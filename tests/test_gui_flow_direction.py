"""Flow-direction napari layer: geometry and colouring. The layer itself is
always added after a solve with signed flows -- there is no toggle."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.results import (
    FLOW_ARROW_COLOUR,
    FLOW_ARROW_DIAMETERS,
    FLOW_ARROW_MIN_WIDTH_UM,
    FLOW_DIRECTION,
    VESSEL_LABELS,
    VESSELS,
    ResultLayers,
    drawn_on_top,
    perturbation_flow_direction_layer_name,
)
from haemolynx.gui.tabs import assign_to_stages
from haemolynx.pipeline import default_schema
from haemolynx.visualization.flow_direction import (
    edge_flow_arrow_zyx,
    edge_flow_direction_sign,
    flow_direction_components,
    flow_direction_rgb,
    flow_direction_rgba,
    flow_direction_vectors,
    flow_heading_deg,
)


def _colormap_name(layer) -> str | None:
    """The map on *layer* as the panel names it: the panel colours through a
    grey-NaN copy of each map (see ``_widget._grey_nan_colormap``)."""
    from haemolynx.gui._widget import _colormap_name as name_of

    return name_of(layer)


def _two_node_edge(*, flow_signed: float, voxels=None) -> nx.MultiGraph:
    """One edge along +z from u=0 at origin to v=1 at z=10."""
    graph = nx.MultiGraph()
    u_pos = np.array([0.0, 0.0, 0.0])
    v_pos = np.array([10.0, 0.0, 0.0])
    graph.add_node(0, pos=u_pos)
    graph.add_node(1, pos=v_pos)
    path = voxels if voxels is not None else [u_pos.tolist(), v_pos.tolist()]
    graph.add_edge(
        0,
        1,
        key=0,
        voxels=path,
        length=10.0,
        flow_signed=flow_signed,
        flow_abs=abs(flow_signed),
        conductance=1.0,
    )
    return graph


def _built_with_flows(graph, **settings) -> ResultLayers:
    results = ResultLayers(settings=dict(settings))
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4)),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
        ),
    )
    volume = SimpleNamespace(
        image=np.zeros((4, 4, 4), dtype=np.uint8),
        skeleton=np.zeros((4, 4, 4), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    results.stage_finished(
        "build_network",
        SimpleNamespace(
            graph=graph,
            volume=volume,
            large_arteriole_mask=None,
            large_venule_mask=None,
            small_arteriole_mask=None,
            small_venule_mask=None,
        ),
    )
    return results


# --- pure geometry (ported from Plotly flow-direction helper) ----------------


def test_positive_flow_signed_means_u_to_v():
    assert edge_flow_direction_sign({"flow_signed": 3.5}) == 1
    assert edge_flow_direction_sign({"flow_signed": -2.0}) == -1
    assert edge_flow_direction_sign({"flow_signed": 0.0}) is None
    assert edge_flow_direction_sign({}) is None


def test_direction_vector_orientation_matches_positive_flow_along_edge():
    """Synthetic 2-node edge: positive flow arrow points along u -> v (+z)."""
    graph = _two_node_edge(flow_signed=4.0)
    vectors, features = flow_direction_vectors(graph)

    assert vectors.shape == (1, 2, 3)
    direction = vectors[0, 1]
    # Centreline is along +z; positive flow must not reverse that tangent.
    assert direction[0] > 0.0
    np.testing.assert_allclose(direction[1:], 0.0, atol=1e-12)
    assert float(np.linalg.norm(direction)) > 0.0
    assert features["flow_signed"].tolist() == [4.0]


def test_negative_flow_reverses_the_arrow():
    positive = flow_direction_vectors(_two_node_edge(flow_signed=2.0))[0][0, 1]
    negative = flow_direction_vectors(_two_node_edge(flow_signed=-2.0))[0][0, 1]
    assert positive[0] > 0.0
    assert negative[0] < 0.0
    np.testing.assert_allclose(positive, -negative)


def test_colour_mapping_uses_exact_flow_magnitude():
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    graph.add_node(2, pos=np.array([10.0, 0.0, 0.0]))
    graph.add_edge(
        0, 1, key=0,
        voxels=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        flow_signed=3.0, flow_abs=3.0,
    )
    graph.add_edge(
        1, 2, key=0,
        voxels=[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        flow_signed=-7.5, flow_abs=7.5,
    )

    _vectors, features = flow_direction_vectors(graph)

    assert features["flow_abs"].tolist() == [3.0, 7.5]
    assert features["flow_signed"].tolist() == [3.0, -7.5]


def test_arrow_helper_rejects_degenerate_polyline():
    assert edge_flow_arrow_zyx(
        np.array([[1.0, 2.0, 3.0]]),
        direction_sign=1,
        arrow_length=1.0,
        lateral_offset=0.1,
    ) is None


def test_flow_direction_components_pure_axes():
    z, y, x = flow_direction_components(np.array([1.0, 0.0, 0.0]))
    assert z == 1.0 and y == 0.0 and x == 0.0
    z, y, x = flow_direction_components(np.array([-1.0, 0.0, 0.0]))
    assert z == -1.0 and y == 0.0 and x == 0.0
    z, y, x = flow_direction_components(np.array([0.0, 2.0, 0.0]))
    assert z == 0.0 and y == 1.0 and x == 0.0
    z, y, x = flow_direction_components(np.array([0.0, -2.0, 0.0]))
    assert z == 0.0 and y == -1.0 and x == 0.0
    z, y, x = flow_direction_components(np.array([0.0, 0.0, 5.0]))
    assert z == 0.0 and y == 0.0 and x == 1.0
    z, y, x = flow_direction_components(np.array([0.0, 0.0, -5.0]))
    assert z == 0.0 and y == 0.0 and x == -1.0


def test_flow_direction_components_diagonal_blends():
    z, y, x = flow_direction_components(np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose([z, y, x], [0.70710678, 0.70710678, 0.0], rtol=1e-5)


def test_flow_direction_components_zero_vector_is_finite():
    z, y, x = flow_direction_components(np.zeros(3))
    assert (z, y, x) == (0.0, 0.0, 0.0)
    assert all(np.isfinite(v) for v in (z, y, x))


def test_flow_direction_rgb_canonical_axes():
    """R=x, G=y, B=z via (component + 1) / 2; +axis is the named channel."""
    np.testing.assert_allclose(flow_direction_rgb([0.0, 0.0, 1.0]), (1.0, 0.5, 0.5))
    np.testing.assert_allclose(flow_direction_rgb([0.0, 1.0, 0.0]), (0.5, 1.0, 0.5))
    np.testing.assert_allclose(flow_direction_rgb([1.0, 0.0, 0.0]), (0.5, 0.5, 1.0))
    np.testing.assert_allclose(flow_direction_rgb([0.0, 0.0, -1.0]), (0.0, 0.5, 0.5))
    np.testing.assert_allclose(flow_direction_rgb([0.0, -1.0, 0.0]), (0.5, 0.0, 0.5))
    np.testing.assert_allclose(flow_direction_rgb([-1.0, 0.0, 0.0]), (0.5, 0.5, 0.0))


def test_flow_direction_rgb_opposites_are_complementary():
    plus = np.asarray(flow_direction_rgb([1.0, 0.0, 0.0]))
    minus = np.asarray(flow_direction_rgb([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(plus + minus, 1.0)
    plus = np.asarray(flow_direction_rgb([0.0, 2.0, 0.0]))
    minus = np.asarray(flow_direction_rgb([0.0, -2.0, 0.0]))
    np.testing.assert_allclose(plus + minus, 1.0)
    plus = np.asarray(flow_direction_rgb([0.0, 0.0, 5.0]))
    minus = np.asarray(flow_direction_rgb([0.0, 0.0, -5.0]))
    np.testing.assert_allclose(plus + minus, 1.0)


def test_flow_direction_rgba_matches_rgb_and_alpha():
    rgba = flow_direction_rgba([1.0], [0.0], [0.0])
    assert rgba.shape == (1, 4)
    np.testing.assert_allclose(rgba[0, :3], flow_direction_rgb([1.0, 0.0, 0.0]))
    assert rgba[0, 3] == pytest.approx(1.0)


def test_flow_dir_rgb_sentinel_in_features():
    _vectors, features = flow_direction_vectors(_two_node_edge(flow_signed=1.0))
    assert "flow_dir_rgb" in features
    assert features["flow_dir_rgb"].shape == (1,)


def test_antiparallel_edges_have_opposite_direction_components():
    positive = flow_direction_vectors(_two_node_edge(flow_signed=2.0))[1]
    negative = flow_direction_vectors(_two_node_edge(flow_signed=-2.0))[1]
    np.testing.assert_allclose(positive["flow_dir_z"], [1.0])
    np.testing.assert_allclose(negative["flow_dir_z"], [-1.0])
    np.testing.assert_allclose(positive["flow_dir_y"], [0.0])
    np.testing.assert_allclose(positive["flow_dir_x"], [0.0])


def test_flow_direction_components_finite_in_features():
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    graph.add_node(2, pos=np.array([5.0, 5.0, 0.0]))
    graph.add_edge(
        0, 1, key=0,
        voxels=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        flow_signed=1e-6, flow_abs=1e-6,  # small, but above the arrow floor
    )
    graph.add_edge(
        1, 2, key=0,
        voxels=[[5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
        flow_signed=-3.0, flow_abs=3.0,
    )
    _vectors, features = flow_direction_vectors(graph)
    for key in ("flow_dir_z", "flow_dir_y", "flow_dir_x"):
        col = np.asarray(features[key], dtype=float)
        assert np.all(np.isfinite(col))
    np.testing.assert_allclose(features["flow_dir_z"], [1.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(features["flow_dir_y"], [0.0, -1.0], rtol=1e-5)
    np.testing.assert_allclose(features["flow_dir_x"], [0.0, 0.0], rtol=1e-5)


# --- ResultLayers / Export-tab toggle ----------------------------------------


def test_flow_direction_layer_always_renders_when_flows_exist():
    """The layer is unconditional now -- no toggle can omit it. export_results
    also re-emits the vessels/vessel-label layers alongside it (so any
    attribute export_results itself wrote, e.g. vascular_community, reaches
    layer.features -- see ResultLayers._from_export_results)."""
    graph = _two_node_edge(flow_signed=1.25)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())

    assert [s.name for s in group.layers] == [VESSELS, VESSEL_LABELS, FLOW_DIRECTION]
    spec = next(s for s in group.layers if s.name == FLOW_DIRECTION)
    assert spec.name == FLOW_DIRECTION
    assert spec.kind == "vectors"
    assert len(spec.data) == 1
    # One colour, over a network coloured by flow; the 3D direction map is
    # still there to colour by.
    assert spec.colour_by is None
    assert spec.options["edge_color"] == FLOW_ARROW_COLOUR
    assert spec.contrast_limits is None
    assert spec.features["flow_abs"].tolist() == [1.25]
    assert "flow_dir_rgb" in spec.features
    assert "flow_heading_deg" in spec.features
    np.testing.assert_allclose(spec.features["flow_dir_z"], [1.0], rtol=1e-5)
    np.testing.assert_allclose(spec.features["flow_dir_y"], [0.0], atol=1e-12)
    np.testing.assert_allclose(spec.features["flow_dir_x"], [0.0], atol=1e-12)
    assert spec.options.get("vector_style") == "triangle"
    assert spec.options.get("length") == 1.0


def test_flow_direction_layer_uses_flow_arrow_scale_setting():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        flow_arrow_scale=2.5,
    )
    group = results.stage_finished("export_results", SimpleNamespace())

    assert [s.name for s in group.layers] == [VESSELS, VESSEL_LABELS, FLOW_DIRECTION]
    spec = next(s for s in group.layers if s.name == FLOW_DIRECTION)
    assert spec.options["length"] == 2.5


def _vessels_of_widths(widths, *, length: float = 100.0) -> nx.MultiGraph:
    """One vessel per width, each *length* µm along +z and carrying flow."""
    graph = nx.MultiGraph()
    for index, width in enumerate(widths):
        start = np.array([0.0, 20.0 * index, 0.0])
        end = start + np.array([length, 0.0, 0.0])
        graph.add_node(2 * index, pos=start)
        graph.add_node(2 * index + 1, pos=end)
        graph.add_edge(2 * index, 2 * index + 1, key=0, voxels=[start.tolist(), end.tolist()],
                       length=length, flow_signed=1e-12, flow_abs=1e-12, diameter_um=width)
    return graph


def _arrow_lengths(vectors) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors)[:, 1], axis=1)


def test_an_arrow_is_as_long_as_its_vessel_is_wide_up_to_half_its_length():
    """Every arrow was a fifth of the median vessel's length (18 µm here)
    whatever its vessel's width; drawn over tubes, a capillary-sized arrow on
    an artery was a speck. Each now runs FLOW_ARROW_DIAMETERS of its own
    vessel's diameters where that is longer, but never past half the vessel."""
    graph = _vessels_of_widths([4.0, 20.0, 80.0])
    uniform, _features = flow_direction_vectors(graph)
    np.testing.assert_allclose(_arrow_lengths(uniform), [18.0, 18.0, 18.0])

    sized, _features = flow_direction_vectors(graph, arrow_diameter_scale=2.0)
    # 4 um: 8 < 18, the uniform length; 20 um: 40; 80 um: 160, cut to 50.
    np.testing.assert_allclose(_arrow_lengths(sized), [18.0, 40.0, 50.0])
    # Same anchors and directions: only the lengths change.
    np.testing.assert_allclose(np.asarray(sized)[:, 0], np.asarray(uniform)[:, 0])


def test_a_vessel_with_no_width_keeps_the_uniform_arrow():
    graph = _vessels_of_widths([20.0, 20.0])
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["diameter_um"] = None
        break
    sized, _features = flow_direction_vectors(graph, arrow_diameter_scale=2.0)
    np.testing.assert_allclose(sorted(_arrow_lengths(sized)), [18.0, 40.0])


def test_flow_arrows_follow_the_vessels_widths_and_go_over_the_tubes():
    """The layer asks for arrows FLOW_ARROW_DIAMETERS of their vessels long,
    heads half the median vessel's width wide (never under the width they had
    before), drawn without a depth test -- the tubes hid every one -- and kept
    over the tubes, a perturbation's arrows too."""
    graph = _vessels_of_widths([4.0, 20.0, 80.0])
    spec = next(
        s for s in _built_with_flows(graph).stage_finished("export_results", SimpleNamespace()).layers
        if s.name == FLOW_DIRECTION
    )
    np.testing.assert_allclose(_arrow_lengths(spec.data), [18.0, FLOW_ARROW_DIAMETERS * 20.0, 50.0])
    assert spec.options["edge_width"] == pytest.approx(10.0), "half the median width, 20 um"
    assert spec.options["blending"] == "translucent_no_depth"

    capillaries = _vessels_of_widths([1.0, 1.0])
    spec = next(
        s for s in _built_with_flows(capillaries).stage_finished("export_results", SimpleNamespace()).layers
        if s.name == FLOW_DIRECTION
    )
    assert spec.options["edge_width"] == FLOW_ARROW_MIN_WIDTH_UM

    assert drawn_on_top(FLOW_DIRECTION)
    assert drawn_on_top(perturbation_flow_direction_layer_name("art_dilate_20"))
    assert not drawn_on_top(VESSELS)
    assert not drawn_on_top("flow direction"), "a user's layer is not ours"


def test_the_flow_arrows_start_hidden_and_then_keep_what_the_user_chose():
    """Over every tube, a network's thousand arrows are a lot to open a run
    with: hidden the first time, and afterwards each export keeps the
    layer as the user left it (a spec shown keeps what is on screen)."""
    results = _built_with_flows(_two_node_edge(flow_signed=1.0))

    def arrows():
        group = results.stage_finished("export_results", SimpleNamespace())
        return next(s for s in group.layers if s.name == FLOW_DIRECTION)

    assert arrows().visible is False
    assert arrows().visible is True


def test_flow_direction_layer_emits_nothing_without_flows():
    graph = _two_node_edge(flow_signed=0.0)
    # Zero signed flow is skipped by the direction helper; strip attributes.
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data.pop("flow_signed", None)
        data.pop("flow_abs", None)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    assert FLOW_DIRECTION not in [spec.name for spec in group.layers]
    # The vessels/vessel-label layers are still re-emitted even with no
    # flow-direction layer to add.
    assert [s.name for s in group.layers] == [VESSELS, VESSEL_LABELS]
    assert "no signed flows" in group.note.lower()


def test_flow_arrow_scale_schema_default_and_requires():
    schema = default_schema()
    setting = schema["flow_arrow_scale"]
    assert setting.kind == "float"
    assert setting.default == 1.0
    assert setting.minimum == 0.1
    assert setting.maximum == 5.0
    assert setting.requires == ("run_haemodynamics",)
    assert setting.section == "Solver and output"


def test_flow_arrow_scale_lives_on_export_tab():
    owner = assign_to_stages(default_schema())
    assert owner["flow_arrow_scale"] == "10. Export"


def test_flow_heading_deg_pure_axes():
    assert flow_heading_deg(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert flow_heading_deg(np.array([-1.0, 0.0, 0.0])) == pytest.approx(180.0)
    assert flow_heading_deg(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert flow_heading_deg(np.array([0.0, -1.0, 0.0])) == pytest.approx(180.0)
    assert flow_heading_deg(np.array([0.0, 0.0, 1.0])) == pytest.approx(90.0)
    assert flow_heading_deg(np.array([0.0, 0.0, -1.0])) == pytest.approx(270.0)


def test_flow_heading_deg_in_features():
    graph = _two_node_edge(flow_signed=1.0)
    _vectors, features = flow_direction_vectors(graph)
    assert "flow_heading_deg" in features
    assert float(features["flow_heading_deg"][0]) == pytest.approx(0.0)


def test_flow_direction_colouring_includes_axis_components_when_enabled():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        flow_direction_colouring=True,
    )
    group = results.stage_finished("export_results", SimpleNamespace())
    spec = next(s for s in group.layers if s.name == FLOW_DIRECTION)
    assert "flow_dir_z" in spec.features
    assert "flow_dir_y" in spec.features
    assert "flow_dir_x" in spec.features
    assert "flow_heading_deg" in spec.features
    assert "flow_dir_rgb" in spec.features
    np.testing.assert_allclose(spec.features["flow_dir_z"], [1.0], rtol=1e-5)


def test_flow_direction_colouring_excludes_axis_components_when_disabled():
    """Regression: flow_direction_colouring was declared but never read
    anywhere -- unchecking it had no effect. The raw axis components are now
    excluded from the colour-by dropdown; the layer itself, flow_abs, and
    the derived flow_dir_rgb/flow_heading_deg encodings stay regardless (see
    the setting's own help text: only "axis components" are gated)."""
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        flow_direction_colouring=False,
    )
    group = results.stage_finished("export_results", SimpleNamespace())
    spec = next(s for s in group.layers if s.name == FLOW_DIRECTION)
    assert "flow_dir_z" not in spec.features
    assert "flow_dir_y" not in spec.features
    assert "flow_dir_x" not in spec.features
    assert "flow_heading_deg" in spec.features
    assert "flow_dir_rgb" in spec.features
    assert "flow_abs" in spec.features

    vessels_spec = next(s for s in group.layers if s.name == VESSELS)
    assert "flow_dir_z" not in vessels_spec.features
    assert "flow_dir_y" not in vessels_spec.features
    assert "flow_dir_x" not in vessels_spec.features


def test_flow_direction_colouring_schema_default_and_requires():
    schema = default_schema()
    setting = schema["flow_direction_colouring"]
    assert setting.kind == "bool"
    assert setting.default is True
    assert setting.requires == ("run_haemodynamics",)
    assert setting.section == "Solver and output"


def test_flow_direction_colouring_lives_on_export_tab():
    owner = assign_to_stages(default_schema())
    assert owner["flow_direction_colouring"] == "10. Export"


def _builds_a_viewer(test):
    """A test taking ``make_napari_viewer`` builds a real viewer: a gui test,
    run by the napari-gui job and skipped where napari is not installed (every
    other job; the fixture is napari's own)."""
    needs_napari = pytest.mark.skipif(
        importlib.util.find_spec("napari") is None, reason="needs napari"
    )
    return pytest.mark.gui(needs_napari(test))


def _perpendicular_arrow_graph() -> nx.MultiGraph:
    """Two edges from the origin: one along +z, one along +y (physical z,y,x)."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([10.0, 0.0, 0.0]))
    graph.add_node(2, pos=np.array([0.0, 10.0, 0.0]))
    graph.add_edge(
        0,
        1,
        key=0,
        voxels=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        flow_signed=1.0,
        flow_abs=1.0,
    )
    graph.add_edge(
        0,
        2,
        key=0,
        voxels=[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        flow_signed=1.0,
        flow_abs=1.0,
    )
    return graph


def _axis_arrow_graph(*, along: str, flow_sign: int = 1) -> nx.MultiGraph:
    """One edge along +axis; ``flow_sign`` sets arrow direction along that axis."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    if along == "z":
        end = np.array([10.0, 0.0, 0.0])
    elif along == "y":
        end = np.array([0.0, 10.0, 0.0])
    elif along == "x":
        end = np.array([0.0, 0.0, 10.0])
    else:
        raise ValueError(along)
    graph.add_node(1, pos=end)
    graph.add_edge(
        0,
        1,
        key=0,
        voxels=[[0.0, 0.0, 0.0], end.tolist()],
        flow_signed=float(flow_sign),
        flow_abs=1.0,
    )
    return graph


def _heading_colours_for_graphs(make_napari_viewer, *graphs) -> list[np.ndarray]:
    from haemolynx.gui._widget import _apply_layers, _colour_layer

    colours = []
    for graph in graphs:
        results = _built_with_flows(graph)
        group = results.stage_finished("export_results", SimpleNamespace())
        viewer = make_napari_viewer()
        _apply_layers(viewer, group)
        layer = viewer.layers[FLOW_DIRECTION]
        _colour_layer(layer, "flow_heading_deg", "continuous")
        colours.append(np.asarray(layer.edge_color, dtype=float)[0, :3])
    return colours


@_builds_a_viewer
def test_flow_heading_y_opposite_colours_differ(make_napari_viewer):
    pos, neg = _heading_colours_for_graphs(
        make_napari_viewer,
        _axis_arrow_graph(along="y", flow_sign=1),
        _axis_arrow_graph(along="y", flow_sign=-1),
    )
    assert not np.allclose(pos, neg, atol=0.02)


@_builds_a_viewer
def test_flow_heading_x_opposite_colours_differ(make_napari_viewer):
    pos, neg = _heading_colours_for_graphs(
        make_napari_viewer,
        _axis_arrow_graph(along="x", flow_sign=1),
        _axis_arrow_graph(along="x", flow_sign=-1),
    )
    assert not np.allclose(pos, neg, atol=0.02)


@_builds_a_viewer
def test_flow_heading_z_vertical_colours_differ(make_napari_viewer):
    pos, neg = _heading_colours_for_graphs(
        make_napari_viewer,
        _axis_arrow_graph(along="z", flow_sign=1),
        _axis_arrow_graph(along="z", flow_sign=-1),
    )
    assert not np.allclose(pos, neg, atol=0.02)


@_builds_a_viewer
def test_flow_heading_deg_uses_fixed_clim(make_napari_viewer):
    from haemolynx.gui._widget import _apply_layers, _colour_layer

    graph = _axis_arrow_graph(along="y", flow_sign=1)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _colour_layer(layer, "flow_heading_deg", "continuous")
    assert layer.edge_contrast_limits == pytest.approx((0.0, 360.0))


@_builds_a_viewer
def test_flow_dir_z_colour_distinguishes_perpendicular_arrows(make_napari_viewer):
    """Axis-aligned +z and +y arrows must differ when coloured by flow_dir_z."""
    from haemolynx.gui._widget import _apply_layers, _colour_layer

    graph = _perpendicular_arrow_graph()
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]

    np.testing.assert_allclose(layer.features["flow_dir_z"], [1.0, 0.0], rtol=1e-5)

    _colour_layer(layer, "flow_dir_z", "continuous")
    colours = np.asarray(layer.edge_color, dtype=float)
    assert colours.shape == (2, 4)
    assert not np.allclose(colours[0, :3], colours[1, :3], atol=0.02)
    assert layer.edge_color_mode == "colormap"
    assert layer.edge_contrast_limits == pytest.approx((-1.0, 1.0))


@_builds_a_viewer
def test_flow_direction_combo_colours_by_flow_dir_z(make_napari_viewer):
    """Colour-by combo re-applies diverging limits without re-running export."""
    from haemolynx.gui._widget import _apply_layers, _attach_colour_scale, _layer_controls

    graph = _perpendicular_arrow_graph()
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _attach_colour_scale(viewer, layer)

    controls = _layer_controls(viewer, layer)
    controls._haemolynx_feature.native.setCurrentText("flow_dir_z")

    colours = np.asarray(layer.edge_color, dtype=float)
    assert not np.allclose(colours[0, :3], colours[1, :3], atol=0.02)
    assert layer.edge_contrast_limits == pytest.approx((-1.0, 1.0))


def _rgb_for_axis_graph(make_napari_viewer, along: str, flow_sign: int) -> np.ndarray:
    from haemolynx.gui._widget import _apply_layers, _colour_layer

    graph = _axis_arrow_graph(along=along, flow_sign=flow_sign)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _colour_layer(layer, "flow_dir_rgb", "direct")
    assert layer.edge_color_mode == "direct"
    return np.asarray(layer.edge_color, dtype=float)[0]


@_builds_a_viewer
def test_flow_direction_layer_defaults_to_one_colour_with_rgb_a_pick_away(make_napari_viewer):
    """Over a network coloured by flow the arrows are one colour, saying which
    way and nothing else; the 3D RGB map (R=x, G=y, B=z) is a pick away."""
    from haemolynx.gui._widget import _active_column, _apply_layers, _colour_layer

    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    spec = next(s for s in group.layers if s.name == FLOW_DIRECTION)
    assert spec.colour_by is None

    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    assert _active_column(layer) is None
    np.testing.assert_allclose(np.asarray(layer.edge_color, dtype=float)[0], (1.0, 1.0, 1.0, 1.0))

    _colour_layer(layer, "flow_dir_rgb", "direct")
    assert _active_column(layer) == "flow_dir_rgb"
    colour = np.asarray(layer.edge_color, dtype=float)[0, :3]
    # +z → blue-ish (R=x, G=y, B=z).
    np.testing.assert_allclose(colour, (0.5, 0.5, 1.0), atol=0.05)


@_builds_a_viewer
def test_flow_arrows_are_drawn_over_the_vessel_tubes(make_napari_viewer):
    """Each arrow lies along its vessel's centreline, inside the tube, which
    hid every one of them while they were depth-tested: they go without the
    depth test and after the tubes in the layer list, under the terminal
    nodes."""
    from haemolynx.gui._widget import _apply_layers
    from haemolynx.gui.results import BOUNDARY_NODES, VESSEL_TUBES

    graph = _two_node_edge(flow_signed=1.0)
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["diameter_um"] = 4.0
    results = _built_with_flows(graph)
    results.stage_finished(
        "assign_boundaries",
        SimpleNamespace(inlet_nodes=[0], outlet_nodes=[1], arteriole_boundary_nodes=[],
                        venule_boundary_nodes=[], graph=graph),
    )
    viewer = make_napari_viewer()
    _apply_layers(viewer, results.stage_finished("solve", SimpleNamespace(
        graph=graph, node_list=[0, 1], pressure=np.array([10.0, 0.0]))))
    _apply_layers(viewer, results.stage_finished("export_results", SimpleNamespace()))

    names = [layer.name for layer in viewer.layers]
    assert VESSEL_TUBES in names
    assert names.index(FLOW_DIRECTION) > names.index(VESSEL_TUBES)
    if BOUNDARY_NODES in names:
        assert names.index(BOUNDARY_NODES) > names.index(FLOW_DIRECTION)
    assert viewer.layers[FLOW_DIRECTION].blending == "translucent_no_depth"


@_builds_a_viewer
def test_flow_dir_rgb_axis_colours_and_complements(make_napari_viewer):
    expected = {
        ("x", 1): (1.0, 0.5, 0.5),
        ("x", -1): (0.0, 0.5, 0.5),
        ("y", 1): (0.5, 1.0, 0.5),
        ("y", -1): (0.5, 0.0, 0.5),
        ("z", 1): (0.5, 0.5, 1.0),
        ("z", -1): (0.5, 0.5, 0.0),
    }
    colours = {
        key: _rgb_for_axis_graph(make_napari_viewer, *key)
        for key in expected
    }
    for key, rgb in expected.items():
        np.testing.assert_allclose(colours[key][:3], rgb, atol=0.05)
        assert colours[key][3] == pytest.approx(1.0)
    for axis in ("x", "y", "z"):
        np.testing.assert_allclose(
            colours[(axis, 1)][:3] + colours[(axis, -1)][:3],
            1.0,
            atol=0.05,
        )


@_builds_a_viewer
def test_flow_dir_rgb_combo_switch_to_flow_abs_uses_colormap(make_napari_viewer):
    from haemolynx.gui._widget import (
        _active_column,
        _apply_layers,
        _attach_colour_scale,
        _layer_controls,
    )

    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _attach_colour_scale(viewer, layer)

    controls = _layer_controls(viewer, layer)
    offered = [
        controls._haemolynx_feature.native.itemText(i)
        for i in range(controls._haemolynx_feature.native.count())
    ]
    assert "flow_dir_rgb" in offered
    assert "flow_heading_deg" in offered
    assert "flow_abs" in offered
    # One colour, no column: the combo names none, rather than its first item,
    # a colouring not on screen that choosing then did nothing for.
    assert controls._haemolynx_feature.native.currentText() == ""

    controls._haemolynx_feature.native.setCurrentText("flow_dir_rgb")
    assert _active_column(layer) == "flow_dir_rgb"
    assert layer.edge_color_mode == "direct"
    controls._haemolynx_feature.native.setCurrentText("flow_abs")
    assert _active_column(layer) == "flow_abs"
    assert layer.edge_color_mode == "colormap"
    assert _colormap_name(layer) == "viridis"


# --- colour-map dropdown on layer controls ----------------------------------


_REQUIRED_COLORMAPS = frozenset({
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray",
    "coolwarm", "RdBu", "seismic", "PiYG",
    "hsv", "twilight", "twilight_shifted",
    "jet", "rainbow", "hot", "cool", "spring", "summer", "autumn", "winter",
    "bone", "copper", "pink",
})


def test_colormap_catalog_lists_a_wide_grouped_set():
    from haemolynx.gui._widget import COLORMAP_GROUPS, colormap_choices

    names = colormap_choices()
    assert _REQUIRED_COLORMAPS <= set(names)
    assert len(names) == len(set(names))
    titles = [title for title, _group in COLORMAP_GROUPS]
    assert titles == ["Sequential", "Diverging", "Cyclic", "Other"]
    sequential = COLORMAP_GROUPS[0][1]
    diverging = COLORMAP_GROUPS[1][1]
    cyclic = COLORMAP_GROUPS[2][1]
    assert sequential.index("viridis") < sequential.index("gray")
    assert "coolwarm" in diverging
    assert "hsv" in cyclic


@_builds_a_viewer
def test_column_defaults_use_the_real_colormap_property(make_napari_viewer):
    """``edge_color_colormap`` is a stray attr; Vectors store the LUT on edge_colormap."""
    from haemolynx.gui._widget import _apply_layers, _colour_layer

    graph = _perpendicular_arrow_graph()
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]

    assert hasattr(layer, "edge_colormap")
    assert not hasattr(type(layer), "edge_color_colormap")

    _colour_layer(layer, "flow_dir_z", "continuous")
    assert _colormap_name(layer) == "coolwarm"
    _colour_layer(layer, "flow_heading_deg", "continuous")
    assert _colormap_name(layer) == "hsv"
    _colour_layer(layer, "flow_abs", "continuous")
    assert _colormap_name(layer) == "viridis"


def _colormap_combo(viewer, layer):
    from haemolynx.gui._widget import _layer_controls

    controls = _layer_controls(viewer, layer)
    return getattr(controls, "_haemolynx_colormap", None)


def _combo_maps(chooser) -> list[str]:
    from haemolynx.gui._widget import _COLORMAP_GROUP_HEADERS

    return [
        chooser.native.itemText(i)
        for i in range(chooser.native.count())
        if chooser.native.itemText(i)
        and chooser.native.itemText(i) not in _COLORMAP_GROUP_HEADERS
    ]


@_builds_a_viewer
def test_flow_direction_colormap_combo_hidden_for_direct_rgb(make_napari_viewer):
    from haemolynx.gui._widget import _apply_layers, _attach_colour_scale

    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _attach_colour_scale(viewer, layer)

    chooser = _colormap_combo(viewer, layer)
    assert chooser is not None
    assert _REQUIRED_COLORMAPS <= set(_combo_maps(chooser))
    assert chooser.shown is False
    assert chooser.native.isEnabled() is False


@_builds_a_viewer
def test_flow_direction_colormap_combo_applies_a_selected_map(make_napari_viewer):
    from haemolynx.gui._widget import (
        _apply_layers,
        _attach_colour_scale,
        _layer_controls,
    )

    graph = _perpendicular_arrow_graph()
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _attach_colour_scale(viewer, layer)

    controls = _layer_controls(viewer, layer)
    controls._haemolynx_feature.native.setCurrentText("flow_dir_z")
    chooser = controls._haemolynx_colormap
    assert chooser.shown is True
    assert chooser.native.isEnabled()
    assert chooser.native.currentText() == "coolwarm"
    assert _colormap_name(layer) == "coolwarm"

    before = np.array(layer.edge_color, copy=True)
    chooser.native.setCurrentText("plasma")
    assert _colormap_name(layer) == "plasma"
    after = np.asarray(layer.edge_color)
    assert not np.allclose(before, after)
    assert layer.edge_contrast_limits == pytest.approx((-1.0, 1.0))


@_builds_a_viewer
def test_colour_by_change_resets_colormap_to_the_column_default(make_napari_viewer):
    from haemolynx.gui._widget import (
        _apply_layers,
        _attach_colour_scale,
        _layer_controls,
    )

    graph = _perpendicular_arrow_graph()
    results = _built_with_flows(graph)
    group = results.stage_finished("export_results", SimpleNamespace())
    viewer = make_napari_viewer()
    _apply_layers(viewer, group)
    layer = viewer.layers[FLOW_DIRECTION]
    _attach_colour_scale(viewer, layer)

    controls = _layer_controls(viewer, layer)
    controls._haemolynx_feature.native.setCurrentText("flow_abs")
    controls._haemolynx_colormap.native.setCurrentText("plasma")
    assert _colormap_name(layer) == "plasma"

    controls._haemolynx_feature.native.setCurrentText("flow_heading_deg")
    assert _colormap_name(layer) == "hsv"
    assert controls._haemolynx_colormap.native.currentText() == "hsv"
    assert layer.edge_contrast_limits == pytest.approx((0.0, 360.0))

    controls._haemolynx_feature.native.setCurrentText("flow_dir_rgb")
    assert layer.edge_color_mode == "direct"
    assert controls._haemolynx_colormap.shown is False


def test_no_arrow_on_a_vessel_whose_flow_is_below_the_carrying_floor():
    """A flow whose sign is rounding gets no arrow. The floor is the
    statistics' own for carrying flow."""
    from haemolynx.gui.results import _flow_direction_drawable_edge_index
    from haemolynx.statistics.flux_heterogeneity import CARRYING_FLOOR_FRACTION
    from haemolynx.visualization.flow_direction import flow_arrow_floor

    graph = nx.MultiGraph()
    for node, x in enumerate((0.0, 5.0, 10.0, 15.0)):
        graph.add_node(node, pos=np.array([0.0, 0.0, x]))
    for u, flow in ((0, 1e-9), (1, 1e-20), (2, -5e-19)):  # the floor: 1e-18
        graph.add_edge(
            u, u + 1, key=0, voxels=[graph.nodes[u]["pos"], graph.nodes[u + 1]["pos"]],
            flow_signed=flow, flow_abs=abs(flow),
        )

    vectors, features = flow_direction_vectors(graph)

    assert flow_arrow_floor(graph) == pytest.approx(CARRYING_FLOOR_FRACTION * 1e-9)
    assert len(vectors) == 1 and features["flow_signed"].tolist() == [1e-9]
    assert _flow_direction_drawable_edge_index(graph).tolist() == [0], "hover rows match the arrows"
