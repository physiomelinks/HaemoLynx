"""Flow-direction napari layer: geometry, colouring, and Export-tab toggle."""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

from haemolynx.gui.results import FLOW_DIRECTION, ResultLayers
from haemolynx.gui.tabs import assign_to_stages
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.progress import STAGES
from haemolynx.visualization.flow_direction import (
    edge_flow_arrow_zyx,
    edge_flow_direction_sign,
    flow_direction_vectors,
    flow_toward_face_from_direction,
)


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


def _built_with_flows(graph, *, show_flow_direction_layer: bool, **settings) -> ResultLayers:
    results = ResultLayers(
        settings={
            "show_flow_direction_layer": show_flow_direction_layer,
            **settings,
        }
    )
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


def test_flow_toward_face_dominant_axis_encoding():
    assert flow_toward_face_from_direction(np.array([1.0, 0.0, 0.0])) == 3
    assert flow_toward_face_from_direction(np.array([-1.0, 0.0, 0.0])) == -3
    assert flow_toward_face_from_direction(np.array([0.0, 2.0, 0.0])) == 2
    assert flow_toward_face_from_direction(np.array([0.0, -2.0, 0.0])) == -2
    assert flow_toward_face_from_direction(np.array([0.0, 0.0, 5.0])) == 1
    assert flow_toward_face_from_direction(np.array([0.0, 0.0, -5.0])) == -1


def test_flow_toward_face_zero_vector_is_finite():
    assert flow_toward_face_from_direction(np.zeros(3)) == 0
    assert np.isfinite(flow_toward_face_from_direction(np.zeros(3)))


def test_antiparallel_edges_have_opposite_flow_toward_face():
    positive = flow_direction_vectors(_two_node_edge(flow_signed=2.0))[1]
    negative = flow_direction_vectors(_two_node_edge(flow_signed=-2.0))[1]
    assert positive["flow_toward_face"][0] == 3
    assert negative["flow_toward_face"][0] == -3
    assert positive["flow_toward_face"][0] == -negative["flow_toward_face"][0]


def test_flow_toward_face_values_are_finite_in_features():
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    graph.add_node(2, pos=np.array([5.0, 5.0, 0.0]))
    graph.add_edge(
        0, 1, key=0,
        voxels=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        flow_signed=1e-16, flow_abs=1e-16,
    )
    graph.add_edge(
        1, 2, key=0,
        voxels=[[5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
        flow_signed=-3.0, flow_abs=3.0,
    )
    _vectors, features = flow_direction_vectors(graph)
    faces = np.asarray(features["flow_toward_face"], dtype=float)
    assert np.all(np.isfinite(faces))
    assert faces.tolist() == [3.0, -2.0]


# --- ResultLayers / Export-tab toggle ----------------------------------------


def test_toggle_off_omits_flow_direction_layer():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(graph, show_flow_direction_layer=False)
    group = results.stage_finished("export_results", SimpleNamespace())
    names = [spec.name for spec in group.layers]
    assert FLOW_DIRECTION not in names
    assert group.layers == ()


def test_toggle_on_with_flows_emits_one_arrow_per_directed_edge():
    graph = _two_node_edge(flow_signed=1.25)
    results = _built_with_flows(graph, show_flow_direction_layer=True)
    group = results.stage_finished("export_results", SimpleNamespace())

    assert len(group.layers) == 1
    spec = group.layers[0]
    assert spec.name == FLOW_DIRECTION
    assert spec.kind == "vectors"
    assert len(spec.data) == 1
    assert spec.colour_by == "flow_abs"
    assert spec.features["flow_abs"].tolist() == [1.25]
    assert spec.features["flow_toward_face"].tolist() == [3]
    assert spec.options.get("vector_style") == "triangle"
    assert spec.options.get("length") == 1.0


def test_flow_direction_layer_uses_flow_arrow_scale_setting():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        show_flow_direction_layer=True,
        flow_arrow_scale=2.5,
    )
    group = results.stage_finished("export_results", SimpleNamespace())

    assert len(group.layers) == 1
    assert group.layers[0].options["length"] == 2.5


def test_toggle_on_without_flows_emits_no_layer():
    graph = _two_node_edge(flow_signed=0.0)
    # Zero signed flow is skipped by the direction helper; strip attributes.
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data.pop("flow_signed", None)
        data.pop("flow_abs", None)
    results = _built_with_flows(graph, show_flow_direction_layer=True)
    group = results.stage_finished("export_results", SimpleNamespace())
    assert FLOW_DIRECTION not in [spec.name for spec in group.layers]
    assert group.layers == ()
    assert "no signed flows" in group.note.lower()


def test_show_flow_direction_layer_schema_default_and_requires():
    schema = default_schema()
    setting = schema["show_flow_direction_layer"]
    assert setting.kind == "bool"
    assert setting.default is True
    assert setting.requires == ("run_haemodynamics",)
    assert setting.section == "Solver and output"


def test_flow_arrow_scale_schema_default_and_requires():
    schema = default_schema()
    setting = schema["flow_arrow_scale"]
    assert setting.kind == "float"
    assert setting.default == 1.0
    assert setting.minimum == 0.1
    assert setting.maximum == 5.0
    assert setting.requires == ("show_flow_direction_layer", "run_haemodynamics")
    assert setting.section == "Solver and output"


def test_flow_arrow_scale_lives_on_export_tab():
    owner = assign_to_stages(default_schema())
    assert owner["flow_arrow_scale"] == "8. Export"


def test_show_flow_direction_layer_lives_on_export_tab():
    """Last tab is 8. Export (STAGES); Solver and output section lands there."""
    assert STAGES[-1].call == "export_results"
    assert STAGES[-1].title == "8. Export"
    owner = assign_to_stages(default_schema())
    assert owner["show_flow_direction_layer"] == "8. Export"


def test_flow_direction_colouring_includes_flow_toward_face_when_enabled():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        show_flow_direction_layer=True,
        flow_direction_colouring=True,
    )
    group = results.stage_finished("export_results", SimpleNamespace())
    spec = group.layers[0]
    assert "flow_toward_face" in spec.features
    assert spec.features["flow_toward_face"].tolist() == [3]


def test_flow_direction_colouring_omits_flow_toward_face_when_disabled():
    graph = _two_node_edge(flow_signed=1.0)
    results = _built_with_flows(
        graph,
        show_flow_direction_layer=True,
        flow_direction_colouring=False,
    )
    group = results.stage_finished("export_results", SimpleNamespace())
    spec = group.layers[0]
    assert "flow_toward_face" not in spec.features
    assert "flow_abs" in spec.features


def test_flow_direction_colouring_schema_default_and_requires():
    schema = default_schema()
    setting = schema["flow_direction_colouring"]
    assert setting.kind == "bool"
    assert setting.default is True
    assert setting.requires == ("show_flow_direction_layer", "run_haemodynamics")
    assert setting.section == "Solver and output"


def test_flow_direction_colouring_lives_on_export_tab():
    owner = assign_to_stages(default_schema())
    assert owner["flow_direction_colouring"] == "8. Export"
