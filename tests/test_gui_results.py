"""Turning a finished stage into layers, checked without a display.

The counterpart of `test_gui_layers.py`, which covers the input direction. As
there, the objects a stage returns are stood in for with `SimpleNamespace` and
a small real graph, so every decision here is testable on every Python the
library supports rather than only where napari and a Qt binding are installed.
The widget test that matches this one is in `test_gui_results_widget.py`, marked
`gui`.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.results import (
    BOUNDARY_COORDINATE_POINT_SIZE,
    BOUNDARY_NODE_POINT_SIZE,
    BOUNDARY_NODES,
    BRANCH_HOVER,
    DEFAULT_VESSEL_COLOUR,
    EDGE_COLUMNS,
    FLOW_DIRECTION,
    IMAGE,
    LAYER_NAMES,
    NODE_POINT_SIZE,
    NODES,
    PERICYTE_POINT_SIZE,
    PERICYTES,
    SKELETON,
    VESSEL_LABEL_POINT_SIZE,
    VESSEL_LABELS,
    VESSELS,
    ResultLayers,
    filter_points_by_z,
    filter_vectors_by_z,
    image_z_extent_um,
    is_ours_name,
    is_z_depth_filtered_layer,
    perturbation_layer_names,
    available_edge_columns,
    edge_features,
    edge_polylines,
    midpoints_of,
    node_points,
    pericyte_points,
    polylines_to_vectors,
)
from haemolynx.visualization.flow_direction import flow_direction_vectors

REPO_ROOT = Path(__file__).resolve().parents[1]


def a_graph(**edge_attributes) -> nx.MultiGraph:
    """Three vessels in a line along z, with positions in physical microns."""
    graph = nx.MultiGraph()
    for node_id, z in enumerate((0.0, 10.0, 20.0, 30.0)):
        graph.add_node(node_id, pos=np.array([z, 0.0, 0.0]))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        graph.add_edge(
            u, v, key=0,
            voxels=[graph.nodes[u]["pos"].tolist(), graph.nodes[v]["pos"].tolist()],
            length=10.0, segment_id=u, **edge_attributes,
        )
    return graph


def network(graph, voxel_size_zyx=(1.0, 1.0, 1.0), **masks):
    volume = SimpleNamespace(
        image=np.zeros((4, 4, 4), dtype=np.uint8),
        skeleton=np.zeros((4, 4, 4), dtype=bool),
        voxel_size_xyz=tuple(reversed(voxel_size_zyx)),
        voxel_size_zyx=voxel_size_zyx,
    )
    return SimpleNamespace(
        graph=graph, volume=volume,
        large_arteriole_mask=masks.get("large_arteriole_mask"),
        large_venule_mask=masks.get("large_venule_mask"),
        small_arteriole_mask=masks.get("small_arteriole_mask"),
        small_venule_mask=masks.get("small_venule_mask"),
    )


def built(graph=None, voxel_size_zyx=(1.0, 1.0, 1.0)) -> ResultLayers:
    """A converter that has already seen a skeletonise and a build_network."""
    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4)), skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=tuple(reversed(voxel_size_zyx)),
            voxel_size_zyx=voxel_size_zyx,
        ),
    )
    results.stage_finished("build_network", network(graph or a_graph(), voxel_size_zyx))
    return results


def spec_named(group, name):
    return next(spec for spec in group.layers if spec.name == name)


# --- the registration test, which is the one that matters --------------------


def test_array_layers_and_graph_layers_land_on_the_same_physical_point():
    """The two coordinate systems must meet.

    Node `pos` is physical microns already -- voxel index times voxel size, done
    when the graph was built -- while `image` and `skeleton` are voxel-indexed.
    So graph layers take scale 1 and array layers take the voxel size. Backwards
    is invisible on isotropic data and wrong on every real stack, which is why
    this fixture is deliberately anisotropic.
    """
    voxel_size_zyx = (2.0, 1.0, 0.5)
    voxel_index = np.array([3.0, 4.0, 5.0])
    physical = voxel_index * np.asarray(voxel_size_zyx)

    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.zeros(3))
    graph.add_node(1, pos=physical)
    graph.add_edge(0, 1, key=0, voxels=[np.zeros(3).tolist(), physical.tolist()],
                   length=float(np.linalg.norm(physical)), segment_id=0)

    results = ResultLayers()
    skeleton_group = results.stage_finished(
        "skeletonise",
        SimpleNamespace(image=np.zeros((8, 8, 8)), skeleton=np.zeros((8, 8, 8), dtype=bool),
                        voxel_size_xyz=(0.5, 1.0, 2.0), voxel_size_zyx=voxel_size_zyx),
    )
    graph_group = results.stage_finished("build_network", network(graph, voxel_size_zyx))

    assert spec_named(skeleton_group, SKELETON).scale == voxel_size_zyx
    assert spec_named(skeleton_group, IMAGE).scale == voxel_size_zyx
    assert spec_named(graph_group, VESSELS).scale == (1.0, 1.0, 1.0)
    assert spec_named(graph_group, NODES).scale == (1.0, 1.0, 1.0)

    # The node sits where voxel (3, 4, 5) of the scaled array sits.
    nodes = spec_named(graph_group, NODES)
    assert np.allclose(nodes.data[1], physical)
    assert np.allclose(nodes.data[1], voxel_index * np.asarray(spec_named(skeleton_group, SKELETON).scale))


def test_skeleton_is_hidden_once_vessels_exist():
    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4)),
            skeleton=np.ones((4, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
        ),
    )
    group = results.stage_finished("build_network", network(a_graph()))
    skeleton = spec_named(group, SKELETON)
    assert skeleton.visible is False


def test_assign_boundaries_keeps_skeleton_hidden_after_cut():
    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4)),
            skeleton=np.ones((4, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
        ),
    )
    results.stage_finished("build_network", network(a_graph()))
    group = results.stage_finished(
        "assign_boundaries",
        SimpleNamespace(
            inlet_nodes=[0],
            outlet_nodes=[3],
            arteriole_boundary_nodes=[],
            venule_boundary_nodes=[],
            graph=a_graph(),
        ),
    )
    skeleton = spec_named(group, SKELETON)
    assert skeleton.visible is False


# --- geometry ----------------------------------------------------------------


def test_every_edge_becomes_a_polyline():
    paths, identity = edge_polylines(a_graph())
    assert len(paths) == 3
    assert identity["edge_index"].tolist() == [0, 1, 2]
    assert identity["u"].tolist() == [0, 1, 2]


def test_an_edge_that_cannot_be_placed_is_dropped_not_fatal():
    """One unplaceable vessel must not cost the others their layer."""
    graph = a_graph()
    graph.add_node(99)
    graph.add_edge(99, 98, key=0)
    graph.add_node(98)
    paths, _identity = edge_polylines(graph)
    assert len(paths) == 3


def test_vectors_reconstruct_the_polylines_they_came_from():
    paths, _ = edge_polylines(a_graph())
    vectors, owner = polylines_to_vectors(paths)

    assert len(vectors) == sum(len(path) - 1 for path in paths)
    assert len(owner) == len(vectors)
    for index, path in enumerate(paths):
        mine = vectors[owner == index]
        assert np.allclose(mine[0, 0], path[0])
        assert np.allclose(mine[:, 0] + mine[:, 1], path[1:])


def test_a_per_edge_value_repeats_across_that_edges_segments():
    paths, identity = edge_polylines(a_graph())
    _vectors, owner = polylines_to_vectors(paths)
    per_segment = identity["edge_index"][owner]
    assert per_segment.tolist() == sorted(per_segment.tolist())
    assert set(per_segment.tolist()) == {0, 1, 2}


def test_midpoints_sit_in_the_middle_of_each_vessel():
    paths, _ = edge_polylines(a_graph())
    assert np.allclose(midpoints_of(paths)[0], [5.0, 0.0, 0.0])


def test_no_geometry_gives_empty_arrays_rather_than_an_error():
    vectors, owner = polylines_to_vectors([])
    assert vectors.shape == (0, 2, 3)
    assert owner.shape == (0,)
    assert midpoints_of([]).shape == (0, 3)


# --- features ----------------------------------------------------------------


def test_a_column_a_stage_has_not_written_is_nan_not_a_keyerror():
    columns = edge_features(a_graph(), ["length", "flow_abs"])
    assert columns["length"].tolist() == [10.0, 10.0, 10.0]
    assert np.isnan(columns["flow_abs"]).all()


def test_a_missing_text_column_is_empty_not_nan():
    columns = edge_features(a_graph(), ["branch_order"])
    assert columns["branch_order"].tolist() == ["", "", ""]


def test_a_sparse_column_keeps_the_values_it_does_have():
    """`set_edge_flows` skips an edge with no conductance, so flows are sparse."""
    graph = a_graph()
    list(graph.edges(keys=True, data=True))[0][3]["flow_abs"] = 1.5
    values = edge_features(graph, ["flow_abs"])["flow_abs"]
    assert values[0] == 1.5
    assert np.isnan(values[1:]).all()


def test_the_vessels_layer_carries_the_identity_of_each_edge():
    group = built().stage_finished("build_network", network(a_graph()))
    vessels = spec_named(group, VESSELS)
    for column in ("edge_index", "u", "v", "key"):
        assert column in vessels.features
        assert len(vessels.features[column]) == len(vessels.data)


def test_a_column_is_empty_until_its_stage_runs_but_never_absent():
    """Declared from the first layer, filled when its stage gets to it.

    This used to assert the opposite -- that a column appeared only once its
    stage had run -- which read as the tidier contract right up until it met
    napari: the layer controls list the columns once, at construction, so a
    column added later never shows up in the "edge feature:" dropdown. Present
    and empty is what a user can actually find.
    """
    graph = a_graph()
    before = spec_named(built(graph).stage_finished(
        "build_network", network(graph)), VESSELS)
    assert "flow_abs" in before.features
    assert np.isnan(np.asarray(before.features["flow_abs"], dtype=float)).all()

    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 2.0
    after = spec_named(built(graph).stage_finished(
        "build_network", network(graph)), VESSELS)
    assert np.allclose(np.asarray(after.features["flow_abs"], dtype=float), 2.0)


def test_the_panel_still_only_offers_columns_that_hold_a_value():
    """`available_edge_columns` keeps its meaning: filled, not merely declared.

    The layer carries every column so napari can see them; this is the separate
    question of which ones are worth colouring by, and it must not drift into
    "all of them" now that the layer holds all of them.
    """
    graph = a_graph()
    assert "flow_abs" not in available_edge_columns(graph)
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 2.0
    assert "flow_abs" in available_edge_columns(graph)


# --- nodes and boundaries ----------------------------------------------------


def test_nodes_without_a_position_are_left_out():
    graph = a_graph()
    graph.add_node("orphan")
    points, ids = node_points(graph)
    assert len(points) == 4
    assert "orphan" not in ids.tolist()


def test_boundary_roles_are_disjoint_and_carry_their_names():
    results = built()
    group = results.stage_finished(
        "assign_boundaries",
        SimpleNamespace(inlet_nodes=[0], outlet_nodes=[3],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=(0, 3)),
    )
    boundaries = spec_named(group, BOUNDARY_NODES)
    assert boundaries.features["role"].tolist() == ["inlet", "outlet"]
    assert boundaries.colour_kind == "categorical"
    assert len(boundaries.data) == 2


def test_a_run_with_no_boundary_nodes_says_so_rather_than_drawing_nothing():
    group = built().stage_finished(
        "assign_boundaries",
        SimpleNamespace(inlet_nodes=[], outlet_nodes=[],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=None),
    )
    assert "No boundary nodes" in group.note


def test_point_layer_sizes_match_the_earlier_viewer_style():
    """Pin napari Points sizes against the branch-hover enlargement regression.

    Graph nodes / boundary nodes / pericytes / vessel-label midpoints have been
    3 / 6 / 4 / 2 since the first viewer commit (and on ``origin/main``).
    Branch hover is no longer a Points overlay; its pickup radius lives in
    ``branch_hover.BRANCH_HOVER_MAX_DISTANCE``. Boundary-tab picking rings stay
    at 8.0 (they are rings, not snapped nodes).
    """
    assert VESSEL_LABEL_POINT_SIZE == 2.0
    assert NODE_POINT_SIZE == 3.0
    assert BOUNDARY_NODE_POINT_SIZE == 6.0
    assert PERICYTE_POINT_SIZE == 4.0
    assert BOUNDARY_COORDINATE_POINT_SIZE == 8.0

    graph = a_graph()
    first = list(graph.edges(keys=True, data=True))[0]
    first[3]["pericyte_centers_um"] = [5.0]
    first[3]["branch_order"] = "B01"

    results = built(graph)
    network_group = results.stage_finished("build_network", network(graph))
    assert spec_named(network_group, VESSEL_LABELS).options["size"] == (
        VESSEL_LABEL_POINT_SIZE
    )
    names = [spec.name for spec in network_group.layers]
    assert BRANCH_HOVER not in names
    assert spec_named(network_group, NODES).options["size"] == NODE_POINT_SIZE

    boundary_group = results.stage_finished(
        "assign_boundaries",
        SimpleNamespace(
            inlet_nodes=[0],
            outlet_nodes=[3],
            arteriole_boundary_nodes=[],
            venule_boundary_nodes=[],
            resistance_node_pair=(0, 3),
        ),
    )
    assert spec_named(boundary_group, NODES).options["size"] == NODE_POINT_SIZE
    assert spec_named(boundary_group, BOUNDARY_NODES).options["size"] == (
        BOUNDARY_NODE_POINT_SIZE
    )

    diameter_group = results.stage_finished(
        "assign_diameters", SimpleNamespace(graph=graph)
    )
    assert spec_named(diameter_group, PERICYTES).options["size"] == (
        PERICYTE_POINT_SIZE
    )

    solve_group = results.stage_finished(
        "solve",
        SimpleNamespace(
            pressure=np.zeros(4),
            node_list=[0, 1, 2, 3],
            equivalent_resistance=1.0,
        ),
    )
    assert spec_named(solve_group, NODES).options["size"] == NODE_POINT_SIZE

    from haemolynx.gui.boundary_picking import BC_COORDINATES, specs_for

    picks = specs_for({"inlet_node_coordinates": [[1.0, 2.0, 3.0]]})
    assert picks[0].name == BC_COORDINATES
    assert picks[0].options["size"] == BOUNDARY_COORDINATE_POINT_SIZE


# --- pressure ----------------------------------------------------------------


def test_pressure_follows_the_node_it_belongs_to_not_its_position_in_the_array():
    """`Solution.pressure` is ordered by `node_list`, which is not id order."""
    results = built()
    group = results.stage_finished(
        "solve",
        SimpleNamespace(pressure=np.array([40.0, 10.0, 30.0, 20.0]),
                        node_list=[3, 0, 2, 1], equivalent_resistance=1.0),
    )
    nodes = spec_named(group, NODES)
    by_id = dict(zip(nodes.features["node_id"].tolist(),
                     nodes.features["pressure"].tolist()))
    assert by_id == {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}


def test_the_equivalent_resistance_is_reported():
    group = built().stage_finished(
        "solve",
        SimpleNamespace(pressure=np.zeros(4), node_list=[0, 1, 2, 3],
                        equivalent_resistance=1.25e16),
    )
    assert "1.2500e+16" in group.note


# --- pericytes ---------------------------------------------------------------


def test_pericytes_come_from_the_edges_own_centres():
    """Not from the periodic derivation, which is wrong for a mask strategy."""
    graph = a_graph()
    first = list(graph.edges(keys=True, data=True))[0]
    first[3]["pericyte_centers_um"] = [5.0]
    first[3]["branch_order"] = "B01"

    points, features = pericyte_points(graph)

    assert len(points) == 1
    assert np.allclose(points[0], [5.0, 0.0, 0.0])
    assert features["arc_length_um"].tolist() == [5.0]
    assert features["branch_order"].tolist() == ["B01"]


def test_no_pericytes_gives_no_layer():
    group = built().stage_finished("assign_diameters", SimpleNamespace(graph=a_graph()))
    assert all(spec.name != PERICYTES for spec in group.layers)


# --- the stages --------------------------------------------------------------


def test_the_input_stage_says_where_its_content_will_appear():
    """Otherwise "stage 1 showed nothing" reads as a bug."""
    group = ResultLayers().stage_finished(
        "segment", SimpleNamespace(image_path=Path("/data/mask.tif"))
    )
    assert group.layers == ()
    assert "mask.tif" in group.note
    assert "next stage" in group.note


def test_the_model_stage_adds_no_layer_but_repaints_the_network():
    """It returns the object it was given; a new layer would be a duplicate."""
    graph = a_graph()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["resistance"] = 1.0
    group = built(graph).stage_finished(
        "build_haemodynamic_model", SimpleNamespace(graph=graph, results={})
    )
    assert group.layers == ()
    assert group.recolour == ((VESSELS, "resistance"),)
    assert "3 of 3" in group.note


def test_the_view_is_turned_to_3d_once_when_the_first_geometry_arrives():
    """A network drawn in a 2D slice is a handful of dots."""
    results = ResultLayers()
    first = results.stage_finished("build_network", network(a_graph()))
    assert first.ndisplay == 3

    again = results.stage_finished("build_network", network(a_graph()))
    assert again.ndisplay is None


def test_masks_are_shown_as_translucent_volume_layers():
    """Enabled masks arrive as Image volumes with the role colours, switched on."""
    from haemolynx.gui.results import MASK_COLOURS, MASK_LAYERS

    art = np.zeros((4, 4, 4), dtype=bool)
    art[1, 1, 1] = True
    group = built(voxel_size_zyx=(2.0, 1.0, 0.5)).stage_finished(
        "build_network",
        network(
            a_graph(),
            voxel_size_zyx=(2.0, 1.0, 0.5),
            large_arteriole_mask=art,
            large_venule_mask=np.zeros((4, 4, 4), dtype=bool),
        ),
    )
    names = {spec.name for spec in group.layers}
    assert MASK_LAYERS["large_arteriole_mask"] in names
    assert MASK_LAYERS["large_venule_mask"] in names
    assert MASK_LAYERS["small_arteriole_mask"] not in names
    assert MASK_LAYERS["small_venule_mask"] not in names

    mask = next(
        spec for spec in group.layers
        if spec.name == MASK_LAYERS["large_arteriole_mask"]
    )
    assert mask.kind == "image"
    assert mask.visible is True
    assert mask.scale == (2.0, 1.0, 0.5)
    assert mask.contrast_limits == (0.0, 1.0)
    assert mask.options["rendering"] == "mip"
    assert mask.options["blending"] == "translucent"
    assert mask.options["mask_colour"] == MASK_COLOURS["large_arteriole_mask"]
    assert mask.data.dtype == np.float32


def test_vessel_mask_volume_layers_skip_missing_and_keep_role_colours():
    from haemolynx.gui.results import (
        MASK_COLOURS,
        MASK_LAYERS,
        vessel_mask_volume_layers,
    )

    layers = vessel_mask_volume_layers(
        {
            "large_arteriole_mask": np.ones((2, 2, 2), dtype=bool),
            "small_venule_mask": np.zeros((2, 2, 2), dtype=bool),
        },
        voxel_size_zyx=(1.5, 1.0, 0.5),
    )
    assert [spec.name for spec in layers] == [
        MASK_LAYERS["large_arteriole_mask"],
        MASK_LAYERS["small_venule_mask"],
    ]
    by_name = {spec.name: spec for spec in layers}
    assert (
        by_name[MASK_LAYERS["large_arteriole_mask"]].options["mask_colour"]
        == MASK_COLOURS["large_arteriole_mask"]
    )
    assert (
        by_name[MASK_LAYERS["small_venule_mask"]].options["mask_colour"]
        == MASK_COLOURS["small_venule_mask"]
    )
    assert by_name[MASK_LAYERS["large_arteriole_mask"]].scale == (1.5, 1.0, 0.5)


def test_mask_role_colours_are_dark_and_light_red_and_green():
    """Large = dark, small = light; arterioles red, venules green."""
    from haemolynx.gui.results import MASK_COLOURS

    large_a = MASK_COLOURS["large_arteriole_mask"]
    small_a = MASK_COLOURS["small_arteriole_mask"]
    large_v = MASK_COLOURS["large_venule_mask"]
    small_v = MASK_COLOURS["small_venule_mask"]
    # Red channel dominates for arterioles; green for venules.
    assert large_a[0] > large_a[1] and large_a[0] > large_a[2]
    assert small_a[0] > small_a[1] and small_a[0] > small_a[2]
    assert large_v[1] > large_v[0] and large_v[1] > large_v[2]
    assert small_v[1] > small_v[0] and small_v[1] > small_v[2]
    # Light roles are brighter (higher RGB sum) than their dark counterparts.
    assert sum(small_a[:3]) > sum(large_a[:3])
    assert sum(small_v[:3]) > sum(large_v[:3])
    # Translucent enough to overlay the main image.
    assert all(0.0 < colour[3] < 1.0 for colour in MASK_COLOURS.values())


def test_the_hover_layer_carries_the_whole_table_and_starts_hidden():
    """A Vectors layer answers no hover query; this is what does."""
    group = built().stage_finished("build_network", network(a_graph()))
    labels = spec_named(group, VESSEL_LABELS)
    assert labels.visible is False
    assert set(labels.features) >= {"edge_index", "u", "v", "key", "length"}


def test_an_unknown_stage_is_ignored_rather_than_raising():
    group = ResultLayers().stage_finished("not_a_stage", object())
    assert group.layers == ()
    assert group.recolour == ()


def test_every_name_a_builder_emits_is_declared():
    """`LAYER_NAMES` is what "clear ours" and "is this ours" work from."""
    results = built()
    results.stage_finished(
        "assign_boundaries",
        SimpleNamespace(inlet_nodes=[0], outlet_nodes=[3],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=None),
    )
    graph = a_graph()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["pericyte_centers_um"] = [5.0]
    results.stage_finished("assign_diameters", SimpleNamespace(graph=graph))
    assert set(results.emitted) <= LAYER_NAMES


def test_every_declared_column_names_a_stage_that_exists():
    from haemolynx.gui.results import OPTIONAL_EDGE_COLUMNS
    from haemolynx.pipeline.progress import STAGES

    known = {stage.call for stage in STAGES}
    assert set(EDGE_COLUMNS.values()) <= known
    assert set(OPTIONAL_EDGE_COLUMNS.values()) <= known


# --- one layer per perturbation ----------------------------------------------


def solved_graph(flow: float = 1e-12) -> nx.MultiGraph:
    """A graph as a re-solve leaves it: flows on edges, pressures on nodes."""
    graph = a_graph()
    for index, node_id in enumerate(graph.nodes):
        graph.nodes[node_id]["pressure"] = 100.0 - 10.0 * index
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["resistance"] = 1e15
        data["conductance"] = 1e-15
        data["flow_abs"] = flow
        data["flow_signed"] = flow
    return graph


def a_perturbation(name: str, flow: float = 1e-12):
    """One `PerturbationResult`, using the real dataclass.

    The real one, so a field renamed in `pipeline/stages.py` fails here rather
    than showing an empty layer list months later.
    """
    from haemolynx.pipeline import PerturbationResult

    return PerturbationResult(
        name=name, type="arteriole_diameter_change", graph=solved_graph(flow)
    )


def a_perturbation_run(*results):
    from haemolynx.pipeline import PerturbationRun

    return PerturbationRun(output_dir=Path("outputs/perturbations"),
                           results=list(results))


def test_two_perturbations_give_two_distinctly_named_vessel_layers():
    """The point of the whole stage: two answers, side by side, told apart."""
    group = built().stage_finished(
        "run_perturbations",
        a_perturbation_run(a_perturbation("art_dilate_20"),
                           a_perturbation("art_constrict_20")),
    )

    vessels = [spec.name for spec in group.layers if spec.kind == "vectors"]
    assert vessels == [
        perturbation_layer_names("art_dilate_20")[0],
        perturbation_layer_names("art_constrict_20")[0],
    ]
    assert len(set(vessels)) == 2


def test_each_perturbation_gets_a_nodes_layer_of_its_own():
    group = built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    assert [spec.name for spec in group.layers] == list(
        perturbation_layer_names("art_dilate_20")
    )


def test_the_baseline_layers_are_untouched():
    """A perturbation must not overwrite the run it is compared against."""
    group = built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    names = {spec.name for spec in group.layers}
    assert VESSELS not in names
    assert NODES not in names
    assert VESSEL_LABELS not in names
    assert group.recolour == ()


def test_the_panel_goes_on_holding_the_baseline_graph():
    """`colour_options` reads the graph this class remembers -- the baseline's.

    A perturbation's graph carries the same columns here, so the check that
    means anything is identity: the remembered graph is still the one the
    earlier stages produced.
    """
    baseline = a_graph()
    results = built(baseline)

    results.stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )

    assert results._graph is baseline


def test_a_perturbations_layers_are_in_microns_like_every_graph_layer():
    """Graph-derived, so already physical: scale 1, not the voxel size."""
    group = built(voxel_size_zyx=(2.0, 1.0, 0.5)).stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    assert [spec.scale for spec in group.layers] == [(1.0, 1.0, 1.0)] * 2


def test_a_perturbations_layers_start_hidden():
    """Identical geometry to the baseline: a visible one just covers it."""
    group = built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    assert all(spec.visible is False for spec in group.layers)
    assert "hidden" in group.note


def test_the_vessels_are_coloured_by_the_same_quantity_as_the_baseline():
    """Comparing like with like is the only thing these layers are for."""
    group = built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    vessels = spec_named(group, perturbation_layer_names("art_dilate_20")[0])
    assert vessels.colour_by == DEFAULT_VESSEL_COLOUR["solve"]
    assert vessels.colour_kind == "continuous"


def test_each_perturbation_carries_its_own_numbers():
    """Two layers built from two graphs, not two views of one."""
    group = built().stage_finished(
        "run_perturbations",
        a_perturbation_run(a_perturbation("faster", flow=4e-12),
                           a_perturbation("slower", flow=1e-12)),
    )
    faster = spec_named(group, perturbation_layer_names("faster")[0])
    slower = spec_named(group, perturbation_layer_names("slower")[0])
    assert faster.features["flow_abs"].max() == pytest.approx(4e-12)
    assert slower.features["flow_abs"].max() == pytest.approx(1e-12)


def test_the_nodes_layer_carries_that_perturbations_pressures():
    """`set_edge_flows` writes them onto the graph; there is no Solution here."""
    group = built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("art_dilate_20"))
    )
    nodes = spec_named(group, perturbation_layer_names("art_dilate_20")[1])
    assert nodes.colour_by == "pressure"
    assert nodes.features["pressure"].tolist() == [100.0, 90.0, 80.0, 70.0]


def test_a_perturbation_with_no_network_gets_no_layer():
    """`type: none` does nothing, so there is nothing to draw."""
    from haemolynx.pipeline import PerturbationResult

    group = built().stage_finished(
        "run_perturbations",
        a_perturbation_run(PerturbationResult(name="off", type="none")),
    )
    assert group.layers == ()
    assert "No perturbation produced a network layer" in group.note


def test_a_failure_is_named_in_the_note_and_the_others_still_drawn():
    from haemolynx.pipeline import PerturbationResult

    group = built().stage_finished(
        "run_perturbations",
        a_perturbation_run(
            a_perturbation("art_dilate_20"),
            PerturbationResult(name="broken", type="pericyte_dilation_sweep",
                               error="ValueError: no inlet"),
        ),
    )
    assert len(group.layers) == 2
    assert "broken" in group.note
    assert "no inlet" in group.note


def test_a_perturbation_layer_is_ours_without_being_in_the_declared_set():
    """`LAYER_NAMES` cannot enumerate a name a config invents; the prefix can."""
    for name in perturbation_layer_names("art_dilate_20"):
        assert is_ours_name(name)
        assert name not in LAYER_NAMES
    assert all(is_ours_name(name) for name in LAYER_NAMES)
    assert not is_ours_name("art_dilate_20 vessels")


def test_a_perturbations_name_is_all_that_separates_its_layers_from_ours():
    """So a perturbation called "vessels" cannot become the baseline's layer."""
    vessels, nodes = perturbation_layer_names("vessels")
    assert vessels != VESSELS
    assert nodes != NODES


# --- a run that was stopped part-way -----------------------------------------


def test_resetting_forgets_the_graph_the_run_had_got_to():
    """The graph is remembered across stages, so an abandoned run leaves one."""
    results = built()
    assert results.colour_options()

    results.reset()

    assert results.colour_options() == []
    assert results.emitted == ()


def test_the_next_run_after_a_reset_starts_from_nothing():
    """Not just emptied: usable again, and back to showing the geometry whole."""
    results = built()
    results.reset()

    group = results.stage_finished("build_network", network(a_graph()))

    assert spec_named(group, VESSELS)
    assert group.ndisplay == 3


# --- it must not need a GUI --------------------------------------------------


def test_the_module_imports_no_gui():
    """The library must import on a machine with no napari and no Qt."""
    probe = (
        "import sys; import haemolynx.gui.results; "
        "print([m for m in sys.modules if m.split('.')[0] in "
        "{'napari', 'magicgui', 'qtpy', 'PyQt6', 'PyQt5', 'PySide6'}])"
    )
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_no_napari_import_appears_in_the_source():
    tree = ast.parse((REPO_ROOT / "src" / "haemolynx" / "gui" / "results.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"napari", "magicgui", "qtpy"}


# --- the topology steps, when asked for --------------------------------------


def test_topology_steps_are_not_drawn_unless_asked_for():
    """Eleven extra rebuilds in the middle of the slowest stage, so: opt in."""
    results = built()
    group = results.stage_finished("topology_step:prune_vascular_stubs", a_graph())
    assert group.layers == ()


def test_a_topology_step_redraws_the_vessels_when_switched_on():
    results = ResultLayers(show_steps=True)
    results.stage_finished("build_network", network(a_graph()))

    group = results.stage_finished("topology_step:prune_vascular_stubs", a_graph())

    assert [spec.name for spec in group.layers] == [VESSELS, VESSEL_LABELS]
    assert "prune_vascular_stubs" in group.title
    assert "3 vessels" in group.note


def test_a_step_does_not_replace_the_graph_the_stage_produced():
    """The step's graph is mid-repair and will change again; do not keep it."""
    results = ResultLayers(show_steps=True)
    finished_graph = a_graph()
    results.stage_finished("build_network", network(finished_graph))

    half_built = nx.MultiGraph()
    half_built.add_node(0, pos=np.zeros(3))
    results.stage_finished("topology_step:collapse_node_clusters", half_built)

    later = results.stage_finished("solve", SimpleNamespace(
        pressure=np.zeros(4), node_list=[0, 1, 2, 3], equivalent_resistance=1.0))
    assert len(spec_named(later, NODES).data) == 4


# --- napari's own layer controls read the column list exactly once ------------


def test_every_column_is_declared_before_its_stage_fills_it():
    """The vessels layer carries the full column set from the first stage.

    napari builds the "edge feature:" dropdown in the layer controls from
    `features.columns` in its constructor and never listens for a features
    change, so a column that only appears at the solve is invisible there for
    the rest of the session. Declaring them all up front -- NaN until filled --
    is what puts flow and pressure in that list.
    """
    from haemolynx.gui.results import EDGE_COLUMNS

    layers = built(a_graph()).stage_finished(
        "build_network", network(a_graph())
    ).layers
    vessels = next(spec for spec in layers if spec.name == VESSELS)

    assert set(EDGE_COLUMNS) <= set(vessels.features), (
        set(EDGE_COLUMNS) - set(vessels.features)
    )
    # And unfilled means empty, not absent or zero: zero is a flow.
    assert np.isnan(np.asarray(vessels.features["flow_abs"], dtype=float)).all()

    nodes = next(spec for spec in layers if spec.name == NODES)
    assert "pressure" in nodes.features
    assert np.isnan(np.asarray(nodes.features["pressure"], dtype=float)).all()


def test_the_column_set_does_not_change_between_stages():
    """Same columns at build_network and at solve, so nothing appears late."""
    graph = a_graph(conductance=1e-18, resistance=1e18)
    results = built(graph)
    early = results.stage_finished("build_network", network(graph))

    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 5e-16
        data["flow_signed"] = -5e-16
        data["pressure_drop"] = 500.0
    late = results.stage_finished(
        "solve",
        SimpleNamespace(node_list=list(graph.nodes),
                        pressure=np.array([1000.0, 900.0, 700.0, 500.0])),
    )

    def columns(group, name):
        return set(next(s for s in group.layers if s.name == name).features)

    assert columns(early, VESSELS) == columns(late, VESSELS)
    assert columns(early, NODES) == columns(late, NODES)


def test_a_declared_column_still_carries_the_value_once_it_is_filled():
    """Declaring early must not mean the real value is lost later."""
    graph = a_graph(conductance=1e-18)
    results = built(graph)
    results.stage_finished("build_network", network(graph))
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 7e-16

    group = results.stage_finished(
        "solve",
        SimpleNamespace(node_list=list(graph.nodes),
                        pressure=np.array([1000.0, 900.0, 700.0, 500.0])),
    )
    vessels = next(s for s in group.layers if s.name == VESSELS)
    assert np.allclose(np.asarray(vessels.features["flow_abs"], dtype=float), 7e-16)


def test_vessel_layers_include_flow_abs_log10_when_setting_enabled():
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    graph = a_graph(conductance=1e-18)
    results = ResultLayers(settings={"flow_log_scale": True})
    results.stage_finished("skeletonise", SimpleNamespace(
        image=np.zeros((4, 4, 4)), skeleton=np.zeros((4, 4, 4), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0),
    ))
    results.stage_finished("build_network", network(graph))
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 1e-12
        data["flow_abs_log10"] = flow_abs_log10_value(1e-12)

    group = results.stage_finished(
        "solve",
        SimpleNamespace(node_list=list(graph.nodes),
                        pressure=np.array([1000.0, 900.0, 700.0, 500.0])),
    )
    vessels = next(s for s in group.layers if s.name == VESSELS)
    assert "flow_abs_log10" in vessels.features
    assert np.allclose(
        np.asarray(vessels.features["flow_abs_log10"], dtype=float),
        flow_abs_log10_value(1e-12),
    )
    assert spec_named(group, VESSELS).colour_by == "flow_abs"


def test_vessel_layers_include_flow_abs_log10_even_when_setting_disabled():
    """Log10 flow is always a layer-panel option once flows exist."""
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    graph = a_graph(conductance=1e-18)
    results = ResultLayers(settings={"flow_log_scale": False})
    results.stage_finished("skeletonise", SimpleNamespace(
        image=np.zeros((4, 4, 4)), skeleton=np.zeros((4, 4, 4), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0),
    ))
    results.stage_finished("build_network", network(graph))
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 1e-12

    group = results.stage_finished(
        "solve",
        SimpleNamespace(node_list=list(graph.nodes),
                        pressure=np.array([1000.0, 900.0, 700.0, 500.0])),
    )
    vessels = next(s for s in group.layers if s.name == VESSELS)
    assert "flow_abs_log10" in vessels.features
    assert np.allclose(
        np.asarray(vessels.features["flow_abs_log10"], dtype=float),
        flow_abs_log10_value(1e-12),
    )


# --- the range a colour bar should span --------------------------------------


def test_trimming_the_extremes_rescues_a_long_tailed_column():
    """Why the panel offers "Fit 1-99%" as well as "Fit all".

    Flow is long-tailed: a few vessels carry orders of magnitude more than the
    rest. Against the full range every other vessel lands in the bottom
    percent of the colormap and the network reads as one flat colour, which is
    exactly the failure that looks like "the flow is not being shown".
    """
    from types import SimpleNamespace

    from haemolynx.gui._widget import _data_range

    values = np.concatenate([np.linspace(1e-16, 2e-16, 200), [1e-9]])
    layer = SimpleNamespace(features={"flow_abs": values})

    full = _data_range(layer, "flow_abs")
    trimmed = _data_range(layer, "flow_abs", 1.0, 99.0)

    assert full == (pytest.approx(1e-16), pytest.approx(1e-9))
    assert trimmed[1] < full[1] / 1000, (full, trimmed)
    # The bulk of the data now spans most of the bar rather than a sliver.
    inside = np.mean((values >= trimmed[0]) & (values <= trimmed[1]))
    assert inside > 0.95


def test_a_column_with_nothing_in_it_has_no_range():
    from types import SimpleNamespace

    from haemolynx.gui._widget import _data_range

    layer = SimpleNamespace(features={"flow_abs": np.full(5, np.nan)})
    assert _data_range(layer, "flow_abs") is None
    assert _data_range(layer, "missing") is None
    assert _data_range(layer, "none") is None


# --- Z depth filter (view-only) ----------------------------------------------


def test_image_z_extent_um_is_voxel_z_times_stack_depth():
    assert image_z_extent_um((2.0, 1.0, 0.5), 8) == pytest.approx(16.0)


def test_z_depth_filter_targets_graph_vectors_and_points_only():
    assert is_z_depth_filtered_layer(VESSELS, "vectors") is True
    assert is_z_depth_filtered_layer(FLOW_DIRECTION, "vectors") is True
    assert is_z_depth_filtered_layer(NODES, "points") is True
    assert is_z_depth_filtered_layer(IMAGE, "image") is False
    assert is_z_depth_filtered_layer(SKELETON, "labels") is False
    assert is_z_depth_filtered_layer("HaemoLynx BC coordinates", "points") is False

def _flow_graph_at_z(*z_values: float) -> nx.MultiGraph:
    """One edge per Z slab; mid-edge arrow anchors sit near z + 5."""
    graph = nx.MultiGraph()
    for z in z_values:
        u, v = int(z), int(z) + 100
        graph.add_node(u, pos=np.array([float(z), 0.0, 0.0]))
        graph.add_node(v, pos=np.array([float(z + 5), 0.0, 0.0]))
        graph.add_edge(
            u,
            v,
            key=0,
            voxels=[[z, 0.0, 0.0], [z + 5, 0.0, 0.0]],
            length=5.0,
            flow_signed=1.0,
            flow_abs=1.0,
            conductance=1.0,
        )
    return graph


def test_filter_flow_direction_vectors_by_z_uses_arrow_origin_z():
    vectors, features = flow_direction_vectors(_flow_graph_at_z(0, 10, 20))
    assert len(vectors) == 3
    assert np.allclose(vectors[:, 0, 0], [5.0, 15.0, 25.0])
    z_mid = 16.0
    filtered, feats, _owner = filter_vectors_by_z(vectors, features, 0.0, z_mid)
    assert len(filtered) == 2
    assert np.all(filtered[:, 0, 0] <= z_mid)
    assert len(feats["flow_abs"]) == len(filtered)


def test_filter_vectors_by_z_keeps_segments_whose_origin_is_in_range():
    paths, identity = edge_polylines(a_graph())
    vectors, owner = polylines_to_vectors(paths)
    features = {"edge_index": identity["edge_index"][owner]}
    z_mid = 15.0
    filtered, feats, _owner = filter_vectors_by_z(vectors, features, 0.0, z_mid)
    # Edge start Z values are 0, 10, 20 along the line graph.
    assert len(filtered) == 2
    assert np.all(filtered[:, 0, 0] <= z_mid)
    assert len(feats["edge_index"]) == len(filtered)


def test_filter_vectors_by_z_at_full_range_restores_all_segments():
    paths, _identity = edge_polylines(a_graph())
    vectors, owner = polylines_to_vectors(paths)
    features = {"edge_index": owner}
    full_z = 30.0
    filtered, feats, _owner = filter_vectors_by_z(vectors, features, 0.0, full_z)
    assert len(filtered) == len(vectors)
    assert len(feats["edge_index"]) == len(filtered)


def test_filter_points_by_z_keeps_midpoints_up_to_z_mid():
    paths, identity = edge_polylines(a_graph())
    midpoints = midpoints_of(paths)
    features = {"edge_index": identity["edge_index"]}
    z_mid = 15.0
    filtered, feats = filter_points_by_z(midpoints, features, 0.0, z_mid)
    assert len(filtered) == 2
    assert np.all(filtered[:, 0] <= z_mid)
    assert len(feats["edge_index"]) == len(filtered)


def test_apply_z_filter_on_viewer_restores_full_range():
    from haemolynx.gui._widget import _apply_z_filter

    graph = a_graph()
    results = built(graph)
    group = results.stage_finished("build_network", network(graph))
    vessels = spec_named(group, VESSELS)
    full_z = 30.0

    class Vectors:
        pass

    layer = Vectors()
    layer.name = VESSELS
    layer.data = np.asarray(vessels.data)
    layer.features = dict(vessels.features)
    layer.metadata = {
        "haemolynx": {
            "z_filter_full": {
                "data": np.asarray(vessels.data),
                "features": {
                    k: np.asarray(v) for k, v in vessels.features.items()
                },
            }
        }
    }
    viewer = SimpleNamespace(layers=[layer])

    _apply_z_filter(viewer, 0.0, 15.0, z_extent=full_z)
    assert len(layer.data) == 2
    _apply_z_filter(viewer, 0.0, full_z, z_extent=full_z)
    assert len(layer.data) == len(vessels.data)


def test_apply_z_filter_on_viewer_filters_flow_direction_layer():
    from haemolynx.gui._widget import _apply_z_filter

    vectors, features = flow_direction_vectors(_flow_graph_at_z(0, 10, 20))
    full_z = 25.0

    class Vectors:
        pass

    layer = Vectors()
    layer.name = FLOW_DIRECTION
    layer.data = np.asarray(vectors)
    layer.features = dict(features)
    layer.metadata = {
        "haemolynx": {
            "z_filter_full": {
                "data": np.asarray(vectors),
                "features": {k: np.asarray(v) for k, v in features.items()},
            }
        }
    }
    viewer = SimpleNamespace(layers=[layer])

    _apply_z_filter(viewer, 0.0, 16.0, z_extent=full_z)
    assert len(layer.data) == 2
    assert np.all(layer.data[:, 0, 0] <= 16.0)
    _apply_z_filter(viewer, 0.0, full_z, z_extent=full_z)
    assert len(layer.data) == len(vectors)

def test_result_layers_image_z_extent_after_skeletonise():
    results = ResultLayers()
    assert results.image_z_extent_um() is None
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((6, 4, 4)),
            skeleton=np.zeros((6, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 2.0),
            voxel_size_zyx=(2.0, 1.0, 1.0),
        ),
    )
    assert results.image_z_extent_um() == pytest.approx(12.0)


def test_an_identical_column_still_gives_a_usable_range():
    """A degenerate range would make napari refuse the limits outright."""
    from types import SimpleNamespace

    from haemolynx.gui._widget import _data_range

    low, high = _data_range(SimpleNamespace(features={"q": np.full(4, 3.0)}), "q")
    assert high > low
