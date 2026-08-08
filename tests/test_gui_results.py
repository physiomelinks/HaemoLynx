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
    BOUNDARY_NODES,
    EDGE_COLUMNS,
    IMAGE,
    LAYER_NAMES,
    NODES,
    PERICYTES,
    SKELETON,
    VESSEL_LABELS,
    VESSELS,
    ResultLayers,
    edge_features,
    edge_polylines,
    midpoints_of,
    node_points,
    pericyte_points,
    polylines_to_vectors,
)

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


def test_columns_appear_only_once_their_stage_has_run():
    graph = a_graph()
    before = built(graph).stage_finished("build_network", network(graph))
    assert "flow_abs" not in spec_named(before, VESSELS).features

    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["flow_abs"] = 2.0
    after = built(graph).stage_finished("build_network", network(graph))
    assert "flow_abs" in spec_named(after, VESSELS).features


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
        SimpleNamespace(starting_nodes=[0], output_nodes=[3],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=(0, 3)),
    )
    boundaries = spec_named(group, BOUNDARY_NODES)
    assert boundaries.features["role"].tolist() == ["starting", "output"]
    assert boundaries.colour_kind == "categorical"
    assert len(boundaries.data) == 2


def test_a_run_with_no_boundary_nodes_says_so_rather_than_drawing_nothing():
    group = built().stage_finished(
        "assign_boundaries",
        SimpleNamespace(starting_nodes=[], output_nodes=[],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=None),
    )
    assert "No boundary nodes" in group.note


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


def test_masks_are_shown_but_not_switched_on():
    group = built().stage_finished(
        "build_network",
        network(a_graph(), large_arteriole_mask=np.zeros((4, 4, 4), dtype=bool)),
    )
    mask = next(spec for spec in group.layers if "arteriole mask" in spec.name)
    assert mask.kind == "labels"
    assert mask.visible is False


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
        SimpleNamespace(starting_nodes=[0], output_nodes=[3],
                        arteriole_boundary_nodes=[], venule_boundary_nodes=[],
                        resistance_node_pair=None),
    )
    graph = a_graph()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["pericyte_centers_um"] = [5.0]
    results.stage_finished("assign_diameters", SimpleNamespace(graph=graph))
    assert set(results.emitted) <= LAYER_NAMES


def test_every_declared_column_names_a_stage_that_exists():
    from haemolynx.pipeline.progress import STAGES

    known = {stage.call for stage in STAGES}
    assert set(EDGE_COLUMNS.values()) <= known


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
