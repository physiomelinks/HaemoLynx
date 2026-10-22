"""Guards on helpers that used to exist as copies in two modules.

Four helpers were defined twice, byte-for-byte or near enough that the copies
had already started to drift (``_sort_nodes`` deduplicated its input in one
module and not the other). These tests pin the single shared implementation and
the identity of the call sites, so a future copy-paste shows up as a failure.

Two further pairs shared a *name* but not behaviour; the last tests here pin the
differences that make them unmergeable.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
import tifffile

from haemolynx.geometry import cumulative_lengths
from haemolynx.graph._helpers import edge_id, sort_nodes
from haemolynx.io import load_binary_mask_and_voxel_size


# --- cumulative_lengths ----------------------------------------------------


def test_cumulative_lengths_starts_at_zero_and_ends_at_total_length():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 4.0, 3.0]])
    lengths = cumulative_lengths(points)
    assert lengths.shape == (3,)
    assert lengths[0] == 0.0
    np.testing.assert_allclose(lengths, [0.0, 3.0, 7.0])


def test_cumulative_lengths_has_one_definition_shared_by_both_consumers():
    from haemolynx import geometry
    from haemolynx.visualization import vtk_io

    assert vtk_io.cumulative_lengths is geometry.cumulative_lengths


# --- edge_id ---------------------------------------------------------------


def test_edge_id_is_orientation_independent():
    assert edge_id(7, 2, 0) == edge_id(2, 7, 0) == (2, 7, 0)


def test_edge_id_keeps_parallel_edges_distinct():
    assert edge_id(2, 7, 0) != edge_id(2, 7, 1)


def test_edge_id_has_one_definition_shared_by_both_consumers():
    from haemolynx.graph import automated_vessel_assignment, branch_order

    assert branch_order.edge_id is edge_id
    assert automated_vessel_assignment._edge_id is edge_id


# --- sort_nodes ------------------------------------------------------------


def test_sort_nodes_orders_mixed_types_that_python_cannot_compare():
    assert sort_nodes({3, "a", 1}) == [1, 3, "a"]


def test_sort_nodes_is_deterministic_regardless_of_input_order():
    assert sort_nodes([5, 1, 22]) == sort_nodes([22, 5, 1]) == [1, 22, 5]


def test_sort_nodes_deduplicates():
    """The two former copies disagreed here; the deduplicating one wins."""
    assert sort_nodes([2, 2, 1]) == [1, 2]


def test_sort_nodes_has_one_definition_shared_by_both_consumers():
    from haemolynx.graph import automated_vessel_assignment, boundaries

    assert boundaries.sort_nodes is sort_nodes
    assert automated_vessel_assignment._sort_nodes is sort_nodes


# --- load_binary_mask_and_voxel_size ---------------------------------------


def test_load_binary_mask_treats_any_positive_voxel_as_foreground(tmp_path):
    mask = np.zeros((3, 4, 5), dtype=np.uint8)
    mask[1, 2, 3] = 7
    path = tmp_path / "mask.tif"
    tifffile.imwrite(str(path), mask)

    loaded, voxel_size_xyz = load_binary_mask_and_voxel_size(path)

    assert loaded.dtype == bool
    assert loaded.shape == (3, 4, 5)
    assert loaded.sum() == 1 and loaded[1, 2, 3]
    assert len(voxel_size_xyz) == 3


def test_load_binary_mask_keeps_one_two_label_masks_sparse(tmp_path):
    """``1/2`` encodings must not become all-True (``> 0`` would)."""
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 2
    path = tmp_path / "one_two.tif"
    tifffile.imwrite(str(path), mask)

    loaded, _voxel_size_xyz = load_binary_mask_and_voxel_size(path)

    assert loaded.dtype == bool
    assert int(loaded.sum()) == 8
    assert bool(loaded[1, 1, 1]) and not bool(loaded[0, 0, 0])


def test_load_binary_mask_rejects_unsupported_format(tmp_path):
    path = tmp_path / "mask.npy"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported pericyte mask format"):
        load_binary_mask_and_voxel_size(path, description="pericyte mask")


def test_load_binary_mask_rejects_non_3d_volume(tmp_path):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected a 3D mask"):
        load_binary_mask_and_voxel_size(path)


def test_load_binary_mask_is_the_one_used_by_both_consumers():
    """The pericyte mask and the 3D distance measurement's masks also need
    their voxel-size metadata status (to apply the run's override and policy,
    as the main image does), so they read through the status variant -- which
    is the one loader the plain one wraps, not a second reading path."""
    from haemolynx.haemodynamics import pericyte_mask
    from haemolynx.io import load
    from haemolynx.statistics import three_dim_distances as distances

    assert (
        pericyte_mask.load_binary_mask_voxel_size_and_status
        is load.load_binary_mask_voxel_size_and_status
    )
    assert (
        distances.load_binary_mask_voxel_size_and_status
        is load.load_binary_mask_voxel_size_and_status
    )


def test_the_plain_mask_loader_reads_through_the_status_one(monkeypatch, tmp_path):
    from haemolynx.io import load

    calls = []

    def fake(mask_path, **kwargs):
        calls.append((mask_path, kwargs))
        return np.ones((2, 2, 2), dtype=bool), (0.5, 0.5, 2.0), {"complete": True}

    monkeypatch.setattr(load, "load_binary_mask_voxel_size_and_status", fake)
    mask, voxel_size_xyz = load.load_binary_mask_and_voxel_size(
        tmp_path / "m.tif", use_memmap=True, memmap_directory=tmp_path
    )

    assert mask.shape == (2, 2, 2) and voxel_size_xyz == (0.5, 0.5, 2.0)
    assert calls and calls[0][1]["use_memmap"] is True
    assert calls[0][1]["memmap_directory"] == tmp_path


# --- deliberately-not-merged pairs -----------------------------------------


def _two_node_graph(voxels) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.asarray([0.0, 0.0, 10.0]))
    graph.add_edge(0, 1, key=0, voxels=voxels)
    return graph


def test_edge_point_accessors_differ_on_a_two_dimensional_polyline():
    """Why drawing an edge and measuring along one could not become one function.

    Anything that draws a vessel pads a 2D polyline out to three columns, since
    a flat graph is still worth showing; the haemodynamics centerline accessor
    rejects it and falls back to the node-to-node segment, because a padded
    polyline would move a projected pericyte.
    """
    from haemolynx.haemodynamics.pericyte_mask import _edge_centerline_points
    from haemolynx.visualization.geometry import edge_polyline

    flat = [[0.0, 0.0], [0.0, 6.0], [0.0, 10.0]]
    graph = _two_node_graph(flat)
    data = graph[0][1][0]

    padded = edge_polyline(graph, 0, 1, data, orient=False, snap=False)
    centerline = _edge_centerline_points(graph, 0, 1, data)

    assert padded.shape == (3, 3)
    assert centerline.shape == (2, 3)


def test_terminal_node_accessors_return_different_shapes():
    """Why ``_terminal_nodes_with_positions`` could not become one function."""
    from haemolynx.graph.automated_vessel_assignment import (
        _terminal_nodes_with_position_pairs,
    )
    from haemolynx.graph.boundaries import _terminal_nodes_and_position_map

    graph = _two_node_graph([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])

    pairs = _terminal_nodes_with_position_pairs(graph)
    terminals, position_map = _terminal_nodes_and_position_map(graph)

    assert [node for node, _ in pairs] == terminals
    assert isinstance(position_map, dict)
    assert set(position_map) == set(terminals)


# --- densify_polyline: vectorised, the same floats as the loop it replaced ------


def _densify_by_loop(points, *, max_step_um):
    """densify_polyline as it was, point by point (the reference)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] <= 1:
        return arr
    dense = [arr[0]]
    for start, end in zip(arr[:-1], arr[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= max_step_um:
            dense.append(end)
            continue
        steps = int(np.ceil(length / max_step_um))
        for step in range(1, steps + 1):
            dense.append(start + (step / steps) * segment)
    return np.asarray(dense, dtype=float)


@pytest.mark.parametrize("max_step_um", [0.25, 0.5, 0.98, 1.0, 3.0])
def test_densify_polyline_gives_the_loops_floats_exactly(max_step_um):
    from haemolynx.graph._helpers import densify_polyline

    rng = np.random.default_rng(20261022)
    paths = [np.cumsum(rng.normal(size=(int(rng.integers(2, 60)), 3)) * scale, axis=0)
             for scale in (0.1, 0.5, 0.98, 1.0, 4.0, 40.0) for _ in range(40)]
    # Steps of exactly one and two step lengths, repeated points, a voxel staircase.
    paths.append(np.array([[0, 0, 0], [0, 0, max_step_um], [0, 0, 3 * max_step_um], [0, 0, 3 * max_step_um]], float))
    paths.append(np.array([[0, 0, 0], [1.0, 0.98, 0.98], [2.0, 0.98, 1.96], [2.0, 1.96, 2.94]]))
    for path in paths:
        expected = _densify_by_loop(path, max_step_um=max_step_um)
        got = densify_polyline(path, max_step_um=max_step_um)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_densify_polyline_leaves_a_point_or_an_empty_path_as_it_is():
    from haemolynx.graph._helpers import densify_polyline

    one = np.array([[1.0, 2.0, 3.0]])
    assert np.array_equal(densify_polyline(one, max_step_um=0.5), one)
    assert densify_polyline(np.empty((0, 3)), max_step_um=0.5).shape == (0, 3)
    with pytest.raises(ValueError):
        densify_polyline(one, max_step_um=0.0)
