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

from ImageLynx.geometry import cumulative_lengths
from ImageLynx.graph._helpers import edge_id, sort_nodes
from ImageLynx.io import load_binary_mask_and_voxel_size


# --- cumulative_lengths ----------------------------------------------------


def test_cumulative_lengths_starts_at_zero_and_ends_at_total_length():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 4.0, 3.0]])
    lengths = cumulative_lengths(points)
    assert lengths.shape == (3,)
    assert lengths[0] == 0.0
    np.testing.assert_allclose(lengths, [0.0, 3.0, 7.0])


def test_cumulative_lengths_has_one_definition_shared_by_both_consumers():
    from ImageLynx import geometry
    from ImageLynx.visualization import vtk_io

    assert vtk_io.cumulative_lengths is geometry.cumulative_lengths


# --- edge_id ---------------------------------------------------------------


def test_edge_id_is_orientation_independent():
    assert edge_id(7, 2, 0) == edge_id(2, 7, 0) == (2, 7, 0)


def test_edge_id_keeps_parallel_edges_distinct():
    assert edge_id(2, 7, 0) != edge_id(2, 7, 1)


def test_edge_id_has_one_definition_shared_by_both_consumers():
    from ImageLynx.graph import automated_vessel_assignment, branch_order

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
    from ImageLynx.graph import automated_vessel_assignment, boundaries

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
    import importlib

    from ImageLynx import io
    from ImageLynx.haemodynamics import pericyte_mask

    distances = importlib.import_module("ImageLynx.statistics.3D_distances")

    assert pericyte_mask.load_binary_mask_and_voxel_size is io.load_binary_mask_and_voxel_size
    assert distances.load_binary_mask_and_voxel_size is io.load_binary_mask_and_voxel_size


# --- deliberately-not-merged pairs -----------------------------------------


def _two_node_graph(voxels) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.asarray([0.0, 0.0, 10.0]))
    graph.add_edge(0, 1, key=0, voxels=voxels)
    return graph


def test_edge_point_accessors_differ_on_a_two_dimensional_polyline():
    """Why ``_edge_points`` could not become one function.

    The VTK exporter pads a 2D polyline out to three columns; the haemodynamics
    centerline accessor rejects it and falls back to the node-to-node segment,
    because a padded polyline would move a projected pericyte.
    """
    from ImageLynx.haemodynamics.pericyte_mask import _edge_centerline_points
    from ImageLynx.visualization.vtk_io import _edge_points_padded_to_3d

    flat = [[0.0, 0.0], [0.0, 6.0], [0.0, 10.0]]
    graph = _two_node_graph(flat)
    data = graph[0][1][0]

    padded = _edge_points_padded_to_3d(0, 1, data, graph)
    centerline = _edge_centerline_points(graph, 0, 1, data)

    assert padded.shape == (3, 3)
    assert centerline.shape == (2, 3)


def test_terminal_node_accessors_return_different_shapes():
    """Why ``_terminal_nodes_with_positions`` could not become one function."""
    from ImageLynx.graph.automated_vessel_assignment import (
        _terminal_nodes_with_position_pairs,
    )
    from ImageLynx.graph.boundaries import _terminal_nodes_and_position_map

    graph = _two_node_graph([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])

    pairs = _terminal_nodes_with_position_pairs(graph)
    terminals, position_map = _terminal_nodes_and_position_map(graph)

    assert [node for node, _ in pairs] == terminals
    assert isinstance(position_map, dict)
    assert set(position_map) == set(terminals)
