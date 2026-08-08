"""One definition of where a vessel is.

The VTK export, the pericyte points and the napari layers all have to agree on
the polyline for an edge, or the same vessel sits in a different place in
ParaView, in an HTML plot and in the viewer. These pin the three questions that
answer costs: is there a polyline, does it run u -> v, and do its ends sit on
the nodes.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.visualization.geometry import as_points, edge_polyline


def _graph(u_pos=(0.0, 0.0, 0.0), v_pos=(0.0, 0.0, 10.0), **edge) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.asarray(u_pos, dtype=float))
    graph.add_node("v", pos=np.asarray(v_pos, dtype=float))
    graph.add_edge("u", "v", key=0, **edge)
    return graph


def _polyline(graph, **kwargs) -> np.ndarray:
    data = graph.edges["u", "v", 0]
    return edge_polyline(graph, "u", "v", data, **kwargs)


# --- shaping ----------------------------------------------------------------


def test_a_polyline_comes_back_as_three_float_columns():
    graph = _graph(voxels=[[0, 0, 0], [0, 0, 5], [0, 0, 10]])
    points = _polyline(graph)
    assert points.shape == (3, 3)
    assert points.dtype == float


def test_a_two_column_polyline_is_padded_not_rejected():
    """A flat graph is still worth drawing; the consumer needs a fixed width."""
    assert as_points([[1, 2], [3, 4]]).tolist() == [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]


def test_an_over_wide_polyline_is_truncated():
    assert as_points([[1, 2, 3, 9], [4, 5, 6, 9]]).tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


def test_a_single_point_is_not_a_polyline():
    with pytest.raises(ValueError, match="at least two points"):
        as_points([[1.0, 2.0, 3.0]])


# --- what to draw when there is no path -------------------------------------


def test_an_edge_with_no_voxels_falls_back_to_its_two_nodes():
    graph = _graph()
    assert _polyline(graph).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]


def test_a_one_point_voxel_path_falls_back_too():
    """Fewer than two voxels is not a path; the nodes still describe the edge."""
    graph = _graph(voxels=[[0, 0, 5]])
    assert _polyline(graph).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]


def test_an_edge_with_neither_path_nor_positions_has_nothing_to_draw():
    graph = nx.MultiGraph()
    graph.add_node("u")
    graph.add_node("v")
    graph.add_edge("u", "v", key=0)
    with pytest.raises(ValueError, match="missing both voxels and node positions"):
        edge_polyline(graph, "u", "v", graph.edges["u", "v", 0])


# --- orientation ------------------------------------------------------------


def test_a_path_running_v_to_u_is_turned_round():
    """Skeleton paths come out in whichever order the tracer walked them."""
    graph = _graph(voxels=[[0, 0, 10], [0, 0, 5], [0, 0, 0]])
    points = _polyline(graph)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[-1].tolist() == [0.0, 0.0, 10.0]


def test_a_path_already_running_u_to_v_is_left_alone():
    graph = _graph(voxels=[[0, 0, 0], [0, 0, 5], [0, 0, 10]])
    assert _polyline(graph)[0].tolist() == [0.0, 0.0, 0.0]


def test_orientation_can_be_switched_off():
    """The pericyte derivation walks the path as stored, unturned."""
    graph = _graph(voxels=[[0, 0, 10], [0, 0, 5], [0, 0, 0]])
    assert _polyline(graph, orient=False, snap=False)[0].tolist() == [0.0, 0.0, 10.0]


# --- snapping ---------------------------------------------------------------


def test_the_ends_are_pulled_onto_the_nodes():
    """Cluster collapse moves a node, leaving its polyline short of it.

    Without this there is a visible gap at every junction in anything that
    draws lines.
    """
    graph = _graph(voxels=[[0, 0, 1], [0, 0, 5], [0, 0, 9]])
    points = _polyline(graph)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[-1].tolist() == [0.0, 0.0, 10.0]
    # The middle is untouched.
    assert points[1].tolist() == [0.0, 0.0, 5.0]


def test_snapping_can_be_switched_off():
    graph = _graph(voxels=[[0, 0, 1], [0, 0, 5], [0, 0, 9]])
    assert _polyline(graph, snap=False)[0].tolist() == [0.0, 0.0, 1.0]


def test_the_stored_path_is_not_modified():
    """The graph is shared; drawing it must not edit it."""
    voxels = [[0, 0, 9], [0, 0, 5], [0, 0, 1]]
    graph = _graph(voxels=voxels)
    _polyline(graph)
    assert voxels == [[0, 0, 9], [0, 0, 5], [0, 0, 1]]


def test_a_node_without_a_position_leaves_that_end_alone():
    graph = _graph(voxels=[[0, 0, 1], [0, 0, 9]])
    del graph.nodes["u"]["pos"]
    points = _polyline(graph)
    assert points[0].tolist() == [0.0, 0.0, 1.0]
    assert points[-1].tolist() == [0.0, 0.0, 10.0]
