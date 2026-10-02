"""Tests for hop-distance fallback and disconnected I/O component cleanup."""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph import (
    FLOW_SOLVED,
    mark_flow_solved_edges,
    remove_components_without_connected_io,
    seed_edges_have_full_mask_coverage,
    select_nodes_at_hop_distance,
)
from haemolynx.pipeline import default_schema


def test_select_nodes_at_hop_distance_exact_ring():
    G = nx.MultiGraph()
    for node in range(5):
        G.add_node(node)
    for node in range(4):
        G.add_edge(node, node + 1)
    assert select_nodes_at_hop_distance(G, [0], 2) == [2]
    assert select_nodes_at_hop_distance(G, [0], 1, exclude_nodes={1}) == []


def test_seed_edges_have_full_mask_coverage_detects_miss():
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1, voxels=[[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    mask = np.zeros((4, 4, 4), dtype=bool)
    covered, uncovered, total = seed_edges_have_full_mask_coverage(G, [0], mask)
    assert covered is False
    assert uncovered == 1
    assert total == 1
    mask[1, 1, 1] = True
    covered, uncovered, total = seed_edges_have_full_mask_coverage(G, [0], mask)
    assert covered is True
    assert uncovered == 0
    assert total == 1


def test_remove_components_without_connected_io_keeps_shared_component():
    G = nx.MultiGraph()
    # Component A: inlet 0 -- 1 -- outlet 2
    G.add_edges_from([(0, 1), (1, 2)])
    # Component B: orphan 10 -- 11
    G.add_edge(10, 11)
    pruned, stats = remove_components_without_connected_io(G, [0], [2])
    assert set(pruned.nodes) == {0, 1, 2}
    assert stats["removed_components"] == 1
    assert stats["removed_nodes"] == 2


def _three_pieces() -> nx.MultiGraph:
    """inlet 0 -- 1 -- outlet 2 (with a parallel 1-2 vessel); inlet 10 -- 11
    with no outlet; 20 -- 21 with no boundary at all."""
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 2), (1, 2), (10, 11), (20, 21)])
    return G


def test_mark_flow_solved_edges_marks_what_removal_would_remove():
    G = _three_pieces()
    inlets, outlets = [0, 10], [2]

    unsolved = mark_flow_solved_edges(G, inlets, outlets)

    assert unsolved == 2
    marks = {(u, v, k): data[FLOW_SOLVED] for u, v, k, data in G.edges(keys=True, data=True)}
    assert marks == {
        (0, 1, 0): True, (1, 2, 0): True, (1, 2, 1): True,
        (10, 11, 0): False, (20, 21, 0): False,
    }
    pruned, _stats = remove_components_without_connected_io(G, inlets, outlets)
    kept = {(u, v, k) for u, v, k in pruned.edges(keys=True)}
    assert kept == {edge for edge, solved in marks.items() if solved}


def test_mark_flow_solved_edges_rewrites_an_earlier_mark():
    G = _three_pieces()
    mark_flow_solved_edges(G, [0], [2])
    # An outlet added at 11 makes 10 -- 11 solvable.
    assert mark_flow_solved_edges(G, [0, 10], [2, 11]) == 1
    assert G.edges[10, 11, 0][FLOW_SOLVED] is True
    assert G.edges[20, 21, 0][FLOW_SOLVED] is False


def test_extras_schema_defaults():
    schema = default_schema()
    assert schema["small_vessel_boundary_fallback_to_hop_distance"].default is True
    assert schema["small_vessel_boundary_fallback_hop_distance"].default == 1
    # Removing disconnected components is now a boundary_handling choice
    # (see tests/test_network_handling.py), off by default as the toggle was.
    assert "remove_disconnected_io_components_after_final_assignment" not in schema
    assert schema["boundary_handling"].default == "leave_unsolved"
    assert schema["write_fast_mode_preassignment_large_vessel_debug_3d_html"].default is False
    assert schema["large_vessel_3d_volume_downsample_stride"].default == 1
