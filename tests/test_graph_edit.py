"""Interactive graph edits: split, delete+collapse, and mask-preferred A*.

These are the pure primitives behind the "Edit" window's Add branch and
Delete edge features (see haemolynx.gui.graph_editor, not written yet at the
time these tests were added) -- no Qt, no napari, just graph/array
operations, so the editing rules are pinned independently of the UI.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import (
    EdgeDraft,
    astar_path,
    commit_new_edge,
    delete_edge_and_collapse,
    insert_node_on_edge,
    mask_cost_field,
    voxel_path_to_microns,
)


# --- insert_node_on_edge -----------------------------------------------------


def _chain_graph() -> nx.MultiGraph:
    """0 --(voxels 0,1,2,3)-- 1, a single straight edge with 4 vertices."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(3.0, 0.0, 0.0))
    G.add_edge(
        0,
        1,
        key=0,
        voxels=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
        length=3.0,
        branch_order="B01",
    )
    return G


def test_insert_node_on_edge_splits_at_the_closest_existing_vertex():
    G = _chain_graph()
    new_node = insert_node_on_edge(G, 0, 1, 0, point_um=(2.1, 0.0, 0.0))

    assert new_node not in (0, 1)
    assert G.nodes[new_node]["pos"] == (2.0, 0.0, 0.0)
    assert not G.has_edge(0, 1)
    assert G.has_edge(0, new_node)
    assert G.has_edge(new_node, 1)
    # branch_order (and any other non-geometry attribute) carries over.
    assert G[0][new_node][0]["branch_order"] == "B01"
    assert G[new_node][1][0]["voxels"] == [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    assert G[0][new_node][0]["voxels"] == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    assert G[0][new_node][0]["length"] == pytest.approx(2.0)
    assert G[new_node][1][0]["length"] == pytest.approx(1.0)


def test_insert_node_on_edge_near_an_end_reuses_the_existing_node():
    G = _chain_graph()
    assert insert_node_on_edge(G, 0, 1, 0, point_um=(0.1, 0.0, 0.0)) == 0
    assert insert_node_on_edge(G, 0, 1, 0, point_um=(2.9, 0.0, 0.0)) == 1
    # Neither call should have mutated the untouched edge.
    assert G.has_edge(0, 1)
    assert G.number_of_nodes() == 2


def test_insert_node_on_edge_falls_back_to_a_straight_line_with_no_voxels():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(10.0, 0.0, 0.0))
    G.add_edge(0, 1, key=0, length=10.0)

    # (6, 0, 0) is closer to node 1's end (10, 0, 0) than node 0's (0, 0, 0):
    # a 2-point line only ever has its two endpoints to snap to.
    new_node = insert_node_on_edge(G, 0, 1, 0, point_um=(6.0, 0.0, 0.0))
    assert new_node == 1
    assert G.has_edge(0, 1)


# --- delete_edge_and_collapse -------------------------------------------------


def _star_graph() -> nx.MultiGraph:
    """A (deg 1) -- C (deg 3) -- B (deg 1), plus C -- D (deg 1)."""
    G = nx.MultiGraph()
    for node, x in (("A", 0.0), ("C", 1.0), ("B", 2.0), ("D", 1.0)):
        y = 0.0 if node != "D" else 1.0
        G.add_node(node, pos=(x, y, 0.0))
    G.add_edge("A", "C", key=0, voxels=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], length=1.0)
    G.add_edge("C", "B", key=0, voxels=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], length=1.0)
    G.add_edge("C", "D", key=0, voxels=[(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], length=1.0)
    return G


def test_delete_edge_collapses_the_endpoint_that_drops_to_degree_two():
    G = _star_graph()
    changed = delete_edge_and_collapse(G, "C", "D", 0)

    # D had no other edge: gone. C dropped from degree 3 to degree 2 (A, B
    # only): collapsed into one continuous A-B edge, C gone too.
    assert "D" not in G
    assert "C" not in G
    assert G.has_edge("A", "B")
    assert G["A"]["B"][0]["voxels"] == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    assert G["A"]["B"][0]["length"] == pytest.approx(2.0)
    assert changed == {"C", "D", "A", "B"}


def test_delete_edge_leaves_an_unrelated_degree_two_node_untouched():
    """Regression: the whole-graph degree2.py sweeps would also merge this;

    a scoped single-edge delete must not.
    """
    G = _star_graph()
    # An unrelated degree-2 chain, X - Y - Z, elsewhere in the same graph.
    G.add_node("X", pos=(5.0, 0.0, 0.0))
    G.add_node("Y", pos=(6.0, 0.0, 0.0))
    G.add_node("Z", pos=(7.0, 0.0, 0.0))
    G.add_edge("X", "Y", key=0, voxels=[(5.0, 0.0, 0.0), (6.0, 0.0, 0.0)], length=1.0)
    G.add_edge("Y", "Z", key=0, voxels=[(6.0, 0.0, 0.0), (7.0, 0.0, 0.0)], length=1.0)

    delete_edge_and_collapse(G, "C", "D", 0)

    assert "Y" in G
    assert G.degree("Y") == 2
    assert G.has_edge("X", "Y")
    assert G.has_edge("Y", "Z")


def test_delete_edge_leaves_a_degree_one_terminal_alone():
    G = _chain_graph()
    changed = delete_edge_and_collapse(G, 0, 1, 0)
    # Both endpoints drop straight to degree 0 (their only edge was this one).
    assert 0 not in G
    assert 1 not in G
    assert changed == {0, 1}


def test_delete_edge_does_not_collapse_two_parallel_edges_into_a_self_loop():
    """A degree-2 node whose two remaining edges both go to the same

    neighbour is a loop, not a pass-through -- must be left alone.
    """
    G = nx.MultiGraph()
    G.add_node("A", pos=(0.0, 0.0, 0.0))
    G.add_node("mid", pos=(1.0, 0.0, 0.0))
    G.add_node("B", pos=(2.0, 0.0, 0.0))
    G.add_edge("A", "mid", key=0, voxels=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], length=1.0)
    G.add_edge("A", "mid", key=1, voxels=[(0.0, 0.0, 0.0), (0.5, 1.0, 0.0), (1.0, 0.0, 0.0)], length=2.0)
    G.add_edge("mid", "B", key=0, voxels=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], length=1.0)

    delete_edge_and_collapse(G, "mid", "B", 0)

    assert "mid" in G
    assert G.degree("mid") == 2
    assert G.number_of_edges("A", "mid") == 2


def test_delete_edge_rejects_an_edge_that_does_not_exist():
    G = _chain_graph()
    with pytest.raises(ValueError, match="No edge"):
        delete_edge_and_collapse(G, 0, 1, key=99)


# --- mask-preferred A* --------------------------------------------------------


def test_astar_prefers_the_segmented_mask_over_a_shorter_background_shortcut():
    """An L-shaped mask corridor vs. the much shorter straight-line diagonal

    that leaves it: the quadratic off-mask penalty must make the corridor
    route cheaper overall, even though it is the longer path in raw voxels.
    """
    mask = np.zeros((5, 20, 20), dtype=bool)
    mask[2, 0:15, 0] = True  # vertical arm
    mask[2, 14, 0:15] = True  # horizontal arm, meeting the vertical arm's end

    cost = mask_cost_field(mask)
    start = (2, 0, 0)
    end = (2, 14, 14)
    path = astar_path(cost, start, end)

    on_mask = mask[tuple(path.astype(int).T)]
    assert on_mask.mean() > 0.9, "path should hug the segmented corridor, not cut the corner"


def test_astar_can_bridge_a_gap_in_the_mask():
    """A short break in an otherwise straight corridor must not block the

    path -- mask-preferred, not mask-restricted.
    """
    mask = np.zeros((5, 3, 20), dtype=bool)
    mask[2, 1, 0:8] = True
    mask[2, 1, 11:20] = True  # voxels 8, 9, 10 are a gap

    cost = mask_cost_field(mask)
    start = (2, 1, 0)
    end = (2, 1, 19)
    path = astar_path(cost, start, end)

    assert tuple(path[0].astype(int)) == start
    assert tuple(path[-1].astype(int)) == end
    # A straight corridor's cheapest path is a straight line; anything much
    # longer would mean the gap forced a detour instead of a direct crossing.
    assert len(path) <= 25


def test_voxel_path_to_microns_scales_per_axis():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    points = voxel_path_to_microns(path, voxel_size_zyx=(2.0, 0.5, 4.0))
    assert points == [(0.0, 0.0, 0.0), (2.0, 1.0, 12.0)]


# --- EdgeDraft / commit_new_edge ----------------------------------------------


def test_edge_draft_extend_does_not_duplicate_the_shared_vertex():
    draft = EdgeDraft(start_node=0, points_um=[(0.0, 0.0, 0.0)])
    draft.extend([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    draft.extend([(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)])

    assert draft.points_um == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    assert draft.last_point == (3.0, 0.0, 0.0)


def test_commit_new_edge_to_an_existing_node():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(2.0, 0.0, 0.0))
    draft = EdgeDraft(start_node=0, points_um=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])

    start, end, key = commit_new_edge(G, draft, end_node=1)

    assert (start, end) == (0, 1)
    assert G.has_edge(0, 1, key)
    assert G[0][1][key]["length"] == pytest.approx(2.0)


def test_commit_new_edge_creates_a_dangling_terminal_when_no_target_was_hit():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    draft = EdgeDraft(start_node=0, points_um=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])

    start, end, key = commit_new_edge(G, draft, end_node=None)

    assert start == 0
    assert end not in (0,)
    assert G.nodes[end]["pos"] == (5.0, 0.0, 0.0)
    assert G.has_edge(0, end, key)
