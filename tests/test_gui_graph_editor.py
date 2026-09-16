"""The "Edit" window's Add branch / Delete edge state machine, without a viewer.

`GraphEditorState` is pure -- a fake click sequence (raw point + an
`EdgeHit`/`NodeHit`/`None`, exactly what `gui.graph_click` would resolve a
real click to) drives it the same way the floating window's mouse callbacks
would, so these tests stand in for the interaction this sandbox's crashing
`make_napari_viewer` cannot exercise directly (see project memory).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.graph_click import EdgeHit, NodeHit
from haemolynx.gui.graph_editor import GraphEditorState


def _line_graph() -> nx.MultiGraph:
    """0 --(0,1)-- 1 --(1,2)-- 2, positions along x, with a real midpoint

    vertex on each edge so a mid-edge click has somewhere genuine to split
    (a plain 2-point straight edge only ever splits at its own ends -- see
    test_graph_edit.py's own test for that fallback case).
    """
    G = nx.MultiGraph()
    for node, x in enumerate((0.0, 5.0, 10.0)):
        G.add_node(node, pos=(x, 0.0, 0.0))
    G.add_edge(
        0, 1, key=0,
        voxels=[(0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (5.0, 0.0, 0.0)], length=5.0,
    )
    G.add_edge(
        1, 2, key=0,
        voxels=[(5.0, 0.0, 0.0), (7.5, 0.0, 0.0), (10.0, 0.0, 0.0)], length=5.0,
    )
    return G


def _flat_cost_field(shape=(20, 20, 20)) -> np.ndarray:
    """A uniform-cost field (no preferred structure) for a plain A* line-search."""
    return np.ones(shape, dtype=float)


def _state(**overrides) -> GraphEditorState:
    kwargs = dict(
        graph=_line_graph(),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        cost_field=_flat_cost_field(),
    )
    kwargs.update(overrides)
    return GraphEditorState(**kwargs)


# --- mode transitions ---------------------------------------------------------


def test_modes_start_idle_and_toggle():
    state = _state()
    assert state.mode == "idle"
    state.start_add()
    assert state.mode == "add"
    state.start_delete()
    assert state.mode == "delete"
    assert state.draft is None
    state.stop()
    assert state.mode == "idle"


# --- Add branch ---------------------------------------------------------------


def test_first_click_must_hit_existing_structure():
    state = _state()
    state.start_add()
    result = state.click_add((1.0, 1.0, 1.0), hit=None)
    assert result == "rejected"
    assert state.draft is None


def test_first_click_on_a_node_anchors_the_draft_there():
    state = _state()
    state.start_add()
    result = state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))
    assert result == "started"
    assert state.draft is not None
    assert state.draft.start_node == 0
    assert state.draft.points_um == [(0.0, 0.0, 0.0)]


def test_first_click_on_an_edge_splits_it_and_anchors_there():
    state = _state()
    state.start_add()
    hit = EdgeHit(u=0, v=1, key=0, point_um=(2.5, 0.0, 0.0))
    state.click_add((2.5, 0.0, 0.0), hit=hit)

    # A new node was inserted on (0, 1) at its own midpoint vertex (2.5, 0, 0);
    # the draft anchors there.
    assert state.draft is not None
    new_node = state.draft.start_node
    assert new_node not in (0, 1, 2)
    assert state.graph.nodes[new_node]["pos"] == (2.5, 0.0, 0.0)
    assert not state.graph.has_edge(0, 1)
    assert state.graph.has_edge(0, new_node)
    assert state.graph.has_edge(new_node, 1)


def test_a_missed_second_click_extends_the_draft_without_finishing():
    state = _state()
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))

    result = state.click_add((3.0, 0.0, 0.0), hit=None)

    assert result == "extended"
    assert state.draft is not None
    assert state.draft.last_point == (3.0, 0.0, 0.0)
    # Nothing committed to the graph yet.
    assert state.graph.number_of_edges() == 2


def test_a_click_on_an_existing_node_finishes_the_draft():
    state = _state()
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))
    state.click_add((5.0, 5.0, 0.0), hit=None)  # a waypoint off the main line

    result = state.click_add((10.0, 0.0, 0.0), hit=NodeHit(node_id=2))

    assert result == "finished"
    assert state.draft is None
    assert state.graph.has_edge(0, 2)
    new_key = next(iter(state.graph[0][2]))
    assert state.graph[0][2][new_key]["voxels"][0] == (0.0, 0.0, 0.0)
    assert state.graph[0][2][new_key]["voxels"][-1] == (10.0, 0.0, 0.0)
    assert "branch_order" not in state.graph[0][2][new_key]


def test_a_click_on_an_existing_edge_splits_it_and_finishes_there():
    state = _state()
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))

    hit = EdgeHit(u=1, v=2, key=0, point_um=(7.5, 0.0, 0.0))
    result = state.click_add((7.5, 0.0, 0.0), hit=hit)

    assert result == "finished"
    assert not state.graph.has_edge(1, 2)
    split_node = next(n for n in state.graph.nodes if state.graph.nodes[n]["pos"] == (7.5, 0.0, 0.0))
    assert state.graph.has_edge(0, split_node)
    assert state.graph.has_edge(split_node, 1)
    assert state.graph.has_edge(split_node, 2)


def test_clicking_back_on_the_drafts_own_start_keeps_drafting():
    state = _state()
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))

    result = state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))

    assert result == "extended"
    assert state.draft is not None
    assert state.graph.number_of_edges() == 2  # no self-loop committed


def test_extending_without_a_cost_field_falls_back_to_a_straight_line():
    """Regression: a missing cost field (e.g. no image layer available when
    the editor opened) used to raise straight out of a click -- it must
    still draw something instead of crashing the interaction."""
    state = _state(cost_field=None)
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))
    result = state.click_add((3.0, 0.0, 0.0), hit=None)
    assert result == "extended"
    assert state.draft.points_um[-1] == (3.0, 0.0, 0.0)


def test_finish_add_commits_a_dangling_terminal():
    state = _state()
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), hit=NodeHit(node_id=0))
    state.click_add((3.0, 0.0, 0.0), hit=None)

    new_node = state.finish_add()

    assert new_node is not None
    assert state.draft is None
    assert state.graph.has_edge(0, new_node)
    assert state.graph.nodes[new_node]["pos"] == (3.0, 0.0, 0.0)


def test_finish_add_with_no_draft_is_a_no_op():
    state = _state()
    state.start_add()
    assert state.finish_add() is None
    assert state.graph.number_of_edges() == 2


def test_cost_field_from_mask_builds_a_real_cost_field():
    state = _state(cost_field=None)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    state.cost_field_from_mask(mask)
    assert state.cost_field is not None
    assert state.cost_field.shape == mask.shape
    assert state.cost_field[2, 2, 2] == pytest.approx(1.0)


# --- Delete edge ----------------------------------------------------------


def test_click_delete_with_no_hit_changes_nothing():
    state = _state()
    state.start_delete()
    changed = state.click_delete(None)
    assert changed == set()
    assert state.graph.number_of_edges() == 2


def test_click_delete_removes_the_hit_edge_and_collapses_degree_two():
    state = _state()
    state.start_delete()
    hit = EdgeHit(u=1, v=2, key=0, point_um=(7.5, 0.0, 0.0))

    changed = state.click_delete(hit)

    # Node 1 drops from degree 2 to degree 1 (kept, no action); node 2's
    # only edge was this one, so it drops to degree 0 and is removed. The
    # collapse/removal rules themselves are exercised fully in
    # test_graph_edit.py -- this only checks the click routes through.
    assert not state.graph.has_edge(1, 2)
    assert 2 not in state.graph
    assert changed == {2}
