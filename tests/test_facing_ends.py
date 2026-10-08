"""``graph.facing_ends``: two dead ends pointing at each other across a gap.

A vessel the segmentation broke leaves a dead end either side of the gap,
each heading for the other. They are joined across that one stretch of
background; two ends side by side, a third vessel in the gap, or a gap longer
than the limit are left alone.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph.facing_ends import join_facing_dead_ends
from haemolynx.preprocessing import MaskSupport


def _support(mask):
    return MaskSupport(mask, (1.0, 1.0, 1.0))


def _tube(mask, axis, centre, radius, lo, hi):
    idx = np.indices(mask.shape)
    others = [a for a in range(3) if a != axis]
    r2 = sum((idx[a] - centre[a]) ** 2 for a in others)
    mask |= (r2 <= radius**2) & (idx[axis] >= lo) & (idx[axis] <= hi)


def _straight(G, u, v):
    path = [list(map(float, G.nodes[u]["pos"])), list(map(float, G.nodes[v]["pos"]))]
    G.add_edge(u, v, voxels=path, length=float(np.linalg.norm(np.subtract(path[1], path[0]))))


def _broken_vessel(gap_from=21, gap_to=25):
    """A vessel along x at (z, y) = (7, 7), its mask missing x = gap_from..gap_to,
    drawn as two pieces ending either side of the gap."""
    mask = np.zeros((15, 20, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 2, 0, 59)
    mask[:, :, gap_from:gap_to + 1] = False
    G = nx.MultiGraph()
    for node, x in (("start", 0.0), ("left", gap_from - 1.0), ("right", gap_to + 1.0), ("end", 59.0)):
        G.add_node(node, pos=np.array([7.0, 7.0, x]))
    _straight(G, "start", "left")
    _straight(G, "right", "end")
    return G, mask


def _joins(G):
    return [(u, v, d) for u, v, d in G.edges(data=True) if d.get("bridge_kind") == "facing_ends"]


def test_a_vessel_broken_by_a_gap_in_the_mask_is_joined_across_it():
    G, mask = _broken_vessel()

    join_facing_dead_ends(G, _support(mask))

    (join,) = _joins(G)
    assert {join[0], join[1]} == {"left", "right"}
    assert join[2]["reconnected"] is True
    assert join[2]["length"] == 6.0
    assert 4.0 <= join[2]["bridge_background_um"] <= 6.0
    assert nx.is_connected(G)
    assert sorted(n for n in G.nodes if G.degree(n) == 1) == ["end", "start"]


def test_two_ends_side_by_side_are_not_joined():
    """Close, but heading the same way: two vessels, not one broken one."""
    mask = np.zeros((15, 30, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 2, 0, 30)
    _tube(mask, 2, (7, 13, 0), 2, 0, 30)
    G = nx.MultiGraph()
    for node, pos in (("a0", (7, 7, 0)), ("a", (7, 7, 30)), ("b0", (7, 13, 0)), ("b", (7, 13, 30))):
        G.add_node(node, pos=np.array(pos, dtype=float))
    _straight(G, "a0", "a")
    _straight(G, "b0", "b")

    join_facing_dead_ends(G, _support(mask))

    assert _joins(G) == []


def test_a_join_that_would_jump_a_third_vessel_is_refused():
    """Two stretches of background with mask between: another vessel in the gap."""
    G, mask = _broken_vessel()
    _tube(mask, 1, (7, 0, 23), 1, 0, 19)  # a vessel along y through the gap

    join_facing_dead_ends(G, _support(mask))

    assert _joins(G) == []


def test_a_gap_longer_than_the_limit_is_left_open():
    G, mask = _broken_vessel(gap_from=21, gap_to=33)  # ends 14 um apart

    join_facing_dead_ends(G, _support(mask), max_gap_um=10.0)
    assert _joins(G) == []
    join_facing_dead_ends(G, _support(mask), max_gap_um=0.0)
    assert _joins(G) == []


def test_an_end_too_short_to_have_a_heading_is_not_joined():
    G, mask = _broken_vessel()
    G.remove_edge("right", "end")
    G.add_node("stub_end", pos=np.array([7.0, 7.0, 28.0]))
    _straight(G, "right", "stub_end")  # 2 um of vessel: no direction to continue

    join_facing_dead_ends(G, _support(mask))

    assert _joins(G) == []


def test_each_end_joins_once_and_the_closest_pair_goes_first():
    """A third end behind the right-hand piece also faces the left end."""
    G, mask = _broken_vessel()
    _tube(mask, 2, (7, 12, 0), 2, 28, 50)
    mask[:, 8:12, 28:34] = True  # the third end's vessel meets the gap region from the side
    G.add_node("third", pos=np.array([7.0, 11.0, 29.0]))
    G.add_node("third0", pos=np.array([7.0, 12.0, 50.0]))
    _straight(G, "third", "third0")

    join_facing_dead_ends(G, _support(mask))

    joins = _joins(G)
    assert len(joins) == 1
    assert {joins[0][0], joins[0][1]} == {"left", "right"}
    assert G.degree("third") == 1


def test_a_heading_is_read_along_a_chain_of_short_edges():
    """Regression: before the last merge a vessel is often several short
    edges through degree-2 nodes; read from its last edge alone, every such
    end was too short to have a direction, and none was joined."""
    G, mask = _broken_vessel()
    G.remove_edge("start", "left")
    previous = "start"
    for k, x in enumerate((4.0, 8.0, 12.0, 16.0, 18.0)):
        G.add_node(f"chain{k}", pos=np.array([7.0, 7.0, x]))
        _straight(G, previous, f"chain{k}")
        previous = f"chain{k}"
    _straight(G, previous, "left")  # the last edge: 2 um

    join_facing_dead_ends(G, _support(mask))

    (join,) = _joins(G)
    assert {join[0], join[1]} == {"left", "right"}
