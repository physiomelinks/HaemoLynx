"""``graph.lumen_loops``: loops lying inside one lumen are Lee-thinning rings.

Lee thinning keeps a mask's topology, so a tunnel through a vessel comes back
as a ring of centreline round it. A vessel loop runs round tissue; a ring runs
round nothing, or a dropout, and is broken -- the vessel drawn once, with no
spur left hanging.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph.lumen_loops import loop_inside_one_lumen, remove_loops_inside_one_lumen
from haemolynx.preprocessing import MaskSupport


def _support(mask):
    return MaskSupport(mask, (1.0, 1.0, 1.0))


def _square_loop(z, lo, hi):
    """A closed square in the plane *z* with corners (lo, lo) and (hi, hi) in (y, x)."""
    corners = [(lo, lo), (lo, hi), (hi, hi), (hi, lo), (lo, lo)]
    return np.array([[z, y, x] for y, x in corners], dtype=float)


def _slab():
    mask = np.zeros((9, 30, 30), dtype=bool)
    mask[2:7, 2:28, 2:28] = True
    return mask


def test_a_loop_round_tissue_is_told_from_a_loop_inside_one_lumen():
    loop = _square_loop(4.0, 6.0, 22.0)
    round_tissue = _slab()
    round_tissue[:, 9:20, 9:20] = False  # 11 um of tissue in the middle
    tunnel = _slab()
    tunnel[:, 14, 14] = False  # a one-voxel dropout through the slab

    assert loop_inside_one_lumen(loop, _support(_slab()))
    assert not loop_inside_one_lumen(loop, _support(round_tissue))
    assert loop_inside_one_lumen(loop, _support(tunnel))


# --- breaking them in a graph -----------------------------------------------


def _tube_mask():
    """A vessel along x at (z, y) = (7, 7), 3 um in radius."""
    mask = np.zeros((15, 20, 60), dtype=bool)
    idx = np.indices(mask.shape)
    mask |= (idx[0] - 7) ** 2 + (idx[1] - 7) ** 2 <= 9
    return mask


def _detour(y: int, x0: float, x1: float) -> list[list[float]]:
    """A centreline from (7, 7, x0) to (7, 7, x1) that steps out to row *y*."""
    off = abs(7 - y)
    step = 1 if y > 7 else -1
    out = [[7.0, 7.0 + step * i, x0 + i] for i in range(off + 1)]
    along = [[7.0, float(y), float(x)] for x in range(int(x0) + off + 1, int(x1) - off)]
    back = [[7.0, 7.0 + step * i, x1 - i] for i in range(off, -1, -1)]
    return out + along + back


def _length(path) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))


def _vessel_with(*rows: int) -> nx.MultiGraph:
    """A vessel A -- u -- v -- B along x, u to v once per *rows* entry: the
    straight centreline for 7, a strand stepping out to that row otherwise."""
    G = nx.MultiGraph()
    for node, x in (("A", 0.0), ("u", 20.0), ("v", 40.0), ("B", 59.0)):
        G.add_node(node, pos=np.array([7.0, 7.0, x]))
    for a, b, x0, x1 in (("A", "u", 0, 20), ("v", "B", 40, 59)):
        path = [[7.0, 7.0, float(x)] for x in range(x0, x1 + 1)]
        G.add_edge(a, b, voxels=path, length=_length(path))
    for row in rows:
        path = _detour(row, 20.0, 40.0)
        G.add_edge("u", "v", voxels=path, length=_length(path), row=row)
    return G


def _rows(G):
    return sorted(d["row"] for *_, d in G.edges(data=True) if "row" in d)


def test_a_ring_inside_one_lumen_loses_its_side_along_the_wall():
    G = _vessel_with(4, 9)  # row 4 runs on the wall, row 9 a voxel inside it

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert _rows(G) == [9]
    assert sorted(G.nodes) == ["A", "B", "u", "v"]
    assert nx.is_connected(G) and nx.cycle_basis(nx.Graph(G)) == []


def test_the_side_kept_is_the_one_down_the_middle_even_when_longer():
    """Regression: the longest side went, and on E14.5 that was often the
    vessel's own centreline, leaving a strand along the rim -- 80k voxels of
    mask uncovered, and a mask diameter read off the rim too small."""
    G = _vessel_with(10)
    middle = [[7.0 + (x % 2), 7.0, float(x)] for x in range(20, 41)]
    G.add_edge("u", "v", voxels=middle, length=_length(middle), row=7)
    assert _length(middle) > _length(_detour(10, 20.0, 40.0))

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert _rows(G) == [7]


def test_a_loop_round_tissue_stays_however_short():
    mask = _tube_mask()
    mask[:, 6:9, 22:39] = False  # 3 um of tissue between the two strands

    G = remove_loops_inside_one_lumen(_vessel_with(4, 9), _support(mask))

    assert _rows(G) == [4, 9]


def test_every_loop_of_a_tangle_in_one_lumen_is_broken():
    """Three strands between the same two junctions: two loops, both rings."""
    G = _vessel_with(4, 7, 9)

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert _rows(G) == [7]
    assert nx.cycle_basis(nx.Graph(G)) == [] and G.number_of_edges() == 3


def test_a_ring_the_network_meets_once_goes_whole():
    """A ring hanging off one node leaves nothing to keep: no spur stays."""
    G = _vessel_with()
    G.add_edge("u", "v", voxels=[[7.0, 7.0, float(x)] for x in range(20, 41)], length=20.0)
    G.add_node("w", pos=np.array([7.0, 7.0, 50.0]))
    for row in (5, 9):
        path = _detour(row, 40.0, 50.0)
        G.add_edge("v", "w", voxels=path, length=_length(path))

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert not G.has_node("w")
    assert sorted(G.nodes) == ["A", "B", "u", "v"]
    assert all(G.degree[n] >= 1 for n in G.nodes)


def test_a_ring_breaks_where_it_leaves_no_dead_end():
    """The ring's longest arc goes, never one edge of it: a branch meeting
    the ring at w keeps a way through, and no new dead end appears."""
    G = _vessel_with(9)
    strand = _detour(4, 20.0, 40.0)
    left, right = strand[:12], strand[11:]
    G.add_node("w", pos=np.array(strand[11]))
    G.add_node("C", pos=np.array(strand[11]) - np.array([0.0, 3.0, 0.0]))
    G.add_edge("u", "w", voxels=left, length=_length(left))
    G.add_edge("w", "v", voxels=right, length=_length(right))
    G.add_edge("w", "C", voxels=[strand[11], list(G.nodes["C"]["pos"])], length=3.0)
    dead_ends = sorted(n for n in G.nodes if G.degree[n] == 1)

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert nx.cycle_basis(nx.Graph(G)) == []
    assert nx.is_connected(G)
    assert sorted(n for n in G.nodes if G.degree[n] == 1) == dead_ends


def test_a_graph_without_loops_is_left_alone():
    G = _vessel_with(9)
    before = sorted(G.edges(keys=True))

    remove_loops_inside_one_lumen(G, _support(_tube_mask()))

    assert sorted(G.edges(keys=True)) == before
