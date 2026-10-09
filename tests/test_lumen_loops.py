"""``graph.lumen_loops``: loops lying inside one lumen are Lee-thinning rings.

Lee thinning keeps a mask's topology, so a tunnel through a vessel comes back
as a ring of centreline round it. A vessel loop runs round tissue; a ring runs
round nothing, or a dropout, and is broken -- the vessel drawn once, with no
spur left hanging.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.lumen_loops import (
    loop_centre_um,
    loop_inside_one_lumen,
    loop_measures,
    loop_sides,
    remove_loops_inside_one_lumen,
)
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


# --- what Manual loop review reads ---------------------------------------------


def _parallel_cycle(G, u, v):
    """The loop the two edges between *u* and *v* close, in order round it."""
    first, second = sorted(G[u][v])
    return [(u, v, first), (v, u, second)]


def test_a_loop_has_one_side_between_each_two_places_the_network_meets_it():
    G = _vessel_with(4, 9)
    cycle = _parallel_cycle(G, "u", "v")

    assert loop_sides(G, cycle) == [[cycle[0]], [cycle[1]]]


def _triangle_off(node_off_it: bool) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 10, 0), 2: (0, 0, 10), 3: (0, -10, -10)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in ((0, 1), (1, 2), (2, 0)):
        G.add_edge(u, v, length=10.0)
    if node_off_it:
        G.add_edge(0, 3, length=14.0)
    return G


def test_a_loop_met_once_is_one_side_and_one_met_nowhere_has_a_side_per_edge():
    cycle = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]

    assert loop_sides(_triangle_off(True), cycle) == [cycle]
    assert loop_sides(_triangle_off(False), cycle) == [[edge] for edge in cycle]


def test_a_boundary_node_on_a_loop_cuts_it_into_sides_too():
    cycle = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]

    assert loop_sides(_triangle_off(True), cycle, cut_at={2}) == [cycle[:2], cycle[2:]]


def _circle(radius: float, centre, e1, e2, count: int = 400) -> np.ndarray:
    e1 = np.asarray(e1, dtype=float) / np.linalg.norm(e1)
    e2 = np.asarray(e2, dtype=float) / np.linalg.norm(e2)
    t = np.linspace(0.0, 2 * np.pi, count + 1)
    return np.asarray(centre, dtype=float) + radius * (
        np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2
    )


def test_loop_measures_read_the_size_and_shape_of_a_circle_in_a_tilted_plane():
    circle = _circle(5.0, (20.0, 30.0, 40.0), (0, 1, 0), (1, 0, 1))

    measures = loop_measures(circle)

    assert measures["loop_length_um"] == pytest.approx(2 * np.pi * 5.0, rel=1e-3)
    assert measures["area_um2"] == pytest.approx(np.pi * 25.0, rel=1e-2)
    assert measures["span_um"] == pytest.approx(10.0, rel=1e-2)
    assert measures["out_of_plane"] == pytest.approx(0.0, abs=1e-6)
    centre = [measures[f"centre_{axis}_um"] for axis in "zyx"]
    assert centre == pytest.approx([20.0, 30.0, 40.0], abs=1e-6)
    assert loop_centre_um(circle) == pytest.approx([20.0, 30.0, 40.0], abs=1e-6)
    for name in ("path_in", "lumen_r_um", "chords_in_lumen", "inside_one_lumen", "wall_side"):
        assert measures[name] is None, "no mask, no mask measures"


def test_a_loop_bent_out_of_its_plane_reads_as_bent_by_as_much():
    """A saddle rising *depth* out of a 6 um circle's plane: its third
    singular value over its second is about depth / 6."""
    t = np.linspace(0.0, 2 * np.pi, 401)
    bends = [
        loop_measures(np.column_stack([depth * np.cos(2 * t), 6.0 * np.cos(t), 6.0 * np.sin(t)]))[
            "out_of_plane"
        ]
        for depth in (0.0, 0.6, 3.0)
    ]

    assert bends[0] == pytest.approx(0.0, abs=1e-6)
    assert bends[0] < bends[1] < bends[2]
    assert bends[1] == pytest.approx(0.1, abs=0.02)
    assert bends[2] == pytest.approx(0.5, abs=0.05)


def test_loop_measures_tell_a_ring_in_one_lumen_from_a_loop_round_tissue():
    loop = _square_loop(4.0, 6.0, 22.0)
    round_tissue = _slab()
    round_tissue[:, 9:20, 9:20] = False

    ring = loop_measures(loop, mask=_slab())
    vessel_loop = loop_measures(loop, mask=round_tissue)

    assert ring["path_in"] == 1.0 and vessel_loop["path_in"] == 1.0
    assert ring["inside_one_lumen"] is True and vessel_loop["inside_one_lumen"] is False
    assert ring["chords_in_lumen"] == 1.0 and ring["chord_bg_median_um"] == 0.0
    assert vessel_loop["chords_in_lumen"] < 0.5 and vessel_loop["chord_bg_median_um"] > 5.0
    # The slab is five voxels deep and the loop runs through its middle.
    assert 1.0 < ring["lumen_r_um"] < 4.0


class _Reads:
    """A mask that remembers the boxes read from it."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.boxes = []

    def __getitem__(self, key):
        self.boxes.append(key)
        return self.array[key]


def test_loop_measures_read_only_a_box_round_the_loop_and_place_it_right():
    loop = _square_loop(4.0, 6.0, 22.0)
    round_tissue = _slab()
    round_tissue[:, 9:20, 9:20] = False
    offset = np.array([30, 60, 70])
    big = np.zeros((60, 120, 140), dtype=bool)
    big[30:39, 60:90, 70:100] = round_tissue
    reads = _Reads(big)

    near = loop_measures(loop, mask=round_tissue)
    far = loop_measures(loop + offset, mask=reads)

    for name in ("path_in", "lumen_r_um", "chords_in_lumen", "chord_bg_median_um", "inside_one_lumen"):
        assert far[name] == pytest.approx(near[name]), name
    (box,) = reads.boxes
    read = np.prod([s.stop - s.start for s in box])
    assert read < big.size / 10


def test_the_wall_side_is_the_side_the_loop_breaker_would_take_out():
    from haemolynx.graph.lumen_loops import iter_short_loops

    G = _vessel_with(4, 9)  # row 4 runs on the wall, row 9 a voxel inside it
    ((cycle, polyline),) = list(iter_short_loops(G))
    sides = loop_sides(G, cycle)
    sides_um = [np.vstack([G.edges[e]["voxels"] for e in side]) for side in sides]
    rows = [G.edges[side[0]]["row"] for side in sides]

    measures = loop_measures(polyline, mask=_tube_mask(), sides_um=sides_um)

    assert rows[measures["wall_side"]] == 4
    radii = measures["side_lumen_r_um"]
    assert radii[rows.index(4)] < radii[rows.index(9)]
    remove_loops_inside_one_lumen(G, _support(_tube_mask()))
    assert _rows(G) == [9], "the breaker took out the same side"
