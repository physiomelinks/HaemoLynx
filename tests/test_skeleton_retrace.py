"""The skeleton re-trace: a stretch of the run's skeleton the graph lost is drawn back
when the loop it closes runs round a hole wider than the short-loop rule's limit
(Harvey's grid N, PLAN_surface_coverage.md), and not when it is beside a kept
vessel, off the mask, a branch, or closes no loop.

Each case is a square ring of tube, its skeleton the ring's centreline, and a graph
that follows three of its four sides: the fourth is the lost stretch."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage

from haemolynx.graph import skeleton_retrace as R
from haemolynx.graph.lumen_loops import DEFAULT_SHORT_LOOP_KEEP_HOLE_UM
from haemolynx.preprocessing.bridge_mask_support import MaskSupport

TUBE_RADIUS_UM = 2.0


def _voxels_along(points_um, spacing):
    """Skeleton voxels on the polyline *points_um*, in order, without repeats."""
    out = []
    for a, b in zip(points_um[:-1], points_um[1:]):
        n = int(np.ceil(np.linalg.norm((b - a) / spacing) * 2)) + 1
        for t in np.linspace(0.0, 1.0, n):
            v = tuple(np.rint((a + t * (b - a)) / spacing).astype(int))
            if not out or out[-1] != v:
                out.append(v)
    return out


class Scene:
    """A volume, its skeleton and mask, and a graph over some of the skeleton."""

    def __init__(self, shape, spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.asarray(spacing, dtype=float)
        self.skeleton = np.zeros(shape, dtype=bool)
        self.G = nx.MultiGraph()

    def line(self, *corners_um, in_graph=True, node_names=None):
        corners = [np.asarray(c, dtype=float) for c in corners_um]
        voxels = _voxels_along(corners, self.spacing)
        for v in voxels:
            self.skeleton[v] = True
        if in_graph:
            a, b = node_names
            for name, at in ((a, corners[0]), (b, corners[-1])):
                self.G.add_node(name, pos=tuple(at))
            path = [tuple(np.asarray(v) * self.spacing) for v in voxels]
            path[0], path[-1] = tuple(corners[0]), tuple(corners[-1])
            length = float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())
            self.G.add_edge(a, b, voxels=path, length=length)
        return voxels

    def mask(self, radius_um=TUBE_RADIUS_UM):
        return ndimage.distance_transform_edt(~self.skeleton, sampling=self.spacing) <= radius_um

    def support(self, mask=None):
        return MaskSupport(self.mask() if mask is None else mask, tuple(self.spacing))


def _ring(side_um=40.0, spacing=(1.0, 1.0, 1.0), plane="yx"):
    """A square ring of side *side_um*, the graph following three sides
    (A-B, B-C, C-D); the fourth side, D-A, is skeleton only."""
    spacing = np.asarray(spacing, dtype=float)
    lo, hi = 16.0, 16.0 + side_um  # on every axis' voxel grid, so the corners are skeleton voxels
    mid = 6.0 * spacing.max()
    if plane == "yx":
        corner = lambda p, q: (mid, p, q)
        shape = (int(2 * mid / spacing[0]) + 1, int((hi + 15) / spacing[1]), int((hi + 15) / spacing[2]))
    else:  # the zx plane: the ring crosses the z axis
        corner = lambda p, q: (p, mid, q)
        shape = (int((hi + 15) / spacing[0]), int(2 * mid / spacing[1]) + 1, int((hi + 15) / spacing[2]))
    A, B, C, D = corner(lo, lo), corner(lo, hi), corner(hi, hi), corner(hi, lo)
    scene = Scene(shape, spacing)
    scene.line(A, B, node_names=("A", "B"))
    scene.line(B, C, node_names=("B", "C"))
    scene.line(C, D, node_names=("C", "D"))
    scene.line(D, A, in_graph=False)
    return scene


def _edges(G):
    return sorted((tuple(sorted((str(u), str(v)))), round(float(d["length"]), 6)) for u, v, d in G.edges(data=True))


@pytest.mark.parametrize("spacing, plane", [((1.0, 1.0, 1.0), "yx"), ((2.0, 1.0, 1.0), "zx")])
def test_a_lost_side_of_a_ring_round_a_wide_hole_is_drawn_back_between_its_two_ends(spacing, plane):
    scene = _ring(spacing=spacing, plane=plane)
    G = scene.G

    R.retrace_lost_skeleton(G, scene.skeleton, scene.support())

    (edge,) = R.retraced_edges(G)
    data = G.edges[edge]
    assert {edge[0], edge[1]} == {"A", "D"}  # its two ends meet the graph at the ring's corners
    assert data[R.RETRACED] is True
    assert data["length"] == pytest.approx(40.0, abs=2.0)
    assert data[R.RETRACED_HOLE_UM] > 20.0  # a 40 um ring of 2 um tube runs round ~35 um of tissue
    path = np.asarray(data["voxels"])
    ends = sorted(tuple(np.round(p, 6)) for p in (path[0], path[-1]))
    assert ends == sorted(tuple(np.round(G.nodes[n]["pos"], 6)) for n in edge[:2])
    assert nx.cycle_basis(nx.Graph(G))  # the ring is closed again


def test_the_hole_limit_decides_and_the_judge_says_why():
    scene = _ring()
    before = _edges(scene.G)

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support(), keep_hole_over_um=1000.0)

    assert _edges(scene.G) == before
    (verdict,) = [v for v in R.judge_lost_skeleton(scene.G, scene.skeleton, scene.support(), keep_hole_over_um=1000.0)
                  if R.piece_length_um(v.piece, scene.spacing) >= R.MIN_PIECE_UM]
    assert verdict.reason == R.REFUSED_SMALL_HOLE
    assert 20.0 < verdict.hole_um < 1000.0
    assert verdict.loop_length_um == pytest.approx(160.0, abs=6.0)


def test_the_default_limit_is_the_short_loop_rules():
    assert R.retrace_lost_skeleton.__kwdefaults__["keep_hole_over_um"] == DEFAULT_SHORT_LOOP_KEEP_HOLE_UM


def test_a_lost_stretch_off_the_mask_is_left_out():
    scene = _ring()
    mask = scene.mask()
    mask[:, 21:51, :19] = False  # take the D-A side's tube out of the segmentation, bar its corners
    before = _edges(scene.G)

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support(mask))

    assert _edges(scene.G) == before
    reasons = {v.reason for v in R.judge_lost_skeleton(scene.G, scene.skeleton, scene.support(mask))
               if R.piece_length_um(v.piece, scene.spacing) >= R.MIN_PIECE_UM}
    assert reasons == {R.REFUSED_OFF_MASK}


def test_a_second_strand_beside_a_kept_vessel_in_its_lumen_is_left_out():
    """Two skeleton strands 4 um apart down one fat tube: the graph keeps one,
    and the other is not a separate tube from it."""
    scene = Scene((13, 30, 70))
    scene.line((6.0, 13.0, 10.0), (6.0, 13.0, 60.0), node_names=("a", "b"))
    scene.line((6.0, 17.0, 14.0), (6.0, 17.0, 56.0), in_graph=False)
    fat = ndimage.distance_transform_edt(~scene.skeleton) <= 5.0
    before = _edges(scene.G)

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support(fat))

    assert _edges(scene.G) == before
    (verdict,) = [v for v in R.judge_lost_skeleton(scene.G, scene.skeleton, scene.support(fat))
                  if R.piece_length_um(v.piece, scene.spacing) >= R.MIN_PIECE_UM]
    assert verdict.reason == R.REFUSED_BESIDE


def test_a_lost_branch_meeting_the_graph_once_is_left_out():
    scene = Scene((13, 60, 70))
    scene.line((6.0, 15.0, 10.0), (6.0, 15.0, 60.0), node_names=("a", "b"))
    scene.line((6.0, 15.0, 35.0), (6.0, 50.0, 35.0), in_graph=False)
    before = _edges(scene.G)

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support())

    assert _edges(scene.G) == before
    (verdict,) = [v for v in R.judge_lost_skeleton(scene.G, scene.skeleton, scene.support())
                  if R.piece_length_um(v.piece, scene.spacing) >= R.MIN_PIECE_UM]
    assert verdict.reason == R.REFUSED_ONE_END


def test_a_stretch_joining_two_separate_networks_closes_no_loop_and_is_left_out():
    scene = Scene((13, 60, 70))
    scene.line((6.0, 10.0, 10.0), (6.0, 10.0, 60.0), node_names=("a", "b"))
    scene.line((6.0, 50.0, 10.0), (6.0, 50.0, 60.0), node_names=("c", "d"))
    scene.line((6.0, 10.0, 35.0), (6.0, 50.0, 35.0), in_graph=False)
    before = _edges(scene.G)

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support())

    assert _edges(scene.G) == before
    (verdict,) = [v for v in R.judge_lost_skeleton(scene.G, scene.skeleton, scene.support())
                  if R.piece_length_um(v.piece, scene.spacing) >= R.MIN_PIECE_UM]
    assert verdict.reason == R.REFUSED_NO_LOOP


def test_a_graph_following_its_whole_skeleton_is_left_exactly_as_it_was():
    scene = _ring()
    D, A = scene.G.nodes["D"]["pos"], scene.G.nodes["A"]["pos"]
    scene.line(D, A, node_names=("D", "A"))
    before = [(u, v, k, dict(d)) for u, v, k, d in scene.G.edges(keys=True, data=True)]

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support())

    assert [(u, v, k, dict(d)) for u, v, k, d in scene.G.edges(keys=True, data=True)] == before
    assert R.lost_skeleton_pieces(scene.G, scene.skeleton, scene.spacing) == []


def test_a_lost_stretch_joins_mid_vessel_by_splitting_it():
    """The ring's lost side ends part-way along a kept vessel, not at a node:
    the vessel is split there, both halves kept and measured."""
    scene = Scene((13, 70, 90))
    scene.line((6.0, 15.0, 10.0), (6.0, 15.0, 80.0), node_names=("a", "b"))
    scene.line((6.0, 55.0, 10.0), (6.0, 55.0, 80.0), node_names=("c", "d"))
    scene.line((6.0, 15.0, 10.0), (6.0, 55.0, 10.0), node_names=("a", "c"))
    scene.line((6.0, 15.0, 50.0), (6.0, 55.0, 50.0), in_graph=False)
    total = sum(d["length"] for *_e, d in scene.G.edges(data=True))

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support())

    (edge,) = R.retraced_edges(scene.G)
    ends = [scene.G.nodes[n]["pos"] for n in edge[:2]]
    assert sorted(round(p[1]) for p in ends) == [15, 55] and all(round(p[2]) == 50 for p in ends)
    assert all(scene.G.degree(n) == 3 for n in edge[:2])
    kept = sum(d["length"] for u, v, d in scene.G.edges(data=True) if not d.get(R.RETRACED))
    assert kept == pytest.approx(total, abs=1e-6)


def _ring_with_a_branch():
    """The ring, a kept vessel E-F across its middle from the B-C side, and a
    lost branch from the lost side's middle to E: the piece meets the graph at
    A, D and E, and its route runs A-D."""
    scene = _ring()
    G = scene.G
    # split B-C at F so the kept vessel E-F joins it there
    from haemolynx.graph.edit import insert_node_on_edge

    (key,) = list(G["B"]["C"])
    F = insert_node_on_edge(G, "B", "C", key, (6.0, 36.0, 56.0), node_id="F")
    scene.line((6.0, 36.0, 31.0), G.nodes[F]["pos"], node_names=("E", F))
    scene.line((6.0, 36.0, 16.0), (6.0, 36.0, 31.0), in_graph=False)
    return scene


def _loop_hole(scene, corners):
    from haemolynx.graph.lumen_loops import loop_hole_um

    loop = np.vstack([np.linspace(np.asarray(a, float), np.asarray(b, float), 41)
                      for a, b in zip(corners, corners[1:] + corners[:1])])
    return loop_hole_um(loop, scene.support())


def test_a_drawn_piece_is_joined_wherever_else_it_meets_the_graph_round_a_wide_hole():
    """Grid N2's drawings: every join Harvey added ran along the piece to
    another place it met the graph. The branch to E closes a loop round the
    ring's lower half, a hole far wider than 10 um."""
    scene = _ring_with_a_branch()

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support())

    joins = [e for e in R.retraced_edges(scene.G) if scene.G.edges[e].get(R.RETRACED_JOIN)]
    (join,) = joins
    on_route, at_e = sorted((np.asarray(scene.G.nodes[n]["pos"]) for n in join[:2]), key=lambda p: p[2])
    assert on_route == pytest.approx((6.0, 36.0, 16.0), abs=1.5)
    assert at_e == pytest.approx((6.0, 36.0, 31.0), abs=1.5)
    assert scene.G.edges[join]["length"] == pytest.approx(15.0, abs=2.5)
    route_end = min(join[:2], key=lambda n: scene.G.nodes[n]["pos"][2])
    assert scene.G.degree(route_end) == 3  # the route was split where the join leaves it
    assert not any(R._PIECE in d for *_e, d in scene.G.edges(data=True))


def test_a_further_join_round_a_hole_under_the_limit_is_left_out():
    scene = _ring_with_a_branch()
    route_hole = _loop_hole(scene, [(6.0, 16.0, 16.0), (6.0, 16.0, 56.0), (6.0, 56.0, 56.0), (6.0, 56.0, 16.0)])
    join_hole = _loop_hole(scene, [(6.0, 36.0, 16.0), (6.0, 36.0, 56.0), (6.0, 56.0, 56.0), (6.0, 56.0, 16.0)])
    assert 10.0 < join_hole < route_hole

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support(), keep_hole_over_um=(join_hole + route_hole) / 2)

    drawn = R.retraced_edges(scene.G)
    assert drawn and not any(scene.G.edges[e].get(R.RETRACED_JOIN) for e in drawn)


def test_a_lost_stretch_through_dropouts_in_the_mask_is_drawn_back():
    """The skeleton is thinned from the mask hole-filled and closed, so it runs on through
    dropouts of the raw segmentation: Harvey's missing vessels on E14.5 were 58-100% in
    the mask, and four at 69-78% were refused at 80%. A third of this stretch is dropout."""
    scene = _ring()
    mask = scene.mask()
    for y0 in (22, 30, 38, 46):
        mask[:, y0:y0 + 3, :19] = False
    piece = max(R.lost_skeleton_pieces(scene.G, scene.skeleton, scene.spacing), key=lambda p: len(p.voxels))
    share = float(np.mean(scene.support(mask).inside(piece.voxels * scene.spacing)))
    assert R.MIN_MASK_FRACTION < share < 0.8

    R.retrace_lost_skeleton(scene.G, scene.skeleton, scene.support(mask))

    (edge,) = R.retraced_edges(scene.G)
    assert {edge[0], edge[1]} == {"A", "D"}
