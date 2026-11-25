"""Small synthetic masks and graphs, one per thing graph building can leave
in a segmented vessel that is not a vessel, each with the right answer known.

Voxels are 1 um on every axis, so voxel indices and physical (z, y, x)
microns are the same numbers. Nodes are named by their position.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np

VOXEL_SIZE = (1.0, 1.0, 1.0)


def capsule(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float], radius: float) -> np.ndarray:
    """Set every voxel within *radius* of the segment p0-p1: a straight tube
    with rounded ends. Returns *mask* for chaining."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    grid = np.stack(np.indices(mask.shape), axis=-1).astype(float)
    d = p1 - p0
    t = np.clip(((grid - p0) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    mask |= np.linalg.norm(grid - (p0 + t[..., None] * d), axis=-1) <= radius
    return mask


def _node(point) -> tuple:
    return tuple(float(round(c, 3)) for c in point)


def polyline_graph(paths: Iterable[Sequence[Sequence[float]]]) -> nx.MultiGraph:
    """One edge per path, sampled every half micron, between nodes at its two
    ends; ends at the same point share a node."""
    G = nx.MultiGraph()
    G.graph["voxel_size"] = VOXEL_SIZE
    for path in paths:
        corners = np.asarray(path, dtype=float)
        dense = [corners[0]]
        for a, b in zip(corners[:-1], corners[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.5)))
            dense.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
        dense = np.asarray(dense)
        u, v = _node(dense[0]), _node(dense[-1])
        for node in (u, v):
            G.add_node(node, pos=np.asarray(node))
        G.add_edge(u, v, voxels=dense.tolist(),
                   length=float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    return G


def wide_vessel() -> np.ndarray:
    """A 12 um vessel along x, centred on (z, y) = (7, 7), 60 um long."""
    return capsule(np.zeros((15, 15, 60), dtype=bool), (7, 7, -10), (7, 7, 70), 6)


def two_strands_in_one_lumen():
    """Lee thinning of a wide lumen: two strands 4 um apart, each its own edge."""
    return wide_vessel(), polyline_graph([[(7, 5, 2), (7, 5, 57)], [(7, 9, 2), (7, 9, 57)]])


def fork_in_one_lumen():
    """Two edges leave one node and run side by side through one lumen."""
    return wide_vessel(), polyline_graph([[(7, 7, 2), (7, 5, 12), (7, 5, 57)],
                                          [(7, 7, 2), (7, 9, 12), (7, 9, 57)]])


def hair_along_a_wall():
    """A dead end leaving a junction and running 20 um along the wide
    vessel's own lumen beside its centreline: a hair, not a branch."""
    return wide_vessel(), polyline_graph([[(7, 7, 2), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)],
                                          [(7, 7, 30), (7, 11, 34), (7, 11, 54)]])


def ring_in_one_lumen():
    """A loop of two edges round nothing: both arcs inside one wide lumen."""
    return wide_vessel(), polyline_graph([[(7, 7, 10), (7, 4, 20), (7, 4, 40), (7, 7, 50)],
                                          [(7, 7, 10), (7, 10, 20), (7, 10, 40), (7, 7, 50)]])


def loop_round_tissue():
    """Two capillaries joined at both ends round a block of tissue: a real
    vessel loop, 12 um of background inside it."""
    mask = np.zeros((9, 30, 60), dtype=bool)
    for a, b in (((4, 6, 8), (4, 6, 52)), ((4, 22, 8), (4, 22, 52)),
                 ((4, 6, 8), (4, 22, 8)), ((4, 6, 52), (4, 22, 52))):
        capsule(mask, a, b, 2.5)
    G = polyline_graph([[(4, 6, 8), (4, 6, 52)], [(4, 22, 8), (4, 22, 52)],
                        [(4, 6, 8), (4, 22, 8)], [(4, 6, 52), (4, 22, 52)]])
    return mask, G


def dead_end_cases():
    """A capillary trunk along x with one dead end of every kind hanging off it,
    each kind on its own junction, plus an isolated single edge.

    Returns ``(mask, G, tips)``: *tips* maps each kind to its dead end's node.
    """
    mask = np.zeros((11, 60, 120), dtype=bool)
    trunk_y = 30
    capsule(mask, (5, trunk_y, 0), (5, trunk_y, 119), 3)
    paths = []
    junctions = [10, 30, 50, 70, 90]
    for x0, x1 in zip([0] + junctions, junctions + [119]):
        paths.append([(5, trunk_y, x0), (5, trunk_y, x1)])
    tips = {}
    # A hair along the trunk's own lumen: its tip never leaves it.
    paths.append([(5, trunk_y, 10), (5, trunk_y + 2, 14), (5, trunk_y + 2, 24)])
    tips["inside_other_lumen"] = _node((5, trunk_y + 2, 24))
    # A branch straight out into background: mostly off the mask.
    paths.append([(5, trunk_y, 30), (5, trunk_y - 20, 30)])
    tips["off_mask"] = _node((5, trunk_y - 20, 30))
    # A branch whose segmented vessel runs on 10 um past the tip.
    capsule(mask, (5, trunk_y, 50), (5, trunk_y + 25, 50), 2)
    paths.append([(5, trunk_y, 50), (5, trunk_y + 15, 50)])
    tips["mask_continues"] = _node((5, trunk_y + 15, 50))
    # A short blind sprout: 5 um on a 3 um trunk, the mask ending with it.
    capsule(mask, (5, trunk_y, 70), (5, trunk_y - 6, 70), 1.5)
    paths.append([(5, trunk_y, 70), (5, trunk_y - 5, 70)])
    tips["short"] = _node((5, trunk_y - 5, 70))
    # A real branch: long, in its own vessel, ending where the mask ends.
    capsule(mask, (5, trunk_y, 90), (5, trunk_y + 21, 90), 2)
    paths.append([(5, trunk_y, 90), (5, trunk_y + 20, 90)])
    tips["real_branch"] = _node((5, trunk_y + 20, 90))
    # An island: one short vessel on its own.
    capsule(mask, (5, 52, 20), (5, 52, 40), 2)
    paths.append([(5, 52, 22), (5, 52, 38)])
    return mask, polyline_graph(paths), tips


def dead_end_beside_vessel():
    """A trunk through a wide lumen, and a dead end leaving it at a junction
    that runs back along inside the same lumen. Returns
    ``(mask, G, tip, junction)``."""
    junction = (7, 7, 10)
    G = polyline_graph([
        [(7, 7, 0), junction], [junction, (7, 7, 59)],
        [junction, (7, 10, 15), (7, 10, 45)],
    ])
    return wide_vessel(), G, _node((7, 10, 45)), _node(junction)


def fork_that_parts():
    """Two edges leave one node side by side through a wide lumen and part
    where it ends, each into a branch of its own. Returns ``(mask, G, ends)``:
    the trunk's open end and the two branch tips."""
    mask = np.zeros((15, 40, 70), dtype=bool)
    capsule(mask, (7, 20, -10), (7, 20, 40), 6)
    capsule(mask, (7, 18, 38), (7, 4, 66), 2.5)
    capsule(mask, (7, 22, 38), (7, 36, 66), 2.5)
    G = polyline_graph([
        [(7, 20, 0), (7, 20, 8)],
        [(7, 20, 8), (7, 18, 14), (7, 18, 38), (7, 5, 64)],
        [(7, 20, 8), (7, 22, 14), (7, 22, 38), (7, 35, 64)],
    ])
    return mask, G, [_node(p) for p in ((7, 20, 0), (7, 5, 64), (7, 35, 64))]


def strand_through_a_trunk():
    """A capillary that enters a wide trunk's lumen from one side branch, runs
    along inside it beside the trunk's centreline, and leaves by another: the
    stretch inside is the trunk drawn twice. Returns ``(mask, G, ends)``: the
    capillary's two tips, then the trunk's two ends."""
    mask = np.zeros((15, 40, 70), dtype=bool)
    capsule(mask, (7, 20, -10), (7, 20, 80), 6)
    capsule(mask, (7, 38, 10), (7, 23, 14), 2)
    capsule(mask, (7, 23, 46), (7, 38, 50), 2)
    G = polyline_graph([
        [(7, 20, 0), (7, 20, 69)],
        [(7, 37, 10), (7, 23, 14), (7, 23, 46), (7, 37, 50)],
    ])
    ends = [_node(p) for p in ((7, 37, 10), (7, 37, 50), (7, 20, 0), (7, 20, 69))]
    return mask, G, ends


# --- Phase 0's remaining cases (PLAN_one_centreline_per_lumen.md) -------------
# Each returns (mask, G, ...) with the plan's right answer in its docstring;
# tests/test_consolidate_lumen.py pins what the clean-up does to each today.


def touching_capillaries():
    """Two capillaries (radius 2.5 um) touching along 20 um with a neck between
    them: their axes 4.5 um apart there, so their tubes overlap 0.5 um and the
    lumen is 2.2 um tall at the neck against 5 um in each tube. Both run from
    the x = 0 face to the x = 59 face, the second bending in to meet the first
    and out again. Right answer: two strands. Returns ``(mask, G, ends)``,
    the straight one's two ends, then the bent one's."""
    mask = np.zeros((9, 30, 60), dtype=bool)
    straight = [(4, 8, 0), (4, 8, 59)]
    bent = [(4, 24, 0), (4, 24, 8), (4, 12.5, 20), (4, 12.5, 40), (4, 24, 52), (4, 24, 59)]
    for path in (straight, bent):
        for a, b in zip(path[:-1], path[1:]):
            capsule(mask, a, b, 2.5)
    G = polyline_graph([straight, bent])
    return mask, G, [_node(p) for p in (straight[0], straight[-1], bent[0], bent[-1])]


def ring_in_a_bending_vessel():
    """A wide vessel (radius 6 um) turning a right angle, with a Lee ring in its
    lumen round the bend: two arcs from a node 35 um before the corner on the
    first leg to one 35 um after it on the second, one each side of the
    centreline, no gap inside the ring. Most of the ring's antipodal chords
    join a point on one leg to a point on the other and cut the bend's inside
    corner, which is background (the plan's loops 18 and 22). Right answer: no
    ring (one strand round the bend). Returns ``(mask, G, ring_ends)``."""
    mask = np.zeros((15, 70, 70), dtype=bool)
    capsule(mask, (7, 55, -10), (7, 55, 50), 6)
    capsule(mask, (7, 55, 50), (7, -10, 50), 6)
    first, second = (7, 55, 15), (7, 20, 50)
    G = polyline_graph([
        [(7, 55, 0), first], [second, (7, 0, 50)],
        [first, (7, 52, 20), (7, 52, 47), (7, 20, 47), second],
        [first, (7, 58, 20), (7, 58, 53), (7, 20, 53), second],
    ])
    return mask, G, [_node(first), _node(second)]


def loop_round_a_small_gap():
    """Two capillaries (radius 2.5 um) parting round a hole 1 um wide and
    3 um long in the segmentation, and meeting again: a loop round a gap
    smaller than a dropout. Right answer, by the plan's decision D1 (marks of
    2026-10-08): the loop stays. Returns ``(mask, G, loop_ends)``."""
    mask = np.zeros((9, 30, 70), dtype=bool)
    left, right = (4, 15, 25), (4, 15, 45)
    capsule(mask, (4, 15, -5), left, 2.5)
    capsule(mask, right, (4, 15, 75), 2.5)
    for y in (13.0, 17.0):
        capsule(mask, left, (4, y, 30), 2.5)
        capsule(mask, (4, y, 30), (4, y, 40), 2.5)
        capsule(mask, (4, y, 40), right, 2.5)
    mask[:, 15, 34:37] = False  # the hole: 1 um across, 3 um along
    G = polyline_graph([
        [(4, 15, 0), left], [right, (4, 15, 69)],
        [left, (4, 13, 30), (4, 13, 40), right],
        [left, (4, 17, 30), (4, 17, 40), right],
    ])
    return mask, G, [_node(left), _node(right)]


def wall_bump():
    """A capillary (radius 2.5 um) along x with a bump on its wall: a ball of
    radius 3 um centred on the wall at x = 30, which Lee thinning draws as a
    spur 3.5 um long from the vessel's centreline into the bump. Right answer:
    no stub. Returns ``(mask, G, tip)``."""
    mask = np.zeros((21, 30, 60), dtype=bool)
    capsule(mask, (10, 12, -5), (10, 12, 65), 2.5)
    capsule(mask, (10, 14.5, 30), (10, 14.5, 30), 3)
    tip = (10, 15.5, 30)
    G = polyline_graph([[(10, 12, 0), (10, 12, 30)], [(10, 12, 30), (10, 12, 59)], [(10, 12, 30), tip]])
    return mask, G, _node(tip)


def branch_at_15_degrees():
    """A branch (radius 2 um) leaving a trunk (radius 3 um) at 15 degrees, as
    cluster collapse leaves it: one node where the two centrelines meet, and
    the branch's first ~10 um inside the trunk's lumen beside the trunk's own
    centreline, so the two edges leaving the node run side by side. Right
    answer: one junction where the branch leaves the lumen, no fork. Returns
    ``(mask, G, junction, branch_tip)``."""
    mask = np.zeros((9, 30, 80), dtype=bool)
    junction = (4, 6, 15)
    capsule(mask, (4, 6, -5), (4, 6, 85), 3)
    direction = np.array([0.0, np.sin(np.radians(15.0)), np.cos(np.radians(15.0))])
    tip = tuple(np.asarray(junction, dtype=float) + 60.0 * direction)
    capsule(mask, junction, tip, 2)
    G = polyline_graph([[(4, 6, 0), junction], [junction, (4, 6, 79)], [junction, tip]])
    return mask, G, _node(junction), _node(tip)


def flared_end():
    """A capillary (radius 2.5 um) whose blind end flares into a bell 6 um in
    radius, which Lee thinning draws as a fishtail: the vessel's centreline
    ends at a node inside the flare, and two prongs 6 um long leave it for the
    bell's rim, all well away from the image's faces. Right answer: no
    fishtail (one strand to the end). Returns ``(mask, G, fork_node,
    prong_tips)``."""
    mask = np.zeros((41, 41, 80), dtype=bool)
    for x in range(0, 50):
        r = 2.5 if x < 35 else 2.5 + 3.5 * (x - 35) / 15.0
        capsule(mask, (20, 20, x), (20, 20, x + 1), r)
    fork = (20, 20, 42)
    tips = [(20, 16, 47), (20, 24, 47)]
    G = polyline_graph([[(20, 20, 0), fork], [fork, tips[0]], [fork, tips[1]]])
    return mask, G, _node(fork), [_node(t) for t in tips]
