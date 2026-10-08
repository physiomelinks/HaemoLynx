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
