"""The Phase 5 grid of PLAN_thick_vessel_bridges.md
(``scripts/thick_bridge_review/build_grid.py``): which stretches it offers as
near-misses beside a thick vessel, and how it spreads its picks. The grid
itself is drawn from a run and never in CI."""
from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from thick_bridge_review.build_grid import DISTANCE_BINS, RATIO_BINS, near_misses, spread


def _tube(shape, a, b, radius):
    p = np.stack(np.indices(shape), axis=-1).astype(float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1) <= radius


def _edge(G, u, v, a, b, **attrs):
    count = int(np.ceil(np.linalg.norm(np.subtract(b, a)))) + 1
    path = [tuple(float(c) for c in q) for q in np.linspace(a, b, count)]
    G.add_node(u, pos=path[0])
    G.add_node(v, pos=path[-1])
    G.add_edge(u, v, voxels=path, length=float(np.linalg.norm(np.subtract(b, a))), **attrs)


def test_near_misses_are_the_vessel_stretches_in_the_region_off_the_tree():
    """A thick vessel 6 um in radius along y: its own centreline on the tree
    is no near-miss; a strand 4 um off the tree through the lumen, and a
    vessel grazing the region's edge, are; a vessel far outside is not."""
    shape = (40, 64, 64)
    region = _tube(shape, (20, 0, 20), (20, 63, 20), 6.0)
    tree_points = np.array([[20.0, y, 20.0] for y in range(64)])
    radius = np.full(len(tree_points), 6.0)
    tolerance = np.maximum(1.5, 0.3 * radius)
    G = nx.MultiGraph()
    _edge(G, "T0", "T1", (20, 2, 20), (20, 61, 20))
    _edge(G, "S0", "S1", (20, 10, 24), (20, 30, 24))
    _edge(G, "G0", "G1", (27, 40, 0), (27, 40, 40))
    _edge(G, "F0", "F1", (5, 5, 5), (5, 60, 5))
    _edge(G, "B0", "B1", (20, 50, 27), (20, 50, 20), is_zero_resistance=True)

    found = near_misses(G, region, cKDTree(tree_points), radius, tolerance, np.ones(3))

    edges = sorted({frozenset(case["edge"][:2]) for case in found}, key=sorted)
    assert edges == sorted([frozenset(("S0", "S1")), frozenset(("G0", "G1"))], key=sorted)
    (strand,) = [c for c in found if set(c["edge"][:2]) == {"S0", "S1"}]
    assert strand["length_um"] == pytest.approx(20.0)
    assert strand["distance_over_tolerance"] == pytest.approx(4.0 / 1.8)
    assert strand["inside_region_share"] == 1.0
    (graze,) = [c for c in found if set(c["edge"][:2]) == {"G0", "G1"}]
    assert graze["inside_region_share"] < 1.0 and graze["distance_to_tree_um"] >= 7.0


def test_the_picks_are_spread_over_both_measures_before_any_cell_gives_two():
    cases = [
        {"distance_over_tolerance": d, "length_over_radius": r, "n": i}
        for i, (d, r) in enumerate([(0.2, 0.5)] * 10 + [(1.2, 0.5), (3.0, 1.0), (5.0, 3.0)])
    ]

    picked = spread(cases, 4, random.Random(0))

    cells = {(np.searchsorted(DISTANCE_BINS, c["distance_over_tolerance"], side="right"),
              np.searchsorted(RATIO_BINS, c["length_over_radius"], side="right")) for c in picked}
    assert len(picked) == 4 and len(cells) == 4
    assert len(spread(cases, 50, random.Random(0))) == len(cases)
    with_must = spread(cases, 4, random.Random(0), must=[12])
    assert cases[12] in with_must and len(with_must) == 4
