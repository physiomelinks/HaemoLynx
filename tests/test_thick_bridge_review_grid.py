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


def test_the_best_threshold_and_its_leave_one_out_score():
    from thick_bridge_review.fit_marks import best_threshold, leave_one_out

    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, np.nan])
    y = np.array([False, False, False, True, True, True, True])
    right, threshold, direction = best_threshold(x, y)
    assert (right, threshold, direction) == (6, 6.5, ">"), "midway between the groups; the unmeasured case not counted"
    assert leave_one_out(x, y) == 6
    # Nothing beats marking all one way: no threshold is offered.
    assert best_threshold(np.array([1.0, 2.0, 3.0]), np.array([True, True, True]))[1] is None
    # One case out of place: the threshold fitted without it calls it wrong.
    noisy = np.array([False, False, True, False, True, True])
    assert leave_one_out(np.arange(6.0), noisy) < best_threshold(np.arange(6.0), noisy)[0]


def test_the_fit_reports_the_rule_against_marking_all_one_way():
    from thick_bridge_review.fit_marks import fit

    rows = [
        {"id": 1, "kind": "bridge", "mark": "B", "length_over_radius": 1.0},
        {"id": 2, "kind": "bridge", "mark": "C", "length_over_radius": 3.0},
        {"id": 3, "kind": "near_miss", "mark": "V", "length_over_radius": 0.5},
        {"id": 4, "kind": "near_miss", "mark": "N", "length_over_radius": 4.0},
        {"id": 5, "kind": "bridge", "mark": "", "length_over_radius": 2.0},
    ]

    text = fit(rows)

    assert "4 cases marked. Marks: B 1, V 1, C 1, N 1." in text
    assert "agrees on **2/4**" in text and "wrong on IDs 2, 4." in text
    assert "no bridges at all (every piece keeps its resistance) 2/4" in text
    assert "## Bridges (2)" in text and "## Near-misses (2)" in text


def test_the_page_offers_the_cases_still_to_mark_with_the_grids_four_marks(tmp_path):
    """``page_data.py``: the interactive page of sheet 2 -- the IDs with no
    mark yet, each drawn as the sheets draw it, on one sheet asking the
    grid's question with its marks."""
    import json

    from thick_bridge_review.page_data import page_case, page_data, unmarked_ids

    shape = (40, 64, 64)
    region = _tube(shape, (20, 0, 20), (20, 63, 20), 6.0)
    piece = [[20.0, 30.0, 27.5], [20.0, 30.0, 20.0]]
    cases = [{"id": i, "vox": [20, 30, 24], "centre_um": [20.0, 30.0, 24.0], "view": "down z", "piece_um": piece}
             for i in (1, 2, 3)]
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "marks.csv").write_text("id,mark\n1,C\n2,\n", encoding="utf-8")
    assert unmarked_ids(tmp_path) == [2, 3]
    case = page_case(cases[0])
    assert case["axis"] == 0 and case["rule"] == "bridge" and case["vox"] == (20, 30, 24)
    G = nx.MultiGraph()
    _edge(G, "T0", "T1", (20, 2, 20), (20, 61, 20))
    _edge(G, "B0", "B1", (20, 30, 27.5), (20, 30, 20), is_zero_resistance=True)
    capture = {"voxel_size_zyx": (1.0, 1.0, 1.0), "binary_mask": region, "thick_vessel_mask": region,
               "thick_tree": np.array([[20, y, 20] for y in range(64)])}

    data = page_data(cases, [2, 3], capture, G)

    (sheet,) = data["sheets"]
    assert [c["id"] for c in sheet["cases"]] == [2, 3]
    assert [m["key"] for m in sheet["marks"]] == ["B", "V", "C", "N", "?"]
    view = sheet["cases"][0]
    assert view["piece"][0] == [27.5, 30.0, 20.0], "[across, down, depth]: x, y, z looking down z"
    assert {line["kind"] for line in view["lines"]} == {"vessel", "bridge"}
    assert view["route"] is None and view["tick"] is None
