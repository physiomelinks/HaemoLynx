"""Grid UR1's builder (``scripts/uncovered_review/build_ur1.py``,
``PLAN_raw_confirmation.md`` Phase 4): the pieces of mask recovery is
handed, what it draws in each, how the 30 are chosen, and the outline each
card draws. The grid itself is built on E14.5 and never in CI."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.preprocessing import MaskSupport
from uncovered_review import build_ur1

SHAPE = (20, 40, 60)
VOXEL = (1.0, 1.0, 1.0)
BLOB_CENTRE = np.array([10.0, 30.0, 30.0])
BLOB_RADIUS = 4.0


def _line(a, b, step=0.5):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    count = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
    return np.linspace(a, b, count)


def _graph(*edges):
    G = nx.MultiGraph()
    for u, v, voxels, attributes in edges:
        G.add_node(u, pos=np.asarray(voxels[0], dtype=float))
        G.add_node(v, pos=np.asarray(voxels[-1], dtype=float))
        G.add_edge(u, v, voxels=voxels, length=float(np.linalg.norm(np.diff(voxels, axis=0), axis=1).sum()),
                   **attributes)
    return G


def _scene():
    """A rod with its centreline, and a ball of mask 20 um from it that no
    centreline covers."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:13, 8:13, 5:55] = True
    zz, yy, xx = np.indices(SHAPE)
    ball = (zz - BLOB_CENTRE[0]) ** 2 + (yy - BLOB_CENTRE[1]) ** 2 + (xx - BLOB_CENTRE[2]) ** 2 <= BLOB_RADIUS ** 2
    mask |= ball
    rod = (0, 1, _line((10, 10, 5), (10, 10, 54)), {})
    return mask, ball, MaskSupport(mask, VOXEL), _graph(rod), rod


def test_the_pieces_are_those_recovery_is_handed_and_each_voxel_finds_its_own():
    mask, ball, support, before, _rod = _scene()
    regions = build_ur1.Regions(before, support)
    assert len(regions.pieces) == 1
    piece = {tuple(v) for v in regions.pieces[0]}
    assert piece == {tuple(v) for v in np.argwhere(ball)}
    assert regions.radii[0] == pytest.approx(BLOB_RADIUS, abs=1.0)
    assert regions.of_voxels([[10, 30, 30], [10, 10, 30], [-1, 0, 0], [19, 39, 59]]).tolist() == [0, -1, -1, -1]
    labels = regions.label_volume()
    assert labels.dtype == np.uint16 and np.array_equal(labels > 0, ball)


def test_what_recovery_drew_is_given_to_the_piece_most_of_it_lies_in_and_its_joins_counted_apart():
    _mask, _ball, support, before, rod = _scene()
    regions = build_ur1.Regions(before, support)
    after = _graph(
        rod,
        (2, 3, _line((10, 30, 27), (10, 30, 33)), {"recovered": True}),
        (2, 4, _line((10, 30, 27), (10, 12, 27)), {"recovered": True, "reconnected": True}),
        (5, 6, _line((10, 10, 40), (10, 10, 45)), {"recovered": True}),  # in no piece
    )
    drawn = build_ur1.drawn_in(after, regions, VOXEL)
    assert set(drawn) == {0}
    assert drawn[0]["joins"] == 1 and len(drawn[0]["edges"]) == 1
    assert drawn[0]["drawn_um"] == pytest.approx(6.0)
    assert len(drawn[0]["paths"]) == 1


def test_the_deepest_points_come_first_and_keep_their_distance():
    points = np.array([[0.0, 0, 0], [0.5, 0, 0], [3.0, 0, 0], [6.0, 0, 0], [6.5, 0, 0]])
    radii = np.array([5.0, 4.9, 3.0, 2.0, 1.9])
    deep = build_ur1.deepest_points(points, radii, count=3)
    assert deep.tolist() == [[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]]


def test_the_thirty_are_ten_recovery_draws_into_and_twenty_it_does_not_each_half_by_depth():
    rows = []
    for k in range(39):
        rows.append({"region": k, "drawn_um": 5.0, "depth_um": float(k % 13) * 20.0, "volume_um3": 100.0 + k,
                     "dextran_median": 0.5, "dextran_over_neighbour": k / 39.0})
    for k in range(39, 400):
        rows.append({"region": k, "drawn_um": 0.0, "depth_um": float(k % 29) * 10.0, "volume_um3": 50.0 + k,
                     "dextran_median": 0.4, "dextran_over_neighbour": None if k % 50 == 0 else (k % 97) / 50.0})
    chosen = build_ur1.choose(rows)
    assert len(chosen) == 30 and len({r["region"] for r in chosen}) == 30
    drawn = [r for r in chosen if r["drawn_um"] > 0]
    others = [r for r in chosen if r["drawn_um"] <= 0]
    assert len(drawn) == 10 and len(others) == 20
    for group, pool in ((drawn, rows[:39]), (others, rows[39:])):
        cut = np.median([r["depth_um"] for r in pool])
        assert sum(r["depth_um"] <= cut for r in group) == len(group) // 2
        # Spread across the local brightness, both ends of it taken.
        values = sorted(build_ur1.local_brightness(r) for r in group)
        everything = sorted(build_ur1.local_brightness(r) for r in pool)
        assert values[0] <= np.percentile(everything, 10) and values[-1] >= np.percentile(everything, 90)
    assert build_ur1.local_brightness({"dextran_over_neighbour": None}) == 0.0


def test_a_short_pool_of_drawn_pieces_is_all_taken_and_the_rest_come_from_the_others():
    rows = [{"region": k, "drawn_um": 5.0 if k < 4 else 0.0, "depth_um": float(k), "volume_um3": 1.0,
             "dextran_median": 0.5, "dextran_over_neighbour": 0.5} for k in range(100)]
    chosen = build_ur1.choose(rows)
    assert len(chosen) == 30 and sum(r["drawn_um"] > 0 for r in chosen) == 4


def test_each_card_outlines_its_piece_in_every_channel():
    mask, _ball, support, before, _rod = _scene()
    regions = build_ur1.Regions(before, support)
    raw = np.where(mask, 1000.0, 50.0)
    drawer = build_ur1.RegionDrawer(before, mask, {"Dextran": raw, "Claudin-5": raw}, VOXEL, regions.label_volume())
    nothing = {"edges": [], "joins": 0, "drawn_um": 0.0, "paths": []}
    planes = build_ur1.views(drawer, 0, regions, nothing, VOXEL)
    assert [p["label"] for p in planes] == ["Its own plane", "Across it"]
    for plane in planes:
        outlines = [o for o in plane["overlays"] if o.get("color") == build_ur1.OUTLINE_COLOUR]
        assert len(outlines) == 1 and outlines[0]["marker"]  # drawn on the stains too
        ring = np.asarray([p for p in outlines[0]["points"] if p is not None])
        # The plane is centred on the ball: its outline runs round the
        # centre at the ball's radius.
        reach = np.linalg.norm(ring, axis=1)
        assert reach.min() >= BLOB_RADIUS - 1.5 and reach.max() <= BLOB_RADIUS + 1.5
        assert len(plane["images"]) == 3


def test_a_fresh_grid_leaves_out_the_pieces_an_earlier_one_showed(tmp_path):
    key = tmp_path / "UR1_key.csv"
    key.write_text("id,centre_zyx_vox\nUR1-01,\"(1, 2, 3)\"\n", encoding="utf-8")
    rows = [{"centre_zyx_vox": (1, 2, 3), "drawn_um": 4.0}, {"centre_zyx_vox": (4, 5, 6), "drawn_um": 0.0},
            {"centre_zyx_vox": (7, 8, 9), "drawn_um": 2.0}]
    shown = build_ur1.shown_centres([key])
    assert shown == {"(1, 2, 3)"}
    assert [r["centre_zyx_vox"] for r in build_ur1.fresh(rows, shown)] == [(4, 5, 6), (7, 8, 9)]
    assert [r["centre_zyx_vox"] for r in build_ur1.fresh(rows, shown, drawn_only=True)] == [(7, 8, 9)]
    # Fewer drawn pieces than the grid holds: every one of them.
    assert len(build_ur1.choose([dict(r, depth_um=1.0, volume_um3=1.0, dextran_median=0.5,
                                      dextran_over_neighbour=0.5, region=i)
                                 for i, r in enumerate(rows) if r["drawn_um"] > 0], drawn_share=30)) == 2


def _marked(values, marks, grid="UR1"):
    rows = [{"id": f"{grid}-{i:02d}", "grid": grid, "region": str(i), "drawn_um": "5.0",
             "dextran_over_neighbour": "" if v is None else str(v)} for i, v in enumerate(values, start=1)]
    return rows, {row["id"]: {"verdict": m} for row, m in zip(rows, marks)}


def test_a_measure_that_splits_the_marks_scores_every_piece_left_out_in_turn():
    from uncovered_review import fit_ur1

    rows, marks = _marked([0.2, 0.3, 0.4, 0.5, 1.5, 1.6, 0.35, 0.45], list("HHHHVVHH"))
    one_way, results = fit_ur1.fit(fit_ur1.judged(rows, marks))
    assert one_way == 6
    best = results[0]
    assert best["measure"] == "dextran_over_neighbour" and best["loo"] == best["n"] == 8
    assert best["above"] and 0.5 < best["threshold"] < 1.5
    # The piece's number is a label, not a measure.
    assert all(r["measure"] != "region" for r in results)


def test_a_fresh_grids_fixed_rules_are_scored_unfitted_and_unsure_marks_left_out():
    from uncovered_review import fit_ur1

    rows, marks = _marked([1.2, 0.9, 1.1, None, 0.5, 0.4], list("VHHV?H"), grid="UR2")
    marked = fit_ur1.judged(rows, marks)
    assert [r["id"] for r in marked] == ["UR2-01", "UR2-02", "UR2-03", "UR2-04", "UR2-06"]
    checked = {c["rule"]: c for c in fit_ur1.check_rules("UR2", marked)}
    local = checked["dextran over its neighbours above 1.05"]
    # 01 V and 1.2: right; 03 H but 1.1: wrong; 04 V with no reading: called H, wrong.
    assert (local["right"], local["n"], local["wrong"]) == (3, 5, ["UR2-03", "UR2-04"])
    assert checked["every piece H (recovery draws nothing)"]["wrong"] == ["UR2-01", "UR2-04"]
    text = fit_ur1.report(rows, marks)
    assert "Unsure or unmarked, left out: UR2-05." in text
    assert "| dextran over its neighbours above 1.05 | 3/5 | UR2-03, UR2-04 |" in text
    assert fit_ur1.check_rules("UR1", marked) == []
