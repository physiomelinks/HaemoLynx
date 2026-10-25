#!/usr/bin/env python3
"""PLAN_bridging.md, after grid P: every case of the wall join's two open
questions on E14.5 that grid P did not mark, for Harvey to mark (his call,
2026-10-25: "grid of the rest").

- **The reach** (``WALL_JOIN_REACH_UM``, 10 um): dead ends past it, up to the
  pool's 15 um, heading for the thick vessel (the chord within the join's
  80 degrees). Grid P marked two, P02 and P12, both J.
- **The in-lumen loop guard**: dead ends the join refused as beside a vessel
  or closing a loop inside one lumen. Grid P marked three: P01 and P13 J,
  P03 N.

The same capture, replay, drawing and J/N question as grid P
(``phase1_rules_grid.py``), the cases of both questions shuffled onto one
sheet so an ID says nothing, numbered on from grid P (P31, ...). Which
question each case answers, and its measures, are in ``key_wall.csv`` only.

    python scripts/thick_junction_review/wall_rest_grid.py \\
        --capture outputs/br/mine/finish_input.pkl \\
        --marked outputs/thick_junction_review/grid_P/key_wall.csv \\
        --out outputs/thick_junction_review/grid_P2 \\
        --page-data outputs/thick_junction_review/grid_P2/page.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

try:
    from thick_junction_review import phase1_rules_grid as grid
except ImportError:  # run as a script, from its own folder
    import phase1_rules_grid as grid

SEED = 20261026
FIRST_ID = 31
#: A pool case this near (um) one grid P marked is that case.
MARKED_WITHIN_UM = 2.0
BEYOND_REACH = "beyond the reach"
LOOP_GUARD = "beside a vessel, or a loop"
QUESTION_OF = {BEYOND_REACH: "reach", LOOP_GUARD: "loop guard"}
KEY_FIELDS = grid.KEY_FIELDS["wall"][:6] + ("question",) + grid.KEY_FIELDS["wall"][6:]


def rest_of_pool(pool, marked_centres, *, max_turn_deg: float) -> list[dict]:
    """The wall join's *pool* cases each open question still needs: past the
    reach and heading for the thick vessel (chord turn under *max_turn_deg*),
    or refused by the in-lumen loop guard -- leaving out any within
    :data:`MARKED_WITHIN_UM` of a centre already marked."""
    marked = np.asarray(marked_centres, float).reshape(-1, 3)
    rest = []
    for case in pool:
        decision = case["decision"]
        if decision == BEYOND_REACH:
            if not np.nan_to_num(case["chord_turn_deg"], nan=180.0) < max_turn_deg:
                continue
        elif decision != LOOP_GUARD:
            continue
        if len(marked) and float(np.min(np.linalg.norm(marked - np.asarray(case["centre"], float), axis=1))) \
                < MARKED_WITHIN_UM:
            continue
        rest.append(dict(case, question=QUESTION_OF[decision]))
    return rest


def _marked_centres(key_csv: Path) -> np.ndarray:
    with open(key_csv, encoding="utf-8") as f:
        return np.asarray([json.loads(r["centre_um"].replace("(", "[").replace(")", "]"))
                           for r in csv.DictReader(f)], float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="e14_run.py --capture's finish_input.pkl (grid P's)")
    parser.add_argument("--marked", required=True, help="grid P's key_wall.csv: its cases are not asked again")
    parser.add_argument("--out", required=True)
    parser.add_argument("--page-data", help="also write the cases as the interactive page's JSON (grid_page.py)")
    args = parser.parse_args(argv)

    import build_grid as phase5

    from haemolynx.graph import IS_ZERO_RESISTANCE
    from haemolynx.graph.build import MAX_GAP_BRIDGE_TURN_DEG

    seen = grid.replay(Path(args.capture))
    cap = seen["cap"]
    vs = np.asarray(cap["voxel_size_zyx"], float)
    mask = np.asarray(cap["binary_mask"], bool)
    region = np.asarray(cap["thick_vessel_mask"], bool)
    tree_points = np.asarray(cap["thick_tree"], float) * vs

    pool = grid.wall_cases(seen)
    picked = rest_of_pool(pool, _marked_centres(Path(args.marked)), max_turn_deg=MAX_GAP_BRIDGE_TURN_DEG)
    random.Random(SEED).shuffle(picked)
    for i, case in enumerate(picked, start=FIRST_ID):
        case["id"] = f"P{i:02d}"
        case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
        shown = case["piece"] if case.get("route") is None else np.vstack([case["piece"], case["route"]])
        case["axis"] = phase5._plane(dict(case, piece=shown), tree_points)
        case["view"] = phase5.PLANES[case["axis"]][0]
        case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drawn = seen["join_in"]
    paths = [(phase5._edge_path(drawn, u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
             for u, v, d in drawn.edges(data=True) if u != v]
    if picked:
        grid._sheet(picked, out / "grid_P2_wall.png", mask, region, tree_points, paths, vs,
                    f"E14.5, grid P2 (wall joins, IDs {picked[0]['id']}-{picked[-1]['id']})   "
                    + grid.QUESTIONS["wall"])
    with open(out / "key_wall.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in sorted(picked, key=lambda c: c["id"]):
            writer.writerow({k: grid._plain(case.get(k)) for k in KEY_FIELDS})
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "rule", "mark"])
        writer.writerows([c["id"], "wall", ""] for c in sorted(picked, key=lambda c: c["id"]))
    if args.page_data:
        Path(args.page_data).write_text(json.dumps({
            "grid": "P2",
            "dataset": "E14.5 MCA, Harvey's Claudin segmentation (grid P2: the wall join's rest)",
            "voxel_size_zyx": [float(v) for v in vs],
            "slab_um": 2 * phase5.HALF_SLAB_UM,
            "sheets": [{
                "rule": "wall", "title": "Wall joins", "question": grid.QUESTIONS["wall"].split("\n")[0],
                "marks": [{"key": k, "label": label} for k, label in grid.MARKS["wall"]],
                "cases": [grid.case_view(case, mask, region, tree_points, paths, vs) for case in picked],
            }],
        }, separators=(",", ":")), encoding="utf-8")
    summary = {
        "pool": len(pool),
        "pool_decisions": grid._count(c["decision"] for c in pool),
        "picked": len(picked),
        "picked_questions": grid._count(c["question"] for c in picked),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
