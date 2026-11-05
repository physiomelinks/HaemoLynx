#!/usr/bin/env python3
"""Fit grid I's marks (PLAN_raw_confirmation.md, Phase 3): which measure says a
short loop is false (closes inside one vessel) rather than real (round tissue)?

Marks from the review page's database (``marks_<grid>/annotations/*.json``):
F = a false loop (remove a side), C = a real loop (keep), ? left out. Every
measure in the grid's key -- the raw image's reading of the hole and of the
loop's walls, and the mask's own loop measures -- is scored leave-one-out
with the one-threshold rule every grid uses, against marking everything one
way.

    python scripts/loop_review/fit_marks_i.py --review outputs/loop_review --grid I
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))
sys.path.insert(0, str(HERE.parent / "bridge_review"))

MEASURES = (
    "raw_hole_lumen", "raw_hole_centre", "raw_loop_wall", "raw_loop_wall_min", "raw_loop_wall_bright",
    "loop_length_um", "lumen_radius_um", "span_um", "area_um2", "out_of_plane", "path_in",
    "chords_in_lumen", "chord_bg_median_um", "n_edges", "n_sides", "hole_points",
)
POSITIVE, NEGATIVE = "F", "C"


def fit(review: Path, grid: str) -> str:
    from fit_marks import loo_threshold
    from fit_marks_m import read_marks

    key = {row["id"]: row for row in csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8"))}
    marks = read_marks(review, grid)
    judged = [(key[i], m) for i, m in marks.items() if i in key and m["verdict"] in (POSITIVE, NEGATIVE)]
    counts = Counter(str(m["verdict"]) for m in marks.values())
    y = np.asarray([m["verdict"] == POSITIVE for _k, m in judged], dtype=bool)
    lines = [f"# Grid {grid}: {POSITIVE} = a false loop (remove a side), {NEGATIVE} = a real loop (keep)", "",
             f"Marks: {dict(sorted(counts.items()))}; judged {len(judged)} of {len(key)}; "
             f"all one way gets {max(y.sum(), (~y).sum()) if len(y) else 0}/{len(judged)}.", "",
             "| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    scored = []
    for measure in MEASURES:
        values = np.asarray([float(k[measure]) if k.get(measure) not in (None, "") else np.nan for k, _m in judged], dtype=float)
        result = loo_threshold(values, y)
        if result is not None:
            scored.append((result["loo"], measure, result))
    for loo, measure, result in sorted(scored, key=lambda s: -s[0]):
        rule = f"{POSITIVE} when {measure} {'>' if result['above'] else '<='} {result['threshold']:.3g}"
        lines.append(f"| {measure} | {loo}/{result['n']} | {rule} | {result['fit_hits']}/{result['n']} |")
    lines += ["", "| id | mark | length µm | r µm | hole lumen | hole centre | loop wall | wall min | path in | chords in lumen | drawn | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in sorted(judged, key=lambda km: km[0]["id"]):
        lines.append(
            f"| {k['id']} | {m['verdict']} | {k['loop_length_um']} | {k['lumen_radius_um']} | {k['raw_hole_lumen']} | "
            f"{k['raw_hole_centre']} | {k['raw_loop_wall']} | {k['raw_loop_wall_min']} | {k['path_in']} | "
            f"{k['chords_in_lumen']} | {m.get('strokes', 0)} | {m['note']} |"
        )
    unmarked = [i for i in key if i not in marks or marks[i]["verdict"] not in (POSITIVE, NEGATIVE)]
    if unmarked:
        lines += ["", f"Not judged: {', '.join(unmarked)}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--grid", default="I")
    args = parser.parse_args(argv)
    review = Path(args.review)
    text = fit(review, args.grid)
    (review / f"FIT_{args.grid}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
