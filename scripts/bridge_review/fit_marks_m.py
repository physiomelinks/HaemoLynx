#!/usr/bin/env python3
"""Fit grid M's marks (PLAN_raw_confirmation.md, Phase 1): which measure says a
vessel runs across a gap in the segmentation?

Marks come from the review page's database (``marks_<grid>/annotations/*.json``):
V = one vessel the segmentation broke (join), N = not (leave), ? = unsure
(left out). Every measure in the grid's key is scored leave-one-out with the
one-threshold rule every grid uses (``scripts/dead_end_review/fit_marks.py``'s
``loo_threshold``), overall and per kind of candidate, against marking
everything one way.

    python scripts/bridge_review/fit_marks_m.py --review outputs/bridge_review --grid M
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))

MEASURES = (
    "raw_gap_bright", "raw_gap_min", "raw_fraction_bright", "raw_longest_dark_um", "raw_wall_between",
    "mask_background_um", "mask_background_stretches", "gap_um", "turn_deg", "chord_um", "end_radius_um",
)
POSITIVE, NEGATIVE = "V", "N"


def read_marks(review: Path, grid: str) -> dict[str, dict]:
    marks: dict[str, dict] = {}
    folder = review / f"marks_{grid}" / "annotations"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        body = json.loads(path.read_text(encoding="utf-8"))
        data = body.get("data", body)
        marks[data.get("id", path.stem)] = {"verdict": data.get("verdict"), "note": data.get("note", "")}
    return marks


def _table(rows, y, measures) -> list[str]:
    from fit_marks import loo_threshold

    lines = ["| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    scored = []
    for measure in measures:
        values = np.asarray([float(k[measure]) if k.get(measure) not in (None, "") else np.nan for k in rows], dtype=float)
        result = loo_threshold(values, y)
        if result is not None:
            scored.append((result["loo"], measure, result))
    for loo, measure, result in sorted(scored, key=lambda s: -s[0]):
        rule = f"{POSITIVE} when {measure} {'>' if result['above'] else '<='} {result['threshold']:.3g}"
        lines.append(f"| {measure} | {loo}/{result['n']} | {rule} | {result['fit_hits']}/{result['n']} |")
    return lines


def fit(review: Path, grid: str) -> str:
    key = {row["id"]: row for row in csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8"))}
    marks = read_marks(review, grid)
    judged = [(key[i], m) for i, m in marks.items() if i in key and m["verdict"] in (POSITIVE, NEGATIVE)]
    counts = Counter(str(m["verdict"]) for m in marks.values())
    y = np.asarray([m["verdict"] == POSITIVE for _k, m in judged], dtype=bool)
    lines = [f"# Grid {grid}: {POSITIVE} = one vessel across the gap (join), {NEGATIVE} = not (leave)", "",
             f"Marks: {dict(sorted(counts.items()))}; judged {len(judged)} of {len(key)}; "
             f"all one way gets {max(y.sum(), (~y).sum()) if len(y) else 0}/{len(judged)}.", ""]
    by_kind = Counter((k["kind"], m["verdict"]) for k, m in judged)
    lines += ["| kind | V | N |", "|---|---|---|"]
    for kind in sorted({k["kind"] for k, _m in judged}):
        lines.append(f"| {kind} | {by_kind[(kind, 'V')]} | {by_kind[(kind, 'N')]} |")
    lines += ["", "## All candidates", ""] + _table([k for k, _m in judged], y, MEASURES)
    for kind in sorted({k["kind"] for k, _m in judged}):
        sub = [(k, m) for k, m in judged if k["kind"] == kind]
        ys = np.asarray([m["verdict"] == POSITIVE for _k, m in sub], dtype=bool)
        if len(sub) >= 4 and len(np.unique(ys)) == 2:
            lines += ["", f"## {kind} only ({len(sub)}; all one way {max(ys.sum(), (~ys).sum())}/{len(sub)})", ""]
            lines += _table([k for k, _m in sub], ys, MEASURES)
    lines += ["", "| id | kind | mark | gap µm | mask bg µm | raw gap bright | raw gap min | raw dark µm | wall between | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in sorted(judged, key=lambda km: km[0]["id"]):
        lines.append(
            f"| {k['id']} | {k['kind']} | {m['verdict']} | {k['gap_um']} | {k['mask_background_um']} | {k['raw_gap_bright']} | "
            f"{k['raw_gap_min']} | {k['raw_longest_dark_um']} | {k['raw_wall_between']} | {m['note']} |"
        )
    unmarked = [i for i in key if i not in marks or marks[i]["verdict"] not in (POSITIVE, NEGATIVE)]
    if unmarked:
        lines += ["", f"Not judged: {', '.join(unmarked)}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--grid", default="M")
    args = parser.parse_args(argv)
    review = Path(args.review)
    text = fit(review, args.grid)
    (review / f"FIT_{args.grid}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
