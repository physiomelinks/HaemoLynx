#!/usr/bin/env python3
"""Fit the doubled-run rule to the marks of grid H (PLAN_clean_graph.md, Phase 0).

Marks come either from the review page's database (``marks_H/annotations/*.json``,
as the dead-end grids' do) or from its "Copy marks as text" button pasted into
``marks_H.txt``: one line per example, ``H-01 D`` with an optional ``(n drawn)``
and ``[note]``. Each example is D (one vessel drawn twice) or T (two vessels,
or a branch leaving); ? and unmarked are left out.

For each candidate measure in the key (the run's length, its length in lumen
radii, the lumen radius) the best one-threshold rule is fitted and scored
leave-one-out against marking everything one way, overall and per place, and
the fit is written to ``FIT.md`` beside the key, as CLAUDE.md's "Marked
example grids" asks.

Usage::

    python scripts/doubled_run_review/fit_marks.py --review outputs/doubled_run_review --grid H
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))

MEASURES = ("run_length_um", "length_over_radius", "lumen_radius_um", "own_edge_length_um")
_LINE = re.compile(r"^\s*(?P<id>[A-Za-z]+-\d+)\s+(?P<mark>[A-Za-z?-])(?:\s*\(\d+ drawn\))?(?:\s*\[(?P<note>.*)\])?\s*$")


def read_marks(review: Path, grid: str) -> dict[str, dict]:
    """``{id: {"verdict", "note"}}`` from the database folder or the pasted text."""
    marks: dict[str, dict] = {}
    folder = review / f"marks_{grid}" / "annotations"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        body = json.loads(path.read_text(encoding="utf-8"))
        data = body.get("data", body)
        marks[data.get("id", path.stem)] = {"verdict": data.get("verdict"), "note": data.get("note", "")}
    text = review / f"marks_{grid}.txt"
    if text.is_file():
        for line in text.read_text(encoding="utf-8").splitlines():
            found = _LINE.match(line)
            if found:
                marks[found["id"]] = {"verdict": found["mark"], "note": found["note"] or ""}
    return marks


def fit(review: Path, grid: str) -> str:
    from fit_marks import loo_threshold

    key = {row["id"]: row for row in csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8"))}
    marks = read_marks(review, grid)
    rows = [(key[i], m) for i, m in marks.items() if i in key and m["verdict"] in ("D", "T")]
    if not rows:
        return f"# Grid {grid}: no D/T marks found\n"
    y = np.asarray([m["verdict"] == "D" for _k, m in rows], dtype=bool)
    counts = Counter(str(m["verdict"]) for m in marks.values())
    lines = [f"# Grid {grid}: D = one vessel drawn twice, T = two vessels / a branch leaving", "",
             f"Marks: {dict(sorted(counts.items()))}; judged {len(rows)}; all one way gets {max(y.sum(), (~y).sum())}/{len(rows)}.",
             ""]
    by_place = Counter((k["place"], m["verdict"]) for k, m in rows)
    lines += ["| place | D | T |", "|---|---|---|"]
    for place in sorted({k["place"] for k, _m in rows}):
        lines.append(f"| {place} | {by_place[(place, 'D')]} | {by_place[(place, 'T')]} |")
    lines += ["", "| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    results = []
    for measure in MEASURES:
        values = np.asarray([float(k[measure]) for k, _m in rows], dtype=float)
        result = loo_threshold(values, y)
        if result is None:
            continue
        results.append((result["loo"], measure, result))
    for loo, measure, result in sorted(results, key=lambda r: -r[0]):
        rule = f"D when {measure} {'>' if result['above'] else '<='} {result['threshold']:.3g}"
        lines.append(f"| {measure} | {loo}/{result['n']} | {rule} | {result['fit_hits']}/{result['n']} |")
    notes = [(k["id"], m["verdict"], m["note"]) for k, m in rows if m["note"]]
    if notes:
        lines += ["", "Notes:"] + [f"- {i} {v}: {n}" for i, v, n in notes]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--grid", default="H")
    args = parser.parse_args(argv)
    review = Path(args.review)
    report = fit(review, args.grid)
    (review / "FIT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
