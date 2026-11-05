#!/usr/bin/env python3
"""Fit grid UR1's marks: does any measure tell a vessel the graph missed (V)
from a piece that is not a vessel of its own (H)?

``PLAN_raw_confirmation.md`` Phase 4. Every numeric column of the key
(``build_ur1.py``) is fitted as one threshold, ``V when value > t`` or
``<= t``, scored leave-one-out and fitted on all, against marking every piece
one way -- over all the judged pieces, and over those mask recovery draws a
centreline into, the only ones a skip rule would change. A "?" is left out.

A fresh grid is also scored by the rules fixed for it before it was marked
(:data:`FRESH_RULES`), unfitted.

Usage::

    python scripts/uncovered_review/fit_ur1.py --key outputs/uncovered_review/UR1_key.csv \\
        --marks outputs/uncovered_review/marks_UR1 --out outputs/uncovered_review/FIT_UR1.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))

from fit_marks import loo_threshold, read_marks  # noqa: E402

#: Key columns that are labels, not measures: the piece's number follows the
#: labelling's scan order (so it stands in for depth, as node ids did on DR1)
#: and the centre voxel is a position.
NOT_MEASURES = frozenset({"id", "grid", "region", "centre_zyx_vox"})


#: The rules each fresh grid checks, fixed before its marks: ``(name, measure,
#: above, threshold)`` -- V when the measure is above (or at most) the
#: threshold -- or ``(name, None, ...)`` for every piece H. UR2's, from UR1
#: (2026-10-25): on the ten pieces recovery draws into, the dextran over the
#: neighbouring centrelines' above 1.05 was V 10/10 leave-one-out (2 V, 8 H),
#: against 8/10 for H every time, which is recovery drawing nothing.
FRESH_RULES: dict[str, tuple[tuple, ...]] = {
    "UR2": (
        ("dextran over its neighbours above 1.05", "dextran_over_neighbour", True, 1.05),
        ("every piece H (recovery draws nothing)", None, True, 0.0),
    ),
}


def check_rules(grid: str, rows: list[dict]) -> list[dict]:
    """*grid*'s fixed rules scored on its judged *rows*, unfitted: right,
    of how many, and the IDs each gets wrong. A piece without the measure is
    called H."""
    results = []
    for name, measure, above, threshold in FRESH_RULES.get(grid, ()):
        wrong = []
        for row in rows:
            value = _number(row.get(measure)) if measure else np.nan
            says_v = bool(np.isfinite(value) and (value > threshold if above else value <= threshold))
            if says_v != (row["mark"] == "V"):
                wrong.append(row["id"])
        results.append({"rule": name, "right": len(rows) - len(wrong), "n": len(rows), "wrong": wrong})
    return results


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return np.nan
    return value


def judged(rows: list[dict], marks: dict[str, dict]) -> list[dict]:
    """The key rows marked V or H, each with its ``mark``."""
    out = []
    for row in rows:
        mark = (marks.get(row["id"]) or {}).get("verdict")
        if mark in ("V", "H"):
            out.append({**row, "mark": mark})
    return out


def measures(rows: list[dict]) -> list[str]:
    """The key's numeric columns, in key order."""
    names = []
    for name in rows[0] if rows else ():
        if name in NOT_MEASURES or name == "mark":
            continue
        if any(np.isfinite(_number(row.get(name))) for row in rows):
            names.append(name)
    return names


def fit(rows: list[dict]) -> tuple[int, list[dict]]:
    """``(all one way, one result per measure)``, best leave-one-out first;
    a measure that cannot be fitted (fewer than four values, or one mark
    among them) is left out."""
    labels = np.asarray([row["mark"] == "V" for row in rows])
    one_way = int(max(labels.sum(), (~labels).sum()))
    results = []
    for name in measures(rows):
        values = np.asarray([_number(row.get(name)) for row in rows], dtype=float)
        result = loo_threshold(values, labels)
        if result is not None:
            results.append({"measure": name, **result})
    results.sort(key=lambda r: (-r["loo"], -r["fit_hits"], r["measure"]))
    return one_way, results


def _rule(result: dict) -> str:
    return f"V when {result['measure']} {'>' if result['above'] else '<='} {result['threshold']:.3g}"


def table(title: str, rows: list[dict]) -> list[str]:
    """One section of the report: the marks, all one way, every measure."""
    v = sum(row["mark"] == "V" for row in rows)
    one_way, results = fit(rows)
    lines = [f"## {title}", "", f"{len(rows)} judged: {v} V, {len(rows) - v} H; all one way gets "
             f"{one_way}/{len(rows)}.", ""]
    if not results:
        return lines + ["Nothing to fit: one mark only, or too few pieces.", ""]
    lines += ["| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    lines += [f"| {r['measure']} | {r['loo']}/{r['n']} | {_rule(r)} | {r['fit_hits']}/{r['n']} |" for r in results]
    return lines + [""]


#: The per-piece table's columns: ``(key column, heading)``.
DETAIL = (
    ("volume_um3", "volume um3"), ("radius_um", "radius um"), ("depth_um", "depth um"),
    ("drawn_um", "drawn um"), ("dextran_median", "dextran"), ("dextran_over_neighbour", "dextran / neighbours"),
    ("claudin_median", "claudin-5"), ("final_uncovered_fraction", "left uncovered"),
)


def report(rows: list[dict], marks: dict[str, dict]) -> str:
    marked = judged(rows, marks)
    unsure = [row["id"] for row in rows if (marks.get(row["id"]) or {}).get("verdict") not in ("V", "H")]
    drawn = [row for row in marked if _number(row.get("drawn_um")) > 0]
    grid = rows[0].get("grid", "UR1") if rows else "UR1"
    lines = [f"# Grid {grid}: V = a vessel the graph missed, H = not a vessel of its own", ""]
    if unsure:
        lines += [f"Unsure or unmarked, left out: {', '.join(unsure)}.", ""]
    checked = check_rules(grid, marked)
    if checked:
        one_way, _results = fit(marked)
        lines += ["## The rules fixed before the marks", "", f"All one way gets {one_way}/{len(marked)}.", "",
                  "| rule | right | wrong |", "|---|---|---|"]
        lines += [f"| {c['rule']} | {c['right']}/{c['n']} | {', '.join(c['wrong'])} |" for c in checked]
        lines.append("")
    lines += table("Every piece", marked)
    lines += table("The pieces mask recovery draws a centreline into", drawn)
    lines += ["## Each piece", "", "| id | mark | evidence | " + " | ".join(h for _k, h in DETAIL) + " |",
              "|" + "---|" * (3 + len(DETAIL))]
    for row in sorted(marked, key=lambda r: r["id"]):
        evidence = ", ".join((marks.get(row["id"]) or {}).get("evidence") or [])
        cells = [f"{_number(row.get(k)):.3g}" if np.isfinite(_number(row.get(k))) else "" for k, _h in DETAIL]
        lines.append(f"| {row['id']} | {row['mark']} | {evidence} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--key", required=True)
    parser.add_argument("--marks", required=True, help="the folder holding annotations/*.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.key, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    text = report(rows, read_marks(Path(args.marks)))
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
