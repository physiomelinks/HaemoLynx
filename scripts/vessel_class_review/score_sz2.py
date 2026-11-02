"""Grid SZ2's marks against Harvey's small-vessel rule (6-15 um, grown from the
large vessels), the fresh check: the rule was fixed before any SZ2 mark.

The rule's call is the key's ``rule`` column (``through_wider_rule`` for the
other reading of "propagate"). Scored overall and per group, against marking
every vessel one way. Then each measure's best cut fitted on SZ1 is applied to
SZ2 unchanged, so the rule is compared with what a width (or any other) cut
fitted on the first grid does on the second.

    python scripts/vessel_class_review/score_sz2.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from score_sz1 import MEASURES, b_order, best_cut

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
GRID = OUT / "SZ2"
REPORT = GRID / "SZ2_result.md"


def read(grid: str):
    marks = {int(r["id"]): r["mark"] for r in csv.DictReader((OUT / grid / f"{grid}_marks.csv").open(encoding="utf-8"))}
    rows = {int(r["id"]): r for r in csv.DictReader((OUT / grid / f"{grid}_key.csv").open(encoding="utf-8"))}
    for r in rows.values():
        r["b_order"] = b_order(r["branch_order"])
    decided = sorted(i for i, m in marks.items() if m in ("S", "C"))
    return marks, rows, decided


def column(rows, ids, name) -> np.ndarray:
    return np.array([float(rows[i][name]) if rows[i][name] not in ("", "nan") else np.nan for i in ids])


def main() -> None:
    marks, rows, decided = read("SZ2")
    small = {i: marks[i] == "S" for i in decided}
    n = len(decided)
    one_way = max(sum(small.values()), n - sum(small.values()))
    counts = Counter(marks.get(i, "unmarked") for i in rows)
    lines = ["# Grid SZ2 result: the fresh check of the small-vessel rule", "",
             f"Marked by Harvey on the SZ2 page, 2026-10-26: {dict(counts)}. Scored on the {n} marked S or C; "
             f"marking all one way gets {one_way}/{n}.", "",
             "| Reading | Right | Wrong |", "|---|---|---|"]
    for column_name, reading in (("rule", "strict: grown only through 6-15 um vessels"),
                                 ("through_wider_rule", "grown through wider vessels too")):
        called = {i: rows[i][column_name] == "small" for i in decided}
        right = sum(called[i] == small[i] for i in decided)
        wrong = ", ".join(f"{i}{'S' if small[i] else 'C'}" for i in decided if called[i] != small[i])
        lines.append(f"| {reading} | {right}/{n} | {wrong or '-'} |")

    lines += ["", "By group (strict reading):", "",
              "| Group | Rule calls | Cases | Marked S | Marked C | Rule right |", "|---|---|---|---|---|---|"]
    for group in ("small", "band_unreached", "under_band", "over_band"):
        ids = [i for i in decided if rows[i]["group"] == group]
        s = sum(small[i] for i in ids)
        right = sum((rows[i]["rule"] == "small") == small[i] for i in ids)
        lines.append(f"| {group} | {'small' if group == 'small' else 'not small'} | {len(ids)} | {s} | "
                     f"{len(ids) - s} | {right}/{len(ids)} |")

    # Each measure's best cut on SZ1, applied unchanged to SZ2.
    m1, rows1, decided1 = read("SZ1")
    small1 = np.array([m1[i] == "S" for i in decided1])
    small2 = np.array([small[i] for i in decided])
    lines += ["", "Each measure's best cut fitted on SZ1, applied unchanged to SZ2, and SZ2's own best cut "
              "(in-sample, an upper bound):", "",
              "| Measure | SZ1 cut | SZ1 right | SZ2 right with SZ1's cut | SZ2's own best |", "|---|---|---|---|---|"]
    for name in MEASURES:
        x1, x2 = column(rows1, decided1, name), column(rows, decided, name)
        ok1, ok2 = np.isfinite(x1), np.isfinite(x2)
        if ok1.sum() < 5 or ok2.sum() < 5:
            continue
        right1, t, above = best_cut(x1[ok1], small1[ok1])
        called2 = x2[ok2] >= t if above else x2[ok2] < t
        right2 = int((called2 == small2[ok2]).sum())
        own, t2, above2 = best_cut(x2[ok2], small2[ok2])
        lines.append(f"| {name} | S {'>=' if above else '<'} {t:.3g} | {right1}/{int(ok1.sum())} | "
                     f"{right2}/{int(ok2.sum())} | {own}/{int(ok2.sum())} (S {'>=' if above2 else '<'} {t2:.3g}) |")

    lines += ["", "Each case, by width:", "",
              "| ID | Mark | Width um | Source | Group | Rule | Branch order | Widest neighbour um |",
              "|---|---|---|---|---|---|---|---|"]
    for i in sorted(rows, key=lambda i: float(rows[i]["diameter_um"])):
        r = rows[i]
        neighbour = f"{float(r['widest_neighbour_um']):.1f}" if r["widest_neighbour_um"] else "-"
        lines.append(f"| {i} | {marks.get(i) or '-'} | {float(r['diameter_um']):.2f} | {r['diameter_source']} | "
                     f"{r['group']} | {r['rule']} | {r['branch_order']} | {neighbour} |")

    # The two grids side by side: how often each width range was marked S.
    bins = ((4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 19))
    lines += ["", "Marked S by width, each grid:", "", "| Width um | SZ1 | SZ2 |", "|---|---|---|"]
    w1 = {i: float(rows1[i]["diameter_um"]) for i in decided1}
    w2 = {i: float(rows[i]["diameter_um"]) for i in decided}
    for lo, hi in bins:
        cells = []
        for w, s in ((w1, {i: m1[i] == "S" for i in decided1}), (w2, small)):
            ids = [i for i in w if lo <= w[i] < hi]
            cells.append(f"{sum(s[i] for i in ids)}/{len(ids)}" if ids else "-")
        lines.append(f"| {lo}-{hi} | {cells[0]} | {cells[1]} |")
    W = np.concatenate([[w1[i] for i in decided1], [w2[i] for i in decided]])
    S = np.concatenate([small1, small2])
    right, t, above = best_cut(W, S)
    loo = sum(((W[j] >= c) if a else (W[j] < c)) == S[j]
              for j in range(len(S)) for _r, c, a in [best_cut(np.delete(W, j), np.delete(S, j))])
    lines += ["", f"Both grids together ({len(S)} marked, {int(S.sum())} S): the best width cut, S "
              f"{'>=' if above else '<'} {t:.3g} um, gets {right}/{len(S)}, leave-one-out {loo}/{len(S)}; "
              f"marking all one way gets {max(int(S.sum()), len(S) - int(S.sum()))}/{len(S)}."]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
