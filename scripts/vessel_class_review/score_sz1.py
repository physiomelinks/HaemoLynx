"""Grid SZ1's marks against every measure in its key: the small-vessel
threshold (Phase 0f of PLAN_vessel_classes_from_segmentation.md).

For each measure, the one threshold (either direction) that gets most marked
vessels right, against marking every one as the larger class, then the same
threshold chosen leaving each vessel out in turn (leave-one-out), and the IDs
the full fit still gets wrong. "?" and unmarked cases are left out.

    python scripts/vessel_class_review/score_sz1.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "outputs/vessel_class_review/SZ1"
REPORT = GRID / "SZ1_result.md"

MEASURES = ("diameter_um", "endothelial_diameter_um", "edt_diameter_um", "ratio_to_widest_neighbour",
            "widest_neighbour_um", "length_um", "length_over_diameter", "ksd_per_um", "depth_um", "b_order")


def b_order(label: str) -> float:
    """B05 -> 5; anything else (Art, Ven, Large, none) -> nan."""
    return float(label[1:]) if label.startswith("B") and label[1:].isdigit() else float("nan")


def best_cut(x: np.ndarray, small: np.ndarray):
    """(right, threshold, small_above) of the best single cut on *x*."""
    best = (-1, None, True)
    values = np.unique(x)
    cuts = np.concatenate([[values[0] - 1], (values[:-1] + values[1:]) / 2, [values[-1] + 1]])
    for t in cuts:
        for above in (True, False):
            called = x >= t if above else x < t
            right = int((called == small).sum())
            if right > best[0]:
                best = (right, float(t), above)
    return best


def main() -> None:
    marks = {int(r["id"]): r["mark"] for r in csv.DictReader((GRID / "SZ1_marks.csv").open(encoding="utf-8"))}
    rows = {int(r["id"]): r for r in csv.DictReader((GRID / "SZ1_key.csv").open(encoding="utf-8"))}
    for r in rows.values():
        r["b_order"] = b_order(r["branch_order"])
    decided = sorted(i for i, m in marks.items() if m in ("S", "C"))
    small = np.array([marks[i] == "S" for i in decided])
    counts = Counter(marks.get(i, "unmarked") for i in rows)
    one_way = max(int(small.sum()), int((~small).sum()))
    lines = ["# Grid SZ1 result", "",
             f"Marked by Harvey on the SZ1 page, 2026-10-26: {dict(counts)}. Scored on the {len(decided)} "
             f"marked S or C; marking all S gets {int(small.sum())}/{len(decided)}.", "",
             "| Measure | Vessels | Best cut | Right | Leave-one-out | Wrong (full fit) |",
             "|---|---|---|---|---|---|"]
    for name in MEASURES:
        ids, x = [], []
        for i in decided:
            value = rows[i].get(name)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                ids.append(i)
                x.append(value)
        if len(ids) < 5:
            continue
        ids = np.array(ids)
        x = np.array(x)
        y = np.array([marks[i] == "S" for i in ids])
        right, t, above = best_cut(x, y)
        loo = 0
        for k in range(len(ids)):
            keep = np.arange(len(ids)) != k
            _r, tk, ak = best_cut(x[keep], y[keep])
            loo += int(((x[k] >= tk) if ak else (x[k] < tk)) == y[k])
        called = x >= t if above else x < t
        wrong = ", ".join(f"{i}{'S' if s else 'C'}" for i, s, c in zip(ids, y, called) if c != s)
        base = max(int(y.sum()), int((~y).sum()))
        lines.append(f"| {name} | {len(ids)} (all one way {base}) | small {'>=' if above else '<'} {t:.2f} | "
                     f"{right}/{len(ids)} | {loo}/{len(ids)} | {wrong or '-'} |")
    lines += ["", "Each case, by width:", "", "| ID | Mark | Width um | Source | Ratio to widest neighbour | Branch order |",
              "|---|---|---|---|---|---|"]
    for i in sorted(rows, key=lambda i: float(rows[i]["diameter_um"])):
        r = rows[i]
        lines.append(f"| {i} | {marks.get(i, '-')} | {float(r['diameter_um']):.2f} | {r['diameter_source']} | "
                     f"{r['ratio_to_widest_neighbour']} | {r['branch_order']} |")
    lines.append(f"\nAll one way: {one_way}/{len(decided)}.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
