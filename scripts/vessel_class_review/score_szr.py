"""Grid SZR's marks against each case's first mark (in grid SZ1 or SZ2): how
often Harvey's small vessel / capillary marks repeat, overall, per grid of
origin, per first mark and per width range.

    python scripts/vessel_class_review/score_szr.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "outputs/vessel_class_review/SZR"
REPORT = GRID / "SZR_result.md"
BAND_UM = (6.0, 15.0)


def width_range(w: float) -> str:
    lo, hi = BAND_UM
    return f"under {lo:g} um" if w < lo else f"{lo:g}-{hi:g} um" if w <= hi else f"over {hi:g} um"


def main() -> None:
    marks = {int(r["id"]): r["mark"] for r in csv.DictReader((GRID / "SZR_marks.csv").open(encoding="utf-8"))}
    rows = sorted(csv.DictReader((GRID / "SZR_key.csv").open(encoding="utf-8")), key=lambda r: int(r["id"]))
    cases = [(int(r["id"]), r["origin_grid"], int(r["origin_id"]), r["origin_mark"], marks.get(int(r["id"])),
              float(r["diameter_um"])) for r in rows]
    same = [c for c in cases if c[3] == c[4]]
    lines = ["# Grid SZR result: do the small vessel / capillary marks repeat?", "",
             f"Marked by Harvey on the SZR page, 2026-10-26. The same mark as the first time: "
             f"**{len(same)}/{len(cases)}**.", "",
             "| Cases | Same mark | First S, now C | First C, now S |", "|---|---|---|---|"]

    def row(label, group):
        lines.append(f"| {label} | {sum(c[3] == c[4] for c in group)}/{len(group)} | "
                     f"{sum(c[3] == 'S' and c[4] == 'C' for c in group)} | "
                     f"{sum(c[3] == 'C' and c[4] == 'S' for c in group)} |")

    for grid in ("SZ1", "SZ2"):
        row(f"from {grid}", [c for c in cases if c[1] == grid])
    for mark in ("S", "C"):
        row(f"first marked {mark}", [c for c in cases if c[3] == mark])
    for label in (width_range(BAND_UM[0] - 1), width_range(BAND_UM[0]), width_range(BAND_UM[1] + 1)):
        row(label, [c for c in cases if width_range(c[5]) == label])
    by_grid = [[sum(c[3] == c[4] for c in cases if c[1] == g), sum(c[3] != c[4] for c in cases if c[1] == g)]
               for g in ("SZ1", "SZ2")]
    lines += ["", f"SZ1 against SZ2, same mark or not: Fisher p = {fisher_exact(by_grid).pvalue:.3f}.", "",
              "When each was marked (the pages' stores, 2026-10-26 UTC): SZ1 20:30-20:33, SZ2 22:46-22:48, "
              "SZR 22:53-22:54. SZ2's cases came back 6 minutes after they were marked and SZ1's after two "
              "hours, so SZ2's agreement may be memory as much as judgement.", "",
              "| ID | From | Its ID there | First | Now | Width um |", "|---|---|---|---|---|---|"]
    for c in cases:
        flag = "" if c[3] == c[4] else " **changed**"
        lines.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]}{flag} | {c[5]:.2f} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
