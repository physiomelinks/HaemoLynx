"""Grid SZR: twenty cases from grids SZ1 and SZ2 shown again, under new
numbers and shuffled, to find how often Harvey's small vessel / capillary
marks repeat.

Within 6-15 um SZ1 was marked mostly small vessel (16/23) and SZ2 mostly
capillary (5/20). Marked again in one sitting, cases that keep their marks
while SZ1's stay S and SZ2's stay C say the samples differ; marks that change
say the judgement moved. So most cases come from that band, half of each
grid's marked S and half C, spread over width, with one under the band from
each grid and SZ2's over-band vessels represented by one. Ten were S and ten C.
The key keeps each case's grid, ID and mark; the page shows none of them.

    python scripts/vessel_class_review/grid_szr.py
"""
from __future__ import annotations

import csv
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
GRID = OUT / "SZR"
SOLVED = OUT / "e14p5_solved.pkl"

SEED = 20261029
BAND_UM = (6.0, 15.0)
#: (grid, widths um, mark, how many)
PICKS = (
    ("SZ1", BAND_UM, "S", 4), ("SZ1", BAND_UM, "C", 4), ("SZ1", (0.0, 6.0), "S", 1), ("SZ1", (0.0, 6.0), "C", 1),
    ("SZ2", BAND_UM, "S", 4), ("SZ2", BAND_UM, "C", 4), ("SZ2", (0.0, 6.0), "C", 1), ("SZ2", (15.0, 99.0), "S", 1),
)


def marked_cases(grid: str) -> list[dict]:
    marks = {r["id"]: r["mark"] for r in csv.DictReader((OUT / grid / f"{grid}_marks.csv").open(encoding="utf-8"))}
    rows = []
    for r in csv.DictReader((OUT / grid / f"{grid}_key.csv").open(encoding="utf-8")):
        if marks.get(r["id"]) in ("S", "C"):
            rows.append({"origin_grid": grid, "origin_id": int(r["id"]), "origin_mark": marks[r["id"]],
                         **{k: v for k, v in r.items() if k != "id"}, "diameter_um": float(r["diameter_um"])})
    return rows


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")

    from case_panels import edge_key, load_scene, save_single
    from grid_sz2 import spread

    rng = random.Random(SEED)
    pools = {grid: marked_cases(grid) for grid in ("SZ1", "SZ2")}
    chosen = []
    for grid, (lo, hi), mark, n in PICKS:
        members = [c for c in pools[grid] if c["origin_mark"] == mark and lo <= c["diameter_um"] < hi]
        got = spread(members, n, rng)[:n]
        if len(got) < n:
            raise SystemExit(f"{grid} {lo}-{hi} um {mark}: {len(members)} cases, {n} wanted")
        chosen += got
    rng.shuffle(chosen)

    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    large = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("large_mask_vessel_type"):
            large |= {(u, v, k), (v, u, k)}
    scene = load_scene(G)
    (GRID / "cases").mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id, c in enumerate(chosen, start=1):
        where = save_single(case_id, edge_key(c), scene, large, GRID / "cases" / f"SZR_{case_id:02d}.png")
        # The same vessel drawn the same way as when it was first marked.
        for name, value in where.items():
            if str(value) != str(c[name]):
                raise SystemExit(f"{c['origin_grid']} {c['origin_id']}: {name} {value} now, {c[name]} then")
        rows.append({"id": case_id, **c})
    with (GRID / "SZR_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases: {Counter((r['origin_grid'], r['origin_mark']) for r in rows)}; widths "
          f"{min(r['diameter_um'] for r in rows):.1f}-{max(r['diameter_um'] for r in rows):.1f} um")


if __name__ == "__main__":
    main()
