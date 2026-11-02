"""Grid SZ1 (Phase 0f of PLAN_vessel_classes_from_segmentation.md): about 30
E14.5 vessels 5-15 um wide, for Harvey to mark small vessel, capillary or
can't tell, so the small-vessel threshold is set by his marks.

Three vessels per 1 um band of the run's final widths (wall to wall), none in
the large masks, none shorter than 10 um, none within 40 um of another case,
then shuffled. The widths and every other measure stay in the key.

    python scripts/vessel_class_review/grid_sz1.py
"""
from __future__ import annotations

import csv
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
GRID = OUT / "SZ1"
SOLVED = OUT / "e14p5_solved.pkl"

BAND_EDGES_UM = np.arange(5.0, 15.01, 1.0)
PER_BAND = 3
MIN_LENGTH_UM = 10.0
MIN_SPACING_UM = 40.0
SEED = 20261027
MEASURED = ("endothelial", "edt_mask")


def candidates(G) -> list[dict]:
    from measures import BRIDGE, curvature_sd, polyline

    out = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get(BRIDGE) or u == v or d.get("large_mask_vessel_type"):
            continue
        w = d.get("diameter_um")
        if w is None or not np.isfinite(w) or not BAND_EDGES_UM[0] <= w < BAND_EDGES_UM[-1]:
            continue
        if d.get("diameter_source") not in MEASURED:
            continue
        length = float(d.get("length") or 0.0)
        if length < MIN_LENGTH_UM:
            continue
        pts = polyline(G, u, v, d)
        neighbours = [float(e.get("diameter_um")) for node in (u, v)
                      for _a, _b, _k, e in G.edges(node, keys=True, data=True)
                      if (_a, _b, _k) not in ((u, v, k), (v, u, k)) and not e.get(BRIDGE)
                      and e.get("diameter_um") is not None and np.isfinite(e.get("diameter_um"))]
        mid = pts[len(pts) // 2]
        out.append({
            "u": u, "v": v, "key": k,
            "diameter_um": round(float(w), 3),
            "diameter_source": d.get("diameter_source"),
            "endothelial_diameter_um": d.get("endothelial_diameter_um") or "",
            "edt_diameter_um": d.get("edt_diameter_um") or "",
            "length_um": round(length, 2),
            "length_over_diameter": round(length / float(w), 3),
            "ksd_per_um": round(curvature_sd(pts), 4) if len(pts) > 3 else "",
            "branch_order": d.get("branch_order") or "",
            "end_degrees": f"{G.degree(u)} {G.degree(v)}",
            "widest_neighbour_um": round(max(neighbours), 3) if neighbours else "",
            "ratio_to_widest_neighbour": round(float(w) / max(neighbours), 3) if neighbours else "",
            "depth_um": round(float(mid[0]), 1),
            "_mid": mid,
        })
    return out


def choose(pool: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    bands = [[] for _ in range(len(BAND_EDGES_UM) - 1)]
    for c in pool:
        bands[int(np.digitize(c["diameter_um"], BAND_EDGES_UM)) - 1].append(c)
    chosen, used_nodes = [], set()
    for band in bands:
        rng.shuffle(band)
        taken = 0
        for c in band:
            if taken >= PER_BAND:
                break
            if {c["u"], c["v"]} & used_nodes:
                continue
            if any(math.dist(c["_mid"], o["_mid"]) < MIN_SPACING_UM for o in chosen):
                continue
            chosen.append(c)
            used_nodes |= {c["u"], c["v"]}
            taken += 1
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")

    from case_panels import load_scene, save_single

    (GRID / "cases").mkdir(parents=True, exist_ok=True)
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    scene = load_scene(G)
    large = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("large_mask_vessel_type"):
            large |= {(u, v, k), (v, u, k)}
    pool = candidates(G)
    chosen = choose(pool)
    rows = []
    for case_id, c in enumerate(chosen, start=1):
        key = (c["u"], c["v"], c["key"])
        where = save_single(case_id, key, scene, large, GRID / "cases" / f"SZ1_{case_id:02d}.png")
        rows.append({"id": case_id, **where, **{k: v for k, v in c.items() if not k.startswith("_")}})
    with (GRID / "SZ1_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bands = Counter(int(r["diameter_um"]) for r in rows)
    print(f"pool {len(pool)}; chosen {len(rows)}; per 1-um band {dict(sorted(bands.items()))}; "
          f"sources {Counter(r['diameter_source'] for r in rows)}")


if __name__ == "__main__":
    main()
