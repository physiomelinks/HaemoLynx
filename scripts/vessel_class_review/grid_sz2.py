"""Grid SZ2: the fresh check of Harvey's small-vessel rule (6-15 um, grown
from the large vessels through vessels in that band), fixed before any SZ2
mark is seen, as CLAUDE.md's marked grids ask of a rule moved after a grid.

Thirty E14.5 vessels not in grid SZ1, none within 40 um of an SZ1 case or of
each other, none in the large masks, at least 10 um long, from four groups the
rule tells apart: those it calls small, those in the band it does not reach,
those under the band and those over it. Each group is spread over width, then
all are shuffled. The rule's call stays in the key.

    python scripts/vessel_class_review/grid_sz2.py
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
GRID = OUT / "SZ2"
SOLVED = OUT / "e14p5_solved.pkl"
SZ1_KEY = OUT / "SZ1" / "SZ1_key.csv"

SEED = 20261028
MIN_SPACING_UM = 40.0
#: (group, widths um, how many): what the rule calls each.
GROUPS = (
    ("small", (6.0, 15.0), 12),
    ("band_unreached", (6.0, 15.0), 8),
    ("under_band", (4.0, 6.0), 5),
    ("over_band", (15.0, 19.0), 5),
)


def spread(members: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to *n* members in order of preference spread evenly over width."""
    members = sorted(members, key=lambda c: c["diameter_um"])
    if len(members) <= n:
        return members
    chunks = []
    for slot in range(n):
        lo, hi = len(members) * slot // n, len(members) * (slot + 1) // n
        chunk = members[lo:hi]
        rng.shuffle(chunk)
        chunks.append(chunk)
    # One from each width slot in turn, so the first n picks span the widths.
    return [chunk[i] for i in range(max(map(len, chunks))) for chunk in chunks if i < len(chunk)]


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")

    from case_panels import edge_key, load_scene, save_single
    from grid_sz1 import candidates
    from small_vessel_rule import SMALL_MAX_UM, SMALL_MIN_UM, small_vessels

    assert (SMALL_MIN_UM, SMALL_MAX_UM) == (6.0, 15.0), "SZ2 checks the 6-15 um rule"
    (GRID / "cases").mkdir(parents=True, exist_ok=True)
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    strict = small_vessels(G, through_wider=False)
    wider = small_vessels(G, through_wider=True)
    sz1 = [r for r in csv.DictReader(SZ1_KEY.open(encoding="utf-8"))]
    sz1_edges = {(frozenset(edge_key(r)[:2]), edge_key(r)[2]) for r in sz1}
    sz1_nodes = {n for r in sz1 for n in edge_key(r)[:2]}

    import grid_sz1

    # The SZ1 pool, widened to every width the four groups take.
    grid_sz1.BAND_EDGES_UM = np.array([4.0, 19.0])
    pool = candidates(G)
    taken_mid = []
    for r in sz1:
        u, v, k = edge_key(r)
        from measures import polyline

        pts = polyline(G, u, v, G.edges[(u, v, k)])
        taken_mid.append(pts[len(pts) // 2])
    by_group = {name: [] for name, _w, _n in GROUPS}
    for c in pool:
        e = (frozenset((c["u"], c["v"])), c["key"])
        if e in sz1_edges or {c["u"], c["v"]} & sz1_nodes:
            continue
        w = c["diameter_um"]
        c["rule"] = "small" if e in strict else "not small"
        c["through_wider_rule"] = "small" if e in wider else "not small"
        if SMALL_MIN_UM <= w <= SMALL_MAX_UM:
            by_group["small" if e in strict else "band_unreached"].append(c)
        elif 4.0 <= w < SMALL_MIN_UM:
            by_group["under_band"].append(c)
        elif SMALL_MAX_UM < w < 19.0:
            by_group["over_band"].append(c)

    rng = random.Random(SEED)
    chosen, used_nodes = [], set()
    for name, _widths, n in GROUPS:
        got = 0
        for c in spread(by_group[name], n, rng):
            if got >= n:
                break
            if {c["u"], c["v"]} & used_nodes:
                continue
            if any(math.dist(c["_mid"], m) < MIN_SPACING_UM for m in taken_mid):
                continue
            c["group"] = name
            chosen.append(c)
            used_nodes |= {c["u"], c["v"]}
            taken_mid.append(c["_mid"])
            got += 1
    rng.shuffle(chosen)

    large = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("large_mask_vessel_type"):
            large |= {(u, v, k), (v, u, k)}
    scene = load_scene(G)
    rows = []
    for case_id, c in enumerate(chosen, start=1):
        key = (c["u"], c["v"], c["key"])
        where = save_single(case_id, key, scene, large, GRID / "cases" / f"SZ2_{case_id:02d}.png")
        rows.append({"id": case_id, **where, **{k: v for k, v in c.items() if not k.startswith("_")}})
    with (GRID / "SZ2_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"pool by group {({k: len(v) for k, v in by_group.items()})}; chosen {Counter(r['group'] for r in rows)}; "
          f"rule calls {Counter(r['rule'] for r in rows)}; widths {min(r['diameter_um'] for r in rows):.1f}-"
          f"{max(r['diameter_um'] for r in rows):.1f} um")


if __name__ == "__main__":
    main()
