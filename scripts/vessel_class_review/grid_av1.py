"""Grid AV1 (Phase 0c of PLAN_vessel_classes_from_segmentation.md): about 30
E14.5 vessels at or above the capillary cut-off, for Harvey to mark arteriole,
venule or can't tell.

Spread across diameter and length over diameter, at most 3 vessels from one
tree, then shuffled,
so an ID gives nothing away. The measures stay in the key, off the image.

    python scripts/vessel_class_review/grid_av1.py
"""
from __future__ import annotations

import csv
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
GRID = OUT / "AV1"
SOLVED = OUT / "e14p5_solved.pkl"
VESSELS_CSV = OUT / "e14p5_vessels.csv"

N_CASES = 30
PER_TREE = 3
MIN_LENGTH_UM = 10.0
SEED = 20261026
CASES_PER_SHEET = 10
#: Diameter strata (um, the run's wall-to-wall widths).
DIAMETER_EDGES_UM = (12.0, 18.0)
QUESTION = ("Is the RED vessel an ARTERIOLE (A), a VENULE (V), or CAN'T TELL (?)")
LEGEND = ("Red: the vessel. Blue: the rest of its tree (vessels >= 8.4 um wide joined to it). "
          "Grey lines: other centrelines.\nPanels: the whole image's centrelines seen down z (yellow box: "
          "the window), then the segmentation, dextran and claudin-5 in the window,\neach projected "
          "through the vessel's own thinnest axis over the depth it spans.")


def _bins(values, edges):
    return np.digitize(np.asarray(values, float), edges)


def choose(rows: list[dict]) -> list[dict]:
    """Up to N_CASES rows spread over pair width ratio x length/diameter,
    PER_TREE at most from one tree."""
    rng = random.Random(SEED)
    pool = [r for r in rows if float(r["length_um"]) >= MIN_LENGTH_UM]
    ld = np.array([float(r["length_over_diameter"]) for r in pool])
    ld_edges = np.nanquantile(ld, [1 / 3, 2 / 3])
    # Spread over diameter and length over diameter: on E14.5 only one vessel
    # had a partner of the other class beside it (phase0b_report.md), so the
    # pair width ratio would spread over nothing.
    cells = defaultdict(list)
    for r in pool:
        cells[(int(_bins([float(r["diameter_um"])], DIAMETER_EDGES_UM)[0]),
               int(_bins([float(r["length_over_diameter"])], ld_edges)[0]))].append(r)
    for members in cells.values():
        rng.shuffle(members)
    chosen, per_tree = [], Counter()
    keys = sorted(cells)
    while len(chosen) < N_CASES and any(cells[k] for k in keys):
        for k in keys:
            while cells[k]:
                r = cells[k].pop()
                if per_tree[r["tree"]] < PER_TREE:
                    chosen.append(r)
                    per_tree[r["tree"]] += 1
                    break
            if len(chosen) >= N_CASES:
                break
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from case_panels import draw_case, edge_key, load_scene, save_single

    GRID.mkdir(parents=True, exist_ok=True)
    (GRID / "cases").mkdir(exist_ok=True)
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    with VESSELS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scene = load_scene(G)
    tree_keys = defaultdict(set)
    for r in rows:
        u, v, k = edge_key(r)
        tree_keys[r["tree"]] |= {(u, v, k), (v, u, k)}

    chosen = choose(rows)
    key_rows = []
    for sheet_no in range(0, len(chosen), CASES_PER_SHEET):
        cases = chosen[sheet_no:sheet_no + CASES_PER_SHEET]
        fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4.1 * len(cases) + 1.2))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Grid AV1 (E14.5 MCA), sheet {sheet_no // CASES_PER_SHEET + 1}: {QUESTION}\n{LEGEND}",
                     fontsize=11, y=0.999)
        for row_no, r in enumerate(cases):
            case_id = sheet_no + row_no + 1
            key, tree = edge_key(r), tree_keys[r["tree"]]
            where = draw_case(axes[row_no], case_id, key, scene, tree)
            save_single(case_id, key, scene, tree, GRID / "cases" / f"AV1_{case_id:02d}.png")
            key_rows.append({"sheet": sheet_no // CASES_PER_SHEET + 1, "id": case_id, **where, **r})
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = GRID / f"AV1_sheet{sheet_no // CASES_PER_SHEET + 1}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(f"wrote {path}", flush=True)
    with (GRID / "AV1_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    print(f"{len(chosen)} cases; trees {Counter(r['tree'] for r in chosen).most_common(5)}; "
          f"truth {Counter(r['truth'] or 'none' for r in chosen)}")


if __name__ == "__main__":
    main()
