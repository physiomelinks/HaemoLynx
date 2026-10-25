#!/usr/bin/env python3
"""The Phase 5 grid's cases (``build_grid.py``) as the interactive page's data,
for ``scripts/thick_junction_review/grid_page.py``: each case still to mark
drawn as the sheets draw it (``phase1_rules_grid.case_view``), on one sheet
with the grid's question and its four marks.

    python scripts/thick_bridge_review/page_data.py \\
        --capture outputs/tb/after/finish_input.pkl --graph outputs/tb/after/e14_graph.pkl \\
        --grid outputs/thick_bridge_review --out outputs/thick_bridge_review/page.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "thick_junction_review"))

#: The page's marks, as the sheets' header has them.
MARKS = (
    ("B", "A bridge: the big vessel's radius between its centreline and a vessel opening into it"),
    ("V", "Vessel: keeps its resistance"),
    ("C", "The big vessel's own centreline"),
    ("N", "Should not be connected at all"),
    ("?", "Unsure"),
)
QUESTION = "What should the red piece be?"

#: The axis each view looks down (``build_grid.PLANES``).
AXIS_OF_VIEW = {"down z": 0, "down y": 1, "down x": 2}


def unmarked_ids(grid: Path) -> list[int]:
    """The grid's IDs with no mark in its ``marks.csv`` (every ID without one)."""
    cases = json.loads((grid / "cases.json").read_text(encoding="utf-8"))
    marked = set()
    marks = grid / "marks.csv"
    if marks.exists():
        with open(marks, newline="", encoding="utf-8") as f:
            marked = {int(r["id"]) for r in csv.DictReader(f) if r.get("mark", "").strip()}
    return sorted(int(c["id"]) for c in cases if int(c["id"]) not in marked)


def page_case(case: dict) -> dict:
    """A ``cases.json`` entry as ``case_view`` takes it."""
    return {
        "id": int(case["id"]), "rule": "bridge", "vox": tuple(int(v) for v in case["vox"]),
        "view": case["view"], "axis": AXIS_OF_VIEW[case["view"]],
        "centre": np.asarray(case["centre_um"], float), "piece": np.asarray(case["piece_um"], float),
    }


def page_data(cases: list[dict], ids, capture: dict, G) -> dict:
    """The page's JSON: one sheet of *ids*, drawn from *capture*'s volumes and *G*."""
    from build_grid import _edge_path
    from haemolynx.graph import IS_ZERO_RESISTANCE
    from phase1_rules_grid import case_view

    vs = np.asarray(capture["voxel_size_zyx"], float)
    mask = np.asarray(capture["binary_mask"], dtype=bool)
    region = np.asarray(capture["thick_vessel_mask"], dtype=bool)
    tree_points = np.asarray(capture["thick_tree"], float) * vs
    paths = [(_edge_path(G, u, v, d), bool(d.get(IS_ZERO_RESISTANCE))) for u, v, d in G.edges(data=True)]
    by_id = {int(c["id"]): c for c in cases}
    views = [case_view(page_case(by_id[i]), mask, region, tree_points, paths, vs) for i in ids]
    return {
        "grid": "TB",
        "dataset": "E14.5 MCA, Harvey's Claudin segmentation",
        "voxel_size_zyx": [float(v) for v in vs],
        "slab_um": 12.0,
        "sheets": [{
            "rule": "bridge", "title": "Bridges and the pieces beside them", "question": QUESTION,
            "marks": [{"key": k, "label": label} for k, label in MARKS],
            "cases": views,
        }],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--grid", required=True, help="build_grid.py's folder: cases.json and marks.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    grid = Path(args.grid)
    cases = json.loads((grid / "cases.json").read_text(encoding="utf-8"))
    ids = unmarked_ids(grid)
    with open(args.capture, "rb") as f:
        capture = pickle.load(f)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    data = page_data(cases, ids, capture, G)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{len(ids)} cases ({ids[0]}-{ids[-1]}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
