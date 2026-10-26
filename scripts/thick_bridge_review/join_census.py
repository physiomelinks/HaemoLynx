#!/usr/bin/env python3
"""A census grid of the thin-arm join's paths (``preprocessing.thick_vessels.
_join_thin_arms_to_fat_ridge``; PLAN_thick_vessel_bridges.md, "connections
that should not exist"; ``docs/notes/thick_vessels.md``).

Every path the join drew that is still a connection in the saved graph and
passes nowhere near a case of an earlier grid (``--exclude``: its
``cases.json``), shuffled into IDs J01, J02, ..., each drawn as the Phase 5
grid draws its cases with the join's path as the red piece, on one
interactive page (``thick_junction_review/grid_page.py``) for Harvey's marks.
No rule is fixed: no measure separated the marks on the Phase 5 grid, so the
key keeps every measure for one to be fitted to these marks afterwards.

Marks: J = joined here (a vessel opening into the big one), C = the big
vessel's own centreline, N = should not be connected at all, ? = unsure.

    python scripts/thick_bridge_review/join_census.py --run <capture> \\
        --exclude outputs/thick_bridge_review --out outputs/join_census_review

``--run`` is what ``capture_steps.py`` writes (``mask.npz`` among it). Writes
``key.csv``, ``cases.json``, ``marks_template.csv``, ``page.json`` and
``grid_J.html`` into ``--out``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "thick_junction_review"))

SEED = 20261026
#: The grid's name: its page title, its IDs' letter and where its marks are kept.
GRID = "J"
#: A path is fresh when no point of it lies within this (um) of an earlier grid's piece.
FRESH_GAP_UM = 3.0
#: A path is still a connection when the saved graph joins its two ends (each
#: within STILL_REACH_UM) by a way no longer than STILL_SLACK times the path
#: plus STILL_EXTRA_UM: the split replaces the path by a straight bridge, and
#: may move the bridge's inner end along the centreline by up to a lumen radius.
STILL_REACH_UM = 3.0
STILL_SLACK = 2.0
STILL_EXTRA_UM = 6.0

MARKS = (
    ("J", "Joined here: a vessel opening into the big one"),
    ("C", "The big vessel's own centreline"),
    ("N", "Should not be connected at all"),
    ("?", "Unsure"),
)
QUESTION = "Should the red path join this vessel to the big vessel's centreline here?"
TITLE = "The thin-arm join's paths"

KEY_FIELDS = (
    "id", "join", "vox", "centre_um", "view", "start_vox", "end_vox", "path_um", "chord_um",
    "approach_cos", "start_degree", "arm_voxels", "other_way_um", "final_bridge_share",
    "earlier_grid_gap_um",
)


def course_um(path: Mapping[str, Any], voxel_size_zyx: Sequence[float]) -> np.ndarray:
    """A join path in um, from its arm's voxel to its tree voxel."""
    from thick_bridge_review.connection_steps import densify

    spacing = np.asarray(voxel_size_zyx, float)
    voxels = np.vstack([path["start"], path["path"], path["end"]]).astype(float)
    return densify(voxels * spacing)


def still_connected(final, course: np.ndarray) -> bool:
    """Whether the saved graph (a ``connection_steps.StepIndex``) still joins
    *course*'s ends (see :data:`STILL_SLACK`)."""
    from thick_bridge_review.connection_steps import polyline_length

    limit = STILL_SLACK * polyline_length(course) + STILL_EXTRA_UM
    local = final.local(course[len(course) // 2])
    return local.path_um(course[0], course[-1], reach_um=STILL_REACH_UM, cutoff_um=limit) <= limit


def carrying_edge(final, course: np.ndarray, reach_um: float = 2.0) -> tuple | None:
    """The saved graph's edge running along most of *course*."""
    local = final.local(course[len(course) // 2])
    best, best_share = None, 0.0
    for e in local.edge_ids:
        share = float((cKDTree(local.index.polylines[e]).query(course, distance_upper_bound=reach_um)[0]
                       <= reach_um).mean())
        if share > best_share:
            best, best_share = local.index.edges[e], share
    return best


def other_way_um(G: nx.MultiGraph, edge: tuple | None) -> float:
    """The shortest way between *edge*'s two nodes through *G* without it
    (inf when it is the only link): how long a loop the connection closes."""
    if edge is None:
        return float("nan")
    u, v, k = edge
    if u == v:
        return 0.0
    H = nx.Graph()
    for a, b, key, data in G.edges(keys=True, data=True):
        if (a, b, key) in ((u, v, k), (v, u, k)) or a == b:
            continue
        weight = float(data.get("length") or 0.0)
        if not H.has_edge(a, b) or H.edges[a, b]["length"] > weight:
            H.add_edge(a, b, length=weight)
    try:
        return float(nx.shortest_path_length(H, u, v, weight="length"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def census_cases(paths: Sequence[Mapping[str, Any]], lee: np.ndarray, final, voxel_size_zyx: Sequence[float],
                 tree_points: np.ndarray, earlier_points: np.ndarray | None, *, seed: int = SEED
                 ) -> list[dict[str, Any]]:
    """The join's *paths* still a connection in *final* and fresh of the
    earlier grid, measured, shuffled and given IDs."""
    from build_grid import PLANES, _plane
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    from join_paths import approach
    from thick_bridge_review.connection_steps import polyline_length

    spacing = np.asarray(voxel_size_zyx, float)
    earlier = cKDTree(earlier_points) if earlier_points is not None and len(earlier_points) else None
    picked = []
    for index, path in enumerate(paths, start=1):
        course = course_um(path, spacing)
        gap = float(earlier.query(course)[0].min()) if earlier is not None else np.inf
        if gap <= FRESH_GAP_UM or not still_connected(final, course):
            continue
        local = final.local(course[len(course) // 2])
        bridge_share = float(local.covered_by(course, lambda d: bool(d.get(IS_ZERO_RESISTANCE)), 2.0).mean())
        centre = course[len(course) // 2]
        case = {"join": index, "piece": course, "centre": centre}
        axis = _plane(case, tree_points)
        case.update({
            "axis": axis, "view": PLANES[axis][0], "rule": "join",
            "vox": tuple(int(v) for v in np.round(centre / spacing)),
            "start_vox": tuple(int(v) for v in path["start"]), "end_vox": tuple(int(v) for v in path["end"]),
            "path_um": round(polyline_length(course), 2),
            "other_way_um": other_way_um(final.G, carrying_edge(final, course)),
            "final_bridge_share": round(bridge_share, 2), "earlier_grid_gap_um": round(gap, 2),
        })
        case.update(approach(path, lee, spacing))
        picked.append(case)
    random.Random(seed).shuffle(picked)
    width = max(2, len(str(len(picked))))
    for number, case in enumerate(picked, start=1):
        case["id"] = f"{GRID}{number:0{width}d}"
    return picked


def key_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        row = {name: case.get(name) for name in KEY_FIELDS}
        row["centre_um"] = tuple(round(float(v), 2) for v in case["centre"])
        for name in ("approach_cos", "chord_um", "other_way_um"):
            value = row[name]
            row[name] = value if value is None or not np.isfinite(value) else round(float(value), 3)
        rows.append(row)
    return rows


def page_data(cases: Sequence[Mapping[str, Any]], mask, region, tree_points, G, voxel_size_zyx) -> dict:
    """The page's JSON: one sheet of *cases*, drawn as the Phase 5 grid draws."""
    from build_grid import _edge_path
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    from phase1_rules_grid import case_view

    vs = np.asarray(voxel_size_zyx, float)
    paths = [(_edge_path(G, u, v, d), bool(d.get(IS_ZERO_RESISTANCE))) for u, v, d in G.edges(data=True)]
    views = [case_view(case, mask, region, tree_points, paths, vs) for case in cases]
    return {
        "grid": GRID,
        "dataset": "E14.5 MCA, Harvey's Claudin segmentation",
        "voxel_size_zyx": [float(v) for v in vs],
        "slab_um": 12.0,
        "sheets": [{
            "rule": "join", "title": TITLE, "question": QUESTION,
            "marks": [{"key": k, "label": label} for k, label in MARKS],
            "cases": views,
        }],
    }


def main(argv: list[str] | None = None) -> int:
    from grid_page import build_page
    from join_paths import join_paths
    from thick_bridge_review.connection_steps import StepIndex, _unpacked, densify

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="what capture_steps.py wrote")
    parser.add_argument("--exclude", action="append", default=[], help="an earlier grid's folder (cases.json)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = np.asarray(json.loads((run / "voxel_size_zyx.json").read_text()), float)
    before = _unpacked(run / "skel_before_join.npz")
    after = _unpacked(run / "skel_after_join.npz")
    tree_voxels = np.load(run / "tree.npy")
    tree = np.zeros_like(before)
    tree[tuple(tree_voxels.T)] = True
    with open(run / "graph_built.pkl", "rb") as f:
        G = pickle.load(f)
    earlier = [densify(c["piece_um"]) for folder in args.exclude
               for c in json.loads((Path(folder) / "cases.json").read_text(encoding="utf-8"))]
    tree_points = tree_voxels.astype(float) * vs
    cases = census_cases(join_paths(before, after, tree), before & ~tree, StepIndex(G), vs, tree_points,
                         np.vstack(earlier) if earlier else None)
    rows = key_rows(cases)
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "cases.json").write_text(json.dumps(
        [dict(row, piece_um=np.round(case["piece"], 3).tolist()) for row, case in zip(rows, cases)], default=str),
        encoding="utf-8")
    (out / "marks_template.csv").write_text("id,mark\n" + "".join(f"{c['id']},\n" for c in cases), encoding="utf-8")
    data = page_data(cases, _unpacked(run / "mask.npz"), _unpacked(run / "thick.npz"), tree_points, G, vs)
    (out / "page.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    page = build_page(data)
    (out / f"grid_{GRID}.html").write_text(page, encoding="utf-8")
    print(f"{len(cases)} join paths -> {out} (page {len(page) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
