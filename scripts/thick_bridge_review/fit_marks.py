#!/usr/bin/env python3
"""Fit Harvey's marks on the Phase 5 grid (``build_grid.py``) of
PLAN_thick_vessel_bridges.md: what each case measures, and how well the
current rule and each single-threshold rule on those measures agree with
the marks, against marking everything one way, with leave-one-out.

Marks: B = a bridge, V = vessel, C = the thick vessel's own centreline,
N = should not be connected at all.

    python scripts/thick_bridge_review/fit_marks.py \\
        --capture outputs/tb/after/finish_input.pkl --graph outputs/tb/after/e14_graph.pkl \\
        --grid outputs/thick_bridge_review

writes ``measures.csv`` and ``FIT.md`` into the grid's folder.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path

import numpy as np

#: The current rule's answer, at the level that moves a resistance: a bridge
#: gives none, a near-miss keeps its own (as vessel or as thick centreline).
RULE_AGREES = {"bridge": {"B"}, "near_miss": {"V", "C"}}

#: The measures a single threshold is tried on, per kind of case.
MEASURES = {
    "bridge": (
        "length_over_radius", "distance_over_tolerance", "share_on_centreline", "axis_cos", "length_um",
        "beyond_in_region", "beyond_dist_over_r", "beyond_axis_cos", "beyond_over_thick", "outer_over_thick",
    ),
    "near_miss": (
        "length_over_radius", "distance_over_tolerance", "dist_over_r_med", "axis_cos", "length_um",
        "inside_region_share",
    ),
}

QUESTIONS = (("is a bridge (B)", {"B"}), ("should not be connected (N)", {"N"}), ("is thick centreline (C)", {"C"}))


def best_threshold(x: np.ndarray, y: np.ndarray):
    """The single threshold on *x* (cases at or under it, or over it, called
    True) that gets most of the booleans *y* right: ``(right, threshold,
    direction)``, threshold None when marking all one way does as well.
    Thresholds lie midway between neighbouring values, so a case left out
    falls on the side its neighbours put it."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    best = (int(max(y.sum(), (~y).sum())), None, "all one way")
    values = np.unique(x)
    for t in (values[:-1] + values[1:]) / 2.0:
        for direction, called in (("<=", x <= t), (">", x > t)):
            right = int((called == y).sum())
            if right > best[0]:
                best = (right, float(t), direction)
    return best


def leave_one_out(x: np.ndarray, y: np.ndarray) -> int:
    """How many of *y* a threshold fitted without each case gets right on it."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    right = 0
    for i in range(len(x)):
        rest = np.ones(len(x), dtype=bool)
        rest[i] = False
        _r, t, direction = best_threshold(x[rest], y[rest])
        if t is None:
            called = bool(y[rest].sum() * 2 > rest.sum())
        else:
            called = bool(x[i] <= t) if direction == "<=" else bool(x[i] > t)
        right += int(called == bool(y[i]))
    return right


def _cos(a, b) -> float:
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def measure(cases, key, capture, G):
    """Each case's measures beyond the key's: its angle to the thick vessel's
    axis, and for a bridge, where the vessel beyond its outer end runs and how
    wide it is against the thick vessel."""
    from scipy.spatial import cKDTree

    from haemolynx.graph import IS_ZERO_RESISTANCE, diagnose_thick_vessel_bridges
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points, points_inside_mask
    from haemolynx.graph.thick_vessel_junctions import _inscribed_radius_um

    vs = np.asarray(capture["voxel_size_zyx"], float)
    tree_points = np.asarray(capture["thick_tree"], float) * vs
    tree = cKDTree(tree_points)
    radius = np.nan_to_num(np.asarray(capture["thick_tree_radius"], float), posinf=0.0)
    thick, mask = capture["thick_vessel_mask"], capture["binary_mask"]
    report = diagnose_thick_vessel_bridges(
        G, thick, capture["thick_tree"], mask, voxel_size_zyx=tuple(vs), tree_radius_um=capture["thick_tree_radius"],
    )
    ends = {frozenset((r["u"], r["v"])): (r["inner_node"], r["outer_node"]) for r in report["bridges"]}
    pos = {n: np.asarray(d["pos"], float) for n, d in G.nodes(data=True)}

    def axis_at(index, reach):
        near = tree_points[tree.query_ball_point(tree_points[index], reach)]
        if len(near) < 3:
            return None
        return np.linalg.svd(near - near.mean(axis=0), full_matrices=False)[2][0]

    rows = []
    for row in key:
        case = cases[int(row["id"])]
        piece = np.asarray(case["piece_um"], float)
        out = dict(row)
        if row["kind"] == "bridge":
            u, v, _k = ast.literal_eval(row["edge"])
            inner, outer = ends[frozenset((u, v))]
            index = int(tree.query(pos[inner])[1])
            thick_r = float(radius[index])
            axis = axis_at(index, max(thick_r, 6.0))
            out["axis_cos"] = _cos(pos[outer] - pos[inner], axis) if axis is not None else np.nan
            out["outer_over_thick"] = float(_inscribed_radius_um(pos[outer][None], mask, vs)[0]) / max(thick_r, 1e-9)
            beyond = None
            for a, b, d in G.edges(outer, data=True):
                other = b if a == outer else a
                if d.get(IS_ZERO_RESISTANCE) or other == outer:
                    continue
                path = densify_polyline(edge_sample_points(outer, other, d, pos), max_step_um=0.5)
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                first = path[arc <= 10.0]
                ring = path[(arc >= 3.0) & (arc <= 13.0)]
                distance, nearest = tree.query(first)
                found = {
                    "beyond_len": float(arc[-1]),
                    "beyond_in_region": float(points_inside_mask(first, thick, voxel_size_zyx=tuple(vs)).mean()),
                    "beyond_dist_over_r": float(np.median(distance / np.maximum(radius[nearest], 1e-9))),
                    "beyond_axis_cos": _cos(first[-1] - first[0], axis) if axis is not None and len(first) > 1 else np.nan,
                    "beyond_over_thick": (
                        float(np.median(_inscribed_radius_um(ring, mask, vs))) / max(thick_r, 1e-9) if len(ring) else np.nan
                    ),
                }
                if beyond is None or found["beyond_len"] > beyond["beyond_len"]:
                    beyond = found
            out.update(beyond or {})
        else:
            index = int(tree.query(piece[len(piece) // 2])[1])
            axis = axis_at(index, max(float(radius[index]), 6.0))
            out["axis_cos"] = _cos(piece[-1] - piece[0], axis) if axis is not None else np.nan
            distance, nearest = tree.query(piece)
            out["dist_over_r_med"] = float(np.median(distance / np.maximum(radius[nearest], 1e-9)))
        rows.append(out)
    return rows


def fit(rows) -> str:
    """FIT.md's text: the current rule against the marks, then every
    question's best single threshold per measure, by kind of case."""
    marked = [r for r in rows if r.get("mark")]
    lines = ["# Thick-vessel bridge grid: the fit", ""]
    agree = [r for r in marked if r["mark"] in RULE_AGREES[r["kind"]]]
    counts = {m: sum(r["mark"] == m for r in marked) for m in "BVCN"}
    best_mark = max(counts, key=counts.get)
    keep = sum(r["mark"] in ("V", "C") for r in marked)
    lines += [
        f"{len(marked)} cases marked. Marks: " + ", ".join(f"{m} {c}" for m, c in counts.items()) + ".",
        "",
        f"- The current rule agrees on **{len(agree)}/{len(marked)}** (a bridge marked B, a near-miss marked V or C);"
        f" wrong on IDs {', '.join(str(r['id']) for r in marked if r not in agree)}.",
        f"- Marking all one way: all {best_mark} {counts[best_mark]}/{len(marked)}; no bridges at all"
        f" (every piece keeps its resistance) {keep}/{len(marked)}.",
        "",
    ]
    for kind, label in (("bridge", "Bridges"), ("near_miss", "Near-misses")):
        group = [r for r in marked if r["kind"] == kind]
        if not group:
            continue
        lines += [f"## {label} ({len(group)})", "",
                  "| question | all one way | best measure | right | leave-one-out |", "|---|---|---|---|---|"]
        for question, target in QUESTIONS:
            y = np.array([r["mark"] in target for r in group])
            results = []
            for name in MEASURES[kind]:
                x = np.array([float(r.get(name, np.nan)) if r.get(name, "") != "" else np.nan for r in group])
                if np.isnan(x).all():
                    continue
                right, t, direction = best_threshold(x, y)
                if t is not None:
                    results.append((right, leave_one_out(x, y), name, direction, t, int((~np.isnan(x)).sum())))
            base = int(max(y.sum(), (~y).sum()))
            if results:
                right, loo, name, direction, t, n = max(results)
                lines.append(f"| {question} | {base}/{len(y)} | {name} {direction} {t:.2f} | {right}/{n} | {loo}/{n} |")
            else:
                lines.append(f"| {question} | {base}/{len(y)} | none beats it | | |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--grid", required=True, help="the grid's folder: key.csv, cases.json, marks.csv")
    args = parser.parse_args(argv)
    grid = Path(args.grid)
    with open(grid / "key.csv", newline="", encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    with open(grid / "marks.csv", newline="", encoding="utf-8") as f:
        marks = {int(r["id"]): r["mark"].strip().upper() for r in csv.DictReader(f) if r["mark"].strip()}
    cases = {int(c["id"]): c for c in json.loads((grid / "cases.json").read_text(encoding="utf-8"))}
    for row in key:
        row["mark"] = marks.get(int(row["id"]), "")
    with open(args.capture, "rb") as f:
        capture = pickle.load(f)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    rows = measure(cases, key, capture, G)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(grid / "measures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = fit(rows)
    (grid / "FIT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
