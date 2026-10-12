#!/usr/bin/env python3
"""Fit the dead-end rules to the marks made on the G1-G3 grids (CLAUDE.md,
"Marked example grids"): join each grid's marks to its key, add measures
computed from the graph, mask and raw image, and score every one-measure
threshold rule by leave-one-out, against marking everything one way.

Usage::

    python scripts/dead_end_review/fit_marks.py --review outputs/dead_end_review \\
        --graph final.pkl --mask mask.npy --raw raw.tif --voxel-size-zyx 1 0.98 0.98
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

#: Which marks each grid's rule separates, as (positive, negative).
QUESTIONS = {"G1": ("X", "S"), "G2": ("F", "I"), "G3": ("J", "N")}
MEANING = {
    "G1": "X = not a vessel end (remove) against S = a real blind end",
    "G2": "F = leaves the image against I = ends inside it",
    "G3": "J = join (same vessel) against N = not the same vessel",
}


def read_marks(folder: Path) -> dict[str, dict]:
    marks = {}
    for path in sorted(folder.glob("annotations/*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        data = body.get("data", body)
        marks[data.get("id", path.stem)] = data
    return marks


def extra_measures(G, rows, mask, raw, vs):
    """Measures the marks' notes point at, per key row (by tip node)."""
    from haemolynx.graph.dead_ends import chain_path, chain_to_junction
    from haemolynx.preprocessing import MaskSupport

    support = MaskSupport(mask, vs)
    out = {}
    for row in rows:
        tip = row["tip"]
        try:
            node = eval(tip, {}, {}) if isinstance(tip, str) else tip
        except Exception:
            node = tip
        if node not in G:
            continue
        _nodes, edges = chain_to_junction(G, node)
        path = chain_path(G, edges, step_um=0.5)
        if len(path) < 2:
            continue
        inside = np.asarray(support.inside(path), dtype=bool)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        last5 = arc <= 5.0
        values = {
            "tip_in_mask": float(inside[0]),
            "last5_in_mask": float(inside[last5].mean()),
            "off_mask_um": float(0.5 * np.count_nonzero(~inside)),
            "chain_over_tip_radius": float(arc[-1] / max(float(support.radius(path[:1])[0]), 0.5)),
            "tip_radius_vs_chain_median": float(
                support.radius(path[:1])[0] / max(np.median(support.radius(path)), 0.5)
            ),
        }
        if raw is not None:
            from scipy import ndimage

            samples = ndimage.map_coordinates(raw, (path / np.asarray(vs)).T, order=1)
            far = samples[arc > 5.0] if np.any(arc > 5.0) else samples
            values["raw_tip_over_chain"] = float(samples[last5].mean() / max(far.mean(), 1e-6))
        out[row["id"]] = values
    return out


def loo_threshold(values: np.ndarray, labels: np.ndarray):
    """Best ``value > t`` or ``value <= t`` rule: its leave-one-out hits, and the
    rule fitted on all of them."""
    def fit(v, y):
        best = (-1, None, None)
        cuts = np.unique(v)
        cuts = np.concatenate([[cuts[0] - 1], (cuts[:-1] + cuts[1:]) / 2, [cuts[-1] + 1]])
        for t in cuts:
            for above in (True, False):
                pred = v > t if above else v <= t
                hits = int(np.sum(pred == y))
                if hits > best[0]:
                    best = (hits, t, above)
        return best

    ok = np.isfinite(values)
    v, y = values[ok], labels[ok]
    if len(v) < 4 or len(np.unique(y)) < 2:
        return None
    loo = 0
    for i in range(len(v)):
        keep = np.arange(len(v)) != i
        _h, t, above = fit(v[keep], y[keep])
        pred = v[i] > t if above else v[i] <= t
        loo += int(pred == y[i])
    hits, t, above = fit(v, y)
    return {"n": len(v), "loo": loo, "fit_hits": hits, "threshold": float(t), "above": above}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    import tifffile

    review = Path(args.review)
    G = pickle.load(open(args.graph, "rb"))
    mask = np.load(args.mask).astype(bool)
    raw = tifffile.imread(args.raw).astype(np.float32) if args.raw else None
    vs = tuple(args.voxel_size_zyx)
    report = ["# Dead-end marks: the rules fitted", ""]
    for grid, (positive, negative) in QUESTIONS.items():
        rows = list(csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8")))
        marks = read_marks(review / f"marks_{grid}")
        extras = extra_measures(G, rows, mask, raw, vs)
        table = []
        for row in rows:
            mark = marks.get(row["id"], {})
            verdict = mark.get("verdict")
            table.append({**row, **extras.get(row["id"], {}), "mark": verdict, "note": mark.get("note", "")})
        keys = list(dict.fromkeys(k for r in table for k in r))
        with open(review / f"{grid}_marked.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
        judged = [r for r in table if r["mark"] in (positive, negative)]
        labels = np.asarray([r["mark"] == positive for r in judged])
        counts = {m: sum(1 for r in table if r["mark"] == m) for m in sorted({r["mark"] for r in table if r["mark"]})}
        baseline = max(labels.sum(), (~labels).sum())
        report += [f"## {grid}: {MEANING[grid]}", "",
                   f"Marks: {counts}; judged {len(judged)}; all one way gets {baseline}/{len(judged)}.", ""]
        # The classifier's own kinds, as they stand.
        kinds = {}
        for r in judged:
            kinds.setdefault(r["kind"], []).append(r["mark"])
        report += ["| kind (classifier) | marks |", "|---|---|"]
        report += [f"| {k} | {', '.join(f'{m}:{v.count(m)}' for m in sorted(set(v)))} |" for k, v in sorted(kinds.items())]
        report += ["", "| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
        results = []
        for key in keys:
            try:
                values = np.asarray([float(r[key]) if r.get(key) not in (None, "", "None") else np.nan for r in judged])
            except (TypeError, ValueError):
                continue
            if np.all(~np.isfinite(values)):
                continue
            fitted = loo_threshold(values, labels)
            if fitted:
                results.append((fitted["loo"], key, fitted))
        for loo, key, fitted in sorted(results, key=lambda x: -x[0])[:12]:
            rule = f"{positive} when {key} {'>' if fitted['above'] else '<='} {fitted['threshold']:.3g}"
            report.append(f"| {key} | {loo}/{fitted['n']} | {rule} | {fitted['fit_hits']}/{fitted['n']} |")
        report.append("")
    text = "\n".join(report)
    (review / "FIT.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
