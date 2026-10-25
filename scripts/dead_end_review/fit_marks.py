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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Which marks each grid's rule separates, as (positive, negative). G1b-G3b
#: are the fresh grids G1'-G3' (PLAN_dead_ends.md Phase 2), asked the same.
QUESTIONS = {
    "G1": ("X", "S"), "G2": ("F", "I"), "G3": ("J", "N"), "G4": ("N", "V"),
    "G1b": ("X", "S"), "G2b": ("F", "I"), "G3b": ("J", "N"),
}
MEANING = {
    "G1": "X = not a vessel end (remove) against S = a real blind end",
    "G2": "F = leaves the image against I = ends inside it",
    "G3": "J = join (same vessel) against N = not the same vessel",
    "G4": "N = not a vessel (remove) against V = a vessel (P, partly, reported apart)",
}
MEANING.update({f"{grid}b": MEANING[grid] for grid in ("G1", "G2", "G3")})


@dataclass(frozen=True)
class Rule:
    """One rule fitted on G1-G3 (FIT.md, 2026-10-12), fixed before the fresh
    grid that checks it was built: it says *positive* when *measure* is
    *above* (or not above) *threshold*; a measure that is unset says the
    negative. A *threshold* of None reads *measure* as a string equal to
    *equals*."""

    name: str
    measure: str
    threshold: float | None = None
    above: bool = True
    equals: str | None = None

    def says_positive(self, row: dict) -> bool:
        value = row.get(self.measure)
        if self.threshold is None:
            return value == self.equals
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(number):
            return False
        return number > self.threshold if self.above else number <= self.threshold


#: The rules each fresh grid checks; the first is the one the grid was spread
#: across. The bar for relying on one is 90% (``graph.8b``).
FRESH_RULES = {
    "G1b": (Rule("beside_share > 0.0104 is X", "beside_share", 0.0104),),
    "G2b": (
        Rule("the classifier's leaves_image is F", "kind", equals="leaves_image"),
        Rule("face_distance_um <= 11.3 is F", "face_distance_um", 11.3, above=False),
        Rule("depth_um <= 10.4 is F", "depth_um", 10.4, above=False),
    ),
    "G3b": (Rule("ahead_turn_deg <= 30.5 is J", "ahead_turn_deg", 30.5, above=False),),
}


def check_rules(grid: str, rows: list[dict], marks: dict[str, dict]) -> list[dict]:
    """Each of *grid*'s :data:`FRESH_RULES` scored on the marks, unfitted:
    ``{"rule", "right", "judged", "baseline", "wrong"}`` -- how many judged
    examples it gets right, against marking all of them one way, and the IDs
    it gets wrong."""
    positive, negative = QUESTIONS[grid]
    judged = [r for r in rows if (marks.get(r["id"]) or {}).get("verdict") in (positive, negative)]
    truth = [marks[r["id"]]["verdict"] == positive for r in judged]
    baseline = max(sum(truth), len(truth) - sum(truth))
    results = []
    for rule in FRESH_RULES.get(grid, ()):
        said = [rule.says_positive(r) for r in judged]
        wrong = [r["id"] for r, s, t in zip(judged, said, truth) if s != t]
        results.append({"rule": rule.name, "right": len(judged) - len(wrong), "judged": len(judged),
                        "baseline": baseline, "wrong": wrong})
    return results


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
    parser.add_argument("--raw-channel", type=int, help="the raw image's channel, for a multi-channel file (z c y x)")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--grids", nargs="*", help="only these grids (each grid's keys name nodes of the graph it was built from)")
    args = parser.parse_args(argv)
    import tifffile

    review = Path(args.review)
    G = pickle.load(open(args.graph, "rb"))
    if args.mask.endswith(".npy"):
        mask = np.load(args.mask).astype(bool)
    else:
        from haemolynx.io.load import _to_binary_volume_for_skeletonization

        mask = np.asarray(_to_binary_volume_for_skeletonization(tifffile.imread(args.mask)), dtype=bool)
    raw = tifffile.imread(args.raw) if args.raw else None
    if raw is not None and args.raw_channel is not None:
        raw = raw[:, args.raw_channel]
    raw = raw.astype(np.float32) if raw is not None else None
    vs = tuple(args.voxel_size_zyx)
    report = ["# Dead-end marks: the rules fitted", ""]
    for grid, (positive, negative) in QUESTIONS.items():
        if args.grids and grid not in args.grids:
            continue
        if not (review / f"{grid}_key.csv").exists() or not (review / f"marks_{grid}").exists():
            continue
        rows = list(csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8")))
        marks = read_marks(review / f"marks_{grid}")
        extras = extra_measures(G, rows, mask, raw, vs)
        table = []
        for row in rows:
            mark = marks.get(row["id"], {})
            verdict = mark.get("verdict")
            table.append({**row, **extras.get(row["id"], {}), "mark": verdict, "note": mark.get("note", ""),
                          "evidence": ";".join(mark.get("evidence") or [])})
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
        checked = check_rules(grid, rows, marks)
        if checked:
            report += ["", "Rules fixed before this grid was built, checked (bar: 90%):", "",
                       "| rule | right | all one way | wrong IDs |", "|---|---|---|---|"]
            report += [f"| {c['rule']} | {c['right']}/{c['judged']} | {c['baseline']}/{c['judged']} | "
                       f"{', '.join(c['wrong']) or '-'} |" for c in checked]
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
    name = "FIT.md" if not args.grids else f"FIT_{'_'.join(args.grids)}.md"
    (review / name).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
