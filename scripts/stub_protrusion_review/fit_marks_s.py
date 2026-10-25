#!/usr/bin/env python3
"""Grid S's marks against its key (``build_grid_s.py``): K (a vessel, keep) or
R (not a vessel, remove) for 30 dead ends the stub prune's protrusion test
decides on E14.5.

Scored, each as right / wrong against marking all one way:

- the rules fixed before the marks (:data:`FIXED_RULES`): today's protrusion
  test, the uncapped reading (16 nearest samples of other centrelines), the
  uncapped reading with the two exemptions ``PLAN_one_centreline_per_lumen.md``
  named (a stub running straight on from a vessel at its junction, a stub at
  an image face), and what the prune call itself did;
- every key measure as one threshold, fitted and leave-one-out
  (``dead_end_review.fit_marks.loo_threshold``);
- every pair of measures as "keep when either is past its threshold",
  leave-one-out.

    python scripts/stub_protrusion_review/fit_marks_s.py --key outputs/stub_protrusion_review/key.csv \\
        --marks outputs/stub_protrusion_review/marks.csv --out outputs/stub_protrusion_review/FIT_S.md
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dead_end_review"))

#: The reading a stub never judged by today's test carries (its 16 nearest
#: samples all its own): it is kept, as one far outside every lumen is.
NEVER_JUDGED = 99.0
COLLINEAR_DEG = 30.0


def _number(text) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.nan
    return NEVER_JUDGED if math.isinf(value) else value


def _flag(text) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


def measures(row: dict) -> dict[str, float]:
    """The key row's measures, and three it implies."""
    out = {name: _number(row.get(name)) for name in (
        "ratio_capped", "ratio_others", "deviation_deg", "face_um", "length_um", "tip_radius_um",
        "parent_radius_um", "threshold_um", "junction_degree", "chain_edges",
    )}
    out["length_over_tip_radius"] = out["length_um"] / out["tip_radius_um"] if out["tip_radius_um"] > 0 else math.nan
    out["length_over_threshold"] = out["length_um"] / out["threshold_um"] if out["threshold_um"] > 0 else math.nan
    out["tip_over_parent_radius"] = (
        out["tip_radius_um"] / out["parent_radius_um"] if out["parent_radius_um"] > 0 else math.nan
    )
    return out


def _removes_uncapped(row: dict) -> bool:
    return _number(row["ratio_others"]) < 1.0


#: Rule name -> (what it is, row -> True when it removes the stub).
FIXED_RULES = {
    "today": ("today's test: today's reading under 1 tip radius", lambda r: _number(r["ratio_capped"]) < 1.0),
    "uncapped": ("the uncapped reading under 1 tip radius", _removes_uncapped),
    "uncapped_but_collinear": (
        f"the uncapped reading, a stub within {COLLINEAR_DEG:g} deg of running straight on kept",
        lambda r: _removes_uncapped(r) and not _number(r["deviation_deg"]) < COLLINEAR_DEG,
    ),
    "uncapped_but_face": (
        "the uncapped reading, a stub within its threshold of a face kept",
        lambda r: _removes_uncapped(r) and not _flag(r["at_face"]),
    ),
    "uncapped_but_both": (
        "the uncapped reading, both exemptions",
        lambda r: _removes_uncapped(r) and not _number(r["deviation_deg"]) < COLLINEAR_DEG
        and not _flag(r["at_face"]),
    ),
    "the_prune": ("what the prune call did (removed by any of its rules)", lambda r: _flag(r["removed_by_call"])),
}


def score(rows: list[dict], marks: dict[str, str], removes) -> tuple[int, list[str]]:
    """(right, the IDs it gets wrong) for a rule *removes* (row -> bool),
    against marks R (remove) and K (keep); other marks are left out."""
    right, wrong = 0, []
    for row in rows:
        mark = marks.get(row["id"])
        if mark not in ("K", "R"):
            continue
        if bool(removes(row)) == (mark == "R"):
            right += 1
        else:
            wrong.append(row["id"])
    return right, wrong


def pair_loo(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> tuple[int, int, tuple]:
    """"Keep when a > s or b > t": (leave-one-out hits, fitted hits, (s, t))."""
    def cuts(v):
        u = np.unique(v[np.isfinite(v)])
        return np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])

    def fit(idx):
        best = (-1, None)
        for s in cuts(a[idx]):
            for t in cuts(b[idx]):
                hits = int(np.sum(((a[idx] > s) | (b[idx] > t)) == keep[idx]))
                if hits > best[0]:
                    best = (hits, (s, t))
        return best

    every = np.arange(len(keep))
    loo = 0
    for i in every:
        _h, (s, t) = fit(every[every != i])
        loo += int(((a[i] > s) or (b[i] > t)) == keep[i])
    hits, cut = fit(every)
    return loo, hits, cut


def report(rows: list[dict], marks: dict[str, str]) -> str:
    from fit_marks import loo_threshold

    judged = [r for r in rows if marks.get(r["id"]) in ("K", "R")]
    n = len(judged)
    keeps = sum(marks[r["id"]] == "K" for r in judged)
    baseline = max(keeps, n - keeps)
    lines = [
        "# Grid S fitted", "",
        f"{n} marked: {keeps} K (keep, a vessel), {n - keeps} R (remove). Marking all one way: {baseline}/{n}.", "",
        "## Rules fixed before the marks", "", "| rule | right | wrong |", "|---|---|---|",
    ]
    for name, (what, removes) in FIXED_RULES.items():
        right, wrong = score(judged, marks, removes)
        lines.append(f"| {what} | {right}/{n} | {', '.join(wrong)} |")
    table = {r["id"]: measures(r) for r in judged}
    labels = np.array([marks[r["id"]] == "K" for r in judged])
    names = list(next(iter(table.values())))
    columns = {name: np.array([table[r["id"]][name] for r in judged], float) for name in names}
    fits = []
    for name in names:
        fitted = loo_threshold(columns[name], labels)
        if fitted is not None:
            fits.append((fitted["loo"], name, fitted))
    lines += ["", "## One measure, one threshold (K side; leave-one-out, then fitted on all)", "",
              "| measure | leave-one-out | rule | fitted |", "|---|---|---|---|"]
    for loo, name, fitted in sorted(fits, key=lambda x: (-x[0], x[1])):
        side = ">" if fitted["above"] else "<="
        lines.append(f"| {name} | {loo}/{fitted['n']} | K when {side} {fitted['threshold']:.3g} | "
                     f"{fitted['fit_hits']}/{fitted['n']} |")
    pairs = []
    complete = [name for name in names if np.isfinite(columns[name]).all()]
    for x, y in itertools.combinations(complete, 2):
        loo, hits, (s, t) = pair_loo(columns[x], columns[y], labels)
        pairs.append((loo, hits, x, y, s, t))
    lines += ["", "## Two measures: K when either is above its threshold (best ten, leave-one-out)", "",
              "| measures | leave-one-out | fitted | thresholds |", "|---|---|---|---|"]
    for loo, hits, x, y, s, t in sorted(pairs, key=lambda p: (-p[0], -p[1]))[:10]:
        lines.append(f"| {x} or {y} | {loo}/{n} | {hits}/{n} | {s:.3g}, {t:.3g} |")
    return "\n".join(lines) + "\n"


def read_marks(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {row["id"]: (row.get("mark") or "").strip().upper() for row in csv.DictReader(f)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--key", required=True)
    parser.add_argument("--marks", required=True, help="a CSV with id and mark (K, R or ?)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.key, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    text = report(rows, read_marks(Path(args.marks)))
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
