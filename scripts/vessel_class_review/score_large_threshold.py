"""Phase 0f of PLAN_vessel_classes_from_segmentation.md: the large-vessel
threshold against E14.5's real large masks.

A vessel is "large" by truth when the run's own overlap rule tagged it in the
arteriole or Vein mask (`large_mask_vessel_type`). Each candidate threshold
calls a vessel large when its width reaches it; scored on the run's final
widths, on its endothelial (raw-image) widths where measured, and on its mask
(EDT) widths. One dataset, a handful of large trees: this screens the
candidates, it fits nothing anyone should ship unchecked.

    python scripts/vessel_class_review/score_large_threshold.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
SOLVED = OUT / "e14p5_solved.pkl"
REPORT = OUT / "phase0f_large_threshold.md"

#: The two candidates the plan names: 2 x the thick-vessel gate's default
#: radius, and Smith et al. 2015's D_min (rat mesentery).
CANDIDATES_UM = (12.0, 16.5)
SWEEP_UM = np.arange(8.0, 30.01, 0.5)
WIDTHS = {
    "final (diameter_um)": "diameter_um",
    "endothelial, where measured": "endothelial_diameter_um",
    "mask EDT (edt_diameter_um)": "edt_diameter_um",
}


def rows(G, attribute):
    out = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("is_zero_resistance") or u == v:
            continue
        w = d.get(attribute)
        if w is None or not np.isfinite(w) or float(w) <= 0:
            continue
        out.append((float(w), d.get("large_mask_vessel_type") in ("arteriole", "venule"),
                    float(d.get("length") or 0.0)))
    return np.array([r[0] for r in out]), np.array([r[1] for r in out]), np.array([r[2] for r in out])


def score(width, large, length, threshold):
    called = width >= threshold
    tp, fp = int((called & large).sum()), int((called & ~large).sum())
    fn, tn = int((~called & large).sum()), int((~called & ~large).sum())
    right_len = float(length[called == large].sum())
    return {
        "threshold": threshold,
        "found": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "right": (tp + tn) / len(width),
        "right_by_length": right_len / max(1e-9, float(length.sum())),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    lines = ["# Phase 0f: the large-vessel threshold against E14.5's large masks", "",
             "Truth: the run's large-mask tag (arteriole or Vein mask). \"Found\" is the share of tagged",
             "vessels at or above the threshold; \"precision\" the share of vessels at or above it that are",
             "tagged; \"right\" counts every vessel; \"by length\" weights each by its length. All one way",
             "(nothing large) is the baseline to beat.", ""]
    for title, attribute in WIDTHS.items():
        width, large, length = rows(G, attribute)
        if not len(width):
            continue
        base = 1.0 - large.mean()
        base_len = float(length[~large].sum() / length.sum())
        lines += [f"## Widths: {title}", "",
                  f"{len(width)} vessels, {int(large.sum())} tagged large. Tagged widths: median "
                  f"{np.median(width[large]):.1f} um (10-90%: {np.percentile(width[large], 10):.1f}-"
                  f"{np.percentile(width[large], 90):.1f}); untagged: median {np.median(width[~large]):.1f} um "
                  f"(90%: {np.percentile(width[~large], 90):.1f}, 99%: {np.percentile(width[~large], 99):.1f}).",
                  f"All one way (nothing large): right {base:.1%}, by length {base_len:.1%}.", "",
                  "| Threshold um | Found | Precision | Right | By length | TP | FP | FN |",
                  "|---|---|---|---|---|---|---|---|"]
        sweep = [score(width, large, length, t) for t in SWEEP_UM]
        best = max(sweep, key=lambda s: (s["right"], s["right_by_length"]))
        shown = {round(t, 1) for t in CANDIDATES_UM} | {round(best["threshold"], 1)}
        for s in sweep + [score(width, large, length, t) for t in CANDIDATES_UM]:
            t = round(s["threshold"], 1)
            if t not in shown:
                continue
            shown.discard(t)
            tag = " (candidate)" if t in CANDIDATES_UM else " (best on this sweep: fitted)"
            lines.append(f"| {t:g}{tag} | {s['found']:.0%} | {s['precision']:.0%} | {s['right']:.1%} | "
                         f"{s['right_by_length']:.1%} | {s['tp']} | {s['fp']} | {s['fn']} |")
        lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
