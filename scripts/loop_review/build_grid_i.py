#!/usr/bin/env python3
"""Grid I of PLAN_raw_confirmation.md: is this short loop real, or false?

Every loop rule so far read the mask (grids A-F of the loop review, 2026-10-08):
whether a gap lies inside the loop and the loop runs through the mask. The raw
image has not been asked. A real loop runs round tissue; a false one closes
inside one vessel (a pinhole in the segmentation, two strands through one
lumen, the wedge at a fork). On an endothelial label (claudin-5, the run's raw
channel since 2026-10-13) the vessels' walls line a real loop's hole, and a
false loop's hole is lumen with no wall between the two sides. This grid shows
30 short loops from one built graph, each drawn in its own plane on the
segmentation with the raw channel beside it, spread by value over the raw
reading of the hole (``RawEvidence.hole_is_lumen``) and shuffled. Marks:
C = a real loop (keep), F = a false loop (remove a side), ? = unsure.

Usage::

    python scripts/loop_review/build_grid_i.py --graph <final.pkl> --mask mask.npy
        --raw <composite.tif> --raw-channel 1 --voxel-size-zyx 1 0.98 0.98
        --out outputs/loop_review
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))
SEED = 20261013
#: Loops longer than this round are not shown: the review list's own limit.
MAX_LOOP_UM = 200.0
#: The hole's middle: interior points this far beyond the wall band.
HOLE_CENTRE_MARGIN_UM = 2.0


def _evenly(items: list, count: int, key) -> list:
    if len(items) <= count:
        return list(items)
    ordered = sorted(items, key=key)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(picks.tolist())]


def _by_value(items: list, count: int, measure: str, tie: str) -> list:
    """*count* candidates spread over *measure* by value: a third each from
    the low, middling and high readings, each third spread over *tie*."""
    bins = (
        [c for c in items if c[measure] <= 0.2],
        [c for c in items if 0.2 < c[measure] < 0.8],
        [c for c in items if c[measure] >= 0.8],
    )
    shares = [count // 3 + (1 if i < count % 3 else 0) for i in range(3)]
    chosen: list = []
    short = 0
    for pool, share in zip(bins, shares):
        picked = _evenly(pool, share + short, key=lambda c: (c[measure], c[tie]))
        short = share + short - len(picked)
        chosen += picked
    if short:
        rest = [c for c in items if c not in chosen]
        chosen += _evenly(rest, short, key=lambda c: (c[measure], c[tie]))
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--raw-channel", type=int, default=None)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="I")
    parser.add_argument("--exclude", action="append", default=[], help="an earlier grid's key: its loops are not shown again")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    from build_grids import Drawer
    from haemolynx.graph.lumen_loops import loop_measures
    from haemolynx.graph.post_processing import short_loops
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing.raw_evidence import (
        BRIGHT_THRESHOLD, ENDOTHELIAL, _along, _tangents, calibrate_raw, load_raw_channel, make_raw_evidence,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    mask = np.load(args.mask).astype(bool)
    raw = load_raw_channel(args.raw, args.raw_channel)
    if raw.shape != mask.shape:
        raise SystemExit(f"raw image {raw.shape} and mask {mask.shape} differ")
    vs = tuple(args.voxel_size_zyx)
    support = MaskSupport(mask, vs)
    calibration = calibrate_raw(raw, mask, vs, stain=ENDOTHELIAL)
    if calibration is None:
        raise SystemExit("The raw channel could not be calibrated as an endothelial label")
    print("calibration:", calibration.describe())
    evidence = make_raw_evidence(raw, mask.shape, vs, calibration)
    shown = set()
    for path in args.exclude:
        shown |= {row["edges"] for row in csv.DictReader(open(path, encoding="utf-8"))}

    candidates = []
    for loop in short_loops(G, search_um=MAX_LOOP_UM):
        points = np.asarray(loop.points_um, dtype=float)
        edges_text = ";".join("-".join(str(x) for x in e) for e in loop.edges)
        if len(points) < 3 or edges_text in shown:
            continue
        radii = support.radius(points)
        radius = float(max(np.median(radii[radii > 0]) if (radii > 0).any() else 1.0, 1.0))
        hole = evidence.hole_is_lumen(points, radius)
        interior, distance = evidence.hole_interior(points, radius)
        centre_points = interior[distance > radius + 2.0 * HOLE_CENTRE_MARGIN_UM] if len(interior) else interior
        samples = _along(np.vstack([points, points[:1]]), evidence._step_um)
        loop_wall = evidence.vesselness(samples, radius, _tangents(samples))
        measures = loop_measures(points, mask=mask, voxel_size_zyx=vs)
        candidates.append(dict(
            edges=edges_text, n_edges=len(loop.edges), n_sides=len(loop.sides),
            loop_length_um=round(float(loop.length_um), 2), lumen_radius_um=round(radius, 2),
            centre_zyx_vox=tuple(int(round(c)) for c in np.asarray(loop.centre_um) / np.asarray(vs)),
            centre_zyx_um=tuple(round(float(c), 1) for c in loop.centre_um),
            span_um=round(float(measures.get("span_um") or 0.0), 2),
            area_um2=round(float(measures.get("area_um2") or 0.0), 1),
            out_of_plane=round(float(measures.get("out_of_plane") or 0.0), 3),
            path_in=round(float(measures.get("path_in") or 0.0), 3),
            chords_in_lumen=round(float(measures.get("chords_in_lumen") or 0.0), 3),
            chord_bg_median_um=round(float(measures.get("chord_bg_median_um") or 0.0), 2),
            inside_one_lumen=bool(measures.get("inside_one_lumen")),
            raw_hole_lumen=round(float(hole), 3) if np.isfinite(hole) else float("nan"),
            raw_hole_centre=round(float(np.median(evidence.brightness(centre_points))), 3) if len(centre_points) else float("nan"),
            raw_loop_wall=round(float(np.median(loop_wall)), 3),
            raw_loop_wall_min=round(float(loop_wall.min()), 3),
            raw_loop_wall_bright=round(float(np.mean(loop_wall >= BRIGHT_THRESHOLD)), 3),
            hole_points=int(len(interior)),
            points=points,
        ))
    judgeable = [c for c in candidates if np.isfinite(c["raw_hole_lumen"])]
    print(f"{len(candidates)} short loops, {len(judgeable)} with a hole wide enough to read")
    chosen = _by_value(judgeable, 30, "raw_hole_lumen", "raw_hole_centre")
    random.Random(args.seed).shuffle(chosen)

    drawer = Drawer(G, mask, raw.astype(np.float32), vs)
    template = (HERE.parent / "dead_end_review" / "template.html").read_text(encoding="utf-8")
    examples, rows = [], []
    for number, c in enumerate(chosen, start=1):
        ident = f"{args.grid}-{number:02d}"
        points = c["points"]
        closed = np.vstack([points, points[:1]])
        drawn = drawer.draw(points, None, [dict(points=closed, color="#ff3b30", width=2.8)],
                            half_thickness=float(np.clip(2 * c["lumen_radius_um"], 2.0, 5.0)), marker=None)
        examples.append({"id": ident, **drawn})
        rows.append({"id": ident, "grid": args.grid, **{k: v for k, v in c.items() if k != "points"}})
    data = {
        "batch": args.grid,
        "title": f"Loop Review {args.grid}: real loop, or false?",
        "question": "Is this loop a real loop round tissue, or a false loop inside one vessel?",
        "lede": ("A short loop of the network (red), drawn in its own plane on the segmentation. Toggle 'Raw image' "
                 "for the claudin-5 (endothelial) channel: a vessel is a bright wall round a dark lumen. A real loop "
                 "runs round tissue, so its vessels' walls line the hole; a false loop closes inside one vessel (a "
                 "pinhole in the segmentation, two strands through one lumen, the wedge at a fork), so its hole is "
                 "lumen with no wall between the two sides."),
        "steps": ["Look at the loop and its hole on the segmentation, then on the raw image.",
                  "C if it is a real loop round tissue: keep it.",
                  "F if it is a false loop inside one vessel: a side should go. Draw which side, if you can.",
                  "Add a note on what decided it."],
        "legend": [{"label": "the loop judged", "color": "#ff3b30"}, {"label": "other centrelines", "color": "#9a9a9a"},
                   {"label": "your drawing", "color": "#ffd60a"}],
        "verdicts": [{"value": "C", "label": "A real loop: keep", "short": "C", "tone": "good"},
                     {"value": "F", "label": "A false loop: remove a side", "short": "F", "tone": "bad"},
                     {"value": "?", "label": "Unsure", "short": "?", "tone": "unsure"}],
        "examples": examples,
    }
    html = (template.replace("__TITLE__", f"Loop Review {args.grid}")
            .replace("__SECOND_PANEL__", "Raw image: claudin-5 (endothelial) channel")
            .replace("__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    (out / f"{args.grid}_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / f"{args.grid}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"{args.grid}_candidates.csv", "w", newline="", encoding="utf-8") as f:
        allrows = [{k: v for k, v in c.items() if k != "points"} for c in candidates]
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in allrows for k in r)))
        writer.writeheader()
        writer.writerows(allrows)
    print(f"{args.grid}: {len(examples)} examples -> {out / (args.grid + '_review.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
