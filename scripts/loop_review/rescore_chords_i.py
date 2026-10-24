#!/usr/bin/env python3
"""Re-score grid I with the raw image read across the loop, not round its hole.

Harvey's marks (27 F, 3 C) were decided on the claudin-5 channel, and the
hole reading (``hole_is_lumen``) called eight false loops walled. What a
person sees in a false loop is that its two sides lie in one wall ring: a
chord from one side to the other crosses no wall. In a real loop the chord
leaves one vessel through its wall, crosses tissue, and enters the other
through its wall. So, for each antipodal chord of the loop (the mask's own
``chords_in_lumen`` test, read from the raw image instead): the wall it
crosses (``wall_between``), the darkest point on it (tissue), and the share of
its length that reads as wall.

    python scripts/loop_review/rescore_chords_i.py --review outputs/loop_review --grid I --graph ... --mask ... --raw ... --raw-channel 1 --voxel-size-zyx ...
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))
sys.path.insert(0, str(HERE.parent / "bridge_review"))


def chord_measures(evidence, points: np.ndarray, radius_um: float) -> dict:
    from haemolynx.preprocessing.raw_evidence import BRIGHT_THRESHOLD
    from haemolynx.graph.lumen_loops import _evenly_round

    ring = _evenly_round(points, evidence._step_um)
    n = len(ring)
    if n < 6:
        return dict(chord_wall_median=np.nan, chord_wall_max=np.nan, chords_walled=np.nan, chord_dark_median=np.nan, chord_bright_share=np.nan, chords=0)
    half = n // 2
    walls, darks, brights = [], [], []
    for i in range(half):
        a, b = ring[i], ring[(i + half) % n]
        if np.linalg.norm(b - a) < 2.0 * radius_um:
            continue
        profile = evidence._chord(a, b)
        if len(profile) < 3:
            continue
        interior = profile[1:-1]
        walls.append(float(evidence.wall_between(a, b)))
        darks.append(float(interior.min()))
        brights.append(float(np.mean(interior >= BRIGHT_THRESHOLD)))
    if not walls:
        return dict(chord_wall_median=np.nan, chord_wall_max=np.nan, chords_walled=np.nan, chord_dark_median=np.nan, chord_bright_share=np.nan, chords=0)
    walls, darks, brights = map(np.asarray, (walls, darks, brights))
    return dict(
        chord_wall_median=round(float(np.median(walls)), 3), chord_wall_max=round(float(walls.max()), 3),
        chords_walled=round(float(np.mean(walls >= 0.3)), 3), chord_dark_median=round(float(np.median(darks)), 3),
        chord_bright_share=round(float(np.median(brights)), 3), chords=int(len(walls)),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--grid", default="I")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--raw-channel", type=int, default=None)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)

    from fit_marks import loo_threshold
    from fit_marks_m import read_marks
    from haemolynx.graph.post_processing import short_loops
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing.raw_evidence import ENDOTHELIAL, calibrate_raw, load_raw_channel, make_raw_evidence

    review = Path(args.review)
    key = {r["id"]: r for r in csv.DictReader(open(review / f"{args.grid}_key.csv", encoding="utf-8"))}
    marks = read_marks(review, args.grid)
    G = pickle.load(open(args.graph, "rb"))
    mask = np.load(args.mask).astype(bool)
    raw = load_raw_channel(args.raw, args.raw_channel)
    vs = tuple(args.voxel_size_zyx)
    _support = MaskSupport(mask, vs)
    evidence = make_raw_evidence(raw, mask.shape, vs, calibrate_raw(raw, mask, vs, stain=ENDOTHELIAL))
    loops = {";".join("-".join(str(x) for x in e) for e in loop.edges): loop for loop in short_loops(G, search_um=200.0)}
    rows = []
    for ident, k in key.items():
        loop = loops.get(k["edges"])
        if loop is None or marks.get(ident, {}).get("verdict") not in ("F", "C"):
            continue
        rows.append((ident, marks[ident]["verdict"], chord_measures(evidence, np.asarray(loop.points_um, float), float(k["lumen_radius_um"]))))
    y = np.asarray([v == "F" for _i, v, _m in rows], dtype=bool)
    lines = [f"## Re-score across the loop ({len(rows)} judged; all one way {max(y.sum(), (~y).sum())}/{len(rows)})", "",
             "| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    for measure in ("chord_wall_median", "chord_wall_max", "chords_walled", "chord_dark_median", "chord_bright_share", "chords"):
        values = np.asarray([float(m[measure]) for _i, _v, m in rows], dtype=float)
        r = loo_threshold(values, y)
        if r:
            lines.append(f"| {measure} | {r['loo']}/{r['n']} | F when {measure} {'>' if r['above'] else '<='} {r['threshold']:.3g} | {r['fit_hits']}/{r['n']} |")
    lines += ["", "| id | mark | wall median | wall max | chords walled | dark median | bright share | chords |", "|---|---|---|---|---|---|---|---|"]
    for i, v, m in rows:
        lines.append(f"| {i} | {v} | {m['chord_wall_median']} | {m['chord_wall_max']} | {m['chords_walled']} | {m['chord_dark_median']} | {m['chord_bright_share']} | {m['chords']} |")
    text = "\n".join(lines) + "\n"
    with open(review / f"FIT_{args.grid}.md", "a", encoding="utf-8") as f:
        f.write("\n" + text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
