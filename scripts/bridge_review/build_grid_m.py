#!/usr/bin/env python3
"""Grid M of PLAN_raw_confirmation.md: does the vessel continue across this gap?

Every rule that joins two pieces of network across background -- the graph
bridges, the facing-dead-end join, the tip-to-side join the dead-end plan
holds back -- reads only the mask: a gap of background under 2 um is a
dropout, more is tissue. The raw image can say what is in the gap. This grid
shows 30 candidates from one built E14.5 graph: pairs of dead ends within
reach of each other (joined or refused, and why), dead ends with another
vessel straight ahead, and the bridges the build drew across background.
Each is drawn in its own plane on the segmentation, with the raw image
beside it (the endothelial channel, claudin-5: a vessel is a bright wall
ring round a dark lumen, Harvey's choice of 2026-10-13), spread over the raw
image's own reading of the gap and shuffled. Marks: V = one vessel the
segmentation broke (join it), N = two vessels or nothing (leave it), ? =
unsure.

Usage::

    python scripts/bridge_review/build_grid_m.py --graph <final.pkl> --mask mask.npy
        --raw <composite.tif> --raw-channel 1 --voxel-size-zyx 1 0.98 0.98
        --out outputs/bridge_review
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
sys.path.insert(0, str(HERE.parent / "doubled_run_review"))
SEED = 20261013
#: How far a pair of dead ends, or a vessel ahead, may be apart to be shown.
SEARCH_GAP_UM = 15.0
SHARES = {"facing": 14, "ahead": 12, "bridge": 4}


def _chord_measures(evidence, support, path_um: np.ndarray) -> dict:
    from haemolynx.graph.facing_ends import _background

    from haemolynx.preprocessing.raw_evidence import BRIGHT_THRESHOLD, _along, _tangents

    radii = support.radius(np.asarray([path_um[0], path_um[-1]]))
    radius = float(max(radii.max(), 1.0))
    signal = evidence.signal_along(path_um, radius)
    stretches, background_um = _background(path_um, support)
    # The gap itself: the chord's samples that lie outside the mask, where
    # the mask rules see nothing and the raw image is the only reading.
    samples = _along(path_um, evidence._step_um)
    vesselness = evidence.vesselness(samples, radius, _tangents(samples))
    outside = ~np.asarray(support.inside(samples), dtype=bool)
    gap_bright = float(np.mean(vesselness[outside] >= BRIGHT_THRESHOLD)) if outside.any() else 1.0
    gap_min = float(vesselness[outside].min()) if outside.any() else float(vesselness.min())
    return dict(
        chord_um=round(float(np.sum(np.linalg.norm(np.diff(path_um, axis=0), axis=1))), 2),
        end_radius_um=round(radius, 2),
        mask_background_stretches=stretches,
        mask_background_um=round(background_um, 2),
        raw_gap_bright=round(gap_bright, 3),
        raw_gap_min=round(gap_min, 3),
        raw_fraction_bright=round(signal.fraction_bright, 3),
        raw_longest_dark_um=round(signal.longest_dark_um, 2),
        raw_wall_between=round(float(evidence.wall_between(path_um[0], path_um[-1])), 3),
        raw_samples=signal.samples,
    )


def _evenly(items: list, count: int, key) -> list:
    if len(items) <= count:
        return list(items)
    ordered = sorted(items, key=key)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(picks.tolist())]


def _by_value(items: list, count: int) -> list:
    """*count* candidates spread over the gap reading by value, not by rank:
    a third each from dark gaps, middling ones and bright ones (on E14.5 two
    thirds of all candidates read fully bright, so a rank spread shows
    mostly those), the bright third spread over its dimmest sample."""
    bins = (
        [c for c in items if c["raw_gap_bright"] <= 0.2],
        [c for c in items if 0.2 < c["raw_gap_bright"] < 0.8],
        [c for c in items if c["raw_gap_bright"] >= 0.8],
    )
    shares = [count // 3 + (1 if i < count % 3 else 0) for i in range(3)]
    chosen: list = []
    short = 0
    for pool, share in zip(bins, shares):
        picked = _evenly(pool, share + short, key=lambda c: (c["raw_gap_bright"], c["raw_gap_min"]))
        short = share + short - len(picked)
        chosen += picked
    if short:  # a bin ran dry: fill from whatever is left, spread the same way
        rest = [c for c in items if c not in chosen]
        chosen += _evenly(rest, short, key=lambda c: (c["raw_gap_bright"], c["raw_gap_min"]))
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw", required=True, help="the raw image (a composite is fine with --raw-channel)")
    parser.add_argument("--raw-channel", type=int, default=None)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="M")
    args = parser.parse_args(argv)

    from build_grids import Drawer, _example
    from haemolynx.graph.dead_ends import classify_dead_ends, image_extent_um
    from haemolynx.graph.facing_ends import facing_dead_end_pairs
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing.raw_evidence import ENDOTHELIAL, calibrate_raw, load_raw_channel, make_raw_evidence

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
    ends = classify_dead_ends(G, extent_um=image_extent_um(mask.shape, vs), support=support)
    pos = lambda n: np.asarray(G.nodes[n]["pos"], dtype=float)

    candidates: dict[str, list[dict]] = {"facing": [], "ahead": [], "bridge": []}
    for pair in facing_dead_end_pairs(G, support, max_gap_um=10.0, search_gap_um=SEARCH_GAP_UM):
        if pair.a not in ends or pair.b not in ends:
            continue
        path = np.vstack([pos(pair.a), pos(pair.b)])
        candidates["facing"].append(dict(
            kind="facing", a=pair.a, b=pair.b, gap_um=round(pair.gap_um, 2), turn_deg=round(pair.turn_deg, 1),
            refused=pair.refused or "", a_kind=ends[pair.a].kind, b_kind=ends[pair.b].kind,
            path=path, **_chord_measures(evidence, support, path),
        ))
    for tip, end in ends.items():
        if end.ahead_um is None or end.heading is None or end.ahead_um > SEARCH_GAP_UM or end.kind in ("leaves_image", "island_end"):
            continue
        target = np.asarray(end.tip_um, dtype=float) + float(end.ahead_um) * np.asarray(end.heading, dtype=float)
        path = np.vstack([np.asarray(end.tip_um, dtype=float), target])
        candidates["ahead"].append(dict(
            kind="ahead", a=tip, b="", gap_um=round(float(end.ahead_um), 2), turn_deg=round(float(end.ahead_turn_deg or 0.0), 1),
            refused="", a_kind=end.kind, b_kind="", ahead_edge=str(end.ahead_edge), path=path,
            **_chord_measures(evidence, support, path),
        ))
    for u, v, k, data in G.edges(keys=True, data=True):
        if float(data.get("bridge_background_um", 0.0) or 0.0) <= 0.0:
            continue
        voxels = np.asarray(data.get("voxels", [pos(u), pos(v)]), dtype=float).reshape(-1, 3)
        candidates["bridge"].append(dict(
            kind="bridge", a=u, b=v, gap_um=round(float(data["bridge_background_um"]), 2), turn_deg=0.0,
            refused=str(data.get("bridge_kind", "")), a_kind="", b_kind="", path=voxels,
            **_chord_measures(evidence, support, voxels),
        ))
    for kind, items in candidates.items():
        print(f"{kind}: {len(items)} candidates")

    chosen: list[dict] = []
    shortfall = 0
    for kind, share in SHARES.items():
        picked = _by_value(candidates[kind], share + shortfall)
        shortfall = share + shortfall - len(picked)
        chosen += picked
    rng = random.Random(SEED)
    rng.shuffle(chosen)

    drawer = Drawer(G, mask, raw.astype(np.float32), vs)
    template = (HERE.parent / "dead_end_review" / "template.html").read_text(encoding="utf-8")
    examples, rows = [], []
    for number, c in enumerate(chosen, start=1):
        ident = f"{args.grid}-{number:02d}"
        if c["kind"] == "facing":
            drawn = _example(drawer, G, ends[c["a"]], partner=ends[c["b"]])
        elif c["kind"] == "ahead":
            drawn = _example(drawer, G, ends[c["a"]], target=c["path"][-1])
        else:
            path = c["path"]
            heading = path[-1] - path[0]
            heading = heading / max(np.linalg.norm(heading), 1e-9)
            drawn = drawer.draw(path, heading, [dict(points=path, color="#ffd60a", width=2.4)],
                                half_thickness=float(np.clip(2 * c["end_radius_um"], 2.0, 5.0)), marker=path[0])
        examples.append({"id": ident, **drawn})
        mid = c["path"][len(c["path"]) // 2]
        row = {k: v for k, v in c.items() if k != "path"}
        rows.append({"id": ident, "grid": args.grid, "mid_zyx_vox": tuple(int(round(x)) for x in mid / np.asarray(vs)),
                     "mid_zyx_um": tuple(round(float(x), 1) for x in mid), **row})
    data = {
        "batch": args.grid,
        "title": f"Raw Bridge Review {args.grid}: does the vessel continue across the gap?",
        "question": "Does one vessel run across this gap in the segmentation?",
        "lede": ("A gap between two pieces of the network: two dead ends facing each other (both red, the gap yellow), "
                 "a dead end with a vessel straight ahead (red, the yellow line to where it would meet it), or a bridge "
                 "the build already drew across background (yellow). The segmentation panel is what the mask rules see. "
                 "Toggle 'Raw image' for the claudin-5 (endothelial) channel: a vessel is a bright wall ring round a "
                 "dark lumen, so a vessel running across the gap shows as walls running across it."),
        "steps": ["Look at the gap on the segmentation, then on the raw image.",
                  "V if one vessel runs across the gap (the segmentation broke it): join.",
                  "N if not (two separate vessels, a real end, nothing there): leave.",
                  "Add a note on what decided it."],
        "legend": [{"label": "the end(s) judged", "color": "#ff3b30"}, {"label": "the gap / the join", "color": "#ffd60a"},
                   {"label": "heading", "color": "#2ad4ff"}, {"label": "other centrelines", "color": "#9a9a9a"}],
        "verdicts": [{"value": "V", "label": "One vessel: join", "short": "V", "tone": "good"},
                     {"value": "N", "label": "Not one vessel: leave", "short": "N", "tone": "bad"},
                     {"value": "?", "label": "Unsure", "short": "?", "tone": "unsure"}],
        "examples": examples,
    }
    html = (template.replace("__TITLE__", f"Raw Bridge Review {args.grid}")
            .replace("__SECOND_PANEL__", "Raw image: claudin-5 (endothelial) channel")
            .replace("__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    (out / f"{args.grid}_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / f"{args.grid}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"{args.grid}_candidates.csv", "w", newline="", encoding="utf-8") as f:
        allrows = [{k: v for k, v in c.items() if k != "path"} for items in candidates.values() for c in items]
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in allrows for k in r)))
        writer.writeheader()
        writer.writerows(allrows)
    print(f"{args.grid}: {len(examples)} examples -> {out / (args.grid + '_review.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
