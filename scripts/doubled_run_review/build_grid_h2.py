#!/usr/bin/env python3
"""Grid H2 of PLAN_clean_graph.md: the dead-end chains a wider "beside" test
would trim away -- are they vessels?

Judging a doubled run with the wider of the two lumen radii made every
capillary fused along a trunk's wall a hair of the trunk, and the trim took
it whole: on E14.5 the graph lost 5.7 mm of centreline (``outputs/
clean_graph_phase1_sym``). Those chains are what this grid shows: 30 dead
ends present in the graph built without that test and absent from the one
built with it, spread across the chain's length, each drawn in its own plane
on the segmentation with the raw dextran channel beside it (the one channel
this run has). Marks: V = a real vessel (keep it), X = not a vessel (a
hair, a second strand: remove it), ? = unsure.

Usage::

    python scripts/doubled_run_review/build_grid_h2.py --before <final.pkl> --after <final.pkl>
        --mask mask.npy --raw C1.tif --voxel-size-zyx 1 0.98 0.98 --out outputs/doubled_run_review
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
SEED = 20261012


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="final.pkl of the build whose chains are judged")
    parser.add_argument("--after", required=True, help="final.pkl of the build that removed some of them")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw", required=True, help="the raw image, one channel, z y x")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="H2")
    args = parser.parse_args(argv)

    import tifffile
    from scipy.spatial import cKDTree

    from build_grids import Drawer, _evenly, _example
    from haemolynx.graph.dead_ends import classify_dead_ends, image_extent_um
    from haemolynx.preprocessing import MaskSupport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = pickle.load(open(args.before, "rb"))
    after = pickle.load(open(args.after, "rb"))
    mask = np.load(args.mask).astype(bool)
    raw = tifffile.imread(args.raw).astype(np.float32)
    if raw.shape != mask.shape:
        raise SystemExit(f"raw image {raw.shape} and mask {mask.shape} differ")
    vs = tuple(args.voxel_size_zyx)
    support = MaskSupport(mask, vs)
    ends = classify_dead_ends(before, extent_um=image_extent_um(mask.shape, vs), support=support)
    kept = np.asarray([after.nodes[n]["pos"] for n in after if after.degree(n) == 1 and "pos" in after.nodes[n]], dtype=float)
    tree = cKDTree(kept) if len(kept) else None
    gone = []
    for tip, end in ends.items():
        if end.kind in ("leaves_image", "island_end"):
            continue
        if tree is None or tree.query(np.asarray(end.tip_um), distance_upper_bound=2.0)[0] > 2.0:
            gone.append(end)
    print(f"{len(gone)} interior dead ends of the first build have no tip within 2 um in the second")
    chosen = _evenly(gone, 30, key=lambda e: e.chain_length_um)
    rng = random.Random(SEED)
    rng.shuffle(chosen)
    drawer = Drawer(before, mask, raw, vs)
    template = (HERE.parent / "dead_end_review" / "template.html").read_text(encoding="utf-8")
    examples, rows = [], []
    for number, end in enumerate(chosen, start=1):
        ident = f"{args.grid}-{number:02d}"
        examples.append({"id": ident, **_example(drawer, before, end)})
        rows.append({"id": ident, "grid": args.grid,
                     "tip_zyx_vox": tuple(int(round(c)) for c in np.asarray(end.tip_um) / np.asarray(vs)),
                     **end.measures()})
    data = {
        "batch": args.grid,
        "title": f"Dead End Review {args.grid}: chains a wider test would trim",
        "question": "Is this dead-end chain a vessel?",
        "lede": ("A dead end (red) that runs beside a wider vessel in the same segmented lumen, which a wider "
                 "'beside' test would cut back to where it leaves that vessel. Its heading is cyan. Toggle "
                 "'Raw image' for the dextran channel, the one channel this run has."),
        "steps": ["Look at the red chain against the segmentation, then against the raw image.",
                  "V if it is a vessel of its own (a capillary along the trunk, a branch): keep it.",
                  "X if it is not (a hair along the wall, the trunk drawn twice): remove it.",
                  "Add a note on what decided it."],
        "legend": [{"label": "the dead end judged", "color": "#ff3b30"}, {"label": "its heading", "color": "#2ad4ff"},
                   {"label": "other centrelines", "color": "#9a9a9a"}, {"label": "your drawing", "color": "#ffd60a"}],
        "verdicts": [{"value": "V", "label": "A vessel: keep", "short": "V", "tone": "good"},
                     {"value": "X", "label": "Not a vessel: remove", "short": "X", "tone": "bad"},
                     {"value": "?", "label": "Unsure", "short": "?", "tone": "unsure"}],
        "examples": examples,
    }
    html = (template.replace("__TITLE__", f"Dead End Review {args.grid}")
            .replace("__SECOND_PANEL__", "Raw image (not the segmentation)")
            .replace("__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    (out / f"{args.grid}_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / f"{args.grid}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.grid}: {len(examples)} examples -> {out / (args.grid + '_review.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
