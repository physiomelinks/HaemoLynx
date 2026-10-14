#!/usr/bin/env python3
"""The gap bridges the bridge gate's new rules refuse (PLAN_bridging.md F2,
F4), drawn for marks: every terminal pair the first graph-building step would
have joined before and now does not, in its own plane on the segmentation.

Each panel: the segmentation slab (white), the two vessels the bridge would
join (red, their last 15 um), the path the gate refused (dashed cyan), other
centrelines (grey dots), a 5 um bar, the bridge's middle as a voxel position.
Header: J = join them (the refusal is wrong), N = leave them apart (right).
The key (reason, lengths) is written beside the image, not on it.

Usage::

    python scripts/thick_junction_review/gate_refusals.py --skeleton <stem>_skeleton.npy \\
        --mask outputs/thick_junction_review/gate_pair2/mask.npy \\
        --voxel-size-zyx 1.0 0.98 0.98 --out outputs/thick_junction_review
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

SEED = 20261014


def _refusals(skeleton, mask, vs):
    """``(G, [(src, tgt, path, reason)])``: the gap bridges the gate refuses
    by turning back (F2) or as a second strand (F4) -- each also re-judged
    with neither rule, so only what the new rules refuse is listed."""
    from skan import csr

    from haemolynx.graph import reconnect
    from haemolynx.graph._helpers import EdgeSampleIndex, duplicates_existing_vessel
    from haemolynx.graph.build import build_graph_segment_skan_stitched_loops
    from haemolynx.preprocessing import MaskSupport

    support = MaskSupport(mask, vs)
    found = []
    original = reconnect.MaskBridges.bridge

    def recording(self, start_um, end_um, preferred=None, *, ends=None, max_turn_deg=None):
        turned, doubled = self.refused_turn, self.refused_duplicate
        result = original(self, start_um, end_um, preferred, ends=ends, max_turn_deg=max_turn_deg)
        if result is not None:
            return result
        if self.refused_turn > turned:
            reason = "turns back on an end (F2)"
        elif self.refused_duplicate > doubled:
            # Only what the old judgement (no middle sample) would have drawn.
            path = reconnect.route_through_mask(self.support.mask, start_um, end_um, self.support.voxel_size_zyx)
            if path is None or not self.support.accepts(path):
                path = np.vstack([start_um, end_um])
            if duplicates_existing_vessel(path, self.G, self.support.inside, self.support.radius,
                                          index=EdgeSampleIndex(self.G)):
                return None
            reason = "second strand, judged at its middle (F4)"
        else:
            return None
        path = reconnect.route_through_mask(self.support.mask, start_um, end_um, self.support.voxel_size_zyx)
        if path is None:
            path = np.vstack([start_um, end_um])
        found.append((ends, np.asarray(path, float), reason))
        return None

    reconnect.MaskBridges.bridge = recording
    try:
        G = build_graph_segment_skan_stitched_loops(
            csr.Skeleton(skeleton), skeleton, voxel_size=vs, reconnect_threshold=10.0, mask_support=support,
        )
    finally:
        reconnect.MaskBridges.bridge = original
    return G, found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    vs = tuple(args.voxel_size_zyx)
    skeleton = np.load(args.skeleton)
    mask = np.load(args.mask).astype(bool)
    G, found = _refusals(skeleton, mask, vs)
    print(f"{len(found)} refused gap bridges")
    if not found:
        return 0
    random.Random(SEED).shuffle(found)

    points = []
    for u, v, d in G.edges(data=True):
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
        points.append(densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=1.0))
    centrelines = np.vstack(points)

    def chain(node):
        (a, b, d), = list(G.edges(node, data=True))[:1]
        other = b if a == node else a
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (node, other)}
        path = densify_polyline(edge_sample_points(node, other, d, pos), max_step_um=0.5)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        return path[arc <= 15.0]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from census import _plane, _slab

    cols = 4
    rows = int(np.ceil(len(found) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.7 * rows + 1.4), squeeze=False)
    fig.suptitle(
        "Gap bridges the gate now refuses: should these two vessel ends be joined?\n"
        "J = yes, join them (the refusal is wrong)   N = no, leave them apart   ? = unsure\n"
        "white = segmentation, red = the two vessels' last 15 um, dashed cyan = the path refused, "
        "grey = other centrelines, magenta bar = 5 um",
        fontsize=11,
    )
    key = []
    for index, ((src, tgt), path, reason) in enumerate(found, start=1):
        ax = axes.ravel()[index - 1]
        a, b = chain(src), chain(tgt)
        focus = np.vstack([a, b, path])
        c, e1, e2, n = _plane(focus, None)
        uv = lambda p: np.column_stack([(np.atleast_2d(p) - c) @ e1, (np.atleast_2d(p) - c) @ e2])
        R = float(np.clip(np.abs(uv(focus)).max() + 6.0, 12.0, 30.0))
        seg, _g = _slab(mask, c, e1, e2, n, np.asarray(vs), R, 3.0)
        ax.imshow(seg.T, cmap="gray", vmin=0, vmax=1.0, origin="lower", extent=(-R, R, -R, R), interpolation="nearest")
        rel = centrelines - c
        near = (np.abs(rel @ n) <= 4.0) & (np.abs(rel @ e1) <= R) & (np.abs(rel @ e2) <= R)
        if near.any():
            q = uv(centrelines[near])
            ax.plot(q[:, 0], q[:, 1], ".", color="#9a9a9a", ms=2.0)
        for piece in (a, b):
            q = uv(piece)
            ax.plot(q[:, 0], q[:, 1], "-", color="#ff3b30", lw=2.4)
        q = uv(path)
        ax.plot(q[:, 0], q[:, 1], "--", color="#2ad4ff", lw=1.6)
        ax.plot([-R + 2, -R + 7], [-R + 2, -R + 2], "-", color="#ff2ad4", lw=3)
        middle = tuple(int(round(x)) for x in path[len(path) // 2] / np.asarray(vs))
        ax.set_title(f"R{index:02d}   vox (z,y,x) = {middle}", fontsize=10, pad=4)
        ax.text(-R + 1.5, R - 2, f"R{index:02d}", color="white", fontsize=12, fontweight="bold", va="top",
                bbox=dict(facecolor="black", edgecolor="none", pad=2))
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_xticks([]); ax.set_yticks([])
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        key.append(dict(id=f"R{index:02d}", reason=reason, path_um=round(length, 2),
                        chord_um=round(float(np.linalg.norm(path[-1] - path[0])), 2), middle_zyx_vox=middle,
                        src=src, tgt=tgt))
    for ax in axes.ravel()[len(found):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.4 / (4.7 * rows + 1.4), wspace=0.04, hspace=0.14)
    out = Path(args.out)
    fig.savefig(out / "grid_R_gate_refusals.png", dpi=105)
    with open(out / "grid_R_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    print(out / "grid_R_gate_refusals.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
