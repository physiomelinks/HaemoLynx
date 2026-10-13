#!/usr/bin/env python3
"""Before and after pictures of the thick-vessel bridges (PLAN_bridging.md
Phase 1): every place the bridges differ between two builds of one graph,
each drawn top-down (a z slab round it, as the viewer shows it) twice, side
by side.

A site is a bridge one build has and the other has not (matched by both its
ends within :data:`MATCH_UM`). Each is labelled by what made the change: a
vessel drawn on from a thick vessel's wall (the after build's edge carries
``joined_at_thick_wall_um``), a fork of the thick centreline (both the new
bridge's ends on the tree), a bridge gone (a taper, or a vessel the old split
removed), or another new bridge. Sites closer than :data:`SITE_UM` are drawn
once.

Usage::

    python scripts/thick_junction_review/before_after.py \\
        --before outputs/thick_junction_review/build \\
        --after outputs/thick_junction_review/build_phase1 \\
        --voxel-size-zyx 1.0 0.98 0.98 --out outputs/thick_junction_review
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

MATCH_UM = 3.0
SITE_UM = 8.0
HALF_WINDOW_UM = 22.0
HALF_SLAB_UM = 6.0
PER_PAGE = 12


def _bridges(G, flag):
    out = []
    for u, v, d in G.edges(data=True):
        if d.get(flag):
            out.append((np.asarray(G.nodes[u]["pos"], float), np.asarray(G.nodes[v]["pos"], float)))
    return out


def _matched(bridge, others):
    a, b = bridge
    for c, d in others:
        if (np.linalg.norm(a - c) <= MATCH_UM and np.linalg.norm(b - d) <= MATCH_UM) or (
            np.linalg.norm(a - d) <= MATCH_UM and np.linalg.norm(b - c) <= MATCH_UM
        ):
            return True
    return False


def _paths(G, flag):
    from haemolynx.graph._helpers import edge_sample_points

    out = []
    for u, v, d in G.edges(data=True):
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
        out.append((np.asarray(edge_sample_points(u, v, d, pos), float), bool(d.get(flag))))
    return out


def _draw(ax, centre, mask, region, tree_points, paths, vs, title):
    z0 = int(round(centre[0] / vs[0]))
    zs = slice(max(z0 - int(HALF_SLAB_UM / vs[0]), 0), z0 + int(HALF_SLAB_UM / vs[0]) + 1)
    y0, x0 = centre[1], centre[2]
    ys = slice(max(int((y0 - HALF_WINDOW_UM) / vs[1]), 0), int((y0 + HALF_WINDOW_UM) / vs[1]) + 1)
    xs = slice(max(int((x0 - HALF_WINDOW_UM) / vs[2]), 0), int((x0 + HALF_WINDOW_UM) / vs[2]) + 1)
    seg = mask[zs, ys, xs].any(axis=0)
    reg = region[zs, ys, xs].any(axis=0)
    extent = (xs.start * vs[2], (xs.start + seg.shape[1]) * vs[2], (ys.start + seg.shape[0]) * vs[1], ys.start * vs[1])
    ax.imshow(np.where(seg, 0.32, 0.0), cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    if reg.any():
        gy = ys.start * vs[1] + (np.arange(reg.shape[0]) + 0.5) * vs[1]
        gx = xs.start * vs[2] + (np.arange(reg.shape[1]) + 0.5) * vs[2]
        ax.contour(gx, gy, reg.astype(float), levels=[0.5], colors="#ff9500", linewidths=0.9)
    near = (np.abs(tree_points[:, 0] - centre[0]) <= HALF_SLAB_UM) & (
        np.abs(tree_points[:, 1] - y0) <= HALF_WINDOW_UM) & (np.abs(tree_points[:, 2] - x0) <= HALF_WINDOW_UM)
    if near.any():
        ax.plot(tree_points[near, 2], tree_points[near, 1], ".", color="#ffd60a", ms=1.2, alpha=0.6)
    for path, is_bridge in paths:
        keep = (np.abs(path[:, 0] - centre[0]) <= HALF_SLAB_UM + 2) & (
            np.abs(path[:, 1] - y0) <= HALF_WINDOW_UM + 5) & (np.abs(path[:, 2] - x0) <= HALF_WINDOW_UM + 5)
        if not keep.any():
            continue
        if is_bridge:
            ax.plot(path[:, 2], path[:, 1], "--", color="#ff2ad4", lw=2.4, dashes=(2.0, 1.4))
        else:
            ax.plot(path[:, 2], path[:, 1], "-", color="#3ddc84", lw=1.8)
    ax.set_xlim(x0 - HALF_WINDOW_UM, x0 + HALF_WINDOW_UM)
    ax.set_ylim(y0 + HALF_WINDOW_UM, y0 - HALF_WINDOW_UM)
    ax.plot([x0 - HALF_WINDOW_UM + 2, x0 - HALF_WINDOW_UM + 12], [y0 + HALF_WINDOW_UM - 2] * 2, "-", color="white", lw=2.5)
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="graph_artefacts.py's --out for the old code")
    parser.add_argument("--after", required=True, help="... for the new code")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    from haemolynx.graph.thick_vessel_junctions import (
        CENTRELINE_TOLERANCE_UM, IS_ZERO_RESISTANCE, JOINED_AT_WALL_UM, thick_vessel_centreline,
    )

    vs = np.asarray(args.voxel_size_zyx, float)
    before = pickle.load(open(Path(args.before) / "working_tree" / "final.pkl", "rb"))
    after = pickle.load(open(Path(args.after) / "working_tree" / "final.pkl", "rb"))
    mask = np.load(Path(args.before) / "mask.npy").astype(bool)
    region = np.load(Path(args.before) / "thick_vessel_mask.npy").astype(bool)
    tree_points = np.argwhere(thick_vessel_centreline(region, tuple(vs))) * vs
    tree = cKDTree(tree_points)

    old, new = _bridges(before, IS_ZERO_RESISTANCE), _bridges(after, IS_ZERO_RESISTANCE)
    joined_ends = [np.asarray(after.nodes[n]["pos"], float) for u, v, d in after.edges(data=True)
                   if d.get(JOINED_AT_WALL_UM) and not d.get(IS_ZERO_RESISTANCE) for n in (u, v)]
    joined_tree = cKDTree(np.asarray(joined_ends)) if joined_ends else None
    sites = []
    for a, b in new:
        if _matched((a, b), old):
            continue
        on_tree = [tree.query(p)[0] <= 2 * CENTRELINE_TOLERANCE_UM for p in (a, b)]
        if joined_tree is not None and min(joined_tree.query(a)[0], joined_tree.query(b)[0]) <= 0.5:
            kind = "new bridge: vessel drawn on from the wall"
        elif all(on_tree):
            kind = "new bridge: fork of the thick centreline"
        else:
            kind = "new bridge: other"
        sites.append((kind, (a + b) / 2))
    for a, b in old:
        if not _matched((a, b), new):
            sites.append(("bridge gone", (a + b) / 2))
    order = {"new bridge: vessel drawn on from the wall": 0, "new bridge: fork of the thick centreline": 1,
             "bridge gone": 2, "new bridge: other": 3}
    sites.sort(key=lambda s: (order[s[0]], tuple(np.round(s[1]))))
    kept = []
    for kind, centre in sites:
        if all(np.linalg.norm(centre - c) > SITE_UM for _k, c in kept):
            kept.append((kind, centre))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "before_after_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "change", "centre_zyx_um", "centre_zyx_vox"])
        for i, (kind, centre) in enumerate(kept, start=1):
            w.writerow([f"S{i:02d}", kind, tuple(np.round(centre, 1)), tuple(int(round(c)) for c in centre / vs)])
    paths_before, paths_after = _paths(before, IS_ZERO_RESISTANCE), _paths(after, IS_ZERO_RESISTANCE)
    pages = [kept[i:i + PER_PAGE] for i in range(0, len(kept), PER_PAGE)]
    written = []
    for p, page in enumerate(pages, start=1):
        rows = int(np.ceil(len(page) / 2))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4.15 * rows + 1.6), squeeze=False)
        fig.suptitle(
            "Thick-vessel bridges on E14.5, before (040170b) and after (Phase 1), top-down: a 12 um z slab\n"
            "grey = segmentation, orange = thick-vessel region, yellow dots = its centreline tree, "
            "green = vessels, magenta dashes = bridges, white bar = 10 um",
            fontsize=11,
        )
        for j, (kind, centre) in enumerate(page):
            number = (p - 1) * PER_PAGE + j + 1
            vox = tuple(int(round(c)) for c in centre / vs)
            r, c = divmod(j, 2)
            _draw(axes[r][2 * c], centre, mask, region, tree_points, paths_before, vs,
                  f"S{number:02d} BEFORE  vox (z,y,x) {vox}")
            _draw(axes[r][2 * c + 1], centre, mask, region, tree_points, paths_after, vs,
                  f"S{number:02d} AFTER  {kind.replace('new bridge: ', '+ ')}")
        for ax in axes.ravel()[2 * len(page):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.4 / (4.15 * rows + 1.6), wspace=0.04, hspace=0.14)
        path = out / f"before_after_{p}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(path)
    counts = {}
    for kind, _c in kept:
        counts[kind] = counts.get(kind, 0) + 1
    print(f"{len(kept)} sites:", counts)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
