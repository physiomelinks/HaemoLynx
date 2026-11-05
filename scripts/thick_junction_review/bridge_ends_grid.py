#!/usr/bin/env python3
"""Two grids for marks on the thick-vessel bridges' open questions
(PLAN_bridging.md section 5, Harvey's pictures 8 and 12):

- **T** -- bridges whose thick-centreline end is not a node splitting one
  thick edge in two: it is the end of a thick edge, or a thick edge and
  another vessel, or no thick edge at all. Question: is the bridge joined to
  the big vessel properly? Y = yes; N = no, it should end on the big
  vessel's centreline (yellow).
- **U** -- vessels running into a thick vessel's lumen that the split left
  as they were because the graph has no thick centreline near enough to
  bridge to. Question: should this vessel get a bridge to the big vessel?
  J = yes; N = no.

Each panel is drawn top-down (a z slab round it, as the viewer shows it):
segmentation grey, the thick region's edge orange, its centreline tree
yellow dots, vessels green, bridges magenta dashes, the element judged red.
Keys (the measures) beside the images.

Usage::

    python scripts/thick_junction_review/bridge_ends_grid.py \\
        --build outputs/thick_junction_review/gate_pair2 --side working_tree \\
        --thick outputs/thick_junction_review/build/thick_vessel_mask.npy \\
        --voxel-size-zyx 1.0 0.98 0.98 --out outputs/thick_junction_review
"""
from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

SEED = 20261014
HALF_WINDOW_UM = 20.0
HALF_SLAB_UM = 6.0
MAX_T = 30


def _paths(G, flag):
    from haemolynx.graph._helpers import edge_sample_points

    out = []
    for u, v, d in G.edges(data=True):
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
        out.append((np.asarray(edge_sample_points(u, v, d, pos), float), bool(d.get(flag))))
    return out


def _draw(ax, centre, mask, region, tree_points, paths, judged, vs, title, ident):
    z0 = int(round(centre[0] / vs[0]))
    zs = slice(max(z0 - int(HALF_SLAB_UM / vs[0]), 0), z0 + int(HALF_SLAB_UM / vs[0]) + 1)
    y0, x0 = centre[1], centre[2]
    ys = slice(max(int((y0 - HALF_WINDOW_UM) / vs[1]), 0), int((y0 + HALF_WINDOW_UM) / vs[1]) + 1)
    xs = slice(max(int((x0 - HALF_WINDOW_UM) / vs[2]), 0), int((x0 + HALF_WINDOW_UM) / vs[2]) + 1)
    seg = mask[zs, ys, xs].any(axis=0)
    reg = region[zs, ys, xs].any(axis=0)
    extent = (xs.start * vs[2], (xs.start + seg.shape[1]) * vs[2], (ys.start + seg.shape[0]) * vs[1], ys.start * vs[1])
    ax.set_facecolor("black")
    ax.imshow(np.where(seg, 0.32, 0.0), cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    if reg.any():
        gy = ys.start * vs[1] + (np.arange(reg.shape[0]) + 0.5) * vs[1]
        gx = xs.start * vs[2] + (np.arange(reg.shape[1]) + 0.5) * vs[2]
        ax.contour(gx, gy, reg.astype(float), levels=[0.5], colors="#ff9500", linewidths=0.9)
    near = (np.abs(tree_points[:, 0] - centre[0]) <= HALF_SLAB_UM) & (
        np.abs(tree_points[:, 1] - y0) <= HALF_WINDOW_UM) & (np.abs(tree_points[:, 2] - x0) <= HALF_WINDOW_UM)
    if near.any():
        ax.plot(tree_points[near, 2], tree_points[near, 1], ".", color="#ffd60a", ms=1.4, alpha=0.7)
    for path, is_bridge in paths:
        keep = (np.abs(path[:, 0] - centre[0]) <= HALF_SLAB_UM + 2) & (
            np.abs(path[:, 1] - y0) <= HALF_WINDOW_UM + 5) & (np.abs(path[:, 2] - x0) <= HALF_WINDOW_UM + 5)
        if not keep.any():
            continue
        if is_bridge:
            ax.plot(path[:, 2], path[:, 1], "--", color="#ff2ad4", lw=2.2, dashes=(2.0, 1.4))
        else:
            ax.plot(path[:, 2], path[:, 1], "-", color="#3ddc84", lw=1.6)
    for path in judged:
        ax.plot(path[:, 2], path[:, 1], "-", color="#ff3b30", lw=3.0, alpha=0.85)
    ax.plot([x0 - HALF_WINDOW_UM + 2, x0 - HALF_WINDOW_UM + 12], [y0 + HALF_WINDOW_UM - 2] * 2, "-", color="white", lw=2.5)
    ax.text(x0 - HALF_WINDOW_UM + 1.5, y0 - HALF_WINDOW_UM + 1.5, ident, color="white", fontsize=12,
            fontweight="bold", va="top", bbox=dict(facecolor="black", edgecolor="none", pad=2))
    ax.set_xlim(x0 - HALF_WINDOW_UM, x0 + HALF_WINDOW_UM)
    ax.set_ylim(y0 + HALF_WINDOW_UM, y0 - HALF_WINDOW_UM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _grid(cases, header, out_png, mask, region, tree_points, vs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 5
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 4.05 * rows + 1.4), squeeze=False)
    fig.suptitle(header, fontsize=11)
    for ax, case in zip(axes.ravel(), cases):
        _draw(ax, case["centre"], mask, region, tree_points, case["paths"], case["judged"], vs,
              f"{case['id']}  vox (z,y,x) {case['vox']}", case["id"])
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.4 / (4.05 * rows + 1.4), wspace=0.04, hspace=0.14)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", required=True)
    parser.add_argument("--side", default="working_tree")
    parser.add_argument("--thick", required=True)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from scipy.spatial import cKDTree

    import haemolynx.graph.thick_vessel_junctions as t
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    vs = np.asarray(args.voxel_size_zyx, float)
    build = Path(args.build)
    mask = np.load(build / "mask.npy").astype(bool)
    region = np.load(args.thick).astype(bool)
    final = pickle.load(open(build / args.side / "final.pkl", "rb"))
    before = pickle.load(open(build / args.side / "before_thick_split.pkl", "rb"))
    tree_vox = t.thick_vessel_centreline(region, tuple(vs))
    tree_points = np.argwhere(tree_vox) * vs
    tree_r = t._inscribed_radius_um(tree_points, mask, vs)
    tree_r = np.where(np.isfinite(tree_r), tree_r, 0.0)
    tol = np.maximum(t.CENTRELINE_TOLERANCE_UM, t.CENTRELINE_TOLERANCE_PER_RADIUS * tree_r)
    tree = cKDTree(tree_points)
    rng = random.Random(SEED)
    out = Path(args.out)

    def pts(G, u, v, d):
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
        return densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=0.5)

    def on_share(p):
        dd, ii = tree.query(p)
        return float(np.mean(dd <= tol[ii]))

    # --- T: bridges whose centreline end does not split one thick edge ----------
    paths_final = _paths(final, t.IS_ZERO_RESISTANCE)
    cases_t = []
    for u, v, d in final.edges(data=True):
        if not d.get(t.IS_ZERO_RESISTANCE):
            continue
        pu, pv = np.asarray(final.nodes[u]["pos"], float), np.asarray(final.nodes[v]["pos"], float)
        end, far = (u, v) if tree.query(pu)[0] <= tree.query(pv)[0] else (v, u)
        vessels = [(a, b, dd) for a, b, dd in final.edges(end, data=True) if not dd.get(t.IS_ZERO_RESISTANCE)]
        thick = sum(on_share(pts(final, a, b, dd)) >= 0.8 for a, b, dd in vessels)
        if thick >= 2:
            continue
        shape = ("at a thick edge's end" if thick == 1 and len(vessels) == 1
                 else "on a thick edge and another vessel" if thick == 1 else "on no thick edge")
        bridge = pts(final, u, v, d)
        pe = np.asarray(final.nodes[end]["pos"], float)
        cases_t.append(dict(centre=pe, judged=[bridge], paths=paths_final, shape=shape,
                            d_end_to_tree_um=round(float(tree.query(pe)[0]), 2),
                            bridge_um=round(float(np.linalg.norm(pu - pv)), 2),
                            vessels_at_end=len(vessels), thick_edges_at_end=int(thick)))
    rng.shuffle(cases_t)
    cases_t = cases_t[:MAX_T]
    for i, case in enumerate(cases_t, start=1):
        case["id"] = f"T{i:02d}"
        case["vox"] = tuple(int(round(c)) for c in case["centre"] / vs)
    if cases_t:
        _grid(cases_t, "Thick-vessel bridges (magenta dashes) whose end is not on a thick edge running on both ways: "
              "is the bridge joined to the big vessel properly?\n"
              "Y = yes, it is right   N = no, it should end on the big vessel's centreline (yellow)   ? = unsure\n"
              "grey = segmentation, orange = thick region, yellow dots = its centreline tree, green = vessels, "
              "red = the bridge judged, white bar = 10 um",
              out / "grid_T_bridge_ends.png", mask, region, tree_points, vs)
        with open(out / "grid_T_key.csv", "w", newline="", encoding="utf-8") as f:
            keys = ["id", "vox", "shape", "d_end_to_tree_um", "bridge_um", "vessels_at_end", "thick_edges_at_end"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: c[k] for k in keys} for c in cases_t)

    # --- U: connectors the split left for want of a centreline near -------------
    captured = []
    original = t._choose_bridges

    def recording(sampled, inside_runs, find, bridged, *rest):
        result = original(sampled, inside_runs, find, bridged, *rest)
        failed = result[1]
        for root in failed:
            runs = [sampled[i].points[s:e + 1] for r, (i, s, e) in enumerate(inside_runs) if find(("run", r)) == root]
            if runs:
                captured.append(np.vstack(runs))
        return result

    t._choose_bridges = recording
    try:
        t.insert_thick_vessel_junction_nodes(before, region, voxel_size_zyx=tuple(vs), segmentation=mask, centreline=tree_vox)
    finally:
        t._choose_bridges = original
    seen, cases_u = [], []
    for run in captured:
        centre = run[len(run) // 2]
        if any(np.linalg.norm(centre - c) < 3.0 for c in seen):
            continue
        seen.append(centre)
        d_tree, it = tree.query(centre)
        cases_u.append(dict(centre=centre, judged=[run], paths=_paths(before, t.IS_ZERO_RESISTANCE),
                            run_um=round(float(np.sum(np.linalg.norm(np.diff(run, axis=0), axis=1))), 2),
                            d_to_tree_um=round(float(d_tree), 2), tree_radius_um=round(float(tree_r[it]), 2)))
    rng.shuffle(cases_u)
    for i, case in enumerate(cases_u, start=1):
        case["id"] = f"U{i:02d}"
        case["vox"] = tuple(int(round(c)) for c in case["centre"] / vs)
    if cases_u:
        _grid(cases_u, "Vessels running into a thick vessel that get no bridge, for want of a big-vessel centreline "
              "in the graph near enough: should this vessel be bridged to the big vessel?\n"
              "J = yes   N = no   ? = unsure     (drawn on the graph before the thick-vessel split)\n"
              "grey = segmentation, orange = thick region, yellow dots = its centreline tree, green = vessels, "
              "red = the vessel's stretch inside the thick region, white bar = 10 um",
              out / "grid_U_unbridged.png", mask, region, tree_points, vs)
        with open(out / "grid_U_key.csv", "w", newline="", encoding="utf-8") as f:
            keys = ["id", "vox", "run_um", "d_to_tree_um", "tree_radius_um"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: c[k] for k in keys} for c in cases_u)
    print(f"T: {len(cases_t)} bridges, U: {len(cases_u)} unbridged vessels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
