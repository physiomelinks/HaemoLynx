#!/usr/bin/env python3
"""Grid V, for marks on the thick-vessel tree fill (PLAN_bridging.md 6a,
``thick_vessel_fill_centreline_gaps``): every gap in the graph's thick
centreline the fill draws from the tree, with the stray strand it takes out
beside it, if any. Question: is the drawn path the big vessel's centreline
here? Y = yes; N = no, leave the graph as it was; K = yes, but the strand
taken out (cyan) should stay.

Each panel is drawn top-down (a z slab round it, as the viewer shows it), on
the graph as the fill found it: segmentation grey, the thick region's edge
orange, its centreline tree yellow dots, vessels green, bridges magenta
dashes, the path drawn red, the strand taken out cyan. The key (the
measures) beside the image.

Usage::

    PYTHONPATH=src python scripts/thick_junction_review/fill_grid.py \\
        --build outputs/thick_junction_review/gate_pair2 --side working_tree \\
        --thick outputs/thick_junction_review/build/thick_vessel_mask.npy \\
        --voxel-size-zyx 1.0 0.98 0.98 --out outputs/thick_junction_review
"""
from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_ends_grid import HALF_SLAB_UM, HALF_WINDOW_UM, _paths  # noqa: E402

SEED = 20261015


def _draw(ax, case, mask, region, tree_points, vs):
    from bridge_ends_grid import _draw as draw_base

    draw_base(ax, case["centre"], mask, region, tree_points, case["paths"], case["judged"], vs,
              f"{case['id']}  vox (z,y,x) {case['vox']}", case["id"])
    for path in case["removed"]:
        ax.plot(path[:, 2], path[:, 1], "-", color="#32d3ff", lw=2.6, alpha=0.9)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", required=True)
    parser.add_argument("--side", default="working_tree")
    parser.add_argument("--thick", required=True)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    import networkx as nx

    import haemolynx.graph.thick_vessel_junctions as t
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    vs = np.asarray(args.voxel_size_zyx, float)
    build = Path(args.build)
    mask = np.load(build / "mask.npy").astype(bool)
    region = np.load(args.thick).astype(bool)
    before = pickle.load(open(build / args.side / "before_thick_split.pkl", "rb"))
    tree_vox = t.thick_vessel_centreline(region, tuple(vs))
    tree_points = np.argwhere(tree_vox) * vs
    tree_r = t._inscribed_radius_um(tree_points, mask, vs)
    tree_r = np.where(np.isfinite(tree_r), tree_r, 0.0)
    tree = cKDTree(tree_points)
    deg = np.array([len(x) - 1 for x in cKDTree(np.argwhere(tree_vox)).query_ball_point(np.argwhere(tree_vox), 1.75)])
    tree_ends = cKDTree(tree_points[deg == 1]) if (deg == 1).any() else None

    def polyline(G, u, v, d):
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
        return densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=0.5)

    snapshot = {}
    removed_strands = []
    removed_edges_of_strays = []
    original_fill = t._fill_tree_gaps
    # Taking out a stray strand went with the detour rule (2026-10-15); the
    # grid as marked was drawn at 51b34f2, which still had it.
    original_remove = getattr(t, "_remove_stray_beside", None)

    def recording_fill(G, *rest):
        snapshot["paths"] = _paths(G, t.IS_ZERO_RESISTANCE)
        before_pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in G.nodes if "pos" in G.nodes[n]}
        snapshot["edges"] = {(u, v, k) for u, v, k in G.edges(keys=True)}
        result = original_fill(G, *rest)
        snapshot["pos"] = dict(before_pos)
        snapshot["pos"].update({n: np.asarray(G.nodes[n]["pos"], float) for n in G.nodes if "pos" in G.nodes[n]})
        snapshot["added"] = [
            (u, v, k, polyline(G, u, v, d)) for u, v, k, d in G.edges(keys=True, data=True) if d.get(t.FROM_TREE)
        ]
        # The graph's own way between each drawn stretch's two ends, before the
        # strays went: is the gap a stretch the graph already joins?
        H = nx.Graph()
        for a, b, d in G.edges(data=True):
            if d.get(t.FROM_TREE) or d.get(t.IS_ZERO_RESISTANCE):
                continue
            w = float(d.get("length") or 0.0)
            if not H.has_edge(a, b) or H[a][b]["length"] > w:
                H.add_edge(a, b, length=w)
        for a, b, d in removed_edges_of_strays:
            H.add_edge(a, b, length=float(d.get("length") or 0.0))
        snapshot["own_way"] = {}
        for u, v, _k, _p in snapshot["added"]:
            try:
                path = nx.shortest_path(H, u, v, weight="length")
                snapshot["own_way"][(u, v)] = (nx.path_weight(H, path, "length"), len(path) - 1, path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                snapshot["own_way"][(u, v)] = (None, None, None)
        return result

    def recording_remove(G, a, b, *rest):
        lines = {(u, v, k): polyline(G, u, v, d) for u, v, k, d in G.edges(keys=True, data=True)}
        data = {(u, v, k): dict(d) for u, v, k, d in G.edges(keys=True, data=True)}
        count = original_remove(G, a, b, *rest)
        if count:
            removed_edges_of_strays.extend((e[0], e[1], data[e]) for e in data if not G.has_edge(*e))
            gone = [lines[e] for e in lines if not G.has_edge(*e)]
            removed_strands.append(np.vstack(gone))
        return count

    t._fill_tree_gaps = recording_fill
    if original_remove is not None:
        t._remove_stray_beside = recording_remove
    try:
        t.insert_thick_vessel_junction_nodes(
            before, region, voxel_size_zyx=tuple(vs), segmentation=mask, centreline=tree_vox, fill_tree_gaps=True,
        )
    finally:
        t._fill_tree_gaps = original_fill
        if original_remove is not None:
            t._remove_stray_beside = original_remove

    def before_fill_pos(n):
        return snapshot["pos"][n]

    # One case per gap: the added edges meeting at shared nodes are one gap
    # (a hub joins three or more sides).
    added = snapshot.get("added", [])
    groups: list[list[int]] = []
    for i, (u, v, _k, _p) in enumerate(added):
        hit = [g for g in groups if any({u, v} & {added[j][0], added[j][1]} for j in g)]
        merged = [i] + [j for g in hit for j in g]
        groups = [g for g in groups if g not in hit] + [merged]
    cases = []
    for g in groups:
        drawn = [added[j][3] for j in g]
        points = np.vstack(drawn)
        strands = [s for s in removed_strands if float(np.min(cKDTree(points).query(s)[0])) <= 1.0]
        length = sum(float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in drawn)
        _d, near = tree.query(points)
        stray_off = (float(np.mean(tree.query(np.vstack(strands))[0])) if strands else None)
        other = cKDTree(np.vstack([p for p, is_bridge in snapshot["paths"] if not is_bridge]))
        u0, v0 = added[g[0]][0], added[g[0]][1]
        own_um, own_hops, own_path = snapshot["own_way"].get((u0, v0), (None, None, None))
        own_off = None
        if own_path is not None and len(g) == 1:
            own_points = np.vstack([np.asarray(before_fill_pos(n)) for n in own_path])
            own_off = round(float(np.max(tree.query(own_points)[0]) / max(float(np.median(tree_r[near])), 0.5)), 2)
        cases.append(dict(
            own_way_um=None if own_um is None else round(own_um, 2),
            own_way_ratio=None if own_um is None else round(own_um / max(length, 1e-6), 2),
            own_way_edges=own_hops, own_way_nodes_max_off_tree_radii=own_off,
            centre=points[len(points) // 2], judged=drawn, removed=strands, paths=snapshot["paths"],
            drawn_um=round(length, 2), sides=len(g) if len(g) > 1 else 2,
            tree_radius_um=round(float(np.median(tree_r[near])), 2),
            strand_removed_um=round(sum(float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) for s in strands), 2),
            strand_mean_off_tree_um=None if stray_off is None else round(stray_off, 2),
            graph_within_2r_um=round(float(np.mean(other.query(points)[0] <= 2 * tree_r[near] + 1.0)), 2),
            to_tree_end_um=None if tree_ends is None else round(float(tree_ends.query(points)[0].min()), 2),
        ))
    rng = random.Random(SEED)
    rng.shuffle(cases)
    for i, case in enumerate(cases, start=1):
        case["id"] = f"V{i:02d}"
        case["vox"] = tuple(int(round(c)) for c in case["centre"] / vs)

    out = Path(args.out)
    cols = 5
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 4.05 * rows + 1.6), squeeze=False)
    fig.suptitle(
        "Thick-vessel centreline drawn from the centreline tree (red) where the graph's own (green) strays off it "
        "or stops: is red the big vessel's centreline here?\n"
        "Y = yes, draw it   N = no, leave the graph as it was   K = yes, but the strand taken out (cyan) should stay"
        "   ? = unsure\n"
        "grey = segmentation, orange = thick region, yellow dots = its centreline tree, green = the graph before, "
        "magenta dashes = bridges, white bar = 10 um", fontsize=11)
    for ax, case in zip(axes.ravel(), cases):
        _draw(ax, case, mask, region, tree_points, vs)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.6 / (4.05 * rows + 1.6), wspace=0.04, hspace=0.14)
    fig.savefig(out / "grid_V_tree_fill.png", dpi=100)
    plt.close(fig)
    keys = ["id", "vox", "own_way_um", "own_way_ratio", "own_way_edges", "own_way_nodes_max_off_tree_radii", "drawn_um", "sides", "tree_radius_um", "strand_removed_um", "strand_mean_off_tree_um",
            "graph_within_2r_um", "to_tree_end_um"]
    with open(out / "grid_V_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: c[k] for k in keys} for c in cases)
    print(f"V: {len(cases)} gaps filled ({len(added)} stretches), {len(removed_strands)} strands taken out; "
          f"half window {HALF_WINDOW_UM} um, slab +/-{HALF_SLAB_UM} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
