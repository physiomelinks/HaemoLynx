#!/usr/bin/env python3
"""PLAN_bridging.md Phase 0: the junctions a thick vessel misses.

Two censuses of one built graph (``scripts/graph_artefacts.py``'s
``final.pkl``, after the thick-vessel split), its mask and its thick-vessel
region, and a grid of about 30 examples for marks:

- **Walls** -- every dead end whose tip lies within the thick vessel's lumen
  radius (+ 10 um) of the thick centreline tree: where the tip is (inside the
  region, or outside it), whether the mask runs on from it to the tree
  (``route_through_mask``, judged by the bridge gate), its dead-end kind, and
  the distances a rule would read. ``walls.csv``.
- **Forks** -- every junction of the graph whose arms all lie on the thick
  centreline tree: the inscribed radius along each arm, so the stretch of a
  branch running inside its trunk's lumen can be measured. ``forks.csv``.

The grid (``grid_W_1-15.png``, ``grid_W_16-30.png``, key ``grid_W_key.csv``)
draws each wall case in its own plane on the segmentation: the dead end's
chain red, the thick region's outline orange, the thick centreline yellow, the
mask route a rule would draw dashed cyan, a 5 um bar, the tip's voxel position.
Header: J = this vessel opens into the big vessel, join it; N = not a junction.

Usage::

    python scripts/thick_junction_review/census.py \\
        --build outputs/thick_junction_review/build --voxel-size-zyx 1.0 0.98 0.98 \\
        --out outputs/thick_junction_review
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np

SEED = 20261013
NEAR_UM = 10.0  # a tip this far outside the lumen is still a candidate
ARM_UM = 20.0  # how far along each arm of a fork the radius is read
ARM_STEP_UM = 1.0


def _plane(focus: np.ndarray, heading: np.ndarray | None):
    centre = focus.mean(axis=0)
    _, _, vh = np.linalg.svd(focus - centre, full_matrices=False)
    e1 = heading / np.linalg.norm(heading) if heading is not None else vh[0]
    e2 = None
    for w in (vh[1], vh[0], vh[2]):
        cand = w - (w @ e1) * e1
        if np.linalg.norm(cand) > 0.2:
            e2 = cand
            break
    if e2 is None:
        e2 = np.cross(e1, [1.0, 0.0, 0.0])
        if np.linalg.norm(e2) < 1e-6:
            e2 = np.cross(e1, [0.0, 1.0, 0.0])
    e2 = e2 / np.linalg.norm(e2)
    return centre, e1, e2, np.cross(e1, e2)


def _slab(volume, centre, e1, e2, n, vs, R, half_thickness, order=0):
    from scipy import ndimage

    g = np.arange(-R, R + 1e-9, 0.25)
    U, V = np.meshgrid(g, g, indexing="ij")
    out = np.zeros(U.shape, dtype=float)
    for w in np.arange(-half_thickness, half_thickness + 1e-9, 0.5):
        P = (centre + U[..., None] * e1 + V[..., None] * e2 + w * n) / vs
        out = np.maximum(out, ndimage.map_coordinates(
            volume.astype(np.float32), P.reshape(-1, 3).T, order=order, mode="constant"
        ).reshape(U.shape))
    return out, g


def _edge_points(G, u, v, data, step=0.5):
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
    return densify_polyline(edge_sample_points(u, v, data, pos), max_step_um=step)


def _distance_to_region(tip_um, region, vs, reach_um=25.0):
    """Distance (um) from *tip_um* to the nearest region voxel, 0 inside;
    inf when none lies within *reach_um*."""
    from scipy.spatial import cKDTree

    lo = np.maximum(np.floor((tip_um - reach_um) / vs).astype(int), 0)
    hi = np.minimum(np.ceil((tip_um + reach_um) / vs).astype(int) + 1, np.asarray(region.shape))
    crop = region[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not crop.any():
        return float("inf")
    voxels = (np.argwhere(crop) + lo) * vs
    d, _ = cKDTree(voxels).query(tip_um)
    return float(d)


def walls(G, mask, region, tree_points, tree_radius, support, vs, extent):
    from scipy.spatial import cKDTree

    from haemolynx.graph._helpers import points_inside_mask
    from haemolynx.graph.dead_ends import classify_dead_ends
    from haemolynx.graph.reconnect import route_through_mask
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    tree = cKDTree(tree_points)
    ends = classify_dead_ends(G, extent_um=extent, support=support)
    rows = []
    for tip, end in ends.items():
        tip_um = np.asarray(end.tip_um, dtype=float)
        d_tree, idx = tree.query(tip_um)
        r = float(tree_radius[idx])
        if not np.isfinite(r):
            r = 0.0
        if d_tree > r + NEAR_UM:
            continue
        target = tree_points[idx]
        inside_region = bool(points_inside_mask(tip_um[None], region, voxel_size_zyx=tuple(vs))[0])
        inside_mask = bool(support.inside(tip_um[None])[0])
        d_region = 0.0 if inside_region else _distance_to_region(tip_um, region, vs)
        straight = np.vstack([tip_um, target])
        s_straight = support.support(straight)
        routed = route_through_mask(mask, tip_um, target, vs)
        s_routed = support.support(routed) if routed is not None else None
        best = s_routed if s_routed is not None and support.accepts_support(s_routed) else s_straight
        route_ok = support.accepts_support(best)
        heading = np.asarray(end.heading, dtype=float) if end.heading is not None else None
        to_tree = target - tip_um
        cos = float("nan")
        if heading is not None and np.linalg.norm(to_tree) > 0:
            cos = float(heading @ to_tree / np.linalg.norm(to_tree))
        junction_on_bridge = any(
            d.get(IS_ZERO_RESISTANCE) for _, _, d in G.edges(end.junction, data=True)
        ) if end.junction in G else False
        if inside_region:
            place = "inside_region"
        elif route_ok:
            place = "wall_mask_continuous"
        else:
            place = "wall_gap"
        rows.append(dict(
            tip=tip, tip_zyx_vox=tuple(int(round(c)) for c in tip_um / vs),
            tip_zyx_um=tuple(round(float(c), 2) for c in tip_um),
            place=place, kind=end.kind, chain_length_um=round(end.chain_length_um, 2),
            tip_radius_um=None if end.tip_radius_um is None else round(end.tip_radius_um, 2),
            junction_radius_um=None if end.junction_radius_um is None else round(end.junction_radius_um, 2),
            d_tree_um=round(float(d_tree), 2), tree_radius_um=round(r, 2),
            d_tree_minus_radius_um=round(float(d_tree) - r, 2),
            d_region_um=round(d_region, 2), inside_mask=inside_mask,
            heading_cos_to_tree=None if np.isnan(cos) else round(cos, 3),
            straight_background_um=round(s_straight.longest_background_um, 2),
            routed_background_um=None if s_routed is None else round(s_routed.longest_background_um, 2),
            routed_inside_fraction=None if s_routed is None else round(s_routed.inside_fraction, 3),
            route_accepted=route_ok, junction_on_bridge=junction_on_bridge,
            mask_run_um=None if end.mask_run_um is None else round(end.mask_run_um, 2),
            ahead_um=None if end.ahead_um is None else round(end.ahead_um, 2),
            depth_um=round(float(tip_um[0]), 1),
            _end=end, _target=target, _route=routed if routed is not None else straight,
        ))
    return rows, len(ends)


def forks(G, tree_points, tree_radius, support, vs):
    """Junctions whose every arm is on the tree, with the inscribed radius
    along each arm."""
    from scipy.spatial import cKDTree

    from haemolynx.graph.thick_vessel_junctions import (
        CENTRELINE_TOLERANCE_PER_RADIUS, CENTRELINE_TOLERANCE_UM, IS_ZERO_RESISTANCE,
    )

    tree = cKDTree(tree_points)
    finite = np.where(np.isfinite(tree_radius), tree_radius, 0.0)
    tolerance = np.maximum(CENTRELINE_TOLERANCE_UM, CENTRELINE_TOLERANCE_PER_RADIUS * finite)

    def on_tree(points):
        d, i = tree.query(points)
        return d <= tolerance[i]

    def arm(start, first_neighbour, first_data):
        """Points along the chain out of *start* through *first_neighbour*,
        to ARM_UM, through degree-2 nodes."""
        pieces, prev, cur, data = [], start, first_neighbour, first_data
        total = 0.0
        while True:
            pts = _edge_points(G, prev, cur, data)
            pieces.append(pts)
            total += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            if total >= ARM_UM or G.degree(cur) != 2:
                break
            nxt = [n for n in G.neighbors(cur) if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            data = min(G[prev][cur].values(), key=lambda d: d.get("length", 0.0) or 0.0)
        path = np.vstack(pieces)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        return path[arc <= ARM_UM], arc[arc <= ARM_UM]

    rows = []
    for n in G.nodes:
        if G.degree(n) < 3 or "pos" not in G.nodes[n]:
            continue
        arms = []
        all_on_tree = True
        bridges = 0
        for u, v, k, d in G.edges(n, keys=True, data=True):
            if d.get(IS_ZERO_RESISTANCE):
                bridges += 1
                continue
            pts = _edge_points(G, u, v, d)
            share = float(np.mean(on_tree(pts))) if len(pts) else 0.0
            if share < 0.8:
                all_on_tree = False
                break
            path, arc = arm(u, v, d)
            radii = support.radius(path)
            samples = np.arange(0.0, min(ARM_UM, arc[-1]) + 1e-9, ARM_STEP_UM)
            prof = np.interp(samples, arc, radii)
            arms.append(dict(to=v, length_um=float(arc[-1]), profile=prof))
        if not all_on_tree or len(arms) < 3:
            continue
        pos = np.asarray(G.nodes[n]["pos"], dtype=float)
        r0 = float(support.radius(pos[None])[0])
        means = [float(np.mean(a["profile"][: int(10 / ARM_STEP_UM) + 1])) for a in arms]
        order = np.argsort(means)[::-1]
        trunk = [arms[i] for i in order[:2]]
        branch_arms = [arms[i] for i in order[2:]]
        for b in branch_arms:
            prof = b["profile"]
            below = np.flatnonzero(prof < 0.6 * r0)
            leaves = float(below[0] * ARM_STEP_UM) if len(below) else float("nan")
            rows.append(dict(
                node=n, node_zyx_vox=tuple(int(round(c)) for c in pos / vs), degree=G.degree(n),
                bridges_at_node=bridges, junction_radius_um=round(r0, 2),
                trunk_radius_10um=tuple(round(float(np.mean(t["profile"][:11])), 2) for t in trunk),
                branch_to=b["to"], branch_radius_at=tuple(round(float(prof[i]), 2) for i in (0, 3, 6, 10, 15) if i < len(prof)),
                branch_leaves_lumen_um=None if np.isnan(leaves) else leaves,
            ))
    return rows


def _draw_grid(examples, mask, region, tree_points, vs, out_png, first_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n = len(examples)
    cols = 5
    rowsn = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(4.2 * cols, 4.7 * rowsn + 1.4))
    fig.suptitle(
        "Thick-vessel wall review W: does this vessel open into the big vessel?\n"
        "J = yes, join it to the big vessel's centreline (yellow)   N = no, not a junction   ? = unsure\n"
        "white = segmentation, orange = the thick-vessel region's edge, yellow = its centreline, "
        "red = the dead end's chain, dashed cyan = the route a rule would draw, magenta bar = 5 um",
        fontsize=11,
    )
    for ax, ex in zip(axes.ravel(), examples):
        end, target, route = ex["_end"], ex["_target"], ex["_route"]
        tip = np.asarray(end.tip_um, dtype=float)
        from haemolynx.graph.dead_ends import chain_path
        chain = chain_path(end_graph(ex), end.chain, step_um=0.5) if end.chain else tip[None]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]) if len(chain) > 1 else np.zeros(1)
        chain = chain[arc <= 25.0]
        heading = np.asarray(end.heading, dtype=float) if end.heading is not None else None
        focus = np.vstack([chain, tip[None], target[None], route])
        c, e1, e2, nrm = _plane(focus, heading)
        uv = lambda p: np.column_stack([(np.atleast_2d(p) - c) @ e1, (np.atleast_2d(p) - c) @ e2])
        R = float(np.clip(np.abs(uv(focus)).max() + 8.0, 15.0, 40.0))
        half = float(np.clip(1.5 * (ex["tree_radius_um"] or 2.0), 2.0, 6.0))
        seg, g = _slab(mask, c, e1, e2, nrm, vs, R, half)
        reg, _ = _slab(region, c, e1, e2, nrm, vs, R, half)
        ext = (-R, R, -R, R)
        ax.imshow(seg.T, cmap="gray", vmin=0, vmax=1.0, origin="lower", extent=ext, interpolation="nearest")
        if reg.any():
            ax.contour(g, g, reg.T, levels=[0.5], colors="#ff9500", linewidths=1.2)
        rel = tree_points - c
        near = (np.abs(rel @ nrm) <= half + 1.5) & (np.abs(rel @ e1) <= R) & (np.abs(rel @ e2) <= R)
        if near.any():
            t = uv(tree_points[near])
            ax.plot(t[:, 0], t[:, 1], ".", color="#ffd60a", ms=2.5)
        rr = uv(route)
        ax.plot(rr[:, 0], rr[:, 1], "--", color="#2ad4ff", lw=1.2)
        ch = uv(chain)
        ax.plot(ch[:, 0], ch[:, 1], "-", color="#ff3b30", lw=2.2)
        (tu, tv), = uv(tip)
        ax.add_patch(Circle((tu, tv), 1.6, fill=False, color="#ff3b30", lw=1.4))
        ax.plot([-R + 3, -R + 8], [-R + 3, -R + 3], "-", color="#ff2ad4", lw=3)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ex['id']}   vox (z,y,x) = {ex['tip_zyx_vox']}", fontsize=10, pad=4)
        ax.text(-R + 2, R - 3, ex["id"], color="white", fontsize=12, fontweight="bold", va="top",
                bbox=dict(facecolor="black", edgecolor="none", pad=2))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.04, hspace=0.12)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


_G = None


def end_graph(_ex):
    return _G


def _evenly(items, count, key):
    items = sorted(items, key=key)
    if len(items) <= count:
        return items
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks))]


def main(argv=None) -> int:
    global _G
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", required=True, help="graph_artefacts.py's --out directory")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-size", type=int, default=30)
    args = parser.parse_args(argv)

    from haemolynx.graph.dead_ends import image_extent_um
    from haemolynx.graph.thick_vessel_junctions import _inscribed_radius_um, thick_vessel_centreline
    from haemolynx.preprocessing import MaskSupport

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = np.asarray(args.voxel_size_zyx, dtype=float)
    G = pickle.load(open(build / "working_tree" / "final.pkl", "rb"))
    _G = G
    mask = np.load(build / "mask.npy").astype(bool)
    region = np.load(build / "thick_vessel_mask.npy").astype(bool)
    print(f"graph {G.number_of_nodes()} nodes / {G.number_of_edges()} edges; region {int(region.sum())} voxels", flush=True)
    tree_vox = thick_vessel_centreline(region, tuple(vs))
    tree_points = np.argwhere(tree_vox) * vs
    tree_radius = _inscribed_radius_um(tree_points, mask, vs)
    unknown = ~np.isfinite(tree_radius)
    if unknown.any():
        tree_radius[unknown] = _inscribed_radius_um(tree_points[unknown], region, vs)
    support = MaskSupport(mask, tuple(vs))
    extent = image_extent_um(mask.shape, tuple(vs))
    print(f"tree {len(tree_points)} points, radius median {np.nanmedian(tree_radius):.2f} um", flush=True)

    rows, n_ends = walls(G, mask, region, tree_points, tree_radius, support, vs, extent)
    public = [k for k in rows[0].keys() if not k.startswith("_")] if rows else []
    with open(out / "walls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=public)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in public})
    fork_rows = forks(G, tree_points, tree_radius, support, vs)
    if fork_rows:
        with open(out / "forks.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fork_rows[0].keys()))
            w.writeheader()
            w.writerows(fork_rows)

    # The grid: spread across place and distance past the lumen, shuffled.
    rng = random.Random(SEED)
    by_place = {}
    # The grid asks whether a vessel opens into the big vessel: an end that
    # leaves the image, or an island's, is not that question.
    for r in rows:
        if r["kind"] in ("leaves_image", "island_end"):
            continue
        by_place.setdefault(r["place"], []).append(r)
    chosen = []
    quota = {"inside_region": 10, "wall_mask_continuous": 12, "wall_gap": 8}
    for place, count in quota.items():
        chosen += _evenly(by_place.get(place, []), count, key=lambda r: r["d_tree_minus_radius_um"])
    spare = [r for v in by_place.values() for r in v if r not in chosen]
    chosen += _evenly(spare, max(0, args.grid_size - len(chosen)), key=lambda r: r["d_tree_minus_radius_um"])
    chosen = chosen[: args.grid_size]
    rng.shuffle(chosen)
    for i, r in enumerate(chosen, start=1):
        r["id"] = f"W{i:02d}"
    with open(out / "grid_W_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + public)
        w.writeheader()
        for r in chosen:
            w.writerow({k: r[k] for k in ["id"] + public})
    half = (len(chosen) + 1) // 2
    _draw_grid(chosen[:half], mask, region, tree_points, vs, out / "grid_W_1-15.png", 1)
    _draw_grid(chosen[half:], mask, region, tree_points, vs, out / "grid_W_16-30.png", half + 1)

    summary = {
        "dead_ends": n_ends,
        "near_thick_vessel": len(rows),
        "by_place": {p: len([r for r in rows if r["place"] == p]) for p in ("inside_region", "wall_mask_continuous", "wall_gap")},
        "interior_by_place": {p: len(v) for p, v in by_place.items()},
        "by_kind": {},
        "by_place_and_kind": {},
        "inside_region_with_mask_route": sum(1 for r in rows if r["place"] == "inside_region" and r["route_accepted"]),
        "junction_on_bridge": sum(1 for r in rows if r["junction_on_bridge"]),
        "forks_on_tree": len(fork_rows),
        "forks_with_a_bridge": sum(1 for r in fork_rows if r["bridges_at_node"]),
        "fork_branch_leaves_lumen_um": [r["branch_leaves_lumen_um"] for r in fork_rows],
        "fork_junction_radius_um": [r["junction_radius_um"] for r in fork_rows],
        "grid": [r["id"] for r in chosen],
    }
    for r in rows:
        summary["by_kind"][r["kind"]] = summary["by_kind"].get(r["kind"], 0) + 1
        k = f"{r['place']}/{r['kind']}"
        summary["by_place_and_kind"][k] = summary["by_place_and_kind"].get(k, 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
