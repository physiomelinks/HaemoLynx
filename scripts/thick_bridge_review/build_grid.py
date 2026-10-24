#!/usr/bin/env python3
"""The marked grid of PLAN_thick_vessel_bridges.md, Phase 5: thick-vessel
bridges and the near-misses beside them, for Harvey's marks.

About 30 bridges (``IS_ZERO_RESISTANCE``) and about 30 near-misses -- stretches
of vessel the split left inside, or within 2 um of, the thick-vessel region
and off its centreline tree: the thick vessel's own path where the graph's
centreline strays, vessels passing through the region's margin, tapers,
vessels the split could not bridge. Each case is measured by how far it lies
from the tree in units of the split's tolerance (a bridge by its inner end, a
near-miss by its nearest point) and by its length over the thick lumen's
radius, and picked spread across both (every bridge of a shape the split
never means to make, and every one drawn round a bend, is in), then all of
them shuffled into one run of IDs, so an ID gives away nothing. The question, in each sheet's header:
what should the red piece be?

    B = a bridge   V = vessel   C = the thick vessel's own centreline
    N = should not be connected at all   ? = unsure

Each case is drawn in its own plane -- looking down whichever axis the piece
and the tree round it extend least along -- on the segmentation (a slab
either side of the plane), the thick-vessel region's edge, the tree, the
graph's centrelines and bridges, a 10 um bar and the case's voxel position.
The measures are in ``key.csv`` beside the sheets, never on them.

    python scripts/thick_bridge_review/build_grid.py \\
        --capture outputs/tb/after/finish_input.pkl --graph outputs/tb/after/graph_built.pkl \\
        --out outputs/thick_bridge_review
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np

SEED = 20261024
HALF_WINDOW_UM = 22.0
HALF_SLAB_UM = 6.0
PER_SHEET = 30
TARGET_BRIDGES = 30
TARGET_NEAR_MISSES = 30
#: A near-miss stretch is at least this long (um): shorter is the split's flicker.
MIN_STRETCH_UM = 2.0
#: ... and lies inside the region or this close (um) to it.
BESIDE_REGION_UM = 2.0
#: Bins the cases are spread across: distance from the tree over the tolerance,
#: and length over the lumen radius.
DISTANCE_BINS = (0.0, 1.0, 1.5, 2.5, 4.0, np.inf)
RATIO_BINS = (0.0, 0.75, 1.25, 2.0, np.inf)

HEADER = (
    "What should the red piece be?   B = a bridge (no resistance: the big vessel's radius between its "
    "centreline and a vessel opening into it)   V = vessel (keeps its resistance)\n"
    "C = the big vessel's own centreline   N = should not be connected at all   ? = unsure\n"
    "grey = segmentation (a slab 6 um either side of the plane), orange = thick-vessel region's edge, "
    "yellow dots = its centreline tree, green = vessels, magenta dashes = bridges, red = the piece judged, "
    "white bar = 10 um"
)

#: Per axis looked down: its name, the axis drawn down the panel and the one
#: drawn across (always in that order in the array, the lower axis first).
PLANES = {0: ("down z", 1, 2), 1: ("down y", 0, 2), 2: ("down x", 0, 1)}


def _edge_path(G, u, v, d):
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in (u, v)}
    return densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=0.5)


def near_misses(G, region, tree, tree_radius, tolerance, vs):
    """Stretches of G's vessels inside, or within :data:`BESIDE_REGION_UM` of,
    the region and further from the tree than its tolerance."""
    from scipy.spatial import cKDTree

    from haemolynx.graph import IS_ZERO_RESISTANCE
    from haemolynx.graph._helpers import points_inside_mask
    from haemolynx.graph.thick_vessel_junctions import FROM_TREE, THICK_TAPER

    region_points = cKDTree(np.argwhere(region) * vs)
    found = []
    for u, v, key, d in G.edges(keys=True, data=True):
        if d.get(IS_ZERO_RESISTANCE) or u == v:
            continue
        path = _edge_path(G, u, v, d)
        if len(path) < 2:
            continue
        to_region = region_points.query(path, distance_upper_bound=BESIDE_REGION_UM + 1.0)[0]
        if not np.isfinite(to_region).any():
            continue
        to_region[points_inside_mask(path, region, voxel_size_zyx=tuple(vs))] = 0.0
        distance, nearest = tree.query(path)
        candidate = (to_region <= BESIDE_REGION_UM) & (distance > tolerance[nearest])
        steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        k = 0
        while k < len(path):
            if not candidate[k]:
                k += 1
                continue
            start = k
            while k + 1 < len(path) and candidate[k + 1]:
                k += 1
            stop = k
            k += 1
            length = float(steps[stop] - steps[start])
            if length < MIN_STRETCH_UM:
                continue
            piece = path[start : stop + 1]
            dist, near = distance[start : stop + 1], nearest[start : stop + 1]
            at = int(np.argmin(dist / tolerance[near]))
            radius = float(tree_radius[near[at]])
            touches = (start > 0 and distance[start - 1] <= tolerance[nearest[start - 1]]) or (
                stop + 1 < len(path) and distance[stop + 1] <= tolerance[nearest[stop + 1]])
            found.append(dict(
                kind="near_miss", edge=(u, v, key), piece=piece, centre=piece[len(piece) // 2],
                length_um=length, local_radius_um=radius,
                length_over_radius=length / radius if radius > 0 else np.inf,
                distance_over_tolerance=float(dist[at] / tolerance[near[at]]),
                distance_to_tree_um=float(dist[at]),
                deepest_over_radius=float(np.min(dist / np.maximum(tree_radius[near], 1e-9))),
                inside_region_share=float(np.mean(to_region[start : stop + 1] == 0.0)),
                touches_centreline=bool(touches),
                edge_length_um=float(d.get("length", steps[-1])),
                taper=bool(d.get(THICK_TAPER)), from_tree=bool(d.get(FROM_TREE)),
                edge_ends_degree=f"{G.degree(u)},{G.degree(v)}",
            ))
    return found


def _bin(value, bins):
    return int(np.searchsorted(bins, value, side="right")) - 1


def spread(cases, target, rng, must=()):
    """Up to *target* cases: every one of *must* (indices into *cases*), then
    the rest taken round-robin over the (distance, ratio) bins."""
    picked = [cases[i] for i in must][:target]
    cells: dict = {}
    for index, case in enumerate(cases):
        if index in must:
            continue
        cell = (_bin(case["distance_over_tolerance"], DISTANCE_BINS), _bin(case["length_over_radius"], RATIO_BINS))
        cells.setdefault(cell, []).append(case)
    for group in cells.values():
        rng.shuffle(group)
    order = sorted(cells)
    while len(picked) < target and any(cells[c] for c in order):
        for cell in order:
            if cells[cell] and len(picked) < target:
                picked.append(cells[cell].pop())
    return picked


def _plane(case, tree_points):
    """The axis to look down: the one the piece and the tree round it extend least along."""
    near = tree_points[np.linalg.norm(tree_points - case["centre"], axis=1) <= 12.0]
    points = np.vstack([case["piece"], near]) if len(near) else case["piece"]
    extent = points.max(axis=0) - points.min(axis=0)
    # Ties go to looking down z, as the viewer opens.
    return int(np.argmin(extent - np.array([0.5, 0.0, 0.0])))


def _draw(ax, case, mask, region, tree_points, paths, vs):
    axis = case["axis"]
    label, h, w = PLANES[axis]
    c = case["centre"]
    lo = np.floor((c - HALF_WINDOW_UM) / vs).astype(int)
    hi = np.ceil((c + HALF_WINDOW_UM) / vs).astype(int) + 1
    lo[axis] = int(np.floor((c[axis] - HALF_SLAB_UM) / vs[axis]))
    hi[axis] = int(np.ceil((c[axis] + HALF_SLAB_UM) / vs[axis])) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, mask.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    # The two axes left are in (h, w) order in every view (h < w), so the
    # slab's projection is indexed [h, w] as drawn: w across, h down.
    image = np.asarray(mask[box]).any(axis=axis)
    regs = np.asarray(region[box]).any(axis=axis)
    extent = (lo[w] * vs[w], hi[w] * vs[w], hi[h] * vs[h], lo[h] * vs[h])
    ax.set_facecolor("black")
    ax.imshow(np.where(image, 0.32, 0.0), cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    if regs.any():
        gh = (lo[h] + np.arange(regs.shape[0]) + 0.5) * vs[h]
        gw = (lo[w] + np.arange(regs.shape[1]) + 0.5) * vs[w]
        ax.contour(gw, gh, regs.astype(float), levels=[0.5], colors="#ff9500", linewidths=0.9)

    def in_view(p, pad=0.0):
        return (np.abs(p[:, axis] - c[axis]) <= HALF_SLAB_UM + pad) & (
            np.abs(p[:, h] - c[h]) <= HALF_WINDOW_UM + 5) & (np.abs(p[:, w] - c[w]) <= HALF_WINDOW_UM + 5)

    near = in_view(tree_points)
    if near.any():
        ax.plot(tree_points[near, w], tree_points[near, h], ".", color="#ffd60a", ms=2.6, alpha=0.9)
    for path, is_bridge in paths:
        keep = in_view(path, pad=2.0)
        if not keep.any():
            continue
        shown = np.where(keep[:, None], path, np.nan)
        if is_bridge:
            ax.plot(shown[:, w], shown[:, h], "--", color="#ff2ad4", lw=2.2, dashes=(2.0, 1.4))
        else:
            ax.plot(shown[:, w], shown[:, h], "-", color="#3ddc84", lw=1.6)
    piece = case["piece"]
    ax.plot(piece[:, w], piece[:, h], "-", color="#ff3b30", lw=3.2, alpha=0.85)
    left, top = c[w] - HALF_WINDOW_UM, c[h] - HALF_WINDOW_UM
    ax.plot([left + 2, left + 12], [c[h] + HALF_WINDOW_UM - 2] * 2, "-", color="white", lw=2.5)
    ax.text(left + 1.5, top + 1.5, str(case["id"]), color="white", fontsize=13, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="none", pad=2))
    ax.set_xlim(c[w] - HALF_WINDOW_UM, c[w] + HALF_WINDOW_UM)
    ax.set_ylim(c[h] + HALF_WINDOW_UM, c[h] - HALF_WINDOW_UM)
    ax.set_xticks([])
    ax.set_yticks([])
    names = "zyx"
    ax.set_title(f"{case['id']}  vox (z,y,x) {case['vox']}  {label}  ({names[w]} across, {names[h]} down)",
                 fontsize=8)


def _sheet(cases, out_png, mask, region, tree_points, paths, vs, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 6
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.75 * rows + 1.6), squeeze=False)
    fig.suptitle(f"{title}\n{HEADER}", fontsize=10)
    for ax, case in zip(axes.ravel(), cases):
        _draw(ax, case, mask, region, tree_points, paths, vs)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.6 / (3.75 * rows + 1.6), wspace=0.04, hspace=0.16)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


KEY_FIELDS = (
    "id", "kind", "shape", "vox", "centre_um", "view", "edge", "length_um", "local_radius_um",
    "length_over_radius", "distance_over_tolerance", "distance_to_tree_um", "share_on_centreline",
    "inner_on_graph_centreline_off_tree", "drawn_path", "outer_at_wall", "deepest_over_radius",
    "inside_region_share", "touches_centreline", "edge_length_um", "taper", "from_tree", "edge_ends_degree",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="e14_run.py --capture's finish_input.pkl")
    parser.add_argument("--graph", required=True, help="the graph build_network saved")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from scipy.spatial import cKDTree

    from haemolynx.graph import IS_ZERO_RESISTANCE, diagnose_thick_vessel_bridges
    from haemolynx.graph.thick_vessel_bridge_shapes import UNINTENDED_SHAPES
    from haemolynx.graph.thick_vessel_junctions import CENTRELINE_TOLERANCE_PER_RADIUS

    with open(args.capture, "rb") as f:
        cap = pickle.load(f)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    vs = np.asarray(cap["voxel_size_zyx"], float)
    mask = np.asarray(cap["binary_mask"], dtype=bool)
    region = np.asarray(cap["thick_vessel_mask"], dtype=bool)
    tree_points = np.asarray(cap["thick_tree"], float) * vs
    tree_radius = np.nan_to_num(np.asarray(cap["thick_tree_radius"], float), posinf=0.0)
    floor = float(cap["settings"].get("thick_bridge_centreline_tolerance_um", 1.5))
    tolerance = np.maximum(floor, CENTRELINE_TOLERANCE_PER_RADIUS * tree_radius)
    tree = cKDTree(tree_points)
    rng = random.Random(SEED)

    report = diagnose_thick_vessel_bridges(
        G, region, cap["thick_tree"], mask, voxel_size_zyx=tuple(vs), tree_radius_um=cap["thick_tree_radius"],
        centreline_tolerance_um=floor,
    )
    bridges = []
    for row in report["bridges"]:
        u, v, key = row["u"], row["v"], row["key"]
        piece = _edge_path(G, u, v, G.edges[u, v, key])
        _d, nearest = tree.query(np.asarray(G.nodes[row["inner_node"]]["pos"], float))
        bridges.append(dict(
            row, kind="bridge", edge=(u, v, key), piece=piece, centre=piece[len(piece) // 2],
            distance_over_tolerance=row["inner_to_tree_um"] / float(tolerance[nearest]),
            distance_to_tree_um=row["inner_to_tree_um"],
        ))
    misses = near_misses(G, region, tree, tree_radius, tolerance, vs)
    # Every bridge of a shape the split never means to make, and every one
    # drawn round a bend, is shown; the rest are spread.
    unusual = [i for i, b in enumerate(bridges) if b["shape"] in UNINTENDED_SHAPES or b["drawn_path"]]
    picked = spread(bridges, TARGET_BRIDGES, rng, must=unusual) + spread(misses, TARGET_NEAR_MISSES, rng)
    rng.shuffle(picked)
    for i, case in enumerate(picked, start=1):
        case["id"] = i
        case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
        case["axis"] = _plane(case, tree_points)
        case["view"] = PLANES[case["axis"]][0]
        case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])

    paths = []
    for u, v, d in G.edges(data=True):
        paths.append((_edge_path(G, u, v, d), bool(d.get(IS_ZERO_RESISTANCE))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = [picked[k : k + PER_SHEET] for k in range(0, len(picked), PER_SHEET)]
    for number, cases in enumerate(sheets, start=1):
        first, last = cases[0]["id"], cases[-1]["id"]
        _sheet(cases, out / f"grid_{number}.png", mask, region, tree_points, paths, vs,
               f"Thick-vessel bridges and the pieces beside them, E14.5 (sheet {number} of {len(sheets)}, "
               f"IDs {first}-{last})")
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in picked:
            writer.writerow({k: case.get(k, "") for k in KEY_FIELDS})
    with open(out / "cases.json", "w", encoding="utf-8") as f:
        json.dump([{**{k: case.get(k) for k in KEY_FIELDS}, "piece_um": np.round(case["piece"], 3).tolist()}
                   for case in picked], f, default=str)
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mark"])
        writer.writerows([case["id"], ""] for case in sorted(picked, key=lambda c: c["id"]))
    print(f"{len(bridges)} bridges and {len(misses)} near-misses found; "
          f"{sum(c['kind'] == 'bridge' for c in picked)} and {sum(c['kind'] == 'near_miss' for c in picked)} "
          f"drawn on {len(sheets)} sheet(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
