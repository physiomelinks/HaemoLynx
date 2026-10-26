#!/usr/bin/env python3
"""The paths the thin-arm join drew (``preprocessing.thick_vessels.
_join_thin_arms_to_fat_ridge``), each measured by how its arm meets it: the
measures tried on the connections near thick vessels that should not exist
(PLAN_thick_vessel_bridges.md, "connections that should not exist";
``docs/notes/thick_vessels.md``).

The join gives every thin skeleton component one path to the thick
centreline, from the component's voxel nearest it. Lee's thin skeleton stops
short of the thick region (``skeleton_thick_vessel_flake_filter_um``), so a
vessel opening into the thick one ends there in a tip pointing at it, and
the join carries it on; a vessel running past the thick one has its nearest
voxel on its side. Each path is measured by the arm's own direction over the
stretch before its start against the path's chord (``approach_cos``, 1
heading straight in, 0 running across) and by the arm's voxels beside the
start (``start_degree``: 1 its tip, 0 an arm of one voxel). On E14.5 neither
separated Harvey's marks.

    python scripts/thick_bridge_review/join_paths.py --run <run> --out <folder>

``--run`` holds ``skel_before_join.npz`` and ``skel_after_join.npz`` (the
thickness-gated skeleton before and after the join, packed bits with their
shape), ``tree.npy`` (the tree's voxels) and ``voxel_size_zyx.json``, as for
``connection_steps.py``. Writes ``joins.csv`` and ``joins.pkl``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: The arm's direction is read over its voxels this far (um) from the join's start.
ARM_REACH_UM = 6.0

_N26 = np.ones((3, 3, 3), dtype=bool)
_OFFSETS = np.array([o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]) - 1


def _neighbours(voxel: np.ndarray, members: set) -> list[tuple]:
    return [tuple(n) for n in (voxel + _OFFSETS).tolist() if tuple(n) in members]


def join_paths(before: np.ndarray, after: np.ndarray, tree: np.ndarray) -> list[dict[str, Any]]:
    """Each path the join drew (a 26-connected piece of *after* not in
    *before*), ordered from its start -- the voxel beside the thin arm it
    joined -- to its end beside the tree (or the skeleton inside the thick
    region). ``{"path": (n, 3) voxels, "start": arm voxel, "end": tree voxel}``;
    a piece touching no arm or no tree is left out."""
    before = np.asarray(before, bool)
    tree = np.asarray(tree, bool)
    lee = before & ~tree
    drawn = np.asarray(after, bool) & ~before
    labels, count = ndimage.label(drawn, structure=_N26)
    paths = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        lo = np.array([max(0, s.start - 1) for s in box])
        hi = np.array([min(d, s.stop + 1) for s, d in zip(box, drawn.shape)])
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        piece = np.argwhere(labels[crop] == index)
        members = {tuple(v) for v in piece.tolist()}
        arm = {tuple(v) for v in np.argwhere(lee[crop]).tolist()}
        ridge = {tuple(v) for v in np.argwhere(before[crop] & ~lee[crop]).tolist()}
        starts = [v for v in members if _neighbours(np.array(v), arm)]
        ends = [v for v in members if _neighbours(np.array(v), ridge)]
        if not starts or not ends:
            continue
        graph = nx.Graph()
        graph.add_nodes_from(members)
        for v in members:
            for n in _neighbours(np.array(v), members):
                graph.add_edge(v, n, weight=float(np.linalg.norm(np.subtract(v, n))))
        best = None
        for s in starts:
            lengths, routes = nx.single_source_dijkstra(graph, s)
            for e in ends:
                if e in lengths and (best is None or lengths[e] > best[0]):
                    best = (lengths[e], routes[e])
        if best is None:
            continue
        route = np.array(best[1]) + lo
        start = np.array(_neighbours(route[0] - lo, arm)[0]) + lo
        end = np.array(_neighbours(route[-1] - lo, ridge)[0]) + lo
        paths.append({"path": route, "start": start, "end": end})
    return paths


def arm_voxels(lee: np.ndarray, start: Sequence[int], voxel_size_zyx: Sequence[float],
               reach_um: float = ARM_REACH_UM) -> np.ndarray:
    """The thin arm's voxels within *reach_um* of *start* along the arm
    (its 26-connected Lee skeleton, walked from *start*)."""
    spacing = np.asarray(voxel_size_zyx, float)
    start = tuple(int(v) for v in start)
    shape = np.asarray(lee.shape)
    seen = {start: 0.0}
    frontier = [start]
    while frontier:
        nxt = []
        for v in frontier:
            for o in _OFFSETS:
                n = tuple(int(c) for c in np.add(v, o))
                if n in seen or np.any(np.array(n) < 0) or np.any(np.array(n) >= shape) or not lee[n]:
                    continue
                d = seen[v] + float(np.linalg.norm(o * spacing))
                if d <= reach_um:
                    seen[n] = d
                    nxt.append(n)
        frontier = nxt
    return np.array(sorted(seen), dtype=int)


def approach(path: dict[str, Any], lee: np.ndarray, voxel_size_zyx: Sequence[float],
             reach_um: float = ARM_REACH_UM) -> dict[str, float]:
    """How the arm meets the join's *path*: ``approach_cos``, |cos| of the
    angle between the arm's direction (the line through its voxels within
    *reach_um* of the start) and the path's chord; ``start_degree``, the
    arm's voxels beside the start (1 = its tip); and the chord's length."""
    spacing = np.asarray(voxel_size_zyx, float)
    start = np.asarray(path["start"], int)
    chord = (np.asarray(path["end"], float) - start) * spacing
    arm = arm_voxels(lee, start, spacing, reach_um)
    degree = sum(1 for o in _OFFSETS if _in(lee, start + o))
    cos = np.nan
    if len(arm) >= 3 and np.linalg.norm(chord) > 0:
        centred = (arm - arm.mean(axis=0)) * spacing
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        cos = float(abs(axis @ chord) / np.linalg.norm(chord))
    return {"approach_cos": cos, "start_degree": int(degree), "chord_um": float(np.linalg.norm(chord)),
            "arm_voxels": int(len(arm))}


def _in(volume: np.ndarray, voxel: np.ndarray) -> bool:
    return bool(np.all(voxel >= 0) and np.all(voxel < volume.shape) and volume[tuple(voxel)])


def main(argv: list[str] | None = None) -> int:
    from thick_bridge_review.connection_steps import _unpacked

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = np.asarray(json.loads((run / "voxel_size_zyx.json").read_text()), float)
    before = _unpacked(run / "skel_before_join.npz")
    after = _unpacked(run / "skel_after_join.npz")
    tree = np.zeros_like(before)
    tree[tuple(np.load(run / "tree.npy").T)] = True
    lee = before & ~tree
    rows = []
    for i, path in enumerate(join_paths(before, after, tree), start=1):
        row = {"join": i, "start": tuple(int(v) for v in path["start"]), "end": tuple(int(v) for v in path["end"]),
               "voxels": len(path["path"])}
        row.update(approach(path, lee, vs))
        row["path_um"] = json.dumps(np.round(path["path"] * vs, 2).tolist())
        rows.append(row)
    with open(out / "joins.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "joins.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"{len(rows)} join paths -> {out / 'joins.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
