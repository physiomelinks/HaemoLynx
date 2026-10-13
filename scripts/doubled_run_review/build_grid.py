#!/usr/bin/env python3
"""Build grid H of PLAN_clean_graph.md: is a doubled run one vessel drawn twice?

Thirty stretches of centreline that the lumen-artefact report counts as
running beside another centreline in the same lumen
(``preprocessing.shadowed_samples``), from one built graph, following
CLAUDE.md's "Marked example grids": ten each from a dead-end chain, the
middle of an edge and a fork at a junction, spread across the run's length
in local lumen radii, then shuffled. Each is drawn in its own plane on the
segmentation alone (what graph building sees): the run red, the rest of its
own centreline orange, the centreline it runs beside blue, the chord between
them at the run's middle yellow, other centrelines grey dots, a 5 um scale
bar. The measures stay off the page, in the key CSV beside it (with the
run's voxel position, for napari).

Marks: D = one vessel drawn twice (resolve it), T = two vessels, or a branch
leaving its parent (leave both), ? = unsure.

Usage::

    python scripts/doubled_run_review/build_grid.py --graph final.pkl --mask mask.npy
        --voxel-size-zyx 1 0.98 0.98 --out outputs/doubled_run_review
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dead_end_review"))
SEED = 20261012
PER_PLACE = 10
PLACES = ("dead-end chain", "mid-edge", "fork")
#: ``(from, to, how many)`` by run length, microns, per place.
BANDS = ((1.0, 2.5, 4), (2.5, 5.0, 3), (5.0, float("inf"), 3))


def doubled_runs(G, support):
    """Every run of consecutive samples (a gap of one allowed) that run
    beside another centreline in the same lumen, with the points of the run
    and of the centreline it runs beside."""
    from scipy.spatial import cKDTree

    from haemolynx.graph._helpers import EdgeSampleIndex, edge_id, edge_sample_points
    from haemolynx.preprocessing.bridge_mask_support import (
        MAX_SHADOW_COSINE, SAME_LUMEN_MARGIN_UM, _densify, _judged_samples, _same_lumen_neighbours,
        _sample_step,
    )

    step = _sample_step(support.voxel_size_zyx)
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    index = EdgeSampleIndex(G)
    ids: dict = {}
    owners = np.asarray([ids.setdefault(o, len(ids)) for o in index.owners], dtype=np.intp)
    by_id = {v: k for k, v in ids.items()}
    paths = {}
    runs = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in node_pos or v not in node_pos:
            continue
        eid = edge_id(u, v, key)
        own = ids.get(eid)
        path = edge_sample_points(u, v, data, node_pos)
        paths[eid] = (u, v, key, np.asarray(path, dtype=float))
        dense = _densify(path, step)
        if len(dense) < 2:
            continue
        reach = 2.0 * float(np.max(support.radius(dense))) + 1.0
        found = index.tree.query_ball_point(dense, r=reach)
        near = np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in found])) if len(found) else np.empty(0, dtype=np.intp)
        near = near[owners[near] != own]
        if not len(near):
            continue
        samples, tangents, judged = _judged_samples(path, support.radius, step_um=step, margin_um=SAME_LUMEN_MARGIN_UM)
        if not judged.any():
            continue
        pts = index.points[near]
        neighbour = _same_lumen_neighbours(
            samples[judged], tangents[judged], cKDTree(pts), pts, support.inside, support.radius,
            step_um=step, margin_um=SAME_LUMEN_MARGIN_UM, max_cosine=MAX_SHADOW_COSINE,
        )
        jsamples = samples[judged]
        beside = neighbour >= 0
        n = len(beside)
        i = 0
        while i < n:
            if not beside[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and (beside[j + 1] or (j + 2 < n and beside[j + 2])):
                j += 1
            hits = [k for k in range(i, j + 1) if neighbour[k] >= 0]
            partner_id = Counter(int(owners[near[neighbour[k]]]) for k in hits).most_common(1)[0][0]
            runs.append(dict(
                own=eid, partner=by_id[partner_id], length=(j - i + 1) * step,
                points=jsamples[i:j + 1], partner_points=pts[[neighbour[k] for k in hits]],
                at_start=i == 0, at_end=j == n - 1,
            ))
            i = j + 1
    return runs, paths


def _nodes(eid) -> set:
    return set(eid[0]) if isinstance(eid[0], frozenset) else {eid[0], eid[1]}


def place_of(G, run, chain_edges, bridges) -> str:
    own, partner = run["own"], run["partner"]
    if own in bridges or partner in bridges:
        return "bridge"
    if own in chain_edges or partner in chain_edges:
        return "dead-end chain"
    if (_nodes(own) & _nodes(partner)) and (run["at_start"] or run["at_end"]):
        return "fork"
    if run["at_start"] or run["at_end"]:
        return "edge end"
    return "mid-edge"


def _evenly(items: list, count: int, key) -> list:
    if len(items) <= count:
        return list(items)
    ordered = sorted(items, key=key)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(picks.tolist())]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True, help=".npy of the binary segmentation")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="H")
    args = parser.parse_args(argv)

    from build_grids import Drawer

    from haemolynx.graph import IS_ZERO_RESISTANCE
    from haemolynx.graph._helpers import edge_id
    from haemolynx.graph.dead_ends import chain_to_junction
    from haemolynx.preprocessing import MaskSupport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    mask = np.load(args.mask).astype(bool)
    vs = tuple(args.voxel_size_zyx)
    support = MaskSupport(mask, vs)
    runs, paths = doubled_runs(G, support)
    bridges = {edge_id(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get(IS_ZERO_RESISTANCE)}
    chain_edges = set()
    for tip in [n for n in G if G.degree(n) == 1]:
        _nodes_, edges = chain_to_junction(G, tip)
        chain_edges.update(edge_id(*e) for e in edges)
    for run in runs:
        run["place"] = place_of(G, run, chain_edges, bridges)
        # The widest lumen along the run and the centreline it runs beside:
        # at a junction's flare the middle of a run can sit at the wall.
        run["radius"] = float(np.max(support.radius(np.vstack([run["points"], run["partner_points"]]))))
        run["ratio"] = run["length"] / max(run["radius"], 1e-6)
    print("runs by place:", Counter(r["place"] for r in runs))

    # Runs are mostly under 2 um (the flare where vessels meet), so an even
    # spread by rank would show almost nothing long: per place, a fixed
    # share from each length band, spread by length within the band.
    chosen = []
    for place in PLACES:
        pool = [r for r in runs if r["place"] == place and r["length"] >= 1.0]
        picked = []
        for lo, hi, count in BANDS:
            band = [r for r in pool if lo <= r["length"] < hi and r not in picked]
            picked += _evenly(band, count, key=lambda r: (r["length"], r["ratio"]))
        if len(picked) < PER_PLACE:
            picked += _evenly([r for r in pool if r not in picked], PER_PLACE - len(picked), key=lambda r: r["length"])
        chosen += picked[:PER_PLACE]
    rng = random.Random(SEED)
    rng.shuffle(chosen)

    drawer = Drawer(G, mask, None, vs)
    examples, rows = [], []
    for number, run in enumerate(chosen, start=1):
        ident = f"{args.grid}-{number:02d}"
        u, v, key, own_path = paths[run["own"]]
        pu, pv, pkey, partner_path = paths[run["partner"]]
        pts = run["points"]
        mid = pts[len(pts) // 2]
        heading = pts[-1] - pts[0] if len(pts) > 1 else None
        if heading is not None and np.linalg.norm(heading) < 1e-6:
            heading = None
        # The partner's point nearest the run's middle: the chord drawn.
        partner_near = run["partner_points"][np.argmin(np.linalg.norm(run["partner_points"] - mid, axis=1))]
        window = 12.0 + run["length"]
        own_window = own_path[np.linalg.norm(own_path - mid, axis=1) <= window]
        partner_window = partner_path[np.linalg.norm(partner_path - mid, axis=1) <= window]
        lines = [
            dict(points=own_window if len(own_window) > 1 else own_path, color="#ff9f0a", width=2.0),
            dict(points=partner_window if len(partner_window) > 1 else partner_path, color="#2a7fff", width=2.4),
            dict(points=pts if len(pts) > 1 else np.vstack([pts, pts + 0.3]), color="#ff3b30", width=3.2),
            dict(points=[mid, partner_near], color="#ffd60a", width=1.8),
        ]
        focus = np.vstack([pts, run["partner_points"], mid[None]])
        example = drawer.draw(focus, heading, lines, half_thickness=float(np.clip(1.5 * run["radius"], 2.0, 5.0)),
                              marker=mid)
        examples.append({"id": ident, **example})
        rows.append({
            "id": ident, "grid": args.grid, "place": run["place"], "run_length_um": round(run["length"], 2),
            "lumen_radius_um": round(run["radius"], 2), "length_over_radius": round(run["ratio"], 2),
            "at_edge_end": bool(run["at_start"] or run["at_end"]),
            "own_edge": f"{u}-{v}-{key}", "partner_edge": f"{pu}-{pv}-{pkey}",
            "own_edge_length_um": round(float(np.linalg.norm(np.diff(own_path, axis=0), axis=1).sum()), 1),
            "partner_edge_length_um": round(float(np.linalg.norm(np.diff(partner_path, axis=0), axis=1).sum()), 1),
            "mid_zyx_vox": tuple(int(round(c)) for c in mid / np.asarray(vs)),
            "mid_zyx_um": tuple(round(float(c), 1) for c in mid),
        })

    template = (HERE.parent / "dead_end_review" / "template.html").read_text(encoding="utf-8")
    data = {
        "batch": args.grid,
        "title": f"Doubled Run Review {args.grid}: is this one vessel drawn twice?",
        "question": "Is the red stretch one vessel drawn twice?",
        "lede": ("The E14.5 segmentation (white = vessel) in the plane of one stretch of centreline (red) that the "
                 "report counts as running beside another centreline (blue) in the same lumen; the rest of the red "
                 "stretch's own centreline is orange, the chord between the two at the stretch's middle is yellow. "
                 "Both panels are the segmentation: graph building never sees the raw image here."),
        "steps": ["Look at whether the red and blue centrelines lie in one segmented lumen along the red stretch, "
                  "or in two vessels (or a branch leaving its parent) that only touch.",
                  "D if it is one vessel drawn twice along the red stretch: one of the two should go there.",
                  "T if they are two vessels, or a branch leaving its parent: both stay.",
                  "Add a note on what decided it, if it helps."],
        "legend": [{"label": "the stretch judged", "color": "#ff3b30"}, {"label": "the rest of its centreline", "color": "#ff9f0a"},
                   {"label": "the centreline it runs beside", "color": "#2a7fff"}, {"label": "the chord between them", "color": "#ffd60a"},
                   {"label": "other centrelines", "color": "#9a9a9a"}],
        "verdicts": [{"value": "D", "label": "One vessel drawn twice", "short": "D", "tone": "bad"},
                     {"value": "T", "label": "Two vessels / a branch leaving", "short": "T", "tone": "good"},
                     {"value": "?", "label": "Unsure", "short": "?", "tone": "unsure"}],
        "examples": examples,
    }
    html = (template.replace("__TITLE__", f"Doubled Run Review {args.grid}")
            .replace("__SECOND_PANEL__", "Segmentation (same slab)")
            .replace("__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    (out / f"{args.grid}_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / f"{args.grid}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"{args.grid}_all_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["place", "run_length_um", "lumen_radius_um", "length_over_radius", "at_edge_end", "mid_zyx_um"])
        for run in runs:
            mid = run["points"][len(run["points"]) // 2]
            writer.writerow([run["place"], round(run["length"], 2), round(run["radius"], 2), round(run["ratio"], 2),
                             bool(run["at_start"] or run["at_end"]), tuple(round(float(c), 1) for c in mid)])
    print(f"{args.grid}: {len(examples)} examples -> {out / (args.grid + '_review.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
