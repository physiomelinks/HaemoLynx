#!/usr/bin/env python3
"""PLAN_one_centreline_per_lumen.md, the stub prune's protrusion test (decision
of 2026-10-11; ``PLAN_clean_graph.md`` 1b; RELEASE_REVIEW ``graph.6d``): a
marked grid of dead ends the test removes or keeps, so a redesigned rule can
be fitted on what Harvey calls a vessel.

The test (``prune.prune_vascular_stubs``, ``STUB_PROTRUSION_RADII`` = 1) prunes a
dead end that never gets more than one of its own radii outside the lumen of
another centreline, whatever its length. It reads only the 16 nearest
centreline samples with the stub's own masked out, so a long stub sees only
its own and is never judged; asking for 16 samples of *other* centrelines
pruned vessels' collinear ends at image faces. The redesign the plan names
(exempt collinear continuations, or stubs at the face) is to be fitted here.

The pool is every dead end the build's stub prunes were handed on E14.5
(``prune_spy.py`` beside this script, installed round a ``run_pipeline_stages``
run: each call's graph and, per chain, both protrusion readings), once each, that the prune's other rules leave to the
protrusion test -- at least its length threshold long or within it of a face,
at least half in the mask, its tip further than the parent's radius + 1 um from
its junction -- and that either reading puts under 2 radii. The grid spreads
over that reading (``ratio_others``), how far the stub turns from running
straight on from a vessel at its junction (``deviation_deg``) and whether its
tip is within its length threshold of an image face, round-robin, then
shuffled. Each case is drawn on the segmentation in its own plane, with the
graph as that prune call was handed it; the measures and today's verdicts are
in ``key.csv``, never on the sheets.

    python scripts/stub_protrusion_review/build_grid_s.py --spy outputs/oc/e14_on/prune_spy \\
        --run outputs/oc/e14_on --out outputs/stub_protrusion_review \\
        --page-data outputs/stub_protrusion_review/page.json

*--run* is the run's own directory: ``mask.npy`` (the segmentation as
``build_network`` binarises it), ``thick_vessel_mask.npy`` and
``thick_vessel_centreline.pkl`` when it has them, and ``results.json`` with its
``voxel_size_zyx``. ``grid_page.py`` turns ``--page-data`` into the page.
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
sys.path.insert(0, str(HERE.parent / "thick_bridge_review"))
sys.path.insert(0, str(HERE.parent / "thick_junction_review"))

SEED = 20261026
COUNT = 30
#: Either protrusion reading under this many tip radii puts a stub in the pool.
POOL_MAX_RATIO = 2.0
RATIO_BINS = (-np.inf, 0.5, 1.0, 1.5, np.inf)
#: A stub turning this little from a vessel at its junction runs straight on from it.
COLLINEAR_DEG = 30.0
#: Two records are one stub when tip and junction are both this near (um).
SAME_STUB_UM = 1.0
#: Leave out a stub this near (um) one on an earlier dead-end grid.
EXCLUDE_NEAR_UM = 6.0

QUESTION = (
    "Is the red dead end a vessel?   K = keep: a vessel (a branch, or a vessel running on to its end or out of "
    "the image)   R = remove: not a vessel (a hair along another vessel's wall, a bump, a spur inside a lumen)   "
    "? = unsure"
)
MARKS = (("K", "Keep: a vessel"), ("R", "Remove: not a vessel"), ("?", "Unsure"))
LEGEND = (
    "grey = segmentation (a slab 6 um either side of the plane), orange = thick-vessel region's edge, "
    "yellow dots = its centreline tree, green = the graph's vessels as the prune saw them, magenta dashes = bridges, "
    "red = the dead end judged, white cross = its junction, white bar = 10 um"
)
KEY_FIELDS = (
    "id", "vox", "centre_um", "view", "tip_um", "junction_um", "call", "caller", "length_um", "tip_radius_um",
    "parent_radius_um", "threshold_um", "ratio_capped", "ratio_others", "deviation_deg", "face_um", "at_face",
    "capped_removes", "others_removes", "removed_by_call", "junction_degree", "chain_edges",
)


def _bin(value, bins) -> int:
    return int(np.searchsorted(bins, value, side="right")) - 1


def read_records(spy: Path) -> list[dict]:
    with open(spy / "records.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool(records: list[dict]) -> list[dict]:
    """One record per stub the protrusion test decides: the call that removed
    it if one did, else the last that saw it."""
    candidates = []
    for r in records:
        if r["half_off_mask"] or r["tip_near_junction"]:
            continue  # the mask rules decide these first
        threshold = r["threshold_um"] or 0.0
        r["at_face"] = r["face_um"] <= threshold
        if r["length_um"] < threshold and not r["at_face"]:
            continue  # the length rule removes it whatever the protrusion
        ratios = [x for x in (r["ratio_capped"], r["ratio_others"]) if x is not None]
        if not ratios or min(ratios) >= POOL_MAX_RATIO:
            continue
        candidates.append(r)
    stubs: list[dict] = []
    for r in candidates:
        tip, junction = np.asarray(r["tip"]), np.asarray(r["junction"])
        for s in stubs:
            if (np.linalg.norm(np.asarray(s["tip"]) - tip) <= SAME_STUB_UM
                    and np.linalg.norm(np.asarray(s["junction"]) - junction) <= SAME_STUB_UM):
                if not s["removed"]:
                    s.update(r)  # the later sighting, or the call that removed it
                break
        else:
            stubs.append(dict(r))
    for s in stubs:
        s["capped_removes"] = s["ratio_capped"] is not None and s["ratio_capped"] < 1.0
        s["others_removes"] = s["ratio_others"] is not None and s["ratio_others"] < 1.0
        s["removed_by_call"] = s["removed"]
    return stubs


def cell(stub: dict) -> tuple:
    ratio = stub["ratio_others"] if stub["ratio_others"] is not None else np.inf
    deviation = stub["deviation_deg"] if stub["deviation_deg"] is not None else 180.0
    return (_bin(min(ratio, 99.0), RATIO_BINS), deviation < COLLINEAR_DEG, bool(stub["at_face"]))


def spread(stubs: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Round-robin over the cells, each cell's stubs shuffled."""
    cells: dict = {}
    for stub in stubs:
        cells.setdefault(cell(stub), []).append(stub)
    for members in cells.values():
        rng.shuffle(members)
    order = sorted(cells)
    picked = []
    while len(picked) < count and any(cells[c] for c in order):
        for c in order:
            if cells[c] and len(picked) < count:
                picked.append(cells[c].pop())
    return picked


def _excluded(centre, others) -> bool:
    return bool(len(others)) and float(np.min(np.linalg.norm(others - centre, axis=1))) < EXCLUDE_NEAR_UM


def _read_key_centres(path: Path, vs) -> np.ndarray:
    """Centres (um) from an earlier grid's key: ``centre_um``, ``tip_um`` or
    a voxel position."""
    centres = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for name, scale in (("centre_um", None), ("tip_um", None), ("tip_zyx_um", None),
                                ("vox", vs), ("tip_vox", vs), ("tip_zyx_vox", vs)):
                text = row.get(name)
                if not text:
                    continue
                values = [float(v) for v in text.replace("(", " ").replace(")", " ").replace("[", " ")
                          .replace("]", " ").replace(",", " ").replace(";", " ").split()]
                if len(values) == 3:
                    centres.append(np.asarray(values) * (1.0 if scale is None else np.asarray(scale)))
                    break
    return np.asarray(centres, float).reshape(-1, 3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spy", required=True, help="prune_spy.py's directory: records.jsonl, call_*.pkl")
    parser.add_argument("--run", required=True, help="the run's directory: mask.npy, thick_vessel_mask.npy, "
                                                      "thick_vessel_centreline.pkl, results.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-key", action="append", default=[], help="an earlier grid's key (repeatable)")
    parser.add_argument("--page-data", help="also write the cases as the interactive page's JSON (grid_page.py)")
    args = parser.parse_args(argv)

    import build_grid as phase5
    import phase1_rules_grid as page

    from haemolynx.graph import IS_ZERO_RESISTANCE, prune

    spy, run, out = Path(args.spy), Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    vs = np.asarray(results["voxel_size_zyx"], float)
    mask = np.load(run / "mask.npy")
    region = np.load(run / "thick_vessel_mask.npy") if (run / "thick_vessel_mask.npy").exists() else np.zeros_like(mask)
    tree_points = np.zeros((0, 3))
    if (run / "thick_vessel_centreline.pkl").exists():
        with open(run / "thick_vessel_centreline.pkl", "rb") as f:
            tree = pickle.load(f)
        tree_points = np.asarray(getattr(tree, "points", tree), float).reshape(-1, 3)
        if tree_points.size and tree_points.max() <= max(mask.shape):  # voxel indices
            tree_points = tree_points * vs

    others = np.vstack([_read_key_centres(Path(k), vs) for k in args.exclude_key]) if args.exclude_key else np.zeros((0, 3))
    stubs = pool(read_records(spy))
    kept = [s for s in stubs if not _excluded(np.asarray(s["tip"]), others)]
    rng = random.Random(SEED)
    picked = spread(kept, COUNT, rng)
    rng.shuffle(picked)

    graphs: dict = {}

    def paths_of(call: int):
        if call not in graphs:
            with open(spy / f"call_{call:03d}.pkl", "rb") as f:
                G = pickle.load(f)
            graphs[call] = [(phase5._edge_path(G, u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
                            for u, v, d in G.edges(data=True) if u != v]
        return graphs[call]

    for i, stub in enumerate(picked, start=1):
        stub["id"] = f"S{i:02d}"
        with open(spy / f"call_{stub['call']:03d}.pkl", "rb") as f:
            G = pickle.load(f)
        # The dead end at the recorded tip, in the graph that call was handed.
        ends = [n for n in G.nodes if G.degree(n) == 1]
        tip = min(ends, key=lambda n: float(np.linalg.norm(np.asarray(G.nodes[n]["pos"], float) - stub["tip"])))
        _nodes, edges = prune._chain(G, tip)
        piece = prune._chain_path(G, edges)
        stub["piece"] = piece
        stub["centre"] = piece[len(piece) // 2] if len(piece) > 2 else np.mean(piece, axis=0)
        stub["tick"] = piece[0]
        stub["vox"] = tuple(int(round(x)) for x in stub["centre"] / vs)
        stub["axis"] = phase5._plane(dict(stub, piece=piece), tree_points)
        stub["view"] = phase5.PLANES[stub["axis"]][0]
        stub["centre_um"] = tuple(round(float(x), 2) for x in stub["centre"])
        stub["tip_um"] = tuple(round(float(x), 2) for x in stub["tip"])
        stub["junction_um"] = tuple(round(float(x), 2) for x in stub["junction"])

    _sheet(picked, out / "grid_S.png", mask, region, tree_points, paths_of, vs,
           f"E14.5, grid S (dead ends and the protrusion test, IDs S01-S{len(picked):02d})   " + QUESTION)
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for stub in sorted(picked, key=lambda s: s["id"]):
            writer.writerow({k: _plain(stub.get(k)) for k in KEY_FIELDS})
    with open(out / "pool.csv", "w", newline="", encoding="utf-8") as f:
        fields = [k for k in KEY_FIELDS if k not in ("id", "vox", "centre_um", "view", "tip_um", "junction_um")]
        writer = csv.DictWriter(f, fieldnames=["excluded", "tip", "junction"] + fields, extrasaction="ignore")
        writer.writeheader()
        for stub in stubs:
            row = {k: _plain(stub.get(k)) for k in fields}
            row.update(excluded=_excluded(np.asarray(stub["tip"]), others),
                       tip=_plain(np.asarray(stub["tip"])), junction=_plain(np.asarray(stub["junction"])))
            writer.writerow(row)
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mark"])
        writer.writerows([s["id"], ""] for s in sorted(picked, key=lambda s: s["id"]))
    summary = {
        "records": len(read_records(spy)), "pool": len(stubs), "after_exclusion": len(kept), "picked": len(picked),
        "pool_cells": _count(str(cell(s)) for s in kept), "picked_cells": _count(str(cell(s)) for s in picked),
        "picked_capped_removes": sum(bool(s["capped_removes"]) for s in picked),
        "picked_others_removes": sum(bool(s["others_removes"]) for s in picked),
        "picked_removed_by_call": sum(bool(s["removed_by_call"]) for s in picked),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if args.page_data:
        cases = []
        for stub in sorted(picked, key=lambda s: s["id"]):
            view = page.case_view(dict(stub, rule="stub", route=None), mask, region, tree_points,
                                  paths_of(stub["call"]), vs)
            cases.append(view)
        Path(args.page_data).write_text(json.dumps({
            "grid": "S",
            "dataset": "E14.5 MCA, Harvey's Claudin segmentation",
            "voxel_size_zyx": [float(v) for v in vs],
            "slab_um": 2 * phase5.HALF_SLAB_UM,
            "sheets": [{"rule": "stub", "title": "Dead ends", "question": QUESTION,
                        "marks": [{"key": k, "label": label} for k, label in MARKS], "cases": cases}],
        }, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def _sheet(cases, out_png, mask, region, tree_points, paths_of, vs, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import phase1_rules_grid as page

    cols = 6
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.75 * rows + 1.6), squeeze=False)
    fig.suptitle(f"{title}\n{LEGEND}", fontsize=10)
    for ax, case in zip(axes.ravel(), sorted(cases, key=lambda c: c["id"])):
        page._draw_case(ax, case, mask, region, tree_points, paths_of(case["call"]), vs)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.6 / (3.75 * rows + 1.6), wspace=0.04,
                        hspace=0.16)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def _plain(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    if isinstance(value, np.ndarray):
        return tuple(round(float(x), 2) for x in value)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (float, int)):
        return tuple(round(float(x), 2) for x in value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
