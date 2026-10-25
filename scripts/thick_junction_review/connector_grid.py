#!/usr/bin/env python3
"""PLAN_bridging.md, after grid P: every connector the thick-vessel split
leaves unbridged on E14.5 -- a vessel running into the thick-vessel region
and on to its centreline tree, kept as vessel because no thick centreline in
the graph lay within its bridge's reach -- for Harvey to mark (his call,
2026-10-25: "grid of the 9").

Two things in the split decide which connectors these are, and neither is
changed until the marks say so:

- **Where the reach is read**: a bridge reaches ``2 r + 5`` um, ``r`` the
  lumen radius at the tree point nearest the exit (``_bridge_reach_um``).
  Read at a thin tree spur, the reach falls short of a wide thick vessel.
- **One exit sinks the connector**: a connector with several exits keeps no
  bridge if any one of them has nothing in reach.

The same capture and drawing as grid P (``phase1_rules_grid.py``), the graph
as the split is handed it, each case's connector stretch in red and the
bridge it would get in cyan (to where it met the thick centreline, else the
nearest thick-centreline sample), shuffled, IDs K01... The measures (both
causes among them) are in ``key_connector.csv`` only.

    python scripts/thick_junction_review/connector_grid.py \\
        --capture outputs/br/mine/finish_input.pkl \\
        --out outputs/thick_junction_review/grid_K \\
        --page-data outputs/thick_junction_review/grid_K/page.json
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import json
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from thick_junction_review import phase1_rules_grid as grid
except ImportError:  # run as a script, from its own folder
    import phase1_rules_grid as grid

SEED = 20261027
#: A tree radius at the exit under this share of the widest within reach
#: is a spur's: the reach is read off a thin side twig of the tree.
SPUR_SHARE = 0.5
QUESTION = (
    "Connectors: the red vessel runs into the big vessel and was left without a bridge.\n"
    "B = bridge it to the big vessel's centreline (cyan dashes: where the bridge would go)   "
    "N = leave it as it is   ? = unsure"
)
MARKS = (("B", "Bridge it to the big vessel's centreline"), ("N", "Leave it as it is"), ("?", "Unsure"))
KEY_FIELDS = ("id", "rule", "vox", "centre_um", "view", "exits", "failed_exits", "tree_radius_at_exit_um",
              "widest_tree_radius_near_um", "spur", "reach_um", "to_centreline_um", "inside_um")


def by_distance(points, where) -> list[tuple[float, np.ndarray]]:
    """``(distance, point)`` for each of *points*, nearest *where* first.
    Ties keep their order: the points themselves are never compared (a
    connector's contacts often hold one sample twice)."""
    return sorted(((float(np.linalg.norm(np.asarray(p, float) - where)), p) for p in points),
                  key=lambda m: m[0])


@contextlib.contextmanager
def failed_connectors_recorded(found: list, *, active=lambda: True):
    """While open, every connector ``thick_vessel_junctions._choose_bridges``
    finds no bridge for (while *active()*) is appended to *found*: its
    stretches inside the region, and each of its exits with the tree radius
    its reach is read from, the reach, how far the thick centreline is, and
    where its bridge would go."""
    from haemolynx.graph import thick_vessel_junctions as tvj

    real = tvj._choose_bridges

    @functools.wraps(real)
    def choose(sampled, inside_runs, find, bridged, tree, tree_radius, edges_at, node_on_centreline,
               node_pos, result, cand_tree, cand_index, contacts, through):
        out = real(sampled, inside_runs, find, bridged, tree, tree_radius, edges_at, node_on_centreline,
                   node_pos, result, cand_tree, cand_index, contacts, through)
        if not out[1] or not active():
            return out
        dropped_from = tvj._decide_what_stays(sampled, inside_runs, find, bridged, through)
        origins = tvj._bridge_origins(sampled, edges_at, node_on_centreline, result)
        for root in out[1]:
            runs = [np.asarray(sampled[i].points[s:t + 1], float)
                    for r, (i, s, t) in enumerate(inside_runs) if find(("run", r)) == root]
            exits = []
            for origin, beside in origins:
                if root not in {dropped_from[c] for c in beside if c in dropped_from}:
                    continue
                where = np.asarray(
                    node_pos[origin[1]] if origin[0] == "node" else sampled[origin[1][0]].points[origin[1][1]],
                    float,
                )
                radius = float(tree_radius[tree.query(where)[1]])
                reach = tvj._bridge_reach_um(radius)
                met = by_distance([sampled[i].points[k] for i, k in contacts.get(root, ())], where)
                near = None if cand_tree is None else cand_tree.query(where)
                to_centreline = min([m[0] for m in met] + ([float(near[0])] if near is not None else []),
                                    default=float("inf"))
                target = met[0][1] if met else (
                    None if near is None else sampled[int(cand_index[near[1]][0])].points[int(cand_index[near[1]][1])]
                )
                around = tree_radius[tree.query_ball_point(where, reach + 10.0)]
                exits.append(dict(
                    where=where, tree_radius_um=radius, reach_um=reach, to_centreline_um=to_centreline,
                    target=None if target is None else np.asarray(target, float),
                    widest_near_um=float(np.max(around)) if len(around) else radius,
                    fails=not to_centreline <= reach,
                ))
            found.append(dict(runs=runs, exits=exits))
        return out

    tvj._choose_bridges = choose
    try:
        yield found
    finally:
        tvj._choose_bridges = real


def connector_case(record: dict) -> dict:
    """One connector's case for the grid: centred on its first failing exit,
    the inside stretch nearest that exit, the bridge that exit would get,
    and the measures of both causes."""
    exits = record["exits"]
    failing = [e for e in exits if e["fails"]] or exits
    first = failing[0]
    runs = [r for r in record["runs"] if len(r)]
    piece = min(runs, key=lambda r: float(np.min(np.linalg.norm(r - first["where"], axis=1))))
    inside = sum(float(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1))) for r in runs)
    return dict(
        rule="connector", centre=first["where"], piece=piece,
        route=None if first["target"] is None else np.vstack([first["where"], first["target"]]),
        exits=len(exits), failed_exits=sum(e["fails"] for e in exits),
        tree_radius_at_exit_um=first["tree_radius_um"], widest_tree_radius_near_um=first["widest_near_um"],
        spur=first["tree_radius_um"] < SPUR_SHARE * first["widest_near_um"],
        reach_um=first["reach_um"], to_centreline_um=first["to_centreline_um"], inside_um=inside,
    )


def replay(capture: Path) -> dict:
    """``finish_network_graph`` on *capture*, recording the connectors the
    build's own split leaves, and the graph that split is handed."""
    from haemolynx import graph
    from haemolynx.pipeline import stages

    with open(capture, "rb") as f:
        cap = pickle.load(f)
    settings = dict(cap["settings"])
    context = stages.network_context(
        settings, cap["binary_mask"], cap["voxel_size_zyx"], cap["thick_vessel_mask"],
        thick_centreline=cap["thick_tree"],
    )
    seen: dict = {"in_build_split": False, "connectors": []}
    real_split = graph.insert_thick_vessel_junction_nodes

    @functools.wraps(real_split)
    def split(G, *a, **k):
        build = k.get("edges") is None and "split_in" not in seen
        if build:
            seen["split_in"] = G.copy()
        seen["in_build_split"] = build
        try:
            return real_split(G, *a, **k)
        finally:
            seen["in_build_split"] = False

    graph.insert_thick_vessel_junction_nodes = split
    try:
        with failed_connectors_recorded(seen["connectors"], active=lambda: seen["in_build_split"]):
            seen["final"] = stages.finish_network_graph(cap["G"].copy(), cap["skeleton"], settings, context)
    finally:
        graph.insert_thick_vessel_junction_nodes = real_split
    seen["cap"] = cap
    return seen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="e14_run.py --capture's finish_input.pkl (grid P's)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--page-data", help="also write the cases as the interactive page's JSON (grid_page.py)")
    args = parser.parse_args(argv)

    import build_grid as phase5

    from haemolynx.graph import IS_ZERO_RESISTANCE

    seen = replay(Path(args.capture))
    cap = seen["cap"]
    vs = np.asarray(cap["voxel_size_zyx"], float)
    mask = np.asarray(cap["binary_mask"], bool)
    region = np.asarray(cap["thick_vessel_mask"], bool)
    tree_points = np.asarray(cap["thick_tree"], float) * vs

    cases = [connector_case(record) for record in seen["connectors"] if record["exits"]]
    random.Random(SEED).shuffle(cases)
    for i, case in enumerate(cases, start=1):
        case["id"] = f"K{i:02d}"
        case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
        shown = case["piece"] if case.get("route") is None else np.vstack([case["piece"], case["route"]])
        case["axis"] = phase5._plane(dict(case, piece=shown), tree_points)
        case["view"] = phase5.PLANES[case["axis"]][0]
        case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drawn = seen["split_in"]
    paths = [(phase5._edge_path(drawn, u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
             for u, v, d in drawn.edges(data=True) if u != v]
    if cases:
        grid._sheet(cases, out / "grid_K_connector.png", mask, region, tree_points, paths, vs,
                    f"E14.5, grid K (connectors, IDs {cases[0]['id']}-{cases[-1]['id']})   " + QUESTION)
    with open(out / "key_connector.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in sorted(cases, key=lambda c: c["id"]):
            writer.writerow({k: grid._plain(case.get(k)) for k in KEY_FIELDS})
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "rule", "mark"])
        writer.writerows([c["id"], "connector", ""] for c in sorted(cases, key=lambda c: c["id"]))
    if args.page_data:
        Path(args.page_data).write_text(json.dumps({
            "grid": "K",
            "dataset": "E14.5 MCA, Harvey's Claudin segmentation (grid K: connectors left unbridged)",
            "voxel_size_zyx": [float(v) for v in vs],
            "slab_um": 2 * phase5.HALF_SLAB_UM,
            "sheets": [{
                "rule": "connector", "title": "Connectors", "question": QUESTION.split("\n")[0],
                "marks": [{"key": k, "label": label} for k, label in MARKS],
                "cases": [grid.case_view(case, mask, region, tree_points, paths, vs) for case in cases],
            }],
        }, separators=(",", ":")), encoding="utf-8")
    summary = {
        "connectors": len(cases),
        "spur_reach": sum(bool(c["spur"]) for c in cases),
        "more_than_one_exit": sum(c["exits"] > 1 for c in cases),
        "sunk_by_one_exit": sum(c["exits"] > c["failed_exits"] for c in cases),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
