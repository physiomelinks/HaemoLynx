#!/usr/bin/env python3
"""Grid SW: every dead end the thick-vessel split left at a thick vessel's
wall in E14.5's surface layer, for Harvey to mark (2026-10-27: "mark which
should join the vein").

On E14.5 the thick region is one piece in the top 38 um, so all 57 of the
run's ``thick_wall`` dead ends lie in the surface layer. The thin-arm join
gives each connected piece of thin skeleton one opening into the thick
region; the split's wall join (``_join_dead_ends_at_walls``) is meant to
draw the rest on to the thick centreline, and refused these: turning more
than 80 degrees onto the nearest centreline point, closing a loop inside one
lumen, closing a loop under ``remove_short_loops_under_um`` round a small
hole, or out of its reach.

A ``capture_steps.py`` run (``thick_bridge_review``) is replayed from the
graph ``finish_network_graph`` was handed, as grid P was
(``phase1_rules_grid.replay``), and every dead end the run's graph calls
``thick_wall`` is drawn as grid P drew its wall cases: the graph the wall
join was handed, the dead end's last 25 um in red, the way the join would be
drawn in cyan dashes (through the mask to the nearest point of the graph's
thick centreline), never the join's answer. All of them are shown, shuffled;
the join's decision and every measure are in ``key_SW.csv`` only.

    python scripts/thick_junction_review/surface_wall_grid.py \\
        --capture outputs/wk/capture --run-graph outputs/wk/new/run/graph_returned.pkl \\
        --config config.yaml --out outputs/thick_junction_review/grid_SW
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "thick_bridge_review"))

import grid_page  # noqa: E402
import phase1_rules_grid as grid  # noqa: E402

SEED = 20261027
GRID = "SW"
SURFACE_UM = 40.0
#: A run-graph dead end this near (um) a case of the replayed pool is that case.
MATCH_UM = 3.0
PER_SHEET = 29
QUESTION = (
    "Should the red dead end be joined to the big vessel?   "
    "J = yes, it opens into the big vessel (cyan dashes: where the wall join would draw it; note it if it "
    "should land elsewhere)   V = no, it runs on to another vessel instead   N = no, leave it as a dead end   "
    "? = unsure"
)
MARKS = (("J", "Join it to the big vessel"), ("V", "It runs on to another vessel instead"),
         ("N", "Leave it as a dead end"), ("?", "Unsure"))
KEY_FIELDS = (
    "id", "vox", "centre_um", "view", "decision", "loop_reason", "loop_hole_um", "past_lumen_um",
    "lumen_radius_um", "chord_turn_deg", "route_turn_deg", "end_length_um", "to_centreline_um", "route_accepted",
    "chain_um", "tip_radius_um", "mask_run_um", "ahead_um", "ahead_turn_deg", "ahead_background_um",
    "thick_turn_deg", "run_node", "pool_node", "crop_number",
)


def unpacked(path: Path) -> np.ndarray:
    z = np.load(path)
    shape = tuple(int(s) for s in z["shape"])
    return np.unpackbits(z["bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)


def finish_input(capture: Path, config: Path) -> dict:
    """``capture_steps.py``'s run folder as ``phase1_rules_grid.replay`` takes
    it: the graph after the last topology step and what the finish reads."""
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline.schema import SCHEMA
    from haemolynx.pipeline.settings import resolve_settings

    settings = resolve_settings(load_config(config, SCHEMA), schema=SCHEMA, config_path=config)
    settings["use_memmap_loading"] = False
    last_topology = sorted(p for p in (capture / "steps").glob("*.pkl") if "_finish_" not in p.name)[-1]
    with last_topology.open("rb") as f:
        G = pickle.load(f)
    return {
        "settings": settings,
        "binary_mask": unpacked(capture / "mask.npz"),
        "voxel_size_zyx": tuple(json.loads((capture / "voxel_size_zyx.json").read_text())),
        "thick_vessel_mask": unpacked(capture / "thick.npz"),
        "thick_tree": np.load(capture / "tree.npy"),
        "skeleton": unpacked(capture / "skeleton.npz"),
        "G": G,
    }


def loop_reason(seen, case, settings) -> tuple[str | None, float | None]:
    """For a case the join refused as beside a vessel or a loop: which, and
    the hole of the short loop it would close (um)."""
    from haemolynx.graph._helpers import EdgeSampleIndex, densify_polyline, duplicates_existing_vessel
    from haemolynx.graph.dead_end_extension import _closes_loop_in_lumen, _loop_through
    from haemolynx.graph.lumen_loops import loop_hole_um
    from haemolynx.graph.thick_vessel_junctions import _centreline_samples
    from haemolynx.preprocessing import MaskSupport

    G0, rule = seen["join_in"], seen["rule_inputs"]
    vs = rule["vs"]
    support = MaskSupport(np.asarray(rule["lumen"], bool), tuple(vs)).limited_to(
        float(settings["bridge_max_background_gap_um"]), float(settings["bridge_min_mask_fraction"]))
    points, owners, samples = _centreline_samples(G0, rule["near_tree"], rule["thick"], vs)
    n = int(case["node"]) if str(case["node"]).lstrip("-").isdigit() else case["node"]
    p = np.asarray(case["centre"], float)
    j = min((j for j in samples.query_ball_point(p, 60.0) if n not in owners[j][:2]),
            key=lambda j: float(np.linalg.norm(points[j] - p)))
    q, edge = points[j], owners[j]
    path = densify_polyline(np.asarray(case["route"], float), max_step_um=0.5)
    distance, nearest = rule["tree"].query(path)
    in_lumen = np.flatnonzero(distance <= np.nan_to_num(rule["tree_radius"][nearest], posinf=0.0))
    outside = path[: int(in_lumen[0]) if len(in_lumen) else len(path)]
    if len(outside) >= 2 and duplicates_existing_vessel(
        outside, G0, support.inside, support.radius, index=EdgeSampleIndex(G0), judge_short_at_middle=True,
    ):
        return "beside a vessel in one lumen", None
    if _closes_loop_in_lumen(G0, n, edge, q, path, support):
        return "a loop inside one lumen", None
    short = float(settings.get("remove_short_loops_under_um") or 0.0)
    loop = _loop_through(G0, n, edge, q, path, short) if short > 0 else None
    if loop is not None:
        return "a short loop round a small hole", loop_hole_um(loop, support)
    return "none found", None


def untried_case(seen, p) -> dict | None:
    """A dead end at *p* the wall join never tried: no stretch of the graph's
    own thick centreline within its reach. Drawn as ``wall_cases`` draws a
    case, its way going to the nearest point of the thick centreline tree."""
    from haemolynx.graph.build import GAP_BRIDGE_TANGENT_LENGTH_UM, _terminal_end
    from haemolynx.graph.cartwheel_guard import _point_along_polyline
    from haemolynx.graph.reconnect import route_through_mask
    from haemolynx.graph.thick_vessel_junctions import WALL_JOIN_REACH_UM, _centreline_samples, _polyline_from
    from haemolynx.preprocessing import MaskSupport

    G0, rule = seen["join_in"], seen["rule_inputs"]
    tree, radius, vs, lumen = rule["tree"], rule["tree_radius"], rule["vs"], rule["lumen"]
    tips = [n for n in G0 if G0.degree(n) == 1 and "pos" in G0.nodes[n]]
    near = min(tips, key=lambda n: float(np.linalg.norm(np.asarray(G0.nodes[n]["pos"], float) - p)))
    if float(np.linalg.norm(np.asarray(G0.nodes[near]["pos"], float) - p)) > MATCH_UM:
        return None
    (a, b, _key, data), = G0.edges(near, keys=True, data=True)
    w = b if a == near else a
    p = np.asarray(G0.nodes[near]["pos"], float)
    distance, nearest = tree.query(p)
    r = float(radius[nearest]) if np.isfinite(radius[nearest]) else 0.0
    q = np.asarray(tree.data[nearest], float)
    points, owners, samples = _centreline_samples(G0, rule["near_tree"], rule["thick"], vs)
    graph_um = None
    if len(points):
        others = [j for j in range(len(points)) if near not in owners[j][:2]]
        graph_um = float(min(np.linalg.norm(points[others] - p, axis=1))) if others else None
    support = MaskSupport(np.asarray(lumen, bool), tuple(vs))
    end = _terminal_end(G0, near, GAP_BRIDGE_TANGENT_LENGTH_UM)
    heading, end_length = (None, 0.0) if end is None else end
    routed = route_through_mask(lumen, p, q, tuple(vs))
    route = routed if routed is not None and support.accepts(routed) else np.vstack([p, q])
    chain = _polyline_from(G0, w, near, data)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    return dict(
        rule="wall", node=str(near), piece=chain[arc >= arc[-1] - 25.0], route=np.asarray(route, float), centre=p,
        decision=(f"never tried: the graph's thick centreline is {graph_um:.0f} um away, past its "
                  f"{r + WALL_JOIN_REACH_UM:.0f} um reach" if graph_um is not None else
                  "never tried: no graph thick centreline"),
        joined=False, past_lumen_um=float(distance) - r, lumen_radius_um=r,
        chord_turn_deg=grid._angle_deg(heading, q - p) if heading is not None else float("nan"),
        route_turn_deg=(grid._angle_deg(heading, _point_along_polyline(np.asarray(route, float),
                                                                       GAP_BRIDGE_TANGENT_LENGTH_UM) - p)
                        if heading is not None else float("nan")),
        end_length_um=float(end_length), to_centreline_um=float(distance),
        route_accepted=routed is not None and support.accepts(routed),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="thick_bridge_review/capture_steps.py's run folder")
    parser.add_argument("--run-graph", required=True, help="the run's solved graph (graph_returned.pkl)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--crop-ends", help="crop_ends.json, to note the crop's numbers in the key")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import build_grid as phase5
    from scipy.spatial import cKDTree

    from haemolynx.graph import IS_ZERO_RESISTANCE
    from haemolynx.graph.dead_ends import THICK_WALL, classify_dead_ends
    from haemolynx.preprocessing import MaskSupport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap_path = out / "finish_input.pkl"
    cap = finish_input(Path(args.capture), Path(args.config))
    with cap_path.open("wb") as f:
        pickle.dump(cap, f, protocol=pickle.HIGHEST_PROTOCOL)
    seen = grid.replay(cap_path)
    final = seen["final"]
    vs = np.asarray(cap["voxel_size_zyx"], float)
    mask = np.asarray(cap["binary_mask"], bool)
    region = np.asarray(cap["thick_vessel_mask"], bool)
    tree_points = np.asarray(cap["thick_tree"], float) * vs
    settings = cap["settings"]

    with open(args.run_graph, "rb") as f:
        run = pickle.load(f)
    tips = [n for n, d in run.nodes(data=True)
            if run.degree(n) == 1 and d.get("dead_end_kind") == THICK_WALL and d["pos"][0] < SURFACE_UM]
    ends = classify_dead_ends(run, support=MaskSupport(mask, tuple(vs)), nodes=tips, stub_radius_multiple=float(
        settings["min_stub_length_radius_multiple"]))
    pool = grid.wall_cases(seen)
    centres = np.vstack([c["centre"] for c in pool])
    pool_tree = cKDTree(centres)
    crop = {}
    if args.crop_ends:
        rows = json.loads(Path(args.crop_ends).read_text(encoding="utf-8"))["dead_ends"]
        crop = {r["tip"]: number for number, r in enumerate(rows, start=1)}

    cases, unmatched = [], []
    for tip in tips:
        p = np.asarray(run.nodes[tip]["pos"], float)
        distance, i = pool_tree.query(p)
        if distance <= MATCH_UM:
            case = dict(pool[i])
        else:
            case = untried_case(seen, p)
            if case is None:
                unmatched.append({"run_node": tip, "centre_um": [round(float(x), 2) for x in p],
                                  "nearest_pool_case_um": round(float(distance), 2)})
                continue
        e = ends[tip]
        case.update(
            run_node=tip, pool_node=case["node"], crop_number=crop.get(tip),
            chain_um=e.chain_length_um, tip_radius_um=e.tip_radius_um, mask_run_um=e.mask_run_um,
            ahead_um=e.ahead_um, ahead_turn_deg=e.ahead_turn_deg, ahead_background_um=e.ahead_background_um,
            thick_turn_deg=e.thick_turn_deg,
        )
        if case["decision"] == "beside a vessel, or a loop":
            case["loop_reason"], case["loop_hole_um"] = loop_reason(seen, case, settings)
        cases.append(case)

    rng = random.Random(SEED)
    rng.shuffle(cases)
    for i, case in enumerate(cases, start=1):
        case["id"] = f"{GRID}{i:02d}"
        case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
        shown = np.vstack([case["piece"], case["route"]])
        case["axis"] = phase5._plane(dict(case, piece=shown), tree_points)
        case["view"] = phase5.PLANES[case["axis"]][0]
        case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])

    # The graph the wall join was handed, so no sheet shows its answer.
    drawn = seen["join_in"]
    paths = [(phase5._edge_path(drawn, u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
             for u, v, d in drawn.edges(data=True) if u != v]
    sheets = []
    for s, start in enumerate(range(0, len(cases), PER_SHEET), start=1):
        part = cases[start:start + PER_SHEET]
        title = (f"E14.5 surface layer, grid {GRID} sheet {s} (IDs {part[0]['id']}-{part[-1]['id']}): dead ends "
                 f"the thick-vessel split left at a big vessel's wall\n{QUESTION}")
        grid._sheet(part, out / f"grid_{GRID}_{s}.png", mask, region, tree_points, paths, vs, title)
        sheets.append({
            "rule": "wall", "title": f"Sheet {s}: {part[0]['id']}-{part[-1]['id']}", "question": QUESTION,
            "marks": [{"key": k, "label": label} for k, label in MARKS],
            "cases": [grid.case_view(case, mask, region, tree_points, paths, vs) for case in part],
        })
    data = {
        "grid": GRID, "dataset": "E14.5 MCA, Harvey's Claudin segmentation (today's run, 9a620be2)",
        "voxel_size_zyx": [float(v) for v in vs], "slab_um": 2 * phase5.HALF_SLAB_UM, "sheets": sheets,
    }
    (out / "page.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    (out / f"grid_{GRID}.html").write_text(grid_page.build_page(data), encoding="utf-8")
    with open(out / f"key_{GRID}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in sorted(cases, key=lambda c: c["id"]):
            writer.writerow({k: grid._plain(case.get(k)) for k in KEY_FIELDS})
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "mark", "note"])
        writer.writerows([c["id"], "", ""] for c in sorted(cases, key=lambda c: c["id"]))
    summary = {
        "replayed_final_edges": final.number_of_edges(), "replayed_final_nodes": final.number_of_nodes(),
        "surface_thick_wall_dead_ends": len(tips), "cases": len(cases), "unmatched": unmatched,
        "decisions": grid._count(c["decision"] for c in cases),
        "loop_reasons": grid._count(c.get("loop_reason") for c in cases if c.get("loop_reason")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
