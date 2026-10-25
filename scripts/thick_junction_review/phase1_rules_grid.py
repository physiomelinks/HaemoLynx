#!/usr/bin/env python3
"""PLAN_bridging.md: a fresh marked grid for the thick-vessel split's three
Phase 1 rules, which were fitted on Harvey's own examples (7 missing bridges,
3 wrong ones) and never checked on a grid they were not fitted on (grid W was
never marked).

- **Wall join** (``_join_dead_ends_at_walls``): a dead end within
  ``WALL_JOIN_REACH_UM`` (10 um) of a thick vessel's lumen, heading for it
  (at most 80 degrees off, its vessel at least 3 um), is drawn on through the
  mask to the thick centreline and bridged.
- **Taper** (``_taper_bridges``): at a thick centreline's end, the bridge
  leaving most nearly along it, within ``TAPER_MIN_COSINE`` (0.7), is the
  thick vessel narrowing and stays vessel.
- **Fork** (``_bridge_thick_forks``): at a fork of the thick centreline, an
  arm at most ``FORK_BRANCH_MAX_RADIUS_FRACTION`` (0.8) of the trunk's radius
  is bridged from the fork to where it leaves the trunk's lumen.

The capture (``finish_network_graph``'s input, ``e14_run.py --capture``) is
replayed with this checkout's code, and each rule's candidates are measured
as the split sees them: every dead end within 15 um of a thick lumen, every
pair a taper is chosen from, every arm of a fork. Each sheet asks one rule's
question, its cases spread across that rule's measures and its
decisions (round-robin over the decisions, then over bins of the measures
within each), then shuffled; cases within 6 um of one on the Phase 5 bridge
grid (``--exclude-key``) are left out, so no place is asked twice. Drawn on
the segmentation, as the split reads it, each case in its own plane, with
the graph as that rule is handed it (never its answer); the measures and the
rule's decision are in ``key_<rule>.csv``, never on the sheets.

    python scripts/thick_junction_review/phase1_rules_grid.py \\
        --capture outputs/br/mine/finish_input.pkl --out outputs/thick_junction_review/grid_P \\
        --exclude-key ../plan-thick-bridges-e32666/outputs/thick_bridge_review/key.csv
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thick_bridge_review"))

SEED = 20261025
#: Dead ends this far past a thick lumen are candidates: the rule's 10 um and
#: 5 um beyond it, so the grid shows where the reach cuts off.
WALL_POOL_PAST_LUMEN_UM = 15.0
#: Leave out a case this near (um) one of the Phase 5 grid's.
EXCLUDE_NEAR_UM = 6.0
#: A tip this near (um) an image face leaves the image: not the question.
FACE_UM = 2.0
COUNTS = {"wall": 14, "taper": 8, "fork": 8}
#: The decisions every case of which is drawn: the rules' few positives.
EVERY = {"wall": "joined", "fork": "bridged"}
TURN_BINS = (0.0, 40.0, 80.0, 120.0, np.inf)
PAST_BINS = (-np.inf, 0.0, 5.0, 10.0, np.inf)
COSINE_BINS = (-np.inf, 0.5, 0.7, 0.85, np.inf)
RATIO_BINS = (0.0, 0.6, 0.8, 0.9, np.inf)

LEGEND = (
    "grey = segmentation (a slab 6 um either side of the plane), orange = thick-vessel region's edge, "
    "yellow dots = its centreline tree, green = the graph's vessels, magenta dashes = bridges, white bar = 10 um"
)
QUESTIONS = {
    "wall": (
        "Wall joins: should the red dead end be joined to the big vessel's centreline?\n"
        "J = yes, it is a vessel opening into the big vessel (cyan dashes: the way a join would be drawn)   "
        "N = no, leave it as a dead end   ? = unsure"
    ),
    "taper": (
        "Tapers: the big vessel's centreline ends, and the red piece leads on to a smaller vessel.\n"
        "T = the big vessel narrows into that vessel: one vessel, no bridge   "
        "S = that vessel opens into the big one from the side: a bridge   ? = unsure"
    ),
    "fork": (
        "Forks: the big vessel's centreline forks. Is the red arm a narrower branch of it?\n"
        "B = yes, a branch: bridge it from the fork to where it leaves the trunk's lumen (white cross)   "
        "E = no, the vessels are of a size: no bridge   ? = unsure"
    ),
}


def _angle_deg(a, b) -> float:
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norms <= 0.0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / norms, -1.0, 1.0))))


def _bin(value, bins) -> int:
    return int(np.searchsorted(bins, value, side="right")) - 1


def _round_robin(groups: dict, count: int) -> list:
    order = sorted(groups)
    picked = []
    while len(picked) < count and any(groups[g] for g in order):
        for g in order:
            if groups[g] and len(picked) < count:
                picked.append(groups[g].pop(0))
    return picked


def _spread(cases, count, rng, cell, every=None):
    """Up to *count* cases: every case the rule decided *every* (its few
    positives), then round-robin over its other decisions, and within each
    decision round-robin over *cell*'s bins of its measures."""
    first = [case for case in cases if case["decision"] == every][:count]
    rng.shuffle(first)
    cases = [case for case in cases if case["decision"] != every]
    count -= len(first)
    by_decision: dict = {}
    for case in cases:
        by_decision.setdefault(case["decision"], []).append(case)
    ordered = {}
    for decision, group in by_decision.items():
        cells: dict = {}
        for case in group:
            cells.setdefault(cell(case), []).append(case)
        for members in cells.values():
            rng.shuffle(members)
        ordered[decision] = _round_robin(cells, len(group))
    return first + _round_robin(ordered, count)


def replay(capture: Path) -> dict:
    """Run ``finish_network_graph`` on *capture* and keep what the build's
    split hands each rule, and what each rule decided."""
    from haemolynx import graph
    from haemolynx.graph import thick_vessel_junctions as tvj
    from haemolynx.pipeline import stages

    with open(capture, "rb") as f:
        cap = pickle.load(f)
    settings = dict(cap["settings"])
    context = stages.network_context(
        settings, cap["binary_mask"], cap["voxel_size_zyx"], cap["thick_vessel_mask"],
        thick_centreline=cap["thick_tree"],
    )
    seen: dict = {"in_build_split": False, "taper_calls": []}

    real_split = graph.insert_thick_vessel_junction_nodes

    @functools.wraps(real_split)
    def split(G, *a, **k):
        build = k.get("edges") is None and "split_out" not in seen
        seen["in_build_split"] = build
        try:
            result = real_split(G, *a, **k)
        finally:
            seen["in_build_split"] = False
        if build:
            seen["split_out"] = result.copy()
        return result

    real_join = tvj._join_dead_ends_at_walls

    @functools.wraps(real_join)
    def join(G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, **k):
        if seen["in_build_split"] and "join_in" not in seen:
            seen["join_in"] = G.copy()
            seen["rule_inputs"] = dict(near_tree=near_tree, tree=tree, tree_radius=np.asarray(tree_radius),
                                       lumen=lumen, thick=thick, vs=np.asarray(vs, float))
            made = real_join(G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, **k)
            seen["join_out"] = G.copy()
            return made
        return real_join(G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, **k)

    real_taper = tvj._taper_bridges

    @functools.wraps(real_taper)
    def taper(G, bridge_ends):
        chosen = real_taper(G, bridge_ends)
        if seen["in_build_split"]:
            seen["taper_calls"].append(_taper_pairs(G, bridge_ends, chosen))
        return chosen

    real_forks = tvj._bridge_thick_forks

    @functools.wraps(real_forks)
    def forks(G, near_tree, lumen, vs, chosen):
        if seen["in_build_split"] and "fork_in" not in seen:
            seen["fork_in"] = G.copy()
            made = real_forks(G, near_tree, lumen, vs, chosen)
            seen["fork_out"] = G.copy()
            return made
        return real_forks(G, near_tree, lumen, vs, chosen)

    graph.insert_thick_vessel_junction_nodes = split
    tvj._join_dead_ends_at_walls = join
    tvj._taper_bridges = taper
    tvj._bridge_thick_forks = forks
    try:
        final = stages.finish_network_graph(cap["G"].copy(), cap["skeleton"], settings, context)
    finally:
        graph.insert_thick_vessel_junction_nodes = real_split
        tvj._join_dead_ends_at_walls = real_join
        tvj._taper_bridges = real_taper
        tvj._bridge_thick_forks = real_forks
    seen["final"] = final
    seen["cap"] = cap
    return seen


def _taper_pairs(G, bridge_ends, chosen):
    """Every (centreline end, vessel start) pair :func:`_taper_bridges` chose
    from, with the cosine it read and whether it chose it."""
    from haemolynx.graph.cartwheel_guard import _point_along_polyline
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, TAPER_DIRECTION_UM, _polyline_from

    rows = []
    for b, ends in enumerate(bridge_ends):
        if ends is None or not G.has_node(ends[0]) or not G.has_node(ends[1]):
            continue
        start, end = ends
        vessels = [(a, c, d) for a, c, d in G.edges(end, data=True) if not d.get(IS_ZERO_RESISTANCE)]
        if len(vessels) != 1:
            continue
        a, c, data = vessels[0]
        other = c if a == end else a
        if other == end:
            continue
        end_pos = np.asarray(G.nodes[end]["pos"], float)
        thick_path = _polyline_from(G, end, other, data)
        outward = end_pos - _point_along_polyline(thick_path, TAPER_DIRECTION_UM)
        start_pos = np.asarray(G.nodes[start]["pos"], float)
        onward = start_pos - end_pos
        norms = float(np.linalg.norm(outward) * np.linalg.norm(onward))
        rows.append(dict(
            end_pos=end_pos, start_pos=start_pos, thick_path=thick_path,
            cosine=float(outward @ onward) / norms if norms > 0 else float("nan"),
            chosen=b in chosen, leaving=sum(1 for e in bridge_ends if e is not None and e[1] == end),
        ))
    return rows


def wall_cases(seen) -> list[dict]:
    from haemolynx.graph.build import (
        GAP_BRIDGE_TANGENT_LENGTH_UM, MAX_GAP_BRIDGE_TURN_DEG, MIN_GAP_BRIDGE_END_LENGTH_UM, _terminal_end,
    )
    from haemolynx.graph.cartwheel_guard import _point_along_polyline
    from haemolynx.graph.reconnect import route_through_mask
    from haemolynx.graph.thick_vessel_junctions import (
        IS_ZERO_RESISTANCE, JOINED_AT_WALL_UM, WALL_JOIN_REACH_UM, _centreline_samples, _polyline_from,
    )
    from haemolynx.preprocessing import MaskSupport

    G0, after, rule = seen["join_in"], seen["join_out"], seen["rule_inputs"]
    tree, radius, vs = rule["tree"], rule["tree_radius"], rule["vs"]
    lumen, thick, near_tree = rule["lumen"], rule["thick"], rule["near_tree"]
    support = MaskSupport(np.asarray(lumen, bool), tuple(vs))
    extent = (np.asarray(np.shape(lumen)) - 1) * vs
    points, owners, samples = _centreline_samples(G0, near_tree, thick, vs)
    cases = []
    for n in G0.nodes:
        if G0.degree(n) != 1 or "pos" not in G0.nodes[n]:
            continue
        (a, b, key, data), = G0.edges(n, keys=True, data=True)
        w = b if a == n else a
        if w == n or data.get(IS_ZERO_RESISTANCE):
            continue
        p = np.asarray(G0.nodes[n]["pos"], float)
        if np.any(p <= FACE_UM) or np.any(p >= extent - FACE_UM):
            continue
        distance, nearest = tree.query(p)
        r = float(radius[nearest]) if np.isfinite(radius[nearest]) else 0.0
        past = float(distance) - r
        if past > WALL_POOL_PAST_LUMEN_UM or near_tree(p[None])[1][0]:
            continue
        found = sorted(
            (float(np.linalg.norm(points[j] - p)), j)
            for j in samples.query_ball_point(p, r + WALL_POOL_PAST_LUMEN_UM + 3.0)
            if n not in owners[j][:2]
        )
        if not found:
            continue
        q = points[found[0][1]]
        end = _terminal_end(G0, n, GAP_BRIDGE_TANGENT_LENGTH_UM)
        heading, end_length = (None, 0.0) if end is None else end
        chord_turn = _angle_deg(heading, q - p) if heading is not None else float("nan")
        routed = route_through_mask(lumen, p, q, tuple(vs))
        route = routed if routed is not None and support.accepts(routed) else np.vstack([p, q])
        route_turn = (
            _angle_deg(heading, _point_along_polyline(np.asarray(route, float), GAP_BRIDGE_TANGENT_LENGTH_UM) - p)
            if heading is not None else float("nan")
        )
        joined = not after.has_node(n) and any(
            d.get(JOINED_AT_WALL_UM) for *_e, d in after.edges(w, data=True)
        ) if after.has_node(w) else False
        if joined:
            why = "joined"
        elif past > WALL_JOIN_REACH_UM:
            why = "beyond the reach"
        elif end_length < MIN_GAP_BRIDGE_END_LENGTH_UM:
            why = "vessel end under 3 um"
        elif np.nan_to_num(chord_turn) > MAX_GAP_BRIDGE_TURN_DEG:
            why = "heading away (chord)"
        elif not support.accepts(np.vstack([p, q])) and (routed is None or not support.accepts(routed)):
            why = "across background"
        elif np.nan_to_num(route_turn) > MAX_GAP_BRIDGE_TURN_DEG:
            why = "heading away (route)"
        else:
            why = "beside a vessel, or a loop"
        chain = _polyline_from(G0, w, n, data)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
        piece = chain[arc >= arc[-1] - 25.0]
        cases.append(dict(
            rule="wall", node=str(n), piece=piece, route=np.asarray(route, float), centre=p,
            decision=why, joined=joined, past_lumen_um=past, lumen_radius_um=r,
            chord_turn_deg=chord_turn, route_turn_deg=route_turn, end_length_um=float(end_length),
            to_centreline_um=float(found[0][0]), route_accepted=routed is not None and support.accepts(routed),
        ))
    return cases


def taper_cases(seen) -> list[dict]:
    merged: dict = {}
    for call in seen["taper_calls"]:
        for row in call:
            key = (tuple(np.round(row["end_pos"], 1)), tuple(np.round(row["start_pos"], 1)))
            if key in merged:
                merged[key]["chosen"] = merged[key]["chosen"] or row["chosen"]
            else:
                merged[key] = dict(row)
    cases = []
    for row in merged.values():
        piece = np.vstack([row["end_pos"], row["start_pos"]])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(row["thick_path"], axis=0), axis=1))]
        cases.append(dict(
            rule="taper", piece=piece, context=row["thick_path"][arc <= 15.0], centre=piece.mean(axis=0),
            decision="taper (left as vessel)" if row["chosen"] else "bridge", joined=row["chosen"],
            cosine=row["cosine"], leaving_the_end=row["leaving"],
            length_um=float(np.linalg.norm(row["start_pos"] - row["end_pos"])),
        ))
    return cases


def fork_cases(seen) -> list[dict]:
    from scipy.spatial import cKDTree

    from haemolynx.graph._helpers import densify_polyline
    from haemolynx.graph.thick_vessel_junctions import (
        FORK_ARM_UM, FORK_BRANCH_MAX_RADIUS_FRACTION, IS_ZERO_RESISTANCE, THICK_FORK_BRIDGE, _SAMPLE_STEP_UM,
        _inscribed_radius_um, _polyline_from,
    )

    G0, after, rule = seen["fork_in"], seen["fork_out"], seen["rule_inputs"]
    near_tree, lumen, vs = rule["near_tree"], rule["lumen"], rule["vs"]
    cases = []
    for n in G0.nodes:
        if G0.degree(n) < 3 or "pos" not in G0.nodes[n]:
            continue
        here = np.asarray(G0.nodes[n]["pos"], float)
        if not near_tree(here[None])[1][0]:
            continue
        arms = []
        for a, b, key, data in G0.edges(n, keys=True, data=True):
            other = b if a == n else a
            if data.get(IS_ZERO_RESISTANCE) or other == n:
                continue
            path = densify_polyline(_polyline_from(G0, n, other, data), max_step_um=_SAMPLE_STEP_UM)
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if near_tree(path[arc <= FORK_ARM_UM])[1].mean() < 0.8:
                continue
            arms.append(dict(path=path, arc=arc))
        if len(arms) < 3:
            continue
        r0 = float(_inscribed_radius_um(here[None], lumen, vs)[0])
        if not np.isfinite(r0) or r0 <= 0.0:
            continue
        for arm in arms:
            window = (arm["arc"] >= r0) & (arm["arc"] <= r0 + FORK_ARM_UM)
            radii = _inscribed_radius_um(arm["path"][window], lumen, vs) if window.any() else np.zeros(0)
            radii = radii[np.isfinite(radii)]
            arm["radius"] = float(np.median(radii)) if len(radii) else float("nan")
        if any(not np.isfinite(arm["radius"]) for arm in arms):
            continue
        arms.sort(key=lambda arm: -arm["radius"])
        trunk, branches = arms[:2], arms[2:]
        trunk_points = np.vstack([arm["path"] for arm in trunk])
        trunk_radii = _inscribed_radius_um(trunk_points, lumen, vs)
        trunk_tree = cKDTree(trunk_points)
        narrowest = min(arm["radius"] for arm in trunk)
        bridged = after.has_node(n) and any(d.get(THICK_FORK_BRIDGE) for *_e, d in after.edges(n, data=True))
        for arm in branches:
            distance, nearest = trunk_tree.query(arm["path"])
            left = np.flatnonzero((distance > trunk_radii[nearest]) & (arm["arc"] > 0.0))
            tick = arm["path"][left[0]] if len(left) else None
            ratio = arm["radius"] / narrowest
            cases.append(dict(
                rule="fork", node=str(n), piece=arm["path"][arm["arc"] <= 15.0], centre=here, tick=tick,
                context=np.vstack([t["path"][t["arc"] <= 15.0] for t in trunk]),
                decision="bridged" if bridged and ratio <= FORK_BRANCH_MAX_RADIUS_FRACTION else "no bridge",
                joined=bool(bridged and ratio <= FORK_BRANCH_MAX_RADIUS_FRACTION),
                ratio=ratio, branch_radius_um=arm["radius"], trunk_radius_um=narrowest, fork_radius_um=r0,
                leaves_trunk_at_um=None if tick is None else float(arm["arc"][left[0]]),
            ))
    return cases


def _excluded(centre, others) -> bool:
    return bool(len(others)) and float(np.min(np.linalg.norm(others - centre, axis=1))) < EXCLUDE_NEAR_UM


def _draw_case(ax, case, mask, region, tree_points, paths, vs):
    import build_grid as phase5

    phase5._draw(ax, case, mask, region, tree_points, paths, vs)
    _label, h, w = phase5.PLANES[case["axis"]]
    if case.get("route") is not None:
        r = case["route"]
        ax.plot(r[:, w], r[:, h], "--", color="#2ad4ff", lw=1.6, dashes=(3.0, 2.0))
    if case.get("tick") is not None:
        t = case["tick"]
        ax.plot([t[w]], [t[h]], "x", color="white", ms=9, mew=2.2)


def _sheet(cases, out_png, mask, region, tree_points, paths, vs, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 5
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.95 * rows + 1.7), squeeze=False)
    fig.suptitle(f"{title}\n{LEGEND}", fontsize=10)
    for ax, case in zip(axes.ravel(), cases):
        _draw_case(ax, case, mask, region, tree_points, paths, vs)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.7 / (3.95 * rows + 1.7), wspace=0.04,
                        hspace=0.16)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


KEY_FIELDS = {
    "wall": ("id", "rule", "vox", "centre_um", "view", "decision", "past_lumen_um", "lumen_radius_um",
             "chord_turn_deg", "route_turn_deg", "end_length_um", "to_centreline_um", "route_accepted", "node"),
    "taper": ("id", "rule", "vox", "centre_um", "view", "decision", "cosine", "leaving_the_end", "length_um"),
    "fork": ("id", "rule", "vox", "centre_um", "view", "decision", "ratio", "branch_radius_um", "trunk_radius_um",
             "fork_radius_um", "leaves_trunk_at_um", "node"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="e14_run.py --capture's finish_input.pkl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-key", help="the Phase 5 grid's key.csv: leave out cases near its")
    args = parser.parse_args(argv)

    import build_grid as phase5

    from haemolynx.graph import IS_ZERO_RESISTANCE

    seen = replay(Path(args.capture))
    cap = seen["cap"]
    vs = np.asarray(cap["voxel_size_zyx"], float)
    mask = np.asarray(cap["binary_mask"], bool)
    region = np.asarray(cap["thick_vessel_mask"], bool)
    tree_points = np.asarray(cap["thick_tree"], float) * vs
    others = np.zeros((0, 3))
    if args.exclude_key:
        with open(args.exclude_key, encoding="utf-8") as f:
            others = np.asarray([json.loads(r["centre_um"].replace("(", "[").replace(")", "]"))
                                 for r in csv.DictReader(f)], float)

    pools = {"wall": wall_cases(seen), "taper": taper_cases(seen), "fork": fork_cases(seen)}
    cells = {
        "wall": lambda c: (_bin(np.nan_to_num(c["route_turn_deg"], nan=180.0), TURN_BINS),
                           _bin(c["past_lumen_um"], PAST_BINS)),
        "taper": lambda c: _bin(np.nan_to_num(c["cosine"], nan=-1.0), COSINE_BINS),
        "fork": lambda c: _bin(c["ratio"], RATIO_BINS),
    }
    rng = random.Random(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first_id = 1
    summary = {}
    picked_all = []
    for rule in ("wall", "taper", "fork"):
        pool = [c for c in pools[rule] if not _excluded(c["centre"], others)]
        picked = _spread(pool, COUNTS[rule], rng, cells[rule], every=EVERY.get(rule))
        rng.shuffle(picked)
        for i, case in enumerate(picked, start=first_id):
            case["id"] = f"P{i:02d}"
            case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
            shown = case["piece"] if case.get("route") is None else np.vstack([case["piece"], case["route"]])
            case["axis"] = phase5._plane(dict(case, piece=shown), tree_points)
            case["view"] = phase5.PLANES[case["axis"]][0]
            case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])
        first_id += len(picked)
        summary[rule] = {
            "pool": len(pools[rule]), "after_exclusion": len(pool), "picked": len(picked),
            "pool_decisions": _count(c["decision"] for c in pools[rule]),
            "picked_decisions": _count(c["decision"] for c in picked),
        }
        # The graph each rule is handed, so no sheet shows the rule's answer:
        # the dead ends before any join, the graph the split samples, the
        # graph before the fork bridges.
        drawn = {"wall": seen["join_in"], "taper": seen["join_out"], "fork": seen["fork_in"]}[rule]
        paths = [(phase5._edge_path(drawn, u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
                 for u, v, d in drawn.edges(data=True) if u != v]
        if picked:
            _sheet(picked, out / f"grid_P_{rule}.png", mask, region, tree_points, paths, vs,
                   f"E14.5, grid P ({rule}s, IDs {picked[0]['id']}-{picked[-1]['id']})   " + QUESTIONS[rule])
        with open(out / f"key_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KEY_FIELDS[rule], extrasaction="ignore")
            writer.writeheader()
            for case in sorted(picked, key=lambda c: c["id"]):
                writer.writerow({k: _plain(case.get(k)) for k in KEY_FIELDS[rule]})
        with open(out / f"pool_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            fields = [k for k in KEY_FIELDS[rule] if k not in ("id", "view")]
            writer = csv.DictWriter(f, fieldnames=["excluded"] + fields, extrasaction="ignore")
            writer.writeheader()
            for case in pools[rule]:
                row = {k: _plain(case.get(k)) for k in fields}
                row["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
                row["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])
                row["excluded"] = _excluded(case["centre"], others)
                writer.writerow(row)
        picked_all += picked
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "rule", "mark"])
        writer.writerows([c["id"], c["rule"], ""] for c in picked_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _plain(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    if isinstance(value, np.ndarray):
        return tuple(round(float(x), 2) for x in value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
