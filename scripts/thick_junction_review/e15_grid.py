#!/usr/bin/env python3
"""PLAN_bridging.md's last grid: E15 (Harvey's Simple Segmentation of
``Composite_150826_E15_cldn5_..._spot1``, the second stack with thick
vessels here), a fresh check of three things fitted on E14.5 alone (his
calls, 2026-10-25):

- **The tree fill** (``_fill_tree_gaps``, ``TREE_FILL_MIN_DETOUR`` 2.1, fitted
  on grid V): every gap in the graph's thick centreline the fill weighs,
  filled or not, the tree's path across it in red. Y = draw it, N = leave the
  graph as it was. The key holds the graph's own way between the gap's sides
  over the tree's path, read to the end rather than cut at 2.1.
- **The wall join** (``_join_dead_ends_at_walls``, its reach 15 um since grid
  P2): dead ends up to 20 um past a thick lumen, joined or not, as grid P
  asked them. J/N.
- **Grid K's two measures**: every connector the split leaves unbridged, as
  grid K asked them (B/N); the key holds each exit on its own (``exits``,
  ``failed_exits``) and the length inside the region (``inside_um``).

One replay of ``outputs/br/e15_run.py``'s capture with this checkout's code,
each rule's cases measured as the split hands them to it; one sheet per
question, its cases spread across the rule's decisions and measures, then
shuffled, IDs E01... on from sheet to sheet, 30 in all: a sheet short of
cases leaves its share to the wall sheet (E15's build on its large masks
gave the fill no gap with the graph on two sides, and the split left no
connector). Each sheet draws the graph as its rule is handed it (never its
answer); the measures and decisions are in ``key_<rule>.csv`` only.

    python scripts/thick_junction_review/e15_grid.py \\
        --capture outputs/br/e15_build/finish_input.pkl \\
        --out outputs/thick_junction_review/grid_E15 \\
        --page-data outputs/thick_junction_review/grid_E15/page.json
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from thick_junction_review import connector_grid as connectors
    from thick_junction_review import phase1_rules_grid as grid
except ImportError:  # run as a script, from its own folder
    import connector_grid as connectors
    import phase1_rules_grid as grid

SEED = 20261026
DATASET = "E15 (Composite_150826_E15_cldn5 spot1), Harvey's Simple Segmentation"
#: Dead ends this far past a thick lumen are wall-join cases: the reach
#: (15 um) and 5 um beyond it.
WALL_POOL_PAST_LUMEN_UM = 20.0
#: The grid's size; the wall sheet takes what the other two leave of it.
TOTAL = 30
COUNTS = {"wall": 12, "fill": 10, "connector": 8}
#: Of the wall sheet, how many the join drew and how many it left past its
#: reach heading for the thick vessel; the rest is every other decision.
WALL_JOINED, WALL_BEYOND = 6, 3
PAST_BINS = (-np.inf, 0.0, 5.0, 10.0, 15.0, np.inf)
RATIO_BINS = (0.0, 1.0, 1.5, 2.1, 3.0, np.inf)
INSIDE_BINS = (0.0, 12.0, 24.4, 40.0, np.inf)
FILLED, LEFT = "filled", "left as it was"

QUESTIONS = {
    "wall": grid.QUESTIONS["wall"],
    "fill": (
        "Tree fill: the big vessel's centreline tree (yellow) runs where no graph centreline follows it.\n"
        "Is the red path the big vessel's centreline here?   Y = yes, draw it   "
        "N = no, leave the graph as it was   ? = unsure"
    ),
    "connector": connectors.QUESTION,
}
MARKS = {
    "wall": grid.MARKS["wall"],
    "fill": (("Y", "Draw it: the big vessel's centreline"), ("N", "Leave the graph as it was"), ("?", "Unsure")),
    "connector": connectors.MARKS,
}
TITLES = {"wall": "Wall joins", "fill": "Tree fill", "connector": "Connectors"}
KEY_FIELDS = {
    "wall": grid.KEY_FIELDS["wall"],
    "fill": ("id", "rule", "vox", "centre_um", "view", "decision", "rule_says", "own_way_ratio", "own_way_um",
             "tree_way_um", "gap_um", "sides", "tree_radius_um", "drawn_share"),
    "connector": connectors.KEY_FIELDS + ("each_exit_on_its_own",),
}


def replay(capture: Path) -> dict:
    """``finish_network_graph`` on *capture*, keeping what the build's own
    split hands the tree fill, the wall join and the bridges: the graph
    before and after each, and the connectors it leaves unbridged."""
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

    real_fill = tvj._fill_tree_gaps

    @functools.wraps(real_fill)
    def fill(G, tree_voxels, vs, tree_points, tree_tolerance, tree_radius, *, near=None, attributes=None):
        first = seen["in_build_split"] and near is None and "fill_in" not in seen
        if first:
            seen["fill_in"] = G.copy()
            seen["fill_args"] = dict(
                tree_voxels=np.asarray(tree_voxels), vs=np.asarray(vs, float),
                tree_points=np.asarray(tree_points, float), tolerance=np.asarray(tree_tolerance, float),
                tree_radius=np.asarray(tree_radius, float),
            )
        made = real_fill(G, tree_voxels, vs, tree_points, tree_tolerance, tree_radius, near=near,
                         attributes=attributes)
        if first:
            seen["fill_out"] = G.copy()
            seen["fill_made"] = made
        return made

    real_join = tvj._join_dead_ends_at_walls

    @functools.wraps(real_join)
    def join(G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, **k):
        first = seen["in_build_split"] and "join_in" not in seen
        if first:
            seen["join_in"] = G.copy()
            seen["rule_inputs"] = dict(near_tree=near_tree, tree=tree, tree_radius=np.asarray(tree_radius),
                                       lumen=lumen, thick=thick, vs=np.asarray(vs, float))
        made = real_join(G, near_tree, tree, tree_radius, lumen, thick, vs, chosen, **k)
        if first:
            seen["join_out"] = G.copy()
        return made

    graph.insert_thick_vessel_junction_nodes = split
    tvj._fill_tree_gaps = fill
    tvj._join_dead_ends_at_walls = join
    try:
        with connectors.failed_connectors_recorded(seen["connectors"], active=lambda: seen["in_build_split"]):
            seen["final"] = stages.finish_network_graph(cap["G"].copy(), cap["skeleton"], settings, context)
    finally:
        graph.insert_thick_vessel_junction_nodes = real_split
        tvj._fill_tree_gaps = real_fill
        tvj._join_dead_ends_at_walls = real_join
    seen["cap"] = cap
    return seen


def fill_cases(seen) -> list[dict]:
    """Every gap the tree fill weighs on the graph it is handed, found as it
    finds them (tree voxels no graph centreline follows, ``TREE_GAP_MIN_UM``
    or longer, followed on two sides or more): the tree's path across it as
    the fill would draw it, the graph's own way between its sides over that
    path (the least over pairs of sides, ``None`` with none), what the rule
    says of that ratio, and whether the fill drew a stretch there."""
    import networkx as nx
    from scipy.spatial import cKDTree

    from haemolynx.graph import thick_vessel_junctions as tvj

    G, after, a = seen["fill_in"], seen["fill_out"], seen["fill_args"]
    vs, tree_vox, tree_points = a["vs"], a["tree_voxels"], a["tree_points"]
    tolerance, radius = a["tolerance"], a["tree_radius"]
    points, owners, arcs = [], [], []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(tvj.IS_ZERO_RESISTANCE) or "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
            continue
        path = tvj.densify_polyline(tvj._polyline_from(G, u, v, data), max_step_um=tvj._SAMPLE_STEP_UM)
        along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        points.append(path)
        owners += [(u, v, key)] * len(path)
        arcs += [(float(x), float(along[-1])) for x in along]
    if not points:
        return []
    points = np.vstack(points)
    samples = cKDTree(points)
    covered = samples.query(tree_points)[0] <= tolerance
    drawn = [
        tvj.densify_polyline(tvj._polyline_from(after, u, v, d), max_step_um=tvj._SAMPLE_STEP_UM)
        for u, v, d in after.edges(data=True) if d.get(tvj.FROM_TREE)
    ]
    drawn_tree = cKDTree(np.vstack(drawn)) if drawn else None

    T = nx.Graph()
    T.add_nodes_from(range(len(tree_vox)))
    for i, j in cKDTree(tree_vox).query_pairs(1.75):
        T.add_edge(int(i), int(j), weight=float(np.linalg.norm((tree_vox[i] - tree_vox[j]) * vs)))
    cases = []
    for gap in nx.connected_components(T.subgraph([i for i in range(len(tree_vox)) if not covered[i]])):
        sides = sorted({j for i in gap for j in T.neighbors(i) if covered[j]})
        if len(sides) < 2:
            continue
        local = T.subgraph(set(gap) | set(sides))
        gap_um = sum(d["weight"] for *_e, d in T.subgraph(gap).edges(data=True)) + 2 * float(vs.min())
        if gap_um < tvj.TREE_GAP_MIN_UM:
            continue
        best, best_own, best_tree = np.inf, np.inf, np.nan
        for x, side in enumerate(sides):
            for other in sides[x + 1:]:
                at = [int(samples.query(tree_points[n])[1]) for n in (side, other)]
                route = nx.shortest_path(local, side, other, weight="weight")
                tree_way = float(tvj.calculate_path_length(
                    [points[at[0]], *(tree_points[n] for n in route[1:-1]), points[at[1]]]
                ))
                own = tvj._graph_way_um(G, *[(owners[k], *arcs[k]) for k in at], cutoff=50.0 * tree_way)
                ratio = own / tree_way if tree_way > 0 else np.inf
                if ratio < best or not np.isfinite(best_tree):
                    best, best_own, best_tree = ratio, own, tree_way
        if len(sides) == 2:
            routes = [nx.shortest_path(local, sides[0], sides[1], weight="weight")]
        else:
            hub = max(gap, key=lambda i: float(radius[i]))
            routes = [r for r in (nx.shortest_path(local, s, hub, weight="weight") for s in sides) if len(r) >= 2]
        # One polyline: out along each side's route to the hub and back.
        order = list(routes[0])
        for r in routes[1:]:
            order += list(reversed(r))[1:] + list(r)[1:]
        piece = tree_points[order]
        share = 0.0 if drawn_tree is None else float(np.mean(drawn_tree.query(piece)[0] <= 1.5))
        cases.append(dict(
            rule="fill", piece=piece, centre=tree_points[max(gap, key=lambda i: float(radius[i]))],
            decision=FILLED if share >= 0.5 else LEFT,
            rule_says=FILLED if best > tvj.TREE_FILL_MIN_DETOUR else LEFT,
            own_way_ratio=None if not np.isfinite(best) else best,
            own_way_um=None if not np.isfinite(best_own) else best_own, tree_way_um=best_tree,
            gap_um=gap_um, sides=len(sides), tree_radius_um=float(np.median(radius[sorted(gap)])),
            drawn_share=share,
        ))
    return cases


def wall_pool(seen) -> list[dict]:
    """Grid P's wall-join cases (:func:`phase1_rules_grid.wall_cases`), out to
    :data:`WALL_POOL_PAST_LUMEN_UM` past the lumen."""
    kept = grid.WALL_POOL_PAST_LUMEN_UM
    grid.WALL_POOL_PAST_LUMEN_UM = WALL_POOL_PAST_LUMEN_UM
    try:
        return grid.wall_cases(seen)
    finally:
        grid.WALL_POOL_PAST_LUMEN_UM = kept


def wall_count(fill_picked: int, connector_picked: int) -> int:
    """The wall sheet's size: :data:`COUNTS`'s, or more where the fill and
    connector sheets came out short of theirs (E15 gave neither a case), so
    the grid still asks :data:`TOTAL`."""
    return max(COUNTS["wall"], TOTAL - fill_picked - connector_picked)


def pick_wall(pool, rng, count: int = COUNTS["wall"]) -> list[dict]:
    """:data:`WALL_JOINED` joins, :data:`WALL_BEYOND` dead ends past the reach
    heading for the thick vessel, both spread over how far past the lumen they
    end, then every other decision round-robin, *count* in all."""
    from haemolynx.graph.build import MAX_GAP_BRIDGE_TURN_DEG

    def past(case):
        return grid._bin(case["past_lumen_um"], PAST_BINS)

    joined = [c for c in pool if c["decision"] == "joined"]
    beyond = [c for c in pool if c["decision"] == "beyond the reach"
              and np.nan_to_num(c["chord_turn_deg"], nan=180.0) < MAX_GAP_BRIDGE_TURN_DEG]
    rest = [c for c in pool if not any(c is d for d in joined + beyond)]
    picked = grid._spread(joined, WALL_JOINED, rng, past)
    picked += grid._spread(beyond, WALL_BEYOND, rng, past)
    cell = lambda c: (grid._bin(np.nan_to_num(c["route_turn_deg"], nan=180.0), grid.TURN_BINS), past(c))  # noqa: E731
    return picked + grid._spread(rest, count - len(picked), rng, cell)


def connector_cases(seen) -> list[dict]:
    cases = []
    for record in seen["connectors"]:
        if record["exits"]:
            case = connectors.connector_case(record)
            case["each_exit_on_its_own"] = "bridged" if case["failed_exits"] < case["exits"] else "left"
            case["decision"] = "left unbridged"
            cases.append(case)
    return cases


def _draw_case(ax, case, mask, region, tree_points, paths, vs):
    grid._draw_case(ax, case, mask, region, tree_points, paths, vs)
    ax.text(0.02, 0.02, f"vox (z,y,x) {case['vox']}  {case['view']}", transform=ax.transAxes, color="white",
            fontsize=7.5, va="bottom", bbox=dict(facecolor="black", edgecolor="none", pad=1.5, alpha=0.7))


def _sheet(cases, out_png, mask, region, tree_points, paths, vs, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 5
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.95 * rows + 1.7), squeeze=False)
    fig.suptitle(f"{title}\n{grid.LEGEND}", fontsize=10)
    for ax, case in zip(axes.ravel(), cases):
        _draw_case(ax, case, mask, region, tree_points, paths, vs)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 1.7 / (3.95 * rows + 1.7), wspace=0.04,
                        hspace=0.16)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True, help="outputs/br/e15_run.py's finish_input.pkl")
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

    pools = {
        "fill": fill_cases(seen) if "fill_in" in seen else [],
        "wall": wall_pool(seen) if "join_in" in seen else [],
        "connector": connector_cases(seen),
    }
    rng = random.Random(SEED)
    picks = {
        "fill": grid._spread(pools["fill"], COUNTS["fill"], rng,
                             lambda c: grid._bin(c["own_way_ratio"] if c["own_way_ratio"] is not None else np.inf,
                                                 RATIO_BINS)),
        "connector": grid._spread(pools["connector"], COUNTS["connector"], rng,
                                  lambda c: grid._bin(c["inside_um"], INSIDE_BINS)),
    }
    picks["wall"] = pick_wall(pools["wall"], rng, wall_count(len(picks["fill"]), len(picks["connector"])))
    # The graph each rule is handed, so no sheet shows its answer.
    drawn = {"fill": seen.get("fill_in"), "wall": seen.get("join_in"), "connector": seen.get("split_in")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first_id, sheets, picked_all, summary = 1, [], [], {}
    for rule in ("fill", "wall", "connector"):
        picked = picks[rule]
        rng.shuffle(picked)
        for i, case in enumerate(picked, start=first_id):
            case["id"] = f"E{i:02d}"
            case["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
            shown = case["piece"] if case.get("route") is None else np.vstack([case["piece"], case["route"]])
            case["axis"] = phase5._plane(dict(case, piece=shown), tree_points)
            case["view"] = phase5.PLANES[case["axis"]][0]
            case["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])
        first_id += len(picked)
        summary[rule] = {
            "pool": len(pools[rule]), "picked": len(picked),
            "pool_decisions": grid._count(c["decision"] for c in pools[rule]),
            "picked_decisions": grid._count(c["decision"] for c in picked),
        }
        if rule == "fill":
            summary[rule]["rule_says"] = grid._count(c["rule_says"] for c in pools[rule])
            summary[rule]["stretches_drawn"] = seen.get("fill_made")
        if rule == "connector":
            summary[rule]["each_exit_on_its_own"] = grid._count(c["each_exit_on_its_own"] for c in pools[rule])
        paths = [] if drawn[rule] is None else [
            (phase5._edge_path(drawn[rule], u, v, d), bool(d.get(IS_ZERO_RESISTANCE)))
            for u, v, d in drawn[rule].edges(data=True) if u != v
        ]
        if picked:
            _sheet(picked, out / f"grid_E15_{rule}.png", mask, region, tree_points, paths, vs,
                   f"E15 ({TITLES[rule].lower()}, IDs {picked[0]['id']}-{picked[-1]['id']})   " + QUESTIONS[rule])
        with open(out / f"key_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KEY_FIELDS[rule], extrasaction="ignore")
            writer.writeheader()
            for case in sorted(picked, key=lambda c: c["id"]):
                writer.writerow({k: grid._plain(case.get(k)) for k in KEY_FIELDS[rule]})
        with open(out / f"pool_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            fields = [k for k in KEY_FIELDS[rule] if k not in ("id", "view")]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for case in pools[rule]:
                row = {k: grid._plain(case.get(k)) for k in fields}
                row["vox"] = tuple(int(round(x)) for x in case["centre"] / vs)
                row["centre_um"] = tuple(round(float(x), 2) for x in case["centre"])
                writer.writerow(row)
        picked_all += picked
        if args.page_data and picked:
            sheets.append({
                "rule": rule, "title": TITLES[rule], "question": QUESTIONS[rule].split("\n")[0],
                "marks": [{"key": k, "label": label} for k, label in MARKS[rule]],
                "cases": [grid.case_view(case, mask, region, tree_points, paths, vs) for case in picked],
            })
    if args.page_data:
        Path(args.page_data).write_text(json.dumps({
            "grid": "E15", "dataset": DATASET,
            "voxel_size_zyx": [float(v) for v in vs],
            "slab_um": 2 * phase5.HALF_SLAB_UM,
            "sheets": sheets,
        }, separators=(",", ":")), encoding="utf-8")
    with open(out / "marks_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "rule", "mark"])
        writer.writerows([c["id"], c["rule"], ""] for c in picked_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
