#!/usr/bin/env python3
"""How much a solved network's flow depends on the boundary condition at the
vessels leaving the image (PLAN_dead_ends.md Phase 4; the release review's
``flow.8b``).

Takes one solved run's graph (resistances, diameters and the solve's node
pressures on it) and solves it again under each ``open_end_model``: closed,
terminal resistance at k = 1, 10 and 100, outlet, and the Fry estimate. For
each it reports the network resistance (pressure drop over the inflow
through the designated inlets), that inflow, what enters and leaves through
the open ends, the share of the centreline that carries flow, the capillary
flow distribution, and the share of vessels whose flow direction differs from
the estimate's -- for the whole network and with a margin next to the image
faces left out, since every method's errors sit near the boundary.

Usage::

    python scripts/open_end_sensitivity.py --graph <run>_checkpoint_solve.pkl \\
        --inlet-pa 4500 --outlet-pa 1000 [--mask mask.npy --voxel-size-zyx 1 0.98 0.98] \\
        [--inlets 1 2 --outlets 3 4] --out outputs/open_end_sensitivity

Without ``--inlets``/``--outlets`` they are read back from the solve: the
dead ends of the solved network held at the inlet and the outlet pressure.
With ``--mask`` every dead end is classified first (``graph.dead_ends``), so
an end the mask runs on from to a face counts as leaving the image; without
it, only the kinds on the graph, else distance to a face.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

MODELS = (
    ("closed", {}),
    ("terminal_resistance", {"terminal_resistance_multiple": 1.0}),
    ("terminal_resistance", {"terminal_resistance_multiple": 10.0}),
    ("terminal_resistance", {"terminal_resistance_multiple": 100.0}),
    ("outlet", {}),
    ("estimated", {}),
)
CAPILLARY_MAX_DIAMETER_UM = 8.0


def _designated_from_solve(G, inlet_pa: float, outlet_pa: float):
    from haemolynx.graph import FLOW_SOLVED

    solved = G.edge_subgraph([e for e in G.edges(keys=True) if G.edges[e].get(FLOW_SOLVED)])
    ends = [n for n in solved if solved.degree(n) == 1]
    pressure = {n: G.nodes[n].get("pressure") for n in ends}
    inlets = [n for n in ends if pressure[n] is not None and np.isclose(pressure[n], inlet_pa)]
    outlets = [n for n in ends if pressure[n] is not None and np.isclose(pressure[n], outlet_pa)]
    return inlets, outlets


def _solve(G, conditions, inlets, outlets, inlet_pa, outlet_pa):
    from haemolynx import haemodynamics as hd

    conductance, nodes = hd.build_conductance_matrix_from_graph(G, solver="sparse")
    matrix, names, index, extra_in, extra_out = hd.open_ends.augment(conductance, nodes, conditions)
    flow = hd.solve_flow_from_conductance_matrix(
        matrix, names, inlet_p_bc=inlet_pa, outlet_p_bc=outlet_pa,
        inlet_nodes=[*inlets, *extra_in], outlet_nodes=[*outlets, *extra_out], node_to_idx=index,
    )
    pressure = flow["pressure"][: len(nodes)]
    hd.set_edge_flows(G, nodes, pressure)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    inflow = hd.resistance.boundary_flow(conductance, pressure, [node_to_idx[n] for n in inlets])
    through = hd.open_ends.open_end_flows(conductance, pressure, node_to_idx, conditions,
                                          inlet_p_bc=inlet_pa, outlet_p_bc=outlet_pa)
    return inflow, through


def _within(G, margin_um, extent):
    if extent is None or margin_um <= 0:
        return lambda u, v: True

    def inside(n):
        p = np.asarray(G.nodes[n]["pos"], dtype=float)
        return float(np.min(np.minimum(p, extent - p))) > margin_um

    return lambda u, v: inside(u) and inside(v)


def measure(G, conditions, inlets, outlets, inlet_pa, outlet_pa, margin_um):
    from haemolynx import graph

    inflow, through = _solve(G, conditions, inlets, outlets, inlet_pa, outlet_pa)
    graph.mark_flow_solved_edges(G, inlets, outlets, conditions.nodes)
    extent = graph.dead_ends.graph_image_extent_um(G)
    rows = {}
    for label, keep in (("whole", _within(G, 0.0, extent)), (f"beyond {margin_um:g} um of a face", _within(G, margin_um, extent))):
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if keep(u, v)]
        total = sum(float(d.get("length") or 0.0) for *_e, d in edges)
        solved = [d for *_e, d in edges if d.get(graph.FLOW_SOLVED)]
        capillary = np.asarray([
            abs(float(d["flow_abs"])) for d in solved
            if d.get("flow_abs") is not None and 0 < float(d.get("diameter_um") or 0) <= CAPILLARY_MAX_DIAMETER_UM
        ])
        rows[label] = {
            "solved_share": sum(float(d.get("length") or 0.0) for d in solved) / total if total else 0.0,
            "capillary_flow_median": float(np.median(capillary)) if len(capillary) else None,
            "capillary_flow_iqr": (float(np.percentile(capillary, 75) - np.percentile(capillary, 25))
                                   if len(capillary) else None),
        }
    return {
        "network_resistance": (inlet_pa - outlet_pa) / inflow if inflow > 0 else float("inf"),
        "designated_inflow": inflow,
        "open_end_in": through["in"],
        "open_end_out": through["out"],
        "open_ends": len(conditions.nodes),
        **{f"{k} [{label}]": v for label, values in rows.items() for k, v in values.items()},
        "signs": {(u, v, k): np.sign(d.get("flow_signed") or 0.0) for u, v, k, d in G.edges(keys=True, data=True)
                  if d.get(graph.FLOW_SOLVED)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--inlet-pa", type=float, required=True)
    parser.add_argument("--outlet-pa", type=float, required=True)
    parser.add_argument("--inlets", nargs="*")
    parser.add_argument("--outlets", nargs="*")
    parser.add_argument("--mask", help=".npy segmentation, to classify dead ends with")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3)
    parser.add_argument("--image-shape", type=int, nargs=3, help="when the graph has no extent and there is no mask")
    parser.add_argument("--margin-um", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from haemolynx import graph, haemodynamics as hd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    if isinstance(G, dict):
        G = G["graph"]
    vs = tuple(args.voxel_size_zyx) if args.voxel_size_zyx else tuple(G.graph.get("image_voxel_size_zyx", (1, 1, 1)))
    if args.mask:
        from haemolynx.preprocessing import MaskSupport

        mask = np.load(args.mask).astype(bool)
        graph.dead_ends.record_image_extent(G, mask.shape, vs)
        graph.dead_ends.classify_and_mark(G, support=MaskSupport(mask, vs))
    elif args.image_shape and "image_extent_um" not in G.graph:
        graph.dead_ends.record_image_extent(G, args.image_shape, vs)
    if args.inlets is not None and args.outlets is not None:
        cast = lambda xs: [int(x) if x.lstrip("-").isdigit() else x for x in xs]
        inlets, outlets = cast(args.inlets), cast(args.outlets)
    else:
        inlets, outlets = _designated_from_solve(G, args.inlet_pa, args.outlet_pa)
    # Only vessels with a resistance take part, as in the solve.
    G = G.edge_subgraph([e for e in G.edges(keys=True) if G.edges[e].get("conductance")]).copy()
    print(f"{G.number_of_edges()} vessels, {len(inlets)} inlets, {len(outlets)} outlets")

    results = []
    for model, extra in MODELS:
        H = copy.deepcopy(G)
        conductance = node_list = None
        if model == "estimated":
            conductance, node_list = hd.build_conductance_matrix_from_graph(H, solver="sparse")
        conditions = hd.open_ends.open_end_conditions(
            model, H, inlet_nodes=inlets, outlet_nodes=outlets, inlet_p_bc=args.inlet_pa,
            outlet_p_bc=args.outlet_pa, conductance=conductance, node_list=node_list, **extra,
        )
        row = measure(H, conditions, inlets, outlets, args.inlet_pa, args.outlet_pa, args.margin_um)
        row["model"] = model + (f" k={extra['terminal_resistance_multiple']:g}" if extra else "")
        row["info"] = json.dumps({k: v for k, v in conditions.info.items() if not isinstance(v, list)})
        results.append(row)
    estimated = next(r for r in results if r["model"] == "estimated")["signs"]
    for row in results:
        shared = [e for e in row["signs"] if e in estimated and row["signs"][e] and estimated[e]]
        row["direction_differs_from_estimate"] = (
            sum(row["signs"][e] != estimated[e] for e in shared) / len(shared) if shared else None
        )
        del row["signs"]
    keys = ["model", *[k for k in results[0] if k != "model"]]
    with open(out / "open_end_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    shown = ["model", "open_ends", "network_resistance", "designated_inflow", "open_end_in", "open_end_out",
             "solved_share [whole]", f"solved_share [beyond {args.margin_um:g} um of a face]",
             "capillary_flow_median [whole]", f"capillary_flow_median [beyond {args.margin_um:g} um of a face]",
             "direction_differs_from_estimate"]

    def cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    lines = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    lines += ["| " + " | ".join(cell(r[k]) for k in shown) + " |" for r in results]
    (out / "REPORT.md").write_text("# Open-end sensitivity\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
