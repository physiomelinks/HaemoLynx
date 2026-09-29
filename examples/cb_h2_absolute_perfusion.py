#!/usr/bin/env python3
"""Absolute perfusion of the six CB networks, against a physiological capillary velocity.

    python3 examples/cb_h2_absolute_perfusion.py

H2 reports every flow quantity as a within-specimen ratio. This measures the absolute scale those
ratios sit on: how much blood each network carries at the frozen 60/20 mmHg, how fast it moves,
and what pressure drop would bring it to a physiological capillary velocity. Reference §13.5 and
S27 of `h2_pipeline_capability_assessment.md` quote these numbers; open item 12 re-derived them
after the rheology resistance fix (`7ea1b36`), which is why this script exists.

Each specimen is solved twice, under the face rule the H2 drivers use and under the band rule
(25% of the axis at each end) it replaced, so the cost of the boundary choice in throughput is
measured rather than assumed. Both rules use `cb_settings.BOUNDARY_AXIS`.

The flow solve is the coupled flow / haematocrit / viscosity solve the H2 drivers run
(`solve_coupled_flow_and_hematocrit`, pressures in mmHg), so its flow is converted to um^3/s
with `poiseuille_flow_to_um3_per_s`. Diameters come from `per_edge_morphometry.csv`, as in
`cb_h2_glomus_perfusion.py`.

Reported per specimen and rule:

- total inlet flow: the flow through every edge touching an inlet, um^3/s;
- flow-weighted velocity: sum(Q v) / sum(Q) over all edges, with v = Q / (pi d^2 / 4). Weighting
  by flow reports the velocity the blood actually experiences, not that of stagnant spurs;
- the pressure drop that would bring that velocity to `TARGET_VELOCITY_UM_S`. The solve is
  linear in the pressure drop (phase separation reads flow fractions, which scaling leaves
  alone), so this is the frozen drop scaled by target / measured.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cb_h2_glomus_perfusion import _attach_diameters, _load_graph      # noqa: E402
from ImageLynx.graph.boundaries import (                               # noqa: E402
    select_boundary_terminal_nodes,
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.haemodynamics.resistance import (                       # noqa: E402
    poiseuille_flow_to_um3_per_s,
)
from ImageLynx.haemodynamics.rheology import (                         # noqa: E402
    report_unconverged,
    rheology_status,
    solve_coupled_flow_and_hematocrit,
)
from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS         # noqa: E402
from ImageLynx import cb_settings                                     # noqa: E402

# Analysis settings come from ImageLynx.cb_settings, which is their single owner.
ROI = cb_settings.ROI_VOXELS
BOUNDARY_AXIS = cb_settings.BOUNDARY_AXIS
FACE_TOLERANCE = cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS
INLET_P, OUTLET_P = cb_settings.INLET_PRESSURE_MMHG, cb_settings.OUTLET_PRESSURE_MMHG

#: Band width of the comparison rule, percent of the axis at each end. It is the band the
#: face rule replaced (S21), measured here only for its throughput.
BAND_PERCENT = 25.0
#: Physiological capillary velocity range, um/s, and the value the pressure is scaled to.
PHYSIOLOGICAL_VELOCITY_UM_S = (200.0, 1000.0)
TARGET_VELOCITY_UM_S = 500.0

RULES = ("face", "band")


def select_boundaries(G, rule):
    """Inlets and outlets under one of the two rules, both on the frozen axis."""
    if rule == "face":
        return select_boundary_terminal_nodes_by_face(
            G, ROI, axis=BOUNDARY_AXIS, face_tolerance_voxels=FACE_TOLERANCE,
            voxel_size=PROCESSING_VOXEL_UM)
    if rule == "band":
        return select_boundary_terminal_nodes(
            G, ROI, edge_percent=BAND_PERCENT, end_percent=BAND_PERCENT,
            axis=BOUNDARY_AXIS, voxel_size=PROCESSING_VOXEL_UM)
    raise ValueError(f"unknown boundary rule {rule!r}; expected one of {RULES}")


def perfusion_summary(G, inlets, pressure_drop_mmhg,
                      target_velocity_um_s=TARGET_VELOCITY_UM_S):
    """Throughput and flow-weighted velocity of a graph the rheology solve has run on.

    Raises if any edge lacks a positive diameter or a finite flow: a missing one would drop
    that edge from both sums and return a plausible, wrong velocity.
    """
    inlet_set = set(inlets)
    flows, velocities = [], []
    inlet_flow = 0.0
    for u, v, key, data in G.edges(keys=True, data=True):
        d = data.get("assigned_diameter_um")
        if d is None or not np.isfinite(d) or d <= 0:
            raise ValueError(f"edge {(u, v, key)} has no usable diameter ({d!r}).")
        q_raw = data.get("flow_abs")
        if q_raw is None or not np.isfinite(q_raw):
            raise ValueError(f"edge {(u, v, key)} has no finite flow_abs ({q_raw!r}); "
                             f"run the rheology solve first.")
        q = poiseuille_flow_to_um3_per_s(abs(float(q_raw)))
        flows.append(q)
        velocities.append(q / (np.pi * float(d) ** 2 / 4.0))
        if u in inlet_set or v in inlet_set:
            inlet_flow += q
    flows = np.asarray(flows)
    total = flows.sum()
    if total <= 0:
        raise ValueError("the network carries no flow, so it has no flow-weighted velocity.")
    velocity = float(np.dot(flows, velocities) / total)
    return {
        "total_inlet_flow_um3_s": float(inlet_flow),
        "flow_weighted_velocity_um_s": velocity,
        "pressure_drop_mmhg": float(pressure_drop_mmhg),
        "pressure_drop_for_target_mmhg":
            float(pressure_drop_mmhg) * target_velocity_um_s / velocity,
    }


def analyse(specimen, rule):
    G = _load_graph(specimen)
    _attach_diameters(G, specimen)
    inlets, outlets = select_boundaries(G, rule)
    G, _pressure = solve_coupled_flow_and_hematocrit(
        G, inlets, outlets, INLET_P, OUTLET_P, **cb_settings.rheology_solver_kwargs())
    rheology = rheology_status(G)
    report_unconverged(rheology, f"{specimen.specimen_id} ({rule} rule)")
    summary = perfusion_summary(G, inlets, INLET_P - OUTLET_P)
    summary.update({
        "inlets": len(inlets),
        "outlets": len(outlets),
        "rheology": rheology,
    })
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="examples/outputs/cb_h2_absolute_perfusion.json")
    ap.add_argument("--specimen", action="append",
                    help="specimen id (repeatable); default all six")
    args = ap.parse_args()

    chosen = [s for s in SPECIMENS if not args.specimen or s.specimen_id in args.specimen]
    if args.specimen and len(chosen) != len(set(args.specimen)):
        known = ", ".join(s.specimen_id for s in SPECIMENS)
        raise SystemExit(f"unknown specimen in {args.specimen}; known: {known}")

    print(f"ROI {ROI[0]}^3, axis {BOUNDARY_AXIS}, {INLET_P:g}/{OUTLET_P:g} mmHg, "
          f"target {TARGET_VELOCITY_UM_S:g} um/s")
    results = []
    for specimen in chosen:
        row = {"specimen_id": specimen.specimen_id, "group": specimen.group}
        for rule in RULES:
            row[rule] = analyse(specimen, rule)
        results.append(row)

    for rule in RULES:
        print(f"\n{rule} rule")
        print(f"  {'spec':7s} {'inlets':>6s} {'outlets':>7s} {'inlet Q um3/s':>14s} "
              f"{'v_fw um/s':>10s} {'dP for target':>14s} {'rheology':>16s}")
        for row in results:
            r = row[rule]
            print(f"  {row['specimen_id']:7s} {r['inlets']:6d} {r['outlets']:7d} "
                  f"{r['total_inlet_flow_um3_s']:14.4g} "
                  f"{r['flow_weighted_velocity_um_s']:10.2f} "
                  f"{r['pressure_drop_for_target_mmhg']:11.0f} mmHg "
                  f"{r['rheology']['rheology_stop_reason']}/"
                  f"{r['rheology']['rheology_iterations']}")
    lo, hi = PHYSIOLOGICAL_VELOCITY_UM_S
    print(f"\nPhysiological capillary velocity {lo:g}-{hi:g} um/s.")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
