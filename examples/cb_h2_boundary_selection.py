#!/usr/bin/env python3
"""T0.2: settling the boundary conditions, by measuring the alternatives.

    python3 examples/cb_h2_boundary_selection.py

S10 established that about 86% of degree-1 nodes in these graphs are interior skeletonisation
spurs rather than vessels crossing a region face, so the band rule assigns arterial pressure
mostly to mask defects. S20 established that the resulting boundary choice moves a
within-specimen shunt ratio by more than calibre error does.

This compares the band rule against a face-crossing rule on the six CB graphs. The quantity is
the shunt ratio, flow through the widest decile of edges over total inlet throughput, and the
comparison is the spread of that ratio as each rule's free parameters move.

**Inputs are the batch outputs**, opened through ``batch_outputs.open_batch_run``: the cached
MultiGraph for topology and ``per_edge_morphometry.csv`` for calibre and length, checked against
the placed ROI before either is read. Until package O of the 2026-10-01 re-run notes
this script read the H1 ParaView export instead, in the VTK frame, with no ROI check. Axes are now
graph axes (z, y, x): axis 1 is y in both frames, while 0 and 2 swap relative to the older logs.

**Inlets and outlets come from the package's own rules**, ``select_boundary_terminal_nodes`` (band)
and ``select_boundary_terminal_nodes_by_face`` (face), so the face rule measured here is the one
the pipeline runs. A rule that raises (an empty band or face) counts as a failed solve. The
face-in + sink-out variant is the face rule in ``universal_sink`` mode: every terminal not on the
inlet face is an outlet. Unlike the older hand-written version it also needs a terminal on the
high face.

The conclusion is that the face rule cuts total sensitivity from 113.5% to 40.0%, and that with
the axis fixed at 1 the residual falls to 8.9% against the band rule's 73.9% (2026-09-28 networks;
13.3% against 75.8%, and 118.8% against 43.1%, on the earlier centred boxes). The ParaView-frame
run gave the same axis-1 figures but 113.7% and 44.9% in total. It read graph-order (z, y, x) node
coordinates against the mask's (x, y, z) bounds and voxel sizes, so on axes 0 and 2 the face cut
used the other axis's extent and spacing. Node coordinates sit on voxel planes, so that dropped
the plane one voxel in from the axis-0 outlet face and the axis-2 inlet face (WKY-A, axis 2: 16
inlets against 31).

Reproduces the numbers quoted in `select_boundary_terminal_nodes_by_face` and in S21 of
`h2_pipeline_capability_assessment.md`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_h2_error_propagation import (                                               # noqa: E402
    ROI, SPECIMENS, load_network, solve_edge_flows,
)
from ImageLynx.graph.boundaries import (                                            # noqa: E402
    select_boundary_terminal_nodes,
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.specimens import PROCESSING_VOXEL_UM                                 # noqa: E402


def boundaries(G, index, axis, mode, tol=1.0, percent=25.0):
    """Inlet and outlet indices under one rule. Raises ``ValueError`` on an empty band or face."""
    if mode == "band":
        inlets, outlets = select_boundary_terminal_nodes(
            G, ROI, edge_percent=percent, end_percent=percent, axis=axis,
            voxel_size=PROCESSING_VOXEL_UM)
    elif mode in ("face", "face_sink"):
        inlets, outlets = select_boundary_terminal_nodes_by_face(
            G, ROI, axis=axis, face_tolerance_voxels=tol, voxel_size=PROCESSING_VOXEL_UM,
            boundary_permeability_mode="universal_sink" if mode == "face_sink" else "caged")
    else:
        raise ValueError(f"unknown boundary mode {mode!r}")
    return (np.array([index[n] for n in inlets], int),
            np.array([index[n] for n in outlets], int))


def ratio(network, axis, mode, tol=1.0, percent=25.0):
    """Widest-decile flow over inlet throughput, or ``None`` if the rule finds no boundary."""
    G, u, v, length, diameter, index = network
    try:
        inlets, outlets = boundaries(G, index, axis, mode, tol, percent)
    except ValueError:
        return None
    q = solve_edge_flows(u, v, length, diameter, inlets, outlets, len(index))
    at_inlet = np.isin(u, inlets) | np.isin(v, inlets)
    value = q[diameter >= np.percentile(diameter, 90)].sum() / q[at_inlet].sum()
    if not np.isfinite(value):
        raise ValueError(f"axis {axis}, {mode}: shunt ratio is {value}; the solve is singular.")
    return float(value)


def _spread(values):
    return (max(values) - min(values)) / np.mean(values) * 100


def main():
    networks = {specimen_id: load_network(specimen_id) for specimen_id in SPECIMENS}
    print("axes are graph axes (z, y, x); inputs are the batch MultiGraph and morphometry CSV\n")

    for label, mode, tol in (("band 25%", "band", 1.0),
                             ("face tol=1", "face", 1.0),
                             ("face tol=2", "face", 2.0),
                             ("face tol=4", "face", 4.0),
                             ("face-in + sink-out", "face_sink", 1.0)):
        spreads, fails = [], 0
        for network in networks.values():
            ok = [r for r in (ratio(network, a, mode, tol) for a in range(3)) if r is not None]
            fails += 3 - len(ok)
            if len(ok) > 1:
                spreads.append(_spread(ok))
        print(f"  {label:20s} mean axis spread {np.mean(spreads):5.1f}%   "
              f"failed solves {fails}/18")

    print("\n=== total sensitivity: both free parameters varied ===")
    for label, mode, params in (("band  (axis x percent 10/25/40)", "band", (10.0, 25.0, 40.0)),
                                ("face  (axis x tol 1/2/4 voxels)", "face", (1.0, 2.0, 4.0))):
        spreads, fails = [], 0
        for network in networks.values():
            vals = []
            for a in range(3):
                for p in params:
                    r = (ratio(network, a, mode, tol=p) if mode == "face"
                         else ratio(network, a, mode, percent=p))
                    if r is None:
                        fails += 1
                    else:
                        vals.append(r)
            if len(vals) > 1:
                spreads.append(_spread(vals))
        print(f"  {label:34s} total spread {np.mean(spreads):5.1f}%   failed {fails}/54")

    print("\n=== which axes are usable in all six specimens? ===")
    for a in range(3):
        ok = [s for s, network in networks.items() if ratio(network, a, "face", 1.0) is not None]
        missing = sorted(set(networks) - set(ok))
        print(f"  face, axis {a}: {len(ok)}/6 solvable  "
              f"{'ALL' if not missing else 'missing ' + ', '.join(missing)}")

    print("\n=== residual sensitivity with the axis fixed ===")
    for label, mode, a, params in (("band  axis1, percent 10/25/40", "band", 1, (10.0, 25.0, 40.0)),
                                   ("face  axis1, tol 1/2/4 voxels", "face", 1, (1.0, 2.0, 4.0))):
        spreads = []
        for network in networks.values():
            vals = [(ratio(network, a, mode, tol=p) if mode == "face"
                     else ratio(network, a, mode, percent=p)) for p in params]
            vals = [r for r in vals if r is not None]
            if len(vals) > 1:
                spreads.append(_spread(vals))
        print(f"  {label:32s} spread {np.mean(spreads):5.1f}%  (per specimen: "
              + ", ".join(f"{x:.0f}%" for x in spreads) + ")")


if __name__ == "__main__":
    main()
