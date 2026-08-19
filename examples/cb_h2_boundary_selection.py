#!/usr/bin/env python3
"""T0.2: settling the boundary conditions, by measuring the alternatives.

    python3 examples/cb_h2_boundary_selection.py

S10 established that about 86% of degree-1 nodes in these graphs are interior skeletonisation
spurs rather than vessels crossing a region face, so the band rule assigns arterial pressure
mostly to mask defects. S20 established that the resulting boundary choice moves a
within-specimen shunt ratio by more than calibre error does.

This compares the band rule against a face-crossing rule on the six CB3 graphs. The quantity is
the shunt ratio, flow through the widest decile of edges over total inlet throughput, and the
comparison is the spread of that ratio as each rule's free parameters move.

The conclusion is that the face rule cuts total sensitivity from 118.8% to 43.1%, and that with
the axis fixed at 1, the only axis solvable in all six specimens, the residual falls to 13.3%
against the band rule's 75.8%. That is below the operative floor S20 reported and below the
27 to 40% effects H1 measures.

Reproduces the numbers quoted in `select_boundary_terminal_nodes_by_face` and in S21 of
`h2_pipeline_capability_assessment.md`.
"""
import sys, numpy as np
sys.path.insert(0, "/home/dsas627/PycharmProjects/ImageLynx/examples")
sys.path.insert(0, "/home/dsas627/PycharmProjects/ImageLynx/src")
from cb_h2_error_propagation import load, solve_edge_flows
from ImageLynx.specimens import SPECIMENS, PROCESSING_VOXEL_UM
VOX = np.asarray(PROCESSING_VOXEL_UM)

def ratio(specimen_id, axis, mode, tol=1.0, percent=25.0):
    u0, v0, length, diameter, nodes, bounds = load(specimen_id)
    ids = np.unique(np.concatenate([u0, v0]))
    remap = {x: i for i, x in enumerate(ids)}
    u = np.array([remap[x] for x in u0]); v = np.array([remap[x] for x in v0])
    node_id = np.asarray(nodes.point_data["node_id"]).astype(int)
    degree = np.asarray(nodes.point_data["degree"]).astype(int)
    pts = nodes.points; lo, hi = bounds[axis]
    term = degree == 1; coord = pts[term][:, axis]; tid = node_id[term]
    t = tol * VOX[::-1][axis]
    if mode.startswith("face"):
        inl = tid[coord <= lo + t]
        out = tid[coord >= hi - t] if mode == "face" else tid[coord > lo + t]
    else:
        e = hi - lo
        inl = tid[coord <= lo + e * percent / 100.0]
        out = tid[coord >= lo + e * (1 - percent / 100.0)]
    inl = np.array([remap[x] for x in inl if x in remap], int)
    out = np.array([remap[x] for x in out if x in remap], int)
    out = out[~np.isin(out, inl)]
    if not len(inl) or not len(out): return None
    q = solve_edge_flows(u, v, length, diameter, inl, out, len(ids))
    if q is None: return None
    at = np.isin(u, inl) | np.isin(v, inl)
    tot = np.abs(q[at]).sum()
    if tot <= 0: return None
    return float(np.abs(q[diameter >= np.percentile(diameter, 90)]).sum() / tot)

for label, mode, tol in (("band 25%", "band", 1.0),
                         ("face tol=1", "face", 1.0),
                         ("face tol=2", "face", 2.0),
                         ("face tol=4", "face", 4.0),
                         ("face-in + sink-out", "face_sink", 1.0)):
    spreads, fails = [], 0
    for s in SPECIMENS:
        vals = [ratio(s.specimen_id, a, mode, tol) for a in range(3)]
        ok = [v for v in vals if v is not None]
        fails += 3 - len(ok)
        if len(ok) > 1:
            spreads.append((max(ok) - min(ok)) / np.mean(ok) * 100)
    print(f"  {label:20s} mean axis spread {np.mean(spreads):5.1f}%   "
          f"failed solves {fails}/18")

print("\n=== total sensitivity: both free parameters varied ===")
for label, mode, params in (("band  (axis x percent 10/25/40)", "band", (10.0, 25.0, 40.0)),
                            ("face  (axis x tol 1/2/4 voxels)", "face", (1.0, 2.0, 4.0))):
    spreads, fails = [], 0
    for s in SPECIMENS:
        vals = []
        for a in range(3):
            for p in params:
                r = (ratio(s.specimen_id, a, mode, tol=p) if mode == "face"
                     else ratio(s.specimen_id, a, mode, percent=p))
                if r is None: fails += 1
                else: vals.append(r)
        if len(vals) > 1:
            spreads.append((max(vals) - min(vals)) / np.mean(vals) * 100)
    print(f"  {label:34s} total spread {np.mean(spreads):5.1f}%   failed {fails}/54")

print("\n=== which axes are usable in all six specimens? ===")
for a in range(3):
    ok = [s.specimen_id for s in SPECIMENS if ratio(s.specimen_id, a, "face", 1.0) is not None]
    print(f"  face, axis {a}: {len(ok)}/6 solvable  {'ALL' if len(ok)==6 else 'missing ' + str(set(x.specimen_id for x in SPECIMENS) - set(ok))}")

print("\n=== residual sensitivity with the axis fixed ===")
for label, mode, a, params in (("band  axis1, percent 10/25/40", "band", 1, (10.0, 25.0, 40.0)),
                               ("face  axis1, tol 1/2/4 voxels", "face", 1, (1.0, 2.0, 4.0))):
    spreads = []
    for s in SPECIMENS:
        vals = [(ratio(s.specimen_id, a, mode, tol=p) if mode == "face"
                 else ratio(s.specimen_id, a, mode, percent=p)) for p in params]
        vals = [v for v in vals if v is not None]
        if len(vals) > 1:
            spreads.append((max(vals) - min(vals)) / np.mean(vals) * 100)
    print(f"  {label:32s} spread {np.mean(spreads):5.1f}%  (per specimen: "
          + ", ".join(f"{x:.0f}%" for x in spreads) + ")")
