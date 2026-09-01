#!/usr/bin/env python3
"""H2 §2.3: glomus-specific 3D hypoxic fraction on a heterogeneous ADR grid.

    python3 examples/cb_h2_hypoxic_fraction.py

§2.3 asks for the segmented TH volume to assign distinct metabolic rates, a higher one for
TH-positive voxels and a lower one for the surrounding stroma, and then for the hypoxic
fraction strictly within the TH-positive volume.

Three defects had to be cleared before this could produce anything (S24, S25, S26). The
conjugate gradient solve was diverging under a non-SPD preconditioner; flow was coupled to the
tissue in the flow solve's own units rather than µm³/s, so the sink exceeded the source by
2.2e4; and each edge's whole flow was recorded against every cell it crossed, so the source
grew with grid resolution and the answer was not grid-convergent.

**Resolution.** 4 µm, not the native 1.866 µm. With the conservation defect fixed the solution
converges: median PO2 moves 27.34, 27.92, 28.21 at 10, 6 and 4 µm, halving its increment each
time and extrapolating to about 28.5. At 4 µm it is within roughly 1% of that limit for a
twenty-seventh of the cost of native resolution.

**The metabolic contrast is an assumption, not a measurement.** Nothing in this study measures
the ratio of glomus to stromal oxygen consumption, so it is a parameter here and the answer is
reported across a range of it rather than at one value.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py                                                              # noqa: E402
from ImageLynx.graph.boundaries import (                                 # noqa: E402
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.haemodynamics.perfusion import (                          # noqa: E402
    PerfusionGrid, build_adr_matrix, map_vessels_to_grid,
    solve_perfusion_steady_state,
)
from ImageLynx.haemodynamics.rheology import (                           # noqa: E402
    solve_coupled_flow_and_hematocrit,
)
from ImageLynx.haemodynamics.tissue_regions import (                     # noqa: E402
    blend_per_cell_rate, mask_bounds_um, mask_fraction_per_cell,
)
from ImageLynx.roi_placement import place_roi                            # noqa: E402
from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS           # noqa: E402
from ImageLynx import cb_settings                                       # noqa: E402

BATCH = Path(__file__).resolve().parents[1] / "examples/outputs/cb_h1_batch"
# Analysis settings come from ImageLynx.cb_settings, which is their single owner.
ROI = cb_settings.ROI_VOXELS
BOUNDARY_AXIS = cb_settings.BOUNDARY_AXIS
TH_THRESHOLD = cb_settings.TH_THRESHOLD
GRID_UM = cb_settings.GRID_UM
INLET_P, OUTLET_P = cb_settings.INLET_PRESSURE_MMHG, cb_settings.OUTLET_PRESSURE_MMHG
BASE_M_MAX = cb_settings.BASE_M_MAX
HYPOXIC_THRESHOLDS = (5.0, 10.0, 20.0)


class PerfConfig:
    """M_max may be a scalar or a per-cell array; the solver uses it elementwise."""

    def __init__(self, m_max):
        self.sigma_diff = 1.5e-9
        self.M_max = m_max
        self.k_reduce = 0.1
        self.C_arterial = 0.13


def _load_graph(specimen):
    with open(next((BATCH / specimen.specimen_id).glob("*_cache/network_graph.pkl")), "rb") as h:
        G = pickle.load(h)
    by_edge = {}
    with open(BATCH / specimen.specimen_id / "per_edge_morphometry.csv") as handle:
        for row in csv.DictReader(handle):
            if row.get("assigned_diameter_um"):
                by_edge[(row["u"], row["v"], row["key"])] = float(row["assigned_diameter_um"])
    for u, v, key, data in G.edges(keys=True, data=True):
        for probe in ((str(u), str(v), str(key)), (str(v), str(u), str(key))):
            if probe in by_edge:
                data["assigned_diameter_um"] = by_edge[probe]
                break
    return G


def _th_mask(specimen):
    bounds = place_roi(specimen, ROI).bounds
    with h5py.File(specimen.th_probabilities_path, "r") as handle:
        block = np.asarray(
            handle["exported_data"][bounds[0], bounds[1], bounds[2], 0], dtype=np.float32)
    if block.max() > 1.5:
        block = block / 255.0
    return block > TH_THRESHOLD


def _unsupplied_pct(q_total):
    """Share of cells receiving no oxygen source at all. Padding raises this by construction."""
    q = np.asarray(q_total, dtype=float)
    return float(100.0 * (q <= 0).mean()) if q.size else float("nan")


def analyse(specimen, contrast, grid_um=GRID_UM, pad_grid=False):
    G = _load_graph(specimen)
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, ROI, axis=BOUNDARY_AXIS, voxel_size=PROCESSING_VOXEL_UM)
    G, _ = solve_coupled_flow_and_hematocrit(G, inlets, outlets, INLET_P, OUTLET_P)

    mask = _th_mask(specimen)
    # Default: the grid stops at the vasculature, and glomus tissue beyond it is dropped (S28).
    # --pad-grid extends it to the segmented volume, which represents that tissue at the cost
    # of solving it with no vessels in it. Neither is free; see the flag's help.
    bounds = mask_bounds_um(mask.shape, PROCESSING_VOXEL_UM) if pad_grid else None
    grid = PerfusionGrid(G, (grid_um, grid_um, grid_um), bounds_zyx=bounds)
    th_fraction = mask_fraction_per_cell(mask, grid, PROCESSING_VOXEL_UM)

    # A contrast of c puts the glomus rate c times the stromal one, holding the volume-weighted
    # mean at BASE_M_MAX so the runs are comparable rather than simply scaled.
    mean_fraction = float(th_fraction.mean())
    stroma = BASE_M_MAX / (1.0 + mean_fraction * (contrast - 1.0))
    m_max = blend_per_cell_rate(th_fraction, tissue_rate=stroma * contrast, stroma_rate=stroma)

    config = PerfConfig(m_max)
    A, q, s = build_adr_matrix(grid, map_vessels_to_grid(G, grid), config)
    po2 = solve_perfusion_steady_state(grid, A, q, s, config)

    # Weighted by TH occupancy, so a cell that is 40% glomus contributes 40% of its volume.
    weight = th_fraction
    total = float(weight.sum())
    result = {
        "specimen_id": specimen.specimen_id, "group": specimen.group,
        "contrast": contrast, "grid_um": grid_um, "cells": int(grid.n_cells),
        "padded_to_segmented_volume": bool(pad_grid),
        "cells_without_vessels_pct": _unsupplied_pct(q),
        "th_volume_fraction": mean_fraction,
        "po2_median_all": float(np.median(po2)),
        "po2_median_th": float(np.average(po2, weights=weight)) if total else float("nan"),
        "po2_median_stroma": (float(np.average(po2, weights=1.0 - weight))
                              if (1.0 - weight).sum() else float("nan")),
    }
    for threshold in HYPOXIC_THRESHOLDS:
        below = (po2 < threshold).astype(float)
        result[f"hypoxic_th_{threshold:g}"] = (float((below * weight).sum() / total)
                                               if total else float("nan"))
        result[f"hypoxic_all_{threshold:g}"] = float(below.mean())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                    help="Glomus metabolic rate as a multiple of stromal. 1.0 is uniform.")
    ap.add_argument("--grid-um", type=float, default=GRID_UM)
    ap.add_argument("--pad-grid", action="store_true",
                    help="Extend the perfusion grid to the segmented volume rather than "
                         "stopping at the vascular bounding box. This represents glomus "
                         "tissue the default drops (4.35%% of SHR-A, 7.54%% of SHR-C), at the "
                         "cost of solving the added cells with no local oxygen source. "
                         "Measured on this cohort that costs 0.7 mmHg of mean PO2 within TH "
                         "and no change in hypoxic fraction, because the diffusion length "
                         "exceeds the unvascularised rim; do not assume that on a sparser bed.")
    ap.add_argument("--out", default="examples/outputs/cb_h2_hypoxic_fraction.json")
    args = ap.parse_args()

    print(f"ROI {ROI[0]}^3, perfusion grid {args.grid_um} um, TH threshold {TH_THRESHOLD}, "
          f"boundary face rule axis {BOUNDARY_AXIS}")
    print(f"Volume-weighted mean metabolic rate held at {BASE_M_MAX} across contrasts.\n")

    rows = []
    for contrast in args.contrast:
        print(f"--- glomus:stroma metabolic contrast {contrast:g} ---")
        print(f"  {'spec':7s} {'TH vol':>7s} {'PO2 TH':>8s} {'PO2 stroma':>11s} "
              + " ".join(f"{'hyp<' + format(t, 'g'):>9s}" for t in HYPOXIC_THRESHOLDS))
        for specimen in SPECIMENS:
            r = analyse(specimen, contrast, args.grid_um, pad_grid=args.pad_grid)
            rows.append(r)
            print(f"  {r['specimen_id']:7s} {100*r['th_volume_fraction']:6.1f}% "
                  f"{r['po2_median_th']:8.2f} {r['po2_median_stroma']:11.2f} "
                  + " ".join(f"{100*r[f'hypoxic_th_{t:g}']:8.2f}%" for t in HYPOXIC_THRESHOLDS))
        for field, label in ([("po2_median_th", "PO2 within TH")]
                             + [(f"hypoxic_th_{t:g}", f"hypoxic fraction < {t:g} mmHg")
                                for t in HYPOXIC_THRESHOLDS]):
            sub = [r for r in rows if r["contrast"] == contrast]
            out = {}
            for group in ("WKY", "SHR"):
                vals = [r[field] for r in sub if r["group"] == group and np.isfinite(r[field])]
                if vals:
                    out[group] = float(np.mean(vals))
            line = f"    {label:30s} " + "   ".join(f"{g} {v:.4g}" for g, v in out.items())
            if len(out) == 2 and out["WKY"]:
                line += f"   SHR/WKY {out['SHR']/out['WKY']:.2f}x"
            print(line)
        print()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {path}")
    print("\nn = 3 per group, exact two-sided permutation floor 2/C(6,3) = 0.10.")


if __name__ == "__main__":
    main()
