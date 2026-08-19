#!/usr/bin/env python3
"""H2 §2.1 functional shunting and §2.2 spatial haematocrit, at the glomus cell level.

    python3 examples/cb_h2_glomus_perfusion.py

Both ask the same question about edges. §2.1: does steady-state flow shunt through
thoroughfare channels that bypass the capillaries penetrating the TH-positive clusters?
§2.2: do the vessels supplying those clusters carry a lower discharge haematocrit than the
rest, which would mean a dense capillary bed largely filled with cell-free plasma?

Edges are classified by the fraction of their centreline lying inside the TH mask, sampled
along the whole polyline rather than at the endpoints: a capillary penetrating a cluster
usually starts and ends in stroma, so an endpoint test would classify exactly the vessels
§2.1 is about as extra-glomus.

Boundary conditions use the face-crossing rule on axis 1 (S21), which is the only axis with
terminals on both faces in all six specimens and which cuts the residual boundary sensitivity
from 75.8% to 13.3%. The band rule this replaces put arterial pressure mostly on interior
skeletonisation spurs.

Diameters come from `per_edge_morphometry.csv` rather than the cached graph, which carries no
calibre: without them the solver silently falls back to 5 µm for every edge.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py                                                            # noqa: E402
from ImageLynx.graph.boundaries import (                               # noqa: E402
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.haemodynamics.rheology import (                         # noqa: E402
    solve_coupled_flow_and_hematocrit,
)
from ImageLynx.haemodynamics.tissue_regions import edge_tissue_fraction  # noqa: E402
from ImageLynx.roi_placement import place_roi                          # noqa: E402
from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS         # noqa: E402

BATCH = Path(__file__).resolve().parents[1] / "examples/outputs/cb_h1_batch"
ROI = (160, 160, 160)
BOUNDARY_AXIS = 1
TH_THRESHOLD = 0.5
#: An edge counts as penetrating when this much of its length lies inside the TH mask.
PENETRATION = 0.5
INLET_P, OUTLET_P = 60.0, 20.0        # mmHg, arteriolar to venular


def _load_graph(specimen):
    path = next((BATCH / specimen.specimen_id).glob("*_cache/network_graph.pkl"))
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _attach_diameters(G, specimen):
    """The cached graph has no calibre; the morphometry export does."""
    path = BATCH / specimen.specimen_id / "per_edge_morphometry.csv"
    by_edge = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            d = row.get("assigned_diameter_um")
            if d:
                by_edge[(row["u"], row["v"], row["key"])] = float(d)
    attached = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        for probe in ((str(u), str(v), str(key)), (str(v), str(u), str(key))):
            if probe in by_edge:
                data["assigned_diameter_um"] = by_edge[probe]
                attached += 1
                break
    return attached


def _th_mask(specimen):
    bounds = place_roi(specimen, ROI).bounds
    with h5py.File(specimen.th_probabilities_path, "r") as handle:
        block = np.asarray(
            handle["exported_data"][bounds[0], bounds[1], bounds[2], 0], dtype=np.float32)
    if block.max() > 1.5:
        block = block / 255.0
    return block > TH_THRESHOLD


def analyse(specimen):
    G = _load_graph(specimen)
    attached = _attach_diameters(G, specimen)
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, ROI, axis=BOUNDARY_AXIS, voxel_size=PROCESSING_VOXEL_UM)

    G, _pressure = solve_coupled_flow_and_hematocrit(
        G, inlets, outlets, INLET_P, OUTLET_P)

    frac = edge_tissue_fraction(G, _th_mask(specimen), PROCESSING_VOXEL_UM)

    rows = []
    for u, v, key, data in G.edges(keys=True, data=True):
        f = frac.get((u, v, key), float("nan"))
        rows.append({
            "th_fraction": f,
            # solve_coupled_flow_and_hematocrit writes flow_abs and flow_signed.
            "flow": float(data.get("flow_abs", float("nan"))),
            "hematocrit": float(data.get("hematocrit", float("nan"))),
            "diameter": float(data.get("assigned_diameter_um", float("nan"))),
            "length": float(data.get("length", 0.0)),
        })
    return {
        "specimen_id": specimen.specimen_id,
        "group": specimen.group,
        "inlets": len(inlets),
        "outlets": len(outlets),
        "diameters_attached": attached,
        "edges": rows,
    }


def _summarise(result):
    rows = [r for r in result["edges"] if np.isfinite(r["th_fraction"])]
    pen = [r for r in rows if r["th_fraction"] >= PENETRATION]
    byp = [r for r in rows if r["th_fraction"] < PENETRATION]
    if not pen or not byp:
        return None

    def q(sub, field):
        vals = np.array([r[field] for r in sub], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    flow_pen, flow_byp = q(pen, "flow"), q(byp, "flow")
    hct_pen, hct_byp = q(pen, "hematocrit"), q(byp, "hematocrit")
    total = sum(r["flow"] for r in rows) or float("nan")
    flow_share = sum(r["flow"] for r in pen) / total
    edge_share = len(pen) / len(rows)
    # The shunt index is what §2.1 actually asks. A flow share equal to the edge share means
    # flow is indifferent to the clusters; below 1 means it is being carried preferentially by
    # the vessels that bypass them, which is the shunting the method proposes to detect. The
    # flow share on its own cannot show this, because it tracks how many edges penetrate,
    # which is itself downstream of the parenchymal volume difference H1 §1.3 reports.
    return {
        "specimen_id": result["specimen_id"], "group": result["group"],
        "n_penetrating": len(pen), "n_bypass": len(byp),
        "edge_share_penetrating": edge_share,
        "shunt_index": flow_share / edge_share if edge_share else float("nan"),
        "flow_share_penetrating": flow_share,
        "median_flow_penetrating": flow_pen, "median_flow_bypass": flow_byp,
        "flow_ratio": flow_pen / flow_byp if flow_byp else float("nan"),
        "median_hct_penetrating": hct_pen, "median_hct_bypass": hct_byp,
        "hct_ratio": hct_pen / hct_byp if hct_byp else float("nan"),
        "median_diameter_penetrating": q(pen, "diameter"),
        "median_diameter_bypass": q(byp, "diameter"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="examples/outputs/cb_h2_glomus_perfusion.json")
    ap.add_argument("--penetration", type=float, default=PENETRATION)
    args = ap.parse_args()

    print(f"ROI {ROI[0]}^3, boundary face rule on axis {BOUNDARY_AXIS}, "
          f"TH threshold {TH_THRESHOLD}, penetration cutoff {args.penetration}")
    summaries = []
    for specimen in SPECIMENS:
        result = analyse(specimen)
        summary = _summarise(result)
        if summary is None:
            print(f"  {specimen.specimen_id}: no edges on one side of the cutoff")
            continue
        summaries.append(summary)
        print(f"  {specimen.specimen_id} ({specimen.group}): "
              f"{result['inlets']} inlets / {result['outlets']} outlets, "
              f"{summary['n_penetrating']} penetrating / {summary['n_bypass']} bypass, "
              f"diameters attached {result['diameters_attached']}/{len(result['edges'])}")

    print(f"\n§2.1 functional shunting")
    print(f"  {'spec':7s} {'edge share':>11s} {'flow share':>11s} {'shunt idx':>10s} "
          f"{'med Q pen':>10s} {'med Q byp':>10s} {'ratio':>7s}")
    for s in summaries:
        print(f"  {s['specimen_id']:7s} {100*s['edge_share_penetrating']:10.1f}% "
              f"{100*s['flow_share_penetrating']:10.1f}% {s['shunt_index']:10.3f} "
              f"{s['median_flow_penetrating']:10.4g} "
              f"{s['median_flow_bypass']:10.4g} {s['flow_ratio']:7.2f}")
    print("  shunt index = flow share / edge share. 1.0 means flow is indifferent to the "
          "clusters;\n  below 1 means the bypassing vessels carry disproportionately more.")

    print(f"\n§2.2 spatial haematocrit")
    print(f"  {'spec':7s} {'Hd pen':>8s} {'Hd byp':>8s} {'ratio':>7s} "
          f"{'d pen':>7s} {'d byp':>7s}")
    for s in summaries:
        print(f"  {s['specimen_id']:7s} {s['median_hct_penetrating']:8.4f} "
              f"{s['median_hct_bypass']:8.4f} {s['hct_ratio']:7.3f} "
              f"{s['median_diameter_penetrating']:6.2f}u {s['median_diameter_bypass']:6.2f}u")

    for field, label in (("shunt_index", "§2.1 shunt index"),
                         ("flow_ratio", "§2.1 median flow ratio pen/byp"),
                         ("hct_ratio", "§2.2 haematocrit ratio pen/byp"),
                         ("flow_share_penetrating", "§2.1 share of flow in penetrating edges")):
        out = {}
        for grp in ("WKY", "SHR"):
            vals = [s[field] for s in summaries
                    if s["group"] == grp and np.isfinite(s[field])]
            if vals:
                out[grp] = float(np.mean(vals))
        line = f"\n  {label}: " + "   ".join(f"{g} {v:.3f}" for g, v in out.items())
        if len(out) == 2:
            line += f"   SHR/WKY {out['SHR']/out['WKY']:.2f}x"
        print(line)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {path}")
    print("\nn = 3 per group, exact two-sided permutation floor 2/C(6,3) = 0.10.")


if __name__ == "__main__":
    main()
