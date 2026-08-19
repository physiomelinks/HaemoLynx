#!/usr/bin/env python3
"""ParaView-ready VTK artefacts for the H2 analysis and the glomus segmentation.

    python3 examples/cb_h2_vtk.py --verify        # check the frames agree, write nothing
    python3 examples/cb_h2_vtk.py                 # write all six specimens

The H1 exporter writes the vascular reconstruction. What it cannot show is anything involving
the second channel or the flow solve: where the glomus clusters sit, which capillaries
penetrate them, what haematocrit those capillaries carry, and what the tissue oxygen field
looks like around them. Those are the four things H2 measures, and they are far easier to judge
by eye than from a table of ratios.

Writes, per specimen, into one shared directory so all six load into a single ParaView session:

    <SPEC>_glomus_prob.vti      TH probability at native 1.866 um, ROI-cropped
    <SPEC>_glomus_surface.vtp   smoothed isosurface of the glomus mask
    <SPEC>_glomus_clusters.vtp  the same surface split into connected nests, each with an id
                                and its volume, so a single cluster can be isolated
    <SPEC>_perfusion.vti        4 um ADR grid: PO2, TH fraction, metabolic rate, flow, source
    <SPEC>_vessels_h2.vtp       centrelines carrying flow, haematocrit, viscosity, TH fraction,
                                transit time and the penetrating/bypassing classification

**Axis convention.** Identical to the H1 exporter, and it is not obvious. Arrays are (z, y, x)
and ImageData is written with ``dimensions = shape``, ``spacing = PROCESSING_VOXEL_UM`` and
``origin`` in the same order, so **VTK's x axis carries the array's z axis**. Graph points are
stored (z, y, x) in micrometres and land the same way. Everything therefore overlays without a
transform, and the perfusion grid, which is 4 um where the glomus mask is 1.866 um, lands in the
same physical frame.

``--verify`` checks that against the data rather than trusting the reasoning. On WKY-C, edges
whose centreline is more than half inside the mask have their midpoint inside the exported
volume 90.7% of the time, against 0.8% for edges less than a tenth inside, and 39.5% if the
volume is transposed. That is the check, not the argument.

Coverage is reported separately and does not block writing. The perfusion grid is the graph's
node bounding box, so a specimen whose vessels stop short of the region edge gets a grid
smaller than the glomus mask: SHR-A loses 4.35% of its glomus volume that way and SHR-C 7.54%.
Those exports are correctly registered and simply have no oxygen field in the gap.
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
import pyvista as pv                                                     # noqa: E402
from scipy import ndimage as ndi                                         # noqa: E402

from ImageLynx.graph.boundaries import (                                 # noqa: E402
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.haemodynamics.perfusion import (                          # noqa: E402
    PerfusionGrid, build_adr_matrix, map_vessels_to_grid,
    solve_perfusion_steady_state,
)
from ImageLynx.haemodynamics.resistance import (                         # noqa: E402
    poiseuille_flow_to_um3_per_s,
)
from ImageLynx.haemodynamics.rheology import (                           # noqa: E402
    solve_coupled_flow_and_hematocrit,
)
from ImageLynx.haemodynamics.tissue_regions import (                     # noqa: E402
    blend_per_cell_rate, edge_tissue_fraction, mask_fraction_per_cell,
)
from ImageLynx.haemodynamics.transit import transit_time_from_inlets     # noqa: E402
from ImageLynx.roi_placement import place_roi                            # noqa: E402
from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS           # noqa: E402

BATCH = Path(__file__).resolve().parents[1] / "examples/outputs/cb_h1_batch"
OUT = Path(__file__).resolve().parents[1] / "examples/outputs/cb_h2_paraview"
ROI = (160, 160, 160)
BOUNDARY_AXIS = 1
TH_THRESHOLD = 0.5
GRID_UM = 4.0
INLET_P, OUTLET_P = 60.0, 20.0
PENETRATION = 0.5
BASE_M_MAX = 0.05


class PerfConfig:
    def __init__(self, m_max):
        self.sigma_diff = 1.5e-9
        self.M_max = m_max
        self.k_reduce = 0.1
        self.C_arterial = 0.13


# --- inputs ---------------------------------------------------------------------------------

def load_graph(specimen):
    """The cached graph, with calibre attached from the morphometry export.

    The graph carries no diameter of its own. Without this the flow solve silently falls back,
    and transit time, which is quadratic in diameter, would be fabricated with it.
    """
    with open(next((BATCH / specimen.specimen_id).glob("*_cache/network_graph.pkl")), "rb") as h:
        G = pickle.load(h)
    by_edge = {}
    with open(BATCH / specimen.specimen_id / "per_edge_morphometry.csv") as handle:
        for row in csv.DictReader(handle):
            if row.get("assigned_diameter_um"):
                by_edge[(row["u"], row["v"], row["key"])] = float(row["assigned_diameter_um"])
    attached = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        for probe in ((str(u), str(v), str(key)), (str(v), str(u), str(key))):
            if probe in by_edge:
                data["assigned_diameter_um"] = by_edge[probe]
                attached += 1
                break
    return G, attached


def load_th(specimen):
    bounds = place_roi(specimen, ROI).bounds
    with h5py.File(specimen.th_probabilities_path, "r") as handle:
        block = np.asarray(
            handle["exported_data"][bounds[0], bounds[1], bounds[2], 0], dtype=np.float32)
    return block / 255.0 if block.max() > 1.5 else block


def solve(specimen):
    """Everything the exports need, computed once."""
    G, attached = load_graph(specimen)
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, ROI, axis=BOUNDARY_AXIS, voxel_size=PROCESSING_VOXEL_UM)
    G, _ = solve_coupled_flow_and_hematocrit(G, inlets, outlets, INLET_P, OUTLET_P)

    prob = load_th(specimen)
    mask = prob > TH_THRESHOLD
    frac = edge_tissue_fraction(G, mask, PROCESSING_VOXEL_UM)
    arrival = transit_time_from_inlets(G, inlets)

    grid = PerfusionGrid(G, (GRID_UM, GRID_UM, GRID_UM))
    th_cell = mask_fraction_per_cell(mask, grid, PROCESSING_VOXEL_UM)
    stroma = BASE_M_MAX / (1.0 + float(th_cell.mean()) * (2.0 - 1.0))
    m_max = blend_per_cell_rate(th_cell, tissue_rate=stroma * 2.0, stroma_rate=stroma)
    config = PerfConfig(m_max)
    A, q_total, s_incoming = build_adr_matrix(grid, map_vessels_to_grid(G, grid), config)
    po2 = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config)

    return dict(graph=G, inlets=inlets, outlets=outlets, attached=attached,
                prob=prob, mask=mask, edge_fraction=frac, arrival=arrival,
                grid=grid, th_cell=th_cell, m_max=m_max,
                q_total=q_total, s_incoming=s_incoming, po2=po2)


# --- writers --------------------------------------------------------------------------------

def _image(shape_zyx, spacing_zyx, origin_zyx=(0.0, 0.0, 0.0)):
    """ImageData whose VTK x axis carries the array's z axis. See the module docstring."""
    img = pv.ImageData()
    img.dimensions = tuple(int(n) + 1 for n in shape_zyx)   # cell data needs point dims + 1
    img.spacing = tuple(float(v) for v in spacing_zyx)
    img.origin = tuple(float(v) for v in origin_zyx)
    return img


def write_glomus_volume(state, stem, out_dir):
    img = _image(state["prob"].shape, PROCESSING_VOXEL_UM)
    img.cell_data["th_probability"] = state["prob"].flatten(order="F")
    img.cell_data["glomus"] = state["mask"].astype(np.uint8).flatten(order="F")
    path = out_dir / f"{stem}_glomus_prob.vti"
    img.save(path)
    return path


def write_glomus_surface(state, stem, decimate, out_dir):
    surface = _glomus_surface(state, decimate)
    path = out_dir / f"{stem}_glomus_surface.vtp"
    surface.save(path)
    return path, surface


def _glomus_surface(state, decimate):
    img = _image(state["mask"].shape, PROCESSING_VOXEL_UM)
    img.cell_data["glomus"] = state["mask"].astype(np.float32).flatten(order="F")
    surface = img.cell_data_to_point_data().contour([0.5], scalars="glomus")
    surface = surface.smooth(n_iter=30, relaxation_factor=0.1)
    if decimate and surface.n_points:
        surface = surface.triangulate().decimate(decimate)
    return surface


def write_glomus_clusters(state, stem, decimate, out_dir):
    """The same surface split into connected nests, each carrying its id and volume.

    Nothing in H1 or H2 measures a per-cluster quantity, so this is for inspection rather than
    analysis: it makes a single nest selectable in ParaView by thresholding on cluster_id.
    """
    labels, n = ndi.label(state["mask"])
    voxel_volume = float(np.prod(PROCESSING_VOXEL_UM))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    volumes = sizes.astype(np.float64) * voxel_volume

    img = _image(labels.shape, PROCESSING_VOXEL_UM)
    img.cell_data["label"] = labels.astype(np.float32).flatten(order="F")
    surface = _glomus_surface(state, decimate)
    if surface.n_points:
        # Sample the label volume at the surface, so every vertex knows its nest.
        sampled = surface.sample(img.cell_data_to_point_data())
        ids = np.rint(np.asarray(sampled["label"])).astype(int)
        ids = np.clip(ids, 0, len(volumes) - 1)
        surface.point_data["cluster_id"] = ids
        surface.point_data["cluster_volume_um3"] = volumes[ids]
    path = out_dir / f"{stem}_glomus_clusters.vtp"
    surface.save(path)
    return path, int(n), volumes


def write_perfusion(state, stem, out_dir):
    grid = state["grid"]
    img = _image(grid.dims, grid.res, grid.min_xyz)
    for name, array in (("PO2_mmHg", state["po2"]),
                        ("th_fraction", state["th_cell"]),
                        ("metabolic_rate", np.asarray(state["m_max"], dtype=float)),
                        ("q_total_um3_s", state["q_total"]),
                        ("s_incoming", state["s_incoming"])):
        img.cell_data[name] = np.asarray(array, dtype=np.float64)
    path = out_dir / f"{stem}_perfusion.vti"
    img.save(path)
    return path


def write_vessels(state, stem, out_dir):
    """Centrelines as polylines, carrying every per-edge quantity H2 measures."""
    G, frac, arrival = state["graph"], state["edge_fraction"], state["arrival"]
    points, lines, cells = [], [], []
    for u, v, key, data in G.edges(keys=True, data=True):
        poly = np.asarray(data.get("voxels", [G.nodes[u]["pos"], G.nodes[v]["pos"]]), float)
        if len(poly) < 2:
            continue
        start = len(points)
        points.extend(poly.tolist())
        lines.extend([len(poly)] + list(range(start, start + len(poly))))
        f = float(frac.get((u, v, key), np.nan))
        reach = max(arrival.get(u, np.inf), arrival.get(v, np.inf))
        cells.append({
            "th_fraction": f,
            "penetrating": 1.0 if f >= PENETRATION else 0.0,
            "flow_um3_s": poiseuille_flow_to_um3_per_s(abs(float(data.get("flow_abs", np.nan)))),
            "hematocrit": float(data.get("hematocrit", np.nan)),
            "viscosity_cP": float(data.get("viscosity", np.nan)),
            "diameter_um": float(data.get("assigned_diameter_um", np.nan)),
            "length_um": float(data.get("length", np.nan)),
            # inf is not writable to VTK; unreachable edges become nan, which ParaView hides.
            "transit_time": float(reach) if np.isfinite(reach) else np.nan,
        })
    mesh = pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines, dtype=np.int64))
    for name in cells[0]:
        mesh.cell_data[name] = np.array([c[name] for c in cells], dtype=float)
    path = out_dir / f"{stem}_vessels_h2.vtp"
    mesh.save(path)
    return path, mesh


# --- verification ---------------------------------------------------------------------------

def verify(specimen, state):
    """Check the three frames against data, not against the reasoning in the docstring."""
    mask, frac, G = state["mask"], state["edge_fraction"], state["graph"]
    voxel = np.asarray(PROCESSING_VOXEL_UM)

    def midpoint_inside(volume):
        hits, fracs = [], []
        for (u, v, key), f in frac.items():
            poly = np.asarray(G.edges[u, v, key].get(
                "voxels", [G.nodes[u]["pos"], G.nodes[v]["pos"]]), float)
            idx = np.floor(poly[len(poly) // 2] / voxel).astype(int)
            if np.all((idx >= 0) & (idx < volume.shape)):
                hits.append(bool(volume[idx[0], idx[1], idx[2]]))
                fracs.append(f)
        return np.array(hits), np.array(fracs)

    hits, fracs = midpoint_inside(mask)
    inside = 100.0 * hits[fracs > PENETRATION].mean()
    outside = 100.0 * hits[fracs < 0.1].mean()
    rev_hits, rev_fracs = midpoint_inside(np.ascontiguousarray(mask.transpose(2, 1, 0)))
    reversed_inside = 100.0 * rev_hits[rev_fracs > PENETRATION].mean()

    grid = state["grid"]
    grid_lo = np.asarray(grid.min_xyz)
    grid_hi = grid_lo + np.asarray(grid.dims) * np.asarray(grid.res)
    mask_hi = np.asarray(mask.shape) * voxel
    contains = bool(np.all(grid_lo <= 0.0) and np.all(grid_hi >= mask_hi - 1e-6))

    # How much glomus tissue the grid does not reach. The grid is the graph's node bounding
    # box, so a specimen whose vessels stop short of the region edge leaves real tissue with
    # no oxygen field. That is a coverage fact about the specimen, not a registration fault,
    # and the two must not share a verdict: conflating them refuses sound exports for SHR-A
    # and SHR-C, whose frames are as well registered as any specimen that passes.
    idx = np.argwhere(mask)
    if idx.size:
        centres = (idx + 0.5) * voxel
        covered = np.all((centres >= grid_lo) & (centres < grid_hi), axis=1)
        outside_pct = 100.0 * float((~covered).mean())
    else:
        outside_pct = 0.0

    perfused = state["q_total"] > 0
    node_cells = set()
    for node in G.nodes:
        pos = G.nodes[node].get("pos")
        if pos is None:
            continue
        cell = grid.get_cell_index(np.asarray(pos, dtype=float))
        if cell != -1:
            node_cells.add(int(cell))
    node_perfused = (100.0 * np.mean([perfused[c] for c in node_cells])) if node_cells else 0.0

    # Registration only. A transposed or shifted volume produces an overlay that renders
    # perfectly and is wrong; a grid that does not span the mask produces one that is right
    # as far as it goes. Only the first is a reason to refuse to write.
    ok = inside > 70.0 and outside < 15.0 and inside > reversed_inside * 1.5
    return {
        "specimen_id": specimen.specimen_id,
        "penetrating_midpoint_inside_pct": round(float(inside), 1),
        "non_penetrating_midpoint_inside_pct": round(float(outside), 1),
        "transposed_control_pct": round(float(reversed_inside), 1),
        "graph_nodes_in_perfused_cells_pct": round(float(node_perfused), 1),
        "perfusion_grid_contains_glomus_volume": contains,
        "glomus_outside_grid_pct": round(outside_pct, 2),
        "ok": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="Check the frames against data and stop without writing.")
    ap.add_argument("--decimate", type=float, default=0.9,
                    help="Surface decimation fraction. 0 disables. The undecimated glomus "
                         "surface runs to tens of megabytes per specimen.")
    ap.add_argument("--specimen", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    chosen = [s for s in SPECIMENS
              if args.specimen is None or s.specimen_id in args.specimen]
    if not args.verify:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary, checks = [], []
    for specimen in chosen:
        state = solve(specimen)
        check = verify(specimen, state)
        checks.append(check)
        status = "ok" if check["ok"] else "FAILED"
        print(f"  {specimen.specimen_id}: frames {status}  "
              f"penetrating {check['penetrating_midpoint_inside_pct']}% inside, "
              f"non-penetrating {check['non_penetrating_midpoint_inside_pct']}%, "
              f"transposed control {check['transposed_control_pct']}%")
        if not check["perfusion_grid_contains_glomus_volume"]:
            print(f"    note: {check['glomus_outside_grid_pct']}% of the glomus volume lies "
                  f"outside the perfusion grid and carries no PO2. The grid is the graph's "
                  f"node bounding box, so this is where the vessels stop, not a frame fault.")
        if args.verify:
            continue
        if not check["ok"]:
            print(f"    refusing to write {specimen.specimen_id}: the frames do not agree, so "
                  f"any overlay would be wrong in a way that still looks plausible.")
            continue

        stem = specimen.specimen_id
        written = {}
        written["glomus_prob"] = str(write_glomus_volume(state, stem, out_dir))
        path, surface = write_glomus_surface(state, stem, args.decimate, out_dir)
        written["glomus_surface"] = str(path)
        path, n_clusters, volumes = write_glomus_clusters(state, stem, args.decimate, out_dir)
        written["glomus_clusters"] = str(path)
        written["perfusion"] = str(write_perfusion(state, stem, out_dir))
        path, mesh = write_vessels(state, stem, out_dir)
        written["vessels_h2"] = str(path)

        big = volumes[volumes > 0]
        summary.append({
            "specimen_id": specimen.specimen_id, "group": specimen.group,
            "diameters_attached": state["attached"],
            "edges": int(mesh.n_cells), "surface_points": int(surface.n_points),
            "glomus_clusters": n_clusters,
            "largest_cluster_um3": float(big.max()) if big.size else 0.0,
            "grid_cells": int(state["grid"].n_cells),
            "po2_median_mmHg": float(np.median(state["po2"])),
            "files": written, "frame_check": check,
        })
        print(f"    wrote 5 files: {mesh.n_cells} edges, {n_clusters} glomus clusters, "
              f"{surface.n_points} surface points, {state['grid'].n_cells} grid cells")

    if args.verify:
        print("\n  --verify only; nothing written.")
        return
    (out_dir / "export_summary.json").write_text(json.dumps(
        {"roi_zyx": list(ROI), "grid_um": GRID_UM, "th_threshold": TH_THRESHOLD,
         "boundary_axis": BOUNDARY_AXIS, "penetration_cutoff": PENETRATION,
         "specimens": summary}, indent=2))
    print(f"\nWrote {len(summary)} specimens to {out_dir}")


if __name__ == "__main__":
    main()
