"""Build ParaView-ready VTK artefacts for the H1 analysis, from outputs already on disk.

The pipeline already writes the reconstruction (``*_vessel_mask.vti``) and the analysed
centrelines (``*_vessels.vtp``) in physical micrometres, so both open in ParaView at true
scale as they stand. What it does not write is the morphometry *onto* that geometry: the
centrelines carry an assigned diameter and a branch order, and nothing else. Colouring the
network by tortuosity, by the EDT diameter that H1 section 1.2 actually reports, or by which
edges retained a known-biased radius, is not possible from the pipeline's own output.

All of it can be recovered without re-running anything. ``per_edge_morphometry.csv`` and
``*_vessels.vtp`` both carry ``(u, v, key)`` for every edge, and the join is exact - 4512 of
4512 cells on WKY-A, with ``assigned_diameter_um`` agreeing on every one, which is what
establishes that the join is correct rather than merely complete. The rows and the skeleton
come from ``batch_outputs.open_batch_run``, which checks the placed ROI before either is read,
and a cell with no row raises.

Writes, per specimen, into one shared directory so all six load into a single ParaView
session:

    <SPEC>_vessels.vtp    analysed centrelines + the full per-edge morphometry
    <SPEC>_nodes.vtp      graph nodes carrying degree - the readout of section 1.1
    <SPEC>_mask.vti       binary reconstruction (copied through, already correct)
    <SPEC>_surface.vtp    pre-extracted smoothed surface of that mask
    <SPEC>_skeleton.vtp   the raw skeleton, before pruning and smoothing

**Axis convention.** The pipeline writes ImageData with ``dimensions = binary.shape`` in
(z, y, x) order, so VTK's x axis carries the array's z axis, and graph points are stored
(z, y, x) in micrometres and land the same way. Everything therefore overlays, and the raw
skeleton is mapped identically. ``--verify`` checks this against the data rather than
trusting the reasoning, and writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ImageLynx.artefact_provenance import read_provenance          # noqa: E402
from ImageLynx.batch_outputs import open_batch_run                  # noqa: E402
from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS     # noqa: E402
from ImageLynx import cb_settings                                 # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "outputs" / "cb_h1_paraview"
FROZEN_THRESHOLD = cb_settings.FROZEN_THRESHOLD

#: Numeric codes for the categorical provenance tags. ParaView cannot colour by a string
#: array, so each tag is written twice: the string for reading, the code for colouring.
CODES = {
    "diameter_provenance": ["measured_edt", "measured_fwhm", "constant",
                            "synthetic_branch_order"],
    "edt_junction_trim": ["trimmed", "untrimmed_too_short", "no_junction", "not_applied"],
    "centreline_smoothing": ["bspline", "bspline_relaxed", "raw_fallback", "raw_too_short",
                             "raw_unchanged"],
}

# Numbers copied from the edge table onto every cell, through the reader, so a blank or
# non-finite cell raises. The pipeline's .vtp already carries the same values; the table is the
# one source. fwhm_diameter_um is left out: the CB pipeline never measures it, so the .vtp's
# array is left as the pipeline wrote it.
FLOAT_COLUMNS = ("length_um", "euclidean_um", "tortuosity", "curvature",
                 "edt_diameter_um", "assigned_diameter_um")
INT_COLUMNS = ("n_centreline_points",)


def stamp(mesh, specimen, extra=None):
    """Identity travels with the geometry, so six specimens can share one session."""
    mesh.field_data["specimen_id"] = np.array([specimen.specimen_id])
    mesh.field_data["group"] = np.array([specimen.group])
    mesh.field_data["group_code"] = np.array([0 if specimen.group == "WKY" else 1])
    mesh.field_data["roi_voxels"] = np.array([160, 160, 160])
    mesh.field_data["voxel_um_zyx"] = np.asarray(PROCESSING_VOXEL_UM, dtype=float)
    mesh.field_data["hysteresis_low"] = np.array([FROZEN_THRESHOLD])
    record = read_provenance(specimen.probabilities_path)
    if record:
        mesh.field_data["classifier_sha256"] = np.array([record["classifier_sha256"]])
        mesh.field_data["classifier_name"] = np.array([record["classifier_name"]])
    for key, value in (extra or {}).items():
        mesh.field_data[key] = np.asarray(value)
    return mesh


def _check_tags(run, keys, edges, column, known):
    unknown = [k for k in keys if edges[k][column] not in known]
    if unknown:
        raise ValueError(
            f"{run.specimen.specimen_id} ({run.run_dir}): {len(unknown)} cells have a {column} "
            f"the export does not know (one of {sorted(known)}): {unknown[:10]}, e.g. "
            f"{edges[unknown[0]][column]!r}.")


def enrich_vessels(run, edges, report):
    """The run's analysed centrelines, with every cell's edge-table row copied onto it.

    The join is strict: a cell whose ``(u, v, key)`` has no row raises, and so does a blank or
    non-finite number in a matched row (read through ``BatchRun.numeric_column``), rather than
    being written as NaN, 0 or "" and read in ParaView as a measurement.
    """
    src = run.run_dir / "resistance_network_vessels.vtp"
    if not src.exists():
        return None
    mesh = pv.read(src)
    keys = list(zip(mesh.cell_data["edge_u"].tolist(),
                    mesh.cell_data["edge_v"].tolist(),
                    mesh.cell_data["edge_key"].tolist()))
    missing = [k for k in keys if k not in edges]
    if missing:
        raise ValueError(
            f"{run.specimen.specimen_id}: {len(missing)} of {len(keys)} cells in {src} have "
            f"no edge-table row: {missing[:10]}.")
    report["vessels_cells"] = mesh.n_cells
    report["vessels_matched"] = len(keys)

    for column in FLOAT_COLUMNS:
        values = run.numeric_column(column)
        mesh.cell_data[column] = np.array([values[k] for k in keys], dtype=float)
    for column in INT_COLUMNS:
        values = run.numeric_column(column)
        fractional = [k for k in keys if not values[k].is_integer()]
        if fractional:
            raise ValueError(
                f"{run.specimen.specimen_id} ({run.run_dir}): {len(fractional)} cells have a "
                f"fractional {column} in the edge table: {fractional[:10]}.")
        mesh.cell_data[column] = np.array([int(values[k]) for k in keys], dtype=np.int32)

    # A tag the export does not know raises rather than being coded as 0 or -1.
    flags = {"False": 0, "True": 1}
    _check_tags(run, keys, edges, "reconnected", flags)
    mesh.cell_data["reconnected"] = np.array(
        [flags[edges[k]["reconnected"]] for k in keys], dtype=np.int8)

    for column, levels in CODES.items():
        lookup = {name: i for i, name in enumerate(levels)}
        _check_tags(run, keys, edges, column, lookup)
        values = [edges[k][column] for k in keys]
        mesh.cell_data[column] = np.asarray(values, dtype=f"<U{max(len(l) for l in levels)}")
        mesh.cell_data[f"{column}_code"] = np.array([lookup[v] for v in values], dtype=np.int8)

    # Radius is the natural glyph scale in ParaView and is worth having ready-made.
    mesh.cell_data["radius_um"] = mesh.cell_data["assigned_diameter_um"] / 2.0
    return stamp(mesh, run.specimen)


def build_nodes(run, edges, report):
    src = run.run_dir / "resistance_network_nodes.vtp"
    if not src.exists():
        return None
    mesh = pv.read(src)
    degree = {}
    for (u, v, _k) in edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    ids = mesh.point_data["node_id"].tolist()
    values = np.array([degree.get(int(i), 0) for i in ids], dtype=np.int32)
    mesh.point_data["degree"] = values
    # Section 1.1 counts branch points: nodes joining three or more distinct segments.
    mesh.point_data["is_branch_node"] = (values >= 3).astype(np.int8)
    mesh.point_data["is_endpoint"] = (values == 1).astype(np.int8)
    report["nodes"] = mesh.n_points
    report["branch_nodes"] = int((values >= 3).sum())
    return stamp(mesh, run.specimen)


def build_skeleton(run, report):
    """Raw skeleton voxels as points, in the same frame as the mask and the centrelines."""
    iz, iy, ix = np.nonzero(run.skeleton())
    vz, vy, vx = PROCESSING_VOXEL_UM
    # Array (z, y, x) maps to VTK (x, y, z) - the convention the mask .vti already uses.
    points = np.column_stack([iz * vz, iy * vy, ix * vx]).astype(float)
    mesh = pv.PolyData(points)
    mesh.point_data["skeleton"] = np.ones(len(points), dtype=np.uint8)
    report["skeleton_voxels"] = int(len(points))
    return stamp(mesh, run.specimen)


def build_surface(run, report, smoothing=30):
    src = run.run_dir / "resistance_network_vessel_mask.vti"
    if not src.exists():
        return None, None
    grid = pv.read(src)
    surface = grid.contour([0.5], scalars="vessel_mask")
    if surface.n_points:
        surface = surface.smooth(n_iter=smoothing, relaxation_factor=0.1)
        surface = surface.compute_normals(auto_orient_normals=True)
    report["surface_points"] = surface.n_points
    return stamp(surface, run.specimen), grid


def verify(specimen, vessels, skeleton, grid):
    """Check the frames agree, rather than trusting the axis reasoning.

    Overhang and inset are reported separately, because only one of them can indicate a
    fault. Centrelines sitting *inside* the mask bounds mean the network does not reach that
    face of the region, which is ordinary anatomy. Centrelines sitting *outside* them would
    mean the frames disagree - except that B-spline smoothing legitimately bulges a
    centreline a fraction of a voxel beyond the voxels it was fitted to, so the test is
    whether the overhang is sub-voxel.
    """
    notes = []
    if vessels is not None and grid is not None:
        vb, gb = np.array(vessels.bounds), np.array(grid.bounds)
        lo_over = np.maximum(gb[::2] - vb[::2], 0.0)     # centreline below the mask minimum
        hi_over = np.maximum(vb[1::2] - gb[1::2], 0.0)   # centreline above the mask maximum
        overhang = float(max(lo_over.max(), hi_over.max()))
        inset = float(max((vb[::2] - gb[::2]).max(), (gb[1::2] - vb[1::2]).max()))
        voxel = float(min(PROCESSING_VOXEL_UM))
        verdict = "sub-voxel, from centreline smoothing" if overhang < voxel else "CHECK"
        notes.append(f"overhang beyond mask {overhang:.2f} um "
                     f"({overhang / voxel:.2f} voxel) - {verdict}")
        reach = ("network does not reach that face" if inset > 0
                 else "network reaches every face")
        notes.append(f"inset from mask face {inset:.1f} um - {reach}")
    if skeleton is not None and grid is not None:
        mask = grid["vessel_mask"].reshape(grid.dimensions, order="F").astype(bool)
        vz, vy, vx = PROCESSING_VOXEL_UM
        idx = np.column_stack([
            np.clip((skeleton.points[:, 0] / vz).round().astype(int), 0, mask.shape[0] - 1),
            np.clip((skeleton.points[:, 1] / vy).round().astype(int), 0, mask.shape[1] - 1),
            np.clip((skeleton.points[:, 2] / vx).round().astype(int), 0, mask.shape[2] - 1)])
        inside = mask[idx[:, 0], idx[:, 1], idx[:, 2]].mean()
        notes.append(f"{inside:.1%} of skeleton points fall inside mask foreground")
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="Check that the exported frames overlay, and stop without "
                             "writing any file (as cb_h2_vtk.py --verify does).")
    args = parser.parse_args()

    if not args.verify:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for specimen in SPECIMENS:
        # Opening refuses morphometry cut anywhere but the placed ROI (open item 27), before
        # the edge table is read.
        run = open_batch_run(specimen)
        edges = run.edge_table()
        report = {}
        vessels = enrich_vessels(run, edges, report)
        nodes = build_nodes(run, edges, report)
        skeleton = build_skeleton(run, report)
        surface, grid = build_surface(run, report)

        if not args.verify:
            stem = OUTPUT / specimen.specimen_id
            if vessels is not None:
                vessels.save(f"{stem}_vessels.vtp")
            if nodes is not None:
                nodes.save(f"{stem}_nodes.vtp")
            if skeleton is not None:
                skeleton.save(f"{stem}_skeleton.vtp")
            if surface is not None:
                surface.save(f"{stem}_surface.vtp")
            if grid is not None:
                stamp(grid, specimen).save(f"{stem}_mask.vti")

        matched = report.get("vessels_matched", 0)
        cells = report.get("vessels_cells", 0)
        print(f"{specimen.specimen_id} ({specimen.group}): "
              f"{matched}/{cells} edges joined, {report.get('nodes', 0)} nodes "
              f"({report.get('branch_nodes', 0)} branch), "
              f"{report.get('skeleton_voxels', 0)} skeleton voxels, "
              f"{report.get('surface_points', 0)} surface points")
        if args.verify:
            for note in verify(specimen, vessels, skeleton, grid):
                print(f"    {note}")
        summary[specimen.specimen_id] = report

    if args.verify:
        print("\n  --verify only; nothing written.")
        return
    (OUTPUT / "export_summary.json").write_text(json.dumps(summary, indent=2))
    # The guide is version-controlled beside the code; examples/outputs/ is gitignored, so a
    # copy travels with the data for anyone handed the directory on its own.
    guide = Path(__file__).resolve().parent / "cb_h1_paraview_guide.md"
    if guide.exists():
        (OUTPUT / "README.md").write_text(guide.read_text())
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
