"""What every file a run writes is, column by column and array by array,
with each one's unit.

The statistics CSVs' rows are defined in
:mod:`haemolynx.statistics.glossary` and its page; this covers every other
output, and says which files those rows are in. Units are spelled as the
glossary spells them (:data:`UNITS`: the glossary's, plus the few an output
file needs that a statistic does not). Column names that are code (lower
case, underscores) carry their unit as a suffix -- ``_um``, ``_pa``,
``_m3_s`` (:data:`UNIT_SUFFIXES`) -- and the VTK files carry theirs in their
field data (``<array>_unit``, from :data:`haemolynx.visualization.vtk_io.ARRAY_UNITS`).

``docs/export_dictionary.md`` is :func:`export_dictionary_markdown`
(``python -m haemolynx.pipeline.export_dictionary`` writes it again).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

from haemolynx.statistics.glossary import UNITS as STATISTICS_UNITS
from haemolynx.visualization.vtk_io import ARRAY_UNITS, POINTS_AXIS_ORDER, POINTS_UNIT

#: Every unit an output's column or array is in: the statistics glossary's,
#: and those only an output file needs.
UNITS: dict[str, str] = {
    **STATISTICS_UNITS,
    "-": "no unit: an identifier, a label or text",
    "flag": "0 or 1 (or False and True): no and yes",
    "percent": "a percentage",
    "voxel": "voxel indices (array positions in the image), not microns",
    "bytes": "bytes of memory",
}

#: The suffix a code-style column name (lower case, underscores) ends in for
#: each unit that has one. A column in a unit not listed here (a count, a
#: fraction, a ratio, a flag, a label) needs no suffix.
UNIT_SUFFIXES: dict[str, str] = {
    "um": "_um",
    "um^2": "_um2",
    "um^3": "_um3",
    "Pa": "_pa",
    "m^3/s": "_m3_s",
    "Pa.s/m^3": "_pa_s_per_m3",
    "m^3/(Pa.s)": "_m3_per_pa_s",
    "s": "_s",
    "mm/s": "_mm_s",
    "degrees": "_deg",
    "percent": "_percent",
    "bytes": "_bytes",
    "voxel": "_vox",
}


@dataclass(frozen=True)
class Column:
    """One column of a CSV, array of a VTK file, or key of a JSON file."""

    name: str
    unit: str
    meaning: str


@dataclass(frozen=True)
class OutputFile:
    """One file a run writes. *name* is its file name, ``{stem}`` standing for
    the run's stem (``pipeline.naming.run_stem``) and ``{name}`` for a
    perturbation's name; *where* is the folder it is written to."""

    name: str
    where: str
    what: str
    columns: tuple[Column, ...] = ()
    columns_note: str = ""


def _c(name: str, unit: str, meaning: str) -> Column:
    return Column(name, unit, meaning)


# -- shared column sets -------------------------------------------------------

_EDGE_ID = (
    _c("u", "-", "Node ID at one end of the vessel (graph node, as in the nodes file's node_id)."),
    _c("v", "-", "Node ID at the other end."),
    _c("key", "-", "Which of the vessels between u and v (0 for the first)."),
)
_BRANCH_ORDER = _c(
    "branch_order", "-",
    "The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO.",
)
_SWEEP_RESULTS = (
    _c("inlet_pressure_pa", "Pa", "Pressure held at the inlets for this point."),
    _c("outlet_pressure_pa", "Pa", "Pressure held at the outlets."),
    _c("total_inlet_flow_m3_s", "m^3/s", "Total flow into the network through its inlets."),
    _c("total_outlet_flow_m3_s", "m^3/s", "Total flow through the outlets (negative: leaving)."),
    _c("flow_balance_error_m3_s", "m^3/s", "Inlet plus outlet flow: 0 when flow is conserved."),
    _c("equivalent_resistance_pa_s_per_m3", "Pa.s/m^3",
       "The network's resistance: inlet minus outlet pressure over total inflow."),
)
_DILATION = (
    _c("dilation_percent", "percent", "Diameter change applied at this point (+10 widens by 10%)."),
    _c("dilation_factor", "dimensionless", "The same change as a scale on the diameter (1.1 for +10%)."),
)
_REVIEW_MEASURES = (
    _c("connector_um", "um", "Close junctions: the length of the vessel joining the two junctions."),
    _c("widest_radius_um", "um", "Close junctions: the widest vessel's radius there."),
    _c("vessels", "vessels", "How many vessels the item involves or meets."),
    _c("length_um", "um", "The vessel's (or the chain's, or the piece's) length."),
    _c("chord_um", "um", "Hairpins: the straight distance between the vessel's ends."),
    _c("ratio", "dimensionless", "Hairpins: length over chord. Diameter jumps: wide over narrow."),
    _c("narrow_um", "um", "Diameter jumps: the narrower diameter."),
    _c("wide_um", "um", "Diameter jumps: the wider diameter."),
    _c("at", "-", "Diameter jumps: pass-through (one vessel into the next) or junction."),
    _c("diameter_um", "um", "The vessel's diameter (the widest vessel's, at a node)."),
    _c("diameter_source", "-", "Where that diameter came from: fwhm, edt, class_median, table, ..."),
    _c("edt_diameter_um", "um", "The mask's own width of the vessel, where measured."),
    _c("fwhm_diameter_um", "um", "The FWHM width of the vessel, where measured."),
    _c("flow_share", "fraction", "The vessel's flow over the network's total inflow."),
    _c("reasons", "-", "Unmeasured diameters: why the width is in doubt."),
    _c("attached_at", "-", "Unsolved pieces: the node IDs it hangs from (;-separated)."),
    _c("disconnected", "flag", "Unsolved pieces: True when it touches nothing that is solved."),
    _c("orders", "-", "Flow against branch orders: the vessels' orders (;-separated)."),
    _c("arteriole_meets_venule", "flag", "Flow against branch orders: an arteriole runs straight into a venule."),
    _c("velocity_mm_s", "mm/s", "Flow outliers: the vessel's mean velocity."),
    _c("class", "-", "Flow outliers: the vessel type it is compared within."),
    _c("class_median_mm_s", "mm/s", "Flow outliers: the median velocity of that type (log scale)."),
    _c("robust_z", "dimensionless", "Flow outliers: how many robust SDs (1.4826 MAD of log10 velocity) from it."),
    _c("role", "-", "Terminal nodes and boundary issues: inlet, outlet, arteriole/venule boundary, open end."),
    _c("other_roles", "-", "Other roles the same node has (;-separated)."),
    _c("leaves_image", "flag", "Terminal nodes: the dead end leaves the image."),
    _c("dead_end_kind", "-", "The node's dead-end kind (graph.dead_ends)."),
    _c("depth_um", "um", "How far inside the image the node is (distance to the nearest face)."),
    _c("issue", "-", "Boundary issues: what is wrong with the node."),
    _c("kinds", "-", "Dead ends: every test the end met (;-separated)."),
    _c("tip_radius_um", "um", "Dead ends: the lumen radius at the tip."),
    _c("junction_radius_um", "um", "Dead ends: the lumen radius where its chain meets a junction."),
    _c("mask_run_um", "um", "Dead ends: how far the segmentation runs on past the tip."),
    _c("ahead_um", "um", "Dead ends: the distance to the vessel straight ahead."),
    _c("ahead_junctions_away", "count", "Dead ends: how many junctions away that vessel is in the graph."),
    _c("face_distance_um", "um", "Dead ends: the tip's distance to the nearest image face."),
    _c("in_mask_fraction", "fraction", "Dead ends: the share of the chain inside the segmentation."),
    _c("ahead_edge", "-", "Dead ends: the vessel straight ahead, as (u, v, key)."),
    _c("ahead_point_um", "um", "Dead ends: the point on it the join would reach, (z, y, x)."),
    _c("shorter_um", "um", "Two in one lumen: the shorter vessel's length."),
    _c("lengths_um", "um", "Two in one lumen: both lengths (;-separated)."),
    _c("cuts_off", "-", "Two in one lumen: for each, whether deleting it cuts vessels off (;-separated)."),
    _c("gap_um", "um", "Facing ends: the gap between the two tips."),
    _c("turn_deg", "degrees", "Facing ends: the larger turn from either end's heading onto the join."),
    _c("refused", "-", "Facing ends: why the build did not join them."),
    _c("volume_um3", "um^3", "Uncovered mask: the region's volume."),
    _c("radius_um", "um", "Uncovered mask: its widest inscribed radius."),
    _c("voxels", "count", "Uncovered mask: its voxel count."),
)


# -- the files ------------------------------------------------------------------

_RUN = "the run's output folder (vtk_output_prefix's)"
_PLOTS = "the plot folder (plot_dir)"
_PERTURBATION = "the perturbation's folder: perturbation_output_dir/{name}_{type}"
_BATCH = "the batch's output folder"

#: The run's own folder and a perturbation's, as :attr:`OutputFile.where`
#: spells them (a batch checks how long its paths there will be).
RUN_FOLDER = _RUN
PERTURBATION_FOLDER = _PERTURBATION

_VTK_POINTS = (
    f"Points are physical ({POINTS_AXIS_ORDER}) microns, as node positions are; "
    f"field data ``points_unit`` ({POINTS_UNIT}) and ``points_axis_order`` say so, "
    "and ``<array>_unit`` gives each array's unit."
)


def _vtk(names: tuple[str, ...], meanings: dict[str, str]) -> tuple[Column, ...]:
    return tuple(_c(name, ARRAY_UNITS[name], meanings[name]) for name in names)


VESSEL_ARRAYS: dict[str, str] = {
    "edge_u": "Node ID at one end (the nodes file's node_id).",
    "edge_v": "Node ID at the other end.",
    "edge_key": "Which of the vessels between edge_u and edge_v (0 for the first).",
    "branch_order": _BRANCH_ORDER.meaning,
    "resistance": "The vessel's hydraulic resistance (Poiseuille, with the run's viscosity law).",
    "conductance": "1 / resistance.",
    "length_um": "The vessel's length along its (smoothed) centreline.",
    "diameter_um": "The vessel's diameter, as the haemodynamics used it.",
    "diameter_source": "Where the diameter came from: fwhm, raw_section, endothelial, edt, class_median, table, bridge, synthetic, ...",
    "is_zero_resistance": "1 for a thick-vessel bridge (a join drawn inside a lumen, given negligible resistance).",
    "flow_solved": "1 where the solve gave the vessel a flow; 0 where it did not (see flow_state).",
    "flow_state": "solved, blind_end (behind a dead end inside the tissue: flow exactly 0), closed_end (behind a face end with no condition: flow exactly 0) or no_boundary (no inlet and outlet reach it: flows blank).",
    "discharge_haematocrit": "The red-cell volume fraction of the blood the vessel carries (with a distributed haematocrit model only).",
    "pressure_u": "Pressure at the edge_u end.",
    "pressure_v": "Pressure at the edge_v end.",
    "pressure_drop": "pressure_u - pressure_v.",
    "flow_signed": "Flow from edge_u to edge_v (negative: the other way).",
    "flow_abs": "The size of the flow, whichever way.",
}
NODE_ARRAYS: dict[str, str] = {
    "node_id": "The graph's node ID (the IDs boundary settings and the panel use).",
    "dead_end_kind": "For a dead end: leaves_image, island_end, artefact, stops_short, broken, blind or unknown; empty for every other node.",
    "pressure": "The node's pressure; blank where no inlet and outlet reach it.",
}
PERICYTE_ARRAYS: dict[str, str] = {
    "edge_u": "Node ID at one end of the vessel the constriction site is on.",
    "edge_v": "Node ID at its other end.",
    "edge_key": "Which of the vessels between them.",
    "branch_order": "That vessel's class.",
}

FILES: tuple[OutputFile, ...] = (
    # -- the model ---------------------------------------------------------
    OutputFile(
        "{stem}_vessels.vtp", _RUN,
        "Every vessel as a polyline (VTK PolyData), one cell per vessel. " + _VTK_POINTS
        + " Field data also holds haemolynx_version, viscosity_law, diameter_basis, "
        "haematocrit and open_end_model. Flow arrays are written once the network is solved.",
        _vtk(tuple(VESSEL_ARRAYS), VESSEL_ARRAYS),
    ),
    OutputFile(
        "{stem}_nodes.vtp", _RUN,
        "Every node as a point: junctions, dead ends, inlets and outlets. " + _VTK_POINTS,
        _vtk(tuple(NODE_ARRAYS), NODE_ARRAYS),
    ),
    OutputFile(
        "{stem}_pericytes.vtp", _RUN,
        "Constriction sites placed along every vessel at the run's constriction spacing "
        "and length (constriction_spacing_um, constriction_length_um). " + _VTK_POINTS,
        _vtk(tuple(PERICYTE_ARRAYS), PERICYTE_ARRAYS),
    ),
    OutputFile(
        "{stem}_tissue_surface.vtp", _RUN,
        "The tissue's surface as a triangle mesh (with measure_tissue_volume). " + _VTK_POINTS
        + " It has no arrays.",
    ),
    OutputFile(
        "{stem}_graph.pkl", _RUN,
        "The network as a pickled networkx MultiGraph: node pos and edge voxels in "
        "physical (z, y, x) microns, every attribute the run set (CLAUDE.md, Graph). "
        "Read it with the same HaemoLynx version.",
    ),
    OutputFile(
        "{stem}_skeleton.npy", _RUN,
        "The skeleton as a NumPy array, voxel-indexed (z, y, x), 1 on the centreline.",
    ),
    OutputFile(
        "{stem}_voxel_size.json", _RUN,
        "The voxel size the run read and used, and where it came from.",
        (
            _c("voxel_size", "um", "The image's voxel size as read, (x, y, z)."),
            _c("run_voxel_size", "um", "The voxel size the run used, (x, y, z) (after any downsampling)."),
            _c("voxel_size_source", "-", "Where it came from: OME header, TIFF tags, override, ..."),
            _c("voxel_metadata_status", "-", "Whether the metadata was found, and its source."),
            _c("voxel_size_policy", "-", "The voxel_size_policy setting."),
            _c("voxel_size_override_xyz", "um", "The voxel_size_override_xyz setting, if any."),
            _c("downsample", "-", "The downsample-on-import record, or null."),
        ),
    ),
    # -- statistics ---------------------------------------------------------
    OutputFile(
        "{stem}_statistics.csv", _RUN,
        "The statistics report, one row per metric (with the Statistics toggle).",
        (
            _c("Section", "-", "The report section the metric is in."),
            _c("Metric", "-", "The metric's name."),
            _c("Value", "-", "Its value (or N/A and why)."),
            _c("Unit", "-", "Its unit."),
            _c("Notes", "-", "Its definition."),
        ),
        "Each row's definition and unit: docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_statistics_connected.csv", _RUN,
        "The geometry and topology statistics again, over the inlet/outlet-connected network only.",
        columns_note="Columns as {stem}_statistics.csv; rows in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_branch_statistics.csv", _RUN,
        "Statistics per branch order, one row per order.",
        columns_note="Every column is defined, with its unit, in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_capillary_bed_joins.csv", _RUN,
        "One row per capillary bed: the B order at which it joins another bed (statistics_capillary_bed_joins).",
        columns_note="Every column is defined, with its unit, in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_tissue_volume.csv", _RUN,
        "The tissue's volume, surface and the vessel densities it gives (measure_tissue_volume).",
        columns_note="Columns as {stem}_statistics.csv; rows in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_betweenness_communities_resistance.json", _RUN,
        "Weighted betweenness and vascular communities, edges weighted by resistance: "
        "the Weighted Betweenness and Communities section of the report, as JSON.",
        columns_note="Keys as the report's rows; docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_betweenness_communities_edge_length.json", _RUN,
        "The same, edges weighted by length.",
        columns_note="Keys as the report's rows; docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_betweenness_communities_edge_flow.json", _RUN,
        "The same, edges weighted by flow.",
        columns_note="Keys as the report's rows; docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{stem}_cell_to_vessel_3d_dual_distances.csv", _RUN,
        "Each object of the cell mask and its distance to the vessels (measurement_3d_to_cell_mask).",
        (
            _c("Object ID", "-", "The object's label in the cell mask."),
            _c("Object Name", "-", "Its name."),
            _c("Voxel Count", "count", "Its size in voxels."),
            _c("Centroid Z (voxel)", "voxel", "Its centroid, z index."),
            _c("Centroid Y (voxel)", "voxel", "Its centroid, y index."),
            _c("Centroid X (voxel)", "voxel", "Its centroid, x index."),
            _c("Centroid Z (um)", "um", "Its centroid, z."),
            _c("Centroid Y (um)", "um", "Its centroid, y."),
            _c("Centroid X (um)", "um", "Its centroid, x."),
            _c("Vessel Contact", "-", "separate, touching, overlapping or inside."),
            _c("Vessel Overlap Voxel Count", "count", "Its voxels inside a vessel."),
            _c("Edge-to-Vessel Distance (um)", "um", "Gap between the object's surface and the nearest vessel (0 touching)."),
            _c("Edge-to-Vessel Nearest Z (um)", "um", "The nearest vessel point, z."),
            _c("Edge-to-Vessel Nearest Y (um)", "um", "The nearest vessel point, y."),
            _c("Edge-to-Vessel Nearest X (um)", "um", "The nearest vessel point, x."),
            _c("Edge-to-Vessel Nearest Coordinate (z,y,x um)", "um", "The same point as one (z, y, x)."),
            _c("Edge-to-Vessel Nearest Graph Edge Label ID", "-", "The vessel whose lumen that point is in."),
            _c("Edge-to-Vessel Nearest Branch Order", "-", "That vessel's class."),
            _c("Centroid-to-Vessel Distance (um)", "um", "Distance from the object's centroid to the nearest vessel."),
            _c("Delta Distance (Centroid-Edge) (um)", "um", "Centroid distance minus surface distance."),
            _c("Centroid-to-Vessel Nearest Z (um)", "um", "The vessel point nearest the centroid, z."),
            _c("Centroid-to-Vessel Nearest Y (um)", "um", "The same, y."),
            _c("Centroid-to-Vessel Nearest X (um)", "um", "The same, x."),
            _c("Centroid-to-Vessel Nearest Coordinate (z,y,x um)", "um", "The same point as one (z, y, x)."),
            _c("Centroid-to-Vessel Nearest Graph Edge Label ID", "-", "The vessel whose lumen that point is in."),
            _c("Centroid-to-Vessel Nearest Branch Order", "-", "That vessel's class."),
        ),
    ),
    OutputFile(
        "{stem}_cell_to_vessel_3d_dual_distances_summary.csv", _RUN,
        "Those distances summarised overall and per branch order of the nearest vessel.",
        (
            _c("Group", "-", "Overall, or a branch order."),
            _c("Measurement Type", "-", "edge_to_vessel or centroid_to_vessel."),
            _c("Object Count", "count", "Objects in the group."),
            _c("Mean Distance (um)", "um", "Mean distance."),
            _c("Median Distance (um)", "um", "Median distance."),
            _c("Std Distance (um)", "um", "Standard deviation of the distance."),
            _c("Min Distance (um)", "um", "Smallest distance."),
            _c("Max Distance (um)", "um", "Largest distance."),
            _c("Mean Delta (Centroid-Edge) (um)", "um", "Mean of centroid minus surface distance."),
            _c("Median Delta (Centroid-Edge) (um)", "um", "Median of the same."),
            _c("Std Delta (Centroid-Edge) (um)", "um", "Its standard deviation."),
        ),
    ),
    OutputFile(
        "{stem}_template_comparison.csv", _RUN,
        "A synthetic run only: each characteristic of the template against the synthetic network.",
        (
            _c("Section", "-", "box, endpoints, counts, lengths, diameters, ..."),
            _c("Characteristic", "-", "What is compared; its unit is in its name, e.g. extent z (um)."),
            _c("Template", "-", "The template's value, in the characteristic's unit."),
            _c("Synthetic", "-", "The synthetic network's value, in the same unit."),
            _c("Ratio", "dimensionless", "Synthetic over template."),
            _c("Tolerance", "-", "The band the generator is held to (exact, rel 0.02, ...), or empty."),
            _c("Within", "-", "yes or no; empty where nothing is fitted."),
        ),
    ),
    # -- the run's record ----------------------------------------------------
    OutputFile(
        "{stem}_run_manifest.json", _RUN,
        "What produced the outputs: the settings as the run started, the HaemoLynx version, "
        "git commit, Python, platform and dependency versions, each input file's size, time and "
        "sha256, and each stage's start, end, wall time (seconds) and peak memory (bytes).",
        (
            _c("stages[].wall_seconds", "s", "The stage's wall time."),
            _c("stages[].peak_rss_bytes", "bytes", "The stage's peak memory."),
            _c("inputs[].size_bytes", "bytes", "An input file's size."),
        ),
        "Every other key is text or a nested record; see pipeline/provenance.py.",
    ),
    OutputFile(
        "{stem}_run_settings.yaml", _RUN,
        "The settings the run started with, as a config file --config loads. Each setting's "
        "unit is in its help (the comments of a config write_default_config writes).",
    ),
    # -- perturbations -------------------------------------------------------
    OutputFile(
        "{name}_summary.csv", _PERTURBATION,
        "One row: the perturbation's network-level result against the baseline.",
        (
            _c("name", "-", "The perturbation's name."),
            _c("type", "-", "Its type (perturbation_type)."),
            _c("overrides", "-", "The settings it changed, as JSON."),
            *_SWEEP_RESULTS[:5],
            _c("equivalent_resistance_pa_s_per_m3", "Pa.s/m^3",
               "The perturbed network's resistance: pressure drop over total inflow."),
            _c("baseline_equivalent_resistance_pa_s_per_m3", "Pa.s/m^3", "The baseline network's, the same way."),
            _c("delta_vs_baseline_pa_s_per_m3", "Pa.s/m^3", "Perturbed minus baseline."),
            _c("percent_change_vs_baseline", "percent", "That change over the baseline."),
        ),
    ),
    OutputFile(
        "{name}_edges.csv", _PERTURBATION,
        "One row per vessel of the perturbed network.",
        (
            *_EDGE_ID,
            _BRANCH_ORDER,
            _c("length_um", "um", "The vessel's length."),
            _c("diameter_um", "um", "Its diameter after the perturbation."),
            _c("resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance after the perturbation."),
            _c("conductance_m3_per_pa_s", "m^3/(Pa.s)", "1 / resistance."),
            _c("pressure_drop_pa", "Pa", "Pressure at u minus pressure at v; blank where no inlet and outlet reach it."),
            _c("flow_signed_m3_s", "m^3/s", "Flow from u to v (negative: the other way); blank as above, exactly 0 behind a dead end."),
            _c("flow_solved", "flag", "True where the solve gave it a flow."),
            _c("flow_state", "-", "solved, blind_end, closed_end or no_boundary (as in the vessels file)."),
            _c("baseline_resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance in the baseline network."),
            _c("resistance_ratio", "dimensionless", "Perturbed over baseline resistance."),
        ),
    ),
    OutputFile(
        "{name}_statistics.csv, {name}_branch_statistics.csv, {name}_capillary_bed_joins.csv, "
        "{name}_betweenness_communities_*.json", _PERTURBATION,
        "The perturbed network's statistics, as the run's own files of those names.",
        columns_note="Columns as the run's; rows in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "{name}_blocked_vessels.csv", _PERTURBATION,
        "A capillary block: one row per vessel it blocked.",
        (
            _c("branch_id", "-", "The vessel's branch ID (its place in the graph's edge order, as the panel shows it)."),
            *_EDGE_ID,
            _BRANCH_ORDER,
            _c("length_um", "um", "The vessel's length."),
            _c("diameter_um", "um", "Its diameter."),
            _c("baseline_resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance before the block."),
            _c("blocked_resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance blocked."),
            _c("baseline_flow_abs_m3_s", "m^3/s", "Its flow before the block."),
            _c("blocked_flow_abs_m3_s", "m^3/s", "Its flow blocked (exactly 0)."),
        ),
    ),
    OutputFile(
        "{name}_block_comparison.csv", _PERTURBATION,
        "A capillary block: how the blocked network's flow differs from the baseline's, one "
        "metric a row (metric, value). Each metric's unit is in its name; the rows are listed "
        "below as columns.",
        (
            _c("metric", "-", "The metric's name."),
            _c("value", "-", "Its value, in the unit its name carries."),
            _c("selection", "-", "How the vessels were chosen."),
            _c("branch_orders", "-", "The orders drawn from (JSON list)."),
            _c("probability", "fraction", "The chance each candidate vessel was blocked."),
            _c("seed", "-", "The random seed."),
            _c("candidate_vessels", "vessels", "Vessels that could be blocked."),
            _c("vessel_ids", "-", "The branch IDs asked for (JSON list)."),
            _c("capillary_block_mask_path", "-", "The mask of blocks, when read from a mask."),
            _c("mask_blocks", "count", "Blobs in that mask."),
            _c("mask_blocks_on_vessels", "count", "Blobs that blocked a vessel."),
            _c("mask_blocks_below_overlap", "count", "Blobs too little inside the segmentation."),
            _c("mask_blocks_off_graph", "count", "Blobs near no vessel."),
            _c("min_overlap_fraction", "fraction", "capillary_block_mask_min_overlap_fraction."),
            _c("vessels_per_block", "-", "capillary_block_mask_vessels_per_block."),
            _c("vessel_mask_source", "-", "What the blobs were read against: the segmentation or the graph's lumens."),
            _c("blocked_vessels", "vessels", "Vessels blocked."),
            _c("resistance_factor", "dimensionless", "What a blocked vessel's resistance was multiplied by."),
            _c("total_inflow_baseline_m3_s", "m^3/s", "Total inflow before the block."),
            _c("total_inflow_blocked_m3_s", "m^3/s", "Total inflow blocked."),
            _c("total_inflow_percent_change", "percent", "Its change."),
            _c("blocked_vessel_length_um", "um", "Length of the blocked vessels."),
            _c("stopped_vessels", "vessels", "Blocked vessels and every vessel fed only through one."),
            _c("stopped_vessel_length_um", "um", "Their length."),
            _c("hypoperfusion_threshold_fraction", "fraction", "The loss of flow that counts as hypoperfused."),
            _c("hypoperfused_vessels", "vessels", "Other vessels that lost at least that share of their flow."),
            _c("hypoperfused_vessel_fraction", "fraction", "Their share of the vessels carrying flow."),
            _c("hypoperfused_length_um", "um", "Their length."),
            _c("hypoperfused_length_fraction", "fraction", "Their share of the network's length."),
            _c("reversed_vessels", "vessels", "Vessels whose flow now runs the other way."),
            _c("outlets", "count", "Outlets compared."),
            _c("outlet_flow_retained_mean", "fraction", "Mean over outlets of blocked over baseline outflow."),
            _c("outlet_flow_retained_min", "fraction", "The least of the same."),
            _c("outlets_losing_half_their_flow", "count", "Outlets keeping half their flow or less."),
            _c("mean_transit_time_s_{baseline|blocked}", "s", "Mean transit time, before / blocked."),
            _c("transit_time_sd_s_{baseline|blocked}", "s", "Its SD."),
            _c("capillary_mean_transit_time_s_{baseline|blocked}", "s", "Capillary mean transit time."),
            _c("capillary_transit_time_heterogeneity_s_{baseline|blocked}", "s", "Capillary transit time heterogeneity (CTH)."),
            _c("transit_time_status_{baseline|blocked}", "-", "Why transit times could not be computed, when not."),
            _c("{capillary|vessel}_{flow|velocity|red_cell_flux|red_cell_velocity}_cv_{baseline|blocked}",
               "dimensionless", "Coefficient of variation over the vessels carrying flow."),
            _c("{vessels|capillaries}_carrying_flow_{baseline|blocked}", "vessels", "Vessels carrying flow."),
            _c("flowing_{vessel|capillary}_length_um_{baseline|blocked}", "um", "Their length."),
            _c("flowing_{vessel|capillary}_volume_um3_{baseline|blocked}", "um^3", "Their volume."),
            _c("flux_heterogeneity_status_{baseline|blocked}", "-", "Why flux heterogeneity could not be computed, when not."),
            _c("draws", "count", "With replicates: how many draws."),
            _c("<metric>_draws_{mean|sd|low|high}", "-",
               "With replicates: each metric's mean, SD and 2.5-97.5% range over the draws, in its own unit."),
        ),
    ),
    OutputFile(
        "{name}_block_replicates.csv", _PERTURBATION,
        "A capillary block with replicates: one row per draw.",
        (
            _c("draw", "-", "The draw's number (1 is the first)."),
            _c("seed", "-", "Its seed."),
            _c("blocked_vessels", "vessels", "Vessels it blocked."),
            _c("<metric>", "-", "Then one column per numeric metric of {name}_block_comparison.csv, in its unit."),
        ),
    ),
    OutputFile(
        "{name}_flow_by_branch_order.csv", _PERTURBATION,
        "A capillary block: the comparison per branch order.",
        (
            _BRANCH_ORDER,
            _c("vessels", "vessels", "Vessels of that order."),
            _c("blocked", "vessels", "Of them, blocked."),
            _c("baseline_mean_flow_m3_s", "m^3/s", "Mean flow before the block."),
            _c("blocked_mean_flow_m3_s", "m^3/s", "Mean flow blocked."),
            _c("mean_flow_percent_change", "percent", "Its change."),
            _c("hypoperfused", "vessels", "Vessels hypoperfused."),
            _c("hypoperfused_length_um", "um", "Their length."),
            _c("reversed", "vessels", "Vessels reversed."),
        ),
    ),
    OutputFile(
        "{name}_mask_blocks.csv", _PERTURBATION,
        "A capillary block read from a mask: one row per blob of the mask.",
        (
            _c("block", "-", "The blob's number."),
            _c("voxels", "count", "Its voxel count."),
            _c("volume_um3", "um^3", "Its volume."),
            _c("centroid_z_um", "um", "Its centroid, z."),
            _c("centroid_y_um", "um", "Its centroid, y."),
            _c("centroid_x_um", "um", "Its centroid, x."),
            _c("voxels_in_vessels", "count", "Its voxels inside the segmentation."),
            _c("overlap_fraction", "fraction", "voxels_in_vessels over voxels."),
            _c("blocks", "flag", "True when it blocks a vessel."),
            _c("blocked_branch_ids", "-", "The branch IDs it blocks (JSON list)."),
            _c("vessel_voxels", "-", "Its voxels in each vessel (JSON)."),
            _c("reason", "-", "Why it blocks nothing, when it does not."),
        ),
    ),
    OutputFile(
        "{arteriole|capillary|pericyte}_dilation_sweep.csv, ..._dilation_pressure_sweep.csv, inlet_pressure_sweep.csv",
        _PERTURBATION,
        "A dilation and/or inlet pressure sweep: one row per grid point.",
        (*_DILATION, *_SWEEP_RESULTS),
    ),
    OutputFile(
        "pericyte_spacing_sweep.csv, pericyte_length_sweep.csv", _PERTURBATION,
        "A pericyte geometry sweep: one row per spacing or length.",
        (
            _c("constriction_spacing_um", "um", "Distance between constriction sites."),
            _c("constriction_length_um", "um", "Length of each constriction."),
            *_DILATION,
            *_SWEEP_RESULTS,
        ),
    ),
    # -- post processing -------------------------------------------------------
    OutputFile(
        "{stem}_connectivity.csv", _RUN,
        "Post processing's connectivity table: one row per vessel and what it joins.",
        (
            _c("Branch ID", "-", "The vessel's branch ID."),
            _c("From node ID", "-", "Its upstream node (by flow, where solved)."),
            _c("To node ID", "-", "Its downstream node."),
            _c("Branch order", "-", "Its class."),
            _c("Length (um)", "um", "Its length."),
            _c("Diameter (um)", "um", "Its diameter."),
            _c("Notes", "-", "Bridge, unsolved, inlet/outlet and so on."),
            _c("Upstream branch IDs", "-", "Vessels feeding it."),
            _c("Downstream branch IDs", "-", "Vessels it feeds."),
            _c("Edge u", "-", "Graph node u."),
            _c("Edge v", "-", "Graph node v."),
            _c("Edge key", "-", "Graph edge key."),
        ),
    ),
    OutputFile(
        "{stem}_loop_review.csv", _RUN,
        "Post processing's short-loop decisions, one row each, with the loop's measures.",
        (
            _c("time", "-", "When it was decided (ISO 8601)."),
            _c("decision", "-", "kept, or side deleted."),
            _c("side_deleted", "-", "The side deleted, if any."),
            _c("wall_side", "-", "The side the build's loop breaker would have taken out."),
            _c("loop_length_um", "um", "The loop's length round."),
            _c("sides", "count", "Its sides."),
            _c("loop_branch_ids", "-", "Its vessels' branch IDs."),
            _c("loop_edges", "-", "Its vessels as (u, v, key)."),
            _c("centre_z_um", "um", "Its centre, z."),
            _c("centre_y_um", "um", "Its centre, y."),
            _c("centre_x_um", "um", "Its centre, x."),
            _c("centre_zyx_vox", "voxel", "Its centre as voxel indices (z y x), to find in napari."),
            _c("span_um", "um", "Its widest extent."),
            _c("area_um2", "um^2", "The area it encloses, in its best-fitting plane."),
            _c("out_of_plane", "dimensionless", "Third over second singular value: 0 for a flat loop."),
            _c("path_in", "fraction", "Share of the loop inside the segmentation."),
            _c("lumen_r_um", "um", "Median lumen radius along it."),
            _c("chords_in_lumen", "fraction", "Share of its chords crossing no more than a dropout of background."),
            _c("chord_bg_median_um", "um", "Their median background: the loop's hole."),
            _c("inside_one_lumen", "flag", "Whether it lies inside one lumen."),
            _c("side_lengths_um", "um", "Each side's length (;-separated)."),
            _c("side_diameters_um", "um", "Each side's diameter (;-separated)."),
            _c("side_lumen_r_um", "um", "Each side's mean lumen radius (;-separated)."),
            _c("zero_resistance_bridges", "count", "Thick-vessel bridges in the loop."),
            _c("recovered_um", "um", "Length of it mask recovery drew."),
            _c("bridged_um", "um", "Length of it that is bridges."),
        ),
    ),
    OutputFile(
        "{stem}_{kind}_review.csv", _RUN,
        "Post processing's decisions on a review list (dead ends, hairpins, ...), one row each, "
        "with the item's measures after the first ten columns.",
        (
            _c("time", "-", "When it was decided (ISO 8601)."),
            _c("decision", "-", "kept, or what was done (deleted, merged, ...)."),
            _c("kind", "-", "The review list (and, for a dead end, its kind)."),
            _c("centre_z_um", "um", "The item's centre, z."),
            _c("centre_y_um", "um", "The item's centre, y."),
            _c("centre_x_um", "um", "The item's centre, x."),
            _c("centre_zyx_vox", "voxel", "Its centre as voxel indices (z y x)."),
            _c("nodes", "-", "Its node IDs (;-separated)."),
            _c("branch_ids", "-", "Its vessels' branch IDs (;-separated)."),
            _c("why", "-", "Why it was listed."),
            *_REVIEW_MEASURES,
        ),
        "Only the measures of the item's own list are filled.",
    ),
    # -- batch -------------------------------------------------------------------
    OutputFile(
        "batch_summary.csv", _BATCH,
        "One row per image of a batch run.",
        (
            _c("stem", "-", "The image's stem."),
            _c("status", "-", "done, failed, skipped (done before), cancelled or not run."),
            _c("condition", "-", "Its condition (A or B), if any."),
            _c("segmented", "-", "Its segmented image."),
            _c("raw", "-", "Its raw image."),
            _c("large_arteriole_mask", "-", "Its large arteriole mask."),
            _c("large_venule_mask", "-", "Its large venule mask."),
            _c("small_arteriole_mask", "-", "Its small arteriole mask."),
            _c("small_venule_mask", "-", "Its small venule mask."),
            _c("haemorun", "-", "The .haemorun it saved."),
            _c("nodes", "count", "Nodes in its network."),
            _c("vessels", "vessels", "Vessels in its network."),
            _c("wall_time_s", "s", "Its wall time."),
            _c(
                "peak_rss_bytes", "bytes",
                "Its peak memory: the largest of its stages' peaks in its run manifest "
                "(not the process's, which in a batch includes every earlier image).",
            ),
            _c("error", "-", "The error, if it failed."),
        ),
    ),
    OutputFile(
        "batch_settings.yaml", _BATCH,
        "The settings the batch ran with, and its folders and conditions as comments.",
    ),
    OutputFile(
        "batch_statistics_by_image.csv", _BATCH,
        "Every image's statistics side by side: Section, Metric, Unit, then one column per "
        "image (its value, in the row's unit). The first rows give each image's condition "
        "(and subject, when paired).",
        columns_note="Metrics as in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "batch_statistics_summary.csv", _BATCH,
        "Per metric and group (each condition, All images): N, Mean, SD, SEM, Median, Q1, Q3, "
        "IQR, Min, Max, Range (in the row's Unit) and CV (dimensionless).",
        columns_note="Metrics as in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "batch_condition_comparison.csv", _BATCH,
        "Two conditions tested per metric: each condition's N, median, mean and range (in the "
        "row's Unit), the difference (Unit) and ratio of medians, Mann-Whitney U or Wilcoxon "
        "(paired) with P, Benjamini-Hochberg Q and the smallest possible P, rank-biserial r, "
        "Welch's or the paired t with P, and Hedges g or Cohen dz (all dimensionless).",
        columns_note="Metrics as in docs/statistics_glossary.md.",
    ),
    OutputFile(
        "batch_condition_pairs.csv", _BATCH,
        "A paired comparison: each subject's pair of images (or the median/mean of its images), "
        "and every image left out with why.",
        (
            _c("Subject", "-", "The subject the images belong to."),
            _c("{A} images", "-", "Condition A's image(s) in the pair."),
            _c("{B} images", "-", "Condition B's."),
            _c("{A} value", "-", "What A's value is: the one image, or the median/mean of N images."),
            _c("{B} value", "-", "The same for B."),
            _c("Left out because", "-", "Why an image was not paired."),
        ),
    ),
    # -- figures -------------------------------------------------------------
    OutputFile(
        "*.png, *.html", _PLOTS,
        "Figures: the skeleton, the graph (final_graph.png, final_graph_3d.html, "
        "geometry_with_branch_orders.png, ...), the node degree distribution, the graph after "
        "each building step (graph_after_<step>.png, with step artifacts on) and a perturbation's "
        "plots and sweep curves in its folder. Axes carry their units.",
    ),
)


# -- the page --------------------------------------------------------------------

def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def export_dictionary_markdown() -> str:
    """The data dictionary as a Markdown page."""
    lines = [
        "# Output data dictionary",
        "",
        "<!-- Written by `python -m haemolynx.pipeline.export_dictionary` from",
        "     src/haemolynx/pipeline/export_dictionary.py; edit that, not this. -->",
        "",
        "Every file a run writes, and every column, array or key in it, with its unit. "
        "`{stem}` is the run's stem (the segmented image's name, or `{template}_synthetic`), "
        "`{name}` a perturbation's name. The rows of the statistics CSVs are defined in "
        "[the statistics glossary](statistics_glossary.md).",
        "",
        "Units are spelled one way everywhere. A column name in code style (lower case, "
        "underscores) carries its unit as a suffix: "
        + ", ".join(f"`{suffix}` ({unit})" for unit, suffix in UNIT_SUFFIXES.items())
        + ". A name without one is a count, a fraction, a ratio, a flag or a label. "
        "The VTK files keep their array names and carry each array's unit in their field "
        "data as `<array>_unit`.",
        "",
        "## Units",
        "",
        "| Unit | Meaning |",
        "|---|---|",
        *(f"| `{_cell(unit)}` | {_cell(meaning)} |" for unit, meaning in UNITS.items()),
    ]
    for item in FILES:
        lines += ["", f"## `{_cell(item.name)}`", "", f"Written to {item.where}. {item.what}"]
        if item.columns_note:
            lines += ["", item.columns_note]
        if item.columns:
            lines += ["", "| Column | Unit | Meaning |", "|---|---|---|"]
            lines += [
                f"| `{_cell(c.name)}` | `{_cell(c.unit)}` | {_cell(c.meaning)} |"
                for c in item.columns
            ]
    return "\n".join(lines) + "\n"


EXPORT_DICTIONARY_PAGE = Path("docs") / "export_dictionary.md"


def write_export_dictionary_markdown(path: Union[str, Path]) -> Path:
    """Write :func:`export_dictionary_markdown` to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_dictionary_markdown(), encoding="utf-8", newline="\n")
    return path


def output_file(name: str) -> OutputFile:
    """The entry for the file named *name* (as :data:`FILES` spells it)."""
    for item in FILES:
        if item.name == name:
            return item
    raise KeyError(name)


def _name_pattern(name: str) -> "re.Pattern[str]":
    """*name* as a regular expression: ``{a|b}`` is either, any other
    ``{...}`` or ``<...>`` stands for any text."""
    parts = []
    for token in re.split(r"(\{[^}]*\}|<[^>]*>)", name):
        if token.startswith("{") and "|" in token:
            parts.append("(?:" + "|".join(re.escape(t) for t in token[1:-1].split("|")) + ")")
        elif token.startswith(("{", "<")):
            parts.append(".+")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts))


def undocumented(file_name: str, names: Iterable[str]) -> list[str]:
    """Those of *names* -- a header's columns, or the metrics a file lists --
    that no column of *file_name*'s entry describes."""
    patterns = [_name_pattern(c.name) for c in output_file(file_name).columns]
    return [n for n in names if not any(p.fullmatch(n) for p in patterns)]


__all__ = [
    "Column",
    "EXPORT_DICTIONARY_PAGE",
    "FILES",
    "OutputFile",
    "PERTURBATION_FOLDER",
    "RUN_FOLDER",
    "UNITS",
    "UNIT_SUFFIXES",
    "export_dictionary_markdown",
    "output_file",
    "undocumented",
    "write_export_dictionary_markdown",
]


if __name__ == "__main__":  # pragma: no cover
    print(write_export_dictionary_markdown(Path(__file__).resolve().parents[3] / EXPORT_DICTIONARY_PAGE))
