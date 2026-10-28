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

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

from haemolynx.statistics.glossary import UNITS as STATISTICS_UNITS
from haemolynx.visualization.vtk_io import ARRAY_UNITS, POINTS_AXIS_ORDER, POINTS_UNIT, RUN_FIELD_DATA

logger = logging.getLogger(__name__)

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
    "The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO. "
    "Under a size convention (classify_vessels_by_size) Unknown_Large, Unknown_Small and "
    "Unknown_Cap are the wide, mid and narrow classes (vessel_size_class).",
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
    _c("two_point", "-", "Two-point resistance nodes: start or end of the pair the resistance is found between, empty for the others."),
    _c("leaves_image", "flag", "Terminal nodes: the dead end leaves the image."),
    _c("dead_end_kind", "-", "The node's dead-end kind (graph.dead_ends)."),
    _c("depth_um", "um", "How far inside the image the node is (distance to the nearest face)."),
    _c("issue", "-", "Boundary issues: what is wrong with the node."),
    _c("kinds", "-", "Dead ends: every test the end met (;-separated)."),
    _c("tip_radius_um", "um", "Dead ends: the lumen radius at the tip."),
    _c("junction_radius_um", "um", "Dead ends: the lumen radius where its chain meets a junction."),
    _c("mask_run_um", "um", "Dead ends: how far the segmentation runs on past the tip."),
    _c("ahead_um", "um", "Dead ends: the distance to the vessel straight ahead."),
    _c("ahead_turn_deg", "degrees", "Dead ends: the turn from the tip's heading onto the vessel ahead."),
    _c("ahead_background_um", "um", "Broken vessels: the background the straight join to the vessel ahead crosses."),
    _c("ahead_junctions_away", "count", "Dead ends: how many junctions away that vessel is in the graph."),
    _c("face_distance_um", "um", "Dead ends: the tip's distance to the nearest image face."),
    _c("in_mask_fraction", "fraction", "Dead ends: the share of the chain inside the segmentation."),
    _c("ahead_edge", "-", "Dead ends: the vessel straight ahead, as (u, v, key)."),
    _c("ahead_point_um", "um", "Dead ends: the point on it the join would reach, (z, y, x)."),
    _c("thick_lumen_um", "um", "Dead ends at thick vessels: how far the tip lies past the thick vessel's lumen (negative inside it)."),
    _c("thick_centreline_um", "um", "Dead ends at thick vessels: the tip's distance to the thick vessel's centreline tree."),
    _c("thick_turn_deg", "degrees", "Dead ends at thick vessels: the turn from the tip's heading onto that centreline."),
    _c("shorter_um", "um", "Two in one lumen: the shorter vessel's length."),
    _c("lengths_um", "um", "Two in one lumen, doubled stretches: both vessels' lengths (;-separated)."),
    _c("cuts_off", "-", "Two in one lumen: for each, whether deleting it cuts vessels off (;-separated)."),
    _c("run_um", "um", "Doubled stretches: how long the stretch beside the other vessel is."),
    _c("vessel_radius_um", "um", "Doubled stretches: the mean lumen radius along the stretch."),
    _c("other_radius_um", "um", "Doubled stretches: the mean lumen radius along the other vessel beside it."),
    _c(
        "seen_from", "-",
        "Doubled stretches: which vessel sees the other beside it, each judged by its own lumen radius: "
        "both, wider_only or narrower_only.",
    ),
    _c("dead_end", "flag", "Doubled stretches: either vessel ends in a dead end."),
    _c("join", "-", "Broken vessels: how it would be joined: ends (two dead ends to each other) or side (a dead end to the side of the vessel ahead)."),
    _c("gap_um", "um", "Broken vessels: the gap between the two tips, or from the tip to the vessel ahead."),
    _c("turn_deg", "degrees", "Broken vessels: the larger turn from either end's heading onto the join (the tip's, for a side join)."),
    _c("refused", "-", "Broken vessels: why the build did not join two facing ends."),
    _c("volume_um3", "um^3", "Uncovered mask: the region's volume."),
    _c("radius_um", "um", "Uncovered mask: its widest inscribed radius."),
    _c("voxels", "count", "Uncovered mask, skeleton the graph does not follow: its voxel count."),
    _c("reason", "-", "Skeleton the graph does not follow: why the skeleton re-trace leaves the piece."),
    _c("hole_um", "um", "Skeleton the graph does not follow: the tissue hole the loop the piece would close runs round."),
    _c("far_fraction", "fraction", "Skeleton the graph does not follow: the share of the piece a separate tube from every centreline."),
    _c("mask_fraction", "fraction", "Skeleton the graph does not follow: the share of the piece in the segmentation."),
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
    "cut_by_image_face": "1 for a vessel on the chain of a dead end that leaves the image: the image faces cut it, so its length is censored.",
    "vessel_size_class": (
        "Under a size convention (classify_vessels_by_size) only: wide, mid or narrow, the "
        "vessel's measured width against the thresholds in the field data "
        "(vessel_size_mid_min_um, vessel_size_wide_min_um); empty on a bridge."
    ),
    "discharge_haematocrit": "The red-cell volume fraction of the blood the vessel carries (with a distributed haematocrit model only); blank where no inlet and outlet reach it, as the flows are.",
    "pressure_u": "Pressure at the edge_u end.",
    "pressure_v": "Pressure at the edge_v end.",
    "pressure_drop": "pressure_u - pressure_v.",
    "flow_signed": "Flow from edge_u to edge_v (negative: the other way).",
    "flow_abs": "The size of the flow, whichever way.",
}
NODE_ARRAYS: dict[str, str] = {
    "node_id": "The graph's node ID (the IDs boundary settings and the panel use).",
    "dead_end_kind": "For a dead end: leaves_image, island_end, artefact, thick_wall, stops_short, broken, blind or unknown; empty for every other node.",
    "boundary_role": "inlet, outlet, open_end_feeds or open_end_drains (an open end leaving the image that the solve gave a condition, by which way its flow runs), open_end (given a condition, no flow), or empty.",
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
        + f" Field data also holds {', '.join(RUN_FIELD_DATA[:-1])} and {RUN_FIELD_DATA[-1]}."
        " Flow arrays are written once the network is solved.",
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
        "{stem}_pericytes.csv", _RUN,
        "Each pericyte of the pericyte measurements' mask (measure_pericytes): where it sits, "
        "its diameter profile's summary, its distance to its vessel's nearest junction and its "
        "branch's resistances. Columns a measurement that is off would fill are blank.",
        (
            _c("pericyte", "-", "The pericyte's number: its connected component's, counted from 1."),
            _c("centroid_z_um", "um", "Its centroid, z."),
            _c("centroid_y_um", "um", "Its centroid, y."),
            _c("centroid_x_um", "um", "Its centroid, x."),
            _c("equivalent_diameter_um", "um", "The diameter of the sphere of its volume."),
            _c("placed", "flag",
               "True when it is placed on a capillary: within the assignment distance of the "
               "nearest one and of a pericyte's size, as the pericyte perturbations place it."),
            _c("reason", "-", "Why it is not placed, when it is not."),
            _c("branch_id", "-", "Its vessel's branch ID (its place in the graph's edge order, as the panel shows it)."),
            *_EDGE_ID,
            _BRANCH_ORDER,
            _c("distance_to_vessel_um", "um", "Its centroid's distance to the nearest capillary's centreline."),
            _c("arc_position_um", "um",
               "Its midline: where it projects onto the vessel's centreline, measured along it from u."),
            _c("vessel_length_um", "um", "The vessel's centreline, u to v."),
            _c("midline_z_um", "um", "Its midline, z."),
            _c("midline_y_um", "um", "Its midline, y."),
            _c("midline_x_um", "um", "Its midline, x."),
            _c("profile_file", "-", "Its diameter profile, in {stem}_pericyte_profiles/."),
            _c("profile_readings", "count", "Readings in the profile."),
            _c("diameter_at_pericyte_um", "um", "The diameter read at its midline (pericyte_profile_reading)."),
            _c("narrowest_diameter_um", "um", "The narrowest resolved reading of its profile."),
            _c("narrowest_offset_um", "um", "Where that reading is, from the midline (negative toward u)."),
            _c("nearest_junction_node", "-",
               "The vessel's nearest junction: the nearer of its ends that three or more vessels "
               "meet at, or where it opens into a thick vessel. Blank when neither end is one."),
            _c("junction_end", "-", "Which end of the vessel it is: u or v."),
            _c("junction_vessels", "vessels", "Vessels meeting there."),
            _c("junction_distance_along_vessel_um", "um", "From the midline to the junction along the vessel."),
            _c("junction_straight_line_distance_um", "um", "From the midline to the junction in a straight line."),
            _c("branch_length_um", "um", "Its branch's length (the one its resistance is solved over)."),
            _c("branch_diameter_um", "um", "The diameter its branch is solved with."),
            _c("branch_haematocrit", "fraction", "Its branch's discharge haematocrit."),
            _c("branch_pericytes", "count", "Measured pericytes on its branch."),
            _c("branch_uniform_resistance_pa_s_per_m3", "Pa.s/m^3",
               "Its branch's resistance as a uniform tube."),
            _c("branch_constricted_resistance_pa_s_per_m3", "Pa.s/m^3",
               "Its branch's resistance narrowed as its pericytes' profiles measure it."),
            _c("branch_resistance_ratio", "dimensionless", "Constricted over uniform."),
        ),
    ),
    OutputFile(
        "{stem}_pericyte_branches.csv", _RUN,
        "Each branch with a measured pericyte on it (measure_pericytes with its diameter "
        "profiles): its resistance as a uniform tube and with the narrowing its pericytes' "
        "profiles measure. The profile scales the branch's own diameter: at each point the "
        "diameter times the reading there over reference_diameter_um; where several profiles "
        "cover a point the nearest pericyte's is read, and beyond them the diameter is kept.",
        (
            _c("branch_id", "-", "The branch's ID (its place in the graph's edge order)."),
            *_EDGE_ID,
            _BRANCH_ORDER,
            _c("length_um", "um", "Its length."),
            _c("diameter_um", "um", "The diameter it is solved with: its uniform diameter."),
            _c("haematocrit", "fraction", "Its discharge haematocrit."),
            _c("pericytes", "count", "Measured pericytes on it."),
            _c("pericyte_numbers", "-", "Their numbers (JSON list), as {stem}_pericytes.csv numbers them."),
            _c("resolved_readings", "count",
               "Their profiles' readings of a closed cross-section of the segmentation, or of a "
               "FWHM that passed its checks (pericyte_profile_reading raw_image_fwhm), at least "
               "2 um wide: what the narrowing is read from."),
            _c("reference_diameter_um", "um",
               "Its typical width in the segmentation: the median of such readings every "
               "pericyte_profile_step_um along its whole length."),
            _c("narrowest_diameter_um", "um", "The narrowest diameter it is given."),
            _c("uniform_resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance as a uniform tube."),
            _c("constricted_resistance_pa_s_per_m3", "Pa.s/m^3", "Its resistance narrowed as measured."),
            _c("resistance_ratio", "dimensionless", "Constricted over uniform."),
        ),
    ),
    OutputFile(
        "{stem}_pericyte_profiles/pericyte_<NNNN>.csv", _RUN,
        "One pericyte's diameter profile (measure_pericytes with its diameter profiles): the "
        "vessel's width at its midline and every pericyte_profile_step_um each way along it, out "
        "to pericyte_profile_reach_um or the vessel's end: the segmentation's, or the raw image's "
        "FWHM (pericyte_profile_reading).",
        (
            _c("offset_um", "um", "From the midline along the vessel: negative toward u, positive toward v."),
            _c("toward_node", "-", "The end of the vessel the offset heads to (blank at the midline)."),
            _c("arc_position_um", "um", "Along the vessel's centreline from u."),
            _c("z_um", "um", "The reading's point, z."),
            _c("y_um", "um", "The reading's point, y."),
            _c("x_um", "um", "The reading's point, x."),
            _c("diameter_um", "um",
               "The vessel's width there: the segmentation's cross-section's area-equivalent "
               "diameter, else twice its inscribed radius; or, with pericyte_profile_reading "
               "raw_image_fwhm, the raw image's FWHM across it, fitted as the Diameters stage "
               "fits its samples. Blank where nothing reads."),
            _c("diameter_method", "-", "cross_section, inscribed_radius, fwhm, or blank."),
            _c("resolved", "flag", "True when the reading is at least 2 um: a width, not a strand of mask."),
            _c("diameter_change_um", "um", "The reading minus the one at the midline."),
            _c("diameter_ratio", "dimensionless", "The reading over the one at the midline."),
            _c("distance_to_vessel_end_um", "um", "Along the vessel to its nearer end."),
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
            _c(
                "Members low", "-",
                "A batch template only: the lowest of its runs' own values, each at the average's "
                "box, in the characteristic's unit.",
            ),
            _c("Members high", "-", "The highest of the same."),
            _c(
                "Within members", "-",
                "yes or no: whether the synthetic value lies between them; empty for a one-run "
                "template, a correlation, or where the synthetic network has no value.",
            ),
        ),
    ),
    # -- the run's record ----------------------------------------------------
    OutputFile(
        "{stem}_run_manifest.json", _RUN,
        "What produced the outputs: the settings as the run started, the HaemoLynx version, "
        "git commit, Python, platform and dependency versions, each input file's size, time and "
        "sha256, and each stage's start, end, wall time (seconds) and peak memory (bytes); "
        "checks.lumen_artefacts holds the lumen-artefact report's counts on the graph graph building "
        "handed on (null where it did not run: the low-RAM option, a synthetic network): pairs of "
        "edges in one lumen, loops inside one lumen, dead-end tips in another vessel's lumen and "
        "whether any was found, as the run's log warns.",
        (
            _c("stages[].wall_seconds", "s", "The stage's wall time."),
            _c("stages[].peak_rss_bytes", "bytes", "The stage's peak memory."),
            _c("inputs[].size_bytes", "bytes", "An input file's size."),
            _c("checks.lumen_artefacts.doubled_centreline_um", "um",
               "Centreline running beside another in the same lumen."),
            _c("checks.lumen_artefacts.centreline_um", "um", "The graph's centreline, as the report samples it."),
            _c("checks.vessel_classes_source.{mid|wide}_min_um", "um",
               "Under a size convention (classify_vessels_by_size) only: the thresholds the "
               "vessels were classed by; beside them counts (vessels per class), basis_counts "
               "(widths per diameter basis) and unsized (vessels with no width, by what became "
               "of them)."),
        ),
        "Every other key is text or a nested record; see pipeline/provenance.py.",
    ),
    OutputFile(
        "{stem}_run_settings.yaml", _RUN,
        "The settings the run started with, as a config file --config loads. Each setting's "
        "unit is in its help (the comments of a config write_default_config writes).",
    ),
    OutputFile(
        "{stem}_citations.txt", _RUN,
        "Plain text: the publications behind the methods this run used, by section, to cite "
        "in a methods section (pipeline/citations.py). Written when export_citations is on.",
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
            _c("baseline_flow_abs_m3_s", "m^3/s", "Its flow before the block; blank where no inlet and outlet reach it."),
            _c("blocked_flow_abs_m3_s", "m^3/s", "Its flow blocked (exactly 0); blank as above."),
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
        "{name}_mask_distribution.csv", _PERTURBATION,
        "A perturbation reading a mask of objects (a pericyte perturbation with "
        "use_pericyte_mask_constriction, or a capillary block read from a mask): how densely its "
        "objects sit on the capillaries and where they lie in the image's depth, one metric a row "
        "(metric, value). Each metric's unit is in its name; the rows are listed below as columns.",
        (
            _c("metric", "-", "The metric's name."),
            _c("value", "-", "Its value, in the unit its name carries."),
            _c("mask_path", "-", "The mask."),
            _c("mask", "-", "Which mask: pericyte_mask or capillary_block_mask."),
            _c("objects", "count", "Objects in the mask (connected components)."),
            _c("objects_on_vessels", "count",
               "Of them, placed on a vessel: a pericyte near enough to a capillary and of a "
               "pericyte's size, a blob that blocks a vessel."),
            _c("objects_on_capillaries", "count", "Of those, on a capillary."),
            _c("capillary_vessels", "vessels",
               "The network's capillaries (B orders, Unknown_Cap: under a size convention, the "
               "narrow class)."),
            _c("capillary_length_um", "um", "Their length."),
            _c("objects_per_mm_capillary", "1/mm", "objects_on_capillaries per mm of capillary_length_um."),
            _c("vessel_{length|diameter|branch_angle}_n", "count",
               "Objects on vessels whose vessel has that measure: a branch angle only where the "
               "vessel leaves a unique lower-order parent."),
            _c("vessel_{length|diameter}_{mean|sd|median|min|max}_um", "um",
               "The length or diameter (width) of the vessel each object sits on, over the objects "
               "on vessels (a vessel holding two counts twice): mean, sample SD, median, least, greatest."),
            _c("vessel_branch_angle_{mean|sd|median|min|max}_deg", "degrees",
               "The same for the vessel's branch angle (its emergence angle, as "
               "{stem}_branch_statistics.csv averages it per order)."),
            _c("image_depth_um", "um", "The image's z extent, its first plane to its last (the crop box's, when cut)."),
            _c("depth_{min|max|range|mean|mode}_um_{all|on_vessels}", "um",
               "The objects' centroid z from the image's first plane, over every object or over "
               "those on vessels: least, greatest, greatest minus least, mean, and the peak of a "
               "Gaussian kernel density of them."),
            _c("depth_mode_bandwidth_um_{all|on_vessels}", "um",
               "That kernel density's bandwidth (Silverman's rule); 0 when the depths are all alike."),
        ),
    ),
    OutputFile(
        "{name}_mask_density_by_branch_order.csv", _PERTURBATION,
        "A perturbation reading a mask of objects: their density per branch order. A pericyte "
        "mask lists the capillary orders, the only ones it places pericytes on.",
        (
            _BRANCH_ORDER,
            _c("vessels", "vessels", "Vessels of that order (thick-vessel bridges are none)."),
            _c("length_um", "um", "Their length."),
            _c("objects", "count",
               "Objects placed on them, each once: a blob on the vessel holding most of its voxels."),
            _c("objects_per_mm", "1/mm", "objects per mm of length_um."),
        ),
    ),
    OutputFile(
        "{name}_mask_objects.csv", _PERTURBATION,
        "A perturbation reading a mask of objects: one row per object of the mask, and the vessel "
        "the perturbation placed it on (blank when on none).",
        (
            _c("object", "-",
               "The object's number: its connected component's, as {name}_mask_blocks.csv numbers a blob."),
            _c("centroid_z_um", "um", "Its centroid, z: its depth from the image's first plane."),
            _c("centroid_y_um", "um", "Its centroid, y."),
            _c("centroid_x_um", "um", "Its centroid, x."),
            _c("volume_um3", "um^3", "Its volume."),
            _c("on_vessel", "flag",
               "True when it was placed on a vessel: a pericyte near enough to a capillary and of a "
               "pericyte's size, a blob that blocks."),
            _c("branch_id", "-", "The vessel's branch ID (its place in the graph's edge order, as the panel shows it)."),
            *_EDGE_ID,
            _BRANCH_ORDER,
            _c("capillary", "flag",
               "True when the vessel is a capillary (B orders, Unknown_Cap: under a size "
               "convention, the narrow class)."),
            _c("vessel_length_um", "um", "The vessel's length."),
            _c("vessel_diameter_um", "um", "Its diameter (width); blank where it has none."),
            _c("vessel_branch_angle_deg", "degrees",
               "Its branch angle: how far it turns from its parent where it leaves it (the emergence "
               "angle {stem}_branch_statistics.csv averages); blank where its junctions have no "
               "unique lower-order parent."),
        ),
    ),
    OutputFile(
        "{name}_pericyte_branches.csv", _PERTURBATION,
        "A pericyte_measured_constriction perturbation: each branch it narrowed, one row each. "
        "The perturbation re-solves the network with each one's resistance scaled by its "
        "resistance_ratio.",
        columns_note="Columns as {stem}_pericyte_branches.csv.",
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
            _c("Size class", "-",
               "Under a size convention (classify_vessels_by_size): wide, mid or narrow by its "
               "measured width; empty otherwise."),
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
            _c("segmented", "-", "Its segmented image (blank where ilastik segments its raw image)."),
            _c("raw", "-", "Its raw image."),
            _c("large_arteriole_mask", "-", "Its large arteriole mask."),
            _c("large_venule_mask", "-", "Its large venule mask."),
            _c("small_arteriole_mask", "-", "Its small arteriole mask."),
            _c("small_venule_mask", "-", "Its small venule mask."),
            _c("pericyte_mask", "-", "Its pericyte mask, from the batch's folder of them."),
            _c("cell_mask", "-", "Its cell mask, from the batch's folder of them."),
            _c(
                "measurement_3d_vessel_mask", "-",
                "Its vessel mask for the 3D distances, from the batch's folder of them.",
            ),
            _c("tissue_raw_tiff", "-", "Its tissue image, from the batch's folder of them."),
            _c("capillary_block_mask", "-", "Its mask of blocks, from the batch's folder of them."),
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
        "{condition}.haemobatch", _BATCH,
        "A synthetic template per condition (batch.haemobatch with none): gzip JSON of each "
        "finished image's templates, by its statistics report and by its graph, with the runs "
        "left out and why and the batch's batch_settings.yaml. A synthetic run loads it as its "
        "template, the runs averaged "
        "(PLAN_batch_templates.md).",
        columns_note="NetworkTemplate fields; see synthetic/batch_template.py.",
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
    from haemolynx.parsers import configure_console_logging

    configure_console_logging()
    logger.info(
        "Wrote %s",
        write_export_dictionary_markdown(Path(__file__).resolve().parents[3] / EXPORT_DICTIONARY_PAGE),
    )
