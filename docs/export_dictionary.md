# Output data dictionary

<!-- Written by `python -m haemolynx.pipeline.export_dictionary` from
     src/haemolynx/pipeline/export_dictionary.py; edit that, not this. -->

Every file a run writes, and every column, array or key in it, with its unit. `{stem}` is the run's stem (the segmented image's name, or `{template}_synthetic`), `{name}` a perturbation's name. The rows of the statistics CSVs are defined in [the statistics glossary](statistics_glossary.md).

Units are spelled one way everywhere. A column name in code style (lower case, underscores) carries its unit as a suffix: `_um` (um), `_um2` (um^2), `_um3` (um^3), `_pa` (Pa), `_m3_s` (m^3/s), `_pa_s_per_m3` (Pa.s/m^3), `_m3_per_pa_s` (m^3/(Pa.s)), `_s` (s), `_mm_s` (mm/s), `_deg` (degrees), `_percent` (percent), `_bytes` (bytes), `_vox` (voxel). A name without one is a count, a fraction, a ratio, a flag or a label. The VTK files keep their array names and carry each array's unit in their field data as `<array>_unit`.

## Units

| Unit | Meaning |
|---|---|
| `um` | micrometres (microns) |
| `um^2` | square micrometres |
| `um^3` | cubic micrometres |
| `um/um^3` | micrometres of vessel per cubic micrometre (length density) |
| `um^2/um^3` | square micrometres of vessel wall per cubic micrometre |
| `1/um` | per micrometre |
| `1/mm` | per millimetre of vessel |
| `1/mm^3` | per cubic millimetre |
| `mm/s` | millimetres per second |
| `nL/min` | nanolitres per minute |
| `m^3/s` | cubic metres per second |
| `m^3/(Pa.s)` | hydraulic conductance, cubic metres per pascal second |
| `Pa` | pascals |
| `Pa.s/m^3` | hydraulic resistance, pascal seconds per cubic metre |
| `s` | seconds |
| `fL` | femtolitres |
| `cells/s` | red cells per second |
| `cells/mm` | red cells per millimetre of vessel |
| `degrees` | degrees of angle |
| `vessels` | a number of vessels (graph edges); as a route length, the vessels along it |
| `count` | a number of things other than vessels (nodes, junctions, beds, loops, samples) |
| `fraction` | a share between 0 and 1 |
| `dimensionless` | a ratio or index with no unit |
| `intensity` | the raw image's own grey values |
| `-` | no unit: an identifier, a label or text |
| `flag` | 0 or 1 (or False and True): no and yes |
| `percent` | a percentage |
| `voxel` | voxel indices (array positions in the image), not microns |
| `bytes` | bytes of memory |

## `{stem}_vessels.vtp`

Written to the run's output folder (vtk_output_prefix's). Every vessel as a polyline (VTK PolyData), one cell per vessel. Points are physical (z, y, x) microns, as node positions are; field data ``points_unit`` (um) and ``points_axis_order`` say so, and ``<array>_unit`` gives each array's unit. Field data also holds haemolynx_version, viscosity_law, diameter_basis, haematocrit, haematocrit_model and open_end_model. Flow arrays are written once the network is solved.

| Column | Unit | Meaning |
|---|---|---|
| `edge_u` | `-` | Node ID at one end (the nodes file's node_id). |
| `edge_v` | `-` | Node ID at the other end. |
| `edge_key` | `-` | Which of the vessels between edge_u and edge_v (0 for the first). |
| `branch_order` | `-` | The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO. |
| `resistance` | `Pa.s/m^3` | The vessel's hydraulic resistance (Poiseuille, with the run's viscosity law). |
| `conductance` | `m^3/(Pa.s)` | 1 / resistance. |
| `length_um` | `um` | The vessel's length along its (smoothed) centreline. |
| `diameter_um` | `um` | The vessel's diameter, as the haemodynamics used it. |
| `diameter_source` | `-` | Where the diameter came from: fwhm, raw_section, endothelial, edt, class_median, table, bridge, synthetic, ... |
| `is_zero_resistance` | `flag` | 1 for a thick-vessel bridge (a join drawn inside a lumen, given negligible resistance). |
| `flow_solved` | `flag` | 1 where the solve gave the vessel a flow; 0 where it did not (see flow_state). |
| `flow_state` | `-` | solved, blind_end (behind a dead end inside the tissue: flow exactly 0), closed_end (behind a face end with no condition: flow exactly 0) or no_boundary (no inlet and outlet reach it: flows blank). |
| `cut_by_image_face` | `flag` | 1 for a vessel on the chain of a dead end that leaves the image: the image faces cut it, so its length is censored. |
| `discharge_haematocrit` | `fraction` | The red-cell volume fraction of the blood the vessel carries (with a distributed haematocrit model only); blank where no inlet and outlet reach it, as the flows are. |
| `pressure_u` | `Pa` | Pressure at the edge_u end. |
| `pressure_v` | `Pa` | Pressure at the edge_v end. |
| `pressure_drop` | `Pa` | pressure_u - pressure_v. |
| `flow_signed` | `m^3/s` | Flow from edge_u to edge_v (negative: the other way). |
| `flow_abs` | `m^3/s` | The size of the flow, whichever way. |

## `{stem}_nodes.vtp`

Written to the run's output folder (vtk_output_prefix's). Every node as a point: junctions, dead ends, inlets and outlets. Points are physical (z, y, x) microns, as node positions are; field data ``points_unit`` (um) and ``points_axis_order`` say so, and ``<array>_unit`` gives each array's unit.

| Column | Unit | Meaning |
|---|---|---|
| `node_id` | `-` | The graph's node ID (the IDs boundary settings and the panel use). |
| `dead_end_kind` | `-` | For a dead end: leaves_image, island_end, artefact, thick_wall, stops_short, broken, blind or unknown; empty for every other node. |
| `boundary_role` | `-` | inlet, outlet, open_end_feeds or open_end_drains (an open end leaving the image that the solve gave a condition, by which way its flow runs), open_end (given a condition, no flow), or empty. |
| `pressure` | `Pa` | The node's pressure; blank where no inlet and outlet reach it. |

## `{stem}_pericytes.vtp`

Written to the run's output folder (vtk_output_prefix's). Constriction sites placed along every vessel at the run's constriction spacing and length (constriction_spacing_um, constriction_length_um). Points are physical (z, y, x) microns, as node positions are; field data ``points_unit`` (um) and ``points_axis_order`` say so, and ``<array>_unit`` gives each array's unit.

| Column | Unit | Meaning |
|---|---|---|
| `edge_u` | `-` | Node ID at one end of the vessel the constriction site is on. |
| `edge_v` | `-` | Node ID at its other end. |
| `edge_key` | `-` | Which of the vessels between them. |
| `branch_order` | `-` | That vessel's class. |

## `{stem}_tissue_surface.vtp`

Written to the run's output folder (vtk_output_prefix's). The tissue's surface as a triangle mesh (with measure_tissue_volume). Points are physical (z, y, x) microns, as node positions are; field data ``points_unit`` (um) and ``points_axis_order`` say so, and ``<array>_unit`` gives each array's unit. It has no arrays.

## `{stem}_graph.pkl`

Written to the run's output folder (vtk_output_prefix's). The network as a pickled networkx MultiGraph: node pos and edge voxels in physical (z, y, x) microns, every attribute the run set (CLAUDE.md, Graph). Read it with the same HaemoLynx version.

## `{stem}_skeleton.npy`

Written to the run's output folder (vtk_output_prefix's). The skeleton as a NumPy array, voxel-indexed (z, y, x), 1 on the centreline.

## `{stem}_voxel_size.json`

Written to the run's output folder (vtk_output_prefix's). The voxel size the run read and used, and where it came from.

| Column | Unit | Meaning |
|---|---|---|
| `voxel_size` | `um` | The image's voxel size as read, (x, y, z). |
| `run_voxel_size` | `um` | The voxel size the run used, (x, y, z) (after any downsampling). |
| `voxel_size_source` | `-` | Where it came from: OME header, TIFF tags, override, ... |
| `voxel_metadata_status` | `-` | Whether the metadata was found, and its source. |
| `voxel_size_policy` | `-` | The voxel_size_policy setting. |
| `voxel_size_override_xyz` | `um` | The voxel_size_override_xyz setting, if any. |
| `downsample` | `-` | The downsample-on-import record, or null. |

## `{stem}_statistics.csv`

Written to the run's output folder (vtk_output_prefix's). The statistics report, one row per metric (with the Statistics toggle).

Each row's definition and unit: docs/statistics_glossary.md.

| Column | Unit | Meaning |
|---|---|---|
| `Section` | `-` | The report section the metric is in. |
| `Metric` | `-` | The metric's name. |
| `Value` | `-` | Its value (or N/A and why). |
| `Unit` | `-` | Its unit. |
| `Notes` | `-` | Its definition. |

## `{stem}_statistics_connected.csv`

Written to the run's output folder (vtk_output_prefix's). The geometry and topology statistics again, over the inlet/outlet-connected network only.

Columns as {stem}_statistics.csv; rows in docs/statistics_glossary.md.

## `{stem}_branch_statistics.csv`

Written to the run's output folder (vtk_output_prefix's). Statistics per branch order, one row per order.

Every column is defined, with its unit, in docs/statistics_glossary.md.

## `{stem}_capillary_bed_joins.csv`

Written to the run's output folder (vtk_output_prefix's). One row per capillary bed: the B order at which it joins another bed (statistics_capillary_bed_joins).

Every column is defined, with its unit, in docs/statistics_glossary.md.

## `{stem}_tissue_volume.csv`

Written to the run's output folder (vtk_output_prefix's). The tissue's volume, surface and the vessel densities it gives (measure_tissue_volume).

Columns as {stem}_statistics.csv; rows in docs/statistics_glossary.md.

## `{stem}_betweenness_communities_resistance.json`

Written to the run's output folder (vtk_output_prefix's). Weighted betweenness and vascular communities, edges weighted by resistance: the Weighted Betweenness and Communities section of the report, as JSON.

Keys as the report's rows; docs/statistics_glossary.md.

## `{stem}_betweenness_communities_edge_length.json`

Written to the run's output folder (vtk_output_prefix's). The same, edges weighted by length.

Keys as the report's rows; docs/statistics_glossary.md.

## `{stem}_betweenness_communities_edge_flow.json`

Written to the run's output folder (vtk_output_prefix's). The same, edges weighted by flow.

Keys as the report's rows; docs/statistics_glossary.md.

## `{stem}_cell_to_vessel_3d_dual_distances.csv`

Written to the run's output folder (vtk_output_prefix's). Each object of the cell mask and its distance to the vessels (measurement_3d_to_cell_mask).

| Column | Unit | Meaning |
|---|---|---|
| `Object ID` | `-` | The object's label in the cell mask. |
| `Object Name` | `-` | Its name. |
| `Voxel Count` | `count` | Its size in voxels. |
| `Centroid Z (voxel)` | `voxel` | Its centroid, z index. |
| `Centroid Y (voxel)` | `voxel` | Its centroid, y index. |
| `Centroid X (voxel)` | `voxel` | Its centroid, x index. |
| `Centroid Z (um)` | `um` | Its centroid, z. |
| `Centroid Y (um)` | `um` | Its centroid, y. |
| `Centroid X (um)` | `um` | Its centroid, x. |
| `Vessel Contact` | `-` | separate, touching, overlapping or inside. |
| `Vessel Overlap Voxel Count` | `count` | Its voxels inside a vessel. |
| `Edge-to-Vessel Distance (um)` | `um` | Gap between the object's surface and the nearest vessel (0 touching). |
| `Edge-to-Vessel Nearest Z (um)` | `um` | The nearest vessel point, z. |
| `Edge-to-Vessel Nearest Y (um)` | `um` | The nearest vessel point, y. |
| `Edge-to-Vessel Nearest X (um)` | `um` | The nearest vessel point, x. |
| `Edge-to-Vessel Nearest Coordinate (z,y,x um)` | `um` | The same point as one (z, y, x). |
| `Edge-to-Vessel Nearest Graph Edge Label ID` | `-` | The vessel whose lumen that point is in. |
| `Edge-to-Vessel Nearest Branch Order` | `-` | That vessel's class. |
| `Centroid-to-Vessel Distance (um)` | `um` | Distance from the object's centroid to the nearest vessel. |
| `Delta Distance (Centroid-Edge) (um)` | `um` | Centroid distance minus surface distance. |
| `Centroid-to-Vessel Nearest Z (um)` | `um` | The vessel point nearest the centroid, z. |
| `Centroid-to-Vessel Nearest Y (um)` | `um` | The same, y. |
| `Centroid-to-Vessel Nearest X (um)` | `um` | The same, x. |
| `Centroid-to-Vessel Nearest Coordinate (z,y,x um)` | `um` | The same point as one (z, y, x). |
| `Centroid-to-Vessel Nearest Graph Edge Label ID` | `-` | The vessel whose lumen that point is in. |
| `Centroid-to-Vessel Nearest Branch Order` | `-` | That vessel's class. |

## `{stem}_cell_to_vessel_3d_dual_distances_summary.csv`

Written to the run's output folder (vtk_output_prefix's). Those distances summarised overall and per branch order of the nearest vessel.

| Column | Unit | Meaning |
|---|---|---|
| `Group` | `-` | Overall, or a branch order. |
| `Measurement Type` | `-` | edge_to_vessel or centroid_to_vessel. |
| `Object Count` | `count` | Objects in the group. |
| `Mean Distance (um)` | `um` | Mean distance. |
| `Median Distance (um)` | `um` | Median distance. |
| `Std Distance (um)` | `um` | Standard deviation of the distance. |
| `Min Distance (um)` | `um` | Smallest distance. |
| `Max Distance (um)` | `um` | Largest distance. |
| `Mean Delta (Centroid-Edge) (um)` | `um` | Mean of centroid minus surface distance. |
| `Median Delta (Centroid-Edge) (um)` | `um` | Median of the same. |
| `Std Delta (Centroid-Edge) (um)` | `um` | Its standard deviation. |

## `{stem}_template_comparison.csv`

Written to the run's output folder (vtk_output_prefix's). A synthetic run only: each characteristic of the template against the synthetic network.

| Column | Unit | Meaning |
|---|---|---|
| `Section` | `-` | box, endpoints, counts, lengths, diameters, ... |
| `Characteristic` | `-` | What is compared; its unit is in its name, e.g. extent z (um). |
| `Template` | `-` | The template's value, in the characteristic's unit. |
| `Synthetic` | `-` | The synthetic network's value, in the same unit. |
| `Ratio` | `dimensionless` | Synthetic over template. |
| `Tolerance` | `-` | The band the generator is held to (exact, rel 0.02, ...), or empty. |
| `Within` | `-` | yes or no; empty where nothing is fitted. |
| `Members low` | `-` | A batch template only: the lowest of its runs' own values, each at the average's box, in the characteristic's unit. |
| `Members high` | `-` | The highest of the same. |
| `Within members` | `-` | yes or no: whether the synthetic value lies between them; empty for a one-run template, a correlation, or where the synthetic network has no value. |

## `{stem}_run_manifest.json`

Written to the run's output folder (vtk_output_prefix's). What produced the outputs: the settings as the run started, the HaemoLynx version, git commit, Python, platform and dependency versions, each input file's size, time and sha256, and each stage's start, end, wall time (seconds) and peak memory (bytes).

Every other key is text or a nested record; see pipeline/provenance.py.

| Column | Unit | Meaning |
|---|---|---|
| `stages[].wall_seconds` | `s` | The stage's wall time. |
| `stages[].peak_rss_bytes` | `bytes` | The stage's peak memory. |
| `inputs[].size_bytes` | `bytes` | An input file's size. |

## `{stem}_run_settings.yaml`

Written to the run's output folder (vtk_output_prefix's). The settings the run started with, as a config file --config loads. Each setting's unit is in its help (the comments of a config write_default_config writes).

## `{name}_summary.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. One row: the perturbation's network-level result against the baseline.

| Column | Unit | Meaning |
|---|---|---|
| `name` | `-` | The perturbation's name. |
| `type` | `-` | Its type (perturbation_type). |
| `overrides` | `-` | The settings it changed, as JSON. |
| `inlet_pressure_pa` | `Pa` | Pressure held at the inlets for this point. |
| `outlet_pressure_pa` | `Pa` | Pressure held at the outlets. |
| `total_inlet_flow_m3_s` | `m^3/s` | Total flow into the network through its inlets. |
| `total_outlet_flow_m3_s` | `m^3/s` | Total flow through the outlets (negative: leaving). |
| `flow_balance_error_m3_s` | `m^3/s` | Inlet plus outlet flow: 0 when flow is conserved. |
| `equivalent_resistance_pa_s_per_m3` | `Pa.s/m^3` | The perturbed network's resistance: pressure drop over total inflow. |
| `baseline_equivalent_resistance_pa_s_per_m3` | `Pa.s/m^3` | The baseline network's, the same way. |
| `delta_vs_baseline_pa_s_per_m3` | `Pa.s/m^3` | Perturbed minus baseline. |
| `percent_change_vs_baseline` | `percent` | That change over the baseline. |

## `{name}_edges.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. One row per vessel of the perturbed network.

| Column | Unit | Meaning |
|---|---|---|
| `u` | `-` | Node ID at one end of the vessel (graph node, as in the nodes file's node_id). |
| `v` | `-` | Node ID at the other end. |
| `key` | `-` | Which of the vessels between u and v (0 for the first). |
| `branch_order` | `-` | The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO. |
| `length_um` | `um` | The vessel's length. |
| `diameter_um` | `um` | Its diameter after the perturbation. |
| `resistance_pa_s_per_m3` | `Pa.s/m^3` | Its resistance after the perturbation. |
| `conductance_m3_per_pa_s` | `m^3/(Pa.s)` | 1 / resistance. |
| `pressure_drop_pa` | `Pa` | Pressure at u minus pressure at v; blank where no inlet and outlet reach it. |
| `flow_signed_m3_s` | `m^3/s` | Flow from u to v (negative: the other way); blank as above, exactly 0 behind a dead end. |
| `flow_solved` | `flag` | True where the solve gave it a flow. |
| `flow_state` | `-` | solved, blind_end, closed_end or no_boundary (as in the vessels file). |
| `baseline_resistance_pa_s_per_m3` | `Pa.s/m^3` | Its resistance in the baseline network. |
| `resistance_ratio` | `dimensionless` | Perturbed over baseline resistance. |

## `{name}_statistics.csv, {name}_branch_statistics.csv, {name}_capillary_bed_joins.csv, {name}_betweenness_communities_*.json`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. The perturbed network's statistics, as the run's own files of those names.

Columns as the run's; rows in docs/statistics_glossary.md.

## `{name}_blocked_vessels.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A capillary block: one row per vessel it blocked.

| Column | Unit | Meaning |
|---|---|---|
| `branch_id` | `-` | The vessel's branch ID (its place in the graph's edge order, as the panel shows it). |
| `u` | `-` | Node ID at one end of the vessel (graph node, as in the nodes file's node_id). |
| `v` | `-` | Node ID at the other end. |
| `key` | `-` | Which of the vessels between u and v (0 for the first). |
| `branch_order` | `-` | The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO. |
| `length_um` | `um` | The vessel's length. |
| `diameter_um` | `um` | Its diameter. |
| `baseline_resistance_pa_s_per_m3` | `Pa.s/m^3` | Its resistance before the block. |
| `blocked_resistance_pa_s_per_m3` | `Pa.s/m^3` | Its resistance blocked. |
| `baseline_flow_abs_m3_s` | `m^3/s` | Its flow before the block; blank where no inlet and outlet reach it. |
| `blocked_flow_abs_m3_s` | `m^3/s` | Its flow blocked (exactly 0); blank as above. |

## `{name}_block_comparison.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A capillary block: how the blocked network's flow differs from the baseline's, one metric a row (metric, value). Each metric's unit is in its name; the rows are listed below as columns.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | `-` | The metric's name. |
| `value` | `-` | Its value, in the unit its name carries. |
| `selection` | `-` | How the vessels were chosen. |
| `branch_orders` | `-` | The orders drawn from (JSON list). |
| `probability` | `fraction` | The chance each candidate vessel was blocked. |
| `seed` | `-` | The random seed. |
| `candidate_vessels` | `vessels` | Vessels that could be blocked. |
| `vessel_ids` | `-` | The branch IDs asked for (JSON list). |
| `capillary_block_mask_path` | `-` | The mask of blocks, when read from a mask. |
| `mask_blocks` | `count` | Blobs in that mask. |
| `mask_blocks_on_vessels` | `count` | Blobs that blocked a vessel. |
| `mask_blocks_below_overlap` | `count` | Blobs too little inside the segmentation. |
| `mask_blocks_off_graph` | `count` | Blobs near no vessel. |
| `min_overlap_fraction` | `fraction` | capillary_block_mask_min_overlap_fraction. |
| `vessels_per_block` | `-` | capillary_block_mask_vessels_per_block. |
| `vessel_mask_source` | `-` | What the blobs were read against: the segmentation or the graph's lumens. |
| `blocked_vessels` | `vessels` | Vessels blocked. |
| `resistance_factor` | `dimensionless` | What a blocked vessel's resistance was multiplied by. |
| `total_inflow_baseline_m3_s` | `m^3/s` | Total inflow before the block. |
| `total_inflow_blocked_m3_s` | `m^3/s` | Total inflow blocked. |
| `total_inflow_percent_change` | `percent` | Its change. |
| `blocked_vessel_length_um` | `um` | Length of the blocked vessels. |
| `stopped_vessels` | `vessels` | Blocked vessels and every vessel fed only through one. |
| `stopped_vessel_length_um` | `um` | Their length. |
| `hypoperfusion_threshold_fraction` | `fraction` | The loss of flow that counts as hypoperfused. |
| `hypoperfused_vessels` | `vessels` | Other vessels that lost at least that share of their flow. |
| `hypoperfused_vessel_fraction` | `fraction` | Their share of the vessels carrying flow. |
| `hypoperfused_length_um` | `um` | Their length. |
| `hypoperfused_length_fraction` | `fraction` | Their share of the network's length. |
| `reversed_vessels` | `vessels` | Vessels whose flow now runs the other way. |
| `outlets` | `count` | Outlets compared. |
| `outlet_flow_retained_mean` | `fraction` | Mean over outlets of blocked over baseline outflow. |
| `outlet_flow_retained_min` | `fraction` | The least of the same. |
| `outlets_losing_half_their_flow` | `count` | Outlets keeping half their flow or less. |
| `mean_transit_time_s_{baseline\|blocked}` | `s` | Mean transit time, before / blocked. |
| `transit_time_sd_s_{baseline\|blocked}` | `s` | Its SD. |
| `capillary_mean_transit_time_s_{baseline\|blocked}` | `s` | Capillary mean transit time. |
| `capillary_transit_time_heterogeneity_s_{baseline\|blocked}` | `s` | Capillary transit time heterogeneity (CTH). |
| `transit_time_status_{baseline\|blocked}` | `-` | Why transit times could not be computed, when not. |
| `{capillary\|vessel}_{flow\|velocity\|red_cell_flux\|red_cell_velocity}_cv_{baseline\|blocked}` | `dimensionless` | Coefficient of variation over the vessels carrying flow. |
| `{vessels\|capillaries}_carrying_flow_{baseline\|blocked}` | `vessels` | Vessels carrying flow. |
| `flowing_{vessel\|capillary}_length_um_{baseline\|blocked}` | `um` | Their length. |
| `flowing_{vessel\|capillary}_volume_um3_{baseline\|blocked}` | `um^3` | Their volume. |
| `flux_heterogeneity_status_{baseline\|blocked}` | `-` | Why flux heterogeneity could not be computed, when not. |
| `draws` | `count` | With replicates: how many draws. |
| `<metric>_draws_{mean\|sd\|low\|high}` | `-` | With replicates: each metric's mean, SD and 2.5-97.5% range over the draws, in its own unit. |

## `{name}_block_replicates.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A capillary block with replicates: one row per draw.

| Column | Unit | Meaning |
|---|---|---|
| `draw` | `-` | The draw's number (1 is the first). |
| `seed` | `-` | Its seed. |
| `blocked_vessels` | `vessels` | Vessels it blocked. |
| `<metric>` | `-` | Then one column per numeric metric of {name}_block_comparison.csv, in its unit. |

## `{name}_flow_by_branch_order.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A capillary block: the comparison per branch order.

| Column | Unit | Meaning |
|---|---|---|
| `branch_order` | `-` | The vessel's class: Art01..., B01..., Ven01..., Large_Art/Large_Ven, Unknown_* or No_BO. |
| `vessels` | `vessels` | Vessels of that order. |
| `blocked` | `vessels` | Of them, blocked. |
| `baseline_mean_flow_m3_s` | `m^3/s` | Mean flow before the block. |
| `blocked_mean_flow_m3_s` | `m^3/s` | Mean flow blocked. |
| `mean_flow_percent_change` | `percent` | Its change. |
| `hypoperfused` | `vessels` | Vessels hypoperfused. |
| `hypoperfused_length_um` | `um` | Their length. |
| `reversed` | `vessels` | Vessels reversed. |

## `{name}_mask_blocks.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A capillary block read from a mask: one row per blob of the mask.

| Column | Unit | Meaning |
|---|---|---|
| `block` | `-` | The blob's number. |
| `voxels` | `count` | Its voxel count. |
| `volume_um3` | `um^3` | Its volume. |
| `centroid_z_um` | `um` | Its centroid, z. |
| `centroid_y_um` | `um` | Its centroid, y. |
| `centroid_x_um` | `um` | Its centroid, x. |
| `voxels_in_vessels` | `count` | Its voxels inside the segmentation. |
| `overlap_fraction` | `fraction` | voxels_in_vessels over voxels. |
| `blocks` | `flag` | True when it blocks a vessel. |
| `blocked_branch_ids` | `-` | The branch IDs it blocks (JSON list). |
| `vessel_voxels` | `-` | Its voxels in each vessel (JSON). |
| `reason` | `-` | Why it blocks nothing, when it does not. |

## `{arteriole\|capillary\|pericyte}_dilation_sweep.csv, ..._dilation_pressure_sweep.csv, inlet_pressure_sweep.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A dilation and/or inlet pressure sweep: one row per grid point.

| Column | Unit | Meaning |
|---|---|---|
| `dilation_percent` | `percent` | Diameter change applied at this point (+10 widens by 10%). |
| `dilation_factor` | `dimensionless` | The same change as a scale on the diameter (1.1 for +10%). |
| `inlet_pressure_pa` | `Pa` | Pressure held at the inlets for this point. |
| `outlet_pressure_pa` | `Pa` | Pressure held at the outlets. |
| `total_inlet_flow_m3_s` | `m^3/s` | Total flow into the network through its inlets. |
| `total_outlet_flow_m3_s` | `m^3/s` | Total flow through the outlets (negative: leaving). |
| `flow_balance_error_m3_s` | `m^3/s` | Inlet plus outlet flow: 0 when flow is conserved. |
| `equivalent_resistance_pa_s_per_m3` | `Pa.s/m^3` | The network's resistance: inlet minus outlet pressure over total inflow. |

## `pericyte_spacing_sweep.csv, pericyte_length_sweep.csv`

Written to the perturbation's folder: perturbation_output_dir/{name}_{type}. A pericyte geometry sweep: one row per spacing or length.

| Column | Unit | Meaning |
|---|---|---|
| `constriction_spacing_um` | `um` | Distance between constriction sites. |
| `constriction_length_um` | `um` | Length of each constriction. |
| `dilation_percent` | `percent` | Diameter change applied at this point (+10 widens by 10%). |
| `dilation_factor` | `dimensionless` | The same change as a scale on the diameter (1.1 for +10%). |
| `inlet_pressure_pa` | `Pa` | Pressure held at the inlets for this point. |
| `outlet_pressure_pa` | `Pa` | Pressure held at the outlets. |
| `total_inlet_flow_m3_s` | `m^3/s` | Total flow into the network through its inlets. |
| `total_outlet_flow_m3_s` | `m^3/s` | Total flow through the outlets (negative: leaving). |
| `flow_balance_error_m3_s` | `m^3/s` | Inlet plus outlet flow: 0 when flow is conserved. |
| `equivalent_resistance_pa_s_per_m3` | `Pa.s/m^3` | The network's resistance: inlet minus outlet pressure over total inflow. |

## `{stem}_connectivity.csv`

Written to the run's output folder (vtk_output_prefix's). Post processing's connectivity table: one row per vessel and what it joins.

| Column | Unit | Meaning |
|---|---|---|
| `Branch ID` | `-` | The vessel's branch ID. |
| `From node ID` | `-` | Its upstream node (by flow, where solved). |
| `To node ID` | `-` | Its downstream node. |
| `Branch order` | `-` | Its class. |
| `Length (um)` | `um` | Its length. |
| `Diameter (um)` | `um` | Its diameter. |
| `Notes` | `-` | Bridge, unsolved, inlet/outlet and so on. |
| `Upstream branch IDs` | `-` | Vessels feeding it. |
| `Downstream branch IDs` | `-` | Vessels it feeds. |
| `Edge u` | `-` | Graph node u. |
| `Edge v` | `-` | Graph node v. |
| `Edge key` | `-` | Graph edge key. |

## `{stem}_loop_review.csv`

Written to the run's output folder (vtk_output_prefix's). Post processing's short-loop decisions, one row each, with the loop's measures.

| Column | Unit | Meaning |
|---|---|---|
| `time` | `-` | When it was decided (ISO 8601). |
| `decision` | `-` | kept, or side deleted. |
| `side_deleted` | `-` | The side deleted, if any. |
| `wall_side` | `-` | The side the build's loop breaker would have taken out. |
| `loop_length_um` | `um` | The loop's length round. |
| `sides` | `count` | Its sides. |
| `loop_branch_ids` | `-` | Its vessels' branch IDs. |
| `loop_edges` | `-` | Its vessels as (u, v, key). |
| `centre_z_um` | `um` | Its centre, z. |
| `centre_y_um` | `um` | Its centre, y. |
| `centre_x_um` | `um` | Its centre, x. |
| `centre_zyx_vox` | `voxel` | Its centre as voxel indices (z y x), to find in napari. |
| `span_um` | `um` | Its widest extent. |
| `area_um2` | `um^2` | The area it encloses, in its best-fitting plane. |
| `out_of_plane` | `dimensionless` | Third over second singular value: 0 for a flat loop. |
| `path_in` | `fraction` | Share of the loop inside the segmentation. |
| `lumen_r_um` | `um` | Median lumen radius along it. |
| `chords_in_lumen` | `fraction` | Share of its chords crossing no more than a dropout of background. |
| `chord_bg_median_um` | `um` | Their median background: the loop's hole. |
| `inside_one_lumen` | `flag` | Whether it lies inside one lumen. |
| `side_lengths_um` | `um` | Each side's length (;-separated). |
| `side_diameters_um` | `um` | Each side's diameter (;-separated). |
| `side_lumen_r_um` | `um` | Each side's mean lumen radius (;-separated). |
| `zero_resistance_bridges` | `count` | Thick-vessel bridges in the loop. |
| `recovered_um` | `um` | Length of it mask recovery drew. |
| `bridged_um` | `um` | Length of it that is bridges. |

## `{stem}_{kind}_review.csv`

Written to the run's output folder (vtk_output_prefix's). Post processing's decisions on a review list (dead ends, hairpins, ...), one row each, with the item's measures after the first ten columns.

Only the measures of the item's own list are filled.

| Column | Unit | Meaning |
|---|---|---|
| `time` | `-` | When it was decided (ISO 8601). |
| `decision` | `-` | kept, or what was done (deleted, merged, ...). |
| `kind` | `-` | The review list (and, for a dead end, its kind). |
| `centre_z_um` | `um` | The item's centre, z. |
| `centre_y_um` | `um` | The item's centre, y. |
| `centre_x_um` | `um` | The item's centre, x. |
| `centre_zyx_vox` | `voxel` | Its centre as voxel indices (z y x). |
| `nodes` | `-` | Its node IDs (;-separated). |
| `branch_ids` | `-` | Its vessels' branch IDs (;-separated). |
| `why` | `-` | Why it was listed. |
| `connector_um` | `um` | Close junctions: the length of the vessel joining the two junctions. |
| `widest_radius_um` | `um` | Close junctions: the widest vessel's radius there. |
| `vessels` | `vessels` | How many vessels the item involves or meets. |
| `length_um` | `um` | The vessel's (or the chain's, or the piece's) length. |
| `chord_um` | `um` | Hairpins: the straight distance between the vessel's ends. |
| `ratio` | `dimensionless` | Hairpins: length over chord. Diameter jumps: wide over narrow. |
| `narrow_um` | `um` | Diameter jumps: the narrower diameter. |
| `wide_um` | `um` | Diameter jumps: the wider diameter. |
| `at` | `-` | Diameter jumps: pass-through (one vessel into the next) or junction. |
| `diameter_um` | `um` | The vessel's diameter (the widest vessel's, at a node). |
| `diameter_source` | `-` | Where that diameter came from: fwhm, edt, class_median, table, ... |
| `edt_diameter_um` | `um` | The mask's own width of the vessel, where measured. |
| `fwhm_diameter_um` | `um` | The FWHM width of the vessel, where measured. |
| `flow_share` | `fraction` | The vessel's flow over the network's total inflow. |
| `reasons` | `-` | Unmeasured diameters: why the width is in doubt. |
| `attached_at` | `-` | Unsolved pieces: the node IDs it hangs from (;-separated). |
| `disconnected` | `flag` | Unsolved pieces: True when it touches nothing that is solved. |
| `orders` | `-` | Flow against branch orders: the vessels' orders (;-separated). |
| `arteriole_meets_venule` | `flag` | Flow against branch orders: an arteriole runs straight into a venule. |
| `velocity_mm_s` | `mm/s` | Flow outliers: the vessel's mean velocity. |
| `class` | `-` | Flow outliers: the vessel type it is compared within. |
| `class_median_mm_s` | `mm/s` | Flow outliers: the median velocity of that type (log scale). |
| `robust_z` | `dimensionless` | Flow outliers: how many robust SDs (1.4826 MAD of log10 velocity) from it. |
| `role` | `-` | Terminal nodes and boundary issues: inlet, outlet, arteriole/venule boundary, open end. |
| `other_roles` | `-` | Other roles the same node has (;-separated). |
| `leaves_image` | `flag` | Terminal nodes: the dead end leaves the image. |
| `dead_end_kind` | `-` | The node's dead-end kind (graph.dead_ends). |
| `depth_um` | `um` | How far inside the image the node is (distance to the nearest face). |
| `issue` | `-` | Boundary issues: what is wrong with the node. |
| `kinds` | `-` | Dead ends: every test the end met (;-separated). |
| `tip_radius_um` | `um` | Dead ends: the lumen radius at the tip. |
| `junction_radius_um` | `um` | Dead ends: the lumen radius where its chain meets a junction. |
| `mask_run_um` | `um` | Dead ends: how far the segmentation runs on past the tip. |
| `ahead_um` | `um` | Dead ends: the distance to the vessel straight ahead. |
| `ahead_turn_deg` | `degrees` | Dead ends: the turn from the tip's heading onto the vessel ahead. |
| `ahead_background_um` | `um` | Broken vessels: the background the straight join to the vessel ahead crosses. |
| `ahead_junctions_away` | `count` | Dead ends: how many junctions away that vessel is in the graph. |
| `face_distance_um` | `um` | Dead ends: the tip's distance to the nearest image face. |
| `in_mask_fraction` | `fraction` | Dead ends: the share of the chain inside the segmentation. |
| `ahead_edge` | `-` | Dead ends: the vessel straight ahead, as (u, v, key). |
| `ahead_point_um` | `um` | Dead ends: the point on it the join would reach, (z, y, x). |
| `thick_lumen_um` | `um` | Dead ends at thick vessels: how far the tip lies past the thick vessel's lumen (negative inside it). |
| `thick_centreline_um` | `um` | Dead ends at thick vessels: the tip's distance to the thick vessel's centreline tree. |
| `thick_turn_deg` | `degrees` | Dead ends at thick vessels: the turn from the tip's heading onto that centreline. |
| `shorter_um` | `um` | Two in one lumen: the shorter vessel's length. |
| `lengths_um` | `um` | Two in one lumen, doubled stretches: both vessels' lengths (;-separated). |
| `cuts_off` | `-` | Two in one lumen: for each, whether deleting it cuts vessels off (;-separated). |
| `run_um` | `um` | Doubled stretches: how long the stretch beside the other vessel is. |
| `vessel_radius_um` | `um` | Doubled stretches: the mean lumen radius along the stretch. |
| `other_radius_um` | `um` | Doubled stretches: the mean lumen radius along the other vessel beside it. |
| `seen_from` | `-` | Doubled stretches: which vessel sees the other beside it, each judged by its own lumen radius: both, wider_only or narrower_only. |
| `dead_end` | `flag` | Doubled stretches: either vessel ends in a dead end. |
| `join` | `-` | Broken vessels: how it would be joined: ends (two dead ends to each other) or side (a dead end to the side of the vessel ahead). |
| `gap_um` | `um` | Broken vessels: the gap between the two tips, or from the tip to the vessel ahead. |
| `turn_deg` | `degrees` | Broken vessels: the larger turn from either end's heading onto the join (the tip's, for a side join). |
| `refused` | `-` | Broken vessels: why the build did not join two facing ends. |
| `volume_um3` | `um^3` | Uncovered mask: the region's volume. |
| `radius_um` | `um` | Uncovered mask: its widest inscribed radius. |
| `voxels` | `count` | Uncovered mask: its voxel count. |

## `batch_summary.csv`

Written to the batch's output folder. One row per image of a batch run.

| Column | Unit | Meaning |
|---|---|---|
| `stem` | `-` | The image's stem. |
| `status` | `-` | done, failed, skipped (done before), cancelled or not run. |
| `condition` | `-` | Its condition (A or B), if any. |
| `segmented` | `-` | Its segmented image (blank where ilastik segments its raw image). |
| `raw` | `-` | Its raw image. |
| `large_arteriole_mask` | `-` | Its large arteriole mask. |
| `large_venule_mask` | `-` | Its large venule mask. |
| `small_arteriole_mask` | `-` | Its small arteriole mask. |
| `small_venule_mask` | `-` | Its small venule mask. |
| `pericyte_mask` | `-` | Its pericyte mask, from the batch's folder of them. |
| `cell_mask` | `-` | Its cell mask, from the batch's folder of them. |
| `measurement_3d_vessel_mask` | `-` | Its vessel mask for the 3D distances, from the batch's folder of them. |
| `tissue_raw_tiff` | `-` | Its tissue image, from the batch's folder of them. |
| `capillary_block_mask` | `-` | Its mask of blocks, from the batch's folder of them. |
| `haemorun` | `-` | The .haemorun it saved. |
| `nodes` | `count` | Nodes in its network. |
| `vessels` | `vessels` | Vessels in its network. |
| `wall_time_s` | `s` | Its wall time. |
| `peak_rss_bytes` | `bytes` | Its peak memory: the largest of its stages' peaks in its run manifest (not the process's, which in a batch includes every earlier image). |
| `error` | `-` | The error, if it failed. |

## `{condition}.haemobatch`

Written to the batch's output folder. A synthetic template per condition (batch.haemobatch with none): gzip JSON of each finished image's templates, by its statistics report and by its graph, with the runs left out and why and the batch's batch_settings.yaml. A synthetic run loads it as its template, the runs averaged (PLAN_batch_templates.md).

NetworkTemplate fields; see synthetic/batch_template.py.

## `batch_settings.yaml`

Written to the batch's output folder. The settings the batch ran with, and its folders and conditions as comments.

## `batch_statistics_by_image.csv`

Written to the batch's output folder. Every image's statistics side by side: Section, Metric, Unit, then one column per image (its value, in the row's unit). The first rows give each image's condition (and subject, when paired).

Metrics as in docs/statistics_glossary.md.

## `batch_statistics_summary.csv`

Written to the batch's output folder. Per metric and group (each condition, All images): N, Mean, SD, SEM, Median, Q1, Q3, IQR, Min, Max, Range (in the row's Unit) and CV (dimensionless).

Metrics as in docs/statistics_glossary.md.

## `batch_condition_comparison.csv`

Written to the batch's output folder. Two conditions tested per metric: each condition's N, median, mean and range (in the row's Unit), the difference (Unit) and ratio of medians, Mann-Whitney U or Wilcoxon (paired) with P, Benjamini-Hochberg Q and the smallest possible P, rank-biserial r, Welch's or the paired t with P, and Hedges g or Cohen dz (all dimensionless).

Metrics as in docs/statistics_glossary.md.

## `batch_condition_pairs.csv`

Written to the batch's output folder. A paired comparison: each subject's pair of images (or the median/mean of its images), and every image left out with why.

| Column | Unit | Meaning |
|---|---|---|
| `Subject` | `-` | The subject the images belong to. |
| `{A} images` | `-` | Condition A's image(s) in the pair. |
| `{B} images` | `-` | Condition B's. |
| `{A} value` | `-` | What A's value is: the one image, or the median/mean of N images. |
| `{B} value` | `-` | The same for B. |
| `Left out because` | `-` | Why an image was not paired. |

## `*.png, *.html`

Written to the plot folder (plot_dir). Figures: the skeleton, the graph (final_graph.png, final_graph_3d.html, geometry_with_branch_orders.png, ...), the node degree distribution, the graph after each building step (graph_after_<step>.png, with step artifacts on) and a perturbation's plots and sweep curves in its folder. Axes carry their units.
