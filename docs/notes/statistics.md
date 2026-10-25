# Statistics and network analyses

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## Network analyses

**Network analyses** (`statistics/inlet_outlet_routes.py`, `transit_time.py`,
`flux_heterogeneity.py`, `territories.py`,
`capillary_bed_joins.py`, `current_flow.py`, `occlusion.py`, `loops.py`, `strahler.py`,
`spectral.py`) — each a
`statistics_*` toggle in the schema's *Connectivity/Network Analysis* section, run inside
`compute_comprehensive_vessel_statistics`. They add rows to the statistics CSV, and most also
write per-vessel edge attributes that the vessels layer offers as colour options (declared in
`gui/results.py`'s `OPTIONAL_EDGE_COLUMNS`). An analysis that cannot run (no inlets/outlets, no
solved flow) writes nothing, so its colour option stays absent. `statistics_route_weighting`
picks the conductance model (topology / length / resistance) the Laplacian-based ones share
through `_flow_system.py`. `capillary_bed_joins` splits the capillaries into one bed per feed
(the nodes the B orders count from, `graph.branch_order.capillary_start_nodes`), takes each
bed's join order as the lowest B order on a watershed vessel with another bed, censors a bed
that never joins (leaves the image / drains to a venule / ends blind), and reports a
Kaplan-Meier median plus the `remove_short_loops_under_um` the graph was built with (that rule
removes the earliest joins); its per-bed table goes to `<stem>_capillary_bed_joins.csv` beside
the statistics CSV, for between-dataset tests (log-rank, a reseeded-feed null) outside the package.
`flux_heterogeneity` (2026-10-19, for the stall work of `PLAN_stall_artifacts.md`): over the
vessels *carrying* flow -- the solve reached them (`vessel_classes.carries_flow`) and their
flow is above `CARRYING_FLOOR_FRACTION` (1e-9) of the network's largest, the block
comparison's own floor, imported from there -- the CV (sample SD / mean) of flow and of mean
velocity, for the capillaries and for every vessel, and the flowing share of the network by
count, length and volume (a cylinder per vessel), with rows by vessel type; it writes
`vessel_velocity_um_s` on each carrying vessel. **The red cells** (2026-10-20,
`haemodynamics/red_cells.py`): on each carrying vessel it also writes `tube_haematocrit`,
`red_cell_velocity_um_s`, `red_cell_flux_per_s` and `red_cell_linear_density_per_mm`,
predicted from the flow and the discharge haematocrit -- the vessel's own
`discharge_haematocrit` under `distributed_iterative`, else the run's `G.graph["haematocrit"]`
-- by the Fahraeus relation of Pries, Secomb, Gaehtgens & Gross 1990 (`H_T/H_D = H_D + (1 -
H_D)(1 + 1.7 e^{-0.415 D} - 0.6 e^{-0.011 D})`, held at 1 below 2.58 um where the fit runs
above it) and red-cell conservation (flux `Q H_D / V`, cell velocity `v_mean H_D / H_T`,
spacing `H_T A / V`), with `red_cell_volume_fl` (Haemodynamics tab, under the haematocrit;
50 fL, a mouse's; rat ~57, human ~90) the mean cell volume `V`; the report gets the same mean /
SD / CV rows for the flux and the red-cell velocity, the mean tube haematocrit and spacing, by
group and by vessel type, `Red Cell Volume (fL)` and `Haematocrit Source`, and the block
comparison the four CVs `_baseline` / `_blocked`. With one haematocrit the flux CV *is* the
flow CV. Predictions, never measurements: they carry every error of the flow (the assumed
pressure drop, the diameters, the viscosity law, what the segmentation missed; `flow.10a` is
still unticked) and the haematocrit model's; cited through `_used_red_cell_transport`.
`compare_block_to_baseline` reports the same
rows `_baseline` / `_blocked`, and counts the blocked network's *stopped* vessels
(`stopped_vessels`, `stopped_vessel_length_um`): the blocked ones and, through
`inlet_to_outlet_vessels`, every vessel fed only through one
(`capillary_block.stopped_vessels`), whose leak -- a millionth of its flow at the default
factor -- is above the floor. After a capillary block's re-solve, the first draw's and every
replicate's, the stage sets `flow_abs`, `flow_signed` and `flow_abs_log10` to exactly 0 on
the perturbation graph's stopped vessels (`capillary_block.zero_stopped_vessel_flows`,
2026-10-19; the pressures stay, since the drop across a block is real), so the perturbation's
own statistics CSV, its edges CSV, its vessels layer and `flow_change_vs_baseline` (0 on a
stopped vessel) agree with the comparison -- which zeroes the same on its copies, for a blocked
network solved by hand. `vessel_classes.carries_flow` reads a flow of exactly 0 as none, so the
Vessel Classes section's "Vessels Carrying Flow" agrees too. `compute_intercapillary_distance`
adds the same spacing over the carrying vessels alone (`... Between Flowing Vessels`) and the
ratio of the two medians.

## Exact betweenness

**Exact betweenness** (`statistics/exact_betweenness.py`, 2026-10-20) — every exact
`betweenness_centrality` in `network_measures.py` goes through `exact_betweenness_centrality`:
full mode's per-node betweenness (the whole network and `_statistics_connected.csv`'s again)
and the summaries' exact branches. From `PARALLEL_BETWEENNESS_MIN_NODES` (2,000) nodes the
per-source searches run in `python -c` workers (networkx's own `_single_source_*` and
`_accumulate_*`, one OpenMP/BLAS thread, `default_worker_count()` capped by free memory), and
this process adds each source's increments in networkx's node order and calls its `_rescale`:
the same floats in the same order, so the same bits. Below the threshold, with one worker, or
when a worker fails, networkx runs it here. Never sampling (`k`) or another library.
The unweighted, undirected pass without endpoints -- the one the statistics make -- is not
networkx's loop but the batched pass (`_BATCHED_PASS`, 2026-10-20): the same search for a
batch of sources at once with numpy (the workers import numpy only), every addition in
networkx's order (`np.add.at` adds in index order, the sorts are stable), so each source's
increments are networkx's bits; it also runs in this process below the threshold, with one
worker, or when the workers fail. Weighted, endpoint and directed passes keep networkx's loop.
One machine state, 29_2_f contra (15,843 nodes): networkx's loop in 8 workers 164 s, the
batched pass here 125 s, in 8 workers 19 s; 29_1_f contra (21,944): 303, 213 and 26 s.
Harvey's AIA batch graph `29_1_b_contra_FITC` (18,253 nodes, 8 workers, a busy machine):
414 s against networkx's ~2,950 s (timed on 300 sources, scaled); a 4,000-node piece of it
122.5 -> 44.7 s, every value's bits equal. `MAX_DEFAULT_WORKERS` (8) is the ceiling.

## statistics/three_dim_distances.py

**`statistics/three_dim_distances.py`** — cell-to-vessel distances (`measurement_3d_to_cell_mask`,
run last in `export_results`). Each object in the cell mask gets its gap to the vessels, and its
centroid's: the distance between the two as voxelised volumes, each voxel the box it fills, so a
cell touching a vessel is 0 um from it (voxel centre to centre it read one voxel step, the z step
along z), with `vessel_contact` saying `separate` / `touching` / `overlapping` / `inside`. The
vessels are `measurement_3d_vessel_mask_path` if set, else the run's own segmentation
(`network.volume.image`), and only without either the graph's tubes of each edge's `diameter_um`
-- until 2026-10-15 the default was the graph's centrelines, a vessel radius too far. The vessel
named is the one whose lumen the nearest vessel point lies in (`_VesselIdentity`: least
`|p - c| - r(c)` over centreline voxels, `r` the lumen radius there), never a bridge -- the
nearest centreline named a capillary running past a wide vessel's wall. Each mask's voxel size is
resolved as the vessel masks' are (none of its own takes the run's), and preflight checks shape
and voxel size from the TIFF headers (`checks.check_cell_mask_measurement`), since a mismatch
used to stop the run at its end. The vessel mask of its own is optional: `must_exist` with a
`placeholder`, which `parsers.check_settings` reads as unset falls back, set must be there
(`Setting.optional_path`; `must_exist` alone made preflight demand it on every run measuring
to the segmentation, and without it the panel opened a save dialogue for it). The same
rule covers `tissue_raw_tiff_path`. `measurement_3d_reference_image_path` (it only checked a shape)
is retired, and so is `edt_mask_path` (2026-10-18): the EDT widths read the run's segmented
input (`stages.segmented_input_path`), since a box of their own carried another dataset's mask
into runs. Objects are read from their bounding boxes in both paths (in RAM, a whole-volume
`labels == id` per object took 16.5 s for 397 cells on 4 M voxels; now 0.6 s).

## statistics/tissue_volume.py

**`statistics/tissue_volume.py`** — `measure_tissue_volume` (`measure_tissue_volume` on the
Additional measurements tab, off by default): the tissue in the raw image (`tissue_raw_tiff_path`,
else the run's raw image, `raw_image_path`) as a closed surface mesh, so vessel density can divide by tissue rather
than by the box the network spans, which counts the image's empty space. Block-averaged onto a
~4 µm working grid with the vessel voxels left out of the threshold (else Otsu splits vessels
from everything else), split by Otsu/Li/triangle or a manual intensity, and the surface placed
**halfway between the two classes' median levels** — where a blurred edge is, whatever the
blur. Vessel voxels then read as the tissue's own level in the normalised-convolution smoothing
(leaving them out pulls the edge inward), and marching cubes runs through the smoothed field,
closed on the image faces. `pipeline/stages.py`'s `tissue_measurement` measures it once per
network (cached on `VesselNetwork.tissue`, keyed on the settings, the raw file and the
segmentation) so the perturbations' statistics divide by the same tissue as the baseline's;
`export_results` writes `{stem}_tissue_volume.csv` and `{stem}_tissue_surface.vtp` and hands
it to the viewer on `Solution.tissue` (`gui/results.py`'s `TISSUE_SURFACE` Surface layer).
Smoothing rounds convex tissue edges by about `tissue_smoothing_sigma_um`; a stack with no
empty space at all logs a warning, since an automatic split then cuts the tissue itself.

## statistics/layout.py (2026-10-25)

Three measures asked for after grid D1d, when E14.5 sat outside every drawn network on them
(`docs/notes/synthetic.md`): **junction shape** (the three angles at each junction of three
vessels, each read 8 um along its centreline: E14.5's are the same read at 4, 8 or 15 um and with
or without a dead-end stub at the junction), **tissue coverage** (a 10 um probe grid through the
image box, at most 200,000 seeded random probes in a larger one, to the nearest centreline sampled
every 2 um) and **vessel columns** (chords of at least 10 um within 30 degrees of z; their spread
by the Clark-Evans ratio of their midpoints in the image plane, with no edge correction). On
E14.5's MCA run: junction smallest angle median 97 degrees, 1.2% acute, angles summing to 356;
0.5% of the box beyond 50 um of a vessel, 90th percentile 34.7 um; columns 21% of the length,
Clark-Evans 0.90. The three cost 0.6 s together on it. The vessel classes gained their 90th
percentile tortuosity beside the mean and median.
