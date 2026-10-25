# HaemoLynx — repository guide for AI assistants

HaemoLynx turns 3D microvascular microscopy into **NetworkX graphs** with haemodynamic edge weights, VTK exports, and network statistics. The pipeline accepts **already-segmented** binary masks (TIFF/H5) or can **run segmentation in-repo via ilastik** (headless) when configured.

**Segmentation is in scope.** HaemoLynx integrates ilastik for pixel classification inference on the main vessel volume and on optional large/small arteriole–venule channels. What stays **outside** this repo is **training** ilastik projects (`.ilp` classifiers): users train those manually in the [ilastik](https://www.ilastik.org/) GUI, then point the pipeline at the exported project and raw/unsegmented images.

---

## Repository layout

```
haemolynx/
├── src/haemolynx/          # Installable package (setuptools, pythonpath = src in pytest)
│   ├── geometry.py         # Polyline arc length, shared by haemodynamics and visualization
│   ├── io/                 # load.py (TIFF/H5 -> skeleton), load_2d.py (a 2D image as a
│   │                       #   z=1 volume), crop.py (crop on import: one box of voxels cut
│   │                       #   from the image and every volume read alongside it, the box
│   │                       #   kept on the graph), downsample.py (downsample on import: the
│   │                       #   segmented image read onto a coarser grid after the crop, every
│   │                       #   mask reduced by the same blocks and the raw image brought onto
│   │                       #   that grid; the record kept on the graph), ilastik.py, voxel_validation.py, axis_order.py,
│   │                       #   raw_volume_cache.py (reuse a memmapped raw volume across runs),
│   │                       #   vessel_masks.py (mask loading/validation),
│   │                       #   vessel_mask_discovery.py (the masks beside the input image,
│   │                       #   found by file name), batch_matching.py (a batch run's
│   │                       #   images matched to their raw images, masks and other files by name)
│   ├── preprocessing/      # skeleton.py — skeletonize, bridge, clean, bundle refinement;
│   │                       #   thick_vessels.py (thickness-gated skeletonisation of fat
│   │                       #   vessels) + thick_vessel_braid_guard.py + thick_vessel_region_cache.py
│   │                       #   (the region, kept beside the skeleton between runs) +
│   │                       #   thick_vessel_tree.py (the large-stack centreline tree, a low-RAM
│   │                       #   toggle: no per-voxel graph, no arm cap); segmentation_cleanup.py
│   │                       #   (pre-skeleton mask cleanup), segmentation_quality.py and
│   │                       #   segmentation_raw_comparison.py ("Check segmented image");
│   │                       #   raw_stain.py (whether the raw image stains the lumen or
│   │                       #   the wall round it: raw_image_stain's auto);
│   │                       #   skeleton_consistency.py (skeleton vs mask), memmap_support.py
│   │                       #   (disk-backed volumes, the low-RAM option), pointwise_distance.py,
│   │                       #   blockwise_edt.py (a distance transform's exact answers, a padded
│   │                       #   block at a time) and slab_labels.py (components labelled a slab
│   │                       #   of planes at a time), the low-RAM option's thick-vessel steps,
│   │                       #   bridge_mask_support.py (whether a bridge path stays in the mask
│   │                       #   and off a vessel already in the same lumen)
│   ├── graph/              # assemble.py (orchestrator) + build, reconnect, optimise, degree2,
│   │                       #   prune, collapse (+ direction_aware_collapse, persistence_collapse,
│   │                       #   cartwheel_guard), branch_order, boundaries, boundary_node_fallback,
│   │                       #   large_vessels, large_vessel_network, cut_at_large_vessel_volumes,
│   │                       #   confidence_vessel_assignment, mask_component_volume,
│   │                       #   mask_continuity (small-mask bridging), small_vessel_redefinition
│   │                       #   (tangential / sandwiched small-mask relabelling),
│   │                       #   diagnostics, validate, _helpers, _platform,
│   │                       #   smoothing.py (centreline smoothing — rewrites `length`),
│   │                       #   thick_vessel_junctions.py (IS_ZERO_RESISTANCE bridges),
│   │                       #   thick_vessel_bridge_shapes.py (what shape each bridge came out),
│   │                       #   communities.py (vascular communities), edit.py (the panel's
│   │                       #   graph edits), post_processing.py (4+ junctions: delete
│   │                       #   vessels / split with a connector; Add vessel's click-by-click
│   │                       #   A* traces through mask or raw data), terminal_node_assignment.py
│   │                       #   (terminal-node assignment), mask_recovery.py (segmented vessels
│   │                       #   the graph lost, traced and joined back through the mask),
│   │                       #   lumen_loops.py (loops lying inside one lumen: Lee rings, broken),
│   │                       #   facing_ends.py (two dead ends heading for each other across a
│   │                       #   gap in the segmentation, joined), network_review.py (what the
│   │                       #   Post processing tab's review lists find, one finder per list),
│   │                       #   dead_ends.py (what each dead end is: the one definition),
│   │                       #   dead_end_extension.py (dead ends drawn on to the face, traced
│   │                       #   on through the mask, joined to the vessel ahead; opt-in),
│   │                       #   skeleton_retrace.py (skeleton the clean-up took round a wide
│   │                       #   hole, drawn back after it)
│   ├── haemodynamics/      # poiseuille, viscosity (the laws), resistance, apply,
│   │                       #   fwhm_diameter.py (FWHM diameters), raw_section.py (the raw
│   │                       #   cross-section fitted where FWHM fails, opt-in), edt_diameter.py
│   │                       #   (mask diameters), endothelial.py (internal diameter inside an
│   │                       #   endothelial stain's wall, FWHM's alternative), sections.py (the
│   │                       #   cross-section sampler/models/worker pool both share),
│   │                       #   fwhm_decoys.py (FWHM where there is no vessel), fwhm_planted.py
│   │                       #   (vessels of known width planted in the raw image, which the
│   │                       #   FWHM optimiser scores accuracy by), endothelial_planted.py
│   │                       #   (the same for the ring fit: hollow vessels of known lumen),
│   │                       #   haematocrit_distribution.py (Pries-Secomb phase
│   │                       #   separation), red_cells.py (tube haematocrit by the Fahraeus
│   │                       #   relation, red-cell velocity, flux and spacing, read off the
│   │                       #   solve), constriction (the one constriction model) + its
│   │                       #   site-choosing strategies: probability, pericyte_mask;
│   │                       #   constriction_strategy (which strategy a run uses),
│   │                       #   pericyte_comparison; perturbations.py (typed settings overrides
│   │                       #   to re-solve for) and what they run: arteriole, capillary,
│   │                       #   capillary_block (+ capillary_block_mask: the vessels a mask's
│   │                       #   blobs sit in), pericyte_sweep, pericyte_geometry_sweep,
│   │                       #   sweep_flows (per-grid-point flows for the viewer);
│   │                       #   open_ends.py (the boundary condition on vessels leaving
│   │                       #   the image, open_end_model)
│   ├── statistics/         # comprehensive.py (compute_comprehensive_vessel_statistics,
│   │                       #   the orchestrator; compute_vessel_statistics_by_measure, the
│   │                       #   same rows a section per measure), topology.py, shape.py, bifurcation.py
│   │                       #   (junction morphometry + per-branch-order aggregation, which
│   │                       #   share the parent/daughter edge classification),
│   │                       #   network_measures.py (betweenness + communities),
│   │                       #   exact_betweenness.py (exact betweenness in worker processes,
│   │                       #   a batch of sources at once, networkx's to the bit),
│   │                       #   network analyses: inlet_outlet_routes (bottlenecks, shunts),
│   │                       #   transit_time, flux_heterogeneity (the CV of flow and velocity
│   │                       #   over the vessels carrying flow, the flowing volume; before and
│   │                       #   after in the block comparison), territories, capillary_bed_joins (the B order
│   │                       #   each feed's capillary bed joins another's, Kaplan-Meier),
│   │                       #   current_flow, occlusion, loops,
│   │                       #   strahler, spectral, over _route_graph.py / _flow_system.py;
│   │                       #   csv_export.py, _sampling.py (one shared seed constant),
│   │                       #   three_dim_distances.py (cell-to-vessel distances),
│   │                       #   tissue_volume.py (the tissue's volume and surface mesh
│   │                       #   from the raw image, what vessel density divides by),
│   │                       #   dead_ends.py (the statistics CSV's Dead Ends section),
│   │                       #   vessel_classes.py (its Vessel Classes section: the same
│   │                       #   measures per class -- large/small arterioles, capillaries,
│   │                       #   small/large venules, or a global run's large/small vessels),
│   │                       #   batch_comparison.py (a batch's statistics side by side:
│   │                       #   each image, each condition's summary, two conditions tested),
│   │                       #   glossary.py (every statistic's definition and unit: the CSVs'
│   │                       #   Unit/Notes, docs/statistics_glossary.md; a new statistic needs a term)
│   ├── optimisation/       # "Optimise settings": search.py (segmentation cleanup,
│   │                       #   Skeletonise and Graph settings), fwhm_search.py (FWHM
│   │                       #   settings), endothelial_search.py (the ring fit's blur, wall
│   │                       #   and gate), candidates/metrics (+ fwhm_ variants), report,
│   │                       #   progress, trial_cache (a trial asked for again is not rerun),
│   │                       #   scorecard (the starting settings against the chosen ones)
│   ├── gui/                # napari plugin: form.py (schema -> form rows, pure),
│   │                       #   tabs.py (one tab per stage), layout.py (each tab's
│   │                       #   boxes and nested Advanced buttons, pure),
│   │                       #   progress.py (what the
│   │                       #   progress bars read, pure), results.py (what each
│   │                       #   stage puts in the viewer, pure), layers.py (an open
│   │                       #   layer -> run settings, pure), boundary_picking.py
│   │                       #   (boundary settings <-> napari Points/Shapes, pure),
│   │                       #   view_snap.py (XY/XZ/YZ plane -> dims order / camera
│   │                       #   directions, pure), canvas_illumination.py (the view
│   │                       #   window's surface lamp), layer_sets.py (baseline/perturbation
│   │                       #   network swap for the view panel's "Showing" menu, pure),
│   │                       #   statistics_view.py (the left panel's Statistics tab: a
│   │                       #   button per measure the last run's statistics found, pure),
│   │                       #   perturbation_editing.py, graph_editor.py
│   │                       #   + graph_click.py (the Edit window), post_processing.py
│   │                       #   (what the "7. Post processing" tab draws, pure), branch_hover.py,
│   │                       #   vessel_tubes.py, run_state.py, run_log.py + log_view.py,
│   │                       #   run_snapshot.py (.haemorun save/load), stage_checkpoints.py
│   │                       #   (re-run from a tab), optimise_progress.py, optimise_review.py
│   │                       #   (an optimiser's proposal before Apply, pure), chrome_tooltips.py
│   │                       #   (hover text for non-setting controls), stage_help.py
│   │                       #   (one help page per tab; the text is human-written),
│   │                       #   help_view.py (the ? window: directory and text),
│   │                       #   vessel_mask_search.py (when the panel looks for vessel
│   │                       #   masks and which rows it fills, pure),
│   │                       #   template_basis.py (Segmentation source set to a picked
│   │                       #   synthetic template's diameter basis, pure),
│   │                       #   batch_run.py (a batch: one run and .haemorun per image,
│   │                       #   no Qt; `python -m haemolynx.gui.batch_run`) + batch_run_cli.py
│   │                       #   (its command line: the one gui module allowed to print),
│   │                       #   _widget.py (the panel), napari.yaml (npe2 manifest)
│   ├── visualization/      # plot.py, vtk_io.py, pipeline_artifacts.py,
│   │                       #   geometry.py (an edge -> a drawable polyline),
│   │                       #   secomb_io.py (a solved network as Secomb's Network.dat,
│   │                       #   for NetFlow and FlowEstimate, and their output read back),
│   │                       #   dilation_curves.py, perturbation_plots.py, flow_direction.py,
│   │                       #   large_vessel_assignment.py, _helpers.py
│   ├── parsers/            # schema.py, config.py, cli.py, checks.py — the settings machinery
│   ├── synthetic/          # network_source=synthetic: template.py (what a run's .haemorun
│   │                       #   says its network is like, by its statistics or its graph),
│   │                       #   generate.py + trees.py + bed.py + polylines.py (a new network
│   │                       #   drawn to it), compare.py (how close it came, per characteristic)
│   └── pipeline/           # A package, not a module: schema.py (the pipeline's 450 settings),
│                           #   settings.py, checks.py (preflight), stages.py (one
│                           #   function per stage + run_pipeline_stages), progress.py
│                           #   (the ordered STAGES + the progress callback), citations.py
│                           #   (what a run used, and how to cite it), provenance.py (the
│                           #   run manifest beside every run's outputs), naming.py (the
│                           #   one stem a run's outputs carry: run_stem), batch.py (one
│                           #   image's settings in a batch, and the batch checks),
│                           #   export_dictionary.py (every output file's columns, arrays and
│                           #   units: docs/export_dictionary.md; a new output column needs an
│                           #   entry), presets.py (the panel's Preset menu: named sets of
│                           #   switches to start from, also offered by --preset); public:
│                           #   default_schema, write_default_config, preflight
├── examples/               # Runnable pipelines and settings (not the core library API surface)
│   ├── resistance_network_pipeline.py        # Main example: config + CLI over haemolynx.pipeline
│   ├── brain_network_pipeline.py             # Whole-brain run: pipeline + pericyte dilation sweep
│   ├── carotid_image_to_model.py             # Carotid dataset: the same pipeline, its own config
│   ├── simple_network_haemodynamics.py       # Hand-built graph: the haemodynamics API on its own
│   ├── *_schema.py / *_config.yaml           # Settings an example adds on top of
│   │                                         #   haemolynx.pipeline.schema; configs are generated
│   ├── pipeline_presets.py                   # The panel's presets + the examples' own, for --preset
│   └── regenerate_configs.py                 # Rewrite every *_config.yaml from its schema
├── tutorials/
│   ├── pipeline_tutorial.ipynb   # **Source of truth** for the step-by-step tutorial
│   ├── pipeline_tutorial.py       # Auto-generated from the notebook (do not edit by hand)
│   ├── tutorial_plots.py          # Inline plot helpers for the notebook
│   └── export_notebook.py         # nbconvert export used by integration tests
├── scripts/                # Developer tools, not part of the package
│   ├── compare_branches.py # Run the pipeline on this branch and on a reference
│   │                       #   ref, and diff the numbers (never runs in CI)
│   ├── branch_comparison/  # metrics/report (pure, unit-tested) + runner
│   ├── graph_artefacts.py  # Rebuild a run's graph step by step and measure its lumen
│   │                       #   artefacts and dead ends (graph_artefacts/: report,
│   │                       #   dead_end_census: by kind, place, depth, step, flow coverage)
│   ├── dead_end_review/    # The marked grids dead ends are calibrated on: G1-G3 (build_grids.py),
│   │                       #   G4 (build_g4.py: segmentation, dextran and claudin-5 side by side),
│   │                       #   DR1 (build_dr1.py: does the vessel go on past the tip?)
│   ├── open_end_sensitivity.py # One solved network under every open_end_model
│   ├── stats_cross_check.py # The statistics against skan, maskel and VesselVio on one mask
│   │                       #   (stats.8a), and a ledger of which HaemoLynx step makes each
│   │                       #   difference (stats_cross_check/; VesselVio, GPL, run as a program)
│   └── secomb_comparison.py # The flow solve against Secomb's NetFlow and FlowEstimate
│                           #   (flow.8a): builds them from clones kept outside the repo
├── tests/                  # Unit + integration tests (see Testing below)
│   ├── conftest.py         # Shared fixtures; matplotlib Agg backend
│   └── integration/        # Full-pipeline and tutorial tests (@pytest.mark.integration)
└── tests/data/             # Small TIFF/H5 fixtures for tests and tutorial input
```

> **Note:** there is an active repo-cleanup effort — see **“Cleanup plan”** at the end of this file.

---

## Segmentation (ilastik)

| In scope (this repo) | Out of scope (manual, upstream) |
|----------------------|----------------------------------|
| Headless ilastik inference via `io.run_ilastik_headless_segmentation` | Training / labelling in ilastik GUI |
| Wiring in `pipeline/stages.py` (`segment` for the main mask, `build_network` for the large/small vessel masks via `io.load_and_validate_vessel_masks`), driven by `use_ilastik_segmentation` and the large/small vessel ilastik flags | Choosing features, labels, or project hyperparameters |
| Loading ilastik-produced masks into the rest of the pipeline | Installing ilastik itself on the user machine |

**Three ilastik hooks** in the main pipeline (all optional):

1. **Main vessel mask** — `use_ilastik_segmentation=True` segments the run's raw image before skeletonization (`raw_image_path` and `raw_image_channel`, `ilastik_classifier_path`, `ilastik_executable`).
2. **Large arteriole/venule masks** — `use_ilastik_large_vessel_segmentation` (requires `use_large_vessel_masks=True`).
3. **Small arteriole/venule masks** — `use_ilastik_small_vessel_segmentation` (requires `use_small_vessel_masks_for_boundary_assignment=True`).

**The vessel masks are segmented from the run's raw image** (2026-10-24, Harvey;
`PLAN_batch_run.md`). Each classifier gets its own channels of it -- `ilastik_arteriole_channels`,
`ilastik_venule_channels` and the two `ilastik_small_*_channels`, counting from 0 like
`raw_image_channel` -- kept separate, or the whole file when unset.
- `io.ilastik.write_ilastik_channels` writes them, plane by plane, to
  `{ilastik_output_dir}/channels/{output stem}_input.tif` for as long as ilastik runs. It keeps
  the file's axes, its kind (ImageJ or OME) and its voxel size, and the file is deleted afterwards.
- The output's provenance record names the channels, so other channels segment again.
- Preflight checks them against the raw image (`checks.check_ilastik_vessel_mask_channels`).
- The separate input files these replace (`ilastik_unsegmented_*_image_path`) still load through
  `migrates`, warning when one was another file than the raw image.

Shared settings: `ilastik_output_dir`, `ilastik_output_suffix`. Every one of them is declared in `pipeline/schema.py`, with its prerequisites; `pipeline/checks.py` (`haemolynx.pipeline.preflight`) is what tells a user which required paths are missing before any work starts. Outputs are named `{image stem}_segmented` for the main image and `{image stem}_{large|small}_{arteriole|venule}_segmented` for the masks: the arteriole and venule classifiers are usually run on the one raw stack, and with one name the venule run overwrote the arteriole's (or, reusing existing output, read it back as its own).

**The vessel label is named, and reuse is by provenance** (2026-10-22, `PLAN_dale_review_fixes.md`
item 4, `io/ilastik.py`). Simple Segmentation numbers the labels 1 to N, so a two-label output is
`{1, 2}` whichever is the vessel; it used to be read with its minority as vessel (inside out where
vessels are the majority) and a one-class output as solid vessel. Each classifier now has its own
label setting -- `ilastik_vessel_label`, `ilastik_arteriole_vessel_label`,
`ilastik_venule_vessel_label`, `ilastik_small_arteriole_vessel_label`,
`ilastik_small_venule_vessel_label` -- a label name or number, **no default**: preflight
(`checks.check_ilastik_vessel_labels`) reads the project's `PixelClassification/LabelNames` and
refuses an unset or unknown label, listing the labels. `run_ilastik_headless_segmentation(vessel_label=)`
rewrites the output in place as 0/1 at that label (`binarise_ilastik_output`; a one-class output
raises and is left as written), so every reader of a segmented image reads it alike, and writes
`{output}.haemolynx.json` beside it: the project's SHA-256, the input's path, size and
modification time, the label, the output's size and time. `ilastik_reuse_existing_output` reuses
an output only when that record matches (`ilastik_output_is_current`); file times alone reused
outputs of a different project copied in with an older time. An output with no record (made
before this) is segmented again once. ilastik is run `--readonly` with `--output_format` from the
suffix (without it ilastik writes the project's saved format, HDF5 by default), and it names a
multipage TIFF `.tiff` whatever it is asked for, which is where the runner looks. Checked by hand
on Windows with ilastik 1.4.1rc2 and Harvey's claudin project (2026-10-22). The generic loader
still guesses the minority for a user-supplied mask of two labels, now with a warning once per
label set (`io.load._warn_minority_guess`).

Users may also supply **pre-segmented** masks only (no ilastik call) — typical for tutorial data and many tests.

**One raw image per run.** The Input tab's `raw_image_path`, `raw_image_channel` and
`raw_image_stain` are the only raw image a run reads: ilastik segments it (the vessel masks'
classifiers too, by their own channels), the Diameters stage
measures from it (`measure_diameters_from_raw_image`), Check segmented image and the optimisers
compare against it, and the tissue volume falls back to it. `raw_image_stain` is `auto`,
`plasma_column` (a lumen label such as dextran) or `endothelial` (the wall, such as claudin-5). Auto
is read once, by `preprocessing/raw_stain.py`'s `read_raw_stain` in `build_network`, and recorded as
`G.graph["raw_image_stain"]` for the Diameters stage. The stain picks the diameter method: FWHM for
a plasma column, the endothelial ring fit for a wall. A raw image that is set is read whatever
else is on, so preflight treats a missing one, or a composite without `raw_image_channel`, as an
error.
**The raw image is optional**: with none entered, every setting that reads it is out of effect
whatever its switch says (`stages.raw_image_is_set`, `ilastik_segments`,
`measures_diameters_from_raw_image`), and preflight warns rather than fails. Those settings declare
`requires=("raw_image_path",)`: a `requires` entry may name a path setting (set / unset,
`parsers.schema.PREREQUISITE_GATE_KINDS`). The panel **greys** a row whose own path gate is unset
rather than hiding it (`gui/form.py`'s `hidden_by`), and a `!gate` rule holds while the gate is out
of effect (`form.unmet_chain`, `Schema._prerequisite_in_effect`). The settings it replaced still
load, through `Setting.migrates` / `Schema.upgrade`. History: `docs/notes/inputs.md`.

---

## End-to-end pipeline (conceptual)

0. **Segmentation (optional)** — ilastik headless on raw TIFF/H5 → binary mask; or skip if input is already segmented  
1. **Load & skeletonize** — `io.load_and_skeletonize_3d_tif` / `_h5`; `preprocessing.preprocess_skeleton_for_graph`  
2. **Vessel masks (optional)** — `io.load_and_validate_vessel_masks` (large/small arteriole/venule; from disk or ilastik)  
3. **Graph build** — `graph.build_graph_from_skeleton` (thirteen topology steps in `graph/assemble.py`), then `graph.smooth_graph_centrelines`  
4. **Boundary & branch order** — manual volume/coordinates or mask-based assignment; `graph.assign_vessel_branch_orders` / hierarchical orders  
5. **Haemodynamics** — `haemodynamics.apply_poiseuille_haemodynamics`, conductance matrix, two-point resistance, flow solve (optionally iterating a distributed haematocrit, `haematocrit_model="distributed_iterative"`), with the boundary condition `open_end_model` puts on vessels leaving the image (`haemodynamics/open_ends.py`)  
6. **Post processing (hand edits)** — on the *solved* network: `post_process` brings vessels edited by hand in the panel (`graph/post_processing.py`) in line with the rest: lengths, mask tags and the boundary nodes they give where the edits touched, branch orders over the whole network, the kinds of the dead ends the edits left, diameters of the edited vessels only (by the run's own methods), zero-resistance bridges into thick vessels. It also records how each drawn vessel sits in the segmentation (`graph.record_drawn_vessel_support`: `drawn_mask_fraction`, `drawn_background_run_um`, `drawn_beside_vessel`), a diagnostic only: Harvey draws vessels where the segmentation is missing, so nothing is refused. Then `solve_post_processed` reruns the haemodynamics on the edited network. A no-op on an unedited network, which is not solved again. With `mid_run_postprocessing` on, a panel run pauses before it (`run_pipeline_stages(stop_after="solve")`) and Continue resumes at it. Details: `docs/notes/pipeline_and_post_processing.md`.  
7. **Perturbations (optional)** — `run_perturbations` re-solves copies of the solved network, once per configured entry in `perturbations` (see `haemodynamics/perturbations.py`)  
8. **Export & stats** — `visualization.graph_to_vtk`, the tissue volume from the raw image (`measure_tissue_volume`, optional), `statistics.compute_comprehensive_vessel_statistics` (with the selected network analyses), vascular communities

`pipeline/stages.py` runs these as ten stage functions: `segment`, `skeletonise`, `build_network`,
`assign_boundaries`, `assign_diameters`, `build_haemodynamic_model`, `solve`, `post_process`,
`run_perturbations`, `export_results`. The vessel masks (step 2) are loaded at the start of `build_network`, and the
statistics, network analyses and communities (step 8) all run inside `export_results`.

**A synthetic network instead of an image** (`network_source=synthetic`, `PLAN_synthetic_network.md`):
the Input tab's `Input` choice draws a network matched to a previous image run's `.haemorun`
(`synthetic_template_path`; `synthetic_template_source`: its saved statistics report, or
distributions off its graph). `segment` reads and checks the template (`synthetic.load_template`; a
template missing a needed statistic is refused, naming the `statistics_*` toggle), `skeletonise`
returns an empty `SkeletonisedVolume(synthetic=True)`, `build_network` draws the graph
(`synthetic.generate_network`), `assign_boundaries` takes the roles the generator marked, the drawn
widths are kept (`diameter_source="synthetic"`) on the template's diameter basis (each edge's
`diameter_basis` is `NetworkTemplate.diameter_basis`, the one most of the template's vessels were
read on; the panel sets `segmentation_source` to it when the template is picked, never on a config
or run load (`gui/template_basis.py`), and preflight warns when they differ), and everything from Diameters on runs
as for an image. Every image-only Input row requires `network_source=segmented_image`, every mask switch reads
as off, `raw_image_is_set` is False, and every output is named `{template stem}_synthetic`
(`pipeline/naming.py`'s `run_stem`, the one answer). Export writes `{stem}_template_comparison.csv`
and a `synthetic_comparison` report section. A synthetic run's `.haemorun` is refused as a template.

The production-style entry point is `examples/resistance_network_pipeline.py` (`image_to_model_pipeline`). The tutorial (`tutorials/pipeline_tutorial.ipynb`) documents segmentation as a prerequisite stage and runs on pre-segmented fixture data by default.

---

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows the venv's interpreter is `.venv\Scripts\python.exe`; run tests as
`.\.venv\Scripts\python.exe -m pytest ...` (a bare `python` may not be on `PATH`).

- **Python:** ≥ 3.9 for the library (CI matrix: 3.9, 3.10, 3.11, 3.12); the napari panel needs 3.11+
- **Dependencies:** `pyproject.toml` only — there is no `requirements.txt`. Extras: `dev`
  (pytest, nbconvert, nbformat, npe2), `notebook` (ipykernel, for running the tutorial
  interactively), `napari` (napari with PyQt6, magicgui, npe2 — the GUI panel) and
  `napari-plugin` (the same, taking the Qt binding from an existing napari installation). `pytest-qt` is deliberately in none of
  them: it aborts a run that has no Qt binding, so only the GUI CI job installs it.
- **Run package tests:** from repo root: `pytest -s` or `pytest`
- **Skip slow tests:** `pytest -m "not slow"`
- **Integration only:** `pytest -m integration`
- **GUI only:** `pytest -m gui` (builds real Qt widgets; needs napari and a display)
- **Headless plotting:** tests set `MPLBACKEND=Agg` in `tests/conftest.py`; CI sets `PYVISTA_OFF_SCREEN=true`

---

## Testing policy (required)

**Every new feature, bug fix, or behaviour change must include rigorous automated tests.** Do not merge logic into `src/` without corresponding coverage in `tests/`.

### What “rigorous” means here

1. **Unit tests** for pure logic (graph transforms, I/O edge cases, voxel validation, branch-order rules, mask overlap logic, haemodynamics math helpers). Use small synthetic fixtures from `tests/conftest.py` or add focused fixtures in the test file.
2. **Regression tests** when fixing a bug — add a test that fails on the old behaviour and passes on the fix.
3. **Integration tests** (`@pytest.mark.integration`, often `@pytest.mark.slow`) for multi-stage flows that touch real fixture data under `tests/data/`. Keep these minimal but representative.
4. **Assert real behaviour**, not implementation details — e.g. node/edge counts, file existence, numeric tolerances on resistance, shape alignment, raised `ValueError` for invalid configs.
5. **No tests that only assert “runs without error”** unless paired with concrete output checks (artifacts, invariants, or golden metrics).
6. **Mark appropriately** — `slow`, `integration`, `plotting`, `gui` per `pyproject.toml` markers.

### Where to put tests

| Change area | Typical test location |
|-------------|----------------------|
Most modules have a test file named after them (`gui/run_snapshot.py` → `tests/test_gui_run_snapshot.py`,
`haemodynamics/capillary_block.py` → `tests/test_capillary_block.py`); check for one before adding a new file.

| Change area | Typical test location |
|-------------|----------------------|
| `src/haemolynx/io/` (incl. ilastik) | `tests/test_io.py`, `tests/test_load_and_validate_vessel_masks.py`, `test_load_2d.py`, `test_axis_order.py`, `test_voxel_validation.py`, `test_raw_volume_cache.py`, `test_crop_on_import.py` (the box, and every volume a run reads cut to it), `test_downsample_on_import.py` (the two choices, the block reduction, a run on the reduced grid, every volume read beside the image placed on it), `test_global_vessel_masks.py` (one mask and no venule mask: loading, the unknown vessel classes, the stages, a whole run), `test_vessel_mask_discovery.py` (masks found by file name), `test_ilastik_vessel_mask_channels.py` (the vessel masks' classifiers given their channels of the raw image: the channel file, reuse, the loader, preflight, the retired input files), `test_batch_matching.py` (a batch's images, raw images and masks by name, own mask folders, ties, a batch of raw images, each image's other files) |
| `src/haemolynx/preprocessing/` | `tests/test_preprocessing.py`, `test_skeleton_bridging.py`, `test_bridge_mask_support.py`, `test_thick_vessel_skeletonisation.py`, `test_thick_vessel_tree.py` (the large-stack tree and the join's bounded last resort; digests of the in-RAM skeleton), `test_thick_vessel_braid_guard.py`, `test_segmentation_cleanup.py`, `test_segmentation_quality.py`, `test_segmentation_raw_comparison.py`, `test_memmap_*.py`, `test_low_ram_*.py` |
| `src/haemolynx/graph/` | `tests/test_graph.py`, `test_graph_assemble.py`, `tests/test_branch_order_hierarchy.py`, `test_centreline_smoothing.py`, `test_graph_communities.py`, `test_graph_edit.py`, `test_graph_thick_vessel_junctions.py` (the split, and one fixture per bridge shape of `thick_vessel_bridge_shapes`), `test_thick_bridge_review_grid.py` (the Phase 5 grid's cases), `test_small_vessel_redefinition.py`, `test_vessel_mask_minority_swap.py`, `test_mask_continuity.py`, `test_mask_recovery.py`, `test_lumen_loops.py`, `test_facing_ends.py`, `test_parallel_duplicates.py`, `test_lumen_artefacts.py` (the lumen-artefact report, on one synthetic case of each kind from `tests/lumen_artefact_fixtures.py`), `test_consolidate_lumen.py`, `test_network_review.py` (each review list's finder), `test_dead_ends.py` (the one dead-end definition, every caller agreeing, on one fixture per kind from `tests/dead_end_fixtures.py`), `test_dead_end_extension.py`, `test_skeleton_retrace.py` (square rings of tube, one side skeleton only, and a branch to join), boundary/assignment tests |
| `src/haemolynx/haemodynamics/` | `tests/test_hemodynamics.py`, `test_viscosity_laws.py`, `test_constriction.py`, `test_haematocrit_distribution.py`, `test_haemodynamics_automated_fwhm.py`, `test_haemodynamics_edt_diameter.py`, `test_raw_section_diameter.py`, `test_fwhm_decoys.py`, `test_fwhm_planted.py`, `test_endothelial_diameter.py`, `test_endothelial_planted.py` (planted vessels read at their lumen at several voxel sizes, walls and blurs), `test_diameter_benchmark.py` (slow: every lumen method's accuracy on known vessels), `test_open_ends.py` (every open-end model, every solver taking it), FWHM/pericyte integration tests |
| Perturbations and sweeps | `tests/test_perturbations.py` (entries, settings, preflight), `test_perturbation_stage.py` (running them), `test_perturbation_outputs.py` (files and layers), `test_pericyte_sweep.py`, `test_pericyte_geometry_sweep.py`, `test_capillary_scaling.py`, `test_arteriole_scaling.py`, `test_capillary_block.py`, `test_capillary_block_mask.py` (which vessels a mask's blobs block, the perturbation, its checks), `test_sweep_flow_layers.py` |
| `src/haemolynx/statistics/` | `tests/test_statistics.py`, `test_statistics_glossary.py` (every row of a run with every statistic on has a term, a definition and a unit; the units a run decides; older CSVs read back alike; the page up to date), `tests/test_three_dim_distances.py`, `test_network_analyses.py`, `test_capillary_bed_joins.py`, `test_inlet_outlet_routes.py`, `test_occlusion_and_current_flow.py`, `test_statistics_without_haemodynamics.py`, `test_exact_betweenness.py` (the parallel pass `==` networkx on every shape, weighting and scaling, in workers and here; the batched pass where path counts pass 2^53, at every batch size; the fallbacks of both worker programs), `test_tissue_volume.py` (the measurement, the density it feeds, its stage wiring and preflight), `test_stats_cross_check.py` (the cross-check against skan on the dense capillary bed's design, pinned), `test_vessel_class_statistics.py` (each class's rows against hand-computed values, both labelling schemes, the export), `test_batch_comparison.py` (conditions from file names, the CSVs read back, summaries, Mann-Whitney/Welch/Benjamini-Hochberg against hand-computed values, the three batch CSVs; paired: subject IDs, one-to-one matching by name, every reason an image is left out, median/mean per subject, Wilcoxon/paired t by hand, Harvey's AIA names) |
| `src/haemolynx/optimisation/` | `tests/test_optimisation_*.py`, `test_fwhm_optimisation_*.py`, `test_endothelial_optimisation_search.py` (a trial's widths `==` the run's; the search on a stand-in ring fit; slow: whole searches at two voxel sizes) |
| `src/haemolynx/visualization/` | `tests/test_visualization.py`, `tests/test_vtk_io.py`, `tests/test_visualization_geometry.py`, `test_pipeline_artifacts.py`, `test_secomb_io.py` |
| `src/haemolynx/gui/` | `tests/test_gui_<module>.py` for the pure modules (`test_gui_form.py`, `test_gui_tabs.py`, `test_gui_progress.py`, `test_gui_results.py`, `test_gui_layers.py`, `test_gui_boundary_picking.py`, `test_gui_view_snap.py`, `test_gui_layout.py`, `test_gui_statistics_view.py`, `test_gui_vessel_mask_search.py`, …); `test_gui_*_widget.py`, `test_gui_widget.py` and `test_gui_view_panel.py` build the real panel; `test_gui_tooltips.py` checks every setting has hover text, and holds the other controls' strings to one voice; `test_gui_hover_text_widget.py` scans the built panel, with and without a viewer, and every dock, window and layer control it opens for a control with none (`tests/hover_text_scan.py`, shared with the Batch run window's scan; the help window waits on Harvey's words); `test_batch_run.py` (the batch runner and its command line, one real two-image batch, one real ilastik batch with a stand-in ilastik), `test_gui_batch_run_widget.py` (the button, the "Batch run" rows, the window: eight rows in view after Check, every control's hover text scanned built; Run pipeline's hand-off), `test_gui_fwhm_optimise_widget.py` / `test_gui_endothelial_optimise_widget.py` (the Diameters tab's two optimisers), `test_gui_presets_widget.py` (the Preset menu: what it sets, what replaces it) |
| `src/haemolynx/pipeline/` | `tests/test_pipeline_schema_api.py`, `test_pipeline_progress.py`, `test_pipeline_invariants.py`, `test_segment_stage.py`, `test_skeletonise_stage.py`, `test_build_network_stage.py` (a build that leaves no vessels), `test_post_process_stage.py`, `test_citations.py`, `test_pipeline_provenance.py` (the run manifest), `test_export_dictionary.py` (every output's documented columns are its writer's, units in VTK field data and CSV names, the page up to date), `test_setting_units.py`, `test_example_configs.py`, `test_batch_settings.py` (one image's batch settings, the batch refusals, the other files a run reads, an ilastik batch's), `test_pipeline_presets.py` (what a preset may name, every setting it names read, usages in any order, `--preset`) |
| `src/haemolynx/parsers/` | `tests/test_parsers_schema.py`, `test_parsers_config.py`, `test_parsers_checks.py`, `test_prefixed_settings.py` (every `fwhm_` / `skeleton_` / `segmentation_cleanup_` setting a parameter of its function or read by name) |
| `src/haemolynx/synthetic/` | `tests/test_synthetic_template.py` (both sources, every refusal, JSON, scaling), `test_synthetic_generate.py` (the graph contract, determinism, the exact counts, the acceptance test against `tests/synthetic_fixtures.py`'s hand-set template, the polyline bender, the bed's arithmetic), `test_synthetic_compare.py`, `test_synthetic_stage.py` (each stage under `network_source=synthetic`, a whole run, preflight), `test_pipeline_naming.py` |
| Any subpackage's `__all__` | `tests/test_public_api.py` (star-imports every subpackage) |
| Packaging / the installed wheel | `tests/test_installed_package.py`, `test_packaging_metadata.py`, `test_napari_manifest.py` |
| Full pipeline / examples | `tests/integration/test_image_to_model_pipeline.py`, `test_nerve_pipeline.py`, `test_axis_order_pipeline.py`, `test_simple_network_haemodynamics.py`, `test_progress_reporting.py`, `test_step_artifacts_setting.py`, `test_dense_capillary_bed.py` (skeleton cleaning and graph building on `tests/dense_capillary_fixtures.py`'s synthetic capillary bed) |
| Tutorial notebook | `tests/integration/test_pipeline_tutorial.py` (exports notebook → `.py`, then runs) |

### Tutorial notebook workflow

- **Edit** `tutorials/pipeline_tutorial.ipynb` only.
- **Regenerate** `tutorials/pipeline_tutorial.py` via:  
  `pytest tests/integration/test_pipeline_tutorial.py`
- Do **not** hand-edit `pipeline_tutorial.py`; it is overwritten by the integration test.

### CI

GitHub Actions (`.github/workflows/pytest-pr.yml`) runs on pull requests, on pushes to `main` and
`Devel_GUI_improvements_HD`, and weekly (Mondays). A newer push to a branch or pull request cancels
the run still going for an older commit (`concurrency`), so only the newest commit is reported; a
push never cancels the weekly run. Six jobs:

- **`lint`** — `ruff check .`, rules `F` and `E9` only (`[tool.ruff]` in `pyproject.toml`). There is
  no formatter: `ruff format` would rewrite most of the codebase, which is Harvey's decision.
  `tutorials/pipeline_tutorial.py` and the notebook are excluded; `__init__.py` re-exports are
  exempt from `F401`.
- **`pytest`** — `pip install -e .[dev]` and `pytest` across a Python 3.9 / 3.10 / 3.11 / 3.12 matrix
  (`fail-fast: false`, so one version failing still reports the others). The 3.11 run measures
  line coverage of `src/haemolynx` without `gui/` (`[tool.coverage]`, no threshold) into the job
  summary and a `coverage-xml` artifact: 92.7% on the not-slow subset at `2e59fb68`.
- **`windows`** — `windows-latest`, Python 3.11, `pytest -m "not slow and not gui" -n auto`.
- **`installed-package`** — builds the wheel, installs it into an empty environment and runs
  `tests/installed_package_smoke.py` from outside the repository, then `tests/test_installed_package.py`.
- **`dependency-floors`** — the suite against the oldest supported versions (`constraints-floor.txt`).
- **`napari-gui`** — Python 3.11 with the `napari` extra and `pytest-qt`: `npe2 validate HaemoLynx`,
  then `xvfb-run -a pytest -m gui`.

The slow and GUI tests, and macOS, are not covered on Windows. Locally, `pytest-xdist` is in `dev`:
`pytest -m "not slow and not gui" -n 10` takes about 8 min on 20 cores (serial: 12-13 min); mind the
RAM, each worker holds its own volumes.

---

## Coding conventions

- **Library code** lives under `src/haemolynx/`. Keep `examples/` thin — orchestration, CLI, presets.
- **Minimal diffs** — match existing naming, types, and import style in the touched module.
- **Settings** — every setting is declared exactly once, in `pipeline/schema.py`, and the examples
  add their own on top of it (`examples/*_schema.py`). A new ilastik path or flag goes there, with
  its `requires`, so `pipeline/checks.py` can check it and the config file and the CLI both grow a
  line for free. A `requires` entry is `name` (a bool on), `!name` (off), `name=value` (a choice
  holding one value) or `name!=value` (holding any other); read them through
  `parsers.parse_prerequisite` rather than by hand. A choice value renamed, or a setting folded
  into another, keeps old configs and saved runs loading through the setting's `value_aliases` /
  `replaces` (read by `Schema.upgrade`) rather than breaking them. Mark a setting `advanced=True` unless most runs
  change it: the panel then shows it behind an Advanced button under the row it depends on (see
  `gui/layout.py`), and `tests/test_gui_layout.py` fails if it lands nowhere. A setting a stage
  reads must not require another stage's skip toggle (`do_skeletonize`, `do_graph_building`):
  with the skeleton or graph loaded, later stages still read it (the thick-vessel split, post
  processing's edited vessels), and the panel would hide it.
  `tests/test_pipeline_schema_api.py` reads every stage function's settings to check it.
- **Generated configs** — `examples/*_config.yaml` are written by `examples/regenerate_configs.py`
  from the schemas, never hand-edited: they carry each setting's help text, units and prerequisites
  as comments. Path values are serialised with `PurePath(...).as_posix()`, so regenerating on
  Windows reproduces the committed files byte for byte instead of rewriting every path with
  backslashes.
- **Input contract** — pipeline expects **binary vessel masks** at skeletonization time. Masks may come from pre-existing files or from **ilastik inference** in this repo; classifier training is always manual.
- **Axis order** — arrays are canonical `(z, y, x)`: axis 0 is the stack axis that overlays and
  projections look through. Loaders take `axis_order` (`"zyx"` default, any permutation of `xyz`)
  and transpose the input to canonical order, so **that setting is how a user picks which axis is z**
  (the `image_axis_order` setting). See `io/axis_order.py`. **`"auto"`** reads each file in the
  order its own labels give (`stated_axis_order`: OME/ImageJ TIFF metadata, an HDF5's `axistags`),
  and a file with none in the segmented image's order, else `zyx`. A run hands its loaders
  `io.axis_order_for_run(settings)` (`"auto"`, or `"auto:<order>"` carrying the image's order),
  never the raw setting, and each loader resolves it against its file (`axis_order_for_file`);
  anything that transposes without a file needs a permutation and refuses `"auto"`. A TIFF page
  is a Y×X plane, so a resliced stack labelled `ZYX` still needs its order chosen by hand.
- **Voxel sizes** — two orders, never mix them. Image metadata is physical **`(x, y, z)`**
  (`voxel_size_xyz`, `io.voxel_validation.resolve_voxel_size_xyz`); anything that scales array
  indices takes per-array-axis spacing in **`(z, y, x)`** (`voxel_size_zyx`). Convert once at the
  loader boundary with `io.voxel_size_zyx_from_xyz`. Passing `xyz` where `zyx` is expected silently
  swaps the z and x spacings — invisible for isotropic voxels, wrong for every real stack
  (see `tests/test_anisotropic_voxel_size.py`). A TIFF's size is read from its OME header's
  `PhysicalSizeX/Y/Z` first (units converted), then, axis by axis, from the resolution tags and
  ImageJ's `spacing` (`io/load.py`'s `_voxel_size_xyz_from_tiff`): tifffile writes one pixel per
  unit into the tags of a file given no resolution, so an OME-TIFF's tags say 1 µm whatever its
  voxels are. Masks and main image must align in shape and
  physical voxel units. **Drawing a run has the same split the other way round:** node `pos` and
  edge `voxels` are physical microns already, so a layer built from the graph takes `scale=(1,1,1)`,
  while `image`, `skeleton` and the masks are voxel-indexed and take `scale=voxel_size_zyx`
  (see `gui/results.py` and its registration test).
- **Boundary coordinates** — every `*_node_coordinates` and `*_node_volumes` setting is physical
  `(z, y, x)` microns, the same units as node `pos`. The `coordinates` method snaps to the
  *nearest* terminal, so a point in voxel indices never fails, it just selects the wrong node;
  `graph.BoundaryCoordinateWarning` reports a snap that went too far and names the voxel-index
  reading when that is what it looks like (`graph/boundaries.py`). The `node_ids` method instead
  takes exactly the nodes in `*_node_ids` -- any node, junctions included, since an
  arteriole/venule boundary usually is one -- and an ID the graph lacks stops the run: IDs name
  nodes of one build of the graph, and rebuilding it renumbers them. The Boundaries tab fills
  these lists by clicking nodes (`gui/boundary_picking.py`'s `toggle_node_id`).
- **Graph** — `nx.MultiGraph` with `pos` on nodes and `voxels` on edges, both in physical
  `(z, y, x)` microns; haemodynamics uses `branch_order` on edges.
- **Edge attributes & units** — `length` (µm), `resistance` (Pa·s/m³), `conductance` (m³/(Pa·s)).
  `length` is measured from `voxels`, and **centreline smoothing rewrites both** (see
  `graph/smoothing.py`), so it is the smoothed curve the haemodynamics and the exports agree on.
  `resistance` and `conductance` are always written together via
  `haemodynamics.poiseuille.set_edge_resistance`. **There is no `weight` attribute** — it used to
  mean physical length at build time and conductance after haemodynamics ran, so statistics read
  conductances back as microns. `graph.assert_no_forbidden_edge_attributes(G)` raises if it
  reappears; NetworkX algorithms must be passed an explicit `weight="length"` / `weight="resistance"`
  rather than relying on their `"weight"` default.
- **Open ends** — a dead end that leaves the image is open in the tissue; `open_end_model`
  chooses its boundary condition: `estimated` (**default**; Fry et al. 2012: pressures that keep
  the network near a target pressure and wall shear stress, so an open end may feed the network
  as well as drain it), `closed`, `terminal_resistance` (drains to the outlet pressure through k
  times its own vessel's resistance) or `outlet`. Like the viscosity law it moves every flow, so
  **flows are not comparable across open-end models**; perturbations may not override it and solve
  with the baseline's conditions (`G.graph["open_end_conditions"]`). See `haemodynamics/open_ends.py`
  below; the E14.5 comparison is in `docs/notes/diameters_and_haemodynamics.md`.
- **Viscosity model** — selectable, and the choice changes every resistance a run produces, so
  **resistances are not comparable across laws or across diameter bases**. The graph records both
  in `G.graph["viscosity_law"]` / `["diameter_basis"]`. All of it is in
  `haemodynamics/viscosity.py`, and every form is pinned to its published equations in
  `tests/test_viscosity_laws.py`.
  - What decides the answer is **which diameter was measured**: `plasma_column` (the channel the
    fluid occupies, what a plasma stain images) or `endothelial` (wall to wall; `anatomical` still
    loads, `normalise_diameter_basis`). The run's basis is **what the segmented vessels are**:
    `segmentation_source` on the Input tab, read everywhere through `viscosity.run_diameter_basis`.
    A diameter read from the raw image carries the basis of *that* measurement on its edge
    (`poiseuille.EDGE_DIAMETER_BASIS`, by `MEASUREMENT_DIAMETER_BASIS`: FWHM and the raw section
    `plasma_column`, the endothelial fit `endothelial`), and `edge_diameter_basis` is read wherever
    a vessel's viscosity is computed. A perturbation may override neither (`INCOMPARABLE_OVERRIDES`).
  - `viscosity_law="pries"` (**default**): the basis picks the form, `plasma_column` the *in vitro*
    law (Pries, Neuhaus & Gaehtgens 1992) and `endothelial` the 1994 *in vivo* law, whose
    `(D/(D−1.1))²` factors stand for the ~1.1 µm endothelial surface layer (applied to a
    plasma-column diameter it counts that layer twice). `pries_viscosity_form` (behind the Advanced
    button under the law) holds every vessel to one form (`in_vivo_1994`, `in_vitro_1992`,
    `in_vivo_2005`) or `auto`; it resolves through `viscosity.run_viscosity_law`, so the graph's
    `viscosity_law` names the form that ran. Plasma is 1.2 mPa·s, with no red-cell-size scaling
    (unlike Secomb's NetFlow).
  - `viscosity_law="capillary_power_law"` is the earlier law, kept for comparison (raises
    `haemodynamics.PlaceholderViscosityWarning` between 7 and 100 µm); `"constant"` is plasma
    everywhere, for separating geometry from viscosity.
  - The in vitro form was wrong until 2026-10-13 (it used the in vivo `μ₀.₄₅`): plasma-column runs
    before then overstate capillary resistance, 3.4× at 5 µm. The ratios between the laws are in
    `docs/notes/diameters_and_haemodynamics.md`.
- **Skeletonization** — use `skimage.morphology.skeletonize(..., method="lee")` via
  `preprocessing.skeletonize_volume`, not deprecated `skeletonize_3d`. Both loaders skeletonize
  through the one `io.load._skeletonize_loaded_volume` helper, which binarizes, skeletonizes **and
  fills holes**: the TIFF and H5 paths had drifted apart on that last step, so one volume saved in
  two formats produced two different graphs. Anything that adds a third loader calls the same helper.
- **Output units** — a column CSV carries each column's unit in its name (`_um`, `_pa`, `_m3_s`,
  `_pa_s_per_m3`: `pipeline/export_dictionary.UNIT_SUFFIXES`); a VTK array keeps its name and gets
  its unit in `visualization/vtk_io.ARRAY_UNITS`, written to the file's field data. A new output
  file, column or array needs its entry in `pipeline/export_dictionary.py`, and the page written
  again (`python -m haemolynx.pipeline.export_dictionary`).
- **Comments** — only for non-obvious domain logic; prefer self-explanatory code.

---

## Key modules (quick reference)

Each entry says what a module is for and the rules a change to it must keep. Why a rule or
default is what it is (the marked grids it was fitted on, benchmarks, E14.5 and patella numbers)
is in [`docs/notes/`](docs/notes/README.md), one file per area, named in each heading below.
**Keep entries here short:** a new measurement history goes in `docs/notes/` or the matching
`PLAN_*.md`, not in this file.

### Inputs and vessel masks (`docs/notes/vessel_masks_and_boundaries.md`, `downsample_and_crop.md`)

- **`io/axis_order.py`** — canonical `(z, y, x)` convention: `normalize_axis_order`, `apply_axis_order`
  (transposes an input volume to canonical order), `voxel_size_zyx_from_xyz` / `voxel_size_xyz_from_zyx`.
- **`io/ilastik.py`** — `run_ilastik_headless_segmentation` (subprocess call to user-installed ilastik + `.ilp` project).
- **`io/vessel_masks.py`** — mask **loading & validation**: `load_large_vessel_masks`,
  `load_and_validate_vessel_masks` (includes the ilastik path for large/small masks). A mask with
  no voxel spacing of its own (an ilastik export) takes the main image's under the `auto` policy.
  The masks loaded last are cached per role, packed one bit a voxel and keyed on the files and
  every cleanup setting (`_LOADED_MASKS`, `clear_vessel_mask_cache`); each read gets a fresh array.
  - **Global masks**: a role given its arteriole/global mask and no venule mask loads as one mask
    of every vessel of that size, `(mask, None, voxel size, None)`. `vessel_masks_are_global(settings)`
    is the one test. Once any role is global the run classes vessels by size alone: no inlets,
    outlets, Large tier or small boundaries from the masks, vessels tagged `graph.GLOBAL_MASK_CLASS`
    by overlap, and `Unknown_Large` / `Unknown_Small` / `Unknown_Cap` in place of branch orders
    (`graph.assign_unknown_vessel_classes`). Statistics class them `large_vessel` / `small_vessel`
    (`GLOBAL_MASK_CATEGORIES`), on neither side. Preflight: `check_global_vessel_masks`.
  - **Masks found by file name** (`io/vessel_mask_discovery.py`'s `find_vessel_masks`, driven by
    `gui/vessel_mask_search.py`): ticking a mask switch, or changing the image paths while one is
    on, fills the role's rows from files beside the images, read by whole words (`large`/`small`,
    arteriole/venule synonyms). Ties are named, never guessed; a row holding a file the user
    chose keeps its role; loading a config or `.haemorun` never searches; nothing searches
    outside the panel.
- **`graph/terminal_node_assignment.py`** — graph **terminal-node assignment** from masks:
  `select_terminal_nodes_from_large_vessel_masks`, `infer_boundary_nodes_from_small_vessel_masks`.
  A terminal the large-vessel cut made carries `CUT_FROM_LARGE_VESSEL` and every selector counts
  it as inside that mask. The progressive dilation (`small_vessel_mask_dilation_microns`) labels each edge at
  the first step that reaches it and re-judges unlabelled ones at every step.
- **`graph/large_vessel_network.py`** — the `mask_stump` method: inlets/outlets where a large mask
  is cut by an image face. A component under `large_vessel_stump_min_component_fraction` of the
  largest, or a contact wider than `STUMP_MAX_FOOTPRINT_CROSS_SECTIONS` (a vessel lying along the
  face), is no stump. Each stump snaps to the end of its own vessel at its face (`_own_face_end`),
  else to the nearest terminal in the mask with a `BoundaryCoordinateWarning`. Checked on E14.5
  only, not a marked grid.
- **`graph/boundary_node_fallback.py`** — `seed_nodes_without_mask_coverage`, read through
  `voxel_size_zyx` (edge voxels are microns, masks voxel-indexed). The large mask of the class
  counts as coverage when large vessels stay in the network, and
  `small_vessel_boundary_fallback_to_hop_distance` *adds* hop neighbours of uncovered seeds to the
  masks' boundary nodes; it never replaces the list.
- **Vessel-mask switches** (`graph/boundaries.py`'s `vessel_masks_in_use`,
  `inlets_outlets_from_vessel_masks`, `assigns_large_vessel_branch_orders`) — the one answer to
  whether a role's masks are used. A switch hidden under one the user unticked keeps its value, so
  mask loading, `assign_boundaries`, a resumed network, preflight, the citations and the panel's
  greying all ask these, **never the raw flags** (`tests/test_vessel_mask_switches.py`).
- **Branch-order tiers** (`graph/branch_order.py`) — the Large_Art/Large_Ven tier is the vessels
  tagged in a large mask (`graph.LARGE_MASK_VESSEL_TYPE`) that connect to an inlet/outlet through
  tagged vessels. Art/Ven start at the hand-off nodes where a vessel leaves that tier, and run on
  along vessels tagged in their own small mask (`graph.SMALL_MASK_VESSEL_TYPE`,
  `_run_on_along_small_mask`), never across a capillary. A capillary leaving a large vessel is B,
  counted from it. With large masks alone the tiers still run (`assign_large_tier_branch_orders`,
  mode `large_tier`). `capillary_start_nodes(..., graph=G)` adds the Large_Art/B and Large_Ven/B
  hand-offs to the feeds and drains. An empty tier or boundary list the run expected warns
  (`MissingLargeVesselTierWarning`, `MissingSmallVesselAssignmentWarning`). The small-mask tiers
  have only been measured on stand-in masks.
- **Boundary nodes after hand edits** — `post_process` tags the vessels an edit made by the same
  overlap rules (`tag_edges_in_large_vessel_masks`, `tag_edges_in_small_vessel_masks`) and
  re-judges boundary nodes only at the nodes the edits touched (`graph.touched_nodes`,
  `post_processing.TOUCHED`), through `_reassign_large_vessel_boundaries` /
  `_reassign_small_vessel_boundaries`. A list is never emptied. The masks reach a Regenerate through
  `PipelineResume`, and the lists post processing leaves are on its checkpoint.
- **Downsample on import** (`io/downsample.py`) — `segmented_image_downsample` and
  `raw_image_downsample` (`none`/`2x`/`4x`/`8x`/`16x`). The segmented image is reduced after the crop,
  binarised first, by block **majority**, with blocks per axis from the voxel size
  (`axis_downsample_factors`, also the optimiser's). The record is `G.graph["downsample_on_import"]`,
  and every volume read beside the image is placed on that grid by it: vessel masks by majority,
  pericyte/cell masks by max (`mask_volume_like_graph`), the raw image by mean
  (`raw_volume_like_graph`). **The raw image must lie on the segmentation's grid**: a finer raw
  factor is read at the segmented one, a coarser one spread back (`spread_onto_grid`), and preflight
  warns either way. The skeleton clean-up's voxel counts keep their physical size
  (`skeleton_voxel_settings_on_grid`). A saved skeleton at another factor is refused.
- **Crop on import** (`io/crop.py`) — the box and every volume read beside the image are cut to it,
  and the box is kept on the graph. Which value is vessel is decided from the *whole file*
  (`io.load.mask_binarization_rule`, applied by `crop_volume(binarise=)`), never from the box. A
  file holding one value is read explicitly, with a warning: 0 and 2 (ilastik background) are
  empty, any other value is vessel (`tests/test_crop_on_import.py`).

### Graph building (`docs/notes/graph_building.md`, `low_ram.md`)

- **`graph/assemble.py`** — `build_graph_from_skeleton`; optional `step_callback(G, label)` after
  each topology step, one per label in `STEP_LABELS` (thirteen of them). The last,
  `consolidate_lumen`, repeats until a round changes nothing (at most `lumen_cleanup_max_rounds`):
  degree-2 merging, the second of two edges through one lumen
  (`lumen_loops.remove_parallel_edges_in_lumen`, never one whose loss cuts vessels off), loops
  inside one lumen, short loops, doubled stretches (`graph/doubled_stretches.py`: rewired, never
  deleted whole, bridges never touched), dangling bridges, the stub prune. `lumen_cleanup` builds
  what it reads from the build's own settings; `keep_apart=` takes a caller's evidence beyond the
  mask, and `along_thick_centreline=` makes the rules keep the strand on a thick vessel's
  centreline tree unless the other is a vessel of its own beside it (`_vessel_beside`).
  `build_network` runs the clean-up again after smoothing and after the thick-vessel split:
  nothing is the last step. Rounds after the first re-judge only what changed
  (`diagnose_parallel_duplicates_in_lumen(memo=)`, `FeatureDistance(remember=)`) and must give the
  same graph as judging everything.
- **Mask-supported bridging** (`preprocessing/bridge_mask_support.py`, `bridge_require_mask_support`,
  on by default) — with the segmented mask, nothing that joins pieces of skeleton or graph
  (closing and `bridge_gaps`, bundle hub links, `connect_skeleton_components`, `graph/reconnect.py`'s
  `MaskBridges`) is drawn across background. A bridge follows a route through the mask
  (`route_through_mask`) crossing at most `bridge_max_background_gap_um` in one run, with
  `bridge_min_mask_fraction` of it in the mask, and never beside a vessel already in the same lumen
  (`MaskSupport.shadows`). The reconnects judge their turn, and the skeleton bridges their facing
  (`_SkeletonBridgeGate._turns_back`), on the path drawn, not the chord. Background is read in pieces
  that divide every axis's half spacing where a step within four times the lookups does
  (`_sample_step`), so a dropout reads exactly. A mask not the skeleton's shape raises (`check_mask_shape`). Graph bridges record `bridge_kind` and
  `bridge_background_um`. Skeleton thinned away entirely by closing is put back. The stub prunes
  judge a dead end as its whole chain to its junction and read the mask
  (`prune.prune_vascular_stubs`, `min_stub_length_radius_multiple`).
- **`graph/mask_recovery.py`** — `recover_uncovered_mask_vessels` (on by default; after the orphan
  reconnect): mask no centreline's lumen covers (`uncovered_mask_voxels`, the cross-section reach
  `CROSS_SECTION_REACH_RADII`), in pieces of at least `recovery_min_region_volume_um3`, is traced
  along the run's own skeleton and joined to the network through the mask. Never an island, never
  across background, never beside an existing vessel, never a loop inside one lumen
  (`lumen_loops.loop_inside_one_lumen`). `coverage_reach_radii=` and `two_ended=` serve
  `remove_interior_dead_ends`. Its worker processes, packed reads and indexes must leave the
  result identical, and so must the order edges come in: a voxel is judged by its 4 nearest
  samples and every sample tied with the 4th (a junction holds one per edge at one point).
  Direction-aware collapse likewise takes tied pairs and groups in node-position order.
- **`graph/facing_ends.py`** — `join_facing_dead_ends` (after mask recovery,
  `facing_dead_end_max_gap_um`, 0 off): two dead ends heading for each other within
  `FACING_MAX_TURN_DEG`, across one stretch of background only, are joined though the gap is more
  than a dropout.
- **`graph/lumen_loops.py`** — `remove_loops_inside_one_lumen`, each round of `consolidate_lumen`
  with a mask: a loop whose antipodal chords mostly cross no more than a dropout loses its arc
  nearest the wall; a loop round tissue stays however short. **Short loops**
  (`remove_short_loops`, `remove_short_loops_under_um`, 200 µm, on): every loop that short loses
  its arc nearest the wall whatever the mask says, except round a hole wider than
  `short_loop_keep_hole_over_um` (10 µm; `loop_hole_um`). 0 leaves every short loop to the review
  list. `preprocessing/raw_evidence.py` (`RawEvidence`) ships no rule: no raw measure beat marking
  all one way.
- **`graph/skeleton_retrace.py`** — `retrace_lost_skeleton` (on; inert while
  `short_loop_keep_hole_over_um` is 0): after the last clean-up, skeleton the graph lost that would
  close a loop round a hole wider than `short_loop_keep_hole_over_um` is drawn back along its own
  skeleton, joined by splitting the nearest vessel (never a bridge), marked `retraced` /
  `retraced_join`. `judge_lost_skeleton` says read-only what it would do with each piece and why.
- **`graph/smoothing.py`** — `smooth_graph_centrelines` takes the voxel staircase out of each
  centreline and **re-measures `length`**, which moves every resistance. It is *not* one of the
  `STEP_LABELS`: it runs in `build_network` before the graph is pickled, so a caller assembling a
  graph by hand gets no smoothing unless it asks. A smoothed path must stay within `max_deviation`
  of a skeleton voxel (or of the path it came from, where that is off the skeleton) and be no
  longer; each edge records `smoothed` / `relaxed` / `kept_raw` / `too_short`. A bridge is never
  merged with a vessel at a degree-2 node.
- **One reader of the mask a run** — `build_network` makes one `preprocessing.MaskSupport` and
  hands it to graph building (`mask_support=`), both clean-ups, the thick-vessel split
  (`MaskSupport.limited_to`) and the dead-end marks; a new step reading the mask there takes it
  rather than packing its own. Under `use_memmap_loading` (the low-RAM option) no step makes a
  whole-volume copy: components through `slab_labels.ForegroundLabels` and slab components,
  distances through `FeatureDistance(by_block=True)` tiles or `blockwise_edt`, unions through
  `memmap_support.MaskUnion`, scratch arrays through `new_memmap_array` (its files are deleted once
  unreferenced). Every such path gives the in-RAM result **bit for bit**
  (`tests/test_low_ram_volume_copies.py`, `test_low_ram_*.py`); coordinates standing in for
  `np.argwhere` keep its Fortran layout, since BLAS rounds by layout. Routing workers start with
  one OpenMP/BLAS thread (`reconnect._routing_environment`).

### Dead ends and open ends (`docs/notes/dead_ends_and_open_ends.md`, `PLAN_dead_ends.md`)

- **`graph/dead_ends.py`** — the one definition of what each dead end (degree-1 node) is, judged
  as its whole chain (`chain_to_junction`): `leaves_image`, `island_end`, `artefact`,
  `stops_short`, `broken`, `blind` (a person's mark: `set_dead_end_kind`, kept through
  reclassifying) or `unknown`. Every module asks it. `build_network` writes `dead_end_kind` on each
  dead end and `image_extent_um` on the graph, so stages without the mask read the same answer
  (`node_leaves_image`, `open_end_nodes`); `post_process` reclassifies only the touched nodes
  (`reclassify_dead_ends`). No rule reached 90% on Harvey's grids, so the kinds only label and
  list, and every repair in `graph/dead_end_extension.py` stays off. `remove_pieces_without_junction`
  and `remove_interior_dead_ends` (Graph tab) are off by default. The fresh grids G1'-G3'
  (`scripts/dead_end_review/build_grids.py --fresh`, rules fixed in `fit_marks.FRESH_RULES`),
  marked by Harvey, confirmed no rule either. `vessels_cut_by_image_faces`: the vessels on a leaving dead end's chain,
  whose lengths are censored (decision D5): flagged in the VTK (`cut_by_image_face`) and the
  connectivity CSV, and given apart in the statistics' Dead Ends section.
- **The raw image at dead ends** (`PLAN_dead_ends_raw.md`) — `RawEvidenceEndothelial.ring_along`
  (E1: a wall ring round the chain, and whose), `inside_walls` (E2: walls enclosing a point, a
  lumen shared with another centreline) and `walls_ahead` (E3: the walls running on past the tip)
  measure only, a window at a time; no rule reads them. Grid DR1 waits on Harvey's marks.
- **Flow states** — the solve writes `graph.FLOW_STATE` (`solved`, `blind_end`, `closed_end`,
  `no_boundary`) beside `FLOW_SOLVED`. `blind_end` and `closed_end` flows and drops are exactly 0:
  **call `graph.zero_flows_behind_dead_ends` after any new `set_edge_flows` on a marked graph.**
  Only `no_boundary` flows are blanked in exports (`graph.flow_is_undefined`). What is blanked is
  one list, `graph.SOLVE_DERIVED_EDGE_ATTRIBUTES`: the VTK, a perturbation's edges and blocked-vessels
  CSVs (`stages._solve_value`) and the viewer's `gui/results.SOLVED_FLOW_COLUMNS` all read it. The viewer draws no
  flow arrow at or below `visualization.flow_direction.flow_arrow_floor`.
- **Pieces cut off by an edit** — `graph.branch_order.assign_branch_orders` pops `branch_order`
  from an edge the search from the inlets does not reach.
- **`haemodynamics/open_ends.py`** — `open_end_conditions(model, G, ...)`: every model becomes
  conductances to two ghost nodes at the inlet and outlet pressure (`augment`), never added to the
  graph; `estimated` reproduces its estimate exactly. The baseline keeps its conditions on the
  graph, and **every solver path augments with them** (the solve stage, the haematocrit loop,
  `_solve_network`, `solve_sweep_point`). `scripts/open_end_sensitivity.py` compares the models.
  `open_end_directions(G)`: which open ends feed the network and which drain it, by their solved
  flow. The network analyses take them as inlets and outlets beside the designated ones
  (`statistics._route_graph._resolve_terminals`, the one place they ask), and the VTK names
  them (`boundary_role`). Flow statistics read `flow_state`, never `== 0`; with
  `statistics_flow_face_margin_um` the per-vessel flow rows leave the vessels by the faces out.

### Thick vessels (`docs/notes/thick_vessels.md`, `PLAN_thick_vessel_bridges.md`, `PLAN_bridging.md`)

- **`preprocessing/thick_vessels.py`** — `use_thick_vessel_skeletonisation`: the fat region (an
  inscribed-radius gate) gets a centreline *tree* instead of Lee's medial sheet, and the thin arms
  fused to it are joined onto it (`_join_thin_arms_to_fat_ridge`: each arm aims at the tree as it
  was before any arm joined). `thick_vessel_braid_guard.py` flags braids left. The region is cached
  beside the saved skeleton, one bit a voxel, keyed on a digest of the mask and the settings that
  decide it (`thick_vessel_region_cache.py`). Under the low-RAM option it matches the in-RAM
  skeleton bit for bit, with one set of workers shared across its passes
  (`shared_volume_tasks.kept_workers`). `skeleton_thick_vessel_large_stack_tree`
  (`thick_vessel_tree.py`, off, low-RAM only) builds the tree without the per-voxel graph; it draws
  the same arms, so it does not keep a network's narrower vessels (waiting on a grid).
- **`graph/thick_vessel_junctions.py`** (`build_network`'s last step; `post_process` for edited
  vessels) — Harvey's rules: a thick vessel is its centreline alone, and every vessel opening into
  it is joined by one `IS_ZERO_RESISTANCE` bridge, straight from where it leaves the thick region
  to the centreline. A bridge is never a dead end nor attached to one, and nothing is ever cut off.
  The centreline is the tree found again from the region; a graph centreline is on it within
  `thick_bridge_centreline_tolerance_um` (1.5 um, unfitted) or 30% of the lumen radius, for the
  split and for every clean-up keeping the strand on it (`stages.thick_bridge_centreline_tolerance_um`).
  Bridges ending within one lumen radius of each other share one node (`_share_centreline_nodes`),
  and one whose straight line would leave the segmentation keeps the way its vessel ran
  (`DRAWN_BRIDGE`). `graph/thick_vessel_bridge_shapes.py`'s `diagnose_thick_vessel_bridges` (read-only;
  logged by `build_network`, a warning for a shape the split never means to make, and in
  `scripts/graph_artefacts.py`) names each bridge's shape. Around that: gaps in the graph's thick
  centreline are drawn from the tree (`thick_vessel_fill_centreline_gaps`, on), dead ends at the
  wall are drawn on to it (`join_dead_ends_at_walls`), a bridge carrying a centreline's end on is a
  taper and stays vessel (`THICK_TAPER`), forks are bridged (`bridge_forks`), centreline ends are
  extended through the segmentation (`thick_vessel_extend_centreline_ends`, on, not yet checked on
  a fresh grid), and blind vessels beyond a bridge the stub prune would take go with it. On their
  fresh grid P (Harvey, 2026-10-25) the wall join and the fork rule beat marking all one way and
  the taper rule did not (`docs/notes/thick_vessels.md`). Bridges
  are not measured (the widest diameter at their ends, `diameter_source="bridge"`), get no branch
  order, and statistics contract them (`contract_bridges`).

### Pipeline (`docs/notes/pipeline_and_post_processing.md`)

- **`pipeline/stages.py`** — one function per stage, each taking settings plus the previous
  stage's dataclass, so a caller can run them one at a time and intervene. `run_pipeline_stages`
  runs all ten and returns the graph; `start_from` / `resume` re-run from a stage on a saved run's
  state (`PipelineResume` carries every boundary list and the thick region), `stop_after` ends after
  a named stage. `post_process` measures only the vessels an edit marked
  (`assign_edge_diameters(edges=...)`), bridges them (`insert_thick_vessel_junction_nodes(edges=...)`),
  re-judges boundary nodes and dead ends at the touched nodes, assigns branch orders over the whole
  network, then `solve_post_processed`. `apply_network_handling(settings, model, boundaries, network)`
  is the Haemodynamics tab's "Network handling": `boundary_handling` acts on every vessel no
  inlet-to-outlet path runs along (`graph.inlet_to_outlet_vessels`) — `leave_unsolved` (default)
  marks them `FLOW_SOLVED=False`, `remove_disconnected` deletes them — and
  `haematocrit_junction_rule="split_junctions"` splits 4+ junctions on a copy. It runs at the start
  of the haemodynamics stage; a caller running stages by hand calls it between `assign_diameters`
  and `build_haemodynamic_model`. The diameter calibrations sample only the vessels it will keep
  (`edges_network_handling_keeps`).
- **`pipeline/progress.py`** — `STAGES`, the run's stages in order (one tab each, one progress bar
  over them): eleven entries for ten stage functions (`solve` shares **6. Haemodynamics**;
  **9. Additional measurements** has no function). `run_pipeline_stages(settings, schema, progress=callback)`
  hands the callback a `ProgressEvent` as each stage starts, finishes (with `seconds`) or fails, and
  one per topology step; `log_progress` is the console consumer. Nothing here imports a GUI.
  **7. Post processing** has no settings: its Review lists (`gui/post_processing.REVIEW_LISTS`) are
  `graph/network_review.py`'s read-only finders, one list per scan. Each decision is a
  `ReviewDecision` or `LoopReview`, matched by position rather than node ids, kept in the
  `.haemorun` and appended to `{stem}_{kind}_review.csv` with its measures, so a rule can be fitted
  to the marks later. Edits go through `graph/post_processing.py` and `graph/edit.py`; a `.haemorun`
  in the old stage order is reordered on load (`with_post_process_checkpoint`).
- **`pipeline/provenance.py`** — every run writes `{stem}_run_manifest.json` and
  `{stem}_run_settings.yaml` (a config `--config` loads) into its output folder: the settings after
  `Schema.upgrade` as the run started, version, git commit and dirty paths, environment, each input
  file's size, mtime and sha256, each stage's times and peak RSS. A failed run writes one too
  (`status: "failed"`). Writing it never fails a run.
- **`pipeline/checks.py`** — `preflight(settings, schema)`, the pre-run checklist: the schema's own
  checks (`parsers/checks.py`) plus what only this pipeline knows — the artefact a skipped stage
  must have left behind, whether the ilastik executable can be found, and whether every output
  path fits Windows' limit while long paths are off (`batch.check_output_path_lengths`, for any
  run). The examples call it before doing any work and exit if it fails.

### Diameters and haemodynamics (`docs/notes/diameters_and_haemodynamics.md`)

- **`haemodynamics/fwhm_diameter.py`** — FWHM vessel-**diameter** measurement from the raw image
  (`measure_edge_diameters_fwhm_from_raw_tiff`, `build_graph_branch_label_volume`; neither is
  re-exported, so import them from the module). The diameter chain is
  FWHM → raw section (`raw_section.py`, `use_raw_section_fallback`, off) → mask (`edt_diameter.py`,
  reading the run's segmented input) → class median (`diameter_class_median_min_edges`) → table
  (`stamp_edge_diameters`). For an endothelial stain `endothelial.py` takes FWHM's place
  (endothelium → mask → class median → table), and its edges carry `diameter_basis="endothelial"`.
  FWHM's guards: `fwhm_min_accepted_samples`, the decoy check (`fwhm_decoys.py`,
  `fwhm_decoy_check`), demotion of flagged widths (`fwhm_demote_flagged_edges`), blur held at the
  image PSF (`fwhm_fix_blur_to_image_psf`); `fwhm_planted.py` scores accuracy for the FWHM
  optimiser. `sections.py`'s `run_edge_tasks` spreads per-edge work over workers reading one
  memory-mapped volume.
- **`haemodynamics/constriction.py`** — the constriction model: diameter profile around a site,
  the resistance integral, and `apply_constriction_sites`, the only place a constricted edge's
  resistance is computed. A strategy supplies *where* the sites are (`ConstrictionSites`);
  `pericyte_mask.py` takes them from a segmented mask, `probability.py` places them periodically
  and activates each with a probability. Its viscosity is the configured law from
  `viscosity.py`, and its resistances are in Pa·s/m³, so a constricted edge is directly
  comparable with `poiseuille.py`'s uniform one — an edge with no sites on it is *exactly*
  `PoiseuilleModel.resistance_of_uniform_segment`, which `tests/test_constriction.py` pins as an
  equality.
- **`haemodynamics/constriction_strategy.py`** — `set_resistances_for_constriction_strategy`, the
  single place the settings pick a strategy, and `constriction_strategy_kwargs`, the single place
  its arguments are read from settings. Used by `apply.py`, `pericyte_comparison.py`, both
  pericyte perturbation branches in `pipeline/stages.py`, `pericyte_sweep.py` and
  `pericyte_geometry_sweep.py` — build the arguments with the helper rather than by hand, or the
  callers drift apart (they did: the mask assignment limits were silently dropped in two of them).
- **`haemodynamics/apply.py`** — high-level Poiseuille application used by examples and tutorial.
  Every resistance computation ends with `apply_final_resistance_overrides`: `custom_edges` at
  their fixed diameter, then negligible resistance on thick-vessel bridges (`IS_ZERO_RESISTANCE`
  edges from `graph/thick_vessel_junctions.py`). **Anything that recomputes resistances must end
  the same way** (perturbations and sweeps call it through `pericyte_sweep.apply_baseline_overrides`),
  or those edges revert to full vessels and the change is reported as the perturbation's effect.
- **`haemodynamics/viscosity.py`** / **`haematocrit_distribution.py`** — the viscosity laws (see
  *Viscosity model* above) and, with `haematocrit_model="distributed_iterative"`, a per-edge
  discharge haematocrit from Pries & Secomb's phase-separation law, iterated with the flow solve
  and under-relaxed each pass. `haematocrit_junction_rule` decides every junction that is not one
  vessel dividing in two: `no_separation` (default), `sequential_bifurcations` or
  `split_junctions`. The default `"fixed"` uses one `haematocrit` everywhere. The solve stage records
  which ran in `G.graph["haematocrit_model"]`, and the VTK writes it to its field data
  (`vtk_io.RUN_FIELD_DATA`). The loop ends with one
  more solve (`final_solve=True`), so a vessel's flow is its drop over its reported resistance; a
  caller that re-solves itself passes `final_solve=False`.
- **`haemodynamics/resistance.py`** — the flow solve and the two-point resistance.
  `haemodynamics_solver` (`dense` default, or `sparse`) is how *every* flow solve holds the network
  (`build_conductance_matrix_from_graph(solver=)`); `equivalent_resistance_solver`
  (`eigendecomposition` default, or `sparse`, `inf` between unconnected nodes). A dense solver is
  warned about when its estimate passes half the free memory (`warn_about_dense_solver_memory`,
  from `build_network` and `solve`). The per-N² memory figures are pinned in
  `tests/test_flow_solvers.py`, so a solve that gains an N × N copy fails there.
  `Solution.network_resistance` is the drop over total inflow; perturbation summaries'
  `equivalent_resistance` is that, not the two-point one.
- **`haemodynamics/perturbations.py`** — a perturbation is a named, typed settings override
  (`PerturbationSpec`) that `run_perturbations` re-solves a copy of the solved network for, written
  into `{name}_{type}` under `perturbation_output_dir`. `PERTURBATION_TYPES` lists them;
  `SETTINGS_FOR_TYPE` says which settings each reads (an override outside it is reported unused and
  not applied); `INCOMPARABLE_OVERRIDES` (viscosity law, raw stain, haematocrit, haematocrit model)
  are refused. The capillary blocks (`CAPILLARY_BLOCK_TYPES`, `is_capillary_block`; including
  `capillary_block_mask`, `haemodynamics/capillary_block_mask.py`) zero the stopped vessels' flows
  after every re-solve (`capillary_block.zero_stopped_vessel_flows`). A sweep writes its grid CSV
  and curves and keeps per-point flows (`sweep_flows.py`) instead of N graph copies.
- **`haemodynamics/red_cells.py`** — tube haematocrit, red-cell velocity, flux and spacing on each
  vessel carrying flow, *predicted* from the flow and the discharge haematocrit by the Fahraeus
  relation (Pries et al. 1990) and red-cell conservation, with `red_cell_volume_fl` the mean cell
  volume. Predictions, never measurements: they carry every error of the flow.

### Statistics (`docs/notes/statistics.md`)

- **Network analyses** (`statistics/inlet_outlet_routes.py`, `transit_time.py`,
  `flux_heterogeneity.py`, `territories.py`, `capillary_bed_joins.py`, `current_flow.py`,
  `occlusion.py`, `loops.py`, `strahler.py`, `spectral.py`) — each a `statistics_*` toggle in the
  schema's *Connectivity/Network Analysis* section, run inside
  `compute_comprehensive_vessel_statistics`. They add rows to the statistics CSV, and per-vessel
  edge attributes offered as colour options must be declared in `gui/results.py`'s
  `OPTIONAL_EDGE_COLUMNS`. An analysis that cannot run writes nothing. `statistics_route_weighting`
  picks the conductance model the Laplacian-based ones share through `_flow_system.py`.
  "Carrying flow" is one definition: `vessel_classes.carries_flow` above `CARRYING_FLOOR_FRACTION`
  of the largest flow.
- **Exact betweenness** (`statistics/exact_betweenness.py`) — every exact
  `betweenness_centrality` goes through `exact_betweenness_centrality`, which spreads the
  per-source searches over workers from `PARALLEL_BETWEENNESS_MIN_NODES` nodes and gives
  networkx's floats bit for bit. The unweighted, undirected pass without endpoints (the one the
  statistics make) is a batched numpy search over many sources at once (`_BATCHED_PASS`), every
  addition in networkx's order, so each source's increments are networkx's bits; weighted,
  endpoint and directed passes keep networkx's loop. Never sampling (`k`) or another library.
- **Vascular communities** (`graph/communities.py`) — the `compute_vascular_communities` toggle,
  independent of Statistics; when Statistics also runs, the two share one partition
  (`pipeline/stages.py`'s `shared_vascular_community_partition`) instead of computing it twice.
- **`statistics/three_dim_distances.py`** — cell-to-vessel distances (`measurement_3d_to_cell_mask`,
  run last in `export_results`): the gap between voxelised volumes, so a cell touching a vessel is
  0 µm from it, with `vessel_contact`. The vessels are `measurement_3d_vessel_mask_path` if set,
  else the run's segmentation, else the graph's tubes; the vessel named is the one whose lumen the
  nearest vessel point lies in, never a bridge. Preflight checks shape and voxel size
  (`checks.check_cell_mask_measurement`). An optional path of this kind is `Setting.optional_path`
  (unset falls back, set must exist).
- **`statistics/tissue_volume.py`** — `measure_tissue_volume` (Additional measurements tab, off by
  default): the tissue in the raw image (`tissue_raw_tiff_path`, else `raw_image_path`) as a closed
  surface mesh, so vessel density can divide by tissue rather than the bounding box. Measured once
  per network (`stages.tissue_measurement`, cached on `VesselNetwork.tissue`) so perturbations
  divide by the same tissue as the baseline; writes `{stem}_tissue_volume.csv` and
  `{stem}_tissue_surface.vtp`.

### Optimisation, exports and the panel (`docs/notes/gui.md`)

- **`optimisation/`** — the panel's **Optimise settings**: a sequential, image-informed search
  over segmentation cleanup, Skeletonise and Graph settings (`search.py`) and, separately, the
  FWHM diameter settings (`fwhm_search.py`; a trial measures through the run's own call, inside
  the whole vessel graph). Candidates come from the image itself
  (`candidates.py`), trials are scored by pure functions (`metrics.py`), and the result is a
  config file plus a readable report (`report.py`). **A trial skeletonises as the run does**
  (`PLAN_optimisers.md`): the panel hands the search the run's grid
  (`stages.load_volume_on_run_grid`), its whole settings, voxel counts on that grid
  (`skeleton_voxel_settings_on_grid` in, `_from_grid` out) and its thick-vessel restriction
  (`stages.thick_vessel_restriction_mask`); plain trials go through
  `io.load._skeletonize_loaded_volume`. Every step's arguments come from the pipeline's own
  builders (`stages.segmentation_cleanup_arguments`, `skeleton_cleanup_arguments`,
  `thick_region_settings` + `thick_vessel_refinement_arguments`, `graph_build_arguments`,
  `centreline_smoothing_arguments`), so a setting a stage starts reading reaches the trials with
  no change here; a setting a caller leaves out is read as its schema default. **A trial's graph
  is the run's**: built through `stages.network_context` and finished by
  `stages.finish_network_graph`, as `build_network` does, so a step added to the finish reaches
  the optimiser for free; the scorecard marks dead ends with `stages.mark_dead_end_kinds` and
  counts them by kind, measures only (no kind is better until a marked grid says so).
  `tests/test_optimisation_search.py` pins a trial's skeleton `==` `skeletonise`'s, its graph `==`
  `build_network`'s edge for edge, and each step's arguments `==` the run's. The search's own
  coarser grid is reduced by block max, not import's majority, and its report says so (D2).
  **Optimise endothelial settings** (`endothelial_search.py`,
  shown instead of the FWHM button when the raw image stains the wall) sweeps the ring fit's
  contrast gate, and its blur and wall only where the image's rings will not calibrate them or the
  settings fix them (then the image's own values, fixed; Harvey's rule); a trial reads
  through `apply.measure_endothelial_diameters` with the run's config, each blur's calibration
  fits made once (`apply.endothelial_calibration_fits`), scored on planted hollow vessels
  (`haemodynamics/endothelial_planted.py`), the sample and decoys. **Check segmented image** is
  `preprocessing/segmentation_quality.py` + `segmentation_raw_comparison.py`.
- **`visualization/geometry.py`** — `edge_polyline`: an edge's `voxels` (or its two node positions)
  turned into a polyline that runs `u`→`v` and touches both nodes. The VTK export, the pericyte
  point derivation and two plotly writers each answered that separately, and drew vessels in
  slightly different places; this is the one answer.
- **`gui/boundary_picking.py`** — the four boundary roles' coordinate and volume settings as napari
  Points and Shapes layers, and back. Pure, like the rest of `gui/` bar `_widget.py` and
  `log_view.py` (the only two that touch Qt, and only inside functions): settings in,
  layer specs out, layer data in, settings out, nothing importing napari. `rectangle_from_box` and
  `box_from_rectangle` are exact inverses, which is what lets an edited layer *be* the setting.
- **`gui/run_snapshot.py`** / **`gui/stage_checkpoints.py`** — save a finished run to a `.haemorun`
  and load it back; keep per-stage snapshots so a tab can re-run from its stage. Save captures on
  the GUI thread and writes on a worker (`save_run_file(background=True)`), gzip level 1
  (`COMPRESS_LEVEL`) into a `.part` file moved over the name once complete.
- **`gui/layout.py`** — where each row sits on its tab: one box per schema section, or grouped by
  hand (`TAB_BOXES`). Every `advanced` setting goes behind an Advanced button anchored on the
  innermost ordinary row it needs, so buttons nest as the settings do. The panel follows each row's
  whole prerequisite chain (`form.prerequisite_chain`). Tests open every button with
  `panel._haemolynx_expand_advanced()`.
- **Statistics tab** (`gui/statistics_view.py`, `_widget.py`'s `_StatisticsPanel`) —
  `export_results` leaves what it measured on the graph as
  `G.graph[pipeline.stages.STATISTICS_REPORT]` (`{"mode", "csv", "sections"}`, one section per
  measure, from `statistics.compute_vessel_statistics_by_measure`); the tab draws a button per
  section. `run_pipeline_stages` drops the report from a resumed network.
- **`gui/layer_sets.py`** — what the view panel's "Showing" menu swaps: the baseline network or
  one perturbation's, every other hidden. A perturbation's `LayerSpec`s carry `layer_set=<its name>`
  (stored in the layer's `OURS` tag); the baseline's are `None`. Sweep sliders sit in the view
  panel's Sweep box. "Colour by", "Colour range" and the "Show" checkboxes (`VIEW_TOGGLES`) act on
  the shown network through the same paths as the layer controls, and a box mirrors its layer's
  `visible` rather than fighting the eye icon.
- **`gui/vessel_tubes.py`** — the vessel tube mesh, display only (the graph and its diameters are
  untouched): `vessel_tube_mesh(vectors, features)` is what `_sync_vessel_tubes` calls. A
  zero-resistance bridge is not vessel and is drawn so (dashed lines, `dash_polyline`; thin flat
  dashes among the tubes). Render quality and the tube-diameter choice (`per_vessel` or `uniform`)
  are session-wide.

### Batch runs and synthetic networks (`docs/notes/batch_run.md`, `synthetic.md`)

- **Batch run** (`PLAN_batch_run.md`) — the pipeline once per segmented image in a folder, or,
  with `use_ilastik_segmentation` on, per raw image (`BatchFolders(from_raw=True)`,
  `batch_from_raw`), each named `{raw stem}_segmented` after the segmentation ilastik makes
  (`io.ilastik.ilastik_segmented_stem`, the one naming rule). `io/batch_matching.py`'s
  `match_batch` pairs each image with its raw image, masks and the other files the run reads
  (`FILE_ROLES`: pericyte, cell, 3D-distance vessel, tissue, blocks; a folder each, matched only
  while read, `batch_file_roles`) by name; ties are named, never guessed. `pipeline/batch.py`:
  `item_settings` (each image's files, its perturbations' too, and its own output folder),
  `check_batch(settings, schema, folders)` (refuses what cannot be per-image: node IDs, synthetic,
  one `FILE_ROLES` file read for every image), `batch_preflight`. ilastik's vessel masks read each
  image's own raw image.
  `gui/batch_run.py`'s `run_batch` runs
  them in turn with fresh state each, records a failure and goes on, rewrites `batch_summary.csv`
  after every image; every image passes its checks or nothing runs. Headless:
  `python -m haemolynx.gui.batch_run` (`batch_run_cli.py`). Conditions and the batch's statistics
  (`statistics/batch_comparison.py`): independent images (Mann-Whitney, Benjamini-Hochberg) or
  paired by subject (Wilcoxon), with `--exclude-section`. In the panel, a set batch is panel state
  (`batch_mode.setup`), not settings. Each image's checks refuse an output path past Windows'
  limit while long paths are off (`check_output_path_lengths`, in every run's preflight). A set batch greys what acts on one
  image (Check segmented image, the optimisers, Run from this stage) and reads the rows it fills
  as set for greying (`BatchSetup.gates`); Load run leaves it, and is refused while a batch runs.
  The Batch run button greys under a synthetic network and while one image runs, never while a
  batch is set or running (`batch_button_greyed`, `RunState.on_change`). Each image's on-disk caches (its decompressed input, checkpoint
  pickles, thick-vessel region) are deleted once it ends (`batch_run.clear_image_caches`).
- **`synthetic/`** — a network drawn to a previous run (see *A synthetic network* above).
  `template.py`'s `NetworkTemplate` (from the saved statistics report or the graph), `generate.py`
  (+ `trees.py`, `bed.py`, `polylines.py`) draws it, `compare.py` scores it per characteristic,
  every band from its one `TOLERANCES` table. The large tier is a trunk handing side branches to
  the bed at its orders' own widths (Murray's law in the Art/Ven tier only), grown in the cells
  the template's trees of that side occupy (`trees.TreeLayer`); the bed is laid out cell by cell
  to the template's capillary density and runs the way its capillaries run
  (`synthetic/organisation.py`, measured on the template, scored in the comparison), along
  Delaunay edges or, with `synthetic_bed_model=voronoi`, a Voronoi tessellation's cell edges.
  The network is closed by default (`synthetic_open_ends=closed`: no vessel leaves the box, all
  flow in at the inlets and out at the outlets, shown in the viewer; `template` cuts the
  template's count at the faces), and the comparison reports what was not drawn
  (`compare.not_drawn`). Still being made physiological: `PLAN_synthetic_physiology.md`. `validate_synthetic_graph` pins the graph contract
  (int ids, `pos`/`voxels` in µm, `length` the polyline's, `diameter_source="synthetic"` and, given
  the template, its `diameter_basis`, no degree-2 node, `dead_end_kind` on every dead end,
  connected).
- **Presets** (`pipeline/presets.py`) — `PRESETS`, named partial configs: the panel's Preset menu
  under Load config... sets the rows a preset names and nothing else, and keeps no state (a config
  or run loaded afterwards puts the menu back to None). Switches and choices only, never a file or
  a threshold. The `USAGE_PRESETS` between them name every setting any of them sets, so one chosen
  after another gives the same run (`tests/test_pipeline_presets.py`); the rest add to a usage.
  `examples/pipeline_presets.py` offers them on `--preset` beside its own (which name files), and
  checks every one against the schema at import. The override engine is `parsers/cli.py`.
---

---

## Help menu (required)

The **?** beside "Show each stage in the viewer" opens the help window (`gui/help_view.py`). Its pages are `gui/stage_help.py`: one entry per panel tab. The right-hand pane is that entry's text. Those words are **human-written**. Do not draft, rewrite, expand, paraphrase, summarise, or fill any entry. An agent may place text the user has already written onto the entry they name, and must not change the words while placing them.

If the user asks for help-menu text to be written, generated, drafted, or filled in — including "write the help", "fill the empty entries", "draft this tab", or any request whose result would be agent-composed prose in that window — **stop before any edit, tool call, or other work on that request** and reply with exactly:

Message from Harvey - Write it yourself don't add AI generated text to the help menu

---

## Marked example grids (required where relevant)

When a rule has to decide what a structure *is* (a vessel loop or a Lee ring, a sprout or a
spur, one lumen or two touching vessels, a real dead end or a broken vessel), or where a threshold
for such a call should sit, do not choose the measure or the threshold by assumption. Calibrate it
against the user's own judgement:

1. **Show a grid of about 30 numbered examples** (IDs 1, 2, 3 …) from the user's real data, and
   send it to them as an image. The user replies with a mark per ID (for example `1 F, 2 C …`).
2. **Draw what the code sees.** For graph-building rules that is the segmentation mask, not the
   raw image. Show each case in its own plane, with the element being judged highlighted, the
   nearby centrelines, a scale bar, and its voxel position so it can be checked in napari.
3. **Spread the examples across the candidate measure, then shuffle them**, so an ID gives away
   nothing about the answer. Put the question, and what each mark means, in the image header
   (e.g. CORRECT = keep, FALSE = remove).
4. **Keep a key** next to the image under `outputs/<topic>_review/` (git-ignored): a CSV mapping
   each ID to every candidate measure, its position and its graph elements. Leave the measures
   off the image itself.
5. **Fit the rule to the marks and report it honestly**: how many examples it gets right,
   compared with marking everything one way, and which IDs it still gets wrong. If the marks
   contradict the measure you assumed, say so and change the measure, not the marks.
6. **Before relying on a rule, check it on a fresh grid** it was not fitted on. Turn each kind of
   case the marks revealed into a synthetic fixture, so the tests pin it.

The first grid was the E14.5 loop review (`outputs/loop_review/`, 2026-10-08). Hole size, the
measure assumed, did not separate the user's marks. Whether a gap lies inside the loop and the
loop runs through the mask did.

---

## What to avoid

- Writing help-menu text. See **Help menu** above: stop and send Harvey's message before doing anything else on that request.
- Committing large generated outputs (`examples/outputs/`, `tutorials/plots/`, `tests/outputs/`) unless the user explicitly asks.
- Editing auto-generated `tutorials/pipeline_tutorial.py` instead of the notebook.
- Adding features without tests (see Testing policy).
- Force-pushing `main`/`master` or amending pushed commits unless explicitly requested.
- Expanding scope into unrelated refactors when fixing a targeted issue.
- Raising a release-review score without evidence in its row, or moving a check's target to make it pass (see **Release review** below).
- Adding dated benchmarks, grid results or E14.5 numbers to this file: they go in `docs/notes/` or the matching `PLAN_*.md` (see **Key modules**).

---

## Useful commands

```bash
# Full test suite
pytest -s

# Fast subset (exclude slow integration)
pytest -m "not slow"

# Single file
pytest tests/test_graph.py -s

# Regenerate tutorial Python from notebook
pytest tests/integration/test_pipeline_tutorial.py -s
```

---

# Release review (TEMPORARY — delete this whole section at the final release)

**`RELEASE_REVIEW.md`** scores every aspect of the software **out of 10**, as sixteen aspects:
graph, diameter and haemodynamic accuracy; statistics; inputs and segmentation; speed; memory;
usability; visualisation; exports; documentation and help; robustness; reproducibility; testing
and CI; packaging; and maintainability. For each one it lists the checks that hold the score
where it is, so it doubles as the list of what to do before release. **`PLAN_release_review.md`**
is the plan for `scripts/release_review/`, a command that will re-measure the automated checks
(steps R1–R8). Delete both files, the scorer and this section once the release gate in
`RELEASE_REVIEW.md` is met. First scoring: 2026-10-10 at `9c6a15a`, lowest score 3, mean 3.8.

Rules for agents:

- **A score comes from its ladder, never from impression.** Each aspect has checks at levels 2,
  4, 6, 8 and 10. Its score is the highest level at which that level's checks, and every lower
  level's, all pass, plus one if half of the next level's checks pass. Write only the number the
  rows give.
- **A check passes only with evidence in its row.** That means a test (`file:line`), an output
  file with its commit and dataset, or a person's marks with their name and the date. If the
  evidence is unknown, the check fails, and the row says what would decide it.
- **H (human) checks belong to the six named testers**: Harvey, Finbar, Mason, George, Rebecca
  and Mike (`RELEASE_REVIEW.md` § Testers). Only George, Rebecca and Mike count as "someone
  other than the developer". Record a tester's result with their name, and never pass an H check
  on your own judgement. `docs.6a`, the help pages, passes only when Harvey
  writes them (see **Help menu**).
- **Never move a check or a target to make it pass.** Record any rubric change in the review
  log, with its reason. As with marked grids: change the measure, not the marks.
- **Re-score with the work.** A commit that changes a check's evidence updates that row, the
  scorecard line and the log in the same commit. A regression lowers the score at once.
- **Choosing work.** When there is room to choose, prefer whatever passes the lowest aspect's
  next check. The scorecard's "Work in order" list is that order.
- **Measure a `git archive` of the commit being scored**, not the live tree: peer sessions edit
  `src/` mid-run.

---

# Cleanup plan (TEMPORARY — delete this whole section once implemented)

> **Status:** Phases 0, 1, 2 and 3 are done (Phase 2's renames on 2026-10-23), Phase 4 all but
> the README note. The finished items and the findings behind the plan are in `docs/notes/cleanup_plan.md`.
> Record a full-run pass count before starting Phase 2, and re-run the full `pytest` (including
> `slow`/`integration`) after **every** rename; no phase may reduce the green count. Tick items as
> they land, and **remove this entire section** when every one is done.

- [x] **Phase 2 renames** — file/module names only, public function names unchanged; `git mv`, one
      commit per rename, imports updated in `src/`, `examples/`, `tests/` and the `__init__.py`
      re-exports:
  - [x] `io/automated_vessel_assignment.py` → `io/vessel_masks.py`
  - [x] `graph/automated_vessel_assignment.py` → `graph/terminal_node_assignment.py`
  - [x] `haemodynamics/automated.py` → `haemodynamics/fwhm_diameter.py`
- [ ] **Phase 4** — a short "preset system" note in the README (schema declarations → preset dicts →
      CLI/YAML overrides); the README does not mention presets today.
- [ ] **When done** — full `pytest` green, the tutorial notebook still exports and runs; update the
      Repository layout and Key modules sections to the new names, drop the misnomer caveats, and
      delete this section.
