# Plan: the two optimisers, in line with the pipeline as it now runs (2026-10-23)

> **Status (2026-10-24): done but for what waits on Harvey or on a large run.** Phases 0-5 and
> the endothelial optimiser (D4) are on `Devel_GUI_improvements_HD`: Phase 2 `f1184f25`, Phase 3
> `9328c668`, Phases 1 and 4 `e1314748`, Phase 5 `399a48f3` + `924b9910`, the endothelial
> optimiser `65068bc1` .. `56be0b79`, B8 `ca5df387` + `aee7da34`. A trial's E14.5 skeleton and graph
> equal the run's, and so do FWHM and ring-fit trials' widths (see **Measured**). The last pieces
> are under **Finishing (2026-10-24)** at the end: the scorecard's dead ends by kind and loops
> inside one lumen, D2's report line, Principle 4's check, the docs.
>
> **Waiting on Harvey:** the two marked grids (D6's thick-vessel grid and the component filter's;
> built, no marks yet); whether the stub sweep may move `min_stub_length_radius_multiple` from the
> 3 radii he decided (see Principle 4's check); FWHM's planted-vessel error of about 0.54 on E14.5
> (B8), which needs looking at before the accuracy term is trusted there.
> **Waiting on a large run** (the machine had under 6 GB free on 2026-10-24): the "Off" refusal's
> 128 B a voxel at real size (about 9.6 GB on E14.5), and Optimise endothelial settings on E14.5
> under `56be0b79`'s rule. Findings were first read in the code at `666cafe8`. Decisions D1-D6 are
> Harvey's (taken 2026-10-23).

## Context

The panel has two optimisers:

- **Optimise settings** (Input tab): `optimisation/search.py`'s
  `optimise_skeleton_and_graph_settings`, a coordinate-descent search over 54 segmentation
  cleanup, Skeletonise and Graph settings (`OPTIMISE_SETTING_NAMES`), each candidate scored by
  running the real cleanup, skeletonisation, skeleton cleaning and graph building on the mask.
- **Optimise FWHM settings** (Diameters tab): `optimisation/fwhm_search.py`'s
  `optimise_fwhm_settings`, the same search over 34 FWHM settings (`FWHM_SETTING_NAMES`), scored
  by measuring a sample of the run's own vessels in the raw image, with decoys and planted
  vessels.

Both were last brought in line on 2026-10-18 (`612f0cba`, downsample on import). Since then, and
before, the pipeline has grown steps the optimisers never run: the clean-up after smoothing, the
thick-vessel split and its bridges, the lost-skeleton retrace, the dead-end kinds, the one
`MaskSupport` per run, network handling's calibration edges, the FWHM mask stop. A trial is meant
to be "the real pipeline call for this candidate"; today it is a partial copy of it, kept by hand.

**The aim:** a trial scores the network, or the widths, that a real run with the same settings
would make, through the same functions the run calls, so the two cannot drift apart again.

## What was found

### A. Optimise settings (`search.py`)

| # | What | Where | Effect |
|---|------|-------|--------|
| A1 | **Downsample on import is ignored for voxel counts.** The search gets the mask already reduced (`load_volume_for_skeletonise`), but the six voxel-counted skeleton settings go in as the user set them, on the image as read. A run divides them by the import factor (`stages.skeleton_voxel_settings_on_grid`); the search does not, and writes its result back on the reduced grid, which the run then divides again. | `_widget.py:6870`, `search.py:2600` | At 2x, a closing radius of 2 is searched as 4 source voxels, and a chosen 2 runs as 1. Wrong in both directions. |
| A2 | **Only the 54 swept settings reach the search.** The panel passes `{name: settings[name] for name in OPTIMISE_SETTING_NAMES}`; every other setting a step reads falls back to the module's own default: `bridge_require_mask_support`, `bridge_max_background_gap_um`, `bridge_min_mask_fraction`, `skeleton_bridge_min_facing_cosine`, `recover_uncovered_mask_vessels`, `recovery_*`, `facing_dead_end_max_gap_um`, `remove_short_loops_under_um`, `short_loop_keep_hole_over_um`, `cartwheel_hub_*`. | `_widget.py:6870`, `search.py:856-1001` | Checked: identical to the run at schema defaults. Wrong as soon as a user changes one (short loops off, recovery off, bridge gates loosened): the search scores graphs built under settings the run will not use. |
| A3 | **`_graph_kwargs` is a hand copy of `stages.graph_build_arguments`** and lacks 7 of its arguments: `cluster_collapse_direction_aware_min_degree` / `_tangent_length_um` (from `cartwheel_hub_*`), `stub_face_protection`, `lumen_cleanup_max_rounds`, `extend_dead_ends_to_faces`, `trace_on_dead_ends`, `join_dead_ends_to_vessel_sides`. `_skeleton_kwargs`, `_thick_vessel_kwargs` and `_cleanup_kwargs` are copies too (the cleanup one matches today). | `search.py:856-1001` vs `stages.py:1734` | Any of these changed by the user is invisible to the search; any argument added to the pipeline later is missed silently. |
| A4 | **The graph a trial scores stops at `build_graph_from_skeleton`.** A run (`build_network`) then smooths with the mask radius (`radius_at`), runs `consolidate_lumen` again, splits at thick vessels (`insert_thick_vessel_junction_nodes`) and consolidates again, retraces lost skeleton, smooths straight steps, optionally removes interior dead ends and pieces without a junction, and marks dead-end kinds. Graph building itself also gets `along_thick_centreline` and `mask_support` in a run, not in a trial. | `search.py:2257`, `stages.py:1990-2313` | The reconnect, collapse, stub and smoothing sweeps judge a graph the run never saves. Smoothing in particular: a run's tolerance is the larger of `centreline_max_deviation` and a fraction of the vessel radius (`smoothing.edge_tolerance_um`), a trial's is `centreline_max_deviation` alone, so the `centreline_max_deviation` sweep is scored on a different rule. |
| A5 | **(The largest drift on E14.5; see Measured.)** **Thick-vessel gating ignores the run's restriction mask** (`_thick_vessel_restriction_mask`, documented in `_thick_vessel_kwargs` as deliberate) and `skeleton_thick_vessel_large_stack_tree`. | `search.py:895` | With large-vessel masks restricting the thick region, the search tunes the thick-vessel radius and refinement for a region the run will not have. |
| A6 | **Low RAM is not honoured.** The panel binarises the whole mask in RAM (`_widget.py:6869`) and reads the whole raw image (`_load_raw_reference_image_or_none`), whatever `use_memmap_loading` says; only then does Auto reduce. | `_widget.py:6700-6870` | On the patella or BigSM stacks this is the in-RAM path the low-RAM option exists to avoid (memory: 48 GB on BigSM). |
| A7 | **Panel gating.** Optimise settings runs on `input_path` under `network_source=synthetic` (no gate); in batch mode it acts on the Input tab's one image (`PLAN_batch_run.md`'s status notes it is not greyed); with ilastik on and no `input_path` it refuses even when a current ilastik output exists. | `_widget.py:14586` | Confusing at best; under synthetic it optimises for an image the run does not read. |
| A9 | **A deselected cleanup group skips the run's cleanup** (corrected 2026-10-23; first written the other way round). Inside the cleanup group, trials and the returned settings agree: the group turns `segmentation_cleanup` on when it keeps a toggle. But with the group deselected the search never cleans the mask at all, even with the user's switch and a toggle on, which a run applies (`stages.skeletonise`). | `search.py` `_run_one_pass` | Latent: E14.5's config has every toggle off. |
| A10 | **The plain skeleton skips the run's hole fill** (found in Phase 0). A trial makes the plain-Lee skeleton with `preprocessing.skeletonize_volume`; a run with `io.load._skeletonize_loaded_volume`, which also fills holes (CLAUDE.md: every loader goes through it). | `search.py:1125` (`_raw_skeleton`) | On E14.5 with thick vessels off, the fill adds 50,441 voxels to the raw skeleton (+49%), and the cleaned skeletons then differ by 25,481 voxels (see **Measured**). |
| A8 | **Stale words.** `optimisation/__init__.py` says the search never imports `haemolynx.io` (it imports `io.downsample` and `io.load`; `fwhm_search` imports `io.downsample`), "all ten" groups (eleven) and "all seven" FWHM groups (eight). `_SEARCH_STEP_CALLS`, Auto's time model, was counted before the short-loop rule and the retrace. | `__init__.py`, `search.py:320` | Misleads the next reader; Auto's estimate may no longer hold. |

### B. Optimise FWHM settings (`fwhm_search.py`)

| # | What | Where | Effect |
|---|------|-------|--------|
| B1 | **Trials do not pass the segmentation to the measurement.** A run calls `measure_edge_diameters_fwhm_from_raw_tiff(..., vessel_mask=...)` (`apply._measure_fwhm_diameters`), so with `fwhm_stop_at_other_vessels_in_mask` on (the default) each profile stops at a neighbouring vessel in the mask. `_FwhmSearch._measure` passes no mask. | `fwhm_search.py:495`, `apply.py:394` | Every trial measures differently from the run it is choosing settings for, most where vessels lie close together. |
| B2 | **The run's widths leak into trials.** The panel hands the optimiser `copy_graph(results._graph)`, whose edges carry the run's FWHM results. An edge that fails in a trial keeps the run's `fwhm_diameter_um` and `fwhm_diameter_samples_um` (`haemodynamics/fwhm_diameter.py`, formerly `automated.py`, rewrites only `fwhm_status` on a failure). `_checked_trial` counts measured edges by `fwhm_diameter_um > 0` for the decoy report, and the regression guard's diameters read `fwhm_diameter_samples_um` from every edge (`fwhm_search.py:581`). | `_widget.py:13851`, `fwhm_search.py:548, 581` | A setting that makes edges fail is credited with the run's widths for them. `fwhm_metrics` itself reads `fwhm_status`, so the main score is right; the guards are not. The pipeline's `_start_measurement_afresh` exists for exactly this. |
| B3 | **Sample and PSF drawn from every edge.** The run never measures zero-resistance bridges (`assign_edge_diameters` sets them apart) and, with `boundary_handling=remove_disconnected`, calibrates only on `edges_network_handling_keeps`. The optimiser's `_representative_subgraph(G, ...)` and `image_psf_from_settings(G, ...)` take every edge. | `fwhm_search.py:982, 1010`, `apply.py:692-725` | Bridges (straight segments inside a thick vessel) and vessels about to be removed are measured and scored, and shape the PSF the trials hold the blur at. |
| B4 | **Its own copy of the measurement call.** `_measure` builds the call itself; the run's `_measure_fwhm_diameters` also passes `use_memmap`, `memmap_directory`, the graph's `image_voxel_size_zyx` and the axis order from its config. | `fwhm_search.py:495` | Same drift risk as A3. |
| B5 | **Run switches not honoured.** Trials always run the decoy check and demotion, whatever `fwhm_decoy_check` / `fwhm_demote_flagged_edges` say, and use `fwhm_decoys.DECOY_SAMPLE_SIZE` rather than `fwhm_decoy_check_sample_size`. | `fwhm_search.py:456-553` | Arguably right for scoring (see D3); the sample size is not. |
| B6 | **The raw-section fallback is unseen.** With `use_raw_section_fallback` on, vessels FWHM fails are measured by the fallback; the optimiser's coverage term does not know. | — | Report-only gap. |
| B8 | **Auto samples too few vessels to judge accuracy** (found checking Phase 3, 2026-10-23). Auto's time model overestimates a trial about fourfold, so on E14.5 it picks 10 of 1,342 vessels (estimated 857 s against its 180 s budget; the whole search took 209 s). With 10, no vessel of known width is planted and only 4 decoys are placed, so the accuracy group (`measurement_model`) is skipped and every choice rests on 1-3 measured vessels. With 200: 43 planted, 53 decoys. | `fwhm_search.py` (`_time_budget_sample`, `estimate_sample_edge_count_for_time_budget`) | The panel's default FWHM search on E14.5 never checked accuracy. **Fixed 2026-10-24** (see **B8** under **Measured**): Auto samples at least 100 vessels, and the time model counts the trials a pass makes and the planted vessels and decoys it measures; within 3% of the E14.5 search. |
| B7 | **The sample is measured as a graph of its own** (found in Phase 0). `_representative_subgraph` returns `G.edge_subgraph(sample)`, and each trial measures that alone, so the neighbouring vessels and junctions FWHM reads round each sampled vessel are missing. | `fwhm_search.py:263, 518` | The largest FWHM drift on E14.5 (see **Measured**): measured as a subgraph, 25 of 47 widths match the run's; measured inside the whole vessel graph with `edges=`, all 47 do, at the same cost. |

## Principles

1. **One function, two callers.** Every settings-to-arguments map and every post-build step has one
   home, in the pipeline; the optimisers call it. No second copy, however small (the precedent:
   `visualization/geometry.edge_polyline`, `stages.graph_build_arguments`).
2. **A trial is the run.** A trial's graph is `build_network`'s graph for those settings on the
   search grid; a trial's widths are `assign_edge_diameters`' FWHM widths on those edges. Pinned
   by equality tests, not tolerances.
3. **Every setting reaches the search; only the swept ones move.** The search reads the run's
   whole settings; `OPTIMISE_SETTING_NAMES` / `FWHM_SETTING_NAMES` say only what it may change.
4. **No new objective without a marked grid.** Making a trial match the run changes no measure
   and needs no grid. Changing *what a sweep minimises* (penalising dead ends, loops) decides what
   a structure is, so it follows CLAUDE.md's marked-grid rules; until then such counts go on the
   scorecard as measures only. Phase 0 also checks that no swept setting's default came from a
   marked grid (a grid-fixed value must not be re-tuned by a proxy cost).
5. **Measure the drift before fixing it**, so each phase's effect on the chosen settings is known
   and reported.

## Phases

### Phase 0 — Measure the drift (no behaviour change)

- A harness under `outputs/optimiser_alignment/` (git-ignored; a `__main__` guard, since spawn
  workers re-run an unguarded script). On E14.5 MCA at its saved config (the reproduction recipe
  in memory), for the starting settings and for the settings the optimiser chooses today:
  - `build_network`'s graph against the optimiser's trial graph (`_Search._build_graph` plus its
    smoothing), at factor 1 and at the factor Auto picks: edges, total length, components,
    skeleton coverage, dead ends by kind, loops inside one lumen, the lumen-artefact report.
  - FWHM: trial widths against `assign_edge_diameters`' widths on the same sampled edges, with
    and without the vessel mask (B1), and with the run's results left on the template (B2).
  - The optimiser's chosen settings with A2's pass-through settings at a non-default value
    (`remove_short_loops_under_um=0`, `recover_uncovered_mask_vessels=False`).
- Check Harvey's batch (processes, free RAM) before any run over 2 GB.
- Record the numbers in this file under **Measured**, with commit and dataset.

### Phase 1 — Shared argument builders (pipeline, no behaviour change)

- In `pipeline/stages.py`, beside `graph_build_arguments` (D1): public
  `segmentation_cleanup_arguments(settings)`, `skeleton_cleanup_arguments(settings, downsample)`
  (the `skeleton_` prefix, the percentage, the bridge gates, tiling, and
  `skeleton_voxel_settings_on_grid`), `thick_skeleton_arguments(settings, binary, voxel_size_xyz,
  crop, downsample)` (today's `_thick_region_arguments` plus the refinement knobs and the
  large-stack tree), `smoothing_arguments(settings)` and `lumen_cleanup_arguments(settings, ...)`.
  `skeletonise` and `build_network` call them. (`lumen_cleanup_arguments` was not built: it has
  one caller, see **Finishing (2026-10-24)**.)
- `search.py` drops `_cleanup_kwargs`, `_skeleton_kwargs`, `_thick_vessel_kwargs` and
  `_graph_kwargs` and calls these. Fixes A3.
- Tests: every argument `skeletonise` / `build_network` pass reaches a trial (monkeypatch the
  step, compare the keyword arguments of a run and of a trial at the same settings).

### Phase 2 — Every setting reaches the search; downsample on import; the skeleton a run makes (A1, A2, A5, A9, A10)

- **A5 first** (Phase 0: the largest drift, and cheap): the panel builds the run's restriction
  mask (`stages._thick_region_arguments(...)["restrict_thick_to_mask"]`, made public) and passes
  it in; the search reduces it onto its grid by block majority and hands it to every thick
  trial. Pinned by the Phase 0 check: with it, a trial's skeleton equals the run's on E14.5
  (0 voxels differ) and the thick trials are 4x faster.
- **A9**: with the cleanup group deselected, the search cleans the mask whenever a run would
  (`segmentation_cleanup` and a toggle on).
- **A10**: the plain trial skeleton goes through `io.load._skeletonize_loaded_volume`, as a run's
  does. Pinned by the Phase 0 check: the run's raw skeleton through the trial's clean-up equals
  the run's saved skeleton (0 voxels differ).

- The panel passes the run's whole settings as `starting_values`; the search keeps deciding only
  `OPTIMISE_SETTING_NAMES`. The function's docstring and `OptimisationResult` say so.
- Downsample on import: the search takes the import `Downsample` record and puts the starting voxel
  counts on the import grid by `skeleton_voxel_settings_on_grid` (one rule) before its own
  `_to_search_grid`; on the way out, a moved value goes back to voxels of the image as read, a kept
  one is returned as the user had it (`_to_full_resolution`'s rule, through the combined factor).
- The report lists the pass-through settings that differ from their defaults, so a reader knows
  what the search was judged under.
- Tests (`test_optimisation_search.py`, `test_downsample_on_import.py`): at 2x import, a closing
  radius of 2 is searched as 1 and comes back 2 when kept; a moved value comes back in source
  voxels; with `remove_short_loops_under_um=0` a trial graph keeps a short loop it otherwise
  breaks; the panel's `starting_values` hold every schema setting.

### Phase 3 — FWHM trials measure what the run measures (B1-B7)

Started 2026-10-23 in its own session ("Fix FWHM optimiser bugs (Phase 3)"), with B7 sent on
after Phase 0 found it.

Independent of Phases 1-2.

- **Sample:** from the vessels the run measures: no `IS_ZERO_RESISTANCE` bridge, and only
  `edges_network_handling_keeps` (the panel has the boundaries checkpoint). One public helper,
  used by `assign_edge_diameters` and the optimiser. The PSF estimate takes the same edges.
- **Fresh template:** before any trial, the sample is cleared of the run's measurements by the
  pipeline's own rule (`_start_measurement_afresh`, made public), keeping or re-measuring
  `edt_diameter_um`, which the decoys and the diameter guess read. Fixes B2.
- **One call:** trials measure through the run's `_measure_fwhm_diameters` (made public), so they
  get the vessel mask, memmap, voxel size and axis order as a run does. Fixes B1, B4.
- **In the whole graph:** a trial measures the sampled vessels inside the whole vessel graph
  (`edges=`), never a subgraph of them, so each sees the neighbours a run sees. Fixes B7.
- **Switches:** honour `fwhm_decoy_check_sample_size`; decoys and demotion as D3 decides.
- **Report:** the share of sampled vessels FWHM leaves to the raw-section fallback when that is on
  (B6), measure only.
- Tests (`test_fwhm_optimisation_search.py`): trial widths `==` `assign_edge_diameters`' widths on
  the same edges and settings, mask stop on; a template carrying run widths on an edge that fails
  scores it failed and leaves it out of the decoy list and the regression guard; bridges and
  removed vessels never sampled; the decoy sample size followed.

### Phase 4 — The graph a trial scores is the run's (A4)

The largest change; it alters what the graph sweeps choose.

- Split `build_network`'s graph work into two public functions it calls, with no change to its
  output (E14.5's graph identical before and after, by the reproduction recipe, plus the suite):
  - `network_context(settings, binary_mask, voxel_size_zyx, thick_vessel_mask, ...)`: the
    `radius_at` sampler, the run's one `MaskSupport`, the thick centreline tree and
    `along_thick_centreline`, built once.
  - `finish_network_graph(G, skeleton, settings, context)`: smoothing (with `radius_at`), the
    clean-up after it, the thick-vessel split and the clean-up after that, the retrace, straight
    steps, the optional interior dead-end removal and junctionless pieces, dead-end kinds.
- The search builds the context once per mask (again after segmentation cleanup moves it), passes
  `mask_support` and `along_thick_centreline` into graph building, and scores every graph group on
  `finish_network_graph(build(...))`. The smoothing group sweeps the smoothing settings through the
  same finish. The trial cache keys on the settings the finish reads.
- The thickness-gated trial skeleton returns its thick region (`return_thick_mask=True`), which
  feeds the finish's split (the restriction mask itself arrives in Phase 2).
- Auto's time model gains a timed "finish" step and `_SEARCH_STEP_CALLS` is counted again on
  E14.5 at 4x and 16x (the test pinning the model's arithmetic is updated, its rule kept). (The
  finish was counted at 2x with Phase 4; 4x and 16x under **Finishing (2026-10-24)**.)
- Scorecard, measures only (Principle 4): dead ends by kind, loops inside one lumen, the
  lumen-artefact report's counts. (Dead ends and pairs in one lumen with Phase 4; the kinds and
  loops inside one lumen in `95941900`.)
- Tests: on `tests/dense_capillary_fixtures.py` and a thick-vessel fixture, a trial's graph equals
  `build_network`'s at the same settings (node positions, edge voxels, lengths); a thick vessel
  restricted by a large mask gives the trial the run's region.

### Phase 5 — Panel, low RAM, words (A6-A8)

- Grey **Optimise settings** under `network_source=synthetic` and in batch mode, and **Optimise
  FWHM settings** under synthetic, each with its reason in the hover text (`chrome_tooltips.py`;
  hover text, not the help menu). With ilastik on and no `input_path`, use the ilastik output
  when `ilastik_output_is_current` says it is the current one; never run ilastik from the button.
- Low RAM: under `use_memmap_loading` the mask is binarised and reduced without a whole in-RAM
  copy (check `block_reduce_volume` on a memmap first), the raw image is read through
  `raw_volume_cache`, and "Off" (factor 1) is refused with a message when the mask will not fit
  in free memory.
- Words: `optimisation/__init__.py`'s dependency list and group counts; CLAUDE.md's
  *Optimisation* entry gains its rule ("trials go through the pipeline's own argument builders
  and `finish_network_graph`; a new step there reaches the optimiser for free"); the history in
  `docs/notes/gui.md`.
- Tests: `test_gui_optimise_settings_widget.py`, `test_gui_fwhm_optimise_widget.py` (greying,
  the ilastik output used), `test_gui_tooltips.py`.

**Done (2026-10-23, `399a48f3` and `924b9910`, merged as `add3241f`; built in its own session
beside Phases 1 and 4; A6, A7 and A8, the CLAUDE.md rule since 2026-10-24):**
- **A7.** Both optimisers grey under synthetic (`graph.network_is_synthetic`), hover text and
  report line `chrome_tooltips.OPTIMISE_SYNTHETIC_TOOLTIP`; batch greying was already in
  (`d4bdfd0c`). `_widget.py`'s `refresh_single_image_buttons` now holds both reasons, runs from
  `apply_prerequisites` on every change of settings, and gives back only the buttons it greyed (a
  run's own greyed button stays greyed). With ilastik segmenting, Optimise settings optimises on
  `stages.current_ilastik_output(settings)` (new, beside `segmented_input_path`: the output's
  sidecar against this run's project, raw image and vessel label) and otherwise refuses, giving
  the reason; `segment` is handed the output (`segmented_path=`), so ilastik never starts. Found
  doing it: before, with ilastik on *and* an `input_path` set, the button's `segment` call ran
  ilastik. A named `input_path` is not used while ilastik segments (a run would analyse ilastik's
  output, not it).
- **A6.** The worker's body is a plain function, `_optimise_settings_for_run`. Under low RAM it
  binarises onto disk and reads the raw image onto disk the way the stages read it
  (`load_single_channel_tiff_volume(use_memmap=True)`, crop and `downsample_raw_volume` on disk,
  superseded copies released). `raw_volume_cache` holds only the segmented image (no channel),
  which `load_volume_on_run_grid` already reads through it; the stages read the raw image as
  above, so that is what "as the pipeline does" came to. `block_reduce_volume` already read a
  memmap a slab at a time; `_downsample_mask` and `_downsample_restriction` (which no longer casts
  first) go straight through it, and `_downsample_intensity`, which cast the whole raw image to
  float64 first (8 B a voxel), casts one plane of blocks at a time, equal bit for bit. "Off" under
  low RAM is refused before binarising when `search.full_resolution_search_refusal` says the mask
  will not fit (`FULL_RESOLUTION_SEARCH_BYTES_PER_VOXEL`, 128 B a voxel, see **Measured**); Auto is not capped.
  Pinned in `test_low_ram_option.py`: on the dense capillary bed with a raw image, low RAM hands
  the search memmaps equal to the in-RAM arrays and the result, every trial and the scorecard are
  the same.
- **A8.** `optimisation/__init__.py`'s dependency list (`io.downsample`, `io.load`; never
  `pipeline` or `gui` -- Phase 1 will change that) and group counts, `search.py`'s "21 settings"
  and "all ten". `fwhm_search.py` already said eight. `_SEARCH_STEP_CALLS` left for Phase 4.
  History in `docs/notes/gui.md`.
- **CLAUDE.md's *Optimisation* rule** landed with the finishing pieces (2026-10-24): trials go
  through the pipeline's argument builders, `network_context` and `finish_network_graph`.
- **Out of this plan's scope, listed:** "Check segmented image" still calls `segment`
  (`gui/_widget.py`'s Check button), so runs ilastik when it is on, and binarises the segmented
  image in RAM whatever `use_memmap_loading` says.

## Order

0 → 1 → 2 → 3 → 4 → 5. Phase 3 needs nothing from 1-2 and can run beside them in its own session.
Phases 1-3 fix plain bugs and are cheap; Phase 4 is where the chosen settings move, and goes
last so its effect is measured on its own. After Phase 0 (2026-10-23): Phase 0 is done, Phase 3 is
under way in its own session, and A5 in Phase 2 is the next most valuable change. After Phase 2
(pushed as `f1184f25`): the search still turns thick vessels off on E14.5, now because of what the
on/off sweep scores (D6), not A5. After 2026-10-24: every phase is done (Phases 1 and 4
`e1314748`, Phase 5 `399a48f3`); what is left waits on Harvey or a large run (see Status).

## Verification

- After each phase: `pytest -m "not slow and not gui"` (13-28 min here), the touched GUI tests in
  the background, the slow optimiser tests, on the final source (no `src` edits mid-run).
- The Phase 0 harness again after Phases 2, 3 and 4: trial graph `==` `build_network`'s at factor
  1; FWHM trial widths `==` the run's; the chosen settings before and after, reported with how
  much each moved and the scorecard.
- `build_network`'s E14.5 graph unchanged by Phase 1 and the split in Phase 4.

## Decisions needed (Harvey)

**Decided 2026-10-23 (Harvey: "go with recommended"):** D1 `pipeline/stages.py`, the optimisers
import it; D2 keep block max, say so in the report; D3 confirmed as Phase 3 built it; D4 yes,
an optimiser for the endothelial ring fit that works on any endothelial dataset (under way in its
own session); D5 accept slower Phase 4 searches; D6 calibrate on a marked grid. Order: verify
Phase 3 on E14.5, then Phase 1, then Phase 4 here; Phase 5 in its own session meanwhile; the
grids built for Harvey to mark.

- **D1. Where the shared builders live.** Recommend `pipeline/stages.py`, where
  `graph_build_arguments` already is; the optimisers then import the pipeline (no cycle: the
  pipeline does not import them). The alternative, a light module both import, keeps the
  optimisers' import small but moves code out of the stage it belongs to.
- **D2. The search's own reduction.** It keeps block max (`docs/notes/downsample_and_crop.md`)
  while import uses block majority. Recommend keeping max (the search finds settings, it is not a
  run) and saying so in the report; or match import's majority.
- **D3. Decoys and demotion in FWHM scoring.** Recommend scoring with both on whatever the run's
  switches say (a loose setting must never win by measuring specks), honouring the sample size,
  and saying so in the report; or follow the run's switches.
- **D4. An optimiser for the endothelial ring fit.** Asked for by Harvey (2026-10-23), built in its
  own session: see **The endothelial optimiser** at the end of this file.
- **D5. Cost of Phase 4.** If the finish makes a 4x E14.5 search more than twice as slow (Phase 0
  says), Auto will pick coarser grids. Recommend accepting it and letting the time model say so;
  the alternative, finishing only some groups' trials, breaks Principle 2.
- **D6. What the thick-vessel on/off sweep scores** (found re-running R1 after Phase 2). It keeps
  whichever raw skeleton has the larger connected share, so on E14.5 thick vessels go off by 0.966
  against 0.953 even with the run's restriction. Keep it; or calibrate a measure on a marked grid
  of thick vessels (a sheet or a centreline), which is what the toggle is for.
  **Grids built 2026-10-24, awaiting Harvey's marks** (git-ignored, with their keys):
  - `outputs/thick_vessel_review/` (`harness/thick_grid.py`): 30 places on E14.5's restricted
    thick region, spread over the local radius and at least 40 um apart, the thick path's skeleton
    and plain Lee's as A and B in random order, a 4 um slab. Marks: A, B, = or X. Candidate
    measures in the key: each skeleton's voxels, junction voxels and braid factor in the region.
  - `outputs/component_filter_review/` (`harness/island_grid.py`): 30 of the 75 pieces
    `skeleton_min_component_percent` = 1 drops under R1's choices, spread over size. Marks: V
    (vessel, keep) or N (noise). Candidate measures in the key: skeleton voxels, the mask
    component's volume, distance to the kept skeleton, whether it touches a face, raw contrast
    (recomputed 2026-10-24: it was the raw mean times a million, see **Finishing**).

## Risks

- Phase 4 makes trials slower; on big stacks Auto then resolves fewer vessels.
- The chosen settings move once trials see the finish; a user's earlier optimised config will
  differ from a new one. The report says which settings moved and why.
- Peer sessions edit this checkout: targeted edits, commits through a private index (memory).
- No `RELEASE_REVIEW.md` row scores the optimisers today; this plan changes no score.

## Out of scope

Optimising once for a whole batch; new objectives on dead ends or
loops before a marked grid; sweeping settings that marked grids fixed; help-menu text.

## Measured

### Phase 0 (2026-10-23, commit `405364d7`, E14.5 MCA)

From a `git archive` of `405364d7` (`outputs/optimiser_alignment/code`), with the harness in
`outputs/optimiser_alignment/harness/` (git-ignored; `summarise.py` prints these tables from the
JSON beside each run). The data is the repo-root `config.yaml` as it stands: Claudin `Simple
Segmentation` mask (287 x 512 x 512, 7.31 M vessel voxels), voxel (z, y, x) = (1.0, 0.98, 0.98) um,
thick vessels on and restricted to the large masks (`skeleton_thick_vessel_restrict_to_mask:
large`), cleanup off, no downsample or crop. The machine was shared with Harvey's idle panel, a
peer's pytest and, from 12:40, the Phase 3 session's suites, so timings are upper bounds.

**The graph (A2-A5).** The run (`segment`, `skeletonise`, `build_network`) against one optimiser
trial fed as the panel feeds it today (the 54 swept settings; `_raw_skeleton`,
`preprocess_skeleton_for_graph`, `build_graph_from_skeleton`, smoothing), and the trial's graph code
on the run's own skeleton:

| measure | run, after build | run, saved | trial | trial code on the run's skeleton |
|---|---|---|---|---|
| edges | 1,370 | 1,406 | 573 | 1,352 |
| centreline um | 52,006 | 47,813 | 29,857 | 47,506 |
| components | 54 | 53 | 75 | 54 |
| dead ends | 456 | 396 | 399 | 447 |
| junctions | 705 | 706 | 247 | 699 |
| run's skeleton traced | 0.669 | 0.660 | 0.103 | 0.663 |
| mask covered | 0.808 | 0.822 | 0.679 | 0.817 |
| short loops | 68 | 89 | 1 | 82 |
| pairs in one lumen | 11 | 15 | 0 | 2 |
| thick-vessel bridges | 0 | 64 | 0 | 0 |
| dead ends: unknown / stops_short / island / leaves image / artefact / broken | 112 / 120 / 96 / 95 / 28 / 5 | 94 / 96 / 94 / 91 / 17 / 4 | 66 / 110 / 132 / 83 / 8 / 0 | 98 / 117 / 96 / 92 / 39 / 5 |

Within 2 um: 45% of the run's saved centreline lies near the trial's (72% the other way); 96-97%
near the trial code's graph on the run's skeleton (both ways).

- **A5 is the whole skeleton drift.** The trial's skeleton has 25,641 voxels to the run's 56,961,
  70,678 of them differing. Unrestricted, the thick region a trial finds is 7.30 M voxels, **99.8%
  of the mask**; the run's, restricted, is 0.88 M (12.0%). Made again with the run's restriction
  mask (`stages._thick_region_arguments`), the trial's skeleton equals the run's: **0 voxels
  differ**, in 160 s against 704 s unrestricted. So on E14.5 every skeleton and graph sweep today
  judges a centreline tree drawn through the whole mask.
- **A3/A4 on the same skeleton** are smaller but real: the saved graph has 51 fewer dead ends (396
  against 447), 54 more edges, 64 bridges and 7 more short loops. The run's own steps after graph
  building, in order (edges / um / dead ends): build 1,370 / 52,006 / 456; smoothing 1,370 / 47,787 /
  456; clean-up 1,321 / 47,303 / 436; thick-vessel split 1,393 / 47,110 / 398; clean-up 1,356 /
  46,833 / 397; retrace 1,406 / 47,935 / 396; smoothing of the retraced 47,831; straight steps
  47,813. The split alone removes 38 dead ends, which is what the stub sweep scores on.
- **A1** (arithmetic, on this config): with a 2x downsample on import, a run uses
  `skeleton_bridge_gap_size` 2, `skeleton_min_branch_length` 2, `skeleton_max_bridge_distance` 2,
  `skeleton_bundle_scan_size` 5 and `skeleton_bundle_hub_min_spacing` 2 (set: 3, 3, 4, 9, 4); the
  optimiser would search them as set. `skeleton_closing_radius` 1 stays 1.

**FWHM (B1-B3, B7).** Dextran (channel 0) read as a plasma column, so FWHM runs; the run's Diameters
call (`apply.assign_edge_diameters`, 504-635 s) measured 470 of 1,406 edges. The optimiser's
200-edge sample of that graph, measured as a trial does today (T0) and with each fix:

| trial | measured by both | trial only | equal to the run | within 1% | p90 error | max error |
|---|---|---|---|---|---|---|
| T0 today: run results left on, no mask, PSF from every edge | 47 | 11 | 21 | 27 | 36% | 62% |
| T1 run results stripped | 47 | 11 | 21 | 27 | 36% | 62% |
| T2 + vessel mask | 47 | 12 | 25 | 28 | 39% | 62% |
| T3 + the run's PSF | 47 | 12 | 25 | 28 | 39% | 62% |
| T4 the sample measured inside the whole vessel graph (`edges=`), mask, run's PSF | 47 | 0 | **47** | 47 | 0 | 0 |
| T5 as T4 without the mask | 47 | 0 | 35 | 42 | 1% | 23% |

- **B7 is the largest FWHM drift**: measured as a subgraph of its own, a sample matches the run on
  25 of 47 widths and gives 12 vessels the run fails a width; inside the whole graph, all 47 match
  and the 12 fail, at the same cost (34 s against 33 s).
- **B1** on its own (T5 against T4): 12 of 47 widths change, by up to 23%.
- **B2** shows only once a candidate fails vessels the run measured: with `fwhm_min_fit_r2` 0.995
  the trial measures 0 vessels, yet today hands the decoy report 47 "measured" vessels and the
  regression guard 212 widths, all the run's; stripped, 0 and 0.
- **B3**: 29 of the 200 sampled edges are bridges (14.5%, against 4.6% of the graph): they carry no
  `edt_diameter_um`, so `_sample_buckets` puts them in its no-width bucket, which gets an equal
  share. None is measured, so they only take sample slots. The PSF is the same with or without
  them here (sigma z, y, x = 2.47, 1.01, 1.01 um).

**What Optimise settings chooses today (R1)**, as the panel calls it: Auto chose 2x (the resolution
cap; typical radius 5.3 um), estimated 703 s and took 2,643 s (3.8x; the machine was loaded). It
moved 9 settings, among them `use_thick_vessel_skeletonisation` True -> False (consistent with A5:
unrestricted, the thick skeleton is a tree through the whole mask, so off scores better), cleanup on
with small volumes under 46 um3 removed, the bundle scan 9 -> 18, `cluster_collapse_method`
direction_aware -> distance_only, and `skeleton_min_component_percent` 0 -> 1. Its own scorecard:
graph components 62 -> 1, input vessels the skeleton reaches 0.58 -> 0.016, skeleton the graph
traces 0.97 -> 0.74, mask the skeleton runs through 0.41 -> 0.56. The component sweep maximises the
largest connected share and guards only a second component of 80% of the largest, so it drops every
island; whether E14.5's islands are noise or vessels is a marked-grid question, not an alignment
one, and is left as an observation. Slowest groups: stub length 658 s, collapse 465 s, gap
bridging 393 s, reconnect 373 s.

**A2 on the chosen settings (R2)**: the same search given the run's whole settings, for a user with
`remove_short_loops_under_um` 0 and `recover_uncovered_mask_vessels` off (which today's panel
never passes, so that user gets R1). It chose the same settings but one:
`min_stub_length_radius_multiple` 2.0 where R1 kept 3.0 (2,167 s, estimated 477 s).

**The graph again, at R1's chosen settings** (thick vessels off, cleanup on): the trial made as
the search makes it, on the cleaned mask (equal to the run's). The skeletons still differ, by
25,481 voxels (trial 63,541, run 53,832): **A10**. On the same cleaned mask, whole-volume and
per-component Lee agree exactly (103,809 voxels), and the run's hole fill adds 50,441 (+49%); the
run's raw skeleton put through the trial's clean-up equals the run's saved skeleton (0 differ).
The graphs (run saved / trial): edges 1,198 / 1,146; um 43,001 / 42,258; dead ends 228 / 244;
short loops 110 / 81; mask covered 0.795 / 0.786; 90-92% of each within 2 um of the other. The
run's steps after building (no split, thick vessels off): build 1,172 edges / 45,921 um / 246 dead
ends; smoothing 42,192 um; clean-up 1,128 / 41,689 / 228; retrace 1,198 / 43,145; smoothing and
straight steps 43,001.

### Phase 2 (2026-10-23, pushed as `f1184f25`, E14.5 MCA)

A1, A2, A5, A9 and A10 done. A trial set up as the panel now sets it up
(`stages.load_volume_on_run_grid`, the run's whole settings, voxel counts on the run's grid,
`stages.thick_vessel_restriction_mask`, the cleanup when it is in effect), against the
skeletons the Phase 0 runs saved (`harness/phase2_check.py`, on the worktree's source):

| config | run's skeleton | trial's | differing | trial |
|---|---|---|---|---|
| `run_base`: thick vessels on, restricted to the large masks | 56,961 | 56,961 | **0** | 125 s (704 s unrestricted) |
| `run_chosen`: R1's settings, thick vessels off, cleanup on | 53,832 | 53,832 | **0** | 97 s |

Not done from Phase 2: the report's list of pass-through settings that differ from their
defaults; it needs the fallback table that Phase 1's shared builders give, so it moves there.

### Phase 1 (2026-10-23, on `1b7fdf04`, landed as `e1314748`, E14.5 MCA)

Done (A3; the deferred report line). `pipeline/stages.py` gains the shared builders
(`segmentation_cleanup_arguments`, `segmentation_cleanup_in_effect`, `skeleton_cleanup_arguments`,
`thick_region_settings`, `thick_vessel_refinement_arguments`, `centreline_smoothing_arguments`, beside
`graph_build_arguments`); `skeletonise` and `build_network` call them, and the search's four copies
are gone (D1: the search imports `pipeline.stages`). A setting a caller leaves out is read as its
schema default, and the report names the settings the trials read but the search does not tune
that are not at their defaults. Graph trials are held in RAM whatever `use_memmap_loading` says.

- `tests/test_optimisation_search.py`: a real `skeletonise` + `build_network` beside a search's
  trials, with six pass-through settings off their defaults: every step (cleanup, thick skeleton,
  skeleton clean-up, graph building) gets the same setting arguments.
- E14.5 (`harness/phase1_check.py`, the worktree's source): `skeletonise` and `build_network` made
  again against Phase 0's run: skeleton 56,961 voxels, **0 differing**; graph 1,228 nodes / 1,406
  edges / 47,812.6581 um, **every edge identical**.

### Phase 4 (2026-10-23, on `1b7fdf04` with Phase 1, landed as `e1314748`, E14.5 MCA)

Done (A4). `pipeline/stages.py`: `NetworkContext` / `network_context` (the radius sampler, the
one `MaskSupport`, the thick centreline tree and what the clean-ups keep its strand by, made once
per mask) and `finish_network_graph` (smoothing by the vessel radius, the clean-up after it, the
thick-vessel split and the clean-up after that, the re-trace, straight steps, the optional
interior dead-end and junctionless-piece removals); `build_network` calls both. The search: a thick
trial keeps its region and tree as `skeletonise` does; graphs are built through the same context
and finished, cached at two levels (graph building by its arguments, the finish by the whole
settings); every graph sweep scores the finished graph, the reconnect sweep's gap candidates
still coming from graph building's own graph; the smoothing sweep counts each edge's
`centreline_smoothing` outcome on the finished graph, its length shrink measured against graph
building's. The scorecard measures finished graphs and adds dead ends, short loops and pairs of
edges in one lumen (measures only). Auto's time model times a finish ("finish" calls counted below).

- `tests/test_optimisation_search.py`: on a thick-vessel fixture and a plain one, a trial's graph
  `==` `build_network`'s, edge for edge (end positions, length, voxels, bridges).
- E14.5 (`harness/phase4_check.py`): `build_network` still makes Phase 0's graph (**identical**,
  1,228 / 1,406), and a trial set up as the panel sets it up scores that graph (**identical**; 263 s
  for the trial's skeleton, clean-up, graph and finish at full resolution).
- **Optimise settings on E14.5 with Phases 1 and 4** (`harness/optimiser_phase4.py`, as the panel
  calls it, Auto 2x), against `f1184f25`: the same choices but these --
  `centreline_max_deviation` 1.0 -> **1.96** and `centreline_smoothing_iterations` 10 -> **5** (the
  smoothing sweep now judges the saved graph, with the run's radius tolerance), and
  `cluster_collapse_persistence_search_multiple` kept at 3 (`f1184f25` moved it to 5). Thick
  vessels still go off (D6) and one component is still kept. Scorecard, start -> chosen: skeleton
  the graph traces 0.712 -> 0.809; the new measures dead ends 288 -> 183, short loops 69 -> 69,
  pairs of edges in one lumen 15 -> 2. **2,608 s against 1,356 s** before Phase 4 (1.9x; D5's limit
  was 2x), the smoothing sweep 891 s of it (40 finished graphs; 40 s before). Auto estimated
  4,110 s with the provisional counts; counted, a search builds 19 graphs and finishes 57, which
  `_SEARCH_STEP_CALLS` now holds.

### Phase 3 checked (2026-10-23, `1b7fdf04`, E14.5 MCA)

`harness/fwhm_end_to_end.py`: dextran (channel 0) read as a plasma column, the run's widths from
its own Diameters call, then the panel's FWHM optimiser call on the old (`405364d7`) and the
fixed (`1b7fdf04`) code.

- **A trial is the run.** One trial at the starting settings on its 200-vessel sample: 56 vessels
  measured by both, **56 equal** (largest difference 0), none measured by one alone, no bridge
  sampled; a strict candidate (`fwhm_min_fit_r2` 0.995) leaves no failed vessel with a width.
- **What Optimise FWHM settings chooses (Auto, every group, one pass):**

| setting | start | chosen at `405364d7` | chosen at `1b7fdf04` |
|---|---|---|---|
| `fwhm_clip_re_rise_fraction_of_center` | 0.08 | 0.05 | 0.05 |
| `fwhm_diameter_guess_um` | 4.0 | None | None |
| `fwhm_min_fit_r2` | 0.85 | 0.85 | 0.90 |
| `fwhm_min_total_extent_multiplier` | 3.0 | 3.0 | 2.0 |

  Scorecard, start -> chosen: vessels with a usable width 0.3 -> 0.3 (old) and 0.1 -> 0.1 (fixed);
  mean fit R2 0.902 -> 0.898 and 0.907 -> 0.943; width spread along a vessel 0.343 -> 0.286 and
  0.182 -> 0.200; no decoy given a width in either. Old 140 s (estimated 752 s), fixed 209 s (896 s).
- **But Auto sampled about 10 vessels in both (B8)**, so these choices rest on 1-3 measured vessels
  and the accuracy group never ran: the comparison says the fixed code measures as the run does,
  not that its choices are better. `harness/fwhm_auto_probe.py`: Auto 10 vessels, 0 planted, 4
  decoys; a fixed 200, 43 planted, 53 decoys.

### Optimise settings again, at `f1184f25` (2026-10-23, E14.5 MCA)

R1 run again as the panel now runs it (`harness/optimiser_rerun.py`, snapshot `../c2`): the same
config, raw image (Claudin, read as endothelial) as the cleanup guard, Auto, every group, one
pass; now with the run's grid, its whole settings and its thick-vessel restriction (444,606
voxels). Against Phase 0's R1 (`harness/compare_r1.py`; settings either run moved):

| setting | starting value | chosen at `405364d7` | chosen at `f1184f25` |
|---|---|---|---|
| `use_thick_vessel_skeletonisation` | True | False | False |
| `segmentation_cleanup` / `_remove_small_volumes` | off / off | on / on | on / on |
| `segmentation_cleanup_remove_small_min_volume_um3` | 5 | 46.1 | 46.1 |
| `skeleton_bundle_scan_size` / `_hub_min_spacing` | 9 / 4 | 18 / 9 | 9 / 4 |
| `skeleton_bundle_max_connections_per_hub` | 8 | 12 | 12 |
| `skeleton_min_component_percent` | 0 | 1 | 1 |
| `cluster_collapse_method` | direction_aware | distance_only | persistence |
| `cluster_collapse_persistence_search_multiple` | 3 | 3 | 5 |
| `min_stub_length_radius_multiple` | 3 | 3 | 2 |

| scorecard | start, `405364d7` | chosen, `405364d7` | start, `f1184f25` | chosen, `f1184f25` |
|---|---|---|---|---|
| mask score (0-10) | 5.995 | 6.019 | 5.995 | 6.019 |
| input vessels the skeleton reaches | 0.577 | 0.016 | 0.504 | 0.016 |
| input mask the skeleton runs through | 0.407 | 0.564 | 0.618 | 0.562 |
| largest connected share of the skeleton | 0.903 | 1.0 | 0.961 | 1.0 |
| skeleton the graph traces | 0.965 | 0.740 | 0.693 | 0.797 |
| graph components | 62 | 1 | 47 | 1 |
| self-loops, cartwheel hubs | 0, 0 | 0, 0 | 0, 0 | 0, 0 |

- **The starting end is now the run's own.** At `405364d7` the scorecard's start was measured on the
  unrestricted thick skeleton: 0.407 of the mask run through, against 0.618 for the run's skeleton.
- **Thick vessels still go off, but no longer because of A5.** The on/off sweep scores only the
  largest connected share of the raw skeleton: confined as the run confines it, on 0.953 and off
  0.966 (Phase 0, unconfined: 0.884 and 0.961). That measure does not see what the thick path is
  for, one centreline through a fat vessel instead of Lee's sheet; changing it is a question for
  a marked grid (Principle 4, D6), so it is recorded, not changed.
- **The graph-side choices moved**: collapse by persistence (search multiple 5) where Phase 0 chose
  distance_only, a stub multiple of 2 where Phase 0 kept 3, and bundle refinement keeping its scan
  size and hub spacing where Phase 0 doubled them. With thick vessels off these trials start from
  the run's own plain skeleton, holes filled (A10), built with the run's settings (A2).
- **Still one component kept**: `skeleton_min_component_percent` 0 -> 1 takes the graph from 47
  components to 1 and the input vessels the skeleton reaches from 0.50 to 0.016, as in Phase 0.
  Unchanged observation; whether E14.5's islands are vessels is a grid question.
- **Time**: 1,356 s against Auto's estimate of 1,229 s (Phase 0: 2,643 s against 703 s); thick
  gating 13 s (72 s). Phase 0's run shared the machine with the Phase 3 session's suites, this one
  with a peer's small pytest, so not all of the difference is Phase 2's.

### Phase 5 (2026-10-23, landed as `399a48f3` + `924b9910`, the dense capillary bed fixture)

What a full-resolution search ("Off") holds in RAM, for `FULL_RESOLUTION_SEARCH_BYTES_PER_VOXEL`:
`optimise_skeleton_and_graph_settings(downsample_factor=1)`, every group, schema defaults, under
`tracemalloc` (numpy's and Python's allocations only), on `tests/dense_capillary_fixtures.py`'s
bed (11% vessel) and on it mirrored to 4x and 8x its voxels (script in the session scratchpad):

| mask | voxels | traced peak | per voxel | search |
|---|---|---|---|---|
| bed | 0.29 M | 56.0 MB | 195 B | 55 s |
| bed x4 (y, x) | 1.15 M | 121.5 MB | 106 B | 209 s |
| bed x8 (y, x, z) | 2.30 M | 266.7 MB | 116 B | 427 s |

The peak grew 76 B a voxel from the first to the second and 126 B from the second to the third,
so it is not yet flat; the constant is 128, rounded up from the larger. Not measured on a stack
the size the refusal is for (E14.5 is 75 M voxels: about 9.6 GB at this rate); a first E14.5
"Off" run under low RAM would check it.

**Two pipeline findings outside this plan:**
- A run that reuses its saved skeleton (`do_skeletonize` off) gets the segmented image uncleaned
  (`stages.skeletonise`, the loaded-skeleton branch, reads `_load_and_crop_volume` and never
  applies `segmentation_cleanup`), so with cleanup on its `build_network` reads a different mask
  from a fresh run with the same settings.
- `io.load._skeletonize_loaded_volume`'s docstring calls the hole fill "a no-op on the normal
  case"; on E14.5's cleaned Claudin mask with plain Lee it adds 49% to the raw skeleton. Whether
  that is right is a question for a grid, not this plan.

### B8 (2026-10-24, landed as `ca5df387` + `aee7da34`, E14.5 MCA)

`harness/fwhm_end_to_end.py --code c4` (a `git archive` of `4724f6ce`, the 100-vessel floor alone)
and `harness/fwhm_b8_probe.py`, as Phase 3's check: dextran (channel 0) as a plasma column, the
panel's call, Auto, every group, one pass. Harvey's idle panel was the only other load.

- **A trial is still the run**: on the 200-vessel sample, 56 measured by both, 56 equal.
- **The sample**: Auto takes 100 of 1,342 vessels (10 before); 12 planted, 40 decoys. The
  accuracy group now runs (16 of 77 trials).
- **The time model** (part b). Auto's probe: 1.64 s fixed, 0.12 s a vessel. One trial on the 100:
  19.5-20.9 s, of it 12.0 s for the sample among every vessel, 0.9 s for the 12 planted and 6.6 s
  for the 40 decoys. The search took **1,488 s, 77 trials** (19.3 s each). The old estimate,
  3,153 s, tripled a whole trial for its checks and assumed 119 trials. Now: the measured 77
  trials (`_TRIALS_PER_PASS`), the fixed cost once a trial, and 0.5 check vessels per sampled
  one (`_CHECK_VESSELS_PER_SAMPLED_VESSEL`; 52 at 100 sampled, 96 at 200): **1,530 s**, pinned in
  `test_auto_estimate_matches_the_e14p5_search_it_was_counted_on`. The budget is still 180 s, so on
  E14.5 Auto always takes the 100-vessel floor, and the panel's search takes about 25 minutes.
- **What it chose** (start -> chosen): `fwhm_profile_model` blurred_lumen -> gaussian,
  `fwhm_clip_profile_to_single_vessel` and `fwhm_enforce_same_edge_locality` on -> off,
  `fwhm_diameter_guess_edge_attribute` edt_diameter_um -> None, `fwhm_diameter_guess_um` 4 -> None,
  `fwhm_max_plateau_shape_ratio` 0.85 -> 0.75. Scorecard: vessels with a usable width 0.05 -> 0.05,
  mean fit R2 0.909 -> 0.923, profile length reached 2.93 -> 1.13, width spread along a vessel
  0.253 -> 0.200, decoys given a width 0.28 -> 0, **planted vessels' width error 0.535 -> 0.557**.
- **Open, not this fix:** FWHM misreads the planted vessels by about half their width at the
  starting settings and at the chosen ones, and the chosen ones are slightly worse (past the
  scorecard's 0.02 tolerance) though each sweep is guarded: later groups traded accuracy for
  their own terms. Only 5 of 100 sampled vessels end with a usable width. Whether that is FWHM on
  this dextran stack or the planting (`fwhm_planted`) needs looking at before the accuracy term
  is trusted on E14.5.

## The endothelial optimiser (D4, 2026-10-23)

Harvey asked for it: the counterpart of Optimise FWHM settings for a raw image that stains the
wall (claudin-5, CD31, isolectin), where the run reads diameters with the ring fit
(`haemodynamics/endothelial.py`). **It must work on any endothelial dataset**: no constant is
fitted to one stack; every candidate is the image's own or a multiple of the setting's value.
Built in its own session (`65068bc1`, `42c4be53`, `a525ad9c`, `56be0b79`; merged as `6f565ac7`
and `718f326d`).

**What it sweeps** (`optimisation/endothelial_search.py`, `ENDOTHELIAL_SETTING_NAMES`):

| group | settings | candidates |
|---|---|---|
| `ring_gate` (first) | `endothelial_min_ring_contrast` | half to twice its value, and the ring fit's own default (4) |
| `blur_and_wall` | `endothelial_psf_sigma_xy_um` / `_z_um` (one joint sweep), then `endothelial_wall_thickness_um` | only where the rings will not calibrate it on their own, or the settings fix it: unset (the image estimates it), the settings' value, the image's estimate fixed, the ends of its 95% interval; otherwise not swept |

The gate goes first because it also decides which rings calibrate the blur and wall: set too
strict, none do and the ring fit reads nothing (found by the slow tests; at a gate of 32 halving
once still read nothing, so the default is always a candidate). Fixed blur and wall matter where
the rings will not calibrate on their own (they disagree on the blur): the run then measures
nothing, and the image's own values, fixed, let it. **Decided (Harvey, 2026-10-24): fixed values
are tried only then, or where the settings already fix them**; where the rings calibrate, the
blur and wall stay the image's estimate and are not swept (a fixed value inside the estimate's
interval is no different from it; on E14.5 the search moved the wall on one vessel, below).

**Read, never swept:** `edt_junction_proximity_exclusion_um` (the ring fit's junction zones) is
the mask-width setting's and moves those widths too, the fallback of every vessel the ring fit
leaves, and the planted vessels have no junction to judge it by. The sampling along a vessel
(2 um), the averaging length (4 um) and the aggregation are not settings; none was added.
`endothelial_workers` changes no width.

**A trial is the run** (Principle 2): the sample is drawn from `apply.measured_vessels` (no
bridge) and network handling's keeps, read among the whole vessel graph with `edges=` through
`apply.measure_endothelial_diameters` with the run's config, from a template cleared by
`clear_edge_measurements` keeping `edt_` (`fwhm_search._fresh_template`). The calibration is the
run's: `calibrate_from_rings` was split into `ring_calibration_fits` (the section fits, which do
not depend on the gate) and the gate, so each blur's fits are made once
(`apply.endothelial_calibration_fits`) and each (blur, gate) calibrated once
(`apply.calibrate_endothelial_rings`), then handed to the measurement (`calibration=`). Pinned:
a trial's widths `==` `assign_edge_diameters`' on the same vessels, with the blur and wall
estimated, with the blur set, and calibrated on a subset.

**Scored by accuracy:** hollow vessels of known lumen planted beside sampled ones
(`haemodynamics/endothelial_planted.py`), drawn with the image's calibrated blur and wall (at
`image_calibration_gate`: the default gate, or the user's where weaker, so a gate under test
cannot leave nothing to plant with), as bright as the sampled vessel's own ring fits, with the
image's noise, clear of every vessel and of the volume's faces. Score = 2 x planted lumen error
(an unread one counts 1) + the sample's real cost (per vessel: 1 without a width, else its
readings' CV) + decoys given a width (sampled vessels moved into vessel-free tissue). The planted
vessels carry the image's blur and wall, so they favour those two by construction; the real
vessels decide between the estimate and fixed values, and the report says so. On synthetic
stacks the planted vessels read within 6% at four voxel sizes (anisotropic 2 x 0.5 x 0.5 um
included), walls 0.8-1.5 um and blurs; no decoy was read at any gate there (the ring fit's
other gates already reject noise), so decoys are covered on a stand-in ring fit.

**Panel:** "Optimise endothelial settings" on the Diameters tab where the FWHM button is, shown
exactly when the ring fit runs (diameters from the raw image, stain endothelial as set or as a
run read an auto, never for a synthetic network, which it also refuses), with options, review
before Apply, progress bars, cancellation and hover text; greyed in batch mode like the others.

**Auto** times reading with the image's own blur and wall, calibrated once (the search reuses
that calibration), against a ten-minute budget for a pass's trials -- three minutes, FWHM's,
sampled 13 of E14.5's vessels -- and never samples 64 or more: from there every trial starts
worker processes (`run_edge_tasks` keeps no pool between calls), a cost a probe of ten does not
see. The calibrations come on top (up to four blurs a pass, about a minute each on E14.5), and
"Every vessel" on a large network is slow for the same reason.

**Measured on E14.5 MCA (2026-10-24, `42c4be53`, before `56be0b79`'s rule)**, the Claudin channel
(raw image channel 1, endothelial) and Phase 0's run graph (`run_base/G_run.pkl`: 1,406 edges, 64
bridges), alone on the machine; harness `outputs/endothelial_optimiser/e14p5_check.py`
(git-ignored, run on a `git archive` of the commit). The search below still swept the blur and
wall against fixed values where E14.5's rings calibrate them, which `56be0b79` stopped:

- The run's Diameters call: 541 of 1,342 vessels measured in 178 s; the rings calibrated at
  sigma_xy 1.08 um (95% 1.05-1.11), sigma_z 2.77 um (2.68-2.86), wall 1.14 um (1.12-1.16).
- **A trial is the run:** one trial at the starting settings on a 40-vessel sample equals the
  run's widths on all 40 (13 of them measured), to the bit.
- Optimise endothelial settings as the panel runs it (Auto, both groups, one pass): 63 vessels
  sampled (the cap), 60 planted vessels drawn like 24 sampled vessels' rings, 38 decoys. Auto
  estimated 12 min 54 s; the search took 6 min 44 s, 9 min with setup. Each calibration at a
  fixed blur costs ~83 s.

| setting | start | chosen | why |
|---|---|---|---|
| `endothelial_min_ring_contrast` | 4 | 3 | planted lumen error 0.521 -> 0.495, CV along a vessel 0.138 -> 0.087, no decoy read (2 nearly ties with 3) |
| `endothelial_psf_sigma_xy_um` / `_z_um` | unset | unset | every fixed candidate scored worse on both the planted vessels and the real ones |
| `endothelial_wall_thickness_um` | unset | 1.125 | one more of 63 sampled vessels read (0.270 -> 0.286); CV worse (0.087 -> 0.118); inside the image's own 95% interval |

| scorecard | start | chosen |
|---|---|---|
| sampled vessels measured | 0.270 | 0.286 |
| width spread along a vessel (CV) | 0.138 | 0.118 (better) |
| decoys given a width | 0 | 0 |
| planted vessels' lumen error | 0.521 | 0.498 (better) |

About half the planted vessels go unread at every setting: they are drawn like E14.5's narrow
vessels, under a blur along z of 2.8 um, which the ring fit does not resolve (as it leaves 801 of
the real ones to the mask width). They add the same to every trial's score.

The gate's move is backed by both accuracy and agreement. **The wall's is not**: it rests on one
vessel, inside the image's own interval. Before this run (at `65068bc1`, Auto broken, a sample
of 10) the search fixed the blur and wall at other values inside their intervals, on 6 read
vessels -- noise.

**Decided (Harvey, 2026-10-24):** fixed blur and wall candidates only where the rings will not
calibrate on their own, or the settings already fix them; done in `56be0b79`. On E14.5 the rings
calibrate (the intervals above), so under that rule the blur and wall stay the image's estimate
and are not swept: the wall's move to 1.125 um above cannot happen, and only the gate group
searches. **Not yet run on E14.5 under the rule** (2026-10-24: under 6 GB free beside a peer's GUI
suite and Harvey's panel); `outputs/endothelial_optimiser/e14p5_check.py` on a `git archive` of
the current commit is the check, expected to choose the gate alone (4 -> 3 before) at about the
6 min 44 s search above less the fixed-blur calibrations.

## Finishing (2026-10-24)

An audit on 2026-10-24 found these pieces outstanding after Phases 0-5, the endothelial optimiser
and B8. Each, and where it stands:

- **The scorecard's dead ends and loops** (Phase 4's scorecard bullet; `95941900`). Trials never
  marked dead-end kinds: `build_network` marks them after `finish_network_graph`. The marking is
  now public, `stages.mark_dead_end_kinds(settings, G, context)`, which `build_network` calls in
  place of its private call and the scorecard calls on its graphs. The scorecard counts each kind
  (`dead ends: leaves image`, `stops short`, `broken`, `artefact`, `unknown`, `island end`; a
  person's `blind` is never on a search's graph) and loops inside one lumen beside short loops and
  pairs of edges in one lumen. The kinds are measures only (Principle 4): `ScoreRow`'s
  `higher_is_better=None` shows a row that is never better or worse, so they raise no warning.
  Pinned in `tests/test_optimisation_search.py`: the scorecard's counts equal the kinds and
  in-lumen loops of `build_network`'s saved graph, with thick vessels and without.
- **D2** (`95941900`): a search on a coarser grid says in its report that the grid was reduced by
  block max, not by the block majority a run's downsample on import uses. In the search's own
  notes rather than `report.py`'s downsample line, which the FWHM report also prints (see below).
- **Docs**: CLAUDE.md's *optimisation/* entry gains the rule (trials through `network_context` and
  `finish_network_graph`, so a step added there reaches the optimiser for free); `docs/notes/gui.md`
  gains Phases 1 and 4, the endothelial optimiser, B8 and these pieces, under a heading naming the
  three optimisers.
- **`lumen_cleanup_arguments`** (Phase 1's list): **not built, and not needed.**
  `finish_network_graph` is the one place a run's settings become `graph.assemble.lumen_cleanup`'s
  arguments (graph building makes its own inside `build_graph_from_skeleton`, from
  `graph_build_arguments`), and the optimiser reaches that place through `finish_network_graph`
  itself, so there is no second copy to drift. `scripts/graph_artefacts/build_side.py` also calls
  `lumen_cleanup`, from a job's recorded build arguments filtered to what each checked-out version
  accepts: it compares branches on older code, where a `stages` builder would not exist. Build one
  when a second caller of the settings map appears.
- **Principle 4's check: no swept default came from a marked grid.** Each of the 92 settings the
  three optimisers sweep (`OPTIMISE_SETTING_NAMES` 54, `FWHM_SETTING_NAMES` 34,
  `ENDOTHELIAL_SETTING_NAMES` 4) was looked up in `docs/notes/`, every `PLAN_*.md`, CLAUDE.md and
  `RELEASE_REVIEW.md` for a grid or marks within two lines, and the six names with hits read:
  `segmentation_cleanup` (clean-graph D4 and raw-confirmation P5 propose a pinhole option for after
  a grid; nothing fitted), `skeleton_thick_vessel_min_radius_um` (surface-coverage D2, a proposal),
  `skeleton_min_component_percent` (this plan's own grid), `min_stub_length` and
  `min_stub_length_radius_multiple` (grid Y's rule for blind vessels beyond a thick-vessel bridge
  reads them; it did not set them), `centreline_max_deviation` (this plan's A4). The values Harvey's
  grids fitted are all on settings or constants the searches read and never move: the short-loop
  rule and its hole limit (`remove_short_loops_under_um`, `short_loop_keep_hole_over_um`; grids I,
  I2, Q, Q2, N, N2), the re-trace, the thick-vessel rules (grids V, W-Z, Y), the bridges' turn rule
  (grid R), the dead-end kinds (G1-G3, P1) and the interior dead-end removal (G5, L1, L2). Phase 2's
  pass-through keeps each at the user's value. **One for Harvey:** the stub multiple's default of 3
  radii is his decision rather than a grid (`PLAN_one_centreline_per_lumen.md` D2, 2026-10-08: do
  not keep real short sprouts, the 3 x radius rule stays), and the stub sweep moves it: to 2 on
  E14.5 at `f1184f25` and with Phase 4. Whether a decision binds like a grid (take the setting out
  of `GRAPH_SETTING_NAMES`) or the sweep may move it is his call.
- **The hole-fill docstring** (`868ac9f0`): `io.load._skeletonize_loaded_volume` no longer calls
  the fill a no-op; it says what A10 measured. Words only.
- **The two grids** (D6; the component filter): **no marks yet** (checked 2026-10-24: no marks
  file beside either). `outputs/thick_vessel_review/` (marks A, B, = or X) and
  `outputs/component_filter_review/` (V or N) wait on Harvey; once marked, fit them by CLAUDE.md's
  *Marked example grids* and check on a fresh grid. **The component key's `raw_contrast` was
  wrong**: `island_grid.py` divided each piece's mean raw by the median raw outside the mask,
  clamped at 1e-6, and E14.5's Claudin channel is 0 outside the mask (52% of those voxels are 0),
  so the column held the piece's mean raw times a million (1.6e7-2.9e8). Recomputed by
  `harness/island_contrast.py` from the grid's saved skeleton, each ID's piece found again by its
  label with its voxel count and centroid checked: `raw_contrast` is now the piece's mean raw over
  the median raw inside the mask (287), 0.055-1.025 across the 30, with `raw_piece_mean`,
  `raw_component_mean` and `raw_local_background_median` (tissue 2-6 um round its mask component;
  0 for 7 of 30) beside it. IDs, labels and `grid.png` are unchanged; the old key is kept as
  `key_raw_contrast_clamped.csv`, and the folder's README says so.
