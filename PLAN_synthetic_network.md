# Synthetic network from a template run (`network_source=synthetic`)

## Context

HaemoLynx only analyses networks that come from an image. Harvey wants an Input-tab option that
instead **generates a synthetic vascular network with the general characteristics of a previous
run**, read from that run's `.haemorun` file (the *template run*). The template must be a
microscopy-image run (never itself synthetic) saved with enough statistics to describe the
network. Uses: a null model with the same gross properties as a real dataset, a shareable stand-in
for raw data, and a pipeline test-bed with known ground truth.

Decisions already taken with Harvey:
- **Source of the characteristics is a dropdown**: `statistics` reads the template's saved
  statistics report (a missing section fails preflight naming the `statistics_*` toggle to switch
  on and rerun); `graph` reads per-vessel distributions off the template graph in the same file.
- **Route**: the generated graph goes straight to the Boundaries stage. Skeletonise/Graph are
  skipped, the generator chooses the inlets, outlets and arteriole/venule boundary nodes, its
  diameters are kept, and Diameters → Haemodynamics → Post processing → Perturbations → Export and
  Statistics run unchanged. (Rasterising a synthetic mask and running the whole pipeline was
  rejected: slower, and the clean-up rules move the network off its targets.)

## What a `.haemorun` gives us (verified)

- `gui/run_snapshot.read_run_snapshot` is pure Python (no Qt). `RunSnapshot.checkpoints` carry a
  graph copy per stage plus the six boundary-node lists; `results_state["settings"]` is what the run
  started with (`input_path`, `raw_image_path`, `segmentation_source`).
- Only the `export_results` checkpoint's graph (or `results_state["graph"]`) has
  `G.graph["statistics_report"] = {"mode", "csv", "sections"}`; a section is `{label: value}`
  (values may be `"N/A (...)"` strings or nested dicts). Report = summary numbers only (means,
  medians, a few percentiles, per-class and per-order tables); no histograms, no degree
  distribution, no inlet/outlet count. The graph has everything per vessel.
- Node/edge contract downstream: int node ids, node `pos` (z,y,x) µm, edge `voxels`/`length`/
  `diameter_um`/`branch_order`, `G.graph["image_voxel_size_zyx"]`/`["voxel_size"]`/
  `["image_extent_um"]` (`graph.dead_ends.record_image_extent`), `dead_end_kind` on degree-1 nodes
  (open-end model), never `weight`.
- `haemodynamics/poiseuille.stamp_edge_diameters` overwrites `diameter_um` from the table unless
  the source is kept (`_KEPT_DIAMETER_SOURCES`) or `override`; Murray's law
  (`statistics/bifurcation._measured_diameter_or_none`) ignores table/class_median/override only.

## Architecture

```
src/haemolynx/synthetic/        # new subpackage (add to haemolynx/__init__, tests/test_public_api SUBPACKAGES)
├── __init__.py                  # __all__: NetworkTemplate, TemplateError, load_template, generate_network,
│                                #   boundary_nodes, compare_run, write_template_comparison_csv, ...
├── template.py                  # NetworkTemplate (+ JSON), template_from_statistics / template_from_graph,
│                                #   the one report reader both use, refusals, lognormal helpers, load_template (memoised)
├── trees.py                     # arterial/venous bifurcating trees, generation by generation from the order table
├── bed.py                       # capillary bed: degree-sequence arithmetic, jittered lattice, Delaunay fill, repair
├── polylines.py                 # vectorised bend-to-length + densify (batch form of secomb_io._fit_length)
├── draft.py                     # NetworkDraft: the network while it is drawn (nodes, roles, edges with
│                                #   lengths and widths, no polylines yet); trees.py and bed.py add to it
├── generate.py                  # generate_network orchestrator, roles, dead-end marks, metadata, validate_synthetic_graph
└── compare.py                   # template vs synthetic rows, tolerances, CSV
src/haemolynx/pipeline/naming.py # run_stem(settings): the output stem for both routes (leaf module)
```

### The template (`synthetic/template.py`)

Frozen, JSON-serialisable dataclasses: `NetworkTemplate` (extent_um, voxel_size_zyx, image_shape,
inlets/outlets as `Endpoint(face, position_rel)`, boundary counts per role, `scheme`
hierarchical/capillary, per-class `ClassTarget` (count, total/mean/median length, mean/median
diameter, tortuosity), per-order `OrderTarget` (label normalised through
`statistics.bifurcation._normalize_branch_order_tag`, edge count, mean length, tortuosity,
emergence angle, mean diameter, CV, optional `Samples`), total length, daughter angle, Murray
exponent (+IQR), intercapillary distance, `LoopTarget` (independent loops, per mm³, length
percentiles), `DeadEndTarget` (leaves_image, interior, per mm, chain-length median/P10/P90),
`Provenance` (path, source, version, run manifest commit when found, notes)); `scaled(f)`,
`to_json`/`from_json`. Helpers `lognormal_from_mean_median` (σ² = 2 ln(mean/median)),
`lognormal_from_mean_cv`, `lognormal_from_median_p10_p90`.

- `template_graph(snapshot)`: the `export_results` checkpoint graph (else the last checkpoint with
  branch orders, else `results_state["graph"]`) and its checkpoint's boundary lists.
- `check_template_run(snapshot, G)` → `TemplateError(reason=...)` in order: not a run / no branch
  orders ("run it through Boundaries"), synthetic (`G.graph["network_source"]=="synthetic"` or the
  settings say so), global-mask classes (`Unknown_*`: "needs arteriole and venule sides"), no
  inlets, no positive `diameter_um` ("run its Diameters stage").
- **One reader for both sources**: `template_from_report(sections, G, checkpoint, ...)`.
  `template_from_statistics` feeds it the saved sections and raises one `TemplateError` listing
  *every* missing required section with its toggles (`missing_statistics` attribute):
  `basic`/`branch_orders` → `statistics`; `vessel_density`, `vessel_classes`, `tortuosity`,
  `murray_law`, `daughter_angles`, `intercapillary_distance`, `dead_ends` → their own toggle;
  `loop_hierarchy` → `statistics_network_analysis` + `statistics_loop_hierarchy`. An `"N/A"` value
  is a fallback with a note, never an error. `template_from_graph` recomputes those same sections
  from the graph (`statistics.compute_vessel_statistics_by_measure` + `compute_branch_order_statistics`,
  so bridges are contracted as the pipeline does), feeds the same reader, then adds per-vessel
  `Samples` (lengths, diameters, tortuosities per order/class, loop lengths, chain lengths).
  Endpoints come from the checkpoint lists for both sources (face by
  `graph.dead_ends.face_distance`).
- Fallbacks (each a provenance note): Murray exponent 3.0; daughter angle 75°; loops unknown → mean
  bed degree 3; no large tier → trees start at Art1/Ven1; no venous orders → outlets attach to the
  bed directly; `capillary` scheme (inlets only) → no trees.
- `load_template(path, source)` memoised on (resolved path, size, mtime, source) so preflight and
  the stage load the file once per process; imports `gui.run_snapshot` lazily (it imports
  `pipeline.stages`).

### The generator (`synthetic/generate.py`, `trees.py`, `bed.py`, `polylines.py`)

`generate_network(template, *, seed=20240917, rng=None, options=GenerationOptions(scale, interior_dead_ends, ...)) -> nx.MultiGraph`,
every draw through one `np.random.default_rng` (house pattern `constriction.resolve_generator`).

1. **Box and roots.** `T = template.scaled(scale)`; one root per template inlet/outlet at its
   relative face position (counts and faces exact).
2. **Trees** (per side, large tier then Art/Ven): generation counts from the order table are hit
   *exactly* — generation 1 one edge per root, then at each generation choose how many tips
   bifurcate (trifurcate only when an odd count forces it; clamp at 3× with a note). Tips chosen
   by space-filling weights (distance to faces and to existing nodes, `cKDTree`). Lengths and
   tortuosity per order from samples or lognormal fits; **Murray's law enforced exactly at every
   junction** with the template's exponent, the asymmetry fitted in closed form to the per-order
   mean diameters; daughter angle drawn round the template mean and split in the Murray-optimal
   proportion; directions pick the azimuth that stays in the box and fills space. A child that
   would leave the box is cut at the face (`leaves_image`).
3. **Leaf roles.** Tips at a tier boundary → `large_*_boundary`; interior leaves → the
   template's number of `arteriole_boundary`/`venule_boundary` nodes (they become bed vertices),
   the rest blind dead ends (or all boundary nodes with `interior_dead_ends=False`).
4. **Capillary bed** (chosen over Voronoi — see Design notes): the degree sequence is computed up
   front from the handshake lemma so that **capillary count, independent loops (cyclomatic number)
   and dead-end count come out exact together** (`degree_sequence_targets`: bed vertices
   `P = E_cap − Λ − R + 1`, degree-1 count = the dead-end target, degree-4 vertices only when the
   loop target needs them). Vertices on a jittered cubic lattice of spacing `(V/P)^{1/3}` (no
   vertex near a tree node or a face); candidate edges from `scipy.spatial.Delaunay` (never
   cross); a degree-constrained greedy fill scored by jittered length and closeness of the junction
   angle to the template's daughter angle; connectivity repair along Delaunay edges (keeps Λ);
   degree-2 merge; the degree-1 vertices nearest the faces projected onto them (`leaves_image`),
   the rest `blind`.
5. **Lengths, diameters.** Capillary lengths by rank-matching chords to the template distribution
   (`L ≥ chord`), one global factor making the class total length exact; interior dead-end chains
   from the chain-length distribution; diameters from samples or lognormal(mean, median);
   `diameter_source="synthetic"`, per-edge `diameter_basis` = the template's.
6. **Polylines**: `bend_to_lengths` (vectorised bisection; a `sin²` envelope so every edge leaves
   its node along the chord — the emergence/daughter angles are read off the first 10 µm),
   `graph._helpers.densify_polyline(max_step_um=0.5)`, `voxels=path.tolist()`,
   `length=calculate_path_length(voxels)`.
7. **Finish**: node attribute `synthetic_roles` (tuple of roles) + `boundary_nodes(G)`; every
   degree-1 node gets `dead_end_kind` (`leaves_image` on a face, else `blind`) with
   `dead_end_kind_source="synthetic"` (new constant in `graph/dead_ends.py`);
   `record_image_extent`, voxel metadata, `G.graph["network_source"]="synthetic"`,
   `G.graph["synthetic_template"]` (provenance + seed/scale/options); `validate_synthetic_graph`
   raises on any contract breach (also used by tests).

Exact by construction: extent, voxel size, shape, inlets/outlets per face, Art/Ven/Large order
counts, Murray exponent, capillary count (± repairs, <1 %), cyclomatic number, dead ends by kind,
every `length`, class total length. Approximate and reported: per-order B counts, mean/median
lengths (chord floors), tortuosity, intercapillary distance, loop-length percentiles, emergence
angles, tree-order mean diameters. Expected time for an E14.5-sized template (~10 k edges):
3–4 s; `template_from_graph` on the template ~10 s once (loop hierarchy).

### Comparison (`synthetic/compare.py`)

`compare_run(template, G, sections=G.graph["statistics_report"]["sections"], ...)` → rows
(section, characteristic, template, synthetic, ratio, tolerance, within) over every target above;
`write_template_comparison_csv` → `{stem}_template_comparison.csv` beside the statistics CSV; the
rows also go into the statistics report (`synthetic_comparison` section) so the Statistics tab
shows them. One `TOLERANCES` table is what the tests assert (exact / rel 3–25 % / abs 0.15
tortuosity, 10° angles, 0.3 Murray).

## Pipeline integration

### Settings (`pipeline/schema.py`, section "Input and segmentation", declared before `input_path`)

| name | kind / default | requires | advanced |
|---|---|---|---|
| `network_source` | choice `segmented_image` \| `synthetic`, default `segmented_image` | – | no |
| `synthetic_template_path` | path, None, `must_exist=True` | `network_source=synthetic` | no |
| `synthetic_template_source` | choice `statistics` \| `graph`, default `statistics` | `network_source=synthetic` | no |
| `synthetic_seed` | int 20240917 (null = fresh) | `network_source=synthetic` | yes |
| `synthetic_scale` | float 1.0, min 0.1, max 10 | `network_source=synthetic` | yes |
| `synthetic_interior_dead_ends` | bool True | `network_source=synthetic` | yes |

Help text ≥ 3 words, no trailing stop; `network_source`'s help says the Skeletonise and Graph tabs'
settings are not read under synthetic. Choice literals repeated in the schema (it stays
numpy-free); constants + `network_is_synthetic(settings)` live in `graph/boundaries.py` beside
`vessel_masks_in_use`, exported from `graph/__init__`; a test pins the choices equal.

Existing rows that gain `"network_source=segmented_image"` in `requires` so they hide under
synthetic (the section is in `form.HIDE_WHEN_UNMET_SECTIONS`; children follow through
`form.prerequisite_chain`): `input_path`, `raw_image_path` (keeps `always_effective`, `migrates`),
`image_axis_order`, `voxel_size_policy`, `voxel_size_override_xyz`, `use_memmap_loading`,
`segmentation_cleanup`, `crop_on_import`, the three check/optimise rows, and
`automated_vessel_assignment` (Boundaries tab). Left ungated: `segmentation_source`,
`mid_run_postprocessing`, every Skeletonise/Graph/Diameters setting. `input_path` is **not** made
`always_effective`: a path the user set and the run ignores deserves the one-line "nothing will read
it" warning; the panel's layer auto-adoption into `input_path` just skips under synthetic.

### Output stem (`pipeline/naming.py`, new leaf module)

`run_stem(settings)` = `input_path.stem` for an image run, `f"{template stem}_synthetic"` under
synthetic, None with neither; `run_input_path(settings)` (None under synthetic). Replace every
`settings['input_path'].stem` read: `stages.py` (18 sites: skeleton/graph paths, boundaries' graph
rewrite, tissue export, every export filename), `checks.check_cached_artefacts`,
`provenance.manifest_location` + `input_files`, `gui/stage_checkpoints._stem_and_output_dir` and
`stems_for_cached_artefacts` (currently via `segmented_input_path`, which returns None under
synthetic). `optimisation/report.py` and `perturbation_plots.py` are image-only / already given a
stem: leave.

### Stage flow (`pipeline/stages.py`)

- `raw_image_is_set` → False under synthetic (switches off ilastik, raw diameters, raw evidence in
  one place); `segmented_input_path` → None (EDT skipped, no mask).
- `SegmentedInputs` gains `synthetic: bool`, `template_path`; `SkeletonisedVolume.image`/`skeleton`
  become `| None` with `synthetic: bool` (readers after the skipped stages already tolerate None;
  `gui/results._from_skeletonise` draws no IMAGE layer).
- `segment`: **first statement** `if network_is_synthetic(settings): return _segment_synthetic(...)`
  (before the `segmented_path` → `input_path` write): coerce the template path, `load_template`
  (cached; `TemplateError` propagates), normalise `vtk_output_prefix`/`final_render_mode`, log the
  template summary.
- `skeletonise`: `if inputs.synthetic: return SkeletonisedVolume(image=None, skeleton=None,
  voxel sizes from the template, synthetic=True)`; writes nothing.
- `build_network`: **before** the mask loading (which reads `image.shape`): with `resumed_graph`
  hand it back after voxel metadata; else `generate_network(...)`, one progress step, raise on an
  empty graph, pickle to `{stem}_graph.pkl`, dense-solver memory warning, return
  `VesselNetwork(graph, volume)`.
- `assign_boundaries`: first statement under synthetic → `_assign_synthetic_boundaries`: fill the six
  run lists in place from `synthetic.boundary_nodes(G)`, raise if no inlet or outlet,
  `resistance_node_pair=(inlets[0], outlets[0])`, return `BoundaryNodes` with no masks. Resume from
  Diameters on already carries the lists. `_assign_branch_orders` unchanged: with outlets and both
  boundary lists present `graph.assign_vessel_branch_orders` runs hierarchical
  (Large_Art/Art/B/Ven/Large_Ven) by itself.
- `graph/boundaries.vessel_masks_in_use` gains `and not network_is_synthetic(settings)` so mask
  loading, the stage, preflight, citations and the panel's greying all read "no masks"
  (`tests/test_vessel_mask_switches.py` pins the one answer).
- `assign_diameters`: unchanged apart from `poiseuille.py`: `DIAMETER_SOURCE_SYNTHETIC="synthetic"`
  added to `DIAMETER_SOURCES`, `_MEASURED_DIAMETER_SOURCES`, `_KEPT_DIAMETER_SOURCES`, and
  `stamp_edge_diameters` keeps it like `override` (counts `"synthetic"`). Murray already reads it.
- `post_process`: guard the thick-vessel bridging when `volume.image is None` and no region.
- `tissue_measurement` → None with no image (preflight refuses first).
- `export_results`: `image_shape = image.shape if image is not None else synthetic_box_shape(G)`
  (inverse of `image_extent_um`) for the statistics' `image_dimensions`; `measurement_3d_to_cell_mask`
  raises under synthetic; 2-D overlays skipped; after statistics, `compare_run` +
  `write_template_comparison_csv` and the `synthetic_comparison` section.

### Preflight (`pipeline/checks.py`)

`check_synthetic_template(settings)` added to `preflight`: empty unless synthetic; suffix must be
`.haemorun`; `load_template` (cached) with `RunSnapshotError` → unreadable, `TemplateError` by
reason → "is itself synthetic", "saved with statistics off / missing sections: switch on <toggles>
on the Additional measurements tab and run its Export again", "no inlets or outlets", "classed by
size alone". Pass line with vessel/inlet/outlet counts and the box. Errors: `measurement_3d_to_cell_mask`,
`measure_tissue_volume`. Warnings for switches on but out of effect: ilastik, raw diameters,
vessel masks ("the generator chooses the boundaries"), thick-vessel restriction/skeletonisation,
`do_skeletonize`/`do_graph_building` off ("generated every run"), `segmentation_source` differing
from the template's basis. The image-only checks (`check_cached_artefacts`, `check_raw_image`,
`check_segmentation_quality`, `check_image_channels`, `check_tissue_volume_image`,
`check_cell_mask_measurement`, `check_thick_vessel_restriction_prerequisites`,
`check_input_is_not_a_large_vessel_mask`) return early. `check_settings`'s `skip` extended under
synthetic with the hidden `must_exist` paths (ilastik classifier, mask paths, cell/tissue masks),
since it reads a path's own `requires` only.

### Panel (minimal)

- `gui/form.py`: `PATH_FILTERS = {"synthetic_template_path": "HaemoLynx run (*.haemorun);;All files (*)"}`
  read in `_options_for` (FileEdit already accepts `filter`); `CHOICE_VALUE_LABELS` and
  `SETTING_ROW_LABELS` for the two choices and the path; `shared_ilastik_host` None under synthetic.
- `gui/layout.py`: the "input" box names start `network_source, synthetic_template_path,
  synthetic_template_source, input_path, ...`; `ADVANCED_TITLES["network_source"]`.
- `gui/boundary_picking.py`: `role_manual_controls_enabled` False for every role under synthetic;
  `role_disabled_note(role, values)` with a synthetic note; `_widget.refresh_role_tabs` uses it and
  `network_source` joins its trigger list.
- `_widget.py`: the "Use open layer" row hidden and layer adoption skipped under synthetic.
- `gui/results.py`: a note layer line for the skipped skeletonise; `_image_shape_z` from the graph's
  extent in `_from_build_network` when None.
- Run from "3. Graph" regenerates with the seed; from Boundaries on the stored graph is handed back.
  Save/Load: the template path relocates by name; a synthetic `.haemorun` records
  `network_source=synthetic` so it is refused as a template.

### Provenance, citations, docs

- `provenance.manifest`: `network_source` and a `synthetic` block (template, source, seed, scale,
  interior_dead_ends); the template file is fingerprinted as an active `must_exist` path.
- `citations.py`: tifffile/scikit-image/skan credits off under synthetic; new
  `NETWORK_GENERATION_CITATIONS` section with one entry whose reason describes our construction
  (degree-constrained random spatial network on jittered lattice points, Delaunay candidate edges,
  matched to the template's statistics) and references Smith et al. 2019, Front. Physiol. 10:233,
  doi 10.3389/fphys.2019.00233 for the organisational requirements a synthetic capillary bed is
  matched on.
- CLAUDE.md: repository-layout line for `synthetic/`, a Key-modules entry, one sentence in the
  end-to-end pipeline; README one line; `python examples/regenerate_configs.py`;
  `PLAN_synthetic_network.md` at the repo root (copy of this plan, the repo's convention). The
  help menu (`gui/stage_help.py`) is not touched.

## Design notes (why)

- **Delaunay fill, not Voronoi**, for the bed: Voronoi vertices are all degree 4, edge lengths span
  two orders of magnitude, infinite ridges need clipping, and `E = 2N` fixes the loop density so it
  can only be tuned by removal — all that clean-up is the work. The degree-sequence fill hits
  count, loops and dead ends exactly and needs no crossing test. A `bed_model="voronoi"` can be
  added later behind an option if a literature-identical bed is ever wanted.
- **Murray exact, order means approximate**: a table whose child/parent ratio exceeds `2^{-1/n}` is
  unattainable under Murray; Murray wins and the comparison CSV shows the gap.
- **Interior dead ends reproduced by default** (`synthetic_interior_dead_ends=True`): they move
  the flow solution, so an "equivalent" network keeps their rate; off puts every dead end at a face.
- **Image box, not the vessel-occupied box**, in v1: the comparison's density rows show any
  mismatch; a `vessel_box` option is a possible follow-up.

## Implementation order (each step leaves the suite green)

1. `synthetic/template.py` data model, JSON, lognormal helpers, `template_from_report`, refusals,
   `template_from_statistics`, `template_from_graph`, `load_template` (+ `tests/test_synthetic_template.py`,
   `tests/synthetic_fixtures.py`: a tiny hand-built hierarchical graph with hand-computed targets, a
   hand-set `fixture_template`, `write_fixture_run(tmp_path, ...)` via `write_run_snapshot`).
2. `polylines.py`, `trees.py`, `bed.py`, `generate.py`, `compare.py` (+ `tests/test_synthetic_generate.py`,
   `tests/test_synthetic_compare.py`; `haemolynx/__init__`, `test_public_api.SUBPACKAGES`).
3. `graph/boundaries.py` constants + `network_is_synthetic` + the `vessel_masks_in_use` guard;
   `graph/dead_ends.SOURCE_SYNTHETIC`; `pipeline/naming.py` and the `.stem` replacements.
4. Schema settings and `requires`; `form.py`/`layout.py`; regenerate configs; `poiseuille.py` stamping.
5. `checks.py` preflight; stage branches in `stages.py`; `tests/test_synthetic_stage.py`.
6. Citations, provenance, panel edits + gui test, docs, `PLAN_synthetic_network.md`.

## Tests (new and touched)

- `tests/test_synthetic_template.py`: both sources agree field by field on the tiny run; each
  section's rows read into the right targets; every missing required section names its toggles
  (parametrised); no report names `statistics`; synthetic / global-mask / no-orders / no-inlets /
  no-diameters refused; N/A Murray → None + note; endpoint faces; JSON round trip; `scaled`
  (counts ×f³, endpoints ×f²); lognormal helpers; export checkpoint preferred; provenance commit.
- `tests/test_synthetic_generate.py`: contract via `validate_synthetic_graph`; same seed identical,
  other seed differs; inlets/outlets on the right faces; `assign_vessel_branch_orders` yields exactly
  the template's label set with exact Art/Ven counts; Murray exponent at every tree junction;
  daughter angle ±10°; capillary count ±3 %, class length ±5 %, cyclomatic exact; dead ends by kind
  exact and `interior_dead_ends=False`; `degree_sequence_targets` identities; no bed edge between
  attached nodes; `scale=1.5`; venous depth 0; capillary-only; large tier on one side;
  `bend_to_lengths`; the acceptance test (every fitted comparison row within tolerance);
  `@pytest.mark.slow` ~10 k edges under 30 s.
- `tests/test_synthetic_compare.py`: rows, ratios, N/A handling, CSV naming, summary.
- `tests/test_synthetic_stage.py`: each stage under synthetic on a template written with
  `write_run_snapshot` from a `tests/data` run with statistics on; `{template}_synthetic_*` outputs
  (graph pickle, statistics CSV, comparison CSV, manifest with the `synthetic` block and the template
  in `inputs`, citations without tifffile/skimage/skan); refusals; `resumed_graph` handed back;
  `assign_boundaries` lists; diameters kept (`stamp_edge_diameters` counts); `raw_image_is_set`,
  `segmented_input_path`, `run_stem` for both routes; `post_process` with thick-vessel on and no image.
- Preflight messages (new `tests/test_pipeline_checks.py` or `test_parsers_checks.py`); `tests/test_vessel_mask_switches.py`
  (False under synthetic whatever the switches hold); `test_gui_form.py` (visible Input rows per
  source, Diameters raw subtree hidden, `PATH_FILTERS == run_snapshot.SAVE_FILTER`, labels);
  `test_gui_layout.py` (rows once, advanced trio behind the `network_source` button, the 50–70 %
  advanced share: +3/+3 keeps it); `test_gui_boundary_picking.py`; one `@pytest.mark.gui` test
  (combo hides image rows, greys the four role tabs with the note, FileEdit carries the filter);
  `test_gui_stage_checkpoints.py` / `test_gui_run_snapshot.py` (stem, Clear, relocation);
  `test_citations.py`, `test_pipeline_provenance.py`, `test_hemodynamics.py` (stamping),
  `test_statistics.py` (Murray reads synthetic widths), `test_parsers_schema.py` (choices pinned).
- Must keep passing: `test_example_configs` (regenerated), `test_setting_units` (`scale` is not a
  measurement word), `test_gui_tooltips`, `test_segment_stage`, `test_skeletonise_stage`,
  `test_pipeline_invariants`, `test_public_api`.

## Verification

1. Unit suites above, then `pytest -m "not slow and not gui"` (13–28 min here), the `slow`
   generator test, and the gui tests for the panel in the background.
2. End to end on real data: run E14.5 (an existing `.haemorun` with statistics on) as the template
   via the panel — Input tab → Synthetic → pick the file → Run; check the viewer shows the network
   after stage 3, the Boundaries role tabs are greyed, the Statistics tab has the comparison
   section, and read `{stem}_template_comparison.csv`: every "exact" row equal, the rest within
   tolerance; time the generation (target < 10 s); save the run and confirm its `.haemorun` is
   refused as a template with the right message; run again with `synthetic_template_source=graph`
   and compare.
3. Preflight messages checked by hand: a `.tif` path, a template saved with `statistics` off, a
   run with `statistics_loop_hierarchy` off.

## Working practice

Other sessions edit this checkout concurrently (uncommitted files from them are present now):
targeted edits only, run the named test files rather than the whole suite between steps, commit
through a private index, and do not edit `src/` while a long test run is in flight.

## Risks

- A `None` `input_path` reaching `.stem` anywhere not moved to `run_stem` (checks, checkpoints,
  provenance are the ones outside `stages.py`).
- `build_network`'s mask loading reads `image.shape` before the resume check: the synthetic early
  return must come first, or a run from the Boundaries tab crashes.
- `segment` writes `segmented_path` into `input_path` first thing: the synthetic branch precedes it.
- Hidden `must_exist` paths (ilastik classifier, masks) still demanded by `check_settings` without
  the extended `skip`.
- `mid_run_postprocessing` works, but mask-reading review lists report `NEEDS_MASK` and Add
  vessel's mask/raw tracers are unavailable; pericyte-mask perturbation strategies fail at their own
  file check (a warning if `check_perturbations` can see the strategy).
- Loading a 100 MB template in preflight stalls the GUI thread for a few seconds the first time; the
  memoised loader keeps it to once per template.

## Status (2026-10-25: done; Harvey's panel check passed)

Implemented 2026-10-18 (`bb831907`, merged `d098de1f`). Since:

- `c8d6624d` (2026-10-18): the large-tier vessels the generator draws are tagged as lying in a
  large mask (`large_mask_vessel_type`), so the branch-order hierarchy finds the Large_Art/Large_Ven
  tier.
- `ce3a7a2e` (2026-10-19): the hidden-settings check (`test_hidden_settings_cannot_affect_a_run.py`)
  is given a template under `network_source=synthetic`.
- `055be569` (2026-10-19): a batch run refuses `network_source=synthetic` (`pipeline/batch.py`).
- `399a48f3` (2026-10-23), `2feb8764` (2026-10-24): the optimisers (segmentation/skeleton/graph,
  FWHM and the endothelial ring fit) are greyed for a synthetic network.
- `5b3b8fdc` (2026-10-24): **the template's diameter basis** (step 5 and the preflight list, never
  built before). `NetworkTemplate.diameter_basis` is the basis most of the template's vessels'
  viscosity was read on (`poiseuille.edge_diameter_basis`: a vessel's own mark, else the run's --
  `G.graph["diameter_basis"]`, else its `segmentation_source`), the run's on a tie, with a note
  when the vessels are on two bases; kept in the JSON, None for a template written before it.
  Every drawn edge carries it (`validate_synthetic_graph(G, template)` checks), stamping keeps it,
  so a synthetic run's viscosity is the template's form of the law whatever its own
  `segmentation_source`; preflight warns when the two differ. The comparison has a "diameter
  basis" row, and **every band is now in one `TOLERANCES` table** in `compare.py` (rows checked
  unchanged before and after on four comparisons, 606 rows).
- `4767c3c6` (2026-10-24): **the tests the plan listed and never had**: Murray's law reads synthetic
  widths (`test_statistics.py`: counted as measured, and the generator's exponent read back at
  every tree junction); stamping keeps them (`test_diameter_assignment.py`, where the stamping tests
  live, not `test_hemodynamics.py`); a synthetic run's checkpoint stem and Clear
  (`test_gui_stage_checkpoints.py`); its template relocated by name and no resume skeleton
  (`test_gui_run_snapshot.py`); the thick-vessel rerun test now asserts the same network, widths,
  resistances and flows as the run without the switch, and no bridge. The slow generator test is
  back to the plan's 30 s: it measured 11-12 s for 9,500 vessels, half of it
  `validate_synthetic_graph` (`np.allclose` four times an edge, each polyline converted twice);
  the same checks written out take 1.3 s, and generation 7-10 s.
- `1234d905` (2026-10-24): verification 3 (below); a `.haemorun` that is not a run now says what to
  choose.

**Verification 3, preflight's messages** (2026-10-24, through `preflight` on the fixture runs;
each is the whole `synthetic_template_path:` error):

- A `.tif`: "... stack.tif is not a .haemorun file. Fix: save the template run from the panel
  (Save run) and point synthetic_template_path at it."
- A `.haemorun` that is not a run: "Could not read ...: UnpicklingError: invalid load key, 'n'.
  Fix: choose a run the panel saved (Save run), of a microscopy-image run."
- Saved with `statistics` off: "The template run was saved with Statistics off, so it has no
  statistics report: switch on statistics on its Additional measurements tab (with
  statistics_vessel_density, statistics_vessel_classes, statistics_tortuosity,
  statistics_murray_law, statistics_daughter_angles, statistics_intercapillary_distance,
  statistics_dead_ends, statistics_network_analysis, statistics_loop_hierarchy) and run its Export
  stage again."
- `statistics_loop_hierarchy` off: "The template run's statistics lack 'loop_hierarchy': switch on
  statistics_network_analysis, statistics_loop_hierarchy on the template run's Additional
  measurements tab and run its Export stage again." Several missing sections are named in one
  error.
- A synthetic run: "The template run is itself a synthetic network: choose the microscopy-image
  run it was generated from."
- A missing file: `check_settings`' "checked: ... Fix: point 'synthetic_template_path' at an
  existing file (the .haemorun file of a finished microscopy-image run ...)".
- Another basis (a warning, not an error): "segmentation_source is plasma_column but the
  template's diameters are endothelial: the drawn vessels keep the template's basis (and its form
  of the viscosity law), while a width given by hand in Post processing is read as plasma_column.
  Fix: set segmentation_source to endothelial."

**Verification 2 (E14.5 as the template, end to end; 2026-10-24, scripted, not through the
panel).** The template: the main checkout's `config.yaml` (Claudin E14.5, large-vessel masks
alone, `segmentation_source=endothelial`), low-RAM, perturbations off, run whole at `2a749a95` (953
s; 1,228 nodes, 1,406 edges; 1 inlet, 2 outlets, 34 + 47 large-vessel hand-off nodes, no
arteriole/venule boundary nodes; every statistics section), and saved as the panel's Save run
would (an Export checkpoint with the graph and the six lists). Harness and outputs:
`outputs/synthetic_e14p5/` (git-ignored) in the `plan-synthetic-e08d0c` worktree. Read as a
template: 1,267 vessels (27 Large_Art over 15 orders, 1,181 capillaries over 24 B orders, 59
Large_Ven over 15), basis endothelial, both sources agree; reading takes 0.9 s (statistics) and
6.2 s (graph). Generation takes **1.2-1.3 s** at three seeds from either source (target < 10 s).
The synthetic run's own `.haemorun` is refused: "The template run is itself a synthetic network:
choose the microscopy-image run it was generated from."

It found three bugs, fixed with tests that fail on the code before (fixture templates draw
exactly as before, edge for edge, at 12 seeds and 9,500 vessels):

- `89cb4b9b`: the network was **not connected** at every seed (2-4 pieces): stars (a junction
  whose every vessel ends blind) beside a dead end of the largest piece, which the connectivity
  repair (now `bed.join_pieces`) gave up on. It now joins junction to junction past the dead end.
- `68d6885e`: with **no Art/Ven tier** (large masks alone), the large tier's leaves were given the
  small tier's role, so the run labelled the large tier Art1..15/Ven1..15 and counted no large
  boundary nodes (68 of 79 exact rows outside). They are now `large_*_boundary`, the run assigns in
  `large_tier` mode.
- `5dda9ac9`: the graph source's default seed drew the one inlet's first vessel out through a face,
  and the side's early return skipped the rule that a tree keeps a leaf on the bed: the inlet was a
  piece of its own.
- `f505c0be` (the comparison): the dead-end total is no longer banded when the template has
  artefact ends or ends of pieces without a junction (17 and 94 here), which one connected
  network cannot have; the kinds it draws were exact (91 leaving the image, 194 inside).

The comparison then (`{stem}_template_comparison.csv`, haemodynamics off -- see the first item
below): **statistics source, 41 of 49 exact rows equal, 50 of 63 banded rows within; graph
source, 30 of 49 and 34 of 49.** Equal: box, voxel size, inlets and outlets by face, diameter basis,
dead ends by kind, capillary count (1,185 / 1,186 of 1,181) and mean diameter (10.40 / 10.33 of
10.40 um). What is outside, and why, is in the first four items below; the rest outside: loops 243
/ 238 of 231 (the generator's note: 231 loops and 277 dead ends cannot both be met with 1,181
capillaries, so 237 are drawn), daughter angle 97 of 114 degrees, average degree 2.47 of 2.31 (the
template's 94 ends of junction-less pieces).

**Decided by Harvey, 2026-10-24** (on the options listed under "Before the decisions" below), and built:

1. The large tier: **a trunk with side branches handed to the bed, its widths the template's per
   order** (Murray's law not enforced in the large tier, its exponent reported only; Murray stays
   exact in the Art/Ven tiers).
2. A root's first vessel: **re-aimed, else shortened, to stay in the box; and no class-length
   rescale stretches a vessel past the template's longest or most tortuous** (the shortfall
   reported instead).
3. The bed: **its lattice spacing from the template's mean capillary chord, filling only as much
   of the box as the count needs, grown from the trees; where the template measured a tissue
   volume, the bed fills that volume instead.**
4. `vessel_box`: **dropped** (the occupied volume of item 3 takes its place).
5. **A Voronoi bed as an option** (`bed_model`), beside the Delaunay fill.
6. The panel half of verification 2: **Harvey checks it himself.**

Built (templates the change does not concern draw as before: the fixture template's networks were
compared edge for edge, position for position, at 12 seeds and 9,500 vessels before and after):

- `78d7b1f0` (1, 2): within the large tier a tip may go on as one child and hand a side branch to
  the bed, the node a large-vessel boundary node taking one bed vessel
  (`trees.trunk_children_per_tip`, `SideResult.handoffs`); its widths are each order's own
  (`draw_root_diameters`), Murray's law and its fitted asymmetry are the Art/Ven tier's only. A
  root's first vessel is re-aimed (up to 12 directions in its cone) when the one drawn leaves the
  box, else shortened inside it, noted; the tree length rescale stops at the class's longest and
  most tortuous (the graph source's own vessels; else 6x the median length and the tortuosity two
  sigmas up the draw), noting the shortfall. The fixture's lengths move by at most 2% on a few
  vessels (the cap); nothing else.
- `ff4f4dcb` (3, 4): `generate.bed_volume_um3`: the template's measured tissue volume, else what
  the bed's vertices take at the template's mean capillary chord apart, never more than the box (a
  bed denser than its box -- the fixture's -- fills it whole, as before). A smaller bed is the
  lattice at its spacing over the box, the points nearest the trees kept
  (`lattice_vertices(grow_from=)`). The graph source reads a measured tissue volume from the saved
  report too. `NetworkTemplate.vessel_box_um` is gone; a template written with it still loads.
- `9e49ddfc` (5): `synthetic_bed_model` (`delaunay`, the default, or `voronoi`; Input tab, behind
  the synthetic Advanced button). The Voronoi bed's vertices are the corners of a tessellation of
  random seeds (`bed.voronoi_vertices`: Delaunay circumcentres, joined across shared faces; corners
  outside the bed's volume or beside the trees left out; more seeds when a small box keeps too
  few; the shortest edges contracted down to the vertex count), and the same fill prunes its cell
  edges to the template's counts. 75-80% of the bed's vessels are cell edges (the record's
  `voronoi_edges`); the repairs draw the rest. Plumbed through the schema (choices pinned to
  `bed.BED_MODELS`), the stage, the form labels, the layout, the manifest, the citation text, the
  example configs.

**E14.5 again, after the decisions** (the same template, haemodynamics on, which now solves:
`outputs/synthetic_e14p5/`). Generation 1.2-1.4 s. Comparison:

| bed / source | exact rows equal | banded rows within |
|---|---|---|
| Delaunay / statistics | 47 of 49 | 57 of 63 |
| Delaunay / graph | 47 of 49 | 56 of 63 |
| Voronoi / statistics | 47 of 49 | 55 of 63 |
| Voronoi / graph | 47 of 49 | 53 of 63 |

Every large-tier order's vessel count is the template's (it was 68 of 79 exact rows outside); the
large classes' mean widths are 18.4 and 21.1 um as measured; capillary length 43.9 mm against 42.5
(was 49.5), mean 36.9 against 36.0 um. Still outside: the large-vessel boundary counts (22 / 49
drawn against 34 / 47: the trunk's hand-offs and leaves, not the template's count), the daughter
angle (97-101 against 114 degrees), the average degree (the template's 94 ends of junction-less
pieces), loops (239-249 against 231: the bed's note, 231 loops and 277 dead ends cannot both be met
with 1,181 capillaries), the capillary median length (32.4 against 28.1 um, 15.4% where the band is
15%), and with the Voronoi bed the tortuosity (1.42 against 1.20) and the intercapillary distance
(12 against 22 um): its cell edges are shorter than the template's chords, so its vessels are
bent further to their lengths. Network resistance: 1.19 (Delaunay) and 1.43 (Voronoi) x 10^15
Pa.s/m^3 against the template's 1.99.

**Segmentation source follows the template** (Harvey, 2026-10-24, after his first panel run, which
ran on a checkout without the decisions and with the row at plasma_column): picking a template, or
switching the Input to synthetic with one picked, sets Segmentation source to the template's
diameter basis (`gui/template_basis.py`, `synthetic.diameter_basis_of_run`: the basis without
reading the whole template, 0.6 s on E14.5); loading a config or a saved run never changes it.

**Harvey's panel check (item 6).** On the Input tab set Input to "Synthetic network from a template
run" and pick `C:\Users\hd01\Documents\GitHub\ImageLynx\.claude\worktrees\plan-synthetic-e08d0c\outputs\synthetic_e14p5\template\e14p5.haemorun`
(the E14.5 template run above; it goes if the `plan-synthetic-e08d0c` worktree is cleaned up), set
an output folder, Run. Check: the network appears after stage 3; the Boundaries role tabs are greyed
with the note that the generator chose them; the Statistics tab has the template comparison
section; Save run, then pick the saved file as a template: preflight refuses it ("The template run
is itself a synthetic network ..."). Try the Advanced button's "Capillary bed model" too.

Harvey's results (2026-10-24, the panel on `43bdcb41`, the E14.5 template above):

- His first run failed at Haemodynamics ("Diameter must exceed 1.1 um for the Pries in vivo law"):
  his checkout was at `9b02d50c`, without the decisions above. Fast-forwarded, **it runs**.
- **The network shows in the viewer** after stage 3.
- Opening the Boundaries tab **hung napari**: Qt's `QTabBar::wheelEvent`, from a disabled current
  tab, looks for the next enabled one with a step of nought when the wheel turns less than a notch,
  and every role tab is disabled under a synthetic network. Fixed in `43bdcb41` (the panel swallows
  the wheel over the role tab bar while its current tab is disabled).
- **The Boundaries role tabs are greyed.**
- **The Statistics tab has the comparison section.** Its button read "Synthetic comparison" (its
  key's words); at his asking it is "Template comparison", with hover text of its own.
- **The saved synthetic run is refused as a template**: "FAILED: synthetic_template_path: The
  template run is itself a synthetic network: choose the microscopy-image run it was generated
  from."
- **The Voronoi bed runs** (`synthetic_bed_model=voronoi`, 2026-10-25). Every check of item 6 passed.

**Before the decisions** (the options as they were put to Harvey):


- **How to draw a large tier like E14.5's.** Its Large_Art/Large_Ven orders are a trunk whose
  order rises at every junction where a capillary leaves it (27 vessels over 15 orders, 1-3 a
  order, widths 15-27 um throughout, 34 hand-off nodes). The generator draws each order as a
  generation of a bifurcating tree and holds Murray's law exactly at the template's exponent
  (1.75): every generation is 0.67 of the last, so the drawn large vessels narrow to **0.02-0.08 um**
  at the deepest generation (class means 4.1 and 1.5 um against 18.4 and 21.1), and the trees'
  leaves give 16 + 29 hand-off nodes, not 34 + 47. On an endothelial template the haemodynamics then stops: the in vivo law is singular
  below 1.1 um ("Diameter must exceed 1.1 um for the Pries in vivo law"). The choices: draw a trunk
  with side branches handed to the bed (Murray then sets the side branch's width, the trunk keeps
  the order's), relax Murray in the large tier, or floor the widths (which only hides it). Until
  then a synthetic run of a large-masks-only template needs `run_haemodynamics` off.
- **A root's first vessel leaving the box.** When it does (the graph source's default seed on
  E14.5), the side's tree is that one vessel, and the class-total rescale stretches it to the
  class's whole length (904 um, tortuosity 14.6). Re-aim or shorten the first vessel, or cap the
  rescale per vessel.
- **The capillary bed's uniform lattice.** E14.5's capillaries are shorter (median 28 um) than the
  lattice spacing its counts give in its box (42 um): the class length comes out 16% over
  (49.5 of 42.5 mm), the median length 40% over, the intercapillary distance 30 of 19-22 um.
  The template's nodes span 99.3% of its image box, so `vessel_box` would not change this: the
  vessels are clustered within the box, which a uniform bed cannot draw.

- **`vessel_box`.** `NetworkTemplate.vessel_box_um` (the box the template's nodes span) is recorded,
  scaled and kept in the JSON, and nothing reads it. The plan names the option ("a possible
  follow-up") but not what it does. To decide: (1) whether the network is drawn in the node box or
  the image box, and whether that is a setting (and its default); (2) where the inlets and outlets
  go (the node box's faces, at the same relative positions, or where they were in the image);
  (3) what the densities divide by -- the image volume, as the template's report did, or the node
  box (every per-mm^3 row then moves by the ratio of the two volumes); (4) the synthetic run's
  `image_extent_um` and statistics `image_dimensions` (the face dead ends are judged against
  them).
- **`bed_model="voronoi"`** ("later, if wanted"): a Voronoi bed as an option beside the Delaunay
  fill, for a literature-identical bed. Whether it is wanted.
- **The panel half of verification 2** (Harvey or a tester): the viewer shows the network after
  stage 3, the Boundaries role tabs are greyed, the Statistics tab has the comparison section.

Built as planned on 2026-10-18, with these departures and findings:

- **Capillary bed**: a degree-constrained Delaunay fill on a jittered lattice (`bed.py`), not
  Voronoi -- the bed's degree sequence is worked out first from the handshake lemma, so the
  capillary count, the loop count and the dead-end count are the template's exactly. When a
  template's loops and dead ends cannot both be met with its capillaries, the loops give (more
  are drawn), never the dead ends; the generated record and the run log say so. Candidates are
  taken near the template's mean capillary chord (not the shortest), which is what keeps the
  tortuosity right; a rerouting pass (`augment`) and a bounded pairing pass fill what the greedy
  fill leaves. Blind ends are set at their chain lengths.
- **Trees**: lengths and root diameters are normalised to each order's mean at draw time (an order
  of two vessels would otherwise miss its mean by the draw); tree class lengths are then rescaled
  to the class totals.
- **Compare-only rows** (no band): the network-wide Murray median (the capillary junctions'
  random widths dominate it; the tree junctions obey Murray exactly), per-order emergence angles
  and tortuosity (read off bent polylines and orders of two vessels).
- **Gates**: a `choice` gate missing from a partial values mapping now reads as its default
  (`parsers.is_prerequisite_met(default=)`, `form.Field.gate_defaults`, `unmet_chain`), so the
  many tests and callers that pass partial dicts keep working after `network_source=segmented_image`
  was added to the Input rows' `requires`. `input_path` was not made `always_effective`.
- **Panel**: the "Use open layer" row hides and layer adoption is refused under synthetic; the role
  tabs grey with `SYNTHETIC_CHOOSES_NOTE`.
- **Results on the fixture template** (`tests/synthetic_fixtures.py`, 178 vessels): 54-55 of 55
  banded rows within band at every seed, generation 0.2 s; the fixture scaled to ~10,000 vessels
  draws in ~5 s (its counts are inconsistent by construction and exercise the fallbacks). A whole
  synthetic run from a generated-and-saved template: 6 s, every output named
  `{template}_synthetic_*`, 62 of 62 banded rows within band.
- **Not yet done** (2026-10-18): verification step 2 (E14.5 as the template, through the panel)
  and the `vessel_box` option: see above.
