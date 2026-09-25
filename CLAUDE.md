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
│   │                       #   z=1 volume), ilastik.py, voxel_validation.py, axis_order.py,
│   │                       #   raw_volume_cache.py (reuse a memmapped raw volume across runs),
│   │                       #   automated_vessel_assignment.py (mask loading/validation)
│   ├── preprocessing/      # skeleton.py — skeletonize, bridge, clean, bundle refinement;
│   │                       #   thick_vessels.py (thickness-gated skeletonisation of fat
│   │                       #   vessels) + thick_vessel_braid_guard.py; segmentation_cleanup.py
│   │                       #   (pre-skeleton mask cleanup), segmentation_quality.py and
│   │                       #   segmentation_raw_comparison.py ("Check segmented image");
│   │                       #   skeleton_consistency.py (skeleton vs mask), memmap_support.py
│   │                       #   (disk-backed volumes, the low-RAM option), pointwise_distance.py
│   ├── graph/              # assemble.py (orchestrator) + build, reconnect, optimise, degree2,
│   │                       #   prune, collapse (+ direction_aware_collapse, persistence_collapse,
│   │                       #   cartwheel_guard), branch_order, boundaries, boundary_node_fallback,
│   │                       #   large_vessels, large_vessel_network, cut_at_large_vessel_volumes,
│   │                       #   confidence_vessel_assignment, mask_component_volume,
│   │                       #   mask_continuity, diagnostics, validate, _helpers, _platform,
│   │                       #   smoothing.py (centreline smoothing — rewrites `length`),
│   │                       #   thick_vessel_junctions.py (IS_ZERO_RESISTANCE bridges),
│   │                       #   communities.py (vascular communities), edit.py (the panel's
│   │                       #   graph edits), automated_vessel_assignment.py (terminal-node
│   │                       #   assignment)
│   ├── haemodynamics/      # poiseuille, viscosity (the laws), resistance, apply,
│   │                       #   automated.py (FWHM diameters), edt_diameter.py (mask-inscribed
│   │                       #   diameters), haematocrit_distribution.py (Pries-Secomb phase
│   │                       #   separation), constriction (the one constriction model) + its
│   │                       #   site-choosing strategies: probability, pericyte_mask;
│   │                       #   constriction_strategy (which strategy a run uses),
│   │                       #   pericyte_comparison; perturbations.py (typed settings overrides
│   │                       #   to re-solve for) and what they run: arteriole, capillary,
│   │                       #   capillary_block, pericyte_sweep, pericyte_geometry_sweep,
│   │                       #   sweep_flows (per-grid-point flows for the viewer)
│   ├── statistics/         # comprehensive.py (compute_comprehensive_vessel_statistics,
│   │                       #   the orchestrator), topology.py, shape.py, bifurcation.py
│   │                       #   (junction morphometry + per-branch-order aggregation, which
│   │                       #   share the parent/daughter edge classification),
│   │                       #   network_measures.py (betweenness + communities),
│   │                       #   network analyses: inlet_outlet_routes (bottlenecks, shunts),
│   │                       #   transit_time, territories, current_flow, occlusion, loops,
│   │                       #   strahler, spectral, over _route_graph.py / _flow_system.py;
│   │                       #   csv_export.py, _sampling.py (one shared seed constant),
│   │                       #   three_dim_distances.py (cell-to-vessel distances)
│   ├── optimisation/       # "Optimise settings": search.py (segmentation cleanup,
│   │                       #   Skeletonise and Graph settings), fwhm_search.py (FWHM
│   │                       #   settings), candidates/metrics (+ fwhm_ variants), report,
│   │                       #   progress
│   ├── gui/                # napari plugin: form.py (schema -> form rows, pure),
│   │                       #   tabs.py (one tab per stage), progress.py (what the
│   │                       #   progress bars read, pure), results.py (what each
│   │                       #   stage puts in the viewer, pure), layers.py (an open
│   │                       #   layer -> run settings, pure), boundary_picking.py
│   │                       #   (boundary settings <-> napari Points/Shapes, pure),
│   │                       #   view_snap.py (XY/XZ/YZ plane -> dims order / camera
│   │                       #   directions, pure), perturbation_editing.py, graph_editor.py
│   │                       #   + graph_click.py (the Edit window), branch_hover.py,
│   │                       #   vessel_tubes.py, run_state.py, run_log.py + log_view.py,
│   │                       #   run_snapshot.py (.haemorun save/load), stage_checkpoints.py
│   │                       #   (re-run from a tab), optimise_progress.py, chrome_tooltips.py
│   │                       #   (hover text for non-setting controls), _widget.py (the
│   │                       #   panel), napari.yaml (npe2 manifest)
│   ├── visualization/      # plot.py, vtk_io.py, pipeline_artifacts.py,
│   │                       #   geometry.py (an edge -> a drawable polyline),
│   │                       #   dilation_curves.py, perturbation_plots.py, flow_direction.py,
│   │                       #   large_vessel_assignment.py, _helpers.py
│   ├── parsers/            # schema.py, config.py, cli.py, checks.py — the settings machinery
│   └── pipeline/           # A package, not a module: schema.py (the pipeline's 366 settings),
│                           #   settings.py, checks.py (preflight), stages.py (one
│                           #   function per stage + run_pipeline_stages), progress.py
│                           #   (the ordered STAGES + the progress callback), citations.py
│                           #   (what a run used, and how to cite it); public:
│                           #   default_schema, write_default_config, preflight
├── examples/               # Runnable pipelines and settings (not the core library API surface)
│   ├── resistance_network_pipeline.py        # Main example: config + CLI over haemolynx.pipeline
│   ├── brain_network_pipeline.py             # Whole-brain run: pipeline + pericyte dilation sweep
│   ├── carotid_image_to_model.py             # Carotid dataset: the same pipeline, its own config
│   ├── simple_network_haemodynamics.py       # Hand-built graph: the haemodynamics API on its own
│   ├── *_schema.py / *_config.yaml           # Settings an example adds on top of
│   │                                         #   haemolynx.pipeline.schema; configs are generated
│   ├── pipeline_presets.py                   # Named partial configs, validated against the schema
│   └── regenerate_configs.py                 # Rewrite every *_config.yaml from its schema
├── tutorials/
│   ├── pipeline_tutorial.ipynb   # **Source of truth** for the step-by-step tutorial
│   ├── pipeline_tutorial.py       # Auto-generated from the notebook (do not edit by hand)
│   ├── tutorial_plots.py          # Inline plot helpers for the notebook
│   └── export_notebook.py         # nbconvert export used by integration tests
├── scripts/                # Developer tools, not part of the package
│   ├── compare_branches.py # Run the pipeline on this branch and on a reference
│   │                       #   ref, and diff the numbers (never runs in CI)
│   └── branch_comparison/  # metrics/report (pure, unit-tested) + runner
├── tests/                  # Unit + integration tests (see Testing below)
│   ├── conftest.py         # Shared fixtures; matplotlib Agg backend
│   └── integration/        # Full-pipeline and tutorial tests (@pytest.mark.integration)
└── tests/data/             # Small TIFF/H5 fixtures for tests and tutorial input
```

> **Note:** there is an active repo-cleanup effort — see **“Cleanup plan”** at the end of this file. Some filenames above are slated to change.

---

## Segmentation (ilastik)

| In scope (this repo) | Out of scope (manual, upstream) |
|----------------------|----------------------------------|
| Headless ilastik inference via `io.run_ilastik_headless_segmentation` | Training / labelling in ilastik GUI |
| Wiring in `pipeline/stages.py` (`segment` for the main mask, `build_network` for the large/small vessel masks via `io.load_and_validate_vessel_masks`), driven by `use_ilastik_segmentation` and the large/small vessel ilastik flags | Choosing features, labels, or project hyperparameters |
| Loading ilastik-produced masks into the rest of the pipeline | Installing ilastik itself on the user machine |

**Three ilastik hooks** in the main pipeline (all optional):

1. **Main vessel mask** — `use_ilastik_segmentation=True` segments the primary input before skeletonization (`ilastik_unsegmented_image_path`, `ilastik_classifier_path`, `ilastik_executable`).
2. **Large arteriole/venule masks** — `use_ilastik_large_vessel_segmentation` (requires `use_large_vessel_masks=True`).
3. **Small arteriole/venule masks** — `use_ilastik_small_vessel_segmentation` (requires `use_small_vessel_masks_for_boundary_assignment=True`).

Shared settings: `ilastik_output_dir`, `ilastik_output_suffix`. Every one of them is declared in `pipeline/schema.py`, with its prerequisites; `pipeline/checks.py` (`haemolynx.pipeline.preflight`) is what tells a user which required paths are missing before any work starts.

Users may also supply **pre-segmented** masks only (no ilastik call) — typical for tutorial data and many tests.

---

## End-to-end pipeline (conceptual)

0. **Segmentation (optional)** — ilastik headless on raw TIFF/H5 → binary mask; or skip if input is already segmented  
1. **Load & skeletonize** — `io.load_and_skeletonize_3d_tif` / `_h5`; `preprocessing.preprocess_skeleton_for_graph`  
2. **Vessel masks (optional)** — `io.load_and_validate_vessel_masks` (large/small arteriole/venule; from disk or ilastik)  
3. **Graph build** — `graph.build_graph_from_skeleton` (eleven topology steps in `graph/assemble.py`), then `graph.smooth_graph_centrelines`  
4. **Boundary & branch order** — manual volume/coordinates or mask-based assignment; `graph.assign_vessel_branch_orders` / hierarchical orders  
5. **Haemodynamics** — `haemodynamics.apply_poiseuille_haemodynamics`, conductance matrix, two-point resistance, flow solve (optionally iterating a distributed haematocrit, `haematocrit_model="distributed_iterative"`)  
6. **Perturbations (optional)** — `run_perturbations` re-solves copies of the solved network, once per configured entry in `perturbations` (see `haemodynamics/perturbations.py`)  
7. **Export & stats** — `visualization.graph_to_vtk`, `statistics.compute_comprehensive_vessel_statistics` (with the selected network analyses), vascular communities

`pipeline/stages.py` runs these as nine stage functions: `segment`, `skeletonise`, `build_network`,
`assign_boundaries`, `assign_diameters`, `build_haemodynamic_model`, `solve`, `run_perturbations`,
`export_results`. The vessel masks (step 2) are loaded at the start of `build_network`, and the
statistics, network analyses and communities (step 7) all run inside `export_results`.

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
| `src/haemolynx/io/` (incl. ilastik) | `tests/test_io.py`, `tests/test_load_and_validate_vessel_masks.py`, `test_load_2d.py`, `test_axis_order.py`, `test_voxel_validation.py`, `test_raw_volume_cache.py` |
| `src/haemolynx/preprocessing/` | `tests/test_preprocessing.py`, `test_skeleton_bridging.py`, `test_thick_vessel_skeletonisation.py`, `test_thick_vessel_braid_guard.py`, `test_segmentation_cleanup.py`, `test_segmentation_quality.py`, `test_segmentation_raw_comparison.py`, `test_memmap_*.py`, `test_low_ram_*.py` |
| `src/haemolynx/graph/` | `tests/test_graph.py`, `test_graph_assemble.py`, `tests/test_branch_order_hierarchy.py`, `test_centreline_smoothing.py`, `test_graph_communities.py`, `test_graph_edit.py`, `test_graph_thick_vessel_junctions.py`, boundary/assignment tests |
| `src/haemolynx/haemodynamics/` | `tests/test_hemodynamics.py`, `test_viscosity_laws.py`, `test_constriction.py`, `test_haematocrit_distribution.py`, `test_haemodynamics_automated_fwhm.py`, `test_haemodynamics_edt_diameter.py`, FWHM/pericyte integration tests |
| Perturbations and sweeps | `tests/test_perturbations.py` (entries, settings, preflight), `test_perturbation_stage.py` (running them), `test_perturbation_outputs.py` (files and layers), `test_pericyte_sweep.py`, `test_pericyte_geometry_sweep.py`, `test_capillary_scaling.py`, `test_arteriole_scaling.py`, `test_capillary_block.py`, `test_sweep_flow_layers.py` |
| `src/haemolynx/statistics/` | `tests/test_statistics.py`, `tests/test_three_dim_distances.py`, `test_network_analyses.py`, `test_inlet_outlet_routes.py`, `test_occlusion_and_current_flow.py`, `test_statistics_without_haemodynamics.py` |
| `src/haemolynx/optimisation/` | `tests/test_optimisation_*.py`, `test_fwhm_optimisation_*.py` |
| `src/haemolynx/visualization/` | `tests/test_visualization.py`, `tests/test_vtk_io.py`, `tests/test_visualization_geometry.py`, `test_pipeline_artifacts.py` |
| `src/haemolynx/gui/` | `tests/test_gui_<module>.py` for the pure modules (`test_gui_form.py`, `test_gui_tabs.py`, `test_gui_progress.py`, `test_gui_results.py`, `test_gui_layers.py`, `test_gui_boundary_picking.py`, `test_gui_view_snap.py`, …); `test_gui_*_widget.py`, `test_gui_widget.py` and `test_gui_view_panel.py` build the real panel; `test_gui_tooltips.py` checks every control has hover text |
| `src/haemolynx/pipeline/` | `tests/test_pipeline_schema_api.py`, `test_pipeline_progress.py`, `test_pipeline_invariants.py`, `test_segment_stage.py`, `test_skeletonise_stage.py`, `test_citations.py`, `test_setting_units.py`, `test_example_configs.py` |
| `src/haemolynx/parsers/` | `tests/test_parsers_schema.py`, `test_parsers_config.py`, `test_parsers_checks.py` |
| Any subpackage's `__all__` | `tests/test_public_api.py` (star-imports every subpackage) |
| Packaging / the installed wheel | `tests/test_installed_package.py`, `test_packaging_metadata.py`, `test_napari_manifest.py` |
| Full pipeline / examples | `tests/integration/test_image_to_model_pipeline.py`, `test_nerve_pipeline.py`, `test_axis_order_pipeline.py`, `test_simple_network_haemodynamics.py`, `test_progress_reporting.py`, `test_step_artifacts_setting.py` |
| Tutorial notebook | `tests/integration/test_pipeline_tutorial.py` (exports notebook → `.py`, then runs) |

### Tutorial notebook workflow

- **Edit** `tutorials/pipeline_tutorial.ipynb` only.
- **Regenerate** `tutorials/pipeline_tutorial.py` via:  
  `pytest tests/integration/test_pipeline_tutorial.py`
- Do **not** hand-edit `pipeline_tutorial.py`; it is overwritten by the integration test.

### CI

GitHub Actions (`.github/workflows/pytest-pr.yml`) runs four jobs on pull requests, all on
`ubuntu-latest`:

- **`pytest`** — `pip install -e .[dev]` and `pytest` across a Python 3.9 / 3.10 / 3.11 / 3.12 matrix
  (`fail-fast: false`, so one version failing still reports the others).
- **`installed-package`** — builds the wheel, installs it into an empty environment and runs
  `tests/installed_package_smoke.py` from outside the repository, then `tests/test_installed_package.py`.
- **`dependency-floors`** — the suite against the oldest supported versions (`constraints-floor.txt`).
- **`napari-gui`** — Python 3.11 with the `napari` extra and `pytest-qt`: `npe2 validate HaemoLynx`,
  then `xvfb-run -a pytest -m gui`.

Nothing in CI runs on Windows, so Windows-only problems (paths, `SYSTEMROOT`, Qt platform plugins)
are only caught locally.

---

## Coding conventions

- **Library code** lives under `src/haemolynx/`. Keep `examples/` thin — orchestration, CLI, presets.
- **Minimal diffs** — match existing naming, types, and import style in the touched module.
- **Settings** — every setting is declared exactly once, in `pipeline/schema.py`, and the examples
  add their own on top of it (`examples/*_schema.py`). A new ilastik path or flag goes there, with
  its `requires`, so `pipeline/checks.py` can check it and the config file and the CLI both grow a
  line for free.
- **Generated configs** — `examples/*_config.yaml` are written by `examples/regenerate_configs.py`
  from the schemas, never hand-edited: they carry each setting's help text, units and prerequisites
  as comments. Path values are serialised with `PurePath(...).as_posix()`, so regenerating on
  Windows reproduces the committed files byte for byte instead of rewriting every path with
  backslashes.
- **Input contract** — pipeline expects **binary vessel masks** at skeletonization time. Masks may come from pre-existing files or from **ilastik inference** in this repo; classifier training is always manual.
- **Axis order** — arrays are canonical `(z, y, x)`: axis 0 is the stack axis that overlays and
  projections look through. Loaders take `axis_order` (`"zyx"` default, any permutation of `xyz`)
  and transpose the input to canonical order, so **that setting is how a user picks which axis is z**
  (the `image_axis_order` setting). See `io/axis_order.py`.
- **Voxel sizes** — two orders, never mix them. Image metadata is physical **`(x, y, z)`**
  (`voxel_size_xyz`, `io.voxel_validation.resolve_voxel_size_xyz`); anything that scales array
  indices takes per-array-axis spacing in **`(z, y, x)`** (`voxel_size_zyx`). Convert once at the
  loader boundary with `io.voxel_size_zyx_from_xyz`. Passing `xyz` where `zyx` is expected silently
  swaps the z and x spacings — invisible for isotropic voxels, wrong for every real stack
  (see `tests/test_anisotropic_voxel_size.py`). Masks and main image must align in shape and
  physical voxel units. **Drawing a run has the same split the other way round:** node `pos` and
  edge `voxels` are physical microns already, so a layer built from the graph takes `scale=(1,1,1)`,
  while `image`, `skeleton` and the masks are voxel-indexed and take `scale=voxel_size_zyx`
  (see `gui/results.py` and its registration test).
- **Boundary coordinates** — every `*_node_coordinates` and `*_node_volumes` setting is physical
  `(z, y, x)` microns, the same units as node `pos`. The `coordinates` method snaps to the
  *nearest* terminal, so a point in voxel indices never fails, it just selects the wrong node;
  `graph.BoundaryCoordinateWarning` reports a snap that went too far and names the voxel-index
  reading when that is what it looks like (`graph/boundaries.py`).
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
- **Viscosity model** — selectable, and the choice changes every resistance a run produces, so
  **resistances are not comparable across laws or across diameter bases**. The graph records both
  in `G.graph["viscosity_law"]` / `["diameter_basis"]`. All of it is in
  `haemodynamics/viscosity.py`.
  - The question that decides the answer is **which diameter you measured**, not which law you
    prefer. `diameter_basis="plasma_column"` (**default**) means the segmented diameter is the
    channel the fluid occupies — what a plasma stain images, which is what this project's data is.
    `"anatomical"` means wall-to-wall, including the ~1.1 µm endothelial surface layer.
  - `viscosity_law="pries"` (**default**) is Pries et al., fitted 3.3–1978 µm, continuous across
    the whole tree, no placeholder and no warning. Reads `diameter_basis`: `plasma_column` selects
    the *in vitro* (tube) form, `anatomical` the *in vivo* form. The in vivo form's
    `(D/(D−1.1))²` factors appear squared, so the leading term carries `(D/(D−1.1))⁴` — that is
    the Poiseuille correction for quoting a resistance against a diameter wider than the channel,
    i.e. **a diameter correction, not a property of blood**. Applying it to a plasma-column
    diameter subtracts the glycocalyx twice and costs ~2.5× in a capillary.
  - `viscosity_law="capillary_power_law"` is the law used before: `µ(d) = 3.0 mPa·s · (5 µm/d)^1.647`
    up to 7 µm, then a constant 3.5 mPa·s, raising `haemodynamics.PlaceholderViscosityWarning`
    between 7 and 100 µm. Kept for comparison with earlier results. It agrees with Pries in vitro
    to 2% at 3 µm and diverges from there (2.0× at 5 µm, 3.2× at 7 µm): a one-point calibration
    whose slope is too steep, crossing below plasma viscosity at 8.7 µm.
  - `viscosity_law="constant"` is plasma everywhere, for separating geometry effects from
    viscosity ones.
  - Switching the default from the power law to Pries **roughly doubles capillary resistance**
    (2.03× at 5 µm). `tests/test_viscosity_laws.py` pins the ratios at each diameter.
- **Skeletonization** — use `skimage.morphology.skeletonize(..., method="lee")` via
  `preprocessing.skeletonize_volume`, not deprecated `skeletonize_3d`. Both loaders skeletonize
  through the one `io.load._skeletonize_loaded_volume` helper, which binarizes, skeletonizes **and
  fills holes**: the TIFF and H5 paths had drifted apart on that last step, so one volume saved in
  two formats produced two different graphs. Anything that adds a third loader calls the same helper.
- **Comments** — only for non-obvious domain logic; prefer self-explanatory code.

---

## Key modules (quick reference)

- **`io/axis_order.py`** — canonical `(z, y, x)` convention: `normalize_axis_order`, `apply_axis_order`
  (transposes an input volume to canonical order), `voxel_size_zyx_from_xyz` / `voxel_size_xyz_from_zyx`.
- **`io/ilastik.py`** — `run_ilastik_headless_segmentation` (subprocess call to user-installed ilastik + `.ilp` project).
- **`io/automated_vessel_assignment.py`** — mask **loading & validation**: `load_large_vessel_masks`, `load_and_validate_vessel_masks` (includes ilastik path for large/small masks). *Despite the name, this is I/O, not graph assignment.*
- **`graph/automated_vessel_assignment.py`** — graph **terminal-node assignment** from masks: `select_terminal_nodes_from_large_vessel_masks`, `infer_boundary_nodes_from_small_vessel_masks`, overlap-resolution + 3D HTML diagnostics. *Same filename as the io module but a different concern — a known source of confusion (see Cleanup Plan).*
- **`graph/assemble.py`** — `build_graph_from_skeleton`; optional `step_callback(G, label)` after each topology step, one per label in `STEP_LABELS` (eleven of them).
- **`graph/smoothing.py`** — `smooth_graph_centrelines`: takes the voxel staircase out of each
  centreline and **re-measures `length`**, which moves every resistance (a path stepping voxel to
  voxel comes back ~7% longer than the vessel it traces). It is *not* one of the `STEP_LABELS` —
  it runs in `pipeline/stages.py`'s `build_network`, after the topology steps and before the graph
  is pickled — so a caller assembling a graph by hand gets no smoothing unless it asks. A smoothed
  path is only accepted if it stays within `max_deviation` of a skeleton voxel and is no longer than
  the path it came from; each edge records which of `smoothed` / `relaxed` / `kept_raw` /
  `too_short` happened to it.
- **`pipeline/stages.py`** — one function per stage (`segment`, `skeletonise`, `build_network`,
  `assign_boundaries`, `assign_diameters`, `build_haemodynamic_model`, `solve`,
  `run_perturbations`, `export_results`), each taking settings plus the previous stage's
  dataclass, so a caller can run them one at a time and intervene. `run_pipeline_stages` is the
  orchestrator that runs all nine in order and returns the graph; its `start_from` / `resume`
  arguments let the panel re-run from a chosen stage using a saved run's state instead of
  recomputing the earlier ones.
- **`pipeline/progress.py`** — `STAGES`, the run's stages in order (the panel draws one tab
  per entry and a progress bar counts them — one list, not two). It has ten entries for nine
  stage functions: `solve` has no tab of its own (it shares **6. Haemodynamics**), and
  **8. Additional measurements** is a tab with no stage function (its settings are read by
  `export_results`). Plus what a run reports through:
  `run_pipeline_stages(settings, schema, progress=callback)` hands the callback a `ProgressEvent`
  as each stage starts, finishes or fails, and one per topology step inside graph building.
  `log_progress` is the ready-made console consumer; the napari panel's bars are the other one.
  Nothing here imports a GUI or a progress-bar library — the callback is the whole mechanism.
- **`pipeline/checks.py`** — `preflight(settings, schema)`, the pre-run checklist: the schema's own
  checks (`parsers/checks.py`) plus what only this pipeline knows — the artefact a skipped stage
  must have left behind, and whether the ilastik executable can be found. The examples call it
  before doing any work and exit if it fails.
- **`haemodynamics/automated.py`** — FWHM vessel-**diameter** measurement from raw TIFF (`measure_edge_diameters_fwhm_from_raw_tiff`, `build_graph_branch_label_volume`; neither is re-exported from `haemolynx.haemodynamics`, so import them from the module). *“automated” is a misnomer; this is diameter estimation.* `haemodynamics/edt_diameter.py` is the cross-check: each edge's diameter from the mask's own inscribed radius.
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
  discharge haematocrit from Pries & Secomb's phase-separation law, iterated with the flow solve.
  The default, `"fixed"`, uses one `haematocrit` everywhere.
- **`haemodynamics/perturbations.py`** — a perturbation is a named, typed settings override
  (`PerturbationSpec`) that `pipeline/stages.py`'s `run_perturbations` re-solves a copy of the
  solved network for, writing into `{name}_{type}` under `perturbation_output_dir`. Thirteen types
  plus `none` (`PERTURBATION_TYPES`): arteriole / pericyte / combined diameter changes, capillary
  blocks, and sweeps over dilation, inlet pressure, pericyte spacing or length.
  `SETTINGS_FOR_TYPE` says which settings each type reads; an override outside it is reported as
  unused and not applied. `INCOMPARABLE_OVERRIDES` (viscosity law, diameter basis, haematocrit,
  haematocrit model) are refused, because they would move every resistance and the difference
  against the baseline would not be the perturbation's effect. A single re-solve writes
  summary/edges/statistics CSVs and plots; a sweep writes its grid CSV and curves, and keeps
  per-point flows (`sweep_flows.py`) so the viewer's slider can swap them without N graph copies.
- **Network analyses** (`statistics/inlet_outlet_routes.py`, `transit_time.py`, `territories.py`,
  `current_flow.py`, `occlusion.py`, `loops.py`, `strahler.py`, `spectral.py`) — each a
  `statistics_*` toggle in the schema's *Connectivity/Network Analysis* section, run inside
  `compute_comprehensive_vessel_statistics`. They add rows to the statistics CSV, and most also
  write per-vessel edge attributes that the vessels layer offers as colour options (declared in
  `gui/results.py`'s `OPTIONAL_EDGE_COLUMNS`). An analysis that cannot run (no inlets/outlets, no
  solved flow) writes nothing, so its colour option stays absent. `statistics_route_weighting`
  picks the conductance model (topology / length / resistance) the Laplacian-based ones share
  through `_flow_system.py`.
- **Vascular communities** (`graph/communities.py`) — the `compute_vascular_communities` toggle,
  independent of Statistics; when Statistics also runs, the two share one partition
  (`pipeline/stages.py`'s `shared_vascular_community_partition`) instead of computing it twice.
- **`preprocessing/thick_vessels.py`** — `use_thick_vessel_skeletonisation`: Lee thinning of the
  fat part of a plasma-labelled mask gives a medial *sheet* (several polylines for one vessel), so
  the fat region (an inscribed-radius gate, not a volume gate) gets a centreline *tree* instead,
  and the thin capillaries fused to it are joined onto that tree; `thick_vessel_braid_guard.py`
  flags braided segments that remain. `graph/thick_vessel_junctions.py` then splits each joined
  edge where it crosses the fat-mask boundary and tags the part inside as an `IS_ZERO_RESISTANCE`
  bridge — the small vessel opening into the lumen, not vessel material (see `apply.py` above).
- **`optimisation/`** — the panel's **Optimise settings**: a sequential, image-informed search
  over segmentation cleanup, Skeletonise and Graph settings (`search.py`) and, separately, the
  FWHM diameter settings (`fwhm_search.py`). Candidates come from the image itself
  (`candidates.py`), trials are scored by pure functions (`metrics.py`), and the result is a
  config file plus a readable report (`report.py`). **Check segmented image** is
  `preprocessing/segmentation_quality.py` + `segmentation_raw_comparison.py`.
- **`gui/run_snapshot.py`** / **`gui/stage_checkpoints.py`** — save a finished run to a
  `.haemorun` file and load it back; keep per-stage snapshots so a tab can re-run from its stage
  (`run_pipeline_stages(start_from=..., resume=...)`) without rebuilding everything before it.
- **`statistics/three_dim_distances.py`** — cell-to-vessel distances.
- **`visualization/geometry.py`** — `edge_polyline`: an edge's `voxels` (or its two node positions)
  turned into a polyline that runs `u`→`v` and touches both nodes. The VTK export, the pericyte
  point derivation and two plotly writers each answered that separately, and drew vessels in
  slightly different places; this is the one answer.
- **`gui/boundary_picking.py`** — the four boundary roles' coordinate and volume settings as napari
  Points and Shapes layers, and back. Pure, like the rest of `gui/` bar `_widget.py` and
  `log_view.py` (the only two that touch Qt, and only inside functions): settings in,
  layer specs out, layer data in, settings out, nothing importing napari. `rectangle_from_box` and
  `box_from_rectangle` are exact inverses, which is what lets an edited layer *be* the setting.
- **`examples/pipeline_presets.py`** — `PRESETS`, named partial configs; every setting name is
  checked against the schema at import, so a preset cannot quietly set something that no longer
  exists. The override engine itself is library code, in `parsers/cli.py` and `parsers/config.py`.

---

## What to avoid

- Committing large generated outputs (`examples/outputs/`, `tutorials/plots/`, `tests/outputs/`) unless the user explicitly asks.
- Editing auto-generated `tutorials/pipeline_tutorial.py` instead of the notebook.
- Adding features without tests (see Testing policy).
- Force-pushing `main`/`master` or amending pushed commits unless explicitly requested.
- Expanding scope into unrelated refactors when fixing a targeted issue.

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

# Cleanup plan (TEMPORARY — delete this whole section once implemented)

> **Status:** partly implemented — Phases 0, 1 and 3 are done, Phase 4 nearly (only the README
> note is left), Phase 2 one rename of four. This is a living checklist. Tick items as PRs land, and
> **remove this entire section** when every phase is done. Until then, treat the descriptions above
> as the *current* state and the items below as the *target* state.
>
> **Baseline (Windows):** the suite has grown from 1,188 to **3,794 tests collected under
> `pytest -m "not slow"`** (4,041 in total, 166 test files) as of 2026-09-25; a fresh full-run pass
> count has not been recorded since. Record one before starting Phase 2, and re-run it (and the
> full `pytest`, including `slow`/`integration`) after **every** rename; no phase may reduce the
> green count. Every behaviour change needs a test (see Testing policy above).

### Goals
Remove duplication, kill dead code, give modules names that match what they do, and make the
examples thin — without regressing the test suite. Work in small, reviewable phases on `devel`,
running tests between each.

### Findings driving this plan
- **Two files named `automated_vessel_assignment.py`** (`io/` = mask loading, `graph/` = terminal-node
  assignment). No shared code — just a confusing name collision.
- **`haemodynamics/automated.py`** is FWHM diameter measurement, not “automation”.
- ~~**`graph/__init__.py` bug:** imports `create_merged_edge_attributes` twice and lists
  `create_merged_edge_attributes_simple` / `_full` in `__all__` — neither is imported, so
  `from haemolynx.graph import *` raises `AttributeError`.~~ **DONE** — see Phase 1.
- ~~**Dependency drift:** `requirements.txt` and `pyproject.toml` disagree.~~ **DONE** —
  `requirements.txt` is deleted; `pyproject.toml` is the only source.
- ~~**Dead code:** `examples/OLD/` (two pre-refactor scripts, imported by nothing).~~ **DONE** — deleted.
- ~~The settings/preset system (`resistance_pipeline_settings.py` + `presets.py` + `preflight.py` +
  `wizard.py`) is **not** duplicated — it’s a clean layered design.~~ **SUPERSEDED** — those four
  files are gone. The constants became schema declarations in `pipeline/schema.py`, `preflight`
  became `pipeline/checks.py`, the override engine became `parsers/cli.py` + `parsers/config.py`,
  and all that is left beside the examples is `pipeline_presets.py`. The wizard was dropped: the
  generated config file, which documents every setting inline, does its job.
- **`examples/carotid_image_to_model.py`** was orphaned and could not even be imported
  (`PLOT_DIR` undefined; `__main__` passed an argument the entry point did not take) — **fixed**:
  it survives as a 91-line runner over `carotid_schema.py` + `carotid_config.yaml`, calling
  `haemolynx.pipeline.run_pipeline_stages` like the other examples. Dale’s segmentation work slots
  into its `use_ilastik_segmentation` path.

### Phase 0 — Housekeeping & safety net (lowest risk)
- [x] Delete `examples/OLD/`.
- [x] Confirm `.gitignore` covers both `.venv/` **and** `venv/`, plus `__pycache__/`, `examples/outputs/`,
      `examples/plots/`, `tutorials/outputs/`, `tutorials/plots/`, `tests/outputs/`, `tests/plots/`,
      `examples/images/`. (None are tracked today — keep it that way.) All ten are listed, plus
      `/outputs/` and `/plots/` for a run started from the repository root.
- [x] Make dependencies single-source: `requirements.txt` deleted, `pyproject.toml` authoritative.
      `pytest` lives in the `dev` extra only and `ipykernel` moved to a new `notebook` extra.
- [x] Run full `pytest`. Commit.

### Phase 1 — Fix the `graph/__init__.py` star-import bug — **DONE**
- [x] Remove the duplicate `create_merged_edge_attributes` import; reconcile `__all__` with what is
      actually imported. The phantom `_simple`/`_full` names are gone, the duplicate import is gone,
      and `from haemolynx.graph import *` succeeds.
- [x] Add a regression test that does `from haemolynx.graph import *` and asserts every name in
      `__all__` is importable, and the same guard for the other subpackages. It landed as
      `tests/test_public_api.py` (not the `test_graph_public_api.py` this plan proposed, because it
      guards every subpackage and not just `graph`), and it also catches duplicates in `__all__` and
      modules whose name is not a valid identifier.
- [x] Run full `pytest`. Commit.

### Phase 2 — Rename modules for clarity (mechanical, test-guarded)
Rename + update all imports in `src/`, `examples/`, `tests/`, and the `__init__.py` re-exports.
Keep the **public function names** the same so the API surface doesn’t move; only file/module names change.
- [ ] `io/automated_vessel_assignment.py` → `io/vessel_masks.py`.
- [ ] `graph/automated_vessel_assignment.py` → `graph/terminal_node_assignment.py`.
- [ ] `haemodynamics/automated.py` → `haemodynamics/fwhm_diameter.py`. It now has many more
      importers than when this item was written: in `src/`, `haemodynamics/__init__.py`, `apply.py`,
      `edt_diameter.py`, `statistics/three_dim_distances.py`, `preprocessing/segmentation_quality.py`,
      `optimisation/__init__.py`, `fwhm_search.py`, `fwhm_candidates.py` and `gui/_widget.py`; plus
      about eight test files (some patch `haemodynamics.automated.<name>` by string, which a plain
      import search misses — grep for the dotted path too).
- [x] `statistics/3D_distances.py` → `statistics/three_dim_distances.py`; the `importlib` hack in
      `statistics/__init__.py` is gone, replaced by a normal
      `from .three_dim_distances import ...`. `tests/test_3d_distances.py` →
      `tests/test_three_dim_distances.py`.
- [ ] Use `git mv` so history is preserved. Run full `pytest` after each rename. Commit per rename.

### Phase 3 — De-duplicate the whole-brain workflow — **DONE**
- [x] The stage runner moved to `src/haemolynx/pipeline/` (a package: `schema`, `settings`, `checks`,
      `stages`, `progress`), so examples share it instead of forking it.
- [x] The pressure/boundary-flow solve and the pericyte dilation sweep moved to
      `src/haemolynx/haemodynamics/pericyte_sweep.py`; the curve plots to
      `src/haemolynx/visualization/dilation_curves.py`.
- [x] `resistance_network_pipeline_for_Alice.py` (1,795 lines) and root-level `AlicePaper.py` are
      replaced by `examples/brain_network_pipeline.py` (84 lines) plus `brain_pipeline_config.yaml`.
      "Alice" is gone from the names; the sweep is described by what it does.
- [x] `tests/test_alice.py` → `tests/test_pericyte_sweep.py`, driving the extracted module.

### Phase 4 — Thin out the examples / consolidate config (larger, do last)
- [x] Stage orchestration lifted into `src/haemolynx/pipeline/`; the example is now config + CLI
      glue (1,282 → 107 lines) and `brain_network_pipeline.py` (84 lines) runs the same stages.
- [x] Split `run_pipeline_stages` into one function per stage. `pipeline/stages.py` has eight —
      `segment`, `skeletonise`, `build_network`, `assign_boundaries`, `assign_diameters`,
      `build_haemodynamic_model`, `solve`, `export_results` — each taking `settings` first and the
      previous stage's dataclass after it. A ninth, `run_perturbations`, has since joined them, and
      `run_pipeline_stages` has grown to ~115 lines with `start_from`/`resume` for re-running from
      a stage. Per-stage *unit* tests are still thin: `segment`, `skeletonise` and
      `run_perturbations` have files of their own (`tests/test_segment_stage.py`,
      `test_skeletonise_stage.py`, `test_perturbation_stage.py`), and the rest are reached through
      `tests/test_pipeline_invariants.py` and the integration runs.
- [ ] Add a short “preset system” note to the README (schema declarations → preset dicts → CLI/YAML
      overrides), since the layering isn’t obvious from filenames. The README does not mention
      presets at all today.
- [x] ~~Populate `examples/local_presets.py` with one realistic example preset.~~ Dropped: the stub is
      gone, and `examples/pipeline_presets.py` ships nine worked presets validated against the schema.

### When done
- [ ] Full `pytest` (incl. `slow`/`integration`) green; tutorial notebook still exports & runs.
- [ ] Update the **Repository layout** and **Key modules** sections above to the new names, drop the
      “Note”/misnomer caveats, and **delete this entire Cleanup plan section.**
