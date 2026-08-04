# ImageLynx — repository guide for AI assistants

ImageLynx turns 3D microvascular microscopy into **NetworkX graphs** with haemodynamic edge weights, VTK exports, and network statistics. The pipeline accepts **already-segmented** binary masks (TIFF/H5) or can **run segmentation in-repo via ilastik** (headless) when configured.

**Segmentation is in scope.** ImageLynx integrates ilastik for pixel classification inference on the main vessel volume and on optional large/small arteriole–venule channels. What stays **outside** this repo is **training** ilastik projects (`.ilp` classifiers): users train those manually in the [ilastik](https://www.ilastik.org/) GUI, then point the pipeline at the exported project and raw/unsegmented images.

---

## Repository layout

```
ImageLynx/
├── src/ImageLynx/          # Installable package (setuptools, pythonpath = src in pytest)
│   ├── io/                 # load.py, ilastik.py, voxel_validation.py, axis_order.py,
│   │                       #   automated_vessel_assignment.py (mask loading/validation)
│   ├── preprocessing/      # skeleton.py — skeletonize, bridge, clean, bundle refinement
│   ├── graph/              # assemble.py (orchestrator) + build, reconnect, optimise, degree2,
│   │                       #   prune, collapse, branch_order, boundaries, large_vessels,
│   │                       #   diagnostics, validate, _helpers,
│   │                       #   automated_vessel_assignment.py (terminal-node assignment)
│   ├── haemodynamics/      # poiseuille, resistance, pipeline, automated.py (FWHM diameters),
│   │                       #   probability, pericyte_mask, pericyte_comparison
│   ├── statistics/         # stats.py, 3D_distances.py (cell-to-vessel; imported via importlib)
│   └── visualization/      # plot.py, vtk_io.py, pipeline_artifacts.py, _helpers.py
├── examples/               # Runnable pipelines and settings (not the core library API surface)
│   ├── resistance_network_pipeline.py        # Main example: config + CLI over ImageLynx.pipeline
│   ├── brain_network_pipeline.py             # Whole-brain run: pipeline + pericyte dilation sweep
│   ├── *_schema.py / *_config.yaml           # Settings: declared once, generated config files
│   ├── resistance_pipeline_settings.py       # Legacy constants (presets/wizard still read these)
│   ├── presets.py          # Preset definitions + CLI/YAML override engine
│   ├── local_presets.py    # User-local preset overrides (stub)
│   ├── preflight.py        # Pre-run validation checklist
│   ├── wizard.py           # Interactive setup
│   └── carotid_image_to_model.py  # Carotid dataset: the same pipeline, its own config
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
| Wiring in `examples/resistance_network_pipeline.py` (`use_ilastik_segmentation`, large/small vessel ilastik flags) | Choosing features, labels, or project hyperparameters |
| Loading ilastik-produced masks into the rest of the pipeline | Installing ilastik itself on the user machine |

**Three ilastik hooks** in the main pipeline (all optional):

1. **Main vessel mask** — `use_ilastik_segmentation=True` segments the primary input before skeletonization (`ilastik_unsegmented_image_path`, `ilastik_classifier_path`, `ilastik_executable`).
2. **Large arteriole/venule masks** — `use_ilastik_large_vessel_segmentation` (requires `use_large_vessel_masks=True`).
3. **Small arteriole/venule masks** — `use_ilastik_small_vessel_segmentation` (requires `use_small_vessel_masks_for_boundary_assignment=True`).

Shared settings: `ilastik_output_dir`, `ilastik_output_suffix`. See `examples/resistance_pipeline_settings.py` and `examples/preflight.py` for required paths when flags are enabled.

Users may also supply **pre-segmented** masks only (no ilastik call) — typical for tutorial data and many tests.

---

## End-to-end pipeline (conceptual)

0. **Segmentation (optional)** — ilastik headless on raw TIFF/H5 → binary mask; or skip if input is already segmented  
1. **Load & skeletonize** — `io.load_and_skeletonize_3d_tif` / `_h5`; `preprocessing.preprocess_skeleton_for_graph`  
2. **Vessel masks (optional)** — `io.load_and_validate_vessel_masks` (large/small arteriole/venule; from disk or ilastik)  
3. **Graph build** — `graph.build_graph_from_skeleton` (multi-step topology pipeline in `graph/assemble.py`)  
4. **Boundary & branch order** — manual volume/coordinates or mask-based assignment; `graph.assign_vessel_branch_orders` / hierarchical orders  
5. **Haemodynamics** — `haemodynamics.apply_poiseuille_haemodynamics`, conductance matrix, two-point resistance, flow solve  
6. **Export & stats** — `visualization.graph_to_vtk`, `statistics.compute_comprehensive_vessel_statistics`

The production-style entry point is `examples/resistance_network_pipeline.py` (`image_to_model_pipeline`). The tutorial (`tutorials/pipeline_tutorial.ipynb`) documents segmentation as a prerequisite stage and runs on pre-segmented fixture data by default.

---

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- **Python:** ≥ 3.9 (CI matrix: 3.9, 3.10, 3.11, 3.12)
- **Dependencies:** `pyproject.toml` only — there is no `requirements.txt`. Extras: `dev`
  (pytest, nbconvert, nbformat), `notebook` (ipykernel, for running the tutorial interactively).
- **Run package tests:** from repo root: `pytest -s` or `pytest`
- **Skip slow tests:** `pytest -m "not slow"`
- **Integration only:** `pytest -m integration`
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
6. **Mark appropriately** — `slow`, `integration`, `plotting` per `pyproject.toml` markers.

### Where to put tests

| Change area | Typical test location |
|-------------|----------------------|
| `src/ImageLynx/io/` (incl. ilastik) | `tests/test_io.py`, `tests/test_load_and_validate_vessel_masks.py` |
| `src/ImageLynx/preprocessing/` | `tests/test_preprocessing.py` |
| `src/ImageLynx/graph/` | `tests/test_graph.py`, `tests/test_branch_order_hierarchy.py`, boundary/assignment tests |
| `src/ImageLynx/haemodynamics/` | `tests/test_hemodynamics.py`, FWHM/pericyte integration tests |
| `src/ImageLynx/statistics/` | `tests/test_statistics.py`, `tests/test_3d_distances.py` |
| `src/ImageLynx/visualization/` | `tests/test_visualization.py`, `tests/test_vtk_io.py` |
| Full pipeline / examples | `tests/integration/test_image_to_model_pipeline.py`, `test_nerve_pipeline.py` |
| Tutorial notebook | `tests/integration/test_pipeline_tutorial.py` (exports notebook → `.py`, then runs) |

### Tutorial notebook workflow

- **Edit** `tutorials/pipeline_tutorial.ipynb` only.
- **Regenerate** `tutorials/pipeline_tutorial.py` via:  
  `pytest tests/integration/test_pipeline_tutorial.py`
- Do **not** hand-edit `pipeline_tutorial.py`; it is overwritten by the integration test.

### CI

GitHub Actions (`.github/workflows/pytest-pr.yml`) runs `pip install -e .[dev]` and `pytest` on pull
requests, across a Python 3.9 / 3.10 / 3.11 / 3.12 matrix (`fail-fast: false`, so one version failing still
reports the others).

---

## Coding conventions

- **Library code** lives under `src/ImageLynx/`. Keep `examples/` thin — orchestration, CLI, presets.
- **Minimal diffs** — match existing naming, types, and import style in the touched module.
- **Input contract** — pipeline expects **binary vessel masks** at skeletonization time. Masks may come from pre-existing files or from **ilastik inference** in this repo; classifier training is always manual. Document new ilastik-related paths/flags in `resistance_pipeline_settings.py` and `preflight.py`.
- **Axis order** — arrays are canonical `(z, y, x)`: axis 0 is the stack axis that overlays and
  projections look through. Loaders take `axis_order` (`"zyx"` default, any permutation of `xyz`)
  and transpose the input to canonical order, so **that setting is how a user picks which axis is z**
  (`IMAGE_AXIS_ORDER` in `examples/resistance_pipeline_settings.py`). See `io/axis_order.py`.
- **Voxel sizes** — two orders, never mix them. Image metadata is physical **`(x, y, z)`**
  (`voxel_size_xyz`, `io.voxel_validation.resolve_voxel_size_xyz`); anything that scales array
  indices takes per-array-axis spacing in **`(z, y, x)`** (`voxel_size_zyx`). Convert once at the
  loader boundary with `io.voxel_size_zyx_from_xyz`. Passing `xyz` where `zyx` is expected silently
  swaps the z and x spacings — invisible for isotropic voxels, wrong for every real stack
  (see `tests/test_anisotropic_voxel_size.py`). Masks and main image must align in shape and
  physical voxel units.
- **Graph** — `nx.MultiGraph` with `pos` on nodes and `voxels` on edges, both in physical
  `(z, y, x)` microns; haemodynamics uses `branch_order` on edges.
- **Edge attributes & units** — `length` (µm), `resistance` (Pa·s/m³), `conductance` (m³/(Pa·s)).
  `resistance` and `conductance` are always written together via
  `haemodynamics.poiseuille.set_edge_resistance`. **There is no `weight` attribute** — it used to
  mean physical length at build time and conductance after haemodynamics ran, so statistics read
  conductances back as microns. `graph.assert_no_forbidden_edge_attributes(G)` raises if it
  reappears; NetworkX algorithms must be passed an explicit `weight="length"` / `weight="resistance"`
  rather than relying on their `"weight"` default.
- **Viscosity model** — piecewise: `µ(d) = 3.0 mPa·s · (5 µm / d)^1.647` up to **7 µm** (above
  ~8.7 µm it would predict blood thinner than plasma), then a constant large-vessel
  `µ = 3.5 mPa·s`. The constant branch is a placeholder — it steps up discontinuously at 7 µm, so
  7–8.4 µm resistances are overestimated. Diameters from 7 µm to 100 µm therefore raise
  `haemodynamics.PlaceholderViscosityWarning`; above ~100 µm the constant is close to the true
  macroscale value and no warning is raised. Modelling the real transition is issue #90, and
  making the law selectable is issue #85.
- **Skeletonization** — use `skimage.morphology.skeletonize(..., method="lee")` via `preprocessing.skeletonize_volume`, not deprecated `skeletonize_3d`.
- **Comments** — only for non-obvious domain logic; prefer self-explanatory code.

---

## Key modules (quick reference)

- **`io/axis_order.py`** — canonical `(z, y, x)` convention: `normalize_axis_order`, `apply_axis_order`
  (transposes an input volume to canonical order), `voxel_size_zyx_from_xyz` / `voxel_size_xyz_from_zyx`.
- **`io/ilastik.py`** — `run_ilastik_headless_segmentation` (subprocess call to user-installed ilastik + `.ilp` project).
- **`io/automated_vessel_assignment.py`** — mask **loading & validation**: `load_large_vessel_masks`, `load_and_validate_vessel_masks` (includes ilastik path for large/small masks). *Despite the name, this is I/O, not graph assignment.*
- **`graph/automated_vessel_assignment.py`** — graph **terminal-node assignment** from masks: `select_terminal_nodes_from_large_vessel_masks`, `infer_boundary_nodes_from_small_vessel_masks`, overlap-resolution + 3D HTML diagnostics. *Same filename as the io module but a different concern — a known source of confusion (see Cleanup Plan).*
- **`graph/assemble.py`** — `build_graph_from_skeleton`; optional `step_callback(G, label)` after each topology step.
- **`haemodynamics/automated.py`** — FWHM vessel-**diameter** measurement from raw TIFF (`measure_edge_diameters_fwhm_from_raw_tiff`, `build_graph_branch_label_volume`). *“automated” is a misnomer; this is diameter estimation.*
- **`haemodynamics/pipeline.py`** — high-level Poiseuille application used by examples and tutorial.
- **`statistics/3D_distances.py`** — cell-to-vessel distances. Module name starts with a digit, so it can’t be imported normally — `statistics/__init__.py` pulls it in via `importlib.import_module`.
- **`examples/resistance_pipeline_settings.py`** — default constants and ilastik toggles; the preset/override engine lives in `examples/presets.py`.

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

> **Status:** approved, not yet implemented. This is a living checklist. Tick items as PRs land,
> and **remove this entire section** when every phase is done. Until then, treat the descriptions
> above as the *current* state and the items below as the *target* state.
>
> **Baseline at time of writing:** `pytest -m "not slow"` → **99 passed, 1 skipped, 8 deselected**.
> Re-run this (and the full `pytest`, including `slow`/`integration`) after **every** phase; no phase
> may reduce the green count. Every behaviour change needs a test (see Testing policy above).

### Goals
Remove duplication, kill dead code, give modules names that match what they do, and make the
examples thin — without regressing the test suite. Work in small, reviewable phases on `devel`,
running tests between each.

### Findings driving this plan
- **Two files named `automated_vessel_assignment.py`** (`io/` = mask loading, `graph/` = terminal-node
  assignment). No shared code — just a confusing name collision.
- **`haemodynamics/automated.py`** is FWHM diameter measurement, not “automation”.
- **`statistics/3D_distances.py`** starts with a digit → can’t be imported normally; loaded via an
  `importlib` hack in `statistics/__init__.py`.
- **`graph/__init__.py` bug:** imports `create_merged_edge_attributes` twice and lists
  `create_merged_edge_attributes_simple` / `_full` in `__all__` — neither is imported, so
  `from ImageLynx.graph import *` raises `AttributeError`. (Confirmed reproducible.)
- ~~**Dependency drift:** `requirements.txt` and `pyproject.toml` disagree.~~ **DONE** —
  `requirements.txt` is deleted; `pyproject.toml` is the only source.
- ~~**Dead code:** `examples/OLD/` (two pre-refactor scripts, imported by nothing).~~ **DONE** — deleted.
- The settings/preset system (`resistance_pipeline_settings.py` + `presets.py` + `preflight.py` +
  `wizard.py`) is **not** duplicated — it’s a clean layered design. Leave its logic alone; just document.
- **`examples/carotid_image_to_model.py`** was orphaned and could not even be imported
  (`PLOT_DIR` undefined; `__main__` passed an argument the entry point did not take) — **fixed**:
  it is now `carotid_schema.py` + `carotid_config.yaml` over `ImageLynx.pipeline`, like the other
  examples. Dale’s segmentation work slots into its `use_ilastik_segmentation` path.

### Phase 0 — Housekeeping & safety net (lowest risk)
- [x] Delete `examples/OLD/`.
- [ ] Confirm `.gitignore` covers both `.venv/` **and** `venv/`, plus `__pycache__/`, `examples/outputs/`,
      `examples/plots/`, `tutorials/outputs/`, `tutorials/plots/`, `tests/outputs/`, `tests/plots/`,
      `examples/images/`. (None are tracked today — keep it that way.)
- [x] Make dependencies single-source: `requirements.txt` deleted, `pyproject.toml` authoritative.
      `pytest` lives in the `dev` extra only and `ipykernel` moved to a new `notebook` extra.
- [ ] Run full `pytest`. Commit.

### Phase 1 — Fix the `graph/__init__.py` star-import bug (small, high value)
- [ ] Remove the duplicate `create_merged_edge_attributes` import; reconcile `__all__` with what is
      actually imported (drop the phantom `_simple`/`_full` names, or import the real symbols if they
      exist in `_helpers.py`).
- [ ] Add a regression test (e.g. `tests/test_graph_public_api.py`) that does `from ImageLynx.graph import *`
      and asserts every name in `__all__` is importable. Do the same guard for the other subpackages’
      `__all__` while we’re here.
- [ ] Run full `pytest`. Commit.

### Phase 2 — Rename modules for clarity (mechanical, test-guarded)
Rename + update all imports in `src/`, `examples/`, `tests/`, and the `__init__.py` re-exports.
Keep the **public function names** the same so the API surface doesn’t move; only file/module names change.
- [ ] `io/automated_vessel_assignment.py` → `io/vessel_masks.py`.
- [ ] `graph/automated_vessel_assignment.py` → `graph/terminal_node_assignment.py`.
- [ ] `haemodynamics/automated.py` → `haemodynamics/fwhm_diameter.py` (update `haemodynamics/__init__.py`,
      `haemodynamics/pipeline.py`, and `statistics/3D_distances.py` which calls `build_graph_branch_label_volume`).
- [ ] `statistics/3D_distances.py` → `statistics/distances_3d.py` (or `cell_distances.py`); drop the
      `importlib` hack in `statistics/__init__.py` for a normal `from .distances_3d import ...`.
      Rename `tests/test_3d_distances.py` references as needed.
- [ ] Use `git mv` so history is preserved. Run full `pytest` after each rename. Commit per rename.

### Phase 3 — De-duplicate the whole-brain workflow — **DONE**
- [x] The stage runner moved to `src/ImageLynx/pipeline.py`, so examples share it instead of forking it.
- [x] The pressure/boundary-flow solve and the pericyte dilation sweep moved to
      `src/ImageLynx/haemodynamics/pericyte_sweep.py`; the curve plots to
      `src/ImageLynx/visualization/dilation_curves.py`.
- [x] `resistance_network_pipeline_for_Alice.py` (1,795 lines) and root-level `AlicePaper.py` are
      replaced by `examples/brain_network_pipeline.py` (73 lines) plus `brain_pipeline_config.yaml`.
      "Alice" is gone from the names; the sweep is described by what it does.
- [x] `tests/test_alice.py` → `tests/test_pericyte_sweep.py`, driving the extracted module.

### Phase 4 — Thin out the examples / consolidate config (larger, do last)
- [x] Stage orchestration lifted into `src/ImageLynx/pipeline.py`; the example is now config + CLI
      glue (1,282 → ~250 lines) and `brain_network_pipeline.py` runs the same stages.
- [ ] Split `ImageLynx.pipeline.run_pipeline_stages` further into one function per stage
      (segmentation → skeletonize → graph → boundary/branch-order → haemodynamics → export/stats)
      with unit tests per stage. It is one ~800-line function today.
- [ ] Add a short “preset system” note to the README (settings constants → preset dicts → CLI/YAML overrides),
      since the layering isn’t obvious from filenames.
- [ ] Populate `examples/local_presets.py` with one realistic example preset (it’s currently an empty stub).

### When done
- [ ] Full `pytest` (incl. `slow`/`integration`) green; tutorial notebook still exports & runs.
- [ ] Update the **Repository layout** and **Key modules** sections above to the new names, drop the
      “Note”/misnomer caveats, and **delete this entire Cleanup plan section.**
