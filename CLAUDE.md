# ImageLynx — repository guide for AI assistants

ImageLynx turns 3D microvascular microscopy into **NetworkX graphs** with haemodynamic edge weights, VTK exports, and network statistics. The pipeline accepts **already-segmented** binary masks (TIFF/H5) or can **run segmentation in-repo via ilastik** (headless) when configured.

**Segmentation is in scope.** ImageLynx integrates ilastik for pixel classification inference on the main vessel volume and on optional large/small arteriole–venule channels. What stays **outside** this repo is **training** ilastik projects (`.ilp` classifiers): users train those manually in the [ilastik](https://www.ilastik.org/) GUI, then point the pipeline at the exported project and raw/unsegmented images.

---

## Repository layout

```
ImageLynx/
├── src/ImageLynx/          # Installable package (setuptools, pythonpath = src in pytest)
│   ├── io/                 # Load TIFF/H5, skeletonize, ilastik, vessel-mask loading
│   ├── preprocessing/      # Skeleton cleaning, bridging, bundle refinement
│   ├── graph/              # Skeleton → graph, topology repair, branch orders, boundaries
│   ├── haemodynamics/      # Poiseuille weights, resistance, flow solve, FWHM diameters, pericytes
│   ├── statistics/         # Network metrics, 3D cell-to-vessel distances
│   └── visualization/      # Matplotlib/plotly plots, VTK export
├── examples/               # Runnable pipelines and settings (not the core library API surface)
│   ├── resistance_network_pipeline.py   # Main end-to-end example / CLI
│   ├── resistance_pipeline_settings.py  # Default parameters and presets
│   ├── preflight.py, wizard.py, presets.py
│   └── carotid_image_to_model.py
├── tutorials/
│   ├── pipeline_tutorial.ipynb   # **Source of truth** for the step-by-step tutorial
│   ├── pipeline_tutorial.py       # Auto-generated from the notebook (do not edit by hand)
│   ├── tutorial_plots.py          # Inline plot helpers for the notebook
│   └── export_notebook.py         # nbconvert export used by integration tests
├── tests/                  # Unit + integration tests (see Testing below)
│   ├── conftest.py         # Shared fixtures; matplotlib Agg backend
│   └── integration/        # Full-pipeline and tutorial tests (@pytest.mark.integration)
└── tests/data/             # Small TIFF/H5 fixtures for tests and tutorial input
```

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

- **Python:** ≥ 3.9 (CI uses 3.10)
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

GitHub Actions (`.github/workflows/pytest-pr.yml`) runs `pip install -e .[dev]` and `pytest` on pull requests.

---

## Coding conventions

- **Library code** lives under `src/ImageLynx/`. Keep `examples/` thin — orchestration, CLI, presets.
- **Minimal diffs** — match existing naming, types, and import style in the touched module.
- **Input contract** — pipeline expects **binary vessel masks** at skeletonization time. Masks may come from pre-existing files or from **ilastik inference** in this repo; classifier training is always manual. Document new ilastik-related paths/flags in `resistance_pipeline_settings.py` and `preflight.py`.
- **Voxel sizes** — use `io.voxel_validation.resolve_voxel_size_xyz`; masks and main image must align in shape and physical voxel units.
- **Graph** — `nx.MultiGraph` with `pos` on nodes and `voxels` on edges; haemodynamics uses `branch_order` on edges.
- **Skeletonization** — use `skimage.morphology.skeletonize(..., method="lee")` via `preprocessing.skeletonize_volume`, not deprecated `skeletonize_3d`.
- **Comments** — only for non-obvious domain logic; prefer self-explanatory code.

---

## Key modules (quick reference)

- **`io/ilastik.py`** — `run_ilastik_headless_segmentation` (subprocess call to user-installed ilastik + `.ilp` project).
- **`io/automated_vessel_assignment.py`** — `load_large_vessel_masks`, `load_and_validate_vessel_masks` (includes ilastik path for large/small masks).
- **`graph/assemble.py`** — `build_graph_from_skeleton`; optional `step_callback(G, label)` after each topology step.
- **`haemodynamics/pipeline.py`** — high-level Poiseuille application used by examples and tutorial.
- **`examples/resistance_pipeline_settings.py`** — constants, ilastik toggles, and presets for the main pipeline script.

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
