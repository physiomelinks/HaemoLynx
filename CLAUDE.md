# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ImageLynx turns 3D microscopy of microvasculature into haemodynamic network models. The package is `src/ImageLynx/`; runnable pipelines and study drivers are in `examples/`. Most current work is the **carotid body (CB) study**: six specimens (WKY-A/B/C normotensive, SHR-A/B/C hypertensive), testing H1 (morphometry) and H2 (perfusion, haematocrit, hypoxia). An older nerve-capillary resistance pipeline shares the package.

## Commands

Use the repo venv (`venv/bin/python`, Python 3.12). CI (`.github/workflows/pytest-pr.yml`) runs `pytest` on pull requests under Python 3.10, after `pip install -e .[dev]`, with `MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true`.

```bash
venv/bin/python -m pytest                                   # full suite
venv/bin/python -m pytest tests/test_rheology_laws.py       # one file
venv/bin/python -m pytest tests/test_cb_settings.py::test_frozen_threshold_lies_on_the_sweep_grid   # one test
venv/bin/python -m pytest -m "not slow and not integration" # skip heavy tests
```

Markers: `slow`, `integration`, `plotting`. Whole-pipeline tests are in `tests/integration/`; small fixture volumes in `tests/data/`. `pyproject.toml` sets `pythonpath = ["src"]`, and `tests/conftest.py` puts `examples/` on `sys.path`, so tests import driver scripts directly. There is no formatter. A pre-commit hook (`scripts/git-hooks/pre-commit`, turned on with `git config core.hooksPath scripts/git-hooks`) runs pyflakes on staged files, failing only on undefined names and redefinitions, then the `not slow and not integration` tests, which it skips (printing one line) when every staged file is a `.md` doc, or when `.git/hook-stamp/fast-tests-ok` matches the staged tree, interpreter and installed packages with no unstaged edits to tracked files (a passing hook run or `scripts/stamp-tests.sh`, which runs the full suite, writes it); so `git commit` usually runs tests, and `--no-verify` skips the hook. The `[dev]` extra is `pytest` and `pyflakes`. Install from `requirements.txt`: `pyproject.toml` omits several runtime deps (`pyyaml`, `joblib`, `optuna`, `dask`, `numba`, `python-igraph`, …), and CI installs only `pyproject.toml`'s, so tests needing them can skip or fail there.

CB study workflow, in order (`python`/`python3` below means the venv interpreter):

```bash
python3 examples/preprocessing/preprocess_cb.py --input <dir> --diagnose            # raw TIFF -> report only
python3 examples/preprocessing/preprocess_cb.py --input <dir> --output-dir ilastik_inputs
python3 examples/preprocessing/preprocess_th.py --input <dir>                        # TH (glomus) channel
# Ilastik prediction with the pooled classifier (external, headless) -> ilastik_probabilities/
python examples/carotid_image_to_model.py --record-provenance                        # stamp maps with the classifier; run right after predicting
python examples/cb_h1_batch.py --stage placement
python examples/cb_h1_batch.py --stage threshold
python examples/cb_h1_batch.py --stage run --threshold 0.95                          # ~6 min/specimen
python examples/cb_h1_batch.py --stage sensitivity                                   # frozen threshold's grid neighbours, seed held; 12 runs (~80 min) -> outputs/cb_h1_sensitivity/
python examples/carotid_image_to_model.py --specimen WKY-A --roi-voxels 160 160 160  # one specimen
python examples/carotid_image_to_model.py --list-specimens                           # also shows each specimen's readiness
python3 examples/cb_h1_th_metrics.py          # H1 §1.3, 1.5 (WKY only; --th-threshold sets the TH cuts)
python3 examples/cb_h1_th_metrics.py --all    # adds SHR; writes cb_h1_th_metrics_all.json (--out overrides)
python3 examples/cb_h1_figures.py
python3 examples/cb_h1_vtk.py                 # VTK of the H1 network (--verify checks frames only)
python3 examples/cb_h1_renders.py             # renders cb_h1_vtk.py's .vtp files
python3 examples/cb_h2_glomus_perfusion.py    # H2 §2.1, 2.2, 2.4 (--penetration)
python3 examples/cb_h2_hypoxic_fraction.py    # H2 §2.3; --pad-grid, --grid-um, --vessel-mapping, --contrast for the variant/sweep JSONs; ~40 min per run
python3 examples/cb_h2_vtk.py                 # VTK of the H2 fields (--verify checks frames only; also --specimen, --pad-grid, --vessel-mapping, --decimate); ~14 min
```

`carotid_image_to_model.py` thresholds the Ilastik probability map itself, with a hysteresis band that defaults to the frozen `cb_settings` values. Every cut is inclusive (`p ≥ t`, `preprocessing.at_or_above`). Given `--roi-voxels`, it places the ROI with `place_roi` and writes `roi_placement.json`; `--roi-centred` opts out. `--boundary-mode` (`caged` / `universal_sink` / `robin_resistance`) sets tissue-edge permeability; it does not choose the inlet/outlet rule. For its flags use `--help`, not `pipeline_cli_arguments.md`.

`examples/preprocessing/prob_to_mask.py` is a standalone tool, not a pipeline step. `cb_h2_absolute_perfusion.py` measures throughput and flow-weighted velocity under the face and band rules (reference §13.5, S30). The other `cb_h2_*.py` scripts (`boundary_selection`, `error_propagation`, `threshold_calibre`) are one-off measurements behind the H2 capability assessment.

Nerve/resistance pipeline: `python examples/resistance_network_pipeline.py --list-presets | --preset NAME | --config file.yaml | --wizard | --preflight-only | --set KEY=VALUE`; `--save-config out.yaml` writes the resolved configuration. Every `image_to_model_pipeline(...)` kwarg is also exposed as `--kwarg-name`. That function lives in the example script itself, not the package; a second copy is in `examples/resistance_network_pipeline_for_Alice.py` (tested by `tests/test_alice.py`), so a change to one may need the other. Its defaults are module-level constants in `examples/resistance_pipeline_settings.py`; named presets are in `examples/presets.py`, with user overrides in `examples/local_presets.py` (`LOCAL_PRESET_DEFINITIONS`).

## Architecture

### Package (`src/ImageLynx/`)

- `preprocessing/` — `image.py` (filters, `at_or_above` inclusive cut, `hysteresis_threshold`, `crop_roi`), `mask.py` (`build_vessel_mask`: probability → binary mask via hysteresis), `skeleton.py`. `pipeline/map_reduce.py` tiles a volume into overlapping chunks, masks them in parallel (joblib) and stitches the result.
- `graph/` — skeleton → `networkx` MultiGraph (`build.py`), then conditioning modules (`prune.py`, `collapse.py`, `reconnect.py`, `degree2.py`, `large_vessels.py`, `optimise.py`). `boundaries.py` picks inlet/outlet terminals (face and band rules, see CB data flow); `branch_order.py` assigns hierarchy; `tiling.py` makes map-reduce chunk boxes.
- `haemodynamics/` — `poiseuille.py` (diameters → resistance, diameter-provenance guards), `resistance.py` (sparse Laplacian solve), `rheology.py` (coupled flow–haematocrit–viscosity: Pries phase separation, one-vs-rest at 3+ daughters, Fåhræus–Lindqvist; `rheology_status(G)` reports how it stopped), `perfusion.py` (3D tissue advection–diffusion–reaction grid coupled to the 1D network, three tiers — see below), `transit.py`, `tissue_regions.py`. Use British spelling: `ImageLynx.hemodynamics` is a deprecated alias. `pericyte_mask.py`, `pericyte_comparison.py` and `probability.py` are frozen for CB work but live for the nerve pipeline.
- `io/` (image and Ilastik loading), `visualization/` (plots, reporting, VTK export).
- `statistics/` — morphometry, threshold selection, cohort-split checks, Optuna auto-tuner. `th_morphometry.py` joins the vessel and TH channels (H1 §1.3, 1.5); `label_placement.py` audits where Ilastik training labels sit.
- CB-study modules at the top level:
  - `specimens.py` — the specimen registry. Holds each specimen's file stems (WKY and SHR are named differently), resolves data roots (`IMAGELYNX_CB_DATA_ROOT`, `IMAGELYNX_CB_ACQUISITION_ROOT`), and holds the single `POOLED_CLASSIFIER` and `PROCESSING_VOXEL_UM`. Each `Specimen` exposes every stage's path as a property (`probabilities_path`, `th_probabilities_path`, …); use these rather than building paths. Preprocessing QC sidecars are committed under `src/ImageLynx/data/preprocessing_qc/`.
  - `cb_settings.py` — frozen analysis constants (ROI size, threshold, hysteresis band, boundary axis, pressures, rheology solver settings, TH threshold, perfusion grid, M_max…). All published H1/H2 numbers use these.
  - `roi_placement.py` — places each specimen's matched ROI on tissue, not the array centre: z from the QC peak slice, y/x from the grayscale centroid over the box's slices.
  - `artefact_provenance.py` — `.provenance.json` sidecars recording which classifier made a probability map.
  - `batch_outputs.py` — `open_batch_run`, the single reader of a batch run (edge table, graph with diameters joined on, skeleton, mask, TH channel), checked against `place_roi` once at open.
  - `batch_compare.py` — compares two batch runs part by part through the reader; behind `examples/cb_compare_batch_runs.py` (exit 0 match, 1 differ, 2 usage error, 3 could not compare).

### CB data flow

`preprocess_cb.py` → Ilastik (one pooled project for all six) → `ilastik_probabilities/*.h5` → `carotid_image_to_model.py`. That script is a phased orchestrator driven by dataclass configs (`PreprocessingConfig`, `SkeletonConfig`, `GraphConfig`, `HaemodynamicsConfig`, `PerfusionConfig`, `VisualizationConfig`, `PipelineConfig`), overridable by YAML `--config` and CLI flags. YAML top-level keys are these class names; a key under the wrong section is dropped with only a warning. `update_dataclass_from_dict` re-runs `__post_init__`, so a YAML cannot set a value the constructor forbids. The two example YAMLs (`examples/config_WKY_normotensive.yaml`, `config_SHR_hypertensive.yaml`) carry only the frozen shared settings and must load to the same dict (tested). `cb_h1_batch.py` passes no `--config`.

Phases: mask (optionally map-reduce chunked) → skeleton → graph → boundary conditions and diameters → flow/rheology solve → perfusion (on by default). Edge resistance comes from the rheology solve (Pries–Secomb viscosity), and everything that reports resistance runs after it. Artefacts (`vessel_mask.npy`, `skeleton.npy`, `network_graph.pkl`) are cached in `<vtk_output_prefix parent>/<image stem>_cache/` (`--use-cache-dir`). The `C1-CB3-…_cache` / `CB3-SHR-…_cache` dirs at the top of `examples/outputs/` come from single-specimen runs, not the batch.

`cb_h1_batch.py --stage run` runs that script once per specimen as a subprocess, into `examples/outputs/cb_h1_batch/<specimen>/` (`specimens.BATCH_RUN_ROOT`, per specimen `Specimen.batch_run_dir`); `--stage sensitivity` writes `examples/outputs/cb_h1_sensitivity/t<low>/<specimen>/`. The H1 and H2 drivers read those runs back through one reader, `batch_outputs.open_batch_run(specimen)` (a sensitivity run is opened by passing `run_dir`). Vocabulary (batch run, edge table, placed ROI) is in `GLOSSARY.md`; the plan and tickets are in `.scratch/batch-run-reader/`.

`src/ImageLynx/batch_outputs.py` documents what the reader checks and hands out. The points easy to get wrong:

- **Strict join.** `graph()` joins the edge table on forward integer `(u, v, key)`, one-to-one. There is no reverse-direction fallback, and nothing is filled with NaN or 0: a missing, extra or duplicate row raises.
- **Numbers through the reader.** Every edge-table number goes through `BatchRun.numeric_column`, which raises on a blank, text, NaN or infinite cell.
- **Drivers read nothing themselves.** A driver does not glob a `*_cache` folder, open the CSV, pickle or TH `.h5`, call `place_roi`/`check_output_roi` (use `run.placement`), or import another driver's underscore-prefixed names (a public loader such as `load_network` is fine).

A new or edited driver that reads a batch run uses the reader and is added to `ON_THE_READER` in `tests/test_drivers_read_through_batch_runs.py`, which checks the source for those leftovers. That file's `ON_THE_READER`, `SHARED_LOADERS` and `PACKAGE_LOADERS` list which drivers read batch runs and how.

Read the unsuffixed outputs in `examples/outputs/`. Suffixed copies (`*_pre_itemNN`, `*_pre_rerun`, dated dirs) are archives of earlier runs, with logs in `rerun_*_logs/`. `examples/outputs_CB_results_0.5/` (an old 0.5-threshold run) and `examples/OLD/` (superseded notebooks and scripts) are archives too, as are `examples/image_to_model.py` (an older driver nothing imports) and `examples/preprocessing/make_ilastik_input.py` (an earlier Ilastik-input builder, replaced by `preprocess_cb.py`). Outputs go stale when a fix lands before the scripts that read it are re-run. Each closed entry in the Open items table at the end of `cb_modelling_reference.md` says in its own words whether it was re-run ("re-run 2026-09-28", "Not re-run: …"); pending re-runs are tracked in the `pipeline_rerun_*_notes.md` files. Check both before quoting a number. To compare a re-run with an archive, use `examples/cb_compare_batch_runs.py OLD_ROOT NEW_ROOT`, which compares all six parts by content (exit 0 when every run matches). Don't compare `network_graph.pkl` by bytes: they change between runs even when the graph does not.

`cb_h1_figures.py` reads every number from the batch and sensitivity edge tables (through `open_batch_run`) and `threshold_selection.json` (written by `--stage threshold`), with p-values from `assess_cohort_split`. A missing table raises.

The pipeline and the H2 drivers share their pressure boundaries but differ in tier and grid. Check these before comparing their outputs:

- **Pressure boundaries — shared.** Both use the face rule (`select_boundary_terminal_nodes_by_face`) on `cb_settings.BOUNDARY_AXIS` (axis 1) with 1-voxel tolerance; it raises on an empty face. The band rule (`select_boundary_terminal_nodes`) remains for the nerve pipeline (with `allow_extreme_fallback=True`) and as the comparison rule in `cb_h2_absolute_perfusion.py`. `cb_h2_error_propagation.py` uses the same face rule but solves plain Poiseuille flow, not the Pries flow, so its error floors are not the H2 flow field.
- **Tissue transport tiers** (`cb_modelling_reference.md` §6.6). Tier 1 is `solve_perfusion_steady_state` (O₂ only). Tier 2 is `solve_coupled_1d3d_perfusion` (endothelial barrier) and is unreachable at the defaults, because the dispatch checks `use_multi_species_model` before `use_endothelial_barrier_model` and both default to `True`. Tier 3 is `solve_multi_species_perfusion` (O₂, CO₂, pH). The pipeline runs Tier 3 and writes a tier-tagged `*_perfusion.vti`, which is not the H2 field: every H2 hypoxia number comes from Tier 1, called directly by `cb_h2_hypoxic_fraction.py` and `cb_h2_vtk.py` with `cb_settings.PerfusionSettings`.
- **Perfusion grid and vessel mapping** (§6.2, §6.8). The pipeline uses a 10 µm grid (`PerfusionConfig.grid_resolution_xyz`) and `map_vessels_to_grid`'s default `vessel_mapping="centreline"`. The H2 drivers use `cb_settings.GRID_UM` (3 µm) and `vessel_mapping="cross_section"`. Only the cross-section mapping converges with grid refinement.

The TH (glomus cell) channel is a parallel track: `preprocess_th.py` (reuses `preprocess_cb.py`'s IO) → its own Ilastik prediction → `Specimen.th_probabilities_path`. It never enters the network pipeline. `cb_h1_th_metrics.py` and the H2 scripts get it through `BatchRun.th_mask`, which crops it to the placed ROI and cuts it at `cb_settings.TH_THRESHOLD`. That cut is strict (`p > t`), unlike the inclusive vessel cuts. Both channels share one grid, so no registration is needed.

## Rules for this repo

Review rules that tests can't fully check (unparseable values raise, a path or constant is defined once, review findings are fixed or declined in writing) are in `CODING_STANDARDS.md`.

- **Don't silently change `cb_settings.py` values.** Changing one invalidates published results in `cb_modelling_reference.md` (§7, §13). Import the constants from `ImageLynx.cb_settings`; don't copy them. `tests/test_cb_settings.py` fails if an `examples/cb_*.py` driver redefines an owned constant as a literal, and checks that the pipeline config and example YAMLs agree with the settings.
- **No silent fallbacks.** Raise rather than substitute a made-up value (e.g. a default 5 µm diameter). See `check_diameter_provenance` in `haemodynamics/poiseuille.py` and `tests/test_silent_fallback_guards.py`.
- **Avoid group-correlated choices.** One classifier, one threshold, one voxel size and one ROI rule for all six specimens. Per-specimen tuning would confound WKY-vs-SHR differences.
- **Commit messages** use a conventional prefix (`fix: …`, `feat: …`, `docs: …`, `refactor: …`), with a GitHub issue number when there is one (`fix (#98): …`). A local `.scratch` ticket goes in the body, not the subject, as a line `Ticket: <feature>/<NN>` (e.g. `Ticket: batch-run-reader/03`). Write the subject as a plain statement of the change.

## Key references

The repo root also holds many plan, handover and scratch files (`*_plan.md`, `chat_transfer_*`, `implement_phase*.py`, `AlicePaper.py`, …). They are not part of the package or the tests. `README.md`, `examples/README.md` (its cache and output paths are wrong), `REMINDERS.md`, `examples/REMINDERS.md` and `TODO.md` are out of date too. Treat only the documents below as current.

- `cb_modelling_reference.md` — the main modelling reference: every method, equation, frozen value and open item, by section. Code comments cite its "open item N" entries (the list at the end) and §11 assumption rows. Check it before changing modelling code, and keep it in sync.
- `haemodynamic_solve_execution_order.md`, `roi_placement_execution_order.md`, `tissue_transport_execution_order.md` — control flow of each stage (Mermaid sources and renders in `figures/`).
- `post_radius_assignment_equations.md` and `post_radius_assignment_equation_flow.md` — the equations after radius assignment, numbered to match the execution-order docs.
- `hypothesis_testing_methods.md` (gitignored) — defines H1 and H2 and the numbered methods (§1.1–1.5, §2.1–2.4) that drivers, docstrings and the whitepapers cite, each with its current status.
- `h1_pipeline_capability_assessment.md` (gitignored) — the 2026-07-30 stage-by-stage review of the pipeline against H1, kept as a dated snapshot, with status updates at the top.
- `h2_pipeline_capability_assessment.md` — defines the numbered findings (S10, S14, S15, …) that H2 docstrings and comments cite.
- `H1_preliminary_results_whitepaper.md`, `H2_preliminary_results_whitepaper.md` — current write-ups of results.
- `pipeline_rerun_*_notes.md` (untracked) — the lettered re-run packages that commit messages cite. The newest (`2026-10-01`, the full re-run on `c190de2`) holds the current package table (O onward), where newly found packages are added; its open rows list outputs on disk that lag the code. Older ones define earlier letters (A … N).
- `open_items_followups.md` (untracked) — working notes on the Open items table in `cb_modelling_reference.md`: which items were closed by which commit, and plans for the rest.
- `CB-SEGMENTATION-METHODS.md` and `examples/preprocessing/README.md` — how the Ilastik project was built and driven, and the preprocessing steps for both channels.

### Obsidian vault (outside the repo)

`/home/dsas627/Desktop/obsidian_vaults/me_bioeng_cb_pipeline` is the companion knowledge base for this pipeline. Code, configs and data stay in the repo. The vault holds everything else. Read it for context on design rationale, run history or the literature behind a choice. **Do not edit it unless the user asks.** It has its own `CLAUDE.md` with writing rules (no em dashes, NZ spelling, thesis-methods prose in `pipeline-notes/`) that apply to any edit made there. Start from its `index.md`. Method rationale is in `pipeline-notes/` (one folder per phase), run history in `experiment-logs/`, and paper notes in `literature_review/`.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<feature>/` (gitignored); never GitHub Issues, since `origin` is not ours. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each ticket file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at the repo root defines **batch run**, **edge table** and **placed ROI**. ADRs will go in `docs/adr/` (none yet). See `docs/agents/domain.md`.
