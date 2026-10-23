# Cleanup plan: the full checklist as it stood

Moved out of CLAUDE.md on 2026-10-22; CLAUDE.md keeps only the items still open.


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
- [x] `io/automated_vessel_assignment.py` → `io/vessel_masks.py`. (2026-10-23)
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
