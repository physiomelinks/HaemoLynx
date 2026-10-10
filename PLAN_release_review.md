# Plan — the release review scorer (TEMPORARY)

> Delete this file, `RELEASE_REVIEW.md`, the "Release review" section of `CLAUDE.md` and
> `scripts/release_review/` at the final release (step R8). Written 2026-10-10 against
> `9c6a15a`.

`RELEASE_REVIEW.md` defines the review: sixteen aspects, each scored 0–10 by a ladder of
checks. Its first scorecard was filled in by hand. This plan turns that into **a command that
produces the scorecard**: it collects the evidence for every check that can be automated, reads
the recorded human marks for the rest, applies the ladder, and writes a dated report with what
moved since the last one.

Why it is needed: the evidence today is scattered. Runtimes sit in memory notes and
`outputs/p6/*.json`, accuracy numbers in review folders, and the one full E14.5 timing is a
single run on one machine. A score that nobody can re-measure drifts, and a score that only goes
up because nobody re-measured it is worse than no score.

---

## Design

- **Shaped like `scripts/branch_comparison/`.** Pure modules (the rubric, the ladder, each
  collector's parsing, the report) are unit-tested in milliseconds; a thin runner does the slow
  work and never runs in CI except the quick tier (R7).
- **One rubric, written down twice, checked to agree.** `scripts/release_review/rubric.py` holds
  every check: `id` (`speed.6a`), aspect, level, kind (`A` automated, `R` reviewed from code or
  docs, `H` human), target, and the collector that decides it. `RELEASE_REVIEW.md` lists the same
  IDs. `tests/test_release_review_rubric.py` parses the document and fails if either side has an
  ID the other lacks or a different level, as `tests/test_gui_layout.py` does for the panel.
- **A check result is data:** `CheckResult(id, status, value, target, evidence, commit, date)`,
  status one of `pass` / `fail` / `unknown`. **`unknown` scores as `fail`**: missing evidence is
  never benefit of the doubt.
- **The ladder is one pure function.** The score is the highest even level whose checks, and
  every lower level's, all pass, plus one if at least half of the next level's checks pass.
  It is tested exhaustively, including that a passing level-8 check cannot lift an aspect with a
  failing level-4 check.
- **Human marks are a tracked file**, `scripts/release_review/human_checks.yaml`: per `H` check,
  who, when, the result and where the evidence is (a SUS sheet, a marked grid's key, a tester's
  notes). The scorer never writes it. An `R` check found by an agent is recorded there too, with
  the file and line that decided it.
- **Tiers, so the cheap part runs often:**

  | Tier | Time | What it runs |
  |---|---|---|
  | `--quick` | < 5 min | static facts: tests collected per marker, CI matrix, packaging metadata, docs counts, code metrics, rubric consistency |
  | `--fixtures` | < 30 min | the pipeline on the committed fixtures and synthetic phantoms; exports read back; the run repeated for identity; the edge-case suite |
  | `--full` | hours | the reference datasets: stage times, peak memory, the lumen-artefact report, planted-vessel diameter error |

- **Output** goes to `outputs/release_review/<date>_<sha>/` (git-ignored): `scorecard.md`,
  `scorecard.json`, the raw evidence, and one appended row per run in
  `outputs/release_review/benchmarks.csv`, so trends per commit can be plotted. The summary table
  is then copied into `RELEASE_REVIEW.md` and a line added to its review log.
- **Reference dataset paths are local**, in an untracked `release_review_datasets.yaml`
  (E14.5 MCA, the Nerve stack, Mason's skeletal-muscle stack, the whole brain), never committed.
- **Shared checkout:** the runner measures a `git archive` of the commit being scored, run with
  `PYTHONPATH` pointing at it, not the live tree, because peer sessions edit `src/` mid-run (see
  the memory note on shared checkouts).

---

## Phases

### R0 — The written review (this change)
- [x] `RELEASE_REVIEW.md`: scoring rules, the sixteen ladders, the first scorecard with evidence
      and gaps, the review log.
- [x] `CLAUDE.md`: a temporary "Release review" section with the rules agents follow.

### R1 — Skeleton and the quick tier
- [ ] `scripts/release_review/` with `rubric.py`, `ladder.py`, `report.py`, `runner.py` and the
      CLI `scripts/release_review.py`.
- [ ] Quick collectors: tests collected per marker (`pytest --collect-only -q -m ...`); the CI
      matrix read from `.github/workflows/*.yml` (operating systems, Python versions, which
      markers run); packaging facts (version, CHANGELOG / CITATION.cff present, wheel builds,
      `npe2 validate`); docs counts (settings with empty help; help pages left empty, **counted
      only, never written**; the tooltip test passes); code metrics (largest modules, TODO count,
      lint configured).
- [ ] Tests: `tests/test_release_review_ladder.py`, `test_release_review_rubric.py` (document vs
      rubric), one test per collector on a fabricated tree.
- **Done when** `python scripts/release_review.py --quick` reproduces the hand-made scorecard's
  quick-tier checks, and each disagreement is resolved by fixing either the collector or the doc.

### R2 — Timing and memory recorded by the pipeline itself
This is product work and **stays after the review is removed**: a user wants to know where a
run's time went.
- [ ] `ProgressEvent` gains `elapsed_s` on a stage or step finishing; the run writes
      `{stem}_run_timings.csv` (stage, step, wall seconds, peak resident memory MB).
- [ ] Peak memory sampled by a light thread (`psutil` as an optional dependency, or
      `ru_maxrss` / Windows `GetProcessMemoryInfo`; decide in the PR). Worker processes report
      their own peaks.
- [ ] The panel's run log prints each stage's time as it finishes.
- [ ] Tests: the CSV exists after a fixture run, has one row per stage, and the times add up to
      within 5% of the run's wall time.
- **Unlocks** `speed.4a`, `speed.6a`, `memory.6a`.

### R3 — The fixture tier
- [ ] Run the pipeline on the three committed fixtures (`Nerve_capillaries_cropped.tif`,
      `seven_vessel_noisy_3d.tif`, `bundled_vessels_8_to_2.h5`).
- [ ] **Edge-case suite** (`robust.6a`): an empty mask, a single straight vessel, a one-voxel-thick
      vessel, a 2D image, strongly anisotropic voxels, vessels touching the image border, no
      inlets or outlets. Each runs every stage and either finishes or stops with a named error,
      never an unhandled exception.
- [ ] **Identity** (`repro.6c`): the same input and settings run twice give byte-identical CSVs
      and identical graphs (bar the export timestamp), with workers on and off.
- [ ] **Exports read back** (`export.4a`, `export.8b`): VTK through `pyvista`/`meshio`, CSVs
      through `csv`, the graph through NetworkX alone, each compared field by field against the
      data dictionary (`export.6a`), so a column the dictionary does not describe fails.
- [ ] **Provenance** (`repro.6a`): every output folder holds the settings, version, commit and
      input checksums.

### R4 — Ground truth for accuracy
The step the release gate depends on most: graph, diameter and flow need 8.
- [ ] **Phantoms** (`graph.8a`, `diameter.8a`): synthetic vessel trees and meshes with known
      centrelines, radii and junctions, rasterised at two voxel sizes (isotropic and 1:1:3) and two
      blur/noise levels, with a raw image to go with each mask. Generated in code from a seed;
      shared with the tests as fixtures.
- [ ] **Metrics** (pure, tested): junction precision and recall (a built junction matches a true
      one within the local radius), centreline length error, loops (first Betti number) against
      the truth, diameter error per method and per width, and flow against the analytic solution
      on the true graph.
- [ ] **Independent solver** (`flow.8a`): export a network to NetFlow's input format, solve it
      there, compare pressures and flows. Needs NetFlow built locally; the comparison script is
      ours.
- [ ] Planted-vessel diameter error on the real image (`fwhm_planted.py`), reported in the full
      tier per dataset.

### R5 — The human review kit
Nothing above 6 in usability, visualisation and docs can be reached without it.
- [ ] `scripts/release_review/usability_tasks.md`: a task list (install, open an image, run,
      find a result, edit a vessel, export, re-run from a stage), with a sheet for times and where
      the tester got stuck.
- [ ] The System Usability Scale: the ten standard items, scored 0–100.
- [ ] A **visual review grid** generator: the panel's views and every saved figure, numbered and
      shuffled, marked GOOD / POOR by a tester, under the **Marked example grids** rules.
- [ ] A **manual diameter protocol** (`diameter.8b`): 30+ vessels measured by hand in napari on
      a Shapes layer, compared with each method.
- [ ] Results recorded in `human_checks.yaml` with names and dates.
- **Testers** (`RELEASE_REVIEW.md` § Testers):
  - **Developers:** Harvey, Finbar and Mason. They do the marked grids, the manual diameters and
    the installs on each OS: Windows (Harvey), macOS (Mason) and Linux (Finbar).
  - **Independent:** George, Rebecca and Mike. They do the task list, the SUS questionnaire and
    the visual grid.
  - The independent three go first, before anyone shows them the panel, because the task list is
    only a first-time test once.

### R6 — The full tier and trends
- [ ] Each reference dataset: stage times and peak memory (from R2), the lumen-artefact report
      (`graph.diagnose_lumen_artefacts` on the saved graph), planted-vessel error, unsolved
      vessels, flow conservation residual.
- [ ] `benchmarks.csv` appended per run; a stage more than 20% slower than the last scored commit
      is flagged in the report (`speed.8b`).
- [ ] Scaling: the E14.5 stack cropped to an eighth, a quarter and a half, and whole
      (`speed.10a`, `memory.10a`).

### R7 — CI (optional)
- [ ] The quick tier as a **non-blocking** PR job that posts the score change.
- [ ] The fixture tier on a weekly schedule. The full tier stays local (the data is not in the
      repository).

### R8 — Release
- [ ] The gate in `RELEASE_REVIEW.md` met and recorded in its log.
- [ ] Tag the release.
- [ ] Delete `RELEASE_REVIEW.md`, this file, the `CLAUDE.md` section and
      `scripts/release_review/`. **Keep** R2's timings and anything from R4 that became test
      fixtures. Decide then whether the benchmark and phantom runner should stay as a developer
      tool under `scripts/`.

---

## Order

R1, then R2, then R4 and R3, then R6. R5 runs alongside them: the testers are named, and it
waits on their time rather than on code. Its task list and SUS sheet are the first thing to
write, so George, Rebecca and Mike can start. The gate is set by the lowest aspect, so the first useful work is
whatever lifts the lowest one. Speed, memory and usability lose their level-4 and level-6 checks
mainly because nothing is measured, so R2 and R5 are worth more than polishing anything already
at 6.

## Decisions for the user

1. **Targets.** Every number marked *(proposed)* in `RELEASE_REVIEW.md` is a first guess. Confirm
   or change each one at the first review and log it. Never change a target later to make a
   score pass.
2. ~~**Testers.**~~ Decided 2026-10-10: Harvey, Finbar, Mason, George, Rebecca and Mike, with
   George, Rebecca and Mike as the independent ones. All three are first-time users and domain
   experts. The operating systems are covered by Harvey (Windows), Mason (macOS) and Finbar
   (Linux).
3. **Reference datasets.** Are E14.5 MCA, Nerve, Mason SM 9W and the whole brain the right four?
4. **New development dependencies:** `psutil` (peak memory) and `pytest-cov` (coverage) in the
   `dev` extra.
5. **NetFlow.** Is building it locally for `flow.8a` acceptable, or is another published network
   with reported flows a better reference?
