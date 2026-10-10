# HaemoLynx release review (TEMPORARY)

> **Delete at the final release**, together with `PLAN_release_review.md`, the "Release review"
> section of `CLAUDE.md` and `scripts/release_review/` once it exists. Until then this file is how
> we judge how far the software is from release, and what to work on next.

Every aspect of the software is scored **out of 10** by a ladder of checks (see
[How scoring works](#how-scoring-works)). A score is never set by impression. It is the highest
level whose checks all pass, and a check passes only with evidence written beside it.
`PLAN_release_review.md` is the plan for a command that re-measures the automated checks. Until
that command exists, the checks are marked by hand.

---

## Scorecard

First scoring: **2026-10-10 at `9c6a15a`**, by Claude from the code, the tests and the recorded
outputs. No human checks have been marked yet. Every target marked *(proposed)* still needs
confirming.

| # | Aspect | Score | What holds it there (the next checks to pass) |
|---|---|---|---|
| 1 | [Graph accuracy](#1-graph-accuracy--graph) | **4** | fitted rules' results not saved (6b); coverage measures disagree (6c); morphology erodes the image faces (6d) |
| 2 | [Diameter accuracy](#2-diameter-accuracy--diameter) | **4** | no recorded benchmark table (6a); planted-vessel error not reported for real runs (6b) |
| 3 | [Haemodynamic accuracy](#3-haemodynamic-accuracy--flow) | **3** | the default "in vitro" Pries law is not the law it cites (4b) |
| 4 | [Statistics](#4-statistics--stats) | **5** | no glossary of definitions and units (6a); 21 of 95 rows carry a unit (6b) |
| 5 | [Inputs and segmentation](#5-inputs-and-segmentation--input) | **5** | ilastik output: vessel label guessed, reuse judged by file time (6c) |
| 6 | [Speed](#6-speed--speed) | **3** | a run does not record its stage times (4a) |
| 7 | [Memory and scale](#7-memory-and-scale--memory) | **4** | peak memory never measured (6a); no stated requirements (6b); the default solvers do not scale (6c) |
| 8 | [Usability](#8-usability--usability) | **4** | no task list walked through (6a); defaults untested end to end (6b); no undo (6c) |
| 9 | [Visualisation](#9-visualisation--visual) | **4** | figures without units (6a); tube frame rate not measured (6c); mixed dpi (6d) |
| 10 | [Exports](#10-exports--export) | **3** | VTK writes unsolved vessels' flows as real numbers (4b) |
| 11 | [Documentation and help](#11-documentation-and-help--docs) | **4** | help pages empty, 0 of 10, for Harvey to write (6a); guide skips Post processing and Perturbations (6b) |
| 12 | [Robustness](#12-robustness--robust) | **4** | no edge-case suite through every stage (6a); open review bugs (6b) |
| 13 | [Reproducibility](#13-reproducibility--repro) | **4** | no settings, version or commit beside the outputs (6a); no end-to-end identity test (6c) |
| 14 | [Testing and CI](#14-testing-and-ci--tests) | **3** | no full-suite pass recorded since 2026-09-25 (4c) |
| 15 | [Installation and packaging](#15-installation-and-packaging--package) | **4** | still 0.1.0 after 457 commits, no CHANGELOG (6a) |
| 16 | [Code maintainability](#16-code-maintainability--code) | **3** | Cleanup plan Phase 2 unfinished (4c) |

**Headline: the lowest score is 3** (haemodynamics, speed, exports, testing, maintainability).
The mean is 3.8. The release gate needs every aspect at 7 or more, and the science aspects at 8.

Most of these scores are low because **nothing has been measured, not because the work is
poor**. The test suite is large (6,064 tests) and checks the analytic cases well. What is
missing almost everywhere is a recorded number on real data, a written target, and someone other
than the developer checking the result. Levels 6 and 8 ask for exactly those three things.

### Work in order

This order follows the gate, which is set by the lowest aspects, and favours the cheap fixes:

1. **Decide the Pries "in vitro" law** (`flow.4b`): a decision for Harvey and Finbar, with the
   options in `PLAN_dale_review_fixes.md`, item 1. It takes haemodynamics from 3 to 4.
2. **Record stage times and peak memory in every run** (`speed.4a`, `memory.6a`, step R2 of the
   plan). It takes speed from 3 to 4 and makes the level-6 speed and memory checks possible.
3. **Fix the VTK export** (`export.4b`, `export.6d`, `export.6c`): blank flow for unsolved
   vessels, add diameter and the solved flag, and stamp the version. This is item 5 of
   `PLAN_dale_review_fixes.md`. It takes exports from 3 to 4 or 5.
4. **Record a full suite run**, GUI and slow tests included, at the current commit (`tests.4c`),
   and start measuring coverage (`tests.6a`). It takes testing from 3 to 4.
5. **Finish Cleanup plan Phase 2**: the three renames (`code.4c`). It takes maintainability from
   3 to 4.
6. **Explain the coverage disagreement** (`graph.6c`) and **pad morphology at the image faces**
   (`graph.6d`, `robust.6b`).
7. **Help pages** (`docs.6a`): Harvey writes these. Agents must not; see `CLAUDE.md` § Help menu.
8. **A usability task list** (`usability.6a`). Then George, Rebecca and Mike, all first-time
   users, walk through it (`usability.8a`, `usability.8b`) and mark the visual grid
   (`visual.8a`). Nobody should show them the panel beforehand. See [Testers](#testers) for
   staggering them.

---

## How scoring works

### Levels

Each aspect has checks at five levels. A level's meaning is the same in every aspect:

| Level | Meaning |
|---|---|
| **2 Exists** | It works, at least once, on real data. |
| **4 Tested** | Automated tests pin the core behaviour, and nothing known is wrong with what they pin. |
| **6 Measured** | It is quantified on reference data at a named commit, and the targets are written down. |
| **8 Meets targets** | It meets the targets on two or more datasets, and someone other than the developer (an [independent tester](#testers)) has checked it. |
| **10 Release-ready** | It is independently validated, guarded against regression automatically, and has no open gaps. |

### The ladder rule

- **Score = the highest even level whose checks, and every lower level's, all pass. Add one if
  at least half the checks at the next level pass.** So 4 means "tested", and 5 means "tested,
  and half-way to measured". The scale ends at 10.
- An aspect whose level-2 checks fail scores 0, or 1 if at least half of them pass.
- A failing low check caps the score, whatever passes above it. A level-8 result cannot carry an
  aspect with a level-4 check failing.

### Evidence

- A check passes **only with evidence in its row**. That can be a test (`file:line`), a
  measurement (an output file, with its commit and dataset), or a person's marks (their name, the
  date, and where the marks are kept). "It should work" is a fail.
- **Unknown counts as fail.** When evidence is missing, the row says what would decide it.
- Check kinds:
  - **A**, automated: a script or test can decide it.
  - **R**, reviewed: someone decides it from reading the code or docs, and records the file and
    line.
  - **H**, human: only one of the [testers](#testers) can pass it, and the row names which one.
    An agent may record that a tester did it and quote where. It never passes one on its own
    judgement.
- Targets marked *(proposed)* are first guesses. Confirm or change each at the first review and
  record it in the log. After that, **a target never moves to make a score pass**. This follows
  the marked-grid rule: change the measure, not the marks. Any rubric change goes in the log with
  its reason.

### Reference datasets

| Name | What | Where |
|---|---|---|
| Fixtures | `Nerve_capillaries_cropped.tif`, `seven_vessel_noisy_3d.tif`, `bundled_vessels_8_to_2.h5` | `tests/data/` |
| E14.5 | MCA stack, 287 × 512 × 512 at 0.98 × 0.98 × 1.0 µm, about 4% vessel | local |
| Nerve | `Nerve_capillaries.tif` (328 MB), the `compare_branches.py` default | local |
| Mason SM | skeletal muscle, 124 × 1024 × 1024 at 1.0 µm | local |
| Brain | the whole-brain run (`examples/brain_network_pipeline.py`) | local |

### Testers

Harvey named these six on 2026-10-10. Only they can pass an H check.

| Tester | Has written HaemoLynx code | Independent ("someone other than the developer") |
|---|---|---|
| Harvey | yes | no |
| Finbar | yes | no |
| Mason | yes (the Post processing tab) | no |
| George | no | **yes** |
| Rebecca | no | **yes** |
| Mike | no | **yes** |

"Has written code" comes from `git shortlog`. A check that asks for someone other than the
developer needs George, Rebecca or Mike: every level-8 "checked by", `graph.10a`, `diameter.10a`,
`stats.10a`, `visual.8a` and `usability.8a`. The developers can do the rest: the marked grids
(`graph.6b`, `graph.8b`), the manual diameters (`diameter.8b`) and the installs on each OS
(`package.6b`).

- **Operating systems** (`package.6b`): Harvey on Windows, Mason on macOS (Mason's commits come
  from a Mac) and Finbar on Linux.
- **First-time users: George, Rebecca and Mike**, all three. None of them has used HaemoLynx, so
  any of them can do the first-time-user checks (`usability.8a`, `docs.8a`, `package.8b`).
- **Domain experts: George, Rebecca and Mike**, all three, for the glossary review (`stats.10a`).
- **Recommended: stagger the first looks.** Each tester can be a first-time user only once,
  and there are three. Spend one now, on the panel as it is, to find the worst problems. Keep the
  other two until the task list, the help pages and the guide gaps (`docs.6a`, `docs.6b`) are
  done. `usability.8a` needs two, so the later pair can pass it.

### The release gate

- Every aspect scores 7 or more. That means it is measured, its targets are written, and it is
  half-way to meeting them with outside checking.
- Graph, diameter, haemodynamics and statistics score 8 or more. The science must meet its
  targets on two datasets, checked by someone other than the developer.
- Nothing is left open in `PLAN_*` files that changes a number a user would report.

When the gate is met, record it in the log, tag the release, and delete this review (`PLAN_release_review.md`, step R8).

### When to re-score

- **With the work.** A commit that changes a check's evidence updates that row, the scorecard
  row and the log, in the same commit. A regression lowers the score at once.
- **A full review** every two weeks, and before any version bump. Every row is re-checked, and
  every number re-measured where a script exists.

---

## The ladders

`By` is the check kind (A, R or H). `Now` is ✓ or ✗ at the last review. A row's ID gives its
aspect and level: `graph.6a` is a level-6 graph check.

### 1. Graph accuracy — `graph`

*Does the network trace every segmented vessel once, along its centre?* This covers
skeletonisation, graph building, smoothing and lengths.

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| graph.2a | A | A graph builds end to end from a real segmented stack. | ✓ | E14.5: 3,146 edges, 51,455 µm (`outputs/graph_artefacts_phase6_baseline/REPORT.md`, 2026-10-09) |
| graph.4a | R | Every topology step in `STEP_LABELS` and smoothing has unit tests. | ✓ | `test_graph_assemble.py`, `test_consolidate_lumen.py`, `test_mask_recovery.py`, `test_facing_ends.py`, `test_lumen_loops.py`, `test_centreline_smoothing.py`, … |
| graph.4b | A | Each known artefact kind has a synthetic fixture that pins it. The kinds are: two vessels in one lumen, a loop inside one lumen, dead ends by kind, and uncovered mask. | ✓ | `tests/lumen_artefact_fixtures.py`, `test_lumen_artefacts.py`, `tests/dense_capillary_fixtures.py`, the Nerve invariants in `test_pipeline_invariants.py:467-513` |
| graph.4c | A | Lengths are within 5% of truth on curves of known length. | ✓ | Helices: total under 2%, worst vessel under 5% (`test_centreline_smoothing.py:184-189`). Anisotropic voxels: `test_anisotropic_voxel_size.py` |
| graph.6a | A | The lumen-artefact report on a reference dataset's saved graph is recorded at a named commit. | ✓ | E14.5 `REPORT.md`: 33 pairs in one lumen, 0 of 246 short loops inside one lumen, 653 dead ends, 77.7% of the mask covered |
| graph.6b | H | Every rule that decides what a structure is (loop, doubled edge, bridge, dead end) was fitted to a marked grid, and the fit is saved: right, wrong, and the all-one-way baseline. | ✗ | About 210 marks over seven grids under `outputs/*_review/`, but the only fit saved is `grid_d_fit.py`. `dead_end_review` has keys and no marks |
| graph.6c | R | The coverage measures agree, or the difference is explained. | ✗ | E14.5 at `e2b4830` gives 77.7% of the mask covered in the artefact report, but 47.1% graph/mask and "979 of 1205 vessels missing" (18.8% represented) in the consistency log. By that log's measure the skeleton itself is only 31.0% represented (`outputs/p6/o2_current.log:10-20`), so what it counts as a "vessel" needs explaining before either number becomes a target |
| graph.6d | R | No correctness bug found in review is open. | ✗ | Unpadded `binary_closing` erodes voxels at the image faces (`PLAN_dale_review_fixes.md`, item 2) |
| graph.8a | A | On phantoms of known topology, at two voxel sizes and two noise levels, junction precision and recall are both ≥ 0.95 and centreline length is within 5% *(proposed)*. | ✗ | No image-to-graph comparison against a known topology exists (plan step R4) |
| graph.8b | H | Each fitted rule agrees with ≥ 90% *(proposed)* of the marks on a fresh grid it was not fitted on. | ✗ | Doubled-edge rule on fresh grid F: 14 of 30 by the survey's reading of the key files, against 17 for marking everything "vessel". Confirm that reading before acting on it |
| graph.8c | A | The artefact report is within its targets on two real datasets. | ✗ | No targets are written yet. The Nerve fixture holds the invariants (0 loops inside one lumen, no off-mask dead ends) |
| graph.10a | H | The graph agrees with another person's manual tracing of a real sub-volume, junction F1 ≥ 0.9 *(proposed)*. | ✗ | |
| graph.10b | A | The phantom scores and the artefact report run on every PR, and a regression fails it. | ✗ | `scripts/compare_branches.py` diffs two branches, but by hand and with no ground truth |

### 2. Diameter accuracy — `diameter`

*Is each vessel's lumen diameter right?* This covers FWHM, the raw-section fit, the mask (EDT),
endothelial, the class median and the table.

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| diameter.2a | A | Every edge gets a diameter, with its source recorded. | ✓ | The FWHM → raw section → mask → class median → table chain; `diameter_source` on edges (`branch_hover.py:25`) |
| diameter.4a | A | Each method is tested on synthetic vessels of known width, with tolerances pinned. | ✓ | `test_diameter_benchmark.py`: raw section ±7–10%, FWHM ±15%, endothelial ±6%, over three orientations, anisotropic blur and noise. EDT ±6–12% (`test_haemodynamics_edt_diameter.py:528-533`) |
| diameter.4b | A | FWHM's false positives are caught on the run's own image. | ✓ | Decoy check, on by default (`fwhm_decoys.py`, `test_fwhm_decoys.py`) |
| diameter.6a | A | A benchmark table (method × width × orientation × noise) is recorded at a commit, not only as pass/fail. | ✗ | The tolerances are pinned, but the measured errors are never written out |
| diameter.6b | A | A real run reports the planted-vessel error and the decoy false-positive rate. | ✗ | `fwhm_planted.py` is used only by the FWHM optimiser, and no planted results are saved |
| diameter.8a | A | Planted-vessel mean relative error is ≤ 10% *(proposed)* for vessels at least twice the PSF wide, on two real datasets. | ✗ | |
| diameter.8b | H | At least 30 vessels measured by hand agree with the method within 15% *(proposed)*, with the bias reported. | ✗ | No manual reference exists (plan step R5) |
| diameter.10a | H | A second annotator, or an independent modality, agrees. | ✗ | |

### 3. Haemodynamic accuracy — `flow`

*Are resistances, flows and haematocrit right?* This covers viscosity, Poiseuille, the solve,
haematocrit, constriction and perturbations.

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| flow.2a | A | Pressures and flows are solved on a real network. | ✓ | E14.5 through solve in 1,834 s (`outputs/p6/o2_current/solve_summary.json`) |
| flow.4a | A | Analytic cases are exact: one Poiseuille tube, series and parallel, Kirchhoff at every node, sparse equal to dense, phase separation and red-cell conservation. | ✓ | `test_simple_network_haemodynamics.py:100-133`, `test_flow_solvers.py:125, 207-217`, `test_haematocrit_distribution.py:135, 173, 231, 293` |
| flow.4b | R | Every law matches the publication it cites, or the citation says what was changed. | ✗ | **The default `plasma_column` "in vitro" Pries law uses the in vivo base `6·exp(−0.085D)`.** Pries 1992's in vitro law is `220·exp(−1.3D)`. That is 3.4× at 5 µm, and the tests pin the hybrid. A decision for Harvey and Finbar (`PLAN_dale_review_fixes.md`, item 1) |
| flow.6a | A | The viscosity and phase-separation laws are checked against published data points, not only re-derived formulas. | ✗ | The formulas are hand-evaluated (`test_viscosity_laws.py:367-373`). There is no comparison with tabulated data |
| flow.6b | A | Flow balance is checked on real data. | ✓ | Nerve (`test_nerve_pipeline.py:141`) |
| flow.6c | A | On each reference dataset, the share of centreline with a solved flow is recorded and the unsolved part is explained. | ✗ | One E14.5 summary lists 1,867 of 3,146 edges unsolved. The script that wrote it is gone, so its meaning is unconfirmed |
| flow.8a | A | An independent solver (NetFlow) on the same network agrees within 1% *(proposed)*, with distributed haematocrit included. | ✗ | NetFlow is cited for the relaxation schedule but never compared |
| flow.8b | A | The sensitivity of network resistance to diameter error and to boundary placement is quantified and reported. | ✗ | |
| flow.10a | H | Flow is compared with measured in vivo flow or red-cell velocity for at least one network, and the differences are explained. | ✗ | |

### 4. Statistics — `stats`

*Is every reported number correct, defined and understood?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| stats.2a | A | A run writes the statistics CSV. | ✓ | `{stem}_statistics.csv`, 95 rows on the synthetic network. Off by default (`schema.py:3452`) |
| stats.4a | A | Core morphometry is checked against hand-computed values on small graphs. | ✓ | `test_statistics.py`: Murray's law, cyclomatic number, fractal dimension of a line and a plane, path efficiency |
| stats.4b | A | Each network analysis is checked against an analytic or brute-force case. | ✓ | Transit times in series and parallel, Kirchhoff index closed form, occlusion by brute force, min-cut, Fiedler on a path, tissue slab and sphere |
| stats.6a | R | Every reported quantity has a definition and a unit a user can read, in a glossary. | ✗ | Only setting help strings and about 10 pattern-based CSV notes (`csv_export.py:97-122`) |
| stats.6b | A | Every row has its unit, parsed correctly and spelled one way. | ✗ | 21 of 95 rows have a unit. "Fractal Dimension (Node Positions)" parses its unit as "Node Positions". Units are spelled microns, um and micron³ |
| stats.6c | A | Sampled quantities use a fixed seed. | ✓ | `REPRODUCIBILITY_SEED = 42` (`statistics/_sampling.py:13`) |
| stats.6d | A | A synthetic network with ground-truth statistics is compared end to end. | ✓ | `test_synthetic_network_statistics.py` |
| stats.8a | A | The statistics are cross-checked against an independent implementation, or against published values for a published network. | ✗ | |
| stats.8b | A | Sampled quantities are shown to be stable across seeds, with the spread reported. | ✗ | |
| stats.10a | H | A domain expert other than the developer (George, Rebecca or Mike) has reviewed the glossary. | ✗ | Waits on the glossary (`stats.6a`) |

### 5. Inputs and segmentation — `input`

*Does every supported input load correctly, and is the segmentation known to be good?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| input.2a | A | A segmented TIFF and an H5 load into the pipeline. | ✓ | `io/load.py:765-776` |
| input.4a | A | Axis order, anisotropic voxels, 2D input, mask shape mismatch and the ilastik call are tested. | ✓ | `test_axis_order.py` (17), `test_anisotropic_voxel_size.py` (7), `test_voxel_validation.py` (21), `test_load_2d.py` (11), `test_load_and_validate_vessel_masks.py` |
| input.4b | A | Missing voxel metadata falls back with a warning, and the run records the fallback. | ✓ | `voxel_validation.py:65-76`; `{stem}_voxel_size.json` records status and source |
| input.6a | A | Preflight catches missing, mismatched and wrong-kind inputs before any work starts. | ✓ | 14 checks (`pipeline/checks.py:595-616`), run before every panel run |
| input.6b | A | "Check segmented image" reports quality numbers against the raw image. | ✓ | `segmentation_quality.py`, `segmentation_raw_comparison.py`. Self-consistency only; see input.10a |
| input.6c | R | The ilastik vessel label is explicit, and output reuse checks the classifier and input, not just file times. | ✗ | Minority-label guess; reuse by mtime (`PLAN_dale_review_fixes.md`, item 4) |
| input.8a | A | A headless ilastik run is tested end to end with a real project on Windows and Linux. | ✗ | |
| input.8b | R | Common microscope formats load with the right voxel sizes: OME-TIFF directly, CZI, ND2, LIF and OME-Zarr through a documented route. | ✗ | OME/ImageJ TIFF works (`io/channels.py`). The others are neither read nor documented |
| input.10a | H | Segmentation Dice against a hand-labelled sub-volume is reported for each reference dataset. | ✗ | |

### 6. Speed — `speed`

*Is a run fast enough that people run it, and do we know where the time goes?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| speed.2a | A | A full run completes on E14.5. | ✓ | 1,834 s through solve (`outputs/p6/o2_current/solve_summary.json`) |
| speed.4a | A | Every run records each stage's wall time, in the log and in a file. | ✗ | `ProgressEvent` has no duration. The run log has no timestamps (`gui/run_log.py:95`) |
| speed.4b | R | The hot spots have been profiled at least once. | ✓ | Per-step build times in `scripts/graph_artefacts/build_side.py`. On E14.5 the build takes 255 s, 134 s of it mask recovery |
| speed.4c | R | Long work runs off the GUI thread. | ✓ | `thread_worker` for the run, both optimisers, segmentation scoring and the review scans (`gui/_widget.py`) |
| speed.6a | A | A per-stage benchmark table exists for each reference dataset at a recorded commit. | ✗ | Only E14.5 graph build per step, and one full run |
| speed.6b | R | The targets are written: fixtures under 2 min, E14.5 through export under 20 min on the development workstation *(proposed)*. | ✗ | |
| speed.6c | H | There is no open report of the panel freezing. | ✗ | `hang_debug.log` (2026-09-29) caught no stuck frame, and it was never closed |
| speed.8a | A | The targets are met on every reference dataset. | ✗ | |
| speed.8b | A | An automated benchmark flags a stage that slows by more than 20%. | ✗ | |
| speed.10a | A | Time is measured against volume at three or more sizes, and no stage grows faster than its volume without a stated reason. | ✗ | |

### 7. Memory and scale — `memory`

*Does it fit, and does it say beforehand when it will not?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| memory.2a | A | E14.5 runs on the development workstation (31.7 GB). | ✓ | `outputs/p6/o2_current.log` |
| memory.4a | A | Disk-backed volumes and the sparse solver exist and are tested. | ✓ | `use_memmap_loading`, `test_memmap_*.py`, `test_low_ram_*.py`; sparse solver equal to dense (`test_flow_solvers.py:125`) |
| memory.4b | A | A worker that runs out of memory falls back instead of losing the run. | ✓ | `run_edge_tasks` falls back to one process (`haemodynamics/sections.py:78-146`). Seen working in `o2_current.log:259-281` |
| memory.6a | A | Peak memory per stage is measured on each reference dataset. | ✗ | Nothing measures peak memory: no psutil, no tracemalloc |
| memory.6b | R | System requirements (RAM for a given volume) are documented for users. | ✗ | |
| memory.6c | R | The default solvers stay within memory at 20,000 nodes. | ✗ | Dense is the default (8N², 3.2 GB at 20k), and so is the eigendecomposition two-point resistance ("hours at 20,000"; `resistance.py:14-25`) |
| memory.8a | A | The largest target dataset (Mason SM or the brain) completes within the documented limits. | ✗ | Mason SM (130 M voxels) has no recorded run |
| memory.8b | A | Preflight warns when a setting will exceed the available memory: the dense solver, eigendecomposition, or the thick-vessel path ignoring memmap. | ✗ | The dense solve only logs a time estimate (`resistance.py:539-547`) |
| memory.10a | A | Memory is measured against volume at three or more sizes, and E14.5 fits in 16 GB *(proposed)*. | ✗ | |

### 8. Usability — `usability`

*Can a biologist get from an image to an answer without help?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| usability.2a | A | The panel runs the whole pipeline. | ✓ | `test_gui_widget.py:468` |
| usability.4a | A | Every setting has hover text with its unit, enforced by a test. | ✓ | `test_gui_tooltips.py`, all 435 settings |
| usability.4b | A | A failed run says why in the panel, and preflight refuses a run that would fail on its inputs. | ✓ | Report box and log dock (`_widget.py:5885`); preflight at `_widget.py:13073` |
| usability.4c | A | Runs save and load, and any tab can re-run from its stage. | ✓ | `.haemorun` (`test_gui_run_snapshot.py`, 30 tests); `stage_checkpoints.py` |
| usability.6a | R | A written task list (install, open, run, inspect, edit, export, re-run) has been walked through by the developer, timed, without the console. | ✗ | No task list exists (plan step R5) |
| usability.6b | A | A segmented TIFF runs end to end with every setting at its default. | ✗ | Not shown. The panel test sets boundary boxes (`test_gui_widget.py:480-486`) |
| usability.6c | A | Hand edits can be undone. | ✗ | No undo. Only Revert or Run from this stage |
| usability.6d | A | Every control, not only every setting, has hover text, checked by scanning the built panel. | ✗ | Non-setting controls are checked against a hand-kept list (`chrome_tooltips.py:72-110`) |
| usability.8a | H | Two or more of the independent testers (George, Rebecca, Mike) complete the task list unaided. | ✗ | No usability testing recorded |
| usability.8b | H | System Usability Scale ≥ 68 *(proposed)* over those testers. | ✗ | |
| usability.10a | H | Five or more users, SUS ≥ 80 *(proposed)*. | ✗ | |

### 9. Visualisation — `visual`

*Do the viewer and the figures show the result truthfully and readably?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| visual.2a | A | Each stage's results appear in napari as it finishes. | ✓ | `gui/results.py` |
| visual.4a | A | Layers are drawn at the right scale (graph in microns, volumes at voxel size), and a test pins it. | ✓ | `test_gui_results.py` (registration) |
| visual.4b | A | Every computed per-vessel quantity can be chosen in Colour by, and unsolved vessels are shown as unsolved. | ✓ | 16 + 30 columns (`results.py:489`), `SOLVED_FLOW_COLUMNS` grey |
| visual.6a | R | Every saved figure has axis labels with units, and a labelled colour bar wherever colour carries data. | ✗ | `plot.py`: colour bar "Edge Weight" with no unit; plotly axes "X/Y/Z" with no units |
| visual.6b | R | The default colour maps are perceptually uniform and colour-blind safe. | ✓ | viridis, coolwarm, cyclic. jet and rainbow are still offered in the picker |
| visual.6c | A | The tubes for E14.5 render interactively, with the frame time measured. | ✗ | |
| visual.6d | R | Every figure is saved at one resolution, ≥ 300 dpi, or as vector. | ✗ | 150, 200 and 300 dpi are all used |
| visual.8a | H | A marked grid of views and figures (GOOD / POOR) is ≥ 80% GOOD *(proposed)*, marked by an independent tester (George, Rebecca or Mike). | ✗ | |
| visual.10a | H | Figures are used in a publication or talk without re-plotting outside HaemoLynx. | ✗ | |

### 10. Exports — `export`

*Can someone else use the outputs without HaemoLynx, and without misreading them?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| export.2a | A | A run writes VTK and CSV. | ✓ | `vtk_io.py:116-217`, `csv_export.py` |
| export.4a | A | Tests read every export back and check its content. | ✓ | `test_vtk_io.py` (23), `test_statistics.py:785`, `test_tissue_volume.py:380`, `test_perturbation_outputs.py`, `test_gui_run_snapshot.py` |
| export.4b | R | No export can be misread: unsolved vessels' flows are blank, not numbers. | ✗ | The VTK writes unsolved vessels' flows as real values and has no solved flag (`PLAN_dale_review_fixes.md`, item 5) |
| export.6a | R | A data dictionary documents every exported file, column or array, and unit. | ✗ | None. The README and the guide do not describe the outputs |
| export.6b | A | Units are present and consistent everywhere, VTK arrays and perturbation CSVs included. | ✗ | VTK arrays have no units. Perturbation resistance and flow columns have none |
| export.6c | A | Outputs carry the HaemoLynx version, the commit and the settings used, with a config beside them. | ✗ | `__version__` appears only in the About text. Settings travel only inside `.haemorun` |
| export.6d | A | The VTK carries each vessel's diameter and its solved flag. | ✗ | |
| export.8a | A | The graph is exported in an open format (GraphML or node-link JSON) besides the pickle. | ✗ | Pickle only |
| export.8b | A | A script checks that each export opens in a third-party tool (ParaView or VTK, pandas, NetworkX). | ✗ | |
| export.10a | A | The export format is versioned, and the previous release's outputs and `.haemorun` files still load (tested). | ✗ | `.haemorun` rejects any version but 1 (`run_snapshot.py:201`) |

### 11. Documentation and help — `docs`

*Can a new user and a new developer learn it from what is written?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| docs.2a | R | The README covers install and a first run. | ✓ | `README.md` (53 lines) |
| docs.4a | A | The tutorial runs in CI. | ✓ | `tests/integration/test_pipeline_tutorial.py`; CI runs the whole suite (`pytest-pr.yml`) |
| docs.4b | A | Every setting has help text. | ✓ | `Setting` refuses empty help; 0 of 435 empty |
| docs.4c | R | A user guide covers configuring a run, the panel and the command line. | ✓ | `tutorials/README.md` (308 lines) |
| docs.6a | H | Every tab's help page has text **written by Harvey**. Agents count the pages and never write them (`CLAUDE.md` § Help menu). | ✗ | 0 of 10 written (`gui/stage_help.py`) |
| docs.6b | R | The user guide covers every tab, including Post processing, Perturbations and Additional measurements, with screenshots. | ✗ | None of these three is in `tutorials/README.md`. No screenshots anywhere |
| docs.6c | R | A CHANGELOG records user-visible changes per version. | ✗ | None |
| docs.6d | R | The examples and tutorial name only things that exist. | ✗ | `pipeline_tutorial.py:607` says colour by `flow`. The arrays are `flow_signed` / `flow_abs` |
| docs.8a | H | A new user reaches a result from the docs alone (shared with usability.8a). | ✗ | |
| docs.8b | R | A methods description lets a paper's methods section be written from it, with citations. | ✗ | `pipeline/citations.py` lists the references per run; there is no methods text |
| docs.10a | R | The docs are published, versioned with releases, and include an API reference. | ✗ | |

### 12. Robustness — `robust`

*Does bad or unusual input give a clear stop, never a wrong answer or a lost run?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| robust.2a | A | Preflight checks the required paths and settings before work starts. | ✓ | `pipeline/checks.py` |
| robust.4a | A | Invalid inputs raise a named error, each with a test: empty skeleton, empty graph, mask shape mismatch, bad axis order. | ✓ | `test_build_network_empty_skeleton.py`, about 109 empty-input tests, `test_shape_mismatch_raises`, `test_axis_order.py`. 252 `pytest.raises` |
| robust.4b | A | A failed stage in the panel keeps the earlier stages' results, and the panel stays usable. | ✓ | `gui/run_state.py`, `stage_checkpoints.py` |
| robust.6a | A | An edge-case suite runs every stage and finishes or names its error: empty mask, one vessel, one voxel thick, 2D, strong anisotropy, vessels touching the border, no inlets or outlets. | ✗ | The pieces exist, but there is no suite through every stage. The 3D-only features are unverified on 2D input (`load_2d.py` docstring) |
| robust.6b | R | No crash or wrong-answer bug found in review is open. | ✗ | Face erosion and the ilastik label guess (`PLAN_dale_review_fixes.md`, items 2 and 4) |
| robust.6c | H | There is no open report of a crash or freeze. | ✗ | The 2026-09-29 freeze (`hang_debug.*`) has no recorded outcome |
| robust.8a | A | Three or more datasets from different labs or tissues run at one commit without code changes. | ✗ | E14.5, Nerve, Mason SM, Ross and carotid exist, but none has a recorded joint run |
| robust.10a | A | Random masks (fuzzing) run with no unhandled exception. | ✗ | |

### 13. Reproducibility — `repro`

*Can a result be traced to what made it, and made again?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| repro.2a | A | Settings save to a config file and load back. | ✓ | `parsers/config.py`, `--save-config` |
| repro.4a | A | Old configs and saved runs upgrade, and unknown keys are refused with a suggestion. | ✓ | `Schema.upgrade` (`parsers/schema.py:562-594`) |
| repro.4b | A | Every random choice has a fixed default seed, and an unseeded replicate's seed is recorded. | ✓ | Seeds 42 and 20240917; `_block_replicates.csv` |
| repro.6a | A | Every output folder holds the settings, the HaemoLynx version, the commit and the input checksums. | ✗ | None of these is written (see export.6c) |
| repro.6b | A | The citations for the methods a run used are written beside its outputs. | ✓ | `{stem}_citations.txt`, on by default (`test_citations.py`) |
| repro.6c | A | The same input and settings, run twice, give identical outputs, tested end to end. | ✗ | No such test. Reconnect adds edges in completion order (`graph/reconnect.py:755-761`) |
| repro.8a | A | Windows and Linux give the same results within tolerance (checked). | ✗ | |
| repro.8b | A | Results do not depend on the worker count (checked). | ✗ | |
| repro.10a | R | A release is archived with a DOI and an example dataset. | ✗ | |

### 14. Testing and CI — `tests`

*Would a regression be caught before a user meets it?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| tests.2a | A | A test suite exists and runs. | ✓ | 6,064 collected: 351 slow, 663 gui, 107 integration |
| tests.4a | A | CI runs unit, integration and GUI tests on every PR. | ✓ | `pytest-pr.yml`: four jobs, Python 3.9–3.12, GUI under xvfb |
| tests.4b | R | Bug fixes carry regression tests (the policy is in `CLAUDE.md`, and recent commits follow it). | ✓ | |
| tests.4c | A | The whole suite, GUI and slow included, has a recorded pass at a commit under two weeks old. | ✗ | The last recorded baseline is 2026-09-25 (`CLAUDE.md` Cleanup plan) |
| tests.6a | A | Line coverage is measured and reported, ≥ 70% of non-GUI code *(proposed)*. | ✗ | No coverage tooling |
| tests.6b | A | CI also runs on pushes to the shared branches or on a schedule, not only on PRs. | ✗ | `pull_request` only |
| tests.6c | A | The pre-commit suite (not slow, not gui) takes under 10 min *(proposed)*. | ✗ | 12 min 40 s; no `pytest-xdist` |
| tests.8a | A | Windows (and macOS) run in CI. | ✗ | ubuntu only. Windows is tested only locally |
| tests.8b | A | Coverage ≥ 80% *(proposed)*, and no known flaky test. | ✗ | |
| tests.10a | A | Coverage ≥ 90% of non-GUI code, with property-based tests for the core maths. | ✗ | |

### 15. Installation and packaging — `package`

*Can anyone install it, on any platform, as a released version?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| package.2a | A | An editable install works. | ✓ | |
| package.4a | A | The wheel is built and used from outside the repository in CI, with dependency floors and the napari manifest checked. | ✓ | `installed-package`, `dependency-floors` and `napari-gui` jobs |
| package.4b | A | A release has been published from the release workflow. | ✓ | 0.1.0 on PyPI, 2026-08-10, tag `v0.1.0` (`release.yml`, Trusted Publishing) |
| package.6a | R | The version moves with user-visible changes, and a CHANGELOG says what changed. | ✗ | Still 0.1.0, 457 commits after the tag. No CHANGELOG |
| package.6b | H | The install instructions have been followed on Windows (Harvey), macOS (Mason) and Linux (Finbar), and recorded. | ✗ | Windows locally and Linux in CI, but no one has followed the README on each and recorded it |
| package.6c | R | `CITATION.cff`, `CONTRIBUTING` and the licence are present. | ✗ | Apache-2.0 licence only |
| package.8a | A | The current release is on PyPI and listed on the napari hub. | ✗ | The napari hub returns 404 |
| package.8b | H | A new user installs it cleanly on each OS. | ✗ | |
| package.10a | A | A conda-forge package exists, and releases are automated end to end. | ✗ | |

### 16. Code maintainability — `code`

*Can someone other than its authors change it safely?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| code.2a | R | An installable package, split into subpackages by concern. | ✓ | 166 modules, 99,806 lines |
| code.4a | A | The public API is guarded. | ✓ | `test_public_api.py` |
| code.4b | A | At least 80% of functions have type hints. | ✓ | 91% of returns and 84% of parameters annotated; `py.typed` ships |
| code.4c | R | The `CLAUDE.md` Cleanup plan is finished. | ✗ | Phase 2 has three renames left (`io/` and `graph/automated_vessel_assignment.py`, `haemodynamics/automated.py`). The README preset note is also open |
| code.6a | A | A linter and formatter are configured and clean in CI. | ✗ | None configured |
| code.6b | R | No module is over 3,000 lines *(proposed)* without a stated reason. | ✗ | `gui/_widget.py` 14,178, `pipeline/stages.py` 5,114. `pipeline/schema.py` (5,867) is declarations, which is a reason |
| code.6c | R | The repository root holds only project files. Personal configs, debug scripts and plans are ignored or kept elsewhere. | ✗ | Five configs, `hang_debug.*`, `PLAN_*.md` and `mason_claude.md` at the root |
| code.8a | A | A type checker is clean on the core subpackages in CI. | ✗ | |
| code.8b | R | A contributor guide gives an architecture overview for people, not only agents. | ✗ | `tutorials/dev_README.md` covers the workflow. The architecture is only in `CLAUDE.md` |
| code.10a | H | A new contributor lands a change by following the guide alone. | ✗ | |

---

## Review log

Add a row for every scoring, and for every change to a check or a target.

| Date | Commit | By | What changed |
|---|---|---|---|
| 2026-10-10 | `9c6a15a` | Claude | First scoring of all 16 aspects from code, tests and recorded outputs. Headline 3, mean 3.8. No H checks marked yet. Every *(proposed)* target still needs Harvey to confirm it. |
| 2026-10-10 | `9c6a15a` | Harvey | Named the testers: Harvey, Finbar, Mason, George, Rebecca and Mike. George, Rebecca and Mike are independent (no commits). `usability.8a` and `visual.8a` now name them instead of saying "someone other than the developer". No score changes. |
| 2026-10-10 | `9c6a15a` | Harvey | George, Rebecca and Mike are all first-time users and all domain experts (`stats.10a`). `package.6b` assigned: Windows to Harvey, macOS to Mason, Linux to Finbar. No score changes. |
