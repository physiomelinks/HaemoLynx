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
| 1 | [Graph accuracy](#1-graph-accuracy--graph) | **4** | fitted rules' results not saved for most grids (6b); coverage measures disagree (6c); the prune's protrusion test cannot tell a hair from a vessel's end (6d) |
| 2 | [Diameter accuracy](#2-diameter-accuracy--diameter) | **4** | no recorded benchmark table (6a); planted-vessel error not reported for real runs (6b) |
| 3 | [Haemodynamic accuracy](#3-haemodynamic-accuracy--flow) | **4** | laws checked against the papers' own descriptions, not against measured data (6a); the solved share not recorded on Nerve (6c) |
| 4 | [Statistics](#4-statistics--stats) | **5** | no glossary of definitions and units (6a); 21 of 95 rows carry a unit (6b). The cross-check against independent tools (8a) passes, held back by these |
| 5 | [Inputs and segmentation](#5-inputs-and-segmentation--input) | **5** | ilastik output: vessel label guessed, reuse judged by file time (6c) |
| 6 | [Speed](#6-speed--speed) | **4** | per-stage tables for the fixtures, E14.5 and Nerve, but not Mason SM or the brain, whose inputs are not on this machine (6a); no written targets (6b); the 2026-09-29 freeze not closed (6c) |
| 7 | [Memory and scale](#7-memory-and-scale--memory) | **4** | peak memory per stage measured on the fixtures, E14.5 and Nerve, but not Mason SM or the brain (6a); no stated requirements (6b); the default solvers do not scale (6c) |
| 8 | [Usability](#8-usability--usability) | **4** | no task list walked through (6a); defaults untested end to end (6b); no undo (6c) |
| 9 | [Visualisation](#9-visualisation--visual) | **4** | figures without units (6a); tube frame rate not measured (6c); mixed dpi (6d) |
| 10 | [Exports](#10-exports--export) | **3** | blind-end vessels keep a flow of exactly 0 in the VTK, flagged, not blank: a decision (4b) |
| 11 | [Documentation and help](#11-documentation-and-help--docs) | **4** | help pages empty, 0 of 10, for Harvey to write (6a); guide skips Post processing and Perturbations (6b) |
| 12 | [Robustness](#12-robustness--robust) | **4** | no edge-case suite through every stage (6a); open review bugs (6b) |
| 13 | [Reproducibility](#13-reproducibility--repro) | **5** | no end-to-end identity test (6c) |
| 14 | [Testing and CI](#14-testing-and-ci--tests) | **3** | no full-suite pass recorded since 2026-09-25 (4c) |
| 15 | [Installation and packaging](#15-installation-and-packaging--package) | **4** | still 0.1.0 after 457 commits, no CHANGELOG (6a) |
| 16 | [Code maintainability](#16-code-maintainability--code) | **3** | Cleanup plan Phase 2 unfinished (4c) |

**Headline: the lowest score is 3** (exports, testing, maintainability).
The mean is 4.0. The release gate needs every aspect at 7 or more, and the science aspects at 8.

Most of these scores are low because **nothing has been measured, not because the work is
poor**. The test suite is large (6,064 tests) and checks the analytic cases well. What is
missing almost everywhere is a recorded number on real data, a written target, and someone other
than the developer checking the result. Levels 6 and 8 ask for exactly those three things.

### Work in order

This order follows the gate, which is set by the lowest aspects, and favours the cheap fixes:

1. ~~**Decide the Pries "in vitro" law** (`flow.4b`)~~. Done 2026-10-13: Harvey chose the
   literal fix, plus Pries & Secomb 2005 as a choice. Haemodynamics went from 3 to 4.
2. ~~**Record stage times and peak memory in every run** (`speed.4a`, `memory.6a`, step R2 of the
   plan)~~. Done 2026-10-15: every run writes a manifest beside its outputs, with each stage's
   time and peak memory, the settings, version, commit and input checksums. Speed went from 3 to
   4 and reproducibility from 4 to 5. The fixtures, E14.5 and Nerve were measured the same day
   at `9b679b8`. `memory.6a` and `speed.6a` still need Mason SM and the brain, whose inputs are
   not on this machine.
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
| graph.6b | H | Every rule that decides what a structure is (loop, doubled edge, bridge, dead end) was fitted to a marked grid, and the fit is saved: right, wrong, and the all-one-way baseline. | ✗ | Dead ends: Harvey marked G1-G3 on 2026-10-12 (87 judged), and the fit is saved (`scripts/dead_end_review/fit_marks.py`, `outputs/dead_end_review/FIT.md`, E14.5 at 43dcf45). No rule reached 90%. The other grids' ~210 marks still have only `grid_d_fit.py`. Thick-vessel bridges: the 13 gap bridges the gate's new rules refuse (turn on the path drawn, a short bridge judged at its middle) were marked by Harvey on 2026-10-14, all 13 N (`outputs/thick_junction_review/grid_R_key.csv`, E14.5 at `95c43fa`): 13/13, the same as the all-one-way baseline, so it shows no wrong refusal and not what the rules miss. The wall-join, taper and fork rules came from Harvey's own examples, not a marked grid (grid W was built and set aside, `outputs/thick_junction_review/REPORT.md`). Grids T and U, Harvey 2026-10-14 (`outputs/thick_junction_review/grid_T_key.csv`, `grid_U_key.csv`): the 30 bridges not ending on a running thick edge all correct (no rule changed); of 12 vessels left unbridged, 9 rightly, 3 missed, no measure separating them (9/12, the all-one-way baseline, no rule built). Short loops: grids Q and Q2, Harvey 2026-10-15, 60 of the short-loop rule's breaks on E14.5 at `75564ff` drawn with the side it took (`outputs/loop_break_review/FIT_Q.md`, `FIT_Q2.md`): no wrong side, but loops round a hole over 10 um were real 29 of 32; hole size 53/60 leave-one-out against 33 all one way, now `short_loop_keep_hole_over_um` (`tests/test_lumen_loops.py`, `test_consolidate_lumen.py`) |
| graph.6c | R | The coverage measures agree, or the difference is explained. | ✗ | E14.5 at `e2b4830` gives 77.7% of the mask covered in the artefact report, but 47.1% graph/mask and "979 of 1205 vessels missing" (18.8% represented) in the consistency log. By that log's measure the skeleton itself is only 31.0% represented (`outputs/p6/o2_current.log:10-20`), so what it counts as a "vessel" needs explaining before either number becomes a target. Explained since (2026-10-12): a "vessel" there is a 26-connected mask component of 2+ voxels. On E14.5 (skeleton in `outputs/t6/skel`) the 832 of 1,205 missing from the skeleton are 0.36% of the mask, the largest 255 voxels, and the log line now reports that share (`test_skeleton_consistency_diagnostics.py:720`). Still open: 77.7% covered in the artefact report against 47.1% graph/mask |
| graph.6d | R | No correctness bug found in review is open. | ✗ | Face erosion fixed at `b99758b` (`test_morphology_at_faces.py`). Open: the prune's protrusion test cannot tell a hair along a wall from a vessel's collinear end (`PLAN_one_centreline_per_lumen.md`, 2026-10-11). Found in review and fixed (PLAN_bridging.md): `insert_node_on_edge` cut an edge against its points' order, and merges drew straight jumps across the gap, 35 um on E14.5 (`32813bc`, `tests/test_graph_edit.py:370`); a bridge's turn was judged on the chord, not the path drawn (`8ec5308`, `tests/test_graph.py:1449`); the thick-vessel wall join targeted flickers off the centreline (`tests/test_graph_thick_vessel_junctions.py:550`) and closed short loops the clean-up then kept (`abdab60`, `:635`) |
| graph.8a | A | On phantoms of known topology, at two voxel sizes and two noise levels, junction precision and recall are both ≥ 0.95 and centreline length is within 5% *(proposed)*. | ✗ | No image-to-graph comparison against a known topology exists (plan step R4) |
| graph.8b | H | Each fitted rule agrees with ≥ 90% *(proposed)* of the marks on a fresh grid it was not fitted on. | ✗ | Doubled-edge rule on fresh grid F: 14 of 30 by the survey's reading of the key files, against 17 for marking everything "vessel". Confirm that reading before acting on it. Short-loop hole rule, fitted on grid Q at 9.2 um, on fresh grid Q2: 23/30, the same as marking all real (Q2 held only loops round holes over 7 um; above the threshold 21 of 26 real), below 90% |
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
| flow.4b | R | Every law matches the publication it cites, or the citation says what was changed. | ✓ | Fixed 2026-10-13, after `5ef7d53`. The in vitro form used the in vivo base `6·exp(−0.085D)`; it is now Pries 1992's `220·exp(−1.3D)`. Each Pries form (in vitro 1992, in vivo 1994, and Pries & Secomb 2005, now a choice) equals its equations as written out from the papers (2005 eqs 5–11, Secomb & Pries 2013 eqs 3–7), at 11 diameters × 4 haematocrits (`test_viscosity_laws.py:278`). A regression test pins the hybrid gone (`:285`). Each form has its own citation, and the power law's says it matches no published law (`citations.py:350`). Phase separation matches 2005 eqs 2–4 (`haematocrit_distribution.py:145-159`); its X0 ≤ 0.49 clamp, which only acts on parents under ~1 µm, is commented at `:143` |
| flow.6a | A | The viscosity and phase-separation laws are checked against published data points, not only re-derived formulas. | ✗ | The Pries forms are checked against what the papers say of them: the in vitro minimum, ~1.25 at 6–7 µm, and ~3.2 above 1 mm (`test_viscosity_laws.py:295`); the 2005 layer, 0.8–1 µm at 10–40 µm (`:310`), peaking at 10.5 µm (`:328`); the 2005 viscosity's minimum at 30–50 µm (`:339`). Those are the papers' descriptions of their own fits, not their measurements. Phase separation has no such check. There is still no comparison with tabulated data |
| flow.6b | A | Flow balance is checked on real data. | ✓ | Nerve (`test_nerve_pipeline.py:141`) |
| flow.6c | A | On each reference dataset, the share of centreline with a solved flow is recorded and the unsolved part is explained. | ✗ | The solve now logs and records why each unsolved vessel carries no flow (`graph.FLOW_STATE`, `test_open_ends.py`; at `b6d9210`). E14.5: 1,539 of 2,839 vessels lie on no inlet-to-outlet path: 585 behind dead ends inside the tissue, 954 in pieces without an inlet and an outlet, 0 behind a closed face end (`outputs/e14_full/run.log`). Nerve not yet recorded |
| flow.8a | A | An independent solver (NetFlow) on the same network agrees within 1% *(proposed)*, with distributed haematocrit included. | ✗ | NetFlow is cited for the relaxation schedule but never compared |
| flow.8b | A | The sensitivity of network resistance to diameter error and to boundary placement is quantified and reported. | ✗ | Only the boundary condition at open ends: across five models network resistance moves under 8% on E14.5 (`scripts/open_end_sensitivity.py`, `outputs/open_end_sensitivity_e14/`, at `cd71285`). Diameter error and boundary placement are not measured |
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
| stats.8a | A | The statistics are cross-checked against an independent implementation, or against published values for a published network. | ✓ | Against skan 0.13.1 (BSD-3), maskel 2.0.2 (MIT, `80acafd`) and VesselVio (GPL-3, `29f5106`, run as its own program), each reading the same mask, by `scripts/stats_cross_check.py` with HaemoLynx's src at `30fd455`. Its ledger names the HaemoLynx step behind each difference. **Synthetic dense capillary bed** (`tests/dense_capillary_fixtures.py`, design known; `outputs/stats_cross_check/bed/`): skan, maskel and VesselVio all find the bed as segmented: 72 segments, 31 branch points, 26 ends, since two dropouts split two vessels. Their lengths are 1,151-1,155 µm against 1,199, because Lee ends stop 1-2 µm short of a face. HaemoLynx finds 66 / 27 / 20 and 1,086 µm, and the tests pin each difference to its step (`test_stats_cross_check.py:210-299`). Skeleton cleaning bridges both dropouts. Cluster collapse folds the 12 µm vessel's two face stubs into their junctions. The short-loop rule removes both parallel 4 µm vessels (a 4 µm hole), and that is all of the raw skeleton left outside its vessels. On one skeleton its length equals skan's. Its diameter is 5.07 µm by cross-section against the design's 5.0; by its inscribed-radius method, skan's definition (2 × EDT), it reads 5.60 and skan 5.66. **E14.5** (`outputs/stats_cross_check/e14/comparison.md`, HaemoLynx with `config.yaml` less the raw image; its large-vessel masks place the thick centrelines): skan and maskel agree exactly, at 132.2 mm and 29,281 segments; VesselVio, pruning ends under 5 µm, gives 81.8 mm. HaemoLynx gives 39.4 mm, 1,571 segments, 680 branch points and 871 ends. Where the definitions coincide, the numbers do: on one skeleton the length is within 0.3%; the length-weighted tortuosity is 1.26 against skan's 1.24; the length-weighted diameter by 2 × EDT is 4.59 µm against 4.53 (VesselVio 4.74). Its cross-section diameter, 7.73 µm, is the area of z-elongated sections, not an error of the same measure. The totals differ by HaemoLynx's graph definition. Skeleton cleaning takes −34.7 mm, collapsing junctions within 5 µm −30.0 mm (15,410 connectors), the lumen clean-up −18.6 mm, smoothing −4.1 mm and the thick split and bridges −2.4 mm. 64% of skan's skeleton lies in the lumen of a vessel HaemoLynx drew (the same vessel again). The other 36% lies outside every one, as does 27% of HaemoLynx's own cleaned skeleton: a graph-completeness gap (`graph.6c`), not a statistics error |
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
| speed.4a | A | Every run records each stage's wall time, in the log and in a file. | ✓ | Since 2026-10-15. In a file: every run's `{stem}_run_manifest.json` gives each stage's start, end and `wall_seconds` (`pipeline/provenance.py`; `test_pipeline_provenance.py:92-130`), on a real staged run of the seven-vessel fixture (`:316`) and on a failed run (`:394`). In the log: a finished or failed stage's `ProgressEvent.seconds` (`progress.py:392`) is on its `log_progress` line, which the panel's run log shows (`test_pipeline_progress.py:394`, `test_gui_run_log.py:593`), and a run's last line lists every stage's time, for a command line that shows no progress (`test_pipeline_provenance.py:133`) |
| speed.4b | R | The hot spots have been profiled at least once. | ✓ | Per-step build times in `scripts/graph_artefacts/build_side.py`. On E14.5 the build takes 255 s, 134 s of it mask recovery. Profiled again 2026-10-12 on a 0.5 G-voxel crop of the 16 G-voxel skeletal-muscle stack (py-spy): all of recovery's first 40 min was the cross-section outline's point-in-mask lookups; after `_follow_rays` + `PackedMask` the E14.5 uncovered scan is 108 -> 18 s and recovery 164 -> 113 s, graph identical (`tests/test_mask_recovery.py` outline tests) |
| speed.4c | R | Long work runs off the GUI thread. | ✓ | `thread_worker` for the run, both optimisers, segmentation scoring and the review scans (`gui/_widget.py`) |
| speed.6a | A | A per-stage benchmark table exists for each reference dataset at a recorded commit. | ✗ | Measured 2026-10-15 at `9b679b8`, from a `git archive` of it, one `run_pipeline_stages` per dataset in its own process: `outputs/release_review/2026-10-15_9b679b8/benchmarks.md` has one table per dataset (each stage's wall time, peak RSS and `peak_rss_method`, with the voxel count and the machine's RAM), beside each run's manifest (`<dataset>_run_manifest.json`) and `<dataset>_stages.csv`; the same rows are in `outputs/release_review/benchmarks.csv`. Whole runs, on the 20-CPU, 31.7 GiB workstation while other sessions ran their own jobs: fixtures 11 s (seven-vessel), 9 s (bundled h5) and 44 s (the 41 M-voxel Nerve crop, with `test_nerve_pipeline.py`'s boundary boxes; on the schema defaults it fails at Solve); E14.5 (75 M voxels, `config.yaml`) 745 s, of which skeletonise 173 s, graph 244 s and diameters 276 s; Nerve (329 M voxels, the `compare_branches.py` default) 521 s, of which skeletonise 263 s and graph 238 s. Fails until Mason SM and the brain are run: Mason SM's stack is on an E: drive that is not attached, and `brain_microvessels.tiff` is not on this machine. Not compared with `speed.6b`'s targets, which are still proposed |
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
| memory.6a | A | Peak memory per stage is measured on each reference dataset. | ✗ | Since 2026-10-15 every run's manifest records each stage's peak resident memory, read with the standard library (psutil is not a dependency): GetProcessMemoryInfo on Windows, getrusage elsewhere; exact for a stage that raised the process peak, else sampled every 0.2 s; worker processes on POSIX only (`test_pipeline_provenance.py:185`, `:202`). Measured 2026-10-15 at `9b679b8` in `speed.6a`'s runs (`outputs/release_review/2026-10-15_9b679b8/`, a table and manifest per dataset). Run peaks: 0.6 GiB (seven-vessel fixture), 0.7 GiB (bundled h5), 2.6 GiB (Nerve crop, 41 M voxels), 5.1 GiB (E14.5, 75 M voxels) and 19.1 GiB (Nerve, 329 M voxels). On the last three the peak comes in graph building, with skeletonising 2-13% below it (both exact: `process_peak`). The later stages are `sampled` lower bounds, and on Windows they leave out worker processes, so a stage run in a process pool (E14.5's diameters, 276 s) under-reports. Fails until Mason SM and the brain are measured; their inputs are not on this machine |
| memory.6b | R | System requirements (RAM for a given volume) are documented for users. | ✗ | |
| memory.6c | R | The default solvers stay within memory at 20,000 nodes. | ✗ | Dense is the default, and so is the eigendecomposition two-point resistance ("hours at 20,000"). Measured 2026-10-15 (peak working set at 2,000-6,000 nodes, Windows): the dense flow solve peaks at 42 N² bytes, 16.8 GB at 20k, not the 8N² (3.2 GB) the help text said; the eigendecomposition at 49 N², 19.6 GB (`resistance.py` `DENSE_FLOW_SOLVE_BYTES_PER_NODE_SQUARED`, pinned against traced allocations in `test_flow_solvers.py` `test_the_estimate_covers_what_*`). The run now warns (memory.8b), but the defaults are unchanged |
| memory.8a | A | The largest target dataset (Mason SM or the brain) completes within the documented limits. | ✗ | Mason SM (130 M voxels) has no recorded run |
| memory.8b | A | Preflight warns when a setting will exceed the available memory: the dense solver, eigendecomposition, or the thick-vessel path ignoring memmap. | ✗ | Since 2026-10-15 the dense solver and the eigendecomposition are warned about when their estimate passes half the free memory, naming the need, the free memory and the switch to sparse (`resistance.warn_about_dense_solver_memory`). Not in preflight, which has no node count: from `build_network` as soon as the network is built, and from `solve` before the matrices are made (`test_build_network_stage.py` `test_the_built_network_warns_*`, `test_flow_solvers.py` `test_the_solve_stage_warns_*`). Still open: the thick-vessel path ignoring memmap |
| memory.10a | A | Memory is measured against volume at three or more sizes, and E14.5 fits in 16 GB *(proposed)*. | ✗ | |

### 8. Usability — `usability`

*Can a biologist get from an image to an answer without help?*

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| usability.2a | A | The panel runs the whole pipeline. | ✓ | `test_gui_widget.py:468` |
| usability.4a | A | Every setting has hover text with its unit, enforced by a test. | ✓ | `test_gui_tooltips.py`, all 435 settings |
| usability.4b | A | A failed run says why in the panel, and preflight refuses a run that would fail on its inputs. | ✓ | Report box and log dock (`_widget.py:5885`); preflight at `_widget.py:13073` |
| usability.4c | A | Runs save and load, and any tab can re-run from its stage. | ✓ | `.haemorun` (`test_gui_run_snapshot.py`, 34 tests); `stage_checkpoints.py` |
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
| export.4b | R | No export can be misread: unsolved vessels' flows are blank, not numbers. | ✗ | Since `b6d9210` the VTK blanks flows and pressures where no inlet and outlet reach (`no_boundary`), and writes `flow_solved` and `flow_state` on every vessel (`test_vtk_io.py:310`). Vessels behind dead ends inside the tissue keep the model's answer, no flow: `flow_abs`, `flow_signed` and `pressure_drop` exactly 0, set after the solve, every perturbation re-solve and every sweep point (before, the solve's rounding, up to 5e-20 m^3/s on E14.5), flagged `flow_solved=0`, `flow_state=blind_end` or `closed_end` (`0c6d13b`, `test_open_ends.py:244`, `:265`, `:290`; the dead-end plan's D4). Harvey decides whether that counts as blank |
| export.6a | R | A data dictionary documents every exported file, column or array, and unit. | ✗ | None. The README and the guide do not describe the outputs |
| export.6b | A | Units are present and consistent everywhere, VTK arrays and perturbation CSVs included. | ✗ | VTK arrays have no units. Perturbation resistance and flow columns have none |
| export.6c | A | Outputs carry the HaemoLynx version, the commit and the settings used, with a config beside them. | ✓ | The VTK vessels file carries the version, viscosity law, diameter basis, haematocrit and open-end model in its field data (`test_vtk_io.py:310`, `b6d9210`). Since 2026-10-15 the run manifest beside every run's outputs carries the version, the commit and the settings, with `{stem}_run_settings.yaml` as the config (`test_pipeline_provenance.py:92`, `:169`, `:519`). Read as the output folder: the VTK and CSV files do not each carry the commit |
| export.6d | A | The VTK carries each vessel's diameter and its solved flag. | ✓ | `diameter_um` and `flow_solved` on every vessel cell (`test_vtk_io.py:310-339`), at `b6d9210` |
| export.8a | A | The graph is exported in an open format (GraphML or node-link JSON) besides the pickle. | ✗ | Pickle only |
| export.8b | A | A script checks that each export opens in a third-party tool (ParaView or VTK, pandas, NetworkX). | ✗ | |
| export.10a | A | The export format is versioned, and the previous release's outputs and `.haemorun` files still load (tested). | ✗ | `.haemorun` rejects any version but 1 (`run_snapshot.py:221`) |

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
| robust.4a | A | Invalid inputs raise a named error, each with a test: empty skeleton, empty graph, mask shape mismatch, bad axis order. | ✓ | `test_build_network_empty_skeleton.py`, about 109 empty-input tests, `test_shape_mismatch_raises`, `test_axis_order.py`. 252 `pytest.raises`. A segmentation not the skeleton's shape now raises in skeleton cleaning and graph building too, where it used to switch the bridge gate off or be read at the wrong voxels (`d5ab09b`, `tests/test_skeleton_bridging.py:688`, `tests/test_graph_assemble.py:750`) |
| robust.4b | A | A failed stage in the panel keeps the earlier stages' results, and the panel stays usable. | ✓ | `gui/run_state.py`, `stage_checkpoints.py` |
| robust.6a | A | An edge-case suite runs every stage and finishes or names its error: empty mask, one vessel, one voxel thick, 2D, strong anisotropy, vessels touching the border, no inlets or outlets. | ✗ | The pieces exist, but there is no suite through every stage. The 3D-only features are unverified on 2D input (`load_2d.py` docstring) |
| robust.6b | R | No crash or wrong-answer bug found in review is open. | ✗ | The ilastik label guess (`PLAN_dale_review_fixes.md`, item 4). Face erosion fixed at `b99758b`; the pericyte perturbation and statistics crashes on thick-vessel bridges fixed at `dd570cb`, and the pericyte perturbation on pieces no inlet and outlet reach at `213476a`. Wrong-answer bugs found in the bridging review fixed: straight jumps across gaps from `insert_node_on_edge` (`32813bc`), a mask of the wrong shape ungating every bridge (`d5ab09b`), the bridging settings hidden for a run loading its saved skeleton while graph building still read them (`b3c085a`) |
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
| repro.6a | A | Every output folder holds the settings, the HaemoLynx version, the commit and the input checksums. | ✓ | Since 2026-10-15, `{stem}_run_manifest.json` beside every run's outputs: from `run_pipeline_stages` (the panel, the carotid example) and from the two examples that call the stages themselves, through `record_run` (`test_pipeline_provenance.py:343`). It holds the settings after `Schema.upgrade`, as the run started (`:150`), the version (`:92`), the commit and dirty paths (`:519`; null for a wheel or with no git, `:476-552`) and each input's sha256 (`:117`, `:328`; sampled above 64 MiB, `:239`), with `{stem}_run_settings.yaml`, which loads back to the same settings (`:169`, `:334`). A failed run writes one too (`:394`) |
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
| 2026-10-12 | after `43dcf45` | Claude | `graph.6c` evidence: the missing-vessel count is explained (mask components; on E14.5 0.36% of the mask), and both missing-vessel log lines now give the volume share. The coverage gap (77.7% against 47.1%) is still unexplained, so 6c stays ✗. No score changes. |
| 2026-10-12 | `b6d9210` | Claude | Dead-end work (`PLAN_dead_ends.md`). `export.6d` passes. Evidence updated for `graph.6b` (dead-end fit saved), `graph.6d` and `robust.6b` (face erosion fixed; bridge crashes fixed), `flow.6c`, `flow.8b`, `export.4b` and `export.6c`; each still fails. No score changes: exports stay 3 on `export.4b`. |
| 2026-10-12 | after `f9cf1d5` | Claude | `speed.4b` evidence: mask recovery profiled on a crop of the 16 G-voxel stack; its cross-section outline now stops each ray at its first voxel out and reads the mask as bits, in worker processes on large stacks. E14.5 build 280 -> 226 s under load, recovery 164 -> 113 s, graph identical. No score changes. |
| 2026-10-12 | `0c6d13b` | Claude | `export.4b` evidence: vessels behind dead ends (`blind_end`, `closed_end`) now carry flows and pressure drops of exactly 0, not the solve's rounding, in the solve, perturbations and sweeps (`test_open_ends.py:244-299`). Still fails until Harvey decides. No score changes. |
| 2026-10-13 | after `5ef7d53` | Claude | `flow.4b` passes: the Pries in vitro law is Pries 1992's, every Pries form is pinned to its published equations, and Pries & Secomb 2005 is a choice (`pries_viscosity_form`). This was Harvey's decision, option A of `PLAN_dale_review_fixes.md` item 1. Haemodynamics goes from 3 to 4, the mean from 3.8 to 3.9. `flow.6a` evidence updated; it still fails. Plasma-column resistances from earlier runs are 2.9–3.8× too high at 5–10 µm. |
| 2026-10-14 | `c62441a` | Claude | Bridging work (`PLAN_bridging.md`, `260d36a..c62441a` after `cf529d8..260d36a`). Evidence updated for `graph.6b` (grid R marked by Harvey, 13/13 at the all-one-way baseline), `graph.6d` and `robust.6b` (bugs found and fixed), `robust.4a` (shape errors in skeleton cleaning and graph building). No score changes: `graph.6b` and `graph.6d` still fail on the other grids' fits and the prune's protrusion test, `robust.6b` on the ilastik label guess. |
| 2026-10-15 | after `10c6923` | Claude | Short-loop hole rule (`short_loop_keep_hole_over_um`, 10 um): evidence added to `graph.6b` (grids Q and Q2 marked and fitted) and `graph.8b` (Q's rule on fresh grid Q2: 23/30, equal to all one way). Both still fail. No score changes. |
| 2026-10-15 | after `f397f62` | Claude | Dense solver memory warning, at Harvey's request: the run warns from `build_network` and `solve` when the dense flow solve (42 N² bytes, measured) or the eigendecomposition (49 N²) would take more than half the free memory. Evidence added to `memory.6c` (the measured peaks; the help text's 3.2 GB at 20k was 5x low, corrected) and `memory.8b` (the warning, tested). Both still fail: the defaults are unchanged, the warning is not in preflight, and the thick-vessel memmap case is not warned. No score changes. |
| 2026-10-15 | after `d7b7f09` | Claude | Run manifest (`pipeline/provenance.py`): every run writes `{stem}_run_manifest.json` and `{stem}_run_settings.yaml` beside its outputs, and each stage's wall time reaches the log. `speed.4a`, `repro.6a` and `export.6c` pass. `memory.6a` evidence updated; it fails until the reference datasets are measured. Speed 3 → 4, reproducibility 4 → 5, mean 3.9 → 4.0; the headline stays 3. `export.6c` is read as the output folder carrying the commit and settings, not each file; exports stay 3 on `export.4b`. |
| 2026-10-15 | `9b679b8` | Claude | Reference runs for `memory.6a` and `speed.6a`: the three fixtures, E14.5 and Nerve, each run once from a `git archive` of `9b679b8`, with a per-stage table and the manifest for each in `outputs/release_review/2026-10-15_9b679b8/`. Evidence added to both rows. Mason SM and the brain were not run because their inputs are not on this machine, so both checks still fail. Times are not compared with `speed.6b`'s proposed targets, which Harvey has yet to confirm. No score changes. |
| 2026-10-15 | `30fd455` | Claude | `stats.8a` passes. The statistics are cross-checked against skan, maskel and VesselVio on the dense capillary bed and on E14.5 (`scripts/stats_cross_check.py`, `tests/test_stats_cross_check.py`). VesselVio is GPL, so it is run as its own program and only its output files are read. Where the definitions coincide the numbers agree within 2.5%. Each other difference is pinned to the HaemoLynx step that makes it. The E14.5 run also measured that 27% of HaemoLynx's cleaned skeleton lies outside every vessel it draws, which belongs to `graph.6c`. No score change: statistics stays 5 on `stats.6a` and `stats.6b`. |
| 2026-10-18 | after `6f05617` | Claude | Low-RAM graph building and diameters (`use_memmap_loading`): one `MaskSupport` a run (one packing of the mask, one surface tree), packed point lookups, the thick tree's inscribed radius measured once and block by block, mask recovery tracing from a piece's own voxels, and unreleased temp memmaps deleted once dropped. E14.5 low-RAM, three runs each from `git archive`s (`outputs/lrb/`, git-ignored): graphs identical before and after diameters; `build_network` 225 -> 233 s median, within the machine's noise; thick split 14.4 -> 9.9 s. On a 210 M-voxel synthetic stack the tree radius peak went 425 -> 72 MB and a long recovered piece 1,419 -> 1 MB. Evidence for `memory.4a` and `speed.4b`; no check changes, no score change. |
