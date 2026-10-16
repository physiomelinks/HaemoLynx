# Plan: no doubled centreline, no internal dead ends, fewer small loops

Written 2026-10-12 against `Devel_GUI_improvements_HD` at **c65749b** (the collapse fix: "with the
fix" in the table below). Every number is from the E14.5 MCA stack, measured with
`scripts/graph_artefacts.py` (`outputs/missed_branches*/`) and a per-item census of its saved
graphs (`outputs/missed_branches_fix/working_tree/{before_thick_split,final}.pkl`, this plan's
section 1.3). It continues `PLAN_one_centreline_per_lumen.md` (phases 0–7 done or closed) and
`PLAN_dead_ends.md` (one definition landed; repairs written but off), and it does not repeat what
they already measured.

| | HEAD (e158d7a) | with the fix (c65749b) |
|---|---:|---:|
| missed skeleton pieces ≥ 20 voxels | 149 | 133 |
| pairs of edges in one lumen | 17 | 10 |
| doubled centreline | 2,073 µm | 1,805 µm |
| dead ends (interior) | 619 | 591 |
| short loops (≤ 200 µm round) | 251 | 242 |
| loops inside one lumen | 7 | 7 |
| mask covered | 77.2 % | 76.6 % |

The three goals, as this plan reads them:

1. **No doubled centreline**: no stretch of centreline running beside another through one
   segmented lumen, by a measure calibrated on your marks (today's measure counts things that are
   not doubling: section 1.3).
2. **No internal dead ends made by graph building**, and every dead end that is left named: the
   vessel leaves the image, the segmentation stops (a real blind end, E14.5's sprouts), or the
   review list says what to do with it. A real blind end is biology, and D2 of the One centreline
   plan keeps it; whether the *solved network* should drop them anyway is decision D1 below.
3. **Fewer small loops**: zero loops inside one lumen (an invariant the thick split now breaks),
   and the false loops among the rest found by a measure that uses the raw image, which no grid
   has tried yet. The loops that are left are then round tissue, and stay.

---

## 1. Where the artefacts are now

### 1.1 Where they are made (step by step, HEAD and c65749b agree to within a few)

| after step | doubled µm | pairs | short loops | in-lumen loops | dead ends |
|---|---:|---:|---:|---:|---:|
| raw skeleton graph (`build_graph_segment…`) | 3,100 | 540 | 2,438 | 1,277 | 6,345 |
| `collapse_node_clusters` | **5,585** | 391 | 734 | 372 | 656 |
| `prune_vascular_stubs` | 4,813 | 316 | 669 | 332 | 501 |
| `recover_uncovered_mask_vessels` | 5,763 | 499 | 922 | 338 | **1,583** |
| `join_facing_dead_ends` | 5,772 | 500 | 925 | 340 | 1,574 |
| `consolidate_lumen` | 3,281 | 16 | 264 | 0 | 689 |
| smoothing + clean-up again | 1,910 | 6 | 236 | 0 | 765 (today's face test) |
| **+ thick-vessel split (saved)** | 1,805 | 10 | 242 | **7** | 713 (today's face test) |

Where the saved graph's dead ends were first made (fix5 build, same code paths): the skeleton
itself 740, cluster collapse 70, **mask recovery 172**, **consolidate_lumen 79**, after the steps
10. So roughly a third of the interior dead ends are made by graph building, not by the skeleton.

Three things the table says:

- **Cluster collapse nearly doubles the doubled centreline** (3,100 → 5,585 µm): merging a
  cluster of close junctions into one node leaves the edges that left those junctions leaving one
  point side by side. 377 of the 391 pairs after it are forks. `consolidate_lumen` then takes the
  pairs out (500 → 16) by **deleting whole edges**, which is what grids D–F were about (56 % of
  the junction-to-junction deletions were vessel) and what makes its 79 dead ends.
- **Mask recovery makes 172 dead ends and adds doubled centreline** (4,813 → 5,763 µm) that the
  clean-up has to take out again.
- **The thick-vessel split is the only step after the last clean-up**, and it makes all 7 loops
  inside one lumen, 4 of the 10 pairs and 33 doubled runs (67 µm): nothing judges the graph after
  it.

### 1.2 Dead ends by kind (saved graph, `graph.dead_ends`, today's `src`)

| kind | n | what it is | chain length p50 / p90 |
|---|---:|---|---|
| `leaves_image` | 114 | open ends | — |
| `island_end` | 232 | ends of components with no junction (left to `remove_disconnected`, D3) | — |
| `stops_short` | 111 | **the mask runs on past the tip**: a vessel the graph lost | 17.5 / 46 µm |
| `broken` | 49 | another vessel straight ahead across one stretch of background, or a lumen it ends in, 2+ junctions away | 12.9 / 30 µm |
| `artefact` | 31 | off the mask, in another lumen, beside a vessel, under the radius rule | 11.4 / 17 µm |
| `unknown` | **522** | interior, none of the above | 15.4 / 39 µm |

219 interior dead ends have another centreline within 10 µm straight ahead (45 within 5 µm).
Your G1 marks (29 judged): 15 not a vessel, 14 real blind ends, and across G1–G3 54 of 87
notes say "not a real vessel"; no mask measure separated them (`FIT.md`), and the one raw
measure tried (`raw_tip_over_chain`) scored 16/29.

The harness's REPORT at c65749b gives 591 interior dead ends; the same `final.pkl` re-measured
with today's `src` gives 713 interior and 114 at a face. The dead-end merge (e158d7a) moved the
face test, so ~120 ends changed side. **One ruler per comparison**: every number in this plan's
acceptance tables is re-measured at the commit it is scored at, as the release review requires.

### 1.3 Doubled centreline, run by run (saved graph; `census.py`, 2026-10-12)

The 1,805 µm is **1,045 separate runs**, judged sample by sample (`shadowed_samples`: a sample
further than lumen radius + 1 µm from both ends of its edge, whose chord to the nearest other
centreline is within 2r + 1 µm, crosses under 1 µm of background, and sits at more than 45° to
its own tangent).

| run length | runs | µm |
|---|---:|---:|
| all (median 1.5 µm, p90 4.4 µm) | 1,045 | 2,192 |
| ≥ 5 µm | 76 | 572 |
| ≥ 10 µm | 9 | 113 |

| where the run is | runs | µm |
|---|---:|---:|
| one edge is a **dead-end chain** (a hair or sprout along a vessel) | 552 | 1,136 |
| **mid-edge** stretch (two strands through one lumen, diverging at both ends) | 309 | 726 |
| **fork**: at the junction the two edges share | 139 | 236 |
| one edge is a thick-vessel bridge | 33 | 67 |
| at an edge end with no shared node | 12 | 28 |
| in one of the 10 reported pairs | 14 | 51 |

So:

- **The pair rule (≥ half an edge beside another) no longer finds the doubling.** 51 of 2,192 µm
  is in pairs. Deleting edges cannot drive this to zero; the target needs stretches resolved.
- **Most runs are under 2 µm and sit where vessels meet** — the flare of a junction or the first
  stretch of a sprout leaving its parent at a shallow angle, past the r + 1 µm end margin but
  still inside the parent's lumen bulge. That is not one vessel drawn twice. Before "no doubled
  centreline" can be a target, **the measure has to say what you mean by doubled** (Phase 0's
  grid). The 76 runs of 5 µm and more (572 µm) are the ones that look like second strands.
- **112 of the 242 short loops have a side with a doubled run ≥ 2 µm, and 72 are triangles**
  (three edges or fewer): the false loops your batch-C drawings showed (a side closing the wedge
  between two branches at a fork; both sides of a slit in one flat vessel) are the doubled runs
  seen from the other side. Resolving the runs removes those loops without a loop rule.

---

## 2. Why the previous phases could not finish it

1. **Delete or keep was the wrong question.** Phase 1 of the One centreline plan resolves a
   pair by deleting one edge whole; grids D–F showed the deleted edge's undoubled remainder is
   vessel 56 % of the time, and no measure separated them. The parked Phase 2 branch
   (`claude/phase2-doubled-stretches`, 5deb1fd) had the right operations (trim, zip, rejoin) but
   combined them with a *cautious* pair rule that kept pairs Phase 1 had deleted, and set a
   minimum doubled run of max(3 µm, 2r), which excluded most forks. The missed-branches session's
   trim-and-rejoin made triangle loops (242 → 422) because a rejoin was allowed to close a loop
   inside one lumen. Neither failure is the operation; both are the rule round it.
2. **Nothing judges the graph after the thick split**, so its bridges reintroduce every artefact
   the clean-up removed.
3. **Every loop measure tried was a mask measure**, and 120 marks showed none generalises. The
   raw image has not been used for loops at all, though D6 of the dead-ends plan adopted it as
   evidence and the dextran channel is black outside vessels (a hole in the mask that is bright
   in the raw image is a pinhole in the segmentation, not tissue).
4. **The prune's hair rule cannot tell a hair along a wall from a vessel's collinear end**
   (`graph.6d`, open since 2026-10-11): it judges protrusion, not "beside".
5. **The dead-end kinds are labels, not repairs.** The repairs exist
   (`graph/dead_end_extension.py`: trace on, extend to a face, join to a side) and are off
   because no rule reached 90 % on a fresh grid. `stops_short` is different from the others: the
   mask running on past the tip is a fact, not a judgement, so tracing on needs a grid of *where
   the trace ends*, not of whether to trace.

---

## 3. Principles (unchanged from the two plans, plus two)

- Measure before changing; one ruler per comparison; fixtures pin every case a grid reveals.
- A rule that decides what a structure *is* follows your marks (CLAUDE.md, marked grids), and
  ships only at ≥ 90 % on a fresh grid; below that it ranks a review list.
- Delete only what is not vessel. **New: where a stretch is doubled, rewire — move the junction
  to the parting point, or attach the remainder there — rather than delete, so the question
  "is the remainder vessel" never has to be answered.** A rewire that would close a loop inside
  one lumen is refused, and that case falls back to today's deletion.
- **New: nothing is the last step.** Whatever edits the graph after `consolidate_lumen` (the
  thick split, hand edits, a future repair) is followed by the clean-up again, with the
  invariants (0 pairs, 0 loops inside one lumen, 0 tips in another lumen) asserted on the graph
  that is saved.
- Peers' files are edited by targeted replacements only, in a worktree, rebased and re-measured
  before landing (memory: shared checkout).

---

## 4. Phases

Each phase is measured on the harness before it lands, its tests live in the files CLAUDE.md
names, and its row in `RELEASE_REVIEW.md` is updated in the same commit.

### Phase 0 — Say what "doubled" means, and make the census a column

**Status 2026-10-12:** grid H built (`scripts/doubled_run_review/build_grid.py`, 10 runs each from
a dead-end chain, mid-edge and a fork, 1.5–17 µm, shuffled) and published as the private page
"Doubled Run Review H" (https://claude.ai/artifact/AB8uy9XjjhvfkP6NyUrDJX; key and all 1,045 runs
in `outputs/doubled_run_review/`). The census is in `diagnose_lumen_artefacts`
(`doubled_runs`, `doubled_runs_by_place`, `doubled_long_runs`, `short_loops_by_side`) and in the
harness's columns, with fixtures for each place (`hair_along_a_wall` added).

**Grid H marked (Harvey, 2026-10-12; read back from the review page, `marks_H.txt`,
drawings in `marks_H_drawings.json`): 19 D, 11 T, no rule.** Of the key's measures the best
scores 19/30 leave-one-out, the all-D baseline; of eleven more computed from the graph and mask
(angle between the tangents, chord gap, gap over the radii, a neck across the chord, radius
ratio, hops between the edges, distance from the junction, share of the shorter edge) the best,
the neck, scores 21/30 (`FIT.md`). Length is not the criterion: four of the six shortest runs
(1.5 µm) are D and the 17 µm run is T; D/T is 6/4, 6/4, 7/3 by place. The drawings show one
traced line for most D and two diverging lines for T: the question is whether the two
centrelines are one vessel, which the mask does not say where it is one lumen either way. So
there is no calibrated minimum run; decision D6 below decides what the stretch step resolves.

- **Grid H (doubled runs).** 30 runs from the saved graph, stratified by run length over the
  local lumen radius (under 0.5 r, 0.5–1 r, 1–2 r, over 2 r) and by place (dead-end chain,
  mid-edge, fork), shuffled, each drawn in its own plane on the segmentation with both
  centrelines, the run highlighted, a scale bar and its voxel position. Marks: **D** = one
  vessel drawn twice (resolve), **T** = two vessels or a branch leaving (leave). Key under
  `outputs/doubled_run_review/`. Fit a minimum run length (absolute, or in radii) and, if the
  marks ask for it, a tangent test (two parallel strands against a branch leaving at an angle).
  Fresh grid H2 before relying on it.
- **The census as code**: `diagnose_lumen_artefacts` reports doubled runs by place and length
  (what `census.py` did by hand), loops by what they run through (a doubled side, a bridge,
  inside one lumen, a triangle), and dead ends by kind and by the step that made them
  (`dead_end_census.made_by_step`, already there). The harness prints them; the Nerve fixture
  and `tests/lumen_artefact_fixtures.py` get one synthetic case per place.
- **Baseline at one commit**, graph as saved, with the split: the acceptance table of section 6
  filled in.

Size S. No behaviour change.

### Phase 1 — Doubled centreline to zero (by Phase 0's measure)

Run inside every `consolidate_lumen` round, after the pair step, on the runs the calibrated
measure counts. The operations are those of the parked branch, with the rules changed:

- **1a. Forks: slide the junction.** Two edges leaving one node side by side: the shared
  stretch becomes one edge (the path nearer the lumen's middle) from the junction to the parting
  point, where a new junction takes both continuations. No deletion, no connector, no loop, no
  dead end. On E14.5 this is where collapse makes its 2,500 µm.
- **1b. Dead-end chains: trim.** A dead-end chain beside another centreline from its junction
  outward is cut back to where it stops being beside; what is left is judged by the prune as
  usual (a hair along a wall becomes nothing; a sprout that runs along its parent and then leaves
  keeps the part that leaves). This replaces the prune's protrusion test (`graph.6d`), which
  stays only as a fallback where no mask is given. A collinear vessel end past a junction is not
  "beside" (its chord runs along its tangent), so the face-end case that broke the 16-sample fix
  cannot recur; a test pins it.
- **1c. Mid-edge stretches: zip both ends.** Two strands through one lumen that diverge at both
  ends: one shared edge between two new junctions, each continuation attached at its parting
  point. Where the stretch runs to an edge's end vertex (the Phase 2 attempt's blocker), that end
  is the parting point.
- **1d. Pairs with a remainder: attach, do not delete.** Where today's pair step would delete an
  edge whose undoubled remainder is more than Phase 0's minimum run, cut it at the parting point
  and attach the remainder to the kept strand there (`graph.edit.insert_node_on_edge`), unless
  that closes a loop inside one lumen (`lumen_loops.loop_inside_one_lumen`), when it is deleted
  as today. Grids D–F are then moot: the remainder stays whatever it is, and it is a dead end only
  if it already was one.
- **1e. The thick split is followed by the clean-up.** `build_network` runs `consolidate_lumen`
  once more after `insert_thick_vessel_junction_nodes`, with bridges exempt from deletion and the
  pair and loop tests reading a bridge as the thick centreline it stands for; the split itself
  refuses a bridge that would run beside an existing bridge (ridge-first already aims at the
  ridge; this is the check on the result). Invariants asserted on the saved graph.
- **1e done (2026-10-12, branch `claude/clean-graph`).** The split's flicker is judged in
  microns (a 1.5 µm dip out of the region, sampled every 0.49 µm, was four samples and came back
  as a vessel between two bridges: three of the seven loops); a degree-2 node beside a bridge is
  never merged; a pair with a bridge is left to the loop step; a bridge left dangling goes
  (`thick_vessel_junctions.remove_dangling_bridges`, in every `consolidate_lumen` round); and
  `build_network` and the harness run `consolidate_lumen` once more after the split. E14.5
  against the Phase 0 baseline (`outputs/clean_graph_phase0` → `clean_graph_phase1e`): loops
  inside one lumen 9 → 0, pairs 11 → 7 (with a bridge 3 → 0), short loops 244 → 231 (through
  a bridge 52 → 41), bridges 138 → 113, dead ends 713 → 712, components 157 → 157, centreline
  −378 µm, mask covered 76.5 → 76.0 % (at the limit: the strands removed ran along the fat
  vessels' walls), build 253 s.
- **1a–1d first measurement (2026-10-12, `graph/doubled_stretches.py`, the parked branch's
  trim / zip / rejoin with today's pair rule kept, bridges never touched, run in every
  clean-up round after the loop step; minimum run 3 µm and two radii until D6):** E14.5
  against Phase 1e (`outputs/clean_graph_phase1_stretch`): doubled 1,782 → 1,611 µm, runs ≥ 5 µm
  76 → 67, on dead-end chains 1,133 → 1,008, mid-edge 730 → 651, dead ends 712 → 704, pairs 7 → 7,
  short loops 231 → 235, mask covered 76.0 → 76.4 %, components 157 → 157, build 327 s. Only a
  tenth of the doubling goes, because the minimum run leaves the short runs, which the marks
  say are doubled too.
- **Every run resolved (D6 a, 2026-10-12, `outputs/clean_graph_phase1_all_runs`)** against
  the 3 µm build: doubled 1,611 → 1,124 µm, pairs 7 → 2, forks 230 → 71 µm, dead ends 704 → 654,
  tips in another lumen 7 → 4, short loops 235 → 238, components 157 → 156, mask covered
  76.4 → 75.4 %, centreline −1.4 mm, build 421 s. Of the 906 µm of runs left, 588 µm are seen
  only from the wider edge (a sample is judged with its own radius, so a narrow centreline 4.5 µm
  from a 6 µm vessel's is not "beside" it from its own side), 198 µm are zips and 234 µm rejoins
  refused by a chord-versus-run-length guard.
- **Symmetric finder tried and reverted (`outputs/clean_graph_phase1_sym`):** judging each
  sample with the wider of the two radii resolved the wide-side runs (doubled 1,124 → 694 µm,
  dead ends 654 → 522, short loops 238 → 171) but made every capillary fused along a trunk's
  wall "beside" the trunk from its own side, and the trim took it whole: centreline −5.7 mm
  (−12 %), mask covered 75.4 → 71.7 %, build 672 s. Fails the kill criteria. The chains it would
  trim are **grid H2** (`scripts/doubled_run_review/build_grid_h2.py`: 30 of them, mask and raw
  dextran panels, V keep / X remove): whether a dead-end chain along a wider vessel is a vessel
  is a marked question, not a finder's. The chord guards stay relaxed (measured alone in
  `outputs/clean_graph_phase1_relaxed`).
- **Relaxed chord guards, every run, own-radius finder (`outputs/clean_graph_phase1_relaxed`,
  the state committed):** against the Phase 0 baseline, doubled 1,799 → 887 µm (runs ≥ 5 µm
  75 → 38), short loops 244 → 199 (with a doubled side 111 → 60), dead ends 713 → 611, loops
  inside one lumen 9 → 0, pairs 11 → 2, tips in another lumen 9 → 7, components 157 → 155;
  costs: centreline 49.5 → 46.0 mm (−3.5 mm: the doubling removed plus the stubs the prune
  takes once a hair is cut back), mask covered 76.5 → 74.8 %, build 253 → 538 s. Both the
  coverage and the centreline criteria are exceeded; D6 (a) chose the doubling over them, and
  the 3 µm build (`clean_graph_phase1_stretch`: doubled 1,611, covered 76.4 %, 327 s) is the
  fallback if Harvey wants the coverage back. Build time is an open item: the stretch finder
  runs over the whole graph every round.
- **Grid H2 marked (Harvey, 2026-10-12): 16 V, 14 X, no rule** (`FIT.md`). Half of the chains
  the wider test would trim are vessels, and no measure (length, radii, share beside the trunk,
  mask ahead, depth) separates them: the best scores 16/30, the all-V baseline. So the finder
  keeps the report's own-radius test, and the doubling only the wider side sees (588 µm on
  E14.5) is **not resolvable by rule**: it goes to the "two vessels in one lumen" review list.
  The target "doubled centreline 0" is therefore "0 by rule where a rule holds, the rest
  listed"; section 6 is updated. **The raw image does not help either** (the raw-confirmation
  session, 2026-10-12, `outputs/doubled_run_review/FIT_raw.md`): a dip in the dextran channel
  between the two centrelines scores 17/30 on H (all-D 19) and 17/30 on H2 (all-V 16); the
  wall between touching capillaries is under a voxel and the PSF has removed it. So the
  doubled-run question stays with the mask rules, and the `keep_apart` hook stays unfilled.
- **Kill criteria, measured per sub-step on the harness:** short loops may not rise (the two
  failed attempts both raised them); components, interior dead ends and build time (+10 %) may
  not rise; centreline length may fall only by the doubled length removed; mask coverage may not
  fall by more than 0.5 points. A sub-step that fails stays off and gets a grid.

Size M–L. Tests: `test_consolidate_lumen.py`, `test_lumen_loops.py`, a new
`test_doubled_stretches.py` with one fixture per operation (fork, hair, slit strand, pair with a
remainder, the loop-closing rejoin refused, the collinear face end kept), and the Nerve
invariants. Expected on E14.5: doubled ≈ 0 by the calibrated measure; pairs 0; loops inside one
lumen 0; of the 112 loops with a doubled side, the false ones gone; `consolidate_lumen`'s 79
dead ends mostly not made.

### Phase 2 — Fewer small loops

- **2a. Zero loops inside one lumen on the saved graph**: Phase 1e. (Today 7, all made by the
  split.)
- **2b. False loops with a non-vessel side** go with Phase 1 (forks, slits, hairs). Measured by
  the census: loops with a doubled side, triangles.
- **2c. The raw-image hole test — grid I.** For each remaining short loop: the raw intensity
  inside the loop's hole (the in-plane gap the loop runs round, read from `fwhm_raw_tiff_path`,
  the dextran channel) against the loop's own vessels and the image background. Bright hole = the
  segmentation missed lumen there (a pinhole: a false loop); dark hole = tissue (real). Grid of
  30, stratified by that ratio, shuffled, drawn on raw *and* mask side by side (the first loop
  grid with the raw image; batches A–C and F were mask only). Fit; fresh grid I2; ship as the
  loop rule (remove the side nearest the wall, as the in-lumen rule does) at ≥ 90 %, else it
  ranks the Short loops review list and the Post processing page shows the raw panel.
- **2d. Pinholes plugged before skeletonisation** (only if 2c's rule holds): in
  `segmentation_cleanup`, fill a tunnel of the mask whose interior is vessel-bright in the raw
  image, so Lee thinning never rings it. The 1-voxel closing that removed 60 % of tunnels merged
  235 components; a plug keyed on the raw image fills only holes the raw image says are lumen.
  Measured by the raw skeleton graph's 2,438 short loops and the final count; mask components
  may not merge.

Size M. Tests: `test_lumen_loops.py` (hole measure on a synthetic pinhole and a synthetic tissue
hole), `test_segmentation_cleanup.py` (a plug fills a bright tunnel and leaves a dark one). The
loops that are left after 2a–2d are round tissue by every measure we have, and the target is
"all that the calibrated rule finds", not a number chosen in advance.

### Phase 3 — No internal dead ends made by graph building; the rest named

- **3a. Stop making them.** Phase 1b/1d remove `consolidate_lumen`'s cause. Mask recovery's 172:
  the census says what kind each is; a recovered strand's free end that is `artefact` (beside a
  vessel: a far-rim strand) is trimmed by 1b, one that is `stops_short` is traced on by 3b, and
  recovery's own join step gets the same attach-at-parting-point rule as 1d where its strand runs
  into a covered lumen rather than ending free. Collapse's 70 are re-measured after c65749b (the
  bounded collapse) and the remainder classified.
- **3b. `stops_short` (111): trace on — grid J.** Turn on `trace_on_dead_ends` in a measured run:
  from each tip follow the mask along its heading until it meets a centreline (join), a face
  (`leaves_image`), another tip (join) or the mask's end (a blind end at its true end), with the
  guards already written (mask support, not beside a vessel in the same lumen, no loop inside one
  lumen, turn limit). The grid is of the **traces**: 30 traced ends, each drawn with the trace,
  marked right / wrong / should have stopped at …; ship on by default at ≥ 90 % on a fresh grid.
  What the trace cannot extend keeps the kind and the review button.
- **3c. `broken` (49) and facing pairs: grid K.** G3's turn rule (`ahead_turn_deg ≤ 30.5°`)
  scored 24/27 fitted; a fresh grid of 30 tip-to-side and facing candidates decides whether
  `join_dead_ends_to_vessel_sides` goes on by default. Below 90 % it stays the review list's
  ranking, as now.
- **3d. `artefact` (31)**: Phase 1b's trim, plus the existing rules; target 0 on the saved graph.
- **3e. `unknown` (522): the first grid with the raw image — grid L (D6, one channel: D5).**
  Your notes say about half are not real vessels, and nothing in the mask separates them. Draw
  30 (stratified by chain length and by what is ahead), each as two panels: mask with
  centrelines, and the raw dextran channel (C1). Marks: **S** real blind end (a lumenised
  sprout), **X** not a vessel, **B** broken (continues). Candidate measures, all new: the raw
  core brightness along the chain's last 5 µm against its parent's; whether the raw signal ends
  where the centreline does or runs on; the raw profile across the tip (a lumen or speckle).
  Fit; fresh grid L2. Whatever reaches 90 % becomes
  the prune's rule for X and the join's for B; what does not is `blind` by default, labelled,
  counted (`statistics/dead_ends.py`) and zero-flow (`FLOW_STATE = blind_end`).
- **3f. The solved network**: with every artefact gone, "no internal dead ends" in the flow is
  `boundary_handling = remove_disconnected`, which already deletes every dead-end branch; if you
  want statistics without blind ends too, a graph-stage setting `remove_blind_ends` (advanced,
  off) prunes every interior dead end whatever its length — decision D1.

Size M–L (3e is mostly grids). Tests: `test_dead_end_extension.py`, `test_dead_ends.py`,
`test_mask_recovery.py`, `test_prune.py`.

### Phase 4 — Validation, targets, docs

- The Nerve fixture and the skeletal-muscle crop (`BigSMconfig.yaml`) as the second real dataset
  for `graph.8c`; the targets of section 6 written into `RELEASE_REVIEW.md` with their evidence;
  `graph.6b` / `graph.8b` rows updated with each grid's fit and fresh-grid score; `graph.6d`
  closed by 1b.
- CLAUDE.md: the consolidation-after-the-split rule, the doubled-run measure, the new settings.
- Build time ≤ +10 % of c65749b (288 s on the harness): the run-finding reads the pair memo's
  sample index, not a new tree per round.

---

## 5. Decisions

**Landed 2026-10-12:** Harvey chose to keep every run resolved; `claude/clean-graph` was merged
with the shared branch (6df1c35) and pushed to `origin/Devel_GUI_improvements_HD`, the local
branch fast-forwarded. Next: build time (the clean-up went 44 → 177 s and the two clean-ups
after the steps 47 → 205 s; `consolidate_lumen` with the stretch step is ~290 s of 538).

**Build time (2026-10-12 evening):** profiled on E14.5's pre-clean-up graph, the stretch step's
first round was 70 s, 59 s of it whole-graph length sums for the log line (now kept in the
editor); the finder itself ~7–11 s a round. The clean-up also ran to its 12-round limit every
time: eleven forks zipped 4 µm one way and then the other (the two-cycle: zip, degree-2 merge,
zip back). Three guards end it (single-sample runs skipped; a fork zipped only where its edges
part at a similar distance; a memo of zipped junctions across rounds), and E14.5 settles in 11
rounds. Cost: doubled 887 → 1,010 µm, loops 199 → 222, dead ends 611 → 646; coverage
74.8 → 75.4 %. Timings could not be compared (loaded machine). **Incremental finder written
(uncommitted, tests pass):** `doubled_stretches(memo=)` keeps each edge's dense path, reach
and runs by the pair memo's signature; an edge is judged again only when it, or a centreline
within its reach plus a step, was added, changed or removed since the last call. **Measured** on
E14.5's pre-clean-up graph, same machine, same final graph (2,591 nodes / 2,745 edges): the
clean-up 305 → 218 s; the stretch step 126 → 50 s (16 s in round one, then 11, 7, 4.5, 2.5 … 1 s);
the prune 79 → 68 s with its centreline index built once per call and trimmed as stubs go
rather than rebuilt (radii read again) every pass. Landed as 774fd57. End-to-end E14.5 with it
(`outputs/clean_graph_phase1_fast`): the same graph as `clean_graph_phase1_settle` node for
node; timings again from a loaded machine (recovery 225 s against ~100 quiet), the clean-up
0.9× recovery's time against 1.3× before. **Then:** the prune's index kept between rounds
(`prune_vascular_stubs(index_memo=)`, each unchanged edge's points and radii reused), measured
next: prune 68 → 60 s (6de3300), and the pair and stretch memos kept on the `LumenCleanup`
across its calls (ea4d949), so the post-split clean-up reuses the post-smoothing verdicts:
end to end (`outputs/clean_graph_phase1_fast2`, same graph) the two clean-ups after the
steps 193 → 150 s under the same load. **Relative to mask recovery, untouched:** the clean-up
1.79× → 0.85×, the after-step work 2.07× → 0.66× (before the stretch step existed, 0.42× and
0.45×). On a quiet machine that is roughly 350 s for the build against 538 s before the
speed-ups and 253 s before the stretch step. Left in the clean-up: the prune's index
build per call (~4–7 s a round), the pair step's first two rounds (23 + 10 s), loops (26 s).

**Thick-vessel region cached (2026-10-13, bcc4625):** the region skeletonisation finds (hole
fill, radius map, body, wall) is kept beside the saved skeleton, one bit a voxel (9 MB on
E14.5), keyed on a digest of the mask's voxels and the settings that decide it
(`preprocessing/thick_vessel_region_cache.py`, `stages.thick_vessel_region_cache`); the
skeletonise stage, the loaded-skeleton path (which used to get no region and make no bridges),
`post_process` on a resumed run and the harness all read it. The "~4 min" figure given for it
earlier was from a loaded machine and wrong: on E14.5 the catchment is 19 s inside
skeletonisation (39 s through `thick_vessel_mask_from_image`, hole fill included), and the
skeletonise stage went 171 → 142 s with the cache, same skeleton voxel for voxel. In a cold
process the read costs 17 s, all of it loading the large-vessel restriction masks for the key,
which the pipeline loads anyway.

**Decided 2026-10-12 (Harvey: "go with the recommendations"):** D1 (a) — graph building makes
no dead ends and names the rest; real blind ends stay, `remove_disconnected` drops them from the
solved network, and `remove_blind_ends` is an optional setting. D2 yes — rewire without a grid,
judged by the kill criteria. D3 grid order H, L, I, then J/K. D4 — pinhole plugging only after
2c holds on a fresh grid, as an option off by default. Work is on branch `claude/clean-graph`
(worktree `.claude/worktrees/clean-graph`, off c65749b).

| # | Question | Recommendation |
|---|---|---|
| D1 | **What "no internal dead ends" means for real blind ends.** (a) None made by graph building, the rest named and kept (E14.5's sprouts are biology; G1 marked 14 of 29 real); the solved network drops them with `remove_disconnected`. (b) None in the graph at all: a `remove_blind_ends` setting, off by default, that prunes every interior dead end. | (a), with (b) as the optional setting for runs where sprouts are noise. The plan is written for (a). |
| D2 | **Rewire without a grid.** Phase 1 keeps every undoubled remainder and attaches it at the parting point, rather than asking whether it is vessel. That is the "never delete vessel" principle, and grids D–F said the question has no measurable answer. | Yes: measured by the kill criteria, not by marks; grid H calibrates only what counts as doubled. |
| D3 | **Grids with the raw image** (H for doubled runs is mask only; I for loops and L for dead ends use the raw dextran channel). Four grids plus their fresh checks is ~240 marks. | Order: H (defines the target), then L (the biggest count), then I, then J/K. |
| D5 | **One channel only** (Harvey, 2026-10-12): every rule and every grid reads the segmentation and the one raw channel it came from. No endothelial or second channel, anywhere. | Decided: the composite's claudin-5 channel is not used; 3e's measures are dextran only. |
| D6 | **What the stretch step resolves, given grid H found no rule.** (a) Every run the report counts (minimum run 0): 19 of 30 marked runs are one vessel drawn twice, and the rewire costs a T only the shared stretch (a branch's first few µm run on its parent's centreline, not deleted). (b) Only runs of 3 µm and two radii, as now: a tenth of the doubling goes, the rest stays. (c) A grid of the T drawings to find a measure the mask has not given. | **Decided (a)** (Harvey, 2026-10-12): resolve every run; measured on E14.5 (`outputs/clean_graph_phase1_all_runs`). The 11 T are the cost; the review list "two vessels in one lumen" stays for them. |
| D4 | **Pinhole plugging in the mask** (2d) changes the segmentation before skeletonisation, so every downstream number moves. | Only after 2c's rule holds on a fresh grid, and as a `segmentation_cleanup` option off by default until the Nerve and muscle datasets confirm it. |

---

## 6. Targets (E14.5, graph as saved, one commit, the harness)

| measure | now (c65749b) | target |
|---|---:|---:|
| doubled centreline, by the report's measure | 1,805 µm | **0 where a rule holds**: every run the narrower side sees is resolved; what only the wider side sees (~590 µm, half of it real capillaries along trunks by grid H2) is listed for review, not removed |
| doubled runs ≥ 5 µm | 76 (572 µm) | as above |
| pairs in one lumen | 10 | 0 |
| loops inside one lumen (after the split) | 7 | 0 |
| short loops with a side that is a doubled run / a triangle | 112 / 72 | only those the raw rule marks real |
| short loops the raw-hole rule removes | — | all it finds at ≥ 90 % |
| interior dead ends made by graph building (collapse + recovery + clean-up) | ~320 | 0, or each named `stops_short`/`broken` and listed |
| `artefact` dead ends | 31 | 0 |
| `stops_short` | 111 | ≤ 10 (the dead-ends plan's figure) |
| `unknown` | 522 | every one `blind`, `broken` or removed by a rule at ≥ 90 % on a fresh grid; the rest `blind`, counted |
| tips in another lumen | 9 | 0 |
| mask covered | 76.6 % | ≥ 76 % |
| components / isolated edges | 153 / 116 | no more |
| build time | 288 s | ≤ 317 s |

Order: 0 → 1 (1a, 1e, 1b, 1d, 1c) → 2a/2b (free) → 3a → grids L, I, J, K as the marks arrive
→ 3b–3e → 2c/2d → 4. Phase 1 needs only grid H; everything in Phase 3 beyond 3a waits for a
grid.

---

## 7. Risks

- **Rewiring is more code than deleting**, and the two earlier attempts regressed loops. Each
  operation lands alone, behind the kill criteria, with its fixture; the loop-closing refusal is
  the guard the failed attempts lacked.
- **A sliding junction moves node positions**, so boundary `node_ids` from an earlier build and
  the smoothing's `max_deviation` check (measured from the path the edge came from) must read
  the new paths; `thick_vessel_junctions` reads junction positions too. Rewires run before
  smoothing and before the split in every round, and once more after the split.
- **Tracing on can re-trace a far rim** (the One centreline plan's lesson): the shadow and
  in-lumen loop guards, and the doubled-run census, measure it.
- **The raw image is not always there**: 2c, 2d and 3e read it when the run has it and fall
  back to the mask rule (or the review list) when not; the Nerve fixture has none.
- **Shared files**: `assemble.py`, `lumen_loops.py`, `prune.py`, `mask_recovery.py`,
  `dead_ends.py`, `stages.py`, `schema.py`, CLAUDE.md. A worktree, rebase, re-measure.
- **The measure moves with the fix.** Phase 0 fixes the ruler first and the acceptance table is
  scored at one commit with a `git archive`, never the live tree.
