# Plan: one centreline per lumen — duplicate edges, erroneous stubs, Lee rings

Written 2026-10-08 against `Devel_GUI_improvements_HD` at **7c0630c**. Every number below
was measured on the E14.5 MCA stack. Sections 1-2 are the original audit; **the current
status of every phase is at the top of section 3** (updated 2026-10-10).

---

## 1. What was measured

**Setup.** The 7 Oct run's saved skeleton (`outputs/...MCAregion_Simple Segmentation_skeleton.npy`)
was rebuilt with `build_graph_from_skeleton` and the repo-root `config.yaml` graph settings
(reconnect 10 µm, orphan 3 µm, direction-aware collapse 5 µm, stubs 10 µm / 1.5 r, mask support
and recovery on), then `smooth_graph_centrelines`. That was done twice:

- at **f096035**, which reproduces the saved 7 Oct `graph.pkl` exactly (579 joins, 2,503 dead
  ends removed, 343 parallels removed, 27 off-mask edges; plus the thick split's 434 nodes);
- at **7c0630c**, this morning's recovery change.

Each graph was audited against the Dropbox mask (`tiff == 1`). The thick-vessel split is not in
the rebuilds; its effect was read off the saved graph instead.

### 1.1 Final graph after smoothing

| | f096035 | 7c0630c (HEAD) |
|---|---:|---:|
| edges / centreline length | 2,908 / 50.9 mm | 3,772 / 57.5 mm |
| mask covered (a centreline within radius + 1 µm) | 63.4 % | 66.1 % |
| **in-lumen duplicate pairs** (`diagnose_parallel_duplicates_in_lumen`) | 72 | **170** |
| centreline running beside another in the same lumen | 2.5 mm (4.9 %) | 3.7 mm (6.4 %) |
| …pairs where one edge is a dead end | 48 | 117 |
| …pairs sharing one node (a fork in one lumen) | 45 | 98 |
| **loops inside one lumen** (current rule) | 6 | 10 |
| **loops round a hole ≤ 3 µm** (rule says "tissue"; these are the leftover rings) | 52 | 57 |
| **interior dead ends** (> 10 µm from an image face) | 911 | 1,253 |
| …tip inside another vessel's lumen | 21 | 79 |
| …shorter than 3 radii of the vessel they leave | 147 | 278 |
| …mask runs on past the tip (a truncated vessel) | 115 | 142 |
| components / of them a single isolated edge | 201 / 137 | 238 / 157 |
| build + smoothing time | 118 s | 310 s |

The saved graph, after the thick split, has 89 pairs. 34 of them involve an `is_zero_resistance`
bridge, and 68 of its 284 short loops run through one.

### 1.2 Where they appear (7c0630c, step by step)

| after step | dup pairs | in-lumen loops | pinhole loops | interior dead ends |
|---|---:|---:|---:|---:|
| degree-2 pass 1 (raw skeleton graph) | 503 | 1,049 | 447 | 6,533 |
| collapse + degree-2 | 324 | 332 | 60 | 837 |
| first stub prune | 320 | 331 | 59 | 757 |
| **mask recovery** | **625** | 355 | **163** | **2,910** |
| loop breaker + parallel removal | 145 | 2 | 99 | 2,992 |
| final prune | 112 | 2 | 99 | 1,236 |
| **final degree-2 merge** | **175** | 3 | 71 | 1,246 |
| off-mask removal | 172 | 1 | 64 | 1,253 |
| **smoothing** | 170 | **10** | 57 | 1,253 |

Where the final graph's interior dead ends first became dead ends: 572 at graph build, 65 at
cluster collapse, **331 at mask recovery**, **106 at the loop/parallel step**, **160 at the final
prune**, 18 later. At f096035 the same steps gave 134, 70 and 27.

### 1.3 What the mask says

- **Most "uncovered" mask is the far side of a lumen that already has a centreline.** Of the 1.04 M
  mask voxels the HEAD graph leaves uncovered, 79 % have a straight line through the mask to
  their nearest centreline. They sit a median 5.6 µm from it, where that centreline's radius
  reads 2.8 µm: the lumen is flattened or the centreline is off-centre. 197 k voxels (19 %) could
  be missed vessels, and 21 k lie in mask islands with no graph at all.
- **Rings start as tunnels in the segmentation.** With its cavities filled the mask has 3,286
  tunnels (first Betti number; Lee thinning rings each one). A one-voxel closing removes 60 % of
  them, so most are pinholes under about 2 µm wide. That closing also merges 235 components and
  adds 72 k voxels, so a blanket closing is not the fix.
- **Duplicate strands are one lumen, not two touching vessels.** Along the chords between the two
  strands of each saved pair, the distance to background dips in the middle (a neck between two
  fused tubes) for only 2 of 86 pairs.
- Of the 200 small graph components, 181 lie in their own mask island (the segmentation has
  1,853 pieces). Isolated edges are mostly a fragmented segmentation, not graph building.

---

## 2. Why it keeps happening

1. **The clean-ups run once, mid-way, and later steps undo them.** `remove_loops_inside_one_lumen`,
   `remove_parallel_edges_in_lumen` and the final prune run on a graph still full of recovery's
   split nodes. The steps after them change the edges: the final degree-2 merge raises the
   duplicate pairs from 112 to 175, smoothing in `pipeline/stages.py` raises in-lumen loops from
   1 to 10, and the thick split adds bridges inside fat lumens. None of the clean-ups runs again.
   The clean-ups also create new dead ends (106 + 160) and islands (components 226 → 340 at the
   loop step), because they delete whole edges and nothing judges what is left.
2. **Duplicates are judged one whole edge at a time.** An edge is flagged only when at least half
   of its middle runs beside another (samples within r + 1 µm of either end are skipped,
   `_shadowing_points`). The verdict depends on where nodes happen to sit. A long edge with a
   duplicated stretch is never flagged, and a flagged edge goes whole, together with whatever
   hangs off it.
3. **Forks in one lumen have no rule.** 98 of 170 pairs share one node: two edges leave a junction
   and run side by side through one lumen (a Lee sheet's two edges at a flared end, or a branch
   leaving at a shallow angle). The degree-2 dedupe needs the same two nodes, the loop breaker
   needs a cycle, and the prune's "inside the parent's lumen" test only measures tip-to-junction
   distance.
4. **Mask recovery traces the far rim of lumens it already has.** Its "uncovered" test is a ball
   of the centreline's own radius. A flattened lumen's far side fails that test, so recovery
   Lee-thins it into a second strand plus hairs and joins them back. At HEAD that adds 300 pairs,
   104 pinhole loops and ~2,150 dead ends. The final prune then removes 3,521 stubs, 3,178 of
   them off the mask or inside their parent's lumen, and 331 survive. f096035's recovery ended
   with zero net coverage gain. 7c0630c gains 2.7 points of coverage for 2.4× the duplicates,
   +340 dead ends and +190 s.
5. **The ring test asks the wrong question.** `loop_inside_one_lumen` counts antipodal chords
   crossing more than 2 µm of background. Your marks (Phase 5) show that hole size is not what
   separates real loops from false ones. The chords also leave a bending vessel and cross
   background *outside* the loop, so a ring with no gap inside it reads as "round tissue"
   (18, 22).
6. **The stub rules read the wrong places.** The radius threshold is read at the node the dead end
   hangs from, which before the last merge is often a degree-2 node inside the stub itself. "Tip
   within parent radius + 1 µm of the junction" misses a hair running along the wall (79 tips end
   inside another vessel's lumen). A stub made of short pieces is eroded piece by piece.
7. **Thick-vessel joins run inside the fat lumen.** `_join_thin_arms_to_fat_ridge` adds each
   arm's new fat voxels to the targets (`fat_now`, `fat_coords`), so later arms join earlier
   bridges. The resulting bridges run beside the ridge and beside each other.
8. **Degree-2 merging skips two-edge rings** (`n1 == n2` in `smart_multigraph_degree2_removal`):
   4 such rings and 10 multi-edge node pairs remain.

---

## 3. The plan

Phases are ordered by payoff per risk. Each phase is its own commit (or a few), with tests in
the files CLAUDE.md names. Every phase is measured on the harness from Phase 0 before it lands.

### Status (2026-10-10)

All on `Devel_GUI_improvements_HD`, pushed.

| phase | state | commit |
|---|---|---|
| 0. Harness and fixtures | done; the harness now measures the graph as `build_network` saves it, thick-vessel split included | 1fe1d6a, e7df990 |
| 1. One clean-up loop at the end of the build | done | 1fe1d6a |
| 2. Doubled stretches | **closed, no code change.** No rule survived the marks; keeping junction-to-junction pair edges (option 2) was measured and rejected | -- |
| 3. Mask recovery | coverage part done; EDT-ridge tracing dropped (recovery's time is routing and the coverage test, not Lee thinning) | 21dc12c |
| 4. Stubs as whole chains | done; the report now judges "short" as the prune does | ae06500, 4e44afc |
| 5. Lee rings | no automatic rule; handed to the manual loop review (7b) | -- |
| 6. Thick-vessel bridges | done: thin arms aim at the fat ridge first, an earlier bridge only past the cap | eb75435 |
| 7b. Manual loop review | done; peers since folded it into the Post processing Review dropdown | 28141a7 |
| 7. Validation, docs | Nerve fixture done (unchanged), invariant test added, CLAUDE.md kept current; **Mason waits for its data drive** | e7df990 |
| (extra) Clean-up speed | radius and pair-verdict memories between rounds: build 339 -> 255 s, graph identical | e2b4830 |

**E14.5 against the acceptance targets** (graph as saved, thick-vessel split included, at e2b4830):

| target | now | met |
|---|---:|:-:|
| loops inside one lumen: 0 | 0 | yes |
| mask covered: >= 63.4% | 77.2% | yes |
| pairs in one lumen: <= 10 | 22 (11 with a thick-vessel bridge) | no |
| doubled centreline: <= 0.5% | ~4% (2,181 um) | no |
| dead ends with tip in another lumen: 0 | 7 (the same 7 before and after the split; 21 before the report fix) | no -- 7 drawn for marks |
| build time: <= f096035 + 20% (~142 s) | 255 s | no |

**Decisions since the first list (section 4):**
- **Phase 2 (2026-10-09).** Batch F, 30 fresh junction-to-junction pair deletions (15 each side
  of a 7.8 um doubled stretch): 17 vessel, 13 junk. The doubled-stretch rule scored 14/30
  (delete all 13, keep all 17); undoubled length > 7.1 um fit F (21/30) and failed on D+E
  (15/31). Pooled, junction-to-junction deletions are 56% vessel (34/61), dead-ended ones 81%
  junk (26/32). Option 2 on E14.5: pairs 22 -> 123, short loops 244 -> 390, doubled +888 um,
  coverage +2.3 points. **Kept today's removal**; the vessels it costs are left to the Post
  processing "two vessels in one lumen" review list.
- **Phase 6 (2026-10-09).** Three variants measured on one frozen copy of the code: current
  (33 pairs, 21 with a bridge, 48 arms unjoined), ridge only (22 / 9 / 73: strands 25
  capillaries), **ridge first (chosen: 22 / 11 / 44)**. Network resistance could only be
  compared between the two new variants (ridge first 18% lower); against current, 9 of 19
  large-vessel boundary snaps moved.
- **Build time (2026-10-09).** Leave mask recovery as it is.
- **Dead ends (2026-10-10).** No dead end escapes the prune. The 29 short interior dead ends
  left are all kept by its image-face protection (tip within 3 radii of a face); most lie 2-3
  radii from the face. Tightening it would need a marked grid; not asked for.

- **Tips inside other lumens (2026-10-11).** 15 of 21 were the report reading a tip as inside
  its own fat-vessel opening (the split cuts it off as a bridge edge); the report now leaves out
  the whole chain. The 7 left are stubs that leave the vessels round them and end inside one:
  three hook back into their own vessel, four end at another vessel -- drawn for marks (join /
  remove / leave), too few on their own to fit a rule.
- **The prune's protrusion test (2026-10-11): left as it is, deliberately.** It asks for the 16
  nearest centreline samples and masks out the stub's own, so a long stub can see only its own.
  Asking for 16 samples of *other* centrelines made it prune a vessel's short straight end past
  a junction at an image face (its protrusion is its length less a radius, and the rule overrides
  the face protection); the 4 E14.5 stubs that changed were all at image faces. The cap was
  hiding a flaw in the rule: it cannot tell a hair along a wall from a vessel's collinear end. A
  fix means redesigning the rule (exempt collinear continuations, or face stubs) on marked cases.

Marks and keys: `outputs/loop_review/` (grids A-F, `grid_F_marks.csv`),
`outputs/dead_end_review/` (`short_dead_ends_key.csv`, `tips_in_lumen.png` + key).

### Status (2026-10-08, uncommitted in the working tree)

- **Phase 0 — done.**
  - `graph.diagnose_lumen_artefacts`, logged in `build_network`'s graph checks.
  - `preprocessing.shadowed_samples`.
  - `scripts/graph_artefacts.py`, plus `tests/lumen_artefact_fixtures.py` and three new test files.
- **Phase 1 — done.**
  - `consolidate_lumen` is now the build's last step, and runs again after smoothing.
  - The pair removal no longer cuts vessels off.
  - On E14.5 against 5fab427, final smoothed graph:

    | | 5fab427 | Phase 1 |
    |---|---:|---:|
    | pairs in one lumen | 124 | 9 (all kept by the guard) |
    | loops inside one lumen | 12 | 0 |
    | dead ends with tip in another lumen | 24 | 9 |
    | dead ends beside another centreline | 28 | 0 |
    | short dead ends | 73 | 41 |
    | isolated edges | 151 | 110 |
    | doubled centreline | 3.3 mm | 2.6 mm |
    | mask covered | 64.7 % | 62.8 % |
    | build time | 388 s | ~430 s |

  - The doubled centreline left is partial overlap, which is Phase 2.
- **Phases 0–1 committed and pushed** (1fe1d6a).
- **Phase 2: attempted 2026-10-08, and it regresses E14.5.** Kept on the local branch
  `claude/phase2-doubled-stretches` (5deb1fd); not merged or pushed. Contents:
  - `graph/doubled_stretches.py`: trim, zip or cut-and-rejoin each doubled stretch;
  - the neck test;
  - a pair removal that only takes an edge lying wholly inside the other's lumen.

  Against 1fe1d6a:

  | | 1fe1d6a | Phase 2 attempt |
  |---|---:|---:|
  | pairs in one lumen | 9 | 54 |
  | doubled centreline | 2,476 µm | 3,182 µm |
  | short loops | 225 | 297 |
  | centreline | 52.3 mm | 57.0 mm |
  | mask covered | 62.8 % | 64.3 % |
  | clean-up time | 100 s | 220 s |

  - **Why:** Phase 1 resolves most pairs by deleting a whole edge, dropping its undoubled
    remainder with it (~4.7 mm on E14.5). The cautious rule keeps those pairs, and the stretch
    step resolves few of them on real data. Most are short forks at junctions, below the 3 µm /
    two-radii minimum, or with their parting point at an edge's end vertex.
  - **What decides it:** is the undoubled remainder Phase 1 deletes vessel or junk? That is a
    judgement call. The next step is a marked grid of those pairs (CLAUDE.md, "Marked example
    grids"), then the rule from the marks.
  - **Reusable now, independent of that decision:** the fixtures, the neck test and the
    rejoin/zip operations.
- **Grids D and E (2026-10-08/09).** You marked 60 of the edges Phase 1 deletes with an
  undoubled part (`outputs/loop_review/grid_D_*`, the Doubled Edge Review page for E):
  - **Dead-ended deletions:** junk 26 of 32 (81 %). Deleting them is right.
  - **Junction-to-junction deletions:** 17 vessel, 14 junk. "Keep all of these" (the dead-end
    rule) failed its fresh check on E: 19/30, against 20/30 for deleting everything.
  - **Within the junction-to-junction ones:** a short doubled stretch (< ~8 µm) points to
    vessel, 21/31 in leave-one-out against 17. Too weak to trust until a fresh batch confirms
    it.

### Phase 0 — Measure first (harness + fixtures)

- **`graph/diagnostics.py`: `diagnose_lumen_artefacts(G, support)`.** One report covering:
  in-lumen pairs; duplicated centreline µm, counted per sample rather than per edge; forks;
  in-lumen and pinhole loops; dead ends by class (inside another lumen, off-mask, wall bump,
  mask continues, blind, image face, island). Log it in `build_network`'s graph checks on the
  graph that is actually saved, i.e. after smoothing and the thick split. Add a per-step version
  under `debug`.
- **`scripts/graph_artefacts/`** (beside `scripts/branch_comparison/`). Rebuild a graph from a
  saved skeleton with a step pickle per `STEP_LABELS` entry, print the §1.2 table, and compare
  two git refs from `git archive` snapshots. This promotes the session's scratch harness
  (`audit_graph.py`, `audit_steps.py`, `rebuild_steps.py`). Datasets: E14.5 MCA, Mason
  (`Mason.yaml`), `Nerve_capillaries_cropped`, `tests/dense_capillary_fixtures.py`.
- **Synthetic fixtures** (`tests/lumen_artefact_fixtures.py`), one per case, each with the right
  answer pinned:
  - a ribbon (flattened) lumen → one strand, with side branches still attached;
  - two capillaries touching with a neck → two strands;
  - a ring inside one lumen with no gap inside it, in a bending vessel → no ring;
  - a loop round a small (≈ 1 µm) gap → the loop stays (your D1 marks);
  - a real tight loop round 4–6 µm of tissue → the loop stays;
  - a wall bump → no stub;
  - a blind branch 2–4 radii long → kept or dropped per decision D2;
  - a branch leaving at 15° whose first stretch is inside the parent → one junction, no fork;
  - a flared vessel end → no fishtail.
- Record the baseline (§1) in the PR description. **Acceptance targets** on E14.5 (to confirm
  with you):
  - duplicated centreline ≤ 0.5 %;
  - in-lumen pairs ≤ 10, all with a neck;
  - 0 in-lumen loops on the saved graph;
  - pinhole loops ≤ 10;
  - 0 dead ends whose tip is inside another lumen;
  - coverage no lower than f096035's 63.4 %;
  - components not increased;
  - build time ≤ f096035's + 20 %.

### Phase 1 — One clean-up loop at the end of the build (fixes cause 1)

- **`graph/assemble.py`.** Replace steps 12–14 and the trailing `remove_edges_off_the_mask` with
  `consolidate_lumen(G, support, ...)` (new `graph/consolidate.py`). It repeats, until nothing
  changes (capped at ~5 rounds):
  1. degree-2 merge;
  2. off-mask removal;
  3. duplicate/fork resolution (Phase 2);
  4. loop breaking;
  5. stub prune (Phase 4).

  The merge comes first so every judgement is made on whole vessels, not recovery's split
  pieces. Nothing in the loop may delete the last connection of a component: an edge whose
  removal would split one is rewired instead (Phase 2). The peer session's new
  `join_facing_dead_ends` (uncommitted, step 15) runs just before it.
- **`pipeline/stages.py` `build_network`.** Run `consolidate_lumen` again after
  `smooth_graph_centrelines` and after `insert_thick_vessel_junction_nodes`, or move smoothing
  inside the loop's last round. Either way, the graph that is pickled is the graph that was
  judged.
- `STEP_LABELS` changes ripple into `pipeline/progress.py`, `gui/progress.py` and
  `tests/test_graph_assemble.py`, `test_pipeline_progress.py`, `test_progress_reporting.py`,
  `test_graph_logging.py`, `test_gui_run_log.py`. Keep the existing labels and add one
  (`consolidate_lumen`) so saved runs and the progress bar still line up. Coordinate the count
  with the peer's step 15.
- Expected on E14.5 at HEAD: the 112 → 175 jump, the 1 → 10 after smoothing, and the 266 dead
  ends the clean-ups create should all go. Verify with the per-step table.

### Phase 2 — Duplicates: judge stretches, zip forks, rewire rather than delete (causes 2, 3)

- **One "same lumen" predicate** in `preprocessing/bridge_mask_support.py`: two paths are one
  lumen along a stretch when the chords between them stay in the mask **and** the distance to
  background shows no neck between them (the interior minimum is not below ~0.6× the smaller
  end radius). That replaces the "< 1 µm of background" test, so two touching capillaries stay
  two even as the duplicate rule gets stricter; on E14.5 that guard spares 2 of 86 pairs.
  Use the predicate in all four places that ask the question today:
  - `path_shadows_existing_vessel` (bridge guards, recovery);
  - `paths_separated_by_background` (degree-2 `are_paths_similar`, direction-aware collapse);
  - `diagnose_parallel_duplicates_in_lumen`;
  - the new resolver.
- **Shadowed runs, not whole edges.** Expose `_shadowing_points` without its early return, as
  per-sample flags. A run of shadowed samples longer than max(2 r, 3 µm) is a duplicated
  stretch. There are three cases:
  - **The run ends at a dead end's tip:** trim the dead end back to the run's start. It is a
    strand beside a vessel. Nothing left outside the parent's lumen means the dead end goes.
  - **The run starts at a shared node (a fork):** *zip*. Insert a node on the kept edge where the
    two part company, re-attach the other edge's remainder there, and drop the shadowed stretch.
    A branch leaving at a shallow angle keeps its own path from the true divergence point.
  - **The run is interior to both edges (a double strand between different nodes):** keep the
    medial strand over the run. Re-attach every edge that met the dropped strand inside the run
    to the nearest point of the kept one (split it there).
- **Medial choice.** Keep the strand with the larger mean mask radius over the shared run only
  (the existing `_wall_hugging`, applied to the run rather than the whole edge). Smoothing can
  then centre it.
- **Lengths and attributes.** Re-measure `length` from `voxels`. Carry `bridge_kind` and
  `recovered` onto the merged edge so the audit can still tell where an edge came from (merges
  drop them today).
- **Tests** in `tests/test_parallel_duplicates.py`, `test_lumen_loops.py`, `test_graph.py`:
  - ribbon lumen → one strand, branches still connected;
  - touching capillaries → two;
  - fork → one junction;
  - a long edge with a 10 µm doubled stretch → trimmed, not deleted;
  - the pair diagnostic is the same whether or not a degree-2 node sits on the edge (a
    regression guard for the 112 → 175 effect).

### Phase 3 — Mask recovery: stop drawing the far rim (cause 4)

**Status 2026-10-09: the coverage part is committed and pushed (21dc12c).**

- **The "explains new mask" filter does not hold up.** Recovered paths that cover more new mask
  per µm survive the clean-up *less* (69 % at 2–5 µm³/µm, 40 % above 20).
- **EDT-ridge tracing** would mostly save the 4.4 mm of 11.6 mm recovery traces that the
  clean-up then removes: a saving in time, little change in the result.
- **A gap in the measurements:** the comparisons so far skip the thick-vessel split, which
  accounted for 34 of 89 pairs in the saved 7 Oct graph. That is Phase 6.

- **The rule.** `uncovered_mask_voxels` follows each sample's cross-section to the wall: 32
  directions, at most 4 radii, and only while the mask runs a radius ahead and behind too, so a
  narrower branch is not hidden.
- **E14.5, against 35da64f:**

  | | 35da64f | Phase 3 |
  |---|---:|---:|
  | doubled centreline | 2,644 µm | 2,168 µm |
  | dead ends | 707 | 631 |
  | short loops (none inside one lumen) | 225 | 249 |
  | mask covered | 77.3 % | 77.7 % |
  | build time | 539 s | 383 s |

- **What recovery adds now:** 4.3k edges instead of 7.4k, and almost no net centreline after
  the clean-up.
- **Not done yet:** tracing pieces as EDT-ridge trees, and the "explains new mask" filter.


Coordinate with whoever owns recovery (7c0630c, this morning) before starting.

- **Coverage means "seen from a centreline through the lumen", not "within one radius".** In
  `uncovered_mask_voxels`, a mask voxel is covered when its straight chord to the nearest
  centreline sample stays in the mask and is no longer than the lumen's half-width across that
  chord. That turns 79 % of today's "uncovered" (820 k voxels) into covered and leaves ~200 k
  that may really be missed vessels. Use the same definition for the coverage the audit reports.
- **Trace a piece as one tree, not a Lee sheet.** Use the EDT-ridge tracer the thick-vessel path
  already has (`preprocessing/thick_vessels.py`: `skeletonize_edt_ridge`,
  `_component_edt_ridge_on_crop`) instead of Lee thinning. Failing that, run Phase 4's ball-cover
  hair test inside the piece before any join is tried.
- **Add a strand only if it explains mask nothing else does.** It must cover at least N µm³ of
  mask that no existing lumen ball covers, and run nowhere beside an existing centreline (the
  Phase 2 predicate). The duplicate check then refuses rims before routing, not after.
- **Target on E14.5:** recovery adds ≈ 0 duplicate pairs, ≈ 0 pinhole loops and < 50 dead ends,
  while keeping 7c0630c's coverage gain measured the new way. Recovery time should drop with the
  traced volume (most of today's 1.1 M voxels are rim).

### Phase 4 — Stubs: a ball-cover test, judged on whole chains (cause 6)

- **Judge the chain, not the piece.** From each tip, walk to the first node of degree ≥ 3 and
  treat the whole walk as the stub. Read the parent radius there, at the real junction. Running
  inside Phase 1's loop after the degree-2 merge mostly guarantees this, but `prune.py` should
  not depend on it.
- **Protrusion test** (replaces "tip within parent radius + 1 µm of the junction" in
  `prune_vascular_stubs.not_a_vessel`):
  - Protrusion = the furthest the stub gets outside every other vessel's lumen, i.e. the max over
    its samples of (distance to the nearest *other* centreline sample − that sample's mask
    radius).
  - Prune when protrusion < k × the stub's own radius near the tip (k ≈ 1–1.5): a wall bump, or a
    hair along the wall, whatever its length.
  - It catches the 79 tips inside another lumen and the 117 dead-end-beside-vessel pairs, and it
    scales with the vessel the way the radius multiple does.
- **The length rule for blind ends** stays as the radius multiple at 3.0 (the in-flight change,
  uncommitted in `schema.py` / `tests/test_prune.py`). Short real sprouts are not kept (decision
  D2). E14.5 has 278 dead ends under 3 r.
- **A tip the mask runs on past** (142) is a truncated vessel. It is a job for recovery and the
  peer's facing-end join, so it is not pruned below `min_stub_length` as now.
- **Islands** (157 single-edge components): left to `boundary_handling = remove_disconnected`
  (decision D3); no new step.

### Phase 5 — Lee rings: does the loop go round a gap? (causes 5, 8)

**Revised 2026-10-08 after D1.** You marked 30 leftover loops
(`outputs/loop_review/loops_to_mark_segmentation.png`, key `loops_to_mark_key.csv`): 22 real,
8 false (1, 3, 8, 18, 20, 22, 25, 28).

- **Hole size does not separate them.** The five smallest holes (about 2.8 µm wide) are all real,
  and the false ones span 2.8–4.8 µm. A loop round a small but real gap is a vessel loop.
- **So both earlier ideas are dropped.** Plugging tunnels before thinning would delete loops you
  call real. A pinhole width relative to the lumen measures the wrong thing.
- **What the false loops have in common:**
  - no gap inside the loop at all (18, 22): the antipodal chords leave a bending vessel, so the
    current test reads "tissue";
  - the gap at the loop's edge rather than inside it (1);
  - a path running through background (28: 29 % off the mask);
  - a loop through fragmented, speckled segmentation (3, 8, 20, 25).
- **Where it stands after 120 marked loops (2026-10-08; 32 false, 88 real):**
  - **No automatic loop rule is ready.** Every measure fitted on one set failed on the next
    fresh set:
    - gap inside the loop: 16/30 on grid A, against 21/30 for keeping everything;
    - bent out of plane: no gain once grid B was added;
    - share of the loop's area that is gap: 23/30 on batch C, against 24/30.
  - **Edge measures don't separate them either.** A same-lumen duplicate edge is in 34 % of
    false loops and 33 % of real ones. Bridges and recovered edges, strand shadowing and mask
    support show no useful gain (best 90/120, against 88/120).
  - **Your drawings (batch C, `outputs/loop_review/annotate_C_drawn_vs_graph.png`) show what a
    false loop is:** a loop with one *side* that isn't a vessel. Either it cuts across the wedge
    between two branches leaving a junction (C23, C24), or it runs down both sides of a slit in
    one flat vessel (C29). Where you traced vessels, the real loops ran wholly along them (9 of
    12 at ≥ 97 %). Local segmentation measures don't capture that judgement.
  - **What it means for the plan:**
    - the current in-lumen rule stays as it is (it already breaks the clear Lee rings, 455 edges
      on E14.5);
    - roughly a quarter of the remaining short loops (~70 of ~275) are false by your eye, and
      fixing them waits on Phases 2–3 or on a manual review;
    - the tracings are effectively hand-drawn centrelines, so they can also score the graph
      beyond loops.
- **Update after grid A (30 fresh loops, marked 2026-10-08): the test below failed and is
  withdrawn.** On fresh loops it got 16/30 right against 21/30 for keeping everything, and 10 of
  the 15 loops it would remove were real (loops round very small gaps). Pooling all 60 marks
  (17 false), the best rule found, scored leave-one-out, is: **false when the loop is bent well
  out of its own plane (third / second singular value of its points > 0.5), or encloses less
  than ~2.7 µm² of gap in its plane (±1 µm)**. That scores 50/60, against 43/60 for keeping
  everything. It would remove ~10 of 17 false loops and ~3 of 43 real ones. Wait for grid B
  (loops in speckled mask) and confirm on a fresh grid before implementing.
- *(Withdrawn)* ~~**New test** in `lumen_loops.loop_inside_one_lumen`, replacing the
  antipodal-chord count.~~ A loop is round tissue only if both hold:
  - its spanning surface holds a gap (background, in the loop's plane ±1 µm) whose centre lies
    inside the loop and away from its own path (deeper than the lumen radius, at least
    0.5 µm²);
  - at least 75 % of the loop's own path is in the mask.

  On the 30 marks this gets **26 of 30 right** (marking all of them "real", which is what the
  code does today, gets 22). It removes 18, 22, 1 and 28 and **removes no loop you called real**;
  it still misses 3, 8, 20 and 25. That was fitted on these 30, so confirm it on a fresh grid of
  30 before relying on it (CLAUDE.md, "Marked example grids").
- **Fragmented-segmentation loops** (3, 8, 20, 25) need another measure (e.g. how broken the
  mask is round the loop). Four examples are too few to fit one: a second grid focused on loops
  through speckled mask would settle it.
- **Two-edge rings and multi-edge pairs** (4 and 10 left): make sure `_Pairs.shortest_cycle`
  tests them (it handles multi-edges; check the `n1 == n2` rings), and let degree-2 merging
  resolve `n1 == n2` through the loop breaker rather than skipping it.
- Smoothing: covered by Phase 1 (rings re-judged after smoothing).

### Phase 6 — Thick-vessel bridges inside the fat lumen (cause 7)

- Try the source first. In `_join_thin_arms_to_fat_ridge`, a later arm targets the original
  ridge only: arms' bridge voxels go into `result` but not into `fat_coords`. Measure the join
  failures this adds (arms that can then no longer reach a ridge within the cap).
- Otherwise, after `insert_thick_vessel_junction_nodes`:
  - re-anchor each `is_zero_resistance` bridge that shadows the fat centreline or another
    bridge (Phase 2 predicate). It becomes a straight run from its boundary node to the nearest
    point of the fat centreline, split there.
  - Two bridges sharing a run become one with a Y at the boundary.
  - These edges are zero-resistance, so flows barely move; check the network resistance anyway.
- Tests: `tests/test_graph_thick_vessel_junctions.py`, `test_thick_vessel_skeletonisation.py`.

### Phase 7b — Manual loop review in the Post processing tab (added 2026-10-08)

The leftover false loops need a person's judgement (Phase 5: 120 marked loops, no automatic
rule). Give that judgement a place in the panel, modelled on the existing "Manual 4+ vessel
junction correction".

- **List:** a "Manual loop review" checkbox on the **7. Post processing** tab (off by default)
  lists the network's short loops, ≤ `LOOP_SEARCH_UM`, shortest first. Each shows its length and
  a link to zoom to it (camera on the loop's centre, in its own plane where the view allows).
- **Edit:** per loop, **Delete this side** (the user picks one of the loop's edges, as they do
  for a 4+ junction split) or **Keep** (marks it reviewed and drops it from the list).
- **Apply:** deletions go through `graph/post_processing.py` like the other hand edits. The
  `post_process` stage then brings the edited network in line and re-solves it, as it already
  does.
- **Reviewed loops persist:** they are stored with the run (`gui/run_snapshot.py`), so a
  reloaded run does not ask again.
- **Train on it:** every Keep and Delete is a labelled example. Write them to
  `outputs/<stem>_loop_review.csv` with the loop's measures, so later batches can be used to
  look for a rule again.
- **Tests:**
  - the loop listing, its order and the zoom target are pure (`gui/post_processing.py`) and
    tested in `tests/test_gui_post_processing.py`;
  - deleting a side through `graph/post_processing.py` is tested in
    `tests/test_post_processing*.py`;
  - one widget test drives the checkbox, list and buttons (`-m gui`).

### Phase 7 — Validation, docs, rollout

- **Before/after on E14.5, Mason and the fixtures.** Report:
  - the §1.1 table and the §1.2 per-step table;
  - skeleton coverage;
  - network resistance and total inflow (duplicates are parallel conductances: removing them
    raises resistance, so say by how much);
  - build time.
- **Invariant test** in `tests/test_pipeline_invariants.py` on `Nerve_capillaries_cropped`: after
  `build_network`, `diagnose_lumen_artefacts` reports no in-lumen pairs without a neck, no
  in-lumen loops, and no dead end whose tip is inside another lumen.
- **CLAUDE.md** (Key modules: `lumen_loops`, `mask_recovery`, `prune`, the new consolidate
  step), schema help text, `python examples/regenerate_configs.py`.
- **Commits.** Per phase, through a private index (shared checkout). Run `pytest -m "not slow
  and not gui"` (~13 min) per phase, and the GUI suite in the background before pushing.

---

## 4. Decisions needed from you

- **D1 — How small a hole is a pinhole? Answered 2026-10-08 by marking 30 loops:** size is not
  the question. A loop round any real gap is a vessel loop; a false one has no gap inside it, or
  runs through background or broken segmentation. See Phase 5.
- **D2 — Short blind ends. Answered 2026-10-08: do not keep real short sprouts.** The 3 × radius
  rule stays as the length rule for blind ends. The protrusion test is added on top of it, to
  catch hairs longer than 3 r that never leave another vessel's lumen.
- **D3 — Mask islands. Answered 2026-10-08: do not keep them; use `remove_disconnected`.**
  `boundary_handling = remove_disconnected` already deletes every component without both an
  inlet and an outlet, single edges and multi-edge islands alike, at the Haemodynamics stage. No
  separate graph-build step is added. (The default, and `config.yaml`, is `leave_unsolved`.)
- **D4 — Recovery's trade-off. Answered 2026-10-08: no.** The 2.7-point coverage gain of
  7c0630c is not worth 2.4× the duplicates. If Phase 3 cannot keep both, fewer duplicates wins:
  fall back to f096035's recovery behaviour (or tighter) rather than keep the extra coverage.

## 5. Risks

- **A stricter duplicate rule could delete a real vessel touching another.** The neck test is the
  guard, and the "touching capillaries" fixture pins it.
- **The new ring test was fitted on 30 examples.** A fresh marked grid has to confirm it before
  it lands, and the "small real gap" fixture pins your D1 marks.
- **Rewiring instead of deleting is more code than deleting.** Each case gets a fixture, and the
  consolidation loop asserts it never increases the component count.
- **Peers edit `assemble.py`, `mask_recovery.py`, `prune.py` and `schema.py` concurrently** (the
  facing-end join and the stub multiple are uncommitted right now). Rebase each phase on the
  latest `origin` and re-run the harness before landing.
