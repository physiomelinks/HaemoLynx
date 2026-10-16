# Plan: bridging — the one gate, every site that uses it, and the junctions thick vessels miss

Written 2026-10-13 against `Devel_GUI_improvements_HD` at **040170b**. Untracked, like the other
`PLAN_*.md` files. Section 1 is a code review (every claim has a `file:line`); nothing in it was
measured on a dataset yet. Section 2 is the measurement that has to come first.

**In one paragraph.** Every join the pipeline draws goes through one gate,
`preprocessing/bridge_mask_support.py`: a path is a vessel when it stays in the segmented mask bar
one dropout (2 µm, 80% in the mask) and does not run beside a centreline already in the same
lumen. The gate is sound and its 76 tests pass. What is wrong sits round it: the thick-vessel
junction step bridges only a vessel whose centreline runs on into the big vessel and touches its
centreline, so a side vessel that *stops at the wall* gets no junction and no other site can reach
the big vessel's centreline (Harvey's screenshot, 2026-10-13: several on one wide vessel); the
direction rule judges the straight chord while the mask route is what is drawn; the optimise and
orphan reconnects' "skeleton path" is the chord; a bridge short enough to sit inside one wide lumen
is never tested for doubling; the skeleton router's window ignores voxel anisotropy; hub spokes
skip the same-lumen test; the refusal log counts tests, not tips; and with a saved skeleton
(`do_skeletonize` off) the panel hides every bridging setting, graph building's included. The plan:
measure the missed junctions on E14.5 and on the screenshot's run and put ~30 of them in a grid
for marks (Phase 0), bridge a vessel that stops at a thick vessel's wall (1), fix the gate's
seven small defects with a test each (2), free the settings from `do_skeletonize` (3), and
re-measure (4).

---

## 0. The map (what the code does today)

One gate, `MaskSupport` ([bridge_mask_support.py](src/haemolynx/preprocessing/bridge_mask_support.py)):

- **Background test** `bridge_is_supported` (l.222): the path cut into pieces of at most half the
  finest voxel, read at midpoints; accepted when the longest background run ≤
  `bridge_max_background_gap_um` (2) and ≥ `bridge_min_mask_fraction` (0.8) of it is mask, the
  fraction waived while all background fits one run.
- **Same-lumen test** `path_shadows_existing_vessel` (l.244): samples further than (lumen radius
  + 1 µm) from both ends are *shadowed* when the nearest existing centreline point lies beside
  them (chord cosine < 0.7), within 2 radii + 1 µm, across < 1 µm of background; a path is a
  duplicate when half its judged samples are.

The sites, in run order, and what each adds on top:

| Site | Where | Reach | Direction rule | Path tried |
|---|---|---|---|---|
| Closing, gap filling | `skeleton.py` `preprocess_skeleton_for_graph` | `skeleton_closing_radius` 2, `skeleton_bridge_gap_size` 3 voxels | — | adds voxels only within one voxel of the mask (`within`) |
| Bundle hub spokes | `_hub_link_gate` l.1109 | bundle window | — | straight; **background test only** |
| Skeleton component bridges | `connect_skeleton_components` l.1476, `_SkeletonBridgeGate` l.1352 | `skeleton_max_bridge_distance` 4 voxels (2 µm) | tip within 60° of the gap, a tip target facing back | straight, then routed (`_bridge_path_through_mask`, pad 3 voxels); up to 8 targets a tip |
| Gap bridges | `build.py` l.393-440 | `graph_reconnect_threshold` 10 µm | ≤ 80° off either end, ends ≥ 3 µm | `MaskBridges`: routed (pad 3 voxels, scaled), then chord |
| Secondary loops | `reconnect.py` `reconnect_secondary_loop_edges` | 6000 voxels | — | **not gated**: ≥ 90% of the path on the skeleton |
| Optimise reconnect | `optimise.py` l.40-195 | 10 µm | as gap bridges | `validate_skeleton_connection`'s line, then routed, then chord |
| Orphan reconnect | `optimise.py` l.253-420 | `final_orphan_reconnect_threshold` 3 µm, **to nodes only** | as gap bridges (a junction target has none) | as optimise |
| Mask recovery | `mask_recovery.py` | 3 µm from a covered lumen | — | routed; own loop rule |
| Facing dead ends | `facing_ends.py` | `facing_dead_end_max_gap_um` 10 µm | ≤ 60°, ends ≥ 3 µm, gap ≤ both ends | chord; **one** background stretch, however long; same-lumen test |
| Thick-vessel junctions | `thick_vessel_junctions.py` l.315 | region | — | straight bridge from where a *connector* stretch leaves the lumen to the tree; **only for a stretch that touches the tree** |

Graph-level sites tag `bridge_kind` and `bridge_background_um` (longest run; the facing join writes
the total, the same thing when there is one stretch).

---

## 1. Findings

Ordered by what they cost a run. **F1 is the one Harvey can see.**

### F1. A side vessel that stops at a thick vessel's wall gets no junction

`insert_thick_vessel_junction_nodes` (l.315) samples every edge against the thick region and the
thick centreline tree, and makes a bridge only for a *connector*: an inside stretch that touches
the tree (module docstring, l.17-29; `_decide_what_stays` l.197, `_bridge_origins` l.269). A
side vessel whose centreline ends inside the region short of the tree is not a connector, and one
whose centreline ends *outside* the region, at the wall, is never sampled inside it. Neither is
bridged. No other site can do it either: the orphan reconnect joins a tip to a **node** within
3 µm, and the thick centreline is one long edge with no node near the wall; the gap bridges join
terminal to terminal; the facing join needs a second dead end. `join_dead_ends_to_vessel_sides`
(`dead_end_extension.py`) is the site built for a tip beside a vessel, and it is off
(PLAN_dead_ends.md gates it on grids G1-G3, where no rule reached 90%).

Harvey's screenshot (2026-10-13, dataset to be named: section 4) shows it on one wide vessel:
four or five side vessels ending within a lumen radius of the yellow centreline, no bridge, each
a dead end. In `graph/dead_ends.py`'s kinds they come out as `stops_short` (mask runs on) or
`artefact` (tip inside a neighbouring lumen), which is wrong both ways: they are junctions.

Why the thin arm did not run on to the tree is a separate question with four candidate answers,
and Phase 0 counts them: (a) `_join_thin_arms_to_fat_ridge` (thick_vessels.py) refused the arm:
longer than `skeleton_thick_vessel_max_bridge_radius_multiple` / `_max_bridge_distance_um`, or the
braid guard; (b) the arm was joined and the stub prune or `_remove_blind_vessels_beyond_bridges`
(l.750) took it back out; (c) the side vessel's segmentation pinches at the wall, so Lee thinning
ended the arm outside the region and the 2 µm skeleton bridge could not reach the body; (d) the
run has no thick region there at all (the inscribed-radius gate), so the wide vessel is a Lee
sheet and the "centreline" the arm should meet is one of several strands.

### F1b. A branch of the thick centreline gets no bridge where it leaves its trunk

Harvey, 2026-10-13, on the same screenshot: the yellow (thick) centreline forks into a smaller
yellow branch with no bridge at the fork. By design the junction step treats every sample on the
tree as thick vessel (l.407: `on_tree` → `_CENTRELINE`), so a branch of the tree is never a
connector and the fork is a plain junction. But the branch's centreline runs inside the trunk's
lumen from the fork out to the trunk's wall, a trunk radius of it, and that stretch is counted as
branch vessel at the branch's diameter: exactly the resistance the bridges exist not to count
(module docstring l.9-11). The rule should be the one every other joining vessel gets: from
where the branch leaves the trunk's lumen (its inscribed radius falling to the branch's own) to
the trunk's centreline, one `IS_ZERO_RESISTANCE` bridge; the branch's own centreline starts at
the wall. Which arm is the trunk is read from the inscribed radius along each arm (Phase 0's
`forks.csv` measures all three), not assumed. Where the two widest arms are not collinear the
fork is a real bifurcation of equals and both daughters get a bridge of the parent's radius.

### F1c. Bridges drawn along the thick vessel, not across it (Harvey's "should not have a bridge")

Harvey's wrong examples (2026-10-13, three screenshots) are bridges running *along* a vessel's
own line: where the thick centreline tree ends inside the region and a thin vessel carries on
from it, the thin vessel's stretch inside the region is treated as a connector, removed, and
replaced by a straight bridge continuing the thick vessel's axis to where the thin vessel leaves
the region. That is a taper, one vessel narrowing, not a vessel opening into another, and it
needs no bridge: the thick centreline should run on into the thin one as vessel.

Measured on the rebuilt E14.5 graph (`outputs/thick_junction_review/bridges.csv`,
`bridge_shapes.txt`), by where each bridge's centreline end sits:

| the bridge's centreline end | n | median cos to the tree's axis | along the axis (cos ≥ 0.7) |
|---|---|---|---|
| splits a thick edge in two (the shape of every correct example) | 34 | 0.18 | 1 |
| at the thick centreline's **end** (one thick edge only) | 28 | 0.29 | 7 |
| on one thick edge and another vessel | 7 | 0.24 | 0 |
| not on a thick edge at all (the 'not connected at one end' pictures) | 9 | 0.49 | 2 |

So 44 of 78 bridges are not the correct shape, and 10 run along the vessel. The 9 off-tree ends
come from `_choose_bridges` (l.620-640) taking "the nearest centreline *sample* within 2 radii +
5 µm" from `cand_tree`, whose candidates are any sample labelled centreline within the tree's
tolerance (up to 30% of the radius, 3.6 µm on a 12 µm vessel): the bridge ends beside the thick
edge, on its own node, and the thick edge is not split there. Harvey's "not connected to the
bridge on one end" pictures (`images/8`, `12`) are that.

Rule for Phase 1 (to confirm with Harvey): a bridge is drawn only from a vessel that enters the
thick vessel from the side; its centreline end always splits a thick edge (snapped onto the
edge's own polyline, never a separate node beside it); a stretch continuing the thick
centreline's end along its axis is the thick vessel tapering and stays vessel. The measure that
separates the two on E14.5 is the bridge's cosine to the tree's local axis (across: median 0.18;
Harvey's wrong ones: 0.7-0.97), checked against his marks before it is relied on.

### F2. The direction rule judges the chord; the mask route is what is drawn

`gap_bridge_pairs_following_ends` tests the straight join against each end's heading (build.py
l.393; optimise.py l.103, l.370); `MaskBridges.bridge` (reconnect.py l.167) then draws
`route_through_mask`'s path, which may leave the terminal at another angle. The route is held to
a window 3 voxels past the ends' bounding box (`MASK_BRIDGE_WINDOW_PAD`, l.104), so the deviation
is bounded, but a fold-back through the mask is exactly what the 80° rule exists to refuse
(memory: gap-bridge-direction-rule) and it never sees the routed path.

### F3. The "skeleton path" the optimise and orphan reconnects prefer is the chord

`validate_skeleton_connection` returns `get_line_points_3d(p1, p2)` (validate.py l.82-95,
_helpers.py l.204): the straight voxel line, once 70% of its points lie within a voxel of the
skeleton. So the `preferred` path is the chord, the mask route is only tried when the chord fails,
and when it fails the chord is measured again as the last candidate. Behaviour is the same as
"chord, then route"; the docstrings at optimise.py l.59 and l.285 and reconnect.py l.150 say
otherwise.

### F4. A bridge short enough to lie inside one wide lumen is never tested for doubling

`_judged_samples` (l.283) leaves out every sample within (radius + 1 µm) of either end. A bridge
of length L with both ends in a lumen of radius r has no judged sample when L < 2(r + 1), and
`path_shadows_existing_vessel` then answers "not a duplicate" (l.278: `judged > 0 and ...`).
Inside a 6 µm-radius vessel that is every bridge under 14 µm: the gap bridges (10 µm) and the
orphan reconnect (3 µm) can draw a second strand there unchallenged. The thick split and
`consolidate_lumen` clean up after it (CLAUDE.md: 21 of E14.5's 33 pairs in one lumen were arms
joined to bridges), which is the wrong place.

### F5. The skeleton router's window ignores voxel anisotropy

`_bridge_path_through_mask` pads by `_BRIDGE_MASK_WINDOW_PAD` = 3 voxels on every axis
(skeleton.py l.1216, l.1242); `route_through_mask` scales the same pad by spacing
(reconnect.py l.117: `reach = ceil(pad / relative)`). On a 0.5 × 0.5 × 2 µm stack a skeleton
bridge may wander 6 µm in z and 1.5 µm in-plane.

### F6. Hub spokes skip the same-lumen test

`_hub_link_gate` (l.1109) runs the background test only. Every other site runs both.

### F7. The skeleton log counts refusals per test, not per tip

`_SkeletonBridgeGate.accept` increments `refused_background` for each of up to 8 targets a tip
tries (l.1384, `_MAX_MASK_TESTED_TARGETS_PER_TIP` l.1341), so "refused N bridge(s) crossing
background" can be several times the tips refused, and a tip whose eight targets were all refused
is counted in neither "declined" nor "refused".

### F8. With a saved skeleton the panel hides every bridging setting

`bridge_require_mask_support` has `requires=("do_skeletonize",)` (schema.py l.2881-2897) and the
gap and fraction settings require both (l.2910, l.2926). `build_network` reads all three
(stages.py l.1220-1222, `graph_build_arguments`), and `facing_dead_end_max_gap_um`,
`remove_short_loops_under_um`, `extend_dead_ends_to_faces`, `trace_on_dead_ends` and
`join_dead_ends_to_vessel_sides` require `bridge_require_mask_support` (l.3048-3113). The panel
reads the whole chain (`form.prerequisite_chain`, l.271), so a run loading its skeleton hides
all eight rows while graph building still reads them. Nothing in `tests/test_gui_layout.py`
or `test_pipeline_schema_api.py` checks that a setting a stage reads is visible when that
stage runs.

### F9. A mask of the wrong shape silently turns the gate off, or is read wrong

`connect_skeleton_components` builds the gate only when `mask_support.mask` is the skeleton's
shape (l.1570) and `preprocess_skeleton_for_graph` likewise (l.2066): a mismatch is silent, and
the run proceeds un-gated. `build_graph_from_skeleton` (assemble.py l.575-585) has no check and
would index the wrong voxels. `robust.4a` says a mask shape mismatch raises a named error; it
does at the loader, not here.

### F10. Half-voxel sampling is exact only on the finest axis

`_sample_step` (l.72) makes pieces an even fraction of the finest voxel. Along a coarser axis
whose spacing is not a whole multiple of that step (0.3/0.3/1.0 µm: step 0.15, 1.0/0.15 is not
an integer) a piece straddles a voxel boundary and an n-voxel dropout can read a step off. On
0.5/0.5/2 and 0.98/0.98/1 stacks it is exact. A nit; the docstring (l.73-76) claims more.

### F11. To verify, not yet a finding

- `np.asanyarray(segmentation_mask, dtype=bool)` (assemble.py l.580, skeleton.py l.2068) copies
  a non-bool memmap into RAM. `_to_binary_volume_for_skeletonization` may already return bool;
  check on the BigSM profile (memory: bigsm-stack-profile-2026-10-12) before calling it a defect.
- The secondary-loop reconnect is not gated. It cannot cross background (≥ 90% of the path on
  the skeleton) but can trace a second strand along a Lee sheet in a wide lumen. Whether that
  happens after the thick split took the sheet out: count on E14.5 (Phase 0's report has the
  pairs-in-one-lumen column per step already).

---

## 2. Phases

Each phase lands on its own, in a worktree off the shared tip (memory: shared-checkout), one
commit per phase, the full `not slow and not gui` suite before each (13-28 min), the E14.5
artefact report after Phases 1 and 2. CLAUDE.md's Mask-supported bridging and thick-vessel
paragraphs change in the same commits.

### Phase 0 — Measure F1, and put it in front of Harvey

1. **Name the run.** The screenshot's dataset and config (section 4, question 1). If it is not
   E14.5, Phase 0 runs on both.
2. **Census of missed junctions.** A finder beside `graph/network_review.py`'s (it already has
   "dead ends by kind"): every dead end whose tip lies within (one thick-vessel lumen radius + the
   dropout) of the thick region, with: distance from tip to the region surface and to the tree,
   whether the tip is inside the region, whether the mask runs on from the tip into the thick
   body (`route_through_mask` from the tip to the nearest tree point, judged by the gate), its
   `dead_end_kind`, and which of (a)-(d) in F1 applies, read from the skeletonise stage's own
   records (the arms `_join_thin_arms_to_fat_ridge` refused, with why; the blind vessels
   `_remove_blind_vessels_beyond_bridges` took out). Run through `scripts/graph_artefacts.py`
   so it is measured on the saved skeleton with the run's settings.
3. **Grid.** About 30 of them, spread across tip-to-tree distance and across (a)-(d), shuffled,
   each in its own plane on the **segmentation mask** with the thick region's outline, the tree,
   the dead end ringed, a scale bar and its voxel position; header "J = this vessel opens into the
   big vessel and should be joined, N = not a junction (leave it)". Key CSV under
   `outputs/thick_junction_review/`. Sent to Harvey as an image. (CLAUDE.md, marked grids.)
4. **Forks** (F1b): every graph junction whose arms all lie on the tree, with the inscribed
   radius along each arm to 20 µm, so the branch's stretch inside the trunk's lumen is measured
   rather than assumed. No grid: whether a fork is a junction is not in question, only where the
   bridge ends, and that is a measurement.
5. **Report.** Counts per cause, the all-J baseline, in `outputs/thick_junction_review/REPORT.md`.

Script: `scripts/thick_junction_review/census.py` on `scripts/graph_artefacts.py`'s build
(`outputs/thick_junction_review/build`, the E14.5 skeleton of 2026-10-07 rebuilt at the current
commit; the saved `graph.pkl` predates the dead-end kinds and the facing join).

Exit: the dominant cause is known and Harvey's marks say how many are junctions.

### Phase 1 — Bridge a vessel that stops at a thick vessel's wall

Shape depends on Phase 0's dominant cause; the three repairs below cover (a)-(c), and (d) is a
thick-region gate question for `PLAN_thick_vessel_bridges.md`.

1. **A dead-end stretch inside the lumen is a connector** (cause a, b): in
   `insert_thick_vessel_junction_nodes`, an inside stretch that ends at a dead end within the
   lumen (inside the tree's inscribed radius at the nearest tree point) is bridged from where it
   entered the lumen to the nearest tree point, by the rule connectors already use for a stretch
   that never led out ("else the nearest centreline point, within 2 radii + 5 µm"); the stretch
   inside the lumen is removed as a connector's is. A stretch that ends inside the region but
   outside the lumen stays a graze.
2. **A tip outside the region whose mask runs on into the thick body** (cause c): in the same
   step, before sampling, extend each dead end within (lumen radius + 2 µm) of the region through
   the mask to the region surface with `route_through_mask`, accepted by the gate (one dropout)
   and by the same-lumen test, then treat it as 1. This is `join_dead_ends_to_vessel_sides`'s
   job narrowed to thick vessels, where the target centreline is one tree and the marks from
   Phase 0 say whether the join is right, so it does not wait on G1-G3's 90%.
2b. **A fork of the tree** (F1b): at each junction whose arms are all tree, the arm whose
   inscribed radius falls fastest is the branch; its samples from the fork to where its radius
   drops to its own (below 0.6 of the fork's, the threshold Phase 0's profiles will set) are
   replaced by one bridge from that point to the fork. Two daughters of equal radius both get one.
   Test: a thick T whose branch is itself thick: one bridge, the branch's first edge starting at
   the trunk's wall, the trunk's own edges untouched.
2c. **No bridge along the axis; every bridge ends on a thick edge** (F1c): in
   `_choose_bridges`, a connector whose exit lies along the tree's local axis from the tree's end
   (cos ≥ the threshold Harvey's marks set, 0.7 to start) is the thick vessel tapering: left as
   vessel, its samples kept. A bridge's centreline target is the nearest point *on a thick
   edge's polyline* (the edge split there, `_rebuild`'s `node_for`), never a free sample within
   the tolerance. Tests: a thick tube narrowing into a thin one gets no bridge and one
   continuous centreline; a side vessel's bridge end has degree 3 with two thick edges.
3. **Dead-end kinds.** A tip Phase 1 bridges is no longer a dead end; one it refuses keeps its
   kind, and `dead_ends.py` gains `thick_wall` ("ends at a thick vessel's wall, not joined") so
   the Post processing review lists it and Trace on from the tip starts there.
4. **Tests** (`tests/test_graph_thick_vessel_junctions.py`): a thin tube meeting a thick tube in
   a T, the thin centreline stopping (i) 1 µm inside the region short of the tree, (ii) 1 µm
   outside it with the mask continuous, (iii) 1 µm outside with 3 µm of background between: one
   `IS_ZERO_RESISTANCE` bridge each for (i) and (ii), none and a `thick_wall` dead end for
   (iii); the bridge's length one lumen radius within 20%; no second strand (the lumen-artefact
   report's pairs count 0). Plus the orphan reconnect on the same T: it must *not* join the tip
   to the thick vessel's node 3 µm along (F1's "to nodes only" stays as it is; the junction step
   owns thick walls).
5. **Measure** on E14.5 with `scripts/graph_artefacts.py --ref 040170b --steps`: missed
   junctions before/after, bridges made, dead ends by kind, pairs in one lumen, mask coverage,
   build time. Report in `outputs/thick_junction_review/REPORT.md`. Check the grid's J marks are
   now joined and the N marks are not; report the misses by ID.

### Phase 2 — The gate's small defects, one test each

In one commit, each with its regression test:

| # | Fix | Test |
|---|---|---|
| F2 | `MaskBridges.bridge` takes the two ends' headings (`_terminal_end`) and re-judges the turn on the accepted path's first `GAP_BRIDGE_TANGENT_LENGTH_UM` (5 µm) from each end; a routed path that folds past 80° is refused as the chord would be. `_SkeletonBridgeGate` the same with `min_facing_cosine`. | `test_graph.py`: a U-shaped mask where the chord is straight and the route folds back; refused. Record the two headings per bridge in the Phase 1 report so a marked grid can check 80° later if E14.5 shows any |
| F3 | Docstrings at optimise.py l.59, l.285, reconnect.py l.150 say "the straight line where the skeleton runs along it"; `MaskBridges.bridge` measures the chord once (drop the final candidate when it equals `preferred`). | `test_graph.py`: the support test is called twice, not three times, on a refused pair (monkeypatch count) |
| F4 | When no sample is judged (L < 2(r+1)), judge the path's middle sample alone: shadowed → duplicate. | `test_bridge_mask_support.py`: a 4 µm bridge beside a strand inside a 6 µm lumen is a duplicate; a 4 µm bridge continuing the strand is not |
| F5 | `_bridge_path_through_mask` pads by `ceil(3 / relative)` per axis, as `route_through_mask` does; or better, the skeleton gate calls `route_through_mask` and `_bridge_path_through_mask` goes. | `test_skeleton_bridging.py`: on 0.5/0.5/2 voxels the routing window is 1 slice in z, 6 in-plane |
| F6 | `_hub_link_gate` runs `support.shadows` against the bundle's skeleton as well. | `test_skeleton_bridging.py`: a spoke beside an existing branch in the same lumen is refused |
| F7 | `refused_background` counts tips whose every target was refused; a per-test count stays at debug level; the summary says "N tip(s) refused". | `test_skeleton_bridging.py`: one tip, three refused targets, the log says 1 |
| F9 | A mask not the skeleton's shape raises `ValueError` naming both shapes, at `preprocess_skeleton_for_graph`, `connect_skeleton_components` and `build_graph_from_skeleton`; `robust.4a`'s row cites it. | one test per entry point |
| F10 | `_sample_step` is the largest step ≤ 0.5 µm that divides **every** axis's spacing within 1e-6, falling back to the present rule; docstring states the condition. | `test_bridge_mask_support.py`: a 1-voxel dropout along z on 0.3/0.3/1.0 reads 1.0 µm |

Then `scripts/graph_artefacts.py` against Phase 1's commit: F2 and F4 can change E14.5 (fewer
bridges, fewer pairs in one lumen); the rest must leave it node for node.

### Phase 3 — Settings a loaded skeleton still needs (F8)

1. `bridge_require_mask_support`, `bridge_max_background_gap_um` and `bridge_min_mask_fraction`
   lose `do_skeletonize` from `requires` (both stages read them; the section stays
   `_PIPELINE_STAGES`, the Advanced button anchors on `bridge_require_mask_support`).
2. A schema test: for every setting `graph_build_arguments` and the skeletonise stage read, its
   prerequisite chain contains no `do_*` toggle of another stage. Written generally: each stage
   function's settings (`parameters_of` + the explicit `settings[...]` reads) against the chain.
3. `tests/test_gui_layout.py`: with `do_skeletonize` off the eight rows are still laid out.

### Phase 4 — Re-measure, record, hand on

1. E14.5 and the screenshot's run: the Phase 0 census again, the lumen-artefact report, dead-end
   census, build time. Numbers into `outputs/thick_junction_review/REPORT.md` and this file's
   status section.
2. `RELEASE_REVIEW.md`: `graph.6d` and `robust.6b` rows cite the fixes; `graph.6b` gains the
   thick-junction grid and its fit; `robust.4a` cites F9's tests. Scores from the ladder only.
3. CLAUDE.md: Mask-supported bridging paragraph (the direction rule now on the drawn path, the
   short-bridge doubling test, the shape error), the thick-vessel paragraph (walls), the settings
   bullet. Memory: a note on what the grid showed.
4. Delete nothing here until Phase 4 lands; then this file goes with the other finished plans.

---

## 3. Order, cost, risk

| Phase | Depends on | Work | Risk |
|---|---|---|---|
| 0 | Harvey naming the run; the E14.5 skeleton on disk | half a day: finder, grid, report | none to the code |
| 1 | Phase 0's marks | 1-2 days | changes E14.5's graph; measured before/after; `PLAN_thick_vessel_bridges.md`'s session owns `thick_vessel_junctions.py`, agree the change with it first |
| 2 | — (can run beside 0) | 1 day | F2/F4 change bridges on real data: report the delta; the rest are no-ops on E14.5 |
| 3 | — | 2 hours | panel layout tests |
| 4 | 1-3 | half a day | — |

Phases 2 and 3 can start today. Phase 1 waits for the grid.

---

## 4. Questions for Harvey

1. **Which run is the screenshot?** Dataset, config and the saved skeleton, so Phase 0 measures
   the same vessel. If it was a run without thickness-gated skeletonisation, F1's cause (d)
   applies and the fix is the gate, not the junction step.
2. **Marks** for the Phase 0 grid when it comes (J / N per ID).
3. Should **F2** (the turn judged on the drawn route) go in without a grid? It only ever refuses
   a bridge the chord rule would have drawn; the Phase 1 report lists each one so they can be
   checked.

---

## 5. Status

- 2026-10-13: review done, plan written. Nothing landed.
- 2026-10-13, later: the screenshot is the E14.5 run (`config.yaml`). Phase 0 measured
  (`outputs/thick_junction_review/REPORT.md`, graph rebuilt at 040170b): 127 dead ends within
  the lumen + 10 µm of the thick tree, 91 of them interior; **84 of those end at a wall with the
  mask continuous from the tip to the big vessel** (67 outside the region, 17 inside it), 7
  across background. Cause (c) dominates (a)/(b) four to one. 16 forks of the tree, 2 with a
  bridge; the branch runs 5-18 µm inside its trunk before its radius falls to 0.6 of the fork's.
  Grid W (30 cases) sent to Harvey. **Harvey replaced it** with his own examples: 7 correct
  bridges given, 10 missing bridges and 10 wrong bridges to follow (`REPORT.md`). Phase 1 is
  fitted to those 20 and checked on grid W afterwards (a fresh set it was not fitted on).
- 2026-10-13, Phase 1 implemented on `claude/thick-junctions` (worktree
  `.claude/worktrees/thick-junctions`): the wall join (F1), one taper per thick end left as vessel
  (F1c, judged per inside stretch so a side vessel at the same end keeps its bridge), fork
  bridges (F1b); the run's mask-support settings passed through (`thick_junction_bridge_arguments`).
  E14.5 in `outputs/thick_junction_review/REPORT.md`: 17 wall joins, 22 tapers, 6 forks. Not
  done: "every bridge ends on a thick edge" (F1c second half) and the 24 connectors with no graph
  centreline near -- both wait on positions for Harvey's pictures 8 and 12. The post-processing
  fixture drew a vessel into a thick end along its axis and expected a bridge (the 2026-10-10
  rule); it now enters from the side, and a new test pins the along-axis case as no bridge.
  `test_the_built_network_draws_each_lumen_once` fails on the base commit 9b5d141 too (no short
  loops left to judge since b1f8777), not by this work.
- 2026-10-13, pushed to origin `Devel_GUI_improvements_HD` (cf529d8..260d36a): Phase 1
  (100293f), plus the taper guard in the clean-up, `insert_node_on_edge` cutting an edge as its
  points run (32813bc: the straight jumps across a gap, S16's 35 um), and no wall join closing a
  short loop (abdab60). Fast suite 5,545 passed. E14.5 against cf529d8: bridges 61 -> 63, dead
  ends 549 -> 545, short loops 0 -> 0, mask covered 71.5 -> 72.1%. Still open from section 2:
  Phase 2 (F2-F10), Phase 3 (F8), every bridge ending on a thick edge, the 11 connectors with no
  graph centreline near, and the 18 in-vessel chords of 5-8.5 um (doubled-stretch rewiring).
- 2026-10-14, Phases 2 and 3 pushed (260d36a..c62441a): F2 (turn judged on the path drawn), F3,
  F4 (short bridges judged at their middle, only where a reconnect draws one -- applied to every
  same-lumen test it cost 1.1 points of mask coverage, d5ab09b -> 95c43fa), F5, F7, F9, F8. F6
  dropped (the spokes replace the bundle's own skeleton, against which every spoke reads as
  doubled); F10 documented only (exact on every axis = the spacings' common divisor, 0.01 um on
  E14.5). The skeleton half of F2 not done (reach 4 voxels, window 3). E14.5 against 260d36a: gap
  bridges 52 -> 42, doubled centreline 798 -> 766 um, dead ends 545 -> 532, mask covered 72.1 ->
  71.9%. The 13 new refusals (grid R) all marked N by Harvey. Fast suite 5,554 passed, plus the
  updated fallback test. Still open: every bridge ending on a thick edge and the 11 connectors with
  no graph centreline near (positions for pictures 8 and 12), the 18 in-vessel chords.
- 2026-10-14, grids T and U marked by Harvey (`outputs/thick_junction_review/REPORT.md`): T all 30
  correct, so "every bridge ends on a thick edge" is dropped; U: 9 of 12 rightly left unbridged,
  U02, U05, U12 real vessels missed, and no measure separates them from the nine, so no rule is
  built. Pushed: `RELEASE_REVIEW.md` evidence rows (e5853c0) and the grid scripts (b842cda). What
  remains open is the graph's own thick centreline not following the tree (81% coverage), behind
  the three misses and five of the long strays, and the 18 in-vessel chords.

---

## 6. Two follow-ups (outlined 2026-10-14, awaiting Harvey's go-ahead)

Measured on E14.5, the `gate_pair2` build at 95c43fa (`outputs/thick_junction_review/`).

### 6a. The graph's thick-vessel centreline does not follow the tree everywhere

The tree (`skeletonize_edt_ridge` of the thick region): 2,797 voxels in 59 pieces, radius median
6.0 um. The final graph's non-bridge centreline covers 82.6% of it (84.6% before the split).
Uncovered: 478 um. Of that, 377 um has a graph strand within two lumen radii but off the tree (the
thick vessel drawn along a stray -- collapse moving nodes, the clean-up keeping the wrong one of two
strands, mask recovery adding one), and 101 um has nothing in the lumen. In 98 gaps, median 2 um
(tolerance flicker), 14 of them 10 um or longer holding 278 um; 158 um lies within 5 um of one of
the tree's 82 ends. Consequences: the split bridges only to graph samples on the tree, so the 12
vessels of grid U go unbridged (three real ones missed), and the long strays are five of grid U's
nine N cases; the thick vessel's length, its branch orders and every bridge's landing point follow
the strays.

Proposal, behind a setting off by default until measured: inside the thick region the tree is the
thick vessel. In the split, before sampling, remove every non-bridge edge stretch that runs in the
region along the thick vessel (on or beside the tree, within two radii) and add the tree itself as
graph edges (skan over the tree voxels, nodes at its junctions and ends, joined to the graph where
the tree leaves the region); then the existing connector rule bridges every vessel leaving the
region to it. Guards: nothing cut off (the split's own guard); a thin vessel only grazing the
region stays; tree spurs under the stub prune's rule dropped. Phases: (0) a grid of the stray
strands and the gaps -- is the stray the thick vessel drawn wrongly, or a thin vessel alongside?
(1) build it with tests (a thick tube whose graph strand wanders; one with a gap; a thin vessel
alongside that must survive); (2) E14.5 before/after: coverage, bridges, grid U's three, dead
ends, mask coverage, build time; (3) on by default if the marks agree.

**Status 2026-10-15.** Built (bb14379, off by default), then fitted on Harvey's grid V: the fill
as first drawn was right on 7 of its 20 gaps; filling only where the graph's own way across a gap
is over 2.1 times the tree's path (`TREE_FILL_MIN_DETOUR`, 307a18b) is right on 18/20 against
13/20 for filling none. Stray removal dropped. E14.5: tree covered 82.6 -> 86.0%, bridges 62 -> 63.
Grid U's three are not tree gaps (U02, U12: reach read on a thin tree spur, one failing exit sinks
the connector; U05: a tree end) and are not changed. Turned on by default at Harvey's call (b633f23, 2026-10-15), before a fresh grid; one on
another stack with thick vessels is still owed (CLAUDE.md, marked grids step 6).

### 6b. The straight in-vessel chords

15 straight steps over 5 um (5.1-8.5 um, 95 um in all, 0.25% of the centreline) in 14 edges; 9 at
an edge's end, 6 mid-edge; all inside the mask, at most about a lumen wide. They are the
doubled-stretch step's chords (`_zip`, `_rejoin`: a remainder joined straight to the partner) and
a few collapse ends, made by the clean-up that runs after smoothing, so never smoothed (a
degree-2 merge also drops `centreline_smoothing`).

Proposal: run `smooth_graph_centrelines` once more, on the edges the clean-up made or changed
after the build's smoothing pass, with the same acceptance (within `max_deviation` of the skeleton
or of the path it came from, never longer). Measured on a copy: the 14 edges go 535.0 -> 513.6 um
(-21.4 um, -4%; per edge -0.15 to -6.7 um), the network -0.06%, and no step over 5 um is left. Their
resistances fall in proportion; flows elsewhere move negligibly. Alternatives: densify the chords
only (no length change, still drawn straight), or leave them. Tests: a doubled-stretch rejoin's
chord is gone after the pass, the edge is no longer than before, and an edge the clean-up did not
touch is left exactly as it was.
