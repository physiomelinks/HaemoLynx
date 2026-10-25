# Graph building: consolidation, bridging, recovery, loops, re-trace, smoothing

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## graph/assemble.py

**`graph/assemble.py`** — `build_graph_from_skeleton`; optional `step_callback(G, label)` after each topology step, one per label in `STEP_LABELS` (thirteen of them).
The last, `consolidate_lumen`, repeats until a round changes nothing: degree-2 merging, then
with the mask edges mostly off it, the second of two edges through one lumen
(`lumen_loops.remove_parallel_edges_in_lumen`, which never takes out an edge whose loss would
cut vessels off -- `cuts_off_vessels`), loops inside one lumen, the stretches of an edge
doubled beside another that the whole-edge pair rule leaves (`graph/doubled_stretches.py`:
a dead end cut back to where it leaves its vessel, a fork zipped to where its edges part, a
mid-edge stretch cut out and its remainders joined to the edge kept -- rewired, never
deleted whole, bridges never touched; every run the report counts, since grids H and H2
(2026-10-12) found no mask measure that tells one vessel drawn twice from two touching, and
the doubling only the wider vessel's side sees is left to the review list), dangling bridges,
and the stub prune; "nothing changed" counts the centreline length. `lumen_cleanup(keep_apart=)`
hands both the pair rule and the stretch step a caller's evidence beyond the mask (a raw-image
test, `PLAN_raw_confirmation.md`) for what to leave alone. `lumen_cleanup(along_thick_centreline=)`
(`build_network` passes it whenever the run has a thick-vessel region:
`thick_vessel_junctions.ThickCentreline.along_um`, the µm of a polyline on the region's
centreline tree within the split's own tolerance) makes the pair, in-lumen loop and stretch
rules keep, of two strands in one lumen, the one with 3 um more on the tree
(`lumen_loops._removal_order`, `THICK_CENTRELINE_LEAD_UM`) -- unless the other is a vessel of
its own beside the thick one: its median lumen radius at least
`VESSEL_BESIDE_THICK_RADIUS_UM` (2.3 um) and lying outside the kept strand's lumen
(`_vessel_beside`), when both stay. Judged by the wall alone, the strand on the tree
went as often as not -- an edge running on past the thick vessel into narrower ones averages
nearer the wall -- and the thick vessels came out broken (Harvey's Forclaude.haemorun, E14.5,
2026-10-15: the graph followed 95% of the tree after skeletonisation and 84.7% built).
Calibrated on Harvey's grids W-Z (2026-10-15, `outputs/thick_vessel_breaks/GRID_*.md`): the
first version (c13ea37, the short-loop rule and the stub prune given it too) was right at 11
of 21 marked places, no better than marking all one way; every place it restored 7 um or more
of the tree was right (12/12 over grids W and X), but keeping the tree strand let the pair and
loop rules take capillaries beside it, the short-loop preference and the stub prune's keep
were wrong wherever they decided (grid W), and grids Y and Z (50 removals beside a kept thick
centreline, marked keep or remove) fitted the beside-vessel test, 39/50 left out against 28
for removing all. As now: 73 of 98 marks across the four grids against 55 for the plain build;
the tree followed 88.5% on E14.5, 3 more loops (95 -> 98). Not yet checked on a fresh grid. Run once in
a fixed order, each left work the others never saw (on E14.5 the last merge alone raised the
pairs in one lumen from 94 to 126). `lumen_cleanup` builds what it reads from the build's own
settings, so `build_network` runs it again after smoothing. E14.5 settles in 9 rounds
(`lumen_cleanup_max_rounds`, 12 by default, `CONSOLIDATION_MAX_ROUNDS`): pairs in one lumen 124 -> 9 (all kept by the guard), loops
inside one lumen 12 -> 0, at a cost of 1.9 points of mask coverage and ~35 s. A round after
the first re-judges only what changed: the pair check keeps each edge's verdict between rounds
(`diagnose_parallel_duplicates_in_lumen(memo=)`; an edge is judged again only when it, or a
centreline within the reach it is judged over, was added, changed or removed -- its verdict
reads nothing else), and the lumen-radius lookup graph building makes remembers the voxels it
has measured (`pointwise_distance.FeatureDistance(remember=)`, up to `LUMEN_RADII_REMEMBERED`;
off by default, since the low-RAM blockwise EDT visits each voxel once and would only pay its
memory). On E14.5 the build went 339 -> 255 s with the graph unchanged node for node. Mask
recovery is now the slow step (~134 s): routing its joins through the mask and finding the
uncovered mask, not tracing its pieces (Lee thinning there is ~5 s).

## Mask-supported bridging

**Mask-supported bridging** (`preprocessing/bridge_mask_support.py`, `bridge_require_mask_support`,
on by default) — with the segmented mask, nothing that joins two pieces of skeleton or graph
(closing and `bridge_gaps`, a bundle hub's links, `connect_skeleton_components`, and graph
building's gap, optimise and orphan reconnects through `graph/reconnect.py`'s `MaskBridges`)
is drawn across background: a bridge follows a route through the mask (`route_through_mask`)
crossing at most `bridge_max_background_gap_um` (2 µm) in one run — a dropout in the
segmentation, not tissue between two vessels — with `bridge_min_mask_fraction` of it in the
mask unless all its background adds up to no more than that run. A path beside a vessel
already in the same lumen is refused too (`MaskSupport.shadows`, the ends excluded so a branch
meeting its parent is not one), so no reconnect draws one segmented vessel twice; a bridge
too short to leave anything judged once those ends are left out (in a lumen 6 um round, any
under 14 um) is judged at its middle -- only where a reconnect draws a bridge
(`judge_short_at_middle`: `MaskBridges`, the skeleton gate, the thick-vessel wall join), not in mask recovery or the
doubled-stretch rule, where on E14.5 it cost 1.1 points of mask coverage. The gap, optimise
and orphan reconnects judge the 80-degree turn on the path drawn, not the chord
(`MaskBridges.bridge(ends=, max_turn_deg=)`): on E14.5 that and the middle sample refused 13
more gap bridges, all 13 marked right by Harvey (grid R, 2026-10-14). A segmentation not the
skeleton's shape raises (`check_mask_shape`), and the three bridging settings require no
stage, so a run loading its saved skeleton keeps them;
`graph.diagnose_lumen_artefacts` reports any pair that does -- with loops inside one lumen and
dead ends by kind, each judged as its whole chain to its junction as the stub prune judges it
(short by the chain's length; its tip in another lumen only if not in one of its own chain's,
a thick-vessel bridge included) -- in `build_network`'s graph checks, on the graph that is saved. A graph bridge records `bridge_kind` and `bridge_background_um`. Closing and gap
filling held to the mask can grow a thin vessel into a rod Lee thinning erases outright, so a
piece of skeleton thinned away entirely is put back as it was. The stub prunes judge a dead end
as its whole chain, tip to the junction it hangs from (length summed, the parent radius read at
that junction; piece by piece, a long dead end in short pieces was worn away), and read the mask
too (`prune.prune_vascular_stubs`): a stub mostly off it, ending inside its parent's lumen, or
never more than its own radius outside another centreline's lumen (`STUB_PROTRUSION_RADII`, with
`lumen_radius_at`: a hair along a vessel's wall, which the tip-to-junction test misses)
goes whatever its length, and one whose tip the mask runs on past is held to
`min_stub_length` as well, while a blind sprout keeps the radius rule
(`min_stub_length_radius_multiple`, 3 radii by default: at 1.5, 3 um on a capillary, Lee's
blind spurs stayed -- 1,037 of E14.5's 1,234 dead ends).

**Bridging loose ends (2026-10-25, `PLAN_bridging.md`).** The skeleton bridges now judge their
facing (`skeleton_bridge_min_facing_cosine`) on the route drawn, read at the point the chord's length
along it, as the reconnects judge their turn (`6e09639a`); on E14.5 (Harvey's Claudin segmentation)
the step draws 1 bridge from 957 branch ends, none refused for it, and the skeleton and graph are
the base's bit for bit. The bridge gate's sample step divides every axis's half spacing where a
step down to a quarter of the finest axis's does (`e87c08e1`): on 0.3 x 0.3 x 1 um voxels a
one-slice z dropout read 0.86-1.14 um by the path's vertices and now reads 1.0; E14.5's 0.98 x 0.98
x 1 um keeps 0.49 um (the half spacings' common divisor is 0.01 um). F11 closed by measurement:
the masks skeleton cleaning and graph building receive are bool already (in RAM, and a bool
memmap under the low-RAM option, on the Nerve fixture), so `np.asanyarray(mask, dtype=bool)`
copies nothing, and graph building is handed the run's `MaskSupport`, so `assemble.py`'s two
conversions never run; the ungated secondary-loop reconnect adds 3 edges on E14.5 (none left
with its mark in the saved graph) and none on the Nerve fixture.

## graph/mask_recovery.py

**`graph/mask_recovery.py`** — `recover_uncovered_mask_vessels` (`recover_uncovered_mask_vessels`,
on by default; its own step after the orphan reconnect, then `consolidate_lumen`):
the mask no centreline's lumen covers (`uncovered_mask_voxels`: within a sample's radius, or
inside its lumen's cross-section, followed out in 32 directions across the vessel to the wall,
at most `CROSS_SECTION_REACH_RADII` = 4 radii, and only while the mask is also there a radius
ahead and behind -- so a flattened lumen's far side is covered, while a branch narrower than
its vessel stops the outline at the wall; on E14.5 79% of the mask the radius rule alone left
uncovered was that far side, and recovery traced it as second strands), in pieces of at least
`recovery_min_region_volume_um3` and wider than a voxel and a half, traced along the run's own
skeleton (the build passes it: `skeleton=`; a piece the skeleton does not run through is left
alone, and only a caller passing none Lee-thins the mask again -- on E14.5, 2026-10-13, two
thirds of what re-thinning drew lay where the skeleton has nothing, combs across wide lumens
the clean-up then took out) and joined to
the network through the mask where an end lies within `final_orphan_reconnect_threshold` of a
covered lumen. Never an island, never across background, never beside an existing vessel,
and never a loop inside one lumen: after a strand's first join (nearest end first), a further
join, or a traced path closing a ring, is kept only when the loop it closes runs round tissue
-- its chords cross more than a dropout (`bridge_max_background_gap_um`) of background
(`lumen_loops.loop_inside_one_lumen`). Joining every free end made tangles of short loops (on
E14.5, 3,232 under 30 µm against 101 before the step). Its duplicate check and node ids are
kept indexed (`EdgeSampleIndex.added_near`, `_NodeIds`): scanned per path, they were 95% of
the step's time. The join loop reads each strand's edges from the strand's own nodes, keeps
the samples recovered so far in trees that double in size (`_RecoveredSamples`: loose until
`_LOOSE_SAMPLES`, then merged) rather than one tree rebuilt per strand, and groups the
uncovered mask by labelling a slab of slices at a time (`_components`) rather than a KD-tree
pair search: on a 43,000-edge skeletal-muscle crop the whole-graph scan and the rebuild were
three quarters of the step, and the pair search holds 13 pairs a voxel, gigabytes on a large
stack. The cross-section outline (`_follow_rays`) follows each ray only to its first
voxel out, with the samples sorted by reach so a chunk is not kept going by its widest vessel,
reads the mask as bits (`preprocessing.PackedMask`: 2 GB for a 16 G-voxel stack, in RAM, which
every `MaskSupport.inside` switches to after `PACK_AFTER_POINTS` lookups) and spreads its chunks
over worker processes (`preprocessing.shared_volume_tasks.run_volume_tasks`, the pool the
section diameter methods use) once there are `_WORKER_MIN_SAMPLES`; the outline is per sample
and identical with any of that (E14.5's uncovered scan went 108 -> 18 s, bit for bit). Without
a mask, or with both settings off, the two steps leave the graph as it is.
`recover_uncovered_mask_vessels(coverage_reach_radii=, two_ended=)` (2026-10-21): the first is
`uncovered_mask_voxels`' cross-section reach (0: a centreline's own radius covers), the second
takes recovered edges left hanging from a dead end out again until none is
(`keep_two_ended_recovery`, `G.graph["recovered_one_ended_removed"]`); both for
`remove_interior_dead_ends` -- after removing E14.5's interior dead ends, recovery as it is
drew 1,376 of its 2,024 edges as dead ends.

## graph/facing_ends.py

**`graph/facing_ends.py`** — `join_facing_dead_ends`, its own step after mask recovery
(`join_facing_dead_ends`, run with a mask and `bridge_require_mask_support`;
`facing_dead_end_max_gap_um`, 10 µm, 0 off): two dead ends pointing at each other -- the
straight join within `FACING_MAX_TURN_DEG` (60°) of each end's heading, read along its whole
chain back to the junction -- are joined across up to that gap, though it is more than a
dropout, because one vessel broken by the segmentation is the one case where crossing more
background is the vessel. One stretch of background only (never a third vessel), each end at
least 3 µm long, the gap no longer than the two together, not beside a vessel in the same
lumen, closest pairs first. Rare by design: of E14.5's 284 dead-end pairs within 10 µm, 143
were siblings off one junction, 57 side by side and only 15 pointed at each other.

## graph/lumen_loops.py

**`graph/lumen_loops.py`** — `remove_loops_inside_one_lumen`, run in every round of
`consolidate_lumen` whenever `bridge_require_mask_support` has a mask:
Lee thinning keeps the mask's topology, so every tunnel through a vessel comes back as a ring,
and graph building closes more loops inside one segmented vessel. Each edge's shortest loop
(within `LOOP_SEARCH_UM`) is tested by `loop_inside_one_lumen` -- at least half its antipodal
chords crossing no more than a dropout of background -- and one inside a lumen loses its arc
(the run of edges between two places the network meets it) nearest the wall, so the vessel's
middle is what stays and no spur is left; a ring the network meets once goes whole. A loop
round tissue stays however short. Removing the longest arc instead kept rim strands and left
80k more voxels of E14.5 uncovered.

## Short loops

**Short loops** (`lumen_loops.remove_short_loops`, `remove_short_loops_under_um`, **200 µm, on by
default** since 2026-10-13) — in every `consolidate_lumen` round, after the in-lumen loop step,
every loop up to that long round loses its arc nearest the wall whatever the mask says of it.
Harvey's grids I and I2 (`outputs/loop_review/FIT_I*.md`, judged on the claudin-5 channel):
55 of 60 short loops of E14.5 close inside one vessel, the in-lumen rule found none of them
(the background it reads between the two sides as tissue was not tissue there), and no
measure of the mask or the raw image found the five real ones (46-179 µm) -- so the rule is
the trivial one, right 92-93 % at the cost of one real loop in twelve. Except round a wide
hole (`short_loop_keep_hole_over_um`, **10 µm** by default since 2026-10-15): a loop whose
hole -- `lumen_loops.loop_hole_um`, the median background its antipodal chords cross,
`loop_measures`' `chord_bg_median_um` -- is wider is left alone. Grids Q and Q2
(`outputs/loop_break_review/FIT_Q*.md`, 60 of the rule's breaks on E14.5, each drawn with
the side it took): where a side should go it took the right one every time, but most
loops round a wide hole were real, and breaking one merged it with its neighbour into a
longer loop that was broken in turn (44% of the first pass's breaks). Over 10 µm 29 of 32
were real, under it 24 of 28 false: 53/60 leave-one-out against 33 all one way. Fitted on
Q (9.2 µm), held on Q2's big holes (21 of 26 real); grid I/I2's "real or false?" marks had
called 6 of 9 big-hole loops false, and the six shown again blind in Q2 all came back
real. E14.5 skeleton reach 68.5 -> 70.5%, 40 short loops left for the review list.
0 breaks every short loop. The Short loops review list shows what is left, and 0 for the
length leaves every short loop to it. `preprocessing/raw_evidence.py` (`RawEvidence`: the
raw channel read in windows where the mask is silent, calibrated once per build and kept on
`G.graph["raw_evidence"]`) shipped no rule: on grids H, H2 (one lumen or two), M, M2 (a
vessel across a gap) and I, I2 (loop holes) no raw measure came near 90 %, and most did no
better than marking everything one way (`PLAN_raw_confirmation.md`; on H the run's own
dextran brightness reached 21/30, the mask's best, against 19). See *The raw image in graph
building* below.

## graph/skeleton_retrace.py

**`graph/skeleton_retrace.py`** — `retrace_lost_skeleton` (`retrace_lost_skeleton`, **on by
default** since 2026-10-18, after grid N2; no effect while `short_loop_keep_hole_over_um` is 0,
which keeps no loop for its hole; `PLAN_surface_coverage.md`): after `build_network`'s
last clean-up (so no rule takes it again) and before `smooth_straight_steps`, a stretch of the
run's skeleton the graph lost is drawn back along its own skeleton when the loop it closes with
the graph runs round a hole wider than `short_loop_keep_hole_over_um` (the same setting and the
same measure, `lumen_loops.loop_hole_um`). A stretch is a 26-connected run of skeleton no
centreline comes within 3 um of, with at least 20 um of skeleton (`piece_length_um`, so a
coarse z axis counts length, not voxels), half of it in the mask (80% refused 4 of the vessels
Harvey drew as missing, at 69-78%), half of it further than two
lumen radii + 3 um from every centreline, meeting reached skeleton in two places; it is drawn
between the two contacts farthest apart through it (not the whole tangle), joined by splitting
the nearest vessel (never a bridge). Only the two short joins face the bridge gate: the route
between is the run's own skeleton, which crosses 3-10 um dropouts of E14.5's raw mask along
vessels grid N marked V. Never beside a vessel in the same lumen (`duplicates_existing_vessel`).
Each further place the piece meets the graph is then joined along its skeleton under the same
tests, the hole read on the loop that join closes (`_join_place`, edge mark `retraced_join`):
every join Harvey added on grid N2 ran that way, but the hole test draws only 1 of his 4 -- a
join closing a loop round the large vessel's own lumen reads a hole of ~0. Edges carry
`retraced` and `retraced_hole_um`, and `build_network` smooths them.
`judge_lost_skeleton` says, read-only, what it would do with each piece and why (`Retrace`,
the `REFUSED_*` reasons). Harvey's grid N (2026-10-18, 30 of the pool): every piece round a
hole over 10.3 um was a vessel (9/9), 17 of 21 round a smaller one were not; at the 10 um
limit (fitted on Q/Q2, not N) 25/30 against 17 for all X; the pair and in-lumen loop rules,
which never read hole size, had taken 12 of the 13 V. Grid N2 (the 21 vessels it drew on
E14.5): 19 V, 1 X, 1 unsure; 10 of the 11 at no grid N place V; 6 of the 7 a pair in one lumen
by the mask V. A* routes (Add vessel's mask and raw tracers) matched Harvey's drawn routes no
better than the skeleton (48% vs 41-42% of his lines), so the skeleton route stays. Harvey then
drew every vessel the surface still lacked (68 strokes, 1.7 mm): 62% of it has skeleton under
it, which the hole and separation tests leave out (the drawn pieces mostly close loops round
holes under 10 um), and 38% has none, beside the large vessels (the thick-vessel gate); of 504
lost surface pieces only the mask share separated cleanly, so it is half, not 80%
(`outputs/surface_review/missing_*.py`). On E14.5 (the 2026-10-16 run): 35 re-traced edges,
1.35 mm, 49 s; skeleton reach 71.3 -> 73.6% (67.0 -> 70.0% on the surface beside the large
vessels), mask covered 74.5 -> 75.4%, 30.8% of Harvey's missing length covered (26.1% at 80%);
pairs in one lumen by the mask 4 -> 14, short loops 42 -> 63, dead ends unchanged.

## graph/smoothing.py

**`graph/smoothing.py`** — `smooth_graph_centrelines`: takes the voxel staircase out of each
centreline and **re-measures `length`**, which moves every resistance (a path stepping voxel to
voxel comes back ~7% longer than the vessel it traces). It is *not* one of the `STEP_LABELS` —
it runs in `pipeline/stages.py`'s `build_network`, after the topology steps and before the graph
is pickled — so a caller assembling a graph by hand gets no smoothing unless it asks.
`build_network` then runs `consolidate_lumen` again on the smoothed curves (smoothing alone
turned E14.5's 2 loops inside one lumen into 12), and once more after the thick-vessel split
(the split alone made 7, through its bridges): nothing is the last step. The clean-up knows a
bridge -- never merged with a vessel at a degree-2 node, a pair with one left to the loop
step, and one left dangling by a removal taken out
(`thick_vessel_junctions.remove_dangling_bridges`) -- and the split judges a flicker in
microns, not samples (a 1.5 um dip out of the region, sampled every 0.49 um, came back as a
vessel between two bridges). A smoothed
path is only accepted if it stays within `max_deviation` of a skeleton voxel and is no longer than
the path it came from; each edge records which of `smoothed` / `relaxed` / `kept_raw` /
`too_short` happened to it. Where the path it came from is itself off the skeleton -- a straight
bridge the graph build added across a gap, or the first step off a node cluster collapse moved --
the distance is measured from that path instead; judged against the skeleton alone, every edge
with a bridge was kept raw.

## The raw image in graph building

`PLAN_raw_confirmation.md`. `preprocessing/raw_evidence.py` reads the run's raw image in windows
where the mask is silent, calibrated once a build (`stages._record_raw_evidence`, after the graph
and the dead-end kinds) and kept on `G.graph["raw_evidence"]`. No rule reads it.

**What a build pays for it (2026-10-25, E14.5, at `8204478d`).** `build_network` from the
thick-bridges session's capture of Harvey's Claudin segmentation, twice, each in a fresh process
(`outputs/raw_confirmation/budget.py`, git-ignored, in worktree `plan-raw-confirmation-0887be`):
without the raw image 211.9 s and a peak of 4,912 MB; with it (claudin-5, the composite's channel
1) 209.8 s and 4,913 MB. The raw step, loading the channel and calibrating it, took 1.2 s. It holds
the channel (uint16, 144 MiB) while it calibrates and then lets it go. The process peak was 2.2 GB
when the step started and did not rise during it; the build's peak comes later. The plan's
budgets (decision D6: +5 % build time, +50 MB on E14.5) hold: +0.6 % and +1 MB of peak. The
"+50 MB resident" was written for windows and samples; read in RAM the channel itself is three
times that, for the second it takes, and under the build's peak. No window is read, since no rule
asks. The budget for the 16.4 G-voxel skeletal-muscle stack (≤ 200 MB, ≤ 10 s of calibration)
is not measured: its segmentation is on this machine, but its raw image only as a 1,000 x 1,000
crop at an unknown offset. `scripts/graph_artefacts.py --raw-cost` makes the same comparison for
any config with a raw image, in its report.

**Uncovered mask, grid UR1 (2026-10-25).** Mask recovery is handed 1,749 pieces of uncovered
mask on E14.5 (the graph after the orphan reconnect, its own two filters) and draws a
centreline into 39 of them; in the finished graph a median 78 % of a piece is still uncovered.
The dextran channel (not the segmentation's own here: it was made from claudin-5) reads mostly
depth: the top 30 um a median 1.27 of the run's lumen level, every band below 0.31-0.41, while
read over the centrelines within 25 um it reads a median 0.42-0.61 in every band. E2
(`inside_walls`) finds no lumen at the deepest points of 1,724 of the 1,749. Grid UR1 (30: ten
recovery draws into, twenty it does not, each half by depth, spread across the local reading;
`scripts/uncovered_review/build_ur1.py`), marked by Harvey the same day: 26 H (not a vessel of
its own), 4 V. No measure beats all H over the 30 (`fit_ur1.py`, `FIT_UR1.md`). Of the ten recovery
draws into, 8 are H, and the dextran over the neighbouring centrelines' above 1.05 separates them
10/10 leave-one-out on 2 V. Grid UR2 (the other 29 pieces recovery draws into) checked that and
"every piece H" (recovery drawing nothing), both fixed before its marks: Harvey marked 12 V,
17 H, and the rules got 16/29 and 17/29, all one way 17/29 (`FIT_UR2.md`). Over all 39 pieces
recovery draws into, 14 are vessels of their own. UR2's best measure, a piece more than 9.15 um
from the nearest centreline, 23/29 left out, does worse than all H on UR1 (23/30 against 26);
over the 39, 31 (79 %). So recovery draws what it can, as before.
