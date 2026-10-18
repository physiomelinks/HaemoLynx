# Plan: the surface plexus and the vessels beside the large ones

Written 2026-10-18 against `Devel_GUI_improvements_HD` at **1e66bce**, from Harvey's E14.5 MCA run of
2026-10-16 11:17 (`C:/Users/hd01/outputs/..._graph.pkl`, run manifest: 1e66bce, clean tree). It builds
on `PLAN_clean_graph.md` (the clean-up rules and their grids), `PLAN_thick_vessel_bridges.md` /
`PLAN_bridging.md` §6 (the thick-vessel tree, the bridges) and the graph-gap reports
(`outputs/graph_gaps/REPORT.md`, `ref_75564ff/REPORT.md`), and does not repeat what they measured.

Harvey's report (2026-10-16): the network looks better, but on the surface layer, in line with the
large vessels, many skeletonised vessels are not covered by the graph. Two kinds, from his screenshots:

- **A. The skeleton is there and the graph does not follow it** -- e.g. an edge that loops back,
  where the graph could follow the centreline and make the missing edges.
- **B. There is no skeleton** -- segmented vessels beside the large arteriole with no centreline at all.
- (2026-10-18, a third screenshot) **branches with a skeleton but no vessel**, beside the arteriole:
  case A again. The skeleton there is drawn as bands two or three dots wide, as Lee leaves a
  flattened lumen.
- (2026-10-18, a fourth) a long skeleton strand running beside the large venule mask, covered
  by the skeleton but by no graph vessel: case A, the near-large stratum of grid N.
- (2026-10-18, a fifth) a vessel beside the large arteriole that does not connect to it, because
  the stretch of segmentation joining them has no skeleton: case B, a dropped join (section 1.3).
- (2026-10-18, a sixth) three skeletonised vessels in the plexus missing from the graph, each
  joining places the graph reaches round holes that look 6-14 um across: case A, sides of loops
  near the short-loop rule's 10 um hole threshold (section 1.2).
- (2026-10-18, a seventh) a broad stretch of segmentation between holes, missing from both the
  skeleton and the graph: mask no skeleton covers, nearly always a wide flat sheet beside a
  centreline (section 1.3, grid O).

Everything below is measured in `outputs/surface_gaps/` (git-ignored): `harness.py` rebuilds the run
from its own skeleton and settings with a snapshot after every topology step and clean-up sub-step
(359 s; it reproduces the GUI graph exactly, 1,752 nodes / 1,683 edges / 40,508 um), `attribute.py`
and `lost_pieces.py` attribute what the graph lost, `no_skeleton.py` and `plain_vs_gated.py` find
mask with no skeleton, `surface_overview.png` draws it. "Reached" = a graph centreline within 3 um
of a skeleton voxel; "far" = more than 2 x lumen radius + 3 um from every kept centreline, i.e. a
separate tube by the mask; "near large" = within 10 um of the large arteriole/venule masks.

---

## 1. Where the misses are

### 1.1 The surface slab is two thirds of the mask, and the graph reaches least of it there

The stack is 287 x 512 x 512. **z 0-39 holds 2.05 M of the 3.07 M mask voxels (67%)**: the
perineural plexus and the pial large vessels (every large-mask voxel but ~300 lies at z < 40).

| place | skeleton voxels | reached by the graph | lost for good |
|---|---:|---:|---:|
| surface (z < 40), near large | 23,139 | **67.0%** | 7,641 |
| surface, other | 16,821 | 68.6% | 5,282 |
| deep (z >= 40) | 33,338 | 75.8% | 8,077 |
| all | 73,298 | 71.3% | 21,000 |

By 10-slice bin the reach is 64-74% over z 0-39, 73-92% over z 40-199, and 64-80% again over
z 200-289, the second, deeper plexus (no large vessels there).

### 1.2 Case A: what takes skeleton the graph had (surface, near large; voxels, [of them far])

| cause | near large | surface other | deep |
|---|---:|---:|---:|
| pairs in one lumen | 1,743 [1,055] | 1,298 [647] | 1,012 [365] |
| short loops (200 um, hole <= 10 um) | 1,570 [773] | 1,113 [555] | 920 [390] |
| loops inside one lumen | 1,330 [687] | 825 [389] | 1,327 [571] |
| cluster collapse | 1,002 [300] | 1,037 [269] | 2,398 [662] |
| stub prune | 949 [321] | 383 [92] | 1,439 [614] |
| thick-vessel split | 476 [270] | 6 | 0 |
| smoothing, degree-2, off-mask, stretches, other | 571 | 620 | 981 |

Near the large vessels the three one-lumen rules take 4,643 of the 7,641 voxels, and **2,515 of
those are far** -- at the end they are a separate tube from everything the graph kept, though each
was removed as one of two strands in one lumen.

As pieces (26-connected runs of lost skeleton, `lost_pieces.csv`): near the large vessels, **45
pieces of >= 20 voxels, 3,789 voxels, are far paths joining two places the graph reaches** -- half
of all that place loses. Across the stack the pool such a piece defines (far, >= 20 voxels, touching
the graph, >= 80% in the mask) is **72 pieces, 5,274 voxels**: 3,103 near large, 1,182 other surface,
989 deep; by cause in-lumen loops 2,137, pairs 1,558, short loops 1,236, prune 141, split 140.
64 of the 72 are paths between two reached places and 8 are branches (one contact).
32 of the 72 were beside another strand when removed (the strand they were judged against went
later: a cascade); the other 40 were already a separate tube by the mask when a one-lumen rule took
them -- the mask joins them (no background between), which on the surface plexus, where vessels
are fused into sheets, is not the same as being one vessel.

**The loops they would close** (`surface_review/n_loop_measures.py`, all 72; the piece from one
reached place to the farthest other, then the final graph's shortest way back, measured with
`lumen_loops.loop_measures` as the clean-up measures loops): 67 close a loop, 1 joins two
components, 4 are branches. 24 of the loops are inside one lumen by the mask (hole ~0); the hole
quartiles over all 67 are 0 / 4.3 / 11.2 um, and 48 are loops the short-loop rule would break
now (<= 200 um round, hole <= 10 um). The rest close loops round wider holes, up to a 531 um loop
round a 65 um hole taken by the pair rule, which never reads hole size. For many pieces the
straight line to the nearest kept centreline crosses no background at all (median 0 um), though
they are more than 2 r + 3 um from it: the mask joins them as one sheet. Grid N spreads over
this: 10 cases inside one lumen, 9 round holes of 3-10 um, 10 round wider holes or longer than
200 um, 1 with no loop. These columns are in `N_key.csv`.

Mask recovery re-reached 2,175 voxels the collapse and prune had dropped (921 on the surface);
855 are lost again in the clean-up.

### 1.3 Case B: what the thick-vessel gate never skeletonises

Plain Lee thinning of the hole-filled mask (no thick gate, no skeleton clean-up) against the run's
skeleton: **5,246 plain-Lee voxels in 183 pieces of >= 10 voxels lie more than 3 um from any
skeleton voxel, and 4,905 of them are on the surface near the large vessels**: 3,796 inside the
thick region, 877 in the 3 um shell beyond it, 204 at 3-6 um. Elsewhere it is 106 (other surface)
and 235 (deep): the skeleton clean-up costs little; the gate is the cause.

Why: `skeleton_thick_vessel_restrict_to_mask = large` makes every segmented voxel inside the large
arteriole/venule masks the thick body (`thick_vessel_object_mask`: the restriction is
authoritative), plus a 3 um wall (`skeleton_thick_vessel_wall_absorption_um`): 765,911 voxels, 333 k
of them outside the masks. That region is drawn only as the EDT-ridge tree, and thin voxels within
`skeleton_thick_vessel_flake_filter_um` (3 um) of it are never Lee-thinned
(`skeletonize_thickness_gated`'s `lee_crop`). So **a capillary inside a large-vessel mask, or within
~6 um of one, has no skeleton**, however long it runs beside the large vessel -- in
`surface_overview.png` (yellow) the missing centreline lies along and inside the arteriole (red)
and venule (green) masks.

**Most of them are joins.** Of the 157 such pieces on the surface beside the large vessels (grid
P's pool, `surface_review/missing_joins.py`), 128 run from the run's thin skeleton (within the 3 um
the comparison leaves out, plus a voxel and a half) into the thick region: plain Lee would join a
thin vessel to the thick one there (median 5.9 um of it outside the region, quartiles 3.0-10.9).
The gated skeleton's own arm join (`_join_thin_arms_to_fat_ridge`) usually joins that vessel
somewhere else, but at 21 of the 128 the final graph has a dead end within 6 um of where the thin
vessel meets the piece (`join_dead_ends.py`): the vessel stops short of the large one, as in
Harvey's fifth screenshot. That count misses a join dropped from a vessel that runs on, which
leaves no dead end. 23 of grid P's 30 cases are joins, 3 of them with a dead end (P-06, P-07,
P-23); both columns are in `P_key.csv`.

What the measure cannot say: how much of the 3,796 voxels is Lee's medial sheet inside the large
vessel's own lumen -- exactly what the gate exists to remove -- and how much is capillaries lying
on or beside it. The venule mask on the left of the overview is a wide sheet, and plain Lee draws a
mesh through it. That is a call for marks (section 3, grid P).

**Mask no skeleton covers, away from the large vessels** (Harvey's seventh screenshot: a broad
stretch of segmentation between holes with neither skeleton nor graph; `no_skeleton_shape.py`).
By mask recovery's own cover rule (a sample's radius, and its lumen's cross-section out to 4
radii), 1,184 pieces of mask (490 k um^3, 17%) have no skeleton through them. Almost none is a
vessel with no centreline: in only 3 of the 307 pieces on the surface away from the large
vessels (306 of 107 k um^3), 7 of 239 beside them and 36 of 638 deep is the piece's widest point
more than 2 r + 3 um from every skeleton voxel. The rest lie beside a centreline that already
runs in the same lumen, and stretch out across it: wide flat sheets of segmentation, about
4 um thick (the inscribed radius, ~2 um, reads the thin dimension) and tens of um across, with
one centreline down them. Whether such a sheet is one flattened vessel or several plexus vessels
fused by the segmentation is the question grids H/H2 could not answer from the mask; the dextran
might (two bright ridges across a fused sheet, one across a flat vessel). A grid of these sheet
pieces (grid O, below) is the way to find out, after N and P. One caution for the screenshots: if
a view is a single slice, a centreline a slice or two above or below is not drawn there.

---

## 2. Principles (from the plans this builds on)

- **Rules that decide what a structure is follow marks** (CLAUDE.md, marked grids): ~30 numbered
  cases, drawn on the mask in their own plane, shuffled, a key under `outputs/<topic>_review/`, the
  fit reported against marking everything one way, then a fresh grid before it is relied on.
- The history says caution: grid S (2026-10-13, lost skeleton pieces over the whole stack) came back
  9 V / 21 X with no measure beating all-X; grids H/H2 found no mask measure for one vessel vs two
  touching. The new grids therefore sample only the pools defined above (surface, far paths; Lee
  strands under the gate), which no grid has judged.
- **Draw along the skeleton, never across background**: every join goes through the one gate
  (`MaskSupport`, `bridge_max_background_gap_um`), with the loop guards the clean-up already has
  (`loop_inside_one_lumen`, the 10 um hole rule) and nothing beside a vessel in the same lumen.
- **Nothing the clean-up undoes**: a re-traced strand is added after the last clean-up, or marked so
  the clean-up after it leaves it alone (as `THICK_TAPER` and bridges are), not re-judged by the
  rule that took it.
- **Off by default** until a fresh grid confirms (as `thick_vessel_extend_centreline_ends`), and
  after any default change, `-m slow` on `test_pipeline_invariants` and the integration graph tests
  (the 7cc17cf lesson).

---

## 3. Phases

### Phase 0 -- Make the census a report, and anchor Harvey's examples (small)

- Fold `surface_gaps/attribute.py`'s "reach by place" into `scripts/graph_artefacts.py` (a
  `--by-place` section: z slab, distance to the large masks, the thick region), and add the
  plain-Lee comparison as a skeleton-stage diagnostic beside `diagnose_vessels_missing_from_skeleton`
  ("centreline the thick gate drops", um, near/inside the large masks). Tests on a synthetic thick
  tube with a capillary 4 um beside it.
- Harvey to give the voxel positions of his seven screenshots (the loop-back edge, the branches
  beside the arteriole with no skeleton, the branches with a skeleton but no vessel, the strand
  beside the venule, the vessel that does not connect to the arteriole, the three plexus
  vessels, the sheet with neither), so both grids
  include them as named cases. They could not be found from the pictures: matched against every
  mask slice and every slab projection in all eight orientations
  (`surface_gaps/locate_screenshot.py`, `locate_projection.py`), nothing correlates better than
  chance, so the views are probably a rotated 3D camera. The cursor position napari shows in its
  status bar, or a Points layer saved over each place, is enough.

### Phase 1 -- Grid P: the centreline the thick gate drops (case B)

30 plain-Lee strands from the 183 pieces, spread over inside-the-large-mask / wall shell /
flake shell and over the candidate measures, shuffled. Each drawn on the mask in its own plane,
with the large-mask outline, the thick tree, the graph nearby, the dextran channel beside it, a
scale bar and its voxel position. Question: **V** -- a vessel of its own, draw it; **X** -- the
large vessel's own lumen or wall, leave it. Key measures: the strand's distance from the tree less
the tree's local radius (outside the trunk's lumen?), its median lumen radius, its length, whether
it leaves the thick region at both ends, background between it and the trunk, its fraction inside
the large mask. Prior: the beside-vessel test the clean-up already uses (`_vessel_beside`: radius
>= 2.3 um and outside the kept strand's lumen, fitted on grids Y/Z).

Build, if a measure fits: `skeleton_thick_vessel_keep_side_vessels` (off by default) -- Lee-thin
the thick region and its flake shell too, keep only the Lee paths the fitted test calls a vessel of
its own, and join them to the tree through `_join_thin_arms_to_fat_ridge` as any thin arm. The
region and its cache are unchanged (only the skeleton is). Tests: a thin tube lying along a thick one
inside its mask gets a centreline; the thick tube's own medial sheet does not.

A narrower fix for the dropped joins alone, if the marks call the joins V: where a thin arm's
end stops within the flake shell and the arm join made no bridge from it, take plain Lee's path
from that end to the tree (through the mask, under the bridge gate) as the join. It touches only
the 128 places, not the large vessels' own lumens.

Cheaper lever measured first, in the same session: wall absorption and flake filter at 1.5 um each
instead of 3 -- how much of the 4,905 voxels comes back, at what cost in wall flakes (grids W-Z's
flake cases).

If the marks say the large masks themselves cover capillaries (the venule mask's sheet), that is a
mask problem, not a gate problem: decision D2.

**Marked 2026-10-18: not built** (Status below, `FIT_P.md`). The narrower join fix stays open:
grid P did not ask whether a thin vessel should connect to the large one.

### Phase 2 -- Grid N: skeleton paths the clean-up took (case A)

**Marked 2026-10-18: build the re-trace with the hole gate** (Status below, `FIT_N.md`): a pool
piece is drawn back when the loop it closes with the final graph runs round a hole wider than
`short_loop_keep_hole_over_um` (10 um, the same setting), along its contact path only.

**Built 2026-10-18, uncommitted** (`graph/skeleton_retrace.py`, setting `retrace_lost_skeleton`,
off by default; `tests/test_skeleton_retrace.py`, wiring in `tests/test_build_network_stage.py`).
Two changes from the plan, both measured: the bridge gate judges only the two joins to the graph,
not the skeleton route between (E14.5's skeleton crosses 3-10 um dropouts of the raw mask along
5 of grid N's V pieces, which the full gate refused), and the size limit is 20 um of skeleton,
not 20 voxels (a coarse z axis). On grid N it agrees with Harvey 25/30 (wrong N-16, N-21, N-25,
N-29, N-30), as the hole measure did. E14.5 (`surface_review/measure_retrace.py`, the rebuild
before its last smoothing step, re-traced and smoothed as `build_network` does):

| | run | re-traced |
|---|---:|---:|
| vessels drawn back | | 21 (1.09 mm; 2 of them pieces of 15-19 voxels outside grid N's pool) |
| skeleton reach, all / surface beside large / surface other / deep | 71.3 / 67.0 / 68.6 / 75.8% | 73.2 / 69.4 / 72.3 / 76.2% |
| mask covered (recovery's cover rule) | 74.5% | 75.2% |
| pairs in one lumen (mask) | 4 | 12 (7 with a re-traced vessel) |
| doubled centreline | 935 um | 1,104 um |
| short loops (<= 200 um) | 42 | 54 |
| loops inside one lumen / interior dead ends | 0 / 531 | 0 / 531 |
| time | | 34 s |

The new pairs are the call grid N says the mask gets wrong on this pool (a re-traced vessel
beside a shorter kept one through a fused sheet). Next: a fresh grid of what it draws, with
those pairs in it, before it goes on by default.

**Grid N2 sent 2026-10-18 (awaiting marks):** all 21 vessels the re-trace drew on E14.5, V keep /
X remove, mask + dextran, shuffled (seed 20261019). Page
https://claude.ai/artifact/TYSjCA9SWdTfhGEcNTzWag; key `surface_review/N2_key.csv` (hole, pair
in one lumen, place, and `same_place_as_N`: 10 cases sit at a grid N place, 11 are fresh);
builder `build_grid_n2.py`. Its precision on the 11 fresh cases is the fresh check.

**Grid N2 marked 2026-10-18** (`surface_review/FIT_N2.md`): 19 V / 1 X (N2-18) / 1 ? (N2-15);
**10 of the 11 fresh cases V**; 6 of the 7 pairs in one lumen V. Harvey's notes: missing joins
(N2-12, N2-14), all along the piece to another place it meets the graph -- now drawn
(`_join_place`, hole-gated: 4 joins on E14.5, 1 of his 4 matched; the other three close loops
round the large vessel's lumen, hole ~0, which needs a join grid); and smoother routes (N2-13,
15, 16, 17) -- A* through the mask or the dextran matched his lines no better than the skeleton
(41-42% vs 48%), so the skeleton route stays. E14.5 with joins: 29 vessels, 1.17 mm, 48 s, reach
71.3 -> 73.3%, pairs 4 -> 14, short loops 42 -> 56.

**On by default since 2026-10-18** (Harvey's call after grid N2), uncommitted. At
`short_loop_keep_hole_over_um` 0 it draws nothing (that limit keeps no loop for its hole). On the
invariant tests' Nerve bed it judges 2 lost pieces, both under 20 um, and draws nothing: the default
build is unchanged there (129 edges, 5,477 um), so the each-lumen-once invariant stands as it was. On
E14.5 the default build gains the 21 vessels and 4 joins above.

30 pieces from the 72-piece pool, stratified by cause (in-lumen loops, pairs, short loops, the rest)
and place (near large, other surface, and 5 deep for contrast), each drawn with the piece
highlighted, the kept graph, the mask plane and dextran, plus the strand the rule judged it
against where there was one. Question: **V** -- draw it (the graph should follow this skeleton);
**X** -- leave it out. Measures in the key: far fraction, length, radius, whether it was beside a
strand when removed (cascade), contacts, the background its chords to the nearest kept strand
cross, hole size where it closes a loop.

Build, if the pool (or a measure within it) fits: `retrace_lost_skeleton` (off by default), a step
after the last clean-up and before `smooth_straight_steps`: each qualifying piece drawn along its
skeleton voxels, joined where it meets skeleton the graph reaches (to the nearest graph sample,
through the gate), refused where it would close a loop inside one lumen, a short loop round a hole
<= 10 um, or run beside a kept vessel in one lumen; edges marked `retraced`. Smoothed like any edge.
Tests: a ring of two separate tubes fused by the mask at both ends keeps both sides; a strand
doubled in one lumen is not re-added; nothing is drawn across background; an unedited graph with no
pool is left exactly as it was.

The root-cause alternative -- stopping the one-lumen rules from taking a far strand in the first
place -- is what grids H/H2 and S could not support; it is only worth trying if grid N shows the
40 "already far when removed" pieces are V and the 32 cascades are not.

### Phase 2b -- Grid O: the sheets no skeleton crosses (after N and P)

30 of the sheet pieces (beside a centreline, >= 200 um^3, surface first), each drawn across the
sheet with the dextran beside it and the centreline that runs down it. Question: **V** -- there
is a vessel here the skeleton misses (draw it); **X** -- one wide vessel, the centreline drawn is
enough. Key measures: width across the sheet against its thickness, distance of the far edge from
the centreline, the number of bright dextran ridges across it, its share inside the thick region.
If a measure fits, the fix belongs to skeletonisation (a second centreline along a fused sheet),
not to graph building.

### Phase 3 -- Validate and hand on

- E14.5 before/after, each variant on its own, both together: reach by place, mask coverage, pairs
  in one lumen, in-lumen loops, short loops, dead ends, doubled centreline, bridges, build time.
- A fresh grid for each rule before it goes on by default; a second stack with large vessels
  (CLAUDE.md step 6; one is still owed for the tree fill too).
- What is left goes to a new Post processing review list, "Skeleton the graph does not follow"
  (`graph/network_review.py` finder over the lost pieces, Trace on through Add vessel), so the
  remainder is reviewable rather than invisible.
- CLAUDE.md (Key modules: the thick gate, the build's steps), RELEASE_REVIEW.md rows the reach
  numbers feed (graph accuracy), the changed settings' help text (not the help pages).

---

### Status (2026-10-18)

Grids N and P sent together, mask plus dextran (D1, D3 decided by Harvey). Built by
`outputs/surface_review/build_grids_np.py` (seed 20261018; keys `N_key.csv`, `P_key.csv`; static
`grid_N.png`, `grid_P.png`). Pages, marks saved in each page's `annotations` collection:
N https://claude.ai/artifact/AfPmDDFUaBrE46N6FGhr1b, P https://claude.ai/artifact/YMVSfK7t9PsUPL5n89GNEp.
N: 15 near large / 9 other surface / 6 deep from the 72-piece pool, causes in turn, spread by
distance from the kept graph. P: 15 inside a large-vessel mask / 15 outside it, from the 157
plain-Lee pieces on the surface beside the large vessels, spread by how far outside the thick
tree's lumen they lie.

**Both marked by Harvey, 2026-10-18** (`outputs/surface_review/FIT_P.md`, `FIT_N.md`).

- **Grid P: 5 V / 25 X; Phase 1's build is dropped.** The gate is right about what plain Lee would
  draw: 25 of 30 dropped lines are the large vessel itself. Piece size fits (V when > 58 voxels,
  29/30 leave-one-out), but the drawings show it marks places, not centrelines: Harvey's 17
  strokes (538 um) follow the red line for 35% of their length, and lie 78% in the thick region,
  62% inside a large-vessel mask, 8% in the shell. The prior (outside the trunk's lumen, grids
  Y/Z) scores 21/30, worse than all X. Inside the large masks the vessels he drew are fused with
  the large lumen (median inscribed radius 3.7 um), so D2's radius gate would release 1-10% of
  them; the dextran along them (slab percentile 0.61) is no brighter than along the X lines (0.58).
  Those vessels need narrower large masks or a hand trace; nothing in the mask finds them.
- **Grid N: 13 V / 17 X; Phase 2's re-trace has its gate.** The hole of the loop a piece would
  close with the final graph decides it: all 9 round a hole over 10.3 um are V, 17 of the 21 at
  or below are X. At the short-loop rule's own 10 um (fitted on Q/Q2, not on N) that is 25/30
  against 17 for all X (wrong N-16, N-21, N-25, N-29, N-30). The pair and in-lumen loop rules,
  which never read hole size, took 12 of the 13 V; the short-loop rule was right 8 times in 9.
  Simple strands are drawn along the piece's contact path; the large tangles (N-08, N-13, N-14)
  are Lee meshes in wide sheets, where Harvey drew smooth vessels the contact path follows for
  11-15%. In the pool, 19 pieces pass (2,106 skeleton voxels, 1,146 beside the large vessels).

## 4. Decisions for Harvey

- **D1. Order.** Recommended: send grids P and N together (both are on the same slab; you mark
  quickly), then build whichever fits first. Alternative: P first, since case B has no workaround in
  the panel (Add vessel can trace case A by hand).
- **D2. The large masks as the thick body.** The restriction is authoritative by design (a narrow
  artery is still the large vessel). If grid P shows the venule mask's sheet covering capillaries,
  either the ilastik masks are retrained (outside this repo) or the body becomes large mask AND
  inscribed radius above half `skeleton_thick_vessel_min_radius_um`. Your call once the marks are in.
- **D3. Channel for the grids.** Mask plus dextran (C1) as before, or the claudin-5 channel as in
  grids Q/Q2, which shows walls between fused plexus vessels better?
- **D4. Positions of your seven examples** (Phase 0).

## 5. Targets

None set before the marks: a reach target would reward re-adding what grid S called artefact. The
report states reach by place, the artefact counts beside it, and the marks each rule was fitted on.

## 6. Risks

- Re-adding doubled strands and false loops: the reason the one-lumen rules exist. Mitigated by the
  far test, the loop guards and the marks; measured by the artefact counts in Phase 3.
- Lee inside the thick region brings back the medial mesh the gate removed (Phase 1's whole
  question). Measured by the in-lumen loops and pairs beside the thick tree.
- Build time: the re-trace adds a pass over ~70 pieces (seconds); Lee-thinning the thick region adds
  its share of the 235 s skeleton stage.

**Harvey's missing-vessel drawing (2026-10-18)** -- every vessel the default build's surface still
lacked, drawn on https://claude.ai/artifact/AbanSnNNMVo87KxuhUXLoL (68 strokes, 1.73 mm;
`surface_review/missing_3d.py`, `missing_fit.py`, `missing_auc.py`): 62% of it has skeleton under
it (lost to pairs 487 um, short loops 298 um), 38% none (beside the thick region, median 1.2 um:
case B, which plain Lee only partly covers). Labelled pieces: of 504 still lost on the surface, 38
drawn. Only the mask share separated cleanly, so MIN_MASK_FRACTION is now 0.5 (every drawn piece
was 58% or more in the mask; at 0.8 four at 69-78% were refused, nothing else let in): 6 new edges
on E14.5, the 4 longest on his strokes, his missing length covered 26.1 -> 30.8%. Every other
relaxation (separation, hole, length) let in more he did not draw than he did; AUCs length 0.78,
loop length 0.67, separation 0.66, hole 0.64, dextran 0.62. The rest is for hand tracing or new
evidence.
