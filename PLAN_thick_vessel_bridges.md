# Plan: thick-vessel bridges as one radius

Status: proposed, 2026-10-09. Supersedes Phase 6 of `PLAN_one_centreline_per_lumen.md`
("Thick-vessel bridges inside the fat lumen"), whose first step (arms aim at the ridge first)
landed in `eb75435`.

## Goal

A thick vessel is represented by its **centreline only**. Every vessel that enters or leaves it
is joined to that centreline by **one bridge**: a zero-resistance edge from where the vessel
leaves the thick vessel's lumen to the nearest point of the thick centreline, i.e. one local
radius long. Its purpose is that the thick vessel's radius is not counted as vessel length (or
resistance) of the vessel joining it.

Everything else inside the thick region keeps its resistance:

- the thick vessel's own centreline, at the thick vessel's diameter;
- a vessel that passes through the thick region's margin without reaching the centreline;
- anything else that is not the stretch from the lumen wall to the centreline.

Out of scope here (separate tracks, see the end): connections that should not exist at all
(grid IDs 9, 18, 19), and how well the large-vessel mask covers the segmented lumen.

## What we found

All on E14.5 MCA (`outputs/...MCAregion_Simple Segmentation_checkpoint_solve.pkl`, built
2026-10-07 with the join code before `eb75435`).

**1. The pieces between two bridges are real.** The marked grid
(`outputs/bridge_sandwich_review/`, 30 of the 54 such pieces): 27 real, 3 should not be
connected, 0 artificial. Marking everything real gets 27/30. The assumed rule (sandwiched means
artificial) is wrong; what is wrong is the bridges either side of them.

**2. Most bridges are not radius-shaped.** `outputs/bridge_audit/bridges.csv`, measured against
the centreline tree the thick-vessel skeletonisation drew (re-run with the code that built the
graph):

| What the bridge is | Count | Median length |
|---|---|---|
| Radius-shaped: centreline end, wall end, length <= 1.5 x the inner end's depth | 55 | 7.2 µm |
| Reaches the centreline, longer than the radius | 54 | 11.1 µm |
| Lies along the thick vessel's own centreline (>= half its length) | 48 | 12.9 µm |
| Ends on another bridge, not the centreline | 58 | 4.2 µm |
| Wall to wall: passes through the margin, never reaches the centreline | 62 | 5.2 µm |
| Other: never reaches the centreline | 116 | 5.8 µm |

So 290 of 393 bridges give zero resistance to something that is vessel (the thick vessel's
centreline, or a vessel grazing the margin), and 112 more are longer or chained.

**3. Where the example came from** (`outputs/bridge_drawing/solid_piece_segmentation.png`): the
solid piece runs along the inner corner where a venule trunk meets a branching venule. The
venule mask covers both limbs but not the corner; the piece is 1 µm from the segmentation's
edge and 4.9 µm from the venule mask, outside the 3 µm margin. It lies on the thin-vessel Lee
skeleton (59 %) and a join path (41 %); its second bridge is 3.3 µm from any skeleton voxel,
i.e. moved by graph building.

**4. The thick centreline cannot be found reliably from the mask alone.** The axis of the thick
region's depth map (a point within 1.5 µm of the deepest point within 3 µm, at least 4 µm deep)
agrees with the centreline tree on 86.6 % of the graph's samples inside the thick region
(precision 87 %, recall 81 %); the segmentation's depth map does worse (83.7 %). One sample in
seven wrong is not good enough to decide resistances by, so the real centreline has to be
carried, with the depth axis only as a logged fallback.

**5. No standard tool does this.** Nearest practice: 1D blood-flow pipelines move junctions
from the centreline branch point into the ostium, where the radius is defined (Bartolo et al.
2024); radius estimation leaves out a dilated zone round branch nodes (MARVEL); a vessel-graph
pipeline subtracts an "inner length" at branch nodes when judging stubs (Drees et al. 2021).
A zero-resistance segment from the ostium to the parent's centreline is the same idea in
network form, and keeps Kirchhoff conservation at one node.

## Root causes

1. **The tagging rule** (`graph/thick_vessel_junctions.py:113`): every inside run of an edge
   that crosses the thick region's boundary is a bridge, whether or not it reaches the
   centreline. Wall-to-wall grazes and runs along the centreline are tagged; edges wholly
   inside are never tagged, even when they are join paths from an off-axis node.
2. **The wall is in the wrong place.** The thick region is the core + geodesic body + a
   `wall_absorption_um` ball (`thick_vessels.py:301-346`), so its edge sits ~3 µm outside the
   segmented wall: even a good bridge takes ~3 µm of the joining vessel.
3. **Degree-2 merging joins a thin arm, its join path and a centreline tail into one edge**
   (when the join met a ridge tip), so one inside run covers both bridge and centreline.
4. **Joins aimed at earlier joins** (fixed for new builds by `eb75435`, except past the cap).
5. **The large-vessel mask does not match the segmented lumen** (corners of junctions, partial
   cover, image edges), so vessels run in and out of the margin.
6. **Graph building moves centrelines after skeletonisation** (preprocessing, cluster
   collapse, smoothing), so a bridge's geometry drifts from the skeleton it came from.
7. **Downstream treats bridges as vessels**: diameters are measured on them (and they are in
   the FWHM/raw-section calibration set, `stages.py:2462`); statistics count them in total
   length, vessel counts, densities and tortuosity (`statistics/shape.py:411-421`); lumen
   diagnostics and most review finders treat them as ordinary edges; the VTK export drops the
   flag (`vtk_io.py:142-189`).

## Design

### A. The thick centreline as a run output

- `skeletonize_thickness_gated(..., return_thick_mask=True)` also returns the centreline tree
  (`fat_skel` before joins, `thick_vessels.py:1407/1970`): `(skeleton, thick, centreline)`.
- Carry `thick_centreline_mask` beside `thick_vessel_mask` everywhere the region goes:
  `SkeletonisedVolume` (`stages.py:156`), `PipelineResume` (:340), `_thick_region_arguments`
  (:593), `thick_vessel_mask_from_image` (:607), `_bridge_edited_vessels` / `post_process`
  (:2577, :2677, :5084), `gui/stage_checkpoints.py` (:163, :487, :677, :701),
  `gui/results.py` state (:1388-1524, :1821), `gui/run_snapshot.py:336`, and the
  `scripts/graph_artefacts` tools.
- A resumed run without it rebuilds it (`thick_vessel_region` grows a `centreline=True` path
  running the same ridge-tree code, ~1-2 min on E14.5); if that fails, the depth-axis test of
  finding 4, with a warning that bridges are approximate.
- Old checkpoints and `.haemorun` files load with the field absent (same default-None pattern
  as the region itself).

### B. The bridge rule

In `insert_thick_vessel_junction_nodes(G, thick_mask, centreline_mask, *, voxel_size_zyx,
segmentation=None, edges=None)`:

1. Label every sample of every edge that touches the thick region: **outside**,
   **centreline** (within `thick_bridge_centreline_tolerance_um`, default 1.5 µm, of the
   centreline mask), or **in the lumen, off the centreline**. Absorb one-sample flickers, as
   `_runs` does now.
2. Split edges where the label changes.
3. Treat the off-centreline pieces as a graph of their own and take its connected components
   ("connectors"):
   - touches the centreline **and** leaves the lumen (an outside piece or a dead end at the
     wall): a **bridge** component;
   - never touches the centreline: **vessel** (a graze through the margin; keeps resistance);
   - touches the centreline only (a spur inside the lumen): vessel; left for the stub prune.
4. Centreline pieces are **vessel**: the thick vessel's own centreline, at its own diameter.

### C. One radius per joining vessel

For each bridge component:

1. Find where each joining vessel leaves the thick vessel's **segmented lumen** (not the
   thick region's edge): along the vessel, the first sample outside the segmentation's thick
   part, or, without a segmentation, the thick region's edge less `wall_absorption_um`.
2. Replace the component with one straight edge per joining vessel, from that wall point to the
   nearest point of the thick centreline, splitting the centreline edge there
   (`graph.edit.insert_node_on_edge`). Its length is the local radius.
3. The joining vessel's stretch between the wall point and the old boundary node becomes part
   of that vessel (resistance at its own diameter).
4. Two joining vessels whose nearest centreline points are within a radius of each other share
   one centreline node (a Y at the wall would put a junction in the lumen).
5. Keep the drawn path instead of the straight one only if the straight one leaves the
   segmentation (a curved thick vessel); log how often.
6. A node left at degree 2 by the removal is merged (the same merge the build uses, keeping
   the bridge/non-bridge guard of `post_processing.py:308`).

Setting: `thick_bridge_one_radius` (advanced, on), under `use_thick_vessel_skeletonisation`,
so the old drawn-path bridges can still be produced for comparison.

### D. Downstream consistency

- **Diameters**: skip bridges in `assign_edge_diameters` and in the calibration set; stamp
  `diameter_source="bridge"`, `diameter_um` = the thick centreline's local diameter (for
  drawing only; resistance is zeroed anyway).
- **Statistics**: bridges out of total length, vessel counts, densities, tortuosity and
  branch-order aggregates; reported as their own row (count, total length) in the CSV.
- **Review finders and lumen diagnostics**: skip bridges in the hairpin finder and the junction
  part of diameter jumps (`network_review.py:338, :402`); count them apart in
  `diagnose_lumen_artefacts`.
- **VTK**: export `is_zero_resistance` (also planned in `PLAN_dale_review_fixes.md:153`).
- **Clean-up order**: run `consolidate_lumen` after the split (as
  `PLAN_one_centreline_per_lumen.md:240` asks), with bridges excluded from parallel-edge and
  loop removal.
- **Hand edits**: `post_process` uses the same rule through `edges=`; diameters of the pieces
  are measured after the split, as now, and a centreline piece never inherits the thin
  vessel's provisional diameter.
- **Resistance**: unchanged (`apply_final_resistance_overrides`), but checked to still run last
  in every perturbation and sweep path.
- **Drawing**: already done (`3ec9e2b`): dashed lines, thin flat-ended tube pieces.

## Phases

Each phase lands on its own commit with its tests; E14.5 numbers are taken from a `git archive`
snapshot, not the live tree.

**Phase 0: measure.** `graph.diagnose_thick_vessel_bridges(G, thick, centreline,
segmentation)` reports the six categories of finding 2 plus bridge length / local radius;
logged in `build_network`'s graph checks beside `diagnose_lumen_artefacts`, and written by the
graph-artefacts script. Synthetic fixture per category in
`tests/test_graph_thick_vessel_junctions.py`. Done when E14.5 reproduces the table above.

**Phase 1: carry the thick centreline** (design A). Update the shape-pinning tests
(`test_gui_stage_checkpoints.py:483-512, :834-851`, `test_gui_run_snapshot.py:194-239`,
`test_gui_results.py:2362-2450`, `test_gui_results_widget.py:815-965`,
`test_thick_vessel_skeletonisation.py:315-348, :1781-1789, :1838, :1866`). Done when a fresh
run, a resumed run and a loaded `.haemorun` all reach the split with the same centreline.

**Phase 2: the bridge rule** (design B). Rewrites
`test_multiple_crossings_tag_only_each_interior_run` (:80) and the fixtures at :55, :155, :183
to include a centreline; `test_post_process_stage.py:288-423` keeps its behaviour (its inside
piece already ends on the centreline). New fixtures: a graze through the margin (no bridge), a
merged arm + join + centreline tail (bridge + centreline vessel), a centreline leaving the mask
and coming back (no bridge), a vessel crossing straight through a thick one (two bridges to one
centreline node).

**Phase 3: one radius** (design C). Fixtures: an oblique join (bridge = radius, the rest
vessel), a bridge chain (each vessel its own radius), two vessels entering side by side (one
centreline node), a curved thick vessel (drawn path kept). Done when every bridge on E14.5 is
radius-shaped (category 1) or logged as a kept drawn path.

**Phase 4: downstream** (design D), with tests in the files CLAUDE.md maps them to
(`test_statistics.py`, `test_network_review.py`, `test_vtk_io.py`,
`test_hemodynamics.py` / `test_thick_vessel_bridge_resistance.py`, `test_post_process_stage.py`).

**Phase 5: calibrate on marks** (CLAUDE.md, "Marked example grids"). A grid of ~30 bridges
and ~30 near-misses (pieces the new rule left as vessel inside or beside the thick region),
spread across the tolerance and the length / radius ratio, shuffled, drawn on the
segmentation in each case's own plane; marks "correct bridge / should be vessel / thick
centreline / should not be connected". Fit the tolerance, report hits against marking all one
way, then check on a fresh grid; turn each kind the marks reveal into a fixture.

**Phase 6: whole-network check.** E14.5 and the Nerve fixture before / after: bridge count
(expected ~ the number of vessels entering thick vessels), bridge length / radius, total vessel
length, network resistance, flows in the vessels that regained resistance; build time; the
`not slow and not gui` suite and the GUI files touching bridges.

## Status (2026-10-11)

Implemented on branch `claude/thick-bridge-radius` (worktree `.claude/worktrees/thick-bridges`),
after Harvey's rules on the first version (2026-10-10): **A** a bridge reaches the thick
vessel's segmented boundary in every case; **B** it connects the centreline to an incoming
vessel; a bridge is never a dead end nor attached to one.

What changed from the design below:

- Design A (carry the centreline) was not needed: the tree is `skeletonize_edt_ridge(thick)`, a
  function of the region, rebuilt at the split in 9 s on E14.5 -- so resume, checkpoints and
  `.haemorun` are untouched, and "rebuild" holds for every run.
- The wall is the thick region's boundary (the segmentation within the margin of the
  large-vessel body), not the inscribed lumen radius: the radius cut left vessel pieces inside
  wide or oblique joins (rule A). The inscribed radius is kept only to tell a stretch entering
  the lumen from one skimming the region's edge.
- The centreline tolerance is max(1.5 um, 0.3 x lumen radius): a wide vessel's centreline wanders
  round the tree. Stretches under 2 um are flicker.
- The thick vessel's own path where its centreline strays off the tree is kept (a stretch with
  no exit, or the shortest way between two places a stretch meets the centreline).
- Nothing is cut off: a stretch whose removal would disconnect anything is left as it was.
- Blind vessels beyond a bridge go with it (the dead-end rule); vessels running out of the
  image stay.

E14.5 (the graph `build_network` hands the split, current code): 150 bridges (old rule: 360
tagged pieces, 44 radius-shaped), each from the region's boundary to the centreline, every one
leading to a junction (132) or out of the image (18); 42 blind vessels removed (529 um);
connected pieces unchanged (155); dead ends 1,127 -> 1,085. Pictures:
`outputs/bridge_radius/`.

Done: the split (Phases 2-3), diameters, branch orders and statistics (Phase 4). Not done:
review finders' bridge handling, the VTK flag (planned in `PLAN_dale_review_fixes.md`), running
`consolidate_lumen` after the split, and the marked grid (Phase 5).

## Decisions (Harvey, 2026-10-09)

1. A bridge ends at the **segmented lumen wall** (the true radius), not the thick region's edge.
2. Bridges are **straight from the wall to the centreline**, re-anchored.
3. Bridges are **left out of the statistics and out of branch-order assignment**.
4. A resumed run with no stored centreline **rebuilds it**.

## Separate tracks

- **Connections that should not exist** (grid IDs 9, 18, 19): joins across gaps between
  vessels that only touch. Needs its own measure and grid; likely in the join caps of
  `_join_thin_arms_to_fat_ridge` or the reconnects.
- **Thick-region coverage of the lumen**: the large-vessel mask misses junction corners and
  parts of the lumen (finding 3). Growing the body through the segmentation up to the local
  lumen radius, instead of a fixed margin, would put fewer real vessels inside the margin.

## Sources

- Bartolo et al. 2024, one-dimensional vascular models from images, junctions moved into the
  ostium: https://arxiv.org/html/2309.08779v2
- MARVEL, dilated branch-node zones left out of radius estimation:
  https://arxiv.org/pdf/2605.25363
- Drees et al. 2021, vessel-graph extraction, "bulge size" with an inner length at branch
  nodes: https://arxiv.org/pdf/2102.03444
- VesselVio, branch-point cliques weighted by radius:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9046271/
