# Plan: dead ends — what each one is, and what the network and the flow do with it

Written 2026-10-10 against `Devel_GUI_improvements_HD` at **e09035d**. Untracked, like the other
`PLAN_*.md` files. Every number was measured on the E14.5 MCA stack (section 2 says which graph).

**In one paragraph.** A dead end is one of seven things: a vessel leaving the image, a vessel the
graph stopped short of, a vessel the segmentation broke, a real blind end (E14.5 sprouts), a
skeleton artefact, the end of an island, or a node a later step left behind. Today the code has
five different "image face" rules, three chain walkers, two meanings of "short" and three of
"inside another lumen", and in the solve every dead end that is not an inlet or outlet is
**closed**. On E14.5 that leaves 60% of the centreline without flow, about a third of it only
because the image face is treated as a wall. The plan: measure (Phase 0), give every dead end one
kind from one definition (1), calibrate the kinds on marked grids (2), put face ends on the face
and give them a boundary condition (3–4), finish the vessels the graph stopped short of or the
segmentation broke (5–6), keep and count the real blind ends (7), and carry the kinds through
exports, statistics and the Post processing review (8–9).

---

## Implementation status (2026-10-12, branch `claude/dead-ends`; checked against `origin/Devel_GUI_improvements_HD` at `60ea7681` on 2026-10-24)

Phases 0–10 landed as **b6d9210d** (2026-10-12). The grids' marks and fits are git-ignored and lie
in the `dead-ends` worktree, `.claude/worktrees/dead-ends/outputs/dead_end_review/`, except G5's.

| Phase | State |
|---|---|
| 0 Measure | done (b6d9210d): `scripts/graph_artefacts/dead_end_census.py` (census by kind, place, depth, step; flow coverage), in the graph-artefacts report and `dead_ends.csv`; fixtures `tests/dead_end_fixtures.py` |
| 1 One definition | done (b6d9210d): `graph/dead_ends.py` (`DEAD_END_KINDS`, `dead_ends.py:56`); diagnostics, review lists, boundary review read it; `build_network` marks `dead_end_kind` and `image_extent_um` (`stages._mark_dead_end_kinds`, `stages.py:2657`). Since then, `post_process` re-judges the kinds at the nodes hand edits touched (`reclassify_dead_ends`, `dead_ends.py:596`; `stages._mark_touched_dead_end_kinds`, `stages.py:4642`; 40c99a4d, 2026-10-18) |
| 2 Grids | marked by Harvey 2026-10-12 (87 judged; `G1_marked.csv`–`G3_marked.csv`); fitted by `scripts/dead_end_review/fit_marks.py`, read in `FIT.md`. No rule reaches 90%: G1 none, G2 the `leaves_image` rule 24/30 (vs 20 all-one-way, 23 distance), G3 turn <= 30.5 deg 24/27 (89%). Grids: G1 https://claude.ai/artifact/Hqtst8xFHBaLtAna4KxdVH, G2 https://claude.ai/artifact/BoBYsUw9Yd18YkczSZLuNx, G3 https://claude.ai/artifact/QNVRrGL9C8syjo8cwBVo5R. G4 (is the chain a real vessel? segmentation, dextran and claudin-5 side by side, two planes; 30 fresh interior dead ends spread by dextran and endothelial contrast; builder `scripts/dead_end_review/build_g4.py`): https://claude.ai/artifact/6DfV4aCrAeH42v8YNd1hTA, marked by Harvey (`G4_marked.csv`, file dated 2026-10-12 18:37; `FIT_G4.md` heads its reading 2026-10-13; N 15, V 12, P 2): every mark cites segmentation shape and context, none the endothelium; best measure claudin-5 chain contrast 20/27 LOO (74%) vs 15 all-one-way, no shape measure better than 17/27. G5 (Phase 5's route join, below; Harvey 2026-10-21, `outputs/stops_short_route_review/G5_marks.csv`). **No fresh grid (G1′, G2′, G3′) has been built**, so no rule is confirmed |
| 3 Face ends, graph | Dale item 2 done, always on (b99758ba, 2026-10-12: `memmap_support.at_faces_as_beyond`, `memmap_support.py:118`); `extend_dead_ends_to_faces` (default off, `schema.py:3483`), `stub_face_protection` (default `threshold`, `schema.py:3528`) and `boundary_open_end_test` (default `distance`, `schema.py:2592`) built, the definition not used by default (G2 did not reach the bar) |
| 4 Face ends, flow | done (b6d9210d): `haemodynamics/open_ends.py` (closed, terminal_resistance, outlet, estimated; `OPEN_END_MODELS`, `open_ends.py:63`), every solver path; `scripts/open_end_sensitivity.py`. D1 decided 2026-10-12: default `estimated` (cd71285c; `schema.py:5101`; see section 7). Flows behind blind and closed ends set to exactly 0 after every solve (`graph.zero_flows_behind_dead_ends`, `prune.py:452`; 0c6d13b3) |
| 5 Stopped short | `trace_on_dead_ends` built, off (`schema.py:3497`). G1 found no rule, and G5 (2026-10-21) scored a route join where the route reaches a vessel two or more junctions away 23/29, the same as never joining (29 judged: 6 J, 6 W, 17 S; `G5_marks.csv`, `G5_measures.csv`), so it stays off. Separately, `remove_interior_dead_ends` (remove each interior dead end's chain, then recover the segmented vessels joined at both ends) was built on by default (94d26a26, 2026-10-22) and turned off the same day (ae814394; `schema.py:3304`): fitted on E14.5 alone, it took a fifth of the nerve crop |
| 6 Broken | `join_dead_ends_to_vessel_sides` built, off (`schema.py:3513`), D3: review only |
| 7 Blind ends | done (b6d9210d): `flow_state` (`prune.py:428`), statistics "Dead Ends" section (`statistics/dead_ends.py`). D4 (2026-10-15, 10c69230): `blind_end` vessels drawn as unsolved, grey, the hover "Unsolved: behind a dead end" (`branch_hover.py:75`); the exports keep 0 with `flow_state=blind_end` |
| 8 Exports | done (b6d9210d): VTK flow_state/flow_solved/length/diameter/source/bridge arrays, NaN only for no_boundary, node dead_end_kind, run metadata (`vtk_io.py:258-339`); perturbation CSV |
| 9 Review | done (b6d9210d): dead-end list grouped by kind (`graph/network_review.py:890`); Join to the vessel ahead, Mark as a real blind end, Mark as leaving the image (`gui/post_processing.py:1228-1234`, `:1290`); trace-on decision written when the trace finishes (`gui/_widget.py:12853`). Hovers name open ends and the dead-end kind of each node (41307bd0, 2026-10-22) |
| 10 Validation | E14.5 baseline: `outputs/de0/REPORT.md`. Full E14.5 run through export (`outputs/e14_full/run.log`): 1,082 dead ends (121 leave the image); with the estimate, 68 open ends get a condition and no vessel stays behind a closed face end. Found and fixed three crashes on the way (pericyte perturbations on bridges and on unreached pieces, statistics on a contracted self-loop). Sensitivity: `outputs/open_end_sensitivity_e14/` (all in the `dead-ends` worktree). Not done: the Nerve invariant of Phase 10 (`tests/test_pipeline_invariants.py` has no dead-end test) |

E14.5 baseline (code of 2026-10-12, graph as saved): 1,077 dead ends -- 115 leave the image, 124 stops short,
42 broken, 36 artefact, 530 unknown, 230 island ends. Final dead ends first appear at: graph build
682, cluster collapse 61, mask recovery 229, end-of-build clean-up 84, after the steps 18.

---

## 0. Who is doing what (read first)

Two sessions are working on parts of this right now. This plan does not touch their parts; it
uses what they land.

| Work | Owner | State (2026-10-10 morning) | This plan |
|---|---|---|---|
| Dead ends whose tip lies **inside another vessel's lumen** | **One centreline per lumen** session (`PLAN_one_centreline_per_lumen.md`, target "dead ends with tip in another lumen: 0") | 21 → **7**: 15 were the report reading the tip against its own chain (cut by the thick split), fixed in `diagnostics.py` (uncommitted, with a test in `test_lumen_artefacts.py`). The 7 left are real, in a grid awaiting your marks J / R / L (`outputs/dead_end_review/tips_in_lumen_grid_key.csv`); T4 and T7 may be missed connections | Not touched. Its rule becomes the `artefact` kind's inside-another-lumen test. T1–T7's marks join G3 (Phase 2): J marks are tip-to-side joins. |
| The stub prune (`graph/prune.py`: whole chains, 3-radius rule, protrusion rule, image-face protection) | same session (its Phase 4, landed `ae06500`, `4e44afc`) | a protrusion bug (it read only the 16 nearest centreline samples, which on a long stub can all be its own) is being fixed, then committed with the report fix; 4 E14.5 stubs change verdict, all near image faces. The 29 face-protected short stubs still have no grid | `prune.py` is not edited until that lands. Phase 3 here changes the face protection, agreed with them. |
| Dead ends the **thick-vessel split** makes | **Bridge visualization** session (`PLAN_thick_vessel_bridges.md`), worktree `.claude/worktrees/thick-bridges`, branch `claude/thick-bridge-radius` | new split measured on E14.5 (1,127 dead ends before the split, 1,146 after); now on its downstream phase (`branch_order.py`, `apply.py`, `stages.py`), uncommitted in the worktree | Not touched. Ask: its Phase 6 reports the dead-end census of section 2 before and after, and "the split leaves no dead end it cannot name" is one of its checks. Re-baseline here after it lands, together with its `consolidate_lumen` after the split. |
| Morphology eroding the image faces (`PLAN_dale_review_fixes.md`, item 2) | unassigned | not started | **Prerequisite of Phase 3.** Face ends move outward by up to the closing radius once it lands. |
| VTK: unsolved flows blank, flags exported (`PLAN_dale_review_fixes.md`, item 5) | unassigned | not started | Phase 8 adds the dead-end kind and flow state on top of it. |
| Release review rows `graph.6b` (dead-end rule fitted to marks), `flow.6c` (unsolved share explained), `flow.8b` (sensitivity to boundary placement), `export.4b` | `RELEASE_REVIEW.md` | all ✗ | Phases 0, 2, 4 and 8 produce their evidence. Update the rows in the same commits. |

Files both peers edit: `graph/assemble.py`, `graph/diagnostics.py`, `pipeline/stages.py`
(`build_network`), `pipeline/schema.py`, `CLAUDE.md`. Work here happens in its own worktree off the
shared tip, lands phase by phase, and is rebased on the tip and re-measured before landing (memory:
shared checkout).

---

## 1. Why dead ends matter

A dead end is a degree-1 node. It matters three ways:

1. **Flow.** A dead end that is not a boundary node is closed: no blood leaves through it. Every
   vessel that only leads to closed dead ends carries no flow, and with the default
   `boundary_handling="leave_unsolved"` it is marked unsolved and drawn grey. On E14.5 that is
   **60% of the centreline** (section 2.2). Most of it is not a property of the tissue but of
   where the image ends and where the segmentation broke.
2. **The graph.** A dead end is one of: a vessel leaving the image, a vessel the graph stopped
   short of, a vessel the segmentation broke, a real blind-ended sprout (E14.5 is embryonic brain,
   where angiogenic sprouts are expected), or an artefact (a skeleton spur, a hair along a wall).
   Only the last should be deleted, and only the sprouts are dead ends in the tissue.
3. **Statistics.** Dead ends bias branch counts, segment lengths, loop counts, connectivity,
   tortuosity, and any flow statistic. A vessel cut by the image face has a censored length. A
   dead-end density is only a biological number (sprouts per mm of vessel) once the other kinds are
   taken out.

---

## 2. What was measured

Two graphs, both read only:

- **A — the last solved E14.5 run** (`outputs/…_checkpoint_solve.pkl`, 7 Oct, `config.yaml`,
  graph code from before Phases 1–6 of `PLAN_one_centreline_per_lumen.md`). The only solved E14.5
  network on disk.
- **B — today's graph as `build_network` saves it**, at `9c6a15a` with the old thick split (the
  Bridge session's capture, `split_old.pkl`): 3,146 edges, 51.4 mm of centreline, 155 components.

Scripts: this session's scratchpad (`census.py`, `whatif.py`, `census_now.py`, `tip_targets.py`,
`near_face.py`, `zprofile.py`); Phase 0 turns them into a tool.

### 2.1 Where the dead ends are (graph B)

| | count |
|---|---:|
| dead ends (degree-1 nodes) | **1,127** |
| within 2 / 5 / 10 / 20 µm of an image face | 144 / 204 / 297 / 555 |
| more than 10 µm from every face ("interior") | 830 |
| …on a chain that reaches a junction | 648 |
| …on an island (a component with no junction) | 182 (115 single-edge islands = 230 dead ends in all) |
| interior chain length, tip to junction, p10 / 25 / 50 / 75 / 90 | 8.4 / 11.1 / 15.7 / 25.2 / 41.8 µm |
| interior, with another centreline straight ahead (≤ 45° off the tip's heading) within 5 / 10 / 15 µm | 20 / 111 / 193 |

The lumen-artefact report on the same build family (`outputs/graph_artefacts_phase6_baseline/REPORT.md`,
653 interior dead ends — the 648 above plus a few): tip in another lumen 19 (7 once the One
centreline session's report fix lands), beside another
centreline 2, mostly off the mask 0, **mask runs on past the tip 114**, shorter than the radius rule
48 (29 since `4e44afc`, all kept by the prune's image-face protection); the rest are blind.

**The fixed 10 µm face margin is wrong both ways.** Following the straight line from each tip to
its nearest face through the segmentation:

| tip's distance to the face | mask reaches the face | partly | no |
|---|---:|---:|---:|
| ≤ 2 µm | 126 | 7 | 11 |
| 2–5 µm | 26 | 9 | 25 |
| 5–10 µm | 7 | 15 | **71** |
| 10–20 µm | **19** | 32 | 207 |

So ~70–100 dead ends counted "at a face" (5–10 µm) are interior ends, and some ends counted
interior run on through the mask to the face. 204 of the 258 tips 10–20 µm from a face are near a
z face.

**Dead ends cluster by depth.** The top 40 µm of the stack (the surface plexus) holds ~70% of the
vessel volume and 29 mm of the 51 mm of centreline, at 14–17 dead ends per mm of centreline.
Deeper, where vessels ingress into the tissue and the signal fades, it is 25–54 per mm. Either
explanation (real sprouts, or segmentation lost with depth) is plausible, so every marked grid in
this plan is stratified by depth.

### 2.2 What the dead ends do to the flow (graph A, and a what-if on B)

Graph A's boundary nodes: 21 (9 inlets at 4,500 Pa, 12 outlets at 1,000 Pa), all in the largest
component. **6 of the 21 sit 11–23 µm inside the image**, not at a face (the large-vessel stump
snaps noted in memory as 10–42 µm off).

| graph A | edges | centreline |
|---|---:|---:|
| solved | 1,452 | 20.2 mm (**40%**) |
| unsolved, hanging off the perfused component | | 13.0 mm |
| unsolved, in the 200 other components | | 17.7 mm |

The 957 edges of the other components carry no `resistance` at all, and their 1,131 nodes carry
`pressure = 0.0` — a number, not a blank (the VTK item covers edges, not node pressures).

What-if, topology only (an upper bound: a vessel counts once it lies on a path between two open
ends, whatever its flow would be):

| opened as boundaries | graph A | graph B |
|---|---:|---:|
| today's 21 boundary nodes | 40% | — |
| + every dead end within 2 µm of a face | — | 50% |
| + every dead end within 10 µm of a face | 59% | 58% |
| + every dead end within 20 µm of a face | 68% | 69% |

With every face end ≤ 10 µm open (graph B), what still carries no flow is **16.4 mm hanging off the
flowing network behind interior dead ends** and 5.1 mm of islands. Joining the 111 tips that have a
centreline ahead within 10 µm to it would bring 2.7 mm of that onto through paths (if every join
were right).

**What this says.** The biggest effect of dead ends on results is the closed image faces: worth
~10 mm (20% of the centreline) of flow. The second is the trees behind interior dead ends (~16 mm),
part real sprouts and part broken vessels; telling those apart needs marks. The graph-cleanliness
dead ends (spurs, hairs) are already mostly handled by the stub prune and the peers' work.

### 2.3 What others do (literature, 2026-10-10; sources at the end)

No published pipeline sorts dead ends into kinds with one rule. Everyone treats the image boundary
separately from interior ends, and interior ends are either repaired by a classifier trained on
human marks, deleted, or left without flow. **Nobody reports what the remaining interior dead ends
do to the flow or the statistics**, so Phase 0's numbers and Phase 4's sensitivity report are new.

- **How many dead ends to expect.** Kaufhold et al. 2012 (adult mouse cortex): ~3,000 ends, most
  where vessels leave the volume; only ~300 (~10%) interior. Cassot et al. 2006 (human cortex):
  8.1% of segments end free, **rising with depth to over 15% below 170 µm**, blamed mostly on
  incomplete filling and poor contrast. E14.5 is far off both: **74% of its dead ends are
  interior** (830 of 1,127), and ~20% of its segments end free. Embryonic sprouts explain part of
  that; Cassot's depth trend is the same pattern as section 2.1's, which points at filling and
  contrast for the rest (decision D7).
- **Repairing interior ends** (the evidence for Phases 5–6). Kaufhold 2012 bridged gaps by
  *threshold relaxation* in the raw image near each end and accepted a bridge by a classifier
  trained on 3D human labels (raw, mask and graph shown together): 88% precision and recall for
  bridging, 94% for deletion. Ji, Kleinfeld et al. 2021 (whole mouse brain) scored ends with a
  random forest on segment length, radius, the intensity drop at the end, the distance to the
  nearest unconnected end and how well two ends' directions line up; linkers were found along an
  intensity-weighted shortest path in the raw image and accepted by a second forest (length,
  straightness, signal-to-noise, radius); all classifiers > 95% against manual labels; 3
  connectivity errors in 2,133 segments checked. Both use the **raw image**, not only the mask.
- **Spur rules** (the check on today's prune). Tsai et al. 2009: a hair is shorter than its
  vessel's radius + 2 voxels, removed until none are left. Drees et al. 2021 (Voreen): bulge size
  = (length − the length inside the parent's lumen + tip radius) / the edge's own mean radius,
  removed below **3.0 for healthy blood vessels** (1.5 for irregular ones), iterated with degree-2
  merging, so it judges whole terminal chains — the same choice HaemoLynx made (3 radii, whole
  chain). **One difference matters**: Voreen subtracts the part inside the parent's lumen and
  scales by the stub's *own* radius; HaemoLynx measures from the junction node and scales by the
  radius *at the junction*. For a capillary off a capillary the two agree; for a capillary of
  radius r off a vessel of radius 5r, HaemoLynx requires ~15r and Voreen ~7r, so **today's rule
  prunes small branches 7r–15r long off large vessels that Voreen would keep**. Whether those are
  vessels is a question for marks: G1 is stratified by the parent / stub radius ratio (Phase 2).
  (This is about how the 3-radius rule is scaled, not decision D2, which keeps the rule.) Ji 2021 deletes free ends ≤ 2 µm; VesselVio
  uses fixed 5 µm; ClearMap 2 and VesSAP do no spur pruning at all.
- **Embryonic sprouts.** No published count of sprouts per mm, or of sprout length, at E12–E15
  (so E14.5's blind-end density would be new data). Tip-cell filopodia are 200–300 nm wide
  (Wilsch-Bräuninger et al. 2024, E14.5 neocortex), far below the voxel and unfilled by a plasma
  dye: **a dead end in a dextran image is a lumenised sprout, or a vessel the dye did not fill, or
  a segmentation loss — never a filopodium.** Coelho-Santos et al. 2021 (postnatal cortex) found
  that unconnected lumenised sprouts were *sometimes not filled by the injected dye* and found them
  by overlaying an endothelial label: the reason for D6 (use the endothelial channel). Hahn et al.
  2021 (E13.5 whole brain, no spur pruning): mean segment length 13.8 µm, segment density 3× adult
  cortex.
- **Boundary conditions at the image faces** (Phase 4):
  - *Closed or removed*: Blinder et al. 2013 (cut capillaries effectively carry no flow; cut
    surface arterioles get a pressure through a resistance R_m); Gagnon et al. 2015 removed
    capillary segments cut by the field of view to get a closed graph.
  - *Estimated* (Fry, Lee, Smith & Secomb 2012, *Microcirculation* 19:530–538): unknown boundary
    nodes get no conservation equation; the solver minimises the deviation of pressures from
    p0 = 31 mmHg and of wall shear stresses from |τ0| = 15 dyn/cm², settling flow directions
    iteratively. On rat mesentery with only 2–3 known boundary conditions, 9.2% of flow
    directions were wrong (3.8–18.4%), clustered near the unknown boundaries. Sweeney et al. 2018
    used it for 97.7% of the boundary conditions of a cortical network (388 boundary nodes).
    **This is the established method for networks with many unknown boundaries.**
  - *By vessel type*: Schmid et al. 2017 set pial arteriole pressures from measured
    pressure–diameter data and pial venules at 10 mmHg, and took the cut ends' pressures from an
    artificial network grown around the real one, after trimming 12.5% of the width off each side.
    Gould et al. 2017 used periodic lateral faces.
  - *Coupled to a continuum*: Peyrounette et al. 2018 couple truncated vessels to a homogenised
    capillary model; plain pressure continuity at the coupling gave ~29% pressure and ~150% flow
    error, their coupling < 1%.
  - *Calibrated*: Rasmussen 2025 sets each boundary pressure as a weighted mean of interior
    pressures plus an offset, calibrated in a Bayesian way; 13–16% of nodes were boundaries, ~80%
    of them capillaries; uncertainty concentrated within a few generations of the boundary.
  - *Terminal resistance on cut capillaries* (Dale's "Robin" mode): **no published example found**;
    Blinder's R_m on cut arterioles is the nearest.
  - Interior blind ends in flow models: d'Esposito et al. 2018 assigned 33% of internal nodes zero
    flow; Gould et al. 2017 joined dead-end loops to the capillary bed "to ensure flow through every
    vessel". Nobody measured the effect.
- **Every method's errors sit near the boundary** (Fry, Rasmussen, Schmid's trimming). So flow
  statistics should be reportable with a margin next to the faces left out (Phase 7).

---

## 3. How the code treats dead ends today

Mapped 2026-10-10 at e09035d (file:line on that commit).

### 3.1 In pipeline order

| Where | What it does to dead ends |
|---|---|
| Skeleton (`preprocessing/skeleton.py:1477` `connect_skeleton_components`) | bridges a skeleton tip to another component, ≤ `skeleton_max_bridge_distance`, facing ≤ 60°, mask-supported |
| Build step 1 (`graph/build.py:357-440`) and step 3 (`graph/optimise.py:37`) | terminal–terminal gap bridges ≤ `graph_reconnect_threshold` (10 µm), turn ≤ 80°, each end ≥ 3 µm judged on its **last edge** (`_terminal_end`), mask-supported |
| Step 5 `collapse_node_clusters` (`graph/collapse.py:30`) | merges every node within `cluster_collapse_distance` (5 µm): a dead end whose tip is within 5 µm of its own junction becomes a self-loop and is deleted; one within 5 µm of another vessel's node is joined to it. **No mask, direction or face rule**, and it runs before the prune |
| Step 7 and every `consolidate_lumen` round: `prune_vascular_stubs` (`graph/prune.py:130`) | the whole chain tip → junction; threshold `min_stub_length_radius_multiple` (3.0) × radius at the junction, raised to `min_stub_length` (10 µm) when the mask runs on past the tip; removed whatever the length when off the mask, ending in its parent's lumen, or never protruding a tip radius out of other lumens; **kept when the tip is within its own threshold of a face** (`at_an_image_face:209`, not a setting) |
| Step 9 `remove_edges_for_self_connected_nodes` | can leave the stick of a "lollipop" as a dead end |
| Step 10 orphan reconnect (`optimise.py:253`) | degree-0/1 nodes to any node ≤ `final_orphan_reconnect_threshold` (3 µm), 80°, mask-supported |
| Step 11 mask recovery (`graph/mask_recovery.py:739`) | creates dead ends (free ends it could not route); trims refused spurs unless `_on_image_face` (≤ 1 voxel, or a 4-voxel mask walk) |
| Step 12 `join_facing_dead_ends` (`graph/facing_ends.py:170`) | joins two tips pointing at each other: ≤ `facing_dead_end_max_gap_um` (10), turns ≤ 60°, headings along the whole chain, one stretch of background |
| Step 13 `consolidate_lumen` | `remove_edges_off_the_mask` **creates dead ends** after the facing join has run, so the broken vessels it leaves are never joined; the final prune runs only with a mask and `bridge_require_mask_support` |
| `build_network` after the steps | smoothing (lengths ~7% shorter), a second `consolidate_lumen` (prune on smoothed lengths), then the thick split (Bridge session) |
| `diagnose_lumen_artefacts` / `classify_dead_ends` (`graph/diagnostics.py:540-611`) | tips within a **fixed 10 µm** of a face are not classified at all; the rest read the **last edge** (except "short", which reads the chain) |
| Boundaries (`graph/boundaries.py`) | a terminal is any degree-1 node; `boundary_open_end_max_distance_um` (10) filters `edge_percent` and `degree_1_from_inlet` only; `mask_stump` snaps to the nearest terminal anywhere; mask-overlap assignment has no face test; `cut_network_at_large_vessel_volumes` makes terminals at the mask surface |
| Network handling and solve (`stages.py:2800`, `resistance.py:421`) | Dirichlet pressure at inlets and outlets only; **every other dead end is a free node: closed, zero flow**; vessels off every inlet-to-outlet path get `FLOW_SOLVED=False` (`leave_unsolved`) or are deleted (`remove_disconnected`); nodes with no path to a boundary keep pressure 0 |
| Post processing (`gui/post_processing.py`, `graph/network_review.py:745`) | "Dead ends" list (Delete / Trace on from the tip / Keep), "Facing dead ends not joined" (Join / Keep), "Unsolved pieces", "Boundary nodes" (flags an open end at a face that is no boundary node); "Prune disconnected and dead-end branches" button |
| Statistics, VTK | **no dead-end count anywhere**; VTK writes no solved flag, degree or role; flow statistics drop zero or non-finite flow and never read `FLOW_SOLVED` |

### 3.2 Inconsistencies (each one fixed in Phase 1 unless noted)

1. **Five "image face" rules**: the prune (tip within the stub's own threshold, often 6–20 µm;
   extent from the skeleton), diagnostics and both review lists (fixed 10 µm, extent from the mask),
   boundaries (`boundary_open_end_max_distance_um`, extent from the image; the boundary review
   ignores the setting), mask recovery (≤ 1 voxel or a 4-voxel mask walk), `mask_stump` (the mask
   touches a face slice). A stub 7 µm from a face can be pruned while diagnostics call it "at the
   face" and boundaries call it open.
2. **Last edge against whole chain**: diagnostics' off-mask, inside-another-lumen, beside-vessel and
   mask-continues; the gap, orphan and recovery end lengths; and the optimiser's stub sweep
   (`optimisation/candidates.py:601`) read the last edge. The prune and the facing join read the chain.
3. **"Inside another lumen" three ways**: tip within radius + 1 µm of its own junction (prune),
   never protruding one tip radius (prune), tip within radius + 1 µm of the nearest other sample
   (diagnostics). *The One centreline per lumen session is working on this one; take its answer.*
4. **"Short" two ways**: `min_stub_length_radius_multiple = 0` means 3.0 in `stages.py:1046` but
   "never short" in `gui/post_processing.py:1577`; diagnostics have no `min_stub_length` raise.
5. **"Dead end" two ways**: diagnostics drop isolated single edges only (an isolated chain is two
   interior dead ends); network handling's "dead_end" count includes loops and self-loops.
6. **Three chain walkers**: `prune._chain`, `network_review._chain_to_junction`,
   `facing_ends._heading`, with different stop rules.
7. **Three turn limits and three tangent lengths**: 80° (bridges, orphan, recovery), 60° (facing
   join, skeleton), 90° (facing review); 5 µm, 3 µm, 4 voxels.
8. **Cluster collapse removes and reconnects short dead ends with no rule** (left as is in
   Phase 1; measured in Phase 0, decided in Phase 5).
9. **Joining before removing**: the facing join (step 12) runs before `remove_edges_off_the_mask`
   (step 13) (Phase 6).
10. **"No flow" three ways**: `FLOW_SOLVED`, the haematocrit `_FLOW_EPSILON`, statistics' `== 0`
    (Phase 8).
11. Small bugs: "traced on" is recorded before any trace is made (`gui/_widget.py:10681`);
    `_edge_data` / `_chain_path` defined twice and `_vessels_not_dead_ends` unused in `prune.py`;
    `stages.py:1095` and `:1168` say fifteen topology steps (there are 13);
    `facing_dead_end_max_gap_um` has no `requires` on `bridge_require_mask_support` though it is
    gated by it; the graph editor window (`graph/edit.py:393`) can leave a dangling terminal.

### 3.3 What nothing acts on

- Interior dead ends of no kind ("a blind end, or a broken vessel"): only listed for review.
- Tips the mask runs on past: the prune deletes the short ones; nothing extends any of them.
- Open ends at a face that are not boundary nodes: closed, zero flow, unsolved.
- Tips at a face: never classified. Isolated single edges: absent from the Dead ends list.
- Interior dead ends chosen as inlets/outlets (coordinates, volume, node_ids, mask overlap,
  `mask_stump`): only flagged by the review.
- Facing pairs the join refused, and tip-to-side gaps: only listed, or not at all.
- Dead ends left by a hand deletion.

---

## 4. The kinds, and what each should become

One kind per dead end, decided by one function (Phase 1), calibrated on marks (Phase 2), stored on
the node as `dead_end_kind`.

| Kind | Recognised by (first proposal; Phase 2 calibrates) | What it is | Graph | Flow | Statistics | Phase |
|---|---|---|---|---|---|---|
| `leaves_image` | tip within max(tip radius, 2 voxels) of a face, **or** the mask runs from the tip to a face along its heading within a few radii | the vessel continues outside the field of view | never pruned; traced on to the face if it stops short | **open boundary** (model: decision D1) | censored length; never counted in a dead-end density | 1, 3, 4 |
| `boundary` | an inlet or outlet, whatever chose it | a designated boundary | must also be `leaves_image`, else flagged | Dirichlet, as today | — | 3 |
| `stops_short` | the mask runs on more than a radius past the tip (today's `mask_continues`) | a vessel the graph stopped short of | traced on through the mask until it meets the network, a face, another tip or the mask's end, then re-classified | whatever it joins | — | 5 |
| `broken` | facing tips (today) **or** a tip pointing at another vessel's side across a gap, with evidence the vessel continues (mask, raw image, endothelial channel) | a vessel the segmentation broke | joined, automatically only above a calibrated evidence threshold, else listed | carries flow once joined | — | 6 |
| `artefact` | off the mask, ends in its parent's lumen, never leaves other lumens (the peers' rule), blind and shorter than 3 radii | a skeleton spur, wall bump, hair along a wall | pruned (today's prune, one definition) | — | — | 1 |
| `blind` | none of the above, and the marks say it is real | a real blind end: on E14.5, a lumenised sprout | kept | **zero by topology**: a solved value, not "unsolved" | counted: blind ends per mm of vessel and per mm³ of tissue, their length | 7 |
| `unknown` | interior, none of the above, no evidence either way | blind end or broken vessel | kept, listed for review | as `blind` until reviewed | reported apart | 2, 9 |
| `island_end` | an end of a component with no junction | a fragment | kept (D3 of the One centreline plan: `remove_disconnected` when wanted) | no boundary | islands reported apart (count, length) | 1, 7 |

Dead ends **made by later steps** keep the kind of what they are, but the census (Phase 0) also
records which step made each one: the thick split (the Bridge session's check), the large-vessel
cut (deliberate terminals at the mask surface: `boundary` when assigned, else flagged), hand
deletions and the graph editor window (Phase 9).

---

## 5. Principles

1. **One definition.** A new `graph/dead_ends.py` holds the one chain walker, the one tip heading,
   the one face test and the one classifier. Every module that asks the question calls it:
   diagnostics, the review lists, boundaries, the prune's face protection, mask recovery's face
   test, the facing join's chain walk. A test puts the same fixture to every caller and requires
   one answer.
2. **Whole chains**, tip to the first junction, everywhere; the last edge alone nowhere.
3. **Measure before changing, and calibrate on marks** (CLAUDE.md, "Marked example grids"). Every
   threshold starts where today's code has it.
4. **Delete only what is not vessel.** Only the prune deletes a dead end. Every other step that
   leaves one is accounted for by the census.
5. **One meaning for "no flow"**: an edge attribute `flow_state` ∈ {`solved`, `blind_end`,
   `no_boundary`, `island`, `off_path_loop`} beside the boolean `FLOW_SOLVED`, read by the viewer,
   statistics and exports.
6. **Results stay comparable.** The face-end boundary model is recorded on the graph
   (`G.graph["open_end_model"]`), like `viscosity_law`, and perturbations may not override it.
7. **Peers' modules are not edited while their work is in flight** (`prune.py`, `diagnostics.py`'s
   inside-another-lumen rule, `thick_vessel_junctions.py`). The seams are agreed with them first.

---

## 6. Phases

Each phase lands on its own commit(s) with its tests (CLAUDE.md, Testing policy), measured on E14.5
from a `git archive` snapshot, and on Mason and the Nerve fixture from Phase 3 on.

### Phase 0 — Measure (no behaviour change)

- **Dead-end census in `scripts/graph_artefacts/`**, beside the lumen-artefact report:
  - by place: face bands (2 / 5 / 10 / 20 µm) × whether the mask reaches the face, per face axis;
  - by kind (today's `classify_dead_ends`, then Phase 1's);
  - by depth slab (20 µm);
  - chain length and chain length / radius percentiles;
  - what lies ahead of each tip (another vessel's side, another tip, a face) within 5 / 10 / 15 µm;
  - **which step made each dead end**, from the per-step pickles the harness already writes,
    including what cluster collapse silently removed or joined (inconsistency 8).
- **Flow coverage** from a solved run (`scripts/graph_artefacts/flow_coverage.py`): the solved
  share of the centreline; unsolved length by reason (behind a closed face end, behind an interior
  dead end, island, loop off every path); each boundary node's distance to the nearest face; and
  the topology what-if of section 2.2.
- **Record E14.5 at a named commit** (after the Bridge session's split lands), plus Mason and the
  Nerve fixture. That is the evidence `flow.6c` asks for; update its row.
- **Fixtures** (`tests/dead_end_fixtures.py`), one per kind, each with its answer:
  1. a vessel leaving through a face;
  2. a vessel whose graph stops 8 µm short of a face while the mask runs on to it;
  3. an interior tip 7 µm from a face whose mask does **not** reach the face (not a face end);
  4. a tip the mask runs on 20 µm past, ending in the tissue;
  5. two tips facing across a 6 µm gap (reuse `test_facing_ends.py`'s);
  6. a tip pointing at another vessel's side across a 5 µm gap;
  7. a blind sprout 4 radii long;
  8. a Lee spur and a hair along a wall (reuse `tests/lumen_artefact_fixtures.py`);
  9. an island;
  10. a tip within 5 µm of its own junction (for cluster collapse).
- Tests: `tests/test_graph_artefacts_tool.py` (census rows on the fixtures),
  `tests/test_dead_end_fixtures.py` (each fixture is what it says).

Size S–M. **Can start now**: it touches only `scripts/` and new test files.

### Phase 1 — One definition

- **`graph/dead_ends.py`**: `chain_to_junction`, `tip_heading`,
  `leaves_image(G, node, extent, support=None)`, `classify_dead_end(...) -> DeadEnd(kind,
  measures)` (the measures are what the grids are spread across), `classify_dead_ends(G, ...)`, and
  `mark_dead_end_kinds(G)`, which writes `dead_end_kind` on the nodes.
- **Callers switched**: `diagnostics._judge_dead_ends` (face tips classified, not skipped),
  `network_review.find_dead_ends` and `find_boundary_issues`, `boundaries.open_terminal_nodes`,
  `mask_recovery._on_image_face`, the facing join's chain walk, and
  `optimisation/candidates.terminal_edge_lengths_um`. **Not yet** the prune's face protection
  (that changes the graph: Phase 3). The inside-another-lumen rule is taken from the One
  centreline session as it lands.
- `build_network` writes `dead_end_kind` after the thick split; `post_process` refreshes it; the
  hover (`gui/branch_hover.py`) names the kind instead of "(open end)".
- The small bugs of 3.2 item 11, and the multiple-0 disagreement (item 4).
- Tests (new `tests/test_dead_ends.py`): every fixture gets its kind; **every caller gives the same
  verdict on the same fixture** (the one-definition test); the face test's two cases (fixtures 2
  and 3); chains through degree-2 nodes and through bridges.
- E14.5: the graph does not change, only the counts move between kinds. Check node for node.

Size M. **Starts when the One centreline session's tips work has landed** (it edits
`diagnostics.py` and `prune.py`), or now in a worktree and rebased onto it.

### Phase 2 — Marked grids (calibration; `graph.6b`)

Three grids of ~30 from E14.5, built from the Phase 1 measures, spread across the candidate
measures and shuffled (CLAUDE.md procedure; keys under `outputs/dead_end_review/`; the interactive
review page in `outputs/loop_review/review_page/`, so the user can draw where a vessel goes):

- **G1, interior dead ends.** Stratified by depth (surface 0–40 µm against deep) and by the parent /
  stub radius ratio (decision D8), and spread across the features the published classifiers use
  (Kaufhold 2012, Ji 2021): chain length / radius, the stub's own radius, the raw intensity drop at
  the tip, how far the mask runs on, the distance to the nearest unconnected end and how well the
  two headings line up, and what lies ahead.
  Marks: **S** real blind end (sprout), **B** broken (draw where it continues), **T** the graph
  stopped short (the mask continues), **X** artefact (delete), **F** leaves the image, **?** unsure.
- **G2, near a face.** Tips 2–25 µm from a face, including the 29 face-protected short stubs
  already keyed in `outputs/dead_end_review/short_dead_ends_key.csv`. Marks: **F** leaves the
  image, **I** interior (and if interior, S / B / X).
- **G3, joins.** The 111 tip-to-side candidates and the facing pairs the join refused. Marks:
  **J** the join is right, **N** not the same vessel.

What to draw: the segmentation in the tip's own plane (what the code sees), the element judged
highlighted, nearby centrelines, a scale bar and the voxel position. **For G1 and G3, add a
raw-image panel**: "sprout or broken" is a question about the tissue, not the mask, and the header
says so. Where the run has an endothelial channel, add a third panel.

Fit each rule and report it honestly against marking everything one way, with the IDs it gets
wrong. **Check it on a fresh grid** (G1', G3') before Phases 5–7 rely on it. Turn every kind the
marks reveal into a fixture. Save each fit (right, wrong, baseline) beside its key: that is the
evidence `graph.6b` lacks today ("`dead_end_review` has keys and no marks").

Size: M of agent time, plus the marking. **G2 can start right after Phase 0.**

### Phase 3 — Face ends in the graph

- **Prerequisite:** `PLAN_dale_review_fixes.md` item 2 (pad, operate, crop), so closing stops
  eroding the face slices. Do it here if nobody has.
- **Trace on to the face.** A `leaves_image` tip short of the face (fixture 2) is extended through
  the mask to the face (`route_through_mask`), so the boundary node sits on the face and the
  vessel's length is not cut short. New step `extend_dead_ends` in `graph/dead_ends.py`, after
  `join_facing_dead_ends` and before `consolidate_lumen` (Phase 5 adds the interior case to the
  same step). Agree the STEP_LABELS change with both peers and fix the stale counts at the same
  time.
- **The prune's face protection becomes `leaves_image`**: a real face end is never pruned, and a
  near-face interior spur is judged like any other. This is the One centreline session's prune;
  agree it with them.
- **Boundaries**: `mask_stump`, mask-overlap assignment and `edge_percent` take only
  `leaves_image` terminals, falling back with a warning as `edge_percent` does today. The six
  E14.5 boundary nodes 11–23 µm inside the image should move to the face. The boundary review uses
  the same test. `boundary_open_end_max_distance_um` folds into the face test (`replaces`, so old
  configs load).
- Tests: fixtures 1–3 through `build_graph_from_skeleton`; a large-vessel stump ending short of the
  face gets its boundary node on the face; `test_boundary_open_ends.py` still passes through the
  alias; `test_prune.py`'s face-stub test (`:111`) rewritten to the new rule on purpose.
- E14.5 before/after: face ends, boundary nodes' distance to the face (target: all within the tip
  radius + 1 µm), interior dead ends, coverage; the One centreline plan's targets no worse.

Size M. After Phase 1 and Dale item 2.

### Phase 4 — Face ends in the flow (decision D1)

Today a face end is closed. Options, all kept selectable so results can be compared (an
`open_end_model` setting, advanced, in the Haemodynamics tab's Network handling box):

- **(a) `closed`** — today: no flow through any face end.
- **(b) `pressure_by_type`** — each open end gets a pressure for its vessel class (arteriolar,
  capillary, venular) from the run's Art / Cap / Ven labels and a published pressure–diameter
  relation. Falls back to (d) where an end has no label.
- **(c) `estimated`** — Fry, Lee, Smith & Secomb 2012 (section 2.3): the open ends get no
  conservation equation, and the solve minimises the deviation of pressures from a target p0 and
  of wall shear stresses from a target |τ0| (31 mmHg and 15 dyn/cm² in the paper; scaled to the run's
  own inlet and outlet pressures here, which are themselves a setting), with the designated inlets
  and outlets fixed and flow directions settled iteratively. **The established method** (Sweeney
  et al. 2018 estimated 97.7% of a cortical network's boundary conditions this way). The most code:
  a constrained least-squares solve with an outer sign loop, in both matrix paths. Its targets must
  be in the units and pressure range of the run (`inlet_p_bc` 4,500 Pa, `outlet_p_bc` 1,000 Pa).
- **(d) `terminal_resistance`** — each open end drains through a resistance to a reference
  (venous) pressure: R = k × its own vessel's resistance, or the resistance of the bed beyond the
  face estimated from the network's own capillary density. One parameter. Dale's branch did this
  with an uncalibrated k = 10 (its "Robin" ghost node, `df8c3ab`). **No published use on cut
  capillaries was found**; Blinder et al. 2013's R_m on cut arterioles is the nearest. Cheap, so
  useful as a comparator.
- **(e) `outlet`** — every open end at the outlet pressure (Dale's "universal sink"). Simple and
  biased toward draining; listed for completeness, not recommended.
- *Not planned*: coupling to a homogenised capillary continuum (Peyrounette et al. 2018) and the
  Bayesian boundary calibration of Rasmussen 2025. Both are research projects; (c) gets most of
  the benefit.

Built whatever D1 says:

- The solve takes **open ends as a third boundary role** with a boundary-condition object
  (pressure, resistance-to-pressure, or unknown), in both the dense and the sparse paths
  (`haemodynamics/resistance.py`). `inlet_to_outlet_vessels` and `mark_flow_solved_edges` treat
  open ends as possible sources and sinks.
- Haematocrit: an open end with inflow brings in the inlet haematocrit, and the
  distributed-haematocrit loop treats it as an inlet for that pass.
- **"Network resistance" with open ends**: the pressure drop over the inflow through the designated
  inlets, with the net flow through open ends reported beside it. The analyses that read inlets and
  outlets (routes, transit time, territories, current flow, occlusion) classify each open end by
  its solved flow sign.
- Perturbations and sweeps keep the baseline's open-end conditions; `open_end_model` joins
  `INCOMPARABLE_OVERRIDES`.
- **Sensitivity report** (`flow.8b`): network resistance, total inflow, the capillary flow
  distribution and the share of flow directions that change, under (a), (b), (c) and (d) with
  k = 1, 10, 100, on E14.5 and Mason, for the whole network and with a margin next to the faces
  left out (the literature puts every method's errors near the boundary). Without it, D1 is a
  guess. Order of building: the open-end role and (a), (d) first (cheap), then (c), then (b).
- Tests: a tube cut by a face carries flow under (b) and (d) and none under (a); Kirchhoff at every
  node, the ghost node included; dense equals sparse; red cells conserved with inflowing open ends;
  a perturbation cannot change the model; old runs load as (a).

Size L. **The infrastructure can start now** (its files are not the peers'); the default waits for
D1.

### Phase 5 — Vessels the graph stopped short of (`stops_short`; 114 on E14.5)

- First find out **why**, from Phase 0's per-step record: recovery skips them (the continuation
  lies inside the tip's own lumen ball, or under `recovery_min_region_volume_um3`), cluster
  collapse or the prune shortened them, or Lee thinning retreated from a flared end.
- **Trace on** (the interior case of Phase 3's `extend_dead_ends`): from each `stops_short` tip,
  follow the mask along its heading (the mask route of Post processing's Add vessel) until the path
  meets a centreline (join), a face (it becomes `leaves_image`), another tip (join), or the end of
  the mask (stop: now a blind end at the true end). Guards: mask support; not beside a vessel in the
  same lumen (`MaskSupport.shadows`); not closing a loop inside one lumen
  (`lumen_loops.loop_inside_one_lumen`); a turn limit.
- The prune's mask-continues raise (threshold → `min_stub_length`) then only meets tips the trace
  could not extend.
- Decide inconsistency 8 here: give cluster collapse the face and mask rules, or leave it and log
  what it removes.
- Calibrated on G1's T marks. Tests: fixture 4; a trace that would run beside another vessel stops.
- E14.5 targets: `stops_short` ≤ 10; pairs in one lumen, doubled centreline and loops inside one
  lumen no worse. The One centreline plan learned that recovery re-traces far rims: watch them.

Size M. After G1. It is mask recovery's territory, so agree it with the One centreline session.

### Phase 6 — Vessels the segmentation broke (`broken`)

- **Tip-to-side joins**: extend `facing_ends` so a tip whose heading meets another vessel's side
  within the gap is joined by splitting that vessel (`graph.edit.insert_node_on_edge`). Keep the
  facing join's guards (one stretch of background; not beside a vessel in the same lumen), and add
  two: not a sibling (the graph path from the tip's junction to the target is longer than a few gap
  lengths), and a plausible angle at the target.
- **Evidence beyond the mask**, where the run has it, as Kaufhold 2012 and Ji 2021 did:
  - the candidate path found in the **raw image**, not only the mask: an intensity-weighted
    shortest path from the tip (the Add vessel tracer's raw route), or the mask grown by lowering
    the threshold near the tip ("threshold relaxation");
  - the path's intensity against the vessel's own and the local background (signal-to-noise),
    its length and straightness, and the radii either side;
  - the **endothelial channel** (`endothelial_channel`). A dextran image only shows perfused
    lumen, so a gap can be a collapsed or unfilled stretch that the endothelium still shows
    (Coelho-Santos et al. 2021 found unfilled sprouts only this way).

  The published classifiers reach 88–95% against human labels with features like these; G3 says
  whether a simple threshold on them is enough here, or whether a small classifier fitted to the
  marks (and to the Post processing review's saved decisions) is needed.
- The broken vessels `remove_edges_off_the_mask` leaves (inconsistency 9): count them in Phase 0;
  if they matter, run the join again after consolidation's off-mask removal.
- Automatic only above the evidence threshold G3 calibrates (decision D3). The rest go to the review
  list, renamed "Broken vessels", which shows tip-to-side candidates too.
- E14.5 expectation: up to +2.7 mm on through paths (an upper bound; section 2.2).

Size M. After G3.

### Phase 7 — Real blind ends: keep, label, count

- Kept. D2 of the One centreline plan stands: blind ends under 3 radii are still pruned.
- **`flow_state = "blind_end"`** for a vessel whose only reason for carrying no flow is that it
  leads to blind ends: its zero flow is the solution, not an unknown. The viewer draws it apart from
  unsolved, the hover says so, and every flow statistic states whether it includes them.
- **A "Dead ends" section in the statistics CSV**: counts by kind; blind ends per mm of vessel and
  per mm³ of tissue (with `measure_tissue_volume`); the blind-end length distribution (sprout
  length); the share of vessels cut by the image faces and whether length statistics include them
  (decision D5); islands (count, length). Glossary entries (`stats.6a`).
- Flow statistics can leave out a margin next to the image faces (an advanced setting, off by
  default), since every boundary model's errors sit there.
- On E14.5, blind-end density by depth becomes a biological readout once G1 says the blind ends are
  real. No published sprout density exists for E12–E15 brain, so it would be new data, and worth
  checking against an endothelial stain before anyone reports it.

Size S–M. After Phase 1; the biology after Phase 2.

### Phase 8 — Exports and blanks

- With `PLAN_dale_review_fixes.md` item 5 (unsolved flows blank in the VTK, flags exported): add
  `flow_state` per edge, and `dead_end_kind` and the boundary role per node.
- Nodes with no path to a boundary keep `pressure = 0.0` (`resistance.py:514-535`; 1,131 nodes on
  graph A). Write NaN in the exports, and on the node too if nothing relies on the 0 (check
  `gui/results.py`, `statistics/` and the perturbation CSVs first).
- `network_measures.py`, `inlet_outlet_routes.py` and the other flow statistics read `flow_state`
  instead of testing `== 0` or non-finite values (inconsistency 10).
- The connectivity CSV notes a dead end at either end, by kind.
- Tests: `test_vtk_io.py`, `test_statistics.py`, `test_graph_connectivity.py`.

Size S. After Phase 1 and Dale item 5.

### Phase 9 — Post processing review

- The Dead ends list groups by kind, with actions per kind:
  - `leaves_image`: **Make it an open end**, Keep;
  - `stops_short`: **Trace on** (Phase 5's proposal drawn; Accept or Edit);
  - `broken`: **Join** (proposal drawn);
  - `artefact`: Delete;
  - `blind`, `unknown`: **Mark as blind end**, Trace on, Delete, Keep.
  Isolated single edges are listed too (today they are absent).
- Record the decision after the action, not before (3.2 item 11). Append it to
  `{stem}_dead_end_review.csv` with the item's measures, like the loop review, so later rule fits
  can use the marks, and keep it with the run (`.haemorun`'s `review_decisions`).
- A dead end left by a hand deletion, or by the graph editor window, appears in the list at once.
- Tests: `test_gui_post_processing.py` (pure: grouping, actions per kind, CSV rows),
  `test_network_review.py`, one widget test (`-m gui`).

Size M. After Phases 1, 5 and 6.

### Phase 10 — Validation and rollout

- Before/after for every phase on E14.5, Mason and Nerve: the census, flow coverage, network
  resistance, build time. The One centreline plan's targets must not get worse.
- An invariant in `tests/test_pipeline_invariants.py` (Nerve): every dead end has a kind, none is
  `stops_short`, and every boundary node is `leaves_image` or deliberately interior.
- CLAUDE.md (Key modules: `graph/dead_ends.py`; the open-end model beside the viscosity model),
  schema help text, `python examples/regenerate_configs.py`, the release-review rows, memory.

---

## 7. Decisions needed

**Decided 2026-10-12 (Harvey): every recommendation below is adopted, and the plan is to be
implemented.** How each one is carried out:

- D1, **decided 2026-10-12: `estimated` is the default** (separate commit, revertable). On E14.5 at
  this branch the estimate converged in 7 passes, nothing clamped. 39 of 68 open ends feed the
  network and 29 drain it, at 1,168-2,459 Pa. Through the open ends 30% as much blood enters as
  through the designated inlets. Its flow directions agree with closed on 86% of vessels, but
  with the drain-only models (terminal resistance, outlet) on only 56-65%. Network resistance
  moves under 8% across all five; solved share 57.7% closed, 64.9% open
  (`outputs/open_end_sensitivity_e14/REPORT.md`). Runs that use it cite Fry et al. 2012.
- D1 as first set out: the open-end role is built with (a) closed, (d) terminal resistance, (e) outlet and (c) Fry
  estimation ((b) needs a published pressure–diameter table and is not built). (a) stays the
  default until the sensitivity report has run; the default is then chosen from it.
- D2, D3, D7, D8 wait for the user's marks on G1–G3. Until then every rule that changes the graph
  ships behind a setting that keeps today's behaviour, and the classification (read only) uses the
  first proposals of section 4.
- D4, D5, D6: built as recommended.

**The marks (Harvey, 2026-10-12; `outputs/dead_end_review/FIT.md`).**

- D2: the definition of section 4 gets 24/30 on G2, against 23/30 for distance. The defaults stay
  on distance until a fresh grid confirms it. Its misses are tips at the deep z face, which were
  marked as ending inside, and one tip 2.0 µm from a face.
- D3: review only. The best rule gets 89% and has not been checked on a fresh grid.
- D7: open. Surface and deep dead ends are marked artefact equally often (8/14 and 7/15).
- D8: no change. The radius ratio does not separate the marks.
- New: 54 of 87 marks say the dead end is not a real vessel, and no measure here separates them.
  That calls for the raw image in the grid (D6), not a threshold.

| # | Question | Who | Recommendation |
|---|---|---|---|
| D1 | **The boundary condition at open face ends**, and the default | Harvey, Finbar | Build the open-end role with (a) and (d), then (c) Fry estimation, the literature's standard; run the sensitivity report; choose the default from it, most likely (c). Keep (a) as the default until then, so no number moves silently |
| D2 | What counts as leaving the image | Harvey, by G2's marks | The rule of section 4, with thresholds from the marks |
| D3 | Joins across gaps: automatic above a calibrated threshold, or review only | Harvey | Automatic only where G3 and a fresh grid agree on ≥ 90%; the rest listed |
| D4 | A blind end's flow: "zero flow (blind end)" or "unsolved", in the viewer and the statistics | Harvey | `blind_end`, drawn apart; each statistic says what it includes. **Decided 2026-10-15 (Harvey): unsolved.** The viewer draws `blind_end` vessels grey with their flows blank, and the hover reads "Unsolved: behind a dead end". The exports keep 0 with `flow_state=blind_end`, also Harvey's choice (same day) |
| D5 | Length statistics with vessels cut by the image faces | Harvey | Report both, with the censored ones flagged |
| D6 | The raw image and the endothelial channel as evidence for broken against blind | Harvey | Yes when the run has them; the mask alone otherwise. Kaufhold and Ji repair gaps from the raw image, and Coelho-Santos found unfilled sprouts only with an endothelial label |
| D7 | The deep tissue's dead ends: biology or lost segmentation | Harvey, by G1's depth strata | Cassot et al. 2006 saw the same rise with depth and blamed filling and contrast. If G1 agrees, fix it upstream (segmentation, ilastik), not in graph building |
| D8 | The stub prune's scale: the radius at the junction (today) or the stub's own radius with the part inside the parent left out (Voreen) | Harvey, by G1's radius-ratio strata | Decide from the marks; agree any change with the One centreline session, whose prune it is |

**Where each decision stands (checked 2026-10-24 against `origin/Devel_GUI_improvements_HD` at
`60ea7681`).**

| # | State | Evidence |
|---|---|---|
| D1 | taken: `estimated` is the default | cd71285c; `schema.py:5101` |
| D2 | open: the defaults stay on distance; the fresh grid G2′ that would confirm the definition was never built | `boundary_open_end_test` default `distance` (`schema.py:2592`), `stub_face_protection` default `threshold` (`schema.py:3528`); no G2′ in `outputs/` |
| D3 | review only: joins listed, never automatic | `join_dead_ends_to_vessel_sides` default off (`schema.py:3513`); G3 best 24/27 fitted (`FIT.md`) |
| D4 | taken 2026-10-15: unsolved in the viewer, 0 with `flow_state=blind_end` in the exports | 10c69230; `branch_hover.py:75` |
| D5 | partly built: the share of vessel length on chains cut by the image faces is reported; the length statistics are not yet given both with and without them, nor the cut vessels flagged in them | `statistics/dead_ends.py:96-103` |
| D6 | adopted, no rule: G4 put raw and endothelial contrast in the grid, best 20/27 LOO (74%); no dead-end rule reads the raw image | `FIT_G4.md`; `preprocessing/raw_evidence.py` is read by no `graph/` module |
| D7 | open: the deep dead ends are no more often artefacts than the surface ones, so no upstream fix follows | `FIT.md` (G1, 8 X / 6 S at 0–40 µm, 7 X / 8 S deeper) |
| D8 | no change: the prune keeps the radius at the junction | `FIT.md` (G1 radius ratio median 1.88 X, 1.74 S) |

Every repair ships off: no rule reached 90% on G1–G3 (`FIT.md`, 2026-10-12), and G5's route join
for `stops_short` scored 23/29, the same as never joining
(`outputs/stops_short_route_review/G5_marks.csv`, Harvey, 2026-10-21).

---

## 8. Targets (proposed; confirm, then fixed as the release review requires)

On E14.5, graph as saved:

- every dead end has a kind, from one definition used by every module (pinned by a test);
- every boundary node lies within its tip radius + 1 µm of a face, unless placed deliberately
  (then flagged);
- face ends short of the face with the mask reaching it: 0;
- `stops_short`: ≤ 10 (114 today);
- the fitted classifier: ≥ 90% on a fresh grid (`graph.8b`'s proposed figure);
- pairs in one lumen, doubled centreline, loops inside one lumen, tips in another lumen: no worse;
- unsolved centreline: 100% explained by `flow_state`, and the through-path share recorded for each
  open-end model;
- build time: ≤ +10%.

---

## 9. Risks

- **Opening face ends moves every haemodynamic number** in a run. So (a) stays the default until D1,
  the model is recorded on the graph, and perturbations refuse to change it.
- **Wrong joins make false loops.** Joins need evidence and marks, and the Short loops review
  catches what slips through.
- **Tracing on can re-trace a far rim**, the One centreline plan's lesson about recovery. The shadow
  and in-lumen loop tests guard it, and the pairs and doubled centreline measure it.
- **Over-joining deletes biology**: E14.5's sprouts are real. The depth strata of the grids show
  whether the deep dead ends are sprouts or losses.
- **The counts change meaning**: the face test moves ends between face and interior, so the
  lumen-artefact report's dead-end rows mean something else after Phase 1. Say so in the commit and
  in the release-review log.
- **Shared files**: `assemble.py` (STEP_LABELS), `stages.py`, `diagnostics.py`, `schema.py`,
  CLAUDE.md. Use a worktree, rebase and re-measure before landing, and agree the STEP_LABELS change
  with both peers.

---

## 10. Order

| Phase | Depends on | Main files | Size | Start |
|---|---|---|---|---|
| 0 Measure | — | `scripts/graph_artefacts/`, new tests | S–M | now |
| 1 One definition | the One centreline session's tips work | new `graph/dead_ends.py`; diagnostics, review, boundaries, recovery, facing ends | M | when that lands (worktree now) |
| 2 Grids | 0 (G2); 1 (G1, G3) | `outputs/dead_end_review/` | M + marking | G2 after 0 |
| 3 Face ends, graph | 1, Dale item 2 | `dead_ends.py`, `prune.py` (agreed), `boundaries.py`, `large_vessel_network.py` | M | after 1 |
| 4 Face ends, flow | infrastructure: none; default: D1 | `haemodynamics/resistance.py`, `graph/connectivity.py`, the solve in `stages.py`, `haematocrit_distribution.py`, `perturbations.py` | L | infrastructure now |
| 5 Stopped short | G1 | `dead_ends.py`, mask recovery (agreed) | M | after G1 |
| 6 Broken | G3 | `facing_ends.py`, `graph/edit.py` | M | after G3 |
| 7 Blind ends | 1; the biology after 2 | `prune.py` (`flow_state`), `statistics/`, `gui/results.py` | S–M | after 1 |
| 8 Exports | 1, Dale item 5 | `vtk_io.py`, `statistics/`, `connectivity.py` | S | after 1 |
| 9 Review | 1, 5, 6 | `gui/post_processing.py`, `network_review.py`, `_widget.py` | M | after 6 |
| 10 Validation | every phase | harness, invariants, docs | — | throughout |

The two biggest results, in order: Phase 4 (about a fifth of E14.5's centreline gains flow), and
Phases 5–6 with the Phase 2 grids (the 16 mm hanging behind interior dead ends, split into what is
real and what is broken).

---

## Sources

Gathered 2026-10-10 by web search. Not verified: Kirst et al. 2020's dead-end statistics,
VesselGraph's bulge-size values, Lorthois 2011's boundary tests, Linninger's handling of cut
vessels, and any embryonic sprout density (none was found).

- Kaufhold et al. 2012, gap bridging and strand deletion learned from labels:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3443315
- Ji, Kleinfeld et al. 2021, whole-brain microvasculature, end and linker classifiers:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8525211
- Tsai et al. 2009, "hair" rule: https://pmc.ncbi.nlm.nih.gov/articles/PMC4972024
- Blinder et al. 2013, cortical angioarchitecture and flow:
  https://neurophysics.ucsd.edu/publications/Phil_Tsai_Pablo_Blinder_NAT_NEURO.pdf
- Drees et al. 2021, Voreen vessel graphs, bulge size: https://arxiv.org/html/2102.03444v1
- VesselVio: https://pmc.ncbi.nlm.nih.gov/articles/PMC9046271/
- ClearMap 2 graph processing (code):
  https://raw.githubusercontent.com/ChristophKirst/ClearMap2/master/ClearMap/Analysis/Graphs/GraphProcessing.py
- Damseh et al. 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6546554
- VesSAP, Todorov et al. 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7591801
- Cassot et al. 2006, free ends rising with depth:
  http://www.zib.de/userpage//prohaska/2005/docs/Cassot_et_al_Microcirculation2006.pdf
- Hahn et al. 2021, E13.5 whole brain: https://pmc.ncbi.nlm.nih.gov/articles/PMC8217891/
- Wilsch-Bräuninger et al. 2024, E12.5/E14.5 tip-cell processes:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10945550
- Coelho-Santos et al. 2021, sprouts unfilled by dye, found by endothelial label:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8256089
- Fantin et al. 2010, hindbrain sprouts: https://pmc.ncbi.nlm.nih.gov/articles/PMC2938310/
- Fry, Lee, Smith & Secomb 2012, boundary conditions estimated from target pressure and shear:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3407827
- Sweeney et al. 2018, cortex with the Fry method: https://pmc.ncbi.nlm.nih.gov/articles/PMC5778006
- Schmid et al. 2017, depth-dependent cortical flow: https://pmc.ncbi.nlm.nih.gov/articles/PMC5347440/
- Gagnon et al. 2015, cut capillaries removed: https://pmc.ncbi.nlm.nih.gov/articles/PMC4339366
- Gould et al. 2017: https://pmc.ncbi.nlm.nih.gov/articles/PMC5363755
- Peyrounette et al. 2018, coupling to a capillary continuum:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5764267/
- Rasmussen 2025, calibrated boundary pressures: https://pmc.ncbi.nlm.nih.gov/articles/PMC12450458/
- d'Esposito et al. 2018, zero flow at internal nodes:
  https://www.biorxiv.org/content/10.1101/219865.full.pdf

