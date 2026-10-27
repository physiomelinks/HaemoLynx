# Dead ends and open ends

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## Dead ends and cut-off pieces after hand edits

**Dead ends and cut-off pieces after hand edits** (2026-10-18) — `post_process` classifies the dead ends among the touched nodes against the run's segmentation (`_mark_touched_dead_end_kinds`: a `MaskSupport` over `network.volume.image`, `graph.dead_ends.reclassify_dead_ends`), as `build_network` classified every dead end; it used to classify none, so the end of a drawn vessel, or the one a deletion left, had no `dead_end_kind` and the open-end boundary condition, `FLOW_STATE` and the VTK fell back to `node_leaves_image`'s distance rule -- a vessel drawn to 7 um short of a face with the segmented vessel running on through it was not an open end -- and a node a vessel was drawn from kept the mark it had at degree 1. A touched node that is a dead end no longer loses the classifier's mark; a person's review mark (`blind`) stays whatever its degree, since the edit can be undone and nothing reads it meanwhile. Without a network, a synthetic one or an empty segmentation nothing is judged. And `graph.branch_order.assign_branch_orders` now pops `branch_order` from an edge the search from the inlets does not reach (it already did from bridges): after a deletion cut a piece off, its vessels kept the orders they had (B04, and the table's 4 um with them); a fresh build gives such a piece none. The stamp has no table entry for a vessel with no order, so the diameter it had stays, counted `unset`. The hierarchical passes skip (not pop) the edges another pass writes (`included_edges`/`excluded_edges`), so only an edge no pass reaches loses its order.

## graph/dead_ends.py

**`graph/dead_ends.py`** — what each dead end (degree-1 node) is, the one definition every
module asks (PLAN_dead_ends.md): judged as its whole chain (`chain_to_junction`, the prune's
walk), it is `leaves_image` (its tip touches a face, or the segmented vessel runs on from it to
a face within `FACE_REACH_RADII` of its radius: `leaves_image`), `island_end`, `artefact` (off
the mask, inside a neighbouring lumen, beside a vessel, under the prune's radius rule),
`stops_short` (the mask runs on past it), `broken` (another vessel straight ahead across one
stretch of background, or a lumen it ends in, that is not one of its neighbours -- 2+ junctions
away, the One centreline session's marks), `blind` (a person said so: `set_dead_end_kind`,
kept through reclassifying) or `unknown`. These are first proposals. Harvey marked grids
G1-G3 (`scripts/dead_end_review/`) on 2026-10-12, and no rule reached 90%. The face rule
scored 24/30, against 23 by distance and 20 for all one way. Most ends were marked "not a real
vessel", and no measure separated those. So the kinds only label and list, and every repair
below stays off. `build_network` marks every dead end's kind on
its node (`dead_end_kind`) and the image extent on the graph (`image_extent_um`), so stages
without the mask -- boundaries, the solve, statistics, the VTK -- read the same answer
(`node_leaves_image`, `open_end_nodes`); `post_process` judges the dead ends the hand edits
left the same way (`reclassify_dead_ends`, over the touched nodes alone: a touched node that
is a dead end no longer loses the classifier's kind, a person's stays). Diagnostics, the
review lists and the boundary review all classify through it; a fixed `image_face_margin_um` is only a caller's explicit choice.
`remove_pieces_without_junction` (Graph tab, off by default, 2026-10-20) removes every piece
with no junction at the end of graph building, before the kinds are marked and the graph is
saved (`remove_pieces_without_junction`: the chains whose dead ends are `island_end`, any
length, anywhere; rings and lone nodes stay; the count and length go on
`G.graph["pieces_without_junction_removed"]` / `_um`). Harvey's AIA patella grid P1
(`outputs/patella_dead_end_review/`) marked 11 of 14 such pieces specks and 3 real vessel the
segmentation cut off, and none of 30 patella dead ends a real blind end.
`remove_interior_dead_ends` (Graph tab, off by default -- on from 2026-10-22, turned off
again the same day by Harvey: on the nerve test crop it took 26 chains, 129 -> 87 vessels and
5,477 -> 4,404 um, and the seven-vessel fixture 10 vessels -> 1;
`recovery_short_loop_keep_hole_over_um`, 5.3 um, behind the same Advanced button;
`stages._remove_and_recover_interior_dead_ends`, after `smooth_straight_steps` and before
the junctionless pieces): every interior dead end's chain (`stops_short`, `unknown`,
`artefact`, `broken`; not `leaves_image`, an island's or a person's `blind`) is removed to
its junction (`dead_ends.remove_interior_dead_ends`: all chains found first, so the answer
is order-free, and a junction left with nothing goes too -- a tree with no vessel leaving
the image goes whole, and the empty-build error names the setting); then mask recovery runs
with coverage by a centreline's own radius alone and keeps only what joins the network at
both ends (`recover_uncovered_mask_vessels(coverage_reach_radii=0, two_ended=True)`,
`keep_two_ended_recovery`); what it drew is marked `interior_recovery`, smoothed and bridged
into thick vessels like any vessel; the lumen clean-up runs with the 5.3 um hole limit; the
interior dead ends it leaves are removed a second time. Counts on the graph
(`interior_dead_ends_removed`, `_um`, `interior_recovery_drawn_um`, `_kept_um`). Fitted to
Harvey's marks on E14.5 alone (2026-10-21, `outputs/stops_short_route_review/`): G5 (a
trace-on along a route through the mask joined no better than never joining, 23/29 each;
17 of 29 `stops_short` ends "stop here"), the remove-then-recover graph he called the best
yet, L1 (at 10 um the short-loop rule broke a real loop 16 times in 24, 12 of 16 of them on
recovered strands) and the fresh L2 (5.3 um 19/29 against 13 at 10 um and 16 keeping all;
below 5.3 um about even). On E14.5: 211 chains, 7.4 mm, out; 6.2 mm drawn back joined at
both ends; 43.9 mm, 194 dead ends and 310 loops after, against 47.8 mm, 396 and 231 built.
The same graph under the low-RAM option (tests/test_build_network_stage.py). Not yet
tried on another dataset.
Speed, output identical (2026-10-22): `densify_polyline` is vectorised (the loop's floats,
pinned against it in tests/test_shared_helpers.py; 8.5x on E14.5's polylines) and
`_fill_tree_gaps` keeps each edge's samples between fills rather than resampling the graph
after each (2.8 s a fill). The step 196 -> 163 s on E14.5 (one pair); whole builds 633/658 s
old against 653/600/673 s new, within the machine's noise, every node, edge and attribute
identical, in RAM and low-RAM (`outputs/stops_short_route_review/speed_check.py`,
`build_check.py`). Recovery's routes (~35 s: a window's EDT and MCP each) are left: a shared
window changes the route.
The solve writes `graph.FLOW_STATE` beside `FLOW_SOLVED`: `solved`, `blind_end` (behind dead
ends inside the tissue: drawn unsolved, grey, while exports keep its 0 -- Harvey, 2026-10-15), `closed_end`
(behind a face end with no boundary condition) or `no_boundary` (no inlet and outlet: the
only flows exports blank, `graph.flow_is_undefined`). The solve, each perturbation re-solve
and each sweep point then set `blind_end` and `closed_end` flows and pressure drops to exactly
0, not the solve's rounding (`graph.zero_flows_behind_dead_ends`: call it after any new
`set_edge_flows` on a marked graph). In the viewer (2026-10-21, after Harvey read arrows on
dead-end vessels as a fault): the vessels from an open end the solve gave a condition to its
junction hover as "Solved: fed/drains through an open end ({model})"
(`gui/branch_hover.open_end_chain_text`, from `G.graph["open_end_conditions"]` and the node
pressures); a node's hover says "open end: leaves the image" only where `node_leaves_image`,
else "dead end: {kind}" (`node_end_text`, the nodes layers' `end` column) -- every degree-1
node read "open end"; and no flow arrow is drawn on a vessel at or below the statistics' floor
for carrying flow (`visualization.flow_direction.flow_arrow_floor`, `carrying_floor`: 1e-9 of
the largest), where its sign is rounding. On E14.5 that hides no arrow: the open ends' arrows
are real flow (`estimated` fed through 34, drained through 53), the least 1.7e-20 m^3/s, 1.6e-8
of the largest -- small, not rounding.
`graph/dead_end_extension.py` holds the
repairs the grids gate, each off until then: `extend_dead_ends_to_faces`,
`trace_on_dead_ends`, `join_dead_ends_to_vessel_sides` (run in the facing-join step), with
`stub_face_protection` and `boundary_open_end_test` choosing `leaves_image` over the old
distances. Morphology (closing, opening) reads past an image face as the vessel running on
(`memmap_support.at_faces_as_beyond`), so no vessel the image cut is trimmed at the face.

## haemodynamics/open_ends.py

**`haemodynamics/open_ends.py`** — `open_end_conditions(model, G, ...)`: the boundary condition
on dead ends leaving the image (in a piece with a designated inlet or outlet). Every model
becomes conductances to two ghost nodes held at the inlet and the outlet pressure
(`augment`), never added to the graph: `terminal_resistance` drains to the outlet level,
`outlet` holds the end at it, and `estimated` (Fry et al. 2012, a constrained least squares
over the free pressures with flow directions iterated until those of the vessels carrying flow
settle) becomes, exactly, the conductance carrying each end's estimated flow at its estimated
pressure -- so the baseline reproduces the estimate and perturbations and sweeps (whose unit
solve scales with the drop) keep it. The baseline keeps its conditions on the graph; every
solver path (the solve stage, the haematocrit loop, `_solve_network`, `solve_sweep_point`)
augments with them. `scripts/open_end_sensitivity.py` re-solves one network under each model.

## Finishing PLAN_dead_ends.md (2026-10-24)

What landed on `claude/plan-dead-ends-36f84f`, and why each choice:

- **Fresh grids G1', G2', G3'** (`scripts/dead_end_review/build_grids.py --fresh`; pages and keys
  `outputs/dead_end_review/G1b`-`G3b`, git-ignored; `FRESH.md` beside them). Built from E14.5
  (Harvey's Claudin segmentation, config.yaml) at fc1d8607, replayed from the thick-bridges
  session's capture of `finish_network_graph`'s input: 394 dead ends (91 leaves_image, 95
  stops_short, 93 unknown, 17 artefact, 4 broken, 94 island ends) against 1,082 on the graph
  G1-G3 came from. No tip within 3 um of one shown on G1-G5 or the stub and tips-in-lumen keys;
  no dead end on two grids. The rules each checks were fixed first (`fit_marks.FRESH_RULES`):
  G1' `beside_share > 0.0104` is X; G2' the classifier's `leaves_image` (and distance <= 11.3 um,
  depth <= 10.4 um) is F; G3' `ahead_turn_deg <= 30.5` is J. G3' looks 15 um ahead, not 10: at
  10 um the graph has 14 interior tips with a vessel ahead, 4 of them under 30.5 deg.
- **Node pressures stay 0 on the graph.** Transit time refuses a network with any non-finite node
  pressure, and a resumed run reads node pressures back as its solution, so NaN on the node would
  have switched transit time off on every network with a piece no inlet and outlet reach. The VTK
  already blanks them; the viewer's nodes layer now does (`gui/results.unsolved_node_pressures`:
  a node whose every vessel is unsolved, the viewer's own rule for vessels).
- **Open ends in the network analyses.** `_resolve_terminals` adds every open end the solve gave a
  condition, as an inlet where blood enters through it and an outlet where it leaves
  (`open_end_directions`, above the carrying floor). Pinned as an equivalence: each analysis gives
  exactly what it gives with those ends designated.
- **Branch-order pressure drop** left vessels behind dead ends and in pieces with no inlet and
  outlet in, each with a drop of exactly 0: the order's mean read low. They are out now.
- **"Dead end" counts in network handling** counted loops off one node, parallel pairs and
  self-loops; those are `removed_loop_*` now.

## The fresh grids marked (2026-10-25)

Harvey marked G1'-G3' (90 marks; `outputs/dead_end_review/FIT_G1b_G2b_G3b.md` in the
`plan-dead-ends-36f84f` worktree). None of the rules fixed beforehand reaches 90%:

| grid | rule | right | all one way |
|---|---|---|---|
| G1' | `beside_share > 0.0104` is X | 13/21 | 12/21 |
| G2' | the classifier's `leaves_image` is F | 14/26 | 18/26 |
| G2' | `face_distance_um <= 11.3` is F | 18/26 | 18/26 |
| G3' | `ahead_turn_deg <= 30.5` is J | 21/30 | 29/30 |

- The turn-angle order the review lists had been given on G3's fit was taken out again.
- G3': 1 of 30 joins was the same vessel. Joins stay review only.
- G1' by depth: below 40 um, 12 S / 2 X / 1 T; in the top 40 um, 7 X / 6 T / 2 B, no S. G1 (another
  segmentation) showed no difference.
- G1' and `stops_short`: 7 of the classifier's 9 `stops_short` tips were marked S, 2 X, none T;
  Harvey's 7 T marks fell on `unknown`, `broken` and `artefact`. Nothing separates T from the
  rest (best 24/30 against 23/30).
- G2': tips near the deep z face marked as leaving even 12-14 um out; some the definition calls
  leaving (x faces, one at the top face) marked as ending inside.

## The raw image at dead ends: Phases A and C of PLAN_dead_ends_raw.md (2026-10-25)

`preprocessing/raw_evidence.py` gained the three wall readings dead ends need (`bafd3aa2`):
`ring_along` (E1), `inside_walls` (E2) and `walls_ahead` (E3). They measure only, and nothing reads
them yet. Grid DR1 (`scripts/dead_end_review/build_dr1.py`, `9debc4d1`) asks of 30 fresh E14.5
dead ends whether the chain is a vessel and whether the vessel goes on past the tip. It went to
Harvey on 2026-10-25.

On E14.5 (claudin-5, 394 dead ends, one process) the ring fit costs 0.032 s a section, `walls_ahead`
7 ms a dead end and `inside_walls` 2 ms a point; `ring_calibration` took 86 s on 8 workers. There:

- the vesselness share reads "walled" at three in four interior tips;
- the fitted rings find none at half of them, because a capillary's lumen fills in under the
  2.7 um z blur;
- E2 puts most centrelines on the wall;
- E3's run spreads (median 6 um).

The full numbers and why DR1 is spread by the fitted rings are in the plan's Phase C and Risks.

Harvey marked DR1 the same day (`outputs/dead_end_review/FIT_DR1.md` in worktree
`plan-dead-ends-raw-04d14a`), and no rule reaches 90%:

- Whether the chain is a vessel (21 V / 9 N): every E1 and E2 measure scores 21/30 leave-one-out,
  the same as all V. Only chain length or a tip in the top planes reach 22/30.
- Whether it goes on past the tip (9 Y / 12 E, 7 unsure): E3's run over 11 um scores 15/21 against
  12/21. Its misses are vessels whose walls stop along the straight line ahead, the vessel likely
  turning.

E4 (`route_ahead`, `8f2a0a3f`), at Harvey's call after DR1: a tracker that follows the walls from
one lumen radius past the tip, steering by the wall all round over three steps ahead, within 60 deg
of the heading, and stopping heading out of the image. Those last two came after reading DR1's
marked dead ends: at 100 deg it turned into neighbours and along the image face. On E14.5 it costs
0.13 s a dead end and runs a median 10 um against E3's 6. Grid DR2 (`9a906d10`) judges it on 30
fresh dead ends, with its three rules fixed in `fit_marks.FRESH_RULES["DR2"]` beforehand.

Harvey marked DR2 the same day (`outputs/dead_end_review/FIT_DR2.md`), and the fresh check fails:

- **The rules fixed beforehand** (E3's DR1 cut, E4's run, E4 running on) score 16/26 each,
  against 18/26 for calling every dead end "ends here".
- **No E3 or E4 measure** fits better than that.
- **The route** was right 18 times in 26, and nothing E4 measures says when.

Across G4, DR1 and DR2 the marks follow shape in context, not the endothelial stain. Phase D
builds nothing. Harvey closed the plan the same day without Phase B: E1-E4 stay in
`raw_evidence.py` as tested measures, and nothing (the classifier, the census, `dead_ends.csv`, the
review lists) reads them.

## Dead ends at a thick vessel's wall (2026-10-25, PLAN_bridging.md 1.3 and 0.2)

`thick_wall`: a dead end whose tip lies within the lumen's radius plus `WALL_JOIN_REACH_UM`
(15 um since grid P2, 10 um before) of the thick vessels' centreline tree, and further than the split's tolerance of it
(`thick_bridge_centreline_tolerance_um`, or 30% of the radius) -- the dead ends the split's wall
join looks at -- which the join left. It takes the place of `stops_short`, `broken` and
`unknown`; an `artefact` stays one (it is no vessel's end), as do `leaves_image` and
`island_end`. The tree is the one the split bridges to: `stages.mark_dead_end_kinds` records it
on the graph (`record_thick_centreline`, `G.graph["thick_centreline"]`: its points, the lumen's
radius at each, the tolerance floor), so post processing's re-judging, the review lists and the
statistics read the same kind without the thick region, as they read `image_extent_um`. A graph
built before has no record, and no `thick_wall` ends until it is built again. It is an interior
kind (`remove_interior_dead_ends` takes it, as it took the kinds it replaces) and counts in the
interior dead-end density; the statistics' Dead Ends section gives it its own row. The repairs in
`dead_end_extension.py` (off) read `stops_short` and `broken` only, so they pass these ends by.

The Post processing tab lists them in "Dead ends at thick vessels"
(`network_review.find_dead_ends_at_thick_walls`, after "Dead ends", nearest the lumen first;
the "Dead ends" list keeps them too, under its Show filter). Each item carries how far the tip
stopped from the lumen (negative inside it), the turn from its heading onto the thick centreline,
and the point of the graph's own thick centreline nearest the tree there (`thick_edge`,
`thick_point_um` on the `DeadEnd`; `ahead_edge` / `ahead_point_um` on the item): "Join to the
vessel ahead" lands there, and post processing bridges the stretch inside the lumen. Delete takes
the dead end's own chain only.

E14.5 (Harvey's Claudin segmentation, the build at `155699f4`; `outputs/thick_junction_review/thick_wall_2026-10-25/`): 36 of
393 dead ends are `thick_wall` (20 were `stops_short`, 16 `unknown`); of the 58 tips within reach
the rest leave the image (15), are artefacts (7) or lie on the tree (4). 30 of the 36 turn more
than 80 degrees from the way to the thick centreline (all but one more than 60): vessels running
beside the thick one or away from it, like the wall join's 4 turn refusals on grid P, all of
which Harvey marked N ("not a junction"). 2 stop inside the lumen; 30 have a join point on the graph's thick
centreline. The record is 2,822 tree points (45 kB), and judging every dead end with it took no
longer than without.

## The Nerve fixture's default run, and the facing join's stretches (2026-10-27)

The Nerve fixture's run at the defaults stops at the solve: the inlets and outlets
`edge_percent` picks lie in separate pieces of the graph. They always have since `edge_percent`
became the default (`a9649822`, 2026-08-08); until the dense solve's guard (`f85b8f60`,
2026-09-03) the run went on and solved every flow in the unjoined pieces to zero, with a warning.
The segmentation is 36 separate pieces (26-connected) and nothing in it joins the top face to the
bottom: the inlets' piece and the outlets' piece join only across jumps of 15.9, 3.7, 14.2 and
25.2 um of background, through three other pieces, while no vessel the build draws crosses more
than 6.2 um. The figures and scripts are in `outputs/de_fresh/` (git-ignored).

The 3.7 um jump was the facing join's to make (two dead ends 5.1 um apart, turns of 14 and 28
degrees) and it refused it as crossing two stretches of background. Both vessels are one voxel
thin, staircases of voxels touching only at their edges, and the straight line read background
for 0.4 um where it passed between two of them. `facing_ends._background`, which the `broken` kind
reads too, now takes background between two mask voxels that touch as no stretch
(`tests/test_facing_ends.py`, `tests/test_dead_ends.py`, on `dead_end_fixtures.thin_diagonal_pair`).
Nerve at the defaults: 129 -> 128 edges, 12 -> 11 pieces, 22 -> 20 `unknown` dead ends, the
inlets and outlets still apart. E14.5 (the thick-bridges capture): the graph identical, digest
`cfe70e01`, every dead end's kind the same; its 9 pairs within 10 um all turn too far.
