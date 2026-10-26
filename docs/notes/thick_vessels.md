# Thick-vessel skeletonisation, the large-stack tree and the bridges

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## preprocessing/thick_vessels.py

**`preprocessing/thick_vessels.py`** — `use_thick_vessel_skeletonisation`: Lee thinning of the
fat part of a plasma-labelled mask gives a medial *sheet* (several polylines for one vessel), so
the fat region (an inscribed-radius gate, not a volume gate) gets a centreline *tree* instead,
and the thin capillaries fused to it are joined onto that tree (`_join_thin_arms_to_fat_ridge`:
each arm aims at the tree as it was before any arm joined, and at an earlier arm's bridge only
when the tree is past the bridge cap -- aimed at whatever was nearest, a second capillary joined
the first one's bridge by the wall and the bridges ran side by side through the fat lumen, 21 of
E14.5's 33 pairs in one lumen); `thick_vessel_braid_guard.py` flags braided segments that
remain. Under the low-RAM option (2026-10-18) no step makes a distance map or label volume of
the stack (`blockwise_edt.py`, `slab_labels.py`) and the skeleton and region are bit for bit
the in-RAM ones (synthetic networks, trunks wider than 16 voxels, anisotropic, cut by the faces,
30k-voxel blocks); on a 300 M-voxel synthetic stack (2.7% foreground) it took 211 s at 3.9 GB
private, 2.6 GB above the process's start, against the old low-RAM path's >83 min (stopped). The
thick region is still a byte a voxel in RAM (the skeleton is drawn on disk since 2026-10-19,
and handed on with the tree's voxels: see **One reader of the mask a run**), and the centreline tree's
Dijkstra graph (built straight into CSR) is ~360 B a voxel of the largest fat component, built
whole; the join's last resort, ~400 B a voxel of the connected structure it walks. Its hole fill
labels the background a slab of planes at a time (`skeleton._fill_binary_holes_by_slab`, no label
volume), and the catchment's blocks run in worker processes (`blockwise_edt.near_blocks`, sized by
`parallel_plan` from cores and free memory): catchment 90 -> 51 s and Lee step 72 -> 35 s on that
stack with 14 workers, 7.5 GB with theirs counted. One set of workers serves the hole fill, every
pass of the region and the Lee step's catchment (`shared_volume_tasks.kept_workers`, stopped
before Lee thins here: a worker imports the whole package as it starts, ~17 s with a dozen at
once, and each pass started its own), each mapping its volumes for one task and flushing what it
writes; the free memory the plan and the re-check before each task read (2 GB kept free) is read
once this process's disk-backed volumes are written out and their pages handed back
(`memmap_support.release_resident_pages`): on Windows a mapped file's pages are not available
while in a working set or unwritten on the modified list (14.7 GB of it on the patella, emptying
at ~55 MB/s), so the re-check took the run's own volumes for the machine's memory and held the
patella's catchment to one task at a time (region 925 -> 1,296 s at `dcc41c25`; 916 s with the
re-check off). 75 M-voxel stack: region 132 -> 42 s (45 s in one process); 300 M: catchment
85 -> 25 s, both bit for bit (the patella with this not yet timed). **The large-stack tree**
(`skeleton_thick_vessel_large_stack_tree`, off by default, under the low-RAM option only;
`preprocessing/thick_vessel_tree.py`, 2026-10-19) is for a fat region that is one network
crossing the stack -- Harvey's patella at a 3 um gate, 111 M of its 144 M fat voxels in one
component, ~40 GB of that CSR. Each component's tree is built as `_component_tree` builds it
(root and far end, arms farthest first, the same tests), from shortest paths found over the
voxels themselves (`shortest_path_parents`: bucketed label-correcting, neighbours three to a
row by searchsorted on the sorted keys, a distance and a parent a voxel kept -- scipy's
distances bit for bit, ties to the lower voxel number, so a route of exactly equal cost can
step a voxel aside), the farthest candidate from cells re-measured only near what the tree
gained (`_FarthestCandidates`), and no arm cap. A component of at most
`EXACT_TREE_MAX_VOXELS` (4 M) keeps scipy's walk, so its tree is the in-RAM one bar the cap
(`tests/test_thick_vessel_tree.py`, with digests of the in-RAM skeleton from 4c96e9c5); the
voxels are grouped by component in one array filled as the slabs are labelled, the radius
read along `distances_at_points`' own block rows on up to 4 threads, small components on
threads, and the join's last resort bounded (`_bounded_structure_path`: a box round the
arm's ends, 41 voxels padding doubling, at most 64 M box voxels and 2 M searched; an arm
with no path is left unjoined and logged). E14.5 (Harvey's settings) with it: thick region,
tree and skeleton bit for bit the in-RAM ones. Patella tree step (1312 s at `1d9bf1ab`, its
region read back): labelling 120 s, 5,644 small components 330 s, the big one's radius
119 s, its shortest paths 740 s (~6.6 us a voxel; threads gave nothing there), arms 8 s;
peak 22.2 GB commit, 9.7 GB above the region and skeleton buffer (6 GB unthreaded), during the radius
(four block transforms and the domain's surface tree at once). The whole
`skeletonize_thickness_gated` under it (`e375545b`, region read back, skeleton on disk,
2026-10-20): Lee 343 s, tree 1,079 s, the thin-arm join 2,383 s for 43,254 arms (5,800
joined, 4,215 past the cap, 33,239 with no fat in their structure, 41 left unjoined by the
bounded search), 4,031 s in all at 10.25 GB peak commit; the join's own parts are in its
closing log line (paths 907 s, the nearest-fat tree rebuilt per bridge 552 s, the cap's radius
465 s, the bounded search 360 s). Before the bridge numbers were carried it ran 5.6 arms/s
against 20. **It draws the same arms,
so it does not keep a network's narrower vessels**: arms aim only at voxels of at least 0.4
of the component's widest radius, so the patella's 111 M-voxel component got its main path
and 2 arms (263,706 tree voxels in all; 68% of fat voxels over 20 um from the tree, against
3.5% on E14.5), and the fat voxels no arm reaches are not handed to Lee. What to do there
(hand them back, or aim arms by local radius) waits on Harvey and a marked grid. The region itself (`thick_vessel_region`: the hole fill, the radius map, the body and its
wall; 39 s on E14.5) is kept beside the saved skeleton, one bit a voxel
(`preprocessing/thick_vessel_region_cache.py`, `stages.thick_vessel_region_cache`), keyed on a
digest of the mask's voxels and the settings that decide it, so nothing else can go stale: a
run with the same mask and settings reads it back (`skeletonize_thickness_gated(thick_region=)`),
a run loading its skeleton (`do_skeletonize` off) gets it too, where it used to get none and
make no bridges (with segmentation cleanup on it missed until 2026-10-23: the loaded image was
not cleaned, so its digest was not the fresh run's; both now go through
`stages._mask_skeleton_is_made_from`), and `post_process` finding it again for a resumed run keeps what it finds;
a mask with no thick region is remembered as well. `graph/thick_vessel_junctions.py` (`build_network`'s last step, and `post_process` for
hand-edited vessels) then makes the bridges, by Harvey's rules: a thick vessel is its centreline
alone, and every vessel opening into it is joined by one `IS_ZERO_RESISTANCE` bridge, straight
from where it leaves the thick region (the segmentation within the margin of the large-vessel
body: the thick vessel's segmented boundary) to the centreline; a bridge is never a dead end nor
attached to one. The centreline is the tree `skeletonize_edt_ridge` draws through the region,
found again from the region (9 s on E14.5, so nothing new is carried through resume or a
`.haemorun`). Where the graph's own thick centreline leaves a gap in the tree (3 um or more) and has no way across it under 2.1 times as long, the tree's path is drawn in first (`thick_vessel_fill_centreline_gaps`, on by default since 2026-10-15; fitted on Harvey's grid V, 18/20 against 13 for drawing none, not yet checked on a fresh grid; in post-processing only a gap an edited vessel reaches, as far as its bridge would look, and what is drawn is marked edited so the stage measures it); a sample is on it within 1.5 um or 30% of the local lumen radius, and a stretch
near it counts only if long enough (3 um at an edge's end, 6 um within). Every edge reaching the
region is labelled outside / centreline / inside: inside stretches joined through their nodes
that lead out of the region and meet the centreline, or come within one lumen radius of it, are
removed and each exit gets a straight bridge to where the stretch met the centreline (else the
nearest centreline point, within 2 radii + 5 um; the design's nearest point for every bridge, C.2,
was not built, and Harvey kept the vessel's own line on 2026-10-26, the Phase 5 grid's marks not
separating long bridges from short); a stretch that only skims the region's edge
stays vessel, and the thick vessel's own path where its centreline strays off the tree (the
shortest way between two places a stretch meets the centreline, or a stretch that never leads
out) stays as it is. Stretches under 2 um are flicker. Nothing is ever cut off: a stretch whose
removal would disconnect anything is left as it was. A vessel beyond a bridge that ends blind
inside the image goes with its bridge (one running out of the image stays) when the stub prune
would take it as a stub too: shorter than `min_stub_length` and than
`min_stub_length_radius_multiple` times the thick vessel's lumen radius at the tree point
nearest the bridge (`_remove_blind_vessels_beyond_bridges(stub_length_um=)`; Harvey's grid Y,
2026-10-15: the four marked keep were 34-154 um, stretches of the thick vessel itself and long
capillaries, the two marked remove 6 and 16 um). On E14.5: 150
bridges, each from the region's boundary to the centreline (median length 1.24 lumen radii),
every one leading to a junction or the image's edge, against the old rule's 360 inside pieces of
which 44 were radius-shaped; 42 blind vessels (529 um) removed. Bridges are not measured
(`assign_edge_diameters` runs on the vessels alone; a bridge takes the widest diameter at its
ends, `diameter_source="bridge"`), get no branch order and are no step in one
(`graph/branch_order.py`), and statistics contract them (`contract_bridges`: the vessel meets
the thick centreline directly, sharing the graph's attribute dicts so per-vessel results land on
it) and report them on their own rows. See `PLAN_thick_vessel_bridges.md`. Three cases since,
from Harvey's examples (2026-10-13, `PLAN_bridging.md` F1-F1c): a dead end stopping within
`WALL_JOIN_REACH_UM` (15 um since grid P2, 10 um before) of the thick lumen and heading for it (the gap bridges' 80-degree,
3 um rule, judged on the path drawn as for the reconnects, `reconnect.path_continues_ends`)
is first drawn on through the mask to the thick centreline, under the bridge gate and
the same-lumen guard (a short join judged at its middle) and the loop guards, never closing a loop shorter than the run's
`remove_short_loops_under_um` round a hole no wider than `short_loop_keep_hole_over_um`
-- one the clean-up would break (on E14.5 all 10 short loops the build left ran through such
a join), then bridged like any vessel (`join_dead_ends_at_walls`; the
edge records `joined_at_thick_wall_um`) -- skeletonisation often ends a thin arm at the wall;
a bridge carrying a thick centreline's end on along its own line was, until 2026-10-25, the
thick vessel narrowing and stayed vessel (`TAPER_MIN_COSINE`, 45 degrees, marked `THICK_TAPER`;
dropped by Harvey after grid P, below: it is bridged like any other now, and the clean-up still
leaves an edge marked `THICK_TAPER` alone, as it does a bridge, for graphs saved before); and at
a fork of the thick centreline an arm at most
`FORK_BRANCH_MAX_RADIUS_FRACTION` (0.8) of the trunk's radius gets a bridge over its stretch
inside the trunk's lumen (`bridge_forks`). `thick_vessel_extend_centreline_ends`
(`extend_thick_ends`, **on by default** since 2026-10-20, Harvey's call; not yet checked on a
fresh grid): where the graph's
thick centreline stops at a dead end while the segmented vessel runs on, it is drawn on, along
the tree where the tree runs on with no graph beside it, then through the segmentation (a
centre-seeking route leaving within 60 degrees of its heading) to the first vessel or dead end
within `EXTEND_REACH_UM` (40 um); an image face reached first means the vessel leaves the image
and nothing is drawn (`_extend_thick_centreline_ends`, edge mark `EXTENDED_UM`). At grid W's
16, 20 and 23 the tree ended where the graph did, so following the tree alone could not make
Harvey's joins. Grid Y (`outputs/thick_vessel_breaks/GRID_Y_extension.md`, all 18 such
places on E14.5's two segmentations): 17/18 against 14 for drawing every one, wrong at a 4 um
step out to a face; leave-one-out about 15-16, the reach resting on one example. On Harvey's
E14.5 Claudin segmentation (2026-10-20, rebuilt from his run's skeleton) it drew 12 thick
centrelines on: dead ends in vessels of 5 um radius or more 124 -> 116, `stops_short` 102 -> 96.

## Bridge shapes, shared nodes and drawn bridges (2026-10-24)

`PLAN_thick_vessel_bridges.md`'s last items, on branch `claude/plan-thick-bridges-e32666`
(`d0ca6387`..`a857e40b`). **The bridge-shape report** (`graph/thick_vessel_bridge_shapes.py`,
`diagnose_thick_vessel_bridges`, logged by `build_network` after the split and written by
`scripts/graph_artefacts.py`) puts each bridge in one of the 2026-10-09 audit's six shapes, or
`fork`. Judged by the tree alone on that audit's graph (E14.5's 2026-10-07 checkpoint, 393 bridges)
it finds 273 in the four shapes the split never means to make, against the audit's 284, with the
same median lengths; the audit read the radius at the inner node with a fixed 1.5 um tolerance,
the report the tree's radius with the split's own tolerance. On the current rule every bridge
ending off the tree ends on the graph's own thick centreline where it strays, 2-6 um away, so by
default that counts as reaching the centreline; and a bridge "lies along" it only with 3 um on the
tree, since one shorter than twice the tolerance has most of its length there whatever its
direction. **Design C.4**: bridges ending within one lumen radius of each other share one node
(`_share_centreline_nodes`); on Harvey's E14.5 Claudin segmentation (the graph handed to the split
at `00611a47`, replayed) it moved 11 of 73, and the clean-up after the split then took a different
path at one tangle at the image face (x < 30 um, y 95-150 um), where the split already joins two
nodes by a 4 um bridge and a 47 um vessel: saved graph 1,226 -> 1,221 nodes, vessel length 47,314
-> 47,139 um. There C.4 moves two bridge ends onto the junctions beside them (2.8 and 7 um away),
and the clean-up then takes out a 93 um and a 63 um vessel running along the thick vessel just
outside its region (in the segmentation; 8.7 and 2.3 um median from the tree) and shortens a 57 um
one to 30 um, 152 um of the 175. Drawn for Harvey on both graphs, down z and down x, and kept at his
call (2026-10-26): the tangle reads better with C.4. A judgement on one place, not a marked grid.
**Design C.5**: a bridge whose straight line would cross more than 2 um of background
keeps the way its vessel ran through the lumen (`DRAWN_BRIDGE`): 1 of 73 on E14.5.
**`thick_bridge_centreline_tolerance_um`** (1.5 um, unfitted) replaces the constant;
`thick_bridge_one_radius` was not made (the drawn-path rule it would switch back to is gone, and
Phase 6 runs it from a script), and stays dropped at Harvey's call (2026-10-26); that script is
kept only in the git-ignored `outputs/tb/` of worktree `plan-thick-bridges-e32666`. The Phase 5 grid (30 bridges, 30 near-misses, E14.5 at `47edef60`,
`scripts/thick_bridge_review/build_grid.py`) went to Harvey on 2026-10-24 for marks. **Phase 6**,
E14.5 built and solved at `47edef60` with the rule before 43dcf453 and with this one: bridges 123
(1,847 um) -> 63 (505 um), length over radius median 1.31 -> 1.08, vessel length 46,471 -> 47,139
um, network resistance 5.70e15 -> 2.79e15 Pa.s/m^3 (open ends closed 1.22e16 -> 3.79e15, same
inlet and outlets), `build_network` 218 -> 245 s (the split 0.4 -> 14.3 s); the 55 old bridges
now mostly vessel carry a median 6.4 times their old flow. The thick vessel's own edges sum to
about the same resistance both ways, so the network's fall lies elsewhere (not traced). The Nerve
fixture has no thick vessel (largest inscribed radius 2.8 um).

## The split with a loaded skeleton; grid P; the census again (2026-10-25)

`PLAN_bridging.md`'s last items (`6e09639a`..`f7f1b71f`). With `do_skeletonize` off the region is
read back beside the skeleton and graph building splits as before, so
`use_thick_vessel_skeletonisation`, the region's settings and the split's (gap fill, end
extension, centreline tolerance, large-stack tree) no longer require `do_skeletonize`; the thin-arm
join, the flake filter and the braiding check, read only while skeletonising, do. **Grid P**
(`scripts/thick_junction_review/phase1_rules_grid.py`, sent to Harvey for marks) puts the wall
join, the taper and the fork rule to a grid they were not fitted on: on Harvey's Claudin
segmentation the join's pool is 92 dead ends within 15 um of a thick lumen (5 joined, 51 refused
turning away, 7 beside a vessel or closing a loop, 29 past the 10 um reach), the taper's 26 pairs
(15 tapers), the fork rule's 12 arms (5 bridged). **The census again** (one skeleton and mask,
`040170b` against today): dead ends near a thick vessel 100 -> 54, of them facing it within 80
degrees 21 -> 8, connectors left for want of a centreline 21 -> 9. Grid U's three misses are not
at today's 9 (`outputs/thick_junction_review/REPORT.md`). **Grid P marked** (Harvey, 2026-10-25, on
its interactive page, `scripts/thick_junction_review/grid_page.py`): the wall join 10/14 against 9/14
for all J (its joins and its turn refusals all right; misses 2 beyond the reach, 2 by the in-lumen
loop guard, right on 1 of 3), the fork rule 6/8 against 4/8 (misses at ratios 0.92 and 0.94 among E
at 0.90-0.98), the taper rule 5/8 against 7/8 for all S: 3 of its 4 tapers marked as side entries to
bridge, at cosines 0.80-0.97. **The taper rule dropped** (Harvey's call, 2026-10-25): a vessel carrying
on from a thick centreline's end is bridged like any other (grid P's taper sheet: 7/8, wrong on P15,
the one T). E14.5, one run each to solve on Harvey's Claudin segmentation, with the rule and
without: tapers left as vessel 15 -> 0, bridges in the built graph 63 (505 um) -> 75 (582 um; 57
radius-shaped, 13 longer than the radius, 4 at forks, 1 along the centreline: no new shape), vessel
47,139 -> 47,012 um, dead ends 394 -> 393, vessels solved 1,101 -> 1,119, network resistance
2.96e15 -> 2.65e15 Pa.s/m^3 (-10%), build 196 -> 194 s. The clean-up still leaves an edge marked
`THICK_TAPER` alone, for graphs saved before. **A dead end the wall join leaves** (within its
reach of the lumen, off the tree) is a `thick_wall` dead end since 2026-10-25, judged on the tree
the build records on the graph and listed in Post processing's "Dead ends at thick vessels"
(`dead_ends_and_open_ends.md`): 36 of E14.5's 393. **Harvey's calls on what grid P left** (2026-10-25):
a fork of two vessels of a size keeps no bridge (his marks: 4 near-equal forks E, 2 B); the wall
join's reach and in-lumen loop guard go to grid P2 (`scripts/thick_junction_review/wall_rest_grid.py`:
every unmarked case of each on E14.5, 5 and 4), the 9 connectors the split leaves unbridged to
grid K (`connector_grid.py`: 2 read their reach at a thin tree spur, 1 sunk by one exit), and the
tree fill to a fresh grid on E15, which waits on that stack's pixel size. **P2 and K marked** (the
same day; `outputs/thick_junction_review/grid_P2/FIT_P2.md`, `grid_K/FIT_K.md`): every version of
the join replayed on E14.5 and scored on all 22 marked wall cases (15 J, 7 N): as it is 12/22, the
reach at 15 um 14/22 (two more joins, both J, none N), the in-lumen loop guard or the beside check
dropped 12/22 (the short-loop guard then refuses the same joins). Harvey: the reach 15 um, the
guard kept. Grid K: 4 B, 5 N; today's split 5/9, the reach read at the widest tree radius 4/9 (the
marks contradict the spur cause), each exit bridged on its own 6/9; nothing changes until the E15
grid checks that and the length inside the region (8/9 here, 6/9 leave-one-out).
**A thick vessel's end at the image face is never dropped** (found raising the reach): on
E14.5 at 15 um a wall join put a node on the arteriole's centreline 17 um from the image face,
past the tree's end (skeletonisation stops the tree short of the face). The stretch from there to
the face became an edge of its own, inside the region and off the tree; met at that node by the
joined vessel's stretch, the split took both for one connector, replaced it by the joined
vessel's bridge, and the arteriole's inlet at the face went with it: the inlet snapped to another
vessel, the Large_Art tier fell from 35 vessels to 8 and the network resistance rose fourfold. An
inside run ending at a dead end on an image face is now the thick vessel's own path, kept like its
path through a connector (`_at_image_face`;
`test_a_thick_vessels_end_at_the_image_face_stays_when_a_vessel_joins_beside_it`).

**The Phase 5 grid, sheet 1 marked (Harvey, 2026-10-25).** IDs 1-30 (17 bridges, 13 near-misses;
`outputs/thick_bridge_review/marks.csv`, fit by `scripts/thick_bridge_review/fit_marks.py` into
`FIT.md`, E14.5 at `47edef60`): B 3, V 3, C 13, N 11. The current rule agrees on 9 of 30 (2 of the
17 bridges marked a bridge, 7 of the 13 near-misses marked vessel or thick centreline), against 13
for marking all C and 16 for making no bridge at all. 7 bridges are the thick vessel's own
centreline by Harvey's marks and 6 should not be connected, as should 5 near-misses. Neither
measure the plan assumed -- distance from the tree over the tolerance, length over the lumen radius
-- separates the marks: every single threshold on them, and on the bridge's angle to the thick
vessel's axis, the share of it on the tree, and where and how wide the vessel beyond it runs, gains
at most 3 cases over marking all one way and loses them under leave-one-out, bar one: a bridge's
lumen radius at its outer end over the thick vessel's radius at most 0.35 calls it thick
centreline on 13 of 17 bridges, leave-one-out 13, against 10 all one way, on a margin of 0.33
against 0.36.

**Sheet 2 marked (Harvey, 2026-10-25)**, the fresh check, on an interactive page
(https://claude.ai/artifact/6UKspdmpdnnWcZLLqtXNeB, `page_data.py` and grid P's `grid_page.py`;
marks in its `marks` collection, copied into `marks.csv`): IDs 31-60 (13 bridges, 17 near-misses),
B 7, V 1, C 8, N 14. Both rules fixed before the marks fail. The rule at `47edef60` agrees on 12 of
30, against 14 for marking all N (9 for no bridge at all). Sheet 1's radius at the outer end gets
8 of the 13 bridges right against 11 all one way: the three bridges under 0.35 are all marked N,
not C. Over all 60 (B 10, V 4, C 21, N 25; `FIT.md`, sheet 1's alone kept as `FIT_sheet1.md`) the
rule agrees on 21 against 25 for all N. 23 of the 30 bridges are not bridges by the marks: 9 are
thick centreline, 12 should not be connected and 2 are vessel. No single threshold gains more than
4 cases over marking all one way under leave-one-out. The best is a near-miss within 1.41 times the
tolerance of the tree read as thick centreline, 22 of 30 against 18; next, a bridge across the
thick vessel's axis (|cos| at most 0.16) read as not to be connected, 21 of 30 against 18. Neither
was fixed beforehand, and neither is near 90%. No rule is built and
`thick_bridge_centreline_tolerance_um` stays at 1.5 um. The largest group is what should not be
connected at all, 25 of the 60: that is graph near the thick vessels, not a bridge's shape (12 of
the 25 drawn by the thin-arm join, below), and no measure here finds it. The grid's bridges are those of
`47edef60`, before the taper rule was dropped, the wall join's reach raised to 15 um and equal
forks left unbridged (above): its forks, IDs 4 and 28, and any taper among its near-misses are
judged as that commit made them.

## Decision D2: the large masks as the thick body, gated by radius (2026-10-26)

`PLAN_surface_coverage.md` D2: with `skeleton_thick_vessel_restrict_to_mask: large` every segmented
voxel inside the large-vessel masks is the thick body, so a capillary inside or within the wall
of a large mask gets no Lee centreline. D2's alternative, the body as large mask AND inscribed
radius (of the hole-filled segmentation) of at least half `skeleton_thick_vessel_min_radius_um`
(1.5 um on E14.5), measured on E14.5 at origin `6d5c1bb1` (Harvey's Claudin segmentation, the
main checkout's config, each run to the solve with the restriction swapped for the run; no code
changed; `outputs/sc/d2/d2_run.py`, `d2_score.py`, `REPORT_D2.md` in worktree
`quirky-goodall-bbcb6f`):

| | today | D2 |
|---|---:|---:|
| large-mask segmentation the gate takes off the body | | 27,169 of 428,964 voxels |
| large-mask segmentation left outside the thick region | 0 | 207 voxels (wall absorption puts the rest back) |
| thick region | 878,982 voxels | 870,313 (-1%) |
| skeleton voxels inside the large masks | 3,460 | 3,542 |
| plain-Lee centreline the gate drops | 7,462 um | 7,278 um (2,815 inside the large masks, from 2,897) |
| Harvey's grid P strokes (538 um) with skeleton / graph within 2 um | 49.8 / 39.0% | 49.9 / 37.2% |
| his missing-vessel drawing (1.73 mm) with skeleton within 2 / graph within 3 um | 30.1 / 35.8% | 31.4 / 37.1% |
| pairs in one lumen / doubled centreline / dead ends | 17 / 975 um / 206 | 15 / 935 um / 200 |
| thick-vessel bridges | 77 | 89 |
| network resistance | 1.93e15 | 2.78e15, not comparable: the inlet moved (below) |

The radius gate takes the narrow edge of the large masks off the body, and the 3 um wall
absorption grows the body straight back over it: the vessels Harvey drew inside the large masks
are fused with the large lumen (grid P: median inscribed radius 3.7 um), so a radius floor does
not separate them, as grid P predicted (1-10% released). D2 gives his drawn vessels no more
centreline (+0.1 point on grid P's strokes, +1.3 on the drawing) and is not built. Narrower
large masks remain the other way (retrained ilastik masks, outside this repo).

Found in passing: the 1% change of the thick region moved the arteriole inlet the `mask_stump`
method picks on the x = 497 um face from y 173 um (node 3984 today) to y 438 um, with the same
three outlets; the solve's resistance follows the inlet, so it cannot be compared between the
two runs. A small change of the thick body choosing another stump is worth a look in
`graph/large_vessel_network.py`.

## Where the connections that should not exist come from (2026-10-26)

The Phase 5 grid's 25 N cases ("should not be connected at all"), traced on E14.5 (Harvey's
Claudin segmentation, `config.yaml`) built again at `26ba5ab1`: all 25 are still there, 12 as
bridges and 13 as near-misses. `scripts/thick_bridge_review/capture_steps.py` runs to
`build_network` keeping the skeleton before and after the thin-arm join, the graph after every
topology step and every step after them, and the graph the split was handed (E14.5: 660 s, about
5 GB). `connection_steps.py` looks for each case's course (a near-miss's piece; for a bridge, its
vessel's way from the region's edge to the centreline in the graph the split was handed) in every
step's graph, and puts it down to the first step of the last unbroken run that has it; where the
skeleton's own graph has it, to the skeleton step whose output first joins its two ends. What made
each (voxel z, y, x; b bridge, n near-miss):

| step | cases | IDs |
|---|---|---|
| thin-arm join (`_join_thin_arms_to_fat_ridge`) | 12 | b 13 (15, 284, 360), 18 (26, 162, 1), 22 (13, 269, 396), 35 (11, 370, 502), 39 (10, 279, 236), 41 (6, 136, 163), 52 (25, 253, 467), 56 (15, 246, 411); n 25 (10, 227, 190), 34 (7, 428, 269), 53 (29, 87, 102), 55 (7, 67, 186) |
| skeleton clean-up (gap filling, closing, bundles, thinned again) | 5 | b 24 (22, 470, 78), 54 (15, 339, 450); n 5 (8, 261, 378), 31 (12, 78, 154), 57 (4, 421, 333) |
| mask recovery | 2 | b 26 (20, 253, 482), the wall join's mark on it too; n 48 (28, 236, 497) |
| the tree itself | 2 | b 4 (6, 309, 85), a fork bridge; n 23 (3, 367, 286) |
| lumen clean-up | 2 | n 8 (20, 278, 415), at the end of graph building; n 43 (17, 337, 116), taken out by it and drawn again after the split |
| Lee on the thin catchment | 1 | n 42 (19, 384, 123) |
| the split's end extension | 1 | n 11 (28, 58, 153) |

The clean-up thins the whole skeleton again (25,549 voxels added, 28,980 taken on E14.5), moving a
join's path two or three voxels, so a skeleton step is judged by whether its skeleton joins the
course's ends within 3 um, not by covering it. **The join is the largest group, and it is also how
every other vessel meets a thick one.** It gives each thin skeleton component one path to the
ridge, from the component's voxel nearest it; Lee stops 3 um short of the region, and 90 of
E14.5's 212 paths start on a one-voxel arm that cut leaves (47 of the 212 are still a connection
in the saved graph). Of all 60 Phase 5 cases the join made 32: N 12, C 12, B 6, V 2. Of today's 77
bridges it made 49 (the split 12, the skeleton clean-up 6, the tree 4, other steps 6). **No measure
is worth a grid.** On those 32 (all one way 20/32): the arm's direction against the path
(`join_paths.py`'s `approach_cos`) is undefined on 19, the arm being one voxel, and where defined
spans N 0.03-0.90 and C 0.14-0.94; a path from a one-voxel arm read as real 22/32; a connection
that is the only link to what lies beyond it, read as N, 22/32; the claudin-5 wall along the way in
(`RawEvidenceEndothelial.wall_between`) is under 0.07 on 10 of 12 N and 16 of 20 others read from
the region's edge, and read from 6 um out along the joined vessel peaks (over 1) on 2 of 10 N
against 8 of 16 others, the wrong way. Of the grid's 30 bridges only 2 leave the side of a vessel
running on past them (13, N; 7, C); the other 28 end their vessel. None gains more than 2 of 32
over marking all one way (`FIT.md`'s measures, on all 60, at most 4 of 30 under leave-one-out), so
no grid was built and nothing in the build changed. The 12 C the join made say where
the tree misses the thick vessel's own centreline: the region-coverage track would change what the
join is asked to connect. Harness and traces: `outputs/nc/` in worktree `angry-chandrasekhar-eb21c0`
(git-ignored; `trace_N/TRACE.md`, `trace_bridges/`, `joins/`).

**Grid J, a census of the join's paths** (sent to Harvey 2026-10-26, unmarked;
`scripts/thick_bridge_review/join_census.py`): every path the join drew on E14.5 that the saved
graph still joins and that passes nowhere within 3 um of a Phase 5 piece, 25 of the 212 (only 54
of the 212 arms are in the saved graph at all), shuffled into J01-J25 and drawn as Phase 5 draws,
the join's path in red. Marks J joined here, C the big vessel's own centreline, N should not be
connected at all. No rule is fixed: the key (`outputs/join_census_review/key.csv`, same worktree)
keeps the arm's approach, its voxels beside the start (0 on 10, 1 on 9, 2 on 6), whether the saved
graph has a bridge along it (16), the loop it closes (12 the only link), for a rule to be fitted to
the marks. Page: https://claude.ai/artifact/P35gsPJAhs4A9dKmCi5HQV (marks in its `marks`
collection).
