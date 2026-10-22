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
make no bridges, and `post_process` finding it again for a resumed run keeps what it finds;
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
nearest centreline point, within 2 radii + 5 um); a stretch that only skims the region's edge
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
`WALL_JOIN_REACH_UM` (10 um) of the thick lumen and heading for it (the gap bridges' 80-degree,
3 um rule, judged on the path drawn as for the reconnects, `reconnect.path_continues_ends`)
is first drawn on through the mask to the thick centreline, under the bridge gate and
the same-lumen guard (a short join judged at its middle) and the loop guards, never closing a loop shorter than the run's
`remove_short_loops_under_um` round a hole no wider than `short_loop_keep_hole_over_um`
-- one the clean-up would break (on E14.5 all 10 short loops the build left ran through such
a join), then bridged like any vessel (`join_dead_ends_at_walls`; the
edge records `joined_at_thick_wall_um`) -- skeletonisation often ends a thin arm at the wall;
a bridge carrying a thick centreline's end on along its own line (`TAPER_MIN_COSINE`, 45
degrees), the one most along it at each end, is the thick vessel narrowing and stays vessel
(an elbow at the end keeps its bridge, and so does a side vessel leaving the same end), marked
`THICK_TAPER` so the clean-up after the split leaves it alone as it does a bridge (never one of
a pair in one lumen, a doubled stretch or a short loop's side; degree-2 merging keeps a taper
apart from a vessel as it does a bridge, `taper_meets_vessel`: merged, the mark spread along
the whole vessel and on with every merge, and dropped, the clean-up rewired E14.5's tapers as
second strands); and at a fork of the thick centreline an arm at most
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
