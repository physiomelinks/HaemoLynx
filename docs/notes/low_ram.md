# The low-RAM option (use_memmap_loading): one mask reader, no whole-volume copies, timings

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## One reader of the mask a run

**One reader of the mask a run** (2026-10-18, for `use_memmap_loading` on large stacks) —
`build_network` makes one `preprocessing.MaskSupport` over the binary mask and hands it to graph
building (`build_graph_from_skeleton(mask_support=)`), both clean-ups after it, the thick-vessel
split (`insert_thick_vessel_junction_nodes(mask_support=)`, under its own limits through
`MaskSupport.limited_to`) and the dead-end marks: the mask is packed to bits once and one
surface KD-tree is built a run, where each step packed it again and the split's wall join built
a second tree. Once packed, every point-in-mask read uses the bits: the lumen test of degree-2
merging and collapse (`mask_lumen_test(support=)`, `MaskSupport.lookup_volume`), the lumen
radius's "is this voxel background" (`FeatureDistance(packed=)`, set by the support when it
measures the very same array), and the EDT diameters' section planes and radii on a memmapped
mask. The same answers, from RAM rather than a disk page read at random -- which only pays once
the mask is out of the page cache: on E14.5 (75 M voxels, cached) a lookup cost ~30 ns either
way. The thick tree's inscribed radius is measured once (`ThickCentreline.radius_um` ->
`tree_radius_um=`) and a block of its box at a time, reading only blocks within 60 um of a point
(`_inscribed_radius_um`); mask recovery traces a piece along the skeleton from the piece's own
voxels (`_trace_skeleton_in_region`, skan's paths from the coordinate graph `graph.build` builds)
rather than a padded window round it. Both bit for bit; on a 210 M-voxel stack (traced numpy
peak) the radius went 425 -> 72 MB and a 24,000-voxel diagonal piece 1,419 -> 1 MB
(`outputs/lrb/peaks.py`, git-ignored). E14.5 low-RAM, three runs each from a `git archive`
(`outputs/lrb/bench.py`): every graph identical to the old code's, node for node, edge for edge,
every attribute, before and after diameters; `build_network` 225 -> 233 s median, within the
machine's noise (the in-build clean-up ranged 49-140 s on both, 74.3 s each under the profiler
back to back), the thick-vessel split 14.4 -> 9.9 s (every run), diameters unchanged (EDT
86 -> 88 s); built again on `6f05617` with the skeleton re-trace, identical, 218 -> 209 s. The skeleton A* helpers in `graph/_helpers.py` and degree-2's overlap count read
the skeleton a voxel at a time but no pipeline step calls them.
Temp files: a `new_memmap_array` array nobody releases has its file deleted once it and its
views are gone (`weakref.finalize`; on Windows, which keeps a mapped file, at the next
`new_memmap_array` or at exit). Unreleased were the skeleton `skeletonise` hands on, the float32
raw image the raw evidence and the Diameters stage load (two 301 MB files a low-RAM E14.5 run,
a volume's worth of float32 each on any stack) and the volumes tests compare -- 2,806 files in
one machine's `%TEMP%`. Killing a run still leaves its files.
**No whole-volume copies under the low-RAM option** (2026-10-19, for a 5.5 G-voxel stack on
33 GB; the centreline tree's CSR, the join's last resort, Lee tiling and the dense solvers are
separate): every step below gives the run bit for bit, and its tests are in
`tests/test_low_ram_volume_copies.py` beside the modules' own. Reconnect's routing workers
start with one OpenMP/BLAS thread (`reconnect._routing_environment`: threadpoolctl holds only
in the parent, and each worker's OpenBLAS committed ~659 MB, which ran an E14.5 build out of
paging file) and are at most `shared_volume_tasks.default_worker_count()`. The lumen radius
reads tiles of the surface (`FeatureDistance(by_block=True)`, through
`MaskSupport(low_memory=)` and `mask_radius_sampler(low_memory=)`: a k-d tree per 96-voxel
tile, made when a query first needs it, least recently used dropped past 256 MB or, when
memory is free, past a quarter of it shared by every such lookup alive
(`SURFACE_TILE_MEMORY_FRACTION`, read again each 64 MB: 256 MB alone held a tenth of the
uncropped patella's 2.4 GB of tiles and mask recovery remade tiles all step), searched ring
by ring until the best is no further than the nearest voxel unsearched) instead of one tree
over the whole surface (~69 B a surface voxel, 7-9 GB). A tile is made from the packed bits
a byte at a time (`PackedMask.box`), its surface by a separable 3x3x3 dilation
(`pointwise_distance._dilate_box`, scipy's own answer) and its tree unbalanced and
uncompacted (twice as fast to build and query; the tree only narrows candidates, so the same
distances). Skeletonisation hands on the thick
tree it drew as its voxels (`SkeletonisedVolume.thick_vessel_centreline`,
`thick_vessel_centreline_voxels`, `edt_ridge_voxels`), which the split, `ThickCentreline` and
post-processing take in place of a volume of it, so `build_network` neither redraws it (a
volume of zeros and a Dijkstra over every fat voxel) nor holds a volume for it. The gated
skeleton is drawn on disk; the region cache and the vessel-mask cache unpack a slab at a time
onto disk (`memmap_support.unpack_bits_by_slab`), and a cached "no region" is a broadcast
False. Components whose sizes, membership or voxels are wanted come from slabs, not an int32
label memmap (22 GB of dirty pages each, which emptied the page cache on a patella run):
`skeleton._SlabComponents` for the fraction filter, the connectivity stats, both
small-component drops, the bridging and `skeletonize_by_component` (each component's mask
written as the slabs pass its box and thinned once its last plane is read). The thin-arm join
looks component numbers up at the skeleton's voxels only, int32; a bridge's new fat voxels take
theirs along the bridge drawn (two 26-adjacent voxels of one mask share its component:
`thick_vessels._numbers_carried_along`, from the arm voxel it starts at and the tree voxel it
stopped against), and only a voxel neither reaches is read from its slab labelled again as it
was (`slab_labels.NumbersAnywhere`; read voxel by voxel, those reads were four fifths of the
patella's join, 2026-10-20). It held 16 B a fat voxel, twice.
Every `label` of a whole vessel or pericyte mask is `slab_labels.ForegroundLabels` (each
foreground voxel's key and int32 number, 12 B a foreground voxel, numbered as `label` numbers;
scipy's own volume is 8 B a voxel from 2**31 voxels): the volume filter, the minority swap,
the opposite-attached and overlap clean-ups (the last ~130 GB on 5.5 G voxels), the stump
points, the small-mask redefinition, the continuity descriptors, the quality metric and the
pericyte centroids (scipy's `center_of_mass` is a C-order bincount, done on the foreground);
the masks are read and cleaned onto disk (`load_and_validate_vessel_masks(use_memmap=)`).
Post-processing binarises the segmentation once, on disk, for its three readers and one
support (`stages._PostProcessSegmentation`); the overlay projection counts and maxes a slab at
a time; mask recovery's worker bits go to the run's `memmap_directory`
(`MaskSupport.memmap_directory`), not `%TEMP%`. Checked against `4c96e9c5`
(`outputs/lvc/old_vs_new.py`, git-ignored: 60 results of the changed functions, dtypes too,
identical) and on E14.5 (low-RAM, skeletonise through diameters, `outputs/lvc/e14/bench.py`,
a `git archive` of `a82c7e9` against this commit: the skeleton, the built graph and the graph
after diameters identical, node for node, edge for edge, every attribute; private peaks
skeletonise 2.74 -> 2.30 GB, build_network 3.32 -> 2.98 GB, boundaries 2.61 -> 2.01 GB; the
radius sampler 3.8 s -> 0 and the tree build_network drew 5.9 s -> 0; skeletonise 643 -> 792 s
and build_network 207 -> 187 s, one run each with other jobs holding most of the RAM, so not
a timing). Peaks, old -> new, on a 120 M-voxel synthetic
stack, each in a fresh process (`outputs/lvc/peaks.py`, private bytes; the new ones are a
32 M-voxel slab's work and do not grow with the stack, the old ones do): a routing worker
627 -> 16 MB (20 cores); lumen radius, 20,000 lookups, 154 -> 44 MB; the tree build_network
drew, 363 MB and 8.8 s -> nothing; region cache read 244 -> 62 MB; connectivity stats 458 MB
of label file -> 372 MB in RAM and no file; Lee by component 572 MB of temp files -> 368 MB
and the result's file; the vessel-mask volume filter 1,376 -> 252 MB; the overlap clean-up
2,064 MB and 41 s -> 409 MB and 4.4 s; post-processing's binarisations 574 MB and 5.2 s ->
367 MB and 1.3 s; the projection 344 -> 367 MB (no smaller at this size, constant past it).
The gated skeleton (1 B a voxel) and the join's lookups (16 B a fat voxel, twice: 4.6 GB on
the patella's 144 M) are gone from RAM by construction.
Coordinates that stand in for `np.argwhere` (`ForegroundLabels.coords` / `component_coords`,
the skeleton's slab components, the thick tree's voxels) are laid out as it lays them out,
Fortran order (the transpose of the index arrays): a BLAS product of them rounds by the
layout, and C-ordered rows moved a small-mask continuity bridge (one piece's principal axis
flipped sign; fixed in `b6c602a`, pinned in the volume-copies tests). Since then, also under
the low-RAM option: small-mask continuity reads its four distance transforms at the voxels
its gates ask about (`mask_continuity._DistanceAt`: the lumen radius's tiles, and scipy's
"no feature" reading -- every voxel that far from one at index (-1, 0, 0) -- where a mask is
empty), draws a bridge in its own box (`_bridge_box_from_line`; a rejected candidate used to
come with a volume of zeros) and reads the other type's masks together where asked; the
arteriole | venule unions are read where asked (`memmap_support.MaskUnion`: the large-vessel
cut, the small-mask redefinition's labelling) or written on disk a slab at a time
(`as_global_vessel_mask(use_memmap=)`, the thick-vessel restriction), and the quality gate
counts its overlap a slab at a time; a vessel mask's dilation is compared with its radius a
block at a time (`dilate_binary_mask_by_microns(low_memory=)`, through the loader and each
step of the confidence assignment) -- scipy's own transform also "dilates" an empty mask into
the volume's corner, and so does this; and segmentation cleanup's reconnect labels its
components a slab at a time (`ForegroundLabels`, read by boxes) instead of an int32 label
file. Identical to `a82c7e9` on random fixtures (`outputs/lvc/continuity_check.py`,
`union_check.py`). The small-mask progressive dilation and the hand-edit tags read their
distances from the surface's tiles too (`_distances_to_mask_microns(low_memory=)`). Peaks,
`b6c602a` -> `3bdb785`, on the same 120 M-voxel stack (`outputs/lvc/peaks2.py`, traced numpy,
the same answers): continuity (on 40 M voxels of it) 2,712 MB and 62 s -> 207 MB and 6.4 s; a
2 um dilation 5,608 MB and 33 s -> 580 MB and 20 s (only voxels with mask inside the reach's
box are measured); the union's 1 B a voxel goes to disk; the reconnect's label file (4 B a
voxel: 744 -> 515 MB of temp files) is gone, its RAM peak unchanged at 2.0 GB -- the blockwise
EDT's block, whose float64 transform is still a file (8 B a voxel). E14.5 low-RAM, a `git
archive` of `b6c602a` against it with `3bdb785`'s changes (`outputs/lvc/e14b/`, git-ignored,
stopped after the boundaries, the small masks `make_masks.py`'s stand-ins): four runs reaching
every one of these steps but the hand-edit tags -- A the cleanup's reconnect (12 bridges), the
thick restriction over all four masks and a 2 um dilation on load; B a global small mask (the
large pair read as their union); C the cut, the confidence assignment, the redefinition (5
pieces moved) and continuity (0 of 11 bridges); D the cut, the legacy progressive dilation
and continuity (4 of 77 bridges) -- gave the skeleton, the thick region, the built graph,
every mask after the boundaries, every boundary list and the graph after them identical, node
for node, edge for edge, every attribute (A stops in both at the large venule mask's missing
stump). Private peaks: the boundaries 7.2 -> 2.1 GB (C) and 7.4 -> 2.3 GB (D); skeletonise,
which loads the dilated large masks for the restriction, 5.0 -> 1.8 GB (B, C); build_network
unchanged (2.9-3.2 GB). Times not compared: a shared machine.
**Two of these in every run** (2026-10-22): timed in RAM against the low-RAM path on one
commit, only continuity's pointwise distances and the block dilation were faster, so normal
runs use them too, keeping their results in RAM (`low_memory` now only puts them on disk):
on a 60 M-voxel synthetic stack continuity 52-60 s and 2.9 GB -> 13 s and 0.5 GB, a 2 um
dilation 33 -> 21 s median of three (noisy) and 2.9 -> 0.76 GB. Continuity still makes the
whole transforms when `use_gpu_mask_continuity_acceleration` finds a GPU. The same masks as
`94d26a26`'s in-RAM path on 484 random cases (80 continuity, 74 drawing bridges; 404
dilations), pinned against scipy's whole transforms in `tests/test_low_ram_volume_copies.py`.
The rest gained nothing in RAM: the overlap and volume-filter clean-ups already share one
algorithm, the surface tiles and point distances tie, and the confidence assignment keeps its
one transform thresholded per step. The low-RAM reconnect refuses big windows (not the same
graph), and the low-RAM skeletonisation steps were slower one by one on that synthetic stack.
**A whole E14.5 run is faster low-RAM, not slower** (2026-10-22, `56912ef4` from a `git
archive`, the repo-root `config.yaml` with only `use_memmap_loading` changed; four runs in the
order in RAM, low RAM, low RAM, in RAM; `outputs/lowram_ab/`, git-ignored): skeletonise
190 / 89 / 91 / 223 s, build_network 427 / 234 / 240 / 437 s, diameters 278 / 319 / 262 s
(no consistent difference), the other stages equal; totals 938 / 685 / 635 s. The last run
had two other Python processes starting during it, so its stages from Diameters on (576 s
there) are not counted. All four graphs identical, every attribute. Part of the gain is the
read-only consistency checks the low-RAM option skips (49 s in skeletonise and 104 s in
build_network in RAM), the rest Lee thinning by component (9 vs 57 s), the blockwise wall
transform (1 vs 21 s), mask recovery (50 vs 86 s) and the lumen clean-up (2-3x). Peaks: 6.1 GB
in RAM; 2.6 GB low RAM in the main process, 4.4-4.5 GB with skeletonise's workers.
