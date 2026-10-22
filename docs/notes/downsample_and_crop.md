# Downsample and crop on import

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## Downsample on import

**Downsample on import** (`io/downsample.py`, 2026-10-18) — two Input tab choices, each
`none`/`2x`/`4x`/`8x`/`16x`: `segmented_image_downsample` (under `segmentation_source`) and
`raw_image_downsample` (under `raw_image_stain`). The segmented image is reduced as it is
loaded, after `crop_on_import`'s box is cut -- binarised first, then by block *majority* (a coarse
voxel is vessel when at least half its block was; 2026-10-19: reduced as read, block max let
ilastik's 1/2 labels' background win every partial block, an erosion, and on binary masks max
widened every vessel ~a coarse voxel and merged neighbours) -- the blocks chosen per axis from the
run's voxel size so a coarse z axis is reduced by less (`axis_downsample_factors`, the rule the
optimiser's own "Optimisation downsampling" uses: one definition, `optimisation/search.py`
imports it), and the run's voxel size is the reduced grid's -- `SkeletonisedVolume.downsample`,
`G.graph["downsample_on_import"]` (`record_on_graph`: the image's `Downsample` and the raw
factor, refreshed from the settings at `build_network`, `assign_diameters`, `post_process` and
`export_results`). Every volume read beside the image is placed on the grid by that record:
the large/small vessel masks (`load_and_validate_vessel_masks(downsample=)`, majority too, the
mask's own voxel size scaled with it, the cache keyed on it), the pericyte and cell masks
(`mask_volume_like_graph`: crop, then block max), and the raw image (`raw_volume_like_graph`:
crop, then block mean -- FWHM/endothelial/raw section, the stain reading, the raw evidence, the
tissue volume, the panel's checks and Add-vessel tracing). **The raw image has to lie on the
segmentation's grid** (every reader compares the two voxel for voxel), so the raw choice only
changes anything when it is *coarser*: a finer one is read at the segmented image's factor, a
coarser one is reduced by its own blocks and each coarse voxel spread over the grid
(`spread_onto_grid`: blockier, no smaller); preflight warns either way
(`checks.check_downsample_on_import`), and ilastik reads the whole raw file regardless. The
`voxel_size_override_xyz` is the image's as read and is scaled too; `load_volume_for_skeletonise`
then returns the reduced grid's voxel size already resolved (status `RUN_GRID_SOURCE`), which
`run_voxel_size_xyz` takes as it is -- the panel's check/optimise and preflight read through it.
A saved skeleton at another factor is refused as one from another crop box is. The skeleton
clean-up's voxel counts (`stages.SKELETON_VOXEL_SETTINGS`) keep their physical size on the grid
(`skeleton_voxel_settings_on_grid`: divided by the finest axis's factor, half up, at least 1).
Measured on E14.5 (`outputs/ds2/`, 2026-10-19, built network against the full-resolution one,
share of its length within max(3 um, lumen radius); a one-voxel shift of the image gives 93.5%):
2x 63.9% before, 72.6% after (69.1% from the majority alone); 4x 30.1% -> 52.4%. What is
still lost is thin vessels (r < 2.5 um: 36% -> 53% kept, 91% under the shift) and short dead
ends, mostly at Lee thinning on the coarse grid; block max keeps more thin ones at 4x but a
third of its graph is then false joins. The optimiser's own downsampling keeps block max. Positions: a
coarse voxel's index times its voxel size is where the first voxel of its block was, so the
crop record's offset is measured in the source grid's voxel size. Not yet tried on a real stack.
**A crop is binarised as the whole file reads** (2026-10-20): which value is vessel (the
minority of an ilastik 1/2 export) is decided from every value the whole file holds, counted a
slab at a time (`io.load.mask_binarization_rule`), then applied to the box as it is cut
(`crop_volume(binarise=)`); the segmented image's rule is kept per file
(`stages._segmented_image_binarization_rule`). Decided on the box, a box holding only label 2
read as solid vessel and one where vessel is the majority came back inside out. The vessel,
pericyte and cell masks were already binarised whole before the cut. A file holding one value
only is read explicitly, with a warning: 0 and 2 (ilastik background) empty, any other value
vessel everywhere -- it was `> 0`, so Harvey's `28_3_d_ipsi_FITC_Simple Segmentation_large.h5`,
label 2 in all 17.1 G voxels, loaded as one solid large vessel and every vessel of the run was
classed Unknown_Large. Pinned in `tests/test_crop_on_import.py`.
