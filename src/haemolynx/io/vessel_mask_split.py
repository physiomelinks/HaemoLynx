"""Large-vessel masks split by width into large and small ones.

A run given only a large arteriole and venule mask (no small masks) may cut
each into the part at least ``large_vessel_mask_split_diameter_um`` wide,
kept as its large mask, and the rest, the small mask of the same side
(``split_large_vessel_masks_by_diameter``, ``graph.small_vessel_masks_split_from_large``).

A voxel is wide where a ball of that diameter fits inside the vessel around
it. The vessel is the run's segmented image, and the mask only says which
side a voxel is on: a classifier's arteriole or venule mask need not follow
the vessel's walls (E14.5's are 4-7 um thick along vessels measured about
20 um wide, so read by their own width most of their large vessels were
small). Without the segmentation each mask is read as the vessel itself:
its opening by the ball.

It is read off the distance transform. The centres are the mask's voxels at
least the ball's radius from the vessel's background, and the wide part is
every voxel of the mask within that radius of a centre, plus half the
coarsest voxel spacing: without it the wall's staircase on an anisotropic
grid leaves a shell of narrow voxels round a wide vessel (a 16 um tube at
2 x 0.6 x 0.6 um kept 97%, in 25 pieces, with none; all of it with half a
voxel). Distances are to the nearest background voxel's centre, as the
thick-vessel gate and the mask's inscribed radius read them.

Both steps go through :func:`~haemolynx.preprocessing.blockwise_edt.near_distances_by_block`,
which compares a distance with any value up to its reach exactly as the
whole transform does, a padded block at a time and only where there is mask:
the same masks whatever the blocks, and on disk under the low-RAM option
(*use_memmap*).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["split_large_vessel_masks_by_diameter", "wide_part_of_mask"]


def _new_volume(shape, *, use_memmap: bool, memmap_directory) -> np.ndarray:
    if not use_memmap:
        return np.zeros(shape, dtype=bool)
    from haemolynx.preprocessing.memmap_support import new_memmap_array

    return new_memmap_array(shape, bool, directory=memmap_directory)


#: Voxels in a padded block, at least. Large-vessel masks are sparse, and only
#: blocks holding mask are transformed: E14.5's arteriole mask (0.2% of a
#: 75 M-voxel stack) split at 12 um in 4.8 s at 500 k, 12 s at 2 M, 36 s at
#: 32 M, the same masks each time.
SPLIT_BLOCK_VOXELS = 500_000


def _wide_part(
    mask, diameter_um, voxel_size_zyx, use_memmap, memmap_directory, segmentation=None, extra_reach_um=0.0
):
    """:func:`wide_part_of_mask`, with the mask's voxel count and the wide part's.

    *extra_reach_um* takes the wide part that much further from the centres:
    a mask dilated at load past the walls of the segmentation it is measured
    on stays wide out to its own edge."""
    from haemolynx.preprocessing.blockwise_edt import near_distances_by_block, reach_voxels

    diameter = float(diameter_um)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(f"diameter_um must be a positive width in microns, got {diameter_um!r}.")
    shape = tuple(int(s) for s in mask.shape)
    if segmentation is not None and tuple(int(s) for s in segmentation.shape) != shape:
        raise ValueError(
            f"The segmentation the widths are read on is {tuple(segmentation.shape)}, "
            f"the mask {shape}: they must lie on one grid."
        )
    sampling = tuple(float(v) for v in voxel_size_zyx)
    radius = 0.5 * diameter
    reach = radius + 0.5 * max(sampling) + max(0.0, float(extra_reach_um))
    # Blocks whose cores are at least twice their halo: a block's transform
    # answers each comparison as the whole one does, whatever its size.
    block_voxels = max(SPLIT_BLOCK_VOXELS, int(np.prod(4 * reach_voxels(reach, sampling))))

    def in_mask(box):
        return np.asarray(mask[box], dtype=bool)

    def in_vessel(box):
        return np.asarray(segmentation[box], dtype=bool)

    centres = _new_volume(shape, use_memmap=use_memmap, memmap_directory=memmap_directory)
    mask_voxels = 0
    found = False
    # The vessel's background is the features: each mask voxel's distance
    # from it (0 for a voxel off the segmentation, which is no centre).
    for core, where, distance in near_distances_by_block(
        in_mask if segmentation is None else in_vessel,
        in_mask,
        shape,
        sampling=sampling,
        reach=radius,
        block_voxels=block_voxels,
    ):
        mask_voxels += int(distance.size)
        deep = distance >= radius
        if deep.any():
            found = True
            block = np.zeros(where.shape, dtype=bool)
            block[where] = deep
            centres[core] = block
    wide = _new_volume(shape, use_memmap=use_memmap, memmap_directory=memmap_directory)
    wide_voxels = 0
    if not found:
        return wide, mask_voxels, wide_voxels
    # The centres are the features: each mask voxel's distance from one.
    for core, where, distance in near_distances_by_block(
        lambda box: ~np.asarray(centres[box], dtype=bool),
        in_mask,
        shape,
        sampling=sampling,
        reach=reach,
        block_voxels=block_voxels,
    ):
        near = distance <= reach
        wide_voxels += int(np.count_nonzero(near))
        block = np.zeros(where.shape, dtype=bool)
        block[where] = near
        wide[core] = block
    return wide, mask_voxels, wide_voxels


def wide_part_of_mask(
    mask: np.ndarray,
    *,
    diameter_um: float,
    voxel_size_zyx: tuple[float, float, float],
    segmentation: np.ndarray | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """The voxels of *mask* where a ball *diameter_um* wide fits inside the
    vessel around them (the module's rule), as a new boolean volume: on disk
    under *use_memmap*, read a block at a time. The vessel is *segmentation*
    (anything a box of it can be read from as booleans), or *mask* itself."""
    return _wide_part(mask, diameter_um, voxel_size_zyx, use_memmap, memmap_directory, segmentation)[0]


def split_large_vessel_masks_by_diameter(
    arteriole_mask: np.ndarray,
    venule_mask: np.ndarray | None,
    *,
    diameter_um: float,
    voxel_size_xyz: tuple[float, float, float],
    segmentation: np.ndarray | None = None,
    dilation_um: float = 0.0,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """``(large arteriole, large venule, small arteriole, small venule)``: each
    mask's part at least *diameter_um* wide (:func:`wide_part_of_mask`), and
    the rest of it.

    *venule_mask* None is a global mask of every large vessel, split the same
    way into a global large and a global small mask. *segmentation* is the
    run's segmented image on the masks' grid, read as the run binarises it,
    which the widths are read on; None reads each mask's own. *dilation_um* is
    the large masks' load-time dilation (``large_vessel_mask_dilation_microns``),
    which widened every mask by twice itself: read on the segmentation, the
    wide part reaches that much further, to the mask's dilated edge; read on
    the mask, it is added to the diameter. Either way the width is the
    vessel's own. *voxel_size_xyz* is the main image's ``(x, y, z)`` voxel
    size, which the masks share.
    """
    from .axis_order import voxel_size_zyx_from_xyz

    voxel_size_zyx = voxel_size_zyx_from_xyz(voxel_size_xyz)
    dilation = max(0.0, float(dilation_um or 0.0))
    diameter = float(diameter_um) + (0.0 if segmentation is not None else 2.0 * dilation)
    extra_reach = dilation if segmentation is not None else 0.0

    def split(mask: np.ndarray | None, name: str):
        if mask is None:
            return None, None
        wide, total, kept = _wide_part(
            mask, diameter, voxel_size_zyx, use_memmap, memmap_directory, segmentation, extra_reach
        )
        if use_memmap:
            from haemolynx.preprocessing.memmap_support import combine_by_slab

            narrow = combine_by_slab(
                lambda m, w: m.astype(bool) & ~w, mask, wide, directory=memmap_directory
            )
        else:
            narrow = np.asarray(mask, dtype=bool) & ~wide
        logger.info(
            f"Large-vessel {name} mask split at {float(diameter_um):.3f} um wide, "
            f"read on {'the segmentation' if segmentation is not None else 'the mask'}: "
            f"{kept} of {total} voxels stay large, {total - kept} are the small "
            f"{name} mask."
        )
        return wide, narrow

    large_arteriole, small_arteriole = split(
        arteriole_mask, "arteriole" if venule_mask is not None else "global"
    )
    large_venule, small_venule = split(venule_mask, "venule")
    return large_arteriole, large_venule, small_arteriole, small_venule
