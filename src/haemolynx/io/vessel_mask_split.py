"""Large-vessel masks split by width into large and small ones.

A run given only a large arteriole and venule mask (no small masks) may cut
each into the part at least ``large_vessel_mask_split_diameter_um`` wide,
kept as its large mask, and the rest, the small mask of the same side
(``split_large_vessel_masks_by_diameter``, ``graph.small_vessel_masks_split_from_large``).

A voxel is wide where a ball of that diameter fits inside its mask around it:
the mask's opening by that ball. It is read off the distance transform. The
centres are the voxels at least the ball's radius from the background, and
the wide part is every voxel of the mask within that radius of a centre, plus
half the coarsest voxel spacing: without it the wall's staircase on an
anisotropic grid leaves a shell of narrow voxels round a wide vessel (a
16 um tube at 2 x 0.6 x 0.6 um kept 97%, in 25 pieces, with none; all of
it with half a voxel). Distances are to the nearest background voxel's
centre, as the thick-vessel gate and the mask's inscribed radius read them.

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


def _wide_part(mask, diameter_um, voxel_size_zyx, use_memmap, memmap_directory):
    """:func:`wide_part_of_mask`, with the mask's voxel count and the wide part's."""
    from haemolynx.preprocessing.blockwise_edt import near_distances_by_block, reach_voxels

    diameter = float(diameter_um)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(f"diameter_um must be a positive width in microns, got {diameter_um!r}.")
    shape = tuple(int(s) for s in mask.shape)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    radius = 0.5 * diameter
    reach = radius + 0.5 * max(sampling)
    # Blocks whose cores are at least twice their halo: a block's transform
    # answers each comparison as the whole one does, whatever its size.
    block_voxels = max(SPLIT_BLOCK_VOXELS, int(np.prod(4 * reach_voxels(reach, sampling))))

    def in_mask(box):
        return np.asarray(mask[box], dtype=bool)

    centres = _new_volume(shape, use_memmap=use_memmap, memmap_directory=memmap_directory)
    mask_voxels = 0
    found = False
    # The background is the features: each mask voxel's distance from it.
    for core, where, distance in near_distances_by_block(
        in_mask, in_mask, shape, sampling=sampling, reach=radius, block_voxels=block_voxels
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
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """The voxels of *mask* where a ball *diameter_um* wide fits inside it
    around them (the module's rule), as a new boolean volume: on disk under
    *use_memmap*, read a block at a time."""
    return _wide_part(mask, diameter_um, voxel_size_zyx, use_memmap, memmap_directory)[0]


def split_large_vessel_masks_by_diameter(
    arteriole_mask: np.ndarray,
    venule_mask: np.ndarray | None,
    *,
    diameter_um: float,
    voxel_size_xyz: tuple[float, float, float],
    dilation_um: float = 0.0,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """``(large arteriole, large venule, small arteriole, small venule)``: each
    mask's part at least *diameter_um* wide (:func:`wide_part_of_mask`), and
    the rest of it.

    *venule_mask* None is a global mask of every large vessel, split the same
    way into a global large and a global small mask. *dilation_um* is the
    large masks' load-time dilation (``large_vessel_mask_dilation_microns``),
    which widened every vessel by twice itself: it is added to the diameter,
    so the width is the vessel's own. *voxel_size_xyz* is the main image's
    ``(x, y, z)`` voxel size, which the masks share.
    """
    from .axis_order import voxel_size_zyx_from_xyz

    voxel_size_zyx = voxel_size_zyx_from_xyz(voxel_size_xyz)
    diameter = float(diameter_um) + 2.0 * max(0.0, float(dilation_um or 0.0))

    def split(mask: np.ndarray | None, name: str):
        if mask is None:
            return None, None
        wide, total, kept = _wide_part(mask, diameter, voxel_size_zyx, use_memmap, memmap_directory)
        if use_memmap:
            from haemolynx.preprocessing.memmap_support import combine_by_slab

            narrow = combine_by_slab(
                lambda m, w: m.astype(bool) & ~w, mask, wide, directory=memmap_directory
            )
        else:
            narrow = np.asarray(mask, dtype=bool) & ~wide
        logger.info(
            f"Large-vessel {name} mask split at {float(diameter_um):.3f} um wide: "
            f"{kept} of {total} voxels stay large, {total - kept} are the small "
            f"{name} mask."
        )
        return wide, narrow

    large_arteriole, small_arteriole = split(
        arteriole_mask, "arteriole" if venule_mask is not None else "global"
    )
    large_venule, small_venule = split(venule_mask, "venule")
    return large_arteriole, large_venule, small_arteriole, small_venule
