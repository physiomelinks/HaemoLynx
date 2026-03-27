"""Helpers for large-vessel mask preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def dilate_binary_mask_by_microns(
    mask: np.ndarray,
    dilation_microns: float,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Dilate a binary mask by a physical radius in microns.

    Uses an EDT threshold in physical units, which supports anisotropic voxels and
    is typically faster than repeated binary dilation for larger radii.
    """
    if dilation_microns <= 0:
        return mask.astype(bool, copy=False)

    binary_mask = mask.astype(bool, copy=False)
    distance_from_mask = distance_transform_edt(~binary_mask, sampling=voxel_size_xyz)
    return binary_mask | (distance_from_mask <= float(dilation_microns))


def dilate_large_vessel_masks_by_microns(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    dilation_microns: float,
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dilate optional arteriole/venule masks by a micron distance."""
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask

    if dilation_microns <= 0:
        return (
            large_arteriole_mask.astype(bool, copy=False),
            large_venule_mask.astype(bool, copy=False),
        )

    return (
        dilate_binary_mask_by_microns(
            large_arteriole_mask,
            dilation_microns=dilation_microns,
            voxel_size_xyz=voxel_size_xyz,
        ),
        dilate_binary_mask_by_microns(
            large_venule_mask,
            dilation_microns=dilation_microns,
            voxel_size_xyz=voxel_size_xyz,
        ),
    )
