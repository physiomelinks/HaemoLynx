"""Helpers for large-vessel mask preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label


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
    # distance_transform_edt sampling follows array axis order (z, y, x).
    sampling_zyx = (
        float(voxel_size_xyz[2]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[0]),
    )
    distance_from_mask = distance_transform_edt(~binary_mask, sampling=sampling_zyx)
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


def exclude_smaller_overlapping_large_vessel_components(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Remove smaller overlapping connected components between large-vessel masks.

    For each overlapping arteriole/venule component pair, the smaller full
    component is removed. This is intended to suppress segmentation leakage where
    a small fragment of one class sits inside a much larger component of the
    opposite class.
    """
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask

    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    overlap = arteriole_mask & venule_mask
    if not np.any(overlap):
        return arteriole_mask, venule_mask

    # 26-connectivity for 3D component labelling.
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    arteriole_labels, _ = label(arteriole_mask, structure=structure)
    venule_labels, _ = label(venule_mask, structure=structure)

    arteriole_sizes = np.bincount(arteriole_labels.ravel())
    venule_sizes = np.bincount(venule_labels.ravel())

    overlap_arteriole_labels = arteriole_labels[overlap]
    overlap_venule_labels = venule_labels[overlap]
    overlapping_pairs = np.unique(
        np.column_stack([overlap_arteriole_labels, overlap_venule_labels]),
        axis=0,
    )

    remove_arteriole_labels: set[int] = set()
    remove_venule_labels: set[int] = set()
    for arteriole_label, venule_label in overlapping_pairs:
        if arteriole_label == 0 or venule_label == 0:
            continue
        arteriole_size = int(arteriole_sizes[int(arteriole_label)])
        venule_size = int(venule_sizes[int(venule_label)])
        if arteriole_size < venule_size:
            remove_arteriole_labels.add(int(arteriole_label))
        elif venule_size < arteriole_size:
            remove_venule_labels.add(int(venule_label))
        else:
            # Deterministic tie-break for reproducible output.
            remove_venule_labels.add(int(venule_label))

    cleaned_arteriole_mask = arteriole_mask.copy()
    cleaned_venule_mask = venule_mask.copy()
    for component_label in remove_arteriole_labels:
        cleaned_arteriole_mask[arteriole_labels == component_label] = False
    for component_label in remove_venule_labels:
        cleaned_venule_mask[venule_labels == component_label] = False

    return cleaned_arteriole_mask, cleaned_venule_mask
