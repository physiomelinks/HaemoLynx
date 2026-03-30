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
    """Remove overlap voxels from the smaller component in each overlap pair.

    For each overlapping arteriole/venule component pair, the smaller full
    component is identified, and only the voxels in the *overlap region* are
    removed from that smaller component. This suppresses class-leakage overlap
    while preserving non-overlapping parts of the component.
    """
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask

    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    return _exclude_smaller_overlapping_mask_components(
        large_arteriole_mask,
        large_venule_mask,
    )


def exclude_smaller_overlapping_small_vessel_components(
    small_arteriole_mask: np.ndarray | None,
    small_venule_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Small-vessel equivalent of overlap cleanup used for large-vessel masks."""
    if small_arteriole_mask is None or small_venule_mask is None:
        return small_arteriole_mask, small_venule_mask

    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    return _exclude_smaller_overlapping_mask_components(
        small_arteriole_mask,
        small_venule_mask,
    )


def _exclude_smaller_overlapping_mask_components(
    arteriole_mask_raw: np.ndarray,
    venule_mask_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove overlap voxels from the smaller component in each overlap pair."""
    arteriole_mask = arteriole_mask_raw.astype(bool, copy=False)
    venule_mask = venule_mask_raw.astype(bool, copy=False)
    overlap = arteriole_mask & venule_mask
    if not np.any(overlap):
        return arteriole_mask, venule_mask
    initial_overlap_voxels = int(np.count_nonzero(overlap))

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

    overlap_pairs_to_remove_from_arteriole: list[tuple[int, int]] = []
    overlap_pairs_to_remove_from_venule: list[tuple[int, int]] = []
    for arteriole_label, venule_label in overlapping_pairs:
        if arteriole_label == 0 or venule_label == 0:
            continue
        arteriole_size = int(arteriole_sizes[int(arteriole_label)])
        venule_size = int(venule_sizes[int(venule_label)])
        if arteriole_size < venule_size:
            overlap_pairs_to_remove_from_arteriole.append(
                (int(arteriole_label), int(venule_label))
            )
        elif venule_size < arteriole_size:
            overlap_pairs_to_remove_from_venule.append(
                (int(arteriole_label), int(venule_label))
            )
        else:
            overlap_pairs_to_remove_from_venule.append(
                (int(arteriole_label), int(venule_label))
            )

    cleaned_arteriole_mask = arteriole_mask.copy()
    cleaned_venule_mask = venule_mask.copy()
    for arteriole_label, venule_label in overlap_pairs_to_remove_from_arteriole:
        pair_overlap = (
            overlap
            & (arteriole_labels == int(arteriole_label))
            & (venule_labels == int(venule_label))
        )
        cleaned_arteriole_mask[pair_overlap] = False
    for arteriole_label, venule_label in overlap_pairs_to_remove_from_venule:
        pair_overlap = (
            overlap
            & (arteriole_labels == int(arteriole_label))
            & (venule_labels == int(venule_label))
        )
        cleaned_venule_mask[pair_overlap] = False

    remaining_overlap_voxels = int(
        np.count_nonzero(cleaned_arteriole_mask & cleaned_venule_mask)
    )
    removed_overlap_voxels = int(initial_overlap_voxels - remaining_overlap_voxels)
    print(
        "Mask overlap cleanup summary: "
        f"initial_overlap_voxels={initial_overlap_voxels}, "
        f"removed_overlap_voxels={removed_overlap_voxels}, "
        f"remaining_overlap_voxels={remaining_overlap_voxels}."
    )

    return cleaned_arteriole_mask, cleaned_venule_mask
