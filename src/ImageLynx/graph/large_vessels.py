"""Helpers for large-vessel mask preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label


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


def remove_small_opposite_attached_large_vessel_components(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    *,
    voxel_size_xyz: tuple[float, float, float],
    max_component_volume_um3: float,
    max_attach_distance_microns: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, dict[str, float]]]:
    """Remove tiny opposite-type large-vessel components attached near mask surfaces.

    A component is removed when:
    - its physical volume is <= `max_component_volume_um3`, and
    - its minimum distance to the opposite mask is <= `max_attach_distance_microns`.
    """
    stats = {
        "arteriole": {
            "removed_component_count": 0.0,
            "removed_voxel_count": 0.0,
            "removed_volume_um3": 0.0,
            "candidate_component_count": 0.0,
            "total_component_count": 0.0,
            "too_large_component_count": 0.0,
            "near_opposite_component_count": 0.0,
            "near_opposite_too_large_component_count": 0.0,
            "min_component_distance_microns": float("inf"),
        },
        "venule": {
            "removed_component_count": 0.0,
            "removed_voxel_count": 0.0,
            "removed_volume_um3": 0.0,
            "candidate_component_count": 0.0,
            "total_component_count": 0.0,
            "too_large_component_count": 0.0,
            "near_opposite_component_count": 0.0,
            "near_opposite_too_large_component_count": 0.0,
            "min_component_distance_microns": float("inf"),
        },
    }
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask, stats
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    max_volume = float(max_component_volume_um3)
    max_dist = float(max_attach_distance_microns)
    if max_volume <= 0 or max_dist < 0:
        return (
            large_arteriole_mask.astype(bool, copy=False),
            large_venule_mask.astype(bool, copy=False),
            stats,
        )

    voxel_volume_um3 = (
        float(voxel_size_xyz[0]) * float(voxel_size_xyz[1]) * float(voxel_size_xyz[2])
    )
    sampling_zyx = (
        float(voxel_size_xyz[2]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[0]),
    )
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    def _clean_one_side(
        mask_raw: np.ndarray,
        opposite_mask_raw: np.ndarray,
        *,
        side_key: str,
    ) -> np.ndarray:
        mask = mask_raw.astype(bool, copy=False)
        opposite_mask = opposite_mask_raw.astype(bool, copy=False)
        labels, component_count = label(mask, structure=structure)
        stats[side_key]["total_component_count"] = float(component_count)
        if component_count <= 0:
            stats[side_key]["min_component_distance_microns"] = 0.0
            return mask
        component_sizes = np.bincount(labels.ravel())
        candidate_ids: list[int] = []
        too_large_ids: list[int] = []
        for component_id in range(1, int(component_count) + 1):
            voxel_count = int(component_sizes[int(component_id)])
            volume = float(voxel_count) * float(voxel_volume_um3)
            if volume <= max_volume:
                candidate_ids.append(int(component_id))
            else:
                too_large_ids.append(int(component_id))
        stats[side_key]["candidate_component_count"] = float(len(candidate_ids))
        stats[side_key]["too_large_component_count"] = float(len(too_large_ids))

        distance_to_opposite = distance_transform_edt(~opposite_mask, sampling=sampling_zyx)
        component_slices = find_objects(labels, max_label=int(component_count))
        near_opposite_all = 0
        near_opposite_too_large = 0
        min_component_distance = float("inf")
        for component_id in range(1, int(component_count) + 1):
            comp_slice = component_slices[int(component_id) - 1]
            if comp_slice is None:
                continue
            local_labels = labels[comp_slice]
            local_component = local_labels == int(component_id)
            if not np.any(local_component):
                continue
            local_distance = distance_to_opposite[comp_slice][local_component]
            if local_distance.size == 0:
                continue
            min_distance = float(np.min(local_distance))
            if min_distance < min_component_distance:
                min_component_distance = min_distance
            if min_distance <= max_dist:
                near_opposite_all += 1
                if component_id in too_large_ids:
                    near_opposite_too_large += 1
        if not np.isfinite(min_component_distance):
            min_component_distance = 0.0
        stats[side_key]["near_opposite_component_count"] = float(near_opposite_all)
        stats[side_key]["near_opposite_too_large_component_count"] = float(
            near_opposite_too_large
        )
        stats[side_key]["min_component_distance_microns"] = float(min_component_distance)

        if not candidate_ids:
            return mask

        cleaned = mask.copy()
        removed_components = 0
        removed_voxels = 0
        for component_id in candidate_ids:
            comp_slice = component_slices[int(component_id) - 1]
            if comp_slice is None:
                continue
            local_labels = labels[comp_slice]
            local_component = local_labels == int(component_id)
            if not np.any(local_component):
                continue
            local_distance = distance_to_opposite[comp_slice][local_component]
            if local_distance.size == 0:
                continue
            min_distance = float(np.min(local_distance))
            if min_distance <= max_dist:
                removed_components += 1
                comp_voxels = int(np.count_nonzero(local_component))
                removed_voxels += comp_voxels
                local_clean = cleaned[comp_slice]
                local_clean[local_component] = False
                cleaned[comp_slice] = local_clean

        stats[side_key]["removed_component_count"] = float(removed_components)
        stats[side_key]["removed_voxel_count"] = float(removed_voxels)
        stats[side_key]["removed_volume_um3"] = float(removed_voxels) * float(voxel_volume_um3)
        return cleaned

    cleaned_arteriole = _clean_one_side(
        large_arteriole_mask,
        large_venule_mask,
        side_key="arteriole",
    )
    cleaned_venule = _clean_one_side(
        large_venule_mask,
        large_arteriole_mask,
        side_key="venule",
    )
    return cleaned_arteriole, cleaned_venule, stats


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
