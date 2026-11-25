"""Mask-component volume filtering helpers."""
from __future__ import annotations

from typing import Any

import numpy as np


def remove_small_mask_components_by_volume(
    mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    min_component_volume_um3: float,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove connected components below a physical-volume threshold.

    Args:
        mask: 3D binary-like input mask (canonical ``(z, y, x)`` array order).
        voxel_size_xyz: Physical voxel size in microns ``(x, y, z)``.
        min_component_volume_um3: Minimum kept component volume (microns^3).
            Components with strictly smaller volume are removed. Set ``<= 0``
            to disable filtering.
        low_memory: The low-RAM option: the mask returned is on disk (in
            *memmap_directory*). Either way the components are labelled a
            slab of planes at a time and held at their voxels
            (``preprocessing.slab_labels.ForegroundLabels``), not in a label
            volume (8 bytes a voxel once a stack has 2**31 voxels).
    """
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    def _volume() -> np.ndarray:
        if low_memory:
            from haemolynx.preprocessing.memmap_support import new_memmap_array

            return new_memmap_array(mask_bool.shape, bool, directory=memmap_directory)
        return np.empty(mask_bool.shape, dtype=bool)

    def _copy() -> np.ndarray:
        if not low_memory:
            return mask_bool.copy()
        from haemolynx.preprocessing.memmap_support import slab_step

        out = _volume()
        step = slab_step(mask_bool.shape)
        for start in range(0, mask_bool.shape[0], step):
            out[start:start + step] = mask_bool[start:start + step]
        return out

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask_bool.shape}.")

    vx, vy, vz = (
        float(voxel_size_xyz[0]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[2]),
    )
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError(
            "voxel_size_xyz must contain positive values. "
            f"Got {voxel_size_xyz}."
        )

    threshold_um3 = float(min_component_volume_um3)
    if threshold_um3 <= 0:
        return _copy(), {
            "removed_component_count": 0,
            "removed_voxel_count": 0,
            "removed_volume_um3": 0.0,
            "kept_component_count": int(1 if np.any(mask_bool) else 0),
            "threshold_um3": threshold_um3,
        }

    labels = ForegroundLabels(mask_bool, np.ones((3, 3, 3), dtype=np.uint8))
    n_labels = labels.count
    if int(n_labels) <= 0:
        return _copy(), {
            "removed_component_count": 0,
            "removed_voxel_count": 0,
            "removed_volume_um3": 0.0,
            "kept_component_count": 0,
            "threshold_um3": threshold_um3,
        }

    counts = labels.sizes
    voxel_volume_um3 = vx * vy * vz
    component_volumes_um3 = counts.astype(float) * float(voxel_volume_um3)

    keep_lookup = np.zeros(int(n_labels) + 1, dtype=bool)
    keep_lookup[1:] = component_volumes_um3[1:] >= threshold_um3
    filtered_mask = labels.write(_volume(), keep_lookup)

    removed_component_labels = np.where(~keep_lookup[1:])[0] + 1
    removed_voxel_count = int(
        sum(int(counts[int(lbl)]) for lbl in removed_component_labels)
    )
    removed_component_count = int(removed_component_labels.size)
    kept_component_count = int(np.count_nonzero(keep_lookup[1:]))

    return filtered_mask, {
        "removed_component_count": removed_component_count,
        "removed_voxel_count": removed_voxel_count,
        "removed_volume_um3": float(removed_voxel_count * voxel_volume_um3),
        "kept_component_count": kept_component_count,
        "threshold_um3": threshold_um3,
    }


def remove_small_vessel_components_by_volume(
    arteriole_mask: np.ndarray | None,
    venule_mask: np.ndarray | None,
    *,
    voxel_size_xyz: tuple[float, float, float],
    min_component_volume_um3: float,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Apply volume-threshold filtering to an arteriole/venule mask pair."""
    if arteriole_mask is None or venule_mask is None:
        return arteriole_mask, venule_mask, {
            "arteriole": None,
            "venule": None,
            "threshold_um3": float(min_component_volume_um3),
        }
    if arteriole_mask.shape != venule_mask.shape:
        raise ValueError(
            "arteriole_mask and venule_mask must share a shape. "
            f"Got {arteriole_mask.shape} and {venule_mask.shape}."
        )

    cleaned_arteriole, arteriole_stats = remove_small_mask_components_by_volume(
        arteriole_mask,
        voxel_size_xyz=voxel_size_xyz,
        min_component_volume_um3=min_component_volume_um3,
        low_memory=low_memory,
        memmap_directory=memmap_directory,
    )
    cleaned_venule, venule_stats = remove_small_mask_components_by_volume(
        venule_mask,
        voxel_size_xyz=voxel_size_xyz,
        min_component_volume_um3=min_component_volume_um3,
        low_memory=low_memory,
        memmap_directory=memmap_directory,
    )
    return cleaned_arteriole, cleaned_venule, {
        "arteriole": arteriole_stats,
        "venule": venule_stats,
        "threshold_um3": float(min_component_volume_um3),
    }
