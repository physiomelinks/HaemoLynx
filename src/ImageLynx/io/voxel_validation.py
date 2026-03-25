"""Helpers for validating and resolving voxel-size metadata."""
from __future__ import annotations

import numpy as np


def validate_voxel_size_xyz(
    voxel_size_xyz,
    *,
    label: str,
) -> tuple[float, float, float]:
    """Validate and normalize voxel-size triplets to (x, y, z) floats."""
    arr = np.asarray(voxel_size_xyz, dtype=float).ravel()
    if arr.size != 3 or np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(
            f"{label} must be 3 finite positive values, got {voxel_size_xyz}."
        )
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def resolve_voxel_size_xyz(
    metadata_voxel_size_xyz: tuple[float, float, float],
    metadata_status: dict[str, object] | None,
    voxel_size_override_xyz,
    voxel_size_policy: str,
) -> tuple[tuple[float, float, float], str]:
    """Resolve final voxel size from metadata and optional manual override."""
    policy = str(voxel_size_policy).strip().lower()
    if policy not in {"auto", "override", "metadata_only"}:
        raise ValueError(
            "voxel_size_policy must be one of: 'auto', 'override', 'metadata_only'. "
            f"Got: {voxel_size_policy}"
        )
    metadata_voxel_size_xyz = validate_voxel_size_xyz(
        metadata_voxel_size_xyz,
        label="metadata voxel size",
    )
    override_xyz = None
    if voxel_size_override_xyz is not None:
        override_xyz = validate_voxel_size_xyz(
            voxel_size_override_xyz,
            label="voxel_size_override_xyz",
        )

    metadata_state = str((metadata_status or {}).get("status", "missing")).lower()
    metadata_is_reliable = metadata_state == "complete"

    if policy == "override":
        if override_xyz is None:
            raise ValueError(
                "voxel_size_policy='override' requires voxel_size_override_xyz."
            )
        return override_xyz, "manual_override"
    if policy == "metadata_only":
        if not metadata_is_reliable:
            raise ValueError(
                "voxel_size_policy='metadata_only' requires complete metadata. "
                f"Current metadata status: {metadata_state}"
            )
        return metadata_voxel_size_xyz, "metadata"

    if metadata_is_reliable:
        return metadata_voxel_size_xyz, "metadata"
    if override_xyz is not None:
        return override_xyz, "manual_override"
    return metadata_voxel_size_xyz, "metadata_fallback"

