"""Helpers for optional automated large-vessel mask loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .load import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    resolve_image_path_with_optional_zip,
)


def _load_mask_image(mask_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a mask image and return (image, voxel_size_xyz)."""
    suffix = mask_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_tif_with_voxel_size(
            str(mask_path)
        )
        return image, (float(voxel_x), float(voxel_y), float(voxel_z))
    if suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_h5_with_voxel_size(
            str(mask_path)
        )
        return image, (float(voxel_x), float(voxel_y), float(voxel_z))
    raise ValueError(
        f"Unsupported mask format '{suffix}' for {mask_path}. "
        "Supported formats: .tif, .tiff, .h5"
    )


def load_large_vessel_masks(
    enabled: bool,
    large_arteriole_mask_path: str | Path | None = None,
    large_venule_mask_path: str | Path | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Load optional large arteriole/venule masks with strict pairing rules.

    When enabled is False, no mask paths are allowed and (None, None) is returned.
    When enabled is True, both mask paths are required and both are loaded.
    """
    has_arteriole = large_arteriole_mask_path is not None
    has_venule = large_venule_mask_path is not None

    if not enabled:
        if has_arteriole or has_venule:
            raise ValueError(
                "Large-vessel masks are disabled. Set enabled=True to provide "
                "large_arteriole_mask_path and large_venule_mask_path."
            )
        return None, None, None, None

    if has_arteriole != has_venule:
        raise ValueError(
            "Either provide both large_arteriole_mask_path and "
            "large_venule_mask_path, or provide neither."
        )
    if not has_arteriole:
        raise ValueError(
            "Large-vessel masks are enabled, but mask paths are missing. "
            "Provide both large_arteriole_mask_path and large_venule_mask_path."
        )

    arteriole_path = resolve_image_path_with_optional_zip(Path(large_arteriole_mask_path))
    venule_path = resolve_image_path_with_optional_zip(Path(large_venule_mask_path))
    large_arteriole_mask, large_arteriole_voxel_size = _load_mask_image(arteriole_path)
    large_venule_mask, large_venule_voxel_size = _load_mask_image(venule_path)
    return (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_voxel_size,
        large_venule_voxel_size,
    )
