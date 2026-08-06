"""I/O utilities for loading and simplifying image data."""
from .load import (
    crop_tiff_volume_from_corners,
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    load_binary_mask_and_voxel_size,
    load_volume_and_voxel_size,
    resolve_image_path_with_optional_zip,
    simplify_to_3d,
)
from .ilastik import run_ilastik_headless_segmentation
from .automated_vessel_assignment import (
    load_and_validate_vessel_masks,
    vessel_mask_arguments,
    load_large_vessel_masks,
)
from .voxel_validation import resolve_voxel_size_xyz, validate_voxel_size_xyz
from .axis_order import (
    CANONICAL_AXIS_ORDER,
    VALID_AXIS_ORDERS,
    apply_axis_order,
    axis_order_transpose,
    normalize_axis_order,
    voxel_size_xyz_from_zyx,
    voxel_size_zyx_from_xyz,
)
from ..preprocessing import bridge_gaps

__all__ = [
    "load_3d_tif_with_voxel_size",
    "load_3d_h5_with_voxel_size",
    "load_and_skeletonize_3d_tif",
    "load_and_skeletonize_3d_h5",
    "crop_tiff_volume_from_corners",
    "load_binary_mask_and_voxel_size",
    "load_volume_and_voxel_size",
    "resolve_image_path_with_optional_zip",
    "simplify_to_3d",
    "bridge_gaps",
    "run_ilastik_headless_segmentation",
    "load_large_vessel_masks",
    "load_and_validate_vessel_masks",
    "vessel_mask_arguments",
    "validate_voxel_size_xyz",
    "resolve_voxel_size_xyz",
    "CANONICAL_AXIS_ORDER",
    "VALID_AXIS_ORDERS",
    "normalize_axis_order",
    "axis_order_transpose",
    "apply_axis_order",
    "voxel_size_zyx_from_xyz",
    "voxel_size_xyz_from_zyx",
]
