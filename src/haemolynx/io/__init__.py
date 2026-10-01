"""I/O utilities for loading and simplifying image data."""
from .load import (
    crop_tiff_volume_from_corners,
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    read_voxel_size_xyz,
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    load_binary_mask_and_voxel_size,
    load_volume_and_voxel_size,
    resolve_image_path_with_optional_zip,
    simplify_to_3d,
)
from .load_2d import (
    load_image_with_voxel_size_2d_aware,
    promote_2d_to_single_slice_volume,
)
from .raw_volume_cache import load_raw_volume_reusing_cache
from .ilastik import run_ilastik_headless_segmentation
from .automated_vessel_assignment import (
    load_and_validate_vessel_masks,
    vessel_mask_arguments,
    load_large_vessel_masks,
)
from .voxel_validation import resolve_voxel_size_xyz, validate_voxel_size_xyz
from .channels import TiffChannel, tiff_channel_axis, tiff_channels
from .axis_order import (
    CANONICAL_AXIS_ORDER,
    VALID_AXIS_ORDERS,
    apply_axis_order,
    axis_order_transpose,
    file_axis_spacing_from_xyz,
    normalize_axis_order,
    voxel_size_xyz_from_file_axes,
    voxel_size_xyz_from_zyx,
    voxel_size_zyx_from_xyz,
)
from ..preprocessing import bridge_gaps

__all__ = [
    "TiffChannel",
    "tiff_channel_axis",
    "tiff_channels",
    "load_3d_tif_with_voxel_size",
    "read_voxel_size_xyz",
    "load_3d_h5_with_voxel_size",
    "load_and_skeletonize_3d_tif",
    "load_and_skeletonize_3d_h5",
    "crop_tiff_volume_from_corners",
    "load_binary_mask_and_voxel_size",
    "load_volume_and_voxel_size",
    "resolve_image_path_with_optional_zip",
    "simplify_to_3d",
    "load_image_with_voxel_size_2d_aware",
    "promote_2d_to_single_slice_volume",
    "load_raw_volume_reusing_cache",
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
    "voxel_size_xyz_from_file_axes",
    "file_axis_spacing_from_xyz",
    "axis_order_transpose",
    "apply_axis_order",
    "voxel_size_zyx_from_xyz",
    "voxel_size_xyz_from_zyx",
]
