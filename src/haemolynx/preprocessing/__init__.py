"""Preprocessing: skeleton cleaning, bridging, skeletonization."""
from .skeleton import (
    bridge_gaps,
    close_binary_mask,
    connect_skeleton_components,
    preprocess_skeleton_for_graph,
    skeletonize_voxel_bundles_into_paths,
    log_skeleton_connectivity_stats,
    skeletonize_volume,
    skeletonize_3d,
)

__all__ = [
    "bridge_gaps",
    "close_binary_mask",
    "connect_skeleton_components",
    "preprocess_skeleton_for_graph",
    "skeletonize_voxel_bundles_into_paths",
    "log_skeleton_connectivity_stats",
    "skeletonize_volume",
    "skeletonize_3d",  # deprecated alias
]
