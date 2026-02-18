"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


def _skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Skeletonize 3D binary image. Uses skeletonize with Lee method."""
    from skimage.morphology import skeletonize
    return skeletonize(img, method="lee")


def bridge_gaps(binary_skeleton: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill small gaps in binary skeleton using distance transform."""
    dist = distance_transform_edt(~binary_skeleton)
    fill_mask = (dist <= max_gap) & (~binary_skeleton)
    return binary_skeleton | fill_mask


def skeletonize_3d_safe(img: np.ndarray) -> np.ndarray:
    """Safe 3D skeletonization wrapper."""
    return _skeletonize_3d(img.astype(bool))


def preprocess_skeleton_for_graph(
    skeleton_image: np.ndarray, min_branch_length: int = 5
) -> np.ndarray:
    """Remove small objects and re-skeletonize for graph building."""
    cleaned = remove_small_objects(skeleton_image, min_size=min_branch_length)
    cleaned = _skeletonize_3d(cleaned.astype(bool))
    return cleaned.astype(bool)
