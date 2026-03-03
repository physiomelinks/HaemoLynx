"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes, label, binary_dilation,  map_coordinates, distance_transform_edt
from skimage.morphology import remove_small_objects, skeletonize_3d

logger = logging.getLogger(__name__)

def bridge_gaps(binary_skeleton, max_gap=4):
    dist = distance_transform_edt(~binary_skeleton)
    fill_mask = (dist <= max_gap) & (~binary_skeleton)
    filled = binary_skeleton | fill_mask
    return filled

def load_and_skeletonize_3d_tif(filepath, voxel_size=1.0):
    logger.debug("Loading and skeletonizing TIFF...")
    image = tifffile.imread(filepath)
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)
    skeleton = skeletonize_3d(img_as_bool(bridged))
    cleaned = remove_small_objects(skeleton_image, min_size=min_branch_length)
    cleaned = skeletonize_3D(cleaned > 0)
    return image, skeleton, cleaned.astype(bool)

