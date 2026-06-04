"""Image-level operations: ROI cropping, smoothing, and pre-processing."""
import logging
import numpy as np

logger = logging.getLogger(__name__)

def crop_roi(
    image: np.ndarray,
    sub_volume_percentage: float = 1.0,
    offset_z: float = 0.0,
    offset_y: float = 0.0,
    offset_x: float = 0.0,
) -> np.ndarray:
    """Crop a 3D ROI from a larger volume using percentage and offsets.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x).
    sub_volume_percentage:
        Factor to determine the size of the ROI relative to the original image 
        (0.0 to 1.0).
    offset_z, offset_y, offset_x:
        Offsets from the image center as a fraction of the total dimension 
        (-0.5 to 0.5).
    """
    if not (0 < sub_volume_percentage <= 1.0):
        logger.warning("sub_volume_percentage must be between 0 and 1. Using full volume.")
        return image

    orig_shape = image.shape
    # Calculate target dimensions
    target_dims = [max(1, int(dim * sub_volume_percentage)) for dim in orig_shape]
    
    # Calculate centers and offsets in voxel units
    centers = [dim / 2.0 for dim in orig_shape]
    voxel_offsets = [
        int(orig_shape[0] * offset_z),
        int(orig_shape[1] * offset_y),
        int(orig_shape[2] * offset_x)
    ]
    
    sub_centers = [center + offset for center, offset in zip(centers, voxel_offsets)]
    
    # Calculate slice bounds
    starts = [max(0, int(center - target / 2.0)) for center, target in zip(sub_centers, target_dims)]
    ends = [min(orig, start + target) for orig, start, target in zip(orig_shape, starts, target_dims)]
    
    # Final alignment check to preserve target size if possible
    for i in range(3):
        if ends[i] - starts[i] < target_dims[i]:
            # Try to expand 'start' if 'end' hit the boundary
            if ends[i] == orig_shape[i]:
                starts[i] = max(0, orig_shape[i] - target_dims[i])
            # Re-check 'end' if 'start' hit the boundary
            ends[i] = min(orig_shape[i], starts[i] + target_dims[i])
    
    logger.info("Cropped ROI: %s -> %s", orig_shape, image[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]].shape)
    return image[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
