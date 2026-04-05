"""Image-level operations: ROI cropping, smoothing, and pre-processing."""
import logging
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import opening, binary_opening, ball

logger = logging.getLogger(__name__)

def smooth_probability_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to a probability map to reduce noise.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x) of probabilities.
    sigma:
        Standard deviation for Gaussian kernel.
    """
    logger.info("Smoothing probability map with sigma=%.2f", sigma)
    return gaussian_filter(image.astype(np.float32), sigma=sigma)

def median_filter_image(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply median filter to a 3D volume to reduce noise.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x).
    size:
        Size of the median filter window (e.g., 3 for 3x3x3).
    """
    logger.info("Applying median filter with size=%d", size)
    return median_filter(image.astype(np.float32), size=size)

def morphological_opening(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply morphological opening to a 3D volume.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x).
    radius:
        Radius of the ball-shaped structuring element.
    """
    logger.info("Applying morphological opening with radius=%d", radius)
    footprint = ball(radius)
    if image.dtype == bool:
        return binary_opening(image, footprint=footprint)
    else:
        return opening(image, footprint=footprint)

def hysteresis_threshold(
    image: np.ndarray, low: float = 0.3, high: float = 0.7
) -> np.ndarray:
    """Apply hysteresis thresholding to a probability map.

    Pixels above 'high' are seeds. Any pixel above 'low' that is connected 
    to a seed is kept.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x) of probabilities.
    low:
        Lower threshold for connectivity.
    high:
        Upper threshold for seeds.
    """
    logger.info("Applying hysteresis thresholding (low=%.2f, high=%.2f)", low, high)
    return apply_hysteresis_threshold(image, low, high)

def calculate_entropy_map(probability_volume: np.ndarray) -> np.ndarray:
    """Calculate voxel-wise Shannon entropy from a multi-channel probability volume.

    Assumes shape is (Z, Y, X, C) or (Z, C, Y, X). Automatically detects the channel axis.

    Parameters
    ----------
    probability_volume:
        Input 4D array of probabilities.
    """
    # Detect channel axis (smallest dimension)
    dims = np.array(probability_volume.shape)
    c_axis = int(np.argmin(dims))
    n_classes = dims[c_axis]
    
    logger.info("Calculating Shannon entropy map (channel axis=%d, classes=%d)", c_axis, n_classes)
    
    # Add a tiny epsilon to prevent log2(0) errors
    epsilon = 1e-10
    prob_safe = np.clip(probability_volume, epsilon, 1.0)
    
    # Calculate entropy: -sum(p * log2(p))
    entropy_map = -np.sum(prob_safe * np.log2(prob_safe), axis=c_axis)
    
    # Normalize entropy between 0 and 1
    max_entropy = np.log2(n_classes)
    normalized_entropy = entropy_map / max_entropy
    
    return normalized_entropy

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

    orig_shape = np.array(image.shape)
    
    # Identify the channel dimension (smallest dim, typically size 2 for Ilastik)
    # Spatial dims are usually much larger.
    if image.ndim == 4:
        c_axis = int(np.argmin(orig_shape))
        spatial_axes = [i for i in range(4) if i != c_axis]
        logger.info("4D Crop: Identified channel axis %d, spatial axes %s", c_axis, spatial_axes)
    else:
        spatial_axes = list(range(min(3, image.ndim)))
        c_axis = None

    # Calculate target dimensions for spatial dims
    target_dims = {}
    for ax in spatial_axes:
        target_dims[ax] = max(1, int(orig_shape[ax] * sub_volume_percentage))
    
    # Calculate centers and offsets for spatial dims
    offsets = [offset_z, offset_y, offset_x]
    
    slices = [slice(None)] * image.ndim
    for i, ax in enumerate(spatial_axes):
        center = orig_shape[ax] / 2.0
        offset_voxels = int(orig_shape[ax] * offsets[i])
        sub_center = center + offset_voxels
        
        start = max(0, int(sub_center - target_dims[ax] / 2.0))
        end = min(orig_shape[ax], start + target_dims[ax])
        
        # Alignment check
        if end - start < target_dims[ax]:
            if end == orig_shape[ax]:
                start = max(0, orig_shape[ax] - target_dims[ax])
            end = min(orig_shape[ax], start + target_dims[ax])
            
        slices[ax] = slice(start, end)
    
    cropped = image[tuple(slices)]
    logger.info("Cropped ROI: %s -> %s", tuple(orig_shape), cropped.shape)
    return cropped
