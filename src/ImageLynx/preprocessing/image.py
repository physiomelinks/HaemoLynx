"""Image-level operations: ROI cropping, smoothing, and pre-processing."""
import logging
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import opening, binary_opening, ball

logger = logging.getLogger(__name__)

def _is_dask_array(arr):
    try:
        import dask.array as da
        return isinstance(arr, da.Array)
    except ImportError:
        return False

def smooth_probability_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to a probability map to reduce noise.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x) of probabilities.
    sigma:
        Standard deviation for Gaussian kernel.
    """
    if sigma <= 0:
        return image
        
    logger.info("Smoothing probability map with sigma=%.2f", sigma)
    
    if _is_dask_array(image):
        import dask.array as da
        # Use 4*sigma depth for better Gaussian approximation at boundaries
        depth = int(np.ceil(4 * sigma))
        return image.map_overlap(
            smooth_probability_map,
            depth=depth,
            boundary="reflect",
            sigma=sigma,
            dtype=np.float32
        )

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
    if size <= 0:
        return image
        
    logger.info("Applying median filter with size=%d", size)
    
    if _is_dask_array(image):
        import dask.array as da
        depth = size // 2 + 1
        return image.map_overlap(
            median_filter_image,
            depth=depth,
            boundary="reflect",
            size=size,
            dtype=np.float32
        )

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
    if radius <= 0:
        return image
        
    logger.info("Applying morphological opening with radius=%d", radius)
    
    if _is_dask_array(image):
        import dask.array as da
        depth = radius + 1
        return image.map_overlap(
            morphological_opening,
            depth=depth,
            boundary="reflect",
            radius=radius,
            dtype=image.dtype
        )

    footprint = ball(radius)
    if image.dtype == bool:
        return binary_opening(image, footprint=footprint)
    else:
        return opening(image, footprint=footprint)

def morphological_closing(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply morphological closing to a 3D volume.

    Parameters
    ----------
    image:
        Input 3D array (z, y, x).
    radius:
        Radius of the ball-shaped structuring element.
    """
    if radius <= 0:
        return image
        
    logger.info("Applying morphological closing with radius=%d", radius)
    
    if _is_dask_array(image):
        import dask.array as da
        depth = radius + 1
        return image.map_overlap(
            morphological_closing,
            depth=depth,
            boundary="reflect",
            radius=radius,
            dtype=image.dtype
        )

    footprint = ball(radius)
    if image.dtype == bool:
        from skimage.morphology import binary_closing
        return binary_closing(image, footprint=footprint)
    else:
        from skimage.morphology import closing
        return closing(image, footprint=footprint)

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
    
    if _is_dask_array(image):
        # Hysteresis is a global operation (connected components).
        # map_overlap might not be perfect for VERY long vessels, 
        # but for typical chunks and vessel sizes it should work okay.
        # Alternatively, we could compute it here if it's the last step.
        logger.warning("Applying Hysteresis Thresholding to Dask array - results may differ slightly at chunk boundaries.")
        import dask.array as da
        return image.map_overlap(
            hysteresis_threshold,
            depth=5, # Overlap to help connectivity
            boundary=0,
            low=low,
            high=high,
            dtype=bool
        )

    return apply_hysteresis_threshold(image, low, high)

def joint_hysteresis_threshold(
    probability_map: np.ndarray,
    entropy_map: np.ndarray,
    low: float = 0.2,
    high: float = 0.4,
    shannon_core: float = 0.6,
    shannon_max: float = 0.95
) -> np.ndarray:
    """Apply dual-criteria morphological reconstruction based on probability and entropy.

    Pixels above 'high' and below 'shannon_core' are seeds. 
    Any pixel above 'low' and below 'shannon_max' that is connected to a seed is kept.

    Parameters
    ----------
    probability_map:
        Input 3D array (z, y, x) of probabilities.
    entropy_map:
        Input 3D array (z, y, x) of Shannon entropy.
    low:
        Lower threshold for connectivity.
    high:
        Upper threshold for seeds.
    shannon_core:
        Maximum entropy for a seed voxel.
    shannon_max:
        Maximum entropy for a candidate voxel.
    """
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    if shannon_core > shannon_max:
        raise ValueError(f"shannon_core ({shannon_core}) must be <= shannon_max ({shannon_max})")

    logger.info("Applying joint hysteresis (low=%.2f, high=%.2f, core=%.2f, max=%.2f)", low, high, shannon_core, shannon_max)

    if _is_dask_array(probability_map) or _is_dask_array(entropy_map):
        logger.warning("Applying Joint Hysteresis to Dask array - results may differ slightly at chunk boundaries.")
        import dask.array as da
        
        return da.map_overlap(
            joint_hysteresis_threshold,
            probability_map,
            entropy_map,
            depth=5,
            boundary=0,
            dtype=bool,
            low=low,
            high=high,
            shannon_core=shannon_core,
            shannon_max=shannon_max,
        )

    from skimage.morphology import reconstruction

    seed_mask = (probability_map >= high) & (entropy_map <= shannon_core)
    candidate_mask = (probability_map >= low) & (entropy_map <= shannon_max)
    
    # Reconstruct seeds into candidate mask
    return reconstruction(seed=seed_mask, mask=candidate_mask, method='dilation').astype(bool)

def calculate_entropy_map(probability_volume: np.ndarray) -> np.ndarray:
    """Calculate voxel-wise Shannon entropy from a multi-channel probability volume.

    Assumes shape is (Z, Y, X, C) or (Z, C, Y, X). Automatically detects the channel axis.

    Parameters
    ----------
    probability_volume:
        Input 4D array of probabilities.
    """
    if _is_dask_array(probability_volume):
        import dask.array as da
        
        # Detect channel axis
        dims = np.array(probability_volume.shape)
        c_axis = int(np.argmin(dims))
        n_classes = dims[c_axis]
        
        logger.info("Calculating Shannon entropy map (Dask) (channel axis=%d, classes=%d)", c_axis, n_classes)
        
        epsilon = 1e-10
        # Use da functions for lazy evaluation
        prob_safe = da.clip(probability_volume, epsilon, 1.0)
        entropy_map = -da.sum(prob_safe * da.log2(prob_safe), axis=c_axis)
        
        max_entropy = np.log2(n_classes)
        return entropy_map / max_entropy

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
