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
    
def simplify_to_3d(image):
    """
    Simplify 4D or 5D datasets to 3D by removing the fourth dimension (channels).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image array (3D, 4D, or 5D)
    
    Returns:
    --------
    numpy.ndarray
        3D image array
    """
    if image.ndim == 3:
        return image
    elif image.ndim == 4:
        # Remove the 4th dimension (channels) - take first channel
        return image[:, :, :, 0]
    elif image.ndim == 5:
        # Remove the 4th and 5th dimensions - take first channel and first of 5th dim
        return image[:, :, :, 0, 0]
    elif image.ndim < 3:
        raise ValueError(f"Image has {image.ndim} dimensions. Need at least 3D data.")
    else:
        # For higher dimensions, take first 3 spatial dimensions
        logger.warning(f"Image has {image.ndim} dimensions. Taking first 3 spatial dimensions and first channel.")
        return image[:, :, :, 0, 0, 0] if image.ndim == 6 else image[:, :, :, 0]
    
    return image

def load_and_skeletonize_3d_tif(filepath, voxel_size=1.0):
    logger.debug("Loading and skeletonizing TIFF...")
    image = tifffile.imread(filepath)
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)
    skeleton = skeletonize_3d(img_as_bool(bridged))
    skeleton = remove_small_objects(skeleton_image, min_size=min_branch_length)
    skeleton = skeletonize_3D(cleaned > 0)
    return image, skeleton.astype(bool)

def load_and_skeletonize_3d_h5(filepath, dataset_name=None, voxel_size=1.0):
    logger.debug("Loading and skeletonizing H5...")
    
    with h5py.File(filepath, 'r') as f:
        # If no dataset name specified, use the first available dataset
        if dataset_name is None:
            # Get the first dataset key
            dataset_keys = list(f.keys())
            if not dataset_keys:
                raise ValueError("No datasets found in the H5 file")
            dataset_name = dataset_keys[0]
            logger.debug(f"No dataset specified, using first available: '{dataset_name}'")
        
        # Check if the specified dataset exists
        if dataset_name not in f:
            available_datasets = list(f.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
        
        # Load the image data
        image = f[dataset_name][:]
        
        # Convert to numpy array if needed and ensure proper data type
        image = np.array(image)
        
        logger.debug(f"Original image shape: {image.shape}")
    
    # Handle 4D and 5D datasets by squeezing singleton dimensions
    image = simplify_to_3d(image)
    logger.debug(f"Simplified image shape: {image.shape}")
    
    # Apply the same processing pipeline as the original function
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)  
    skeleton = skeletonize_3d(img_as_bool(bridged))
    skeleton = remove_small_objects(skeleton_image, min_size=min_branch_length)
    skeleton = skeletonize_3D(cleaned > 0)
    
    return image, skeleton.astype(bool)

