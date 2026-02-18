"""Load 3D images from TIFF or HDF5 and produce skeleton."""
import logging

import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.util import img_as_bool

from ..preprocessing.skeleton import bridge_gaps, skeletonize_3d_safe

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


def load_and_skeletonize_3d_tif(filepath: str, voxel_size: float = 1.0):
    """Load a TIFF stack, threshold, fill holes, bridge gaps, skeletonize."""
    logger.debug("Loading and skeletonizing TIFF...")
    image = tifffile.imread(filepath)
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)
    skeleton = skeletonize_3d_safe(img_as_bool(bridged))
    return image, skeleton


def load_and_skeletonize_3d_h5(
    filepath: str, dataset_name: str, voxel_size: float = 1.0
):
    """Load an HDF5 dataset, simplify to 3D, then skeletonize."""
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    logger.debug("Loading and skeletonizing H5...")
    with h5py.File(filepath, "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {available}"
            )
        image = np.array(f[dataset_name][:])
    logger.debug("Original image shape: %s", image.shape)
    image = simplify_to_3d(image)
    logger.debug("Simplified image shape: %s", image.shape)
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)
    skeleton = skeletonize_3d_safe(img_as_bool(bridged))
    return image, skeleton


def simplify_to_3d(image: np.ndarray) -> np.ndarray:
    """Reduce image to 3D by taking first spatial/channel slice."""
    if image.ndim == 3:
        return image
    if image.ndim < 3:
        raise ValueError(f"Image has {image.ndim} dimensions. Need at least 3D.")
    logger.warning(
        "Image has %d dimensions. Taking first 3 spatial + first channel.",
        image.ndim,
    )
    if image.ndim == 6:
        return image[:, :, :, 0, 0, 0]
    return image[:, :, :, 0]
