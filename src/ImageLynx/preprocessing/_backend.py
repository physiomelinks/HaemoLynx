import logging

logger = logging.getLogger(__name__)

# Attempt to import GPU-accelerated libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cucim.skimage.morphology as cp_morphology
    import cucim.skimage.filters as cp_filters
    import cucim.skimage.transform as cp_transform
    HAS_GPU = True
    logger.info("GPU acceleration available (cupy/cucim).")
except ImportError:
    import numpy as cp
    HAS_GPU = False
    logger.info("GPU acceleration not available. Falling back to CPU (scipy/skimage).")

import scipy.ndimage as sp_ndimage
import skimage.morphology as sp_morphology
import skimage.filters as sp_filters
import skimage.transform as sp_transform

def get_ndimage():
    return cp_ndimage if HAS_GPU else sp_ndimage

def get_morphology():
    return cp_morphology if HAS_GPU else sp_morphology

def get_filters():
    return cp_filters if HAS_GPU else sp_filters

def get_transform():
    return cp_transform if HAS_GPU else sp_transform

def to_gpu(arr):
    if HAS_GPU and not isinstance(arr, cp.get_array_module(arr).ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    if HAS_GPU:
        return cp.asnumpy(arr)
    return arr
