import logging
import numpy as np

logger = logging.getLogger(__name__)

HAS_GPU = False
cupy = None
cp_ndimage = None
cp_morphology = None
cp_filters = None
cp_transform = None

def _check_gpu():
    global HAS_GPU, cupy, cp_ndimage, cp_morphology, cp_filters, cp_transform
    if HAS_GPU:
        return True
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as nd
        import cucim.skimage.morphology as morph
        import cucim.skimage.filters as filt
        import cucim.skimage.transform as trans
        cupy = cp
        cp_ndimage = nd
        cp_morphology = morph
        cp_filters = filt
        cp_transform = trans
        HAS_GPU = True
        return True
    except ImportError:
        return False

# Initial check
_check_gpu()

def get_ndimage():
    _check_gpu()
    return cp_ndimage if HAS_GPU else sp_ndimage

def get_morphology():
    _check_gpu()
    return cp_morphology if HAS_GPU else sp_morphology

def get_filters():
    _check_gpu()
    return cp_filters if HAS_GPU else sp_filters

def get_transform():
    _check_gpu()
    return cp_transform if HAS_GPU else sp_transform

def to_gpu(arr):
    if _check_gpu():
        if cupy.get_array_module(arr) is cupy:
            return arr
        return cupy.asarray(arr)
    return arr

def to_cpu(arr):
    if _check_gpu():
        if cupy.get_array_module(arr) is cupy:
            return cupy.asnumpy(arr)
    return arr
