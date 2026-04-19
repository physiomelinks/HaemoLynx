import numpy as np
import pytest
from ImageLynx.preprocessing import _backend

def test_backend_conversions_preserve_data():
    """Ensure that the CPU-GPU roundtrip is lossless for all standard dtypes."""
    # Test with different data types common in the pipeline
    # Note: np.bool_ or bool is used for masks
    for dtype in [np.float32, bool, np.uint8]:
        # Small random 3D volume
        original = (np.random.rand(5, 5, 5) > 0.5).astype(dtype)
        
        # Simulating the pipeline flow: CPU -> GPU -> CPU
        gpu_version = _backend.to_gpu(original)
        cpu_version = _backend.to_cpu(gpu_version)
        
        # Assertions: Metadata and content should be identical
        assert cpu_version.shape == original.shape
        # Note: cupy might return np.bool_ when given bool, or vice-versa. 
        # We check equivalent types.
        if dtype == bool:
            assert cpu_version.dtype in [bool, np.bool_]
        else:
            assert cpu_version.dtype == original.dtype
            
        np.testing.assert_array_equal(cpu_version, original)

def test_backend_module_availability():
    """Verify that the backend always provides functional modules."""
    assert _backend.get_ndimage() is not None
    assert _backend.get_filters() is not None
    assert _backend.get_morphology() is not None
    assert _backend.get_transform() is not None
    
    # Verify the modules actually have the critical functions we rely on
    assert hasattr(_backend.get_ndimage(), 'gaussian_filter')
    assert hasattr(_backend.get_filters(), 'apply_hysteresis_threshold')
    assert hasattr(_backend.get_morphology(), 'skeletonize')
    assert hasattr(_backend.get_transform(), 'resize')

def test_has_gpu_flag_consistency():
    """Verify HAS_GPU matches the actual import availability."""
    try:
        import cupy
        import cupyx.scipy.ndimage
        import cucim.skimage.morphology
        expected = True
    except ImportError:
        expected = False
    
    assert _backend.HAS_GPU == expected
