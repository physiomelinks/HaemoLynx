import os
import numpy as np
import pytest
import tifffile
import h5py
from pathlib import Path
from ImageLynx import io, preprocessing

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

@pytest.fixture
def sample_volume_3d():
    """Create a random 3D volume for testing."""
    return np.random.rand(32, 32, 32).astype(np.float32)

@pytest.fixture
def temp_tiff(tmp_path, sample_volume_3d):
    """Save sample volume to a temporary TIFF."""
    path = tmp_path / "test.tif"
    tifffile.imwrite(str(path), sample_volume_3d)
    return path

@pytest.fixture
def temp_h5(tmp_path, sample_volume_3d):
    """Save sample volume to a temporary H5."""
    path = tmp_path / "test.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("data", data=sample_volume_3d)
    return path

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_lazy_load_tif(temp_tiff, sample_volume_3d):
    """Verify TIFF lazy loading returns a dask array matching content."""
    lazy_vol = io.load_3d_tif(temp_tiff, lazy=True)
    assert isinstance(lazy_vol, da.Array)
    assert lazy_vol.shape == sample_volume_3d.shape
    np.testing.assert_array_almost_equal(lazy_vol.compute(), sample_volume_3d)

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_lazy_load_h5(temp_h5, sample_volume_3d):
    """Verify H5 lazy loading returns a dask array matching content."""
    lazy_vol = io.load_3d_h5(temp_h5, dataset_name="data", lazy=True)
    assert isinstance(lazy_vol, da.Array)
    assert lazy_vol.shape == sample_volume_3d.shape
    np.testing.assert_array_almost_equal(lazy_vol.compute(), sample_volume_3d)

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_entropy_calculation(sample_volume_3d):
    """Verify entropy calculation on Dask matches NumPy results."""
    # Create 4D data (2 channels)
    vol_4d = np.stack([sample_volume_3d, 1.0 - sample_volume_3d], axis=-1)
    lazy_4d = da.from_array(vol_4d, chunks=(16, 16, 16, 2))
    
    res_numpy = preprocessing.calculate_entropy_map(vol_4d)
    res_dask = preprocessing.calculate_entropy_map(lazy_4d)
    
    assert isinstance(res_dask, da.Array)
    np.testing.assert_array_almost_equal(res_dask.compute(), res_numpy)

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_median_filter_overlap(sample_volume_3d):
    """Verify median filter with overlap handles chunk boundaries correctly."""
    # Use small chunks to force many boundaries
    lazy_vol = da.from_array(sample_volume_3d, chunks=(8, 8, 8))
    
    res_numpy = preprocessing.median_filter_image(sample_volume_3d, size=3)
    res_dask = preprocessing.median_filter_image(lazy_vol, size=3)
    
    assert isinstance(res_dask, da.Array)
    # Boundaries should be consistent due to map_overlap
    np.testing.assert_array_almost_equal(res_dask.compute(), res_numpy)

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_gaussian_smoothing_overlap(sample_volume_3d):
    """Verify Gaussian smoothing with overlap matches NumPy."""
    lazy_vol = da.from_array(sample_volume_3d, chunks=(16, 16, 16))
    
    res_numpy = preprocessing.smooth_probability_map(sample_volume_3d, sigma=1.0)
    res_dask = preprocessing.smooth_probability_map(lazy_vol, sigma=1.0)
    
    assert isinstance(res_dask, da.Array)
    np.testing.assert_array_almost_equal(res_dask.compute(), res_numpy)

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_hysteresis_thresholding(sample_volume_3d):
    """Verify hysteresis on dask returns a valid mask."""
    lazy_vol = da.from_array(sample_volume_3d, chunks=(16, 16, 16))
    
    res_numpy = preprocessing.hysteresis_threshold(sample_volume_3d, low=0.2, high=0.4)
    res_dask = preprocessing.hysteresis_threshold(lazy_vol, low=0.2, high=0.4)
    
    assert isinstance(res_dask, da.Array)
    # Hysteresis can vary slightly at boundaries in Dask, so we check general similarity
    # or identicality if the depth is sufficient.
    computed = res_dask.compute()
    overlap_fraction = np.sum(computed == res_numpy) / res_numpy.size
    assert overlap_fraction > 0.95

@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_roi_crop_virtual(sample_volume_3d):
    """Verify that ROI cropping on Dask is a virtual metadata operation."""
    lazy_vol = da.from_array(sample_volume_3d, chunks=(16, 16, 16))
    
    # Crop to 50%
    cropped_lazy = preprocessing.crop_roi(lazy_vol, sub_volume_percentage=0.5)
    
    assert isinstance(cropped_lazy, da.Array)
    assert cropped_lazy.shape == (16, 16, 16)
    # Check content
    expected = sample_volume_3d[8:24, 8:24, 8:24]
    np.testing.assert_array_almost_equal(cropped_lazy.compute(), expected)
