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
    """Create a random 3D volume for testing, the same field every run."""
    return np.random.default_rng(0).random((32, 32, 32)).astype(np.float32)

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
def test_dask_hysteresis_thresholding():
    """Hysteresis on dask equals the NumPy result where every tube's seed is in overlap reach.

    On a random field a plain cut at ``low`` agrees with hysteresis almost everywhere, so the
    field here is tubes in the band between low and high, some carrying a seed and some not.
    Two seeded tubes cross a chunk face with the seed on the far side, no more than the
    overlap depth (5) away, so a chunk only keeps its half by seeing its neighbour's seed.
    """
    low, high, band, seed = 0.2, 0.4, 0.3, 0.6
    field = np.random.default_rng(0).random((32, 32, 32)).astype(np.float32) * 0.15
    field[4, 4, 2:14] = band
    field[4, 4, 8] = seed                  # seeded, inside one chunk
    field[4, 10, 2:14] = band              # unseeded, inside one chunk
    field[20, 20, 12:21] = band
    field[20, 20, 15] = seed               # seeded, crosses the x = 16 face
    field[26, 26, 12:21] = band            # unseeded, crosses the x = 16 face
    field[12:21, 8, 24] = band
    field[17, 8, 24] = seed                # seeded, crosses the z = 16 face

    res_numpy = preprocessing.hysteresis_threshold(field, low=low, high=high)
    res_dask = preprocessing.hysteresis_threshold(
        da.from_array(field, chunks=(16, 16, 16)), low=low, high=high)

    assert isinstance(res_dask, da.Array)
    assert res_numpy[4, 4, 2:14].all() and res_numpy[20, 20, 12:21].all()
    assert res_numpy[12:21, 8, 24].all()
    assert not res_numpy[4, 10].any() and not res_numpy[26, 26].any(), (
        "the unseeded tubes must drop, or a plain cut at low would pass")
    np.testing.assert_array_equal(res_dask.compute(), res_numpy)

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
