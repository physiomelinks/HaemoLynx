"""Tests for io module."""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from ImageLynx.io import (
    crop_tiff_volume_from_corners,
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    bridge_gaps,
    simplify_to_3d,
    get_tif_spacing
)

def test_bridge_gaps():
    arr = np.zeros((5, 5, 5), dtype=bool)
    arr[2, 2, :] = True
    arr[2, 2, 1] = False  # 1-voxel gap
    result = bridge_gaps(arr, max_gap=2)
    assert result[2, 2, 1]


def test_simplify_to_3d():
    img3 = np.random.rand(4, 4, 4)
    assert simplify_to_3d(img3).shape == (4, 4, 4)
    img4 = np.random.rand(4, 4, 4, 2)
    out = simplify_to_3d(img4)
    assert out.shape == (4, 4, 4)


def test_simplify_to_3d_raises():
    with pytest.raises(ValueError):
        simplify_to_3d(np.zeros((3, 3)))


def test_load_and_skeletonize_3d_tif(tmp_path):
    f = tmp_path / "test.tif"
    img = np.random.randint(0, 255, (6, 6, 6), dtype=np.uint16)
    import tifffile
    tifffile.imwrite(f, img)
    image, skeleton, vx, vy, vz = load_and_skeletonize_3d_tif(str(f))
    assert image.shape == (6, 6, 6)
    assert skeleton.shape == (6, 6, 6)
    assert skeleton.dtype == bool


def test_crop_tiff_volume_from_corners(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    arr = np.arange(10 * 8 * 6, dtype=np.uint16).reshape((10, 8, 6))
    import tifffile

    tifffile.imwrite(src, arr)
    info = crop_tiff_volume_from_corners(
        src,
        dst,
        corner_a=(9, 7, 5),
        corner_b=(7, 4, 0),
    )

    out = tifffile.imread(dst)
    assert out.shape == (3, 4, 6)
    assert tuple(info["source_shape"]) == (10, 8, 6)
    assert tuple(info["cropped_shape"]) == (3, 4, 6)


def test_get_tif_spacing(tmp_path):
    import tifffile
    f = tmp_path / "test_spacing.tif"
    img = np.zeros((2, 2, 2), dtype=np.uint8)
    
    # Test setting specific spacing (Z, Y, X) = (2.0, 0.5, 0.5)
    # tifffile uses resolution = (1/X, 1/Y) in cm/inch etc.
    # spacing tag is for Z.
    tifffile.imwrite(
        f,
        img,
        resolution=(2.0, 2.0), # 1/0.5
        metadata={'spacing': 2.0},
        imagej=True
    )
    
    spacing = get_tif_spacing(str(f))
    assert spacing == (2.0, 0.5, 0.5)
    
    # Test fallback to defaults
    f2 = tmp_path / "test_no_spacing.tif"
    tifffile.imwrite(f2, img)
    spacing_default = get_tif_spacing(str(f2))
    assert spacing_default == (1.0, 1.0, 1.0)
