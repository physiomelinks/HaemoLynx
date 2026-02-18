"""Tests for io module."""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from ImageLynx.io import (
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    bridge_gaps,
    simplify_to_3d,
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
    image, skeleton = load_and_skeletonize_3d_tif(str(f))
    assert image.shape == (6, 6, 6)
    assert skeleton.shape == (6, 6, 6)
    assert skeleton.dtype == bool
