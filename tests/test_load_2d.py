"""A 2D image reaches the pipeline as a single-slice (z=1) volume with a
logged warning, instead of silently passing through 2D (the TIFF path) or
raising (the H5 path) -- see haemolynx.io.load_2d's module docstring.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.io import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    load_image_with_voxel_size_2d_aware,
    promote_2d_to_single_slice_volume,
)

# --- promote_2d_to_single_slice_volume --------------------------------------


def test_promote_2d_to_single_slice_volume_adds_a_leading_z_axis():
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    promoted, was_2d = promote_2d_to_single_slice_volume(image)

    assert was_2d is True
    assert promoted.shape == (1, 6, 8)
    assert np.array_equal(promoted[0], image)


def test_promote_2d_to_single_slice_volume_leaves_3d_unchanged():
    image = np.zeros((4, 6, 8), dtype=np.uint8)
    promoted, was_2d = promote_2d_to_single_slice_volume(image)

    assert was_2d is False
    assert promoted is image


# --- load_image_with_voxel_size_2d_aware: TIFF ------------------------------


def test_2d_aware_loader_promotes_a_2d_tiff_and_warns(tmp_path, caplog):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.random.randint(0, 255, (12, 9), dtype=np.uint16))

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        image, _x, _y, _z, status = load_image_with_voxel_size_2d_aware(
            str(path), input_format="tif"
        )

    assert image.shape == (1, 12, 9)
    assert status["promoted_from_2d"] is True
    assert any("2D image" in r.message for r in caplog.records)


def test_2d_aware_loader_promotes_a_2d_tiff_with_non_default_axis_order(tmp_path, caplog):
    """Regression: a 2D TIFF loaded with a non-default axis_order used to
    raise inside apply_axis_order (a 2-axis array has no (z, y, x) to
    reorder) instead of being promoted like the default-axis_order case."""
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.random.randint(0, 255, (12, 9), dtype=np.uint16))

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        image, _x, _y, _z, status = load_image_with_voxel_size_2d_aware(
            str(path), input_format="tif", axis_order="xyz"
        )

    assert image.shape == (1, 12, 9)
    assert status["promoted_from_2d"] is True


def test_2d_aware_loader_passes_a_3d_tiff_through_unchanged(tmp_path, caplog):
    path = tmp_path / "volume.tif"
    raw = np.random.randint(0, 255, (4, 6, 8), dtype=np.uint16)
    tifffile.imwrite(path, raw)

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        image, x, y, z, status = load_image_with_voxel_size_2d_aware(
            str(path), input_format="tif"
        )
    plain_image, plain_x, plain_y, plain_z, plain_status = load_3d_tif_with_voxel_size(
        str(path)
    )

    assert image.shape == (4, 6, 8)
    assert np.array_equal(image, plain_image)
    assert (x, y, z) == (plain_x, plain_y, plain_z)
    assert status == plain_status
    assert "promoted_from_2d" not in status
    assert not any(r.name == "haemolynx.io.load_2d" for r in caplog.records)


# --- load_image_with_voxel_size_2d_aware: H5 --------------------------------


def test_2d_aware_loader_promotes_a_2d_h5_and_warns(tmp_path, caplog):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((10, 14), dtype=np.uint8))

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        image, _x, _y, _z, status = load_image_with_voxel_size_2d_aware(
            str(path), input_format="h5"
        )

    assert image.shape == (1, 10, 14)
    assert status["promoted_from_2d"] is True
    assert any("2D image" in r.message for r in caplog.records)


def test_2d_aware_loader_passes_a_3d_h5_through_unchanged(tmp_path, caplog):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "volume.h5"
    raw = np.random.randint(0, 255, (4, 6, 8)).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=raw)

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        image, x, y, z, status = load_image_with_voxel_size_2d_aware(
            str(path), input_format="h5"
        )
    plain_image, plain_x, plain_y, plain_z, plain_status = load_3d_h5_with_voxel_size(
        str(path), "data"
    )

    assert image.shape == (4, 6, 8)
    assert np.array_equal(image, plain_image)
    assert (x, y, z) == (plain_x, plain_y, plain_z)
    assert status == plain_status
    assert "promoted_from_2d" not in status
    assert not any(r.name == "haemolynx.io.load_2d" for r in caplog.records)


def test_2d_aware_loader_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="input_format"):
        load_image_with_voxel_size_2d_aware(
            str(tmp_path / "whatever.png"), input_format="png"
        )


# --- backward compatibility of load_3d_h5_with_voxel_size's default --------


def test_load_3d_h5_with_voxel_size_still_rejects_2d_by_default(tmp_path):
    """allow_2d defaults to False; every existing caller (not going through
    load_2d) must keep raising for a 2D dataset exactly as before."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((10, 14), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected 3D image"):
        load_3d_h5_with_voxel_size(str(path), "data")


def test_load_3d_tif_with_voxel_size_still_rejects_2d_with_non_default_axis_order_by_default(
    tmp_path,
):
    """allow_2d defaults to False; every existing caller (not going through
    load_2d) must keep raising for a 2D TIFF with a non-default axis_order
    exactly as before."""
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.random.randint(0, 255, (12, 9), dtype=np.uint16))

    with pytest.raises(ValueError, match="requires a 3D volume"):
        load_3d_tif_with_voxel_size(str(path), axis_order="xyz")
