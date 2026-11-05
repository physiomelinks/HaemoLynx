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


def test_promote_2d_to_single_slice_volume_preserves_a_memmap_for_the_3d_case(tmp_path):
    """Regression: the 3D (was_2d=False) branch used to go through
    np.asarray, which silently downcasts an np.memmap to a plain ndarray --
    defeating use_memmap_loading for every 3D input, since this promotion
    runs unconditionally inside load_image_with_voxel_size_2d_aware."""
    path = tmp_path / "volume.tif"
    tifffile.imwrite(path, np.zeros((4, 6, 8), dtype=np.uint8))
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray(out="memmap")

    promoted, was_2d = promote_2d_to_single_slice_volume(image)

    assert was_2d is False
    assert isinstance(promoted, np.memmap)


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


def test_a_2d_images_z_spacing_is_its_finest_in_plane_one(tmp_path):
    """Regression: a 2D image read as a stack of one kept 1 um of z when its
    file gave none, so with 2 um pixels the single plane was the finest
    axis: a downsample spent its factor on z (2x reduced nothing) and every
    skeleton voxel count meant half a pixel."""
    from haemolynx.io.downsample import axis_downsample_factors
    from haemolynx.io.load_2d import load_image_with_voxel_size_2d_aware

    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, np.zeros((40, 40), dtype=np.uint8), resolution=(1 / 2.0, 1 / 3.0))

    image, x, y, z, status = load_image_with_voxel_size_2d_aware(str(path), input_format="tif")

    assert image.shape == (1, 40, 40)
    assert (x, y, z) == pytest.approx((2.0, 3.0, 2.0))
    # Still missing, so auto takes an override; and no 1 um warning for it.
    assert status["missing_axes"] == ["z"] and status["z_from_in_plane"]
    from haemolynx.io import resolve_voxel_size_xyz

    assert resolve_voxel_size_xyz((x, y, z), status, [5.0, 5.0, 5.0], "auto")[0] == (5.0, 5.0, 5.0)
    assert resolve_voxel_size_xyz((x, y, z), status, None, "auto")[0] == pytest.approx((2.0, 3.0, 2.0))
    # x is reduced by the whole factor, as on a 3D image of these pixels.
    assert axis_downsample_factors(2, (z, y, x))[2] == 2


def test_a_2d_image_with_no_pixel_size_keeps_its_fallback(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "plane.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((10, 12), dtype=np.uint8))

    _image, x, y, z, status = load_image_with_voxel_size_2d_aware(str(path), input_format="h5")

    assert (x, y, z) == (1.0, 1.0, 1.0)
    assert "z_from_in_plane" not in status
