"""Tests for input axis-order handling and canonical (z, y, x) conversion."""
import numpy as np
import pytest
import tifffile

from ImageLynx.io import (
    CANONICAL_AXIS_ORDER,
    VALID_AXIS_ORDERS,
    apply_axis_order,
    axis_order_transpose,
    load_3d_tif_with_voxel_size,
    normalize_axis_order,
    voxel_size_xyz_from_zyx,
    voxel_size_zyx_from_xyz,
)


def _distinct_shape_volume() -> np.ndarray:
    """Volume whose axes have distinct lengths, so transposes are detectable."""
    return np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)


def test_normalize_axis_order_accepts_permutations_and_normalizes_case():
    assert normalize_axis_order("ZYX") == "zyx"
    assert normalize_axis_order(" xyz ") == "xyz"
    for order in VALID_AXIS_ORDERS:
        assert normalize_axis_order(order) == order


@pytest.mark.parametrize("bad", ["zy", "zyxx", "zzz", "abc", "", "z y x", None, 3])
def test_normalize_axis_order_rejects_non_permutations(bad):
    with pytest.raises(ValueError, match="permutation of 'xyz'|must be a string"):
        normalize_axis_order(bad)


def test_axis_order_transpose_maps_named_axis_to_canonical_position():
    # Volume stored (x, y, z): z is axis 2, y is axis 1, x is axis 0.
    assert axis_order_transpose("xyz") == (2, 1, 0)
    assert axis_order_transpose("zyx") == (0, 1, 2)
    # Volume stored (y, z, x): z is axis 1, y is axis 0, x is axis 2.
    assert axis_order_transpose("yzx") == (1, 0, 2)


def test_apply_axis_order_is_identity_for_canonical_order():
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, CANONICAL_AXIS_ORDER)
    assert result.shape == volume.shape
    assert np.array_equal(result, volume)


def test_apply_axis_order_moves_the_named_z_axis_to_axis_zero():
    # Shape (2, 3, 4) stored as (x, y, z) means x=2, y=3, z=4.
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, "xyz")
    assert result.shape == (4, 3, 2)
    # A voxel at file index (x=1, y=2, z=3) must land at canonical (z=3, y=2, x=1).
    assert result[3, 2, 1] == volume[1, 2, 3]


@pytest.mark.parametrize("axis_order", VALID_AXIS_ORDERS)
def test_apply_axis_order_preserves_voxel_values_for_every_permutation(axis_order):
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, axis_order)
    assert sorted(result.shape) == sorted(volume.shape)
    assert np.array_equal(np.sort(result.ravel()), np.sort(volume.ravel()))
    # The axis named "z" must become axis 0 with its original length.
    assert result.shape[0] == volume.shape[axis_order.index("z")]
    assert result.shape[1] == volume.shape[axis_order.index("y")]
    assert result.shape[2] == volume.shape[axis_order.index("x")]


def test_apply_axis_order_rejects_non_3d_volume_for_non_canonical_order():
    with pytest.raises(ValueError, match="requires a 3D volume"):
        apply_axis_order(np.zeros((4, 4)), "xyz")


def test_voxel_size_conversion_reverses_axis_order():
    assert voxel_size_zyx_from_xyz((0.5, 0.6, 2.0)) == (2.0, 0.6, 0.5)
    assert voxel_size_xyz_from_zyx((2.0, 0.6, 0.5)) == (0.5, 0.6, 2.0)


def test_voxel_size_conversion_round_trips():
    voxel_size_xyz = (0.325, 0.4, 3.0)
    assert voxel_size_xyz_from_zyx(voxel_size_zyx_from_xyz(voxel_size_xyz)) == voxel_size_xyz


@pytest.mark.parametrize("bad", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
def test_voxel_size_conversion_rejects_wrong_length(bad):
    with pytest.raises(ValueError, match="must have length 3"):
        voxel_size_zyx_from_xyz(bad)


def test_tif_loader_applies_axis_order(tmp_path):
    """A file written as (x, y, z) loads transposed to canonical (z, y, x)."""
    volume = _distinct_shape_volume()  # (x=2, y=3, z=4) under axis_order="xyz"
    path = tmp_path / "xyz_volume.tif"
    tifffile.imwrite(str(path), volume)

    canonical, _vx, _vy, _vz, _status = load_3d_tif_with_voxel_size(
        str(path), axis_order="xyz"
    )
    assert canonical.shape == (4, 3, 2)
    assert canonical[3, 2, 1] == volume[1, 2, 3]

    as_stored, _vx, _vy, _vz, _status = load_3d_tif_with_voxel_size(str(path))
    assert as_stored.shape == volume.shape
