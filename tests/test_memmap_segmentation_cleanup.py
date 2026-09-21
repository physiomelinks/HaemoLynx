"""``use_memmap_loading`` changes where segmentation-cleanup's own full-volume
working buffers (distance transforms, connected-component labels, the
gaussian-smoothing float cast) live, not what they compute.

Every test compares a memmap-backed result against the ordinary in-RAM one
on the same small fixture: the invariant pinned throughout is that turning
the toggle on never changes a value, only where it is stored.
"""
from __future__ import annotations

import numpy as np

from haemolynx.preprocessing import segmentation_cleanup as sc


def _cylinder_along_x(
    shape: tuple[int, int, int], *, z: float, y: float, radius: float, x0: int, x1: int
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


def _sphere(
    shape: tuple[int, int, int], center: tuple[float, float, float], radius: float
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    cz, cy, cx = center
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def _dumbbell(shape, *, sphere_radius, neck_radius, x_a, x_b):
    sphere_a = _sphere(shape, (10, 10, x_a), sphere_radius)
    sphere_b = _sphere(shape, (10, 10, x_b), sphere_radius)
    neck = _cylinder_along_x(shape, z=10, y=10, radius=neck_radius, x0=x_a, x1=x_b)
    return sphere_a | sphere_b | neck


# --- reconnect_vessel_like_components ----------------------------------------


def test_reconnect_gives_the_same_result_with_memmap_on():
    shape = (10, 10, 40)
    mask = _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=0, x1=14)
    mask |= _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=20, x1=34)

    memmap_cleaned, memmap_stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0, use_memmap=True
    )
    eager_cleaned, eager_stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0, use_memmap=False
    )

    assert np.array_equal(memmap_cleaned, eager_cleaned)
    assert memmap_stats == eager_stats
    assert memmap_stats["accepted_bridges"] == 1


def test_reconnect_with_memmap_on_still_returns_early_with_fewer_than_two_components():
    shape = (10, 10, 10)
    mask = _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=0, x1=8)

    cleaned, stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=True
    )

    assert np.array_equal(cleaned, mask)
    assert stats["attempted_bridges"] == 0


# --- split_narrow_neck_components ---------------------------------------------


def test_split_gives_the_same_result_with_memmap_on():
    shape = (20, 20, 35)
    mask = _dumbbell(shape, sphere_radius=5.0, neck_radius=1.5, x_a=6, x_b=26)

    memmap_cleaned, memmap_stats = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=True
    )
    eager_cleaned, eager_stats = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=False
    )

    assert np.array_equal(memmap_cleaned, eager_cleaned)
    assert memmap_stats == eager_stats
    assert memmap_stats["cuts_made"] == 1


def test_split_with_memmap_on_still_returns_early_for_an_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)

    cleaned, stats = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=True
    )

    assert np.array_equal(cleaned, mask)
    assert stats["bodies_found"] == 0


def test_split_with_memmap_on_still_returns_early_with_fewer_than_two_peaks():
    shape = (20, 20, 35)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=8)

    cleaned, stats = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=True
    )

    assert np.array_equal(cleaned, mask)
    assert stats["cuts_made"] == 0


# --- smooth_vessel_surfaces (gaussian) ----------------------------------------


def test_smooth_gaussian_gives_the_same_result_with_memmap_on():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    flip = rng.random(shape) < 0.05
    noisy[flip] = ~noisy[flip]

    memmap_result = sc.smooth_vessel_surfaces(
        noisy, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=1.5, use_memmap=True
    )
    eager_result = sc.smooth_vessel_surfaces(
        noisy, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=1.5, use_memmap=False
    )

    assert np.array_equal(memmap_result, eager_result)


def test_smooth_morphological_ignores_use_memmap():
    """The morphological method never leaves boolean, so there is nothing
    for the flag to redirect -- it must still be accepted, not raise."""
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)

    memmap_result = sc.smooth_vessel_surfaces(
        mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        method="morphological",
        morphological_radius_um=1.0,
        use_memmap=True,
    )
    eager_result = sc.smooth_vessel_surfaces(
        mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        method="morphological",
        morphological_radius_um=1.0,
        use_memmap=False,
    )

    assert np.array_equal(memmap_result, eager_result)


# --- remove_small_segmented_volumes -------------------------------------------


def test_remove_small_segmented_volumes_gives_the_same_result_with_memmap_on():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 4:6, 4:6] = True  # a real vessel segment, kept
    mask[0, 0, 0] = True  # a one-voxel speck, dropped

    memmap_result = sc.remove_small_segmented_volumes(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=5.0, use_memmap=True
    )
    eager_result = sc.remove_small_segmented_volumes(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=5.0, use_memmap=False
    )

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    assert not memmap_result[0, 0, 0], "the one-voxel speck should have been dropped"
    assert memmap_result[4, 5, 5], "the real vessel segment should survive"


def test_remove_small_segmented_volumes_with_memmap_on_still_no_ops_below_threshold():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    result = sc.remove_small_segmented_volumes(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=0.0, use_memmap=True
    )

    assert result is mask


# --- clean_segmented_mask_for_skeletonisation ---------------------------------


def test_clean_segmented_mask_gives_the_same_result_with_memmap_on():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 4:6, 4:6] = True
    speck = mask.copy()
    speck[0, 0, 0] = True

    memmap_cleaned, memmap_raw = sc.clean_segmented_mask_for_skeletonisation(
        speck,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        remove_small_volumes=True,
        remove_small_min_volume_um3=5.0,
        use_memmap=True,
    )
    eager_cleaned, eager_raw = sc.clean_segmented_mask_for_skeletonisation(
        speck,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        remove_small_volumes=True,
        remove_small_min_volume_um3=5.0,
        use_memmap=False,
    )

    assert isinstance(memmap_raw, np.memmap)
    assert not isinstance(eager_raw, np.memmap)
    assert np.array_equal(memmap_cleaned, eager_cleaned)
    assert np.array_equal(memmap_raw, eager_raw)
    assert np.array_equal(memmap_raw, speck)
