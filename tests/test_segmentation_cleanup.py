"""Pre-skeletonization cleanup of the raw segmented binary vessel mask."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.preprocessing import segmentation_cleanup as sc
from test_thick_vessel_skeletonisation import _disk_tube_along_axis


def _cylinder_along_x(
    shape: tuple[int, int, int], *, z: float, y: float, radius: float, x0: int, x1: int
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


# --- reconnect_vessel_like_components ----------------------------------------


def test_reconnect_bridges_two_collinear_tube_fragments_with_a_gap():
    shape = (10, 10, 40)
    mask = _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=0, x1=14)
    mask |= _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=20, x1=34)

    cleaned, stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0
    )

    assert stats["accepted_bridges"] == 1
    labeled, count = sc._connected_components(cleaned)
    assert count == 1


def test_reconnect_does_not_bridge_a_perpendicular_fragment_pair():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=1.0, x0=0, x1=8)
    # A second fragment near the first's end, running along z instead of x.
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((yy - 10.0) ** 2 + (xx - 12.0) ** 2)
    mask |= (radial <= 1.0) & (zz >= 0) & (zz <= 8)

    _cleaned, stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0,
        max_axis_angle_degrees=30.0,
    )

    assert stats["accepted_bridges"] == 0
    assert stats["rejected_reasons"]


def test_reconnect_does_not_bridge_two_blob_shaped_components():
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[2:8, 2:8, 10:16] = True

    _cleaned, stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0
    )

    assert stats["accepted_bridges"] == 0
    assert "source_not_cylindrical" in stats["rejected_reasons"] or (
        "target_not_cylindrical" in stats["rejected_reasons"]
    )


def test_reconnect_leaves_a_lone_tube_with_no_neighbor_unchanged():
    shape = (10, 10, 30)
    mask = _cylinder_along_x(shape, z=5, y=5, radius=1.0, x0=0, x1=25)

    cleaned, stats = sc.reconnect_vessel_like_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0
    )

    assert stats["attempted_bridges"] == 0
    assert np.array_equal(cleaned, mask)


def test_reconnect_gates_bridge_distance_in_physical_microns_not_voxels():
    """A voxel-count gap that is far in microns along a coarse axis must not
    bridge, while the same voxel-count gap along a fine axis (a shorter
    physical distance) does -- proving the gate reads voxel_size_zyx, not
    raw voxel counts."""
    shape = (30, 10, 10)
    voxel_size_zyx = (2.0, 0.4, 0.4)

    far = np.zeros(shape, dtype=bool)
    far[0:10, 4:6, 4:6] = True
    far[15:25, 4:6, 4:6] = True  # 5-voxel gap along z * 2.0um = 10um
    _cleaned_far, stats_far = sc.reconnect_vessel_like_components(
        far, voxel_size_zyx=voxel_size_zyx, max_bridge_distance_um=8.0
    )
    assert stats_far["accepted_bridges"] == 0

    near = np.zeros(shape, dtype=bool)
    near[0:10, 4:6, 4:6] = True
    near[13:23, 4:6, 4:6] = True  # 3-voxel gap along z * 2.0um = 6um
    _cleaned_near, stats_near = sc.reconnect_vessel_like_components(
        near, voxel_size_zyx=voxel_size_zyx, max_bridge_distance_um=8.0
    )
    assert stats_near["accepted_bridges"] == 1


# --- smooth_vessel_surfaces ---------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def test_smooth_reduces_surface_roughness_of_a_jagged_tube():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    flip = rng.random(shape) < 0.05
    noisy[flip] = ~noisy[flip]

    smoothed = sc.smooth_vessel_surfaces(noisy, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=1.5)

    assert _iou(smoothed, clean) > _iou(noisy, clean)


def test_smooth_leaves_a_clean_tube_nearly_unchanged():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)

    smoothed = sc.smooth_vessel_surfaces(clean, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=0.6)

    assert _iou(smoothed, clean) > 0.9


def test_smooth_sigma_zero_is_a_no_op():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, :] = True
    out = sc.smooth_vessel_surfaces(mask, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=0.0)
    assert np.array_equal(out, mask)
    assert out is mask


# --- remove_small_segmented_volumes -------------------------------------------


def test_remove_small_drops_a_speck_below_threshold_and_keeps_the_vessel():
    shape = (10, 10, 10)
    vessel = np.zeros(shape, dtype=bool)
    vessel[2:8, 4:6, 4:6] = True  # 6*2*2 = 24 voxels
    speck = np.zeros(shape, dtype=bool)
    speck[0, 0, 0] = True  # 1 voxel
    combo = vessel | speck

    out = sc.remove_small_segmented_volumes(
        combo, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=5.0
    )

    assert not out[0, 0, 0]
    assert np.array_equal(out & vessel, vessel)


def test_remove_small_keeps_a_component_just_above_threshold():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[0:2, 0:3, 0:1] = True  # 6 voxels, above a 5-voxel threshold

    out = sc.remove_small_segmented_volumes(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=5.0
    )

    assert np.array_equal(out, mask)


def test_remove_small_min_volume_zero_is_a_no_op():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = True
    out = sc.remove_small_segmented_volumes(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_volume_um3=0.0
    )
    assert out is mask


# --- clean_segmented_mask_for_skeletonisation (orchestrator) -----------------


def test_clean_segmented_mask_all_off_returns_input_unchanged_and_calls_nothing(
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setattr(
        sc, "reconnect_vessel_like_components",
        lambda *a, **k: calls.append("reconnect") or (None, {}),
    )
    monkeypatch.setattr(
        sc, "smooth_vessel_surfaces", lambda *a, **k: calls.append("smooth") or None
    )
    monkeypatch.setattr(
        sc, "remove_small_segmented_volumes",
        lambda *a, **k: calls.append("remove_small") or None,
    )

    image = np.zeros((4, 4, 4), dtype=bool)
    cleaned, raw = sc.clean_segmented_mask_for_skeletonisation(
        image, voxel_size_zyx=(1.0, 1.0, 1.0)
    )

    assert calls == []
    assert cleaned is image
    assert raw is None


def test_clean_segmented_mask_runs_reconnect_then_smooth_then_remove_small_in_order(
    monkeypatch,
):
    order: list[str] = []

    def fake_reconnect(mask, **kwargs):
        order.append("reconnect")
        return mask, {}

    def fake_smooth(mask, **kwargs):
        order.append("smooth")
        return mask

    def fake_remove_small(mask, **kwargs):
        order.append("remove_small")
        return mask

    monkeypatch.setattr(sc, "reconnect_vessel_like_components", fake_reconnect)
    monkeypatch.setattr(sc, "smooth_vessel_surfaces", fake_smooth)
    monkeypatch.setattr(sc, "remove_small_segmented_volumes", fake_remove_small)

    image = np.zeros((4, 4, 4), dtype=bool)
    sc.clean_segmented_mask_for_skeletonisation(
        image,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        reconnect_gaps=True,
        smooth_surfaces=True,
        remove_small_volumes=True,
    )

    assert order == ["reconnect", "smooth", "remove_small"]


def test_clean_segmented_mask_returns_raw_only_when_a_step_ran():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 4:6, 4:6] = True
    speck = mask.copy()
    speck[0, 0, 0] = True

    cleaned, raw = sc.clean_segmented_mask_for_skeletonisation(
        speck, voxel_size_zyx=(1.0, 1.0, 1.0), remove_small_volumes=True,
        remove_small_min_volume_um3=5.0,
    )

    assert raw is not None
    assert np.array_equal(raw, speck)
    assert not cleaned[0, 0, 0]
