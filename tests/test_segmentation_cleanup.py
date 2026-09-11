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


def _sphere(
    shape: tuple[int, int, int], center: tuple[float, float, float], radius: float
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    cz, cy, cx = center
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


# --- fill_enclosed_cavities ---------------------------------------------------


def test_fill_cavities_fills_an_internal_void_but_not_outside_background():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False  # an enclosed cavity, imaging-noise-style

    filled = sc.fill_enclosed_cavities(mask)

    assert filled[5, 5, 5]  # the cavity itself
    assert np.array_equal(filled[2:8, 2:8, 2:8], np.ones((6, 6, 6), dtype=bool))
    assert not filled[0, 0, 0]  # true background, outside the shell entirely


def test_fill_cavities_leaves_a_solid_mask_unchanged():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 2:8, 2:8] = True

    filled = sc.fill_enclosed_cavities(mask)

    assert np.array_equal(filled, mask)


def test_fill_cavities_does_not_fill_a_cavity_open_to_the_background():
    """A void that reaches the mask's own boundary is not "enclosed" -- it
    is background, and must stay background."""
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 2:6] = False  # channel from the cavity out through a face

    filled = sc.fill_enclosed_cavities(mask)

    assert not filled[5, 5, 3]
    assert np.array_equal(filled, mask)


def test_fill_cavities_empty_mask_is_a_no_op():
    mask = np.zeros((5, 5, 5), dtype=bool)
    assert np.array_equal(sc.fill_enclosed_cavities(mask), mask)


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


# --- split_narrow_neck_components ---------------------------------------------


def _dumbbell(shape, *, sphere_radius, neck_radius, x_a, x_b):
    sphere_a = _sphere(shape, (10, 10, x_a), sphere_radius)
    sphere_b = _sphere(shape, (10, 10, x_b), sphere_radius)
    neck = _cylinder_along_x(shape, z=10, y=10, radius=neck_radius, x0=x_a, x1=x_b)
    return sphere_a | sphere_b | neck


def test_split_cuts_a_dumbbell_at_its_genuine_pinch():
    shape = (20, 20, 35)
    mask = _dumbbell(shape, sphere_radius=5.0, neck_radius=1.5, x_a=6, x_b=26)
    _labeled, before = sc._connected_components(mask)
    assert before == 1

    cleaned, stats = sc.split_narrow_neck_components(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert stats["cuts_made"] == 1
    _labeled, after = sc._connected_components(cleaned)
    assert after == 2


def test_split_leaves_a_uniform_radius_tube_as_one_component():
    """Watershed may still seed multiple markers along a flat ridge, but a
    uniform vessel has no narrowing at any interface, so nothing is cut."""
    shape = (20, 20, 35)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=32)

    cleaned, stats = sc.split_narrow_neck_components(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert stats["cuts_made"] == 0
    _labeled, after = sc._connected_components(cleaned)
    assert after == 1
    assert np.array_equal(cleaned, mask)


def test_split_leaves_already_separate_bodies_untouched():
    shape = (20, 20, 35)
    mask = _sphere(shape, (10, 10, 6), 5.0) | _sphere(shape, (10, 10, 26), 5.0)

    cleaned, stats = sc.split_narrow_neck_components(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert stats["adjacent_pairs"] == 0
    assert stats["cuts_made"] == 0
    assert np.array_equal(cleaned, mask)


def test_split_min_body_radius_guard_skips_a_body_too_thin_to_judge():
    shape = (20, 20, 35)
    # An asymmetric dumbbell: one normal-sized sphere, one tiny one -- too
    # thin to trust as a real "second vessel body" rather than noise.
    big_sphere = _sphere(shape, (10, 10, 6), 5.0)
    tiny_sphere = _sphere(shape, (10, 10, 26), 0.8)
    neck = _cylinder_along_x(shape, z=10, y=10, radius=0.6, x0=6, x1=26)
    mask = big_sphere | tiny_sphere | neck
    # neck_radius=0.6 alone (lenient ratio) would otherwise cut -- isolate the
    # size guard by asking for a body radius the tiny end cannot reach.
    cleaned, stats = sc.split_narrow_neck_components(
        mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        min_pinch_radius_ratio=0.99,
        min_body_radius_um=2.0,
    )

    assert stats["cuts_made"] == 0
    assert "body_too_thin" in stats["rejected_reasons"]
    assert np.array_equal(cleaned, mask)


def test_split_pinch_radius_ratio_gates_how_much_narrowing_counts():
    """The same mild narrowing is rejected under a strict ratio and accepted
    under a lenient one -- proving the ratio, not a fixed radius, decides."""
    shape = (20, 20, 35)
    mask = _dumbbell(shape, sphere_radius=5.0, neck_radius=3.5, x_a=6, x_b=26)

    _cleaned_strict, stats_strict = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_pinch_radius_ratio=0.5
    )
    assert stats_strict["cuts_made"] == 0

    _cleaned_lenient, stats_lenient = sc.split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), min_pinch_radius_ratio=0.9
    )
    assert stats_lenient["cuts_made"] == 1


def test_split_empty_mask_is_a_no_op():
    mask = np.zeros((10, 10, 10), dtype=bool)
    cleaned, stats = sc.split_narrow_neck_components(mask, voxel_size_zyx=(1.0, 1.0, 1.0))
    assert np.array_equal(cleaned, mask)
    assert stats["bodies_found"] == 0


# --- remove_surface_whiskers ---------------------------------------------------


def test_remove_whiskers_strips_a_thin_attached_spike_but_keeps_the_body():
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    # A single-voxel-wide spike attached to the body's own surface (y=14),
    # sticking straight out to y=18.
    spike = np.zeros(shape, dtype=bool)
    spike[10, 14:19, 9] = True
    mask = body | spike
    assert mask[10, 18, 9] and not body[10, 18, 9]

    out = sc.remove_surface_whiskers(mask, voxel_size_zyx=(1.0, 1.0, 1.0), whisker_radius_um=1.0)

    assert not out[10, 18, 9]
    assert _iou(out, body) > 0.9


def test_remove_whiskers_leaves_a_thick_clean_tube_nearly_unchanged():
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=5.0, x0=2, x1=17)

    out = sc.remove_surface_whiskers(body, voxel_size_zyx=(1.0, 1.0, 1.0), whisker_radius_um=1.0)

    assert _iou(out, body) > 0.9


def test_remove_whiskers_radius_zero_is_a_no_op():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, :] = True
    out = sc.remove_surface_whiskers(mask, voxel_size_zyx=(1.0, 1.0, 1.0), whisker_radius_um=0.0)
    assert np.array_equal(out, mask)
    assert out is mask


def test_remove_whiskers_converts_physical_radius_per_axis_anisotropically(monkeypatch):
    """The structuring element radius must come from voxel_size_zyx per axis,
    not a single scalar -- else an anisotropic dataset opens more
    aggressively along its coarser axis than its finer ones."""
    captured = {}

    def fake_ellipsoid(radius_voxels):
        captured["radius_voxels"] = radius_voxels
        return np.ones((1, 1, 1), dtype=bool)

    monkeypatch.setattr(sc, "_ellipsoid_structure", fake_ellipsoid)
    monkeypatch.setattr(sc, "binary_opening", lambda mask, structure=None: mask)

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    sc.remove_surface_whiskers(mask, voxel_size_zyx=(2.0, 0.5, 0.25), whisker_radius_um=1.0)

    assert captured["radius_voxels"] == (1, 2, 4)


def test_ellipsoid_structure_shape_matches_per_axis_radius():
    structure = sc._ellipsoid_structure((1, 3, 2))
    assert structure.shape == (3, 7, 5)
    assert structure[1, 3, 2]


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


def test_smooth_unknown_method_raises_value_error():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, :] = True
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        sc.smooth_vessel_surfaces(mask, voxel_size_zyx=(1.0, 1.0, 1.0), method="bogus")


def test_smooth_morphological_reduces_surface_roughness_of_a_jagged_tube():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    flip = rng.random(shape) < 0.05
    noisy[flip] = ~noisy[flip]

    smoothed = sc.smooth_vessel_surfaces(
        noisy, voxel_size_zyx=(1.0, 1.0, 1.0), method="morphological", morphological_radius_um=1.5
    )

    assert _iou(smoothed, clean) > _iou(noisy, clean)


def test_smooth_morphological_leaves_a_clean_tube_nearly_unchanged():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)

    smoothed = sc.smooth_vessel_surfaces(
        clean, voxel_size_zyx=(1.0, 1.0, 1.0), method="morphological", morphological_radius_um=0.6
    )

    assert _iou(smoothed, clean) > 0.9


def test_smooth_morphological_radius_zero_is_a_no_op():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, :] = True
    out = sc.smooth_vessel_surfaces(
        mask, voxel_size_zyx=(1.0, 1.0, 1.0), method="morphological", morphological_radius_um=0.0
    )
    assert np.array_equal(out, mask)
    assert out is mask


def test_smooth_morphological_preserves_cross_section_better_than_gaussian_on_a_thin_vessel():
    """The curvature bias this feature exists to avoid: gaussian
    blur-then-rethreshold shrinks a round vessel's cross-section (a
    threshold-driven effect that grows with curvature, i.e. worse the
    thinner the vessel), while morphological closing-then-opening -- no
    threshold step -- shrinks it markedly less at the same physical scale."""
    shape = (30, 30, 30)
    tube = _cylinder_along_x(shape, z=15, y=15, radius=5.0, x0=2, x1=27)
    mid_x = 14  # well inside the tube body, away from its flat end caps
    original_area = int(tube[:, :, mid_x].sum())

    gaussian = sc.smooth_vessel_surfaces(
        tube, voxel_size_zyx=(1.0, 1.0, 1.0), sigma_um=1.5, method="gaussian"
    )
    morphological = sc.smooth_vessel_surfaces(
        tube,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        method="morphological",
        morphological_radius_um=1.5,
    )

    gaussian_area = int(gaussian[:, :, mid_x].sum())
    morphological_area = int(morphological[:, :, mid_x].sum())

    assert gaussian_area < original_area  # the bias this feature exists to avoid
    assert morphological_area > gaussian_area
    assert abs(morphological_area - original_area) < abs(gaussian_area - original_area)


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
        sc, "fill_enclosed_cavities",
        lambda *a, **k: calls.append("fill_cavities") or None,
    )
    monkeypatch.setattr(
        sc, "remove_surface_whiskers",
        lambda *a, **k: calls.append("remove_whiskers") or None,
    )
    monkeypatch.setattr(
        sc, "split_narrow_neck_components",
        lambda *a, **k: calls.append("split_narrow_necks") or (None, {}),
    )
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


def test_clean_segmented_mask_runs_every_step_in_order(monkeypatch):
    order: list[str] = []

    def fake_fill_cavities(mask, **kwargs):
        order.append("fill_cavities")
        return mask

    def fake_whiskers(mask, **kwargs):
        order.append("remove_whiskers")
        return mask

    def fake_split(mask, **kwargs):
        order.append("split_narrow_necks")
        return mask, {}

    def fake_reconnect(mask, **kwargs):
        order.append("reconnect")
        return mask, {}

    def fake_smooth(mask, **kwargs):
        order.append("smooth")
        return mask

    def fake_remove_small(mask, **kwargs):
        order.append("remove_small")
        return mask

    monkeypatch.setattr(sc, "fill_enclosed_cavities", fake_fill_cavities)
    monkeypatch.setattr(sc, "remove_surface_whiskers", fake_whiskers)
    monkeypatch.setattr(sc, "split_narrow_neck_components", fake_split)
    monkeypatch.setattr(sc, "reconnect_vessel_like_components", fake_reconnect)
    monkeypatch.setattr(sc, "smooth_vessel_surfaces", fake_smooth)
    monkeypatch.setattr(sc, "remove_small_segmented_volumes", fake_remove_small)

    image = np.zeros((4, 4, 4), dtype=bool)
    sc.clean_segmented_mask_for_skeletonisation(
        image,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        fill_cavities=True,
        remove_whiskers=True,
        split_narrow_necks=True,
        reconnect_gaps=True,
        smooth_surfaces=True,
        remove_small_volumes=True,
    )

    assert order == [
        "fill_cavities",
        "remove_whiskers",
        "split_narrow_necks",
        "reconnect",
        "smooth",
        "remove_small",
    ]


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
