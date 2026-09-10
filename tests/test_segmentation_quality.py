"""Score a raw segmented mask, 0-10, before it is ever skeletonised."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.preprocessing import segmentation_quality as sq


def _cylinder_along_x(
    shape: tuple[int, int, int], *, z: float, y: float, radius: float, x0: int, x1: int
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


# --- degenerate input -----------------------------------------------------


def test_empty_mask_scores_zero_on_every_axis():
    mask = np.zeros((10, 10, 10), dtype=bool)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.total == 0.0
    assert score.fragmentation == 0.0
    assert score.connectivity == 0.0
    assert score.boundary_vessels == 0.0
    assert score.noise == 0.0
    assert score.resolution == 0.0


def test_total_is_the_sum_of_the_five_sub_scores():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=5.0, x0=5, x1=14)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.total == pytest.approx(
        score.fragmentation
        + score.connectivity
        + score.boundary_vessels
        + score.noise
        + score.resolution
    )
    assert 0.0 <= score.total <= 10.0


# --- fragmentation ----------------------------------------------------------


def test_fragmentation_score_drops_as_small_speck_count_grows():
    shape = (30, 30, 30)
    body = _cylinder_along_x(shape, z=15, y=15, radius=5.0, x0=5, x1=24)

    clean = body.copy()
    speckled = body.copy()
    rng = np.random.default_rng(0)
    for _ in range(15):
        z, y, x = rng.integers(0, 30, size=3)
        speckled[z, y, x] = True

    score_clean = sq.score_segmented_mask(clean, voxel_size_zyx=(1.0, 1.0, 1.0))
    score_speckled = sq.score_segmented_mask(speckled, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score_clean.small_fragment_count == 0
    assert score_speckled.small_fragment_count > 0
    assert score_speckled.fragmentation < score_clean.fragmentation


def test_a_single_component_has_no_small_fragments():
    shape = (15, 15, 15)
    mask = _cylinder_along_x(shape, z=7, y=7, radius=3.0, x0=2, x1=12)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.component_count == 1
    assert score.small_fragment_count == 0
    assert score.fragmentation == pytest.approx(2.0)


# --- connectivity -------------------------------------------------------------


def test_connectivity_score_is_full_marks_for_one_connected_network():
    shape = (15, 15, 15)
    mask = _cylinder_along_x(shape, z=7, y=7, radius=3.0, x0=2, x1=12)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.largest_fraction == pytest.approx(1.0)
    assert score.connectivity == pytest.approx(2.0)


def test_connectivity_score_drops_when_the_network_splits_roughly_in_two():
    shape = (15, 15, 40)
    half_a = _cylinder_along_x(shape, z=7, y=7, radius=3.0, x0=0, x1=14)
    half_b = _cylinder_along_x(shape, z=7, y=7, radius=3.0, x0=25, x1=39)
    mask = half_a | half_b

    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.largest_fraction == pytest.approx(0.5, abs=0.05)
    assert score.connectivity == pytest.approx(1.0, abs=0.1)


# --- boundary vessels ---------------------------------------------------------


def test_boundary_score_is_full_marks_when_nothing_touches_the_edge():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=3.0, x0=5, x1=14)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.boundary_patch_count == 0
    assert score.boundary_vessels == pytest.approx(2.0)


def test_boundary_score_stays_full_marks_for_a_normal_two_ended_vessel():
    """One inlet, one outlet -- the allowance exists exactly for this case."""
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=3.0, x0=0, x1=19)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.boundary_patch_count == 2
    assert score.boundary_vessels == pytest.approx(2.0)


def test_boundary_score_drops_as_more_places_touch_the_edge():
    shape = (20, 20, 20)
    trunk = _cylinder_along_x(shape, z=10, y=10, radius=2.0, x0=0, x1=19)
    # Two extra branches poking out of two more faces, on top of the
    # trunk's own two ends (x=0 and x=19) -- four boundary patches in all.
    extra = np.zeros(shape, dtype=bool)
    extra[0:3, 8:12, 8:12] = True  # touches the z=0 face
    extra[8:12, 0:3, 8:12] = True  # touches the y=0 face
    mask = trunk | extra

    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.boundary_patch_count > 2
    assert score.boundary_vessels < 2.0


# --- noise ---------------------------------------------------------------------


def test_noise_score_is_high_for_a_clean_smooth_tube():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.noise > 1.5


def test_noise_score_drops_for_a_jagged_surface():
    shape = (20, 20, 20)
    clean = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    rng = np.random.default_rng(1)
    jagged = clean.copy()
    flip = rng.random(shape) < 0.08
    jagged[flip] = ~jagged[flip]

    score_clean = sq.score_segmented_mask(clean, voxel_size_zyx=(1.0, 1.0, 1.0))
    score_jagged = sq.score_segmented_mask(jagged, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score_jagged.noise < score_clean.noise


# --- resolution ------------------------------------------------------------------


def test_resolution_score_is_high_for_a_well_sampled_vessel():
    # The score reads the *median* distance-to-background over the whole
    # solid cylinder, not its nominal radius -- most of a filled disk's own
    # area sits near its edge, so the median is well below the radius
    # (~0.29x for a 2D disk); a big enough radius still saturates the score.
    shape = (30, 30, 30)
    mask = _cylinder_along_x(shape, z=15, y=15, radius=12.0, x0=2, x1=27)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.resolution == pytest.approx(2.0)


def test_resolution_score_drops_for_a_vessel_barely_wider_than_a_voxel():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=0.9, x0=2, x1=17)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert score.resolution < 1.0


def test_resolution_score_reads_physical_microns_not_voxels():
    """The same voxel-radius vessel scores differently once the coarsest
    axis is expressed in coarser physical units -- proving the score reads
    voxel_size_zyx, not a raw voxel count."""
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=3.0, x0=2, x1=17)

    fine = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))
    coarse = sq.score_segmented_mask(mask, voxel_size_zyx=(4.0, 1.0, 1.0))

    assert coarse.resolution < fine.resolution


# --- report text -----------------------------------------------------------------


def test_report_text_includes_the_total_and_every_sub_score():
    shape = (15, 15, 15)
    mask = _cylinder_along_x(shape, z=7, y=7, radius=3.0, x0=2, x1=12)
    score = sq.score_segmented_mask(mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    text = sq.format_segmentation_quality_report(score)

    assert f"{score.total:.1f}/10" in text
    assert "fragmentation" in text
    assert "connectivity" in text
    assert "boundary vessels" in text
    assert "noise" in text
    assert "resolution" in text
