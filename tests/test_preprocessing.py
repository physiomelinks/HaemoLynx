"""Tests for preprocessing module."""
import pytest
import numpy as np

from ImageLynx.preprocessing import (
    bridge_gaps,
    close_binary_mask,
    skeletonize_3d,
    preprocess_skeleton_for_graph,
)


def test_bridge_gaps(small_binary_3d):
    result = bridge_gaps(small_binary_3d, max_gap=1)
    assert result.shape == small_binary_3d.shape
    assert np.any(result)


def test_bridge_gaps_dilates_but_close_binary_mask_does_not():
    """Why the mask cleanup path uses close_binary_mask rather than bridge_gaps.

    bridge_gaps is a plain dilation - it never erodes back - so an isolated solid object
    grows by max_gap in every direction. On a vessel mask that adds a voxel of radius to
    every vessel unconditionally, and because cross-sectional area goes as the square of the
    radius the bias is not size-neutral: +1 voxel is +125% area on a 2-voxel radius but only
    +36% on a 6-voxel one, so narrow capillaries are inflated hardest. A closing bridges the
    same gaps while leaving an isolated object's size untouched.
    """
    volume = np.zeros((21, 21, 21), dtype=bool)
    volume[8:13, 8:13, 8:13] = True   # isolated 5x5x5 cube, well clear of the borders
    original_count = int(volume.sum())

    dilated = bridge_gaps(volume, max_gap=1)
    closed = close_binary_mask(volume, radius=1)

    assert int(dilated.sum()) > original_count, "bridge_gaps is expected to dilate"
    assert dilated[7, 10, 10], "bridge_gaps should expand outward by one voxel"

    assert int(closed.sum()) == original_count, (
        "closing changed the size of an isolated object "
        f"({original_count} -> {int(closed.sum())} voxels)"
    )
    assert not closed[7, 10, 10], "closing must not expand boundaries"


def test_close_binary_mask_bridges_a_gap_between_thick_structures():
    """The behaviour the mask path relies on: a closing does still bridge a narrow gap.

    This holds for thick structures. It does not hold for a 1-voxel-thick skeleton, where
    the erosion step removes the bridge again - which is why bridge_gaps is kept for that
    case rather than being replaced outright.
    """
    volume = np.zeros((21, 21, 21), dtype=bool)
    volume[8:13, 8:13, 4:10] = True    # two thick blocks separated by a 1-voxel gap at x=10
    volume[8:13, 8:13, 11:17] = True

    closed = close_binary_mask(volume, radius=1)

    assert closed[10, 10, 10], "the gap between two thick structures was not bridged"


def test_close_binary_mask_is_a_noop_for_non_positive_radius():
    volume = np.zeros((10, 10, 10), dtype=bool)
    volume[5, 5, 5] = True
    assert np.array_equal(close_binary_mask(volume, radius=0), volume)


def test_skeletonize_3d(small_binary_3d):
    out = skeletonize_3d(small_binary_3d)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool


def test_preprocess_skeleton_for_graph(small_binary_3d):
    out = preprocess_skeleton_for_graph(small_binary_3d, min_branch_length=2)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool


# --- Greyscale denoising at capillary scale (#98 Phase A) ---------------------------------

def test_median_filtering_the_probability_field_destroys_capillaries():
    """A 3x3x3 median spans 5.6 um; a 6 um capillary is 3.2 voxels across.

    Any filter whose support is comparable to the structure width deletes the structure
    rather than cleaning it up, and this one ran on the float probability field *before*
    thresholding, so it destroyed the gradients hysteresis depends on as well as the signal.

    The filter was there to suppress isolated speckle. It does that - but a connected
    component size filter applied *after* thresholding does the same job without moving a
    single boundary, which is where speckle removal belongs. Measured on the fixture below:

        approach                              foreground  components  recall
        truth (clean probability)                    900           3   100.0%
        threshold only                              1644         714   100.0%
        median 3 -> threshold                        181           3    20.1%
        threshold -> 50-voxel size filter            913           3   100.0%
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    from ImageLynx.preprocessing import median_filter_image, hysteresis_threshold

    rng = np.random.default_rng(0)
    shape = (60, 60, 60)
    _, yy, xx = np.ogrid[:60, :60, :60]

    # Three parallel 6 um capillaries with a soft probability boundary, 12 voxels apart.
    prob = np.zeros(shape, dtype=np.float32)
    for cy in (18, 30, 42):
        r = np.sqrt((yy - cy) ** 2 + (xx - 30) ** 2)
        prob = np.maximum(prob, 1.0 / (1.0 + np.exp((r - 1.6) * 2.2))).astype(np.float32)

    truth = hysteresis_threshold(prob, low=0.65, high=0.75)
    speckle = (rng.random(shape) < 0.004) * rng.uniform(0.7, 1.0, shape)
    noisy = np.clip(prob + speckle.astype(np.float32), 0, 1).astype(np.float32)

    def recall(binary):
        return float((binary & truth).sum()) / float(truth.sum())

    plain = hysteresis_threshold(noisy, low=0.65, high=0.75)
    medianed = hysteresis_threshold(median_filter_image(noisy, size=3), low=0.65, high=0.75)
    sized = remove_small_objects(plain, min_size=50, connectivity=3)

    # Both suppress the speckle equally well.
    assert label(plain, connectivity=3).max() > 500
    assert label(medianed, connectivity=3).max() == 3
    assert label(sized, connectivity=3).max() == 3

    # Only one of them keeps the vessels.
    assert recall(medianed) < 0.25, "median filtering is meant to be shown deleting the anatomy"
    assert recall(sized) == pytest.approx(1.0)

    # And it thins what survives, which propagates to resistance as r^-4.
    edt_kw = dict(sampling=(1.866, 1.866, 1.866))
    r90 = lambda b: float(np.percentile(distance_transform_edt(b, **edt_kw)[b], 90))
    assert r90(medianed) < r90(truth)
    assert r90(sized) == pytest.approx(r90(truth))


def test_pipeline_does_not_median_filter_the_probability_field():
    """The pipeline shipped median_filter_size = 3, i.e. exactly the filter above."""
    C = pytest.importorskip("carotid_image_to_model")
    pre = C.PreprocessingConfig()
    assert pre.median_filter_size == 0
    # The two morphological radii were already 0 for the same capillary-scale reason.
    assert pre.morphological_opening_radius == 0
    assert pre.morphological_closing_radius == 0
    assert pre.probability_smoothing_sigma == 0.0
