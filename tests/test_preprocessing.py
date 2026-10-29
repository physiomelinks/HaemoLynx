"""Tests for preprocessing module."""
import pytest
import numpy as np

from ImageLynx.preprocessing import (
    bridge_gaps,
    close_binary_mask,
    preprocess_skeleton_for_graph,
)
from ImageLynx.cb_settings import ROI_VOXELS
from ImageLynx.specimens import SPECIMENS


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


def test_close_binary_mask_is_a_noop_for_non_positive_radius():
    volume = np.zeros((10, 10, 10), dtype=bool)
    volume[5, 5, 5] = True
    assert np.array_equal(close_binary_mask(volume, radius=0), volume)


def test_preprocess_skeleton_for_graph():
    """A stray voxel is pruned, not bridged into the skeleton.

    The input is a real skeleton (a 1-voxel line along x), the function's documented input,
    plus one voxel four voxels off it. Pruning at min_branch_length=2 drops the stray voxel
    before reconnection runs; without pruning it is bridged in (13 voxels, three outside the
    input). A square rod of even thickness would not do: skimage skeletonises it to nothing.
    """
    from skimage.measure import label

    line = np.zeros((10, 10, 10), dtype=bool)
    line[5, 5, :] = True
    volume = line.copy()
    volume[1, 5, 5] = True

    out = preprocess_skeleton_for_graph(volume, min_branch_length=2)

    assert np.array_equal(out, line)
    assert not (out & ~volume).any()
    assert label(out, connectivity=3).max() == 1


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


# --- Closing at the domain boundary (#98 Phase B) -----------------------------------------

def test_closing_does_not_delete_vessels_at_the_domain_boundary():
    """A closing is extensive: X is a subset of X closed. This one was not.

    scipy erodes against a zero border, so every voxel touching the edge of the array was
    removed. On a boundary-crossing tube the entire first and last slice disappeared - 13
    voxels to 0 - which moves the point at which a vessel terminates one slice inside the
    domain. Inlet and outlet identification is defined by vessels reaching the boundary, so
    this silently changed which nodes became inlets.
    """
    from skimage.measure import label
    from ImageLynx.preprocessing.skeleton import close_binary_mask

    mask = np.zeros((20, 20, 20), dtype=bool)
    _, yy, xx = np.ogrid[:20, :20, :20]
    mask |= np.broadcast_to(((yy - 10) ** 2 + (xx - 10) ** 2) <= 2 ** 2, mask.shape)
    assert mask[0].sum() == 13 and mask[-1].sum() == 13

    closed = close_binary_mask(mask, radius=1)

    assert closed[0].sum() == 13, "the boundary slice was eroded away"
    assert closed[-1].sum() == 13
    assert np.all(closed >= mask), "closing must never remove foreground"

    # And it must still do the job it is there for.
    gapped = mask.copy()
    gapped[9:11] = False
    assert label(gapped, connectivity=3).max() == 2
    assert label(close_binary_mask(gapped, radius=2), connectivity=3).max() == 1


def test_closing_twice_is_a_no_op():
    """Closing is idempotent for a fixed structuring element.

    The CB pipeline used to call close_binary_mask twice in succession, at closing_radius
    and then at bridge_gap_size. The second call could never do anything at equal radii, and
    at unequal ones it was a second, differently sized closing wearing the name of a bridge.
    bridge_gap_size has been removed from SkeletonConfig; this test is what stops a second
    closing parameter being reintroduced on the belief that radii accumulate. They do not:
    one call at radius 2 is a different, larger operation than two calls at radius 1.
    """
    from ImageLynx.preprocessing.skeleton import close_binary_mask

    rng = np.random.default_rng(0)
    mask = rng.random((24, 24, 24)) < 0.15

    once = close_binary_mask(mask, radius=1)
    assert np.array_equal(close_binary_mask(once, radius=1), once)
    assert not np.array_equal(close_binary_mask(mask, radius=2), once)


def test_absolute_roi_gives_matched_volumes_across_differently_sized_specimens():
    """A percentage ROI is not a matched sample when the specimens differ in size.

    SHR volumes average 89 Mvoxel against 63 for WKY, so the same percentage crops a larger
    absolute box from SHR - reintroducing the extent confound that any raw count is already
    exposed to. A comparison needs identical absolute dimensions.
    """
    from ImageLynx.preprocessing import crop_roi

    wky_c = np.zeros((435, 315, 255), dtype=np.float32)
    shr_b = np.zeros((495, 483, 399), dtype=np.float32)

    # The percentage path is what it is: same fraction, very different absolute size.
    assert crop_roi(wky_c, 0.45).size * 2.5 < crop_roi(shr_b, 0.45).size

    # The absolute path gives the same box from both.
    box = (160, 160, 160)
    assert crop_roi(wky_c, size_zyx=box).shape == box
    assert crop_roi(shr_b, size_zyx=box).shape == box


def test_an_absolute_roi_larger_than_the_volume_is_clipped_not_padded():
    """Silently padding would invent tissue; the caller has to see the smaller box."""
    from ImageLynx.preprocessing import crop_roi

    volume = np.zeros((30, 40, 50), dtype=np.float32)
    assert crop_roi(volume, size_zyx=(100, 100, 100)).shape == (30, 40, 50)


def test_absolute_roi_handles_a_trailing_channel_axis():
    """Probability volumes arrive as (z, y, x, c) and must not be cropped on the class axis."""
    from ImageLynx.preprocessing import crop_roi

    volume = np.zeros((60, 60, 60, 2), dtype=np.float32)
    assert crop_roi(volume, size_zyx=(20, 20, 20)).shape == (20, 20, 20, 2)


#: Every extent the offsets are checked at: the odd and even ones next to the ROI size, and
#: every axis extent of the six specimens (which include 435, 315, 495 and 456, the extents
#: the one-voxel-low box was first found on).
OFFSET_EXTENTS = sorted(
    {160, 161, 200, 201, 377, 520, 521}
    | {extent for specimen in SPECIMENS for extent in specimen.shape_zyx}
)


@pytest.mark.parametrize("extent", OFFSET_EXTENTS)
def test_offsets_reproduce_the_placement_bounds_exactly(extent):
    """crop_roi from centre_to_offsets lands on RoiPlacement.bounds.

    Truncating the offset and the start separately put the box one voxel low on odd extents
    with the centre above the midpoint - always low, never high, so it did not average out.
    Every legal centre is checked, on each of the three axes in turn.
    """
    from ImageLynx.preprocessing import crop_roi
    from ImageLynx.roi_placement import RoiPlacement, centre_to_offsets

    size = ROI_VOXELS[0]
    for axis in range(3):
        # A full-size box on all three axes, with only the tested axis longer than the box.
        shape = [size, size, size]
        shape[axis] = extent
        line = np.arange(extent, dtype=np.int32).reshape(
            [extent if a == axis else 1 for a in range(3)])
        volume = np.broadcast_to(line, shape)
        for centre in range(size // 2, extent - (size - size // 2) + 1):
            centre_zyx = [size // 2] * 3
            centre_zyx[axis] = centre
            placement = RoiPlacement("X", tuple(centre_zyx), (size,) * 3,
                                     centre_to_offsets(centre_zyx, volume.shape), None, "test")
            expected = volume[placement.bounds]
            offsets = dict(zip(("offset_z", "offset_y", "offset_x"), placement.offsets_zyx))
            got = crop_roi(volume, size_zyx=(size,) * 3, **offsets)
            assert got.shape == expected.shape == (size,) * 3, (axis, centre)
            assert got.flat[0] == expected.flat[0], f"axis {axis}, centre {centre}"


def test_a_zero_offset_on_an_odd_extent_centres_on_the_lower_middle_voxel():
    """Halves round down, matching clamp_centre's extent // 2.

    On an even extent the same rule gives the exactly centred box.
    """
    from ImageLynx.preprocessing import crop_roi

    volume = np.arange(21, dtype=np.int32)[:, None, None] * np.ones((1, 4, 4), dtype=np.int32)
    got = crop_roi(volume, size_zyx=(5, 4, 4))
    assert list(got[:, 0, 0]) == [8, 9, 10, 11, 12]

    even = np.arange(60 * 60 * 60, dtype=np.float32).reshape(60, 60, 60)
    centred = crop_roi(even, size_zyx=(20, 20, 20))
    assert centred.shape == (20, 20, 20)
    assert centred[0, 0, 0] == even[20, 20, 20]
