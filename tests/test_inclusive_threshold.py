"""Every CB vessel cut keeps the probability level that sits on its threshold (open item 17).

The Ilastik export is quantised to hundredths and every CB threshold is a whole number of
hundredths, so a strict `>` dropped a whole level: `p > 0.90` was `p >= 0.91`. The selector's
plain cut and the pipeline's hysteresis (flood and seed) are now inclusive. The hysteresis is
otherwise skimage's: face connectivity, low clipped to high.
"""
import numpy as np
import pytest
from skimage.filters import apply_hysteresis_threshold

from ImageLynx.preprocessing import at_or_above, hysteresis_threshold


def _quantised(shape=(24, 24, 24), seed=0):
    rng = np.random.default_rng(seed)
    return (np.round(rng.random(shape) * 100.0).astype(np.float32) / np.float32(100.0))


@pytest.mark.parametrize("t", [0.30, 0.50, 0.85, 0.90, 0.93, 0.95, 0.99, 1.0])
def test_a_float32_level_equal_to_the_threshold_is_kept(t):
    prob = _quantised()
    on_level = prob == np.float32(t)
    assert on_level.any()
    assert at_or_above(prob, t)[on_level].all()
    assert np.array_equal(at_or_above(prob, t), (prob > np.float32(t)) | on_level)


def test_at_or_above_works_on_float64_and_integer_arrays():
    assert at_or_above(np.array([0.89, 0.9, 0.91]), 0.9).tolist() == [False, True, True]
    assert at_or_above(np.array([1, 2, 3]), 2).tolist() == [False, True, True]


def test_on_a_field_without_ties_it_is_skimage_exactly():
    """Only the comparison changed: same labelling, same connectivity."""
    rng = np.random.default_rng(1)
    prob = rng.random((32, 32, 32))           # continuous, so nothing sits on a bound
    for low, high in [(0.3, 0.7), (0.5, 0.9), (0.62, 0.95)]:
        ours = hysteresis_threshold(prob, low=low, high=high)
        theirs = apply_hysteresis_threshold(prob, low, high)
        assert np.array_equal(ours, theirs), (low, high)


def test_a_voxel_exactly_at_high_seeds_and_one_exactly_at_low_floods():
    prob = np.zeros((1, 1, 5), dtype=np.float32)
    prob[0, 0, :] = [0.0, 0.9, 0.95, 0.9, 0.0]
    mask = hysteresis_threshold(prob, low=0.9, high=0.95)
    assert mask[0, 0].tolist() == [False, True, True, True, False]
    # skimage's strict cut kept nothing: 0.95 is not above 0.95, so there was no seed
    assert not apply_hysteresis_threshold(prob, 0.9, 0.95).any()


def test_connectivity_is_face_only_as_in_skimage():
    prob = np.zeros((1, 3, 3), dtype=np.float32)
    prob[0, 0, 0] = 1.0          # seed
    prob[0, 1, 1] = 0.9          # diagonal neighbour only
    mask = hysteresis_threshold(prob, low=0.9, high=0.95)
    assert mask[0, 0, 0] and not mask[0, 1, 1]


def test_a_flood_threshold_above_the_seed_is_clipped_to_it():
    prob = _quantised(seed=2)
    assert np.array_equal(hysteresis_threshold(prob, low=0.97, high=0.95),
                          hysteresis_threshold(prob, low=0.95, high=0.95))


def test_the_quantised_band_keeps_the_levels_on_both_bounds():
    prob = _quantised(seed=3)
    mask = hysteresis_threshold(prob, low=0.90, high=0.95)
    assert mask[prob == np.float32(0.90)].any()
    # every kept voxel is at or above the flood threshold
    assert (prob[mask] >= np.float32(0.90)).all()
