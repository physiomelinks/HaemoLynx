"""`build_vessel_mask` must be the mask the CB pipeline builds (open item 41).

It is a copy of the recipe in `carotid_image_to_model.py` (`_apply_preprocessing_filters` then
`_preprocess_local_mask`), not a refactor of it, so this test is what stops the two drifting.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ImageLynx import cb_settings
from ImageLynx.preprocessing import build_vessel_mask


def _field(shape=(48, 40, 40), seed=3):
    """A smooth probability field quantised to hundredths, as the Ilastik export is.

    Saturated cores (p = 1.0 reaches the 0.999 seed), unseeded islands, holes and structures
    touching the z faces, so every step of the recipe has something to act on.
    """
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.standard_normal(shape), sigma=2.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return np.round(np.clip(1.4 * noise - 0.2, 0.0, 1.0), 2).astype(np.float32)


def _pipeline_mask(prob, low, high):
    from carotid_image_to_model import (
        GraphConfig, PipelineConfig, PreprocessingConfig, SkeletonConfig,
        _preprocess_local_mask,
    )

    pre = PreprocessingConfig()
    pre.hysteresis_threshold_low, pre.hysteresis_threshold_high = low, high
    _, binary = _preprocess_local_mask(
        prob, None, pre, SkeletonConfig(), GraphConfig(), PipelineConfig())
    return binary


@pytest.mark.parametrize("low", [0.90, 0.93, cb_settings.FROZEN_THRESHOLD, 0.97])
def test_the_mask_matches_the_pipeline_at_the_cb_defaults(low):
    prob = _field()
    expected = _pipeline_mask(prob, low, cb_settings.HYSTERESIS_HIGH)
    got = build_vessel_mask(prob, low, cb_settings.HYSTERESIS_HIGH)
    assert expected.any()
    np.testing.assert_array_equal(got, expected)


def test_a_seed_at_or_below_the_flood_threshold_raises():
    with pytest.raises(ValueError, match="seed"):
        build_vessel_mask(_field(), 0.95, 0.95)
