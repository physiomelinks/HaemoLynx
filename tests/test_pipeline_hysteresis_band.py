"""The pipeline's hysteresis band is the frozen one, and nothing fills it in silently (open item 1).

Every H1 run passed ``--hysteresis-low 0.90`` and got a 0.95 seed, while ``PreprocessingConfig``
declared 0.65 / 0.75 and the mask step fell back to 0.2 / 0.4 for a missing key. The config now
reads the band from ``cb_settings``, a low given alone always takes its seed from the settings
offset, and the mask step raises on a missing bound.
"""
from dataclasses import asdict

import numpy as np
import pytest

from ImageLynx import cb_settings

C = pytest.importorskip("carotid_image_to_model")


# --- CLI overrides -------------------------------------------------------------------------

def test_no_override_keeps_the_frozen_band():
    config = C.PreprocessingConfig()
    C._apply_hysteresis_overrides(config)
    assert config.hysteresis_threshold_low == cb_settings.HYSTERESIS_LOW
    assert config.hysteresis_threshold_high == cb_settings.HYSTERESIS_HIGH


@pytest.mark.parametrize("low, expected_high", [
    (0.90, 0.95),   # what cb_h1_batch.py --stage run passes
    (0.85, 0.90),   # below the configured seed: the old rule left it at 0.95
    (0.70, 0.75),
    (0.97, 0.999),  # capped below 1
])
def test_a_low_alone_takes_its_seed_from_the_settings_offset(low, expected_high):
    config = C.PreprocessingConfig()
    C._apply_hysteresis_overrides(config, low=low)
    assert config.hysteresis_threshold_low == low
    assert config.hysteresis_threshold_high == pytest.approx(expected_high)


def test_an_explicit_high_wins_over_the_offset():
    config = C.PreprocessingConfig()
    C._apply_hysteresis_overrides(config, low=0.80, high=0.97)
    assert (config.hysteresis_threshold_low, config.hysteresis_threshold_high) == (0.80, 0.97)


def test_a_high_alone_keeps_the_configured_low():
    config = C.PreprocessingConfig()
    C._apply_hysteresis_overrides(config, high=0.97)
    assert config.hysteresis_threshold_low == cb_settings.HYSTERESIS_LOW
    assert config.hysteresis_threshold_high == 0.97


@pytest.mark.parametrize("low, high", [(0.90, 0.90), (0.90, 0.85), (None, 0.80)])
def test_a_seed_at_or_below_the_flood_threshold_raises(low, high):
    with pytest.raises(ValueError, match="must be above low"):
        C._apply_hysteresis_overrides(C.PreprocessingConfig(), low=low, high=high)


# --- Mask step -----------------------------------------------------------------------------

def _probability_field():
    """Two 1-voxel tubes along x between the frozen bounds: one seeded, one not.

    A voxel just below the flood threshold touches the seeded tube. The frozen band keeps only
    the seeded tube; the old 0.65 / 0.75 default would have grown into that voxel as well.
    Values follow cb_settings, on the hundredths grid the Ilastik export uses.
    """
    low, high = cb_settings.HYSTERESIS_LOW, cb_settings.HYSTERESIS_HIGH
    between = np.float32(round(low + 0.01, 2))
    assert low < between < high
    prob = np.zeros((12, 12, 12), dtype=np.float32)
    prob[6, 6, 2:10] = between
    prob[6, 6, 5] = 1.0                                    # the seed
    prob[6, 7, 5] = np.float32(round(low - 0.05, 2))       # touches the seeded tube
    prob[3, 3, 2:10] = between
    expected = np.zeros(prob.shape, dtype=bool)
    expected[6, 6, 2:10] = True
    return prob, expected


def test_the_mask_step_thresholds_at_the_frozen_band():
    prob, expected = _probability_field()
    config = asdict(C.PreprocessingConfig())
    _, binary = C._apply_preprocessing_filters(prob, None, config)
    np.testing.assert_array_equal(binary, expected)


@pytest.mark.parametrize("missing", ["hysteresis_threshold_low", "hysteresis_threshold_high"])
def test_the_mask_step_refuses_a_missing_bound(missing):
    prob, _ = _probability_field()
    config = asdict(C.PreprocessingConfig())
    del config[missing]
    with pytest.raises(KeyError, match=missing):
        C._apply_preprocessing_filters(prob, None, config)
