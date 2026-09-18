"""Unit tests for haemolynx.optimisation.fwhm_candidates -- synthetic data only.

Every generator must (a) always include the schema default/current value
(except the two exclusion-zone functions' own documented safety cap), and
(b) degrade to something sane on pathological input rather than raising.
"""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.optimisation import fwhm_candidates as c


# ---------------------------------------------------------------------------
# Group 1: exclusion zones
# ---------------------------------------------------------------------------
def test_exclusion_zone_candidates_include_zero_and_default():
    lengths = np.array([5.0, 10.0, 15.0, 20.0, 40.0])
    result = c.exclusion_zone_candidates(lengths, default=10.0)
    assert 0.0 in result
    assert 10.0 in result
    assert result == sorted(result)


def test_exclusion_zone_candidates_empty_lengths_falls_back():
    result = c.exclusion_zone_candidates(np.array([]), default=10.0)
    assert result == sorted({0.0, 10.0})


def test_exclusion_zone_candidates_percentile_candidates_capped_by_edge_length():
    """A default far larger than the sampled edges' own length must not
    stop a length-scaled alternative from being offered -- the percentile
    candidates are capped at MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH of the
    median edge length, not at the (much larger) default itself."""
    from haemolynx.haemodynamics.automated import MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH

    lengths = np.full(20, 8.0)
    result = c.exclusion_zone_candidates(lengths, default=10.0)
    cap = MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH * 8.0
    non_zero_non_default = [v for v in result if v not in (0.0, 10.0)]
    assert non_zero_non_default, "must propose at least one length-scaled alternative"
    assert all(v <= cap + 1e-9 for v in non_zero_non_default)


# ---------------------------------------------------------------------------
# Group 2: extent and widening
# ---------------------------------------------------------------------------
def test_half_extent_candidates_include_default():
    diameters = np.array([2.0, 3.0, 4.0, 5.0])
    result = c.half_extent_candidates(diameters, default=6.0)
    assert 6.0 in result
    assert all(v > 0 for v in result)


def test_half_extent_candidates_empty_diameters_falls_back_to_default():
    result = c.half_extent_candidates(np.array([]), default=6.0)
    assert result == [6.0]


def test_min_total_extent_multiplier_candidates_include_default_and_bounded():
    result = c.min_total_extent_multiplier_candidates(default=3.0)
    assert 3.0 in result
    assert all(v >= 1.0 for v in result)


def test_max_transverse_widen_passes_candidates_include_default_and_nonnegative():
    result = c.max_transverse_widen_passes_candidates(default=5)
    assert 5 in result
    assert all(v >= 0 for v in result)


def test_max_transverse_widen_passes_candidates_default_zero_never_negative():
    result = c.max_transverse_widen_passes_candidates(default=0)
    assert 0 in result
    assert all(v >= 0 for v in result)


# ---------------------------------------------------------------------------
# Group 3: central-lobe clipping
# ---------------------------------------------------------------------------
def test_clip_min_drop_fraction_candidates_include_default_and_bounded():
    result = c.clip_min_drop_fraction_candidates(default=0.35)
    assert 0.35 in result
    assert all(0.0 <= v <= 1.0 for v in result)


def test_clip_re_rise_fraction_candidates_include_default_and_bounded():
    result = c.clip_re_rise_fraction_candidates(default=0.08)
    assert 0.08 in result
    assert all(0.0 <= v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# Group 4: same-edge geometry
# ---------------------------------------------------------------------------
def test_same_edge_arc_window_um_candidates_always_offers_none():
    result = c.same_edge_arc_window_um_candidates(3.0)
    assert None in result
    assert 3.0 in result
    assert result[0] is None  # None sorts first


def test_same_edge_arc_window_um_candidates_default_none_only_offers_none():
    result = c.same_edge_arc_window_um_candidates(None)
    assert result == [None]


def test_same_edge_arc_separation_candidates_um_include_default():
    distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = c.same_edge_arc_separation_candidates_um(distances, default=6.0)
    assert 6.0 in result


def test_same_edge_arc_separation_candidates_um_empty_falls_back_to_default():
    result = c.same_edge_arc_separation_candidates_um(np.array([]), default=6.0)
    assert result == [6.0]


def test_same_edge_arc_window_min_candidates_include_default():
    distances = np.array([1.0, 2.0, 3.0])
    result = c.same_edge_arc_window_min_candidates(distances, default=1.0)
    assert 1.0 in result


def test_same_edge_arc_window_multiplier_candidates_include_default_and_bounded():
    result = c.same_edge_arc_window_multiplier_candidates(default=1.0)
    assert 1.0 in result
    assert all(v >= 0.0 for v in result)


def test_nonlocal_same_edge_half_extent_factor_candidates_include_default():
    result = c.nonlocal_same_edge_half_extent_factor_candidates(default=0.45)
    assert 0.45 in result
    assert all(v >= 0.0 for v in result)


# ---------------------------------------------------------------------------
# Group 5: baseline estimation
# ---------------------------------------------------------------------------
def test_baseline_wing_fraction_candidates_include_default_and_bounded():
    result = c.baseline_wing_fraction_candidates(default=0.2)
    assert 0.2 in result
    assert all(0.0 < v < 0.5 for v in result)


def test_baseline_constraint_half_width_ptp_candidates_include_default_and_bounded():
    result = c.baseline_constraint_half_width_ptp_candidates(default=0.35)
    assert 0.35 in result
    assert all(0.0 <= v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# Group 6: diameter guess
# ---------------------------------------------------------------------------
def test_diameter_guess_edge_attribute_candidates_without_edt():
    result = c.diameter_guess_edge_attribute_candidates(has_populated_edt_diameter=False)
    assert result == [None, "diameter_um"]


def test_diameter_guess_edge_attribute_candidates_with_edt():
    """Regression: fwhm_diameter_guess_edge_attribute defaults to
    'edt_diameter_um', but that attribute is entirely unpopulated whenever
    use_edt_diameter_crosscheck is off -- offering it as a candidate in
    that case would spend a real trial on a setting that changes nothing
    (the exact 'silently inert' trap found and fixed manually this session)."""
    result = c.diameter_guess_edge_attribute_candidates(has_populated_edt_diameter=True)
    assert result == [None, "diameter_um", "edt_diameter_um"]


def test_diameter_guess_um_candidates_always_offers_none():
    result = c.diameter_guess_um_candidates(np.array([2.0, 3.0, 4.0, 5.0]))
    assert result[0] is None


def test_diameter_guess_um_candidates_empty_diameters_only_offers_none():
    result = c.diameter_guess_um_candidates(np.array([]))
    assert result == [None]


# ---------------------------------------------------------------------------
# Group 7: rejection gates
# ---------------------------------------------------------------------------
def test_max_fit_center_offset_um_candidates_include_default():
    result = c.max_fit_center_offset_um_candidates(default=1.5)
    assert 1.5 in result
    assert all(v >= 0.0 for v in result)


def test_max_fit_center_offset_fraction_candidates_include_default():
    result = c.max_fit_center_offset_fraction_candidates(default=0.3)
    assert 0.3 in result
    assert all(v >= 0.0 for v in result)


def test_min_fit_r2_candidates_include_default_and_bounded():
    result = c.min_fit_r2_candidates(default=0.85)
    assert 0.85 in result
    assert all(0.0 <= v <= 1.0 for v in result)


def test_max_plateau_shape_ratio_candidates_include_default_and_bounded():
    result = c.max_plateau_shape_ratio_candidates(default=0.85)
    assert 0.85 in result
    assert all(0.0 <= v <= 1.0 for v in result)
