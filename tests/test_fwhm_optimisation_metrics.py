"""Unit tests for haemolynx.optimisation.fwhm_metrics -- hand-built graphs
and summaries, no real FWHM measurement."""
from __future__ import annotations

import networkx as nx
import pytest

from haemolynx.optimisation import fwhm_metrics as met


def _graph_with_edges(edges: list[dict]) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for i, data in enumerate(edges):
        G.add_edge(f"u{i}", f"v{i}", **data)
    return G


def test_fwhm_measurement_quality_all_measured_perfect_fit():
    G = _graph_with_edges(
        [
            {
                "fwhm_status": "measured",
                "fwhm_diameter_samples_um": [4.0, 4.0],
                "fwhm_diameter_r2_samples": [1.0, 1.0],
                "fwhm_profile_lines_phys": [
                    [[0.0, 0.0, -6.0], [0.0, 0.0, 6.0]],
                    [[0.0, 0.0, -6.0], [0.0, 0.0, 6.0]],
                ],
            },
        ]
    )
    quality = met.fwhm_measurement_quality(G, {"edges_measured": 1}, min_total_extent_multiplier=3.0)
    assert quality.n_edges_total == 1
    assert quality.n_edges_measured == 1
    assert quality.measured_fraction == pytest.approx(1.0)
    assert quality.mean_fit_r2 == pytest.approx(1.0)
    assert quality.median_fit_r2 == pytest.approx(1.0)
    # achieved extent = 12um, target = 3.0 * 4.0 = 12um -> ratio 1.0
    assert quality.mean_achieved_extent_ratio == pytest.approx(1.0)
    assert quality.median_diameter_cv == pytest.approx(0.0)


def test_fwhm_measurement_quality_partial_measurement():
    G = _graph_with_edges(
        [
            {"fwhm_status": "measured", "fwhm_diameter_samples_um": [4.0], "fwhm_diameter_r2_samples": [0.9]},
            {"fwhm_status": "failed:fwhm_failed"},
        ]
    )
    quality = met.fwhm_measurement_quality(G, {"edges_measured": 1}, min_total_extent_multiplier=3.0)
    assert quality.n_edges_total == 2
    assert quality.measured_fraction == pytest.approx(0.5)


def test_fwhm_measurement_quality_no_edges_is_zero_not_a_crash():
    G = nx.MultiGraph()
    quality = met.fwhm_measurement_quality(G, {"edges_measured": 0}, min_total_extent_multiplier=3.0)
    assert quality.n_edges_total == 0
    assert quality.measured_fraction == 0.0
    assert quality.mean_fit_r2 == 0.0
    assert quality.mean_achieved_extent_ratio == 0.0
    assert quality.median_diameter_cv == 0.0


def test_fwhm_measurement_quality_diameter_cv_from_inconsistent_samples():
    G = _graph_with_edges(
        [
            {
                "fwhm_status": "measured",
                "fwhm_diameter_samples_um": [2.0, 4.0, 6.0],
                "fwhm_diameter_r2_samples": [0.9, 0.9, 0.9],
            }
        ]
    )
    quality = met.fwhm_measurement_quality(G, {"edges_measured": 1}, min_total_extent_multiplier=3.0)
    assert quality.median_diameter_cv > 0.0


def test_fwhm_measurement_quality_no_profile_lines_gives_zero_extent_ratio():
    """A caller that ran without store_profile_debug=True must not crash --
    just get an uninformative-but-safe 0.0 ratio."""
    G = _graph_with_edges(
        [{"fwhm_status": "measured", "fwhm_diameter_samples_um": [4.0], "fwhm_diameter_r2_samples": [0.9]}]
    )
    quality = met.fwhm_measurement_quality(G, {"edges_measured": 1}, min_total_extent_multiplier=3.0)
    assert quality.mean_achieved_extent_ratio == 0.0


def test_score_prefers_higher_measured_fraction_and_fit_quality():
    good = met.FwhmMeasurementQuality(
        n_edges_total=10, n_edges_measured=9, measured_fraction=0.9,
        mean_fit_r2=0.95, median_fit_r2=0.95, mean_achieved_extent_ratio=1.0,
        median_diameter_cv=0.0,
    )
    bad = met.FwhmMeasurementQuality(
        n_edges_total=10, n_edges_measured=2, measured_fraction=0.2,
        mean_fit_r2=0.5, median_fit_r2=0.5, mean_achieved_extent_ratio=0.3,
        median_diameter_cv=0.5,
    )
    assert good.score < bad.score


def test_rejection_gates_score_multiplicative():
    high = met.FwhmMeasurementQuality(
        n_edges_total=10, n_edges_measured=10, measured_fraction=1.0,
        mean_fit_r2=0.9, median_fit_r2=0.9, mean_achieved_extent_ratio=1.0,
        median_diameter_cv=0.0,
    )
    zero_r2 = met.FwhmMeasurementQuality(
        n_edges_total=10, n_edges_measured=10, measured_fraction=1.0,
        mean_fit_r2=0.0, median_fit_r2=0.0, mean_achieved_extent_ratio=1.0,
        median_diameter_cv=0.0,
    )
    assert met.rejection_gates_score(high) == pytest.approx(-0.9)
    assert met.rejection_gates_score(zero_r2) == pytest.approx(0.0)
    assert met.rejection_gates_score(high) < met.rejection_gates_score(zero_r2)


def test_regression_penalty_zero_when_current_meets_or_beats_baseline():
    assert met.regression_penalty(current=0.9, baseline=0.9, tolerance=0.05) == 0.0
    assert met.regression_penalty(current=0.95, baseline=0.9, tolerance=0.05) == 0.0


def test_regression_penalty_within_tolerance_is_free():
    assert met.regression_penalty(current=0.87, baseline=0.9, tolerance=0.05) == 0.0


def test_regression_penalty_beyond_tolerance_is_guarded():
    assert met.regression_penalty(current=0.5, baseline=0.9, tolerance=0.05) == met._GUARD_PENALTY


def _quality(**fields) -> met.FwhmMeasurementQuality:
    base = dict(
        n_edges_total=10, n_edges_measured=10, measured_fraction=1.0,
        mean_fit_r2=0.9, median_fit_r2=0.9, mean_achieved_extent_ratio=1.0,
        median_diameter_cv=0.0,
    )
    base.update(fields)
    return met.FwhmMeasurementQuality(**base)


def test_usable_fraction_leaves_out_widths_the_run_sets_aside():
    G = _graph_with_edges(
        [
            {"fwhm_status": "measured", "fwhm_diameter_r2_samples": [0.9]},
            {"fwhm_status": "measured", "fwhm_diameter_r2_samples": [0.9],
             "fwhm_demoted": "speck_width"},
            {"fwhm_status": "measured", "fwhm_diameter_r2_samples": [0.9],
             "fwhm_demoted": "edt_disagreement", "fwhm_low_confidence_vs_edt": True},
            {"fwhm_status": "failed:fwhm_failed"},
        ]
    )
    quality = met.fwhm_measurement_quality(
        G, {"edges_measured": 3}, min_total_extent_multiplier=3.0,
        decoy_report={"false_positive_rate": 0.25},
    )
    assert quality.measured_fraction == pytest.approx(0.75)
    assert quality.n_edges_demoted == 2
    assert quality.usable_fraction == pytest.approx(0.25)
    assert quality.decoy_false_positive_rate == pytest.approx(0.25)
    assert quality.edt_disagreement_fraction == pytest.approx(1 / 3)


def test_without_checks_usable_is_measured_and_the_score_is_unchanged():
    quality = _quality(n_edges_measured=8, measured_fraction=0.8)
    assert quality.usable_fraction == pytest.approx(0.8)
    assert quality.score == pytest.approx(-0.8 - 0.5 * 0.9 - 0.25 * 1.0)


def test_a_setting_that_measures_more_by_fitting_specks_does_not_win():
    """Regression: the score rewarded measuring more edges with no accuracy
    term, so a gate loose enough to fit specks won. Here the loose setting
    measures every edge but reads half its decoys and has two widths in the
    speck range; the strict one measures eight, all usable, and no decoy."""
    loose = _quality(n_edges_measured=10, measured_fraction=1.0, n_edges_demoted=2,
                     decoy_false_positive_rate=0.5)
    strict = _quality(n_edges_measured=8, measured_fraction=0.8)

    assert strict.score < loose.score
    assert met.rejection_gates_score(strict) < met.rejection_gates_score(loose)


def test_rejection_gates_score_counts_usable_widths_and_charges_decoys():
    quality = _quality(n_edges_demoted=5, decoy_false_positive_rate=0.2)
    assert met.rejection_gates_score(quality) == pytest.approx(-(0.5 * 0.9) + 0.2)
