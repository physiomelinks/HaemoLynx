"""Red cells in a solved vessel (``haemodynamics/red_cells.py``): the Fahraeus
tube haematocrit of Pries et al. 1990, and the red-cell velocity, flux and
spacing that red-cell conservation gives from it, against hand-computed values."""
from __future__ import annotations

import math

import pytest

from haemolynx.haemodynamics.red_cells import (
    DEFAULT_RED_CELL_VOLUME_FL,
    FAHRAEUS_FIT_MIN_DIAMETER_UM,
    RED_CELL_EDGE_ATTRIBUTES,
    fahraeus_ratio,
    red_cell_flux_per_s,
    red_cell_linear_density_per_mm,
    red_cell_velocity_um_s,
    tube_haematocrit,
    vessel_discharge_haematocrit,
)


def _published_ratio(h, d):
    """Pries, Secomb, Gaehtgens and Gross 1990, written out here, not imported."""
    return h + (1.0 - h) * (1.0 + 1.7 * math.exp(-0.415 * d) - 0.6 * math.exp(-0.011 * d))


# --- the Fahraeus relation ------------------------------------------------------------


@pytest.mark.parametrize("h, d", [(0.45, 10.0), (0.2, 5.0), (0.3, 30.0), (0.0, 8.0), (0.6, 4.0)])
def test_the_ratio_is_pries_1990s_fit(h, d):
    assert fahraeus_ratio(h, d) == pytest.approx(_published_ratio(h, d))


def test_two_hand_checked_points():
    # 1 + 1.7 e^-4.15 - 0.6 e^-0.11 = 0.4893; 0.45 + 0.55 x 0.4893 = 0.7191.
    assert fahraeus_ratio(0.45, 10.0) == pytest.approx(0.7191, abs=5e-4)
    # 1 + 1.7 e^-2.075 - 0.6 e^-0.055 = 0.6456; 0.2 + 0.8 x 0.6456 = 0.7165.
    assert fahraeus_ratio(0.2, 5.0) == pytest.approx(0.7165, abs=5e-4)


def test_the_fahraeus_effect_is_strongest_around_ten_microns_and_gone_in_wide_vessels():
    ratios = {d: fahraeus_ratio(0.45, d) for d in (4.0, 10.0, 30.0, 100.0, 1000.0)}
    assert ratios[10.0] < ratios[4.0]
    assert ratios[10.0] < ratios[30.0] < ratios[100.0] < ratios[1000.0]
    assert ratios[1000.0] == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 < r <= 1.0 for r in ratios.values())


def test_below_the_fit_the_ratio_is_held_at_one():
    """The fit crosses 1 at 2.58 um whatever the haematocrit (the crossing
    is where its diameter term alone reaches 1); held at 1 below it."""
    assert _published_ratio(0.45, 2.0) > 1.0
    assert _published_ratio(0.0, 2.0) > 1.0
    assert fahraeus_ratio(0.45, 2.0) == 1.0
    assert fahraeus_ratio(0.0, 1.0) == 1.0
    assert fahraeus_ratio(0.45, FAHRAEUS_FIT_MIN_DIAMETER_UM) == pytest.approx(1.0, abs=2e-3)
    assert fahraeus_ratio(0.45, 3.0) < 1.0
    assert _published_ratio(0.45, 2.7) < 1.0 < _published_ratio(0.45, 2.5)


def test_tube_haematocrit_is_the_discharge_haematocrit_times_the_ratio():
    assert tube_haematocrit(0.45, 10.0) == pytest.approx(0.45 * _published_ratio(0.45, 10.0))
    assert tube_haematocrit(0.0, 10.0) == 0.0
    assert tube_haematocrit(0.45, 10.0) < 0.45


@pytest.mark.parametrize("h", [-0.1, 1.0, 1.5, math.nan])
def test_a_haematocrit_outside_zero_to_one_is_refused(h):
    with pytest.raises(ValueError, match="discharge_haematocrit"):
        fahraeus_ratio(h, 10.0)
    with pytest.raises(ValueError, match="discharge_haematocrit"):
        red_cell_flux_per_s(1e-13, h, 50.0)


@pytest.mark.parametrize("d", [0.0, -5.0, math.nan])
def test_a_diameter_that_is_not_positive_is_refused(d):
    with pytest.raises(ValueError, match="diameter_um"):
        tube_haematocrit(0.45, d)
    with pytest.raises(ValueError, match="diameter_um"):
        red_cell_linear_density_per_mm(d, 0.45, 50.0)


@pytest.mark.parametrize("v", [0.0, -50.0, math.nan])
def test_a_red_cell_volume_that_is_not_positive_is_refused(v):
    with pytest.raises(ValueError, match="red_cell_volume_fl"):
        red_cell_flux_per_s(1e-13, 0.45, v)
    with pytest.raises(ValueError, match="red_cell_volume_fl"):
        red_cell_linear_density_per_mm(6.0, 0.45, v)


# --- red-cell conservation -------------------------------------------------------------


def test_flux_is_flow_times_discharge_haematocrit_over_the_cell_volume():
    # 100 pL/s at a haematocrit of 0.45 is 45 pL/s of cells: 900 cells of 50 fL.
    assert red_cell_flux_per_s(1e-13, 0.45, 50.0) == pytest.approx(900.0)
    # Half the cell volume, twice the cells; the flow's sign is not a direction here.
    assert red_cell_flux_per_s(1e-13, 0.45, 25.0) == pytest.approx(1800.0)
    assert red_cell_flux_per_s(-1e-13, 0.45, 50.0) == pytest.approx(900.0)
    assert red_cell_flux_per_s(1e-13, 0.0, 50.0) == 0.0


def test_red_cell_velocity_is_the_mean_velocity_over_the_ratio():
    v_mean = 1000.0  # um/s
    v_rbc = red_cell_velocity_um_s(v_mean, 0.45, 10.0)
    assert v_rbc == pytest.approx(v_mean / _published_ratio(0.45, 10.0))
    assert v_rbc > v_mean
    # Conservation: the cells' velocity at the tube haematocrit carries the
    # discharge haematocrit at the mean velocity.
    assert v_rbc * tube_haematocrit(0.45, 10.0) == pytest.approx(v_mean * 0.45)
    # In a vessel the fit does not cover, cells move with the plasma.
    assert red_cell_velocity_um_s(v_mean, 0.45, 2.0) == v_mean


def test_linear_density_is_the_tube_haematocrit_filling_the_cross_section():
    # A 6 um vessel: pi 3^2 = 28.27 um^2, 28,274 fL per mm of vessel.
    h_t = tube_haematocrit(0.3, 6.0)
    expected = h_t * math.pi * 9.0 * 1000.0 / 50.0
    assert red_cell_linear_density_per_mm(6.0, 0.3, 50.0) == pytest.approx(expected)
    assert red_cell_linear_density_per_mm(6.0, 0.0, 50.0) == 0.0


def test_flux_equals_linear_density_times_red_cell_velocity():
    """The three measures agree with each other on any vessel: cells per mm
    times mm per second is cells per second."""
    for flow, h, d, v in [(1e-13, 0.45, 6.0, 50.0), (3e-12, 0.2, 20.0, 90.0), (2e-14, 0.6, 4.0, 45.0)]:
        v_mean_um_s = flow / (math.pi * (0.5 * d * 1e-6) ** 2) * 1e6
        v_rbc_mm_s = red_cell_velocity_um_s(v_mean_um_s, h, d) * 1e-3
        assert red_cell_linear_density_per_mm(d, h, v) * v_rbc_mm_s == pytest.approx(
            red_cell_flux_per_s(flow, h, v), rel=1e-12
        )


def test_the_default_cell_volume_is_a_mouse_red_cell():
    assert DEFAULT_RED_CELL_VOLUME_FL == 50.0
    assert len(RED_CELL_EDGE_ATTRIBUTES) == 4


# --- which haematocrit a vessel is counted at ------------------------------------------


def test_a_vessels_own_discharge_haematocrit_comes_before_the_runs():
    assert vessel_discharge_haematocrit({"discharge_haematocrit": 0.3}, 0.45) == (0.3, True)
    assert vessel_discharge_haematocrit({}, 0.45) == (0.45, False)
    assert vessel_discharge_haematocrit({"discharge_haematocrit": 0.0}, 0.45) == (0.0, True)


@pytest.mark.parametrize("bad", [None, "x", math.nan, 1.0, -0.2])
def test_a_haematocrit_that_is_no_fraction_falls_through(bad):
    assert vessel_discharge_haematocrit({"discharge_haematocrit": bad}, 0.45) == (0.45, False)
    assert vessel_discharge_haematocrit({}, bad) == (None, False)
