"""Open item 19 (a, b): the Tier 3 CO2 curve is McHardy's, in mmol/L, as its source defines it.

The code used to take McHardy's whole-blood curve, which is in vol%, call it mmol/L, multiply it
by haematocrit and add dissolved CO2 a second time. Its Haldane term had no source and was about
a third of the measured effect. These tests pin the curve to McHardy (1967) as quoted by
Mallat & Vallet (2021), and the Haldane size to Loeppky et al. (1983).
"""
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import calculate_blood_co2_content

_VOL_PCT_PER_MMOL_L = 2.226  # 1 mmol CO2 = 22.26 mL STPD

# PO2 at which the fixed-P50 Hill curve gives McHardy's reference saturation of 95%.
_PO2_AT_S95 = 26.0 * 19.0 ** (1.0 / 2.7)


def _mchardy_reference_mmol_L(pco2):
    """McHardy at his reference state (Hb 15 g/dL, SaO2 95%): only the power-law term is left."""
    return 11.02 * pco2 ** 0.396 / _VOL_PCT_PER_MMOL_L


def test_the_reference_state_matches_mchardy_in_mmol_per_litre():
    """47.5 vol% at PCO2 40 is 21.3 mmol/L. The old code gave 24.4 (+14%)."""
    c = calculate_blood_co2_content(40.0, 0.45, _PO2_AT_S95)
    np.testing.assert_allclose(c, 21.3, rtol=0.01)


def test_dissolved_co2_is_not_added_a_second_time():
    """McHardy is total content. At H 0.45 and S 95% nothing else may be added to it."""
    for pco2 in (20.0, 40.0, 60.0):
        np.testing.assert_allclose(calculate_blood_co2_content(pco2, 0.45, _PO2_AT_S95),
                                   _mchardy_reference_mmol_L(pco2), rtol=1e-12)


def test_the_slope_at_pco2_40_is_mchardys_not_the_old_one():
    """About 0.21 mmol/L/mmHg. The old code gave 0.28 (+38%)."""
    slope = (calculate_blood_co2_content(41.0, 0.45, 100.0)
             - calculate_blood_co2_content(39.0, 0.45, 100.0)) / 2.0
    assert 0.19 < slope < 0.23


def test_the_haldane_effect_is_the_measured_size():
    """Full desaturation adds about 2.5 mmol/L (Loeppky) to 2.7 (McHardy). The old code gave 0.9."""
    haldane = calculate_blood_co2_content(40.0, 0.45, 1e-3) - calculate_blood_co2_content(40.0, 0.45, 100.0)
    assert 2.0 < haldane < 3.5


def test_haematocrit_enters_through_mchardys_hb_term():
    """More haemoglobin carries more CO2; H 0.45 is Hb 15 g/dL, where the Hb term is zero.

    At S 95% the Haldane term is zero for every H (it now scales with Hb), leaving the Hb term.
    """
    contents = [calculate_blood_co2_content(40.0, h, _PO2_AT_S95) for h in (0.0, 0.2, 0.45, 0.8)]
    assert np.all(np.diff(contents) > 0)
    hb_term_at_h0 = contents[2] - contents[0]
    np.testing.assert_allclose(hb_term_at_h0, 15.0 * 0.015 * 40.0 / _VOL_PCT_PER_MMOL_L, rtol=1e-12)


@pytest.mark.parametrize("hematocrit", [0.0, 0.2, 0.45, 0.8])
@pytest.mark.parametrize("po2", [1.0, 20.0, 100.0])
def test_content_rises_with_pco2_over_the_tissue_range(hematocrit, po2):
    """Tier 3 inverts this curve with brentq, so it must be monotonic. At H 0 it peaks near
    135 mmHg, above any tissue PCO2."""
    pco2 = np.linspace(0.01, 120.0, 2000)
    c = np.array([calculate_blood_co2_content(p, hematocrit, po2) for p in pco2])
    assert np.all(np.diff(c) > 0)


@pytest.mark.parametrize("pco2", [0.0, -5.0])
def test_non_positive_pco2_gives_zero(pco2):
    assert calculate_blood_co2_content(pco2, 0.45, 100.0) == 0.0


@pytest.mark.parametrize("hematocrit, share", [(0.0, 0.0), (0.225, 0.5), (0.45, 1.0), (0.9, 2.0)])
def test_the_haldane_effect_scales_with_haemoglobin(hematocrit, share):
    """Deoxygenation raises CO2 capacity through haemoglobin, so plasma has no Haldane effect.

    McHardy's saturation term was fitted at Hb 15 g/dL and is scaled by Hb / 15. Unscaled, a
    plasma-skimmed vessel losing O2 gained CO2 capacity and its PCO2 fell below arterial.
    """
    haldane = (calculate_blood_co2_content(40.0, hematocrit, 1e-3)
               - calculate_blood_co2_content(40.0, hematocrit, 100.0))
    full = (calculate_blood_co2_content(40.0, 0.45, 1e-3)
            - calculate_blood_co2_content(40.0, 0.45, 100.0))
    np.testing.assert_allclose(haldane, share * full, rtol=1e-12, atol=1e-15)
