"""Open item 21: Tier 3's per-cell exchange step is implicit.

The step took each cell's wall flux at the blood pressure on entry and subtracted flux / q from
the content. Once P A alpha exceeded q dC/dP (the measured wall permeability, capillary flow,
the flat top of the O2 curve) one step went past blood-tissue equilibrium, the Picard loop
stalled, and tissue PCO2 came out below arterial. The flux is now taken at the outlet pressure,
solved for by ``_implicit_cell_outlet_pressure``.
"""
import logging
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse.linalg

from ImageLynx.haemodynamics.perfusion import (
    _implicit_cell_outlet_pressure,
    calculate_blood_co2_content,
    calculate_blood_oxygen_content,
    solve_multi_species_perfusion,
)
from test_perfusion_march_flow_units import _ALPHA_O2, _AREA, _chain, _Config, _H, _tier3_expected


def _o2(p):
    return calculate_blood_oxygen_content(p, _H, 40.0, 7.4)


def _co2(p):
    return calculate_blood_co2_content(p, _H, 60.0)


# (content function, inlet pressure, tissue pressure): O2 leaves the blood, CO2 enters it.
_CASES = {"O2": (_o2, 95.0, 30.0), "CO2": (_co2, 40.0, 46.0)}


@pytest.mark.parametrize("species", _CASES)
@pytest.mark.parametrize("g", [1e-6, 1e-3, 1e-1, 1e1, 1e3, 1e6])
def test_outlet_solves_the_balance_and_lies_between_tissue_and_inlet(species, g):
    content, p_in, p_t = _CASES[species]
    c_in = content(p_in)
    p_out = _implicit_cell_outlet_pressure(content, c_in, g, p_t, species)
    # brentq stops on a pressure tolerance, so the content residual scales with g.
    assert content(p_out) + g * (p_out - p_t) == pytest.approx(c_in, rel=1e-9, abs=g * 1e-11)
    assert min(p_t, p_in) - 1e-9 <= p_out <= max(p_t, p_in) + 1e-9


@pytest.mark.parametrize("species", _CASES)
def test_weak_wall_keeps_the_inlet_and_strong_wall_reaches_the_tissue(species):
    content, p_in, p_t = _CASES[species]
    c_in = content(p_in)
    assert _implicit_cell_outlet_pressure(content, c_in, 1e-12, p_t, species) == pytest.approx(p_in, abs=1e-6)
    assert _implicit_cell_outlet_pressure(content, c_in, 1e9, p_t, species) == pytest.approx(p_t, abs=1e-6)


def test_blood_loses_exactly_the_wall_flux():
    q = 1e3
    k = 9.1e-2 * 1e4 * _AREA * _ALPHA_O2
    content, p_in, p_t = _CASES["O2"]
    c_in = content(p_in)
    p_out = _implicit_cell_outlet_pressure(content, c_in, k / q, p_t, "O2")
    assert q * (c_in - content(p_out)) == pytest.approx(k * (p_out - p_t), rel=1e-9)


def test_a_step_the_explicit_update_overshoots_stays_above_the_tissue():
    """At the measured permeability and capillary flow the old step went past equilibrium."""
    q = 1e3
    g = 9.1e-2 * 1e4 * _AREA * _ALPHA_O2 / q
    content, p_in, p_t = _CASES["O2"]
    c_in = content(p_in)
    # The old explicit update lands below tissue-equilibrium content.
    assert c_in - g * (p_in - p_t) < content(p_t)
    p_out = _implicit_cell_outlet_pressure(content, c_in, g, p_t, "O2")
    assert p_t < p_out < p_in


def test_no_bracket_raises_instead_of_keeping_an_old_value():
    def capped(p):
        return min(_o2(p), 1.0)

    with pytest.raises(ValueError, match="Implicit O2 step found no root"):
        _implicit_cell_outlet_pressure(capped, 50.0, 1e-6, 0.0, "O2")


@pytest.mark.parametrize("q", [1e2, 1e3])
def test_tier3_reaches_the_hand_calculated_answer_at_the_measured_permeability(q, caplog, monkeypatch):
    """The old step never settled here; tissue PCO2 came out below arterial.

    The Picard loop is slow at this permeability (thousands of iterations at 1e2 um^3/s), so the
    tolerance is tight and the cap high: this checks where the loop ends up, not how fast. The
    tissue CG is tightened too. At its hard-coded rtol 1e-5 (open item 6), warm-started from the
    last field, it returns that field unchanged once a Picard step is smaller than its
    tolerance, and the loop stops 0.03-0.25 mmHg short of the answer here.
    """
    cg = scipy.sparse.linalg.cg

    def tight_cg(A, b, **kwargs):
        return cg(A, b, **{**kwargs, "rtol": 1e-13, "maxiter": 5000})

    monkeypatch.setattr(scipy.sparse.linalg, "cg", tight_cg)
    config = replace(_Config(), permeability_o2_cm_s=9.1e-2, permeability_co2_cm_s=9.1e-2,
                     picard_tolerance=1e-10, picard_max_iterations=5000)
    grid, G, cells = _chain(q)
    with caplog.at_level(logging.WARNING, logger="ImageLynx.haemodynamics.perfusion"):
        po2, pco2, _ = solve_multi_species_perfusion(grid, G, [0], cells, config)
    assert not [r for r in caplog.records if "max_iter" in r.getMessage()]
    want_po2, want_pco2 = _tier3_expected(q, config)
    np.testing.assert_allclose(po2, want_po2, atol=1e-3)
    np.testing.assert_allclose(pco2, want_pco2, atol=1e-3)
    # Tissue consumes O2 and produces CO2, so it can't sit on the wrong side of arterial.
    assert np.all(po2 <= config.po2_arterial_mmHg)
    assert np.all(pco2 >= config.pco2_arterial)
