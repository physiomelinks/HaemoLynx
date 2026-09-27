"""Open item 23: Tier 3's Picard loop reaches its fixed point at capillary flow.

The loop lagged the tissue behind the blood: each iteration moved it C' / (C' + P A alpha / q)
of the way, which at the measured wall permeability meant hundreds to thousands of iterations
(the default cap is 50). Its tissue CG, at a hard-coded rtol 1e-5 and warm-started, handed back
the last field once a step was smaller than that, and the relative-change test then read zero.

Now each update is linearised through the blood's actual response and solved exactly by LU, with
guarded Anderson acceleration on top, and the loop stops on the nonlinear residual.
"""
import logging
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from carotid_image_to_model import PerfusionConfig
from ImageLynx.haemodynamics import perfusion as pf
from ImageLynx.haemodynamics.perfusion import (
    _AndersonMixer,
    _blood_response_conductance,
    _implicit_cell_outlet_pressure,
    _relative_residual,
    calculate_blood_co2_content,
    calculate_blood_oxygen_content,
    solve_multi_species_perfusion,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S
from test_perfusion_junction_state import _MEASURED, _network

_LOGGER = "ImageLynx.haemodynamics.perfusion"


def _y_network(q):
    """One 6 um feeder splitting into two branches across a 100 um box."""
    G = nx.MultiGraph()
    pos = {0: np.array([50., 50., 5.]), 1: np.array([50., 50., 45.]),
           2: np.array([20., 20., 95.]), 3: np.array([80., 80., 95.])}
    for n, p in pos.items():
        G.add_node(n, pos=p)
    for u, v, qq in [(0, 1, q), (1, 2, q / 2), (1, 3, q / 2)]:
        f = qq / POISEUILLE_FLOW_TO_UM3_PER_S
        G.add_edge(u, v, key=0, length=float(np.linalg.norm(pos[v] - pos[u])), flow_abs=f,
                   flow_signed=f, assigned_diameter_um=6.0, hematocrit=0.45,
                   voxels=[pos[u] + (pos[v] - pos[u]) * t for t in np.linspace(0, 1, 21)])
    grid = pf.PerfusionGrid(G, (10.0, 10.0, 10.0))
    return grid, G, pf.map_vessels_to_grid(G, grid)


def _solve(grid, G, starts, cells, config):
    return solve_multi_species_perfusion(grid, G, starts, cells, config, return_info=True)


def _cases():
    yield from ((f"Y {q:g}", *_y_network(q), [0], PerfusionConfig()) for q in (1e3, 1e4, 1e5))
    yield ("plasma-skimmed", *_network([(0, 1, 1e3, 0.45), (1, 2, 5e2, 0.89),
                                        (1, 3, 5e2, 0.01)]), [0], _MEASURED)
    yield ("merge", *_network([(0, 2, 4e2, 0.30), (1, 2, 6e2, 0.60), (2, 3, 1e3, 0.48)]),
           [0, 1], _MEASURED)


@pytest.mark.parametrize("name, grid, G, cells, starts, config", list(_cases()),
                         ids=[c[0] for c in _cases()])
def test_the_default_settings_converge_and_land_on_the_fixed_point(name, grid, G, cells, starts,
                                                                   config):
    """Before: 378 and 97 iterations on the Y network at 1e3 and 1e4 (plain Picard); the
    plasma-skimmed and merging cases took 75-243 with Anderson alone. The cases built on
    _MEASURED carry its 1e-4; each runs at the pipeline default here (1e-5 since open item 6)."""
    default = PerfusionConfig()
    assert default.picard_max_iterations == 50 and default.picard_tolerance == 1e-5
    config = replace(config, picard_max_iterations=default.picard_max_iterations,
                     picard_tolerance=default.picard_tolerance)
    po2, pco2, _, info = _solve(grid, G, starts, cells, config)
    assert info["converged"]
    assert info["iterations"] <= 25
    tight_po2, tight_pco2, _, tight = _solve(
        grid, G, starts, cells, replace(config, picard_tolerance=1e-12, picard_max_iterations=200))
    assert tight["converged"]
    # At 1e-4 the Y network sat within 0.023 mmHg of the fixed point; 1e-5 is tighter.
    np.testing.assert_allclose(po2, tight_po2, atol=0.05)
    np.testing.assert_allclose(pco2, tight_pco2, atol=0.05)


def test_the_tissue_solve_is_exact_and_does_not_use_cg(monkeypatch):
    import scipy.sparse.linalg

    def refuse(*args, **kwargs):
        raise AssertionError("Tier 3 called CG")

    monkeypatch.setattr(scipy.sparse.linalg, "cg", refuse)
    grid, G, cells = _y_network(1e4)
    _, _, _, info = _solve(grid, G, [0], cells, PerfusionConfig())
    assert info["converged"]


def test_a_converged_run_reports_a_residual_below_the_tolerance(caplog):
    grid, G, cells = _y_network(1e3)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _, _, _, info = _solve(grid, G, [0], cells, PerfusionConfig())
    assert info["converged"]
    assert max(info["residual_o2"], info["residual_co2"]) < PerfusionConfig().picard_tolerance
    assert not [r for r in caplog.records if "max_iter" in r.getMessage()]


def test_a_capped_run_says_it_did_not_converge(caplog):
    config = replace(PerfusionConfig(), picard_max_iterations=2)
    grid, G, cells = _y_network(1e3)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _, _, _, info = _solve(grid, G, [0], cells, config)
    assert info == {**info, "converged": False, "iterations": 2}
    assert max(info["residual_o2"], info["residual_co2"]) >= config.picard_tolerance
    assert [r for r in caplog.records if "max_iter" in r.getMessage()]
    # The default return shape is unchanged.
    assert len(solve_multi_species_perfusion(grid, G, [0], cells, config)) == 3


@pytest.mark.parametrize("species", ["O2", "CO2"])
@pytest.mark.parametrize("q", [1e1, 1e3, 1e5])
def test_the_blood_response_is_the_slope_of_the_implicit_step(species, q):
    """d(flux)/d(P_tissue) through the implicit step is -k C' / (C' + k / q)."""
    h, k = 0.45, 9.1e2 * 100.0 * (1.34e-3 if species == "O2" else 0.03)
    if species == "O2":
        content, c_in, p_t = (lambda p: calculate_blood_oxygen_content(p, h, 40.0, 7.4)), \
            calculate_blood_oxygen_content(95.0, h, 40.0, 7.4), 30.0
    else:
        content, c_in, p_t = (lambda p: calculate_blood_co2_content(p, h, 60.0)), \
            calculate_blood_co2_content(40.0, h, 60.0), 46.0

    def flux(pt):
        return k * (_implicit_cell_outlet_pressure(content, c_in, k / q, pt, species) - pt)

    d = 1e-4
    numeric = -(flux(p_t + d) - flux(p_t - d)) / (2 * d)
    p_out = _implicit_cell_outlet_pressure(content, c_in, k / q, p_t, species)
    assert _blood_response_conductance(content, p_out, k, q) == pytest.approx(numeric, rel=1e-4)
    # At high flow the blood holds its pressure and the response is the full wall conductance;
    # at low flow it follows the tissue and the response is much smaller.
    assert _blood_response_conductance(content, p_out, k, q) <= k


def test_the_residual_is_zero_at_the_solution_and_in_mmhg_relative_to_the_field():
    A = sp.diags([[-1.0] * 3, [3.0] * 4, [-1.0] * 3], [-1, 0, 1]).tocsr()
    x = np.array([10.0, 20.0, 30.0, 40.0])
    b = A @ x
    scale = np.full(4, 3.0)
    assert _relative_residual(A, x, b, scale) == pytest.approx(0.0, abs=1e-15)
    # A 1 mmol/s imbalance in one cell, scale 3 per mmHg, field max 40: (1/3)/40.
    assert _relative_residual(A, x, b + np.array([0, 1.0, 0, 0]), scale) == pytest.approx(1 / 120)


def test_anderson_speeds_up_a_slow_contraction_and_clips_at_zero():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    M = Q @ np.diag([0.99, 0.95, 0.9, 0.5, 0.2, 0.1]) @ Q.T
    fixed = rng.uniform(1.0, 2.0, size=6)
    b = (np.eye(6) - M) @ fixed

    def iterations(depth):
        mixer, x = _AndersonMixer(depth), np.zeros(6)
        for n in range(2000):
            gx = M @ x + b
            r = np.linalg.norm(gx - x)
            if r < 1e-10:
                return n
            x = mixer.update(x, gx, r)
        return 2000

    assert iterations(5) < 30 < iterations(0)
    assert np.all(_AndersonMixer(5).update(np.zeros(2), np.array([-1.0, 1.0]), 1.0) >= 0.0)


def test_anderson_drops_its_history_when_the_residual_rises():
    mixer = _AndersonMixer(5)
    mixer.update(np.zeros(2), np.ones(2), 1.0)
    mixer.update(np.ones(2), np.full(2, 1.5), 0.5)
    assert len(mixer.fs) == 2
    out = mixer.update(np.full(2, 1.5), np.full(2, 1.7), 0.9)  # rose from 0.5
    assert len(mixer.fs) == 1
    np.testing.assert_array_equal(out, np.full(2, 1.7))  # plain step
