"""Open item 20: the Tier 2 and Tier 3 march divides by edge flow in um^3/s.

``c -= flux / q`` has flux in mmol/L um^3/s, so q has to be in um^3/s. The march read edge
``flow_abs`` raw, in the flow solve's mmHg um^3 / cP, so q was 1.33e5 times too small and the
answer did not depend on flow. These tests put a known flow through a two-cell chain and check
the blood-side drop into the second cell against a hand calculation.

The cells are made nearly independent (diffusivity ~0) and the sink zero-order (large
k_reduce), so at steady state each cell's wall flux equals M V. Blood entering cell 2 has lost
exactly M V / q of content, which is the quantity the unit error got wrong.
"""
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import brentq

from ImageLynx.haemodynamics.perfusion import (
    calculate_blood_co2_content,
    calculate_blood_oxygen_content,
    calculate_ph_from_pco2,
    map_vessels_to_grid,
    PerfusionGrid,
    solve_coupled_1d3d_perfusion,
    solve_multi_species_perfusion,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S

# The solver's own solubilities (perfusion.py, solve_multi_species_perfusion).
_ALPHA_O2 = 1.34e-3
_ALPHA_CO2 = 0.03

_V_CELL = 1000.0
_AREA = 100.0
_H = 0.45


class _TwoCellGrid:
    n_cells = 2
    cell_volume = _V_CELL
    dims = (2, 1, 1)
    res = (10.0, 10.0, 10.0)


@dataclass
class _Config:
    #: Zero-order sink. M V / (P A alpha_O2) = 40 mmHg of wall drop in Tier 3.
    M_max: float = 40.0 * 1.0 * _AREA * _ALPHA_O2 / _V_CELL
    k_reduce: float = 1000.0
    permeability_o2_cm_s: float = 1.0e-4
    permeability_co2_cm_s: float = 1.0e-4
    #: Effectively zero, so the two cells exchange only through the blood.
    sigma_diff: float = 1e-30
    sigma_diff_co2: float = 1e-30
    po2_arterial_mmHg: float = 100.0
    pco2_arterial: float = 40.0
    systemic_hematocrit: float = _H
    respiratory_quotient: float = 0.82
    hco3_tissue: float = 24.0
    picard_max_iterations: int = 200
    picard_tolerance: float = 1e-10


def _chain(q_um3_s, flow_abs="solver units"):
    """One edge through two cells. Cell 0 is upstream; edge_to_cells keeps insertion order."""
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    attrs = dict(flow_signed=1.0, hematocrit=_H, length=20.0)
    if flow_abs == "solver units":
        attrs["flow_abs"] = q_um3_s / POISEUILLE_FLOW_TO_UM3_PER_S
    G.add_edge(0, 1, key=0, **attrs)
    cells = {i: [{'edge': (0, 1, 0), 'flow': q_um3_s, 'hematocrit': _H, 'length': 10.0,
                  'surface_area': _AREA}] for i in (0, 1)}
    return _TwoCellGrid(), G, cells


def _tier3_expected(q, c):
    """Tissue PO2 and PCO2 in both cells, stepping the march by hand in the solver's order."""
    p_o2 = c.permeability_o2_cm_s * 1e4
    p_co2 = c.permeability_co2_cm_s * 1e4
    mv_o2 = c.M_max * _V_CELL
    mv_co2 = mv_o2 * c.respiratory_quotient

    po2_b1, pco2_b1 = c.po2_arterial_mmHg, c.pco2_arterial
    po2_t1 = po2_b1 - mv_o2 / (p_o2 * _AREA * _ALPHA_O2)
    pco2_t1 = pco2_b1 + mv_co2 / (p_co2 * _AREA * _ALPHA_CO2)
    ph_t1 = calculate_ph_from_pco2(pco2_t1, c.hco3_tissue)

    # Cell 1's wall fluxes are M V out for O2 and M V RQ in for CO2.
    c_o2 = calculate_blood_oxygen_content(po2_b1, _H, pco2_b1, 7.4) - mv_o2 / q
    c_co2 = calculate_blood_co2_content(pco2_b1, _H, po2_b1) + mv_co2 / q
    po2_b2 = brentq(lambda p: calculate_blood_oxygen_content(p, _H, pco2_b1, ph_t1) - c_o2, 0.0, 150.0)
    pco2_b2 = brentq(lambda p: calculate_blood_co2_content(p, _H, po2_b2) - c_co2, 0.0, 150.0)

    po2_t2 = po2_b2 - mv_o2 / (p_o2 * _AREA * _ALPHA_O2)
    pco2_t2 = pco2_b2 + mv_co2 / (p_co2 * _AREA * _ALPHA_CO2)
    return np.array([po2_t1, po2_t2]), np.array([pco2_t1, pco2_t2])


def _tier2_expected(q, c):
    """Tier 2's wall flux is P A dPO2, without solubility (open item 22)."""
    p_o2 = c.permeability_o2_cm_s * 1e4
    mv = c.M_max * _V_CELL
    po2_t1 = c.po2_arterial_mmHg - mv / (p_o2 * _AREA)
    c2 = calculate_blood_oxygen_content(c.po2_arterial_mmHg, _H) - mv / q
    po2_b2 = brentq(lambda p: calculate_blood_oxygen_content(p, _H) - c2, 0.0, 150.0)
    return np.array([po2_t1, po2_b2 - mv / (p_o2 * _AREA)])


def _run_tier3(q, config=_Config()):
    grid, G, cells = _chain(q)
    return solve_multi_species_perfusion(grid, G, [0], cells, config)


def _run_tier2(q, config=_Config()):
    grid, G, cells = _chain(q)
    return solve_coupled_1d3d_perfusion(grid, G, [0], cells, config)


@pytest.mark.parametrize("q", [100.0, 1000.0])
def test_tier3_blood_side_drop_matches_the_hand_calculation(q):
    po2, pco2, _ = _run_tier3(q)
    want_po2, want_pco2 = _tier3_expected(q, _Config())
    np.testing.assert_allclose(po2, want_po2, atol=1e-3)
    np.testing.assert_allclose(pco2, want_pco2, atol=1e-3)
    # The chain only tests the march if the blood-side loss moves cell 2 measurably. Compared
    # with infinite flow, not with cell 1: the Bohr shift at cell 1's acidic pH can put cell 2
    # above cell 1.
    assert _tier3_expected(1e12, _Config())[0][1] - want_po2[1] > 0.05


# Tier 2 has no alpha in its wall flux (open item 22), so its wall term is ~750x Tier 3's and
# the explicit march overshoots below ~1e2 um^3/s (as item 21 does for Tier 3). These flows are
# above that.
@pytest.mark.parametrize("q", [300.0, 1000.0])
def test_tier2_blood_side_drop_matches_the_hand_calculation(q):
    po2 = _run_tier2(q)
    want = _tier2_expected(q, _Config())
    np.testing.assert_allclose(po2, want, atol=1e-3)
    assert _tier2_expected(1e12, _Config())[1] - want[1] > 0.5


def test_tier3_depends_on_flow():
    """Before the fix the answer was the same at any flow."""
    low = _run_tier3(1e2)[0][1]
    high = _run_tier3(1e4)[0][1]
    assert low < high - 5.0


def test_tier2_depends_on_flow():
    low = _run_tier2(3e2)[1]
    high = _run_tier2(1e5)[1]
    assert low < high - 2.0


@pytest.mark.parametrize("solver", [solve_multi_species_perfusion, solve_coupled_1d3d_perfusion])
def test_a_flowing_edge_without_flow_abs_is_refused(solver):
    grid, G, cells = _chain(1e3, flow_abs=None)
    with pytest.raises(ValueError, match="no 'flow_abs'"):
        solver(grid, G, [0], cells, _Config())


@pytest.mark.parametrize("solver", [solve_multi_species_perfusion, solve_coupled_1d3d_perfusion])
def test_mismatched_conversion_factors_are_refused(solver):
    """A grid mapped with flow left in solver units, solved with the default factor."""
    pts = [np.array([t, 15.0, 15.0]) for t in np.linspace(5.0, 25.0, 11)]
    G = nx.MultiGraph()
    G.add_node(0, pos=pts[0])
    G.add_node(1, pos=pts[-1])
    G.add_edge(0, 1, key=0, length=20.0, flow_abs=50.0, flow_signed=50.0,
               assigned_diameter_um=8.0, hematocrit=_H, voxels=pts)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    cells = map_vessels_to_grid(G, grid, flow_to_um3_per_s=1.0)
    with pytest.raises(ValueError, match="different conversion factors"):
        solver(grid, G, [0], cells, _Config())
    # The same factor on both sides is accepted.
    solver(grid, G, [0], cells, replace(_Config(), picard_max_iterations=2), flow_to_um3_per_s=1.0)
