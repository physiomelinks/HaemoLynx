"""Open item 22: Tier 1 diffusion carries O2 solubility, and its Newton solve reaches the fixed point.

Every term of Tier 1's balance is an O2 flux in mmol/L um^3/s except diffusion, which was
sigma * face / spacing * dPO2 in mmHg um^3/s: about 1/alpha = 750 times too strong. The slab
below has an analytic answer that only the alpha version reproduces.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import (
    ALPHA_O2_MMOL_PER_L_MMHG,
    PerfusionGrid,
    build_adr_matrix,
    calculate_blood_oxygen_content,
    cell_discharge_hematocrit,
    map_vessels_to_grid,
    solve_perfusion_steady_state,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S

_D = 1.5e-9          # m^2/s, the configured sigma_diff
_M = 0.05            # mmol/L/s
_P0 = 50.0           # mmHg held at both ends of the slab
_H_STEP = 4.0        # um
_N = 11


class _SlabGrid:
    """A line of cells along z. Only what build_adr_matrix and the solver read."""
    dims = (_N, 1, 1)
    res = (_H_STEP, _H_STEP, _H_STEP)
    n_cells = _N
    cell_volume = _H_STEP ** 3


@dataclass
class _Config:
    sigma_diff: float = _D
    M_max: float = _M
    #: Zero-order sink: M(P) = M_max everywhere the tissue is above a fraction of a mmHg.
    k_reduce: float = 1000.0
    po2_arterial_mmHg: float = _P0
    #: Plasma, so blood content is alpha P and the end cells are held by a linear term.
    systemic_hematocrit: float = 0.0
    #: Solver stop settings the tiers now read (open item 6), at Tier 1's old hard-coded values.
    picard_max_iterations: int = 50
    picard_tolerance: float = 1e-5


def _slab():
    """End cells supplied by a flow large enough to pin them at the arterial PO2."""
    q = 1e9
    ends = {i: [{"flow": q, "hematocrit": 0.0, "length_fraction": 1.0}] for i in (0, _N - 1)}
    return build_adr_matrix(_SlabGrid(), ends, _Config())


_SLAB_H = np.zeros(_N)  # plasma in both end cells


def _analytic():
    """D alpha P'' = M between two fixed ends: P = P0 - M / (2 D alpha) x (L - x)."""
    x = np.arange(_N) * _H_STEP
    return _P0 - _M / (2.0 * _D * 1e12 * ALPHA_O2_MMOL_PER_L_MMHG) * x * (x[-1] - x)


def test_slab_matches_the_krogh_profile_with_solubility():
    A, q, s = _slab()
    po2, info = solve_perfusion_steady_state(_SlabGrid(), A, q, s, _Config(),
                                             cell_hematocrit=_SLAB_H, return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(po2, _analytic(), atol=1e-3)
    # The profile has to be measurable for this to test anything: a ~5 mmHg dip at mid-slab.
    # Without alpha it was 0.0067 mmHg.
    assert _P0 - po2[_N // 2] > 4.9


def test_face_conductance_is_d_alpha():
    A, _, _ = _slab()
    expected = _D * 1e12 * ALPHA_O2_MMOL_PER_L_MMHG * _H_STEP
    assert A[0, 1] == pytest.approx(-expected, rel=1e-12)


def _capillary_network(q_um3_s=2.0e3):
    """One capillary through a small grid: the Hill curve's steep and flat parts both matter."""
    G = nx.MultiGraph()
    pts = [np.array([t, 20.0, 20.0]) for t in np.linspace(2.0, 38.0, 19)]
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([40.0, 40.0, 40.0]))
    G.add_edge(0, 1, key=0, length=36.0, flow_abs=q_um3_s / POISEUILLE_FLOW_TO_UM3_PER_S,
               flow_signed=q_um3_s / POISEUILLE_FLOW_TO_UM3_PER_S,
               assigned_diameter_um=6.0, hematocrit=0.45, voxels=pts)
    grid = PerfusionGrid(G, (4.0, 4.0, 4.0))
    cells = map_vessels_to_grid(G, grid)
    A, q, s = build_adr_matrix(grid, cells, _NetworkConfig())
    return grid, A, q, s, cell_discharge_hematocrit(cells, grid.n_cells)


@dataclass
class _NetworkConfig:
    sigma_diff: float = _D
    M_max: float = _M
    k_reduce: float = 0.1
    po2_arterial_mmHg: float = 100.0
    systemic_hematocrit: float = 0.45
    #: Solver stop settings the tiers now read (open item 6), at Tier 1's old hard-coded values.
    picard_max_iterations: int = 50
    picard_tolerance: float = 1e-5


def test_newton_reaches_the_fixed_point_on_a_capillary_grid():
    """The balance is checked here from its definition, not from the solver's own residual."""
    grid, A, q, s, h = _capillary_network()
    config = _NetworkConfig()
    po2, info = solve_perfusion_steady_state(grid, A, q, s, config, cell_hematocrit=h,
                                             return_info=True)
    assert info["converged"] and info["iterations"] < 50

    washout = np.array([q[i] * calculate_blood_oxygen_content(po2[i], h[i])
                        if q[i] > 0 else 0.0 for i in range(grid.n_cells)])
    consumption = config.M_max * (1.0 - np.exp(-config.k_reduce * po2)) * grid.cell_volume
    imbalance = A @ po2 + 1e-6 * po2 - (s - washout - consumption)
    # Scaled to mmHg by the cell's own conductance, as the solver's stop test does.
    assert np.max(np.abs(imbalance) / A.diagonal()) < 1e-3
    # The field is not trivially at a bound: tissue far from the capillary sits well below it.
    assert po2.min() < 90.0 and po2.max() > po2.min() + 5.0


def test_mass_balance_closes_at_the_fixed_point():
    """Diffusion only moves O2 between cells, so delivery - washout = consumption overall."""
    grid, A, q, s, h = _capillary_network()
    config = _NetworkConfig()
    po2 = solve_perfusion_steady_state(grid, A, q, s, config, cell_hematocrit=h)
    washout = sum(q[i] * calculate_blood_oxygen_content(po2[i], h[i])
                  for i in np.flatnonzero(q > 0))
    consumption = float((config.M_max * (1.0 - np.exp(-config.k_reduce * po2))).sum()
                        * grid.cell_volume)
    assert s.sum() - washout == pytest.approx(consumption, rel=1e-4)


def test_zero_flow_and_no_metabolism_is_already_converged():
    A, q, s = _slab()
    config = _Config(M_max=0.0)
    q0, s0 = np.zeros_like(q), np.zeros_like(s)
    po2, info = solve_perfusion_steady_state(_SlabGrid(), A, q0, s0, config,
                                             cell_hematocrit=_SLAB_H, return_info=True)
    assert info["converged"] and info["iterations"] == 0
    np.testing.assert_array_equal(po2, 0.0)
