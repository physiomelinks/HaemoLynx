"""Open item 29: Tier 1's washout uses each cell's own haematocrit, the same as its source.

The source was evaluated at each edge's haematocrit and the washout at ``systemic_hematocrit``.
A cell fed by an edge above 0.45 then received more O2 at arterial PO2 than it could wash out at
any PO2 short of hundreds of mmHg, carried by dissolved O2 alone. On WKY-A that phantom source
was 55 times the tissue's whole demand, and with solubility in the diffusion (item 22) the field
went hyperoxic. Content is affine in H, so washing out at the cell's flow-weighted H is exact.
"""
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import brentq

from ImageLynx.haemodynamics.perfusion import (
    calculate_blood_oxygen_content,
    cell_discharge_hematocrit,
    solve_perfusion_steady_state,
)


class _OneCell:
    n_cells = 1
    cell_volume = 1000.0


@dataclass
class _Config:
    M_max: float = 0.0
    k_reduce: float = 1000.0
    po2_arterial_mmHg: float = 100.0
    systemic_hematocrit: float = 0.45


def _one_cell(vessels, config):
    """One well-mixed cell, no diffusion: s and q built as build_adr_matrix builds them."""
    q = np.array([sum(v["flow"] * v["length_fraction"] for v in vessels)])
    s = np.array([sum(v["flow"] * v["length_fraction"]
                      * calculate_blood_oxygen_content(config.po2_arterial_mmHg, v["hematocrit"])
                      for v in vessels)])
    h = cell_discharge_hematocrit({0: vessels}, 1)
    return solve_perfusion_steady_state(_OneCell(), sp.csr_matrix([[0.0]]), q, s, config,
                                        cell_hematocrit=h, return_info=True)


@pytest.mark.parametrize("h_edge", [0.1, 0.45, 0.6, 0.8])
def test_no_consumption_leaves_tissue_at_arterial_po2_whatever_the_edge_haematocrit(h_edge):
    """With systemic washout, H 0.6 gave hundreds of mmHg here and H 0.1 far below arterial."""
    vessels = [{"flow": 1.0e3, "hematocrit": h_edge, "length_fraction": 1.0}]
    po2, info = _one_cell(vessels, _Config())
    assert info["converged"]
    # Within the solver's stop test: 1e-5 of the largest PO2.
    assert po2[0] == pytest.approx(100.0, abs=1e-3)


@pytest.mark.parametrize("h_edge", [0.2, 0.6])
def test_fick_balance_at_the_edge_haematocrit(h_edge):
    """C_venous = C_art - M V / q, inverted on the edge's own content curve."""
    q, config = 2.0e3, _Config(M_max=0.05)
    vessels = [{"flow": q, "hematocrit": h_edge, "length_fraction": 1.0}]
    c_venous = calculate_blood_oxygen_content(100.0, h_edge) - config.M_max * _OneCell.cell_volume / q
    want = brentq(lambda p: calculate_blood_oxygen_content(p, h_edge) - c_venous, 0.0, 100.0,
                  xtol=1e-12)
    po2, info = _one_cell(vessels, config)
    assert info["converged"]
    assert po2[0] == pytest.approx(want, abs=1e-3)


def test_cell_haematocrit_is_flow_and_length_weighted():
    vessels = {0: [{"flow": 1.0, "hematocrit": 0.2, "length_fraction": 1.0},
                   {"flow": 3.0, "hematocrit": 0.6, "length_fraction": 0.5}],
               2: [{"flow": 5.0, "hematocrit": 0.4, "length_fraction": 1.0}]}
    h = cell_discharge_hematocrit(vessels, 3)
    np.testing.assert_allclose(h, [(0.2 + 1.5 * 0.6) / 2.5, 0.0, 0.4])


def test_washout_at_the_mixed_haematocrit_equals_the_sum_over_vessels():
    """Content is affine in H, so one mixed H reproduces the per-vessel washout exactly."""
    vessels = [{"flow": 1.0, "hematocrit": 0.15, "length_fraction": 1.0},
               {"flow": 4.0, "hematocrit": 0.7, "length_fraction": 0.25}]
    h_mix = cell_discharge_hematocrit({0: vessels}, 1)[0]
    q = sum(v["flow"] * v["length_fraction"] for v in vessels)
    for p in (5.0, 26.0, 60.0):
        per_vessel = sum(v["flow"] * v["length_fraction"]
                         * calculate_blood_oxygen_content(p, v["hematocrit"]) for v in vessels)
        assert q * calculate_blood_oxygen_content(p, h_mix) == pytest.approx(per_vessel, rel=1e-12)


def test_systemic_haematocrit_no_longer_moves_tier1():
    vessels = [{"flow": 2.0e3, "hematocrit": 0.3, "length_fraction": 1.0}]
    a, _ = _one_cell(vessels, _Config(M_max=0.05, systemic_hematocrit=0.45))
    b, _ = _one_cell(vessels, _Config(M_max=0.05, systemic_hematocrit=0.30))
    np.testing.assert_array_equal(a, b)


def test_a_perfused_cell_without_a_haematocrit_is_refused():
    q = np.array([1.0e3, 0.0])
    s = q * calculate_blood_oxygen_content(100.0, 0.45)
    A = sp.csr_matrix((2, 2))

    class _TwoCells:
        n_cells = 2
        cell_volume = 1000.0

    with pytest.raises(ValueError, match="haematocrit"):
        solve_perfusion_steady_state(_TwoCells(), A, q, s, _Config(),
                                     cell_hematocrit=np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="haematocrit"):
        solve_perfusion_steady_state(_TwoCells(), A, q, s, _Config(),
                                     cell_hematocrit=np.array([0.45]))
