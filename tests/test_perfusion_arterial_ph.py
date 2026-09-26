"""Open item 25: Tier 3 arterial blood enters at the Henderson-Hasselbalch pH.

Tissue pH is Henderson-Hasselbalch on tissue PCO2 and ``hco3_tissue``, and inside each cell the
blood's O2 is read at the tissue pH. Arterial blood was seeded at a literal pH 7.4, while the same
formula gives 7.401 at PCO2 40 / HCO3 24, so blood changed pH at the first cell with no exchange
behind it. Through Bohr and Haldane that left tissue PCO2 below arterial (7e-4 mmHg on WKY-A).

With no metabolism, nothing is exchanged at steady state, so every cell must sit at the arterial
pressures exactly.
"""
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import solve_multi_species_perfusion
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S

_V_CELL = 1000.0
_AREA = 100.0
_H = 0.45
_N = 3


class _ChainGrid:
    n_cells = _N
    cell_volume = _V_CELL
    dims = (_N, 1, 1)
    res = (10.0, 10.0, 10.0)


@dataclass
class _Config:
    M_max: float = 0.0
    k_reduce: float = 0.1
    respiratory_quotient: float = 0.82
    permeability_o2_cm_s: float = 9.1e-2
    permeability_co2_cm_s: float = 9.1e-2
    sigma_diff: float = 1.5e-9
    sigma_diff_co2: float = 1.6e-9
    po2_arterial_mmHg: float = 100.0
    pco2_arterial: float = 40.0
    hco3_tissue: float = 24.0
    picard_max_iterations: int = 200
    picard_tolerance: float = 1e-12


def _solve(config, q_um3_s=1e3):
    """One edge through a chain of cells, at capillary flow."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, flow_signed=1.0, flow_abs=q_um3_s / POISEUILLE_FLOW_TO_UM3_PER_S,
               hematocrit=_H, length=10.0 * _N)
    cells = {i: [{'edge': (0, 1, 0), 'flow': q_um3_s, 'hematocrit': _H, 'length': 10.0,
                  'surface_area': _AREA}] for i in range(_N)}
    po2, pco2, _, info = solve_multi_species_perfusion(_ChainGrid(), G, [0], cells, config,
                                                       return_info=True)
    assert info["converged"]
    return po2, pco2


@pytest.mark.parametrize("pco2_art, hco3", [(40.0, 24.0), (45.0, 20.0)])
def test_without_metabolism_tissue_sits_at_the_arterial_pressures(pco2_art, hco3):
    config = _Config(pco2_arterial=pco2_art, hco3_tissue=hco3)
    po2, pco2 = _solve(config)
    np.testing.assert_allclose(po2, config.po2_arterial_mmHg, rtol=0, atol=1e-8)
    np.testing.assert_allclose(pco2, config.pco2_arterial, rtol=0, atol=1e-8)


def test_with_metabolism_tissue_pco2_never_falls_below_arterial():
    config = replace(_Config(), M_max=0.005)
    _, pco2 = _solve(config)
    assert np.all(pco2 >= config.pco2_arterial)
