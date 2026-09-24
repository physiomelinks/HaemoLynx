"""Open items 3 and 4: arterial PO2 and systemic haematocrit come from the config.

Both used to be written out inside the solver bodies as well as declared in the config, so
changing the config value changed nothing in some solvers. Each test below changes the
config value and checks the solver's answer moves with it, which a hard-coded copy would not.
"""
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import pytest

from ImageLynx import cb_settings
from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    build_adr_matrix,
    calculate_blood_oxygen_content,
    map_vessels_to_grid,
    solve_coupled_1d3d_perfusion,
    solve_multi_species_perfusion,
    solve_perfusion_steady_state,
)


@dataclass
class _Config:
    sigma_diff: float = 1.5e-9
    M_max: float = 0.05
    k_reduce: float = 0.1
    po2_arterial_mmHg: float = 100.0
    systemic_hematocrit: float = 0.45
    permeability_o2_cm_s: float = 1.0e-4


def _one_edge(hematocrit=0.45):
    G = nx.MultiGraph()
    pts = [np.array([t, 15.0, 15.0]) for t in np.linspace(5.0, 25.0, 11)]
    G.add_node(0, pos=pts[0])
    G.add_node(1, pos=pts[-1])
    G.add_edge(0, 1, key=0, length=20.0, flow_abs=50.0, flow_signed=50.0,
               assigned_diameter_um=8.0, hematocrit=hematocrit, voxels=pts)
    return G


def _tier1(config, hematocrit=0.45):
    G = _one_edge(hematocrit)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    A, q, s = build_adr_matrix(grid, map_vessels_to_grid(G, grid), config)
    return s, solve_perfusion_steady_state(grid, A, q, s, config)


def test_the_tier1_source_uses_the_configured_arterial_po2():
    source_100, _ = _tier1(_Config())
    source_60, _ = _tier1(_Config(po2_arterial_mmHg=60.0))
    ratio = calculate_blood_oxygen_content(60.0, 0.45) / calculate_blood_oxygen_content(100.0, 0.45)
    np.testing.assert_allclose(source_60, source_100 * ratio, rtol=1e-12)


def test_the_tier1_washout_uses_the_configured_systemic_haematocrit():
    """Open item 4. The source is fixed by the edge haematocrit, so only the washout can move."""
    source_a, po2_a = _tier1(_Config(systemic_hematocrit=0.45))
    source_b, po2_b = _tier1(_Config(systemic_hematocrit=0.30))
    np.testing.assert_array_equal(source_a, source_b)
    assert not np.allclose(po2_a, po2_b)


def _coupled(config):
    G = _one_edge()
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    return solve_coupled_1d3d_perfusion(grid, G, [0], map_vessels_to_grid(G, grid), config)


def test_the_coupled_solver_uses_the_configured_arterial_po2():
    assert not np.allclose(_coupled(_Config()), _coupled(_Config(po2_arterial_mmHg=60.0)))


@dataclass
class _MultiConfig(_Config):
    #: Low enough that the tissue is not anoxic at either arterial PO2 the test compares.
    M_max: float = 1e-4
    sigma_diff_co2: float = 3.0e-8
    permeability_co2_cm_s: float = 2.0e-3
    respiratory_quotient: float = 0.82
    pco2_arterial: float = 40.0
    hco3_tissue: float = 24.0
    picard_max_iterations: int = 50
    picard_tolerance: float = 1e-4


def _multi(config):
    G = _one_edge()
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    return solve_multi_species_perfusion(grid, G, [0], map_vessels_to_grid(G, grid), config)[0]


def test_the_multi_species_solver_uses_the_configured_arterial_po2():
    assert not np.allclose(_multi(_MultiConfig()), _multi(_MultiConfig(po2_arterial_mmHg=60.0)))


@pytest.mark.parametrize("field", ["po2_arterial_mmHg", "systemic_hematocrit"])
def test_a_config_without_the_field_is_refused_not_defaulted(field):
    """The multi-species solver read both through getattr with a default."""

    class Partial:
        pass

    config = Partial()
    for name, value in vars(_MultiConfig()).items():
        if name != field:
            setattr(config, name, value)
    with pytest.raises(AttributeError, match=field):
        _multi(config)


def test_the_h2_settings_match_the_values_that_were_hard_coded():
    """Moving the values into config must not move a published H2 number."""
    settings = cb_settings.PerfusionSettings()
    assert settings.po2_arterial_mmHg == 100.0
    assert settings.systemic_hematocrit == 0.45
    source_settings, po2_settings = _tier1(replace(settings, M_max=0.05))
    source_default, po2_default = _tier1(_Config(sigma_diff=settings.sigma_diff,
                                                 k_reduce=settings.k_reduce))
    np.testing.assert_array_equal(source_settings, source_default)
    np.testing.assert_array_equal(po2_settings, po2_default)
