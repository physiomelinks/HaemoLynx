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
    sigma_diff_co2: float = 1.6e-9
    permeability_co2_cm_s: float = 1.0e-4
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


@pytest.mark.parametrize("field", [
    # systemic_hematocrit is not here: Tier 3 read it only for the arterial blood it gave a
    # node with no inflow, which now raises instead (open item 24).
    "po2_arterial_mmHg",
    # Loose end A2: the rest of Tier 3's fields were getattr fallbacks too.
    "M_max", "k_reduce", "respiratory_quotient", "hco3_tissue", "permeability_o2_cm_s",
    "permeability_co2_cm_s", "pco2_arterial", "picard_max_iterations", "picard_tolerance",
    "sigma_diff_co2",
])
def test_a_config_without_the_field_is_refused_not_defaulted(field):
    """The multi-species solver read every one of these through getattr with a default.

    The sharpest case was M_max, whose fallback of 0.05 is 10x PerfusionConfig's 0.005.
    """

    class Partial:
        pass

    config = Partial()
    for name, value in vars(_MultiConfig()).items():
        if name != field:
            setattr(config, name, value)
    with pytest.raises(AttributeError, match=field):
        _multi(config)


def test_the_pipeline_reads_the_tier_switch_without_a_default():
    """Which perfusion tier runs was chosen through getattr(..., False) as well."""
    from pathlib import Path

    source = (Path(__file__).parent.parent / "examples" / "carotid_image_to_model.py").read_text()
    assert "getattr(perf_config, 'use_multi_species_model'" not in source
    assert "perf_config.use_multi_species_model" in source


def test_the_pipeline_reads_the_barrier_switch_without_a_default():
    """The Tier 2 switch, just below the Tier 3 one, was read through getattr(..., False) too."""
    from pathlib import Path

    source = (Path(__file__).parent.parent / "examples" / "carotid_image_to_model.py").read_text()
    assert "getattr(perf_config, 'use_endothelial_barrier_model'" not in source
    assert "perf_config.use_endothelial_barrier_model" in source


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


# Open item 18. Tier 3 multiplies each gas's diffusivity and wall permeability by its solubility,
# and alpha_CO2 is already ~22x alpha_O2. A second ~20x in D_CO2 and P_CO2 made CO2 move ~450x
# faster than O2; the measured Krogh ratio (D * alpha) in rat muscle is ~21 (Kawashiro 1975).
# These are the solver's own solubilities (perfusion.py, solve_multi_species_perfusion).
_ALPHA_O2 = 1.34e-3
_ALPHA_CO2 = 0.03


def _pipeline_perfusion_config():
    C = pytest.importorskip("carotid_image_to_model")
    return C.PerfusionConfig()


def test_the_default_co2_to_o2_krogh_ratio_is_near_the_measured_21():
    config = _pipeline_perfusion_config()
    ratio = (config.sigma_diff_co2 * _ALPHA_CO2) / (config.sigma_diff * _ALPHA_O2)
    assert 15.0 < ratio < 30.0, ratio


def test_the_default_wall_permeability_is_one_value_for_both_gases():
    """Dash & Bassingthwaighte (2006) use one capillary PS for O2 and CO2; alpha does the rest."""
    config = _pipeline_perfusion_config()
    assert config.permeability_co2_cm_s == config.permeability_o2_cm_s


# Open item 19 (c). The wall permeability was 1e-4 cm/s with no source, 10^2-10^4x below a D/delta
# estimate, and Tier 3 ran wall-limited. Liu, Eskin & Hellums (1994) measured the O2 mass-transfer
# coefficient of a human umbilical vein endothelial monolayer: k = 1.22e-10 mol/(cm^2 s mmHg).
# The solver's wall flux is P * A * alpha_O2 * dPO2, so P = k / alpha_O2 reproduces k.
_LIU_1994_K_MOL_PER_CM2_S_MMHG = 1.22e-10
_ALPHA_O2_MOL_PER_CM3_MMHG = _ALPHA_O2 * 1e-6  # mmol/L = 1e-6 mol/cm^3


def test_the_default_o2_wall_permeability_is_liu_1994_measured_value():
    config = _pipeline_perfusion_config()
    expected = _LIU_1994_K_MOL_PER_CM2_S_MMHG / _ALPHA_O2_MOL_PER_CM3_MMHG
    np.testing.assert_allclose(config.permeability_o2_cm_s, expected, rtol=0.01)


@pytest.mark.parametrize("name", ["config_WKY_normotensive.yaml", "config_SHR_hypertensive.yaml"])
def test_the_example_yamls_carry_the_same_wall_permeabilities(name):
    """Both YAMLs restate the wall permeabilities; keep them in step with PerfusionConfig."""
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    config = _pipeline_perfusion_config()

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for child in node.values():
                found = find(child, key)
                if found is not None:
                    return found
        return None

    loaded = yaml.safe_load((Path(__file__).parent.parent / "examples" / name).read_text())
    for key in ("permeability_o2_cm_s", "permeability_co2_cm_s"):
        assert float(find(loaded, key)) == getattr(config, key), key


def test_the_multi_species_solver_runs_at_the_corrected_co2_values():
    config = _MultiConfig()
    G = _one_edge()
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    po2, pco2, ph = solve_multi_species_perfusion(grid, G, [0], map_vessels_to_grid(G, grid), config)
    for field in (po2, pco2, ph):
        assert np.all(np.isfinite(field))
    # Tissue produces CO2, so no cell can sit below the arterial PCO2.
    assert np.all(pco2 >= config.pco2_arterial - 1e-9)
