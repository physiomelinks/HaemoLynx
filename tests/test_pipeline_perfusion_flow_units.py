"""Open item 31: the pipeline's perfusion step converted flow to um^3/s twice.

``carotid_image_to_model.py`` gives the flow solve its pressures in mPa, so with viscosity in cP
(mPa s) and lengths in um its edge flow is already in um^3/s. The perfusion library's default
``flow_to_um3_per_s`` (1.33e5) assumes mmHg, as the H2 drivers pass. The pipeline took that
default, so its perfusion saw flow 1.33e5 times too high. It now passes its own factor, 1.0.
"""
import ast
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from carotid_image_to_model import (
    _MPA_PER_MMHG,
    _perfusion_flow_to_um3_per_s,
    HaemodynamicsConfig,
    PerfusionConfig,
)
from ImageLynx.haemodynamics.perfusion import (
    map_vessels_to_grid,
    PerfusionGrid,
    solve_coupled_1d3d_perfusion,
    solve_multi_species_perfusion,
)
from ImageLynx.haemodynamics.resistance import PASCALS_PER_MMHG, POISEUILLE_FLOW_TO_UM3_PER_S
from ImageLynx.haemodynamics.rheology import solve_coupled_flow_and_hematocrit

_PIPELINE = Path(__file__).resolve().parents[1] / "examples" / "carotid_image_to_model.py"
_PERFUSION_CALLS = {"map_vessels_to_grid", "solve_multi_species_perfusion",
                    "solve_coupled_1d3d_perfusion"}

_D_UM = 8.0
_L_UM = 60.0


def test_the_pipeline_factor_is_one():
    assert _perfusion_flow_to_um3_per_s() == pytest.approx(1.0, rel=1e-12)
    # Derived, not written down: the library's mmHg factor over mPa per mmHg.
    assert _MPA_PER_MMHG == pytest.approx(PASCALS_PER_MMHG * 1e3, rel=1e-15)
    assert _perfusion_flow_to_um3_per_s() == pytest.approx(
        POISEUILLE_FLOW_TO_UM3_PER_S / (PASCALS_PER_MMHG * 1e3), rel=1e-15)


def test_the_config_pressures_are_in_mpa():
    hemo = HaemodynamicsConfig()
    assert hemo.input_p_bc / _MPA_PER_MMHG == pytest.approx(60.0)
    assert hemo.output_p_bc / _MPA_PER_MMHG == pytest.approx(20.0)


def _tube_solved_at_pipeline_pressures():
    """One straight tube, solved by the pipeline's own flow solve at its default pressures."""
    pts = [np.array([t, 15.0, 15.0]) for t in np.linspace(0.0, _L_UM, 13)]
    G = nx.MultiGraph()
    G.add_node(0, pos=pts[0])
    G.add_node(1, pos=pts[-1])
    G.add_edge(0, 1, key=0, length=_L_UM, fwhm_diameter_um=_D_UM, assigned_diameter_um=_D_UM,
               voxels=pts)
    hemo = HaemodynamicsConfig()
    G, _ = solve_coupled_flow_and_hematocrit(G, [0], [1], hemo.input_p_bc, hemo.output_p_bc,
                                             systemic_hematocrit=0.45, max_iterations=1)
    return G, hemo


def _si_poiseuille_um3_per_s(dp_mmhg, mu_cp, d_um, l_um):
    """Q = pi d^4 dP / (128 mu L), worked in Pa, Pa s and m, then taken to um^3/s."""
    dp_pa = dp_mmhg * PASCALS_PER_MMHG
    mu_pa_s = mu_cp * 1e-3
    d_m, l_m = d_um * 1e-6, l_um * 1e-6
    return np.pi * d_m ** 4 * dp_pa / (128.0 * mu_pa_s * l_m) * 1e18


def test_pipeline_flow_times_its_factor_is_the_si_poiseuille_flow():
    G, hemo = _tube_solved_at_pipeline_pressures()
    data = G[0][1][0]
    dp_mmhg = (hemo.input_p_bc - hemo.output_p_bc) / _MPA_PER_MMHG
    want = _si_poiseuille_um3_per_s(dp_mmhg, data["viscosity"], _D_UM, _L_UM)

    assert data["flow_abs"] * _perfusion_flow_to_um3_per_s() == pytest.approx(want, rel=1e-6)
    # The library default, which the pipeline used, is 1.33e5 times too high here.
    assert data["flow_abs"] * POISEUILLE_FLOW_TO_UM3_PER_S == pytest.approx(
        want * POISEUILLE_FLOW_TO_UM3_PER_S, rel=1e-6)


def test_the_grid_gets_the_si_flow_with_the_pipeline_factor():
    G, hemo = _tube_solved_at_pipeline_pressures()
    dp_mmhg = (hemo.input_p_bc - hemo.output_p_bc) / _MPA_PER_MMHG
    want = _si_poiseuille_um3_per_s(dp_mmhg, G[0][1][0]["viscosity"], _D_UM, _L_UM)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))

    fixed = map_vessels_to_grid(G, grid, flow_to_um3_per_s=_perfusion_flow_to_um3_per_s())
    old = map_vessels_to_grid(G, grid)

    fixed_flows = [v["flow"] for cell in fixed.values() for v in cell]
    old_flows = [v["flow"] for cell in old.values() for v in cell]
    assert fixed_flows
    np.testing.assert_allclose(fixed_flows, want, rtol=1e-6)
    np.testing.assert_allclose(old_flows, want * POISEUILLE_FLOW_TO_UM3_PER_S, rtol=1e-6)


@pytest.mark.parametrize("solver", [solve_multi_species_perfusion, solve_coupled_1d3d_perfusion])
def test_a_solver_left_on_the_default_factor_is_refused(solver):
    """A missed call site cannot pass quietly: the map and the solver must agree."""
    G, _ = _tube_solved_at_pipeline_pressures()
    for _, _, d in G.edges(data=True):
        d["flow_signed"] = d["flow_abs"]
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    cells = map_vessels_to_grid(G, grid, flow_to_um3_per_s=_perfusion_flow_to_um3_per_s())
    with pytest.raises(ValueError, match="different conversion factors"):
        solver(grid, G, [0], cells, PerfusionConfig())


def _perfusion_calls_in_pipeline():
    tree = ast.parse(_PIPELINE.read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _PERFUSION_CALLS:
                calls.append((name, node))
    return calls


def test_every_pipeline_perfusion_call_passes_the_pipeline_factor():
    calls = _perfusion_calls_in_pipeline()
    assert {name for name, _ in calls} == _PERFUSION_CALLS
    for name, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert "flow_to_um3_per_s" in kw, f"{name} at line {node.lineno} takes the mmHg default"
        value = kw["flow_to_um3_per_s"]
        assert isinstance(value, ast.Name) and value.id == "flow_to_um3_per_s", (
            f"{name} at line {node.lineno} is not given the pipeline's factor")
