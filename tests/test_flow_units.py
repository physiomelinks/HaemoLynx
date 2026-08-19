"""T1.5: the flow solve's units, and what makes them commensurable with the tissue sink.

The flow solve computes Q = dP / R with R = 128 mu L / (pi d^4), taking pressure in mmHg,
viscosity in cP and lengths in um. That mixture is not an SI-derived system, so Q is not in
um^3/s. The metabolic sink is in mmol/L/s times um^3, which is. Coupling them without a
conversion asked the tissue to consume 2.2e4 times the oxygen the blood delivered, and the
steady-state PO2 was correctly zero everywhere.
"""
import numpy as np
import pytest

from ImageLynx.haemodynamics.resistance import (
    POISEUILLE_FLOW_TO_UM3_PER_S,
    poiseuille_flow_to_um3_per_s,
)

PA_PER_MMHG = 133.322387415


def test_the_factor_is_derived_from_the_unit_definitions_not_fitted():
    """dP in mmHg over R in cP/um^3 gives mmHg um^3 / cP; the factor is the SI bridge."""
    expected = PA_PER_MMHG / (1e-3 * (1e-6) ** 2) * 1e-18
    assert POISEUILLE_FLOW_TO_UM3_PER_S == pytest.approx(expected * 1e6, rel=1e-9)
    assert POISEUILLE_FLOW_TO_UM3_PER_S == pytest.approx(PA_PER_MMHG * 1e3, rel=1e-12)


def test_a_single_tube_matches_an_independent_si_computation():
    """Poiseuille worked twice: once in the pipeline's units, once wholly in SI."""
    d_um, l_um, mu_cp, dp_mmhg = 8.0, 300.0, 4.0, 40.0

    r_pipeline = (128.0 * mu_cp * l_um) / (np.pi * d_um ** 4)
    q_pipeline = dp_mmhg / r_pipeline

    r_si = (128.0 * mu_cp * 1e-3 * l_um * 1e-6) / (np.pi * (d_um * 1e-6) ** 4)
    q_si_um3_s = (dp_mmhg * PA_PER_MMHG) / r_si * 1e18

    assert poiseuille_flow_to_um3_per_s(q_pipeline) == pytest.approx(q_si_um3_s, rel=1e-9)


def test_the_conversion_is_linear_and_handles_arrays():
    q = np.array([0.0, 1.0, 2.5, 1e-4])
    out = poiseuille_flow_to_um3_per_s(q)
    assert out.shape == q.shape
    np.testing.assert_allclose(out, q * POISEUILLE_FLOW_TO_UM3_PER_S)
    assert poiseuille_flow_to_um3_per_s(0.0) == 0.0


def test_the_converted_flow_gives_a_physically_readable_velocity():
    """A capillary velocity should land in um/s, not in an arbitrary scale."""
    d_um, l_um, mu_cp, dp_mmhg = 8.0, 300.0, 4.0, 1.0
    q = dp_mmhg / ((128.0 * mu_cp * l_um) / (np.pi * d_um ** 4))
    velocity = poiseuille_flow_to_um3_per_s(q) / (np.pi * (d_um / 2) ** 2)
    assert 10.0 < velocity < 1e6, f"{velocity} um/s is not a readable capillary velocity"


def test_the_grid_coupling_applies_the_conversion():
    import networkx as nx

    from ImageLynx.haemodynamics.perfusion import PerfusionGrid, map_vessels_to_grid

    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([30.0, 30.0, 30.0]))
    G.add_edge(1, 2, key=0, length=20.0, flow_abs=3.0, assigned_diameter_um=8.0,
               voxels=[np.array([t, t, t]) for t in np.linspace(2.0, 28.0, 6)])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))

    converted = map_vessels_to_grid(G, grid)
    raw = map_vessels_to_grid(G, grid, flow_to_um3_per_s=1.0)

    total_converted = sum(v["flow"] for cell in converted.values() for v in cell)
    total_raw = sum(v["flow"] for cell in raw.values() for v in cell)
    assert total_converted == pytest.approx(total_raw * POISEUILLE_FLOW_TO_UM3_PER_S)


def test_passing_one_leaves_flow_in_solver_units():
    """An escape for callers comparing against pre-conversion output."""
    assert poiseuille_flow_to_um3_per_s(7.0, factor=1.0) == pytest.approx(7.0)
