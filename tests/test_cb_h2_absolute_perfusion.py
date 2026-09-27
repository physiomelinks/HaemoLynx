"""The absolute-perfusion summary behind reference §13.5 and S27 (open item 12).

Those numbers were quoted for a month with no script behind them, and were computed on a
rheology loop that inflated every resistance about 200-540x. So the summary is checked against
Hagen-Poiseuille by hand, and the pressure extrapolation against a second solve rather than
against its own formula.
"""
import networkx as nx
import numpy as np
import pytest

from cb_h2_absolute_perfusion import (
    TARGET_VELOCITY_UM_S, perfusion_summary, select_boundaries,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S
from ImageLynx.haemodynamics.rheology import (
    calculate_pries_secomb_viscosity,
    solve_coupled_flow_and_hematocrit,
)

P_IN, P_OUT = 60.0, 20.0


def _y_network():
    """15 um parent splitting into 10 and 5 um daughters, so phase separation is active."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=50.0, assigned_diameter_um=15.0)
    G.add_edge(1, 2, key=0, length=80.0, assigned_diameter_um=10.0)
    G.add_edge(1, 3, key=0, length=80.0, assigned_diameter_um=5.0)
    return G


def _solve(G, inlets, outlets, p_in, p_out=P_OUT):
    G, _ = solve_coupled_flow_and_hematocrit(
        G, inlets, outlets, p_in, p_out, max_iterations=200, tolerance=1e-12)
    return G


def test_a_single_tube_matches_hagen_poiseuille_in_um3_per_s():
    """One edge carries no phase separation, so Q = dP pi d^4 / (128 mu_PS(d, 0.45) L)."""
    d, length = 6.0, 100.0
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=length, assigned_diameter_um=d)
    G = _solve(G, [0], [1], P_IN)

    summary = perfusion_summary(G, [0], P_IN - P_OUT)

    mu = calculate_pries_secomb_viscosity(d, 0.45)
    q = (P_IN - P_OUT) * np.pi * d ** 4 / (128.0 * mu * length) * POISEUILLE_FLOW_TO_UM3_PER_S
    assert summary["total_inlet_flow_um3_s"] == pytest.approx(q, rel=1e-9)
    assert summary["flow_weighted_velocity_um_s"] == pytest.approx(
        q / (np.pi * d ** 2 / 4.0), rel=1e-9)


def test_flow_weighted_velocity_and_inlet_flow_match_a_hand_calculation():
    """Two edges with set flows: velocity weighted by flow, inlet flow only at the inlet."""
    G = nx.MultiGraph()
    G.add_edge("in", "j", key=0, assigned_diameter_um=4.0, flow_abs=3.0)
    G.add_edge("j", "out", key=0, assigned_diameter_um=8.0, flow_abs=1.0)

    summary = perfusion_summary(G, ["in"], 40.0)

    q = np.array([3.0, 1.0]) * POISEUILLE_FLOW_TO_UM3_PER_S
    v = q / (np.pi * np.array([4.0, 8.0]) ** 2 / 4.0)
    assert summary["total_inlet_flow_um3_s"] == pytest.approx(q[0], rel=1e-12)
    assert summary["flow_weighted_velocity_um_s"] == pytest.approx(
        (q * v).sum() / q.sum(), rel=1e-12)


def test_the_pressure_for_the_target_velocity_reaches_it_when_solved():
    """The extrapolation assumes the solve is linear in dP; a second solve checks that.

    Phase separation reads flow fractions, which a uniform scaling leaves unchanged, so every
    flow and every haematocrit should scale exactly with the pressure drop.
    """
    base = _solve(_y_network(), [0], [2, 3], P_IN)
    summary = perfusion_summary(base, [0], P_IN - P_OUT)
    needed = summary["pressure_drop_for_target_mmhg"]

    scaled = _solve(_y_network(), [0], [2, 3], P_OUT + needed)
    again = perfusion_summary(scaled, [0], needed)

    assert again["flow_weighted_velocity_um_s"] == pytest.approx(TARGET_VELOCITY_UM_S, rel=1e-6)
    ratio = needed / (P_IN - P_OUT)
    for u, v, k, data in base.edges(keys=True, data=True):
        other = scaled[u][v][k]
        assert other["flow_abs"] == pytest.approx(ratio * data["flow_abs"], rel=1e-6)
        assert other["hematocrit"] == pytest.approx(data["hematocrit"], rel=1e-6)


def test_a_missing_diameter_raises():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, flow_abs=1.0)
    with pytest.raises(ValueError, match="diameter"):
        perfusion_summary(G, [0], 40.0)


def test_a_missing_flow_raises():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, assigned_diameter_um=5.0)
    with pytest.raises(ValueError, match="flow_abs"):
        perfusion_summary(G, [0], 40.0)


def test_a_network_with_no_flow_raises():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, assigned_diameter_um=5.0, flow_abs=0.0)
    with pytest.raises(ValueError, match="no flow"):
        perfusion_summary(G, [0], 40.0)


def test_an_unknown_boundary_rule_raises():
    with pytest.raises(ValueError, match="unknown boundary rule"):
        select_boundaries(nx.MultiGraph(), "extreme")
