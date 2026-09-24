"""Current-flow shares, single-vessel occlusion impact and occlusion curves."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics.resistance import (
    build_conductance_matrix_from_graph,
    set_edge_flows,
    solve_flow_from_conductance_matrix,
)
from haemolynx.statistics.current_flow import compute_current_flow
from haemolynx.statistics.occlusion import (
    compute_occlusion_curves,
    compute_single_vessel_occlusion_impact,
)


def _vessel(G, u, v, g, length=10.0, **extra):
    return G.add_edge(u, v, conductance=float(g), resistance=1.0 / float(g), length=float(length), **extra)


def _meshed_network():
    """Inlet 0 -> outlet 9 through a small mesh: a single feeding vessel
    (0-1, a bridge), a cross-linked middle, a doubled 4-5 pair, parallel
    exits 5-9 / 5-6-9, and a dead-end spur 3-7-8."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 4, branch_order="Art1")
    _vessel(G, 1, 2, 1, branch_order="Art2")
    _vessel(G, 1, 3, 2, branch_order="Art2")
    _vessel(G, 2, 4, 1, branch_order="B01")
    _vessel(G, 3, 4, 3, branch_order="B01")
    _vessel(G, 2, 3, 0.5, branch_order="B02")
    _vessel(G, 4, 5, 2, branch_order="Ven2")
    _vessel(G, 4, 5, 1, branch_order="Ven2")
    _vessel(G, 5, 9, 5, branch_order="Ven1")
    _vessel(G, 5, 6, 1, branch_order="Ven1")
    _vessel(G, 6, 9, 1, branch_order="Ven1")
    _vessel(G, 3, 7, 1, branch_order="B02")
    _vessel(G, 7, 8, 1)
    return G


def _brute_force_flows(G, inlets, outlets):
    """Per-vessel signed flows and total inflow at inlet=1, outlet=0, via the
    pipeline's own dense solver; all zero when inlet and outlet are cut apart."""
    C, node_list = build_conductance_matrix_from_graph(G)
    H = G.copy()
    try:
        solved = solve_flow_from_conductance_matrix(
            C, node_list, inlet_p_bc=1.0, outlet_p_bc=0.0,
            inlet_nodes=list(inlets), outlet_nodes=list(outlets),
        )
    except ValueError:
        return {(u, v, k): 0.0 for u, v, k in H.edges(keys=True)}, 0.0
    set_edge_flows(H, node_list, solved["pressure"])
    flows = {(u, v, k): d.get("flow_signed", 0.0) for u, v, k, d in H.edges(keys=True, data=True)}
    total = 0.0
    for (u, v, k), f in flows.items():
        if u in inlets and v not in inlets:
            total += f
        elif v in inlets and u not in inlets:
            total -= f
    return flows, total


# --- current flow ------------------------------------------------------------------


def test_parallel_vessels_split_current_by_conductance():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 1)
    _vessel(G, 0, 1, 3)
    result = compute_current_flow(G, [0], [1], weighting="resistance")
    shares = sorted(d["current_flow_share"] for _u, _v, d in G.edges(data=True))
    assert shares == pytest.approx([0.25, 0.75])
    assert result["Equivalent Inlet-Outlet Resistance"] == pytest.approx(1 / 4)


def test_series_resistance_adds_and_every_series_vessel_carries_everything():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 1, length=10)
    _vessel(G, 1, 2, 1, length=30)
    result = compute_current_flow(G, [0], [2], weighting="length")
    assert result["Equivalent Inlet-Outlet Resistance"] == pytest.approx(40.0)
    shares = [d["current_flow_share"] for _u, _v, d in G.edges(data=True)]
    assert shares == pytest.approx([1.0, 1.0])


def test_current_flow_matches_the_pipelines_own_flow_solve():
    G = _meshed_network()
    compute_current_flow(G, [0], [9], weighting="resistance")
    flows, total = _brute_force_flows(G, {0}, {9})
    for u, v, k, d in G.edges(keys=True, data=True):
        assert d["current_flow_share"] == pytest.approx(abs(flows[(u, v, k)]) / total, abs=1e-9)
    assert G[7][8][0]["current_flow_share"] == pytest.approx(0.0)


def test_kirchhoff_index_matches_the_path_graph_closed_form():
    """A unit-resistance path of n nodes has Kirchhoff index (n^3 - n) / 6."""
    n = 6
    G = nx.MultiGraph()
    for i in range(n - 1):
        _vessel(G, i, i + 1, 1)
    result = compute_current_flow(G, [0], [n - 1], weighting="topology", statistics_mode="full")
    assert result["Kirchhoff Index"] == pytest.approx((n**3 - n) / 6)


def test_kirchhoff_index_is_full_mode_only():
    G = _meshed_network()
    result = compute_current_flow(G, [0], [9], statistics_mode="fast")
    assert str(result["Kirchhoff Index"]).startswith("N/A")


# --- single-vessel occlusion impact -------------------------------------------------


def test_occlusion_impact_matches_brute_force_removal_of_every_vessel():
    G = _meshed_network()
    compute_single_vessel_occlusion_impact(
        G, [0], [9], weighting="resistance", statistics_mode="full", hypoperfusion_fraction=0.5
    )
    base_flows, base_total = _brute_force_flows(G, {0}, {9})
    for u, v, k in list(G.edges(keys=True)):
        H = G.copy()
        H.remove_edge(u, v, k)
        flows, total = _brute_force_flows(H, {0}, {9})
        expected_loss = 1.0 - total / base_total
        expected_hypo = sum(
            G[a][b][c]["length"]
            for (a, b, c), f in base_flows.items()
            if (a, b, c) != (u, v, k)
            and abs(f) > 1e-12
            and abs(flows.get((a, b, c), 0.0)) <= 0.5 * abs(f)
        )
        data = G[u][v][k]
        assert data["occlusion_flow_loss"] == pytest.approx(expected_loss, abs=1e-9), (u, v, k)
        assert data["occlusion_hypoperfused_length_um"] == pytest.approx(expected_hypo), (u, v, k)


def test_blocking_the_only_feeding_vessel_stops_all_flow_and_a_spur_costs_nothing():
    G = _meshed_network()
    result = compute_single_vessel_occlusion_impact(G, [0], [9], weighting="resistance")
    assert G[0][1][0]["occlusion_flow_loss"] == pytest.approx(1.0)
    assert G[7][8][0]["occlusion_flow_loss"] == pytest.approx(0.0)
    assert G[3][7][0]["occlusion_flow_loss"] == pytest.approx(0.0)
    network = result["Occlusion Impact (Network)"]
    assert network["Most Critical Vessel"][:3] == (0, 1, 0)
    assert network["Vessels Losing Over 99% Of Flow"] == 1


def test_occlusion_impact_is_grouped_by_vessel_type_and_branch_order():
    G = _meshed_network()
    result = compute_single_vessel_occlusion_impact(G, [0], [9], weighting="resistance")
    by_type = result["Occlusion Impact By Vessel Type"]
    assert list(by_type) == ["arteriole", "capillary", "venule", "unassigned"]
    assert by_type["arteriole"]["Vessel Count"] == 3
    assert by_type["arteriole"]["Max Flow Loss"] == pytest.approx(1.0)
    assert by_type["unassigned"]["Vessel Count"] == 1  # the 7-8 spur tip

    by_order = result["Occlusion Impact By Branch Order"]
    assert list(by_order["arteriole"]) == ["Art1", "Art2"]
    assert list(by_order["capillary"]) == ["BO1", "BO2"]
    assert list(by_order["venule"]) == ["Ven1", "Ven2"]
    assert by_order["venule"]["Ven2"]["Vessel Count"] == 2


def test_fast_mode_caps_evaluated_vessels_by_flow_power():
    G = _meshed_network()
    result = compute_single_vessel_occlusion_impact(
        G, [0], [9], weighting="resistance", statistics_mode="fast", max_evaluated_fast=3
    )
    evaluated = [d["occlusion_flow_loss"] for _u, _v, d in G.edges(data=True)]
    assert sum(1 for x in evaluated if math.isnan(x)) > 0
    assert G[0][1][0]["occlusion_flow_loss"] == pytest.approx(1.0)  # highest power: kept
    assert "not evaluated" in result["Occlusion Impact Method"]
    assert result["Occlusion Impact (Network)"]["Evaluated Count"] < 13


def test_occlusion_impact_without_terminals_writes_nothing():
    G = _meshed_network()
    result = compute_single_vessel_occlusion_impact(G, [], [9])
    assert str(result["Occlusion Impact Status"]).startswith("N/A")
    assert all("occlusion_flow_loss" not in d for _u, _v, d in G.edges(data=True))


# --- occlusion curves -------------------------------------------------------------


def test_every_curve_starts_intact_and_never_recovers():
    G = _meshed_network()
    for _u, _v, d in G.edges(data=True):
        d["diameter_um"] = 1.0 / d["resistance"]
    result = compute_occlusion_curves(G, [0], [9], weighting="resistance")
    for name in ("random", "highest_current_flow_share", "narrowest_diameter"):
        curve = result[f"Occlusion Curve ({name})"]["Curve [fraction removed, flow fraction, outlets reachable]"]
        flows = [row[1] for row in curve]
        assert flows[0] == pytest.approx(1.0)
        assert curve[0][2] == pytest.approx(1.0)
        if name != "random":  # the random curve is an average, not one removal sequence
            assert all(b <= a + 1e-9 for a, b in zip(flows, flows[1:])), name


def test_a_chain_loses_all_flow_at_the_first_removal():
    G = nx.MultiGraph()
    for i in range(10):
        _vessel(G, i, i + 1, 1)
    result = compute_occlusion_curves(G, [0], [10], max_fraction=0.5)
    curve = result["Occlusion Curve (highest_current_flow_share)"]
    assert curve["Curve [fraction removed, flow fraction, outlets reachable]"][1][1] == pytest.approx(0.0)
    # Flow is 1 before the first step and 0 after it: halfway, interpolated.
    from haemolynx.statistics.occlusion import FAST_CURVE_STEPS

    assert curve["Removal Fraction At 50% Flow"] == pytest.approx(0.5 * 0.5 / FAST_CURVE_STEPS)
    assert result["Occlusion Curve (narrowest_diameter)"].startswith("N/A")


def test_busiest_first_removal_hurts_more_than_random():
    grid = nx.grid_2d_graph(6, 6)
    G = nx.MultiGraph()
    for u, v in grid.edges():
        _vessel(G, u, v, 1)
    result = compute_occlusion_curves(G, [(0, 0)], [(5, 5)], weighting="topology")
    targeted = result["Occlusion Curve (highest_current_flow_share)"]["Robustness Index"]
    random_ = result["Occlusion Curve (random)"]["Robustness Index"]
    assert targeted < random_
