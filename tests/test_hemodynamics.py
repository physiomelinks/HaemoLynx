"""Tests for haemodynamics module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.haemodynamics import (
    PoiseuilleModel,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)

MODEL = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)


def test_calculate_viscosity():
    assert MODEL.calculate_viscosity(1.0) == 1.0
    assert MODEL.calculate_viscosity(2.0) < 1.0


def test_get_diameter_at_position():
    d = MODEL.get_diameter_at_position(0, 100, 5.0, 4.0)
    assert 4.0 <= d <= 5.0


def test_resistance_integrand():
    r = MODEL.resistance_integrand(10, 100, 5.0, 4.0)
    assert r > 0


def test_calculate_integrated_resistance():
    R = MODEL.calculate_integrated_resistance(50.0, 5.0, 4.0, num_points=50)
    assert R > 0
    assert R < float("inf")
    assert MODEL.calculate_integrated_resistance(0, 5, 4) == float("inf")


def test_set_poiseuille_resistances_with_constrictions(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    config = {"BO1": {"d1": 6.2, "d2": 6.2}}
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(G, config)
    assert isinstance(out_graph, nx.MultiGraph)
    assert res["resistances_set"] >= 0


def test_set_poiseuille_resistances_with_constrictions_fwhm_baseline(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G[0][1][0]["fwhm_diameter_um"] = 4.0
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(
        G,
        {"BO1": 10.0},
        prefer_edge_fwhm_baseline=True,
        constriction_factor_by_branch_order={"BO1": 0.8},
    )
    assert isinstance(out_graph, nx.MultiGraph)
    assert res["resistances_set"] == 1
    assert res["used_fwhm_baseline"] == 1
    assert out_graph[0][1][0]["resistance"] > 0


def test_set_poiseuille_resistances_with_constrictions_fwhm_fallback(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(
        G,
        {"BO1": 10.0},
        prefer_edge_fwhm_baseline=True,
        constriction_factor_by_branch_order={"BO1": 0.5},
    )
    assert res["used_fwhm_baseline"] == 0
    assert res["resistances_set"] == 1
    r_fallback = out_graph[0][1][0]["resistance"]

    G2 = multigraph_with_branch_order.copy()
    G2, _ = MODEL.set_poiseuille_resistances_with_constrictions(
        G2,
        {"BO1": {"d1": 10.0, "d2": 5.0}},
    )
    assert np.isclose(r_fallback, G2[0][1][0]["resistance"])


def test_set_poiseuille_edge_resistances(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    out_graph, res = MODEL.set_poiseuille_edge_resistances(
        G, [(0, 1)], 6.0, use_resistance=True
    )
    assert isinstance(out_graph, nx.MultiGraph)
    assert "updated" in res


def test_calc_laplacian_from_conductance_matrix():
    C = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    L = calc_laplacian_from_conductance_matrix(C)
    assert np.allclose(L, L.T)
    assert np.allclose(np.sum(L, axis=1), 0)


def test_build_conductance_matrix_from_graph():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, resistance=2.0)
    G.add_edge(0, 1, resistance=4.0)  # Parallel edge conductances should be summed.
    G.add_edge(1, 2, resistance=1.0)
    G.add_edge(0, 2, resistance=-3.0)  # Non-positive resistance is ignored.

    C, node_list = build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    i0 = node_to_idx[0]
    i1 = node_to_idx[1]
    i2 = node_to_idx[2]

    assert C.shape == (3, 3)
    assert np.allclose(C, C.T)
    assert np.isclose(C[i0, i1], (1.0 / 2.0) + (1.0 / 4.0))
    assert np.isclose(C[i1, i2], 1.0)
    assert np.isclose(C[i0, i2], 0.0)


def test_calc_two_point_from_laplacian_matrix_nodeID():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, resistance=1)
    G.add_edge(1, 2, resistance=1)
    C = np.zeros((3, 3))
    C[0, 1] = C[1, 0] = 1
    C[1, 2] = C[2, 1] = 1
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)
    assert R > 0
