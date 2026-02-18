"""Tests for hemodynamics module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.hemodynamics import (
    calculate_viscosity,
    get_diameter_at_position,
    resistance_integrand,
    calculate_integrated_resistance,
    set_poiseuille_weights_with_constrictions,
    set_poiseuille_edge_weights,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)


def test_calculate_viscosity():
    assert calculate_viscosity(1.0) == 1.0
    assert calculate_viscosity(2.0) < 1.0


def test_get_diameter_at_position():
    d = get_diameter_at_position(0, 100, 5.0, 4.0)
    assert 4.0 <= d <= 5.0


def test_resistance_integrand():
    r = resistance_integrand(10, 100, 5.0, 4.0)
    assert r > 0


def test_calculate_integrated_resistance():
    R = calculate_integrated_resistance(50.0, 5.0, 4.0, num_points=50)
    assert R > 0
    assert R < float("inf")
    assert calculate_integrated_resistance(0, 5, 4) == float("inf")


def test_set_poiseuille_weights_with_constrictions(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    config = {"BO1": {"d1": 6.2, "d2": 6.2}}
    res = set_poiseuille_weights_with_constrictions(G, config)
    assert res["weights_set"] >= 0


def test_set_poiseuille_edge_weights(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    res = set_poiseuille_edge_weights(G, [(0, 1)], 6.0, use_resistance=False)
    assert "updated" in res


def test_calc_laplacian_from_conductance_matrix():
    C = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    L = calc_laplacian_from_conductance_matrix(C)
    assert np.allclose(L, L.T)
    assert np.allclose(np.sum(L, axis=1), 0)


def test_calc_two_point_from_laplacian_matrix_nodeID():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    C = np.zeros((3, 3))
    C[0, 1] = C[1, 0] = 1
    C[1, 2] = C[2, 1] = 1
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)
    assert R > 0
