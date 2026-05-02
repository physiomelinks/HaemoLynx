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


def test_set_poiseuille_edge_resistances(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    out_graph, res = MODEL.set_poiseuille_edge_resistances(
        G, [(0, 1)], 6.0
    )
    assert isinstance(out_graph, nx.MultiGraph)
    assert "updated" in res


def test_calc_laplacian_from_conductance_matrix():
    import scipy.sparse as sp
    C = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    L = calc_laplacian_from_conductance_matrix(C)
    L_dense = L.toarray()
    assert np.allclose(L_dense, L_dense.T)
    assert np.allclose(np.sum(L_dense, axis=1), 0)


def test_build_conductance_matrix_from_graph():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, resistance=1.5)
    G.add_edge(0, 1, resistance=2.5)  # Parallel edge should be summed.
    G.add_edge(1, 2, resistance=1.0)
    G.add_edge(0, 2, resistance=-3.0)  # Non-positive resistances are ignored.

    C, node_list = build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    i0 = node_to_idx[0]
    i1 = node_to_idx[1]
    i2 = node_to_idx[2]

    assert C.shape == (3, 3)
    C_dense = C.toarray()
    assert np.allclose(C_dense, C_dense.T)
    # 1/1.5 + 1/2.5 = 0.666... + 0.4 = 1.0666...
    assert np.isclose(C_dense[i0, i1], 1.0666666666666667)
    assert np.isclose(C_dense[i1, i2], 1.0)
    assert np.isclose(C_dense[i0, i2], 0.0)


def test_calc_two_point_from_laplacian_matrix_nodeID():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, resistance=1)
    G.add_edge(1, 2, resistance=1)
    import scipy.sparse as sp
    C = sp.csr_matrix([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)
    assert R > 0

def test_solve_system_smart_routing(monkeypatch):
    from ImageLynx.haemodynamics.resistance import _solve_system_smart
    import scipy.sparse as sp
    import numpy as np
    
    A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    
    # Test direct solver path
    x = _solve_system_smart(A, b, iterative_threshold=50000)
    assert np.allclose(x, [0.66666667, 0.33333333])
    
    # Test iterative solver path
    x_iter = _solve_system_smart(A, b, iterative_threshold=1)
    assert np.allclose(x_iter, [0.66666667, 0.33333333])


def test_solve_system_smart_singular_fallback():
    from ImageLynx.haemodynamics.resistance import _solve_system_smart
    import scipy.sparse as sp
    import numpy as np
    
    # Singular matrix
    A = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 0.0])
    
    # Direct solver singular fallback
    x = _solve_system_smart(A, b, iterative_threshold=50000)
    assert x is not None  # Should not crash
    
    # Iterative solver singular fallback
    x_iter = _solve_system_smart(A, b, iterative_threshold=1)
    assert x_iter is not None  # Should not crash


def test_solve_system_smart_preconditioner_failure(monkeypatch):
    from ImageLynx.haemodynamics.resistance import _solve_system_smart
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg
    import numpy as np
    
    A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    
    def mock_spilu(*args, **kwargs):
        raise RuntimeError("Mock spilu failure")
        
    monkeypatch.setattr(splinalg, "spilu", mock_spilu)
    
    # Should fall back to lsqr and not crash
    x = _solve_system_smart(A, b, iterative_threshold=1)
    assert np.allclose(x, [0.66666667, 0.33333333])

