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

def test_boundary_mode_universal_sink():
    from ImageLynx.graph.boundaries import select_boundary_terminal_nodes
    import networkx as nx
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 50, 50)) # Z-top
    G.add_node(1, pos=(50, 50, 50)) # Center
    G.add_node(2, pos=(100, 50, 50)) # Z-bottom
    G.add_node(3, pos=(50, 0, 50)) # Y-edge
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    
    # Universal sink should grab node 3 as an outlet
    start, out = select_boundary_terminal_nodes(
        G, (101, 101, 101), edge_percent=25.0, end_percent=25.0, axis=0, boundary_permeability_mode="universal_sink"
    )
    assert 0 in start
    assert 2 in out
    assert 3 in out # Swept up!

def test_robin_matrix_ghost_node_generation():
    from ImageLynx.haemodynamics.resistance import build_conductance_matrix_from_graph
    import networkx as nx
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2, is_robin_boundary=True)
    G.add_edge(0, 1, resistance=1.0)
    G.add_edge(1, 2, resistance=2.0)
    
    # Matrix should be 4x4 (3 nodes + 1 ghost)
    cond, nodes = build_conductance_matrix_from_graph(G, robin_multiplier=10.0)
    assert cond.shape == (4, 4)
    assert "ROBIN_GHOST_NODE" in nodes
    
    ghost_idx = nodes.index("ROBIN_GHOST_NODE")
    node2_idx = nodes.index(2)
    
    # Check that Robin resistance was properly scaled: 
    # Edge to node 2 has R=2.0. Multiplier is 10. Ghost R = 20.0. Conductance = 1/20 = 0.05
    assert np.isclose(cond[ghost_idx, node2_idx], 0.05)

def test_robin_vs_sink_flow_conservation(monkeypatch):
    from ImageLynx.haemodynamics.resistance import build_conductance_matrix_from_graph, solve_flow_from_conductance_matrix
    import networkx as nx
    import pyvista as pv
    
    # Mock pv.read to return a dummy mesh
    class DummyMesh:
        def __init__(self):
            self.cell_data = {
                "edge_u": [0, 1, 1],
                "edge_v": [1, 2, 3],
                "resistance": [1.0, 1.0, 1.0]
            }
            self.n_cells = 3
        def save(self, *args, **kwargs):
            pass
    monkeypatch.setattr(pv, "read", lambda x: DummyMesh())
    
    G = nx.MultiGraph()
    G.add_node(0, pos=(0, 50, 50)) # Inlet
    G.add_node(1, pos=(50, 50, 50)) # Hub
    G.add_node(2, pos=(100, 50, 50)) # Outlet
    G.add_node(3, pos=(50, 0, 50), is_robin_boundary=True) # X/Y dead-end
    G.add_edge(0, 1, key=0, resistance=1.0)
    G.add_edge(1, 2, key=0, resistance=1.0)
    G.add_edge(1, 3, key=0, resistance=1.0)
    
    cond, nodes = build_conductance_matrix_from_graph(G, robin_multiplier=1.0)
    # Output nodes must explicitly include the ghost node!
    outputs = [2, "ROBIN_GHOST_NODE"] 
    
    vtk_export_in = {
        "point_data": {}, 
        "cell_data": {}, 
        "vessels_path": "dummy.vtp",
        "edges_u": [0, 1, 1],
        "edges_v": [1, 2, 3],
        "edges_k": [0, 0, 0]
    }
    
    p_nodes, vtk_export = solve_flow_from_conductance_matrix(
        cond, nodes, 100.0, 0.0, [0], outputs, vtk_export_in
    )
    
    # Prove mass conservation using the output pressures directly
    p_dict = dict(zip(p_nodes["node_list"], p_nodes["pressure"]))
    
    flow_0_1 = (p_dict[0] - p_dict[1]) / 1.0
    flow_1_2 = (p_dict[1] - p_dict[2]) / 1.0
    flow_1_3 = (p_dict[1] - p_dict[3]) / 1.0
    
    # In robin mode, the flow OUT of node 3 goes to the Ghost Node
    flow_3_ghost = (p_dict[3] - p_dict["ROBIN_GHOST_NODE"]) / 1.0 # robin_multiplier=1.0 * resistance=1.0
    
    flow_in = flow_0_1
    flow_out = flow_1_2 + flow_3_ghost
                
    assert np.isclose(flow_in, flow_out, atol=1e-8)

