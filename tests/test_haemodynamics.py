"""Tests for haemodynamics module."""
import numpy as np
import networkx as nx
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from ImageLynx.haemodynamics import (
    PoiseuilleModel,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)

MODEL = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)


def test_the_integrated_resistance_of_a_zero_length_vessel_is_infinite():
    assert MODEL.calculate_integrated_resistance(0, 5, 4) == float("inf")


def test_set_poiseuille_resistances_with_constrictions(multigraph_with_branch_order):
    """With d1 = d2 and no measured diameter, the 5 um edge is a straight 6.2 um tube."""
    G = multigraph_with_branch_order.copy()
    config = {"BO1": {"d1": 6.2, "d2": 6.2}}
    out_graph, res = MODEL.set_poiseuille_resistances_with_constrictions(G, config)
    edge = out_graph.edges[0, 1, 0]
    assert res["resistances_set"] == 1
    assert edge["assigned_diameter_um"] == 6.2
    assert edge["diameter_provenance"] == "synthetic_branch_order"
    assert edge["resistance"] == pytest.approx(128 * 5 / (np.pi * 6.2 ** 5.647), rel=1e-12)


def test_set_poiseuille_edge_resistances(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    absent = (0, 7)
    out_graph, res = MODEL.set_poiseuille_edge_resistances(
        G, [(0, 1), absent], 6.0
    )
    edge = out_graph.edges[0, 1, 0]
    assert edge["resistance"] == pytest.approx(
        128 * 6 ** -1.647 * 5 / (np.pi * 6 ** 4), rel=1e-12)
    assert edge["assigned_diameter_um"] == 6.0
    assert [item["edge"] for item in res["updated"]] == [(0, 1, 0)]
    assert res["not_found"] == [absent]


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
    C = sp.csr_matrix([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    L = calc_laplacian_from_conductance_matrix(C)
    R = calc_two_point_from_laplacian_matrix_nodeID(L, G, 0, 2)
    assert R == pytest.approx(2.0)

def test_solve_system_smart_routing(monkeypatch):
    from ImageLynx.haemodynamics.resistance import _solve_system_smart

    A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    
    # Test direct solver path
    x = _solve_system_smart(A, b, iterative_threshold=50000)
    assert np.allclose(x, [0.66666667, 0.33333333])
    
    # Test iterative solver path
    x_iter = _solve_system_smart(A, b, iterative_threshold=1)
    assert np.allclose(x_iter, [0.66666667, 0.33333333])


def test_solve_system_smart_raises_on_a_singular_system():
    """A singular network Laplacian is a graph defect, not a solve to approximate.

    This used to fall back to lsqr in both branches. Least squares answers a singular
    Laplacian with a minimum-norm pressure field over a component that nothing is driving:
    well-formed, and meaningless. The graph-level largest-component prune is what normally
    prevents the condition, so a silent fallback here hid the prune being turned off.
    """
    from ImageLynx.haemodynamics.resistance import _solve_system_smart

    A = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="singular|did not converge"):
        _solve_system_smart(A, b, iterative_threshold=50000)

    with pytest.raises(ValueError, match="singular|did not converge|failed"):
        _solve_system_smart(A, b, iterative_threshold=1)


def test_solve_system_smart_preconditioner_failure(monkeypatch):
    from ImageLynx.haemodynamics.resistance import _solve_system_smart

    A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    
    def mock_spilu(*args, **kwargs):
        raise RuntimeError("Mock spilu failure")
        
    monkeypatch.setattr(splinalg, "spilu", mock_spilu)

    # A preconditioner that cannot be built is reported, not papered over with lsqr:
    # the fallback returned a field that looked solved and was not.
    with pytest.raises(ValueError, match="Preconditioning"):
        _solve_system_smart(A, b, iterative_threshold=1)

def test_boundary_mode_universal_sink():
    from ImageLynx.graph.boundaries import select_boundary_terminal_nodes
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

def test_a_robin_ghost_node_takes_its_share_of_the_outflow(monkeypatch):
    """Inlet 0 at 100, outlet 2 and the ghost at 0, every resistance 1, the ghost's included.

    Node 1 settles at 40 and the Robin node 3 at 20, so 60 flows in, 40 leaves through node 2
    and 20 through the ghost. Balance at node 1 alone would hold with no flow to the ghost.
    """
    from ImageLynx.haemodynamics.resistance import solve_flow_from_conductance_matrix
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
    
    p_dict = dict(zip(p_nodes["node_list"], p_nodes["pressure"]))

    # Every resistance is 1, so each flow is the pressure drop across its edge. The ghost
    # edge's resistance is robin_multiplier 1.0 times node 3's resistance 1.0.
    flow_in = p_dict[0] - p_dict[1]
    to_node_2 = p_dict[1] - p_dict[2]
    to_ghost = p_dict[3] - p_dict["ROBIN_GHOST_NODE"]

    assert flow_in == pytest.approx(60.0)
    assert to_node_2 == pytest.approx(40.0)
    assert to_ghost == pytest.approx(20.0)

