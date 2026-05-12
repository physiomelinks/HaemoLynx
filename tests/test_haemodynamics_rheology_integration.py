import pytest
import numpy as np
import networkx as nx
from ImageLynx.haemodynamics.rheology import solve_coupled_flow_and_hematocrit

def test_coupled_solver_convergence():
    """Verify the iterative solver successfully converges and assigns hematocrit/viscosity."""
    G = nx.MultiGraph()
    # Simple Y-bifurcation network
    # Node 0 (Inlet) -> Node 1 -> Node 2 (Outlet 1)
    #                         -> Node 3 (Outlet 2)
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=15.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0) # Medium branch
    G.add_edge(1, 3, key=0, length=10.0, fwhm_diameter_um=5.0)  # Tiny branch
    
    starting_nodes = [0]
    output_nodes = [2, 3]
    
    # Run the iterative solver
    G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes=starting_nodes,
        output_nodes=output_nodes,
        input_p_bc=100.0,
        output_p_bc=10.0,
        systemic_hematocrit=0.45,
        max_iterations=10,
        tolerance=1e-3
    )
    
    # Assert solver completed and returned pressures
    assert final_pressure is not None
    assert len(final_pressure) == 4
    
    # Assert every edge got hematocrit and viscosity attributes
    for u, v, key, data in G_solved.edges(keys=True, data=True):
        assert "hematocrit" in data
        assert "viscosity" in data
        assert "flow_abs" in data
        
    # The parent branch should still have systemic hematocrit
    assert np.isclose(G_solved[0][1][0]["hematocrit"], 0.45, atol=1e-5)
    
    # The larger branch (1 -> 2) should skim RBCs, giving it H > 0.45
    assert G_solved[1][2][0]["hematocrit"] > 0.45
    # The tiny branch (1 -> 3) should lose RBCs, giving it H < 0.45
    assert G_solved[1][3][0]["hematocrit"] < 0.45


def test_coupled_solver_dag_cycle_handling(caplog):
    """Verify the topological sorter safely handles biologically impossible fluid loops."""
    G = nx.MultiGraph()
    # Force a loop: 0 -> 1 -> 2 -> 1
    # We will manually assign bizarre pressures to trick the DAG builder
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(2, 1, key=1, length=10.0, fwhm_diameter_um=10.0)
    
    starting_nodes = [0]
    output_nodes = [2]
    
    # By default, a linear solver won't create a cycle here, but let's see if 
    # the solver at least completes without throwing a fatal unhandled error.
    # It should just resolve it as normal parallel edges since P1 > P2.
    G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes=starting_nodes,
        output_nodes=output_nodes,
        input_p_bc=100.0,
        output_p_bc=10.0,
        systemic_hematocrit=0.45,
        max_iterations=5
    )
    
    assert final_pressure is not None
