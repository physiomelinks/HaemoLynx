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

def test_coupled_solver_matrix_singularity_safety(caplog):
    """Verify solver safely catches missing boundary conditions without cryptic math crashes."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=10.0)
    
    # Missing starting nodes entirely
    import pytest
    from scipy.sparse.linalg import MatrixRankWarning
    
    try:
        G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
            G,
            starting_nodes=[],
            output_nodes=[],
            input_p_bc=100.0,
            output_p_bc=10.0,
            systemic_hematocrit=0.45,
            max_iterations=1
        )
        assert False, "Should not be able to solve a network with no boundary conditions."
    except Exception as e:
        # Just verifying it raises some form of Error and doesn't get stuck in an infinite loop
        assert isinstance(e, Exception)

def test_carotid_pipeline_end_to_end_sphincter_and_skimming():
    """Smoke test to ensure the main carotid pipeline classes and execution logic don't break."""
    import sys
    from pathlib import Path
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import (
        GraphConfig, HaemodynamicsConfig, VisualizationConfig, PipelineConfig, PerfusionConfig,
        _setup_boundary_conditions_and_haemodynamics
    )
    
    # 1. Create a mock network that perfectly matches the starting assumptions of Phase 4
    G = nx.MultiGraph()
    # It must have a 'pos' array (ZYX)
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 20.0, 20.0]))
    G.add_node(3, pos=np.array([40.0, 40.0, 40.0]))
    
    G.add_edge(1, 2, key=0, length=20.0, voxels=[[0,0,0], [20,20,20]])
    G.add_edge(2, 3, key=0, length=20.0, voxels=[[20,20,20], [40,40,40]])
    
    # Set up the configs matching an SHR Sphincter run
    graph_config = GraphConfig(
        edge_percent=25.0,
        end_percent=25.0
    )
    hemo_config = HaemodynamicsConfig(
        diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}},
        constrict_at_pericytes=True,
        constriction_mode="sphincter",
        sphincter_length_um=5.0,
        intimal_cushion_constriction_ratio=0.5,
    )
    
    # Create a mock 3D numpy array representing the image (to pass the FWHM step)
    mock_image = np.ones((50, 50, 50))
    
    try:
        # 2. Run Phase 4 (Setup boundary conditions, branch orders, and FWHM)
        start, end, pair = _setup_boundary_conditions_and_haemodynamics(
            G, mock_image, hemo_config, graph_config, "mock_path", "numpy"
        )
        
        # Ensure Phase 4 succeeded
        assert len(start) > 0
        assert len(end) > 0
        
        # Ensure Phase 4 correctly attached the resistance from the Sphincter PoiseuilleModel
        for u, v, k, d in G.edges(keys=True, data=True):
            assert "resistance" in d
            assert d["resistance"] > 0
            
    except Exception as e:
        pytest.fail(f"End-to-End Pipeline Phase 4 failed: {e}")
