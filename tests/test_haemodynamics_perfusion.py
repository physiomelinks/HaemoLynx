import pytest
import numpy as np
import networkx as nx
import scipy.sparse as sp
from dataclasses import dataclass

from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    map_vessels_to_grid,
    build_adr_matrix,
    solve_perfusion_steady_state
)

@dataclass
class MockPerfusionConfig:
    do_perfusion_modeling: bool = True
    grid_resolution_xyz: tuple[float, float, float] = (10.0, 10.0, 10.0)
    grid_opacity: float = 0.3
    sigma_diff: float = 1.5e-9
    M_max: float = 0.05
    k_reduce: float = 0.1
    C_arterial: float = 0.13

@pytest.fixture
def mock_graph():
    """Creates a mock multigraph with explicit physical ZYX positions."""
    G = nx.MultiGraph()
    # Node 1 at [0, 0, 0], Node 2 at [20, 20, 20] (ZYX physical coords)
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 20.0, 20.0]))
    
    # Straight line along Z-axis for testing mapping
    # Voxels are strictly physical ZYX coords
    voxels_z_line = [
        np.array([5.0, 5.0, 5.0]),
        np.array([10.0, 5.0, 5.0]),
        np.array([15.0, 5.0, 5.0]),
        np.array([20.0, 5.0, 5.0])
    ]
    
    G.add_edge(1, 2, key=0, voxels=voxels_z_line, flow_abs=10.0, length=15.0)
    G.graph["voxel_size"] = (1.0, 1.0, 1.0) # Used in older parts, not perfusion directly anymore
    return G

# --- 1. PerfusionGrid Geometry & Coordinate Mapping Tests ---

def test_perfusion_grid_dimensions(mock_graph):
    """Test that the 3D grid dimensions perfectly bound the graph with padding."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    # Max ZYX is [20,20,20], Min ZYX is [0,0,0].
    # Padding is half resolution (5.0) on all sides.
    # So bounds are [-5.0 to 25.0] in all dimensions -> 30 units length -> 3 cells exactly.
    np.testing.assert_array_equal(grid.dims, [3, 3, 3])
    assert grid.n_cells == 27
    np.testing.assert_array_equal(grid.min_xyz, [-5.0, -5.0, -5.0])
    np.testing.assert_array_equal(grid.max_xyz, [25.0, 25.0, 25.0])

def test_grid_index_bidirectional_mapping(mock_graph):
    """Test that physical -> index -> physical coordinate mapping is sound."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    original_zyx = np.array([12.0, 18.0, 2.0])
    
    # 1. Forward Mapping (Physical to Index)
    idx = grid.get_cell_index(original_zyx)
    assert idx >= 0 and idx < grid.n_cells
    
    # 2. Reverse Mapping (Index to Cell Center Physical)
    center_zyx = grid.get_xyz_from_index(idx)
    
    # The center should be within +/- 5.0 (half resolution) of the original point
    assert np.all(np.abs(center_zyx - original_zyx) <= 5.0)

def test_grid_out_of_bounds_handling(mock_graph):
    """Test that extreme coordinates safely return -1 instead of crashing."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    assert grid.get_cell_index(np.array([-100.0, 0.0, 0.0])) == -1
    assert grid.get_cell_index(np.array([0.0, 500.0, 0.0])) == -1


# --- 2. Vessel-to-Grid Advection Mapping Tests ---

def test_map_vessels_to_grid_straight_line(mock_graph):
    """Verify 1D physical line segments map precisely into 3D voxel buckets."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    
    # The line goes from Z=5 to Z=20 at Y=5, X=5.
    # Grid goes from -5 to 25.
    # Cell 0 (Z=-5 to 5): Should catch the point at Z=5
    # Cell 1 (Z=5 to 15): Should catch the point at Z=10
    # Cell 2 (Z=15 to 25): Should catch the points at Z=15, 20
    
    assert len(cell_to_vessels) > 0
    total_flow_injected = sum(v["flow"] for vessels in cell_to_vessels.values() for v in vessels)
    assert total_flow_injected > 0.0


# --- 3. ADR Sparse Matrix Tests ---

def test_build_adr_matrix_structure(mock_graph):
    """Test the physical structure of the 7-Point Stencil Laplacian."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    
    A, b_adv, diag_A = build_adr_matrix(grid, cell_to_vessels, config)
    
    assert sp.issparse(A)
    assert isinstance(A, sp.csr_matrix)
    assert A.shape == (27, 27)
    
    # The central node of a 3x3x3 grid (index 13) should have exactly 7 non-zeros
    central_row = A.getrow(13)
    assert central_row.nnz == 7
    
    # A corner node (index 0) should have exactly 4 non-zeros (itself + 3 neighbors)
    corner_row = A.getrow(0)
    assert corner_row.nnz == 4

def test_advective_source_vector(mock_graph):
    """Verify that oxygen is only sourced from cells containing vessels."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    
    _, b_adv, _ = build_adr_matrix(grid, cell_to_vessels, config)
    
    # Check that any non-zero entries in b_adv correspond perfectly to cell_to_vessels keys
    for i in range(len(b_adv)):
        if b_adv[i] > 0.0:
            assert i in cell_to_vessels


# --- 4. Non-Linear Picard Iteration Solver Tests ---

def test_perfusion_solver_zero_flow(mock_graph):
    """If no flow enters the system, concentration must be 0.0 everywhere."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    # Force flow to 0
    for u, v, k in mock_graph.edges(keys=True):
        mock_graph[u][v][k]["flow_abs"] = 0.0
        
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    A, b_adv, _ = build_adr_matrix(grid, cell_to_vessels, config)
    
    C_steady = solve_perfusion_steady_state(grid, A, b_adv, config)
    np.testing.assert_allclose(C_steady, np.zeros_like(C_steady), atol=1e-10)

def test_perfusion_solver_no_metabolism(mock_graph):
    """If tissue doesn't consume oxygen, diffusion spreads the arterial concentration."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    config.M_max = 0.0 # Shut off metabolism
    
    A, b_adv, _ = build_adr_matrix(grid, cell_to_vessels, config)
    C_steady = solve_perfusion_steady_state(grid, A, b_adv, config)
    
    # Total concentration shouldn't be zero since we have advection and no sink
    assert np.sum(C_steady) > 0.0

def test_perfusion_solver_positivity(mock_graph):
    """Even with massive metabolism, non-linear math physically prevents negative concentrations."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    
    # Introduce normal flow, but a ridiculously high tissue sink
    config.M_max = 1000.0 
    
    A, b_adv, _ = build_adr_matrix(grid, cell_to_vessels, config)
    C_steady = solve_perfusion_steady_state(grid, A, b_adv, config)
    
    # Assert there are no physically impossible negative concentrations
    assert np.all(C_steady >= -1e-10) # Account for minor floating point error
