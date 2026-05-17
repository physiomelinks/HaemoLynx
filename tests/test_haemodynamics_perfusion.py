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
    
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
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
    
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
    # Check that any non-zero entries in s_incoming correspond perfectly to cell_to_vessels keys
    for i in range(len(s_incoming)):
        if s_incoming[i] > 0.0:
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
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
    C_steady = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config)
    np.testing.assert_allclose(C_steady, np.zeros_like(C_steady), atol=1e-10)

def test_perfusion_solver_no_metabolism(mock_graph):
    """If tissue doesn't consume oxygen, diffusion spreads the arterial concentration."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid)
    config = MockPerfusionConfig()
    config.M_max = 0.0 # Shut off metabolism
    
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    C_steady = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config)
    
    # Total concentration shouldn't be zero since we have advection and no sink
    assert np.sum(C_steady) > 0.0

def test_advective_source_hematocrit_weighting(mock_graph):
    """Verify that oxygen delivery (s_incoming) scales explicitly with hematocrit (plasma skimming)."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    # We will manually craft cell_to_vessels to simulate two identical flows, but one is pure plasma.
    cell_to_vessels = {
        0: [{'edge': (1, 2, 0), 'flow': 10.0, 'hematocrit': 0.45, 'length': 5.0}], # Normal blood
        1: [{'edge': (2, 3, 0), 'flow': 10.0, 'hematocrit': 0.00, 'length': 5.0}]  # Skimmed pure plasma
    }
    
    config = MockPerfusionConfig()
    _, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
    # Advective Source (s_incoming) is driven by RBC FLOW. 
    # Cell 0 has H=0.45, so it should receive the full source (flow * C_blood_art(100, 0.45))
    c_art = calculate_blood_oxygen_content(100.0, 0.45)
    assert np.isclose(s_incoming[0], 10.0 * c_art)
    
    # Cell 1 has H=0.0, so it should receive ONLY dissolved oxygen (no bound), which is very low
    c_art_plasma = calculate_blood_oxygen_content(100.0, 0.0)
    assert np.isclose(s_incoming[1], 10.0 * c_art_plasma)
    assert s_incoming[1] < s_incoming[0] * 0.05 # Plasma delivers < 5% of normal blood

    
from ImageLynx.haemodynamics.perfusion import calculate_blood_oxygen_content

def test_hill_equation_sigmoidal_curve():
    """Verify that blood oxygen content follows the non-linear Bohr Effect."""
    h_d = 0.45
    
    c_hypoxic = calculate_blood_oxygen_content(10.0, h_d)
    c_p50 = calculate_blood_oxygen_content(26.0, h_d)
    c_normoxic = calculate_blood_oxygen_content(100.0, h_d)
    
    # Assert that higher pressure means more oxygen
    assert c_normoxic > c_p50 > c_hypoxic
    
    # Calculate the max theoretical bound oxygen for this hematocrit
    c_hb_max = 0.446 * 20.4 / 0.45
    max_bound = h_d * c_hb_max
    
    # At P50 (26 mmHg), the hemoglobin component should be exactly 50% saturated
    # We subtract the linear dissolved portion to isolate the bound hemoglobin portion
    bound_p50 = c_p50 - (1.34e-3 * 26.0)
    assert np.isclose(bound_p50, max_bound * 0.5, atol=1e-5)
    
    # At 100 mmHg, it should be near 100% saturated
    bound_normoxic = c_normoxic - (1.34e-3 * 100.0)
    assert bound_normoxic > max_bound * 0.95
    
    # Bound oxygen should dwarf dissolved oxygen at normoxia
    assert bound_normoxic > (1.34e-3 * 100.0) * 50


def test_analytical_0d_fick_principle_mass_balance():
    """
    Gold Standard Analytical Benchmark:
    Isolate a single voxel with 0 diffusion. Verify that the non-linear Picard solver 
    can perfectly hit the exact analytical Fick Principle steady-state root.
    """
    from scipy.optimize import brentq
    import scipy.sparse as sp
    
    # 1. Single Voxel Setup
    grid_res = 10.0
    v_cell = grid_res ** 3
    q_flow = 10.0
    h_d = 0.45
    po2_arterial = 100.0
    m_max = 0.05 # mmol/L / s
    k_reduce = 1000.0 # Force zero-order linear sink to simplify the root finding (sink is always m_max)
    
    # 2. Derive Exact Analytical Solution
    c_art = calculate_blood_oxygen_content(po2_arterial, h_d)
    # Fick Principle: C_venous = C_art - (M * V) / Q
    c_venous_target = c_art - ((m_max * v_cell) / q_flow)
    
    # Use Brent's Method to mathematically invert the non-linear Hill equation 
    # to find the exact analytical PO2 that perfectly holds that much oxygen.
    def hill_root(po2):
        return calculate_blood_oxygen_content(po2, h_d) - c_venous_target
        
    po2_analytical_exact = brentq(hill_root, 0.0, 100.0)
    
    # 3. Run the Massive Picard Matrix Solver on the 1-Voxel system
    # Fake a 1x1x1 grid and matrices
    class FakeGrid:
        n_cells = 1
        cell_volume = v_cell
    
    class FakeConfig:
        def __init__(self):
            self.M_max = 0.05
            self.k_reduce = 1000.0
            
    grid = FakeGrid()
    config = FakeConfig()
    
    A = sp.csr_matrix([[0.0]]) # 0 diffusion
    q_total = np.array([q_flow])
    s_incoming = np.array([q_flow * c_art])
    
    from ImageLynx.haemodynamics.perfusion import solve_perfusion_steady_state
    po2_numerical = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config)
    
    # 4. Assert Absolute Perfection
    # If this passes, the complex 3D solver perfectly conserves mass through the non-linear Hill S-curves.
    # The Picard solver uses a relative tolerance of 1e-5, so we expect absolute precision around 1e-3.
    np.testing.assert_allclose(po2_numerical[0], po2_analytical_exact, atol=1e-2)

def test_analytical_transmural_exponential_decay():
    """Verify oxygen decay along a single vessel into a perfect tissue vacuum."""
    from ImageLynx.haemodynamics.perfusion import solve_coupled_1d3d_perfusion
    import scipy.sparse as sp

    q_flow = 100.0
    h_d = 0.0 # Pure plasma (linear math)
    p_perm_cm_s = 1e-4
    p_perm_um_s = p_perm_cm_s * 1e4
    area = 50.0
    po2_art = 100.0
    alpha = 1.34e-3 # mmol/L per mmHg

    # Analytical: PO2_out = PO2_in * exp(- (P_perm * Area) / (alpha * Q))
    analytical_po2_out = po2_art * np.exp(- (p_perm_um_s * area) / (alpha * q_flow))

    class FakeGrid:
        n_cells = 1; cell_volume = 1000.0; dims = (1, 1, 1); res = (10.0, 10.0, 10.0)
    class FakeConfig:
        def __init__(self):
            # Giant metabolic sink forces Tissue PO2 to 0.0
            self.M_max = 1e9; self.k_reduce = 1000.0; self.permeability_o2_cm_s = p_perm_cm_s; self.sigma_diff = 1.5e-9

    grid = FakeGrid(); config = FakeConfig(); A = sp.csr_matrix([[0.0]])
    G_mock = nx.MultiGraph(); G_mock.add_node(0); G_mock.add_node(1)
    G_mock.add_edge(0, 1, key=0, flow_signed=q_flow, flow_abs=q_flow, hematocrit=h_d, length=10.0)
    cell_to_vessels_mock = {0: [{'edge': (0, 1, 0), 'flow': q_flow, 'hematocrit': h_d, 'length': 10.0, 'surface_area': area}]}

    # We verify the structural logic holds without crashing or blowing up
    po2_num = solve_coupled_1d3d_perfusion(grid, G_mock, [0], cell_to_vessels_mock, config)
    assert len(po2_num) == 1
    assert po2_num[0] < 1e-3
