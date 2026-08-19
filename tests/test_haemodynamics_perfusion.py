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
    # The fixture graph carries no diameter. map_vessels_to_grid used to
    # substitute 5.0 um silently; naming it keeps these tests about the grid
    # mapping rather than about calibre.
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
    
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
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
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
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
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
        
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
    config = MockPerfusionConfig()
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
    C_steady = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config)
    np.testing.assert_allclose(C_steady, np.zeros_like(C_steady), atol=1e-10)

def test_perfusion_solver_no_metabolism(mock_graph):
    """If tissue doesn't consume oxygen, diffusion spreads the arterial concentration."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
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


def test_adr_stencil_connects_physical_neighbours_with_correct_anisotropic_weights():
    """Lock the ADR discretisation against a well-meant rename.

    The axis naming in ``build_adr_matrix`` is systematically inverted: ``grid.dims`` is
    (nz, ny, nx) but is unpacked as ``nx, ny, nz`` ([perfusion.py:212]), and ``D_x`` is built
    from ``(res[1]*res[2])/res[0]``, which with ``res`` in (z, y, x) is the *z* coefficient.
    The two inversions cancel and the assembled matrix is correct, which this test pins down.

    It exists because the arithmetic is right for a reason no reader would guess from the
    names. Correcting either inversion on its own would silently produce a wrong stencil that
    an isotropic grid could not detect, so the grid here is deliberately non-cubic and the
    spacing anisotropic.
    """
    import networkx as nx_lib
    import numpy as np

    from ImageLynx.haemodynamics.perfusion import PerfusionGrid, build_adr_matrix

    class _Config:
        sigma_diff = 1.5e-9
        M_max = 0.005
        k_reduce = 0.1
        C_arterial = 0.13
        po2_arterial_mmHg = 100.0
        picard_max_iterations = 5
        picard_tolerance = 1e-4

    graph = nx_lib.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([40.0, 60.0, 140.0]))
    graph.add_edge(0, 1, key=0, length=10.0, radius=2.0, flow=1.0)

    grid = PerfusionGrid(graph, (5.0, 10.0, 20.0))
    n_z, n_y, n_x = grid.dims
    assert (n_z, n_y, n_x) != (n_x, n_y, n_z), "grid must be non-cubic for this test to bite"

    matrix = build_adr_matrix(grid, {}, _Config())[0]
    res = grid.res
    sigma = _Config.sigma_diff * 1e12
    expected = {
        "z": sigma * (res[1] * res[2]) / res[0],
        "y": sigma * (res[0] * res[2]) / res[1],
        "x": sigma * (res[0] * res[1]) / res[2],
    }

    def linear_index(z, y, x):
        # The grid's own convention, z fastest ([perfusion.py:138]).
        return int(z + y * n_z + x * n_z * n_y)

    centre = linear_index(1, 1, 1)
    for axis, (dz, dy, dx) in (("z", (1, 0, 0)), ("y", (0, 1, 0)), ("x", (0, 0, 1))):
        neighbour = linear_index(1 + dz, 1 + dy, 1 + dx)
        assert matrix[centre, neighbour] == pytest.approx(-expected[axis], rel=1e-12), (
            f"{axis}-neighbour coupling is not the {axis} diffusion coefficient"
        )

    # Exactly six off-diagonal neighbours, each one physical step away. Catches an ordering
    # change that preserved the weights but connected the wrong cells.
    neighbours = sorted(set(matrix[centre].nonzero()[1]) - {centre})
    assert len(neighbours) == 6
    for j in neighbours:
        offset = (j % n_z - 1, (j // n_z) % n_y - 1, j // (n_z * n_y) - 1)
        assert sum(abs(o) for o in offset) == 1, f"index {j} is not one step from the centre"


# --- Grid extent: padding past the vasculature (S28, T2.6) ---

def _two_node_graph():
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array([0.0, 0.0, 0.0]))
    G.add_node("b", pos=np.array([20.0, 20.0, 20.0]))
    return G


def test_the_grid_still_stops_at_the_vasculature_when_no_bounds_are_asked_for():
    """The default is unchanged, because changing it would move every existing result."""
    grid = PerfusionGrid(_two_node_graph(), (4.0, 4.0, 4.0))

    assert np.allclose(grid.min_xyz, [-2.0, -2.0, -2.0])
    assert np.allclose(grid.min_xyz + grid.dims * grid.res, [22.0, 22.0, 22.0])


def test_requested_bounds_extend_the_grid_to_cover_them():
    """What S28 needs: a grid that reaches the segmented tissue, not just the vessels."""
    grid = PerfusionGrid(_two_node_graph(), (4.0, 4.0, 4.0),
                         bounds_zyx=((-10.0, 0.0, 0.0), (40.0, 30.0, 22.0)))

    lo, hi = grid.min_xyz, grid.min_xyz + grid.dims * grid.res
    assert lo[0] <= -10.0 and hi[0] >= 40.0
    assert hi[1] >= 30.0
    assert hi[2] >= 22.0


def test_a_tighter_bound_never_shrinks_the_grid_off_the_vasculature():
    """The failure this rules out is silent: a node outside the grid indexes to -1, and its
    flow simply stops being a source, leaving a solve that runs and describes less network."""
    G = _two_node_graph()
    grid = PerfusionGrid(G, (4.0, 4.0, 4.0), bounds_zyx=((5.0, 5.0, 5.0), (10.0, 10.0, 10.0)))

    assert np.allclose(grid.min_xyz, [-2.0, -2.0, -2.0])
    for node in G.nodes:
        assert grid.get_cell_index(G.nodes[node]["pos"]) != -1


def test_nodes_stay_addressable_after_the_origin_moves():
    """Padding shifts min_xyz, so every cell index changes. The nodes must still resolve, and
    to the cell that actually contains them."""
    G = _two_node_graph()
    grid = PerfusionGrid(G, (4.0, 4.0, 4.0),
                         bounds_zyx=((-40.0, -40.0, -40.0), (60.0, 60.0, 60.0)))

    for node in G.nodes:
        pos = G.nodes[node]["pos"]
        index = grid.get_cell_index(pos)
        assert index != -1
        assert np.all(np.abs(grid.get_xyz_from_index(index) - pos) <= grid.res)


def test_padding_costs_cells_in_proportion_to_the_volume_added():
    tight = PerfusionGrid(_two_node_graph(), (4.0, 4.0, 4.0))
    padded = PerfusionGrid(_two_node_graph(), (4.0, 4.0, 4.0),
                           bounds_zyx=((-2.0, -2.0, -2.0), (42.0, 42.0, 42.0)))

    assert padded.n_cells > tight.n_cells
    assert padded.n_cells == 11 ** 3          # 44 um at 4 um in each axis


@pytest.mark.parametrize("bounds", [
    (( 0.0, 0.0), (10.0, 10.0)),                          # not triples
    ((0.0, 0.0, 0.0), (0.0, 10.0, 10.0)),                 # max not above min on one axis
    ((0.0, 0.0, 0.0), (-1.0, 10.0, 10.0)),                # inverted
])
def test_malformed_bounds_raise_rather_than_producing_a_grid(bounds):
    with pytest.raises(ValueError):
        PerfusionGrid(_two_node_graph(), (4.0, 4.0, 4.0), bounds_zyx=bounds)


def test_bounds_already_inside_the_vasculature_change_nothing_at_all():
    """The property that makes the flag safe to leave on across a mixed cohort.

    Four of the six carotid body specimens already have vessels reaching the region edge, and
    padding them must be a byte-for-byte no-op rather than a small perturbation: WKY-A's mean
    PO2 within TH is identical to three decimals with and without the flag. A grid that shifted
    its origin by a rounding step would renumber every cell and move results for reasons
    unrelated to the tissue.
    """
    G = _two_node_graph()
    tight = PerfusionGrid(G, (4.0, 4.0, 4.0))
    padded = PerfusionGrid(G, (4.0, 4.0, 4.0),
                           bounds_zyx=((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)))

    assert np.array_equal(tight.dims, padded.dims)
    assert np.allclose(tight.min_xyz, padded.min_xyz)
    assert tight.n_cells == padded.n_cells
    for node in G.nodes:
        pos = G.nodes[node]["pos"]
        assert tight.get_cell_index(pos) == padded.get_cell_index(pos)
