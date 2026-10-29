import pytest
import numpy as np
import networkx as nx
from dataclasses import dataclass

from ImageLynx.haemodynamics.perfusion import (
    ALPHA_O2_MMOL_PER_L_MMHG,
    PerfusionGrid,
    map_vessels_to_grid,
    build_adr_matrix,
    solve_perfusion_steady_state,
    calculate_blood_oxygen_content,
    cell_discharge_hematocrit,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S

@dataclass
class MockPerfusionConfig:
    sigma_diff: float = 1.5e-9
    M_max: float = 0.05
    k_reduce: float = 0.1
    po2_arterial_mmHg: float = 100.0
    systemic_hematocrit: float = 0.45
    #: Solver stop settings the tiers now read, at Tier 1's old hard-coded values.
    picard_max_iterations: int = 50
    picard_tolerance: float = 1e-5

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
    
    G.add_edge(1, 2, key=0, voxels=voxels_z_line, flow_abs=10.0, length=15.0, hematocrit=0.45)
    return G

# --- 1. PerfusionGrid Geometry & Coordinate Mapping Tests ---

@pytest.mark.parametrize("res, n_per_axis", [(10.0, 3), (4.0, 6)])
def test_perfusion_grid_dimensions(mock_graph, res, n_per_axis):
    """With no bounds asked for, the grid is the vasculature padded by half a cell.

    The default is unchanged, because changing it would move every existing result.
    """
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(res, res, res))

    # Max ZYX is [20,20,20], Min ZYX is [0,0,0], padded by half a cell on all sides:
    # [-5, 25] at 10 um is 3 cells, [-2, 22] at 4 um is 6.
    np.testing.assert_array_equal(grid.dims, [n_per_axis] * 3)
    assert grid.n_cells == n_per_axis ** 3
    np.testing.assert_array_equal(grid.min_xyz, [-res / 2] * 3)
    np.testing.assert_array_equal(grid.max_xyz, [20.0 + res / 2] * 3)
    np.testing.assert_array_equal(grid.min_xyz + grid.dims * grid.res, [20.0 + res / 2] * 3)

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
    # The fixture's points sit on cell faces in y and x, and at z = 5 and 15. This line keeps
    # at least 2 um off every face, so the expected cells don't depend on how a face point
    # rounds.
    line = [np.array([z, 7.0, 7.0]) for z in (7.0, 12.0, 17.0, 22.0)]
    mock_graph[1][2][0]["voxels"] = line
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    # The fixture graph carries no diameter. map_vessels_to_grid used to
    # substitute 5.0 um silently; naming it keeps these tests about the grid
    # mapping rather than about calibre.
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)

    # The grid runs from -5 to 25 um in 10 um cells, so the line is in the middle cell in y
    # and x, and in z it crosses from the middle cell (z = 7, 12) into the top one (17, 22).
    # Linear index z + 3 y + 9 x: 1 + 3 + 9 = 13 and 2 + 3 + 9 = 14.
    assert sorted(cell_to_vessels) == [13, 14]
    assert [grid.get_cell_index(p) for p in line] == [13, 13, 14, 14]
    for idx in (13, 14):
        (entry,) = cell_to_vessels[idx]
        assert entry["edge"] == (1, 2, 0)
        assert entry["length_fraction"] == pytest.approx(0.5, rel=1e-12)
        # Each cell records the edge's whole flow, converted to um^3/s; the share is the
        # length fraction above.
        assert entry["flow"] == pytest.approx(10.0 * POISEUILLE_FLOW_TO_UM3_PER_S, rel=1e-12)


# --- 3. Non-Linear Picard Iteration Solver Tests ---

def test_perfusion_solver_zero_flow(mock_graph):
    """If no flow enters the system, concentration must be 0.0 everywhere."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    # Force flow to 0
    for u, v, k in mock_graph.edges(keys=True):
        mock_graph[u][v][k]["flow_abs"] = 0.0
        
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
    config = MockPerfusionConfig()
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    
    C_steady = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config,
                                            cell_hematocrit=cell_discharge_hematocrit(cell_to_vessels, grid.n_cells))
    np.testing.assert_allclose(C_steady, np.zeros_like(C_steady), atol=1e-10)

def test_perfusion_solver_no_metabolism(mock_graph):
    """If tissue doesn't consume oxygen, diffusion spreads the arterial PO2 to every cell."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    cell_to_vessels = map_vessels_to_grid(mock_graph, grid, default_diameter_um=5.0)
    config = MockPerfusionConfig()
    config.M_max = 0.0 # Shut off metabolism
    
    A, q_total, s_incoming = build_adr_matrix(grid, cell_to_vessels, config)
    C_steady = solve_perfusion_steady_state(grid, A, q_total, s_incoming, config,
                                            cell_hematocrit=cell_discharge_hematocrit(cell_to_vessels, grid.n_cells))

    # With no sink, every cell, perfused or not, settles at the arterial PO2.
    np.testing.assert_allclose(C_steady, config.po2_arterial_mmHg, rtol=0, atol=1e-3)

def test_advective_source_hematocrit_weighting(mock_graph):
    """Verify that oxygen delivery (s_incoming) scales explicitly with hematocrit (plasma skimming)."""
    grid = PerfusionGrid(mock_graph, grid_resolution_xyz=(10.0, 10.0, 10.0))
    
    # We will manually craft cell_to_vessels to simulate two identical flows, but one is pure plasma.
    cell_to_vessels = {
        # Each edge lies wholly in its one cell, so its whole flow is shared there.
        0: [{'edge': (1, 2, 0), 'flow': 10.0, 'hematocrit': 0.45, 'length': 5.0,
             'length_fraction': 1.0}], # Normal blood
        1: [{'edge': (2, 3, 0), 'flow': 10.0, 'hematocrit': 0.00, 'length': 5.0,
             'length_fraction': 1.0}]  # Skimmed pure plasma
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
    # Oxygen is sourced only in cells holding a vessel.
    assert not s_incoming[2:].any()


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


# --- 4. ADR Sparse Matrix Tests ---

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
    class _Config:
        sigma_diff = 1.5e-9
        M_max = 0.005
        k_reduce = 0.1
        po2_arterial_mmHg = 100.0
        picard_max_iterations = 5
        picard_tolerance = 1e-4

    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.array([40.0, 60.0, 140.0]))
    graph.add_edge(0, 1, key=0, length=10.0, radius=2.0, flow=1.0)

    grid = PerfusionGrid(graph, (5.0, 10.0, 20.0))
    n_z, n_y, n_x = grid.dims
    assert (n_z, n_y, n_x) != (n_x, n_y, n_z), "grid must be non-cubic for this test to bite"

    matrix = build_adr_matrix(grid, {}, _Config())[0]
    res = grid.res
    # D alpha: the conductance takes a PO2 difference to an O2 flux.
    sigma = _Config.sigma_diff * 1e12 * ALPHA_O2_MMOL_PER_L_MMHG
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

    # A corner cell has three neighbours, so its row holds four non-zeros with the diagonal.
    assert matrix[linear_index(0, 0, 0)].nnz == 4


# --- Grid extent: padding past the vasculature ---

def _two_node_graph():
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array([0.0, 0.0, 0.0]))
    G.add_node("b", pos=np.array([20.0, 20.0, 20.0]))
    return G


def test_requested_bounds_extend_the_grid_to_cover_them():
    """A grid that reaches the segmented tissue, not just the vessels."""
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
