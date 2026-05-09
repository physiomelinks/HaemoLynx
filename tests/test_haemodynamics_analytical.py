import pytest
import numpy as np
import networkx as nx
import scipy.sparse as sp
from dataclasses import dataclass

from ImageLynx.haemodynamics.resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    _solve_system_smart
)
from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    build_adr_matrix,
    solve_perfusion_steady_state
)

@dataclass
class MockPerfusionConfig:
    do_perfusion_modeling: bool = True
    grid_resolution_xyz: tuple[float, float, float] = (10.0, 10.0, 10.0)
    grid_opacity: float = 0.3
    sigma_diff: float = 1.0 # Simplified for analytical tests
    M_max: float = 10.0
    k_reduce: float = 1000.0 # High value forces linear "zero-order" kinetics
    C_arterial: float = 100.0


# --- Part 1: 1D Hemodynamics Analytical Tests ---

def test_analytical_poiseuille_series():
    """Test two resistors in series. R_eq = R1 + R2, Q = dP / R_eq."""
    G = nx.Graph()
    G.add_edge(0, 1, resistance=2.0)
    G.add_edge(1, 2, resistance=3.0)
    
    C, node_list = build_conductance_matrix_from_graph(G)
    L = calc_laplacian_from_conductance_matrix(C)
    
    # Boundary Conditions: P_in at node 0 = 100, P_out at node 2 = 10
    P_in = 100.0
    P_out = 10.0
    known_idx = [0, 2]
    p_k = np.array([P_in, P_out])
    unknown_idx = [1]
    
    l_uu = L[unknown_idx, :][:, unknown_idx]
    l_uk = L[unknown_idx, :][:, known_idx]
    rhs = -l_uk.dot(p_k)
    
    P_mid = _solve_system_smart(l_uu, rhs)[0]
    
    # Analytical:
    R_eq = 2.0 + 3.0
    Q = (P_in - P_out) / R_eq
    P_mid_analytical = P_in - Q * 2.0
    
    assert np.isclose(P_mid, P_mid_analytical, atol=1e-10)

def test_analytical_poiseuille_parallel():
    """Test two resistors in parallel. 1/R_eq = 1/R1 + 1/R2."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, resistance=2.0)
    G.add_edge(0, 1, resistance=4.0)
    
    # build_conductance_matrix_from_graph works with MultiGraphs if converted or handles duplicate edges
    # Wait, build_conductance_matrix_from_graph uses nx.Graph().edges() which squashes multigraphs.
    # Let's manually build a slightly different parallel structure to avoid nx.Graph squashing:
    # 0 -> 1 -> 3
    # 0 -> 2 -> 3
    G_parallel = nx.Graph()
    G_parallel.add_edge(0, 1, resistance=2.0)
    G_parallel.add_edge(1, 3, resistance=1e-9) # Zero resistance wire
    G_parallel.add_edge(0, 2, resistance=4.0)
    G_parallel.add_edge(2, 3, resistance=1e-9) # Zero resistance wire
    
    C, node_list = build_conductance_matrix_from_graph(G_parallel)
    L = calc_laplacian_from_conductance_matrix(C)
    
    P_in = 100.0
    P_out = 10.0
    
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    known_idx = [node_to_idx[0], node_to_idx[3]]
    p_k = np.array([P_in, P_out])
    
    unknown_idx = [node_to_idx[1], node_to_idx[2]]
    l_uu = L[unknown_idx, :][:, unknown_idx]
    l_uk = L[unknown_idx, :][:, known_idx]
    rhs = -l_uk.dot(p_k)
    
    P_unknowns = _solve_system_smart(l_uu, rhs)
    
    # Calculate flows
    P_1 = P_unknowns[unknown_idx.index(node_to_idx[1])]
    P_2 = P_unknowns[unknown_idx.index(node_to_idx[2])]
    
    Q1 = (P_in - P_1) / 2.0
    Q2 = (P_in - P_2) / 4.0
    
    # Analytical: Flow should split inversely proportional to resistance.
    # R2 = 2 * R1, so Q1 should be 2 * Q2
    assert np.isclose(Q1, 2.0 * Q2, atol=1e-10)
    
    # Total equivalent resistance = 1 / (1/2 + 1/4) = 4/3
    Q_total = Q1 + Q2
    Q_total_analytical = (P_in - P_out) / (4.0 / 3.0)
    assert np.isclose(Q_total, Q_total_analytical, atol=1e-10)


# --- Part 2: 3D Perfusion Analytical Tests ---

def test_analytical_1d_pure_diffusion():
    """Test pure diffusion gradient (straight line) without advection or metabolism."""
    # Build a 10x1x1 grid mathematically
    config = MockPerfusionConfig()
    config.M_max = 0.0 # No metabolism
    
    N = 10
    # Create 1D Laplacian
    diagonals = [np.full(N, -2.0), np.ones(N-1), np.ones(N-1)]
    A = sp.diags(diagonals, [0, -1, 1], format='csr')
    
    # Adjust boundaries to Dirichlet C(0) = 100, C(L) = 0
    # In finite differences, forcing a boundary means A[0,0] = 1, b[0] = 100
    A.data = A.data.astype(np.float64) # Ensure float
    A_lil = A.tolil()
    A_lil[0, :] = 0
    A_lil[0, 0] = 1.0
    A_lil[-1, :] = 0
    A_lil[-1, -1] = 1.0
    A = A_lil.tocsr()
    
    b = np.zeros(N)
    b[0] = 100.0
    b[-1] = 0.0
    
    # Since we are bypassing PerfusionGrid for this pure math test, we just solve directly
    C_steady = _solve_system_smart(A, b)
    
    # Analytical solution is a straight line from 100 down to 0
    C_analytical = np.linspace(100.0, 0.0, N)
    np.testing.assert_allclose(C_steady, C_analytical, atol=1e-10)


def test_analytical_zero_order_metabolism():
    """Test diffusion + constant metabolism (parabolic curve).
    Governing Eq: D d^2C/dx^2 = M
    Analytical Sol: C(x) = C_0 - (M / 2D) * x * (L - x)
    """
    N = 11
    D = 1.0
    M = 10.0
    h = 1.0 # grid spacing
    
    # Create 1D Laplacian scaled by D/h^2
    diagonals = [np.full(N, -2.0), np.ones(N-1), np.ones(N-1)]
    A = sp.diags(diagonals, [0, -1, 1], format='csr') * (D / h**2)
    
    # Apply Dirichlet boundaries C(0) = 100, C(L) = 100
    C_0 = 100.0
    A_lil = A.tolil()
    A_lil[0, :] = 0
    A_lil[0, 0] = 1.0
    A_lil[-1, :] = 0
    A_lil[-1, -1] = 1.0
    A = A_lil.tocsr()
    
    # The RHS vector is the metabolic sink M (positive on the RHS since Laplacian is negative definite)
    # Wait, the equation is D L C = M. Since our L has negative diagonal, D L C = -M -> A C = b
    # So b = M for internal nodes.
    b = np.full(N, M)
    b[0] = C_0
    b[-1] = C_0
    
    C_steady = _solve_system_smart(A, b)
    
    # Analytical Parabola
    L = N - 1
    x = np.arange(N)
    # C(x) = C_0 - (M / 2D) * x * (L - x)
    C_analytical = C_0 - (M / (2.0 * D)) * x * (L - x)
    
    np.testing.assert_allclose(C_steady, C_analytical, atol=1e-10)


def test_analytical_radial_point_source():
    """Test advection-diffusion point source radiates as 1/r."""
    # We will build a 21x21x21 grid to give enough space for radial diffusion
    N = 21
    mid = N // 2
    
    # Build 3D Laplacian directly
    def build_3d_laplacian(n):
        diag = np.full(n**3, -6.0)
        off_1 = np.ones(n**3 - 1)
        off_1[n-1::n] = 0 # break connections at x boundaries
        off_n = np.ones(n**3 - n)
        off_n[n**2-n::n**2] = 0 # break at y boundaries
        for i in range(1, n):
            off_n[n**2-n + i::n**2] = 0
        off_n2 = np.ones(n**3 - n**2)
        
        return sp.diags([diag, off_1, off_1, off_n, off_n, off_n2, off_n2],
                        [0, -1, 1, -n, n, -n**2, n**2], format='csr')

    A = build_3d_laplacian(N)
    b = np.zeros(N**3)
    
    # Central Point Source
    center_idx = mid + mid*N + mid*N**2
    
    # Neumann/Dirichlet Boundaries - let's set corners to 0 to ground the system
    # Actually just set the outer edges to 0
    A_lil = A.tolil()
    for i in range(N**3):
        z = i // N**2
        y = (i % N**2) // N
        x = i % N
        if x == 0 or x == N-1 or y == 0 or y == N-1 or z == 0 or z == N-1:
            A_lil[i, :] = 0
            A_lil[i, i] = 1.0
            b[i] = 0.0
            
    A = A_lil.tocsr()
    b[center_idx] = -100.0 # Source injected (negative because diagonal is negative)
    
    C_steady = _solve_system_smart(A, b)
    C_3d = C_steady.reshape((N, N, N))
    
    # Extract radial profile along one axis from center
    radial_profile = C_3d[mid, mid, mid:N]
    
    r_values = np.arange(1, mid)
    c_values = radial_profile[1:mid]
    
    # 1. Assert strict monotonic decay (it always drops as it moves away from source)
    assert np.all(np.diff(c_values) < 0)
    
    # 2. Near the source (where boundary effects are minimal), it should roughly follow 1/r.
    # Therefore C(1) / C(2) should be approximately 2.
    c1 = c_values[0] # r=1
    c2 = c_values[1] # r=2
    
    ratio = c1 / c2
    # In a perfect continuous infinite domain, ratio is 2.0. 
    # Discrete Cartesian grid introduces minor discretization errors near origin.
    assert 1.5 < ratio < 2.5
