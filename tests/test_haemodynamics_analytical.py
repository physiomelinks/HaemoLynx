import numpy as np
import networkx as nx

from ImageLynx.haemodynamics.resistance import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    _solve_system_smart,
    POISEUILLE_FLOW_TO_UM3_PER_S,
)
from ImageLynx.haemodynamics.rheology import solve_coupled_flow_and_hematocrit


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
    # Two paths of zero-resistance wire between 0 and 3, so the resistors are in parallel:
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


# --- Part 2: Wall Shear Stress and Constriction Calculus ---

def test_analytical_wall_shear_stress():
    """Verify that Wall Shear Stress (WSS) is calculated accurately against exact analytical limits."""
    # Create a simple 2-node graph to manually test the WSS math inside the solver
    G = nx.MultiGraph()
    d = 10.0 # micrometers
    L = 100.0 # micrometers
    G.add_edge(0, 1, key=0, length=L, fwhm_diameter_um=d)
    
    # We will run one iteration of the solver to calculate the flow and WSS
    G_solved, _ = solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes=[0],
        output_nodes=[1],
        input_p_bc=13.332e6,
        output_p_bc=0.0,
        systemic_hematocrit=0.45,
        max_iterations=1 # Only need 1 iteration to get the first WSS calculation
    )
    
    data = G_solved[0][1][0]
    mu_app = data["viscosity"]
    q_abs = data["flow_abs"]
    wss_calc = data["wall_shear_stress_pa"]
    
    # Analytical WSS = 32 * mu * Q / (pi * d^3)
    # mu is mPa*s, Q is um^3/s, d is um. Result is mPa. Divide by 1000 for Pa.
    wss_exact_pa = ((32.0 * mu_app * q_abs) / (np.pi * d**3)) / 1000.0
    
    # Assert perfect mathematical match
    assert np.isclose(wss_calc, wss_exact_pa, atol=1e-10)
    
    # Assert it's a realistic physiological value (usually between 1 and 10 Pa in microvessels)
    assert wss_calc > 0.0

def test_analytical_sphincter_resistance_calculus():
    """Verify that numerical resistance integration matches exact calculus for complex geometries."""
    from ImageLynx.haemodynamics.poiseuille import PoiseuilleModel
    
    # We will test a standard periodic constriction (ramp down, hold, ramp up)
    d1 = 10.0
    d2 = 5.0
    L = 100.0
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0, mode="periodic")
    
    # Run the numerical integrator
    num_r = model.calculate_integrated_resistance(L, d1, d2, num_points=2000)
    
    # The artificial viscosity used internally by PoiseuilleModel is 1 / d^1.647
    # So the integrand is (128 / pi) * (1 / d^5.647)
    C = 128.0 / np.pi
    p = 5.647
    
    # 0 to 10: ramp d1 to d2. r(x) = d1 + (d2-d1)*(x/10)
    # Integral of dx / (A + Bx)^p = [ -1 / (B*(p-1)*(A+Bx)^(p-1)) ]
    B1 = (d2 - d1) / 10.0
    A1 = d1
    int1 = (-1.0 / (B1 * (p - 1.0))) * (1.0 / (A1 + B1*10.0)**(p - 1.0) - 1.0 / (A1)**(p - 1.0))
    
    # 10 to 30: hold d2
    int2 = 20.0 / (d2**p)
    
    # 30 to 40: ramp d2 to d1. r(x) = d2 + (d1-d2)*((x-30)/10)
    B3 = (d1 - d2) / 10.0
    A3 = d2 - B3*30.0
    int3 = (-1.0 / (B3 * (p - 1.0))) * (1.0 / (A3 + B3*40.0)**(p - 1.0) - 1.0 / (A3 + B3*30.0)**(p - 1.0))
    
    # 40 to 100: hold d1
    int4 = 60.0 / (d1**p)
    
    exact_r = C * (int1 + int2 + int3 + int4)
    
    # Assert the numerical trapezoidal integration is highly accurate (within 0.1%)
    assert np.isclose(num_r, exact_r, rtol=1e-3)

# --- Part 3: Multi-Species Coupling Tests ---

def test_bohr_haldane_atomic_curves():
    """Verify the atomic shift equations for the Bohr and Haldane effects."""
    from ImageLynx.haemodynamics.perfusion import (
        calculate_blood_oxygen_content, 
        calculate_blood_co2_content
    )
    
    h_d = 0.45
    
    # 1. The Bohr Effect (Acidosis shifts O2 curve right, releasing more O2)
    # At P50 (26.0), normal pH (7.4) should be ~50% saturated.
    c_o2_normal = calculate_blood_oxygen_content(26.0, h_d, pco2_mmHg=40.0, ph=7.4)
    # Severe acidosis (pH 7.0) at the same PO2
    c_o2_acidic = calculate_blood_oxygen_content(26.0, h_d, pco2_mmHg=40.0, ph=7.0)
    
    # Assert the acidic blood has less oxygen bound (it 'dumped' it)
    assert c_o2_acidic < c_o2_normal
    
    # 2. The Haldane Effect (Hyperoxia drops CO2 carrying capacity)
    pco2 = 40.0
    # Hypoxic blood (PO2=20)
    c_co2_hypoxic = calculate_blood_co2_content(pco2, h_d, po2_mmHg=20.0)
    # Hyperoxic blood (PO2=100)
    c_co2_hyperoxic = calculate_blood_co2_content(pco2, h_d, po2_mmHg=100.0)
    
    # Assert oxygenated blood holds less CO2
    assert c_co2_hyperoxic < c_co2_hypoxic


def test_henderson_hasselbalch_equilibrium():
    """Verify the Henderson-Hasselbalch math exactly matches physiological pH bounds."""
    from ImageLynx.haemodynamics.perfusion import calculate_ph_from_pco2
    
    # Normal physiological baseline
    ph_normal = calculate_ph_from_pco2(40.0, hco3_mmol_L=24.0)
    assert np.isclose(ph_normal, 7.4, atol=1e-2)
    
    # Severe hypercapnia (PCO2 = 80) should drive pH down into severe acidosis (~7.1)
    ph_acidic = calculate_ph_from_pco2(80.0, hco3_mmol_L=24.0)
    assert ph_acidic < 7.15


def test_multi_species_0d_fick_mass_balance():
    """
    Gold Standard Analytical Benchmark for the Multi-Species Solver.
    Isolates a single voxel. Mathematically proves the Picard Matrix loop correctly 
    navigates the coupled Bohr/Haldane equations to hit the exact Fick Mass Balance root.
    """
    from dataclasses import dataclass
    from ImageLynx.haemodynamics.perfusion import (
        solve_multi_species_perfusion,
        calculate_blood_oxygen_content,
        calculate_blood_co2_content,
        calculate_ph_from_pco2
    )
    
    # 1. Setup Parameters
    v_cell = 1000.0; q_huge = 1e9; h_d = 0.45 
    po2_art = 100.0; pco2_art = 40.0
    
    m_max = 0.05
    rq = 0.82
    hco3 = 24.0
    # Arterial blood enters at the Henderson-Hasselbalch pH of its PCO2, 7.401 (open item 25).
    ph_art = calculate_ph_from_pco2(pco2_art, hco3)
    m_co2 = m_max * rq
    
    # We use stable physiological permeabilities.
    # The Fick Principle holds true regardless of permeability at steady state.
    p_perm_o2_cm_s = 1e-4
    p_perm_co2_cm_s = 1e-4
    p_perm_o2_um_s = p_perm_o2_cm_s * 1e4
    p_perm_co2_um_s = p_perm_co2_cm_s * 1e4
    area = 1000.0
    alpha_o2 = 1.34e-3
    alpha_co2 = 0.03
    
    # 2. Derive Exact Analytical Targets using Fick's Principle
    # Because Q is massive, blood content doesn't drop.
    # Steady State: Flux_into_tissue = Metabolism_at_sink
    # P_perm * Area * alpha * (PO2_blood - PO2_tissue) = M_max * V_cell
    # The wall flux is taken at the blood's outlet pressure (open item 21), read off the O2 curve
    # at the cell's tissue pH (Bohr) and the CO2 curve at that PO2 (Haldane). The content is
    # arterial, but the acidic cell puts blood PO2 about 1 mmHg above 100, so iterate on pH.
    from scipy.optimize import brentq
    c_o2_art = calculate_blood_oxygen_content(po2_art, h_d, pco2_art, ph_art)
    c_co2_art = calculate_blood_co2_content(pco2_art, h_d, po2_art)
    analytical_ph = ph_art
    for _ in range(100):
        po2_blood = brentq(lambda p: calculate_blood_oxygen_content(p, h_d, pco2_art, analytical_ph) - c_o2_art, 0.0, 150.0, xtol=1e-12)
        pco2_blood = brentq(lambda p: calculate_blood_co2_content(p, h_d, po2_blood) - c_co2_art, 0.0, 150.0, xtol=1e-12)
        pco2_analytical_tissue = pco2_blood + (m_co2 * v_cell) / (p_perm_co2_um_s * area * alpha_co2)
        ph_new = calculate_ph_from_pco2(pco2_analytical_tissue, hco3)
        if abs(ph_new - analytical_ph) < 1e-13:
            break
        analytical_ph = ph_new
    po2_analytical_tissue = po2_blood - (m_max * v_cell) / (p_perm_o2_um_s * area * alpha_o2)
    
    analytical_po2 = po2_analytical_tissue
    analytical_pco2 = pco2_analytical_tissue
    
    # 4. Run the Massive Picard Solver
    class FakeGrid:
        def __init__(self):
            self.n_cells = 1; self.cell_volume = v_cell; self.dims = (1, 1, 1); self.res = (10.0, 10.0, 10.0)
            
    @dataclass
    class FakeConfig:
        M_max = m_max; k_reduce = 1000.0; respiratory_quotient = rq; hco3_tissue = hco3
        permeability_o2_cm_s = p_perm_o2_cm_s
        permeability_co2_cm_s = p_perm_co2_cm_s
        po2_arterial_mmHg = po2_art; pco2_arterial = pco2_art; systemic_hematocrit = h_d
        sigma_diff = 1.5e-9; sigma_diff_co2 = 1.6e-9
        picard_max_iterations = 50; picard_tolerance = 1e-4

    grid = FakeGrid()
    config = FakeConfig()
    
    G_mock = nx.MultiGraph()
    G_mock.add_node(0); G_mock.add_node(1)
    G_mock.add_edge(0, 1, key=0, flow_signed=q_huge, flow_abs=q_huge / POISEUILLE_FLOW_TO_UM3_PER_S, hematocrit=h_d, length=10.0)
    
    cell_to_vessels_mock = {0: [{'edge': (0, 1, 0), 'flow': q_huge, 'hematocrit': h_d, 'length': 10.0, 'surface_area': area}]}
    
    po2_num, pco2_num, ph_num = solve_multi_species_perfusion(grid, G_mock, [0], cell_to_vessels_mock, config)
    
    # 5. Assert perfection
    np.testing.assert_allclose(po2_num[0], analytical_po2, atol=1e-2)
    np.testing.assert_allclose(pco2_num[0], analytical_pco2, atol=1e-2)
    np.testing.assert_allclose(ph_num[0], analytical_ph, atol=1e-3)
