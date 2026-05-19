"""
Empirical In-Vivo Rheology Models (Pries and Secomb).
Handles the Fåhræus–Lindqvist effect (diameter-dependent viscosity) and 
Plasma Skimming (unequal hematocrit splitting at bifurcations).
"""
import numpy as np
import networkx as nx

def calculate_pries_secomb_viscosity(diameter_um: float, hematocrit: float, mu_plasma: float = 1.2) -> float:
    """
    Calculates the apparent in-vivo blood viscosity using the empirical Pries-Secomb model.
    Accounts for the Fåhræus–Lindqvist effect where viscosity drops in small capillaries 
    but skyrockets when diameters approach RBC dimensions (< 5um).

    Parameters:
    -----------
    diameter_um : float
        Vessel diameter in micrometers.
    hematocrit : float
        Discharge hematocrit (fraction 0.0 to 1.0).
    mu_plasma : float
        Viscosity of blood plasma (default ~1.2 cP or mPa*s).

    Returns:
    --------
    float
        Apparent dynamic viscosity.
    """
    if diameter_um <= 0 or hematocrit <= 0.0:
        return mu_plasma

    D = diameter_um
    H = hematocrit

    # To avoid mathematical singularities for extremely small vessels, cap D at 3.0 um 
    # (RBCs physically cannot enter vessels smaller than ~3 um without lysing)
    D = max(D, 3.0)

    # 1. Calculate relative apparent viscosity at a standard hematocrit of 0.45
    # Pries et al. (1992) empirical formulation
    mu_45 = 220.0 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * (D ** 0.645))

    # 2. Shape parameter C describing hematocrit dependence
    C_shape = (0.8 + np.exp(-0.075 * D)) * (-1.0 + 1.0 / (1.0 + 10**-11 * D**12)) + (1.0 / (1.0 + 10**-11 * D**12))

    # 3. Calculate relative apparent viscosity at actual hematocrit H
    mu_rel = 1.0 + (mu_45 - 1.0) * (((1.0 - H)**C_shape - 1.0) / ((1.0 - 0.45)**C_shape - 1.0))

    # 4. Correct for the cell-depleted layer near the vessel wall (D / (D - 1.1))^2
    # In-vivo empirical adjustment
    mu_app = mu_rel * ((D / (D - 1.1)) ** 2)

    return float(mu_app * mu_plasma)


def calculate_phase_separation_hematocrit(
    q_in: float, h_in: float, 
    q_out1: float, d_out1: float, 
    q_out2: float, d_out2: float
) -> tuple[float, float]:
    """
    Calculates the phase separation of Red Blood Cells (Plasma Skimming) at a diverging bifurcation.
    RBCs disproportionately favor the branch with higher flow velocity and larger diameter.

    Parameters:
    -----------
    q_in : float
        Total flow entering the bifurcation.
    h_in : float
        Hematocrit entering the bifurcation.
    q_out1, q_out2 : float
        Volumetric blood flow into branch 1 and branch 2.
    d_out1, d_out2 : float
        Diameters of branch 1 and branch 2 in micrometers.

    Returns:
    --------
    tuple[float, float]
        (hematocrit_out1, hematocrit_out2)
    """
    # Prevent division by zero or biologically impossible negative flows
    if q_in <= 1e-12 or h_in <= 0.0:
        return 0.0, 0.0

    fq1 = q_out1 / q_in
    fq2 = q_out2 / q_in

    # If almost all flow goes to one branch, RBCs follow entirely
    if fq1 < 1e-6:
        return 0.0, h_in * (q_in / max(q_out2, 1e-12))
    if fq2 < 1e-6:
        return h_in * (q_in / max(q_out1, 1e-12)), 0.0

    # Critical flow fraction where RBCs completely fail to enter a branch (skimming threshold)
    # Empirically, RBCs struggle to enter branches drawing less than ~5% of flow
    x0 = 0.05 

    if fq1 <= x0:
        fq_e1 = 0.0
    elif fq1 >= 1.0 - x0:
        fq_e1 = 1.0
    else:
        # Pries-Secomb Logistic Skimming Function
        # A defines the asymmetry of the bifurcation based on diameters
        A = -13.29 * ((d_out1**2 / d_out2**2) - 1) / ((d_out1**2 / d_out2**2) + 1) * (1 - h_in) / d_out1
        
        # B controls the steepness of the skimming curve
        B = 1.0 + 6.98 * (1 - h_in) / d_out1
        
        # Logit transformation
        logit_fq = np.log((fq1 - x0) / (1.0 - fq1 - x0))
        
        logit_fe = A + B * logit_fq
        fq_e1 = 1.0 / (1.0 + np.exp(-logit_fe))

    # Mass Conservation of RBCs
    fq_e2 = 1.0 - fq_e1
    
    # Convert RBC flux fractions back to local hematocrit concentrations (H = Flux_RBC / Total_Flow)
    h_out1 = h_in * (fq_e1 / fq1)
    h_out2 = h_in * (fq_e2 / fq2)

    # Physical bounds check
    h_out1 = min(max(h_out1, 0.0), 0.95)
    h_out2 = min(max(h_out2, 0.0), 0.95)

    return float(h_out1), float(h_out2)


def solve_coupled_flow_and_hematocrit(
    G: nx.MultiGraph, 
    starting_nodes: list[int],
    output_nodes: list[int],
    input_p_bc: float,
    output_p_bc: float,
    systemic_hematocrit: float = 0.45,
    max_iterations: int = 15,
    tolerance: float = 1e-4
) -> tuple[nx.MultiGraph, np.ndarray]:
    """
    Solves the highly non-linear coupled system of Flow, Resistance, and Hematocrit.
    
    Algorithm:
    1. Assume uniform hematocrit and calculate baseline Pries-Secomb viscosities/resistances.
    2. Solve the linear Poiseuille flow equations to get flow directions and magnitudes.
    3. Traverse the network topologically from Inlets to Outlets (Directed Acyclic Graph).
    4. At every bifurcation, calculate plasma skimming (phase separation) to assign new hematocrit values to child edges.
    5. Update viscosities and resistances based on the new hematocrit distribution.
    6. Repeat until flow changes fall below tolerance.
    """
    from .resistance import build_conductance_matrix_from_graph, calc_laplacian_from_conductance_matrix, _solve_system_smart
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialization: Assign baseline hematocrit and viscosity
    for u, v, key, data in G.edges(keys=True, data=True):
        diameter = data.get("assigned_diameter_um", data.get("fwhm_diameter_um", 5.0))
        if diameter is None or diameter <= 0:
            diameter = 5.0
            
        data["hematocrit"] = systemic_hematocrit
        mu_app = calculate_pries_secomb_viscosity(diameter, systemic_hematocrit)
        data["viscosity"] = mu_app
        
        length = data.get("length", 10.0)
        data["resistance"] = (128.0 * mu_app * length) / (np.pi * diameter**4)

    iteration = 0
    max_flow_diff = float('inf')
    previous_flows = {}
    final_pressure = None
    
    while iteration < max_iterations and max_flow_diff > tolerance:
        logger.info(f"--- Flow-Hematocrit Iteration {iteration+1} ---")
        
        # 1. Build Conductance and Laplacian
        conductance, node_list = build_conductance_matrix_from_graph(G)
        laplacian = calc_laplacian_from_conductance_matrix(conductance)
        
        n_nodes = len(node_list)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        pressure = np.zeros(n_nodes, dtype=float)
        
        # 2. Apply Boundary Conditions
        bc_idx_to_p = {}
        for n in starting_nodes:
            if n in node_to_idx:
                bc_idx_to_p[node_to_idx[n]] = float(input_p_bc)
        for n in output_nodes:
            if n in node_to_idx:
                bc_idx_to_p[node_to_idx[n]] = float(output_p_bc)
                
        known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
        for idx in known_idx:
            pressure[idx] = bc_idx_to_p[idx]
        unknown_idx = np.array(sorted(set(range(n_nodes)).difference(set(known_idx))), dtype=int)
        
        # 3. Solve Pressure Matrix
        if len(unknown_idx) > 0:
            l_uu = laplacian[unknown_idx, :][:, unknown_idx]
            l_uk = laplacian[unknown_idx, :][:, known_idx]
            p_k = pressure[known_idx]
            rhs = -l_uk.dot(p_k)
            pressure[unknown_idx] = _solve_system_smart(l_uu, rhs)
            
        final_pressure = pressure
        
        # 4. Calculate Flows & Build Directed Acyclic Graph (DAG)
        DAG = nx.MultiDiGraph()
        DAG.add_nodes_from(G.nodes(data=True))
        
        current_flows = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            p_u = pressure[node_to_idx[u]]
            p_v = pressure[node_to_idx[v]]
            r = data["resistance"]
            
            flow_signed = (1.0 / r) * (p_u - p_v)
            flow_abs = abs(flow_signed)
            
            current_flows[(u, v, key)] = flow_abs
            data["flow_abs"] = flow_abs
            data["flow_signed"] = flow_signed
            
            # Direct the edge from high pressure to low pressure
            if flow_signed > 0:
                DAG.add_edge(u, v, key=key, **data)
            else:
                DAG.add_edge(v, u, key=key, **data)
                
        # 5. Check Convergence
        if iteration > 0:
            diffs = [abs(current_flows[k] - previous_flows[k]) for k in current_flows]
            max_flow_diff = max(diffs) if diffs else 0.0
            logger.info(f"  Max Flow Diff: {max_flow_diff:.6e}")
            if max_flow_diff <= tolerance:
                logger.info("  -> Converged!")
                break
                
        previous_flows = current_flows.copy()
        
        # 6. Topologically Traverse DAG and Distribute Hematocrit
        try:
            topological_order = list(nx.topological_sort(DAG))
        except nx.NetworkXUnfeasible:
            logger.warning("  Cycle detected in flow directions! Cannot topologically sort. Breaking iteration.")
            break
            
        # Reset node incoming hematocrit accumulators
        node_h_in = {n: 0.0 for n in DAG.nodes()}
        node_q_in = {n: 0.0 for n in DAG.nodes()}
        
        # Force Systemic Hematocrit at all Inlets
        for n in starting_nodes:
            node_h_in[n] = systemic_hematocrit
            node_q_in[n] = 1.0 # Dummy >0 to prevent div by zero at root
            
        for node in topological_order:
            # Calculate mixed hematocrit at this node
            if node_q_in[node] > 0:
                h_mix = node_h_in[node] / node_q_in[node]
            else:
                h_mix = systemic_hematocrit
                
            out_edges = list(DAG.out_edges(node, data=True, keys=True))
            
            if len(out_edges) == 0:
                continue
            elif len(out_edges) == 1:
                # Direct pass-through
                v, k, data = out_edges[0][1], out_edges[0][2], out_edges[0][3]
                u = node
                G[u][v][k]["hematocrit"] = h_mix
                data["hematocrit"] = h_mix
                node_h_in[v] += h_mix * data["flow_abs"]
                node_q_in[v] += data["flow_abs"]
            elif len(out_edges) == 2:
                # Bifurcation -> Plasma Skimming
                e1, e2 = out_edges[0], out_edges[1]
                q1 = e1[3]["flow_abs"]
                d1 = e1[3].get("assigned_diameter_um", e1[3].get("fwhm_diameter_um", 5.0))
                q2 = e2[3]["flow_abs"]
                d2 = e2[3].get("assigned_diameter_um", e2[3].get("fwhm_diameter_um", 5.0))
                
                h1, h2 = calculate_phase_separation_hematocrit(
                    q1 + q2, h_mix, q1, d1, q2, d2
                )
                
                G[node][e1[1]][e1[2]]["hematocrit"] = h1
                e1[3]["hematocrit"] = h1
                G[node][e2[1]][e2[2]]["hematocrit"] = h2
                e2[3]["hematocrit"] = h2
                
                node_h_in[e1[1]] += h1 * q1
                node_q_in[e1[1]] += q1
                
                node_h_in[e2[1]] += h2 * q2
                node_q_in[e2[1]] += q2
            else:
                # Trifurcation+ -> Just proportional mixing (Phase separation equations only work for Y-splits)
                for _, v, k, data in out_edges:
                    G[node][v][k]["hematocrit"] = h_mix
                    data["hematocrit"] = h_mix
                    node_h_in[v] += h_mix * data["flow_abs"]
                    node_q_in[v] += data["flow_abs"]

        # 7. Update Graph Viscosities and Resistances for next iteration
        for u, v, key, data in G.edges(keys=True, data=True):
            # The DAG data dictionary is a reference to the G data dictionary, so hematocrit is already updated
            h = data["hematocrit"]
            d = data.get("assigned_diameter_um", data.get("fwhm_diameter_um", 5.0))
            if d is None or d <= 0:
                d = 5.0
                
            mu_app = calculate_pries_secomb_viscosity(d, h)
            data["viscosity"] = mu_app
            
            # To preserve the complex geometric integration of sphincters/pericytes,
            # we scale the resistance by the ratio of the new in-vivo viscosity to the old artificial viscosity,
            # rather than overwriting it with a straight-tube approximation.
            if "original_resistance" not in data:
                # Save the base resistance from Phase 4
                data["original_resistance"] = data.get("resistance", (128.0 * 1.0 * data.get("length", 10.0)) / (np.pi * d**4))
                
            # The old viscosity formula used in poiseuille.py was: 1.0 / d^1.647
            mu_old = 1.0 / (d ** 1.647)
            
            # Scale the resistance geometrically
            data["resistance"] = data["original_resistance"] * (mu_app / mu_old)
            
            # WSS = (32 * mu * Q) / (pi * D^3)
            # Units: mu is in mPa*s (cP), Q is in um^3/s, D is in um
            # To get WSS in Pa: (mPa*s * um^3/s) / um^3 = mPa. So WSS is in mPa.
            # Convert mPa to Pa by dividing by 1000.
            q_abs = data.get("flow_abs", 0.0)
            wss_mPa = (32.0 * mu_app * q_abs) / (np.pi * d**3)
            data["wall_shear_stress_pa"] = wss_mPa / 1000.0
            
        iteration += 1

    return G, final_pressure
