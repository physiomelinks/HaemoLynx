"""
Empirical In-Vivo Rheology Models (Pries and Secomb).
Handles the Fåhræus–Lindqvist effect (diameter-dependent viscosity) and 
Plasma Skimming (unequal hematocrit splitting at bifurcations).
"""
import numpy as np
import networkx as nx

#: The two Pries-Secomb relations for relative apparent viscosity at a discharge haematocrit
#: of 0.45. They differ only in the first term, and that term is the whole disagreement.
#:
#: ``in_vitro`` is Pries et al. (1992), fitted to blood in glass tubes.
#: ``in_vivo`` is Pries et al. (1994), fitted to microvessels in living tissue, where the
#: endothelial surface layer narrows the effective lumen and raises resistance.
#:
#: At D = 8 um and H = 0.45 the two give apparent viscosities differing by roughly 3.4x, and
#: resistance is linear in viscosity, so this is not a refinement.
PRIES_SECOMB_LAWS = ("in_vivo", "in_vitro")


def _mu_45(diameter_um: float, law: str) -> float:
    """Relative apparent viscosity at H = 0.45, under the requested relation."""
    d = diameter_um
    tail = 3.2 - 2.44 * np.exp(-0.06 * (d ** 0.645))
    if law == "in_vitro":
        return 220.0 * np.exp(-1.3 * d) + tail
    return 6.0 * np.exp(-0.085 * d) + tail


def calculate_pries_secomb_viscosity(
    diameter_um: float,
    hematocrit: float,
    mu_plasma: float = 1.2,
    law: str = "in_vivo",
) -> float:
    """Apparent blood viscosity, accounting for the Fahraeus-Lindqvist effect.

    ``law`` selects the relation, and the wall-layer correction follows it rather than being
    applied unconditionally. A glass tube has no endothelial surface layer, so correcting for
    one there is not a refinement of the in vitro law but a departure from both.

    This function previously combined the in vitro base with the in vivo wall correction,
    which is neither. **in vivo is the default**: H2 models perfusion of living tissue, where
    the surface layer is present, and the glass-tube relation is the special case.

    Parameters
    ----------
    diameter_um:
        Vessel diameter. Capped below at 3.0 um, since an RBC cannot pass a smaller lumen
        intact and the relation runs away there.
    hematocrit:
        Discharge haematocrit, 0 to 1. Returns plasma viscosity at zero.
    mu_plasma:
        Plasma viscosity, default 1.2 cP.
    law:
        ``"in_vivo"`` (default) or ``"in_vitro"``.
    """
    if law not in PRIES_SECOMB_LAWS:
        raise ValueError(
            f"Unknown viscosity law {law!r}. Expected one of {PRIES_SECOMB_LAWS}. "
            f"The two differ by about 3.4x in the capillary range, so there is no safe default "
            f"beyond the in vivo relation this pipeline uses."
        )
    if diameter_um <= 0 or hematocrit <= 0.0:
        return mu_plasma

    d = max(float(diameter_um), 3.0)
    h = float(hematocrit)

    mu_45 = _mu_45(d, law)

    # Shape parameter for the haematocrit dependence. Common to both relations.
    tail = 1.0 / (1.0 + 1e-11 * d ** 12)
    c_shape = (0.8 + np.exp(-0.075 * d)) * (-1.0 + tail) + tail
    h_term = ((1.0 - h) ** c_shape - 1.0) / ((1.0 - 0.45) ** c_shape - 1.0)

    if law == "in_vitro":
        mu_rel = 1.0 + (mu_45 - 1.0) * h_term
    else:
        # The cell-depleted layer near the wall, W = (D / (D - 1.1))^2. It appears *twice* in
        # the published in vivo relation, once scaling the haematocrit term inside the bracket
        # and once outside it. Applying it once understates apparent viscosity by 1.26x at
        # 8 um and 2.2x at 3 um, which is the calibre range every vessel here sits in.
        wall = (d / (d - 1.1)) ** 2
        mu_rel = (1.0 + (mu_45 - 1.0) * h_term * wall) * wall

    return float(mu_rel * mu_plasma)


def calculate_phase_separation_hematocrit(
    q_in: float, h_in: float, 
    q_out1: float, d_out1: float, 
    q_out2: float, d_out2: float,
    d_parent: float,
) -> tuple[float, float]:
    """
    Calculates the phase separation of Red Blood Cells (Plasma Skimming) at a diverging bifurcation.
    RBCs disproportionately favor the branch with higher flow velocity and larger diameter.

    Implements the empirical phase separation law of Pries et al. (1989) in its later
    parametrisation, as printed in Rasmussen, Secomb & Pries (2018):

        A  = -13.29 * ((D1/D2)^2 - 1) / ((D1/D2)^2 + 1) * (1 - H) / D_F
        B  = 1 + 6.98 * (1 - H) / D_F
        X0 = 0.964 * (1 - H) / D_F
        logit FQ_E1 = A + B * logit[(FQ_1 - X0) / (1 - 2 X0)]

    with FQ_E1 = 0 for FQ_1 <= X0 and 1 for FQ_1 >= 1 - X0. Rasmussen et al. attribute the
    constants to a regression in Pries, Reglin & Secomb (2003); other sources call the same
    form Pries & Secomb (2005). Which of the two first printed them is unconfirmed.

    All three terms scale with the feeding vessel diameter D_F, not a daughter's, which makes
    the result independent of which branch is labelled 1.

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
    d_parent : float
        Diameter of the feeding (parent) vessel D_F in micrometers.

    Returns:
    --------
    tuple[float, float]
        (hematocrit_out1, hematocrit_out2). If X0 >= 0.5, which happens only for D_F below
        about 1.06 um at H = 0.45, the law is undefined and both branches get h_in.
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

    # Minimum flow fraction a branch must draw to receive any red cells (Pries et al.)
    x0 = 0.964 * (1 - h_in) / d_parent

    # 1 - 2 X0 <= 0 leaves no flow range for the logistic curve; fall back to proportional
    # splitting rather than extrapolating the regression that far below its data.
    if x0 >= 0.5:
        return float(h_in), float(h_in)

    if fq1 <= x0:
        fq_e1 = 0.0
    elif fq1 >= 1.0 - x0:
        fq_e1 = 1.0
    else:
        # Pries-Secomb Logistic Skimming Function
        # A defines the asymmetry of the bifurcation based on diameters
        A = -13.29 * ((d_out1**2 / d_out2**2) - 1) / ((d_out1**2 / d_out2**2) + 1) * (1 - h_in) / d_parent
        
        # B controls the steepness of the skimming curve
        B = 1.0 + 6.98 * (1 - h_in) / d_parent
        
        # Logit transformation; (FQ-X0)/(1-FQ-X0) is logit[(FQ-X0)/(1-2 X0)] rearranged
        logit_fq = np.log((fq1 - x0) / (1.0 - fq1 - x0))
        
        logit_fe = A + B * logit_fq
        fq_e1 = 1.0 / (1.0 + np.exp(-logit_fe))

    # Mass Conservation of RBCs
    fq_e2 = 1.0 - fq_e1
    
    # Convert RBC flux fractions back to local hematocrit concentrations (H = Flux_RBC / Total_Flow)
    h_out1 = h_in * (fq_e1 / fq1)
    h_out2 = h_in * (fq_e2 / fq2)

    # Physical bounds check. Not part of the Pries law, and RBC flux is no longer conserved
    # if it ever fires; it has not been seen to for daughters of 3-30 um.
    h_out1 = min(max(h_out1, 0.0), 0.95)
    h_out2 = min(max(h_out2, 0.0), 0.95)

    return float(h_out1), float(h_out2)


def _edge_diameter_um(data, default_diameter_um=None):
    """The edge's measured diameter, or the caller's deliberate stand-in.

    This used to substitute 5.0 um silently whenever the attribute was absent or
    non-positive - open item 9 in ``cb_modelling_reference.md``. Resistance goes as the
    fourth power of diameter, so a cached graph carrying no calibre solved at a uniform
    5 um for every edge and produced a flow field that was arithmetically fine and meant
    nothing. ``map_vessels_to_grid`` and ``edge_transit_times`` already raised on exactly
    this condition; this function makes the third site agree with them.
    """
    diameter = data.get("assigned_diameter_um", data.get("fwhm_diameter_um"))
    if diameter is not None and float(diameter) > 0:
        return float(diameter)
    return None if default_diameter_um is None else float(default_diameter_um)


def _require_diameters(G, default_diameter_um=None):
    """Raise unless every edge carries a usable diameter."""
    missing = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
               if _edge_diameter_um(d, default_diameter_um) is None]
    if missing:
        shown = ", ".join(str(e) for e in missing[:3])
        raise ValueError(
            f"{len(missing)} of {G.number_of_edges()} edges have no usable diameter "
            f"(absent or non-positive), for example {shown}. Resistance goes as the fourth "
            f"power of diameter, so a substituted calibre produces a fabricated flow field "
            f"rather than an approximate one. Assign diameters first, or pass "
            f"default_diameter_um to model the unmeasured edges at a stated calibre."
        )


def feeding_vessel_diameter(
    DAG: nx.MultiDiGraph, node, d1: float, d2: float, default_diameter_um=None
) -> tuple[float, bool]:
    """
    The feeding diameter D_F at a diverging bifurcation, for the phase separation law.

    One incoming edge gives its diameter. Several (a merge that splits again) give the
    flow-weighted mean of their diameters. No incoming edge with flow falls back to the
    larger daughter, and the second return value is True so callers can count the fallbacks.
    """
    q_sum = 0.0
    qd_sum = 0.0
    for _, _, data in DAG.in_edges(node, data=True):
        q = data["flow_abs"]
        q_sum += q
        qd_sum += q * _edge_diameter_um(data, default_diameter_um)
    if q_sum > 0.0:
        return qd_sum / q_sum, False
    return max(d1, d2), True


def _upstream_viscosity(diameter_um: float) -> float:
    """The viscosity every upstream resistance setter bakes in: 1 / D^1.647.

    poiseuille.py, pericyte_mask.py and probability.py all use it.
    """
    return 1.0 / (diameter_um ** 1.647)


def _record_original_resistance(data: dict, diameter_um: float) -> None:
    """Save the resistance the edge arrived with, once, before the solver rescales it.

    An edge with no resistance gets straight-tube Poiseuille with the upstream viscosity, so
    the rescale reduces to 128 mu_app L / (pi D^4). An edge that already carries
    original_resistance, from an earlier call on the same graph, keeps it: re-saving would
    rescale an already rescaled value.
    """
    if "original_resistance" in data:
        return
    resistance = data.get("resistance")
    if resistance is None:
        length = data.get("length", 10.0)
        resistance = (
            128.0 * _upstream_viscosity(diameter_um) * length
        ) / (np.pi * diameter_um**4)
    data["original_resistance"] = resistance


def _rescaled_resistance(data: dict, diameter_um: float, mu_app: float) -> float:
    """The arriving resistance with its upstream viscosity swapped for mu_app.

    Resistance is linear in viscosity, so this keeps any sphincter or pericyte geometry
    integrated upstream. For a constricted edge it is approximate: upstream integrated
    1 / d(x)^1.647 along the edge, while this ratio uses the one edge diameter.
    """
    return data["original_resistance"] * (mu_app / _upstream_viscosity(diameter_um))


def solve_coupled_flow_and_hematocrit(
    G: nx.MultiGraph, 
    starting_nodes: list[int],
    output_nodes: list[int],
    input_p_bc: float,
    output_p_bc: float,
    systemic_hematocrit: float = 0.45,
    max_iterations: int = 15,
    tolerance: float = 1e-4,
    default_diameter_um: float | None = None,
    relaxation: float = 0.5,
) -> tuple[nx.MultiGraph, np.ndarray]:
    """
    Solves the highly non-linear coupled system of Flow, Resistance, and Hematocrit.
    
    Algorithm:
    1. Assume uniform hematocrit and calculate baseline Pries-Secomb viscosities/resistances.
    2. Solve the linear Poiseuille flow equations to get flow directions and magnitudes.
    3. Traverse the network topologically from Inlets to Outlets (Directed Acyclic Graph).
    4. At every bifurcation, calculate plasma skimming (phase separation) to assign new hematocrit values to child edges.
    5. Under-relax the new hematocrit against the previous pass, then update viscosities and
       resistances from it.
    6. Repeat until flow changes fall below tolerance.

    Why the loop stopped is written to ``G.graph``, since a cycle or iteration-limit exit
    otherwise returns a graph indistinguishable from a converged one:

    - ``rheology_stop_reason``: ``"converged"``, ``"flow_cycle"`` (the flow directions could
      not be sorted topologically) or ``"max_iterations"``.
    - ``rheology_iterations``: number of pressure solves performed.
    - ``rheology_max_flow_change``: last max absolute change in edge flow between passes, or
      None if fewer than two passes ran.

    ``relaxation`` is the step taken towards the new hematocrit each pass,
    H = H_old + relaxation * (H_skim - H_old). At 1.0 (undamped) a 15 -> 10/5 um Y-junction
    under the in vivo viscosity law flips between two states every pass and never converges;
    0.5 settles it.
    """
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(f"relaxation must be in (0, 1], got {relaxation}.")
    from .resistance import build_conductance_matrix_from_graph, calc_laplacian_from_conductance_matrix, _solve_system_smart
    import logging
    logger = logging.getLogger(__name__)
    _require_diameters(G, default_diameter_um)
    
    # Initialization: Assign baseline hematocrit and viscosity. The arriving resistance is
    # recorded before it is touched; overwriting it here used to discard any upstream
    # sphincter or pericyte geometry and count viscosity twice in the update below.
    for u, v, key, data in G.edges(keys=True, data=True):
        diameter = _edge_diameter_um(data, default_diameter_um)
            
        data["hematocrit"] = systemic_hematocrit
        mu_app = calculate_pries_secomb_viscosity(diameter, systemic_hematocrit)
        data["viscosity"] = mu_app

        _record_original_resistance(data, diameter)
        data["resistance"] = _rescaled_resistance(data, diameter, mu_app)

    iteration = 0
    max_flow_diff = float('inf')
    previous_flows = {}
    final_pressure = None
    stop_reason = "max_iterations"
    
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
                stop_reason = "converged"
                break
                
        previous_flows = current_flows.copy()
        
        # 6. Topologically Traverse DAG and Distribute Hematocrit
        try:
            topological_order = list(nx.topological_sort(DAG))
        except nx.NetworkXUnfeasible:
            logger.warning("  Cycle detected in flow directions! Cannot topologically sort. Breaking iteration.")
            stop_reason = "flow_cycle"
            break
            
        previous_hematocrit = {
            (u, v, key): data["hematocrit"] for u, v, key, data in G.edges(keys=True, data=True)
        }

        # Reset node incoming hematocrit accumulators
        node_h_in = {n: 0.0 for n in DAG.nodes()}
        node_q_in = {n: 0.0 for n in DAG.nodes()}
        
        # Force Systemic Hematocrit at all Inlets
        for n in starting_nodes:
            node_h_in[n] = systemic_hematocrit
            node_q_in[n] = 1.0 # Dummy >0 to prevent div by zero at root

        # Bifurcations with no inflowing parent, where D_F falls back to the larger daughter
        n_parent_fallbacks = 0
            
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
                d1 = _edge_diameter_um(e1[3], default_diameter_um)
                q2 = e2[3]["flow_abs"]
                d2 = _edge_diameter_um(e2[3], default_diameter_um)
                
                d_parent, fell_back = feeding_vessel_diameter(
                    DAG, node, d1, d2, default_diameter_um
                )
                n_parent_fallbacks += fell_back
                h1, h2 = calculate_phase_separation_hematocrit(
                    q1 + q2, h_mix, q1, d1, q2, d2, d_parent
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

        if n_parent_fallbacks:
            logger.info(
                f"  {n_parent_fallbacks} bifurcation(s) had no inflowing parent; "
                "D_F fell back to the larger daughter diameter."
            )

        # 7. Update Graph Viscosities and Resistances for next iteration
        for u, v, key, data in G.edges(keys=True, data=True):
            # The traversal wrote the skimmed hematocrit into G; step only part of the way to it.
            h_old = previous_hematocrit[(u, v, key)]
            h = h_old + relaxation * (data["hematocrit"] - h_old)
            data["hematocrit"] = h
            d = _edge_diameter_um(data, default_diameter_um)
                
            mu_app = calculate_pries_secomb_viscosity(d, h)
            data["viscosity"] = mu_app
            
            # Rescale the arriving resistance rather than overwrite it with a straight tube,
            # so sphincter and pericyte geometry survives.
            data["resistance"] = _rescaled_resistance(data, d, mu_app)
            
            # WSS = (32 * mu * Q) / (pi * D^3)
            # Units: mu is in mPa*s (cP), Q is in um^3/s, D is in um
            # To get WSS in Pa: (mPa*s * um^3/s) / um^3 = mPa. So WSS is in mPa.
            # Convert mPa to Pa by dividing by 1000.
            q_abs = data.get("flow_abs", 0.0)
            wss_mPa = (32.0 * mu_app * q_abs) / (np.pi * d**3)
            data["wall_shear_stress_pa"] = wss_mPa / 1000.0
            
        iteration += 1

    # A break leaves the pass that triggered it uncounted.
    iterations = iteration if stop_reason == "max_iterations" else iteration + 1
    if stop_reason == "max_iterations":
        logger.warning(f"  Rheology did not converge within {max_iterations} iterations.")
    G.graph["rheology_stop_reason"] = stop_reason
    G.graph["rheology_iterations"] = iterations
    G.graph["rheology_max_flow_change"] = None if np.isinf(max_flow_diff) else float(max_flow_diff)

    return G, final_pressure
