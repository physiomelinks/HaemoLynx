import numpy as np
import networkx as nx
from numba import jit
import logging
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

def calculate_blood_oxygen_content(po2_mmHg: float, hematocrit: float) -> float:
    """
    Calculates total oxygen content in blood (mmol/L) using the Hill Equation.
    Includes both dissolved plasma O2 and non-linear hemoglobin-bound O2.
    
    Parameters:
    -----------
    po2_mmHg : float
        Partial pressure of oxygen in mmHg.
    hematocrit : float
        Volume fraction of red blood cells (e.g., 0.45).
        
    Returns:
    --------
    float
        Total oxygen concentration in mmol/L.
    """
    if po2_mmHg <= 0.0:
        return 0.0
        
    # Constants for physiological human/rat blood
    alpha_o2 = 1.34e-3  # Solubility of O2 in plasma (mmol/L per mmHg)
    p50 = 26.0          # PO2 at which Hb is 50% saturated (mmHg)
    hill_n = 2.7        # Hill coefficient (curve steepness)
    
    # Max O2 carrying capacity of RBCs (approx 20.4 ml O2 / 100ml blood at H=1.0)
    # 1 ml O2 / 100ml = 0.446 mmol/L
    c_hb_max = 0.446 * 20.4 / 0.45 # Scale to pure RBC
    
    # 1. Linear Dissolved O2 (Henry's Law)
    dissolved = alpha_o2 * po2_mmHg
    
    # 2. Non-Linear Bound O2 (Hill Equation)
    saturation = (po2_mmHg ** hill_n) / ((po2_mmHg ** hill_n) + (p50 ** hill_n))
    bound = hematocrit * c_hb_max * saturation
    
    return float(dissolved + bound)


class PerfusionGrid:
    """
    A 3D structured grid for tissue diffusion modeling.
    Coordinates are natively handled in [z, y, x] to perfectly align with ImageLynx graph conventions
    and VTK exports without flipping.
    """
    def __init__(self, G: nx.MultiGraph, grid_resolution_xyz: Tuple[float, float, float]):
        # 1. Get physical bounds from graph nodes
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            raise ValueError("Graph G must have 'pos' attributes (z, y, x).")
            
        # ImageLynx convention: pos is [z, y, x] in physical units (micrometers)
        nodes_zyx = np.array(list(pos.values()))
        
        # We assume resolution is passed as (x,y,z), so we flip it to (z,y,x) to match
        self.res = np.array([grid_resolution_xyz[2], grid_resolution_xyz[1], grid_resolution_xyz[0]], dtype=float)
        
        # Pad by half resolution to ensure all nodes are inside
        self.min_xyz = np.min(nodes_zyx, axis=0) - self.res * 0.5  # min_xyz is actually min_zyx here
        self.max_xyz = np.max(nodes_zyx, axis=0) + self.res * 0.5  # max_xyz is actually max_zyx here
        
        self.dims = np.ceil((self.max_xyz - self.min_xyz) / self.res).astype(int)
        self.n_cells = int(np.prod(self.dims))
        
        # Calculate volumes for the CellML blueprint
        self.cell_volume = float(np.prod(self.res))
        
        logger.info(f"Generated 3D Perfusion Grid: {self.dims[0]}x{self.dims[1]}x{self.dims[2]} (ZYX) "
                    f"({self.n_cells} cells) at resolution {self.res}µm")

    def get_cell_index(self, xyz: np.ndarray) -> int:
        """Map a physical point (z,y,x) to a linear grid index."""
        return _numba_get_linear_index(xyz, self.min_xyz, self.res, self.dims)

    def get_xyz_from_index(self, index: int) -> np.ndarray:
        """Map a linear index back to physical center-of-cell (z,y,x) coordinates."""
        # index = z + y*nz + x*nz*ny
        nz, ny = self.dims[0], self.dims[1]
        ix = index // (nz * ny)
        iy = (index % (nz * ny)) // nz
        iz = index % nz
        
        indices = np.array([iz, iy, ix], dtype=float)
        return self.min_xyz + (indices + 0.5) * self.res

@jit(nopython=True, cache=True)
def _numba_get_linear_index(pos_xyz, min_xyz, res, dims):
    # pos_xyz and min_xyz are actually (z, y, x)
    rel = pos_xyz - min_xyz
    idx_z = int(rel[0] / res[0])
    idx_y = int(rel[1] / res[1])
    idx_x = int(rel[2] / res[2])
    
    if idx_z < 0 or idx_z >= dims[0] or \
       idx_y < 0 or idx_y >= dims[1] or \
       idx_x < 0 or idx_x >= dims[2]:
        return -1
        
    # Linear index (z fastest)
    return idx_z + idx_y * dims[0] + idx_x * dims[0] * dims[1]

def map_vessels_to_grid(G: nx.MultiGraph, grid: PerfusionGrid) -> Dict[int, List[Dict[str, Any]]]:
    """
    Step 2: Map 1D vessel segments (edges) to the 3D tissue grid cells.
    Returns:
        Mapping of linear_cell_index -> list of segments passing through that cell.
        Each segment info includes the edge ID, flow, and length in that cell.
    """
    cell_to_vessels = {}
    
    for u, v, key, data in G.edges(keys=True, data=True):
        voxels = data.get("voxels")
        flow = data.get("flow_abs", 0.0)
        edge_len = data.get("length", 0.0)
        
        diameter = data.get("assigned_diameter_um", data.get("fwhm_diameter_um", 5.0))
        if diameter is None or diameter <= 0:
            diameter = 5.0
        radius = diameter / 2.0
        
        if voxels is None or len(voxels) < 2:
            continue
            
        # ImageLynx edges store 'voxels' natively in physical ZYX space from build.py!
        # No spacing multiplication needed here.
        vox_phys_zyx = np.array(voxels, dtype=float)
        
        # Incremental length per voxel segment
        # In a real model, we'd use line-plane intersection, but for high-res microscopy,
        # point-sampling the voxels is a robust and fast approximation.
        len_per_vox = edge_len / (len(voxels) - 1) if len(voxels) > 1 else 0.0

        for i in range(len(vox_phys_zyx)):
            zyx = vox_phys_zyx[i]
            idx = grid.get_cell_index(zyx)
            
            if idx != -1:
                if idx not in cell_to_vessels:
                    cell_to_vessels[idx] = []
                
                # Check if this edge is already registered in this specific cell
                found = False
                for item in cell_to_vessels[idx]:
                    if item['edge'] == (u, v, key):
                        item['length'] += len_per_vox
                        item['surface_area'] += 2.0 * np.pi * radius * len_per_vox
                        found = True
                        break
                
                if not found:
                    cell_to_vessels[idx].append({
                        'edge': (u, v, key),
                        'flow': flow,
                        'hematocrit': data.get("hematocrit", 0.45),
                        'length': len_per_vox,
                        'surface_area': 2.0 * np.pi * radius * len_per_vox
                    })
                    
    logger.info(f"Vessel-to-Grid mapping complete. {len(cell_to_vessels)} tissue cells are perfused by vessels.")
    return cell_to_vessels


def build_adr_matrix(grid: PerfusionGrid, cell_to_vessels: Dict[int, List[Dict[str, Any]]], perf_config) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Step 4: Build the pure Diffusion sparse matrix and Advection vectors.
    Returns:
        A: scipy.sparse.csr_matrix (Constant LHS matrix for Diffusion ONLY)
        q_total: np.ndarray (Total bulk flow through each voxel)
        s_incoming: np.ndarray (Fixed arterial oxygen content entering each voxel)
    """
    import scipy.sparse as sp
    
    N = grid.n_cells
    nx, ny, nz = grid.dims
    res = grid.res
    
    # Convert diffusion coefficient from m^2/s to µm^2/s
    sigma_diff_um2_s = perf_config.sigma_diff * 1e12
    
    # Diffusive conductance between cells (µm^3/s)
    D_x = sigma_diff_um2_s * (res[1] * res[2]) / res[0]
    D_y = sigma_diff_um2_s * (res[0] * res[2]) / res[1]
    D_z = sigma_diff_um2_s * (res[0] * res[1]) / res[2]
    
    rows, cols, data = [], [], []
    diag_A = np.zeros(N, dtype=np.float64)
    q_total = np.zeros(N, dtype=np.float64)
    s_incoming = np.zeros(N, dtype=np.float64)
    
    po2_arterial = 100.0 # mmHg
    
    # Advection arrays (Vessel coupling)
    for idx, vessels in cell_to_vessels.items():
        total_q = sum(v['flow'] for v in vessels)
        q_total[idx] = total_q
        
        # Calculate exactly how much oxygen is delivered to this cell based on the Hill Equation
        # S_incoming = Sum( Q * C_blood_arterial )
        total_o2_flux = 0.0
        for v in vessels:
            h = v.get('hematocrit', 0.45)
            c_art = calculate_blood_oxygen_content(po2_arterial, h)
            total_o2_flux += v['flow'] * c_art
            
        s_incoming[idx] = total_o2_flux

    # Build diffusion matrix (Standard 7-point stencil)
    logger.info("Building 3D Diffusion sparse matrix...")
    
    # x-direction edges
    idx_x = np.arange(N).reshape((nz, ny, nx))
    left = idx_x[:, :, :-1].flatten()
    right = idx_x[:, :, 1:].flatten()
    rows.extend(left); cols.extend(right); data.extend([-D_x] * len(left))
    rows.extend(right); cols.extend(left); data.extend([-D_x] * len(right))
    np.add.at(diag_A, left, D_x)
    np.add.at(diag_A, right, D_x)
    
    # y-direction edges
    bottom = idx_x[:, :-1, :].flatten()
    top = idx_x[:, 1:, :].flatten()
    rows.extend(bottom); cols.extend(top); data.extend([-D_y] * len(bottom))
    rows.extend(top); cols.extend(bottom); data.extend([-D_y] * len(top))
    np.add.at(diag_A, bottom, D_y)
    np.add.at(diag_A, top, D_y)
    
    # z-direction edges
    back = idx_x[:-1, :, :].flatten()
    front = idx_x[1:, :, :].flatten()
    rows.extend(back); cols.extend(front); data.extend([-D_z] * len(back))
    rows.extend(front); cols.extend(back); data.extend([-D_z] * len(front))
    np.add.at(diag_A, back, D_z)
    np.add.at(diag_A, front, D_z)
    
    # Add diagonal elements
    all_indices = np.arange(N)
    rows.extend(all_indices)
    cols.extend(all_indices)
    data.extend(diag_A)
    
    # We add a tiny regularization factor to the diagonal to prevent the matrix from being 
    # perfectly singular (since Neumann BCs mean pure diffusion has a null space).
    # But since we no longer have Q on the diagonal, we must add a tiny sink to stabilize CG.
    tiny_sink = 1e-12
    data = np.array(data)
    diag_mask = (np.array(rows) == np.array(cols))
    data[diag_mask] += tiny_sink
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    logger.info(f"Diffusion Matrix constructed. Shape: {A.shape}, Non-zeros: {A.nnz}")
    
    return A, q_total, s_incoming


def solve_perfusion_steady_state(grid: PerfusionGrid, A: Any, q_total: np.ndarray, s_incoming: np.ndarray, perf_config) -> np.ndarray:
    """
    Step 5: Solve the Non-Linear Steady-State Perfusion system using Picard Iteration.
    Solves for tissue PO2 (mmHg).
    """
    import scipy.sparse.linalg as splinalg
    
    N = grid.n_cells
    PO2 = np.zeros(N, dtype=np.float64) # Initial guess (0.0 mmHg everywhere)
    
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    V_cell = grid.cell_volume
    
    # We need a system-wide baseline hematocrit for the venous washout calculation
    # For a perfect voxel-level solution, we'd store a weighted average H_D per voxel,
    # but for stability we can assume the washout matches systemic 0.45, or we can approximate.
    # We will use 0.45 as the baseline for the tissue equilibrium curve.
    h_baseline = 0.45
    
    max_iter = 50
    tolerance = 1e-5
    
    logger.info("Initializing ILU preconditioner for steady-state solver...")
    # Add a tiny diagonal regularizer to A to ensure ILU succeeds if entirely disconnected
    A_reg = A.copy()
    A_reg.setdiag(A_reg.diagonal() + 1e-6)
    
    # NUMERICAL STABILIZATION:
    # Because A is purely diffusion, its rows sum to 0. Solving A*x = b fails if sum(b) != 0.
    # The non-linear advective washout acts as a sink on the RHS, which is highly unstable for CG.
    # We apply a mathematical trick: Add a linear pseudo-washout to the LHS diagonal,
    # and add the exact same term to the RHS. The true steady-state roots remain identical,
    # but the LHS matrix becomes strictly diagonally dominant and highly invertible.
    # Increasing gamma_relax dampens the Picard step size, preventing sigmoidal oscillations.
    gamma_relax = 0.5 # Effective linearized slope
    pseudo_washout_diag = q_total * gamma_relax
    A_stable = A_reg.copy()
    A_stable.setdiag(A_stable.diagonal() + pseudo_washout_diag)
    
    try:
        ilu = splinalg.spilu(A_stable.tocsc(), drop_tol=1e-4, fill_factor=10)
        M_pre = splinalg.LinearOperator(A_stable.shape, ilu.solve)
    except Exception as e:
        logger.warning(f"ILU preconditioning failed: {e}. Falling back to standard CG.")
        M_pre = None

    logger.info("Starting Non-Linear Picard Iteration loop solving for PO2...")
    for iteration in range(max_iter):
        PO2_clamped = np.maximum(PO2, 0.0)
        
        # 1. Compute non-linear metabolic sink based on current PO2
        # M(PO2) = M_max * (1 - exp(-k * PO2))
        M_reduced = M_max * (1.0 - np.exp(-k_reduce * PO2_clamped))
        
        # 2. Compute dynamic Advective Washout
        # Voxel loses oxygen based on blood leaving at local tissue PO2
        s_washout = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if q_total[i] > 0:
                c_venous = calculate_blood_oxygen_content(PO2_clamped[i], h_baseline)
                s_washout[i] = q_total[i] * c_venous
                
        # 3. Construct the full RHS: b = Advection_In - Advection_Out - Metabolic_Sink + Pseudo_Washout
        b = s_incoming - s_washout - (M_reduced * V_cell) + (pseudo_washout_diag * PO2_clamped)
        
        # 4. Solve the linear system A_stable * PO2_new = b
        PO2_new, info = splinalg.cg(A_stable, b, M=M_pre, x0=PO2, rtol=1e-6, maxiter=1000)
        
        if info != 0:
            logger.warning(f"CG Solver did not converge perfectly at iteration {iteration} (info={info})")
            
        # Prevent non-physical negative pressures which cause Picard oscillation
        PO2_new = np.maximum(PO2_new, 0.0)
        
        # 5. Check convergence
        diff = np.linalg.norm(PO2_new - PO2) / (np.linalg.norm(PO2_new) + 1e-12)
        logger.debug(f"  Iteration {iteration+1}: Relative change = {diff:.6e}")
        
        PO2 = PO2_new
        
        if diff < tolerance:
            logger.info(f"Steady-state perfusion converged successfully after {iteration+1} iterations.")
            break
    else:
        logger.warning(f"Picard iteration hit max_iter ({max_iter}) without reaching tolerance {tolerance}.")
        
    return PO2


def solve_coupled_1d3d_perfusion(grid: PerfusionGrid, G: nx.MultiGraph, starting_nodes: list, cell_to_vessels: Dict, perf_config) -> np.ndarray:
    """
    Solve the Fully Coupled 1D-3D Steady-State Perfusion system using Picard Iteration.
    Solves for tissue PO2 (mmHg) and Blood PO2 simultaneously using an endothelial barrier model.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg
    from scipy.optimize import brentq
    import networkx as nx
    
    N = grid.n_cells
    nx_dim, ny_dim, nz_dim = grid.dims
    res = grid.res
    
    sigma_diff_um2_s = perf_config.sigma_diff * 1e12
    D_x = sigma_diff_um2_s * (res[1] * res[2]) / res[0]
    D_y = sigma_diff_um2_s * (res[0] * res[2]) / res[1]
    D_z = sigma_diff_um2_s * (res[0] * res[1]) / res[2]
    
    rows, cols, data = [], [], []
    diag_A = np.zeros(N, dtype=np.float64)
    
    idx_x = np.arange(N).reshape((nz_dim, ny_dim, nx_dim))
    left = idx_x[:, :, :-1].flatten(); right = idx_x[:, :, 1:].flatten()
    rows.extend(left); cols.extend(right); data.extend([-D_x] * len(left))
    rows.extend(right); cols.extend(left); data.extend([-D_x] * len(right))
    np.add.at(diag_A, left, D_x); np.add.at(diag_A, right, D_x)
    
    bottom = idx_x[:, :-1, :].flatten(); top = idx_x[:, 1:, :].flatten()
    rows.extend(bottom); cols.extend(top); data.extend([-D_y] * len(bottom))
    rows.extend(top); cols.extend(bottom); data.extend([-D_y] * len(top))
    np.add.at(diag_A, bottom, D_y); np.add.at(diag_A, top, D_y)
    
    back = idx_x[:-1, :, :].flatten(); front = idx_x[1:, :, :].flatten()
    rows.extend(back); cols.extend(front); data.extend([-D_z] * len(back))
    rows.extend(front); cols.extend(back); data.extend([-D_z] * len(front))
    np.add.at(diag_A, back, D_z); np.add.at(diag_A, front, D_z)
    
    all_indices = np.arange(N)
    rows.extend(all_indices); cols.extend(all_indices); data.extend(diag_A)
    data = np.array(data)
    diag_mask = (np.array(rows) == np.array(cols))
    data[diag_mask] += 1e-12 
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    PO2_tissue = np.zeros(N, dtype=np.float64)
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    V_cell = grid.cell_volume
    P_perm = perf_config.permeability_o2_cm_s * 1e4 # um/s
    po2_arterial = 100.0
    
    edge_to_cells = {}
    q_total = np.zeros(N)
    for cell_idx, vessels in cell_to_vessels.items():
        q_total[cell_idx] = sum(v['flow'] for v in vessels)
        for v in vessels:
            edge = v['edge']
            if edge not in edge_to_cells: edge_to_cells[edge] = []
            edge_to_cells[edge].append({'cell_idx': cell_idx, 'surface_area': v.get('surface_area', 100.0), 'flow': v['flow']})
            
    DAG = nx.MultiDiGraph()
    for u, v, key, e_data in G.edges(keys=True, data=True):
        f = e_data.get("flow_signed", 0.0)
        if f > 0: DAG.add_edge(u, v, key=key, **e_data)
        elif f < 0: DAG.add_edge(v, u, key=key, **e_data)
            
    try:
        topo_order = list(nx.topological_sort(DAG))
    except nx.NetworkXUnfeasible:
        topo_order = list(G.nodes())
        
    A_stable = A.copy()
    gamma_relax = 0.5
    pseudo_washout_diag = q_total * gamma_relax
    A_stable.setdiag(A_stable.diagonal() + pseudo_washout_diag)
    try:
        ilu = splinalg.spilu(A_stable.tocsc(), drop_tol=1e-4, fill_factor=10)
        M_pre = splinalg.LinearOperator(A_stable.shape, ilu.solve)
    except Exception:
        M_pre = None

    logger.info("Starting Fully Coupled 1D-3D Picard Loop...")
    for iteration in range(50):
        PO2_clamped = np.maximum(PO2_tissue, 0.0)
        M_red = M_max * (1.0 - np.exp(-k_reduce * PO2_clamped))
        
        node_o2_flux_in = {n: 0.0 for n in DAG.nodes()}
        node_q_in = {n: 0.0 for n in DAG.nodes()}
        for n in starting_nodes:
            if n in DAG.nodes:
                for succ in DAG.successors(n):
                    for k, d in DAG[n][succ].items():
                        h = d.get("hematocrit", 0.45)
                        node_o2_flux_in[n] += calculate_blood_oxygen_content(po2_arterial, h) * d.get("flow_abs", 0.0)
                        node_q_in[n] += d.get("flow_abs", 0.0)
        
        cell_transmural_flux = np.zeros(N, dtype=np.float64)
        for node in topo_order:
            c_mix = node_o2_flux_in[node] / node_q_in[node] if node_q_in[node] > 0 else calculate_blood_oxygen_content(po2_arterial, 0.45)
            for _, v, k, e_data in DAG.out_edges(node, data=True, keys=True):
                edge_key = (node, v, k)
                if edge_key not in edge_to_cells: edge_key = (v, node, k)
                q = e_data.get("flow_abs", 0.0)
                h = e_data.get("hematocrit", 0.45)
                try:
                    po2_current = brentq(lambda p: calculate_blood_oxygen_content(p, h) - c_mix, 0.0, 150.0)
                except ValueError: po2_current = po2_arterial if c_mix > 0 else 0.0
                
                c_current = c_mix
                for cell in edge_to_cells.get(edge_key, []):
                    flux = P_perm * cell['surface_area'] * max(0.0, po2_current - PO2_clamped[cell['cell_idx']])
                    if q > 0:
                        c_current = max(0.0, c_current - (flux / q))
                        try:
                            po2_current = brentq(lambda p: calculate_blood_oxygen_content(p, h) - c_current, 0.0, 150.0)
                        except ValueError: po2_current = 0.0
                    cell_transmural_flux[cell['cell_idx']] += flux
                node_o2_flux_in[v] += c_current * q; node_q_in[v] += q
                
        b = cell_transmural_flux - (M_red * V_cell) + (pseudo_washout_diag * PO2_clamped)
        PO2_new, info = splinalg.cg(A_stable, b, M=M_pre, x0=PO2_tissue, rtol=1e-6, maxiter=1000)
        PO2_new = np.maximum(PO2_new, 0.0)
        diff = np.linalg.norm(PO2_new - PO2_tissue) / (np.linalg.norm(PO2_new) + 1e-12)
        PO2_tissue = PO2_new
        if diff < 1e-4:
            logger.info(f"Coupled solver converged after {iteration+1} iterations.")
            break
    return PO2_tissue
