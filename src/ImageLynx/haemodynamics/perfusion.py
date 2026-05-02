import numpy as np
import networkx as nx
from numba import jit
import logging
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class PerfusionGrid:
    """
    A 3D structured grid for tissue diffusion modeling.
    Coordinates are stored in [x, y, z] to match typical physiological modeling conventions.
    """
    def __init__(self, G: nx.MultiGraph, grid_resolution_xyz: Tuple[float, float, float]):
        # 1. Get physical bounds from graph nodes
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            raise ValueError("Graph G must have 'pos' attributes (z, y, x).")
            
        nodes_zyx = np.array(list(pos.values()))
        # ImageLynx convention: pos is [z, y, x] in physical units (micrometers)
        nodes_xyz = nodes_zyx[:, [2, 1, 0]]
        
        self.res = np.array(grid_resolution_xyz, dtype=float)
        # Pad by half resolution to ensure all nodes are inside
        self.min_xyz = np.min(nodes_xyz, axis=0) - self.res * 0.5
        self.max_xyz = np.max(nodes_xyz, axis=0) + self.res * 0.5
        
        self.dims = np.ceil((self.max_xyz - self.min_xyz) / self.res).astype(int)
        self.n_cells = int(np.prod(self.dims))
        
        # Calculate volumes for the CellML blueprint
        self.cell_volume = float(np.prod(self.res))
        
        logger.info(f"Generated 3D Perfusion Grid: {self.dims[0]}x{self.dims[1]}x{self.dims[2]} "
                    f"({self.n_cells} cells) at resolution {grid_resolution_xyz}µm")

    def get_cell_index(self, xyz: np.ndarray) -> int:
        """Map a physical point to a linear grid index."""
        return _numba_get_linear_index(xyz, self.min_xyz, self.res, self.dims)

    def get_xyz_from_index(self, index: int) -> np.ndarray:
        """Map a linear index back to physical center-of-cell XYZ coordinates."""
        # index = x + y*nx + z*nx*ny
        nx, ny = self.dims[0], self.dims[1]
        iz = index // (nx * ny)
        iy = (index % (nx * ny)) // nx
        ix = index % nx
        
        indices = np.array([ix, iy, iz], dtype=float)
        return self.min_xyz + (indices + 0.5) * self.res

@jit(nopython=True, cache=True)
def _numba_get_linear_index(pos_xyz, min_xyz, res, dims):
    rel = pos_xyz - min_xyz
    idx_x = int(rel[0] / res[0])
    idx_y = int(rel[1] / res[1])
    idx_z = int(rel[2] / res[2])
    
    if idx_x < 0 or idx_x >= dims[0] or \
       idx_y < 0 or idx_y >= dims[1] or \
       idx_z < 0 or idx_z >= dims[2]:
        return -1
        
    # Linear index (x fastest)
    return idx_x + idx_y * dims[0] + idx_z * dims[0] * dims[1]

def map_vessels_to_grid(G: nx.MultiGraph, grid: PerfusionGrid) -> Dict[int, List[Dict[str, Any]]]:
    """
    Step 2: Map 1D vessel segments (edges) to the 3D tissue grid cells.
    Returns:
        Mapping of linear_cell_index -> list of segments passing through that cell.
        Each segment info includes the edge ID, flow, and length in that cell.
    """
    cell_to_vessels = {}
    
    # Get spacing from graph metadata to convert voxels to physical
    spacing = np.array(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))

    for u, v, key, data in G.edges(keys=True, data=True):
        voxels = data.get("voxels")
        flow = data.get("flow_abs", 0.0)
        edge_len = data.get("length", 0.0)
        
        if voxels is None or len(voxels) < 2:
            continue
            
        # Convert voxels (zyx image) to physical xyz
        vox_arr = np.array(voxels, dtype=float)
        # Apply spacing to match physical scale of G.nodes['pos']
        vox_phys_xyz = np.zeros_like(vox_arr)
        vox_phys_xyz[:, 0] = vox_arr[:, 2] * spacing[2] # x
        vox_phys_xyz[:, 1] = vox_arr[:, 1] * spacing[1] # y
        vox_phys_xyz[:, 2] = vox_arr[:, 0] * spacing[0] # z
        
        # Incremental length per voxel segment
        # In a real model, we'd use line-plane intersection, but for high-res microscopy,
        # point-sampling the voxels is a robust and fast approximation.
        len_per_vox = edge_len / (len(voxels) - 1) if len(voxels) > 1 else 0.0

        for i in range(len(vox_phys_xyz)):
            xyz = vox_phys_xyz[i]
            idx = grid.get_cell_index(xyz)
            
            if idx != -1:
                if idx not in cell_to_vessels:
                    cell_to_vessels[idx] = []
                
                # Check if this edge is already registered in this specific cell
                found = False
                for item in cell_to_vessels[idx]:
                    if item['edge'] == (u, v, key):
                        item['length'] += len_per_vox
                        found = True
                        break
                
                if not found:
                    cell_to_vessels[idx].append({
                        'edge': (u, v, key),
                        'flow': flow,
                        'length': len_per_vox
                    })
                    
    logger.info(f"Vessel-to-Grid mapping complete. {len(cell_to_vessels)} tissue cells are perfused by vessels.")
    return cell_to_vessels


def build_adr_matrix(grid: PerfusionGrid, cell_to_vessels: Dict[int, List[Dict[str, Any]]], perf_config) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Step 4: Build the Advection-Diffusion-Reaction (ADR) sparse matrix.
    Returns:
        A: scipy.sparse.csr_matrix (Constant LHS matrix for Diffusion and Advection)
        b_adv: np.ndarray (Constant RHS vector for Advection source terms)
        D_diag: np.ndarray (Main diagonal of D, used for reference if needed)
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
    
    # Pre-allocate sparse matrix components
    rows = []
    cols = []
    data = []
    
    b_adv = np.zeros(N, dtype=np.float64)
    diag_A = np.zeros(N, dtype=np.float64)
    
    # Advection source terms (Vessel coupling)
    for idx, vessels in cell_to_vessels.items():
        total_q = sum(v['flow'] for v in vessels)
        # Add advective washout to diagonal
        diag_A[idx] += total_q
        # Add advective source to RHS
        b_adv[idx] += total_q * perf_config.C_arterial

    # Build diffusion matrix (Standard 7-point stencil)
    # Using Numba for speed is possible, but vectorized construction is also fast.
    # We will build it directly.
    logger.info("Building 3D Advection-Diffusion sparse matrix...")
    
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
    
    # Prevent completely disconnected, non-perfused cells from being strictly singular
    # by adding a tiny regularization factor to the diagonal if needed.
    # However, ILU can often handle it if there's connection to perfused cells.
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    logger.info(f"ADR Matrix constructed. Shape: {A.shape}, Non-zeros: {A.nnz}")
    
    return A, b_adv, diag_A


def solve_perfusion_steady_state(grid: PerfusionGrid, A: Any, b_adv: np.ndarray, perf_config) -> np.ndarray:
    """
    Step 5: Solve the Non-Linear Steady-State Perfusion system using Picard Iteration.
    """
    import scipy.sparse.linalg as splinalg
    
    N = grid.n_cells
    C = np.zeros(N, dtype=np.float64) # Initial guess (0.0 mmol/L everywhere)
    
    M_max = perf_config.M_max
    k_reduce = perf_config.k_reduce
    V_cell = grid.cell_volume
    
    max_iter = 50
    tolerance = 1e-5
    
    logger.info("Initializing ILU preconditioner for steady-state solver...")
    # Because A is diagonally dominant, ILU works very well
    try:
        ilu = splinalg.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
        M_pre = splinalg.LinearOperator(A.shape, ilu.solve)
    except Exception as e:
        logger.warning(f"ILU preconditioning failed: {e}. Falling back to standard CG.")
        M_pre = None

    logger.info("Starting Non-Linear Picard Iteration loop...")
    for iteration in range(max_iter):
        # 1. Compute non-linear metabolic sink based on current concentration
        # M(C) = M_max * (1 - exp(-k * C))
        # Ensure C doesn't drop below 0 physically
        C_clamped = np.maximum(C, 0.0)
        M_reduced = M_max * (1.0 - np.exp(-k_reduce * C_clamped))
        
        # 2. Construct the full RHS: b = Advection_Source - Metabolic_Sink
        # Note: units of M_reduced * V_cell naturally balance with D and Q (as derived)
        b = b_adv - (M_reduced * V_cell)
        
        # 3. Solve the linear system A * C_new = b
        C_new, info = splinalg.cg(A, b, M=M_pre, x0=C, rtol=1e-6, maxiter=1000)
        
        if info != 0:
            logger.warning(f"CG Solver did not converge perfectly at iteration {iteration} (info={info})")
            
        # 4. Check convergence
        diff = np.linalg.norm(C_new - C) / (np.linalg.norm(C_new) + 1e-12)
        logger.debug(f"  Iteration {iteration+1}: Relative change = {diff:.6e}")
        
        C = C_new
        
        if diff < tolerance:
            logger.info(f"Steady-state perfusion converged successfully after {iteration+1} iterations.")
            break
    else:
        logger.warning(f"Picard iteration hit max_iter ({max_iter}) without reaching tolerance {tolerance}.")
        
    return C
