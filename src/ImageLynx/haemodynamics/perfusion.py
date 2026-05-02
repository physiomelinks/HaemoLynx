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
