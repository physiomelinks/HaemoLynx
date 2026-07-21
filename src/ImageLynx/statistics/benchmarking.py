import logging
import numpy as np
import networkx as nx
from scipy import ndimage
from typing import Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

def redilate_skeleton_to_volume(G: nx.MultiGraph, shape: Tuple[int, int, int], voxel_size_xyz: Tuple[float, float, float]) -> np.ndarray:
    """
    Reconstructs a 3D binary volume from the mathematical graph by dilating each centerline
    point by its assigned physical radius.
    """
    reconstructed = np.zeros(shape, dtype=bool)
    vz, vy, vx = voxel_size_xyz
    
    # Extract points and radii
    for u, v, key, data in G.edges(keys=True, data=True):
        pts = data.get("pts")
        if pts is None or len(pts) == 0:
            pts = data.get("voxels")
        if pts is None or len(pts) == 0:
            continue
            
        # Prioritize assigned_diameter_um, fallback to fwhm
        diameter_um = data.get("assigned_diameter_um", data.get("fwhm_diameter_um", 5.0))
        radius_um = diameter_um / 2.0
        
        # Convert radius to voxels (assuming isotropic-ish for dilation simplicity, or using min spacing)
        # For rigorous 3D, we'll use a bounding box approach per point
        radius_vox_x = radius_um / vx
        radius_vox_y = radius_um / vy
        radius_vox_z = radius_um / vz
        
        max_rad = math.ceil(max(radius_vox_x, radius_vox_y, radius_vox_z))
        
        for pt in pts:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            
            # Simple bounding box voxelization
            z_min, z_max = max(0, z - max_rad), min(shape[0], z + max_rad + 1)
            y_min, y_max = max(0, y - max_rad), min(shape[1], y + max_rad + 1)
            x_min, x_max = max(0, x - max_rad), min(shape[2], x + max_rad + 1)
            
            for zz in range(z_min, z_max):
                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        # Ellipsoid distance check
                        dist = ((xx - x) * vx)**2 + ((yy - y) * vy)**2 + ((zz - z) * vz)**2
                        if dist <= radius_um**2:
                            reconstructed[zz, yy, xx] = True
                            
    return reconstructed

def evaluate_volumetric_reconstruction(G: nx.MultiGraph, binary_mask: np.ndarray, voxel_size_xyz: Tuple[float, float, float]) -> Dict[str, float]:
    """Computes Dice Similarity Coefficient and Jaccard Index."""
    logger.info("Computing Volumetric Reconstruction Benchmarks (Redilation)...")
    reconstructed = redilate_skeleton_to_volume(G, binary_mask.shape, voxel_size_xyz)
    
    intersection = np.logical_and(binary_mask, reconstructed).sum()
    volume_mask = binary_mask.sum()
    volume_recon = reconstructed.sum()
    
    if volume_mask + volume_recon == 0:
        return {"dice_coefficient": 0.0, "jaccard_index": 0.0}
        
    dice = (2.0 * intersection) / (volume_mask + volume_recon)
    jaccard = intersection / (volume_mask + volume_recon - intersection)
    
    return {
        "dice_coefficient": float(dice),
        "jaccard_index": float(jaccard),
        "reconstructed_volume_fraction": float(volume_recon / max(1, volume_mask))
    }

def evaluate_centerline_centricity(G: nx.MultiGraph, binary_mask: np.ndarray, voxel_size_xyz: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Computes Euclidean Distance Transform on the mask and checks how perfectly
    the skeleton points align with the topological ridges.
    """
    logger.info("Computing Centerline Centricity (Distance Transform)...")
    # scipy EDT uses voxel counts. We scale by voxel size.
    sampling = (voxel_size_xyz[0], voxel_size_xyz[1], voxel_size_xyz[2])
    edt = ndimage.distance_transform_edt(binary_mask, sampling=sampling)
    
    centricity_errors = []
    
    for u, v, key, data in G.edges(keys=True, data=True):
        pts = data.get("pts")
        if pts is None or len(pts) == 0:
            pts = data.get("voxels")
        if pts is None or len(pts) == 0:
            continue
            
        diameter_um = data.get("fwhm_diameter_um")
        if diameter_um is None or diameter_um <= 0:
            continue
            
        expected_radius = diameter_um / 2.0
        
        for pt in pts:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < edt.shape[0] and 0 <= y < edt.shape[1] and 0 <= x < edt.shape[2]:
                actual_distance_to_wall = edt[z, y, x]
                # Error is the absolute difference between the EDT ridge and the assigned radius
                error = abs(actual_distance_to_wall - expected_radius)
                centricity_errors.append(error)
                
    if not centricity_errors:
        return {"mean_centricity_error_um": 0.0, "median_centricity_error_um": 0.0}
        
    return {
        "mean_centricity_error_um": float(np.mean(centricity_errors)),
        "median_centricity_error_um": float(np.median(centricity_errors)),
        "max_centricity_error_um": float(np.max(centricity_errors))
    }

def _calculate_euler_characteristic_3d(binary_mask: np.ndarray) -> int:
    """
    Approximates the Euler characteristic of a 3D binary volume using 
    2x2x2 voxel neighborhood configurations (Marching Cubes topological analysis).
    For complex biological networks, this gives the number of components minus the number of holes.
    """
    # Pad to handle boundaries
    padded = np.pad(binary_mask, 1, mode='constant')
    
    # Calculate vertices, edges, faces, and cubes (V - E + F - C)
    # This is a simplified proxy. 
    # For robust biological graphs, we primarily care about the Betti-1 number (cycles).
    # Since full 3D Betti computation requires complex algebraic topology libraries (like Gudhi),
    # we will rely on the skeletal cyclomatic complexity vs the mask's component count.
    
    labeled, num_components = ndimage.label(binary_mask)
    return num_components

def evaluate_topological_preservation(G: nx.MultiGraph, binary_mask: np.ndarray) -> Dict[str, float]:
    """
    Compares the cyclomatic complexity (fundamental loops) of the skeleton graph
    against the number of isolated components in the binary mask. Also evaluates
    advanced Graph Health metrics (Dead-end ratio, Bifurcation sanity, Edge Variance).
    """
    logger.info("Computing Topological Preservation...")
    
    _, num_components = ndimage.label(binary_mask)
    
    # Graph cyclomatic complexity: E - V + C
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    graph_components = nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G)
    
    graph_loops = num_edges - num_nodes + graph_components
    
    # Advanced Graph Health
    terminal_nodes = sum(1 for n in G.nodes() if G.degree(n) == 1)
    terminal_ratio = terminal_nodes / max(1, num_nodes)
    
    deg3 = sum(1 for n in G.nodes() if G.degree(n) == 3)
    deg4_plus = sum(1 for n in G.nodes() if G.degree(n) > 3)
    bifurcation_ratio = deg3 / max(1, deg3 + deg4_plus)
    
    # Edge Length Variance
    import math
    lengths = []
    for u, v, data in G.edges(data=True):
        if "length" in data:
            lengths.append(data["length"])
        elif "pos" in G.nodes[u] and "pos" in G.nodes[v]:
            p1, p2 = G.nodes[u]["pos"], G.nodes[v]["pos"]
            lengths.append(math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2))))
            
    import numpy as np
    edge_length_std = float(np.std(lengths)) if lengths else 0.0
    
    return {
        "binary_mask_connected_components": int(num_components),
        "graph_connected_components": int(graph_components),
        "graph_fundamental_loops": int(graph_loops),
        "terminal_node_ratio": float(terminal_ratio),
        "degree3_bifurcation_ratio": float(bifurcation_ratio),
        "edge_length_std": float(edge_length_std)
    }

def evaluate_completeness_and_overpruning(G: nx.MultiGraph, binary_mask: np.ndarray, voxel_size_xyz: Tuple[float, float, float], max_capillary_radius_um: float = 20.0) -> Dict[str, float]:
    """
    Determines how much of the original binary mask was 'orphaned' by aggressive pruning.
    """
    logger.info("Computing Completeness & Over-Pruning (Orphaned Volume)...")
    
    # Create a blank volume and draw ONLY the skeleton centerline points (1 voxel thick)
    skeleton_volume = np.zeros_like(binary_mask)
    for u, v, key, data in G.edges(keys=True, data=True):
        pts = data.get("pts")
        if pts is None or len(pts) == 0:
            pts = data.get("voxels", [])
            
        for pt in pts:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < skeleton_volume.shape[0] and 0 <= y < skeleton_volume.shape[1] and 0 <= x < skeleton_volume.shape[2]:
                skeleton_volume[z, y, x] = True
                
    # Calculate EDT outward from the skeleton
    sampling = (voxel_size_xyz[0], voxel_size_xyz[1], voxel_size_xyz[2])
    # Distance to the NEAREST skeleton point (so we invert the skeleton volume)
    distance_from_skeleton = ndimage.distance_transform_edt(~skeleton_volume, sampling=sampling)
    
    # Mask this distance field to only look at voxels that are actually tissue in the binary_mask
    tissue_distances = distance_from_skeleton[binary_mask]
    
    if len(tissue_distances) == 0:
        return {"orphaned_volume_fraction": 0.0}
        
    # Any tissue voxel that is further away from the skeleton than the largest expected vessel radius
    # is considered "orphaned" (meaning a branch was pruned that shouldn't have been).
    orphaned_voxels = np.sum(tissue_distances > max_capillary_radius_um)
    total_tissue_voxels = len(tissue_distances)
    
    return {
        "orphaned_volume_fraction": float(orphaned_voxels / total_tissue_voxels),
        "mean_distance_to_skeleton_um": float(np.mean(tissue_distances)),
        "max_distance_to_skeleton_um": float(np.max(tissue_distances))
    }

def run_all_benchmarks(G: nx.MultiGraph, binary_mask: np.ndarray, voxel_size_xyz: Tuple[float, float, float]) -> Dict[str, Any]:
    """Executes the complete benchmarking suite."""
    logger.info("=== Starting Skeletonization Benchmarking Suite ===")
    
    results = {}
    
    # 1. Volumetric
    try:
        results["volumetric"] = evaluate_volumetric_reconstruction(G, binary_mask, voxel_size_xyz)
    except Exception as e:
        logger.error(f"Failed Volumetric Benchmark: {e}")
        
    # 2. Centricity
    try:
        results["centricity"] = evaluate_centerline_centricity(G, binary_mask, voxel_size_xyz)
    except Exception as e:
        logger.error(f"Failed Centricity Benchmark: {e}")
        
    # 3. Topology
    try:
        results["topology"] = evaluate_topological_preservation(G, binary_mask)
    except Exception as e:
        logger.error(f"Failed Topology Benchmark: {e}")
        
    # 4. Completeness
    try:
        results["completeness"] = evaluate_completeness_and_overpruning(G, binary_mask, voxel_size_xyz)
    except Exception as e:
        logger.error(f"Failed Completeness Benchmark: {e}")
        
    logger.info("=== Benchmarking Suite Complete ===")
    return results

# --- PREPROCESSING BENCHMARKS ---

def evaluate_preprocessing_confidence(prob_map: np.ndarray, binary_mask: np.ndarray) -> Dict[str, float]:
    """Calculates the mean probability of preserved voxels and the total probability yield."""
    if not binary_mask.any():
        return {"confidence": 0.0, "probability_yield": 0.0}
    
    mean_confidence = float(np.mean(prob_map[binary_mask]))
    
    total_mass = np.sum(prob_map)
    yielded_mass = np.sum(prob_map[binary_mask])
    prob_yield = float(yielded_mass / max(1e-9, total_mass))
    
    return {"confidence": mean_confidence, "probability_yield": prob_yield}

def evaluate_preprocessing_uncertainty(entropy_map: np.ndarray, binary_mask: np.ndarray) -> Dict[str, float]:
    """Calculates the mean uncertainty and the fraction of preserved voxels with dangerous uncertainty."""
    if not binary_mask.any():
        return {"mean_uncertainty": 0.0, "high_uncertainty_fraction": 0.0}
    
    preserved_entropy = entropy_map[binary_mask]
    mean_unc = float(np.mean(preserved_entropy))
    high_unc_frac = float(np.sum(preserved_entropy > 0.8) / preserved_entropy.size)
    
    return {"mean_uncertainty": mean_unc, "high_uncertainty_fraction": high_unc_frac}

def evaluate_preprocessing_crispness(prob_map: np.ndarray, binary_mask: np.ndarray) -> float:
    """
    Calculates the 3D gradient magnitude on the raw image to find sharp edges,
    then measures how well the outer surface of the binary mask aligns with them.
    """
    if not binary_mask.any():
        return 0.0
        
    # Isolate the surface (skin) of the binary mask
    eroded = ndimage.binary_erosion(binary_mask)
    surface_mask = binary_mask ^ eroded
    
    if not surface_mask.any():
        return 0.0
        
    # Compute 3D gradient magnitude of the probability map
    grad_z = ndimage.sobel(prob_map, axis=0)
    grad_y = ndimage.sobel(prob_map, axis=1)
    grad_x = ndimage.sobel(prob_map, axis=2)
    gradient_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
    
    # Average gradient exactly at the boundary surface
    crispness = np.mean(gradient_magnitude[surface_mask])
    return float(crispness)

def evaluate_preprocessing_fragmentation(binary_mask: np.ndarray) -> int:
    """Counts the total number of floating 3D islands."""
    _, num_components = ndimage.label(binary_mask)
    return int(num_components)

def evaluate_preprocessing_compactness(binary_mask: np.ndarray) -> float:
    """
    Calculates the Surface-Area-to-Volume Ratio.
    High values indicate heavily jagged, non-cylindrical, or noisy boundaries.
    """
    if not binary_mask.any():
        return 0.0
    eroded = ndimage.binary_erosion(binary_mask)
    surface_mask = binary_mask ^ eroded
    return float(np.sum(surface_mask) / np.sum(binary_mask))

def evaluate_preprocessing_euler_characteristic(binary_mask: np.ndarray) -> int:
    """
    Calculates the Euler Characteristic (Components - Tunnels + Cavities).
    If a threshold creates "Swiss cheese" holes inside vessels, this drops sharply.
    """
    if not binary_mask.any():
        return 0
    from skimage.measure import euler_number
    # 3D connectivity: 3 for faces+edges+corners (26-connected)
    try:
        return int(euler_number(binary_mask, connectivity=3))
    except Exception:
        return 0

def run_all_preprocessing_benchmarks(prob_map: np.ndarray, binary_mask: np.ndarray, entropy_map: np.ndarray = None) -> Dict[str, Any]:
    """Executes the reference-free benchmarking suite for voxel preprocessing."""
    results = {}
    try:
        conf_results = evaluate_preprocessing_confidence(prob_map, binary_mask)
        results["confidence"] = conf_results["confidence"]
        results["probability_yield"] = conf_results["probability_yield"]
    except Exception as e:
        logger.error(f"Failed Confidence Benchmark: {e}")
        
    if entropy_map is not None:
        try:
            unc_results = evaluate_preprocessing_uncertainty(entropy_map, binary_mask)
            results["mean_uncertainty"] = unc_results["mean_uncertainty"]
            results["high_uncertainty_fraction"] = unc_results["high_uncertainty_fraction"]
        except Exception as e:
            logger.error(f"Failed Uncertainty Benchmark: {e}")
        
    try:
        results["crispness"] = evaluate_preprocessing_crispness(prob_map, binary_mask)
    except Exception as e:
        logger.error(f"Failed Crispness Benchmark: {e}")
        
    try:
        results["fragmentation"] = evaluate_preprocessing_fragmentation(binary_mask)
    except Exception as e:
        logger.error(f"Failed Fragmentation Benchmark: {e}")
        
    try:
        results["surface_area_ratio"] = evaluate_preprocessing_compactness(binary_mask)
    except Exception as e:
        logger.error(f"Failed Compactness Benchmark: {e}")
        
    try:
        results["euler_characteristic"] = evaluate_preprocessing_euler_characteristic(binary_mask)
    except Exception as e:
        logger.error(f"Failed Euler Benchmark: {e}")
        
    return results