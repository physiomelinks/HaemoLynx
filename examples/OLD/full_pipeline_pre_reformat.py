# %%
#Import dependencies
import logging

logger = logging.getLogger("vessel_analysis")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import networkx as nx
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, label, binary_dilation,  map_coordinates, distance_transform_edt
from skimage.util import img_as_bool
from scipy.spatial import KDTree, cKDTree
from skimage.morphology import remove_small_objects, binary_dilation, skeletonize_3d
from scipy.interpolate import splprep, splev
from collections import defaultdict, deque
import pandas as pd
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.graph import route_through_array
import math
import plotly.graph_objects as go
from skimage.measure import profile_line
from functools import lru_cache
import os
from typing import List, Tuple, Dict, Any, Union, Optional
try:
    from skan import csr
except ImportError:
    pass
from plotly.subplots import make_subplots
from scipy.integrate import quad
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import matplotlib.animation as animation
import random
import threading
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage

# the below is for the new ImageLynx package
from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization

# %%
#User input
# PARAMETERS
# TODO This wasn't used.
# filepath = "C:\\Users\\hd01\\Dropbox\\Modelling - nerve vasculature\\2425_Fem_DAPI_FITCdextran_NG2dsRed_24112023_a_2425_Fem_DAPI_FI...laments_manuallyaddedfilaments_06112024.ims Resolution Level 1 - C=10_cropped.tif"


diameter_by_branch_order = {
        'BO1': 6.2,  # Arterioles
        'BO2': 4.0,   # BO1
        'BO3': 5.0,   # BO2
        'BO4': 5.0,   
        'BO5': 4.0,
        'BO6': 4.0,   
        'BO7': 4.0,   
        'BO8': 4.0, 
        'BO9': 4.0,   
        'B10': 4.0,
        'B11': 4.0,   
        'B12': 4.0,   
        'B13': 4.0,
        'B14': 4.0,
        'B15': 4.0,   
        'B16': 4.0,   
        'B17': 4.0, 
        'B18': 4.0,
        'B19': 4.0,   
        'B20': 4.0,   
        'B21': 4.0,
        'B22': 4.0,   
        'B23': 4.0,
        'B24': 4.0,
        'B25': 4.0,   
        'B26': 4.0,
    }

#Diameter configuration with d1 (passive) and d2 (constricted) diameters
diameter_by_branch_order_enhanced = {
    'BO1': {'d1': 6.2, 'd2': 6.2},  # d1 is passive, d2 is constricted
    'BO2': {'d1': 4.0, 'd2': 3.2},
    'BO3': {'d1': 5.0, 'd2': 4.0},
    'BO4': {'d1': 5.0, 'd2': 4.0},
    'BO5': {'d1': 4.0, 'd2': 3.2},
    'BO6': {'d1': 4.0, 'd2': 3.2},
    'BO7': {'d1': 4.0, 'd2': 3.2},
    'BO8': {'d1': 4.0, 'd2': 3.2},
    'BO9': {'d1': 4.0, 'd2': 3.2},
    'B10': {'d1': 4.0, 'd2': 3.2},
    'B11': {'d1': 4.0, 'd2': 3.2},
    'B12': {'d1': 4.0, 'd2': 3.2},
    'B13': {'d1': 4.0, 'd2': 3.2},
    'B14': {'d1': 4.0, 'd2': 3.2},
    'B15': {'d1': 4.0, 'd2': 3.2},
    'B16': {'d1': 4.0, 'd2': 3.2},
    'B17': {'d1': 4.0, 'd2': 3.2},
    'B18': {'d1': 4.0, 'd2': 3.2},
    'B19': {'d1': 4.0, 'd2': 3.2},
    'B20': {'d1': 4.0, 'd2': 3.2},
    'B21': {'d1': 4.0, 'd2': 3.2},
    'B22': {'d1': 4.0, 'd2': 3.2},
    'B23': {'d1': 4.0, 'd2': 3.2},
    'B24': {'d1': 4.0, 'd2': 3.2},
    'B25': {'d1': 4.0, 'd2': 3.2},
    'B26': {'d1': 4.0, 'd2': 3.2},
}

#Arteriole start point (ie first node of tree, B01 = terminal arteriole) 
starting_nodes = [426, 184, 509, 494, 483]   
#Endoneurial vessel ID
custom_edges = [
        (103, 262),
        (103, 104),
        (309, 363),
        (363, 746),
        (363, 745),
        (746, 874),
        (745, 766),
        (874, 1140),
        (221, 309),
        (103, 106),
        (34, 222),
        (222, 258),
        (233, 236),
        (123, 176),
        (234, 235),
        (35, 65),
        (32, 35),
        (260,290),
        (290,846),
        (290,846),
        (766, 846),
        (766, 845)
    ]  


# %%
###Load file via tiff or via h5###


def load_and_skeletonize_3d_tif(filepath, voxel_size=1.0):
    logger.debug("Loading and skeletonizing TIFF...")
    image = tifffile.imread(filepath)
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)
    skeleton = skeletonize_3d(img_as_bool(bridged))
    return image, skeleton

def bridge_gaps(binary_skeleton, max_gap=4):
    dist = distance_transform_edt(~binary_skeleton)
    fill_mask = (dist <= max_gap) & (~binary_skeleton)
    filled = binary_skeleton | fill_mask
    return filled

def load_and_skeletonize_3d_h5(filepath, dataset_name=None, voxel_size=1.0):
    logger.debug("Loading and skeletonizing H5...")
    
    with h5py.File(filepath, 'r') as f:
        # Check if the specified dataset exists
        if dataset_name not in f:
            available_datasets = list(f.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
        
        # Load the image data
        image = f[dataset_name][:]
        
        # Convert to numpy array if needed and ensure proper data type
        image = np.array(image)
        
        logger.debug(f"Original image shape: {image.shape}")
    
    # Handle 4D and 5D datasets by squeezing singleton dimensions
    image = simplify_to_3d(image)
    logger.debug(f"Simplified image shape: {image.shape}")
    
    # Apply the same processing pipeline as the original function
    threshold = threshold_otsu(image)
    binary = image > threshold
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled)  # Note: you'll need to define or import bridge_gaps
    skeleton = skeletonize_3d(img_as_bool(bridged))
    
    return image, skeleton

def simplify_to_3d(image):
    if image.ndim == 3:
        return image
    elif image.ndim < 3:
        raise ValueError(f"Image has {image.ndim} dimensions. Need at least 3D data.")
    else:
        # For higher dimensions, take first 3 spatial dimensions
        logger.warning(f"Image has {image.ndim} dimensions. Taking first 3 spatial dimensions and first channel.")
        return image[:, :, :, 0, 0, 0] if image.ndim == 6 else image[:, :, :, 0]
    
    return image

image, skeleton = load_and_skeletonize_3d_tif("C://Users//hd01//Dropbox//710//For density analysis//4w Males//C1-Zstack_animal1_4w_male_1587_19102024.tif")
#image, skeleton = load_and_skeletonize_3d_h5('C://Users//hd01//Dropbox//710//For density analysis//4w Males//C1-Zstack2_animal1_4w_male_1587_19102024_Simple Segmentation.h5')


# %%
###Skeleton cleaning###
def preprocess_skeleton_for_graph(skeleton_image, min_branch_length=5):
    """
    Removes small objects to limit degree 2 nodes.
    """
    # Remove small connected components
    cleaned = remove_small_objects(skeleton_image, min_size=min_branch_length)
    
    # Re-skeletonize to ensure proper skeleton properties
    cleaned = skeletonize_3D(cleaned > 0)
    
    return cleaned.astype(bool)

skeleton = preprocess_skeleton_for_graph(skeleton, min_branch_length=10)
sk = csr.Skeleton(skeleton)

# %%
def build_graph_segment_skan_stitched_loops(
    sk, skeleton_image, debug=False, reconnect_threshold=3.0, 
    max_voxel_graph_size=100000, use_spatial_index=True
):
 
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if sk is None or skeleton_image is None:
        raise ValueError("sk and skeleton_image cannot be None")
    
    if sk.n_paths == 0:
        logger.warning("No paths found in skeleton")
        return nx.Graph(), []
    
    paths = [(i, sk.path_coordinates(i)) for i in range(sk.n_paths)]
    skel = skeleton_image
    ndim = skel.ndim
    
    # Memory-efficient loop detection
    foreground = np.argwhere(skel)
    voxel_loops = []
    
    # Only build voxel graph if manageable size
    if len(foreground) <= max_voxel_graph_size:
        offsets = np.argwhere(generate_binary_structure(ndim, 1)) - 1
        voxel_graph = nx.Graph()
        
        def process_pt_batch(pts_batch):
            """Process points in batches to reduce memory overhead"""
            edges = []
            for pt in pts_batch:
                for off in offsets:
                    nb = pt + off
                    if (np.all(nb >= 0) and np.all(nb < skel.shape) and 
                        skel[tuple(nb)]):
                        edges.append((tuple(pt), tuple(nb)))
            return edges
        
        # Process in batches to control memory usage
        batch_size = min(1000, len(foreground))
        batches = [foreground[i:i + batch_size] 
                  for i in range(0, len(foreground), batch_size)]
        
        max_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_edges in executor.map(process_pt_batch, batches):
                voxel_graph.add_edges_from(batch_edges)
        
        try:
            voxel_loops = nx.cycle_basis(voxel_graph)
            if debug:
                logger.debug(f"Found {len(voxel_loops)} voxel loops")
        except Exception as e:
            logger.warning(f"Loop detection failed: {e}")
            voxel_loops = []
    else:
        if debug:
            logger.warning(f"Skeleton too large ({len(foreground)} voxels) for loop detection")
    
    # Create loop voxel set with normalized coordinates
    loop_vox = set()
    for loop in voxel_loops:
        for v in loop:
            # Ensure consistent coordinate representation
            if isinstance(v, (list, tuple, np.ndarray)):
                loop_vox.add(tuple(np.round(v).astype(int)))
    
    def make_segment_safe(pid_path):
        """Safely create segment with validation"""
        pid, path = pid_path
        if len(path) < 2:
            return None
        
        # Validate path coordinates are within bounds
        path_array = np.array(path)
        if (np.any(path_array < 0) or 
            np.any(path_array >= np.array(skel.shape))):
            if debug:
                logger.warning(f"Path {pid} contains out-of-bounds coordinates")
            # Clip to valid bounds
            path_array = np.clip(path_array, 0, np.array(skel.shape) - 1)
        
        # Convert to integer coordinates consistently
        segment = [tuple(np.round(p).astype(int)) for p in path_array]
        
        # Remove duplicate consecutive points
        unique_segment = [segment[0]]
        for i in range(1, len(segment)):
            if segment[i] != segment[i-1]:
                unique_segment.append(segment[i])
        
        return unique_segment if len(unique_segment) >= 2 else None
    
    # Process segments with better error handling
    max_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        segments = [s for s in executor.map(make_segment_safe, paths) if s]
    
    if not segments:
        logger.warning("No valid segments found")
        return nx.Graph(), voxel_loops
    
    # Build graph with consistent coordinate handling
    G = nx.Graph()
    loop_edges = set()
    mapping = {}
    
    for seg_idx, seg in enumerate(segments):
        if len(seg) < 2:
            continue
            
        u_vox, v_vox = seg[0], seg[-1]
        
        # Create unique node IDs
        uid = mapping.setdefault(u_vox, len(mapping))
        vid = mapping.setdefault(v_vox, len(mapping))
        
        # Store positions as numpy arrays for consistency
        u_pos = np.array(u_vox, dtype=float)
        v_pos = np.array(v_vox, dtype=float)
        
        # Add nodes with minimal data (avoid redundant voxel storage)
        if not G.has_node(uid):
            G.add_node(uid, pos=u_pos)
        if not G.has_node(vid):
            G.add_node(vid, pos=v_pos)
        
        # Calculate distance properly
        seg_array = np.array(seg, dtype=float)
        if len(seg_array) > 1:
            distances = np.linalg.norm(np.diff(seg_array, axis=0), axis=1)
            total_dist = np.sum(distances)
        else:
            total_dist = 0.0
        
        # Add edge with essential data only
        G.add_edge(uid, vid, 
                  weight=max(total_dist, 1e-6),  # Avoid zero weights
                  length=len(seg), 
                  voxels=seg,
                  segment_id=seg_idx)
        
        # Track loop edges with normalized representation
        if u_vox in loop_vox and v_vox in loop_vox:
            edge = tuple(sorted([uid, vid]))  # Normalize edge representation
            loop_edges.add(edge)
    
    # Efficient terminal reconnection
    if reconnect_threshold and reconnect_threshold > 0:
        terminals = [n for n in G.nodes if G.degree[n] == 1]
        
        if len(terminals) > 1:
            # Use spatial indexing for efficiency
            if use_spatial_index and len(terminals) > 10:
                terminal_coords = np.array([G.nodes[n]['pos'] for n in terminals])
                tree = cKDTree(terminal_coords)
                
                # Find all pairs within threshold
                pairs_indices = tree.query_pairs(reconnect_threshold)
                pairs = []
                
                for i, j in pairs_indices:
                    src, tgt = terminals[i], terminals[j]
                    edge_norm = tuple(sorted([src, tgt]))
                    
                    # Skip if already connected or in loop
                    if (G.has_edge(src, tgt) or edge_norm in loop_edges or
                        G.degree[src] >= 3 or G.degree[tgt] >= 3):
                        continue
                    
                    dist = np.linalg.norm(terminal_coords[i] - terminal_coords[j])
                    pairs.append((dist, src, tgt))
            else:
                # Fallback to O(n²) for small numbers of terminals
                pairs = []
                for i, src in enumerate(terminals):
                    for j in range(i + 1, len(terminals)):
                        tgt = terminals[j]
                        edge_norm = tuple(sorted([src, tgt]))
                        
                        if (edge_norm in loop_edges or
                            G.degree[src] >= 3 or G.degree[tgt] >= 3):
                            continue
                        
                        src_pos = np.array(G.nodes[src]['pos'])
                        tgt_pos = np.array(G.nodes[tgt]['pos'])
                        dist = np.linalg.norm(src_pos - tgt_pos)
                        
                        if dist <= reconnect_threshold:
                            pairs.append((dist, src, tgt))
            
            # Process reconnections in order of distance
            heapq.heapify(pairs)
            reconnected = 0
            
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                
                # Re-validate conditions (graph may have changed)
                if (G.has_edge(src, tgt) or 
                    G.degree[src] >= 3 or G.degree[tgt] >= 3):
                    continue
                
                # Create reconnection edge
                src_pos = np.array(G.nodes[src]['pos'])
                tgt_pos = np.array(G.nodes[tgt]['pos'])
                
                G.add_edge(src, tgt, 
                          weight=max(dist, 1e-6),
                          length=2,
                          voxels=[tuple(src_pos.astype(int)), 
                                 tuple(tgt_pos.astype(int))],
                          reconnected=True)
                
                reconnected += 1
                if debug:
                    logger.debug(f"Reconnected {src}-{tgt}, d={dist:.2f}")
            
            if debug and reconnected > 0:
                logger.info(f"Reconnected {reconnected} terminal pairs")
    
    # Final validation
    isolated_nodes = [n for n in G.nodes if G.degree[n] == 0]
    if isolated_nodes:
        G.remove_nodes_from(isolated_nodes)
        if debug:
            logger.warning(f"Removed {len(isolated_nodes)} isolated nodes")
    
    if debug:
        logger.info(f"Final graph: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges, "
                   f"{len(loop_edges)} loop edges")
    
    return G, voxel_loops, loop_edges

def reconnect_secondary_loop_edges(
    G, skeleton,
    voxel_size=(1,1,1),
    min_length_voxels=30,
    max_length_voxels=6000,
    max_distance=6000.0,
    margin=10,
    k_paths=5,
    min_overlap=0.9, #0.7
    min_geom_dev=8.0,
    repulsion_sigma=2.0,
    max_workers=None,
    debug=True,
    max_cache_size=1000
):
    
    logger = logging.getLogger(__name__)
    
    # Input validation
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise ValueError("G must be a NetworkX Graph or MultiGraph")
    if skeleton is None or skeleton.size == 0:
        raise ValueError("skeleton cannot be None or empty")
    
    # Convert to MultiGraph if it isn't already
    if not isinstance(G, nx.MultiGraph):
        if debug:
            logger.info("Converting Graph to MultiGraph to support parallel edges")
        G = nx.MultiGraph(G)
    
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        logger.warning("No node positions found")
        return G
    
    deg = dict(G.degree())
    skeleton_copy = skeleton.astype(bool)
    
    # Pre-compute distance transform (expensive operation)
    try:
        base_cost = 1 + distance_transform_edt(~skeleton_copy) ** 2
    except Exception as e:
        logger.error(f"Failed to compute distance transform: {e}")
        return G

    # Thread-safe cache with size limit
    cache_lock = threading.Lock()
    sub_cache = {}
    cache_access_order = []

    def manage_cache(key, value=None):
        """Thread-safe cache management with LRU eviction"""
        with cache_lock:
            if value is not None:  # Adding to cache
                if key in sub_cache:
                    # Move to end
                    cache_access_order.remove(key)
                    cache_access_order.append(key)
                else:
                    # Add new entry
                    if len(sub_cache) >= max_cache_size:
                        # Evict oldest
                        oldest = cache_access_order.pop(0)
                        del sub_cache[oldest]
                    sub_cache[key] = value
                    cache_access_order.append(key)
                return value
            else:  # Getting from cache
                if key in sub_cache:
                    # Move to end (most recently used)
                    cache_access_order.remove(key)
                    cache_access_order.append(key)
                    return sub_cache[key]
                return None

    def make_repulsion_safe(orig_voxels, sub_shape):
        """Safely create repulsion field with bounds checking"""
        if not orig_voxels or not sub_shape or any(s <= 0 for s in sub_shape):
            return np.zeros(sub_shape, dtype=float)
        
        mask = np.zeros(sub_shape, dtype=float)
        valid_count = 0
        
        for coords in orig_voxels:
            if len(coords) >= 3:
                x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
                if (0 <= x < sub_shape[0] and 
                    0 <= y < sub_shape[1] and 
                    0 <= z < sub_shape[2]):
                    mask[x, y, z] = 1.0
                    valid_count += 1
        
        if valid_count == 0:
            return np.zeros(sub_shape, dtype=float)
        
        try:
            # Create stronger repulsion field
            repulsion_field = gaussian_filter(mask, sigma=repulsion_sigma)
            
            # Scale repulsion strength based on distance from skeleton
            # Higher repulsion where skeleton is present
            max_repulsion = np.max(repulsion_field)
            if max_repulsion > 0:
                # Normalize and amplify repulsion
                repulsion_field = (repulsion_field / max_repulsion) * 50.0  # Strong repulsion
                
                # Add extra penalty directly on original path voxels
                high_penalty_mask = mask > 0
                repulsion_field[high_penalty_mask] += 100.0  # Very high cost on exact path
            
            return repulsion_field
        except Exception as e:
            logger.warning(f"Gaussian filter failed: {e}")
            # Fallback: direct high cost on original voxels
            fallback_mask = mask.copy()
            fallback_mask[mask > 0] = 100.0  # High penalty on original path
            return fallback_mask

    # Find candidate pairs with better filtering
    candidates = []
    for u, v, key, data in G.edges(data=True, keys=True):
        if (deg[u] == 2 and deg[v] == 2 and 
            data.get("voxels") and
            u in pos and v in pos and
            not data.get("secondary", False)):  # Only consider primary edges
            
            distance = np.linalg.norm(np.subtract(pos[u], pos[v]))
            if distance <= max_distance:
                candidates.append((u, v, distance))
    
    # Remove duplicates (since MultiGraph might have multiple edges between same nodes)
    seen_pairs = set()
    unique_candidates = []
    for u, v, dist in candidates:
        pair = tuple(sorted([u, v]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_candidates.append((u, v, dist))
    
    # Sort by distance for better cache efficiency
    unique_candidates.sort(key=lambda x: x[2])
    pairs = [(u, v) for u, v, _ in unique_candidates]
    
    if debug:
        logger.info(f"{len(pairs)} candidate deg‑2 pairs")

    def attempt_reconnect(pair_data):
        """Attempt to find alternative paths between nodes"""
        u, v, node_positions = pair_data
        try:
            # Use explicitly passed node_positions to avoid closure issues
            if u not in node_positions or v not in node_positions:
                return None
            
            pu, pv = np.array(node_positions[u]), np.array(node_positions[v])
            u_vox = np.round(pu / voxel_size).astype(int)
            v_vox = np.round(pv / voxel_size).astype(int)

            # Get original voxels with validation (from primary edge only)
            primary_edge_data = None
            for key, edge_data in G[u][v].items():
                if not edge_data.get("secondary", False):
                    primary_edge_data = edge_data
                    break
            
            if not primary_edge_data:
                return None
                
            orig_voxels_raw = primary_edge_data.get("voxels", [])
            if not orig_voxels_raw:
                return None

            # Convert and validate original voxels
            orig_voxels = []
            for vox in orig_voxels_raw:
                if isinstance(vox, (list, tuple, np.ndarray)) and len(vox) >= 3:
                    vox_coords = np.round(np.array(vox) / voxel_size).astype(int)
                    if np.all(vox_coords >= 0) and np.all(vox_coords < skeleton_copy.shape):
                        orig_voxels.append(vox_coords)

            if not orig_voxels:
                return None

            best_paths = []

            # Try different expansion levels
            for expansion in [0, 10, 25, 50]:
                ext = margin + expansion
                minc = np.maximum(np.minimum(u_vox, v_vox) - ext, 0)
                maxc = np.minimum(np.maximum(u_vox, v_vox) + ext + 1, skeleton_copy.shape)
                
                # Validate subvolume
                if np.any(minc >= maxc):
                    continue
                
                cache_key = (*minc, *maxc)
                cached_result = manage_cache(cache_key)
                
                if cached_result is None:
                    # Create subvolume cost map
                    try:
                        sub_cost = base_cost[minc[0]:maxc[0], 
                                           minc[1]:maxc[1], 
                                           minc[2]:maxc[2]].copy()
                        
                        if sub_cost.size == 0:
                            continue
                        
                        # Add repulsion from original path
                        orig_rel = [vox - minc for vox in orig_voxels]
                        repulsion = make_repulsion_safe(orig_rel, sub_cost.shape)
                        sub_cost = sub_cost + repulsion
                        
                        cached_result = (sub_cost, minc)
                        manage_cache(cache_key, cached_result)
                        
                    except Exception as e:
                        if debug:
                            logger.warning(f"Subvolume creation failed for {u}-{v}: {e}")
                        continue
                
                sub_cost, minc = cached_result
                
                # Calculate local coordinates
                ru = u_vox - minc
                rv = v_vox - minc
                
                # Validate bounds
                if (np.any(ru < 0) or np.any(rv < 0) or 
                    np.any(ru >= sub_cost.shape) or np.any(rv >= sub_cost.shape)):
                    continue

                # Use faster pathfinding algorithm
                try:
                    path_coords, cost = route_through_array(
                        sub_cost, tuple(ru), tuple(rv), fully_connected=True
                    )
                    
                    if path_coords is None or len(path_coords) < min_length_voxels:
                        continue
                    
                    path_coords = np.array(path_coords)
                    path_length = len(path_coords)
                    
                    if path_length > max_length_voxels:
                        continue
                    
                    # Convert to absolute coordinates
                    abs_coords = path_coords + minc
                    
                    # Validate coordinates are within skeleton bounds
                    if (np.any(abs_coords < 0) or 
                        np.any(abs_coords >= skeleton.shape)):
                        continue
                    
                    # Calculate metrics safely
                    try:
                        x, y, z = abs_coords.T
                        skeleton_hits = skeleton[x, y, z]
                        overlap = np.sum(skeleton_hits) / path_length
                        
                        if overlap < min_overlap:
                            continue
                        
                        # Calculate geometric deviation (how different from original)
                        orig_coords = np.array(orig_voxels)
                        hausdorff_dist = max(
                            directed_hausdorff(orig_coords, abs_coords)[0],
                            directed_hausdorff(abs_coords, orig_coords)[0]
                        )
                        
                        if hausdorff_dist < min_geom_dev:
                            if debug:
                                logger.debug(f"Path too similar to original (dev={hausdorff_dist:.1f} < {min_geom_dev})")
                            continue
                        
                        # Additional novelty check: path overlap percentage
                        orig_set = set(tuple(coord) for coord in orig_coords)
                        new_set = set(tuple(coord) for coord in abs_coords)
                        overlap_voxels = len(orig_set.intersection(new_set))
                        path_similarity = overlap_voxels / min(len(orig_set), len(new_set))
                        
                        # Reject paths that are too similar (>70% overlap)
                        if path_similarity > 0.7:
                            if debug:
                                logger.debug(f"Path too similar (voxel overlap={path_similarity:.2f})")
                            continue
                        
                        # Convert back to physical coordinates
                        vox3d = (abs_coords * voxel_size).tolist()
                        
                        # Calculate path novelty metrics
                        unique_voxels = len(new_set - orig_set)
                        path_novelty = unique_voxels / len(new_set)  # Fraction of unique voxels
                        
                        best_paths.append({
                            'voxels': vox3d,
                            'overlap': overlap,
                            'deviation': hausdorff_dist,
                            'length': path_length,
                            'cost': cost,
                            'novelty': path_novelty,  # How different from original
                            'voxel_similarity': path_similarity  # For debugging
                        })
                        
                        # Stop if we have enough good paths
                        if len(best_paths) >= k_paths:
                            break
                            
                    except Exception as e:
                        if debug:
                            logger.warning(f"Metric calculation failed: {e}")
                        continue
                        
                except Exception as e:
                    if debug:
                        logger.warning(f"Pathfinding failed for {u}-{v}: {e}")
                    continue
                
                # Break if we found good paths
                if best_paths:
                    break

            if not best_paths:
                return None

            # Sort by novelty first, then overlap and deviation
            best_paths.sort(key=lambda p: (-p['novelty'], -p['overlap'], -p['deviation']))
            
            return u, v, best_paths[:k_paths]

        except Exception as e:
            if debug:
                logger.error(f"Attempt failed for {u}-{v}: {e}")
            return None

    # Process pairs with thread-safe edge addition
    added = 0
    edge_lock = threading.Lock()
    
    max_workers = max_workers or min(4, len(pairs))
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Pass node positions explicitly to avoid closure issues
            pair_data_list = [(u, v, pos) for u, v in pairs]
            future_to_pair = {executor.submit(attempt_reconnect, pair_data): (pair_data[0], pair_data[1]) 
                             for pair_data in pair_data_list}
            
            for future in as_completed(future_to_pair):
                u, v = future_to_pair[future]
                try:
                    result = future.result()
                    if not result:
                        continue
                    
                    u, v, candidates = result
                    
                    if debug:
                        logger.info(f"{u}-{v} → {len(candidates)} candidates")
                    
                    # Thread-safe edge addition
                    with edge_lock:
                        # Check if there's already a secondary edge between these nodes
                        has_secondary = False
                        if G.has_edge(u, v):
                            for key, edge_data in G[u][v].items():
                                if edge_data.get("secondary", False):
                                    has_secondary = True
                                    break
                        
                        if has_secondary:
                            if debug:
                                logger.info(f"{u}-{v} already has secondary edge")
                            continue
                        
                        # Add new parallel edge (MultiGraph automatically assigns new key)
                        best = candidates[0]
                        new_edge_key = G.add_edge(u, v,
                                                weight=best['length'],
                                                voxels=best['voxels'],
                                                length=best['length'],
                                                overlap=best['overlap'],
                                                deviation=best['deviation'],
                                                novelty=best['novelty'],
                                                secondary=True)
                        
                        added += 1
                        if debug:
                            logger.info(f"Added secondary edge {u}-{v} (key={new_edge_key}): "
                                      f"novelty={best['novelty']:.2f}, overlap={best['overlap']:.2f}, "
                                      f"dev={best['deviation']:.1f}, length={best['length']}")
                            
                            # Show how different it is from original
                            if 'voxel_similarity' in best:
                                logger.info(f"   Path similarity to original: {best['voxel_similarity']:.2f} "
                                          f"(lower is more novel)")
                            
                            # Verify original edge is preserved
                            primary_count = sum(1 for key, data in G[u][v].items() 
                                              if not data.get("secondary", False))
                            secondary_count = sum(1 for key, data in G[u][v].items() 
                                                if data.get("secondary", False))
                            logger.info(f"   Edge {u}-{v} now has {primary_count} primary + "
                                      f"{secondary_count} secondary edges")
                            
                except Exception as e:
                    if debug:
                        logger.error(f"Processing failed for {u}-{v}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Threading failed: {e}")
        # Return original graph on threading failure
        return G

    # Clear cache to free memory
    with cache_lock:
        sub_cache.clear()
        cache_access_order.clear()

    if debug:
        logger.info(f"Done: added {added} secondary edges")

    return G

def optimise_graph_topology_fixed(
    G, voxel_loops, loop_edges, skeleton_data=None, debug=False, reconnect_threshold=3.0,
    use_spatial_index=True, remove_degree2_nodes=True, 
    consolidation_threshold=2.0, improve_junctions=True,
    preserve_multigraph=True, validate_reconnections=True,
    aggressive_degree2_cleanup_level=1  # 0=conservative, 1=normal, 2=aggressive, 3=extra aggressive
):

    logger = logging.getLogger(__name__)
    
# Improved terminal reconnection with skeleton validation
    if reconnect_threshold and reconnect_threshold > 0:
        valid_nodes = [n for n in G.nodes() if 'pos' in G.nodes[n]]
        terminals = [n for n in valid_nodes if G.degree[n] == 1]
        
        if len(terminals) > 1:
            # Get terminal pairs within threshold (existing logic)
            if use_spatial_index and len(terminals) > 10:
                terminal_coords = np.array([G.nodes[n]['pos'] for n in terminals])
                tree = cKDTree(terminal_coords)
                pairs_indices = tree.query_pairs(reconnect_threshold)
                pairs = []
                
                for i, j in pairs_indices:
                    src, tgt = terminals[i], terminals[j]
                    edge_norm = tuple(sorted([src, tgt]))
                    
                    # Skip if already connected or in loop
                    if (G.has_edge(src, tgt) or edge_norm in loop_edges or
                        G.degree[src] >= 3 or G.degree[tgt] >= 3):
                        continue
                    
                    dist = np.linalg.norm(terminal_coords[i] - terminal_coords[j])
                    pairs.append((dist, src, tgt))
            else:
                # Fallback logic (keep existing)
                pairs = []
                for i, src in enumerate(terminals):
                    if 'pos' not in G.nodes[src]:
                        continue
                    for j in range(i + 1, len(terminals)):
                        tgt = terminals[j]
                        if 'pos' not in G.nodes[tgt]:
                            continue
                            
                        edge_norm = tuple(sorted([src, tgt]))
                        
                        if (edge_norm in loop_edges or
                            G.degree[src] >= 3 or G.degree[tgt] >= 3):
                            continue
                        
                        src_pos = np.array(G.nodes[src]['pos'])
                        tgt_pos = np.array(G.nodes[tgt]['pos'])
                        dist = np.linalg.norm(src_pos - tgt_pos)
                        
                        if dist <= reconnect_threshold:
                            pairs.append((dist, src, tgt))
            
            #Validate and create proper reconnections
            heapq.heapify(pairs)
            reconnected = 0
            
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                
                # Re-validate conditions
                if (not G.has_node(src) or not G.has_node(tgt) or
                    'pos' not in G.nodes[src] or 'pos' not in G.nodes[tgt] or
                    G.has_edge(src, tgt) or 
                    G.degree[src] >= 3 or G.degree[tgt] >= 3):
                    continue
                
                src_pos = np.array(G.nodes[src]['pos'])
                tgt_pos = np.array(G.nodes[tgt]['pos'])
                
                #Validate connection against skeleton data
                if validate_reconnections and skeleton_data is not None:
                    connection_valid, voxel_path = validate_skeleton_connection(
                        skeleton_data, src_pos, tgt_pos, max_gap=reconnect_threshold
                    )
                    
                    if not connection_valid:
                        if debug:
                            logger.debug(f"Skipped reconnection {src}-{tgt}: no skeleton path")
                        continue
                    
                    # Create edge with proper voxel path
                    add_edge_safe(G, src, tgt, 
                              weight=max(dist, 1e-6),
                              length=len(voxel_path) if voxel_path else dist,
                              voxels=voxel_path if voxel_path else [tuple(src_pos.astype(int)), 
                                                                   tuple(tgt_pos.astype(int))],
                              reconnected=True,
                              validated=True)
                else:
                    # OPTION 1: More conservative reconnection (recommended)
                    # Only reconnect if distance is very small (likely genuine gap)
                    conservative_threshold = min(reconnect_threshold * 0.5, 1.5)
                    if dist > conservative_threshold:
                        if debug:
                            logger.debug(f"Skipped reconnection {src}-{tgt}: distance {dist:.2f} > conservative threshold {conservative_threshold:.2f}")
                        continue
                    
                    # OPTION 2: Create edge with proper distance-based properties
                    add_edge_safe(G, src, tgt, 
                              weight=max(dist, 1e-6),
                              length=dist,  # Use actual distance, not arbitrary value
                              voxels=[tuple(src_pos.astype(int)), 
                                     tuple(tgt_pos.astype(int))],
                              reconnected=True,
                              conservative=True)
                
                reconnected += 1
                if debug:
                    logger.debug(f"Reconnected {src}-{tgt}, d={dist:.2f}")
            
            if debug and reconnected > 0:
                logger.info(f"Reconnected {reconnected} terminal pairs")

    
    return G, voxel_loops

def validate_skeleton_connection(skeleton_data, pos1, pos2, max_gap=3.0):
    """
    Validate that there's actually a skeleton path between two positions.
    
    Args:
        skeleton_data: Binary skeleton image (3D numpy array)
        pos1, pos2: Positions to connect
        max_gap: Maximum allowed gap in skeleton
        
    Returns:
        (is_valid, voxel_path): Tuple of validation result and path voxels
    """
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize_3d
    from skimage.measure import label
    
    try:
        # Convert positions to integer coordinates
        p1 = np.round(pos1).astype(int)
        p2 = np.round(pos2).astype(int)
        
        # Check bounds
        if (not (0 <= p1[0] < skeleton_data.shape[0] and 
                0 <= p1[1] < skeleton_data.shape[1] and 
                0 <= p1[2] < skeleton_data.shape[2]) or
            not (0 <= p2[0] < skeleton_data.shape[0] and 
                0 <= p2[1] < skeleton_data.shape[1] and 
                0 <= p2[2] < skeleton_data.shape[2])):
            return False, None
        
        # Method 1: Check if positions are connected in skeleton
        # Create a small region around the line between points
        line_points = get_line_points_3d(p1, p2)
        
        # Check if most line points are near skeleton voxels
        skeleton_nearby = 0
        for point in line_points:
            # Check small neighborhood around each point
            region = skeleton_data[
                max(0, point[0]-1):min(skeleton_data.shape[0], point[0]+2),
                max(0, point[1]-1):min(skeleton_data.shape[1], point[1]+2),
                max(0, point[2]-1):min(skeleton_data.shape[2], point[2]+2)
            ]
            if np.any(region > 0):
                skeleton_nearby += 1
        
        # Connection is valid if most of the line follows skeleton
        connection_ratio = skeleton_nearby / len(line_points)
        is_valid = connection_ratio > 0.7  # 70% of path should be near skeleton
        
        if is_valid:
            return True, line_points
        else:
            return False, None
            
    except Exception as e:
        # If validation fails, be conservative
        return False, None

def safer_simple_remove_all_degree2_nodes(G, max_degree=4, debug=False, max_iterations=100, 
                                         max_edge_length_ratio=2.0):
    logger = logging.getLogger(__name__)
    total_removed = 0
    skipped_long_edges = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    
    if debug:
        initial_degree2 = len([n for n in G.nodes() if G.degree[n] == 2])
        logger.info(f"Starting safer cleanup with {initial_degree2} degree-2 nodes")
    
    for iteration in range(max_iterations):
        removed_this_iter = 0
        
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        
        if not degree2_nodes:
            break
        
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
                
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
                
            n1, n2 = neighbors
            
            # Don't create high-degree nodes
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            
            # Get edge data
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            node_pos = G.nodes[node].get('pos', None)
            
            if not edge1_data_list or not edge2_data_list:
                continue
            
            # SAFETY CHECK: Validate that merged edge won't be too long/artificial
            if 'pos' in G.nodes[n1] and 'pos' in G.nodes[n2] and node_pos is not None:
                # Calculate current path length through the degree-2 node
                pos1 = np.array(G.nodes[n1]['pos'])
                pos2 = np.array(G.nodes[n2]['pos'])
                node_pos_arr = np.array(node_pos)
                
                current_path_length = (np.linalg.norm(pos1 - node_pos_arr) + 
                                     np.linalg.norm(node_pos_arr - pos2))
                direct_distance = np.linalg.norm(pos2 - pos1)
                
                # Skip if the direct connection would be much shorter than the path
                # (indicates the degree-2 node represents a genuine bend/curve)
                if current_path_length > direct_distance * max_edge_length_ratio:
                    if debug:
                        logger.debug(f"Skipped removing node {node}: would create short-cut "
                                   f"(path: {current_path_length:.2f}, direct: {direct_distance:.2f})")
                    skipped_long_edges += 1
                    continue
            
            # Proceed with removal
            G.remove_node(node)
            
            # Create properly merged edges
            if is_multigraph:
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_attrs = create_merged_edge_attributes(
                            edge1_data, edge2_data, node_pos
                        )
                        add_edge_safe(G, n1, n2, **merged_attrs)
            else:
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                
                merged_attrs = create_merged_edge_attributes(
                    edge1_data, edge2_data, node_pos
                )
                
                if has_edge_safe(G, n1, n2):
                    remove_edge_safe(G, n1, n2)
                
                add_edge_safe(G, n1, n2, **merged_attrs)
            
            removed_this_iter += 1
            total_removed += 1
        
        if debug and removed_this_iter > 0:
            remaining = len([n for n in G.nodes() if G.degree[n] == 2])
            logger.debug(f"Iteration {iteration + 1}: removed {removed_this_iter}, {remaining} remain")
        
        if removed_this_iter == 0:
            break
    
    if debug:
        final_degree2 = len([n for n in G.nodes() if G.degree[n] == 2])
        logger.info(f"Safer cleanup: removed {total_removed}, skipped {skipped_long_edges} long edges")
        logger.info(f"Final degree-2 nodes: {final_degree2}")
    
    return G

def trivial_remove_all_degree2_nodes(G, max_degree=4, debug=False):
    
    total_removed = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    
    # Keep iterating until no more degree-2 nodes
    max_iterations = 100
    for iteration in range(max_iterations):
        
        # Find all degree-2 nodes
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        
        if not degree2_nodes:
            break  # Done!
        
        removed_this_iter = 0
        
        for node in degree2_nodes:
            # Double-check it's still degree-2 (may have changed during iteration)
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            
            n1, n2 = neighbors
            
            # ONLY constraint: don't create super high-degree nodes
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            
            # Get the node's position
            node_pos = G.nodes[node].get('pos', None)
            
            # Get edge data from both edges
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            
            if not edge1_data_list or not edge2_data_list:
                continue
            
            # STEP 1: Remove the degree-2 node
            G.remove_node(node)
            
            # STEP 2: Create single edge between neighbors with merged path
            if is_multigraph:
                # For multigraphs, create edges for all combinations
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_edge = create_trivial_merged_edge(edge1_data, edge2_data, node_pos)
                        G.add_edge(n1, n2, **merged_edge)
            else:
                # For simple graphs, merge into single edge
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                
                merged_edge = create_trivial_merged_edge(edge1_data, edge2_data, node_pos)
                
                # Remove existing edge if present
                if G.has_edge(n1, n2):
                    G.remove_edge(n1, n2)
                
                # Add the new merged edge
                G.add_edge(n1, n2, **merged_edge)
            
            removed_this_iter += 1
            total_removed += 1
            
            if debug:
                print(f"Removed degree-2 node {node}, connected {n1}-{n2}")
        
        if debug:
            remaining = len([n for n in G.nodes() if G.degree[n] == 2])
            print(f"Iteration {iteration + 1}: removed {removed_this_iter}, {remaining} degree-2 nodes remain")
        
        if removed_this_iter == 0:
            break  # No progress, stop
    
    final_count = len([n for n in G.nodes() if G.degree[n] == 2])
    
    if debug:
        print(f"TRIVIAL REMOVAL COMPLETE:")
        print(f"  Total removed: {total_removed}")
        print(f"  Final degree-2 nodes: {final_count}")
    
    return G
def create_trivial_merged_edge(edge1_data, edge2_data, removed_node_pos):
    """
    Create merged edge with exact topology preservation.
    Just concatenate paths and include the removed node's position.
    """
    
    # Get voxel paths
    voxels1 = edge1_data.get('voxels', [])
    voxels2 = edge2_data.get('voxels', [])
    
    # Create merged path: path1 + removed_node + path2
    merged_voxels = []
    
    # Add first path
    if voxels1:
        merged_voxels.extend(voxels1)
    
    # Add the removed node's position (this preserves exact topology)
    if removed_node_pos is not None:
        node_voxel = tuple(removed_node_pos)
        # Only add if not already present (avoid duplicates)
        if not merged_voxels or merged_voxels[-1] != node_voxel:
            merged_voxels.append(node_voxel)
    
    # Add second path  
    if voxels2:
        # Skip first voxel of second path if it's the same as removed node
        start_idx = 0
        if (voxels2 and removed_node_pos is not None and 
            len(voxels2) > 0 and tuple(voxels2[0]) == tuple(removed_node_pos)):
            start_idx = 1
        merged_voxels.extend(voxels2[start_idx:])
    
    # Merge other attributes simply
    merged_attributes = {
        'weight': edge1_data.get('weight', 0) + edge2_data.get('weight', 0),
        'length': edge1_data.get('length', 0) + edge2_data.get('length', 0),
        'voxels': merged_voxels,
        'merged': True,
        'trivial_merge': True,
        'removed_node_pos': removed_node_pos
    }
    
    # Preserve any other attributes from the first edge
    for key, value in edge1_data.items():
        if key not in merged_attributes:
            merged_attributes[key] = value
    
    return merged_attributes

def smart_multigraph_degree2_removal(G, skeleton_data=None, max_degree=4, debug=False):
    """
    Smart degree-2 removal for MultiGraphs.
    
    CORE LOGIC:
    1. Find degree-2 nodes
    2. Remove the node
    3. Merge the two connecting edges
    4. Improve straight edges using skeleton data
    5. Preserve curved edge topology
    6. Never exceed max_degree
    """
    import networkx as nx
    import numpy as np
    
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("This function is designed specifically for MultiGraphs")
    
    print("=== SMART MULTIGRAPH DEGREE-2 REMOVAL ===")
    
    total_removed = 0
    
    for iteration in range(100):
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        
        if not degree2_nodes:
            break
        
        removed_this_iter = 0
        
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            
            n1, n2 = neighbors
            
            # Check degree constraints
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            
            # Get positions
            node_pos = G.nodes[node].get('pos', None)
            n1_pos = G.nodes[n1].get('pos', None)
            n2_pos = G.nodes[n2].get('pos', None)
            
            if node_pos is None or n1_pos is None or n2_pos is None:
                continue
            
            # Get all edges between node and its neighbors
            edges_to_n1 = list(G[node][n1].values()) if G.has_edge(node, n1) else []
            edges_to_n2 = list(G[node][n2].values()) if G.has_edge(node, n2) else []
            
            if not edges_to_n1 or not edges_to_n2:
                continue
            
            # Remove the degree-2 node
            G.remove_node(node)
            
            # Merge each pair of edges intelligently
            for edge1_data in edges_to_n1:
                for edge2_data in edges_to_n2:
                    
                    voxels1 = edge1_data.get('voxels', [])
                    voxels2 = edge2_data.get('voxels', [])
                    
                    if debug:
                        print(f"  Merging edges: {len(voxels1)} + {len(voxels2)} voxels")
                    
                    # Create merged edge with topology preservation/improvement
                    merged_voxels = merge_edges_with_topology_improvement(
                        voxels1, voxels2,
                        np.array(n1_pos), np.array(node_pos), np.array(n2_pos),
                        skeleton_data, debug
                    )
                    
                    # Calculate attributes for merged edge
                    merged_attrs = {
                        'weight': edge1_data.get('weight', 0) + edge2_data.get('weight', 0),
                        'length': calculate_path_length(merged_voxels),
                        'voxels': merged_voxels,
                        'merged': True,
                        'original_edges': 2
                    }
                    
                    # Check for similar edges to avoid duplicates
                    should_add, replace_key = should_add_merged_edge(
                        G, n1, n2, merged_voxels, merged_attrs, debug
                    )
                    
                    if should_add:
                        if replace_key is not None:
                            G.remove_edge(n1, n2, key=replace_key)
                            if debug:
                                print(f"     Replaced inferior edge")
                        
                        G.add_edge(n1, n2, **merged_attrs)
                        if debug:
                            print(f"     Added merged edge (length: {merged_attrs['length']:.1f})")
                    else:
                        if debug:
                            print(f"     Skipped duplicate edge")
            
            removed_this_iter += 1
            total_removed += 1
        
        if removed_this_iter == 0:
            break
    
    if debug:
        final_count = len([n for n in G.nodes() if G.degree[n] == 2])
        print(f"Smart removal: {total_removed} removed, {final_count} remain")
    
    return G

def merge_edges_with_topology_improvement(voxels1, voxels2, pos1, node_pos, pos2, skeleton_data, debug=False):
    """
    Merge two edges while improving straight segments using skeleton data.
    
    Strategy:
    - Curved edges: preserve as-is
    - Straight edges: improve using skeleton topology
    - Mixed: improve straight part, preserve curved part
    """
    import numpy as np
    
    # Analyze edge types
    is_curved1 = is_path_curved(voxels1)
    is_curved2 = is_path_curved(voxels2)
    
    if debug:
        print(f"    Edge1: {'CURVED' if is_curved1 else 'STRAIGHT'} ({len(voxels1)} voxels)")
        print(f"    Edge2: {'CURVED' if is_curved2 else 'STRAIGHT'} ({len(voxels2)} voxels)")
    
    # Case 1: Both edges are curved - preserve both
    if is_curved1 and is_curved2:
        merged = merge_curved_edges(voxels1, voxels2, node_pos, debug)
        if debug:
            print(f"    → Merged curved edges: {len(merged)} voxels")
        return merged
    
    # Case 2: Mixed curved/straight - improve straight edge
    elif is_curved1 and not is_curved2:
        improved_voxels2 = improve_straight_edge_with_skeleton(
            node_pos, pos2, skeleton_data, debug
        )
        if improved_voxels2:
            merged = merge_curved_edges(voxels1, improved_voxels2, node_pos, debug)
            if debug:
                print(f"    → Improved straight edge2 and merged: {len(merged)} voxels")
            return merged
        else:
            # Fallback: use original straight edge
            merged = merge_curved_edges(voxels1, voxels2, node_pos, debug)
            if debug:
                print(f"    → Could not improve edge2, used original: {len(merged)} voxels")
            return merged
    
    elif not is_curved1 and is_curved2:
        improved_voxels1 = improve_straight_edge_with_skeleton(
            pos1, node_pos, skeleton_data, debug
        )
        if improved_voxels1:
            merged = merge_curved_edges(improved_voxels1, voxels2, node_pos, debug)
            if debug:
                print(f"    → Improved straight edge1 and merged: {len(merged)} voxels")
            return merged
        else:
            # Fallback: use original straight edge
            merged = merge_curved_edges(voxels1, voxels2, node_pos, debug)
            if debug:
                print(f"    → Could not improve edge1, used original: {len(merged)} voxels")
            return merged
    
    # Case 3: Both edges are straight - try to improve entire path
    else:
        improved_full_path = improve_straight_path_with_skeleton(
            pos1, pos2, skeleton_data, debug
        )
        if improved_full_path:
            if debug:
                print(f"    → Improved entire straight path: {len(improved_full_path)} voxels")
            return improved_full_path
        else:
            # Fallback: merge original straight edges
            merged = merge_curved_edges(voxels1, voxels2, node_pos, debug)
            if debug:
                print(f"    → Could not improve path, merged originals: {len(merged)} voxels")
            return merged
def prune_vascular_stubs(G: Union[nx.Graph, nx.MultiGraph], 
                        min_stub_length: float = 10.0,
                        max_iterations: int = 100,
                        debug: bool = False, 
                        voxel_size: Tuple[float, float, float] = (1, 1, 1)) -> Union[nx.Graph, nx.MultiGraph]:
    """
    Iteratively removes short terminal stubs from a vascular network graph until convergence.
    
    This function identifies terminal nodes (degree 1) that are connected by edges shorter 
    than the specified threshold and removes them. The process is repeated until no more 
    short stubs are found, as removing one stub may expose new terminal nodes.
    """
    # Input validation
    if min_stub_length < 0:
        raise ValueError("min_stub_length must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must be a 3-tuple")
    
    # Work on a copy to avoid modifying the original
    G_pruned = G.copy()
    
    if G_pruned.number_of_nodes() == 0:
        return G_pruned
    
    total_removed = 0
    iteration = 0
    graph_type = "MultiGraph" if isinstance(G_pruned, nx.MultiGraph) else "Graph"
    
    if debug:
        print(f"Starting stub pruning on {graph_type} with {G_pruned.number_of_nodes()} nodes")
        print(f"Minimum stub length threshold: {min_stub_length}")
    
    while iteration < max_iterations:
        iteration += 1
        nodes_before = G_pruned.number_of_nodes()
        
        if nodes_before == 0:
            break
            
        # Find and remove short terminal stubs in this iteration
        nodes_to_remove = []
        
        # Find all terminal nodes (degree 1)
        terminal_nodes = [node for node in G_pruned.nodes() if G_pruned.degree(node) == 1]
        
        for node in terminal_nodes:
            if node not in G_pruned:  # Node might have been removed already
                continue
                
            neighbors = list(G_pruned.neighbors(node))
            if not neighbors:  # Isolated node
                nodes_to_remove.append(node)
                continue
                
            neighbor = neighbors[0]
            
            # Calculate edge length
            try:
                if isinstance(G_pruned, nx.MultiGraph):
                    # For MultiGraph, consider all parallel edges and use the minimum length
                    edge_data_list = list(G_pruned[node][neighbor].values())
                    edge_length = min(
                        calculate_edge_length(node, neighbor, edge_data, voxel_size) 
                        for edge_data in edge_data_list
                    )
                else:
                    # For regular Graph
                    edge_data = G_pruned[node][neighbor]
                    edge_length = calculate_edge_length(node, neighbor, edge_data, voxel_size)
                
                # Check if this is a short stub
                if edge_length < min_stub_length:
                    nodes_to_remove.append(node)
                    if debug:
                        print(f"  Iteration {iteration}: Marking node {node} for removal "
                              f"(stub length: {edge_length:.2f})")
                        
            except Exception as e:
                if debug:
                    print(f"  Warning: Could not calculate edge length for node {node}: {e}")
                # If we can't calculate length, assume it's a stub to be safe
                nodes_to_remove.append(node)
        
        # Remove the identified nodes
        G_pruned.remove_nodes_from(nodes_to_remove)
        
        nodes_after = G_pruned.number_of_nodes()
        removed_this_iteration = nodes_before - nodes_after
        total_removed += removed_this_iteration
        
        if debug:
            print(f"  Iteration {iteration}: Removed {removed_this_iteration} nodes "
                  f"({nodes_after} remaining)")
        
        # Check for convergence
        if removed_this_iteration == 0:
            if debug:
                print(f"Convergence reached after {iteration} iterations")
            break
    
    else:
        # Loop completed without break (max iterations reached)
        raise RuntimeWarning(
            f"Maximum iterations ({max_iterations}) reached without full convergence. "
            f"Consider increasing max_iterations or checking for issues in the graph."
        )
    
    if debug:
        efficiency = (total_removed / G.number_of_nodes()) * 100 if G.number_of_nodes() > 0 else 0
        print(f"\nPruning complete:")
        print(f"  Total nodes removed: {total_removed}")
        print(f"  Final node count: {G_pruned.number_of_nodes()}")
        print(f"  Pruning efficiency: {efficiency:.1f}%")
        
        # Additional diagnostics
        remaining_terminals = [n for n in G_pruned.nodes() if G_pruned.degree(n) == 1]
        print(f"  Remaining terminal nodes: {len(remaining_terminals)}")
    
    return G_pruned

#Produce graph
G, loops, edges = build_graph_segment_skan_stitched_loops(
        sk, skeleton,
        debug=True,                    
        max_voxel_graph_size=50000)

#Improve graph (run in order)
G = reconnect_secondary_loop_edges(G, skeleton, debug=True) 
G, voxel_loops = optimise_graph_topology_fixed(
    G, loops, edges, 
    skeleton,  # Pass original skeleton
    reconnect_threshold=3.0,           # Conservative threshold
    validate_reconnections=True,
    aggressive_degree2_cleanup_level=0
)
G = safer_simple_remove_all_degree2_nodes(G, max_degree=5, max_edge_length_ratio=2.0)
G = trivial_remove_all_degree2_nodes(G, max_degree=5, debug=True)
G = smart_multigraph_degree2_removal(G, skeleton)
G = prune_vascular_stubs(G)
G = smart_multigraph_degree2_removal(G, skeleton)

#Assign branch orders and resistance:
def assign_branch_orders(G, starting_nodes):
    # Initialize tracking structures
    edge_distances = {}  # {(u, v, key): min_distance}
    node_distances = {}  # {node: min_distance}
    
    # Initialize BFS queue with all starting nodes at distance 0
    queue = deque()
    for start_node in starting_nodes:
        if start_node in G.nodes():
            queue.append((start_node, 0))
            node_distances[start_node] = 0
        else:
            logger.warning(f"Starting node {start_node} not found in graph")
    
    # BFS to find minimum distances to all nodes
    while queue:
        current_node, distance = queue.popleft()
        
        # Skip if we've already found a shorter path to this node
        if current_node in node_distances and node_distances[current_node] < distance:
            continue
            
        # Explore all neighboring nodes
        for neighbor in G.neighbors(current_node):
            new_distance = distance + 1
            
            # Only update if this is a shorter path
            if neighbor not in node_distances or node_distances[neighbor] > new_distance:
                node_distances[neighbor] = new_distance
                queue.append((neighbor, new_distance))
    
    # Assign branch orders to edges based on node distances
    results = {
        'edges_assigned': 0,
        'edges_skipped': 0,
        'branch_order_counts': defaultdict(int),
        'unreachable_edges': []
    }
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get minimum distance to either endpoint
        u_dist = node_distances.get(u, float('inf'))
        v_dist = node_distances.get(v, float('inf'))
        
        if u_dist == float('inf') and v_dist == float('inf'):
            results['unreachable_edges'].append((u, v, key))
            results['edges_skipped'] += 1
            continue
        
        # Edge branch order is minimum distance to either endpoint + 1
        # (since edge connects nodes at different levels)
        edge_distance = min(u_dist, v_dist) + 1
        branch_order = f"BO{edge_distance}"
        
        # Assign branch order attribute
        G[u][v][key]['branch_order'] = branch_order
        
        results['edges_assigned'] += 1
        results['branch_order_counts'][branch_order] += 1
        
        logger.debug(f"Edge ({u}, {v}, {key}): {branch_order} "
                    f"(u_dist={u_dist}, v_dist={v_dist})")
    
    return results

assign_branch_orders(G, starting_nodes)



# %%
#Analysis/statistics functions for downstream stuff
#***For epineurial vessels***
#Constricted pericytes
set_poiseuille_weights_with_constrictions(G, diameter_by_branch_order_enhanced)

def set_poiseuille_weights_with_constrictions(G, diameter_by_branch_order):
    """
    Set edge weights using integrated Poiseuille's law with diameter constrictions.
    Weight = 1 / (integrated resistance over vessel length)
    
    Parameters:
    -----------
    G : networkx.MultiGraph
        The multigraph with branch_order and length attributes
    diameter_by_branch_order : dict
        Dictionary mapping branch order strings to diameter dictionaries
        e.g., {'BO1': {'d1': 6.2, 'd2': 4.0}, 'BO2': {'d1': 4.5, 'd2': 3.0}}
        where d1 is passive diameter and d2 is constricted diameter
        
    Returns:
    --------
    dict : Summary of weight assignments
    """
    results = {
        'weights_set': 0,
        'missing_branch_order': [],
        'missing_length': [],
        'unknown_branch_order': [],
        'invalid_length': [],
        'invalid_diameter': [],
        'diameter_calculations': {}  # Track diameter info for each branch order
    }
    
    print(f"=== Enhanced Poiseuille Weight Calculation (With Constrictions) ===")
    print(f"Formula: Weight = 1 / (∫ resistance dx)")
    print(f"Resistance per unit length = (128 * μ) / (π * d^4)")
    print(f"Viscosity calculation: μ = 1 / diameter^1.647")
    print(f"Constriction pattern: d1 → d2 → d1 over 40μm, spaced 100μm apart")
    print(f"Units: diameter and length in micrometers (μm)")
    print()
    
    # Validate and display diameter information
    for branch_order, diameters in diameter_by_branch_order.items():
        if not isinstance(diameters, dict) or 'd1' not in diameters or 'd2' not in diameters:
            print(f"Error: {branch_order} must have dict with 'd1' and 'd2' keys")
            continue
        
        d1, d2 = diameters['d1'], diameters['d2']
        if d1 <= 0 or d2 <= 0:
            print(f"Warning: Invalid diameters for {branch_order}: d1={d1}, d2={d2}")
            continue
        
        results['diameter_calculations'][branch_order] = {
            'd1_passive': d1,
            'd2_constricted': d2,
            'viscosity_d1': calculate_viscosity(d1),
            'viscosity_d2': calculate_viscosity(d2)
        }
        print(f"{branch_order}: d1={d1}μm (μ={calculate_viscosity(d1):.6f}), "
              f"d2={d2}μm (μ={calculate_viscosity(d2):.6f})")
    
    print()
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Check for branch order
        branch_order = data.get('branch_order', None)
        if branch_order is None:
            results['missing_branch_order'].append((u, v, key))
            continue
        
        # Check for length
        length = data.get('length', None)
        if length is None:
            results['missing_length'].append((u, v, key))
            continue
        
        if length <= 0:
            results['invalid_length'].append((u, v, key, length))
            continue
        
        # Get diameters for this branch order
        diameters = diameter_by_branch_order.get(branch_order, None)
        if diameters is None:
            results['unknown_branch_order'].append((u, v, key, branch_order))
            continue
        
        if not isinstance(diameters, dict) or 'd1' not in diameters or 'd2' not in diameters:
            results['invalid_diameter'].append((u, v, key, branch_order, "missing d1/d2"))
            continue
        
        d1, d2 = diameters['d1'], diameters['d2']
        if d1 <= 0 or d2 <= 0:
            results['invalid_diameter'].append((u, v, key, branch_order, f"d1={d1}, d2={d2}"))
            continue
        
        # Calculate integrated resistance
        try:
            total_resistance = calculate_integrated_resistance(length, d1, d2)
            
            # Weight is conductance (inverse of resistance)
            weight = 1.0 / total_resistance
            
            # Store old weight for comparison
            old_weight = data.get('weight', None)
            
            # Set new weight
            G[u][v][key]['weight'] = weight
            
            results['weights_set'] += 1
            
            # Debug output for first few edges
            if results['weights_set'] <= 5:
                print(f"Edge ({u}, {v}, {key}): {branch_order}, length={length:.3f}μm, "
                      f"d1={d1}μm, d2={d2}μm, resistance={total_resistance:.6f}, weight={weight:.6f}")
                
        except Exception as e:
            print(f"Error calculating resistance for edge ({u}, {v}, {key}): {e}")
            results['invalid_diameter'].append((u, v, key, branch_order, str(e)))
            continue
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Weights successfully set: {results['weights_set']}")
    if results['missing_branch_order']:
        print(f"Edges missing branch_order: {len(results['missing_branch_order'])}")
    if results['missing_length']:
        print(f"Edges missing length: {len(results['missing_length'])}")
    if results['unknown_branch_order']:
        print(f"Edges with unknown branch_order: {len(results['unknown_branch_order'])}")
    if results['invalid_length']:
        print(f"Edges with invalid length: {len(results['invalid_length'])}")
    if results['invalid_diameter']:
        print(f"Edges with invalid diameter: {len(results['invalid_diameter'])}")
    
    return results

#Passive diameter pericytes
#set_poiseuille_weights(G, diameter_by_branch_order)
def calculate_viscosity(diameter):
    """Calculate viscosity using the relationship: μ = 1 / diameter^1.647"""
    return 1.0 / (diameter ** 1.647)

def resistance_integrand(position, length, d1, d2):
    """
    Calculate resistance per unit length at a given position.
    
    Resistance per unit length = (128 * viscosity) / (π * diameter^4)
    """
    diameter = get_diameter_at_position(position, length, d1, d2)
    viscosity = calculate_viscosity(diameter)
    return (128.0 * viscosity) / (np.pi * diameter**4)

def calculate_integrated_resistance(length, d1, d2, num_points=1000):
    """
    Calculate total resistance by integrating over vessel length.
    
    Parameters:
    -----------
    length : float
        Vessel length in micrometers
    d1 : float
        Passive diameter in micrometers
    d2 : float
        Constricted diameter in micrometers
    num_points : int
        Number of points for numerical integration
        
    Returns:
    --------
    float : Total resistance of the vessel
    """
    if length <= 0:
        return float('inf')
    
    # Use numerical integration (trapezoidal rule for efficiency)
    positions = np.linspace(0, length, num_points)
    resistances = [resistance_integrand(pos, length, d1, d2) for pos in positions]
    
    # Integrate using trapezoidal rule
    total_resistance = np.trapz(resistances, dx=length/(num_points-1))
    
    return total_resistance

#*** For endoneurial vessel subset (defined by user input custom_edges)***
set_poiseuille_edge_weights(G, custom_edges, 6, use_resistance=False)

def set_poiseuille_edge_weights(G, custom_edges, edge_diameter, use_resistance=True):
    """
    use_resistance : bool
        If True: weight = resistance = (128 * μ * L) / (π * d^4)
        If False: weight = conductance = (π * d^4) / (128 * μ * L)
        
    Returns:
    --------
    dict : Summary of changes made
    """
    import numpy as np
    
    PI = np.pi
    results = {
        'updated': [],
        'not_found': [],
        'no_length': [],
        'multiple_edges': [],
        'invalid_diameter': [],
        'formula_used': 'resistance' if use_resistance else 'conductance',
        'parameters': {
            'edge_diameter': edge_diameter,
            'calculated_viscosity': None
        }
    }
    
    # Validate diameter
    if edge_diameter <= 0:
        logger.error(f"Invalid edge_diameter {edge_diameter}. Must be positive.")
        results['invalid_diameter'].append(edge_diameter)
        return results
    
    # Calculate viscosity using the specified formula
    viscosity = 1 / (edge_diameter ** 1.647)
    results['parameters']['calculated_viscosity'] = viscosity
    
    # Handle both dict and list formats
    if isinstance(custom_edges, dict):
        edge_pairs = custom_edges.keys()
    else:
        edge_pairs = custom_edges
    
    print(f"=== Poiseuille Weight Calculation ===")
    print(f"Formula: {'Resistance' if use_resistance else 'Conductance'}")
    print(f"Edge Diameter: {edge_diameter} μm")
    print(f"Calculated Viscosity: {viscosity:.6f} (from 1/diameter^1.647)")
    if use_resistance:
        print(f"Weight = (128 * μ * L) / (π * d^4)")
    else:
        print(f"Weight = (π * d^4) / (128 * μ * L)")
    print()
    
    for edge_pair in edge_pairs:
        u, v = edge_pair
        
        # Check if edge exists (avoid processing same edge twice)
        edges_found = []
        
        # Check (u, v) first, then (v, u) only if (u, v) doesn't exist
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            for key in edge_data.keys():
                edges_found.append((u, v, key))
        elif u != v and G.has_edge(v, u):
            edge_data = G.get_edge_data(v, u)
            for key in edge_data.keys():
                edges_found.append((v, u, key))
        
        if not edges_found:
            results['not_found'].append(edge_pair)
            logger.warning(f"Edge {edge_pair} not found in graph")
            continue
        
        if len(edges_found) > 1:
            results['multiple_edges'].append((edge_pair, len(edges_found)))
            logger.info(f"Multiple edges found for {edge_pair}: {len(edges_found)} edges")
        
        # Process each edge found
        for u_actual, v_actual, key in edges_found:
            edge_data = G[u_actual][v_actual][key]
            
            # Get length attribute (vessel_length)
            vessel_length = edge_data.get('length', None)
            
            if vessel_length is None:
                results['no_length'].append((u_actual, v_actual, key))
                logger.warning(f"No 'length' attribute found for edge ({u_actual}, {v_actual}, key={key})")
                continue
            
            if vessel_length <= 0:
                logger.warning(f"Invalid length {vessel_length} for edge ({u_actual}, {v_actual}, key={key})")
                continue
            
            # Calculate weight using the specified Poiseuille's law formula
            if use_resistance:
                # Resistance: (128 * μ * L) / (π * d^4)
                new_weight = (128.0 * viscosity * vessel_length) / (PI * edge_diameter**4)
            else:
                # Conductance: (π * d^4) / (128 * μ * L)  
                new_weight = (PI * edge_diameter**4) / (128.0 * viscosity * vessel_length)
            
            # Store old weight for logging
            old_weight = edge_data.get('weight', None)
            
            # Set new weight
            G[u_actual][v_actual][key]['weight'] = new_weight
            
            results['updated'].append({
                'edge': (u_actual, v_actual, key),
                'vessel_length': vessel_length,
                'edge_diameter': edge_diameter,
                'calculated_viscosity': viscosity,
                'old_weight': old_weight,
                'new_weight': new_weight
            })
            
            logger.debug(f"Updated edge ({u_actual}, {v_actual}, key={key}): "
                        f"length={vessel_length}μm, diameter={edge_diameter}μm, "
                        f"viscosity={viscosity:.6f}, new_weight={new_weight:.6f}")
    
    return results
#Visualise and produce statistics
plot_node_degree_distribution(G)
def plot_node_degree_distribution(G, title="Node Degree Distribution"):
    """
    Enhanced plotting function with more details
    """
    import matplotlib.pyplot as plt
    
    degrees = [deg for _, deg in G.degree()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(1, max(degrees)+2), align='left', rwidth=0.8, alpha=0.7)
    plt.xlabel("Node Degree")
    plt.ylabel("Count")
    plt.title(f"{title} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
    plt.grid(True, alpha=0.3)
    
    # Add text with statistics
    degree_counts = {}
    for deg in degrees:
        degree_counts[deg] = degree_counts.get(deg, 0) + 1
    
    stats_text = f"Degree distribution:\n"
    for deg in sorted(degree_counts.keys()):
        stats_text += f"  {deg}: {degree_counts[deg]} nodes\n"
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return degree_counts

visualize_edges_and_nodes(image, G)

def visualize_edges_and_nodes(image, G):
    projection = np.max(image, axis=0)
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 10))
    plt.imshow(projection, cmap='gray')
    for u, v, d in G.edges(data=True):
        path = d.get("voxels", [])
        if len(path) > 1:
            path = np.array(path)
            plt.plot(path[:, 2], path[:, 1], color='cyan', linewidth=0.5)
    coords = np.array(list(pos.values()))
    plt.scatter(coords[:, 2], coords[:, 1], c='red', s=3)
    plt.title("Overlay: Edges and Nodes on Z-Projection")
    plt.axis('off')
    plt.show()
    
stats = compute_comprehensive_vessel_statistics(
    G, 
    node_positions=your_positions_dict,
    voxel_size=(1, 1, 1),  # in microns
    image_dimensions=(x_size, y_size, z_size)  # in voxels
)

def compute_comprehensive_vessel_statistics(G, node_positions=None, voxel_size=(1.0, 1.0, 1.0), image_dimensions=None):
    # Handle MultiGraph by converting to simple graph for some calculations
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    if is_multigraph:
        # For multigraphs, we'll work with the original for edge-specific measures
        # and create a simple graph for topology measures
        G_simple = nx.Graph(G) if not G.is_directed() else nx.DiGraph(G)
    else:
        G_simple = G
    
    # Basic graph statistics
    basic_stats = compute_basic_statistics(G, is_multigraph)
    
    # Tortuosity measures
    tortuosity_stats = compute_tortuosity_measures(G, node_positions, is_multigraph)
    
    # Branching analysis
    branching_stats = compute_branching_statistics(G_simple, node_positions)
    
    # Tree asymmetry
    asymmetry_stats = compute_tree_asymmetry(G_simple)
    
    # Fractal dimension
    fractal_stats = compute_fractal_dimension(G_simple, node_positions)
    
    # Path efficiency
    efficiency_stats = compute_path_efficiency(G, is_multigraph)
    
    # Vessel density
    density_stats = compute_vessel_density(G, node_positions, voxel_size, image_dimensions, is_multigraph)
    
    # Combine all statistics
    all_stats = {
        **basic_stats,
        **tortuosity_stats,
        **branching_stats,
        **asymmetry_stats,
        **fractal_stats,
        **efficiency_stats,
        **density_stats
    }
    
    return all_stats

def compute_basic_statistics(G, is_multigraph):
    """Compute basic graph statistics including edge length measures."""
    
    # Extract edge weights/lengths
    if is_multigraph:
        edge_weights = []
        for u, v, key, edge_data in G.edges(keys=True, data=True):
            if "weight" in edge_data:
                edge_weights.append(edge_data["weight"])
            elif "length" in edge_data:
                edge_weights.append(edge_data["length"])
    else:
        edge_weights = []
        for u, v, edge_data in G.edges(data=True):
            if "weight" in edge_data:
                edge_weights.append(edge_data["weight"])
            elif "length" in edge_data:
                edge_weights.append(edge_data["length"])
    
    # Calculate node degrees
    node_degrees = [G.degree(node) for node in G.nodes()]
    
    # Compute statistics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    total_edge_length = sum(edge_weights) if edge_weights else 0
    avg_degree = sum(node_degrees) / len(node_degrees) if node_degrees else 0
    avg_edge_length = sum(edge_weights) / len(edge_weights) if edge_weights else 0
    
    return {
        "Total Nodes": total_nodes,
        "Total Edges": total_edges,
        "Total Edge Length (microns)": total_edge_length,
        "Average Edge Length (microns)": avg_edge_length,
        "Average Degree": avg_degree
    }

def compute_tortuosity_measures(G, node_positions, is_multigraph):
    """Compute tortuosity index and average curvature measures."""
    
    if node_positions is None:
        return {
            "Average Tortuosity Index": "N/A (no position data)",
            "Average Curvature": "N/A (no position data)"
        }
    
    tortuosity_indices = []
    curvatures = []
    
    # Get all paths in the vessel tree
    if is_multigraph:
        edges_iter = G.edges(keys=True, data=True)
    else:
        edges_iter = G.edges(data=True)
    
    for edge_info in edges_iter:
        if is_multigraph:
            u, v, key, edge_data = edge_info
        else:
            u, v, edge_data = edge_info
        
        if u in node_positions and v in node_positions:
            pos_u = np.array(node_positions[u])
            pos_v = np.array(node_positions[v])
            
            # Straight-line distance
            straight_distance = euclidean(pos_u, pos_v)
            
            # Actual path length (from edge weight/length)
            path_length = edge_data.get("weight", edge_data.get("length", straight_distance))
            
            if straight_distance > 0:
                # Tortuosity index = actual path length / straight-line distance
                tortuosity_index = path_length / straight_distance
                tortuosity_indices.append(tortuosity_index)
                
                # Simple curvature approximation (more sophisticated methods would need intermediate points)
                # Curvature ≈ (path_length - straight_distance) / path_length
                if path_length > 0:
                    curvature = (path_length - straight_distance) / path_length
                    curvatures.append(curvature)
    
    avg_tortuosity = np.mean(tortuosity_indices) if tortuosity_indices else 0
    avg_curvature = np.mean(curvatures) if curvatures else 0
    
    return {
        "Average Tortuosity Index": avg_tortuosity,
        "Average Curvature": avg_curvature
    }

def compute_branching_statistics(G, node_positions):
    """Compute average branching angle and related statistics."""
    
    if node_positions is None:
        return {"Average Branching Angle (degrees)": "N/A (no position data)"}
    
    branching_angles = []
    
    # Find branching points (nodes with degree > 2)
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 3:  # Branching point
            # Calculate angles between all pairs of connected edges
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if node in node_positions and neighbors[i] in node_positions and neighbors[j] in node_positions:
                        center = np.array(node_positions[node])
                        point1 = np.array(node_positions[neighbors[i]])
                        point2 = np.array(node_positions[neighbors[j]])
                        
                        # Calculate vectors
                        vec1 = point1 - center
                        vec2 = point2 - center
                        
                        # Calculate angle
                        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                            angle = np.arccos(cos_angle)
                            branching_angles.append(np.degrees(angle))
    
    avg_branching_angle = np.mean(branching_angles) if branching_angles else 0
    
    return {
        "Average Branching Angle (degrees)": avg_branching_angle,
        "Number of Branching Points": len([n for n in G.nodes() if G.degree(n) > 2])
    }

def compute_tree_asymmetry(G):
    """Compute tree asymmetry index."""
    
    if not nx.is_tree(G):
        # For non-trees, we'll compute asymmetry on the spanning tree
        if nx.is_connected(G):
            G = nx.minimum_spanning_tree(G)
        else:
            return {"Tree Asymmetry Index": "N/A (disconnected graph)"}
    
    def calculate_subtree_asymmetry(node, parent=None):
        """Calculate asymmetry for subtrees rooted at node."""
        children = [n for n in G.neighbors(node) if n != parent]
        
        if len(children) == 0:
            return 0, 1  # asymmetry, size
        
        child_sizes = []
        total_asymmetry = 0
        
        for child in children:
            child_asymmetry, child_size = calculate_subtree_asymmetry(child, node)
            child_sizes.append(child_size)
            total_asymmetry += child_asymmetry
        
        # Calculate asymmetry at this node
        if len(child_sizes) >= 2:
            # Asymmetry = difference between largest and smallest subtree sizes
            node_asymmetry = max(child_sizes) - min(child_sizes)
        else:
            node_asymmetry = 0
        
        total_asymmetry += node_asymmetry
        total_size = sum(child_sizes) + 1
        
        return total_asymmetry, total_size
    
    # Find a reasonable root (node with highest degree or arbitrary if all equal)
    root = max(G.nodes(), key=G.degree)
    asymmetry, size = calculate_subtree_asymmetry(root)
    
    # Normalize by tree size
    normalized_asymmetry = asymmetry / size if size > 0 else 0
    
    return {"Tree Asymmetry Index": normalized_asymmetry}

def compute_fractal_dimension(G, node_positions):
    """Compute fractal dimension using box-counting method."""
    
    if node_positions is None or len(node_positions) < 2:
        return {"Fractal Dimension": "N/A (insufficient position data)"}
    
    # Get all positions
    positions = np.array([node_positions[node] for node in G.nodes() if node in node_positions])
    
    if len(positions) < 2:
        return {"Fractal Dimension": "N/A (insufficient position data)"}
    
    # Box-counting method
    box_sizes = []
    box_counts = []
    
    # Determine range of box sizes
    coord_ranges = positions.max(axis=0) - positions.min(axis=0)
    max_range = np.max(coord_ranges)
    
    # Use logarithmically spaced box sizes
    min_box_size = max_range / 100
    max_box_size = max_range / 2
    
    for box_size in np.logspace(np.log10(min_box_size), np.log10(max_box_size), 10):
        # Count boxes that contain at least one point
        min_coords = positions.min(axis=0)
        box_indices = ((positions - min_coords) / box_size).astype(int)
        unique_boxes = len(set(tuple(box_idx) for box_idx in box_indices))
        
        box_sizes.append(box_size)
        box_counts.append(unique_boxes)
    
    # Calculate fractal dimension using linear regression in log-log space
    if len(box_sizes) > 1 and all(count > 0 for count in box_counts):
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Linear regression: log(count) = -fractal_dim * log(size) + const
        fractal_dim = -np.polyfit(log_sizes, log_counts, 1)[0]
    else:
        fractal_dim = 0
    
    return {"Fractal Dimension": fractal_dim}

def compute_path_efficiency(G, is_multigraph):
    """Compute path efficiency measures using length as weight."""
    
    # For efficiency calculations, use simple graph
    if is_multigraph:
        G_simple = nx.Graph()
        G_simple.add_nodes_from(G.nodes())
        
        # Add edges with minimum weight for multiple edges
        edge_weights = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            weight = data.get("weight", data.get("length", 1))
            if (u, v) not in edge_weights or weight < edge_weights[(u, v)]:
                edge_weights[(u, v)] = weight
        
        for (u, v), weight in edge_weights.items():
            G_simple.add_edge(u, v, weight=weight)
    else:
        G_simple = G
    
    if not nx.is_connected(G_simple):
        return {"Path Efficiency": "N/A (disconnected graph)"}
    
    # Calculate average shortest path length weighted by edge lengths
    try:
        # Use weight attribute for shortest path calculations
        path_lengths = []
        nodes = list(G_simple.nodes())
        
        for i, source in enumerate(nodes):
            for target in nodes[i+1:]:
                try:
                    path_length = nx.shortest_path_length(G_simple, source, target, weight='weight')
                    path_lengths.append(path_length)
                except nx.NetworkXNoPath:
                    continue
        
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        
        # Efficiency is inverse of path length
        efficiency = 1 / avg_path_length if avg_path_length > 0 else 0
        
    except Exception:
        efficiency = 0
    
    return {
        "Path Efficiency": efficiency,
        "Average Shortest Path Length (microns)": avg_path_length if 'avg_path_length' in locals() else 0
    }

def compute_vessel_density(G, node_positions, voxel_size, image_dimensions, is_multigraph):
    """Compute vessel density in two ways: within vessel volume and whole image."""
    
    # Get total vessel length
    if is_multigraph:
        edge_lengths = [
            edge_data.get("weight", edge_data.get("length", 0))
            for u, v, key, edge_data in G.edges(keys=True, data=True)
        ]
    else:
        edge_lengths = [
            edge_data.get("weight", edge_data.get("length", 0))
            for u, v, edge_data in G.edges(data=True)
        ]
    
    total_vessel_length = sum(edge_lengths)
    
    density_stats = {"Total Vessel Length (microns)": total_vessel_length}
    
    # Density within vessel-occupied volume
    if node_positions is not None and len(node_positions) > 0:
        positions = np.array([node_positions[node] for node in G.nodes() if node in node_positions])
        
        if len(positions) > 0:
            # Calculate bounding box of vessels
            min_coords = positions.min(axis=0)
            max_coords = positions.max(axis=0)
            vessel_volume = np.prod(max_coords - min_coords)  # in cubic microns
            
            vessel_density_in_tissue = total_vessel_length / vessel_volume if vessel_volume > 0 else 0
            density_stats["Vessel Density in Tissue (microns/micron³)"] = vessel_density_in_tissue
            density_stats["Vessel-Occupied Volume (micron³)"] = vessel_volume
        else:
            density_stats["Vessel Density in Tissue (microns/micron³)"] = "N/A (no position data)"
    
    # Density in whole image
    if image_dimensions is not None and voxel_size is not None:
        # Convert image dimensions to microns
        image_volume_microns = np.prod([dim * voxel for dim, voxel in zip(image_dimensions, voxel_size)])
        vessel_density_whole_image = total_vessel_length / image_volume_microns if image_volume_microns > 0 else 0
        
        density_stats["Vessel Density in Whole Image (microns/micron³)"] = vessel_density_whole_image
        density_stats["Total Image Volume (micron³)"] = image_volume_microns
    else:
        density_stats["Vessel Density in Whole Image (microns/micron³)"] = "N/A (no image dimension data)"
    
    return density_stats



visualize_geometry_with_branch_orders(image, G, color_palette='rainbow', group_above=8)

def visualize_geometry_with_branch_orders(image, G, figsize=(12, 10), color_palette='viridis', 
                                        node_color='red', node_size=3, edge_linewidth=0.8,
                                        show_legend=True, background_cmap='gray', 
                                        save_path=None, dpi=300, alpha=0.8, reverse_gradient=True,
                                        group_above=None):
    
    # Create Z-projection of the image
    projection = np.max(image, axis=0)
    
    # Extract branch orders from edges
    all_branch_orders = set()
    visualizable_branch_orders = set()
    edge_branch_orders = {}
    edge_paths = {}
    
    edges_with_bo = 0
    edges_with_voxels = 0
    edges_with_both = 0
    
    for u, v, key, data in G.edges(keys=True, data=True):
        bo = data.get('branch_order', 'No_BO')
        path = data.get('voxels', [])
        
        if bo != 'No_BO':
            edges_with_bo += 1
            all_branch_orders.add(bo)
        
        if len(path) > 1:
            edges_with_voxels += 1
            
        if bo != 'No_BO' and len(path) > 1:
            edges_with_both += 1
            visualizable_branch_orders.add(bo)
        
        edge_branch_orders[(u, v, key)] = bo
        edge_paths[(u, v, key)] = path
    
    # Sort branch orders for consistent coloring
    all_branch_orders = sort_branch_orders_numerically(list(all_branch_orders))
    visualizable_branch_orders = sort_branch_orders_numerically(list(visualizable_branch_orders))
    
    print(f"=== Debugging Information ===")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Edges with branch_order: {edges_with_bo}")
    print(f"Edges with voxel paths: {edges_with_voxels}")
    print(f"Edges with both (visualizable): {edges_with_both}")
    print(f"All branch orders found: {all_branch_orders}")
    print(f"Visualizable branch orders: {visualizable_branch_orders}")
    
    # Use visualizable branch orders for the actual plotting
    branch_orders = visualizable_branch_orders if visualizable_branch_orders else all_branch_orders
    
    # Create color mapping with optional gradient reversal and grouping
    color_mapping = create_color_mapping(branch_orders, color_palette, reverse_gradient, group_above)
    
    # Count actual edges that will be plotted for each branch order
    actual_edge_counts = {}
    for bo in branch_orders:
        actual_edge_counts[bo] = 0
        for (u, v, key), edge_bo in edge_branch_orders.items():
            if edge_bo == bo:
                path = edge_paths[(u, v, key)]
                if len(path) > 1:
                    actual_edge_counts[bo] += 1
    
    # Create legend groups if grouping is enabled
    legend_orders, legend_counts = group_branch_orders_for_legend(branch_orders, group_above, actual_edge_counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display background image
    ax.imshow(projection, cmap=background_cmap)
    
    # Draw edges by branch order
    legend_elements = []
    
    for bo in branch_orders:
        # Collect all paths for this branch order
        bo_paths = []
        
        for (u, v, key), edge_bo in edge_branch_orders.items():
            if edge_bo == bo:
                path = edge_paths[(u, v, key)]
                if len(path) > 1:
                    bo_paths.append(np.array(path))
        
        # Plot all paths for this branch order
        color = color_mapping[bo]
        plotted_count = 0
        for path in bo_paths:
            ax.plot(path[:, 2], path[:, 1], color=color, 
                   linewidth=edge_linewidth, alpha=alpha)
            plotted_count += 1
    
    # Create legend using grouped orders
    for legend_bo in legend_orders:
        if legend_bo in color_mapping:
            color = color_mapping[legend_bo]
            count = legend_counts[legend_bo]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, 
                                            label=f"{legend_bo} ({count} edges)", alpha=alpha))
    
    # Draw nodes
    pos = nx.get_node_attributes(G, 'pos')
    if pos:
        coords = np.array(list(pos.values()))
        ax.scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
    
    # Add legend
    if show_legend and legend_elements:
        ax.legend(handles=legend_elements, title='Branch Orders', 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Set title and styling
    ax.set_title("Network Geometry with Branch Order Colors", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Geometry visualization saved to: {save_path}")
    
    plt.show()
    
    return fig, ax, color_mapping


fig, ax, weight_range, cmap = visualize_geometry_with_edge_weights(image, G, min_weight=0.1, max_weight=5.5, use_inverse=True)

def visualize_geometry_with_edge_weights(image, G, figsize=(12, 10), color_palette='viridis', node_color='red', node_size=3, edge_linewidth=0.8, show_legend=True, background_cmap='gray', save_path=None, dpi=300, alpha=0.8, min_weight=None, max_weight=None, legend_bins=5, reverse_gradient=False, use_inverse=True):
    
    # Create Z-projection of the image
    projection = np.max(image, axis=0)
    
    # Extract weights from edges and optionally calculate inverse
    edge_weights = {}
    edge_paths = {}
    weights_list = []
    
    edges_with_weights = 0
    edges_with_voxels = 0
    edges_with_both = 0
    zero_weight_edges = 0
    
    for u, v, key, data in G.edges(keys=True, data=True):
        weight = data.get('weight', None)
        path = data.get('voxels', [])
        
        if weight is not None:
            edges_with_weights += 1
            
            # Handle inverse weight calculation
            if use_inverse:
                if weight == 0:
                    print(f"WARNING: Edge ({u}, {v}, {key}) has weight=0, cannot calculate 1/weight. Skipping this edge.")
                    zero_weight_edges += 1
                    processed_weight = None
                else:
                    processed_weight = 1.0 / weight
                    weights_list.append(processed_weight)
            else:
                processed_weight = weight
                weights_list.append(processed_weight)
        else:
            processed_weight = None
        
        if len(path) > 1:
            edges_with_voxels += 1
            
        if processed_weight is not None and len(path) > 1:
            edges_with_both += 1
        
        edge_weights[(u, v, key)] = processed_weight
        edge_paths[(u, v, key)] = path
    
    print(f"Debugging Information")
    print(f"Using inverse weights (1/weight): {use_inverse}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Edges with weights: {edges_with_weights}")
    if use_inverse and zero_weight_edges > 0:
        print(f"Edges with zero weights (skipped): {zero_weight_edges}")
    print(f"Edges with voxel paths: {edges_with_voxels}")
    print(f"Edges with both (visualizable): {edges_with_both}")
    
    if not weights_list:
        print("WARNING: No edges with weights found!")
        return None, None, None, None
    
    # Determine weight range
    data_min_weight = min(weights_list)
    data_max_weight = max(weights_list)
    
    # Use provided range or data range
    final_min_weight = min_weight if min_weight is not None else data_min_weight
    final_max_weight = max_weight if max_weight is not None else data_max_weight
    
    weight_type = "1/weight" if use_inverse else "weight"
    print(f"Data {weight_type} range: [{data_min_weight:.6f}, {data_max_weight:.6f}]")
    print(f"Color mapping range: [{final_min_weight:.6f}, {final_max_weight:.6f}]")
    
    # Create colormap and normalization
    cmap = plt.get_cmap(color_palette)
    if reverse_gradient:
        cmap = cmap.reversed()
    
    norm = Normalize(vmin=final_min_weight, vmax=final_max_weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display background image
    ax.imshow(projection, cmap=background_cmap)
    
    # Draw edges colored by weight
    plotted_edges = 0
    weight_counts = {}  # For tracking how many edges at each weight range
    
    for (u, v, key), weight in edge_weights.items():
        if weight is not None:
            path = edge_paths[(u, v, key)]
            if len(path) > 1:
                # Get color for this weight
                color = cmap(norm(weight))
                
                # Plot the path
                path_array = np.array(path)
                ax.plot(path_array[:, 2], path_array[:, 1], color=color, 
                       linewidth=edge_linewidth, alpha=alpha)
                plotted_edges += 1
    
    print(f"Plotted {plotted_edges} edges with weights and paths")
    
    # Draw nodes
    pos = nx.get_node_attributes(G, 'pos')
    if pos:
        coords = np.array(list(pos.values()))
        ax.scatter(coords[:, 2], coords[:, 1], c=node_color, s=node_size)
    
    # Add colorbar legend
    if show_legend and plotted_edges > 0:
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar_label = "1/Weight" if use_inverse else "Edge Weight"
        cbar.set_label(cbar_label, rotation=270, labelpad=20)
        
        # Add some statistics to the legend
        stats_text = f"Data range: [{data_min_weight:.6f}, {data_max_weight:.6f}]\n"
        stats_text += f"Type: {weight_type}\n"
        stats_text += f"Plotted edges: {plotted_edges}"
        ax.text(1.02, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title and styling
    title_suffix = " (1/Weight)" if use_inverse else " (Weight)"
    title = f"Network Geometry Colored by Edge{title_suffix}"
    if min_weight is not None or max_weight is not None:
        title += f"\n(Fixed range: [{final_min_weight:.6f}, {final_max_weight:.6f}])"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Weight-based visualization saved to: {save_path}")
    
    plt.show()
    
    return fig, ax, (final_min_weight, final_max_weight), cmap


# %%
#Code for network resistance

def calc_laplacian_from_conductance_matrix(C):
    if not np.allclose(C, C.T):
        raise ValueError("Conductance matrix must be symmetric")
    if not np.all(np.diagonal(C) == 0):
        raise ValueError("Conductance matrix diagonal must be zero")
    diag = np.sum(C, axis=1)
    L = np.diag(diag) - C
    return L

def calc_two_point_from_laplacian_matrix_nodeID(L, G, node_id1, node_id2):
    """ This function calculates the two point resistance between two nodes in a network given the Laplacian matrix.
    Args:
        L (np.array): Laplacian matrix of the network (hermitian)
        G (networkx.MultiGraph): The multigraph containing the nodes
        node_id1: ID of node 1 (as it appears in the multigraph)
        node_id2: ID of node 2 (as it appears in the multigraph)
    """
    
    # Get the node list from the multigraph to create the mapping
    node_list = list(G.nodes())
    
    # Create mapping from node ID to matrix index
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    
    # Convert node IDs to matrix indices
    try:
        node_idx1 = node_to_idx[node_id1]
        node_idx2 = node_to_idx[node_id2]
    except KeyError as e:
        raise ValueError(f"Node {e} not found in the multigraph")
    
    # calculate the eigenvalues and orthonormal eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # calculate the effective resistance between the two nodes
    R = 0
    for II in range(1, len(eigvals)):
        if eigvals[II] > 1e-10:
            R += (1/eigvals[II]) * (eigvecs[node_idx1, II] - eigvecs[node_idx2, II])**2
    
    return R



