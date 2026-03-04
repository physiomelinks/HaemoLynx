"""Internal helpers for graph operations."""
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import networkx as nx

def add_edge_safe(G, u, v, **attr):
    return G.add_edge(u, v, **attr)

def has_edge_safe(G: Union[nx.Graph, nx.MultiGraph], u: int, v: int) -> bool:
    """Check if edge exists between u and v."""
    return G.has_edge(u, v)

def remove_edge_safe(G, u, v):
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        # Remove all edges between u and v
        if G.has_edge(u, v):
            keys_to_remove = list(G[u][v].keys())
            for key in keys_to_remove:
                G.remove_edge(u, v, key)
    else:
        if G.has_edge(u, v):
            G.remove_edge(u, v)


def get_all_edge_data(G, u, v):
    """
    Get all edge data between two nodes (for multigraphs, returns list of all parallel edges).
    """
    import networkx as nx
    
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if G.has_edge(u, v):
            return list(G[u][v].values())
        return []
    else:
        edge_data = G.get_edge_data(u, v)
        return [edge_data] if edge_data is not None else []


def create_merged_edge_attributes(edge1_data, edge2_data, node_pos):
    """
    Create merged edge attributes from two edges and the removed node position.
    """
    
    # Get original voxel paths
    voxels1 = edge1_data.get('voxels', [])
    voxels2 = edge2_data.get('voxels', [])
    
    # CRITICAL FIX: Properly orient and merge voxel paths
    merged_voxels = []
    
    if voxels1 and voxels2 and node_pos is not None:
        # Convert to numpy arrays for easier manipulation
        voxels1_arr = [np.array(v) for v in voxels1]
        voxels2_arr = [np.array(v) for v in voxels2]
        node_pos_arr = np.array(node_pos)
        
        # Determine which end of each voxel path connects to the removed node
        # Check distances to find the correct orientation
        
        # For voxels1: which end is closer to the removed node?
        if len(voxels1_arr) > 0:
            dist_to_start1 = np.linalg.norm(voxels1_arr[0] - node_pos_arr)
            dist_to_end1 = np.linalg.norm(voxels1_arr[-1] - node_pos_arr)
            
            if dist_to_start1 < dist_to_end1:
                # Node connects to start of voxels1, so reverse it
                voxels1_oriented = voxels1[::-1]
            else:
                # Node connects to end of voxels1, keep original order
                voxels1_oriented = voxels1
        else:
            voxels1_oriented = []
        
        # For voxels2: which end is closer to the removed node?
        if len(voxels2_arr) > 0:
            dist_to_start2 = np.linalg.norm(voxels2_arr[0] - node_pos_arr)
            dist_to_end2 = np.linalg.norm(voxels2_arr[-1] - node_pos_arr)
            
            if dist_to_start2 < dist_to_end2:
                # Node connects to start of voxels2, keep original order
                voxels2_oriented = voxels2
            else:
                # Node connects to end of voxels2, so reverse it
                voxels2_oriented = voxels2[::-1]
        else:
            voxels2_oriented = []
        
        # Build continuous path: voxels1_oriented + node_position + voxels2_oriented
        merged_voxels = []
        
        # Add first path (leading TO the removed node)
        if voxels1_oriented:
            merged_voxels.extend(voxels1_oriented)
        
        # Add the removed node position (ensuring no duplicates)
        node_voxel = tuple(int(round(x)) for x in node_pos)
        if not merged_voxels or merged_voxels[-1] != node_voxel:
            merged_voxels.append(node_voxel)
        
        # Add second path (leading FROM the removed node)
        if voxels2_oriented:
            # Skip first voxel of second path if it's the same as removed node
            start_idx = 1 if (voxels2_oriented and 
                            tuple(int(round(x)) for x in voxels2_oriented[0]) == node_voxel) else 0
            merged_voxels.extend(voxels2_oriented[start_idx:])
    
    else:
        # Fallback: simple concatenation if we can't do proper orientation
        if voxels1:
            merged_voxels.extend(voxels1)
        
        if node_pos is not None:
            node_voxel = tuple(int(round(x)) for x in node_pos)
            if node_voxel not in merged_voxels:
                merged_voxels.append(node_voxel)
        
        if voxels2:
            merged_voxels.extend(voxels2)
    
    # Validate the merged path for continuity
    if len(merged_voxels) > 1:
        max_gap = validate_voxel_path_continuity(merged_voxels)
        if max_gap > 5.0:  # Large gap indicates problematic merge
            print(f"WARNING: Large gap ({max_gap:.2f}) in merged voxel path - may be discontinuous")
    
    # Calculate proper length from the merged voxel path
    length_from_voxels = calculate_voxel_path_length(merged_voxels) if merged_voxels else 0
    length_additive = edge1_data.get('length', 0) + edge2_data.get('length', 0)
    
    # Use voxel path length if available and reasonable, otherwise use additive
    if length_from_voxels > 0 and abs(length_from_voxels - length_additive) < length_additive * 0.5:
        final_length = length_from_voxels
    else:
        final_length = length_additive
    
    # Create merged attributes
    merged_attrs = {
        'weight': max(edge1_data.get('weight', 0) + edge2_data.get('weight', 0), 1e-6),
        'length': final_length,
        'voxels': merged_voxels,  # Properly oriented and continuous path
        'merged': True,
        'simple_merge': True,
        'removed_node_pos': node_pos,
        'voxel_path_length': length_from_voxels,
        'additive_length': length_additive
    }
    
    # Preserve other attributes from first edge
    for key, value in edge1_data.items():
        if key not in merged_attrs:
            merged_attrs[key] = value
    
    return merged_attrs



def get_line_points_3d(p1, p2):
    """
    Get 3D line points between two positions using Bresenham-like algorithm.
    """
    import numpy as np
    
    # Simple linear interpolation for 3D line
    distance = np.linalg.norm(p2 - p1)
    num_points = max(int(distance) + 1, 2)
    
    line_points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        point = p1 + t * (p2 - p1)
        line_points.append(tuple(np.round(point).astype(int)))
    
    return line_points


def calculate_path_length(voxels):
    """Calculate length as sum of distances between consecutive voxels."""
    import numpy as np
    
    if len(voxels) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(voxels) - 1):
        p1 = np.array(voxels[i])
        p2 = np.array(voxels[i + 1])
        total += np.linalg.norm(p2 - p1)
    
    return total


def calculate_edge_length(node1: int, node2: int, edge_data: dict, voxel_size: Tuple[float, float, float] = (1, 1, 1)) -> float:
    """
    Calculate the length of an edge between two nodes.
    -------
    float
        Edge length
    """
    # If length is pre-calculated in edge data
    if 'length' in edge_data:
        return edge_data['length']
    
    # If we have coordinate information, calculate Euclidean distance
    if 'pos' in edge_data or ('x' in edge_data and 'y' in edge_data):
        if 'pos' in edge_data:
            pos1, pos2 = edge_data['pos']
        else:
            pos1 = (edge_data.get('x1', 0), edge_data.get('y1', 0), edge_data.get('z1', 0))
            pos2 = (edge_data.get('x2', 0), edge_data.get('y2', 0), edge_data.get('z2', 0))
        
        # Calculate distance accounting for voxel size
        diff = np.array(pos2) - np.array(pos1)
        scaled_diff = diff * np.array(voxel_size)
        return np.linalg.norm(scaled_diff)
    
    # If we have weight, use that
    if 'weight' in edge_data:
        return edge_data['weight']
    # Fallback
    return 1.0
    


# For merge_edges_with_topology_improvement
def is_path_curved(voxels: List, ratio_threshold: float = 1.15) -> bool:
    """True if path length / straight-line distance > threshold."""
    if len(voxels) < 3:
        return False
    arr = np.array(voxels, dtype=float)
    path_len = calculate_path_length(voxels)
    straight = np.linalg.norm(arr[-1] - arr[0])
    if straight < 1e-10:
        return True
    return path_len / straight > ratio_threshold


def merge_curved_edges(voxels1, voxels2, connection_pos, debug=False):
    """
    Merge two edge paths at a connection point, preserving topology.
    """
    
    connection_pos = np.array(connection_pos)
    
    # Orient voxels1 to end at connection_pos
    oriented_voxels1 = orient_path_to_endpoint(voxels1, connection_pos)
    
    # Orient voxels2 to start from connection_pos  
    oriented_voxels2 = orient_path_from_startpoint(voxels2, connection_pos)
    
    # Merge the paths
    merged = []
    
    # Add first path
    if oriented_voxels1:
        merged.extend(oriented_voxels1)
    
    # Add connection point if not already present
    connection_voxel = tuple(connection_pos.astype(int))
    if not merged or tuple(merged[-1]) != connection_voxel:
        merged.append(connection_voxel)
    
    # Add second path (skip duplicate connection point)
    if oriented_voxels2:
        start_idx = 1 if (len(oriented_voxels2) > 0 and 
                         tuple(oriented_voxels2[0]) == connection_voxel) else 0
        merged.extend(oriented_voxels2[start_idx:])
    
    return merged


def improve_straight_edge_with_skeleton(
    pos_a: np.ndarray, pos_b: np.ndarray, skeleton_data: np.ndarray, debug: bool = False
) -> List:
    """Find skeleton-based path between two points (for straight edges)."""
    from skimage.graph import route_through_array
    pa = np.round(pos_a).astype(int)
    pb = np.round(pos_b).astype(int)
    cost = 1 + (1 - skeleton_data.astype(float)) * 100
    try:
        path, _ = route_through_array(cost, tuple(pa), tuple(pb), fully_connected=True)
        return [tuple(p) for p in path]
    except Exception:
        return []


def improve_straight_edge_with_skeleton(start_pos, end_pos, skeleton_data, debug=False):
    """
    Improve a straight edge by tracing through skeleton topology.
    Returns improved voxel path or None if improvement not possible.
    """
    if skeleton_data is None:
        return None
    
    traced_path = trace_skeleton_path(skeleton_data, start_pos, end_pos, debug)
    
    if traced_path and len(traced_path) >= 2:
        # Verify the traced path is actually better (longer/more curved)
        if is_path_curved(traced_path) or len(traced_path) > 3:
            return traced_path
    
    return None

def trace_skeleton_path(skeleton_data, start_pos, end_pos, debug=False):
    """
    Trace path through skeleton data from start_pos to end_pos using A* pathfinding.
    
    Args:
        skeleton_data: 3D binary array where 1s represent skeleton voxels,
                      OR dict with 'skeleton' key containing the binary array,
                      OR list of skeleton voxel coordinates
        start_pos: Starting position (3D coordinates)
        end_pos: Ending position (3D coordinates)
        debug: Whether to print debug info
    
    Returns:
        List of voxel coordinates [(x,y,z), ...] or None if no path found
    """

    
    if debug:
        print(f"       Tracing skeleton from {start_pos} to {end_pos}")
    
    # Parse skeleton data into binary array
    skeleton_array = parse_skeleton_data(skeleton_data)
    if skeleton_array is None:
        if debug:
            print(f"       Could not parse skeleton data")
        return None
    
    # Find nearest skeleton voxels to start and end positions
    start_skeleton = find_nearest_skeleton_voxel(skeleton_array, start_pos)
    end_skeleton = find_nearest_skeleton_voxel(skeleton_array, end_pos)
    
    if start_skeleton is None or end_skeleton is None:
        if debug:
            print(f"       Could not find skeleton voxels near start/end positions")
        return None
    
    if debug:
        start_dist = np.linalg.norm(np.array(start_pos) - np.array(start_skeleton))
        end_dist = np.linalg.norm(np.array(end_pos) - np.array(end_skeleton))
        print(f"       Start skeleton voxel: {start_skeleton} (dist: {start_dist:.1f})")
        print(f"       End skeleton voxel: {end_skeleton} (dist: {end_dist:.1f})")
    
    # Use A* to find path through skeleton
    path = astar_skeleton_path(skeleton_array, start_skeleton, end_skeleton, debug)
    
    if path:
        if debug:
            print(f"       Found skeleton path with {len(path)} voxels")
        return path
    else:
        if debug:
            print(f"       No skeleton path found")
        return None

def are_paths_similar(voxels1, voxels2, tolerance=3.0):
    """Check if two paths connect similar endpoints."""
    import numpy as np
    
    if len(voxels1) < 2 or len(voxels2) < 2:
        return False
    
    start1, end1 = np.array(voxels1[0]), np.array(voxels1[-1])
    start2, end2 = np.array(voxels2[0]), np.array(voxels2[-1])
    
    # Check both orientations
    dist_same = np.linalg.norm(start1 - start2) + np.linalg.norm(end1 - end2)
    dist_flipped = np.linalg.norm(start1 - end2) + np.linalg.norm(end1 - start2)
    
    return min(dist_same, dist_flipped) <= tolerance * 2
    
def should_add_merged_edge(G, n1, n2, new_voxels, new_attrs, debug=False):
    """
    Check if we should add this merged edge, avoiding duplicates.
    """
    import numpy as np
    
    if not G.has_edge(n1, n2):
        return True, None
    
    new_length = new_attrs.get('length', 0)
    new_is_curved = is_path_curved(new_voxels)
    
    # Check existing edges for similar paths
    for edge_key, edge_data in G[n1][n2].items():
        existing_voxels = edge_data.get('voxels', [])
        existing_length = edge_data.get('length', 0)
        existing_is_curved = is_path_curved(existing_voxels)
        
        # Check if paths are similar
        if are_paths_similar(new_voxels, existing_voxels):
            # Prefer curved over straight
            if new_is_curved and not existing_is_curved:
                if debug:
                    print(f"     Replacing straight with curved path")
                return True, edge_key
            elif not new_is_curved and existing_is_curved:
                if debug:
                    print(f"     Keeping existing curved over new straight")
                return False, None
            else:
                # Same type - prefer shorter
                if new_length < existing_length * 0.95:
                    if debug:
                        print(f"     Replacing with shorter path ({new_length:.1f} vs {existing_length:.1f})")
                    return True, edge_key
                else:
                    if debug:
                        print(f"     Keeping existing shorter path ({existing_length:.1f} vs {new_length:.1f})")
                    return False, None
    
    return True, None
