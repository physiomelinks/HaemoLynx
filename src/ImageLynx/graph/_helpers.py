"""Internal helpers for graph operations."""
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import networkx as nx
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from ..coords import physical_xyz_to_index_zyx, index_zyx_to_physical_xyz

def add_edge_safe(G, u, v, **attr):
    return G.add_edge(u, v, **attr)


def _as_xyz_array(point: Any) -> np.ndarray:
    """Normalize point-like input to a 3-vector float array."""
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"Expected point with >=3 values, got shape {arr.shape}")
    return arr[:3]


def physical_point_to_voxel_index(
    point: Any,
    voxel_size: Tuple[float, float, float],
    *,
    clip_shape: Tuple[int, int, int] = None,
) -> Tuple[int, int, int]:
    """Convert physical xyz point to nearest voxel index using voxel size."""
    vs = np.asarray(voxel_size, dtype=float)
    idx = np.round(_as_xyz_array(point) / vs).astype(int)
    if clip_shape is not None:
        shape = np.asarray(clip_shape, dtype=int)
        idx = np.clip(idx, 0, shape - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


def physical_points_to_voxel_indices(
    points: Any,
    voxel_size: Tuple[float, float, float],
    *,
    clip_shape: Tuple[int, int, int] = None,
) -> np.ndarray:
    """Convert N x 3 physical xyz points to nearest voxel indices."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected points with shape (N, 3)")
    vs = np.asarray(voxel_size, dtype=float)
    idx = np.round(arr[:, :3] / vs).astype(int)
    if clip_shape is not None:
        shape = np.asarray(clip_shape, dtype=int)
        idx = np.clip(idx, 0, shape - 1)
    return idx


def _points_allclose(a: Any, b: Any, atol: float = 1e-6) -> bool:
    """Check point equality in continuous coordinate space."""
    return bool(np.allclose(_as_xyz_array(a), _as_xyz_array(b), atol=atol))

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
        node_point = tuple(_as_xyz_array(node_pos).tolist())
        if not merged_voxels or not _points_allclose(merged_voxels[-1], node_point):
            merged_voxels.append(node_point)
        
        # Add second path (leading FROM the removed node)
        if voxels2_oriented:
            # Skip first voxel of second path if it's the same as removed node
            start_idx = 1 if (voxels2_oriented and 
                            _points_allclose(voxels2_oriented[0], node_point)) else 0
            merged_voxels.extend(voxels2_oriented[start_idx:])
    
    else:
        # Fallback: simple concatenation if we can't do proper orientation
        if voxels1:
            merged_voxels.extend(voxels1)
        
        if node_pos is not None:
            node_point = tuple(_as_xyz_array(node_pos).tolist())
            if not any(_points_allclose(v, node_point) for v in merged_voxels):
                merged_voxels.append(node_point)
        
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

def calculate_voxel_path_length(voxels):
    """
    Calculate the actual length along a voxel path.
    """
    if not voxels or len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

def validate_voxel_path_continuity(voxels):
    """
    Check if a voxel path is continuous and return the maximum gap.
    """
    if len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    return float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

def merge_edge_voxels_at_node(voxels1: List, voxels2: List, node_pos: Any) -> List:
    """Concatenate two edge voxel paths at the removed node with orientation."""
    part1 = orient_voxel_path_to_node(voxels1, node_pos, node_should_be_start=False)
    part2 = orient_voxel_path_to_node(voxels2, node_pos, node_should_be_start=True)
    merged = list(part1)
    if node_pos is not None:
        node_point = tuple(_as_xyz_array(node_pos).tolist())
        if not merged or not _points_allclose(merged[-1], node_point):
            merged.append(node_point)
    if part2:
        start_idx = 0
        if node_pos is not None and _points_allclose(part2[0], node_pos):
            start_idx = 1
        merged.extend(part2[start_idx:])
    return merged

def _voxel_key(point: Any) -> Tuple[int, ...]:
    arr = np.asarray(point, dtype=float)
    return tuple(np.round(arr).astype(int))

def orient_voxel_path_to_node(
    voxels: List, node_pos: Any, *, node_should_be_start: bool
) -> List:
    """Orient a voxel path so the removed node is at desired endpoint."""
    if not voxels:
        return []
    oriented = list(voxels)
    if node_pos is None:
        return oriented
    node_key = _voxel_key(node_pos)
    start_key = _voxel_key(oriented[0])
    end_key = _voxel_key(oriented[-1])
    if node_should_be_start:
        if start_key != node_key and end_key == node_key:
            oriented.reverse()
    else:
        if end_key != node_key and start_key == node_key:
            oriented.reverse()
    return oriented

def get_line_points_3d(p1, p2):
    """
    Get 3D line points between two positions using Bresenham-like algorithm.
    """
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
    if len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def snap_edge_endpoints_to_node_positions(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    update_length: bool = True,
    update_weight_if_matches_length: bool = True,
    max_snap_distance: Optional[float] = None,
    reference_skeleton_data: Optional[np.ndarray] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    prefer_skeleton_reference: bool = False,
    skeleton_search_radius_voxels: int = 10,
    atol: float = 1e-6,
) -> Dict[str, int]:
    """Align edge voxel-path endpoints to incident node positions.

    Topology edits can leave edge voxel paths slightly detached from node ``pos``.
    This routine orients each path to best match ``(u, v)``, snaps first/last
    points to node positions, and optionally updates edge length.
    """
    stats = {
        "edges_seen": 0,
        "edges_snapped": 0,
        "edges_reoriented": 0,
        "edges_skipped_distance": 0,
        "endpoints_using_skeleton_reference": 0,
    }
    skeleton_array = (
        parse_skeleton_data(reference_skeleton_data)
        if (prefer_skeleton_reference and reference_skeleton_data is not None)
        else None
    )
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_multi else (
        (u, v, 0, data) for u, v, data in G.edges(data=True)
    )

    for u, v, _, data in edge_iter:
        stats["edges_seen"] += 1
        voxels = data.get("voxels")
        if not voxels or len(voxels) < 2:
            continue
        if "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
            continue

        pts = np.asarray(voxels, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue

        u_pos = np.asarray(G.nodes[u]["pos"], dtype=float)
        v_pos = np.asarray(G.nodes[v]["pos"], dtype=float)
        if u_pos.size < 3 or v_pos.size < 3:
            continue
        u3 = u_pos[:3]
        v3 = v_pos[:3]

        u_target = u3
        v_target = v3
        if skeleton_array is not None:
            u_idx = physical_xyz_to_index_zyx(u3, voxel_size)
            v_idx = physical_xyz_to_index_zyx(v3, voxel_size)
            u_nearest = find_nearest_skeleton_voxel(
                skeleton_array,
                u_idx,
                max_search_radius=int(skeleton_search_radius_voxels),
            )
            v_nearest = find_nearest_skeleton_voxel(
                skeleton_array,
                v_idx,
                max_search_radius=int(skeleton_search_radius_voxels),
            )
            if u_nearest is not None:
                u_target = np.asarray(index_zyx_to_physical_xyz(u_nearest, voxel_size), dtype=float)[:3]
                stats["endpoints_using_skeleton_reference"] += 1
            if v_nearest is not None:
                v_target = np.asarray(index_zyx_to_physical_xyz(v_nearest, voxel_size), dtype=float)[:3]
                stats["endpoints_using_skeleton_reference"] += 1

        direct_start = float(np.linalg.norm(pts[0] - u_target))
        direct_end = float(np.linalg.norm(pts[-1] - v_target))
        flipped_start = float(np.linalg.norm(pts[0] - v_target))
        flipped_end = float(np.linalg.norm(pts[-1] - u_target))
        direct_cost = direct_start + direct_end
        flipped_cost = flipped_start + flipped_end
        use_flipped = flipped_cost < direct_cost
        if use_flipped:
            cand_start_dist = flipped_start
            cand_end_dist = flipped_end
        else:
            cand_start_dist = direct_start
            cand_end_dist = direct_end
        if max_snap_distance is not None:
            if max(cand_start_dist, cand_end_dist) > float(max_snap_distance):
                stats["edges_skipped_distance"] += 1
                continue
        if use_flipped:
            pts = pts[::-1].copy()
            stats["edges_reoriented"] += 1

        before_start = pts[0].copy()
        before_end = pts[-1].copy()
        pts[0] = u_target
        pts[-1] = v_target

        snapped = not (
            np.allclose(before_start, u_target, atol=atol)
            and np.allclose(before_end, v_target, atol=atol)
        )
        if not snapped:
            continue

        data["voxels"] = pts.tolist()
        if update_length:
            new_length = calculate_path_length(data["voxels"])
            old_length = data.get("length", None)
            data["length"] = new_length
            if (
                update_weight_if_matches_length
                and old_length is not None
                and np.isfinite(old_length)
                and np.isfinite(data.get("weight", np.nan))
                and np.isclose(float(data["weight"]), float(old_length), atol=max(atol, 1e-9))
            ):
                data["weight"] = new_length
        stats["edges_snapped"] += 1

    return stats


def diagnose_edge_endpoint_node_alignment(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    atol: float = 1e-6,
    sample_limit: int = 20,
) -> Dict[str, Any]:
    """Report edges whose voxel endpoints do not align with incident nodes."""
    stats: Dict[str, Any] = {
        "edges_seen": 0,
        "edges_with_voxels": 0,
        "edges_missing_node_pos": 0,
        "misaligned_edges": 0,
        "max_misalignment": 0.0,
        "sample_misaligned_edges": [],
    }
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_multi else (
        (u, v, 0, data) for u, v, data in G.edges(data=True)
    )

    for u, v, k, data in edge_iter:
        stats["edges_seen"] += 1
        voxels = data.get("voxels")
        if not voxels or len(voxels) < 2:
            continue
        stats["edges_with_voxels"] += 1
        if "pos" not in G.nodes[u] or "pos" not in G.nodes[v]:
            stats["edges_missing_node_pos"] += 1
            continue

        pts = np.asarray(voxels, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        u3 = np.asarray(G.nodes[u]["pos"], dtype=float)[:3]
        v3 = np.asarray(G.nodes[v]["pos"], dtype=float)[:3]
        if u3.size < 3 or v3.size < 3:
            stats["edges_missing_node_pos"] += 1
            continue

        direct = float(np.linalg.norm(pts[0] - u3) + np.linalg.norm(pts[-1] - v3))
        flipped = float(np.linalg.norm(pts[0] - v3) + np.linalg.norm(pts[-1] - u3))
        mismatch = min(direct, flipped)
        if mismatch > float(atol):
            stats["misaligned_edges"] += 1
            if mismatch > float(stats["max_misalignment"]):
                stats["max_misalignment"] = mismatch
            if len(stats["sample_misaligned_edges"]) < int(sample_limit):
                stats["sample_misaligned_edges"].append((int(u), int(v), int(k), mismatch))
    return stats


def diagnose_edge_endpoint_skeleton_alignment(
    G: Union[nx.Graph, nx.MultiGraph],
    skeleton_data: np.ndarray,
    *,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_search_radius_voxels: int = 2,
    sample_limit: int = 20,
) -> Dict[str, Any]:
    """Report how well edge endpoints lie on the provided skeleton."""
    skeleton_array = parse_skeleton_data(skeleton_data)
    if skeleton_array is None:
        raise ValueError("skeleton_data could not be parsed into a 3D array")

    stats: Dict[str, Any] = {
        "edges_seen": 0,
        "endpoints_seen": 0,
        "off_skeleton_endpoints": 0,
        "max_distance_to_skeleton": 0.0,
        "sample_off_skeleton_endpoints": [],
    }
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_multi else (
        (u, v, 0, data) for u, v, data in G.edges(data=True)
    )

    for u, v, k, data in edge_iter:
        stats["edges_seen"] += 1
        voxels = data.get("voxels")
        if not voxels or len(voxels) < 2:
            continue
        pts = np.asarray(voxels, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue

        for endpoint in (pts[0], pts[-1]):
            stats["endpoints_seen"] += 1
            endpoint_idx = physical_xyz_to_index_zyx(endpoint, voxel_size)
            nearest = find_nearest_skeleton_voxel(
                skeleton_array,
                endpoint_idx,
                max_search_radius=int(max_search_radius_voxels),
            )
            if nearest is None:
                stats["off_skeleton_endpoints"] += 1
                if len(stats["sample_off_skeleton_endpoints"]) < int(sample_limit):
                    stats["sample_off_skeleton_endpoints"].append((int(u), int(v), int(k), None))
                continue
            nearest_phys = np.asarray(index_zyx_to_physical_xyz(nearest, voxel_size), dtype=float)
            dist = float(np.linalg.norm(np.asarray(endpoint, dtype=float) - nearest_phys))
            if dist > float(stats["max_distance_to_skeleton"]):
                stats["max_distance_to_skeleton"] = dist
            if dist > 1e-6:
                stats["off_skeleton_endpoints"] += 1
                if len(stats["sample_off_skeleton_endpoints"]) < int(sample_limit):
                    stats["sample_off_skeleton_endpoints"].append((int(u), int(v), int(k), dist))
    return stats


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
    
    # If we have coordinate information, calculate Euclidean distance.
    # In this codebase, edge/node coordinates are stored in physical units.
    if 'pos' in edge_data or ('x' in edge_data and 'y' in edge_data):
        if 'pos' in edge_data:
            pos1, pos2 = edge_data['pos']
        else:
            pos1 = (edge_data.get('x1', 0), edge_data.get('y1', 0), edge_data.get('z1', 0))
            pos2 = (edge_data.get('x2', 0), edge_data.get('y2', 0), edge_data.get('z2', 0))
        # Positions here are physical coordinates (x, y, z), so do not rescale.
        diff = np.array(pos2) - np.array(pos1)
        return float(np.linalg.norm(diff))
    
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
    connection_point = tuple(connection_pos.astype(float).tolist())
    if not merged or not _points_allclose(merged[-1], connection_point):
        merged.append(connection_point)
    
    # Add second path (skip duplicate connection point)
    if oriented_voxels2:
        start_idx = 1 if (
            len(oriented_voxels2) > 0
            and _points_allclose(oriented_voxels2[0], connection_point)
        ) else 0
        merged.extend(oriented_voxels2[start_idx:])
    
    return merged

def orient_path_to_endpoint(voxels, target_pos):
    """Orient path so it ends at target_pos."""
    if not voxels:
        return []
    
    target = np.array(target_pos)
    start_dist = np.linalg.norm(np.array(voxels[0]) - target)
    end_dist = np.linalg.norm(np.array(voxels[-1]) - target)
    
    if start_dist < end_dist:
        return voxels[::-1]  # Reverse to end at target
    else:
        return list(voxels)  # Already ends at target


def orient_path_from_startpoint(voxels, target_pos):
    """Orient path so it starts from target_pos."""
    if not voxels:
        return []
    
    target = np.array(target_pos)
    start_dist = np.linalg.norm(np.array(voxels[0]) - target)
    end_dist = np.linalg.norm(np.array(voxels[-1]) - target)
    
    if start_dist < end_dist:
        return list(voxels)  # Already starts from target
    else:
        return voxels[::-1]  # Reverse to start from target

def improve_straight_edge_with_skeleton(start_pos, end_pos, skeleton_data, debug=False, voxel_size=(1.0, 1.0, 1.0)):
    """
    Improve a straight edge by tracing through skeleton topology.
    Returns improved voxel path (physical coords) or None if not possible.
    """
    if skeleton_data is None:
        return None
    
    traced_path = trace_skeleton_path(skeleton_data, start_pos, end_pos, debug, voxel_size=voxel_size)
    
    if traced_path and len(traced_path) >= 2:
        if is_path_curved(traced_path) or len(traced_path) > 3:
            return traced_path
    
    return None

def trace_skeleton_path(skeleton_data, start_pos, end_pos, debug=False, voxel_size=(1.0, 1.0, 1.0)):
    """
    Trace path through skeleton data from start_pos to end_pos using A* pathfinding.
    
    Positions are in physical units; *voxel_size* converts them to array
    indices for look-ups.  The returned path is converted back to physical
    coordinates.
    """
    vs = np.asarray(voxel_size, dtype=float)
    start_vox = physical_xyz_to_index_zyx(start_pos, vs)
    end_vox = physical_xyz_to_index_zyx(end_pos, vs)
    
    if debug:
        print(f"       Tracing skeleton from {start_pos} (vox {start_vox}) to {end_pos} (vox {end_vox})")
    
    skeleton_array = parse_skeleton_data(skeleton_data)
    if skeleton_array is None:
        if debug:
            print(f"       Could not parse skeleton data")
        return None
    
    start_skeleton = find_nearest_skeleton_voxel(skeleton_array, start_vox)
    end_skeleton = find_nearest_skeleton_voxel(skeleton_array, end_vox)
    
    if start_skeleton is None or end_skeleton is None:
        if debug:
            print(f"       Could not find skeleton voxels near start/end positions")
        return None
    
    if debug:
        start_dist = np.linalg.norm(np.array(start_pos) - np.array(start_skeleton))
        end_dist = np.linalg.norm(np.array(end_pos) - np.array(end_skeleton))
        print(f"       Start skeleton voxel: {start_skeleton} (dist: {start_dist:.1f})")
        print(f"       End skeleton voxel: {end_skeleton} (dist: {end_dist:.1f})")
    
    path = astar_skeleton_path(skeleton_array, start_skeleton, end_skeleton, debug)
    
    if path:
        if debug:
            print(f"       Found skeleton path with {len(path)} voxels")
        phys_path = [index_zyx_to_physical_xyz(p, vs).tolist() for p in path]
        return phys_path
    else:
        if debug:
            print(f"       No skeleton path found")
        return None

def parse_skeleton_data(skeleton_data):
    """
    Parse skeleton data into a 3D binary numpy array.
    Handles multiple input formats.
    """
    if skeleton_data is None:
        return None
    
    # Case 1: Already a numpy array
    if isinstance(skeleton_data, np.ndarray):
        if skeleton_data.ndim == 3:
            return skeleton_data if skeleton_data.dtype == bool else skeleton_data.astype(bool)
        else:
            return None
    
    # Case 2: Dictionary with skeleton key
    elif isinstance(skeleton_data, dict):
        if 'skeleton' in skeleton_data:
            skel = skeleton_data['skeleton']
            if isinstance(skel, np.ndarray) and skel.ndim == 3:
                return skel.astype(bool)
        # Try other common keys
        for key in ['binary', 'mask', 'data', 'array']:
            if key in skeleton_data:
                skel = skeleton_data[key]
                if isinstance(skel, np.ndarray) and skel.ndim == 3:
                    return skel.astype(bool)
        return None
    
    # Case 3: List of coordinates - convert to binary array
    elif isinstance(skeleton_data, (list, tuple)):
        if len(skeleton_data) > 0:
            coords = np.array(skeleton_data)
            if coords.ndim == 2 and coords.shape[1] == 3:
                # Create binary array from coordinates
                max_coords = coords.max(axis=0) + 1
                binary_array = np.zeros(max_coords, dtype=bool)
                for coord in coords:
                    binary_array[tuple(coord)] = True
                return binary_array
        return None
    
    else:
        return None


def find_nearest_skeleton_voxel(skeleton_array, target_pos, max_search_radius=10):
    """
    Find the nearest skeleton voxel to target_pos within search radius.
    """
    target = np.array(target_pos, dtype=int)
    shape = skeleton_array.shape
    
    if (target >= 0).all() and (target < shape).all():
        if skeleton_array[tuple(target)]:
            return tuple(target)
    
    for radius in range(1, max_search_radius + 1):
        lo = np.maximum(target - radius, 0)
        hi = np.minimum(target + radius + 1, shape)
        
        sub = skeleton_array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not np.any(sub):
            continue
        
        local_hits = np.argwhere(sub) + lo
        dists = np.linalg.norm(local_hits - target, axis=1)
        within = dists <= radius
        if np.any(within):
            best = int(np.argmin(np.where(within, dists, np.inf)))
            return tuple(local_hits[best])
    
    return None


def astar_skeleton_path(skeleton_array, start, end, debug=False):
    """
    A* pathfinding through skeleton voxels only.
    """
    import heapq
    from collections import defaultdict
    
    start = tuple(start)
    end = tuple(end)
    
    if start == end:
        return [start]
    
    ex, ey, ez = end
    sx, sy, sz = skeleton_array.shape

    open_set = [(0, 0, start)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    
    closed_set = set()
    
    _OFFSETS_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    
    iterations = 0
    max_iterations = 50000
    
    while open_set and iterations < max_iterations:
        iterations += 1
        
        _, current_g, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        cx, cy, cz = current
        cur_g = g_score[current]
        
        for dx, dy, dz in _OFFSETS_26:
            nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
            
            if not (0 <= nx_ < sx and 0 <= ny_ < sy and 0 <= nz_ < sz):
                continue
            if not skeleton_array[nx_, ny_, nz_]:
                continue
            
            neighbor = (nx_, ny_, nz_)
            if neighbor in closed_set:
                continue
            
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
            tentative_g = cur_g + distance
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                hdx = nx_ - ex
                hdy = ny_ - ey
                hdz = nz_ - ez
                f = tentative_g + (hdx*hdx + hdy*hdy + hdz*hdz) ** 0.5
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    
    return None


def _chaikin_once(points: np.ndarray) -> np.ndarray:
    """Apply one Chaikin subdivision pass while preserving endpoints."""
    if points.shape[0] <= 2:
        return points.copy()
    out = [points[0]]
    for i in range(points.shape[0] - 1):
        p0 = points[i]
        p1 = points[i + 1]
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        out.extend([q, r])
    out.append(points[-1])
    return np.asarray(out, dtype=float)


def _resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample polyline to a fixed number of points by arc length."""
    if points.shape[0] <= 1 or n_points <= 1:
        return points[:1].copy()
    if points.shape[0] == n_points:
        return points.copy()

    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(deltas)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        out = np.repeat(points[:1], n_points, axis=0)
        out[0] = points[0]
        out[-1] = points[-1]
        return out

    targets = np.linspace(0.0, total, n_points)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    z = np.interp(targets, cumulative, points[:, 2])
    out = np.column_stack((x, y, z))
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def chaikin_smooth_polyline(points: Any, iterations: int = 2) -> np.ndarray:
    """Smooth a polyline in continuous space via Chaikin corner cutting."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    iterations = max(int(iterations), 0)
    out = arr.copy()
    for _ in range(iterations):
        out = _chaikin_once(out)
    if out.shape[0] != arr.shape[0]:
        out = _resample_polyline(out, arr.shape[0])
    # Keep original endpoints exactly.
    out[0] = arr[0]
    out[-1] = arr[-1]
    return out


def bspline_smooth_polyline(points: Any, smoothness: float = 0.75) -> np.ndarray:
    """
    Smooth a polyline in continuous space using a parametric B-spline.

    Returns the same number of points and preserves endpoints exactly.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return arr

    # Remove consecutive duplicates to prevent spline fitting failures.
    dedup = [arr[0]]
    for i in range(1, arr.shape[0]):
        if not np.allclose(arr[i], arr[i - 1], atol=1e-12):
            dedup.append(arr[i])
    fit_arr = np.asarray(dedup, dtype=float)
    if fit_arr.shape[0] < 4:
        return arr.copy()

    n_points = arr.shape[0]
    deltas = np.linalg.norm(np.diff(fit_arr, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(deltas)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return arr.copy()

    u = cumulative / total
    k = min(3, fit_arr.shape[0] - 1)
    try:
        s_val = max(float(smoothness), 0.0) * float(fit_arr.shape[0])
        tck, _ = splprep(fit_arr.T, u=u, s=s_val, k=k)
        u_new = np.linspace(0.0, 1.0, n_points)
        x_new, y_new, z_new = splev(u_new, tck)
        out = np.column_stack((x_new, y_new, z_new))
        out[0] = arr[0]
        out[-1] = arr[-1]
        return out
    except Exception:
        return arr.copy()


def _blend_polyline(original: np.ndarray, smoothed: np.ndarray, alpha: float) -> np.ndarray:
    """Blend original and smoothed polylines with fixed endpoints."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mixed = (1.0 - alpha) * original + alpha * smoothed
    mixed[0] = original[0]
    mixed[-1] = original[-1]
    return mixed


def _skeleton_kdtree_physical(
    skeleton_array: np.ndarray, voxel_size: Tuple[float, float, float]
) -> Tuple[cKDTree, np.ndarray]:
    """Build physical-space KD-tree for skeleton support voxels."""
    hits = np.argwhere(skeleton_array)
    if hits.size == 0:
        return None, None
    phys = hits.astype(float) * np.asarray(voxel_size, dtype=float)
    return cKDTree(phys), phys


def _polyline_within_skeleton_distance(
    points: np.ndarray, skeleton_tree: cKDTree, max_distance: float
) -> bool:
    """Validate every point in polyline is close to skeleton support."""
    if skeleton_tree is None:
        return False
    dists, _ = skeleton_tree.query(points, k=1)
    return bool(np.all(np.asarray(dists, dtype=float) <= float(max_distance)))


def smooth_graph_edge_centerlines_continuous(
    G: Union[nx.Graph, nx.MultiGraph],
    skeleton_data: Any,
    *,
    smoothing_options: Optional[Dict[str, Any]] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smoothing_method: str = "bspline",
    chaikin_iterations: int = 2,
    bspline_smoothness: float = 0.75,
    max_distance_vox: float = 1.0,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Smooth all edge centerlines in physical space with configurable method.

    Smoothing is accepted only when every generated point remains within
    ``max_distance_vox * min(voxel_size)`` of skeleton support in physical space.
    """
    options = dict(smoothing_options or {})
    method = str(options.get("method", smoothing_method)).strip().lower()
    bspline_smoothness = float(options.get("s", bspline_smoothness))
    max_distance_vox = float(options.get("max_vox_dist_from_skel", max_distance_vox))
    if bspline_smoothness < 0.0:
        bspline_smoothness = 0.0
    if max_distance_vox <= 0.0:
        max_distance_vox = 1.0

    if method not in {"chaikin", "bspline"}:
        raise ValueError(
            f"Unsupported smoothing_method='{method}'. "
            "Expected 'chaikin' or 'bspline'."
        )

    def _make_candidate(original_points: np.ndarray, iterations: int) -> np.ndarray:
        if method == "chaikin":
            return chaikin_smooth_polyline(original_points, iterations=iterations)
        return bspline_smooth_polyline(original_points, smoothness=bspline_smoothness)

    skeleton_array = parse_skeleton_data(skeleton_data)
    if skeleton_array is None:
        return {
            "method": method,
            "smoothed_edges": 0,
            "fallback_edges": 0,
            "skipped_edges": 0,
        }

    skeleton_tree, _ = _skeleton_kdtree_physical(skeleton_array, voxel_size)
    max_distance_phys = float(max_distance_vox) * float(np.min(np.asarray(voxel_size, dtype=float)))

    smoothed_edges = 0
    relaxed_edges = 0
    fallback_edges = 0
    skipped_edges = 0

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_iter = list(G.edges(keys=True, data=True))
        for u, v, key, data in edge_iter:
            voxels = data.get("voxels")
            if not voxels or len(voxels) < 3:
                skipped_edges += 1
                continue
            original = np.asarray(voxels, dtype=float)
            accepted = None
            was_relaxed = False

            iteration_candidates = (
                range(int(chaikin_iterations), -1, -1)
                if method == "chaikin"
                else [0]
            )
            for iters in iteration_candidates:
                candidate = _make_candidate(original, iters)
                if _polyline_within_skeleton_distance(candidate, skeleton_tree, max_distance_phys):
                    accepted = candidate
                    break

                # If still outside, relax smoothing strength (continuous blend).
                for alpha in (0.8, 0.6, 0.4, 0.2, 0.1, 0.05):
                    relaxed = _blend_polyline(original, candidate, alpha)
                    if _polyline_within_skeleton_distance(relaxed, skeleton_tree, max_distance_phys):
                        accepted = relaxed
                        was_relaxed = True
                        break
                if accepted is not None:
                    break

            if accepted is not None and not np.allclose(accepted, original, atol=1e-8):
                new_len = calculate_path_length(accepted.tolist())
                data["voxels"] = accepted.tolist()
                data["length"] = new_len
                data["weight"] = max(new_len, 1e-6)
                smoothed_edges += 1
                if was_relaxed:
                    relaxed_edges += 1
            else:
                fallback_edges += 1
    else:
        edge_iter = list(G.edges(data=True))
        for u, v, data in edge_iter:
            voxels = data.get("voxels")
            if not voxels or len(voxels) < 3:
                skipped_edges += 1
                continue
            original = np.asarray(voxels, dtype=float)
            accepted = None
            was_relaxed = False

            iteration_candidates = (
                range(int(chaikin_iterations), -1, -1)
                if method == "chaikin"
                else [0]
            )
            for iters in iteration_candidates:
                candidate = _make_candidate(original, iters)
                if _polyline_within_skeleton_distance(candidate, skeleton_tree, max_distance_phys):
                    accepted = candidate
                    break
                for alpha in (0.8, 0.6, 0.4, 0.2, 0.1, 0.05):
                    relaxed = _blend_polyline(original, candidate, alpha)
                    if _polyline_within_skeleton_distance(relaxed, skeleton_tree, max_distance_phys):
                        accepted = relaxed
                        was_relaxed = True
                        break
                if accepted is not None:
                    break

            if accepted is not None and not np.allclose(accepted, original, atol=1e-8):
                new_len = calculate_path_length(accepted.tolist())
                data["voxels"] = accepted.tolist()
                data["length"] = new_len
                data["weight"] = max(new_len, 1e-6)
                smoothed_edges += 1
                if was_relaxed:
                    relaxed_edges += 1
            else:
                fallback_edges += 1

    if debug:
        print(
            "Continuous centerline smoothing: "
            f"method={method}, smoothed="
            f"{smoothed_edges}, relaxed={relaxed_edges}, "
            f"fallback={fallback_edges}, skipped={skipped_edges}"
        )
    return {
        "method": method,
        "smoothed_edges": smoothed_edges,
        "relaxed_edges": relaxed_edges,
        "fallback_edges": fallback_edges,
        "skipped_edges": skipped_edges,
    }


def line_voxels_3d(start, end):
    """Rasterize a 3D line segment to integer voxels (inclusive endpoints)."""
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    steps = int(np.max(np.abs(end_arr - start_arr)))
    if steps <= 0:
        return [tuple(np.round(start_arr).astype(int))]

    out = []
    for i in range(steps + 1):
        t = i / steps
        p = start_arr + t * (end_arr - start_arr)
        key = tuple(np.round(p).astype(int))
        if not out or out[-1] != key:
            out.append(key)
    return out


def has_skeleton_line_of_sight(skeleton_array, a, b):
    """True when straight line from a to b lies entirely on skeleton voxels."""
    shape = skeleton_array.shape
    for voxel in line_voxels_3d(a, b):
        vx, vy, vz = voxel
        if not (0 <= vx < shape[0] and 0 <= vy < shape[1] and 0 <= vz < shape[2]):
            return False
        if not bool(skeleton_array[vx, vy, vz]):
            return False
    return True


def smooth_skeleton_path_voxels(path, skeleton_array):
    """
    Shortcut a voxel path while keeping every shortcut segment on the skeleton.

    The smoothed path always preserves endpoints and never introduces off-skeleton
    coordinates.
    """
    if path is None:
        return None
    voxels = [tuple(np.asarray(p, dtype=int)) for p in path]
    if len(voxels) <= 2:
        return voxels

    simplified = [voxels[0]]
    anchor_idx = 0
    last_idx = len(voxels) - 1
    while anchor_idx < last_idx:
        next_idx = anchor_idx + 1
        for cand_idx in range(last_idx, anchor_idx, -1):
            if has_skeleton_line_of_sight(
                skeleton_array,
                voxels[anchor_idx],
                voxels[cand_idx],
            ):
                next_idx = cand_idx
                break
        simplified.append(voxels[next_idx])
        anchor_idx = next_idx

    # Light prune for local near-collinear triples, still LOS-constrained.
    pruned = [simplified[0]]
    for i in range(1, len(simplified) - 1):
        prev_v = pruned[-1]
        cur_v = simplified[i]
        nxt_v = simplified[i + 1]
        if has_skeleton_line_of_sight(skeleton_array, prev_v, nxt_v):
            continue
        pruned.append(cur_v)
    pruned.append(simplified[-1])
    return pruned

def are_paths_similar(voxels1, voxels2, tolerance=3.0):
    """Check if two paths connect similar endpoints."""
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

def voxel_path_overlap_ratio(path_a: List, path_b: List) -> float:
    """Return overlap ratio between two voxel paths based on rounded voxels."""
    if not path_a or not path_b:
        return 0.0
    set_a = set(_voxel_key(p) for p in path_a)
    set_b = set(_voxel_key(p) for p in path_b)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a.intersection(set_b))
    return overlap / max(len(set_a), len(set_b))

def improve_straight_path_with_skeleton(start_pos, end_pos, skeleton_data, debug=False, voxel_size=(1.0, 1.0, 1.0)):
    """
    Improve an entire straight path between two endpoints using skeleton.
    """
    return improve_straight_edge_with_skeleton(start_pos, end_pos, skeleton_data, debug, voxel_size=voxel_size)