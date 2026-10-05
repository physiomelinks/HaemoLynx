"""Internal helpers for graph operations."""
import logging
from typing import List, Tuple, Dict, Any, Callable, Union

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def edge_id(u: Any, v: Any, key: Any) -> Tuple[Any, Any, Any]:
    """Orientation-independent id for a MultiGraph edge.

    ``(u, v, key)`` and ``(v, u, key)`` name the same edge, so callers that key
    dicts or sets by edge must normalise first or they will count it twice.
    """
    return (u, v, key) if u <= v else (v, u, key)


def sort_nodes(nodes) -> List[Any]:
    """Deterministic node order for reproducible output.

    Node ids can be a mix of types, which is unorderable in Python 3, so they
    are ordered by type name then string form. Duplicates are dropped.
    """
    return sorted(set(nodes), key=lambda n: (str(type(n)), str(n)))


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

    merged_voxels = merge_voxel_paths_at_node(voxels1, voxels2, node_pos)


    # Validate the merged path for continuity
    if len(merged_voxels) > 1:
        max_gap = validate_voxel_path_continuity(merged_voxels)
        if max_gap > 5.0:  # Large gap indicates problematic merge
            logger.warning(f"Large gap ({max_gap:.2f}) in merged voxel path - may be discontinuous")

    # Length always comes from the merged path. The additive sum is kept only as a
    # diagnostic: the two diverge legitimately when an upstream step re-routes a
    # path through the skeleton, and the path is the truth in that case.
    length_from_voxels = calculate_path_length(merged_voxels)
    length_additive = edge1_data.get('length', 0) + edge2_data.get('length', 0)
    final_length = length_from_voxels
    
    # Create merged attributes
    merged_attrs = {
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

def validate_voxel_path_continuity(voxels):
    """
    Check if a voxel path is continuous and return the maximum gap.
    """
    if len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    return float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

#: Two path points closer than this (microns, per axis) are the same point.
#: Edge paths meeting at a shared node terminate on the identical coordinate, so
#: this only absorbs floating-point noise.
JUNCTION_TOLERANCE_UM = 1e-6


def _points_coincide(
    point_a: Any, point_b: Any, tolerance: float = JUNCTION_TOLERANCE_UM
) -> bool:
    """Compare two physical points without quantising them to voxel indices."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    if a.shape != b.shape:
        return False
    return bool(np.all(np.abs(a - b) <= tolerance))


def _as_point(point: Any) -> Tuple[float, ...]:
    return tuple(float(c) for c in np.asarray(point, dtype=float).ravel())


def merge_voxel_paths_at_node(
    voxels1: List,
    voxels2: List,
    node_pos: Any,
    *,
    tolerance: float = JUNCTION_TOLERANCE_UM,
) -> List[Tuple[float, ...]]:
    """Join two edge voxel paths at their shared node, in physical microns.

    Paths and ``node_pos`` are physical ``(z, y, x)`` coordinates, so the
    junction is inserted at its exact position rather than being rounded to an
    integer voxel index. Both incident edges already terminate on the shared
    node, so the insertion is normally a no-op — it only fires when an upstream
    step (e.g. skeleton re-routing) left a path short of the junction.

    This is the single implementation behind :func:`merge_curved_edges`,
    :func:`merge_edge_voxels_at_node` and :func:`create_merged_edge_attributes`,
    which previously quantised the junction in three different ways.
    """
    if node_pos is None:
        return [_as_point(p) for p in list(voxels1) + list(voxels2)]

    node_point = _as_point(node_pos)
    part1 = [_as_point(p) for p in orient_path_to_endpoint(voxels1, node_point)]
    part2 = [_as_point(p) for p in orient_path_from_startpoint(voxels2, node_point)]

    merged = list(part1)
    if not merged or not _points_coincide(merged[-1], node_point, tolerance):
        merged.append(node_point)

    start_idx = 1 if (part2 and _points_coincide(part2[0], node_point, tolerance)) else 0
    merged.extend(part2[start_idx:])
    return merged


def merge_edge_voxels_at_node(voxels1: List, voxels2: List, node_pos: Any) -> List:
    """Concatenate two edge voxel paths at the removed node with orientation."""
    return merge_voxel_paths_at_node(voxels1, voxels2, node_pos)

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
    """Length of a voxel path in microns: sum of consecutive point distances."""
    if not voxels or len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


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
    
    # Fallback
    return 1.0
    


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
    return merge_voxel_paths_at_node(voxels1, voxels2, connection_pos)

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

def trace_skeleton_path(skeleton_data, start_pos, end_pos, debug=False, voxel_size=(1.0, 1.0, 1.0)):
    """
    Trace path through skeleton data from start_pos to end_pos using A* pathfinding.
    
    Positions are in physical units; *voxel_size* converts them to array
    indices for look-ups.  The returned path is converted back to physical
    coordinates. The search for the nearest skeleton voxel and the path's
    step costs are physical too: in voxels, the path with fewest voxels won
    over the shortest one, and a merged edge's ``voxels`` and ``length``
    followed it.
    """
    vs = np.asarray(voxel_size, dtype=float)
    relative = vs / float(vs.min())
    start_vox = np.round(np.asarray(start_pos, dtype=float) / vs).astype(int)
    end_vox = np.round(np.asarray(end_pos, dtype=float) / vs).astype(int)
    
    if debug:
        logger.debug(f"       Tracing skeleton from {start_pos} (vox {start_vox}) to {end_pos} (vox {end_vox})")
    
    skeleton_array = parse_skeleton_data(skeleton_data)
    if skeleton_array is None:
        if debug:
            logger.debug(f"       Could not parse skeleton data")
        return None
    
    start_skeleton = find_nearest_skeleton_voxel(skeleton_array, start_vox, spacing=relative)
    end_skeleton = find_nearest_skeleton_voxel(skeleton_array, end_vox, spacing=relative)
    
    if start_skeleton is None or end_skeleton is None:
        if debug:
            logger.debug(f"       Could not find skeleton voxels near start/end positions")
        return None
    
    if debug:
        start_dist = np.linalg.norm(np.array(start_pos) - np.array(start_skeleton))
        end_dist = np.linalg.norm(np.array(end_pos) - np.array(end_skeleton))
        logger.debug(f"       Start skeleton voxel: {start_skeleton} (dist: {start_dist:.1f})")
        logger.debug(f"       End skeleton voxel: {end_skeleton} (dist: {end_dist:.1f})")
    
    path = astar_skeleton_path(skeleton_array, start_skeleton, end_skeleton, debug, spacing=relative)
    
    if path:
        if debug:
            logger.debug(f"       Found skeleton path with {len(path)} voxels")
        phys_path = [(np.array(p, dtype=float) * vs).tolist() for p in path]
        return phys_path
    else:
        if debug:
            logger.debug(f"       No skeleton path found")
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


def find_nearest_skeleton_voxel(skeleton_array, target_pos, max_search_radius=10, spacing=None):
    """
    Find the nearest skeleton voxel to target_pos within search radius.

    Radius and distance in voxels of the finest axis, each axis weighted by
    *spacing* (per-axis spacing relative to the finest; cube voxels when
    ``None``), so the search reaches the same physical distance every way.
    """
    target = np.array(target_pos, dtype=int)
    shape = skeleton_array.shape
    relative = np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)

    if (target >= 0).all() and (target < shape).all():
        if skeleton_array[tuple(target)]:
            return tuple(target)

    for radius in range(1, max_search_radius + 1):
        reach = np.ceil(radius / relative - 1e-9).astype(int)
        lo = np.maximum(target - reach, 0)
        hi = np.minimum(target + reach + 1, shape)

        sub = skeleton_array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not np.any(sub):
            continue

        local_hits = np.argwhere(sub) + lo
        dists = np.linalg.norm((local_hits - target) * relative, axis=1)
        within = dists <= radius
        if np.any(within):
            best = int(np.argmin(np.where(within, dists, np.inf)))
            return tuple(local_hits[best])
    
    return None


def astar_skeleton_path(skeleton_array, start, end, debug=False, spacing=None):
    """
    A* pathfinding through skeleton voxels only, each step costing its
    physical length (*spacing*, per axis relative to the finest; cube voxels
    when ``None``).
    """
    import heapq
    from collections import defaultdict
    
    start = tuple(start)
    end = tuple(end)
    
    if start == end:
        return [start]
    
    ex, ey, ez = end
    sx, sy, sz = skeleton_array.shape
    wx, wy, wz = (1.0, 1.0, 1.0) if spacing is None else (float(s) for s in spacing)

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
            
            distance = ((dx * wx) ** 2 + (dy * wy) ** 2 + (dz * wz) ** 2) ** 0.5
            tentative_g = cur_g + distance
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                hdx = nx_ - ex
                hdy = ny_ - ey
                hdz = nz_ - ez
                f = tentative_g + ((hdx * wx) ** 2 + (hdy * wy) ** 2 + (hdz * wz) ** 2) ** 0.5
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    
    return None

def path_separation(
    path_a: List, path_b: List, step_um: float = 0.5, *, ignore_within=None
) -> float:
    """Furthest either polyline strays from the other (symmetric Hausdorff distance).

    Both paths are sampled every *step_um* along their length, so the result
    is within ``step_um / 2`` of the exact distance however sparse the inputs.

    *ignore_within*, a ``(centre, radius)`` pair, leaves the samples of either
    path within *radius* of *centre* out of the measurement (they are still
    what the other path's samples are measured to); 0.0 when nothing is left
    to measure.
    """
    a = densify_polyline(np.asarray(path_a, dtype=float), max_step_um=step_um)
    b = densify_polyline(np.asarray(path_b, dtype=float), max_step_um=step_um)
    if a.ndim != 2 or b.ndim != 2 or len(a) == 0 or len(b) == 0:
        return float("inf")
    from_a, from_b = a, b
    if ignore_within is not None:
        centre, radius = np.asarray(ignore_within[0], dtype=float), float(ignore_within[1])
        from_a = a[np.linalg.norm(a - centre, axis=1) > radius]
        from_b = b[np.linalg.norm(b - centre, axis=1) > radius]
    a_to_b = cKDTree(b).query(from_a)[0].max() if len(from_a) else 0.0
    b_to_a = cKDTree(a).query(from_b)[0].max() if len(from_b) else 0.0
    return float(max(a_to_b, b_to_a))


def paths_separated_by_background(
    path_a: List,
    path_b: List,
    inside_lumen: Callable[[np.ndarray], np.ndarray],
    *,
    step_um: float = 0.5,
    min_gap_um: float = 1.0,
    ignore_within=None,
) -> bool:
    """Whether the segmentation shows two vessels along two paths, not one.

    Each sample of either path (every *step_um*) is joined by a straight
    chord to the nearest point of the other. A chord with both ends in the
    lumen *separates* the paths when at least *min_gap_um* of it crosses
    background: two lumens with tissue between them. The paths are two
    vessels when at least half of those chords separate them.

    A chord with an end outside the lumen is not counted: a path running
    outside the mask (a straight shortcut, a gap bridge) is no evidence of a
    second vessel. Nor is a chord shorter than ``2 * min_gap_um``, where the
    paths meet. With no chord left to judge, the paths are one vessel.

    *inside_lumen* maps ``(N, 3)`` physical ``(z, y, x)`` points to one bool
    each (see ``assemble.mask_lumen_test``); *ignore_within* is as for
    :func:`path_separation`.
    """
    a = densify_polyline(np.asarray(path_a, dtype=float), max_step_um=step_um)
    b = densify_polyline(np.asarray(path_b, dtype=float), max_step_um=step_um)
    if a.ndim != 2 or b.ndim != 2 or len(a) < 2 or len(b) < 2:
        return False
    starts, ends = [], []
    for here, there in ((a, b), (b, a)):
        if ignore_within is not None:
            centre, radius = np.asarray(ignore_within[0], dtype=float), float(ignore_within[1])
            here = here[np.linalg.norm(here - centre, axis=1) > radius]
        if len(here):
            starts.append(here)
            ends.append(there[cKDTree(there).query(here)[1]])
    if not starts:
        return False
    p, q = np.vstack(starts), np.vstack(ends)
    length = np.linalg.norm(q - p, axis=1)
    long_enough = length >= 2.0 * min_gap_um
    p, q, length = p[long_enough], q[long_enough], length[long_enough]
    if not len(p):
        return False
    # Every chord gets as many samples as the longest needs, so each sample
    # stands for length / (n - 1) of its own chord.
    n = int(np.ceil(length.max() / step_um)) + 1
    t = np.linspace(0.0, 1.0, n)
    samples = p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]
    inside = np.asarray(inside_lumen(samples.reshape(-1, 3)), dtype=bool).reshape(len(p), n)
    judged = inside[:, 0] & inside[:, -1]
    if not judged.any():
        return False
    background_um = (~inside).sum(axis=1) * length / (n - 1)
    separating = judged & (background_um >= min_gap_um)
    return bool(separating.sum() >= 0.5 * judged.sum())


def are_paths_similar(voxels1, voxels2, tolerance=3.0, inside_lumen=None):
    """True when two paths are the same vessel.

    They must join the same end points, and either never stray further than
    *tolerance* microns from each other or, given *inside_lumen* (the
    segmentation the skeleton came from), run through one lumen: not
    separated by background (:func:`paths_separated_by_background`). Shared
    end points alone are not enough: two distinct vessels between the same
    two junctions (a loop) share them too, and treating those as one path
    deletes a vessel. Distance alone is not enough either: Lee thinning
    leaves two or more strands 5-10 um apart through one wide vessel, and
    those are one vessel.
    """
    if len(voxels1) < 2 or len(voxels2) < 2:
        return False
    
    start1, end1 = np.array(voxels1[0]), np.array(voxels1[-1])
    start2, end2 = np.array(voxels2[0]), np.array(voxels2[-1])
    
    # Check both orientations
    dist_same = np.linalg.norm(start1 - start2) + np.linalg.norm(end1 - end2)
    dist_flipped = np.linalg.norm(start1 - end2) + np.linalg.norm(end1 - start2)
    
    if min(dist_same, dist_flipped) > tolerance * 2:
        return False
    if path_separation(voxels1, voxels2) <= tolerance:
        return True
    return inside_lumen is not None and not paths_separated_by_background(
        voxels1, voxels2, inside_lumen
    )
    
def should_add_merged_edge(G, n1, n2, new_voxels, new_attrs, debug=False, inside_lumen=None):
    """
    Check if we should add this merged edge, avoiding duplicates.

    *inside_lumen*, when given, lets a duplicate be judged by the
    segmentation as well as by distance (see :func:`are_paths_similar`).
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
        if are_paths_similar(new_voxels, existing_voxels, inside_lumen=inside_lumen):
            # Prefer curved over straight
            if new_is_curved and not existing_is_curved:
                if debug:
                    logger.debug(f"     Replacing straight with curved path")
                return True, edge_key
            elif not new_is_curved and existing_is_curved:
                if debug:
                    logger.debug(f"     Keeping existing curved over new straight")
                return False, None
            else:
                # Same type - prefer shorter
                if new_length < existing_length * 0.95:
                    if debug:
                        logger.debug(f"     Replacing with shorter path ({new_length:.1f} vs {existing_length:.1f})")
                    return True, edge_key
                else:
                    if debug:
                        logger.debug(f"     Keeping existing shorter path ({existing_length:.1f} vs {new_length:.1f})")
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


def points_inside_mask(
    points_zyx: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: Tuple[float, float, float],
) -> np.ndarray:
    """True per point where the physical point falls in a True (interior) mask voxel.

    Vectorized over all of an edge's (densified) sample points at once --
    the densified polyline can have hundreds of points per edge, and this
    runs for every edge in the graph on every pipeline run that reaches
    this stage.
    """
    points = np.asarray(points_zyx, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_zyx must be (N, 3), got shape {points.shape}.")
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_zyx must be three positive values, got {voxel_size_zyx}."
        )
    idx = np.rint(points / voxel_size).astype(int)
    mask_shape = np.asarray(mask.shape, dtype=int)
    in_bounds = np.all((idx >= 0) & (idx < mask_shape), axis=1)
    inside = np.zeros(n, dtype=bool)
    valid = idx[in_bounds]
    if valid.size:
        inside[in_bounds] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return inside


def edge_sample_points(
    u: Any,
    v: Any,
    edge_data: Dict[str, Any],
    node_pos: Dict[Any, np.ndarray],
) -> np.ndarray:
    """Physical polyline for an edge, oriented from ``u`` toward ``v``."""
    pos_u = np.asarray(node_pos[u], dtype=float)
    pos_v = np.asarray(node_pos[v], dtype=float)
    voxels = edge_data.get("voxels")
    if voxels is not None:
        arr = np.asarray(voxels, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
            oriented = orient_path_from_startpoint(arr.tolist(), pos_u)
            return np.asarray(oriented, dtype=float)
    return np.vstack([pos_u.reshape(1, 3), pos_v.reshape(1, 3)])


def densify_polyline(
    points: np.ndarray,
    *,
    max_step_um: float,
) -> np.ndarray:
    """Sample a polyline so consecutive points are at most ``max_step_um`` apart.

    Sparse centreline voxels (or a two-endpoint fallback) can leave a straight
    segment with both ends outside a mask while every interior mask voxel along
    the chord is missed. Densifying before the inside/outside test catches those
    crossings.
    """
    if max_step_um <= 0:
        raise ValueError(f"max_step_um must be > 0, got {max_step_um}.")
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return arr
    if arr.shape[0] == 1:
        return arr
    dense: List[np.ndarray] = [arr[0]]
    for start, end in zip(arr[:-1], arr[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= max_step_um:
            dense.append(end)
            continue
        steps = int(np.ceil(length / max_step_um))
        for step in range(1, steps + 1):
            dense.append(start + (step / steps) * segment)
    return np.asarray(dense, dtype=float)


class EdgeSampleIndex:
    """Every edge's centreline points in one KD-tree, for
    :func:`duplicates_existing_vessel`: built once per pass, not per question.

    Paths a pass adds after building it go in with :meth:`add`; they are
    judged against separately, so the tree is never rebuilt. Each edge is
    sampled at least every *step_um*, so a straight edge stored as its two
    ends still has a point beside every point along it.
    """

    def __init__(self, G: nx.MultiGraph, step_um: float = 1.0):
        from haemolynx.preprocessing.bridge_mask_support import _densify

        node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
        points, owners = [], []
        for u, v, key, data in G.edges(keys=True, data=True):
            if u not in node_pos or v not in node_pos:
                continue
            path = _densify(edge_sample_points(u, v, data, node_pos), step_um)
            points.append(path)
            owners.extend([edge_id(u, v, key)] * len(path))
        self.points = np.vstack(points) if points else np.empty((0, 3))
        self.owners = owners
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.added: List[np.ndarray] = []
        # Each added path's bounding box, (lo, hi) per row, grown by doubling:
        # mask recovery adds thousands of paths and asks about each new one,
        # and a box test per added path in Python made that quadratic.
        self._boxes = np.empty((16, 2, 3))

    def add(self, path: Any) -> None:
        from haemolynx.preprocessing.bridge_mask_support import _densify

        dense = _densify(np.asarray(path, dtype=float).reshape(-1, 3), 1.0)
        if len(self.added) == len(self._boxes):
            self._boxes = np.concatenate([self._boxes, np.empty_like(self._boxes)])
        self._boxes[len(self.added)] = (dense.min(axis=0), dense.max(axis=0))
        self.added.append(dense)

    def added_near(self, lo: np.ndarray, hi: np.ndarray) -> List[np.ndarray]:
        """The added paths whose bounding box meets the box *lo*..*hi*, in
        the order they were added."""
        boxes = self._boxes[: len(self.added)]
        meets = np.all(boxes[:, 1] >= lo, axis=1) & np.all(boxes[:, 0] <= hi, axis=1)
        return [self.added[i] for i in np.flatnonzero(meets)]


def duplicates_existing_vessel(
    new_path: Any,
    G: nx.MultiGraph,
    inside_lumen: Callable[[np.ndarray], np.ndarray],
    radius_at: Callable[[np.ndarray], np.ndarray],
    *,
    index: "EdgeSampleIndex | None" = None,
    step_um: float = 0.5,
) -> bool:
    """Whether a path about to be added to *G* runs beside one of its edges
    in the same lumen: a second, parallel vessel inside one segmented branch.

    The test is ``preprocessing.bridge_mask_support.path_shadows_existing_vessel``
    against every edge's centreline (*index*, built from *G* when ``None``)
    and, separately, every path :meth:`EdgeSampleIndex.add` recorded since.
    The new path's own two ends are left out, so a branch meeting the vessel
    it joins, or a bridge continuing one, is not a duplicate. *inside_lumen*
    and *radius_at* map ``(N, 3)`` physical points to a bool / the lumen
    radius in microns each.
    """
    from haemolynx.preprocessing.bridge_mask_support import path_shadows_existing_vessel

    index = EdgeSampleIndex(G) if index is None else index
    if index.tree is not None and path_shadows_existing_vessel(
        new_path, index.tree, index.points, inside_lumen, radius_at, step_um=step_um
    ):
        return True
    if not index.added:
        return False
    path = np.asarray(new_path, dtype=float).reshape(-1, 3)
    reach = 2.0 * float(np.max(radius_at(path))) + 1.0
    lo, hi = path.min(axis=0) - reach, path.max(axis=0) + reach
    nearby = index.added_near(lo, hi)
    return bool(nearby) and path_shadows_existing_vessel(
        path, None, np.vstack(nearby), inside_lumen, radius_at, step_um=step_um
    )


def next_node_id(G: nx.MultiGraph, reserved: "set[Any]") -> int:
    """A numeric node id not already used by ``G`` or ``reserved``."""
    numeric = [
        int(n)
        for n in list(G.nodes) + list(reserved)
        if isinstance(n, (int, np.integer))
    ]
    return (max(numeric) if numeric else -1) + 1