"""Internal helpers for graph operations."""
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import networkx as nx


def add_edge_safe(
    G: Union[nx.Graph, nx.MultiGraph], u: int, v: int, **attrs: Any
) -> None:
    """Add edge to Graph or MultiGraph with given attributes."""
    G.add_edge(u, v, **attrs)


def has_edge_safe(G: Union[nx.Graph, nx.MultiGraph], u: int, v: int) -> bool:
    """Check if edge exists between u and v."""
    return G.has_edge(u, v)


def remove_edge_safe(
    G: Union[nx.Graph, nx.MultiGraph], u: int, v: int, key: Any = None
) -> None:
    """Remove edge(s) between u and v."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if key is not None:
            if G.has_edge(u, v, key):
                G.remove_edge(u, v, key)
        else:
            while G.has_edge(u, v):
                keys = list(G[u][v].keys())
                G.remove_edge(u, v, keys[0])
    else:
        if G.has_edge(u, v):
            G.remove_edge(u, v)


def get_all_edge_data(
    G: Union[nx.Graph, nx.MultiGraph], u: int, v: int
) -> List[dict]:
    """Return list of edge data dicts (all parallel edges for MultiGraph)."""
    if not G.has_edge(u, v):
        return []
    ed = G.get_edge_data(u, v)
    if isinstance(ed, dict):
        # MultiGraph: ed = {key: {attrs}, ...}
        if isinstance(list(ed.values())[0], dict):
            return [d.copy() for d in ed.values()]
        return [ed.copy()]
    return [ed.copy()]


def create_merged_edge_attributes(
    edge1_data: dict, edge2_data: dict, node_pos: Any
) -> dict:
    """Merge two edge attributes when removing a degree-2 node."""
    voxels1 = edge1_data.get("voxels", [])
    voxels2 = edge2_data.get("voxels", [])
    merged_voxels = []
    if voxels1:
        merged_voxels.extend(voxels1)
    if node_pos is not None:
        node_voxel = tuple(np.array(node_pos).astype(int))
        if not merged_voxels or merged_voxels[-1] != node_voxel:
            merged_voxels.append(node_voxel)
    if voxels2:
        start_idx = 0
        if (
            voxels2
            and node_pos is not None
            and len(voxels2) > 0
            and tuple(np.array(voxels2[0]).astype(int)) == tuple(np.array(node_pos).astype(int))
        ):
            start_idx = 1
        merged_voxels.extend(voxels2[start_idx:])

    return {
        "weight": edge1_data.get("weight", 0) + edge2_data.get("weight", 0),
        "length": edge1_data.get("length", 0) + edge2_data.get("length", 0),
        "voxels": merged_voxels,
        "merged": True,
        "removed_node_pos": node_pos,
    }


def get_line_points_3d(p1: np.ndarray, p2: np.ndarray) -> List[Tuple[int, ...]]:
    """Return list of voxel coordinates along 3D line from p1 to p2."""
    p1 = np.round(p1).astype(int)
    p2 = np.round(p2).astype(int)
    n = max(
        abs(p2[0] - p1[0]),
        abs(p2[1] - p1[1]),
        abs(p2[2] - p1[2]),
        1,
    )
    t = np.linspace(0, 1, n + 1)
    pts = []
    for ti in t:
        pt = np.round(p1 + ti * (p2 - p1)).astype(int)
        pts.append(tuple(pt))
    return pts


def calculate_path_length(voxels: List) -> float:
    """Sum of Euclidean distances between consecutive voxels."""
    if len(voxels) < 2:
        return 0.0
    arr = np.array(voxels, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def calculate_edge_length(
    node: int,
    neighbor: int,
    edge_data: dict,
    voxel_size: Tuple[float, float, float],
) -> float:
    """Edge length from voxels or Euclidean distance * voxel scale."""
    voxels = edge_data.get("voxels", [])
    if len(voxels) >= 2:
        return calculate_path_length(
            [tuple(np.array(v) * np.array(voxel_size)) for v in voxels]
        )
    length = edge_data.get("length", edge_data.get("weight", 0))
    if length > 0:
        return float(length)
    return 0.0


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


def merge_curved_edges(
    voxels1: List, voxels2: List, node_pos: np.ndarray, debug: bool = False
) -> List:
    """Concatenate two curved paths at junction node."""
    merged = []
    if voxels1:
        merged.extend(voxels1)
    if node_pos is not None:
        v = tuple(np.round(node_pos).astype(int))
        if not merged or merged[-1] != v:
            merged.append(v)
    if voxels2:
        v0 = tuple(np.round(np.array(voxels2[0])).astype(int))
        np_v = tuple(np.round(node_pos).astype(int))
        start = 1 if v0 == np_v else 0
        merged.extend(voxels2[start:])
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


def improve_straight_path_with_skeleton(
    pos1: np.ndarray, pos2: np.ndarray, skeleton_data: np.ndarray, debug: bool = False
) -> List:
    """Find skeleton path between two points."""
    return improve_straight_edge_with_skeleton(pos1, pos2, skeleton_data, debug)


def should_add_merged_edge(
    G: nx.MultiGraph,
    n1: int,
    n2: int,
    merged_voxels: List,
    merged_attrs: dict,
    debug: bool = False,
) -> Tuple[bool, Any]:
    """Whether to add merged edge; return (should_add, replace_key or None)."""
    if not G.has_edge(n1, n2):
        return True, None
    for key, data in G[n1][n2].items():
        ev = data.get("voxels", [])
        if len(ev) > 0 and len(merged_voxels) > 0:
            sim = len(set(map(tuple, ev)) & set(map(tuple, merged_voxels)))
            sim /= max(len(ev), len(merged_voxels))
            if sim > 0.9 and merged_attrs.get("length", 0) < data.get("length", float("inf")):
                return True, key
            if sim > 0.9 and merged_attrs.get("length", 0) >= data.get("length", 0):
                return False, None
    return True, None
