"""Prune short terminal stubs from vascular graph."""
from typing import Tuple, Union

import networkx as nx

from ._helpers import calculate_edge_length

def prune_vascular_stubs(
    G: Union[nx.Graph, nx.MultiGraph],
    min_stub_length: float = 10.0,
    max_iterations: int = 100,
    debug: bool = False,
    voxel_size: Tuple[float, float, float] = (1, 1, 1),
) -> Union[nx.Graph, nx.MultiGraph]:
    """Iteratively remove short terminal stubs until convergence.

    ``min_stub_length`` is in the units ``voxel_size`` puts edge lengths in - MICRONS whenever a
    real voxel size is passed, voxels only at (1, 1, 1). The default of 10.0 is a bare number
    inherited from when everything ran at unit spacing, where it happened to mean 10 voxels;
    callers should pass an explicit, physically justified value rather than rely on it.

    This cannot change beta-1. Only degree-1 nodes are removed, and a degree-1 node lies on no
    cycle, so vascular loop topology - the H1 section 1.1 readout - is untouched at any
    threshold. It does change the per-edge length and tortuosity distributions, since it removes
    the shortest terminal segments first.
    """
    if min_stub_length < 0:
        raise ValueError("min_stub_length must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must be a 3-tuple")

    G_pruned = G.copy()
    if G_pruned.number_of_nodes() == 0:
        return G_pruned

    total_removed = 0
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        nodes_before = G_pruned.number_of_nodes()
        if nodes_before == 0:
            break
        nodes_to_remove = []
        terminal_nodes = [
            n for n in G_pruned.nodes() if G_pruned.degree(n) == 1
        ]
        for node in terminal_nodes:
            if node not in G_pruned:
                continue
            neighbors = list(G_pruned.neighbors(node))
            if not neighbors:
                nodes_to_remove.append(node)
                continue
            neighbor = neighbors[0]
            try:
                if isinstance(G_pruned, nx.MultiGraph):
                    edge_data_list = list(G_pruned[node][neighbor].values())
                    edge_length = min(
                        calculate_edge_length(
                            node, neighbor, ed, voxel_size
                        )
                        for ed in edge_data_list
                    )
                else:
                    edge_data = G_pruned[node][neighbor]
                    edge_length = calculate_edge_length(
                        node, neighbor, edge_data, voxel_size
                    )
                if edge_length < min_stub_length:
                    nodes_to_remove.append(node)
                    if debug:
                        print(
                            f"  Iteration {iteration}: Marking node {node} "
                            f"(stub length: {edge_length:.2f})"
                        )
            except Exception as e:
                if debug:
                    print(f"  Warning: Could not calculate edge length: {e}")
                nodes_to_remove.append(node)

        G_pruned.remove_nodes_from(nodes_to_remove)
        nodes_after = G_pruned.number_of_nodes()
        removed_this_iteration = nodes_before - nodes_after
        total_removed += removed_this_iteration

        if debug:
            print(
                f"  Iteration {iteration}: Removed {removed_this_iteration} "
                f"({nodes_after} remaining)"
            )
        if removed_this_iteration == 0:
            if debug:
                print(f"Convergence reached after {iteration} iterations")
            break

    if debug:
        print(f"\nPruning complete: Total nodes removed: {total_removed}")
    return G_pruned

def remove_edges_for_self_connected_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove edges for nodes that are connected to themselves with no nodes in between."""
    G_pruned = G.copy()
    for node in G_pruned.nodes():
        if node in G_pruned.neighbors(node):
            G_pruned.remove_edge(node, node)
    return G_pruned

def resolve_core_dead_ends(
    G: nx.MultiGraph,
    image_shape: tuple[int, ...],
    voxel_size_xyz: tuple[float, float, float],
    mode: str = "none",
    safe_zone_percent: float = 5.0,
    max_stitch_distance_um: float = 15.0,
    max_degree: int = 4
) -> dict:
    """
    Plan A (Eradicate) and Plan B (Stitch) implementation for resolving Degree-1
    dead-ends that fall deep within the internal core of the sub-volume, 
    preserving valid inlets/outlets in the boundary safe zones.
    """
    if mode not in ["eradicate", "stitch"]:
        return {}
        
    initial_edges = G.number_of_edges()
    if initial_edges == 0:
        return {}
        
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Node 'pos' is in physical units but image_shape is in voxels, so the extent has to be
    # scaled by the spacing before the two are compared.
    z_max, y_max, x_max = [
        float(s - 1) * float(v) for s, v in zip(image_shape, voxel_size_xyz)
    ]
    z_marg = z_max * (safe_zone_percent / 100.0)
    y_marg = y_max * (safe_zone_percent / 100.0)
    x_marg = x_max * (safe_zone_percent / 100.0)
    
    node_pos = nx.get_node_attributes(G, "pos")
    
    def is_core(n):
        pos = node_pos.get(n)
        if pos is None: 
            return False
        z, y, x = pos
        in_z_safe = (z <= z_marg) or (z >= z_max - z_marg)
        in_y_safe = (y <= y_marg) or (y >= y_max - y_marg)
        in_x_safe = (x <= x_marg) or (x >= x_max - x_marg)
        return not (in_z_safe or in_y_safe or in_x_safe)
        
    edges_added = 0
    edges_removed = 0
    fallback_eradicated = 0
    
    core_deg1 = [n for n in G.nodes() if G.degree(n) == 1 and is_core(n)]
    fallback_queue = []
    
    if mode == "stitch" and core_deg1:
        for n in core_deg1:
            if G.degree(n) != 1:
                continue
                
            # Filter valid targets (Must not exceed degree limit, must not be self or already connected)
            valid_nodes = [
                t for t in G.nodes() 
                if G.degree(t) < max_degree and t != n and not G.has_edge(n, t) and t in node_pos
            ]
            
            if not valid_nodes:
                fallback_queue.append(n)
                continue
            
            # Convert voxel coordinates to physical microns for distance check
            v_size = np.array(voxel_size_xyz)
            pts = np.array([node_pos[t] for t in valid_nodes]) * v_size
            n_pt = np.array(node_pos[n]) * v_size
            
            tree = cKDTree(pts)
            dist, idx = tree.query(n_pt)
            
            if dist <= max_stitch_distance_um:
                target = valid_nodes[idx]
                G.add_edge(n, target, length=float(dist), is_stitched=True)
                edges_added += 1
            else:
                fallback_queue.append(n)
                
    # Eradicate Mode (or Plan B fallbacks)
    queue = []
    if mode == "eradicate":
        queue = [n for n in G.nodes() if G.degree(n) == 1 and is_core(n)]
    else:
        queue = fallback_queue
        
    while queue:
        curr = queue.pop()
        if curr not in G or G.degree(curr) != 1:
            continue
        # Get the neighbor before we delete the node
        neighbor = list(G.neighbors(curr))[0]
        
        # Delete the dead end (recursively eats the branch backward)
        G.remove_node(curr)
        edges_removed += 1
        
        if mode == "stitch":
            fallback_eradicated += 1
            
        # If the deletion caused the neighbor to become a new dead-end, queue it up!
        # Note: We do NOT check is_core(neighbor) here. If a branch dies in the core, 
        # we eradicate the ENTIRE branch all the way back to its Degree-3 origin, 
        # even if it started in the boundary safe zone!
        if G.degree(neighbor) == 1:
            queue.append(neighbor)
            
    pct_added = (edges_added / initial_edges) * 100.0 if initial_edges > 0 else 0.0
    pct_removed = (edges_removed / initial_edges) * 100.0 if initial_edges > 0 else 0.0
    
    return {
        "initial_edges": initial_edges,
        "edges_added": edges_added,
        "edges_added_pct": round(pct_added, 2),
        "edges_removed": edges_removed,
        "edges_removed_pct": round(pct_removed, 2),
        "fallback_eradicated": fallback_eradicated
    }
