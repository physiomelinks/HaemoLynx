"""Assign branch orders to graph edges via BFS from boundary nodes."""
import logging
from collections import defaultdict, deque

import networkx as nx

logger = logging.getLogger(__name__)


def _edge_id(u: int, v: int, key: int) -> tuple[int, int, int]:
    """Return an orientation-independent edge id for MultiGraph edges."""
    return (u, v, key) if u <= v else (v, u, key)


def _compute_node_distances(
    G: nx.MultiGraph,
    start_nodes: list[int],
    stop_nodes: set[int] | None = None,
) -> dict[int, int]:
    """Compute unweighted BFS node distances with optional stop nodes using igraph."""
    stop_nodes = stop_nodes or set()
    
    try:
        import igraph as ig
        # Convert to igraph for fast C-based traversals
        ig_G = ig.Graph.from_networkx(G)
        
        # To respect stop_nodes, we make the graph directed and remove outgoing 
        # edges from the stop nodes. This allows paths to reach them, but not 
        # pass through them.
        ig_G.to_directed(mode="mutual")
        
        name_to_vid = {v["_nx_name"]: v.index for v in ig_G.vs}
        
        if stop_nodes:
            stop_vids = [name_to_vid[n] for n in stop_nodes if n in name_to_vid]
            edges_to_delete = []
            for vid in stop_vids:
                edges_to_delete.extend(ig_G.incident(vid, mode="out"))
            ig_G.delete_edges(edges_to_delete)
            
        start_vids = [name_to_vid[n] for n in start_nodes if n in name_to_vid]
        
        if not start_vids:
            return {}
            
        # Get distances from all start nodes to all other nodes
        dists = ig_G.distances(source=start_vids, mode="out")
        
        # Find the minimum distance to each node from any start node
        min_dists = [min(d[j] for d in dists) for j in range(ig_G.vcount())]
        
        node_distances = {}
        for i, v in enumerate(ig_G.vs):
            d = min_dists[i]
            if d < float('inf'):
                node_distances[v["_nx_name"]] = int(d)
                
        return node_distances
        
    except ImportError:
        # Fallback to pure Python BFS if igraph is not available
        node_distances: dict[int, int] = {}
        from collections import deque
        queue: deque[tuple[int, int]] = deque()

        for start_node in start_nodes:
            if start_node in G.nodes():
                if start_node not in node_distances:
                    node_distances[start_node] = 0
                    queue.append((start_node, 0))
            else:
                logger.warning("Starting node %s not found in graph", start_node)

        while queue:
            current_node, distance = queue.popleft()
            if (
                current_node in node_distances
                and node_distances[current_node] < distance
            ):
                continue
            if current_node in stop_nodes:
                continue
            for neighbor in G.neighbors(current_node):
                new_distance = distance + 1
                if (
                    neighbor not in node_distances
                    or node_distances[neighbor] > new_distance
                ):
                    node_distances[neighbor] = new_distance
                    queue.append((neighbor, new_distance))
        return node_distances


def assign_branch_orders(
    G: nx.MultiGraph,
    starting_nodes: list[int],
    prefix: str = "B",
    stop_nodes: set[int] | None = None,
    excluded_edges: set[tuple[int, int, int]] | None = None,
    included_edges: set[tuple[int, int, int]] | None = None,
) -> dict:
    """Assign branch-order labels to edges from BFS distance to starting nodes."""
    node_distances = _compute_node_distances(
        G,
        starting_nodes,
        stop_nodes=stop_nodes,
    )
    excluded_edges = excluded_edges or set()
    included_edges = included_edges or set()

    results = {
        "edges_assigned": 0,
        "edges_skipped": 0,
        "branch_order_counts": defaultdict(int),
        "unreachable_edges": [],
        "excluded_edges": 0,
    }

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_identifier = _edge_id(u, v, key)
        if included_edges and edge_identifier not in included_edges:
            results["edges_skipped"] += 1
            continue
        if edge_identifier in excluded_edges:
            results["excluded_edges"] += 1
            results["edges_skipped"] += 1
            continue
        u_dist = node_distances.get(u, float("inf"))
        v_dist = node_distances.get(v, float("inf"))
        if u_dist == float("inf") and v_dist == float("inf"):
            results["unreachable_edges"].append((u, v, key))
            results["edges_skipped"] += 1
            continue
        edge_distance = min(u_dist, v_dist) + 1
        branch_order = f"{prefix}{edge_distance:02d}" if prefix == "B" else f"{prefix}{edge_distance}"
        G[u][v][key]["branch_order"] = branch_order
        results["edges_assigned"] += 1
        results["branch_order_counts"][branch_order] += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Edge (%s, %s, %s): %s (u_dist=%s, v_dist=%s)",
                u,
                v,
                key,
                branch_order,
                u_dist,
                v_dist,
            )
    return results


def assign_hierarchical_branch_orders(
    G: nx.MultiGraph,
    starting_nodes: list[int],
    output_nodes: list[int],
    arteriole_boundary_nodes: list[int],
    venule_boundary_nodes: list[int],
) -> dict:
    """
    Assign branch orders in three stages:
    1) Arteriole edges (Art1, Art2, ...) from starting nodes up to arteriole boundary.
    2) Venule edges (Ven1, Ven2, ...) from output nodes up to venule boundary.
    3) Capillary edges (B01, B02, ...) from arteriole boundary, excluding Art/Ven edges.
    """
    arteriole_boundary_set = set(arteriole_boundary_nodes)
    venule_boundary_set = set(venule_boundary_nodes)

    arteriole_node_distances = _compute_node_distances(
        G,
        starting_nodes,
        stop_nodes=arteriole_boundary_set,
    )
    arteriole_nodes = set(arteriole_node_distances.keys())
    arteriole_edges = {
        _edge_id(u, v, key)
        for u, v, key in G.edges(keys=True)
        if u in arteriole_nodes and v in arteriole_nodes
    }
    arteriole_results = assign_branch_orders(
        G,
        starting_nodes,
        prefix="Art",
        stop_nodes=arteriole_boundary_set,
        included_edges=arteriole_edges,
    )

    venule_node_distances = _compute_node_distances(
        G,
        output_nodes,
        stop_nodes=venule_boundary_set,
    )
    venule_nodes = set(venule_node_distances.keys())
    venule_edges = {
        _edge_id(u, v, key)
        for u, v, key in G.edges(keys=True)
        if u in venule_nodes and v in venule_nodes
    }
    venule_results = assign_branch_orders(
        G,
        output_nodes,
        prefix="Ven",
        stop_nodes=venule_boundary_set,
        excluded_edges=arteriole_edges,
        included_edges=venule_edges,
    )

    capillary_start_nodes = arteriole_boundary_nodes if arteriole_boundary_nodes else starting_nodes
    capillary_excluded_edges = arteriole_edges | venule_edges
    capillary_results = assign_branch_orders(
        G,
        capillary_start_nodes,
        prefix="B",
        excluded_edges=capillary_excluded_edges,
    )

    return {
        "arteriole": arteriole_results,
        "venule": venule_results,
        "capillary": capillary_results,
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "excluded_capillary_edge_count": len(capillary_excluded_edges),
    }
