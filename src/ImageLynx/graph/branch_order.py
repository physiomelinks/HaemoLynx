"""Assign branch orders to graph edges via BFS from starting nodes."""
import logging
from collections import defaultdict, deque

import networkx as nx

logger = logging.getLogger(__name__)


def assign_branch_orders(
    G: nx.MultiGraph, starting_nodes: list
) -> dict:
    """Assign branch order to each edge based on BFS distance from starting nodes."""
    edge_distances = {}
    node_distances = {}
    queue = deque()
    for start_node in starting_nodes:
        if start_node in G.nodes():
            queue.append((start_node, 0))
            node_distances[start_node] = 0
        else:
            logger.warning("Starting node %s not found in graph", start_node)

    while queue:
        current_node, distance = queue.popleft()
        if (
            current_node in node_distances
            and node_distances[current_node] < distance
        ):
            continue
        for neighbor in G.neighbors(current_node):
            new_distance = distance + 1
            if (
                neighbor not in node_distances
                or node_distances[neighbor] > new_distance
            ):
                node_distances[neighbor] = new_distance
                queue.append((neighbor, new_distance))

    results = {
        "edges_assigned": 0,
        "edges_skipped": 0,
        "branch_order_counts": defaultdict(int),
        "unreachable_edges": [],
    }

    for u, v, key, data in G.edges(keys=True, data=True):
        u_dist = node_distances.get(u, float("inf"))
        v_dist = node_distances.get(v, float("inf"))
        if u_dist == float("inf") and v_dist == float("inf"):
            results["unreachable_edges"].append((u, v, key))
            results["edges_skipped"] += 1
            continue
        edge_distance = min(u_dist, v_dist) + 1
        branch_order = f"B{edge_distance:02d}"
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
