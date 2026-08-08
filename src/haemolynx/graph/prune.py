"""Prune short terminal stubs from vascular graph."""
import logging
from typing import Tuple, Union

import networkx as nx

from ._helpers import calculate_edge_length

logger = logging.getLogger(__name__)

def prune_vascular_stubs(
    G: Union[nx.Graph, nx.MultiGraph],
    min_stub_length: float = 10.0,
    max_iterations: int = 100,
    debug: bool = False,
    voxel_size: Tuple[float, float, float] = (1, 1, 1),
) -> Union[nx.Graph, nx.MultiGraph]:
    """Iteratively remove short terminal stubs until convergence."""
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
                        logger.debug(
                            f"  Iteration {iteration}: Marking node {node} "
                            f"(stub length: {edge_length:.2f})"
                        )
            except Exception as e:
                if debug:
                    logger.debug(f"  Warning: Could not calculate edge length: {e}")
                nodes_to_remove.append(node)

        G_pruned.remove_nodes_from(nodes_to_remove)
        nodes_after = G_pruned.number_of_nodes()
        removed_this_iteration = nodes_before - nodes_after
        total_removed += removed_this_iteration

        if debug:
            logger.debug(
                f"  Iteration {iteration}: Removed {removed_this_iteration} "
                f"({nodes_after} remaining)"
            )
        if removed_this_iteration == 0:
            if debug:
                logger.debug(f"Convergence reached after {iteration} iterations")
            break

    if debug:
        logger.debug(f"Pruning complete: Total nodes removed: {total_removed}")
    return G_pruned

def remove_edges_for_self_connected_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove edges for nodes that are connected to themselves with no nodes in between."""
    G_pruned = G.copy()
    for node in G_pruned.nodes():
        if node in G_pruned.neighbors(node):
            G_pruned.remove_edge(node, node)
    return G_pruned
