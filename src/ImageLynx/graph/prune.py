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
    self_loops = list(nx.selfloop_edges(G_pruned, keys=True)) if isinstance(G_pruned, nx.MultiGraph) else list(nx.selfloop_edges(G_pruned))
    G_pruned.remove_edges_from(self_loops)
    return G_pruned


def remove_isolated_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove degree-0 nodes from the graph."""
    G_pruned = G.copy()
    isolated_nodes = [n for n in G_pruned.nodes() if G_pruned.degree(n) == 0]
    if isolated_nodes:
        G_pruned.remove_nodes_from(isolated_nodes)
    return G_pruned


def remove_components_without_connected_io(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: list[int],
    output_nodes: list[int],
) -> tuple[Union[nx.Graph, nx.MultiGraph], dict[str, int]]:
    """Keep only connected components containing both start and output nodes.

    Components that do not include at least one node from each boundary set
    are removed. Node IDs not present in ``G`` are ignored.
    """
    start_node_set = {
        int(node_id) for node_id in starting_nodes if int(node_id) in G.nodes
    }
    output_node_set = {
        int(node_id) for node_id in output_nodes if int(node_id) in G.nodes
    }

    keep_nodes: set[int] = set()
    removed_component_count = 0
    removed_node_count = 0

    for component_nodes in nx.connected_components(G):
        component_node_set = {int(node_id) for node_id in component_nodes}
        has_start_node = bool(component_node_set.intersection(start_node_set))
        has_output_node = bool(component_node_set.intersection(output_node_set))
        if has_start_node and has_output_node:
            keep_nodes.update(component_node_set)
        else:
            removed_component_count += 1
            removed_node_count += len(component_node_set)

    if removed_component_count <= 0:
        return G.copy(), {
            "removed_components": 0,
            "removed_nodes": 0,
            "remaining_nodes": int(G.number_of_nodes()),
        }

    G_pruned = G.subgraph(keep_nodes).copy()
    return G_pruned, {
        "removed_components": int(removed_component_count),
        "removed_nodes": int(removed_node_count),
        "remaining_nodes": int(G_pruned.number_of_nodes()),
    }
