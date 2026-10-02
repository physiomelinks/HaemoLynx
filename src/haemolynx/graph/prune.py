"""Prune short terminal stubs from vascular graph."""
import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from ._helpers import calculate_edge_length

logger = logging.getLogger(__name__)


def prune_vascular_stubs(
    G: Union[nx.Graph, nx.MultiGraph],
    min_stub_length: float = 10.0,
    max_iterations: int = 100,
    debug: bool = False,
    voxel_size: Tuple[float, float, float] = (1, 1, 1),
    *,
    radius_at: Optional[Callable[[np.ndarray], float]] = None,
    radius_multiple: float = 0.0,
    image_extent_um: Optional[Sequence[float]] = None,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Iteratively remove short terminal stubs until convergence.

    A stub is a terminal node's edge; it is removed when shorter than its
    threshold. With *radius_at* (the vessel radius, in microns, at a
    position) and a positive *radius_multiple*, the threshold is that many
    radii of the vessel the stub leaves -- read at the junction it hangs from
    -- so a skeleton spur on a wide vessel, as long as that vessel is thick,
    goes, while a real short capillary end survives; one fixed length could
    only ever do one of the two. *min_stub_length* is the threshold wherever
    no radius can be read.

    With *image_extent_um* (the image's ``(z, y, x)`` extent in microns, node
    positions running from 0 to it), a terminal no further from an image face
    than its own threshold is kept: that is a vessel the image cut through --
    exactly the open ends inlets and outlets are chosen from -- not a spur.
    """
    if min_stub_length < 0:
        raise ValueError("min_stub_length must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must be a 3-tuple")
    if radius_multiple < 0:
        raise ValueError("radius_multiple must be non-negative")

    G_pruned = G.copy()
    if G_pruned.number_of_nodes() == 0:
        return G_pruned

    extent = None if image_extent_um is None else np.asarray(image_extent_um, dtype=float)

    def threshold_for(junction: Any) -> float:
        if radius_at is not None and radius_multiple > 0 and "pos" in G_pruned.nodes[junction]:
            radius = float(radius_at(np.asarray(G_pruned.nodes[junction]["pos"], dtype=float)))
            if radius > 0:
                return radius_multiple * radius
        return float(min_stub_length)

    def at_an_image_face(node: Any, threshold: float) -> bool:
        if extent is None or "pos" not in G_pruned.nodes[node]:
            return False
        position = np.asarray(G_pruned.nodes[node]["pos"], dtype=float)
        return float(np.min(np.minimum(position, extent - position))) <= threshold

    total_removed = 0
    protected: set = set()
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
                threshold = threshold_for(neighbor)
                if edge_length < threshold and at_an_image_face(node, threshold):
                    protected.add(node)
                    continue
                if edge_length < threshold:
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

    kept_at_faces = sum(1 for node in protected if node in G_pruned)
    logger.info(
        "Pruning complete: removed %d terminal stubs (%s), kept %d short one(s) "
        "at an image face, graph now has %d nodes / %d edges",
        total_removed,
        (
            f"threshold {radius_multiple:g} x the parent vessel's radius"
            if radius_at is not None and radius_multiple > 0
            else f"threshold {float(min_stub_length):g} um"
        ),
        kept_at_faces,
        G_pruned.number_of_nodes(),
        G_pruned.number_of_edges(),
    )
    return G_pruned


#: Edge attribute the flow solve writes: False on a vessel in a branch or tree
#: without both an inlet and an outlet -- no pressure difference drives a
#: flow through it, so its flow and pressures are not a result -- and True
#: everywhere else. Exactly the vessels
#: :func:`remove_components_without_connected_io` would remove are False.
FLOW_SOLVED = "flow_solved"


def _components_by_io(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
):
    """Each connected component's nodes, and whether it holds both a start
    and an output node. Node IDs not present in ``G`` are ignored."""
    start_node_set = {
        int(node_id) for node_id in starting_nodes if int(node_id) in G.nodes
    }
    output_node_set = {
        int(node_id) for node_id in output_nodes if int(node_id) in G.nodes
    }
    for component_nodes in nx.connected_components(G):
        component_node_set = {int(node_id) for node_id in component_nodes}
        yield component_node_set, bool(
            component_node_set & start_node_set and component_node_set & output_node_set
        )


def mark_flow_solved_edges(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
) -> int:
    """Write :data:`FLOW_SOLVED` on every edge of *G*; return how many are
    unsolved (in a component without both a start and an output node)."""
    unsolved_nodes: set[int] = set()
    for component_node_set, has_io in _components_by_io(G, starting_nodes, output_nodes):
        if not has_io:
            unsolved_nodes |= component_node_set
    unsolved = 0
    for u, _v, data in G.edges(data=True):
        # One end tells: both ends of an edge share a component.
        solved = int(u) not in unsolved_nodes
        data[FLOW_SOLVED] = solved
        unsolved += not solved
    return unsolved


def remove_components_without_connected_io(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: list[int],
    output_nodes: list[int],
) -> tuple[Union[nx.Graph, nx.MultiGraph], dict[str, int]]:
    """Keep only connected components containing both start and output nodes.

    Components that do not include at least one node from each boundary set
    are removed. Node IDs not present in ``G`` are ignored.
    """
    keep_nodes: set[int] = set()
    removed_component_count = 0
    removed_node_count = 0

    for component_node_set, has_io in _components_by_io(G, starting_nodes, output_nodes):
        if has_io:
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


def remove_edges_for_self_connected_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove edges for nodes that are connected to themselves with no nodes in between."""
    G_pruned = G.copy()
    for node in G_pruned.nodes():
        if node in G_pruned.neighbors(node):
            G_pruned.remove_edge(node, node)
    return G_pruned
