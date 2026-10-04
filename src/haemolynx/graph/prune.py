"""Prune short terminal stubs from vascular graph."""
import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from ._helpers import calculate_edge_length, densify_polyline, edge_sample_points

logger = logging.getLogger(__name__)

#: How far back from its tip a stub's outward direction is read.
STUB_TANGENT_LENGTH_UM = 3.0


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
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mask_continues_at: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
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

    The mask can say more about a stub (each read only when given, so
    without them nothing changes):

    - *inside_lumen* (``points_um -> bool``): a stub less than half of which
      lies in the mask is pruned whatever its length -- not a vessel there.
    - *radius_at*: a stub whose tip lies within the parent's radius + 1 um of
      the junction it hangs from is pruned whatever its length -- a Lee spur
      inside the parent's own lumen. Only with *inside_lumen* too.
    - *mask_continues_at* ``(tip_um, outward_direction) -> bool``: where the
      mask runs on past the tip, the tip is not a vessel end but a vessel
      the network lost track of, so the threshold is the larger of the
      radius rule and *min_stub_length*. A tip at a dead end of the mask (a
      blind sprout) keeps the radius rule.

    Stubs at an image face are kept as before; the two whatever-its-length
    rules apply there too, since neither is a vessel the image cut through.
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

    def stub_path(node: Any, junction: Any) -> Optional[np.ndarray]:
        if "pos" not in G_pruned.nodes[node] or "pos" not in G_pruned.nodes[junction]:
            return None
        node_pos = {n: np.asarray(G_pruned.nodes[n]["pos"], dtype=float) for n in (node, junction)}
        data = G_pruned[junction][node]
        if G_pruned.is_multigraph():
            data = next(iter(data.values()))
        return densify_polyline(edge_sample_points(junction, node, data, node_pos), max_step_um=0.5)

    def not_a_vessel(path: np.ndarray, junction: Any) -> bool:
        if 2 * int(np.count_nonzero(inside_lumen(path))) < len(path):
            return True
        if radius_at is None:
            return False
        junction_pos = path[0]
        parent_radius = float(radius_at(junction_pos))
        return parent_radius > 0 and float(np.linalg.norm(path[-1] - junction_pos)) < parent_radius + 1.0

    def mask_continues(path: np.ndarray) -> bool:
        reverse = path[::-1]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(reverse, axis=0), axis=1))])
        tip = reverse[0]
        outward = tip - reverse[min(int(np.searchsorted(arc, STUB_TANGENT_LENGTH_UM)), len(reverse) - 1)]
        norm = float(np.linalg.norm(outward))
        return norm > 0 and bool(mask_continues_at(tip, outward / norm))

    total_removed = 0
    not_vessels = 0
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
                path = (
                    stub_path(node, neighbor)
                    if inside_lumen is not None or mask_continues_at is not None
                    else None
                )
                if path is not None and len(path) > 1:
                    if inside_lumen is not None and not_a_vessel(path, neighbor):
                        nodes_to_remove.append(node)
                        not_vessels += 1
                        continue
                    if mask_continues_at is not None and mask_continues(path):
                        threshold = max(threshold, float(min_stub_length))
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
    if inside_lumen is not None:
        logger.info(
            "Pruning: %d of those stubs were mostly outside the mask or inside their "
            "parent's lumen, and went whatever their length",
            not_vessels,
        )
    return G_pruned


#: Edge attribute the flow solve writes: False on a vessel no inlet-to-outlet
#: path runs along -- in a branch or tree without both an inlet and an
#: outlet, or a dead end hanging off one that has both -- since no pressure
#: difference drives a flow through it, so its flow and pressures are not a
#: result; True everywhere else. Exactly the vessels
#: :func:`haemolynx.graph.remove_vessels_off_inlet_outlet_paths` would
#: remove are False.
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
    unsolved (on no path from a start to an output node, by
    :func:`haemolynx.graph.inlet_to_outlet_vessels`)."""
    from .connectivity import inlet_to_outlet_vessels

    multi = G if G.is_multigraph() else nx.MultiGraph(G)
    through = {
        (frozenset((u, v)), k)
        for u, v, k in inlet_to_outlet_vessels(multi, list(starting_nodes), list(output_nodes))
    }
    edges = (
        G.edges(keys=True, data=True)
        if G.is_multigraph()
        else ((u, v, 0, data) for u, v, data in G.edges(data=True))
    )
    unsolved = 0
    for u, v, k, data in edges:
        solved = (frozenset((u, v)), k) in through
        data[FLOW_SOLVED] = solved
        unsolved += not solved
    return unsolved


def edges_with_connected_io(
    G: nx.MultiGraph,
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
) -> list[tuple[int, int, int]]:
    """The edges ``(u, v, key)`` of *G* in a component holding both a start
    and an output node: the ones :func:`remove_components_without_connected_io`
    keeps."""
    kept_nodes: set[int] = set()
    for component_node_set, has_io in _components_by_io(G, starting_nodes, output_nodes):
        if has_io:
            kept_nodes |= component_node_set
    # One end tells: both ends of an edge share a component.
    return [(u, v, key) for u, v, key in G.edges(keys=True) if int(u) in kept_nodes]


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
