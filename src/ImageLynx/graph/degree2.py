"""Degree-2 node removal and edge merging."""
import logging
from typing import List, Tuple, Union, Any

import numpy as np
import networkx as nx

from ._helpers import (
    get_all_edge_data,
    create_merged_edge_attributes,
    voxel_path_overlap_ratio,
    add_edge_safe,
    has_edge_safe,
    remove_edge_safe,
    is_path_curved,
    merge_curved_edges,
    improve_straight_edge_with_skeleton,
    improve_straight_path_with_skeleton,
    should_add_merged_edge,
    calculate_path_length,
    merge_edge_voxels_at_node,
)

logger = logging.getLogger(__name__)


def _compute_skeleton_overlap(
    voxels: List, skeleton_data: np.ndarray
) -> float:
    """Fraction of voxel path coordinates that lie on the skeleton."""
    if not voxels or skeleton_data is None or skeleton_data.size == 0:
        return 0.0
    shape = skeleton_data.shape
    on_skeleton = 0
    total = 0
    for v in voxels:
        coords = tuple(int(round(c)) for c in v)
        if all(0 <= coords[i] < shape[i] for i in range(len(coords))):
            total += 1
            if skeleton_data[coords]:
                on_skeleton += 1
        else:
            total += 1
    return on_skeleton / total if total > 0 else 0.0


def _should_replace_existing_simple_edge(
    G: nx.Graph,
    n1: Any,
    n2: Any,
    merged_voxels: List,
    overlap_threshold: float = 0.9,
) -> bool:
    """Only replace existing edge when voxel overlap is significant."""
    if not G.has_edge(n1, n2):
        return True
    existing = G.get_edge_data(n1, n2) or {}
    existing_voxels = existing.get("voxels", [])
    overlap = voxel_path_overlap_ratio(existing_voxels, merged_voxels)
    return overlap >= overlap_threshold


def safer_simple_remove_all_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    debug: bool = False,
    max_iterations: int = 100,
    max_edge_length_ratio: float = 2.0,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove degree-2 nodes with safety check for shortcut creation."""
    total_removed = 0
    skipped_long_edges = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))

    if debug:
        initial_degree2 = len([n for n in G.nodes() if G.degree[n] == 2])
        logger.info("Starting safer cleanup with %d degree-2 nodes", initial_degree2)

    for iteration in range(max_iterations):
        removed_this_iter = 0
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        if not degree2_nodes:
            break
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            node_pos = G.nodes[node].get("pos", None)
            if not edge1_data_list or not edge2_data_list:
                continue
            if "pos" in G.nodes[n1] and "pos" in G.nodes[n2] and node_pos is not None:
                pos1 = np.array(G.nodes[n1]["pos"])
                pos2 = np.array(G.nodes[n2]["pos"])
                node_pos_arr = np.array(node_pos)
                current_path_length = np.linalg.norm(pos1 - node_pos_arr) + np.linalg.norm(
                    node_pos_arr - pos2
                )
                direct_distance = np.linalg.norm(pos2 - pos1)
                if current_path_length > direct_distance * max_edge_length_ratio:
                    if debug:
                        logger.debug(
                            "Skipped removing node %s: would create shortcut", node
                        )
                    skipped_long_edges += 1
                    continue
            G.remove_node(node)
            if is_multigraph:
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_attrs = create_merged_edge_attributes(
                            edge1_data, edge2_data, node_pos
                        )
                        add_edge_safe(G, n1, n2, **merged_attrs)
            else:
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                merged_attrs = create_merged_edge_attributes(
                    edge1_data, edge2_data, node_pos
                )
                if has_edge_safe(G, n1, n2):
                    if _should_replace_existing_simple_edge(
                        G, n1, n2, merged_attrs.get("voxels", [])
                    ):
                        remove_edge_safe(G, n1, n2)
                    else:
                        if debug:
                            logger.debug(
                                "Preserved existing edge %s-%s (low overlap with merged path)",
                                n1,
                                n2,
                            )
                        continue
                add_edge_safe(G, n1, n2, **merged_attrs)
            removed_this_iter += 1
            total_removed += 1
        if debug and removed_this_iter > 0:
            remaining = len([n for n in G.nodes() if G.degree[n] == 2])
            logger.debug(
                "Iteration %d: removed %d, %d remain",
                iteration + 1,
                removed_this_iter,
                remaining,
            )
        if removed_this_iter == 0:
            break

    if debug:
        final_degree2 = len([n for n in G.nodes() if G.degree[n] == 2])
        logger.info(
            "Safer cleanup: removed %d, skipped %d long edges, final degree-2: %d",
            total_removed,
            skipped_long_edges,
            final_degree2,
        )
    return G


def trivial_remove_all_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    debug: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove all degree-2 nodes by merging adjacent edges."""
    total_removed = 0
    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    for iteration in range(100):
        degree2_nodes = [n for n in G.nodes() if G.degree[n] == 2]
        if not degree2_nodes:
            break
        removed_this_iter = 0
        for node in degree2_nodes:
            if not G.has_node(node) or G.degree[node] != 2:
                continue
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue
            node_pos = G.nodes[node].get("pos", None)
            edge1_data_list = get_all_edge_data(G, node, n1)
            edge2_data_list = get_all_edge_data(G, node, n2)
            if not edge1_data_list or not edge2_data_list:
                continue
            G.remove_node(node)
            if is_multigraph:
                for edge1_data in edge1_data_list:
                    for edge2_data in edge2_data_list:
                        merged_edge = create_trivial_merged_edge(
                            edge1_data, edge2_data, node_pos
                        )
                        G.add_edge(n1, n2, **merged_edge)
            else:
                edge1_data = edge1_data_list[0]
                edge2_data = edge2_data_list[0]
                merged_edge = create_trivial_merged_edge(
                    edge1_data, edge2_data, node_pos
                )
                if G.has_edge(n1, n2):
                    if _should_replace_existing_simple_edge(
                        G, n1, n2, merged_edge.get("voxels", [])
                    ):
                        G.remove_edge(n1, n2)
                    else:
                        if debug:
                            logger.debug(
                                "Preserved existing edge %s-%s (low overlap with trivial merged path)",
                                n1,
                                n2,
                            )
                        continue
                G.add_edge(n1, n2, **merged_edge)
            removed_this_iter += 1
            total_removed += 1
            if debug:
                logger.debug("Removed degree-2 node %s, connected %s-%s", node, n1, n2)
        if removed_this_iter == 0:
            break
    if debug:
        logger.info("Trivial removal: total removed %d", total_removed)
    return G


def create_trivial_merged_edge(
    edge1_data: dict, edge2_data: dict, removed_node_pos: Any
) -> dict:
    """Create merged edge with exact topology preservation."""
    voxels1 = edge1_data.get("voxels", [])
    voxels2 = edge2_data.get("voxels", [])
    merged_voxels = merge_edge_voxels_at_node(voxels1, voxels2, removed_node_pos)
    merged_attributes = {
        "weight": edge1_data.get("weight", 0) + edge2_data.get("weight", 0),
        "length": edge1_data.get("length", 0) + edge2_data.get("length", 0),
        "voxels": merged_voxels,
        "merged": True,
        "trivial_merge": True,
        "removed_node_pos": removed_node_pos,
    }
    for key, value in edge1_data.items():
        if key not in merged_attributes:
            merged_attributes[key] = value
    return merged_attributes


def merge_edges_with_topology_improvement(
    voxels1: List,
    voxels2: List,
    pos1: np.ndarray,
    node_pos: np.ndarray,
    pos2: np.ndarray,
    skeleton_data,
    debug: bool = False,
    voxel_size: tuple = (1.0, 1.0, 1.0),
) -> List:
    """Merge two edges while improving straight segments using skeleton."""
    if skeleton_data is None or skeleton_data.size == 0:
        return merge_curved_edges(voxels1, voxels2, node_pos, debug)
    is_curved1 = is_path_curved(voxels1)
    is_curved2 = is_path_curved(voxels2)
    if is_curved1 and is_curved2:
        return merge_curved_edges(voxels1, voxels2, node_pos, debug)
    if is_curved1 and not is_curved2:
        improved_voxels2 = improve_straight_edge_with_skeleton(
            node_pos, pos2, skeleton_data, debug, voxel_size=voxel_size
        )
        if improved_voxels2:
            return merge_curved_edges(voxels1, improved_voxels2, node_pos, debug)
        return merge_curved_edges(voxels1, voxels2, node_pos, debug)
    if not is_curved1 and is_curved2:
        improved_voxels1 = improve_straight_edge_with_skeleton(
            pos1, node_pos, skeleton_data, debug, voxel_size=voxel_size
        )
        if improved_voxels1:
            return merge_curved_edges(improved_voxels1, voxels2, node_pos, debug)
        return merge_curved_edges(voxels1, voxels2, node_pos, debug)
    improved_full_path = improve_straight_path_with_skeleton(
        pos1, pos2, skeleton_data, debug, voxel_size=voxel_size
    )
    if improved_full_path:
        return improved_full_path
    return merge_curved_edges(voxels1, voxels2, node_pos, debug)


def smart_multigraph_degree2_removal(
    G: nx.MultiGraph,
    skeleton_data: np.ndarray = None,
    max_degree: int = 4,
    debug: bool = False,
    max_iterations: int = 500,
) -> nx.MultiGraph:
    """Smart degree-2 removal for MultiGraphs with topology improvement."""
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("This function is designed for MultiGraphs")

    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))

    total_removed = 0
    for iteration in range(max_iterations):
        removed_this_iter = 0

        for node in list(G.nodes()):
            if not G.has_node(node) or G.degree[node] != 2:
                continue

            edges = list(G.edges(node, keys=True, data=True))
            if len(edges) != 2:
                continue

            _, n1, k1, d1 = edges[0]
            _, n2, k2, d2 = edges[1]

            if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
                continue

            node_pos = G.nodes[node].get("pos", None)
            n1_pos = G.nodes[n1].get("pos", None)
            n2_pos = G.nodes[n2].get("pos", None)
            if node_pos is None or n1_pos is None or n2_pos is None:
                continue

            voxels1 = d1.get("voxels", [])
            voxels2 = d2.get("voxels", [])

            G.remove_node(node)

            merged_voxels = merge_edges_with_topology_improvement(
                voxels1,
                voxels2,
                np.array(n1_pos),
                np.array(node_pos),
                np.array(n2_pos),
                skeleton_data,
                debug,
                voxel_size=vs,
            )
            merged_attrs = {
                "weight": d1.get("weight", 0) + d2.get("weight", 0),
                "length": calculate_path_length(merged_voxels),
                "voxels": merged_voxels,
                "merged": True,
                "original_edges": 2,
            }

            should_add, replace_key = should_add_merged_edge(
                G, n1, n2, merged_voxels, merged_attrs, debug
            )
            if should_add:
                if replace_key is not None:
                    G.remove_edge(n1, n2, key=replace_key)
                G.add_edge(n1, n2, **merged_attrs)

            removed_this_iter += 1
            total_removed += 1

        if removed_this_iter == 0:
            break

    if debug:
        logger.info("Smart removal: %d removed", total_removed)
    return G