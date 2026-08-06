"""Optimise graph topology: terminal reconnection with skeleton validation."""
import logging
import heapq

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from ._helpers import (
    CONSERVATIVE_RECONNECT_CAP_UM,
    RECONNECT_THRESHOLD_UM,
    add_edge_safe,
    calculate_path_length,
)
from .validate import validate_skeleton_connection

logger = logging.getLogger(__name__)


def _physical_path_length(points) -> float:
    """Compute 3D polyline length in physical units."""
    if not points or len(points) < 2:
        return 0.0
    return float(calculate_path_length(points))


def _find_reconnection_candidates_batch(
    src_batch, G_node_attrs, G_edge_presence,
    target_nodes, target_coords, tree, reconnect_threshold
):
    """Find reconnection pairs for a batch of source nodes using a KD-tree."""
    candidates = []
    for src in src_batch:
        if src not in G_node_attrs:
            continue
        src_pos = np.array(G_node_attrs[src]["pos"], dtype=float)
        idxs = tree.query_ball_point(src_pos, reconnect_threshold)
        for idx in idxs:
            tgt = target_nodes[idx]
            if tgt == src:
                continue
            # Note: G_edge_presence is a set of sorted node pairs for fast check
            if tuple(sorted([src, tgt])) in G_edge_presence:
                continue
            dist = float(np.linalg.norm(src_pos - target_coords[idx]))
            if dist <= reconnect_threshold:
                candidates.append((dist, src, tgt))
    return candidates


def optimise_graph_topology_fixed(
    G,
    voxel_loops,
    loop_edges,
    skeleton_data=None,
    debug=False,
    reconnect_threshold=RECONNECT_THRESHOLD_UM,
    use_spatial_index=True,
    remove_degree2_nodes=True,
    # consolidation_threshold=2.0 removed: declared here and consumed nowhere, in this
    # implementation or the legacy one it was ported from. It was not a miscalibrated
    # threshold that 2705b38 re-denominated from voxels to microns, as first assumed - it
    # never had an effect in either unit, which is also why Stage 11 step 3 measured as an
    # exact no-op.
    improve_junctions=True,
    preserve_multigraph=True,
    validate_reconnections=True,
    aggressive_degree2_cleanup_level=1,
):
    """Reconnect nearby terminals with optional skeleton validation.

    reconnect_threshold is in MICRONS, not voxels: it is compared against node "pos",
    which is stored in physical units. See RECONNECT_THRESHOLD_UM in _helpers.
    """
    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))

    if reconnect_threshold and reconnect_threshold > 0:
        valid_nodes = [n for n in G.nodes() if "pos" in G.nodes[n]]
        terminals = [n for n in valid_nodes if G.degree[n] == 1]

        if len(terminals) > 1:
            if use_spatial_index and len(terminals) > 10:
                terminal_coords = np.array([G.nodes[n]["pos"] for n in terminals])
                tree = cKDTree(terminal_coords)
                pairs_indices = tree.query_pairs(reconnect_threshold)
                pairs = []
                for i, j in pairs_indices:
                    src, tgt = terminals[i], terminals[j]
                    edge_norm = tuple(sorted([src, tgt]))
                    if (
                        G.has_edge(src, tgt)
                        or edge_norm in loop_edges
                        or G.degree[src] > 1
                        or G.degree[tgt] > 1
                    ):
                        continue
                    dist = np.linalg.norm(terminal_coords[i] - terminal_coords[j])
                    pairs.append((dist, src, tgt))
            else:
                pairs = []
                for i, src in enumerate(terminals):
                    if "pos" not in G.nodes[src]:
                        continue
                    for j in range(i + 1, len(terminals)):
                        tgt = terminals[j]
                        if "pos" not in G.nodes[tgt]:
                            continue
                        edge_norm = tuple(sorted([src, tgt]))
                        if (
                            edge_norm in loop_edges
                            or G.degree[src] > 1
                            or G.degree[tgt] > 1
                        ):
                            continue
                        src_pos = np.array(G.nodes[src]["pos"])
                        tgt_pos = np.array(G.nodes[tgt]["pos"])
                        dist = np.linalg.norm(src_pos - tgt_pos)
                        if dist <= reconnect_threshold:
                            pairs.append((dist, src, tgt))

            heapq.heapify(pairs)
            reconnected = 0
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                if (
                    not G.has_node(src)
                    or not G.has_node(tgt)
                    or "pos" not in G.nodes[src]
                    or "pos" not in G.nodes[tgt]
                    or G.has_edge(src, tgt)
                    or G.degree[src] > 1
                    or G.degree[tgt] > 1
                ):
                    continue
                src_pos = np.array(G.nodes[src]["pos"])
                tgt_pos = np.array(G.nodes[tgt]["pos"])

                if validate_reconnections and skeleton_data is not None:
                    connection_valid, voxel_path = validate_skeleton_connection(
                        skeleton_data, src_pos, tgt_pos, max_gap=reconnect_threshold,
                        voxel_size=vs,
                    )
                    if not connection_valid:
                        if debug:
                            logger.debug("Skipped reconnection %s-%s: no skeleton path", src, tgt)
                        continue
                    vs_arr = np.asarray(vs, dtype=float)
                    if voxel_path:
                        phys_path = [(np.array(p, dtype=float) * vs_arr).tolist() for p in voxel_path]
                    else:
                        phys_path = [src_pos.tolist(), tgt_pos.tolist()]
                    path_length = _physical_path_length(phys_path)
                    add_edge_safe(
                        G,
                        src,
                        tgt,
                        weight=max(dist, 1e-6),
                        length=path_length if path_length > 0 else dist,
                        voxels=phys_path,
                        reconnected=True,
                        validated=True,
                    )
                else:
                    conservative_threshold = min(reconnect_threshold * 0.5, CONSERVATIVE_RECONNECT_CAP_UM)
                    if dist > conservative_threshold:
                        if debug:
                            logger.debug(
                                "Skipped reconnection %s-%s: dist %.2f > threshold %.2f",
                                src,
                                tgt,
                                dist,
                                conservative_threshold,
                            )
                        continue
                    add_edge_safe(
                        G,
                        src,
                        tgt,
                        weight=max(dist, 1e-6),
                        length=dist,
                        voxels=[src_pos.tolist(), tgt_pos.tolist()],
                        reconnected=True,
                        conservative=True,
                    )
                reconnected += 1
                if debug:
                    logger.debug("Reconnected %s-%s, d=%.2f", src, tgt, dist)
            if debug and reconnected > 0:
                logger.info("Reconnected %d terminal pairs", reconnected)

    return G, voxel_loops


def reconnect_orphan_and_dangling_nodes(
    G: nx.MultiGraph,
    skeleton_data=None,
    reconnect_threshold: float = 3.0,
    include_degree1: bool = True,
    max_new_edges_per_node: int = 1,
    validate_reconnections: bool = True,
    debug: bool = False,
) -> nx.MultiGraph:
    """Reconnect degree-0/degree-1 nodes to nearby nodes via skeleton path."""
    if reconnect_threshold <= 0:
        return G
    if max_new_edges_per_node < 1:
        return G
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("This function is designed for MultiGraphs")

    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))
    vs_arr = np.asarray(vs, dtype=float)

    valid_nodes = [n for n in G.nodes if "pos" in G.nodes[n]]
    if len(valid_nodes) < 2:
        return G

    target_nodes = []
    for node in valid_nodes:
        degree = G.degree[node]
        if degree >= 1:
            target_nodes.append(node)
    if len(target_nodes) < 1:
        return G

    source_nodes = []
    for node in valid_nodes:
        degree = G.degree[node]
        if degree == 0 or (include_degree1 and degree == 1):
            source_nodes.append(node)
    if not source_nodes:
        return G

    target_coords = np.array([G.nodes[n]["pos"] for n in target_nodes], dtype=float)
    tree = cKDTree(target_coords)

    candidate_pairs = []
    for src in source_nodes:
        src_pos = np.array(G.nodes[src]["pos"], dtype=float)
        idxs = tree.query_ball_point(src_pos, reconnect_threshold)
        for idx in idxs:
            tgt = target_nodes[idx]
            if tgt == src:
                continue
            if G.has_edge(src, tgt):
                continue
            dist = float(np.linalg.norm(src_pos - np.array(G.nodes[tgt]["pos"], dtype=float)))
            if dist <= reconnect_threshold:
                candidate_pairs.append((dist, src, tgt))

    if not candidate_pairs:
        return G

    heapq.heapify(candidate_pairs)
    added_edges_per_node = {n: 0 for n in source_nodes}
    reconnect_count = 0

    while candidate_pairs:
        dist, src, tgt = heapq.heappop(candidate_pairs)
        if not G.has_node(src) or not G.has_node(tgt):
            continue
        if "pos" not in G.nodes[src] or "pos" not in G.nodes[tgt]:
            continue
        if G.has_edge(src, tgt):
            continue
        if added_edges_per_node.get(src, 0) >= max_new_edges_per_node:
            continue
        if G.degree[src] > 1 and include_degree1:
            continue

        src_pos = np.array(G.nodes[src]["pos"], dtype=float)
        tgt_pos = np.array(G.nodes[tgt]["pos"], dtype=float)
        voxel_path = None

        if validate_reconnections and skeleton_data is not None:
            connection_valid, voxel_path = validate_skeleton_connection(
                skeleton_data,
                src_pos,
                tgt_pos,
                max_gap=reconnect_threshold,
                voxel_size=vs,
            )
            if not connection_valid:
                continue
            if voxel_path:
                phys_path = [(np.array(p, dtype=float) * vs_arr).tolist() for p in voxel_path]
            else:
                phys_path = [src_pos.tolist(), tgt_pos.tolist()]
        else:
            phys_path = [src_pos.tolist(), tgt_pos.tolist()]

        length = _physical_path_length(phys_path)
        if length <= 0:
            length = float(np.linalg.norm(tgt_pos - src_pos))
        add_edge_safe(
            G,
            src,
            tgt,
            weight=max(length, 1e-6),
            length=length,
            voxels=phys_path,
            reconnected=True,
            orphan_reconnect=True,
            validated=bool(validate_reconnections and skeleton_data is not None),
        )
        added_edges_per_node[src] = added_edges_per_node.get(src, 0) + 1
        reconnect_count += 1

    if debug and reconnect_count > 0:
        logger.info("Reconnected %d orphan/dangling node edge(s)", reconnect_count)
    return G
