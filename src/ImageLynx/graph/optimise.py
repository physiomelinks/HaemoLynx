"""Optimise graph topology: terminal reconnection with skeleton validation."""
import logging
import heapq

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from ._helpers import add_edge_safe
from .validate import validate_skeleton_connection

logger = logging.getLogger(__name__)


def optimise_graph_topology_fixed(
    G,
    voxel_loops,
    loop_edges,
    skeleton_data=None,
    debug=False,
    reconnect_threshold=3.0,
    use_spatial_index=True,
    remove_degree2_nodes=True,
    consolidation_threshold=2.0,
    improve_junctions=True,
    preserve_multigraph=True,
    validate_reconnections=True,
    aggressive_degree2_cleanup_level=1,
):
    """Reconnect nearby terminals with optional skeleton validation."""
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
                    add_edge_safe(
                        G,
                        src,
                        tgt,
                        weight=max(dist, 1e-6),
                        length=len(voxel_path) * float(np.linalg.norm(vs_arr)) if voxel_path else dist,
                        voxels=phys_path,
                        reconnected=True,
                        validated=True,
                    )
                else:
                    conservative_threshold = min(reconnect_threshold * 0.5, 1.5)
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
