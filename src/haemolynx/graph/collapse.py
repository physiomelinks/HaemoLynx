"""Collapse spatially co-located node clusters into single representative nodes.

On larger vessels the skeletonisation can produce many short branches that
terminate in close proximity, creating dense bundles of nodes.  This module
detects those bundles via spatial proximity and merges every cluster into one
representative node, rewiring edges and preserving voxel paths.
"""
import logging
from typing import Union

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _pick_representative(G: Union[nx.Graph, nx.MultiGraph], cluster: list) -> int:
    """Pick the best representative for a cluster of nodes.

    Preference order:
    1. Highest degree (most connected — most likely a real junction)
    2. Ties broken by lowest node id for determinism
    """
    return max(cluster, key=lambda n: (G.degree(n), -n))


def collapse_node_clusters(
    G: Union[nx.Graph, nx.MultiGraph],
    distance_threshold: float = 5.0,
    debug: bool = False,
    max_iterations: int = 10,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Collapse clusters of nodes that are within *distance_threshold* of each
    other into single representative nodes.

    Parameters
    ----------
    G : nx.Graph or nx.MultiGraph
        The vascular graph (modified in-place on a copy).
    distance_threshold : float
        Maximum Euclidean distance between two node positions for them to be
        considered part of the same spatial cluster.
    debug : bool
        Emit detailed log messages.
    max_iterations : int
        Safety cap — clustering is repeated until no more merges are found or
        this limit is reached (each pass may expose new clusters).

    Returns
    -------
    Same type as *G*, with node clusters collapsed.
    """
    G = G.copy()
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    total_merged = 0

    for iteration in range(max_iterations):
        nodes_with_pos = [
            (n, np.array(G.nodes[n]["pos"], dtype=float))
            for n in G.nodes()
            if "pos" in G.nodes[n]
        ]
        if len(nodes_with_pos) < 2:
            break

        node_ids = [n for n, _ in nodes_with_pos]
        coords = np.array([p for _, p in nodes_with_pos])

        tree = cKDTree(coords)
        pairs = tree.query_pairs(distance_threshold)
        if not pairs:
            break

        # Build an undirected proximity graph and find connected components to
        # get the clusters.
        proximity = nx.Graph()
        proximity.add_nodes_from(range(len(node_ids)))
        for i, j in pairs:
            proximity.add_edge(i, j)

        merged_this_iter = 0
        for component in nx.connected_components(proximity):
            if len(component) < 2:
                continue

            cluster = [node_ids[i] for i in component]
            # Ensure all cluster nodes still exist (earlier merge might have
            # removed some).
            cluster = [n for n in cluster if G.has_node(n)]
            if len(cluster) < 2:
                continue

            rep = _pick_representative(G, cluster)
            others = [n for n in cluster if n != rep]

            if debug:
                logger.info(
                    "Collapsing cluster of %d nodes %s -> representative %s",
                    len(cluster),
                    cluster,
                    rep,
                )

            # Update the representative position to the centroid of the
            # cluster so it sits at a geometrically central location.
            cluster_positions = np.array(
                [G.nodes[n]["pos"] for n in cluster if "pos" in G.nodes[n]]
            )
            G.nodes[rep]["pos"] = cluster_positions.mean(axis=0)

            for other in others:
                if not G.has_node(other):
                    continue
                _rewire_edges(G, other, rep, is_multi, debug)
                G.remove_node(other)
                merged_this_iter += 1

        total_merged += merged_this_iter
        if debug:
            logger.info(
                "Iteration %d: merged %d nodes (%d total)",
                iteration + 1,
                merged_this_iter,
                total_merged,
            )
        if merged_this_iter == 0:
            break

    # Final cleanup: remove self-loops that may have been created.
    self_loops = list(nx.selfloop_edges(G, keys=True)) if is_multi else [
        (u, v) for u, v in nx.selfloop_edges(G)
    ]
    if self_loops:
        if is_multi:
            for u, v, k in self_loops:
                G.remove_edge(u, v, key=k)
        else:
            G.remove_edges_from(self_loops)
        if debug:
            logger.info("Removed %d self-loops after collapsing", len(self_loops))

    if debug or total_merged > 0:
        logger.info(
            "collapse_node_clusters: merged %d nodes, graph now has %d nodes / %d edges",
            total_merged,
            G.number_of_nodes(),
            G.number_of_edges(),
        )
    return G


def _patch_voxel_endpoint(data: dict, old_pos: np.ndarray, new_pos: np.ndarray) -> dict:
    """Return a shallow copy of *data* with the voxel path endpoint closest to
    *old_pos* replaced by *new_pos* (as integer tuple)."""
    data = dict(data)
    voxels = data.get("voxels")
    if not voxels or len(voxels) < 2:
        return data

    new_voxel = tuple(np.round(new_pos).astype(int))
    old_voxel = tuple(np.round(old_pos).astype(int))

    voxels = list(voxels)
    start_key = tuple(np.round(np.asarray(voxels[0])).astype(int))
    end_key = tuple(np.round(np.asarray(voxels[-1])).astype(int))

    if start_key == old_voxel or np.linalg.norm(np.asarray(voxels[0], dtype=float) - old_pos) < \
            np.linalg.norm(np.asarray(voxels[-1], dtype=float) - old_pos):
        voxels[0] = new_voxel
    else:
        voxels[-1] = new_voxel

    data["voxels"] = voxels
    return data


def _rewire_edges(
    G: Union[nx.Graph, nx.MultiGraph],
    old_node: int,
    new_node: int,
    is_multi: bool,
    debug: bool,
) -> None:
    """Move every edge incident to *old_node* so it connects to *new_node*
    instead, preserving edge attributes and patching voxel endpoints."""
    old_pos = np.asarray(G.nodes[old_node].get("pos", [0, 0, 0]), dtype=float)
    new_pos = np.asarray(G.nodes[new_node].get("pos", [0, 0, 0]), dtype=float)

    if is_multi:
        edges = list(G.edges(old_node, data=True, keys=True))
        for u, v, key, data in edges:
            neighbor = v if u == old_node else u
            if neighbor == new_node:
                continue
            patched = _patch_voxel_endpoint(data, old_pos, new_pos)
            G.add_edge(new_node, neighbor, **patched)
    else:
        edges = list(G.edges(old_node, data=True))
        for u, v, data in edges:
            neighbor = v if u == old_node else u
            if neighbor == new_node:
                continue
            patched = _patch_voxel_endpoint(data, old_pos, new_pos)
            if not G.has_edge(new_node, neighbor):
                G.add_edge(new_node, neighbor, **patched)
            else:
                existing = G[new_node][neighbor]
                existing_len = existing.get("length", float("inf"))
                new_len = patched.get("length", float("inf"))
                if new_len < existing_len:
                    G.remove_edge(new_node, neighbor)
                    G.add_edge(new_node, neighbor, **patched)
