"""Utilities for boundary-node fallback selection logic."""
from __future__ import annotations

import numpy as np
import networkx as nx


def select_nodes_at_hop_distance(
    G: nx.Graph,
    source_nodes: list[int],
    hop_distance: int,
    *,
    exclude_nodes: set[int] | None = None,
) -> list[int]:
    """Collect unique nodes exactly ``hop_distance`` edges from any source node."""
    hop = int(hop_distance)
    if hop < 1:
        raise ValueError(f"hop_distance must be >= 1, got {hop_distance}.")

    excluded = set() if exclude_nodes is None else set(exclude_nodes)
    selected: set[int] = set()
    for src in source_nodes:
        if src not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, src, cutoff=hop)
        for node_id, distance in lengths.items():
            if distance == hop and node_id not in excluded:
                selected.add(int(node_id))
    return sorted(selected)


def seed_edges_have_full_mask_coverage(
    G: nx.MultiGraph,
    seed_nodes: list[int],
    mask: np.ndarray,
) -> tuple[bool, int, int]:
    """Return whether all seed-adjacent edges overlap the given mask.

    Returns
    -------
    (all_covered, uncovered_edge_count, total_seed_edge_count)
    """
    total_edges = 0
    uncovered_edges = 0
    z_max, y_max, x_max = mask.shape
    seen_edges: set[tuple[int, int, int]] = set()

    for seed in seed_nodes:
        if seed not in G:
            continue
        for u, v, key, edge_data in G.edges(seed, keys=True, data=True):
            edge_id = (int(min(u, v)), int(max(u, v)), int(key))
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            total_edges += 1

            voxels = edge_data.get("voxels")
            if not voxels:
                uncovered_edges += 1
                continue

            covered = False
            for voxel in voxels:
                if voxel is None or len(voxel) < 3:
                    continue
                z = int(round(float(voxel[0])))
                y = int(round(float(voxel[1])))
                x = int(round(float(voxel[2])))
                if z < 0 or y < 0 or x < 0 or z >= z_max or y >= y_max or x >= x_max:
                    continue
                if bool(mask[z, y, x]):
                    covered = True
                    break
            if not covered:
                uncovered_edges += 1

    return (uncovered_edges == 0), int(uncovered_edges), int(total_edges)
