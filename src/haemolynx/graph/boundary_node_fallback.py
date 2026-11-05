"""Utilities for boundary-node fallback selection logic."""
from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
import numpy as np

from ._helpers import points_inside_mask


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


def _edge_points(G: nx.Graph, u: Any, v: Any, edge_data: dict) -> np.ndarray:
    """An edge's centreline samples, or its two node positions without them."""
    voxels = edge_data.get("voxels")
    if voxels:
        arr = np.asarray([row for row in voxels if row is not None and len(row) >= 3], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3 and len(arr):
            return arr[:, :3]
    points = [G.nodes[n].get("pos") for n in (u, v)]
    points = [np.asarray(p, dtype=float) for p in points if p is not None]
    return np.asarray(points, dtype=float).reshape(-1, 3)


def seed_nodes_without_mask_coverage(
    G: nx.MultiGraph,
    seed_nodes: Iterable[Any],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> list[Any]:
    """The seed nodes none of whose edges touches *mask*, in the order given.

    A seed's edge is covered when any of its centreline samples (``voxels``,
    physical ``(z, y, x)`` microns, like node ``pos``) falls in a True voxel
    of *mask*, read through *voxel_size_zyx* -- the mask is voxel-indexed.
    Rounding microns straight to indices, as this used to, read the wrong
    voxels on every stack whose voxels are not 1 um, and the coverage test
    then failed for edges lying wholly inside the mask.
    """
    uncovered: list[Any] = []
    for seed in seed_nodes:
        if seed not in G:
            continue
        edges = (
            G.edges(seed, keys=True, data=True)
            if isinstance(G, nx.MultiGraph)
            else ((u, v, 0, d) for u, v, d in G.edges(seed, data=True))
        )
        covered = False
        for u, v, _key, edge_data in edges:
            points = _edge_points(G, u, v, edge_data)
            if len(points) and points_inside_mask(points, mask, voxel_size_zyx=voxel_size_zyx).any():
                covered = True
                break
        if not covered:
            uncovered.append(seed)
    return uncovered


def seed_edges_have_full_mask_coverage(
    G: nx.MultiGraph,
    seed_nodes: list[int],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[bool, int, int]:
    """Whether every seed-adjacent edge overlaps *mask*.

    Returns ``(all_covered, uncovered_edge_count, total_seed_edge_count)``.
    Edge ``voxels`` are physical ``(z, y, x)`` microns; *voxel_size_zyx*
    maps them onto the mask's voxels. See
    :func:`seed_nodes_without_mask_coverage` for the per-seed form the
    pipeline's fallback uses.
    """
    total_edges = 0
    uncovered_edges = 0
    seen_edges: set[tuple[Any, Any, int]] = set()
    for seed in seed_nodes:
        if seed not in G:
            continue
        edges = (
            G.edges(seed, keys=True, data=True)
            if isinstance(G, nx.MultiGraph)
            else ((u, v, 0, d) for u, v, d in G.edges(seed, data=True))
        )
        for u, v, key, edge_data in edges:
            edge_id = (min(u, v), max(u, v), key)
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            total_edges += 1
            points = _edge_points(G, u, v, edge_data)
            if not len(points) or not points_inside_mask(
                points, mask, voxel_size_zyx=voxel_size_zyx
            ).any():
                uncovered_edges += 1
    return (uncovered_edges == 0), int(uncovered_edges), int(total_edges)
