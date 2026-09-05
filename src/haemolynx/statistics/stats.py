"""Vessel network statistics."""
from __future__ import annotations

from collections import deque
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from networkx.algorithms.community import greedy_modularity_communities

from haemolynx.geometry import cumulative_lengths
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes
from haemolynx.visualization.geometry import edge_polyline

#Need to add in bifurcation ratios

def compute_basic_statistics(
    G: Union[nx.Graph, nx.MultiGraph], is_multigraph: bool
) -> Dict[str, Any]:
    """Compute basic graph statistics."""
    if is_multigraph:
        edge_lengths = [
            d["length"]
            for _, _, _, d in G.edges(keys=True, data=True)
            if d.get("length") is not None
        ]
    else:
        edge_lengths = [
            d["length"]
            for _, _, d in G.edges(data=True)
            if d.get("length") is not None
        ]
    node_degrees = [G.degree(n) for n in G.nodes()]
    return {
        "Total Nodes": G.number_of_nodes(),
        "Total Edges": G.number_of_edges(),
        "Total Edge Length (microns)": sum(edge_lengths) if edge_lengths else 0,
        "Average Edge Length (microns)": (
            sum(edge_lengths) / len(edge_lengths) if edge_lengths else 0
        ),
        "Average Degree": sum(node_degrees) / len(node_degrees) if node_degrees else 0,
    }

#Update with newer tortuosity values HD gave to Anna
def compute_tortuosity_measures(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict],
    is_multigraph: bool,
) -> Dict[str, Any]:
    """Compute tortuosity index and curvature."""
    if node_positions is None:
        return {
            "Average Tortuosity Index": "N/A (no position data)",
            "Average Curvature": "N/A (no position data)",
        }
    tortuosity_indices = []
    curvatures = []
    it = G.edges(keys=True, data=True) if is_multigraph else G.edges(data=True)
    for item in it:
        u, v = item[0], item[1]
        edge_data = item[-1]
        if u in node_positions and v in node_positions:
            pos_u = np.array(node_positions[u])
            pos_v = np.array(node_positions[v])
            straight = euclidean(pos_u, pos_v)
            path_length = edge_data.get("length", straight)
            if straight > 0:
                tortuosity_indices.append(path_length / straight)
                if path_length > 0:
                    curvatures.append((path_length - straight) / path_length)
    return {
        "Average Tortuosity Index": (
            np.mean(tortuosity_indices) if tortuosity_indices else 0
        ),
        "Average Curvature": np.mean(curvatures) if curvatures else 0,
    }


def compute_branching_statistics(G: nx.Graph) -> Dict[str, Any]:
    """Count branching points (degree > 2 nodes).

    This used to also report "Average Branching Angle (degrees)": every
    pair of a junction's neighbours, angled node-to-node in a straight line
    and pooled into one network-wide mean. compute_branch_order_statistics's
    "Mean Emergence Angle (degrees)" (compute_emergence_angles_by_branch_order)
    replaced it with a more rigorous measurement -- one parent versus each
    daughter, along the local centreline tangent rather than a straight
    node-to-node line, broken down per branch order rather than pooled --
    so the cruder version was removed rather than left alongside it.
    """
    return {
        "Number of Branching Points": len(
            [n for n in G.nodes() if G.degree(n) > 2]
        ),
    }


def _tree_asymmetry_from_root(G: nx.Graph, root: Any) -> tuple[float, int]:
    """Iterative post-order asymmetry/subtree-size accumulation, rooted at *root*.

    A recursive version of this (one call per node, following the tree down
    from the root) hits Python's default recursion limit on a long,
    sparsely-branching stretch of vessel -- a real shape, not a pathological
    one, once a network has been simplified to a spanning tree. BFS order
    reversed is a safe post-order for a tree: every node's children appear
    strictly later than it in BFS, so processing nodes in reverse guarantees
    each node's children are already resolved.
    """
    parent: Dict[Any, Any] = {root: None}
    order: list[Any] = [root]
    queue: deque = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor == parent[node] or neighbor in parent:
                continue
            parent[neighbor] = node
            order.append(neighbor)
            queue.append(neighbor)

    result: Dict[Any, tuple[float, int]] = {}
    for node in reversed(order):
        children = [
            result[neighbor]
            for neighbor in G.neighbors(node)
            if neighbor != parent[node]
        ]
        if not children:
            result[node] = (0.0, 1)
            continue
        child_sizes = [size for _asymmetry, size in children]
        total = sum(asymmetry for asymmetry, _size in children)
        node_a = max(child_sizes) - min(child_sizes) if len(child_sizes) >= 2 else 0
        result[node] = (total + node_a, sum(child_sizes) + 1)
    return result[root]


def compute_tree_asymmetry(G: nx.Graph) -> Dict[str, Any]:
    """Compute tree asymmetry index."""
    if G.number_of_nodes() == 0:
        return {"Tree Asymmetry Index": "N/A (empty graph)"}

    if not nx.is_tree(G):
        if nx.is_connected(G):
            # There is no "weight" edge attribute on these graphs (see
            # graph.assert_no_forbidden_edge_attributes); without an explicit
            # weight=, every edge is treated as weight 1 and the "minimum"
            # spanning tree is really an arbitrary one picked by Kruskal's
            # tie-breaking, not the vessel network's own shortest-path
            # skeleton. A real vascular graph has loops (capillary beds
            # anastomose), so this branch is the common case, not an edge
            # case.
            G = nx.minimum_spanning_tree(G.copy(), weight="length")
        else:
            return {"Tree Asymmetry Index": "N/A (disconnected graph)"}
    root = max(G.nodes(), key=G.degree)
    asymmetry, size = _tree_asymmetry_from_root(G, root)
    return {
        "Tree Asymmetry Index": asymmetry / size if size > 0 else 0
    }


def _box_counting_fractal_dimension(points: np.ndarray) -> float:
    """Box-counting fractal-dimension estimate for a physical point cloud."""
    max_range = np.max(points.max(axis=0) - points.min(axis=0))
    min_bs = max_range / 100
    max_bs = max_range / 2
    box_sizes, box_counts = [], []
    for bs in np.logspace(np.log10(min_bs), np.log10(max_bs), 10):
        min_c = points.min(axis=0)
        indices = ((points - min_c) / bs).astype(int)
        box_sizes.append(bs)
        box_counts.append(len(set(tuple(i) for i in indices)))
    if len(box_sizes) > 1 and all(c > 0 for c in box_counts):
        return float(-np.polyfit(np.log(box_sizes), np.log(box_counts), 1)[0])
    return 0.0


def _centreline_points(G: Union[nx.Graph, nx.MultiGraph]) -> np.ndarray:
    """Every point along every edge's real centreline, concatenated.

    Reads each edge's ``voxels`` polyline (the smoothed centreline, where
    available) via the same `edge_polyline` the vessel-tube drawing and the
    emergence-angle tangent both use, so this is the network's actual
    physical shape -- not just the branch/terminal points left after
    topology simplification.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)
    chunks: list[np.ndarray] = []
    for item in edge_iter:
        u, v = item[0], item[1]
        data = item[-1]
        try:
            chunks.append(edge_polyline(G, u, v, data))
        except (TypeError, ValueError):
            continue
    if not chunks:
        return np.empty((0, 3))
    return np.concatenate(chunks, axis=0)


def compute_fractal_dimension(
    G: Union[nx.Graph, nx.MultiGraph], node_positions: Optional[dict]
) -> Dict[str, Any]:
    """Compute two box-counting fractal-dimension estimates.

    "Fractal Dimension (Node Positions)" counts only the branch/terminal
    points left after topology simplification -- fast, but blind to the
    vessel's actual path shape between junctions, so it understates spatial
    complexity. "Fractal Dimension (Centreline)" counts every point along
    every edge's real centreline instead, matching standard vascular
    fractal-dimension methodology, at the cost of many more points to bin.
    The two are not expected to agree; both are reported explicitly rather
    than folded into one "Fractal Dimension" value.
    """
    result: Dict[str, Any] = {}

    if node_positions is None or len(node_positions) < 2:
        result["Fractal Dimension (Node Positions)"] = "N/A (insufficient position data)"
    else:
        node_points = np.array(
            [node_positions[n] for n in G.nodes() if n in node_positions]
        )
        if len(node_points) < 2:
            result["Fractal Dimension (Node Positions)"] = "N/A (insufficient position data)"
        else:
            result["Fractal Dimension (Node Positions)"] = _box_counting_fractal_dimension(
                node_points
            )

    centreline_points = _centreline_points(G)
    if len(centreline_points) < 2:
        result["Fractal Dimension (Centreline)"] = "N/A (insufficient position data)"
    else:
        result["Fractal Dimension (Centreline)"] = _box_counting_fractal_dimension(
            centreline_points
        )

    return result


def compute_path_efficiency(
    G: Union[nx.Graph, nx.MultiGraph],
    is_multigraph: bool,
    max_pairs: Optional[int] = 5000,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Compute path efficiency from weighted shortest-path lengths.

    Path efficiency is defined here as the inverse of the mean weighted
    shortest-path length across all unique node pairs.
    """
    if is_multigraph:
        # Reduce a MultiGraph to a simple graph by keeping the lightest edge
        # between each unordered node pair.
        G_s = nx.Graph()
        G_s.add_nodes_from(G.nodes())
        ew = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            w = d.get("length", 1)
            uv = (u, v)
            if uv not in ew or w < ew[uv]:
                ew[uv] = w
        for (u, v), w in ew.items():
            G_s.add_edge(u, v, length=w)
    else:
        G_s = G

    # Efficiency is undefined for disconnected graphs because some node pairs
    # have infinite path length.
    if not nx.is_connected(G_s):
        return {"Path Efficiency": "N/A (disconnected graph)"}

    # For a connected graph, every pair should have a valid path.
    # Unexpected failures should surface as errors rather than being silently
    # swallowed, otherwise statistics can look valid while being wrong.
    path_lengths = []
    nodes = list(G_s.nodes())
    total_pairs = (len(nodes) * (len(nodes) - 1)) // 2

    # For large graphs, estimate efficiency from a bounded random sample of
    # node pairs to avoid very long runtimes.
    if max_pairs is not None and total_pairs > max_pairs:
        rng = np.random.default_rng(rng_seed)
        sampled_pairs = set()
        while len(sampled_pairs) < max_pairs:
            i = int(rng.integers(0, len(nodes)))
            j = int(rng.integers(0, len(nodes)))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            sampled_pairs.add((i, j))

        pairs = [(nodes[i], nodes[j]) for i, j in sampled_pairs]
    else:
        pairs = []
        for i, src in enumerate(nodes):
            for tgt in nodes[i + 1 :]:
                pairs.append((src, tgt))

    for src, tgt in pairs:
        try:
            pl = nx.shortest_path_length(G_s, src, tgt, weight="length")
        except nx.NetworkXNoPath as exc:
            raise RuntimeError(
                f"No path between connected-graph nodes {src} and {tgt}"
            ) from exc
        path_lengths.append(pl)

    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    efficiency = 1 / avg_path_length if avg_path_length > 0 else 0
    return {
        "Path Efficiency": efficiency,
        "Average Shortest Path Length (microns)": avg_path_length,
        "Path Efficiency Pair Sample Size": len(path_lengths),
        "Path Efficiency Pair Coverage": (
            len(path_lengths) / total_pairs if total_pairs > 0 else 0
        ),
    }


def compute_vessel_density(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict],
    voxel_size,
    image_dimensions,
    is_multigraph: bool,
) -> Dict[str, Any]:
    """Compute vessel density."""
    if is_multigraph:
        lengths = [
            d.get("length", 0)
            for _, _, _, d in G.edges(keys=True, data=True)
        ]
    else:
        lengths = [
            d.get("length", 0)
            for _, _, d in G.edges(data=True)
        ]
    total_length = sum(lengths)
    out = {"Total Vessel Length (microns)": total_length}
    if node_positions and len(node_positions) > 0:
        positions = np.array(
            [node_positions[n] for n in G.nodes() if n in node_positions]
        )
        if len(positions) > 0:
            vol = np.prod(
                positions.max(axis=0) - positions.min(axis=0)
            )
            out["Vessel Density in Tissue (microns/micron³)"] = (
                total_length / vol if vol > 0 else 0
            )
            out["Vessel-Occupied Volume (micron³)"] = vol
        else:
            out["Vessel Density in Tissue (microns/micron³)"] = (
                "N/A (no position data)"
            )
    else:
        out["Vessel Density in Tissue (microns/micron³)"] = "N/A (no position data)"

    if image_dimensions is not None and voxel_size is not None:
        img_vol = np.prod(
            [d * v for d, v in zip(image_dimensions, voxel_size)]
        )
        out["Vessel Density in Whole Image (microns/micron³)"] = (
            total_length / img_vol if img_vol > 0 else 0
        )
        out["Total Image Volume (micron³)"] = img_vol
    else:
        out["Vessel Density in Whole Image (microns/micron³)"] = (
            "N/A (no image dimension data)"
        )
    return out


def compute_intercapillary_distance(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    max_neighbors: int = 64,
) -> Dict[str, Any]:
    """Nearest-neighbour spacing between non-adjacent vessels.

    For each edge, the distance from its own centreline to the closest
    point on any *other* edge that does not share a node with it -- a
    shared node is trivially close at the junction itself, not a
    meaningful measure of tissue spacing. This is the key input to
    Krogh-cylinder oxygen-diffusion modelling: how far apart two vessels
    actually sit, not how far apart their branch points are.

    Approximate by design: each edge's own polyline (``voxels``, or its two
    node positions when that is all there is) is queried against a single
    global k-nearest-neighbour tree of every edge's sampled points, taking
    the closest of up to *max_neighbors* results that belongs to an allowed
    (non-self, non-adjacent) edge. An edge whose nearest *max_neighbors*
    sampled points are all on itself or an adjacent edge -- possible next to
    one very densely sampled, very long neighbour -- is left out of the
    average rather than searched exhaustively.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    raw_edges = (
        list(G.edges(keys=True, data=True))
        if is_mg
        else [(u, v, 0, d) for u, v, d in G.edges(data=True)]
    )

    edge_points: list = []
    incident: Dict[Any, set] = {}
    for u, v, _key, data in raw_edges:
        try:
            points = np.asarray(edge_polyline(G, u, v, data), dtype=float)
        except (TypeError, ValueError):
            continue
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        idx = len(edge_points)
        edge_points.append((points, u, v))
        incident.setdefault(u, set()).add(idx)
        incident.setdefault(v, set()).add(idx)

    if len(edge_points) < 2:
        return {
            "Mean Intercapillary Distance (microns)": "N/A (fewer than two edges)",
            "Median Intercapillary Distance (microns)": "N/A (fewer than two edges)",
            "Intercapillary Distance Sample Count": 0,
        }

    all_points = np.concatenate([pts for pts, _u, _v in edge_points], axis=0)
    point_edge_idx = np.concatenate(
        [
            np.full(pts.shape[0], i, dtype=np.intp)
            for i, (pts, _u, _v) in enumerate(edge_points)
        ]
    )
    tree = cKDTree(all_points)
    k = min(int(max_neighbors), all_points.shape[0])

    distances: list[float] = []
    for i, (points, u, v) in enumerate(edge_points):
        forbidden = {i} | incident.get(u, set()) | incident.get(v, set())
        dists, idxs = tree.query(points, k=k)
        if k == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)
        best: Optional[float] = None
        for row_d, row_i in zip(dists, idxs):
            for d, point_i in zip(np.atleast_1d(row_d), np.atleast_1d(row_i)):
                if int(point_edge_idx[point_i]) in forbidden:
                    continue
                d = float(d)
                if best is None or d < best:
                    best = d
                break  # results are sorted ascending; first allowed one wins
        if best is not None:
            distances.append(best)

    if not distances:
        return {
            "Mean Intercapillary Distance (microns)": (
                "N/A (no edge found an unrelated neighbour)"
            ),
            "Median Intercapillary Distance (microns)": (
                "N/A (no edge found an unrelated neighbour)"
            ),
            "Intercapillary Distance Sample Count": 0,
        }
    return {
        "Mean Intercapillary Distance (microns)": float(np.mean(distances)),
        "Median Intercapillary Distance (microns)": float(np.median(distances)),
        "Intercapillary Distance Sample Count": len(distances),
    }


def compute_communities_summary(
    G: nx.Graph, max_nodes_exact: int = 1500
) -> Dict[str, Any]:
    """Compute community statistics with runtime guards."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"Community Count": 0}

    if n_nodes <= max_nodes_exact:
        communities = list(greedy_modularity_communities(G))
        sizes = [len(c) for c in communities]
        return {
            "Community Count": len(communities),
            "Largest Community Size": max(sizes) if sizes else 0,
            "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
            "Community Method": "greedy_modularity",
        }

    # Fallback for large graphs: connected components are fast and stable.
    components = list(nx.connected_components(G))
    sizes = [len(c) for c in components]
    return {
        "Community Count": len(components),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": "connected_components_fallback",
    }


def compute_communities(G: nx.Graph):
    # Plain topological communities. Weighted community detection (by
    # resistance, geometry/length, or solved |flow|) is
    # compute_weighted_communities_summary, all three reported together by
    # compute_betweenness_and_community_measurements.
    return list(greedy_modularity_communities(G))


def compute_betweenness_summary(
    G: nx.Graph,
    max_nodes_exact: int = 1000,
    approx_k: int = 128,
    seed: int = 42,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute compact betweenness summary, avoiding huge outputs."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G)
        method = "exact"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(G, k=k, seed=seed)
        method = f"approx_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": top,
        "Betweenness Method": method,
    }


def compute_betweenness(G: nx.Graph):
    # Plain topological betweenness. Weighted betweenness (by resistance,
    # geometry/length, or solved |flow|) is compute_weighted_betweenness_
    # summary, all three reported together by
    # compute_betweenness_and_community_measurements.
    return nx.betweenness_centrality(G)


def _simple_graph_with_edge_attr(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    transform: Optional[Callable[[float], float]] = None,
    target_attr: str = "analysis_weight",
) -> nx.Graph:
    """Build a simple graph carrying one transformed edge attribute.

    For MultiGraph inputs, parallel edges are collapsed by taking the smallest
    transformed value, which is appropriate for path-based distance weights.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_s = nx.Graph()
    G_s.add_nodes_from(G.nodes())

    if transform is None:
        transform = lambda x: x  # noqa: E731

    if is_mg:
        best = {}
        for u, v, _, data in G.edges(keys=True, data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            uv = tuple(sorted((u, v)))
            if uv not in best or transformed < best[uv]:
                best[uv] = float(transformed)
        for (u, v), val in best.items():
            G_s.add_edge(u, v, **{target_attr: val})
    else:
        for u, v, data in G.edges(data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            existing = G_s.get_edge_data(u, v, default={}).get(target_attr)
            if existing is None or transformed < existing:
                G_s.add_edge(u, v, **{target_attr: float(transformed)})
    return G_s


def compute_weighted_betweenness_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = 1000,
    approx_k: int = 128,
    seed: int = 42,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute weighted betweenness summary from a chosen edge attribute."""
    transform = (lambda x: 1.0 / x) if inverse_source_attr else None
    G_s = _simple_graph_with_edge_attr(
        G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
    )
    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G_s, weight="analysis_weight")
        method = "exact_weighted"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(
            G_s, k=k, seed=seed, weight="analysis_weight"
        )
        method = f"approx_weighted_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": [
            {"node": node, "value": float(value)} for node, value in top
        ],
        "Betweenness Method": method,
    }


def compute_weighted_communities_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = 1500,
) -> Dict[str, Any]:
    """Compute weighted community summary using greedy modularity."""
    transform = (lambda x: 1.0 / x) if inverse_source_attr else None
    G_s = _simple_graph_with_edge_attr(
        G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
    )
    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return {"Community Count": 0}

    if n_nodes <= max_nodes_exact:
        communities = list(greedy_modularity_communities(G_s, weight="analysis_weight"))
        sizes = [len(c) for c in communities]
        return {
            "Community Count": len(communities),
            "Largest Community Size": max(sizes) if sizes else 0,
            "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
            "Community Method": "greedy_modularity_weighted",
        }

    components = list(nx.connected_components(G_s))
    sizes = [len(c) for c in components]
    return {
        "Community Count": len(components),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": "connected_components_fallback",
    }


def compute_betweenness_and_community_measurements(
    G: Union[nx.Graph, nx.MultiGraph],
) -> Dict[str, Dict[str, Any]]:
    """Compute weighted betweenness/community using three edge distance models.

    - "edge_resistance": Poiseuille resistance as the shortest-path distance
      -- the vessels that impede flow least are the most central.
    - "edge_length": physical length as the distance -- purely geometric,
      independent of haemodynamics.
    - "edge_flow_abs": inverse of the solved absolute flow (``flow_abs``,
      written by haemodynamics.resistance.set_edge_flows) as the distance --
      vessels carrying the most flow are treated as the shortest, most
      travelled paths. Only meaningful once flow has been solved; edges with
      no flow (or none solved yet) drop out the same way a missing
      resistance or length would.

    All three are reported together rather than one at a time, so a caller
    never has to guess which distance model a given number came from.
    """
    resistance_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="resistance", inverse_source_attr=False
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="resistance", inverse_source_attr=False
        ),
    }
    edge_length_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="length", inverse_source_attr=False
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="length", inverse_source_attr=False
        ),
    }
    edge_flow_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="flow_abs", inverse_source_attr=True
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="flow_abs", inverse_source_attr=True
        ),
    }
    return {
        "edge_resistance": resistance_results,
        "edge_length": edge_length_results,
        "edge_flow_abs": edge_flow_results,
    }


def compute_comprehensive_vessel_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
    statistics_mode: str = "fast",
) -> Dict[str, Any]:
    """Combine all vessel statistics.

    Length-based metrics read the ``length`` edge attribute (microns). They never
    read ``resistance``/``conductance``, so running haemodynamics first cannot
    change a reported length.
    """
    assert_no_forbidden_edge_attributes(G, context="vessel statistics")
    valid_modes = {"fast", "full"}
    if statistics_mode not in valid_modes:
        raise ValueError(
            f"Invalid statistics_mode='{statistics_mode}'. "
            f"Choose one of {sorted(valid_modes)}."
        )

    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = (
        (nx.Graph(G) if not G.is_directed() else nx.DiGraph(G))
        if is_mg
        else G
    )

    base = {
        **compute_basic_statistics(G, is_mg),
        **compute_tortuosity_measures(G, node_positions, is_mg),
        **compute_branching_statistics(G_simple),
        **compute_tree_asymmetry(G_simple),
        # The original G, not G_simple: collapsing parallel edges to build
        # G_simple keeps only one edge's data per node pair, which would
        # silently drop other parallel edges' centreline points here.
        **compute_fractal_dimension(G, node_positions),
        **compute_vessel_density(
            G, node_positions, voxel_size, image_dimensions, is_mg
        ),
        # The original G, not G_simple, for the same reason as fractal
        # dimension: a junction's parent/daughter diameters live on specific
        # parallel edges that G_simple would collapse away.
        **compute_murray_law_compliance(G),
        **compute_daughter_daughter_angles(G),
        # The original G, not G_simple: each parallel edge is its own
        # vessel with its own spacing to its neighbours.
        **compute_intercapillary_distance(G),
    }

    if statistics_mode == "full":
        return {
            **base,
            **compute_path_efficiency(G, is_mg, max_pairs=None),
            "communities": compute_communities(G_simple),
            **compute_betweenness(G_simple),
            "Statistics Mode": "full",
        }

    return {
        **base,
        **compute_path_efficiency(G, is_mg),
        **compute_communities_summary(G_simple),
        **compute_betweenness_summary(G_simple),
        "Statistics Mode": "fast",
    }


def _flatten_statistics_dict(
    stats: Dict[str, Any], prefix: str = ""
) -> list[tuple[str, Any]]:
    """Flatten nested statistics dict for tabular export."""
    flattened: list[tuple[str, Any]] = []
    for key, value in stats.items():
        full_key = f"{prefix} > {key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.extend(_flatten_statistics_dict(value, full_key))
        else:
            flattened.append((full_key, value))
    return flattened


def _split_metric_path(metric_path: str) -> tuple[str, str]:
    """Split flattened key path into section and metric."""
    if " > " not in metric_path:
        return "Core Statistics", metric_path
    section, metric = metric_path.rsplit(" > ", 1)
    return section, metric


def _extract_unit(metric_name: str) -> tuple[str, str]:
    """Extract trailing '(unit)' from metric labels when present."""
    m = re.search(r"\(([^()]+)\)\s*$", metric_name)
    if not m:
        return metric_name, ""
    unit = m.group(1).strip()
    clean_metric = re.sub(r"\s*\([^()]+\)\s*$", "", metric_name).strip()
    return clean_metric, unit


def _format_stat_value(metric_name: str, value: Any) -> tuple[str, str]:
    """Format metric value plus a short human-readable note."""
    if isinstance(value, str):
        if value.startswith("N/A"):
            return value, "Not computable with current inputs."
        return value, ""

    if isinstance(value, (bool, np.bool_)):
        return ("True" if bool(value) else "False"), ""

    if isinstance(value, (int, np.integer)):
        return str(int(value)), ""

    if isinstance(value, (float, np.floating)):
        f_val = float(value)
        if "Coverage" in metric_name:
            return f"{f_val:.2%}", "Fraction of all node-pairs included."
        return f"{f_val:.6g}", ""

    if isinstance(value, (list, tuple, set)):
        serializable = list(value) if not isinstance(value, list) else value
        try:
            return json.dumps(serializable), "Serialized collection value."
        except TypeError:
            return str(serializable), "Collection converted to text."

    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True), "Serialized nested object."
        except TypeError:
            return str(value), "Nested object converted to text."

    return str(value), ""


def _annotation_for_metric(metric_name: str) -> str:
    """Return short metric-specific annotation used in CSV notes."""
    if metric_name == "Statistics Mode":
        return "fast=sampled/compact metrics, full=heavier/exact metrics."
    if "Tortuosity" in metric_name:
        return "Higher values indicate less straight vessels."
    if "Curvature" in metric_name:
        return "Relative deviation from straight vessel segments."
    if "Branching Points" in metric_name:
        return (
            "Count of junctions (degree > 2); see the per-branch-order "
            "Mean Emergence Angle for angle detail."
        )
    if "Asymmetry" in metric_name:
        return "Tree imbalance estimate after simplification to a tree."
    if "Fractal Dimension" in metric_name:
        return "Box-counting estimate of structural complexity."
    if "Path Efficiency" in metric_name:
        return "Inverse of mean shortest-path distance."
    if "Betweenness" in metric_name:
        return "Centrality based on shortest-path traffic."
    if "Community" in metric_name:
        return "Subnetwork partition summary."
    if "Density" in metric_name:
        return "Total vessel length normalized by available volume."
    return ""


def export_statistics_to_csv(
    stats: Dict[str, Any],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export statistics dictionary to a readable annotated CSV file.

    The CSV is long-format with one row per metric and includes:
    section, metric name, value, parsed unit, and notes.
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flattened = _flatten_statistics_dict(stats)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric", "Value", "Unit", "Notes"])
        writer.writerow(
            [
                "Metadata",
                "Exported At (UTC)",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "",
                "Generated by HaemoLynx statistics exporter.",
            ]
        )
        writer.writerow(
            [
                "Metadata",
                "Metric Count",
                str(len(flattened)),
                "",
                "Number of metric rows in this file.",
            ]
        )

        for metric_path, value in flattened:
            section, metric = _split_metric_path(metric_path)
            metric_clean, unit = _extract_unit(metric)
            formatted_value, value_note = _format_stat_value(metric_clean, value)
            metric_note = _annotation_for_metric(metric_clean)
            notes = " ".join(n for n in [value_note, metric_note] if n).strip()
            writer.writerow([section, metric_clean, formatted_value, unit, notes])

    return output_path


def _normalize_branch_order_tag(tag: Any) -> Optional[str]:
    """Normalize branch-order labels to ArtN / BON / VenN where possible."""
    if tag is None:
        return None
    label = str(tag).strip()
    if not label:
        return None
    m = re.match(r"^(art|ven|bo|b)\s*0*(\d+)$", label, flags=re.IGNORECASE)
    if not m:
        return label
    prefix = m.group(1).lower()
    n = int(m.group(2))
    if prefix == "art":
        return f"Art{n}"
    if prefix == "ven":
        return f"Ven{n}"
    return f"BO{n}"


def _branch_order_sort_key(tag: str) -> tuple[int, int, str]:
    """Sort as Art1..ArtN, BO1..BON, Ven1..VenN, then unknown labels."""
    m = re.match(r"^(art|ven|bo)\s*(\d+)$", str(tag), flags=re.IGNORECASE)
    if not m:
        return (3, 0, str(tag))
    prefix = m.group(1).lower()
    n = int(m.group(2))
    group = {"art": 0, "bo": 1, "ven": 2}.get(prefix, 3)
    return (group, n, str(tag))


def _incident_edge_items(
    G: Union[nx.Graph, nx.MultiGraph], node: Any
) -> list[tuple[Any, Any, Any, dict]]:
    """Incident edges as ``(u, v, key, data)`` with ``u`` equal to ``node``."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return list(G.edges(node, keys=True, data=True))
    return [(u, v, None, d) for u, v, d in G.edges(node, data=True)]


def _iter_parent_daughter_junctions(
    G: Union[nx.Graph, nx.MultiGraph],
):
    """Junctions with a unique lowest-branch-order-rank parent and >=1 daughter.

    Yields ``(node, parent_item, daughter_items)`` where each item is
    ``(u, v, key, data, tag)``. This is the same junction definition
    :func:`compute_emergence_angles_by_branch_order` uses (the unique
    lowest-rank incident edge is the parent, every other labelled edge a
    daughter; tied ranks, unlabelled-only, or degree < 3 contribute
    nothing) -- shared here so a second and third junction-level statistic
    (Murray's law, daughter-daughter angle) do not each redefine "parent"
    on their own.
    """
    for node in G.nodes():
        if int(G.degree(node)) < 3:
            continue
        labelled: list[tuple[Any, Any, Any, dict, str]] = []
        for u, v, key, data in _incident_edge_items(G, node):
            if u == v:
                continue
            tag = _normalize_branch_order_tag(data.get("branch_order"))
            if not tag:
                continue
            labelled.append((u, v, key, data, tag))
        if len(labelled) < 2:
            continue
        ranks = [_branch_order_sort_key(item[4]) for item in labelled]
        min_rank = min(ranks)
        parent_indices = [i for i, rank in enumerate(ranks) if rank == min_rank]
        if len(parent_indices) != 1:
            continue
        parent_i = parent_indices[0]
        daughter_items = [item for i, item in enumerate(labelled) if i != parent_i]
        yield node, labelled[parent_i], daughter_items


def _point_along_polyline(
    points: np.ndarray, distance_um: float
) -> Optional[np.ndarray]:
    """Interpolate a point this far along a polyline, clamped to its length."""
    lengths = cumulative_lengths(points)
    total = float(lengths[-1])
    if total <= 0.0:
        return None
    target = min(max(float(distance_um), 0.0), total)
    if target <= 0.0:
        for i in range(1, len(points)):
            if float(lengths[i]) > 0.0:
                return np.asarray(points[i], dtype=float)
        return None
    idx = int(np.searchsorted(lengths, target, side="left"))
    idx = min(max(idx, 1), len(points) - 1)
    t0 = float(lengths[idx - 1])
    t1 = float(lengths[idx])
    if t1 <= t0:
        return np.asarray(points[idx], dtype=float)
    frac = (target - t0) / (t1 - t0)
    start = np.asarray(points[idx - 1], dtype=float)
    end = np.asarray(points[idx], dtype=float)
    return start + frac * (end - start)


def _outgoing_unit_tangent(
    G: Union[nx.Graph, nx.MultiGraph],
    node: Any,
    u: Any,
    v: Any,
    data: dict,
    tangent_length_um: float,
) -> Optional[np.ndarray]:
    """Unit vector leaving ``node`` along this edge's local centreline."""
    try:
        points = edge_polyline(G, u, v, data)
    except (TypeError, ValueError):
        return None
    if node == u:
        path = points
    elif node == v:
        path = points[::-1]
    else:
        return None
    dest = _point_along_polyline(path, tangent_length_um)
    if dest is None:
        return None
    vec = np.asarray(dest, dtype=float) - np.asarray(path[0], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def _angle_between_unit_vectors(a: np.ndarray, b: np.ndarray) -> float:
    cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _empty_branch_order_record(tag: str) -> Dict[str, Any]:
    return {
        "Branch Order": tag,
        "Edge Count": 0,
        "Mean Length (microns)": 0.0,
        "Mean Tortuosity Index": "N/A (no position data)",
        "Tortuosity Sample Count": 0,
        "Mean Emergence Angle (degrees)": "N/A (no unique parent junction)",
        "Emergence Angle Sample Count": 0,
        "Mean Pressure Drop (Pa)": "N/A (no flow solved)",
        "Total Pressure Drop (Pa)": "N/A (no flow solved)",
        "Pressure Drop Sample Count": 0,
        "Mean Diameter (microns)": "N/A (no diameter assigned)",
        "Diameter Coefficient of Variation": "N/A (no diameter assigned)",
        "Diameter Sample Count": 0,
        "_diameter_sum_sq": 0.0,
    }


def compute_emergence_angles_by_branch_order(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Angle each daughter leaves its parent, grouped by the daughter's order.

    At a junction the parent is the unique incident edge with the lowest
    branch-order rank (Art* before BO* before Ven*, then the numeric index).
    Each other labelled incident edge is a daughter. The emergence angle is
    the deflection of the daughter's outgoing centreline tangent from the
    parent's incoming tangent: 0° continues the parent, 90° leaves at a
    right angle.

    Junctions with no unique lowest-order parent (tied ranks, unlabelled
    edges only, or degree < 3) contribute nothing, so root segments have
    no emergence angle.

    Each junction is evaluated independently, but an edge that is a daughter
    at one end can also be a daughter at its other end -- neither endpoint's
    local minimum, the way a capillary bridging two comparable-order
    neighbourhoods looks, not a strict parent -> child step. That edge's
    emergence still counts once, at whichever of its two junctions is
    reached first, not once per end.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    # An edge with neither endpoint the local minimum rank (a capillary
    # bridging two comparable-order neighbourhoods, not a strict parent ->
    # child step) is a "daughter" at both of its junctions. Each junction is
    # processed independently, so without this, that one edge's emergence
    # would be added twice -- once per end -- to the same branch order's
    # sum/count. An edge id (not just (u, v): parallel edges between the same
    # two nodes are different edges) makes sure each edge contributes at most
    # once, no matter which of its two junctions is visited first.
    seen_daughters: set[tuple[Any, Any]] = set()

    for node in G.nodes():
        if int(G.degree(node)) < 3:
            continue
        labelled: list[tuple[Any, Any, Any, dict, str]] = []
        for u, v, key, data in _incident_edge_items(G, node):
            if u == v:
                continue
            tag = _normalize_branch_order_tag(data.get("branch_order"))
            if not tag:
                continue
            labelled.append((u, v, key, data, tag))
        if len(labelled) < 2:
            continue
        ranks = [_branch_order_sort_key(item[4]) for item in labelled]
        min_rank = min(ranks)
        parent_indices = [i for i, rank in enumerate(ranks) if rank == min_rank]
        if len(parent_indices) != 1:
            continue
        parent_i = parent_indices[0]
        p_u, p_v, _p_key, p_data, _parent_tag = labelled[parent_i]
        parent_out = _outgoing_unit_tangent(
            G, node, p_u, p_v, p_data, tangent_length_um
        )
        if parent_out is None:
            continue
        parent_in = -parent_out
        for i, (u, v, key, data, tag) in enumerate(labelled):
            if i == parent_i:
                continue
            edge_id = (frozenset((u, v)), key)
            if edge_id in seen_daughters:
                continue
            daughter_out = _outgoing_unit_tangent(
                G, node, u, v, data, tangent_length_um
            )
            if daughter_out is None:
                continue
            angle = _angle_between_unit_vectors(parent_in, daughter_out)
            sums[tag] = sums.get(tag, 0.0) + angle
            counts[tag] = counts.get(tag, 0) + 1
            seen_daughters.add(edge_id)

    ordered: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(counts, key=_branch_order_sort_key):
        n = counts[tag]
        ordered[tag] = {
            "Branch Order": tag,
            "Mean Emergence Angle (degrees)": sums[tag] / n,
            "Emergence Angle Sample Count": n,
        }
    return ordered


def compute_murray_law_compliance(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    exponent: float = 3.0,
) -> Dict[str, Any]:
    """How well parent^n ~= sum(daughter^n) holds at each qualifying junction.

    Murray's law (the classic ``exponent=3``) predicts the diameter
    relationship a minimum-total-work vascular network would have at every
    bifurcation; a ratio of 1.0 is ideal, and systematic deviation from it
    is a standard marker of abnormal (e.g. tumour) vasculature. Uses the
    same junction definition as :func:`compute_emergence_angles_by_branch_order`
    (see :func:`_iter_parent_daughter_junctions`): the unique lowest-rank
    incident edge is the parent, every other labelled edge a daughter.

    A junction missing a positive diameter on the parent or on any one of
    its daughters is skipped entirely -- a ratio computed from a partial
    set of daughters is not comparable to one computed from all of them.
    """
    ratios: list[float] = []
    for _node, parent_item, daughter_items in _iter_parent_daughter_junctions(G):
        if not daughter_items:
            continue
        parent_d = _positive_diameter_or_none(parent_item[3].get("diameter_um"))
        if parent_d is None:
            continue
        daughter_sum = 0.0
        for daughter_item in daughter_items:
            daughter_d = _positive_diameter_or_none(daughter_item[3].get("diameter_um"))
            if daughter_d is None:
                daughter_sum = None
                break
            daughter_sum += daughter_d**exponent
        if daughter_sum is None:
            continue
        ratios.append(daughter_sum / (parent_d**exponent))

    if not ratios:
        return {
            "Mean Murray Ratio": "N/A (no junction had diameters on every branch)",
            "Murray Ratio Sample Count": 0,
            "Murray Law Exponent": exponent,
        }
    return {
        "Mean Murray Ratio": float(np.mean(ratios)),
        "Murray Ratio Sample Count": len(ratios),
        "Murray Law Exponent": exponent,
    }


def _positive_diameter_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if value_f > 0 else None


def compute_daughter_daughter_angles(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Any]:
    """Angle between each pair of sibling daughters at a bifurcation.

    Distinct from :func:`compute_emergence_angles_by_branch_order` (parent
    vs. each daughter): this is the angle between the daughters themselves,
    the other half of classic bifurcation morphometry. Uses the same
    junction definition (see :func:`_iter_parent_daughter_junctions`) so a
    junction only contributes here if it also had a well-defined parent,
    keeping the two angles comparable measurements of the same events.

    A junction with more than two daughters (a trifurcation or wider)
    contributes one angle per pair, not just adjacent ones -- there is no
    inherent ordering among daughters at the same junction.
    """
    angles: list[float] = []
    for node, _parent_item, daughter_items in _iter_parent_daughter_junctions(G):
        if len(daughter_items) < 2:
            continue
        tangents: list[np.ndarray] = []
        for u, v, _key, data, _tag in daughter_items:
            tangent = _outgoing_unit_tangent(G, node, u, v, data, tangent_length_um)
            if tangent is not None:
                tangents.append(tangent)
        for i in range(len(tangents)):
            for j in range(i + 1, len(tangents)):
                angles.append(_angle_between_unit_vectors(tangents[i], tangents[j]))

    if not angles:
        return {
            "Mean Daughter-Daughter Angle (degrees)": (
                "N/A (no bifurcation with two measurable daughters)"
            ),
            "Daughter-Daughter Angle Sample Count": 0,
        }
    return {
        "Mean Daughter-Daughter Angle (degrees)": float(np.mean(angles)),
        "Daughter-Daughter Angle Sample Count": len(angles),
    }


def compute_branch_order_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute mean length, tortuosity, and emergence angle per branch order.

    Returns a dictionary keyed by branch-order label with a compact summary.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)

    by_tag: Dict[str, Dict[str, Any]] = {}
    for item in edge_iter:
        u, v = item[0], item[1]
        data = item[-1]
        normalized_tag = _normalize_branch_order_tag(data.get("branch_order"))
        if not normalized_tag:
            continue

        length = data.get("length", 0.0)
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            continue
        if length_f <= 0:
            continue

        if normalized_tag not in by_tag:
            by_tag[normalized_tag] = _empty_branch_order_record(normalized_tag)
        rec = by_tag[normalized_tag]
        rec["Edge Count"] += 1
        rec["Mean Length (microns)"] += length_f

        if (
            node_positions is not None
            and u in node_positions
            and v in node_positions
        ):
            pos_u = np.array(node_positions[u], dtype=float)
            pos_v = np.array(node_positions[v], dtype=float)
            straight = euclidean(pos_u, pos_v)
            if straight > 0:
                tort = length_f / straight
                if rec["Mean Tortuosity Index"] == "N/A (no position data)":
                    rec["Mean Tortuosity Index"] = 0.0
                rec["Mean Tortuosity Index"] += tort
                rec["Tortuosity Sample Count"] += 1

        # Only present once haemodynamics has run and flow has been solved
        # (haemodynamics.resistance.set_edge_flows). The sign of a single
        # edge's drop is an artefact of its arbitrary (u, v) storage order,
        # not physically meaningful, so this aggregates magnitude -- "how
        # much of the network's total pressure loss happens in this order",
        # the classic answer being mostly small arterioles, not capillaries.
        pressure_drop = data.get("pressure_drop")
        if pressure_drop is not None:
            try:
                drop_abs = abs(float(pressure_drop))
            except (TypeError, ValueError):
                drop_abs = None
            if drop_abs is not None:
                if rec["Mean Pressure Drop (Pa)"] == "N/A (no flow solved)":
                    rec["Mean Pressure Drop (Pa)"] = 0.0
                    rec["Total Pressure Drop (Pa)"] = 0.0
                rec["Mean Pressure Drop (Pa)"] += drop_abs
                rec["Total Pressure Drop (Pa)"] += drop_abs
                rec["Pressure Drop Sample Count"] += 1

        # Only present once haemodynamics has assigned a diameter
        # (haemodynamics.poiseuille), whether measured (FWHM) or from the
        # branch-order table. Caliber distribution by generation is one of
        # the most basic readouts in vascular morphometry, and this data
        # already carries it -- it was just never reported on its own.
        diameter = data.get("diameter_um")
        if diameter is not None:
            try:
                diameter_f = float(diameter)
            except (TypeError, ValueError):
                diameter_f = None
            if diameter_f is not None and diameter_f > 0:
                if rec["Mean Diameter (microns)"] == "N/A (no diameter assigned)":
                    rec["Mean Diameter (microns)"] = 0.0
                rec["Mean Diameter (microns)"] += diameter_f
                rec["_diameter_sum_sq"] += diameter_f * diameter_f
                rec["Diameter Sample Count"] += 1

    for rec in by_tag.values():
        edge_count = int(rec["Edge Count"])
        if edge_count > 0:
            rec["Mean Length (microns)"] = rec["Mean Length (microns)"] / edge_count
        t_samples = int(rec["Tortuosity Sample Count"])
        if t_samples > 0 and isinstance(rec["Mean Tortuosity Index"], (int, float)):
            rec["Mean Tortuosity Index"] = rec["Mean Tortuosity Index"] / t_samples
        elif t_samples == 0:
            rec["Mean Tortuosity Index"] = "N/A (insufficient position data)"
        p_samples = int(rec["Pressure Drop Sample Count"])
        if p_samples > 0 and isinstance(rec["Mean Pressure Drop (Pa)"], (int, float)):
            rec["Mean Pressure Drop (Pa)"] = rec["Mean Pressure Drop (Pa)"] / p_samples
        elif p_samples == 0:
            rec["Mean Pressure Drop (Pa)"] = "N/A (no flow solved)"
            rec["Total Pressure Drop (Pa)"] = "N/A (no flow solved)"

    emergence = compute_emergence_angles_by_branch_order(
        G, tangent_length_um=tangent_length_um
    )
    for tag, emergence_rec in emergence.items():
        if tag not in by_tag:
            by_tag[tag] = _empty_branch_order_record(tag)
        rec = by_tag[tag]
        rec["Mean Emergence Angle (degrees)"] = emergence_rec[
            "Mean Emergence Angle (degrees)"
        ]
        rec["Emergence Angle Sample Count"] = emergence_rec[
            "Emergence Angle Sample Count"
        ]

    # After the emergence merge, not before: a tag with no length>0 edges but
    # a real emergence angle gets its record created for the first time
    # right above, and _diameter_sum_sq (an internal accumulator, never
    # meant to reach the caller) must be dropped from every record, not just
    # the ones the main loop already saw.
    for rec in by_tag.values():
        d_samples = int(rec["Diameter Sample Count"])
        if d_samples > 0 and isinstance(rec["Mean Diameter (microns)"], (int, float)):
            mean_d = rec["Mean Diameter (microns)"] / d_samples
            variance = max(0.0, rec["_diameter_sum_sq"] / d_samples - mean_d * mean_d)
            rec["Mean Diameter (microns)"] = mean_d
            rec["Diameter Coefficient of Variation"] = (
                (variance**0.5) / mean_d if mean_d > 0 else 0.0
            )
        else:
            rec["Mean Diameter (microns)"] = "N/A (no diameter assigned)"
            rec["Diameter Coefficient of Variation"] = "N/A (no diameter assigned)"
        del rec["_diameter_sum_sq"]

    ordered = {
        k: by_tag[k]
        for k in sorted(by_tag.keys(), key=_branch_order_sort_key)
    }
    return ordered


def export_branch_order_statistics_to_csv(
    branch_order_stats: Dict[str, Dict[str, Any]],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export per-branch-order summary statistics to readable CSV."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Branch Order",
                "Edge Count",
                "Mean Length (microns)",
                "Mean Tortuosity Index",
                "Mean Emergence Angle (degrees)",
                "Mean Pressure Drop (Pa)",
                "Total Pressure Drop (Pa)",
                "Mean Diameter (microns)",
                "Diameter Coefficient of Variation",
                "Notes",
            ]
        )
        writer.writerow(
            [
                "# Ordered by vessel class",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Art* first, then BO*, then Ven*.",
            ]
        )
        for branch_tag in sorted(branch_order_stats.keys(), key=_branch_order_sort_key):
            rec = branch_order_stats[branch_tag]
            mean_len = rec.get("Mean Length (microns)", 0.0)
            if isinstance(mean_len, (int, float, np.integer, np.floating)):
                mean_len_s = f"{float(mean_len):.6g}"
            else:
                mean_len_s = str(mean_len)
            mean_tort = rec.get("Mean Tortuosity Index", "N/A")
            if isinstance(mean_tort, (int, float, np.integer, np.floating)):
                mean_tort_s = f"{float(mean_tort):.6g}"
                notes = ["Mean tortuosity is path length / straight distance."]
            else:
                mean_tort_s = str(mean_tort)
                notes = [
                    "Tortuosity unavailable (missing/insufficient node positions)."
                ]
            mean_angle = rec.get(
                "Mean Emergence Angle (degrees)",
                "N/A (no unique parent junction)",
            )
            if isinstance(mean_angle, (int, float, np.integer, np.floating)):
                mean_angle_s = f"{float(mean_angle):.6g}"
                notes.append(
                    "Mean emergence angle is the deflection from the unique "
                    "lower-order parent (0 degrees = collinear)."
                )
            else:
                mean_angle_s = str(mean_angle)
                notes.append(
                    "Emergence angle unavailable (no unique lower-order "
                    "parent junction)."
                )
            mean_drop = rec.get("Mean Pressure Drop (Pa)", "N/A (no flow solved)")
            total_drop = rec.get("Total Pressure Drop (Pa)", "N/A (no flow solved)")
            if isinstance(mean_drop, (int, float, np.integer, np.floating)):
                mean_drop_s = f"{float(mean_drop):.6g}"
                total_drop_s = f"{float(total_drop):.6g}"
                notes.append(
                    "Pressure drop is |pressure_u - pressure_v|, only present "
                    "once flow has been solved; total is this order's share "
                    "of the network's overall pressure loss."
                )
            else:
                mean_drop_s = str(mean_drop)
                total_drop_s = str(total_drop)
                notes.append("Pressure drop unavailable (flow not solved).")
            mean_diam = rec.get("Mean Diameter (microns)", "N/A (no diameter assigned)")
            diam_cv = rec.get(
                "Diameter Coefficient of Variation", "N/A (no diameter assigned)"
            )
            if isinstance(mean_diam, (int, float, np.integer, np.floating)):
                mean_diam_s = f"{float(mean_diam):.6g}"
                diam_cv_s = f"{float(diam_cv):.6g}"
                notes.append(
                    "Diameter coefficient of variation is its standard "
                    "deviation / mean across this order's edges."
                )
            else:
                mean_diam_s = str(mean_diam)
                diam_cv_s = str(diam_cv)
                notes.append("Diameter unavailable (no diameter assigned).")
            writer.writerow(
                [
                    branch_tag,
                    int(rec.get("Edge Count", 0)),
                    mean_len_s,
                    mean_tort_s,
                    mean_angle_s,
                    mean_drop_s,
                    total_drop_s,
                    mean_diam_s,
                    diam_cv_s,
                    " ".join(notes),
                ]
            )

    return output_path
