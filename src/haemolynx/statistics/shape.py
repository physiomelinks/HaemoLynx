"""Geometric/spatial vessel-network measures: shape, density, and spacing."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

from haemolynx.visualization.geometry import edge_polyline

from ._sampling import REPRODUCIBILITY_SEED

#: Above this many unique node pairs, path efficiency is estimated from a
#: bounded random sample of pairs instead of every one, to avoid a runtime
#: quadratic in node count on a large network.
DEFAULT_PATH_EFFICIENCY_MAX_PAIRS = 5000


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


def _box_counting_fractal_dimension(points: np.ndarray) -> float:
    """Box-counting fractal-dimension estimate for a physical point cloud.

    Returns 0.0 -- the same fallback already used below when the box counts
    do not admit a fit -- for a degenerate cloud with no spatial extent
    (every point coincident, or numerically indistinguishable), rather than
    feeding a zero range into log-spaced box sizes: `np.log10(0)` is `-inf`,
    which propagates to non-finite box sizes and leaves `np.polyfit` fitting
    a degenerate system, raising `LinAlgError: SVD did not converge`.
    """
    max_range = np.max(points.max(axis=0) - points.min(axis=0))
    if not np.isfinite(max_range) or max_range <= 0:
        return 0.0
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
    max_pairs: Optional[int] = DEFAULT_PATH_EFFICIENCY_MAX_PAIRS,
    rng_seed: int = REPRODUCIBILITY_SEED,
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
