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


#: Box sizes tried between the smallest and the largest (log-spaced).
_FRACTAL_BOX_SIZES = 16
#: The fewest boxes a size may be covered by and still count as scaling:
#: fewer, and where the grid happens to fall decides the count.
_FRACTAL_MIN_BOXES = 10
#: Grid offsets tried at each box size, as fractions of a box; the fewest
#: boxes over them is the count (the usual guard against the grid's
#: alignment).
_FRACTAL_GRID_OFFSETS = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.75, 0.5], [0.75, 0.25, 0.25]]
)
#: The centreline is sampled this often (um) for its box count.
FRACTAL_CENTRELINE_STEP_UM = 1.0


def _occupied_boxes(points: np.ndarray, size: float) -> int:
    """The fewest boxes of edge *size* that cover *points*, over the grid
    offsets in :data:`_FRACTAL_GRID_OFFSETS`."""
    origin = points.min(axis=0)
    fewest = None
    for offset in _FRACTAL_GRID_OFFSETS[:, : points.shape[1]]:
        index = np.floor((points - origin) / size + offset).astype(np.int64)
        index -= index.min(axis=0)
        span = index.max(axis=0) + 1
        keys = index[:, 0]
        for axis in range(1, index.shape[1]):
            keys = keys * span[axis] + index[:, axis]
        count = int(np.unique(keys).size)
        fewest = count if fewest is None else min(fewest, count)
    return int(fewest)


def box_counting_fractal_dimension(
    points: np.ndarray, *, sampling_step_um: Optional[float] = None
) -> Dict[str, Any]:
    """Box-counting dimension of a physical point cloud, and how well it fits.

    Box sizes run from twice the cloud's sampling step (*sampling_step_um*,
    else the median distance between neighbouring points) -- below that the
    count measures how the points were sampled, not the network -- to half
    its extent, and only the sizes still covered by at least
    :data:`_FRACTAL_MIN_BOXES` boxes are fitted. Counted with ``np.unique`` on
    integer box keys, the fewest over a few grid offsets.

    Returns ``dimension`` (the fitted slope), ``r_squared`` of the log-log
    fit, the ``box_min_um``/``box_max_um`` fitted over and how many ``sizes``;
    a cloud with no extent, or too few box sizes to fit a line to, has
    ``dimension`` 0.0 and ``r_squared`` NaN.
    """
    none = {"dimension": 0.0, "r_squared": float("nan"), "box_min_um": float("nan"),
            "box_max_um": float("nan"), "sizes": 0}
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return none
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if not np.isfinite(extent) or extent <= 0:
        return none
    if sampling_step_um is None:
        gaps, _ = cKDTree(points).query(points, k=2)
        gaps = gaps[:, 1][gaps[:, 1] > 0]
        sampling_step_um = float(np.median(gaps)) if gaps.size else extent
    smallest = max(2.0 * float(sampling_step_um), extent / 1000.0)
    largest = extent / 2.0
    if not smallest < largest:
        return none
    sizes = np.logspace(np.log10(smallest), np.log10(largest), _FRACTAL_BOX_SIZES)
    counts = np.array([_occupied_boxes(points, size) for size in sizes], dtype=float)
    scaling = counts >= _FRACTAL_MIN_BOXES
    if int(scaling.sum()) < 2:
        return none
    sizes, counts = sizes[scaling], counts[scaling]
    log_s, log_n = np.log(sizes), np.log(counts)
    slope, intercept = np.polyfit(log_s, log_n, 1)
    residual = log_n - (slope * log_s + intercept)
    spread = float(np.sum((log_n - log_n.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / spread if spread > 0 else float("nan")
    return {
        "dimension": float(-slope),
        "r_squared": r_squared,
        "box_min_um": float(sizes.min()),
        "box_max_um": float(sizes.max()),
        "sizes": int(sizes.size),
    }


def _box_counting_fractal_dimension(points: np.ndarray) -> float:
    """:func:`box_counting_fractal_dimension`'s dimension alone; 0.0 for a
    degenerate cloud (every point coincident)."""
    return box_counting_fractal_dimension(points)["dimension"]


def _centreline_points(
    G: Union[nx.Graph, nx.MultiGraph], *, step_um: Optional[float] = None
) -> np.ndarray:
    """Every point along every edge's real centreline, concatenated.

    Reads each edge's ``voxels`` polyline (the smoothed centreline, where
    available) via the same `edge_polyline` the vessel-tube drawing and the
    emergence-angle tangent both use, so this is the network's actual
    physical shape -- not just the branch/terminal points left after
    topology simplification. With *step_um*, each polyline is resampled
    evenly at that step (see :func:`haemolynx.geometry.resample_at_step`).
    """
    from haemolynx.geometry import resample_at_step

    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)
    chunks: list[np.ndarray] = []
    for item in edge_iter:
        u, v = item[0], item[1]
        data = item[-1]
        try:
            polyline = edge_polyline(G, u, v, data)
        except (TypeError, ValueError):
            continue
        chunks.append(resample_at_step(polyline, step_um) if step_um else polyline)
    if not chunks:
        return np.empty((0, 3))
    return np.concatenate(chunks, axis=0)


def _fractal_entries(name: str, fit: Dict[str, Any]) -> Dict[str, Any]:
    if not np.isfinite(fit["r_squared"]):
        return {name: fit["dimension"], f"{name} R^2": "N/A (no scaling range to fit)"}
    return {
        name: fit["dimension"],
        f"{name} R^2": fit["r_squared"],
        f"{name} Box Sizes (microns)": f"{fit['box_min_um']:.3g}-{fit['box_max_um']:.3g}",
    }


def compute_fractal_dimension(
    G: Union[nx.Graph, nx.MultiGraph], node_positions: Optional[dict]
) -> Dict[str, Any]:
    """Compute two box-counting fractal-dimension estimates.

    "Fractal Dimension (Node Positions)" counts only the branch/terminal
    points left after topology simplification -- fast, but blind to the
    vessel's actual path shape between junctions, so it understates spatial
    complexity. "Fractal Dimension (Centreline)" counts points every
    :data:`FRACTAL_CENTRELINE_STEP_UM` along every edge's real centreline
    instead, matching standard vascular fractal-dimension methodology. The
    two are not expected to agree; both are reported explicitly rather than
    folded into one "Fractal Dimension" value.

    Each comes with the R^2 of its log-log fit and the box sizes it was fitted
    over (see :func:`box_counting_fractal_dimension`): a dimension from a
    range where the network does not scale -- a fixed range used to be fitted
    whatever the counts did -- says little, and the R^2 is what shows it.
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
            result.update(
                _fractal_entries(
                    "Fractal Dimension (Node Positions)",
                    box_counting_fractal_dimension(node_points),
                )
            )

    centreline_points = _centreline_points(G, step_um=FRACTAL_CENTRELINE_STEP_UM)
    if len(centreline_points) < 2:
        result["Fractal Dimension (Centreline)"] = "N/A (insufficient position data)"
    else:
        result.update(
            _fractal_entries(
                "Fractal Dimension (Centreline)",
                box_counting_fractal_dimension(
                    centreline_points, sampling_step_um=FRACTAL_CENTRELINE_STEP_UM
                ),
            )
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

        # The same sampled pairs, searched once per distinct source rather
        # than once per pair: the per-pair loop took 29 s on a 2,203-node
        # network, longer than the exact mean below.
        by_source: dict[int, list[int]] = {}
        for i, j in sampled_pairs:
            by_source.setdefault(i, []).append(j)
        adjacency = _length_adjacency(G_s, nodes)
        for sources, distances in _distances_from(adjacency, sorted(by_source)):
            for row, source in enumerate(sources):
                reached = distances[row, by_source[source]]
                if not np.all(np.isfinite(reached)):
                    raise RuntimeError(
                        f"No path between connected-graph node {nodes[source]} and a sampled partner"
                    )
                path_lengths.extend(float(value) for value in reached)
    else:
        avg_path_length = _mean_shortest_path_length_over_all_pairs(G_s, nodes)
        efficiency = 1 / avg_path_length if avg_path_length > 0 else 0
        return {
            "Path Efficiency": efficiency,
            "Average Shortest Path Length (microns)": avg_path_length,
            "Path Efficiency Pair Sample Size": total_pairs,
            "Path Efficiency Pair Coverage": 1.0 if total_pairs > 0 else 0,
        }

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


#: Sources per batch of the all-pairs search: bounds the distance block held
#: at once to this many rows of the node count.
_ALL_PAIRS_SOURCE_BATCH = 256


def _mean_shortest_path_length_over_all_pairs(G_s: nx.Graph, nodes: list) -> float:
    """Mean ``length``-weighted shortest path over every unordered node pair.

    One single-source search per node, in compiled code, rather than one
    search per *pair*: a pair-at-a-time loop ran ~4 million Dijkstra searches
    on a 2,800-node network, which kept Export busy for hours in ``full``
    statistics mode. The same answer, *G_s* being connected. A missing
    ``length`` counts as 1, as NetworkX's own weighted search does.
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    total = 0.0
    for sources, distances in _distances_from(_length_adjacency(G_s, nodes), range(n)):
        for row, source in enumerate(sources):
            later = distances[row, source + 1 :]
            if not np.all(np.isfinite(later)):
                raise RuntimeError(
                    f"No path from connected-graph node {nodes[source]} to every other node"
                )
            total += float(later.sum())
    return total / (n * (n - 1) / 2)


def _length_adjacency(G_s: nx.Graph, nodes: list):
    """*G_s* as a sparse ``length`` matrix over *nodes* (their list positions).

    The lightest of any parallel edges, no self-loops, and a missing
    ``length`` as 1, as NetworkX's own weighted search reads it.
    """
    from scipy.sparse import csr_matrix

    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    lightest: dict[tuple[int, int], float] = {}
    for u, v, data in G_s.edges(data=True):
        i, j = sorted((index[u], index[v]))
        if i == j:
            continue
        weight = float(data.get("length", 1))
        if (i, j) not in lightest or weight < lightest[(i, j)]:
            lightest[(i, j)] = weight
    rows, cols = zip(*lightest) if lightest else ((), ())
    # Explicit zero-length edges stay edges in a sparse graph built this way.
    return csr_matrix(
        (np.fromiter(lightest.values(), dtype=float), (rows, cols)), shape=(n, n)
    )


def _distances_from(adjacency, sources):
    """``(batch, distances)`` per batch of *sources*: one search per source."""
    from scipy.sparse.csgraph import dijkstra

    sources = np.asarray(list(sources), dtype=int)
    for start in range(0, len(sources), _ALL_PAIRS_SOURCE_BATCH):
        batch = sources[start : start + _ALL_PAIRS_SOURCE_BATCH]
        yield batch, dijkstra(adjacency, directed=False, indices=batch)


def compute_vessel_density(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict],
    voxel_size,
    image_dimensions,
    is_multigraph: bool,
    tissue_volume_um3: Optional[float] = None,
    vessel_volume_um3: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute vessel density.

    "In Tissue" divides by *tissue_volume_um3* when it is given -- the tissue
    measured from the raw image (:mod:`~haemolynx.statistics.tissue_volume`) --
    and otherwise by the box the network's nodes span, which counts any empty
    space inside that box as tissue. *vessel_volume_um3*, the segmented vessel
    volume inside the tissue, adds the vascular volume fraction.
    """
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
    if tissue_volume_um3 is not None:
        tissue_volume = float(tissue_volume_um3)
        out["Tissue Volume (micron³)"] = tissue_volume
        out["Vessel Density in Tissue (microns/micron³)"] = (
            total_length / tissue_volume if tissue_volume > 0 else 0
        )
        out["Tissue Volume Source"] = "tissue surface measured from the raw image"
        if vessel_volume_um3 is not None:
            out["Vessel Volume in Tissue (micron³)"] = float(vessel_volume_um3)
            out["Vascular Volume Fraction in Tissue"] = (
                float(vessel_volume_um3) / tissue_volume if tissue_volume > 0 else 0
            )
    elif "Vessel-Occupied Volume (micron³)" in out:
        out["Tissue Volume Source"] = "bounding box of the network's nodes"

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


#: Spacing of the points each vessel is sampled at for the intercapillary
#: distance, in microns: fine next to a spacing of tens of microns, coarse
#: enough to keep a whole network's points few.
INTERCAPILLARY_SAMPLE_STEP_UM = 2.0

#: How many nearest points each query looks through for one on an unrelated
#: vessel, widening to the next when none of them is.
_INTERCAPILLARY_NEIGHBOUR_COUNTS = (64, 512, 4096)


def compute_intercapillary_distance(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    sample_step_um: float = INTERCAPILLARY_SAMPLE_STEP_UM,
) -> Dict[str, Any]:
    """Spacing between non-adjacent vessels.

    Each edge is sampled every *sample_step_um* along its centreline
    (``voxels``, or its two node positions when that is all there is), each
    sample's distance to the nearest point of any *other* edge that does not
    share a node with it is taken -- a shared node is trivially close at the
    junction itself -- and the edge's value is the median of those distances
    along it. Reported as the mean and median over edges. This is the key
    input to Krogh-cylinder oxygen-diffusion modelling: how far the tissue
    between two vessels reaches.

    The median along the vessel, not its closest approach: a vessel's nearest
    unrelated neighbour is usually a vessel two junctions away, met right at
    the junction, so the closest approach ran low for almost every vessel.
    Samples evenly spaced in microns, not the centreline's own vertices, so a
    stretch along a coarse z axis counts for its length.
    """
    from haemolynx.geometry import resample_at_step

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
        edge_points.append((resample_at_step(points, sample_step_um), u, v))
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
    neighbour_counts = [k for k in _INTERCAPILLARY_NEIGHBOUR_COUNTS if k < all_points.shape[0]]
    neighbour_counts.append(all_points.shape[0])

    distances: list[float] = []
    for i, (points, u, v) in enumerate(edge_points):
        forbidden = np.fromiter(
            {i} | incident.get(u, set()) | incident.get(v, set()), dtype=np.intp
        )
        nearest = np.full(points.shape[0], np.nan)
        pending = np.arange(points.shape[0])
        for k in neighbour_counts:
            dists, idxs = tree.query(points[pending], k=k)
            dists = np.asarray(dists, dtype=float).reshape(len(pending), -1)
            idxs = np.asarray(idxs).reshape(len(pending), -1)
            allowed = ~np.isin(point_edge_idx[idxs], forbidden)
            found = allowed.any(axis=1)
            # Results come nearest first: the first allowed one is the nearest.
            first = np.argmax(allowed, axis=1)
            nearest[pending[found]] = dists[found, first[found]]
            pending = pending[~found]
            if pending.size == 0:
                break
        if np.isfinite(nearest).any():
            distances.append(float(np.nanmedian(nearest)))

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
