"""Per-setting candidate-value generation, derived from the segmented image.

Every function here looks at a real measurement of the mask, skeleton, or
graph in front of it -- an inscribed-radius distribution, an inter-component
gap distribution, a terminal-edge-length distribution -- and turns it into a
short list of candidate values worth actually trying. None of these functions
run the pipeline or score anything; :mod:`.search` does that, one setting (or,
for the settings too small to search independently, one short joint sweep) at
a time, so the total number of real preprocessing/graph-building calls stays
additive across settings rather than the product of every group's grid.

Every candidate list always includes the schema default, and every function
degrades to ``[default]`` on pathological input (an empty mask, a single
component, no gaps) rather than raising -- ``np.percentile`` of an empty array
raises, and a settings optimiser must never crash on a bad image, it should
just fall back to leaving that setting alone.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

from haemolynx.preprocessing import inscribed_radius_map, needs_thick_vessel_treatment

#: Below this inscribed radius, thickness-gated skeletonisation cannot matter
#: for any candidate this module would ever propose -- used only to decide
#: whether the thick-vessel group is worth searching at all.
_THICK_VESSEL_GUARD_RADIUS_UM = 2.0

_26_NEIGHBOUR_KERNEL = np.ones((3, 3, 3), dtype=int)
_26_NEIGHBOUR_KERNEL[1, 1, 1] = 0


def _percentiles(values: np.ndarray, percentiles: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return []
    return [float(np.percentile(values, p)) for p in percentiles]


def _clip_round(values: set[float], *, lo: float, hi: float, as_int: bool) -> list:
    kept = sorted(v for v in values if lo <= v <= hi)
    if as_int:
        return sorted({int(round(v)) for v in kept})
    return kept


# ---------------------------------------------------------------------------
# Group 1: thick-vessel gating
# ---------------------------------------------------------------------------
def thick_vessel_worth_checking(
    raw_mask: np.ndarray, voxel_size_zyx: tuple[float, float, float]
) -> bool:
    """Whether the mask has anything fat enough for thickness gating to matter."""
    return needs_thick_vessel_treatment(
        raw_mask, min_radius_um=_THICK_VESSEL_GUARD_RADIUS_UM, voxel_size_zyx=voxel_size_zyx
    )


def thick_vessel_min_radius_candidates(
    raw_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    default: float,
) -> list[float]:
    """Percentiles of the mask's own inscribed-radius distribution.

    The setting's meaning *is* "radius above which a region counts as fat", so
    candidates come directly from percentiles of that same measurement rather
    than an arbitrary multiplier grid.
    """
    radius_map = inscribed_radius_map(raw_mask, voxel_size_zyx)
    nonzero = radius_map[radius_map > 0]
    candidates = {float(default)} | set(_percentiles(nonzero, (75.0, 90.0, 95.0, 99.0)))
    return sorted(v for v in candidates if v > 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 2: min branch length
# ---------------------------------------------------------------------------
def min_branch_length_candidates(
    component_sizes: tuple[int, ...], default: int, ceiling: int = 20
) -> list[int]:
    """Percentiles of the raw skeleton's own per-component voxel-count distribution."""
    sizes = np.asarray(component_sizes, dtype=float)
    candidates = {0, int(default)} | set(
        int(round(v)) for v in _percentiles(sizes, (5.0, 10.0, 25.0))
    )
    return sorted(v for v in candidates if 0 <= v <= ceiling) or [int(default)]


# ---------------------------------------------------------------------------
# Group 3: bundle refinement
# ---------------------------------------------------------------------------
def bundle_scan_size_candidates(
    raw_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    default: int,
) -> list[int]:
    """Odd window sizes bracketing the mask's own median vessel radius."""
    radius_map = inscribed_radius_map(raw_mask, voxel_size_zyx)
    nonzero = radius_map[radius_map > 0]
    if nonzero.size == 0:
        return [int(default)]
    median_radius_um = float(np.median(nonzero))
    spacing = min(float(v) for v in voxel_size_zyx) or 1.0
    median_radius_voxels = median_radius_um / spacing
    candidates = {int(default)}
    for multiplier in (2.0, 4.0, 6.0):
        size = int(round(median_radius_voxels * multiplier))
        size += 1 - (size % 2)  # nearest odd
        candidates.add(size)
    return sorted(v for v in candidates if 5 <= v <= 25) or [int(default)]


def bundle_density_fraction_candidates(
    raw_mask: np.ndarray,
    scan_size: int,
    default: float,
) -> list[float]:
    """Percentiles of the local-density map at the chosen scan size."""
    from scipy.ndimage import uniform_filter

    density = uniform_filter(raw_mask.astype(float), size=scan_size)
    nonzero = density[density > 0]
    candidates = {float(default)} | set(_percentiles(nonzero, (60.0, 75.0, 90.0)))
    return sorted(v for v in candidates if 0.0 < v <= 1.0) or [float(default)]


def estimate_bundle_max_connections(raw_skeleton: np.ndarray) -> int:
    """Observed worst-case junction fan-out: the busiest voxel's neighbour count.

    A single derived value rather than a swept candidate -- like
    ``bundle_hub_min_spacing``, this keeps the bundle-refinement group to two
    short sweeps (scan size, then density fraction) instead of a four-way grid.
    """
    skeleton_bool = np.asarray(raw_skeleton, dtype=bool)
    if not skeleton_bool.any():
        return 8
    neighbour_counts = convolve(
        skeleton_bool.astype(int), _26_NEIGHBOUR_KERNEL, mode="constant"
    )
    observed_max = int(neighbour_counts[skeleton_bool].max())
    return max(4, min(observed_max, 12))


def bundle_hub_min_spacing_for(scan_size: int) -> int:
    """Derived, not searched: half the scan window, floor of 1."""
    return max(1, scan_size // 2)


# ---------------------------------------------------------------------------
# Group 4: closing radius
# ---------------------------------------------------------------------------
def closing_radius_candidates(
    gap_distances_voxels: np.ndarray, default: int, cap: int = 5
) -> list[int]:
    candidates = {0, 1, 2, int(default)}
    if gap_distances_voxels.size:
        candidates.add(int(np.ceil(np.percentile(gap_distances_voxels, 25))))
    return sorted(v for v in candidates if 0 <= v <= cap) or [int(default)]


# ---------------------------------------------------------------------------
# Group 5: gap bridging
# ---------------------------------------------------------------------------
def bridge_gap_size_candidates(
    gap_distances_voxels: np.ndarray, default: int
) -> list[int]:
    candidates = {0, int(default)} | set(
        int(round(v)) for v in _percentiles(gap_distances_voxels, (10.0, 25.0))
    )
    return sorted(v for v in candidates if v >= 0) or [int(default)]


def max_bridge_distance_candidates(
    gap_distances_voxels: np.ndarray, default: int
) -> list[int]:
    candidates = {int(default)} | set(
        int(round(v)) for v in _percentiles(gap_distances_voxels, (50.0, 75.0, 90.0))
    )
    return sorted(v for v in candidates if v >= 0) or [int(default)]


# ---------------------------------------------------------------------------
# Group 6: component connectivity + minimum component percent
# ---------------------------------------------------------------------------
def component_connectivity_candidates() -> list[int]:
    return [1, 2, 3]


def min_component_percent_candidates(
    component_sizes: tuple[int, ...], voxel_count: int, default: float
) -> list[float]:
    """Percent candidates from the sharpest drop in the sorted component sizes.

    ``component_sizes`` is largest-first (as
    :func:`haemolynx.preprocessing.compute_skeleton_connectivity_stats`
    returns it). The elbow is the pair of consecutive sizes with the biggest
    ratio -- everything past it looks like noise relative to what is kept.
    """
    candidates = {0.0, float(default), 1.0, 5.0}
    sizes = list(component_sizes)
    if len(sizes) >= 2 and voxel_count > 0:
        ratios = [
            sizes[i] / sizes[i + 1] if sizes[i + 1] > 0 else float("inf")
            for i in range(len(sizes) - 1)
        ]
        elbow = int(np.argmax(ratios))
        threshold_size = (sizes[elbow] + sizes[elbow + 1]) / 2.0
        candidates.add(round(100.0 * threshold_size / voxel_count, 4))
    return sorted(v for v in candidates if 0.0 <= v <= 100.0)


# ---------------------------------------------------------------------------
# Group 7: reconnect thresholds
# ---------------------------------------------------------------------------
def reconnect_threshold_candidates(
    gap_distances_um: np.ndarray, default: float
) -> list[float]:
    candidates = {float(default)} | set(_percentiles(gap_distances_um, (25.0, 50.0, 75.0, 90.0)))
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def orphan_threshold_candidates(
    reconnect_threshold: float, default: float
) -> list[float]:
    """Fractions of the winning reconnect threshold, mirroring the schema
    defaults' own ~0.3 ratio (``final_orphan_reconnect_threshold=3.0`` against
    ``graph_reconnect_threshold=10.0``)."""
    candidates = {float(default)} | {
        round(reconnect_threshold * ratio, 6) for ratio in (0.25, 0.5, 1.0)
    }
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 8: cluster collapse distance
# ---------------------------------------------------------------------------
def cluster_collapse_distance_candidates(
    node_gap_distances_um: np.ndarray, default: float
) -> list[float]:
    candidates = {float(default)} | set(
        _percentiles(node_gap_distances_um, (5.0, 10.0, 25.0))
    )
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 9: minimum stub length
# ---------------------------------------------------------------------------
def min_stub_length_candidates(
    terminal_edge_lengths_um: np.ndarray, default: float
) -> list[float]:
    candidates = {float(default)} | set(
        _percentiles(terminal_edge_lengths_um, (10.0, 25.0, 40.0))
    )
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 10: centreline smoothing
# ---------------------------------------------------------------------------
def centreline_smoothing_method_candidates() -> list[str]:
    return ["taubin", "chaikin"]


def centreline_smoothing_iterations_candidates(method: str) -> list[int]:
    """Iteration counts to try, method-dependent.

    Taubin re-averages a fixed number of points each pass, so cost is linear
    in iterations. Chaikin's corner-cutting *doubles* the point count every
    pass (``_chaikin_once`` in ``graph/smoothing.py``), so the same {5,10,15,20}
    grid that is cheap for Taubin is a 2**20-fold blow-up for Chaikin -- its
    own function default is 2, and a handful of passes already gets close to
    the limit curve, so candidates stay small.
    """
    if method == "chaikin":
        return [1, 2, 3, 4]
    return [5, 10, 15, 20]


def centreline_max_deviation_candidates(
    voxel_size_zyx: tuple[float, float, float], default: float
) -> list[float]:
    one_voxel_um = min(float(v) for v in voxel_size_zyx) or 1.0
    candidates = {float(default)} | {round(one_voxel_um * m, 6) for m in (0.5, 1.0, 2.0, 4.0)}
    return sorted(v for v in candidates if v > 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Graph measurements feeding groups 7-9 -- the graph-side analogue of
# haemolynx.preprocessing.inter_component_gap_distances, since a graph's
# ``pos`` is already physical microns and has no array to label.
# ---------------------------------------------------------------------------
def graph_component_gap_distances_um(G: nx.Graph) -> np.ndarray:
    """Nearest-node distance between every pair of *G*'s connected components."""
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return np.array([], dtype=float)

    positions = nx.get_node_attributes(G, "pos")
    trees: list[cKDTree] = []
    point_sets: list[np.ndarray] = []
    for component in components:
        points = np.asarray(
            [positions[n] for n in component if n in positions], dtype=float
        )
        if points.size == 0:
            continue
        point_sets.append(points)
        trees.append(cKDTree(points))

    distances: list[float] = []
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            dists, _ = trees[j].query(point_sets[i])
            distances.append(float(dists.min()))
    return np.sort(np.asarray(distances, dtype=float))


def nearest_neighbour_node_distances_um(G: nx.Graph) -> np.ndarray:
    """Every node's distance to the nearest *other* node, regardless of edges.

    Feeds the cluster-collapse-distance candidates: a near-duplicate pair of
    junction nodes sits close together whether or not an edge already joins
    them, which is exactly what that setting merges.
    """
    positions = nx.get_node_attributes(G, "pos")
    if len(positions) < 2:
        return np.array([], dtype=float)
    nodes = list(positions)
    points = np.asarray([positions[n] for n in nodes], dtype=float)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return np.sort(dists[:, 1])


def terminal_edge_lengths_um(G: nx.Graph) -> np.ndarray:
    """Length of every edge touching a degree-1 (terminal) node."""
    degrees = dict(G.degree())
    lengths: list[float] = []
    for u, v, data in G.edges(data=True):
        if degrees.get(u) == 1 or degrees.get(v) == 1:
            length = data.get("length")
            if length is not None:
                lengths.append(float(length))
    return np.asarray(lengths, dtype=float)
