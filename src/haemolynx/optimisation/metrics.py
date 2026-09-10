"""Pure quality/consistency scoring for skeleton- and graph-optimisation trials.

Every trial :mod:`haemolynx.optimisation.search` runs needs a number (or a
small record of numbers) saying how good the result is. Some of that already
exists elsewhere in the codebase -- :func:`haemolynx.graph.diagnose_degree2_nodes`,
:func:`haemolynx.preprocessing.compute_skeleton_connectivity_stats`, the counts
:func:`haemolynx.graph.smooth_graph_centrelines` already returns -- reused here,
not reimplemented. What is new is bundled into this module: a fragmentation/
defect score for a graph, and a gap-vs-fusion signal for skeleton cleanup, since
neither `graph` nor `preprocessing` has a use for those outside optimisation.

No napari, no Qt, no file I/O: everything here takes an array or a graph and
returns a plain, hashable-friendly dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import cKDTree

from haemolynx.graph import diagnose_degree2_nodes
from haemolynx.preprocessing import braid_factor as _braid_factor_along_axis


def _resolve_connectivity(ndim: int, connectivity: int | None) -> int:
    if connectivity is None:
        return ndim
    return max(1, min(int(connectivity), ndim))


@dataclass(frozen=True)
class GraphTopologyMetrics:
    """A snapshot of how clean a graph's topology is.

    Cheap to compute (no re-labelling of a volume, just graph algorithms), so
    this is the metric the reconnect-threshold and cluster-collapse trials
    score against.
    """

    n_nodes: int
    n_edges: int
    n_components: int
    n_selfloops: int
    n_isolates: int
    total_degree2: int

    @property
    def score(self) -> float:
        """Lower is better. Components and self-loops are the costly defects:
        a surviving self-loop after ``build_graph_from_skeleton`` (which already
        removes self-connected nodes internally) signals a genuinely bad
        reconnection, not noise, so it is weighted heavily."""
        return (
            float(max(self.n_components - 1, 0))
            + 5.0 * float(self.n_selfloops)
            + 0.5 * float(self.n_isolates)
        )


def graph_topology_metrics(G: nx.Graph) -> GraphTopologyMetrics:
    """Fragmentation/defect counts for *G*."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return GraphTopologyMetrics(0, 0, 0, 0, 0, 0)
    n_components = nx.number_connected_components(G)
    n_selfloops = nx.number_of_selfloops(G)
    n_isolates = sum(1 for _ in nx.isolates(G))
    total_degree2 = int(diagnose_degree2_nodes(G)["total_degree2"])
    return GraphTopologyMetrics(
        n_nodes=n_nodes,
        n_edges=G.number_of_edges(),
        n_components=n_components,
        n_selfloops=n_selfloops,
        n_isolates=n_isolates,
        total_degree2=total_degree2,
    )


@dataclass(frozen=True)
class GapFusionSignal:
    """Whether cleaning a skeleton sealed real gaps, or fused distinct vessels.

    A candidate that merges two raw components separated by much more than a
    typical vessel diameter is more likely fusing two distinct vessels than
    sealing a real gap in one -- that is what ``largest_gap_bridged_ratio`` is
    for; the closing-radius and gap-bridging trials reject any candidate whose
    ratio crosses a guard threshold.
    """

    components_before: int
    components_after: int
    largest_gap_bridged_um: float
    largest_gap_bridged_ratio: float

    @property
    def components_merged(self) -> int:
        return max(0, self.components_before - self.components_after)


def gap_vs_fusion_signal(
    raw_skeleton: np.ndarray,
    cleaned_skeleton: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    component_connectivity: int | None = None,
    typical_radius_um: float = 1.0,
) -> GapFusionSignal:
    """How far apart were the raw components that ended up merged in *cleaned*?

    Labels both skeletons, finds which raw components a cleaning step actually
    touched (the voxels *cleaned* has that *raw* did not -- closing/bridging
    additions), and for every pair of raw components touched by the same
    addition, measures how far apart they originally were. ``largest_gap_bridged_um``
    is the worst such merge, 0.0 if nothing was merged.
    """
    raw = np.asarray(raw_skeleton, dtype=bool)
    cleaned = np.asarray(cleaned_skeleton, dtype=bool)
    conn = _resolve_connectivity(raw.ndim, component_connectivity)
    structure = generate_binary_structure(raw.ndim, conn)
    raw_labeled, n_before = label(raw, structure=structure)
    _, n_after = label(cleaned, structure=structure)

    if n_before <= 1 or not raw.any():
        return GapFusionSignal(int(n_before), int(n_after), 0.0, 0.0)

    added = cleaned & ~raw
    if not added.any():
        # Cleaning only removed voxels (or changed none): it drew no bridge,
        # so it cannot have merged components.
        return GapFusionSignal(int(n_before), int(n_after), 0.0, 0.0)

    spacing = np.asarray(voxel_size_zyx, dtype=float)
    raw_coords = np.argwhere(raw)
    raw_labels_at = raw_labeled[tuple(raw_coords.T)]
    raw_physical = raw_coords * spacing
    tree = cKDTree(raw_physical)

    added_physical = np.argwhere(added) * spacing
    _, nearest_idx = tree.query(added_physical)
    touched_labels = sorted(set(int(v) for v in raw_labels_at[nearest_idx]))

    if len(touched_labels) < 2:
        return GapFusionSignal(int(n_before), int(n_after), 0.0, 0.0)

    per_component_physical = {
        lbl: raw_physical[raw_labels_at == lbl] for lbl in touched_labels
    }
    per_component_trees = {
        lbl: cKDTree(coords) for lbl, coords in per_component_physical.items()
    }

    largest_gap = 0.0
    for i, lbl_a in enumerate(touched_labels):
        for lbl_b in touched_labels[i + 1:]:
            dists, _ = per_component_trees[lbl_b].query(per_component_physical[lbl_a])
            largest_gap = max(largest_gap, float(dists.min()))

    ratio = largest_gap / max(2.0 * float(typical_radius_um), 1e-6)
    return GapFusionSignal(
        components_before=int(n_before),
        components_after=int(n_after),
        largest_gap_bridged_um=largest_gap,
        largest_gap_bridged_ratio=ratio,
    )


@dataclass(frozen=True)
class SmoothingQuality:
    """How much of the graph a centreline-smoothing trial actually smoothed.

    Built directly from :func:`haemolynx.graph.smooth_graph_centrelines`'s own
    returned counts -- no new measurement of the graph itself, just a score
    over numbers that already exist.
    """

    smoothed: int
    relaxed: int
    kept_raw: int
    too_short: int
    length_before_um: float
    length_after_um: float

    @property
    def total(self) -> int:
        return self.smoothed + self.relaxed + self.kept_raw + self.too_short

    @property
    def smoothed_fraction(self) -> float:
        return (self.smoothed + self.relaxed) / self.total if self.total else 0.0

    @property
    def length_shrink_fraction(self) -> float:
        if self.length_before_um <= 0:
            return 0.0
        return 1.0 - (self.length_after_um / self.length_before_um)


def smoothing_quality(
    counts: Mapping[str, int], length_before_um: float, length_after_um: float
) -> SmoothingQuality:
    return SmoothingQuality(
        smoothed=int(counts.get("smoothed", 0)),
        relaxed=int(counts.get("relaxed", 0)),
        kept_raw=int(counts.get("kept_raw", 0)),
        too_short=int(counts.get("too_short", 0)),
        length_before_um=float(length_before_um),
        length_after_um=float(length_after_um),
    )


def total_edge_length(G: nx.Graph) -> float:
    """Sum of every edge's ``length`` attribute, 0.0 for edges missing one."""
    return float(sum(d.get("length", 0.0) or 0.0 for _, _, d in G.edges(data=True)))


def braid_factor_along_long_axis(skeleton: np.ndarray) -> float:
    """:func:`haemolynx.preprocessing.braid_factor`, checked along the
    skeleton's own longest axis of extent.

    The thick-vessel refinement settings (wall absorption, flake filter, the
    two bridge caps, bridge-radius smoothing) exist specifically to stop
    ``skeletonize_thickness_gated`` leaving a medial *sheet* -- several
    parallel strands -- instead of one centreline through the fat region.
    ``braid_factor`` is already the codebase's own measure of that (~1 for a
    single line, higher for a sheet), but it needs an axis running roughly
    along the vessel's own direction of travel: checked across that
    direction instead, even a single clean line reads as heavily braided,
    because nearly every voxel then falls into the same one or two slices
    (counted the other way -- see the module docstring's fixture, "axis 2 is
    the trunk axis"). A fixture-specific caller knows its own trunk axis; an
    optimiser does not, so this measures the skeleton's own bounding-box
    extent and uses whichever axis it is longest along.
    """
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    if not skeleton_bool.any():
        return 0.0
    coords = np.argwhere(skeleton_bool)
    extents = coords.max(axis=0) - coords.min(axis=0)
    long_axis = int(np.argmax(extents))
    return _braid_factor_along_axis(skeleton_bool, axis=long_axis)
