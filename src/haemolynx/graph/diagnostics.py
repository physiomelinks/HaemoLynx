"""Diagnostics utilities for graph topology cleanup."""
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np

from ._helpers import get_all_edge_data


def diagnose_degree2_nodes(
    G: Union[nx.Graph, nx.MultiGraph],
    max_degree: int = 4,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    """Summarize remaining degree-2 nodes and why smart cleanup may skip them."""
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    if sample_limit < 1:
        raise ValueError("sample_limit must be >= 1")

    degree2_nodes: List[Any] = [n for n in G.nodes() if G.degree[n] == 2]
    reason_nodes: Dict[str, List[Any]] = {
        "neighbors_not_2": [],
        "high_degree_neighbor": [],
        "missing_pos": [],
        "missing_edge_data": [],
        "eligible_for_smart_removal": [],
    }

    for node in degree2_nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) != 2:
            reason_nodes["neighbors_not_2"].append(node)
            continue

        n1, n2 = neighbors
        if G.degree[n1] >= max_degree or G.degree[n2] >= max_degree:
            reason_nodes["high_degree_neighbor"].append(node)
            continue

        node_pos = G.nodes[node].get("pos")
        n1_pos = G.nodes[n1].get("pos")
        n2_pos = G.nodes[n2].get("pos")
        if node_pos is None or n1_pos is None or n2_pos is None:
            reason_nodes["missing_pos"].append(node)
            continue

        edge1_data_list = get_all_edge_data(G, node, n1)
        edge2_data_list = get_all_edge_data(G, node, n2)
        if not edge1_data_list or not edge2_data_list:
            reason_nodes["missing_edge_data"].append(node)
            continue

        reason_nodes["eligible_for_smart_removal"].append(node)

    reason_counts = {k: len(v) for k, v in reason_nodes.items()}
    reason_examples = {k: v[:sample_limit] for k, v in reason_nodes.items() if v}

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "total_degree2": len(degree2_nodes),
        "max_degree_threshold": max_degree,
        "reason_counts": reason_counts,
        "reason_examples": reason_examples,
    }


def format_degree2_diagnostics_report(report: Dict[str, Any]) -> str:
    """Format degree-2 diagnostics into a compact multiline report."""
    lines = [
        "Degree-2 diagnostics:",
        (
            f"  total_nodes={report.get('total_nodes', 0)}, "
            f"total_edges={report.get('total_edges', 0)}, "
            f"total_degree2={report.get('total_degree2', 0)}, "
            f"max_degree_threshold={report.get('max_degree_threshold', 0)}"
        ),
    ]

    counts = report.get("reason_counts", {})
    examples = report.get("reason_examples", {})
    for key in (
        "neighbors_not_2",
        "high_degree_neighbor",
        "missing_pos",
        "missing_edge_data",
        "eligible_for_smart_removal",
    ):
        count = counts.get(key, 0)
        sample = examples.get(key, [])
        lines.append(f"  {key}: {count}" + (f" (sample={sample})" if sample else ""))

    return "\n".join(lines)


def _rasterize_graph_edges(
    G: Union[nx.Graph, nx.MultiGraph],
    shape: tuple,
    spacing: np.ndarray,
) -> np.ndarray:
    """Every edge's ``voxels`` polyline (physical microns), converted back
    to voxel indices and rasterised onto a volume of the given *shape*.

    Shared by the two graph-vs-raster consistency checks below, so a graph
    with no ``voxels`` on an edge, or a point that rounds just past the
    array edge, is handled identically by both.
    """
    covered = np.zeros(shape, dtype=bool)
    for _u, _v, data in G.edges(data=True):
        voxels = data.get("voxels")
        if voxels is None or len(voxels) == 0:
            continue
        indices = np.round(np.asarray(voxels, dtype=float) / spacing).astype(int)
        for axis in range(indices.shape[1]):
            indices[:, axis] = np.clip(indices[:, axis], 0, shape[axis] - 1)
        covered[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return covered


def diagnose_skeleton_graph_consistency(
    G: Union[nx.Graph, nx.MultiGraph],
    skeleton: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Fraction of *skeleton* the finished graph's edges still trace.

    Every edge's own ``voxels`` polyline (physical microns) is converted
    back to voxel indices and rasterised onto a volume the skeleton's own
    shape; the fraction of skeleton foreground voxels landing on that
    rasterised set is how much of the original skeleton this graph's
    topology repair -- reconnection, degree-2 removal, cluster collapse,
    stub pruning, orphan repair -- still accounts for by the time it is
    finished. A steep drop points at over-aggressive pruning or collapsing,
    not at skeletonisation itself, which this never touches.

    An approximate reading, not a precise one: centreline smoothing moves an
    edge's points off the original skeleton voxels on purpose (see
    ``graph.smoothing``), and a handful of interior voxels absorbed by
    ``collapse_node_clusters`` at a merged junction are expected to go
    unmatched even in a healthy run. It is a coarse "did a large chunk of
    the skeleton go missing" check, not a per-voxel audit.
    """
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    skeleton_voxel_count = int(skeleton_bool.sum())
    if skeleton_voxel_count == 0:
        return {
            "skeleton_voxel_count": 0,
            "graph_voxel_count": 0,
            "matched_voxel_count": 0,
            "coverage_fraction": 1.0,
        }

    spacing = np.asarray([float(v) for v in voxel_size_zyx], dtype=float)
    covered = _rasterize_graph_edges(G, skeleton_bool.shape, spacing)

    matched_voxel_count = int((covered & skeleton_bool).sum())
    return {
        "skeleton_voxel_count": skeleton_voxel_count,
        "graph_voxel_count": int(covered.sum()),
        "matched_voxel_count": matched_voxel_count,
        "coverage_fraction": matched_voxel_count / skeleton_voxel_count,
    }


def format_skeleton_graph_consistency_report(report: Dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_skeleton_graph_consistency`."""
    return (
        "Skeleton/graph consistency: "
        f"{report.get('matched_voxel_count', 0)} of "
        f"{report.get('skeleton_voxel_count', 0)} skeleton voxels are still "
        f"traced by the graph's edges "
        f"({report.get('coverage_fraction', 1.0):.1%})."
    )


def diagnose_graph_mask_consistency(
    G: Union[nx.Graph, nx.MultiGraph],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Fraction of the segmented image the finished graph's edges run through.

    The third leg of the consistency triangle:
    ``preprocessing.skeleton_consistency.diagnose_skeleton_mask_consistency``
    checks the skeleton against the mask before any graph exists,
    :func:`diagnose_skeleton_graph_consistency` above checks the graph
    against the skeleton it was built from; this checks the graph directly
    against the original segmented image, so a loss that compounds across
    both earlier steps -- individually healthy skeleton/mask and
    skeleton/graph numbers that still add up to a poor graph/mask number --
    has somewhere to show up.

    Uses the same local-radius-plus-one-voxel-diagonal criterion as
    ``diagnose_skeleton_mask_consistency`` (see that function's docstring
    for the empirical justification): a mask voxel counts as "explained"
    once the nearest point on the graph's own rasterised edges is no
    farther from it than that point's own inscribed radius plus that
    margin. *mask* is read via the same canonical
    ``io.load._to_binary_volume_for_skeletonization`` binarisation the
    other two consistency checks use, not a plain ``!= 0`` test, for the
    same reason.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.skeleton_consistency import _explained_by_local_radius

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        return {
            "mask_voxel_count": 0,
            "graph_voxel_count": 0,
            "explained_voxel_count": 0,
            "coverage_fraction": 1.0,
        }

    spacing = np.asarray([float(v) for v in voxel_size_zyx], dtype=float)
    covered = _rasterize_graph_edges(G, mask_bool.shape, spacing)

    explained = _explained_by_local_radius(
        covered, mask_bool, voxel_size_zyx=voxel_size_zyx
    )
    explained_voxel_count = int(explained.sum())
    return {
        "mask_voxel_count": mask_voxel_count,
        "graph_voxel_count": int(covered.sum()),
        "explained_voxel_count": explained_voxel_count,
        "coverage_fraction": explained_voxel_count / mask_voxel_count,
    }


def format_graph_mask_consistency_report(report: Dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_graph_mask_consistency`."""
    return (
        "Graph/mask consistency: "
        f"{report.get('explained_voxel_count', 0)} of "
        f"{report.get('mask_voxel_count', 0)} segmented-image voxels are "
        f"within their own local radius (plus discretisation margin) of "
        f"the graph's edges ({report.get('coverage_fraction', 1.0):.1%})."
    )


def diagnose_vessels_missing_from_graph(
    G: Union[nx.Graph, nx.MultiGraph],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
    min_vessel_voxels: int = 2,
) -> Dict[str, Any]:
    """Whole segmented-image vessels the finished graph drops entirely.

    The graph-side counterpart of
    ``preprocessing.skeleton_consistency.diagnose_vessels_missing_from_skeleton``
    -- the inverse question to :func:`diagnose_graph_mask_consistency`'s
    coverage fraction, which can stay high even while a whole small vessel
    is unrepresented, as long as it is a small enough slice of total
    volume. This instead treats each connected component of the mask as
    one candidate vessel and asks whether the graph's own rasterised edges
    explain *any* of it at all -- catching a vessel a topology-repair pass
    (pruning, orphan removal, cluster collapse) dropped whole, which a
    single blended percentage can hide. See
    ``preprocessing.skeleton_consistency._missing_mask_components`` for the
    noise-vessel size floor and why it matters.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.skeleton_consistency import (
        _explained_by_local_radius,
        _missing_mask_components,
    )

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    spacing = np.asarray([float(v) for v in voxel_size_zyx], dtype=float)
    covered = _rasterize_graph_edges(G, mask_bool.shape, spacing)

    explained = _explained_by_local_radius(
        covered, mask_bool, voxel_size_zyx=voxel_size_zyx
    )
    return _missing_mask_components(
        explained, mask_bool, min_vessel_voxels=min_vessel_voxels
    )


def diagnose_graph_against_mask(
    G: Union[nx.Graph, nx.MultiGraph],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
    min_vessel_voxels: int = 2,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """``(diagnose_graph_mask_consistency(...),
    diagnose_vessels_missing_from_graph(...))`` for the same pair, from one
    rasterisation, one local-radius map and one distance transform of the
    graph's edges instead of two of each -- the same two reports.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.skeleton_consistency import (
        _explained_by_local_radius,
        _missing_mask_components,
    )

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        consistency = {
            "mask_voxel_count": 0,
            "graph_voxel_count": 0,
            "explained_voxel_count": 0,
            "coverage_fraction": 1.0,
        }
        return consistency, _missing_mask_components(
            mask_bool, mask_bool, min_vessel_voxels=min_vessel_voxels
        )
    spacing = np.asarray([float(v) for v in voxel_size_zyx], dtype=float)
    covered = _rasterize_graph_edges(G, mask_bool.shape, spacing)
    explained = _explained_by_local_radius(covered, mask_bool, voxel_size_zyx=voxel_size_zyx)
    explained_voxel_count = int(explained.sum())
    consistency = {
        "mask_voxel_count": mask_voxel_count,
        "graph_voxel_count": int(covered.sum()),
        "explained_voxel_count": explained_voxel_count,
        "coverage_fraction": explained_voxel_count / mask_voxel_count,
    }
    return consistency, _missing_mask_components(
        explained, mask_bool, min_vessel_voxels=min_vessel_voxels
    )


def diagnose_parallel_duplicates_in_lumen(
    G: Union[nx.Graph, nx.MultiGraph],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
    mask_support=None,
) -> Dict[str, Any]:
    """Pairs of edges that run beside each other in one lumen over at least
    half of either's length: one segmented vessel represented twice.

    Each edge is judged against every other edge's centreline by
    ``preprocessing.bridge_mask_support.path_shadows_existing_vessel`` --
    the guard every reconnect and the mask recovery apply to a path before
    adding it -- so two vessels with background between them, a branch
    meeting the vessel it leaves, and two edges continuing each other are
    not pairs. *mask_support*, a ``MaskSupport`` over *mask*, is used in
    place of building one.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.bridge_mask_support import (
        MaskSupport,
        _densify,
        _sample_step,
        _shadowing_points,
    )
    from scipy.spatial import cKDTree

    from ._helpers import EdgeSampleIndex, edge_id, edge_sample_points

    if mask_support is None:
        mask_support = MaskSupport(_to_binary_volume_for_skeletonization(mask), voxel_size_zyx)
    step = _sample_step(mask_support.voxel_size_zyx)
    index = EdgeSampleIndex(G)
    ids: Dict[Any, int] = {}
    owners = np.asarray([ids.setdefault(owner, len(ids)) for owner in index.owners], dtype=np.intp)
    edges = list(ids)
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    pairs = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        own = ids.get(edge_id(u, v, key))
        if own is None:
            continue
        path = _densify(edge_sample_points(u, v, data, node_pos), step)
        reach = 2.0 * float(np.max(mask_support.radius(path))) + 1.0
        found = index.tree.query_ball_point(path, r=reach)
        near = np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in found]))
        near = near[owners[near] != own]
        if not len(near):
            continue
        judged, shadowing = _shadowing_points(
            path, cKDTree(index.points[near]), index.points[near],
            mask_support.inside, mask_support.radius, step_um=step,
        )
        if judged and 2 * len(shadowing) >= judged:
            partner = edges[int(np.bincount(owners[near[shadowing]]).argmax())]
            pairs.add(tuple(sorted((edges[own], partner), key=str)))
    return {
        "edge_count": len(edges),
        "duplicate_pair_count": len(pairs),
        "duplicate_pairs": sorted(pairs, key=str),
    }


def format_parallel_duplicates_report(report: Dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_parallel_duplicates_in_lumen`."""
    pairs = report.get("duplicate_pairs", [])
    return (
        "Parallel duplicates in a lumen: "
        f"{report.get('duplicate_pair_count', 0)} pair(s) of the graph's "
        f"{report.get('edge_count', 0)} edges run beside each other inside one "
        "segmented vessel"
        + (f" (e.g. {pairs[:3]})." if pairs else ".")
    )


#: A dead end this close to an image face, in microns, is a vessel the image
#: cut through: the open ends inlets and outlets are chosen from.
IMAGE_FACE_MARGIN_UM = 10.0

#: The dead-end kinds :func:`diagnose_lumen_artefacts` counts, in report order.
DEAD_END_KINDS = (
    "inside_other_lumen",
    "beside_vessel",
    "off_mask",
    "mask_continues",
    "short",
)


def _sample_weights(samples: np.ndarray) -> np.ndarray:
    """The length of centreline each sample of a polyline stands for."""
    if len(samples) < 2:
        return np.zeros(len(samples))
    spacing = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    return np.concatenate([[spacing[0] / 2], (spacing[:-1] + spacing[1:]) / 2, [spacing[-1] / 2]])


def diagnose_lumen_artefacts(
    G: Union[nx.Graph, nx.MultiGraph],
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
    mask_support=None,
    stub_radius_multiple: float = 3.0,
    image_face_margin_um: float = IMAGE_FACE_MARGIN_UM,
) -> Dict[str, Any]:
    """What graph building can leave in a segmented vessel that is not a
    vessel, measured against the mask: one report, read-only.

    - **Two edges in one lumen**: the pairs
      :func:`diagnose_parallel_duplicates_in_lumen` reports, those of them
      sharing one node (a *fork*: two edges leaving a junction side by side
      through one lumen), and the length of centreline, counted sample by
      sample (``preprocessing.shadowed_samples``), that runs beside another
      in the same lumen -- a doubled stretch of a long edge included.
    - **Loops inside one lumen**: of each edge's shortest loop
      (``lumen_loops.iter_short_loops``), those
      ``lumen_loops.loop_inside_one_lumen`` says run round no tissue.
    - **Dead ends** further than *image_face_margin_um* from every image
      face, by kind (one dead end can be several): its tip inside another
      vessel's lumen; at least half of it beside another centreline in the
      same lumen; less than half of it in the mask; the mask running on past
      its tip (``assemble.mask_continues_past``); shorter than
      *stub_radius_multiple* radii of the vessel it leaves. Isolated single
      edges are counted apart.

    *mask_support*, a ``MaskSupport`` over *mask*, is used in place of
    building one; without one *mask* is binarised as the loaders do.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.bridge_mask_support import (
        MaskSupport,
        _densify,
        _sample_step,
        shadowed_samples,
    )
    from scipy.spatial import cKDTree

    from ._helpers import EdgeSampleIndex, edge_id, edge_sample_points
    from .assemble import mask_continues_past
    from .lumen_loops import iter_short_loops, loop_inside_one_lumen

    support = mask_support
    if support is None:
        support = MaskSupport(_to_binary_volume_for_skeletonization(mask), voxel_size_zyx)
    step = _sample_step(support.voxel_size_zyx)
    duplicates = diagnose_parallel_duplicates_in_lumen(
        G, support.mask, voxel_size_zyx=support.voxel_size_zyx, mask_support=support
    )
    forks = [
        (a, b) for a, b in duplicates["duplicate_pairs"]
        if len({a[0], a[1]} & {b[0], b[1]}) == 1
    ]

    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    index = EdgeSampleIndex(G)
    ids: Dict[Any, int] = {}
    owners = np.asarray([ids.setdefault(owner, len(ids)) for owner in index.owners], dtype=np.intp)

    def other_points_near(points: np.ndarray, own: Any, reach: float) -> np.ndarray:
        if index.tree is None:
            return np.empty(0, dtype=np.intp)
        found = index.tree.query_ball_point(points, r=reach)
        near = np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in np.atleast_1d(found)]))
        return near[owners[near] != own] if own is not None else near

    total_um = duplicated_um = 0.0
    beside_share: Dict[Any, float] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in node_pos or v not in node_pos:
            continue
        own = ids.get(edge_id(u, v, key))
        path = edge_sample_points(u, v, data, node_pos)
        dense = _densify(path, step)
        if len(dense) < 2:
            continue
        total_um += float(_sample_weights(dense).sum())
        near = other_points_near(dense, own, 2.0 * float(np.max(support.radius(dense))) + 1.0)
        if not len(near):
            continue
        samples, judged, beside = shadowed_samples(
            path, cKDTree(index.points[near]), index.points[near],
            support.inside, support.radius, step_um=step,
        )
        duplicated_um += float(_sample_weights(samples)[beside].sum())
        if judged.any():
            beside_share[edge_id(u, v, key)] = float(beside[judged].mean())

    loop_count = 0
    loops_in_lumen: List[Any] = []
    for cycle, polyline in iter_short_loops(G):
        loop_count += 1
        if loop_inside_one_lumen(polyline, support):
            loops_in_lumen.append(cycle)

    extent = (np.asarray(support.mask.shape, dtype=float) - 1.0) * np.asarray(support.voxel_size_zyx)
    continues = mask_continues_past(support)
    dead_ends: Dict[str, List[Any]] = {kind: [] for kind in DEAD_END_KINDS}
    isolated = sum(1 for u, v in G.edges() if u != v and G.degree[u] == 1 and G.degree[v] == 1)
    at_face = interior = 0
    for node in G.nodes:
        if G.degree[node] != 1 or node not in node_pos:
            continue
        (_, other, key, data), = G.edges(node, keys=True, data=True)
        if G.degree[other] == 1 or other not in node_pos:
            continue
        tip = node_pos[node]
        if float(np.min(np.minimum(tip, extent - tip))) <= image_face_margin_um:
            at_face += 1
            continue
        interior += 1
        path = _densify(edge_sample_points(other, node, data, node_pos), step)
        if len(path) < 2:
            continue
        if 2 * int(np.count_nonzero(support.inside(path))) < len(path):
            dead_ends["off_mask"].append(node)
        own = ids.get(edge_id(node, other, key))
        near = other_points_near(tip.reshape(1, 3), own, 2.0 * float(support.radius(tip.reshape(1, 3))[0]) + 30.0)
        if len(near):
            distance = np.linalg.norm(index.points[near] - tip, axis=1)
            nearest = index.points[near[int(np.argmin(distance))]].reshape(1, 3)
            if float(distance.min()) <= float(support.radius(nearest)[0]) + 1.0:
                dead_ends["inside_other_lumen"].append(node)
        if beside_share.get(edge_id(node, other, key), 0.0) >= 0.5:
            dead_ends["beside_vessel"].append(node)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path[::-1], axis=0), axis=1))])
        outward = tip - path[::-1][min(int(np.searchsorted(arc, 3.0)), len(path) - 1)]
        norm = float(np.linalg.norm(outward))
        if norm > 0 and continues(tip, outward / norm):
            dead_ends["mask_continues"].append(node)
        parent_radius = float(support.radius(node_pos[other].reshape(1, 3))[0])
        length = float(data.get("length") or _sample_weights(path).sum())
        if parent_radius > 0 and length < stub_radius_multiple * parent_radius:
            dead_ends["short"].append(node)

    return {
        "edge_count": duplicates["edge_count"],
        "centreline_um": total_um,
        "duplicate_pair_count": duplicates["duplicate_pair_count"],
        "duplicate_pairs": duplicates["duplicate_pairs"],
        "fork_pair_count": len(forks),
        "duplicated_centreline_um": duplicated_um,
        "duplicated_centreline_fraction": duplicated_um / total_um if total_um else 0.0,
        "short_loop_count": loop_count,
        "loops_inside_one_lumen": len(loops_in_lumen),
        "loop_cycles_inside_one_lumen": loops_in_lumen,
        "dead_ends_at_image_face": at_face,
        "interior_dead_ends": interior,
        "isolated_single_edges": isolated,
        "dead_end_counts": {kind: len(nodes) for kind, nodes in dead_ends.items()},
        "dead_ends": dead_ends,
        "stub_radius_multiple": float(stub_radius_multiple),
    }


def lumen_artefacts_found(report: Dict[str, Any]) -> bool:
    """Whether :func:`diagnose_lumen_artefacts` found anything that is never
    a vessel: two edges in one lumen, a loop inside one, or a dead end whose
    tip lies in another vessel's lumen."""
    return bool(
        report.get("duplicate_pair_count", 0)
        or report.get("loops_inside_one_lumen", 0)
        or report.get("dead_end_counts", {}).get("inside_other_lumen", 0)
    )


def format_lumen_artefacts_report(report: Dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_lumen_artefacts`."""
    counts = report.get("dead_end_counts", {})
    return (
        "Lumen artefacts: "
        f"{report.get('duplicate_pair_count', 0)} pair(s) of edges share one lumen "
        f"({report.get('fork_pair_count', 0)} leaving one node side by side), and "
        f"{report.get('duplicated_centreline_um', 0.0):.0f} um "
        f"({report.get('duplicated_centreline_fraction', 0.0):.1%}) of centreline runs beside "
        "another in the same lumen; "
        f"{report.get('loops_inside_one_lumen', 0)} of {report.get('short_loop_count', 0)} "
        "short loops lie inside one lumen; "
        f"{report.get('interior_dead_ends', 0)} dead ends away from the image faces "
        f"({counts.get('inside_other_lumen', 0)} ending inside another vessel's lumen, "
        f"{counts.get('beside_vessel', 0)} beside another centreline, "
        f"{counts.get('off_mask', 0)} mostly off the mask, "
        f"{counts.get('mask_continues', 0)} where the mask runs on past the tip, "
        f"{counts.get('short', 0)} shorter than "
        f"{report.get('stub_radius_multiple', 3.0):g} radii of their vessel), "
        f"{report.get('dead_ends_at_image_face', 0)} at an image face, and "
        f"{report.get('isolated_single_edges', 0)} isolated single edge(s)."
    )


def format_vessels_missing_from_graph_report(report: Dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_vessels_missing_from_graph`."""
    return (
        "Vessels missing from graph: "
        f"{report.get('missing_vessel_count', 0)} of "
        f"{report.get('vessel_count', 0)} segmented-image vessels have no "
        f"graph edge anywhere within their own local radius "
        f"({report.get('explained_vessel_fraction', 1.0):.1%} represented)."
    )
