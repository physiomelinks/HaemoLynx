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
    shape = skeleton_bool.shape
    covered = np.zeros(shape, dtype=bool)
    for _u, _v, data in G.edges(data=True):
        voxels = data.get("voxels")
        if voxels is None or len(voxels) == 0:
            continue
        indices = np.round(np.asarray(voxels, dtype=float) / spacing).astype(int)
        for axis in range(indices.shape[1]):
            indices[:, axis] = np.clip(indices[:, axis], 0, shape[axis] - 1)
        covered[indices[:, 0], indices[:, 1], indices[:, 2]] = True

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
