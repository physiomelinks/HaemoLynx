"""Cut vascular graphs at large arteriole/venule mask volume boundaries.

When a centreline intersects a large-vessel mask, the portion inside the volume
is removed and crossing edges are split so new degree-1 terminals sit on the
exterior side of the boundary. Optional cleanup then drops small connected
components that remain entirely outside the volume.

Polarity (do not invert): mask True = interior = remove; mask False = exterior
= keep. The cut volume is the union (OR) of the large arteriole and venule
masks.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from haemolynx.io.load import _to_binary_volume_for_skeletonization

from ._helpers import calculate_path_length, orient_path_from_startpoint
from .automated_vessel_assignment import _position_to_mask_index

logger = logging.getLogger(__name__)


def _as_large_vessel_foreground(mask: np.ndarray, *, role: str) -> np.ndarray:
    """Boolean foreground mask; rejects raw ``dtype=bool`` casts of 1/2 labels."""
    arr = np.asarray(mask)
    if arr.dtype == bool:
        binary = arr
    else:
        binary = np.asarray(
            _to_binary_volume_for_skeletonization(arr), dtype=bool
        )
        # A constant non-zero volume is an empty large-vessel mask (background
        # label only). Shared skeleton binarisation maps that to all-True via
        # ``arr > 0``, which would delete the whole network at cut time.
        values = np.unique(arr)
        if values.size == 1 and values[0] != 0 and bool(binary.all()):
            binary = np.zeros(arr.shape, dtype=bool)
    fill = float(np.mean(binary)) if binary.size else 0.0
    if fill > 0.5:
        logger.warning(
            "large_%s_mask fills %.1f%% of voxels before the network cut; "
            "large-vessel cut volumes should be sparse foreground. A raw "
            "``astype(bool)`` / nonzero cast on a 1/2-encoded mask marks every "
            "voxel True and removes the whole network.",
            role,
            100.0 * fill,
        )
    return binary


def _combined_large_vessel_mask(
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
) -> np.ndarray:
    arteriole = _as_large_vessel_foreground(
        large_arteriole_mask, role="arteriole"
    )
    venule = _as_large_vessel_foreground(large_venule_mask, role="venule")
    if arteriole.shape != venule.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {arteriole.shape} and {venule.shape}."
        )
    # Union of interiors to remove — both masks participate.
    return arteriole | venule


def _point_inside_mask(
    point_zyx: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> bool:
    """True when the physical point falls in a True (interior) mask voxel."""
    idx = _position_to_mask_index(
        np.asarray(point_zyx, dtype=float),
        voxel_size_zyx=voxel_size_zyx,
        mask_shape=mask.shape,
    )
    if idx is None:
        return False
    return bool(mask[idx])


def _edge_sample_points(
    u: Any,
    v: Any,
    edge_data: dict[str, Any],
    node_pos: dict[Any, np.ndarray],
) -> np.ndarray:
    """Physical polyline for an edge, oriented from ``u`` toward ``v``."""
    pos_u = np.asarray(node_pos[u], dtype=float)
    pos_v = np.asarray(node_pos[v], dtype=float)
    voxels = edge_data.get("voxels")
    if voxels is not None:
        arr = np.asarray(voxels, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
            oriented = orient_path_from_startpoint(arr.tolist(), pos_u)
            return np.asarray(oriented, dtype=float)
    return np.vstack([pos_u.reshape(1, 3), pos_v.reshape(1, 3)])


def _densify_polyline(
    points: np.ndarray,
    *,
    max_step_um: float,
) -> np.ndarray:
    """Sample a polyline so consecutive points are at most ``max_step_um`` apart.

    Sparse centreline voxels (or a two-endpoint fallback) can leave a straight
    segment with both ends outside a mask while every interior mask voxel along
    the chord is missed. Densifying before the inside/outside test catches those
    crossings.
    """
    if max_step_um <= 0:
        raise ValueError(f"max_step_um must be > 0, got {max_step_um}.")
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return arr
    if arr.shape[0] == 1:
        return arr
    dense: list[np.ndarray] = [arr[0]]
    for start, end in zip(arr[:-1], arr[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= max_step_um:
            dense.append(end)
            continue
        steps = int(np.ceil(length / max_step_um))
        for step in range(1, steps + 1):
            dense.append(start + (step / steps) * segment)
    return np.asarray(dense, dtype=float)


def _exterior_runs(inside_flags: list[bool]) -> list[tuple[int, int]]:
    """Inclusive index ranges of contiguous exterior (not-inside) samples.

    These are the runs that remain after the cut. Interior runs are discarded.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_inside in enumerate(inside_flags):
        if not is_inside:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(inside_flags) - 1))
    return runs


def _next_node_id(G: nx.MultiGraph, reserved: set[Any]) -> int:
    numeric = [
        int(n)
        for n in list(G.nodes) + list(reserved)
        if isinstance(n, (int, np.integer))
    ]
    return (max(numeric) if numeric else -1) + 1


def _edge_attrs_for_segment(
    edge_data: dict[str, Any],
    segment_points: np.ndarray,
) -> dict[str, Any]:
    attrs = {
        key: value
        for key, value in edge_data.items()
        if key not in ("voxels", "length")
    }
    points = [tuple(float(c) for c in row) for row in segment_points]
    attrs["voxels"] = points
    attrs["length"] = float(calculate_path_length(points))
    return attrs


def _remove_small_components_by_edge_count(
    G: nx.MultiGraph,
    *,
    max_edge_count: int,
) -> tuple[nx.MultiGraph, int]:
    """Drop connected components whose edge count is strictly below threshold."""
    if max_edge_count < 1:
        raise ValueError(
            f"orphaned_branch_max_edge_count must be >= 1, got {max_edge_count}."
        )
    keep_nodes: set[Any] = set()
    removed_components = 0
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if int(subgraph.number_of_edges()) < int(max_edge_count):
            removed_components += 1
            continue
        keep_nodes.update(component)
    if removed_components == 0:
        return G, 0
    return G.subgraph(keep_nodes).copy(), removed_components


def cut_graph_at_large_vessel_volumes(
    G: nx.MultiGraph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    enabled: bool = True,
    remove_orphaned_branches: bool = False,
    orphaned_branch_max_edge_count: int = 3,
) -> nx.MultiGraph:
    """Cut ``G`` at large-vessel mask boundaries and drop interior geometry.

    Parameters
    ----------
    enabled
        When False, return a copy of ``G`` unchanged (pipeline toggle off).
        When True, each centreline is densified to about one sample per voxel
        before the inside/outside test.
    remove_orphaned_branches
        After cutting, remove connected components whose edge count is strictly
        less than ``orphaned_branch_max_edge_count``.
    orphaned_branch_max_edge_count
        Edge-count threshold for optional orphan cleanup (remove if
        ``edge_count < threshold``).

    Notes
    -----
    A sample point is interior when it maps into a True voxel of the OR of the
    arteriole and venule masks. Fully interior edges are dropped. Crossing
    edges keep only exterior runs; new degree-1 nodes are created at the last
    exterior sample of each cut so remaining edges contain zero interior
    voxels.
    """
    if not enabled:
        return G.copy()

    mask = _combined_large_vessel_mask(large_arteriole_mask, large_venule_mask)
    result = nx.MultiGraph()
    result.graph.update(G.graph)

    node_pos: dict[Any, np.ndarray] = {}
    for node_id, data in G.nodes(data=True):
        result.add_node(node_id, **dict(data))
        if "pos" in data:
            node_pos[node_id] = np.asarray(data["pos"], dtype=float)

    reserved_ids: set[Any] = set(result.nodes)
    edges_kept = 0
    edges_dropped_interior = 0
    edges_split = 0
    cut_nodes_created = 0

    edge_iter = (
        G.edges(keys=True, data=True)
        if isinstance(G, nx.MultiGraph)
        else ((u, v, 0, d) for u, v, d in G.edges(data=True))
    )

    for u, v, _key, edge_data in edge_iter:
        if u not in node_pos or v not in node_pos:
            result.add_edge(u, v, **dict(edge_data))
            edges_kept += 1
            continue

        points = _edge_sample_points(u, v, edge_data, node_pos)
        sample_step = min(float(spacing) for spacing in voxel_size_zyx)
        points = _densify_polyline(points, max_step_um=sample_step)
        inside_flags = [
            _point_inside_mask(point, mask, voxel_size_zyx=voxel_size_zyx)
            for point in points
        ]

        if all(inside_flags):
            edges_dropped_interior += 1
            continue

        if not any(inside_flags):
            result.add_edge(u, v, **dict(edge_data))
            edges_kept += 1
            continue

        runs = _exterior_runs(inside_flags)
        if not runs:
            edges_dropped_interior += 1
            continue

        edges_split += 1
        for run_start, run_end in runs:
            segment = points[run_start : run_end + 1]
            # A lone exterior sample cannot form a vessel segment. When it is an
            # original endpoint, that node remains as the degree-1 cut terminal
            # via its other exterior edges; otherwise there is nothing to keep.
            if segment.shape[0] < 2:
                continue

            touches_u = run_start == 0
            touches_v = run_end == len(points) - 1

            if touches_u:
                start_node = u
            else:
                start_node = _next_node_id(result, reserved_ids)
                reserved_ids.add(start_node)
                start_pos = tuple(float(c) for c in segment[0])
                result.add_node(start_node, pos=start_pos)
                node_pos[start_node] = np.asarray(start_pos, dtype=float)
                cut_nodes_created += 1

            if touches_v:
                end_node = v
            else:
                end_node = _next_node_id(result, reserved_ids)
                reserved_ids.add(end_node)
                end_pos = tuple(float(c) for c in segment[-1])
                result.add_node(end_node, pos=end_pos)
                node_pos[end_node] = np.asarray(end_pos, dtype=float)
                cut_nodes_created += 1

            if start_node == end_node:
                continue

            result.add_edge(
                start_node,
                end_node,
                **_edge_attrs_for_segment(edge_data, segment),
            )

    isolated = [n for n in result.nodes if result.degree(n) == 0]
    result.remove_nodes_from(isolated)

    removed_orphan_components = 0
    if remove_orphaned_branches:
        result, removed_orphan_components = _remove_small_components_by_edge_count(
            result,
            max_edge_count=int(orphaned_branch_max_edge_count),
        )

    logger.info(
        "Cut graph at large-vessel volumes: "
        f"kept_unsplit={edges_kept}, dropped_interior={edges_dropped_interior}, "
        f"split_crossing={edges_split}, cut_nodes_created={cut_nodes_created}, "
        f"removed_isolated_nodes={len(isolated)}, "
        f"removed_orphan_components={removed_orphan_components}, "
        f"remaining_nodes={result.number_of_nodes()}, "
        f"remaining_edges={result.number_of_edges()}."
    )
    considered = edges_kept + edges_dropped_interior + edges_split
    if considered > 0 and edges_dropped_interior / considered >= 0.9:
        logger.warning(
            "Large-vessel cut removed %.0f%% of edges as fully interior "
            "(%d/%d). That usually means the main input network was already "
            "confined to a large arteriole/venule mask (e.g. input_path set to "
            "the large-venule TIFF). Cut polarity still keeps exterior / removes "
            "interior; point input_path at the full vessel segmentation instead.",
            100.0 * edges_dropped_interior / considered,
            edges_dropped_interior,
            considered,
        )
    return result
