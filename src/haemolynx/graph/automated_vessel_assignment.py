"""Automatic terminal-node assignment from arteriole/venule masks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

# Aliased because this module also uses ``edge_id`` as a local variable name.
from ._helpers import edge_id as _edge_id, sort_nodes as _sort_nodes
from .large_vessels import (
    exclude_smaller_overlapping_large_vessel_components,
    exclude_smaller_overlapping_small_vessel_components,
)

logger = logging.getLogger(__name__)


def _terminal_nodes_with_position_pairs(G: nx.Graph) -> list[tuple[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals: list[tuple[Any, np.ndarray]] = []
    for node_id, degree in G.degree():
        if degree != 1 or node_id not in node_pos:
            continue
        terminals.append((node_id, np.asarray(node_pos[node_id], dtype=float)))
    return terminals


def _position_to_mask_index(
    position_zyx: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_zyx must be three positive values, got {voxel_size_zyx}."
        )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a 3D mask shape, got {mask_shape}.")

    voxel_index = np.rint(position_zyx / voxel_size).astype(int)
    if np.any(voxel_index < 0):
        return None
    if np.any(voxel_index >= np.asarray(mask_shape, dtype=int)):
        return None
    return (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))


def _terminal_edge_sample_points(
    G: nx.Graph,
    node_id: Any,
    node_pos: np.ndarray,
    *,
    max_sample_points: int = 25,
) -> np.ndarray:
    """Collect sample points near a terminal node along its incident edge."""
    if max_sample_points <= 0:
        return np.asarray([node_pos], dtype=float)

    edge_voxels: np.ndarray | None = None
    if isinstance(G, nx.MultiGraph):
        incident_edges = list(G.edges(node_id, keys=True, data=True))
        if incident_edges:
            edge_data = incident_edges[0][3]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr
    else:
        incident_edges = list(G.edges(node_id, data=True))
        if incident_edges:
            edge_data = incident_edges[0][2]
            voxels = edge_data.get("voxels")
            if voxels is not None:
                arr = np.asarray(voxels, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
                    edge_voxels = arr

    if edge_voxels is None:
        return np.asarray([node_pos], dtype=float)

    distances = np.linalg.norm(edge_voxels - node_pos.reshape(1, 3), axis=1)
    nearest_idx = np.argsort(distances)[: max_sample_points]
    samples = edge_voxels[nearest_idx]
    samples = np.vstack([samples, node_pos.reshape(1, 3)])
    return np.unique(samples, axis=0)


def _mask_midpoint_physical(
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    return np.mean(points_zyx.astype(float), axis=0) * voxel_size


def _mask_principal_axis(mask: np.ndarray) -> int:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return 0
    spans = np.ptp(points_zyx.astype(float), axis=0)
    return int(np.argmax(spans))


def _cross_section_midpoint_physical(
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    intersection_point: np.ndarray | None,
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0 or intersection_point is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    axis = _mask_principal_axis(mask)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    intersection_index = np.rint(intersection_point / voxel_size).astype(int)
    target_slice = int(intersection_index[axis])
    slice_coords = points_zyx[:, axis]
    in_slice = points_zyx[slice_coords == target_slice]
    if in_slice.size == 0:
        nearest_slice = int(
            np.unique(slice_coords)[
                int(np.argmin(np.abs(np.unique(slice_coords) - target_slice)))
            ]
        )
        in_slice = points_zyx[slice_coords == nearest_slice]
    if in_slice.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    return np.mean(in_slice.astype(float), axis=0) * voxel_size


def _overlap_fraction_and_intersection(
    sample_points: np.ndarray,
    mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    node_pos: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    valid_points: list[np.ndarray] = []
    in_mask_points: list[np.ndarray] = []
    for point in sample_points:
        mask_index = _position_to_mask_index(
            point,
            voxel_size_zyx=voxel_size_zyx,
            mask_shape=mask.shape,
        )
        if mask_index is None:
            continue
        valid_points.append(point)
        if mask[mask_index]:
            in_mask_points.append(point)
    if not valid_points:
        return 0.0, None
    overlap_fraction = float(len(in_mask_points)) / float(len(valid_points))
    if not in_mask_points:
        return overlap_fraction, None
    in_mask_arr = np.asarray(in_mask_points, dtype=float)
    dists = np.linalg.norm(in_mask_arr - node_pos.reshape(1, 3), axis=1)
    return overlap_fraction, in_mask_arr[int(np.argmin(dists))]


def resolve_overlapping_terminal_node_assignment(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    max_sample_points: int = 25,
) -> str:
    """Resolve input/output assignment for a terminal node in both masks.

    Decision rule:
    1) Prefer shorter distance from overlap-entry point to vessel cross-section
       midpoint at the entry slice.
    2) If tied, prefer shorter distance to vessel volume midpoint.
    3) If still tied, use higher local overlap percentage near the node.
    """
    metrics = compute_overlapping_terminal_assignment_metrics(
        G,
        node_id,
        node_pos=node_pos,
        large_arteriole_mask=large_arteriole_mask,
        large_venule_mask=large_venule_mask,
        voxel_size_zyx=voxel_size_zyx,
        max_sample_points=max_sample_points,
    )
    arteriole_overlap = float(metrics["arteriole_overlap_fraction"])
    venule_overlap = float(metrics["venule_overlap_fraction"])
    arteriole_cross_section_dist = float(metrics["arteriole_cross_section_midpoint_distance"])
    venule_cross_section_dist = float(metrics["venule_cross_section_midpoint_distance"])
    if arteriole_cross_section_dist < venule_cross_section_dist:
        return "inlet"
    if venule_cross_section_dist < arteriole_cross_section_dist:
        return "outlet"

    arteriole_dist = float(metrics["arteriole_midpoint_distance"])
    venule_dist = float(metrics["venule_midpoint_distance"])
    if arteriole_dist < venule_dist:
        return "inlet"
    if venule_dist < arteriole_dist:
        return "outlet"

    if arteriole_overlap > venule_overlap:
        return "inlet"
    if venule_overlap > arteriole_overlap:
        return "outlet"
    # Final deterministic tie-break.
    return "inlet"


def compute_overlapping_terminal_assignment_metrics(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    max_sample_points: int = 25,
) -> dict[str, Any]:
    """Compute overlap and midpoint-distance metrics for overlap resolution."""
    samples = _terminal_edge_sample_points(
        G,
        node_id,
        node_pos,
        max_sample_points=max_sample_points,
    )
    arteriole_overlap, arteriole_intersection = _overlap_fraction_and_intersection(
        samples, large_arteriole_mask, voxel_size_zyx, node_pos
    )
    venule_overlap, venule_intersection = _overlap_fraction_and_intersection(
        samples, large_venule_mask, voxel_size_zyx, node_pos
    )
    arteriole_mid = _mask_midpoint_physical(large_arteriole_mask, voxel_size_zyx)
    venule_mid = _mask_midpoint_physical(large_venule_mask, voxel_size_zyx)
    arteriole_cross_section_mid = _cross_section_midpoint_physical(
        large_arteriole_mask, voxel_size_zyx, arteriole_intersection
    )
    venule_cross_section_mid = _cross_section_midpoint_physical(
        large_venule_mask, voxel_size_zyx, venule_intersection
    )
    arteriole_dist = np.inf
    venule_dist = np.inf
    arteriole_cross_section_dist = np.inf
    venule_cross_section_dist = np.inf
    if arteriole_intersection is not None and np.all(np.isfinite(arteriole_mid)):
        arteriole_dist = float(np.linalg.norm(arteriole_intersection - arteriole_mid))
    if venule_intersection is not None and np.all(np.isfinite(venule_mid)):
        venule_dist = float(np.linalg.norm(venule_intersection - venule_mid))
    if arteriole_intersection is not None and np.all(np.isfinite(arteriole_cross_section_mid)):
        arteriole_cross_section_dist = float(
            np.linalg.norm(arteriole_intersection - arteriole_cross_section_mid)
        )
    if venule_intersection is not None and np.all(np.isfinite(venule_cross_section_mid)):
        venule_cross_section_dist = float(
            np.linalg.norm(venule_intersection - venule_cross_section_mid)
        )
    return {
        "arteriole_overlap_fraction": float(arteriole_overlap),
        "venule_overlap_fraction": float(venule_overlap),
        "arteriole_cross_section_midpoint_distance": float(arteriole_cross_section_dist),
        "venule_cross_section_midpoint_distance": float(venule_cross_section_dist),
        "arteriole_midpoint_distance": float(arteriole_dist),
        "venule_midpoint_distance": float(venule_dist),
        "arteriole_intersection": None if arteriole_intersection is None else arteriole_intersection.copy(),
        "venule_intersection": None if venule_intersection is None else venule_intersection.copy(),
        "arteriole_cross_section_midpoint": arteriole_cross_section_mid.copy(),
        "venule_cross_section_midpoint": venule_cross_section_mid.copy(),
        "arteriole_midpoint": arteriole_mid.copy(),
        "venule_midpoint": venule_mid.copy(),
    }


def select_terminal_nodes_from_large_vessel_masks(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    terminal_node_ids: set[Any] | None = None,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Assign degree-1 nodes to input/output groups by vessel-mask overlap.

    When ``exclude_smaller_overlapping_volumes`` is True, overlapping large-vessel
    components are pre-cleaned by removing only overlap voxels from the smaller
    component in each overlap pair before node assignment.
    ``overlap_parallel_workers`` is accepted for API compatibility with the
    progressive/fast-mode callers; the current path resolves overlaps sequentially.
    """
    del overlap_parallel_workers  # reserved for parallel overlap resolution
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = (
            exclude_smaller_overlapping_large_vessel_components(
                arteriole_mask,
                venule_mask,
            )
        )
        if cleaned_arteriole is None or cleaned_venule is None:
            raise RuntimeError(
                "Internal error: expected cleaned masks when large masks are provided."
            )
        arteriole_mask = cleaned_arteriole
        venule_mask = cleaned_venule
    terminal_nodes = _terminal_nodes_with_position_pairs(G)
    if terminal_node_ids is not None:
        allowed = set(terminal_node_ids)
        terminal_nodes = [
            (node_id, node_pos)
            for node_id, node_pos in terminal_nodes
            if node_id in allowed
        ]
    if not terminal_nodes:
        return [], []

    inlet_nodes: set[Any] = set()
    outlet_nodes: set[Any] = set()
    for node_id, node_pos in terminal_nodes:
        index_zyx = _position_to_mask_index(
            node_pos,
            voxel_size_zyx=voxel_size_zyx,
            mask_shape=arteriole_mask.shape,
        )
        if index_zyx is None:
            continue
        in_arteriole = bool(arteriole_mask[index_zyx])
        in_venule = bool(venule_mask[index_zyx])
        if in_arteriole and in_venule and not allow_overlap:
            assignment = resolve_overlapping_terminal_node_assignment(
                G,
                node_id,
                node_pos=node_pos,
                large_arteriole_mask=arteriole_mask,
                large_venule_mask=venule_mask,
                voxel_size_zyx=voxel_size_zyx,
            )
            if assignment == "inlet":
                inlet_nodes.add(node_id)
            else:
                outlet_nodes.add(node_id)
            continue
        if in_arteriole:
            inlet_nodes.add(node_id)
        if in_venule:
            outlet_nodes.add(node_id)

    if not allow_overlap:
        outlet_nodes -= inlet_nodes

    return _sort_nodes(inlet_nodes), _sort_nodes(outlet_nodes)


def _edge_sample_points_from_data(
    edge_data: dict[str, Any],
    endpoint_positions: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return unique physical sample points for an edge."""
    voxels = edge_data.get("voxels")
    if voxels is not None:
        arr = np.asarray(voxels, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.size > 0:
            return np.unique(arr, axis=0)
    p_u, p_v = endpoint_positions
    return np.unique(
        np.vstack([p_u.reshape(1, 3), p_v.reshape(1, 3)]),
        axis=0,
    )


def _sample_overlap_fraction(
    sample_points: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> float:
    """Return fraction of valid edge sample points that fall inside a mask."""
    valid_count = 0
    in_mask_count = 0
    for point in sample_points:
        idx = _position_to_mask_index(
            point,
            voxel_size_zyx=voxel_size_zyx,
            mask_shape=mask.shape,
        )
        if idx is None:
            continue
        valid_count += 1
        if bool(mask[idx]):
            in_mask_count += 1
    if valid_count == 0:
        return 0.0
    return float(in_mask_count) / float(valid_count)


#: An unlabelled edge joining two edges of one vessel type takes that type
#: when at least this fraction of ``minimum_overlap_fraction`` of it lies in
#: that type's mask: hysteresis, so one under-covered stretch of a vessel does
#: not read as the vessel ending twice.
GAP_FILL_FRACTION_OF_THRESHOLD = 0.5


def _label_edge_ids(G: nx.Graph) -> list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]]:
    """``(u, v, edge id, data)`` for every edge, the id in this module's form."""
    if isinstance(G, nx.MultiGraph):
        return [(u, v, _edge_id(u, v, key), data) for u, v, key, data in G.edges(keys=True, data=True)]
    return [
        (u, v, ((u, v, 0) if u <= v else (v, u, 0)), data) for u, v, data in G.edges(data=True)
    ]


def _fill_label_gaps(
    edges: list[tuple[Any, Any, tuple[Any, Any, int], dict[str, Any]]],
    fractions: dict[tuple[Any, Any, int], tuple[float, float]],
    labelled: dict[str, set[tuple[Any, Any, int]]],
    *,
    low_fraction: float,
) -> int:
    """Label every unlabelled edge whose two ends each meet another edge of
    one type and which lies at least *low_fraction* in that type's mask;
    repeat until nothing changes. Returns how many edges were filled."""
    filled = 0
    ends = {edge_id: (u, v) for u, v, edge_id, _data in edges}
    data_of = {edge_id: data for _u, _v, edge_id, data in edges}
    while True:
        touching: dict[str, dict[Any, set]] = {name: {} for name in labelled}
        for name, edge_ids in labelled.items():
            for edge_id in edge_ids:
                for node in ends.get(edge_id, ()):
                    touching[name].setdefault(node, set()).add(edge_id)
        taken = set().union(*labelled.values())
        added: list[tuple[str, tuple[Any, Any, int]]] = []
        for edge_id, (art_fraction, ven_fraction) in fractions.items():
            if edge_id in taken:
                continue
            u, v = ends[edge_id]
            options = []
            for name, fraction in (("arteriole", art_fraction), ("venule", ven_fraction)):
                if fraction < low_fraction:
                    continue
                if touching[name].get(u, set()) - {edge_id} and touching[name].get(v, set()) - {edge_id}:
                    options.append((fraction, name))
            if options:
                added.append((max(options)[1], edge_id))
        if not added:
            return filled
        for name, edge_id in added:
            labelled[name].add(edge_id)
            data_of[edge_id]["mask_vessel_type"] = name
            filled += 1


def infer_boundary_nodes_from_small_vessel_masks(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> dict[str, Any]:
    """Label mask-overlapping edges/nodes and infer arteriole/venule boundaries.

    Edges are marked as arteriole/venule when the fraction of sampled edge points
    inside the corresponding small-vessel mask meets `minimum_overlap_fraction`.
    Then, with hysteresis, an unlabelled edge joining two edges of one type
    takes that type when at least ``GAP_FILL_FRACTION_OF_THRESHOLD`` of that
    fraction of it lies in the type's mask: one under-covered edge in the
    middle of an arteriole used to leave a gap, and both its ends were then
    reported as the arteriole meeting the capillary bed. Associated endpoint
    nodes are given the same mask vessel type. Boundary nodes are the
    labeled-mask nodes that connect to at least one unlabeled edge, i.e.
    where the small-vessel mask region transitions into the capillary bed.
    """
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    if not (0.0 <= float(minimum_overlap_fraction) <= 1.0):
        raise ValueError(
            "minimum_overlap_fraction must be in [0.0, 1.0]. "
            f"Got {minimum_overlap_fraction}."
        )

    arteriole_mask = small_arteriole_mask.astype(bool, copy=False)
    venule_mask = small_venule_mask.astype(bool, copy=False)
    del overlap_parallel_workers  # reserved for parallel edge classification
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = (
            exclude_smaller_overlapping_small_vessel_components(
                arteriole_mask,
                venule_mask,
            )
        )
        if cleaned_arteriole is None or cleaned_venule is None:
            raise RuntimeError(
                "Internal error: expected cleaned masks when small masks are provided."
            )
        arteriole_mask = cleaned_arteriole
        venule_mask = cleaned_venule
    node_positions = nx.get_node_attributes(G, "pos")
    if not node_positions:
        raise ValueError("Graph has no node positions ('pos').")

    # Clear previous mask labels to keep output deterministic between reruns.
    for _, attrs in G.nodes(data=True):
        attrs.pop("mask_vessel_type", None)
    edges = _label_edge_ids(G)
    for _u, _v, _edge, attrs in edges:
        attrs.pop("mask_vessel_type", None)

    arteriole_edges: set[tuple[Any, Any, int]] = set()
    venule_edges: set[tuple[Any, Any, int]] = set()
    overlap_edges = 0
    fractions: dict[tuple[Any, Any, int], tuple[float, float]] = {}
    threshold = float(minimum_overlap_fraction)

    for u, v, edge_id, edge_data in edges:
        if u not in node_positions or v not in node_positions:
            continue
        pu = np.asarray(node_positions[u], dtype=float)
        pv = np.asarray(node_positions[v], dtype=float)
        samples = _edge_sample_points_from_data(edge_data, (pu, pv))
        arteriole_fraction = _sample_overlap_fraction(
            samples, arteriole_mask, voxel_size_zyx=voxel_size_zyx
        )
        venule_fraction = _sample_overlap_fraction(
            samples, venule_mask, voxel_size_zyx=voxel_size_zyx
        )
        fractions[edge_id] = (arteriole_fraction, venule_fraction)
        in_arteriole = arteriole_fraction >= threshold
        in_venule = venule_fraction >= threshold
        if in_arteriole and in_venule:
            overlap_edges += 1
            if allow_overlap:
                arteriole_edges.add(edge_id)
                venule_edges.add(edge_id)
                edge_data["mask_vessel_type"] = "overlap"
            elif arteriole_fraction >= venule_fraction:
                arteriole_edges.add(edge_id)
                edge_data["mask_vessel_type"] = "arteriole"
            else:
                venule_edges.add(edge_id)
                edge_data["mask_vessel_type"] = "venule"
            continue
        if in_arteriole:
            arteriole_edges.add(edge_id)
            edge_data["mask_vessel_type"] = "arteriole"
        elif in_venule:
            venule_edges.add(edge_id)
            edge_data["mask_vessel_type"] = "venule"

    gap_filled = _fill_label_gaps(
        edges,
        fractions,
        {"arteriole": arteriole_edges, "venule": venule_edges},
        low_fraction=GAP_FILL_FRACTION_OF_THRESHOLD * threshold,
    )
    if gap_filled:
        logger.info(
            "Small-vessel masks: %d under-covered edge(s) between two edges of one "
            "vessel type took that type (at least %.2f of the edge in its mask).",
            gap_filled,
            GAP_FILL_FRACTION_OF_THRESHOLD * threshold,
        )

    arteriole_nodes: set[Any] = set()
    venule_nodes: set[Any] = set()
    for u, v, key in arteriole_edges:
        arteriole_nodes.add(u)
        arteriole_nodes.add(v)
    for u, v, key in venule_edges:
        venule_nodes.add(u)
        venule_nodes.add(v)

    if not allow_overlap:
        overlapping_nodes = arteriole_nodes & venule_nodes
        venule_nodes -= overlapping_nodes

    for node_id in arteriole_nodes:
        if node_id in G.nodes:
            G.nodes[node_id]["mask_vessel_type"] = "arteriole"
    for node_id in venule_nodes:
        if node_id in G.nodes and G.nodes[node_id].get("mask_vessel_type") != "arteriole":
            G.nodes[node_id]["mask_vessel_type"] = "venule"

    incident: dict[Any, set[tuple[Any, Any, int]]] = {}
    for u, v, edge_id, _data in edges:
        incident.setdefault(u, set()).add(edge_id)
        incident.setdefault(v, set()).add(edge_id)

    def _boundary_nodes_for(edge_ids: set[tuple[Any, Any, int]], labeled_nodes: set[Any]) -> list[Any]:
        # Transition point from mask-labeled region to non-labeled region.
        return _sort_nodes(
            node_id
            for node_id in labeled_nodes
            if any(edge_id not in edge_ids for edge_id in incident.get(node_id, ()))
        )

    arteriole_boundary_nodes = _boundary_nodes_for(arteriole_edges, arteriole_nodes)
    venule_boundary_nodes = _boundary_nodes_for(venule_edges, venule_nodes)

    return {
        "arteriole_boundary_nodes": arteriole_boundary_nodes,
        "venule_boundary_nodes": venule_boundary_nodes,
        "arteriole_nodes": _sort_nodes(arteriole_nodes),
        "venule_nodes": _sort_nodes(venule_nodes),
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "overlap_edge_count": overlap_edges,
        "gap_filled_edge_count": gap_filled,
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
    }


def filter_io_nodes_to_terminal_degree1(
    G: nx.Graph,
    input_nodes: list[Any],
    output_nodes: list[Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Keep only degree-1 nodes in input/output assignments.

    Returns:
        (filtered_input_nodes, filtered_output_nodes, dropped_input_nodes, dropped_output_nodes)
    """
    degree1_nodes = {node_id for node_id, degree in G.degree() if int(degree) == 1}
    dropped_input_nodes = [node_id for node_id in input_nodes if node_id not in degree1_nodes]
    dropped_output_nodes = [node_id for node_id in output_nodes if node_id not in degree1_nodes]
    filtered_input_nodes = [node_id for node_id in input_nodes if node_id in degree1_nodes]
    filtered_output_nodes = [node_id for node_id in output_nodes if node_id in degree1_nodes]
    return (
        filtered_input_nodes,
        filtered_output_nodes,
        dropped_input_nodes,
        dropped_output_nodes,
    )


def _build_dilation_schedule_microns(
    *,
    max_dilation_microns: float,
    dilation_step_microns: float,
) -> list[float]:
    """Create dilation schedule including 0 and the exact max dilation."""
    max_dilation = float(max_dilation_microns)
    step = float(dilation_step_microns)
    if max_dilation < 0:
        raise ValueError(
            "max_dilation_microns must be >= 0. "
            f"Got {max_dilation_microns}."
        )
    if step <= 0:
        raise ValueError(
            "dilation_step_microns must be > 0. "
            f"Got {dilation_step_microns}."
        )

    schedule: list[float] = [0.0]
    if max_dilation <= 0:
        return schedule

    epsilon = 1e-9
    current = step
    while current < (max_dilation - epsilon):
        schedule.append(float(current))
        current += step
    if abs(schedule[-1] - max_dilation) > epsilon:
        schedule.append(float(max_dilation))
    return schedule


def _distances_to_mask_microns(
    mask: np.ndarray,
    indices: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Each of *indices*' distance to the nearest voxel of *mask*, in microns
    (0 inside it; ``inf`` when the mask is empty).

    Read point by point from the mask's surface (a KD-tree over it), not from
    a whole-volume distance transform: the same values
    ``distance_transform_edt(~mask, sampling=voxel_size_zyx)`` gives there.
    """
    from haemolynx.preprocessing.pointwise_distance import FeatureDistance

    binary = np.asanyarray(mask, dtype=bool)
    if len(indices) == 0:
        return np.zeros(0, dtype=float)
    if not binary.any():
        return np.full(len(indices), np.inf)
    if binary.all():
        return np.zeros(len(indices), dtype=float)
    distance = FeatureDistance(binary, feature_value=True, sampling=voxel_size_zyx)
    return distance.at(np.asarray(indices, dtype=np.intp))


def select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Assign I/O nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first, followed by fixed dilation
    increments (default: 5 microns) up to `max_dilation_microns`. A terminal
    is locked at the first step whose dilation reaches it -- inside the
    arteriole mask grown by that much, an inlet; the venule mask, an outlet --
    and cannot be reassigned at a later step.

    Each terminal's distance to each mask is read once, point by point, and
    its step is the first one at least that far: the same answer growing both
    masks step by step gives, without building two whole-volume masks per step
    to read a handful of voxels from. A terminal both masks reach at the same
    step goes to the nearer one -- it used to be settled by comparing
    whole-mask midpoints, scanning both masks for every such terminal --
    and an exact tie to the arteriole, as before.

    ``exclude_smaller_overlapping_volumes`` cleans the two masks' overlap
    once, before any distance is read. ``overlap_parallel_workers`` is
    accepted for API compatibility and unused.
    """
    del overlap_parallel_workers
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=max_dilation_microns,
        dilation_step_microns=dilation_step_microns,
    )

    terminal_nodes = _terminal_nodes_with_position_pairs(G)
    if not terminal_nodes:
        return [], []

    arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    venule_mask = large_venule_mask.astype(bool, copy=False)
    if exclude_smaller_overlapping_volumes:
        cleaned_arteriole, cleaned_venule = exclude_smaller_overlapping_large_vessel_components(
            arteriole_mask, venule_mask
        )
        if cleaned_arteriole is not None and cleaned_venule is not None:
            arteriole_mask, venule_mask = cleaned_arteriole, cleaned_venule

    inside: list[tuple[Any, tuple[int, int, int]]] = []
    for node_id, node_pos in terminal_nodes:
        index = _position_to_mask_index(
            node_pos, voxel_size_zyx=voxel_size_zyx, mask_shape=arteriole_mask.shape
        )
        if index is not None:
            inside.append((node_id, index))
    indices = np.asarray([index for _node, index in inside], dtype=np.intp).reshape(-1, 3)
    to_arteriole = _distances_to_mask_microns(arteriole_mask, indices, voxel_size_zyx=voxel_size_zyx)
    to_venule = _distances_to_mask_microns(venule_mask, indices, voxel_size_zyx=voxel_size_zyx)

    steps = np.asarray(schedule, dtype=float)

    def first_step(distance: float) -> int | None:
        reached = np.flatnonzero(steps >= float(distance) - 1e-9)
        return int(reached[0]) if reached.size else None

    assigned_inputs: set[Any] = set()
    assigned_outputs: set[Any] = set()
    per_step = [[0, 0] for _ in schedule]
    for (node_id, _index), d_art, d_ven in zip(inside, to_arteriole, to_venule):
        step_art, step_ven = first_step(d_art), first_step(d_ven)
        if step_art is None and step_ven is None:
            continue
        if step_ven is None or (step_art is not None and step_art < step_ven):
            to_inlet = True
        elif step_art is None or step_ven < step_art:
            to_inlet = False
        else:
            # Both reach it at the same step: the nearer mask, and on an exact
            # tie (or allow_overlap) the arteriole, as the step-mask version did.
            to_inlet = allow_overlap or float(d_art) <= float(d_ven)
        step = step_art if to_inlet else step_ven
        if to_inlet:
            assigned_inputs.add(node_id)
            per_step[step][0] += 1
        else:
            assigned_outputs.add(node_id)
            per_step[step][1] += 1

    logger.info(
        "Automated large-vessel progressive assignment: %d step(s), max_dilation=%.3f microns, "
        "step=%.3f microns; new (inputs, outputs) per step: %s; %d of %d terminals unassigned.",
        len(schedule),
        float(max_dilation_microns),
        float(dilation_step_microns),
        ", ".join(f"{d:g} um: {i}/{o}" for d, (i, o) in zip(schedule, per_step)),
        len(terminal_nodes) - len(assigned_inputs) - len(assigned_outputs),
        len(terminal_nodes),
    )
    return _sort_nodes(assigned_inputs), _sort_nodes(assigned_outputs - assigned_inputs)


def _distance_from_mask_microns(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Compute physical EDT distance from binary mask (in microns)."""
    binary_mask = mask.astype(bool, copy=False)
    sampling_zyx = (
        float(voxel_size_zyx[0]),
        float(voxel_size_zyx[1]),
        float(voxel_size_zyx[2]),
    )
    return distance_transform_edt(~binary_mask, sampling=sampling_zyx)


def _dilated_mask_from_cached_distance(
    base_mask: np.ndarray,
    distance_from_mask: np.ndarray,
    *,
    dilation_microns: float,
) -> np.ndarray:
    """Apply dilation threshold using a cached EDT distance volume."""
    dilation = float(dilation_microns)
    if dilation <= 0:
        return base_mask.astype(bool, copy=False)
    return base_mask.astype(bool, copy=False) | (distance_from_mask <= dilation)


def _downsample_binary_mask_max(mask: np.ndarray, stride: int) -> np.ndarray:
    """Downsample a 3D binary mask via block max-pooling."""
    if stride <= 1:
        return mask.astype(bool, copy=False)

    z, y, x = mask.shape
    pad_z = (-z) % stride
    pad_y = (-y) % stride
    pad_x = (-x) % stride
    if pad_z or pad_y or pad_x:
        padded = np.pad(
            mask.astype(bool, copy=False),
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=False,
        )
    else:
        padded = mask.astype(bool, copy=False)

    z2, y2, x2 = padded.shape
    pooled = padded.reshape(
        z2 // stride,
        stride,
        y2 // stride,
        stride,
        x2 // stride,
        stride,
    )
    return np.max(pooled, axis=(1, 3, 5))


def _recompute_small_vessel_boundary_state_from_edge_labels(
    G: nx.Graph,
    *,
    allow_overlap: bool,
    minimum_overlap_fraction: float,
) -> dict[str, Any]:
    """Rebuild boundary/node labelling from current edge ``mask_vessel_type`` values."""
    arteriole_edges: set[tuple[Any, Any, int]] = set()
    venule_edges: set[tuple[Any, Any, int]] = set()
    overlap_edge_count = 0
    if isinstance(G, nx.MultiGraph):
        for u, v, key, attrs in G.edges(keys=True, data=True):
            vt = attrs.get("mask_vessel_type")
            if vt is None:
                continue
            eid = _edge_id(u, v, key)
            if vt == "arteriole":
                arteriole_edges.add(eid)
            elif vt == "venule":
                venule_edges.add(eid)
            elif vt == "overlap":
                overlap_edge_count += 1
                arteriole_edges.add(eid)
                venule_edges.add(eid)
    else:
        for u, v, attrs in G.edges(data=True):
            vt = attrs.get("mask_vessel_type")
            if vt is None:
                continue
            eid = (u, v, 0) if u <= v else (v, u, 0)
            if vt == "arteriole":
                arteriole_edges.add(eid)
            elif vt == "venule":
                venule_edges.add(eid)
            elif vt == "overlap":
                overlap_edge_count += 1
                arteriole_edges.add(eid)
                venule_edges.add(eid)

    arteriole_nodes: set[Any] = set()
    venule_nodes: set[Any] = set()
    for u, v, key in arteriole_edges:
        arteriole_nodes.add(u)
        arteriole_nodes.add(v)
    for u, v, key in venule_edges:
        venule_nodes.add(u)
        venule_nodes.add(v)

    if not allow_overlap:
        overlapping_nodes = arteriole_nodes & venule_nodes
        venule_nodes -= overlapping_nodes

    for node_id in G.nodes:
        G.nodes[node_id].pop("mask_vessel_type", None)
    for node_id in arteriole_nodes:
        if node_id in G.nodes:
            G.nodes[node_id]["mask_vessel_type"] = "arteriole"
    for node_id in venule_nodes:
        if node_id in G.nodes and G.nodes[node_id].get("mask_vessel_type") != "arteriole":
            G.nodes[node_id]["mask_vessel_type"] = "venule"

    def _boundary_nodes_for(
        edge_ids: set[tuple[Any, Any, int]], labeled_nodes: set[Any]
    ) -> list[Any]:
        boundaries: set[Any] = set()
        for node_id in labeled_nodes:
            incident_all: set[tuple[Any, Any, int]] = set()
            if isinstance(G, nx.MultiGraph):
                for nu, nv, nkey in G.edges(node_id, keys=True):
                    incident_all.add(_edge_id(nu, nv, nkey))
            else:
                for nu, nv in G.edges(node_id):
                    edge_id = (nu, nv, 0) if nu <= nv else (nv, nu, 0)
                    incident_all.add(edge_id)
            if any(eid not in edge_ids for eid in incident_all):
                boundaries.add(node_id)
        return _sort_nodes(boundaries)

    arteriole_boundary_nodes = _boundary_nodes_for(arteriole_edges, arteriole_nodes)
    venule_boundary_nodes = _boundary_nodes_for(venule_edges, venule_nodes)

    return {
        "arteriole_boundary_nodes": arteriole_boundary_nodes,
        "venule_boundary_nodes": venule_boundary_nodes,
        "arteriole_nodes": _sort_nodes(arteriole_nodes),
        "venule_nodes": _sort_nodes(venule_nodes),
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "overlap_edge_count": overlap_edge_count,
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
    }


def infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> dict[str, Any]:
    """Infer boundary nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first. When the next scheduled
    dilation exceeds 1 micron, an extra 1 micron step is inserted so sparse masks
    can label nearest-vessel edges before large dilations smear the capillary gap.

    Edge ``mask_vessel_type`` labels from the first step where any edge is
    classified are locked (including explicit "unlabeled" gaps); later dilations
    cannot reassign locked edges. Boundary nodes from merged edge labels are
    accumulated across steps.
    """
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=max_dilation_microns,
        dilation_step_microns=dilation_step_microns,
    )
    if (
        len(schedule) >= 2
        and float(schedule[0]) == 0.0
        and float(schedule[1]) > 1.0
        and float(max_dilation_microns) >= 1.0
    ):
        schedule = sorted({float(s) for s in schedule} | {1.0})

    locked_edge_vessel_type: dict[tuple[Any, Any, int], str | None] = {}

    all_nodes = set(G.nodes)
    assigned_arteriole_boundary: set[Any] = set()
    assigned_venule_boundary: set[Any] = set()
    assigned_arteriole_nodes: set[Any] = set()
    assigned_venule_nodes: set[Any] = set()
    assigned_arteriole_edges: set[tuple[Any, Any, int]] = set()
    assigned_venule_edges: set[tuple[Any, Any, int]] = set()
    remaining_boundary_nodes = set(all_nodes)
    remaining_label_nodes = set(all_nodes)
    if isinstance(G, nx.MultiGraph):
        all_edge_ids = {_edge_id(u, v, key) for u, v, key in G.edges(keys=True)}
    else:
        all_edge_ids = {(u, v, 0) if u <= v else (v, u, 0) for u, v in G.edges()}
    remaining_label_edges = set(all_edge_ids)

    base_arteriole_mask = small_arteriole_mask.astype(bool, copy=False)
    base_venule_mask = small_venule_mask.astype(bool, copy=False)
    arteriole_distance_from_mask = _distance_from_mask_microns(
        base_arteriole_mask,
        voxel_size_zyx=voxel_size_zyx,
    )
    venule_distance_from_mask = _distance_from_mask_microns(
        base_venule_mask,
        voxel_size_zyx=voxel_size_zyx,
    )
    latest_result: dict[str, Any] = {
        "arteriole_edge_count": 0,
        "venule_edge_count": 0,
        "overlap_edge_count": 0,
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
    }

    logger.info(
        "Small-vessel progressive boundary assignment: "
        f"{len(schedule)} step(s), max_dilation={float(max_dilation_microns):.3f} microns, "
        f"step={float(dilation_step_microns):.3f} microns."
    )
    def _iter_edge_attr_items() -> list[tuple[tuple[Any, Any, int], dict[str, Any]]]:
        if isinstance(G, nx.MultiGraph):
            return [
                (_edge_id(u, v, key), edge_data)
                for u, v, key, edge_data in G.edges(keys=True, data=True)
            ]
        return [
            (((u, v, 0) if u <= v else (v, u, 0)), edge_data)
            for u, v, edge_data in G.edges(data=True)
        ]

    for step_idx, dilation_microns in enumerate(schedule, start=1):
        if not remaining_boundary_nodes and not remaining_label_nodes:
            logger.info(
                "Small-vessel progressive boundary assignment: "
                "all graph nodes assigned before final dilation step."
            )
            break
        if dilation_microns <= 0:
            step_arteriole_mask = base_arteriole_mask
            step_venule_mask = base_venule_mask
        else:
            step_arteriole_mask = _dilated_mask_from_cached_distance(
                base_arteriole_mask,
                arteriole_distance_from_mask,
                dilation_microns=float(dilation_microns),
            )
            step_venule_mask = _dilated_mask_from_cached_distance(
                base_venule_mask,
                venule_distance_from_mask,
                dilation_microns=float(dilation_microns),
            )

        infer_boundary_nodes_from_small_vessel_masks(
            G,
            small_arteriole_mask=step_arteriole_mask,
            small_venule_mask=step_venule_mask,
            voxel_size_zyx=voxel_size_zyx,
            minimum_overlap_fraction=minimum_overlap_fraction,
            allow_overlap=allow_overlap,
            exclude_smaller_overlapping_volumes=exclude_smaller_overlapping_volumes,
            overlap_parallel_workers=overlap_parallel_workers,
        )
        inferred_types: dict[tuple[Any, Any, int], str | None] = {}
        for edge_id, edge_data in _iter_edge_attr_items():
            inferred_types[edge_id] = edge_data.get("mask_vessel_type")

        for edge_id, edge_data in _iter_edge_attr_items():
            if edge_id in locked_edge_vessel_type:
                lock_val = locked_edge_vessel_type[edge_id]
                if lock_val is None:
                    edge_data.pop("mask_vessel_type", None)
                else:
                    edge_data["mask_vessel_type"] = lock_val

        labeled_any = any(v is not None for v in inferred_types.values())
        if labeled_any:
            for edge_id, vt in inferred_types.items():
                if edge_id not in locked_edge_vessel_type:
                    locked_edge_vessel_type[edge_id] = vt

        step_result = _recompute_small_vessel_boundary_state_from_edge_labels(
            G,
            allow_overlap=allow_overlap,
            minimum_overlap_fraction=minimum_overlap_fraction,
        )
        latest_result = step_result

        step_arteriole_nodes = [
            node_id
            for node_id in step_result.get("arteriole_nodes", [])
            if node_id in remaining_label_nodes
        ]
        for node_id in step_arteriole_nodes:
            assigned_arteriole_nodes.add(node_id)
            remaining_label_nodes.discard(node_id)

        step_venule_nodes = [
            node_id
            for node_id in step_result.get("venule_nodes", [])
            if node_id in remaining_label_nodes
        ]
        for node_id in step_venule_nodes:
            assigned_venule_nodes.add(node_id)
            remaining_label_nodes.discard(node_id)

        step_new_arteriole_edges = 0
        step_new_venule_edges = 0
        for edge_id, edge_data in _iter_edge_attr_items():
            if edge_id not in remaining_label_edges:
                continue
            vessel_type = edge_data.get("mask_vessel_type")
            if vessel_type == "arteriole":
                assigned_arteriole_edges.add(edge_id)
                remaining_label_edges.discard(edge_id)
                step_new_arteriole_edges += 1
            elif vessel_type == "venule":
                if edge_id not in assigned_arteriole_edges:
                    assigned_venule_edges.add(edge_id)
                remaining_label_edges.discard(edge_id)
                step_new_venule_edges += 1
            elif vessel_type == "overlap":
                # Preserve deterministic single-label edge assignment under overlap.
                assigned_arteriole_edges.add(edge_id)
                remaining_label_edges.discard(edge_id)
                step_new_arteriole_edges += 1

        step_arteriole_boundary = [
            node_id
            for node_id in step_result.get("arteriole_boundary_nodes", [])
            if node_id in remaining_boundary_nodes
        ]
        for node_id in step_arteriole_boundary:
            assigned_arteriole_boundary.add(node_id)
            remaining_boundary_nodes.discard(node_id)

        step_venule_boundary = [
            node_id
            for node_id in step_result.get("venule_boundary_nodes", [])
            if node_id in remaining_boundary_nodes
        ]
        for node_id in step_venule_boundary:
            assigned_venule_boundary.add(node_id)
            remaining_boundary_nodes.discard(node_id)

        logger.info(
            "Small-vessel progressive boundary assignment step "
            f"{step_idx}/{len(schedule)} "
            f"(dilation={float(dilation_microns):.3f} microns): "
            f"new_arteriole_boundary={len(step_arteriole_boundary)}, "
            f"new_venule_boundary={len(step_venule_boundary)}, "
            f"new_arteriole_edges={step_new_arteriole_edges}, "
            f"new_venule_edges={step_new_venule_edges}, "
            f"remaining_boundary_nodes={len(remaining_boundary_nodes)}."
        )

    # Reapply locked assignments so graph labels reflect returned progressive results.
    for _, attrs in G.nodes(data=True):
        attrs.pop("mask_vessel_type", None)
    if isinstance(G, nx.MultiGraph):
        edge_attr_by_id = {
            _edge_id(u, v, key): edge_data
            for u, v, key, edge_data in G.edges(keys=True, data=True)
        }
    else:
        edge_attr_by_id = {
            ((u, v, 0) if u <= v else (v, u, 0)): edge_data
            for u, v, edge_data in G.edges(data=True)
        }
    for edge_data in edge_attr_by_id.values():
        edge_data.pop("mask_vessel_type", None)
    for edge_id in assigned_arteriole_edges:
        edge_data = edge_attr_by_id.get(edge_id)
        if edge_data is not None:
            edge_data["mask_vessel_type"] = "arteriole"
    for edge_id in (assigned_venule_edges - assigned_arteriole_edges):
        edge_data = edge_attr_by_id.get(edge_id)
        if edge_data is not None:
            edge_data["mask_vessel_type"] = "venule"
    for node_id in assigned_arteriole_nodes:
        if node_id in G.nodes:
            G.nodes[node_id]["mask_vessel_type"] = "arteriole"
    for node_id in (assigned_venule_nodes - assigned_arteriole_nodes):
        if node_id in G.nodes:
            G.nodes[node_id]["mask_vessel_type"] = "venule"

    return {
        "arteriole_boundary_nodes": _sort_nodes(assigned_arteriole_boundary),
        "venule_boundary_nodes": _sort_nodes(assigned_venule_boundary - assigned_arteriole_boundary),
        "arteriole_nodes": _sort_nodes(assigned_arteriole_nodes),
        "venule_nodes": _sort_nodes(assigned_venule_nodes - assigned_arteriole_nodes),
        "arteriole_edge_count": int(latest_result.get("arteriole_edge_count", 0)),
        "venule_edge_count": int(latest_result.get("venule_edge_count", 0)),
        "overlap_edge_count": int(latest_result.get("overlap_edge_count", 0)),
        "minimum_overlap_fraction": float(
            latest_result.get("minimum_overlap_fraction", minimum_overlap_fraction)
        ),
    }


def write_automated_vessel_assignment_3d_html(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    input_nodes: list[Any],
    outlet_nodes: list[Any],
    voxel_size_zyx: tuple[float, float, float],
    output_html_path: str | Path,
) -> bool:
    """Write interactive 3D HTML showing masks, graph, and selected nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Edges from node positions; graph positions are stored as (z, y, x).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]
    else:
        edge_iter = G.edges(data=True)
        for u, v, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[2]), float(pv[2]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[0]), float(pv[0]), None]

    input_set = set(input_nodes)
    output_set = set(outlet_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        if not np.any(mask):
            return
        z_scale, y_scale, x_scale = voxel_size_zyx
        zz, yy, xx = np.indices(mask.shape, dtype=float)
        fig.add_trace(
            go.Volume(
                x=(xx * float(x_scale)).ravel(),
                y=(yy * float(y_scale)).ravel(),
                z=(zz * float(z_scale)).ravel(),
                value=mask.astype(float).ravel(),
                isomin=0.5,
                isomax=1.0,
                opacity=0.12,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=name,
            )
        )

    fig = go.Figure()
    _add_volume_trace(
        large_arteriole_mask.astype(bool, copy=False),
        name="Arteriole Mask Volume",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        large_venule_mask.astype(bool, copy=False),
        name="Venule Mask Volume",
        color="#FF3EA5",
        fig=fig,
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(0, 200, 255, 0.7)", width=5),
            name="Edges",
        )
    )
    if other_nodes:
        ox, oy, oz = _coords(other_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=4, color="#9E9E9E"),
                name="Other Nodes",
            )
        )
    if input_nodes:
        ix, iy, iz = _coords(input_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=8, color="#00FF7F"),
                name="Input Nodes",
            )
        )
    if outlet_nodes:
        ox, oy, oz = _coords(outlet_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=8, color="#FF3EA5"),
                name="Output Nodes",
            )
        )
    fig.update_layout(
        title="Automated Vessel Assignment (3D)",
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return True


def write_small_vessel_mask_boundary_labelling_3d_html(
    G: nx.Graph,
    *,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    arteriole_boundary_nodes: list[Any],
    venule_boundary_nodes: list[Any],
    voxel_size_zyx: tuple[float, float, float],
    output_html_path: str | Path,
    title: str = "Small Vessel Mask Boundary Labelling (3D)",
) -> bool:
    """Write interactive 3D HTML: small-vessel masks, mask-labelled edges, boundary nodes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raise ValueError("Graph has no node positions ('pos').")
    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_line_lists() -> tuple[list[float | None], list[float | None], list[float | None]]:
        return [], [], []

    segs: dict[str, tuple[list[float | None], list[float | None], list[float | None]]] = {
        "capillary": _empty_line_lists(),
        "arteriole": _empty_line_lists(),
        "venule": _empty_line_lists(),
        "overlap": _empty_line_lists(),
    }

    def _push_edge(kind: str, pu: np.ndarray, pv: np.ndarray) -> None:
        lx, ly, lz = segs[kind]
        lx += [float(pu[2]), float(pv[2]), None]
        ly += [float(pu[1]), float(pv[1]), None]
        lz += [float(pu[0]), float(pv[0]), None]

    if isinstance(G, nx.MultiGraph):
        for u, v, _k, edge_data in G.edges(keys=True, data=True):
            if u not in pos or v not in pos:
                continue
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            vt = edge_data.get("mask_vessel_type")
            kind = (
                vt
                if vt in ("arteriole", "venule", "overlap")
                else "capillary"
            )
            _push_edge(kind, pu, pv)
    else:
        for u, v, edge_data in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            vt = edge_data.get("mask_vessel_type")
            kind = (
                vt
                if vt in ("arteriole", "venule", "overlap")
                else "capillary"
            )
            _push_edge(kind, pu, pv)

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        if not np.any(mask):
            return
        z_scale, y_scale, x_scale = voxel_size_zyx
        zz, yy, xx = np.indices(mask.shape, dtype=float)
        fig.add_trace(
            go.Volume(
                x=(xx * float(x_scale)).ravel(),
                y=(yy * float(y_scale)).ravel(),
                z=(zz * float(z_scale)).ravel(),
                value=mask.astype(float).ravel(),
                isomin=0.5,
                isomax=1.0,
                opacity=0.12,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=name,
            )
        )

    art_b = set(arteriole_boundary_nodes)
    ven_b = set(venule_boundary_nodes)
    neutral_nodes: list[Any] = []
    art_interior: list[Any] = []
    ven_interior: list[Any] = []
    for n in G.nodes:
        if n in art_b or n in ven_b:
            continue
        t = G.nodes[n].get("mask_vessel_type")
        if t == "arteriole":
            art_interior.append(n)
        elif t == "venule":
            ven_interior.append(n)
        else:
            neutral_nodes.append(n)

    fig = go.Figure()
    _add_volume_trace(
        small_arteriole_mask.astype(bool, copy=False),
        name="Small arteriole mask",
        color="#00FF7F",
        fig=fig,
    )
    _add_volume_trace(
        small_venule_mask.astype(bool, copy=False),
        name="Small venule mask",
        color="#FF3EA5",
        fig=fig,
    )

    cap_x, cap_y, cap_z = segs["capillary"]
    if cap_x:
        fig.add_trace(
            go.Scatter3d(
                x=cap_x,
                y=cap_y,
                z=cap_z,
                mode="lines",
                line=dict(color="rgba(0, 200, 255, 0.45)", width=3),
                name="Edges (capillary / unlabelled)",
            )
        )
    a_x, a_y, a_z = segs["arteriole"]
    if a_x:
        fig.add_trace(
            go.Scatter3d(
                x=a_x,
                y=a_y,
                z=a_z,
                mode="lines",
                line=dict(color="rgba(0, 220, 120, 0.9)", width=5),
                name="Edges (arteriole mask)",
            )
        )
    v_x, v_y, v_z = segs["venule"]
    if v_x:
        fig.add_trace(
            go.Scatter3d(
                x=v_x,
                y=v_y,
                z=v_z,
                mode="lines",
                line=dict(color="rgba(255, 62, 165, 0.9)", width=5),
                name="Edges (venule mask)",
            )
        )
    o_x, o_y, o_z = segs["overlap"]
    if o_x:
        fig.add_trace(
            go.Scatter3d(
                x=o_x,
                y=o_y,
                z=o_z,
                mode="lines",
                line=dict(color="rgba(255, 200, 0, 0.95)", width=6),
                name="Edges (overlap)",
            )
        )

    if neutral_nodes:
        nx_, ny_, nz_ = _coords(neutral_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=nx_,
                y=ny_,
                z=nz_,
                mode="markers",
                marker=dict(size=4, color="#9E9E9E"),
                name="Nodes (unlabelled)",
            )
        )
    if art_interior:
        ix, iy, iz = _coords(art_interior)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=6, color="#00CC66"),
                name="Nodes (arteriole interior)",
            )
        )
    if ven_interior:
        vx, vy, vz = _coords(ven_interior)
        fig.add_trace(
            go.Scatter3d(
                x=vx,
                y=vy,
                z=vz,
                mode="markers",
                marker=dict(size=6, color="#E040A0"),
                name="Nodes (venule interior)",
            )
        )
    if arteriole_boundary_nodes:
        bx, by, bz = _coords(list(arteriole_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(size=11, color="#00FF7F", symbol="diamond", line=dict(width=1, color="#004422")),
                name="Arteriole boundary",
            )
        )
    if venule_boundary_nodes:
        bx, by, bz = _coords(list(venule_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(size=11, color="#FF3EA5", symbol="diamond", line=dict(width=1, color="#440022")),
                name="Venule boundary",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return True
