"""Automatic terminal-node assignment from arteriole/venule masks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any

import networkx as nx
import numpy as np

from ..coords import physical_xyz_to_index_zyx, index_zyx_to_physical_xyz
from .large_vessels import (
    dilate_large_vessel_masks_by_microns,
    exclude_smaller_overlapping_large_vessel_components,
    exclude_smaller_overlapping_small_vessel_components,
)


def _sort_nodes(nodes: set[Any]) -> list[Any]:
    return sorted(nodes, key=lambda n: (str(type(n)), str(n)))


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


def _edge_id(u: Any, v: Any, key: int) -> tuple[Any, Any, int]:
    """Orientation-independent edge id for MultiGraph edges."""
    return (u, v, key) if u <= v else (v, u, key)


def _terminal_nodes_with_positions(G: nx.Graph) -> list[tuple[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals: list[tuple[Any, np.ndarray]] = []
    for node_id, degree in G.degree():
        if degree != 1 or node_id not in node_pos:
            continue
        terminals.append((node_id, np.asarray(node_pos[node_id], dtype=float)))
    return terminals


def _position_to_mask_index(
    position_xyz: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int] | None:
    voxel_size = np.asarray(voxel_size_xyz, dtype=float)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            f"voxel_size_xyz must be three positive values, got {voxel_size_xyz}."
        )
    if len(mask_shape) != 3:
        raise ValueError(f"Expected a 3D mask shape, got {mask_shape}.")

    voxel_index = physical_xyz_to_index_zyx(position_xyz, voxel_size_xyz)
    if np.any(voxel_index < 0):
        return None
    if np.any(voxel_index >= np.asarray(mask_shape, dtype=int)):
        return None
    return (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2]))


def _nonzero_bbox_slices_zyx(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    """Return tight z/y/x bounding slices for nonzero mask voxels."""
    coords = np.argwhere(mask.astype(bool, copy=False))
    if coords.size == 0:
        return None
    mins = coords.min(axis=0).astype(int)
    maxs = coords.max(axis=0).astype(int) + 1
    return (
        slice(int(mins[0]), int(maxs[0])),
        slice(int(mins[1]), int(maxs[1])),
        slice(int(mins[2]), int(maxs[2])),
    )


def _point_in_bbox_zyx(index_zyx: tuple[int, int, int], bbox: tuple[slice, slice, slice]) -> bool:
    z, y, x = index_zyx
    z_slice, y_slice, x_slice = bbox
    return (
        int(z_slice.start) <= z < int(z_slice.stop)
        and int(y_slice.start) <= y < int(y_slice.stop)
        and int(x_slice.start) <= x < int(x_slice.stop)
    )


def _combine_bboxes_zyx(
    bboxes: list[tuple[slice, slice, slice] | None],
) -> tuple[slice, slice, slice] | None:
    """Return a union bbox from multiple optional bboxes."""
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        return None
    z_min = min(int(bbox[0].start) for bbox in valid)
    y_min = min(int(bbox[1].start) for bbox in valid)
    x_min = min(int(bbox[2].start) for bbox in valid)
    z_max = max(int(bbox[0].stop) for bbox in valid)
    y_max = max(int(bbox[1].stop) for bbox in valid)
    x_max = max(int(bbox[2].stop) for bbox in valid)
    return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))


def _sample_points_to_valid_indices_zyx(
    sample_points: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    mask_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sample points (x,y,z) to valid rounded indices and points."""
    points = np.asarray(sample_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError(
            f"voxel_size_xyz must be three positive values, got {voxel_size_xyz}."
        )

    indices_zyx = np.rint(
        np.column_stack(
            [
                points[:, 2] / vz,
                points[:, 1] / vy,
                points[:, 0] / vx,
            ]
        )
    ).astype(int)
    shape = np.asarray(mask_shape, dtype=int)
    valid_index_mask = np.all(indices_zyx >= 0, axis=1) & np.all(
        indices_zyx < shape.reshape(1, 3),
        axis=1,
    )
    if not np.any(valid_index_mask):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    return points[valid_index_mask], indices_zyx[valid_index_mask]


def _sample_overlap_fraction_from_valid_indices(
    valid_indices_zyx: np.ndarray,
    mask: np.ndarray,
) -> float:
    if valid_indices_zyx.size == 0:
        return 0.0
    in_mask_mask = mask[
        valid_indices_zyx[:, 0],
        valid_indices_zyx[:, 1],
        valid_indices_zyx[:, 2],
    ].astype(bool)
    return float(np.count_nonzero(in_mask_mask)) / float(valid_indices_zyx.shape[0])


def _indices_intersect_bbox_zyx(
    indices_zyx: np.ndarray,
    bbox: tuple[slice, slice, slice],
) -> bool:
    if indices_zyx.size == 0:
        return False
    z_slice, y_slice, x_slice = bbox
    in_bbox = (
        (indices_zyx[:, 0] >= int(z_slice.start))
        & (indices_zyx[:, 0] < int(z_slice.stop))
        & (indices_zyx[:, 1] >= int(y_slice.start))
        & (indices_zyx[:, 1] < int(y_slice.stop))
        & (indices_zyx[:, 2] >= int(x_slice.start))
        & (indices_zyx[:, 2] < int(x_slice.stop))
    )
    return bool(np.any(in_bbox))


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

    if edge_voxels.shape[0] <= max_sample_points:
        samples = edge_voxels
        samples = np.vstack([samples, node_pos.reshape(1, 3)])
        return np.unique(samples, axis=0)

    distances = np.linalg.norm(edge_voxels - node_pos.reshape(1, 3), axis=1)
    nearest_idx = np.argpartition(distances, max_sample_points - 1)[:max_sample_points]
    samples = edge_voxels[nearest_idx]
    samples = np.vstack([samples, node_pos.reshape(1, 3)])
    return np.unique(samples, axis=0)


def _mask_midpoint_physical(
    mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    midpoint_zyx = np.mean(points_zyx.astype(float), axis=0)
    return index_zyx_to_physical_xyz(midpoint_zyx, voxel_size_xyz)


def _mask_principal_axis(mask: np.ndarray) -> int:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return 0
    spans = np.ptp(points_zyx.astype(float), axis=0)
    return int(np.argmax(spans))


def _prepare_mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    """Precompute geometry for repeated overlap-resolution queries."""
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0:
        return {
            "empty": True,
            "midpoint_zyx": None,
            "principal_axis": 0,
            "slice_midpoints_zyx": {},
            "slice_values": np.asarray([], dtype=int),
        }

    points_float = points_zyx.astype(float)
    axis = int(np.argmax(np.ptp(points_float, axis=0)))
    midpoint_zyx = np.mean(points_float, axis=0)

    slice_coords = points_zyx[:, axis].astype(int)
    unique_slices = np.unique(slice_coords)
    slice_midpoints_zyx: dict[int, np.ndarray] = {}
    for slice_value in unique_slices:
        in_slice = points_float[slice_coords == slice_value]
        if in_slice.size == 0:
            continue
        slice_midpoints_zyx[int(slice_value)] = np.mean(in_slice, axis=0)

    return {
        "empty": False,
        "midpoint_zyx": midpoint_zyx,
        "principal_axis": axis,
        "slice_midpoints_zyx": slice_midpoints_zyx,
        "slice_values": unique_slices.astype(int),
    }


def _mask_midpoint_from_geometry(
    geometry: dict[str, Any],
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    if bool(geometry.get("empty", True)):
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    midpoint_zyx = geometry.get("midpoint_zyx")
    if midpoint_zyx is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    return index_zyx_to_physical_xyz(np.asarray(midpoint_zyx, dtype=float), voxel_size_xyz)


def _cross_section_midpoint_from_geometry(
    geometry: dict[str, Any],
    voxel_size_xyz: tuple[float, float, float],
    intersection_point: np.ndarray | None,
) -> np.ndarray:
    if bool(geometry.get("empty", True)) or intersection_point is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)

    axis = int(geometry.get("principal_axis", 0))
    intersection_index = physical_xyz_to_index_zyx(intersection_point, voxel_size_xyz)
    target_slice = int(intersection_index[axis])

    slice_midpoints_zyx = geometry.get("slice_midpoints_zyx", {})
    if target_slice in slice_midpoints_zyx:
        midpoint_zyx = np.asarray(slice_midpoints_zyx[target_slice], dtype=float)
        return index_zyx_to_physical_xyz(midpoint_zyx, voxel_size_xyz)

    slice_values = np.asarray(geometry.get("slice_values", np.asarray([], dtype=int)), dtype=int)
    if slice_values.size == 0:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    nearest_slice = int(slice_values[int(np.argmin(np.abs(slice_values - target_slice)))])
    midpoint_zyx = slice_midpoints_zyx.get(nearest_slice)
    if midpoint_zyx is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    return index_zyx_to_physical_xyz(np.asarray(midpoint_zyx, dtype=float), voxel_size_xyz)


def _cross_section_midpoint_physical(
    mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    intersection_point: np.ndarray | None,
) -> np.ndarray:
    points_zyx = np.argwhere(mask.astype(bool, copy=False))
    if points_zyx.size == 0 or intersection_point is None:
        return np.asarray([np.inf, np.inf, np.inf], dtype=float)
    axis = _mask_principal_axis(mask)
    intersection_index = physical_xyz_to_index_zyx(intersection_point, voxel_size_xyz)
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
    midpoint_zyx = np.mean(in_slice.astype(float), axis=0)
    return index_zyx_to_physical_xyz(midpoint_zyx, voxel_size_xyz)


def _overlap_fraction_and_intersection(
    sample_points: np.ndarray,
    mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    node_pos: np.ndarray,
) -> tuple[float, np.ndarray | None]:
    valid_points, valid_indices_zyx = _sample_points_to_valid_indices_zyx(
        sample_points,
        voxel_size_xyz=voxel_size_xyz,
        mask_shape=mask.shape,
    )
    if valid_points.size == 0 or valid_indices_zyx.size == 0:
        return 0.0, None

    in_mask_mask = mask[
        valid_indices_zyx[:, 0],
        valid_indices_zyx[:, 1],
        valid_indices_zyx[:, 2],
    ].astype(bool)
    overlap_fraction = float(np.count_nonzero(in_mask_mask)) / float(valid_points.shape[0])
    if not np.any(in_mask_mask):
        return overlap_fraction, None

    in_mask_arr = valid_points[in_mask_mask]
    dists = np.linalg.norm(in_mask_arr - node_pos.reshape(1, 3), axis=1)
    return overlap_fraction, in_mask_arr[int(np.argmin(dists))]


def resolve_overlapping_terminal_node_assignment(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    max_sample_points: int = 25,
    arteriole_geometry: dict[str, Any] | None = None,
    venule_geometry: dict[str, Any] | None = None,
    terminal_sample_points: np.ndarray | None = None,
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
        voxel_size_xyz=voxel_size_xyz,
        max_sample_points=max_sample_points,
        arteriole_geometry=arteriole_geometry,
        venule_geometry=venule_geometry,
        terminal_sample_points=terminal_sample_points,
    )
    arteriole_overlap = float(metrics["arteriole_overlap_fraction"])
    venule_overlap = float(metrics["venule_overlap_fraction"])
    arteriole_cross_section_dist = float(metrics["arteriole_cross_section_midpoint_distance"])
    venule_cross_section_dist = float(metrics["venule_cross_section_midpoint_distance"])
    if arteriole_cross_section_dist < venule_cross_section_dist:
        return "input"
    if venule_cross_section_dist < arteriole_cross_section_dist:
        return "output"

    arteriole_dist = float(metrics["arteriole_midpoint_distance"])
    venule_dist = float(metrics["venule_midpoint_distance"])
    if arteriole_dist < venule_dist:
        return "input"
    if venule_dist < arteriole_dist:
        return "output"

    if arteriole_overlap > venule_overlap:
        return "input"
    if venule_overlap > arteriole_overlap:
        return "output"
    # Final deterministic tie-break.
    return "input"


def compute_overlapping_terminal_assignment_metrics(
    G: nx.Graph,
    node_id: Any,
    *,
    node_pos: np.ndarray,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    max_sample_points: int = 25,
    arteriole_geometry: dict[str, Any] | None = None,
    venule_geometry: dict[str, Any] | None = None,
    terminal_sample_points: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute overlap and midpoint-distance metrics for overlap resolution."""
    if terminal_sample_points is None:
        samples = _terminal_edge_sample_points(
            G,
            node_id,
            node_pos,
            max_sample_points=max_sample_points,
        )
    else:
        samples = np.asarray(terminal_sample_points, dtype=float)
    arteriole_overlap, arteriole_intersection = _overlap_fraction_and_intersection(
        samples, large_arteriole_mask, voxel_size_xyz, node_pos
    )
    venule_overlap, venule_intersection = _overlap_fraction_and_intersection(
        samples, large_venule_mask, voxel_size_xyz, node_pos
    )
    arteriole_geom = (
        arteriole_geometry
        if arteriole_geometry is not None
        else _prepare_mask_geometry(large_arteriole_mask)
    )
    venule_geom = (
        venule_geometry
        if venule_geometry is not None
        else _prepare_mask_geometry(large_venule_mask)
    )
    arteriole_mid = _mask_midpoint_from_geometry(arteriole_geom, voxel_size_xyz)
    venule_mid = _mask_midpoint_from_geometry(venule_geom, voxel_size_xyz)
    arteriole_cross_section_mid = _cross_section_midpoint_from_geometry(
        arteriole_geom, voxel_size_xyz, arteriole_intersection
    )
    venule_cross_section_mid = _cross_section_midpoint_from_geometry(
        venule_geom, voxel_size_xyz, venule_intersection
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
    voxel_size_xyz: tuple[float, float, float],
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Assign degree-1 nodes to input/output groups by vessel-mask overlap.

    When `exclude_smaller_overlapping_volumes=True`, overlapping large-vessel
    components are pre-cleaned by removing only overlap voxels from the smaller
    component in each overlap pair before node assignment.
    """
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
    terminal_nodes = _terminal_nodes_with_positions(G)
    if not terminal_nodes:
        return [], []

    start_time_s = time.perf_counter()
    print(
        "Automated large-vessel assignment: "
        f"processing {len(terminal_nodes)} terminal node(s)."
    )

    # Defer expensive overlap geometry preparation until we actually find
    # overlap candidates that require tie-break resolution.
    arteriole_geometry: dict[str, Any] | None = None
    venule_geometry: dict[str, Any] | None = None
    arteriole_bbox = _nonzero_bbox_slices_zyx(arteriole_mask)
    venule_bbox = _nonzero_bbox_slices_zyx(venule_mask)
    union_bbox = _combine_bboxes_zyx([arteriole_bbox, venule_bbox])

    starting_nodes: set[Any] = set()
    output_nodes: set[Any] = set()
    overlap_candidates: list[tuple[Any, np.ndarray]] = []
    progress_stride = max(1, len(terminal_nodes) // 10)
    for idx, (node_id, node_pos) in enumerate(terminal_nodes, start=1):
        if idx == 1 or idx % progress_stride == 0 or idx == len(terminal_nodes):
            elapsed = time.perf_counter() - start_time_s
            print(
                "Automated large-vessel assignment progress: "
                f"{idx}/{len(terminal_nodes)} terminals "
                f"({elapsed:.1f}s elapsed)."
            )
        index_zyx = _position_to_mask_index(
            node_pos,
            voxel_size_xyz=voxel_size_xyz,
            mask_shape=arteriole_mask.shape,
        )
        if index_zyx is None:
            continue
        if union_bbox is not None and not _point_in_bbox_zyx(index_zyx, union_bbox):
            # Fast reject for terminals that cannot overlap either mask.
            continue
        in_arteriole = bool(arteriole_mask[index_zyx])
        in_venule = bool(venule_mask[index_zyx])
        if in_arteriole and in_venule and not allow_overlap:
            overlap_candidates.append((node_id, node_pos))
            continue
        if in_arteriole:
            starting_nodes.add(node_id)
        if in_venule:
            output_nodes.add(node_id)

    if overlap_candidates and not allow_overlap:
        print(
            "Automated large-vessel overlap-resolution: "
            f"{len(overlap_candidates)} overlap terminal(s) requiring tie-break."
        )
        arteriole_geometry = _prepare_mask_geometry(arteriole_mask)
        venule_geometry = _prepare_mask_geometry(venule_mask)

        def _resolve_one(candidate: tuple[Any, np.ndarray]) -> tuple[Any, str]:
            node_id, node_pos = candidate
            terminal_sample_points = _terminal_edge_sample_points(
                G,
                node_id,
                node_pos,
            )
            assignment = resolve_overlapping_terminal_node_assignment(
                G,
                node_id,
                node_pos=node_pos,
                large_arteriole_mask=arteriole_mask,
                large_venule_mask=venule_mask,
                voxel_size_xyz=voxel_size_xyz,
                arteriole_geometry=arteriole_geometry,
                venule_geometry=venule_geometry,
                terminal_sample_points=terminal_sample_points,
            )
            return node_id, assignment

        overlap_stride = max(1, len(overlap_candidates) // 10)
        if overlap_parallel_workers > 0 and len(overlap_candidates) > 1:
            resolved: list[tuple[Any, str]] = []
            with ThreadPoolExecutor(max_workers=int(overlap_parallel_workers)) as executor:
                futures = [executor.submit(_resolve_one, candidate) for candidate in overlap_candidates]
                for done_idx, future in enumerate(as_completed(futures), start=1):
                    resolved.append(future.result())
                    if (
                        done_idx == 1
                        or done_idx % overlap_stride == 0
                        or done_idx == len(overlap_candidates)
                    ):
                        elapsed = time.perf_counter() - start_time_s
                        print(
                            "Automated large-vessel overlap-resolution progress: "
                            f"{done_idx}/{len(overlap_candidates)} tie-break terminal(s) "
                            f"({elapsed:.1f}s elapsed)."
                        )
        else:
            resolved = []
            for done_idx, candidate in enumerate(overlap_candidates, start=1):
                resolved.append(_resolve_one(candidate))
                if (
                    done_idx == 1
                    or done_idx % overlap_stride == 0
                    or done_idx == len(overlap_candidates)
                ):
                    elapsed = time.perf_counter() - start_time_s
                    print(
                        "Automated large-vessel overlap-resolution progress: "
                        f"{done_idx}/{len(overlap_candidates)} tie-break terminal(s) "
                        f"({elapsed:.1f}s elapsed)."
                    )

        for node_id, assignment in resolved:
            if assignment == "input":
                starting_nodes.add(node_id)
            else:
                output_nodes.add(node_id)

    if not allow_overlap:
        output_nodes -= starting_nodes

    filtered_inputs, filtered_outputs, _dropped_inputs, _dropped_outputs = (
        filter_io_nodes_to_terminal_degree1(
            G,
            _sort_nodes(starting_nodes),
            _sort_nodes(output_nodes),
        )
    )
    total_elapsed = time.perf_counter() - start_time_s
    print(
        "Automated large-vessel assignment complete (step candidate totals): "
        f"inputs={len(filtered_inputs)}, outputs={len(filtered_outputs)}, "
        f"elapsed={total_elapsed:.1f}s."
    )
    return filtered_inputs, filtered_outputs


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


def select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
    G: nx.Graph,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Assign I/O nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first, followed by fixed dilation
    increments (default: 5 microns) up to `max_dilation_microns`.

    Nodes assigned at an earlier step are locked and cannot be reassigned at a
    later step, even if they overlap the opposite mask after additional dilation.
    """
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    schedule = _build_dilation_schedule_microns(
        max_dilation_microns=max_dilation_microns,
        dilation_step_microns=dilation_step_microns,
    )

    terminal_nodes = _terminal_nodes_with_positions(G)
    if not terminal_nodes:
        return [], []
    terminal_node_ids = {node_id for node_id, _ in terminal_nodes}

    assigned_inputs: set[Any] = set()
    assigned_outputs: set[Any] = set()
    remaining_terminal_ids = set(terminal_node_ids)

    base_arteriole_mask = large_arteriole_mask.astype(bool, copy=False)
    base_venule_mask = large_venule_mask.astype(bool, copy=False)

    print(
        "Automated large-vessel progressive assignment: "
        f"{len(schedule)} step(s), max_dilation={float(max_dilation_microns):.3f} microns, "
        f"step={float(dilation_step_microns):.3f} microns."
    )
    for step_idx, dilation_microns in enumerate(schedule, start=1):
        if not remaining_terminal_ids:
            print(
                "Automated large-vessel progressive assignment: "
                "all terminal nodes assigned before final dilation step."
            )
            break
        if dilation_microns <= 0:
            step_arteriole_mask = base_arteriole_mask
            step_venule_mask = base_venule_mask
        else:
            dilated_arteriole, dilated_venule = dilate_large_vessel_masks_by_microns(
                large_arteriole_mask=base_arteriole_mask,
                large_venule_mask=base_venule_mask,
                dilation_microns=float(dilation_microns),
                voxel_size_xyz=voxel_size_xyz,
            )
            if dilated_arteriole is None or dilated_venule is None:
                raise RuntimeError(
                    "Internal error: dilation unexpectedly returned None masks."
                )
            step_arteriole_mask = dilated_arteriole
            step_venule_mask = dilated_venule

        step_inputs, step_outputs = select_terminal_nodes_from_large_vessel_masks(
            G,
            large_arteriole_mask=step_arteriole_mask,
            large_venule_mask=step_venule_mask,
            voxel_size_xyz=voxel_size_xyz,
            allow_overlap=allow_overlap,
            exclude_smaller_overlapping_volumes=exclude_smaller_overlapping_volumes,
            overlap_parallel_workers=overlap_parallel_workers,
        )

        newly_assigned_inputs = [
            node_id
            for node_id in step_inputs
            if node_id in remaining_terminal_ids
        ]
        for node_id in newly_assigned_inputs:
            assigned_inputs.add(node_id)
            remaining_terminal_ids.discard(node_id)

        newly_assigned_outputs = [
            node_id
            for node_id in step_outputs
            if node_id in remaining_terminal_ids
        ]
        for node_id in newly_assigned_outputs:
            assigned_outputs.add(node_id)
            remaining_terminal_ids.discard(node_id)

        print(
            "Automated large-vessel progressive assignment step "
            f"{step_idx}/{len(schedule)} "
            f"(dilation={float(dilation_microns):.3f} microns): "
            f"step_total_inputs={len(step_inputs)}, "
            f"step_total_outputs={len(step_outputs)}, "
            f"new_inputs={len(newly_assigned_inputs)}, "
            f"new_outputs={len(newly_assigned_outputs)}, "
            f"remaining_terminals={len(remaining_terminal_ids)}."
        )

    sorted_inputs = _sort_nodes(assigned_inputs)
    sorted_outputs = _sort_nodes(assigned_outputs - assigned_inputs)
    return sorted_inputs, sorted_outputs


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
    voxel_size_xyz: tuple[float, float, float],
) -> float:
    """Return fraction of valid edge sample points that fall inside a mask."""
    _, valid_indices_zyx = _sample_points_to_valid_indices_zyx(
        sample_points,
        voxel_size_xyz=voxel_size_xyz,
        mask_shape=mask.shape,
    )
    return _sample_overlap_fraction_from_valid_indices(valid_indices_zyx, mask)


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


def infer_boundary_nodes_from_small_vessel_masks(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> dict[str, Any]:
    """Label mask-overlapping edges/nodes and infer arteriole/venule boundaries.

    Edges are marked as arteriole/venule when the fraction of sampled edge points
    inside the corresponding small-vessel mask meets `minimum_overlap_fraction`.
    Associated endpoint nodes are given the same mask vessel type. Boundary nodes
    are the labeled-mask nodes that connect to at least one unlabeled edge, i.e.
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
    union_bbox = _combine_bboxes_zyx(
        [_nonzero_bbox_slices_zyx(arteriole_mask), _nonzero_bbox_slices_zyx(venule_mask)]
    )
    node_positions = nx.get_node_attributes(G, "pos")
    if not node_positions:
        raise ValueError("Graph has no node positions ('pos').")

    # Clear previous mask labels to keep output deterministic between reruns.
    for _, attrs in G.nodes(data=True):
        attrs.pop("mask_vessel_type", None)
    if isinstance(G, nx.MultiGraph):
        edge_iter_reset = G.edges(keys=True, data=True)
        for _u, _v, _k, attrs in edge_iter_reset:
            attrs.pop("mask_vessel_type", None)
    else:
        edge_iter_reset = G.edges(data=True)
        for _u, _v, attrs in edge_iter_reset:
            attrs.pop("mask_vessel_type", None)

    arteriole_edges: set[tuple[Any, Any, int]] = set()
    venule_edges: set[tuple[Any, Any, int]] = set()
    overlap_edges = 0
    edge_samples_cache: dict[tuple[Any, Any, int], np.ndarray] = {}
    edge_valid_indices_cache: dict[tuple[Any, Any, int], np.ndarray] = {}
    edge_attr_by_id: dict[tuple[Any, Any, int], dict[str, Any]] = {}
    total_edges = int(G.number_of_edges())
    processed_edges = 0
    start_time_s = time.perf_counter()
    print(
        "Small-vessel boundary assignment: "
        f"processing {total_edges} edge(s)."
    )
    progress_stride = max(1, total_edges // 10) if total_edges > 0 else 1

    edge_records: list[
        tuple[tuple[Any, Any, int], np.ndarray, np.ndarray, dict[str, Any]]
    ] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, key, edge_data in edge_iter:
            if u not in node_positions or v not in node_positions:
                continue
            edge_id = _edge_id(u, v, key)
            edge_records.append(
                (
                    edge_id,
                    np.asarray(node_positions[u], dtype=float),
                    np.asarray(node_positions[v], dtype=float),
                    edge_data,
                )
            )
    else:
        edge_iter = G.edges(data=True)
        for u, v, edge_data in edge_iter:
            if u not in node_positions or v not in node_positions:
                continue
            edge_id = (u, v, 0) if u <= v else (v, u, 0)
            edge_records.append(
                (
                    edge_id,
                    np.asarray(node_positions[u], dtype=float),
                    np.asarray(node_positions[v], dtype=float),
                    edge_data,
                )
            )

    def _classify_edge(
        edge_id: tuple[Any, Any, int],
        pu: np.ndarray,
        pv: np.ndarray,
        edge_data: dict[str, Any],
        *,
        use_cache: bool,
    ) -> tuple[tuple[Any, Any, int], str | None, bool]:
        if use_cache:
            samples = edge_samples_cache.get(edge_id)
            if samples is None:
                samples = _edge_sample_points_from_data(edge_data, (pu, pv))
                edge_samples_cache[edge_id] = samples
            valid_indices_zyx = edge_valid_indices_cache.get(edge_id)
            if valid_indices_zyx is None:
                _, valid_indices_zyx = _sample_points_to_valid_indices_zyx(
                    samples,
                    voxel_size_xyz=voxel_size_xyz,
                    mask_shape=arteriole_mask.shape,
                )
                edge_valid_indices_cache[edge_id] = valid_indices_zyx
        else:
            samples = _edge_sample_points_from_data(edge_data, (pu, pv))
            _, valid_indices_zyx = _sample_points_to_valid_indices_zyx(
                samples,
                voxel_size_xyz=voxel_size_xyz,
                mask_shape=arteriole_mask.shape,
            )
        if valid_indices_zyx.size == 0:
            return edge_id, None, False
        if union_bbox is not None and not _indices_intersect_bbox_zyx(valid_indices_zyx, union_bbox):
            return edge_id, None, False

        arteriole_fraction = _sample_overlap_fraction_from_valid_indices(
            valid_indices_zyx,
            arteriole_mask,
        )
        venule_fraction = _sample_overlap_fraction_from_valid_indices(
            valid_indices_zyx,
            venule_mask,
        )
        in_arteriole = arteriole_fraction >= float(minimum_overlap_fraction)
        in_venule = venule_fraction >= float(minimum_overlap_fraction)
        if in_arteriole and in_venule:
            if allow_overlap:
                return edge_id, "overlap", True
            if arteriole_fraction >= venule_fraction:
                return edge_id, "arteriole", True
            return edge_id, "venule", True
        if in_arteriole:
            return edge_id, "arteriole", False
        if in_venule:
            return edge_id, "venule", False
        return edge_id, None, False

    worker_count = int(overlap_parallel_workers)
    if worker_count > 0 and len(edge_records) > 1:
        results: list[tuple[tuple[Any, Any, int], str | None, bool]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_classify_edge, edge_id, pu, pv, edge_data, use_cache=False)
                for edge_id, pu, pv, edge_data in edge_records
            ]
            for done_idx, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if done_idx % progress_stride == 0 or done_idx == len(edge_records):
                    elapsed = time.perf_counter() - start_time_s
                    print(
                        "Small-vessel boundary assignment progress: "
                        f"{done_idx}/{len(edge_records)} edges ({elapsed:.1f}s elapsed)."
                    )
    else:
        results = []
        for processed_edges, (edge_id, pu, pv, edge_data) in enumerate(edge_records, start=1):
            results.append(
                _classify_edge(edge_id, pu, pv, edge_data, use_cache=True)
            )
            if processed_edges % progress_stride == 0 or processed_edges == len(edge_records):
                elapsed = time.perf_counter() - start_time_s
                print(
                    "Small-vessel boundary assignment progress: "
                    f"{processed_edges}/{len(edge_records)} edges ({elapsed:.1f}s elapsed)."
                )

    for edge_id, _pu, _pv, edge_data in edge_records:
        edge_attr_by_id[edge_id] = edge_data
    for edge_id, vessel_type, was_overlap in results:
        if vessel_type is None:
            continue
        if was_overlap:
            overlap_edges += 1
        if vessel_type == "arteriole":
            arteriole_edges.add(edge_id)
        elif vessel_type == "venule":
            venule_edges.add(edge_id)
        elif vessel_type == "overlap":
            arteriole_edges.add(edge_id)
            venule_edges.add(edge_id)
        edge_attr_by_id[edge_id]["mask_vessel_type"] = vessel_type

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

    def _boundary_nodes_for(edge_ids: set[tuple[Any, Any, int]], labeled_nodes: set[Any]) -> list[Any]:
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
            # Transition point from mask-labeled region to non-labeled region.
            if any(edge_id not in edge_ids for edge_id in incident_all):
                boundaries.add(node_id)
        return _sort_nodes(boundaries)

    arteriole_boundary_nodes = _boundary_nodes_for(arteriole_edges, arteriole_nodes)
    venule_boundary_nodes = _boundary_nodes_for(venule_edges, venule_nodes)
    total_elapsed = time.perf_counter() - start_time_s
    print(
        "Small-vessel boundary assignment complete: "
        f"arteriole_boundary_nodes={len(arteriole_boundary_nodes)}, "
        f"venule_boundary_nodes={len(venule_boundary_nodes)}, "
        f"elapsed={total_elapsed:.1f}s."
    )

    return {
        "arteriole_boundary_nodes": arteriole_boundary_nodes,
        "venule_boundary_nodes": venule_boundary_nodes,
        "arteriole_nodes": _sort_nodes(arteriole_nodes),
        "venule_nodes": _sort_nodes(venule_nodes),
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "overlap_edge_count": overlap_edges,
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
    }


def infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
    G: nx.Graph,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    max_dilation_microns: float,
    dilation_step_microns: float = 5.0,
    minimum_overlap_fraction: float = 0.5,
    allow_overlap: bool = False,
    exclude_smaller_overlapping_volumes: bool = False,
    overlap_parallel_workers: int = 0,
) -> dict[str, Any]:
    """Infer boundary nodes over progressive dilation steps without reassignment.

    Assignment steps always include 0 microns first, followed by fixed dilation
    increments (default: 5 microns) up to `max_dilation_microns`.

    Nodes assigned to arteriole/venule boundary groups at earlier steps are
    locked and cannot be reassigned in subsequent steps.
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

    all_nodes = set(G.nodes)
    assigned_arteriole_boundary: set[Any] = set()
    assigned_venule_boundary: set[Any] = set()
    assigned_arteriole_nodes: set[Any] = set()
    assigned_venule_nodes: set[Any] = set()
    remaining_boundary_nodes = set(all_nodes)
    remaining_label_nodes = set(all_nodes)

    base_arteriole_mask = small_arteriole_mask.astype(bool, copy=False)
    base_venule_mask = small_venule_mask.astype(bool, copy=False)
    latest_result: dict[str, Any] = {
        "arteriole_edge_count": 0,
        "venule_edge_count": 0,
        "overlap_edge_count": 0,
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
    }

    print(
        "Small-vessel progressive boundary assignment: "
        f"{len(schedule)} step(s), max_dilation={float(max_dilation_microns):.3f} microns, "
        f"step={float(dilation_step_microns):.3f} microns."
    )
    for step_idx, dilation_microns in enumerate(schedule, start=1):
        if not remaining_boundary_nodes and not remaining_label_nodes:
            print(
                "Small-vessel progressive boundary assignment: "
                "all graph nodes assigned before final dilation step."
            )
            break
        if dilation_microns <= 0:
            step_arteriole_mask = base_arteriole_mask
            step_venule_mask = base_venule_mask
        else:
            dilated_arteriole, dilated_venule = dilate_large_vessel_masks_by_microns(
                large_arteriole_mask=base_arteriole_mask,
                large_venule_mask=base_venule_mask,
                dilation_microns=float(dilation_microns),
                voxel_size_xyz=voxel_size_xyz,
            )
            if dilated_arteriole is None or dilated_venule is None:
                raise RuntimeError(
                    "Internal error: dilation unexpectedly returned None masks."
                )
            step_arteriole_mask = dilated_arteriole
            step_venule_mask = dilated_venule

        step_result = infer_boundary_nodes_from_small_vessel_masks(
            G,
            small_arteriole_mask=step_arteriole_mask,
            small_venule_mask=step_venule_mask,
            voxel_size_xyz=voxel_size_xyz,
            minimum_overlap_fraction=minimum_overlap_fraction,
            allow_overlap=allow_overlap,
            exclude_smaller_overlapping_volumes=exclude_smaller_overlapping_volumes,
            overlap_parallel_workers=overlap_parallel_workers,
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

        print(
            "Small-vessel progressive boundary assignment step "
            f"{step_idx}/{len(schedule)} "
            f"(dilation={float(dilation_microns):.3f} microns): "
            f"new_arteriole_boundary={len(step_arteriole_boundary)}, "
            f"new_venule_boundary={len(step_venule_boundary)}, "
            f"remaining_boundary_nodes={len(remaining_boundary_nodes)}."
        )

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
    output_nodes: list[Any],
    voxel_size_xyz: tuple[float, float, float],
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

    # Edges from node positions stored as (x, y, z).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    if isinstance(G, nx.MultiGraph):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, _k, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[0]), float(pv[0]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[2]), float(pv[2]), None]
    else:
        edge_iter = G.edges(data=True)
        for u, v, _data in edge_iter:
            pu = np.asarray(pos[u], dtype=float)
            pv = np.asarray(pos[v], dtype=float)
            edge_x += [float(pu[0]), float(pv[0]), None]
            edge_y += [float(pu[1]), float(pv[1]), None]
            edge_z += [float(pu[2]), float(pv[2]), None]

    input_set = set(input_nodes)
    output_set = set(output_nodes)
    other_nodes = [n for n in G.nodes if n not in input_set and n not in output_set]

    def _coords(nodes: list[Any]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        mask_bool = mask.astype(bool, copy=False)
        bbox = _nonzero_bbox_slices_zyx(mask_bool)
        if bbox is None:
            return
        x_scale, y_scale, z_scale = voxel_size_xyz
        z_slice, y_slice, x_slice = bbox
        cropped = mask_bool[z_slice, y_slice, x_slice]
        zz, yy, xx = np.indices(cropped.shape, dtype=float)
        xx = xx + float(x_slice.start)
        yy = yy + float(y_slice.start)
        zz = zz + float(z_slice.start)
        fig.add_trace(
            go.Volume(
                x=(xx * float(x_scale)).ravel(),
                y=(yy * float(y_scale)).ravel(),
                z=(zz * float(z_scale)).ravel(),
                value=cropped.astype(float).ravel(),
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
    if output_nodes:
        ox, oy, oz = _coords(output_nodes)
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
    voxel_size_xyz: tuple[float, float, float],
    output_html_path: str | Path,
    title: str = "Small Vessel Mask Boundary Labelling (3D)",
    volume_downsample_stride: int = 1,
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
    stride = max(1, int(volume_downsample_stride))

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
        lx += [float(pu[0]), float(pv[0]), None]
        ly += [float(pu[1]), float(pv[1]), None]
        lz += [float(pu[2]), float(pv[2]), None]

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
        xs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        return xs, ys, zs

    def _add_volume_trace(mask: np.ndarray, *, name: str, color: str, fig: Any) -> None:
        mask_bool = mask.astype(bool, copy=False)
        bbox = _nonzero_bbox_slices_zyx(mask_bool)
        if bbox is None:
            return
        x_scale, y_scale, z_scale = voxel_size_xyz
        z_slice, y_slice, x_slice = bbox
        cropped = mask_bool[z_slice, y_slice, x_slice]
        downsampled = _downsample_binary_mask_max(cropped, stride)
        if not np.any(downsampled):
            downsampled = cropped
            effective_stride = 1
        else:
            effective_stride = stride
        zz, yy, xx = np.indices(downsampled.shape, dtype=float)
        xx = (xx * float(effective_stride)) + float(x_slice.start)
        yy = (yy * float(effective_stride)) + float(y_slice.start)
        zz = (zz * float(effective_stride)) + float(z_slice.start)
        fig.add_trace(
            go.Volume(
                x=(xx * float(x_scale)).ravel(),
                y=(yy * float(y_scale)).ravel(),
                z=(zz * float(z_scale)).ravel(),
                value=downsampled.astype(float).ravel(),
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
