"""Pericyte-mask driven constriction mapping for Poiseuille edge weights."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree

from ImageLynx.io import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    resolve_image_path_with_optional_zip,
)
from .probability import (
    is_capillary_branch_order,
    select_active_pericyte_indices,
    validate_active_pericyte_indices,
)


def _load_binary_mask_and_voxel_size(
    mask_path: str | Path,
    *,
    h5_dataset_name: str | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a binary 3D mask and return (mask_bool, voxel_size_xyz)."""
    path = resolve_image_path_with_optional_zip(Path(mask_path))
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z = load_3d_tif_with_voxel_size(str(path))
    elif suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z = load_3d_h5_with_voxel_size(
            str(path),
            dataset_name=h5_dataset_name,
        )
    else:
        raise ValueError(
            f"Unsupported pericyte mask format '{suffix}'. "
            "Expected .tif, .tiff, or .h5."
        )
    if image.ndim != 3:
        raise ValueError(
            f"Pericyte mask must be 3D, got shape={tuple(image.shape)}."
        )
    mask = np.asarray(image) > 0
    return (
        mask,
        (float(voxel_x), float(voxel_y), float(voxel_z)),
    )


def _extract_pericyte_centroids_physical(
    mask_bool: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Return connected-component centroids in physical coordinates."""
    labels, n_labels = label(mask_bool)
    if n_labels <= 0:
        return np.empty((0, 3), dtype=float)
    indices = list(range(1, n_labels + 1))
    centroids_idx = np.asarray(center_of_mass(mask_bool, labels, indices), dtype=float)
    if centroids_idx.ndim == 1:
        centroids_idx = centroids_idx.reshape(1, 3)
    spacing = np.asarray(voxel_size_xyz, dtype=float).reshape(1, 3)
    return centroids_idx * spacing


def _extract_pericyte_component_properties(
    mask_bool: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids_phys, equivalent_diameters_um) per connected component."""
    labels, n_labels = label(mask_bool)
    if n_labels <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    indices = list(range(1, n_labels + 1))
    centroids_idx = np.asarray(center_of_mass(mask_bool, labels, indices), dtype=float)
    if centroids_idx.ndim == 1:
        centroids_idx = centroids_idx.reshape(1, 3)
    spacing = np.asarray(voxel_size_xyz, dtype=float)
    voxel_volume = float(np.prod(spacing))
    counts = np.asarray(
        [(labels == idx).sum() for idx in indices],
        dtype=float,
    )
    component_volumes_um3 = counts * voxel_volume
    # Equivalent sphere diameter from volume.
    equivalent_diameters_um = ((6.0 * component_volumes_um3) / np.pi) ** (1.0 / 3.0)
    return centroids_idx * spacing.reshape(1, 3), equivalent_diameters_um


def _edge_points(
    graph: nx.Graph,
    u: Any,
    v: Any,
    edge_data: dict[str, Any],
) -> np.ndarray:
    voxels = edge_data.get("voxels")
    if voxels is not None:
        pts = np.asarray(voxels, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 3:
            return pts
    p0 = np.asarray(graph.nodes[u]["pos"], dtype=float)
    p1 = np.asarray(graph.nodes[v]["pos"], dtype=float)
    return np.vstack([p0, p1])


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def _project_point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
    cumulative_lengths: np.ndarray,
) -> tuple[float, float]:
    """Return (arc_length_position, min_distance_to_polyline)."""
    best_dist_sq = float("inf")
    best_s = 0.0
    for idx in range(polyline.shape[0] - 1):
        a = polyline[idx]
        b = polyline[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0:
            candidate = a
            t = 0.0
        else:
            t = float(np.dot(point - a, ab) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            candidate = a + t * ab
        dist_sq = float(np.dot(point - candidate, point - candidate))
        if dist_sq < best_dist_sq:
            seg_len = float(np.linalg.norm(ab))
            best_dist_sq = dist_sq
            best_s = float(cumulative_lengths[idx] + t * seg_len)
    return best_s, float(np.sqrt(best_dist_sq))


@dataclass(frozen=True)
class _EdgeRecord:
    u: Any
    v: Any
    key: Any
    points: np.ndarray
    cumulative_lengths: np.ndarray
    length: float


def _build_edge_records(graph: nx.Graph) -> list[_EdgeRecord]:
    records: list[_EdgeRecord] = []
    if isinstance(graph, nx.MultiGraph):
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))
    for u, v, key, edge_data in edge_iter:
        if not is_capillary_branch_order(edge_data.get("branch_order")):
            # Rule: only capillary branches can receive pericyte assignments.
            continue
        points = _edge_points(graph, u, v, edge_data)
        if points.shape[0] < 2:
            continue
        cumulative_lengths = _cumulative_lengths(points)
        length = float(cumulative_lengths[-1])
        if length <= 0:
            continue
        records.append(
            _EdgeRecord(
                u=u,
                v=v,
                key=key,
                points=points,
                cumulative_lengths=cumulative_lengths,
                length=length,
            )
        )
    return records


def _assign_centroids_to_edges(
    edge_records: list[_EdgeRecord],
    pericyte_centroids_phys: np.ndarray,
) -> dict[int, tuple[tuple[Any, Any, Any], float, float]]:
    """Project each centroid to nearest edge and return mapping by centroid index."""
    if not edge_records or pericyte_centroids_phys.size == 0:
        return {}
    point_bank: list[np.ndarray] = []
    point_to_edge_idx: list[int] = []
    for edge_idx, record in enumerate(edge_records):
        point_bank.extend(record.points)
        point_to_edge_idx.extend([edge_idx] * record.points.shape[0])
    tree = cKDTree(np.asarray(point_bank, dtype=float))
    projections: dict[int, tuple[tuple[Any, Any, Any], float, float]] = {}
    for centroid_idx, centroid in enumerate(np.asarray(pericyte_centroids_phys, dtype=float)):
        _, nearest_point_idx = tree.query(centroid)
        edge_idx = int(point_to_edge_idx[int(nearest_point_idx)])
        record = edge_records[edge_idx]
        s, dist_um = _project_point_to_polyline(
            centroid,
            record.points,
            record.cumulative_lengths,
        )
        edge_key = (record.u, record.v, record.key)
        projections[int(centroid_idx)] = (edge_key, float(s), float(dist_um))
    return projections


def _diameter_at_position_from_pericytes(
    position: float,
    d1: float,
    d2: float,
    constriction_centers: list[float],
    constriction_length: float,
) -> float:
    """Piecewise linear constriction around each centroid (d1->d2->d1)."""
    if not constriction_centers or constriction_length <= 0:
        return d1

    half_window = float(constriction_length) / 2.0
    ramp_width = float(constriction_length) / 4.0
    plateau_half = float(constriction_length) / 4.0

    diameter = float(d1)
    for center in constriction_centers:
        distance_from_center = abs(float(position) - float(center))
        if distance_from_center >= half_window:
            continue
        if distance_from_center <= plateau_half:
            local_diameter = float(d2)
        else:
            if ramp_width <= 0:
                local_diameter = float(d1)
            else:
                alpha = (distance_from_center - plateau_half) / ramp_width
                local_diameter = float(d2 + (d1 - d2) * alpha)
        diameter = min(diameter, local_diameter)
    return diameter


def _integrated_resistance_from_centroid_constrictions(
    *,
    length: float,
    d1: float,
    d2: float,
    constriction_centers: list[float],
    constriction_length: float,
    num_points: int,
) -> float:
    if length <= 0:
        return float("inf")
    positions = np.linspace(0.0, float(length), int(num_points))
    diameters = np.asarray(
        [
            _diameter_at_position_from_pericytes(
                position=pos,
                d1=d1,
                d2=d2,
                constriction_centers=constriction_centers,
                constriction_length=constriction_length,
            )
            for pos in positions
        ],
        dtype=float,
    )
    diameters = np.clip(diameters, a_min=1e-9, a_max=None)
    viscosity = 1.0 / (diameters ** 1.647)
    resistance_per_length = (128.0 * viscosity) / (np.pi * (diameters ** 4))
    integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integ(resistance_per_length, x=positions))


def _resolve_d1_d2_for_edge(
    *,
    edge_data: dict[str, Any],
    branch_order: str,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    prefer_edge_fwhm_baseline: bool,
) -> tuple[float, float, bool]:
    used_fwhm_baseline = False
    if prefer_edge_fwhm_baseline:
        spec = diameter_by_branch_order.get(branch_order)
        if spec is None:
            raise ValueError(
                f"No fallback baseline diameter for branch_order '{branch_order}'."
            )
        if isinstance(spec, dict):
            raise ValueError(
                "With prefer_edge_fwhm_baseline=True, diameter_by_branch_order must "
                f"map '{branch_order}' to a numeric fallback baseline diameter."
            )
        fallback_d1 = float(spec)
        fwhm_d = edge_data.get("fwhm_diameter_um")
        if fwhm_d is not None and float(fwhm_d) > 0:
            d1 = float(fwhm_d)
            used_fwhm_baseline = True
        else:
            d1 = fallback_d1
        factor = None
        if constriction_factor_by_branch_order is not None:
            factor = constriction_factor_by_branch_order.get(branch_order)
        if factor is None:
            raise ValueError(
                f"No constriction factor for branch_order '{branch_order}'."
            )
        d2 = d1 * float(factor)
        return d1, d2, used_fwhm_baseline

    spec = diameter_by_branch_order.get(branch_order)
    if spec is None:
        raise ValueError(f"No diameter mapping for branch_order '{branch_order}'.")
    if isinstance(spec, dict):
        if "d1" not in spec or "d2" not in spec:
            raise ValueError(
                f"Invalid diameter dict for '{branch_order}'. Expected keys d1 and d2."
            )
        return float(spec["d1"]), float(spec["d2"]), used_fwhm_baseline

    d1 = float(spec)
    factor = None
    if constriction_factor_by_branch_order is not None:
        factor = constriction_factor_by_branch_order.get(branch_order)
    if factor is None:
        raise ValueError(
            f"No constriction factor for branch_order '{branch_order}' in scalar mode."
        )
    d2 = d1 * float(factor)
    return d1, d2, used_fwhm_baseline


def set_poiseuille_weights_with_pericyte_mask(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    pericyte_mask_path: str | Path,
    pericyte_mask_h5_dataset_name: str | None = None,
    prefer_edge_fwhm_baseline: bool = False,
    constriction_length: float = 40.0,
    num_integration_points: int = 1000,
    use_probabilistic_constriction: bool = False,
    constriction_probability: float = 1.0,
    active_pericyte_indices: list[int] | None = None,
    max_assignment_distance_um: float | None = 3.0,
    min_pericyte_diameter_um: float | None = 5.0,
    max_pericyte_diameter_um: float | None = 12.0,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Set conductance weights using pericyte centroids from a mask volume.

    Each connected component in ``pericyte_mask_path`` is treated as one pericyte.
    The component centroid is projected to the nearest graph edge and used as a
    constriction center. Diameter is ``d2`` in the local core around that center
    and linearly ramps to ``d1`` towards the edge of the constriction window.
    """
    if constriction_length <= 0:
        raise ValueError(
            f"constriction_length must be > 0, got {constriction_length}."
        )
    if num_integration_points < 3:
        raise ValueError(
            f"num_integration_points must be >= 3, got {num_integration_points}."
        )

    mask_bool, mask_voxel_size = _load_binary_mask_and_voxel_size(
        pericyte_mask_path,
        h5_dataset_name=pericyte_mask_h5_dataset_name,
    )
    if (
        min_pericyte_diameter_um is not None
        and max_pericyte_diameter_um is not None
        and float(min_pericyte_diameter_um) > float(max_pericyte_diameter_um)
    ):
        raise ValueError(
            "min_pericyte_diameter_um cannot be greater than max_pericyte_diameter_um."
        )

    all_centroids_phys, equivalent_diameters_um = _extract_pericyte_component_properties(
        mask_bool,
        mask_voxel_size,
    )
    total_pericytes = int(all_centroids_phys.shape[0])
    edge_records = _build_edge_records(graph)
    projection_by_centroid = _assign_centroids_to_edges(
        edge_records,
        all_centroids_phys,
    )
    eligible_indices: list[int] = []
    for centroid_idx, (_, _, dist_um) in projection_by_centroid.items():
        diameter_um = float(equivalent_diameters_um[int(centroid_idx)])
        passes_distance = (
            max_assignment_distance_um is None
            or float(dist_um) <= float(max_assignment_distance_um)
        )
        passes_min_diameter = (
            min_pericyte_diameter_um is None
            or diameter_um >= float(min_pericyte_diameter_um)
        )
        passes_max_diameter = (
            max_pericyte_diameter_um is None
            or diameter_um <= float(max_pericyte_diameter_um)
        )
        if passes_distance and passes_min_diameter and passes_max_diameter:
            eligible_indices.append(int(centroid_idx))
    eligible_indices = sorted(eligible_indices)
    eligible_set = set(eligible_indices)

    if active_pericyte_indices is not None:
        preselected_indices = validate_active_pericyte_indices(
            active_pericyte_indices,
            total_pericytes=total_pericytes,
        )
        selected_indices = [idx for idx in preselected_indices if idx in eligible_set]
        probabilistic_mode = bool(use_probabilistic_constriction)
    elif use_probabilistic_constriction:
        selected_from_eligible = select_active_pericyte_indices(
            total_pericytes=len(eligible_indices),
            constriction_probability=float(constriction_probability),
        )
        selected_indices = [eligible_indices[idx] for idx in selected_from_eligible]
        probabilistic_mode = True
    else:
        selected_indices = list(eligible_indices)
        probabilistic_mode = False

    assigned_centers_by_edge: dict[tuple[Any, Any, Any], list[float]] = {}
    assignment_distances: list[float] = []
    for centroid_idx in selected_indices:
        projection = projection_by_centroid.get(int(centroid_idx))
        if projection is None:
            continue
        edge_key, s_um, dist_um = projection
        assigned_centers_by_edge.setdefault(edge_key, []).append(float(s_um))
        assignment_distances.append(float(dist_um))

    results: dict[str, Any] = {
        "weights_set": 0,
        "pericyte_count": total_pericytes,
        "eligible_pericyte_count": int(len(eligible_indices)),
        "max_assignment_distance_um": (
            None if max_assignment_distance_um is None else float(max_assignment_distance_um)
        ),
        "min_pericyte_diameter_um": (
            None if min_pericyte_diameter_um is None else float(min_pericyte_diameter_um)
        ),
        "max_pericyte_diameter_um": (
            None if max_pericyte_diameter_um is None else float(max_pericyte_diameter_um)
        ),
        "active_pericyte_count": int(len(selected_indices)),
        "active_pericyte_indices": [int(idx) for idx in selected_indices],
        "equivalent_diameter_um_mean_all": (
            float(np.mean(equivalent_diameters_um)) if equivalent_diameters_um.size else 0.0
        ),
        "probabilistic_constriction_enabled": probabilistic_mode,
        "constriction_probability": float(constriction_probability),
        "edges_with_pericytes": int(len(assigned_centers_by_edge)),
        "used_fwhm_baseline": 0,
        "mask_voxel_size_xyz": tuple(float(v) for v in mask_voxel_size),
        "assignment_distance_um_mean": (
            float(np.mean(assignment_distances)) if assignment_distances else 0.0
        ),
        "assignment_distance_um_max": (
            float(np.max(assignment_distances)) if assignment_distances else 0.0
        ),
    }

    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        branch_order = edge_data.get("branch_order")
        if branch_order is None:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) missing required 'branch_order' attribute."
            )
        length = edge_data.get("length")
        if length is None or float(length) <= 0:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) has invalid length: {length}."
            )
        d1, d2, used_fwhm = _resolve_d1_d2_for_edge(
            edge_data=edge_data,
            branch_order=str(branch_order),
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=constriction_factor_by_branch_order,
            prefer_edge_fwhm_baseline=bool(prefer_edge_fwhm_baseline),
        )
        if used_fwhm:
            results["used_fwhm_baseline"] += 1
        if d1 <= 0 or d2 <= 0:
            raise ValueError(
                f"Edge ({u}, {v}, {key}) has non-positive diameters d1={d1}, d2={d2}."
            )

        centers = assigned_centers_by_edge.get((u, v, key), [])
        total_resistance = _integrated_resistance_from_centroid_constrictions(
            length=float(length),
            d1=float(d1),
            d2=float(d2),
            constriction_centers=centers,
            constriction_length=float(constriction_length),
            num_points=int(num_integration_points),
        )
        graph[u][v][key]["weight"] = 1.0 / float(total_resistance)
        graph[u][v][key]["pericyte_count_assigned"] = int(len(centers))
        graph[u][v][key]["pericyte_centers_um"] = [float(s) for s in centers]
        results["weights_set"] += 1
    return graph, results
