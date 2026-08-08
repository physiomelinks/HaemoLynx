"""Constriction sites taken from a segmented pericyte mask.

Each connected component of the mask is one pericyte; its centroid is projected
onto the nearest capillary and becomes a constriction site there. The narrowing
itself, and how it becomes an edge resistance, is
:mod:`haemolynx.haemodynamics.constriction`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree

from haemolynx.geometry import cumulative_lengths
from haemolynx.io import (
    CANONICAL_AXIS_ORDER,
    load_binary_mask_and_voxel_size,
    voxel_size_zyx_from_xyz,
)
from .constriction import (
    apply_constriction_sites,
    is_capillary_branch_order,
    require_enough_integration_points,
    require_positive_constriction_length,
    select_active_pericyte_indices,
    validate_active_pericyte_indices,
)

#: Names the pericyte mask in loader errors, so a bad path or format says which
#: of the pipeline's several masks was at fault.
PERICYTE_MASK_DESCRIPTION = "pericyte mask"


def _extract_pericyte_centroids_physical(
    mask_bool: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Return connected-component centroids in physical coordinates."""
    labels, n_labels = label(mask_bool)
    if n_labels <= 0:
        return np.empty((0, 3), dtype=float)
    indices = list(range(1, n_labels + 1))
    centroids_idx = np.asarray(center_of_mass(mask_bool, labels, indices), dtype=float)
    if centroids_idx.ndim == 1:
        centroids_idx = centroids_idx.reshape(1, 3)
    spacing = np.asarray(voxel_size_zyx, dtype=float).reshape(1, 3)
    return centroids_idx * spacing


def _extract_pericyte_component_properties(
    mask_bool: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids_phys, equivalent_diameters_um) per connected component."""
    labels, n_labels = label(mask_bool)
    if n_labels <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    indices = list(range(1, n_labels + 1))
    centroids_idx = np.asarray(center_of_mass(mask_bool, labels, indices), dtype=float)
    if centroids_idx.ndim == 1:
        centroids_idx = centroids_idx.reshape(1, 3)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    counts = np.asarray(
        [(labels == idx).sum() for idx in indices],
        dtype=float,
    )
    component_volumes_um3 = counts * voxel_volume
    # Equivalent sphere diameter from volume.
    equivalent_diameters_um = ((6.0 * component_volumes_um3) / np.pi) ** (1.0 / 3.0)
    return centroids_idx * spacing.reshape(1, 3), equivalent_diameters_um


def _edge_centerline_points(
    graph: nx.Graph,
    u: Any,
    v: Any,
    edge_data: dict[str, Any],
) -> np.ndarray:
    """Centerline of one edge as strict ``(n, 3)`` physical points.

    Anything that is not already an ``(n>=2, 3)`` array of voxel coordinates —
    including a 2D or padded polyline — is discarded in favour of the straight
    node-to-node segment, because pericyte centroids are projected onto this in
    3D and a reshaped polyline would move the projection.
    """
    voxels = edge_data.get("voxels")
    if voxels is not None:
        pts = np.asarray(voxels, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 3:
            return pts
    p0 = np.asarray(graph.nodes[u]["pos"], dtype=float)
    p1 = np.asarray(graph.nodes[v]["pos"], dtype=float)
    return np.vstack([p0, p1])


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
        points = _edge_centerline_points(graph, u, v, edge_data)
        if points.shape[0] < 2:
            continue
        arc_lengths = cumulative_lengths(points)
        length = float(arc_lengths[-1])
        if length <= 0:
            continue
        records.append(
            _EdgeRecord(
                u=u,
                v=v,
                key=key,
                points=points,
                cumulative_lengths=arc_lengths,
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


class MaskConstrictionSites:
    """Sites already fixed to edges by projecting mask centroids onto them.

    The mask decides everything before the run reaches an edge, so this simply
    hands back what was assigned to ``(u, v, key)`` and reports how the
    assignment went.
    """

    def __init__(
        self,
        *,
        assigned_centers_by_edge: dict[tuple[Any, Any, Any], list[float]],
        summary_fields: dict[str, Any],
    ) -> None:
        self._assigned_centers_by_edge = assigned_centers_by_edge
        self._summary_fields = summary_fields

    def centers_for_edge(
        self,
        u: Any,
        v: Any,
        key: Any,
        edge_data: dict[str, Any],
        *,
        length: float,
    ) -> list[float]:
        return self._assigned_centers_by_edge.get((u, v, key), [])

    def summary(self) -> dict[str, Any]:
        return dict(self._summary_fields)


def set_poiseuille_resistances_with_pericyte_mask(
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
    axis_order: str = CANONICAL_AXIS_ORDER,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Set edge resistance/conductance using pericyte centroids from a mask volume.

    Each connected component in ``pericyte_mask_path`` is treated as one pericyte.
    The component centroid is projected to the nearest graph edge and used as a
    constriction center. Diameter is ``d2`` in the local core around that center
    and linearly ramps to ``d1`` towards the edge of the constriction window.

    With ``use_probabilistic_constriction=True`` the active cohort is drawn from
    ``rng`` if given, else from a generator built on ``seed``; ``seed=None``
    means a different cohort on every call.
    """
    require_positive_constriction_length(constriction_length)
    require_enough_integration_points(num_integration_points)

    mask_bool, mask_voxel_size = load_binary_mask_and_voxel_size(
        pericyte_mask_path,
        h5_dataset_name=pericyte_mask_h5_dataset_name,
        axis_order=axis_order,
        description=PERICYTE_MASK_DESCRIPTION,
    )
    mask_voxel_size_zyx = voxel_size_zyx_from_xyz(mask_voxel_size)
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
        mask_voxel_size_zyx,
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
            rng=rng,
            seed=seed,
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

    assignment_summary: dict[str, Any] = {
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
        "mask_voxel_size_xyz": tuple(float(v) for v in mask_voxel_size),
        "assignment_distance_um_mean": (
            float(np.mean(assignment_distances)) if assignment_distances else 0.0
        ),
        "assignment_distance_um_max": (
            float(np.max(assignment_distances)) if assignment_distances else 0.0
        ),
    }

    sites = MaskConstrictionSites(
        assigned_centers_by_edge=assigned_centers_by_edge,
        summary_fields=assignment_summary,
    )
    return apply_constriction_sites(
        graph,
        sites,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order=constriction_factor_by_branch_order,
        prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
        constriction_length=constriction_length,
        num_integration_points=num_integration_points,
    )
