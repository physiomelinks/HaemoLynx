"""Type-locked cylinder continuity helpers for small-vessel masks."""
from __future__ import annotations

from typing import Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from scipy.spatial import cKDTree

_GPU_EDT_AVAILABLE: bool | None = None
_GPU_CP: Any = None
_GPU_NDIMAGE: Any = None


def _init_gpu_edt_backend() -> bool:
    global _GPU_EDT_AVAILABLE, _GPU_CP, _GPU_NDIMAGE
    if _GPU_EDT_AVAILABLE is not None:
        return bool(_GPU_EDT_AVAILABLE)
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy import ndimage as cpx_ndimage  # type: ignore

        _GPU_CP = cp
        _GPU_NDIMAGE = cpx_ndimage
        _GPU_EDT_AVAILABLE = True
    except Exception:
        _GPU_CP = None
        _GPU_NDIMAGE = None
        _GPU_EDT_AVAILABLE = False
    return bool(_GPU_EDT_AVAILABLE)


def _edt(
    mask: np.ndarray,
    *,
    sampling_zyx: tuple[float, float, float],
    use_gpu_acceleration: bool,
) -> np.ndarray:
    if not bool(use_gpu_acceleration):
        return distance_transform_edt(mask, sampling=sampling_zyx)
    if not _init_gpu_edt_backend():
        return distance_transform_edt(mask, sampling=sampling_zyx)
    try:
        cp = _GPU_CP
        cpx_ndimage = _GPU_NDIMAGE
        gpu_mask = cp.asarray(mask)
        gpu_result = cpx_ndimage.distance_transform_edt(gpu_mask, sampling=sampling_zyx)
        return cp.asnumpy(gpu_result)
    except Exception:
        return distance_transform_edt(mask, sampling=sampling_zyx)


def _voxel_size_xyz_to_sampling_zyx(
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(voxel_size_xyz[2]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[0]),
    )


def _line_indices_zyx(start_zyx: np.ndarray, end_zyx: np.ndarray) -> np.ndarray:
    delta = end_zyx.astype(float) - start_zyx.astype(float)
    step_count = int(max(2, np.ceil(np.linalg.norm(delta)) + 1))
    t = np.linspace(0.0, 1.0, step_count, dtype=float)
    points = np.rint(
        start_zyx.reshape(1, 3) * (1.0 - t.reshape(-1, 1))
        + end_zyx.reshape(1, 3) * t.reshape(-1, 1)
    ).astype(int)
    return np.unique(points, axis=0)


def _clip_indices_to_shape(indices_zyx: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if indices_zyx.size == 0:
        return indices_zyx
    valid = np.all(indices_zyx >= 0, axis=1) & (
        indices_zyx[:, 0] < int(shape[0])
    ) & (indices_zyx[:, 1] < int(shape[1])) & (indices_zyx[:, 2] < int(shape[2]))
    return indices_zyx[valid]


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, count = label(mask.astype(bool, copy=False), structure=structure)
    return labeled, int(count)


def _component_descriptors(
    mask: np.ndarray,
    edt_inside: np.ndarray | None = None,
    *,
    labeled: np.ndarray | None = None,
    count: int | None = None,
) -> dict[int, dict[str, Any]]:
    if labeled is None or count is None:
        labeled, count = _connected_components(mask)
    descriptors: dict[int, dict[str, Any]] = {}
    if int(count) <= 0:
        return descriptors
    component_slices = find_objects(labeled, max_label=int(count))
    for component_id in range(1, int(count) + 1):
        component_slice = component_slices[component_id - 1] if component_slices else None
        if component_slice is None:
            continue
        labeled_roi = labeled[component_slice]
        local_coords = np.argwhere(labeled_roi == int(component_id))
        if local_coords.size == 0:
            continue
        offset = np.asarray(
            [
                int(component_slice[0].start),
                int(component_slice[1].start),
                int(component_slice[2].start),
            ],
            dtype=int,
        )
        coords = local_coords + offset.reshape(1, 3)
        if coords.size == 0:
            continue
        centroid = np.mean(coords.astype(float), axis=0)
        if coords.shape[0] >= 3:
            cov = np.cov(coords.astype(float).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            principal_axis = eigvecs[:, 0]
            linearity = float((eigvals[0] - eigvals[1]) / max(1e-9, float(eigvals[0])))
        else:
            principal_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
            linearity = 0.0
        projections = coords.astype(float) @ principal_axis.reshape(3, 1)
        min_idx = int(np.argmin(projections[:, 0]))
        max_idx = int(np.argmax(projections[:, 0]))
        end_a = coords[min_idx].astype(int)
        end_b = coords[max_idx].astype(int)
        if edt_inside is not None:
            radii = edt_inside[
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
            ]
            median_radius = float(np.median(radii)) if radii.size else 0.0
        else:
            median_radius = 0.0
        descriptors[component_id] = {
            "component_id": int(component_id),
            "coords_zyx": coords,
            "centroid_zyx": centroid,
            "principal_axis_zyx": principal_axis / max(1e-9, np.linalg.norm(principal_axis)),
            "linearity": float(linearity),
            "endpoints_zyx": (end_a, end_b),
            "median_radius_microns": float(median_radius),
            "size_voxels": int(coords.shape[0]),
        }
    return descriptors


def _bbox_from_masks_and_points(
    *,
    shape_zyx: tuple[int, int, int],
    masks: list[np.ndarray],
    points_zyx: np.ndarray | None,
    margin_vox_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    mins = np.asarray([shape_zyx[0], shape_zyx[1], shape_zyx[2]], dtype=int)
    maxs = np.asarray([0, 0, 0], dtype=int)
    found = False
    for mask in masks:
        coords = np.argwhere(mask.astype(bool, copy=False))
        if coords.size == 0:
            continue
        found = True
        mins = np.minimum(mins, coords.min(axis=0).astype(int))
        maxs = np.maximum(maxs, coords.max(axis=0).astype(int) + 1)
    if points_zyx is not None and points_zyx.size > 0:
        pts = np.asarray(points_zyx, dtype=int)
        found = True
        mins = np.minimum(mins, pts.min(axis=0).astype(int))
        maxs = np.maximum(maxs, pts.max(axis=0).astype(int) + 1)
    if not found:
        return (
            slice(0, int(shape_zyx[0])),
            slice(0, int(shape_zyx[1])),
            slice(0, int(shape_zyx[2])),
        )

    margin = np.asarray(margin_vox_zyx, dtype=int)
    mins = np.maximum(np.asarray([0, 0, 0], dtype=int), mins - margin)
    maxs = np.minimum(np.asarray(shape_zyx, dtype=int), maxs + margin)
    return (
        slice(int(mins[0]), int(maxs[0])),
        slice(int(mins[1]), int(maxs[1])),
        slice(int(mins[2]), int(maxs[2])),
    )


def _global_to_local_index_zyx(
    index_zyx: np.ndarray,
    bbox_zyx: tuple[slice, slice, slice] | None,
) -> np.ndarray:
    if bbox_zyx is None:
        return np.asarray(index_zyx, dtype=int)
    z_slice, y_slice, x_slice = bbox_zyx
    idx = np.asarray(index_zyx, dtype=int)
    return np.asarray(
        [
            int(idx[0]) - int(z_slice.start),
            int(idx[1]) - int(y_slice.start),
            int(idx[2]) - int(x_slice.start),
        ],
        dtype=int,
    )


def _global_to_local_indices_zyx(
    indices_zyx: np.ndarray,
    bbox_zyx: tuple[slice, slice, slice] | None,
) -> np.ndarray:
    if bbox_zyx is None:
        return np.asarray(indices_zyx, dtype=int)
    z_slice, y_slice, x_slice = bbox_zyx
    indices = np.asarray(indices_zyx, dtype=int)
    out = indices.copy()
    out[:, 0] = out[:, 0] - int(z_slice.start)
    out[:, 1] = out[:, 1] - int(y_slice.start)
    out[:, 2] = out[:, 2] - int(x_slice.start)
    return out


def _local_signed_gradient(
    signed_distance: np.ndarray,
    *,
    index_zyx: tuple[int, int, int],
    sampling_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Central-difference signed-distance gradient at one voxel."""
    z, y, x = int(index_zyx[0]), int(index_zyx[1]), int(index_zyx[2])
    z_max, y_max, x_max = (
        int(signed_distance.shape[0]) - 1,
        int(signed_distance.shape[1]) - 1,
        int(signed_distance.shape[2]) - 1,
    )
    z_lo, z_hi = max(0, z - 1), min(z_max, z + 1)
    y_lo, y_hi = max(0, y - 1), min(y_max, y + 1)
    x_lo, x_hi = max(0, x - 1), min(x_max, x + 1)
    dz_den = max(1e-9, float((z_hi - z_lo) * sampling_zyx[0]))
    dy_den = max(1e-9, float((y_hi - y_lo) * sampling_zyx[1]))
    dx_den = max(1e-9, float((x_hi - x_lo) * sampling_zyx[2]))
    gz = float(signed_distance[z_hi, y, x] - signed_distance[z_lo, y, x]) / dz_den
    gy = float(signed_distance[z, y_hi, x] - signed_distance[z, y_lo, x]) / dy_den
    gx = float(signed_distance[z, y, x_hi] - signed_distance[z, y, x_lo]) / dx_den
    return np.asarray([gz, gy, gx], dtype=float)


def _evaluate_endpoint_tangency_to_large_mask(
    endpoint_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    *,
    signed_distance_to_large: np.ndarray,
    distance_to_opposite_large: np.ndarray,
    sampling_zyx: tuple[float, float, float],
    max_contact_distance_microns: float,
    tangency_cosine_max: float,
    opposite_exclusion_distance_microns: float,
    tangency_weight: float,
    opposite_penalty_weight: float,
    roi_bbox_zyx: tuple[slice, slice, slice] | None = None,
    precomputed_distance_to_large_microns: np.ndarray | None = None,
) -> dict[str, Any]:
    idx = np.asarray(endpoint_zyx, dtype=int)
    local_idx = _global_to_local_index_zyx(idx, roi_bbox_zyx)
    z, y, x = int(local_idx[0]), int(local_idx[1]), int(local_idx[2])
    shape = signed_distance_to_large.shape
    if not (0 <= z < int(shape[0]) and 0 <= y < int(shape[1]) and 0 <= x < int(shape[2])):
        return {
            "valid": False,
            "reason": "endpoint_out_of_bounds",
            "score": float(np.inf),
            "distance_to_large_microns": float(np.inf),
            "tangency_cosine": float(1.0),
            "distance_to_opposite_large_microns": float(np.inf),
        }

    if precomputed_distance_to_large_microns is not None:
        d_large = float(precomputed_distance_to_large_microns[z, y, x])
    else:
        d_large = float(abs(signed_distance_to_large[z, y, x]))
    if d_large > float(max_contact_distance_microns):
        return {
            "valid": False,
            "reason": "too_far_from_large_mask",
            "score": float(np.inf),
            "distance_to_large_microns": float(d_large),
            "tangency_cosine": float(1.0),
            "distance_to_opposite_large_microns": float(distance_to_opposite_large[z, y, x]),
        }
    normal = _local_signed_gradient(
        signed_distance_to_large,
        index_zyx=(z, y, x),
        sampling_zyx=sampling_zyx,
    )
    n_norm = float(np.linalg.norm(normal))
    if n_norm <= 1e-9:
        return {
            "valid": False,
            "reason": "undefined_boundary_normal",
            "score": float(np.inf),
            "distance_to_large_microns": float(d_large),
            "tangency_cosine": float(1.0),
            "distance_to_opposite_large_microns": float(distance_to_opposite_large[z, y, x]),
        }
    normal_hat = normal / n_norm
    axis = np.asarray(axis_zyx, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        axis_hat = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        axis_hat = axis / axis_norm
    tangency_cos = float(abs(np.dot(axis_hat, normal_hat)))
    if tangency_cos > float(tangency_cosine_max):
        return {
            "valid": False,
            "reason": "not_tangential_to_large_mask",
            "score": float(np.inf),
            "distance_to_large_microns": float(d_large),
            "tangency_cosine": float(tangency_cos),
            "distance_to_opposite_large_microns": float(distance_to_opposite_large[z, y, x]),
        }

    d_opp = float(distance_to_opposite_large[z, y, x])
    opposite_penalty = max(0.0, float(opposite_exclusion_distance_microns) - d_opp)
    score = (
        float(d_large)
        + float(tangency_weight) * float(tangency_cos)
        + float(opposite_penalty_weight) * float(opposite_penalty)
    )
    return {
        "valid": True,
        "reason": "ok",
        "score": float(score),
        "distance_to_large_microns": float(d_large),
        "tangency_cosine": float(tangency_cos),
        "distance_to_opposite_large_microns": float(d_opp),
    }


def _best_neighbor_for_endpoint(
    *,
    current_component_id: int,
    endpoint_zyx: np.ndarray,
    current_axis_zyx: np.ndarray,
    target_type: str,
    descriptors: dict[int, dict[str, Any]],
    endpoint_lookup_by_type: dict[str, dict[str, Any]],
    max_endpoint_distance_microns: float,
    min_facing_cosine: float,
    max_axis_angle_degrees: float,
    sampling_zyx: tuple[float, float, float],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    e = np.asarray(endpoint_zyx, dtype=float)
    spacing = np.asarray(sampling_zyx, dtype=float)
    query_point = e * spacing
    current_axis = np.asarray(current_axis_zyx, dtype=float)
    lookup = endpoint_lookup_by_type.get(str(target_type)) or {}
    tree = lookup.get("tree")
    endpoint_points = lookup.get("endpoint_points_zyx")
    endpoint_component_ids = lookup.get("endpoint_component_ids")
    if tree is None or endpoint_points is None or endpoint_component_ids is None:
        return None
    candidate_indices = tree.query_ball_point(
        query_point,
        r=float(max_endpoint_distance_microns),
    )
    for endpoint_idx in candidate_indices:
        other_id = int(endpoint_component_ids[int(endpoint_idx)])
        if int(other_id) == int(current_component_id):
            continue
        other = descriptors.get(int(other_id))
        if other is None:
            continue
        other_axis = np.asarray(other["principal_axis_zyx"], dtype=float)
        axis_angle = _axis_angle_deg(current_axis, other_axis)
        if axis_angle > float(max_axis_angle_degrees):
            continue
        other_endpoint = np.asarray(endpoint_points[int(endpoint_idx)], dtype=float)
        endpoint_distance = float(np.linalg.norm((other_endpoint - e) * spacing))
        v = np.asarray(other_endpoint, dtype=float) - e
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-9:
            continue
        v_hat = v / v_norm
        cur_axis = current_axis.copy()
        oth_axis = other_axis.copy()
        if float(np.dot(cur_axis, v_hat)) < 0.0:
            cur_axis = -cur_axis
        if float(np.dot(oth_axis, -v_hat)) < 0.0:
            oth_axis = -oth_axis
        cur_face = float(np.dot(cur_axis / max(1e-9, np.linalg.norm(cur_axis)), v_hat))
        oth_face = float(np.dot(oth_axis / max(1e-9, np.linalg.norm(oth_axis)), -v_hat))
        if cur_face < float(min_facing_cosine) or oth_face < float(min_facing_cosine):
            continue
        candidate = {
            "component_id": int(other_id),
            "distance_microns": float(endpoint_distance),
            "axis_angle_degrees": float(axis_angle),
            "current_facing": float(cur_face),
            "other_facing": float(oth_face),
        }
        if best is None or candidate["distance_microns"] < best["distance_microns"]:
            best = candidate
    return best


def _reassign_sandwiched_components(
    *,
    arteriole_mask: np.ndarray,
    venule_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    max_endpoint_distance_microns: float,
    min_facing_cosine: float,
    max_axis_angle_degrees: float,
    parallel_workers: int = 0,
) -> dict[str, Any]:
    all_small = arteriole_mask.astype(bool, copy=False) | venule_mask.astype(bool, copy=False)
    sampling_zyx = _voxel_size_xyz_to_sampling_zyx(voxel_size_xyz)
    labeled, comp_count = _connected_components(all_small)
    descriptors = _component_descriptors(
        all_small,
        None,
        labeled=labeled,
        count=comp_count,
    )
    component_type: dict[int, str] = {}
    for comp_id, desc in descriptors.items():
        coords = desc["coords_zyx"]
        art_frac = float(
            np.count_nonzero(arteriole_mask[coords[:, 0], coords[:, 1], coords[:, 2]])
        ) / float(max(1, coords.shape[0]))
        component_type[int(comp_id)] = "arteriole" if art_frac >= 0.5 else "venule"
    endpoint_lookup_by_type: dict[str, dict[str, Any]] = {}
    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    for vessel_type in ("arteriole", "venule"):
        endpoint_points: list[np.ndarray] = []
        endpoint_component_ids: list[int] = []
        for comp_id, desc in descriptors.items():
            if component_type.get(int(comp_id), "") != vessel_type:
                continue
            end_a, end_b = desc["endpoints_zyx"]
            endpoint_points.append(np.asarray(end_a, dtype=float))
            endpoint_component_ids.append(int(comp_id))
            endpoint_points.append(np.asarray(end_b, dtype=float))
            endpoint_component_ids.append(int(comp_id))
        if not endpoint_points:
            endpoint_lookup_by_type[vessel_type] = {
                "tree": None,
                "endpoint_points_zyx": np.empty((0, 3), dtype=float),
                "endpoint_component_ids": np.empty((0,), dtype=int),
            }
            continue
        endpoint_arr = np.asarray(endpoint_points, dtype=float)
        endpoint_lookup_by_type[vessel_type] = {
            "tree": cKDTree(endpoint_arr * spacing),
            "endpoint_points_zyx": endpoint_arr,
            "endpoint_component_ids": np.asarray(endpoint_component_ids, dtype=int),
        }

    def _evaluate_one_component(comp_id: int) -> tuple[int, str | None]:
        desc = descriptors[int(comp_id)]
        current_type = component_type.get(int(comp_id), "")
        if current_type not in {"arteriole", "venule"}:
            return int(comp_id), None
        target_type = "venule" if current_type == "arteriole" else "arteriole"
        axis = np.asarray(desc["principal_axis_zyx"], dtype=float)
        end_a, end_b = desc["endpoints_zyx"]
        support_a = _best_neighbor_for_endpoint(
            current_component_id=int(comp_id),
            endpoint_zyx=np.asarray(end_a, dtype=int),
            current_axis_zyx=axis,
            target_type=target_type,
            descriptors=descriptors,
            endpoint_lookup_by_type=endpoint_lookup_by_type,
            max_endpoint_distance_microns=max_endpoint_distance_microns,
            min_facing_cosine=min_facing_cosine,
            max_axis_angle_degrees=max_axis_angle_degrees,
            sampling_zyx=sampling_zyx,
        )
        support_b = _best_neighbor_for_endpoint(
            current_component_id=int(comp_id),
            endpoint_zyx=np.asarray(end_b, dtype=int),
            current_axis_zyx=axis,
            target_type=target_type,
            descriptors=descriptors,
            endpoint_lookup_by_type=endpoint_lookup_by_type,
            max_endpoint_distance_microns=max_endpoint_distance_microns,
            min_facing_cosine=min_facing_cosine,
            max_axis_angle_degrees=max_axis_angle_degrees,
            sampling_zyx=sampling_zyx,
        )
        if support_a is None or support_b is None:
            return int(comp_id), None
        # Prefer true "middle between two labeled volumes": different neighbors at each end.
        if int(support_a["component_id"]) == int(support_b["component_id"]):
            return int(comp_id), None
        return int(comp_id), str(target_type)

    flips: dict[int, str] = {}
    component_ids = sorted(int(cid) for cid in descriptors.keys())
    worker_count = int(parallel_workers)
    if worker_count > 0 and len(component_ids) > 1:
        results: list[tuple[int, str | None]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_evaluate_one_component, int(comp_id)): int(comp_id)
                for comp_id in component_ids
            }
            for future in as_completed(future_map):
                results.append(future.result())
        for comp_id, target_type in sorted(results, key=lambda row: int(row[0])):
            if target_type is None:
                continue
            flips[int(comp_id)] = str(target_type)
    else:
        for comp_id in component_ids:
            decided_comp_id, target_type = _evaluate_one_component(int(comp_id))
            if target_type is None:
                continue
            flips[int(decided_comp_id)] = str(target_type)

    out_art = arteriole_mask.astype(bool, copy=False).copy()
    out_ven = venule_mask.astype(bool, copy=False).copy()
    flip_to_art = 0
    flip_to_ven = 0
    for comp_id, new_type in flips.items():
        coords = np.argwhere(labeled == int(comp_id))
        if coords.size == 0:
            continue
        if new_type == "arteriole":
            out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = False
            out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            flip_to_art += 1
        else:
            out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = False
            out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            flip_to_ven += 1

    return {
        "small_arteriole_mask": out_art,
        "small_venule_mask": out_ven,
        "stats": {
            "component_count": int(comp_count),
            "sandwiched_flips_to_arteriole": int(flip_to_art),
            "sandwiched_flips_to_venule": int(flip_to_ven),
        },
    }


def redefine_small_masks_from_large_tangential_contact(
    *,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    voxel_size_xyz: tuple[float, float, float],
    enable_redefinition: bool = True,
    max_contact_distance_microns: float = 12.0,
    touch_distance_microns: float = 3.0,
    tangency_cosine_max: float = 0.35,
    reassignment_margin: float = 0.10,
    opposite_exclusion_distance_microns: float = 3.0,
    tangency_weight: float = 4.0,
    opposite_penalty_weight: float = 4.0,
    enable_sandwiched_component_reassignment: bool = True,
    sandwiched_max_endpoint_distance_microns: float = 12.0,
    sandwiched_min_facing_cosine: float = 0.82,
    sandwiched_max_axis_angle_degrees: float = 45.0,
    reassignment_parallel_workers: int = 0,
    use_gpu_acceleration: bool = False,
) -> dict[str, Any]:
    """Redefine small-vessel labels using tangential contact to large masks.

    Components are only redefined when their endpoint is near and tangential to
    one large-vessel class, and the score margin over the opposite class is large.
    """
    art_small = small_arteriole_mask.astype(bool, copy=False)
    ven_small = small_venule_mask.astype(bool, copy=False)
    if art_small.shape != ven_small.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {art_small.shape} and {ven_small.shape}."
        )
    if not enable_redefinition:
        return {
            "small_arteriole_mask": art_small.copy(),
            "small_venule_mask": ven_small.copy(),
            "stats": {
                "redefinition_enabled": False,
                "reassigned_to_arteriole": 0,
                "reassigned_to_venule": 0,
                "unresolved_components": 0,
                "component_count": 0,
            },
        }
    have_large_masks = large_arteriole_mask is not None and large_venule_mask is not None
    sampling_zyx = _voxel_size_xyz_to_sampling_zyx(voxel_size_xyz)
    all_small = art_small | ven_small
    total_start_s = time.perf_counter()
    _labeled_all, comp_count = _connected_components(all_small)
    descriptors = _component_descriptors(
        all_small,
        None,
        labeled=_labeled_all,
        count=comp_count,
    )
    if have_large_masks:
        large_art = large_arteriole_mask.astype(bool, copy=False)
        large_ven = large_venule_mask.astype(bool, copy=False)
        if large_art.shape != art_small.shape or large_ven.shape != art_small.shape:
            raise ValueError(
                "All small/large masks must share the same shape for redefinition."
            )
        endpoint_points: list[np.ndarray] = []
        for desc in descriptors.values():
            end_a, end_b = desc["endpoints_zyx"]
            endpoint_points.append(np.asarray(end_a, dtype=int))
            endpoint_points.append(np.asarray(end_b, dtype=int))
        endpoints_arr = (
            np.asarray(endpoint_points, dtype=int)
            if endpoint_points
            else np.empty((0, 3), dtype=int)
        )
        max_margin_microns = max(
            float(max_contact_distance_microns),
            float(touch_distance_microns),
            float(opposite_exclusion_distance_microns),
        )
        margin_zyx = (
            int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[0])))),
            int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[1])))),
            int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[2])))),
        )
        roi_bbox_zyx = _bbox_from_masks_and_points(
            shape_zyx=art_small.shape,
            masks=[large_art, large_ven],
            points_zyx=endpoints_arr,
            margin_vox_zyx=margin_zyx,
        )
        z_slice, y_slice, x_slice = roi_bbox_zyx
        art_roi = large_art[z_slice, y_slice, x_slice]
        ven_roi = large_ven[z_slice, y_slice, x_slice]
        dist_to_art = _edt(
            ~art_roi,
            sampling_zyx=sampling_zyx,
            use_gpu_acceleration=bool(use_gpu_acceleration),
        )
        dist_to_ven = _edt(
            ~ven_roi,
            sampling_zyx=sampling_zyx,
            use_gpu_acceleration=bool(use_gpu_acceleration),
        )
        inside_art = _edt(
            art_roi,
            sampling_zyx=sampling_zyx,
            use_gpu_acceleration=bool(use_gpu_acceleration),
        )
        inside_ven = _edt(
            ven_roi,
            sampling_zyx=sampling_zyx,
            use_gpu_acceleration=bool(use_gpu_acceleration),
        )
        signed_art = dist_to_art - inside_art
        signed_ven = dist_to_ven - inside_ven
    else:
        roi_bbox_zyx = None
        signed_art = np.zeros_like(art_small, dtype=float)
        signed_ven = np.zeros_like(ven_small, dtype=float)
        dist_to_art = np.full_like(art_small, fill_value=np.inf, dtype=float)
        dist_to_ven = np.full_like(ven_small, fill_value=np.inf, dtype=float)

    out_art = art_small.copy()
    out_ven = ven_small.copy()

    reassigned_to_art = 0
    reassigned_to_ven = 0
    unresolved = 0
    setup_phase_elapsed_s = time.perf_counter() - total_start_s
    tangential_phase_start_s = time.perf_counter()
    if have_large_masks:
        def _evaluate_component_reassignment(
            comp_id: int,
            desc: dict[str, Any],
        ) -> tuple[int, str, np.ndarray]:
            coords = desc["coords_zyx"]
            if coords.size == 0:
                return int(comp_id), "skip", coords
            source_art_fraction = float(
                np.count_nonzero(art_small[coords[:, 0], coords[:, 1], coords[:, 2]])
            ) / float(coords.shape[0])
            source_type = "arteriole" if source_art_fraction >= 0.5 else "venule"
            axis = np.asarray(desc["principal_axis_zyx"], dtype=float)
            endpoints = desc["endpoints_zyx"]

            best_art = {
                "valid": False,
                "score": float(np.inf),
                "distance_to_large_microns": float(np.inf),
                "tangency_cosine": float(1.0),
            }
            best_ven = {
                "valid": False,
                "score": float(np.inf),
                "distance_to_large_microns": float(np.inf),
                "tangency_cosine": float(1.0),
            }
            for endpoint in endpoints:
                endpoint_idx_local = _global_to_local_index_zyx(
                    np.asarray(endpoint, dtype=int),
                    roi_bbox_zyx,
                )
                ez, ey, ex = (
                    int(endpoint_idx_local[0]),
                    int(endpoint_idx_local[1]),
                    int(endpoint_idx_local[2]),
                )
                dist_art_fast = float(np.inf)
                dist_ven_fast = float(np.inf)
                if (
                    0 <= ez < int(dist_to_art.shape[0])
                    and 0 <= ey < int(dist_to_art.shape[1])
                    and 0 <= ex < int(dist_to_art.shape[2])
                ):
                    dist_art_fast = float(dist_to_art[ez, ey, ex])
                    dist_ven_fast = float(dist_to_ven[ez, ey, ex])
                # Cheap prefilter before gradient/tangency math.
                if (
                    dist_art_fast > float(max_contact_distance_microns)
                    and dist_ven_fast > float(max_contact_distance_microns)
                ):
                    continue
                art_eval = _evaluate_endpoint_tangency_to_large_mask(
                    np.asarray(endpoint, dtype=int),
                    axis,
                    signed_distance_to_large=signed_art,
                    distance_to_opposite_large=dist_to_ven,
                    sampling_zyx=sampling_zyx,
                    max_contact_distance_microns=float(max_contact_distance_microns),
                    tangency_cosine_max=float(tangency_cosine_max),
                    opposite_exclusion_distance_microns=float(
                        opposite_exclusion_distance_microns
                    ),
                    tangency_weight=float(tangency_weight),
                    opposite_penalty_weight=float(opposite_penalty_weight),
                    roi_bbox_zyx=roi_bbox_zyx,
                    precomputed_distance_to_large_microns=dist_to_art,
                )
                if art_eval["score"] < best_art["score"]:
                    best_art = art_eval
                ven_eval = _evaluate_endpoint_tangency_to_large_mask(
                    np.asarray(endpoint, dtype=int),
                    axis,
                    signed_distance_to_large=signed_ven,
                    distance_to_opposite_large=dist_to_art,
                    sampling_zyx=sampling_zyx,
                    max_contact_distance_microns=float(max_contact_distance_microns),
                    tangency_cosine_max=float(tangency_cosine_max),
                    opposite_exclusion_distance_microns=float(
                        opposite_exclusion_distance_microns
                    ),
                    tangency_weight=float(tangency_weight),
                    opposite_penalty_weight=float(opposite_penalty_weight),
                    roi_bbox_zyx=roi_bbox_zyx,
                    precomputed_distance_to_large_microns=dist_to_ven,
                )
                if ven_eval["score"] < best_ven["score"]:
                    best_ven = ven_eval

            art_valid = bool(best_art["valid"]) and (
                float(best_art["distance_to_large_microns"])
                <= float(touch_distance_microns)
            )
            ven_valid = bool(best_ven["valid"]) and (
                float(best_ven["distance_to_large_microns"])
                <= float(touch_distance_microns)
            )
            target_type: str | None = None
            if art_valid and not ven_valid:
                target_type = "arteriole"
            elif ven_valid and not art_valid:
                target_type = "venule"
            elif art_valid and ven_valid:
                if (float(best_ven["score"]) - float(best_art["score"])) >= float(
                    reassignment_margin
                ):
                    target_type = "arteriole"
                elif (float(best_art["score"]) - float(best_ven["score"])) >= float(
                    reassignment_margin
                ):
                    target_type = "venule"
                else:
                    return int(comp_id), "unresolved", coords
            else:
                return int(comp_id), "unresolved", coords

            if target_type == source_type:
                return int(comp_id), "keep", coords
            if target_type == "arteriole":
                return int(comp_id), "to_arteriole", coords
            else:
                return int(comp_id), "to_venule", coords

        component_ids = sorted(int(k) for k in descriptors.keys())
        results: list[tuple[int, str, np.ndarray]] = []
        worker_count = int(reassignment_parallel_workers)
        if worker_count > 0 and len(component_ids) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_component_reassignment,
                        int(comp_id),
                        descriptors[int(comp_id)],
                    ): int(comp_id)
                    for comp_id in component_ids
                }
                for future in as_completed(future_map):
                    results.append(future.result())
        else:
            for comp_id in component_ids:
                results.append(
                    _evaluate_component_reassignment(
                        int(comp_id),
                        descriptors[int(comp_id)],
                    )
                )

        # Deterministic apply order for reproducible outputs.
        for _comp_id, decision, coords in sorted(results, key=lambda row: int(row[0])):
            if decision == "unresolved":
                unresolved += 1
                continue
            if decision == "to_arteriole":
                out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = False
                out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                reassigned_to_art += 1
            elif decision == "to_venule":
                out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = False
                out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                reassigned_to_ven += 1
    tangential_phase_elapsed_s = time.perf_counter() - tangential_phase_start_s

    sandwiched_stats = {
        "component_count": 0,
        "sandwiched_flips_to_arteriole": 0,
        "sandwiched_flips_to_venule": 0,
    }
    sandwiched_phase_start_s = time.perf_counter()
    if bool(enable_sandwiched_component_reassignment):
        sandwiched = _reassign_sandwiched_components(
            arteriole_mask=out_art,
            venule_mask=out_ven,
            voxel_size_xyz=voxel_size_xyz,
            max_endpoint_distance_microns=float(
                sandwiched_max_endpoint_distance_microns
            ),
            min_facing_cosine=float(sandwiched_min_facing_cosine),
            max_axis_angle_degrees=float(sandwiched_max_axis_angle_degrees),
            parallel_workers=int(reassignment_parallel_workers),
        )
        out_art = np.asarray(sandwiched["small_arteriole_mask"], dtype=bool)
        out_ven = np.asarray(sandwiched["small_venule_mask"], dtype=bool)
        sandwiched_stats = dict(sandwiched["stats"])
    sandwiched_phase_elapsed_s = time.perf_counter() - sandwiched_phase_start_s

    overlap = out_art & out_ven
    if np.any(overlap):
        out_ven = out_ven & (~overlap)

    return {
        "small_arteriole_mask": out_art.astype(bool, copy=False),
        "small_venule_mask": out_ven.astype(bool, copy=False),
        "stats": {
            "redefinition_enabled": True,
            "reassigned_to_arteriole": int(reassigned_to_art),
            "reassigned_to_venule": int(reassigned_to_ven),
            "unresolved_components": int(unresolved),
            "component_count": int(comp_count),
            "used_large_mask_tangency": bool(have_large_masks),
            "max_contact_distance_microns": float(max_contact_distance_microns),
            "touch_distance_microns": float(touch_distance_microns),
            "tangency_cosine_max": float(tangency_cosine_max),
            "reassignment_margin": float(reassignment_margin),
            "sandwiched_flips_to_arteriole": int(
                sandwiched_stats.get("sandwiched_flips_to_arteriole", 0)
            ),
            "sandwiched_flips_to_venule": int(
                sandwiched_stats.get("sandwiched_flips_to_venule", 0)
            ),
            "tangential_phase_elapsed_s": float(tangential_phase_elapsed_s),
            "sandwiched_phase_elapsed_s": float(sandwiched_phase_elapsed_s),
            "setup_phase_elapsed_s": float(setup_phase_elapsed_s),
            "total_elapsed_s": float(time.perf_counter() - total_start_s),
            "gpu_acceleration_requested": bool(use_gpu_acceleration),
            "gpu_acceleration_available": bool(_init_gpu_edt_backend())
            if bool(use_gpu_acceleration)
            else False,
        },
    }


def _axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    na = axis_a / max(1e-9, np.linalg.norm(axis_a))
    nb = axis_b / max(1e-9, np.linalg.norm(axis_b))
    cosine = float(np.clip(np.abs(np.dot(na, nb)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _bridge_mask_from_line(
    line_zyx: np.ndarray,
    shape: tuple[int, int, int],
    *,
    radius_voxels: int,
) -> np.ndarray:
    bridge = np.zeros(shape, dtype=bool)
    if line_zyx.size == 0:
        return bridge
    line_zyx = _clip_indices_to_shape(line_zyx, shape)
    if line_zyx.size == 0:
        return bridge
    bridge[line_zyx[:, 0], line_zyx[:, 1], line_zyx[:, 2]] = True
    if int(radius_voxels) > 0:
        bridge = binary_dilation(
            bridge,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(radius_voxels),
        )
    return bridge


def _min_endpoint_distance_microns(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    sampling_zyx: tuple[float, float, float],
) -> float:
    source_endpoints = source["endpoints_zyx"]
    target_endpoints = target["endpoints_zyx"]
    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    min_dist = np.inf
    for s_ep in source_endpoints:
        s = np.asarray(s_ep, dtype=float)
        for t_ep in target_endpoints:
            t = np.asarray(t_ep, dtype=float)
            d = float(np.linalg.norm((t - s).reshape(1, 3) * spacing))
            if d < min_dist:
                min_dist = d
    return float(min_dist)


def _attempt_cylinder_bridge(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    shape: tuple[int, int, int],
    dist_to_same_type_microns: np.ndarray,
    dist_to_opposite_type_microns: np.ndarray,
    sampling_zyx: tuple[float, float, float],
    max_bridge_distance_microns: float,
    corridor_max_distance_microns: float,
    opposite_exclusion_distance_microns: float,
    min_cylindricality: float,
    max_axis_angle_degrees: float,
    min_facing_cosine: float,
    max_radius_ratio: float,
    enforce_cylinder_only: bool,
    roi_bbox_zyx: tuple[slice, slice, slice] | None = None,
) -> tuple[bool, np.ndarray, str]:
    if enforce_cylinder_only:
        if float(source["linearity"]) < float(min_cylindricality):
            return False, np.zeros(shape, dtype=bool), "source_not_cylindrical"
        if float(target["linearity"]) < float(min_cylindricality):
            return False, np.zeros(shape, dtype=bool), "target_not_cylindrical"
        angle = _axis_angle_deg(
            np.asarray(source["principal_axis_zyx"], dtype=float),
            np.asarray(target["principal_axis_zyx"], dtype=float),
        )
        if angle > float(max_axis_angle_degrees):
            return False, np.zeros(shape, dtype=bool), "axis_mismatch"
        src_r = max(1e-6, float(source["median_radius_microns"]))
        tgt_r = max(1e-6, float(target["median_radius_microns"]))
        ratio = max(src_r, tgt_r) / min(src_r, tgt_r)
        if ratio > float(max_radius_ratio):
            return False, np.zeros(shape, dtype=bool), "radius_ratio_mismatch"

    source_endpoints = source["endpoints_zyx"]
    target_endpoints = target["endpoints_zyx"]
    pairs = [
        (np.asarray(source_endpoints[0], dtype=int), np.asarray(target_endpoints[0], dtype=int)),
        (np.asarray(source_endpoints[0], dtype=int), np.asarray(target_endpoints[1], dtype=int)),
        (np.asarray(source_endpoints[1], dtype=int), np.asarray(target_endpoints[0], dtype=int)),
        (np.asarray(source_endpoints[1], dtype=int), np.asarray(target_endpoints[1], dtype=int)),
    ]
    best_pair = None
    best_distance = np.inf
    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    for p0, p1 in pairs:
        dist_microns = float(np.linalg.norm((p1 - p0).astype(float) * spacing))
        if dist_microns < best_distance:
            best_distance = dist_microns
            best_pair = (p0, p1)
    if best_pair is None:
        return False, np.zeros(shape, dtype=bool), "no_endpoint_pair"
    p0_best, p1_best = best_pair
    if best_distance > float(max_bridge_distance_microns):
        return False, np.zeros(shape, dtype=bool), "bridge_too_long"

    if enforce_cylinder_only:
        v = (p1_best - p0_best).astype(float)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-9:
            return False, np.zeros(shape, dtype=bool), "degenerate_endpoint_vector"
        v_hat = v / v_norm
        source_axis = np.asarray(source["principal_axis_zyx"], dtype=float)
        target_axis = np.asarray(target["principal_axis_zyx"], dtype=float)
        # Orient source axis to point toward target, and target axis to point away from target.
        if float(np.dot(source_axis, v_hat)) < 0.0:
            source_axis = -source_axis
        if float(np.dot(target_axis, v_hat)) > 0.0:
            target_axis = -target_axis
        source_facing = float(np.dot(source_axis, v_hat))
        target_facing = float(np.dot(target_axis, -v_hat))
        facing_threshold = float(min_facing_cosine)
        if source_facing < facing_threshold or target_facing < facing_threshold:
            return False, np.zeros(shape, dtype=bool), "endpoint_facing_mismatch"

    line_zyx = _line_indices_zyx(p0_best, p1_best)
    line_zyx = _clip_indices_to_shape(line_zyx, shape)
    if line_zyx.size == 0:
        return False, np.zeros(shape, dtype=bool), "empty_line"

    line_local_zyx = _global_to_local_indices_zyx(line_zyx, roi_bbox_zyx)
    same_d = dist_to_same_type_microns[
        line_local_zyx[:, 0],
        line_local_zyx[:, 1],
        line_local_zyx[:, 2],
    ]
    opposite_d = dist_to_opposite_type_microns[
        line_local_zyx[:, 0],
        line_local_zyx[:, 1],
        line_local_zyx[:, 2],
    ]
    if np.any(opposite_d < float(opposite_exclusion_distance_microns)):
        return False, np.zeros(shape, dtype=bool), "cross_type_exclusion"
    if np.any(same_d > float(corridor_max_distance_microns)):
        return False, np.zeros(shape, dtype=bool), "outside_same_type_corridor"

    voxel_scale = min(float(v) for v in sampling_zyx)
    avg_radius_microns = 0.5 * (
        float(source["median_radius_microns"]) + float(target["median_radius_microns"])
    )
    bridge_radius_voxels = int(max(0, np.rint(avg_radius_microns / max(1e-9, voxel_scale))))
    bridge_radius_voxels = int(min(bridge_radius_voxels, 3))
    bridge = _bridge_mask_from_line(
        line_zyx,
        shape,
        radius_voxels=bridge_radius_voxels,
    )
    return True, bridge, "bridged"


def _enforce_type_locked_continuity_for_small_mask(
    *,
    small_mask: np.ndarray,
    large_mask: np.ndarray | None,
    opposite_mask: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    allow_small_to_large: bool,
    allow_small_to_small: bool,
    enforce_cylinder_only: bool,
    min_cylindricality: float,
    max_axis_angle_degrees: float,
    min_facing_cosine: float,
    max_radius_ratio: float,
    max_bridge_distance_microns: float,
    corridor_max_distance_microns: float,
    opposite_exclusion_distance_microns: float,
    use_gpu_acceleration: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_start_s = time.perf_counter()
    small_binary = small_mask.astype(bool, copy=False)
    large_binary = (
        np.zeros_like(small_binary, dtype=bool)
        if large_mask is None
        else large_mask.astype(bool, copy=False)
    )
    opposite_binary = opposite_mask.astype(bool, copy=False)
    sampling_zyx = _voxel_size_xyz_to_sampling_zyx(voxel_size_xyz)
    edt_inside_small = _edt(
        small_binary,
        sampling_zyx=sampling_zyx,
        use_gpu_acceleration=bool(use_gpu_acceleration),
    )
    small_desc = _component_descriptors(small_binary, edt_inside_small)
    if not small_desc:
        return small_binary.copy(), {
            "attempted_bridges": 0,
            "accepted_bridges": 0,
            "rejected_reasons": {},
        }
    labeled_large, _ = _connected_components(large_binary)
    large_desc: dict[int, dict[str, Any]] = {}
    if allow_small_to_large and np.any(large_binary):
        edt_inside_large = _edt(
            large_binary,
            sampling_zyx=sampling_zyx,
            use_gpu_acceleration=bool(use_gpu_acceleration),
        )
        large_desc = _component_descriptors(large_binary, edt_inside_large)

    updated_small = small_binary.copy()
    source_candidates: dict[int, list[dict[str, Any]]] = {}
    centroid_prefilter_radius = float(max_bridge_distance_microns) + float(
        corridor_max_distance_microns
    )
    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    large_tree: cKDTree | None = None
    large_ids: list[int] = []
    if large_desc:
        large_ids = sorted(large_desc.keys())
        large_centroids = np.asarray(
            [np.asarray(large_desc[cid]["centroid_zyx"], dtype=float) for cid in large_ids],
            dtype=float,
        )
        large_tree = cKDTree(large_centroids * spacing)
    small_tree: cKDTree | None = None
    small_ids = sorted(small_desc.keys())
    if small_ids:
        small_centroids = np.asarray(
            [np.asarray(small_desc[cid]["centroid_zyx"], dtype=float) for cid in small_ids],
            dtype=float,
        )
        small_tree = cKDTree(small_centroids * spacing)
    source_build_start_s = time.perf_counter()
    source_ids = sorted(small_desc.keys())
    for source_id in source_ids:
        source = small_desc[source_id]
        source_centroid = np.asarray(source["centroid_zyx"], dtype=float)
        source_point = source_centroid.reshape(1, 3) * spacing
        candidate_targets: list[dict[str, Any]] = []
        if allow_small_to_large and large_tree is not None:
            idxs = large_tree.query_ball_point(source_point.ravel(), r=centroid_prefilter_radius)
            for idx in idxs:
                cid = int(large_ids[int(idx)])
                candidate_targets.append(large_desc[cid])
        if allow_small_to_small and small_tree is not None:
            idxs = small_tree.query_ball_point(source_point.ravel(), r=centroid_prefilter_radius)
            for idx in idxs:
                cid = int(small_ids[int(idx)])
                if cid == int(source_id):
                    continue
                candidate = small_desc[cid]
                if int(candidate["size_voxels"]) >= int(source["size_voxels"]):
                    candidate_targets.append(candidate)
        if not candidate_targets:
            continue

        # Guard against duplicate targets when queried from multiple pools.
        uniq: dict[int, dict[str, Any]] = {}
        for candidate in candidate_targets:
            uniq[int(candidate["component_id"])] = candidate
        candidate_targets = list(uniq.values())

        # Keep an explicit radius gate for safety and deterministic behavior.
        candidate_targets = [
            c
            for c in candidate_targets
            if float(
                np.linalg.norm(
                    (np.asarray(c["centroid_zyx"], dtype=float) - source_centroid)
                    * np.asarray(sampling_zyx, dtype=float)
                )
            )
            <= centroid_prefilter_radius
        ]
        if not candidate_targets:
            continue

        candidate_targets = sorted(
            candidate_targets,
            key=lambda c: float(
                np.linalg.norm(np.asarray(c["centroid_zyx"], dtype=float) - source_centroid)
            ),
        )

        # Skip if source small component already touches any large component.
        if allow_small_to_large and np.any(large_binary):
            src_vox = source["coords_zyx"]
            if np.any(
                labeled_large[src_vox[:, 0], src_vox[:, 1], src_vox[:, 2]] > 0
            ):
                continue
        source_candidates[int(source_id)] = candidate_targets

    candidate_build_elapsed_s = time.perf_counter() - source_build_start_s
    if not source_candidates:
        return updated_small, {
            "attempted_bridges": 0,
            "accepted_bridges": 0,
            "rejected_reasons": {},
            "prefiltered_out_count": 0,
            "candidate_build_elapsed_s": float(candidate_build_elapsed_s),
            "distance_setup_elapsed_s": 0.0,
            "bridge_loop_elapsed_s": 0.0,
            "total_elapsed_s": float(time.perf_counter() - total_start_s),
            "roi_shape_zyx": (0, 0, 0),
        }

    # Compute expensive distance volumes only on candidate ROI.
    distance_setup_start_s = time.perf_counter()
    candidate_points: list[np.ndarray] = []
    for source_id in source_ids:
        if int(source_id) not in source_candidates:
            continue
        src = small_desc[int(source_id)]
        src_a, src_b = src["endpoints_zyx"]
        candidate_points.append(np.asarray(src_a, dtype=int))
        candidate_points.append(np.asarray(src_b, dtype=int))
        for tgt in source_candidates[int(source_id)]:
            tgt_a, tgt_b = tgt["endpoints_zyx"]
            candidate_points.append(np.asarray(tgt_a, dtype=int))
            candidate_points.append(np.asarray(tgt_b, dtype=int))
    points_arr = (
        np.asarray(candidate_points, dtype=int)
        if candidate_points
        else np.empty((0, 3), dtype=int)
    )
    max_margin_microns = max(
        float(max_bridge_distance_microns),
        float(corridor_max_distance_microns),
        float(opposite_exclusion_distance_microns),
    )
    margin_zyx = (
        int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[0])))),
        int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[1])))),
        int(np.ceil(max_margin_microns / max(1e-9, float(sampling_zyx[2])))),
    )
    roi_bbox_zyx = _bbox_from_masks_and_points(
        shape_zyx=small_binary.shape,
        masks=[small_binary, large_binary],
        points_zyx=points_arr,
        margin_vox_zyx=margin_zyx,
    )
    z_slice, y_slice, x_slice = roi_bbox_zyx
    roi_shape_zyx = (
        int(z_slice.stop) - int(z_slice.start),
        int(y_slice.stop) - int(y_slice.start),
        int(x_slice.stop) - int(x_slice.start),
    )
    dist_to_same_type = _edt(
        ~(small_binary | large_binary)[z_slice, y_slice, x_slice],
        sampling_zyx=sampling_zyx,
        use_gpu_acceleration=bool(use_gpu_acceleration),
    )
    dist_to_opposite_type = _edt(
        ~opposite_binary[z_slice, y_slice, x_slice],
        sampling_zyx=sampling_zyx,
        use_gpu_acceleration=bool(use_gpu_acceleration),
    )
    distance_setup_elapsed_s = time.perf_counter() - distance_setup_start_s
    attempted = 0
    accepted = 0
    rejected_reasons: dict[str, int] = {}
    prefiltered_out_count = 0
    bridge_loop_start_s = time.perf_counter()
    for source_id in source_ids:
        if int(source_id) not in source_candidates:
            continue
        source = small_desc[source_id]
        for target in source_candidates[int(source_id)]:
            # Necessary-condition prefilters to avoid expensive line sampling.
            if enforce_cylinder_only:
                source_axis = np.asarray(source["principal_axis_zyx"], dtype=float)
                target_axis = np.asarray(target["principal_axis_zyx"], dtype=float)
                axis_angle = _axis_angle_deg(source_axis, target_axis)
                if axis_angle > float(max_axis_angle_degrees):
                    prefiltered_out_count += 1
                    continue
                src_r = max(1e-6, float(source["median_radius_microns"]))
                tgt_r = max(1e-6, float(target["median_radius_microns"]))
                ratio = max(src_r, tgt_r) / min(src_r, tgt_r)
                if ratio > float(max_radius_ratio):
                    prefiltered_out_count += 1
                    continue
            endpoint_dist = _min_endpoint_distance_microns(
                source,
                target,
                sampling_zyx=sampling_zyx,
            )
            if endpoint_dist > float(max_bridge_distance_microns):
                prefiltered_out_count += 1
                continue

            attempted += 1
            ok, bridge_mask, reason = _attempt_cylinder_bridge(
                source,
                target,
                shape=small_binary.shape,
                dist_to_same_type_microns=dist_to_same_type,
                dist_to_opposite_type_microns=dist_to_opposite_type,
                sampling_zyx=sampling_zyx,
                max_bridge_distance_microns=float(max_bridge_distance_microns),
                corridor_max_distance_microns=float(corridor_max_distance_microns),
                opposite_exclusion_distance_microns=float(opposite_exclusion_distance_microns),
                min_cylindricality=float(min_cylindricality),
                max_axis_angle_degrees=float(max_axis_angle_degrees),
                min_facing_cosine=float(min_facing_cosine),
                max_radius_ratio=float(max_radius_ratio),
                enforce_cylinder_only=bool(enforce_cylinder_only),
                roi_bbox_zyx=roi_bbox_zyx,
            )
            if not ok:
                rejected_reasons[reason] = int(rejected_reasons.get(reason, 0) + 1)
                continue
            updated_small |= bridge_mask
            accepted += 1
            break
    bridge_loop_elapsed_s = time.perf_counter() - bridge_loop_start_s

    return updated_small, {
        "attempted_bridges": int(attempted),
        "accepted_bridges": int(accepted),
        "rejected_reasons": rejected_reasons,
        "prefiltered_out_count": int(prefiltered_out_count),
        "candidate_build_elapsed_s": float(candidate_build_elapsed_s),
        "distance_setup_elapsed_s": float(distance_setup_elapsed_s),
        "bridge_loop_elapsed_s": float(bridge_loop_elapsed_s),
        "total_elapsed_s": float(time.perf_counter() - total_start_s),
        "roi_shape_zyx": roi_shape_zyx,
    }


def enforce_small_vessel_mask_continuity(
    *,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    voxel_size_xyz: tuple[float, float, float],
    enable_continuity: bool = True,
    allow_small_to_large: bool = True,
    allow_small_to_small: bool = True,
    enforce_cylinder_only: bool = True,
    min_cylindricality: float = 0.45,
    max_axis_angle_degrees: float = 45.0,
    min_facing_cosine: float = 0.82,
    max_radius_ratio: float = 3.0,
    max_bridge_distance_microns: float = 35.0,
    corridor_max_distance_microns: float = 12.0,
    opposite_exclusion_distance_microns: float = 3.0,
    use_gpu_acceleration: bool = False,
) -> dict[str, Any]:
    """Bridge same-type small-vessel mask gaps with type-locked cylinder links.

    Allowed connections are strictly:
    - small venule -> large venule
    - small venule -> small venule
    - small arteriole -> large arteriole
    - small arteriole -> small arteriole
    """
    art_small = small_arteriole_mask.astype(bool, copy=False)
    ven_small = small_venule_mask.astype(bool, copy=False)
    if art_small.shape != ven_small.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {art_small.shape} and {ven_small.shape}."
        )
    if large_arteriole_mask is not None and large_arteriole_mask.shape != art_small.shape:
        raise ValueError(
            "large_arteriole_mask shape must match small masks. "
            f"Got {large_arteriole_mask.shape} and {art_small.shape}."
        )
    if large_venule_mask is not None and large_venule_mask.shape != ven_small.shape:
        raise ValueError(
            "large_venule_mask shape must match small masks. "
            f"Got {large_venule_mask.shape} and {ven_small.shape}."
        )
    if not enable_continuity:
        return {
            "small_arteriole_mask": art_small.copy(),
            "small_venule_mask": ven_small.copy(),
            "stats": {
                "continuity_enabled": False,
                "arteriole": {"attempted_bridges": 0, "accepted_bridges": 0, "rejected_reasons": {}},
                "venule": {"attempted_bridges": 0, "accepted_bridges": 0, "rejected_reasons": {}},
            },
        }

    updated_ven, ven_stats = _enforce_type_locked_continuity_for_small_mask(
        small_mask=ven_small,
        large_mask=large_venule_mask,
        opposite_mask=art_small | (
            np.zeros_like(art_small, dtype=bool)
            if large_arteriole_mask is None
            else large_arteriole_mask.astype(bool, copy=False)
        ),
        voxel_size_xyz=voxel_size_xyz,
        allow_small_to_large=allow_small_to_large,
        allow_small_to_small=allow_small_to_small,
        enforce_cylinder_only=enforce_cylinder_only,
        min_cylindricality=min_cylindricality,
        max_axis_angle_degrees=max_axis_angle_degrees,
        min_facing_cosine=min_facing_cosine,
        max_radius_ratio=max_radius_ratio,
        max_bridge_distance_microns=max_bridge_distance_microns,
        corridor_max_distance_microns=corridor_max_distance_microns,
        opposite_exclusion_distance_microns=opposite_exclusion_distance_microns,
        use_gpu_acceleration=bool(use_gpu_acceleration),
    )
    updated_art, art_stats = _enforce_type_locked_continuity_for_small_mask(
        small_mask=art_small,
        large_mask=large_arteriole_mask,
        opposite_mask=updated_ven | (
            np.zeros_like(ven_small, dtype=bool)
            if large_venule_mask is None
            else large_venule_mask.astype(bool, copy=False)
        ),
        voxel_size_xyz=voxel_size_xyz,
        allow_small_to_large=allow_small_to_large,
        allow_small_to_small=allow_small_to_small,
        enforce_cylinder_only=enforce_cylinder_only,
        min_cylindricality=min_cylindricality,
        max_axis_angle_degrees=max_axis_angle_degrees,
        min_facing_cosine=min_facing_cosine,
        max_radius_ratio=max_radius_ratio,
        max_bridge_distance_microns=max_bridge_distance_microns,
        corridor_max_distance_microns=corridor_max_distance_microns,
        opposite_exclusion_distance_microns=opposite_exclusion_distance_microns,
        use_gpu_acceleration=bool(use_gpu_acceleration),
    )
    # Keep strict type separation after bridging.
    overlap = updated_art & updated_ven
    if np.any(overlap):
        updated_ven = updated_ven & (~overlap)

    return {
        "small_arteriole_mask": updated_art.astype(bool, copy=False),
        "small_venule_mask": updated_ven.astype(bool, copy=False),
        "stats": {
            "continuity_enabled": True,
            "arteriole": art_stats,
            "venule": ven_stats,
        },
    }
