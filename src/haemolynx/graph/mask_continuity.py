"""Type-locked cylinder continuity helpers for small-vessel masks."""
from __future__ import annotations

from typing import Any
import time

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


def _voxel_size_zyx_to_sampling_zyx(
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return EDT sampling in array axis order (z, y, x)."""
    return (
        float(voxel_size_zyx[0]),
        float(voxel_size_zyx[1]),
        float(voxel_size_zyx[2]),
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
    voxel_size_zyx: tuple[float, float, float],
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
    sampling_zyx = _voxel_size_zyx_to_sampling_zyx(voxel_size_zyx)
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
            # Cheap endpoint-distance prefilter; axis/facing/radius checks run in
            # `_attempt_cylinder_bridge` so rejection reasons match actual gates.
            endpoint_dist = _min_endpoint_distance_microns(
                source,
                target,
                sampling_zyx=sampling_zyx,
            )
            if endpoint_dist > float(max_bridge_distance_microns):
                prefiltered_out_count += 1
                rejected_reasons["bridge_too_long"] = int(
                    rejected_reasons.get("bridge_too_long", 0) + 1
                )
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
    voxel_size_zyx: tuple[float, float, float],
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
        voxel_size_zyx=voxel_size_zyx,
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
        voxel_size_zyx=voxel_size_zyx,
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
