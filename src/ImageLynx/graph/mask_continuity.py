"""Type-locked cylinder continuity helpers for small-vessel masks."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


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


def _component_descriptors(mask: np.ndarray, edt_inside: np.ndarray) -> dict[int, dict[str, Any]]:
    labeled, count = _connected_components(mask)
    descriptors: dict[int, dict[str, Any]] = {}
    if count <= 0:
        return descriptors
    for component_id in range(1, count + 1):
        coords = np.argwhere(labeled == component_id)
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
        radii = edt_inside[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ]
        median_radius = float(np.median(radii)) if radii.size else 0.0
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


def _signed_distance_and_gradient(
    mask: np.ndarray,
    *,
    sampling_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    binary = mask.astype(bool, copy=False)
    outside_dist = distance_transform_edt(~binary, sampling=sampling_zyx)
    inside_dist = distance_transform_edt(binary, sampling=sampling_zyx)
    signed = outside_dist - inside_dist
    grad = np.gradient(
        signed,
        float(sampling_zyx[0]),
        float(sampling_zyx[1]),
        float(sampling_zyx[2]),
    )
    return signed, (np.asarray(grad[0]), np.asarray(grad[1]), np.asarray(grad[2]))


def _evaluate_endpoint_tangency_to_large_mask(
    endpoint_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    *,
    signed_distance_to_large: np.ndarray,
    grad_signed_distance_to_large: tuple[np.ndarray, np.ndarray, np.ndarray],
    distance_to_opposite_large: np.ndarray,
    max_contact_distance_microns: float,
    tangency_cosine_max: float,
    opposite_exclusion_distance_microns: float,
    tangency_weight: float,
    opposite_penalty_weight: float,
) -> dict[str, Any]:
    idx = np.asarray(endpoint_zyx, dtype=int)
    z, y, x = int(idx[0]), int(idx[1]), int(idx[2])
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
    gz = float(grad_signed_distance_to_large[0][z, y, x])
    gy = float(grad_signed_distance_to_large[1][z, y, x])
    gx = float(grad_signed_distance_to_large[2][z, y, x])
    normal = np.asarray([gz, gy, gx], dtype=float)
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
    component_type: dict[int, str],
    max_endpoint_distance_microns: float,
    min_facing_cosine: float,
    max_axis_angle_degrees: float,
    sampling_zyx: tuple[float, float, float],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    e = np.asarray(endpoint_zyx, dtype=float)
    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    current_axis = np.asarray(current_axis_zyx, dtype=float)
    for other_id, other in descriptors.items():
        if int(other_id) == int(current_component_id):
            continue
        if component_type.get(int(other_id), "") != str(target_type):
            continue
        other_axis = np.asarray(other["principal_axis_zyx"], dtype=float)
        axis_angle = _axis_angle_deg(current_axis, other_axis)
        if axis_angle > float(max_axis_angle_degrees):
            continue
        other_endpoints = other["endpoints_zyx"]
        p0 = np.asarray(other_endpoints[0], dtype=float)
        p1 = np.asarray(other_endpoints[1], dtype=float)
        d0 = float(np.linalg.norm((p0 - e).reshape(1, 3) * spacing))
        d1 = float(np.linalg.norm((p1 - e).reshape(1, 3) * spacing))
        other_endpoint = p0 if d0 <= d1 else p1
        endpoint_distance = min(d0, d1)
        if endpoint_distance > float(max_endpoint_distance_microns):
            continue
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
) -> dict[str, Any]:
    all_small = arteriole_mask.astype(bool, copy=False) | venule_mask.astype(bool, copy=False)
    sampling_zyx = _voxel_size_xyz_to_sampling_zyx(voxel_size_xyz)
    edt_inside_all = distance_transform_edt(all_small, sampling=sampling_zyx)
    labeled, comp_count = _connected_components(all_small)
    descriptors = _component_descriptors(all_small, edt_inside_all)
    component_type: dict[int, str] = {}
    for comp_id, desc in descriptors.items():
        coords = desc["coords_zyx"]
        art_frac = float(
            np.count_nonzero(arteriole_mask[coords[:, 0], coords[:, 1], coords[:, 2]])
        ) / float(max(1, coords.shape[0]))
        component_type[int(comp_id)] = "arteriole" if art_frac >= 0.5 else "venule"

    flips: dict[int, str] = {}
    for comp_id, desc in descriptors.items():
        current_type = component_type.get(int(comp_id), "")
        if current_type not in {"arteriole", "venule"}:
            continue
        target_type = "venule" if current_type == "arteriole" else "arteriole"
        axis = np.asarray(desc["principal_axis_zyx"], dtype=float)
        end_a, end_b = desc["endpoints_zyx"]
        support_a = _best_neighbor_for_endpoint(
            current_component_id=int(comp_id),
            endpoint_zyx=np.asarray(end_a, dtype=int),
            current_axis_zyx=axis,
            target_type=target_type,
            descriptors=descriptors,
            component_type=component_type,
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
            component_type=component_type,
            max_endpoint_distance_microns=max_endpoint_distance_microns,
            min_facing_cosine=min_facing_cosine,
            max_axis_angle_degrees=max_axis_angle_degrees,
            sampling_zyx=sampling_zyx,
        )
        if support_a is None or support_b is None:
            continue
        # Prefer true "middle between two labeled volumes": different neighbors at each end.
        if int(support_a["component_id"]) == int(support_b["component_id"]):
            continue
        flips[int(comp_id)] = target_type

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
    if have_large_masks:
        large_art = large_arteriole_mask.astype(bool, copy=False)
        large_ven = large_venule_mask.astype(bool, copy=False)
        if large_art.shape != art_small.shape or large_ven.shape != art_small.shape:
            raise ValueError(
                "All small/large masks must share the same shape for redefinition."
            )
        signed_art, grad_art = _signed_distance_and_gradient(
            large_art,
            sampling_zyx=sampling_zyx,
        )
        signed_ven, grad_ven = _signed_distance_and_gradient(
            large_ven,
            sampling_zyx=sampling_zyx,
        )
        dist_to_art = distance_transform_edt(~large_art, sampling=sampling_zyx)
        dist_to_ven = distance_transform_edt(~large_ven, sampling=sampling_zyx)
    else:
        signed_art = np.zeros_like(art_small, dtype=float)
        signed_ven = np.zeros_like(ven_small, dtype=float)
        zeros_grad = np.zeros_like(art_small, dtype=float)
        grad_art = (zeros_grad, zeros_grad, zeros_grad)
        grad_ven = (zeros_grad, zeros_grad, zeros_grad)
        dist_to_art = np.full_like(art_small, fill_value=np.inf, dtype=float)
        dist_to_ven = np.full_like(ven_small, fill_value=np.inf, dtype=float)

    out_art = art_small.copy()
    out_ven = ven_small.copy()

    all_small = art_small | ven_small
    edt_inside_all_small = distance_transform_edt(all_small, sampling=sampling_zyx)
    _labeled_all, comp_count = _connected_components(all_small)
    descriptors = _component_descriptors(all_small, edt_inside_all_small)

    reassigned_to_art = 0
    reassigned_to_ven = 0
    unresolved = 0
    if have_large_masks:
        for comp_id, desc in descriptors.items():
            coords = desc["coords_zyx"]
            if coords.size == 0:
                continue
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
                art_eval = _evaluate_endpoint_tangency_to_large_mask(
                    np.asarray(endpoint, dtype=int),
                    axis,
                    signed_distance_to_large=signed_art,
                    grad_signed_distance_to_large=grad_art,
                    distance_to_opposite_large=dist_to_ven,
                    max_contact_distance_microns=float(max_contact_distance_microns),
                    tangency_cosine_max=float(tangency_cosine_max),
                    opposite_exclusion_distance_microns=float(
                        opposite_exclusion_distance_microns
                    ),
                    tangency_weight=float(tangency_weight),
                    opposite_penalty_weight=float(opposite_penalty_weight),
                )
                if art_eval["score"] < best_art["score"]:
                    best_art = art_eval
                ven_eval = _evaluate_endpoint_tangency_to_large_mask(
                    np.asarray(endpoint, dtype=int),
                    axis,
                    signed_distance_to_large=signed_ven,
                    grad_signed_distance_to_large=grad_ven,
                    distance_to_opposite_large=dist_to_art,
                    max_contact_distance_microns=float(max_contact_distance_microns),
                    tangency_cosine_max=float(tangency_cosine_max),
                    opposite_exclusion_distance_microns=float(
                        opposite_exclusion_distance_microns
                    ),
                    tangency_weight=float(tangency_weight),
                    opposite_penalty_weight=float(opposite_penalty_weight),
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
                    unresolved += 1
                    continue
            else:
                unresolved += 1
                continue

            if target_type == source_type:
                continue

            if target_type == "arteriole":
                out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = False
                out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                reassigned_to_art += 1
            else:
                out_art[coords[:, 0], coords[:, 1], coords[:, 2]] = False
                out_ven[coords[:, 0], coords[:, 1], coords[:, 2]] = True
                reassigned_to_ven += 1

    sandwiched_stats = {
        "component_count": 0,
        "sandwiched_flips_to_arteriole": 0,
        "sandwiched_flips_to_venule": 0,
    }
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
        )
        out_art = np.asarray(sandwiched["small_arteriole_mask"], dtype=bool)
        out_ven = np.asarray(sandwiched["small_venule_mask"], dtype=bool)
        sandwiched_stats = dict(sandwiched["stats"])

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

    same_d = dist_to_same_type_microns[
        line_zyx[:, 0],
        line_zyx[:, 1],
        line_zyx[:, 2],
    ]
    opposite_d = dist_to_opposite_type_microns[
        line_zyx[:, 0],
        line_zyx[:, 1],
        line_zyx[:, 2],
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
) -> tuple[np.ndarray, dict[str, Any]]:
    small_binary = small_mask.astype(bool, copy=False)
    large_binary = (
        np.zeros_like(small_binary, dtype=bool)
        if large_mask is None
        else large_mask.astype(bool, copy=False)
    )
    opposite_binary = opposite_mask.astype(bool, copy=False)
    sampling_zyx = _voxel_size_xyz_to_sampling_zyx(voxel_size_xyz)
    edt_inside_small = distance_transform_edt(small_binary, sampling=sampling_zyx)
    small_desc = _component_descriptors(small_binary, edt_inside_small)
    if not small_desc:
        return small_binary.copy(), {
            "attempted_bridges": 0,
            "accepted_bridges": 0,
            "rejected_reasons": {},
        }
    labeled_large, _ = _connected_components(large_binary)
    dist_to_same_type = distance_transform_edt(~(small_binary | large_binary), sampling=sampling_zyx)
    dist_to_opposite_type = distance_transform_edt(~opposite_binary, sampling=sampling_zyx)
    edt_inside_large = distance_transform_edt(large_binary, sampling=sampling_zyx)
    large_desc = _component_descriptors(large_binary, edt_inside_large) if np.any(large_binary) else {}

    updated_small = small_binary.copy()
    attempted = 0
    accepted = 0
    rejected_reasons: dict[str, int] = {}

    source_ids = sorted(small_desc.keys())
    for source_id in source_ids:
        source = small_desc[source_id]
        source_centroid = np.asarray(source["centroid_zyx"], dtype=float)

        candidate_targets: list[dict[str, Any]] = []
        if allow_small_to_large and large_desc:
            candidate_targets.extend(large_desc.values())
        if allow_small_to_small:
            for candidate_id, candidate in small_desc.items():
                if int(candidate_id) == int(source_id):
                    continue
                if int(candidate["size_voxels"]) >= int(source["size_voxels"]):
                    candidate_targets.append(candidate)
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

        for target in candidate_targets:
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
            )
            if not ok:
                rejected_reasons[reason] = int(rejected_reasons.get(reason, 0) + 1)
                continue
            updated_small |= bridge_mask
            accepted += 1
            break

    return updated_small, {
        "attempted_bridges": int(attempted),
        "accepted_bridges": int(accepted),
        "rejected_reasons": rejected_reasons,
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
