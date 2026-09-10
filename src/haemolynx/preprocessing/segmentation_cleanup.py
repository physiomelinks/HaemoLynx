"""Pre-skeletonization cleanup of the raw segmented binary vessel mask.

Three independently-toggleable steps, always applied reconnect -> smooth ->
remove-small when more than one is on (see
:func:`clean_segmented_mask_for_skeletonisation` for why that order).  All
distance/volume parameters are physical (microns / cubic microns), sampled
via ``voxel_size_zyx`` -- unlike :func:`haemolynx.preprocessing.skeleton.
close_binary_mask`/``bridge_gaps``, which are voxel-unit only, a real
problem for this project's anisotropic datasets (e.g. a ``(1.0, 0.4, 0.4)``
zyx voxel size, where a "radius=2 voxel" ball is 2.0 microns in z but only
0.8 microns in x/y).

Nothing here imports :mod:`haemolynx.io` or :mod:`haemolynx.graph` --
``preprocessing`` sits at the bottom of this package's dependency stack (both
of those already import *from* ``preprocessing``), so every function here
takes and returns an already-boolean array; canonical binarisation of
whatever a loader produced stays the caller's job
(``io.load._to_binary_volume_for_skeletonization`` is the one place that
decides which label value is foreground).

The reconnect step's component-descriptor and cylinder-bridge machinery is a
single-mask relative of :mod:`haemolynx.graph.mask_continuity`'s type-locked
small-vessel continuity (``enforce_small_vessel_mask_continuity``), which
solves a different, narrower problem (arteriole/venule redefinition against
large masks, post-boundary-assignment) with the same PCA/endpoint/cylinder
gating idea. Its helpers are module-private and live one layer above this
one, so the pieces this module needs are re-implemented here rather than
imported.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects

__all__ = [
    "reconnect_vessel_like_components",
    "smooth_vessel_surfaces",
    "remove_small_segmented_volumes",
    "clean_segmented_mask_for_skeletonisation",
]

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, count = label(mask.astype(bool, copy=False), structure=_STRUCTURE_26)
    return labeled, int(count)


def _component_descriptors(
    *, labeled: np.ndarray, count: int, edt_inside: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Per labeled component: centroid, PCA principal axis, a "linearity"
    (cylindricality) score, the two endpoints along that axis, and a median
    local radius in physical microns.

    Linearity is ``(eigval[0] - eigval[1]) / eigval[0]`` on the covariance of
    the component's own voxel coordinates -- near 1 for an elongated,
    tube-like component, near 0 for a blob. PCA runs on raw voxel indices,
    not physical coordinates (matching ``graph.mask_continuity``'s own
    precedent): a genuinely anisotropic dataset biases this somewhat toward
    the more finely sampled axes, a known, accepted limitation of the same
    approach already used in production elsewhere in this codebase.

    *edt_inside* is a Euclidean distance transform of the *whole* mask,
    computed once by the caller with physical ``sampling=voxel_size_zyx``
    (so it is correct for an anisotropic dataset) -- the median of its
    values over just this component's own voxels is a real, local radius
    estimate, not a crude area/length approximation.
    """
    from scipy.ndimage import find_objects

    descriptors: dict[int, dict[str, Any]] = {}
    if count <= 0:
        return descriptors
    slices = find_objects(labeled, max_label=count)
    for component_id in range(1, count + 1):
        component_slice = slices[component_id - 1] if slices else None
        if component_slice is None:
            continue
        local = labeled[component_slice]
        local_coords = np.argwhere(local == component_id)
        if local_coords.size == 0:
            continue
        offset = np.asarray(
            [int(s.start) for s in component_slice], dtype=int
        ).reshape(1, 3)
        coords = local_coords + offset
        centroid = np.mean(coords.astype(float), axis=0)
        if coords.shape[0] >= 3:
            cov = np.cov(coords.astype(float).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            principal_axis = eigvecs[:, order[0]]
            linearity = float(
                (eigvals[0] - eigvals[1]) / max(1e-9, float(eigvals[0]))
            )
        else:
            principal_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
            linearity = 0.0
        norm = float(np.linalg.norm(principal_axis))
        principal_axis = principal_axis / norm if norm > 1e-9 else principal_axis
        projections = coords.astype(float) @ principal_axis.reshape(3, 1)
        end_a = coords[int(np.argmin(projections[:, 0]))]
        end_b = coords[int(np.argmax(projections[:, 0]))]
        radii = edt_inside[coords[:, 0], coords[:, 1], coords[:, 2]]
        median_radius_um = float(np.median(radii)) if radii.size else 0.0
        descriptors[component_id] = {
            "component_id": int(component_id),
            "coords": coords,
            "centroid": centroid,
            "principal_axis": principal_axis,
            "linearity": linearity,
            "endpoints": (end_a.astype(int), end_b.astype(int)),
            "median_radius_um": float(median_radius_um),
            "size_voxels": int(coords.shape[0]),
        }
    return descriptors


def _axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    na = axis_a / max(1e-9, float(np.linalg.norm(axis_a)))
    nb = axis_b / max(1e-9, float(np.linalg.norm(axis_b)))
    cosine = float(np.clip(abs(float(np.dot(na, nb))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _line_indices(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end.astype(float) - start.astype(float)
    steps = int(max(2, np.ceil(float(np.linalg.norm(delta))) + 1))
    t = np.linspace(0.0, 1.0, steps, dtype=float).reshape(-1, 1)
    points = np.rint(start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t)
    return np.unique(points.astype(int), axis=0)


def _bridge_mask_from_line(
    line: np.ndarray, shape: tuple[int, int, int], *, radius_voxels: int
) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    bridge = np.zeros(shape, dtype=bool)
    if line.size == 0:
        return bridge
    valid = np.all(line >= 0, axis=1) & np.all(
        line < np.asarray(shape).reshape(1, 3), axis=1
    )
    line = line[valid]
    if line.size == 0:
        return bridge
    bridge[line[:, 0], line[:, 1], line[:, 2]] = True
    if radius_voxels > 0:
        bridge = binary_dilation(
            bridge, structure=_STRUCTURE_26, iterations=int(radius_voxels)
        )
    return bridge


def _attempt_tube_bridge(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    shape: tuple[int, int, int],
    sampling_zyx: tuple[float, float, float],
    max_bridge_distance_microns: float,
    min_cylindricality: float,
    max_axis_angle_degrees: float,
    min_facing_cosine: float,
    max_radius_ratio: float,
) -> tuple[bool, np.ndarray, str]:
    """Single-mask analog of ``graph.mask_continuity._attempt_cylinder_
    bridge``, minus the same-type-corridor / opposite-type-exclusion checks
    (those exist there for bridging within one of two *type-locked* masks;
    there is only one mask here).
    """
    empty = np.zeros(shape, dtype=bool)
    if float(source["linearity"]) < float(min_cylindricality):
        return False, empty, "source_not_cylindrical"
    if float(target["linearity"]) < float(min_cylindricality):
        return False, empty, "target_not_cylindrical"

    spacing = np.asarray(sampling_zyx, dtype=float).reshape(1, 3)
    pairs = [
        (a, b) for a in source["endpoints"] for b in target["endpoints"]
    ]
    best_pair, best_distance = None, np.inf
    for p0, p1 in pairs:
        distance = float(np.linalg.norm((p1 - p0).astype(float) * spacing[0]))
        if distance < best_distance:
            best_distance, best_pair = distance, (p0, p1)
    if best_pair is None:
        return False, empty, "no_endpoint_pair"
    if best_distance > float(max_bridge_distance_microns):
        return False, empty, "bridge_too_long"
    p0, p1 = best_pair

    v = (p1 - p0).astype(float) * spacing[0]
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-9:
        return False, empty, "degenerate_endpoint_vector"
    v_hat = v / v_norm
    source_axis = np.asarray(source["principal_axis"], dtype=float) * spacing[0]
    target_axis = np.asarray(target["principal_axis"], dtype=float) * spacing[0]
    source_axis = source_axis / max(1e-9, float(np.linalg.norm(source_axis)))
    target_axis = target_axis / max(1e-9, float(np.linalg.norm(target_axis)))
    # Orient the source axis to point toward the target, and the target axis
    # to point back toward the source -- "facing" is then how well each
    # fragment's own tube direction actually aims at the other.
    if float(np.dot(source_axis, v_hat)) < 0.0:
        source_axis = -source_axis
    if float(np.dot(target_axis, v_hat)) > 0.0:
        target_axis = -target_axis
    source_facing = float(np.dot(source_axis, v_hat))
    target_facing = float(np.dot(target_axis, -v_hat))
    if source_facing < float(min_facing_cosine) or target_facing < float(min_facing_cosine):
        return False, empty, "endpoint_facing_mismatch"

    angle = _axis_angle_deg(source["principal_axis"], target["principal_axis"])
    if angle > float(max_axis_angle_degrees):
        return False, empty, "axis_mismatch"

    src_r = max(1e-6, float(source["median_radius_um"]))
    tgt_r = max(1e-6, float(target["median_radius_um"]))
    ratio = max(src_r, tgt_r) / min(src_r, tgt_r)
    if ratio > float(max_radius_ratio):
        return False, empty, "radius_ratio_mismatch"

    line = _line_indices(p0, p1)
    # The finest axis's voxel size is the conservative conversion basis
    # (matching graph.mask_continuity's own precedent): using the coarsest
    # axis instead would under-cover the bridge along the finer axes.
    voxel_scale_um = min(float(v) for v in sampling_zyx)
    avg_radius_voxels = 0.5 * (src_r + tgt_r) / max(1e-9, voxel_scale_um)
    bridge_radius_voxels = int(min(3, max(0, round(avg_radius_voxels))))
    bridge = _bridge_mask_from_line(line, shape, radius_voxels=bridge_radius_voxels)
    return True, bridge, "bridged"


def reconnect_vessel_like_components(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    max_bridge_distance_um: float = 30.0,
    min_cylindricality: float = 0.5,
    max_axis_angle_degrees: float = 30.0,
    min_facing_cosine: float = 0.85,
    max_radius_ratio: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bridge disconnected mask components that look like the same vessel
    tube continuing.

    Labels 26-connected components, builds a ``cKDTree`` over every
    component's two endpoints (in physical microns), and for each component
    queries its nearby-endpoint candidates sorted shortest-first. Each
    candidate is gated through :func:`_attempt_tube_bridge` (both ends
    cylindrical enough, close enough, well-aligned, similarly sized);
    accepted bridges are OR'd into the mask and their two components
    unioned (union-find, the same greedy shortest-first pattern
    :func:`haemolynx.preprocessing.skeleton.connect_skeleton_components`
    already uses for the skeleton), so an already-merged pair is never
    re-attempted. Single-pass: a chain of 3+ fragments may need this run
    twice to fully close.

    Returns ``(mask, stats)`` where ``stats`` has ``attempted_bridges``,
    ``accepted_bridges`` and ``rejected_reasons`` (a ``dict[str, int]``
    counting why each rejected candidate failed).
    """
    mask = np.asarray(mask, dtype=bool)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    labeled, count = _connected_components(mask)
    stats: dict[str, Any] = {
        "attempted_bridges": 0,
        "accepted_bridges": 0,
        "rejected_reasons": {},
    }
    if count < 2:
        return mask, stats

    edt_inside = distance_transform_edt(mask, sampling=sampling)
    descriptors = _component_descriptors(labeled=labeled, count=count, edt_inside=edt_inside)
    component_ids = sorted(descriptors)
    spacing = np.asarray(sampling, dtype=float).reshape(1, 3)
    endpoint_points: list[np.ndarray] = []
    endpoint_owner: list[int] = []
    for cid in component_ids:
        for endpoint in descriptors[cid]["endpoints"]:
            endpoint_points.append(endpoint.astype(float))
            endpoint_owner.append(cid)
    endpoints_arr = np.asarray(endpoint_points, dtype=float)
    tree = cKDTree(endpoints_arr * spacing)

    parent = {cid: cid for cid in component_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    candidates: list[tuple[float, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i, cid in enumerate(component_ids):
        for endpoint in descriptors[cid]["endpoints"]:
            nearby = tree.query_ball_point(
                endpoint.astype(float) * sampling, r=float(max_bridge_distance_um)
            )
            for idx in nearby:
                other_cid = endpoint_owner[int(idx)]
                if other_cid == cid:
                    continue
                pair = (min(cid, other_cid), max(cid, other_cid))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                distance = float(
                    np.linalg.norm(
                        (endpoints_arr[int(idx)] - endpoint.astype(float)) * spacing[0]
                    )
                )
                candidates.append((distance, cid, other_cid))
    candidates.sort(key=lambda row: row[0])

    result = mask.copy()
    shape = mask.shape
    for _distance, cid_a, cid_b in candidates:
        if find(cid_a) == find(cid_b):
            continue
        stats["attempted_bridges"] += 1
        accepted, bridge, reason = _attempt_tube_bridge(
            descriptors[cid_a],
            descriptors[cid_b],
            shape=shape,
            sampling_zyx=sampling,
            max_bridge_distance_microns=max_bridge_distance_um,
            min_cylindricality=min_cylindricality,
            max_axis_angle_degrees=max_axis_angle_degrees,
            min_facing_cosine=min_facing_cosine,
            max_radius_ratio=max_radius_ratio,
        )
        if accepted:
            result |= bridge
            parent[find(cid_a)] = find(cid_b)
            stats["accepted_bridges"] += 1
        else:
            stats["rejected_reasons"][reason] = stats["rejected_reasons"].get(reason, 0) + 1

    return result, stats


def smooth_vessel_surfaces(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    sigma_um: float = 1.0,
) -> np.ndarray:
    """Gaussian-blur-then-rethreshold: reduces surface noise, pulling a
    ragged mask toward a smoother, more cylindrical shape.

    Anisotropy-aware: ``sigma_voxels[axis] = sigma_um / voxel_size_zyx[axis]``,
    so a ``(1.0, 0.4, 0.4)`` zyx dataset smooths the same physical distance
    on every axis despite sampling y/x 2.5x more densely than z. One
    vectorised ``scipy.ndimage.gaussian_filter`` call; no per-voxel loop.
    ``sigma_um <= 0`` is a no-op.
    """
    mask = np.asarray(mask, dtype=bool)
    if float(sigma_um) <= 0.0:
        return mask
    sigma_voxels = tuple(
        float(sigma_um) / max(1e-9, float(v)) for v in voxel_size_zyx
    )
    blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma_voxels)
    return blurred > 0.5


def remove_small_segmented_volumes(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_volume_um3: float = 5.0,
) -> np.ndarray:
    """Drop connected components smaller than ``min_volume_um3``.

    The same ``skimage.morphology.remove_small_objects`` this codebase
    already uses on the *skeleton*
    (:func:`haemolynx.preprocessing.skeleton.preprocess_skeleton_for_graph`),
    here on the raw mask instead. ``min_size`` in voxels is
    ``min_volume_um3`` divided by one voxel's physical volume.
    ``min_volume_um3 <= 0`` is a no-op.
    """
    mask = np.asarray(mask, dtype=bool)
    if float(min_volume_um3) <= 0.0:
        return mask
    voxel_volume = (
        float(voxel_size_zyx[0]) * float(voxel_size_zyx[1]) * float(voxel_size_zyx[2])
    )
    min_size_voxels = int(np.ceil(float(min_volume_um3) / max(1e-9, voxel_volume)))
    return remove_small_objects(mask, min_size=min_size_voxels, connectivity=3)


def clean_segmented_mask_for_skeletonisation(
    image: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    reconnect_gaps: bool = False,
    reconnect_max_bridge_distance_um: float = 30.0,
    reconnect_min_cylindricality: float = 0.5,
    reconnect_max_axis_angle_degrees: float = 30.0,
    reconnect_min_facing_cosine: float = 0.85,
    reconnect_max_radius_ratio: float = 3.0,
    smooth_surfaces: bool = False,
    smooth_sigma_um: float = 1.0,
    remove_small_volumes: bool = False,
    remove_small_min_volume_um3: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reconnect -> smooth -> remove-small, each independently toggleable.

    Order matters: reconnect first so fragments merge before size-filtering
    (a two-piece vessel that is individually below the volume threshold
    survives once bridged); smooth in the middle so the bridge geometry
    itself also gets smoothed, not left a hard-edged stub; remove-small last
    so voxels smoothing erodes off a marginal component are still caught by
    the same pass.

    *image* must already be canonically binarised by the caller -- this
    module has no ``io`` dependency (see the module docstring). Returns
    ``(cleaned, raw)`` where ``raw`` is ``None`` (no copy, no extra memory)
    when every step is off; ``raw`` is the pre-cleanup boolean array only
    when at least one step actually ran, which is what feeds the GUI's
    corrected-vs-raw comparison toggle.
    """
    if not (reconnect_gaps or smooth_surfaces or remove_small_volumes):
        return image, None
    raw = np.asarray(image, dtype=bool).copy()
    cleaned = raw
    if reconnect_gaps:
        cleaned, _stats = reconnect_vessel_like_components(
            cleaned,
            voxel_size_zyx=voxel_size_zyx,
            max_bridge_distance_um=reconnect_max_bridge_distance_um,
            min_cylindricality=reconnect_min_cylindricality,
            max_axis_angle_degrees=reconnect_max_axis_angle_degrees,
            min_facing_cosine=reconnect_min_facing_cosine,
            max_radius_ratio=reconnect_max_radius_ratio,
        )
    if smooth_surfaces:
        cleaned = smooth_vessel_surfaces(
            cleaned, voxel_size_zyx=voxel_size_zyx, sigma_um=smooth_sigma_um
        )
    if remove_small_volumes:
        cleaned = remove_small_segmented_volumes(
            cleaned, voxel_size_zyx=voxel_size_zyx, min_volume_um3=remove_small_min_volume_um3
        )
    return cleaned.astype(bool, copy=False), raw
