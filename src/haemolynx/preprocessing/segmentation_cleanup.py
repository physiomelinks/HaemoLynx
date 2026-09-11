"""Pre-skeletonization cleanup of the raw segmented binary vessel mask.

Six independently-toggleable steps, always applied fill-cavities ->
remove-whiskers -> split-narrow-necks -> reconnect -> smooth -> remove-small
when more than one is on (see
:func:`clean_segmented_mask_for_skeletonisation` for why that order). All
distance/volume parameters are physical (microns / cubic microns), sampled
via ``voxel_size_zyx`` -- unlike
:func:`haemolynx.preprocessing.skeleton.close_binary_mask`/``bridge_gaps``,
which are voxel-unit only, a real problem for this project's anisotropic
datasets (e.g. a ``(1.0, 0.4, 0.4)`` zyx voxel size, where a "radius=2
voxel" ball is 2.0 microns in z but only 0.8 microns in x/y).

:func:`split_narrow_neck_components` is the inverse of
:func:`reconnect_vessel_like_components`: reconnect repairs a false split
(one vessel broken into fragments by a segmentation gap); split repairs a
false merge (two distinct, touching vessels segmented as one blob), cutting
at a genuine pinch -- a watershed ridge between two mask "bodies" whose own
cross-sectional radius there is markedly narrower than either body, not
just wherever a marker-controlled watershed happens to place a boundary
along an otherwise uniform vessel.

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
imported -- except the crash-avoiding 3x3 eigen helpers below, shared with
:mod:`haemolynx.preprocessing.thick_vessels`, which solves the identical
"BLAS crashes on this environment" problem for the identical 3x3
covariance case.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from .skeleton import fill_binary_holes
from .thick_vessels import (
    _dominant_eigenvector_3x3,
    _matvec_3x3,
    _project_onto_axis,
    _symmetric_covariance_3x3,
)

__all__ = [
    "fill_enclosed_cavities",
    "reconnect_vessel_like_components",
    "smooth_vessel_surfaces",
    "remove_small_segmented_volumes",
    "split_narrow_neck_components",
    "remove_surface_whiskers",
    "clean_segmented_mask_for_skeletonisation",
]

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def fill_enclosed_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill small internal air-gaps/voids -- imaging noise inside an
    otherwise solid vessel lumen -- that would otherwise survive into the
    skeleton as a spurious closed loop (a handle in the topology, not a
    real vessel branch).

    Reuses :func:`haemolynx.preprocessing.skeleton.fill_binary_holes`, the
    same function ``use_thick_vessel_skeletonisation``'s
    ``skeleton_fill_mask_holes_before_thickness`` already applies -- but
    that only runs on the thickness-gated branch of skeletonisation
    (:func:`haemolynx.preprocessing.thick_vessels.skeletonize_thickness_gated`).
    Doing it here, once, on the mask itself, covers the plain-Lee path too,
    and does it before every later step in this module -- whisker removal,
    splitting, reconnecting -- rather than leaving a hollow lumen to
    distort their own radius/linearity measurements (an enclosed cavity
    would otherwise be counted as *not* part of the vessel).

    Purely topological -- an enclosed background region either exists or
    it does not, with nothing to size -- so unlike every other step here,
    this has no physical parameter to make anisotropy-aware.
    """
    return fill_binary_holes(np.asarray(mask, dtype=bool))


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, count = label(mask.astype(bool, copy=False), structure=_STRUCTURE_26)
    return labeled, int(count)


def _outer_3x3(vector: np.ndarray) -> np.ndarray:
    """*vector* (x) *vector* as an explicit 3x3 -- not ``np.outer``, kept in
    the same elementwise style as ``thick_vessels._symmetric_covariance_3x3``
    so nothing here reintroduces a BLAS-backed call.
    """
    result = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            result[i, j] = float(vector[i] * vector[j])
    return result


def _rayleigh_quotient_3x3(matrix: np.ndarray, vector: np.ndarray) -> float:
    """``v^T A v / v^T v`` -- *vector*'s own eigenvalue when it is (close to)
    one of *matrix*'s eigenvectors, without a full eigendecomposition.
    """
    mv = _matvec_3x3(matrix, vector)
    numerator = float(mv[0] * vector[0] + mv[1] * vector[1] + mv[2] * vector[2])
    denominator = float(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    return numerator / denominator if denominator > 1e-300 else 0.0


def _top_two_eigen_3x3(
    matrix: np.ndarray,
) -> tuple[tuple[float, np.ndarray], tuple[float, np.ndarray]]:
    """The two largest (eigenvalue, eigenvector) pairs of a symmetric 3x3
    matrix, in ``np.linalg.eigh``'s own descending order, without ``eigh``
    itself -- it crashes natively on this environment's broken NumPy/BLAS
    build for a non-trivial matrix (see
    ``haemolynx.preprocessing.thick_vessels``, which hit the identical
    fault). Power iteration finds the dominant pair
    (:func:`haemolynx.preprocessing.thick_vessels._dominant_eigenvector_3x3`,
    plus its Rayleigh-quotient eigenvalue); Hotelling deflation --
    subtracting that eigenvector's own outer-product contribution -- leaves
    a matrix whose new dominant eigenvector is the original's second, found
    the same way. Only the top two are ever needed here (a linearity
    ratio), so stopping there is the whole answer, not an approximation of
    a full decomposition.
    """
    axis0 = _dominant_eigenvector_3x3(matrix)
    eigval0 = _rayleigh_quotient_3x3(matrix, axis0)
    deflated = matrix - eigval0 * _outer_3x3(axis0)
    axis1 = _dominant_eigenvector_3x3(deflated)
    eigval1 = _rayleigh_quotient_3x3(matrix, axis1)
    return (eigval0, axis0), (eigval1, axis1)


def _component_descriptors(
    *, labeled: np.ndarray, count: int, edt_inside: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Per labeled component: centroid, PCA principal axis, a "linearity"
    (cylindricality) score, the two endpoints along that axis, and a median
    local radius in physical microns.

    Linearity is ``(eigval[0] - eigval[1]) / eigval[0]`` on the covariance of
    the component's own voxel coordinates -- near 1 for an elongated,
    tube-like component, near 0 for a blob -- computed via
    :func:`_top_two_eigen_3x3` (power iteration plus one deflation step),
    not ``np.linalg.eigh``, which crashes natively on this environment's
    broken NumPy/BLAS build for a non-trivial component. PCA runs on raw
    voxel indices, not physical coordinates (matching
    ``graph.mask_continuity``'s own precedent): a genuinely anisotropic
    dataset biases this somewhat toward the more finely sampled axes, a
    known, accepted limitation of the same approach already used in
    production elsewhere in this codebase.

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
        coords_float = coords.astype(float)
        centroid = np.mean(coords_float, axis=0)
        if coords.shape[0] >= 3:
            cov = _symmetric_covariance_3x3(coords_float - centroid)
            (eigval0, principal_axis), (eigval1, _axis1) = _top_two_eigen_3x3(cov)
            linearity = float(
                (eigval0 - eigval1) / max(1e-9, float(eigval0))
            )
        else:
            principal_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
            linearity = 0.0
        norm = float(np.linalg.norm(principal_axis))
        principal_axis = principal_axis / norm if norm > 1e-9 else principal_axis
        projections = _project_onto_axis(coords_float, principal_axis)
        end_a = coords[int(np.argmin(projections))]
        end_b = coords[int(np.argmax(projections))]
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


def _adjacent_label_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Every unordered pair of distinct positive labels that touch (26-conn).

    Zero-padding the array first means every one of the 26 shifted views can
    be compared against the unpadded array elementwise, with no wrap-around
    false adjacency at the volume's own edges.
    """
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    shape = labels.shape
    pairs: set[tuple[int, int]] = set()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                shifted = padded[
                    1 + dz : 1 + dz + shape[0],
                    1 + dy : 1 + dy + shape[1],
                    1 + dx : 1 + dx + shape[2],
                ]
                both_fg = (labels > 0) & (shifted > 0) & (labels != shifted)
                if not both_fg.any():
                    continue
                for a, b in zip(labels[both_fg].tolist(), shifted[both_fg].tolist()):
                    pairs.add((a, b) if a < b else (b, a))
    return pairs


def split_narrow_neck_components(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_marker_separation_um: float = 10.0,
    min_pinch_radius_ratio: float = 0.6,
    min_body_radius_um: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cut the mask apart at genuine pinch points -- narrow necks where two
    separate vessels have been fused by segmentation blur, not a real
    vessel's own natural taper.

    Marker-controlled watershed of the mask's own physical distance
    transform: each local radius maximum at least ``min_marker_separation_um``
    from its neighbours seeds one "body" (matching
    ``skimage``'s own touching-object-splitting recipe of ``peak_local_max``
    + ``watershed(-distance, markers, mask=...)``). For every pair of bodies
    that end up touching, the interface between them -- the watershed ridge
    -- is only cut when its own narrowest point (the minimum distance-
    transform value along it) is at least ``min_pinch_radius_ratio`` thinner
    than *both* bodies' own peak radius (the distance-transform value at the
    marker that seeded each one, not a mean/median over its whole catchment,
    which can swallow a long stretch of a genuinely thin neck and read as
    thin itself); a uniform-radius vessel that watershed happens to split
    into two markers has no such narrowing at the interface and is left
    untouched, physically still one component.

    Returns ``(mask, stats)`` where ``stats`` has ``bodies_found``,
    ``adjacent_pairs``, ``cuts_made`` and ``rejected_reasons`` (a
    ``dict[str, int]``), mirroring :func:`reconnect_vessel_like_components`.
    """
    mask = np.asarray(mask, dtype=bool)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    stats: dict[str, Any] = {
        "bodies_found": 0,
        "adjacent_pairs": 0,
        "cuts_made": 0,
        "rejected_reasons": {},
    }
    if not mask.any():
        return mask, stats

    edt = distance_transform_edt(mask, sampling=sampling)
    voxel_scale_um = min(sampling)
    min_distance_voxels = max(
        1, int(round(float(min_marker_separation_um) / max(1e-9, voxel_scale_um)))
    )
    # exclude_border defaults to min_distance, which would blank out the
    # whole array whenever a volume's own extent is not much bigger than
    # the marker-separation distance (real vessels legitimately run close
    # to a stack's edge) -- physical proximity to another marker is already
    # what min_distance enforces, so a plain edge is not disqualifying.
    peaks = peak_local_max(
        edt,
        min_distance=min_distance_voxels,
        labels=mask.astype(np.int32),
        exclude_border=False,
    )
    if peaks.shape[0] < 2:
        return mask, stats

    peak_mask = np.zeros(mask.shape, dtype=bool)
    peak_mask[tuple(peaks.T)] = True
    markers, _n_markers = label(peak_mask, structure=_STRUCTURE_26)
    labels = watershed(-edt, markers, mask=mask)
    n_labels = int(labels.max())
    stats["bodies_found"] = n_labels
    if n_labels < 2:
        return mask, stats

    pairs = _adjacent_label_pairs(labels)
    stats["adjacent_pairs"] = len(pairs)
    to_remove = np.zeros(mask.shape, dtype=bool)
    for label_a, label_b in pairs:
        region_a = labels == label_a
        region_b = labels == label_b
        # Each body's own peak EDT, i.e. the radius at the marker that seeded
        # it -- not the mean/median over the whole watershed catchment, which
        # for a body whose basin swallows a long stretch of a thin neck would
        # be dragged down by all those low-EDT neck voxels and never read as
        # "thick" at all.
        radius_a = float(edt[region_a].max()) if region_a.any() else 0.0
        radius_b = float(edt[region_b].max()) if region_b.any() else 0.0
        if radius_a < min_body_radius_um or radius_b < min_body_radius_um:
            stats["rejected_reasons"]["body_too_thin"] = (
                stats["rejected_reasons"].get("body_too_thin", 0) + 1
            )
            continue
        interface = (region_a & binary_dilation(region_b, structure=_STRUCTURE_26)) | (
            region_b & binary_dilation(region_a, structure=_STRUCTURE_26)
        )
        if not interface.any():
            continue
        # The interface's own *peak* EDT, not its minimum: any interface
        # cross-section, thin neck or full-bore tube alike, always runs from
        # its own surface (EDT near 0) up to its centre -- the surface edge
        # is not evidence of a pinch, only the centre value is a genuine
        # local radius to compare against the two bodies'.
        neck_radius = float(edt[interface].max())
        body_radius = min(radius_a, radius_b)
        if neck_radius <= float(min_pinch_radius_ratio) * body_radius:
            to_remove |= interface
            stats["cuts_made"] += 1
        else:
            stats["rejected_reasons"]["not_narrow_enough"] = (
                stats["rejected_reasons"].get("not_narrow_enough", 0) + 1
            )

    if not to_remove.any():
        return mask, stats
    return mask & ~to_remove, stats


def _ellipsoid_structure(radius_voxels: tuple[int, int, int]) -> np.ndarray:
    """Boolean footprint of the ellipsoid ``radius_voxels`` describes.

    Per-axis radii, not a single scalar, so an anisotropic ``voxel_size_zyx``
    still opens a physically round (not axis-stretched) neighbourhood.
    """
    rz, ry, rx = (max(1, int(r)) for r in radius_voxels)
    zz, yy, xx = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    return (zz / rz) ** 2 + (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0


def remove_surface_whiskers(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    whisker_radius_um: float = 1.0,
) -> np.ndarray:
    """Morphological opening: strips thin surface spikes still attached to
    an otherwise clean vessel, before they can survive into the skeleton as
    spurious degree-1 stub branches.

    Erosion by a ``whisker_radius_um`` ball removes anything with no core
    that thick (a 1-2 voxel whisker has none), then dilation by the same
    ball restores the vessel body's own size -- a whisker does not reappear
    since erosion left nothing there to dilate back from. The structuring
    element is an anisotropy-aware ellipsoid (:func:`_ellipsoid_structure`),
    matching :func:`smooth_vessel_surfaces`'s own precedent, so a
    ``(1.0, 0.4, 0.4)`` zyx dataset does not strip more aggressively along
    the coarser z axis than the finer y/x ones. ``whisker_radius_um <= 0``
    is a no-op.
    """
    mask = np.asarray(mask, dtype=bool)
    if float(whisker_radius_um) <= 0.0:
        return mask
    radius_voxels = tuple(
        max(1, int(round(float(whisker_radius_um) / max(1e-9, float(v)))))
        for v in voxel_size_zyx
    )
    structure = _ellipsoid_structure(radius_voxels)
    return binary_opening(mask, structure=structure)


_SMOOTH_METHODS = ("gaussian", "morphological")


def _smooth_vessel_surfaces_morphological(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    radius_um: float,
) -> np.ndarray:
    """Closing-then-opening with an anisotropic ellipsoid structuring
    element: the curvature-preserving alternative to blur-then-rethreshold.

    Closing first fills small surface concavities (imaging-noise dents)
    without adding to the vessel's own width; opening second removes small
    protrusions the same way -- the standard morphological pair for
    smoothing a binary shape without a re-threshold step, so it does not
    carry that step's own curvature-dependent bias (see
    :func:`smooth_vessel_surfaces`). Reuses :func:`_ellipsoid_structure`, the
    same anisotropy-aware footprint :func:`remove_surface_whiskers` already
    uses, so a ``(1.0, 0.4, 0.4)`` zyx dataset closes/opens the same
    physical distance on every axis. ``radius_um <= 0`` is a no-op.
    """
    if float(radius_um) <= 0.0:
        return mask
    radius_voxels = tuple(
        max(1, int(round(float(radius_um) / max(1e-9, float(v)))))
        for v in voxel_size_zyx
    )
    structure = _ellipsoid_structure(radius_voxels)
    return binary_opening(binary_closing(mask, structure=structure), structure=structure)


def smooth_vessel_surfaces(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    sigma_um: float = 1.0,
    method: str = "gaussian",
    morphological_radius_um: float = 1.0,
) -> np.ndarray:
    """Reduce surface noise, pulling a ragged mask toward a smoother, more
    cylindrical shape.

    Two methods, chosen by *method*:

    ``"gaussian"`` (default) blurs then rethresholds at 0.5 -- one vectorised
    ``scipy.ndimage.gaussian_filter`` call, anisotropy-aware
    (``sigma_voxels[axis] = sigma_um / voxel_size_zyx[axis]``, so a
    ``(1.0, 0.4, 0.4)`` zyx dataset smooths the same physical distance on
    every axis despite sampling y/x 2.5x more densely than z). Known,
    accepted tradeoff: a blur softens a thin tube's edge by roughly the same
    physical amount on both the inside and outside, but re-thresholding at
    0.5 keeps only the inside half -- a small, systematic narrowing whose
    size grows with the mask's local curvature, i.e. worst on exactly the
    thin, highly-curved vessels whose diameter (and therefore resistance)
    this step must not distort.

    ``"morphological"`` (:func:`_smooth_vessel_surfaces_morphological`)
    closes then opens with an anisotropic ellipsoid structuring element
    instead -- no threshold step, so it does not carry that same bias, at
    the cost of coarser (voxel-radius-grained, not continuous-sigma) size
    control and more compute for a large *radius_um*. Reach for this when
    diameter accuracy on thin/curved vessels matters more than smoothing a
    perfectly continuous amount.

    ``sigma_um <= 0`` (gaussian) or ``morphological_radius_um <= 0``
    (morphological) is a no-op for the method actually selected.
    """
    mask = np.asarray(mask, dtype=bool)
    if method == "morphological":
        return _smooth_vessel_surfaces_morphological(
            mask, voxel_size_zyx=voxel_size_zyx, radius_um=morphological_radius_um
        )
    if method != "gaussian":
        raise ValueError(
            f"Unknown smoothing method {method!r}. Known: {', '.join(_SMOOTH_METHODS)}."
        )
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
    fill_cavities: bool = False,
    remove_whiskers: bool = False,
    whisker_radius_um: float = 1.0,
    split_narrow_necks: bool = False,
    split_min_marker_separation_um: float = 10.0,
    split_min_pinch_radius_ratio: float = 0.6,
    split_min_body_radius_um: float = 1.0,
    reconnect_gaps: bool = False,
    reconnect_max_bridge_distance_um: float = 30.0,
    reconnect_min_cylindricality: float = 0.5,
    reconnect_max_axis_angle_degrees: float = 30.0,
    reconnect_min_facing_cosine: float = 0.85,
    reconnect_max_radius_ratio: float = 3.0,
    smooth_surfaces: bool = False,
    smooth_sigma_um: float = 1.0,
    smooth_method: str = "gaussian",
    smooth_morphological_radius_um: float = 1.0,
    remove_small_volumes: bool = False,
    remove_small_min_volume_um3: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill-cavities -> remove-whiskers -> split-narrow-necks -> reconnect ->
    smooth -> remove-small, each independently toggleable.

    Order matters: cavity filling first, so a hollow lumen from imaging
    noise does not throw off every later step's own radius/linearity
    measurements (an enclosed cavity would otherwise read as *not* part of
    the vessel); whisker removal next, so a spike does not throw off the
    radius/axis measurements every later step relies on; split before
    reconnect so a genuine false-merge is cut apart before reconnect ever
    gets a chance to characterise (and potentially re-bridge) the same
    fused pair -- running them in the other order could have reconnect
    immediately re-joining the very neck split just cut; reconnect before
    smooth so fragments merge before size-filtering (a two-piece vessel
    that is individually below the volume threshold survives once
    bridged); smooth before remove-small so the bridge/cut geometry itself
    also gets smoothed, not left a hard-edged stub, with remove-small last
    so voxels smoothing erodes off a marginal component are still caught by
    the same pass.

    *image* must already be canonically binarised by the caller -- this
    module has no ``io`` dependency (see the module docstring). Returns
    ``(cleaned, raw)`` where ``raw`` is ``None`` (no copy, no extra memory)
    when every step is off; ``raw`` is the pre-cleanup boolean array only
    when at least one step actually ran, which is what feeds the GUI's
    corrected-vs-raw comparison toggle.
    """
    if not (
        fill_cavities
        or remove_whiskers
        or split_narrow_necks
        or reconnect_gaps
        or smooth_surfaces
        or remove_small_volumes
    ):
        return image, None
    raw = np.asarray(image, dtype=bool).copy()
    cleaned = raw
    if fill_cavities:
        cleaned = fill_enclosed_cavities(cleaned)
    if remove_whiskers:
        cleaned = remove_surface_whiskers(
            cleaned, voxel_size_zyx=voxel_size_zyx, whisker_radius_um=whisker_radius_um
        )
    if split_narrow_necks:
        cleaned, _stats = split_narrow_neck_components(
            cleaned,
            voxel_size_zyx=voxel_size_zyx,
            min_marker_separation_um=split_min_marker_separation_um,
            min_pinch_radius_ratio=split_min_pinch_radius_ratio,
            min_body_radius_um=split_min_body_radius_um,
        )
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
            cleaned,
            voxel_size_zyx=voxel_size_zyx,
            sigma_um=smooth_sigma_um,
            method=smooth_method,
            morphological_radius_um=smooth_morphological_radius_um,
        )
    if remove_small_volumes:
        cleaned = remove_small_segmented_volumes(
            cleaned, voxel_size_zyx=voxel_size_zyx, min_volume_um3=remove_small_min_volume_um3
        )
    return cleaned.astype(bool, copy=False), raw
