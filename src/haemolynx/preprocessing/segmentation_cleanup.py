"""Pre-skeletonization cleanup of the raw segmented binary vessel mask.

Seven independently-toggleable steps, always applied fill-cavities ->
remove-whiskers -> split-narrow-necks -> close-small-gaps -> reconnect ->
smooth -> remove-small when more than one is on (see
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

:func:`close_small_gaps` is a lighter, indiscriminate complement to
:func:`reconnect_vessel_like_components`: a single-voxel-scale dropout
right at a junction leaves a fragment too short for a principal axis to be
well-defined, so reconnect's own cylindricality gate correctly declines to
bridge it -- but a gap that size needs no shape test to trust, so a small
anisotropy-aware morphological closing handles it instead, before
reconnect ever sees it.

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

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
    maximum_filter,
)
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .skeleton import drop_small_components, fill_binary_holes
from .memmap_support import (
    map_blockwise,
    map_by_slab,
    new_memmap_array,
    release_memmap_array,
    release_superseded,
)
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
    "close_small_gaps",
    "clean_segmented_mask_for_skeletonisation",
]

logger = logging.getLogger(__name__)

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def _log_step_stats(step: str, stats: dict[str, Any]) -> None:
    """One line saying what a cleanup step did: its counts, then why it
    declined what it declined, most common reason first."""
    counts = ", ".join(
        f"{key}={value}" for key, value in stats.items() if key != "rejected_reasons"
    )
    reasons = sorted(
        (stats.get("rejected_reasons") or {}).items(), key=lambda item: (-item[1], item[0])
    )
    declined = ", ".join(f"{reason} x{count}" for reason, count in reasons) or "none"
    logger.info("Segmentation cleanup, %s: %s; declined: %s.", step, counts, declined)


def _morphology(
    mask: np.ndarray,
    op,
    *,
    reach: int,
    use_memmap: bool,
    memmap_directory: str | Path | None,
) -> np.ndarray:
    """``op(mask)``; under *use_memmap* (the low-RAM option) one padded block
    at a time into a new memmap, exact because *op* reads no further than
    *reach* voxels (see ``memmap_support.iter_blocks``)."""
    if not use_memmap:
        return op(np.asarray(mask, dtype=bool))
    out = new_memmap_array(mask.shape, bool, directory=memmap_directory)
    return map_blockwise(
        np.asanyarray(mask, dtype=bool), lambda block: op(block), out, halo=int(reach)
    )


def fill_enclosed_cavities(
    mask: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
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

    *use_memmap* and *memmap_directory* are forwarded to
    :func:`haemolynx.preprocessing.skeleton.fill_binary_holes`.
    """
    return fill_binary_holes(
        np.asanyarray(mask, dtype=bool), use_memmap=use_memmap, memmap_directory=memmap_directory
    )


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


#: Farthest-point extremities examined per component as candidate tips: two
#: for a straight or curved fragment's ends, more for a branched one's.
_MAX_TIPS_PER_COMPONENT = 6
#: At most this many voxels of one component take part in the farthest-point
#: search (every n-th voxel beyond it); each extremity is then refined in its
#: own neighbourhood, so the subsampling only decides roughly where to look.
_TIP_SEARCH_MAX_POINTS = 200_000
#: A tip's neighbourhood reaches this many local radii back into its
#: component: far enough for a stable axis, short enough to follow a bend.
_TIP_NEIGHBOURHOOD_RADII = 4.0
#: ...and never less than this many of the coarsest voxel spacing.
_TIP_NEIGHBOURHOOD_MIN_VOXELS = 3.0
#: A point is a tip -- the end of a tube, not the side of one -- when its
#: neighbourhood's centroid lies at least this fraction of the neighbourhood
#: radius behind it along the local axis. A tip with its neighbourhood
#: centred on it has its centroid about half a radius back; a point on a
#: tube's side has it level.
_TIP_MIN_ENDNESS = 0.3


def _box_around(
    center_idx: np.ndarray, reach_um: float, sampling: np.ndarray, shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """The voxel box reaching *reach_um* around *center_idx* on every axis."""
    half = np.ceil(float(reach_um) / sampling).astype(int)
    lo = np.maximum(center_idx - half, 0)
    hi = np.minimum(center_idx + half + 1, np.asarray(shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _principal_axis_and_linearity(points_um: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """``(unit axis, linearity, centroid)`` of a physical point cloud.

    Linearity is ``(lambda0 - lambda1) / lambda0`` of its covariance: near 1
    for a tube, near 0 for a blob. In microns, so an anisotropic voxel grid
    does not tilt the axis toward its finely sampled axes.
    """
    centroid = points_um.mean(axis=0)
    if points_um.shape[0] < 3:
        return np.asarray([1.0, 0.0, 0.0]), 0.0, centroid
    cov = _symmetric_covariance_3x3(points_um - centroid)
    (eigval0, axis), (eigval1, _axis1) = _top_two_eigen_3x3(cov)
    linearity = float((eigval0 - eigval1) / max(1e-12, float(eigval0)))
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0])
    return axis, linearity, centroid


def _tip_descriptor(
    start_idx: np.ndarray,
    *,
    component_id: int,
    labeled: np.ndarray,
    edt_inside: np.ndarray,
    sampling: np.ndarray,
) -> dict[str, Any]:
    """The local shape of *component_id* around the extremity *start_idx*.

    The neighbourhood is the component's own voxels within a few local radii
    of the extremity -- read from a small box of *labeled*, not the whole
    component -- and its radius is re-estimated from that neighbourhood until
    the two agree (twice is enough). Everything is in microns.
    """
    min_reach = _TIP_NEIGHBOURHOOD_MIN_VOXELS * float(sampling.max())
    reach = min_reach
    start_um = start_idx.astype(float) * sampling
    near_idx = start_idx.reshape(1, 3)
    radius = 0.0
    for _ in range(3):
        box = _box_around(start_idx, reach, sampling, labeled.shape)
        offset = np.asarray([s.start for s in box])
        own = np.asarray(labeled[box]) == component_id
        local = np.argwhere(own) + offset
        points = local.astype(float) * sampling
        keep = np.linalg.norm(points - start_um, axis=1) <= reach
        if keep.any():
            near_idx = local[keep]
        # The radius is read on the medial ridge (local maxima of the distance
        # transform): a statistic over every voxel sits near the surface,
        # where most of a tube's voxels are, and reads well short of it.
        edt_box = np.asarray(edt_inside[box], dtype=float)
        ridge = own & (edt_box > 0) & (maximum_filter(edt_box, footprint=_STRUCTURE_26) == edt_box)
        ridge_idx = np.argwhere(ridge) + offset
        ridge_near = np.linalg.norm(ridge_idx.astype(float) * sampling - start_um, axis=1) <= reach
        radii = np.asarray(edt_inside[tuple(ridge_idx[ridge_near].T)], dtype=float)
        if radii.size == 0:
            radii = np.asarray(edt_inside[tuple(near_idx.T)], dtype=float)
        radius = float(np.median(radii)) if radii.size else 0.0
        new_reach = max(_TIP_NEIGHBOURHOOD_RADII * radius, min_reach)
        if abs(new_reach - reach) <= 1e-9 * max(new_reach, 1.0):
            break
        reach = new_reach

    near_um = near_idx.astype(float) * sampling
    axis, linearity, centroid = _principal_axis_and_linearity(near_um)
    projection = _project_onto_axis(near_um - centroid, axis)
    # Orient outward: from the neighbourhood's centroid toward the extremity.
    if float(np.dot(start_um - centroid, axis)) < 0.0:
        axis = -axis
        projection = -projection
    # The tip point is the centre of the end face: the voxels within half the
    # finest spacing of the furthest one out. Measuring gaps between end-face
    # centres, not between arbitrary corner voxels, is what makes a distance
    # gate exact for a tube whose cross-section is several voxels wide.
    if projection.size:
        end_face = projection >= float(projection.max()) - 0.5 * float(sampling.min())
        tip_um = near_um[end_face].mean(axis=0)
    else:
        tip_um = start_um
    endness = float(np.dot(tip_um - centroid, axis)) / max(reach, 1e-12)
    return {
        "point_um": tip_um,
        "axis": axis,
        "linearity": float(linearity) if near_um.shape[0] >= 3 else 0.0,
        "radius_um": float(radius),
        "endness": endness,
    }


def _extremities(points_um: np.ndarray, *, min_separation_um: float) -> list[int]:
    """Farthest-point sampling: indices of up to ``_MAX_TIPS_PER_COMPONENT``
    points of *points_um*, each as far as possible from those already taken,
    stopping once the next would be within *min_separation_um* of one."""
    if points_um.shape[0] == 0:
        return []
    centroid = points_um.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points_um - centroid, axis=1)))
    chosen = [first]
    nearest = np.linalg.norm(points_um - points_um[first], axis=1)
    while len(chosen) < _MAX_TIPS_PER_COMPONENT:
        candidate = int(np.argmax(nearest))
        if float(nearest[candidate]) < float(min_separation_um):
            break
        chosen.append(candidate)
        nearest = np.minimum(nearest, np.linalg.norm(points_um - points_um[candidate], axis=1))
    return chosen


def _component_descriptors(
    *,
    labeled: np.ndarray,
    count: int,
    edt_inside: np.ndarray,
    sampling: tuple[float, float, float] | None = None,
) -> dict[int, dict[str, Any]]:
    """Per labeled component: its size and its candidate *tips*, each with the
    local axis, linearity, radius and "endness" of the component around it.

    Earlier versions fitted one principal axis to the whole component and
    took its two extreme voxels: a curved or branched fragment then read as
    a blob (low global linearity) and was never bridged, and a thick-and-thin
    fragment's median radius described neither part. Now every extremity
    (:func:`_extremities`) is described by its own neighbourhood
    (:func:`_tip_descriptor`), in microns.

    *edt_inside* is a Euclidean distance transform of the whole mask in
    physical units, computed once by the caller. The component's full voxel
    list is not kept once it has been described.
    """
    spacing = np.asarray(sampling if sampling is not None else (1.0, 1.0, 1.0), dtype=float)
    descriptors: dict[int, dict[str, Any]] = {}
    if count <= 0:
        return descriptors
    slices = find_objects(labeled, max_label=count)
    min_separation = 2.0 * _TIP_NEIGHBOURHOOD_MIN_VOXELS * float(spacing.max())
    for component_id in range(1, count + 1):
        component_slice = slices[component_id - 1] if slices else None
        if component_slice is None:
            continue
        local = np.asarray(labeled[component_slice])
        coords = np.argwhere(local == component_id)
        if coords.size == 0:
            continue
        offset = np.asarray([int(s.start) for s in component_slice], dtype=int).reshape(1, 3)
        size = int(coords.shape[0])
        stride = max(1, size // _TIP_SEARCH_MAX_POINTS)
        sample = coords[::stride] + offset
        del coords, local
        tips = [
            _tip_descriptor(
                sample[index],
                component_id=component_id,
                labeled=labeled,
                edt_inside=edt_inside,
                sampling=spacing,
            )
            for index in _extremities(
                sample.astype(float) * spacing, min_separation_um=min_separation
            )
        ]
        descriptors[component_id] = {
            "component_id": int(component_id),
            "size_voxels": size,
            "tips": tips,
        }
    return descriptors


def _axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    na = axis_a / max(1e-9, float(np.linalg.norm(axis_a)))
    nb = axis_b / max(1e-9, float(np.linalg.norm(axis_b)))
    cosine = float(np.clip(abs(float(np.dot(na, nb))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _tube_between(
    start_um: np.ndarray,
    end_um: np.ndarray,
    radius_um: float,
    *,
    sampling: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[tuple[slice, ...], np.ndarray] | None:
    """The voxels within *radius_um* of the segment *start_um*-*end_um*, as
    ``(slices, local)`` so ``result[slices] |= local`` ORs it in.

    Sized in microns on each axis -- a wide vessel's gap gets a bridge as wide
    as the vessel, not a fixed few voxels -- and always containing the voxels
    the segment itself passes through, so even a bridge thinner than a voxel
    still connects.
    """
    reach = float(radius_um) + float(sampling.max())
    lo_um = np.minimum(start_um, end_um) - reach
    hi_um = np.maximum(start_um, end_um) + reach
    lo = np.maximum(np.floor(lo_um / sampling).astype(int), 0)
    hi = np.minimum(np.ceil(hi_um / sampling).astype(int) + 1, np.asarray(shape))
    if np.any(hi <= lo):
        return None
    grids = np.meshgrid(
        *(np.arange(int(a), int(b), dtype=float) * s for a, b, s in zip(lo, hi, sampling)),
        indexing="ij",
    )
    points = np.stack([g.ravel() for g in grids], axis=1)
    segment = end_um - start_um
    length_sq = float(np.dot(segment, segment))
    if length_sq > 0.0:
        t = np.clip(_project_onto_axis(points - start_um, segment) / length_sq, 0.0, 1.0)
    else:
        t = np.zeros(points.shape[0])
    distance = np.linalg.norm(points - (start_um + t[:, None] * segment), axis=1)
    local = (distance <= float(radius_um) + 1e-9).reshape(tuple(hi - lo))
    steps = int(max(2, np.ceil(np.sqrt(length_sq) / float(sampling.min())) + 1))
    along = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
    line = np.rint((start_um + along * segment) / sampling).astype(int) - lo
    inside = np.all((line >= 0) & (line < (hi - lo)), axis=1)
    line = line[inside]
    local[line[:, 0], line[:, 1], line[:, 2]] = True
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), local


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
) -> tuple[bool, tuple[tuple[slice, ...], np.ndarray] | None, str]:
    """Whether two *tips* (see :func:`_tip_descriptor`) are the same tube
    continuing across a gap, and if so the bridge that joins them.

    Single-mask analog of ``graph.mask_continuity._attempt_cylinder_bridge``,
    minus the same-type-corridor / opposite-type-exclusion checks (those
    exist there for bridging within one of two *type-locked* masks; there is
    only one mask here). Every test reads the two tips' own neighbourhoods,
    not their whole components.
    """
    spacing = np.asarray(sampling_zyx, dtype=float)
    if float(source["linearity"]) < float(min_cylindricality):
        return False, None, "source_not_cylindrical"
    if float(target["linearity"]) < float(min_cylindricality):
        return False, None, "target_not_cylindrical"
    if float(source["endness"]) < _TIP_MIN_ENDNESS:
        return False, None, "source_not_a_tip"
    if float(target["endness"]) < _TIP_MIN_ENDNESS:
        return False, None, "target_not_a_tip"

    start = np.asarray(source["point_um"], dtype=float)
    end = np.asarray(target["point_um"], dtype=float)
    v = end - start
    distance = float(np.linalg.norm(v))
    if distance > float(max_bridge_distance_microns) + 1e-9:
        return False, None, "bridge_too_long"
    if distance <= 1e-9:
        return False, None, "degenerate_endpoint_vector"
    v_hat = v / distance
    # Each tip's axis already points outward, away from its own component,
    # so two ends of one tube facing each other both point along the gap.
    source_facing = float(np.dot(source["axis"], v_hat))
    target_facing = float(np.dot(target["axis"], -v_hat))
    if source_facing < float(min_facing_cosine) or target_facing < float(min_facing_cosine):
        return False, None, "endpoint_facing_mismatch"

    if _axis_angle_deg(source["axis"], target["axis"]) > float(max_axis_angle_degrees):
        return False, None, "axis_mismatch"

    src_r = max(1e-6, float(source["radius_um"]))
    tgt_r = max(1e-6, float(target["radius_um"]))
    if max(src_r, tgt_r) / min(src_r, tgt_r) > float(max_radius_ratio):
        return False, None, "radius_ratio_mismatch"

    bridge = _tube_between(start, end, 0.5 * (src_r + tgt_r), sampling=spacing, shape=shape)
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
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bridge disconnected mask components that look like the same vessel
    tube continuing.

    Labels 26-connected components and finds each one's candidate *tips* --
    its farthest-point extremities, each described by its own neighbourhood:
    a local axis, linearity and radius, in microns (see
    :func:`_component_descriptors`). Pairs of tips on different components
    within *max_bridge_distance_um* of each other are tried shortest-first,
    each gated through :func:`_attempt_tube_bridge` (both ends tubular, both
    genuinely ends, facing each other, aligned, similarly sized). An accepted
    bridge is a tube as wide as the two ends' mean radius, ORed into the
    mask, and its two components are unioned (union-find, the same greedy
    shortest-first pattern
    :func:`haemolynx.preprocessing.skeleton.connect_skeleton_components`
    uses for the skeleton), so an already-merged pair is never re-attempted.

    Only a component's extremities are tried as tips, so a gap in the middle
    of a large network (not near one of its few extremities) is not found
    here; this step joins fragments, which are small enough for their ends to
    be their extremities.

    Returns ``(mask, stats)`` where ``stats`` has ``attempted_bridges``,
    ``accepted_bridges`` and ``rejected_reasons`` (a ``dict[str, int]``
    counting why each rejected candidate failed).

    *use_memmap* (the low-RAM option) keeps every volume-sized array on
    disk: the component labels, ``edt_inside`` -- computed one padded block
    at a time (:func:`haemolynx.preprocessing.pointwise_distance.
    distance_transform_edt_blockwise`), since scipy's own transform needs
    ~70 bytes of RAM per voxel whatever it writes into -- and the returned
    mask. *memmap_directory* is forwarded to :func:`new_memmap_array`.
    """
    mask = np.asanyarray(mask, dtype=bool) if use_memmap else np.asarray(mask, dtype=bool)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    stats: dict[str, Any] = {
        "attempted_bridges": 0,
        "accepted_bridges": 0,
        "rejected_reasons": {},
    }
    owned: list[np.ndarray] = []
    try:
        if use_memmap:
            labeled = new_memmap_array(mask.shape, np.int32, directory=memmap_directory)
            owned.append(labeled)
            count = int(label(mask, structure=_STRUCTURE_26, output=labeled))
        else:
            labeled, count = _connected_components(mask)
        if count < 2:
            return mask, stats

        if use_memmap:
            from .pointwise_distance import distance_transform_edt_blockwise

            edt_inside = new_memmap_array(mask.shape, np.float64, directory=memmap_directory)
            owned.append(edt_inside)
            distance_transform_edt_blockwise(mask, edt_inside, sampling=sampling)
        else:
            edt_inside = distance_transform_edt(mask, sampling=sampling)
        return _reconnect_using_edt(
            mask,
            sampling=sampling,
            labeled=labeled,
            count=count,
            edt_inside=edt_inside,
            stats=stats,
            max_bridge_distance_um=max_bridge_distance_um,
            min_cylindricality=min_cylindricality,
            max_axis_angle_degrees=max_axis_angle_degrees,
            min_facing_cosine=min_facing_cosine,
            max_radius_ratio=max_radius_ratio,
            result=(
                map_by_slab(
                    mask,
                    lambda slab: slab.copy(),
                    new_memmap_array(mask.shape, bool, directory=memmap_directory),
                )
                if use_memmap
                else None
            ),
        )
    finally:
        for array in owned:
            release_memmap_array(array)


def _reconnect_using_edt(
    mask: np.ndarray,
    *,
    sampling: tuple[float, float, float],
    labeled: np.ndarray,
    count: int,
    edt_inside: np.ndarray,
    stats: dict[str, Any],
    max_bridge_distance_um: float,
    min_cylindricality: float,
    max_axis_angle_degrees: float,
    min_facing_cosine: float,
    max_radius_ratio: float,
    result: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The rest of :func:`reconnect_vessel_like_components`, split out so its
    caller can release memmap-backed arrays in a ``finally`` clause around
    whichever return path this takes. *result*, when given, is a copy of
    *mask* to bridge into (a disk-backed one, under the low-RAM option)."""
    descriptors = _component_descriptors(
        labeled=labeled, count=count, edt_inside=edt_inside, sampling=sampling
    )
    tips: list[dict[str, Any]] = []
    owners: list[int] = []
    for cid in sorted(descriptors):
        for tip in descriptors[cid]["tips"]:
            tips.append(tip)
            owners.append(cid)
    if len(tips) < 2:
        return (mask.copy() if result is None else result), stats

    points = np.asarray([tip["point_um"] for tip in tips], dtype=float)
    tree = cKDTree(points)
    candidates: list[tuple[float, int, int]] = []
    for i, j in sorted(tree.query_pairs(r=float(max_bridge_distance_um) + 1e-9)):
        if owners[i] == owners[j]:
            continue
        candidates.append((float(np.linalg.norm(points[i] - points[j])), i, j))
    candidates.sort(key=lambda row: row[0])

    parent = {cid: cid for cid in descriptors}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if result is None:
        result = mask.copy()
    shape = mask.shape
    for _distance, i, j in candidates:
        if find(owners[i]) == find(owners[j]):
            continue
        stats["attempted_bridges"] += 1
        accepted, bridge, reason = _attempt_tube_bridge(
            tips[i],
            tips[j],
            shape=shape,
            sampling_zyx=sampling,
            max_bridge_distance_microns=max_bridge_distance_um,
            min_cylindricality=min_cylindricality,
            max_axis_angle_degrees=max_axis_angle_degrees,
            min_facing_cosine=min_facing_cosine,
            max_radius_ratio=max_radius_ratio,
        )
        if accepted:
            if bridge is not None:
                slices, local = bridge
                result[slices] |= local
            parent[find(owners[i])] = find(owners[j])
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
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
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

    *use_memmap*, when True, writes ``edt`` -- a full-volume ``float64``
    distance transform, the widest allocation this function owns for the
    whole run -- to a disk-backed buffer instead of a fresh in-RAM one.
    ``watershed``'s own output array has no way to redirect to disk (no
    ``out=`` parameter in this dependency), so that one allocation stays
    in RAM regardless. *memmap_directory* is forwarded to
    :func:`new_memmap_array`.
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

    if use_memmap:
        # Blockwise, not scipy's own call into a memmap `distances=`: that still
        # holds ~70 bytes per voxel of its own temporaries in RAM.
        from .pointwise_distance import distance_transform_edt_blockwise

        edt = new_memmap_array(mask.shape, np.float64, directory=memmap_directory)
        try:
            distance_transform_edt_blockwise(mask, edt, sampling=sampling)
        except ValueError:
            # An all-foreground mask: scipy's result is not a distance to
            # anything, and only its own whole-volume call reproduces it.
            distance_transform_edt(mask, sampling=sampling, distances=edt)
    else:
        edt = distance_transform_edt(mask, sampling=sampling)
    try:
        return _split_using_edt(
            mask,
            sampling=sampling,
            edt=edt,
            stats=stats,
            min_marker_separation_um=min_marker_separation_um,
            min_pinch_radius_ratio=min_pinch_radius_ratio,
            min_body_radius_um=min_body_radius_um,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    finally:
        if use_memmap:
            release_memmap_array(edt)


def _marker_peaks(
    edt: np.ndarray,
    mask: np.ndarray,
    *,
    sampling: tuple[float, float, float],
    min_marker_separation_um: float,
) -> np.ndarray:
    """The EDT's local maxima at least *min_marker_separation_um* apart on
    every axis -- the watershed's markers, strongest first.

    ``peak_local_max``'s ``min_distance`` is one voxel count for every axis:
    converted with the finest spacing, a 10 um separation became 40 um along
    a 2 um z, so two vessel bodies fused one above the other were rarely
    split. On anisotropic voxels the neighbourhood is a box of the separation
    in microns on each axis and the peaks are thinned by the same physical
    (Chebyshev, as skimage's own) distance; on cube voxels it is the plain
    ``min_distance`` call, as before.
    """
    spacing = np.asarray(sampling, dtype=float)
    separation = float(min_marker_separation_um)
    # exclude_border defaults to min_distance, which would blank out the
    # whole array whenever a volume's own extent is not much bigger than
    # the marker-separation distance (real vessels legitimately run close
    # to a stack's edge) -- physical proximity to another marker is already
    # what min_distance enforces, so a plain edge is not disqualifying.
    if np.allclose(spacing, spacing[0]):
        return peak_local_max(
            edt,
            min_distance=max(1, int(round(separation / max(1e-9, float(spacing[0]))))),
            labels=mask.astype(np.int32),
            exclude_border=False,
        )
    half = [max(1, int(round(separation / max(1e-9, s)))) for s in spacing]
    candidates = peak_local_max(
        edt,
        footprint=np.ones([2 * h + 1 for h in half], dtype=bool),
        labels=mask.astype(np.int32),
        exclude_border=False,
    )
    if candidates.shape[0] < 2:
        return candidates
    from scipy.spatial import cKDTree

    physical = candidates * spacing
    tree = cKDTree(physical)
    suppressed = np.zeros(len(candidates), dtype=bool)
    kept = []
    for index in range(len(candidates)):  # strongest first, as peak_local_max returns them
        if suppressed[index]:
            continue
        kept.append(index)
        near = tree.query_ball_point(physical[index], r=separation * (1 - 1e-9), p=np.inf)
        suppressed[near] = True
    return candidates[kept]


def _split_using_edt(
    mask: np.ndarray,
    *,
    sampling: tuple[float, float, float],
    edt: np.ndarray,
    stats: dict[str, Any],
    min_marker_separation_um: float,
    min_pinch_radius_ratio: float,
    min_body_radius_um: float,
    use_memmap: bool,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The rest of :func:`split_narrow_neck_components`, split out so its
    caller can release a memmap-backed ``edt`` in a ``finally`` clause
    around whichever return path this takes."""
    peaks = _marker_peaks(
        edt, mask, sampling=sampling, min_marker_separation_um=min_marker_separation_um
    )
    if peaks.shape[0] < 2:
        return mask, stats

    peak_mask = np.zeros(mask.shape, dtype=bool)
    peak_mask[tuple(peaks.T)] = True
    if use_memmap:
        markers = new_memmap_array(mask.shape, np.int32, directory=memmap_directory)
        label(peak_mask, structure=_STRUCTURE_26, output=markers)
    else:
        markers, _n_markers = label(peak_mask, structure=_STRUCTURE_26)
    try:
        labels = watershed(-edt, markers, mask=mask)
    finally:
        if use_memmap:
            release_memmap_array(markers)
    n_labels = int(labels.max())
    stats["bodies_found"] = n_labels
    if n_labels < 2:
        return mask, stats

    pairs = _adjacent_label_pairs(labels)
    stats["adjacent_pairs"] = len(pairs)
    to_remove = np.zeros(mask.shape, dtype=bool)
    # One label -> bounding-box pass shared by every pair below, instead of
    # each pair's own `labels == label_x` / `binary_dilation(...)` scanning
    # and allocating over the *entire* volume -- real, branchy vasculature
    # can watershed into many bodies with many touching pairs, and every one
    # of those full-volume passes is wasted work outside the two bodies'
    # own (usually tiny, relative to the whole stack) local neighbourhood.
    bboxes = find_objects(labels, max_label=n_labels)
    for label_a, label_b in pairs:
        bbox_a = bboxes[label_a - 1]
        bbox_b = bboxes[label_b - 1]
        if bbox_a is None or bbox_b is None:
            continue
        # The pair's shared bounding box, padded by 1 voxel -- exactly
        # enough margin that a single-iteration `binary_dilation` (below)
        # sees the same neighbours it would against the full volume, since
        # both bodies' own true extents are already fully inside the
        # unpadded union (`find_objects` gives each label's own tight box).
        crop = tuple(
            slice(max(0, min(a.start, b.start) - 1), min(dim, max(a.stop, b.stop) + 1))
            for a, b, dim in zip(bbox_a, bbox_b, mask.shape)
        )
        labels_crop = labels[crop]
        edt_crop = edt[crop]
        region_a = labels_crop == label_a
        region_b = labels_crop == label_b
        # Each body's own peak EDT, i.e. the radius at the marker that seeded
        # it -- not the mean/median over the whole watershed catchment, which
        # for a body whose basin swallows a long stretch of a thin neck would
        # be dragged down by all those low-EDT neck voxels and never read as
        # "thick" at all.
        radius_a = float(edt_crop[region_a].max()) if region_a.any() else 0.0
        radius_b = float(edt_crop[region_b].max()) if region_b.any() else 0.0
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
        neck_radius = float(edt_crop[interface].max())
        body_radius = min(radius_a, radius_b)
        if neck_radius <= float(min_pinch_radius_ratio) * body_radius:
            to_remove[crop] |= interface
            stats["cuts_made"] += 1
        else:
            stats["rejected_reasons"]["not_narrow_enough"] = (
                stats["rejected_reasons"].get("not_narrow_enough", 0) + 1
            )

    if not to_remove.any():
        return mask, stats
    return mask & ~to_remove, stats


def _physical_ball(
    radius_um: float, voxel_size_zyx: tuple[float, float, float]
) -> tuple[np.ndarray, int]:
    """The footprint of a ball of *radius_um* microns, and its reach in voxels.

    The offsets whose physical distance from the centre is within the radius:
    a coarse axis reaches ``radius / its spacing`` voxels -- none at all when
    the radius is under one of its voxels. Rounding every axis up to at least
    one voxel (as the footprint used to) turned a 1 um
    whisker radius into 2 um along a 2 um z, and opening then deleted a 4 um
    capillary two slices thick outright. The finest axis still reaches at
    least one voxel, so a radius under a voxel acts on the smallest scale the
    grid has, as it did.
    """
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    radius = max(float(radius_um), float(spacing.min()))
    reach = np.floor(radius / spacing + 1e-9).astype(int)
    offsets = np.ogrid[tuple(slice(-r, r + 1) for r in reach)]
    structure = sum((axis * s) ** 2 for axis, s in zip(offsets, spacing)) <= radius**2 + 1e-9
    return structure, int(reach.max())


def remove_surface_whiskers(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    whisker_radius_um: float = 1.0,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Morphological opening: strips thin surface spikes still attached to
    an otherwise clean vessel, before they can survive into the skeleton as
    spurious degree-1 stub branches.

    Erosion by a ``whisker_radius_um`` ball removes anything with no core
    that thick (a 1-2 voxel whisker has none), then dilation by the same
    ball restores the vessel body's own size -- a whisker does not reappear
    since erosion left nothing there to dilate back from. The structuring
    element is an anisotropy-aware ellipsoid (:func:`_physical_ball`),
    matching :func:`smooth_vessel_surfaces`'s own precedent, so a
    ``(1.0, 0.4, 0.4)`` zyx dataset does not strip more aggressively along
    the coarser z axis than the finer y/x ones. ``whisker_radius_um <= 0``
    is a no-op.

    *use_memmap* (the low-RAM option) opens one padded block at a time into
    a disk-backed array: erosion then dilation each read one radius, so a
    halo of two radii makes every block exact.
    """
    mask = np.asanyarray(mask, dtype=bool) if use_memmap else np.asarray(mask, dtype=bool)
    if float(whisker_radius_um) <= 0.0:
        return mask
    structure, reach = _physical_ball(whisker_radius_um, voxel_size_zyx)
    return _morphology(
        mask,
        lambda m: binary_opening(m, structure=structure),
        reach=2 * reach,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )


def close_small_gaps(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    closing_radius_um: float = 0.5,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Anisotropy-aware morphological closing, at a much smaller, more
    conservative scale than
    :func:`haemolynx.preprocessing.skeleton.close_binary_mask`'s own
    voxel-only closing: bridges single-voxel-scale dropouts within one
    vessel, indiscriminately -- no shape test, unlike
    :func:`reconnect_vessel_like_components`.

    Meant for gaps too small, and too close to a junction, for reconnect's
    own PCA-based cylindricality gate to confidently accept -- a short
    fragment right at a branch point has a poorly-defined principal axis,
    so reconnect correctly declines to bridge it, but a gap that size needs
    no shape test to trust: dilation by a ``closing_radius_um`` ball merges
    it, then erosion by the same ball removes the added surface layer
    everywhere except right at the bridge, leaving genuinely separate
    vessels further apart than ``closing_radius_um`` untouched. Reuses the
    same anisotropy-aware ellipsoid footprint (:func:`_physical_ball`)
    :func:`remove_surface_whiskers` already uses, so a ``(1.0, 0.4, 0.4)``
    zyx dataset closes the same physical distance on every axis.
    ``closing_radius_um <= 0`` is a no-op.

    *use_memmap* (the low-RAM option) closes one padded block at a time into
    a disk-backed array, with a halo of two radii -- exact, as for
    :func:`remove_surface_whiskers`.
    """
    mask = np.asanyarray(mask, dtype=bool) if use_memmap else np.asarray(mask, dtype=bool)
    if float(closing_radius_um) <= 0.0:
        return mask
    structure, reach = _physical_ball(closing_radius_um, voxel_size_zyx)
    return _morphology(
        mask,
        lambda m: binary_closing(m, structure=structure),
        reach=2 * reach,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )


_SMOOTH_METHODS = ("gaussian", "morphological")


def _smooth_vessel_surfaces_morphological(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    radius_um: float,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Closing-then-opening with an anisotropic ellipsoid structuring
    element: the curvature-preserving alternative to blur-then-rethreshold.

    Closing first fills small surface concavities (imaging-noise dents)
    without adding to the vessel's own width; opening second removes small
    protrusions the same way -- the standard morphological pair for
    smoothing a binary shape without a re-threshold step, so it does not
    carry that step's own curvature-dependent bias (see
    :func:`smooth_vessel_surfaces`). Reuses :func:`_physical_ball`, the
    same anisotropy-aware footprint :func:`remove_surface_whiskers` already
    uses, so a ``(1.0, 0.4, 0.4)`` zyx dataset closes/opens the same
    physical distance on every axis. ``radius_um <= 0`` is a no-op.
    """
    if float(radius_um) <= 0.0:
        return mask
    structure, reach = _physical_ball(radius_um, voxel_size_zyx)
    return _morphology(
        mask,
        lambda m: binary_opening(binary_closing(m, structure=structure), structure=structure),
        reach=4 * reach,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )


def smooth_vessel_surfaces(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    sigma_um: float = 1.0,
    method: str = "gaussian",
    morphological_radius_um: float = 1.0,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
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

    *use_memmap*, when True, runs either method one padded block at a time
    into a disk-backed mask. Exact both ways: the gaussian's halo is its own
    kernel radius, and four radii cover the morphological closing then
    opening. *memmap_directory* is forwarded to :func:`new_memmap_array`.
    """
    mask = np.asanyarray(mask, dtype=bool) if use_memmap else np.asarray(mask, dtype=bool)
    if method == "morphological":
        return _smooth_vessel_surfaces_morphological(
            mask,
            voxel_size_zyx=voxel_size_zyx,
            radius_um=morphological_radius_um,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
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
    if use_memmap:
        # One padded block at a time, in RAM. gaussian_filter truncates each
        # axis's kernel at int(4 * sigma + 0.5) voxels and computes every
        # output from its own window alone, so a halo that wide gives each
        # core exactly the whole-volume float32 values -- instead of blurring
        # a volume-sized float32 file into another one on disk.
        halo = max(int(4.0 * float(s) + 0.5) for s in sigma_voxels)
        return map_blockwise(
            mask,
            lambda block: gaussian_filter(block.astype(np.float32), sigma=sigma_voxels) > 0.5,
            new_memmap_array(mask.shape, bool, directory=memmap_directory),
            halo=halo,
        )
    blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma_voxels)
    return blurred > 0.5


def remove_small_segmented_volumes(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_volume_um3: float = 5.0,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Drop connected components smaller than ``min_volume_um3``.

    The same :func:`haemolynx.preprocessing.skeleton.drop_small_components`
    this codebase already uses on the *skeleton*
    (:func:`haemolynx.preprocessing.skeleton.preprocess_skeleton_for_graph`),
    here on the raw mask instead. ``min_size`` in voxels is
    ``min_volume_um3`` divided by one voxel's physical volume.
    ``min_volume_um3 <= 0`` is a no-op. *use_memmap* and *memmap_directory*
    are forwarded to it; see its own docstring for what they change.
    """
    mask = np.asanyarray(mask, dtype=bool)
    if float(min_volume_um3) <= 0.0:
        return mask
    voxel_volume = (
        float(voxel_size_zyx[0]) * float(voxel_size_zyx[1]) * float(voxel_size_zyx[2])
    )
    min_size_voxels = int(np.ceil(float(min_volume_um3) / max(1e-9, voxel_volume)))
    return drop_small_components(
        mask,
        min_size=min_size_voxels,
        connectivity=3,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )


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
    close_gaps: bool = False,
    close_gaps_radius_um: float = 0.5,
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
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill-cavities -> remove-whiskers -> split-narrow-necks ->
    close-small-gaps -> reconnect -> smooth -> remove-small, each
    independently toggleable.

    Order matters: cavity filling first, so a hollow lumen from imaging
    noise does not throw off every later step's own radius/linearity
    measurements (an enclosed cavity would otherwise read as *not* part of
    the vessel); whisker removal next, so a spike does not throw off the
    radius/axis measurements every later step relies on; split before
    reconnect so a genuine false-merge is cut apart before reconnect ever
    gets a chance to characterise (and potentially re-bridge) the same
    fused pair -- running them in the other order could have reconnect
    immediately re-joining the very neck split just cut; close-small-gaps
    right before reconnect, so trivial single-voxel-scale dropouts are
    already bridged indiscriminately by the time reconnect runs, leaving
    it to spend its own shape-gated, longer-range bridging only on
    fragments genuinely too far apart or too ambiguous for a small,
    untargeted closing to have already fixed; reconnect before smooth so
    fragments merge before size-filtering (a two-piece vessel that is
    individually below the volume threshold survives once bridged); smooth
    before remove-small so the bridge/cut geometry itself also gets
    smoothed, not left a hard-edged stub, with remove-small last so voxels
    smoothing erodes off a marginal component are still caught by the same
    pass.

    *image* must already be canonically binarised by the caller -- this
    module has no ``io`` dependency (see the module docstring). Returns
    ``(cleaned, raw)`` where ``raw`` is ``None`` (no copy, no extra memory)
    when every step is off; ``raw`` is the pre-cleanup boolean array only
    when at least one step actually ran, which is what feeds the GUI's
    corrected-vs-raw comparison toggle.

    *use_memmap*, forwarded to :func:`split_narrow_neck_components`,
    :func:`reconnect_vessel_like_components`, :func:`smooth_vessel_surfaces`
    and :func:`remove_small_segmented_volumes`, also decides whether ``raw``
    itself (a full-volume boolean copy, kept alive for the rest of the run
    rather than released here) is disk-backed. The caller owns its backing
    file the same way it would own a plain array's memory.
    *memmap_directory* is forwarded to :func:`new_memmap_array` and to each
    of those four functions in turn.
    """
    if not (
        fill_cavities
        or remove_whiskers
        or split_narrow_necks
        or close_gaps
        or reconnect_gaps
        or smooth_surfaces
        or remove_small_volumes
    ):
        return image, None
    if use_memmap:
        raw = new_memmap_array(image.shape, bool, directory=memmap_directory)
        raw[:] = image
    else:
        raw = np.asarray(image, dtype=bool).copy()
    cleaned = raw

    def advance(new: np.ndarray) -> np.ndarray:
        # Under use_memmap each step writes a fresh volume-sized file; the one
        # it replaces is nobody else's (raw is returned to the caller).
        release_superseded(cleaned, new, keep=raw)
        return new

    if fill_cavities:
        cleaned = advance(
            fill_enclosed_cavities(
                cleaned, use_memmap=use_memmap, memmap_directory=memmap_directory
            )
        )
    if remove_whiskers:
        cleaned = advance(
            remove_surface_whiskers(
                cleaned,
                voxel_size_zyx=voxel_size_zyx,
                whisker_radius_um=whisker_radius_um,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )
    if split_narrow_necks:
        split, split_stats = split_narrow_neck_components(
            cleaned,
            voxel_size_zyx=voxel_size_zyx,
            min_marker_separation_um=split_min_marker_separation_um,
            min_pinch_radius_ratio=split_min_pinch_radius_ratio,
            min_body_radius_um=split_min_body_radius_um,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
        _log_step_stats("split narrow necks", split_stats)
        cleaned = advance(split)
    if close_gaps:
        cleaned = advance(
            close_small_gaps(
                cleaned,
                voxel_size_zyx=voxel_size_zyx,
                closing_radius_um=close_gaps_radius_um,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )
    if reconnect_gaps:
        reconnected, reconnect_stats = reconnect_vessel_like_components(
            cleaned,
            voxel_size_zyx=voxel_size_zyx,
            max_bridge_distance_um=reconnect_max_bridge_distance_um,
            min_cylindricality=reconnect_min_cylindricality,
            max_axis_angle_degrees=reconnect_max_axis_angle_degrees,
            min_facing_cosine=reconnect_min_facing_cosine,
            max_radius_ratio=reconnect_max_radius_ratio,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
        _log_step_stats("reconnect gaps", reconnect_stats)
        cleaned = advance(reconnected)
    if smooth_surfaces:
        cleaned = advance(
            smooth_vessel_surfaces(
                cleaned,
                voxel_size_zyx=voxel_size_zyx,
                sigma_um=smooth_sigma_um,
                method=smooth_method,
                morphological_radius_um=smooth_morphological_radius_um,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )
    if remove_small_volumes:
        cleaned = advance(
            remove_small_segmented_volumes(
                cleaned,
                voxel_size_zyx=voxel_size_zyx,
                min_volume_um3=remove_small_min_volume_um3,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )
    return cleaned.astype(bool, copy=False), raw
