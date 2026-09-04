"""Thickness-gated skeletonisation for fat regions of a plasma-labelled mask.

Plasma-column labelling fills the lumen, and the pipeline's main input is
typically **one connected binary object**: a fat trunk fused to capillaries,
not a separate tube. Lee thinning of the fat part of that object yields a
medial *sheet* (several polylines for one vessel). The thin part is already
fine.

This module does **not** change the pipeline. The functions here are what a
later wiring step would call.

A voxel is in the fat region when it belongs to the geodesic reconstruction of
an inscribed-radius core (see :func:`thick_vessel_object_mask`). Capillaries
fused into the same object stay out. The fat catchment is then given a
centreline *tree*: one inverted-EDT geodesic per arm, rejecting paths that
run parallel to an existing line (the medial-sheet artefact).

The radius gate is locked by
``tests/test_thick_vessel_skeletonisation.py`` on a fused plasma-labelled
fixture. Volume (µm³) of the whole object is not the gate.
"""
from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

from .skeleton import fill_binary_holes, skeletonize_volume, _draw_line_3d

# Inscribed radius (EDT of the binary mask, microns) at which Lee begins to
# produce a multi-polyline medial sheet in the fat part of a plasma-labelled
# object (elongated filled column fused to capillaries, one connected mask).
# Locked by ``test_the_locked_radius_threshold_matches_the_measured_onset``.
THICK_VESSEL_MIN_RADIUS_UM: float = 6.0

#: Mean skeleton voxels per occupied axial slice above this counts as a braid.
BRAID_FACTOR_LIMIT: float = 2.0

_OFFSETS_26 = tuple(
    (i, j, k)
    for i in (-1, 0, 1)
    for j in (-1, 0, 1)
    for k in (-1, 0, 1)
    if not (i == 0 and j == 0 and k == 0)
)


def inscribed_radius_map(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Per-voxel inscribed radius in microns: EDT of the binary mask."""
    mask = np.asarray(binary, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float64)
    return distance_transform_edt(mask, sampling=tuple(float(v) for v in voxel_size_zyx))


def max_inscribed_radius_um(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Largest inscribed radius in microns, or 0.0 for an empty mask."""
    radii = inscribed_radius_map(binary, voxel_size_zyx)
    return float(radii.max()) if radii.size else 0.0


def foreground_volume_um3(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Physical volume of the binary foreground."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    return float(np.asarray(binary, dtype=bool).sum()) * voxel_volume


def thick_vessel_object_mask(
    binary: np.ndarray,
    *,
    min_radius_um: float,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Fat-region voxels of a (possibly single) connected plasma-labelled mask.

    1. Core: inscribed radius >= *min_radius_um*.
    2. Body: geodesic reconstruction of that core through voxels fatter than
       half the threshold, so a flattened plasma column's interior is included
       without walking down fused capillaries (those stay below the half-gate).
    3. Wall: Euclidean ball of that half-gate around the body, so the plasma
       column's surface is not left for Lee to mesh, while capillaries more
       than that far from the body stay on the Lee path.
    """
    mask = np.asarray(binary, dtype=bool)
    if min_radius_um <= 0.0 or not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    radius_map = inscribed_radius_map(mask, voxel_size_zyx)
    thick_core = mask & (radius_map >= float(min_radius_um))
    if not thick_core.any():
        return np.zeros(mask.shape, dtype=bool)

    t_low = 0.5 * float(min_radius_um)
    allowed = mask & (radius_map >= t_low)
    body = binary_propagation(thick_core, mask=allowed)
    dist_to_body = distance_transform_edt(
        ~body, sampling=tuple(float(v) for v in voxel_size_zyx)
    )
    return mask & (dist_to_body <= t_low)


def braid_factor(
    skeleton: np.ndarray,
    *,
    axis: int = 0,
) -> float:
    """Mean skeleton-voxel count per axial slice that contains any skeleton.

    A single centreline is ~1. A medial sheet is several voxels in the same
    slice. Empty skeletons return 0.0.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if not skel.any():
        return 0.0
    counts = skel.sum(axis=tuple(i for i in range(skel.ndim) if i != axis))
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return 0.0
    return float(occupied.mean())


def lee_braid_factor(
    binary: np.ndarray,
    *,
    axis: int = 0,
) -> float:
    """:func:`braid_factor` of Lee thinning applied to *binary*."""
    return braid_factor(skeletonize_volume(binary), axis=axis)


def lee_sheet_excess(
    binary: np.ndarray,
    region: np.ndarray,
) -> float:
    """How many times more Lee skeleton voxels sit in *region* than an EDT ridge.

    A single centreline is ~1. A medial sheet is several times that. Empty
    regions return 0.0. *region* should be the fat catchment, not the whole
    fused object, so attached capillaries do not inflate the ratio.
    """
    roi = np.asarray(binary, dtype=bool) & np.asarray(region, dtype=bool)
    if not roi.any():
        return 0.0
    lee = skeletonize_volume(binary) & roi
    ridge = skeletonize_edt_ridge(roi)
    n_ridge = int(ridge.sum())
    if n_ridge == 0:
        return float(lee.sum())
    return float(lee.sum()) / float(n_ridge)


def _linear_index(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    z, y, x = shape
    return coords[:, 0] * (y * x) + coords[:, 1] * x + coords[:, 2]


def _from_linear(index: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    yx = shape[1] * shape[2]
    z, rem = divmod(int(index), yx)
    y, x = divmod(rem, shape[2])
    return z, y, x


def _dijkstra_parents(
    binary: np.ndarray,
    cost: np.ndarray,
    root: tuple[int, int, int],
) -> np.ndarray:
    """Parent linear-index map for a 26-connected Dijkstra on *binary*."""
    shape = binary.shape
    n = int(np.prod(shape))
    parent = np.full(n, -1, dtype=np.int32)
    dist = np.full(n, np.inf, dtype=np.float64)
    root_i = int(_linear_index(np.asarray([root], dtype=int), shape)[0])
    dist[root_i] = 0.0
    parent[root_i] = root_i
    heap: list[tuple[float, int]] = [(0.0, root_i)]
    z_max, y_max, x_max = shape
    binary_f = np.asarray(binary, dtype=bool)

    while heap:
        d, idx = heapq.heappop(heap)
        if d > dist[idx]:
            continue
        z, y, x = _from_linear(idx, shape)
        for dz, dy, dx in _OFFSETS_26:
            nz, ny, nx_ = z + dz, y + dy, x + dx
            if not (0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx_ < x_max):
                continue
            if not binary_f[nz, ny, nx_]:
                continue
            nidx = (nz * y_max + ny) * x_max + nx_
            step = float(cost[nz, ny, nx_])
            if not np.isfinite(step) or step <= 0.0:
                continue
            nd = d + step
            if nd < dist[nidx]:
                dist[nidx] = nd
                parent[nidx] = idx
                heapq.heappush(heap, (nd, nidx))
    return parent


def _traceback(
    parent: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    start_i = int(_linear_index(np.asarray([start], dtype=int), shape)[0])
    end_i = int(_linear_index(np.asarray([end], dtype=int), shape)[0])
    path: list[tuple[int, int, int]] = []
    idx = end_i
    seen = set()
    while idx >= 0 and idx not in seen:
        seen.add(idx)
        path.append(_from_linear(idx, shape))
        if idx == start_i:
            break
        nxt = int(parent[idx])
        if nxt == idx:
            break
        idx = nxt
    path.reverse()
    return path


def _principal_endpoints(coords: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Voxels at the ends of the component's long axis."""
    centered = coords.astype(float) - coords.astype(float).mean(axis=0)
    if coords.shape[0] < 2:
        voxel = tuple(int(v) for v in coords[0])
        return voxel, voxel
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    projection = centered @ axis
    start = tuple(int(v) for v in coords[int(np.argmin(projection))])
    end = tuple(int(v) for v in coords[int(np.argmax(projection))])
    return start, end


def _draw_path(result: np.ndarray, path: list[tuple[int, int, int]]) -> None:
    for voxel in path:
        result[voxel] = True


def _cover_around_path(
    path: list[tuple[int, int, int]],
    radius_voxels: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _draw_path(mask, path)
    iterations = max(1, int(radius_voxels))
    return binary_dilation(
        mask, structure=generate_binary_structure(3, 3), iterations=iterations
    )


def _local_principal_axis(
    component: np.ndarray,
    point: tuple[int, int, int],
    radius: float,
) -> np.ndarray | None:
    """Long axis of *component* in a ball around *point*, or None if too small."""
    r = max(2, int(np.ceil(radius)))
    z, y, x = (int(v) for v in point)
    z0, z1 = max(0, z - r), min(component.shape[0], z + r + 1)
    y0, y1 = max(0, y - r), min(component.shape[1], y + r + 1)
    x0, x1 = max(0, x - r), min(component.shape[2], x + r + 1)
    crop = component[z0:z1, y0:y1, x0:x1]
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    ball = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    coords = np.argwhere(crop & ball)
    if coords.shape[0] < 8:
        return None
    centered = coords.astype(float) - coords.astype(float).mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim != 2 or cov.shape != (3, 3):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, int(np.argmax(eigvals))]


def _path_cuts_across_lumen(
    path: list[tuple[int, int, int]],
    component: np.ndarray,
    max_edt: float,
) -> bool:
    """True when *path* crosses a tube's width instead of following an arm.

    A flattened plasma column's medial geometry is a sheet: the farthest
    high-EDT voxel from the trunk is across the lumen, and the geodesic to it
    is a short perpendicular stub. A real arm's tip has a local long axis
    aligned with that geodesic.
    """
    if len(path) < 2:
        return True
    start = np.asarray(path[0], dtype=float)
    end = np.asarray(path[-1], dtype=float)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return True
    direction /= norm
    axis = _local_principal_axis(
        component, path[-1], radius=max(3.0, 1.25 * float(max_edt))
    )
    if axis is None:
        return False
    return abs(float(np.dot(direction, axis))) < 0.5


def _trim_to_tree(
    path: list[tuple[int, int, int]],
    tree: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Keep the segment from the far end until it first meets *tree*."""
    trimmed: list[tuple[int, int, int]] = []
    for voxel in reversed(path):
        trimmed.append(voxel)
        if tree[voxel]:
            break
    trimmed.reverse()
    return trimmed


def _component_edt_ridge(component: np.ndarray, edt: np.ndarray) -> np.ndarray:
    """Centreline tree: trunk geodesic plus every arm that is not a sheet duplicate."""
    result = np.zeros(component.shape, dtype=bool)
    coords = np.argwhere(component)
    if coords.size == 0:
        return result

    local_edt = np.where(component, edt, 0.0)
    max_edt = float(local_edt.max())
    cost = np.where(component, 1.0 / (np.square(local_edt) + 1e-6), np.inf)
    root, far = _principal_endpoints(coords)
    parent = _dijkstra_parents(component, cost, root)
    main = _traceback(parent, root, far, component.shape)
    if len(main) < 2:
        centre = tuple(int(v) for v in coords[int(np.argmax(local_edt[tuple(coords.T)]))])
        result[centre] = True
        return result
    _draw_path(result, main)

    cover_r = max(1, int(round(max_edt)))
    covered = _cover_around_path(main, cover_r, component.shape)
    min_arm_voxels = max(4, int(2.0 * max_edt))
    high = component & (local_edt >= 0.4 * max_edt)

    for _ in range(12):
        candidates = high & ~covered
        if not candidates.any():
            break
        dist_to_tree = distance_transform_edt(~result)
        far_dist = float(dist_to_tree[candidates].max())
        # Remaining high-EDT voxels sitting beside the trunk are the sheet, not arms.
        if far_dist < 2.0 * max_edt:
            break
        cand_coords = np.argwhere(candidates)
        target = tuple(
            int(v)
            for v in cand_coords[int(np.argmax(dist_to_tree[tuple(cand_coords.T)]))]
        )
        branch = _trim_to_tree(
            _traceback(parent, root, target, component.shape),
            result,
        )
        if len(branch) < min_arm_voxels:
            covered[target] = True
            continue
        if _path_cuts_across_lumen(branch, component, max_edt):
            covered |= _cover_around_path(branch, cover_r, component.shape)
            continue
        _draw_path(result, branch)
        covered |= _cover_around_path(branch, cover_r, component.shape)
    return result


def skeletonize_edt_ridge(binary: np.ndarray) -> np.ndarray:
    """Centreline tree per connected component: every fat arm, not a medial sheet."""
    mask = np.asarray(binary, dtype=bool)
    result = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return result
    edt = distance_transform_edt(mask)
    labeled, n_labels = label(mask)
    for component_id in range(1, int(n_labels) + 1):
        component = labeled == component_id
        result |= _component_edt_ridge(component, edt)
    return result.astype(bool)


def _stitch_thin_to_thick(
    skeleton: np.ndarray,
    thick_object: np.ndarray,
) -> np.ndarray:
    """One link per thin skeleton fragment that meets the fat-region boundary."""
    result = skeleton.copy()
    structure = generate_binary_structure(3, 1)
    interface = binary_dilation(thick_object, structure=structure) & ~thick_object
    thin_at_cut = result & interface
    thick_skel = result & thick_object
    if not thin_at_cut.any() or not thick_skel.any():
        return result
    labeled, n_labels = label(thin_at_cut)
    thick_coords = np.argwhere(thick_skel)
    for component_id in range(1, int(n_labels) + 1):
        pts = np.argwhere(labeled == component_id)
        # Nearest interface voxel of this fragment to the ridge.
        delta = pts[:, None, :] - thick_coords[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        pair = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
        _draw_line_3d(result, pts[pair[0]], thick_coords[pair[1]])
    return result


def skeletonize_thickness_gated(
    binary: np.ndarray,
    *,
    min_radius_um: float = THICK_VESSEL_MIN_RADIUS_UM,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_mask_holes: bool = True,
) -> np.ndarray:
    """Lee on the thin catchment; an EDT-ridge tree (every arm) inside the fat catchment.

    ``min_radius_um <= 0`` is the current pipeline behaviour: Lee on the whole
    mask. Capillaries fused into the same object never enter the ridge path
    unless their own EDT peak crosses the threshold.
    """
    mask = np.asarray(binary, dtype=bool)
    if fill_mask_holes:
        mask = fill_binary_holes(mask)
    if float(min_radius_um) <= 0.0:
        return skeletonize_volume(mask).astype(bool)

    thick = thick_vessel_object_mask(
        mask, min_radius_um=float(min_radius_um), voxel_size_zyx=voxel_size_zyx
    )
    if not thick.any():
        return skeletonize_volume(mask).astype(bool)

    thin = mask & ~thick
    result = np.zeros(mask.shape, dtype=bool)
    if thin.any():
        result |= skeletonize_volume(thin).astype(bool)
    result |= skeletonize_edt_ridge(thick)
    # Do not draw chords through the fat region: leftover thin fragments at
    # the cut are nearby endpoints, which graph-building already reconnects.
    return result.astype(bool)


def needs_thick_vessel_treatment(
    binary: np.ndarray,
    *,
    min_radius_um: float = THICK_VESSEL_MIN_RADIUS_UM,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bool:
    """True when any EDT peak of the (single) object is fat enough to treat."""
    return max_inscribed_radius_um(binary, voxel_size_zyx) >= float(min_radius_um)


def characterisation_rows(
    cases: Iterable[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]],
) -> list[dict[str, float | str | bool]]:
    """Radius, volumes, and Lee braid of the fat plasma column along its long axis.

    Each case is ``(name, mask, fat_roi, spacing)``. Braid is measured on the
    fat ROI alone (the plasma column) so junction voxels with fused capillaries
    do not set the gate. Axis 2 is the trunk axis of the fixture.
    """
    rows: list[dict[str, float | str | bool]] = []
    for name, mask, fat_roi, spacing in cases:
        radius = max_inscribed_radius_um(mask, spacing)
        volume = foreground_volume_um3(mask, spacing)
        fat_volume = foreground_volume_um3(fat_roi, spacing)
        braid = lee_braid_factor(fat_roi, axis=2)
        rows.append(
            {
                "name": name,
                "max_radius_um": radius,
                "volume_um3": volume,
                "fat_volume_um3": fat_volume,
                "lee_braid_factor": braid,
                "lee_sheets": bool(braid > BRAID_FACTOR_LIMIT),
            }
        )
    return rows
