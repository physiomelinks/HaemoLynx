"""Thickness-gated skeletonisation for fat regions of a plasma-labelled mask.

Plasma-column labelling fills the lumen, and the pipeline's main input is
typically **one connected binary object**: a fat trunk fused to capillaries,
not a separate tube. Lee thinning of the fat part of that object yields a
medial *sheet* (several polylines for one vessel). The thin part is already
fine.

This module is used by ``skeletonise`` when
``use_thick_vessel_skeletonisation`` is on.

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

import gc
import logging
import time
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .skeleton import fill_binary_holes, skeletonize_volume

logger = logging.getLogger(__name__)

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


class _ForegroundIndex:
    """Map ``(z, y, x)`` to a compact foreground id without a dense volume.

    Packed keys plus ``searchsorted`` keep Dijkstra's neighbour lookup off a
    ``prod(shape)`` int32 table, which on a fat bbox the size of a stack is
    hundreds of megabytes. Tests index ``index_of[start]`` and
    ``index_of[nz, ny, nx]``.
    """

    __slots__ = ("_keys", "_n", "_nyx", "_nx")

    def __init__(self, fg_coords: np.ndarray, shape: tuple[int, int, int]):
        self._nyx = int(shape[1]) * int(shape[2])
        self._nx = int(shape[2])
        self._keys = (
            fg_coords[:, 0].astype(np.int64, copy=False) * self._nyx
            + fg_coords[:, 1].astype(np.int64, copy=False) * self._nx
            + fg_coords[:, 2].astype(np.int64, copy=False)
        )
        self._n = int(self._keys.size)

    def _pack(self, z, y, x) -> np.ndarray:
        return (
            np.asarray(z, dtype=np.int64) * self._nyx
            + np.asarray(y, dtype=np.int64) * self._nx
            + np.asarray(x, dtype=np.int64)
        )

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 3:
            raise IndexError("ForegroundIndex expects (z, y, x)")
        keys = self._pack(item[0], item[1], item[2])
        scalar = np.ndim(keys) == 0
        keys_a = np.atleast_1d(np.asarray(keys, dtype=np.int64))
        idx = np.searchsorted(self._keys, keys_a)
        valid = idx < self._n
        match = np.zeros(keys_a.shape, dtype=bool)
        if self._n:
            match[valid] = self._keys[idx[valid]] == keys_a[valid]
        out = np.full(keys_a.shape, -1, dtype=np.int32)
        out[match] = idx[match].astype(np.int32, copy=False)
        if scalar:
            return int(out[0])
        return out


def _aabb_slices(
    p0: tuple[int, int, int],
    p1: tuple[int, int, int],
    shape: tuple[int, ...],
    *,
    pad: int,
) -> tuple[tuple[slice, slice, slice], tuple[int, int, int]]:
    """Tight AABB around two voxels, padded, with no full-volume allocation."""
    slices: list[slice] = []
    origin: list[int] = []
    for a, b, dim in zip(p0, p1, shape):
        lo = max(0, min(int(a), int(b)) - int(pad))
        hi = min(int(dim), max(int(a), int(b)) + 1 + int(pad))
        slices.append(slice(lo, hi))
        origin.append(lo)
    return (slices[0], slices[1], slices[2]), (origin[0], origin[1], origin[2])


def _line_voxels(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """26-connected straight line from *start* to *end* (Chebyshev stepping)."""
    sz, sy, sx = (int(v) for v in start)
    ez, ey, ex = (int(v) for v in end)
    n = max(abs(ez - sz), abs(ey - sy), abs(ex - sx))
    if n <= 0:
        return [(sz, sy, sx)]
    voxels: list[tuple[int, int, int]] = []
    prev: tuple[int, int, int] | None = None
    for i in range(n + 1):
        t = i / n
        voxel = (
            int(round(sz + t * (ez - sz))),
            int(round(sy + t * (ey - sy))),
            int(round(sx + t * (ex - sx))),
        )
        if voxel != prev:
            voxels.append(voxel)
            prev = voxel
    return voxels


def _touches_tree(tree: np.ndarray, voxel: tuple[int, int, int]) -> bool:
    """True if *voxel* is on *tree* or 26-adjacent to it."""
    z, y, x = (int(v) for v in voxel)
    z0, z1 = max(0, z - 1), min(tree.shape[0], z + 2)
    y0, y1 = max(0, y - 1), min(tree.shape[1], y + 2)
    x0, x1 = max(0, x - 1), min(tree.shape[2], x + 2)
    return bool(tree[z0:z1, y0:y1, x0:x1].any())


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
    wall_absorption_um: float | None = None,
) -> np.ndarray:
    """Fat-region voxels of a (possibly single) connected plasma-labelled mask.

    1. Core: inscribed radius >= *min_radius_um*.
    2. Body: geodesic reconstruction of that core through voxels fatter than
       half *min_radius_um*, so a flattened plasma column's interior is
       included without walking down fused capillaries (those stay below
       that gate). Always half *min_radius_um* -- this is about correctly
       tracing the fat trunk's own (possibly irregular) shape, not about how
       much of a neighbouring vessel gets absorbed, so *wall_absorption_um*
       does not affect it.
    3. Wall: Euclidean ball of *wall_absorption_um* around the body, so the
       plasma column's surface is not left for Lee to mesh, while capillaries
       farther than that from the body stay on the Lee path.

    *wall_absorption_um* also swallows the first *wall_absorption_um* of any
    real vessel fused directly onto the trunk's surface -- the fat region has
    no way to tell a small vessel's own base from the trunk's own surface
    roughness, so a larger value costs more of every fused vessel's near-wall
    length. ``None`` derives it as half of *min_radius_um*, the previous
    fixed behaviour (both steps 2 and 3 then use the same value, as before
    this parameter existed).
    """
    mask = np.asarray(binary, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    if min_radius_um <= 0.0 or not mask.any():
        return out

    bbox = _foreground_bbox(mask, pad=1)
    if bbox is None:
        return out
    crop = mask[bbox]
    logger.info(
        "thick_vessel_object_mask: cropped to %s (%d voxels, %d foreground) "
        "from input shape %s",
        crop.shape,
        int(crop.size),
        int(crop.sum()),
        mask.shape,
    )
    t0 = time.perf_counter()
    radius_map = inscribed_radius_map(crop, voxel_size_zyx)
    thick_core = crop & (radius_map >= float(min_radius_um))
    logger.info(
        "thick_vessel_object_mask: inscribed radius map took %.2fs (%d core voxels)",
        time.perf_counter() - t0,
        int(thick_core.sum()),
    )
    if not thick_core.any():
        return out

    propagation_gate = 0.5 * float(min_radius_um)
    wall_radius = (
        propagation_gate
        if wall_absorption_um is None
        else max(0.0, float(wall_absorption_um))
    )
    allowed = crop & (radius_map >= propagation_gate)
    # radius_map is float64 over the whole crop -- as large as any array in
    # this function -- and everything from here on only needs the two
    # boolean masks already derived from it.
    del radius_map
    gc.collect()
    t1 = time.perf_counter()
    body = binary_propagation(thick_core, mask=allowed)
    logger.info(
        "thick_vessel_object_mask: geodesic body propagation took %.2fs (%d body voxels)",
        time.perf_counter() - t1,
        int(body.sum()),
    )
    t2 = time.perf_counter()
    dist_to_body = distance_transform_edt(
        ~body, sampling=tuple(float(v) for v in voxel_size_zyx)
    )
    out[bbox] = crop & (dist_to_body <= wall_radius)
    logger.info(
        "thick_vessel_object_mask: wall distance transform took %.2fs (%d fat voxels total)",
        time.perf_counter() - t2,
        int(out.sum()),
    )
    return out


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


def _foreground_bbox(mask: np.ndarray, *, pad: int = 1) -> tuple[slice, slice, slice] | None:
    """Tight slices around foreground, padded, or None if *mask* is empty."""
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.ndim != 3:
        coords = np.argwhere(mask_b)
        if coords.size == 0:
            return None
        lo = np.maximum(coords.min(axis=0) - int(pad), 0)
        hi = np.minimum(coords.max(axis=0) + 1 + int(pad), mask_b.shape)
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    z_any = mask_b.any(axis=(1, 2))
    if not z_any.any():
        return None
    y_any = mask_b.any(axis=(0, 2))
    x_any = mask_b.any(axis=(0, 1))
    z0 = int(np.argmax(z_any))
    z1 = int(z_any.size - np.argmax(z_any[::-1]))
    y0 = int(np.argmax(y_any))
    y1 = int(y_any.size - np.argmax(y_any[::-1]))
    x0 = int(np.argmax(x_any))
    x1 = int(x_any.size - np.argmax(x_any[::-1]))
    p = int(pad)
    return (
        slice(max(0, z0 - p), min(mask_b.shape[0], z1 + p)),
        slice(max(0, y0 - p), min(mask_b.shape[1], y1 + p)),
        slice(max(0, x0 - p), min(mask_b.shape[2], x1 + p)),
    )


def _expand_slices(
    slc: tuple[slice, ...],
    shape: tuple[int, ...],
    *,
    pad: int = 1,
) -> tuple[slice, ...]:
    return tuple(
        slice(max(0, int(s.start) - int(pad)), min(int(dim), int(s.stop) + int(pad)))
        for s, dim in zip(slc, shape)
    )


def _skeletonize_foreground(mask: np.ndarray) -> np.ndarray:
    """Lee thinning of each connected component, cropped so empty space is not walked."""
    mask_b = np.asarray(mask, dtype=bool)
    result = np.zeros(mask_b.shape, dtype=bool)
    if not mask_b.any():
        return result
    structure = generate_binary_structure(mask_b.ndim, mask_b.ndim)
    labeled, n_labels = label(mask_b, structure=structure)
    if n_labels == 0:
        return result
    for component_id, slc in enumerate(find_objects(labeled), start=1):
        if slc is None:
            continue
        padded = _expand_slices(slc, mask_b.shape, pad=1)
        result[padded] |= skeletonize_volume(labeled[padded] == component_id)
    return result


def _dijkstra_parents(
    binary: np.ndarray,
    cost: np.ndarray,
    root: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, _ForegroundIndex] | None:
    """Sparse 26-connected Dijkstra on foreground voxels of *binary*.

    Returns ``(parent, fg_coords, index_of)`` where *parent* and *fg_coords*
    are length-N (N = foreground count) and *index_of* maps a voxel in the
    array to that compact id, or -1. ``None`` if *root* is not foreground.

    Arrays are sized to the foreground, not ``prod(shape)``: a microscopy
    stack must not allocate a float64 per background voxel. Neighbour lookup
    uses packed ``z,y,x`` keys and ``searchsorted`` rather than a dense
    ``index_of`` volume. The walk itself is SciPy's C Dijkstra on a
    26-neighbour CSR graph of those voxels.
    """
    binary_f = np.asarray(binary, dtype=bool)
    fg_coords = np.argwhere(binary_f)
    n_fg = int(fg_coords.shape[0])
    if n_fg == 0:
        return None
    index_of = _ForegroundIndex(fg_coords, binary_f.shape)
    root_i = int(index_of[root])
    if root_i < 0:
        return None

    zyx = fg_coords.astype(np.intp, copy=False)
    z_max, y_max, x_max = binary_f.shape
    cost_a = np.asarray(cost, dtype=np.float64)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    for dz, dy, dx in _OFFSETS_26:
        nz = zyx[:, 0] + int(dz)
        ny = zyx[:, 1] + int(dy)
        nx_ = zyx[:, 2] + int(dx)
        in_bounds = (
            (nz >= 0)
            & (nz < z_max)
            & (ny >= 0)
            & (ny < y_max)
            & (nx_ >= 0)
            & (nx_ < x_max)
        )
        if not np.any(in_bounds):
            continue
        src = np.flatnonzero(in_bounds)
        nidx = index_of[nz[in_bounds], ny[in_bounds], nx_[in_bounds]]
        connected = nidx >= 0
        if not np.any(connected):
            continue
        nbr_z = nz[in_bounds][connected]
        nbr_y = ny[in_bounds][connected]
        nbr_x = nx_[in_bounds][connected]
        weights = cost_a[nbr_z, nbr_y, nbr_x]
        finite = np.isfinite(weights) & (weights > 0.0)
        if not np.any(finite):
            continue
        row_chunks.append(src[connected][finite].astype(np.int32, copy=False))
        col_chunks.append(nidx[connected][finite].astype(np.int32, copy=False))
        data_chunks.append(weights[finite])

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        weights_all = np.concatenate(data_chunks)
        graph = csr_matrix((weights_all, (rows, cols)), shape=(n_fg, n_fg))
    else:
        graph = csr_matrix((n_fg, n_fg))
    _dist, predecessors = dijkstra(
        graph,
        directed=True,
        indices=root_i,
        return_predecessors=True,
    )
    parent = np.asarray(predecessors, dtype=np.int32)
    parent[root_i] = root_i
    parent[parent < 0] = -1
    return parent, fg_coords, index_of


def _traceback(
    parent: np.ndarray,
    fg_coords: np.ndarray,
    index_of: _ForegroundIndex,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    start_i = int(index_of[start])
    end_i = int(index_of[end])
    if start_i < 0 or end_i < 0:
        return []
    path: list[tuple[int, int, int]] = []
    idx = end_i
    seen: set[int] = set()
    while idx >= 0 and idx not in seen:
        seen.add(idx)
        path.append(tuple(int(v) for v in fg_coords[idx]))
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
    if not path:
        return mask
    radius = max(1, int(radius_voxels))
    pts = np.asarray(path, dtype=int)
    lo = np.maximum(pts.min(axis=0) - radius, 0)
    hi = np.minimum(pts.max(axis=0) + radius + 1, shape)
    slc = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = np.zeros(tuple(int(b - a) for a, b in zip(lo, hi)), dtype=bool)
    local[tuple((pts - lo).T)] = True
    mask[slc] = distance_transform_edt(~local) <= radius
    return mask


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
    """Keep the segment from the far end until it first meets *tree*.

    Meeting is 26-adjacency, not only exact overlap: a branch that runs
    beside the existing tree and later joins it would close a cycle.
    """
    trimmed: list[tuple[int, int, int]] = []
    for voxel in reversed(path):
        trimmed.append(voxel)
        if _touches_tree(tree, voxel):
            break
    trimmed.reverse()
    return trimmed


def _component_edt_ridge_on_crop(component: np.ndarray, edt: np.ndarray) -> np.ndarray:
    """Centreline tree on an already-cropped fat component."""
    result = np.zeros(component.shape, dtype=bool)
    coords = np.argwhere(component)
    if coords.size == 0:
        return result

    local_edt = np.where(component, edt, 0.0)
    max_edt = float(local_edt.max())
    cost = np.where(component, 1.0 / (np.square(local_edt) + 1e-6), np.inf)
    root, far = _principal_endpoints(coords)
    walked = _dijkstra_parents(component, cost, root)
    if walked is None:
        centre = tuple(int(v) for v in coords[int(np.argmax(local_edt[tuple(coords.T)]))])
        result[centre] = True
        return result
    parent, fg_coords, index_of = walked
    main = _traceback(parent, fg_coords, index_of, root, far)
    if len(main) < 2:
        centre = tuple(int(v) for v in coords[int(np.argmax(local_edt[tuple(coords.T)]))])
        result[centre] = True
        return result
    _draw_path(result, main)

    cover_r = max(1, int(round(max_edt)))
    covered = _cover_around_path(main, cover_r, component.shape)
    min_arm_voxels = max(4, int(2.0 * max_edt))
    high = component & (local_edt >= 0.4 * max_edt)
    # Every voxel this loop ever adds to `result` is exactly one of `main`
    # or an accepted `branch` -- both already known as coordinate lists, so
    # the tree can grow by appending them instead of re-deriving the whole
    # (up to component-sized) set via np.argwhere(result) on every accepted
    # arm. On a real multi-million-voxel fat component with many arms, that
    # full-array rescan per acceptance was most of this function's cost.
    tree_coords = np.asarray(main, dtype=np.intp)
    tree_kdt = cKDTree(tree_coords.astype(np.float64, copy=False))

    # A real network's single connected fat catchment can legitimately have
    # far more than a dozen arms (a whole fused sub-network, not one trunk);
    # 12 was an arbitrary safety bound, not a correctness threshold -- the
    # real stopping conditions are the two `break`s below. Each iteration
    # was already made to clear a whole rejected branch's neighbourhood
    # rather than one voxel, so raising this costs one KD-tree query per
    # additional *accepted* arm, not per candidate voxel.
    max_arms = 500
    for _ in range(max_arms):
        candidates = high & ~covered
        if not candidates.any():
            break
        cand_coords = np.argwhere(candidates)
        dists, _ = tree_kdt.query(cand_coords.astype(np.float64, copy=False), k=1)
        far_dist = float(np.max(dists))
        # Remaining high-EDT voxels sitting beside the trunk are the sheet, not arms.
        if far_dist < 2.0 * max_edt:
            break
        target = tuple(int(v) for v in cand_coords[int(np.argmax(dists))])
        branch = _trim_to_tree(
            _traceback(parent, fg_coords, index_of, root, target),
            result,
        )
        if len(branch) < min_arm_voxels:
            # Cover the rejected branch's own neighbourhood so the same
            # short stub is not re-traced next iteration -- but only a
            # small fixed margin, not the full cover_r (the TRUNK's own
            # radius, which can be 10-20+ voxels for a wide vessel). A
            # too-short stub and a separate, genuinely long-enough arm can
            # sit within cover_r of each other; covering that whole
            # neighbourhood pre-emptively removed the real arm's
            # candidate voxels before they were ever tried, silently
            # dropping it from the centreline tree.
            covered |= _cover_around_path(branch, min(2, cover_r), component.shape)
            continue
        if _touches_tree(result, branch[-1]):
            # Tip already on the tree: this geodesic would close a loop.
            covered[target] = True
            continue
        if _path_cuts_across_lumen(branch, component, max_edt):
            covered |= _cover_around_path(branch, cover_r, component.shape)
            continue
        _draw_path(result, branch)
        covered |= _cover_around_path(branch, cover_r, component.shape)
        tree_coords = np.concatenate(
            [tree_coords, np.asarray(branch, dtype=np.intp)], axis=0
        )
        tree_kdt = cKDTree(tree_coords.astype(np.float64, copy=False))
    else:
        # Loop ran out of iterations without a break, i.e. without ever
        # deciding "no arms left" or "what's left is the sheet, not an arm" --
        # real candidate arms may still remain, silently dropped from the tree.
        remaining = high & ~covered
        if remaining.any():
            logger.warning(
                "Fat-vessel centreline hit the %d-arm cap with %d high-EDT "
                "voxels still uncovered; some arms may be missing from the "
                "reconstructed centreline.",
                max_arms,
                int(remaining.sum()),
            )
    return result


def skeletonize_edt_ridge(binary: np.ndarray) -> np.ndarray:
    """Centreline tree per connected component: every fat arm, not a medial sheet."""
    mask = np.asarray(binary, dtype=bool)
    result = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return result
    bbox = _foreground_bbox(mask, pad=1)
    if bbox is None:
        return result
    crop = mask[bbox]
    edt = distance_transform_edt(crop)
    # 26-connected, matching every other label() call in this module. The
    # default (6-connected, face-only) can split a solid fat blob that is
    # only diagonally connected at a jagged mask boundary into two
    # "components", each building its own independent centreline tree with
    # no cross-component tree-adjacency check -- the two trees can then end
    # up 26-adjacent at more than one voxel near the split, producing a
    # genuine cycle in the unioned result once graph-building reads it back.
    labeled, n_labels = label(crop, structure=generate_binary_structure(3, 3))
    if n_labels == 0:
        return result
    logger.info(
        "skeletonize_edt_ridge: %d fat component(s) in a %s crop (%d voxels)",
        int(n_labels),
        crop.shape,
        int(crop.sum()),
    )
    crop_result = np.zeros(crop.shape, dtype=bool)
    for component_id, slc in enumerate(find_objects(labeled), start=1):
        if slc is None:
            continue
        component_mask = labeled[slc] == component_id
        t_component = time.perf_counter()
        crop_result[slc] |= _component_edt_ridge_on_crop(component_mask, edt[slc])
        if n_labels > 1 or int(component_mask.sum()) > 10_000:
            logger.info(
                "skeletonize_edt_ridge: component %d/%d (%d voxels) took %.2fs",
                component_id,
                int(n_labels),
                int(component_mask.sum()),
                time.perf_counter() - t_component,
            )
    result[bbox] = crop_result
    return result.astype(bool)


def _shift_path(
    path: list[tuple[int, int, int]],
    origin: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    oz, oy, ox = origin
    return [(int(z) + oz, int(y) + oy, int(x) + ox) for z, y, x in path]


def _geodesic_on_crop(
    crop: np.ndarray,
    local_start: tuple[int, int, int],
    local_end: tuple[int, int, int],
    *,
    precomputed_cost: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Inverted-EDT geodesic in a cropped boolean mask, or empty if unreachable.

    *precomputed_cost* skips the EDT (the expensive part for a large crop)
    when the caller already has one for this exact *crop* -- see
    :func:`_path_through_mask`'s *fallback_cost_fn*, cached per physically
    connected structure across every arm in it rather than recomputed once
    per arm.
    """
    if not crop[local_start] or not crop[local_end]:
        return []
    if precomputed_cost is not None:
        cost = precomputed_cost
    else:
        edt = distance_transform_edt(crop)
        cost = np.where(crop, 1.0 / (np.square(edt) + 1e-6), np.inf)
    walked = _dijkstra_parents(crop, cost, local_start)
    if walked is None:
        return []
    parent, fg_coords, index_of = walked
    path = _traceback(parent, fg_coords, index_of, local_start, local_end)
    if len(path) < 2:
        return []
    return path


def _path_through_mask(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    allowed: np.ndarray,
    *,
    fallback_cost_fn: Callable[[], np.ndarray] | None = None,
) -> list[tuple[int, int, int]]:
    """Geodesic in *allowed* from *start* to *end*, falling back to a straight line.

    Prefers a straight line when every voxel is in *allowed*, then a dilated-line
    corridor at growing radii, then Dijkstra on the whole of *allowed*. Does not
    allocate a full-stack seeds array just to bbox those points.

    *fallback_cost_fn*, called only if every corridor attempt fails, returns
    the Dijkstra cost array for the whole of *allowed* (same shape) -- the
    caller's chance to memoize one EDT across every arm that shares this
    *allowed* instead of paying for it again per arm, without paying for it
    at all when (as for most arms) a corridor attempt already succeeds.
    """
    start = tuple(int(v) for v in start)
    end = tuple(int(v) for v in end)
    allowed_b = np.asarray(allowed, dtype=bool)
    if start == end:
        return [start]
    if not allowed_b[start] or not allowed_b[end]:
        return [start, end]
    line = _line_voxels(start, end)
    if all(allowed_b[p] for p in line):
        return line

    struct26 = generate_binary_structure(3, 3)
    # Radii stay cheap because the corridor's cost scales with the box
    # around (start, end) plus this padding, not with the whole of
    # `allowed` -- wider radii here are what let a real winding path (round
    # a trunk, past a neck) resolve without ever reaching the full-`allowed`
    # fallback below, which is by far the most expensive step available.
    for radius in (2, 5, 15, 40):
        slc, origin = _aabb_slices(start, end, allowed_b.shape, pad=int(radius) + 1)
        crop_allowed = allowed_b[slc]
        local_start = tuple(int(s - o) for s, o in zip(start, origin))
        local_end = tuple(int(e - o) for e, o in zip(end, origin))
        painted = np.zeros(crop_allowed.shape, dtype=bool)
        cz, cy, cx = painted.shape
        for z, y, x in _line_voxels(local_start, local_end):
            if 0 <= z < cz and 0 <= y < cy and 0 <= x < cx:
                painted[z, y, x] = True
        corridor = (
            binary_dilation(painted, structure=struct26, iterations=int(radius))
            & crop_allowed
        )
        corridor[local_start] = True
        corridor[local_end] = True
        local_path = _geodesic_on_crop(corridor, local_start, local_end)
        if len(local_path) >= 2:
            return _shift_path(local_path, origin)

    slc, origin = _aabb_slices(start, end, allowed_b.shape, pad=2)
    local_path = _geodesic_on_crop(
        allowed_b[slc],
        tuple(int(s - o) for s, o in zip(start, origin)),
        tuple(int(e - o) for e, o in zip(end, origin)),
    )
    if len(local_path) >= 2:
        return _shift_path(local_path, origin)

    # Every attempt above stayed inside a box tight around start/end. A real
    # geodesic can need to leave that box (wind around the trunk, detour past
    # a neck) even though start and end are both foreground voxels of one
    # connected `allowed`. Without this, the caller silently drew no bridge
    # at all: it marks only start/end (already True) and calls the arm
    # joined because `end` sits on the ridge, leaving the arm's own voxels
    # 26-disconnected from everything else -- an isolated fragment that
    # vanishes from the graph the moment anything downstream drops small
    # disconnected components. Fall back to Dijkstra on the whole mask: no
    # crop, so no box to be too tight, at the cost of one full-mask walk.
    full_path = _geodesic_on_crop(
        allowed_b,
        start,
        end,
        precomputed_cost=fallback_cost_fn() if fallback_cost_fn is not None else None,
    )
    if len(full_path) >= 2:
        return full_path
    logger.warning(
        "Could not join a thin vessel arm to the fat ridge: no path from %s "
        "to %s within the connected mask. This arm stays a disconnected "
        "fragment and may be dropped downstream.",
        start,
        end,
    )
    return [start, end]


def _join_thin_arms_to_fat_ridge(
    skeleton: np.ndarray,
    thick: np.ndarray,
    allowed: np.ndarray,
    *,
    min_arm_extent_voxels: float = 4.0,
) -> np.ndarray:
    """Connect each thin-vessel skeleton that leaves the fat wall to the fat ridge.

    Thin skeleton CCs that never extend beyond the wall (Lee flakes / wrapping
    mesh on the fat surface) are dropped. Joining those would draw a sheet of
    chords. A fused capillary does extend, and its Lee polyline must meet the
    ridge.
    """
    result = np.asarray(skeleton, dtype=bool).copy()
    thick_b = np.asarray(thick, dtype=bool)
    allowed_b = np.asarray(allowed, dtype=bool)
    thin_skel = result & ~thick_b
    fat_skel = result & thick_b
    if not fat_skel.any():
        return result
    if not thin_skel.any():
        return fat_skel
    structure = generate_binary_structure(3, 3)
    labeled, n_labels = label(thin_skel, structure=structure)
    if n_labels == 0:
        return fat_skel

    thin_coords = np.argwhere(thin_skel)
    labels_at = labeled[tuple(thin_coords.T)]
    # Per physically connected structure (thin + thick together), not one
    # EDT over their shared bounding box: real fat/thin material can span
    # most of a large stack, so that box is close to the whole image, and a
    # dense EDT that size is what actually ran the process out of memory (a
    # 287x512x512 stack: "Unable to allocate 1.35 GiB for an array with
    # shape (180948686,)"). Each connected roi component is a sliver of
    # that by comparison -- same reasoning as the per-arm join scoping
    # below, applied to this earlier length-filter measurement too.
    roi = thin_skel | thick_b
    roi_labeled, _n_roi = label(roi, structure=structure)
    roi_objects = find_objects(roi_labeled)
    roi_ids_at_thin = roi_labeled[tuple(thin_coords.T)]
    dists = np.full(len(thin_coords), np.inf, dtype=np.float64)
    for roi_id in np.unique(roi_ids_at_thin):
        if roi_id == 0:
            continue
        slc = roi_objects[int(roi_id) - 1]
        if slc is None:
            continue
        here = roi_ids_at_thin == roi_id
        if not thick_b[slc].any():
            # No fat material anywhere in this thin arm's own connected
            # structure: it cannot be measured against a wall it has none
            # of. Leave it at +inf, so it clears the length filter and is
            # kept as-is -- the per-arm join loop below reaches the same
            # "nothing to join to" conclusion for exactly this case.
            continue
        origin = np.array([int(s.start) for s in slc], dtype=int)
        dist_crop = distance_transform_edt(~thick_b[slc])
        local = thin_coords[here] - origin
        dists[here] = dist_crop[tuple(local.T)]

    max_dist = np.zeros(int(n_labels) + 1, dtype=np.float64)
    np.maximum.at(max_dist, labels_at, dists)
    arm_ids = np.flatnonzero(max_dist >= float(min_arm_extent_voxels))
    arm_ids = arm_ids[arm_ids > 0]
    objects = find_objects(labeled)

    keep = fat_skel.copy()
    for component_id in arm_ids:
        slc = objects[int(component_id) - 1]
        if slc is None:
            continue
        keep[slc] |= labeled[slc] == int(component_id)
    result = keep
    if arm_ids.size == 0:
        return result

    fat_coords = np.argwhere(result & thick_b)
    if fat_coords.size == 0:
        return result
    # Which physically connected structure (of the real mask, not just the
    # fat/thin split) each voxel belongs to. A thin arm's Euclidean-nearest
    # fat voxel can sit in a different, merely-nearby vessel network that it
    # is not actually connected to at all -- no path can ever exist between
    # them, full-mask search or not. Scoping the nearest-neighbour search to
    # fat voxels sharing the arm's own component rules that out up front.
    allowed_components, _n_allowed = label(allowed_b, structure=structure)
    component_objects = find_objects(allowed_components)
    fat_component_labels = allowed_components[tuple(fat_coords.T)]
    fat_kdt = cKDTree(fat_coords.astype(np.float64, copy=False))
    fat_now = result & thick_b

    # Memoized per physically-connected structure: the fallback's EDT
    # depends only on that structure's shape, not on which arm needed it,
    # so a component with many arms needing the fallback (a real fused
    # sub-network, not a handful of capillaries) pays for it once instead
    # of once per arm.
    cost_cache: dict[int, np.ndarray] = {}

    def _fallback_cost_for(label_key: int, local_mask: np.ndarray) -> np.ndarray:
        cached = cost_cache.get(label_key)
        if cached is not None:
            return cached
        local_edt = distance_transform_edt(local_mask)
        cost = np.where(local_mask, 1.0 / (np.square(local_edt) + 1e-6), np.inf)
        cost_cache[label_key] = cost
        return cost

    logger.info(
        "_join_thin_arms_to_fat_ridge: joining up to %d thin-arm components to the "
        "fat ridge",
        int(arm_ids.size),
    )
    join_step_start = time.perf_counter()
    last_log_time = join_step_start
    for arm_index, component_id in enumerate(arm_ids, start=1):
        slc = objects[int(component_id) - 1]
        if slc is None:
            continue
        padded = _expand_slices(slc, labeled.shape, pad=1)
        component = labeled[padded] == int(component_id)
        if (binary_dilation(component, structure=structure) & fat_now[padded]).any():
            continue
        pts_local = np.argwhere(component)
        if pts_local.size == 0:
            continue
        origin_p = np.array([int(s.start) for s in padded], dtype=int)
        pts = pts_local + origin_p
        arm_component_label = int(allowed_components[tuple(pts[0])])
        same_component = fat_component_labels == arm_component_label
        if not np.any(same_component):
            # No fat ridge material anywhere in this arm's own physically
            # connected structure: it is either a fully self-contained small
            # network with no fat trunk of its own (already complete, needs
            # no joining) or genuinely isolated. Either way there is nothing
            # to join it to -- keep it as already-drawn Lee output, and do
            # not manufacture a connection to an unrelated nearby network.
            continue
        scoped_fat_coords = fat_coords[same_component]
        scoped_kdt = (
            fat_kdt
            if same_component.all()
            else cKDTree(scoped_fat_coords.astype(np.float64, copy=False))
        )
        _d, nn = scoped_kdt.query(pts.astype(np.float64, copy=False), k=1)
        nearest = int(np.argmin(np.atleast_1d(_d)))
        start = tuple(int(v) for v in pts[nearest])
        end = tuple(int(v) for v in scoped_fat_coords[int(np.atleast_1d(nn)[nearest])])
        # Search only this arm's own physically connected structure, not the
        # whole image: _path_through_mask's own local searches are already
        # small, but its last-resort fallback runs a full EDT + Dijkstra over
        # everything passed to it, and one connected structure is a sliver of
        # a real multi-vessel stack. Without this, that fallback firing even
        # a handful of times on a real image is what actually froze the run,
        # not the join logic itself.
        component_slc = component_objects[arm_component_label - 1]
        component_origin = np.array([int(s.start) for s in component_slc], dtype=int)
        local_allowed = allowed_b[component_slc]
        local_start = tuple(int(v) for v in (np.array(start) - component_origin))
        local_end = tuple(int(v) for v in (np.array(end) - component_origin))
        local_path = _path_through_mask(
            local_start,
            local_end,
            local_allowed,
            fallback_cost_fn=lambda: _fallback_cost_for(
                arm_component_label, local_allowed
            ),
        )
        path = [
            tuple(int(v) for v in (np.array(voxel) + component_origin))
            for voxel in local_path
        ]
        # fat_now must stay exactly what it was before this arm while the
        # walk below tests against it (the walk stops on first touching the
        # *pre-existing* ridge, not on becoming its own touch); voxels this
        # arm adds are folded in afterwards instead.
        drawn: list[tuple[int, int, int]] = []
        for voxel in path:
            if not allowed_b[voxel]:
                continue
            result[voxel] = True
            drawn.append(voxel)
            if _touches_tree(fat_now, voxel):
                break
        # Requery against what's now connected, including this join's own
        # bridge and any earlier arm: otherwise every later arm keeps
        # targeting the pre-join ridge alone, picking a farther "nearest"
        # point than the one this loop just made available. Every voxel
        # this arm could have added to the fat set is already known from
        # `drawn` -- no need to recompute `result & thick_b` or re-argwhere
        # it (both full-image scans) to find out what changed.
        newly_fat = [voxel for voxel in drawn if thick_b[voxel]]
        if newly_fat:
            new_coords = np.asarray(newly_fat, dtype=np.intp)
            fat_now[tuple(new_coords.T)] = True
            fat_coords = np.concatenate([fat_coords, new_coords], axis=0)
            fat_component_labels = np.concatenate(
                [fat_component_labels, allowed_components[tuple(new_coords.T)]]
            )
            fat_kdt = cKDTree(fat_coords.astype(np.float64, copy=False))
        if arm_index % 200 == 0 or time.perf_counter() - last_log_time > 30.0:
            now = time.perf_counter()
            logger.info(
                "_join_thin_arms_to_fat_ridge: %d/%d arm components processed "
                "(%.2fs since last checkpoint, %.2fs total)",
                arm_index,
                int(arm_ids.size),
                now - last_log_time,
                now - join_step_start,
            )
            last_log_time = now
    return result


def skeletonize_thickness_gated(
    binary: np.ndarray,
    *,
    min_radius_um: float = THICK_VESSEL_MIN_RADIUS_UM,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_mask_holes: bool = True,
    wall_absorption_um: float | None = None,
    flake_filter_um: float | None = None,
) -> np.ndarray:
    """Lee on the thin catchment; an EDT-ridge tree (every arm) inside the fat catchment.

    ``min_radius_um <= 0`` is the current pipeline behaviour: Lee on the whole
    mask. Capillaries fused into the same object never enter the ridge path
    unless their own EDT peak crosses the threshold.

    *wall_absorption_um* is how far around the fat body's core the wall
    absorbs surface roughness (and, unavoidably, the first stretch of any
    real vessel fused directly onto it) into the catchment; see
    :func:`thick_vessel_object_mask`. ``None`` derives it as half of
    *min_radius_um*.

    *flake_filter_um* is how far beyond the fat wall a thin-vessel skeleton
    fragment must reach to be kept rather than dropped as a Lee-thinning
    flake of the wall's own surface; see :func:`_join_thin_arms_to_fat_ridge`.
    ``None`` derives it as ``max(4 voxels, 0.75 * min_radius_um)``, the
    previous fixed behaviour. Together, a vessel fused to a fat trunk needs
    to reach *wall_absorption_um* + *flake_filter_um* beyond its attachment
    point to survive at all -- shortening either recovers shorter real
    vessels at the cost of letting more wall-wrap flakes through.
    """
    mask = np.asarray(binary, dtype=bool)
    if fill_mask_holes:
        bbox = _foreground_bbox(mask, pad=0)
        if bbox is not None:
            filled = fill_binary_holes(mask[bbox])
            mask = mask.copy()
            mask[bbox] = filled
    if float(min_radius_um) <= 0.0:
        return skeletonize_volume(mask).astype(bool)

    logger.info(
        "skeletonize_thickness_gated: computing fat catchment (input shape %s, "
        "%d foreground voxels)",
        mask.shape,
        int(mask.sum()),
    )
    t0 = time.perf_counter()
    thick = thick_vessel_object_mask(
        mask,
        min_radius_um=float(min_radius_um),
        voxel_size_zyx=voxel_size_zyx,
        wall_absorption_um=wall_absorption_um,
    )
    t_catchment = time.perf_counter() - t0
    if not thick.any():
        return skeletonize_volume(mask).astype(bool)
    # thick_vessel_object_mask's own large locals (the radius map, the
    # geodesic body, the wall distance transform -- each up to the size of
    # the input volume) are unreachable now that it has returned, but a big
    # freed block does not always get handed back to the OS the moment its
    # refcount drops; collecting explicitly before the next big allocation
    # (Lee-thinning, then the ridge tree, then joining) gives it the best
    # chance to.
    gc.collect()

    thin = mask & ~thick
    spacing = min(float(v) for v in voxel_size_zyx)
    if flake_filter_um is None:
        min_arm_extent = max(4.0, 0.75 * float(min_radius_um) / max(spacing, 1e-6))
    else:
        min_arm_extent = max(0.0, float(flake_filter_um) / max(spacing, 1e-6))
    result = np.zeros(mask.shape, dtype=bool)
    logger.info(
        "skeletonize_thickness_gated: fat catchment done in %.2fs (%d fat, %d thin "
        "voxels); Lee-thinning the thin catchment",
        t_catchment,
        int(thick.sum()),
        int(thin.sum()),
    )
    t1 = time.perf_counter()
    if thin.any():
        # Do not Lee the leftover fat-wall shell: those CCs never extend away
        # from thick, and Lee of that wrap is the looped mesh beside the ridge.
        bbox = _foreground_bbox(thin | thick, pad=1)
        if bbox is None:
            result |= _skeletonize_foreground(thin)
        else:
            dist_crop = distance_transform_edt(~thick[bbox])
            lee_crop = thin[bbox] & (dist_crop >= float(min_arm_extent))
            if lee_crop.any():
                result[bbox] |= _skeletonize_foreground(lee_crop)
    t_lee = time.perf_counter() - t1
    gc.collect()
    logger.info(
        "skeletonize_thickness_gated: Lee-thinning done in %.2fs; building the "
        "fat-catchment centreline tree",
        t_lee,
    )
    t2 = time.perf_counter()
    result |= skeletonize_edt_ridge(thick)
    t_ridge = time.perf_counter() - t2
    gc.collect()
    logger.info(
        "skeletonize_thickness_gated: centreline tree done in %.2fs; joining thin "
        "arms to the fat ridge",
        t_ridge,
    )
    t3 = time.perf_counter()
    result = _join_thin_arms_to_fat_ridge(
        result,
        thick,
        mask,
        min_arm_extent_voxels=min_arm_extent,
    ).astype(bool)
    t_join = time.perf_counter() - t3
    logger.info(
        "Thickness-gated skeletonisation: shape=%s fat=%d voxels thin=%d voxels "
        "catchment=%.3fs lee=%.3fs ridge=%.3fs join=%.3fs",
        mask.shape,
        int(thick.sum()),
        int(thin.sum()),
        t_catchment,
        t_lee,
        t_ridge,
        t_join,
    )
    return result


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
