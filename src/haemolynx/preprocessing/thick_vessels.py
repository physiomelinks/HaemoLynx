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
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
    maximum_filter,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import blockwise_edt, memmap_support, slab_labels
from .skeleton import fill_binary_holes, skeletonize_by_component, skeletonize_volume

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


def _relative_spacing(voxel_size_zyx) -> np.ndarray:
    """Per-axis spacing in voxels of the finest axis: 1 there, more along a
    coarser axis; all ones without a voxel size.

    The centreline tree below works in these units: its distances, radii and
    thresholds are the same numbers as before on cube voxels, and physical on
    any other -- where, counted in voxels, a round trunk read as a flat
    ellipse and its ridge as a band (the medial sheet this tree exists to
    avoid).
    """
    if voxel_size_zyx is None:
        return np.ones(3)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    return spacing / float(spacing.min())


def _step_factor(offset, relative_spacing: np.ndarray) -> float:
    """How much longer a voxel step along *offset* is, physically, than the
    same step on cube voxels: 1 on cube voxels, and a step along a coarse
    axis its relative spacing."""
    step = np.asarray(offset, dtype=float)
    return float(np.linalg.norm(step * relative_spacing) / np.linalg.norm(step))


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

    @property
    def keys(self) -> np.ndarray:
        """The foreground's packed C-order keys, sorted."""
        return self._keys


def _box_members(
    keys: np.ndarray, shape: tuple[int, int, int], box: tuple[slice, slice, slice]
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, ids)``: where the voxels of the sorted C-order *keys*
    (packed as :class:`_ForegroundIndex` packs them) fall inside *box* -- each
    one's flat C-order position in the box, and its index into *keys*."""
    (z0, z1), (y0, y1), (x0, x1) = (
        s.indices(int(n))[:2] for s, n in zip(box, shape)
    )
    width = max(0, x1 - x0)
    empty = np.empty(0, dtype=np.int64)
    if z1 <= z0 or y1 <= y0 or width == 0 or not len(keys):
        return empty, empty
    zz, yy = np.meshgrid(
        np.arange(z0, z1, dtype=np.int64), np.arange(y0, y1, dtype=np.int64), indexing="ij"
    )
    starts = (zz.reshape(-1) * int(shape[1]) + yy.reshape(-1)) * int(shape[2]) + x0
    first = np.searchsorted(keys, starts)
    counts = np.searchsorted(keys, starts + width) - first
    total = int(counts.sum())
    if total == 0:
        return empty, empty
    row = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
    ids = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    ids += np.repeat(first, counts)
    return row * width + (keys[ids] - starts[row]), ids


class _VoxelBoxes:
    """A set of voxels, read a box at a time as a dense boolean array -- what
    code that slices a whole-volume mask (``mask[z0:z1, y0:y1, x0:x1]``,
    ``mask.shape``) needs, without the volume.

    *keys* are the voxels' packed C-order keys, sorted; *members*, when
    given, is a boolean per key, and only keys marked True are in the set
    (it may change between reads).
    """

    def __init__(self, keys: np.ndarray, shape, members: np.ndarray | None = None):
        self._keys = keys
        self.shape = tuple(int(s) for s in shape)
        self._members = members

    def __getitem__(self, box) -> np.ndarray:
        box = tuple(slice(*s.indices(n)[:2]) for s, n in zip(box, self.shape))
        out = np.zeros(tuple(max(0, s.stop - s.start) for s in box), dtype=bool)
        positions, ids = _box_members(self._keys, self.shape, box)
        if self._members is not None:
            positions = positions[self._members[ids]]
        out.reshape(-1)[positions] = True
        return out


class _Both:
    """``a & b`` of two whole-volume masks, read a box at a time."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a = a
        self._b = b
        self.shape = tuple(a.shape)

    def __getitem__(self, box) -> np.ndarray:
        return np.asarray(self._a[box], dtype=bool) & np.asarray(self._b[box], dtype=bool)


def _keys_of(coords: np.ndarray, shape) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return (coords[:, 0] * int(shape[1]) + coords[:, 1]) * int(shape[2]) + coords[:, 2]


def _coords_of(keys: np.ndarray, shape) -> np.ndarray:
    return np.column_stack(np.unravel_index(np.asarray(keys, dtype=np.int64), tuple(shape))).astype(
        np.intp, copy=False
    )


def _keys_where(planes, shape, *, block_voxels: int | None = None) -> np.ndarray:
    """Packed C-order keys of the voxels where ``planes(z0, z1)`` is True,
    read a slab of z-planes at a time."""
    plane = int(shape[1]) * int(shape[2])
    step = memmap_support.slab_step(shape, block_voxels)
    parts = [
        np.flatnonzero(np.asarray(planes(z0, min(z0 + step, shape[0])), dtype=bool)).astype(np.int64)
        + z0 * plane
        for z0 in range(0, int(shape[0]), step)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


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


_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def medial_ridge_mask(radii: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """Voxels of *source_mask* whose own *radii* value is a local maximum
    among its 26 neighbours -- the medial ridge, one voxel per
    centreline-ish point rather than one per foreground voxel.

    *radii* is taken as given (e.g. the full mask's own
    :func:`inscribed_radius_map`) so a caller that wants the ridge of only
    *part* of a mask -- :mod:`haemolynx.preprocessing.segmentation_quality`
    restricts to components that are not small fragments -- still compares
    against each voxel's true neighbourhood, not one recomputed from the
    restricted subset alone (which would change distances near whatever got
    excluded).
    """
    return np.asarray(source_mask, dtype=bool) & (
        maximum_filter(radii, footprint=_STRUCTURE_26) == radii
    )


def medial_ridge_radii_um(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Inscribed radius, sampled only at the mask's own medial ridge.

    Most of a solid vessel's volume sits near its own surface, where
    distance-to-background is small, so a statistic (e.g. a median) taken
    over every foreground voxel systematically underestimates the vessel's
    true radius -- confirmed on a synthetic cylinder to read roughly 3-4x
    too low. Callers that want a genuine "typical vessel radius"
    (:mod:`haemolynx.preprocessing.segmentation_quality`'s resolution
    score, :mod:`haemolynx.optimisation.search`'s own scale reference) both
    need this, not the raw per-voxel map.
    """
    mask = np.asarray(binary, dtype=bool)
    if not mask.any():
        return np.zeros(0, dtype=np.float64)
    radii = inscribed_radius_map(mask, voxel_size_zyx)
    return radii[medial_ridge_mask(radii, mask)]


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
    restrict_to_mask: np.ndarray | None = None,
    use_memmap: bool = False,
    memmap_directory=None,
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

    *restrict_to_mask*, when given, is authoritative: every foreground
    voxel inside it becomes the fat body directly, and steps 1 and 2 (the
    radius-seeded core and its geodesic corridor) do not run at all --
    *min_radius_um* only still sets step 3's own wall radius (via
    *wall_absorption_um*'s default). This matters because a real vessel can
    fail to reach *min_radius_um* anywhere in the image and still
    genuinely be the large vessel a mask says it is -- an artery imaged
    before full muscularisation can measure narrower everywhere than its
    paired vein, for instance -- in which case the ordinary radius-seeded
    path would never find a core to propagate from at all, no matter how
    good the mask is. Step 3 still runs on this mask-defined body exactly
    as it would on a geodesic one, so nearby fused-vessel/surface
    roughness just past the mask's own edge is still absorbed the same way
    -- the result is not strictly confined to the mask's own footprint.

    *use_memmap* (the low-RAM option) never makes a distance map: every
    distance here is only compared with a radius of a few microns, and a
    block's own distance transform with that much context answers each
    comparison exactly as the whole one does (:mod:`.blockwise_edt`). The
    two radius gates go to one byte a voxel on disk, the geodesic body is
    found by labelling that a slab at a time (:mod:`.slab_labels`), and the
    wall is measured only in blocks with body within reach. Only the result
    is a volume in RAM. Same result either way.
    """
    mask = np.asarray(binary, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    if min_radius_um <= 0.0 or not mask.any():
        return out
    arguments = dict(
        min_radius_um=float(min_radius_um),
        voxel_size_zyx=voxel_size_zyx,
        wall_absorption_um=wall_absorption_um,
        restrict_to_mask=restrict_to_mask,
    )
    if use_memmap:
        _thick_vessel_object_mask_by_block(out, mask, **arguments, memmap_directory=memmap_directory)
    else:
        sampling = tuple(float(v) for v in voxel_size_zyx)
        _thick_vessel_object_mask_into(
            out,
            mask,
            **arguments,
            distance=lambda features: distance_transform_edt(features, sampling=sampling),
        )
    return out


def _thick_vessel_object_mask_into(
    out: np.ndarray,
    mask: np.ndarray,
    *,
    min_radius_um: float,
    voxel_size_zyx: tuple[float, float, float],
    wall_absorption_um: float | None,
    restrict_to_mask: np.ndarray | None,
    distance,
) -> None:
    """:func:`thick_vessel_object_mask`'s body, writing into *out*, with every
    distance transform taken through *distance*."""
    bbox = _foreground_bbox(mask, pad=1)
    if bbox is None:
        return
    crop = mask[bbox]
    logger.info(
        "thick_vessel_object_mask: cropped to %s (%d voxels, %d foreground) "
        "from input shape %s",
        crop.shape,
        int(crop.size),
        int(crop.sum()),
        mask.shape,
    )
    wall_radius = (
        0.5 * float(min_radius_um)
        if wall_absorption_um is None
        else max(0.0, float(wall_absorption_um))
    )

    if restrict_to_mask is not None:
        body = crop & np.asarray(restrict_to_mask, dtype=bool)[bbox]
        if not body.any():
            return
        t0 = time.perf_counter()
        dist_to_body = distance(~body)
        out[bbox] = crop & (dist_to_body <= wall_radius)
        logger.info(
            "thick_vessel_object_mask: restricted to mask directly (no "
            "radius test), wall distance transform took %.2fs (%d fat "
            "voxels total)",
            time.perf_counter() - t0,
            int(out.sum()),
        )
        return

    t0 = time.perf_counter()
    radius_map = distance(crop)
    thick_core = crop & (radius_map >= float(min_radius_um))
    logger.info(
        "thick_vessel_object_mask: inscribed radius map took %.2fs (%d core voxels)",
        time.perf_counter() - t0,
        int(thick_core.sum()),
    )
    if not thick_core.any():
        return

    propagation_gate = 0.5 * float(min_radius_um)
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
    dist_to_body = distance(~body)
    out[bbox] = crop & (dist_to_body <= wall_radius)
    logger.info(
        "thick_vessel_object_mask: wall distance transform took %.2fs (%d fat voxels total)",
        time.perf_counter() - t2,
        int(out.sum()),
    )


def _thick_vessel_object_mask_by_block(
    out: np.ndarray,
    mask: np.ndarray,
    *,
    min_radius_um: float,
    voxel_size_zyx: tuple[float, float, float],
    wall_absorption_um: float | None,
    restrict_to_mask: np.ndarray | None,
    memmap_directory,
) -> None:
    """:func:`_thick_vessel_object_mask_into` for the low-RAM option: the same
    three steps on the same crop, a block or a slab at a time.

    1. Each block's radius map with ``ceil(min_radius_um / spacing) + 1``
       voxels of context decides both gates (radius at least *min_radius_um*,
       and at least half that) as the whole crop's does, written as one byte a
       voxel to disk: 2 for the core, 1 for the rest of the corridor.
    2. ``binary_propagation`` of the core through the corridor is the
       corridor's face-connected components that hold a core voxel: labelled
       a slab at a time, then written over the codes as the body.
    3. The wall, with context for *wall_absorption_um*, only where body lies
       within reach of the block.
    """
    bbox = _foreground_bbox(mask, pad=1)
    if bbox is None:
        return
    crop = mask[bbox]
    n_foreground = int(np.count_nonzero(crop))
    logger.info(
        "thick_vessel_object_mask: cropped to %s (%d voxels, %d foreground) "
        "from input shape %s",
        crop.shape,
        int(crop.size),
        n_foreground,
        mask.shape,
    )
    sampling = tuple(float(v) for v in voxel_size_zyx)
    if n_foreground == crop.size:
        # No background anywhere: scipy's radius map is not a distance to
        # anything, and only its whole-array call reproduces it.
        _thick_vessel_object_mask_into(
            out,
            mask,
            min_radius_um=min_radius_um,
            voxel_size_zyx=voxel_size_zyx,
            wall_absorption_um=wall_absorption_um,
            restrict_to_mask=restrict_to_mask,
            distance=lambda features: distance_transform_edt(features, sampling=sampling),
        )
        return
    wall_radius = (
        0.5 * float(min_radius_um)
        if wall_absorption_um is None
        else max(0.0, float(wall_absorption_um))
    )
    out_crop = out[bbox]

    if restrict_to_mask is not None:
        t0 = time.perf_counter()
        block_voxels, workers = blockwise_edt.parallel_plan(crop.shape)
        _absorb_wall_by_block(
            out_crop,
            (mask, np.asarray(restrict_to_mask, dtype=bool)),
            _outside_restricted_body,
            dict(origin=tuple(int(s.start) for s in bbox), wall=wall_radius),
            sampling=sampling,
            block_voxels=block_voxels,
            workers=workers,
            memmap_directory=memmap_directory,
        )
        logger.info(
            "thick_vessel_object_mask: restricted to mask directly (no "
            "radius test), wall distance transform took %.2fs (%d fat "
            "voxels total)",
            time.perf_counter() - t0,
            int(np.count_nonzero(out_crop)),
        )
        return

    block_voxels, workers = blockwise_edt.parallel_plan(crop.shape)
    if workers > 1:
        logger.info(
            "thick_vessel_object_mask: %d worker processes, blocks of %d voxels",
            workers,
            block_voxels,
        )
    origin = tuple(int(s.start) for s in bbox)
    codes = memmap_support.new_memmap_array(crop.shape, np.uint8, directory=memmap_directory)
    try:
        t0 = time.perf_counter()
        n_core = 0
        params = dict(origin=origin, min_radius=float(min_radius_um), gate=0.5 * float(min_radius_um))
        for core, (in_core, corridor, n) in blockwise_edt.near_blocks(
            _crop_of, _crop_of, _radius_codes, (mask,), crop.shape,
            sampling=sampling, reach=float(min_radius_um), params=params,
            block_voxels=block_voxels, workers=workers, memmap_directory=memmap_directory,
        ):
            shape = tuple(c.stop - c.start for c in core)
            codes[core] = (
                2 * blockwise_edt.unpack(in_core, shape).astype(np.uint8)
                + blockwise_edt.unpack(corridor, shape).astype(np.uint8)
            )
            n_core += n
        logger.info(
            "thick_vessel_object_mask: inscribed radius map took %.2fs (%d core voxels)",
            time.perf_counter() - t0,
            n_core,
        )
        if n_core == 0:
            return
        t1 = time.perf_counter()
        n_body = _propagate_core_by_slab(codes)
        logger.info(
            "thick_vessel_object_mask: geodesic body propagation took %.2fs (%d body voxels)",
            time.perf_counter() - t1,
            n_body,
        )
        t2 = time.perf_counter()
        _absorb_wall_by_block(
            out_crop,
            (mask, codes),
            _outside_coded_body,
            dict(origin=origin, wall=wall_radius),
            sampling=sampling,
            block_voxels=block_voxels,
            workers=workers,
            memmap_directory=memmap_directory,
        )
        logger.info(
            "thick_vessel_object_mask: wall distance transform took %.2fs (%d fat voxels total)",
            time.perf_counter() - t2,
            int(np.count_nonzero(out_crop)),
        )
    finally:
        memmap_support.release_memmap_array(codes)


def _propagate_core_by_slab(codes: np.ndarray) -> int:
    """``binary_propagation(codes == 2, mask=codes >= 1)``, written over
    *codes* as 1 on the result and 0 elsewhere; returns its voxel count.

    Propagation through scipy's default structure reaches exactly the
    face-connected components of the mask that hold a seed, so those are
    found by labelling a slab at a time and labelling it again to write.
    """
    structure = generate_binary_structure(3, 1)

    def corridor(z0, z1):
        return np.asarray(codes[z0:z1]) >= 1

    seeded = []

    def visit(z0, z1, labels, offset):
        found = labels[np.asarray(codes[z0:z1]) == 2]
        if found.size:
            seeded.append(np.unique(found).astype(np.int64) + offset)

    components = slab_labels.label_by_slab(corridor, codes.shape, structure=structure, visit=visit)
    keep = np.zeros(components.count + 1, dtype=bool)
    if seeded:
        keep[components.number[np.concatenate(seeded)]] = True
    n_body = 0
    for z0, z1, labels, offset in components.replay(corridor):
        body = keep[components.numbers(labels, offset)]
        codes[z0:z1] = body
        n_body += int(np.count_nonzero(body))
    return n_body


def _absorb_wall_by_block(
    out_crop: np.ndarray,
    volumes: tuple[np.ndarray, ...],
    outside_body,
    params: dict,
    *,
    sampling: tuple[float, float, float],
    block_voxels: int,
    workers: int,
    memmap_directory,
) -> None:
    """``out_crop[...] = crop & (distance_transform_edt(~body) <= wall)``, a
    block at a time over *workers* processes: the crop is ``volumes[0]`` at
    ``params["origin"]``, and ``outside_body(volumes, box, params)`` gives
    ``~body`` over a box. A block with no body within the wall is left False
    without a transform."""
    for core, bits in blockwise_edt.near_blocks(
        outside_body, _crop_of, _within_wall, volumes, out_crop.shape,
        sampling=sampling, reach=float(params["wall"]), params=params,
        block_voxels=block_voxels, workers=workers, memmap_directory=memmap_directory,
    ):
        out_crop[core] = blockwise_edt.unpack(bits, tuple(c.stop - c.start for c in core))


# The low-RAM fat catchment's blocks, as blockwise_edt.near_blocks reads and
# decides them -- module-level, so worker processes can be sent them. Each
# volume is the whole stack; params["origin"] is where the crop starts in it.


def _in_stack(box: tuple[slice, ...], params: dict) -> tuple[slice, ...]:
    return tuple(slice(s.start + o, s.stop + o) for s, o in zip(box, params["origin"]))


def _crop_of(volumes, box, params) -> np.ndarray:
    return np.asarray(volumes[0][_in_stack(box, params)], dtype=bool)


def _outside_coded_body(volumes, box, params) -> np.ndarray:
    """``~body``: *volumes[1]* holds the body as 1, over the crop's box."""
    return np.asarray(volumes[1][box]) != 1


def _outside_restricted_body(volumes, box, params) -> np.ndarray:
    """``~(crop & restriction)``, *volumes[1]* the restriction mask."""
    at = _in_stack(box, params)
    return ~(np.asarray(volumes[0][at], dtype=bool) & np.asarray(volumes[1][at], dtype=bool))


def _outside_thick(volumes, box, params) -> np.ndarray:
    return ~np.asarray(volumes[1][_in_stack(box, params)], dtype=bool)


def _thin_catchment(volumes, box, params) -> np.ndarray:
    at = _in_stack(box, params)
    return np.asarray(volumes[0][at], dtype=bool) & ~np.asarray(volumes[1][at], dtype=bool)


def _radius_codes(where, distance, params):
    """The core (radius at least ``min_radius``) and the rest of the corridor
    (at least ``gate``), as bits, and how many core voxels."""
    in_core = distance >= params["min_radius"]
    core = np.zeros(where.shape, dtype=bool)
    corridor = np.zeros(where.shape, dtype=bool)
    core[where] = in_core
    corridor[where] = ~in_core & (distance >= params["gate"])
    return blockwise_edt.pack(core), blockwise_edt.pack(corridor), int(np.count_nonzero(in_core))


def _within_wall(where, distance, params):
    block = np.zeros(where.shape, dtype=bool)
    block[where] = distance <= params["wall"]
    return blockwise_edt.pack(block)


def _reaching(where, distance, params):
    """The thin catchment at least ``reach`` from the fat region, as bits,
    with how many thin voxels and how many of them reach."""
    block = np.zeros(where.shape, dtype=bool)
    block[where] = distance >= params["reach"]
    return blockwise_edt.pack(block), int(np.count_nonzero(where)), int(np.count_nonzero(block))


def diagnose_mask_restriction_alignment(
    masks: Mapping[str, np.ndarray | None],
    segmented_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """How much of each named restriction mask actually falls on real
    segmented foreground.

    ``thick_vessel_object_mask``'s *restrict_to_mask* seeds the fat body
    directly from mask membership, with no radius test at all (see its own
    docstring) -- so the restriction is only as good as the mask's own
    agreement with where the vessel actually is in the segmented image. A
    low fraction here means much of the mask sits on background: it is
    misaligned with, or a different shape/size than, the vessel as it
    actually appears in the segmented image, and wall absorption (a fixed
    distance from wherever the mask *did* land on real foreground) may not
    be enough to compensate for that on its own.

    *masks* maps a name (e.g. ``"large_arteriole_mask"``) to its array, or
    ``None`` for a role that was not loaded -- skipped, along with any mask
    with no voxels of its own (nothing to compare). Returns one report per
    remaining mask, in the order given.
    """
    seg = np.asarray(segmented_mask, dtype=bool)
    reports: list[dict[str, Any]] = []
    for role, mask in masks.items():
        if mask is None:
            continue
        restrict = np.asarray(mask, dtype=bool)
        n_mask = int(restrict.sum())
        if n_mask == 0:
            continue
        n_overlap = int((restrict & seg).sum())
        reports.append(
            {
                "role": role,
                "mask_voxels": n_mask,
                "overlap_voxels": n_overlap,
                "overlap_fraction": n_overlap / n_mask,
            }
        )
    return reports


def format_mask_restriction_alignment_report(report: Mapping[str, Any]) -> str:
    """A one-line summary of one entry from :func:`diagnose_mask_restriction_alignment`."""
    return (
        f"Thick-vessel mask restriction: {report['role']} has "
        f"{report['overlap_voxels']} of {report['mask_voxels']} voxels "
        f"({report['overlap_fraction']:.1%}) on the segmented image -- the "
        "rest sit on background, which usually means the mask is "
        "misaligned with, or a different shape than, the vessel as it "
        "actually appears in the segmented image."
    )


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
    return _bbox_of_any(z_any, mask_b.any(axis=(0, 2)), mask_b.any(axis=(0, 1)), mask_b.shape, pad)


def _foreground_bbox_by_slab(planes, shape, *, pad: int = 1) -> tuple[slice, slice, slice] | None:
    """:func:`_foreground_bbox` of the volume ``planes(z0, z1)`` reads, a slab
    of z-planes at a time."""
    z_any = np.zeros(int(shape[0]), dtype=bool)
    y_any = np.zeros(int(shape[1]), dtype=bool)
    x_any = np.zeros(int(shape[2]), dtype=bool)
    step = memmap_support.slab_step(shape)
    for z0 in range(0, int(shape[0]), step):
        slab = np.asarray(planes(z0, min(z0 + step, int(shape[0]))), dtype=bool)
        z_any[z0:z0 + len(slab)] = slab.any(axis=(1, 2))
        y_any |= slab.any(axis=(0, 2))
        x_any |= slab.any(axis=(0, 1))
    if not z_any.any():
        return None
    return _bbox_of_any(z_any, y_any, x_any, shape, pad)


def _bbox_of_any(z_any, y_any, x_any, shape, pad) -> tuple[slice, slice, slice]:
    z0 = int(np.argmax(z_any))
    z1 = int(z_any.size - np.argmax(z_any[::-1]))
    y0 = int(np.argmax(y_any))
    y1 = int(y_any.size - np.argmax(y_any[::-1]))
    x0 = int(np.argmax(x_any))
    x1 = int(x_any.size - np.argmax(x_any[::-1]))
    p = int(pad)
    return (
        slice(max(0, z0 - p), min(int(shape[0]), z1 + p)),
        slice(max(0, y0 - p), min(int(shape[1]), y1 + p)),
        slice(max(0, x0 - p), min(int(shape[2]), x1 + p)),
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


def _build_dijkstra_graph(
    binary: np.ndarray, cost: np.ndarray, relative_spacing: np.ndarray | None = None
):
    """The sparse 26-neighbour weighted graph over foreground voxels of *binary*.

    A step into a voxel costs that voxel's *cost* times :func:`_step_factor`:
    on an anisotropic crop a step along the coarse axis covers more tissue,
    and costed per voxel it came four times cheaper per micron.

    This -- not the Dijkstra walk itself -- is the expensive part of a call
    to :func:`_dijkstra_parents` on a large crop: it means a Python loop over
    26 neighbour offsets, boolean masking and an ``_ForegroundIndex`` build
    (sorting all foreground coordinates), all sized to the crop. A caller
    that runs several Dijkstra calls on the same (*binary*, *cost*) pair with
    a different *root* each time -- :func:`_join_thin_arms_to_fat_ridge`'s
    fallback, once per arm that needs it in a given physically connected
    structure -- should build this once and pass it back in via
    *precomputed_graph*, rather than pay for an identical rebuild per call
    (measured at ~90% of the per-call cost on a ~840k-voxel crop).

    Returns ``(graph, fg_coords, index_of)`` -- see :func:`_dijkstra_parents`.
    """
    binary_f = np.asarray(binary, dtype=bool)
    fg_coords = np.argwhere(binary_f)
    cost_a = np.asarray(cost, dtype=np.float64)
    return _build_dijkstra_graph_from_coords(
        fg_coords, binary_f.shape, cost_a[tuple(fg_coords.T)], relative_spacing
    )


def _build_dijkstra_graph_from_coords(
    fg_coords: np.ndarray,
    shape: tuple[int, int, int],
    cost_at: np.ndarray,
    relative_spacing: np.ndarray | None = None,
):
    """:func:`_build_dijkstra_graph` from the foreground's own voxels
    (``(n, 3)``, C order, in a box of *shape*) and the cost of stepping into
    each, with nothing the size of the box.

    Written straight into the CSR arrays a z-plane of voxels at a time --
    counted in one pass, filled in a second -- each voxel's 26 neighbours
    read from an index of the three planes round it. Gathered as row, column
    and weight chunks and converted, the edges were held about three times
    over (~1.1 KB a voxel of a solid fat component, the low-RAM run's peak)
    where the graph itself is 12 bytes an edge. The offsets run in C order,
    so each row's neighbours come out sorted, as the conversion sorted them.
    """
    n_fg = int(fg_coords.shape[0])
    index_of = _ForegroundIndex(fg_coords, shape)
    z_max, y_max, x_max = (int(s) for s in shape)
    cost_a = np.asarray(cost_at, dtype=np.float64)
    relative = np.ones(3) if relative_spacing is None else np.asarray(relative_spacing, float)
    factors = np.array([_step_factor(offset, relative) for offset in _OFFSETS_26])
    offsets = np.array(_OFFSETS_26, dtype=np.intp)
    # Each plane's voxels are a run of the C-ordered list; indexed with a
    # border of -1, so a neighbour off the box's edge reads as none.
    bounds = np.searchsorted(fg_coords[:, 0], np.arange(z_max + 1))
    ys = fg_coords[:, 1].astype(np.intp) + 1
    xs = fg_coords[:, 2].astype(np.intp) + 1
    planes: dict[int, np.ndarray | None] = {}

    def plane(z: int) -> np.ndarray | None:
        if z not in planes:
            lo, hi = int(bounds[z]), int(bounds[z + 1])
            if hi > lo:
                index = np.full((y_max + 2, x_max + 2), -1, dtype=np.int32)
                index[ys[lo:hi], xs[lo:hi]] = np.arange(lo, hi, dtype=np.int32)
            else:
                index = None
            planes[z] = index
            for old in [k for k in planes if k < z - 1]:
                del planes[old]
        return planes[z]

    def steps(z: int):
        """``(neighbours, weights, kept)`` of plane *z*'s voxels, a row of
        26 per voxel; *kept* marks the steps that are edges."""
        lo, hi = int(bounds[z]), int(bounds[z + 1])
        y, x = ys[lo:hi], xs[lo:hi]
        neighbours = np.full((hi - lo, len(offsets)), -1, dtype=np.int32)
        for k, (dz, dy, dx) in enumerate(offsets):
            index = plane(z + int(dz)) if 0 <= z + int(dz) < z_max else None
            if index is not None:
                neighbours[:, k] = index[y + dy, x + dx]
        kept = neighbours >= 0
        weights = np.zeros(neighbours.shape, dtype=np.float64)
        weights[kept] = cost_a[neighbours[kept]] * np.broadcast_to(factors, neighbours.shape)[kept]
        kept &= np.isfinite(weights) & (weights > 0.0)
        return neighbours, weights, kept

    occupied = [z for z in range(z_max) if bounds[z + 1] > bounds[z]]
    counts = np.zeros(n_fg, dtype=np.int64)
    for z in occupied:
        _neighbours, _weights, kept = steps(z)
        counts[bounds[z]:bounds[z + 1]] = kept.sum(axis=1)
    planes.clear()
    nnz = int(counts.sum())
    index_type = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n_fg + 1, dtype=index_type)
    np.cumsum(counts, out=indptr[1:])
    del counts
    indices = np.empty(nnz, dtype=index_type)
    data = np.empty(nnz, dtype=np.float64)
    for z in occupied:
        neighbours, weights, kept = steps(z)
        start, stop = int(indptr[bounds[z]]), int(indptr[bounds[z + 1]])
        indices[start:stop] = neighbours[kept]
        data[start:stop] = weights[kept]
    planes.clear()
    graph = csr_matrix((data, indices, indptr), shape=(n_fg, n_fg))
    return graph, fg_coords, index_of


def _dijkstra_parents(
    binary: np.ndarray,
    cost: np.ndarray,
    root: tuple[int, int, int],
    *,
    precomputed_graph=None,
    relative_spacing: np.ndarray | None = None,
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

    *precomputed_graph*, from :func:`_build_dijkstra_graph`, skips rebuilding
    that graph -- see its docstring for why that matters for a caller
    re-running this on the same (*binary*, *cost*) with a different *root*.
    """
    if precomputed_graph is not None:
        graph, fg_coords, index_of = precomputed_graph
    else:
        graph, fg_coords, index_of = _build_dijkstra_graph(binary, cost, relative_spacing)
    n_fg = int(fg_coords.shape[0])
    if n_fg == 0:
        return None
    root_i = int(index_of[root])
    if root_i < 0:
        return None

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


def _symmetric_covariance_3x3(centered: np.ndarray) -> np.ndarray:
    """Covariance matrix of *centered* (N x 3, already mean-subtracted).

    Six elementwise sums rather than ``np.cov``: ``np.cov`` crashes the
    interpreter outright (no Python exception, a native fault) on some
    Windows NumPy/BLAS builds, even though what it computes here is nothing
    more than that -- no linear-algebra library involved either way.
    """
    divisor = max(centered.shape[0] - 1, 1)
    cov = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            value = float(np.sum(centered[:, i] * centered[:, j])) / divisor
            cov[i, j] = value
            cov[j, i] = value
    return cov


def _matvec_3x3(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """*matrix* @ *vector* for a 3x3 matrix, by explicit scalar arithmetic.

    Not ``matrix @ vector``: matrix-vector product crashes the interpreter
    outright on some Windows NumPy/BLAS builds too, for a large enough left
    operand -- not just the full decompositions this module's other
    LAPACK-avoidance helpers exist for (see :func:`_dominant_eigenvector_3x3`
    and :func:`_project_onto_axis`, its other caller). A fixed 3x3 case never
    dispatches to a matrix routine at all when spelled out this way.
    """
    return np.array(
        [
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1] + matrix[0, 2] * vector[2],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1] + matrix[1, 2] * vector[2],
            matrix[2, 0] * vector[0] + matrix[2, 1] * vector[1] + matrix[2, 2] * vector[2],
        ]
    )


def _project_onto_axis(centered: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """*centered* (N x 3) projected onto unit vector *axis*, elementwise.

    Not ``centered @ axis``: see :func:`_matvec_3x3` -- the same crash, this
    time for the (potentially large) N x 3 case rather than a fixed 3x3 one.
    """
    return centered[:, 0] * axis[0] + centered[:, 1] * axis[1] + centered[:, 2] * axis[2]


def _dominant_eigenvector_3x3(matrix: np.ndarray, *, iterations: int = 100) -> np.ndarray:
    """Unit eigenvector for the largest eigenvalue of a symmetric 3x3 matrix.

    Power iteration -- repeatedly multiply a seed vector by *matrix* and
    renormalise, which converges to the dominant eigenvector geometrically in
    the eigenvalue gap ratio -- rather than ``np.linalg.eigh``, which crashes
    the interpreter outright (no Python exception, a native fault) on some
    Windows NumPy/BLAS builds; ``scipy.linalg.eigh`` crashes identically, so
    this is not a numpy-specific problem to route around by switching
    library. Every caller below only ever wanted this one eigenvector, never
    a full decomposition or the eigenvalues themselves, so this is not an
    approximation of what they needed -- just a different, LAPACK-free way to
    get exactly that. 100 iterations is generous for a 3x3 matrix: real
    convergence is geometric in the (typically well-separated) eigenvalue
    gap and settles in well under 20.
    """
    vector = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    for _ in range(iterations):
        vector = _matvec_3x3(matrix, vector)
        norm = float(np.sqrt(np.sum(vector * vector)))
        if norm < 1e-300:
            # matrix is (numerically) the zero matrix: every direction is as
            # arbitrary -- and as valid -- an answer as any other.
            return np.array([1.0, 0.0, 0.0])
        vector = vector / norm
    return vector


def _principal_axis(coords: np.ndarray) -> np.ndarray:
    """Unit vector along *coords*' (N x 3) own direction of greatest spread.

    *coords* that is not N x 3 (every real caller's own argwhere() on a 3D
    array always is) has no principal axis to compute against -- an
    arbitrary axis is as valid an answer as any other, matching
    :func:`_dominant_eigenvector_3x3`'s own zero-matrix fallback, rather
    than an opaque IndexError out of :func:`_symmetric_covariance_3x3`.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return np.array([1.0, 0.0, 0.0])
    centered = coords.astype(float) - coords.astype(float).mean(axis=0)
    return _dominant_eigenvector_3x3(_symmetric_covariance_3x3(centered))


def _principal_endpoints(
    coords: np.ndarray, relative_spacing: np.ndarray | None = None
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Voxels at the ends of the component's long axis, measured physically."""
    if coords.shape[0] < 2:
        voxel = tuple(int(v) for v in coords[0])
        return voxel, voxel
    scaled = coords.astype(float) * (1.0 if relative_spacing is None else relative_spacing)
    centered = scaled - scaled.mean(axis=0)
    axis = _dominant_eigenvector_3x3(_symmetric_covariance_3x3(centered))
    projection = _project_onto_axis(centered, axis)
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
    relative_spacing: np.ndarray | None = None,
) -> np.ndarray:
    """Voxels within *radius_voxels* voxels of the finest axis of *path*,
    measured physically."""
    mask = np.zeros(shape, dtype=bool)
    if not path:
        return mask
    radius = max(1, int(radius_voxels))
    relative = np.ones(3) if relative_spacing is None else relative_spacing
    reach = np.ceil(radius / relative).astype(int)
    pts = np.asarray(path, dtype=int)
    lo = np.maximum(pts.min(axis=0) - reach, 0)
    hi = np.minimum(pts.max(axis=0) + reach + 1, shape)
    slc = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = np.zeros(tuple(int(b - a) for a, b in zip(lo, hi)), dtype=bool)
    local[tuple((pts - lo).T)] = True
    mask[slc] = distance_transform_edt(~local, sampling=tuple(relative)) <= radius
    return mask


def _covered_around_path(
    path: list[tuple[int, int, int]],
    radius_voxels: int,
    shape: tuple[int, int, int],
    relative_spacing: np.ndarray | None,
    index_of: _ForegroundIndex,
    fg_coords: np.ndarray,
) -> np.ndarray:
    """Which foreground voxels :func:`_cover_around_path` marks, as ids into
    *fg_coords* -- from the voxels themselves and a KD-tree of *path*, not a
    distance transform of the path's box (for a trunk running the length of
    a stack, the box is the stack).

    A voxel is covered when its distance to the nearest path voxel, by
    ``distance_transform_edt``'s own arithmetic, is within the radius, as
    there; the KD-tree only finds the candidates.
    """
    if not path:
        return np.empty(0, dtype=np.int64)
    radius = max(1, int(radius_voxels))
    relative = np.ones(3) if relative_spacing is None else np.asarray(relative_spacing, float)
    reach = np.ceil(radius / relative).astype(int)
    pts = np.asarray(path, dtype=int)
    lo = np.maximum(pts.min(axis=0) - reach, 0)
    hi = np.minimum(pts.max(axis=0) + reach + 1, shape)
    _positions, ids = _box_members(
        index_of.keys, shape, tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    )
    if not ids.size:
        return ids
    query = fg_coords[ids].astype(np.int64)
    tree = cKDTree(pts.astype(np.float64) * relative)
    # The tree's own distances differ from scipy's in the last bits, so the
    # bound has slack and the verdict is scipy's arithmetic.
    slack = radius * 1e-9 + 1e-9
    distance, nearest = tree.query(
        query.astype(np.float64) * relative, k=1, distance_upper_bound=radius + slack
    )
    found = np.flatnonzero(np.isfinite(distance))
    within = np.zeros(len(ids), dtype=bool)
    if found.size:
        exact = blockwise_edt.displacement_distances(pts[nearest[found]] - query[found], relative)
        within[found] = exact <= radius
        # A tie the tree broke the other way: another path voxel as near,
        # whose distance by scipy's arithmetic is within the radius.
        for row in found[~within[found]]:
            near = tree.query_ball_point(query[row].astype(np.float64) * relative, radius + slack)
            if near:
                within[row] = bool(
                    np.any(
                        blockwise_edt.displacement_distances(pts[near] - query[row], relative)
                        <= radius
                    )
                )
    return ids[within]


def _local_principal_axis(
    component: np.ndarray,
    point: tuple[int, int, int],
    radius: float,
    relative_spacing: np.ndarray | None = None,
) -> np.ndarray | None:
    """Long axis of *component* in a ball around *point*, or None if too small
    -- ball and axis in voxels of the finest axis, measured physically."""
    r = max(2, int(np.ceil(radius)))
    relative = np.ones(3) if relative_spacing is None else relative_spacing
    rz, ry, rx = (int(np.ceil(r / s)) for s in relative)
    z, y, x = (int(v) for v in point)
    z0, z1 = max(0, z - rz), min(component.shape[0], z + rz + 1)
    y0, y1 = max(0, y - ry), min(component.shape[1], y + ry + 1)
    x0, x1 = max(0, x - rx), min(component.shape[2], x + rx + 1)
    crop = component[z0:z1, y0:y1, x0:x1]
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    ball = (
        ((zz - z) * relative[0]) ** 2 + ((yy - y) * relative[1]) ** 2
        + ((xx - x) * relative[2]) ** 2 <= r * r
    )
    coords = np.argwhere(crop & ball)
    if coords.shape[0] < 8:
        return None
    return _principal_axis(coords * relative)


def _path_cuts_across_lumen(
    path: list[tuple[int, int, int]],
    component: np.ndarray,
    max_edt: float,
    relative_spacing: np.ndarray | None = None,
) -> bool:
    """True when *path* crosses a tube's width instead of following an arm.

    A flattened plasma column's medial geometry is a sheet: the farthest
    high-EDT voxel from the trunk is across the lumen, and the geodesic to it
    is a short perpendicular stub. A real arm's tip has a local long axis
    aligned with that geodesic.
    """
    if len(path) < 2:
        return True
    relative = np.ones(3) if relative_spacing is None else relative_spacing
    start = np.asarray(path[0], dtype=float) * relative
    end = np.asarray(path[-1], dtype=float) * relative
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return True
    direction /= norm
    # Anchored at the path's midpoint, not its tip: a real arm's tip sits on
    # the component's rounded end-cap, where the local neighbourhood is
    # ball-shaped rather than tube-shaped and PCA has no reliable long axis
    # (confirmed on a symmetric 4-arm cross fixture -- the tip-anchored axis
    # picked up curvature from the cap and rejected a genuine arm). The
    # midpoint sits inside the tube on both a real arm and a short
    # perpendicular sheet-crossing stub, so it reads the same in the case
    # this check exists to catch.
    axis = _local_principal_axis(
        component, path[len(path) // 2], radius=max(3.0, 1.25 * float(max_edt)),
        relative_spacing=relative,
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


def _path_length(path: list[tuple[int, int, int]], relative_spacing: np.ndarray) -> float:
    """A voxel path's length in voxels of the finest axis, counting its first
    voxel and weighting each step by :func:`_step_factor` -- on cube voxels,
    its voxel count."""
    if not path:
        return 0.0
    steps = np.diff(np.asarray(path, dtype=float), axis=0)
    if steps.size == 0:
        return 1.0
    voxel_steps = np.sqrt(np.sum(steps * steps, axis=1))
    physical = np.sqrt(np.sum((steps * relative_spacing) ** 2, axis=1))
    return 1.0 + float(np.sum(physical / voxel_steps))


def _component_edt_ridge_on_crop(
    component: np.ndarray, edt: np.ndarray, relative_spacing: np.ndarray | None = None
) -> np.ndarray:
    """Centreline tree on an already-cropped fat component. *edt* and every
    distance here are in voxels of the finest axis (see
    :func:`_relative_spacing`)."""
    component = np.asarray(component, dtype=bool)
    result = np.zeros(component.shape, dtype=bool)
    coords = np.argwhere(component)
    if coords.size == 0:
        return result
    tree = _component_tree(coords, np.asarray(edt)[tuple(coords.T)], component.shape, relative_spacing)
    result[tuple(tree.T)] = True
    return result


def _component_tree(
    coords: np.ndarray,
    edt_at: np.ndarray,
    shape: tuple[int, int, int],
    relative_spacing: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`_component_edt_ridge_on_crop` from the component's own voxels:
    *coords* (``(n, 3)``, C order, in its box of *shape*) and *edt_at*, the
    EDT at each. Returns the tree's voxels, in that box.

    Nothing here is the size of the box: a fat network that crosses a stack
    has a box the size of the stack, and the cost, radius and coverage maps
    over it -- with a distance transform of the trunk's own box -- were tens
    of bytes a voxel of it.
    """
    relative = np.ones(3) if relative_spacing is None else np.asarray(relative_spacing, float)
    coords = np.asarray(coords, dtype=np.intp)
    edt_at = np.asarray(edt_at)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.intp)

    max_edt = float(edt_at.max())
    if len(coords) < int(np.prod(shape)):
        # The box's other voxels read 0 in the component's radius map.
        max_edt = max(max_edt, 0.0)
    cost = 1.0 / (np.square(edt_at) + 1e-6)
    root, far = _principal_endpoints(coords, relative)
    graph = _build_dijkstra_graph_from_coords(coords, shape, cost, relative)
    walked = _dijkstra_parents(None, None, root, precomputed_graph=graph)
    if walked is None:
        return coords[[int(np.argmax(edt_at))]]
    parent, fg_coords, index_of = walked
    main = _traceback(parent, fg_coords, index_of, root, far)
    if len(main) < 2:
        return coords[[int(np.argmax(edt_at))]]
    in_tree = np.zeros(len(coords), dtype=bool)
    result = _VoxelBoxes(index_of.keys, shape, members=in_tree)
    component = _VoxelBoxes(index_of.keys, shape)

    def draw(path):
        in_tree[index_of[tuple(np.asarray(path, dtype=np.intp).T)]] = True

    def cover(path, radius):
        covered[_covered_around_path(path, radius, shape, relative, index_of, coords)] = True

    draw(main)
    cover_r = max(1, int(round(max_edt)))
    covered = np.zeros(len(coords), dtype=bool)
    cover(main, cover_r)
    min_arm_voxels = max(4, int(2.0 * max_edt))
    high = edt_at >= 0.4 * max_edt
    # Every voxel this loop ever adds to the tree is exactly one of `main`
    # or an accepted `branch` -- both already known as coordinate lists, so
    # the tree can grow by appending them instead of re-deriving the whole
    # (up to component-sized) set on every accepted arm. On a real
    # multi-million-voxel fat component with many arms, that full rescan per
    # acceptance was most of this function's cost.
    tree_coords = np.asarray(main, dtype=np.intp)
    tree_kdt = cKDTree(tree_coords.astype(np.float64, copy=False) * relative)

    # A real network's single connected fat catchment can legitimately have
    # far more than a dozen arms (a whole fused sub-network, not one trunk);
    # 12 was an arbitrary safety bound, not a correctness threshold -- the
    # real stopping conditions are the two `break`s below. Each iteration
    # was already made to clear a whole rejected branch's neighbourhood
    # rather than one voxel, so raising this costs one KD-tree query per
    # additional *accepted* arm, not per candidate voxel.
    max_arms = 500
    for _ in range(max_arms):
        candidates = np.flatnonzero(high & ~covered)
        if not candidates.size:
            break
        cand_coords = coords[candidates]
        dists, _ = tree_kdt.query(cand_coords.astype(np.float64, copy=False) * relative, k=1)
        far_dist = float(np.max(dists))
        # Remaining high-EDT voxels sitting beside the trunk are the sheet, not arms.
        if far_dist < 2.0 * max_edt:
            break
        target = tuple(int(v) for v in cand_coords[int(np.argmax(dists))])
        branch = _trim_to_tree(
            _traceback(parent, fg_coords, index_of, root, target),
            result,
        )
        if _path_length(branch, relative) < min_arm_voxels:
            # Cover the rejected branch's own neighbourhood so the same
            # short stub is not re-traced next iteration -- but only a
            # small fixed margin, not the full cover_r (the TRUNK's own
            # radius, which can be 10-20+ voxels for a wide vessel). A
            # too-short stub and a separate, genuinely long-enough arm can
            # sit within cover_r of each other; covering that whole
            # neighbourhood pre-emptively removed the real arm's
            # candidate voxels before they were ever tried, silently
            # dropping it from the centreline tree.
            cover(branch, min(2, cover_r))
            continue
        if _touches_tree(result, branch[-1]):
            # Tip already on the tree: this geodesic would close a loop.
            covered[index_of[target]] = True
            continue
        if _path_cuts_across_lumen(branch, component, max_edt, relative):
            cover(branch, cover_r)
            continue
        draw(branch)
        cover(branch, cover_r)
        tree_coords = np.concatenate(
            [tree_coords, np.asarray(branch, dtype=np.intp)], axis=0
        )
        tree_kdt = cKDTree(tree_coords.astype(np.float64, copy=False) * relative)
    else:
        # Loop ran out of iterations without a break, i.e. without ever
        # deciding "no arms left" or "what's left is the sheet, not an arm" --
        # real candidate arms may still remain, silently dropped from the tree.
        remaining = int(np.count_nonzero(high & ~covered))
        if remaining:
            logger.warning(
                "Fat-vessel centreline hit the %d-arm cap with %d high-EDT "
                "voxels still uncovered; some arms may be missing from the "
                "reconstructed centreline.",
                max_arms,
                remaining,
            )
    return coords[in_tree]


#: 26-connected, matching every other label() call in this module. The
#: default (6-connected, face-only) can split a solid fat blob that is only
#: diagonally connected at a jagged mask boundary into two "components",
#: each building its own independent centreline tree with no
#: cross-component tree-adjacency check -- the two trees can then end up
#: 26-adjacent at more than one voxel near the split, producing a genuine
#: cycle in the unioned result once graph-building reads it back.
_STRUCTURE_FULL = generate_binary_structure(3, 3)


def skeletonize_edt_ridge(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    *,
    use_memmap: bool = False,
) -> np.ndarray:
    """Centreline tree per connected component: every fat arm, not a medial sheet.

    Distances, costs and the tree's thresholds are physical (see
    :func:`_relative_spacing`); without *voxel_size_zyx* every voxel is a cube.

    *use_memmap* (the low-RAM option) finds the components a slab at a time
    and each one's distance map from its own box padded by a voxel -- the
    nearest background to any of its voxels lies in it (see
    :func:`_join_thin_arms_to_fat_ridge`'s local radius) -- a block at a
    time when that box is large, instead of a distance transform and int32
    labels over the whole region's box. Same tree.
    """
    relative = _relative_spacing(voxel_size_zyx)
    mask = np.asarray(binary, dtype=bool)
    result = np.zeros(mask.shape, dtype=bool)
    if mask.any():
        _edt_ridge_into(result, mask, relative, low_ram=use_memmap)
    return result


def edt_ridge_voxels(
    binary: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    *,
    use_memmap: bool = False,
) -> np.ndarray:
    """:func:`skeletonize_edt_ridge`'s tree as its voxels, ``(n, ndim)`` in
    C order -- ``np.argwhere`` of the tree -- without a volume to draw it in."""
    mask = np.asarray(binary, dtype=bool)
    drawn: list[np.ndarray] = []
    if mask.any():
        _edt_ridge_into(None, mask, _relative_spacing(voxel_size_zyx), low_ram=use_memmap, drawn=drawn)
    return drawn_voxels(drawn, mask.shape)


def drawn_voxels(drawn: list[np.ndarray], shape) -> np.ndarray:
    """The voxels of the *drawn* pieces (each ``(n, ndim)``) once each, in C
    order: ``np.argwhere`` of a volume they were drawn into."""
    if not drawn:
        return np.empty((0, len(shape)), dtype=np.intp)
    keys = np.unique(np.concatenate([np.ravel_multi_index(tuple(v.T), shape) for v in drawn]))
    return np.column_stack(np.unravel_index(keys, shape)).astype(np.intp)


def _any(volume: np.ndarray) -> bool:
    """``volume.any()``, read once for a broadcast scalar (a region found
    before to be empty is one: no volume of False to read through)."""
    volume = np.asarray(volume)
    if volume.size and not any(volume.strides):
        return bool(volume.flat[0])
    return bool(volume.any())


def _edt_ridge_into(
    out: np.ndarray | None,
    mask: np.ndarray,
    relative: np.ndarray,
    *,
    low_ram: bool,
    drawn: list[np.ndarray] | None = None,
) -> None:
    """Draw :func:`skeletonize_edt_ridge`'s tree of *mask* into *out* (None:
    nowhere), and add each component's tree voxels to *drawn* when given."""
    if low_ram:
        _edt_ridge_by_component(out, mask, relative, drawn=drawn)
        return
    bbox = _foreground_bbox(mask, pad=1)
    if bbox is None:
        return
    crop = mask[bbox]
    edt = distance_transform_edt(crop, sampling=tuple(relative))
    labeled, n_labels = label(crop, structure=_STRUCTURE_FULL)
    if n_labels == 0:
        return
    logger.info(
        "skeletonize_edt_ridge: %d fat component(s) in a %s crop (%d voxels)",
        int(n_labels),
        crop.shape,
        int(crop.sum()),
    )
    out_crop = None if out is None else out[bbox]
    first = 0 if drawn is None else len(drawn)
    for component_id, slc in enumerate(find_objects(labeled), start=1):
        if slc is None:
            continue
        coords = np.argwhere(labeled[slc] == component_id)
        _draw_component_tree(
            out_crop, coords, edt[slc][tuple(coords.T)], slc, relative, component_id, int(n_labels),
            drawn=drawn,
        )
    if drawn is not None:
        origin = np.array([s.start for s in bbox], dtype=np.intp)
        drawn[first:] = [voxels + origin for voxels in drawn[first:]]


def _edt_ridge_by_component(
    out: np.ndarray | None, mask: np.ndarray, relative: np.ndarray, *, drawn: list | None = None
) -> None:
    """:func:`_edt_ridge_into` for the low-RAM option: each component's
    voxels from a slab-at-a-time labelling, and its radius map from its own
    padded box."""
    components, keys, numbers = slab_labels.voxels_by_component(
        lambda z0, z1: mask[z0:z1], mask.shape, structure=_STRUCTURE_FULL
    )
    if components.count == 0:
        return
    logger.info(
        "skeletonize_edt_ridge: %d fat component(s) in a %s volume (%d voxels)",
        components.count,
        mask.shape,
        len(keys),
    )
    order = np.argsort(numbers, kind="stable")
    bounds = np.searchsorted(numbers[order], np.arange(1, components.count + 2))
    del numbers
    for component_id in range(1, components.count + 1):
        coords = _coords_of(keys[order[bounds[component_id - 1]:bounds[component_id]]], mask.shape)
        box = components.box(component_id)
        edt_at = blockwise_edt.distances_at_points(
            lambda region: mask[region],
            _expand_slices(box, mask.shape, pad=1),
            coords,
            sampling=relative,
        )
        origin = np.array([s.start for s in box], dtype=np.intp)
        _draw_component_tree(
            out, coords - origin, edt_at, box, relative, component_id, components.count,
            drawn=drawn,
        )


def _draw_component_tree(
    out: np.ndarray | None,
    coords: np.ndarray,
    edt_at: np.ndarray,
    box: tuple[slice, slice, slice],
    relative: np.ndarray,
    component_id: int,
    n_components: int,
    *,
    drawn: list | None = None,
) -> None:
    """Draw the tree of the component at *coords* (in its tight *box* of
    *out*) into *out* (None: nowhere), and add its voxels (*out*'s
    coordinates) to *drawn* when given, logging the slow ones."""
    t_component = time.perf_counter()
    shape = tuple(int(s.stop) - int(s.start) for s in box)
    tree = _component_tree(coords, edt_at, shape, relative)
    origin = np.array([int(s.start) for s in box], dtype=np.intp)
    voxels = tree + origin
    if out is not None:
        out[tuple(voxels.T)] = True
    if drawn is not None:
        drawn.append(voxels)
    if n_components > 1 or len(coords) > 10_000:
        logger.info(
            "skeletonize_edt_ridge: component %d/%d (%d voxels) took %.2fs",
            component_id,
            int(n_components),
            len(coords),
            time.perf_counter() - t_component,
        )


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
    precomputed_graph=None,
    relative_spacing: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Inverted-EDT geodesic in a cropped boolean mask, or empty if unreachable.

    *precomputed_cost* skips the EDT (the expensive part for a large crop)
    when the caller already has one for this exact *crop*. *precomputed_graph*
    (from :func:`_build_dijkstra_graph`) goes further and also skips building
    the Dijkstra graph itself -- see :func:`_path_through_mask`'s
    *fallback_graph_fn*, cached per physically connected structure across
    every arm in it rather than rebuilt once per arm.
    """
    if not crop[local_start] or not crop[local_end]:
        return []
    if precomputed_graph is not None:
        walked = _dijkstra_parents(crop, None, local_start, precomputed_graph=precomputed_graph)
    else:
        relative = np.ones(3) if relative_spacing is None else relative_spacing
        if precomputed_cost is not None:
            cost = precomputed_cost
        else:
            edt = distance_transform_edt(crop, sampling=tuple(relative))
            cost = np.where(crop, 1.0 / (np.square(edt) + 1e-6), np.inf)
        walked = _dijkstra_parents(crop, cost, local_start, relative_spacing=relative)
    if walked is None:
        return []
    parent, fg_coords, index_of = walked
    path = _traceback(parent, fg_coords, index_of, local_start, local_end)
    if len(path) < 2:
        return []
    return path


def _box_dilation(mask: np.ndarray, reach) -> np.ndarray:
    """*mask* dilated by a box reaching ``reach[axis]`` voxels along each axis.

    One line per axis in turn: a box is the sum of its three edges, so this is
    the same dilation as one ``np.ones`` box structure. Handed that box,
    scipy's dilation allocates the *square* of its voxel count -- a 40-voxel
    corridor's 81**3 box asks for terabytes -- and that ran the thin-arm join
    out of memory on a real stack.
    """
    dilated = np.asarray(mask, dtype=bool)
    for axis, r in enumerate(reach):
        if int(r) <= 0:
            continue
        line = [1] * dilated.ndim
        line[axis] = 2 * int(r) + 1
        dilated = binary_dilation(dilated, structure=np.ones(line, dtype=bool))
    return dilated


def _path_through_mask(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    allowed: np.ndarray,
    *,
    fallback_graph_fn=None,
    relative_spacing: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Geodesic in *allowed* from *start* to *end*, falling back to a straight line.

    Corridor radii are in voxels of the finest axis and the geodesics costed
    physically (see :func:`_relative_spacing`).

    Prefers a straight line when every voxel is in *allowed*, then a dilated-line
    corridor at growing radii, then Dijkstra on the whole of *allowed*. Does not
    allocate a full-stack seeds array just to bbox those points.

    *fallback_graph_fn*, called only if every corridor attempt fails, returns
    the Dijkstra graph (from :func:`_build_dijkstra_graph`) for the whole of
    *allowed* -- the caller's chance to memoize one EDT-and-graph build across
    every arm that shares this *allowed*, instead of rebuilding an identical
    graph per arm (the dominant cost when several arms in one physically
    connected structure all need this fallback), without paying for it at
    all when (as for most arms) a corridor attempt already succeeds.
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

    relative = np.ones(3) if relative_spacing is None else np.asarray(relative_spacing, float)
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
        # A box reaching *radius* finest-axis voxels, at least one along every
        # axis -- *radius* iterations of the 26-neighbour cube on cube voxels.
        reach = [max(1, int(round(radius / s))) for s in relative]
        corridor = _box_dilation(painted, reach) & crop_allowed
        corridor[local_start] = True
        corridor[local_end] = True
        local_path = _geodesic_on_crop(
            corridor, local_start, local_end, relative_spacing=relative
        )
        if len(local_path) >= 2:
            return _shift_path(local_path, origin)

    slc, origin = _aabb_slices(start, end, allowed_b.shape, pad=2)
    local_path = _geodesic_on_crop(
        allowed_b[slc],
        tuple(int(s - o) for s, o in zip(start, origin)),
        tuple(int(e - o) for e, o in zip(end, origin)),
        relative_spacing=relative,
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
        precomputed_graph=fallback_graph_fn() if fallback_graph_fn is not None else None,
        relative_spacing=relative,
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
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_arm_extent_um: float = 4.0,
    max_bridge_distance_um: float | None = None,
    max_bridge_radius_multiple: float | None = None,
    radius_smoothing_um: float = 10.0,
    arms_reach_extent: bool = False,
    in_place: bool = False,
    use_memmap: bool = False,
) -> np.ndarray:
    """Connect each thin-vessel skeleton that leaves the fat wall to the fat ridge.

    Thin skeleton CCs that never extend beyond the wall (Lee flakes / wrapping
    mesh on the fat surface) are dropped. Joining those would draw a sheet of
    chords. A fused capillary does extend, and its Lee polyline must meet the
    ridge. "Extend" means reaching at least *min_arm_extent_um* microns from
    the fat region, measured physically: in voxels, an arm leaving along a
    coarse z axis had to reach several times further than one leaving
    in-plane before it counted as a vessel.

    *arms_reach_extent* says every thin skeleton component already reaches
    that far, so the filter would keep them all and is not measured:
    :func:`skeletonize_thickness_gated` Lee-thins only voxels at least
    *min_arm_extent_um* from the fat region, Lee keeps a subset of what it
    thins, and a component's own box sees no fat voxel the whole region does
    not -- so every voxel of every component measures at least as far here.
    *in_place* draws into *skeleton* itself instead of a copy.

    Nearest-neighbour search and the caps below are evaluated in physical
    microns (via *voxel_size_zyx*), not raw voxel-index distance -- the same
    convention as the rest of the pipeline's geometry, and required for an
    anisotropic stack where a voxel step means a different physical distance
    on each axis.

    *max_bridge_distance_um*, when given, caps how far an arm's nearest
    fat-ridge point may be (in microns) before the join is refused (arm left
    disconnected, same as when its own structure has no fat material at
    all). A thin vessel merging into a fat one should attach near where it
    actually touches it -- without a cap, the nearest-in-distance ridge
    point can be arbitrarily far along the fat trunk's own length, which is
    not the same anatomical event.

    *max_bridge_radius_multiple*, when given, derives a per-arm cap instead
    of a single fixed one: *that* multiple of the fat vessel's own local
    radius (in microns) at the candidate ridge point -- an arm attaching at
    the surface of a wide trunk is expected to bridge roughly the trunk's
    own half-width to reach its centreline, so the cap has to scale with
    that local width rather than with any fixed constant (a thin capillary
    and a major trunk have very different radii, and a cap sized for the
    smaller one rejects every legitimate attachment to the larger). When
    both this and *max_bridge_distance_um* are given, the tighter of the
    two applies.

    The local radius behind *max_bridge_radius_multiple* is not read as a
    single raw sample at the candidate point -- it is the maximum radius
    within *radius_smoothing_um* microns of it in every direction. The
    ridge can pass through a transient narrow waist between two wider
    lobes, or taper toward the fat/thin classification threshold at its own
    tail end; sampling exactly there would otherwise report an
    unrepresentatively small radius even though the trunk is genuinely wide
    again a short distance further along the very same ridge.

    Arms are joined one at a time, and each aims at the fat ridge first --
    the ridge as it was before any arm was joined -- so its bridge runs from
    the wall to the centreline, the way a capillary opens into a vessel.
    Only an arm whose nearest ridge point is past the cap joins the nearest
    earlier bridge instead (within the cap), so an arm that can reach the
    network only through another arm's opening is still joined. Aimed at
    whatever was nearest, the second of two capillaries entering side by
    side joined the first one's bridge near the wall, and the bridges ran
    beside each other and the ridge inside the fat lumen: on E14.5 MCA, 21
    of the 33 pairs of edges in one lumen once the thick-vessel split
    (``graph.insert_thick_vessel_junction_nodes``) cut them out.

    No volume-sized array is made beyond the copy *in_place* skips: the
    thin components, the region's, and the mask's are labelled a slab of
    z-planes at a time (:mod:`.slab_labels`) and kept only at the voxels
    asked about; the fat skeleton is read box by box from its voxel list.
    In RAM a slab is the whole volume -- ``label``'s own call. *use_memmap*
    (the low-RAM option) labels slabs of
    ``memmap_support.LOW_MEMORY_BLOCK_VOXELS`` instead, and measures each
    distance from blocks around the voxels asked about
    (:mod:`.blockwise_edt`) rather than a transform of a whole component's
    box. Same result either way.
    """
    result = skeleton if in_place else np.array(skeleton, dtype=bool)
    thick_b = np.asarray(thick, dtype=bool)
    allowed_b = np.asarray(allowed, dtype=bool)
    shape = tuple(result.shape)
    scale = np.array(voxel_size_zyx, dtype=np.float64)
    relative_scale = _relative_spacing(scale)
    block = None if use_memmap else max(1, int(np.prod(shape)))
    structure = _STRUCTURE_FULL

    fat_keys = _keys_where(
        lambda z0, z1: result[z0:z1] & thick_b[z0:z1], shape, block_voxels=block
    )
    if fat_keys.size == 0:
        return result
    thin_components, thin_keys, thin_numbers = slab_labels.voxels_by_component(
        lambda z0, z1: result[z0:z1] & ~thick_b[z0:z1], shape, structure=structure,
        block_voxels=block,
    )
    if thin_keys.size == 0 or thin_components.count == 0:
        return result
    n_labels = thin_components.count
    thin_coords = _coords_of(thin_keys, shape)
    del thin_keys

    # The physically connected structures of thin skeleton and fat region
    # together, at every voxel of either: the length filter scopes each
    # arm's distance to its own one, and the bridge cap reads the fat
    # vessel's radius over the one its candidate point is in.
    #
    # Under the low-RAM option the numbers are looked up from the start at
    # the skeleton's voxels only, and at a voxel a bridge adds later from
    # its slab labelled again (slab_labels.NumbersAnywhere): looked up at
    # every fat voxel, each lookup was 16 bytes a fat voxel, two at once.
    # The structures were labelled before the flake filter and the bridges
    # changed the skeleton outside the fat region, so a slab is labelled
    # again as it was then: *roi_changed* holds the voxels whose skeleton
    # value has changed since (each one outside the fat region, where the
    # skeleton alone decides).
    roi_components = roi_at = None
    roi_changed: set[int] = set()
    plane = int(shape[1]) * int(shape[2])

    def _skeleton_changed(voxels: np.ndarray) -> None:
        if roi_at is not None and use_memmap:
            roi_changed.symmetric_difference_update(
                np.unique(np.ravel_multi_index(tuple(np.asarray(voxels).reshape(-1, 3).T), shape)).tolist()
            )

    def _roi_planes_then(z0: int, z1: int) -> np.ndarray:
        planes = np.asarray(result[z0:z1], dtype=bool) | np.asarray(thick_b[z0:z1], dtype=bool)
        here = [key - z0 * plane for key in roi_changed if z0 * plane <= key < z1 * plane]
        if here:
            flat = planes.reshape(-1)
            flat[here] = ~flat[here]
        return planes

    if not arms_reach_extent or max_bridge_radius_multiple is not None:
        if use_memmap:
            roi_components, roi_known = slab_labels.numbers_at(
                lambda z0, z1: result[z0:z1] | thick_b[z0:z1],
                lambda z0, z1: result[z0:z1],
                shape,
                structure=structure,
                block_voxels=block,
            )
            roi_at = slab_labels.NumbersAnywhere(roi_components, roi_known, _roi_planes_then)
        else:
            roi_components, roi_keys, roi_numbers = slab_labels.voxels_by_component(
                lambda z0, z1: result[z0:z1] | thick_b[z0:z1], shape, structure=structure,
                block_voxels=block,
            )
            roi_at = slab_labels.ComponentLookup(
                roi_keys, roi_numbers.astype(roi_components.number_dtype(), copy=False), shape
            )
            del roi_keys, roi_numbers

    if arms_reach_extent:
        arm_ids = np.arange(1, n_labels + 1)
    else:
        # Per physically connected structure (thin + thick together), not one
        # EDT over their shared bounding box: real fat/thin material can span
        # most of a large stack, so that box is close to the whole image, and a
        # dense EDT that size is what actually ran the process out of memory (a
        # 287x512x512 stack: "Unable to allocate 1.35 GiB for an array with
        # shape (180948686,)"). Each connected roi component is a sliver of
        # that by comparison -- same reasoning as the per-arm join scoping
        # below, applied to this earlier length-filter measurement too.
        roi_ids_at_thin = roi_at[tuple(thin_coords.T)]
        dists = np.full(len(thin_coords), np.inf, dtype=np.float64)
        for roi_id in np.unique(roi_ids_at_thin):
            if roi_id == 0:
                continue
            slc = roi_components.box(int(roi_id))
            here = roi_ids_at_thin == roi_id
            if not thick_b[slc].any():
                # No fat material anywhere in this thin arm's own connected
                # structure: it cannot be measured against a wall it has none
                # of. Leave it at +inf, so it clears the length filter and is
                # kept as-is -- the per-arm join loop below reaches the same
                # "nothing to join to" conclusion for exactly this case.
                continue
            if use_memmap:
                dists[here] = blockwise_edt.distances_at_points(
                    lambda region: ~np.asarray(thick_b[region], dtype=bool),
                    slc,
                    thin_coords[here],
                    sampling=scale,
                )
                continue
            origin = np.array([int(s.start) for s in slc], dtype=int)
            dist_crop = distance_transform_edt(~thick_b[slc], sampling=tuple(scale))
            local = thin_coords[here] - origin
            dists[here] = dist_crop[tuple(local.T)]

        max_dist = np.zeros(int(n_labels) + 1, dtype=np.float64)
        np.maximum.at(max_dist, thin_numbers, dists)
        arm_ids = np.flatnonzero(max_dist >= float(min_arm_extent_um))
        arm_ids = arm_ids[arm_ids > 0]
        is_arm = np.zeros(int(n_labels) + 1, dtype=bool)
        is_arm[arm_ids] = True
        dropped = ~is_arm[thin_numbers]
        if dropped.any():
            result[tuple(thin_coords[dropped].T)] = False
            _skeleton_changed(thin_coords[dropped])
    if arm_ids.size == 0:
        return result
    # Each arm's voxels, in C order, as its own labelled box would list them.
    by_arm = np.argsort(thin_numbers, kind="stable")
    arm_bounds = np.searchsorted(thin_numbers[by_arm], np.arange(1, n_labels + 2))

    fat_coords = _coords_of(fat_keys, shape)
    # Which physically connected structure (of the real mask, not just the
    # fat/thin split) each voxel belongs to. A thin arm's Euclidean-nearest
    # fat voxel can sit in a different, merely-nearby vessel network that it
    # is not actually connected to at all -- no path can ever exist between
    # them, full-mask search or not. Scoping the nearest-neighbour search to
    # fat voxels sharing the arm's own component rules that out up front.
    # Looked up at every arm and fat-region voxel: an arm's own, the ridge's,
    # and any a bridge adds.
    allowed_components, allowed_at = slab_labels.numbers_at(
        lambda z0, z1: allowed_b[z0:z1],
        (lambda z0, z1: result[z0:z1]) if use_memmap else (lambda z0, z1: result[z0:z1] | thick_b[z0:z1]),
        shape,
        structure=structure,
        block_voxels=block,
    )
    if use_memmap:
        # A bridge's voxels in the fat region are read where they land.
        allowed_at = slab_labels.NumbersAnywhere(
            allowed_components, allowed_at, lambda z0, z1: allowed_b[z0:z1]
        )
    fat_component_labels = allowed_at[tuple(fat_coords.T)]
    fat_kdt = cKDTree(fat_coords.astype(np.float64, copy=False) * scale)
    # The ridge and every bridge voxel inside the fat region so far: exactly
    # the skeleton's fat voxels, as long as each bridge is drawn only once
    # its walk (which must see the region as it was before) is over.
    fat_now = _Both(result, thick_b)
    fat_skel = _VoxelBoxes(fat_keys, shape)

    # What each arm aims at first: the ridge before any arm was joined (no
    # arm voxel is fat, so that is fat_skel), per physically connected
    # structure -- built once per structure, not once per arm. fat_coords
    # grows with every bridge (a new array each time), these never do.
    ridge_coords = fat_coords
    ridge_component_labels = fat_component_labels
    ridge_trees: dict[int, tuple[np.ndarray, cKDTree] | None] = {}

    def _ridge_tree_for(label_key: int):
        if label_key not in ridge_trees:
            coords = ridge_coords[ridge_component_labels == label_key]
            ridge_trees[label_key] = (
                (coords, cKDTree(coords.astype(np.float64, copy=False) * scale))
                if len(coords)
                else None
            )
        return ridge_trees[label_key]

    def _nearest_point(points_um: np.ndarray, coords: np.ndarray, tree: cKDTree):
        """``(index into the arm's points, distance in microns, target voxel)``
        of the arm point nearest any of *coords*."""
        distance, index = tree.query(points_um, k=1)
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        nearest = int(np.argmin(distance))
        target = tuple(int(v) for v in coords[int(index[nearest])])
        return nearest, float(distance[nearest]), target

    # Memoized per physically-connected structure: the fallback's EDT and
    # Dijkstra graph depend only on that structure's shape, not on which arm
    # needed it, so a component with many arms needing the fallback (a real
    # fused sub-network, not a handful of capillaries) pays for building
    # them once instead of once per arm -- measured at ~90% of the fallback's
    # per-arm cost on a large component (see _build_dijkstra_graph).
    graph_cache: dict[int, tuple] = {}

    def _fallback_graph_for(label_key: int, local_mask: np.ndarray) -> tuple:
        cached = graph_cache.get(label_key)
        if cached is not None:
            return cached
        if use_memmap:
            built = _structure_graph(label_key)
        else:
            local_edt = distance_transform_edt(local_mask, sampling=tuple(relative_scale))
            cost = np.where(local_mask, 1.0 / (np.square(local_edt) + 1e-6), np.inf)
            built = _build_dijkstra_graph(local_mask, cost, relative_scale)
        graph_cache[label_key] = built
        return built

    def _structure_graph(label_key: int) -> tuple:
        """The fallback's graph for the low-RAM option: the structure's own
        voxels, not every voxel of its box, each costed from its exact
        distance in the box (a block at a time) -- no distance map of the
        box, and nothing for the box's other structures. Those are never
        reached from the arm, so the walk is the same step for step."""
        box = allowed_components.box(label_key)
        origin = np.array([int(s.start) for s in box], dtype=np.int64)
        plane = int(shape[1]) * int(shape[2])
        keys = []
        for z0, z1, labels, offset in allowed_components.replay(lambda a, b: allowed_b[a:b]):
            if z1 <= box[0].start or z0 >= box[0].stop:
                continue
            mine = np.flatnonzero(allowed_components.numbers(labels, offset).reshape(-1) == label_key)
            keys.append(mine.astype(np.int64) + int(z0) * plane)
        coords = _coords_of(np.concatenate(keys), shape)
        distance = blockwise_edt.distances_at_points(
            lambda region: allowed_b[region], box, coords, sampling=relative_scale
        )
        cost = 1.0 / (np.square(distance) + 1e-6)
        box_shape = tuple(int(s.stop) - int(s.start) for s in box)
        return _build_dijkstra_graph_from_coords(coords - origin, box_shape, cost, relative_scale)

    # Memoized per roi component (see the flake-length filter above, which
    # scopes its own EDT the same way): the fat vessel's own local radius at
    # a candidate ridge point, in microns -- how far a surface attachment
    # there is expected to have to bridge to reach the centreline.
    #
    # Smoothed by a maximum filter over a small physical neighbourhood
    # (radius_smoothing_um in every direction) rather than read as a single
    # raw sample: the ridge can run through a transient narrow waist -- a
    # brief neck between two wider lobes, or the tail end of the fat body's
    # own reconstruction tapering toward the classification threshold --
    # and a candidate landing exactly there would otherwise report an
    # unrepresentatively small radius even though the trunk is genuinely
    # wide again a few microns further along. The window only needs to
    # span a typical neck, not reach into an unrelated part of the network,
    # so this stays a local correction, not a way to see a wide vessel from
    # far away.
    radius_crop_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    radius_at_cache: dict[tuple[int, int, int], float] = {}
    # Each roi box's fat surface, once a window deep inside a wide vessel
    # needed it (see blockwise_edt.distances_at_points).
    radius_surface_cache: dict = {}
    footprint = tuple(
        max(1, 2 * int(np.ceil(radius_smoothing_um / max(s, 1e-6))) + 1) for s in scale
    )

    def _local_fat_radius_um_at(voxel: tuple[int, int, int]) -> float:
        roi_id = int(roi_at[voxel])
        if roi_id == 0:
            return 0.0
        # Padded by 1 beyond the component's own tight bounding box: a
        # solid component's real surface can coincide exactly with that
        # box's edge (nothing of this label exists past it, by
        # definition of "tight"), so the unpadded crop can be entirely
        # foreground there with no background left to measure distance
        # to. One voxel of padding always reveals that surface -- the
        # true nearest background to any interior point is either
        # already inside the tight box (an internal cavity) or exactly
        # one step beyond its edge; nothing needs more.
        padded_slc = _expand_slices(roi_components.box(roi_id), shape, pad=1)
        if use_memmap:
            return _windowed_fat_radius_um(voxel, padded_slc)
        cached = radius_crop_cache.get(roi_id)
        if cached is None:
            padded_origin = np.array([int(s.start) for s in padded_slc], dtype=int)
            crop = distance_transform_edt(
                thick_b[padded_slc], sampling=tuple(scale)
            )
            smoothed = maximum_filter(crop, size=footprint).astype(
                np.float32, copy=False
            )
            cached = (padded_origin, smoothed)
            radius_crop_cache[roi_id] = cached
        origin, crop = cached
        local = tuple(int(v) for v in (np.array(voxel) - origin))
        return float(crop[local])

    def _windowed_fat_radius_um(voxel, padded_slc) -> float:
        """The same reading from the window alone: the maximum filter's
        value at *voxel* is the largest radius in the window around it --
        clipped to the padded box, which is where scipy's reflection at the
        box's faces takes it -- and only fat voxels have a radius above 0."""
        cached = radius_at_cache.get(voxel)
        if cached is None:
            window = tuple(
                slice(max(int(p.start), int(v) - f // 2), min(int(p.stop), int(v) + f // 2 + 1))
                for p, v, f in zip(padded_slc, voxel, footprint)
            )
            points = np.argwhere(np.asarray(thick_b[window], dtype=bool))
            points += np.array([int(w.start) for w in window])
            radii = blockwise_edt.distances_at_points(
                lambda region: thick_b[region],
                padded_slc,
                points,
                sampling=scale,
                surface_cache=radius_surface_cache,
            )
            cached = float(np.float32(radii.max())) if len(radii) else 0.0
            radius_at_cache[voxel] = cached
        return cached

    def _bridge_cap_um(end: tuple[int, int, int]) -> float | None:
        """How long a bridge ending at *end* may be: the tighter of the fixed
        cap and the radius multiple there, None when neither is set."""
        cap = max_bridge_distance_um
        if max_bridge_radius_multiple is not None:
            radius_cap = float(max_bridge_radius_multiple) * _local_fat_radius_um_at(end)
            cap = radius_cap if cap is None else min(cap, radius_cap)
        return cap

    logger.info(
        "_join_thin_arms_to_fat_ridge: joining up to %d thin-arm components to the "
        "fat ridge",
        int(arm_ids.size),
    )
    join_step_start = time.perf_counter()
    last_log_time = join_step_start
    for arm_index, component_id in enumerate(arm_ids, start=1):
        pts = thin_coords[by_arm[arm_bounds[component_id - 1]:arm_bounds[component_id]]]
        if pts.size == 0:
            continue
        padded = _expand_slices(thin_components.box(int(component_id)), shape, pad=1)
        origin_p = np.array([int(s.start) for s in padded], dtype=int)
        component = np.zeros(tuple(int(s.stop) - int(s.start) for s in padded), dtype=bool)
        component[tuple((pts - origin_p).T)] = True
        if (binary_dilation(component, structure=structure) & fat_now[padded]).any():
            continue
        arm_component_label = int(allowed_at[tuple(pts[0])])
        same_component = fat_component_labels == arm_component_label
        if not np.any(same_component):
            # No fat ridge material anywhere in this arm's own physically
            # connected structure: it is either a fully self-contained small
            # network with no fat trunk of its own (already complete, needs
            # no joining) or genuinely isolated. Either way there is nothing
            # to join it to -- keep it as already-drawn Lee output, and do
            # not manufacture a connection to an unrelated nearby network.
            continue
        arm_points_um = pts.astype(np.float64, copy=False) * scale
        # The ridge first; the nearest of everything already connected (the
        # ridge and every earlier bridge) only when the ridge is past the cap.
        # The walk below stops on what the arm aimed at.
        ridge = _ridge_tree_for(arm_component_label)
        stop_on = fat_skel
        if ridge is not None:
            nearest, nearest_distance, end = _nearest_point(arm_points_um, *ridge)
            effective_cap = _bridge_cap_um(end)
        if ridge is None or (effective_cap is not None and nearest_distance > effective_cap):
            scoped_fat_coords = fat_coords[same_component]
            scoped_kdt = (
                fat_kdt
                if same_component.all()
                else cKDTree(scoped_fat_coords.astype(np.float64, copy=False) * scale)
            )
            nearest, nearest_distance, end = _nearest_point(
                arm_points_um, scoped_fat_coords, scoped_kdt
            )
            effective_cap = _bridge_cap_um(end)
            stop_on = fat_now
        start = tuple(int(v) for v in pts[nearest])
        if effective_cap is not None and nearest_distance > effective_cap:
            logger.info(
                "Arm's nearest fat ridge point is %.1f microns away, past "
                "the %.1f-micron bridge cap; left disconnected (a thin "
                "vessel should merge into a fat vessel near where it "
                "actually touches it, not bridge to an arbitrarily distant "
                "point on the ridge).",
                nearest_distance,
                effective_cap,
            )
            continue
        # Search only this arm's own physically connected structure, not the
        # whole image: _path_through_mask's own local searches are already
        # small, but its last-resort fallback runs a full EDT + Dijkstra over
        # everything passed to it, and one connected structure is a sliver of
        # a real multi-vessel stack. Without this, that fallback firing even
        # a handful of times on a real image is what actually froze the run,
        # not the join logic itself.
        component_slc = allowed_components.box(arm_component_label)
        component_origin = np.array([int(s.start) for s in component_slc], dtype=int)
        local_allowed = allowed_b[component_slc]
        local_start = tuple(int(v) for v in (np.array(start) - component_origin))
        local_end = tuple(int(v) for v in (np.array(end) - component_origin))
        local_path = _path_through_mask(
            local_start,
            local_end,
            local_allowed,
            fallback_graph_fn=lambda: _fallback_graph_for(
                arm_component_label, local_allowed
            ),
            relative_spacing=relative_scale,
        )
        path = [
            tuple(int(v) for v in (np.array(voxel) + component_origin))
            for voxel in local_path
        ]
        # The walk stops on first touching what the arm aimed at as it was
        # before this arm -- the ridge alone, for an arm aimed at the ridge,
        # not an earlier bridge it passes -- so this arm's voxels are drawn
        # only once it is over (fat_now reads them from the skeleton).
        drawn: list[tuple[int, int, int]] = []
        for voxel in path:
            if not allowed_b[voxel]:
                continue
            drawn.append(voxel)
            if _touches_tree(stop_on, voxel):
                break
        if use_memmap and roi_at is not None:
            _skeleton_changed(np.array(
                [voxel for voxel in drawn if not result[voxel] and not thick_b[voxel]], dtype=np.intp
            ))
        for voxel in drawn:
            result[voxel] = True
        # This join's own bridge becomes something a later arm can fall back
        # on, when its ridge is past the cap (see above): an arm that can
        # reach the network only through another arm's opening. Every voxel
        # this arm could have added to the fat set is already known from
        # `drawn` -- no need to recompute `result & thick_b` or re-argwhere
        # it (both full-image scans) to find out what changed.
        newly_fat = [voxel for voxel in drawn if thick_b[voxel]]
        if newly_fat:
            new_coords = np.asarray(newly_fat, dtype=np.intp)
            fat_coords = np.concatenate([fat_coords, new_coords], axis=0)
            fat_component_labels = np.concatenate(
                [fat_component_labels, allowed_at[tuple(new_coords.T)]]
            )
            fat_kdt = cKDTree(fat_coords.astype(np.float64, copy=False) * scale)
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


@contextmanager
def _hole_filled_mask(
    binary: np.ndarray, fill_mask_holes: bool, *, use_memmap: bool, memmap_directory
) -> Iterator[np.ndarray]:
    """*binary* as bool, its enclosed holes filled when *fill_mask_holes*,
    for the ``with`` block.

    Under *use_memmap* the filled copy is a disk-backed array, released when
    the block ends -- in RAM it was one more byte a voxel of the whole volume.
    """
    mask = np.asarray(binary, dtype=bool)
    bbox = _foreground_bbox(mask, pad=0) if fill_mask_holes else None
    if bbox is None:
        yield mask
        return
    filled = fill_binary_holes(mask[bbox], use_memmap=use_memmap, memmap_directory=memmap_directory)
    owned = None
    try:
        if use_memmap:
            owned = memmap_support.new_memmap_array(mask.shape, bool, directory=memmap_directory)
            memmap_support.map_by_slab(mask, lambda slab: slab, owned)
            memmap_support.map_by_slab(filled, lambda slab: slab, owned[bbox])
            mask = owned
        else:
            mask = mask.copy()
            mask[bbox] = filled
    except BaseException:
        if owned is not None:
            memmap_support.release_memmap_array(owned)
        raise
    finally:
        if isinstance(filled, np.memmap):
            memmap_support.release_memmap_array(filled)
    try:
        yield mask
    finally:
        if owned is not None:
            memmap_support.release_memmap_array(owned)


def thick_vessel_region(
    binary: np.ndarray,
    *,
    min_radius_um: float = THICK_VESSEL_MIN_RADIUS_UM,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_mask_holes: bool = True,
    wall_absorption_um: float | None = None,
    restrict_thick_to_mask: np.ndarray | None = None,
    use_memmap: bool = False,
    memmap_directory=None,
) -> np.ndarray | None:
    """The thick-vessel region :func:`skeletonize_thickness_gated` returns
    with ``return_thick_mask=True``, without skeletonising anything.

    For a resumed run that needs the region (where a hand-drawn vessel
    opens into a thick one) but only has the segmentation: the same hole
    filling and :func:`thick_vessel_object_mask` call, so the same voxels.
    ``None`` where that function returns ``None`` too -- *min_radius_um* 0,
    or no thick region found.
    """
    if float(min_radius_um) <= 0.0:
        return None
    with _hole_filled_mask(
        binary, fill_mask_holes, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as mask:
        thick = thick_vessel_object_mask(
            mask,
            min_radius_um=float(min_radius_um),
            voxel_size_zyx=voxel_size_zyx,
            wall_absorption_um=wall_absorption_um,
            restrict_to_mask=restrict_thick_to_mask,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    return thick if thick.any() else None


def _lee_thin_catchment_by_block(
    result: np.ndarray,
    mask: np.ndarray,
    thick: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_arm_extent_um: float,
    lee_kwargs: dict,
) -> int:
    """For the low-RAM option, :func:`skeletonize_thickness_gated`'s Lee step:
    the thin catchment's voxels at least *min_arm_extent_um* from the fat
    region -- each decided from its block's own distance transform with that
    much context -- written to disk, Lee-thinned, and drawn into *result*.
    Returns how many voxels the thin catchment has."""
    shape = mask.shape
    bbox = _foreground_bbox_by_slab(
        lambda z0, z1: np.asarray(mask[z0:z1], dtype=bool) | np.asarray(thick[z0:z1], dtype=bool),
        shape,
        pad=1,
    )
    if bbox is None:
        return 0
    crop_shape = tuple(int(s.stop) - int(s.start) for s in bbox)
    reach = float(min_arm_extent_um)
    block_voxels, workers = blockwise_edt.parallel_plan(crop_shape)
    lee_input = memmap_support.new_memmap_array(
        crop_shape, bool, directory=lee_kwargs["memmap_directory"]
    )
    n_thin = 0
    try:
        n_lee = 0
        for core, (bits, n_where, n_reaching) in blockwise_edt.near_blocks(
            _outside_thick, _thin_catchment, _reaching, (mask, thick), crop_shape,
            sampling=tuple(float(v) for v in voxel_size_zyx), reach=reach,
            params=dict(origin=tuple(int(s.start) for s in bbox), reach=reach),
            block_voxels=block_voxels, workers=workers,
            memmap_directory=lee_kwargs["memmap_directory"],
        ):
            lee_input[core] = blockwise_edt.unpack(bits, tuple(c.stop - c.start for c in core))
            n_thin += n_where
            n_lee += n_reaching
        if n_lee:
            skeleton = skeletonize_by_component(lee_input, **lee_kwargs)
            try:
                target = result[bbox]
                step = memmap_support.slab_step(target.shape)
                for z0 in range(0, target.shape[0], step):
                    target[z0:z0 + step] |= np.asarray(skeleton[z0:z0 + step], dtype=bool)
            finally:
                if isinstance(skeleton, np.memmap):
                    memmap_support.release_memmap_array(skeleton)
    finally:
        memmap_support.release_memmap_array(lee_input)
    return n_thin


def skeletonize_thickness_gated(
    binary: np.ndarray,
    *,
    min_radius_um: float = THICK_VESSEL_MIN_RADIUS_UM,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_mask_holes: bool = True,
    wall_absorption_um: float | None = None,
    restrict_thick_to_mask: np.ndarray | None = None,
    flake_filter_um: float | None = None,
    max_bridge_radius_multiple: float | None = None,
    max_bridge_distance_um: float | None = None,
    bridge_radius_smoothing_um: float = 10.0,
    return_thick_mask: bool = False,
    use_memmap: bool = False,
    memmap_directory=None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
    thick_region: np.ndarray | None = None,
    centreline_voxels: list | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
    """Lee on the thin catchment; an EDT-ridge tree (every arm) inside the fat catchment.

    *thick_region*, when given, is the fat catchment found before from this
    mask and these arguments (:func:`thick_vessel_region`'s answer, all
    False where it found none -- see
    :mod:`~haemolynx.preprocessing.thick_vessel_region_cache`), and finding
    it, the slow part, is skipped. Nothing else reads it.

    ``min_radius_um <= 0`` is the current pipeline behaviour: Lee on the whole
    mask. Capillaries fused into the same object never enter the ridge path
    unless their own EDT peak crosses the threshold.

    *wall_absorption_um* is how far around the fat body's core the wall
    absorbs surface roughness (and, unavoidably, the first stretch of any
    real vessel fused directly onto it) into the catchment; see
    :func:`thick_vessel_object_mask`. ``None`` derives it as half of
    *min_radius_um*.

    *restrict_thick_to_mask*, when given, is forwarded to
    :func:`thick_vessel_object_mask`'s own *restrict_to_mask* -- every
    foreground voxel inside this mask becomes fat directly, with no
    *min_radius_um* test at all. A caller with independent ground truth
    for where the real large vessels are (e.g. a known large/small vessel
    mask) does not need local radius to agree first: a real vessel can
    measure narrower than *min_radius_um* everywhere in the image and
    still genuinely be the vessel the mask says it is. If the restriction
    leaves no fat region at all, this falls back to plain Lee on the whole
    mask, the same path taken when no fat trunk is present.

    *flake_filter_um* is how far beyond the fat wall a thin-vessel skeleton
    fragment must reach to be kept rather than dropped as a Lee-thinning
    flake of the wall's own surface; see :func:`_join_thin_arms_to_fat_ridge`.
    ``None`` derives it as ``max(4 voxels, 0.75 * min_radius_um)``, the
    previous fixed behaviour. Together, a vessel fused to a fat trunk needs
    to reach *wall_absorption_um* + *flake_filter_um* beyond its attachment
    point to survive at all -- shortening either recovers shorter real
    vessels at the cost of letting more wall-wrap flakes through.

    *max_bridge_radius_multiple*, when given, caps how far a surviving arm
    may bridge to reach the fat ridge, as a multiple of the fat vessel's own
    *local* radius (in microns) at the candidate ridge point -- not the
    fixed *min_radius_um* threshold that only decides fat-vs-thin, since a
    major trunk's own radius is typically many times that threshold and an
    arm attaching to it must cross roughly that much distance to reach its
    centreline. ``None`` leaves the join search unbounded; see
    :func:`_join_thin_arms_to_fat_ridge`.

    *max_bridge_distance_um*, when given, is an additional fixed cap in
    microns, independent of local radius -- the tighter of the two applies
    when both are given. Useful when the radius-based cap alone is too
    tight (the candidate point sits right at the fat/thin classification
    boundary, where local radius reads close to its own minimum even for a
    trunk that is genuinely wide a short distance further along the same
    ridge) or too loose for a particular dataset; see
    :func:`_join_thin_arms_to_fat_ridge`.

    *bridge_radius_smoothing_um* widens that local-radius reading to the
    widest one found within this many microns of the candidate point, in
    every direction, rather than a single raw sample: the ridge can pass
    through a transient narrow waist between two wider lobes, or taper
    toward the fat/thin threshold at its own tail end, and an arm landing
    exactly there would otherwise see an unrepresentatively small radius
    even though the trunk is genuinely wide again a short distance further
    along the same ridge. ``0`` reproduces the raw single-point reading.

    *return_thick_mask*, when ``True``, returns ``(skeleton, thick)``
    instead of a bare array -- *thick* is ``None`` on every path that never
    computed one (``min_radius_um <= 0``, or no fat region found).

    *centreline_voxels*, a list, is given the fat region's centreline tree
    as it is drawn: its voxels in C order (:func:`edt_ridge_voxels` of
    *thick*, which graph building reads), so nothing draws it again.

    *use_memmap* (the low-RAM option) makes no distance map and no label
    volume. Every distance here is either compared with a few microns or
    needed at some voxels only, and a block's own transform with enough
    context answers both exactly (:mod:`.blockwise_edt`); components are
    labelled a slab of planes at a time (:mod:`.slab_labels`). The
    hole-filled mask and the Lee input are on disk, and Lee thinning runs
    through :func:`~haemolynx.preprocessing.skeleton.skeletonize_by_component`
    (with the ``tile_*`` settings). What stays in RAM is the thick region, a
    byte a voxel, and lists of the fat region's voxels;
    the skeleton returned is on disk (a ``new_memmap_array``). Same skeleton
    and region either way.
    """
    lee_kwargs = {
        "use_memmap": use_memmap,
        "memmap_directory": memmap_directory,
        "tile_large_components": tile_large_components,
        "tile_max_voxels": tile_max_voxels,
        "tile_halo_voxels": tile_halo_voxels,
    }
    # Tiling only applies under use_memmap (see skeletonize_by_component).
    low_ram = bool(use_memmap)

    def lee(volume: np.ndarray) -> np.ndarray:
        if not low_ram:
            return _skeletonize_foreground(volume)
        return skeletonize_by_component(volume, **lee_kwargs)

    def whole_lee(volume: np.ndarray) -> np.ndarray:
        """Lee of the whole mask; under low-RAM the one
        ``skeletonize_by_component`` wrote on disk, not a copy in RAM."""
        return lee(volume) if low_ram else skeletonize_volume(volume).astype(bool)

    with _hole_filled_mask(
        binary, fill_mask_holes, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as mask:
        if float(min_radius_um) <= 0.0:
            skeleton = whole_lee(mask)
            return (skeleton, None) if return_thick_mask else skeleton

        t0 = time.perf_counter()
        if thick_region is not None:
            thick = np.asarray(thick_region, dtype=bool)
            if thick.shape != mask.shape:
                raise ValueError(
                    f"The thick-vessel region given is {thick.shape} voxels but the mask "
                    f"is {mask.shape}: it was found in a different image."
                )
            logger.info(
                "skeletonize_thickness_gated: fat catchment found before (input shape %s, "
                "%d foreground voxels)",
                mask.shape,
                int(np.count_nonzero(mask)),
            )
        else:
            logger.info(
                "skeletonize_thickness_gated: computing fat catchment (input shape %s, "
                "%d foreground voxels)",
                mask.shape,
                int(np.count_nonzero(mask)),
            )
            thick = thick_vessel_object_mask(
                mask,
                min_radius_um=float(min_radius_um),
                voxel_size_zyx=voxel_size_zyx,
                wall_absorption_um=wall_absorption_um,
                restrict_to_mask=restrict_thick_to_mask,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        t_catchment = time.perf_counter() - t0
        if not _any(thick):
            skeleton = whole_lee(mask)
            return (skeleton, None) if return_thick_mask else skeleton
        # thick_vessel_object_mask's own large locals are unreachable now
        # that it has returned, but a big freed block does not always get
        # handed back to the OS the moment its refcount drops; collecting
        # explicitly before the next big allocation (Lee-thinning, then the
        # ridge tree, then joining) gives it the best chance to.
        gc.collect()

        # In microns, compared against a distance measured in microns: this was a
        # voxel count against a voxel-index distance, so on an anisotropic stack
        # an arm leaving the trunk along the coarse z axis had to reach several
        # times further than one leaving in-plane before it was kept.
        spacing = min(float(v) for v in voxel_size_zyx)
        if flake_filter_um is None:
            min_arm_extent_um = max(4.0 * spacing, 0.75 * float(min_radius_um))
        else:
            min_arm_extent_um = max(0.0, float(flake_filter_um))
        n_thick = int(np.count_nonzero(thick))
        # Under low-RAM the skeleton is drawn on disk: held through the rest
        # of skeletonisation, it was a byte a voxel of RAM.
        result = (
            memmap_support.new_memmap_array(mask.shape, bool, directory=memmap_directory)
            if low_ram
            else np.zeros(mask.shape, dtype=bool)
        )
        logger.info(
            "skeletonize_thickness_gated: fat catchment done in %.2fs (%d fat voxels); "
            "Lee-thinning the thin catchment",
            t_catchment,
            n_thick,
        )
        t1 = time.perf_counter()
        # Do not Lee the leftover fat-wall shell: those CCs never extend away
        # from thick, and Lee of that wrap is the looped mesh beside the ridge.
        if low_ram:
            n_thin = _lee_thin_catchment_by_block(
                result,
                mask,
                thick,
                voxel_size_zyx=voxel_size_zyx,
                min_arm_extent_um=min_arm_extent_um,
                lee_kwargs=lee_kwargs,
            )
        else:
            thin = mask & ~thick
            n_thin = int(thin.sum())
            if n_thin:
                bbox = _foreground_bbox(thin | thick, pad=1)
                if bbox is None:
                    result |= lee(thin)
                else:
                    dist_crop = distance_transform_edt(
                        ~thick[bbox], sampling=tuple(float(v) for v in voxel_size_zyx)
                    )
                    lee_crop = thin[bbox] & (dist_crop >= float(min_arm_extent_um))
                    del dist_crop
                    if lee_crop.any():
                        result[bbox] |= lee(lee_crop)
            del thin
        t_lee = time.perf_counter() - t1
        gc.collect()
        logger.info(
            "skeletonize_thickness_gated: Lee-thinning done in %.2fs (%d thin voxels); "
            "building the fat-catchment centreline tree",
            t_lee,
            n_thin,
        )
        t2 = time.perf_counter()
        drawn = [] if centreline_voxels is not None else None
        _edt_ridge_into(result, thick, _relative_spacing(voxel_size_zyx), low_ram=low_ram, drawn=drawn)
        if centreline_voxels is not None:
            centreline_voxels.append(drawn_voxels(drawn, thick.shape))
        del drawn
        t_ridge = time.perf_counter() - t2
        gc.collect()
        logger.info(
            "skeletonize_thickness_gated: centreline tree done in %.2fs; joining thin "
            "arms to the fat ridge",
            t_ridge,
        )
        t3 = time.perf_counter()
        # Every thin skeleton voxel is Lee output of the catchment cut to
        # min_arm_extent_um from the fat region, so the join's length filter
        # would keep every arm (see its arms_reach_extent).
        result = np.asanyarray(
            _join_thin_arms_to_fat_ridge(
                result,
                thick,
                mask,
                voxel_size_zyx=voxel_size_zyx,
                min_arm_extent_um=min_arm_extent_um,
                max_bridge_radius_multiple=max_bridge_radius_multiple,
                max_bridge_distance_um=max_bridge_distance_um,
                radius_smoothing_um=bridge_radius_smoothing_um,
                arms_reach_extent=True,
                in_place=True,
                use_memmap=low_ram,
            ),
            dtype=bool,
        )
        t_join = time.perf_counter() - t3
        logger.info(
            "Thickness-gated skeletonisation: shape=%s fat=%d voxels thin=%d voxels "
            "catchment=%.3fs lee=%.3fs ridge=%.3fs join=%.3fs",
            mask.shape,
            n_thick,
            n_thin,
            t_catchment,
            t_lee,
            t_ridge,
            t_join,
        )
        return (result, thick) if return_thick_mask else result


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
