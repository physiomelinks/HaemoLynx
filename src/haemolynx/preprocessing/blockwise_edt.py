"""Exact distance-transform answers for a volume, a padded block at a time.

The low-RAM option's thick-vessel steps ask two things of a Euclidean
distance transform: whether each voxel is nearer or farther than a few
microns (the fat catchment's radius gates and wall, the flake filter), and
its exact value at some voxels (the centreline tree's costs, the bridge
cap's local radius). Neither needs the whole float64 map, nor the ~70 bytes
a voxel scipy takes to make it.

Both take scipy's own transform of padded blocks. A block's transform never
reads a voxel nearer than its true distance -- it sees fewer features -- and
reads the true one wherever the nearest feature lies inside the padded block:

* compared with a distance *t*, a halo of ``ceil(t / spacing) + 1`` voxels
  along each axis holds every feature nearer than *t*, so the comparison
  comes out as on the whole volume (:func:`near_distances_by_block`); a
  block with no feature in reach needs no transform at all;
* for exact values, a voxel's distance is kept once it is below its distance
  to the padded block's edge, and the halo doubles for the voxels left
  (:func:`distances_at_points`).

Volume faces are where the whole volume's transform stops too, so blocks
there need nothing special. Distances are worked out from scipy's feature
transform with scipy's own arithmetic (:func:`distances_at`), only at the
voxels asked about -- the feature transform alone is about 21 bytes a voxel.
Where two features are exactly equidistant, the caveat in
:mod:`.pointwise_distance` applies.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import memmap_support

#: ``volume(slices)``: the array ``distance_transform_edt`` would be handed,
#: over a box of the volume; its zeros are the features.
Box = Callable[[tuple[slice, ...]], np.ndarray]

#: Context, in voxels of the finest axis, :func:`distances_at_points` starts
#: from before doubling it.
START_REACH_VOXELS = 16


def feature_transform(block: np.ndarray, sampling: Sequence[float]) -> np.ndarray:
    """scipy's feature transform of *block*: the nearest zero of each voxel."""
    return distance_transform_edt(
        block, sampling=sampling, return_distances=False, return_indices=True
    )


def distances_at(features: np.ndarray, coords: np.ndarray, sampling: Sequence[float]) -> np.ndarray:
    """``distance_transform_edt``'s distances at *coords* (``(n, ndim)``, in
    the block *features* was taken of), from its feature transform by its own
    arithmetic -- the values the whole transform has there, bit for bit."""
    ndim = features.shape[0]
    coords = np.asarray(coords, dtype=np.intp).reshape(-1, ndim)
    at = tuple(coords.T)
    dt = np.empty((ndim, len(coords)), dtype=np.int64)
    for axis in range(ndim):
        dt[axis] = features[axis][at] - coords[:, axis]
    return _distances(dt, sampling)


def displacement_distances(displacement: np.ndarray, sampling: Sequence[float]) -> np.ndarray:
    """The length of each integer *displacement* (``(n, ndim)``) by
    ``distance_transform_edt``'s arithmetic: what it reads at a voxel whose
    nearest feature is that far away."""
    displacement = np.asarray(displacement, dtype=np.int64)
    return _distances(np.ascontiguousarray(displacement.reshape(-1, len(sampling)).T), sampling)


def _distances(dt: np.ndarray, sampling: Sequence[float]) -> np.ndarray:
    dt = np.asarray(dt).astype(np.float64)
    for axis in range(len(sampling)):
        dt[axis, ...] *= sampling[axis]
    np.multiply(dt, dt, dt)
    # scipy sums its (ndim, ...) array over axis 0 one axis after another,
    # (a0 + a1) + a2; spelled out, so a short array -- which numpy may sum
    # pairwise -- adds them in the same order.
    total = dt[0].copy()
    for axis in range(1, dt.shape[0]):
        total += dt[axis]
    return np.sqrt(total)


def reach_voxels(reach: float, sampling: Sequence[float]) -> np.ndarray:
    """Voxels along each axis that hold every feature within *reach*, plus one."""
    return np.ceil(float(reach) / np.asarray(sampling, dtype=np.float64)).astype(np.int64) + 1


def _core_extent(shape, halo, block_voxels: int) -> list[int]:
    """Core sizes for tiling *shape* so a core padded by *halo* holds at most
    *block_voxels* voxels -- or as close as halving cores gets, when the halo
    alone is wider than that."""
    shape = [int(s) for s in shape]
    core = list(shape)

    def padded(extents):
        return int(np.prod([min(s, c + 2 * int(h)) for s, c, h in zip(shape, extents, halo)]))

    while padded(core) > block_voxels:
        best = None
        for axis in range(len(core)):
            if core[axis] <= 1:
                continue
            trial = list(core)
            trial[axis] = (trial[axis] + 1) // 2
            if padded(trial) < padded(core) and (best is None or core[axis] > core[best]):
                best = axis
        if best is None:
            break
        core[best] = (core[best] + 1) // 2
    return core


def _tiles(shape, core) -> Iterator[tuple[slice, ...]]:
    for starts in itertools.product(*(range(0, int(s), int(c)) for s, c in zip(shape, core))):
        yield tuple(slice(a, min(a + int(c), int(s))) for a, c, s in zip(starts, core, shape))


def _pad(core: tuple[slice, ...], halo, shape) -> tuple[slice, ...]:
    return tuple(
        slice(max(0, c.start - int(h)), min(int(s), c.stop + int(h)))
        for c, h, s in zip(core, halo, shape)
    )


def near_distances_by_block(
    edt_input: Box,
    wanted: Box,
    shape: tuple[int, ...],
    *,
    sampling: Sequence[float],
    reach: float,
    block_voxels: int | None = None,
) -> Iterator[tuple[tuple[slice, ...], np.ndarray, np.ndarray]]:
    """``(core, where, distance)`` for each block of a volume of *shape* that
    holds a wanted voxel.

    *where* is ``wanted(core)``; *distance*, one value per True voxel of it in
    C order, is ``distance_transform_edt(edt_input(whole volume), sampling)``
    there wherever that is at most *reach*, and something larger than *reach*
    elsewhere (``inf`` when no feature is within reach of the block). So it
    compares with any distance up to *reach* exactly as the whole transform
    does. Only meaningful when the volume has a feature somewhere: scipy's
    transform of a volume without one is not a distance to anything.
    """
    sampling = np.asarray(sampling, dtype=np.float64)
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    halo = reach_voxels(reach, sampling)
    for core in _tiles(shape, _core_extent(shape, halo, block_voxels)):
        where = np.asarray(wanted(core), dtype=bool)
        if not where.any():
            continue
        padded = _pad(core, halo, shape)
        block = np.asarray(edt_input(padded), dtype=bool)
        if block.all():
            yield core, where, np.full(int(np.count_nonzero(where)), np.inf)
            continue
        coords = np.argwhere(where) + np.array([c.start - p.start for c, p in zip(core, padded)])
        yield core, where, distances_at(feature_transform(block, sampling), coords, sampling)


def distances_at_points(
    edt_input: Box,
    domain: tuple[slice, ...],
    points: np.ndarray,
    *,
    sampling: Sequence[float],
    block_voxels: int | None = None,
    reach: float | None = None,
) -> np.ndarray:
    """``distance_transform_edt(edt_input(domain), sampling)`` at *points*
    (``(n, ndim)`` voxels inside the box *domain*, in the volume's own
    coordinates), without transforming more than about *block_voxels* voxels
    at once.

    A domain that fits is transformed whole. Otherwise each point is first
    read from a block padded by *reach* (``START_REACH_VOXELS`` voxels of the
    finest axis when not given), kept where it is nearer than anything past
    that block could be, and read again with twice the reach where it is not.
    """
    sampling = np.asarray(sampling, dtype=np.float64)
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    lo = np.array([int(s.start) for s in domain], dtype=np.int64)
    shape = np.array([int(s.stop) - int(s.start) for s in domain], dtype=np.int64)
    local = np.asarray(points, dtype=np.int64).reshape(-1, len(domain)) - lo
    out = np.empty(len(local), dtype=np.float64)
    if not len(local):
        return out
    if int(np.prod(shape)) <= int(block_voxels):
        block = np.asarray(edt_input(tuple(domain)), dtype=bool)
        return distances_at(feature_transform(block, sampling), local, sampling)

    reach_um = float(START_REACH_VOXELS * sampling.min()) if reach is None else float(reach)
    todo = np.arange(len(local))
    while todo.size:
        halo = np.minimum(reach_voxels(reach_um, sampling), shape)
        core = np.array(_core_extent(shape, halo, int(block_voxels)), dtype=np.int64)
        cell = local[todo] // core
        order = np.lexsort(cell.T[::-1])
        todo, cell = todo[order], cell[order]
        starts = np.flatnonzero(np.r_[True, np.any(np.diff(cell, axis=0) != 0, axis=1)])
        unsure = []
        for start, stop in zip(starts, np.r_[starts[1:], len(todo)]):
            idx = todo[start:stop]
            c_lo = cell[start] * core
            c_hi = np.minimum(c_lo + core, shape)
            p_lo = np.maximum(c_lo - halo, 0)
            p_hi = np.minimum(c_hi + halo, shape)
            whole = bool(np.all(p_lo == 0) and np.all(p_hi == shape))
            block = np.asarray(
                edt_input(tuple(slice(int(a + o), int(b + o)) for a, b, o in zip(p_lo, p_hi, lo))),
                dtype=bool,
            )
            if block.all() and not whole:
                unsure.append(idx)
                continue
            here = local[idx]
            distance = distances_at(feature_transform(block, sampling), here - p_lo, sampling)
            # Nothing past the padded block is nearer than its edge: a
            # feature there is a step beyond an inner face of the block.
            bound = np.full(len(idx), np.inf)
            for axis in range(len(shape)):
                if p_lo[axis] > 0:
                    bound = np.minimum(bound, (here[:, axis] - p_lo[axis] + 1) * sampling[axis])
                if p_hi[axis] < shape[axis]:
                    bound = np.minimum(bound, (p_hi[axis] - here[:, axis]) * sampling[axis])
            # Strictly nearer, with room for rounding, so no feature past the
            # block can be one the whole transform would have tied with it.
            sure = distance < bound * (1.0 - 1e-12)
            out[idx[sure]] = distance[sure]
            unsure.append(idx[~sure])
        todo = np.concatenate(unsure) if unsure else np.empty(0, dtype=np.int64)
        reach_um *= 2.0
    return out
