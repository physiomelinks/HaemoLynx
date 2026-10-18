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
  (:func:`distances_at_points`); voxels deeper than a halo the budget can
  hold are measured to the features on the object's surface instead, the
  only ones that can be nearest.

Volume faces are where the whole volume's transform stops too, so blocks
there need nothing special. Distances are worked out from scipy's feature
transform with scipy's own arithmetic (:func:`distances_at`), only at the
voxels asked about -- the feature transform alone is about 21 bytes a voxel.
Where two features are exactly equidistant, the caveat in
:mod:`.pointwise_distance` applies.
"""
from __future__ import annotations

import itertools
import os
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

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


#: Bytes of RAM a worker holds per voxel of its padded block, allowing for a
#: dense one: scipy's feature transform is 13 (measured), the block and its
#: wanted voxels a few more, and each wanted voxel's coordinates and distance
#: ~56 -- a quarter of a block wanted.
BLOCK_BYTES_PER_VOXEL = 32
#: A worker's own interpreter and imports, single-threaded (~150 MB measured).
WORKER_BASE_BYTES = 200 << 20
#: Smaller blocks than this are mostly halo: the budget then buys fewer workers.
MIN_PARALLEL_BLOCK_VOXELS = 2_000_000
#: Volumes smaller than this are transformed in this process: starting the
#: workers (each imports the package, ~7 s) costs more than they save.
PARALLEL_MIN_VOXELS = 64_000_000
#: The share of the memory free now that the workers may take between them.
PARALLEL_MEMORY_FRACTION = 0.5


def parallel_plan(shape, block_voxels: int | None = None) -> tuple[int, int]:
    """``(block_voxels, workers)`` for :func:`near_blocks` on a volume of
    *shape*: as many workers as there are cores but one and as half the
    memory free now holds with blocks of :data:`MIN_PARALLEL_BLOCK_VOXELS`
    (at :data:`BLOCK_BYTES_PER_VOXEL`), then blocks as large as that share
    allows, up to *block_voxels*. One worker -- this process -- below
    :data:`PARALLEL_MIN_VOXELS`. The block size changes which blocks there
    are, never the answer."""
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    block_voxels = int(block_voxels)
    total = int(np.prod([int(s) for s in shape], dtype=np.int64))
    cores = max(1, (os.cpu_count() or 1) - 1)
    if total < PARALLEL_MIN_VOXELS or cores == 1:
        return block_voxels, 1
    from haemolynx.haemodynamics.resistance import available_memory_bytes

    available = available_memory_bytes()
    if available is None:
        return block_voxels, 1
    budget = PARALLEL_MEMORY_FRACTION * float(available)
    smallest = MIN_PARALLEL_BLOCK_VOXELS * BLOCK_BYTES_PER_VOXEL + WORKER_BASE_BYTES
    workers = int(min(cores, budget // smallest, -(-total // MIN_PARALLEL_BLOCK_VOXELS)))
    if workers <= 1:
        return block_voxels, 1
    block = int(min(block_voxels, (budget / workers - WORKER_BASE_BYTES) // BLOCK_BYTES_PER_VOXEL))
    return block, workers


def near_blocks(
    edt_input,
    wanted,
    decide,
    volumes: tuple[np.ndarray, ...],
    shape: tuple[int, ...],
    *,
    sampling: Sequence[float],
    reach: float,
    params=None,
    block_voxels: int | None = None,
    workers: int = 1,
    memmap_directory=None,
) -> Iterator[tuple[tuple[slice, ...], object]]:
    """``(core, decide(where, distance, params))`` for each block of
    :func:`near_distances_by_block` that holds a wanted voxel, in its order,
    with *where* and *distance* as it gives them -- the blocks transformed in
    *workers* processes when more than one.

    The volume is read through ``edt_input(volumes, box, params)`` and
    ``wanted(volumes, box, params)``, *volumes* a tuple the workers map from
    disk (:func:`~.shared_volume_tasks.iter_volume_tasks`: a whole memmap as
    it is, anything else copied once). *edt_input*, *wanted* and *decide*
    must be module-level functions, and *decide*'s result small enough to
    send back: the blocks are independent, so each is transformed and
    decided wherever it runs, identically.
    """
    from .shared_volume_tasks import iter_volume_tasks

    sampling_t = tuple(float(v) for v in sampling)
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    halo = reach_voxels(reach, sampling_t)
    cores = list(_tiles(shape, _core_extent(shape, halo, int(block_voxels))))
    payloads = [
        (edt_input, wanted, decide, core, _pad(core, halo, shape), sampling_t, params)
        for core in cores
    ]
    results = iter_volume_tasks(
        _near_block, payloads, volume=tuple(volumes), workers=int(workers),
        memmap_directory=memmap_directory,
    )
    for core, result in zip(cores, results):
        if result is not None:
            yield core, result


def _near_block(volumes, payload):
    """One block of :func:`near_blocks`: None when it holds no wanted voxel."""
    edt_input, wanted, decide, core, padded, sampling, params = payload
    where = np.asarray(wanted(volumes, core, params), dtype=bool)
    if not where.any():
        return None
    block = np.asarray(edt_input(volumes, padded, params), dtype=bool)
    if block.all():
        distance = np.full(int(np.count_nonzero(where)), np.inf)
    else:
        coords = np.argwhere(where) + np.array([c.start - p.start for c, p in zip(core, padded)])
        distance = distances_at(feature_transform(block, sampling), coords, sampling)
    return decide(where, distance, params)


def pack(block: np.ndarray) -> np.ndarray:
    """A boolean block as bits, for sending back from a worker."""
    return np.packbits(np.asarray(block, dtype=bool), axis=None)


def unpack(bits: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """:func:`pack`'s block again."""
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def distances_at_points(
    edt_input: Box,
    domain: tuple[slice, ...],
    points: np.ndarray,
    *,
    sampling: Sequence[float],
    block_voxels: int | None = None,
    reach: float | None = None,
    surface_cache: dict | None = None,
) -> np.ndarray:
    """``distance_transform_edt(edt_input(domain), sampling)`` at *points*
    (``(n, ndim)`` voxels inside the box *domain*, in the volume's own
    coordinates), without transforming more than about *block_voxels* voxels
    at once.

    A domain that fits is transformed whole. Otherwise the points are read
    in groups, each from the box round its points padded by *reach*
    (``START_REACH_VOXELS`` voxels of the finest axis when not given); a
    point is kept where it is nearer than anything past that box could be,
    and read again with twice the reach where it is not. Points still left
    once a reach no longer fits the budget -- deep inside a vessel hundreds
    of voxels wide -- are measured to the features on the domain's surface
    (:func:`_surface_distances`), kept in *surface_cache* when given so a
    caller asking about one domain again does not find them again.
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
        if not _workable(shape, halo, core, int(block_voxels)):
            break
        cell = local[todo] // core
        order = np.lexsort(cell.T[::-1])
        todo, cell = todo[order], cell[order]
        starts = np.flatnonzero(np.r_[True, np.any(np.diff(cell, axis=0) != 0, axis=1)])
        unsure = []
        for start, stop in zip(starts, np.r_[starts[1:], len(todo)]):
            idx = todo[start:stop]
            here = local[idx]
            # The box round this cell's points, not the whole cell: a few
            # points (a bridge's window) need no more than their own context.
            p_lo = np.maximum(here.min(axis=0) - halo, 0)
            p_hi = np.minimum(here.max(axis=0) + 1 + halo, shape)
            whole = bool(np.all(p_lo == 0) and np.all(p_hi == shape))
            block = np.asarray(
                edt_input(tuple(slice(int(a + o), int(b + o)) for a, b, o in zip(p_lo, p_hi, lo))),
                dtype=bool,
            )
            if block.all() and not whole:
                unsure.append(idx)
                continue
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
    if todo.size:
        out[todo] = _surface_distances(
            edt_input, domain, local[todo], sampling, int(block_voxels), surface_cache
        )
    return out


def _workable(shape, halo, core, block_voxels: int) -> bool:
    """Whether cores of *core* padded by *halo* fit the budget with each core
    at least half its halo across (or the whole axis): past that, a block is
    mostly context and each holds too few points to be worth its transform."""
    padded = int(np.prod([min(int(s), int(c) + 2 * int(h)) for s, c, h in zip(shape, core, halo)]))
    return padded <= block_voxels and all(
        int(c) >= min(int(s), max(1, (int(h) + 1) // 2)) for s, c, h in zip(shape, core, halo)
    )


def _surface_distances(
    edt_input: Box,
    domain: tuple[slice, ...],
    local: np.ndarray,
    sampling: np.ndarray,
    block_voxels: int,
    cache: dict | None,
) -> np.ndarray:
    """The transform's distances at *local* (points in *domain*'s own
    coordinates) from the features that touch a non-feature along an axis.

    The nearest feature to a voxel is always one of those: one step from it
    along any axis it is displaced on, towards the voxel, is nearer the voxel
    and inside the domain, so must be a non-feature. Found a block at a time
    and kept in a k-d tree -- a vessel's surface, not its volume.
    """
    key = tuple((int(s.start), int(s.stop)) for s in domain)
    found = None if cache is None else cache.get(key)
    if found is None:
        surface = _surface_features(edt_input, domain, block_voxels)
        found = (surface, cKDTree(surface * sampling) if len(surface) else None)
        if cache is not None:
            cache[key] = found
    surface, tree = found
    if tree is None:
        # No feature anywhere in the domain: only scipy's own whole transform
        # says what it reads there.
        block = np.asarray(edt_input(tuple(domain)), dtype=bool)
        return distances_at(feature_transform(block, sampling), local, sampling)
    _distance, nearest = tree.query(local * sampling, k=1)
    return displacement_distances(surface[np.asarray(nearest)] - local, sampling)


def _surface_features(edt_input: Box, domain: tuple[slice, ...], block_voxels: int) -> np.ndarray:
    """``(n, ndim)`` features of *domain* (in its own coordinates) with a
    non-feature beside them along an axis, a block at a time."""
    shape = [int(s.stop) - int(s.start) for s in domain]
    lo = [int(s.start) for s in domain]
    one = [1] * len(shape)
    found = []
    for core in _tiles(shape, _core_extent(shape, one, block_voxels)):
        padded = _pad(core, one, shape)
        block = np.asarray(
            edt_input(tuple(slice(p.start + o, p.stop + o) for p, o in zip(padded, lo))), dtype=bool
        )
        beside = np.zeros(block.shape, dtype=bool)
        for axis in range(block.ndim):
            ahead = [slice(None)] * block.ndim
            behind = [slice(None)] * block.ndim
            ahead[axis], behind[axis] = slice(1, None), slice(None, -1)
            beside[tuple(behind)] |= block[tuple(ahead)]
            beside[tuple(ahead)] |= block[tuple(behind)]
        inner = tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(core, padded))
        touching = ~block[inner] & beside[inner]
        found.append(np.argwhere(touching) + np.array([c.start for c in core]))
    return np.concatenate(found).astype(np.int64) if found else np.empty((0, len(shape)), np.int64)
