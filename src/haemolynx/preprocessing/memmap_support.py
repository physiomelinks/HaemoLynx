"""Disk-backed arrays for volumes too large to hold in RAM.

``use_memmap_loading`` trades RAM for disk and speed on a machine that
cannot hold a whole image volume outright -- decompressing a TIFF/H5 file
into a dense array costs exactly ``shape * dtype.itemsize`` bytes regardless
of how well the *file* compresses, and several preprocessing steps need an
extra same-size (or wider-dtype) buffer on top of that. tifffile already has
its own built-in, self-cleaning memmap support
(``TiffFile.asarray(out="memmap")``, used directly in :mod:`haemolynx.io.load`);
this module gives the H5 loader and the full-volume intermediate arrays that
matter most here in ``preprocessing`` (connected-component labels, distance
transforms) the same thing. Lives in ``preprocessing`` rather than ``io``
because ``preprocessing`` has no dependency on ``io`` (``io`` imports *from*
``preprocessing``, never the other way -- see ``segmentation_cleanup``'s own
module docstring) and this is needed on both sides of that boundary.

Windows will not delete a file that is still memory-mapped -- unlike POSIX,
where an unlinked-but-open file just stays around until the last handle
closes. Waiting for Python's garbage collector to drop the last reference is
not reliable enough to build on (a reference cycle, or a caller that keeps
one around a little longer than expected, defers it indefinitely), so
:func:`release_memmap_array` closes the array's own ``mmap.mmap`` object
directly -- an immediate, deterministic unmap independent of anyone else's
reference count -- confirmed empirically on this checkout's Windows: closing
the mmap released the OS lock immediately even with another reference to the
(now unusable) array still alive.
"""
from __future__ import annotations

import atexit
import itertools
import logging
import os
import tempfile
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

#: Most voxels one padded block of :func:`iter_blocks` may hold. scipy's
#: morphology and local filters need a few bytes per voxel on top of their
#: input, so a block stays in the low hundreds of MB; a distance transform,
#: at roughly 70 bytes per voxel of scipy's own internal temporaries, stays
#: around 2 GB. Deliberately a volume budget rather than a slice count: one
#: z-slice of a wide stack can already be tens of millions of voxels.
LOW_MEMORY_BLOCK_VOXELS = 32_000_000


def iter_blocks(
    shape: tuple[int, ...], *, halo: int, block_voxels: int = LOW_MEMORY_BLOCK_VOXELS
) -> Iterator[tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]]:
    """Yield ``(padded, core_in_padded, core)`` slice tuples tiling *shape*.

    The cores partition *shape* exactly; each padded region extends its core
    by *halo* voxels along every axis, clipped to the volume. An operation
    whose output at a voxel depends only on input within *halo* voxels of it
    (along every axis) therefore gives, on each core, exactly what it gives
    on the whole volume -- including at the volume's own edges, where the
    padded region stops at the same boundary a whole-volume call sees.

    Cores are shrunk by halving the longest one until a padded block fits
    *block_voxels*; a halo wide enough that even a single-voxel core cannot
    fit is accepted rather than looped on forever.
    """
    shape = tuple(int(s) for s in shape)
    halo = max(0, int(halo))
    core = list(shape)

    def padded_voxels(extents: list[int]) -> int:
        return int(np.prod([min(s, c + 2 * halo) for s, c in zip(shape, extents)]))

    while padded_voxels(core) > block_voxels:
        axis = int(np.argmax(core))
        if core[axis] <= 1:
            break
        core[axis] = (core[axis] + 1) // 2

    for starts in itertools.product(*(range(0, s, max(1, c)) for s, c in zip(shape, core))):
        core_slices = tuple(
            slice(start, min(start + c, s)) for start, c, s in zip(starts, core, shape)
        )
        padded = tuple(
            slice(max(0, cs.start - halo), min(s, cs.stop + halo))
            for cs, s in zip(core_slices, shape)
        )
        inner = tuple(
            slice(cs.start - ps.start, cs.stop - ps.start)
            for cs, ps in zip(core_slices, padded)
        )
        yield padded, inner, core_slices


def map_blockwise(
    source: np.ndarray,
    op: Callable[[np.ndarray], np.ndarray],
    out: np.ndarray,
    *,
    halo: int,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
) -> np.ndarray:
    """``out[core] = op(source[padded])[core_in_padded]`` for every block.

    Exactly ``out[...] = op(source)`` for any *op* whose reach is at most
    *halo* voxels (see :func:`iter_blocks`), with no array larger than one
    padded block ever held in RAM -- *source* and *out* can both be memmaps.
    """
    for padded, inner, core in iter_blocks(source.shape, halo=halo, block_voxels=block_voxels):
        out[core] = op(np.asarray(source[padded]))[inner]
    return out


def at_faces_as_beyond(op: Callable[[np.ndarray], np.ndarray], reach: int) -> Callable[[np.ndarray], np.ndarray]:
    """*op* run on its input padded by *reach* voxels of edge replication, and
    cropped back: what morphology reads past an image face is the face itself,
    a vessel the image cut running straight on past it.

    scipy reads outside the array as background, so a closing's erosion eats
    the first and last *r* slices of every vessel through a face (it stops
    being extensive), and an opening caps it there -- moving every face end,
    and the boundary nodes chosen from them, up to *r* voxels into the image
    (PLAN_dale_review_fixes.md, item 2). Used block by block it stays exact: a
    block's own halo is at least *reach* deep wherever it is not a face, so
    the padding only ever stands in for what lies beyond the image.
    """
    reach = int(reach)

    def padded(block: np.ndarray) -> np.ndarray:
        if reach <= 0:
            return op(block)
        grown = np.pad(np.asarray(block), reach, mode="edge")
        crop = tuple(slice(reach, reach + size) for size in block.shape)
        return np.asarray(op(grown))[crop]

    return padded


#: Backing files of :func:`new_memmap_array` arrays dropped without
#: :func:`release_memmap_array` whose deletion Windows refused while the map
#: was still open: tried again at the next :func:`new_memmap_array`, and as
#: the process exits.
_DROPPED_FILES: set[str] = set()
#: Each live array's finaliser, by backing file, so a release cancels it.
_FINALISERS: dict[str, weakref.finalize] = {}


def _delete_dropped(path: str) -> None:
    """The finaliser of an array whose last reference has gone: delete its
    file. It runs while the array's map is still open, so on Windows the file
    waits for :func:`_delete_dropped_files`."""
    _FINALISERS.pop(path, None)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        _DROPPED_FILES.add(path)


@atexit.register
def _delete_dropped_files() -> None:
    for path in list(_DROPPED_FILES):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            continue
        _DROPPED_FILES.discard(path)


def new_memmap_array(
    shape: tuple[int, ...], dtype, *, directory: str | Path | None = None
) -> np.memmap:
    """A fresh disk-backed array of *shape*/*dtype*, in a new temp file.

    The backing file outlives this call -- pass the result to
    :func:`release_memmap_array` once it is no longer needed. One nobody
    releases is deleted once the array, and every view of it, is gone (on
    Windows, which will not delete a file while it is mapped, at the next
    call here or as the process exits): stage outputs such as the skeleton
    ``skeletonise`` hands on, the float32 raw image the raw evidence and the
    Diameters stage read (two 301 MB files a low-RAM E14.5 run), and the
    volumes tests compare were left in the temp directory by every run --
    2,800 of them on one machine (2026-10-18). Killing the process outright
    still leaves them.
    *directory* overrides where the temp file is created (the pipeline's
    ``memmap_directory`` setting, when the OS temp drive is the wrong place
    for a volume-sized file -- too small, or too slow); the default is
    wherever ``tempfile`` already puts them (``$TMPDIR``/``%TEMP%``). Created
    if it does not already exist.
    """
    if _DROPPED_FILES:
        _delete_dropped_files()
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(
        prefix="haemolynx_", suffix=".memmap", dir=str(directory) if directory else None
    )
    os.close(fd)
    array = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(int(s) for s in shape))
    key = os.path.abspath(path)
    _FINALISERS[key] = weakref.finalize(array, _delete_dropped, key)
    return array


def release_memmap_array(array: np.memmap) -> Path:
    """Close *array*'s own memory map and delete its backing file.

    The caller must not use *array* again afterwards -- this leaves its
    Python object dangling, the same as any array whose backing storage has
    been removed. Returns the removed path (useful for logging/tests).
    Failure to remove the file is logged, not raised: a leftover temp file
    is a cleanup nuisance, not a reason to fail a pipeline run that has
    already produced its result.
    """
    path = Path(array.filename)
    finaliser = _FINALISERS.pop(os.path.abspath(path), None)
    if finaliser is not None:
        finaliser.detach()
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()
    try:
        path.unlink()
    except OSError:
        logger.warning("Could not remove memmap file %s", path, exc_info=True)
    return path


@contextmanager
def temporary_memmap_array(
    shape: tuple[int, ...], dtype, *, directory: str | Path | None = None
) -> Iterator[np.memmap]:
    """A disk-backed array whose backing file is removed when the block exits.

    For a full-volume intermediate (a connected-component label array, a
    distance-transform buffer) that is only needed within one step -- unlike
    the loaded image itself, which a caller keeps for the rest of the run
    and releases explicitly with :func:`release_memmap_array` when it is
    finally done with it. Do not let the yielded array escape the ``with``
    block: it is invalid the moment the block exits.
    """
    array = new_memmap_array(shape, dtype, directory=directory)
    try:
        yield array
    finally:
        release_memmap_array(array)


def release_superseded(previous: np.ndarray, new: np.ndarray, *, keep: np.ndarray) -> None:
    """Delete *previous*'s backing file once *new* has replaced it.

    For a chain of steps that each write a fresh volume-sized memmap: the one
    a step replaces is nobody else's, so it is deleted there and then rather
    than left in the temp directory until the OS gets round to it. Only a
    memmap that is neither *new* itself (a step that changed nothing hands
    its input straight back) nor *keep* (the caller's own input, theirs to
    keep or release) is touched -- so this is a no-op on plain arrays.
    """
    if isinstance(previous, np.memmap) and previous is not new and previous is not keep:
        release_memmap_array(previous)


def unpack_bits_by_slab(packed: np.ndarray, shape, *, directory=None) -> np.memmap:
    """``np.unpackbits(packed)[:n].reshape(shape).astype(bool)`` (*packed*
    from ``np.packbits`` of a volume of *shape*, big bit order) written into
    a new disk-backed volume a slab of planes at a time: each slab's bits
    unpacked from the bytes that hold them, cut where the slab starts and
    stops within its first and last byte. Unpacked whole, the bits were a
    byte a voxel and then a copy of that as booleans: two volumes in RAM."""
    shape = tuple(int(s) for s in shape)
    out = new_memmap_array(shape, bool, directory=directory)
    plane = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    step = slab_step(shape)
    try:
        for z0 in range(0, shape[0], step):
            z1 = min(z0 + step, shape[0])
            first, last = z0 * plane, z1 * plane
            byte0 = first // 8
            bits = np.unpackbits(np.asarray(packed[byte0:(last + 7) // 8]))
            offset = first - 8 * byte0
            out[z0:z1] = bits[offset:offset + last - first].reshape((z1 - z0,) + shape[1:]).astype(bool)
    except BaseException:
        release_memmap_array(out)
        raise
    return out


def argwhere_by_slab(
    volume: np.ndarray,
    predicate: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
) -> np.ndarray:
    """``np.argwhere(predicate(volume))`` without a whole-volume temporary.

    Reads *volume* a slab of whole axis-0 slices at a time, so the result is
    in exactly the C order a whole-volume ``np.argwhere`` gives -- same rows,
    same order. *predicate* defaults to truthiness.
    """
    shape = volume.shape
    step = slab_step(shape, block_voxels)
    parts = []
    for start in range(0, shape[0], step):
        slab = np.asarray(volume[start:start + step])
        found = np.argwhere(slab if predicate is None else predicate(slab))
        found[:, 0] += start
        parts.append(found)
    if not parts:
        return np.empty((0, volume.ndim), dtype=np.intp)
    return np.concatenate(parts)


def map_by_slab(
    source: np.ndarray,
    op: Callable[[np.ndarray], np.ndarray],
    out: np.ndarray,
    *,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
) -> np.ndarray:
    """``out[...] = op(source)`` for an elementwise *op*, a slab at a time."""
    step = slab_step(source.shape, block_voxels)
    for start in range(0, source.shape[0], step):
        out[start:start + step] = op(np.asarray(source[start:start + step]))
    return out


def slab_step(shape: tuple[int, ...], block_voxels: int | None = None) -> int:
    """Whole axis-0 slices per slab, so a slab holds about *block_voxels*
    (:data:`LOW_MEMORY_BLOCK_VOXELS` when not given)."""
    if block_voxels is None:
        block_voxels = LOW_MEMORY_BLOCK_VOXELS
    slice_voxels = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    return max(1, block_voxels // max(slice_voxels, 1))


def bincount_by_slab(
    labels: np.ndarray,
    *,
    where: np.ndarray | None = None,
    minlength: int = 0,
    block_voxels: int | None = None,
) -> np.ndarray:
    """``np.bincount(labels.ravel())`` -- or of ``labels[where]`` -- a slab at a
    time. ``np.bincount`` casts its whole input to ``intp``, so on an int32
    label memmap the plain call is an 8-byte-per-voxel copy of the volume."""
    counts = np.zeros(max(int(minlength), 0), dtype=np.int64)
    step = slab_step(labels.shape, block_voxels)
    for start in range(0, labels.shape[0], step):
        slab = np.asarray(labels[start:start + step])
        if where is not None:
            slab = slab[np.asarray(where[start:start + step], dtype=bool)]
        part = np.bincount(slab.ravel(), minlength=len(counts))
        if len(part) > len(counts):
            counts = np.concatenate([counts, np.zeros(len(part) - len(counts), np.int64)])
        counts[: len(part)] += part
    return counts


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def delete_when_unmapped(array: np.memmap) -> np.memmap:
    """Delete *array*'s backing file once nothing maps it any more.

    For a volume handed to something that decides its lifetime -- a napari
    layer -- rather than released explicitly: the file goes when the last
    view of the mapping is garbage-collected, instead of waiting in the temp
    directory. Returns *array* for chaining.
    """
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and getattr(array, "filename", None):
        import weakref

        weakref.finalize(mapping, _remove_quietly, str(array.filename))
    return array
