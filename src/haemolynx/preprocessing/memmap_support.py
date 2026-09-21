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

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


def new_memmap_array(
    shape: tuple[int, ...], dtype, *, directory: str | Path | None = None
) -> np.memmap:
    """A fresh disk-backed array of *shape*/*dtype*, in a new temp file.

    The backing file outlives this call -- pass the result to
    :func:`release_memmap_array` once it is no longer needed, or leave it
    for the OS to reclaim from the temp directory eventually (no worse than
    what killing the process outright would already leave behind).
    *directory* overrides where the temp file is created (the pipeline's
    ``memmap_directory`` setting, when the OS temp drive is the wrong place
    for a volume-sized file -- too small, or too slow); the default is
    wherever ``tempfile`` already puts them (``$TMPDIR``/``%TEMP%``). Created
    if it does not already exist.
    """
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(
        prefix="haemolynx_", suffix=".memmap", dir=str(directory) if directory else None
    )
    os.close(fd)
    return np.memmap(path, dtype=dtype, mode="w+", shape=tuple(int(s) for s in shape))


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
