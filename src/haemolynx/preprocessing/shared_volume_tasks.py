"""Independent per-item work over one volume, spread across worker processes.

A task that reads a volume and depends only on its own payload -- one vessel's
cross-sections, one chunk of centreline samples' lumen outlines -- gives the
same answer whichever process computes it, so a list of them can be shared
out. The workers read the volume from one temporary memory-mapped file
rather than each receiving a copy, so memory stays one volume's worth
(:func:`run_volume_tasks`). The haemodynamics section methods and graph
building's mask recovery share this; it lives in ``preprocessing`` because
that is below both of them.
"""
from __future__ import annotations

import logging
import mmap
import os
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_DEFAULT_WORKERS",
    "default_worker_count",
    "iter_volume_tasks",
    "open_whole_file",
    "run_volume_tasks",
    "whole_file_spec",
]

#: The most workers :func:`default_worker_count` gives: on a 20-thread
#: machine 500 real vessels' endothelial rings took 170 s in one process,
#: 41 s in 8 and 54 s in 19 -- past eight, starting workers and sharing the
#: physical cores costs more than the extra ones save.
MAX_DEFAULT_WORKERS = 8

_WORKER_VOLUME: np.ndarray | tuple[np.ndarray, ...] | None = None

#: Memory :func:`iter_volume_tasks` leaves free for the rest of the machine:
#: another task goes out only while this much and the task's own size are
#: available, so a run shrinks to fewer tasks at once (one at least) as
#: something else grows, rather than taking the machine to nothing free.
MEMORY_RESERVE_BYTES = 2 << 30

_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def default_worker_count() -> int:
    """Up to :data:`MAX_DEFAULT_WORKERS`, leaving a core for the rest of the machine."""
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) - 1))


def _load_worker_volume(
    specs: list[tuple[str, str, tuple[int, ...], int]], as_tuple: bool
) -> None:
    global _WORKER_VOLUME
    volumes = tuple(
        np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape, offset=offset)
        for path, dtype, shape, offset in specs
    )
    _WORKER_VOLUME = volumes if as_tuple else volumes[0]


def whole_file_spec(array: np.ndarray) -> tuple[str, str, tuple[int, ...], int]:
    """How a worker opens *array* for writing (:func:`open_whole_file`): it
    must be the whole of a file-backed memmap."""
    spec = _whole_file(array)
    if spec is None:
        raise ValueError("workers can write only into the whole of a file-backed memmap")
    return spec


def open_whole_file(spec: tuple[str, str, tuple[int, ...], int], mode: str = "r+") -> np.memmap:
    """The memmap *spec* (from :func:`whole_file_spec`) describes, opened in
    this process -- a worker writing its own part of a shared output."""
    path, dtype, shape, offset = spec
    return np.memmap(path, dtype=np.dtype(dtype), mode=mode, shape=tuple(shape), offset=int(offset))


def _whole_file(array: np.ndarray) -> tuple[str, str, tuple[int, ...], int] | None:
    """``(path, dtype, shape, offset)`` when *array* is the whole of a
    file-backed memmap -- itself, or a plain view of all of it -- so workers
    can map that file rather than a copy; else None."""
    root = array
    while root is not None and not (isinstance(root, np.memmap) and isinstance(root.base, mmap.mmap)):
        root = root.base if isinstance(root, np.ndarray) else None
    if root is None or not root.filename:
        return None
    same = (
        array.shape == root.shape
        and array.dtype == root.dtype
        and array.flags.c_contiguous
        and array.__array_interface__["data"][0] == root.__array_interface__["data"][0]
    )
    if not same:
        return None
    return str(root.filename), root.dtype.str, tuple(int(s) for s in root.shape), int(root.offset)


def _run_on_worker_volume(task_and_payload):
    task, payload = task_and_payload
    return task(_WORKER_VOLUME, payload)


def run_volume_tasks(
    task, payloads, *, volume: np.ndarray | tuple[np.ndarray, ...], workers: int = 1,
    min_payloads: int = 2, memmap_directory=None,
):
    """``[task(volume, payload) for payload in payloads]``, spread over
    *workers* processes when there are at least *min_payloads* of them.

    Each payload's result depends only on the payload and the volume, so the
    results are the same, in the same order, however many processes compute
    them. The workers read the volume from one temporary memory-mapped file
    rather than each receiving a copy. *volume* may be a tuple of volumes (an
    image and its mask, say), each shared the same way and handed to *task*
    as that tuple; a boolean one stays boolean, an integer one keeps its
    dtype, any other is read as float32. *task* must be a module-level
    function (it is sent to the workers by name). Falls back to this process,
    with a warning, if workers cannot be started.
    """
    payloads = list(payloads)
    if workers <= 1 or len(payloads) < min_payloads:
        return [task(volume, payload) for payload in payloads]
    from concurrent.futures import ProcessPoolExecutor

    from .memmap_support import new_memmap_array, release_memmap_array

    volumes = volume if isinstance(volume, tuple) else (volume,)
    shared = [
        new_memmap_array(
            each.shape,
            each.dtype if each.dtype == np.bool_ or np.issubdtype(each.dtype, np.integer) else np.float32,
            directory=memmap_directory,
        )
        for each in volumes
    ]
    # Workers read these as they load numpy: one native thread each, since
    # the workers already fill the cores and a BLAS/OpenMP team in each would
    # oversubscribe them.
    single_threaded = {name: "1" for name in _NATIVE_THREAD_VARIABLES}
    saved = {name: os.environ.get(name) for name in single_threaded}
    try:
        for copy, each in zip(shared, volumes):
            copy[...] = each
            copy.flush()
        chunk = max(1, len(payloads) // (4 * workers))
        os.environ.update(single_threaded)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_load_worker_volume,
                initargs=(
                    [(str(copy.filename), copy.dtype.str, tuple(copy.shape), 0) for copy in shared],
                    isinstance(volume, tuple),
                ),
            ) as pool:
                return list(pool.map(_run_on_worker_volume, [(task, p) for p in payloads], chunksize=chunk))
        except (OSError, RuntimeError) as error:  # e.g. no process spawning here
            logger.warning(
                "Could not start %d worker processes (%s); measuring in this one.", workers, error
            )
            return [task(volume, payload) for payload in payloads]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for copy in shared:
            release_memmap_array(copy)


def _room_for(task_bytes: int | None) -> bool:
    """Whether one more task of *task_bytes* leaves :data:`MEMORY_RESERVE_BYTES`
    free (True when either is unknown)."""
    if not task_bytes:
        return True
    from haemolynx.haemodynamics.resistance import available_memory_bytes

    available = available_memory_bytes()
    return available is None or available >= MEMORY_RESERVE_BYTES + int(task_bytes)


def iter_volume_tasks(
    task, payloads, *, volume: np.ndarray | tuple[np.ndarray, ...], workers: int = 1,
    min_payloads: int = 2, memmap_directory=None, task_bytes: int | None = None,
    on_low_memory=None,
):
    """:func:`run_volume_tasks` one result at a time, in order, with only a
    few more payloads out at once than there are *workers* -- for results
    that together would be a volume's worth.

    A volume that is already the whole of a file-backed memmap is mapped by
    the workers as it is; any other is copied to one first. Falls back to
    this process, with a warning, for whatever is left if the workers fail.

    *task_bytes*, a task's memory, has the free memory read again before
    each further task goes out: below :data:`MEMORY_RESERVE_BYTES` and that,
    it waits for one out to finish instead (calling *on_low_memory* first --
    flushing the caller's memmaps, say, whose unwritten pages are not free --
    at most every few seconds). The results do not depend on how many run at
    once.
    """
    payloads = list(payloads)
    if workers <= 1 or len(payloads) < min_payloads:
        for payload in payloads:
            yield task(volume, payload)
        return
    from concurrent.futures import ProcessPoolExecutor

    from .memmap_support import new_memmap_array, release_memmap_array

    volumes = volume if isinstance(volume, tuple) else (volume,)
    specs = []
    copies = []
    single_threaded = {name: "1" for name in _NATIVE_THREAD_VARIABLES}
    saved = {name: os.environ.get(name) for name in single_threaded}
    done = 0
    try:
        for each in volumes:
            spec = _whole_file(each)
            if spec is None:
                copy = new_memmap_array(
                    each.shape,
                    each.dtype if each.dtype == np.bool_ or np.issubdtype(each.dtype, np.integer) else np.float32,
                    directory=memmap_directory,
                )
                copies.append(copy)
                step = max(1, (32 << 20) // max(1, int(np.prod(each.shape[1:], dtype=np.int64)) * copy.itemsize))
                for start in range(0, each.shape[0], step):
                    copy[start:start + step] = each[start:start + step]
                copy.flush()
                spec = (str(copy.filename), copy.dtype.str, tuple(copy.shape), 0)
            specs.append(spec)
        os.environ.update(single_threaded)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_load_worker_volume,
                initargs=(specs, isinstance(volume, tuple)),
            ) as pool:
                pending = deque()
                queued = deque(payloads)
                window = workers + max(2, workers // 2)
                flushed = [float("-inf")]

                def refill():
                    while queued and len(pending) < window:
                        if pending and not _room_for(task_bytes):
                            if on_low_memory is not None and time.monotonic() - flushed[0] > 5.0:
                                flushed[0] = time.monotonic()
                                on_low_memory()
                            if not _room_for(task_bytes):
                                return
                        pending.append(pool.submit(_run_on_worker_volume, (task, queued.popleft())))

                refill()
                while pending:
                    result = pending.popleft().result()
                    done += 1
                    refill()
                    yield result
        except (OSError, RuntimeError) as error:  # e.g. no process spawning here
            logger.warning(
                "Could not run %d worker processes (%s); finishing in this one.", workers, error
            )
            for payload in payloads[done:]:
                yield task(volume, payload)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for copy in copies:
            release_memmap_array(copy)
