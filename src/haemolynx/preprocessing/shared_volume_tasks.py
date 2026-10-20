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
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_DEFAULT_WORKERS",
    "close_kept_workers",
    "default_worker_count",
    "iter_volume_tasks",
    "kept_workers",
    "open_whole_file",
    "run_volume_tasks",
    "whole_file_spec",
    "write_whole_file_part",
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


def write_whole_file_part(spec: tuple[str, str, tuple[int, ...], int], key, value) -> None:
    """``open_whole_file(spec)[key] = value``, on disk before it returns. A
    write left to Windows waits on the modified page list, not free memory,
    until its mapped page writer gets to it: on a 5.5 G-voxel stack that list
    held 14.7 GB of the run's volumes, emptying at ~55 MB/s, where a flush
    wrote 1 GB in 0.23 s."""
    out = open_whole_file(spec)
    out[key] = value
    out.flush()
    del out


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


#: The workers :func:`kept_workers` keeps between :func:`iter_volume_tasks`
#: calls: how many blocks deep it is, the pool and its size, whether a call
#: has the pool now, and whether to stop it once that call is done.
_KEPT: dict = {"depth": 0, "pool": None, "size": 0, "busy": False, "close": False}
_KEPT_LOCK = threading.Lock()


@contextmanager
def kept_workers() -> Iterator[None]:
    """Within the block, :func:`iter_volume_tasks` hands its worker processes
    on to the next call instead of stopping them, and they stop as the
    outermost block ends (or at :func:`close_kept_workers`). For several
    passes over one stack in a row: a worker imports the whole package as it
    starts, ~17 s with a dozen starting at once on a 20-core machine, and the
    low-RAM thick region's hole fill, radius map, two propagation passes and
    wall each started their own -- on a 75 M-voxel stack the region took 132 s
    in workers and 45 s in one process. A call wanting more workers than are
    kept starts that many in their place; one wanting fewer runs no more than
    it asked for at once."""
    with _KEPT_LOCK:
        _KEPT["depth"] += 1
    try:
        yield
    finally:
        with _KEPT_LOCK:
            _KEPT["depth"] -= 1
            outermost = _KEPT["depth"] == 0
        if outermost:
            close_kept_workers()


def close_kept_workers() -> None:
    """Stop the workers :func:`kept_workers` keeps -- before a step in this
    process that wants their memory; a later call in the block starts new
    ones. One a call is using stops when that call is done."""
    with _KEPT_LOCK:
        if _KEPT["busy"]:
            _KEPT["close"] = True
            return
        pool = _KEPT["pool"]
        _KEPT["pool"], _KEPT["size"] = None, 0
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def _started() -> int:
    return os.getpid()


def _start_pool(workers: int):
    """A pool of *workers* processes, every one started now, with one native
    thread each (see :func:`run_volume_tasks`): on 'spawn' a pool starts its
    processes as tasks arrive, which for a kept pool would be after the
    environment is put back."""
    from concurrent.futures import ProcessPoolExecutor

    single_threaded = {name: "1" for name in _NATIVE_THREAD_VARIABLES}
    saved = {name: os.environ.get(name) for name in single_threaded}
    os.environ.update(single_threaded)
    try:
        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            for future in [pool.submit(_started) for _ in range(workers)]:
                future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        return pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _pool_for(workers: int):
    """``(pool, kept)``: the kept pool inside :func:`kept_workers` (started,
    or started again larger, as *workers* needs), else a new one of
    *workers* for this call alone."""
    with _KEPT_LOCK:
        kept = _KEPT["depth"] > 0 and not _KEPT["busy"]
        if kept:
            _KEPT["busy"] = True
            pool, size = _KEPT["pool"], _KEPT["size"]
    if not kept:
        return _start_pool(workers), False
    try:
        if pool is None or size < workers:
            with _KEPT_LOCK:
                _KEPT["pool"], _KEPT["size"] = None, 0
            if pool is not None:
                pool.shutdown(wait=True)
            pool = _start_pool(workers)
            with _KEPT_LOCK:
                _KEPT["pool"], _KEPT["size"] = pool, workers
    except BaseException:
        with _KEPT_LOCK:
            _KEPT["busy"] = False
        raise
    return pool, True


def _give_back(pool, *, broken: bool = False) -> None:
    """A call's kept *pool* free again -- stopped if it broke, if the block
    has ended or if :func:`close_kept_workers` asked meanwhile."""
    with _KEPT_LOCK:
        _KEPT["busy"] = False
        stop = broken or _KEPT["depth"] == 0 or _KEPT["close"]
        _KEPT["close"] = False
        if stop and _KEPT["pool"] is pool:
            _KEPT["pool"], _KEPT["size"] = None, 0
    if stop:
        pool.shutdown(wait=True, cancel_futures=True)


def _run_job(job):
    """One task in a worker, its volumes mapped for it alone: their pages
    leave the worker's memory with it rather than piling up there, where the
    free memory read here (the next plan, the re-check) counts them used."""
    task, payload, specs, as_tuple = job
    volumes = tuple(
        np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=tuple(shape), offset=int(offset))
        for path, dtype, shape, offset in specs
    )
    return task(volumes if as_tuple else volumes[0], payload)


def iter_volume_tasks(
    task, payloads, *, volume: np.ndarray | tuple[np.ndarray, ...], workers: int = 1,
    min_payloads: int = 2, memmap_directory=None, task_bytes: int | None = None,
    on_low_memory=None,
):
    """:func:`run_volume_tasks` one result at a time, in order, with only a
    few more payloads out at once than there are *workers* -- for results
    that together would be a volume's worth. No more than *workers* run at
    once, and another goes out as soon as any is done.

    A volume that is already the whole of a file-backed memmap is mapped by
    the workers as it is; any other is copied to one first. Falls back to
    this process, with a warning, for whatever is left if the workers fail.
    Inside :func:`kept_workers` the workers are the ones kept.

    *task_bytes*, a task's memory, has the free memory read again before
    each further task goes out: below :data:`MEMORY_RESERVE_BYTES` and that,
    it waits for one out to finish instead. Before it waits -- at most every
    few seconds -- this process's own disk-backed volumes are written out and
    their pages handed back (:func:`~.memmap_support.release_resident_pages`:
    on Windows they are not free memory until then), and *on_low_memory*
    called. The results do not depend on how many run at once.
    """
    payloads = list(payloads)
    if workers <= 1 or len(payloads) < min_payloads:
        for payload in payloads:
            yield task(volume, payload)
        return
    from concurrent.futures import FIRST_COMPLETED, wait

    from .memmap_support import new_memmap_array, release_memmap_array, release_resident_pages

    volumes = volume if isinstance(volume, tuple) else (volume,)
    specs = []
    copies = []
    done = 0
    pool = None
    kept = False
    broken = False
    pending = deque()
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
        try:
            pool, kept = _pool_for(workers)
            as_tuple = isinstance(volume, tuple)
            queued = deque(payloads)
            window = workers + max(2, workers // 2)
            flushed = [float("-inf")]

            def refill():
                while queued and len(pending) < window:
                    running = sum(1 for future in pending if not future.done())
                    if running >= workers:
                        return
                    if running and not _room_for(task_bytes):
                        if time.monotonic() - flushed[0] > 5.0:
                            # This process's own volumes' pages are not free
                            # memory on Windows until handed back.
                            flushed[0] = time.monotonic()
                            release_resident_pages(*volumes)
                            if on_low_memory is not None:
                                on_low_memory()
                        if not _room_for(task_bytes):
                            return
                    pending.append(pool.submit(_run_job, (task, queued.popleft(), specs, as_tuple)))

            refill()
            while pending:
                if not pending[0].done():
                    wait([future for future in pending if not future.done()], return_when=FIRST_COMPLETED)
                    refill()
                    continue
                result = pending.popleft().result()
                done += 1
                refill()
                yield result
        except (OSError, RuntimeError) as error:  # e.g. no process spawning here
            logger.warning(
                "Could not run %d worker processes (%s); finishing in this one.", workers, error
            )
            broken = True
            for future in pending:
                future.cancel()
            pending.clear()
            for payload in payloads[done:]:
                yield task(volume, payload)
    finally:
        for future in pending:
            future.cancel()
        if pool is not None:
            if kept:
                wait(pending)
                _give_back(pool, broken=broken)
            else:
                pool.shutdown(wait=True, cancel_futures=True)
        for copy in copies:
            release_memmap_array(copy)
