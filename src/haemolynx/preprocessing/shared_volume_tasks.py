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
import os

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["MAX_DEFAULT_WORKERS", "default_worker_count", "run_volume_tasks"]

#: The most workers :func:`default_worker_count` gives: on a 20-thread
#: machine 500 real vessels' endothelial rings took 170 s in one process,
#: 41 s in 8 and 54 s in 19 -- past eight, starting workers and sharing the
#: physical cores costs more than the extra ones save.
MAX_DEFAULT_WORKERS = 8

_WORKER_VOLUME: np.ndarray | tuple[np.ndarray, ...] | None = None

_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def default_worker_count() -> int:
    """Up to :data:`MAX_DEFAULT_WORKERS`, leaving a core for the rest of the machine."""
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) - 1))


def _load_worker_volume(specs: list[tuple[str, str, tuple[int, ...]]], as_tuple: bool) -> None:
    global _WORKER_VOLUME
    volumes = tuple(
        np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape) for path, dtype, shape in specs
    )
    _WORKER_VOLUME = volumes if as_tuple else volumes[0]


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
                    [(str(copy.filename), copy.dtype.str, tuple(copy.shape)) for copy in shared],
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
