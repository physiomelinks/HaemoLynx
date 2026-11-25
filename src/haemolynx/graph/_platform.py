"""OS-specific *how* for graph building, not *what*.

The skeleton and topology algorithms are the same on every platform. Linux
timings for windowed distance transforms and ball-dilation cutovers stay as
they were measured. Windows only changes execution: skip thread pools whose
work holds the GIL, cap nested native threads inside GIL-releasing scipy
calls, and compile skan/Numba before the real volume is in RAM.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def python_loops_use_threads(platform: str | None = None) -> bool:
    """Whether GIL-bound Python loops should run on a thread pool.

    Those loops do not release the GIL, so extra threads are overhead. Linux
    keeps the existing pool; Windows does not.
    """
    if platform is None:
        platform = sys.platform
    return platform != "win32"


def iter_python_work(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    *,
    platform: str | None = None,
) -> Iterator[R]:
    """Apply *fn* to *items* in input order, threaded only when that helps."""
    if platform is None:
        platform = sys.platform
    sequence = list(items)
    if (
        not sequence
        or max_workers <= 1
        or len(sequence) == 1
        or not python_loops_use_threads(platform)
    ):
        for item in sequence:
            yield fn(item)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(fn, sequence)


def map_python_work(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    *,
    platform: str | None = None,
) -> list[R]:
    """Eager form of :func:`iter_python_work`."""
    return list(iter_python_work(fn, items, max_workers, platform=platform))


@contextmanager
def nested_native_thread_limit(platform: str | None = None):
    """Cap OpenMP/BLAS inside reconnect workers on Windows only.

    Windowed ``distance_transform_edt`` releases the GIL, so several windows
    are meant to run at once. If scipy's native library also opens a thread
    team per call, Windows oversubscribes and the Linux-measured window budget
    stops being a saving. One native thread per worker keeps the outer pool;
    Linux does not enter this.
    """
    if platform is None:
        platform = sys.platform
    if platform != "win32":
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1):
        yield


def skan_numba_warmup_skeleton(platform: str | None = None):
    """A tiny skeleton used to pay Numba's compile cost, or ``None`` to skip.

    Some skan functions are ``cache=False``, so every process LLVM-compiles
    them. Doing that while a full stack is live is a Windows commit-charge
    spike. Linux leaves the first real ``Skeleton()`` to compile as before.
    """
    if platform is None:
        platform = sys.platform
    if platform != "win32":
        return None
    import numpy as np

    tiny = np.zeros((5, 5, 5), dtype=bool)
    tiny[2, 2, 1:4] = True
    return tiny
