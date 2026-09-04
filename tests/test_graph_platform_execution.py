"""Windows execution choices for graph building must not change results.

Linux keeps the thread pools and int32 labelling that were timed there.
Windows skips GIL-bound pools, compiles skan on a tiny volume first, and
labels background as int16 when it fits. The mask and the graph stay the same.
"""
from __future__ import annotations

import numpy as np

from haemolynx.graph._platform import (
    iter_python_work,
    map_python_work,
    nested_native_thread_limit,
    python_loops_use_threads,
    skan_numba_warmup_skeleton,
)
from haemolynx.preprocessing.skeleton import (
    _label_inverted_background,
    fill_binary_holes,
)


def test_python_loops_are_threaded_off_windows_only():
    assert python_loops_use_threads("linux") is True
    assert python_loops_use_threads("darwin") is True
    assert python_loops_use_threads("win32") is False


def test_python_work_preserves_input_order_on_both_platforms():
    items = [3, 1, 2]

    def double(x):
        return x * 2

    threaded = map_python_work(double, items, max_workers=4, platform="linux")
    sequential = map_python_work(double, items, max_workers=4, platform="win32")
    assert threaded == [6, 2, 4]
    assert sequential == [6, 2, 4]


def test_python_work_does_not_open_a_pool_on_windows(monkeypatch):
    from haemolynx.graph import _platform as plat

    def boom(*_args, **_kwargs):
        raise AssertionError("ThreadPoolExecutor must not run on Windows")

    monkeypatch.setattr(plat, "ThreadPoolExecutor", boom)
    assert list(iter_python_work(lambda x: x, [1, 2, 3], max_workers=4, platform="win32")) == [
        1,
        2,
        3,
    ]


def test_python_work_still_opens_a_pool_on_linux(monkeypatch):
    from haemolynx.graph import _platform as plat

    seen = []

    class _FakePool:
        def __init__(self, max_workers):
            seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    monkeypatch.setattr(plat, "ThreadPoolExecutor", _FakePool)
    assert map_python_work(lambda x: x + 1, [1, 2], max_workers=4, platform="linux") == [2, 3]
    assert seen == [4]


def test_nested_native_thread_limit_is_a_no_op_off_windows(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "threadpoolctl":
            raise AssertionError("threadpoolctl imported off Windows")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with nested_native_thread_limit("linux"):
        pass


def test_skan_numba_warmup_is_windows_only():
    assert skan_numba_warmup_skeleton("linux") is None
    assert skan_numba_warmup_skeleton("darwin") is None
    warmup = skan_numba_warmup_skeleton("win32")
    assert warmup is not None
    assert warmup.shape == (5, 5, 5)
    assert warmup.dtype == bool
    assert int(warmup.sum()) == 3


def test_windows_compact_background_labels_match_int32_ids():
    """int16 on Windows is a smaller array of the same ids, not a different labelling."""
    volume = np.zeros((12, 12, 12), dtype=bool)
    volume[3:8, 3:8, 3:8] = True
    volume[4:7, 4:7, 4:7] = False
    inverted = ~volume
    compact, n_compact = _label_inverted_background(inverted, platform="win32")
    full, n_full = _label_inverted_background(inverted, platform="linux")
    assert n_compact == n_full
    assert compact.dtype == np.int16
    assert np.array_equal(compact.astype(np.int32), np.asarray(full, dtype=np.int32))


def test_windows_compact_hole_fill_still_matches_scipy():
    from scipy.ndimage import binary_fill_holes
    from haemolynx.preprocessing import skeleton as skeleton_mod

    volume = np.zeros((12, 12, 12), dtype=bool)
    volume[3:8, 3:8, 3:8] = True
    volume[4:7, 4:7, 4:7] = False

    original = skeleton_mod.sys.platform
    try:
        skeleton_mod.sys.platform = "win32"
        ours = fill_binary_holes(volume)
    finally:
        skeleton_mod.sys.platform = original
    assert np.array_equal(ours, binary_fill_holes(volume))


def test_windows_label_falls_back_when_int16_cannot_hold_the_ids(monkeypatch):
    from haemolynx.preprocessing import skeleton as skeleton_mod

    real_label = skeleton_mod.label

    def fake_label(inverted, output=None):
        if output is not None and getattr(output, "dtype", None) == np.dtype(np.int16):
            raise RuntimeError("too many labels")
        return real_label(inverted)

    monkeypatch.setattr(skeleton_mod, "label", fake_label)
    inverted = np.ones((4, 4, 4), dtype=bool)
    labeled, n_labels = _label_inverted_background(inverted, platform="win32")
    assert n_labels >= 0
    assert labeled.dtype != np.int16
