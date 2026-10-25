"""``preprocessing.shared_volume_tasks``: per-item work over one volume, in workers."""
from __future__ import annotations

import numpy as np

from haemolynx.preprocessing.shared_volume_tasks import default_worker_count, run_volume_tasks


def _dtype_and_sum(volume, payload):
    return volume.dtype.str, int(volume[payload].sum()), payload


def test_workers_read_an_integer_volume_as_it_is_and_answer_in_order():
    bits = np.arange(40, dtype=np.uint8).reshape(4, 10)
    payloads = [i for i in range(4)]
    alone = run_volume_tasks(_dtype_and_sum, payloads, volume=bits, workers=1)
    pooled = run_volume_tasks(_dtype_and_sum, payloads, volume=bits, workers=2, min_payloads=1)
    assert pooled == alone == [(bits.dtype.str, int(bits[i].sum()), i) for i in payloads]


def _dies_in_a_worker(volume, payload):
    """Ends any worker it runs in at once, as the OS ends one that ran out of
    memory; in the parent process it answers."""
    import multiprocessing
    import os

    if multiprocessing.parent_process() is not None:
        os._exit(1)
    return payload, int(volume[payload].sum())


def test_a_worker_killed_mid_run_falls_back_to_this_process_with_the_same_answers(caplog):
    """The run is not lost when a worker dies (RELEASE_REVIEW.md memory.4b):
    the pool breaks, and every payload is measured here instead."""
    volume = np.arange(40, dtype=np.int32).reshape(4, 10)
    payloads = list(range(4))

    pooled = run_volume_tasks(_dies_in_a_worker, payloads, volume=volume, workers=2, min_payloads=1)

    assert pooled == [(p, int(volume[p].sum())) for p in payloads]
    assert "measuring in this one" in caplog.text


def test_a_worker_killed_mid_stream_finishes_what_is_left_here(tmp_path, caplog):
    from haemolynx.preprocessing.shared_volume_tasks import iter_volume_tasks

    volume = np.arange(60, dtype=np.int32).reshape(6, 10)
    payloads = list(range(6))

    streamed = list(iter_volume_tasks(
        _dies_in_a_worker, payloads, volume=volume, workers=2, memmap_directory=tmp_path,
    ))

    assert streamed == [(p, int(volume[p].sum())) for p in payloads]
    assert "finishing in this one" in caplog.text


def test_too_few_payloads_stay_in_this_process():
    seen = run_volume_tasks(_dtype_and_sum, [0], volume=np.ones((2, 2), dtype=bool), workers=4, min_payloads=2)
    assert seen == [("|b1", 2, 0)]


def test_the_default_worker_count_leaves_a_core_and_is_at_least_one():
    assert 1 <= default_worker_count() <= 8


def _row_sum_and_dtypes(volumes, payload):
    image, mask = volumes
    return payload, int(image[payload].sum()), bool(mask[payload].any()), image.dtype.str


def test_iter_volume_tasks_answers_in_order_and_maps_a_whole_memmap_as_it_is(tmp_path, monkeypatch):
    """Results come back one at a time, in order, as run_volume_tasks's do; a
    volume already the whole of a memmap is mapped by the workers from its
    own file, and only the one in RAM is copied."""
    import haemolynx.preprocessing.memmap_support as memmap_support
    from haemolynx.preprocessing.shared_volume_tasks import iter_volume_tasks

    on_disk = memmap_support.new_memmap_array((12, 7), np.int32, directory=tmp_path)
    on_disk[...] = np.arange(84, dtype=np.int32).reshape(12, 7)
    on_disk.flush()
    in_ram = np.zeros((12, 7), dtype=bool)
    in_ram[::3] = True
    copies = []
    real = memmap_support.new_memmap_array
    monkeypatch.setattr(
        memmap_support, "new_memmap_array", lambda *a, **k: copies.append(a[0]) or real(*a, **k)
    )
    payloads = list(range(12))

    pooled = list(
        iter_volume_tasks(
            _row_sum_and_dtypes, payloads, volume=(np.asarray(on_disk), in_ram), workers=2,
            memmap_directory=tmp_path,
        )
    )

    alone = [_row_sum_and_dtypes((on_disk, in_ram), p) for p in payloads]
    assert pooled == alone
    assert copies == [(12, 7)], "only the volume in RAM is copied"
    memmap_support.release_memmap_array(on_disk)


def test_only_the_whole_of_a_memmap_is_shared_by_its_file(tmp_path):
    from haemolynx.preprocessing.memmap_support import new_memmap_array, release_memmap_array
    from haemolynx.preprocessing.shared_volume_tasks import _whole_file

    whole = new_memmap_array((4, 5, 6), bool, directory=tmp_path)
    npy = np.lib.format.open_memmap(str(tmp_path / "a.npy"), mode="w+", dtype=np.uint8, shape=(3, 4))

    assert _whole_file(whole) == (str(whole.filename), "|b1", (4, 5, 6), 0)
    assert _whole_file(np.asarray(whole))[:3] == (str(whole.filename), "|b1", (4, 5, 6))
    assert _whole_file(npy)[3] > 0, "a .npy file's header comes first"
    assert _whole_file(whole[1:]) is None
    assert _whole_file(whole[:, :, ::2]) is None
    assert _whole_file(np.zeros((4, 5, 6), dtype=bool)) is None
    release_memmap_array(whole)
    del npy


def _timed_nap(volume, payload):
    import time

    start = time.monotonic()
    time.sleep(0.4)
    return payload, start, time.monotonic()


def test_free_memory_running_short_holds_back_further_tasks(tmp_path, monkeypatch):
    """With *task_bytes* given, free memory is read again before each task
    goes out: short of the reserve, the run waits for the one out (flushing
    first) instead of starting more -- one at a time, the same results."""
    import haemolynx.haemodynamics.resistance as resistance
    from haemolynx.preprocessing import shared_volume_tasks
    from haemolynx.preprocessing.shared_volume_tasks import iter_volume_tasks

    volume = np.zeros((4, 4), dtype=bool)
    flushes = []
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: shared_volume_tasks.MEMORY_RESERVE_BYTES)

    short = list(
        iter_volume_tasks(
            _timed_nap, list(range(4)), volume=volume, workers=2, memmap_directory=tmp_path,
            task_bytes=1 << 20, on_low_memory=lambda: flushes.append(1),
        )
    )

    assert [p for p, _s, _e in short] == list(range(4))
    spans = sorted((start, end) for _p, start, end in short)
    assert all(later[0] >= earlier[1] for earlier, later in zip(spans, spans[1:])), "one at a time"
    assert flushes, "the caller's memmaps were flushed first"

    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 64 << 30)
    plenty = list(
        iter_volume_tasks(
            _timed_nap, list(range(4)), volume=volume, workers=2, memmap_directory=tmp_path,
            task_bytes=1 << 20,
        )
    )
    spans = sorted((start, end) for _p, start, end in plenty)
    assert any(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:])), "two at once"


def _pid_and_nap(volume, payload):
    import os
    import time

    start = time.monotonic()
    time.sleep(0.3)
    return payload, os.getpid(), start, time.monotonic(), int(volume[payload].sum())


def _most_at_once(spans):
    edges = sorted([(start, 1) for start, _end in spans] + [(end, -1) for _start, end in spans])
    most = running = 0
    for _t, step in edges:
        running += step
        most = max(most, running)
    return most


def test_kept_workers_carry_one_pool_through_several_calls(tmp_path):
    """Inside kept_workers, a second call runs in the processes the first
    started (they import the whole package as they start); a call wanting
    fewer runs no more than that at once, one wanting more starts a larger
    pool; outside the block every call has its own. Same answers throughout."""
    import psutil

    from haemolynx.preprocessing import shared_volume_tasks
    from haemolynx.preprocessing.shared_volume_tasks import iter_volume_tasks, kept_workers

    volume = np.arange(24, dtype=np.int32).reshape(6, 4)
    payloads = list(range(6))
    expected = [int(volume[p].sum()) for p in payloads]

    def run(workers):
        out = list(iter_volume_tasks(
            _pid_and_nap, payloads, volume=volume, workers=workers, memmap_directory=tmp_path,
        ))
        assert [p for p, *_ in out] == payloads and [s for *_, s in out] == expected
        return {pid for _p, pid, *_ in out}, [(start, end) for _p, _pid, start, end, _s in out]

    with kept_workers():
        first, _spans = run(3)
        fewer, spans = run(2)
        more, _spans = run(4)
        again, _spans = run(4)
        kept_pids = set(more)
    assert fewer <= first, "the second call ran in the first call's workers"
    assert _most_at_once(spans) <= 2, "no more at once than the call asked for"
    assert not more & first, "a larger pool started in place of the smaller"
    assert again == more
    assert shared_volume_tasks._KEPT["pool"] is None, "stopped as the block ended"
    assert not [pid for pid in kept_pids if psutil.pid_exists(pid) and psutil.Process(pid).is_running()
                and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]

    alone, _spans = run(2)
    assert not alone & more, "outside the block a call starts its own"


def test_close_kept_workers_stops_them_before_the_block_ends(tmp_path):
    from haemolynx.preprocessing import shared_volume_tasks
    from haemolynx.preprocessing.shared_volume_tasks import close_kept_workers, iter_volume_tasks, kept_workers

    volume = np.ones((4, 4), dtype=bool)
    with kept_workers():
        before = {pid for _p, pid, *_ in iter_volume_tasks(
            _pid_and_nap, list(range(4)), volume=volume, workers=2, memmap_directory=tmp_path,
        )}
        close_kept_workers()
        assert shared_volume_tasks._KEPT["pool"] is None
        after = {pid for _p, pid, *_ in iter_volume_tasks(
            _pid_and_nap, list(range(4)), volume=volume, workers=2, memmap_directory=tmp_path,
        )}
    assert not before & after
    assert shared_volume_tasks._KEPT["pool"] is None


def test_handing_this_process_s_own_pages_back_can_be_the_room_it_needed(tmp_path, monkeypatch):
    """Short of memory, a run first hands back its own volumes' pages (on
    Windows not free memory until then); when that is the room it needed, the
    next task goes out at once rather than waiting for the one out."""
    import haemolynx.haemodynamics.resistance as resistance
    from haemolynx.preprocessing import memmap_support, shared_volume_tasks
    from haemolynx.preprocessing.shared_volume_tasks import iter_volume_tasks

    volume = memmap_support.new_memmap_array((4, 4), bool, directory=tmp_path)
    handed_back = []
    monkeypatch.setattr(memmap_support, "release_resident_pages", lambda *a: handed_back.append(a))
    monkeypatch.setattr(
        resistance,
        "available_memory_bytes",
        lambda: (64 << 30) if handed_back else shared_volume_tasks.MEMORY_RESERVE_BYTES,
    )

    out = list(
        iter_volume_tasks(
            _timed_nap, list(range(4)), volume=volume, workers=2, memmap_directory=tmp_path,
            task_bytes=1 << 20,
        )
    )

    assert [p for p, _s, _e in out] == list(range(4))
    assert handed_back and any(each is volume for each in handed_back[0])
    spans = sorted((start, end) for _p, start, end in out)
    assert any(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:])), "two at once"
    memmap_support.release_memmap_array(volume)
