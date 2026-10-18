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
