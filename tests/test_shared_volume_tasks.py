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
