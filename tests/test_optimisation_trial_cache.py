"""Unit tests for haemolynx.optimisation.trial_cache -- pure, synthetic data only."""
from __future__ import annotations

import gc

import networkx as nx
import numpy as np
import pytest

from haemolynx.optimisation.trial_cache import TrialCache, freeze


class _Counting:
    """A trial function that counts its calls and returns a fresh array."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> np.ndarray:
        self.calls += 1
        return self.value.copy()


def _volume() -> np.ndarray:
    volume = np.zeros((4, 5, 6), dtype=bool)
    volume[1:3, 2, 1:5] = True
    return volume


def test_the_same_inputs_and_settings_run_the_trial_once():
    cache = TrialCache()
    source = _volume()
    trial = _Counting(source)

    first = cache.get_or_compute((source,), {"radius": 2.0}, trial)
    second = cache.get_or_compute((source,), {"radius": 2.0}, trial)

    assert trial.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)
    np.testing.assert_array_equal(second, first)
    assert second.dtype == bool and second.shape == source.shape


def test_different_settings_run_the_trial_again():
    cache = TrialCache()
    source = _volume()
    trial = _Counting(source)

    cache.get_or_compute((source,), {"radius": 2.0}, trial)
    cache.get_or_compute((source,), {"radius": 3.0}, trial)

    assert trial.calls == 2


def test_an_input_is_recognised_by_identity_not_content():
    """A search replaces its mask with a new array when a cleanup step is
    decided; a result computed from the old one must not come back, even
    when the two happen to hold the same voxels."""
    cache = TrialCache()
    source = _volume()
    trial = _Counting(source)

    cache.get_or_compute((source,), {"radius": 2.0}, trial)
    cache.get_or_compute((source.copy(),), {"radius": 2.0}, trial)

    assert trial.calls == 2


def test_a_result_from_an_input_that_no_longer_exists_is_not_returned():
    cache = TrialCache()
    trial = _Counting(_volume())

    source = _volume()
    cache.get_or_compute((source,), {}, trial)
    del source
    gc.collect()
    # A new array may reuse the old one's id; the weak reference tells them apart.
    for _ in range(20):
        cache.get_or_compute((_volume(),), {}, trial)

    assert trial.calls == 21


def test_what_a_caller_does_to_its_result_never_reaches_the_next_caller():
    cache = TrialCache()
    source = _volume()
    trial = _Counting(source)

    first = cache.get_or_compute((source,), {}, trial)
    first[:] = False
    second = cache.get_or_compute((source,), {}, trial)
    second[0, 0, 0] = True
    third = cache.get_or_compute((source,), {}, trial)

    np.testing.assert_array_equal(third, source)


def test_a_graph_result_comes_back_as_a_copy():
    cache = TrialCache()
    source = _volume()

    def build() -> nx.MultiGraph:
        G = nx.MultiGraph()
        G.add_edge(0, 1, length=3.0)
        return G

    first = cache.get_or_compute((source,), {}, build)
    first.add_edge(1, 2, length=1.0)
    second = cache.get_or_compute((source,), {}, build)
    second.edges[0, 1, 0]["length"] = 99.0
    third = cache.get_or_compute((source,), {}, build)

    assert third.number_of_edges() == 1
    assert third.edges[0, 1, 0]["length"] == 3.0


def test_a_trial_that_raises_is_not_stored():
    cache = TrialCache()
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cache.get_or_compute((), {"x": 1}, failing)

    assert calls["n"] == 2
    assert len(cache) == 0


def test_the_oldest_result_is_dropped_first():
    cache = TrialCache(max_entries=2, max_bytes=None)
    source = _volume()
    trial = _Counting(source)

    for radius in (1.0, 2.0, 3.0):
        cache.get_or_compute((source,), {"radius": radius}, trial)
    cache.get_or_compute((source,), {"radius": 3.0}, trial)  # kept
    cache.get_or_compute((source,), {"radius": 1.0}, trial)  # dropped: runs again

    assert trial.calls == 4


def test_a_byte_budget_keeps_the_newest_result_even_when_it_alone_is_over():
    cache = TrialCache(max_entries=None, max_bytes=1)
    source = np.ones((64, 64), dtype=float)
    trial = _Counting(source)

    cache.get_or_compute((source,), {"k": 1}, trial)
    cache.get_or_compute((source,), {"k": 2}, trial)
    cache.get_or_compute((source,), {"k": 2}, trial)

    assert len(cache) == 1
    assert trial.calls == 2


def test_boolean_volumes_are_stored_eight_voxels_to_a_byte():
    cache = TrialCache()
    source = np.ones((80, 80, 80), dtype=bool)
    cache.get_or_compute((source,), {}, lambda: source.copy())

    assert cache._bytes == source.size // 8


def test_unhashable_settings_run_the_trial_every_time_rather_than_fail():
    cache = TrialCache()
    trial = _Counting(_volume())
    unhashable = {"labels": [{1, 2}]}

    cache.get_or_compute((), unhashable, trial)
    cache.get_or_compute((), unhashable, trial)

    assert trial.calls == 2


def test_freeze_ignores_key_order_and_turns_sequences_into_tuples():
    assert freeze({"a": 1, "b": [1, 2]}) == freeze({"b": (1, 2), "a": 1})
    assert freeze({"voxel": (0.5, 0.5, 2.0)}) != freeze({"voxel": (2.0, 0.5, 0.5)})
    assert freeze(np.float64(2.5)) == 2.5
    with pytest.raises(TypeError):
        freeze({"s": {1, 2}})
