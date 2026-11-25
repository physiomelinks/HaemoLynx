"""Reusing a trial's result when an optimiser asks for the same trial again.

Both searches recompute what they have already computed: a sweep's baseline is
the same trial its current value gets as a candidate, and the skeleton or graph
a sweep leaves behind is its winner's trial, derived once more. A
:class:`TrialCache` remembers the last few results of one trial function, keyed
by the settings the trial read and by the arrays it read them from, and hands
back a copy -- what a caller does to its result never reaches the next caller.

An input array is recognised by identity, not content, and the cache holds only
a weak reference to it: once a search replaces its mask or skeleton with a new
array, results computed from the old one can no longer be returned.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence, TypeVar

import networkx as nx
import numpy as np

T = TypeVar("T")

#: Results kept per cache, oldest dropped first.
DEFAULT_MAX_ENTRIES = 12
#: Bytes of stored arrays kept per cache; the newest result is kept even when
#: it alone is larger. Boolean volumes are stored eight to a byte.
DEFAULT_MAX_BYTES = 256 * 2**20


def freeze(value: Any) -> Hashable:
    """*value* as a hashable key: mappings as sorted item tuples, sequences and
    arrays as tuples. Raises ``TypeError`` for anything else unhashable."""
    if isinstance(value, Mapping):
        return tuple(sorted(((str(k), freeze(v)) for k, v in value.items()), key=lambda kv: kv[0]))
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    hash(value)
    return value


class _Stored:
    """One result, stored so that every restore is a fresh copy."""

    def __init__(self, value: Any) -> None:
        self.kind = "plain"
        self.nbytes = 0
        if isinstance(value, np.ndarray) and value.dtype == bool:
            self.kind = "bool_array"
            self.shape = value.shape
            self.data = np.packbits(value, axis=None)
            self.nbytes = int(self.data.nbytes)
        elif isinstance(value, np.ndarray):
            self.kind = "array"
            self.data = value.copy()
            self.nbytes = int(self.data.nbytes)
        elif isinstance(value, nx.Graph):
            self.kind = "graph"
            self.data = value.copy()
        elif isinstance(value, tuple):
            self.kind = "tuple"
            self.data = tuple(_Stored(v) for v in value)
            self.nbytes = sum(item.nbytes for item in self.data)
        else:
            self.data = value

    def restore(self) -> Any:
        if self.kind == "bool_array":
            size = int(np.prod(self.shape))
            return np.unpackbits(self.data, count=size).reshape(self.shape).view(bool)
        if self.kind in ("array", "graph"):
            return self.data.copy()
        if self.kind == "tuple":
            return tuple(item.restore() for item in self.data)
        return self.data


def _reference(obj: Any) -> Callable[[], Any]:
    if obj is None:
        return lambda: None
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


class TrialCache:
    """The recent results of one trial function.

    :meth:`get_or_compute` returns a copy of the result stored for the same
    *inputs* (by identity) and *settings* (by value), or runs *compute* and
    stores what it returns. A *compute* that raises stores nothing, so a
    failing trial fails again the next time it is asked for.
    """

    def __init__(
        self,
        max_entries: Optional[int] = DEFAULT_MAX_ENTRIES,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[tuple[Callable[[], Any], ...], _Stored]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get_or_compute(self, inputs: Sequence[Any], settings: Any, compute: Callable[[], T]) -> T:
        try:
            key = (tuple(id(obj) for obj in inputs), freeze(settings))
        except TypeError:
            self.misses += 1
            return compute()
        entry = self._entries.get(key)
        if entry is not None:
            references, stored = entry
            if all(ref() is obj for ref, obj in zip(references, inputs)):
                self._entries.move_to_end(key)
                self.hits += 1
                return stored.restore()
            self._drop(key)
        value = compute()
        self.misses += 1
        stored = _Stored(value)
        self._entries[key] = (tuple(_reference(obj) for obj in inputs), stored)
        self._bytes += stored.nbytes
        self._evict()
        return value

    def _drop(self, key: Hashable) -> None:
        _references, stored = self._entries.pop(key)
        self._bytes -= stored.nbytes

    def _evict(self) -> None:
        while len(self._entries) > 1 and (
            (self.max_entries is not None and len(self._entries) > self.max_entries)
            or (self.max_bytes is not None and self._bytes > self.max_bytes)
        ):
            self._drop(next(iter(self._entries)))
