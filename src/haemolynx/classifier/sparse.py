"""A labels volume holding only what has been painted, for images far larger than memory.

A brush stroke touches a few thousand voxels of a volume that may hold a
hundred billion: :class:`SparseLabels` keeps the volume as chunks, one only
where something was written, and answers any other read with zeros. napari
paints into it and draws a plane of it as it would a NumPy array; the
classifier lists its painted voxels (:meth:`SparseLabels.points`) without
scanning the whole volume.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

#: The chunk the volume is kept in, (z, y, x): thin in z, as strokes are
#: painted a plane at a time.
CHUNK = (8, 128, 128)


class SparseLabels:
    """A uint8 labels volume of *shape*, all 0 until written. Indexed as a
    NumPy array is with slices and integers (a box), or with index arrays on
    every axis (points, as napari paints)."""

    def __init__(self, shape, dtype=np.uint8, chunk=CHUNK):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.chunk = tuple(int(c) for c in chunk)
        self._chunks: dict[tuple[int, ...], np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def __len__(self) -> int:
        return self.shape[0]

    def _key(self, key) -> tuple[list, bool]:
        """``(parts, points)``: each axis's indices -- an index array for a
        slice or for points, an int for an integer -- and whether the key
        names points (index arrays) rather than a box."""
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = key.index(Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim - len(key) + 1) + key[at + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim:
            raise IndexError(f"too many indices for a {self.ndim}D volume")
        points = any(not isinstance(k, (slice, int, np.integer)) for k in key)
        parts: list = []
        for part, n in zip(key, self.shape):
            if isinstance(part, slice):
                if points:
                    raise IndexError("index arrays on every axis, or slices and integers, not both")
                parts.append(np.arange(*part.indices(n)))
            elif isinstance(part, (int, np.integer)):
                parts.append(int(part) + (n if part < 0 else 0))
            else:
                parts.append(np.asarray(part, dtype=np.int64))
        return parts, points

    # -------------------------------------------------------------- reading

    def __getitem__(self, key: Any) -> np.ndarray:
        parts, points = self._key(key)
        if points:
            arrays = np.broadcast_arrays(*[np.asarray(p, dtype=np.int64) for p in parts])
            values = self._read_points(np.stack([a.ravel() for a in arrays], axis=1))
            return values.reshape(arrays[0].shape)
        axes = [np.atleast_1d(np.asarray(p, dtype=np.int64)) for p in parts]
        from .memory import refuse_if_too_large

        refuse_if_too_large(
            int(np.prod([len(a) for a in axes], dtype=np.int64)) * self.dtype.itemsize, "the painted labels"
        )
        out = np.zeros(tuple(len(a) for a in axes), dtype=self.dtype)
        keep = [len(a) for a, p in zip(axes, parts) if not isinstance(p, int)]
        with self._lock:
            chunks = list(self._chunks.items())
        if not chunks or not out.size:
            return out.reshape(keep)
        lows = [int(a.min()) for a in axes]
        highs = [int(a.max()) for a in axes]
        for index, data in chunks:
            origin = [i * c for i, c in zip(index, self.chunk)]
            if any(o > hi or o + n <= lo for o, n, lo, hi in zip(origin, data.shape, lows, highs)):
                continue
            where, local = [], []
            for axis, a in enumerate(axes):
                inside = (a >= origin[axis]) & (a < origin[axis] + data.shape[axis])
                where.append(np.nonzero(inside)[0])
                local.append(a[inside] - origin[axis])
            if all(len(w) for w in where):
                out[np.ix_(*where)] = data[np.ix_(*local)]
        return out.reshape(keep)

    def _read_points(self, points: np.ndarray) -> np.ndarray:
        out = np.zeros(len(points), dtype=self.dtype)
        if not len(points):
            return out
        index = points // np.asarray(self.chunk)
        with self._lock:
            for key in {tuple(row) for row in index.tolist()}:
                data = self._chunks.get(key)
                if data is None:
                    continue
                rows = np.all(index == key, axis=1)
                local = points[rows] - np.asarray(key) * np.asarray(self.chunk)
                out[rows] = data[tuple(local.T)]
        return out

    def __array__(self, dtype=None, copy=None):
        whole = self[(slice(None),) * self.ndim]
        return whole if dtype is None else whole.astype(dtype)

    # -------------------------------------------------------------- writing

    def __setitem__(self, key: Any, value) -> None:
        parts, points = self._key(key)
        if points:
            arrays = np.broadcast_arrays(*[np.asarray(p, dtype=np.int64) for p in parts])
            flat = np.stack([a.ravel() for a in arrays], axis=1)
            values = np.broadcast_to(np.asarray(value, dtype=self.dtype), arrays[0].shape).ravel()
        else:
            axes = [np.atleast_1d(np.asarray(p, dtype=np.int64)) for p in parts]
            grids = np.meshgrid(*axes, indexing="ij")
            flat = np.stack([g.ravel() for g in grids], axis=1)
            box = tuple(len(a) for a in axes)
            given = np.asarray(value, dtype=self.dtype)
            kept = [len(a) for a, p in zip(axes, parts) if not isinstance(p, int)]
            values = np.broadcast_to(given.reshape(given.shape if given.ndim == 0 else kept), kept).reshape(box).ravel()
        self._write(flat, values)

    def _write(self, points: np.ndarray, values: np.ndarray) -> None:
        if not len(points):
            return
        index = points // np.asarray(self.chunk)
        order = np.lexsort(index.T[::-1])
        index, points, values = index[order], points[order], values[order]
        breaks = np.nonzero(np.any(np.diff(index, axis=0) != 0, axis=1))[0] + 1
        with self._lock:
            for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(index)]):
                key = tuple(int(v) for v in index[start])
                data = self._chunks.get(key)
                if data is None:
                    if not values[start:stop].any():
                        continue
                    extent = [min(c, n - k * c) for k, c, n in zip(key, self.chunk, self.shape)]
                    data = np.zeros(extent, dtype=self.dtype)
                    self._chunks[key] = data
                local = points[start:stop] - np.asarray(key) * np.asarray(self.chunk)
                data[tuple(local.T)] = values[start:stop]
                if not data.any():
                    del self._chunks[key]

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    # ------------------------------------------------------------- listing

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, 3) int32, values (n,))`` of every nonzero voxel, in
        C order, as ``np.argwhere`` on the dense volume gives them."""
        with self._lock:
            chunks = sorted(self._chunks.items())
        found, values = [], []
        for key, data in chunks:
            local = np.argwhere(data)
            if len(local):
                found.append(local + np.asarray(key) * np.asarray(self.chunk))
                values.append(data[tuple(local.T)])
        if not found:
            return np.empty((0, self.ndim), np.int32), np.empty(0, self.dtype)
        points = np.concatenate(found).astype(np.int32)
        values = np.concatenate(values)
        order = np.lexsort(points.T[::-1])
        return points[order], values[order]


def painted_points(labels) -> tuple[np.ndarray, np.ndarray]:
    """``(points, values)`` of every nonzero voxel of *labels*: a
    :class:`SparseLabels` lists its own, any other array is scanned."""
    if isinstance(labels, SparseLabels):
        return labels.points()
    array = np.asarray(labels)
    points = np.argwhere(array > 0).astype(np.int32)
    return points, array[tuple(points.T)]
