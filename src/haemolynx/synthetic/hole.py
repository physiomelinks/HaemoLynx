"""A box inside a synthetic network's that the network leaves empty
(``PLAN_synthetic_surround.md`` phase 2): the image's own box, which its own
graph fills, inside the surround drawn round it.

A :class:`Hole` is an axis-aligned box in the network's frame. The generator
keeps what it draws out of it: a tree vessel ends before it would enter it
(:meth:`Hole.entry_fraction`), the bed lays no vertex in it or within a margin
of it (:meth:`Hole.contains`) and joins no two vertices across it
(:meth:`Hole.crosses`), and a vessel's bow is turned off it (:meth:`Hole.depth`).
Its faces are not in it: a vessel may run along one, and the image's open ends
will lie on them.

Every hole check is here, and the generator asks one only when it was given a
hole, so a network drawn without one is drawn as it always was
(``tests/test_synthetic_hole.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: How far inside a face a point must be to be in the hole (um): a vessel
#: drawn along a face, or ending on one, is not in it.
FACE_TOLERANCE_UM = 1e-6
#: Points read along each step of a path when judging how deep it goes in.
DEPTH_SUBDIVISIONS = 8


@dataclass(frozen=True)
class Hole:
    """The box from :attr:`lo_um` to :attr:`hi_um`, ``(z, y, x)`` microns in
    the network's frame. See the module."""

    lo_um: tuple[float, float, float]
    hi_um: tuple[float, float, float]

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.hi_um, dtype=float) <= np.asarray(self.lo_um, dtype=float)):
            raise ValueError(f"a hole needs positive extent, got lo {self.lo_um} and hi {self.hi_um}")

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.lo_um, dtype=float)

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.hi_um, dtype=float)

    def volume_in(self, extent_um: Sequence[float]) -> float:
        """The hole's volume inside the box from 0 to *extent_um*."""
        extent = np.asarray(extent_um, dtype=float)
        sides = np.clip(np.minimum(self.hi, extent) - np.maximum(self.lo, 0.0), 0.0, None)
        return float(np.prod(sides))

    def share_of(self, extent_um: Sequence[float]) -> float:
        """The share of the box from 0 to *extent_um* the hole takes."""
        volume = float(np.prod(np.asarray(extent_um, dtype=float)))
        return self.volume_in(extent_um) / volume if volume > 0 else 0.0

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Whether each of *points* (``(n, 3)``, or one point) lies inside the
        hole grown by *margin* (um), more than :data:`FACE_TOLERANCE_UM` in."""
        p = np.asarray(points, dtype=float)
        grown = float(margin) - FACE_TOLERANCE_UM
        return np.all((p > self.lo - grown) & (p < self.hi + grown), axis=-1)

    def _clip(self, start: np.ndarray, end: np.ndarray, grown: float) -> Optional[tuple[float, float]]:
        """The stretch ``(t0, t1)`` of *start* -> *end* inside the hole grown
        by *grown*, as fractions of it, or None (Liang-Barsky)."""
        lo, hi = self.lo - grown, self.hi + grown
        t0, t1 = 0.0, 1.0
        d = end - start
        for axis in range(3):
            if d[axis] == 0.0:
                if not lo[axis] < start[axis] < hi[axis]:
                    return None
                continue
            ta, tb = (lo[axis] - start[axis]) / d[axis], (hi[axis] - start[axis]) / d[axis]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 >= t1:
                return None
        return t0, t1

    def crosses(self, start: Sequence[float], end: Sequence[float], clearance: float = 0.0) -> bool:
        """Whether the segment *start* -> *end* passes through the hole grown
        by *clearance* (um): along a face, or touching it, does not."""
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        clip = self._clip(a, b, float(clearance) - FACE_TOLERANCE_UM)
        return clip is not None and bool((clip[1] - clip[0]) * float(np.linalg.norm(b - a)) > FACE_TOLERANCE_UM)

    def crosses_each(self, starts: np.ndarray, ends: np.ndarray, clearance: float = 0.0) -> np.ndarray:
        """:meth:`crosses` for each pair of rows of *starts* and *ends*."""
        starts = np.asarray(starts, dtype=float).reshape(-1, 3)
        ends = np.asarray(ends, dtype=float).reshape(-1, 3)
        grown = float(clearance) - FACE_TOLERANCE_UM
        lo, hi = self.lo - grown, self.hi + grown
        d = ends - starts
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - starts) / d, (hi - starts) / d
        flat = d == 0.0
        inside_flat = (starts > lo) & (starts < hi)
        near = np.where(flat, np.where(inside_flat, -np.inf, np.inf), np.minimum(ta, tb))
        far = np.where(flat, np.where(inside_flat, np.inf, -np.inf), np.maximum(ta, tb))
        t0 = np.maximum(near.max(axis=1), 0.0)
        t1 = np.minimum(far.min(axis=1), 1.0)
        return (t1 - t0) * np.linalg.norm(d, axis=1) > FACE_TOLERANCE_UM

    def entry_fraction(self, start: Sequence[float], end: Sequence[float], clearance: float = 0.0) -> float:
        """Where the segment *start* -> *end* enters the hole grown by
        *clearance*, as a fraction of it (1 when it never does). From a
        *start* already within *clearance* of the hole, where it enters the
        hole itself."""
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if clearance > 0 and bool(self.contains(a, clearance)):
            clearance = 0.0
        if not self.crosses(a, b, clearance):
            return 1.0
        clip = self._clip(a, b, float(clearance) - FACE_TOLERANCE_UM)
        return 1.0 if clip is None else float(max(clip[0], 0.0))

    def depth(self, paths: np.ndarray, subdivisions: int = DEPTH_SUBDIVISIONS) -> np.ndarray:
        """How far each path (``(E, P, 3)``) goes into the hole: the deepest
        of its points' distance in from the nearest face, read at its
        vertices and *subdivisions* points along each step between them (a
        step between two vertices outside can cut a corner); 0 when none is in."""
        p = np.asarray(paths, dtype=float)
        if not len(p):
            return np.zeros(0)
        if subdivisions > 1 and p.shape[1] > 1:
            t = np.linspace(0.0, 1.0, int(subdivisions), endpoint=False)[None, None, :, None]
            between = p[:, :-1, None, :] + t * (p[:, 1:, None, :] - p[:, :-1, None, :])
            p = np.concatenate([between.reshape(len(p), -1, 3), p[:, -1:, :]], axis=1)
        inward = np.minimum(p - self.lo, self.hi - p).min(axis=-1)
        return np.maximum(inward.max(axis=-1), 0.0)

    def entered_by(self, paths: np.ndarray) -> np.ndarray:
        """Whether any step of each path (``(E, P, 3)``) passes through the
        hole (:meth:`crosses`, exactly)."""
        p = np.asarray(paths, dtype=float)
        if not len(p) or p.shape[1] < 2:
            return np.zeros(len(p), dtype=bool)
        steps = self.crosses_each(p[:, :-1, :].reshape(-1, 3), p[:, 1:, :].reshape(-1, 3))
        return steps.reshape(len(p), -1).any(axis=1)

    def grown(self, margin: float) -> "Hole":
        """The hole grown by *margin* (um) on every side."""
        return Hole(tuple(float(v) for v in self.lo - margin), tuple(float(v) for v in self.hi + margin))

    def push_out(
        self, point: Sequence[float], axes: Sequence[int], margin: float = 0.0, extent_um: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """*point* moved, along one of *axes*, out of the hole grown by
        *margin* by the shortest way that stays in the box from 0 to
        *extent_um*; as it was when already out, or when no way does."""
        p = np.asarray(point, dtype=float).copy()
        if not axes or not bool(self.contains(p, margin)):
            return p
        lo, hi = self.lo - margin, self.hi + margin
        top = np.full(3, np.inf) if extent_um is None else np.asarray(extent_um, dtype=float)
        moves = [(p[a] - lo[a], a, lo[a]) for a in axes if lo[a] >= 0.0]
        moves += [(hi[a] - p[a], a, hi[a]) for a in axes if hi[a] <= top[a]]
        if not moves:
            return p
        _distance, axis, value = min(moves)
        p[axis] = value
        return p

    def to_record(self) -> dict:
        return {"lo_um": [float(v) for v in self.lo_um], "hi_um": [float(v) for v in self.hi_um]}

    @classmethod
    def from_record(cls, record: Optional[dict]) -> Optional["Hole"]:
        if not record:
            return None
        return cls(tuple(float(v) for v in record["lo_um"]), tuple(float(v) for v in record["hi_um"]))


def centred_hole(extent_um: Sequence[float], inner_um: Sequence[float]) -> Hole:
    """A box of *inner_um* in the middle of the box from 0 to *extent_um*:
    an image's own box inside a surround drawn round it."""
    extent, inner = np.asarray(extent_um, dtype=float), np.asarray(inner_um, dtype=float)
    if np.any(inner > extent):
        raise ValueError(f"a hole of {inner.tolist()} um does not fit a box of {extent.tolist()} um")
    lo = (extent - inner) / 2.0
    return Hole(tuple(float(v) for v in lo), tuple(float(v) for v in lo + inner))


#: The graph key a network drawn round a hole records it under.
GRAPH_HOLE = "synthetic_hole_um"


__all__ = ["FACE_TOLERANCE_UM", "GRAPH_HOLE", "Hole", "centred_hole"]
