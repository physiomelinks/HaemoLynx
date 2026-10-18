"""Polylines of a chosen length between two fixed ends.

A synthetic vessel is drawn as a smooth curve from one node to the next whose
arc length is the length it was given -- the vessel's tortuosity -- and which
leaves each node *along the chord*, so the emergence and daughter angles the
statistics read off the first ten microns of a vessel are the angles the
generator chose, not the bend's. The offset from the chord is
``A * sin^2(pi s)`` in one perpendicular direction and
``A/2 * sin(2 pi s + psi) * sin^2(pi s)`` in the other, random phases, the
amplitude ``A`` found by bisection; ``sin^2`` has zero slope at both ends.
:func:`bend_to_lengths` does every edge at once (the batch form of
``visualization/secomb_io._fit_length``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from haemolynx.graph._helpers import calculate_path_length, densify_polyline

#: Vertices per bent edge before densifying: the arc length is the polyline's
#: own, so densifying (linear between them) keeps it exactly.
DEFAULT_POINTS = 17


def perpendicular_frame(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each unit tangent of *tangents* (E, 3)."""
    t = np.asarray(tangents, dtype=float)
    helper = np.where(np.abs(t[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    n1 = np.cross(t, helper)
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-12)
    n2 = np.cross(t, n1)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    return n1, n2


def _lengths(paths: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)


def bend_to_lengths(
    starts: np.ndarray,
    ends: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    *,
    points: int = DEFAULT_POINTS,
    iterations: int = 60,
) -> list[np.ndarray]:
    """One polyline per edge from ``starts[i]`` to ``ends[i]`` whose length is
    ``targets[i]``; a target no longer than the chord gives the straight
    line (two points). Every random phase is drawn from *rng*."""
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    ends = np.asarray(ends, dtype=float).reshape(-1, 3)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    count = len(starts)
    if count == 0:
        return []
    chords = ends - starts
    chord_lengths = np.linalg.norm(chords, axis=1)
    straight = (targets <= chord_lengths * (1.0 + 1e-9)) | (chord_lengths <= 0.0)
    out: list[Optional[np.ndarray]] = [None] * count
    for i in np.flatnonzero(straight):
        out[i] = np.stack([starts[i], ends[i]])
    bent = np.flatnonzero(~straight)
    if len(bent) == 0:
        return out  # type: ignore[return-value]

    b_starts, b_chords, b_lengths, b_targets = starts[bent], chords[bent], chord_lengths[bent], targets[bent]
    tangents = b_chords / b_lengths[:, None]
    n1, n2 = perpendicular_frame(tangents)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=len(bent))
    psi = rng.uniform(0.0, 2.0 * np.pi, size=len(bent))
    d1 = np.cos(phi)[:, None] * n1 + np.sin(phi)[:, None] * n2
    d2 = -np.sin(phi)[:, None] * n1 + np.cos(phi)[:, None] * n2
    s = np.linspace(0.0, 1.0, max(int(points), 3))
    envelope = np.sin(np.pi * s) ** 2
    base = b_starts[:, None, :] + s[None, :, None] * b_chords[:, None, :]
    wobble = 0.5 * np.sin(2.0 * np.pi * s[None, :] + psi[:, None]) * envelope[None, :]
    shape = envelope[None, :, None] * d1[:, None, :] + wobble[:, :, None] * d2[:, None, :]

    def paths(amplitude: np.ndarray) -> np.ndarray:
        return base + amplitude[:, None, None] * shape

    low = np.zeros(len(bent))
    high = np.maximum(b_targets, b_lengths)
    for _ in range(64):
        short = _lengths(paths(high)) < b_targets
        if not short.any():
            break
        high[short] *= 2.0
    for _ in range(int(iterations)):
        mid = 0.5 * (low + high)
        below = _lengths(paths(mid)) < b_targets
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    final = paths(0.5 * (low + high))
    for j, i in enumerate(bent):
        out[i] = final[j]
    return out  # type: ignore[return-value]


def bend_path(points: Sequence[np.ndarray], target: float, rng: np.random.Generator, *, bend_points: int = DEFAULT_POINTS) -> np.ndarray:
    """A polyline through *points* (two or more; the interior ones are kept)
    of length *target*: each leg bent to its share of the target by chord."""
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        raise ValueError("a path needs at least two points")
    chords = np.array([np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])])
    total = float(chords.sum())
    shares = chords / total * float(target) if total > 0 else np.full(len(chords), float(target) / len(chords))
    legs = bend_to_lengths(np.stack(pts[:-1]), np.stack(pts[1:]), shares, rng, points=bend_points)
    path = [legs[0]]
    for leg in legs[1:]:
        path.append(leg[1:])
    return np.concatenate(path, axis=0)


def voxels_and_length(path: np.ndarray, *, step_um: float = 0.5) -> tuple[list, float]:
    """``(voxels, length)`` of a path densified to *step_um*: the list the
    graph's ``voxels`` holds and the length measured along it."""
    dense = densify_polyline(np.asarray(path, dtype=float), max_step_um=step_um)
    voxels = [[float(v) for v in point] for point in dense]
    return voxels, calculate_path_length(voxels)


def densify_paths(paths: Sequence[np.ndarray], *, step_um: float = 0.5) -> tuple[list, np.ndarray]:
    """Every path of *paths* sampled so no step exceeds *step_um*, keeping
    its vertices (so each length is kept exactly), all at once: the
    ``(voxels, length)`` of each, as :func:`voxels_and_length` gives them."""
    if not paths:
        return [], np.zeros(0)
    counts = np.array([len(p) for p in paths])
    stacked = np.concatenate([np.asarray(p, dtype=float) for p in paths], axis=0)
    # Segment s runs from stacked[i] to stacked[i + 1] for every i that is
    # not a path's last point.
    last = np.cumsum(counts) - 1
    is_start = np.ones(len(stacked), dtype=bool)
    is_start[last] = False
    starts = np.flatnonzero(is_start)
    a, b = stacked[starts], stacked[starts + 1]
    seg_len = np.linalg.norm(b - a, axis=1)
    steps = np.maximum(np.ceil(seg_len / float(step_um) - 1e-12).astype(int), 1)
    total = int(steps.sum())
    seg_of = np.repeat(np.arange(len(starts)), steps)
    offsets = np.repeat(np.cumsum(steps) - steps, steps)
    j = np.arange(total) - offsets
    t = (j / steps[seg_of])[:, None]
    dense = a[seg_of] + t * (b[seg_of] - a[seg_of])
    path_of_seg = np.repeat(np.arange(len(paths)), counts - 1)
    points_per_path = np.bincount(path_of_seg, weights=steps, minlength=len(paths)).astype(int)
    lengths = np.bincount(path_of_seg, weights=seg_len, minlength=len(paths))
    out = []
    cursor = 0
    for i, n in enumerate(points_per_path):
        chunk = np.concatenate([dense[cursor:cursor + n], np.asarray(paths[i], dtype=float)[-1:]], axis=0)
        out.append(chunk.tolist())
        cursor += n
    return out, lengths


__all__ = ["DEFAULT_POINTS", "bend_path", "bend_to_lengths", "densify_paths", "perpendicular_frame", "voxels_and_length"]
