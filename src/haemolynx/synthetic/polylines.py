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
``visualization/secomb_io._fit_length``). Given the box, a bow that would
leave it is turned about its chord until it stays inside (or leaves it
least): the turn keeps its length exactly. Given a curvature to reach (the
template's, ``synthetic.organisation``), a vessel is drawn with as many lobes
(:func:`lobes`, 1 to :data:`MAX_LOBES`) as bring its turn per 10 um nearest
it, at the same length: a curve of several lobes turns more than one bow of
the same tortuosity.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from haemolynx.graph._helpers import calculate_path_length, densify_polyline

from .organisation import CURVATURE_STEP_UM, _curvature

#: Vertices per bent edge before densifying: the arc length is the polyline's
#: own, so densifying (linear between them) keeps it exactly.
DEFAULT_POINTS = 17
#: Turns about the chord tried for a bow that leaves the box.
BOW_TURNS = 12
#: The most lobes a vessel's curve is given to reach the template's curvature,
#: and the vertices it is then drawn with (enough for that many).
MAX_LOBES = 4
LOBE_POINTS = 8 * MAX_LOBES + 1


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
    extent_um: Optional[np.ndarray] = None,
    curvature_deg: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """One polyline per edge from ``starts[i]`` to ``ends[i]`` whose length is
    ``targets[i]``; a target no longer than the chord gives the straight
    line (two points). Every random phase is drawn from *rng*. With
    *extent_um*, a bow leaving the box is turned about its chord to stay in
    it (no draw: the turns are fixed). With *curvature_deg* (per edge, NaN
    for none), an edge long enough to read it on takes the number of lobes,
    1 to :data:`MAX_LOBES`, whose curve turns nearest that many degrees per
    10 um (see :func:`lobes`)."""
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
    wanted = None if curvature_deg is None else np.asarray(curvature_deg, dtype=float).reshape(-1)[bent]
    if wanted is not None:
        points = max(int(points), LOBE_POINTS)
    s = np.linspace(0.0, 1.0, max(int(points), 3))
    envelope = np.sin(np.pi * s) ** 2
    base = b_starts[:, None, :] + s[None, :, None] * b_chords[:, None, :]
    wobble = 0.5 * np.sin(2.0 * np.pi * s[None, :] + psi[:, None]) * envelope[None, :]
    # Each edge's main offset: the one bow unless its curvature asks for more lobes.
    lobe_shape = np.repeat(envelope[None, :], len(bent), axis=0)

    def fit(rows: np.ndarray, main: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The amplitude bringing each of *rows* to its length with the
        offsets *main*, and the paths."""
        shape = main[:, :, None] * d1[rows][:, None, :] + wobble[rows][:, :, None] * d2[rows][:, None, :]

        def paths(amplitude: np.ndarray) -> np.ndarray:
            return base[rows] + amplitude[:, None, None] * shape

        low = np.zeros(len(rows))
        high = np.maximum(b_targets[rows], b_lengths[rows])
        for _ in range(64):
            short = _lengths(paths(high)) < b_targets[rows]
            if not short.any():
                break
            high[short] *= 2.0
        for _ in range(int(iterations)):
            mid = 0.5 * (low + high)
            below = _lengths(paths(mid)) < b_targets[rows]
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        amplitude = 0.5 * (low + high)
        return amplitude, paths(amplitude)

    amplitude, final = fit(np.arange(len(bent)), lobe_shape)
    if wanted is not None:
        chooses = np.flatnonzero(np.isfinite(wanted) & (b_targets >= 3.0 * CURVATURE_STEP_UM))
        if len(chooses):
            error = np.abs(np.array([_path_curvature(final[i]) for i in chooses]) - wanted[chooses])
            for k in range(2, MAX_LOBES + 1):
                main = np.repeat(lobes(s, k)[None, :], len(chooses), axis=0)
                k_amplitude, k_paths = fit(chooses, main)
                k_error = np.abs(np.array([_path_curvature(path) for path in k_paths]) - wanted[chooses])
                better = k_error < error
                rows = chooses[better]
                amplitude[rows], final[rows], lobe_shape[rows] = k_amplitude[better], k_paths[better], main[better]
                error[better] = k_error[better]
    if extent_um is not None:
        extent = np.asarray(extent_um, dtype=float)
        excursion = _excursion(final, extent)
        for turn in range(1, BOW_TURNS):
            outside = np.flatnonzero(excursion > 1e-9)
            if not len(outside):
                break
            theta = 2.0 * np.pi * turn / BOW_TURNS
            r1 = np.cos(theta) * d1[outside] + np.sin(theta) * d2[outside]
            r2 = -np.sin(theta) * d1[outside] + np.cos(theta) * d2[outside]
            turned = base[outside] + amplitude[outside, None, None] * (
                lobe_shape[outside][:, :, None] * r1[:, None, :] + wobble[outside][:, :, None] * r2[:, None, :]
            )
            better = _excursion(turned, extent) < excursion[outside]
            final[outside[better]] = turned[better]
            excursion[outside[better]] = _excursion(turned[better], extent)
    for j, i in enumerate(bent):
        out[i] = final[j]
    return out  # type: ignore[return-value]


def lobes(s: np.ndarray, k: int) -> np.ndarray:
    """``sin(pi s) sin(k pi s)``: *k* lobes along the chord, alternating
    sides, flat at both ends (so a vessel leaves its node along the chord);
    one lobe is the bow ``sin^2(pi s)``."""
    return np.sin(np.pi * s) * np.sin(k * np.pi * s)


def _path_curvature(path: np.ndarray) -> float:
    """Mean turn (degrees) per :data:`CURVATURE_STEP_UM` along *path*, as the
    organisation reads it (``organisation._curvature``); NaN when too short."""
    value = _curvature([np.asarray(path, dtype=float)])
    return float("nan") if value is None else value


def _excursion(paths: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """How far each path (E, P, 3) goes outside the box, 0 when it stays in."""
    below = np.max(-paths, axis=(1, 2))
    above = np.max(paths - extent[None, None, :], axis=(1, 2))
    return np.maximum(np.maximum(below, above), 0.0)


def bend_path(
    points: Sequence[np.ndarray],
    target: float,
    rng: np.random.Generator,
    *,
    bend_points: int = DEFAULT_POINTS,
    extent_um: Optional[np.ndarray] = None,
    curvature_deg: Optional[float] = None,
) -> np.ndarray:
    """A polyline through *points* (two or more; the interior ones are kept)
    of length *target*: each leg bent to its share of the target by chord
    (inside the box *extent_um*, toward *curvature_deg*, as
    :func:`bend_to_lengths` draws them)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        raise ValueError("a path needs at least two points")
    chords = np.array([np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])])
    total = float(chords.sum())
    shares = chords / total * float(target) if total > 0 else np.full(len(chords), float(target) / len(chords))
    curvature = None if curvature_deg is None else np.full(len(shares), float(curvature_deg))
    legs = bend_to_lengths(
        np.stack(pts[:-1]), np.stack(pts[1:]), shares, rng, points=bend_points, extent_um=extent_um, curvature_deg=curvature
    )
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
