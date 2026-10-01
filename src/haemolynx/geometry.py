"""Polyline geometry shared by graph consumers.

Centerline polylines are used by more than one subpackage — haemodynamics walks
them to place constrictions, visualization walks them to emit VTK cells — so the
arc-length parameterisation they both need lives here rather than in either one.
"""
from __future__ import annotations

import numpy as np


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    """Arc length at each vertex of a polyline, starting at 0.

    ``points`` is an ``(n, 3)`` array of physical ``(z, y, x)`` coordinates in
    microns; the returned array has ``n`` entries, the last being the total
    length.
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def resample_at_step(points: np.ndarray, step_um: float) -> np.ndarray:
    """Points evenly spaced along a polyline, at most *step_um* apart, from its
    first point to its last.

    Unlike its own vertices, which a centreline places a voxel apart -- a
    quarter of the length along z of what they span in-plane on a typical
    stack -- these weight every micron of the vessel the same, so a statistic
    taken over them is one over the vessel's length.
    """
    if step_um <= 0:
        raise ValueError(f"step_um must be > 0, got {step_um}.")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return points
    lengths = cumulative_lengths(points)
    total = float(lengths[-1])
    if total <= 0.0:
        return points  # every point coincident: nothing to space out
    keep = np.concatenate(([True], np.diff(lengths) > 0.0))
    points, lengths = points[keep], lengths[keep]
    targets = np.linspace(0.0, total, int(np.ceil(total / float(step_um))) + 1)
    return np.column_stack(
        [np.interp(targets, lengths, points[:, axis]) for axis in range(points.shape[1])]
    )
