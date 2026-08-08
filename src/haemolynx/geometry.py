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
