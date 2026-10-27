"""The one rule for whether two values hold the same content, shared by every comparer.

``batch_compare`` (with ``atol = 0``), ``reference_run`` and the VTK comparer in
``driver_snapshots`` all decide equality here, so their rules cannot drift apart.

- A float ``new`` passes against its reference ``ref`` when
  ``|new - ref| <= rtol * |ref| + atol``. ``rtol`` scales by the reference side.
- A NaN equals a NaN and nothing else; an infinity equals only itself.
- Everything else is exact, and types are strict: ``1`` is not ``1.0``, a ``numpy.float64``
  is not a ``float``, a tuple is not a list, and arrays must share shape and dtype.

``max_scaled_atol`` gives the absolute floor the reference run uses, ``rtol`` times the largest
finite magnitude of the values it is scaled by, so values near zero do not fail on rounding.
"""
from __future__ import annotations

import numpy as np


def within(ref, new, rtol: float = 0.0, atol: float = 0.0) -> np.ndarray:
    """Element by element, whether ``new`` passes against ``ref`` under the rule."""
    ref, new = np.asarray(ref, dtype=float), np.asarray(new, dtype=float)
    with np.errstate(invalid="ignore"):
        result = np.abs(new - ref) <= rtol * np.abs(ref) + atol
    nan = np.isnan(ref) | np.isnan(new)
    infinite = (np.isinf(ref) | np.isinf(new)) & ~nan
    result = np.where(infinite, ref == new, result)
    return np.where(nan, np.isnan(ref) & np.isnan(new), result)


def close(ref, new, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Whether the float ``new`` passes against ``ref`` under the rule."""
    return bool(within(ref, new, rtol, atol))


def max_scaled_atol(values, rtol: float) -> float:
    """``rtol`` times the largest finite magnitude among ``values``; 0 when there is none."""
    values = np.abs(np.asarray(values, dtype=float).ravel())
    values = values[np.isfinite(values)]
    return float(rtol * values.max()) if values.size else 0.0


def float_pairs(ref, new):
    """The float leaves of two values side by side, or None when they differ in anything but
    a float: a type, a length, a key, a shape or dtype, an integer or text."""
    if type(ref) is not type(new):
        return None
    if isinstance(ref, dict):
        if ref.keys() != new.keys():
            return None
        return _pairs_of(ref.values(), (new[key] for key in ref))
    if isinstance(ref, (list, tuple)):
        if len(ref) != len(new):
            return None
        return _pairs_of(ref, new)
    if isinstance(ref, np.ndarray):
        if ref.shape != new.shape or ref.dtype != new.dtype:
            return None
        if ref.dtype.kind == "f":
            return list(zip(ref.ravel().tolist(), new.ravel().tolist()))
        return [] if np.array_equal(ref, new) else None
    if isinstance(ref, (float, np.floating)):
        return [(float(ref), float(new))]
    return [] if bool(ref == new) else None


def _pairs_of(refs, news):
    pairs = []
    for ref, new in zip(refs, news):
        inner = float_pairs(ref, new)
        if inner is None:
            return None
        pairs += inner
    return pairs


def values_equal(ref, new, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Same types and structure, and every float leaf under the rule."""
    pairs = float_pairs(ref, new)
    if pairs is None:
        return False
    return not pairs or bool(within(*zip(*pairs), rtol, atol).all())


def gap_text(ref, new) -> str:
    """The largest absolute and relative gap between paired floats, as a report clause
    (``", max abs 2e-05, max rel 1e-06"``), or "" when no pair is finite on both sides. A gap
    from a reference of 0 is infinitely relative."""
    ref, new = np.asarray(ref, dtype=float).ravel(), np.asarray(new, dtype=float).ravel()
    finite = np.isfinite(ref) & np.isfinite(new)
    if not finite.any():
        return ""
    gaps, scale = np.abs(new[finite] - ref[finite]), np.abs(ref[finite])
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(gaps == 0, 0.0, gaps / scale)
    return f", max abs {gaps.max():.3g}, max rel {relative.max():.3g}"
