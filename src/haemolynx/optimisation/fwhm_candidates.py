"""Per-setting candidate-value generation for the FWHM diameter-measurement
settings, derived from the graph's own edges and a baseline probe measurement.

Mirrors :mod:`.candidates`'s own pattern exactly: every function here looks at
a real measurement -- the sampled subgraph's own edge-length distribution, its
own inter-sample arc-length spacing, a single baseline probe run's own
observed diameter/pass-count distribution -- and turns it into a short list of
candidate values worth actually trying. None of these functions run the real
measurement or score anything; :mod:`.fwhm_search` does that, one setting (or
small joint group) at a time.

Every candidate list always includes the schema default (or the current
value, when a caller is re-optimising from a non-default starting point), and
every function degrades to ``[default]`` on pathological input (a single
edge, all-equal lengths, an empty baseline) rather than raising -- the same
contract :mod:`.candidates` documents for the Skeletonise/Graph search.

The one deliberate exception is the two exclusion-zone candidate functions'
own safety cap: a candidate at or beyond
:data:`haemolynx.haemodynamics.automated.MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH`
of the sampled subgraph's own median edge length would be silently clamped
right back down to that bound by
``measure_edge_diameters_fwhm_from_raw_tiff``'s own per-edge
``_length_scaled_exclusion`` -- proposing it as a distinct candidate would
just re-evaluate, at real measurement cost, a value indistinguishable from
one already tried.
"""
from __future__ import annotations

import numpy as np

from haemolynx.haemodynamics.automated import MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH


def _percentiles(values: np.ndarray, percentiles: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return []
    return [float(np.percentile(values, p)) for p in percentiles]


def _sorted_with_none_first(values: set) -> list:
    """Sort a candidate set that may contain ``None`` ("auto") among floats.

    Mirrors :func:`haemolynx.optimisation.candidates._sorted_with_none_first`
    -- duplicated rather than imported since that one is private to its own
    module.
    """
    return sorted(values, key=lambda v: (0, 0.0) if v is None else (1, float(v)))


# ---------------------------------------------------------------------------
# Group 1: exclusion zones
# ---------------------------------------------------------------------------
def exclusion_zone_candidates(edge_lengths_um: np.ndarray, default: float) -> list[float]:
    """Candidates for ``fwhm_branch_endpoint_exclusion_um`` /
    ``fwhm_junction_proximity_exclusion_um`` -- 0 (off), the current/default
    value, and half of the sampled subgraph's own 10th/25th percentile edge
    length, each capped at :data:`MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH` of
    the subgraph's median edge length (see module docstring)."""
    lengths = np.asarray(edge_lengths_um, dtype=float)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    candidates = {0.0, float(default)}
    if lengths.size:
        cap = MAX_EXCLUSION_FRACTION_OF_EDGE_LENGTH * float(np.median(lengths))
        for p in _percentiles(lengths, (10.0, 25.0)):
            candidates.add(round(min(0.5 * p, cap), 6))
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 2: extent and widening
# ---------------------------------------------------------------------------
def half_extent_candidates(baseline_diameters_um: np.ndarray, default: float) -> list[float]:
    """Candidates for ``fwhm_transverse_half_extent_um`` -- the current/default
    value plus half-extents derived from a baseline probe run's own observed
    diameter distribution (median, 75th percentile), each scaled by the
    schema's own ``min_total_extent_multiplier`` semantics: an initial
    half-extent of ``0.5 * multiplier_guess * diameter`` gives the widen loop
    a first pass already close to its target instead of starting from
    scratch. Uses a fixed multiplier of 3 (the schema default for
    ``fwhm_min_total_extent_multiplier``) since this candidate set is
    generated before that setting's own sweep runs."""
    diameters = np.asarray(baseline_diameters_um, dtype=float)
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    candidates = {float(default)}
    for d in _percentiles(diameters, (50.0, 75.0)):
        candidates.add(round(0.5 * 3.0 * d, 6))
    return sorted(v for v in candidates if v > 0.0) or [float(default)]


def min_total_extent_multiplier_candidates(default: float = 3.0) -> list[float]:
    """A fixed grid bracketing the schema default -- no image measurement maps
    directly onto "how many diameters wide should the sampled profile be"."""
    candidates = {float(default)} | {2.0, 3.0, 4.0}
    return sorted(v for v in candidates if v >= 1.0) or [float(default)]


def max_transverse_widen_passes_candidates(default: int) -> list[int]:
    """A fixed grid bracketing the schema default -- the widen loop's own
    per-sample pass count is not surfaced by
    ``measure_edge_diameters_fwhm_from_raw_tiff``'s summary (it is decided
    fresh, per sample, inside the loop -- see its module docstring), so
    unlike the other extent-and-widening candidates there is no observed
    per-edge count to derive a data-informed grid from. Each extra pass
    costs one more real sample-and-fit call per remaining sample per trial,
    so the grid stays small."""
    candidates = {int(default), max(0, int(default) - 2), int(default) + 2}
    return sorted(v for v in candidates if v >= 0) or [int(default)]


# ---------------------------------------------------------------------------
# Group 3: central-lobe clipping
# ---------------------------------------------------------------------------
def clip_min_drop_fraction_candidates(default: float = 0.35) -> list[float]:
    candidates = {float(default)} | {0.2, 0.35, 0.5}
    return sorted(v for v in candidates if 0.0 <= v <= 1.0) or [float(default)]


def clip_re_rise_fraction_candidates(default: float = 0.08) -> list[float]:
    candidates = {float(default)} | {0.05, 0.08, 0.15}
    return sorted(v for v in candidates if 0.0 <= v <= 1.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 4: same-edge geometry
# ---------------------------------------------------------------------------
def same_edge_arc_window_um_candidates(default: float | None) -> list[float | None]:
    """``None`` (the adaptive ``max(min, multiplier x diameter estimate)``
    window -- see ``_local_same_edge_window``'s own docstring) plus the
    current/default absolute override.

    The schema default (3.0) is a concrete number, not ``None`` -- meaning a
    typical run never actually exercises the adaptive path at all, silently
    leaving ``fwhm_same_edge_arc_window_multiplier`` and
    ``fwhm_same_edge_arc_window_min_um`` inert (``_local_same_edge_window``
    returns the fixed override immediately whenever it is not ``None``, the
    same "silently inert" trap this session's own FWHM bug hunt found and
    fixed for ``fwhm_diameter_guess_edge_attribute``). Offering ``None`` as a
    real candidate here is what lets those two settings' own sweeps
    (:mod:`.fwhm_search`'s own ``_group_same_edge_geometry``, which only
    runs them once this setting's own sweep picks ``None``) actually change
    anything.
    """
    candidates: set[float | None] = {None}
    if default is not None:
        candidates.add(float(default))
    return _sorted_with_none_first(candidates)


def same_edge_arc_separation_candidates_um(
    inter_sample_arc_distances_um: np.ndarray, default: float
) -> list[float]:
    """Candidates for ``fwhm_nonlocal_same_edge_arc_separation_um`` -- the
    current/default value plus percentiles of the sampled subgraph's own
    measured inter-sample arc-length spacing (25th/50th/75th percentile),
    the actual scale a "how far along the same edge counts as nonlocal"
    setting should be judged against."""
    distances = np.asarray(inter_sample_arc_distances_um, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    candidates = {float(default)} | set(_percentiles(distances, (25.0, 50.0, 75.0)))
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def same_edge_arc_window_min_candidates(
    inter_sample_arc_distances_um: np.ndarray, default: float
) -> list[float]:
    distances = np.asarray(inter_sample_arc_distances_um, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    candidates = {float(default)} | set(_percentiles(distances, (10.0, 25.0)))
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def same_edge_arc_window_multiplier_candidates(default: float = 1.0) -> list[float]:
    """A fixed grid bracketing the schema default -- a dimensionless factor
    with no image measurement that maps onto it directly."""
    candidates = {float(default)} | {0.5, 1.0, 1.5}
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def nonlocal_same_edge_half_extent_factor_candidates(default: float = 0.45) -> list[float]:
    candidates = {float(default)} | {0.3, 0.45, 0.6}
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 5: baseline estimation
# ---------------------------------------------------------------------------
def baseline_wing_fraction_candidates(default: float = 0.2) -> list[float]:
    candidates = {float(default)} | {0.1, 0.2, 0.3}
    return sorted(v for v in candidates if 0.0 < v < 0.5) or [float(default)]


def baseline_constraint_half_width_ptp_candidates(default: float = 0.35) -> list[float]:
    candidates = {float(default)} | {0.2, 0.35, 0.5}
    return sorted(v for v in candidates if 0.0 <= v <= 1.0) or [float(default)]


# ---------------------------------------------------------------------------
# Group 6: diameter guess
# ---------------------------------------------------------------------------
def diameter_guess_edge_attribute_candidates(
    has_populated_edt_diameter: bool,
) -> list[str | None]:
    """``[None, "diameter_um"]`` always, plus ``"edt_diameter_um"`` only when
    at least one sampled edge actually carries a positive value for it --
    the exact "silently inert" trap fixed manually on this same FWHM code
    this session (``use_edt_diameter_crosscheck=False`` leaves
    ``edt_diameter_um`` entirely unpopulated, so offering it as a candidate
    would spend a real trial on a setting that changes nothing)."""
    candidates: list[str | None] = [None, "diameter_um"]
    if has_populated_edt_diameter:
        candidates.append("edt_diameter_um")
    return candidates


def diameter_guess_um_candidates(baseline_diameters_um: np.ndarray) -> list[float | None]:
    """``None`` (the schema default -- no single global guess) plus the
    sampled subgraph's own median and 25th-percentile baseline diameter, a
    data-informed fallback for edges no per-edge attribute can seed."""
    diameters = np.asarray(baseline_diameters_um, dtype=float)
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    candidates: set[float] = set(_percentiles(diameters, (25.0, 50.0)))
    result: list[float | None] = [None] + sorted(round(v, 6) for v in candidates)
    return result


# ---------------------------------------------------------------------------
# Group 7: rejection gates
# ---------------------------------------------------------------------------
def max_fit_center_offset_um_candidates(default: float = 1.5) -> list[float]:
    candidates = {float(default)} | {1.0, 1.5, 2.5}
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def max_fit_center_offset_fraction_candidates(default: float = 0.3) -> list[float]:
    candidates = {float(default)} | {0.2, 0.3, 0.4}
    return sorted(v for v in candidates if v >= 0.0) or [float(default)]


def min_fit_r2_candidates(default: float = 0.85) -> list[float]:
    candidates = {float(default)} | {0.7, 0.85, 0.9}
    return sorted(v for v in candidates if 0.0 <= v <= 1.0) or [float(default)]


def max_plateau_shape_ratio_candidates(default: float = 0.85) -> list[float]:
    candidates = {float(default)} | {0.75, 0.85, 0.95}
    return sorted(v for v in candidates if 0.0 <= v <= 1.0) or [float(default)]
