"""Pure quality scoring for FWHM diameter-measurement optimisation trials.

Mirrors :mod:`.metrics`'s own shape: a frozen dataclass built from one real
``measure_edge_diameters_fwhm_from_raw_tiff`` call's own summary dict and the
graph it wrote its per-edge results onto, plus a regression-penalty guard
ported from :mod:`.search` (that one is private to the skeleton/graph search,
kept here rather than imported so this module has no dependency on it).

No napari, no Qt, no file I/O -- everything here takes a graph and a summary
dict and returns a plain, hashable-friendly dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx
import numpy as np

#: A candidate rejected by a guard scores this much worse than any real
#: trial -- same magnitude as `search._GUARD_PENALTY`, kept as an independent
#: constant so this module has no dependency on that one.
_GUARD_PENALTY = 1000.0


@dataclass(frozen=True)
class FwhmMeasurementQuality:
    """A snapshot of how good one FWHM measurement trial's results are.

    Built from the sampled subgraph a trial actually ran
    ``measure_edge_diameters_fwhm_from_raw_tiff`` against, after that call
    returns -- every field reads what that call itself already wrote onto
    the graph's edges (``fwhm_status``, ``fwhm_diameter_r2_samples``,
    ``fwhm_diameter_samples_um``, and -- only when the trial ran with
    ``store_profile_debug=True`` -- ``fwhm_profile_lines_phys``), never a
    second, independent measurement.
    """

    n_edges_total: int
    n_edges_measured: int
    #: Higher is better: the direct fix for "most edges have no measurement
    #: at all" (the exclusion-zone bug found and fixed manually this session).
    measured_fraction: float
    #: Higher is better: mean of every accepted sample's own Gaussian fit R^2
    #: across every measured edge.
    mean_fit_r2: float
    median_fit_r2: float
    #: Higher is better, up to ~1.0: how close the achieved transverse
    #: profile length came to `min_total_extent_multiplier` x the fitted
    #: diameter -- the direct fix for "the line width is still not 3x".
    mean_achieved_extent_ratio: float
    #: Lower is better: median, across edges with >=2 accepted samples, of
    #: that edge's own coefficient of variation across its accepted
    #: per-sample diameters -- a same-edge-consistency signal.
    median_diameter_cv: float

    @property
    def score(self) -> float:
        """Lower is better -- the general-purpose combination every sweep
        group other than ``rejection_gates`` scores its candidates with
        (see :func:`rejection_gates_score` for that group's own combined
        score). Weights favour measuring more edges first (the biggest,
        most concrete failure mode this optimiser exists to fix), then fit
        quality and achieved extent, with same-edge consistency as a light
        tie-breaker.
        """
        return (
            -self.measured_fraction
            - 0.5 * self.mean_fit_r2
            - 0.25 * min(self.mean_achieved_extent_ratio, 1.0)
            + 0.1 * self.median_diameter_cv
        )


def fwhm_measurement_quality(
    G: nx.MultiGraph,
    summary: Mapping[str, Any],
    *,
    min_total_extent_multiplier: float,
) -> FwhmMeasurementQuality:
    """Build a :class:`FwhmMeasurementQuality` from one trial's own graph and
    the summary dict ``measure_edge_diameters_fwhm_from_raw_tiff`` returned
    for it."""
    n_edges_total = G.number_of_edges()
    n_edges_measured = int(summary.get("edges_measured", 0))
    measured_fraction = (
        float(n_edges_measured) / float(n_edges_total) if n_edges_total > 0 else 0.0
    )

    all_r2: list[float] = []
    extent_ratios: list[float] = []
    diameter_cvs: list[float] = []
    mult = max(float(min_total_extent_multiplier), 1e-9)

    for _u, _v, data in G.edges(data=True):
        if data.get("fwhm_status") != "measured":
            continue
        r2_samples = data.get("fwhm_diameter_r2_samples") or []
        all_r2.extend(float(r2) for r2 in r2_samples if r2 is not None)

        diameters = data.get("fwhm_diameter_samples_um") or []
        if len(diameters) >= 2:
            arr = np.asarray(diameters, dtype=float)
            mean_d = float(np.mean(arr))
            if mean_d > 0:
                diameter_cvs.append(float(np.std(arr) / mean_d))

        profile_lines = data.get("fwhm_profile_lines_phys") or []
        for line, diameter in zip(profile_lines, diameters):
            line_arr = np.asarray(line, dtype=float)
            if line_arr.shape[0] < 2 or diameter <= 0:
                continue
            achieved_extent = float(np.linalg.norm(line_arr[-1] - line_arr[0]))
            extent_ratios.append(achieved_extent / (mult * float(diameter)))

    return FwhmMeasurementQuality(
        n_edges_total=n_edges_total,
        n_edges_measured=n_edges_measured,
        measured_fraction=measured_fraction,
        mean_fit_r2=float(np.mean(all_r2)) if all_r2 else 0.0,
        median_fit_r2=float(np.median(all_r2)) if all_r2 else 0.0,
        mean_achieved_extent_ratio=float(np.mean(extent_ratios)) if extent_ratios else 0.0,
        median_diameter_cv=float(np.median(diameter_cvs)) if diameter_cvs else 0.0,
    )


def rejection_gates_score(quality: FwhmMeasurementQuality) -> float:
    """The ``rejection_gates`` group's own combined score: a looser gate
    that measures more edges but craters fit quality must not look like a
    win, and vice versa -- multiplying the two, not adding them, means
    either one collapsing to ~0 drags the whole score down."""
    return -(quality.measured_fraction * quality.mean_fit_r2)


def regression_penalty(current: float, baseline: float, tolerance: float) -> float:
    """`_GUARD_PENALTY` when *current* has fallen more than *tolerance*
    below *baseline*, else 0 -- ported from
    `haemolynx.optimisation.search._Search._regression_penalty` (private to
    that class) so a candidate that regresses `measured_fraction` or
    `mean_fit_r2` below its own group's pre-sweep baseline never wins just
    because some other term in the combined score improved."""
    return _GUARD_PENALTY if (baseline - current) > tolerance else 0.0
