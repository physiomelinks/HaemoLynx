"""Sequential, image-informed search over the FWHM diameter-measurement
settings (the Diameters tab's "5. FWHM" block).

:func:`optimise_fwhm_settings` is :mod:`.search`'s own coordinate-descent
pattern -- one setting (or small joint group sharing a prerequisite toggle)
at a time, holding everything else fixed, every candidate scored by actually
running the run's own FWHM measurement
(:func:`haemolynx.haemodynamics.apply.measure_fwhm_diameters`, with the run's
haemodynamics config and segmentation) -- applied to the eight FWHM setting
groups (:data:`FWHM_GROUP_NAMES`) instead of the Skeletonise/Graph ones.

Running the real measurement on every one of a real graph's thousands of
edges for every one of ~30 real candidate trials would be far too slow, so
each trial measures one fixed sample of the vessels instead
(:func:`_representative_edges`), picked once up front from the vessels a run
measures and calibrates on -- no zero-resistance bridge, none that network
handling removes. It measures them among the whole network, as a run does:
a line stops at the vessel beside and a sample keeps clear of a junction
whether or not that neighbour was sampled, so a trial's widths are the
run's. The vessels start as the run's FWHM finds them, with none of the
widths the run left on them (:func:`_fresh_template`). The caller's own
graph is left as it was unless it asks for the winners to be measured on it
(``measure_full_graph``): the panel only wants the settings, and measuring
every edge of a large network can take longer than the search itself.

Everything a trial reads is fixed for the run but its settings, so a trial
asked for twice -- every sweep's baseline is its current value's own
candidate -- is measured once (:class:`~.trial_cache.TrialCache`). A sweep
keeps the current value unless a candidate beats it by
:data:`.search.SWEEP_MIN_IMPROVEMENT`. A candidate that raises is scored as a
loss and the sweep continues; if every candidate in a sweep fails, the
setting simply keeps its incoming value -- same fallback as :mod:`.search`.

Given the segmented vessel mask, every trial is also checked the way a run
checks its own FWHM widths: a fixed set of decoys (sampled vessels moved into
vessel-free tissue, :mod:`haemolynx.haemodynamics.fwhm_decoys`) is measured
with the same settings, widths in the speck range they set or disagreeing
with the mask's own width are set aside, and the score counts only the widths
left plus the share of decoys given a width -- so a setting loose enough to
fit specks does not win by measuring more. The decoys and the demotion score
every trial whatever ``fwhm_decoy_check`` and ``fwhm_demote_flagged_edges``
say for the run, and the report says so. Every trial also holds FWHM's blur
at the image PSF the run would use (``fwhm_fix_blur_to_image_psf``).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import networkx as nx
import numpy as np

from haemolynx.haemodynamics import apply, fwhm_diameter, fwhm_decoys, fwhm_planted
from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig, image_psf_from_settings
from haemolynx.haemodynamics.poiseuille import (
    clear_edge_measurements,
    flag_fwhm_edt_disagreement,
    mark_fwhm_demotions,
)
from haemolynx.io.load import _to_binary_volume_for_skeletonization

from . import fwhm_candidates as cand
from . import fwhm_metrics as met
from .progress import ProgressCallback
from .scorecard import ScoreRow
from .search import OptimisationResult, _SweepBookkeeping
from .trial_cache import TrialCache

#: The eight independently selectable groups this search runs, in the order
#: :meth:`_FwhmSearch.run` runs them: the measurement model first, since it
#: changes what every other setting acts on; exclusion and extent decide
#: *which* samples exist at all before clipping/geometry/baseline decide how
#: they are fitted, and rejection gates run last since they act on the
#: already-tuned fits.
FWHM_GROUP_NAMES: tuple[str, ...] = (
    "measurement_model",
    "exclusion_zones",
    "extent_and_widening",
    "central_lobe_clipping",
    "same_edge_geometry",
    "baseline_estimation",
    "diameter_guess",
    "rejection_gates",
)

#: A short, human-readable label for each group, for a GUI checkbox list --
#: mirrors :data:`.search.GROUP_LABELS`.
FWHM_GROUP_LABELS: dict[str, str] = {
    "measurement_model": "Profile model and sampling (judged on planted vessels)",
    "exclusion_zones": "Exclusion zones (branch endpoints, junctions)",
    "extent_and_widening": "Transverse extent and widening",
    "central_lobe_clipping": "Central-lobe clipping",
    "same_edge_geometry": "Same-edge locality geometry",
    "baseline_estimation": "Profile baseline estimation",
    "diameter_guess": "Per-edge diameter guess",
    "rejection_gates": "Rejection-gate thresholds",
}

#: Every FWHM setting this search can decide, across all eight groups --
#: a setting missing from :attr:`OptimisationResult.settings` simply kept
#: its starting value (its own group was disabled, or a conditional
#: sub-sweep never ran because its prerequisite toggle stayed off).
FWHM_SETTING_NAMES: tuple[str, ...] = (
    "fwhm_transverse_sampling_mode",
    "fwhm_profile_model",
    "fwhm_longitudinal_average_um",
    "fwhm_sample_spacing_along_edge_um",
    "fwhm_min_accepted_samples",
    "fwhm_edge_diameter_aggregation",
    "fwhm_branch_endpoint_exclusion_um",
    "fwhm_junction_proximity_exclusion_um",
    "fwhm_transverse_half_extent_um",
    "fwhm_min_total_extent_multiplier",
    "fwhm_max_transverse_widen_passes",
    "fwhm_clip_profile_to_single_vessel",
    "fwhm_clip_min_drop_fraction_of_center",
    "fwhm_clip_re_rise_fraction_of_center",
    "fwhm_enforce_same_edge_locality",
    "fwhm_same_edge_arc_window_um",
    "fwhm_same_edge_arc_window_multiplier",
    "fwhm_same_edge_arc_window_min_um",
    "fwhm_cap_half_extent_by_nonlocal_same_edge_distance",
    "fwhm_nonlocal_same_edge_arc_separation_um",
    "fwhm_nonlocal_same_edge_half_extent_factor",
    "fwhm_profile_baseline_mode",
    "fwhm_profile_baseline_wing_fraction",
    "fwhm_constrain_fitted_baseline",
    "fwhm_baseline_constraint_half_width_ptp",
    "fwhm_diameter_guess_edge_attribute",
    "fwhm_diameter_guess_um",
    "fwhm_reject_samples_with_center_offset",
    "fwhm_max_fit_center_offset_um",
    "fwhm_max_fit_center_offset_fraction_of_diameter",
    "fwhm_reject_samples_with_low_fit_r2",
    "fwhm_min_fit_r2",
    "fwhm_reject_samples_with_plateau_shape",
    "fwhm_max_plateau_shape_ratio",
)

#: An upper bound on how many sweeps a run does, for progress display --
#: mirrors `.search._GROUP_TOTAL_UPPER_BOUND`. Guarded sub-sweeps that never
#: run (a toggle stayed off) mean a real run can finish before reaching this.
_SWEEP_TOTAL_UPPER_BOUND = 34
#: The trials a pass of every group measures, for Auto's time estimate:
#: counted on E14.5 (77 at 4724f6ce, 100 vessels sampled; repeated settings
#: come from the trial cache and are not counted). The estimate used to take
#: 34 sweeps of 3.5 candidates, 119 trials (PLAN_optimisers.md, B8).
_TRIALS_PER_PASS = 77
#: The planted vessels and decoys a trial also measures, per sampled vessel,
#: with the segmentation: 52 for 100 sampled and 96 for 200 on E14.5. Fewer
#: than are asked for are placed (room is needed round each), and their
#: small graphs pay little of a trial's fixed cost.
_CHECK_VESSELS_PER_SAMPLED_VESSEL = 0.5

#: "Auto" sample-size selection's own time budget, in seconds -- smaller
#: than the skeleton/graph search's five minutes since this search's own
#: per-trial cost scales with edge count, not voxel volume, and a
#: representative sample is deliberately small to begin with.
DEFAULT_AUTO_SAMPLE_TARGET_SECONDS = 180.0
#: Never sample fewer edges than this -- a handful of edges cannot
#: represent a real network's own length distribution.
_MIN_SAMPLE_EDGE_COUNT = 10
#: What "Auto" never samples fewer than (a smaller pool is sampled whole):
#: planting puts a vessel of known width beside about one sampled vessel in
#: five (43 of 200 on E14.5), and the accuracy group needs some planted. A
#: budget too small for this shows in the estimate; it is not met by sampling
#: so few vessels that accuracy is never checked (PLAN_optimisers.md, B8).
_MIN_AUTO_SAMPLE_EDGE_COUNT = 100

#: A candidate rejected by a guard scores this much worse than any real
#: trial -- see `.fwhm_metrics._GUARD_PENALTY`.
_GUARD_PENALTY = met._GUARD_PENALTY
#: Guard every sweep's own combined score against the one failure mode none
#: of the individual metrics can see on their own: a candidate that
#: improves its own score while quietly leaving far fewer edges a usable
#: width, or fitting them far worse, than the group's own pre-sweep baseline
#: did -- see `.fwhm_metrics.regression_penalty`. Usable, not merely
#: measured: a gate that stops FWHM fitting specks measures fewer edges and
#: loses nothing the run would have used. Reasoned, not empirically
#: fitted: small enough that ordinary sample-to-sample noise between two
#: similar candidates never falsely trips the guard.
_MAX_USABLE_FRACTION_REGRESSION = 0.05
_MAX_FIT_R2_REGRESSION = 0.05
#: ...but never less than this many edges' worth of the sample: on a sample
#: of ten, 5% is half an edge, and a candidate measuring one edge fewer
#: tripped the guard while one measuring an edge more was rewarded -- a pull
#: towards looser settings. One edge lost is allowed, two are not.
_USABLE_EDGES_REGRESSION_ALLOWED = 1.5

#: The before/after scorecard (see :mod:`.scorecard`): each measure's name,
#: the :class:`.fwhm_metrics.FwhmMeasurementQuality` field it reads, whether
#: higher is better, and how far it may move the wrong way and still count
#: as the same (``None``: the sample-size tolerance the guard uses).
_SCORECARD_MEASURES: tuple[tuple[str, str, bool, Optional[float]], ...] = (
    ("sampled vessels with a usable width", "usable_fraction", True, None),
    ("mean fit R2", "mean_fit_r2", True, 0.02),
    ("profile length reached (of the target)", "mean_achieved_extent_ratio", True, 0.05),
    ("width spread along a vessel (CV)", "median_diameter_cv", False, 0.02),
    ("decoys given a width", "decoy_false_positive_rate", False, 0.0),
    ("planted vessels' width error", "planted_error", False, 0.02),
)


def _scorecard(
    before: met.FwhmMeasurementQuality, after: met.FwhmMeasurementQuality
) -> tuple[ScoreRow, ...]:
    usable_tolerance = _FwhmSearch._usable_fraction_tolerance(before.n_edges_total)
    return tuple(
        ScoreRow(
            name,
            float(getattr(before, field)),
            float(getattr(after, field)),
            higher_is_better,
            usable_tolerance if tolerance is None else tolerance,
        )
        for name, field, higher_is_better, tolerance in _SCORECARD_MEASURES
        if field != "planted_error" or before.n_planted
    )


#: The FWHM/EDT disagreement ratio a width is set aside at, when the
#: settings do not give ``edt_fwhm_disagreement_warn_ratio`` -- its default.
_DEFAULT_EDT_WARN_RATIO = 1.5


def _measurement_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    """``measure_edge_diameters_fwhm_from_raw_tiff`` keyword arguments from a
    settings dict -- every ``fwhm_``-prefixed name the function actually
    accepts, prefix stripped.

    A local reimplementation of
    ``haemolynx.parsers.config.prefixed_arguments`` rather than an import of
    it: that function lives in ``haemolynx.parsers``, a dependency this
    package's own docstring does not claim (see
    ``haemolynx.optimisation``'s module docstring) and this one function
    does not need the rest of that module for.
    """
    prefix = "fwhm_"
    valid = set(inspect.signature(fwhm_diameter.measure_edge_diameters_fwhm_from_raw_tiff).parameters)
    return {
        name[len(prefix):]: value
        for name, value in settings.items()
        if name.startswith(prefix) and name[len(prefix):] in valid
    }


#: Edge attributes the sample is spread by, first that most edges carry.
_SAMPLE_WIDTH_ATTRIBUTES = ("edt_diameter_um", "diameter_um")


def _sample_buckets(G: nx.MultiGraph, edge_keys, n_buckets: int) -> np.ndarray:
    """Which of up to *n_buckets* (plus one, for edges with no width) each
    edge is drawn from: by width when most edges have one
    (:data:`_SAMPLE_WIDTH_ATTRIBUTES`), in bins even in log width between
    the narrowest and the widest -- widths cluster, and quantile bins of 38
    capillaries at 3 um and two vessels at 12 um all fell on 3 um -- else by
    length, in quantile bins."""
    for name in _SAMPLE_WIDTH_ATTRIBUTES:
        widths = np.asarray(
            [float(G.edges[e].get(name) or 0.0) for e in edge_keys], dtype=float
        )
        widths[~np.isfinite(widths)] = 0.0
        has_width = widths > 0
        if np.count_nonzero(has_width) * 2 < len(edge_keys):
            continue
        logs = np.log(np.where(has_width, widths, 1.0))
        lo, hi = float(logs[has_width].min()), float(logs[has_width].max())
        span = hi - lo
        bins = np.zeros(len(edge_keys), dtype=int) if span <= 0 else np.clip(
            np.floor((logs - lo) / span * n_buckets).astype(int), 0, n_buckets - 1
        )
        return np.where(has_width, bins, n_buckets)
    lengths = np.asarray(
        [float(G.edges[e].get("length", 0.0) or 0.0) for e in edge_keys], dtype=float
    )
    quantile_edges = np.quantile(lengths, np.linspace(0.0, 1.0, n_buckets + 1))
    return np.clip(np.searchsorted(quantile_edges[1:-1], lengths, side="right"), 0, n_buckets - 1)


def _representative_edges(
    G: nx.MultiGraph, sample_edge_count: int, pool: Optional[Iterable[tuple]] = None
) -> list[tuple]:
    """A fixed-size sample ``(u, v, key)`` of *G*'s own edges -- of *pool*
    when given -- spread over their widths.

    Bucketed by each edge's width -- its mask (EDT) width, or its diameter,
    when most edges have one; its ``length`` otherwise (see
    :func:`_sample_buckets`) -- then a fixed number drawn from each bucket. A plain random sample of a network
    of mostly capillaries would rarely draw enough of the few wide vessels
    to say anything about how well they measure, and width, not length, is
    what decides how FWHM fares on a vessel. Deterministic (seeded), so a
    repeated call against the same graph and count picks the same edges.
    """
    edge_keys = list(G.edges(keys=True)) if pool is None else [tuple(edge) for edge in pool]
    total_edges = len(edge_keys)
    sample_edge_count = max(1, min(int(sample_edge_count), total_edges))
    if sample_edge_count >= total_edges:
        return edge_keys

    bucket_of = _sample_buckets(G, edge_keys, min(5, max(1, sample_edge_count)))
    buckets = [int(b) for b in np.unique(bucket_of)]

    rng = np.random.default_rng(0)
    per_bucket = max(1, sample_edge_count // len(buckets))
    selected: list[int] = []
    for bucket in buckets:
        idx_in_bucket = np.flatnonzero(bucket_of == bucket)
        rng.shuffle(idx_in_bucket)
        selected.extend(int(i) for i in idx_in_bucket[:per_bucket])

    if len(selected) < sample_edge_count:
        taken = set(selected)
        remaining = np.array([i for i in range(total_edges) if i not in taken], dtype=int)
        rng.shuffle(remaining)
        selected.extend(int(i) for i in remaining[: sample_edge_count - len(selected)])
    selected = selected[:sample_edge_count]

    return [edge_keys[i] for i in selected]


#: What a run has measured on its vessels when its FWHM starts: the
#: segmentation's own widths, with ``use_edt_diameter_crosscheck`` on --
#: FWHM's first guess at each vessel's width.
_MEASURED_BEFORE_FWHM = ("edt_",)

#: The branch-order diameter table a config made here carries: measuring
#: widths reads none of it, but a config must have one.
_UNREAD_DIAMETER_TABLE = {"diameter_by_branch_order": {"unread": 1.0}}


def _measurement_config(
    starting_values: Mapping[str, Any],
    *,
    voxel_size_zyx: tuple[float, float, float],
    axis_order: str = fwhm_diameter.CANONICAL_AXIS_ORDER,
    raw_tiff_path: str | Path | None = None,
    raw_channel: int | None = None,
) -> HaemodynamicsApplyConfig:
    """The haemodynamics config a run measures widths with, made from
    *starting_values*, for a caller without the run's own: the panel passes
    that one, as the Diameters stage builds it."""
    edt = {
        name: value
        for name, value in starting_values.items()
        if name.startswith("edt_") or name == "use_edt_diameter_crosscheck"
    }
    fwhm = {name: value for name, value in starting_values.items() if name not in edt}
    fwhm.update(use_fwhm_edge_diameters=True, fwhm_raw_tiff_path=raw_tiff_path, fwhm_raw_channel=raw_channel)
    return HaemodynamicsApplyConfig(
        diameters=dict(_UNREAD_DIAMETER_TABLE),
        fwhm=fwhm,
        edt=edt,
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        axis_order=axis_order,
    )


def _with_fwhm_settings(
    config: HaemodynamicsApplyConfig, settings: Mapping[str, Any]
) -> HaemodynamicsApplyConfig:
    """*config* holding *settings*' values of the settings this search
    decides (:data:`FWHM_SETTING_NAMES`); every other setting is the run's."""
    return replace(
        config,
        fwhm={**config.fwhm, **{name: settings[name] for name in FWHM_SETTING_NAMES if name in settings}},
    )


def _has_mask_widths(G: nx.MultiGraph, edges: Iterable[tuple]) -> bool:
    return any(float(G.edges[edge].get("edt_diameter_um") or 0.0) > 0.0 for edge in edges)


def _fresh_template(
    G: nx.MultiGraph, config: HaemodynamicsApplyConfig, vessel_mask: Optional[np.ndarray]
) -> nx.MultiGraph:
    """The vessels of *G* a run measures (no zero-resistance bridge), as the
    run's FWHM finds them, in a copy: every measurement *G* carries cleared
    by the pipeline's own rule
    (:func:`~haemolynx.haemodynamics.poiseuille.clear_edge_measurements`) --
    the panel hands over the run's graph, and a trial must not be credited
    with the run's widths -- with no width kept as provisional, since an
    image's vessels have none before the Diameters stage. The segmentation's
    own widths stay (:data:`_MEASURED_BEFORE_FWHM`): where the run measures
    them (``use_edt_diameter_crosscheck``) they are its own, measured again
    here, as the run measures them, when *G* has none."""
    template = apply.measured_vessels(G).copy()
    for _u, _v, data in template.edges(data=True):
        clear_edge_measurements(data, keep_prefixes=_MEASURED_BEFORE_FWHM)
    if (
        vessel_mask is not None
        and config.use_edt_diameter_crosscheck
        and not _has_mask_widths(template, template.edges(keys=True))
    ):
        apply.measure_edt_diameters(template, config, mask_volume=vessel_mask)
    return template


def _sample_edge_count_from_probe_seconds(
    probe_seconds: float,
    probe_edge_count: int,
    total_edges: int,
    target_seconds: float,
    fixed_seconds: float = 0.0,
    check_vessels_per_sampled_vessel: float = 0.0,
) -> int:
    """Pure arithmetic half of :func:`estimate_sample_edge_count_for_time_budget`,
    split out so the decision itself is directly testable without timing
    anything real -- mirrors `.search._factor_from_probe_seconds`.
    *probe_seconds* is one trial measuring *probe_edge_count* edges,
    *fixed_seconds* of it what a trial costs however few it measures;
    *check_vessels_per_sampled_vessel* the planted vessels and decoys it
    measures besides."""
    if probe_seconds <= 0.0 or probe_edge_count <= 0:
        return total_edges
    per_vessel = _per_sampled_vessel_seconds(
        probe_seconds, probe_edge_count, fixed_seconds, check_vessels_per_sampled_vessel
    )
    budget_edges = int(
        (target_seconds - _TRIALS_PER_PASS * fixed_seconds) / max(per_vessel * _TRIALS_PER_PASS, 1e-12)
    )
    return min(max(_MIN_AUTO_SAMPLE_EDGE_COUNT, budget_edges), total_edges)


def _per_sampled_vessel_seconds(
    probe_seconds: float, probe_edge_count: int, fixed_seconds: float, check_vessels_per_sampled_vessel: float
) -> float:
    per_edge_seconds = max(probe_seconds - fixed_seconds, 0.0) / float(max(probe_edge_count, 1))
    return per_edge_seconds * (1.0 + check_vessels_per_sampled_vessel)


def _estimated_search_seconds(
    probe_seconds: float,
    probe_edge_count: int,
    sample_edge_count: int,
    fixed_seconds: float = 0.0,
    check_vessels_per_sampled_vessel: float = 0.0,
) -> float:
    """How long a pass on *sample_edge_count* edges is estimated to take --
    the arithmetic :func:`_sample_edge_count_from_probe_seconds` compares
    with its budget."""
    per_vessel = _per_sampled_vessel_seconds(
        probe_seconds, probe_edge_count, fixed_seconds, check_vessels_per_sampled_vessel
    )
    return _TRIALS_PER_PASS * (fixed_seconds + per_vessel * sample_edge_count)


def estimate_sample_edge_count_for_time_budget(
    G: nx.MultiGraph,
    *,
    raw_volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    psf_sigma_zyx: tuple[float, float, float] | None = None,
    config: Optional[HaemodynamicsApplyConfig] = None,
) -> int:
    """The largest sample-edge count estimated to keep a full search under
    *target_seconds* on the machine it actually runs on.

    Times real trials -- the run's measurement of a small probe of *this*
    graph's own vessels among all of them, and of none -- then extrapolates
    linearly in edge count (this search's own cost scales per-edge, unlike
    the skeleton/graph search's cubic voxel-volume scaling) across the
    :data:`_TRIALS_PER_PASS` trials a pass makes, each paying what a trial
    costs whatever it measures (the whole network painted into its label
    volume) once. Never fewer than :data:`_MIN_AUTO_SAMPLE_EDGE_COUNT`
    edges, whatever the budget. Falls back to every edge (no
    subsampling) if the probe itself cannot run at all, so "Auto" never
    fails a run just because estimating its own runtime did. *config* is
    the run's haemodynamics config, made from *starting_values* when not
    given.
    """
    if config is None:
        config = _measurement_config(starting_values, voxel_size_zyx=voxel_size_zyx)
    config = _with_fwhm_settings(config, starting_values)
    template = _fresh_template(G, config, None)
    return _time_budget_sample(
        lambda edges: apply.measure_fwhm_diameters(
            template.copy(), config, raw_volume=raw_volume, psf_sigma_zyx=psf_sigma_zyx, edges=edges
        ),
        template,
        list(template.edges(keys=True)),
        target_seconds=target_seconds,
    )[0]


def _time_budget_sample(
    measure: Callable[[list[tuple]], Any],
    G: nx.MultiGraph,
    pool: list[tuple],
    *,
    target_seconds: float,
    check_vessels_per_sampled_vessel: float = 0.0,
) -> tuple[int, Optional[float]]:
    """:func:`estimate_sample_edge_count_for_time_budget`'s count of *pool*'s
    edges, timing ``measure(edges)`` -- one trial measuring *edges* -- and
    the search time it estimated for that count -- ``None`` when it did not
    time a probe -- which the run's report sets beside the time the search
    took. A trial measures *check_vessels_per_sampled_vessel* planted vessels
    and decoys besides each sampled one."""
    total_edges = len(pool)
    if total_edges <= _MIN_SAMPLE_EDGE_COUNT:
        return total_edges, None
    probe_edge_count = min(10, total_edges)
    probe_edges = _representative_edges(G, probe_edge_count, pool=pool)

    try:
        # One untimed warm-up call first -- see `.search`'s own
        # `estimate_downsample_factor_for_time_budget` for why a fresh
        # process's first real call pays a one-time setup cost that would
        # otherwise inflate the estimate by orders of magnitude.
        measure(probe_edges)
        t0 = time.perf_counter()
        measure([])
        fixed_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        measure(probe_edges)
        probe_seconds = time.perf_counter() - t0
    except Exception:  # noqa: BLE001 - estimating runtime must never block a real run
        return total_edges, None

    count = _sample_edge_count_from_probe_seconds(
        probe_seconds, probe_edge_count, total_edges, target_seconds, fixed_seconds,
        check_vessels_per_sampled_vessel,
    )
    if probe_seconds <= 0.0:
        return count, None
    return count, _estimated_search_seconds(
        probe_seconds, probe_edge_count, count, fixed_seconds, check_vessels_per_sampled_vessel
    )


class _FwhmSearch(_SweepBookkeeping):
    """Mutable state for one call to :func:`optimise_fwhm_settings`."""

    def __init__(
        self,
        template: nx.MultiGraph,
        *,
        raw_volume: np.ndarray,
        voxel_size_zyx: tuple[float, float, float],
        starting_values: Mapping[str, Any],
        progress: Optional[ProgressCallback],
        enabled_groups: Optional[Iterable[str]] = None,
        vessel_mask: Optional[np.ndarray] = None,
        psf_sigma_zyx: tuple[float, float, float] | None = None,
        planting_psf_um: tuple[float, float, float] | None = None,
        sample_edges: Optional[Iterable[tuple]] = None,
        config: Optional[HaemodynamicsApplyConfig] = None,
        pool_size: Optional[int] = None,
    ) -> None:
        #: The vessels a run measures, as its FWHM finds them (see
        #: `_fresh_template`). Never mutated by a trial -- every trial copies
        #: this fresh (see `_run_trial`), matching `.search._Search`'s own
        #: "always re-derive from source" discipline, since a measurement
        #: call writes its results directly onto the graph's own edges.
        self.template = template
        #: The vessels every trial measures, among all of the template's,
        #: and is scored on; every edge of the template unless given.
        self.sample_edges: list[tuple] = (
            list(template.edges(keys=True)) if sample_edges is None
            else [tuple(edge) for edge in sample_edges]
        )
        #: How many vessels the sample was drawn from.
        self.pool_size = len(self.sample_edges) if pool_size is None else int(pool_size)
        self.raw_volume = raw_volume
        self.voxel_size_zyx = voxel_size_zyx
        self.current: dict[str, Any] = dict(starting_values)
        #: The run's haemodynamics config: every trial measures with it,
        #: holding the trial's own values of :data:`FWHM_SETTING_NAMES`.
        self.config = config if config is not None else _measurement_config(
            starting_values, voxel_size_zyx=voxel_size_zyx
        )
        #: The segmentation, binarised: what a run's lines stop at the vessel
        #: beside by (``fwhm_stop_at_other_vessels_in_mask``).
        self.vessel_mask = vessel_mask
        #: The image PSF every trial holds FWHM's blur at, while
        #: ``fwhm_fix_blur_to_image_psf`` is on.
        self.psf_sigma_zyx = psf_sigma_zyx
        self.edt_warn_ratio = float(
            self.config.edt_setting("edt_fwhm_disagreement_warn_ratio") or _DEFAULT_EDT_WARN_RATIO
        )
        #: One decoy per sampled edge (up to the decoy check's own sample
        #: size), placed once so every trial is judged on the same tissue.
        self.decoy_probe: nx.MultiGraph | None = None
        #: Vessels of known width drawn into a copy of the raw image beside
        #: sampled ones (see `haemodynamics.fwhm_planted`), with the blur the
        #: image shows -- *planting_psf_um*, whatever the trials hold theirs at.
        self.planting_psf_um = planting_psf_um
        self.planted: Optional[fwhm_planted.PlantedVessels] = None
        if vessel_mask is not None:
            self._prepare_checks(vessel_mask)
        super().__init__(
            progress=progress,
            enabled_groups=enabled_groups,
            group_total=len(FWHM_GROUP_NAMES) * 4,
        )
        self.passes_run: int = 0
        #: Every trial's outcome, by the settings it measured with: nothing
        #: else a trial reads changes during a run (see `_trial_outcome`).
        self._outcomes = TrialCache(max_entries=None, max_bytes=None)

    # -- real-measurement trial helpers ------------------------------------------
    def _sample_data(self) -> Iterable[dict[str, Any]]:
        """The template's attributes of each sampled edge."""
        return (self.template.edges[edge] for edge in self.sample_edges)

    def _prepare_checks(self, vessel_mask: np.ndarray) -> None:
        """Place the sampled vessels' decoys and plant their known-width
        vessels -- once, before any trial -- each guided by its vessel's mask
        width: the run's own, or, where the run measures none, one measured
        here as a run would, on a copy no trial reads, since a run's FWHM
        would have none to start from."""
        guide = self.template.edge_subgraph(self.sample_edges)
        if not _has_mask_widths(guide, self.sample_edges):
            guide = guide.copy()
            apply.measure_edt_diameters(guide, self.config, mask_volume=vessel_mask)
        edges = list(self.sample_edges)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(edges))
        self.decoy_probe = fwhm_decoys.decoy_probe_graph(
            guide,
            [edges[i] for i in order[: apply.fwhm_decoy_sample_size(self.config)]],
            vessel_mask,
            self.voxel_size_zyx,
            rng=rng,
            guide_attribute="edt_diameter_um",
            copy_attributes=("diameter_um",),
        )
        self.planted = fwhm_planted.plant_vessels(
            self.raw_volume,
            guide,
            [edges[i] for i in order],
            vessel_mask,
            self.voxel_size_zyx,
            image_psf_um=self.planting_psf_um,
            rng=np.random.default_rng(1),
        )

    def _measure(
        self,
        graph: nx.MultiGraph,
        settings: Mapping[str, Any],
        *,
        volume: Optional[np.ndarray] = None,
        edges: Optional[Iterable[tuple]] = None,
    ) -> dict[str, Any]:
        """The run's FWHM measurement of *graph* (of its *edges*, among all
        of them, when given) with *settings* on *volume* (the raw image
        unless given), its blur held at the image PSF while
        ``fwhm_fix_blur_to_image_psf`` is on and the profile model has one."""
        config = _with_fwhm_settings(self.config, settings)
        fixed_blur = bool(config.fwhm_setting("fwhm_fix_blur_to_image_psf", True)) and (
            config.fwhm_setting("fwhm_profile_model", "blurred_lumen") == "blurred_lumen"
        )
        return apply.measure_fwhm_diameters(
            graph,
            config,
            raw_volume=self.raw_volume if volume is None else volume,
            vessel_mask=self.vessel_mask,
            psf_sigma_zyx=self.psf_sigma_zyx if fixed_blur else None,
            edges=edges,
        )

    def _run_trial(self, overrides: Mapping[str, Any]) -> tuple[nx.MultiGraph, dict[str, Any]]:
        """Run the real measurement of the sampled vessels on a fresh copy of
        the template with `self.current` plus *overrides* -- never a previous
        trial's own output."""
        settings = {**self.current, **overrides}
        graph_trial = self.template.copy()
        summary = self._measure(graph_trial, settings, edges=self.sample_edges)
        return graph_trial, summary

    def _checked_trial(
        self, overrides: Mapping[str, Any]
    ) -> tuple[nx.MultiGraph, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        """:meth:`_run_trial`, then the run's own checks on its widths: the
        decoys measured with the same settings, the EDT cross-check, and the
        widths either sets aside marked ``fwhm_demoted``; and the planted
        vessels measured with them too. Returns the sampled vessels of the
        trial and the decoy and planted-vessel reports, each ``None`` without
        its vessels. Only widths the trial measured count: an edge that
        failed is ``fwhm_status`` failed whatever widths it carries."""
        graph_trial, summary = self._run_trial(overrides)
        sample = graph_trial.edge_subgraph(self.sample_edges)
        settings = {**self.current, **overrides}
        planted_report = None
        if self.planted is not None:
            planted = self.planted.probe.copy()
            self._measure(planted, settings, volume=self.planted.volume)
            planted_report = fwhm_planted.planted_width_report(planted)
        decoy_report = None
        if self.decoy_probe is not None:
            probe = self.decoy_probe.copy()
            self._measure(probe, settings)
            measured = [
                (u, v, key)
                for u, v, key, data in sample.edges(keys=True, data=True)
                if data.get("fwhm_status") == "measured"
            ]
            decoy_report = fwhm_decoys.speck_width_report(sample, probe, measured)
        flag_fwhm_edt_disagreement(sample, warn_ratio=self.edt_warn_ratio)
        mark_fwhm_demotions(sample, enabled=True)
        return sample, summary, decoy_report, planted_report

    def _trial_outcome(
        self, overrides: Mapping[str, Any]
    ) -> tuple[met.FwhmMeasurementQuality, np.ndarray]:
        """:meth:`_checked_trial`'s quality and accepted per-sample diameters
        for ``self.current`` plus *overrides*, measured once per distinct set
        of settings. The sample, its decoys, the raw image, the PSF and the
        EDT ratio are fixed for the run, so the settings a measurement reads
        are the whole key."""
        settings = {**self.current, **overrides}
        key = (
            _measurement_kwargs(settings),
            bool(settings.get("fwhm_fix_blur_to_image_psf", True)),
            settings.get("fwhm_min_total_extent_multiplier"),
        )

        def measure() -> tuple[met.FwhmMeasurementQuality, np.ndarray]:
            sample, summary, decoy_report, planted_report = self._checked_trial(overrides)
            quality = met.fwhm_measurement_quality(
                sample,
                summary,
                min_total_extent_multiplier=float(settings["fwhm_min_total_extent_multiplier"]),
                decoy_report=decoy_report,
                planted_report=planted_report,
            )
            diameters: list[float] = []
            for _u, _v, data in sample.edges(data=True):
                if data.get("fwhm_status") == "measured":
                    diameters.extend(float(d) for d in (data.get("fwhm_diameter_samples_um") or []))
            return quality, np.asarray(diameters, dtype=float)

        return self._outcomes.get_or_compute((), key, measure)

    def _quality(self, overrides: Mapping[str, Any]) -> met.FwhmMeasurementQuality:
        return self._trial_outcome(overrides)[0]

    def _baseline_diameters_um(self) -> np.ndarray:
        """This edge sample's own accepted FWHM diameters under
        `self.current` right now (`self.current` keeps changing as earlier
        groups decide their own settings)."""
        return self._trial_outcome({})[1]

    def _inter_sample_arc_distances_um(self) -> np.ndarray:
        """The actual along-edge spacing between consecutive samples this
        sample subgraph's own edges get under the current
        ``fwhm_sample_spacing_along_edge_um``, computed directly (same
        formula `measure_edge_diameters_fwhm_from_raw_tiff` itself uses to
        place samples) rather than by running a real measurement -- this is
        plain arithmetic on each edge's own ``length``, not something a
        Gaussian fit could tell us anything a re-derivation cannot."""
        nominal = float(self.current["fwhm_sample_spacing_along_edge_um"])
        if nominal <= 0:
            return np.array([], dtype=float)
        spacings: list[float] = []
        for data in self._sample_data():
            total_len = float(data.get("length", 0.0) or 0.0)
            if total_len <= 0:
                continue
            n_samples = max(1, int(np.floor(total_len / nominal)) + 1)
            if n_samples > 1:
                spacings.append(total_len / (n_samples - 1))
        return np.asarray(spacings, dtype=float)

    # -- sweeps -------------------------------------------------------------------
    @staticmethod
    def _usable_fraction_tolerance(n_edges: int) -> float:
        """How far `usable_fraction` may fall below a sweep's baseline, on a
        sample of *n_edges*, before the guard trips:
        :data:`_MAX_USABLE_FRACTION_REGRESSION`, or
        :data:`_USABLE_EDGES_REGRESSION_ALLOWED` edges of the sample when that
        is more."""
        return max(
            _MAX_USABLE_FRACTION_REGRESSION, _USABLE_EDGES_REGRESSION_ALLOWED / max(int(n_edges), 1)
        )

    def _guarded_sweep(
        self, group: str, setting: str, candidate_values: list, *, score_fn=None
    ) -> Any:
        """:meth:`_sweep`, guarded against a candidate that improves its own
        combined score while regressing `usable_fraction` or
        `mean_fit_r2` below this sub-sweep's own pre-sweep baseline (see
        `.fwhm_metrics.regression_penalty`) -- the workhorse behind every
        group method below except where a group-specific *score_fn* is
        needed (`rejection_gates`).
        """
        score_of = score_fn or (lambda quality: quality.score)
        baseline = self._quality({})
        usable_tolerance = self._usable_fraction_tolerance(baseline.n_edges_total)

        def cost_fn(value: Any) -> float:
            quality = self._quality({setting: value})
            penalty = met.regression_penalty(
                quality.usable_fraction, baseline.usable_fraction, usable_tolerance
            )
            penalty += met.regression_penalty(
                quality.mean_fit_r2, baseline.mean_fit_r2, _MAX_FIT_R2_REGRESSION
            )
            self._note(
                usable_fraction=quality.usable_fraction,
                mean_fit_r2=quality.mean_fit_r2,
                extent_ratio=quality.mean_achieved_extent_ratio,
                diameter_cv=quality.median_diameter_cv,
                decoy_false_positive_rate=quality.decoy_false_positive_rate,
                guard_penalty=penalty,
                **({"planted_width_error": quality.planted_error} if quality.n_planted else {}),
            )
            return score_of(quality) + penalty

        return self._sweep(group, setting, candidate_values, cost_fn)

    # -- group 0: measurement model -------------------------------------------------
    def _group_measurement_model(self) -> None:
        """Which way profiles are drawn, how they are fitted, how much each is
        averaged along the vessel, how closely they are spaced, how many an
        edge needs and how they are pooled: the settings that decide whether
        a width is right. Coverage and fit quality cannot judge them -- a
        Gaussian fits a plasma-filled lumen well and reads it narrow -- so
        they are swept only with vessels planted at a known width to judge
        them by; without the run's segmentation to plant beside, they keep
        their values."""
        group = "measurement_model"
        if self.planted is None:
            self._record(
                group, "measurement_model", None, 0.0,
                note="skipped: no vessels planted to judge accuracy by (needs the run's segmentation)",
            )
            return
        self._guarded_sweep(
            group, "fwhm_transverse_sampling_mode", ["in_plane_yx", "true_3d_perpendicular"]
        )
        self._guarded_sweep(group, "fwhm_profile_model", ["blurred_lumen", "gaussian"])
        self._guarded_sweep(
            group,
            "fwhm_longitudinal_average_um",
            cand.longitudinal_average_candidates(self.current["fwhm_longitudinal_average_um"]),
        )
        self._guarded_sweep(
            group,
            "fwhm_sample_spacing_along_edge_um",
            cand.sample_spacing_candidates(self.current["fwhm_sample_spacing_along_edge_um"]),
        )
        self._guarded_sweep(
            group,
            "fwhm_min_accepted_samples",
            cand.min_accepted_samples_candidates(self.current["fwhm_min_accepted_samples"]),
        )
        self._guarded_sweep(group, "fwhm_edge_diameter_aggregation", ["median", "mean"])

    # -- group 1: exclusion zones --------------------------------------------------
    def _group_exclusion_zones(self) -> None:
        group = "exclusion_zones"
        edge_lengths = np.asarray(
            [float(data.get("length", 0.0) or 0.0) for data in self._sample_data()], dtype=float
        )
        self._guarded_sweep(
            group,
            "fwhm_branch_endpoint_exclusion_um",
            cand.exclusion_zone_candidates(edge_lengths, self.current["fwhm_branch_endpoint_exclusion_um"]),
        )
        self._guarded_sweep(
            group,
            "fwhm_junction_proximity_exclusion_um",
            cand.exclusion_zone_candidates(edge_lengths, self.current["fwhm_junction_proximity_exclusion_um"]),
        )

    # -- group 2: extent and widening -----------------------------------------------
    def _group_extent_and_widening(self) -> None:
        group = "extent_and_widening"
        baseline_diameters = self._baseline_diameters_um()
        self._guarded_sweep(
            group,
            "fwhm_transverse_half_extent_um",
            cand.half_extent_candidates(baseline_diameters, self.current["fwhm_transverse_half_extent_um"]),
        )
        self._guarded_sweep(
            group,
            "fwhm_min_total_extent_multiplier",
            cand.min_total_extent_multiplier_candidates(self.current["fwhm_min_total_extent_multiplier"]),
        )
        self._guarded_sweep(
            group,
            "fwhm_max_transverse_widen_passes",
            cand.max_transverse_widen_passes_candidates(self.current["fwhm_max_transverse_widen_passes"]),
        )

    # -- group 3: central-lobe clipping ----------------------------------------------
    def _group_central_lobe_clipping(self) -> None:
        group = "central_lobe_clipping"
        self._guarded_sweep(group, "fwhm_clip_profile_to_single_vessel", [False, True])
        if self.current["fwhm_clip_profile_to_single_vessel"]:
            self._guarded_sweep(
                group,
                "fwhm_clip_min_drop_fraction_of_center",
                cand.clip_min_drop_fraction_candidates(self.current["fwhm_clip_min_drop_fraction_of_center"]),
            )
            self._guarded_sweep(
                group,
                "fwhm_clip_re_rise_fraction_of_center",
                cand.clip_re_rise_fraction_candidates(self.current["fwhm_clip_re_rise_fraction_of_center"]),
            )

    # -- group 4: same-edge geometry --------------------------------------------------
    def _group_same_edge_geometry(self) -> None:
        group = "same_edge_geometry"
        self._guarded_sweep(group, "fwhm_enforce_same_edge_locality", [False, True])
        if self.current["fwhm_enforce_same_edge_locality"]:
            self._guarded_sweep(
                group,
                "fwhm_same_edge_arc_window_um",
                cand.same_edge_arc_window_um_candidates(self.current["fwhm_same_edge_arc_window_um"]),
            )
            if self.current["fwhm_same_edge_arc_window_um"] is None:
                arc_distances = self._inter_sample_arc_distances_um()
                self._guarded_sweep(
                    group,
                    "fwhm_same_edge_arc_window_min_um",
                    cand.same_edge_arc_window_min_candidates(
                        arc_distances, self.current["fwhm_same_edge_arc_window_min_um"]
                    ),
                )
                self._guarded_sweep(
                    group,
                    "fwhm_same_edge_arc_window_multiplier",
                    cand.same_edge_arc_window_multiplier_candidates(
                        self.current["fwhm_same_edge_arc_window_multiplier"]
                    ),
                )
        self._guarded_sweep(group, "fwhm_cap_half_extent_by_nonlocal_same_edge_distance", [False, True])
        if self.current["fwhm_cap_half_extent_by_nonlocal_same_edge_distance"]:
            arc_distances = self._inter_sample_arc_distances_um()
            self._guarded_sweep(
                group,
                "fwhm_nonlocal_same_edge_arc_separation_um",
                cand.same_edge_arc_separation_candidates_um(
                    arc_distances, self.current["fwhm_nonlocal_same_edge_arc_separation_um"]
                ),
            )
            self._guarded_sweep(
                group,
                "fwhm_nonlocal_same_edge_half_extent_factor",
                cand.nonlocal_same_edge_half_extent_factor_candidates(
                    self.current["fwhm_nonlocal_same_edge_half_extent_factor"]
                ),
            )

    # -- group 5: baseline estimation -----------------------------------------------
    def _group_baseline_estimation(self) -> None:
        group = "baseline_estimation"
        self._guarded_sweep(group, "fwhm_profile_baseline_mode", ["wings", "percentile"])
        self._guarded_sweep(
            group,
            "fwhm_profile_baseline_wing_fraction",
            cand.baseline_wing_fraction_candidates(self.current["fwhm_profile_baseline_wing_fraction"]),
        )
        self._guarded_sweep(group, "fwhm_constrain_fitted_baseline", [False, True])
        if self.current["fwhm_constrain_fitted_baseline"]:
            self._guarded_sweep(
                group,
                "fwhm_baseline_constraint_half_width_ptp",
                cand.baseline_constraint_half_width_ptp_candidates(
                    self.current["fwhm_baseline_constraint_half_width_ptp"]
                ),
            )

    # -- group 6: diameter guess -----------------------------------------------------
    def _group_diameter_guess(self) -> None:
        group = "diameter_guess"
        has_edt = _has_mask_widths(self.template, self.sample_edges)
        self._guarded_sweep(
            group,
            "fwhm_diameter_guess_edge_attribute",
            cand.diameter_guess_edge_attribute_candidates(has_edt),
        )
        baseline_diameters = self._baseline_diameters_um()
        self._guarded_sweep(
            group,
            "fwhm_diameter_guess_um",
            cand.diameter_guess_um_candidates(baseline_diameters),
        )

    # -- group 7: rejection gates ------------------------------------------------------
    def _group_rejection_gates(self) -> None:
        group = "rejection_gates"
        self._guarded_sweep(
            group, "fwhm_reject_samples_with_center_offset", [False, True],
            score_fn=met.rejection_gates_score,
        )
        if self.current["fwhm_reject_samples_with_center_offset"]:
            self._guarded_sweep(
                group,
                "fwhm_max_fit_center_offset_um",
                cand.max_fit_center_offset_um_candidates(self.current["fwhm_max_fit_center_offset_um"]),
                score_fn=met.rejection_gates_score,
            )
            self._guarded_sweep(
                group,
                "fwhm_max_fit_center_offset_fraction_of_diameter",
                cand.max_fit_center_offset_fraction_candidates(
                    self.current["fwhm_max_fit_center_offset_fraction_of_diameter"]
                ),
                score_fn=met.rejection_gates_score,
            )
        self._guarded_sweep(
            group, "fwhm_reject_samples_with_low_fit_r2", [False, True],
            score_fn=met.rejection_gates_score,
        )
        if self.current["fwhm_reject_samples_with_low_fit_r2"]:
            self._guarded_sweep(
                group,
                "fwhm_min_fit_r2",
                cand.min_fit_r2_candidates(self.current["fwhm_min_fit_r2"]),
                score_fn=met.rejection_gates_score,
            )
        self._guarded_sweep(
            group, "fwhm_reject_samples_with_plateau_shape", [False, True],
            score_fn=met.rejection_gates_score,
        )
        if self.current["fwhm_reject_samples_with_plateau_shape"]:
            self._guarded_sweep(
                group,
                "fwhm_max_plateau_shape_ratio",
                cand.max_plateau_shape_ratio_candidates(self.current["fwhm_max_plateau_shape_ratio"]),
                score_fn=met.rejection_gates_score,
            )

    # -- orchestration --------------------------------------------------------------
    def run(self, *, max_passes: int = 1) -> None:
        """Run the fixed group sequence, optionally repeating it to
        convergence -- see `.search._Search.run`'s own docstring for why
        this is safe to repeat."""
        max_passes = max(1, int(max_passes))
        if max_passes > 1:
            self._group_total *= max_passes
        for _pass_index in range(max_passes):
            before = dict(self.current)
            self.passes_run += 1
            self._run_one_pass()
            if self.current == before:
                break

    def _run_one_pass(self) -> None:
        for name, method in (
            ("measurement_model", self._group_measurement_model),
            ("exclusion_zones", self._group_exclusion_zones),
            ("extent_and_widening", self._group_extent_and_widening),
            ("central_lobe_clipping", self._group_central_lobe_clipping),
            ("same_edge_geometry", self._group_same_edge_geometry),
            ("baseline_estimation", self._group_baseline_estimation),
            ("diameter_guess", self._group_diameter_guess),
            ("rejection_gates", self._group_rejection_gates),
        ):
            self._run_group(name, method)


def _prepared_search(
    G: nx.MultiGraph,
    *,
    raw_tiff_path: str | Path | None = None,
    voxel_size_zyx: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    progress: Optional[ProgressCallback] = None,
    sample_edge_count: Optional[int] = None,
    auto_sample_target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    groups: Optional[Iterable[str]] = None,
    axis_order: str = fwhm_diameter.CANONICAL_AXIS_ORDER,
    raw_channel: int | None = None,
    vessel_mask: Optional[np.ndarray] = None,
    config: Optional[HaemodynamicsApplyConfig] = None,
    calibration_edges: Optional[Iterable[tuple]] = None,
) -> tuple[_FwhmSearch, Optional[float]]:
    """The search :func:`optimise_fwhm_settings` runs, with everything it
    measures with read and made once -- the raw image and segmentation as a
    run reads them, the vessels as its FWHM finds them, the image PSF, the
    sample -- and the time "Auto" estimated it to take (``None`` when the
    sample size was given)."""
    if config is None:
        if raw_tiff_path is None:
            raise ValueError("The FWHM settings optimiser needs the raw image: raw_tiff_path or a config.")
        config = _measurement_config(
            starting_values, voxel_size_zyx=voxel_size_zyx, axis_order=axis_order,
            raw_tiff_path=raw_tiff_path, raw_channel=raw_channel,
        )
    config = _with_fwhm_settings(config, starting_values)
    vessels = apply.measured_vessels(G)
    # What the run calibrates on (network handling's keeps), less bridges.
    pool = (
        list(vessels.edges(keys=True)) if calibration_edges is None
        else [tuple(edge) for edge in calibration_edges if vessels.has_edge(*edge)]
    )
    total_edges = len(pool)
    if total_edges == 0 and calibration_edges is not None and vessels.number_of_edges():
        raise ValueError(
            "The run calibrates its widths on none of the network's vessels: with "
            "boundary_handling=remove_disconnected, no vessel lies on a path from an inlet to "
            "an outlet."
        )
    if total_edges == 0:
        raise ValueError("Graph has no edges to measure FWHM diameters on.")

    # Cut to the crop_on_import box G was built in, and reduced onto its
    # grid, when the image was; made float32 once, not by every trial.
    raw_volume = fwhm_diameter.float32_volume(
        apply.load_fwhm_raw_volume_like_graph(config, G),
        use_memmap=config.use_memmap,
        memmap_directory=config.memmap_directory,
    )
    if vessel_mask is not None:
        # Read as a run reads it: an ilastik "Simple Segmentation" is labels
        # 1/2, both true as bool.
        vessel_mask = _to_binary_volume_for_skeletonization(
            vessel_mask, use_memmap=config.use_memmap, memmap_directory=config.memmap_directory
        )
    voxel_sz = tuple(float(v) for v in G.graph.get("image_voxel_size_zyx", config.voxel_size_zyx))
    template = _fresh_template(vessels, config, vessel_mask)
    calibration = None if calibration_edges is None else pool
    # The blur the image shows, estimated as for the blurred-lumen model
    # whatever the run starts with: the planted vessels are drawn with it,
    # and a trial switching to that model holds its fit at it.
    image_psf, _psf_details = image_psf_from_settings(
        template,
        {**config.fwhm, "fwhm_fix_blur_to_image_psf": True, "fwhm_profile_model": "blurred_lumen"},
        voxel_size_zyx=config.voxel_size_zyx,
        raw_volume=raw_volume,
        vessel_mask=vessel_mask,
        edges=calibration,
        memmap_directory=config.memmap_directory,
    )
    psf_sigma_zyx = (
        image_psf if bool(config.fwhm_setting("fwhm_fix_blur_to_image_psf", True)) else None
    )

    estimated_seconds: Optional[float] = None
    if sample_edge_count is None:
        # With the segmentation a trial also measures planted vessels and decoys.
        sample_edge_count, estimated_seconds = _time_budget_sample(
            lambda edges: apply.measure_fwhm_diameters(
                template.copy(), config, raw_volume=raw_volume, vessel_mask=vessel_mask,
                psf_sigma_zyx=psf_sigma_zyx, edges=edges,
            ),
            vessels,
            pool,
            target_seconds=auto_sample_target_seconds,
            check_vessels_per_sampled_vessel=(
                _CHECK_VESSELS_PER_SAMPLED_VESSEL if vessel_mask is not None else 0.0
            ),
        )

    search = _FwhmSearch(
        template,
        sample_edges=_representative_edges(vessels, sample_edge_count, pool=pool),
        pool_size=total_edges,
        config=config,
        raw_volume=raw_volume,
        voxel_size_zyx=voxel_sz,
        starting_values=starting_values,
        progress=progress,
        enabled_groups=groups,
        vessel_mask=vessel_mask,
        psf_sigma_zyx=psf_sigma_zyx,
        planting_psf_um=image_psf,
    )
    return search, estimated_seconds


def _sample_note(search: "_FwhmSearch") -> str:
    """How many vessels the trials measured: a sample of the vessels, in the
    image as read -- never a coarser copy of it, which is what the report's
    downsampling line (``OptimisationResult.downsample_factor``) means."""
    return (
        f"Measured a sample of {len(search.sample_edges)} of the {search.pool_size} vessels the run "
        "measures, each among the whole vessel graph as a run measures it."
    )


def _scoring_note(with_decoys: bool) -> str:
    """What every trial is scored with whatever the run's own switches say
    (decision D3, PLAN_optimisers.md)."""
    if with_decoys:
        return (
            "Every trial was scored with the decoy check and the demotion of flagged widths on, "
            "whatever fwhm_decoy_check and fwhm_demote_flagged_edges say for the run: a setting "
            "loose enough to fit specks must not win by measuring them."
        )
    return (
        "Every trial was scored with the demotion of flagged widths on, whatever "
        "fwhm_demote_flagged_edges says for the run; with no segmentation there were no decoys "
        "to score them on (fwhm_decoy_check)."
    )


def _fallback_note(
    before: met.FwhmMeasurementQuality, after: met.FwhmMeasurementQuality, *, demoted_too: bool
) -> str:
    """How many sampled vessels FWHM leaves to the raw-section fallback --
    those it gives no width, and with ``fwhm_demote_flagged_edges`` those
    whose width is set aside -- at the starting settings and the chosen
    ones. A measure only: the score does not read it."""

    def left(quality: met.FwhmMeasurementQuality) -> int:
        return (
            quality.n_edges_total
            - quality.n_edges_measured
            + (quality.n_edges_demoted if demoted_too else 0)
        )

    return (
        f"use_raw_section_fallback is on: FWHM leaves {left(before)} of {before.n_edges_total} "
        f"sampled vessels to the raw-section fallback at your starting settings and {left(after)} "
        "at these (no FWHM width" + (", or one set aside" if demoted_too else "") + "); "
        "a measure only, not part of the score."
    )


def optimise_fwhm_settings(
    G: nx.MultiGraph,
    *,
    raw_tiff_path: str | Path | None = None,
    voxel_size_zyx: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    progress: Optional[ProgressCallback] = None,
    sample_edge_count: Optional[int] = None,
    auto_sample_target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    groups: Optional[Iterable[str]] = None,
    max_passes: int = 1,
    axis_order: str = fwhm_diameter.CANONICAL_AXIS_ORDER,
    raw_channel: int | None = None,
    vessel_mask: Optional[np.ndarray] = None,
    measure_full_graph: bool = False,
    config: Optional[HaemodynamicsApplyConfig] = None,
    calibration_edges: Optional[Iterable[tuple]] = None,
) -> OptimisationResult:
    """Empirically choose every FWHM setting in :data:`FWHM_SETTING_NAMES`
    for *G* against its raw image.

    *config* is the run's haemodynamics config, as the Diameters stage
    builds it: every trial measures with it -- its FWHM settings, the raw
    image it names, its axis order and low-RAM option -- holding the trial's
    own values of :data:`FWHM_SETTING_NAMES`, so a trial's widths are the
    run's. Without it one is made from *starting_values*, the raw image at
    *raw_tiff_path* (*axis_order*, *raw_channel*) and *voxel_size_zyx*.

    *starting_values* must have an entry for every FWHM measurement setting
    (see `_measurement_kwargs`) -- the settings this run does not manage to
    improve on simply keep their starting value. It may also carry
    ``raw_section_psf_sigma_xy_um`` / ``raw_section_psf_sigma_z_um`` (the image
    PSF, estimated from the image's wide vessels when unset) and
    ``edt_fwhm_disagreement_warn_ratio``, which are read and never changed.

    *vessel_mask*, the segmented vessel mask *G* was built from, stops each
    line at the vessel beside as a run's does, and turns on the decoy and
    planted-vessel checks every trial is scored with (see the module
    docstring); without it a trial is scored on its measurement alone.

    *calibration_edges* (``(u, v, key)``) are the vessels a run calibrates
    on -- network handling's keeps
    (``pipeline.stages.edges_network_handling_keeps``); the sample and the
    image PSF come from these alone, while every vessel is still measured
    round them. Zero-resistance bridges are never sampled.

    *sample_edge_count* trades accuracy for speed: ``None`` (the default)
    auto-selects a count estimated to keep the whole search under
    *auto_sample_target_seconds* via
    :func:`estimate_sample_edge_count_for_time_budget`; any other value
    fixes the sample size directly (a value at or above the vessel count
    samples every vessel). The search always runs against a fixed,
    width-stratified sample (see :func:`_representative_edges`), and never
    changes *G* -- unless *measure_full_graph* asks for the winning settings
    to be measured on every vessel of *G* at the end, which on a large
    network can take longer than the search did.

    *groups* restricts the search to some of :data:`FWHM_GROUP_NAMES` --
    ``None`` (the default) runs all eight; any other setting simply keeps
    its starting value, exactly as if every one of its candidates had
    failed.

    *max_passes* repeats the whole group sequence (coordinate descent) up to
    this many times, each pass starting from where the previous one left
    every setting -- see `.search._Search.run`'s own docstring for why
    repeating the sequence is safe. Stops as soon as a pass changes nothing.
    """
    search, estimated_seconds = _prepared_search(
        G,
        raw_tiff_path=raw_tiff_path,
        voxel_size_zyx=voxel_size_zyx,
        starting_values=starting_values,
        progress=progress,
        sample_edge_count=sample_edge_count,
        auto_sample_target_seconds=auto_sample_target_seconds,
        groups=groups,
        axis_order=axis_order,
        raw_channel=raw_channel,
        vessel_mask=vessel_mask,
        config=config,
        calibration_edges=calibration_edges,
    )
    # The scorecard's starting end: the first sweep's own baseline, so the
    # trial cache makes it free.
    try:
        before: Optional[met.FwhmMeasurementQuality] = search._quality({})
    except Exception:  # noqa: BLE001 - a scorecard must never fail the search it reports on
        before = None
    start = time.perf_counter()
    search.run(max_passes=max_passes)
    seconds = time.perf_counter() - start
    after: Optional[met.FwhmMeasurementQuality] = None
    try:
        after = search._quality({}) if before is not None else None
        scorecard = _scorecard(before, after) if after is not None else ()
    except Exception:  # noqa: BLE001
        scorecard = ()

    notes = [_sample_note(search), _scoring_note(search.decoy_probe is not None)]
    if bool(search.config.fwhm_setting("use_raw_section_fallback", False)) and after is not None:
        notes.append(
            _fallback_note(
                before, after,
                demoted_too=bool(search.config.fwhm_setting("fwhm_demote_flagged_edges", True)),
            )
        )

    settings = {name: search.current[name] for name in FWHM_SETTING_NAMES if name in search.current}

    if measure_full_graph:
        search._measure(apply.measured_vessels(G), search.current)

    return OptimisationResult(
        settings=settings,
        trials=tuple(search.trials),
        groups_run=tuple(dict.fromkeys(search.groups_run)),
        passes_run=search.passes_run,
        seconds=seconds,
        estimated_seconds=estimated_seconds,
        group_seconds=dict(search.group_seconds),
        scorecard=scorecard,
        notes=tuple(notes),
    )
