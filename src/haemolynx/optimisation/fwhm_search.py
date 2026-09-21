"""Sequential, image-informed search over the FWHM diameter-measurement
settings (the Diameters tab's "5. FWHM" block).

:func:`optimise_fwhm_settings` is :mod:`.search`'s own coordinate-descent
pattern -- one setting (or small joint group sharing a prerequisite toggle)
at a time, holding everything else fixed, every candidate scored by actually
running the real
:func:`haemolynx.haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`
-- applied to the seven FWHM setting groups
(:data:`FWHM_GROUP_NAMES`) instead of the Skeletonise/Graph ones.

Running the real measurement on every one of a real graph's thousands of
edges for every one of ~30 real candidate trials would be far too slow, so
this search runs against one representative subgraph
(:func:`_representative_subgraph`) instead -- a fixed-size, length-stratified
sample of the real graph's own edges, picked once up front -- and only
applies the winning settings to the caller's full graph *once*, for real, at
the end (mirroring how the skeleton/graph search re-derives
``self.current_skeleton`` at full fidelity rather than leaving it at a
downsampled approximation).

A candidate that raises is scored as a loss and the sweep continues; if every
candidate in a sweep fails, the setting simply keeps its incoming value --
same fallback as :mod:`.search`.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import networkx as nx
import numpy as np

from haemolynx.haemodynamics import automated

from . import fwhm_candidates as cand
from . import fwhm_metrics as met
from .progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
    ProgressCallback,
)
from .search import OptimisationResult, _SweepBookkeeping

#: The seven independently selectable groups this search runs, in the order
#: :meth:`_FwhmSearch.run` runs them: exclusion and extent decide *which*
#: samples exist at all before clipping/geometry/baseline decide how they
#: are fitted, and rejection gates run last since they act on the
#: already-tuned fits.
FWHM_GROUP_NAMES: tuple[str, ...] = (
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
    "exclusion_zones": "Exclusion zones (branch endpoints, junctions)",
    "extent_and_widening": "Transverse extent and widening",
    "central_lobe_clipping": "Central-lobe clipping",
    "same_edge_geometry": "Same-edge locality geometry",
    "baseline_estimation": "Profile baseline estimation",
    "diameter_guess": "Per-edge diameter guess",
    "rejection_gates": "Rejection-gate thresholds",
}

#: Every FWHM setting this search can decide, across all seven groups --
#: a setting missing from :attr:`OptimisationResult.settings` simply kept
#: its starting value (its own group was disabled, or a conditional
#: sub-sweep never ran because its prerequisite toggle stayed off).
FWHM_SETTING_NAMES: tuple[str, ...] = (
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

#: An upper bound on how many sweeps a run does, for progress display and
#: for the time-budget estimate below -- mirrors
#: `.search._GROUP_TOTAL_UPPER_BOUND`. Guarded sub-sweeps that never run
#: (a toggle stayed off) mean a real run can finish before reaching this.
_SWEEP_TOTAL_UPPER_BOUND = 28
#: Reasoned, not measured: this search's own candidate lists mostly have
#: 3-4 entries each (a handful have 2, for a bare on/off toggle) -- used
#: only to convert a per-edge timing probe into a whole-run time estimate.
_AVERAGE_CANDIDATES_PER_SWEEP = 3.5

#: "Auto" sample-size selection's own time budget, in seconds -- smaller
#: than the skeleton/graph search's five minutes since this search's own
#: per-trial cost scales with edge count, not voxel volume, and a
#: representative sample is deliberately small to begin with.
DEFAULT_AUTO_SAMPLE_TARGET_SECONDS = 180.0
#: Never sample fewer edges than this -- a handful of edges cannot
#: represent a real network's own length distribution.
_MIN_SAMPLE_EDGE_COUNT = 10

#: A candidate rejected by a guard scores this much worse than any real
#: trial -- see `.fwhm_metrics._GUARD_PENALTY`.
_GUARD_PENALTY = met._GUARD_PENALTY
#: Guard every sweep's own combined score against the one failure mode none
#: of the individual metrics can see on their own: a candidate that
#: improves its own score while quietly measuring far fewer edges, or
#: fitting them far worse, than the group's own pre-sweep baseline did --
#: see `.fwhm_metrics.regression_penalty`. Reasoned, not empirically
#: fitted: small enough that ordinary sample-to-sample noise between two
#: similar candidates never falsely trips the guard.
_MAX_MEASURED_FRACTION_REGRESSION = 0.05
_MAX_FIT_R2_REGRESSION = 0.05


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
    valid = set(inspect.signature(automated.measure_edge_diameters_fwhm_from_raw_tiff).parameters)
    return {
        name[len(prefix):]: value
        for name, value in settings.items()
        if name.startswith(prefix) and name[len(prefix):] in valid
    }


def _representative_subgraph(G: nx.MultiGraph, sample_edge_count: int) -> nx.MultiGraph:
    """A fixed-size, length-stratified sample of *G*'s own edges.

    Quantile-bucketed by each edge's own ``length`` (so short, typical, and
    long edges are all represented, not just whichever the network happens
    to have most of), then a fixed number drawn from each bucket -- a plain
    random sample of a network with a long tail of short capillaries and a
    few long trunks would rarely draw enough of the rare, long ones to say
    anything about how well they measure. Deterministic (seeded), so a
    repeated call against the same graph and count picks the same edges.
    """
    total_edges = G.number_of_edges()
    sample_edge_count = max(1, min(int(sample_edge_count), total_edges))
    if sample_edge_count >= total_edges:
        return G.copy()

    edge_keys = list(G.edges(keys=True))
    lengths = np.asarray(
        [float(G.edges[e].get("length", 0.0) or 0.0) for e in edge_keys], dtype=float
    )
    n_buckets = min(5, max(1, sample_edge_count))
    quantile_edges = np.quantile(lengths, np.linspace(0.0, 1.0, n_buckets + 1))
    bucket_of = np.clip(
        np.searchsorted(quantile_edges[1:-1], lengths, side="right"), 0, n_buckets - 1
    )

    rng = np.random.default_rng(0)
    per_bucket = max(1, sample_edge_count // n_buckets)
    selected: list[int] = []
    for bucket in range(n_buckets):
        idx_in_bucket = np.flatnonzero(bucket_of == bucket)
        rng.shuffle(idx_in_bucket)
        selected.extend(int(i) for i in idx_in_bucket[:per_bucket])

    if len(selected) < sample_edge_count:
        remaining = np.array(
            [i for i in range(total_edges) if i not in set(selected)], dtype=int
        )
        rng.shuffle(remaining)
        selected.extend(int(i) for i in remaining[: sample_edge_count - len(selected)])
    selected = selected[:sample_edge_count]

    chosen_edges = [edge_keys[i] for i in selected]
    return G.edge_subgraph(chosen_edges).copy()


def _sample_edge_count_from_probe_seconds(
    probe_seconds: float, probe_edge_count: int, total_edges: int, target_seconds: float
) -> int:
    """Pure arithmetic half of :func:`estimate_sample_edge_count_for_time_budget`,
    split out so the decision itself is directly testable without timing
    anything real -- mirrors `.search._factor_from_probe_seconds`."""
    if probe_seconds <= 0.0 or probe_edge_count <= 0:
        return total_edges
    per_edge_seconds = probe_seconds / float(probe_edge_count)
    total_calls = _SWEEP_TOTAL_UPPER_BOUND * _AVERAGE_CANDIDATES_PER_SWEEP
    budget_edges = int(target_seconds / max(per_edge_seconds * total_calls, 1e-12))
    return max(_MIN_SAMPLE_EDGE_COUNT, min(budget_edges, total_edges))


def estimate_sample_edge_count_for_time_budget(
    G: nx.MultiGraph,
    *,
    raw_volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
) -> int:
    """The largest sample-edge count estimated to keep a full search under
    *target_seconds* on the machine it actually runs on.

    Times one real measurement call on a small probe subgraph of *this*
    graph's own edges, then extrapolates linearly in edge count (this
    search's own cost scales per-edge, unlike the skeleton/graph search's
    cubic voxel-volume scaling) across the roughly
    :data:`_SWEEP_TOTAL_UPPER_BOUND` x :data:`_AVERAGE_CANDIDATES_PER_SWEEP`
    real trials a full run makes. Falls back to every edge (no subsampling)
    if the probe itself cannot run at all, so "Auto" never fails a run just
    because estimating its own runtime did.
    """
    total_edges = G.number_of_edges()
    if total_edges <= _MIN_SAMPLE_EDGE_COUNT:
        return total_edges
    probe_edge_count = min(10, total_edges)
    probe_graph = _representative_subgraph(G, probe_edge_count)
    kwargs = _measurement_kwargs(starting_values)

    def _probe_once() -> None:
        automated.measure_edge_diameters_fwhm_from_raw_tiff(
            probe_graph.copy(),
            raw_volume=raw_volume,
            voxel_size_zyx=voxel_size_zyx,
            store_profile_debug=False,
            **kwargs,
        )

    try:
        # One untimed warm-up call first -- see `.search`'s own
        # `estimate_downsample_factor_for_time_budget` for why a fresh
        # process's first real call pays a one-time setup cost that would
        # otherwise inflate the estimate by orders of magnitude.
        _probe_once()
        t0 = time.perf_counter()
        _probe_once()
        probe_seconds = time.perf_counter() - t0
    except Exception:  # noqa: BLE001 - estimating runtime must never block a real run
        return total_edges

    return _sample_edge_count_from_probe_seconds(
        probe_seconds, probe_edge_count, total_edges, target_seconds
    )


class _FwhmSearch(_SweepBookkeeping):
    """Mutable state for one call to :func:`optimise_fwhm_settings`."""

    def __init__(
        self,
        sample_graph: nx.MultiGraph,
        *,
        raw_volume: np.ndarray,
        voxel_size_zyx: tuple[float, float, float],
        starting_values: Mapping[str, Any],
        progress: Optional[ProgressCallback],
        enabled_groups: Optional[Iterable[str]] = None,
    ) -> None:
        #: Never mutated -- every trial copies this fresh (see `_run_trial`),
        #: matching `.search._Search`'s own "always re-derive from source"
        #: discipline, since a measurement call writes its results directly
        #: onto the graph's own edges.
        self.sample_graph_template = sample_graph
        self.raw_volume = raw_volume
        self.voxel_size_zyx = voxel_size_zyx
        self.current: dict[str, Any] = dict(starting_values)
        super().__init__(
            progress=progress,
            enabled_groups=enabled_groups,
            group_total=len(FWHM_GROUP_NAMES) * 4,
        )
        self.passes_run: int = 0

    # -- real-measurement trial helpers ------------------------------------------
    def _run_trial(self, overrides: Mapping[str, Any]) -> tuple[nx.MultiGraph, dict[str, Any]]:
        """Run the real measurement on a fresh copy of the sample subgraph
        with `self.current` plus *overrides* -- never a previous trial's own
        output."""
        settings = {**self.current, **overrides}
        graph_trial = self.sample_graph_template.copy()
        summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
            graph_trial,
            raw_volume=self.raw_volume,
            voxel_size_zyx=self.voxel_size_zyx,
            store_profile_debug=True,
            **_measurement_kwargs(settings),
        )
        return graph_trial, summary

    def _quality(self, overrides: Mapping[str, Any]) -> met.FwhmMeasurementQuality:
        graph_trial, summary = self._run_trial(overrides)
        mult = float({**self.current, **overrides}["fwhm_min_total_extent_multiplier"])
        return met.fwhm_measurement_quality(graph_trial, summary, min_total_extent_multiplier=mult)

    def _baseline_diameters_um(self) -> np.ndarray:
        """This edge sample's own accepted FWHM diameters under
        `self.current` right now -- a fresh real measurement, not cached
        across calls at different points in the run (`self.current` keeps
        changing as earlier groups decide their own settings)."""
        graph_trial, _summary = self._run_trial({})
        diameters: list[float] = []
        for _u, _v, data in graph_trial.edges(data=True):
            diameters.extend(float(d) for d in (data.get("fwhm_diameter_samples_um") or []))
        return np.asarray(diameters, dtype=float)

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
        for _u, _v, data in self.sample_graph_template.edges(data=True):
            total_len = float(data.get("length", 0.0) or 0.0)
            if total_len <= 0:
                continue
            n_samples = max(1, int(np.floor(total_len / nominal)) + 1)
            if n_samples > 1:
                spacings.append(total_len / (n_samples - 1))
        return np.asarray(spacings, dtype=float)

    # -- sweeps -------------------------------------------------------------------
    def _sweep(self, group: str, setting: str, candidate_values: list, cost_fn) -> Any:
        self._emit(GROUP_STARTED, group)
        best_score = float("inf")
        best_value = self.current.get(setting)
        for index, value in enumerate(candidate_values):
            note = ""
            try:
                score = float(cost_fn(value))
            except Exception as error:  # noqa: BLE001 - one bad candidate must not end the sweep
                score = float("inf")
                note = f"failed: {error}"
            self._record(group, setting, value, score, note)
            self._emit(CANDIDATE_EVALUATED, group, candidate_index=index, candidate_total=len(candidate_values))
            if score < best_score:
                best_score, best_value = score, value
        self.current[setting] = best_value
        self._emit(GROUP_FINISHED, group, winner={setting: best_value})
        self._group_index += 1
        return best_value

    def _guarded_sweep(
        self, group: str, setting: str, candidate_values: list, *, score_fn=None
    ) -> Any:
        """:meth:`_sweep`, guarded against a candidate that improves its own
        combined score while regressing `measured_fraction` or
        `mean_fit_r2` below this sub-sweep's own pre-sweep baseline (see
        `.fwhm_metrics.regression_penalty`) -- the workhorse behind every
        group method below except where a group-specific *score_fn* is
        needed (`rejection_gates`).
        """
        score_of = score_fn or (lambda quality: quality.score)
        baseline = self._quality({})

        def cost_fn(value: Any) -> float:
            quality = self._quality({setting: value})
            penalty = met.regression_penalty(
                quality.measured_fraction, baseline.measured_fraction, _MAX_MEASURED_FRACTION_REGRESSION
            )
            penalty += met.regression_penalty(
                quality.mean_fit_r2, baseline.mean_fit_r2, _MAX_FIT_R2_REGRESSION
            )
            return score_of(quality) + penalty

        return self._sweep(group, setting, candidate_values, cost_fn)

    # -- group 1: exclusion zones --------------------------------------------------
    def _group_exclusion_zones(self) -> None:
        group = "exclusion_zones"
        edge_lengths = np.asarray(
            [
                float(data.get("length", 0.0) or 0.0)
                for _u, _v, data in self.sample_graph_template.edges(data=True)
            ],
            dtype=float,
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
        has_edt = any(
            float(data.get("edt_diameter_um") or 0.0) > 0.0
            for _u, _v, data in self.sample_graph_template.edges(data=True)
        )
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
            ("exclusion_zones", self._group_exclusion_zones),
            ("extent_and_widening", self._group_extent_and_widening),
            ("central_lobe_clipping", self._group_central_lobe_clipping),
            ("same_edge_geometry", self._group_same_edge_geometry),
            ("baseline_estimation", self._group_baseline_estimation),
            ("diameter_guess", self._group_diameter_guess),
            ("rejection_gates", self._group_rejection_gates),
        ):
            if self._group_enabled(name):
                method()


def optimise_fwhm_settings(
    G: nx.MultiGraph,
    *,
    raw_tiff_path: str | Path,
    voxel_size_zyx: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    progress: Optional[ProgressCallback] = None,
    sample_edge_count: Optional[int] = None,
    auto_sample_target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    groups: Optional[Iterable[str]] = None,
    max_passes: int = 1,
    axis_order: str = automated.CANONICAL_AXIS_ORDER,
) -> OptimisationResult:
    """Empirically choose every FWHM setting in :data:`FWHM_SETTING_NAMES`
    for *G* against the raw image at *raw_tiff_path*.

    *starting_values* must have an entry for every FWHM measurement setting
    (see `_measurement_kwargs`) -- the settings this run does not manage to
    improve on simply keep their starting value.

    *sample_edge_count* trades accuracy for speed: ``None`` (the default)
    auto-selects a count estimated to keep the whole search under
    *auto_sample_target_seconds* via
    :func:`estimate_sample_edge_count_for_time_budget`; any other value
    fixes the sample size directly (a value at or above *G*'s own edge count
    runs against every edge, no subsampling). The search always runs
    against a fixed, length-stratified sample (see
    :func:`_representative_subgraph`) rather than the full graph, and
    applies the winning settings to *G* itself, for real, exactly once at
    the end -- *G* is never mutated before that point.

    *groups* restricts the search to some of :data:`FWHM_GROUP_NAMES` --
    ``None`` (the default) runs all seven; any other setting simply keeps
    its starting value, exactly as if every one of its candidates had
    failed.

    *max_passes* repeats the whole group sequence (coordinate descent) up to
    this many times, each pass starting from where the previous one left
    every setting -- see `.search._Search.run`'s own docstring for why
    repeating the sequence is safe. Stops as soon as a pass changes nothing.
    """
    total_edges = G.number_of_edges()
    if total_edges == 0:
        raise ValueError("Graph has no edges to measure FWHM diameters on.")

    raw_volume = automated.load_single_channel_tiff_volume(raw_tiff_path, axis_order=axis_order)

    if sample_edge_count is None:
        sample_edge_count = estimate_sample_edge_count_for_time_budget(
            G,
            raw_volume=raw_volume,
            voxel_size_zyx=voxel_size_zyx,
            starting_values=starting_values,
            target_seconds=auto_sample_target_seconds,
        )
    sample_edge_count = max(1, min(int(sample_edge_count), total_edges))

    sample_graph = _representative_subgraph(G, sample_edge_count)

    search = _FwhmSearch(
        sample_graph,
        raw_volume=raw_volume,
        voxel_size_zyx=voxel_size_zyx,
        starting_values=starting_values,
        progress=progress,
        enabled_groups=groups,
    )
    search.run(max_passes=max_passes)

    settings = {name: search.current[name] for name in FWHM_SETTING_NAMES if name in search.current}

    # Apply the winning settings to the caller's full graph, for real, once
    # -- mirrors `optimise_skeleton_and_graph_settings` re-deriving
    # `self.current_skeleton` at full fidelity rather than leaving it at a
    # downsampled approximation.
    final_settings = {**starting_values, **settings}
    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_volume=raw_volume,
        voxel_size_zyx=voxel_size_zyx,
        store_profile_debug=True,
        **_measurement_kwargs(final_settings),
    )

    return OptimisationResult(
        settings=settings,
        trials=tuple(search.trials),
        downsample_factor=max(1, round(total_edges / sample_edge_count)),
        groups_run=tuple(dict.fromkeys(search.groups_run)),
        passes_run=search.passes_run,
    )
