"""Sequential, image-informed search over segmentation-cleanup, Skeletonise
and Graph tab settings.

:func:`optimise_skeleton_and_graph_settings` decides every setting in
:data:`OPTIMISE_SETTING_NAMES` by running a fixed sequence of *sweeps*, each
varying one setting (or, for the seven segmentation-cleanup steps' own knobs,
the four bundle-refinement knobs, the five thick-vessel refinement knobs, the
three centreline-smoothing knobs, and the cluster-collapse method plus its own
one method-specific knob, a short joint sweep over the settings one function
call actually shares) while holding every other setting at its current value.
The order matches the order the real pipeline itself applies these parameters
-- segmentation cleanup runs first, directly on the raw mask, in the same
fixed order ``clean_segmented_mask_for_skeletonisation`` itself applies its
own steps (fill cavities, remove whiskers, split narrow necks, close small
gaps, reconnect, smooth, remove small volumes); then thick-vessel gating
decides which raw skeleton every later sweep even sees, then min-branch-length,
bundle refinement, closing, gap-bridging, and finally
component-connectivity/filtering -- so that at every sweep, "everything not
yet decided" really is what the real pipeline call would still use by default.
The graph-side settings (reconnect thresholds, cluster collapse, stub
pruning, centreline smoothing) run afterwards, each on the single
already-decided skeleton or graph the previous sweep produced.

Segmentation cleanup's own sweeps score each candidate with
:func:`haemolynx.preprocessing.score_segmented_mask` (the same 0-10
fragmentation/connectivity/boundary/noise/resolution score the "Check
segmented image" button reports) rather than a skeleton-connectivity or
braid metric -- these settings act on the mask itself, before any skeleton
exists to measure.

This is coordinate-descent, not a joint grid search over all 21 settings at
once (which would be combinatorial): every sweep still calls the real
``preprocess_skeleton_for_graph`` / ``build_graph_from_skeleton`` /
``smooth_graph_centrelines`` for every candidate it tries and scores the
result with :mod:`.metrics`, so every setting gets a genuine, image-specific
empirical test -- just one setting (or small joint group) at a time, which
keeps the total number of real pipeline calls additive across settings rather
than their product.

A candidate that raises is scored as a loss and the sweep continues; if every
candidate in a sweep fails, the setting simply keeps its incoming value. A
sweep that has nothing to build on (an empty mask, a skeleton with no
foreground) never crashes -- the guarded groups (thick-vessel gating) skip
outright, and the metric functions all handle empty input -- except the graph
sweeps, which need a real graph to hold anything on: if every
``graph_reconnect_threshold`` candidate fails to build a graph at all, the
search raises, because there is nothing later sweeps could run on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from haemolynx import graph as graph_mod
from haemolynx import preprocessing

from . import candidates as cand
from . import metrics as met
from .progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
    OptimisationEvent,
    ProgressCallback,
)

#: Settings this search decides, on the Input tab's segmentation-cleanup
#: block (the raw mask, before skeletonisation) -- in the same fixed order
#: `clean_segmented_mask_for_skeletonisation` itself applies them.
SEGMENTATION_CLEANUP_SETTING_NAMES: tuple[str, ...] = (
    "segmentation_cleanup_fill_cavities",
    "segmentation_cleanup_remove_whiskers",
    "segmentation_cleanup_whisker_radius_um",
    "segmentation_cleanup_split_narrow_necks",
    "segmentation_cleanup_split_min_marker_separation_um",
    "segmentation_cleanup_split_min_pinch_radius_ratio",
    "segmentation_cleanup_split_min_body_radius_um",
    "segmentation_cleanup_close_gaps",
    "segmentation_cleanup_close_gaps_radius_um",
    "segmentation_cleanup_reconnect_gaps",
    "segmentation_cleanup_reconnect_max_bridge_distance_um",
    "segmentation_cleanup_reconnect_min_cylindricality",
    "segmentation_cleanup_reconnect_max_axis_angle_degrees",
    "segmentation_cleanup_reconnect_min_facing_cosine",
    "segmentation_cleanup_reconnect_max_radius_ratio",
    "segmentation_cleanup_smooth_surfaces",
    "segmentation_cleanup_smooth_method",
    "segmentation_cleanup_smooth_sigma_um",
    "segmentation_cleanup_smooth_morphological_radius_um",
    "segmentation_cleanup_remove_small_volumes",
    "segmentation_cleanup_remove_small_min_volume_um3",
)

#: Settings this search decides, on the Skeletonise tab.
SKELETON_SETTING_NAMES: tuple[str, ...] = (
    "use_thick_vessel_skeletonisation",
    "skeleton_thick_vessel_min_radius_um",
    "skeleton_fill_mask_holes_before_thickness",
    "skeleton_thick_vessel_wall_absorption_um",
    "skeleton_thick_vessel_flake_filter_um",
    "skeleton_thick_vessel_max_bridge_radius_multiple",
    "skeleton_thick_vessel_max_bridge_distance_um",
    "skeleton_thick_vessel_bridge_radius_smoothing_um",
    "skeleton_min_branch_length",
    "skeleton_bundle_scan_size",
    "skeleton_bundle_density_fraction",
    "skeleton_bundle_max_connections_per_hub",
    "skeleton_bundle_hub_min_spacing",
    "skeleton_closing_radius",
    "skeleton_bridge_gap_size",
    "skeleton_max_bridge_distance",
    "skeleton_component_connectivity",
    "skeleton_min_component_percent",
)

#: Settings this search decides, on the Graph tab.
GRAPH_SETTING_NAMES: tuple[str, ...] = (
    "graph_reconnect_threshold",
    "final_orphan_reconnect_threshold",
    "cluster_collapse_distance",
    "cluster_collapse_method",
    "cluster_collapse_max_radial_dispersion",
    "cluster_collapse_persistence_search_multiple",
    "min_stub_length",
    "smooth_centrelines",
    "centreline_smoothing_method",
    "centreline_smoothing_iterations",
    "centreline_max_deviation",
)

OPTIMISE_SETTING_NAMES: tuple[str, ...] = (
    SEGMENTATION_CLEANUP_SETTING_NAMES + SKELETON_SETTING_NAMES + GRAPH_SETTING_NAMES
)

#: Deliberately not in scope: the cartwheel-hub-guard settings
#: (detect_cartwheel_hub_artifacts, cartwheel_hub_min_degree,
#: cartwheel_hub_max_radial_dispersion, cartwheel_hub_tangent_length_um) and
#: the *_consistency_warn_below / missing_vessel_* settings are all read-only
#: QC checks -- they only decide whether a warning is logged, never anything
#: about the skeleton or graph itself. There is no "better" or "worse" value
#: for a warning threshold in the sense the rest of this search means it, so
#: nothing here empirically tests them; they stay exactly as the GUI/config
#: already had them.

#: An upper bound on how many sweeps a run does, for progress display. Guarded
#: sweeps that are skipped mean a real run can finish before reaching this.
_GROUP_TOTAL_UPPER_BOUND = 44

#: The eleven independently selectable groups this search runs, in the order
#: :meth:`_Search.run` runs them -- a GUI's "choose optimisation types"
#: checkbox list is built from this and :data:`GROUP_LABELS`, one checkbox
#: per name.
GROUP_NAMES: tuple[str, ...] = (
    "segmentation_cleanup",
    "thick_vessel_gating",
    "min_branch_length",
    "bundle_refinement",
    "closing_radius",
    "gap_bridging",
    "connectivity_and_component_filter",
    "reconnect_thresholds",
    "cluster_collapse_distance",
    "min_stub_length",
    "centreline_smoothing",
)

#: A short, human-readable label for each group, for a GUI checkbox list.
GROUP_LABELS: dict[str, str] = {
    "segmentation_cleanup": "Segmentation cleanup (mask, before skeletonisation)",
    "thick_vessel_gating": "Thick-vessel gating (on/off, radius, refinement)",
    "min_branch_length": "Minimum branch length",
    "bundle_refinement": "Bundle refinement (confluence hubs)",
    "closing_radius": "Closing radius",
    "gap_bridging": "Gap bridging",
    "connectivity_and_component_filter": "Component connectivity + filtering",
    "reconnect_thresholds": "Graph reconnect thresholds",
    "cluster_collapse_distance": "Cluster collapse",
    "min_stub_length": "Minimum stub length",
    "centreline_smoothing": "Centreline smoothing",
}

#: Settings measured in voxels of whatever grid the search actually ran on.
#: When the search runs on a downsampled copy for speed (see
#: :func:`resolve_auto_downsample_factor`), these need scaling back up by the
#: downsample factor before they mean anything on the full-resolution grid a
#: real run skeletonises. Every other numeric setting this search decides is
#: already resolution-independent: a physical micron distance (scaled voxel
#: size already accounts for those), a fraction, a count, or a mode choice.
_VOXEL_SCALED_SETTING_NAMES: tuple[str, ...] = (
    "skeleton_closing_radius",
    "skeleton_bridge_gap_size",
    "skeleton_max_bridge_distance",
    "skeleton_min_branch_length",
    "skeleton_bundle_scan_size",
    "skeleton_bundle_hub_min_spacing",
)

#: Downsampling factors "Optimisation downsampling" offers, "off" (1) first.
DOWNSAMPLE_FACTORS: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Above this many voxels, "Auto" downsampling starts choosing a factor > 1 --
#: chosen so a typical sweep's real preprocessing/graph-building calls stay in
#: the tens-of-seconds range rather than minutes on a whole-brain-scale stack.
AUTO_DOWNSAMPLE_TARGET_VOXELS = 20_000_000


def resolve_auto_downsample_factor(shape: tuple[int, ...]) -> int:
    """The smallest offered factor bringing *shape* under the search's own target size."""
    total = 1
    for dim in shape:
        total *= int(dim)
    for factor in DOWNSAMPLE_FACTORS:
        if total / (factor ** 3) <= AUTO_DOWNSAMPLE_TARGET_VOXELS:
            return factor
    return DOWNSAMPLE_FACTORS[-1]


def _downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Block-max reduction: a downsampled voxel is foreground if any voxel in
    its block was. Plain striding could skip clean over a thin vessel that
    happens to fall between the sampled points; this cannot lose one."""
    if factor <= 1:
        return mask
    from skimage.measure import block_reduce

    return block_reduce(mask, block_size=(factor, factor, factor), func=np.max)


def _downsample_intensity(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean reduction for a raw intensity image -- unlike the mask's
    own block-*max* (foreground survives if any voxel in the block was),
    an intensity value should average over its block, or Otsu thresholding
    the downsampled copy would systematically read brighter than the real
    volume and bias every added/removed-voxel comparison."""
    if factor <= 1:
        return image
    from skimage.measure import block_reduce

    return block_reduce(image, block_size=(factor, factor, factor), func=np.mean)

#: Reject a closing/bridging candidate that merges components further apart
#: than this many typical-vessel-radii -- more likely two distinct vessels
#: than one gap.
_GAP_FUSION_RATIO_GUARD = 2.0
#: Reject a min-branch-length candidate that strips more of the raw skeleton
#: than this fraction.
_MAX_VOXELS_REMOVED_FRACTION = 0.15
#: Reject a min-stub-length candidate that removes more than this fraction of
#: total graph length.
_MAX_GRAPH_LENGTH_REMOVED_FRACTION = 0.05
#: Reject a centreline-smoothing candidate that shrinks total length by more
#: than this fraction (the over-smoothing failure mode `graph.smoothing`'s own
#: docstring warns about).
_MAX_LENGTH_SHRINK_FRACTION = 0.15
#: A candidate rejected by a guard scores this much worse than any real trial.
_GUARD_PENALTY = 1000.0

#: Guard several groups' own narrow proxy metrics against the one failure
#: mode none of them can see: a candidate that improves its own score while
#: quietly dropping real coverage of the segmented mask -- measured against
#: the existing, already-tested `preprocessing.diagnose_skeleton_mask_consistency`/
#: `diagnose_vessels_missing_from_skeleton`/`graph.diagnose_skeleton_graph_consistency`
#: (previously computed only as end-of-run warning diagnostics, never used
#: to choose between candidates). Reasoned, not empirically fitted: small
#: enough that ordinary EDT/labelling rounding noise between two very
#: similar candidates never falsely trips the guard, big enough to catch a
#: candidate that genuinely erodes coverage rather than one that's a coin
#: flip away from the baseline.
_MAX_COVERAGE_FRACTION_REGRESSION = 0.02
_MAX_MISSING_VESSEL_FRACTION_REGRESSION = 0.02
#: Same idea, applied to the "does a segmentation-cleanup candidate invent
#: foreground the raw reference image does not support" check
#: (`preprocessing.compare_segmentation_to_raw_image`) -- only ever
#: evaluated when a raw image is actually supplied.
_MAX_RAW_ADDED_FRACTION_REGRESSION = 0.02

#: Below this physical volume, a mask connected component is segmentation
#: noise, not a real vessel worth protecting from pruning -- matches
#: `preprocessing.segmentation_raw_comparison.DEFAULT_MISSED_STRUCTURE_MIN_VOLUME_UM3`'s
#: own reasoning, adopted here for the same reason it was adopted there: a
#: real run against noisy real data showed `diagnose_vessels_missing_from_skeleton`'s
#: own default floor (``min_vessel_voxels=2``) counts tiny segmentation-noise
#: flecks in the mask as "real vessels" the guard must protect, which made
#: `skeleton_min_branch_length` -- whose entire purpose is pruning exactly
#: that noise -- unable to prune anything at all (every non-zero candidate
#: got guard-rejected). A voxel-count floor also does not generalise across
#: resolutions the way a physical volume does.
_MIN_VESSEL_VOLUME_UM3_FOR_GUARD = 20.0


@dataclass(frozen=True)
class TrialRecord:
    """One candidate value, tried and scored, kept for the run's report."""

    group: str
    setting: str
    value: Any
    score: float
    note: str = ""


@dataclass(frozen=True)
class OptimisationResult:
    """The winning value for every setting this search decided, and the trials tried."""

    settings: dict[str, Any]
    trials: tuple[TrialRecord, ...]
    #: 1 when the search ran at full resolution; > 1 when it ran on a
    #: downsampled copy (see :func:`resolve_auto_downsample_factor`) and the
    #: voxel-scaled settings were multiplied back up before being returned.
    downsample_factor: int = 1
    #: Which of :data:`GROUP_NAMES` actually ran; a name missing here kept its
    #: starting value untouched.
    groups_run: tuple[str, ...] = field(default_factory=lambda: GROUP_NAMES)


def _skeleton_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    """``preprocess_skeleton_for_graph`` keyword arguments from a settings dict."""
    return dict(
        min_branch_length=int(settings["skeleton_min_branch_length"]),
        max_bridge_distance=int(settings["skeleton_max_bridge_distance"]),
        component_connectivity=int(settings["skeleton_component_connectivity"]),
        min_component_fraction=float(settings["skeleton_min_component_percent"]) / 100.0,
        closing_radius=int(settings["skeleton_closing_radius"]),
        bridge_gap_size=int(settings["skeleton_bridge_gap_size"]),
        bundle_scan_size=int(settings["skeleton_bundle_scan_size"]),
        bundle_density_fraction=float(settings["skeleton_bundle_density_fraction"]),
        bundle_max_connections_per_hub=int(settings["skeleton_bundle_max_connections_per_hub"]),
        bundle_hub_min_spacing=int(settings["skeleton_bundle_hub_min_spacing"]),
    )


def _thick_vessel_kwargs(
    settings: Mapping[str, Any], voxel_size_zyx: tuple[float, float, float]
) -> dict[str, Any]:
    """``skeletonize_thickness_gated`` keyword arguments from a settings dict.

    ``restrict_thick_to_mask`` is deliberately not threaded through: it needs
    the configured large/small vessel masks loaded from disk, which would
    pull ``haemolynx.io`` and mask-loading into this package's dependencies
    for a setting that names *which other settings to trust*, not a
    quality knob with a better or worse value an image measurement could
    inform -- it stays whatever the GUI/config already had it at.
    """
    return dict(
        min_radius_um=float(settings["skeleton_thick_vessel_min_radius_um"]),
        voxel_size_zyx=voxel_size_zyx,
        fill_mask_holes=bool(settings["skeleton_fill_mask_holes_before_thickness"]),
        wall_absorption_um=settings["skeleton_thick_vessel_wall_absorption_um"],
        flake_filter_um=settings["skeleton_thick_vessel_flake_filter_um"],
        max_bridge_radius_multiple=settings["skeleton_thick_vessel_max_bridge_radius_multiple"],
        max_bridge_distance_um=settings["skeleton_thick_vessel_max_bridge_distance_um"],
        bridge_radius_smoothing_um=float(settings["skeleton_thick_vessel_bridge_radius_smoothing_um"]),
    )


def _raw_skeleton(
    mask: np.ndarray, settings: Mapping[str, Any], voxel_size_zyx: tuple[float, float, float]
) -> np.ndarray:
    if settings["use_thick_vessel_skeletonisation"]:
        return preprocessing.skeletonize_thickness_gated(
            mask, **_thick_vessel_kwargs(settings, voxel_size_zyx)
        )
    return preprocessing.skeletonize_volume(mask)


class _Search:
    """Mutable state for one call to :func:`optimise_skeleton_and_graph_settings`."""

    def __init__(
        self,
        raw_mask: np.ndarray,
        voxel_size_xyz: tuple[float, float, float],
        starting_values: Mapping[str, Any],
        progress: Optional[ProgressCallback],
        enabled_groups: Optional[Iterable[str]] = None,
        raw_image: Optional[np.ndarray] = None,
    ) -> None:
        self.raw_mask = np.asarray(raw_mask, dtype=bool)
        #: An optional raw (unsegmented) reference image, already resampled
        #: to match `raw_mask`'s own shape (see `optimise_skeleton_and_graph_settings`).
        #: `None` (the default) is today's exact behaviour -- only
        #: `segmentation_cleanup` reads this, and only to add an extra,
        #: additive guard (see `_group_segmentation_cleanup`), never to
        #: change what the group otherwise decides.
        if raw_image is not None and raw_image.shape != self.raw_mask.shape:
            raise ValueError(
                "raw_image shape does not match the segmented mask shape: "
                f"{raw_image.shape} != {self.raw_mask.shape}"
            )
        self.raw_image = raw_image
        #: The mask exactly as given, before any segmentation-cleanup
        #: candidate is tried -- every cleanup trial re-cleans this same
        #: fixed input (never the previous trial's own output), matching how
        #: `_preprocess_trial` always re-runs from `self.raw_skeleton`. Kept
        #: even when the segmentation-cleanup group is disabled, since
        #: `self.raw_mask` may still be reassigned to a cleaned copy once
        #: that group actually runs.
        self.original_raw_mask: np.ndarray = self.raw_mask
        voxel_size_xyz = tuple(float(v) for v in voxel_size_xyz)
        # zyx is xyz reversed -- the same relationship
        # haemolynx.io.voxel_size_zyx_from_xyz encodes, kept local here so
        # `optimisation` never imports `haemolynx.io`.
        self.voxel_size_zyx: tuple[float, float, float] = tuple(reversed(voxel_size_xyz))
        self.current: dict[str, Any] = dict(starting_values)
        self.progress = progress
        #: ``None`` means every group runs; otherwise only the named ones do
        #: -- see :meth:`_group_enabled`.
        self.enabled_groups: Optional[frozenset[str]] = (
            frozenset(enabled_groups) if enabled_groups is not None else None
        )
        self.groups_run: list[str] = []
        self.trials: list[TrialRecord] = []
        self.raw_skeleton: np.ndarray = np.zeros_like(self.raw_mask)
        self.current_skeleton: np.ndarray = self.raw_skeleton
        self.current_graph = None
        self._group_index = 0
        #: Set fresh at the top of `_group_closing_radius`/`_group_gap_bridging`
        #: (each group's own entering state) -- `_fusion_cost` is a shared
        #: method, not a sweep-local closure, so its own coverage-regression
        #: guard needs this as instance state rather than a captured local.
        self._fusion_baseline_coverage: float = 1.0

        radius_map = preprocessing.inscribed_radius_map(self.raw_mask, self.voxel_size_zyx)
        nonzero = radius_map[radius_map > 0]
        self.typical_radius_um = float(np.median(nonzero)) if nonzero.size else 1.0

    def _group_enabled(self, name: str) -> bool:
        enabled = self.enabled_groups is None or name in self.enabled_groups
        if enabled:
            self.groups_run.append(name)
        return enabled

    # -- progress / bookkeeping -------------------------------------------------
    def _emit(self, kind: str, group_name: str, **extra: Any) -> None:
        if self.progress is None:
            return
        self.progress(
            OptimisationEvent(
                kind=kind,
                group_index=self._group_index,
                group_total=_GROUP_TOTAL_UPPER_BOUND,
                group_name=group_name,
                **extra,
            )
        )

    def _record(self, group: str, setting: str, value: Any, score: float, note: str = "") -> None:
        self.trials.append(TrialRecord(group=group, setting=setting, value=value, score=score, note=note))

    def _sweep(self, group: str, setting: str, candidate_values: list, cost_fn) -> Any:
        """Try every candidate for one setting; keep the lowest-cost one.

        *cost_fn(value) -> float* runs the real pipeline code for one candidate
        and returns its cost (lower is better); it may raise, which scores that
        candidate as a loss without ending the sweep. If every candidate fails,
        the setting keeps whatever value it already had in ``self.current``.
        """
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

    # -- skeleton-side trial helpers ---------------------------------------------
    def _preprocess_trial(self, overrides: Mapping[str, Any]) -> np.ndarray:
        settings = {**self.current, **overrides}
        return preprocessing.preprocess_skeleton_for_graph(self.raw_skeleton, **_skeleton_kwargs(settings))

    def _connectivity(self) -> Optional[int]:
        return int(self.current["skeleton_component_connectivity"])

    # -- group 0: segmentation cleanup (raw mask, before skeletonisation) --------
    def _cleanup_kwargs(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        """``clean_segmented_mask_for_skeletonisation`` keyword arguments from
        a settings dict -- the ``segmentation_cleanup_``-prefixed schema names
        stripped down to the plain names that function actually takes."""
        return dict(
            fill_cavities=bool(settings["segmentation_cleanup_fill_cavities"]),
            remove_whiskers=bool(settings["segmentation_cleanup_remove_whiskers"]),
            whisker_radius_um=float(settings["segmentation_cleanup_whisker_radius_um"]),
            split_narrow_necks=bool(settings["segmentation_cleanup_split_narrow_necks"]),
            split_min_marker_separation_um=float(
                settings["segmentation_cleanup_split_min_marker_separation_um"]
            ),
            split_min_pinch_radius_ratio=float(
                settings["segmentation_cleanup_split_min_pinch_radius_ratio"]
            ),
            split_min_body_radius_um=float(settings["segmentation_cleanup_split_min_body_radius_um"]),
            close_gaps=bool(settings["segmentation_cleanup_close_gaps"]),
            close_gaps_radius_um=float(settings["segmentation_cleanup_close_gaps_radius_um"]),
            reconnect_gaps=bool(settings["segmentation_cleanup_reconnect_gaps"]),
            reconnect_max_bridge_distance_um=float(
                settings["segmentation_cleanup_reconnect_max_bridge_distance_um"]
            ),
            reconnect_min_cylindricality=float(
                settings["segmentation_cleanup_reconnect_min_cylindricality"]
            ),
            reconnect_max_axis_angle_degrees=float(
                settings["segmentation_cleanup_reconnect_max_axis_angle_degrees"]
            ),
            reconnect_min_facing_cosine=float(
                settings["segmentation_cleanup_reconnect_min_facing_cosine"]
            ),
            reconnect_max_radius_ratio=float(
                settings["segmentation_cleanup_reconnect_max_radius_ratio"]
            ),
            smooth_surfaces=bool(settings["segmentation_cleanup_smooth_surfaces"]),
            smooth_sigma_um=float(settings["segmentation_cleanup_smooth_sigma_um"]),
            smooth_method=str(settings["segmentation_cleanup_smooth_method"]),
            smooth_morphological_radius_um=float(
                settings["segmentation_cleanup_smooth_morphological_radius_um"]
            ),
            remove_small_volumes=bool(settings["segmentation_cleanup_remove_small_volumes"]),
            remove_small_min_volume_um3=float(
                settings["segmentation_cleanup_remove_small_min_volume_um3"]
            ),
        )

    def _cleanup_trial(self, overrides: Mapping[str, Any]) -> np.ndarray:
        """Re-clean :attr:`original_raw_mask` (never a previous trial's own
        output -- matches how ``_preprocess_trial`` always re-runs from
        ``self.raw_skeleton``) with ``self.current`` plus *overrides*."""
        settings = {**self.current, **overrides}
        cleaned, _raw = preprocessing.clean_segmented_mask_for_skeletonisation(
            self.original_raw_mask,
            voxel_size_zyx=self.voxel_size_zyx,
            **self._cleanup_kwargs(settings),
        )
        return cleaned

    def _group_segmentation_cleanup(self) -> None:
        """Choose the seven segmentation-cleanup toggles and their own knobs,
        each scored on the resulting mask's own quality score -- these act
        directly on the mask, before any skeleton exists to measure instead.
        """
        group = "segmentation_cleanup"

        # Optional: when a raw reference image is supplied (see
        # `optimise_skeleton_and_graph_settings`'s own `raw_image`
        # parameter), guard every candidate against inventing foreground
        # the raw signal does not support -- `score_segmented_mask` alone
        # cannot see this, since it only ever looks at the mask's own
        # geometry (see `preprocessing.segmentation_raw_comparison`'s own
        # module docstring for why this is a genuinely different signal).
        baseline_added_fraction: Optional[float] = None
        if self.raw_image is not None:
            baseline_added_fraction = preprocessing.compare_segmentation_to_raw_image(
                self._cleanup_trial({}), self.raw_image, voxel_size_zyx=self.voxel_size_zyx
            ).added_fraction

        def quality(mask_trial: np.ndarray) -> float:
            score = -preprocessing.score_segmented_mask(
                mask_trial, voxel_size_zyx=self.voxel_size_zyx
            ).total
            if baseline_added_fraction is not None:
                added_fraction = preprocessing.compare_segmentation_to_raw_image(
                    mask_trial, self.raw_image, voxel_size_zyx=self.voxel_size_zyx
                ).added_fraction
                score += self._regression_penalty(
                    1.0 - added_fraction,
                    1.0 - baseline_added_fraction,
                    _MAX_RAW_ADDED_FRACTION_REGRESSION,
                )
            return score

        self._sweep(
            group, "segmentation_cleanup_fill_cavities", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_fill_cavities": value})),
        )

        self._sweep(
            group, "segmentation_cleanup_remove_whiskers", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_remove_whiskers": value})),
        )
        if self.current["segmentation_cleanup_remove_whiskers"]:
            self._sweep(
                group, "segmentation_cleanup_whisker_radius_um",
                cand.small_radius_candidates(
                    self.voxel_size_zyx, float(self.current["segmentation_cleanup_whisker_radius_um"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_whisker_radius_um": value})
                ),
            )

        self._sweep(
            group, "segmentation_cleanup_split_narrow_necks", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_split_narrow_necks": value})),
        )
        if self.current["segmentation_cleanup_split_narrow_necks"]:
            self._sweep(
                group, "segmentation_cleanup_split_min_marker_separation_um",
                cand.split_marker_separation_candidates(
                    self.typical_radius_um,
                    float(self.current["segmentation_cleanup_split_min_marker_separation_um"]),
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_split_min_marker_separation_um": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_split_min_pinch_radius_ratio",
                cand.split_pinch_radius_ratio_candidates(
                    float(self.current["segmentation_cleanup_split_min_pinch_radius_ratio"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_split_min_pinch_radius_ratio": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_split_min_body_radius_um",
                cand.split_min_body_radius_candidates(
                    self.typical_radius_um,
                    float(self.current["segmentation_cleanup_split_min_body_radius_um"]),
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_split_min_body_radius_um": value})
                ),
            )

        self._sweep(
            group, "segmentation_cleanup_close_gaps", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_close_gaps": value})),
        )
        if self.current["segmentation_cleanup_close_gaps"]:
            self._sweep(
                group, "segmentation_cleanup_close_gaps_radius_um",
                cand.small_radius_candidates(
                    self.voxel_size_zyx, float(self.current["segmentation_cleanup_close_gaps_radius_um"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_close_gaps_radius_um": value})
                ),
            )

        self._sweep(
            group, "segmentation_cleanup_reconnect_gaps", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_reconnect_gaps": value})),
        )
        if self.current["segmentation_cleanup_reconnect_gaps"]:
            gap_distances_um = cand.mask_component_gap_distances_um(
                self.original_raw_mask, self.voxel_size_zyx
            )
            self._sweep(
                group, "segmentation_cleanup_reconnect_max_bridge_distance_um",
                cand.reconnect_max_bridge_distance_candidates(
                    gap_distances_um,
                    float(self.current["segmentation_cleanup_reconnect_max_bridge_distance_um"]),
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_reconnect_max_bridge_distance_um": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_reconnect_min_cylindricality",
                cand.reconnect_min_cylindricality_candidates(
                    float(self.current["segmentation_cleanup_reconnect_min_cylindricality"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_reconnect_min_cylindricality": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_reconnect_max_axis_angle_degrees",
                cand.reconnect_max_axis_angle_candidates(
                    float(self.current["segmentation_cleanup_reconnect_max_axis_angle_degrees"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_reconnect_max_axis_angle_degrees": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_reconnect_min_facing_cosine",
                cand.reconnect_min_facing_cosine_candidates(
                    float(self.current["segmentation_cleanup_reconnect_min_facing_cosine"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_reconnect_min_facing_cosine": value})
                ),
            )
            self._sweep(
                group, "segmentation_cleanup_reconnect_max_radius_ratio",
                cand.reconnect_max_radius_ratio_candidates(
                    float(self.current["segmentation_cleanup_reconnect_max_radius_ratio"])
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_reconnect_max_radius_ratio": value})
                ),
            )

        self._sweep(
            group, "segmentation_cleanup_smooth_surfaces", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_smooth_surfaces": value})),
        )
        if self.current["segmentation_cleanup_smooth_surfaces"]:
            self._sweep(
                group, "segmentation_cleanup_smooth_method",
                cand.smooth_method_candidates(),
                lambda value: quality(self._cleanup_trial({"segmentation_cleanup_smooth_method": value})),
            )
            if self.current["segmentation_cleanup_smooth_method"] == "gaussian":
                self._sweep(
                    group, "segmentation_cleanup_smooth_sigma_um",
                    cand.small_radius_candidates(
                        self.voxel_size_zyx, float(self.current["segmentation_cleanup_smooth_sigma_um"])
                    ),
                    lambda value: quality(
                        self._cleanup_trial({"segmentation_cleanup_smooth_sigma_um": value})
                    ),
                )
            else:
                self._sweep(
                    group, "segmentation_cleanup_smooth_morphological_radius_um",
                    cand.small_radius_candidates(
                        self.voxel_size_zyx,
                        float(self.current["segmentation_cleanup_smooth_morphological_radius_um"]),
                    ),
                    lambda value: quality(
                        self._cleanup_trial({"segmentation_cleanup_smooth_morphological_radius_um": value})
                    ),
                )

        self._sweep(
            group, "segmentation_cleanup_remove_small_volumes", [False, True],
            lambda value: quality(self._cleanup_trial({"segmentation_cleanup_remove_small_volumes": value})),
        )
        if self.current["segmentation_cleanup_remove_small_volumes"]:
            self._sweep(
                group, "segmentation_cleanup_remove_small_min_volume_um3",
                cand.remove_small_min_volume_candidates(
                    self.original_raw_mask,
                    self.voxel_size_zyx,
                    float(self.current["segmentation_cleanup_remove_small_min_volume_um3"]),
                ),
                lambda value: quality(
                    self._cleanup_trial({"segmentation_cleanup_remove_small_min_volume_um3": value})
                ),
            )

        self.raw_mask = self._cleanup_trial({})
        # The typical-radius scale used by later groups' own candidate
        # generation (thick-vessel refinement, bundle scan size) should
        # reflect the mask cleanup actually decided on, not the pre-cleanup
        # input measured in `__init__`.
        radius_map = preprocessing.inscribed_radius_map(self.raw_mask, self.voxel_size_zyx)
        nonzero = radius_map[radius_map > 0]
        if nonzero.size:
            self.typical_radius_um = float(np.median(nonzero))

    # -- shared input-data-consistency guards -------------------------------------
    # Every helper below returns a "higher is better" fraction so
    # `_regression_penalty` has one uniform shape, and every one reuses an
    # existing, already-tested diagnostic that previously only ran as an
    # end-of-a-real-run warning (`pipeline/stages.py`) -- never used to
    # choose between candidates until now.
    def _vessels_represented_fraction(self, skeleton: np.ndarray) -> float:
        """Fraction of self.raw_mask's own real vessels (see
        preprocessing.diagnose_vessels_missing_from_skeleton) with at least
        one skeleton voxel anywhere in them -- catches a whole vessel
        dropped, which a blended coverage percentage can hide.

        Uses `_MIN_VESSEL_VOLUME_UM3_FOR_GUARD`, not
        `diagnose_vessels_missing_from_skeleton`'s own default
        ``min_vessel_voxels=2`` -- see that constant's own docstring for
        the real-data run that showed why the bare default is far too
        permissive here.
        """
        voxel_volume_um3 = (
            float(self.voxel_size_zyx[0]) * float(self.voxel_size_zyx[1]) * float(self.voxel_size_zyx[2])
        )
        min_vessel_voxels = max(
            2, int(round(_MIN_VESSEL_VOLUME_UM3_FOR_GUARD / max(1e-9, voxel_volume_um3)))
        )
        report = preprocessing.diagnose_vessels_missing_from_skeleton(
            skeleton, self.raw_mask, voxel_size_zyx=self.voxel_size_zyx,
            min_vessel_voxels=min_vessel_voxels,
        )
        return float(report["explained_vessel_fraction"])

    def _skeleton_mask_coverage_fraction(self, skeleton: np.ndarray) -> float:
        """Fraction of self.raw_mask's own volume the skeleton still runs
        through. See preprocessing.diagnose_skeleton_mask_consistency."""
        report = preprocessing.diagnose_skeleton_mask_consistency(
            skeleton, self.raw_mask, voxel_size_zyx=self.voxel_size_zyx
        )
        return float(report["coverage_fraction"])

    def _skeleton_graph_coverage_fraction(self, G) -> float:
        """Fraction of self.current_skeleton the graph's own rasterised
        edges still trace. See graph.diagnose_skeleton_graph_consistency --
        the cheapest of the three (no EDT, no labelling), so used across
        every graph-level sweep below."""
        report = graph_mod.diagnose_skeleton_graph_consistency(
            G, self.current_skeleton, voxel_size_zyx=self.voxel_size_zyx
        )
        return float(report["coverage_fraction"])

    @staticmethod
    def _regression_penalty(current: float, baseline: float, tolerance: float) -> float:
        """`_GUARD_PENALTY` when *current* has fallen more than *tolerance*
        below *baseline*, else 0 -- purely additive: a candidate that does
        not regress pays nothing and is scored exactly as it was before
        this guard existed."""
        return _GUARD_PENALTY if (baseline - current) > tolerance else 0.0

    # -- group 1: thick-vessel gating ---------------------------------------------
    def _group_thick_vessel_gating(self) -> None:
        group = "thick_vessel_gating"
        if not cand.thick_vessel_worth_checking(self.raw_mask, self.voxel_size_zyx):
            self.current["use_thick_vessel_skeletonisation"] = False
            self._record(group, "use_thick_vessel_skeletonisation", False, 0.0, note="skipped: nothing fat enough")
            self.raw_skeleton = preprocessing.skeletonize_volume(self.raw_mask)
            return

        def cost_for(skeleton: np.ndarray, on: bool) -> float:
            stats = preprocessing.compute_skeleton_connectivity_stats(skeleton, self._connectivity())
            # A tiny bias toward "off" so a marginal gain does not turn on a
            # second algorithm for its own sake (Occam's razor).
            return -stats.largest_fraction + (1e-3 if on else 0.0)

        def trial_on_off(value: bool) -> float:
            skeleton = _raw_skeleton(self.raw_mask, {**self.current, "use_thick_vessel_skeletonisation": value}, self.voxel_size_zyx)
            return cost_for(skeleton, value)

        self._sweep(group, "use_thick_vessel_skeletonisation", [False, True], trial_on_off)

        if not self.current["use_thick_vessel_skeletonisation"]:
            self.raw_skeleton = preprocessing.skeletonize_volume(self.raw_mask)
            return

        def trial_with(overrides: Mapping[str, Any]) -> np.ndarray:
            kwargs = _thick_vessel_kwargs({**self.current, **overrides}, self.voxel_size_zyx)
            return preprocessing.skeletonize_thickness_gated(self.raw_mask, **kwargs)

        radius_candidates = cand.thick_vessel_min_radius_candidates(
            self.raw_mask, self.voxel_size_zyx, self.current["skeleton_thick_vessel_min_radius_um"]
        )
        self._sweep(
            group, "skeleton_thick_vessel_min_radius_um", radius_candidates,
            lambda value: cost_for(trial_with({"skeleton_thick_vessel_min_radius_um": value}), True),
        )

        self._sweep(
            group, "skeleton_fill_mask_holes_before_thickness", [True, False],
            lambda value: cost_for(trial_with({"skeleton_fill_mask_holes_before_thickness": value}), True),
        )

        # The five refinement settings below (wall absorption, flake filter,
        # the two bridge caps, bridge-radius smoothing) exist specifically to
        # stop the fat region's centreline coming out as a medial *sheet*
        # rather than one line, so they are scored on braid factor -- the
        # measure this codebase already uses for exactly that failure mode --
        # rather than on connectivity alone.
        baseline_skeleton_for_refinement = trial_with({})
        baseline_vessels_represented = self._vessels_represented_fraction(
            baseline_skeleton_for_refinement
        )

        def cost_braid(skeleton: np.ndarray) -> float:
            stats = preprocessing.compute_skeleton_connectivity_stats(skeleton, self._connectivity())
            penalty = self._regression_penalty(
                self._vessels_represented_fraction(skeleton),
                baseline_vessels_represented,
                _MAX_MISSING_VESSEL_FRACTION_REGRESSION,
            )
            return met.braid_factor_along_long_axis(skeleton) + (1.0 - stats.largest_fraction) + penalty

        typical_thick_radius_um = cand.typical_thick_vessel_radius_um(
            self.raw_mask, self.voxel_size_zyx, float(self.current["skeleton_thick_vessel_min_radius_um"])
        )

        self._sweep(
            group, "skeleton_thick_vessel_wall_absorption_um",
            cand.thick_vessel_wall_absorption_candidates(typical_thick_radius_um),
            lambda value: cost_braid(trial_with({"skeleton_thick_vessel_wall_absorption_um": value})),
        )
        self._sweep(
            group, "skeleton_thick_vessel_flake_filter_um",
            cand.thick_vessel_flake_filter_candidates(self.voxel_size_zyx),
            lambda value: cost_braid(trial_with({"skeleton_thick_vessel_flake_filter_um": value})),
        )
        self._sweep(
            group, "skeleton_thick_vessel_max_bridge_radius_multiple",
            cand.thick_vessel_max_bridge_radius_multiple_candidates(
                float(self.current["skeleton_thick_vessel_max_bridge_radius_multiple"])
            ),
            lambda value: cost_braid(trial_with({"skeleton_thick_vessel_max_bridge_radius_multiple": value})),
        )
        # The raw mask's own connected-component gaps stand in for "how far
        # apart are the fragments a bridge might need to reach" -- the real
        # skeleton does not exist yet at this point in the very first group.
        gap_distances_um = preprocessing.inter_component_gap_distances(
            self.raw_mask, self._connectivity(), self.voxel_size_zyx
        )
        self._sweep(
            group, "skeleton_thick_vessel_max_bridge_distance_um",
            cand.thick_vessel_max_bridge_distance_candidates(gap_distances_um),
            lambda value: cost_braid(trial_with({"skeleton_thick_vessel_max_bridge_distance_um": value})),
        )
        self._sweep(
            group, "skeleton_thick_vessel_bridge_radius_smoothing_um",
            cand.thick_vessel_bridge_radius_smoothing_candidates(typical_thick_radius_um),
            lambda value: cost_braid(trial_with({"skeleton_thick_vessel_bridge_radius_smoothing_um": value})),
        )

        self.raw_skeleton = _raw_skeleton(self.raw_mask, self.current, self.voxel_size_zyx)

    # -- group 2: minimum branch length --------------------------------------------
    def _group_min_branch_length(self) -> None:
        group = "min_branch_length"
        raw_stats = preprocessing.compute_skeleton_connectivity_stats(self.raw_skeleton, self._connectivity())
        candidate_values = cand.min_branch_length_candidates(
            raw_stats.component_sizes, int(self.current["skeleton_min_branch_length"])
        )
        # Baseline through the *whole* `_preprocess_trial` pipeline at the
        # settings entering this group (closing/bridging/bundle knobs still
        # at their own not-yet-optimised current values) -- not the bare
        # pre-pipeline `self.raw_skeleton`, which every candidate below is
        # NOT compared against (every candidate goes through the full
        # pipeline too). This fixes a real, pre-existing bug the
        # voxels-removed guard already had, not just the new
        # vessels-missing guard: a real, noisy dataset showed the full
        # pipeline's own closing/bridging/bundle-refinement steps (at their
        # not-yet-tuned defaults) discarding ~34% of the raw skeleton's
        # voxels even at min_branch_length=0, which alone already exceeded
        # `_MAX_VOXELS_REMOVED_FRACTION` regardless of this sweep's own
        # candidate value -- every real candidate was rejected, no matter
        # how little it actually pruned (see project memory).
        baseline_cleaned = self._preprocess_trial({})
        baseline_voxel_count = preprocessing.compute_skeleton_connectivity_stats(
            baseline_cleaned, self._connectivity()
        ).voxel_count
        baseline_vessels_represented = self._vessels_represented_fraction(baseline_cleaned)

        def cost(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_min_branch_length": value})
            stats = preprocessing.compute_skeleton_connectivity_stats(cleaned, self._connectivity())
            removed_fraction = (
                1.0 - stats.voxel_count / baseline_voxel_count if baseline_voxel_count else 0.0
            )
            penalty = _GUARD_PENALTY if removed_fraction > _MAX_VOXELS_REMOVED_FRACTION else 0.0
            penalty += self._regression_penalty(
                self._vessels_represented_fraction(cleaned),
                baseline_vessels_represented,
                _MAX_MISSING_VESSEL_FRACTION_REGRESSION,
            )
            return -stats.largest_fraction + penalty

        self._sweep(group, "skeleton_min_branch_length", candidate_values, cost)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 3: bundle refinement -------------------------------------------------
    def _group_bundle_refinement(self) -> None:
        group = "bundle_refinement"
        baseline_components = preprocessing.compute_skeleton_connectivity_stats(
            self.current_skeleton, self._connectivity()
        ).n_components
        baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)

        def leftover_density(cleaned: np.ndarray, scan_size: int, density_fraction: float) -> float:
            from scipy.ndimage import uniform_filter

            density_after = uniform_filter(cleaned.astype(float), size=scan_size)
            return float((density_after >= density_fraction).sum())

        def guard(cleaned: np.ndarray) -> float:
            n_components = preprocessing.compute_skeleton_connectivity_stats(
                cleaned, self._connectivity()
            ).n_components
            penalty = _GUARD_PENALTY if n_components > baseline_components else 0.0
            penalty += self._regression_penalty(
                self._skeleton_mask_coverage_fraction(cleaned),
                baseline_coverage,
                _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            return penalty

        scan_candidates = cand.bundle_scan_size_candidates(
            self.raw_mask, self.voxel_size_zyx, int(self.current["skeleton_bundle_scan_size"])
        )

        def cost_scan(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_bundle_scan_size": value})
            density_fraction = float(self.current["skeleton_bundle_density_fraction"])
            return leftover_density(cleaned, value, density_fraction) + guard(cleaned)

        self._sweep(group, "skeleton_bundle_scan_size", scan_candidates, cost_scan)

        density_candidates = cand.bundle_density_fraction_candidates(
            self.raw_mask, int(self.current["skeleton_bundle_scan_size"]),
            float(self.current["skeleton_bundle_density_fraction"]),
        )
        scan_size = int(self.current["skeleton_bundle_scan_size"])

        def cost_density(value: float) -> float:
            cleaned = self._preprocess_trial({"skeleton_bundle_density_fraction": value})
            return leftover_density(cleaned, scan_size, value) + guard(cleaned)

        self._sweep(group, "skeleton_bundle_density_fraction", density_candidates, cost_density)

        self.current["skeleton_bundle_max_connections_per_hub"] = cand.estimate_bundle_max_connections(
            self.raw_skeleton
        )
        self.current["skeleton_bundle_hub_min_spacing"] = cand.bundle_hub_min_spacing_for(
            int(self.current["skeleton_bundle_scan_size"])
        )
        self._record(
            group, "skeleton_bundle_max_connections_per_hub",
            self.current["skeleton_bundle_max_connections_per_hub"], 0.0,
            note="derived from observed skeleton branching, not swept",
        )
        self._record(
            group, "skeleton_bundle_hub_min_spacing",
            self.current["skeleton_bundle_hub_min_spacing"], 0.0,
            note="derived as half the scan window, not swept",
        )
        self.current_skeleton = self._preprocess_trial({})

    # -- group 4: closing radius -----------------------------------------------------
    def _gap_distances_voxels(self) -> np.ndarray:
        return preprocessing.inter_component_gap_distances(self.current_skeleton, self._connectivity())

    def _fusion_cost(self, cleaned: np.ndarray) -> float:
        signal = met.gap_vs_fusion_signal(
            self.current_skeleton, cleaned, voxel_size_zyx=self.voxel_size_zyx,
            component_connectivity=self._connectivity(), typical_radius_um=self.typical_radius_um,
        )
        penalty = _GUARD_PENALTY if signal.largest_gap_bridged_ratio > _GAP_FUSION_RATIO_GUARD else 0.0
        penalty += self._regression_penalty(
            self._skeleton_mask_coverage_fraction(cleaned),
            self._fusion_baseline_coverage,
            _MAX_COVERAGE_FRACTION_REGRESSION,
        )
        return -float(signal.components_merged) + penalty

    def _group_closing_radius(self) -> None:
        group = "closing_radius"
        self._fusion_baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)
        candidate_values = cand.closing_radius_candidates(
            self._gap_distances_voxels(), int(self.current["skeleton_closing_radius"])
        )

        def cost(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_closing_radius": value})
            return self._fusion_cost(cleaned)

        self._sweep(group, "skeleton_closing_radius", candidate_values, cost)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 5: gap bridging --------------------------------------------------------
    def _group_gap_bridging(self) -> None:
        group = "gap_bridging"
        self._fusion_baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)
        gaps = self._gap_distances_voxels()
        bridge_candidates = cand.bridge_gap_size_candidates(gaps, int(self.current["skeleton_bridge_gap_size"]))

        def cost_bridge(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_bridge_gap_size": value})
            return self._fusion_cost(cleaned)

        self._sweep(group, "skeleton_bridge_gap_size", bridge_candidates, cost_bridge)
        self.current_skeleton = self._preprocess_trial({})

        gaps = self._gap_distances_voxels()
        max_distance_candidates = cand.max_bridge_distance_candidates(
            gaps, int(self.current["skeleton_max_bridge_distance"])
        )

        def cost_max_distance(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_max_bridge_distance": value})
            return self._fusion_cost(cleaned)

        self._sweep(group, "skeleton_max_bridge_distance", max_distance_candidates, cost_max_distance)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 6: component connectivity + minimum component percent -----------------
    def _group_connectivity_and_component_filter(self) -> None:
        group = "connectivity_and_component_filter"
        baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)

        def cost_connectivity(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_component_connectivity": value})
            stats = preprocessing.compute_skeleton_connectivity_stats(cleaned, value)
            penalty = self._regression_penalty(
                self._skeleton_mask_coverage_fraction(cleaned),
                baseline_coverage,
                _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            return -stats.largest_fraction + penalty

        self._sweep(group, "skeleton_component_connectivity", cand.component_connectivity_candidates(), cost_connectivity)
        self.current_skeleton = self._preprocess_trial({})

        connectivity = self._connectivity()
        before = preprocessing.compute_skeleton_connectivity_stats(self.current_skeleton, connectivity)
        percent_candidates = cand.min_component_percent_candidates(
            before.component_sizes, before.voxel_count, float(self.current["skeleton_min_component_percent"])
        )
        # A genuinely bilateral vessel tree (a second component within 20% of
        # the largest) must not be discarded as noise.
        protect_second_component = (
            len(before.component_sizes) >= 2
            and before.component_sizes[1] >= 0.8 * before.component_sizes[0]
        )

        def cost_percent(value: float) -> float:
            # No coverage-regression guard here, deliberately: discarding a
            # genuinely small, noise-scale component is this sweep's own
            # purpose, and necessarily costs some raw-mask coverage by
            # design -- `protect_second_component` below is the targeted
            # guard against the one real risk (a genuine second vessel tree
            # mistaken for noise), not a blanket coverage floor that would
            # fight the setting's own intent.
            cleaned = self._preprocess_trial({"skeleton_min_component_percent": value})
            stats = preprocessing.compute_skeleton_connectivity_stats(cleaned, connectivity)
            penalty = 0.0
            if protect_second_component:
                min_size_removed = value / 100.0 * before.voxel_count
                if min_size_removed > before.component_sizes[1]:
                    penalty = _GUARD_PENALTY
            return -stats.largest_fraction + penalty

        self._sweep(group, "skeleton_min_component_percent", percent_candidates, cost_percent)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 7: reconnect thresholds -------------------------------------------------
    def _build_graph(self, overrides: Mapping[str, Any]):
        settings = {**self.current, **overrides}
        return graph_mod.build_graph_from_skeleton(
            self.current_skeleton,
            voxel_size=self.voxel_size_zyx,
            graph_reconnect_threshold=float(settings["graph_reconnect_threshold"]),
            final_orphan_reconnect_threshold=float(settings["final_orphan_reconnect_threshold"]),
            cluster_collapse_distance=float(settings["cluster_collapse_distance"]),
            min_stub_length=float(settings["min_stub_length"]),
            cluster_collapse_method=str(settings["cluster_collapse_method"]),
            cluster_collapse_max_radial_dispersion=float(settings["cluster_collapse_max_radial_dispersion"]),
            cluster_collapse_persistence_search_multiple=float(
                settings["cluster_collapse_persistence_search_multiple"]
            ),
        )

    def _group_reconnect_thresholds(self) -> None:
        group = "reconnect_thresholds"
        baseline_graph = self._build_graph(
            {"graph_reconnect_threshold": 0.0, "final_orphan_reconnect_threshold": 0.0, "min_stub_length": 0.0}
        )
        gap_um = cand.graph_component_gap_distances_um(baseline_graph)
        reconnect_candidates = cand.reconnect_threshold_candidates(
            gap_um, float(self.current["graph_reconnect_threshold"])
        )
        # The graph built from the settings as they stand entering this
        # group -- not `baseline_graph` above, which is a zero-threshold
        # probe purely for gap-distance candidate generation -- is the
        # right reference for "did a threshold choice reduce how much of
        # the skeleton the graph still traces".
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self._build_graph({}))

        def cost_reconnect(value: float) -> float:
            G = self._build_graph({"graph_reconnect_threshold": value})
            penalty = self._regression_penalty(
                self._skeleton_graph_coverage_fraction(G),
                baseline_graph_coverage,
                _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            return met.graph_topology_metrics(G).score + penalty

        self._sweep(group, "graph_reconnect_threshold", reconnect_candidates, cost_reconnect)

        orphan_candidates = cand.orphan_threshold_candidates(
            float(self.current["graph_reconnect_threshold"]), float(self.current["final_orphan_reconnect_threshold"])
        )

        def cost_orphan(value: float) -> float:
            G = self._build_graph({"final_orphan_reconnect_threshold": value})
            penalty = self._regression_penalty(
                self._skeleton_graph_coverage_fraction(G),
                baseline_graph_coverage,
                _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            return met.graph_topology_metrics(G).score + penalty

        self._sweep(group, "final_orphan_reconnect_threshold", orphan_candidates, cost_orphan)

        self.current_graph = self._build_graph({})
        if self.current_graph.number_of_nodes() == 0:
            raise RuntimeError(
                "No graph could be built from the optimised skeleton at any "
                "candidate reconnect threshold; cannot continue optimising "
                "graph settings."
            )

    # -- group 8: cluster collapse (distance, method, and the method's own knob) -------
    def _group_cluster_collapse(self) -> None:
        group = "cluster_collapse_distance"
        baseline = met.graph_topology_metrics(self.current_graph)
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self.current_graph)

        def cost_for(key: str, value: Any) -> float:
            G = self._build_graph({key: value})
            metrics = met.graph_topology_metrics(G)
            fragmentation_penalty = _GUARD_PENALTY * max(0, metrics.n_components - baseline.n_components)
            fragmentation_penalty += self._regression_penalty(
                self._skeleton_graph_coverage_fraction(G),
                baseline_graph_coverage,
                _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            return float(metrics.total_degree2) + fragmentation_penalty

        node_gaps = cand.nearest_neighbour_node_distances_um(self.current_graph)
        candidate_values = cand.cluster_collapse_distance_candidates(
            node_gaps, float(self.current["cluster_collapse_distance"])
        )
        self._sweep(
            group, "cluster_collapse_distance", candidate_values,
            lambda value: cost_for("cluster_collapse_distance", value),
        )
        self.current_graph = self._build_graph({})

        self._sweep(
            group, "cluster_collapse_method", cand.cluster_collapse_method_candidates(),
            lambda value: cost_for("cluster_collapse_method", value),
        )
        self.current_graph = self._build_graph({})

        # The other two settings are each read by exactly one method, so only
        # the one the search just chose is worth a sweep of its own.
        method = self.current["cluster_collapse_method"]
        if method == "direction_aware":
            self._sweep(
                group, "cluster_collapse_max_radial_dispersion",
                cand.cluster_collapse_max_radial_dispersion_candidates(
                    float(self.current["cluster_collapse_max_radial_dispersion"])
                ),
                lambda value: cost_for("cluster_collapse_max_radial_dispersion", value),
            )
            self.current_graph = self._build_graph({})
        elif method == "persistence":
            self._sweep(
                group, "cluster_collapse_persistence_search_multiple",
                cand.cluster_collapse_persistence_search_multiple_candidates(
                    float(self.current["cluster_collapse_persistence_search_multiple"])
                ),
                lambda value: cost_for("cluster_collapse_persistence_search_multiple", value),
            )
            self.current_graph = self._build_graph({})

    # -- group 9: minimum stub length -----------------------------------------------
    def _group_min_stub_length(self) -> None:
        # No `_skeleton_graph_coverage_fraction` guard here, deliberately --
        # unlike reconnect_thresholds/cluster_collapse_distance, pruning a
        # terminal stub is this sweep's own purpose and necessarily reduces
        # how much of the skeleton the graph still traces; a coverage-
        # regression guard would fight the setting's own intent exactly the
        # way it would for skeleton_min_component_percent (see that sweep's
        # own comment). `_MAX_GRAPH_LENGTH_REMOVED_FRACTION` below is the
        # existing, purpose-built guard against removing too much.
        group = "min_stub_length"
        terminal_lengths = cand.terminal_edge_lengths_um(self.current_graph)
        candidate_values = cand.min_stub_length_candidates(
            terminal_lengths, float(self.current["min_stub_length"])
        )
        unpruned = self._build_graph({"min_stub_length": 0.0})
        baseline_nodes = unpruned.number_of_nodes()
        baseline_length = met.total_edge_length(unpruned)

        def cost(value: float) -> float:
            G = self._build_graph({"min_stub_length": value})
            stubs_removed = max(0, baseline_nodes - G.number_of_nodes())
            length_removed = max(0.0, baseline_length - met.total_edge_length(G))
            if baseline_length > 0 and length_removed / baseline_length > _MAX_GRAPH_LENGTH_REMOVED_FRACTION:
                return _GUARD_PENALTY
            if length_removed <= 0:
                return 0.0
            return -(stubs_removed / length_removed)

        self._sweep(group, "min_stub_length", candidate_values, cost)
        self.current_graph = self._build_graph({})

    # -- group 10: centreline smoothing ------------------------------------------------
    def _group_centreline_smoothing(self) -> None:
        group = "centreline_smoothing"
        if not bool(self.current.get("smooth_centrelines", True)):
            return
        self._emit(GROUP_STARTED, group)
        methods = cand.centreline_smoothing_method_candidates()
        deviations = cand.centreline_max_deviation_candidates(
            self.voxel_size_zyx, float(self.current["centreline_max_deviation"])
        )
        combos = [
            (m, it, dev)
            for m in methods
            for it in cand.centreline_smoothing_iterations_candidates(m)
            for dev in deviations
        ]
        length_before = met.total_edge_length(self.current_graph)

        best_score = float("inf")
        best_combo = (
            self.current["centreline_smoothing_method"],
            int(self.current["centreline_smoothing_iterations"]),
            float(self.current["centreline_max_deviation"]),
        )
        for index, (method, iterations, deviation) in enumerate(combos):
            note = ""
            try:
                trial_graph = self.current_graph.copy()
                counts = graph_mod.smooth_graph_centrelines(
                    trial_graph, self.current_skeleton, voxel_size_zyx=self.voxel_size_zyx,
                    method=method, iterations=iterations, max_deviation=deviation,
                )
                quality = met.smoothing_quality(counts, length_before, met.total_edge_length(trial_graph))
                penalty = _GUARD_PENALTY if quality.length_shrink_fraction > _MAX_LENGTH_SHRINK_FRACTION else 0.0
                score = -quality.smoothed_fraction + penalty
            except Exception as error:  # noqa: BLE001
                score = float("inf")
                note = f"failed: {error}"
            self._record(group, "centreline_smoothing", (method, iterations, deviation), score, note)
            self._emit(CANDIDATE_EVALUATED, group, candidate_index=index, candidate_total=len(combos))
            if score < best_score:
                best_score, best_combo = score, (method, iterations, deviation)

        (
            self.current["centreline_smoothing_method"],
            self.current["centreline_smoothing_iterations"],
            self.current["centreline_max_deviation"],
        ) = best_combo
        self._emit(
            GROUP_FINISHED, group,
            winner={
                "centreline_smoothing_method": best_combo[0],
                "centreline_smoothing_iterations": best_combo[1],
                "centreline_max_deviation": best_combo[2],
            },
        )
        self._group_index += 1

    # -- orchestration ------------------------------------------------------------
    def run(self) -> None:
        if not self.raw_mask.any():
            self.raw_skeleton = self.raw_mask.copy()
            self.current_skeleton = self.raw_skeleton
            return

        if self._group_enabled("segmentation_cleanup"):
            self._group_segmentation_cleanup()
            if not self.raw_mask.any():
                self.raw_skeleton = self.raw_mask.copy()
                self.current_skeleton = self.raw_skeleton
                return

        if self._group_enabled("thick_vessel_gating"):
            self._group_thick_vessel_gating()
        else:
            self.raw_skeleton = _raw_skeleton(self.raw_mask, self.current, self.voxel_size_zyx)

        for name, method in (
            ("min_branch_length", self._group_min_branch_length),
            ("bundle_refinement", self._group_bundle_refinement),
            ("closing_radius", self._group_closing_radius),
            ("gap_bridging", self._group_gap_bridging),
            ("connectivity_and_component_filter", self._group_connectivity_and_component_filter),
        ):
            if self._group_enabled(name):
                method()
        # Whichever of the skeleton-cleaning groups ran -- or none did, every
        # one deselected -- this makes sure `current_skeleton` reflects every
        # decision actually in `self.current`, cheaply (one more real call).
        self.current_skeleton = self._preprocess_trial({})

        if not self.current_skeleton.any():
            return

        if self._group_enabled("reconnect_thresholds"):
            self._group_reconnect_thresholds()
        else:
            self.current_graph = self._build_graph({})
            if self.current_graph.number_of_nodes() == 0:
                raise RuntimeError(
                    "No graph could be built from the optimised skeleton with "
                    "the current graph settings; cannot continue optimising "
                    "graph settings."
                )

        for name, method in (
            ("cluster_collapse_distance", self._group_cluster_collapse),
            ("min_stub_length", self._group_min_stub_length),
            ("centreline_smoothing", self._group_centreline_smoothing),
        ):
            if self._group_enabled(name):
                method()


def optimise_skeleton_and_graph_settings(
    raw_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    progress: Optional[ProgressCallback] = None,
    downsample_factor: Optional[int] = None,
    groups: Optional[Iterable[str]] = None,
    raw_image: Optional[np.ndarray] = None,
) -> OptimisationResult:
    """Empirically choose every Skeletonise/Graph tab setting for *raw_mask*.

    *starting_values* must have an entry for every name in
    :data:`OPTIMISE_SETTING_NAMES` -- the settings this run does not manage to
    improve on simply keep their starting value. See the module docstring for
    the search strategy.

    *downsample_factor* trades accuracy for speed on a large volume: ``None``
    (the default) auto-detects from voxel count via
    :func:`resolve_auto_downsample_factor`, ``1`` runs at full resolution, and
    ``2``/``4``/``8``/``16`` (:data:`DOWNSAMPLE_FACTORS`) run the whole search
    on a block-max-reduced copy of *raw_mask* with *voxel_size_xyz* scaled up
    to match. Micron-based settings come back unaffected by this (the scaled
    voxel size already accounts for it); the handful of settings measured in
    voxels (:data:`_VOXEL_SCALED_SETTING_NAMES`) are multiplied back up by the
    factor before being returned, so they are correct against the
    full-resolution volume a real run actually skeletonises.

    *groups* restricts the search to some of :data:`GROUP_NAMES` -- ``None``
    (the default) runs all ten; any other setting simply keeps its starting
    value, exactly as if every one of its candidates had failed.

    *raw_image* is an optional raw (unsegmented) reference image, same shape
    as *raw_mask*, same physical dataset -- when given, the
    ``segmentation_cleanup`` group additionally guards its candidates against
    the raw signal (see :func:`haemolynx.preprocessing.compare_segmentation_to_raw_image`);
    ``None`` (the default) is today's exact behaviour. Downsampled the same
    way *raw_mask* is (block-mean, not block-max -- an intensity value
    averages over its block rather than taking the brightest voxel in it).
    """
    raw_mask = np.asarray(raw_mask, dtype=bool)
    voxel_size_xyz = tuple(float(v) for v in voxel_size_xyz)
    factor = int(downsample_factor) if downsample_factor else resolve_auto_downsample_factor(raw_mask.shape)
    factor = factor if factor in DOWNSAMPLE_FACTORS else 1

    if factor > 1:
        search_mask = _downsample_mask(raw_mask, factor)
        search_voxel_size_xyz = tuple(v * factor for v in voxel_size_xyz)
        search_raw_image = (
            _downsample_intensity(np.asarray(raw_image), factor) if raw_image is not None else None
        )
    else:
        search_mask = raw_mask
        search_voxel_size_xyz = voxel_size_xyz
        search_raw_image = raw_image

    search = _Search(
        search_mask, search_voxel_size_xyz, starting_values, progress,
        enabled_groups=groups, raw_image=search_raw_image,
    )
    search.run()
    settings = {name: search.current[name] for name in OPTIMISE_SETTING_NAMES if name in search.current}
    if factor > 1:
        for name in _VOXEL_SCALED_SETTING_NAMES:
            if settings.get(name) is not None:
                settings[name] = int(round(settings[name] * factor))
    return OptimisationResult(
        settings=settings,
        trials=tuple(search.trials),
        downsample_factor=factor,
        groups_run=tuple(dict.fromkeys(search.groups_run)),
    )
