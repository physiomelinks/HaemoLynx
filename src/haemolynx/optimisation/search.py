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
One exception: ``preprocess_skeleton_for_graph`` applies min-branch-length
*after* bridging (a short piece of a broken vessel is joined before it can be
filtered), but it is still swept first here -- every sweep runs the whole
real call, so the order only decides which setting is fixed first.
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

A sweep keeps a setting's current value unless another candidate scores
lower by more than :data:`SWEEP_MIN_IMPROVEMENT` -- a tie, or a difference
too small to mean anything, leaves it where the user had it. Trials a search
has just run (a sweep's baseline is its current value's own trial; the
skeleton or graph it leaves behind is its winner's) come back from a
:class:`~haemolynx.optimisation.trial_cache.TrialCache` instead of running
again.

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

import math
import numbers
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from haemolynx import graph as graph_mod
from haemolynx import preprocessing
from haemolynx.graph import cartwheel_guard
from haemolynx.preprocessing.bridge_mask_support import (
    DEFAULT_MAX_BACKGROUND_GAP_UM,
    DEFAULT_MIN_MASK_FRACTION,
)

from . import candidates as cand
from . import metrics as met
from .progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
    OptimisationEvent,
    ProgressCallback,
)
from .scorecard import ScoreRow
from .trial_cache import TrialCache

#: Settings this search decides, on the Input tab's segmentation-cleanup
#: block (the raw mask, before skeletonisation) -- in the same fixed order
#: `clean_segmented_mask_for_skeletonisation` itself applies them.
SEGMENTATION_CLEANUP_SETTING_NAMES: tuple[str, ...] = (
    "segmentation_cleanup",
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
    "skeleton_bridge_weight_by_segmentation",
    "skeleton_bridge_z_distance_weight",
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
    "min_stub_length_radius_multiple",
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
#:
#: Also deliberately not swept: skeleton_bridge_weight_by_segmentation and
#: skeleton_bridge_z_distance_weight. Both change *how* a bridge within
#: skeleton_max_bridge_distance is drawn (mask-hugging vs. straight line;
#: how much a z-gap counts against xy), not whether the resulting skeleton
#: scores better on any metric this search evaluates -- they are safety/bias
#: knobs the user sets deliberately, so they pass through untouched like the
#: guard settings above.

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


def axis_downsample_factors(
    factor: int, voxel_size_zyx: Optional[Sequence[float]] = None
) -> tuple[int, ...]:
    """How much each axis is reduced by for an offered *factor*.

    *factor* applies to the finest axes; a coarser one is reduced by only as
    much as brings its voxels nearest the same size (at least 1), so the
    search grid is as close to isotropic as the factors allow. On a
    0.5 x 0.5 x 2 um stack, factor 4 gives 2 um voxels on every axis, where
    reducing each axis by 4 gave 2 x 2 x 8 um -- a vessel's z extent four
    times coarser than the resolution the search judged it at in-plane.
    Without *voxel_size_zyx*, every axis by *factor*.
    """
    factor = max(1, int(factor))
    if voxel_size_zyx is None:
        return (factor, factor, factor)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    finest = float(spacing.min())
    return tuple(max(1, int(round(factor * finest / float(s)))) for s in spacing)


def _voxel_reduction(factor: int, voxel_size_zyx: Optional[Sequence[float]]) -> int:
    """How many voxels become one at *factor*: the product of the axes'."""
    return int(np.prod(axis_downsample_factors(factor, voxel_size_zyx)))


def resolve_auto_downsample_factor(
    shape: tuple[int, ...], voxel_size_zyx: Optional[Sequence[float]] = None
) -> int:
    """The smallest offered factor bringing *shape* under the search's own target size."""
    total = 1
    for dim in shape:
        total *= int(dim)
    for factor in DOWNSAMPLE_FACTORS:
        if total / _voxel_reduction(factor, voxel_size_zyx) <= AUTO_DOWNSAMPLE_TARGET_VOXELS:
            return factor
    return DOWNSAMPLE_FACTORS[-1]


def _downsample_mask(
    mask: np.ndarray, factor: int, voxel_size_zyx: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Block-max reduction: a downsampled voxel is foreground if any voxel in
    its block was. Plain striding could skip clean over a thin vessel that
    happens to fall between the sampled points; this cannot lose one. Blocks
    are :func:`axis_downsample_factors` of *factor*."""
    blocks = axis_downsample_factors(factor, voxel_size_zyx)
    if max(blocks) <= 1:
        return mask
    from skimage.measure import block_reduce

    return block_reduce(mask, block_size=blocks, func=np.max)


def _downsample_intensity(
    image: np.ndarray, factor: int, voxel_size_zyx: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Block-mean reduction for a raw intensity image -- unlike the mask's
    own block-*max* (foreground survives if any voxel in the block was),
    an intensity value should average over its block, or Otsu thresholding
    the downsampled copy would systematically read brighter than the real
    volume and bias every added/removed-voxel comparison. The same blocks as
    :func:`_downsample_mask`."""
    blocks = axis_downsample_factors(factor, voxel_size_zyx)
    if max(blocks) <= 1:
        return image
    from skimage.measure import block_reduce

    return block_reduce(image, block_size=blocks, func=np.mean)


#: "Auto" downsampling's own time budget, in seconds, when a caller asks it
#: to target a runtime instead of (or as well as) a voxel count -- see
#: :func:`estimate_downsample_factor_for_time_budget`.
DEFAULT_AUTO_DOWNSAMPLE_TARGET_SECONDS = 300.0

#: "Auto" never searches a grid so coarse that a typical vessel's radius (see
#: :func:`_typical_vessel_radius_um`) is fewer than this many of its voxels --
#: a typical vessel at least three voxels across -- whatever the time budget
#: allows. On a 0.98 um E14.5 stack with a typical radius of 3.7 um that is
#: 2x (1.9 voxels), not 4x (0.9); the time budget alone had chosen 16x, which
#: put every capillary in a single ~16 um voxel.
AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS = 1.5

#: "Auto" measures the typical radius on the densest crop of the
#: full-resolution mask holding at most this many voxels, not the whole
#: stack: the distance transform took 38 s on a 75M-voxel one.
_AUTO_RADIUS_CROP_VOXELS = 8_000_000


#: How many times a search runs each of its costly steps, counted on the
#: E14.5 MCA stack (287 x 512 x 512): real searches at 16x and 4x made within
#: two of the same calls of each, the scorecard's own two runs not counted.
#: A mask whose thick vessels win (use_thick_vessel_skeletonisation chosen)
#: runs the thick skeleton about fifteen more times, so its estimate reads low.
_SEARCH_STEP_CALLS: dict[str, int] = {
    "cleanup": 15,
    "mask score": 24,
    "plain skeleton": 2,
    "thick skeleton": 2,
    "preprocess": 22,
    "graph": 19,
    "smooth": 40,
}


@dataclass(frozen=True)
class _SearchTimeModel:
    """A whole search's time: each costly step timed once on one search grid
    of *probe_voxels* voxels, times how often a search runs it
    (:data:`_SEARCH_STEP_CALLS`) -- and, for any other grid, scaled with its
    voxel count.

    Timed on the grid itself, because a coarser one does not predict it.
    What this replaced timed one whole cleanup-skeletonise-build-graph cycle
    at 16x, scaled it with voxel count and multiplied it by 44: it read a 4x
    search on the E14.5 stack as 637 s, and the search took 57 s. A search
    repeats the cheap steps far more often than the costly ones, and the
    cheap ones are not cheap on a very coarse grid, where vessels merge into
    blobs whose tangled skeletons are slow to turn into graphs -- a graph
    build took 2 s at 8x there and 1.2 s at 4x. Scaling with voxel count is
    cautious for a finer grid, where cost grows more slowly than voxels do.
    """

    step_seconds: Mapping[str, float]
    probe_voxels: float

    def search_seconds(self, voxels: float) -> float:
        one_search = sum(
            _SEARCH_STEP_CALLS.get(name, 0) * seconds for name, seconds in self.step_seconds.items()
        )
        return one_search * float(voxels) / max(float(self.probe_voxels), 1.0)


@dataclass(frozen=True)
class _AutoDownsample:
    """What "Auto" chose, and why -- see :func:`_auto_downsample`."""

    factor: int
    #: The search time it estimated for :attr:`factor`, or ``None`` when it
    #: fell back to the voxel-count heuristic.
    estimated_seconds: Optional[float] = None
    #: The typical vessel radius it measured, and the coarsest factor that
    #: keeps it :data:`AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS` voxels; ``None``
    #: for a mask with no vessel to measure.
    typical_radius_um: Optional[float] = None
    resolution_cap: Optional[int] = None


def estimate_downsample_factor_for_time_budget(
    raw_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    target_seconds: float = DEFAULT_AUTO_DOWNSAMPLE_TARGET_SECONDS,
    use_thick_vessel_skeletonisation: bool = False,
) -> int:
    """The most-detail (smallest) offered factor estimated to keep a full
    search under *target_seconds* on the machine it actually runs on -- and
    never coarser than one that still resolves the mask's typical vessel.

    :data:`AUTO_DOWNSAMPLE_TARGET_VOXELS` assumes a fixed voxels-per-second
    rate, which is wrong in two ways a real dataset exposes: it does not
    know how fast *this* machine is, and it does not know how
    topologically complex *this* mask's own vessel network is. This times
    each of a search's costly steps once on *this* mask, on the grid the
    resolution cap below allows (the coarsest offered, without one), and
    adds them up as often as a search runs each one
    (:class:`_SearchTimeModel`); a finer grid is chosen only if that time,
    scaled with its voxel count, still fits the budget.

    The time budget is then capped by resolution: no factor at which the
    mask's typical vessel radius is fewer than
    :data:`AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS` search-grid voxels. A
    search on a grid that cannot see the vessels decides nothing about
    them, however quickly it runs.

    Falls back to :func:`resolve_auto_downsample_factor` (the voxel-count
    heuristic, still capped by resolution) if the probe itself cannot run at
    all -- an empty mask, or any other error timing it -- so "Auto" never
    fails a run just because estimating its own runtime did.
    """
    return _auto_downsample(
        raw_mask,
        voxel_size_zyx,
        target_seconds=target_seconds,
        starting_values={"use_thick_vessel_skeletonisation": use_thick_vessel_skeletonisation},
    ).factor


def _auto_downsample(
    raw_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    target_seconds: float,
    starting_values: Optional[Mapping[str, Any]] = None,
) -> _AutoDownsample:
    """:func:`estimate_downsample_factor_for_time_budget`'s factor, with the
    search time it estimated and the resolution cap it applied, for the
    run's report. *starting_values* give the timed steps their settings."""
    typical_radius_um = (
        _typical_vessel_radius_um(_densest_crop(raw_mask, _AUTO_RADIUS_CROP_VOXELS), voxel_size_zyx)
        if raw_mask.any()
        else None
    )
    cap = (
        _coarsest_factor_resolving(typical_radius_um, voxel_size_zyx)
        if typical_radius_um is not None
        else None
    )

    def capped(factor: int) -> int:
        return factor if cap is None else min(factor, cap)

    model = _probe_time_model(
        raw_mask, voxel_size_zyx,
        factor=cap if cap is not None else DOWNSAMPLE_FACTORS[-1],
        starting_values=starting_values,
    )
    if model is None:
        return _AutoDownsample(
            capped(resolve_auto_downsample_factor(raw_mask.shape, voxel_size_zyx)),
            typical_radius_um=typical_radius_um,
            resolution_cap=cap,
        )
    total_voxels = float(raw_mask.size)
    factor = capped(
        _factor_from_time_model(model, total_voxels, target_seconds, voxel_size_zyx=voxel_size_zyx)
    )
    return _AutoDownsample(
        factor,
        estimated_seconds=model.search_seconds(total_voxels / _voxel_reduction(factor, voxel_size_zyx)),
        typical_radius_um=typical_radius_um,
        resolution_cap=cap,
    )


def _probe_time_model(
    raw_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    factor: int,
    starting_values: Optional[Mapping[str, Any]] = None,
) -> Optional[_SearchTimeModel]:
    """Each of a search's costly steps (see :data:`_SEARCH_STEP_CALLS`)
    timed once on the search grid *factor* gives, with *starting_values*
    where they give the step's settings; ``None`` when there is nothing to
    time (an empty mask) or timing it failed."""
    settings = dict(starting_values or {})

    def from_settings(make, *args) -> dict[str, Any]:
        """A step's keyword arguments from *settings*, or none -- the
        function's own defaults -- when they do not give them all."""
        try:
            return make(settings, *args)
        except (KeyError, TypeError, ValueError):
            return {}

    def probe_grid(factor: int) -> tuple[np.ndarray, tuple[float, ...]]:
        mask = _downsample_mask(raw_mask, factor, voxel_size_zyx)
        voxel = tuple(
            float(v) * f for v, f in zip(voxel_size_zyx, axis_downsample_factors(factor, voxel_size_zyx))
        )
        return mask, voxel

    def time_steps(mask: np.ndarray, voxel: tuple[float, ...]) -> dict[str, float]:
        seconds: dict[str, float] = {}

        def timed(name: str, step, *args, **kwargs):
            start = time.perf_counter()
            out = step(*args, **kwargs)
            seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - start
            return out

        cleaned, _raw = timed(
            "cleanup", preprocessing.clean_segmented_mask_for_skeletonisation,
            mask, voxel_size_zyx=voxel, **from_settings(_cleanup_kwargs),
        )
        timed("mask score", preprocessing.score_segmented_mask, cleaned, voxel_size_zyx=voxel)
        plain = timed("plain skeleton", preprocessing.skeletonize_volume, mask)
        thick = timed(
            "thick skeleton", preprocessing.skeletonize_thickness_gated,
            mask, **{"voxel_size_zyx": voxel, **from_settings(_thick_vessel_kwargs, voxel)},
        )
        raw_skeleton = thick if settings.get("use_thick_vessel_skeletonisation") else plain
        radius_map = preprocessing.inscribed_radius_map(mask, voxel)  # once per search: untimed

        def preprocess_and_check() -> np.ndarray:
            skeleton = preprocessing.preprocess_skeleton_for_graph(
                raw_skeleton, segmentation_mask=mask, voxel_size_zyx=voxel, **from_settings(_skeleton_kwargs)
            )
            preprocessing.diagnose_skeleton_mask_consistency(
                skeleton, mask, voxel_size_zyx=voxel, local_radius_map=radius_map
            )
            return skeleton

        skeleton = timed("preprocess", preprocess_and_check)

        def build_and_check():
            graph = graph_mod.build_graph_from_skeleton(
                skeleton,
                segmentation_mask=mask,
                **{"voxel_size": voxel, **from_settings(_graph_kwargs, voxel)},
            )
            graph_mod.diagnose_skeleton_graph_consistency(graph, skeleton, voxel_size_zyx=voxel)
            return graph

        graph = timed("graph", build_and_check)
        smoothing = {
            key: settings[name]
            for key, name in (
                ("method", "centreline_smoothing_method"),
                ("iterations", "centreline_smoothing_iterations"),
                ("max_deviation", "centreline_max_deviation"),
            )
            if name in settings
        }
        timed(
            "smooth", graph_mod.smooth_graph_centrelines,
            graph.copy(), skeleton, voxel_size_zyx=voxel, **smoothing,
        )
        return seconds

    try:
        coarse_mask, coarse_voxel = probe_grid(DOWNSAMPLE_FACTORS[-1])
        if not coarse_mask.any():
            return None
        probe_mask, probe_voxel = probe_grid(factor)
        # One untimed warm-up pass first, on the coarsest grid where it is
        # cheapest: scipy/skimage/networkx do a lot of
        # lazy, one-time work (imports, numba/compiled-kernel setup, disk
        # cache checks) on their own first real use in a fresh process --
        # confirmed on this machine to cost seconds on a call that settles
        # to milliseconds immediately after. Timing that cold-start cost
        # would inflate the estimate by orders of magnitude and pick a far
        # coarser factor than the search actually needs. Only the warming
        # matters, so a grid too coarse to keep a skeleton at all is no loss.
        try:
            time_steps(coarse_mask, coarse_voxel)
        except Exception:  # noqa: BLE001
            pass
        step_seconds = time_steps(probe_mask, probe_voxel)
    except Exception:  # noqa: BLE001 - estimating runtime must never block a real run
        return None
    return _SearchTimeModel(step_seconds, float(probe_mask.size))


def _factor_from_time_model(
    model: _SearchTimeModel,
    total_voxels: float,
    target_seconds: float,
    *,
    voxel_size_zyx: Optional[Sequence[float]] = None,
) -> int:
    """The most-detail offered factor whose estimated search time (on a
    volume of *total_voxels*, reduced as :func:`axis_downsample_factors`
    reduces it) fits within *target_seconds*; the coarsest when none does.
    Pure arithmetic, split out from :func:`_auto_downsample` so the decision
    itself is directly testable without timing anything real."""
    best = DOWNSAMPLE_FACTORS[-1]
    for factor in reversed(DOWNSAMPLE_FACTORS):
        voxels = total_voxels / _voxel_reduction(factor, voxel_size_zyx)
        if model.search_seconds(voxels) <= target_seconds:
            best = factor
        else:
            break
    return best


def _coarsest_factor_resolving(typical_radius_um: float, voxel_size_zyx: Sequence[float]) -> int:
    """The largest offered factor whose search grid keeps *typical_radius_um*
    at least :data:`AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS` voxels, judged
    in-plane as :func:`_typical_vessel_radius_um` judges it; 1 when even the
    full-resolution grid does not."""
    best = DOWNSAMPLE_FACTORS[0]
    for factor in DOWNSAMPLE_FACTORS:
        axis_factors = axis_downsample_factors(factor, voxel_size_zyx)
        in_plane_voxel_um = max(
            float(v) * f for v, f in zip(tuple(voxel_size_zyx)[1:], axis_factors[1:])
        )
        if typical_radius_um / in_plane_voxel_um >= AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS:
            best = factor
        else:
            break
    return best


def _densest_crop(mask: np.ndarray, max_voxels: int) -> np.ndarray:
    """*mask* itself when it holds at most *max_voxels*, else the crop of
    about that many voxels -- as near a cube as the mask's shape allows --
    with the most foreground, among crops tiling the whole volume (the last
    on each axis flush with its end): one pass over the mask."""
    if mask.size <= max_voxels:
        return mask
    crop_shape = [0] * mask.ndim
    remaining = float(max_voxels)
    for done, axis in enumerate(np.argsort(mask.shape)):
        # The small nudge keeps a whole-number root whole: 27000 ** (1 / 3) is 29.999...
        side = int(remaining ** (1.0 / (mask.ndim - done)) + 1e-9)
        crop_shape[axis] = max(1, min(int(mask.shape[axis]), side))
        remaining /= crop_shape[axis]
    starts_per_axis = [
        sorted(set(range(0, n - c + 1, c)) | {n - c}) for n, c in zip(mask.shape, crop_shape)
    ]
    best_slices, best_count = None, -1
    for starts in np.array(np.meshgrid(*starts_per_axis, indexing="ij")).reshape(mask.ndim, -1).T:
        slices = tuple(slice(int(s), int(s) + c) for s, c in zip(starts, crop_shape))
        count = int(np.count_nonzero(mask[slices]))
        if count > best_count:
            best_slices, best_count = slices, count
    return mask[best_slices]


def _typical_vessel_radius_um(mask: np.ndarray, voxel_size_zyx: Sequence[float]) -> Optional[float]:
    """*mask*'s own typical vessel radius, or ``None`` for an empty mask.

    Medial-ridge median (see ``preprocessing.medial_ridge_radii_um``),
    further restricted to ridge points at or above the same
    ``DEFAULT_TARGET_VOXELS_ACROSS_RADIUS`` floor "Check segmented
    image" already treats as too discretisation-biased for a reliable
    radius/diameter reading. Without that floor, a dataset with a lot
    of capillaries at or near the voxel-resolution limit drags the
    "typical" scale down to something far smaller than any vessel a
    person could actually point to -- exactly the scale every
    size-based candidate in this module (closing/smoothing radii,
    split thresholds, bundle scan size) is calibrated against, so an
    unreliable, too-small scale reference lets those candidates
    propose a closing/smoothing radius comparable to or larger than a
    real vessel's own diameter (confirmed on real data to erase most
    of the vasculature before it is even re-thresholded). Falls back
    to the unfiltered ridge median when nothing clears the floor --
    still better than every foreground voxel, and this module has no
    more reliable number to offer for a mask that is entirely
    borderline-resolved.
    """
    ridge_radii = preprocessing.medial_ridge_radii_um(mask, tuple(voxel_size_zyx))
    if not ridge_radii.size:
        return None
    # Judged in-plane, as "Check segmented image" judges resolution: a
    # floor of three of the coarsest voxels was 6 um on a 2 um z, which
    # excluded every capillary and left the large vessels setting the
    # "typical" scale -- the too-loose caps this floor exists to prevent.
    in_plane_voxel_um = max(float(v) for v in tuple(voxel_size_zyx)[1:])
    reliable = ridge_radii[
        ridge_radii >= preprocessing.DEFAULT_TARGET_VOXELS_ACROSS_RADIUS * in_plane_voxel_um
    ]
    return float(np.median(reliable)) if reliable.size else float(np.median(ridge_radii))


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
#: The symmetric case: does a candidate erase real, raw-supported
#: foreground the way over-aggressive smoothing/closing can (confirmed on
#: real data: a smoothing sigma several times too large erased ~88% of a
#: real capillary network, none of it flagged as "added", since nothing
#: was invented -- it was all removed). `added_fraction` alone only ever
#: caught over-segmentation; this is what catches under-segmentation from
#: the same raw-image comparison, at no extra cost (both fractions come
#: from the one call).
_MAX_RAW_REMOVED_FRACTION_REGRESSION = 0.02

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

#: Without a raw image to compare a cleaned mask with, cleanup is guarded by
#: the input mask's own real vessels -- its components of at least
#: `_MIN_VESSEL_VOLUME_UM3_FOR_GUARD`, the floor the skeleton guards use. A
#: vessel counts as kept while at least this share of its voxels is still
#: foreground: a smoothing radius several times too large erased ~88% of a
#: real capillary network (see `_MAX_RAW_REMOVED_FRACTION_REGRESSION`), and
#: nothing else in the mask's own score sees that.
_MIN_SHARE_OF_A_VESSEL_KEPT = 0.5
#: A cleanup candidate keeping this much less of the input's real vessels
#: than its sweep's starting point did is rejected.
_MAX_REAL_VESSELS_LOST_REGRESSION = 0.02


#: A sweep moves a setting off its current value only for a candidate scoring
#: lower by more than this share of the current value's own score -- and by
#: more than this much outright, for a score of order one or less. Scores are
#: deterministic, so this is not about noise: it stops a setting changing for
#: a difference too small to mean anything (a thousandth of the largest
#: component's share), and makes a tie keep what the user had.
SWEEP_MIN_IMPROVEMENT = 1e-3


@dataclass(frozen=True)
class TrialRecord:
    """One candidate value, tried and scored, kept for the run's report."""

    group: str
    setting: str
    value: Any
    score: float
    note: str = ""
    #: How long scoring this candidate took.
    seconds: float = field(default=0.0, compare=False)
    #: The measurements its score was made from (see each sweep's cost), by name.
    metrics: Mapping[str, float] = field(default_factory=dict, compare=False)
    #: Which sweep tried it, counted from 0 across the run; -1 for a value
    #: recorded without a sweep (derived, or skipped).
    sweep: int = field(default=-1, compare=False)
    #: Whether it was the setting's value when its sweep started.
    incumbent: bool = field(default=False, compare=False)
    #: Whether its sweep chose it.
    chosen: bool = field(default=False, compare=False)


def _same_value(a: Any, b: Any) -> bool:
    """Whether a candidate is a setting's current value: numbers equal to
    rounding (a refined candidate is rounded to six places), booleans only
    to booleans, tuples item by item."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _chosen_index(
    values: Sequence[Any],
    scores: Sequence[float],
    incumbent: Any,
    *,
    min_improvement: float = SWEEP_MIN_IMPROVEMENT,
) -> Optional[int]:
    """Which candidate a sweep keeps: the lowest-scoring one (the first, on a
    tie), unless the incumbent -- the setting's current value -- was among
    them and nothing beats it by *min_improvement* (see
    :data:`SWEEP_MIN_IMPROVEMENT`). ``None`` when every candidate failed."""
    scored = [i for i, score in enumerate(scores) if score < math.inf]
    if not scored:
        return None
    best = min(scored, key=lambda i: scores[i])
    incumbents = [i for i in scored if _same_value(values[i], incumbent)]
    if incumbents:
        held = min(incumbents, key=lambda i: scores[i])
        margin = min_improvement * max(1.0, abs(scores[held]))
        if scores[best] >= scores[held] - margin:
            return held
    return best


@dataclass(frozen=True)
class OptimisationResult:
    """The winning value for every setting this search decided, and the trials tried."""

    settings: dict[str, Any]
    trials: tuple[TrialRecord, ...]
    #: 1 when the search ran at full resolution; > 1 when it ran on a
    #: downsampled copy (see :func:`resolve_auto_downsample_factor`) and the
    #: voxel-scaled settings were multiplied back up before being returned.
    #: The finest axes' factor; :attr:`downsample_factors_zyx` has every axis's.
    downsample_factor: int = 1
    #: Which of :data:`GROUP_NAMES` actually ran; a name missing here kept its
    #: starting value untouched.
    groups_run: tuple[str, ...] = field(default_factory=lambda: GROUP_NAMES)
    #: How many passes through the group sequence actually ran -- see
    #: :func:`optimise_skeleton_and_graph_settings`'s own ``max_passes``. 1
    #: unless a multi-pass run was requested and needed more than one pass
    #: to converge.
    passes_run: int = 1
    #: How much each axis was reduced by (see :func:`axis_downsample_factors`).
    downsample_factors_zyx: tuple[int, ...] = (1, 1, 1)
    #: How long the search took, from its first sweep to its last.
    seconds: float = 0.0
    #: How long "Auto" estimated it would take, when it chose the sample or
    #: downsampling from a timed probe; ``None`` otherwise.
    estimated_seconds: Optional[float] = None
    #: Time spent in each group that ran, summed over passes.
    group_seconds: Mapping[str, float] = field(default_factory=dict)
    #: The typical vessel radius "Auto" measured, and the coarsest factor it
    #: allowed for it (see :data:`AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS`);
    #: ``None`` when the factor was given, or there was no vessel to measure.
    typical_radius_um: Optional[float] = None
    resolution_cap: Optional[int] = None
    #: Voxel-counted settings whose starting value is under one voxel of the
    #: search grid, which the search did not move: returned as they were.
    finer_than_search_grid: tuple[str, ...] = ()
    #: The starting settings against the chosen ones, on the same measures
    #: (see :mod:`.scorecard`); empty when they could not be measured.
    scorecard: tuple[ScoreRow, ...] = ()


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
        bridge_weight_by_segmentation=bool(settings["skeleton_bridge_weight_by_segmentation"]),
        bridge_z_distance_weight=float(settings["skeleton_bridge_z_distance_weight"]),
        bridge_min_facing_cosine=float(
            settings.get(
                "skeleton_bridge_min_facing_cosine",
                preprocessing.skeleton.DEFAULT_BRIDGE_MIN_FACING_COSINE,
            )
        ),
        **_bridge_gate_kwargs(settings),
    )


def _bridge_gate_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The mask-support gate on bridges, which skeleton cleaning and graph
    building share."""
    return dict(
        bridge_require_mask_support=bool(settings.get("bridge_require_mask_support", True)),
        bridge_max_background_gap_um=float(
            settings.get("bridge_max_background_gap_um", DEFAULT_MAX_BACKGROUND_GAP_UM)
        ),
        bridge_min_mask_fraction=float(
            settings.get("bridge_min_mask_fraction", DEFAULT_MIN_MASK_FRACTION)
        ),
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


def _cleanup_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
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


def _graph_kwargs(settings: Mapping[str, Any], voxel_size_zyx: Sequence[float]) -> dict[str, Any]:
    """``build_graph_from_skeleton`` keyword arguments from a settings dict,
    all but the stub-radius sampler (which needs the mask)."""
    return dict(
        voxel_size=tuple(voxel_size_zyx),
        graph_reconnect_threshold=float(settings["graph_reconnect_threshold"]),
        final_orphan_reconnect_threshold=float(settings["final_orphan_reconnect_threshold"]),
        cluster_collapse_distance=float(settings["cluster_collapse_distance"]),
        min_stub_length=float(settings["min_stub_length"]),
        min_stub_length_radius_multiple=float(settings.get("min_stub_length_radius_multiple", 0.0) or 0.0),
        cluster_collapse_method=str(settings["cluster_collapse_method"]),
        cluster_collapse_max_radial_dispersion=float(settings["cluster_collapse_max_radial_dispersion"]),
        cluster_collapse_persistence_search_multiple=float(
            settings["cluster_collapse_persistence_search_multiple"]
        ),
        **_bridge_gate_kwargs(settings),
        recover_uncovered_mask_vessels=bool(settings.get("recover_uncovered_mask_vessels", True)),
        recovery_min_region_volume_um3=float(
            settings.get(
                "recovery_min_region_volume_um3",
                graph_mod.mask_recovery.DEFAULT_MIN_REGION_VOLUME_UM3,
            )
        ),
        recovery_min_length_um=float(
            settings.get("recovery_min_length_um", graph_mod.mask_recovery.DEFAULT_MIN_LENGTH_UM)
        ),
        facing_dead_end_max_gap_um=float(
            settings.get("facing_dead_end_max_gap_um", graph_mod.facing_ends.DEFAULT_FACING_MAX_GAP_UM)
        ),
    )


def _cartwheel_hub_count(G, settings: Mapping[str, Any]) -> int:
    """How many of *G*'s nodes are "cartwheel" artifacts -- one node
    with many spoke edges radiating in every direction instead of the
    two or three branches a real vessel junction has (see
    `graph.cartwheel_guard`'s own module docstring). Uses the same
    detection thresholds a real pipeline run's own end-of-run check
    would (`cartwheel_hub_min_degree`/`cartwheel_hub_max_radial_dispersion`/
    `cartwheel_hub_tangent_length_um`) when present in *settings*,
    the function's own reasoned defaults otherwise -- these three are
    deliberately not settings this search tunes (see this module's own
    docstring), so a caller's `starting_values` will not always include
    them.
    """
    return len(
        graph_mod.detect_cartwheel_hubs(
            G,
            min_degree=int(settings.get("cartwheel_hub_min_degree", cartwheel_guard.DEFAULT_MIN_DEGREE)),
            max_radial_dispersion=float(
                settings.get(
                    "cartwheel_hub_max_radial_dispersion", cartwheel_guard.DEFAULT_MAX_RADIAL_DISPERSION
                )
            ),
            tangent_length_um=float(
                settings.get("cartwheel_hub_tangent_length_um", cartwheel_guard.DEFAULT_TANGENT_LENGTH_UM)
            ),
        )
    )


#: The before/after scorecard (see :mod:`.scorecard`): each measure's name,
#: whether higher is better, and how far it may move the wrong way and still
#: count as the same. The coverage tolerances are the guards' own.
_SCORECARD_MEASURES: tuple[tuple[str, bool, float], ...] = (
    ("mask score (0-10)", True, 0.05),
    ("input vessels the skeleton reaches", True, 0.02),
    ("input mask the skeleton runs through", True, 0.02),
    ("largest connected share of the skeleton", True, 0.01),
    ("skeleton the graph traces", True, 0.02),
    ("graph components", False, 0.0),
    ("self-loops", False, 0.0),
    ("cartwheel hubs", False, 0.0),
)


def _scorecard_measures(
    input_mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    settings: Mapping[str, Any],
    *,
    input_radius_map: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Everything this search decides -- cleanup, skeleton, graph -- run
    from *input_mask* with *settings*, and measured against *input_mask*
    itself, so the settings a search started from and the ones it chose are
    judged alike. Cleanup applies only while its master switch is on, as in
    a real run."""
    voxel = tuple(float(v) for v in voxel_size_zyx)
    mask = np.asarray(input_mask, dtype=bool)
    if settings.get("segmentation_cleanup"):
        cleaned, _raw = preprocessing.clean_segmented_mask_for_skeletonisation(
            mask, voxel_size_zyx=voxel, **_cleanup_kwargs(settings)
        )
    else:
        cleaned = mask
    skeleton = preprocessing.preprocess_skeleton_for_graph(
        _raw_skeleton(cleaned, settings, voxel),
        segmentation_mask=cleaned,
        voxel_size_zyx=voxel,
        **_skeleton_kwargs(settings),
    )
    kwargs = _graph_kwargs(settings, voxel)
    stub_radius_at = (
        graph_mod.assemble.mask_radius_sampler(cleaned, voxel, 1.0)
        if kwargs["min_stub_length_radius_multiple"] > 0
        else None
    )
    G = graph_mod.build_graph_from_skeleton(
        skeleton, stub_radius_at=stub_radius_at, segmentation_mask=cleaned, **kwargs
    )

    if input_radius_map is None:
        input_radius_map = preprocessing.inscribed_radius_map(mask, voxel)
    voxel_volume_um3 = float(np.prod(voxel))
    min_vessel_voxels = max(2, int(round(_MIN_VESSEL_VOLUME_UM3_FOR_GUARD / max(1e-9, voxel_volume_um3))))
    connectivity = int(settings["skeleton_component_connectivity"])
    topology = met.graph_topology_metrics(G)
    return {
        "mask score (0-10)": float(preprocessing.score_segmented_mask(cleaned, voxel_size_zyx=voxel).total),
        "input vessels the skeleton reaches": float(
            preprocessing.diagnose_vessels_missing_from_skeleton(
                skeleton, mask, voxel_size_zyx=voxel, min_vessel_voxels=min_vessel_voxels,
                local_radius_map=input_radius_map,
            )["explained_vessel_fraction"]
        ),
        "input mask the skeleton runs through": float(
            preprocessing.diagnose_skeleton_mask_consistency(
                skeleton, mask, voxel_size_zyx=voxel, local_radius_map=input_radius_map,
            )["coverage_fraction"]
        ),
        "largest connected share of the skeleton": float(
            preprocessing.compute_skeleton_connectivity_stats(skeleton, connectivity).largest_fraction
        ),
        "skeleton the graph traces": float(
            graph_mod.diagnose_skeleton_graph_consistency(G, skeleton, voxel_size_zyx=voxel)[
                "coverage_fraction"
            ]
        ),
        "graph components": float(topology.n_components),
        "self-loops": float(topology.n_selfloops),
        "cartwheel hubs": float(_cartwheel_hub_count(G, settings)),
    }


def _scorecard(before: Mapping[str, float], after: Mapping[str, float]) -> tuple[ScoreRow, ...]:
    return tuple(
        ScoreRow(name, before[name], after[name], higher_is_better, tolerance)
        for name, higher_is_better, tolerance in _SCORECARD_MEASURES
        if name in before and name in after
    )


def _raw_skeleton(
    mask: np.ndarray, settings: Mapping[str, Any], voxel_size_zyx: tuple[float, float, float]
) -> np.ndarray:
    if settings["use_thick_vessel_skeletonisation"]:
        return preprocessing.skeletonize_thickness_gated(
            mask, **_thick_vessel_kwargs(settings, voxel_size_zyx)
        )
    return preprocessing.skeletonize_volume(mask)


class _SweepBookkeeping:
    """Progress emission, trial recording, group gating and the one-setting
    sweep itself, shared by every sweep engine (:class:`_Search`, and
    :class:`haemolynx.optimisation.fwhm_search._FwhmSearch`).

    Deliberately narrow: only what never depends on what a "trial" or
    "group" actually runs. The convergence loop, the sweep strategies, and
    what one trial measures differ enough between the two searches
    (mask/graph preprocessing vs. FWHM measurement quality) that sharing
    more than this would force one engine's shape onto the other's real
    logic.
    """

    def __init__(
        self,
        *,
        progress: Optional[ProgressCallback],
        enabled_groups: Optional[Iterable[str]],
        group_total: int,
    ) -> None:
        self.progress = progress
        #: ``None`` means every group runs; otherwise only the named ones do
        #: -- see :meth:`_group_enabled`.
        self.enabled_groups: Optional[frozenset[str]] = (
            frozenset(enabled_groups) if enabled_groups is not None else None
        )
        self.groups_run: list[str] = []
        self.trials: list[TrialRecord] = []
        #: Progress denominator -- see each subclass's own ``run``: scaled up
        #: before a multi-pass run so a progress bar's fraction never exceeds
        #: 1 just because convergence needed more than one pass through the
        #: group sequence.
        self._group_total: int = group_total
        self._group_index = 0
        #: Time spent in each group, summed over passes -- see :meth:`_run_group`.
        self.group_seconds: dict[str, float] = {}
        #: What the candidate being scored has noted -- see :meth:`_note`.
        self._trial_metrics: dict[str, float] = {}

    def _group_enabled(self, name: str) -> bool:
        enabled = self.enabled_groups is None or name in self.enabled_groups
        if enabled:
            self.groups_run.append(name)
        return enabled

    def _run_group(self, name: str, method: Callable[[], None]) -> bool:
        """Run one group's *method* if it is enabled, timing it into
        :attr:`group_seconds`; whether it ran."""
        if not self._group_enabled(name):
            return False
        start = time.perf_counter()
        try:
            method()
        finally:
            self.group_seconds[name] = self.group_seconds.get(name, 0.0) + (
                time.perf_counter() - start
            )
        return True

    def _note(self, **metrics: float) -> None:
        """Keep named measurements with the candidate being scored, for the
        report to say why the winner won. A cost function calls this; called
        anywhere else, it is forgotten when the next candidate starts."""
        self._trial_metrics.update({name: float(value) for name, value in metrics.items()})

    # -- sweeps -----------------------------------------------------------------
    def _sweep(self, group: str, setting: str, candidate_values: Sequence[Any], cost_fn) -> Any:
        """Try every candidate for one setting and keep the one
        :func:`_chosen_index` picks: the lowest-cost one, unless the current
        value scores within :data:`SWEEP_MIN_IMPROVEMENT` of it.

        *cost_fn(value) -> float* runs the real pipeline code for one candidate
        and returns its cost (lower is better); it may raise, which scores that
        candidate as a loss without ending the sweep. If every candidate fails,
        the setting keeps whatever value it already had in ``self.current``.
        """
        winner = self._run_candidates(
            group, setting, candidate_values, cost_fn, incumbent=self.current.get(setting)
        )
        self.current[setting] = winner
        self._emit(GROUP_FINISHED, group, winner={setting: winner})
        self._group_index += 1
        return winner

    def _run_candidates(
        self, group: str, setting: str, candidate_values: Sequence[Any], cost_fn, *, incumbent: Any
    ) -> Any:
        """The trial loop behind :meth:`_sweep`, for a sweep that applies its
        winner itself (a joint sweep over several settings): emits the start
        and every candidate, records every trial, and returns the chosen
        value -- *incumbent* itself when it was kept, or when every
        candidate failed."""
        self._emit(GROUP_STARTED, group)
        values = list(candidate_values)
        outcomes: list[tuple[float, str, float, dict[str, float]]] = []
        for index, value in enumerate(values):
            note = ""
            self._trial_metrics = {}
            start = time.perf_counter()
            try:
                score = float(cost_fn(value))
            except Exception as error:  # noqa: BLE001 - one bad candidate must not end the sweep
                score = float("inf")
                note = f"failed: {error}"
            outcomes.append((score, note, time.perf_counter() - start, dict(self._trial_metrics)))
            self._emit(CANDIDATE_EVALUATED, group, candidate_index=index, candidate_total=len(values))

        chosen = _chosen_index(values, [outcome[0] for outcome in outcomes], incumbent)
        for index, (value, (score, note, seconds, metrics)) in enumerate(zip(values, outcomes)):
            self._record(
                group, setting, value, score, note,
                seconds=seconds, metrics=metrics, sweep=self._group_index,
                incumbent=_same_value(value, incumbent), chosen=index == chosen,
            )
        if chosen is None or _same_value(values[chosen], incumbent):
            return incumbent
        return values[chosen]

    # -- progress / bookkeeping -------------------------------------------------
    def _emit(self, kind: str, group_name: str, **extra: Any) -> None:
        if self.progress is None:
            return
        self.progress(
            OptimisationEvent(
                kind=kind,
                group_index=self._group_index,
                group_total=self._group_total,
                group_name=group_name,
                **extra,
            )
        )

    def _record(
        self, group: str, setting: str, value: Any, score: float, note: str = "", **details: Any
    ) -> None:
        """Keep one trial for the report; *details* are :class:`TrialRecord`'s
        optional fields."""
        self.trials.append(
            TrialRecord(group=group, setting=setting, value=value, score=score, note=note, **details)
        )


class _Search(_SweepBookkeeping):
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
        # Always a fresh copy, even when raw_mask is already a bool array
        # (np.asarray would then return the caller's own object unchanged):
        # self.original_raw_mask below must never alias memory the caller
        # still owns, since nothing downstream is allowed to assume it can
        # safely mutate `raw_mask` in place just because this search does not.
        self.raw_mask = np.array(raw_mask, dtype=bool, copy=True)
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
        super().__init__(
            progress=progress,
            enabled_groups=enabled_groups,
            group_total=_GROUP_TOTAL_UPPER_BOUND,
        )
        #: How many passes :meth:`run` actually executed -- 1 unless
        #: ``max_passes`` > 1 and convergence took more than one pass.
        self.passes_run: int = 0
        self.raw_skeleton: np.ndarray = np.zeros_like(self.raw_mask)
        self.current_skeleton: np.ndarray = self.raw_skeleton
        self.current_graph = None
        #: Set fresh at the top of `_group_closing_radius`/`_group_gap_bridging`
        #: (each group's own entering state) -- `_fusion_cost` is a shared
        #: method, not a sweep-local closure, so its own coverage-regression
        #: guard needs this as instance state rather than a captured local.
        self._fusion_baseline_coverage: float = 1.0
        #: (id(self.raw_mask), radius_map) -- see `_raw_mask_local_radius_map`.
        self._raw_mask_local_radius_map_cache: Optional[tuple[int, np.ndarray]] = None
        #: The input mask's components and which are real vessels -- see
        #: `_input_vessels`.
        self._input_vessels_cache: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        #: Precomputed once (Otsu threshold + labelled foreground of
        #: `self.raw_image`) so `_group_segmentation_cleanup` does not redo
        #: that -- mask-independent -- work for every one of its ~20
        #: sequential sub-sweeps' worth of candidates. `None` when no raw
        #: image was supplied, exactly like `self.raw_image` itself.
        self._raw_image_foreground: Optional[preprocessing.RawImageForeground] = (
            preprocessing.analyze_raw_image_foreground(
                self.raw_image, voxel_size_zyx=self.voxel_size_zyx
            )
            if self.raw_image is not None
            else None
        )
        #: The last few results of each kind of trial (see `trial_cache`):
        #: a sweep's baseline is its current value's own candidate trial, and
        #: the skeleton or graph a sweep leaves behind is its winner's.
        self._cleanup_cache = TrialCache()
        self._skeletonise_cache = TrialCache()
        self._preprocess_cache = TrialCache()
        self._graph_cache = TrialCache(max_entries=8)

        self.typical_radius_um = self._measure_typical_radius_um(self.raw_mask) or 1.0

    def _measure_typical_radius_um(self, mask: np.ndarray) -> Optional[float]:
        """*mask*'s own typical vessel radius on this search's grid -- see
        :func:`_typical_vessel_radius_um`."""
        return _typical_vessel_radius_um(mask, self.voxel_size_zyx)

    def _sweep_with_refinement(
        self,
        group: str,
        setting: str,
        candidate_values: list,
        cost_fn,
        *,
        refine_fractions: tuple[float, ...] = (0.75, 1.25),
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> Any:
        """:meth:`_sweep`, then one more focused round of candidates around
        the winner.

        The coarse multiplier grids this search's candidate generators use
        (0.5x/1x/2x/4x the coarsest voxel spacing, or of a measured typical
        radius) are deliberately sparse -- cheap to evaluate, but a
        genuinely better value can sit between two grid points, which a
        single coarse pass can never find. This tries *refine_fractions* of
        the coarse winner (0.75x/1.25x by default: a modest, local
        neighbourhood, not another multiplicative octave) alongside the
        winner itself, so refinement can only match or improve on the
        coarse result, never regress it -- if the winner is already a local
        optimum, the second round just re-confirms it at the cost of a
        couple of extra real evaluations.

        *min_value*/*max_value* bound the refined candidates the same way
        the coarse generator was bounded (e.g. a caller-supplied
        ``typical_radius_um`` cap) -- refining outward must not reintroduce
        a candidate the coarse grid excluded for real safety reasons, not
        just because it was not on the grid.

        Only meaningful for numeric settings; every other setting's
        candidates keep exactly today's coarse-grid-only behaviour, since
        this method is opt-in per call site, not automatic for every
        :meth:`_sweep` call.
        """
        winner = self._sweep(group, setting, candidate_values, cost_fn)
        if not isinstance(winner, (int, float)) or isinstance(winner, bool):
            return winner
        winner = float(winner)
        refine_candidates = {winner} | {round(winner * f, 6) for f in refine_fractions}
        refine_candidates = {
            v for v in refine_candidates
            if v > min_value and (max_value is None or v <= max_value)
        }
        if len(refine_candidates) <= 1:
            return winner
        return self._sweep(group, setting, sorted(refine_candidates), cost_fn)

    # -- skeleton-side trial helpers ---------------------------------------------
    def _preprocess_trial(self, overrides: Mapping[str, Any]) -> np.ndarray:
        kwargs = _skeleton_kwargs({**self.current, **overrides})
        return self._preprocess_cache.get_or_compute(
            (self.raw_skeleton, self.raw_mask),
            (kwargs, self.voxel_size_zyx),
            lambda: preprocessing.preprocess_skeleton_for_graph(
                self.raw_skeleton,
                segmentation_mask=self.raw_mask,
                voxel_size_zyx=self.voxel_size_zyx,
                **kwargs,
            ),
        )

    def _raw_skeleton_trial(self, overrides: Mapping[str, Any]) -> np.ndarray:
        """:func:`_raw_skeleton` of ``self.raw_mask`` with ``self.current``
        plus *overrides*: the thickness-gated skeleton when that is on, else
        the plain one."""
        settings = {**self.current, **overrides}
        if settings["use_thick_vessel_skeletonisation"]:
            key: Any = ("thickness_gated", _thick_vessel_kwargs(settings, self.voxel_size_zyx))
        else:
            key = ("plain",)
        return self._skeletonise_cache.get_or_compute(
            (self.raw_mask,), key,
            lambda: _raw_skeleton(self.raw_mask, settings, self.voxel_size_zyx),
        )

    def _connectivity(self) -> Optional[int]:
        return int(self.current["skeleton_component_connectivity"])

    # -- group 0: segmentation cleanup (raw mask, before skeletonisation) --------
    def _cleanup_trial(self, overrides: Mapping[str, Any]) -> np.ndarray:
        """Re-clean :attr:`original_raw_mask` (never a previous trial's own
        output -- matches how ``_preprocess_trial`` always re-runs from
        ``self.raw_skeleton``) with ``self.current`` plus *overrides*."""
        kwargs = _cleanup_kwargs({**self.current, **overrides})

        def clean() -> np.ndarray:
            cleaned, _raw = preprocessing.clean_segmented_mask_for_skeletonisation(
                self.original_raw_mask, voxel_size_zyx=self.voxel_size_zyx, **kwargs,
            )
            return cleaned

        return self._cleanup_cache.get_or_compute(
            (self.original_raw_mask,), (kwargs, self.voxel_size_zyx), clean
        )

    def _group_segmentation_cleanup(self) -> None:
        """Choose the seven segmentation-cleanup toggles and their own knobs,
        each scored on the resulting mask's own quality score -- these act
        directly on the mask, before any skeleton exists to measure instead.
        """
        group = "segmentation_cleanup"

        # Optional: when a raw reference image is supplied (see
        # `optimise_skeleton_and_graph_settings`'s own `raw_image`
        # parameter), guard every candidate against inventing foreground the
        # raw signal does not support *and* against erasing real foreground
        # the raw signal does support -- `score_segmented_mask` alone cannot
        # see either, since it only ever looks at the mask's own geometry
        # (see `preprocessing.segmentation_raw_comparison`'s own module
        # docstring for why this is a genuinely different signal). Both
        # directions come from the one `compare_segmentation_to_raw_image`
        # call, so guarding the second costs nothing extra.
        baseline_comparison: Optional[preprocessing.SegmentationRawComparison] = None
        # Without a raw image: the share of the input's real vessels this
        # sub-sweep's starting point keeps (see `_real_vessels_kept_fraction`).
        baseline_vessels_kept: Optional[float] = None

        def quality(mask_trial: np.ndarray) -> float:
            mask_score = preprocessing.score_segmented_mask(
                mask_trial, voxel_size_zyx=self.voxel_size_zyx
            ).total
            self._note(mask_score=mask_score)
            score = -mask_score
            if baseline_vessels_kept is not None:
                vessels_kept = self._real_vessels_kept_fraction(mask_trial)
                penalty = self._regression_penalty(
                    vessels_kept, baseline_vessels_kept, _MAX_REAL_VESSELS_LOST_REGRESSION,
                )
                self._note(real_vessels_kept=vessels_kept, guard_penalty=penalty)
                score += penalty
            if baseline_comparison is not None:
                comparison = preprocessing.compare_segmentation_to_raw_image(
                    mask_trial, self.raw_image, voxel_size_zyx=self.voxel_size_zyx,
                    precomputed=self._raw_image_foreground,
                )
                penalty = self._regression_penalty(
                    1.0 - comparison.added_fraction,
                    1.0 - baseline_comparison.added_fraction,
                    _MAX_RAW_ADDED_FRACTION_REGRESSION,
                )
                penalty += self._regression_penalty(
                    1.0 - comparison.removed_fraction,
                    1.0 - baseline_comparison.removed_fraction,
                    _MAX_RAW_REMOVED_FRACTION_REGRESSION,
                )
                self._note(
                    raw_added_fraction=comparison.added_fraction,
                    raw_removed_fraction=comparison.removed_fraction,
                    guard_penalty=penalty,
                )
                score += penalty
            return score

        def sweep_cleanup(setting: str, candidate_values: list, *, refine: bool = False) -> Any:
            """`self._sweep` for one segmentation-cleanup setting, refreshing
            the raw-image baseline immediately beforehand.

            Every sub-sweep in this group can move `self.current`, which
            `self._cleanup_trial({})` (what the baseline is measured
            against) reads -- computing the baseline once at group entry
            would leave it fixed at the *pre-group* mask while later
            sub-sweeps' candidates are judged against whatever every prior
            sub-sweep in this same group already decided, silently
            misattributing their own effect on `added_fraction`/
            `removed_fraction` to whichever setting happens to run next
            (the same "matched processing stage" bug `_group_min_branch_length`
            was fixed for -- see project memory). Refreshing here keeps the
            baseline at exactly this sub-sweep's own starting point instead.

            *refine*, for the radius-style settings that use
            `cand.small_radius_candidates`'s coarse voxel-scale grid, adds
            one focused round around the coarse winner (see
            `_sweep_with_refinement`), capped at this mask's own
            `typical_radius_um` -- the same safety bound the coarse grid
            itself was generated with, so refining cannot reintroduce the
            failure mode that bound exists for.
            """
            nonlocal baseline_comparison, baseline_vessels_kept
            if self.raw_image is not None:
                baseline_comparison = preprocessing.compare_segmentation_to_raw_image(
                    self._cleanup_trial({}), self.raw_image, voxel_size_zyx=self.voxel_size_zyx,
                    precomputed=self._raw_image_foreground,
                )
            else:
                baseline_vessels_kept = self._real_vessels_kept_fraction(self._cleanup_trial({}))
            cost_fn = lambda value: quality(self._cleanup_trial({setting: value}))
            if refine:
                return self._sweep_with_refinement(
                    group, setting, candidate_values, cost_fn, max_value=self.typical_radius_um,
                )
            return self._sweep(group, setting, candidate_values, cost_fn)

        sweep_cleanup("segmentation_cleanup_fill_cavities", [False, True])

        sweep_cleanup("segmentation_cleanup_remove_whiskers", [False, True])
        if self.current["segmentation_cleanup_remove_whiskers"]:
            sweep_cleanup(
                "segmentation_cleanup_whisker_radius_um",
                cand.small_radius_candidates(
                    self.voxel_size_zyx,
                    float(self.current["segmentation_cleanup_whisker_radius_um"]),
                    typical_radius_um=self.typical_radius_um,
                    footprint=True,
                ),
                refine=True,
            )

        sweep_cleanup("segmentation_cleanup_split_narrow_necks", [False, True])
        if self.current["segmentation_cleanup_split_narrow_necks"]:
            sweep_cleanup(
                "segmentation_cleanup_split_min_marker_separation_um",
                cand.split_marker_separation_candidates(
                    self.typical_radius_um,
                    float(self.current["segmentation_cleanup_split_min_marker_separation_um"]),
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_split_min_pinch_radius_ratio",
                cand.split_pinch_radius_ratio_candidates(
                    float(self.current["segmentation_cleanup_split_min_pinch_radius_ratio"])
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_split_min_body_radius_um",
                cand.split_min_body_radius_candidates(
                    self.typical_radius_um,
                    float(self.current["segmentation_cleanup_split_min_body_radius_um"]),
                ),
            )

        sweep_cleanup("segmentation_cleanup_close_gaps", [False, True])
        if self.current["segmentation_cleanup_close_gaps"]:
            sweep_cleanup(
                "segmentation_cleanup_close_gaps_radius_um",
                cand.small_radius_candidates(
                    self.voxel_size_zyx,
                    float(self.current["segmentation_cleanup_close_gaps_radius_um"]),
                    typical_radius_um=self.typical_radius_um,
                    footprint=True,
                ),
                refine=True,
            )

        sweep_cleanup("segmentation_cleanup_reconnect_gaps", [False, True])
        if self.current["segmentation_cleanup_reconnect_gaps"]:
            gap_distances_um = cand.mask_component_gap_distances_um(
                self.original_raw_mask, self.voxel_size_zyx
            )
            sweep_cleanup(
                "segmentation_cleanup_reconnect_max_bridge_distance_um",
                cand.reconnect_max_bridge_distance_candidates(
                    gap_distances_um,
                    float(self.current["segmentation_cleanup_reconnect_max_bridge_distance_um"]),
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_reconnect_min_cylindricality",
                cand.reconnect_min_cylindricality_candidates(
                    float(self.current["segmentation_cleanup_reconnect_min_cylindricality"])
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_reconnect_max_axis_angle_degrees",
                cand.reconnect_max_axis_angle_candidates(
                    float(self.current["segmentation_cleanup_reconnect_max_axis_angle_degrees"])
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_reconnect_min_facing_cosine",
                cand.reconnect_min_facing_cosine_candidates(
                    float(self.current["segmentation_cleanup_reconnect_min_facing_cosine"])
                ),
            )
            sweep_cleanup(
                "segmentation_cleanup_reconnect_max_radius_ratio",
                cand.reconnect_max_radius_ratio_candidates(
                    float(self.current["segmentation_cleanup_reconnect_max_radius_ratio"])
                ),
            )

        sweep_cleanup("segmentation_cleanup_smooth_surfaces", [False, True])
        if self.current["segmentation_cleanup_smooth_surfaces"]:
            sweep_cleanup(
                "segmentation_cleanup_smooth_method", cand.smooth_method_candidates(),
            )
            if self.current["segmentation_cleanup_smooth_method"] == "gaussian":
                sweep_cleanup(
                    "segmentation_cleanup_smooth_sigma_um",
                    cand.small_radius_candidates(
                        self.voxel_size_zyx,
                        float(self.current["segmentation_cleanup_smooth_sigma_um"]),
                        typical_radius_um=self.typical_radius_um,
                    ),
                    refine=True,
                )
            else:
                sweep_cleanup(
                    "segmentation_cleanup_smooth_morphological_radius_um",
                    cand.small_radius_candidates(
                        self.voxel_size_zyx,
                        float(self.current["segmentation_cleanup_smooth_morphological_radius_um"]),
                        typical_radius_um=self.typical_radius_um,
                        footprint=True,
                    ),
                    refine=True,
                )

        sweep_cleanup("segmentation_cleanup_remove_small_volumes", [False, True])
        if self.current["segmentation_cleanup_remove_small_volumes"]:
            sweep_cleanup(
                "segmentation_cleanup_remove_small_min_volume_um3",
                cand.remove_small_min_volume_candidates(
                    self.original_raw_mask,
                    self.voxel_size_zyx,
                    float(self.current["segmentation_cleanup_remove_small_min_volume_um3"]),
                ),
            )

        # The seven toggles above are only read at all while this master
        # switch is on (see pipeline.schema's segmentation_cleanup and
        # pipeline.stages.skeletonise) -- a real run applying these chosen
        # settings must not silently ignore what this group just decided
        # was worth turning on.
        if any(
            self.current[flag]
            for flag in (
                "segmentation_cleanup_fill_cavities",
                "segmentation_cleanup_remove_whiskers",
                "segmentation_cleanup_split_narrow_necks",
                "segmentation_cleanup_close_gaps",
                "segmentation_cleanup_reconnect_gaps",
                "segmentation_cleanup_smooth_surfaces",
                "segmentation_cleanup_remove_small_volumes",
            )
        ):
            self.current["segmentation_cleanup"] = True

        self.raw_mask = self._cleanup_trial({})
        # The typical-radius scale used by later groups' own candidate
        # generation (thick-vessel refinement, bundle scan size) should
        # reflect the mask cleanup actually decided on, not the pre-cleanup
        # input measured in `__init__`.
        measured = self._measure_typical_radius_um(self.raw_mask)
        if measured is not None:
            self.typical_radius_um = measured

    # -- shared input-data-consistency guards -------------------------------------
    # Every helper below returns a "higher is better" fraction so
    # `_regression_penalty` has one uniform shape, and every one reuses an
    # existing, already-tested diagnostic that previously only ran as an
    # end-of-a-real-run warning (`pipeline/stages.py`) -- never used to
    # choose between candidates until now.
    def _raw_mask_local_radius_map(self) -> np.ndarray:
        """`inscribed_radius_map` of `self.raw_mask`'s own canonical
        binarisation, cached against `id(self.raw_mask)`.

        `self.raw_mask` is reassigned to a new array exactly once, at the
        end of `_group_segmentation_cleanup`, and never mutated in place
        afterward -- so its identity is a valid cache key for the rest of
        a run. Every later group's own guard calls (`_vessels_represented_fraction`,
        `_skeleton_mask_coverage_fraction`) would otherwise redo this same
        full-volume EDT from scratch on every single candidate, dozens of
        times over in a real run, for a mask that has not changed since the
        previous call.
        """
        from haemolynx.io.load import _to_binary_volume_for_skeletonization

        key = id(self.raw_mask)
        cached = self._raw_mask_local_radius_map_cache
        if cached is None or cached[0] != key:
            mask_bool = _to_binary_volume_for_skeletonization(self.raw_mask)
            radius_map = preprocessing.inscribed_radius_map(mask_bool, self.voxel_size_zyx)
            self._raw_mask_local_radius_map_cache = (key, radius_map)
            return radius_map
        return cached[1]

    def _input_vessels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The input mask's components (26-connected), each one's voxel
        count, and which are real vessels (at least
        `_MIN_VESSEL_VOLUME_UM3_FOR_GUARD`) -- labelled once per search."""
        if self._input_vessels_cache is None:
            from scipy.ndimage import label as label_components

            labels, n_components = label_components(
                self.original_raw_mask, structure=np.ones((3, 3, 3), dtype=bool)
            )
            sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
            voxel_volume_um3 = float(np.prod(self.voxel_size_zyx))
            min_voxels = max(
                2, int(round(_MIN_VESSEL_VOLUME_UM3_FOR_GUARD / max(1e-9, voxel_volume_um3)))
            )
            real = sizes >= min_voxels
            real[0] = False
            self._input_vessels_cache = (labels, sizes, real)
        return self._input_vessels_cache

    def _real_vessels_kept_fraction(self, mask: np.ndarray) -> float:
        """Share of the input mask's real vessels (see `_input_vessels`)
        that *mask* keeps at least `_MIN_SHARE_OF_A_VESSEL_KEPT` of; 1.0
        when the input has none."""
        labels, sizes, real = self._input_vessels()
        if not real.any():
            return 1.0
        kept_voxels = np.bincount(labels[np.asarray(mask, dtype=bool)], minlength=sizes.size)
        kept = real & (kept_voxels >= _MIN_SHARE_OF_A_VESSEL_KEPT * sizes)
        return float(np.count_nonzero(kept)) / float(np.count_nonzero(real))

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
            local_radius_map=self._raw_mask_local_radius_map(),
        )
        return float(report["explained_vessel_fraction"])

    def _skeleton_mask_coverage_fraction(self, skeleton: np.ndarray) -> float:
        """Fraction of self.raw_mask's own volume the skeleton still runs
        through. See preprocessing.diagnose_skeleton_mask_consistency."""
        report = preprocessing.diagnose_skeleton_mask_consistency(
            skeleton, self.raw_mask, voxel_size_zyx=self.voxel_size_zyx,
            local_radius_map=self._raw_mask_local_radius_map(),
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

    def _cartwheel_hub_count(self, G) -> int:
        """How many of *G*'s nodes are "cartwheel" artifacts -- one node
        with many spoke edges radiating in every direction instead of the
        two or three branches a real vessel junction has (see
        `graph.cartwheel_guard`'s own module docstring). Uses the same
        detection thresholds a real pipeline run's own end-of-run check
        would (`cartwheel_hub_min_degree`/`cartwheel_hub_max_radial_dispersion`/
        `cartwheel_hub_tangent_length_um`) when present in `self.current`,
        the function's own reasoned defaults otherwise -- these three are
        deliberately not settings this search tunes (see this module's own
        docstring), so a caller's `starting_values` will not always include
        them.
        """
        return _cartwheel_hub_count(G, self.current)

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
            self.raw_skeleton = self._raw_skeleton_trial({})
            return

        def cost_for(skeleton: np.ndarray, on: bool) -> float:
            stats = preprocessing.compute_skeleton_connectivity_stats(skeleton, self._connectivity())
            self._note(largest_fraction=stats.largest_fraction)
            # A tiny bias toward "off" so a marginal gain does not turn on a
            # second algorithm for its own sake (Occam's razor).
            return -stats.largest_fraction + (1e-3 if on else 0.0)

        def trial_on_off(value: bool) -> float:
            skeleton = self._raw_skeleton_trial({"use_thick_vessel_skeletonisation": value})
            return cost_for(skeleton, value)

        self._sweep(group, "use_thick_vessel_skeletonisation", [False, True], trial_on_off)

        if not self.current["use_thick_vessel_skeletonisation"]:
            self.raw_skeleton = self._raw_skeleton_trial({})
            return

        def trial_with(overrides: Mapping[str, Any]) -> np.ndarray:
            return self._raw_skeleton_trial(overrides)

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
        baseline_vessels_represented: float = 0.0

        def cost_braid(skeleton: np.ndarray) -> float:
            stats = preprocessing.compute_skeleton_connectivity_stats(skeleton, self._connectivity())
            vessels_represented = self._vessels_represented_fraction(skeleton)
            penalty = self._regression_penalty(
                vessels_represented,
                baseline_vessels_represented,
                _MAX_MISSING_VESSEL_FRACTION_REGRESSION,
            )
            braid = met.braid_factor_along_long_axis(skeleton)
            self._note(
                braid_factor=braid,
                largest_fraction=stats.largest_fraction,
                vessels_represented=vessels_represented,
                guard_penalty=penalty,
            )
            return braid + (1.0 - stats.largest_fraction) + penalty

        def sweep_refinement(setting: str, candidate_values: list) -> Any:
            """`self._sweep` for one thick-vessel refinement setting,
            refreshing the vessels-represented baseline immediately
            beforehand -- see `_group_segmentation_cleanup`'s own
            `sweep_cleanup` for why a baseline shared across sequential
            sub-sweeps must be refreshed at each one's own starting point
            rather than fixed once for the whole group.
            """
            nonlocal baseline_vessels_represented
            baseline_vessels_represented = self._vessels_represented_fraction(trial_with({}))
            return self._sweep(
                group, setting, candidate_values,
                lambda value: cost_braid(trial_with({setting: value})),
            )

        typical_thick_radius_um = cand.typical_thick_vessel_radius_um(
            self.raw_mask, self.voxel_size_zyx, float(self.current["skeleton_thick_vessel_min_radius_um"])
        )

        sweep_refinement(
            "skeleton_thick_vessel_wall_absorption_um",
            cand.thick_vessel_wall_absorption_candidates(typical_thick_radius_um),
        )
        sweep_refinement(
            "skeleton_thick_vessel_flake_filter_um",
            cand.thick_vessel_flake_filter_candidates(self.voxel_size_zyx),
        )
        sweep_refinement(
            "skeleton_thick_vessel_max_bridge_radius_multiple",
            cand.thick_vessel_max_bridge_radius_multiple_candidates(
                float(self.current["skeleton_thick_vessel_max_bridge_radius_multiple"])
            ),
        )
        # The raw mask's own connected-component gaps stand in for "how far
        # apart are the fragments a bridge might need to reach" -- the real
        # skeleton does not exist yet at this point in the very first group.
        gap_distances_um = preprocessing.inter_component_gap_distances(
            self.raw_mask, self._connectivity(), self.voxel_size_zyx
        )
        sweep_refinement(
            "skeleton_thick_vessel_max_bridge_distance_um",
            cand.thick_vessel_max_bridge_distance_candidates(gap_distances_um),
        )
        sweep_refinement(
            "skeleton_thick_vessel_bridge_radius_smoothing_um",
            cand.thick_vessel_bridge_radius_smoothing_candidates(typical_thick_radius_um),
        )

        self.raw_skeleton = self._raw_skeleton_trial({})

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
            vessels_represented = self._vessels_represented_fraction(cleaned)
            penalty += self._regression_penalty(
                vessels_represented,
                baseline_vessels_represented,
                _MAX_MISSING_VESSEL_FRACTION_REGRESSION,
            )
            self._note(
                largest_fraction=stats.largest_fraction,
                voxels_removed_fraction=removed_fraction,
                vessels_represented=vessels_represented,
                guard_penalty=penalty,
            )
            return -stats.largest_fraction + penalty

        self._sweep(group, "skeleton_min_branch_length", candidate_values, cost)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 3: bundle refinement -------------------------------------------------
    def _group_bundle_refinement(self) -> None:
        group = "bundle_refinement"

        def guard_baseline() -> tuple[int, float]:
            """(n_components, coverage_fraction) for the skeleton
            `self.current` would produce right now -- matches
            `self.current_skeleton` at group entry (nothing in `self.current`
            has moved yet), but must be recomputed after this group's own
            first sub-sweep decides a winner, or the second sub-sweep's
            guard would judge its candidates against a stale, pre-sweep
            reference (see project memory: "matched processing stage").
            """
            cleaned = self._preprocess_trial({})
            n_components = preprocessing.compute_skeleton_connectivity_stats(
                cleaned, self._connectivity()
            ).n_components
            return n_components, self._skeleton_mask_coverage_fraction(cleaned)

        baseline_components, baseline_coverage = guard_baseline()

        def leftover_density(cleaned: np.ndarray, scan_size: int, density_fraction: float) -> float:
            from scipy.ndimage import uniform_filter

            from haemolynx.preprocessing.skeleton import bundle_scan_window

            window = bundle_scan_window(int(scan_size), self.voxel_size_zyx, cleaned.ndim)
            density_after = uniform_filter(cleaned.astype(float), size=window)
            leftover = float((density_after >= density_fraction).sum())
            self._note(dense_voxels_left=leftover)
            return leftover

        def guard(cleaned: np.ndarray) -> float:
            n_components = preprocessing.compute_skeleton_connectivity_stats(
                cleaned, self._connectivity()
            ).n_components
            penalty = _GUARD_PENALTY if n_components > baseline_components else 0.0
            coverage = self._skeleton_mask_coverage_fraction(cleaned)
            penalty += self._regression_penalty(
                coverage, baseline_coverage, _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            self._note(n_components=n_components, mask_coverage=coverage, guard_penalty=penalty)
            return penalty

        scan_candidates = cand.bundle_scan_size_candidates(
            self.raw_mask, self.voxel_size_zyx, int(self.current["skeleton_bundle_scan_size"])
        )

        def cost_scan(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_bundle_scan_size": value})
            density_fraction = float(self.current["skeleton_bundle_density_fraction"])
            return leftover_density(cleaned, value, density_fraction) + guard(cleaned)

        self._sweep(group, "skeleton_bundle_scan_size", scan_candidates, cost_scan)

        # Refresh: the scan-size sweep above may have moved self.current, so
        # `guard`'s own baseline must be recomputed at this sub-sweep's own
        # starting point (see `guard_baseline`'s own docstring).
        baseline_components, baseline_coverage = guard_baseline()

        density_candidates = cand.bundle_density_fraction_candidates(
            self.raw_mask, int(self.current["skeleton_bundle_scan_size"]),
            float(self.current["skeleton_bundle_density_fraction"]),
            voxel_size_zyx=self.voxel_size_zyx,
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
    def _gap_distances_finest_voxels(self, *, z_distance_weight: float = 1.0) -> np.ndarray:
        """Gaps in the unit the closing radius, the bridge gap size and the
        maximum bridge distance are all read in: physical distance, counted in
        voxels of the finest axis. Only the maximum bridge distance is
        compared with ``skeleton_bridge_z_distance_weight`` on top, so only it
        passes *z_distance_weight*. Counted in plain voxels, a gap of two 2 um
        slices offered a closing radius of 2 -- 1 um, which never reaches
        across a slice."""
        spacing = np.asarray(self.voxel_size_zyx, dtype=float)
        return preprocessing.inter_component_gap_distances(
            self.current_skeleton,
            self._connectivity(),
            voxel_size_zyx=tuple(spacing / float(spacing.min())),
            z_distance_weight=float(z_distance_weight),
        )

    def _fusion_cost(self, cleaned: np.ndarray) -> float:
        signal = met.gap_vs_fusion_signal(
            self.current_skeleton, cleaned, voxel_size_zyx=self.voxel_size_zyx,
            component_connectivity=self._connectivity(), typical_radius_um=self.typical_radius_um,
        )
        penalty = _GUARD_PENALTY if signal.largest_gap_bridged_ratio > _GAP_FUSION_RATIO_GUARD else 0.0
        coverage = self._skeleton_mask_coverage_fraction(cleaned)
        penalty += self._regression_penalty(
            coverage, self._fusion_baseline_coverage, _MAX_COVERAGE_FRACTION_REGRESSION,
        )
        self._note(
            components_merged=signal.components_merged,
            largest_gap_bridged_ratio=signal.largest_gap_bridged_ratio,
            mask_coverage=coverage,
            guard_penalty=penalty,
        )
        return -float(signal.components_merged) + penalty

    def _group_closing_radius(self) -> None:
        group = "closing_radius"
        self._fusion_baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)
        candidate_values = cand.closing_radius_candidates(
            self._gap_distances_finest_voxels(), int(self.current["skeleton_closing_radius"])
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
        gaps = self._gap_distances_finest_voxels()
        bridge_candidates = cand.bridge_gap_size_candidates(gaps, int(self.current["skeleton_bridge_gap_size"]))

        def cost_bridge(value: int) -> float:
            cleaned = self._preprocess_trial({"skeleton_bridge_gap_size": value})
            return self._fusion_cost(cleaned)

        self._sweep(group, "skeleton_bridge_gap_size", bridge_candidates, cost_bridge)
        self.current_skeleton = self._preprocess_trial({})
        # Refresh: `_fusion_cost` reads `self._fusion_baseline_coverage` as
        # shared instance state (see its own docstring), so the bridge-gap
        # sweep just above moving `self.current_skeleton` must be reflected
        # here before the max-bridge-distance sweep below judges its own
        # candidates against it (see project memory: "matched processing
        # stage").
        self._fusion_baseline_coverage = self._skeleton_mask_coverage_fraction(self.current_skeleton)

        gaps = self._gap_distances_finest_voxels(
            z_distance_weight=float(self.current["skeleton_bridge_z_distance_weight"])
        )
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
            coverage = self._skeleton_mask_coverage_fraction(cleaned)
            penalty = self._regression_penalty(
                coverage, baseline_coverage, _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            self._note(
                largest_fraction=stats.largest_fraction, mask_coverage=coverage, guard_penalty=penalty,
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
            self._note(largest_fraction=stats.largest_fraction, guard_penalty=penalty)
            return -stats.largest_fraction + penalty

        self._sweep(group, "skeleton_min_component_percent", percent_candidates, cost_percent)
        self.current_skeleton = self._preprocess_trial({})

    # -- group 7: reconnect thresholds -------------------------------------------------
    def _stub_radius_sampler(self):
        """The mask's radius at a position, built once for every graph this
        search builds (see ``graph.assemble.mask_radius_sampler``)."""
        cached = getattr(self, "_stub_radius_at", None)
        # Rebuilt if segmentation cleanup has since replaced the mask.
        if cached is None or cached[0] is not self.raw_mask:
            cached = (
                self.raw_mask,
                graph_mod.assemble.mask_radius_sampler(self.raw_mask, self.voxel_size_zyx, 1.0),
            )
            self._stub_radius_at = cached
        return cached[1]

    def _build_graph(self, overrides: Mapping[str, Any]):
        kwargs = _graph_kwargs({**self.current, **overrides}, self.voxel_size_zyx)
        radius_multiple = kwargs["min_stub_length_radius_multiple"]
        return self._graph_cache.get_or_compute(
            # The mask too: the stub-radius sampler and the duplicate-edge
            # check read it.
            (self.current_skeleton, self.raw_mask),
            kwargs,
            lambda: graph_mod.build_graph_from_skeleton(
                self.current_skeleton,
                stub_radius_at=self._stub_radius_sampler() if radius_multiple > 0 else None,
                segmentation_mask=self.raw_mask,
                **kwargs,
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

        def topology_cost(G) -> float:
            coverage = self._skeleton_graph_coverage_fraction(G)
            penalty = self._regression_penalty(
                coverage, baseline_graph_coverage, _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            topology = met.graph_topology_metrics(G)
            self._note(
                n_components=topology.n_components,
                self_loops=topology.n_selfloops,
                isolated_nodes=topology.n_isolates,
                skeleton_coverage=coverage,
                guard_penalty=penalty,
            )
            return topology.score + penalty

        def cost_reconnect(value: float) -> float:
            return topology_cost(self._build_graph({"graph_reconnect_threshold": value}))

        self._sweep(group, "graph_reconnect_threshold", reconnect_candidates, cost_reconnect)

        # Refresh: the reconnect-threshold sweep above may have moved
        # self.current["graph_reconnect_threshold"], which this baseline's
        # own self._build_graph({}) call must reflect before the
        # orphan-threshold sweep below judges its candidates against it
        # (see project memory: "matched processing stage").
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self._build_graph({}))

        orphan_candidates = cand.orphan_threshold_candidates(
            float(self.current["graph_reconnect_threshold"]), float(self.current["final_orphan_reconnect_threshold"])
        )

        def cost_orphan(value: float) -> float:
            return topology_cost(self._build_graph({"final_orphan_reconnect_threshold": value}))

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
        """Collapsing a cluster of nearby nodes into one representative is
        exactly the mechanism `graph.cartwheel_guard`'s own module docstring
        names as the known cause of a "cartwheel" hub: a real vessel
        junction's daughters continue in some coherent direction, and a
        collapse distance too generous for this graph produces a
        representative with edges radiating every which way instead --
        geometrically valid (no fragmentation, no coverage loss), but not a
        plausible vessel junction. `graph_topology_metrics` alone cannot see
        this (it only counts nodes/edges/components/self-loops/degree-2
        nodes, none of which a cartwheel hub necessarily moves), so a
        candidate that introduces new ones is guarded the same way a
        candidate that introduces new components already is.
        """
        group = "cluster_collapse_distance"
        baseline = met.graph_topology_metrics(self.current_graph)
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self.current_graph)
        baseline_hub_count = self._cartwheel_hub_count(self.current_graph)

        def cost_for(key: str, value: Any) -> float:
            G = self._build_graph({key: value})
            metrics = met.graph_topology_metrics(G)
            coverage = self._skeleton_graph_coverage_fraction(G)
            hubs = self._cartwheel_hub_count(G)
            fragmentation_penalty = _GUARD_PENALTY * max(0, metrics.n_components - baseline.n_components)
            fragmentation_penalty += self._regression_penalty(
                coverage, baseline_graph_coverage, _MAX_COVERAGE_FRACTION_REGRESSION,
            )
            fragmentation_penalty += _GUARD_PENALTY * max(0, hubs - baseline_hub_count)
            self._note(
                degree2_nodes=metrics.total_degree2,
                n_components=metrics.n_components,
                skeleton_coverage=coverage,
                cartwheel_hubs=hubs,
                guard_penalty=fragmentation_penalty,
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
        # Refresh: the distance sweep above may have moved
        # self.current["cluster_collapse_distance"], which `cost_for`'s own
        # baseline must reflect before the method sweep below judges its
        # candidates against it (see project memory: "matched processing
        # stage").
        baseline = met.graph_topology_metrics(self.current_graph)
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self.current_graph)
        baseline_hub_count = self._cartwheel_hub_count(self.current_graph)

        self._sweep(
            group, "cluster_collapse_method", cand.cluster_collapse_method_candidates(),
            lambda value: cost_for("cluster_collapse_method", value),
        )
        self.current_graph = self._build_graph({})
        # Refresh again for the same reason, ahead of the method-specific
        # knob sweep below.
        baseline = met.graph_topology_metrics(self.current_graph)
        baseline_graph_coverage = self._skeleton_graph_coverage_fraction(self.current_graph)
        baseline_hub_count = self._cartwheel_hub_count(self.current_graph)

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
        # Whichever threshold a real run would judge stubs by: a multiple of
        # the parent vessel's radius when that is on, else the fixed length.
        by_radius = float(self.current.get("min_stub_length_radius_multiple", 0.0) or 0.0) > 0
        setting = "min_stub_length_radius_multiple" if by_radius else "min_stub_length"
        if by_radius:
            candidate_values = sorted(
                {0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, float(self.current[setting])}
            )
        else:
            terminal_lengths = cand.terminal_edge_lengths_um(self.current_graph)
            candidate_values = cand.min_stub_length_candidates(
                terminal_lengths, float(self.current["min_stub_length"])
            )
        unpruned = self._build_graph(
            {"min_stub_length": 0.0, "min_stub_length_radius_multiple": 0.0}
        )
        baseline_nodes = unpruned.number_of_nodes()
        baseline_length = met.total_edge_length(unpruned)

        def cost(value: float) -> float:
            G = self._build_graph({setting: value})
            stubs_removed = max(0, baseline_nodes - G.number_of_nodes())
            length_removed = max(0.0, baseline_length - met.total_edge_length(G))
            self._note(
                nodes_removed=stubs_removed,
                length_removed_fraction=length_removed / baseline_length if baseline_length > 0 else 0.0,
            )
            if baseline_length > 0 and length_removed / baseline_length > _MAX_GRAPH_LENGTH_REMOVED_FRACTION:
                return _GUARD_PENALTY
            if length_removed <= 0:
                return 0.0
            return -(stubs_removed / length_removed)

        self._sweep(group, setting, candidate_values, cost)
        self.current_graph = self._build_graph({})

    # -- group 10: centreline smoothing ------------------------------------------------
    def _group_centreline_smoothing(self) -> None:
        group = "centreline_smoothing"
        if not bool(self.current.get("smooth_centrelines", True)):
            return
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

        def cost(combo: tuple[str, int, float]) -> float:
            method, iterations, deviation = combo
            trial_graph = self.current_graph.copy()
            counts = graph_mod.smooth_graph_centrelines(
                trial_graph, self.current_skeleton, voxel_size_zyx=self.voxel_size_zyx,
                method=method, iterations=iterations, max_deviation=deviation,
            )
            quality = met.smoothing_quality(counts, length_before, met.total_edge_length(trial_graph))
            penalty = _GUARD_PENALTY if quality.length_shrink_fraction > _MAX_LENGTH_SHRINK_FRACTION else 0.0
            self._note(
                smoothed_fraction=quality.smoothed_fraction,
                length_shrink_fraction=quality.length_shrink_fraction,
                guard_penalty=penalty,
            )
            return -quality.smoothed_fraction + penalty

        best_combo = self._run_candidates(
            group, "centreline_smoothing", combos, cost,
            incumbent=(
                self.current["centreline_smoothing_method"],
                int(self.current["centreline_smoothing_iterations"]),
                float(self.current["centreline_max_deviation"]),
            ),
        )
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
    def run(self, *, max_passes: int = 1) -> None:
        """Run the fixed group sequence, optionally repeating it to convergence.

        Coordinate descent -- tuning one setting (or small joint group) at a
        time, holding every other setting at its current value -- cannot see
        an interaction where two settings only help *together* (e.g. a
        segmentation-cleanup step that only wants a smaller closing radius
        once another step is also on): the single pass this module was built
        around decides each setting once, in a fixed order, and never
        revisits it. *max_passes* > 1 repeats the whole sequence, each pass
        starting from where the previous one left `self.current`, using this
        codebase's own established "always re-derive from source, never from
        a previous trial's own output" discipline (`_cleanup_trial` re-cleans
        `self.original_raw_mask`, `_preprocess_trial` re-runs from
        `self.raw_skeleton`) -- the same discipline that already makes one
        setting's own sweep safe to re-run is what makes re-running the
        *whole* sequence safe too. Stops as soon as a pass changes nothing
        (converged: another pass would just repeat it), so a dataset with no
        real cross-group interaction pays for at most one extra, cheap
        confirmation pass, not the full budget every time.
        """
        max_passes = max(1, int(max_passes))
        if max_passes > 1:
            self._group_total = _GROUP_TOTAL_UPPER_BOUND * max_passes
        for _pass_index in range(max_passes):
            before = dict(self.current)
            self.passes_run += 1
            self._run_one_pass()
            if self.current == before:
                break

    def _run_one_pass(self) -> None:
        if not self.raw_mask.any():
            self.raw_skeleton = self.raw_mask.copy()
            self.current_skeleton = self.raw_skeleton
            return

        if self._run_group("segmentation_cleanup", self._group_segmentation_cleanup):
            if not self.raw_mask.any():
                self.raw_skeleton = self.raw_mask.copy()
                self.current_skeleton = self.raw_skeleton
                return

        if not self._run_group("thick_vessel_gating", self._group_thick_vessel_gating):
            self.raw_skeleton = self._raw_skeleton_trial({})

        for name, method in (
            ("min_branch_length", self._group_min_branch_length),
            ("bundle_refinement", self._group_bundle_refinement),
            ("closing_radius", self._group_closing_radius),
            ("gap_bridging", self._group_gap_bridging),
            ("connectivity_and_component_filter", self._group_connectivity_and_component_filter),
        ):
            self._run_group(name, method)
        # Whichever of the skeleton-cleaning groups ran -- or none did, every
        # one deselected -- this makes sure `current_skeleton` reflects every
        # decision actually in `self.current`, cheaply (one more real call).
        self.current_skeleton = self._preprocess_trial({})

        if not self.current_skeleton.any():
            return

        if not self._run_group("reconnect_thresholds", self._group_reconnect_thresholds):
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
            self._run_group(name, method)


def _to_search_grid(starting_values: Mapping[str, Any], factor: int) -> dict[str, Any]:
    """*starting_values* with every voxel-counted setting
    (:data:`_VOXEL_SCALED_SETTING_NAMES`) in voxels of a search grid
    *factor* times coarser, rounded half up.

    Regression: they went in unchanged, so a closing radius of 2 was searched
    as 2 voxels of a 16x grid -- 32 full-resolution voxels -- and, kept,
    came back as 32."""
    values = dict(starting_values)
    if factor > 1:
        for name in _VOXEL_SCALED_SETTING_NAMES:
            if values.get(name) is not None:
                values[name] = int(math.floor(float(values[name]) / factor + 0.5))
    return values


def _to_full_resolution(
    settings: Mapping[str, Any],
    starting_values: Mapping[str, Any],
    search_starting_values: Mapping[str, Any],
    factor: int,
    *,
    bundle_refinement_ran: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """*settings*, decided on a grid *factor* times coarser, with every
    voxel-counted setting back in full-resolution voxels -- and the ones
    left as they were because they are finer than the search grid.

    A setting the search kept is returned as the user had it, not as its
    round trip through the coarse grid (2 voxels at 16x is 0, and 0 back is
    0); one the search moved is its search-grid value times *factor*. The
    bundle hub spacing is derived again from the final scan size, as a
    full-resolution search derives it.
    """
    settings = dict(settings)
    finer: list[str] = []
    for name in _VOXEL_SCALED_SETTING_NAMES:
        if settings.get(name) is None:
            continue
        start = starting_values.get(name)
        if start is not None and settings[name] == search_starting_values.get(name):
            settings[name] = start
            if start and not search_starting_values[name]:
                finer.append(name)
        else:
            settings[name] = int(round(settings[name] * factor))
    if bundle_refinement_ran and settings.get("skeleton_bundle_scan_size") is not None:
        settings["skeleton_bundle_hub_min_spacing"] = cand.bundle_hub_min_spacing_for(
            int(settings["skeleton_bundle_scan_size"])
        )
        finer = [name for name in finer if name != "skeleton_bundle_hub_min_spacing"]
    return settings, tuple(finer)


def optimise_skeleton_and_graph_settings(
    raw_mask: np.ndarray,
    *,
    voxel_size_xyz: tuple[float, float, float],
    starting_values: Mapping[str, Any],
    progress: Optional[ProgressCallback] = None,
    downsample_factor: Optional[int] = None,
    groups: Optional[Iterable[str]] = None,
    raw_image: Optional[np.ndarray] = None,
    max_passes: int = 1,
    auto_downsample_target_seconds: float = DEFAULT_AUTO_DOWNSAMPLE_TARGET_SECONDS,
) -> OptimisationResult:
    """Empirically choose every Skeletonise/Graph tab setting for *raw_mask*.

    *starting_values* must have an entry for every name in
    :data:`OPTIMISE_SETTING_NAMES` -- the settings this run does not manage to
    improve on simply keep their starting value. See the module docstring for
    the search strategy.

    *downsample_factor* trades accuracy for speed on a large volume: ``None``
    (the default) auto-detects a factor estimated to keep the whole search
    under *auto_downsample_target_seconds*, and never coarser than one that
    still resolves the mask's typical vessel, via
    :func:`estimate_downsample_factor_for_time_budget` (which times real
    evaluations on *this* mask rather than assuming a fixed
    voxels-per-second rate -- see its own docstring), ``1`` runs at full
    resolution, and ``2``/``4``/``8``/``16`` (:data:`DOWNSAMPLE_FACTORS`)
    run the whole search on a block-max-reduced copy of *raw_mask* with
    *voxel_size_xyz* scaled up to match. Micron-based settings come back
    unaffected by this (the scaled voxel size already accounts for it);
    the handful of settings measured in voxels
    (:data:`_VOXEL_SCALED_SETTING_NAMES`) go into the search in its own
    grid's voxels and come back in full-resolution ones -- as they were,
    when the search kept them (see :func:`_to_full_resolution`) -- so they
    are correct against the full-resolution volume a real run actually
    skeletonises.

    *auto_downsample_target_seconds* is only consulted when
    *downsample_factor* is ``None`` -- the runtime budget "Auto" aims for,
    default five minutes.

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

    *max_passes* repeats the whole group sequence (coordinate descent) up to
    this many times, each pass starting from where the previous one left
    every setting, to catch an improvement that only shows up once two
    settings have *both* moved from their starting values -- a single pass
    (the default, ``1``, today's exact behaviour) decides each setting once
    and never revisits it. Stops as soon as a pass changes nothing, so a
    dataset with no such interaction costs at most one extra, cheap
    confirmation pass, not the full budget every time; see
    :meth:`_Search.run`'s own docstring for why repeating the sequence is
    safe. `OptimisationResult.passes_run` reports how many actually ran.
    """
    raw_mask = np.asarray(raw_mask, dtype=bool)
    voxel_size_xyz = tuple(float(v) for v in voxel_size_xyz)
    voxel_size_zyx = tuple(reversed(voxel_size_xyz))
    auto: Optional[_AutoDownsample] = None
    if downsample_factor:
        factor = int(downsample_factor)
    else:
        auto = _auto_downsample(
            raw_mask,
            voxel_size_zyx,
            target_seconds=auto_downsample_target_seconds,
            starting_values=starting_values,
        )
        factor = auto.factor
    if factor not in DOWNSAMPLE_FACTORS:
        factor, auto = 1, None
    axis_factors = axis_downsample_factors(factor, voxel_size_zyx)

    if max(axis_factors) > 1:
        search_mask = _downsample_mask(raw_mask, factor, voxel_size_zyx)
        search_voxel_size_xyz = tuple(
            v * f for v, f in zip(voxel_size_xyz, reversed(axis_factors))
        )
        search_raw_image = (
            _downsample_intensity(np.asarray(raw_image), factor, voxel_size_zyx)
            if raw_image is not None
            else None
        )
    else:
        search_mask = raw_mask
        search_voxel_size_xyz = voxel_size_xyz
        search_raw_image = raw_image

    # Voxel counts of the finest axis, which the search grid reduces by the
    # full factor: into its voxels on the way in, back out on the way out.
    search_starting_values = _to_search_grid(starting_values, factor)
    search = _Search(
        search_mask, search_voxel_size_xyz, search_starting_values, progress,
        enabled_groups=groups, raw_image=search_raw_image,
    )
    # The scorecard's two ends, on the search's own grid: before any sweep
    # moves anything, and after the last.
    input_radius_map: Optional[np.ndarray] = None

    def measured(settings: Mapping[str, Any]) -> dict[str, float]:
        nonlocal input_radius_map
        try:
            if input_radius_map is None and search_mask.any():
                input_radius_map = preprocessing.inscribed_radius_map(
                    search_mask, tuple(reversed(search_voxel_size_xyz))
                )
            return _scorecard_measures(
                search_mask, tuple(reversed(search_voxel_size_xyz)), settings,
                input_radius_map=input_radius_map,
            )
        except Exception:  # noqa: BLE001 - a scorecard must never fail the search it reports on
            return {}

    before = measured({**starting_values, **search_starting_values})
    start = time.perf_counter()
    search.run(max_passes=max_passes)
    seconds = time.perf_counter() - start
    after = measured({**starting_values, **search.current})
    settings = {name: search.current[name] for name in OPTIMISE_SETTING_NAMES if name in search.current}
    finer_than_search_grid: tuple[str, ...] = ()
    if factor > 1:
        settings, finer_than_search_grid = _to_full_resolution(
            settings, starting_values, search_starting_values, factor,
            bundle_refinement_ran="bundle_refinement" in search.groups_run,
        )
    return OptimisationResult(
        settings=settings,
        trials=tuple(search.trials),
        downsample_factor=factor,
        groups_run=tuple(dict.fromkeys(search.groups_run)),
        passes_run=search.passes_run,
        downsample_factors_zyx=axis_factors,
        seconds=seconds,
        estimated_seconds=auto.estimated_seconds if auto is not None else None,
        group_seconds=dict(search.group_seconds),
        typical_radius_um=auto.typical_radius_um if auto is not None else None,
        resolution_cap=auto.resolution_cap if auto is not None else None,
        finer_than_search_grid=finer_than_search_grid,
        scorecard=_scorecard(before, after),
    )
