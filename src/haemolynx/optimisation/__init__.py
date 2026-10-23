"""Empirically choosing pipeline settings from real image/graph data.

Given a segmented binary mask, :func:`optimise_skeleton_and_graph_settings`
runs the real preprocessing/graph-building code with a sequence of
image-derived candidate values, scores each with :mod:`.metrics`, and returns
the winning value for every setting on the Skeletonise and Graph tabs.
:func:`.fwhm_search.optimise_fwhm_settings` is the same coordinate-descent
pattern applied to the FWHM diameter-measurement settings on the Diameters
tab: it measures a representative sample of the graph's own vessels
through the run's own FWHM call,
:func:`haemolynx.haemodynamics.apply.measure_fwhm_diameters`, scored with
:mod:`.fwhm_metrics`, instead of a proxy metric.

The skeleton/graph search, over its eleven groups (:data:`GROUP_NAMES`),
depends only on ``numpy``, ``scipy``, ``networkx``, and the sibling
``haemolynx.preprocessing`` / ``haemolynx.graph`` / ``haemolynx.io``
subpackages -- the last for two of the run's own steps, never to read a file:
the block reduction a search grid is made by (``io.downsample``) and the
binarise-skeletonise-fill every loader skeletonises with (``io.load``). It
never imports ``haemolynx.pipeline`` or ``haemolynx.gui``, which hand it the
run's mask, grid and settings, so it is testable with tiny synthetic arrays
under plain pytest, no disk I/O and no GUI toolkit. The FWHM search, over its
eight groups (:data:`FWHM_GROUP_NAMES`), adds ``haemolynx.haemodynamics``
(``apply``: the run's own measurement calls and haemodynamics config) and
``io.load``'s binarisation of the segmentation -- still never
``haemolynx.pipeline`` or ``haemolynx.gui``, which hand it the run's config,
so the same "small synthetic array, no GUI, no disk I/O beyond a throwaway
tiff" testability holds for it too.
"""
from .fwhm_metrics import (
    FwhmMeasurementQuality,
    fwhm_measurement_quality,
    regression_penalty as fwhm_regression_penalty,
    rejection_gates_score,
)
from .fwhm_search import (
    DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    FWHM_GROUP_LABELS,
    FWHM_GROUP_NAMES,
    FWHM_SETTING_NAMES,
    estimate_sample_edge_count_for_time_budget,
    optimise_fwhm_settings,
)
from .metrics import (
    GapFusionSignal,
    GraphTopologyMetrics,
    SmoothingQuality,
    braid_factor_along_long_axis,
    gap_vs_fusion_signal,
    graph_topology_metrics,
    smoothing_quality,
    total_edge_length,
)
from .progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
    KINDS,
    OptimisationEvent,
)
from .report import build_report_text, config_filename
from .search import (
    AUTO_DOWNSAMPLE_TARGET_VOXELS,
    DEFAULT_AUTO_DOWNSAMPLE_TARGET_SECONDS,
    DOWNSAMPLE_FACTORS,
    GRAPH_SETTING_NAMES,
    GROUP_LABELS,
    GROUP_NAMES,
    OPTIMISE_SETTING_NAMES,
    SKELETON_SETTING_NAMES,
    OptimisationResult,
    TrialRecord,
    estimate_downsample_factor_for_time_budget,
    full_resolution_search_refusal,
    optimise_skeleton_and_graph_settings,
    resolve_auto_downsample_factor,
)

__all__ = [
    "FwhmMeasurementQuality",
    "fwhm_measurement_quality",
    "fwhm_regression_penalty",
    "rejection_gates_score",
    "DEFAULT_AUTO_SAMPLE_TARGET_SECONDS",
    "FWHM_GROUP_LABELS",
    "FWHM_GROUP_NAMES",
    "FWHM_SETTING_NAMES",
    "estimate_sample_edge_count_for_time_budget",
    "optimise_fwhm_settings",
    "GapFusionSignal",
    "GraphTopologyMetrics",
    "SmoothingQuality",
    "braid_factor_along_long_axis",
    "gap_vs_fusion_signal",
    "graph_topology_metrics",
    "smoothing_quality",
    "total_edge_length",
    "CANDIDATE_EVALUATED",
    "GROUP_FINISHED",
    "GROUP_STARTED",
    "KINDS",
    "OptimisationEvent",
    "build_report_text",
    "config_filename",
    "AUTO_DOWNSAMPLE_TARGET_VOXELS",
    "DEFAULT_AUTO_DOWNSAMPLE_TARGET_SECONDS",
    "DOWNSAMPLE_FACTORS",
    "GRAPH_SETTING_NAMES",
    "GROUP_LABELS",
    "GROUP_NAMES",
    "OPTIMISE_SETTING_NAMES",
    "SKELETON_SETTING_NAMES",
    "OptimisationResult",
    "TrialRecord",
    "estimate_downsample_factor_for_time_budget",
    "full_resolution_search_refusal",
    "optimise_skeleton_and_graph_settings",
    "resolve_auto_downsample_factor",
]
