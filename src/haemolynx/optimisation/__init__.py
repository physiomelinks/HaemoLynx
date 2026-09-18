"""Empirically choosing pipeline settings from real image/graph data.

Given a segmented binary mask, :func:`optimise_skeleton_and_graph_settings`
runs the real preprocessing/graph-building code with a sequence of
image-derived candidate values, scores each with :mod:`.metrics`, and returns
the winning value for every setting on the Skeletonise and Graph tabs.
:func:`.fwhm_search.optimise_fwhm_settings` is the same coordinate-descent
pattern applied to the FWHM diameter-measurement settings on the Diameters
tab: it runs the real
:func:`haemolynx.haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`
on a representative sample of the graph's own edges, scored with
:mod:`.fwhm_metrics`, instead of a proxy metric.

The skeleton/graph search depends only on ``numpy``, ``scipy``, ``networkx``,
and the sibling ``haemolynx.preprocessing`` / ``haemolynx.graph`` /
``haemolynx.statistics`` subpackages -- never on ``haemolynx.pipeline``,
``haemolynx.io``, or ``haemolynx.gui`` -- so it is testable with tiny
synthetic arrays under plain pytest, no disk I/O and no GUI toolkit. The FWHM
search adds one more dependency to that list, on
``haemolynx.haemodynamics.automated`` specifically (the one real measurement
function it calls) -- still never on ``haemolynx.pipeline``/``haemolynx.io``/
``haemolynx.gui``, so the same "small synthetic array, no GUI, no disk I/O
beyond a throwaway tiff" testability holds for it too.
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
    "optimise_skeleton_and_graph_settings",
    "resolve_auto_downsample_factor",
]
