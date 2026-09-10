"""Empirically choosing Skeletonise/Graph pipeline settings from a segmented image.

Given a segmented binary mask, :func:`optimise_skeleton_and_graph_settings`
runs the real preprocessing/graph-building code with a sequence of
image-derived candidate values, scores each with :mod:`.metrics`, and returns
the winning value for every setting on the Skeletonise and Graph tabs.

Depends only on ``numpy``, ``scipy``, ``networkx``, and the sibling
``haemolynx.preprocessing`` / ``haemolynx.graph`` / ``haemolynx.statistics``
subpackages -- never on ``haemolynx.pipeline``, ``haemolynx.io``, or
``haemolynx.gui`` -- so the whole search is testable with tiny synthetic arrays
under plain pytest, no disk I/O and no GUI toolkit.
"""
from .metrics import (
    GapFusionSignal,
    GraphTopologyMetrics,
    SmoothingQuality,
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
    GRAPH_SETTING_NAMES,
    OPTIMISE_SETTING_NAMES,
    SKELETON_SETTING_NAMES,
    OptimisationResult,
    TrialRecord,
    optimise_skeleton_and_graph_settings,
)

__all__ = [
    "GapFusionSignal",
    "GraphTopologyMetrics",
    "SmoothingQuality",
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
    "GRAPH_SETTING_NAMES",
    "OPTIMISE_SETTING_NAMES",
    "SKELETON_SETTING_NAMES",
    "OptimisationResult",
    "TrialRecord",
    "optimise_skeleton_and_graph_settings",
]
