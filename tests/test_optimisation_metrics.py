"""Unit tests for haemolynx.optimisation.metrics -- pure, synthetic data only."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.optimisation.metrics import (
    GapFusionSignal,
    GraphTopologyMetrics,
    gap_vs_fusion_signal,
    graph_topology_metrics,
    smoothing_quality,
    total_edge_length,
)


def test_graph_topology_metrics_on_empty_graph():
    metrics = graph_topology_metrics(nx.MultiGraph())
    assert metrics == GraphTopologyMetrics(0, 0, 0, 0, 0, 0)
    assert metrics.score == 0.0


def test_graph_topology_metrics_counts_components_selfloops_isolates():
    G = nx.MultiGraph()
    # Component 1: a 3-node path A-B-C (B has degree 2).
    G.add_edge("A", "B", length=1.0)
    G.add_edge("B", "C", length=1.0)
    # Component 2: an isolated node.
    G.add_node("D")
    # Component 3: a self-loop plus one real edge.
    G.add_edge("E", "E", length=1.0)
    G.add_edge("E", "F", length=1.0)

    metrics = graph_topology_metrics(G)
    assert metrics.n_nodes == 6
    assert metrics.n_components == 3
    assert metrics.n_selfloops == 1
    assert metrics.n_isolates == 1
    assert metrics.total_degree2 == 1  # only B
    assert metrics.score == pytest.approx(2 + 5 * 1 + 0.5 * 1)


def test_gap_vs_fusion_signal_no_merge_when_nothing_added():
    raw = np.zeros((1, 1, 20), dtype=bool)
    raw[0, 0, 0] = True
    raw[0, 0, 10] = True
    cleaned = raw.copy()  # identical: nothing bridged
    signal = gap_vs_fusion_signal(raw, cleaned, component_connectivity=1)
    assert signal.components_before == 2
    assert signal.components_after == 2
    assert signal.largest_gap_bridged_um == 0.0
    assert signal.components_merged == 0


def test_gap_vs_fusion_signal_measures_a_real_bridge():
    raw = np.zeros((1, 1, 20), dtype=bool)
    raw[0, 0, 0] = True
    raw[0, 0, 10] = True
    cleaned = raw.copy()
    cleaned[0, 0, 1:10] = True  # bridge drawn between the two raw components
    signal = gap_vs_fusion_signal(
        raw, cleaned, component_connectivity=1, typical_radius_um=1.0
    )
    assert signal.components_before == 2
    assert signal.components_after == 1
    assert signal.components_merged == 1
    assert signal.largest_gap_bridged_um == pytest.approx(10.0)
    assert signal.largest_gap_bridged_ratio == pytest.approx(10.0 / 2.0)


def test_gap_vs_fusion_signal_single_component_is_a_no_op():
    raw = np.zeros((1, 1, 10), dtype=bool)
    raw[0, 0, 2:5] = True
    signal = gap_vs_fusion_signal(raw, raw.copy(), component_connectivity=1)
    assert signal == GapFusionSignal(1, 1, 0.0, 0.0)


def test_smoothing_quality_fraction_and_shrink():
    counts = {"smoothed": 6, "relaxed": 2, "kept_raw": 1, "too_short": 1}
    quality = smoothing_quality(counts, length_before_um=100.0, length_after_um=93.0)
    assert quality.total == 10
    assert quality.smoothed_fraction == pytest.approx(0.8)
    assert quality.length_shrink_fraction == pytest.approx(0.07)


def test_smoothing_quality_handles_zero_length_before():
    quality = smoothing_quality({}, length_before_um=0.0, length_after_um=0.0)
    assert quality.total == 0
    assert quality.smoothed_fraction == 0.0
    assert quality.length_shrink_fraction == 0.0


def test_total_edge_length_sums_present_lengths_only():
    G = nx.MultiGraph()
    G.add_edge("A", "B", length=3.0)
    G.add_edge("B", "C", length=4.5)
    G.add_edge("C", "D")  # no length attribute
    assert total_edge_length(G) == pytest.approx(7.5)
