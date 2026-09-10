"""Unit tests for haemolynx.optimisation.candidates -- synthetic data only.

Every generator must (a) always include the schema default, and (b) degrade to
``[default]`` on pathological input rather than raising.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.optimisation import candidates as c


# ---------------------------------------------------------------------------
# Group 1: thick-vessel gating
# ---------------------------------------------------------------------------
def test_thick_vessel_worth_checking_false_for_thin_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[4:6, 4:6, :] = True  # a 2-voxel-radius rod, well under the guard
    assert c.thick_vessel_worth_checking(mask, (1.0, 1.0, 1.0)) is False


def test_thick_vessel_worth_checking_true_for_fat_mask():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:18, 2:18, 2:18] = True  # a big solid block, inscribed radius > 2um
    assert c.thick_vessel_worth_checking(mask, (1.0, 1.0, 1.0)) is True


def test_thick_vessel_min_radius_candidates_empty_mask_falls_back_to_default():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert c.thick_vessel_min_radius_candidates(empty, (1.0, 1.0, 1.0), default=6.0) == [6.0]


def test_thick_vessel_min_radius_candidates_include_default():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:18, 2:18, 2:18] = True
    result = c.thick_vessel_min_radius_candidates(mask, (1.0, 1.0, 1.0), default=6.0)
    assert 6.0 in result
    assert all(v > 0 for v in result)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# Group 2: min branch length
# ---------------------------------------------------------------------------
def test_min_branch_length_candidates_empty_falls_back_to_default_and_zero():
    result = c.min_branch_length_candidates((), default=3)
    assert result == [0, 3]


def test_min_branch_length_candidates_respects_ceiling():
    sizes = tuple(range(1, 200))
    result = c.min_branch_length_candidates(sizes, default=3, ceiling=20)
    assert all(0 <= v <= 20 for v in result)
    assert 3 in result
    assert 0 in result


# ---------------------------------------------------------------------------
# Group 3: bundle refinement
# ---------------------------------------------------------------------------
def test_bundle_scan_size_candidates_empty_mask_falls_back_to_default():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert c.bundle_scan_size_candidates(empty, (1.0, 1.0, 1.0), default=9) == [9]


def test_bundle_scan_size_candidates_are_odd_and_bounded():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:25, 5:25, 5:25] = True
    result = c.bundle_scan_size_candidates(mask, (1.0, 1.0, 1.0), default=9)
    assert 9 in result
    assert all(5 <= v <= 25 for v in result)
    assert all(v % 2 == 1 for v in result)


def test_bundle_density_fraction_candidates_empty_mask_falls_back_to_default():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert c.bundle_density_fraction_candidates(empty, scan_size=9, default=0.35) == [0.35]


def test_bundle_density_fraction_candidates_bounded_unit_interval():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    result = c.bundle_density_fraction_candidates(mask, scan_size=5, default=0.35)
    assert 0.35 in result
    assert all(0.0 < v <= 1.0 for v in result)


def test_estimate_bundle_max_connections_on_empty_skeleton_returns_default():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert c.estimate_bundle_max_connections(empty) == 8


def test_estimate_bundle_max_connections_on_a_simple_line_is_low():
    skel = np.zeros((10, 10, 10), dtype=bool)
    skel[2:8, 5, 5] = True
    result = c.estimate_bundle_max_connections(skel)
    assert 4 <= result <= 12


def test_bundle_hub_min_spacing_for_matches_documented_formula():
    assert c.bundle_hub_min_spacing_for(9) == 4
    assert c.bundle_hub_min_spacing_for(1) == 1
    assert c.bundle_hub_min_spacing_for(0) == 1


# ---------------------------------------------------------------------------
# Groups 4-5: closing radius, gap bridging
# ---------------------------------------------------------------------------
def test_closing_radius_candidates_empty_gaps_falls_back_to_defaults():
    result = c.closing_radius_candidates(np.array([]), default=2)
    assert result == [0, 1, 2]


def test_closing_radius_candidates_respect_cap():
    gaps = np.array([100.0, 200.0])
    result = c.closing_radius_candidates(gaps, default=2, cap=5)
    assert all(v <= 5 for v in result)


def test_bridge_gap_size_candidates_empty_falls_back_to_default_and_zero():
    result = c.bridge_gap_size_candidates(np.array([]), default=3)
    assert result == [0, 3]


def test_max_bridge_distance_candidates_include_percentiles():
    gaps = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    result = c.max_bridge_distance_candidates(gaps, default=4)
    assert 4 in result
    assert max(result) >= int(round(np.percentile(gaps, 90)))


# ---------------------------------------------------------------------------
# Group 6: connectivity + component percent
# ---------------------------------------------------------------------------
def test_component_connectivity_candidates_are_fixed():
    assert c.component_connectivity_candidates() == [1, 2, 3]


def test_min_component_percent_candidates_finds_the_elbow():
    # A dominant component (1000) plus a cluster of tiny noise components.
    sizes = (1000, 900, 5, 4, 3, 2, 1)
    result = c.min_component_percent_candidates(sizes, voxel_count=sum(sizes), default=0.0)
    assert 0.0 in result
    assert 1.0 in result
    assert 5.0 in result
    assert all(0.0 <= v <= 100.0 for v in result)


def test_min_component_percent_candidates_handles_single_component():
    result = c.min_component_percent_candidates((100,), voxel_count=100, default=0.0)
    assert result == [0.0, 1.0, 5.0]


# ---------------------------------------------------------------------------
# Group 7: reconnect thresholds
# ---------------------------------------------------------------------------
def test_reconnect_threshold_candidates_empty_falls_back_to_default():
    assert c.reconnect_threshold_candidates(np.array([]), default=10.0) == [10.0]


def test_orphan_threshold_candidates_are_fractions_of_reconnect():
    result = c.orphan_threshold_candidates(reconnect_threshold=10.0, default=3.0)
    assert 3.0 in result
    assert pytest.approx(2.5) in result  # 0.25 * 10.0
    assert pytest.approx(5.0) in result  # 0.5 * 10.0
    assert pytest.approx(10.0) in result  # 1.0 * 10.0


# ---------------------------------------------------------------------------
# Group 8-9: cluster collapse, min stub length
# ---------------------------------------------------------------------------
def test_cluster_collapse_distance_candidates_empty_falls_back_to_default():
    assert c.cluster_collapse_distance_candidates(np.array([]), default=5.0) == [5.0]


def test_min_stub_length_candidates_empty_falls_back_to_default():
    assert c.min_stub_length_candidates(np.array([]), default=10.0) == [10.0]


# ---------------------------------------------------------------------------
# Group 10: centreline smoothing
# ---------------------------------------------------------------------------
def test_centreline_smoothing_method_candidates():
    assert c.centreline_smoothing_method_candidates() == ["taubin", "chaikin"]


def test_centreline_smoothing_iterations_candidates_for_taubin():
    assert c.centreline_smoothing_iterations_candidates("taubin") == [5, 10, 15, 20]


def test_centreline_smoothing_iterations_candidates_for_chaikin_stay_small():
    # Chaikin doubles its point count every pass, so candidates must stay
    # small enough that even the largest does not blow up.
    result = c.centreline_smoothing_iterations_candidates("chaikin")
    assert max(result) <= 6


def test_centreline_max_deviation_candidates_scale_with_voxel_size():
    result = c.centreline_max_deviation_candidates((2.0, 2.0, 2.0), default=1.0)
    assert 1.0 in result
    assert min(result) == pytest.approx(1.0)  # 0.5 voxel of 2.0um clipped by default tie
    assert max(result) == pytest.approx(8.0)  # 4 * 2.0um


# ---------------------------------------------------------------------------
# Graph measurement helpers
# ---------------------------------------------------------------------------
def _graph_with_two_components():
    G = nx.MultiGraph()
    G.add_node("A", pos=(0.0, 0.0, 0.0))
    G.add_node("B", pos=(0.0, 0.0, 1.0))
    G.add_edge("A", "B", length=1.0)
    G.add_node("C", pos=(0.0, 0.0, 11.0))
    G.add_node("D", pos=(0.0, 0.0, 12.0))
    G.add_edge("C", "D", length=1.0)
    return G


def test_graph_component_gap_distances_um_single_component_is_empty():
    G = nx.MultiGraph()
    G.add_node("A", pos=(0.0, 0.0, 0.0))
    G.add_node("B", pos=(0.0, 0.0, 1.0))
    G.add_edge("A", "B", length=1.0)
    assert c.graph_component_gap_distances_um(G).size == 0


def test_graph_component_gap_distances_um_measures_known_gap():
    G = _graph_with_two_components()
    gaps = c.graph_component_gap_distances_um(G)
    assert gaps.size == 1
    assert gaps[0] == pytest.approx(10.0)  # B at z=1 to C at z=11


def test_nearest_neighbour_node_distances_um_excludes_self():
    G = _graph_with_two_components()
    dists = c.nearest_neighbour_node_distances_um(G)
    assert dists.size == 4
    assert np.all(dists > 0)
    assert dists.min() == pytest.approx(1.0)


def test_nearest_neighbour_node_distances_um_needs_two_nodes():
    G = nx.MultiGraph()
    G.add_node("A", pos=(0.0, 0.0, 0.0))
    assert c.nearest_neighbour_node_distances_um(G).size == 0


def test_terminal_edge_lengths_um_only_counts_degree_one_endpoints():
    G = nx.MultiGraph()
    G.add_edge("A", "B", length=5.0)  # A is degree-1 terminal
    G.add_edge("B", "C", length=3.0)  # C is degree-1 terminal
    G.add_edge("C", "D", length=2.0)
    G.add_edge("D", "A", length=4.0)  # closes a loop: A and D become degree 2
    lengths = c.terminal_edge_lengths_um(G)
    assert lengths.size == 0  # no degree-1 nodes once the loop closes


def test_terminal_edge_lengths_um_finds_a_dangling_stub():
    G = nx.MultiGraph()
    G.add_edge("A", "B", length=5.0)
    G.add_edge("B", "C", length=3.0)
    G.add_edge("C", "D", length=2.0)  # A and D are both degree-1 terminals
    lengths = c.terminal_edge_lengths_um(G)
    assert sorted(lengths.tolist()) == [2.0, 5.0]
