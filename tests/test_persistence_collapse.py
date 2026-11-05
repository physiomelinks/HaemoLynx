"""Persistence-based node-cluster collapse: hand-verified geometry, no pipeline involved.

See graph/persistence_collapse.py's own docstring for the cited mathematics
(Carlsson & Memoli 2010's single-linkage / 0-dimensional persistent homology
correspondence; Chazal, Guibas, Oudot & Skraba 2013's ToMATo persistence-gap
clustering) this module's algorithm is built from.
"""
from __future__ import annotations

import math

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.cartwheel_guard import detect_cartwheel_hubs
from haemolynx.graph.collapse import collapse_node_clusters
from haemolynx.graph.persistence_collapse import (
    DEFAULT_SEARCH_RADIUS_MULTIPLE,
    _minimum_spanning_tree_edges,
    _persistence_cutoff,
    collapse_node_clusters_persistence,
)


# --- the pure math: MST edges and the gap they imply ------------------------


def test_mst_edges_of_a_line_of_three_points_are_the_two_short_hops():
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    edges = _minimum_spanning_tree_edges(coords)
    weights = sorted(w for _, _, w in edges)
    assert weights == pytest.approx([1.0, 2.0])


def test_mst_edges_of_fewer_than_two_points_is_empty():
    assert _minimum_spanning_tree_edges(np.array([[0.0, 0.0, 0.0]])) == []
    assert _minimum_spanning_tree_edges(np.zeros((0, 3))) == []


def test_persistence_cutoff_finds_the_big_relative_gap():
    # 1.0, 1.1 are close together (noise-scale); 4.0 is a clear jump after.
    assert _persistence_cutoff([1.0, 1.1, 4.0]) == pytest.approx(1.1)


def test_persistence_cutoff_is_none_for_a_smooth_run_with_no_gap():
    # Each step is the same small increment -- no elbow to find.
    assert _persistence_cutoff([1.0, 1.5, 2.0, 2.5, 3.0]) is None


def test_persistence_cutoff_needs_at_least_three_distances():
    """Regression test: with exactly 2 weights (a 3-node cluster's MST has
    exactly 1 edge pair, hence 1 gap), that single gap is *always* the
    entire spread by construction, so relative_gaps[0] comes out to 1.0
    whenever the two weights differ at all -- even a near-uniform pair like
    3.0 and 3.05 (a 1.6% difference) would previously read as a "clear
    gap" and get flagged, which is wrong: the statistic needs at least two
    gaps to judge whether one of them stands out."""
    assert _persistence_cutoff([]) is None
    assert _persistence_cutoff([1.0]) is None
    assert _persistence_cutoff([3.0, 3.05]) is None
    assert _persistence_cutoff([1.0, 100.0]) is None  # even a huge, "obvious" gap


# --- the core claim: splits at a real gap, agrees with distance_only when
# there is none ---------------------------------------------------------


def test_two_tight_pairs_stay_separate_where_distance_only_merges_them_all():
    """0-1 are 1 um apart, 2-3 are 1 um apart, and the pairs are ~3.5-5.5 um
    from each other -- all four sit within the default 5 um collapse
    distance of *something* in the group, so distance_only's single-linkage
    chaining merges all four into one node. The persistence gap between
    "within-pair" (1 um) and "between-pair" (3.5 um) distances is exactly
    the kind of natural, locally-set scale this method is for."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([4.5, 0.0, 0.0]))
    G.add_node(3, pos=np.array([5.5, 0.0, 0.0]))
    for node, direction in ((0, [0, 1, 0]), (1, [0, -1, 0]), (2, [0, 0, 1]), (3, [0, 0, -1])):
        ext_id = 100 + node
        ext_pos = np.asarray(direction, dtype=float) * 20.0 + G.nodes[node]["pos"]
        G.add_node(ext_id, pos=ext_pos)
        G.add_edge(
            node, ext_id, length=20.0,
            voxels=[G.nodes[node]["pos"].tolist(), ext_pos.tolist()],
        )

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    persistent = collapse_node_clusters_persistence(G, distance_threshold=5.0)

    assert legacy.number_of_nodes() == 5, "fixture bug: distance_only did not merge all four"
    assert persistent.number_of_nodes() == 6
    degrees = sorted(dict(persistent.degree()).values(), reverse=True)
    assert degrees[:2] == [2, 2], "each pair should merge into its own degree-2 representative"


def test_a_cluster_with_no_internal_structure_matches_distance_only():
    """A hub with no genuine sub-structure -- points scattered with roughly
    uniform spacing around one location -- has no persistence gap to find,
    so the method falls back to exactly what distance_only would do."""
    rng = np.random.default_rng(0)
    G = nx.MultiGraph()
    n_spokes = 8
    for i in range(n_spokes):
        G.add_node(i, pos=np.array([0.0, 0.0, 0.0]) + rng.normal(scale=0.3, size=3))
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = direction * 20.0
        ext_id = 100 + i
        G.add_node(ext_id, pos=ext_pos)
        c_pos = G.nodes[i]["pos"]
        G.add_edge(
            i, ext_id, length=float(np.linalg.norm(ext_pos - c_pos)),
            voxels=[c_pos.tolist(), ext_pos.tolist()],
        )

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    persistent = collapse_node_clusters_persistence(G, distance_threshold=5.0)

    assert persistent.number_of_nodes() == legacy.number_of_nodes()
    assert persistent.number_of_edges() == legacy.number_of_edges()


def test_a_three_node_near_uniform_cluster_merges_all_three_like_distance_only():
    """Regression test for the 3-node persistence-cutoff degeneracy: three
    nodes at 0, 3.0 and 6.05 microns apart (a near-uniform run, ~1.6%
    difference between the two hops) have no genuine persistence gap -- a
    cluster this small can never provide one (see
    test_persistence_cutoff_needs_at_least_three_distances) -- so all three
    should collapse into one node, exactly like distance_only, rather than
    splitting off the third node because a single 2-weight MST trivially
    reads as "all gap"."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([3.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([6.05, 0.0, 0.0]))
    for u, v in ((0, 1), (1, 2)):
        G.add_edge(u, v, length=float(np.linalg.norm(G.nodes[v]["pos"] - G.nodes[u]["pos"])))

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    persistent = collapse_node_clusters_persistence(G, distance_threshold=5.0)

    assert legacy.number_of_nodes() == 1, "fixture bug: distance_only did not merge all three"
    assert persistent.number_of_nodes() == 1


def test_a_second_pass_can_remerge_a_correct_split_which_is_why_the_default_is_one_pass():
    """Documents the reason max_iterations defaults to 1 (see the module's
    own docstring): once the two-tight-pairs fixture above is correctly
    split, the two surviving representatives are ~4.5 um apart -- only one
    distance, too few to show a gap -- so a second pass falls back to plain
    distance_threshold and remerges them, exactly reproducing distance_only.
    This is expected, not a bug in the fallback itself; the fix is not to
    keep iterating past the pass that actually had the local geometry to
    reason about."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([4.5, 0.0, 0.0]))
    G.add_node(3, pos=np.array([5.5, 0.0, 0.0]))
    for node, direction in ((0, [0, 1, 0]), (1, [0, -1, 0]), (2, [0, 0, 1]), (3, [0, 0, -1])):
        ext_id = 100 + node
        ext_pos = np.asarray(direction, dtype=float) * 20.0 + G.nodes[node]["pos"]
        G.add_node(ext_id, pos=ext_pos)
        G.add_edge(
            node, ext_id, length=20.0,
            voxels=[G.nodes[node]["pos"].tolist(), ext_pos.tolist()],
        )

    one_pass = collapse_node_clusters_persistence(G, distance_threshold=5.0, max_iterations=1)
    many_passes = collapse_node_clusters_persistence(
        G, distance_threshold=5.0, max_iterations=10
    )

    assert one_pass.number_of_nodes() == 6
    assert many_passes.number_of_nodes() == 5


def test_no_gap_falls_back_to_merging_within_the_ordinary_distance_threshold():
    """A cluster with three roughly-evenly-spaced points and no elbow must
    still merge everything within cluster_collapse_distance, matching
    distance_only, rather than merging nothing at all."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.5, 0.0, 0.0]))
    G.add_node(2, pos=np.array([3.0, 0.0, 0.0]))  # evenly spaced: 1.5, 1.5 -- no gap

    out = collapse_node_clusters_persistence(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 1


# --- validation --------------------------------------------------------


def test_a_negative_search_radius_multiple_is_rejected():
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="search_radius_multiple"):
        collapse_node_clusters_persistence(G, search_radius_multiple=-1.0)


def test_a_zero_search_radius_multiple_is_legal_and_merges_nothing():
    """0.0 is schema-legal (minimum=0.0) -- must degrade gracefully (find no
    candidates) rather than raise, matching this codebase's own established
    rule that a schema-legal boundary must never crash a run."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))

    out = collapse_node_clusters_persistence(G, distance_threshold=5.0, search_radius_multiple=0.0)

    assert out.number_of_nodes() == 2


def test_a_graph_with_fewer_than_two_positioned_nodes_is_left_alone():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    out = collapse_node_clusters_persistence(G, distance_threshold=5.0)
    assert list(out.nodes) == [0]


def test_default_search_radius_multiple_constant_is_a_real_multiplier():
    assert DEFAULT_SEARCH_RADIUS_MULTIPLE > 1.0


# --- integration with the cartwheel guard -----------------------------------


def _four_pairs_wheel_graph() -> nx.MultiGraph:
    """Four pairs of hub nodes arranged around a small circle (pair centres
    ~3 um apart), each pair's two members ~0.3 um from each other; every one
    of the eight hub nodes gets its own distinct spoke direction (evenly
    spread), so a full 8-way merge looks wheel-shaped while two natural
    distance scales exist: within-pair (~0.3 um) and between-pair (~4-5 um)."""
    G = nx.MultiGraph()
    n_pairs = 4
    pair_radius = 3.0
    member_offset = 0.3
    node_positions: dict[int, np.ndarray] = {}
    hub_id = 0
    for i in range(n_pairs):
        pair_angle = 2 * math.pi * i / n_pairs
        centre = np.array([0.0, pair_radius * math.cos(pair_angle), pair_radius * math.sin(pair_angle)])
        tangent = np.array([0.0, -math.sin(pair_angle), math.cos(pair_angle)])
        for sign in (-1, 1):
            pos = centre + tangent * member_offset / 2 * sign
            G.add_node(hub_id, pos=pos)
            node_positions[hub_id] = pos
            hub_id += 1

    ext_id = 1000
    for node in range(hub_id):
        angle = 2 * math.pi * node / hub_id  # each hub node gets its own distinct direction
        direction = np.array([0.0, math.cos(angle), math.sin(angle)])
        ext_pos = node_positions[node] + direction * 30.0
        G.add_node(ext_id, pos=ext_pos)
        G.add_edge(
            node, ext_id, length=30.0,
            voxels=[node_positions[node].tolist(), ext_pos.tolist()],
        )
        ext_id += 1
    return G


def test_the_split_result_is_not_flagged_by_the_cartwheel_guard_at_its_own_defaults():
    """The whole point, cross-checked against the guard this collapse
    problem was first diagnosed through, at the guard's own realistic
    default thresholds (min_degree=6, max_radial_dispersion=0.5): four
    tight pairs, chained by proximity into one component under
    distance_only, merge into a single flaggable degree-8 cartwheel; under
    persistence, the natural within-pair/between-pair gap keeps them as
    four separate degree-2 nodes, none of which are anywhere close to
    flaggable."""
    G = _four_pairs_wheel_graph()

    legacy = collapse_node_clusters(G, distance_threshold=5.0)
    persistent = collapse_node_clusters_persistence(G, distance_threshold=5.0)

    legacy_hubs = detect_cartwheel_hubs(legacy, min_degree=6, max_radial_dispersion=0.5)
    assert legacy_hubs, "fixture bug: distance_only did not even produce a flaggable hub"
    assert max(dict(legacy.degree()).values()) == 8

    persistent_hubs = detect_cartwheel_hubs(persistent, min_degree=6, max_radial_dispersion=0.5)
    assert persistent_hubs == []
    assert sorted(dict(persistent.degree()).values(), reverse=True)[:4] == [2, 2, 2, 2]
