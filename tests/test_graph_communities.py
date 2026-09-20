"""Unit tests for haemolynx.graph.communities."""
from __future__ import annotations

import networkx as nx
import pytest

from haemolynx.graph import communities as gc


def _two_clusters_joined_by_one_bridge() -> nx.MultiGraph:
    """Two dense triangles (0,1,2) and (3,4,5), joined only by edge (2, 3).

    A textbook case for modularity detection: the bridge is the one edge
    greedy modularity should NOT put inside either triangle's community.
    """
    G = nx.MultiGraph()
    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        G.add_edge(u, v, length=1.0, resistance=1.0, flow_abs=1.0)
    G.add_edge(2, 3, length=1.0, resistance=1.0, flow_abs=1.0)
    return G


# ---------------------------------------------------------------------------
# simple_graph_with_edge_attr
# ---------------------------------------------------------------------------
def test_simple_graph_with_edge_attr_collapses_parallel_edges_to_the_smallest():
    G = nx.MultiGraph()
    G.add_edge(0, 1, resistance=5.0)
    G.add_edge(0, 1, resistance=2.0)
    G_s = gc.simple_graph_with_edge_attr(G, source_attr="resistance")
    assert G_s.number_of_edges() == 1
    assert G_s[0][1]["analysis_weight"] == pytest.approx(2.0)


def test_simple_graph_with_edge_attr_skips_missing_or_nonpositive_values():
    G = nx.MultiGraph()
    G.add_edge(0, 1, resistance=1.0)
    G.add_edge(1, 2)  # no resistance attribute at all
    G.add_edge(2, 3, resistance=-1.0)
    G_s = gc.simple_graph_with_edge_attr(G, source_attr="resistance")
    assert set(G_s.edges()) == {(0, 1)}
    assert set(G_s.nodes()) == {0, 1, 2, 3}  # every node still present


def test_simple_graph_with_edge_attr_applies_transform():
    G = nx.Graph()
    G.add_edge(0, 1, flow_abs=4.0)
    G_s = gc.simple_graph_with_edge_attr(
        G, source_attr="flow_abs", transform=lambda x: 1.0 / x
    )
    assert G_s[0][1]["analysis_weight"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# communities_for_weighting
# ---------------------------------------------------------------------------
def test_communities_for_weighting_rejects_unknown_weighting():
    G = nx.Graph()
    G.add_edge(0, 1)
    with pytest.raises(ValueError, match="Unknown weighting"):
        gc.communities_for_weighting(G, "not_a_real_weighting")


@pytest.mark.parametrize("weighting", gc.COMMUNITY_WEIGHTINGS)
def test_communities_for_weighting_separates_the_two_clusters(weighting):
    G = _two_clusters_joined_by_one_bridge()
    communities, method = gc.communities_for_weighting(G, weighting)
    assert method in ("greedy_modularity", "greedy_modularity_weighted")
    node_sets = [set(c) for c in communities]
    assert {0, 1, 2} in node_sets
    assert {3, 4, 5} in node_sets
    assert len(communities) == 2


def test_communities_for_weighting_topology_accepts_a_multigraph_directly():
    """Regression: greedy_modularity_communities needs a simple graph --
    calling it on a raw MultiGraph must not raise."""
    G = _two_clusters_joined_by_one_bridge()
    communities, _method = gc.communities_for_weighting(G, "topology")
    assert sum(len(c) for c in communities) == G.number_of_nodes()


def test_communities_for_weighting_empty_graph_returns_nothing():
    G = nx.Graph()
    communities, method = gc.communities_for_weighting(G, "topology")
    assert communities == []
    assert method == "greedy_modularity"


def test_communities_for_weighting_degenerates_gracefully_with_no_populated_attribute():
    """Regression: "resistance"/"flow" weighting before haemodynamics has
    run (no edge carries that attribute) must not raise -- it degenerates
    to every node its own singleton community."""
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    communities, method = gc.communities_for_weighting(G, "resistance")
    assert method == "greedy_modularity_weighted"
    assert sorted(len(c) for c in communities) == [1, 1, 1]


def test_communities_for_weighting_falls_back_to_connected_components_above_the_node_cap():
    G = _two_clusters_joined_by_one_bridge()
    communities, method = gc.communities_for_weighting(G, "topology", max_nodes_exact=2)
    assert method == "connected_components_fallback"
    assert len(communities) == 1  # the bridge keeps everything in one component
    assert set(communities[0]) == set(G.nodes())


# ---------------------------------------------------------------------------
# assign_vascular_communities
# ---------------------------------------------------------------------------
def test_assign_vascular_communities_labels_domains_and_the_bridge_as_boundary():
    G = _two_clusters_joined_by_one_bridge()
    summary = gc.assign_vascular_communities(G, weighting="topology")

    assert summary.community_count == 2
    assert summary.boundary_edge_count == 1
    assert summary.sizes == (3, 3)
    assert summary.weighting == "topology"

    labels = {frozenset((u, v)): data["vascular_community"] for u, v, data in G.edges(data=True)}
    assert labels[frozenset((2, 3))] == gc.BOUNDARY_LABEL
    # Every other edge keeps a real domain label, and the two triangles
    # never share one.
    triangle_a_labels = {labels[frozenset(pair)] for pair in [(0, 1), (1, 2), (0, 2)]}
    triangle_b_labels = {labels[frozenset(pair)] for pair in [(3, 4), (4, 5), (3, 5)]}
    assert len(triangle_a_labels) == 1
    assert len(triangle_b_labels) == 1
    assert triangle_a_labels != triangle_b_labels
    assert triangle_a_labels | triangle_b_labels <= {"D01", "D02"}


def test_assign_vascular_communities_largest_domain_is_d01():
    G = nx.MultiGraph()
    # A 4-node clique (the larger domain) plus a lone pair, bridged once.
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        G.add_edge(u, v, length=1.0)
    G.add_edge(4, 5, length=1.0)
    G.add_edge(3, 4, length=1.0)  # the bridge
    summary = gc.assign_vascular_communities(G, weighting="topology")
    assert summary.sizes[0] >= summary.sizes[-1]

    big_domain_edge = G[0][1][0]["vascular_community"]
    small_domain_edge = G[4][5][0]["vascular_community"]
    assert big_domain_edge == "D01"
    assert small_domain_edge != "D01"
    assert small_domain_edge != gc.BOUNDARY_LABEL


def test_assign_vascular_communities_is_deterministic_across_repeated_calls():
    G = _two_clusters_joined_by_one_bridge()
    gc.assign_vascular_communities(G, weighting="topology")
    first = {(u, v, k): d["vascular_community"] for u, v, k, d in G.edges(keys=True, data=True)}
    gc.assign_vascular_communities(G, weighting="topology")
    second = {(u, v, k): d["vascular_community"] for u, v, k, d in G.edges(keys=True, data=True)}
    assert first == second


def test_assign_vascular_communities_handles_multigraph_parallel_edges():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=1.0)
    G.add_edge(0, 1, length=1.0)  # a second, parallel edge
    G.add_edge(1, 2, length=1.0)
    summary = gc.assign_vascular_communities(G, weighting="topology")
    assert summary.boundary_edge_count == 0
    labels = [data["vascular_community"] for _u, _v, data in G.edges(data=True)]
    assert len(labels) == 3
    assert len(set(labels)) == 1  # one small, fully-connected component


def test_assign_vascular_communities_empty_graph_does_not_raise():
    G = nx.MultiGraph()
    summary = gc.assign_vascular_communities(G, weighting="topology")
    assert summary.community_count == 0
    assert summary.boundary_edge_count == 0


def test_assign_vascular_communities_rejects_unknown_weighting():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=1.0)
    with pytest.raises(ValueError, match="Unknown weighting"):
        gc.assign_vascular_communities(G, weighting="not_a_real_weighting")


def test_assign_vascular_communities_uses_a_custom_attribute_name():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=1.0)
    gc.assign_vascular_communities(G, weighting="topology", attribute="domain")
    assert "domain" in G[0][1][0]
    assert "vascular_community" not in G[0][1][0]
