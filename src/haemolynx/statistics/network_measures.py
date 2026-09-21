"""Betweenness centrality and community/modularity partitioning, plain
topological and weighted by resistance/length/solved flow."""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from haemolynx.graph.communities import (
    DEFAULT_MAX_NODES_EXACT,
    communities_for_weighting,
    simple_graph_with_edge_attr,
)

from ._sampling import REPRODUCIBILITY_SEED

#: Above this many nodes, betweenness centrality falls back to an
#: approximate computation (`DEFAULT_BETWEENNESS_APPROX_K` source nodes)
#: instead of the exact one -- lower than
#: `graph.communities.DEFAULT_MAX_NODES_EXACT` (modularity's own threshold)
#: because betweenness is more expensive per node.
DEFAULT_BETWEENNESS_MAX_NODES_EXACT = 1000
#: How many source nodes an approximate betweenness computation samples,
#: once a graph is too large for the exact one.
DEFAULT_BETWEENNESS_APPROX_K = 128
#: How many of the highest-betweenness nodes a summary reports by name.
DEFAULT_BETWEENNESS_TOP_N = 5


def compute_communities_summary(
    G: nx.Graph, max_nodes_exact: int = DEFAULT_MAX_NODES_EXACT
) -> Dict[str, Any]:
    """Compute community statistics with runtime guards."""
    if G.number_of_nodes() == 0:
        return {"Community Count": 0}

    communities, method = communities_for_weighting(G, "topology", max_nodes_exact)
    sizes = [len(c) for c in communities]
    return {
        "Community Count": len(communities),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": method,
    }


def compute_communities(G: nx.Graph):
    # Plain topological communities. Weighted community detection (by
    # resistance, geometry/length, or solved |flow|) is
    # compute_weighted_communities_summary, all three reported together by
    # compute_betweenness_and_community_measurements.
    return list(greedy_modularity_communities(G))


def compute_betweenness_summary(
    G: nx.Graph,
    max_nodes_exact: int = DEFAULT_BETWEENNESS_MAX_NODES_EXACT,
    approx_k: int = DEFAULT_BETWEENNESS_APPROX_K,
    seed: int = REPRODUCIBILITY_SEED,
    top_n: int = DEFAULT_BETWEENNESS_TOP_N,
) -> Dict[str, Any]:
    """Compute compact betweenness summary, avoiding huge outputs."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G)
        method = "exact"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(G, k=k, seed=seed)
        method = f"approx_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": top,
        "Betweenness Method": method,
    }


def compute_betweenness(G: nx.Graph):
    # Plain topological betweenness. Weighted betweenness (by resistance,
    # geometry/length, or solved |flow|) is compute_weighted_betweenness_
    # summary, all three reported together by
    # compute_betweenness_and_community_measurements.
    return nx.betweenness_centrality(G)


def compute_weighted_betweenness_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = DEFAULT_BETWEENNESS_MAX_NODES_EXACT,
    approx_k: int = DEFAULT_BETWEENNESS_APPROX_K,
    seed: int = REPRODUCIBILITY_SEED,
    top_n: int = DEFAULT_BETWEENNESS_TOP_N,
) -> Dict[str, Any]:
    """Compute weighted betweenness summary from a chosen edge attribute."""
    transform = (lambda x: 1.0 / x) if inverse_source_attr else None
    G_s = simple_graph_with_edge_attr(
        G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
    )
    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return {"Betweenness Mean": 0.0, "Betweenness Max": 0.0}

    if n_nodes <= max_nodes_exact:
        bet = nx.betweenness_centrality(G_s, weight="analysis_weight")
        method = "exact_weighted"
    else:
        k = min(approx_k, n_nodes)
        bet = nx.betweenness_centrality(
            G_s, k=k, seed=seed, weight="analysis_weight"
        )
        method = f"approx_weighted_k={k}"

    values = list(bet.values())
    top = sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "Betweenness Mean": float(np.mean(values)) if values else 0.0,
        "Betweenness Max": float(np.max(values)) if values else 0.0,
        "Betweenness Top Nodes": [
            {"node": node, "value": float(value)} for node, value in top
        ],
        "Betweenness Method": method,
    }


#: `compute_betweenness_and_community_measurements`'s three source
#: attributes, translated to `graph.communities`'s own weighting names, so
#: this function's public `source_attr`/`inverse_source_attr` signature
#: (unchanged, for backward compatibility) can still delegate to
#: `communities_for_weighting`.
_WEIGHTING_FOR_SOURCE_ATTR = {
    ("resistance", False): "resistance",
    ("length", False): "length",
    ("flow_abs", True): "flow",
}


def compute_weighted_communities_summary(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    inverse_source_attr: bool = False,
    max_nodes_exact: int = DEFAULT_MAX_NODES_EXACT,
    precomputed: "tuple[list, str] | None" = None,
) -> Dict[str, Any]:
    """Compute weighted community summary using greedy modularity.

    *precomputed*, when given, is a ``(communities, method)`` pair already
    produced by :func:`~haemolynx.graph.communities.communities_for_weighting`
    for this exact (*source_attr*, *inverse_source_attr*) -- lets a caller
    that already partitioned this graph for the same weighting elsewhere
    (e.g. :func:`haemolynx.graph.communities.assign_vascular_communities`)
    skip running greedy modularity on it a second time. Not validated
    against *source_attr*/*inverse_source_attr* for cost reasons; passing a
    partition for a different weighting is the caller's error.
    """
    weighting = _WEIGHTING_FOR_SOURCE_ATTR.get((source_attr, inverse_source_attr))
    if precomputed is not None:
        communities, method = precomputed
    elif weighting is not None:
        communities, method = communities_for_weighting(G, weighting, max_nodes_exact)
    else:
        # An arbitrary (source_attr, inverse_source_attr) pair not among the
        # three named weightings `graph.communities` knows about -- fall
        # back to the same computation inline, so this function's public
        # signature stays general rather than restricted to those three.
        transform = (lambda x: 1.0 / x) if inverse_source_attr else None
        G_s = simple_graph_with_edge_attr(
            G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
        )
        n_nodes = G_s.number_of_nodes()
        if n_nodes == 0:
            communities, method = [], "greedy_modularity_weighted"
        elif n_nodes <= max_nodes_exact:
            communities = list(greedy_modularity_communities(G_s, weight="analysis_weight"))
            method = "greedy_modularity_weighted"
        else:
            communities = list(nx.connected_components(G_s))
            method = "connected_components_fallback"

    if not communities and method != "connected_components_fallback":
        return {"Community Count": 0}
    sizes = [len(c) for c in communities]
    return {
        "Community Count": len(communities),
        "Largest Community Size": max(sizes) if sizes else 0,
        "Mean Community Size": float(np.mean(sizes)) if sizes else 0,
        "Community Method": method,
    }


def compute_betweenness_and_community_measurements(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    precomputed_communities: "Dict[str, tuple[list, str]] | None" = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute weighted betweenness/community using three edge distance models.

    - "edge_resistance": Poiseuille resistance as the shortest-path distance
      -- the vessels that impede flow least are the most central.
    - "edge_length": physical length as the distance -- purely geometric,
      independent of haemodynamics.
    - "edge_flow_abs": inverse of the solved absolute flow (``flow_abs``,
      written by haemodynamics.resistance.set_edge_flows) as the distance --
      vessels carrying the most flow are treated as the shortest, most
      travelled paths. Only meaningful once flow has been solved; edges with
      no flow (or none solved yet) drop out the same way a missing
      resistance or length would.

    All three are reported together rather than one at a time, so a caller
    never has to guess which distance model a given number came from.

    *precomputed_communities*, when given, maps a
    :data:`haemolynx.graph.communities.COMMUNITY_WEIGHTINGS` name
    ("resistance"/"length"/"flow") to a ``(communities, method)`` pair
    already computed for this graph -- e.g. by
    :func:`haemolynx.graph.communities.assign_vascular_communities` for the
    same weighting -- so that one weighting's community partition is not
    computed twice.
    """
    precomputed_communities = precomputed_communities or {}
    resistance_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="resistance", inverse_source_attr=False
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="resistance", inverse_source_attr=False,
            precomputed=precomputed_communities.get("resistance"),
        ),
    }
    edge_length_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="length", inverse_source_attr=False
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="length", inverse_source_attr=False,
            precomputed=precomputed_communities.get("length"),
        ),
    }
    edge_flow_results = {
        "Betweenness": compute_weighted_betweenness_summary(
            G, source_attr="flow_abs", inverse_source_attr=True
        ),
        "Communities": compute_weighted_communities_summary(
            G, source_attr="flow_abs", inverse_source_attr=True,
            precomputed=precomputed_communities.get("flow"),
        ),
    }
    return {
        "edge_resistance": resistance_results,
        "edge_length": edge_length_results,
        "edge_flow_abs": edge_flow_results,
    }
