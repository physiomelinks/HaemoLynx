"""Partitioning a vascular graph into communities/domains by modularity.

Blinder et al.'s "cortical angiome" work shows the cortical vasculature
organizes into spatially clustered, modular sub-networks ("vascular
domains"), found by graph community/modularity detection --
:func:`assign_vascular_communities` is that same style of partition, written
onto the graph as an edge attribute so it can be used to colour the vessels
layer, the way :func:`haemolynx.graph.branch_order.assign_branch_orders`
writes ``branch_order``.

:func:`communities_for_weighting` -- the actual partitioning step -- also
backs :mod:`haemolynx.statistics.network_measures`'s own community-count
summaries (``compute_communities_summary``, ``compute_weighted_communities_summary``),
which import it from here rather than duplicating it: :mod:`haemolynx.statistics`
already depends on :mod:`haemolynx.graph` (for
:func:`haemolynx.graph.validate.assert_no_forbidden_edge_attributes`), never
the other way round, so this module has no dependency on ``statistics``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

#: The four ways a vascular-community partition can be weighted -- shared by
#: :func:`communities_for_weighting`/:func:`assign_vascular_communities` and
#: ``pipeline.schema``'s own ``vascular_community_weighting`` choice, so the
#: schema's ``choices`` tuple and this module's own validation never drift
#: apart.
COMMUNITY_WEIGHTINGS: tuple[str, ...] = ("topology", "resistance", "length", "flow")

#: (source edge attribute, whether to invert it before treating it as a
#: shortest-path-style distance) for each weighting other than "topology" --
#: identical to the three distance models
#: :func:`haemolynx.statistics.network_measures.compute_betweenness_and_community_measurements`
#: already uses, so a vessel's domain here matches the community count that
#: report quotes for the same weighting.
_WEIGHTING_SOURCE_ATTR: dict[str, tuple[str, bool]] = {
    "resistance": ("resistance", False),
    "length": ("length", False),
    "flow": ("flow_abs", True),
}

#: Above this many nodes, exact greedy-modularity partitioning is replaced by
#: connected components -- same guard, same threshold, as
#: ``statistics.network_measures``'s own community-summary functions.
DEFAULT_MAX_NODES_EXACT = 1500

#: An edge whose two endpoints fall in different communities gets this label
#: instead of a domain id -- see :func:`assign_vascular_communities`.
BOUNDARY_LABEL = "boundary"


def simple_graph_with_edge_attr(
    G: Union[nx.Graph, nx.MultiGraph],
    source_attr: str,
    transform: Optional[Callable[[float], float]] = None,
    target_attr: str = "analysis_weight",
) -> nx.Graph:
    """Build a simple graph carrying one transformed edge attribute.

    For MultiGraph inputs, parallel edges are collapsed by taking the smallest
    transformed value, which is appropriate for path-based distance weights.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_s = nx.Graph()
    G_s.add_nodes_from(G.nodes())

    if transform is None:
        transform = lambda x: x  # noqa: E731

    if is_mg:
        best = {}
        for u, v, _, data in G.edges(keys=True, data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            uv = tuple(sorted((u, v)))
            if uv not in best or transformed < best[uv]:
                best[uv] = float(transformed)
        for (u, v), val in best.items():
            G_s.add_edge(u, v, **{target_attr: val})
    else:
        for u, v, data in G.edges(data=True):
            raw = data.get(source_attr)
            if raw is None or raw <= 0:
                continue
            transformed = transform(raw)
            if transformed is None or transformed <= 0:
                continue
            existing = G_s.get_edge_data(u, v, default={}).get(target_attr)
            if existing is None or transformed < existing:
                G_s.add_edge(u, v, **{target_attr: float(transformed)})
    return G_s


def communities_for_weighting(
    G: Union[nx.Graph, nx.MultiGraph],
    weighting: str,
    max_nodes_exact: int = DEFAULT_MAX_NODES_EXACT,
) -> tuple[list[frozenset], str]:
    """The node partition greedy modularity finds for *weighting*, and the
    method name that produced it.

    ``"topology"`` runs on the graph's own unweighted structure (a MultiGraph
    is first collapsed to a simple graph -- ``greedy_modularity_communities``
    needs one, and parallel edges carry no extra topological information for
    it anyway). ``"resistance"``/``"length"``/``"flow"`` build a simple graph
    carrying that one transformed edge attribute (see
    :func:`simple_graph_with_edge_attr`) and run weighted modularity on it.

    Above *max_nodes_exact*, falls back to connected components (fast and
    stable, if not modular) -- same guard already used for the statistics
    report's own community counts. Never raises on an edgeless graph:
    ``greedy_modularity_communities`` returns every node as its own singleton
    community in that case (confirmed empirically), which is what a
    weighting with no populated source attribute (e.g. "resistance" before
    haemodynamics has run) degenerates to.
    """
    if weighting not in COMMUNITY_WEIGHTINGS:
        raise ValueError(
            f"Unknown weighting={weighting!r}; choose from {COMMUNITY_WEIGHTINGS}."
        )

    if weighting == "topology":
        is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
        G_s = nx.Graph(G) if is_mg else G
        weight_kwargs: dict = {}
        exact_method = "greedy_modularity"
    else:
        source_attr, inverse = _WEIGHTING_SOURCE_ATTR[weighting]
        transform = (lambda x: 1.0 / x) if inverse else None
        G_s = simple_graph_with_edge_attr(
            G, source_attr=source_attr, transform=transform, target_attr="analysis_weight"
        )
        weight_kwargs = {"weight": "analysis_weight"}
        exact_method = "greedy_modularity_weighted"

    n_nodes = G_s.number_of_nodes()
    if n_nodes == 0:
        return [], exact_method
    if n_nodes <= max_nodes_exact:
        return list(greedy_modularity_communities(G_s, **weight_kwargs)), exact_method
    return list(nx.connected_components(G_s)), "connected_components_fallback"


@dataclass(frozen=True)
class VascularCommunitySummary:
    """What :func:`assign_vascular_communities` did to one graph."""

    community_count: int
    #: Community sizes, largest first -- same order as the domain labels.
    sizes: tuple[int, ...]
    boundary_edge_count: int
    method: str
    weighting: str


def assign_vascular_communities(
    G: nx.MultiGraph,
    *,
    weighting: str = "topology",
    max_nodes_exact: int = DEFAULT_MAX_NODES_EXACT,
    attribute: str = "vascular_community",
) -> VascularCommunitySummary:
    """Partition *G* into vascular communities/domains and write each edge's
    own domain onto ``attribute``.

    Communities are ranked by size (largest first) and labelled ``"D01"``,
    ``"D02"``, ... -- ``"D01"`` is always the network's largest domain,
    matching ``branch_order``'s own zero-padded convention. Ranking by
    ``(-len(community), min(community))`` rather than raw set order makes the
    labelling deterministic across repeated calls, regardless of how
    :func:`communities_for_weighting`'s own algorithm happens to order same-
    size communities.

    An edge whose two endpoints land in different communities -- a domain
    boundary -- gets :data:`BOUNDARY_LABEL` instead of either endpoint's own
    domain, so a boundary is a visible, distinct category rather than an
    edge that arbitrarily looks like it belongs to one side.
    """
    communities, method = communities_for_weighting(G, weighting, max_nodes_exact)
    ordered = sorted(communities, key=lambda community: (-len(community), min(community)))
    width = max(2, len(str(len(ordered))))

    node_to_label: dict = {}
    for index, community in enumerate(ordered, start=1):
        label = f"D{index:0{width}d}"
        for node in community:
            node_to_label[node] = label

    boundary_edge_count = 0
    for u, v, key in G.edges(keys=True):
        label_u = node_to_label.get(u)
        label_v = node_to_label.get(v)
        if label_u is not None and label_u == label_v:
            G[u][v][key][attribute] = label_u
        else:
            G[u][v][key][attribute] = BOUNDARY_LABEL
            boundary_edge_count += 1

    return VascularCommunitySummary(
        community_count=len(ordered),
        sizes=tuple(len(community) for community in ordered),
        boundary_edge_count=boundary_edge_count,
        method=method,
        weighting=weighting,
    )
