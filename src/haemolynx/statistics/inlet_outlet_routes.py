"""Inlet->outlet bottlenecks and shunts.

Both analyses look at the network only through the routes blood can take
from an inlet (a large-arteriole-mask terminal or a picked inlet node) to an
outlet (the venous equivalent):

* a **bottleneck** is where that route system is narrowest -- the minimum
  edge cut separating every inlet from every outlet, plus each vessel's
  share of the shortest inlet->outlet routes;
* a **shunt** is an inlet->outlet route that is much shorter than the
  network's typical one, i.e. one that bypasses the bulk of the bed.

Unlike every other statistic, both also *write edge attributes* onto the
graph they are given, so the vessels layer can be coloured by them (see
``haemolynx.gui.results.OPTIONAL_EDGE_COLUMNS``). Attributes from an earlier
call are cleared first, and nothing is written when there is no route to
analyse -- so an absent attribute always means "not computed", never
"left over from a previous run".
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._route_graph import (
    ROUTE_WEIGHTINGS,
    _clear_edge_attributes,
    _edge_weight_and_capacity,
    _effective_weighting,
    _iter_edges,
    _resolve_terminals,
    _route_graph,
    _set_edge,
    _terminal_problem,
    _unavailable,
)
from ._sampling import REPRODUCIBILITY_SEED

__all__ = [
    "ROUTE_WEIGHTINGS",
    "BOTTLENECK_EDGE_ATTRIBUTES",
    "SHUNT_EDGE_ATTRIBUTES",
    "compute_inlet_outlet_bottlenecks",
    "compute_inlet_outlet_shunts",
]

BOTTLENECK_EDGE_ATTRIBUTES: tuple[str, ...] = ("bottleneck_route_share", "bottleneck_min_cut")
SHUNT_EDGE_ATTRIBUTES: tuple[str, ...] = ("shunt_route_ratio", "shunt")

MIN_CUT_LABEL = "min_cut"
NOT_MIN_CUT_LABEL = "not_min_cut"
SHUNT_LABEL = "shunt"
NOT_SHUNT_LABEL = "not_shunt"

#: Above this many inlets, ``fast`` mode samples this many of them as route
#: sources for the route-share score. Each source is one weighted
#: shortest-path sweep of the whole network (~0.5 s on a 27k-node, 78k-edge
#: lattice), so this keeps ``fast`` to seconds; ``full`` uses every inlet.
DEFAULT_ROUTE_SHARE_MAX_SOURCES = 16
DEFAULT_TOP_N = 10

_SOURCE = ("__inlet_outlet_routes_source__",)
_SINK = ("__inlet_outlet_routes_sink__",)


def compute_inlet_outlet_bottlenecks(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
    statistics_mode: str = "fast",
    max_sources: int = DEFAULT_ROUTE_SHARE_MAX_SOURCES,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Where the inlet->outlet route system is narrowest.

    **Minimum cut**: the smallest-capacity set of vessels whose occlusion
    disconnects every inlet from every outlet (max-flow/min-cut between a
    super-source on the inlets and a super-sink on the outlets). Capacity is
    one per vessel for ``topology``/``length`` -- the fewest vessels -- and
    conductance for ``resistance`` -- the hydraulically narrowest
    cross-section. Parallel vessels between the same two junctions add
    their capacities, so a doubled vessel is never a bottleneck on its own.

    **Route share**: per vessel, the fraction of shortest inlet->outlet
    routes (one per connected inlet/outlet pair, weighted by *weighting*)
    that pass through it -- 1.0 on a trunk every route uses. Between
    parallel vessels the share goes to the shortest; the others carry none
    of the *shortest* routes. ``fast`` mode samples at most *max_sources*
    inlets as route sources on a network with more than that many.

    Writes ``bottleneck_route_share`` (float) and ``bottleneck_min_cut``
    (``"min_cut"``/``"not_min_cut"``) onto every edge of *G*, and returns a
    summary. A node given as both inlet and outlet is left out of both.
    """
    if statistics_mode not in ("fast", "full"):
        raise ValueError(f"Invalid statistics_mode={statistics_mode!r}; use 'fast' or 'full'.")
    _clear_edge_attributes(G, BOTTLENECK_EDGE_ATTRIBUTES)
    effective, weighting_label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    routes = _route_graph(G, effective)
    H = routes.graph
    problem = _terminal_problem(inlets, outlets, H)
    if problem is not None:
        return _unavailable("Bottleneck", problem, weighting_label, both)

    # --- minimum cut --------------------------------------------------------
    # Capacities are rescaled to at most 1 before the flow solve: SI
    # conductances are ~1e-16, small enough to trip the solver's own
    # floating-point tolerances; the reported capacity is scaled back.
    max_capacity = max(data["capacity"] for _u, _v, data in H.edges(data=True))
    D = nx.DiGraph()
    for u, v, data in H.edges(data=True):
        capacity = data["capacity"] / max_capacity
        D.add_edge(u, v, capacity=capacity)
        D.add_edge(v, u, capacity=capacity)
    for inlet in inlets:
        D.add_edge(_SOURCE, inlet)  # no capacity attribute = unbounded
    for outlet in outlets:
        D.add_edge(outlet, _SINK)
    cut_value, (source_side, _sink_side) = nx.minimum_cut(D, _SOURCE, _SINK)
    cut_pairs = {
        frozenset((u, v))
        for u, v in H.edges()
        if (u in source_side) != (v in source_side)
    }

    # --- route share --------------------------------------------------------
    sources = sorted(inlets, key=repr)
    method = "exact"
    if statistics_mode == "fast" and len(sources) > max_sources:
        sources = random.Random(REPRODUCIBILITY_SEED).sample(sources, max_sources)
        method = f"sampled_sources={max_sources}"
    component_of = {}
    for index, component in enumerate(nx.connected_components(H)):
        for node in component:
            component_of[node] = index
    outlet_count_by_component: dict = {}
    for outlet in outlets:
        outlet_count_by_component[component_of[outlet]] = (
            outlet_count_by_component.get(component_of[outlet], 0) + 1
        )
    routed_pairs = sum(outlet_count_by_component.get(component_of[s], 0) for s in sources)
    # A directed copy, so each inlet->outlet pair is counted exactly once
    # (the undirected subset betweenness halves its counts).
    directed = H.to_directed()
    raw = nx.edge_betweenness_centrality_subset(
        directed,
        sources=sources,
        targets=list(outlets),
        normalized=False,
        weight=None if effective == "topology" else "route_weight",
    )
    share_by_pair: dict = {}
    for (u, v), value in raw.items():
        pair = frozenset((u, v))
        share_by_pair[pair] = share_by_pair.get(pair, 0.0) + float(value)
    if routed_pairs:
        share_by_pair = {pair: value / routed_pairs for pair, value in share_by_pair.items()}

    # --- write attributes back onto G --------------------------------------
    shares: list[tuple[float, Any, Any, Any]] = []
    for _u, _v, _k, data in _iter_edges(G):
        data["bottleneck_route_share"] = 0.0
        data["bottleneck_min_cut"] = NOT_MIN_CUT_LABEL
    cut_edge_count = 0
    for pair, members in routes.members.items():
        best = min(members, key=lambda m: m[3])
        share = min(1.0, share_by_pair.get(pair, 0.0))
        u, v, key, _weight = best
        _set_edge(G, u, v, key, "bottleneck_route_share", share)
        shares.append((share, u, v, key))
        if pair in cut_pairs:
            for mu, mv, mkey, _w in members:
                _set_edge(G, mu, mv, mkey, "bottleneck_min_cut", MIN_CUT_LABEL)
                cut_edge_count += 1

    values = [
        float(data["bottleneck_route_share"]) for _u, _v, _k, data in _iter_edges(G)
    ]
    shares.sort(key=lambda row: row[0], reverse=True)
    top = [(u, v, key, round(share, 4)) for share, u, v, key in shares[:top_n] if share > 0]
    result = {
        "Bottleneck Route Weighting": weighting_label,
        "Bottleneck Min-Cut Edge Count": cut_edge_count,
        "Bottleneck Min-Cut Capacity": float(cut_value) * max_capacity,
        "Bottleneck Min-Cut Capacity Unit": (
            "conductance (m^3/(Pa.s))" if effective == "resistance" else "vessels"
        ),
        "Bottleneck Route Share Max": max(values) if values else 0.0,
        "Bottleneck Route Share Mean": float(np.mean(values)) if values else 0.0,
        "Bottleneck Top Edges": top,
        "Bottleneck Route Share Method": method,
    }
    if both:
        result["Bottleneck Nodes Both Inlet And Outlet (excluded)"] = both
    return result


def compute_inlet_outlet_shunts(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
    max_route_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Short inlet->outlet routes that bypass the bulk of the network.

    For every vessel, its *route length* is the shortest inlet->outlet route
    that passes through it in the forward direction: shortest distance from
    any inlet to its near end (not passing an outlet), plus the vessel, plus
    shortest distance from its far end to any outlet (not passing an inlet),
    where "forward" means the far end is both farther from the inlets and
    closer to the outlets. That condition guarantees the two halves share no
    node, so every length is a real simple route rather than a walk that
    doubles back -- without it, the bed vessels right beside a shunt would
    borrow the shunt's own shortness. A vessel with no forward direction (a
    dead-end spur, a lateral link between two equally distant points) has no
    route and a NaN ratio.

    Dividing by the median route length over every vessel that has one
    gives ``shunt_route_ratio``: about 1 for a typical capillary-bed vessel,
    well below 1 for a vessel on a direct arteriole-to-venule connection. A
    ratio at or below *max_route_fraction* is a shunt. Each connected group
    of shunt vessels touching both an inlet and an outlet is counted as one
    shunt pathway. When the graph carries a solved ``flow_abs``, also
    reports what fraction of the flow reaching the outlets arrives through
    shunt vessels.

    Writes ``shunt_route_ratio`` (float; NaN on a vessel with no route) and
    ``shunt`` (``"shunt"``/``"not_shunt"``) onto every edge of *G*.
    """
    if not (max_route_fraction >= 0.0):
        raise ValueError(f"max_route_fraction must be >= 0, got {max_route_fraction!r}")
    _clear_edge_attributes(G, SHUNT_EDGE_ATTRIBUTES)
    effective, weighting_label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    routes = _route_graph(G, effective)
    H = routes.graph
    problem = _terminal_problem(inlets, outlets, H)
    if problem is not None:
        return _unavailable("Shunt", problem, weighting_label, both)

    # The inlet half of a route never passes through an outlet, nor the
    # outlet half through an inlet -- otherwise a bed vessel beside the
    # inlet would get a fake short route "inlet -> vessel -> back through
    # the inlet -> shunt".
    d_in = nx.multi_source_dijkstra_path_length(
        H.subgraph(set(H) - outlets), inlets, weight="route_weight"
    )
    d_out = nx.multi_source_dijkstra_path_length(
        H.subgraph(set(H) - inlets), outlets, weight="route_weight"
    )

    def from_inlets(node: Any) -> float:
        return math.inf if node in outlets else d_in.get(node, math.inf)

    def to_outlets(node: Any) -> float:
        return math.inf if node in inlets else d_out.get(node, math.inf)

    def route_length(u: Any, v: Any, weight: float) -> float:
        # Only a *forward* orientation counts -- farther from the inlets and
        # closer to the outlets at its far end. That guarantees the inlet
        # half and outlet half share no node, so the length is that of a
        # real simple route, not a walk that doubles back. A dead-end spur
        # or a lateral link has no forward orientation, and so no route.
        best = math.inf
        for a, b in ((u, v), (v, u)):
            ahead, behind = from_inlets(a), to_outlets(b)
            if (
                math.isfinite(ahead)
                and math.isfinite(behind)
                and ahead < from_inlets(b)
                and behind < to_outlets(a)
            ):
                best = min(best, ahead + weight + behind)
        return best

    through: list[tuple[Any, Any, Any, float]] = []
    for u, v, key, data in _iter_edges(G):
        if u == v:
            through.append((u, v, key, math.inf))
            continue
        weight, _capacity = _edge_weight_and_capacity(data, effective)
        through.append((u, v, key, route_length(u, v, weight)))

    finite = [length for *_rest, length in through if math.isfinite(length)]
    if not finite:
        return _unavailable("Shunt", "no forward inlet-outlet route", weighting_label, both)
    reference = float(np.median(finite))
    shunt_graph = nx.Graph()
    shunt_edges: set = set()
    for u, v, key, length in through:
        if math.isfinite(length) and reference > 0:
            ratio = length / reference
        else:
            ratio = float("nan")
        is_shunt = math.isfinite(ratio) and ratio <= max_route_fraction
        _set_edge(G, u, v, key, "shunt_route_ratio", ratio)
        _set_edge(G, u, v, key, "shunt", SHUNT_LABEL if is_shunt else NOT_SHUNT_LABEL)
        if is_shunt:
            shunt_graph.add_edge(u, v)
            shunt_edges.add((u, v, key))

    pathway_count = sum(
        1
        for component in nx.connected_components(shunt_graph)
        if component & inlets and component & outlets
    )

    total_edges = G.number_of_edges()
    result = {
        "Shunt Route Weighting": weighting_label,
        "Shunt Route Fraction Threshold": float(max_route_fraction),
        "Shunt Edge Count": len(shunt_edges),
        "Shunt Edge Fraction": len(shunt_edges) / total_edges if total_edges else 0.0,
        "Shunt Pathway Count": pathway_count,
        "Shortest Inlet-Outlet Route": float(min(finite)),
        "Median Inlet-Outlet Route": reference,
        "Shunt Outflow Fraction": _shunt_outflow_fraction(G, outlets, shunt_edges),
    }
    if both:
        result["Shunt Nodes Both Inlet And Outlet (excluded)"] = both
    return result


def _shunt_outflow_fraction(G: nx.Graph, outlets: set, shunt_edges: set) -> Any:
    total = 0.0
    shunted = 0.0
    for u, v, key, data in _iter_edges(G):
        if u == v or (u not in outlets and v not in outlets):
            continue
        flow = data.get("flow_abs")
        try:
            flow = abs(float(flow))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(flow):
            continue
        total += flow
        if (u, v, key) in shunt_edges:
            shunted += flow
    if total <= 0.0:
        return "N/A (no solved flow at the outlets)"
    return shunted / total
