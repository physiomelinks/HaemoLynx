"""Shared plumbing for the inlet/outlet- and edge-annotating network analyses.

Every analysis built on this writes per-vessel results onto the graph as
edge attributes (so the vessels layer can be coloured by them) and measures
distances/conductances in one of :data:`ROUTE_WEIGHTINGS`, chosen by the
``statistics_route_weighting`` setting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

import networkx as nx

#: The distance models a route can be measured in. ``topology`` counts
#: vessels, ``length`` uses centreline length (um), ``resistance`` uses
#: hydraulic resistance for routes and conductance for capacity/current.
ROUTE_WEIGHTINGS: tuple[str, ...] = ("topology", "length", "resistance")


def _iter_edges(G: nx.Graph) -> Iterator[tuple[Any, Any, Any, dict]]:
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, None, data


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _effective_weighting(G: nx.Graph, weighting: str) -> tuple[str, str]:
    """The weighting actually usable on *G*, and how to report it.

    Resistance needs a finite positive ``resistance`` and ``conductance`` on
    every edge (haemodynamics has run); length needs a positive ``length``.
    Anything missing falls back one step (resistance -> length -> topology)
    and says so, rather than silently mixing units.
    """
    if weighting not in ROUTE_WEIGHTINGS:
        raise ValueError(
            f"Unknown route weighting {weighting!r}. Choose one of {ROUTE_WEIGHTINGS}."
        )
    edges = [data for _u, _v, _k, data in _iter_edges(G)]
    if weighting == "resistance":
        if all(
            _positive_finite(d.get("resistance")) and _positive_finite(d.get("conductance"))
            for d in edges
        ):
            return "resistance", "resistance"
        weighting, fallback_from = "length", "resistance"
    else:
        fallback_from = None
    if weighting == "length":
        if all(_positive_finite(d.get("length")) for d in edges):
            label = "length" if fallback_from is None else f"length ({fallback_from} unavailable)"
            return "length", label
        fallback_from = fallback_from or "length"
    if weighting == "topology" and fallback_from is None:
        return "topology", "topology"
    return "topology", f"topology ({fallback_from} unavailable)"


def _edge_weight_and_capacity(data: dict, weighting: str) -> tuple[float, float]:
    """(route weight, cut capacity) for one vessel under *weighting*."""
    if weighting == "resistance":
        return float(data["resistance"]), float(data["conductance"])
    if weighting == "length":
        return float(data["length"]), 1.0
    return 1.0, 1.0


def _edge_conductance(data: dict, weighting: str) -> float:
    """Conductance of one vessel under *weighting*: one per vessel for
    ``topology``, 1/length for ``length`` (Poiseuille at a uniform
    diameter), the real ``conductance`` for ``resistance``."""
    if weighting == "resistance":
        return float(data["conductance"])
    if weighting == "length":
        return 1.0 / float(data["length"])
    return 1.0


@dataclass
class _RouteGraph:
    """A simple graph over *G* for route/cut computations.

    ``graph`` has one edge per connected node pair, with ``route_weight``
    (the shortest parallel vessel's) and ``capacity`` (summed over parallel
    vessels, since each is a real extra route). ``members`` maps each pair
    back to its original ``(u, v, key, weight)`` edges.
    """

    graph: nx.Graph
    members: dict = field(default_factory=dict)


def _route_graph(G: nx.Graph, weighting: str) -> _RouteGraph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    members: dict = {}
    for u, v, key, data in _iter_edges(G):
        if u == v:
            continue  # a self-loop never takes blood anywhere
        weight, capacity = _edge_weight_and_capacity(data, weighting)
        pair = frozenset((u, v))
        members.setdefault(pair, []).append((u, v, key, weight))
        if H.has_edge(u, v):
            edge = H[u][v]
            edge["route_weight"] = min(edge["route_weight"], weight)
            edge["capacity"] += capacity
        else:
            H.add_edge(u, v, route_weight=weight, capacity=capacity)
    return _RouteGraph(graph=H, members=members)


def _resolve_terminals(
    G: nx.Graph, inlet_nodes: Optional[Iterable[Any]], outlet_nodes: Optional[Iterable[Any]]
) -> tuple[set, set, int]:
    """Inlets and outlets present in *G*, minus any node given as both."""
    inlets = {n for n in (inlet_nodes or ()) if n in G}
    outlets = {n for n in (outlet_nodes or ()) if n in G}
    both = inlets & outlets
    return inlets - both, outlets - both, len(both)


def _clear_edge_attributes(G: nx.Graph, names: Iterable[str]) -> None:
    names = tuple(names)
    for _u, _v, _k, data in _iter_edges(G):
        for name in names:
            data.pop(name, None)


def _set_edge(G: nx.Graph, u: Any, v: Any, key: Any, name: str, value: Any) -> None:
    if key is None:
        G[u][v][name] = value
    else:
        G[u][v][key][name] = value


def _unavailable(
    prefix: str, reason: str, weighting_label: Optional[str] = None, both: int = 0
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if weighting_label is not None:
        result[f"{prefix} Route Weighting"] = weighting_label
    result[f"{prefix} Status"] = f"N/A ({reason})"
    if both:
        result[f"{prefix} Nodes Both Inlet And Outlet (excluded)"] = both
    return result


def _component_index(H: nx.Graph) -> dict:
    component_of = {}
    for index, component in enumerate(nx.connected_components(H)):
        for node in component:
            component_of[node] = index
    return component_of


def _terminal_problem(inlets: set, outlets: set, H: nx.Graph) -> Optional[str]:
    if not inlets or not outlets:
        return "no inlet and outlet nodes to route between"
    component_of = _component_index(H)
    inlet_components = {component_of[n] for n in inlets}
    if not any(component_of[n] in inlet_components for n in outlets):
        return "no inlet is connected to any outlet"
    return None
