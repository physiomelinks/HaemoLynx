"""Current-flow ("electrical") betweenness and resistance distance.

The physics-consistent counterpart of shortest-route betweenness: a
pressure-driven vascular network is a resistor network, so blood does not
take only the shortest route -- it splits across every parallel route in
inverse proportion to its resistance (Kirchhoff). Each vessel's *current-flow
share* is the fraction of the total inlet->outlet flow passing through it
under that split.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._flow_system import TerminalFlowSystem
from ._route_graph import (
    _clear_edge_attributes,
    _edge_conductance,
    _effective_weighting,
    _iter_edges,
    _resolve_terminals,
    _route_graph,
    _set_edge,
    _terminal_problem,
    _unavailable,
)

__all__ = ["CURRENT_FLOW_EDGE_ATTRIBUTES", "compute_current_flow", "resistance_unit"]

CURRENT_FLOW_EDGE_ATTRIBUTES: tuple[str, ...] = ("current_flow_share",)

#: Largest network (nodes) the Kirchhoff index is computed for: it needs
#: every Laplacian eigenvalue, a dense O(n^3) eigendecomposition.
KIRCHHOFF_MAX_NODES = 3000
DEFAULT_TOP_N = 10


def resistance_unit(weighting: str) -> str:
    """The unit a resistance comes out in under each conductance model."""
    return {
        "resistance": "Pa.s/m^3",
        "length": "um (resistance proportional to length)",
        "topology": "vessels (unit resistance per vessel)",
    }[weighting]


def _kirchhoff_index(G: nx.Graph, weighting: str) -> Any:
    """Sum of resistance distances over every node pair of the largest
    connected component: ``n * sum(1 / lambda_i)`` over the nonzero
    eigenvalues of the conductance-weighted Laplacian."""
    simple = nx.Graph()
    for u, v, _key, data in _iter_edges(G):
        if u == v:
            continue
        g = _edge_conductance(data, weighting)
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += g
        else:
            simple.add_edge(u, v, weight=g)
    if simple.number_of_nodes() == 0:
        return "N/A (no vessels)"
    largest = max(nx.connected_components(simple), key=len)
    n = len(largest)
    if n > KIRCHHOFF_MAX_NODES:
        return f"N/A (largest component has {n} nodes; exact needs <= {KIRCHHOFF_MAX_NODES})"
    if n < 2:
        return 0.0
    sub = simple.subgraph(largest)
    scale = max(d["weight"] for _u, _v, d in sub.edges(data=True))
    laplacian = nx.laplacian_matrix(sub, weight="weight").toarray() / scale
    eigenvalues = np.linalg.eigvalsh(laplacian)[1:]
    return float(n * np.sum(1.0 / eigenvalues) / scale)


def compute_current_flow(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
    statistics_mode: str = "fast",
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Each vessel's share of the inlet->outlet current, plus the network's
    equivalent inlet->outlet resistance and (on networks up to
    :data:`KIRCHHOFF_MAX_NODES`, ``full`` mode only) its Kirchhoff index.

    Conductance follows *weighting*: one per vessel, 1/length, or the real
    solved conductance. Writes ``current_flow_share`` onto every edge of *G*.
    """
    _clear_edge_attributes(G, CURRENT_FLOW_EDGE_ATTRIBUTES)
    effective, label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    problem = _terminal_problem(inlets, outlets, _route_graph(G, effective).graph)
    if problem is not None:
        return _unavailable("Current Flow", problem, label, both)

    system = TerminalFlowSystem(G, inlets, outlets, effective)
    shares = system.flow_fraction()
    for _u, _v, _k, data in _iter_edges(G):
        data["current_flow_share"] = 0.0
    for (u, v, key), share in zip(system.edges, shares):
        _set_edge(G, u, v, key, "current_flow_share", float(share))

    order = np.argsort(-shares)[:top_n]
    result: Dict[str, Any] = {
        "Current Flow Route Weighting": label,
        "Equivalent Inlet-Outlet Resistance": system.equivalent_resistance,
        "Equivalent Inlet-Outlet Resistance Unit": resistance_unit(effective),
        "Current Flow Share Max": float(shares.max()) if shares.size else 0.0,
        "Current Flow Share Mean": float(shares.mean()) if shares.size else 0.0,
        "Current Flow Top Vessels": [
            (*system.edges[i], round(float(shares[i]), 4)) for i in order if shares[i] > 0
        ],
        "Kirchhoff Index": (
            _kirchhoff_index(G, effective)
            if statistics_mode == "full"
            else "N/A (full statistics mode only)"
        ),
    }
    if both:
        result["Current Flow Nodes Both Inlet And Outlet (excluded)"] = both
    return result
