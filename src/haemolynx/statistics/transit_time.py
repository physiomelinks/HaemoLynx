"""Blood transit times and capillary transit-time heterogeneity (CTH).

Each vessel's transit time is its volume divided by its flow. Following the
solved flow from the inlets, the arrival-time distribution at every node is
the flow-weighted mixture of its inflowing vessels' distributions, so its
mean and variance propagate *exactly* in one pass over the nodes in order
of decreasing pressure (a pressure-driven flow never runs uphill, so that
order is a topological order of the flow). At the outlets this gives the
network's mean transit time (MTT) and its spread; restricting the sum to
capillary vessels gives capillary MTT and CTH, the heterogeneity measure
linked to oxygen-extraction efficiency.

Assumes plug flow along each vessel (mean velocity, no dispersion within a
vessel) and uses the solved (plasma/mean) flow -- not red-cell velocity.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._route_graph import (
    _clear_edge_attributes,
    _iter_edges,
    _positive_finite,
    _resolve_terminals,
    _set_edge,
    _unavailable,
)
from .bifurcation import branch_order_category

__all__ = ["TRANSIT_TIME_EDGE_ATTRIBUTES", "compute_transit_times"]

TRANSIT_TIME_EDGE_ATTRIBUTES: tuple[str, ...] = ("transit_time_s", "arrival_time_s")
ASSUMPTIONS = "plug flow per vessel; solved mean (plasma) flow, not red-cell velocity"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _moments_summary(weight: float, first: float, second: float) -> tuple[float, float]:
    mean = first / weight
    variance = max(0.0, second / weight - mean * mean)
    return mean, math.sqrt(variance)


def compute_transit_times(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """Transit time of every vessel, and the inlet->outlet transit-time
    distribution (mean, SD, CV) for whole paths and for capillary time only.

    Needs a solved flow (``flow_abs`` on edges, ``pressure`` on nodes) and
    ``diameter_um``/``length`` on every vessel. Writes ``transit_time_s``
    and ``arrival_time_s`` (mean arrival time at the vessel's downstream
    end) onto every edge.
    """
    _clear_edge_attributes(G, TRANSIT_TIME_EDGE_ATTRIBUTES)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    if not inlets or not outlets:
        return _unavailable("Transit Time", "no inlet and outlet nodes", both=both)
    edges = list(_iter_edges(G))
    if not all(_finite(G.nodes[n].get("pressure")) for n in G.nodes if G.degree(n)) or not all(
        _finite(d.get("flow_abs")) for _u, _v, _k, d in edges
    ):
        return _unavailable("Transit Time", "needs a solved flow (run haemodynamics)", both=both)
    if not all(_positive_finite(d.get("diameter_um")) and _positive_finite(d.get("length"))
               for _u, _v, _k, d in edges):
        return _unavailable("Transit Time", "needs a diameter and length on every vessel", both=both)

    pressure = {n: float(G.nodes[n].get("pressure", math.nan)) for n in G.nodes}
    flow_max = max((float(d["flow_abs"]) for *_r, d in edges), default=0.0)
    tiny = 1e-12 * flow_max
    inflows: dict = defaultdict(list)  # downstream node -> [(upstream, tau, tau_cap, q, edge)]
    outflow: dict = defaultdict(float)  # upstream node -> flow leaving it through vessels
    transit = {}
    for u, v, key, data in edges:
        q = float(data["flow_abs"])
        if u == v or q <= tiny or math.isnan(pressure[u]) or math.isnan(pressure[v]):
            continue
        upstream, downstream = (u, v) if pressure[u] >= pressure[v] else (v, u)
        radius_m = float(data["diameter_um"]) * 0.5e-6
        volume_m3 = math.pi * radius_m**2 * float(data["length"]) * 1e-6
        tau = volume_m3 / q
        tau_cap = tau if branch_order_category(data.get("branch_order")) == "capillary" else 0.0
        transit[(u, v, key)] = tau
        inflows[downstream].append((upstream, tau, tau_cap, q, (u, v, key)))
        outflow[upstream] += q

    # (weight-normalised) first/second moments of whole-path and capillary time
    mean, second, mean_c, second_c = {}, {}, {}, {}
    for inlet in inlets:
        mean[inlet] = second[inlet] = mean_c[inlet] = second_c[inlet] = 0.0
    arrival = {}
    for node in sorted(G.nodes, key=lambda n: -pressure[n] if not math.isnan(pressure[n]) else math.inf):
        if node in inlets or node not in inflows:
            continue
        w = m1 = m2 = c1 = c2 = 0.0
        for upstream, tau, tau_cap, q, edge in inflows[node]:
            if upstream not in mean:
                continue  # fed from a region no inlet reaches
            mu, su, cu, scu = mean[upstream], second[upstream], mean_c[upstream], second_c[upstream]
            arrival[edge] = mu + tau
            w += q
            m1 += q * (mu + tau)
            m2 += q * (su + 2 * mu * tau + tau * tau)
            c1 += q * (cu + tau_cap)
            c2 += q * (scu + 2 * cu * tau_cap + tau_cap * tau_cap)
        if w > 0:
            mean[node], second[node] = m1 / w, m2 / w
            mean_c[node], second_c[node] = c1 / w, c2 / w

    # Blood leaves the network at an outlet at the rate it arrives there net
    # of what drains on through the outlet to a lower-pressure one -- counting
    # every inflow would count that through-flow at both outlets. What leaves
    # carries the outlet's own mixed arrival-time distribution.
    w = m1 = m2 = c1 = c2 = 0.0
    for outlet in outlets:
        if outlet not in mean or outlet in inlets:
            continue
        reached = sum(q for up, _t, _tc, q, _e in inflows.get(outlet, ()) if up in mean)
        leaving = max(0.0, reached - outflow.get(outlet, 0.0))
        w += leaving
        m1 += leaving * mean[outlet]
        m2 += leaving * second[outlet]
        c1 += leaving * mean_c[outlet]
        c2 += leaving * second_c[outlet]
    if w <= 0:
        return _unavailable("Transit Time", "no flow reaches an outlet from an inlet", both=both)

    # Only now the measure has an answer: nothing is written when it cannot run.
    for u, v, key, _data in edges:
        _set_edge(G, u, v, key, "transit_time_s", transit.get((u, v, key), math.nan))
        _set_edge(G, u, v, key, "arrival_time_s", arrival.get((u, v, key), math.nan))

    mtt, sd = _moments_summary(w, m1, m2)
    result: Dict[str, Any] = {
        "Transit Time Assumptions": ASSUMPTIONS,
        "Mean Transit Time (s)": mtt,
        "Transit Time SD (s)": sd,
        "Transit Time CV": sd / mtt if mtt > 0 else math.nan,
    }
    has_capillaries = any(
        branch_order_category(d.get("branch_order")) == "capillary" for *_r, d in edges
    )
    if has_capillaries:
        cmtt, cth = _moments_summary(w, c1, c2)
        result.update(
            {
                "Capillary Mean Transit Time (s)": cmtt,
                "Capillary Transit Time Heterogeneity CTH (s)": cth,
                "Capillary Transit Time CV": cth / cmtt if cmtt > 0 else math.nan,
            }
        )
    else:
        result["Capillary Transit Time Status"] = "N/A (no vessels classified as capillaries)"

    by_type: dict = defaultdict(lambda: [0.0, 0.0, []])
    for (u, v, key), tau in transit.items():
        data = G[u][v][key] if key is not None else G[u][v]
        q = float(data["flow_abs"])
        entry = by_type[branch_order_category(data.get("branch_order"))]
        entry[0] += q * tau
        entry[1] += q
        entry[2].append(tau)
    result["Vessel Transit Time By Vessel Type"] = {
        category: {
            "Flow-Weighted Mean (s)": weighted / weight,
            "Median (s)": float(np.median(taus)),
            "Vessel Count": len(taus),
        }
        for category, (weighted, weight, taus) in by_type.items()
    }
    if both:
        result["Transit Time Nodes Both Inlet And Outlet (excluded)"] = both
    return result
