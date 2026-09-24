"""Arteriolar/venular perfusion territories and the watersheds between them.

Each vessel belongs to the inlet (arteriolar territory) and the outlet
(venular territory) that reach it most cheaply, under the same route
weighting as the other route analyses. A *watershed* vessel sits on the
border between two territories -- its two ends are reached first from
different inlets (or outlets) -- the zones classically most vulnerable to
hypoperfusion, being the furthest from any single supply.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._route_graph import (
    _clear_edge_attributes,
    _effective_weighting,
    _iter_edges,
    _positive_finite,
    _resolve_terminals,
    _route_graph,
    _set_edge,
    _unavailable,
)

__all__ = ["TERRITORY_EDGE_ATTRIBUTES", "compute_perfusion_territories"]

TERRITORY_EDGE_ATTRIBUTES: tuple[str, ...] = (
    "arteriolar_territory",
    "venular_territory",
    "watershed",
    "arteriolar_watershed_margin",
)
UNREACHED = "unreached"


def _two_nearest_sources(H: nx.Graph, sources: list) -> dict:
    """For every node, its (distance, source) to the nearest two *distinct*
    sources -- a Dijkstra that lets each node be settled once per source,
    at most twice in total."""
    best: dict = defaultdict(list)
    counter = itertools.count()
    heap = [(0.0, next(counter), s, s) for s in sources]
    heapq.heapify(heap)
    while heap:
        distance, _tie, node, source = heapq.heappop(heap)
        entries = best[node]
        if len(entries) >= 2 or any(src == source for _d, src in entries):
            continue
        entries.append((distance, source))
        for neighbor, attrs in H[node].items():
            known = best.get(neighbor, ())
            if len(known) < 2 and not any(src == source for _d, src in known):
                heapq.heappush(
                    heap, (distance + attrs["route_weight"], next(counter), neighbor, source)
                )
    return best


def _labels(terminals: set, prefix: str) -> dict:
    ordered = sorted(terminals, key=repr)
    width = max(2, len(str(len(ordered))))
    return {t: f"{prefix}{i + 1:0{width}d}" for i, t in enumerate(ordered)}


def _margin(entries: list) -> float:
    if len(entries) < 2:
        return 1.0
    (d1, _s1), (d2, _s2) = entries[0], entries[1]
    return (d2 - d1) / (d2 + d1) if d2 + d1 > 0 else 0.0


def _side_summary(prefix: str, labels: list, watershed: list, lengths: list) -> Dict[str, Any]:
    reached = [label for label in labels if label != UNREACHED]
    sizes: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for label, length in zip(labels, lengths):
        if label != UNREACHED:
            sizes[label] += length
            counts[label] += 1
    total_length = float(sum(lengths))
    shed_length = float(sum(l for l, w in zip(lengths, watershed) if w))
    result: Dict[str, Any] = {
        f"{prefix} Territory Count": len(counts),
        f"{prefix} Watershed Vessel Count": int(sum(watershed)),
        f"{prefix} Watershed Length Fraction": shed_length / total_length if total_length else 0.0,
        f"{prefix} Unreached Vessel Count": len(labels) - len(reached),
    }
    if counts:
        vessel_counts = np.asarray(list(counts.values()))
        length_sums = np.asarray(list(sizes.values()))
        result.update(
            {
                f"{prefix} Territory Size Min (vessels)": int(vessel_counts.min()),
                f"{prefix} Territory Size Median (vessels)": float(np.median(vessel_counts)),
                f"{prefix} Territory Size Max (vessels)": int(vessel_counts.max()),
                f"{prefix} Territory Length Median": float(np.median(length_sums)),
            }
        )
    return result


def compute_perfusion_territories(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
) -> Dict[str, Any]:
    """Arteriolar and venular territories, watershed vessels between them,
    and how close each vessel is to a border.

    Territory labels are ``A01, A02, ...`` per inlet and ``V01, ...`` per
    outlet (in node-id order). ``arteriolar_watershed_margin`` is
    ``(d2 - d1) / (d2 + d1)`` for the nearest two inlets at distances
    ``d1 <= d2`` -- 0 on a border, 1 deep inside one territory (or with a
    single reachable inlet) -- the lower of the vessel's two ends.
    ``watershed`` is ``none``/``arteriolar``/``venular``/``both``.
    """
    _clear_edge_attributes(G, TERRITORY_EDGE_ATTRIBUTES)
    effective, label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    if not inlets or not outlets:
        return _unavailable("Territory", "no inlet and outlet nodes", label, both)
    H = _route_graph(G, effective).graph
    arterial = _two_nearest_sources(H, sorted(inlets, key=repr))
    venous = _two_nearest_sources(H, sorted(outlets, key=repr))
    art_names = _labels(inlets, "A")
    ven_names = _labels(outlets, "V")

    def nearest(entries_by_node: dict, node: Any) -> Any:
        entries = entries_by_node.get(node)
        return entries[0] if entries else None

    def edge_territory(entries_by_node: dict, names: dict, u: Any, v: Any) -> tuple[str, bool]:
        a, b = nearest(entries_by_node, u), nearest(entries_by_node, v)
        if a is None and b is None:
            return UNREACHED, False
        closer = min((x for x in (a, b) if x is not None), key=lambda e: e[0])
        border = a is not None and b is not None and a[1] != b[1]
        return names[closer[1]], border

    art_labels, ven_labels, art_shed, ven_shed, lengths = [], [], [], [], []
    for u, v, key, data in _iter_edges(G):
        art, art_border = edge_territory(arterial, art_names, u, v)
        ven, ven_border = edge_territory(venous, ven_names, u, v)
        watershed = {
            (False, False): "none",
            (True, False): "arteriolar",
            (False, True): "venular",
            (True, True): "both",
        }[(art_border, ven_border)]
        margin = min(_margin(arterial.get(u, [])), _margin(arterial.get(v, [])))
        if art == UNREACHED:
            margin = float("nan")
        _set_edge(G, u, v, key, "arteriolar_territory", art)
        _set_edge(G, u, v, key, "venular_territory", ven)
        _set_edge(G, u, v, key, "watershed", watershed)
        _set_edge(G, u, v, key, "arteriolar_watershed_margin", margin)
        art_labels.append(art)
        ven_labels.append(ven)
        art_shed.append(art_border)
        ven_shed.append(ven_border)
        lengths.append(float(data["length"]) if _positive_finite(data.get("length")) else 1.0)

    result: Dict[str, Any] = {"Territory Route Weighting": label}
    result.update(_side_summary("Arteriolar", art_labels, art_shed, lengths))
    result.update(_side_summary("Venular", ven_labels, ven_shed, lengths))
    if both:
        result["Territory Nodes Both Inlet And Outlet (excluded)"] = both
    return result
