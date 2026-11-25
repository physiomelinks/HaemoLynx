"""Horton-Strahler ordering of the arterial and venous trees, and Horton's laws.

Strahler order is defined on trees: a terminal segment is order 1, and a
segment's order goes up by one only where two segments of the same highest
order join. So it is computed separately on the arterial tree (Large_Art and
Art vessels, rooted at the inlets) and the venous tree (Ven and Large_Ven,
rooted at the outlets), each taken as the shortest-length spanning tree from
its roots; the capillary mesh between them is left unordered, as are
vessels closing a loop within a tree (anastomoses). On a network with no
arteriole/venule classification it falls back to the whole network's tree
from the inlets, flagged approximate.

Horton's laws: per order, the number, mean length and mean diameter of
*elements* (consecutive segments of the same order, merged) change by a
roughly constant ratio per order; R_B, R_L, R_D are those ratios from a
log-linear fit.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._route_graph import (
    _clear_edge_attributes,
    _effective_weighting,
    _iter_edges,
    _positive_finite,
    _set_edge,
)
from .bifurcation import branch_order_category

__all__ = ["STRAHLER_EDGE_ATTRIBUTES", "compute_strahler_orders"]

STRAHLER_EDGE_ATTRIBUTES: tuple[str, ...] = ("strahler_order",)
ARTERIAL = frozenset({"large_arteriole", "arteriole"})
VENOUS = frozenset({"venule", "large_venule"})


def _tree_orders(G, edge_ids: list, roots: set, weighting: str) -> tuple[dict, list]:
    """Strahler order per original edge on the shortest-path tree of the
    vessels *edge_ids* from *roots*, and the tree's elements."""
    S = nx.Graph()
    members: dict = defaultdict(list)
    for u, v, key in edge_ids:
        data = G[u][v][key] if key is not None else G[u][v]
        w = float(data["length"]) if weighting == "length" else 1.0
        members[frozenset((u, v))].append((u, v, key, w))
        if not S.has_edge(u, v) or w < S[u][v]["weight"]:
            S.add_edge(u, v, weight=w)
    roots = {r for r in roots if r in S}
    if not roots:
        return {}, []
    _dist, paths = nx.multi_source_dijkstra(S, roots, weight="weight")
    parent = {n: p[-2] for n, p in paths.items() if len(p) > 1}
    children: dict = defaultdict(list)
    for node, par in parent.items():
        children[par].append(node)

    order: dict = {}
    for node in sorted(parent, key=lambda n: -_dist[n]):  # deepest first
        kids = [order[c] for c in children.get(node, ())]
        if not kids:
            order[node] = 1
        else:
            top = max(kids)
            order[node] = top + 1 if kids.count(top) >= 2 else top

    edge_order: dict = {}
    tree_edge: dict = {}
    for node, par in parent.items():
        best = min(members[frozenset((node, par))], key=lambda m: m[3])
        edge_order[best[:3]] = order[node]
        tree_edge[node] = best

    elements = []  # (order, [original edges])
    for node, par in parent.items():
        k = order[node]
        if par in parent and order[par] == k:
            continue  # not the top of its element
        chain = [tree_edge[node][:3]]
        current = node
        while True:
            same = [c for c in children.get(current, ()) if order[c] == k]
            if not same:
                break
            current = same[0]
            chain.append(tree_edge[current][:3])
        elements.append((k, chain))
    return edge_order, elements


def _horton(G, elements: list, weighting: str) -> Dict[str, Any]:
    by_order: dict = defaultdict(lambda: {"count": 0, "lengths": [], "diameters": []})
    for k, chain in elements:
        entry = by_order[k]
        entry["count"] += 1
        length = 0.0
        diameter_weighted = 0.0
        has_diameter = True
        for u, v, key in chain:
            data = G[u][v][key] if key is not None else G[u][v]
            seg = float(data["length"]) if weighting == "length" else 1.0
            length += seg
            if _positive_finite(data.get("diameter_um")):
                diameter_weighted += seg * float(data["diameter_um"])
            else:
                has_diameter = False
        entry["lengths"].append(length)
        if has_diameter and length > 0:
            entry["diameters"].append(diameter_weighted / length)

    unit = "um" if weighting == "length" else "vessels"
    orders = sorted(by_order)
    table = {}
    for k in orders:
        entry = by_order[k]
        row: Dict[str, Any] = {
            "Element Count": entry["count"],
            f"Mean Element Length ({unit})": float(np.mean(entry["lengths"])),
        }
        if entry["diameters"]:
            row["Mean Element Diameter (um)"] = float(np.mean(entry["diameters"]))
        table[str(k)] = row
    result: Dict[str, Any] = {"Max Order": max(orders) if orders else 0, "Orders": table}

    def ratio(values: list, invert: bool) -> Any:
        pairs = [(k, v) for k, v in zip(orders, values) if v is not None and v > 0]
        if len(pairs) < 2:
            return "N/A (fewer than two orders)"
        ks, vs = zip(*pairs)
        slope = np.polyfit(np.asarray(ks, float), np.log(np.asarray(vs, float)), 1)[0]
        return float(math.exp(-slope if invert else slope))

    result["Bifurcation Ratio R_B"] = ratio([by_order[k]["count"] for k in orders], invert=True)
    result["Length Ratio R_L"] = ratio([float(np.mean(by_order[k]["lengths"])) for k in orders], invert=False)
    result["Diameter Ratio R_D"] = ratio(
        [float(np.mean(by_order[k]["diameters"])) if by_order[k]["diameters"] else None for k in orders],
        invert=False,
    )
    return result


def compute_strahler_orders(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """Strahler orders and Horton ratios for the arterial and venous trees.

    Writes ``strahler_order`` onto every edge (NaN on capillaries,
    anastomoses and anything not reached from a root). Length-weighted
    trees and element lengths (vessel counts without lengths).
    """
    _clear_edge_attributes(G, STRAHLER_EDGE_ATTRIBUTES)
    weighting, _label = _effective_weighting(G, "length")
    weighting = "length" if weighting == "length" else "topology"
    inlets = {n for n in (inlet_nodes or ()) if n in G}
    outlets = {n for n in (outlet_nodes or ()) if n in G}

    arterial, venous, everything = [], [], []
    for u, v, key, data in _iter_edges(G):
        if u == v:
            continue
        category = branch_order_category(data.get("branch_order"))
        everything.append((u, v, key))
        if category in ARTERIAL:
            arterial.append((u, v, key))
        elif category in VENOUS:
            venous.append((u, v, key))

    sides = []
    if arterial or venous:
        method = "arterial and venous trees (capillaries unordered)"
        if arterial:
            sides.append(("Arterial", arterial, inlets))
        if venous:
            sides.append(("Venous", venous, outlets))
    else:
        method = "approximate: no arteriole/venule classification, whole-network tree from the inlets"
        sides.append(("Network", everything, inlets))

    result: Dict[str, Any] = {"Strahler Method": method}
    if not any(roots for _n, _e, roots in sides):
        result["Strahler Status"] = "N/A (no inlet/outlet nodes to root the trees at)"
        return result

    all_orders: dict = {}
    for name, edge_ids, roots in sides:
        orders, elements = _tree_orders(G, edge_ids, roots, weighting)
        if not orders:
            result[f"Strahler {name}"] = "N/A (no root on this tree)"
            continue
        all_orders.update(orders)
        summary = _horton(G, elements, weighting)
        summary["Unordered Vessels (anastomoses or unreached)"] = len(edge_ids) - len(orders)
        result[f"Strahler {name}"] = summary

    for u, v, key, _data in _iter_edges(G):
        _set_edge(G, u, v, key, "strahler_order", float(all_orders.get((u, v, key), math.nan)))
    return result
