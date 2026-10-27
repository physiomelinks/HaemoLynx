"""How many capillary branch orders lie between the arterioles and the venules.

Capillary B orders are counted from the *feeds* -- the arteriole boundary
nodes and, with a Large tier, where capillaries leave the large arteriole
(``graph.branch_order``'s
:func:`~haemolynx.graph.branch_order.capillary_start_nodes`) -- so a route
from an arteriole through capillaries B01, B02 ... Bn into a venule holds n
branch orders. n is counted for every capillary that drains into a venule
(has one end at a *drain*: a venule boundary node, or where a capillary joins
the large venule) as the fewest capillaries on a route from a feed ending in
it, through capillaries alone and past no other drain. The summary is n's
mean, median, SD and range over those capillaries.

The same routes are measured in microns: per draining capillary, the
shortest route by centreline length (the vessels' ``length``) from a feed
ending in it, through capillaries alone and past no other drain -- not
necessarily the route with the fewest capillaries. Its minimum is the
shortest arteriole-to-venule route along the capillary bed.

n is not read from the draining capillary's own B order. The orders are
counted through the whole network, drains included, so a capillary meeting a
venule is numbered from whichever of its ends is nearer a feed -- which can be
the venule's, reached along another capillary -- and its label then
understates the route it ends.

Without an Art, Ven or Large tier the B orders count every vessel from the
inlets, so there is no arteriole or venule to lie between: it reports N/A.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.branch_order import _tier_and_number
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges

from ._route_graph import _iter_edges, _positive_finite, _unavailable

__all__ = ["SECTION", "compute_capillary_path_orders"]

SECTION = "Capillary Orders Between Arteriole and Venule"
_PREFIX = "Capillary Path Orders"
_ORDERS = "Capillary Orders Between Arteriole and Venule"
_LENGTHS = "Capillary Route Length Between Arteriole and Venule"
_CAPILLARY_TIERS = frozenset({"b", "bo"})
_ARTERIOLE_VENULE_TIERS = frozenset({"art", "large_art", "ven", "large_ven"})


def _tier(data: dict) -> Optional[str]:
    parsed = _tier_and_number(data.get("branch_order"))
    return parsed[0] if parsed is not None else None


def _hops_from_feeds(adjacency: dict, feeds: Iterable[Any], drains: set) -> dict:
    """Per node, the fewest capillaries from a feed to it. A drain is reached
    but not passed through: a route ends at the venule it drains into."""
    hops = {feed: 0 for feed in feeds}
    queue = deque(hops)
    while queue:
        node = queue.popleft()
        if node in drains:
            continue
        for neighbour, _length in adjacency[node]:
            if neighbour not in hops:
                hops[neighbour] = hops[node] + 1
                queue.append(neighbour)
    return hops


def _microns_from_feeds(adjacency: dict, feeds: Sequence[Any], drains: set) -> dict:
    """Per node, the shortest capillary route from a feed to it, in microns,
    passing no drain as :func:`_hops_from_feeds` does."""
    heap = [(0.0, index, feed) for index, feed in enumerate(feeds)]
    heapq.heapify(heap)
    counter = itertools.count(len(heap))
    microns: dict = {}
    while heap:
        distance, _tie, node = heapq.heappop(heap)
        if node in microns:
            continue
        microns[node] = distance
        if node in drains:
            continue
        for neighbour, length in adjacency[node]:
            if neighbour not in microns:
                heapq.heappush(heap, (distance + length, next(counter), neighbour))
    return microns


def _summary(name: str, values: np.ndarray, unit: str = "") -> Dict[str, Any]:
    return {
        f"{name}, Mean{unit}": float(values.mean()),
        f"{name}, Median{unit}": float(np.median(values)),
        f"{name}, SD{unit}": float(values.std(ddof=1)) if len(values) > 1 else "N/A (one route)",
        f"{name}, Min{unit}": values.min().item(),
        f"{name}, Max{unit}": values.max().item(),
    }


def compute_capillary_path_orders(
    G: nx.Graph,
    feed_nodes: Optional[Iterable[Any]],
    drain_nodes: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """The capillary branch orders between the arterioles and the venules:
    see the module docstring.

    *feed_nodes* are the nodes the capillary B orders were counted from,
    *drain_nodes* the venous ones the capillaries drain into
    (:func:`~haemolynx.graph.branch_order.capillary_start_nodes` gives both);
    a node given as both is a feed. Capillaries are the vessels with a
    ``B``/``BO`` ``branch_order``; a zero-resistance bridge is no step. A
    capillary with both ends at drains runs between two venules and is no
    route's last vessel. Reports why when it cannot count a route, and why
    there are no route lengths when a capillary has no positive ``length``.
    """
    work, became = G, {}
    if any(data.get(IS_ZERO_RESISTANCE) for *_e, data in _iter_edges(G)):
        work, became = contract_bridges(G)

    tiers = [_tier(data) for *_e, data in _iter_edges(work)]
    capillaries = [
        (u, v, data)
        for (u, v, _k, data), tier in zip(_iter_edges(work), tiers)
        if tier in _CAPILLARY_TIERS
    ]
    if not capillaries:
        return {SECTION: _unavailable(_PREFIX, "no capillary branch orders")}
    if not _ARTERIOLE_VENULE_TIERS & set(tiers):
        return {
            SECTION: _unavailable(
                _PREFIX,
                "no arteriole or venule branch orders: the B orders count every vessel from the inlets",
            )
        }

    lengths_known = all(_positive_finite(data.get("length")) for *_e, data in capillaries)
    adjacency: dict = defaultdict(list)
    for u, v, data in capillaries:
        if u != v:
            length = float(data["length"]) if lengths_known else 0.0
            adjacency[u].append((v, length))
            adjacency[v].append((u, length))
    feeds = {became.get(n, n) for n in (feed_nodes or ())} & adjacency.keys()
    if not feeds:
        return {SECTION: _unavailable(_PREFIX, "no capillary feed nodes")}
    drains = ({became.get(n, n) for n in (drain_nodes or ())} & adjacency.keys()) - feeds
    feeds = sorted(feeds, key=repr)
    hops = _hops_from_feeds(adjacency, feeds, drains)
    microns = _microns_from_feeds(adjacency, feeds, drains) if lengths_known else {}

    routes = 0
    orders: list[int] = []
    lengths: list[float] = []
    for u, v, data in capillaries:
        if u == v or (u in drains) == (v in drains):
            continue
        routes += 1
        inner = v if u in drains else u
        if inner in hops:
            orders.append(hops[inner] + 1)
            if lengths_known:
                lengths.append(microns[inner] + float(data["length"]))
    if not routes:
        return {SECTION: _unavailable(_PREFIX, "no capillary drains into a venule")}

    rows: Dict[str, Any] = {}
    if orders:
        rows.update(_summary(_ORDERS, np.asarray(orders)))
        if lengths_known:
            rows.update(_summary(_LENGTHS, np.asarray(lengths), " (microns)"))
        else:
            rows[f"{_LENGTHS} Status"] = "N/A (vessel lengths unavailable)"
    else:
        rows[f"{_PREFIX} Status"] = "N/A (no arteriole reaches a venule through capillaries)"
    rows["Capillaries Draining to a Venule"] = routes
    rows["Capillaries Draining to a Venule No Arteriole Reaches"] = routes - len(orders)
    return {SECTION: rows}
