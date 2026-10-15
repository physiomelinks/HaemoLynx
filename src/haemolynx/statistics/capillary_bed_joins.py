"""At which capillary branch order the beds of different arteriolar feeds join.

Capillary B orders are counted from every *feed* at once -- the arteriole
boundary nodes, or the inlets without them (``graph.branch_order``'s
:func:`~haemolynx.graph.branch_order.capillary_start_nodes`) -- so each
capillary belongs to the *bed* of the feed that reaches it first. Two beds
meet on a watershed vessel, one whose two ends were reached first from
different feeds: the rule ``territories`` applies to inlets, here with the
feeds as sources. A bed's *join order* is the lowest B order on any of its
watershed vessels -- the order at which blood from its feed can first meet
another feed's (single linkage of the beds over the B-order filtration).

A bed that never meets another is *censored*: it leaves the image (so it may
join beyond the field of view), drains into a venule, or ends blind. Join
orders are therefore time-to-event data, summarised by a Kaplan-Meier
estimate; tests between datasets (log-rank, a reseeded-feed permutation null)
are left to the per-bed table this writes beside the statistics CSV.

Building the graph breaks every loop under ``remove_short_loops_under_um``
round, so the earliest joins may be missing; the summary reports the value.
"""
from __future__ import annotations

import csv
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import networkx as nx
import numpy as np

from haemolynx.graph.branch_order import _tier_and_number
from haemolynx.graph.dead_ends import graph_image_extent_um, node_leaves_image
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges

from ._route_graph import (
    _clear_edge_attributes,
    _edge_weight_and_capacity,
    _effective_weighting,
    _iter_edges,
    _positive_finite,
    _set_edge,
    _unavailable,
)
from .territories import _labels

__all__ = [
    "BED_JOIN_EDGE_ATTRIBUTES",
    "BED_TABLE",
    "BED_TABLE_COLUMNS",
    "CENSOR_REASONS",
    "compute_capillary_bed_joins",
    "export_capillary_bed_joins_csv",
    "kaplan_meier",
    "kaplan_meier_median",
]

#: Per vessel: the bed it belongs to (its feed's label, ``F01``...; a
#: capillary no feed reaches is ``unreached``) and that bed's join order (NaN
#: for a censored bed). Only capillaries carry them.
BED_JOIN_EDGE_ATTRIBUTES: tuple[str, ...] = ("capillary_bed", "bed_join_order")
UNREACHED = "unreached"

#: Where the per-bed table waits on ``G.graph`` until
#: :func:`export_capillary_bed_joins_csv` writes it.
BED_TABLE = "capillary_bed_join_table"

#: Why a bed never joined another, in the order one is chosen when several
#: apply: an open end first, since the bed may join beyond the image.
LEAVES_IMAGE = "leaves_image"
REACHES_VENULE = "reaches_venule"
BLIND_END = "blind_end"
CENSOR_REASONS: tuple[str, ...] = (LEAVES_IMAGE, REACHES_VENULE, BLIND_END)

WITHIN_TREE = "within_tree"
BETWEEN_TREES = "between_trees"

#: The per-bed table's columns. ``join_order``/``join_distance_um`` are the
#: time-to-event pair: where the bed joins, or -- for a censored bed -- the
#: highest B order and the furthest distance it was followed to.
BED_TABLE_COLUMNS: tuple[str, ...] = (
    "bed",
    "feed_node",
    "arteriolar_tree",
    "vessel_count",
    "bed_length_um",
    "join_order",
    "join_distance_um",
    "censored",
    "censor_reason",
    "joins_bed",
    "join_class",
)

SECTION = "Capillary Bed Joins"
_PREFIX = "Capillary Bed Join"
_CENSOR_ROWS = {
    LEAVES_IMAGE: "Capillary Beds Censored, Leaving the Image",
    REACHES_VENULE: "Capillary Beds Censored, Draining to a Venule",
    BLIND_END: "Capillary Beds Censored, Ending Blind",
}


def kaplan_meier(times: Iterable[float], joined: Iterable[bool]) -> tuple[np.ndarray, np.ndarray]:
    """The Kaplan-Meier fraction of beds not yet joined.

    Returns the distinct join times and the fraction just after each. A bed
    censored at a time is still at risk at it -- censoring follows the joins
    at the same time, the usual convention.
    """
    times = np.asarray(list(times), dtype=float)
    joined = np.asarray(list(joined), dtype=bool)
    event_times = np.unique(times[joined])
    at_risk = np.asarray([(times >= t).sum() for t in event_times], dtype=float)
    events = np.asarray([(joined & (times == t)).sum() for t in event_times], dtype=float)
    return event_times, np.cumprod(1.0 - events / at_risk)


def kaplan_meier_median(times: Iterable[float], joined: Iterable[bool]) -> float:
    """The first time by which the Kaplan-Meier estimate says half the beds
    have joined; NaN when it never says so (too few join, or too many are
    censored first)."""
    event_times, survival = kaplan_meier(times, joined)
    reached = np.flatnonzero(survival <= 0.5)
    return float(event_times[reached[0]]) if len(reached) else float("nan")


def _capillary_order(data: dict) -> Optional[int]:
    parsed = _tier_and_number(data.get("branch_order"))
    if parsed is None or parsed[0] not in {"b", "bo"}:
        return None
    return parsed[1]


def _is_arteriolar(data: dict) -> bool:
    parsed = _tier_and_number(data.get("branch_order"))
    return parsed is not None and parsed[0] in {"art", "large_art"}


def _length(data: dict) -> float:
    return float(data["length"]) if _positive_finite(data.get("length")) else 0.0


@dataclass
class _Bed:
    label: str
    feed: Any
    tree: Optional[str] = None
    vessels: int = 0
    length_um: float = 0.0
    max_order: int = 0
    reach_um: float = 0.0
    leaves_image: bool = False
    reaches_venule: bool = False
    #: (join order, join distance, the other bed's label), lowest first.
    join: Optional[tuple] = None

    @property
    def censor_reason(self) -> str:
        if self.join is not None:
            return ""
        if self.leaves_image:
            return LEAVES_IMAGE
        if self.reaches_venule:
            return REACHES_VENULE
        return BLIND_END


def _nearest_feeds(adjacency: dict, feeds: Sequence[Any], drains: set) -> dict:
    """Per node, ``(distance, microns, feed index)`` of the feed reaching it
    first. A drain node is reached but not passed through: a bed ends at the
    venule it drains to."""
    counter = itertools.count()
    heap = [(0.0, 0.0, index, next(counter), feed) for index, feed in enumerate(feeds)]
    heapq.heapify(heap)
    nearest: dict = {}
    while heap:
        distance, microns, index, _tie, node = heapq.heappop(heap)
        if node in nearest:
            continue
        nearest[node] = (distance, microns, index)
        if node in drains:
            continue
        for neighbour, weight, length in adjacency[node]:
            if neighbour not in nearest:
                heapq.heappush(
                    heap, (distance + weight, microns + length, index, next(counter), neighbour)
                )
    return nearest


def _arteriolar_trees(G: nx.Graph, feeds: Sequence[Any]) -> list[Optional[str]]:
    """Per feed, the arteriolar tree (``T01``...) it hangs from -- the piece
    of Art/Large_Art vessels it touches -- or None for a feed on none."""
    parent: dict = {}

    def find(x: Any) -> Any:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _key, data in _iter_edges(G):
        if _is_arteriolar(data):
            parent[find(u)] = find(v)
    names: dict = {}
    trees: list[Optional[str]] = []
    for feed in feeds:
        if feed not in parent:
            trees.append(None)
            continue
        root = find(feed)
        names.setdefault(root, f"T{len(names) + 1:02d}")
        trees.append(names[root])
    return trees


def compute_capillary_bed_joins(
    G: nx.Graph,
    feed_nodes: Optional[Iterable[Any]],
    drain_nodes: Optional[Iterable[Any]] = None,
    *,
    weighting: str = "topology",
    short_loop_removal_um: Optional[float] = None,
) -> Dict[str, Any]:
    """Each feed's capillary bed, where it first joins another's, and a
    Kaplan-Meier summary of those join orders: see the module docstring.

    *feed_nodes* are the nodes the capillary B orders were counted from,
    *drain_nodes* the venous ones their beds drain to
    (:func:`~haemolynx.graph.branch_order.capillary_start_nodes` gives both).
    Capillaries are the vessels with a ``B``/``BO`` ``branch_order``; a
    zero-resistance bridge is no step. *weighting* is a route weighting
    (``topology`` -- the hop count B orders are -- ``length`` or
    ``resistance``) deciding which feed reaches a capillary first; the join
    order is always the watershed vessel's own B order. The join distance is
    half the length of the shortest feed-to-feed path through that vessel --
    how far blood from each feed travels before the two meet.

    Writes :data:`BED_JOIN_EDGE_ATTRIBUTES` on the capillaries and leaves the
    per-bed table on ``G.graph[BED_TABLE]`` for
    :func:`export_capillary_bed_joins_csv`. *short_loop_removal_um*, when
    given, is the ``remove_short_loops_under_um`` the graph was built with,
    reported alongside. Without capillary B orders or a feed among them,
    reports why and writes nothing.
    """
    _clear_edge_attributes(G, BED_JOIN_EDGE_ATTRIBUTES)
    G.graph.pop(BED_TABLE, None)
    work, became = G, {}
    if any(data.get(IS_ZERO_RESISTANCE) for *_e, data in _iter_edges(G)):
        work, became = contract_bridges(G)

    capillaries = []
    for u, v, key, data in _iter_edges(work):
        order = _capillary_order(data)
        if order is not None:
            capillaries.append((u, v, key, data, order))
    if not capillaries:
        return {SECTION: _unavailable(_PREFIX, "no capillary branch orders")}
    on_capillaries = nx.MultiGraph()
    on_capillaries.add_edges_from((u, v, data) for u, v, _k, data, _o in capillaries)
    # Each feed as the contracted graph names it, and as the caller did.
    given: dict = {}
    for node in feed_nodes or ():
        given.setdefault(became.get(node, node), node)
    feeds = sorted((n for n in given if n in on_capillaries), key=lambda n: repr(given[n]))
    if not feeds:
        return {SECTION: _unavailable(_PREFIX, "no capillary feed nodes")}

    effective, label = _effective_weighting(on_capillaries, weighting)
    lengths_known = all(_positive_finite(data.get("length")) for *_e, data, _o in capillaries)
    adjacency: dict = defaultdict(list)
    for u, v, _key, data, _order in capillaries:
        if u != v:
            weight = _edge_weight_and_capacity(data, effective)[0]
            adjacency[u].append((v, weight, _length(data)))
            adjacency[v].append((u, weight, _length(data)))
    drains = {became.get(n, n) for n in (drain_nodes or ())} - set(feeds)
    nearest = _nearest_feeds(adjacency, feeds, drains)

    labels = _labels({given[feed] for feed in feeds}, "F")
    beds = [_Bed(label=labels[given[feed]], feed=given[feed]) for feed in feeds]
    for bed, tree in zip(beds, _arteriolar_trees(work, feeds)):
        bed.tree = tree
    extent = graph_image_extent_um(work)

    def leaves(node: Any) -> bool:
        return work.degree(node) == 1 and node_leaves_image(work, node, extent)

    owners: list = []
    unreached = 0
    for u, v, key, data, order in capillaries:
        ends = [(n, nearest[n]) for n in (u, v) if n in nearest]
        if not ends:
            owners.append((u, v, key, None))
            unreached += 1
            continue
        owner, entry = min(
            [(n, e) for n, e in ends if n not in drains] or ends, key=lambda end: end[1]
        )
        far = v if owner == u else u
        far_entry = nearest.get(far)
        bed = beds[entry[2]]
        length = _length(data)
        bed.vessels += 1
        bed.length_um += length
        bed.max_order = max(bed.max_order, order)
        # Furthest point of the vessel from the feed: midway round a loop
        # within the bed, else its far end.
        same_bed = far_entry is not None and far_entry[2] == entry[2]
        far_microns = far_entry[1] if same_bed else entry[1] + length
        bed.reach_um = max(bed.reach_um, (entry[1] + length + far_microns) / 2.0)
        bed.reaches_venule |= u in drains or v in drains
        bed.leaves_image |= leaves(u) or leaves(v)
        owners.append((u, v, key, entry[2]))

        a, b = nearest.get(u), nearest.get(v)
        if a is None or b is None or a[2] == b[2] or u in drains or v in drains:
            continue
        distance = (a[1] + length + b[1]) / 2.0
        for mine, theirs in ((a[2], b[2]), (b[2], a[2])):
            candidate = (order, distance, beds[theirs].label)
            if beds[mine].join is None or candidate < beds[mine].join:
                beds[mine].join = candidate

    for u, v, key, index in owners:
        if index is None:
            _set_edge(work, u, v, key, "capillary_bed", UNREACHED)
            _set_edge(work, u, v, key, "bed_join_order", float("nan"))
            continue
        bed = beds[index]
        _set_edge(work, u, v, key, "capillary_bed", bed.label)
        join_order = float(bed.join[0]) if bed.join else float("nan")
        _set_edge(work, u, v, key, "bed_join_order", join_order)

    by_label = {bed.label: bed for bed in beds}
    table = []
    for bed in beds:
        partner = by_label[bed.join[2]] if bed.join else None
        join_class = ""
        if partner is not None and bed.tree is not None and partner.tree is not None:
            join_class = WITHIN_TREE if bed.tree == partner.tree else BETWEEN_TREES
        order = bed.join[0] if bed.join else bed.max_order
        distance = bed.join[1] if bed.join else bed.reach_um
        table.append(
            {
                "bed": bed.label,
                "feed_node": bed.feed,
                "arteriolar_tree": bed.tree or "",
                "vessel_count": bed.vessels,
                "bed_length_um": bed.length_um if lengths_known else "",
                "join_order": order,
                "join_distance_um": distance if lengths_known else "",
                "censored": bed.join is None,
                "censor_reason": bed.censor_reason,
                "joins_bed": partner.label if partner is not None else "",
                "join_class": join_class,
            }
        )
    G.graph[BED_TABLE] = table

    joined = [bed.join is not None for bed in beds]
    rows: Dict[str, Any] = {
        f"{_PREFIX} Route Weighting": label,
        "Capillary Beds": len(beds),
        "Capillary Beds Joined": int(sum(joined)),
        "Capillary Beds Joined (fraction)": float(np.mean(joined)),
    }
    for reason, row in _CENSOR_ROWS.items():
        rows[row] = sum(1 for bed in beds if bed.censor_reason == reason)
    median = kaplan_meier_median([r["join_order"] for r in table], joined)
    rows["Capillary Bed Join Order, Kaplan-Meier Median"] = (
        median if np.isfinite(median) else "N/A (fewer than half the beds join)"
    )
    if lengths_known:
        median = kaplan_meier_median([r["join_distance_um"] for r in table], joined)
        rows["Capillary Bed Join Distance, Kaplan-Meier Median (microns)"] = (
            median if np.isfinite(median) else "N/A (fewer than half the beds join)"
        )
    else:
        rows["Capillary Bed Join Distance, Kaplan-Meier Median (microns)"] = (
            "N/A (vessel lengths unavailable)"
        )
    if any(bed.tree is not None for bed in beds):
        classes = [r["join_class"] for r in table]
        rows["Capillary Beds Joining Within Their Arteriolar Tree"] = classes.count(WITHIN_TREE)
        rows["Capillary Beds Joining Another Arteriolar Tree"] = classes.count(BETWEEN_TREES)
    if unreached:
        rows["Capillaries No Feed Reaches"] = unreached
    if short_loop_removal_um is not None:
        rows["Graph Built With Short Loops Removed Under (microns)"] = float(short_loop_removal_um)
    return {SECTION: rows}


def export_capillary_bed_joins_csv(
    G: nx.Graph, output_csv_path: Union[str, Path]
) -> Optional[Path]:
    """Write the per-bed table :func:`compute_capillary_bed_joins` left on
    *G* (one row per bed, :data:`BED_TABLE_COLUMNS`) and take it off the
    graph. None, and no file, when the analysis did not run."""
    table = G.graph.pop(BED_TABLE, None)
    if not table:
        return None
    path = Path(output_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BED_TABLE_COLUMNS)
        writer.writeheader()
        for row in table:
            writer.writerow(
                {
                    name: (f"{value:.6g}" if isinstance(value, float) else value)
                    for name, value in row.items()
                }
            )
    return path
