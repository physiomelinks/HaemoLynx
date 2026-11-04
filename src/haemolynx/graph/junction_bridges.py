"""Zero-resistance bridges where a narrow vessel opens into a wider one, at
junctions no thick vessel is part of.

A centreline runs to the middle of the junction, so a vessel opening into a
wider one counts the stretch across that vessel's lumen -- about its radius --
as its own length, and its resistance with it. The thick-vessel split takes
that stretch out where a vessel opens into a thick one
(:mod:`.thick_vessel_junctions`); this does it at every other junction, from
the widths the Diameters stage measured: each vessel narrower than
:data:`NARROWER_THAN` of the widest vessel there is cut as far along its
centreline as that vessel's radius, and the stretch becomes a bridge. A vessel
running through the junction, and vessels of one width meeting, stay whole.

A bridge here is an ``IS_ZERO_RESISTANCE`` edge like the thick split's: its
resistance is negligible, it is not measured, it gets no branch order, and
the statistics contract it. Unlike the thick split's, the branch-order search
crosses it as the vessel it was cut from (it keeps that vessel's mask tags),
so every vessel's order is what it was without it. It is marked
:data:`JUNCTION_BRIDGE`, and the node where it meets its vessel
:data:`JUNCTION_BRIDGE_END`, so :func:`remove_junction_bridges` can put the
vessel back whole.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

from haemolynx.geometry import cumulative_lengths

from ._helpers import calculate_path_length, edge_id, edge_sample_points, next_node_id, sort_nodes
from .crop import LENGTH_EDGE_ATTRIBUTES
from .dead_ends import THICK_CENTRELINE
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

__all__ = [
    "JUNCTION_BRIDGE",
    "JUNCTION_BRIDGE_END",
    "MIN_VESSEL_LEFT_UM",
    "NARROWER_THAN",
    "has_junction_bridges",
    "insert_junction_bridges",
    "rebridge_junctions",
    "remove_junction_bridges",
]

#: Edge attribute: a zero-resistance bridge :func:`insert_junction_bridges` drew.
JUNCTION_BRIDGE = "junction_bridge"

#: Node attribute: where a junction bridge meets the vessel it was cut from.
JUNCTION_BRIDGE_END = "junction_bridge_end"

#: A vessel narrower than this share of the widest vessel at its junction is
#: bridged: the share the thick split's fork rule calls a branch
#: (``thick_vessel_junctions.FORK_BRANCH_MAX_RADIUS_FRACTION``), so the two
#: halves of a vessel running through, measured a little apart, are not.
NARROWER_THAN = 0.8

#: A vessel its bridges would leave shorter than this (um) lies across the
#: wider vessel's lumen, and is left whole.
MIN_VESSEL_LEFT_UM = 1.0

#: Two arc lengths this close (um) are one point: a cut on a vertex adds none.
_SAME_POINT_UM = 1e-9


def _width_um(data: Mapping[str, Any]) -> float | None:
    try:
        width = float(data.get("diameter_um"))
    except (TypeError, ValueError):
        return None
    return width if np.isfinite(width) and width > 0 else None


class _ThickLumen:
    """Whether a point lies in a thick vessel's lumen: within the lumen's
    radius (the split's tolerance where that is unknown) of the centreline
    tree graph building kept on the graph (``dead_ends.record_thick_centreline``)."""

    def __init__(self, G: nx.MultiGraph):
        self._tree = None
        record = G.graph.get(THICK_CENTRELINE)
        if not record:
            return
        points = np.asarray(record["points_um"], dtype=float).reshape(-1, 3)
        if not len(points):
            return
        from scipy.spatial import cKDTree

        radius = np.asarray(record["radius_um"], dtype=float).reshape(-1)
        self._reach = np.fmax(radius, float(record.get("tolerance_um") or 0.0))
        self._tree = cKDTree(points)

    def holds(self, point: Any) -> bool:
        if self._tree is None or point is None:
            return False
        distance, index = self._tree.query(np.asarray(point, dtype=float).reshape(3))
        return bool(distance <= self._reach[index])


def _point_at(path: np.ndarray, along: np.ndarray, s: float) -> np.ndarray:
    i = int(np.clip(np.searchsorted(along, s, side="right") - 1, 0, len(path) - 2))
    step = along[i + 1] - along[i]
    t = 0.0 if step <= 0 else (s - along[i]) / step
    return path[i] + t * (path[i + 1] - path[i])


def _piece(path: np.ndarray, along: np.ndarray, start: float, stop: float) -> np.ndarray:
    """The stretch of *path* from arc length *start* to *stop*."""
    inside = path[(along > start + _SAME_POINT_UM) & (along < stop - _SAME_POINT_UM)]
    return np.vstack([_point_at(path, along, start), inside, _point_at(path, along, stop)])


def _vessel(data: Mapping[str, Any], points: np.ndarray) -> dict[str, Any]:
    attrs = {k: v for k, v in data.items() if k not in ("voxels", "length", *LENGTH_EDGE_ATTRIBUTES)}
    attrs["voxels"] = points.tolist()
    attrs["length"] = float(calculate_path_length(attrs["voxels"]))
    return attrs


def _bridge(data: Mapping[str, Any], points: np.ndarray, width_um: float) -> dict[str, Any]:
    """The bridge cut from vessel *data*. It keeps the vessel's mask tags, so
    it lies in the same tier as the vessel it stands for at the junction,
    and none of what was measured of the vessel or labelled on it."""
    from haemolynx.haemodynamics.poiseuille import clear_edge_measurements

    from .branch_order import BRANCH_ORDER_BASIS
    from .post_processing import EDITED

    attrs = _vessel(data, points)
    clear_edge_measurements(attrs)
    for name in ("branch_order", BRANCH_ORDER_BASIS, EDITED):
        attrs.pop(name, None)
    attrs.update({
        IS_ZERO_RESISTANCE: True,
        JUNCTION_BRIDGE: True,
        # As every bridge: the widest vessel at its ends, never measured.
        "diameter_um": float(width_um),
        "diameter_source": "bridge",
    })
    return attrs


def has_junction_bridges(G: nx.MultiGraph) -> bool:
    """Whether *G* carries a bridge :func:`insert_junction_bridges` drew."""
    return any(data.get(JUNCTION_BRIDGE) for _u, _v, data in G.edges(data=True))


def insert_junction_bridges(G: nx.MultiGraph, *, nodes: Iterable[Any] | None = None) -> dict[str, int]:
    """Bridge, in place, each vessel narrower than :data:`NARROWER_THAN` of
    the widest vessel at a junction, as far along it from the junction as
    that vessel's radius, at each junction of *nodes* (default every node)
    that no bridge meets and no thick vessel's lumen holds.

    Widths are each vessel's ``diameter_um``; a junction where a vessel has
    none is left as it is. A vessel the bridges at its ends would leave
    shorter than :data:`MIN_VESSEL_LEFT_UM` keeps its whole length, and so
    does one a person made of a bridge on the Post processing tab
    (``post_processing.BRIDGE_SET_BY_HAND`` False). Returns
    counts: ``bridges`` drawn, ``junctions`` bridged at, and the junctions
    and vessels left as they were, by why.
    """
    from .post_processing import BRIDGE_SET_BY_HAND

    counts: Counter = Counter({"bridges": 0, "junctions": 0})
    thick = _ThickLumen(G)
    wanted: dict[tuple, dict[Any, float]] = {}
    for junction in sort_nodes(G.nodes if nodes is None else (n for n in nodes if n in G)):
        if G.degree(junction) < 3:
            continue
        incident = list(G.edges(junction, keys=True, data=True))
        if any(data.get(IS_ZERO_RESISTANCE) for *_edge, data in incident):
            counts["junctions_at_bridges"] += 1
            continue
        if thick.holds(G.nodes[junction].get("pos")):
            counts["junctions_in_thick_vessels"] += 1
            continue
        widths = [_width_um(data) for *_edge, data in incident]
        if G.nodes[junction].get("pos") is None or any(width is None for width in widths):
            counts["junctions_unmeasured"] += 1
            continue
        widest = max(widths)
        for (_j, other, key, data), width in zip(incident, widths):
            if data.get(BRIDGE_SET_BY_HAND) is False:
                counts["vessels_unbridged_by_hand"] += 1
            elif other != junction and width < NARROWER_THAN * widest:
                wanted.setdefault(edge_id(junction, other, key), {})[junction] = widest

    reserved: set[Any] = set()
    bridged_at: set[Any] = set()
    for (u, v, key), widest_at in wanted.items():
        data = G.edges[u, v, key]
        path = edge_sample_points(u, v, data, {n: G.nodes[n]["pos"] for n in (u, v)})
        along = cumulative_lengths(path)
        total = float(along[-1])
        at_u, at_v = widest_at.get(u, 0.0) / 2.0, widest_at.get(v, 0.0) / 2.0
        if total - at_u - at_v < MIN_VESSEL_LEFT_UM:
            counts["vessels_too_short"] += 1
            continue

        def end_node(s: float) -> Any:
            node = next_node_id(G, reserved)
            reserved.add(node)
            G.add_node(node, pos=tuple(float(c) for c in _point_at(path, along, s)), **{JUNCTION_BRIDGE_END: True})
            return node

        start, stop = (end_node(at_u) if at_u else u), (end_node(total - at_v) if at_v else v)
        G.remove_edge(u, v, key)
        if at_u:
            G.add_edge(u, start, **_bridge(data, _piece(path, along, 0.0, at_u), widest_at[u]))
        G.add_edge(start, stop, **_vessel(data, _piece(path, along, at_u, total - at_v)))
        if at_v:
            G.add_edge(stop, v, **_bridge(data, _piece(path, along, total - at_v, total), widest_at[v]))
        counts["bridges"] += bool(at_u) + bool(at_v)
        bridged_at.update(widest_at)
    counts["junctions"] = len(bridged_at)
    return dict(counts)


def _junction_and_end(G: nx.MultiGraph, u: Any, v: Any) -> tuple[Any, Any]:
    """A junction bridge's ``(junction, end)``: the end is the node marked
    :data:`JUNCTION_BRIDGE_END`, or with neither or both marked, the one
    fewer vessels meet."""
    marked = [bool(G.nodes[n].get(JUNCTION_BRIDGE_END)) for n in (u, v)]
    if marked[0] != marked[1]:
        return (v, u) if marked[0] else (u, v)
    return (u, v) if G.degree(u) >= G.degree(v) else (v, u)


def remove_junction_bridges(
    G: nx.MultiGraph, *, nodes: Iterable[Any] | None = None, keep: Iterable[Any] = ()
) -> list[Any]:
    """Put back, in place, each vessel :func:`insert_junction_bridges` cut at
    a junction of *nodes* (default every junction): its bridge joined onto
    it again through the node they met at. A junction counts as one of
    *nodes* when either end of one of its bridges is, and then every bridge
    there goes, so it can be judged afresh.

    A bridge stays where its end node is in *keep* (a boundary node), or
    where an edit has since joined another vessel there; one whose vessel an
    edit took goes with its end node. Returns the junctions bridges were
    taken from, in node order.
    """
    bridges = [
        (u, v, key, *_junction_and_end(G, u, v))
        for u, v, key, data in G.edges(keys=True, data=True)
        if data.get(JUNCTION_BRIDGE) and u != v
    ]
    if nodes is not None:
        chosen = set(nodes)
        at = {junction for u, v, _k, junction, _end in bridges if u in chosen or v in chosen}
        bridges = [bridge for bridge in bridges if bridge[3] in at]
    keep = set(keep)
    removed_from: set[Any] = set()
    for u, v, key, junction, end in bridges:
        if not G.has_edge(u, v, key) or end in keep:
            continue
        others = [
            (a, b, k, d) for a, b, k, d in G.edges(end, keys=True, data=True)
            if not (k == key and {a, b} == {u, v})
        ]
        if not others:
            G.remove_edge(u, v, key)
            G.remove_node(end)
        elif len(others) == 1 and not others[0][3].get(IS_ZERO_RESISTANCE) and others[0][1] != end:
            _end, far, far_key, vessel = others[0]
            positions = {n: G.nodes[n]["pos"] for n in (junction, end, far)}
            bridge_path = edge_sample_points(junction, end, G.edges[u, v, key], positions)
            vessel_path = edge_sample_points(end, far, vessel, positions)
            attrs = _vessel(vessel, np.vstack([bridge_path, vessel_path[1:]]))
            G.remove_node(end)
            G.add_edge(junction, far, **attrs)
        else:
            continue
        removed_from.add(junction)
    return sort_nodes(removed_from)


def rebridge_junctions(G: nx.MultiGraph, nodes: Iterable[Any], *, keep: Iterable[Any] = ()) -> dict[str, int]:
    """Judge the junctions at *nodes* afresh from the widths now on *G*: the
    junction bridges there put back (:func:`remove_junction_bridges`), then
    drawn again (:func:`insert_junction_bridges`)."""
    nodes = [n for n in nodes if n in G]
    junctions = remove_junction_bridges(G, nodes=nodes, keep=keep)
    return insert_junction_bridges(G, nodes=[*nodes, *junctions])
