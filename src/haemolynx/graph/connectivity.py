"""How a network is connected, as a flat table: one row per vessel.

Behind the "Export connectivity CSV" button of the post-processing tab, and
usable on any graph. Each vessel (edge) runs *from* one node *to* another,
oriented away from the inlets: the end nearer an inlet, by path length along
the vessels, is the "from" end. That needs no solved flow, so the table is the
same before and after Haemodynamics runs.

A vessel splitting into two (a diverging bifurcation) shows as one row whose
``Downstream branch IDs`` lists both daughters; two merging into one, as a row
whose ``Upstream branch IDs`` lists both. An inlet vessel has no "from" node
and an outlet vessel no "to" node, when that end is the network's open tip.

The IDs are the ones every other output uses, so a simulation result can be
mapped back: ``Branch ID`` is the vessel's index in ``G.edges(keys=True)`` --
the viewer's branchID and the order of the VTK export's vessel cells -- node
IDs are the graph's own (the VTK's ``node_id``, the ``.pkl``'s nodes), and
``Edge u`` / ``Edge v`` / ``Edge key`` are the vessel's key in the ``.pkl``
graph and the VTK's ``edge_u`` / ``edge_v`` / ``edge_key`` cell data.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

__all__ = [
    "CONNECTIVITY_COLUMNS",
    "connectivity_rows",
    "inlet_to_outlet_vessels",
    "remove_vessels_off_inlet_outlet_paths",
    "write_connectivity_csv",
]

#: The CSV's header, one entry per :func:`connectivity_rows` cell.
CONNECTIVITY_COLUMNS = (
    "Branch ID",
    "From node ID",
    "To node ID",
    "Branch order",
    "Length (um)",
    "Diameter (um)",
    "Notes",
    "Upstream branch IDs",
    "Downstream branch IDs",
    "Edge u",
    "Edge v",
    "Edge key",
)


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _edge_length(data: dict) -> float:
    try:
        length = float(data.get("length"))
    except (TypeError, ValueError):
        return 1.0
    return length if np.isfinite(length) and length >= 0 else 1.0


def inlet_to_outlet_vessels(
    G: nx.MultiGraph,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
) -> set[tuple[Any, Any, Any]]:
    """The vessels ``(u, v, key)`` some inlet-to-outlet path runs through.

    A vessel is kept when a path from an inlet to an outlet, visiting no node
    twice, uses it: one end leads back to an inlet and the other on to an
    outlet, by different vessels. Dead-end branches, loops hanging off a
    single node, self-loops and pieces with no inlet or no outlet carry no
    flow, and are left out. Exact and linear-time: with every inlet joined to
    a virtual source, every outlet to a virtual sink and the source to the
    sink, a vessel is kept exactly when it shares a biconnected component
    with that source-sink edge. Empty when there is no inlet or no outlet.
    """
    inlets = [n for n in dict.fromkeys(inlet_nodes) if n in G]
    outlets = [n for n in dict.fromkeys(outlet_nodes) if n in G]
    if not inlets or not outlets:
        return set()
    source, sink = object(), object()
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from((u, v) for u, v in G.edges() if u != v)
    H.add_edges_from((source, n) for n in inlets)
    H.add_edges_from((sink, n) for n in outlets)
    H.add_edge(source, sink)
    through: set[frozenset] = set()
    for component in nx.biconnected_component_edges(H):
        pairs = {frozenset(edge) for edge in component}
        if frozenset((source, sink)) in pairs:
            through = pairs
            break
    return {
        (u, v, k)
        for u, v, k in G.edges(keys=True)
        if u != v and frozenset((u, v)) in through
    }


def _summed_length(data: dict) -> float:
    try:
        length = float(data.get("length"))
    except (TypeError, ValueError):
        return 0.0
    return length if np.isfinite(length) else 0.0


def remove_vessels_off_inlet_outlet_paths(
    G: nx.MultiGraph,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
) -> tuple[nx.MultiGraph, dict[str, float]]:
    """A copy of *G* holding only the vessels :func:`inlet_to_outlet_vessels`
    keeps, and the counts of what went.

    That removes every piece with no inlet or no outlet, as
    :func:`haemolynx.graph.remove_components_without_connected_io` does, and
    also what hangs off a piece that has both: dead-end branches and trees,
    loops attached to the rest at a single node, and self-loops -- no
    pressure difference drives a flow along any of them. The inlets and
    outlets of a kept piece stay, as does every node a kept vessel ends on.

    The counts separate the two: ``removed_components`` with
    ``removed_component_vessels`` / ``_nodes`` / ``_length_um``, and
    ``removed_dead_end_vessels`` / ``_nodes`` / ``_length_um``; then
    ``remaining_nodes`` and ``remaining_vessels``.
    """
    kept_edges = inlet_to_outlet_vessels(G, inlet_nodes, outlet_nodes)
    boundary = {n for n in (*inlet_nodes, *outlet_nodes) if n in G}
    inlets = {n for n in inlet_nodes if n in G}
    outlets = {n for n in outlet_nodes if n in G}
    in_kept_piece: set[Any] = set()
    removed_components = 0
    for component in nx.connected_components(G):
        if component & inlets and component & outlets:
            in_kept_piece |= component
        else:
            removed_components += 1
    kept_nodes = {n for u, v, _k in kept_edges for n in (u, v)}
    kept_nodes |= boundary & in_kept_piece

    stats: dict[str, float] = {
        "removed_components": removed_components,
        "removed_component_vessels": 0,
        "removed_component_nodes": 0,
        "removed_component_length_um": 0.0,
        "removed_dead_end_vessels": 0,
        "removed_dead_end_nodes": 0,
        "removed_dead_end_length_um": 0.0,
    }
    removed_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if (u, v, k) in kept_edges:
            continue
        removed_edges.append((u, v, k))
        kind = "dead_end" if u in in_kept_piece else "component"
        stats[f"removed_{kind}_vessels"] += 1
        stats[f"removed_{kind}_length_um"] += _summed_length(data)
    removed_nodes = [n for n in G if n not in kept_nodes]
    for n in removed_nodes:
        kind = "dead_end" if n in in_kept_piece else "component"
        stats[f"removed_{kind}_nodes"] += 1

    pruned = G.copy()
    pruned.remove_edges_from(removed_edges)
    pruned.remove_nodes_from(removed_nodes)
    stats["remaining_nodes"] = pruned.number_of_nodes()
    stats["remaining_vessels"] = pruned.number_of_edges()
    return pruned, stats


def connectivity_rows(
    G: nx.MultiGraph,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
    *,
    arteriole_boundary_nodes: Iterable[Any] = (),
    venule_boundary_nodes: Iterable[Any] = (),
    only_inlet_to_outlet: bool = False,
) -> list[dict[str, str]]:
    """One row per vessel of *G*, keyed by :data:`CONNECTIVITY_COLUMNS`.

    Rows run from the inlets outwards (by the distance of each vessel's
    "from" end from the nearest inlet, then branchID). A vessel in a piece of
    the network with no inlet keeps the graph's own end order and says so in
    its notes, as do dead ends (a tip that is no inlet or outlet), self-loops
    and the arteriole/venule boundary roles of its ends.

    With *only_inlet_to_outlet*, only the vessels
    :func:`inlet_to_outlet_vessels` keeps are listed, and upstream/downstream
    vessels and open tips are read among those alone. Branch and node IDs are
    *G*'s either way, so they still match the VTK and ``.pkl`` exports.
    """
    inlets = {n for n in inlet_nodes if n in G}
    outlets = {n for n in outlet_nodes if n in G}
    arterioles = set(arteriole_boundary_nodes)
    venules = set(venule_boundary_nodes)

    distance: dict[Any, float] = {}
    if inlets:
        distance = nx.multi_source_dijkstra_path_length(
            G, inlets, weight=lambda _u, _v, data: min(_edge_length(d) for d in data.values())
        )

    edges = list(G.edges(keys=True, data=True))
    kept = (
        inlet_to_outlet_vessels(G, list(inlets), list(outlets))
        if only_inlet_to_outlet
        else None
    )
    degree: dict[Any, int] = {}
    oriented: list[tuple[int, Any, Any, Any, dict, bool]] = []
    for branch_id, (u, v, k, data) in enumerate(edges):
        if kept is not None and (u, v, k) not in kept:
            continue
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        reached = u in distance and v in distance
        start, end = u, v
        if reached and distance[v] < distance[u]:
            start, end = v, u
        oriented.append((branch_id, start, end, k, data, reached))

    into: dict[Any, list[int]] = {}
    out_of: dict[Any, list[int]] = {}
    for branch_id, start, end, _k, _data, _reached in oriented:
        out_of.setdefault(start, []).append(branch_id)
        into.setdefault(end, []).append(branch_id)

    def role_notes(node: Any) -> list[str]:
        notes = []
        if node in arterioles:
            notes.append("arteriole boundary")
        if node in venules:
            notes.append("venule boundary")
        return notes

    keyed = []
    for branch_id, start, end, k, data, reached in oriented:
        notes: list[str] = []
        from_cell, to_cell = str(start), str(end)
        if start == end:
            notes.append("Self-loop")
        else:
            if start in inlets:
                notes.append("Inlet")
                if degree[start] == 1:
                    from_cell = ""
            if end in outlets:
                notes.append("Outlet")
                if degree[end] == 1:
                    to_cell = ""
            elif end not in inlets and degree[end] == 1:
                notes.append("Dead end")
        for node in dict.fromkeys((start, end)):
            notes.extend(role_notes(node))
        if not reached:
            notes.append("Not connected to an inlet")
        upstream = [b for b in into.get(start, ()) if b != branch_id]
        downstream = [b for b in out_of.get(end, ()) if b != branch_id]
        order = data.get("branch_order")
        keyed.append((
            (distance.get(start, np.inf), branch_id),
            {
                "Branch ID": str(branch_id),
                "From node ID": from_cell,
                "To node ID": to_cell,
                "Branch order": "" if order is None else str(order),
                "Length (um)": _number(data.get("length")),
                "Diameter (um)": _number(data.get("diameter_um")),
                "Notes": "; ".join(notes),
                "Upstream branch IDs": ", ".join(map(str, upstream)),
                "Downstream branch IDs": ", ".join(map(str, downstream)),
                "Edge u": str(edges[branch_id][0]),
                "Edge v": str(edges[branch_id][1]),
                "Edge key": str(k),
            },
        ))
    keyed.sort(key=lambda item: item[0])
    return [row for _key, row in keyed]


def write_connectivity_csv(path: Path | str, rows: Sequence[dict[str, str]]) -> Path:
    """Write :func:`connectivity_rows` to *path*. Returns the path written."""
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONNECTIVITY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return target
