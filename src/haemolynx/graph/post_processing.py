"""Post-processing a finished network: junctions where four or more vessels meet.

Pure graph logic behind the panel's "10. Post processing" tab -- no Qt, no
napari -- so every rule here is testable on a hand-built graph. A vessel is
an edge and a junction a node, as everywhere else; a vessel's *branchID* is
its position in ``G.edges(keys=True)``, the same index the viewer's hover
shows (see :func:`edge_keys`).

Three things a user can do at a junction of degree four or more:

* :func:`delete_vessels` removes one or more of its vessels, tidying each
  touched node the way :func:`haemolynx.graph.edit.delete_edge_and_collapse`
  does (degree 0 removed, a plain degree-2 pass-through merged into one
  vessel) -- but never removing or merging a protected (boundary) node, which
  a later solve would otherwise fail on.
* :func:`split_junction` peels vessels off into new nodes joined to the
  original by a short connector vessel (15 um by default), until the
  junction is a bifurcation.
* leave it as it is, which needs no function.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np

from ._helpers import calculate_path_length, next_node_id
from .degree2 import create_trivial_merged_edge

__all__ = [
    "DEFAULT_SPLIT_CONNECTOR_LENGTH_UM",
    "JunctionVessel",
    "delete_vessels",
    "edge_keys",
    "high_degree_junctions",
    "junction_vessels",
    "split_junction",
]

EdgeKey = tuple[Any, Any, Any]

#: Length of the connector vessel :func:`split_junction` inserts.
DEFAULT_SPLIT_CONNECTOR_LENGTH_UM = 15.0

#: How far along a vessel its direction out of a junction is measured.
_DIRECTION_SAMPLE_UM = 10.0


@dataclass(frozen=True)
class JunctionVessel:
    """One vessel meeting a junction, as the post-processing table lists it."""

    branch_id: int
    u: Any
    v: Any
    key: Any
    #: The vessel's other end (the junction itself for a self-loop).
    other_node: Any
    length_um: float | None
    diameter_um: float | None
    branch_order: str | None

    @property
    def edge(self) -> EdgeKey:
        return (self.u, self.v, self.key)


def edge_keys(G: nx.MultiGraph) -> list[EdgeKey]:
    """Every edge's ``(u, v, key)``, indexed by branchID."""
    return [(u, v, k) for u, v, k in G.edges(keys=True)]


def high_degree_junctions(G: nx.MultiGraph, *, min_degree: int = 4) -> list[Any]:
    """Nodes where at least *min_degree* vessels meet, busiest first.

    Ties keep the graph's own node order, so the list reads the same on every
    scan of an unchanged graph.
    """
    found = [(node, degree) for node, degree in G.degree() if degree >= min_degree]
    found.sort(key=lambda item: -item[1])
    return [node for node, _ in found]


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def junction_vessels(G: nx.MultiGraph, node: Any) -> list[JunctionVessel]:
    """The vessels meeting at *node*, in branchID order."""
    if node not in G:
        raise ValueError(f"Node {node!r} is not in the network")
    rows = []
    for branch_id, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        if node not in (u, v):
            continue
        order = data.get("branch_order")
        rows.append(
            JunctionVessel(
                branch_id=branch_id,
                u=u,
                v=v,
                key=k,
                other_node=v if u == node else u,
                length_um=_optional_float(data.get("length")),
                diameter_um=_optional_float(data.get("diameter_um")),
                branch_order=None if order is None else str(order),
            )
        )
    return rows


def delete_vessels(
    G: nx.MultiGraph,
    edges: Iterable[EdgeKey],
    *,
    protected: Iterable[Any] = (),
) -> set[Any]:
    """Remove *edges*, then tidy every node they touched.

    A touched node left with no vessels is removed; one left as a plain
    pass-through (two vessels to two different neighbours) is merged into one
    continuous vessel, as :func:`haemolynx.graph.edit.delete_edge_and_collapse`
    does. *protected* nodes (inlets, outlets, arteriole/venule boundaries) are
    never merged, and a deletion that would leave one with no vessel at all is
    refused before anything changes: the solve needs every boundary node.

    Returns the node ids whose drawing changed.
    """
    selected = list(dict.fromkeys(tuple(edge) for edge in edges))
    for u, v, k in selected:
        if not G.has_edge(u, v, key=k):
            raise ValueError(f"No vessel ({u!r}, {v!r}, {k!r}) to delete")
    protected_set = set(protected)
    loss: Counter = Counter()
    for u, v, _k in selected:
        loss[u] += 1
        loss[v] += 1
    stranded = [
        node for node in loss if node in protected_set and G.degree(node) - loss[node] <= 0
    ]
    if stranded:
        names = ", ".join(str(node) for node in stranded)
        raise ValueError(
            f"Deleting these vessels would leave boundary node(s) {names} with no "
            "vessel; the solve needs every inlet, outlet and boundary node"
        )
    for u, v, k in selected:
        G.remove_edge(u, v, k)

    changed: set[Any] = set()
    for node in loss:
        if not G.has_node(node) or node in protected_set:
            continue
        degree = G.degree(node)
        if degree == 0:
            G.remove_node(node)
            changed.add(node)
            continue
        if degree != 2:
            continue
        incident = list(G.edges(node, keys=True, data=True))
        if len(incident) != 2:
            continue  # a self-loop counts twice towards degree
        (_, n1, _k1, d1), (_, n2, _k2, d2) = incident
        if n1 == n2:
            continue  # two vessels to one neighbour: a loop, not a pass-through
        merged = create_trivial_merged_edge(d1, d2, G.nodes[node].get("pos"))
        G.remove_node(node)
        G.add_edge(n1, n2, **merged)
        changed.update({node, n1, n2})
    return changed


def _node_position(G: nx.MultiGraph, node: Any) -> np.ndarray:
    pos = G.nodes[node].get("pos")
    if pos is None:
        raise ValueError(f"Node {node!r} has no position")
    return np.asarray(pos, dtype=float)[:3]


def _path_from(G: nx.MultiGraph, node: Any, other: Any, data: dict) -> np.ndarray:
    """The vessel's path as (n, 3) points, starting at the *node* end."""
    start = _node_position(G, node)
    voxels = data.get("voxels")
    if voxels is not None and len(voxels) >= 2:
        points = np.asarray([np.asarray(p, dtype=float)[:3] for p in voxels])
    else:
        points = np.vstack([start, _node_position(G, other)])
    if np.linalg.norm(points[-1] - start) < np.linalg.norm(points[0] - start):
        points = points[::-1]
    return points


def _direction(points: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Unit direction a vessel leaves *start* in, over its first few microns."""
    travelled = 0.0
    target = points[-1]
    for a, b in zip(points[:-1], points[1:]):
        travelled += float(np.linalg.norm(b - a))
        if travelled >= _DIRECTION_SAMPLE_UM:
            target = b
            break
    vector = target - start
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros(3)


def split_junction(
    G: nx.MultiGraph,
    node: Any,
    *,
    connector_length_um: float = DEFAULT_SPLIT_CONNECTOR_LENGTH_UM,
    reserved_ids: Iterable[Any] = (),
) -> list[Any]:
    """Split *node* into bifurcations joined by short connector vessels.

    While more than three vessels meet at *node*, the two leaving it in the
    most similar directions -- the pair most likely to be daughters of one
    parent -- move to a new node *connector_length_um* out along their mean
    direction, joined back to *node* by a straight connector vessel of that
    length. A degree-4 junction becomes two bifurcations; degree 5 takes two
    peels. Each moved vessel's path loses the part the connector now covers,
    and its length is re-measured from what is left. *node* keeps its id (so a
    boundary role on it survives); new ids avoid *reserved_ids*.

    Connector vessels carry ``junction_split_connector=True`` and no diameter
    or branch order: a regenerate from Diameters assigns both.

    Returns the new node ids.
    """
    length = float(connector_length_um)
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"Connector length must be positive, got {connector_length_um!r}")
    if node not in G:
        raise ValueError(f"Node {node!r} is not in the network")
    origin = _node_position(G, node)
    reserved = set(reserved_ids)
    connectors: set[tuple[Any, Any, Any]] = set()
    new_nodes: list[Any] = []

    while G.degree(node) > 3:
        candidates = [
            (other, k, data)
            for _, other, k, data in G.edges(node, keys=True, data=True)
            if other != node
            and (node, other, k) not in connectors
            and (other, node, k) not in connectors
        ]
        if len(candidates) < 2:
            break
        paths = [_path_from(G, node, other, data) for other, _k, data in candidates]
        directions = [_direction(path, origin) for path in paths]
        best: tuple[int, int] | None = None
        best_score = -np.inf
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                score = float(np.dot(directions[i], directions[j]))
                if score > best_score:
                    best, best_score = (i, j), score
        assert best is not None
        i, j = best
        axis = directions[i] + directions[j]
        if np.linalg.norm(axis) < 1e-9:
            axis = directions[i]
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        new_pos = origin + length * axis

        new_node = next_node_id(G, reserved | set(new_nodes))
        G.add_node(new_node, pos=new_pos)
        for index in (i, j):
            other, k, data = candidates[index]
            path = paths[index]
            rest = list(path[1:])
            while len(rest) > 1 and float(np.dot(rest[0] - origin, axis)) <= length:
                rest.pop(0)
            points = [tuple(float(c) for c in new_pos)] + [
                tuple(float(c) for c in p) for p in rest
            ]
            attrs = {key: value for key, value in data.items() if key not in ("voxels", "length")}
            attrs["voxels"] = points
            attrs["length"] = float(calculate_path_length(points))
            G.remove_edge(node, other, k)
            G.add_edge(new_node, other, **attrs)
        key = G.add_edge(
            node,
            new_node,
            voxels=[tuple(float(c) for c in origin), tuple(float(c) for c in new_pos)],
            length=length,
            junction_split_connector=True,
        )
        connectors.add((node, new_node, key))
        new_nodes.append(new_node)
    return new_nodes

