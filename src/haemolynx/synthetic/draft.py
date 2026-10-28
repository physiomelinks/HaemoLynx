"""The network while it is being drawn: nodes, their roles, and edges with
the lengths and diameters they were given but no polylines yet.

:mod:`.trees` and :mod:`.bed` add to one :class:`NetworkDraft`; :mod:`.generate`
turns it into the ``nx.MultiGraph`` the pipeline takes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

#: What an edge was drawn as: the vessel class the generator meant it to be.
TREE = "tree"
BED = "bed"
#: A vessel continuing an image's own out to the box (``generate.Join``):
#: drawn at that vessel's width, never rescaled to the template's totals.
TRUNK = "trunk"


@dataclass
class EdgeDraft:
    u: int
    v: int
    #: ``large_arteriole`` / ``arteriole`` / ``capillary`` / ``venule`` / ``large_venule``.
    category: str
    #: The order label the generator meant (``Art2``...), None for a capillary.
    order: Optional[str]
    #: The length the polyline will be bent to; None until lengths are drawn.
    length_um: Optional[float]
    diameter_um: Optional[float]
    #: Interior points the polyline must pass through (a merged degree-2 vertex).
    via: list = field(default_factory=list)
    part: str = TREE


@dataclass
class NetworkDraft:
    positions: list = field(default_factory=list)
    roles: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    #: Nodes taken out again (merged degree-2 vertices, isolated ones): left
    #: out of the graph, their ids never reused.
    removed: set = field(default_factory=set)
    #: Node attributes the graph carries over as they are (a join's key).
    marks: dict = field(default_factory=dict)

    def add_node(self, position: Any, *roles: str) -> int:
        node = len(self.positions)
        self.positions.append(np.asarray(position, dtype=float).reshape(3).copy())
        if roles:
            self.roles[node] = tuple(roles)
        return node

    def add_role(self, node: int, role: str) -> None:
        current = self.roles.get(node, ())
        if role not in current:
            self.roles[node] = current + (role,)

    def add_edge(self, u: int, v: int, **fields: Any) -> EdgeDraft:
        edge = EdgeDraft(u=u, v=v, **fields)
        self.edges.append(edge)
        return edge

    def position(self, node: int) -> np.ndarray:
        return self.positions[node]

    def degree(self) -> np.ndarray:
        counts = np.zeros(len(self.positions), dtype=int)
        for edge in self.edges:
            counts[edge.u] += 1
            counts[edge.v] += 1
        return counts

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


__all__ = ["BED", "TREE", "TRUNK", "EdgeDraft", "NetworkDraft"]
