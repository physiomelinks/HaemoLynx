"""The image's network inside a synthetic one (``PLAN_synthetic_surround.md``).

A synthetic network drawn round an image run's own graph, to the run's own
template, joined at the vessels the image faces cut. This module holds what is
built of it so far: the image's open ends, read as the surround will continue
them (:func:`read_open_ends`).

Each open end is an :class:`ImageOpenEnd`: where it leaves the image, which
way its vessel heads out, how wide it is, what class of vessel it is (its
branch order, or a synthetic network's own class), which way blood ran
through it in the run's solve (``haemodynamics.open_ends.open_end_directions``)
and whether the run designated it an inlet or an outlet. Its
:attr:`~ImageOpenEnd.continuation` is what the surround draws beyond it: a
capillary joins the synthetic bed; every arteriole and venule runs on as a
trunk to the outer box's face, a new inlet or outlet there (Harvey,
2026-10-27, D5: the solved direction was right at no more ends than a coin,
59 of 124 on E14.5's harness runs, and a trunk at 97).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.dead_ends import (
    chain_path,
    chain_to_junction,
    graph_image_extent_um,
    open_end_nodes,
    tip_heading,
)
from haemolynx.haemodynamics.open_ends import open_end_directions
from haemolynx.statistics.bifurcation import branch_order_category

from .generate import SYNTHETIC_CLASS
from .template import FACES

#: The vessel sides an open end can be on.
ARTERIAL = "arterial"
VENOUS = "venous"
CAPILLARY = "capillary"
UNKNOWN = "unknown"
_SIDE_OF_CATEGORY = {
    "large_arteriole": ARTERIAL,
    "arteriole": ARTERIAL,
    "capillary": CAPILLARY,
    "venule": VENOUS,
    "large_venule": VENOUS,
}
#: What the surround draws beyond an open end (the plan's table).
BED = "bed"
TRUNK = "trunk"
CONTINUATIONS: tuple[str, ...] = (BED, TRUNK)


def vessel_side(data: dict) -> str:
    """The side a vessel is on (:data:`ARTERIAL`, :data:`VENOUS`,
    :data:`CAPILLARY` or :data:`UNKNOWN`) by its class (:func:`_category`)."""
    return _SIDE_OF_CATEGORY.get(_category(data)[1] or "", UNKNOWN)


@dataclass(frozen=True)
class ImageOpenEnd:
    """One vessel the image's faces cut. See the module."""

    node: Any
    position_um: tuple[float, float, float]
    #: The face it leaves by (``template.FACES``): the nearest.
    face: str
    #: The unit vector out of the image along its vessel, None for a chain
    #: with no length.
    heading: Optional[tuple[float, float, float]]
    diameter_um: Optional[float]
    #: Its vessel's branch-order label, and that label's class
    #: (``graph.branch_order.BRANCH_ORDER_CATEGORY_SEQUENCE``), or a
    #: synthetic network's own class; None for neither.
    order: Optional[str]
    category: Optional[str]
    #: ``open_ends.FEEDS`` (blood entered the image through it in the run's
    #: solve), ``open_ends.DRAINS``, or None (not solved, or under the
    #: carrying floor).
    direction: Optional[str]
    #: ``"inlet"`` or ``"outlet"`` when the run designated it one.
    designated: Optional[str] = None

    @property
    def side(self) -> str:
        """:data:`ARTERIAL`, :data:`VENOUS`, :data:`CAPILLARY` or :data:`UNKNOWN`."""
        return _SIDE_OF_CATEGORY.get(self.category or "", UNKNOWN)

    @property
    def continuation(self) -> str:
        """What the surround draws beyond it (:data:`CONTINUATIONS`): a
        designated inlet or outlet, and every arteriole and venule, whichever
        way its blood ran, runs on as a trunk (D5); a capillary, or a vessel
        of no known class, joins the bed."""
        if self.designated is not None or self.side in (ARTERIAL, VENOUS):
            return TRUNK
        return BED


def _face(position: np.ndarray, extent: np.ndarray) -> str:
    both = np.concatenate([position, extent - position])
    return FACES[int(np.argmin(both))]


def _category(data: dict) -> tuple[Optional[str], Optional[str]]:
    """``(order, category)`` of a vessel: its branch order and that order's
    class, else a synthetic network's own class."""
    order = data.get("branch_order")
    if order is not None:
        category = branch_order_category(str(order))
        return str(order), (category if category in _SIDE_OF_CATEGORY else None)
    category = data.get(SYNTHETIC_CLASS)
    return None, (category if category in _SIDE_OF_CATEGORY else None)


def read_open_ends(
    G: nx.MultiGraph,
    *,
    inlets: Sequence[Any] = (),
    outlets: Sequence[Any] = (),
    extent_um: Optional[Sequence[float]] = None,
) -> list[ImageOpenEnd]:
    """Every dead end of *G* that leaves the image
    (``graph.dead_ends.open_end_nodes``), and each designated inlet and
    outlet that is a dead end, as an :class:`ImageOpenEnd`, in node order.
    *extent_um* defaults to the graph's own (``image_extent_um``)."""
    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)
    if extent is None:
        raise ValueError("the graph records no image_extent_um: pass extent_um")
    roles = {**{n: "outlet" for n in outlets}, **{n: "inlet" for n in inlets}}
    ends = set(open_end_nodes(G, extent_um=extent))
    ends |= {n for n in roles if n in G and G.degree(n) == 1}
    directions = open_end_directions(G)
    out: list[ImageOpenEnd] = []
    for node in sorted(ends, key=lambda n: (str(type(n)), n)):
        _nodes, edges = chain_to_junction(G, node)
        heading = tip_heading(chain_path(G, edges)) if edges else None
        a, b, key = edges[0] if edges else next(iter(G.edges(node, keys=True)))
        data = G.edges[a, b, key]
        order, category = _category(data)
        diameter = data.get("diameter_um")
        position = np.asarray(G.nodes[node]["pos"], dtype=float)[:3]
        out.append(
            ImageOpenEnd(
                node=node,
                position_um=tuple(float(v) for v in position),
                face=_face(position, extent),
                heading=None if heading is None else tuple(float(v) for v in heading),
                diameter_um=None if diameter is None else float(diameter),
                order=order,
                category=category,
                direction=directions.get(node),
                designated=roles.get(node),
            )
        )
    return out


__all__ = [
    "ARTERIAL",
    "BED",
    "CAPILLARY",
    "CONTINUATIONS",
    "ImageOpenEnd",
    "TRUNK",
    "UNKNOWN",
    "VENOUS",
    "read_open_ends",
    "vessel_side",
]
