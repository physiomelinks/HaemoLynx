"""The image's network inside a synthetic one (``PLAN_synthetic_surround.md``).

A synthetic network drawn round an image run's own graph, to the run's own
template, joined at the vessels the image faces cut: the image's open ends,
read as the surround continues them (:func:`read_open_ends`), and the
network drawn round the image and joined to it (:func:`surround_network`).

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

The surround's box is the image's scaled (``synthetic_scale`` along
``synthetic_scale_axes``) with the image's box in its middle as a hole
(``synthetic.hole``). The image's graph is moved into the surround's frame
unchanged; the network drawn round it (``generate_network(joins=)``) joins each
open end where its vessel meets the hole's face, by a drawn piece of the same
vessel where its tip lies short of it (:data:`SURROUND_CONNECTOR`). An open end on
a face the box does not grow past (a z face, scaled along x and y) is left
open. Every edge says which network it is (:data:`NETWORK_PART`), and
:func:`validate_surrounded_graph` checks the join.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.crop import SOLVE_EDGE_ATTRIBUTES, SOLVE_NODE_ATTRIBUTES
from haemolynx.graph.dead_ends import (
    DEAD_END_KIND,
    DEAD_END_KIND_SOURCE,
    chain_path,
    chain_to_junction,
    graph_image_extent_um,
    open_end_nodes,
    tip_heading,
)
from haemolynx.haemodynamics.open_ends import open_end_directions
from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_SYNTHETIC
from haemolynx.statistics.bifurcation import branch_order_category

from .generate import (
    DEFAULT_SEED,
    JOIN_BED,
    JOIN_TRUNK,
    SURROUND_JOIN,
    SYNTHETIC_CLASS,
    SYNTHETIC_ROLES,
    GenerationOptions,
    Join,
    generate_network,
)
from .hole import GRAPH_HOLE, Hole, centred_hole
from .polylines import voxels_and_length
from .template import FACES, GRAPH_NETWORK_SOURCE, NetworkTemplate

#: What ``G.graph[GRAPH_NETWORK_SOURCE]`` holds on a surrounded network.
SURROUNDED = "synthetic_surround"
#: Graph keys: where the image's box lies in the surround's frame (its
#: ``lo`` corner), and what was joined (:func:`surround_network`).
IMAGE_OFFSET = "image_offset_um"
SURROUND_RECORD = "synthetic_surround"
#: Edge attribute: which network a vessel is, :data:`IMAGE_PART` or
#: :data:`SYNTHETIC_PART`.
NETWORK_PART = "network_part"
IMAGE_PART = "image"
SYNTHETIC_PART = "synthetic"
#: Edge attribute: a drawn piece of an image's vessel from its tip to the
#: hole's face, where the surround joins it.
SURROUND_CONNECTOR = "surround_connector"
#: Node attribute: an image node's own id.
IMAGE_NODE_ID = "image_node_id"
#: A tip this near its face point is joined there, with no connector (um).
JOIN_TOLERANCE_UM = 1e-6

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


_SIDE_CATEGORIES = {ARTERIAL: ("large_arteriole", "arteriole"), VENOUS: ("large_venule", "venule")}


def _join_for(end: ImageOpenEnd, offset: np.ndarray, hole: Hole) -> Join:
    """The join the surround makes at *end*: on its face of the hole, where
    its vessel meets it."""
    axis = "zyx".index(end.face[0])
    point = np.asarray(end.position_um, dtype=float) + offset
    point[axis] = hole.lo[axis] if end.face.endswith("0") else hole.hi[axis]
    if end.continuation != TRUNK:
        return Join(position_um=tuple(float(v) for v in point), kind=JOIN_BED, key=end.node)
    side = end.side if end.side in _SIDE_CATEGORIES else (ARTERIAL if end.designated == "inlet" else VENOUS)
    category = end.category if end.category in _SIDE_CATEGORIES[side] else _SIDE_CATEGORIES[side][1]
    return Join(
        position_um=tuple(float(v) for v in point),
        kind=JOIN_TRUNK,
        key=end.node,
        side=side,
        heading=end.heading,
        diameter_um=end.diameter_um,
        category=category,
        order=end.order,
    )


def _translated(voxels: Any, offset: np.ndarray) -> Any:
    if voxels is None:
        return None
    return (np.asarray(voxels, dtype=float).reshape(-1, 3) + offset).tolist()


def surround_network(
    image_graph: nx.MultiGraph,
    template: NetworkTemplate,
    *,
    inlets: Sequence[Any] = (),
    outlets: Sequence[Any] = (),
    roles: Optional[Mapping[str, Sequence[Any]]] = None,
    options: Optional[GenerationOptions] = None,
    seed: Optional[int] = DEFAULT_SEED,
    rng: Optional[np.random.Generator] = None,
    validate: bool = True,
) -> nx.MultiGraph:
    """*image_graph* inside a synthetic network drawn round it to
    *template* (the image run's own), one graph. See the module.

    The box is *template*'s scaled by ``options.scale`` (above 1) along
    ``options.scale_axes``. *inlets* and *outlets* are the image run's; their
    trunks' ends on the box's faces take their roles. *roles* (role ->
    nodes) are the image run's other boundary nodes, kept on its nodes.
    The image's solve (flows, pressures) is not carried over."""
    options = options or GenerationOptions()
    if not float(options.scale) > 1.0:
        raise ValueError(
            f"a surround needs a box larger than the image: synthetic_scale above 1, got {options.scale!r}"
        )
    if any(not isinstance(n, (int, np.integer)) for n in image_graph.nodes):
        raise ValueError("the image graph's nodes must be ints")
    own = graph_image_extent_um(image_graph)
    image_extent = np.asarray(template.extent_um if own is None else own, dtype=float)
    outer = np.asarray(template.scaled(options.scale, options.scale_axes).extent_um, dtype=float)
    hole = centred_hole(outer, image_extent)
    offset = hole.lo

    ends = read_open_ends(image_graph, inlets=inlets, outlets=outlets, extent_um=image_extent)
    joins: list[Join] = []
    left_open: list = []
    for end in ends:
        axis = "zyx".index(end.face[0])
        if outer[axis] <= image_extent[axis] * (1.0 + 1e-9):
            left_open.append(end.node)  # the box does not grow past this face (D3)
        else:
            joins.append(_join_for(end, offset, hole))
    shell = generate_network(
        template, seed=seed, rng=rng, options=replace(options, hole=hole), validate=validate, joins=joins
    )

    G = nx.MultiGraph()
    G.graph.update(shell.graph)
    G.graph[GRAPH_NETWORK_SOURCE] = SURROUNDED
    G.graph[IMAGE_OFFSET] = tuple(float(v) for v in offset)
    for node, data in image_graph.nodes(data=True):
        kept = {k: v for k, v in data.items() if k not in SOLVE_NODE_ATTRIBUTES}
        kept["pos"] = np.asarray(data["pos"], dtype=float)[:3] + offset
        kept[IMAGE_NODE_ID] = node
        G.add_node(node, **kept)
    for u, v, key, data in image_graph.edges(keys=True, data=True):
        kept = {k: val for k, val in data.items() if k not in SOLVE_EDGE_ATTRIBUTES}
        if data.get("voxels") is not None:
            kept["voxels"] = _translated(data["voxels"], offset)
        kept[NETWORK_PART] = IMAGE_PART
        G.add_edge(u, v, key=key, **kept)
    base = max((int(n) for n in image_graph.nodes), default=-1) + 1
    for node, data in shell.nodes(data=True):
        G.add_node(base + node, **data)
    for u, v, data in shell.edges(data=True):
        G.add_edge(base + u, base + v, **{**data, NETWORK_PART: SYNTHETIC_PART})

    by_key = {data[SURROUND_JOIN]: base + node for node, data in shell.nodes(data=True) if SURROUND_JOIN in data}
    connectors = 0
    for join in joins:
        tip, at_face = join.key, by_key[join.key]
        for node in (tip, at_face):
            G.nodes[node].pop(DEAD_END_KIND, None)
            G.nodes[node].pop(DEAD_END_KIND_SOURCE, None)
        start, end = G.nodes[tip]["pos"], G.nodes[at_face]["pos"]
        if float(np.linalg.norm(end - start)) <= JOIN_TOLERANCE_UM:
            for _a, other, data in list(G.edges(at_face, data=True)):
                G.add_edge(tip, other, **data)
            G.remove_node(at_face)
            continue
        (_t, _o, vessel), = G.edges(tip, data=True)
        voxels, length = voxels_and_length(np.stack([start, end]))
        attributes = {
            k: vessel[k] for k in (SYNTHETIC_CLASS, "branch_order", "diameter_basis") if vessel.get(k) is not None
        }
        G.add_edge(
            tip, at_face, voxels=voxels, length=length, diameter_um=float(vessel.get("diameter_um") or join.diameter_um or 5.0),
            diameter_source=DIAMETER_SOURCE_SYNTHETIC, **attributes, **{NETWORK_PART: SYNTHETIC_PART, SURROUND_CONNECTOR: True},
        )
        connectors += 1

    joined = {join.key for join in joins}
    for node in joined:
        # A joined inlet's or outlet's role is its trunk's end's now.
        kept = tuple(r for r in G.nodes[node].get(SYNTHETIC_ROLES, ()) if r not in ("inlet", "outlet"))
        if kept:
            G.nodes[node][SYNTHETIC_ROLES] = kept
        else:
            G.nodes[node].pop(SYNTHETIC_ROLES, None)
    image_roles = {**(roles or {}), "inlet": list(inlets), "outlet": list(outlets)}
    for role, nodes in image_roles.items():
        for node in nodes:
            if node not in G or (role in ("inlet", "outlet") and node in joined):
                continue  # a joined inlet's or outlet's role is its trunk's end's now
            current = tuple(G.nodes[node].get(SYNTHETIC_ROLES, ()))
            if role not in current:
                G.nodes[node][SYNTHETIC_ROLES] = current + (role,)
    G.graph[SURROUND_RECORD] = {
        "image_extent_um": [float(v) for v in image_extent],
        "joined": sorted(joined),
        "left_open": sorted(left_open),
        "trunks": sum(1 for j in joins if j.kind == JOIN_TRUNK),
        "bed_joins": sum(1 for j in joins if j.kind == JOIN_BED),
        "connectors": connectors,
    }
    if validate:
        validate_surrounded_graph(G, image_graph)
    return G


def _same(a: Any, b: Any) -> bool:
    """Equal values, arrays element for element, NaN equal to NaN."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f"))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return bool(a == b)


def validate_surrounded_graph(G: nx.MultiGraph, image_graph: nx.MultiGraph) -> None:
    """Raise ``ValueError`` unless *G* is *image_graph* moved into the
    surround's frame unchanged (every node and vessel, every attribute but
    the solve's and the dead-end kinds of the ends joined) inside a drawn
    network nothing of which lies in the hole, every open end joined or
    left open as the record says, and the whole connected."""
    problems: list[str] = []
    offset = np.asarray(G.graph.get(IMAGE_OFFSET), dtype=float)
    hole = Hole.from_record(G.graph.get(GRAPH_HOLE))
    record = G.graph.get(SURROUND_RECORD) or {}
    joined = set(record.get("joined", ()))
    changed = {"pos", DEAD_END_KIND, DEAD_END_KIND_SOURCE, SYNTHETIC_ROLES, *SOLVE_NODE_ATTRIBUTES}
    for node, data in image_graph.nodes(data=True):
        if node not in G:
            problems.append(f"image node {node} is missing")
            continue
        if not np.array_equal(G.nodes[node]["pos"], np.asarray(data["pos"], dtype=float)[:3] + offset):
            problems.append(f"image node {node} moved")
        if any(not _same(G.nodes[node].get(k), v) for k, v in data.items() if k not in changed):
            problems.append(f"image node {node}'s attributes changed")
        if node not in joined and DEAD_END_KIND in data and G.nodes[node].get(DEAD_END_KIND) != data[DEAD_END_KIND]:
            problems.append(f"image node {node}'s dead-end kind changed")
    for u, v, key, data in image_graph.edges(keys=True, data=True):
        if not G.has_edge(u, v, key):
            problems.append(f"image vessel {u}-{v} is missing")
            continue
        drawn = G.edges[u, v, key]
        if drawn.get(NETWORK_PART) != IMAGE_PART:
            problems.append(f"image vessel {u}-{v} is not marked {IMAGE_PART}")
        for name, value in data.items():
            if name in SOLVE_EDGE_ATTRIBUTES:
                continue
            if name == "voxels":
                if value is not None and not np.array_equal(
                    np.asarray(drawn["voxels"]), np.asarray(value, dtype=float).reshape(-1, 3) + offset
                ):
                    problems.append(f"image vessel {u}-{v}'s centreline moved")
            elif not _same(drawn.get(name), value):
                problems.append(f"image vessel {u}-{v}'s {name} changed")
    if hole is None:
        problems.append("no hole recorded")
    else:
        for node, data in G.nodes(data=True):
            if IMAGE_NODE_ID not in data and bool(hole.contains(data["pos"])):
                problems.append(f"drawn node {node} lies in the hole")
        for u, v, data in G.edges(data=True):
            if data.get(NETWORK_PART) == SYNTHETIC_PART and not data.get(SURROUND_CONNECTOR):
                if data.get("voxels") and bool(hole.contains(np.asarray(data["voxels"], float)).any()):
                    problems.append(f"drawn vessel {u}-{v} enters the hole")
    for node in joined:
        if node in G and G.degree(node) < 2:
            problems.append(f"open end {node} was not joined")
    for node in record.get("left_open", ()):
        if node in G and G.degree(node) != 1:
            problems.append(f"open end {node}, left open, has degree {G.degree(node)}")
    if G.number_of_nodes() and not nx.is_connected(G):
        pieces = nx.number_connected_components(G)
        image_pieces = nx.number_connected_components(image_graph)
        if pieces > image_pieces:
            problems.append(f"the surround leaves {pieces} pieces (the image had {image_pieces})")
    if problems:
        raise ValueError("surrounded network: " + "; ".join(problems[:12]) + (" ..." if len(problems) > 12 else ""))


__all__ = [
    "ARTERIAL",
    "BED",
    "CAPILLARY",
    "CONTINUATIONS",
    "IMAGE_NODE_ID",
    "IMAGE_OFFSET",
    "IMAGE_PART",
    "ImageOpenEnd",
    "NETWORK_PART",
    "SURROUNDED",
    "SURROUND_CONNECTOR",
    "SURROUND_RECORD",
    "SYNTHETIC_PART",
    "TRUNK",
    "UNKNOWN",
    "VENOUS",
    "read_open_ends",
    "surround_network",
    "validate_surrounded_graph",
    "vessel_side",
]
