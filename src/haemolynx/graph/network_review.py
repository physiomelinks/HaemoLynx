"""What the review lists of the "7. Post processing" tab find in a network.

Pure graph logic behind the tab's Review dropdown -- no Qt, no napari -- one
finder per list, each returning :class:`ReviewItem` s in the order the list
shows them. The tab's two older lists keep their own code: 4+ junctions
(:func:`haemolynx.graph.post_processing.high_degree_junctions`) and short
loops (:func:`haemolynx.graph.post_processing.short_loops`).

The thresholds here only decide what is *listed*. Nothing changes until a
person acts on an item, and every decision is written down with the item's
measures, so that a rule can later be fitted to the marks rather than to an
assumption (CLAUDE.md, "Marked example grids").
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from ._helpers import calculate_path_length, edge_sample_points
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

__all__ = [
    "BOUNDARY_NODES",
    "CLOSE_JUNCTIONS",
    "DEAD_ENDS",
    "DIAMETER_JUMPS",
    "BROKEN_VESSELS",
    "FACING_ENDS",
    "JOIN_ENDS",
    "JOIN_SIDE",
    "FLOW_AGAINST_ORDER",
    "FLOW_OUTLIERS",
    "HAIRPINS",
    "REVIEW_KINDS",
    "TERMINAL_NODES",
    "THICK_WALL_ENDS",
    "ReviewItem",
    "TWO_IN_ONE_LUMEN",
    "UNCOVERED_MASK",
    "UNMEASURED_DIAMETERS",
    "UNSOLVED_PIECES",
    "find_boundary_issues",
    "find_close_junctions",
    "find_dead_ends",
    "find_dead_ends_at_thick_walls",
    "find_diameter_jumps",
    "find_broken_vessels",
    "find_facing_ends",
    "find_flow_against_branch_order",
    "find_flow_outliers",
    "find_hairpins",
    "find_terminal_nodes",
    "find_two_in_one_lumen",
    "find_uncovered_mask",
    "find_unmeasured_diameters",
    "find_unsolved_pieces",
    "flow_reference",
    "total_inflow",
    "within",
    "z_range_around",
]

EdgeKey = tuple[Any, Any, Any]

CLOSE_JUNCTIONS = "close_junctions"
HAIRPINS = "hairpins"
DEAD_ENDS = "dead_ends"
#: The "Dead ends at thick vessels" list (PLAN_bridging.md 0.2, 1.3): every
#: dead end the definition calls ``thick_wall``.
THICK_WALL_ENDS = "thick_wall_ends"
TWO_IN_ONE_LUMEN = "two_in_one_lumen"
FACING_ENDS = "facing_ends"
#: The "Broken vessels" review list (PLAN_dead_ends.md Phase 6): the facing
#: dead ends the join left open, and the dead ends the definition calls
#: ``broken`` with another vessel's side straight ahead. Its key stays
#: :data:`FACING_ENDS`, what saved decisions and review CSVs name it by.
BROKEN_VESSELS = FACING_ENDS
#: How a "Broken vessels" item would be joined (its ``join`` measure): two
#: dead ends to each other, or a dead end to the side of the vessel ahead.
JOIN_ENDS = "ends"
JOIN_SIDE = "side"
UNCOVERED_MASK = "uncovered_mask"
UNSOLVED_PIECES = "unsolved_pieces"
FLOW_AGAINST_ORDER = "flow_against_order"
FLOW_OUTLIERS = "flow_outliers"
BOUNDARY_NODES = "boundary_nodes"
TERMINAL_NODES = "terminal_nodes"
UNMEASURED_DIAMETERS = "unmeasured_diameters"
DIAMETER_JUMPS = "diameter_jumps"
#: Every kind a finder here returns, in the order the dropdown lists them.
REVIEW_KINDS = (
    CLOSE_JUNCTIONS,
    HAIRPINS,
    DEAD_ENDS,
    THICK_WALL_ENDS,
    TWO_IN_ONE_LUMEN,
    FACING_ENDS,
    UNCOVERED_MASK,
    UNSOLVED_PIECES,
    FLOW_AGAINST_ORDER,
    FLOW_OUTLIERS,
    BOUNDARY_NODES,
    TERMINAL_NODES,
    UNMEASURED_DIAMETERS,
    DIAMETER_JUMPS,
)

#: Hairpins: a vessel at least this many times longer than the straight line
#: between its ends, and at least :data:`HAIRPIN_MIN_LENGTH_UM` long.
HAIRPIN_MIN_RATIO = 2.0
HAIRPIN_MIN_LENGTH_UM = 10.0

#: Facing ends: the review looks this many times further than the join's own
#: gap, and at pairs turning up to :data:`FACING_REVIEW_MAX_TURN_DEG` -- past
#: a right angle the ends point away from each other.
FACING_REVIEW_GAP_FACTOR = 2.0
FACING_REVIEW_MAX_TURN_DEG = 90.0

#: Flow outliers: a mean velocity more than this many robust standard
#: deviations (1.4826 MAD, in log10) from its vessel class's median, over a
#: class of at least :data:`FLOW_OUTLIER_MIN_CLASS` solved vessels -- and at
#: least :data:`FLOW_OUTLIER_MIN_FOLD` times faster or slower than it, or a
#: class whose vessels all flow alike would list a vessel 15% off.
FLOW_OUTLIER_ROBUST_Z = 3.5
FLOW_OUTLIER_MIN_CLASS = 8
FLOW_OUTLIER_MIN_FOLD = 3.0

#: Boundary nodes carrying less than this share of the network's inflow.
BOUNDARY_MIN_FLOW_SHARE = 0.001

#: Diameter jumps: one measured vessel this many times wider than the next
#: along a chain, or a daughter this many times wider than its parent.
DIAMETER_CHAIN_RATIO = 2.0
DIAMETER_DAUGHTER_RATIO = 1.5


@dataclass(frozen=True)
class ReviewItem:
    """One thing a review list asks a person to look at."""

    #: One of :data:`REVIEW_KINDS`.
    kind: str
    #: The nodes it is about (a tip, a junction, the two ends of a gap).
    nodes: tuple[Any, ...]
    #: Its vessels ``(u, v, key)``: what the tab's table lists and colours.
    edges: tuple[EdgeKey, ...]
    #: Where it is, physical microns ``(z, y, x)``: what decisions are matched by.
    centre_um: tuple[float, float, float]
    #: One line saying why it is listed.
    why: str
    #: Its measures, for the table and the decisions CSV.
    measures: Mapping[str, Any] = field(default_factory=dict, compare=False)
    #: Points to fit the view to, physical microns.
    points_um: np.ndarray = field(default_factory=lambda: np.empty((0, 3)), compare=False, repr=False)


# --- shared helpers ------------------------------------------------------------


def _node_pos(G: nx.MultiGraph) -> dict[Any, np.ndarray]:
    return {n: np.asarray(d["pos"], dtype=float)[:3] for n, d in G.nodes(data=True) if d.get("pos") is not None}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _path(G: nx.MultiGraph, edge: EdgeKey, pos: Mapping[Any, np.ndarray]) -> np.ndarray:
    u, v, key = edge
    if u not in pos or v not in pos:
        return np.empty((0, 3))
    return np.asarray(edge_sample_points(u, v, G.edges[u, v, key], pos), dtype=float).reshape(-1, 3)


def _length(G: nx.MultiGraph, edge: EdgeKey, pos: Mapping[Any, np.ndarray]) -> float:
    length = _number(G.edges[edge].get("length"))
    if length is not None:
        return length
    return calculate_path_length(_path(G, edge, pos).tolist())


def _points(G: nx.MultiGraph, edges: Iterable[EdgeKey], pos, extra=()) -> np.ndarray:
    parts = [_path(G, edge, pos) for edge in edges]
    parts += [np.asarray(p, dtype=float).reshape(-1, 3) for p in extra]
    parts = [p for p in parts if len(p)]
    return np.vstack(parts) if parts else np.empty((0, 3))


def _centre(points: np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    centre = points.mean(axis=0) if len(points) else np.asarray(fallback, dtype=float)
    return tuple(float(c) for c in centre)  # type: ignore[return-value]


def _namer(G: nx.MultiGraph):
    """``name(u, v, key)``: the edge as ``G.edges(keys=True)`` lists it --
    from whichever end comes first in the graph's node order -- so an edge
    found from either end is one item, under the name its branchID has."""
    order = {node: i for i, node in enumerate(G.nodes)}

    def name(u: Any, v: Any, key: Any) -> EdgeKey:
        return (u, v, key) if order.get(u, -1) <= order.get(v, -1) else (v, u, key)

    return name


def _chain_to_junction(G: nx.MultiGraph, tip: Any) -> tuple[list[EdgeKey], Any]:
    """The vessels from a dead end back to the first node that is not a
    plain pass-through, tip first, each named from the tip's side, and that
    node: the one chain walk every dead-end question shares
    (:func:`haemolynx.graph.dead_ends.chain_to_junction`)."""
    from .dead_ends import chain_to_junction

    nodes, edges = chain_to_junction(G, tip)
    return list(edges), nodes[-1]


def _diameter(data: Mapping[str, Any]) -> float | None:
    diameter = _number(data.get("diameter_um"))
    return diameter if diameter is not None and diameter > 0 else None


def _is_solved(data: Mapping[str, Any]) -> bool:
    from .prune import FLOW_SOLVED

    return data.get(FLOW_SOLVED) is not False and _number(data.get("flow_abs")) is not None


def total_inflow(G: nx.MultiGraph, inlets: Iterable[Any] = ()) -> float | None:
    """The flow into the network through *inlets* (m^3/s), read from the solve:
    at each inlet, what leaves it along its vessels towards lower pressure.
    None without inlets, node pressures or flows."""
    total = 0.0
    found = False
    for inlet in dict.fromkeys(inlets):
        if inlet not in G:
            continue
        p_in = _number(G.nodes[inlet].get("pressure"))
        if p_in is None:
            continue
        for _a, other, data in G.edges(inlet, data=True):
            flow = _number(data.get("flow_abs"))
            p_other = _number(G.nodes[other].get("pressure"))
            if flow is None or p_other is None:
                continue
            found = True
            total += flow if p_other < p_in else -flow
    return total if found and total > 0 else None


def _flow_share(data: Mapping[str, Any], reference: float | None) -> float | None:
    flow = _number(data.get("flow_abs"))
    if flow is None or not reference:
        return None
    return flow / reference


def flow_reference(G: nx.MultiGraph, inlets: Iterable[Any]) -> float | None:
    """What a flow share is a share of: the inflow, else the largest flow."""
    inflow = total_inflow(G, inlets)
    if inflow:
        return inflow
    flows = [f for f in (_number(d.get("flow_abs")) for _u, _v, d in G.edges(data=True)) if f]
    return max(flows) if flows else None


def within(items: Iterable[ReviewItem], centre_um: Sequence[float] | None, radius_um: float) -> list[ReviewItem]:
    """The *items* whose centre lies within *radius_um* of *centre_um*; all
    of them for a radius of 0 or no centre."""
    items = list(items)
    if centre_um is None or not radius_um or radius_um <= 0:
        return items
    centre = np.asarray(centre_um, dtype=float)[-3:]
    return [item for item in items if float(np.linalg.norm(np.asarray(item.centre_um) - centre)) <= radius_um]


def z_range_around(
    centre_um: Sequence[float] | None,
    radius_um: float,
    voxel_size_zyx: Sequence[float],
    depth: int,
) -> tuple[int, int] | None:
    """The slices ``(first, stop)`` of a *depth*-slice volume within
    *radius_um* of *centre_um*, so a mask scan reads only those; None (every
    slice) for a radius of 0 or no centre."""
    if centre_um is None or not radius_um or radius_um <= 0:
        return None
    z = float(np.asarray(centre_um, dtype=float)[-3])
    dz = float(voxel_size_zyx[0])
    first = max(int(np.floor((z - radius_um) / dz)), 0)
    stop = min(int(np.ceil((z + radius_um) / dz)) + 1, int(depth))
    return first, max(stop, first)


# --- needing only the graph ------------------------------------------------------


def find_close_junctions(G: nx.MultiGraph) -> list[ReviewItem]:
    """Two junctions joined by a vessel shorter than the widest radius of the
    other vessels meeting them: one junction drawn as two, which the 4+
    junction list misses. Closest first. Thick-vessel bridges, short by
    design, are not counted."""
    pos = _node_pos(G)
    name = _namer(G)
    items = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if u == v or data.get(IS_ZERO_RESISTANCE) or G.degree(u) < 3 or G.degree(v) < 3:
            continue
        edge = (u, v, key)
        others = list(dict.fromkeys(
            name(a, b, k) for node in (u, v) for a, b, k in G.edges(node, keys=True)
            if {a, b} != {u, v} or k != key
        ))
        radius = max((d / 2.0 for d in (_diameter(G.edges[e]) for e in others) if d), default=0.0)
        length = _length(G, edge, pos)
        if radius <= 0 or length >= radius:
            continue
        combined = G.degree(u) + G.degree(v) - 2
        points = _points(G, [edge], pos)
        items.append(ReviewItem(
            kind=CLOSE_JUNCTIONS,
            nodes=(u, v),
            edges=(edge, *others),
            centre_um=_centre(points),
            why=(
                f"Junctions {u} and {v} are {length:.3g} µm apart, within the widest "
                f"vessel's radius ({radius:.3g} µm): one {combined}-vessel junction drawn as two."
            ),
            measures={"connector_um": length, "widest_radius_um": radius, "vessels": combined},
            points_um=_points(G, (edge, *others), pos),
        ))
    items.sort(key=lambda item: item.measures["connector_um"])
    return items


def find_hairpins(
    G: nx.MultiGraph,
    *,
    min_ratio: float = HAIRPIN_MIN_RATIO,
    min_length_um: float = HAIRPIN_MIN_LENGTH_UM,
) -> list[ReviewItem]:
    """Vessels at least *min_ratio* times longer than the straight line
    between their ends, and *min_length_um* long: a trace that doubled back.
    Most doubled first; a loop on one node is the short-loop list's.
    Thick-vessel bridges are not vessels, and are left out."""
    pos = _node_pos(G)
    items = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if u == v or data.get(IS_ZERO_RESISTANCE):
            continue
        edge = (u, v, key)
        path = _path(G, edge, pos)
        if len(path) < 2:
            continue
        length = _length(G, edge, pos)
        chord = float(np.linalg.norm(path[-1] - path[0]))
        ratio = length / chord if chord > 0 else float("inf")
        if length < min_length_um or ratio < min_ratio:
            continue
        items.append(ReviewItem(
            kind=HAIRPINS,
            nodes=(u, v),
            edges=(edge,),
            centre_um=_centre(path),
            why=f"A {length:.3g} µm vessel whose ends are {chord:.3g} µm apart: it doubles back.",
            measures={"length_um": length, "chord_um": chord, "ratio": ratio},
            points_um=path,
        ))
    items.sort(key=lambda item: -item.measures["ratio"])
    return items


def find_diameter_jumps(
    G: nx.MultiGraph,
    *,
    chain_ratio: float = DIAMETER_CHAIN_RATIO,
    daughter_ratio: float = DIAMETER_DAUGHTER_RATIO,
) -> list[ReviewItem]:
    """Where a measured diameter jumps: at a pass-through node one vessel at
    least *chain_ratio* times the other, and at a junction a daughter at least
    *daughter_ratio* times its parent (the parent as the statistics read it:
    the one lowest-ranked branch order). Only measured diameters count, and
    thick-vessel bridges are left out. Largest jump first."""
    from haemolynx.statistics.bifurcation import _iter_parent_daughter_junctions, _measured_diameter_or_none

    pos = _node_pos(G)
    name = _namer(G)
    items = []
    for node in G.nodes:
        incident = [(a, b, k) for a, b, k in G.edges(node, keys=True) if a != b]
        if G.degree(node) != 2 or len(incident) != 2:
            continue
        datas = [G.edges[e] for e in incident]
        if any(d.get(IS_ZERO_RESISTANCE) for d in datas):
            continue
        widths = [_measured_diameter_or_none(d) for d in datas]
        if None in widths:
            continue
        ratio = max(widths) / min(widths)
        if ratio < chain_ratio:
            continue
        edges = tuple(name(*e) for e in incident)
        items.append(ReviewItem(
            kind=DIAMETER_JUMPS,
            nodes=(node,),
            edges=edges,
            centre_um=_centre([pos[node]]) if node in pos else _centre(_points(G, edges, pos)),
            why=f"One vessel runs on into another {ratio:.3g}× its width ({min(widths):.3g} → {max(widths):.3g} µm).",
            measures={"ratio": ratio, "narrow_um": min(widths), "wide_um": max(widths), "at": "pass-through"},
            points_um=_points(G, edges, pos),
        ))
    for node, parent, daughters in _iter_parent_daughter_junctions(G):
        # A bridge has no branch order of its own, so it is never one of
        # these; one a hand edit labelled is still not a vessel.
        if parent[3].get(IS_ZERO_RESISTANCE):
            continue
        daughters = [item for item in daughters if not item[3].get(IS_ZERO_RESISTANCE)]
        parent_width = _measured_diameter_or_none(parent[3])
        if parent_width is None:
            continue
        wide = [
            (item, width) for item in daughters
            if (width := _measured_diameter_or_none(item[3])) is not None and width >= daughter_ratio * parent_width
        ]
        if not wide:
            continue
        widest = max(width for _item, width in wide)
        edges = tuple(name(*item[:3]) for item in (parent, *(item for item, _w in wide)))
        items.append(ReviewItem(
            kind=DIAMETER_JUMPS,
            nodes=(node,),
            edges=edges,
            centre_um=_centre([pos[node]]) if node in pos else _centre(_points(G, edges, pos)),
            why=(
                f"A daughter {widest / parent_width:.3g}× as wide as its parent "
                f"({parent_width:.3g} → {widest:.3g} µm, parent {parent[3].get('branch_order')})."
            ),
            measures={"ratio": widest / parent_width, "narrow_um": parent_width, "wide_um": widest, "at": "junction"},
            points_um=_points(G, edges, pos),
        ))
    items.sort(key=lambda item: -item.measures["ratio"])
    return items


def find_unmeasured_diameters(G: nx.MultiGraph, inlets: Iterable[Any] = ()) -> list[ReviewItem]:
    """Vessels whose diameter is not a measurement of their own -- the class
    median or the branch-order table -- or whose FWHM width a check called
    into doubt (set aside, at odds with the mask's width, or in the range
    FWHM reads off specks). Largest share of the network's flow first,
    since that is where a wrong width moves the result most."""
    from haemolynx.haemodynamics.poiseuille import (
        DIAMETER_SOURCE_CLASS_MEDIAN,
        DIAMETER_SOURCE_TABLE,
        fwhm_demotion_reason,
    )

    pos = _node_pos(G)
    reference = flow_reference(G, inlets)
    items = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(IS_ZERO_RESISTANCE):
            continue
        source = data.get("diameter_source")
        reasons = []
        if source == DIAMETER_SOURCE_CLASS_MEDIAN:
            reasons.append("its class's median width")
        elif source == DIAMETER_SOURCE_TABLE:
            reasons.append("the branch-order table's width")
        demoted = fwhm_demotion_reason(data)
        if demoted:
            reasons.append(f"FWHM set aside ({demoted})")
        elif data.get("fwhm_low_confidence_vs_edt"):
            reasons.append("FWHM at odds with the mask's width")
        elif data.get("fwhm_in_speck_width_range"):
            reasons.append("FWHM in the width range of specks")
        if not reasons:
            continue
        edge = (u, v, key)
        share = _flow_share(data, reference) if _is_solved(data) else None
        path = _path(G, edge, pos)
        said = "; ".join(reasons)
        items.append(ReviewItem(
            kind=UNMEASURED_DIAMETERS,
            nodes=(u, v),
            edges=(edge,),
            centre_um=_centre(path),
            why=f"Its width is {said}." if source in (DIAMETER_SOURCE_CLASS_MEDIAN, DIAMETER_SOURCE_TABLE)
            else f"{said[0].upper()}{said[1:]}.",
            measures={
                "diameter_um": _diameter(data),
                "diameter_source": source,
                "edt_diameter_um": _number(data.get("edt_diameter_um")),
                "fwhm_diameter_um": _number(data.get("fwhm_diameter_um")),
                "flow_share": share,
                "reasons": "; ".join(reasons),
            },
            points_um=path,
        ))
    items.sort(key=lambda item: -(item.measures["flow_share"] or 0.0))
    return items


# --- needing the solve -------------------------------------------------------------


def find_unsolved_pieces(G: nx.MultiGraph, inlets: Iterable[Any], outlets: Iterable[Any]) -> list[ReviewItem]:
    """The vessels no inlet-to-outlet path runs along, piece by piece
    (:func:`~haemolynx.graph.connectivity.unsolved_pieces`), longest first: a
    long one is more often a missed connection than debris."""
    from .connectivity import unsolved_pieces

    pos = _node_pos(G)
    items = []
    for piece in unsolved_pieces(G, list(inlets), list(outlets)):
        points = _points(G, piece.edges, pos)
        where = (
            f"hangs off node(s) {', '.join(str(n) for n in piece.attached_at)}"
            if piece.attached_at else "is a piece of its own"
        )
        items.append(ReviewItem(
            kind=UNSOLVED_PIECES,
            nodes=piece.attached_at,
            edges=piece.edges,
            centre_um=_centre(points),
            why=f"{len(piece.edges)} vessel(s), {piece.length_um:.3g} µm, no inlet-to-outlet path: it {where}.",
            measures={
                "vessels": len(piece.edges),
                "length_um": piece.length_um,
                "attached_at": ";".join(str(n) for n in piece.attached_at),
                "disconnected": piece.disconnected,
            },
            points_um=points,
        ))
    return items


def _downhill(G: nx.MultiGraph, edge: EdgeKey) -> tuple[Any, Any] | None:
    """``(upstream, downstream)`` ends of a solved vessel, by node pressure."""
    u, v, _key = edge
    p_u, p_v = _number(G.nodes[u].get("pressure")), _number(G.nodes[v].get("pressure"))
    if p_u is None or p_v is None or p_u == p_v:
        return None
    return (u, v) if p_u > p_v else (v, u)


def find_flow_against_branch_order(G: nx.MultiGraph, inlets: Iterable[Any] = ()) -> list[ReviewItem]:
    """Junctions where the solved flow runs against the branch orders: blood
    passing from a vessel into one nearer the arterial side (generations from
    the capillary bed, :func:`~haemolynx.graph.branch_order.branch_order_signed_values`,
    falling along the flow), or an arteriole meeting a venule with no
    capillary there. Usually a boundary or a branch order assigned wrongly,
    or a false connection. Largest share of the flow first."""
    from .branch_order import branch_order_signed_values

    pos = _node_pos(G)
    solved = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if u != v and _is_solved(d)]
    generation = dict(zip(
        solved, branch_order_signed_values(G.edges[e].get("branch_order") for e in solved)
    ))
    reference = flow_reference(G, inlets)
    into: dict[Any, list[EdgeKey]] = {}
    out_of: dict[Any, list[EdgeKey]] = {}
    for edge in solved:
        ends = _downhill(G, edge)
        if ends is not None:
            out_of.setdefault(ends[0], []).append(edge)
            into.setdefault(ends[1], []).append(edge)
    items = []
    for node in G.nodes:
        feeding, draining = into.get(node, []), out_of.get(node, [])
        backwards = [
            (a, b) for a in feeding for b in draining
            if np.isfinite(generation[a]) and np.isfinite(generation[b]) and generation[b] < generation[a]
        ]
        present = [generation[e] for e in (*feeding, *draining) if np.isfinite(generation[e])]
        shunt = any(g < 0 for g in present) and any(g > 0 for g in present) and not any(g == 0 for g in present)
        if not backwards and not shunt:
            continue
        edges = tuple(dict.fromkeys(e for pair in backwards for e in pair)) or tuple(feeding + draining)
        labels = {e: str(G.edges[e].get("branch_order")) for e in edges}
        share = max((_flow_share(G.edges[e], reference) or 0.0) for e in edges)
        if backwards:
            a, b = backwards[0]
            why = f"Flow runs from {labels[a]} into {labels[b]}, back towards the arterial side."
        else:
            why = "An arteriole meets a venule here with no capillary between them."
        items.append(ReviewItem(
            kind=FLOW_AGAINST_ORDER,
            nodes=(node,),
            edges=edges,
            centre_um=_centre([pos[node]]) if node in pos else _centre(_points(G, edges, pos)),
            why=why,
            measures={
                "flow_share": share,
                "orders": ";".join(labels[e] for e in edges),
                "arteriole_meets_venule": shunt,
            },
            points_um=_points(G, edges, pos),
        ))
    items.sort(key=lambda item: -item.measures["flow_share"])
    return items


def find_flow_outliers(
    G: nx.MultiGraph,
    *,
    robust_z: float = FLOW_OUTLIER_ROBUST_Z,
    min_class: int = FLOW_OUTLIER_MIN_CLASS,
    min_fold: float = FLOW_OUTLIER_MIN_FOLD,
) -> list[ReviewItem]:
    """Solved vessels whose mean velocity (flow over the cross-section of
    their diameter) lies more than *robust_z* robust standard deviations from
    the median of their vessel class, in log10, and at least *min_fold* times
    off it -- compared within arterioles, capillaries, venules and so on, so
    a fast arteriole is not an outlier among capillaries. A solved vessel
    with no flow at all is listed too. Furthest from its class first."""
    from haemolynx.statistics.bifurcation import branch_order_category

    pos = _node_pos(G)
    by_class: dict[str, list[tuple[EdgeKey, float]]] = {}
    still = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if u == v or not _is_solved(data) or data.get(IS_ZERO_RESISTANCE):
            continue
        diameter = _diameter(data)
        if diameter is None:
            continue
        flow = float(data["flow_abs"])
        if flow <= 0:
            still.append((u, v, key))
            continue
        area = np.pi * (diameter * 0.5e-6) ** 2
        by_class.setdefault(branch_order_category(data.get("branch_order")), []).append(
            ((u, v, key), flow / area * 1e3)
        )
    items = []

    def item(edge, velocity, kind_of, median, z, why) -> ReviewItem:
        path = _path(G, edge, pos)
        return ReviewItem(
            kind=FLOW_OUTLIERS,
            nodes=edge[:2],
            edges=(edge,),
            centre_um=_centre(path),
            why=why,
            measures={"velocity_mm_s": velocity, "class": kind_of, "class_median_mm_s": median, "robust_z": z},
            points_um=path,
        )

    for edge in still:
        kind_of = branch_order_category(G.edges[edge].get("branch_order"))
        items.append(item(edge, 0.0, kind_of, None, float("-inf"), "A solved vessel with no flow through it."))
    for kind_of, members in by_class.items():
        if len(members) < min_class:
            continue
        logs = np.log10([velocity for _e, velocity in members])
        median = float(np.median(logs))
        spread = 1.4826 * float(np.median(np.abs(logs - median)))
        if spread <= 0:
            continue
        for (edge, velocity), value in zip(members, logs):
            z = (float(value) - median) / spread
            if abs(z) <= robust_z or abs(float(value) - median) < np.log10(min_fold):
                continue
            typical = 10.0 ** median
            items.append(item(
                edge, velocity, kind_of, typical, z,
                f"{velocity:.3g} mm/s, against a median of {typical:.3g} mm/s among the "
                f"{len(members)} {kind_of.replace('_', ' ')} vessels ({abs(z):.3g} robust SDs "
                f"{'faster' if z > 0 else 'slower'}).",
            ))
    items.sort(key=lambda it: -abs(it.measures["robust_z"]))
    return items


def _face_readers(
    G: nx.MultiGraph,
    pos: Mapping[Any, np.ndarray],
    extent_um: Sequence[float] | None,
    image_face_margin_um: float | None,
):
    """Two readers of where a node lies against the image's faces: its depth
    inside the image in microns, and whether it leaves the image (None when
    neither can be told). A dead end leaves it by the one definition
    (:func:`haemolynx.graph.dead_ends.node_leaves_image`); any other node
    when it sits on a face. *image_face_margin_um*, when given, replaces both
    with a fixed distance."""
    from .dead_ends import graph_image_extent_um, leaves_image, node_leaves_image

    extent = np.asarray(extent_um, dtype=float) if extent_um is not None else graph_image_extent_um(G)

    def depth(node) -> float | None:
        if extent is None or node not in pos:
            return None
        return float(np.min(np.minimum(pos[node], extent - pos[node])))

    def leaves(node) -> bool | None:
        if image_face_margin_um is not None:
            inside = depth(node)
            return None if inside is None else inside <= float(image_face_margin_um)
        if G.degree(node) != 1:
            # Not a dead end: whether it at least sits on a face.
            if extent is None or node not in pos:
                return None
            return leaves_image(pos[node], extent, voxel_size_zyx=G.graph.get("image_voxel_size_zyx"))[0]
        if extent is None and G.nodes[node].get("dead_end_kind") is None:
            return None
        return node_leaves_image(G, node, extent)

    return depth, leaves


def find_terminal_nodes(
    G: nx.MultiGraph,
    roles: Mapping[str, Iterable[Any]],
    *,
    extent_um: Sequence[float] | None = None,
) -> list[ReviewItem]:
    """The terminal nodes, for a person to make one an inlet or an outlet or
    take the role away: the inlets, then the outlets, then the dead ends
    leaving the image that are neither -- the vessels the image cut, where
    blood can come in or go out -- then every other dead end; the dead ends
    widest first. A vessel with a dead end at both ends (a piece on its own)
    is left out: nothing flows through it whatever its ends are. *roles* maps
    ``"inlet"``, ``"outlet"`` and the other boundary roles to their nodes.

    Whether a dead end leaves the image is the one definition
    (:func:`haemolynx.graph.dead_ends.node_leaves_image`), against the
    image's extent *extent_um* or the graph's own."""
    pos = _node_pos(G)
    name = _namer(G)
    inlets = [n for n in dict.fromkeys(roles.get("inlet") or ()) if n in G]
    outlets = [n for n in dict.fromkeys(roles.get("outlet") or ()) if n in G and n not in inlets]
    others: dict[Any, list[str]] = {}
    for role, nodes in roles.items():
        if role in ("inlet", "outlet"):
            continue
        for node in nodes or ():
            others.setdefault(node, []).append(str(role))
    depth, leaves = _face_readers(G, pos, extent_um, None)
    reference = flow_reference(G, inlets)

    def item(node, order: int, role: str) -> tuple[tuple, ReviewItem]:
        edges = tuple(dict.fromkeys(name(a, b, k) for a, b, k in G.edges(node, keys=True)))
        widths = [w for w in (_diameter(G.edges[e]) for e in edges) if w is not None]
        flows = [s for s in (_flow_share(G.edges[e], reference) for e in edges if _is_solved(G.edges[e]))
                 if s is not None]
        out = leaves(node)
        inside = depth(node)
        kind = G.nodes[node].get("dead_end_kind")
        where = (
            "leaving the image" if out
            else f"{inside:.3g} µm inside the image" if inside is not None
            else "where the image's faces are not known"
        )
        if role:
            why = f"Node {node} is an {role}, {where}"
            if G.degree(node) != 1:
                why += f"; not a dead end: {G.degree(node)} vessels meet there"
            why += "."
        elif out:
            why = (f"Node {node} is a dead end {where}, neither inlet nor outlet: the vessel the image cut "
                   "is an open end in the solve (open_end_model) unless it is made one.")
        else:
            why = f"Node {node} is a dead end {where}" + (f" ({str(kind).replace('_', ' ')})." if kind else ".")
        widest = max(widths) if widths else None
        found = ReviewItem(
            kind=TERMINAL_NODES,
            nodes=(node,),
            edges=edges,
            centre_um=_centre([pos[node]]) if node in pos else _centre(_points(G, edges, pos)),
            why=why,
            measures={
                "role": role,
                "other_roles": ";".join(others.get(node, ())),
                "vessels": G.degree(node),
                "leaves_image": out,
                "dead_end_kind": kind,
                "depth_um": inside,
                "diameter_um": widest,
                "flow_share": max(flows) if flows else None,
            },
            points_um=_points(G, edges, pos, extra=[pos[node]] if node in pos else ()),
        )
        return (order, -(widest or 0.0)), found

    ranked = [item(node, 0, "inlet") for node in inlets] + [item(node, 1, "outlet") for node in outlets]
    for node in G.nodes:
        if node in inlets or node in outlets or G.degree(node) != 1:
            continue
        (_a, other, _k), = G.edges(node, keys=True)
        if G.degree(other) == 1:
            continue
        ranked.append(item(node, 2 if leaves(node) else 3, ""))
    # Inlets and outlets keep their lists' order; the dead ends go widest first.
    ranked.sort(key=lambda pair: pair[0] if pair[0][0] >= 2 else (pair[0][0], 0.0))
    return [found for _key, found in ranked]


def find_boundary_issues(
    G: nx.MultiGraph,
    roles: Mapping[str, Iterable[Any]],
    *,
    extent_um: Sequence[float] | None = None,
    image_face_margin_um: float | None = None,
    min_flow_share: float = BOUNDARY_MIN_FLOW_SHARE,
) -> list[ReviewItem]:
    """What looks wrong with the boundary nodes, issue by issue: an inlet or
    outlet that is not a dead end; one that does not leave the image; a dead
    end that leaves the image but is no boundary node -- a vessel the image
    cut, closed in the solve unless the run gives open ends a boundary
    condition; and an inlet or outlet carrying less than *min_flow_share* of
    the network's inflow. *roles* maps ``"inlet"``, ``"outlet"`` and the
    other boundary roles to their nodes.

    Whether a dead end leaves the image is the one definition
    (:func:`haemolynx.graph.dead_ends.node_leaves_image`: its kind when the
    run marked one, else its distance to a face, the image's extent
    *extent_um* or the graph's own). *image_face_margin_um*, when given,
    replaces it with a fixed distance."""
    pos = _node_pos(G)
    name = _namer(G)
    role_of: dict[Any, str] = {}
    for role, nodes in roles.items():
        for node in nodes or ():
            role_of.setdefault(node, str(role))
    ends = [n for n in (*(roles.get("inlet") or ()), *(roles.get("outlet") or ())) if n in G]
    depth, leaves = _face_readers(G, pos, extent_um, image_face_margin_um)

    reference = flow_reference(G, roles.get("inlet") or ())
    found: list[tuple[int, ReviewItem]] = []

    def add(order: int, node, issue: str, why: str) -> None:
        edges = tuple(dict.fromkeys(name(a, b, k) for a, b, k in G.edges(node, keys=True)))
        flows = [(_flow_share(G.edges[e], reference) or 0.0) for e in edges if _is_solved(G.edges[e])]
        found.append((order, ReviewItem(
            kind=BOUNDARY_NODES,
            nodes=(node,),
            edges=edges,
            centre_um=_centre([pos[node]]) if node in pos else _centre(_points(G, edges, pos)),
            why=why,
            measures={
                "role": role_of.get(node, ""),
                "issue": issue,
                "vessels": G.degree(node),
                "depth_um": depth(node),
                "flow_share": max(flows) if flows else None,
            },
            points_um=_points(G, edges, pos, extra=[pos[node]] if node in pos else ()),
        )))

    for node in dict.fromkeys(ends):
        role = role_of[node]
        if G.degree(node) != 1:
            add(0, node, "not a dead end", f"The {role} node {node} is not a dead end: {G.degree(node)} vessels meet there.")
        if leaves(node) is False:
            inside = depth(node)
            add(1, node, "inside the image",
                f"The {role} node {node} does not leave the image"
                + (f": it lies {inside:.3g} µm inside it." if inside is not None else "."))
        edges = list(G.edges(node, keys=True))
        shares = [_flow_share(G.edges[e], reference) for e in edges if _is_solved(G.edges[e])]
        shares = [s for s in shares if s is not None]
        if shares and max(shares) < min_flow_share:
            add(3, node, "little flow", f"The {role} node {node} carries {max(shares):.2%} of the inflow.")
    for node in G.nodes:
        if node in role_of or G.degree(node) != 1:
            continue
        (_a, other, _k), = G.edges(node, keys=True)
        if G.degree(other) == 1:
            continue
        if leaves(node):
            add(2, node, "open end at a face",
                f"Node {node} is a dead end leaving the image but no boundary node: the vessel the image cut "
                "is closed in the solve unless open ends get a boundary condition (open_end_model).")
    found.sort(key=lambda pair: pair[0])
    return [item for _order, item in found]


# --- needing the segmented mask ------------------------------------------------------


#: What each kind of dead end is, as the dead-end list says it, in the order
#: the list groups them.
DEAD_END_WORDS = {
    "artefact": "not a vessel end",
    "thick_wall": "it stops at a thick vessel's wall, not joined to it",
    "stops_short": "the segmented vessel runs on past its tip",
    "broken": "another vessel lies straight ahead across a gap",
    "unknown": "a blind end, or a broken vessel",
    "blind": "a real blind end (marked by hand)",
    "island_end": "an end of a piece with no junction",
}
_ARTEFACT_WORDS = {
    "inside_other_lumen": "its tip lies inside another vessel's lumen",
    "beside_vessel": "it runs beside another vessel in the same lumen",
    "off_mask": "it lies mostly off the segmented mask",
    "mask_continues": "the segmented vessel runs on past its tip",
}


def find_dead_ends(
    G: nx.MultiGraph,
    support,
    *,
    stub_radius_multiple: float = 3.0,
    image_face_margin_um: float | None = None,
) -> list[ReviewItem]:
    """Every dead end that does not leave the image, with its kind
    (:mod:`haemolynx.graph.dead_ends`, judged against the mask) and its chain
    back to the junction, grouped by kind in :data:`DEAD_END_WORDS` order --
    artefacts first -- and longest first within each. (Ranking by the turn
    onto the vessel ahead, ``ahead_turn_deg``, was tried: G3 fitted it 24/27,
    but the fresh grid G3′ scored it 21/30 against 29/30 for "no join" every
    time, Harvey 2026-10-25, so it stays a measure, not an order.) A piece with no junction is listed
    once, not once per end. *image_face_margin_um*, when given, replaces the
    face test with a fixed distance."""
    from .dead_ends import LEAVES_IMAGE, classify_dead_ends, image_extent_um

    ends = classify_dead_ends(
        G, extent_um=image_extent_um(support.mask.shape, support.voxel_size_zyx), support=support,
        stub_radius_multiple=stub_radius_multiple, face_margin_um=image_face_margin_um,
    )
    order = list(DEAD_END_WORDS)
    pos = _node_pos(G)
    name = _namer(G)
    seen: set = set()
    items = []
    for tip, end in ends.items():
        if end.kind == LEAVES_IMAGE:
            continue
        chain = tuple(name(*e) for e in end.chain)
        if frozenset(chain) in seen:
            continue
        seen.add(frozenset(chain))
        points = _points(G, chain, pos)
        details = [_ARTEFACT_WORDS[f] for f in end.flags if f in _ARTEFACT_WORDS]
        if "short" in end.flags:
            details.append(f"it is shorter than {stub_radius_multiple:g} radii of the vessel it leaves")
        why = f"A dead end ({end.kind.replace('_', ' ')}): {DEAD_END_WORDS.get(end.kind, end.kind)}"
        why += ("; " + "; ".join(details) if details and end.kind == "artefact" else "") + "."
        measures = {
            "kind": end.kind,
            "kinds": ";".join(end.flags),
            "length_um": end.chain_length_um,
            "vessels": len(chain),
        }
        for key in ("tip_radius_um", "junction_radius_um", "mask_run_um", "ahead_um", "ahead_turn_deg",
                    "ahead_junctions_away", "face_distance_um", "in_mask_fraction"):
            measures[key] = getattr(end, key)
        # Where "Join to the vessel ahead" joins it: the vessel straight ahead
        # and the point the heading meets it.
        if end.ahead_um is not None and end.ahead_edge is not None and end.heading is not None:
            measures["ahead_edge"] = tuple(name(*end.ahead_edge))
            measures["ahead_point_um"] = tuple(
                float(c) for c in np.asarray(end.tip_um) + float(end.ahead_um) * np.asarray(end.heading)
            )
        rank = order.index(end.kind) if end.kind in order else len(order)
        items.append((rank, -end.chain_length_um, ReviewItem(
            kind=DEAD_ENDS,
            nodes=(tip, end.junction),
            edges=chain,
            centre_um=_centre([pos[tip]]) if tip in pos else _centre(points),
            why=why,
            measures=measures,
            points_um=points,
        )))
    items.sort(key=lambda entry: entry[:2])
    return [item for _rank, _length, item in items]


def find_dead_ends_at_thick_walls(
    G: nx.MultiGraph,
    support,
    *,
    stub_radius_multiple: float = 3.0,
) -> list[ReviewItem]:
    """Every dead end the definition calls ``thick_wall`` (:mod:`haemolynx.graph.dead_ends`):
    a vessel stopping within reach of a thick vessel's lumen, off its
    centreline, that the thick-vessel split did not join to it -- a vessel
    opening into the thick one the wall join refused (turning away, across
    background, beside a vessel in the same lumen, closing a loop), or one
    ending beside it. Nearest the lumen first. Each carries how far its tip
    lies past the lumen, the turn onto the thick centreline, and where on the
    graph's own thick centreline a join lands (``ahead_edge``,
    ``ahead_point_um``: "Join to the vessel ahead" joins it there, and post
    processing bridges the stretch inside the lumen). Empty for a graph
    without a thick centreline recorded (``dead_ends.record_thick_centreline``)."""
    from .dead_ends import THICK_WALL, classify_dead_ends, image_extent_um

    ends = classify_dead_ends(
        G, extent_um=image_extent_um(support.mask.shape, support.voxel_size_zyx), support=support,
        stub_radius_multiple=stub_radius_multiple,
    )
    pos = _node_pos(G)
    name = _namer(G)
    items = []
    for tip, end in ends.items():
        if end.kind != THICK_WALL:
            continue
        chain = tuple(name(*e) for e in end.chain)
        edge = None if end.thick_edge is None or not G.has_edge(*end.thick_edge) else tuple(name(*end.thick_edge))
        point = None if edge is None else tuple(float(c) for c in end.thick_point_um)
        inside = (end.thick_lumen_um or 0.0) < 0.0
        turn = "" if end.thick_turn_deg is None else f", turning {end.thick_turn_deg:.0f}° onto its centreline"
        why = (
            f"Dead end {tip} stops "
            + (f"inside a thick vessel's lumen, {-end.thick_lumen_um:.3g} µm in" if inside
               else f"{end.thick_lumen_um:.3g} µm from a thick vessel's lumen")
            + f"{turn}, not joined to it: a vessel opening into it, or one ending beside it."
        )
        measures = {
            "kind": end.kind,
            "length_um": end.chain_length_um,
            "vessels": len(chain),
            "thick_lumen_um": end.thick_lumen_um,
            "thick_centreline_um": end.thick_centreline_um,
            "thick_turn_deg": end.thick_turn_deg,
            "tip_radius_um": end.tip_radius_um,
            "mask_run_um": end.mask_run_um,
            "in_mask_fraction": end.in_mask_fraction,
            "ahead_edge": edge,
            "ahead_point_um": point,
        }
        points = _points(G, chain, pos, extra=[np.asarray(point)] if point is not None else ())
        items.append((float(end.thick_lumen_um), ReviewItem(
            kind=THICK_WALL_ENDS,
            nodes=(tip, end.junction),
            edges=chain,
            centre_um=_centre([pos[tip]]) if tip in pos else _centre(points),
            why=why,
            measures=measures,
            points_um=points,
        )))
    items.sort(key=lambda entry: entry[0])
    return [item for _distance, item in items]


def find_two_in_one_lumen(G: nx.MultiGraph, support) -> list[ReviewItem]:
    """Pairs of vessels running side by side through one segmented vessel
    (:func:`~haemolynx.graph.diagnostics.diagnose_parallel_duplicates_in_lumen`),
    longest first, saying for each whether deleting it would cut vessels off
    (:func:`~haemolynx.graph.lumen_loops.cuts_off_vessels`)."""
    from .diagnostics import diagnose_parallel_duplicates_in_lumen
    from .lumen_loops import cuts_off_vessels

    pos = _node_pos(G)
    name = _namer(G)
    report = diagnose_parallel_duplicates_in_lumen(
        G, support.mask, voxel_size_zyx=support.voxel_size_zyx, mask_support=support
    )
    items = []
    for a, b in report["duplicate_pairs"]:
        edges = tuple(name(*e) for e in (a, b))
        if not all(G.has_edge(*e) for e in edges):
            continue
        lengths = [_length(G, e, pos) for e in edges]
        cuts = [cuts_off_vessels(G, e) for e in edges]
        points = _points(G, edges, pos)
        items.append(ReviewItem(
            kind=TWO_IN_ONE_LUMEN,
            nodes=tuple(dict.fromkeys(n for e in edges for n in e[:2])),
            edges=edges,
            centre_um=_centre(points),
            why=(
                "Two vessels run side by side through one segmented vessel"
                + (": deleting either would cut vessels off." if all(cuts) else ".")
            ),
            measures={
                "shorter_um": min(lengths),
                "lengths_um": ";".join(f"{x:.3g}" for x in lengths),
                "cuts_off": ";".join(str(c) for c in cuts),
            },
            points_um=points,
        ))
    items.sort(key=lambda item: -item.measures["shorter_um"])
    return items


def find_facing_ends(
    G: nx.MultiGraph,
    support,
    *,
    max_gap_um: float | None = None,
    max_turn_deg: float | None = None,
    gap_factor: float = FACING_REVIEW_GAP_FACTOR,
    review_turn_deg: float = FACING_REVIEW_MAX_TURN_DEG,
) -> list[ReviewItem]:
    """Pairs of dead ends heading for each other that the join left open
    (:func:`~haemolynx.graph.facing_ends.facing_dead_end_pairs`), looking
    *gap_factor* times as far as the join (*max_gap_um*, the run's
    ``facing_dead_end_max_gap_um``) and at turns up to *review_turn_deg*,
    each with why it was not joined. Closest first."""
    from .facing_ends import DEFAULT_FACING_MAX_GAP_UM, FACING_MAX_TURN_DEG, facing_dead_end_pairs

    gap = DEFAULT_FACING_MAX_GAP_UM if max_gap_um is None else float(max_gap_um)
    turn = FACING_MAX_TURN_DEG if max_turn_deg is None else float(max_turn_deg)
    if gap <= 0:
        gap = DEFAULT_FACING_MAX_GAP_UM
    pos = _node_pos(G)
    name = _namer(G)
    items = []
    for pair in facing_dead_end_pairs(
        G, support, max_gap_um=gap, max_turn_deg=turn, search_gap_um=gap * gap_factor,
    ):
        if pair.turn_deg > review_turn_deg:
            continue
        chains = list(dict.fromkeys(name(*e) for tip in (pair.a, pair.b) for e in _chain_to_junction(G, tip)[0]))
        points = _points(G, chains, pos)
        ends = [pos[n] for n in (pair.a, pair.b) if n in pos]
        items.append(ReviewItem(
            kind=FACING_ENDS,
            nodes=(pair.a, pair.b),
            edges=tuple(chains),
            centre_um=_centre(ends, fallback=_centre(points)),
            why=(
                f"Dead ends {pair.a} and {pair.b} head for each other across {pair.gap_um:.3g} µm "
                f"(turning {pair.turn_deg:.0f}°), not joined: {pair.refused or 'they would be joined'}."
            ),
            measures={"gap_um": pair.gap_um, "turn_deg": pair.turn_deg, "refused": pair.refused or ""},
            points_um=points,
        ))
    return items


def find_broken_vessels(
    G: nx.MultiGraph,
    support,
    *,
    max_gap_um: float | None = None,
    stub_radius_multiple: float = 3.0,
) -> list[ReviewItem]:
    """The "Broken vessels" list: every pair of facing dead ends the join
    left open (:func:`find_facing_ends`), and every dead end the definition
    calls ``broken`` with another vessel's side straight ahead, not one of
    its neighbours (``join_dead_ends_to_vessel_sides``, off by default,
    decision D3) -- each with how it would be joined (``join``:
    :data:`JOIN_ENDS` or :data:`JOIN_SIDE`) and the same measures, so their
    decisions share one CSV. A pair whose ends both leave the image is two
    vessels running out of it, not one broken vessel, and is left out.
    Closest first. On the fresh grid G3′ (Harvey, 2026-10-25) 1 of 30 such
    joins was the same vessel."""
    from .dead_ends import BROKEN, LEAVES_IMAGE, classify_dead_ends, image_extent_um, leaves_image

    extent = image_extent_um(support.mask.shape, support.voxel_size_zyx)
    ends = classify_dead_ends(G, extent_um=extent, support=support, stub_radius_multiple=stub_radius_multiple)

    def runs_out(node: Any) -> bool:
        # The face test itself: an end of a piece with no junction is an
        # island end first, whether or not it leaves the image.
        end = ends.get(node)
        return end is not None and (end.kind == LEAVES_IMAGE or leaves_image(
            end.tip_um, extent, heading=end.heading, support=support, tip_radius_um=end.tip_radius_um,
            voxel_size_zyx=support.voxel_size_zyx,
        )[0])

    blank = {"ahead_edge": None, "ahead_point_um": None, "ahead_junctions_away": None, "ahead_background_um": None}
    items = [
        replace(item, measures={"join": JOIN_ENDS, **item.measures, **blank})
        for item in find_facing_ends(G, support, max_gap_um=max_gap_um)
        if not all(runs_out(n) for n in item.nodes)
    ]
    paired = {n for item in items for n in item.nodes}
    pos = _node_pos(G)
    name = _namer(G)
    for tip, end in ends.items():
        if (end.kind != BROKEN or tip in paired or end.ahead_edge is None or end.ahead_um is None
                or end.heading is None or not G.has_edge(*end.ahead_edge)):
            continue
        chain = [name(*e) for e in end.chain]
        ahead = tuple(name(*end.ahead_edge))
        target = np.asarray(end.tip_um) + float(end.ahead_um) * np.asarray(end.heading)
        away = "not reached in the network" if end.ahead_junctions_away is None else (
            f"{end.ahead_junctions_away} junctions away in the network"
        )
        items.append(ReviewItem(
            kind=FACING_ENDS,
            nodes=(tip, end.junction),
            edges=tuple(dict.fromkeys([*chain, ahead])),
            centre_um=_centre([np.asarray(end.tip_um), target]),
            why=(
                f"Dead end {tip} points at the side of a vessel {end.ahead_um:.3g} µm ahead (turning "
                f"{end.ahead_turn_deg:.0f}°), {away}, across "
                f"{float(end.ahead_background_um or 0.0):.3g} µm of background: a vessel the segmentation "
                "broke, or a real end. Not joined: tip-to-side joins are a review decision."
            ),
            measures={
                "join": JOIN_SIDE,
                "gap_um": float(end.ahead_um),
                "turn_deg": float(end.ahead_turn_deg),
                "refused": "",
                "ahead_edge": ahead,
                "ahead_point_um": tuple(float(c) for c in target),
                "ahead_junctions_away": end.ahead_junctions_away,
                "ahead_background_um": end.ahead_background_um,
            },
            points_um=_points(G, chain, pos),
        ))
    items.sort(key=lambda item: (item.measures["gap_um"], item.measures["turn_deg"]))
    return items


def find_uncovered_mask(
    G: nx.MultiGraph,
    support,
    *,
    min_region_volume_um3: float | None = None,
    z_range: tuple[int, int] | None = None,
) -> list[ReviewItem]:
    """Segmented vessels no centreline covers
    (:func:`~haemolynx.graph.mask_recovery.uncovered_mask_regions`), largest
    first: a branch the graph lost, for Add vessel to trace."""
    from .mask_recovery import DEFAULT_MIN_REGION_VOLUME_UM3, uncovered_mask_regions

    volume = DEFAULT_MIN_REGION_VOLUME_UM3 if min_region_volume_um3 is None else float(min_region_volume_um3)
    items = []
    for region in uncovered_mask_regions(G, support, min_region_volume_um3=volume, z_range=z_range):
        items.append(ReviewItem(
            kind=UNCOVERED_MASK,
            nodes=(),
            edges=(),
            centre_um=region.centre_um,
            why=(
                f"{region.volume_um3:.3g} µm³ of segmented vessel, up to {2 * region.radius_um:.3g} µm "
                "across, that no centreline covers."
            ),
            measures={"volume_um3": region.volume_um3, "radius_um": region.radius_um, "voxels": region.voxel_count},
            points_um=region.sample_um,
        ))
    return items
