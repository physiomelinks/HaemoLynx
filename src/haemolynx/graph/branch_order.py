"""Assign branch orders to graph edges via BFS from boundary nodes."""
from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any

import networkx as nx
import numpy as np

from ._helpers import edge_id, sort_nodes
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

logger = logging.getLogger(__name__)

PostAssignCallback = Callable[[nx.MultiGraph], None]

#: The order this module's own hierarchical tiers are assigned in --
#: Large_Art outermost on the arterial side, then Art, then the plain
#: capillary tier (``prefix="B"``, see :func:`assign_branch_orders`), then
#: Ven, then Large_Ven outermost on the venous side. The single source of
#: truth for "which vessel category sorts before which": both
#: ``statistics.bifurcation`` and ``visualization._helpers`` derive their own
#: branch-order sort-rank tables from this sequence's own order (each
#: mapping its own label-prefix spelling, e.g. "b" vs "bo", to one of these
#: five category names) rather than each keeping an independent copy of the
#: ranking itself, which could silently drift from this module's real
#: hierarchy -- and from each other.
BRANCH_ORDER_CATEGORY_SEQUENCE: tuple[str, ...] = (
    "large_arteriole",
    "arteriole",
    "capillary",
    "venule",
    "large_venule",
)

#: The branch-order labels of a run whose vessel masks are global
#: (``io.vessel_masks_are_global``): one mask of every large vessel and/or one
#: of every small vessel, with no arteriole/venule split. A vessel's size class
#: is then known but not which side of the capillary bed it lies on, so no
#: order is counted (:func:`assign_unknown_vessel_classes`). None of them
#: starts with B, Art or Ven, which the label-prefix tests read as tiers.
UNKNOWN_LARGE_VESSEL = "Unknown_Large"
UNKNOWN_SMALL_VESSEL = "Unknown_Small"
UNKNOWN_CAPILLARY = "Unknown_Cap"
UNKNOWN_VESSEL_CLASSES: tuple[str, ...] = (
    UNKNOWN_LARGE_VESSEL,
    UNKNOWN_SMALL_VESSEL,
    UNKNOWN_CAPILLARY,
)

#: Edge attribute naming the global mask a vessel lies in, ``"large"`` or
#: ``"small"`` (absent: neither, a capillary). Written by
#: ``graph.tag_vessels_in_global_masks``; read by
#: :func:`assign_unknown_vessel_classes`.
GLOBAL_MASK_CLASS = "global_mask_class"
_UNKNOWN_CLASS_BY_MASK = {"large": UNKNOWN_LARGE_VESSEL, "small": UNKNOWN_SMALL_VESSEL}

#: Edge attribute naming the large-vessel mask a vessel lies in,
#: ``"arteriole"`` or ``"venule"`` (absent: neither). Written by
#: :func:`tag_large_mask_vessel_types` when a run adds a Large_Art/Large_Ven
#: tier; read by :func:`assign_hierarchical_branch_orders`.
LARGE_MASK_VESSEL_TYPE = "large_mask_vessel_type"

#: Edge attribute naming the small-vessel mask a vessel lies in,
#: ``"arteriole"`` or ``"venule"`` (absent: neither, or the run's boundary
#: nodes did not come from the small masks). Written by
#: :func:`tag_small_mask_vessel_types` from the labelling the Boundaries
#: stage reads its arteriole/venule boundary nodes from; read by
#: :func:`assign_hierarchical_branch_orders`, which runs the Art/Ven tier on
#: along it, and by the Post processing stage, which judges a hand-edited
#: vessel beside the others.
SMALL_MASK_VESSEL_TYPE = "small_mask_vessel_type"


def _tag_mask_vessel_types(
    G: nx.MultiGraph,
    attribute: str,
    arteriole_edges: Iterable[tuple[Any, Any, Any]],
    venule_edges: Iterable[tuple[Any, Any, Any]],
) -> None:
    arteriole = {edge_id(*edge) for edge in arteriole_edges}
    venule = {edge_id(*edge) for edge in venule_edges}
    for u, v, key, data in G.edges(keys=True, data=True):
        identifier = edge_id(u, v, key)
        if identifier in arteriole:
            data[attribute] = "arteriole"
        elif identifier in venule:
            data[attribute] = "venule"
        else:
            data.pop(attribute, None)


def tag_large_mask_vessel_types(
    G: nx.MultiGraph,
    arteriole_edges: Iterable[tuple[Any, Any, Any]],
    venule_edges: Iterable[tuple[Any, Any, Any]],
) -> None:
    """Tag each ``(u, v, key)`` edge with the large mask it lies in, as
    :data:`LARGE_MASK_VESSEL_TYPE`, and clear the tag from every other edge.
    An edge in both lists is the arteriole's, as the masks' labelling decides
    a tie."""
    _tag_mask_vessel_types(G, LARGE_MASK_VESSEL_TYPE, arteriole_edges, venule_edges)


def tag_small_mask_vessel_types(
    G: nx.MultiGraph,
    arteriole_edges: Iterable[tuple[Any, Any, Any]] = (),
    venule_edges: Iterable[tuple[Any, Any, Any]] = (),
) -> None:
    """Tag each ``(u, v, key)`` edge with the small mask it lies in, as
    :data:`SMALL_MASK_VESSEL_TYPE`, and clear the tag from every other edge
    (from all of them, given no edges: a run whose boundary nodes did not
    come from the small masks). An edge in both lists is the arteriole's."""
    _tag_mask_vessel_types(G, SMALL_MASK_VESSEL_TYPE, arteriole_edges, venule_edges)


_TIER_LABEL = re.compile(r"^(.*?)(\d+)\s*$")

#: Label prefix -> (tier along the flow path, side of the capillary bed).
#: Ven and Large_Ven are numbered from the outlet (graph.branch_order), so
#: along the path their numbers run down: Ven2 comes before Ven1.
_FLOW_PATH_TIERS = {
    "large_art": (0, -1),
    "art": (1, -1),
    "b": (2, 0),
    "bo": (2, 0),
    "ven": (3, 1),
    "large_ven": (4, 1),
}


def _tier_and_number(label: Any) -> tuple[str, int] | None:
    """``("art", 3)`` for ``Art3``; None for a label with no trailing number."""
    match = _TIER_LABEL.match(str(label or "").strip())
    if match is None:
        return None
    return match.group(1).lower(), int(match.group(2))


def flow_path_key(label: Any) -> tuple:
    """Sort key putting branch-order labels in the order blood passes them.

    Any other label keeps the order the plots use for it (after every
    branch-order tier; labels with no number last), so this also serves text
    columns that are not branch orders.
    """
    parsed = _tier_and_number(label)
    if parsed is None:
        return (99, 0, str(label))
    prefix, number = parsed
    tier = _FLOW_PATH_TIERS.get(prefix)
    if tier is None:
        return (len(_FLOW_PATH_TIERS), number, str(label))
    position, side = tier
    return (position, -number if side > 0 else number, str(label))


def branch_order_signed_values(labels: Iterable[Any]) -> np.ndarray:
    """Generations from the capillary bed, per label, for *labels* together.

    Capillaries are 0. Upstream of them, the arteriole that feeds them is
    -1 and the count grows through Art to the most upstream Large_Art (so
    ``Art`` n is ``-(Art_max - n + 1)``, and a Large_Art lies beyond every
    Art). Downstream, the venule that drains them is +1 and the count grows
    through Ven to Large_Ven1 at the outlet. The maxima are the network's
    own; a label of no known tier is NaN.
    """
    labels = [str(label) for label in labels]
    most: dict[str, int] = {}
    for label in set(labels):
        parsed = _tier_and_number(label)
        if parsed is not None:
            most[parsed[0]] = max(most.get(parsed[0], 0), parsed[1])
    art, ven = most.get("art", 0), most.get("ven", 0)

    def signed(label: str) -> float:
        parsed = _tier_and_number(label)
        if parsed is None:
            return float("nan")
        prefix, number = parsed
        if prefix in {"b", "bo"}:
            return 0.0
        if prefix == "art":
            return -float(art - number + 1)
        if prefix == "large_art":
            return -float(art + most["large_art"] - number + 1)
        if prefix == "ven":
            return float(ven - number + 1)
        if prefix == "large_ven":
            return float(ven + most["large_ven"] - number + 1)
        return float("nan")

    return np.asarray([signed(label) for label in labels], dtype=float)


def branch_order_rank_values(labels: Iterable[Any]) -> np.ndarray:
    """Each label's place along the flow path, 1..N over the labels present."""
    labels = [str(label) for label in labels]
    present = sorted({label for label in labels if _tier_and_number(label)}, key=flow_path_key)
    rank = {label: float(index) for index, label in enumerate(present, start=1)}
    return np.asarray([rank.get(label, np.nan) for label in labels], dtype=float)


class MissingSmallVesselAssignmentWarning(UserWarning):
    """Strict branch-order mode ran without small arteriole/venule terminals.

    Capillary ``B*`` orders are still assigned from the inlet nodes; Art*/Ven*
    labelling is skipped because no small-vessel (or manual A/V) terminals
    were provided.
    """


class MissingLargeVesselTierWarning(UserWarning):
    """A Large_Art or Large_Ven tier was asked for, but no vessel tagged in
    that large mask (:data:`LARGE_MASK_VESSEL_TYPE`) reaches an inlet or
    outlet, so the tier is empty and Art or Ven is counted as without one."""


def _compute_node_distances(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    stop_nodes: set[int] | None = None,
) -> dict[int, int]:
    """Compute unweighted BFS node distances with optional stop nodes.

    A zero-resistance bridge is no step: it is the radius of the thick vessel
    a vessel opens into, so the vessel is one order from the thick vessel,
    not two.
    """
    stop_nodes = stop_nodes or set()
    node_distances: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()

    for inlet_node in inlet_nodes:
        if inlet_node in G.nodes():
            if inlet_node not in node_distances:
                node_distances[inlet_node] = 0
                queue.append((inlet_node, 0))
        else:
            logger.warning("Inlet node %s not found in graph", inlet_node)

    while queue:
        current_node, distance = queue.popleft()
        if (
            current_node in node_distances
            and node_distances[current_node] < distance
        ):
            continue
        if current_node in stop_nodes:
            continue
        for neighbor, between in G.adj[current_node].items():
            datas = between.values() if G.is_multigraph() else (between,)
            step = 0 if all(d.get(IS_ZERO_RESISTANCE) for d in datas) else 1
            new_distance = distance + step
            if (
                neighbor not in node_distances
                or node_distances[neighbor] > new_distance
            ):
                node_distances[neighbor] = new_distance
                if step:
                    queue.append((neighbor, new_distance))
                else:
                    queue.appendleft((neighbor, new_distance))
    return node_distances


def assign_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    prefix: str = "B",
    stop_nodes: set[int] | None = None,
    excluded_edges: set[tuple[int, int, int]] | None = None,
    included_edges: set[tuple[int, int, int]] | None = None,
) -> dict:
    """Assign branch-order labels to edges from BFS distance to inlet nodes.

    A zero-resistance bridge gets none (and loses any it had): it is not a
    vessel of any order. Nor has an edge the search from the inlets does not
    reach -- a piece of the network cut off from them -- which loses any it
    had too: a hand edit that cuts a piece off used to leave its vessels with
    the orders of the network they were cut from (an edge left out by
    *included_edges* or *excluded_edges* is another pass's, and keeps its).
    """
    node_distances = _compute_node_distances(
        G,
        inlet_nodes,
        stop_nodes=stop_nodes,
    )
    excluded_edges = excluded_edges or set()
    included_edges = included_edges or set()

    results = {
        "edges_assigned": 0,
        "edges_skipped": 0,
        "branch_order_counts": defaultdict(int),
        "unreachable_edges": [],
        "excluded_edges": 0,
        "bridges": 0,
    }

    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(IS_ZERO_RESISTANCE):
            data.pop("branch_order", None)
            results["bridges"] += 1
            continue
        edge_identifier = edge_id(u, v, key)
        if included_edges and edge_identifier not in included_edges:
            results["edges_skipped"] += 1
            continue
        if edge_identifier in excluded_edges:
            results["excluded_edges"] += 1
            results["edges_skipped"] += 1
            continue
        u_dist = node_distances.get(u, float("inf"))
        v_dist = node_distances.get(v, float("inf"))
        if u_dist == float("inf") and v_dist == float("inf"):
            data.pop("branch_order", None)
            results["unreachable_edges"].append((u, v, key))
            results["edges_skipped"] += 1
            continue
        edge_distance = min(u_dist, v_dist) + 1
        branch_order = f"{prefix}{edge_distance:02d}" if prefix == "B" else f"{prefix}{edge_distance}"
        G[u][v][key]["branch_order"] = branch_order
        results["edges_assigned"] += 1
        results["branch_order_counts"][branch_order] += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Edge (%s, %s, %s): %s (u_dist=%s, v_dist=%s)",
                u,
                v,
                key,
                branch_order,
                u_dist,
                v_dist,
            )
    return results


def _large_vessel_tier(
    G: nx.MultiGraph, roots: Iterable[Any], vessel: str
) -> set[tuple[Any, Any, Any]]:
    """The edges tagged *vessel* (:data:`LARGE_MASK_VESSEL_TYPE`) that a walk
    from *roots* along tagged edges alone reaches: the large vessel from the
    inlet or outlet to where its mask ends, past every vessel that leaves it
    on the way."""
    tagged: dict[Any, list[tuple[Any, tuple[Any, Any, Any]]]] = defaultdict(list)
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(LARGE_MASK_VESSEL_TYPE) == vessel:
            identifier = edge_id(u, v, key)
            tagged[u].append((v, identifier))
            tagged[v].append((u, identifier))
    seen = {root for root in roots if root in tagged}
    queue = deque(seen)
    tier: set[tuple[Any, Any, Any]] = set()
    while queue:
        node = queue.popleft()
        for neighbour, identifier in tagged[node]:
            tier.add(identifier)
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return tier


def _edge_ends(edges: Iterable[tuple[Any, Any, Any]]) -> set[Any]:
    return {node for u, v, _key in edges for node in (u, v)}


def _hand_off_nodes(G: nx.MultiGraph, tier: set[tuple[Any, Any, Any]]) -> list[Any]:
    """The nodes of a Large tier where a vessel outside it leaves it."""
    return sort_nodes(
        node
        for node in _edge_ends(tier)
        if any(edge_id(node, other, key) not in tier for _n, other, key in G.edges(node, keys=True))
    )


def _closed_side_branches(
    G: nx.MultiGraph,
    tier_nodes: set[Any],
    stop_nodes: set[Any],
    far_side_nodes: set[Any],
    excluded_edges: set[tuple[Any, Any, Any]],
) -> set[tuple[Any, Any, Any]]:
    """The edges of the side branches leaving a Large tier that close off at
    *stop_nodes* (the arteriole or venule boundary nodes) before reaching
    *far_side_nodes* (the other side of the capillary bed).

    The edges outside *excluded_edges* fall into pieces joined at every node
    that is neither a tier node nor a stop node. A piece meeting the tier is
    a side branch: a small vessel, kept, when it touches no far-side node; a
    capillary leaving the large vessel, left out, when it joins the bed that
    drains to the far side.
    """
    parent: dict[tuple[Any, Any, Any], tuple[Any, Any, Any]] = {}

    def root(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    incident: dict[Any, list[tuple[Any, Any, Any]]] = defaultdict(list)
    for u, v, key in G.edges(keys=True):
        identifier = edge_id(u, v, key)
        if identifier in excluded_edges:
            continue
        parent[identifier] = identifier
        incident[u].append(identifier)
        if v != u:
            incident[v].append(identifier)
    joined_at = tier_nodes | stop_nodes
    for node, identifiers in incident.items():
        if node in joined_at:
            continue
        first = root(identifiers[0])
        for identifier in identifiers[1:]:
            parent[root(identifier)] = first
    pieces: dict[tuple[Any, Any, Any], list[tuple[Any, Any, Any]]] = defaultdict(list)
    for identifier in parent:
        pieces[root(identifier)].append(identifier)
    kept: set[tuple[Any, Any, Any]] = set()
    for piece in pieces.values():
        ends = _edge_ends(piece)
        if ends & tier_nodes and not ends & far_side_nodes:
            kept.update(piece)
    return kept


def _run_on_along_small_mask(
    G: nx.MultiGraph,
    edges: set[tuple[Any, Any, Any]],
    from_nodes: set[Any],
    *,
    vessel: str,
    excluded_edges: set[tuple[Any, Any, Any]],
) -> set[tuple[Any, Any, Any]]:
    """The vessels tagged *vessel* in the small mask
    (:data:`SMALL_MASK_VESSEL_TYPE`) outside *edges*, an Art or Ven tier, and
    *excluded_edges* that a walk from *from_nodes* along such vessels alone
    reaches.

    The small masks' labelling makes every junction where a capillary leaves
    a small vessel one of its boundary nodes, so the search that found
    *edges* stopped at the first of them; the tier runs on along the vessel's
    mask past them, but never across a capillary. With no tags (boundary
    nodes picked by hand) there is nothing to add.
    """
    tagged: dict[Any, list[tuple[Any, tuple[Any, Any, Any]]]] = defaultdict(list)
    for u, v, key, data in G.edges(keys=True, data=True):
        identifier = edge_id(u, v, key)
        if (
            data.get(SMALL_MASK_VESSEL_TYPE) == vessel
            and identifier not in edges
            and identifier not in excluded_edges
        ):
            tagged[u].append((v, identifier))
            tagged[v].append((u, identifier))
    seen = {node for node in from_nodes if node in tagged}
    queue = deque(seen)
    added: set[tuple[Any, Any, Any]] = set()
    while queue:
        node = queue.popleft()
        for other, identifier in tagged[node]:
            added.add(identifier)
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return added


def _tier_from_seeds(
    G: nx.MultiGraph,
    seeds: list[Any],
    *,
    prefix: str,
    vessel: str,
    stop_nodes: set[Any],
    excluded_edges: set[tuple[Any, Any, Any]],
) -> tuple[dict, set[tuple[Any, Any, Any]]]:
    """The Art or Ven tier with no Large tier on its side: every edge both
    of whose ends a search from *seeds* reaches before *stop_nodes*, run on
    along the small mask (:func:`_run_on_along_small_mask`)."""
    reached = set(_compute_node_distances(G, seeds, stop_nodes=stop_nodes))
    edges = {
        edge_id(u, v, key)
        for u, v, key in G.edges(keys=True)
        if u in reached and v in reached
    }
    added = _run_on_along_small_mask(
        G,
        edges,
        set(seeds) | _edge_ends(edges - excluded_edges),
        vessel=vessel,
        excluded_edges=excluded_edges,
    )
    if added:
        # Counted along the tier's own vessels, through the boundary nodes
        # it now runs past.
        edges |= added
        view = G.edge_subgraph(edges)
        results = assign_branch_orders(
            view,
            [node for node in seeds if node in view],
            prefix=prefix,
            excluded_edges=excluded_edges,
        )
    else:
        results = assign_branch_orders(
            G,
            seeds,
            prefix=prefix,
            stop_nodes=stop_nodes,
            excluded_edges=excluded_edges,
            included_edges=edges,
        )
    return results, edges


def _tier_from_large_vessel(
    G: nx.MultiGraph,
    hand_offs: list[Any],
    tier: set[tuple[Any, Any, Any]],
    *,
    prefix: str,
    vessel: str,
    stop_nodes: set[Any],
    far_side_nodes: set[Any],
    excluded_edges: set[tuple[Any, Any, Any]],
) -> tuple[dict, set[tuple[Any, Any, Any]]]:
    """The Art or Ven tier beside a Large tier: the side branches that close
    off at *stop_nodes* (:func:`_closed_side_branches`), run on along the
    small mask (:func:`_run_on_along_small_mask`), counted from the hand-off
    nodes they leave the large vessel at."""
    edges = _closed_side_branches(
        G, _edge_ends(tier), stop_nodes, far_side_nodes, excluded_edges
    )
    added = _run_on_along_small_mask(
        G, edges, set(hand_offs) | _edge_ends(edges), vessel=vessel, excluded_edges=excluded_edges
    )
    edges |= added
    view = G.edge_subgraph(edges)
    # A hand-off is the start of its side branches even when the small
    # vessel's own labelling also made it a boundary node.
    starts = [node for node in hand_offs if node in view]
    # Run on along the small mask, the tier is counted through the boundary
    # nodes it now runs past.
    results = assign_branch_orders(
        view, starts, prefix=prefix, stop_nodes=set() if added else stop_nodes - set(starts)
    )
    return results, edges


def _large_tier_or_warn(
    G: nx.MultiGraph, roots: list[Any], vessel: str, boundary_nodes: list[Any]
) -> set[tuple[Any, Any, Any]]:
    if not boundary_nodes:
        return set()
    tier = _large_vessel_tier(G, roots, vessel)
    if not tier:
        name, ends, small = (
            ("Large_Art", "an inlet", "Art")
            if vessel == "arteriole"
            else ("Large_Ven", "an outlet", "Ven")
        )
        warnings.warn(
            f"{len(boundary_nodes)} {name} boundary node(s) given, but no vessel tagged "
            f"in the large {vessel} mask ({LARGE_MASK_VESSEL_TYPE!r}) reaches {ends}: "
            f"there is no {name} tier, and {small} is counted as without one.",
            MissingLargeVesselTierWarning,
            stacklevel=3,
        )
    return tier


def assign_hierarchical_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    outlet_nodes: list[int],
    arteriole_boundary_nodes: list[int],
    venule_boundary_nodes: list[int],
    *,
    large_arteriole_boundary_nodes: list[int] | None = None,
    large_venule_boundary_nodes: list[int] | None = None,
) -> dict:
    """Assign Art*/Ven*/B* branch orders using arteriole and venule boundaries.

    ``large_arteriole_boundary_nodes``/``large_venule_boundary_nodes``, when
    non-empty, add a ``Large_Art``/``Large_Ven`` tier "outside" the existing
    ``Art``/``Ven`` tier: a large vessel kept in the network (see
    ``graph.large_vessel_network``), from the true image-edge inlet/outlet to
    where the large-vessel mask's own coverage ends. The tier is the edges
    tagged as lying in that mask (:data:`LARGE_MASK_VESSEL_TYPE`, see
    :func:`tag_large_mask_vessel_types`) that connect to an inlet/outlet
    through tagged edges alone, numbered along them -- so it runs on past the
    junctions where vessels leave it. Those junctions are the hand-off nodes
    the ``Art``/``Ven`` tier starts from, and it takes only the side branches
    that close off at arteriole/venule boundary nodes without reaching the
    other side of the bed (the outlets, the venule boundary nodes and the
    Large_Ven tier for Art; their arterial counterparts for Ven). A capillary
    leaving a large vessel directly is a capillary: the ``B`` orders count
    from the hand-off nodes as well as from the arteriole boundary nodes.

    With boundary nodes from the small masks, every junction where a
    capillary leaves a small vessel is one of them, not only where its mask
    ends. The vessels lying in those masks are tagged
    (:data:`SMALL_MASK_VESSEL_TYPE`, see :func:`tag_small_mask_vessel_types`),
    and the ``Art``/``Ven`` tier, with or without a Large tier, runs on from
    where its search stopped along the vessels tagged in its small mask: past
    the capillaries leaving it, which the ``B`` orders count from those
    junctions, but never across one -- a piece of the small mask joined to
    the tier only through capillaries is not in it. Boundary nodes picked by
    hand come with no tags: the tiers end at them.

    A tagged mask fragment that does not connect to an inlet/outlet is not
    in the tier. A tier asked for that comes out empty warns
    (:class:`MissingLargeVesselTierWarning`) and leaves Art or Ven as without
    one. Leaving both lists empty (the default) reproduces the behaviour
    without a large tier exactly, whatever tags the graph carries.
    """
    arteriole_boundary_set = set(arteriole_boundary_nodes)
    venule_boundary_set = set(venule_boundary_nodes)
    large_arteriole_edges = _large_tier_or_warn(
        G, inlet_nodes, "arteriole", list(large_arteriole_boundary_nodes or [])
    )
    large_venule_edges = _large_tier_or_warn(
        G, outlet_nodes, "venule", list(large_venule_boundary_nodes or [])
    )
    large_edges = large_arteriole_edges | large_venule_edges

    large_arteriole_results = None
    arteriole_hand_offs: list[Any] = []
    if large_arteriole_edges:
        view = G.edge_subgraph(large_arteriole_edges)
        large_arteriole_results = assign_branch_orders(
            view, [node for node in inlet_nodes if node in view], prefix="Large_Art"
        )
        arteriole_hand_offs = _hand_off_nodes(G, large_arteriole_edges)
    large_venule_results = None
    venule_hand_offs: list[Any] = []
    if large_venule_edges:
        view = G.edge_subgraph(large_venule_edges)
        large_venule_results = assign_branch_orders(
            view, [node for node in outlet_nodes if node in view], prefix="Large_Ven"
        )
        venule_hand_offs = _hand_off_nodes(G, large_venule_edges)

    if large_arteriole_edges:
        arteriole_results, arteriole_edges = _tier_from_large_vessel(
            G,
            arteriole_hand_offs,
            large_arteriole_edges,
            prefix="Art",
            vessel="arteriole",
            stop_nodes=arteriole_boundary_set,
            far_side_nodes=(
                set(outlet_nodes) | venule_boundary_set | _edge_ends(large_venule_edges)
            ),
            excluded_edges=large_edges,
        )
    else:
        arteriole_results, arteriole_edges = _tier_from_seeds(
            G,
            inlet_nodes,
            prefix="Art",
            vessel="arteriole",
            stop_nodes=arteriole_boundary_set,
            excluded_edges=large_edges,
        )

    if large_venule_edges:
        venule_results, venule_edges = _tier_from_large_vessel(
            G,
            venule_hand_offs,
            large_venule_edges,
            prefix="Ven",
            vessel="venule",
            stop_nodes=venule_boundary_set,
            far_side_nodes=(
                set(inlet_nodes) | arteriole_boundary_set | _edge_ends(large_arteriole_edges)
            ),
            excluded_edges=large_edges | arteriole_edges,
        )
    else:
        venule_results, venule_edges = _tier_from_seeds(
            G,
            outlet_nodes,
            prefix="Ven",
            vessel="venule",
            stop_nodes=venule_boundary_set,
            excluded_edges=arteriole_edges | large_edges,
        )

    capillary_start_nodes = list(
        arteriole_boundary_nodes if arteriole_boundary_nodes else inlet_nodes
    )
    capillary_start_nodes += [
        node for node in arteriole_hand_offs if node not in set(capillary_start_nodes)
    ]
    capillary_excluded_edges = arteriole_edges | venule_edges | large_edges
    capillary_results = assign_branch_orders(
        G,
        capillary_start_nodes,
        prefix="B",
        excluded_edges=capillary_excluded_edges,
    )

    return {
        "arteriole": arteriole_results,
        "venule": venule_results,
        "capillary": capillary_results,
        "large_arteriole": large_arteriole_results,
        "large_venule": large_venule_results,
        "arteriole_edge_count": len(arteriole_edges),
        "venule_edge_count": len(venule_edges),
        "large_arteriole_edge_count": len(large_arteriole_edges),
        "large_venule_edge_count": len(large_venule_edges),
        "excluded_capillary_edge_count": len(capillary_excluded_edges),
    }


def assign_large_tier_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[Any],
    outlet_nodes: list[Any],
    *,
    large_arteriole_boundary_nodes: list[Any] | None = None,
    large_venule_boundary_nodes: list[Any] | None = None,
) -> dict | None:
    """Large_Art/Large_Ven and B orders, with no small arterioles or venules.

    For a run given large-vessel masks alone: no arteriole or venule boundary
    nodes, so there is no Art/Ven tier, but the large vessels are known. The
    Large tiers are read as :func:`assign_hierarchical_branch_orders` reads
    them, and every other vessel is a capillary, its ``B`` order counted from
    the inlets and from where it leaves the Large_Art tier. Hierarchical mode
    used to be the only way to a Large tier, so such a run fell back to ``B``
    orders from the inlets and its large arteriole was B01.

    Returns None, writing nothing, when neither tier has a vessel.
    """
    large_arteriole_edges = _large_tier_or_warn(
        G, inlet_nodes, "arteriole", list(large_arteriole_boundary_nodes or [])
    )
    large_venule_edges = _large_tier_or_warn(
        G, outlet_nodes, "venule", list(large_venule_boundary_nodes or [])
    )
    if not large_arteriole_edges and not large_venule_edges:
        return None
    large_edges = large_arteriole_edges | large_venule_edges
    results: dict[str, Any] = {"large_arteriole": None, "large_venule": None}
    arteriole_hand_offs: list[Any] = []
    for name, edges, roots, prefix in (
        ("large_arteriole", large_arteriole_edges, inlet_nodes, "Large_Art"),
        ("large_venule", large_venule_edges, outlet_nodes, "Large_Ven"),
    ):
        if not edges:
            continue
        view = G.edge_subgraph(edges)
        results[name] = assign_branch_orders(
            view, [node for node in roots if node in view], prefix=prefix
        )
        if prefix == "Large_Art":
            arteriole_hand_offs = _hand_off_nodes(G, edges)
    starts = list(inlet_nodes) + [node for node in arteriole_hand_offs if node not in set(inlet_nodes)]
    results["capillary"] = assign_branch_orders(G, starts, prefix="B", excluded_edges=large_edges)
    results["large_arteriole_edge_count"] = len(large_arteriole_edges)
    results["large_venule_edge_count"] = len(large_venule_edges)
    return results


def assign_unknown_vessel_classes(
    G: nx.MultiGraph, *, post_assign_callback: PostAssignCallback | None = None
) -> dict[str, Any]:
    """Label every vessel with its unknown class instead of a branch order.

    For a run with global vessel masks: a vessel tagged
    :data:`GLOBAL_MASK_CLASS` ``"large"`` is :data:`UNKNOWN_LARGE_VESSEL`,
    ``"small"`` :data:`UNKNOWN_SMALL_VESSEL`, and every other one
    :data:`UNKNOWN_CAPILLARY`, whether or not an inlet reaches it. A
    zero-resistance bridge gets none, as in :func:`assign_branch_orders`.
    Returns ``{"mode": "unknown", "counts": {label: edges}, "bridges": n}``.
    """
    counts: dict[str, int] = defaultdict(int)
    bridges = 0
    for _u, _v, data in G.edges(data=True):
        if data.get(IS_ZERO_RESISTANCE):
            data.pop("branch_order", None)
            bridges += 1
            continue
        label = _UNKNOWN_CLASS_BY_MASK.get(data.get(GLOBAL_MASK_CLASS), UNKNOWN_CAPILLARY)
        data["branch_order"] = label
        counts[label] += 1
    if post_assign_callback is not None:
        post_assign_callback(G)
    return {"mode": "unknown", "counts": dict(counts), "bridges": bridges}


def uses_hierarchical_branch_orders(
    outlet_nodes: Iterable[Any] | None,
    arteriole_boundary_nodes: Iterable[Any] | None,
    venule_boundary_nodes: Iterable[Any] | None,
) -> bool:
    """Whether :func:`assign_vessel_branch_orders` assigns Art*/Ven*/B*
    tiers (all three node sets non-empty) rather than B* orders from the
    inlets alone."""
    return bool(
        list(arteriole_boundary_nodes or [])
        and list(venule_boundary_nodes or [])
        and list(outlet_nodes or [])
    )


def large_tier_capillary_hand_offs(G: nx.MultiGraph) -> tuple[list[Any], list[Any]]:
    """The nodes where a capillary leaves the Large_Art tier, and where one
    joins the Large_Ven tier, read from the edges' branch orders: a node with
    both a ``Large_Art`` (``Large_Ven``) and a ``B`` vessel.

    The first are those hand-off nodes :func:`assign_hierarchical_branch_orders`
    counts B orders from that a capillary leaves; both are empty without a
    Large tier. Read on a graph with its bridges contracted (as statistics
    see it), a capillary opening into the large vessel through a bridge meets
    it at the bridge's centreline end.
    """
    tiers_at: dict[Any, set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        parsed = _tier_and_number(data.get("branch_order"))
        if parsed is not None:
            tiers_at[u].add(parsed[0])
            tiers_at[v].add(parsed[0])

    def meeting_capillaries(tier: str) -> list[Any]:
        return sort_nodes(
            node for node, tiers in tiers_at.items() if tier in tiers and tiers & {"b", "bo"}
        )

    return meeting_capillaries("large_art"), meeting_capillaries("large_ven")


def capillary_start_nodes(
    inlet_nodes: Iterable[Any] | None,
    outlet_nodes: Iterable[Any] | None,
    arteriole_boundary_nodes: Iterable[Any] | None,
    venule_boundary_nodes: Iterable[Any] | None,
    *,
    graph: nx.MultiGraph | None = None,
) -> tuple[list[Any], list[Any]]:
    """The nodes :func:`assign_vessel_branch_orders` counts capillary B
    orders from, and the venous nodes on the bed's far side: the arteriole
    and venule boundary nodes in hierarchical mode, else the inlets and
    outlets.

    Given the labelled *graph*, hierarchical mode adds where capillaries
    leave the Large_Art tier and join the Large_Ven tier
    (:func:`large_tier_capillary_hand_offs`): a capillary leaving a large
    arteriole is counted from it, and one joining the large vein drains
    there. Without a Large tier that adds nothing. Without hierarchical mode
    the feeds are the inlets and the drains the outlets, with the same
    hand-offs added (:func:`assign_large_tier_branch_orders`' tiers).
    """
    if uses_hierarchical_branch_orders(outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes):
        feeds, drains = list(arteriole_boundary_nodes), list(venule_boundary_nodes)
    else:
        feeds, drains = list(inlet_nodes or []), list(outlet_nodes or [])
    if graph is not None:
        arterial, venous = large_tier_capillary_hand_offs(graph)
        known_feeds, known_drains = set(feeds), set(drains)
        feeds += [node for node in arterial if node not in known_feeds]
        drains += [node for node in venous if node not in known_drains]
    return feeds, drains


def assign_vessel_branch_orders(
    G: nx.MultiGraph,
    inlet_nodes: list[int],
    *,
    outlet_nodes: list[int] | None = None,
    arteriole_boundary_nodes: list[int] | None = None,
    venule_boundary_nodes: list[int] | None = None,
    large_arteriole_boundary_nodes: list[int] | None = None,
    large_venule_boundary_nodes: list[int] | None = None,
    strict_hierarchical: bool = False,
    expects_hierarchical: bool = False,
    post_assign_callback: PostAssignCallback | None = None,
) -> dict[str, Any]:
    """
    Assign branch orders on ``G`` using capillary-only or hierarchical rules.

    When arteriole boundary nodes, venule boundary nodes, and outlet nodes are
    all non-empty, uses :func:`assign_hierarchical_branch_orders`; otherwise
    assigns capillary ``B*`` orders from ``inlet_nodes`` only.

    Parameters
    ----------
    G
        Vascular graph (modified in place).
    inlet_nodes
        Inlet / arteriole-side seed nodes for capillary or hierarchical assignment.
    outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes
        Optional node sets for hierarchical assignment.
    large_arteriole_boundary_nodes, large_venule_boundary_nodes
        Optional node sets adding a ``Large_Art``/``Large_Ven`` tier outside
        the ``Art``/``Ven`` tier, read from the vessels tagged
        :data:`LARGE_MASK_VESSEL_TYPE` (see :func:`assign_hierarchical_branch_orders`);
        without arteriole/venule boundary nodes, the Large tiers and ``B``
        orders alone (:func:`assign_large_tier_branch_orders`).
    strict_hierarchical
        When hierarchical prerequisites are missing: raise if small-vessel
        (or manual A/V) assignment was expected, otherwise warn and fall back
        to capillary ``B*`` orders.
    expects_hierarchical
        Set when small-vessel masks or manual arteriole/venule methods were
        supposed to supply hierarchical terminals.
    post_assign_callback
        Optional ``callback(G)`` after orders are written (e.g. for plotting).

    Returns
    -------
    dict
        Summary with ``mode`` of ``"hierarchical"``, ``"large_tier"``, ``"capillary"``,
        or ``"skipped"``.
    """
    if not inlet_nodes:
        return {"mode": "skipped", "reason": "no_inlet_nodes"}

    outlet_nodes = list(outlet_nodes or [])
    arteriole_boundary_nodes = list(arteriole_boundary_nodes or [])
    venule_boundary_nodes = list(venule_boundary_nodes or [])
    large_arteriole_boundary_nodes = list(large_arteriole_boundary_nodes or [])
    large_venule_boundary_nodes = list(large_venule_boundary_nodes or [])

    use_hierarchical = uses_hierarchical_branch_orders(
        outlet_nodes, arteriole_boundary_nodes, venule_boundary_nodes
    )
    large_tier_only = not use_hierarchical and bool(
        large_arteriole_boundary_nodes or large_venule_boundary_nodes
    )
    # A run given large masks alone asks for no Art/Ven tier: its Large tier
    # is all it expects.
    if strict_hierarchical and not use_hierarchical and (expects_hierarchical or not large_tier_only):
        if expects_hierarchical:
            raise ValueError(
                "Strict branch-order assignment is enabled, but hierarchical "
                "assignment prerequisites are missing. "
                f"Need non-empty outlet_nodes, arteriole_boundary_nodes, and "
                f"venule_boundary_nodes. Got counts: "
                f"outlet_nodes={len(outlet_nodes)}, "
                f"arteriole_boundary_nodes={len(arteriole_boundary_nodes)}, "
                f"venule_boundary_nodes={len(venule_boundary_nodes)}. "
                "Fix mask inputs/thresholds or disable strict_hierarchical."
            )
        warnings.warn(
            "Strict branch-order assignment is enabled, but small arterioles "
            "and venules were not assigned (no arteriole/venule boundary "
            "nodes). Assigning capillary branch orders (B01, B02, …) from "
            "inlet nodes only.",
            MissingSmallVesselAssignmentWarning,
            stacklevel=2,
        )
    elif expects_hierarchical and not use_hierarchical:
        # Without strict assignment this fell back in silence: small masks
        # finding no venule boundary node left a run with no Art tier either.
        empty = [
            name
            for name, nodes in (
                ("outlet_nodes", outlet_nodes),
                ("arteriole_boundary_nodes", arteriole_boundary_nodes),
                ("venule_boundary_nodes", venule_boundary_nodes),
            )
            if not nodes
        ]
        warnings.warn(
            "Arteriole and venule boundary nodes were expected (small-vessel masks or "
            f"a manual pick), but {', '.join(empty)} came out empty, so there is no "
            "Art*/Ven* tier on either side: "
            + (
                "Large_Art/Large_Ven along the large masks and capillary B* orders beside them."
                if large_tier_only
                else "capillary B* orders from the inlets only."
            ),
            MissingSmallVesselAssignmentWarning,
            stacklevel=2,
        )

    if use_hierarchical:
        branch_results = assign_hierarchical_branch_orders(
            G,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
            arteriole_boundary_nodes=arteriole_boundary_nodes,
            venule_boundary_nodes=venule_boundary_nodes,
            large_arteriole_boundary_nodes=large_arteriole_boundary_nodes,
            large_venule_boundary_nodes=large_venule_boundary_nodes,
        )
        summary: dict[str, Any] = {"mode": "hierarchical", **branch_results}
    else:
        large_results = (
            assign_large_tier_branch_orders(
                G,
                inlet_nodes,
                outlet_nodes,
                large_arteriole_boundary_nodes=large_arteriole_boundary_nodes,
                large_venule_boundary_nodes=large_venule_boundary_nodes,
            )
            if large_tier_only
            else None
        )
        if large_results is not None:
            summary = {"mode": "large_tier", **large_results}
        else:
            capillary_results = assign_branch_orders(G, inlet_nodes)
            summary = {"mode": "capillary", "capillary": capillary_results}

    if post_assign_callback is not None:
        post_assign_callback(G)

    return summary
