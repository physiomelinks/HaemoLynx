"""Prune short terminal stubs from vascular graph."""
import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from ._helpers import calculate_edge_length, densify_polyline, edge_sample_points

logger = logging.getLogger(__name__)

#: How far back from its tip a stub's outward direction is read.
STUB_TANGENT_LENGTH_UM = 3.0

#: A dead end that never gets further than this many of its own radii (read
#: at its tip) outside the lumen of another centreline is a hair along that
#: vessel's wall or a bump on it, whatever its length -- the tip-to-junction
#: test misses one that runs along the wall.
STUB_PROTRUSION_RADII = 1.0


def _chain(G: nx.MultiGraph, tip: Any) -> Tuple[list, list]:
    """``(nodes, edges)`` from a dead end's *tip* through degree-2 nodes to
    the first node that is not one: the junction it hangs from (or, for an
    isolated chain, its other end). Edges are ``(a, b, key)``, *a* nearer
    the tip."""
    nodes, edges = [tip], []
    current, came = tip, None
    while True:
        incident = (
            G.edges(current, keys=True) if G.is_multigraph()
            else ((current, other, None) for _node, other in G.edges(current))
        )
        onward = [
            (other, key) for _node, other, key in incident
            if other != current and (frozenset((current, other)), key) != came
        ]
        if len(onward) != 1:
            break
        other, key = onward[0]
        if other in nodes:
            break
        edges.append((current, other, key))
        came = (frozenset((current, other)), key)
        nodes.append(other)
        current = other
        if G.degree[other] != 2:
            break
    return nodes, edges


def _edge_data(G, edge) -> dict:
    a, b, key = edge
    return G.edges[a, b, key] if G.is_multigraph() else G.edges[a, b]


def _chain_path(G: nx.MultiGraph, edges: list) -> np.ndarray:
    """A chain's points from its junction to its tip, every half micron."""
    pieces = []
    for a, b, key in reversed(edges):
        node_pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
        piece = edge_sample_points(b, a, _edge_data(G, (a, b, key)), node_pos)
        pieces.append(piece if not pieces else piece[1:])
    return densify_polyline(np.vstack(pieces), max_step_um=0.5)


def _centreline_index(G: nx.MultiGraph, lumen_radius_at, memo: Optional[dict] = None):
    """Every edge's centreline points with their lumen radii, for judging a
    stub against the other vessels: ``(tree, radii, owner per point, owner
    id per edge, points)``, or ``None`` for a graph without edges.

    *memo*, a dict a caller passes to every call on one graph and one mask,
    keeps each edge's points and radii under the signature of its path
    (``diagnostics._edge_signature``): an edge unchanged since the last
    call reads no lumen radius again. The radii were the cost -- the
    clean-up prunes every round, and a round changes a few edges.
    """
    from scipy.spatial import cKDTree

    from .diagnostics import _edge_signature

    remembered = memo if memo is not None else {}
    kept: dict = {}
    points, radii, owners, ids = [], [], [], {}
    every = (
        G.edges(keys=True, data=True) if G.is_multigraph()
        else ((u, v, None, data) for u, v, data in G.edges(data=True))
    )
    for u, v, key, data in every:
        if u == v:
            continue
        node_pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
        signature = _edge_signature((u, v, key), data, node_pos[u], node_pos[v]) if memo is not None else None
        if signature in remembered:
            piece, piece_radii = remembered[signature]
        else:
            piece = densify_polyline(edge_sample_points(u, v, data, node_pos), max_step_um=1.0)
            piece_radii = np.asarray(lumen_radius_at(piece), dtype=float).reshape(-1)
        if signature is not None:
            kept[signature] = (piece, piece_radii)
        owner = ids.setdefault((frozenset((u, v)), key), len(ids))
        points.append(piece)
        radii.append(piece_radii)
        owners.extend([owner] * len(piece))
    if memo is not None:
        memo.clear()
        memo.update(kept)
    if not points:
        return None
    points = np.vstack(points)
    return cKDTree(points), np.concatenate(radii), np.asarray(owners, dtype=np.intp), ids, points


def _index_without(index, dead_owners: set):
    """*index* with the points of the edges *dead_owners* own left out: the
    tree is rebuilt from the points kept, no lumen radius read again -- the
    radii were the cost (built afresh every pass, the index was most of the
    prune's time, and within one call only stubs go)."""
    from scipy.spatial import cKDTree

    if index is None or not dead_owners:
        return index
    _tree, radii, owners, ids, points = index
    keep = ~np.isin(owners, np.asarray(sorted(dead_owners), dtype=np.intp))
    if keep.all():
        return index
    if not keep.any():
        return None
    return cKDTree(points[keep]), radii[keep], owners[keep], ids, points[keep]


def prune_vascular_stubs(
    G: Union[nx.Graph, nx.MultiGraph],
    min_stub_length: float = 10.0,
    max_iterations: int = 100,
    debug: bool = False,
    voxel_size: Tuple[float, float, float] = (1, 1, 1),
    *,
    radius_at: Optional[Callable[[np.ndarray], float]] = None,
    radius_multiple: float = 0.0,
    image_extent_um: Optional[Sequence[float]] = None,
    inside_lumen: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mask_continues_at: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    lumen_radius_at: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    image_face_test: Optional[Callable[[Any, np.ndarray], bool]] = None,
    index_memo: Optional[dict] = None,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Iteratively remove short terminal stubs until convergence.

    A stub is a dead end's chain: from its tip through degree-2 nodes to the
    junction it hangs from, judged whole -- its length is the chain's, and
    the parent radius is read at that junction, not at a node inside the
    stub. (Judged piece by piece, a long dead end in short pieces was worn
    away one piece at a time.) It is removed when shorter than its
    threshold. With *radius_at* (the vessel radius, in microns, at a
    position) and a positive *radius_multiple*, the threshold is that many
    radii of the vessel the stub leaves -- read at the junction it hangs from
    -- so a skeleton spur on a wide vessel, as long as that vessel is thick,
    goes, while a real short capillary end survives; one fixed length could
    only ever do one of the two. *min_stub_length* is the threshold wherever
    no radius can be read.

    With *image_extent_um* (the image's ``(z, y, x)`` extent in microns, node
    positions running from 0 to it), a terminal no further from an image face
    than its own threshold is kept: that is a vessel the image cut through --
    exactly the open ends inlets and outlets are chosen from -- not a spur.

    The mask can say more about a stub (each read only when given, so
    without them nothing changes):

    - *inside_lumen* (``points_um -> bool``): a stub less than half of which
      lies in the mask is pruned whatever its length -- not a vessel there.
    - *radius_at*: a stub whose tip lies within the parent's radius + 1 um of
      the junction it hangs from is pruned whatever its length -- a Lee spur
      inside the parent's own lumen. Only with *inside_lumen* too.
    - *mask_continues_at* ``(tip_um, outward_direction) -> bool``: where the
      mask runs on past the tip, the tip is not a vessel end but a vessel
      the network lost track of, so the threshold is the larger of the
      radius rule and *min_stub_length*. A tip at a dead end of the mask (a
      blind sprout) keeps the radius rule.
    - *lumen_radius_at* (``points_um -> radii``, with *inside_lumen*): a stub
      that never gets more than :data:`STUB_PROTRUSION_RADII` of its own
      radius outside the lumen of some other centreline is pruned whatever
      its length -- a hair along that vessel's wall. All of it must stay
      inside, so a real branch beside a hair, running on past the hair's
      tip, is kept.

    Stubs at an image face are kept as before; the two whatever-its-length
    rules apply there too, since neither is a vessel the image cut through.
    *index_memo*, a dict passed to every call on one graph and mask, keeps
    the centreline index's points and radii per unchanged edge between
    calls (``_centreline_index``). *image_face_test* ``(tip, path_from_junction) -> bool``, when given,
    decides which short stubs are at a face instead of *image_extent_um*'s
    "within its own threshold" -- the dead-end definition's
    ``leaves_image`` (``assemble.lumen_cleanup``'s ``stub_face_protection``).
    """
    if min_stub_length < 0:
        raise ValueError("min_stub_length must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must be a 3-tuple")
    if radius_multiple < 0:
        raise ValueError("radius_multiple must be non-negative")

    G_pruned = G.copy()
    if G_pruned.number_of_nodes() == 0:
        return G_pruned

    extent = None if image_extent_um is None else np.asarray(image_extent_um, dtype=float)

    def threshold_for(junction: Any) -> float:
        if radius_at is not None and radius_multiple > 0 and "pos" in G_pruned.nodes[junction]:
            radius = float(radius_at(np.asarray(G_pruned.nodes[junction]["pos"], dtype=float)))
            if radius > 0:
                return radius_multiple * radius
        return float(min_stub_length)

    def at_an_image_face(node: Any, threshold: float) -> bool:
        if extent is None or "pos" not in G_pruned.nodes[node]:
            return False
        position = np.asarray(G_pruned.nodes[node]["pos"], dtype=float)
        return float(np.min(np.minimum(position, extent - position))) <= threshold

    def stub_path(nodes: list, edges: list) -> Optional[np.ndarray]:
        if any("pos" not in G_pruned.nodes[n] for n in nodes):
            return None
        return _chain_path(G_pruned, edges)

    def inside_other_lumens(path: np.ndarray, edges: list, index) -> bool:
        """Whether the stub never gets more than its own radius outside the
        lumen of a centreline that is not its own."""
        tree, radii, owners, ids, _points = index
        own = [ids.get((frozenset((a, b)), key), -1) for a, b, key in edges]
        k = min(16, len(owners))
        distance, nearest = tree.query(path, k=k)
        distance, nearest = distance.reshape(len(path), k), nearest.reshape(len(path), k)
        excess = np.where(np.isin(owners[nearest], own), np.inf, distance - radii[nearest])
        protrusion = float(np.max(np.min(excess, axis=1)))
        tip_radius = float(np.asarray(lumen_radius_at(path[-1:]), dtype=float).reshape(-1)[0])
        return protrusion < STUB_PROTRUSION_RADII * tip_radius

    def not_a_vessel(path: np.ndarray, junction: Any) -> bool:
        if 2 * int(np.count_nonzero(inside_lumen(path))) < len(path):
            return True
        if radius_at is None:
            return False
        junction_pos = path[0]
        parent_radius = float(radius_at(junction_pos))
        return parent_radius > 0 and float(np.linalg.norm(path[-1] - junction_pos)) < parent_radius + 1.0

    def mask_continues(path: np.ndarray) -> bool:
        reverse = path[::-1]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(reverse, axis=0), axis=1))])
        tip = reverse[0]
        outward = tip - reverse[min(int(np.searchsorted(arc, STUB_TANGENT_LENGTH_UM)), len(reverse) - 1)]
        norm = float(np.linalg.norm(outward))
        return norm > 0 and bool(mask_continues_at(tip, outward / norm))

    total_removed = 0
    not_vessels = 0
    protected: set = set()
    iteration = 0
    index = None

    while iteration < max_iterations:
        iteration += 1
        nodes_before = G_pruned.number_of_nodes()
        if nodes_before == 0:
            break
        nodes_to_remove = []
        terminal_nodes = [
            n for n in G_pruned.nodes() if G_pruned.degree(n) == 1
        ]
        nodes_to_remove.extend(n for n in G_pruned.nodes() if G_pruned.degree(n) == 0)
        chains = {node: _chain(G_pruned, node) for node in terminal_nodes}
        if inside_lumen is not None and lumen_radius_at is not None and iteration == 1:
            index = _centreline_index(G_pruned, lumen_radius_at, memo=index_memo)
        for node in terminal_nodes:
            nodes, edges = chains[node]
            junction = nodes[-1]
            # An isolated chain goes whole; a dead end leaves its junction.
            goes = nodes if G_pruned.degree(junction) <= 1 else nodes[:-1]
            try:
                edge_length = sum(
                    calculate_edge_length(a, b, _edge_data(G_pruned, (a, b, key)), voxel_size)
                    for a, b, key in edges
                )
                threshold = threshold_for(junction)
                path = (
                    stub_path(nodes, edges)
                    if inside_lumen is not None or mask_continues_at is not None
                    else None
                )
                if path is not None and len(path) > 1:
                    if inside_lumen is not None and (
                        not_a_vessel(path, junction)
                        or (index is not None and inside_other_lumens(path, edges, index))
                    ):
                        nodes_to_remove.extend(goes)
                        not_vessels += 1
                        continue
                    if mask_continues_at is not None and mask_continues(path):
                        threshold = max(threshold, float(min_stub_length))
                at_face = (
                    image_face_test(node, path)
                    if image_face_test is not None and path is not None and len(path) > 1
                    else at_an_image_face(node, threshold)
                ) if edge_length < threshold else False
                if at_face:
                    protected.add(node)
                    continue
                if edge_length < threshold:
                    nodes_to_remove.extend(goes)
                    if debug:
                        logger.debug(
                            f"  Iteration {iteration}: Marking node {node} "
                            f"(stub length: {edge_length:.2f})"
                        )
            except Exception as e:
                if debug:
                    logger.debug(f"  Warning: Could not calculate edge length: {e}")
                nodes_to_remove.extend(goes)

        if index is not None and nodes_to_remove:
            # The stubs about to go leave the index: their points would be
            # read as other vessels' lumens by the stubs judged next pass.
            dead = {
                index[3].get((frozenset((a, b)), key)) for n in set(nodes_to_remove)
                for a, b, key in (
                    G_pruned.edges(n, keys=True) if G_pruned.is_multigraph()
                    else ((a, b, None) for a, b in G_pruned.edges(n))
                )
            }
            index = _index_without(index, {d for d in dead if d is not None})
        G_pruned.remove_nodes_from(nodes_to_remove)
        nodes_after = G_pruned.number_of_nodes()
        removed_this_iteration = nodes_before - nodes_after
        total_removed += removed_this_iteration

        if debug:
            logger.debug(
                f"  Iteration {iteration}: Removed {removed_this_iteration} "
                f"({nodes_after} remaining)"
            )
        if removed_this_iteration == 0:
            if debug:
                logger.debug(f"Convergence reached after {iteration} iterations")
            break

    kept_at_faces = sum(1 for node in protected if node in G_pruned)
    logger.info(
        "Pruning complete: removed %d terminal stubs (%s), kept %d short one(s) "
        "at an image face, graph now has %d nodes / %d edges",
        total_removed,
        (
            f"threshold {radius_multiple:g} x the parent vessel's radius"
            if radius_at is not None and radius_multiple > 0
            else f"threshold {float(min_stub_length):g} um"
        ),
        kept_at_faces,
        G_pruned.number_of_nodes(),
        G_pruned.number_of_edges(),
    )
    if inside_lumen is not None:
        logger.info(
            "Pruning: %d of those stubs were mostly outside the mask or never left "
            "another vessel's lumen, and went whatever their length",
            not_vessels,
        )
    return G_pruned


#: Edge attribute the flow solve writes: False on a vessel no inlet-to-outlet
#: path runs along -- in a branch or tree without both an inlet and an
#: outlet, or a dead end hanging off one that has both -- since no pressure
#: difference drives a flow through it, so its flow and pressures are not a
#: result; True everywhere else. Exactly the vessels
#: :func:`haemolynx.graph.remove_vessels_off_inlet_outlet_paths` would
#: remove are False.
FLOW_SOLVED = "flow_solved"

#: Edge attribute the flow solve writes beside :data:`FLOW_SOLVED`: why a
#: vessel carries flow or not (PLAN_dead_ends.md, decision D4). ``solved`` --
#: on a path between two boundaries; ``blind_end`` -- hanging off the flowing
#: network behind interior dead ends (or a loop off every path): its zero flow
#: is the answer, not an unknown; ``closed_end`` -- hanging behind a dead end
#: that leaves the image but was given no boundary condition
#: (``open_end_model="closed"``); ``no_boundary`` -- in a piece of the network
#: no flow reaches at all (no inlet and outlet).
FLOW_STATE = "flow_state"
FLOW_STATES = ("solved", "blind_end", "closed_end", "no_boundary")

#: Edge attributes the flow solve writes (``haemodynamics.set_edge_flows``,
#: and the haematocrit loop's ``discharge_haematocrit``): the one list of
#: what every export blanks where :func:`flow_is_undefined` (the VTK vessels
#: file, a perturbation's edges CSV) and the viewer blanks on every unsolved
#: vessel (``gui.results.SOLVED_FLOW_COLUMNS``).
SOLVE_DERIVED_EDGE_ATTRIBUTES = (
    "pressure_u", "pressure_v", "pressure_drop", "flow_signed", "flow_abs", "flow_abs_log10",
    "discharge_haematocrit",
)


def flow_is_undefined(data: dict) -> bool:
    """Whether a vessel's solved flow and pressures are no answer at all: in
    a piece of the network no inlet and outlet reach (``no_boundary``), where
    the solve left pressure 0 for want of any. A vessel hanging off the
    flowing network behind dead ends (``blind_end``, ``closed_end``) has a
    flow of zero and its junction's pressure, which *are* the answer, and the
    flow state says why. A graph solved before flow states existed: every
    unsolved vessel."""
    state = data.get(FLOW_STATE)
    if state is not None:
        return state == "no_boundary"
    return data.get(FLOW_SOLVED) is False


def zero_flows_behind_dead_ends(G: Union[nx.Graph, nx.MultiGraph]) -> int:
    """Set ``flow_abs``, ``flow_signed`` and ``pressure_drop`` to exactly 0.0
    (``flow_abs_log10`` to match) on every vessel whose :data:`FLOW_STATE` is
    ``blind_end`` or ``closed_end``; return how many. Nothing drives a flow
    through them, so what the pressure solve left there is rounding (up to
    5e-20 m^3/s on E14.5, either sign). Their ``pressure_u``/``pressure_v``
    stay as solved: the junction's pressure, which is the answer. Only the
    attributes a solve wrote are touched. Run after
    ``haemodynamics.set_edge_flows`` on any graph whose flow states are marked
    (:func:`mark_flow_solved_edges`, or copied from the baseline that was)."""
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    zero = {"pressure_drop": 0.0, "flow_signed": 0.0, "flow_abs": 0.0,
            "flow_abs_log10": flow_abs_log10_value(0.0)}
    zeroed = 0
    for *_edge, data in G.edges(data=True):
        if data.get(FLOW_STATE) in ("blind_end", "closed_end") and "flow_signed" in data:
            for name, value in zero.items():
                if name in data:
                    data[name] = value
            zeroed += 1
    return zeroed


def _components_by_io(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
):
    """Each connected component's nodes, and whether it holds both a start
    and an output node. Node IDs not present in ``G`` are ignored."""
    start_node_set = {
        int(node_id) for node_id in starting_nodes if int(node_id) in G.nodes
    }
    output_node_set = {
        int(node_id) for node_id in output_nodes if int(node_id) in G.nodes
    }
    for component_nodes in nx.connected_components(G):
        component_node_set = {int(node_id) for node_id in component_nodes}
        yield component_node_set, bool(
            component_node_set & start_node_set and component_node_set & output_node_set
        )


def mark_flow_solved_edges(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
    open_ends: Sequence[Any] = (),
) -> int:
    """Write :data:`FLOW_SOLVED` and :data:`FLOW_STATE` on every edge of *G*;
    return how many are unsolved (on no path from a start to an output node,
    by :func:`haemolynx.graph.inlet_to_outlet_vessels`). *open_ends* -- the
    vessels leaving the image the solve gave a boundary condition -- can be
    either, so a path from one to another carries flow too."""
    from .connectivity import inlet_to_outlet_vessels

    multi = G if G.is_multigraph() else nx.MultiGraph(G)
    opened = list(open_ends)
    through = {
        (frozenset((u, v)), k)
        for u, v, k in inlet_to_outlet_vessels(
            multi, [*starting_nodes, *opened], [*output_nodes, *opened]
        )
    }
    edges = (
        G.edges(keys=True, data=True)
        if G.is_multigraph()
        else ((u, v, 0, data) for u, v, data in G.edges(data=True))
    )
    unsolved = 0
    for u, v, k, data in list(edges):
        solved = (frozenset((u, v)), k) in through
        data[FLOW_SOLVED] = solved
        data[FLOW_STATE] = "solved" if solved else None
        unsolved += not solved
    _mark_flow_states(multi if G.is_multigraph() else G, set(opened))
    return unsolved


def _mark_flow_states(G, opened: set) -> None:
    """Why each unsolved vessel carries no flow (:data:`FLOW_STATE`), piece by
    piece of the unsolved network."""
    from .dead_ends import node_leaves_image

    edges = (
        list(G.edges(keys=True, data=True)) if G.is_multigraph()
        else [(u, v, 0, d) for u, v, d in G.edges(data=True)]
    )
    flowing = {n for u, v, _k, d in edges if d.get(FLOW_SOLVED) for n in (u, v)}
    rest = nx.Graph()
    for u, v, _k, d in edges:
        if not d.get(FLOW_SOLVED):
            rest.add_edge(u, v)
    state_of = {}
    for piece in nx.connected_components(rest):
        if not piece & flowing:
            state = "no_boundary"
        elif any(G.degree(n) == 1 and n not in opened and node_leaves_image(G, n) for n in piece):
            state = "closed_end"
        else:
            state = "blind_end"
        for node in piece:
            state_of[node] = state
    for u, v, _k, d in edges:
        if not d.get(FLOW_SOLVED):
            d[FLOW_STATE] = state_of.get(u, "no_boundary")


def edges_with_connected_io(
    G: nx.MultiGraph,
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
) -> list[tuple[int, int, int]]:
    """The edges ``(u, v, key)`` of *G* in a component holding both a start
    and an output node: the ones :func:`remove_components_without_connected_io`
    keeps."""
    kept_nodes: set[int] = set()
    for component_node_set, has_io in _components_by_io(G, starting_nodes, output_nodes):
        if has_io:
            kept_nodes |= component_node_set
    # One end tells: both ends of an edge share a component.
    return [(u, v, key) for u, v, key in G.edges(keys=True) if int(u) in kept_nodes]


def remove_components_without_connected_io(
    G: Union[nx.Graph, nx.MultiGraph],
    starting_nodes: list[int],
    output_nodes: list[int],
) -> tuple[Union[nx.Graph, nx.MultiGraph], dict[str, int]]:
    """Keep only connected components containing both start and output nodes.

    Components that do not include at least one node from each boundary set
    are removed. Node IDs not present in ``G`` are ignored.
    """
    keep_nodes: set[int] = set()
    removed_component_count = 0
    removed_node_count = 0

    for component_node_set, has_io in _components_by_io(G, starting_nodes, output_nodes):
        if has_io:
            keep_nodes.update(component_node_set)
        else:
            removed_component_count += 1
            removed_node_count += len(component_node_set)

    if removed_component_count <= 0:
        return G.copy(), {
            "removed_components": 0,
            "removed_nodes": 0,
            "remaining_nodes": int(G.number_of_nodes()),
        }

    G_pruned = G.subgraph(keep_nodes).copy()
    return G_pruned, {
        "removed_components": int(removed_component_count),
        "removed_nodes": int(removed_node_count),
        "remaining_nodes": int(G_pruned.number_of_nodes()),
    }


def remove_edges_for_self_connected_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """Remove edges for nodes that are connected to themselves with no nodes in between."""
    G_pruned = G.copy()
    for node in G_pruned.nodes():
        if node in G_pruned.neighbors(node):
            G_pruned.remove_edge(node, node)
    return G_pruned
