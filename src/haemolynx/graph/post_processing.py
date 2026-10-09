"""Post-processing a network by hand: junctions where four or more vessels meet.

Pure graph logic behind the panel's "7. Post processing" tab -- no Qt, no
napari -- so every rule here is testable on a hand-built graph. A vessel is
an edge and a junction a node, as everywhere else; a vessel's *branchID* is
its position in ``G.edges(keys=True)``, the same index the viewer's hover
shows (see :func:`edge_keys`).

Every edit marks what it did, for the pipeline's ``post_process`` stage to
bring in line with the rest of the network (its length, branch order,
diameter and thick-vessel bridge): each vessel it created or rewrote carries
:data:`EDITED`, and the graph carries :data:`PENDING` -- set by a deletion
too, which changes branch orders downstream of it without leaving a vessel
to mark. :func:`has_pending_edits` asks; the stage clears both.

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

And, for the tab's click-in-the-viewer edits: :func:`add_traced_vessel` adds
a vessel traced click by click -- from a node, or from a point on a vessel
where :func:`split_vessel_at` forms a new node, through waypoints joined by
:func:`trace_path` (A* through the segmented mask, or through the raw image's
intensities with :class:`IntensityCostField`), to another node or vessel --
with the mean diameter of the vessels already at its ends
(:func:`mean_incident_diameter`). :func:`vessel_path_between` and
:func:`add_vessel_between` are the two-node version. After deleting,
:func:`prune_disconnected_branches` removes whatever the deletions cut off
from every inlet-to-outlet piece, and every dead end that reaches no outlet.

For Manual loop review: :func:`short_loops` lists the network's short loops,
shortest first, each split into the sides a person chooses between, and
:func:`delete_loop_side` takes one side out.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from ._helpers import calculate_path_length, next_node_id
from .degree2 import create_trivial_merged_edge
from .edit import astar_path, voxel_path_to_microns
from .connectivity import remove_vessels_off_inlet_outlet_paths
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

__all__ = [
    "APPLIED",
    "DEFAULT_SPLIT_CONNECTOR_LENGTH_UM",
    "EDITED",
    "IntensityCostField",
    "JunctionVessel",
    "MIN_ROUTED_INSIDE_FRACTION",
    "PENDING",
    "SPLIT_SNAP_UM",
    "ShortLoop",
    "TracedVessel",
    "VesselEnd",
    "add_traced_vessel",
    "add_vessel_between",
    "clear_edit_marks",
    "delete_loop_side",
    "delete_vessels",
    "edge_keys",
    "edited_edges",
    "has_pending_edits",
    "high_degree_junctions",
    "junction_vessels",
    "mark_edited",
    "mean_incident_diameter",
    "prune_disconnected_branches",
    "short_loops",
    "smooth_traced_path",
    "split_high_degree_junctions",
    "split_junction",
    "split_vessel_at",
    "trace_path",
    "vessel_path",
    "vessel_path_between",
]

EdgeKey = tuple[Any, Any, Any]

#: Edge attribute: an edit here created or rewrote this vessel, so the
#: ``post_process`` stage measures it again the way graph building's vessels
#: were measured.
EDITED = "post_processing_edited"
#: Graph attribute: the network was edited here since it was last regenerated.
PENDING = "post_processing_pending"
#: Graph attribute: edits made here were brought into this network (set by the
#: ``post_process`` stage, and kept) -- work a rerun from an earlier stage drops.
APPLIED = "post_processing_applied"

#: Length of the connector vessel :func:`split_junction` inserts.
DEFAULT_SPLIT_CONNECTOR_LENGTH_UM = 15.0

#: A routed new vessel must run at least this fraction of its points inside
#: the segmented mask; less, and the route has left the vessel -- draw it
#: straight instead.
MIN_ROUTED_INSIDE_FRACTION = 0.5

#: How far along a vessel its direction out of a junction is measured.
_DIRECTION_SAMPLE_UM = 10.0

#: A traced vessel ending on a vessel this close (along it) to one of that
#: vessel's ends joins the node there instead of cutting off a sliver.
SPLIT_SNAP_UM = 1.0

#: Raw-intensity routing (:class:`IntensityCostField`): the brightest voxels
#: of a window cost 1 and the dimmest ``1 / RAW_COST_FLOOR``, so a route
#: follows a bright vessel rather than cutting across dark tissue.
RAW_COST_FLOOR = 0.01
#: Window percentiles taken as black and white: robust to a few hot pixels.
RAW_COST_PERCENTILES = (1.0, 99.5)
#: Gaussian blur (voxels) before costing, so a route does not hop between
#: bright specks of noise.
RAW_COST_SMOOTHING_SIGMA_VOX = 1.0


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


def mark_edited(G: nx.MultiGraph, edges: Iterable[EdgeKey] = ()) -> None:
    """Mark *edges* :data:`EDITED` and *G* :data:`PENDING` (with no edges: *G* only)."""
    for u, v, k in edges:
        G.edges[u, v, k][EDITED] = True
    G.graph[PENDING] = True


def edited_edges(G: nx.MultiGraph) -> list[EdgeKey]:
    """The vessels an edit created or rewrote since the last regenerate."""
    return [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get(EDITED)]


def has_pending_edits(G: nx.MultiGraph | None) -> bool:
    """Whether *G* was edited here since it was last regenerated."""
    if G is None:
        return False
    return bool(G.graph.get(PENDING)) or bool(edited_edges(G))


def clear_edit_marks(G: nx.MultiGraph) -> None:
    """Forget every edit mark: the network is in line with itself again."""
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        data.pop(EDITED, None)
    G.graph.pop(PENDING, None)


def _is_bridge(data: Mapping[str, Any]) -> bool:
    return bool(data.get(IS_ZERO_RESISTANCE))


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

    Thick-vessel bridges (:data:`~haemolynx.graph.IS_ZERO_RESISTANCE`, the
    opening of a small vessel into a big one's lumen) are kept apart from the
    vessels they open: a bridge and an ordinary vessel meeting at a
    pass-through are not merged -- the merged vessel would take one of the
    two's resistance rule for both -- and a bridge a deletion leaves hanging
    from a thick vessel, with nothing beyond it, is deleted with the vessel
    it opened.

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
    queue = list(loss)
    while queue:
        node = queue.pop(0)
        if not G.has_node(node) or node in protected_set:
            continue
        degree = G.degree(node)
        if degree == 0:
            G.remove_node(node)
            changed.add(node)
            continue
        if degree == 1:
            ((_, other, k, data),) = G.edges(node, keys=True, data=True)
            if _is_bridge(data):
                G.remove_edge(node, other, k)
                G.remove_node(node)
                changed.update({node, other})
                queue.append(other)
            continue
        if degree != 2:
            continue
        incident = list(G.edges(node, keys=True, data=True))
        if len(incident) != 2:
            continue  # a self-loop counts twice towards degree
        (_, n1, _k1, d1), (_, n2, _k2, d2) = incident
        if n1 == n2:
            continue  # two vessels to one neighbour: a loop, not a pass-through
        if _is_bridge(d1) != _is_bridge(d2):
            continue  # a bridge and the vessel it opens: two vessels, not one
        merged = create_trivial_merged_edge(d1, d2, G.nodes[node].get("pos"))
        G.remove_node(node)
        key = G.add_edge(n1, n2, **merged)
        mark_edited(G, [(n1, n2, key)])
        changed.update({node, n1, n2})
    mark_edited(G)
    return changed


@dataclass(frozen=True)
class ShortLoop:
    """One short loop of the network, as Manual loop review lists it."""

    #: Its ``(a, b, key)`` edges, in order round it.
    edges: tuple[EdgeKey, ...]
    #: Those edges in runs between the places the rest of the network -- or
    #: a boundary node -- meets the loop (:func:`~haemolynx.graph.lumen_loops.loop_sides`):
    #: what Delete this side takes out, one run at a time.
    sides: tuple[tuple[EdgeKey, ...], ...]
    #: Summed vessel length round it, in microns.
    length_um: float
    #: The middle of the loop, physical microns ``(z, y, x)``.
    centre_um: tuple[float, float, float]
    #: Its points in order round it, physical microns ``(z, y, x)``.
    points_um: np.ndarray = field(compare=False, repr=False)


def short_loops(
    G: nx.MultiGraph, *, search_um: float | None = None, cut_at: Iterable[Any] = ()
) -> list[ShortLoop]:
    """The loops of *G* shorter than *search_um* round (default
    :data:`~haemolynx.graph.lumen_loops.LOOP_SEARCH_UM`), shortest first.

    Each is some vessel's shortest loop, listed once
    (:func:`~haemolynx.graph.lumen_loops.iter_short_loops`, the loops graph
    building judges); loops of equal length keep the order they were found
    in, so an unchanged graph lists the same way every time. Their sides are
    cut at the nodes of *cut_at* (boundary nodes) as well as wherever another
    vessel leaves them.
    """
    from .lumen_loops import LOOP_SEARCH_UM, iter_short_loops, loop_centre_um, loop_sides

    limit = LOOP_SEARCH_UM if search_um is None else float(search_um)
    cut = frozenset(cut_at)
    found = []
    for cycle, polyline in iter_short_loops(G, limit):
        edges = tuple(tuple(edge) for edge in cycle)
        found.append(
            ShortLoop(
                edges=edges,
                sides=tuple(tuple(tuple(e) for e in side) for side in loop_sides(G, edges, cut_at=cut)),
                length_um=float(sum(_optional_float(G.edges[e].get("length")) or 0.0 for e in edges)),
                centre_um=tuple(float(c) for c in loop_centre_um(polyline)),
                points_um=np.asarray(polyline, dtype=float),
            )
        )
    found.sort(key=lambda loop: loop.length_um)
    return found


def delete_loop_side(
    G: nx.MultiGraph, loop: ShortLoop, side: int, *, protected: Iterable[Any] = ()
) -> set[Any]:
    """Delete side *side* of *loop* -- its run of vessels -- through
    :func:`delete_vessels`, which tidies the nodes it touched and never
    strands a *protected* (boundary) node. Raises ``ValueError`` for a side
    the loop does not have, or one an earlier edit already changed.
    Returns the node ids whose drawing changed."""
    if not 0 <= int(side) < len(loop.sides):
        raise ValueError(f"The loop has sides 1 to {len(loop.sides)}, not {int(side) + 1}")
    return delete_vessels(G, loop.sides[int(side)], protected=protected)


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
    mark: bool = True,
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

    Connector vessels carry ``junction_split_connector=True`` and, like every
    vessel this module adds, the mean diameter of the vessels that met at
    *node* as a manual override (:func:`mean_incident_diameter`; none when
    none of them has a diameter). With *mark* (the tab's edits), connectors
    and moved vessels are marked :data:`EDITED`, so the ``post_process``
    stage measures them and assigns their branch order.

    Returns the new node ids.
    """
    length = float(connector_length_um)
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"Connector length must be positive, got {connector_length_um!r}")
    if node not in G:
        raise ValueError(f"Node {node!r} is not in the network")
    origin = _node_position(G, node)
    connector_diameter = mean_incident_diameter(G, [node])
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
            moved = G.add_edge(new_node, other, **attrs)
            if mark:
                mark_edited(G, [(new_node, other, moved)])
        connector_attrs: dict[str, Any] = {
            "voxels": [tuple(float(c) for c in origin), tuple(float(c) for c in new_pos)],
            "length": length,
            "junction_split_connector": True,
        }
        if connector_diameter is not None:
            from haemolynx.haemodynamics.poiseuille import set_edge_diameter_override

            set_edge_diameter_override(connector_attrs, connector_diameter)
        key = G.add_edge(node, new_node, **connector_attrs)
        if mark:
            mark_edited(G, [(node, new_node, key)])
        connectors.add((node, new_node, key))
        new_nodes.append(new_node)
    return new_nodes


def split_high_degree_junctions(
    G: nx.MultiGraph,
    *,
    connector_length_um: float | None = None,
) -> dict[Any, list[Any]]:
    """:func:`split_junction` every junction of four or more vessels, for a run.

    What the ``split_junctions`` haematocrit junction rule does to the network
    before its model is built, so the phase-separation law has one parent and
    two daughters at every junction. Unlike the Post processing tab's splits,
    these come after the ``post_process`` stage and are not marked for it, so
    each connector also takes a ``branch_order`` here -- without one it would
    get no resistance. It takes the order the two vessels peeled onto it
    share, else the wider one's: the connector is the first stretch of their
    common trunk.

    Without *connector_length_um*, each junction's connectors are as long as
    its vessels are wide (:func:`mean_incident_diameter`, which is also the
    connector's diameter). A four-way node in a traced network is usually two
    bifurcations that graph building merged, and they were about that far
    apart; a fixed length would re-plumb a capillary bed and fall inside an
    arteriole's own junction. A junction with no diameters takes
    :data:`DEFAULT_SPLIT_CONNECTOR_LENGTH_UM`.

    Returns ``{junction: [new node ids]}`` for every junction split.
    """
    split: dict[Any, list[Any]] = {}
    for node in high_degree_junctions(G):
        length = connector_length_um
        if length is None:
            length = mean_incident_diameter(G, [node]) or DEFAULT_SPLIT_CONNECTOR_LENGTH_UM
        new_nodes = split_junction(G, node, connector_length_um=length, mark=False)
        if not new_nodes:
            continue
        split[node] = new_nodes
        for new_node in new_nodes:
            connector = None
            peeled = []
            for _u, _v, _k, data in G.edges(new_node, keys=True, data=True):
                if data.get("junction_split_connector") and connector is None:
                    connector = data
                else:
                    peeled.append(data)
            labelled = [data for data in peeled if data.get("branch_order") is not None]
            if connector is None or not labelled:
                continue
            widest = max(labelled, key=lambda data: _optional_float(data.get("diameter_um")) or 0.0)
            connector["branch_order"] = widest["branch_order"]
    return split


def mean_incident_diameter(G: nx.MultiGraph, nodes: Iterable[Any]) -> float | None:
    """Mean ``diameter_um`` of the vessels meeting at any of *nodes*.

    Each vessel counts once even when it joins two of the nodes; vessels with
    no usable diameter are skipped. None when none has one.
    """
    seen: set = set()
    diameters: list[float] = []
    for node in nodes:
        if node not in G:
            continue
        for u, v, k, data in G.edges(node, keys=True, data=True):
            key = (frozenset((u, v)), k)
            if key in seen:
                continue
            seen.add(key)
            diameter = _optional_float(data.get("diameter_um"))
            if diameter is not None and diameter > 0:
                diameters.append(diameter)
    return float(np.mean(diameters)) if diameters else None


def _inside_fraction(points_um: Sequence[Sequence[float]], mask: Any, voxel_size_zyx) -> float:
    scale = np.asarray(voxel_size_zyx, dtype=float)
    shape = np.asarray(np.shape(mask))
    index = np.clip(np.round(np.asarray(points_um, dtype=float) / scale).astype(int), 0, shape - 1)
    values = np.asarray([bool(mask[tuple(i)]) for i in index])
    return float(values.mean()) if len(values) else 0.0


def vessel_path_between(
    G: nx.MultiGraph,
    a: Any,
    b: Any,
    *,
    cost_field: Any = None,
    mask: Any = None,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    min_inside_fraction: float = MIN_ROUTED_INSIDE_FRACTION,
) -> tuple[list[tuple[float, float, float]], str]:
    """The path a new vessel from *a* to *b* follows, and how it was found.

    Routed through *cost_field* (``graph.mask_cost_field`` of the segmented
    *mask*) with :func:`haemolynx.graph.astar_path` when both are given, and
    kept when at least *min_inside_fraction* of it lies inside the mask;
    otherwise -- no image, a route that left the vessel, or routing that
    failed -- the straight line from *a* to *b*. Returns ``(points_um,
    "routed" | "straight")``; the path always starts and ends exactly on the
    two nodes.
    """
    start = _node_position(G, a)
    end = _node_position(G, b)
    straight = [tuple(float(c) for c in start), tuple(float(c) for c in end)]
    if cost_field is None or mask is None:
        return straight, "straight"
    scale = np.asarray(voxel_size_zyx, dtype=float)
    try:
        path_vox = astar_path(cost_field, start / scale, end / scale, voxel_size_zyx=scale)
        points = voxel_path_to_microns(path_vox, scale)
    except Exception:  # noqa: BLE001 - any routing failure falls back to straight
        return straight, "straight"
    if len(points) < 2 or _inside_fraction(points, mask, scale) < min_inside_fraction:
        return straight, "straight"
    points[0], points[-1] = straight[0], straight[1]
    return points, "routed"


def add_vessel_between(
    G: nx.MultiGraph,
    a: Any,
    b: Any,
    points_um: Sequence[Sequence[float]] | None = None,
    *,
    diameter_um: float | None = None,
) -> EdgeKey:
    """Add a vessel from node *a* to node *b* along *points_um*.

    *points_um* defaults to the straight line between the two. Its length is
    measured from the path. With *diameter_um*, the vessel carries it as a
    manual override (``diameter_source="override"``), which the Diameters
    stage keeps over the branch-order table. Returns the new edge's
    ``(u, v, key)``.
    """
    if a == b:
        raise ValueError("A vessel needs two different nodes")
    for node in (a, b):
        if node not in G:
            raise ValueError(f"Node {node!r} is not in the network")
    if points_um is None:
        points_um = [_node_position(G, a), _node_position(G, b)]
    points = [tuple(float(c) for c in np.asarray(p, dtype=float)[:3]) for p in points_um]
    if len(points) < 2:
        raise ValueError("A vessel path needs at least two points")
    attrs: dict[str, Any] = {
        "voxels": points,
        "length": float(calculate_path_length(points)),
        "post_processing_added": True,
    }
    if diameter_um is not None:
        from haemolynx.haemodynamics.poiseuille import set_edge_diameter_override

        set_edge_diameter_override(attrs, diameter_um)
    key = G.add_edge(a, b, **attrs)
    mark_edited(G, [(a, b, key)])
    return (a, b, key)


def vessel_path(G: nx.MultiGraph, edge: EdgeKey) -> np.ndarray:
    """Vessel *edge*'s path as (n, 3) microns, running from its ``u`` end.

    Its ``voxels`` when it has two or more, else the straight line between
    its nodes.
    """
    u, v, k = edge
    if not G.has_edge(u, v, key=k):
        raise ValueError(f"No vessel ({u!r}, {v!r}, {k!r}) in the network")
    return _path_from(G, u, v, G.edges[u, v, k])


def _dedupe(points: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
    """*points* as tuples, without consecutive repeats."""
    kept: list[tuple[float, float, float]] = []
    for point in points:
        p = tuple(float(c) for c in np.asarray(point, dtype=float)[:3])
        if not kept or np.linalg.norm(np.subtract(p, kept[-1])) > 1e-9:
            kept.append(p)
    return kept


@dataclass(frozen=True)
class _PlaceOnVessel:
    """Where on a vessel's path a point falls."""

    #: The path segment it falls on (``points[segment]`` to ``points[segment + 1]``).
    segment: int
    point: np.ndarray
    #: Distance along the path from its ``u`` end, and the path's whole length.
    along_um: float
    total_um: float


def _place_on_vessel(points: np.ndarray, target: Sequence[float]) -> _PlaceOnVessel:
    """The point of path *points* nearest *target*."""
    target = np.asarray(target, dtype=float)[:3]
    a, b = points[:-1], points[1:]
    d = b - a
    length_sq = np.einsum("ij,ij->i", d, d)
    safe = np.where(length_sq > 0, length_sq, 1.0)
    t = np.clip(np.einsum("ij,ij->i", target - a, d) / safe, 0.0, 1.0)
    t = np.where(length_sq > 0, t, 0.0)
    closest = a + t[:, None] * d
    i = int(np.argmin(np.linalg.norm(closest - target, axis=1)))
    lengths = np.sqrt(length_sq)
    return _PlaceOnVessel(
        segment=i,
        point=closest[i],
        along_um=float(lengths[:i].sum() + t[i] * lengths[i]),
        total_um=float(lengths.sum()),
    )


def _snapped_end(edge: EdgeKey, place: _PlaceOnVessel, snap_um: float) -> Any | None:
    """The node at an end of *edge* that *place* is within *snap_um* of, if any."""
    u, v, _k = edge
    if place.along_um <= snap_um:
        return u
    if place.total_um - place.along_um <= snap_um:
        return v
    return None


def split_vessel_at(
    G: nx.MultiGraph,
    edge: EdgeKey,
    point_um: Sequence[float],
    *,
    reserved_ids: Iterable[Any] = (),
    snap_um: float = SPLIT_SNAP_UM,
) -> Any:
    """Cut vessel *edge* in two at the point of its path nearest *point_um*.

    A new node goes exactly there -- between two of the path's points if
    that is where it falls, not at the nearest one, so a straight two-point
    vessel can be split too. The two halves keep every attribute of the
    vessel (its diameter and where it came from included) except its path
    and ``length``, which are measured from each half: the two lengths add up
    to the vessel's. Within *snap_um* of either end, nothing is cut and the
    node at that end is returned instead. New ids avoid *reserved_ids*.

    Returns the node at the cut.
    """
    points = vessel_path(G, edge)
    place = _place_on_vessel(points, point_um)
    snapped = _snapped_end(edge, place, snap_um)
    if snapped is not None:
        return snapped
    u, v, k = edge
    data = G.edges[u, v, k]
    node = next_node_id(G, set(reserved_ids))
    G.add_node(node, pos=np.asarray(place.point, dtype=float))
    halves = (
        (u, node, _dedupe([*points[: place.segment + 1], place.point])),
        (node, v, _dedupe([place.point, *points[place.segment + 1 :]])),
    )
    for a, b, path in halves:
        attrs = {key: value for key, value in data.items() if key not in ("voxels", "length")}
        if len(path) < 2:
            path = _dedupe([_node_position(G, a), _node_position(G, b)])
        attrs["voxels"] = path
        attrs["length"] = float(calculate_path_length(path))
        half = G.add_edge(a, b, **attrs)
        mark_edited(G, [(a, b, half)])
    G.remove_edge(u, v, k)
    return node


class IntensityCostField:
    """A routing cost from raw intensities, worked out one window at a time.

    The raw-image counterpart of :func:`haemolynx.graph.mask_cost_field`, and
    shaped like its windowed form: ``shape`` plus slicing by a tuple of
    unit-step slices, which is all :func:`haemolynx.graph.astar_path` uses.
    Each window is blurred (:data:`RAW_COST_SMOOTHING_SIGMA_VOX`, with enough
    context that the blur has no edge), scaled between its own
    :data:`RAW_COST_PERCENTILES` and costed ``1 / max(t, RAW_COST_FLOOR)``:
    1 on the brightest voxels, 100 on the dimmest. A cost relative to the
    window keeps it fair to a vessel that is dim across the whole stack, and
    one route is always found within a single window.
    """

    def __init__(
        self,
        raw: Any,
        *,
        sigma_vox: float = RAW_COST_SMOOTHING_SIGMA_VOX,
        percentiles: tuple[float, float] = RAW_COST_PERCENTILES,
        floor: float = RAW_COST_FLOOR,
    ):
        shape = tuple(int(n) for n in np.shape(raw))
        if len(shape) != 3:
            raise ValueError(f"Raw intensities need a 3D (z, y, x) volume, got shape {shape}")
        self.raw = raw
        self.shape = shape
        self.ndim = 3
        self.sigma_vox = float(sigma_vox)
        self.percentiles = tuple(float(p) for p in percentiles)
        self.floor = float(floor)

    def __getitem__(self, key) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        window = tuple(slice(*k.indices(n)) for k, n in zip(key, self.shape))
        if any(k.step != 1 for k in window):
            raise IndexError("IntensityCostField supports unit-step slices only")
        lo = np.array([k.start for k in window])
        hi = np.maximum(np.array([k.stop for k in window]), lo)
        pad = int(np.ceil(3.0 * self.sigma_vox)) if self.sigma_vox > 0 else 0
        plo = np.maximum(lo - pad, 0)
        phi = np.minimum(hi + pad, self.shape)
        crop = np.asarray(
            self.raw[tuple(slice(a, b) for a, b in zip(plo, phi))], dtype=np.float32
        )
        if self.sigma_vox > 0 and crop.size:
            crop = gaussian_filter(crop, self.sigma_vox)
        values = crop[tuple(slice(a - p, b - p) for a, b, p in zip(lo, hi, plo))]
        if not values.size:
            return np.ones(values.shape)
        black, white = np.percentile(values, self.percentiles)
        if not white > black:
            return np.ones(values.shape)
        brightness = np.clip((values - black) / (white - black), self.floor, 1.0)
        return 1.0 / brightness.astype(np.float64)


def trace_path(
    cost_field: Any,
    start_um: Sequence[float],
    end_um: Sequence[float],
    *,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> list[tuple[float, float, float]]:
    """The cheapest path from *start_um* to *end_um* through *cost_field*.

    :func:`haemolynx.graph.astar_path` over a window round the two points --
    *cost_field* is :func:`haemolynx.graph.mask_cost_field` of the segmented
    mask or an :class:`IntensityCostField` of the raw image -- in microns,
    starting and ending exactly on the two points. With no *cost_field*, the
    straight line between them.
    """
    start = np.asarray(start_um, dtype=float)[:3]
    end = np.asarray(end_um, dtype=float)[:3]
    straight = _dedupe([start, end])
    if cost_field is None or len(straight) < 2:
        return straight
    scale = np.asarray(voxel_size_zyx, dtype=float)
    points = voxel_path_to_microns(
        astar_path(cost_field, start / scale, end / scale, voxel_size_zyx=scale), scale
    )
    if len(points) < 2:
        return straight
    return _dedupe([start, *points[1:-1], end])


def smooth_traced_path(
    points_um: Sequence[Sequence[float]],
    *,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    method: str = "taubin",
    iterations: int = 10,
    max_deviation: float | None = None,
) -> tuple[list[tuple[float, float, float]], str]:
    """A traced path with the voxel staircase taken out, as graph building does.

    An A* path steps voxel to voxel, which makes it longer than the vessel
    it follows (by ~7%; see :mod:`haemolynx.graph.smoothing`) -- and the
    vessels the pipeline built were smoothed. The same smoother and the same
    acceptance rule apply, judged against the traced path itself: the result
    may not stray further from it than :func:`haemolynx.graph.smoothing.edge_tolerance_um`
    allows, nor be longer. It must also still pass every traced point within
    that distance: each straight leg between two clicks counts as a bridge
    (no point lies along it), and a chord cutting off a trace that turns back
    on itself lies along the way back. Returns the path and what happened to it
    (``smoothed``, ``relaxed``, ``kept_raw`` or ``too_short``).
    """
    from scipy.spatial import cKDTree

    from .smoothing import DEFAULT_MAX_DEVIATION_UM, _accept, edge_tolerance_um, smooth_polyline

    original = np.asarray(_dedupe(points_um), dtype=float)
    if len(original) < 3:
        return _dedupe(original), "too_short"
    smoothed = smooth_polyline(original, method=method, iterations=iterations)
    tolerance = edge_tolerance_um(
        original,
        max_deviation=DEFAULT_MAX_DEVIATION_UM if max_deviation is None else float(max_deviation),
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
    )
    accepted, outcome = _accept(
        original, smoothed, cKDTree(original), tolerance, must_pass=original
    )
    return _dedupe(accepted), outcome


@dataclass(frozen=True)
class VesselEnd:
    """Where a traced vessel starts or ends: an existing node, or a point on
    an existing vessel, where :func:`add_traced_vessel` will form a new node."""

    node: Any = None
    edge: EdgeKey | None = None
    point_um: tuple[float, float, float] | None = None

    @classmethod
    def at_node(cls, node: Any) -> "VesselEnd":
        return cls(node=node)

    @classmethod
    def on_vessel(cls, edge: EdgeKey, point_um: Sequence[float]) -> "VesselEnd":
        return cls(
            edge=tuple(edge),
            point_um=tuple(float(c) for c in np.asarray(point_um, dtype=float)[:3]),
        )

    @property
    def on_a_vessel(self) -> bool:
        return self.edge is not None

    def position(self, G: nx.MultiGraph) -> np.ndarray:
        """Where it is, in microns: the node's ``pos``, or the point on the vessel."""
        if self.edge is not None:
            return np.asarray(self.point_um, dtype=float)
        return _node_position(G, self.node)


@dataclass(frozen=True)
class TracedVessel:
    """What :func:`add_traced_vessel` did."""

    edge: EdgeKey
    #: Nodes formed on existing vessels, in the order the vessel reaches them.
    new_nodes: tuple[Any, ...]
    #: The vessels those nodes cut in two, as they were before the cut.
    split_vessels: tuple[EdgeKey, ...]
    diameter_um: float | None
    #: What smoothing did to the path (see :func:`smooth_traced_path`), or
    #: None when it was not smoothed.
    smoothing: str | None


def _resolve_end(G: nx.MultiGraph, end: VesselEnd, snap_um: float):
    """``(node, place)``: the node *end* is at, or where on its vessel to cut."""
    if not end.on_a_vessel:
        if end.node not in G:
            raise ValueError(f"Node {end.node!r} is not in the network")
        return end.node, None
    place = _place_on_vessel(vessel_path(G, end.edge), end.point_um)
    snapped = _snapped_end(end.edge, place, snap_um)
    return (snapped, None) if snapped is not None else (None, place)


def add_traced_vessel(
    G: nx.MultiGraph,
    start: VesselEnd,
    end: VesselEnd,
    points_um: Sequence[Sequence[float]],
    *,
    reserved_ids: Iterable[Any] = (),
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    smoothing: Mapping[str, Any] | None = None,
    snap_um: float = SPLIT_SNAP_UM,
) -> TracedVessel:
    """Add the vessel traced along *points_um* from *start* to *end*.

    An end on an existing vessel cuts that vessel there with
    :func:`split_vessel_at` (a new node, two halves each measured again);
    both ends on one vessel cut it twice. The new vessel's path runs from its
    start node to its end node through *points_um* (the traced points; its
    first and last are moved onto the nodes), smoothed like the pipeline's
    own centrelines when *smoothing* -- keyword arguments of
    :func:`smooth_traced_path`: ``method``, ``iterations``, ``max_deviation``
    -- is given, and its ``length`` is measured from that path. It carries
    ``post_processing_added=True`` and, as a manual override, the mean
    diameter of the vessels at its two ends (:func:`mean_incident_diameter`,
    halves of a cut vessel included).

    A vessel from a node back to itself is refused before anything changes.
    New ids avoid *reserved_ids*.
    """
    # Both ends first, so a refusal leaves G as it was.
    start_node, start_place = _resolve_end(G, start, snap_um)
    end_node, end_place = _resolve_end(G, end, snap_um)
    if start_node is not None and start_node == end_node:
        raise ValueError(
            f"This vessel would start and end at node {start_node}: finish it on "
            "a different node or vessel"
        )
    if (
        start_place is not None
        and end_place is not None
        and start.edge == end.edge
        and abs(start_place.along_um - end_place.along_um) <= snap_um
    ):
        raise ValueError(
            "This vessel would start and end at the same point: finish it on a "
            "different node or vessel"
        )
    points = _dedupe(points_um)
    if len(points) < 2:
        raise ValueError("A traced vessel needs at least two points")

    reserved = set(reserved_ids)
    new_nodes: list[Any] = []
    split_vessels: list[EdgeKey] = []
    if start_node is None:
        start_node = split_vessel_at(
            G, start.edge, start.point_um, reserved_ids=reserved, snap_um=snap_um
        )
        new_nodes.append(start_node)
        split_vessels.append(tuple(start.edge))
    if end_node is None:
        edge = tuple(end.edge)
        if not G.has_edge(*edge):
            # The start's cut was on this vessel too: cut whichever half holds the end.
            edge = min(
                ((a, b, k) for a, b, k in G.edges(start_node, keys=True)),
                key=lambda half: float(
                    np.linalg.norm(
                        _place_on_vessel(vessel_path(G, half), end.point_um).point
                        - np.asarray(end.point_um, dtype=float)
                    )
                ),
            )
        end_node = split_vessel_at(G, edge, end.point_um, reserved_ids=reserved, snap_um=snap_um)
        new_nodes.append(end_node)
        split_vessels.append(tuple(end.edge))

    points[0] = tuple(float(c) for c in _node_position(G, start_node))
    points[-1] = tuple(float(c) for c in _node_position(G, end_node))
    points = _dedupe(points)
    if len(points) < 2:
        points = [points[0], points[0]]
    outcome = None
    if smoothing is not None:
        points, outcome = smooth_traced_path(
            points, voxel_size_zyx=voxel_size_zyx, **dict(smoothing)
        )
    diameter = mean_incident_diameter(G, [start_node, end_node])
    edge = add_vessel_between(G, start_node, end_node, points, diameter_um=diameter)
    if outcome is not None:
        G.edges[edge]["centreline_smoothing"] = outcome
    return TracedVessel(
        edge=edge,
        new_nodes=tuple(new_nodes),
        split_vessels=tuple(split_vessels),
        diameter_um=diameter,
        smoothing=outcome,
    )


def prune_disconnected_branches(
    G: nx.MultiGraph,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Drop every vessel of *G* that no inlet-to-outlet path runs along.

    What a deletion leaves behind when it cut the last vessel joining a
    branch to the rest -- a piece that blood cannot cross, which has no
    boundary condition to solve with -- and every dead-end branch, tree or
    loop that reaches no outlet. The rule is
    :func:`haemolynx.graph.remove_vessels_off_inlet_outlet_paths`'s, the
    one ``boundary_handling="remove_disconnected"`` applies; this adds
    ``removed_nodes``, ``removed_vessels`` and ``removed_boundary_nodes``
    (inlets or outlets that sat on a dropped piece) to its counts, and
    refuses to drop everything.
    """
    pruned, stats = remove_vessels_off_inlet_outlet_paths(G, inlet_nodes, outlet_nodes)
    if G.number_of_edges() and not pruned.number_of_edges():
        raise ValueError(
            "No piece of the network has both an inlet and an outlet; pruning "
            "would remove every vessel"
        )
    boundary = dict.fromkeys([*inlet_nodes, *outlet_nodes])
    stats = dict(stats)
    stats["removed_nodes"] = G.number_of_nodes() - pruned.number_of_nodes()
    stats["removed_vessels"] = G.number_of_edges() - pruned.number_of_edges()
    stats["removed_boundary_nodes"] = [n for n in boundary if n in G and n not in pruned]
    if stats["removed_vessels"] or G.graph.get(PENDING):
        mark_edited(pruned)
    return pruned, stats
