"""What the "7. Post processing" tab draws and lists, described without napari.

Pure, like :mod:`haemolynx.gui.results`: a graph in, colours, layer specs
and table rows out. ``_widget.py`` owns the Qt page,
applies these to the viewer and calls the graph edits in
:mod:`haemolynx.graph.post_processing`.

The tab does not draw a second copy of the network. It recolours the vessels
layer that is already there -- so the colours land on the 3D tubes as well as
the lines, and a click only swaps colours instead of rebuilding a layer:

* grey -- every other vessel;
* green -- a vessel Add vessel drew;
* cyan -- one of the vessels at the junction being looked at;
* yellow -- a vessel selected in the tab's table.

Its own layers: :data:`HIGH_DEGREE_JUNCTIONS` rings every node where four or
more vessels meet; while Add vessel traces, :data:`NEW_VESSEL_TRACE` draws
the path so far and :data:`NEW_VESSEL_POINTS` the clicks along it
(:func:`trace_layers`); and :data:`ADDED_NODES` marks the nodes Add vessel
formed on existing vessels (:func:`added_nodes_layer`).

The Review dropdown (:data:`REVIEW_LISTS`) picks which list a scan makes.
Short loops lists the network's short loops still to review
(:func:`loops_to_review`, :func:`loop_label`), the chosen loop's sides
(:func:`loop_side_rows`) and where the camera goes to look at it
(:func:`loop_view`); the loop is cyan and the side selected yellow, as at a
junction. Each decision is a :class:`LoopReview`, kept with the run and
written as a :func:`loop_review_row` to ``{stem}_loop_review.csv``.

Every other list is one of :mod:`haemolynx.graph.network_review`'s: its
items (:func:`review_label`), the chosen item's vessels
(:func:`review_table_rows`), a ring round each item
(:func:`review_marker_layer`), and the buttons :class:`ReviewList` names.
Each decision is a :class:`ReviewDecision`, kept with the run and written as
a :func:`review_record` to ``{stem}_{kind}_review.csv``. A regenerate is
summed up before and after by :func:`network_summary` and
:func:`summary_change`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from haemolynx.graph.post_processing import (
    APPLIED,
    JunctionVessel,
    VesselEnd,
    edge_keys,
    has_pending_edits,
    high_degree_junctions,
    junction_vessels,
)
from haemolynx.gui.results import PREFIX, LayerSpec, node_points, polylines_to_vectors

__all__ = [
    "ACTION_DELETE_ITEM",
    "ACTION_DELETE_SELECTED",
    "ACTION_JOIN",
    "ACTION_KEEP",
    "ACTION_LABELS",
    "ACTION_MERGE",
    "ACTION_SET_DIAMETER",
    "ACTION_TRACE_ON",
    "NEEDS_BOUNDARIES",
    "NEEDS_FLOW",
    "NEEDS_MASK",
    "NetworkSummary",
    "REVIEW_CSV_COLUMNS",
    "REVIEW_JUNCTIONS",
    "REVIEW_LISTS",
    "REVIEW_LIST_BY_KEY",
    "REVIEW_LIST_BY_LABEL",
    "REVIEW_LOOPS",
    "REVIEW_MARKERS",
    "REVIEW_MATCH_UM",
    "REVIEW_NONE",
    "REVIEW_REGION",
    "REVIEW_TABLE_COLUMNS",
    "ReviewDecision",
    "ReviewList",
    "append_review_record",
    "find_review_items",
    "review_mask_support",
    "review_needs_missing",
    "item_still_there",
    "items_to_review",
    "network_summary",
    "review_box_um",
    "review_csv_path",
    "review_label",
    "review_marker_layer",
    "review_record",
    "review_region_layer",
    "review_table_rows",
    "summary_change",
    "view_centre_zyx",
    "ALL_VESSELS_IN_VIEWER",
    "CONNECTIVITY_EXPORT_CHOICES",
    "INLET_TO_OUTLET_ONLY",
    "default_connectivity_csv_path",
    "ADDED",
    "ADDED_NODES",
    "AT_JUNCTION",
    "CONNECTED",
    "HIGH_DEGREE_JUNCTIONS",
    "JUNCTION_TABLE_COLUMNS",
    "KEPT",
    "LOOP_MATCH_LENGTH_FRACTION",
    "LOOP_MATCH_UM",
    "LOOP_REVIEW_COLUMNS",
    "LOOP_TABLE_COLUMNS",
    "LOOP_ZOOM_MARGIN",
    "LOOP_ZOOM_MIN_BOX_UM",
    "LoopReview",
    "LoopView",
    "NEW_VESSEL_POINTS",
    "NEW_VESSEL_TRACE",
    "NetworkScan",
    "POST_PROCESSING_LAYERS",
    "SELECTED",
    "SIDE_DELETED",
    "STATUS_COLOURS",
    "TRACE_SOURCES",
    "TRACE_THROUGH_MASK",
    "TRACE_THROUGH_RAW",
    "VesselTrace",
    "added_nodes_layer",
    "added_vessel_ids",
    "append_loop_review",
    "branch_id_of",
    "camera_center_for",
    "describe_vessels",
    "edits_lost_by_running_from",
    "junction_label",
    "junction_marker_layer",
    "junction_table_rows",
    "loop_branch_ids",
    "loop_label",
    "loop_review_csv_path",
    "loop_review_row",
    "loop_side_rows",
    "loop_view",
    "loops_to_review",
    "nearest_node",
    "parse_branch_ids",
    "point_on_path_under_click",
    "point_under_click",
    "scan_network",
    "status_colours",
    "trace_layers",
    "vessel_status",
    "zoom_for_canvas",
]

HIGH_DEGREE_JUNCTIONS = f"{PREFIX}4+ junctions"
#: The path of the vessel Add vessel is tracing, drawn after every click.
NEW_VESSEL_TRACE = f"{PREFIX}new vessel trace"
#: Where that vessel starts and each point clicked along it.
NEW_VESSEL_POINTS = f"{PREFIX}new vessel points"
#: The nodes Add vessel formed on existing vessels, until Regenerate.
ADDED_NODES = f"{PREFIX}added nodes"
#: A ring round each item of the review list shown, and the voxels of the
#: piece of uncovered mask chosen in it.
REVIEW_MARKERS = f"{PREFIX}review items"
REVIEW_REGION = f"{PREFIX}review region"
#: Every layer this tab adds, for removing them all at once.
POST_PROCESSING_LAYERS = (
    HIGH_DEGREE_JUNCTIONS, NEW_VESSEL_TRACE, NEW_VESSEL_POINTS, ADDED_NODES, REVIEW_MARKERS, REVIEW_REGION,
)

CONNECTED = "connected"
ADDED = "added"
AT_JUNCTION = "at junction"
SELECTED = "selected"
#: RGBA per status, in the precedence :func:`vessel_status` applies them.
STATUS_COLOURS: dict[str, tuple[float, float, float, float]] = {
    SELECTED: (1.0, 0.9, 0.0, 1.0),
    AT_JUNCTION: (0.0, 0.85, 1.0, 1.0),
    ADDED: (0.35, 1.0, 0.35, 1.0),
    CONNECTED: (0.62, 0.62, 0.62, 1.0),
}

#: What Add vessel traces through, as the tab's dropdown lists them.
TRACE_THROUGH_MASK = "Segmented mask"
TRACE_THROUGH_RAW = "Raw data"
TRACE_SOURCES = (TRACE_THROUGH_MASK, TRACE_THROUGH_RAW)

#: Colours of the trace in progress: the path and the points clicked along
#: it are orange; a node that will form (or has formed) on an existing
#: vessel is green, like the vessels Add vessel draws; an existing node the
#: trace starts from is white.
TRACE_COLOUR = (1.0, 0.55, 0.0, 1.0)
NEW_NODE_COLOUR = STATUS_COLOURS[ADDED]
START_NODE_COLOUR = (1.0, 1.0, 1.0, 1.0)
#: Marker size (microns): a little larger than a node, so it rings one.
TRACE_POINT_SIZE = 5.0
TRACE_LINE_WIDTH = 1.2

#: Header of the junction table, one column per :func:`junction_table_rows` cell.
JUNCTION_TABLE_COLUMNS = ("branchID", "length (µm)", "diameter (µm)", "branch order", "other end")


@dataclass(frozen=True)
class NetworkScan:
    """One look at the network: what the tab lists."""

    junctions: tuple[Any, ...]
    vessel_count: int

    @property
    def summary(self) -> str:
        return (
            f"{len(self.junctions)} junction(s) where 4+ vessels meet, "
            f"in {self.vessel_count} vessels."
        )


def scan_network(graph: Any) -> NetworkScan:
    """The 4+ junctions of *graph*."""
    return NetworkScan(
        junctions=tuple(high_degree_junctions(graph)),
        vessel_count=len(edge_keys(graph)),
    )


def vessel_status(
    edge_index: Sequence[int],
    *,
    at_junction: Iterable[int] = (),
    selected: Iterable[int] = (),
    added: Iterable[int] = (),
) -> np.ndarray:
    """Each drawn segment's status, from the branchID it belongs to.

    *edge_index* is the vessels layer's own per-segment ``edge_index`` column.
    A selected vessel reads as selected even though it is also at the
    junction, and a vessel at the junction as at the junction even though
    Add vessel drew it (*added*).
    """
    index = np.asarray(edge_index, dtype=int)
    status = np.full(index.shape, CONNECTED, dtype=object)
    for label, ids in ((ADDED, added), (AT_JUNCTION, at_junction), (SELECTED, selected)):
        chosen = np.fromiter((int(i) for i in ids), dtype=int)
        if chosen.size:
            status[np.isin(index, chosen)] = label
    return status


def status_colours(status: Sequence[str]) -> np.ndarray:
    """(n, 4) RGBA for :func:`vessel_status`'s labels."""
    labels = np.asarray(status, dtype=object)
    colours = np.empty((len(labels), 4), dtype=float)
    for label, rgba in STATUS_COLOURS.items():
        colours[labels == label] = rgba
    return colours


def added_vessel_ids(graph: Any) -> list[int]:
    """branchIDs of the vessels Add vessel drew (``post_processing_added``)."""
    return [
        branch_id
        for branch_id, (_u, _v, _k, data) in enumerate(graph.edges(keys=True, data=True))
        if data.get("post_processing_added")
    ]


def _branch_index(graph: Any) -> dict[tuple[Any, Any, Any], int]:
    """branchID by edge key, under both ``(u, v, key)`` and ``(v, u, key)``."""
    index: dict[tuple[Any, Any, Any], int] = {}
    for i, (u, v, k) in enumerate(edge_keys(graph)):
        index.setdefault((v, u, k), i)
        index[(u, v, k)] = i
    return index


def branch_id_of(graph: Any, edge: Sequence[Any]) -> int | None:
    """The branchID of vessel *edge*, named either way round; None if it has none.

    ``G.edges(keys=True)`` lists an edge from whichever end comes first in
    the graph's node order, not necessarily the way it was added.
    """
    return _branch_index(graph).get(tuple(edge))


def describe_vessels(graph: Any, edges: Iterable[tuple[Any, Any, Any]]) -> str:
    """The vessels *edges* as the tab's log names them, in branchID order.

    ``"branchID 12 (node 3-7, 25.1 µm, 5 µm)"`` per vessel, joined by ``"; "``
    -- its branchID, both ends, length and diameter where known. Call it
    before an edit: a deletion renumbers every later branchID.
    """
    index = _branch_index(graph)
    wanted = sorted((index[tuple(e)], tuple(e)) for e in edges if tuple(e) in index)
    parts = []
    for branch_id, (u, v, k) in wanted:
        data = graph.edges[u, v, k]
        details = [f"node {u}-{v}"]
        for attr in ("length", "diameter_um"):
            value = data.get(attr)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                details.append(f"{number:.3g} µm")
        parts.append(f"branchID {branch_id} ({', '.join(details)})")
    return "; ".join(parts)


def junction_label(graph: Any, node: Any, decision: str | None = None) -> str:
    """One line of the junction list: its id, how many vessels, what was decided."""
    text = f"Node {node} - {graph.degree(node)} vessels"
    return f"{text} ({decision})" if decision else text


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def junction_table_rows(graph: Any, node: Any) -> tuple[list[JunctionVessel], list[tuple[str, ...]]]:
    """The vessels at *node* and their table cells, in the same order."""
    vessels = junction_vessels(graph, node)
    rows = [
        (
            str(vessel.branch_id),
            _cell(vessel.length_um),
            _cell(vessel.diameter_um),
            _cell(vessel.branch_order),
            str(vessel.other_node),
        )
        for vessel in vessels
    ]
    return vessels, rows


def junction_marker_layer(graph: Any, scan: NetworkScan) -> LayerSpec:
    """A magenta ring on each node where four or more vessels meet."""
    points, ids = node_points(graph, scan.junctions)
    return LayerSpec(
        kind="points",
        name=HIGH_DEGREE_JUNCTIONS,
        data=points,
        features={
            "node_id": ids,
            "degree": np.asarray([graph.degree(n) for n in ids], dtype=float),
        },
        options={
            "size": 8.0,
            "face_color": "transparent",
            "border_color": "magenta",
            "border_width": 0.2,
            "out_of_slice_display": True,
        },
    )


def parse_branch_ids(text: str, vessel_count: int) -> list[int]:
    """The branchIDs typed into the tab, e.g. ``"12, 40 311"``, in order, once each.

    Commas, semicolons and spaces all separate IDs. Raises ``ValueError``
    naming what is wrong: nothing typed, something that is not a whole
    number, or an ID the network does not have (0 to *vessel_count* - 1).
    """
    tokens = [t for t in str(text).replace(",", " ").replace(";", " ").split() if t]
    if not tokens:
        raise ValueError("Type one or more branchIDs, e.g. 12, 40")
    ids: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError:
            raise ValueError(f"{token!r} is not a branchID (a whole number)") from None
        if not 0 <= value < vessel_count:
            raise ValueError(
                f"branchID {value} is not in this network (0 to {vessel_count - 1})"
            )
        if value not in ids:
            ids.append(value)
    return ids


#: How far, in microns, a click may land from a node and still pick it.
NODE_PICK_DISTANCE_UM = 3.0


def nearest_node(
    graph: Any,
    position: Sequence[float],
    *,
    view_direction: Sequence[float] | None = None,
    dims: Sequence[int] | None = None,
    max_distance: float = NODE_PICK_DISTANCE_UM,
) -> Any | None:
    """The node a click at *position* (microns, ``(z, y, x)``) landed on.

    In 3D the click is a ray along *view_direction*: the node nearest that
    ray wins, so what is under the cursor is picked however deep it sits.
    Otherwise the plain distance over the displayed *dims* counts. None when
    no node is within *max_distance*. Worked out from the graph itself, not
    from napari's point picking, which needs the layer drawn on screen.
    """
    points, ids = node_points(graph)
    if not len(points):
        return None
    point = np.asarray(position, dtype=float)[-3:]
    axes = [a for a in (dims or (0, 1, 2)) if 0 <= int(a) < 3]
    delta = points[:, axes] - point[axes]
    view = None if view_direction is None else np.asarray(view_direction, dtype=float)[-3:]
    if view is not None and len(axes) == 3 and np.linalg.norm(view) > 1e-12:
        view = view / np.linalg.norm(view)
        delta = delta - np.outer(delta @ view, view)
    distance = np.linalg.norm(delta, axis=1)
    best = int(np.argmin(distance))
    return ids[best] if distance[best] <= max_distance else None


def _view_ray(view_direction: Sequence[float] | None, axes: Sequence[int]) -> np.ndarray | None:
    """The unit view direction of a 3D click, or None for a 2D one."""
    if view_direction is None or len(axes) != 3:
        return None
    view = np.asarray(view_direction, dtype=float)[-3:]
    norm = float(np.linalg.norm(view))
    return view / norm if norm > 1e-12 else None


def _flattener(view_direction: Sequence[float] | None, dims: Sequence[int] | None):
    """Map (n, 3) points onto what the screen shows, as :func:`nearest_node` does.

    In 3D, across the view ray (whatever lies under the cursor, at any depth,
    lands on the cursor); otherwise onto the displayed axes.
    """
    axes = [a for a in (dims or (0, 1, 2)) if 0 <= int(a) < 3]
    view = _view_ray(view_direction, axes)

    def flatten(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if view is not None:
            return points - np.outer(points @ view, view)
        return points[:, axes]

    return flatten


def point_on_path_under_click(
    path_um: Sequence[Sequence[float]],
    position: Sequence[float],
    *,
    view_direction: Sequence[float] | None = None,
    dims: Sequence[int] | None = None,
) -> np.ndarray:
    """The point of a vessel's path under a click at *position*, in microns.

    Where on a clicked vessel a new node forms: the point of *path_um* that
    looks nearest the cursor -- measured across the view ray in 3D, or over
    the displayed axes in 2D -- which may lie between two of its points.
    """
    points = np.asarray(path_um, dtype=float)[:, :3]
    if len(points) < 2:
        return points[0].copy()
    flatten = _flattener(view_direction, dims)
    a, b = points[:-1], points[1:]
    fa, fb = flatten(a), flatten(b)
    click = flatten(np.asarray(position, dtype=float)[-3:][None, :])[0]
    d = fb - fa
    length_sq = np.einsum("ij,ij->i", d, d)
    safe = np.where(length_sq > 0, length_sq, 1.0)
    t = np.where(
        length_sq > 0, np.clip(np.einsum("ij,ij->i", click - fa, d) / safe, 0.0, 1.0), 0.0
    )
    i = int(np.argmin(np.linalg.norm(fa + t[:, None] * d - click, axis=1)))
    return a[i] + t[i] * (b[i] - a[i])


def point_under_click(
    position: Sequence[float],
    *,
    view_direction: Sequence[float] | None = None,
    dims: Sequence[int] | None = None,
    volume: Any = None,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    pick: str = "first",
    near_um: Sequence[float] | None = None,
) -> np.ndarray:
    """Where a click that missed every node and vessel is, in microns.

    In 2D, the click itself, in the slice on screen. In 3D a click is a ray,
    and its depth comes from *volume* (voxel-indexed, *voxel_size_zyx*
    apart): with *pick* ``"first"`` (a segmented mask) the middle of the
    first run of mask voxels along the ray -- the vessel in front; with
    ``"brightest"`` (raw data) the brightest voxel along it -- what a
    maximum-intensity view shows there. With no volume, or a ray that meets
    no vessel, the point on the ray nearest *near_um* (the trace's last
    point), so the trace carries on at the depth it had.
    """
    click = np.asarray(position, dtype=float)[-3:]
    axes = [a for a in (dims or (0, 1, 2)) if 0 <= int(a) < 3]
    view = _view_ray(view_direction, axes)
    if view is None:
        return click
    if volume is not None:
        scale = np.asarray(voxel_size_zyx, dtype=float)
        shape = np.asarray(np.shape(volume)[-3:])
        lo, hi = -0.5 * scale, (shape - 0.5) * scale
        enter, leave = -np.inf, np.inf
        for axis in range(3):
            if abs(view[axis]) < 1e-12:
                if not lo[axis] <= click[axis] <= hi[axis]:
                    enter, leave = np.inf, -np.inf
                continue
            t1 = (lo[axis] - click[axis]) / view[axis]
            t2 = (hi[axis] - click[axis]) / view[axis]
            enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
        if enter < leave:
            ts = np.arange(enter, leave, 0.5 * float(scale.min()))
            samples = click + ts[:, None] * view
            index = np.clip(np.round(samples / scale).astype(int), 0, shape - 1)
            values = np.asarray(volume[tuple(index.T)], dtype=float)
            if pick == "brightest":
                if values.size and np.nanmax(values) > np.nanmin(values):
                    return samples[int(np.nanargmax(values))]
            else:
                inside = values > 0
                if inside.any():
                    first = int(np.argmax(inside))
                    rest = inside[first:]
                    run = int(np.argmin(rest)) if not rest.all() else len(rest)
                    return samples[first + (run - 1) // 2]
    if near_um is not None:
        near = np.asarray(near_um, dtype=float)[:3]
        return click + float(np.dot(near - click, view)) * view
    return click


@dataclass
class VesselTrace:
    """A vessel Add vessel is tracing: where it starts and its path so far.

    The network is not touched until the trace finishes (see
    :func:`haemolynx.graph.add_traced_vessel`), so giving up leaves nothing
    behind; until then :func:`trace_layers` draws it.
    """

    start: VesselEnd
    #: The path so far, in microns, from the start through every waypoint.
    points_um: list[tuple[float, float, float]]
    #: The points clicked after the start, in order.
    waypoints_um: list[tuple[float, float, float]] = field(default_factory=list)
    #: What each segment was traced through, in order.
    routes: list[str] = field(default_factory=list)

    @classmethod
    def starting_at(cls, start: VesselEnd, graph: Any) -> "VesselTrace":
        point = tuple(float(c) for c in start.position(graph))
        return cls(start=start, points_um=[point])

    @property
    def last_point(self) -> tuple[float, float, float]:
        return self.points_um[-1]

    def extended(self, segment_um: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
        """The path with *segment_um*, which starts at :attr:`last_point`, on the end."""
        rest = [tuple(float(c) for c in p) for p in list(segment_um)[1:]]
        return [*self.points_um, *rest]

    def extend(
        self, segment_um: Sequence[Sequence[float]], waypoint_um: Sequence[float], route: str
    ) -> None:
        """Trace on to *waypoint_um* along *segment_um*, found through *route*."""
        self.points_um = self.extended(segment_um)
        self.waypoints_um.append(tuple(float(c) for c in waypoint_um))
        self.routes.append(route)


def trace_layers(trace: VesselTrace) -> tuple[LayerSpec, ...]:
    """What the viewer draws of *trace*: its path once it has one, and its points.

    The start is green when a new node will form there (it is on a vessel)
    and white when it is an existing node; every point clicked since is
    orange, like the path through them.
    """
    specs: list[LayerSpec] = []
    path = np.asarray(trace.points_um, dtype=float)
    if len(path) >= 2:
        vectors, _owner = polylines_to_vectors([path])
        specs.append(
            LayerSpec(
                kind="vectors",
                name=NEW_VESSEL_TRACE,
                data=vectors,
                options={
                    "vector_style": "line",
                    "edge_width": TRACE_LINE_WIDTH,
                    "edge_color": list(TRACE_COLOUR),
                    "out_of_slice_display": True,
                },
            )
        )
    start_role = "new node" if trace.start.on_a_vessel else "start node"
    roles = [start_role] + ["waypoint"] * len(trace.waypoints_um)
    colour = {
        "new node": NEW_NODE_COLOUR,
        "start node": START_NODE_COLOUR,
        "waypoint": TRACE_COLOUR,
    }
    specs.append(
        LayerSpec(
            kind="points",
            name=NEW_VESSEL_POINTS,
            data=np.asarray([trace.points_um[0], *trace.waypoints_um], dtype=float),
            features={"role": np.asarray(roles, dtype=object)},
            options={
                "size": TRACE_POINT_SIZE,
                "face_color": np.asarray([colour[r] for r in roles], dtype=float),
                "border_color": "black",
                "border_width": 0.1,
                "out_of_slice_display": True,
            },
        )
    )
    return tuple(specs)


def added_nodes_layer(graph: Any, nodes: Iterable[Any]) -> LayerSpec:
    """A green marker on each of *nodes* still in *graph*: the nodes Add vessel
    formed where a new vessel met an existing one."""
    points, ids = node_points(graph, [n for n in dict.fromkeys(nodes) if n in graph])
    return LayerSpec(
        kind="points",
        name=ADDED_NODES,
        data=points,
        features={"node_id": ids},
        options={
            "size": TRACE_POINT_SIZE,
            "face_color": list(NEW_NODE_COLOUR),
            "border_color": "black",
            "border_width": 0.1,
            "out_of_slice_display": True,
        },
    )


def camera_center_for(
    position_zyx: Sequence[float], displayed: Sequence[int]
) -> tuple[float, float, float]:
    """napari ``camera.center`` that puts *position_zyx* mid-screen.

    The camera's centre is given in the displayed dims' own order (which a
    snap to XZ or YZ permutes), padded at the front to three values when only
    two dims are displayed -- napari reads the last two in 2D.
    """
    centre = [float(position_zyx[axis]) for axis in displayed]
    while len(centre) < 3:
        centre.insert(0, 0.0)
    return tuple(centre[-3:])  # type: ignore[return-value]


def zoom_for_canvas(canvas_size_px: Sequence[float], box_um: float) -> float | None:
    """napari camera zoom (screen pixels per micron) that fits a *box_um* box.

    Worked out from the canvas size directly rather than by resetting the view
    first, which redraws the whole scene once more on every click. None when
    either is unusable, so the caller keeps its zoom.
    """
    sizes = [float(v) for v in canvas_size_px if float(v) > 0]
    if not sizes or box_um <= 0:
        return None
    return min(sizes) / float(box_um)


def edits_lost_by_running_from(
    start_from: str | None, working_graph: Any, post_processed_graph: Any
) -> bool:
    """Whether a run starting at *start_from* would throw hand edits away.

    The tab comes after Haemodynamics, so a run from Haemodynamics or earlier
    starts again from a graph that never had the edits: the ones still in the tab's
    *working_graph*, and the ones the ``post_process`` stage already brought
    into the network (*post_processed_graph*, its checkpoint's, marked
    :data:`~haemolynx.graph.post_processing.APPLIED`).
    """
    from haemolynx.pipeline.progress import STAGES

    order = [stage.call for stage in STAGES if stage.call]
    if start_from not in order:
        return False
    if order.index(start_from) >= order.index("post_process"):
        return False
    if has_pending_edits(working_graph):
        return True
    graph_attrs = getattr(post_processed_graph, "graph", None) or {}
    return bool(graph_attrs.get(APPLIED))


def default_connectivity_csv_path(values: Any) -> str:
    """Suggested name for the connectivity CSV: beside the run's VTK output.

    ``{stem}_connectivity.csv`` in the folder of ``vtk_output_prefix`` when that
    folder exists on this machine (a run loaded from another one may name a
    folder that does not), otherwise just the filename.
    """
    from pathlib import Path

    from haemolynx.gui.stage_checkpoints import output_dir_from_prefix

    prefix = values.get("vtk_output_prefix") if values else None
    stem = Path(str(prefix)).name if prefix else "haemolynx"
    name = f"{stem}_connectivity.csv"
    folder = output_dir_from_prefix(prefix)
    return str(folder / name) if folder is not None and folder.is_dir() else name


#: The Export tab's connectivity choices, in dropdown order.
ALL_VESSELS_IN_VIEWER = "All vessels in the viewer"
INLET_TO_OUTLET_ONLY = "Only vessels between an inlet and an outlet"
CONNECTIVITY_EXPORT_CHOICES = (ALL_VESSELS_IN_VIEWER, INLET_TO_OUTLET_ONLY)


# --- Manual loop review ---------------------------------------------------------

#: Header of the table of the chosen loop's sides, one column per
#: :func:`loop_side_rows` cell.
LOOP_TABLE_COLUMNS = ("side", "branchIDs", "length (µm)", "diameter (µm)", "from node", "to node")

#: What was decided about a loop: kept as it is, or opened by deleting a side.
KEPT = "kept"
SIDE_DELETED = "side deleted"

#: A listed loop is one already reviewed when its centre is within this many
#: microns of the reviewed one's and its length within this many microns or
#: :data:`LOOP_MATCH_LENGTH_FRACTION` of it, whichever is more. Positions, not
#: node ids: deleting a side merges the vessels left at its ends, which
#: renumbers the edges of the loops beside it.
LOOP_MATCH_UM = 1.0
LOOP_MATCH_LENGTH_FRACTION = 0.05

#: The smallest box (microns) zooming to a loop fits on screen.
LOOP_ZOOM_MIN_BOX_UM = 20.0
#: The box is the loop's widest extent times this, so its surroundings show.
LOOP_ZOOM_MARGIN = 1.6


@dataclass(frozen=True)
class LoopReview:
    """What a person decided about one short loop, kept with the run."""

    centre_um: tuple[float, float, float]
    length_um: float
    #: :data:`KEPT` or :data:`SIDE_DELETED`.
    decision: str
    #: The side deleted, numbered from 1 as the table lists them.
    side: int | None = None
    #: When it was decided (ISO 8601, to the second).
    time: str = ""

    @classmethod
    def of(cls, loop: Any, decision: str, *, side: int | None = None, time: str = "") -> "LoopReview":
        return cls(
            centre_um=tuple(float(c) for c in loop.centre_um),
            length_um=float(loop.length_um),
            decision=str(decision),
            side=None if side is None else int(side),
            time=str(time),
        )

    def matches(self, loop: Any) -> bool:
        """Whether *loop* (a :class:`~haemolynx.graph.ShortLoop`) is the one reviewed."""
        offset = np.linalg.norm(np.asarray(loop.centre_um, dtype=float) - np.asarray(self.centre_um))
        tolerance = max(LOOP_MATCH_UM, LOOP_MATCH_LENGTH_FRACTION * float(self.length_um))
        return bool(offset <= LOOP_MATCH_UM and abs(float(loop.length_um) - self.length_um) <= tolerance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "centre_um": list(self.centre_um),
            "length_um": self.length_um,
            "decision": self.decision,
            "side": self.side,
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LoopReview | None":
        """One review back from :meth:`as_dict`; None for anything unreadable."""
        try:
            centre = tuple(float(c) for c in data["centre_um"])
            if len(centre) != 3:
                return None
            side = data.get("side")
            return cls(
                centre_um=centre,  # type: ignore[arg-type]
                length_um=float(data["length_um"]),
                decision=str(data["decision"]),
                side=None if side is None else int(side),
                time=str(data.get("time") or ""),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


def loops_to_review(loops: Iterable[Any], reviews: Iterable[LoopReview]) -> list[Any]:
    """*loops* less the ones a review kept, in the same order.

    A loop whose side was deleted is listed again if it is back -- a run
    from an earlier stage rebuilt the network without that edit.
    """
    kept = [review for review in reviews if review.decision == KEPT]
    return [loop for loop in loops if not any(review.matches(loop) for review in kept)]


def loop_label(loop: Any, number: int) -> str:
    """One line of the loop list: its place in it, length and number of sides."""
    return f"Loop {number}: {loop.length_um:.3g} µm round, {len(loop.sides)} side(s)"


def _side_ids(index: dict, side: Iterable[Sequence[Any]]) -> list[int]:
    return [index[tuple(edge)] for edge in side if tuple(edge) in index]


def loop_branch_ids(graph: Any, loop: Any, side: int | None = None) -> list[int]:
    """branchIDs of *loop*'s vessels still in *graph* -- of side *side* only
    (numbered from 0) when given."""
    index = _branch_index(graph)
    if side is None:
        return _side_ids(index, loop.edges)
    if not 0 <= side < len(loop.sides):
        return []
    return _side_ids(index, loop.sides[side])


def loop_side_rows(graph: Any, loop: Any) -> list[tuple[str, ...]]:
    """The table cells of *loop*'s sides, in order round it: number, its
    vessels' branchIDs, summed length, length-weighted diameter and the
    nodes it runs between."""
    index = _branch_index(graph)
    rows = []
    for number, side in enumerate(loop.sides, start=1):
        lengths, diameters = [], []
        for edge in side:
            data = graph.edges[tuple(edge)] if graph.has_edge(*edge) else {}
            try:
                length = float(data.get("length"))
            except (TypeError, ValueError):
                length = float("nan")
            try:
                diameter = float(data.get("diameter_um"))
            except (TypeError, ValueError):
                diameter = float("nan")
            lengths.append(length)
            diameters.append(diameter)
        length = np.asarray(lengths, dtype=float)
        diameter = np.asarray(diameters, dtype=float)
        usable = np.isfinite(length) & np.isfinite(diameter) & (length > 0)
        rows.append(
            (
                str(number),
                ", ".join(str(i) for i in _side_ids(index, side)),
                _cell(float(np.nansum(length))) if np.isfinite(length).any() else "",
                _cell(float(np.average(diameter[usable], weights=length[usable]))) if usable.any() else "",
                str(side[0][0]),
                str(side[-1][1]),
            )
        )
    return rows


@dataclass(frozen=True)
class LoopView:
    """Where the camera goes to look at a loop: physical ``(z, y, x)``."""

    centre_um: tuple[float, float, float]
    #: The box, in microns, to fit on screen.
    box_um: float
    #: In a 3D view, look along the loop's normal with this up: unit vectors.
    view_direction: tuple[float, float, float]
    up_direction: tuple[float, float, float]


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def loop_view(
    points_um: Sequence[Sequence[float]],
    *,
    towards: Sequence[float] | None = None,
    up: Sequence[float] | None = None,
) -> LoopView:
    """The view of a loop in its own plane: centred on it, fitting its
    widest extent with room round it, looking along its normal.

    The normal is turned to agree with *towards* (the camera's current view
    direction, so a zoom from one loop to the next does not flip the scene),
    and up is *up* laid into the loop's plane, or the loop's long axis when
    *up* is along its normal.
    """
    from haemolynx.graph.lumen_loops import loop_centre_um

    points = np.asarray(points_um, dtype=float).reshape(-1, 3)
    centre = loop_centre_um(points)
    centred = points - centre
    span = float(np.linalg.norm(centred, axis=1).max()) * 2.0 if len(points) else 0.0
    box = max(LOOP_ZOOM_MIN_BOX_UM, LOOP_ZOOM_MARGIN * span)
    normal, long_axis = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if len(points) >= 3 and np.linalg.norm(centred) > 1e-9:
        _u, _s, axes = np.linalg.svd(centred, full_matrices=False)
        normal, long_axis = axes[2], axes[0]
    view = normal
    reference = None if towards is None else _unit(np.asarray(towards, dtype=float)[-3:])
    if reference is not None and float(np.dot(view, reference)) < 0:
        view = -view
    in_plane = None
    if up is not None:
        wanted = np.asarray(up, dtype=float)[-3:]
        in_plane = _unit(wanted - np.dot(wanted, view) * view)
    if in_plane is None:
        in_plane = _unit(long_axis - np.dot(long_axis, view) * view)
    return LoopView(
        centre_um=tuple(float(c) for c in centre),  # type: ignore[arg-type]
        box_um=float(box),
        view_direction=tuple(float(c) + 0.0 for c in view),  # type: ignore[arg-type]
        up_direction=tuple(float(c) + 0.0 for c in in_plane),  # type: ignore[arg-type]
    )


#: Columns of ``{stem}_loop_review.csv``, one row per decision: when and
#: what, the loop's vessels, and the measures of
#: :func:`haemolynx.graph.lumen_loops.loop_measures` a rule can be fitted to.
LOOP_REVIEW_COLUMNS = (
    "time",
    "decision",
    "side_deleted",
    "wall_side",
    "loop_length_um",
    "sides",
    "loop_branch_ids",
    "loop_edges",
    "centre_z_um",
    "centre_y_um",
    "centre_x_um",
    "centre_zyx_vox",
    "span_um",
    "area_um2",
    "out_of_plane",
    "path_in",
    "lumen_r_um",
    "chords_in_lumen",
    "chord_bg_median_um",
    "inside_one_lumen",
    "side_lengths_um",
    "side_diameters_um",
    "side_lumen_r_um",
    "zero_resistance_bridges",
    "recovered_um",
    "bridged_um",
)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}" if np.isfinite(value) else ""
    if isinstance(value, (list, tuple)):
        return ";".join(_csv_value(v) for v in value)
    return str(value)


def loop_review_row(
    graph: Any,
    loop: Any,
    review: LoopReview,
    *,
    mask: Any = None,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    max_background_gap_um: float | None = None,
) -> dict[str, str]:
    """One ``{stem}_loop_review.csv`` row for *review* of *loop*, measured on
    *graph* as it was when the decision was made (before a deletion): the
    measures need the side that is about to go."""
    from haemolynx.graph.lumen_loops import loop_measures
    from haemolynx.graph.post_processing import vessel_path
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    present = [side for side in loop.sides if all(graph.has_edge(*e) for e in side)]
    sides_um = (
        [np.vstack([vessel_path(graph, tuple(e)) for e in side]) for side in loop.sides]
        if len(present) == len(loop.sides) else []
    )
    options: dict[str, Any] = {"mask": mask, "voxel_size_zyx": voxel_size_zyx, "sides_um": sides_um}
    if max_background_gap_um is not None:
        options["max_background_gap_um"] = float(max_background_gap_um)
    measures = loop_measures(loop.points_um, **options)
    rows = loop_side_rows(graph, loop)
    edges = [graph.edges[tuple(e)] for e in loop.edges if graph.has_edge(*e)]

    def length(data) -> float:
        try:
            return float(data.get("length") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    voxel = np.asarray(voxel_size_zyx, dtype=float)
    centre_vox = np.rint(np.asarray(loop.centre_um, dtype=float) / voxel).astype(int)
    wall = measures.get("wall_side")
    row: dict[str, Any] = {
        "time": review.time,
        "decision": review.decision,
        "side_deleted": review.side,
        "wall_side": None if wall is None else int(wall) + 1,
        "sides": len(loop.sides),
        "loop_branch_ids": loop_branch_ids(graph, loop),
        "loop_edges": ";".join(f"{a}-{b}" for a, b, _k in loop.edges),
        "centre_zyx_vox": " ".join(str(int(c)) for c in centre_vox),
        "side_lengths_um": [r[2] for r in rows],
        "side_diameters_um": [r[3] for r in rows],
        "zero_resistance_bridges": sum(bool(d.get(IS_ZERO_RESISTANCE)) for d in edges),
        "recovered_um": sum(length(d) for d in edges if d.get("recovered")),
        "bridged_um": sum(length(d) for d in edges if d.get("bridge_kind")),
    }
    for name in LOOP_REVIEW_COLUMNS:
        if name not in row:
            row[name] = measures.get(name)
    return {name: _csv_value(row[name]) for name in LOOP_REVIEW_COLUMNS}


def loop_review_csv_path(values: Any) -> "Path | None":
    """Where the loop decisions are written: ``{stem}_loop_review.csv`` in the
    folder of ``vtk_output_prefix``. None when there is no such folder on
    this machine, as for a run loaded from another one before it is run."""
    from pathlib import Path

    from haemolynx.gui.stage_checkpoints import output_dir_from_prefix

    prefix = values.get("vtk_output_prefix") if values else None
    if not prefix:
        return None
    folder = output_dir_from_prefix(prefix)
    if folder is None or not folder.is_dir():
        return None
    return Path(folder) / f"{Path(str(prefix)).name}_loop_review.csv"


def append_loop_review(path: Any, rows: Iterable[dict[str, str]]) -> "Path":
    """Add *rows* to the CSV at *path*, writing its header first when the
    file is new or empty: every session's decisions collect in one file."""
    import csv
    from pathlib import Path

    target = Path(path)
    new = not target.is_file() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOOP_REVIEW_COLUMNS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in LOOP_REVIEW_COLUMNS})
    return target


# --- Review lists ----------------------------------------------------------------

#: The Review dropdown's entries that are not a :mod:`haemolynx.graph.network_review`
#: kind: nothing listed, and the two lists with pages of their own.
REVIEW_NONE = "none"
REVIEW_JUNCTIONS = "junctions_4plus"
REVIEW_LOOPS = "short_loops"

#: What a list needs from the run before it can be made.
NEEDS_MASK = "the segmented image in the viewer"
NEEDS_BOUNDARIES = "inlet and outlet nodes"
NEEDS_FLOW = "a solved network"

#: The buttons a review list offers for its chosen item.
ACTION_DELETE_SELECTED = "delete_selected"
ACTION_DELETE_ITEM = "delete_item"
ACTION_MERGE = "merge"
ACTION_TRACE_ON = "trace_on"
ACTION_JOIN = "join"
ACTION_SET_DIAMETER = "set_diameter"
ACTION_KEEP = "keep"
ACTION_LABELS = {
    ACTION_DELETE_SELECTED: "Delete selected vessels",
    ACTION_DELETE_ITEM: "Delete all its vessels",
    ACTION_MERGE: "Merge into one junction",
    ACTION_TRACE_ON: "Trace on from the tip",
    ACTION_JOIN: "Join the two ends",
    ACTION_SET_DIAMETER: "Set diameter of selected vessels to",
    ACTION_KEEP: "Keep",
}


@dataclass(frozen=True)
class ReviewList:
    """One entry of the Review dropdown."""

    key: str
    label: str
    #: What it cannot be made without (``NEEDS_*``).
    needs: tuple[str, ...] = ()
    #: Its item's buttons (``ACTION_*``), in the order they are shown.
    actions: tuple[str, ...] = ()

    @property
    def uses_mask(self) -> bool:
        return NEEDS_MASK in self.needs


def _review_lists() -> tuple[ReviewList, ...]:
    from haemolynx.graph import network_review as nr

    return (
        ReviewList(REVIEW_NONE, "None (edit by clicking only)"),
        ReviewList(REVIEW_JUNCTIONS, "4+ junctions"),
        ReviewList(REVIEW_LOOPS, "Short loops"),
        ReviewList(nr.CLOSE_JUNCTIONS, "Junctions close together", (),
                   (ACTION_MERGE, ACTION_DELETE_SELECTED, ACTION_KEEP)),
        ReviewList(nr.HAIRPINS, "Hairpin vessels", (), (ACTION_DELETE_SELECTED, ACTION_KEEP)),
        ReviewList(nr.DEAD_ENDS, "Dead ends", (NEEDS_MASK,),
                   (ACTION_DELETE_ITEM, ACTION_TRACE_ON, ACTION_KEEP)),
        ReviewList(nr.TWO_IN_ONE_LUMEN, "Two vessels in one lumen", (NEEDS_MASK,),
                   (ACTION_DELETE_SELECTED, ACTION_KEEP)),
        ReviewList(nr.FACING_ENDS, "Facing dead ends not joined", (NEEDS_MASK,), (ACTION_JOIN, ACTION_KEEP)),
        ReviewList(nr.UNCOVERED_MASK, "Segmented vessels without a centreline", (NEEDS_MASK,), (ACTION_KEEP,)),
        ReviewList(nr.UNSOLVED_PIECES, "Unsolved pieces", (NEEDS_BOUNDARIES,),
                   (ACTION_DELETE_ITEM, ACTION_KEEP)),
        ReviewList(nr.FLOW_AGAINST_ORDER, "Flow against branch order", (NEEDS_FLOW,),
                   (ACTION_DELETE_SELECTED, ACTION_KEEP)),
        ReviewList(nr.FLOW_OUTLIERS, "Flow outliers", (NEEDS_FLOW,), (ACTION_SET_DIAMETER, ACTION_KEEP)),
        ReviewList(nr.BOUNDARY_NODES, "Boundary nodes", (NEEDS_BOUNDARIES,), (ACTION_KEEP,)),
        ReviewList(nr.UNMEASURED_DIAMETERS, "Unmeasured diameters", (), (ACTION_SET_DIAMETER, ACTION_KEEP)),
        ReviewList(nr.DIAMETER_JUMPS, "Diameter jumps", (), (ACTION_SET_DIAMETER, ACTION_KEEP)),
    )


#: Every entry of the Review dropdown, in its order; the first is the default.
REVIEW_LISTS: tuple[ReviewList, ...] = _review_lists()
REVIEW_LIST_BY_KEY = {entry.key: entry for entry in REVIEW_LISTS}
REVIEW_LIST_BY_LABEL = {entry.label: entry for entry in REVIEW_LISTS}

#: Header of the table of a review item's vessels.
REVIEW_TABLE_COLUMNS = ("branchID", "length (µm)", "diameter (µm)", "diameter source", "branch order", "flow share")


def _measure(item: Any, name: str, default: Any = None) -> Any:
    value = (item.measures or {}).get(name, default)
    return default if value is None else value


def _g(value: Any) -> str:
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def review_label(item: Any, number: int) -> str:
    """One line of a review list: its place in it and what it is, briefly."""
    from haemolynx.graph import network_review as nr

    kind, m = item.kind, item.measures or {}
    nodes = ", ".join(str(n) for n in item.nodes)
    if kind == nr.CLOSE_JUNCTIONS:
        text = f"Nodes {nodes}: {_g(m.get('connector_um'))} µm apart, {m.get('vessels')} vessels"
    elif kind == nr.HAIRPINS:
        text = f"{_g(m.get('length_um'))} µm vessel, ends {_g(m.get('chord_um'))} µm apart"
    elif kind == nr.DEAD_ENDS:
        kinds = str(m.get("kinds") or "none of the artefact kinds").replace("_", " ").replace(";", ", ")
        text = f"Node {item.nodes[0]}: {kinds}, {_g(m.get('length_um'))} µm"
    elif kind == nr.TWO_IN_ONE_LUMEN:
        text = f"Two vessels side by side, the shorter {_g(m.get('shorter_um'))} µm"
    elif kind == nr.FACING_ENDS:
        text = f"Nodes {nodes}: {_g(m.get('gap_um'))} µm apart, {m.get('refused') or 'joinable'}"
    elif kind == nr.UNCOVERED_MASK:
        text = f"{_g(m.get('volume_um3'))} µm³ of mask, up to {_g(2 * float(m.get('radius_um') or 0))} µm across"
    elif kind == nr.UNSOLVED_PIECES:
        where = f"off node {m.get('attached_at')}" if m.get("attached_at") else "on its own"
        text = f"{m.get('vessels')} vessel(s), {_g(m.get('length_um'))} µm, {where}"
    elif kind == nr.FLOW_AGAINST_ORDER:
        text = f"Node {nodes}: {str(m.get('orders', '')).replace(';', ', ')}"
    elif kind == nr.FLOW_OUTLIERS:
        z = float(m.get("robust_z") or 0.0)
        text = (
            f"{_g(m.get('velocity_mm_s'))} mm/s in a {str(m.get('class', '')).replace('_', ' ')}"
            + ("" if not np.isfinite(z) else f" ({z:+.2g} SD)")
        )
    elif kind == nr.BOUNDARY_NODES:
        text = f"Node {nodes} ({m.get('role') or 'no role'}): {m.get('issue')}"
    elif kind == nr.UNMEASURED_DIAMETERS:
        share = m.get("flow_share")
        text = f"{m.get('diameter_source') or 'flagged'} width" + (
            f", {float(share):.1%} of the flow" if share is not None else ", unsolved"
        )
    elif kind == nr.DIAMETER_JUMPS:
        text = f"Node {nodes}: ×{_g(m.get('ratio'))} ({m.get('at')})"
    else:
        text = item.why
    return f"{number}. {text}"


def review_table_rows(
    graph: Any, item: Any, *, flow_reference: float | None = None
) -> tuple[list[tuple[Any, Any, Any]], list[tuple[str, ...]]]:
    """The vessels of *item* still in *graph*, and their table cells in the
    same order. *flow_reference* is what a flow share is a share of (the
    network's inflow)."""
    index = _branch_index(graph)
    edges, rows = [], []
    for edge in item.edges:
        edge = tuple(edge)
        if not graph.has_edge(*edge):
            continue
        data = graph.edges[edge]
        flow = data.get("flow_abs")
        try:
            share = float(flow) / float(flow_reference) if flow_reference else None
        except (TypeError, ValueError):
            share = None
        edges.append(edge)
        rows.append((
            str(index.get(edge, "")),
            _cell(_optional_number(data.get("length"))),
            _cell(_optional_number(data.get("diameter_um"))),
            str(data.get("diameter_source") or ""),
            str(data.get("branch_order") or ""),
            "" if share is None or not np.isfinite(share) else f"{share:.2%}",
        ))
    return edges, rows


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def review_box_um(item: Any) -> float:
    """The box, in microns, zooming to *item* fits on screen: its points'
    widest reach from its centre, with room round it."""
    points = np.asarray(getattr(item, "points_um", ()), dtype=float).reshape(-1, 3)
    if not len(points):
        return LOOP_ZOOM_MIN_BOX_UM
    span = 2.0 * float(np.linalg.norm(points - np.asarray(item.centre_um), axis=1).max())
    return max(LOOP_ZOOM_MIN_BOX_UM, LOOP_ZOOM_MARGIN * span)


def review_marker_layer(items: Sequence[Any]) -> LayerSpec:
    """A magenta ring round each item of a review list."""
    centres = np.asarray([item.centre_um for item in items], dtype=float).reshape(-1, 3)
    return LayerSpec(
        kind="points",
        name=REVIEW_MARKERS,
        data=centres,
        features={"item": np.arange(1, len(centres) + 1, dtype=float)},
        options={
            "size": 8.0,
            "face_color": "transparent",
            "border_color": "magenta",
            "border_width": 0.2,
            "out_of_slice_display": True,
        },
    )


def review_region_layer(item: Any) -> LayerSpec:
    """The voxels of the chosen piece of uncovered mask, in orange."""
    points = np.asarray(getattr(item, "points_um", ()), dtype=float).reshape(-1, 3)
    return LayerSpec(
        kind="points",
        name=REVIEW_REGION,
        data=points,
        features={},
        options={"size": 1.0, "face_color": TRACE_COLOUR, "border_width": 0.0, "out_of_slice_display": True},
    )


def view_centre_zyx(
    camera_center: Sequence[float], displayed: Sequence[int], dims_point: Sequence[float]
) -> tuple[float, float, float]:
    """The point mid-screen, physical ``(z, y, x)``: the camera centre on the
    displayed dims (in their order, as :func:`camera_center_for` writes it)
    and the slider's point on the others. The inverse of
    :func:`camera_center_for`."""
    point = [float(c) for c in dims_point]
    centre = [float(c) for c in camera_center][-len(displayed):] if displayed else []
    for axis, value in zip(displayed, centre):
        if 0 <= int(axis) < len(point):
            point[int(axis)] = value
    return tuple(point[-3:])  # type: ignore[return-value]


#: A listed item is one already decided when it is of the same kind and its
#: centre lies within this many microns of the decided one's.
REVIEW_MATCH_UM = 1.0


@dataclass(frozen=True)
class ReviewDecision:
    """What a person decided about one item of a review list, kept with the run."""

    kind: str
    centre_um: tuple[float, float, float]
    #: :data:`KEPT`, or what was done (``"deleted"``, ``"merged"`` ...).
    decision: str
    time: str = ""

    @classmethod
    def of(cls, item: Any, decision: str, *, time: str = "") -> "ReviewDecision":
        return cls(
            kind=str(item.kind),
            centre_um=tuple(float(c) for c in item.centre_um),  # type: ignore[arg-type]
            decision=str(decision),
            time=str(time),
        )

    def matches(self, item: Any) -> bool:
        if str(item.kind) != self.kind:
            return False
        offset = np.linalg.norm(np.asarray(item.centre_um, dtype=float) - np.asarray(self.centre_um))
        return bool(offset <= REVIEW_MATCH_UM)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "centre_um": list(self.centre_um), "decision": self.decision, "time": self.time}

    @classmethod
    def from_dict(cls, data: Any) -> "ReviewDecision | None":
        """One decision back from :meth:`as_dict`; None for anything unreadable."""
        try:
            centre = tuple(float(c) for c in data["centre_um"])
            if len(centre) != 3:
                return None
            return cls(
                kind=str(data["kind"]),
                centre_um=centre,  # type: ignore[arg-type]
                decision=str(data["decision"]),
                time=str(data.get("time") or ""),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None


def items_to_review(items: Iterable[Any], decisions: Iterable[ReviewDecision]) -> list[Any]:
    """*items* less the ones a decision kept, in the same order. An item
    whose vessels were edited is listed again if it is back."""
    kept = [decision for decision in decisions if decision.decision == KEPT]
    return [item for item in items if not any(decision.matches(item) for decision in kept)]


def item_still_there(graph: Any, item: Any) -> bool:
    """Whether every vessel and node *item* names is still in *graph*: an
    edit since the scan can take it away."""
    return all(graph.has_edge(*tuple(edge)) for edge in item.edges) and all(n in graph for n in item.nodes)


#: The first columns of ``{stem}_{kind}_review.csv``; each item's own
#: measures follow, so a rule can be fitted to the decisions later.
REVIEW_CSV_COLUMNS = (
    "time",
    "decision",
    "kind",
    "centre_z_um",
    "centre_y_um",
    "centre_x_um",
    "centre_zyx_vox",
    "nodes",
    "branch_ids",
    "why",
)


def review_record(
    graph: Any,
    item: Any,
    decision: ReviewDecision,
    *,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, str]:
    """One CSV row for *decision* about *item*, read off *graph* before the
    edit it records changes it."""
    index = _branch_index(graph)
    centre = np.asarray(item.centre_um, dtype=float)
    voxel = np.rint(centre / np.asarray(voxel_size_zyx, dtype=float)).astype(int)
    row: dict[str, Any] = {
        "time": decision.time,
        "decision": decision.decision,
        "kind": item.kind,
        "centre_z_um": centre[0],
        "centre_y_um": centre[1],
        "centre_x_um": centre[2],
        "centre_zyx_vox": " ".join(str(int(c)) for c in voxel),
        "nodes": ";".join(str(n) for n in item.nodes),
        "branch_ids": [index[tuple(e)] for e in item.edges if tuple(e) in index],
        "why": item.why,
    }
    for name, value in (item.measures or {}).items():
        row.setdefault(str(name), value)
    return {name: _csv_value(value) for name, value in row.items()}


def review_csv_path(values: Any, kind: str) -> "Path | None":
    """Where a review list's decisions are written: ``{stem}_{kind}_review.csv``
    beside the VTK output; None when there is no such folder on this machine."""
    path = loop_review_csv_path(values)
    if path is None:
        return None
    stem = path.name[: -len("_loop_review.csv")]
    return path.with_name(f"{stem}_{kind}_review.csv")


def append_review_record(path: Any, row: dict[str, str]) -> "Path":
    """Add *row* to the CSV at *path*, its header the row's own columns when
    the file is new, else the file's (a column it lacks is left out)."""
    import csv
    from pathlib import Path

    target = Path(path)
    header = list(row)
    if target.is_file() and target.stat().st_size > 0:
        with target.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), header) or header
        new = False
    else:
        new = True
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in header})
    return target


# --- Before and after a regenerate ------------------------------------------------


@dataclass(frozen=True)
class NetworkSummary:
    """The few numbers that say how a regenerate changed the network."""

    vessels: int
    length_um: float
    #: Vessels on no inlet-to-outlet path; None without inlets and outlets.
    unsolved: int | None
    #: Flow in through the inlets (m^3/s); None before a solve.
    inflow: float | None
    #: Inlet-to-outlet pressure drop over the inflow (Pa.s/m^3).
    resistance: float | None


def network_summary(graph: Any, inlets: Sequence[Any], outlets: Sequence[Any]) -> NetworkSummary:
    """:class:`NetworkSummary` of *graph*, its flows read from the solve."""
    from haemolynx.graph.connectivity import inlet_to_outlet_vessels
    from haemolynx.graph.network_review import total_inflow

    length = 0.0
    for _u, _v, data in graph.edges(data=True):
        value = _optional_number(data.get("length"))
        length += value or 0.0
    inlets = [n for n in inlets if n in graph]
    outlets = [n for n in outlets if n in graph]
    unsolved = None
    if inlets and outlets:
        unsolved = graph.number_of_edges() - len(inlet_to_outlet_vessels(graph, inlets, outlets))
    inflow = total_inflow(graph, inlets)
    resistance = None
    pressures_in = [_optional_number(graph.nodes[n].get("pressure")) for n in inlets]
    pressures_out = [_optional_number(graph.nodes[n].get("pressure")) for n in outlets]
    if inflow and pressures_in and pressures_out and None not in pressures_in + pressures_out:
        drop = float(np.mean(pressures_in)) - float(np.mean(pressures_out))
        resistance = drop / inflow if drop else None
    return NetworkSummary(graph.number_of_edges(), length, unsolved, inflow, resistance)


def summary_change(before: NetworkSummary, after: NetworkSummary) -> str:
    """One line saying what a regenerate changed, e.g.
    ``"vessels 1,234 → 1,220; unsolved 40 → 12; network resistance 1.2e+13 →
    1.1e+13 Pa·s/m³ (−8.3%)"``."""

    def change(name: str, a: Any, b: Any, fmt: str, unit: str = "", relative: bool = False) -> str | None:
        if a is None or b is None:
            return None
        text = f"{name} {format(a, fmt)} → {format(b, fmt)}{unit}"
        if relative and a:
            text += f" ({(b - a) / a:+.1%})".replace("-", "−")
        return text

    parts = [
        change("vessels", before.vessels, after.vessels, ","),
        change("length", before.length_um, after.length_um, ",.0f", " µm"),
        change("unsolved", before.unsolved, after.unsolved, ","),
        change("inflow", before.inflow, after.inflow, ".3g", " m³/s", relative=True),
        change("network resistance", before.resistance, after.resistance, ".3g", " Pa·s/m³", relative=True),
    ]
    return "; ".join(p for p in parts if p)


# --- Making a review list ------------------------------------------------------


def review_needs_missing(
    entry: ReviewList, graph: Any, *, has_mask: bool, inlets: Sequence[Any], outlets: Sequence[Any]
) -> str | None:
    """What *entry* needs that this run cannot give it (a ``NEEDS_*``), or None."""
    if NEEDS_MASK in entry.needs and not has_mask:
        return NEEDS_MASK
    if NEEDS_BOUNDARIES in entry.needs and not (
        any(n in graph for n in inlets) and any(n in graph for n in outlets)
    ):
        return NEEDS_BOUNDARIES
    if NEEDS_FLOW in entry.needs and not any(
        _optional_number(data.get("flow_abs")) is not None for _u, _v, data in graph.edges(data=True)
    ):
        return NEEDS_FLOW
    return None


def review_mask_support(image: Any, voxel_size_zyx: Sequence[float], settings: Any = None):
    """The segmented image as the review lists judge vessels against it: a
    ``MaskSupport`` binarised the way the loaders do, held to the run's own
    bridge settings."""
    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.preprocessing.bridge_mask_support import (
        DEFAULT_MAX_BACKGROUND_GAP_UM,
        DEFAULT_MIN_MASK_FRACTION,
        MaskSupport,
    )

    values = settings or {}
    gap = _optional_number(values.get("bridge_max_background_gap_um"))
    fraction = _optional_number(values.get("bridge_min_mask_fraction"))
    return MaskSupport(
        _to_binary_volume_for_skeletonization(np.asanyarray(image)),
        tuple(float(v) for v in voxel_size_zyx),
        max_background_gap_um=DEFAULT_MAX_BACKGROUND_GAP_UM if gap is None else gap,
        min_mask_fraction=DEFAULT_MIN_MASK_FRACTION if fraction is None else fraction,
    )


def find_review_items(
    key: str,
    graph: Any,
    *,
    support: Any = None,
    roles: Mapping[str, Sequence[Any]] | None = None,
    settings: Any = None,
    extent_um: Sequence[float] | None = None,
    z_range: tuple[int, int] | None = None,
) -> list[Any]:
    """Make review list *key* (one of :data:`~haemolynx.graph.network_review.REVIEW_KINDS`)
    on *graph*, with the run's own settings where a list reads one:
    ``min_stub_length_radius_multiple`` (dead ends), ``facing_dead_end_max_gap_um``
    and ``recovery_min_region_volume_um3``. *support* is the segmented image
    (:func:`review_mask_support`), *roles* the boundary nodes by role,
    *extent_um* the image's size in microns and *z_range* the slices a mask
    scan reads."""
    from haemolynx.graph import network_review as nr

    values = settings or {}
    roles = roles or {}
    inlets = tuple(roles.get("inlet") or ())
    outlets = tuple(roles.get("outlet") or ())
    if key == nr.CLOSE_JUNCTIONS:
        return nr.find_close_junctions(graph)
    if key == nr.HAIRPINS:
        return nr.find_hairpins(graph)
    if key == nr.DEAD_ENDS:
        multiple = _optional_number(values.get("min_stub_length_radius_multiple"))
        return nr.find_dead_ends(graph, support, stub_radius_multiple=3.0 if multiple is None else multiple)
    if key == nr.TWO_IN_ONE_LUMEN:
        return nr.find_two_in_one_lumen(graph, support)
    if key == nr.FACING_ENDS:
        return nr.find_facing_ends(graph, support, max_gap_um=_optional_number(values.get("facing_dead_end_max_gap_um")))
    if key == nr.UNCOVERED_MASK:
        return nr.find_uncovered_mask(
            graph, support,
            min_region_volume_um3=_optional_number(values.get("recovery_min_region_volume_um3")),
            z_range=z_range,
        )
    if key == nr.UNSOLVED_PIECES:
        return nr.find_unsolved_pieces(graph, inlets, outlets)
    if key == nr.FLOW_AGAINST_ORDER:
        return nr.find_flow_against_branch_order(graph, inlets)
    if key == nr.FLOW_OUTLIERS:
        return nr.find_flow_outliers(graph)
    if key == nr.BOUNDARY_NODES:
        return nr.find_boundary_issues(graph, roles, extent_um=extent_um)
    if key == nr.UNMEASURED_DIAMETERS:
        return nr.find_unmeasured_diameters(graph, inlets)
    if key == nr.DIAMETER_JUMPS:
        return nr.find_diameter_jumps(graph)
    raise ValueError(f"{key!r} is not a review list")
