"""What the "6. Post processing" tab draws and lists, described without napari.

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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

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
    "NEW_VESSEL_POINTS",
    "NEW_VESSEL_TRACE",
    "NetworkScan",
    "POST_PROCESSING_LAYERS",
    "SELECTED",
    "STATUS_COLOURS",
    "TRACE_SOURCES",
    "TRACE_THROUGH_MASK",
    "TRACE_THROUGH_RAW",
    "VesselTrace",
    "added_nodes_layer",
    "added_vessel_ids",
    "branch_id_of",
    "camera_center_for",
    "describe_vessels",
    "edits_lost_by_running_from",
    "junction_label",
    "junction_marker_layer",
    "junction_table_rows",
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
#: Every layer this tab adds, for removing them all at once.
POST_PROCESSING_LAYERS = (HIGH_DEGREE_JUNCTIONS, NEW_VESSEL_TRACE, NEW_VESSEL_POINTS, ADDED_NODES)

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

    The tab comes after Diameters, so a run from Diameters or earlier starts
    again from a graph that never had the edits: the ones still in the tab's
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
