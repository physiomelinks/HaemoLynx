"""What the "10. Post processing" tab draws and lists, described without napari.

Pure, like :mod:`haemolynx.gui.results`: a graph in, colours, layer specs
and table rows out. ``_widget.py`` owns the Qt page,
applies these to the viewer and calls the graph edits in
:mod:`haemolynx.graph.post_processing`.

The tab does not draw a second copy of the network. It recolours the vessels
layer that is already there -- so the colours land on the 3D tubes as well as
the lines, and a click only swaps colours instead of rebuilding a layer:

* grey -- every other vessel;
* orange -- a dead-end vessel, one no inlet-to-outlet path runs through
  (once the dead-end list has been filled);
* cyan -- one of the vessels at the junction being looked at;
* yellow -- a vessel selected in either of the tab's tables.

The one layer of its own, :data:`HIGH_DEGREE_JUNCTIONS`, rings every node
where four or more vessels meet.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from haemolynx.graph.post_processing import (
    JunctionVessel,
    edge_keys,
    high_degree_junctions,
    junction_vessels,
)
from haemolynx.gui.results import PREFIX, LayerSpec, node_points

__all__ = [
    "AT_JUNCTION",
    "CONNECTED",
    "DEAD_END",
    "DEAD_END_TABLE_COLUMNS",
    "HIGH_DEGREE_JUNCTIONS",
    "JUNCTION_TABLE_COLUMNS",
    "NetworkScan",
    "POST_PROCESSING_LAYERS",
    "SELECTED",
    "STATUS_COLOURS",
    "boundaries_following_graph",
    "camera_center_for",
    "dead_end_table_rows",
    "describe_vessels",
    "junction_label",
    "junction_marker_layer",
    "junction_table_rows",
    "nearest_node",
    "parse_branch_ids",
    "scan_network",
    "status_colours",
    "vessel_midpoint",
    "vessel_status",
    "zoom_for_canvas",
]

HIGH_DEGREE_JUNCTIONS = f"{PREFIX}4+ junctions"
#: Every layer this tab adds, for removing them all at once.
POST_PROCESSING_LAYERS = (HIGH_DEGREE_JUNCTIONS,)

CONNECTED = "connected"
DEAD_END = "dead end"
AT_JUNCTION = "at junction"
SELECTED = "selected"
#: RGBA per status, in the precedence :func:`vessel_status` applies them.
STATUS_COLOURS: dict[str, tuple[float, float, float, float]] = {
    SELECTED: (1.0, 0.9, 0.0, 1.0),
    AT_JUNCTION: (0.0, 0.85, 1.0, 1.0),
    DEAD_END: (1.0, 0.4, 0.1, 1.0),
    CONNECTED: (0.62, 0.62, 0.62, 1.0),
}

#: Header of the junction table, one column per :func:`junction_table_rows` cell.
JUNCTION_TABLE_COLUMNS = ("branchID", "length (µm)", "diameter (µm)", "branch order", "other end")
#: Header of the dead-end table, one column per :func:`dead_end_table_rows` cell.
DEAD_END_TABLE_COLUMNS = ("branchID", "length (µm)", "diameter (µm)", "branch order", "nodes")


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
    dead_end: Iterable[int] = (),
    at_junction: Iterable[int] = (),
    selected: Iterable[int] = (),
) -> np.ndarray:
    """Each drawn segment's status, from the branchID it belongs to.

    *edge_index* is the vessels layer's own per-segment ``edge_index`` column.
    Later statuses win: a selected vessel reads as selected even though it is
    also at the junction or a dead end.
    """
    index = np.asarray(edge_index, dtype=int)
    status = np.full(index.shape, CONNECTED, dtype=object)
    for label, ids in ((DEAD_END, dead_end), (AT_JUNCTION, at_junction), (SELECTED, selected)):
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


def describe_vessels(graph: Any, edges: Iterable[tuple[Any, Any, Any]]) -> str:
    """The vessels *edges* as the tab's log names them, in branchID order.

    ``"branchID 12 (node 3-7, 25.1 µm, 5 µm)"`` per vessel, joined by ``"; "``
    -- its branchID, both ends, length and diameter where known. Call it
    before an edit: a deletion renumbers every later branchID.
    """
    index = {key: i for i, key in enumerate(edge_keys(graph))}
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


def dead_end_table_rows(
    graph: Any, edges: Iterable[tuple[Any, Any, Any]]
) -> tuple[list[int], list[tuple[str, ...]]]:
    """The branchIDs of *edges* and their dead-end table cells, in branchID order."""
    index = {key: i for i, key in enumerate(edge_keys(graph))}
    wanted = sorted((index[tuple(e)], tuple(e)) for e in edges if tuple(e) in index)
    ids, rows = [], []
    for branch_id, (u, v, k) in wanted:
        data = graph.edges[u, v, k]
        order = data.get("branch_order")
        ids.append(branch_id)
        rows.append(
            (
                str(branch_id),
                _cell(_finite(data.get("length"))),
                _cell(_finite(data.get("diameter_um"))),
                _cell(None if order is None else str(order)),
                f"{u}-{v}",
            )
        )
    return ids, rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def vessel_midpoint(graph: Any, edge: tuple[Any, Any, Any]) -> np.ndarray | None:
    """A point halfway along a vessel's path (microns), to zoom the viewer to.

    The middle of its ``voxels``, or the midpoint of its two nodes when it
    has none; None when neither is known.
    """
    u, v, k = edge
    voxels = graph.edges[u, v, k].get("voxels")
    if voxels is not None and len(voxels):
        return np.asarray(voxels[len(voxels) // 2], dtype=float)[:3]
    ends = [graph.nodes[n].get("pos") for n in (u, v)]
    if any(pos is None for pos in ends):
        return None
    return np.mean([np.asarray(pos, dtype=float)[:3] for pos in ends], axis=0)


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


def boundaries_following_graph(resume: Any) -> Any:
    """*resume* (a ``PipelineResume``) with its boundary lists cut to its graph.

    A prune in this tab can drop inlets or outlets that sat on a piece cut
    off from the rest; the run being regenerated must not look for them. The
    resistance node pair is re-picked from what is left when either of its
    nodes went. Everything else on *resume* is kept.
    """
    graph = resume.graph
    if graph is None:
        return resume

    def kept(nodes):
        return tuple(node for node in (nodes or ()) if node in graph)

    inlets, outlets = kept(resume.inlet_nodes), kept(resume.outlet_nodes)
    pair = resume.resistance_node_pair
    if pair is None or any(node not in graph for node in pair):
        pair = (inlets[0], outlets[0]) if inlets and outlets else None
    return replace(
        resume,
        inlet_nodes=inlets,
        outlet_nodes=outlets,
        arteriole_boundary_nodes=kept(resume.arteriole_boundary_nodes),
        venule_boundary_nodes=kept(resume.venule_boundary_nodes),
        resistance_node_pair=pair,
    )
