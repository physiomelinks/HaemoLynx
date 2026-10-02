"""Boundary-based node selection helpers."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Iterable, Mapping

import numpy as np
import networkx as nx

from ._helpers import sort_nodes

logger = logging.getLogger(__name__)


class BoundaryCoordinateWarning(UserWarning):
    """A configured coordinate did not land on the network it points at.

    The ``coordinates`` method snaps each point to the *nearest* terminal, so
    it never fails -- a point on no vessel at all still selects something, and
    the run goes on to solve a network whose inlets are somewhere else
    entirely. This warning is the only sign, so it says which setting, which
    point, and how far the snap went.
    """


#: How far a point may snap before it is reported, in voxels of the graph's own
#: image. A pick read off a viewer is accurate to a few voxels, and pruning and
#: cluster collapse move a terminal a few more; past ten the point is not
#: describing the terminal it selects. Voxels rather than microns so the same
#: number means the same thing on any stack.
SNAP_WARNING_VOXELS = 10.0

#: Microns to fall back on when the graph carries no voxel size.
SNAP_WARNING_MICRONS = 10.0


def _voxel_size_zyx(G: nx.Graph) -> np.ndarray | None:
    """The per-array-axis voxel size ``build_graph_from_skeleton`` recorded."""
    voxel_size = G.graph.get("voxel_size")
    if voxel_size is None:
        return None
    arr = np.asarray(voxel_size, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return None
    return arr


def _snap_warning_distance(voxel_size_zyx: np.ndarray | None) -> float:
    if voxel_size_zyx is None:
        return SNAP_WARNING_MICRONS
    return SNAP_WARNING_VOXELS * float(np.max(voxel_size_zyx))


def _nearest_distance(pos: Mapping[Any, np.ndarray], terminals: list[Any], point: np.ndarray) -> float:
    return min(float(np.linalg.norm(pos[node_id] - point)) for node_id in terminals)


def _terminal_extent(
    pos: Mapping[Any, np.ndarray], terminals: list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """The corners of the box the terminal nodes occupy, in microns."""
    stacked = np.asarray([pos[node_id] for node_id in terminals], dtype=float)
    return stacked.min(axis=0), stacked.max(axis=0)


def warn_about_coordinate_snaps(
    G: nx.Graph,
    points: list[np.ndarray],
    distances: list[float],
    *,
    setting_name: str,
    pos: Mapping[Any, np.ndarray],
    terminals: list[Any],
) -> None:
    """Report coordinates that snapped to a terminal far from where they point.

    Node ``pos`` is physical microns, and so is every coordinate setting. The
    mistake this catches is a coordinate read off a viewer showing voxel
    indices: on the anisotropic stacks this pipeline is built for it lands at a
    fraction of the depth it should, on no vessel, and snaps to whatever
    terminal happens to be nearest. It is detectable because the *right*
    reading is still available -- multiplying the point by the voxel size lands
    it on a terminal, and by a wide margin -- so that is what is checked, and
    the fix is named in the message.

    A point outside the network's own extent is left alone unless it carries
    that signature: pointing at a corner and taking the terminal nearest it is
    a deliberate way to use this method, and it is always "far" from anything.
    """
    voxel_size = _voxel_size_zyx(G)
    threshold = _snap_warning_distance(voxel_size)
    lo, hi = _terminal_extent(pos, terminals)
    for index, (point, distance) in enumerate(zip(points, distances)):
        if distance <= threshold:
            continue
        as_voxel_distance = None
        if voxel_size is not None:
            as_microns = point * voxel_size
            candidate = _nearest_distance(pos, terminals, as_microns)
            if candidate * 2.0 < distance:
                as_voxel_distance = candidate
        # A point outside the network's own extent is the deliberate idiom
        # "take the terminal nearest this corner", so only the reading that
        # says the units are wrong is worth reporting there.
        inside = bool(np.all(point >= lo) and np.all(point <= hi))
        if as_voxel_distance is None and not inside:
            continue
        message = (
            f"{setting_name}[{index}] = {tuple(round(float(v), 1) for v in point)} "
            f"is {distance:.1f} um from the nearest terminal node, which is the "
            "one it selects. Coordinates are physical (z, y, x) microns, the "
            "same units as node positions."
        )
        if as_voxel_distance is not None:
            message += (
                " Read as voxel indices it would be "
                f"{tuple(round(float(v), 1) for v in point * voxel_size)} um, "
                f"{as_voxel_distance:.1f} um from a terminal -- so this "
                "point looks like a voxel index. Fix: multiply it by the "
                f"voxel size {tuple(round(float(v), 4) for v in voxel_size)}."
            )
        warnings.warn(message, BoundaryCoordinateWarning, stacklevel=2)


def select_boundary_terminal_nodes(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    edge_percent: float,
    end_percent: float,
    axis: int = 1,
) -> tuple[list[Any], list[Any]]:
    """Select the degree-1 nodes in the first and last bands along one axis.

    The bands are measured across the span the terminal nodes themselves cover
    along ``axis``, not across the image. Two reasons, both of which made the
    image-relative version select nothing at all:

    * node ``pos`` is in microns while ``image_shape`` counts voxels, so the
      two were only ever the same numbers at 1 micron voxels;
    * a network rarely reaches the edge of its image -- closing, pruning and
      stub removal all pull terminals inward -- so the first and last few
      percent of the *image* routinely hold no terminal node.

    Measured across the terminals, both lists are non-empty whenever two
    terminals differ along ``axis`` and neither percentage is 100: the lowest
    terminal always falls in the first band and the highest in the last.
    ``image_shape`` is what ``axis`` is checked against.
    """
    if not (0.0 <= edge_percent <= 100.0 and 0.0 <= end_percent <= 100.0):
        raise ValueError("edge_percent and end_percent must be in [0, 100].")
    if axis < 0 or axis >= len(image_shape):
        raise ValueError(f"axis={axis} out of bounds for image shape {image_shape}.")

    node_pos = nx.get_node_attributes(G, "pos")
    terminal_nodes = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    if not terminal_nodes:
        return [], []

    def axis_coord(node_id: Any) -> float:
        return float(np.asarray(node_pos[node_id], dtype=float)[axis])

    axis_coords = [axis_coord(node) for node in terminal_nodes]
    lowest, highest = min(axis_coords), max(axis_coords)
    span = highest - lowest
    top_limit = lowest + span * (edge_percent / 100.0)
    bottom_start = highest - span * (end_percent / 100.0)

    inlets = [node for node in terminal_nodes if axis_coord(node) <= top_limit]
    outlets = [node for node in terminal_nodes if axis_coord(node) >= bottom_start]
    inlet_set = set(inlets)
    outlets = [node for node in outlets if node not in inlet_set]

    inlets.sort(key=lambda n: (axis_coord(n), n))
    outlets.sort(key=lambda n: (-axis_coord(n), n))
    return inlets, outlets


#: How close to an image face a terminal must be, in microns, to be an open
#: end -- a vessel the image cut through -- rather than a dead end inside the
#: tissue. Loose, because a skeleton stops short of the face it runs into by
#: up to the vessel's radius.
DEFAULT_OPEN_END_MAX_DISTANCE_UM = 10.0


def open_terminal_nodes(
    G: nx.Graph, image_shape: tuple[int, ...], *, max_distance_um: float
) -> list[Any] | None:
    """The terminals within *max_distance_um* of an image face, in node order.

    ``None`` when the graph does not record its voxel size (or the image is
    not 3D), so the image's extent in microns is unknown and nothing can be
    said either way.
    """
    voxel_size = _voxel_size_zyx(G)
    if voxel_size is None or len(image_shape) != 3:
        return None
    extent = (np.asarray(image_shape, dtype=float) - 1.0) * voxel_size
    terminals, pos = _terminal_nodes_and_position_map(G)
    return [
        node
        for node in terminals
        if float(np.min(np.minimum(pos[node], extent - pos[node]))) <= float(max_distance_um)
    ]


def _keep_open_ends(
    selected: list[Any],
    open_ends: list[Any] | None,
    *,
    method: str,
    node_role: str,
    max_distance_um: float,
) -> list[Any]:
    """*selected* less its interior dead ends -- or all of it, with a warning,
    when none of it is an open end (the network may simply stop short of
    every face, and no boundary at all would fail the run later instead)."""
    if open_ends is None:
        return selected
    open_set = set(open_ends)
    kept = [node for node in selected if node in open_set]
    dropped = len(selected) - len(kept)
    if kept:
        if dropped:
            logger.info(
                "%s %s nodes: left out %d terminal(s) more than %.3g um from every "
                "image face -- dead ends inside the tissue, not vessels the image cut.",
                method, node_role, dropped, max_distance_um,
            )
        return kept
    if selected:
        logger.warning(
            "%s %s nodes: none of the %d candidate terminal(s) lies within %.3g um of "
            "an image face, so interior dead ends are being used as boundaries. Check "
            "the network reaches the image edge, or raise boundary_open_end_max_distance_um.",
            method, node_role, len(selected), max_distance_um,
        )
    return selected


def _terminal_nodes_and_position_map(G: nx.Graph) -> tuple[list[Any], dict[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    pos = {node: np.asarray(node_pos[node], dtype=float) for node in terminals}
    return terminals, pos


def _normalize_point(point: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(tuple(point), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate, got shape {arr.shape}.")
    return arr


def _resolve_node_id(G: nx.Graph, value: Any) -> Any | None:
    """*value* as the graph node it names, or None when it names none.

    Node IDs are ints, but a hand-edited config can quote one ("12") or a
    numpy-typed list can carry ``12.0``; either still means node 12.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):  # True == 1, but nobody means node 1 by it
        return None
    if isinstance(value, float):
        # Before the lookup: 4.0 finds node 4 (they hash alike), and would
        # then go on through the run as a float key.
        if not value.is_integer():
            return None
        value = int(value)
    try:
        if value in G:
            return value
    except TypeError:  # unhashable, e.g. a list typed where an ID belongs
        return None
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        as_int = int(value.strip())
        return as_int if as_int in G else None
    return None


def select_nodes_by_id(
    G: nx.Graph,
    node_ids: Iterable[Any],
    *,
    setting_name: str = "node_ids",
) -> list[Any]:
    """The graph nodes *node_ids* name, raising for any the graph lacks.

    Any node, not only terminals: this is the method for saying exactly which
    node is meant, and an arteriole/venule boundary -- where a vessel hands
    over to the capillaries -- is usually a junction. A missing ID raises
    rather than being skipped, because a node ID names a node in one build of
    the graph: rebuilding with other settings renumbers them, and a run that
    quietly dropped the stale ones would solve with different boundaries.
    """
    resolved: list[Any] = []
    missing: list[Any] = []
    for value in node_ids:
        node = _resolve_node_id(G, value)
        if node is None:
            missing.append(value)
        else:
            resolved.append(node)
    if missing:
        raise ValueError(
            f"{setting_name} lists node ID(s) {missing} that are not in the graph "
            f"({G.number_of_nodes()} nodes). A node ID names a node in one build "
            "of the graph: changing the Skeletonise or Graph settings, or cutting "
            "the network at the large-vessel volumes, can renumber or remove "
            f"nodes. Fix: pick the nodes again on the graph this run builds, or "
            "choose another selection method."
        )
    return resolved


def select_boundary_nodes_by_method(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    method: str,
    node_role: str,
    coordinates: Iterable[Iterable[float]] | None = None,
    volume_boxes: Iterable[tuple[Iterable[float], Iterable[float]]] | None = None,
    edge_percent: float = 10.0,
    end_percent: float = 10.0,
    axis: int = 1,
    exclude_nodes: Iterable[Any] | None = None,
    inlet_nodes_for_distance: Iterable[Any] | None = None,
    distance_from_inlet_node: float = 0.0,
    coordinates_setting_name: str = "coordinates",
    open_end_max_distance_um: float | None = DEFAULT_OPEN_END_MAX_DISTANCE_UM,
    node_ids: Iterable[Any] | None = None,
    node_ids_setting_name: str = "node_ids",
) -> list[Any]:
    """Select boundary nodes for one role using the specified method.

    ``coordinates_setting_name`` only names the setting the points came from,
    so a :class:`BoundaryCoordinateWarning` can say which one to edit;
    ``node_ids_setting_name`` does the same for ``node_ids``.

    ``node_ids`` takes exactly the nodes listed (see
    :func:`select_nodes_by_id`), terminals or not.

    The two methods that choose terminals by position alone,
    ``edge_percent`` and ``degree_1_from_inlet``, keep only *open ends* --
    terminals within *open_end_max_distance_um* of an image face (see
    :func:`open_terminal_nodes`). A terminal deep inside the tissue is a
    vessel's dead end (a segmentation gap, most often): holding it at a
    boundary pressure pushes flow in or out where none enters the tissue.
    ``None`` turns this off. When a method's candidates hold no open end at
    all, it keeps them all and warns. The other methods take what the user
    placed or asked for, unfiltered.
    """
    if node_role not in {"inlet", "outlet"}:
        raise ValueError("node_role must be 'inlet' or 'outlet'.")

    method_norm = str(method).strip().lower()
    excluded = set(exclude_nodes or [])
    selected: list[Any]

    if method_norm == "node_ids":
        # Before the terminal check: these need not be terminals.
        selected = select_nodes_by_id(G, node_ids or [], setting_name=node_ids_setting_name)
        return [node for node in sort_nodes(selected) if node not in excluded]

    terminals, pos = _terminal_nodes_and_position_map(G)
    if not terminals:
        return []

    if method_norm == "all_degree_1":
        selected = terminals
    elif method_norm == "coordinates":
        points = list(coordinates or [])
        if not points:
            return []
        selected = []
        targets: list[np.ndarray] = []
        distances: list[float] = []
        for idx, point in enumerate(points):
            target = _normalize_point(point, name=f"coordinates[{idx}]")
            nearest = min(
                terminals,
                key=lambda node_id: float(np.linalg.norm(pos[node_id] - target)),
            )
            selected.append(nearest)
            targets.append(target)
            distances.append(float(np.linalg.norm(pos[nearest] - target)))
        warn_about_coordinate_snaps(
            G,
            targets,
            distances,
            setting_name=coordinates_setting_name,
            pos=pos,
            terminals=terminals,
        )
    elif method_norm == "volume":
        boxes = list(volume_boxes or [])
        if not boxes:
            return []
        normalized_boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, box in enumerate(boxes):
            corners = tuple(box)
            if len(corners) != 2:
                raise ValueError(
                    "Each volume box must contain exactly two corner points."
                )
            corner_a = _normalize_point(corners[0], name=f"volume_boxes[{idx}][0]")
            corner_b = _normalize_point(corners[1], name=f"volume_boxes[{idx}][1]")
            lo = np.minimum(corner_a, corner_b)
            hi = np.maximum(corner_a, corner_b)
            normalized_boxes.append((lo, hi))
        selected = []
        for node_id in terminals:
            p = pos[node_id]
            if any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in normalized_boxes):
                selected.append(node_id)
    elif method_norm == "edge_percent":
        inlet_nodes, outlet_nodes = select_boundary_terminal_nodes(
            G,
            image_shape,
            edge_percent=edge_percent,
            end_percent=end_percent,
            axis=axis,
        )
        selected = inlet_nodes if node_role == "inlet" else outlet_nodes
        if open_end_max_distance_um is not None:
            selected = _keep_open_ends(
                selected,
                open_terminal_nodes(G, image_shape, max_distance_um=open_end_max_distance_um),
                method=method_norm,
                node_role=node_role,
                max_distance_um=float(open_end_max_distance_um),
            )
    elif method_norm == "degree_1_from_inlet":
        if distance_from_inlet_node < 0:
            raise ValueError("distance_from_inlet_node must be non-negative.")
        node_pos_all = nx.get_node_attributes(G, "pos")
        inlet_positions = [
            np.asarray(node_pos_all[node_id], dtype=float)
            for node_id in (inlet_nodes_for_distance or [])
            if node_id in node_pos_all
        ]
        if not inlet_positions:
            return []
        selected = []
        for node_id in terminals:
            nearest_start_dist = min(
                float(np.linalg.norm(pos[node_id] - start_pos))
                for start_pos in inlet_positions
            )
            if nearest_start_dist > distance_from_inlet_node:
                selected.append(node_id)
        if open_end_max_distance_um is not None:
            selected = _keep_open_ends(
                selected,
                open_terminal_nodes(G, image_shape, max_distance_um=open_end_max_distance_um),
                method=method_norm,
                node_role=node_role,
                max_distance_um=float(open_end_max_distance_um),
            )
    else:
        raise ValueError(
            "Unknown boundary-node method. Supported methods are: "
            "'coordinates', 'all_degree_1', 'volume', 'edge_percent', "
            "'degree_1_from_inlet', 'node_ids'."
        )

    return [node for node in sort_nodes(selected) if node not in excluded]



#: Config settings naming each boundary role's selection method, coordinates,
#: volume boxes and node IDs, plus the ``node_role`` the selector expects.
BOUNDARY_ROLE_SETTINGS: dict[str, dict[str, str]] = {
    "inlet": {
        "method": "inlet_node_selection_method",
        "coordinates": "inlet_node_coordinates",
        "volume_boxes": "inlet_node_volumes",
        "node_ids": "inlet_node_ids",
        "node_role": "inlet",
    },
    "outlet": {
        "method": "outlet_node_selection_method",
        "coordinates": "outlet_node_coordinates",
        "volume_boxes": "outlet_node_volumes",
        "node_ids": "outlet_node_ids",
        "node_role": "outlet",
    },
    "arteriole_boundary": {
        "method": "arteriole_boundary_selection_method",
        "coordinates": "arteriole_boundary_node_coordinates",
        "volume_boxes": "arteriole_boundary_node_volumes",
        "node_ids": "arteriole_boundary_node_ids",
        "node_role": "inlet",
    },
    "venule_boundary": {
        "method": "venule_boundary_selection_method",
        "coordinates": "venule_boundary_node_coordinates",
        "volume_boxes": "venule_boundary_node_volumes",
        "node_ids": "venule_boundary_node_ids",
        "node_role": "outlet",
    },
    "large_vessel_inlet": {
        "method": "large_vessel_inlet_node_selection_method",
        "coordinates": "large_vessel_inlet_node_coordinates",
        "volume_boxes": "large_vessel_inlet_node_volumes",
        "node_ids": "large_vessel_inlet_node_ids",
        "node_role": "inlet",
    },
    "large_vessel_outlet": {
        "method": "large_vessel_outlet_node_selection_method",
        "coordinates": "large_vessel_outlet_node_coordinates",
        "volume_boxes": "large_vessel_outlet_node_volumes",
        "node_ids": "large_vessel_outlet_node_ids",
        "node_role": "outlet",
    },
}


#: Settings every role shares, mapped to the keyword each one fills in
#: :func:`select_boundary_nodes_by_method`. One axis and one pair of bands
#: describe the whole network, so these are not per-role the way the
#: coordinates and volume boxes are.
BOUNDARY_BAND_SETTINGS: dict[str, str] = {
    "boundary_axis": "axis",
    "boundary_first_percent": "edge_percent",
    "boundary_last_percent": "end_percent",
    "boundary_distance_from_inlet_node": "distance_from_inlet_node",
}


#: Roles whose nodes hold a pressure or a flow in the solve. A node ID picked
#: for one of these off a junction is most often a click that missed the
#: terminal beside it, so it is reported; the arteriole/venule boundaries are
#: where branch ordering stops, and a junction is what they usually are.
_FLOW_BOUNDARY_ROLES = frozenset(
    {"inlet", "outlet", "large_vessel_inlet", "large_vessel_outlet"}
)


def _node_id_list(value: Any) -> list[Any]:
    """A node-ID setting as a list: a lone ID typed without brackets is one."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, int, float, np.generic)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _report_node_id_picks(
    G: nx.Graph,
    role: str,
    setting_name: str,
    requested: list[Any],
    chosen: list[Any],
) -> None:
    """Log what a ``node_ids`` pick did not do as asked.

    A listed node an earlier role already took is left out (inlets before
    outlets before the vessel boundaries, as every method does), which would
    otherwise be silent.
    """
    resolved = select_nodes_by_id(G, requested, setting_name=setting_name)
    kept = set(chosen)
    taken = [node for node in sort_nodes(resolved) if node not in kept]
    if taken:
        logger.warning(
            "%s: node(s) %s already belong to an earlier boundary role, so %s "
            "does not take them.", setting_name, taken, role,
        )
    if role in _FLOW_BOUNDARY_ROLES:
        interior = [node for node in chosen if G.degree(node) != 1]
        if interior:
            logger.warning(
                "%s: node(s) %s are not terminals (degree %s), so the %s "
                "boundary is held inside the network rather than where a vessel "
                "leaves the image. Check they are the nodes you meant.",
                setting_name, interior, [G.degree(node) for node in interior],
                role.replace("_", " "),
            )


def select_boundary_nodes_for_role(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    settings: Mapping[str, Any],
    role: str,
    *,
    exclude_nodes: Iterable[Any] | None = None,
) -> list[Any]:
    """Select one role's boundary nodes from the boundary-assignment settings.

    The roles differ only in which settings they read, so naming the role is
    enough; :data:`BOUNDARY_ROLE_SETTINGS` records which those are.

    A method whose settings hold nothing for it to work with raises here,
    naming the setting to change: it would otherwise return an empty list and
    surface much later as "no boundary nodes found", with nothing to say which
    of the settings was the empty one.
    """
    try:
        names = BOUNDARY_ROLE_SETTINGS[role]
    except KeyError:
        known = ", ".join(sorted(BOUNDARY_ROLE_SETTINGS))
        raise ValueError(f"Unknown boundary role {role!r}. Roles are: {known}.") from None

    method = str(settings[names["method"]]).strip().lower()
    coordinates = list(settings.get(names["coordinates"]) or [])
    volume_boxes = list(settings.get(names["volume_boxes"]) or [])
    node_ids = _node_id_list(settings.get(names["node_ids"]))
    inlet_nodes = list(settings.get("inlet_nodes") or [])

    if method == "node_ids":
        if not node_ids:
            raise ValueError(
                f"{names['method']}='node_ids' takes exactly the nodes listed in "
                f"{names['node_ids']}, but that setting is empty. Fix: list the "
                f"node IDs in {names['node_ids']} (or pick them in the viewer on "
                f"the Boundaries tab), or set {names['method']} to "
                "'edge_percent', which needs no node IDs from this dataset."
            )
        chosen = select_boundary_nodes_by_method(
            G,
            image_shape,
            method=method,
            node_role=names["node_role"],
            node_ids=node_ids,
            node_ids_setting_name=names["node_ids"],
            exclude_nodes=exclude_nodes,
        )
        _report_node_id_picks(G, role, names["node_ids"], node_ids, chosen)
        return chosen

    if method == "coordinates" and not coordinates:
        raise ValueError(
            f"{names['method']}='coordinates' takes the terminals nearest to the "
            f"points in {names['coordinates']}, but that setting is empty. Fix: "
            f"list the coordinates in {names['coordinates']}, or set "
            f"{names['method']} to 'edge_percent', which needs no coordinates "
            "from this dataset."
        )
    if method == "volume" and not volume_boxes:
        raise ValueError(
            f"{names['method']}='volume' takes the terminals inside the boxes in "
            f"{names['volume_boxes']}, but that setting is empty. Fix: list the "
            f"(min corner, max corner) boxes in {names['volume_boxes']}, or set "
            f"{names['method']} to 'edge_percent', which needs no boxes from "
            "this dataset."
        )
    if method == "degree_1_from_inlet" and not inlet_nodes:
        raise ValueError(
            f"{names['method']}='degree_1_from_inlet' measures each terminal's "
            "distance from the inlet nodes, and none have been chosen. Fix: "
            f"set {names['method']} to another method, or pick the inlet "
            "nodes first by giving inlet_node_selection_method one of the "
            "other methods."
        )

    # Settings the config does not carry fall through to the selector's own
    # defaults rather than being repeated here.
    band = {
        keyword: settings[name]
        for name, keyword in BOUNDARY_BAND_SETTINGS.items()
        if settings.get(name) is not None
    }
    # Unlike the band settings, an empty value here means "off", not "the
    # selector's default" -- so an absent setting is the only fall-through.
    if "boundary_open_end_max_distance_um" in settings:
        band["open_end_max_distance_um"] = settings["boundary_open_end_max_distance_um"]
    return select_boundary_nodes_by_method(
        G,
        image_shape,
        method=method,
        node_role=names["node_role"],
        coordinates=coordinates,
        volume_boxes=volume_boxes,
        exclude_nodes=exclude_nodes,
        inlet_nodes_for_distance=inlet_nodes,
        coordinates_setting_name=names["coordinates"],
        **band,
    )
