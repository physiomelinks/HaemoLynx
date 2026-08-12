"""Boundary-based node selection helpers."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import networkx as nx

from ._helpers import sort_nodes


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
) -> list[Any]:
    """Select boundary nodes for one role using the specified method."""
    if node_role not in {"inlet", "outlet"}:
        raise ValueError("node_role must be 'inlet' or 'outlet'.")

    terminals, pos = _terminal_nodes_and_position_map(G)
    if not terminals:
        return []

    method_norm = str(method).strip().lower()
    excluded = set(exclude_nodes or [])
    selected: list[Any]

    if method_norm == "all_degree_1":
        selected = terminals
    elif method_norm == "coordinates":
        points = list(coordinates or [])
        if not points:
            return []
        selected = []
        for idx, point in enumerate(points):
            target = _normalize_point(point, name=f"coordinates[{idx}]")
            nearest = min(
                terminals,
                key=lambda node_id: float(np.linalg.norm(pos[node_id] - target)),
            )
            selected.append(nearest)
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
    else:
        raise ValueError(
            "Unknown boundary-node method. Supported methods are: "
            "'coordinates', 'all_degree_1', 'volume', 'edge_percent', "
            "'degree_1_from_inlet'."
        )

    return [node for node in sort_nodes(selected) if node not in excluded]



#: Config settings naming each boundary role's selection method, coordinates and
#: volume boxes, plus the ``node_role`` the selector expects.
BOUNDARY_ROLE_SETTINGS: dict[str, dict[str, str]] = {
    "inlet": {
        "method": "inlet_node_selection_method",
        "coordinates": "inlet_node_coordinates",
        "volume_boxes": "inlet_node_volumes",
        "node_role": "inlet",
    },
    "outlet": {
        "method": "outlet_node_selection_method",
        "coordinates": "outlet_node_coordinates",
        "volume_boxes": "outlet_node_volumes",
        "node_role": "outlet",
    },
    "arteriole_boundary": {
        "method": "arteriole_boundary_selection_method",
        "coordinates": "arteriole_boundary_node_coordinates",
        "volume_boxes": "arteriole_boundary_node_volumes",
        "node_role": "inlet",
    },
    "venule_boundary": {
        "method": "venule_boundary_selection_method",
        "coordinates": "venule_boundary_node_coordinates",
        "volume_boxes": "venule_boundary_node_volumes",
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


def select_boundary_nodes_for_role(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    settings: Mapping[str, Any],
    role: str,
    *,
    exclude_nodes: Iterable[Any] | None = None,
) -> list[Any]:
    """Select one role's boundary nodes from the boundary-assignment settings.

    The four roles differ only in which settings they read, so naming the role
    is enough; :data:`BOUNDARY_ROLE_SETTINGS` records which those are.

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
    inlet_nodes = list(settings.get("inlet_nodes") or [])

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
    return select_boundary_nodes_by_method(
        G,
        image_shape,
        method=method,
        node_role=names["node_role"],
        coordinates=coordinates,
        volume_boxes=volume_boxes,
        exclude_nodes=exclude_nodes,
        inlet_nodes_for_distance=inlet_nodes,
        **band,
    )
