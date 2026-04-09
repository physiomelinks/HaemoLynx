"""Boundary-based node selection helpers."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import networkx as nx


def select_boundary_terminal_nodes(
    G: nx.Graph,
    image_shape: tuple[int, ...],
    *,
    edge_percent: float,
    end_percent: float,
    axis: int = 1,
) -> tuple[list[Any], list[Any]]:
    """Select degree-1 nodes in top and bottom image bands along one axis."""
    if not (0.0 <= edge_percent <= 100.0 and 0.0 <= end_percent <= 100.0):
        raise ValueError("edge_percent and end_percent must be in [0, 100].")
    if axis < 0 or axis >= len(image_shape):
        raise ValueError(f"axis={axis} out of bounds for image shape {image_shape}.")

    node_pos = nx.get_node_attributes(G, "pos")
    terminal_nodes = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    if not terminal_nodes:
        return [], []

    axis_size = float(image_shape[axis] - 1)
    top_limit = axis_size * (edge_percent / 100.0)
    bottom_start = axis_size * (1.0 - (end_percent / 100.0))

    def axis_coord(node_id: Any) -> float:
        return float(np.asarray(node_pos[node_id], dtype=float)[axis])

    starting = [node for node in terminal_nodes if axis_coord(node) <= top_limit]
    outputs = [node for node in terminal_nodes if axis_coord(node) >= bottom_start]
    starting_set = set(starting)
    outputs = [node for node in outputs if node not in starting_set]

    starting.sort(key=lambda n: (axis_coord(n), n))
    outputs.sort(key=lambda n: (-axis_coord(n), n))
    return starting, outputs


def _terminal_nodes_with_positions(G: nx.Graph) -> tuple[list[Any], dict[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    terminals = [node for node, degree in G.degree() if degree == 1 and node in node_pos]
    pos = {node: np.asarray(node_pos[node], dtype=float) for node in terminals}
    return terminals, pos


def _all_nodes_with_positions(G: nx.Graph) -> tuple[list[Any], dict[Any, np.ndarray]]:
    node_pos = nx.get_node_attributes(G, "pos")
    nodes = [node for node in G.nodes() if node in node_pos]
    pos = {node: np.asarray(node_pos[node], dtype=float) for node in nodes}
    return nodes, pos


def _normalize_point(point: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(tuple(point), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate, got shape {arr.shape}.")
    return arr


def _normalize_point_with_order(
    point: Iterable[float],
    *,
    name: str,
    coordinate_order: str,
) -> np.ndarray:
    arr = _normalize_point(point, name=name)
    order = str(coordinate_order).strip().lower()
    if order == "xyz":
        return arr
    if order == "zyx":
        # Convert user-provided (z, y, x) into internal (x, y, z).
        return arr[[2, 1, 0]]
    raise ValueError(
        "coordinate_order must be 'xyz' or 'zyx', "
        f"got {coordinate_order!r}."
    )


def _sort_nodes(nodes: Iterable[Any]) -> list[Any]:
    return sorted(set(nodes), key=lambda n: (str(type(n)), str(n)))


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
    starting_nodes_for_distance: Iterable[Any] | None = None,
    distance_from_starting_node: float = 0.0,
    terminal_only: bool = True,
    coordinate_order: str = "xyz",
) -> list[Any]:
    """Select boundary nodes for one role using the specified method."""
    if node_role not in {"input", "output"}:
        raise ValueError("node_role must be 'input' or 'output'.")

    terminals, terminal_pos = _terminal_nodes_with_positions(G)
    all_nodes, all_pos = _all_nodes_with_positions(G)
    if not all_nodes:
        return []
    candidates = terminals if bool(terminal_only) else all_nodes
    pos = terminal_pos if bool(terminal_only) else all_pos
    if not candidates:
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
        available = [node_id for node_id in candidates if node_id not in excluded]
        if not available:
            return []
        for idx, point in enumerate(points):
            target = _normalize_point_with_order(
                point,
                name=f"coordinates[{idx}]",
                coordinate_order=coordinate_order,
            )
            pool = [node_id for node_id in available if node_id not in selected]
            if not pool:
                break
            nearest = min(
                pool,
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
            corner_a = _normalize_point_with_order(
                corners[0],
                name=f"volume_boxes[{idx}][0]",
                coordinate_order=coordinate_order,
            )
            corner_b = _normalize_point_with_order(
                corners[1],
                name=f"volume_boxes[{idx}][1]",
                coordinate_order=coordinate_order,
            )
            lo = np.minimum(corner_a, corner_b)
            hi = np.maximum(corner_a, corner_b)
            normalized_boxes.append((lo, hi))
        selected = []
        for node_id in candidates:
            p = pos[node_id]
            if any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in normalized_boxes):
                selected.append(node_id)
    elif method_norm == "edge_percent":
        start_nodes, out_nodes = select_boundary_terminal_nodes(
            G,
            image_shape,
            edge_percent=edge_percent,
            end_percent=end_percent,
            axis=axis,
        )
        selected = start_nodes if node_role == "input" else out_nodes
    elif method_norm == "degree_1_from_starting":
        if distance_from_starting_node < 0:
            raise ValueError("distance_from_starting_node must be non-negative.")
        node_pos_all = nx.get_node_attributes(G, "pos")
        starting_positions = [
            np.asarray(node_pos_all[node_id], dtype=float)
            for node_id in (starting_nodes_for_distance or [])
            if node_id in node_pos_all
        ]
        if not starting_positions:
            return []
        selected = []
        for node_id in terminals:
            nearest_start_dist = min(
                float(np.linalg.norm(pos[node_id] - start_pos))
                for start_pos in starting_positions
            )
            if nearest_start_dist > distance_from_starting_node:
                selected.append(node_id)
    else:
        raise ValueError(
            "Unknown boundary-node method. Supported methods are: "
            "'coordinates', 'all_degree_1', 'volume', 'edge_percent', "
            "'degree_1_from_starting'."
        )

    return [node for node in _sort_nodes(selected) if node not in excluded]

