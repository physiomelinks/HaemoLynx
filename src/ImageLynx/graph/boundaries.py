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


def _normalize_point(point: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(tuple(point), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate, got shape {arr.shape}.")
    return arr


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
) -> list[Any]:
    """Select boundary nodes for one role using the specified method."""
    if node_role not in {"input", "output"}:
        raise ValueError("node_role must be 'input' or 'output'.")

    terminals, pos = _terminal_nodes_with_positions(G)
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
        start_nodes, out_nodes = select_boundary_terminal_nodes(
            G,
            image_shape,
            edge_percent=edge_percent,
            end_percent=end_percent,
            axis=axis,
        )
        selected = start_nodes if node_role == "input" else out_nodes
    else:
        raise ValueError(
            "Unknown boundary-node method. Supported methods are: "
            "'coordinates', 'all_degree_1', 'volume', 'edge_percent'."
        )

    return [node for node in _sort_nodes(selected) if node not in excluded]

