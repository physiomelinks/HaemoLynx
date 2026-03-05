"""Boundary-based node selection helpers."""
from __future__ import annotations

from typing import Any

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

