"""Validate skeleton connections between positions."""
from typing import Any, List, Tuple, Optional

import numpy as np
import networkx as nx

from ._helpers import get_line_points_3d

#: Edge attributes carried by a vascular graph, with their units.
EDGE_ATTRIBUTE_UNITS = {
    "length": "um",
    "resistance": "Pa.s/m^3",
    "conductance": "m^3/(Pa.s)",
}

#: Removed in favour of the explicit attributes above. ``weight`` meant physical
#: length at graph-build time and conductance after haemodynamics ran, so any
#: consumer reading it got whichever the last writer happened to mean.
FORBIDDEN_EDGE_ATTRIBUTES = ("weight",)


def assert_no_forbidden_edge_attributes(G: Any, *, context: str = "") -> None:
    """Raise if any edge carries a removed, ambiguous attribute such as ``weight``.

    ``weight`` was overloaded: graph construction stored physical length in it
    and haemodynamics later overwrote it with conductance, so statistics read
    conductances back as microns. Use ``length``, ``resistance`` and
    ``conductance`` instead — see :data:`EDGE_ATTRIBUTE_UNITS`.
    """
    if not isinstance(G, nx.Graph):
        return
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for item in edges:
        data = item[-1]
        for name in FORBIDDEN_EDGE_ATTRIBUTES:
            if name in data:
                where = f" in {context}" if context else ""
                raise ValueError(
                    f"Edge {item[:-1]} carries the removed '{name}' attribute{where}. "
                    f"'{name}' was ambiguous — it held physical length before "
                    "haemodynamics and conductance afterwards. Use the explicit "
                    + ", ".join(f"'{k}' ({v})" for k, v in EDGE_ATTRIBUTE_UNITS.items())
                    + " attributes instead."
                )


def validate_skeleton_connection(
    skeleton_data: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    max_gap: float = 3.0,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[bool, Optional[List]]:
    """
    Validate that there's a skeleton path between two positions.
    Returns (is_valid, voxel_path or None).

    Positions are in physical units; *voxel_size* converts them to array
    indices for skeleton look-ups.
    """
    try:
        vs = np.asarray(voxel_size, dtype=float)
        p1 = np.round(np.asarray(pos1, dtype=float) / vs).astype(int)
        p2 = np.round(np.asarray(pos2, dtype=float) / vs).astype(int)
        if not (
            0 <= p1[0] < skeleton_data.shape[0]
            and 0 <= p1[1] < skeleton_data.shape[1]
            and 0 <= p1[2] < skeleton_data.shape[2]
        ) or not (
            0 <= p2[0] < skeleton_data.shape[0]
            and 0 <= p2[1] < skeleton_data.shape[1]
            and 0 <= p2[2] < skeleton_data.shape[2]
        ):
            return False, None

        line_points = get_line_points_3d(p1, p2)
        skeleton_nearby = 0
        for point in line_points:
            region = skeleton_data[
                max(0, point[0] - 1) : min(skeleton_data.shape[0], point[0] + 2),
                max(0, point[1] - 1) : min(skeleton_data.shape[1], point[1] + 2),
                max(0, point[2] - 1) : min(skeleton_data.shape[2], point[2] + 2),
            ]
            if np.any(region > 0):
                skeleton_nearby += 1
        connection_ratio = skeleton_nearby / len(line_points)
        is_valid = connection_ratio > 0.7
        if is_valid:
            return True, line_points
        return False, None
    except Exception:
        return False, None
