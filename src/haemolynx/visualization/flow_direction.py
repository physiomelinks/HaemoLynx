"""Per-edge flow direction arrows from haemodynamic edge attributes.

Ports the direction logic of
:func:`haemolynx.visualization.large_vessel_assignment.visualize_3d_plotly_large_vessel_assignment_flow_direction`
into pure geometry that stays in physical ``(z, y, x)`` microns -- the same
frame as node ``pos``, edge ``voxels``, and napari Vectors layers.

Positive ``flow_signed`` means flow runs ``u -> v`` (as written by
:func:`haemolynx.haemodynamics.resistance.set_edge_flows`). The centreline is
oriented ``u -> v`` via :func:`haemolynx.visualization.geometry.edge_polyline`,
then reversed when the signed flow is negative.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from haemolynx.haemodynamics.resistance import flow_abs_log10_value
from haemolynx.visualization.geometry import edge_polyline

__all__ = [
    "edge_flow_direction_sign",
    "edge_flow_arrow_zyx",
    "edge_flow_direction_columns",
    "flow_direction_vectors",
    "flow_direction_components",
]


def edge_flow_direction_sign(
    edge_data: Mapping[str, Any],
    *,
    signed_flow_attr: str = "flow_signed",
    direction_attr: str = "edge_direction",
    positive_flow_means_u_to_v: bool = True,
) -> Optional[int]:
    """Infer edge direction sign (+1 for u->v, -1 for v->u, None unknown).

    Same priority as the Plotly flow-direction helper:
    1) numeric sign of ``signed_flow_attr`` (default ``flow_signed``)
    2) string labels on ``direction_attr`` (e.g. ``u_to_v`` / ``v_to_u``)
    """
    flow_val = edge_data.get(signed_flow_attr)
    if flow_val is not None:
        try:
            flow_float = float(flow_val)
            if np.isfinite(flow_float) and flow_float != 0.0:
                if flow_float > 0.0:
                    return 1 if positive_flow_means_u_to_v else -1
                return -1 if positive_flow_means_u_to_v else 1
        except (TypeError, ValueError):
            pass

    dir_val = edge_data.get(direction_attr)
    if dir_val is None:
        return None
    text = str(dir_val).strip().lower()
    if text in {"u_to_v", "uv", "forward", "fwd", "+"}:
        return 1
    if text in {"v_to_u", "vu", "reverse", "rev", "-"}:
        return -1
    return None


def flow_direction_components(direction: np.ndarray) -> tuple[float, float, float]:
    """Signed normalised ``(z, y, x)`` components of a flow direction vector.

    Each component is in ``[-1, +1]``. A zero or non-finite vector returns
    ``(0, 0, 0)``.
    """
    d = np.asarray(direction, dtype=float).reshape(-1)[:3]
    if d.shape[0] < 3 or not np.all(np.isfinite(d)):
        return 0.0, 0.0, 0.0
    norm = float(np.linalg.norm(d))
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0
    unit = d / norm
    return float(unit[0]), float(unit[1]), float(unit[2])


def _polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def edge_flow_arrow_zyx(
    points: np.ndarray,
    *,
    direction_sign: int,
    arrow_length: float,
    lateral_offset: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Nearby arrow anchor and direction vector for an edge (physical z,y,x).

    Port of the Plotly helper's mid-edge tangent + lateral offset, without the
    ``(z,y,x) -> (x,y,z)`` swap that Plotly needs.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        return None
    points = points[:, :3].copy()
    if direction_sign < 0:
        points = points[::-1, :]

    mid_idx = int(points.shape[0] // 2)
    lo = max(0, mid_idx - 1)
    hi = min(points.shape[0] - 1, mid_idx + 1)
    tangent = points[hi] - points[lo]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        tangent = points[-1] - points[0]
        tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        return None
    tangent_unit = tangent / tangent_norm

    # Offset arrow laterally so it is visible next to the edge.
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    lateral = np.cross(tangent_unit, ref)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm <= 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = np.cross(tangent_unit, ref)
        lat_norm = float(np.linalg.norm(lateral))
    if lat_norm <= 1e-12:
        return None
    lateral_unit = lateral / lat_norm

    anchor = points[mid_idx] + (lateral_offset * lateral_unit)
    vector = tangent_unit * float(arrow_length)
    return anchor, vector


def _iter_edges(graph: Any):
    if getattr(graph, "is_multigraph", lambda: False)():
        return graph.edges(keys=True, data=True)
    return ((u, v, 0, data) for u, v, data in graph.edges(data=True))


def edge_flow_direction_columns(graph: Any) -> dict[str, np.ndarray]:
    """Per drawable edge: signed unit ``(z, y, x)`` flow direction components.

    Uses the same edge order and skip rules as
    :func:`haemolynx.gui.results.edge_polylines` so the columns align with
    vessel segment features after ``segment_owner`` indexing.
    """
    dir_z: list[float] = []
    dir_y: list[float] = []
    dir_x: list[float] = []
    if getattr(graph, "is_multigraph", lambda: False)():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for u, v, _key, data in edge_iter:
        try:
            points = edge_polyline(graph, u, v, data)
        except ValueError:
            continue
        direction_sign = edge_flow_direction_sign(data)
        if direction_sign is None:
            dir_z.append(float("nan"))
            dir_y.append(float("nan"))
            dir_x.append(float("nan"))
            continue
        points = np.asarray(points, dtype=float)
        if direction_sign < 0:
            points = points[::-1, :]
        if points.shape[0] < 2:
            dir_z.append(float("nan"))
            dir_y.append(float("nan"))
            dir_x.append(float("nan"))
            continue
        mid_idx = int(points.shape[0] // 2)
        lo = max(0, mid_idx - 1)
        hi = min(points.shape[0] - 1, mid_idx + 1)
        tangent = points[hi] - points[lo]
        dz, dy, dx = flow_direction_components(tangent)
        dir_z.append(dz)
        dir_y.append(dy)
        dir_x.append(dx)

    return {
        "flow_dir_z": np.asarray(dir_z, dtype=float),
        "flow_dir_y": np.asarray(dir_y, dtype=float),
        "flow_dir_x": np.asarray(dir_x, dtype=float),
    }


def flow_direction_vectors(
    graph: Any,
    *,
    signed_flow_attr: str = "flow_signed",
    direction_attr: str = "edge_direction",
    positive_flow_means_u_to_v: bool = True,
    arrow_length_scale: float = 0.18,
    arrow_offset_scale: float = 0.08,
    flow_abs_attr: str = "flow_abs",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One mid-edge arrow per directed edge, plus features for colouring.

    Returns
    -------
    vectors
        Shape ``(N, 2, 3)``: origin and displacement in physical ``(z, y, x)``.
    features
        Parallel columns including ``flow_abs`` (magnitude used for the heatmap),
        ``flow_signed``, and ``flow_dir_z`` / ``flow_dir_y`` / ``flow_dir_x``
        (signed normalised direction components). Empty dict when there are no
        arrows.
    """
    directed: list[tuple[np.ndarray, int, float, float]] = []
    for u, v, _key, data in _iter_edges(graph):
        try:
            points = edge_polyline(graph, u, v, data)
        except ValueError:
            continue
        direction_sign = edge_flow_direction_sign(
            data,
            signed_flow_attr=signed_flow_attr,
            direction_attr=direction_attr,
            positive_flow_means_u_to_v=positive_flow_means_u_to_v,
        )
        if direction_sign is None:
            continue
        signed = data.get(signed_flow_attr)
        try:
            signed_f = float(signed) if signed is not None else float("nan")
        except (TypeError, ValueError):
            signed_f = float("nan")
        abs_val = data.get(flow_abs_attr)
        try:
            if abs_val is not None:
                flow_abs = float(abs_val)
            elif np.isfinite(signed_f):
                flow_abs = abs(signed_f)
            else:
                flow_abs = float("nan")
        except (TypeError, ValueError):
            flow_abs = abs(signed_f) if np.isfinite(signed_f) else float("nan")
        directed.append((np.asarray(points, dtype=float), int(direction_sign),
                         signed_f, flow_abs))

    if not directed:
        return np.empty((0, 2, 3), dtype=float), {}

    lengths = [_polyline_length(pts) for pts, _s, _sf, _fa in directed]
    valid = [float(v) for v in lengths if v > 0]
    base_len = float(np.median(valid)) if valid else 1.0
    arrow_length = max(0.25, base_len * float(arrow_length_scale))
    lateral_offset = max(0.10, base_len * float(arrow_offset_scale))

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    signed_col: list[float] = []
    abs_col: list[float] = []
    log10_col: list[float] = []
    dir_z_col: list[float] = []
    dir_y_col: list[float] = []
    dir_x_col: list[float] = []
    for points, direction_sign, signed_f, flow_abs in directed:
        result = edge_flow_arrow_zyx(
            points,
            direction_sign=direction_sign,
            arrow_length=arrow_length,
            lateral_offset=lateral_offset,
        )
        if result is None:
            continue
        anchor, vector = result
        origins.append(anchor)
        directions.append(vector)
        signed_col.append(signed_f)
        abs_col.append(flow_abs)
        if np.isfinite(flow_abs):
            log10_col.append(flow_abs_log10_value(flow_abs))
        else:
            log10_col.append(float("nan"))
        dz, dy, dx = flow_direction_components(vector)
        dir_z_col.append(dz)
        dir_y_col.append(dy)
        dir_x_col.append(dx)

    if not origins:
        return np.empty((0, 2, 3), dtype=float), {}

    vectors = np.stack(
        [np.stack(origins, axis=0), np.stack(directions, axis=0)], axis=1
    )
    features = {
        "flow_abs": np.asarray(abs_col, dtype=float),
        "flow_abs_log10": np.asarray(log10_col, dtype=float),
        "flow_signed": np.asarray(signed_col, dtype=float),
        "flow_dir_z": np.asarray(dir_z_col, dtype=float),
        "flow_dir_y": np.asarray(dir_y_col, dtype=float),
        "flow_dir_x": np.asarray(dir_x_col, dtype=float),
    }
    return vectors, features
