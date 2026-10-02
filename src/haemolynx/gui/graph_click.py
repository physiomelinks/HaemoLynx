"""Resolving a click on the vessels/nodes napari layers to a graph element.

Pure -- no Qt, no napari import -- so the "Edit" window's Add branch and
Delete edge features are testable against a fake click and layer data
without a live viewer. Reuses :func:`haemolynx.gui.branch_hover.nearest_vector_index`
for the vessels (Vectors) layer polyline hit-test, the same one hover
already uses (see ``_hover_feature_index`` in ``gui/_widget.py``), so click
and hover cannot silently disagree about which segment is under the cursor.
Node hits go through napari's own Points picking (``Layer.get_value``),
exactly like hover does for every layer kind but Vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .branch_hover import BRANCH_HOVER_MAX_DISTANCE, nearest_vector_index

__all__ = ["EdgeHit", "NodeHit", "hit_test_nodes", "hit_test_vessels", "nearest_node_hit"]


@dataclass(frozen=True)
class NodeHit:
    """A click landed on this graph node."""

    node_id: Any


@dataclass(frozen=True)
class EdgeHit:
    """A click landed on this graph edge, at this point along it.

    ``point_um`` is the closest point on the hit segment to the click, in
    the same physical-micron frame as node ``pos`` and edge ``voxels`` --
    where :func:`haemolynx.graph.insert_node_on_edge` would split it.
    """

    u: Any
    v: Any
    key: Any
    point_um: tuple[float, float, float]


def _as_float_vec(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _closest_point_on_segment(
    origin: np.ndarray, direction: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """The point on segment ``origin -> origin + direction`` nearest *point*.

    Uses the full geometry regardless of what :func:`nearest_vector_index`
    projected onto to decide *which* segment was hit -- once a segment is
    chosen, where to split or start a new branch on it is a plain 3D
    question, not a screen-projected one.
    """
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 0.0:
        return origin
    t = float(np.dot(point - origin, direction) / length_sq)
    t = min(1.0, max(0.0, t))
    return origin + t * direction


def hit_test_vessels(
    layer_data: Any,
    features: Mapping[str, Any],
    position: Any,
    *,
    max_distance: float = BRANCH_HOVER_MAX_DISTANCE,
    view_direction: Any | None = None,
    dims: Sequence[int] | None = None,
) -> EdgeHit | None:
    """Which edge a click on the vessels Vectors layer hit, and where on it."""
    index = nearest_vector_index(
        position,
        layer_data,
        max_distance=max_distance,
        view_direction=view_direction,
        dims=dims,
    )
    if index is None:
        return None
    point = _as_float_vec(position)
    data = np.asarray(layer_data, dtype=float)
    if point is None or data.ndim != 3 or index >= data.shape[0]:
        return None
    ndim = min(point.size, data.shape[2])
    origin = data[index, 0, :ndim]
    direction = data[index, 1, :ndim]
    closest = _closest_point_on_segment(origin, direction, point[:ndim])

    try:
        u = features["u"][index]
        v = features["v"][index]
        key = features["key"][index]
    except (KeyError, IndexError, TypeError):
        return None
    return EdgeHit(u=u, v=v, key=key, point_um=tuple(float(c) for c in closest))


def hit_test_nodes(get_value_result: Any, features: Mapping[str, Any]) -> NodeHit | None:
    """Which graph node a native ``Points.get_value(...)`` hit resolved to."""
    index = get_value_result
    if isinstance(index, tuple):
        index = index[0]
    if index is None:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    node_ids = features.get("node_id") if hasattr(features, "get") else None
    if node_ids is None or index < 0 or index >= len(node_ids):
        return None
    return NodeHit(node_id=node_ids[index])


def nearest_node_hit(
    points: Any,
    features: Mapping[str, Any],
    position: Any,
    *,
    max_distance: float,
    view_direction: Any | None = None,
    dims: Sequence[int] | None = None,
) -> NodeHit | None:
    """The node nearest a click, within *max_distance*, or None.

    For a click napari's own Points picking missed: a node dot a few microns
    across is a few pixels on screen in a whole-volume 3D view. Each node is a
    zero-length segment to :func:`nearest_vector_index`, so with a
    *view_direction* the distance is measured across the line of sight -- how
    far from the cursor the node *looks* -- whatever its depth.
    """
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or not len(data):
        return None
    segments = np.stack([data, np.zeros_like(data)], axis=1)
    index = nearest_vector_index(
        position, segments, max_distance=max_distance,
        view_direction=view_direction, dims=dims,
    )
    return hit_test_nodes(index, features)
