"""Branch-hover tooltip text, metric availability, and polyline hit-testing.

Napari Vectors ``_get_value`` always returns ``None``, so hover cannot use the
native query. The vessels (and flow-direction) Vectors layers carry a
``tooltip`` feature column from these helpers, and :func:`nearest_vector_index`
picks the segment under the cursor. Which optional metrics appear in the
layer-controls checkboxes is decided here from what the graph actually carries
-- the same rule ``available_edge_columns`` uses for colouring.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Optional metrics the panel can offer, in display order.
BRANCH_HOVER_METRICS: tuple[str, ...] = (
    "flow",
    "order",
    "diameter",
    "diameter_source",
    "resistance",
    "tortuosity",
    "length",
)

#: Checkbox / tooltip label for each optional metric key.
BRANCH_HOVER_LABELS: dict[str, str] = {
    "flow": "branch flow",
    "order": "branch order",
    "diameter": "branch diameter",
    "diameter_source": "diameter source",
    "resistance": "branch resistance",
    "tortuosity": "branch tortuosity",
    "length": "branch length",
}

#: Graph edge attribute that supplies each metric (``None`` = computed).
_METRIC_ATTR: dict[str, str | None] = {
    "flow": "flow_abs",
    "order": "branch_order",
    "diameter": "diameter_um",
    "diameter_source": "diameter_source",
    "resistance": "resistance",
    "tortuosity": None,
    "length": "length",
}

_TEXT_HOVER_METRICS = frozenset({"order", "diameter_source"})

_BRANCH_ID_LINE = "branchID: {branch_id}"

#: Pickup radius in data coordinates (µm). Matches the old midpoint-circle
#: diameter of 2 so a branch is still easy to hit, but the target is the
#: polyline rather than a marker beside it.
BRANCH_HOVER_MAX_DISTANCE = 2.0


def branch_id_for_edge(
    _u: Any, _v: Any, key: Any, data: Mapping[str, Any]
) -> str:
    """Stable branch identity for a tooltip line.

    Prefer ``segment_id`` (assigned at graph build and used for colouring);
    fall back to the MultiGraph edge key.
    """
    segment_id = data.get("segment_id")
    if segment_id is not None:
        return str(segment_id)
    return str(key)


def edge_tortuosity(
    graph: Any, u: Any, v: Any, data: Mapping[str, Any]
) -> float | None:
    """Path length / Euclidean end-to-end, or ``None`` when not computable."""
    length = data.get("length")
    if length is None:
        return None
    try:
        path_length = float(length)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(path_length) or path_length < 0:
        return None

    pos_u = graph.nodes[u].get("pos") if u in graph.nodes else None
    pos_v = graph.nodes[v].get("pos") if v in graph.nodes else None
    if pos_u is None or pos_v is None:
        return None
    straight = float(np.linalg.norm(np.asarray(pos_u, dtype=float)[:3]
                                    - np.asarray(pos_v, dtype=float)[:3]))
    if not np.isfinite(straight) or straight <= 0:
        return None
    return path_length / straight


def _metric_value(
    graph: Any, u: Any, v: Any, data: Mapping[str, Any], metric: str
) -> Any | None:
    """Raw value for *metric* on one edge, or ``None`` when missing."""
    if metric == "tortuosity":
        return edge_tortuosity(graph, u, v, data)
    attr = _METRIC_ATTR[metric]
    assert attr is not None
    value = data.get(attr)
    if value is None:
        return None
    if metric in _TEXT_HOVER_METRICS:
        text = str(value).strip()
        return text if text else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def available_branch_hover_metrics(graph: Any) -> tuple[str, ...]:
    """Optional metrics this graph can offer, in declared order.

    A metric is available when at least one drawable edge carries a usable
    value for it. Tortuosity is derived from length and node positions, so it
    appears once those exist rather than as a stored edge attribute.
    """
    present: list[str] = []
    for metric in BRANCH_HOVER_METRICS:
        for u, v, _key, data in _iter_edges(graph):
            if _metric_value(graph, u, v, data, metric) is not None:
                present.append(metric)
                break
    return tuple(present)


def format_metric_value(metric: str, value: Any) -> str:
    """Canonical string for one metric value in a tooltip line."""
    if metric in _TEXT_HOVER_METRICS:
        return str(value)
    number = float(value)
    return f"{number:.6g}"


def format_branch_tooltip(
    branch_id: str,
    values: Mapping[str, Any],
    selected: Sequence[str],
) -> str:
    """Deterministic multi-line tooltip text.

    Always starts with ``branchID``. Then one line per *selected* metric that
    both is offered by :data:`BRANCH_HOVER_METRICS` and has a non-``None``
    value in *values*. Unselected and unavailable metrics are omitted.
    """
    lines = [_BRANCH_ID_LINE.format(branch_id=branch_id)]
    for metric in BRANCH_HOVER_METRICS:
        if metric not in selected:
            continue
        if metric not in values or values[metric] is None:
            continue
        label = BRANCH_HOVER_LABELS[metric]
        lines.append(f"{label}: {format_metric_value(metric, values[metric])}")
    return "\n".join(lines)


def panel_metric_options(
    available: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """``(metric_key, checkbox_label)`` pairs the panel should show."""
    return tuple(
        (metric, BRANCH_HOVER_LABELS[metric])
        for metric in BRANCH_HOVER_METRICS
        if metric in available
    )


def filter_selected_metrics(
    selected: Iterable[str],
    available: Sequence[str],
) -> tuple[str, ...]:
    """Keep only metrics that are both selected and currently available."""
    available_set = set(available)
    return tuple(
        metric for metric in BRANCH_HOVER_METRICS
        if metric in selected and metric in available_set
    )


def branch_hover_rows(
    graph: Any,
    selected: Sequence[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Per-drawable-edge branch ids, tooltips, and raw metric columns.

    Edges that cannot be placed (no polyline) are skipped, matching
    :func:`haemolynx.gui.results.edge_polylines`.
    """
    from haemolynx.visualization.geometry import edge_polyline

    available = available_branch_hover_metrics(graph)
    chosen = filter_selected_metrics(selected, available)

    branch_ids: list[str] = []
    tooltips: list[str] = []
    columns: dict[str, list[Any]] = {metric: [] for metric in BRANCH_HOVER_METRICS}

    for u, v, key, data in _iter_edges(graph):
        try:
            edge_polyline(graph, u, v, data)
        except ValueError:
            continue
        branch_id = branch_id_for_edge(u, v, key, data)
        values = {
            metric: _metric_value(graph, u, v, data, metric)
            for metric in BRANCH_HOVER_METRICS
        }
        branch_ids.append(branch_id)
        tooltips.append(format_branch_tooltip(branch_id, values, chosen))
        for metric in BRANCH_HOVER_METRICS:
            columns[metric].append(values[metric])

    features: dict[str, np.ndarray] = {
        "branch_id": np.asarray(branch_ids, dtype=object),
        "tooltip": np.asarray(tooltips, dtype=object),
    }
    for metric in BRANCH_HOVER_METRICS:
        if metric in _TEXT_HOVER_METRICS:
            features[metric] = np.asarray(
                ["" if v is None else str(v) for v in columns[metric]],
                dtype=object,
            )
        else:
            features[metric] = np.asarray(
                [np.nan if v is None else float(v) for v in columns[metric]],
                dtype=float,
            )
    return branch_ids, features


def default_selected_metrics(available: Sequence[str]) -> tuple[str, ...]:
    """Initial checkbox state: every metric the graph can currently offer."""
    return tuple(metric for metric in BRANCH_HOVER_METRICS if metric in available)


def tooltips_from_feature_table(
    features: Mapping[str, Any],
    selected: Sequence[str],
) -> np.ndarray:
    """Rebuild the ``tooltip`` column from an existing features table.

    Used when the user toggles checkboxes: the raw metric columns stay put and
    only the composed strings change.
    """
    branch_ids = [str(v) for v in np.asarray(features["branch_id"]).tolist()]
    n = len(branch_ids)
    chosen = tuple(
        metric for metric in BRANCH_HOVER_METRICS if metric in selected
    )
    tooltips: list[str] = []
    for index in range(n):
        values: dict[str, Any] = {}
        for metric in BRANCH_HOVER_METRICS:
            if metric not in features:
                values[metric] = None
                continue
            raw = np.asarray(features[metric])[index]
            if metric in _TEXT_HOVER_METRICS:
                text = "" if raw is None else str(raw).strip()
                values[metric] = text or None
            else:
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    values[metric] = None
                else:
                    values[metric] = None if not np.isfinite(number) else number
        tooltips.append(format_branch_tooltip(branch_ids[index], values, chosen))
    return np.asarray(tooltips, dtype=object)


def available_metrics_from_features(features: Mapping[str, Any]) -> tuple[str, ...]:
    """Which optional metrics a hover layer's features actually carry values for."""
    present: list[str] = []
    for metric in BRANCH_HOVER_METRICS:
        if metric not in features:
            continue
        column = np.asarray(features[metric])
        if metric in _TEXT_HOVER_METRICS:
            if any(str(v).strip() for v in column.tolist()):
                present.append(metric)
        else:
            numbers = np.asarray(column, dtype=float)
            if np.isfinite(numbers).any():
                present.append(metric)
    return tuple(present)


def _iter_edges(graph: Any):
    if getattr(graph, "is_multigraph", lambda: False)():
        return graph.edges(keys=True, data=True)
    return ((u, v, 0, data) for u, v, data in graph.edges(data=True))


def hover_features_for_segments(
    graph: Any,
    owner: np.ndarray,
    selected: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[str, ...]]:
    """Repeat per-edge hover columns across Vectors segments via *owner*.

    *owner* is the drawable-edge index of each vector segment, the same array
    :func:`haemolynx.gui.results.polylines_to_vectors` returns. Empty when
    there are no segments or no drawable edges.
    """
    available = available_branch_hover_metrics(graph)
    chosen = (
        default_selected_metrics(available)
        if selected is None
        else filter_selected_metrics(selected, available)
    )
    index = np.asarray(owner, dtype=int)
    if index.size == 0:
        return {}, available, chosen
    _ids, features = branch_hover_rows(graph, chosen)
    n_edges = len(features.get("tooltip", ()))
    if n_edges == 0 or int(index.min()) < 0 or int(index.max()) >= n_edges:
        return {}, available, chosen
    return (
        {name: np.asarray(values)[index] for name, values in features.items()},
        available,
        chosen,
    )


def nearest_vector_index(
    position: Any,
    vectors: Any,
    *,
    max_distance: float = BRANCH_HOVER_MAX_DISTANCE,
    view_direction: Any | None = None,
    dims: Sequence[int] | None = None,
) -> int | None:
    """Index of the nearest Vectors segment within *max_distance*, else ``None``.

    *vectors* is napari Vectors data ``(M, 2, D)``: origin and displacement,
    in the same frame as *position*. When *dims* is given, distance uses only
    those axes. When *view_direction* is a non-zero vector (3D camera ray),
    segments are projected onto the view plane through *position* so the
    pickup matches what is under the cursor on screen.
    """
    data = np.asarray(vectors, dtype=float)
    point = _as_float_vec(position)
    if (
        point is None
        or data.ndim != 3
        or data.shape[0] == 0
        or data.shape[1] != 2
    ):
        return None
    origins = np.asarray(data[:, 0, :], dtype=float)
    directions = np.asarray(data[:, 1, :], dtype=float)
    ndim = int(origins.shape[1])
    if dims is not None:
        axes = np.asarray(list(dims), dtype=int)
        axes = axes[(axes >= 0) & (axes < ndim) & (axes < point.size)]
        if axes.size == 0:
            return None
        point = point[axes]
        origins = origins[:, axes]
        directions = directions[:, axes]
    else:
        n = min(int(point.size), ndim)
        point = point[:n]
        origins = origins[:, :n]
        directions = directions[:, :n]

    view = _as_float_vec(view_direction)
    width = int(origins.shape[1])
    if view is not None and view.size >= width:
        view = view[:width]
        norm = float(np.linalg.norm(view))
        if np.isfinite(norm) and norm > 1e-12:
            view = view / norm
            ends = origins + directions
            origins = _project_onto_plane(origins, point, view)
            ends = _project_onto_plane(ends, point, view)
            directions = ends - origins

    length_sq = np.einsum("ij,ij->i", directions, directions)
    delta = point - origins
    t = np.zeros(len(origins), dtype=float)
    nonzero = length_sq > 0.0
    t[nonzero] = (
        np.einsum("ij,ij->i", delta[nonzero], directions[nonzero])
        / length_sq[nonzero]
    )
    t = np.clip(t, 0.0, 1.0)
    closest = origins + t[:, None] * directions
    offset = closest - point
    dist_sq = np.einsum("ij,ij->i", offset, offset)
    dist_sq = np.where(np.isfinite(dist_sq), dist_sq, np.inf)
    if not np.isfinite(dist_sq).any():
        return None
    index = int(np.argmin(dist_sq))
    max_sq = float(max_distance) * float(max_distance)
    if not np.isfinite(max_sq) or dist_sq[index] > max_sq:
        return None
    return index


def _as_float_vec(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _project_onto_plane(
    points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Project *points* onto the plane through *plane_point* with *normal*."""
    rel = points - plane_point
    return points - np.outer(rel @ normal, normal)
