"""Where a vessel is, as a polyline, for whatever wants to draw it.

An edge carries its shape in ``voxels`` -- the skeleton path it came from, in
physical (z, y, x) microns -- or, when it does not, only the positions of the
two nodes it joins. Turning that into something drawable is the same three
questions every time:

* is there a polyline at all, and is it three columns wide?
* does it run from *u* to *v*, or the other way round? Skeleton paths come out
  in whichever order the tracer walked them.
* do its ends sit exactly on the nodes? Cluster collapse moves a node to the
  mean of the cluster it replaced, leaving the polyline a little short of it,
  and a gap at every junction is visible in anything that draws lines.

Four places used to answer them separately: the VTK export, the pericyte point
derivation, and two plotly writers. This is the one answer, so a vessel is in
the same place in ParaView, in an HTML plot and in napari.
"""
from __future__ import annotations

from typing import Any, Mapping

import networkx as nx
import numpy as np

__all__ = ["as_points", "edge_polyline"]


def as_points(path_like: Any) -> np.ndarray:
    """A polyline as exactly three float columns.

    Wider input is truncated and narrower is zero-padded, because a consumer
    needs a fixed width and a flat graph is still worth drawing. Fewer than two
    points is not a polyline and raises.
    """
    arr = np.asarray(path_like, dtype=float)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    if arr.shape[0] < 2:
        raise ValueError("Polyline needs at least two points")
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    if arr.shape[1] < 3:
        arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
    return arr


def _snap_to_nodes(points: np.ndarray, u: Any, v: Any, graph: nx.Graph) -> np.ndarray:
    """Put the ends exactly on the nodes, where the graph knows where they are."""
    out = points.copy()
    u_pos = graph.nodes[u].get("pos")
    v_pos = graph.nodes[v].get("pos")
    if u_pos is not None:
        out[0] = np.asarray(u_pos, dtype=float)[:3]
    if v_pos is not None:
        out[-1] = np.asarray(v_pos, dtype=float)[:3]
    return out


def _orient_to_nodes(points: np.ndarray, u: Any, v: Any, graph: nx.Graph) -> np.ndarray:
    """Run the polyline u -> v, reversing it if that fits the nodes better."""
    u_pos = graph.nodes[u].get("pos")
    v_pos = graph.nodes[v].get("pos")
    if u_pos is None or v_pos is None or len(points) < 2:
        return points

    u_arr = np.asarray(u_pos, dtype=float)[:3]
    v_arr = np.asarray(v_pos, dtype=float)[:3]
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)

    direct_cost = float(np.linalg.norm(start - u_arr) + np.linalg.norm(end - v_arr))
    flipped_cost = float(np.linalg.norm(start - v_arr) + np.linalg.norm(end - u_arr))
    if flipped_cost < direct_cost:
        return points[::-1].copy()
    return points


def edge_polyline(
    graph: nx.Graph,
    u: Any,
    v: Any,
    edge_data: Mapping[str, Any],
    *,
    orient: bool = True,
    snap: bool = True,
) -> np.ndarray:
    """The (n, 3) polyline for one edge, running u -> v with its ends on them.

    Raises ``ValueError`` when the edge has neither a usable ``voxels`` path nor
    positions for both of its nodes -- there is then nothing to draw.
    """
    voxels = edge_data.get("voxels")
    if voxels is not None and len(voxels) >= 2:
        points = as_points(voxels)
    else:
        u_pos = graph.nodes[u].get("pos")
        v_pos = graph.nodes[v].get("pos")
        if u_pos is None or v_pos is None:
            raise ValueError(
                f"Edge ({u}, {v}) is missing both voxels and node positions"
            )
        points = as_points([u_pos, v_pos])

    if orient:
        points = _orient_to_nodes(points, u, v, graph)
    if snap:
        points = _snap_to_nodes(points, u, v, graph)
    return points
