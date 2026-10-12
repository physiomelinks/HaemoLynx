"""Finishing the vessels a dead end cut short (PLAN_dead_ends.md, Phases 3, 5 and 6).

Three repairs, each reading what :mod:`haemolynx.graph.dead_ends` says a dead
end is, each off unless a run turns it on (until marked grids calibrate them,
decisions D2 and D3):

- :func:`extend_dead_ends_to_faces` -- a dead end that leaves the image but
  stops short of the face (the segmented vessel runs on to it, within a few
  radii) is drawn on to the face, so its open end sits where the vessel leaves
  the image and its length is not cut short.
- :func:`trace_on_dead_ends` -- from a dead end the segmented vessel runs on
  past (``stops_short``), the centreline is drawn on through the mask along
  the vessel's heading until it meets another vessel (joined there), reaches
  an image face, or the mask ends (a blind end where the vessel really ends).
- :func:`join_dead_ends_to_vessel_sides` -- a dead end pointing at another
  vessel's side across one stretch of background (``broken``), that vessel not
  one of its neighbours in the network, is joined to it there.

None draws a vessel beside one already in the same lumen
(``duplicates_existing_vessel``), and none closes a loop inside one lumen
(``lumen_loops.loop_inside_one_lumen``): a refused repair leaves the dead end
as it was. Each new edge records what made it (``bridge_kind``).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from ._helpers import EdgeSampleIndex, densify_polyline, duplicates_existing_vessel, edge_sample_points, next_node_id

logger = logging.getLogger(__name__)

#: How far :func:`trace_on_dead_ends` follows the mask past a tip, microns.
TRACE_ON_MAX_UM = 30.0
#: The step it advances by, in microns.
TRACE_STEP_UM = 0.5
EXTENDED_TO_FACE = "extended_to_face"
TRACED_ON = "traced_on"
JOINED_TO_SIDE = "joined_to_side"


def _length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1))) if len(path) > 1 else 0.0


def _loop_through(G: nx.MultiGraph, tip: Any, edge: tuple, point: np.ndarray, join: np.ndarray, max_um: float):
    """The closed path a join from *tip* to *point* on *edge* would make with
    the network, if one shorter than *max_um* exists, else None."""
    u, v, key = edge
    data = G.edges[u, v, key]
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
    along = edge_sample_points(u, v, data, pos)
    index = int(np.argmin(np.linalg.norm(along - point, axis=1)))
    best = None
    for end in (u, v):
        try:
            length, route = nx.single_source_dijkstra(G, end, tip, cutoff=max_um, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if best is None or length < best[0]:
            best = (length, end, route)
    if best is None:
        return None
    _length_um, end, route = best
    piece = along[: index + 1][::-1] if end == u else along[index:]
    pieces = [np.asarray(join, dtype=float), piece]
    for a, b in zip(route[:-1], route[1:]):
        edge_data = min(G[a][b].values(), key=lambda d: d.get("length") or 0.0)
        ends = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
        pieces.append(edge_sample_points(a, b, edge_data, ends))
    return np.vstack(pieces)


def _closes_loop_in_lumen(G, tip, edge, point, join, support) -> bool:
    from .lumen_loops import LOOP_SEARCH_UM, loop_inside_one_lumen

    loop = _loop_through(G, tip, edge, point, join, LOOP_SEARCH_UM)
    return loop is not None and loop_inside_one_lumen(loop, support)


def _join(G, tip, edge, point, path, kind: str, background_um: float = 0.0) -> Any:
    """Split *edge* at *point* and add the edge *path* from *tip* to it."""
    from .edit import insert_node_on_edge

    target = insert_node_on_edge(G, *edge, point)
    path = np.asarray(path, dtype=float)
    path[-1] = np.asarray(G.nodes[target]["pos"], dtype=float)
    G.add_edge(tip, target, voxels=path.tolist(), length=_length(path), reconnected=True,
               bridge_kind=kind, bridge_background_um=float(background_um))
    return target


def _end_at(G, tip, path, kind: str) -> Any:
    """Add a new node at *path*'s end and the edge from *tip* to it."""
    path = np.asarray(path, dtype=float)
    if any(isinstance(n, (int, np.integer)) for n in G.nodes):
        node = next_node_id(G, set())
    else:
        # A graph whose nodes are named by position (a synthetic fixture)
        # keeps naming them so.
        node = tuple(float(round(c, 3)) for c in path[-1])
    G.add_node(node, pos=path[-1].copy())
    G.add_edge(tip, node, voxels=path.tolist(), length=_length(path), reconnected=True, bridge_kind=kind)
    return node


def extend_dead_ends_to_faces(G: nx.MultiGraph, support, *, extent_um: Sequence[float]) -> int:
    """Draw each dead end that leaves the image but stops short of the face
    on to the face, straight along its heading (the mask runs all that way,
    :func:`~haemolynx.graph.dead_ends.leaves_image`). Returns how many."""
    from .dead_ends import LEAVES_IMAGE, _distance_to_box_along, classify_dead_ends

    extent = np.asarray(extent_um, dtype=float)
    ends = classify_dead_ends(G, extent_um=extent, support=support)
    index = None
    extended = 0
    for tip, end in ends.items():
        if end.kind != LEAVES_IMAGE or not end.reaches_face or end.heading is None:
            continue
        start = np.asarray(end.tip_um, dtype=float)
        heading = np.asarray(end.heading, dtype=float)
        reach = _distance_to_box_along(start, heading, extent)
        if not reach or reach <= 0:
            continue
        path = densify_polyline(np.vstack([start, np.clip(start + reach * heading, 0.0, extent)]), max_step_um=1.0)
        if index is None:
            index = EdgeSampleIndex(G)
        if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
            continue
        _end_at(G, tip, path, EXTENDED_TO_FACE)
        index.add(path)
        extended += 1
    if extended:
        logger.info("Drew %d dead end(s) that leave the image on to the image face.", extended)
    return extended


def _nearest_other(centrelines, point: np.ndarray, own: set):
    """``(distance, sample index)`` of the nearest centreline sample not in *own*."""
    found = centrelines.near(point, 30.0, own)
    if not len(found):
        return None, None
    distance = np.linalg.norm(centrelines.points[found] - point, axis=1)
    best = int(np.argmin(distance))
    return float(distance[best]), int(found[best])


def trace_on_dead_ends(
    G: nx.MultiGraph, support, *, extent_um: Sequence[float], max_um: float = TRACE_ON_MAX_UM,
) -> dict[str, int]:
    """Draw the centreline on from each dead end the segmented vessel runs on
    past (``stops_short``), straight along its heading while it stays in the
    mask: joined to another vessel where it enters that vessel's lumen, ended
    at a face it reaches, or at the mask's end -- the vessel's real end.
    Returns how many of each."""
    from ._helpers import edge_id
    from .dead_ends import STOPS_SHORT, _Centrelines, classify_dead_ends

    extent = np.asarray(extent_um, dtype=float)
    ends = classify_dead_ends(G, extent_um=extent, support=support)
    counts = {"joined": 0, "to_face": 0, "to_mask_end": 0, "refused": 0}
    index = None
    for tip, end in ends.items():
        if end.kind != STOPS_SHORT or end.heading is None or tip not in G or G.degree(tip) != 1:
            continue
        centrelines = _Centrelines(G, support)
        own = {centrelines.ids[edge_id(*e)] for e in end.chain if edge_id(*e) in centrelines.ids}
        start = np.asarray(end.tip_um, dtype=float)
        heading = np.asarray(end.heading, dtype=float)
        steps = np.arange(1, int(np.ceil(max_um / TRACE_STEP_UM)) + 1) * TRACE_STEP_UM
        samples = start + steps[:, None] * heading
        inside_box = np.all((samples >= 0.0) & (samples <= extent), axis=1)
        in_mask = np.asarray(support.inside(samples), dtype=bool) & inside_box
        outcome, stop = None, None
        own_radius = float(support.radius(start.reshape(1, 3))[0])
        for i, point in enumerate(samples):
            if not inside_box[i]:
                outcome, stop = "to_face", i
                break
            if not in_mask[i]:
                outcome, stop = "to_mask_end", i
                break
            if steps[i] > own_radius:
                distance, nearest = _nearest_other(centrelines, point, own)
                if distance is not None and distance <= float(centrelines.radii[nearest]):
                    outcome, stop = "joined", i
                    break
        if outcome is None:
            continue
        if outcome == "joined":
            target = centrelines.points[nearest]
            edge = centrelines.edges[int(centrelines.owners[nearest])]
            path = densify_polyline(np.vstack([start, samples[:stop], target]), max_step_um=1.0)
        else:
            last = max(stop - 1, 0)
            if outcome == "to_face":
                path = densify_polyline(np.vstack([start, np.clip(samples[stop], 0.0, extent)]), max_step_um=1.0)
            else:
                if steps[last] < max(own_radius, 1.0):
                    continue
                path = densify_polyline(np.vstack([start, samples[last]]), max_step_um=1.0)
        if index is None:
            index = EdgeSampleIndex(G)
        if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
            counts["refused"] += 1
            continue
        if outcome == "joined":
            if _closes_loop_in_lumen(G, tip, edge, target, path, support):
                counts["refused"] += 1
                continue
            _join(G, tip, edge, target, path, TRACED_ON)
        else:
            _end_at(G, tip, path, TRACED_ON)
        index.add(path)
        counts[outcome] += 1
    if any(counts.values()):
        logger.info(
            "Traced on from dead ends the segmented vessel runs on past: %d joined to another vessel, "
            "%d drawn on to an image face, %d to where the mask ends; %d refused (beside a vessel in the "
            "same lumen, or closing a loop inside one).",
            counts["joined"], counts["to_face"], counts["to_mask_end"], counts["refused"],
        )
    return counts


def join_dead_ends_to_vessel_sides(
    G: nx.MultiGraph, support, *, extent_um: Sequence[float], max_gap_um: Optional[float] = None,
) -> int:
    """Join each dead end pointing at another vessel's side across one stretch
    of background (a ``broken`` dead end with a vessel ahead), that vessel not
    one of its neighbours in the network, to that vessel, straight, where the
    heading meets it. Closest first; each end joins once. Returns how many."""
    from .dead_ends import AHEAD_REACH_UM, BROKEN, classify_dead_ends

    reach = AHEAD_REACH_UM if max_gap_um is None else float(max_gap_um)
    ends = classify_dead_ends(G, extent_um=np.asarray(extent_um, dtype=float), support=support, ahead_reach_um=reach)
    candidates = sorted(
        (end for end in ends.values()
         if end.kind == BROKEN and end.ahead_um is not None and end.ahead_edge is not None and not end.tip_in_other_lumen),
        key=lambda end: end.ahead_um,
    )
    index = None
    joined = 0
    for end in candidates:
        tip, edge = end.tip, tuple(end.ahead_edge)
        if tip not in G or G.degree(tip) != 1 or not G.has_edge(*edge):
            continue
        start = np.asarray(end.tip_um, dtype=float)
        target = start + float(end.ahead_um) * np.asarray(end.heading, dtype=float)
        u, v, key = edge
        along = edge_sample_points(u, v, G.edges[u, v, key], {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)})
        target = along[int(np.argmin(np.linalg.norm(along - target, axis=1)))]
        path = densify_polyline(np.vstack([start, target]), max_step_um=1.0)
        if index is None:
            index = EdgeSampleIndex(G)
        if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
            continue
        if _closes_loop_in_lumen(G, tip, edge, target, path, support):
            continue
        _join(G, tip, edge, target, path, JOINED_TO_SIDE, end.ahead_background_um or 0.0)
        index.add(path)
        joined += 1
    if joined:
        logger.info("Joined %d dead end(s) to the side of the vessel straight ahead across a gap.", joined)
    return joined


__all__ = [
    "EXTENDED_TO_FACE",
    "JOINED_TO_SIDE",
    "TRACED_ON",
    "TRACE_ON_MAX_UM",
    "extend_dead_ends_to_faces",
    "join_dead_ends_to_vessel_sides",
    "trace_on_dead_ends",
]
