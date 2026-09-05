"""Split graph edges where they cross a fat/thick-vessel mask boundary.

When a thin vessel's skeleton is joined to a fat vessel's own centreline
(:func:`preprocessing.thick_vessels._join_thin_arms_to_fat_ridge`), the
resulting merged edge runs from real thin-vessel material, through the join
bridge, into the fat vessel's own centreline -- topology simplification has
no notion that the bridge is different from the vessel on either side of it.

This module finds where such an edge crosses the fat/thick mask boundary and
splits it there, tagging the segment(s) inside the mask so haemodynamics can
give them (near) zero resistance: a bridge represents the small vessel
opening into the big vessel's lumen, not new vessel material with its own
resistance. See :func:`haemodynamics.apply._assign_poiseuille_resistances`
for where :data:`IS_ZERO_RESISTANCE` is read back.

Mirrors :mod:`graph.cut_at_large_vessel_volumes`'s edge-sampling/splitting
approach (same generic helpers, from ``_helpers``), but keeps both sides of
the boundary -- that module drops the interior -- and an edge may cross more
than once.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from ._helpers import (
    calculate_path_length,
    densify_polyline,
    edge_sample_points,
    next_node_id,
    points_inside_mask,
)

logger = logging.getLogger(__name__)

#: Edge attribute marking a segment as a thin-vessel-to-fat-vessel join, not
#: new vessel material -- haemodynamics gives it a negligible resistance
#: instead of computing one from its diameter/length.
IS_ZERO_RESISTANCE = "is_zero_resistance"


def _runs(
    inside_flags: list[bool], *, min_run_length: int = 2
) -> list[tuple[int, int, bool]]:
    """Contiguous ``(start, end inclusive, is_inside)`` runs over the whole sequence.

    A run shorter than *min_run_length* is a single-sample flicker right at
    the boundary (voxel-grid discretisation, not a real crossing) and gets
    absorbed into a neighbouring run rather than becoming its own
    degenerate edge.
    """
    if not inside_flags:
        return []
    raw: list[list[Any]] = []
    start = 0
    current = inside_flags[0]
    for i in range(1, len(inside_flags)):
        if inside_flags[i] != current:
            raw.append([start, i - 1, current])
            start = i
            current = inside_flags[i]
    raw.append([start, len(inside_flags) - 1, current])

    changed = True
    while changed and len(raw) > 1:
        changed = False
        for i, (s, e, _inside) in enumerate(raw):
            if e - s + 1 >= min_run_length:
                continue
            if i == 0:
                raw[1][0] = s
            else:
                raw[i - 1][1] = e
            del raw[i]
            changed = True
            break

    # Absorbing a short run into its neighbour (above) can leave that
    # neighbour newly adjacent to another run of the same is_inside value
    # (the short run used to separate them) -- coalesce those back into one
    # run so a flicker never produces an extra, artificial crossing.
    coalesced: list[list[Any]] = []
    for s, e, inside in raw:
        if coalesced and coalesced[-1][2] == inside:
            coalesced[-1][1] = e
        else:
            coalesced.append([s, e, inside])
    return [(s, e, inside) for s, e, inside in coalesced]


def _segment_attrs(
    edge_data: dict[str, Any],
    segment_points: np.ndarray,
    *,
    is_inside: bool,
) -> dict[str, Any]:
    attrs = {
        key: value
        for key, value in edge_data.items()
        if key not in ("voxels", "length", IS_ZERO_RESISTANCE)
    }
    points = [tuple(float(c) for c in row) for row in segment_points]
    attrs["voxels"] = points
    attrs["length"] = float(calculate_path_length(points))
    if is_inside:
        attrs[IS_ZERO_RESISTANCE] = True
    return attrs


def insert_thick_vessel_junction_nodes(
    G: nx.MultiGraph,
    thick_mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> nx.MultiGraph:
    """Split edges at the ``thick_mask`` boundary; tag the interior segment(s).

    An edge with every sample outside ``thick_mask`` (an ordinary thin
    vessel) or every sample inside it (the fat vessel's own centreline) is
    left untouched -- both are real vessel material with their own
    resistance. An edge that crosses the boundary is split at each
    crossing, keeping both sides; only the segment(s) inside ``thick_mask``
    are tagged :data:`IS_ZERO_RESISTANCE`.
    """
    result = nx.MultiGraph()
    result.graph.update(G.graph)

    node_pos: dict[Any, np.ndarray] = {}
    for node_id, data in G.nodes(data=True):
        result.add_node(node_id, **dict(data))
        if "pos" in data:
            node_pos[node_id] = np.asarray(data["pos"], dtype=float)

    reserved_ids: set[Any] = set(result.nodes)
    edges_split = 0
    segments_tagged = 0

    edge_iter = (
        G.edges(keys=True, data=True)
        if isinstance(G, nx.MultiGraph)
        else ((u, v, 0, d) for u, v, d in G.edges(data=True))
    )

    for u, v, _key, edge_data in edge_iter:
        if u not in node_pos or v not in node_pos:
            result.add_edge(u, v, **dict(edge_data))
            continue

        points = edge_sample_points(u, v, edge_data, node_pos)
        sample_step = min(float(spacing) for spacing in voxel_size_zyx)
        points = densify_polyline(points, max_step_um=sample_step)
        inside_flags = points_inside_mask(
            points, thick_mask, voxel_size_zyx=voxel_size_zyx
        ).tolist()

        runs = _runs(inside_flags)
        if len(runs) <= 1:
            # Fully interior or fully exterior: real vessel material either way.
            result.add_edge(u, v, **dict(edge_data))
            continue

        edges_split += 1
        prev_node = u
        prev_index = 0
        last_index = len(runs) - 1
        for run_index, (_run_start, run_end, is_inside) in enumerate(runs):
            # Start from the previous run's own end index, not this run's
            # start: consecutive segments must share their boundary point,
            # or this edge's voxels would not actually start at prev_node's
            # own position -- a discontinuous polyline one sample short.
            segment = points[prev_index : run_end + 1]
            is_last = run_index == last_index
            if is_last:
                next_node = v
            else:
                next_node = next_node_id(result, reserved_ids)
                reserved_ids.add(next_node)
                next_pos = tuple(float(c) for c in segment[-1])
                result.add_node(next_node, pos=next_pos)
                node_pos[next_node] = np.asarray(next_pos, dtype=float)
            result.add_edge(
                prev_node,
                next_node,
                **_segment_attrs(edge_data, segment, is_inside=is_inside),
            )
            if is_inside:
                segments_tagged += 1
            prev_node = next_node
            prev_index = run_end

    logger.info(
        "Thick-vessel junction split: crossing_edges=%d, "
        "zero_resistance_segments=%d, remaining_nodes=%d, remaining_edges=%d.",
        edges_split,
        segments_tagged,
        result.number_of_nodes(),
        result.number_of_edges(),
    )
    return result
