"""Interactive graph edits: split an edge, delete one and collapse, draw a new one.

Pure graph/array operations only -- no Qt, no napari. :mod:`haemolynx.gui.graph_editor`
is the interactive window that turns mouse clicks into calls here, so the
editing rules themselves are testable without a viewer.

A newly added edge here is deliberately left without ``branch_order``/
``diameter_um``: haemodynamics reads those, not this module, and the "Edit"
window's Regenerate action re-runs branch-order and diameter assignment over
the whole (edited) graph rather than having this module guess a value for
one new edge in isolation (see ``haemodynamics.poiseuille.set_poiseuille_resistances``
for what an edge missing either one costs -- a silent zero-conductance edge).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

from ._helpers import calculate_path_length, next_node_id
from .degree2 import create_trivial_merged_edge

logger = logging.getLogger(__name__)

__all__ = [
    "EdgeDraft",
    "astar_path",
    "commit_new_edge",
    "delete_edge_and_collapse",
    "insert_node_on_edge",
    "mask_cost_field",
    "voxel_path_to_microns",
]


def _edge_attrs_for_segment(
    edge_data: dict[str, Any], segment_points: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """New edge attributes for one slice of an existing edge's path.

    Same pattern as ``graph.cut_at_large_vessel_volumes._edge_attrs_for_segment``:
    keep every attribute except the geometry, which is recomputed from the slice.
    """
    attrs = {k: v for k, v in edge_data.items() if k not in ("voxels", "length")}
    points = [tuple(float(c) for c in row) for row in segment_points]
    attrs["voxels"] = points
    attrs["length"] = float(calculate_path_length(points))
    return attrs


def _closest_vertex_index(points: Sequence[Sequence[float]], target: Sequence[float]) -> int:
    """Index of the polyline vertex closest to *target*."""
    arr = np.asarray(points, dtype=float)
    distances = np.linalg.norm(arr - np.asarray(target, dtype=float), axis=1)
    return int(np.argmin(distances))


def insert_node_on_edge(
    G: nx.MultiGraph,
    u: Any,
    v: Any,
    key: Any,
    point_um: Sequence[float],
    *,
    reserved_ids: set[Any] | None = None,
) -> Any:
    """Split edge ``(u, v, key)`` at the vertex of its own path closest to
    *point_um*, inserting a new node there. Returns the node id at the split
    -- an existing endpoint (``u`` or ``v``) when the closest vertex is
    already one of them, so a click near an edge's own end does not create a
    redundant duplicate node.

    An edge drawn straight (no ``voxels``, per :func:`haemolynx.visualization.geometry.edge_polyline`'s
    own fallback) is split against a synthetic 2-point line between its
    nodes' ``pos`` instead.
    """
    data = G.get_edge_data(u, v, key)
    if data is None:
        raise ValueError(f"No edge ({u!r}, {v!r}, {key!r}) to split")

    voxels = list(data.get("voxels") or [])
    if len(voxels) < 2:
        voxels = [tuple(G.nodes[u]["pos"]), tuple(G.nodes[v]["pos"])]

    index = _closest_vertex_index(voxels, point_um)
    if index == 0:
        return u
    if index == len(voxels) - 1:
        return v

    new_node = next_node_id(G, set(reserved_ids or ()))
    split_point = tuple(float(c) for c in voxels[index])
    G.add_node(new_node, pos=split_point)

    part_a = voxels[: index + 1]
    part_b = voxels[index:]
    G.remove_edge(u, v, key)
    G.add_edge(u, new_node, **_edge_attrs_for_segment(data, part_a))
    G.add_edge(new_node, v, **_edge_attrs_for_segment(data, part_b))
    return new_node


def delete_edge_and_collapse(G: nx.MultiGraph, u: Any, v: Any, key: Any) -> set[Any]:
    """Remove edge ``(u, v, key)``, then tidy up just its own two endpoints.

    An endpoint left at degree 0 is removed outright; one left at degree 2
    (a plain pass-through, its two remaining edges going to two distinct
    neighbours) is collapsed into one continuous edge via
    :func:`haemolynx.graph.degree2.create_trivial_merged_edge`, exactly as a
    real degree-2 node reads everywhere else in this codebase. Deliberately
    scoped to only the two nodes this one deletion touches -- unlike
    :mod:`haemolynx.graph.degree2`'s whole-graph sweeps, a pre-existing
    degree-2 node elsewhere in the graph is left alone.

    Returns the set of node ids the caller's drawn layers need to refresh:
    removed nodes, and the two surviving endpoints of any freshly merged edge.
    """
    if not G.has_edge(u, v, key=key):
        raise ValueError(f"No edge ({u!r}, {v!r}, {key!r}) to delete")
    G.remove_edge(u, v, key)

    changed: set[Any] = set()
    for node in (u, v):
        if not G.has_node(node):
            continue
        degree = G.degree(node)
        if degree == 0:
            G.remove_node(node)
            changed.add(node)
        elif degree == 2:
            incident = list(G.edges(node, keys=True, data=True))
            (_, n1, _k1, d1), (_, n2, _k2, d2) = incident
            if n1 == n2:
                # Two parallel edges to the same neighbour: a loop, not a
                # pass-through -- collapsing it would create a self-loop.
                continue
            node_pos = G.nodes[node].get("pos")
            merged = create_trivial_merged_edge(d1, d2, node_pos)
            G.remove_node(node)
            G.add_edge(n1, n2, **merged)
            changed.update({node, n1, n2})
    return changed


def mask_cost_field(mask: np.ndarray) -> np.ndarray:
    """Routing cost ``1 + d^2``, ``d`` = voxel distance to the nearest True mask voxel.

    Same formula as ``graph.reconnect``'s own ``window_cost``, applied to the
    segmented vessel mask rather than the skeleton: cheap inside or near real
    vessel structure, growing quadratically -- never blocked -- away from it,
    so "Add branch" can still bridge a genuine gap in the segmentation
    instead of refusing to route through unsegmented voxels at all.
    """
    binary = np.asarray(mask, dtype=bool)
    return 1.0 + distance_transform_edt(~binary) ** 2


#: Voxels of context kept around a routed segment's own start/end when
#: cropping *cost_field* for :func:`astar_path`. The cost field itself is
#: already fully computed (unlike ``graph.reconnect``'s own windowing, which
#: exists to avoid recomputing a distance transform) -- this pad only bounds
#: the pathfinding search itself, which otherwise scales with the *whole*
#: array on every single mouse click "Add branch" makes: measured directly
#: against ``route_through_array``, 128**3 voxels costs ~3s and 200**3 ~26s,
#: easily enough to freeze the GUI thread this runs on for a real stack.
#: Matches ``graph.reconnect.COST_WINDOW_PAD`` for the same "far enough to
#: route around a real obstruction, not so far it re-explores the volume"
#: reasoning.
_ASTAR_WINDOW_PAD = 32


def _straight_line_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Evenly spaced integer voxel coordinates from *start* to *end*."""
    n_steps = max(int(np.linalg.norm(end.astype(float) - start.astype(float))) + 1, 2)
    return np.round(
        np.linspace(start.astype(float), end.astype(float), n_steps)
    )


def astar_path(
    cost_field: np.ndarray, start_vox: Sequence[float], end_vox: Sequence[float]
) -> np.ndarray:
    """Cheapest voxel path from *start_vox* to *end_vox* through *cost_field*.

    Thin wrapper over :func:`skimage.graph.route_through_array`, the same
    call ``graph.reconnect`` makes (``fully_connected=True``), so a branch
    drawn here costs a path the same way a reconnected secondary loop edge
    does. Returns voxel-index coordinates, not physical microns -- see
    :func:`voxel_path_to_microns`.

    Runs only over a local window around the two points, padded by
    :data:`_ASTAR_WINDOW_PAD` -- see its own docstring for why. Both points
    are clamped into *cost_field*'s own bounds first, so a click just
    outside the segmented volume (a stray ray in 3D view, an edge case near
    the boundary) routes from the nearest in-bounds voxel instead of
    raising. Never raises: a routing failure inside the window falls back to
    a straight line between the two (clamped) points, matching
    ``preprocessing.skeleton.connect_skeleton_components``'s own
    mask-preferred-not-mask-required philosophy -- a branch always gets
    drawn somewhere, even if not the mask-hugging path this is trying for.
    """
    shape = np.asarray(cost_field.shape)
    start = np.clip(np.round(start_vox).astype(int), 0, shape - 1)
    end = np.clip(np.round(end_vox).astype(int), 0, shape - 1)

    lo = np.maximum(np.minimum(start, end) - _ASTAR_WINDOW_PAD, 0)
    hi = np.minimum(np.maximum(start, end) + _ASTAR_WINDOW_PAD + 1, shape)
    window = cost_field[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    start_local = tuple((start - lo).astype(int))
    end_local = tuple((end - lo).astype(int))
    try:
        path_coords, _cost = route_through_array(
            window, start_local, end_local, fully_connected=True
        )
        return np.asarray(path_coords, dtype=float) + lo
    except Exception:
        logger.debug(
            "A* branch routing failed; falling back to a straight line.", exc_info=True
        )
        return _straight_line_path(start, end)


def voxel_path_to_microns(
    path_vox: np.ndarray, voxel_size_zyx: Sequence[float]
) -> list[tuple[float, float, float]]:
    """A voxel-index path as physical-micron points, per-axis scaled."""
    scale = np.asarray(voxel_size_zyx, dtype=float)
    return [
        tuple(float(c) for c in (row * scale)) for row in np.asarray(path_vox, dtype=float)
    ]


@dataclass
class EdgeDraft:
    """In-progress "Add branch" state: an anchor node and its drawn path so far.

    ``points_um`` always starts pre-seeded with the anchor's own position, so
    :attr:`last_point` never needs graph access to fall back on.
    """

    start_node: Any
    points_um: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def last_point(self) -> tuple[float, float, float]:
        if not self.points_um:
            raise ValueError("EdgeDraft.points_um must be seeded with the start point")
        return self.points_um[-1]

    def extend(self, path_um: Sequence[Sequence[float]]) -> None:
        """Append a newly found path segment (physical microns).

        ``path_um[0]`` is expected to (nearly) coincide with
        :attr:`last_point`; only the rest is appended, so chaining several
        A* segments does not duplicate the shared vertex.
        """
        points = [tuple(float(c) for c in p) for p in path_um]
        if not points:
            return
        if not self.points_um:
            self.points_um = points
            return
        self.points_um.extend(points[1:])


def commit_new_edge(
    G: nx.MultiGraph,
    draft: EdgeDraft,
    end_node: Any | None,
    *,
    reserved_ids: set[Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Add *draft*'s accumulated path to ``G`` as one new edge.

    *end_node* is an existing node id when the draft's path ended by hitting
    one (call :func:`insert_node_on_edge` first if it hit an edge instead,
    and pass the node that returns); ``None`` creates a fresh dangling
    terminal node at the draft's last point (pressing Finish with no target
    hit). Returns ``(start_node, end_node, key)`` for the new edge.
    """
    if end_node is None:
        end_node = next_node_id(G, set(reserved_ids or ()))
        G.add_node(end_node, pos=draft.last_point)
    key = G.add_edge(
        draft.start_node,
        end_node,
        voxels=[tuple(float(c) for c in p) for p in draft.points_um],
        length=float(calculate_path_length(draft.points_um)),
    )
    return draft.start_node, end_node, key
