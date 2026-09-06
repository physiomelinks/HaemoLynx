"""Detect "cartwheel" hub artifacts: one node with many spoke edges radiating
in every direction, instead of the two or three branches a real vessel
junction has.

Where this comes from: nothing in this module changes it, or even names it
directly, but the shape is a known side effect of any step that redraws graph
topology from spatial proximity alone -- collapsing a cluster of nearby nodes
into one representative (``graph.collapse.collapse_node_clusters``) rewires
every edge that used to reach each cluster member so it now reaches the one
representative instead. A cluster of many separate, real vessel branches that
happened to skeletonize with their junctions close together becomes a single
node with all of their edges attached, radiating outward in whatever
direction each original branch actually ran -- visually, a wheel of spokes
around one hub, sitting on top of the vessel network it came from.

There is no reliable way to undo that after the fact -- the cluster's
original, separate positions are gone once collapsed to one representative --
so this module does not try to fix it. It only tells you it happened: a pure,
read-only diagnostic in the same spirit as ``graph.diagnostics``, run against
a finished graph whenever you want to check. Nothing in the mandatory
``build_graph_from_skeleton`` pipeline calls it.

The detection itself is geometric, not tied to *why* a hub formed: a real
vessel junction's daughters generally continue in some coherent direction
(even a busy one has an inflow side and an outflow side); a cartwheel hub's
spokes point every which way, with no such coherence. That is measured with
the mean resultant length of the spokes' outgoing unit-tangent directions --
the same "how uniformly spread are these directions" statistic used for
directional/circular data generally: close to 1 when they mostly agree, close
to 0 when they are spread evenly around the hub.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Union

import numpy as np
import networkx as nx

__all__ = [
    "CartwheelHub",
    "hub_spoke_directions",
    "hub_radial_dispersion",
    "detect_cartwheel_hubs",
    "format_cartwheel_hub_report",
]

#: A real vessel junction is a bifurcation (degree 3) or, occasionally, a
#: trifurcation; a node with more incident edges than this is already
#: unusual enough to be worth checking for the cartwheel shape.
DEFAULT_MIN_DEGREE = 6

#: Mean resultant length below this counts as "spread out enough to be a
#: cartwheel", not merely a busy junction with directional coherence. 0.5 is
#: past the point where the spokes could plausibly be described as leaving
#: "mostly one way" -- see the docstrings below for the geometry.
DEFAULT_MAX_RADIAL_DISPERSION = 0.5

#: How far along an edge's own centreline to look for its outgoing direction,
#: matching the sampling distance statistics.compute_emergence_angles_by_branch_order
#: uses for the analogous "which way does this edge leave this node" question.
DEFAULT_TANGENT_LENGTH_UM = 10.0


@dataclass(frozen=True)
class CartwheelHub:
    """One flagged node: how many spokes, how spread out, and which edges."""

    node: Any
    degree: int
    radial_dispersion: float
    spoke_count: int
    mean_spoke_length_um: float
    #: The incident edges considered, as (neighbor, edge_key) pairs -- edge_key
    #: is None for a plain nx.Graph, and the multigraph key otherwise.
    spokes: tuple[tuple[Any, Any], ...]


def _incident_edge_items(G: Union[nx.Graph, nx.MultiGraph], node: Any):
    """Incident edges as ``(neighbor, key, data)``, ``key`` None for a plain Graph."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, key, data in G.edges(node, keys=True, data=True):
            if u == v:
                continue
            yield (v if u == node else u), key, data
    else:
        for u, v, data in G.edges(node, data=True):
            if u == v:
                continue
            yield (v if u == node else u), None, data


def _point_along_polyline(points: np.ndarray, distance_um: float) -> np.ndarray:
    """The point this far along *points* (from its start), clamped to its end."""
    if len(points) < 2:
        return points[0]
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(deltas)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return points[0]
    target = min(max(float(distance_um), 0.0), total)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 1), len(points) - 1)
    t0, t1 = float(cumulative[idx - 1]), float(cumulative[idx])
    frac = (target - t0) / (t1 - t0) if t1 > t0 else 0.0
    return points[idx - 1] + frac * (points[idx] - points[idx - 1])


def _spoke_direction_and_length(
    G: Union[nx.Graph, nx.MultiGraph],
    node: Any,
    neighbor: Any,
    data: dict,
    *,
    tangent_length_um: float,
) -> tuple[np.ndarray | None, float]:
    """The unit vector leaving *node* along this edge, and the edge's length.

    Prefers the edge's own ``voxels`` centreline, oriented to start at *node*;
    falls back to the straight line to *neighbor*'s position when there is no
    usable path. Returns ``(None, length)`` when neither position nor voxels
    give a well-defined direction (a duplicate point, or missing ``pos``).
    """
    pos_node = G.nodes[node].get("pos")
    if pos_node is None:
        return None, 0.0
    pos_node = np.asarray(pos_node, dtype=float)

    length = data.get("length")
    voxels = data.get("voxels")
    points = np.asarray(voxels, dtype=float) if voxels is not None and len(voxels) >= 2 else None
    if points is not None:
        if np.linalg.norm(points[0] - pos_node) > np.linalg.norm(points[-1] - pos_node):
            points = points[::-1]
        target = _point_along_polyline(points, tangent_length_um)
        vector = target - points[0]
        if length is None:
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    else:
        pos_neighbor = G.nodes[neighbor].get("pos")
        if pos_neighbor is None:
            return None, float(length) if length is not None else 0.0
        pos_neighbor = np.asarray(pos_neighbor, dtype=float)
        vector = pos_neighbor - pos_node
        if length is None:
            length = float(np.linalg.norm(vector))

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None, float(length)
    return vector / norm, float(length)


def hub_spoke_directions(
    G: Union[nx.Graph, nx.MultiGraph],
    node: Any,
    *,
    tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
) -> dict[tuple[Any, Any], np.ndarray]:
    """Every incident edge's outgoing unit direction from *node*.

    Keyed by ``(neighbor, edge_key)`` so parallel edges to the same neighbor
    are kept distinct. An edge contributes nothing (is left out of the
    result) when its direction cannot be determined -- a missing ``pos`` on
    either end, or both ends landing on the same point.
    """
    directions: dict[tuple[Any, Any], np.ndarray] = {}
    for neighbor, key, data in _incident_edge_items(G, node):
        direction, _length = _spoke_direction_and_length(
            G, node, neighbor, data, tangent_length_um=tangent_length_um
        )
        if direction is not None:
            directions[(neighbor, key)] = direction
    return directions


def hub_radial_dispersion(directions: Sequence[np.ndarray]) -> float:
    """How uniformly *directions* (unit vectors) spread out around their hub.

    The mean resultant length: the norm of the average direction. 1.0 means
    every direction agrees exactly; 0.0 means they cancel out completely, the
    signature of spokes spread evenly around a wheel. Fewer than two
    directions cannot disperse at all, so this returns 1.0 (nothing to flag)
    rather than an undefined value.
    """
    if len(directions) < 2:
        return 1.0
    mean_vector = np.mean(np.asarray(directions, dtype=float), axis=0)
    return float(np.linalg.norm(mean_vector))


def detect_cartwheel_hubs(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    min_degree: int = DEFAULT_MIN_DEGREE,
    max_radial_dispersion: float = DEFAULT_MAX_RADIAL_DISPERSION,
    tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
) -> list[CartwheelHub]:
    """Every node whose incident edges radiate in too many directions to be
    a real vessel junction.

    A node qualifies only once it clears both bars: at least *min_degree*
    incident edges, AND a radial dispersion at or below
    *max_radial_dispersion* once directions are computed for them (a node can
    have high degree with low dispersion -- several branches all continuing
    roughly one way -- and that is not flagged). Returned worst-first (lowest
    dispersion, i.e. most spread out, first).
    """
    if min_degree < 2:
        raise ValueError("min_degree must be >= 2")
    if not 0.0 <= max_radial_dispersion <= 1.0:
        raise ValueError("max_radial_dispersion must be in [0.0, 1.0]")

    hubs: list[CartwheelHub] = []
    for node in G.nodes():
        degree = int(G.degree(node))
        if degree < min_degree:
            continue

        spokes: list[tuple[Any, Any]] = []
        directions: list[np.ndarray] = []
        lengths: list[float] = []
        for neighbor, key, data in _incident_edge_items(G, node):
            direction, length = _spoke_direction_and_length(
                G, node, neighbor, data, tangent_length_um=tangent_length_um
            )
            if direction is None:
                continue
            spokes.append((neighbor, key))
            directions.append(direction)
            lengths.append(length)

        if len(directions) < min_degree:
            # Missing position/voxel data hid too many spokes to judge this
            # node fairly against the same bar every other node is held to.
            continue

        dispersion = hub_radial_dispersion(directions)
        if dispersion > max_radial_dispersion:
            continue

        hubs.append(
            CartwheelHub(
                node=node,
                degree=degree,
                radial_dispersion=dispersion,
                spoke_count=len(spokes),
                mean_spoke_length_um=float(np.mean(lengths)) if lengths else 0.0,
                spokes=tuple(spokes),
            )
        )

    hubs.sort(key=lambda hub: hub.radial_dispersion)
    return hubs


def format_cartwheel_hub_report(hubs: Sequence[CartwheelHub]) -> str:
    """A compact multiline report, in the style of format_degree2_diagnostics_report."""
    if not hubs:
        return "Cartwheel hub guard: no hubs flagged."
    lines = [f"Cartwheel hub guard: {len(hubs)} hub(s) flagged."]
    for hub in hubs:
        lines.append(
            f"  node={hub.node}: degree={hub.degree}, "
            f"radial_dispersion={hub.radial_dispersion:.3f}, "
            f"mean_spoke_length_um={hub.mean_spoke_length_um:.1f}"
        )
    return "\n".join(lines)
