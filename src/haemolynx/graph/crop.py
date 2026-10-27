"""A network cut to a box, as an image of that box would have it.

:func:`crop_graph_to_box` keeps what of a network lies in a box: every node
inside it, and every vessel's centreline clipped to it. A vessel crossing a
face is cut there, and the piece inside ends in a new dead end that leaves the
image (``dead_ends.LEAVES_IMAGE``), as a vessel the field of view cuts does.
Each piece records the vessel it came from and the share of it it spans, so
what the uncropped network says about it (its flow) can be read back.

A synthetic network drawn larger and cropped so is a stand-in image whose
beyond is known: ``PLAN_synthetic_surround.md``'s cut-out harness
(``scripts/surround_check.py``).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.geometry import cumulative_lengths

from ._helpers import edge_sample_points
from .dead_ends import DEAD_END_KIND, DEAD_END_KIND_SOURCE, LEAVES_IMAGE, SOURCE_CROP
from .prune import FLOW_SOLVED, FLOW_STATE, SOLVE_DERIVED_EDGE_ATTRIBUTES

#: Each vessel's source: the uncropped edge ``(u, v, key)``; the node of the
#: piece that lies toward ``u``; and the share of the source's centreline the
#: piece spans, ``(start, end)`` measured from ``u``.
CROP_SOURCE_EDGE = "crop_source_edge"
CROP_FROM = "crop_from"
CROP_SPAN = "crop_span"
#: Where the box was, in the uncropped network's frame: ``{"lo": ..., "hi": ...}``.
CROP_BOX = "crop_box_um"
#: The graph-level records a crop keeps: the voxel size and what its physics
#: was read on. Every other one describes the uncropped network (its solve's
#: open-end conditions, its statistics, its thick centreline in its frame).
KEPT_GRAPH_ATTRIBUTES: tuple[str, ...] = (
    "voxel_size",
    "image_voxel_size_zyx",
    "image_voxel_size_xyz",
    "network_source",
    "diameter_basis",
    "viscosity_law",
    "haematocrit_model",
    "raw_image_stain",
)
#: What the uncropped network's solve left on its nodes and vessels.
SOLVE_NODE_ATTRIBUTES: tuple[str, ...] = ("pressure",)
SOLVE_EDGE_ATTRIBUTES: tuple[str, ...] = (*SOLVE_DERIVED_EDGE_ATTRIBUTES, FLOW_SOLVED, FLOW_STATE)
#: A cut vessel's resistance goes with its length: a piece loses them.
LENGTH_EDGE_ATTRIBUTES: tuple[str, ...] = ("resistance", "conductance")
#: A piece shorter than this (um) is a vessel grazing a face or a corner: dropped.
MIN_PIECE_UM = 1e-6


def _clip(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[tuple[float, float]]:
    """The stretch ``(t0, t1)`` of the segment *a* -> *b* inside the box, as
    fractions of it, or None where it misses the box (Liang-Barsky)."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for axis in range(3):
        if d[axis] == 0.0:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                return None
            continue
        ta = (lo[axis] - a[axis]) / d[axis]
        tb = (hi[axis] - a[axis]) / d[axis]
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None
    return t0, t1


def _pieces(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[dict]:
    """The stretches of the polyline *points* inside the box: each its
    points, its start and end as arc length from the first point, and
    whether it starts at the first point and ends at the last."""
    arc = cumulative_lengths(points)
    out: list[dict] = []
    current: Optional[dict] = None
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        clip = _clip(a, b, lo, hi)
        if clip is None:
            if current is not None:
                out.append(current)
                current = None
            continue
        t0, t1 = clip
        step = arc[i + 1] - arc[i]
        if current is not None and t0 > 0.0:
            out.append(current)
            current = None
        if current is None:
            current = {"points": [a + t0 * (b - a)], "start": arc[i] + t0 * step, "from_first": i == 0 and t0 == 0.0}
        current["points"].append(a + t1 * (b - a))
        current["end"] = arc[i] + t1 * step
        current["to_last"] = False
        if t1 < 1.0:
            out.append(current)
            current = None
    if current is not None:
        current["to_last"] = True
        out.append(current)
    return out


def _polyline(G: nx.MultiGraph, u: Any, v: Any, data: dict) -> np.ndarray:
    """The edge's centreline from ``u`` to ``v``, its ends on the nodes."""
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float)[:3] for n in (u, v)}
    points = np.asarray(edge_sample_points(u, v, data, pos), dtype=float).reshape(-1, 3)
    if not np.allclose(points[0], pos[u]):
        points = np.vstack([pos[u], points])
    if not np.allclose(points[-1], pos[v]):
        points = np.vstack([points, pos[v]])
    return points


def crop_graph_to_box(
    G: nx.MultiGraph,
    lo_um: Sequence[float],
    hi_um: Sequence[float],
    *,
    translate: bool = True,
) -> nx.MultiGraph:
    """What of *G* lies in the box from *lo_um* to *hi_um* ``(z, y, x)``, a
    new graph. See the module.

    Nodes inside the box keep their ids and attributes; each vessel's
    centreline is clipped to it, a piece crossing a face ending in a new node
    there (the next free int id) marked ``leaves_image``. A vessel wholly
    inside is kept whole; a piece takes its share of the vessel's ``length``
    and loses its ``resistance`` and ``conductance``. What the uncropped
    network's solve left (pressures, flows, flow states) goes from every node
    and vessel. Every vessel records its source (:data:`CROP_SOURCE_EDGE`,
    :data:`CROP_FROM`, :data:`CROP_SPAN`). With *translate*, positions are
    moved by ``-lo_um`` so the box starts at 0, as every image's does; the
    graph's ``image_extent_um`` is the box's, and :data:`CROP_BOX` where it was.
    """
    lo = np.asarray(lo_um, dtype=float)[:3]
    hi = np.asarray(hi_um, dtype=float)[:3]
    if np.any(hi <= lo):
        raise ValueError(f"the box must have positive extent, got lo {lo.tolist()} and hi {hi.tolist()}")
    eps = 1e-9 * max(1.0, float(np.max(np.abs(np.concatenate([lo, hi])))))
    lo_in, hi_in = lo - eps, hi + eps
    shift = lo if translate else np.zeros(3)

    def inside(position: Any) -> bool:
        p = np.asarray(position, dtype=float)[:3]
        return bool(np.all(p >= lo_in) and np.all(p <= hi_in))

    H = nx.MultiGraph()
    H.graph.update({k: G.graph[k] for k in KEPT_GRAPH_ATTRIBUTES if k in G.graph})
    H.graph["image_extent_um"] = tuple(float(v) for v in hi - lo)
    H.graph[CROP_BOX] = {"lo": tuple(float(v) for v in lo), "hi": tuple(float(v) for v in hi)}
    for node, data in G.nodes(data=True):
        if data.get("pos") is not None and inside(data["pos"]):
            kept = {k: v for k, v in data.items() if k not in SOLVE_NODE_ATTRIBUTES}
            kept["pos"] = np.asarray(data["pos"], dtype=float)[:3] - shift
            H.add_node(node, **kept)
    ints = [int(n) for n in G.nodes if isinstance(n, (int, np.integer))]
    next_id = (max(ints) + 1) if ints else 0

    def new_tip(position: np.ndarray) -> Any:
        nonlocal next_id
        node = next_id if ints or G.number_of_nodes() == 0 else ("crop", next_id)
        next_id += 1
        H.add_node(
            node,
            pos=np.clip(position, lo, hi) - shift,
            **{DEAD_END_KIND: LEAVES_IMAGE, DEAD_END_KIND_SOURCE: SOURCE_CROP},
        )
        return node

    for u, v, key, data in G.edges(keys=True, data=True):
        base = {k: val for k, val in data.items() if k not in SOLVE_EDGE_ATTRIBUTES}
        points = _polyline(G, u, v, data)
        total = float(cumulative_lengths(points)[-1])
        if u in H and v in H and all(inside(p) for p in points):
            if data.get("voxels") is not None:
                base["voxels"] = (np.asarray(data["voxels"], dtype=float).reshape(-1, 3) - shift).tolist()
            H.add_edge(u, v, **{
                **base,
                CROP_SOURCE_EDGE: (u, v, key),
                CROP_FROM: u,
                CROP_SPAN: (0.0, 1.0),
            })
            continue
        for piece in _pieces(points, lo_in, hi_in):
            span = float(piece["end"] - piece["start"])
            if span < MIN_PIECE_UM:
                continue
            path = np.asarray(piece["points"], dtype=float)
            a = u if piece["from_first"] else new_tip(path[0])
            b = v if piece["to_last"] else new_tip(path[-1])
            attributes = {k: val for k, val in base.items() if k not in LENGTH_EDGE_ATTRIBUTES}
            length = data.get("length")
            attributes["length"] = (
                float(length) * span / total if length is not None and total > 0 else span
            )
            path[0], path[-1] = H.nodes[a]["pos"] + shift, H.nodes[b]["pos"] + shift
            attributes.update({
                "voxels": (path - shift).tolist(),
                CROP_SOURCE_EDGE: (u, v, key),
                CROP_FROM: a,
                CROP_SPAN: (float(piece["start"]) / total, float(piece["end"]) / total) if total > 0 else (0.0, 1.0),
            })
            H.add_edge(a, b, **attributes)
    # A node whose every vessel the crop took is left out with them.
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0 and n in G and G.degree(n) > 0])
    return H


__all__ = [
    "CROP_BOX",
    "CROP_FROM",
    "CROP_SOURCE_EDGE",
    "CROP_SPAN",
    "KEPT_GRAPH_ATTRIBUTES",
    "crop_graph_to_box",
]
