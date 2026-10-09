"""Loops lying inside one lumen: Lee-thinning rings, not vessel loops.

Lee thinning keeps a mask's topology, so every tunnel through it -- a speck
of background inside a vessel, two vessels touching twice -- comes back as a
ring of centreline round it, and graph building and mask recovery can close
further loops inside one segmented vessel. A vessel loop runs round tissue;
these run round nothing, or round a dropout. :func:`loop_inside_one_lumen`
tells the two apart from the mask, and :func:`remove_loops_inside_one_lumen`
breaks every such loop in a graph.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from haemolynx.preprocessing.bridge_mask_support import (
    DEFAULT_MAX_BACKGROUND_GAP_UM,
    MaskSupport,
    _sample_step,
)

from ._helpers import edge_sample_points

logger = logging.getLogger(__name__)

#: How long a loop is looked for, in microns. A longer one is left as it is:
#: lying inside one lumen, it would need a lumen some 60 um across.
LOOP_SEARCH_UM = 200.0

#: Stands in for an edge a search must not use: longer than any search.
_UNUSABLE = 1e12


def _chord_background_um(loop_um: np.ndarray, support: MaskSupport) -> np.ndarray | None:
    """The background each antipodal chord of a closed path crosses, in
    microns: from each point of the path, evenly spaced, to the point half
    the path away. None when the path is too short to have chords."""
    loop = np.asarray(loop_um, dtype=float).reshape(-1, 3)
    step = _sample_step(support.voxel_size_zyx)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    count = int(np.ceil(arc[-1] / step))
    if count < 4:
        return None
    along = np.arange(count) * (arc[-1] / count)
    points = np.column_stack([np.interp(along, arc, loop[:, axis]) for axis in range(3)])
    half = count // 2
    start, chords = points[:half], points[half:2 * half] - points[:half]
    lengths = np.linalg.norm(chords, axis=1)
    samples = int(np.ceil(lengths.max() / step)) + 1
    if samples < 2:
        return None
    t = np.linspace(0.0, 1.0, samples)
    flags = support.inside(
        (start[:, None, :] + t[None, :, None] * chords[:, None, :]).reshape(-1, 3)
    ).reshape(half, samples)
    return (~flags).sum(axis=1) * lengths / (samples - 1)


def loop_inside_one_lumen(loop_um: np.ndarray, support: MaskSupport) -> bool:
    """Whether a closed path lies inside one lumen rather than round tissue.

    A vessel loop runs round tissue, so the chord from a point of it to the
    point half the loop away crosses background. Lee thinning of a blob of
    mask, or a piece joined back to the vessel it lies beside, closes loops
    whose chords stay in the mask -- or cross only a dropout, the background
    a bridge may cross (``support.max_background_gap_um``): Lee thinning
    rings a one-voxel tunnel through a vessel. The loop is inside one lumen
    when at least half its chords cross no more than that.
    """
    background_um = _chord_background_um(loop_um, support)
    if background_um is None:
        return True
    dropout = np.sum(background_um <= support.max_background_gap_um + 1e-9)
    return 2 * int(dropout) >= len(background_um)


def _length(data: dict) -> float:
    return float(data.get("length", 0.0) or 0.0)


class _Pairs:
    """The graph as one weighted edge per node pair -- its shortest -- for
    SciPy's Dijkstra, kept in step as edges are taken out."""

    def __init__(self, G: nx.MultiGraph):
        self.G = G
        self.nodes = list(G.nodes)
        self.at = {node: i for i, node in enumerate(self.nodes)}
        seen: set[frozenset] = set()
        self.pairs: list[tuple[Any, Any]] = []
        for u, v in G.edges():
            if u != v and frozenset((u, v)) not in seen:
                seen.add(frozenset((u, v)))
                self.pairs.append((u, v))
        rows = [self.at[u] for u, v in self.pairs] + [self.at[v] for u, v in self.pairs]
        cols = [self.at[v] for u, v in self.pairs] + [self.at[u] for u, v in self.pairs]
        weights = [self.shortest(u, v)[1] for u, v in self.pairs] * 2
        self.matrix = csr_matrix(
            (np.asarray(weights, dtype=float), (rows, cols)), shape=(len(self.nodes),) * 2
        )
        self.matrix.sort_indices()

    def shortest(self, u, v) -> tuple[Any, float]:
        """``(key, weight)`` of the shortest edge joining *u* and *v*, or
        ``(None, unusable)`` when none is left. A zero-length edge weighs a
        trace, as SciPy's sparse graphs can drop an explicit zero."""
        if not self.G.has_edge(u, v):
            return None, _UNUSABLE
        key, data = min(self.G[u][v].items(), key=lambda item: _length(item[1]))
        return key, max(_length(data), 1e-9)

    def _slots(self, u, v) -> list[int]:
        slots = []
        for a, b in ((u, v), (v, u)):
            i, j = self.at[a], self.at[b]
            row = self.matrix.indices[self.matrix.indptr[i]:self.matrix.indptr[i + 1]]
            slots.append(int(self.matrix.indptr[i] + np.searchsorted(row, j)))
        return slots

    def set_weight(self, u, v, weight: float) -> None:
        for slot in self._slots(u, v):
            self.matrix.data[slot] = weight

    def refresh(self, u, v) -> None:
        """Re-read the pair after one of its edges was taken out."""
        self.set_weight(u, v, self.shortest(u, v)[1])

    def shortest_cycle(self, u, v, limit: float):
        """The shortest cycle through the shortest edge joining *u* and *v*,
        as ``[(a, b, key), ...]`` in order round it, or ``None`` when there
        is none within *limit*."""
        key, length = self.shortest(u, v)
        if key is None or length >= limit:
            return None
        best = None
        others = sorted(
            ((k, _length(d)) for k, d in self.G[u][v].items() if k != key), key=lambda kd: kd[1]
        )
        if others and length + others[0][1] < limit:
            best = (length + others[0][1], [(u, v, key), (v, u, others[0][0])])
        self.set_weight(u, v, _UNUSABLE)
        try:
            distance, previous = dijkstra(
                self.matrix, indices=self.at[u], return_predecessors=True, limit=limit - length
            )
        finally:
            self.set_weight(u, v, length)
        reached = float(distance[self.at[v]])
        if np.isfinite(reached) and (best is None or length + reached < best[0]):
            path = [self.at[v]]
            while path[-1] != self.at[u]:
                path.append(int(previous[path[-1]]))
            route = [self.nodes[i] for i in reversed(path)]
            cycle = [(u, v, key)]
            for a, b in zip(route[::-1][:-1], route[::-1][1:]):
                cycle.append((a, b, self.shortest(a, b)[0]))
            best = (length + reached, cycle)
        return None if best is None else best[1]


def _polyline(G: nx.MultiGraph, cycle) -> np.ndarray:
    pieces = []
    for a, b, key in cycle:
        positions = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
        pieces.append(edge_sample_points(a, b, G.edges[a, b, key], positions))
    return np.vstack(pieces)


def iter_short_loops(G: nx.MultiGraph, search_um: float = LOOP_SEARCH_UM):
    """Each distinct loop of *G* that is some edge's shortest loop within
    *search_um*, as ``(cycle, polyline)``: the ``[(a, b, key), ...]`` edges in
    order round it and its points in physical microns. The loops
    :func:`remove_loops_inside_one_lumen` judges, without changing *G*."""
    if G.number_of_edges() == 0:
        return
    pairs = _Pairs(G)
    seen: set[frozenset] = set()
    for u, v in pairs.pairs:
        cycle = pairs.shortest_cycle(u, v, search_um)
        if cycle is None:
            continue
        signature = frozenset((a, b, key) if str(a) <= str(b) else (b, a, key) for a, b, key in cycle)
        if signature in seen:
            continue
        seen.add(signature)
        yield cycle, _polyline(G, cycle)


def _arcs(G: nx.MultiGraph, cycle, cut_at: frozenset = frozenset()):
    """The cycle cut at every node with an edge off it, and at every node of
    *cut_at*: the runs of edges between the places the rest of the network
    meets it."""

    def meets(node) -> bool:
        return G.degree[node] > 2 or node in cut_at

    starts = [meets(a) for a, _b, _key in cycle]
    if not any(starts):
        return []
    first = starts.index(True)
    turned = cycle[first:] + cycle[:first]
    arcs, current = [], []
    for edge in turned:
        if current and meets(edge[0]):
            arcs.append(current)
            current = []
        current.append(edge)
    arcs.append(current)
    return arcs


def loop_sides(G: nx.MultiGraph, cycle, *, cut_at=()) -> list[list[tuple[Any, Any, Any]]]:
    """The sides of a loop: the runs of its ``(a, b, key)`` edges between
    the places the rest of the network meets it -- and any node of *cut_at*,
    such as an outlet the loop runs through -- in order round it.

    A loop met at one place is one side, the whole loop; a loop nothing else
    meets has each of its edges as a side.
    """
    cycle = list(cycle)
    return _arcs(G, cycle, frozenset(cut_at)) or [[edge] for edge in cycle]


#: Mask kept round a loop when :func:`loop_measures` reads it, in microns:
#: wider than any lumen radius read there.
MEASURE_MARGIN_UM = 10.0


def _arc_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _evenly_round(loop: np.ndarray, step_um: float) -> np.ndarray:
    """Points evenly spaced round a closed path, the closing point not repeated."""
    keep = np.concatenate([[True], np.linalg.norm(np.diff(loop, axis=0), axis=1) > 0])
    loop = loop[keep]
    if len(loop) < 2:
        return loop
    if not np.allclose(loop[0], loop[-1]):
        loop = np.vstack([loop, loop[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    count = max(16, int(np.ceil(arc[-1] / step_um)))
    along = np.arange(count) * (arc[-1] / count)
    return np.column_stack([np.interp(along, arc, loop[:, axis]) for axis in range(3)])


def loop_centre_um(loop_um: np.ndarray, step_um: float = 0.5) -> np.ndarray:
    """The middle of a closed path: the mean of points evenly spaced round
    it, so a side drawn with more points does not pull it over."""
    points = _evenly_round(np.asarray(loop_um, dtype=float).reshape(-1, 3), step_um)
    return points.mean(axis=0) if len(points) else np.zeros(3)


def loop_measures(
    loop_um: np.ndarray,
    *,
    mask: Any = None,
    voxel_size_zyx=(1.0, 1.0, 1.0),
    max_background_gap_um: float = DEFAULT_MAX_BACKGROUND_GAP_UM,
    sides_um=(),
) -> dict[str, Any]:
    """What a loop looks like, for a person's judgement of it to be fitted
    against later: its size and shape, and -- given the segmented *mask*
    (voxel-indexed, *voxel_size_zyx* apart) -- the mask along and across it.

    * ``loop_length_um``, ``centre_{z,y,x}_um``, ``span_um`` (its widest
      extent) and ``area_um2`` (enclosed, in its best-fitting plane);
    * ``out_of_plane``: its third over its second singular value, 0 for a
      flat loop -- how far it bends out of that plane;
    * with a mask: ``path_in``, the share of the loop in the mask;
      ``lumen_r_um``, the median lumen radius along it; ``chords_in_lumen``,
      the share of its antipodal chords crossing no more than a dropout
      (*max_background_gap_um*) of background, and ``chord_bg_median_um``
      their median background; ``inside_one_lumen``
      (:func:`loop_inside_one_lumen`); and, for each of *sides_um* (the
      sides' polylines), ``side_lumen_r_um``, its mean lumen radius, and
      ``wall_side``, the side the build's loop breaker would take out
      (nearest the wall, then longest). None for each without a mask.

    Only a box round the loop is read from the mask, so a memory-mapped
    volume is not loaded whole.
    """
    loop = np.asarray(loop_um, dtype=float).reshape(-1, 3)
    sides = [np.asarray(side, dtype=float).reshape(-1, 3) for side in sides_um]
    voxel = np.asarray(voxel_size_zyx, dtype=float)
    points = _evenly_round(loop, 0.5 * float(voxel.min()))
    centre = points.mean(axis=0) if len(points) else np.zeros(3)
    measures: dict[str, Any] = {
        "loop_length_um": _arc_length(loop),
        "centre_z_um": float(centre[0]),
        "centre_y_um": float(centre[1]),
        "centre_x_um": float(centre[2]),
        "span_um": 0.0,
        "area_um2": 0.0,
        "out_of_plane": 0.0,
    }
    if len(points) >= 3:
        centred = points - centre
        _u, singular, axes = np.linalg.svd(centred, full_matrices=False)
        if singular[1] > 1e-9:
            measures["out_of_plane"] = float(singular[2] / singular[1])
        flat = centred @ axes[:2].T
        measures["area_um2"] = float(
            0.5 * abs(np.sum(flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1]))
        )
        sample = points[:: max(1, len(points) // 256)]
        measures["span_um"] = float(
            np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2).max()
        )
    for name in ("path_in", "lumen_r_um", "chords_in_lumen", "chord_bg_median_um", "inside_one_lumen"):
        measures[name] = None
    measures["side_lumen_r_um"] = None
    measures["wall_side"] = None
    if mask is None or len(points) < 2:
        return measures

    everything = np.vstack([loop, *sides]) / voxel
    margin = np.ceil(MEASURE_MARGIN_UM / voxel).astype(int)
    shape = np.asarray(np.shape(mask)[-3:])
    lo = np.clip(np.floor(everything.min(axis=0)).astype(int) - margin, 0, shape)
    hi = np.clip(np.ceil(everything.max(axis=0)).astype(int) + margin + 1, 0, shape)
    crop = np.asarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) > 0
    offset = lo * voxel
    support = MaskSupport(crop, tuple(voxel), max_background_gap_um=max_background_gap_um)
    local = points - offset
    measures["path_in"] = float(np.mean(support.inside(local)))
    measures["lumen_r_um"] = float(np.median(support.radius(local)))
    background = _chord_background_um(loop - offset, support)
    if background is not None:
        measures["chords_in_lumen"] = float(np.mean(background <= max_background_gap_um + 1e-9))
        measures["chord_bg_median_um"] = float(np.median(background))
    measures["inside_one_lumen"] = bool(loop_inside_one_lumen(loop - offset, support))
    if sides:
        radii = [float(np.mean(support.radius(side - offset))) for side in sides]
        measures["side_lumen_r_um"] = radii
        measures["wall_side"] = int(
            min(range(len(sides)), key=lambda i: (radii[i], -_arc_length(sides[i])))
        )
    return measures


def _wall_hugging(G: nx.MultiGraph, run, support: MaskSupport) -> tuple[float, float]:
    """How near the lumen's wall an arc runs -- its mean distance to
    background, smaller first -- and then its length, longer first."""
    polyline = _polyline(G, run)
    return float(np.mean(support.radius(polyline))), -sum(_length(G.edges[e]) for e in run)


def _break(G: nx.MultiGraph, cycle, support: MaskSupport) -> list[tuple[Any, Any, Any]]:
    """Take out the loop's arc running nearest the lumen's wall -- the whole
    loop when only one place meets the network, its wall-most edge when none
    does -- and the nodes left with nothing. Returns the edges taken out.

    The arc kept is the one nearest the lumen's middle: the arc along the
    rim covered less of the lumen, and the diameter read off the mask along
    it would come out too small."""
    arc = min(loop_sides(G, cycle), key=lambda run: _wall_hugging(G, run, support))
    for a, b, key in arc:
        G.remove_edge(a, b, key)
    for a, b, _key in arc:
        for node in (a, b):
            if G.has_node(node) and G.degree[node] == 0:
                G.remove_node(node)
    return arc


def remove_loops_inside_one_lumen(
    G: nx.MultiGraph, support: MaskSupport, *, search_um: float = LOOP_SEARCH_UM
) -> nx.MultiGraph:
    """Break, in place, every loop of *G* lying inside one lumen of
    *support*'s mask (:func:`loop_inside_one_lumen`).

    Each edge's shortest loop within *search_um* is found and tested; one
    inside a lumen loses its arc nearest the wall -- the run of edges between two
    places the rest of the network meets it -- so the vessel goes on, drawn
    once, and no spur is left hanging. A loop only one place meets goes
    whole. Runs until no loop inside a lumen is left; a loop round tissue,
    however short, stays.
    """
    if G.number_of_edges() == 0:
        return G
    removed: list[float] = []
    round_tissue: set[frozenset] = set()
    changed = True
    while changed:
        changed = False
        pairs = _Pairs(G)
        for u, v in pairs.pairs:
            if not G.has_edge(u, v):
                continue
            cycle = pairs.shortest_cycle(u, v, search_um)
            if cycle is None:
                continue
            signature = frozenset((a, b, key) if str(a) <= str(b) else (b, a, key) for a, b, key in cycle)
            if signature in round_tissue:
                continue
            if not loop_inside_one_lumen(_polyline(G, cycle), support):
                round_tissue.add(signature)
                continue
            lengths = {(a, b, key): _length(G.edges[a, b, key]) for a, b, key in cycle}
            for a, b, key in _break(G, cycle, support):
                removed.append(lengths[(a, b, key)])
                if G.has_node(a) and G.has_node(b):
                    pairs.refresh(a, b)
                else:
                    pairs.set_weight(a, b, _UNUSABLE)
            changed = True
    if removed:
        logger.info(
            "Broke loops lying inside one lumen (Lee-thinning rings): took out %d edge(s), %.0f um",
            len(removed),
            float(sum(removed)),
        )
    return G


def _joined_without(G: nx.MultiGraph, u, v) -> bool:
    """Whether *u* and *v* stay connected with the one edge between them taken
    out: searched from both ends at once, without changing *G*."""
    if G.number_of_edges(u, v) > 1:
        return True
    seen = ({u}, {v})
    fronts = ([u], [v])
    while fronts[0] and fronts[1]:
        side = 0 if len(fronts[0]) <= len(fronts[1]) else 1
        grown = []
        for node in fronts[side]:
            for other in G.neighbors(node):
                if {node, other} == {u, v}:
                    continue
                if other in seen[1 - side]:
                    return True
                if other not in seen[side]:
                    seen[side].add(other)
                    grown.append(other)
        fronts = (grown, fronts[1]) if side == 0 else (fronts[0], grown)
    return False


def cuts_off_vessels(G: nx.MultiGraph, edge) -> bool:
    """Whether taking *edge* out would cut part of the network off from the
    rest -- more than the edge's own dead end, which simply goes with it."""
    u, v, _key = edge
    if u == v or G.degree[u] == 1 or G.degree[v] == 1:
        return False
    return not _joined_without(G, u, v)


def remove_parallel_edges_in_lumen(G: nx.MultiGraph, support: MaskSupport) -> nx.MultiGraph:
    """Drop, in place, one edge of each pair
    :func:`graph.diagnostics.diagnose_parallel_duplicates_in_lumen` reports.

    The edge nearer the wall goes (``_wall_hugging``), so the centreline
    that stays is the one nearer the lumen's middle -- unless taking it out
    would cut vessels off from the rest of the network
    (:func:`cuts_off_vessels`), when the other goes instead, or, when that
    would too, both stay. A branch meeting its parent, and two vessels with
    background between them, are not pairs in that report and are left
    alone.
    """
    from haemolynx.graph.diagnostics import diagnose_parallel_duplicates_in_lumen

    report = diagnose_parallel_duplicates_in_lumen(
        G,
        support.mask,
        voxel_size_zyx=tuple(float(v) for v in support.voxel_size_zyx),
        mask_support=support,
    )
    removed = kept_connected = 0
    for left, right in report["duplicate_pairs"]:
        present = [edge for edge in (left, right) if G.has_edge(*edge)]
        if len(present) < 2:
            continue
        candidates = sorted(present, key=lambda edge: _wall_hugging(G, [edge], support))
        drop = next((edge for edge in candidates if not cuts_off_vessels(G, edge)), None)
        if drop is None:
            kept_connected += 1
            continue
        u, v, key = drop
        G.remove_edge(u, v, key)
        removed += 1
        for node in (u, v):
            if G.has_node(node) and G.degree(node) == 0:
                G.remove_node(node)
    G.graph["parallel_lumen_edges_removed"] = int(G.graph.get("parallel_lumen_edges_removed", 0)) + removed
    if removed or kept_connected:
        logger.info(
            "Removed %d parallel edge(s) lying beside another in one lumen; kept %d pair(s) "
            "where either edge was the only link to vessels beyond it",
            removed,
            kept_connected,
        )
    return G


__all__ = [
    "LOOP_SEARCH_UM",
    "MEASURE_MARGIN_UM",
    "cuts_off_vessels",
    "iter_short_loops",
    "loop_centre_um",
    "loop_inside_one_lumen",
    "loop_measures",
    "loop_sides",
    "remove_loops_inside_one_lumen",
    "remove_parallel_edges_in_lumen",
]
