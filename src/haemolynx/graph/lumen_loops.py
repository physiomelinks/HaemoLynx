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
from typing import Any, Callable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import (
    DEFAULT_MAX_BACKGROUND_GAP_UM,
    MaskSupport,
    _sample_step,
)

from ._helpers import densify_polyline, edge_sample_points
from .thick_vessel_junctions import IS_ZERO_RESISTANCE, THICK_TAPER

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


class _LoopMeasures:
    """One mask measure of loops -- whether one lies inside one lumen, the hole
    one runs round -- remembered under each loop's exact points, from one call
    of a loop rule to the next (*memo*, a dict a caller keeps per graph and
    mask: ``assemble.consolidate_lumen``'s). The measures read only the points
    and the mask, so a loop the last call measured is not measured again;
    each call keeps only the loops it asked about. The clean-up calls each
    loop rule nine times on graphs that change in a few places, and measuring
    every loop again was ~half of each call (29_2_f contra, 2026-10-20)."""

    def __init__(self, memo: dict | None, kind: str, measure: Callable[[np.ndarray], Any]):
        self._memo, self._kind, self._measure = memo, kind, measure
        self._before = memo.get(kind, {}) if memo is not None else {}
        self._now: dict = {}

    def __call__(self, loop_um: np.ndarray) -> Any:
        if self._memo is None:
            return self._measure(loop_um)
        key = np.ascontiguousarray(loop_um, dtype=float).tobytes()
        if key in self._now:
            return self._now[key]
        value = self._before[key] if key in self._before else self._measure(loop_um)
        self._now[key] = value
        return value

    def keep(self) -> None:
        if self._memo is not None:
            self._memo[self._kind] = self._now


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


#: Of two strands in one lumen, the one with this much more of its length
#: (µm) on a thick vessel's centreline tree is that vessel and stays.
THICK_CENTRELINE_LEAD_UM = 3.0

#: ...and the other goes, unless it is a vessel of its own beside the thick
#: one: its lumen at least this wide (µm, the median mask radius along it)
#: and lying outside the kept strand's lumen (:data:`BESIDE_THICK_LUMEN_RADII`).
#: Keeping the strand on the tree made the clean-up take capillaries running
#: beside the thick vessel, which had been its only route before (grids W
#: and X). Harvey's grids Y and Z (2026-10-15, E14.5: 50 pair and loop
#: removals beside a kept thick centreline, marked keep or remove): the
#: kept ones mostly 3-8 um in radius, the removed 1-2 um; right 39/50 with
#: this, against 28 for removing all.
VESSEL_BESIDE_THICK_RADIUS_UM = 2.3

#: Outside the kept strand's lumen: the side's median distance from that
#: strand, in the larger of the two lumen radii there. A second strand in the
#: same lumen lies nearer than the lumen's radius, however wide it is.
BESIDE_THICK_LUMEN_RADII = 1.0


def _vessel_beside(G: nx.MultiGraph, side, kept, support: MaskSupport) -> bool:
    """Whether *side* (a run of edge keys) is a vessel of its own beside
    *kept*, not a second strand in its lumen (:data:`VESSEL_BESIDE_THICK_RADIUS_UM`)."""
    side_points = densify_polyline(_polyline(G, side), max_step_um=0.5)
    side_radius = support.radius(side_points)
    if float(np.median(side_radius)) < VESSEL_BESIDE_THICK_RADIUS_UM:
        return False
    kept_points = densify_polyline(_polyline(G, kept), max_step_um=0.5)
    distance, nearest = cKDTree(kept_points).query(side_points)
    lumen = np.maximum(np.maximum(support.radius(kept_points)[nearest], side_radius), 0.5)
    return float(np.median(distance / lumen)) >= BESIDE_THICK_LUMEN_RADII


def _removal_order(
    G: nx.MultiGraph,
    runs: list,
    support: MaskSupport,
    along_centreline: Callable[[np.ndarray], float] | None = None,
) -> list:
    """*runs* (each a list of edge keys) in the order they go: one
    following a thick vessel's centreline tree last
    (*along_centreline*, µm of a polyline on it: a
    ``thick_vessel_junctions.ThickCentreline``'s ``along_um``), and
    otherwise the one nearest the wall first (``_wall_hugging``). Judged by
    the wall alone, a strand on the tree whose edge ran on into narrower
    vessels averaged nearer the wall than a strand lying wholly in the wide
    lumen, and went: the thick vessel was left broken (E14.5, 2026-10-15).

    Empty when the run that would go first, off the tree, is a vessel of its
    own beside the one on it (:func:`_vessel_beside`): the caller then leaves
    the pair or the loop as it is."""
    wall = [_wall_hugging(G, run, support) for run in runs]
    if along_centreline is None:
        order = sorted(range(len(runs)), key=lambda i: wall[i])
    else:
        along = [along_centreline(_polyline(G, run)) for run in runs]
        best = max(along)
        order = sorted(range(len(runs)), key=lambda i: (along[i] >= best - THICK_CENTRELINE_LEAD_UM, wall[i]))
        first = order[0]
        if along[first] < best - THICK_CENTRELINE_LEAD_UM:
            on_tree = max(range(len(runs)), key=lambda i: along[i])
            if _vessel_beside(G, runs[first], runs[on_tree], support):
                return []
    return [runs[i] for i in order]


def _break(
    G: nx.MultiGraph,
    cycle,
    support: MaskSupport,
    along_centreline: Callable[[np.ndarray], float] | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Take out the loop's arc running nearest the lumen's wall -- the whole
    loop when only one place meets the network, its wall-most edge when none
    does -- and the nodes left with nothing. Returns the edges taken out.

    The arc kept is the one nearest the lumen's middle: the arc along the
    rim covered less of the lumen, and the diameter read off the mask along
    it would come out too small. With *along_centreline*, an arc on a thick
    vessel's centreline tree is kept first (:func:`_removal_order`), and
    nothing is taken out when the arc that would go is a vessel beside it."""
    order = _removal_order(G, list(loop_sides(G, cycle)), support, along_centreline)
    if not order:
        return []
    arc = order[0]
    for a, b, key in arc:
        G.remove_edge(a, b, key)
    for a, b, _key in arc:
        for node in (a, b):
            if G.has_node(node) and G.degree[node] == 0:
                G.remove_node(node)
    return arc


def remove_loops_inside_one_lumen(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    search_um: float = LOOP_SEARCH_UM,
    along_centreline: Callable[[np.ndarray], float] | None = None,
    memo: dict | None = None,
) -> nx.MultiGraph:
    """Break, in place, every loop of *G* lying inside one lumen of
    *support*'s mask (:func:`loop_inside_one_lumen`).

    Each edge's shortest loop within *search_um* is found and tested; one
    inside a lumen loses its arc nearest the wall -- the run of edges between two
    places the rest of the network meets it -- so the vessel goes on, drawn
    once, and no spur is left hanging. A loop only one place meets goes
    whole. Runs until no loop inside a lumen is left; a loop round tissue,
    however short, stays. *along_centreline*: see :func:`_removal_order`; a
    loop whose side would be a vessel beside a thick one stays too. *memo*:
    see :class:`_LoopMeasures`.
    """
    if G.number_of_edges() == 0:
        return G
    inside_one_lumen = _LoopMeasures(memo, "inside_one_lumen", lambda loop: loop_inside_one_lumen(loop, support))
    removed: list[float] = []
    # Loops left as they are: round tissue, or through a vessel beside a
    # thick one. Every edge of one finds it again, pass after pass.
    round_tissue: set[frozenset] = set()
    left_beside = 0
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
            if not inside_one_lumen(_polyline(G, cycle)):
                round_tissue.add(signature)
                continue
            lengths = {(a, b, key): _length(G.edges[a, b, key]) for a, b, key in cycle}
            taken = _break(G, cycle, support, along_centreline)
            if not taken:
                round_tissue.add(signature)
                left_beside += 1
                continue
            for a, b, key in taken:
                removed.append(lengths[(a, b, key)])
                if G.has_node(a) and G.has_node(b):
                    pairs.refresh(a, b)
                else:
                    pairs.set_weight(a, b, _UNUSABLE)
            changed = True
    inside_one_lumen.keep()
    if left_beside:
        logger.info(
            "Left %d loop(s) inside one lumen whose side would have been a vessel beside a thick one",
            left_beside,
        )
    if removed:
        logger.info(
            "Broke loops lying inside one lumen (Lee-thinning rings): took out %d edge(s), %.0f um",
            len(removed),
            float(sum(removed)),
        )
    return G


#: Loops up to this long round are removed by :func:`remove_short_loops`
#: when a run asks for it (``remove_short_loops_under_um``): the review
#: list's own limit, so what it removes is what the list would have shown.
DEFAULT_SHORT_LOOP_UM = LOOP_SEARCH_UM

#: A short loop round a tissue hole wider than this -- the median background
#: its antipodal chords cross (:func:`loop_hole_um`, ``loop_measures``'
#: ``chord_bg_median_um``) -- is left to the review list rather than broken
#: (``short_loop_keep_hole_over_um``). Harvey's grids Q and Q2 (2026-10-15,
#: 60 of the rule's breaks on E14.5, shown with the side it took): over 10 um
#: 29 of 32 were real loops, under it 24 of 28 were not; 53/60 right
#: leave-one-out against 33 for marking all one way. The six grid I/I2 loops
#: shown again in Q2 all came back real, the four marked false included.
DEFAULT_SHORT_LOOP_KEEP_HOLE_UM = 10.0


def loop_hole_um(loop_um: np.ndarray, support: MaskSupport) -> float | None:
    """How wide a hole a closed path runs round: the median background its
    antipodal chords cross, in microns -- ``loop_measures``'
    ``chord_bg_median_um``. None when the path is too short to have chords."""
    background = _chord_background_um(loop_um, support)
    return None if background is None else float(np.median(background))


def remove_short_loops(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    max_length_um: float = DEFAULT_SHORT_LOOP_UM,
    keep_hole_over_um: float = DEFAULT_SHORT_LOOP_KEEP_HOLE_UM,
    along_centreline: Callable[[np.ndarray], float] | None = None,
    memo: dict | None = None,
) -> nx.MultiGraph:
    """Break, in place, every loop of *G* up to *max_length_um* round, as
    :func:`remove_loops_inside_one_lumen` breaks a loop inside one lumen --
    the arc nearest the wall goes -- whether or not the mask says it runs
    round tissue, unless the hole it runs round is wider than
    *keep_hole_over_um* (:func:`loop_hole_um`).

    Harvey's grids I and I2 (2026-10-13, 60 short loops of E14.5 judged on
    the claudin-5 channel): 55 closed inside one vessel, and the mask rule
    had found none of them, since the background it reads between the two
    sides as tissue was not tissue there. No measure, of the mask or the raw
    image, found the five real ones (46-179 um round), and no length cut did
    better than taking a side of every short loop, which is right 92-93% of
    the time at the cost of one real loop in twelve. Grids Q and Q2
    (2026-10-15) then showed 60 of the loops this broke on E14.5 with the
    side it took: where a side should go it took the right one every time,
    but most loops round a wide hole were real
    (:data:`DEFAULT_SHORT_LOOP_KEEP_HOLE_UM`), and breaking one merged it
    with its neighbour into a longer loop that was broken in turn. Those
    loops are left to the Short loops review list. 0 for *max_length_um*
    turns the rule off; 0 for *keep_hole_over_um* breaks every short loop.
    *along_centreline*: see :func:`_removal_order`. *memo*: see
    :class:`_LoopMeasures`.
    """
    if G.number_of_edges() == 0 or not max_length_um > 0:
        return G
    hole_um = _LoopMeasures(memo, "hole_um", lambda loop: loop_hole_um(loop, support))
    removed: list[float] = []
    # Whether each loop, by its edges, is left alone -- round a wide hole, or
    # its side a vessel beside a thick one: every edge of a loop left alone
    # finds that loop again, pass after pass.
    wide_hole: dict[frozenset, bool] = {}
    kept_for_hole = 0
    changed = True
    while changed:
        changed = False
        pairs = _Pairs(G)
        for u, v in pairs.pairs:
            if not G.has_edge(u, v):
                continue
            cycle = pairs.shortest_cycle(u, v, max_length_um)
            if cycle is None:
                continue
            lengths = {(a, b, key): _length(G.edges[a, b, key]) for a, b, key in cycle}
            if sum(lengths.values()) > max_length_um:
                continue
            # A loop through a thick-vessel bridge is the join to the thick
            # centreline, not two strands: left to the review list. Taking a
            # side of those cost E14.5 32 of its 110 bridges. A taper, the
            # thick vessel narrowing inside its own lumen, likewise.
            if any(
                G.edges[a, b, key].get(IS_ZERO_RESISTANCE) or G.edges[a, b, key].get(THICK_TAPER)
                for a, b, key in cycle
            ):
                continue
            signature = frozenset((a, b, key) if str(a) <= str(b) else (b, a, key) for a, b, key in cycle)
            if keep_hole_over_um > 0:
                if signature not in wide_hole:
                    hole = hole_um(_polyline(G, cycle))
                    wide_hole[signature] = hole is not None and hole > keep_hole_over_um
                    kept_for_hole += int(wide_hole[signature])
                if wide_hole[signature]:
                    continue
            elif wide_hole.get(signature):
                continue
            taken = _break(G, cycle, support, along_centreline)
            if not taken:
                # Its side would be a vessel beside a thick one: left alone.
                wide_hole[signature] = True
                continue
            for a, b, key in taken:
                removed.append(lengths[(a, b, key)])
                if G.has_node(a) and G.has_node(b):
                    pairs.refresh(a, b)
                else:
                    pairs.set_weight(a, b, _UNUSABLE)
            changed = True
    hole_um.keep()
    if removed or kept_for_hole:
        logger.info(
            "Broke every loop under %.0f um round but those round a hole over %.0f um (grids I, I2, Q, "
            "Q2): took out %d edge(s), %.0f um; left %d loop(s) round a wide hole to the review list",
            max_length_um, keep_hole_over_um, len(removed), float(sum(removed)), kept_for_hole,
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


#: Whether :func:`remove_parallel_edges_in_lumen` rewires the edge of a pair
#: that goes (``doubled_stretches.rewire_pair``) rather than deleting it
#: whole, when a caller does not say. Off: on E14.5 (2026-10-25, the clean-up
#: on one graph either way) rewiring kept 0.9 mm more centreline and 0.7
#: points more of the mask covered, but 9 more short loops -- all round holes
#: of 10 um or more, which grids Q/Q2 found mostly real -- and 5 more interior
#: dead ends, the kept remainders' own tips. PLAN_clean_graph.md's kill
#: criteria allow neither, so it stays off until Harvey decides.
REWIRE_PAIRS = False


def remove_parallel_edges_in_lumen(
    G: nx.MultiGraph, support: MaskSupport, *, memo: dict | None = None,
    keep_apart: Callable[[tuple], bool] | None = None,
    along_centreline: Callable[[np.ndarray], float] | None = None,
    rewire: bool | None = None,
) -> nx.MultiGraph:
    """Take out, in place, one edge of each pair
    :func:`graph.diagnostics.diagnose_parallel_duplicates_in_lumen` reports.

    The edge nearer the wall goes (``_wall_hugging``), so the centreline
    that stays is the one nearer the lumen's middle. With *rewire* (default
    :data:`REWIRE_PAIRS`, off) it goes the way
    ``doubled_stretches.rewire_pair`` takes it -- the stretch beside the
    other cut out, what is left joined to the other where it parts -- and is
    deleted whole only when nothing is left to join, or every join would
    close a loop inside one lumen (PLAN_clean_graph.md 1d); what a rewire
    left (``doubled_stretches.REWIRED``) is deleted, not rewired again, or
    the clean-up never settled. Deleting it is refused where that would cut
    vessels off from the rest of the network (:func:`cuts_off_vessels`): the
    other goes instead, the same two ways, or, when neither can, both stay.
    A branch meeting its parent, and two vessels with background between
    them, are not pairs in that report and are left alone. *memo* is the
    report's own, for a caller asking again and again of one graph
    (``assemble.consolidate_lumen``). *keep_apart*, called with the pair
    ``(left, right)`` of edge keys, is evidence beyond the mask that they
    are two vessels: a pair it returns True for is left alone. With
    *along_centreline*, an edge on a thick vessel's centreline tree is kept
    over one that is not (:func:`_removal_order`), and both stay when the
    other is a vessel of its own beside it.
    """
    from haemolynx.graph.diagnostics import diagnose_parallel_duplicates_in_lumen
    from haemolynx.graph.doubled_stretches import REWIRED, rewire_pair

    report = diagnose_parallel_duplicates_in_lumen(
        G,
        support.mask,
        voxel_size_zyx=tuple(float(v) for v in support.voxel_size_zyx),
        mask_support=support,
        memo=memo,
    )
    if rewire is None:
        rewire = REWIRE_PAIRS
    removed = rewired = kept_connected = kept_beside = 0
    for left, right in report["duplicate_pairs"]:
        present = [edge for edge in (left, right) if G.has_edge(*edge)]
        if len(present) < 2:
            continue
        # A pair with a thick-vessel bridge is left alone: a bridge runs
        # inside the thick lumen by construction, so it is always beside
        # the thick centreline, and it is the join to it, not a second
        # strand. The loop step and the dangling-bridge rule settle those.
        # A taper lies in the thick lumen for the same reason.
        if any(G.edges[edge].get(IS_ZERO_RESISTANCE) or G.edges[edge].get(THICK_TAPER) for edge in present):
            continue
        if keep_apart is not None and keep_apart((left, right)):
            continue
        candidates = [run[0] for run in _removal_order(G, [[edge] for edge in present], support, along_centreline)]
        if not candidates:
            kept_beside += 1
            continue
        for drop in candidates:
            if rewire and not G.edges[drop].get(REWIRED) and rewire_pair(
                G, support, drop, next(e for e in present if e != drop)
            ):
                rewired += 1
                break
            if not cuts_off_vessels(G, drop):
                u, v, key = drop
                G.remove_edge(u, v, key)
                removed += 1
                for node in (u, v):
                    if G.has_node(node) and G.degree(node) == 0:
                        G.remove_node(node)
                break
        else:
            kept_connected += 1
    G.graph["parallel_lumen_edges_removed"] = int(G.graph.get("parallel_lumen_edges_removed", 0)) + removed
    G.graph["parallel_lumen_edges_rewired"] = int(G.graph.get("parallel_lumen_edges_rewired", 0)) + rewired
    if removed or rewired or kept_connected or kept_beside:
        logger.info(
            "Parallel edges lying beside another in one lumen: rewired %d, their stretch cut out and "
            "what was left joined where it parts; removed %d whole; kept %d pair(s) where either "
            "edge was the only link to vessels beyond it, and %d where one was a vessel beside a "
            "thick one",
            rewired,
            removed,
            kept_connected,
            kept_beside,
        )
    return G


__all__ = [
    "LOOP_SEARCH_UM",
    "MEASURE_MARGIN_UM",
    "cuts_off_vessels",
    "iter_short_loops",
    "loop_centre_um",
    "loop_hole_um",
    "loop_inside_one_lumen",
    "loop_measures",
    "loop_sides",
    "remove_loops_inside_one_lumen",
    "remove_parallel_edges_in_lumen",
]
