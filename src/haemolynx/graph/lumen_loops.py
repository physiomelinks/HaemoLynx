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

from haemolynx.preprocessing.bridge_mask_support import MaskSupport, _sample_step

from ._helpers import edge_sample_points

logger = logging.getLogger(__name__)

#: How long a loop is looked for, in microns. A longer one is left as it is:
#: lying inside one lumen, it would need a lumen some 60 um across.
LOOP_SEARCH_UM = 200.0

#: Stands in for an edge a search must not use: longer than any search.
_UNUSABLE = 1e12


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
    loop = np.asarray(loop_um, dtype=float).reshape(-1, 3)
    step = _sample_step(support.voxel_size_zyx)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    count = int(np.ceil(arc[-1] / step))
    if count < 4:
        return True
    along = np.arange(count) * (arc[-1] / count)
    points = np.column_stack([np.interp(along, arc, loop[:, axis]) for axis in range(3)])
    half = count // 2
    start, chords = points[:half], points[half:2 * half] - points[:half]
    lengths = np.linalg.norm(chords, axis=1)
    samples = int(np.ceil(lengths.max() / step)) + 1
    if samples < 2:
        return True
    t = np.linspace(0.0, 1.0, samples)
    flags = support.inside(
        (start[:, None, :] + t[None, :, None] * chords[:, None, :]).reshape(-1, 3)
    ).reshape(half, samples)
    background_um = (~flags).sum(axis=1) * lengths / (samples - 1)
    dropout = np.sum(background_um <= support.max_background_gap_um + 1e-9)
    return 2 * int(dropout) >= half


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


def _arcs(G: nx.MultiGraph, cycle):
    """The cycle cut at every node with an edge off it: the runs of edges
    between the places the rest of the network meets it."""
    meets = [G.degree[a] > 2 for a, _b, _key in cycle]
    if not any(meets):
        return []
    first = meets.index(True)
    turned = cycle[first:] + cycle[:first]
    arcs, current = [], []
    for edge in turned:
        if current and G.degree[edge[0]] > 2:
            arcs.append(current)
            current = []
        current.append(edge)
    arcs.append(current)
    return arcs


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
    arcs = _arcs(G, cycle)
    if not arcs:
        arcs = [[edge] for edge in cycle]
    arc = min(arcs, key=lambda run: _wall_hugging(G, run, support))
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
    "cuts_off_vessels",
    "iter_short_loops",
    "loop_inside_one_lumen",
    "remove_loops_inside_one_lumen",
    "remove_parallel_edges_in_lumen",
]
