"""Stretches of centreline running beside another in one lumen, rewired.

A second edge through one lumen is not always a pair end to end
(``lumen_loops.remove_parallel_edges_in_lumen`` judges whole edges: half of
one beside another). A dead end runs back along the vessel it leaves; two
edges leave a junction side by side and part further on (a fork); a second
strand doubles part of a vessel. On E14.5 only 51 of 1,805 um of doubled
centreline lay in pairs (PLAN_clean_graph.md, section 1.3).

:func:`resolve_doubled_stretches` finds the doubled stretches sample by
sample (the test ``preprocessing.shadowed_samples`` applies, one sample at
a time) and resolves each by its shape, **rewiring rather than deleting**,
so whether the undoubled remainder is vessel never has to be answered:

- a stretch reaching a dead end's tip: the dead end is cut back to where it
  leaves the other centreline (the stub prune then judges what is left);
- a stretch two edges start at a node they share (a fork): they are zipped --
  the one nearer the lumen's middle is kept, split where the two part, and
  the other's remainder starts there;
- a stretch inside an edge: the edge nearer the wall loses it, and its two
  remaining parts are joined to the kept edge where they leave it.

Nothing is cut off: every part that stays is joined to the network, and
every change shortens the network's centreline, so repeating it settles. A
thick-vessel bridge is never a stretch, nor a partner: it runs inside the
thick lumen by construction. The parked first attempt (branch
``claude/phase2-doubled-stretches``) combined these operations with a pair
rule that kept pairs the whole-edge rule had deleted; here the pair rule is
as it was, and this runs after it on what it leaves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import (
    MAX_SHADOW_COSINE,
    SAME_LUMEN_MARGIN_UM,
    MaskSupport,
    _densify,
    _judged_samples,
    _same_lumen_neighbours,
    _sample_step,
)

from ._helpers import EdgeSampleIndex, calculate_path_length, edge_id, edge_sample_points, next_node_id
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

logger = logging.getLogger(__name__)

#: The shortest stretch resolved, in microns and in lumen radii: every run
#: the report counts. Grid H of PLAN_clean_graph.md (Harvey, 2026-10-12: 19
#: of 30 runs one vessel drawn twice, 11 two vessels or a branch leaving)
#: found no length, angle, gap or neck that tells the two apart, so no
#: minimum is calibrated and every run is resolved (decision D6): the cost
#: to two touching vessels is the shared stretch running on one centreline,
#: nothing deleted. The parked attempt's 3 um and two radii left nine tenths
#: of E14.5's doubling in place.
MIN_DOUBLED_RUN_UM = 0.0
MIN_DOUBLED_RUN_RADII = 0.0

#: Centreline a doubled stretch may skip -- samples beside nothing, or beside
#: another vessel -- and still be one stretch.
RUN_GAP_UM = 1.0


def _length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0


@dataclass
class DoubledStretch:
    """One doubled stretch: samples *start*..*stop* of *edge*, sampled from
    its first node to its second, beside *partner*."""

    edge: tuple
    partner: tuple
    samples: np.ndarray
    start: int
    stop: int
    partner_start: np.ndarray
    partner_stop: np.ndarray
    length_um: float
    radius_um: float
    partner_radius_um: float
    at_start: bool
    at_stop: bool


def doubled_stretches(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    min_run_um: float = MIN_DOUBLED_RUN_UM,
    min_run_radii: float = MIN_DOUBLED_RUN_RADII,
) -> list[DoubledStretch]:
    """Every stretch of every edge running beside one other centreline in the
    same lumen for at least *min_run_um* and *min_run_radii* lumen radii,
    longest first. Thick-vessel bridges are neither stretches nor partners."""
    if G.number_of_edges() < 2:
        return []
    step = _sample_step(support.voxel_size_zyx)
    gap_samples = max(1, int(np.ceil(RUN_GAP_UM / step)))
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    index = EdgeSampleIndex(G)
    if index.tree is None:
        return []
    ids: dict[Any, int] = {}
    owners = np.asarray([ids.setdefault(owner, len(ids)) for owner in index.owners], dtype=np.intp)
    edge_of = list(ids)
    bridges = {edge_id(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get(IS_ZERO_RESISTANCE)}
    bridge_ids = {ids[b] for b in bridges if b in ids}
    runs: list[DoubledStretch] = []
    for u, v, key, data in G.edges(keys=True, data=True):
        eid = edge_id(u, v, key)
        own = ids.get(eid)
        if own is None or u == v or eid in bridges:
            continue
        path = edge_sample_points(u, v, data, node_pos)
        dense = _densify(path, step)
        if len(dense) < 2:
            continue
        found = index.tree.query_ball_point(dense, r=2.0 * float(np.max(support.radius(dense))) + 1.0)
        near = np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in found]))
        near = near[(owners[near] != own) & ~np.isin(owners[near], list(bridge_ids))]
        if not len(near):
            continue
        samples, tangents, judged = _judged_samples(path, support.radius, step_um=step, margin_um=SAME_LUMEN_MARGIN_UM)
        if not judged.any():
            continue
        points = index.points[near]
        # Each sample is judged with its own lumen radius, as the report's
        # measure judges it: a stretch counts from the narrower centreline
        # only when it lies within that vessel's own width of the other --
        # inside the wider lumen. Judged with the wider of the two radii
        # instead (tried 2026-10-12), a capillary fused along a trunk's wall
        # counted as beside the trunk from its own side, and the trim then
        # took the whole capillary: on E14.5 5.7 mm of centreline and 3.7
        # points of mask coverage went with 430 um of doubling. What the
        # wider side alone sees (588 of 906 um left on E14.5) is a grid's
        # question, not a finder's.
        neighbour_judged = _same_lumen_neighbours(
            samples[judged], tangents[judged], cKDTree(points), points, support.inside, support.radius,
            step_um=step, margin_um=SAME_LUMEN_MARGIN_UM, max_cosine=MAX_SHADOW_COSINE,
        )
        neighbour = np.full(len(samples), -1, dtype=np.intp)
        neighbour[np.flatnonzero(judged)] = neighbour_judged
        point_of = np.where(neighbour >= 0, near[np.maximum(neighbour, 0)], -1)
        partner = np.where(point_of >= 0, owners[np.maximum(point_of, 0)], -1)
        judged_at = np.flatnonzero(judged)
        first, last = int(judged_at[0]), int(judged_at[-1])
        i = 0
        while i < len(samples):
            if partner[i] < 0:
                i += 1
                continue
            which, start, stop, gap, j = partner[i], i, i, 0, i + 1
            while j < len(samples):
                if partner[j] == which:
                    stop, gap = j, 0
                elif gap < gap_samples:
                    gap += 1
                else:
                    break
                j += 1
            i = stop + 1
            stretch = samples[start:stop + 1]
            length = _length(stretch)
            radius = float(np.mean(support.radius(stretch)))
            if length < max(min_run_um, min_run_radii * radius):
                continue
            beside = point_of[start:stop + 1]
            beside = beside[(beside >= 0) & (partner[start:stop + 1] == which)]
            runs.append(DoubledStretch(
                edge=(u, v, key),
                partner=edge_of[int(which)],
                samples=samples,
                start=start,
                stop=stop,
                partner_start=index.points[point_of[start]],
                partner_stop=index.points[point_of[stop]],
                length_um=length,
                radius_um=radius,
                partner_radius_um=float(np.mean(support.radius(index.points[beside]))),
                at_start=start <= first + gap_samples,
                at_stop=stop >= last - gap_samples,
            ))
    runs.sort(key=lambda run: -run.length_um)
    return runs


class _Editor:
    """Splits and rewires edges, keeping each new edge's attributes."""

    def __init__(self, G: nx.MultiGraph):
        self.G = G
        self.next_id = next_node_id(G, set())
        self.touched: set[tuple] = set()

    def pos(self, node) -> np.ndarray:
        return np.asarray(self.G.nodes[node]["pos"], dtype=float)

    def path(self, edge, from_node) -> np.ndarray:
        """The edge's points, from *from_node* to its other end."""
        u, v, key = edge
        other = v if from_node == u else u
        node_pos = {n: self.pos(n) for n in (u, v)}
        return edge_sample_points(from_node, other, self.G.edges[u, v, key], node_pos)

    def _add(self, a, b, points, template: dict) -> None:
        attrs = {k: val for k, val in template.items() if k not in ("voxels", "length")}
        points = [tuple(float(c) for c in p) for p in points]
        attrs["voxels"] = points
        attrs["length"] = float(calculate_path_length(points))
        key = self.G.add_edge(a, b, **attrs)
        self.touched.add(edge_id(a, b, key))

    def replace(self, edge, pieces) -> None:
        """Take *edge* out and add ``(a, b, points)`` pieces carrying its
        attributes; nodes left with no edge go."""
        u, v, key = edge
        template = dict(self.G.edges[u, v, key])
        self.G.remove_edge(u, v, key)
        self.touched.add(edge_id(u, v, key))
        for a, b, points in pieces:
            if len(points) >= 2:
                self._add(a, b, points, template)
        for node in (u, v):
            if self.G.has_node(node) and self.G.degree(node) == 0:
                self.G.remove_node(node)

    def split(self, edge, from_node, indices):
        """New nodes on *edge* at the vertices *indices* of its path from
        *from_node* (ascending, inside the path); returns them in order."""
        u, v, _key = edge
        other = v if from_node == u else u
        path = self.path(edge, from_node)
        nodes, pieces, previous, previous_index = [], [], from_node, 0
        for index in indices:
            node = self.next_id
            self.next_id += 1
            self.G.add_node(node, pos=np.asarray(path[index], dtype=float))
            pieces.append((previous, node, path[previous_index:index + 1]))
            nodes.append(node)
            previous, previous_index = node, index
        pieces.append((previous, other, path[previous_index:]))
        self.replace(edge, pieces)
        return nodes


def _nearest_inner(path: np.ndarray, point: np.ndarray) -> int | None:
    """Index of *path*'s vertex nearest *point*, or ``None`` when that is one
    of its two ends."""
    index = int(np.argmin(np.linalg.norm(path - point, axis=1)))
    return index if 0 < index < len(path) - 1 else None


def _trim(editor: _Editor, run: DoubledStretch, tip_at_stop: bool) -> bool:
    """Cut a dead end back to the stretch's near end."""
    u, v, _key = run.edge
    if tip_at_stop:
        keep, base = run.samples[: run.start + 1], u
    else:
        keep, base = run.samples[run.stop:][::-1], v
    if len(keep) < 2:
        editor.replace(run.edge, [])
        return True
    tip = editor.next_id
    editor.next_id += 1
    editor.G.add_node(tip, pos=np.asarray(keep[-1], dtype=float))
    editor.replace(run.edge, [(base, tip, keep)])
    return True


def _zip(editor: _Editor, run: DoubledStretch, shared, at_start: bool) -> bool:
    """Make two edges leaving *shared* side by side one up to where they part."""
    G = editor.G
    u, v, _key = run.edge
    samples = run.samples if at_start else run.samples[::-1]
    part = run.stop if at_start else len(run.samples) - 1 - run.start
    partner_part = run.partner_stop if at_start else run.partner_start
    far = v if at_start else u
    partner_path = editor.path(run.partner, shared)
    partner_far = run.partner[1] if run.partner[0] == shared else run.partner[0]
    # The chord from the parting point across to the edge kept is not held
    # to the run's length: the two lie in one lumen (that is what beside
    # means), and a short shared run with a wide gap -- the flare of a wide
    # vessel's junction -- was marked one vessel drawn twice as often as a
    # long one (grid H). A parting point at the partner's far end is a whole
    # edge doubled: a pair, left to the pair rule.
    if run.radius_um <= run.partner_radius_um:
        # This edge is the outer one: the partner is kept and split.
        where = _nearest_inner(partner_path, samples[part])
        if where is None:
            return False
        (node,) = editor.split(run.partner, shared, [where])
        editor.replace(run.edge, [(node, far, np.vstack([editor.pos(node), samples[part:]]))])
    else:
        where = _nearest_inner(partner_path, partner_part)
        inner = _nearest_inner(editor.path(run.edge, shared), samples[part])
        if where is None or inner is None:
            return False
        (node,) = editor.split(run.edge, shared, [inner])
        if not G.has_edge(*run.partner):
            return False
        editor.replace(run.partner, [(node, partner_far, np.vstack([editor.pos(node), partner_path[where:]]))])
    return True


def _rejoin(editor: _Editor, run: DoubledStretch) -> bool:
    """Cut the stretch out of the edge nearer the wall and join its two
    remaining parts to the partner where they leave it."""
    u, v, _key = run.edge
    a, b, _pkey = run.partner
    partner_path = editor.path(run.partner, a)
    ends = {0: a, len(partner_path) - 1: b}

    def at_partner(point):
        """The partner's vertex nearest *point*: its index, or the end node
        it is when that vertex is one of the partner's two ends."""
        index = int(np.argmin(np.linalg.norm(partner_path - point, axis=1)))
        return (index, None) if 0 < index < len(partner_path) - 1 else (None, ends[index])

    first, first_node = at_partner(run.samples[run.start])
    second, second_node = at_partner(run.samples[run.stop])
    if first_node is not None and second_node is not None and first_node == second_node:
        return False
    # Split the partner where the two remainders join it -- once where both
    # join at the same vertex, not at all where they join at its own ends.
    # The chords are not held to the run's length (see _zip).
    indices = sorted({i for i in (first, second) if i is not None})
    nodes = editor.split(run.partner, a, indices) if indices else []
    at = dict(zip(indices, nodes))
    to_start = first_node if first is None else at[first]
    to_stop = second_node if second is None else at[second]
    # A remainder joining the partner at the node it already starts from
    # (the edge leaves a node the partner ends on) would be a self-loop: it
    # has nothing left, and goes.
    pieces = [
        (u, to_start, np.vstack([run.samples[: run.start + 1], editor.pos(to_start)])),
        (to_stop, v, np.vstack([editor.pos(to_stop), run.samples[run.stop:]])),
    ]
    editor.replace(run.edge, [(x, y, points) for x, y, points in pieces if x != y])
    return True


def resolve_doubled_stretches(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    min_run_um: float = MIN_DOUBLED_RUN_UM,
    min_run_radii: float = MIN_DOUBLED_RUN_RADII,
    keep_apart: Callable[[DoubledStretch], bool] | None = None,
) -> nx.MultiGraph:
    """Resolve, in place, every stretch of centreline :func:`doubled_stretches`
    finds running beside another in one lumen (see the module docstring).

    Each edge is changed at most once a call; a stretch whose edge or partner
    has already changed waits for the next call (``assemble.consolidate_lumen``
    repeats until nothing changes). Two edges joining the same two nodes are
    left to degree-2 merging, which drops a duplicate route. *keep_apart*,
    called with each stretch, is evidence beyond the mask that the two
    centrelines are two vessels (the raw image, say): a stretch it returns
    True for is left alone. The mask alone cannot tell (grids H and H2).
    """
    runs = doubled_stretches(G, support, min_run_um=min_run_um, min_run_radii=min_run_radii)
    if not runs:
        return G
    editor = _Editor(G)
    counts = {"trimmed": 0, "zipped": 0, "rejoined": 0}
    removed_um = 0.0
    for run in runs:
        edge, partner = run.edge, run.partner
        if not (G.has_edge(*edge) and G.has_edge(*partner)):
            continue
        if keep_apart is not None and keep_apart(run):
            continue
        if edge_id(*edge) in editor.touched or edge_id(*partner) in editor.touched:
            continue
        u, v, _key = edge
        a, b, _pkey = partner
        if {u, v} == {a, b}:
            continue
        before = sum(float(d.get("length", 0.0) or 0.0) for _x, _y, d in G.edges(data=True))
        # A dead end beside a vessel is cut back; of two dead ends side by
        # side, the one nearer the wall is.
        trims = run.radius_um <= run.partner_radius_um or min(G.degree[a], G.degree[b]) > 1
        done = None
        if run.at_stop and G.degree[v] == 1 and trims:
            done = "trimmed" if _trim(editor, run, tip_at_stop=True) else None
        elif run.at_start and G.degree[u] == 1 and trims:
            done = "trimmed" if _trim(editor, run, tip_at_stop=False) else None
        elif run.at_start and u in (a, b):
            done = "zipped" if _zip(editor, run, u, at_start=True) else None
        elif run.at_stop and v in (a, b):
            done = "zipped" if _zip(editor, run, v, at_start=False) else None
        elif run.radius_um < run.partner_radius_um:
            done = "rejoined" if _rejoin(editor, run) else None
        if done:
            counts[done] += 1
            removed_um += before - sum(float(d.get("length", 0.0) or 0.0) for _x, _y, d in G.edges(data=True))
    if any(counts.values()):
        logger.info(
            "Doubled stretches in one lumen: %d dead end(s) cut back, %d fork(s) zipped, "
            "%d stretch(es) cut out and rejoined; %.0f um of centreline taken out",
            counts["trimmed"], counts["zipped"], counts["rejoined"], removed_um,
        )
    G.graph["doubled_stretches_resolved"] = int(G.graph.get("doubled_stretches_resolved", 0)) + sum(counts.values())
    return G


__all__ = [
    "MIN_DOUBLED_RUN_RADII",
    "MIN_DOUBLED_RUN_UM",
    "DoubledStretch",
    "doubled_stretches",
    "resolve_doubled_stretches",
]
