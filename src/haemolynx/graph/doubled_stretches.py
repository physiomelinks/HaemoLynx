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
from dataclasses import dataclass, replace
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

from ._helpers import (
    EdgeSampleIndex, calculate_path_length, edge_id, edge_sample_points, next_node_id, points_within,
)
from .thick_vessel_junctions import IS_ZERO_RESISTANCE, THICK_TAPER

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
    memo: dict | None = None,
) -> list[DoubledStretch]:
    """Every stretch of every edge running beside one other centreline in the
    same lumen for at least *min_run_um* and *min_run_radii* lumen radii,
    longest first. Thick-vessel bridges are neither stretches nor partners.

    *memo*, a dict a caller passes to every call on one graph (and one
    mask), keeps each edge's verdict for the next call: an edge whose
    centreline is unchanged, with no edge added, changed or removed since
    within the reach it is judged over, is not judged again -- its runs
    read only its own path and the centrelines in that reach. The clean-up
    asks every round of a graph each round changes in a few places; judged
    whole every round, the finder was ~7-11 s a round on E14.5.
    """
    if G.number_of_edges() < 2:
        return []
    from .diagnostics import _edge_signature

    step = _sample_step(support.voxel_size_zyx)
    gap_samples = max(1, int(np.ceil(RUN_GAP_UM / step)))
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    index = EdgeSampleIndex(G)
    if index.tree is None:
        return []
    ids: dict[Any, int] = {}
    owners = np.asarray([ids.setdefault(owner, len(ids)) for owner in index.owners], dtype=np.intp)
    edge_of = list(ids)
    # Bridges and tapers run inside a thick lumen by construction: neither a
    # stretch nor a partner.
    bridges = {
        edge_id(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
        if d.get(IS_ZERO_RESISTANCE) or d.get(THICK_TAPER)
    }
    bridge_ids = np.asarray(sorted(ids[b] for b in bridges if b in ids), dtype=np.intp)
    remembered: dict = memo.get("edges", {}) if memo is not None else {}
    entries = []
    for u, v, key, data in G.edges(keys=True, data=True):
        eid = edge_id(u, v, key)
        own = ids.get(eid)
        if own is None or u == v or eid in bridges or u not in node_pos or v not in node_pos:
            continue
        signature = _edge_signature(eid, data, node_pos[u], node_pos[v]) if memo is not None else None
        if signature in remembered:
            dense, reach, _runs = remembered[signature]
            path = None
        else:
            path = edge_sample_points(u, v, data, node_pos)
            dense = _densify(path, step)
            if len(dense) < 2:
                continue
            reach = 2.0 * float(np.max(support.radius(dense))) + 1.0
        entries.append((u, v, key, data, signature, path, dense, reach))
    changed_tree = None
    if remembered:
        present = {e[4] for e in entries}
        changed = [remembered[sig][0] for sig in remembered if sig not in present]
        changed += [e[6] for e in entries if e[4] not in remembered]
        if changed:
            changed_tree = cKDTree(np.vstack(changed))
    kept: dict = {}
    runs: list[DoubledStretch] = []
    for u, v, key, data, signature, path, dense, reach in entries:
        if signature in remembered:
            # The judging path is denser than the index's samples a verdict
            # reads, so a change one sample step further out still counts.
            untouched = changed_tree is None or not np.isfinite(
                changed_tree.query(dense, k=1, distance_upper_bound=reach + step)[0]
            ).any()
            if untouched:
                runs.extend(remembered[signature][2])
                kept[signature] = remembered[signature]
                continue
            path = edge_sample_points(u, v, data, node_pos)
        edge_runs = _judge_edge(
            u, v, key, path, dense, reach, ids[edge_id(u, v, key)], index, owners, edge_of, bridge_ids, support,
            step=step, gap_samples=gap_samples, min_run_um=min_run_um, min_run_radii=min_run_radii,
        )
        runs.extend(edge_runs)
        if signature is not None:
            kept[signature] = (dense, reach, edge_runs)
    if memo is not None:
        memo["edges"] = kept
    runs.sort(key=lambda run: -run.length_um)
    return runs


def _judge_edge(u, v, key, path, dense, reach, own, index, owners, edge_of, bridge_ids, support, *,
                step, gap_samples, min_run_um, min_run_radii) -> list[DoubledStretch]:
    """The doubled stretches of one edge: its judged samples against every
    other centreline's points within *reach* of its path."""
    near = points_within(index.tree, dense, reach)
    near = near[(owners[near] != own) & ~np.isin(owners[near], bridge_ids)]
    if not len(near):
        return []
    samples, tangents, judged = _judged_samples(path, support.radius, step_um=step, margin_um=SAME_LUMEN_MARGIN_UM)
    if not judged.any():
        return []
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
    runs: list[DoubledStretch] = []
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
    return runs


class _Editor:
    """Splits and rewires edges, keeping each new edge's attributes."""

    def __init__(self, G: nx.MultiGraph):
        self.G = G
        self.next_id = next_node_id(G, set())
        self.touched: set[tuple] = set()
        #: Centreline taken out less centreline added, microns: kept here as
        #: edges go and come, not read off the whole graph after each change
        #: (that sum, twice per stretch, was 59 of the step's 70 s on E14.5).
        self.removed_um = 0.0
        #: Positions (rounded) of the junctions zips have left behind, across
        #: the clean-up's rounds: a zip that would put its new junction back
        #: on one is the previous round's zip undone, and is refused.
        self.zipped_at: set = set()

    @staticmethod
    def place(point) -> tuple:
        return tuple(np.round(np.asarray(point, dtype=float) * 2.0).astype(int).tolist())

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
        self.removed_um -= attrs["length"]
        key = self.G.add_edge(a, b, **attrs)
        self.touched.add(edge_id(a, b, key))

    def replace(self, edge, pieces) -> None:
        """Take *edge* out and add ``(a, b, points)`` pieces carrying its
        attributes; nodes left with no edge go."""
        u, v, key = edge
        template = dict(self.G.edges[u, v, key])
        self.removed_um += float(template.get("length", 0.0) or 0.0)
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


def _part_together(along_edge: float, along_partner: float, min_shared_um: float) -> bool:
    """Whether two edges leaving one node part at about the same distance
    along each (within a factor of two, and past *min_shared_um*). A fork
    does; where the parting points disagree, the edge re-attached at the
    other's parting point doubled back along the stretch kept, and the next
    round zipped it the other way: on E14.5 11 forks moved 4 um to and fro
    to the round limit."""
    if along_edge < min_shared_um or along_partner < min_shared_um:
        return False
    return 0.5 * along_edge <= along_partner <= 2.0 * along_edge


def _zip(editor: _Editor, run: DoubledStretch, shared, at_start: bool, min_shared_um: float = 0.0) -> bool:
    """Make two edges leaving *shared* side by side one up to where they part.

    A parting point less than *min_shared_um* along the edge kept from
    *shared* is refused: zipped there, the roles reverse at the new
    junction and the next round zips the other way (E14.5: 11 forks
    moving 0.4 um back and forth to the round limit)."""
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
        if where is None or not _part_together(_length(samples[: part + 1]), _length(partner_path[: where + 1]), min_shared_um):
            return False
        if editor.place(partner_path[where]) in editor.zipped_at:
            return False
        editor.zipped_at.add(editor.place(editor.pos(shared)))
        (node,) = editor.split(run.partner, shared, [where])
        editor.replace(run.edge, [(node, far, np.vstack([editor.pos(node), samples[part:]]))])
    else:
        where = _nearest_inner(partner_path, partner_part)
        inner = _nearest_inner(editor.path(run.edge, shared), samples[part])
        if where is None or inner is None or not _part_together(
            _length(samples[: part + 1]), _length(partner_path[: where + 1]), min_shared_um
        ):
            return False
        if editor.place(editor.path(run.edge, shared)[inner]) in editor.zipped_at:
            return False
        editor.zipped_at.add(editor.place(editor.pos(shared)))
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


def _mostly_on(along_centreline: Callable[[np.ndarray], float], points: np.ndarray) -> bool:
    """Whether at least half of *points* lies on a thick vessel's centreline
    tree. A stretch with no length on it never does: counted as on, every
    partner stretch narrowed to one point turned capillaries far from any
    thick vessel into the wall side and rewired them."""
    on = along_centreline(points) if len(points) > 1 else 0.0
    return on > 0.0 and on >= 0.5 * _length(points)


def _partner_stretch(G: nx.MultiGraph, run: DoubledStretch) -> np.ndarray:
    """The partner's centreline between where *run* starts and stops beside it."""
    a, b, key = run.partner
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
    path = np.asarray(edge_sample_points(a, b, G.edges[a, b, key], pos), dtype=float)
    if len(path) < 2:
        return path
    ends = cKDTree(path).query(np.vstack([run.partner_start, run.partner_stop]))[1]
    lo, hi = sorted(int(i) for i in ends)
    return path[lo : hi + 1]


def resolve_doubled_stretches(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    min_run_um: float = MIN_DOUBLED_RUN_UM,
    min_run_radii: float = MIN_DOUBLED_RUN_RADII,
    keep_apart: Callable[[DoubledStretch], bool] | None = None,
    memo: dict | None = None,
    along_centreline: Callable[[np.ndarray], float] | None = None,
) -> nx.MultiGraph:
    """Resolve, in place, every stretch of centreline :func:`doubled_stretches`
    finds running beside another in one lumen (see the module docstring).

    Each edge is changed at most once a call; a stretch whose edge or partner
    has already changed waits for the next call (``assemble.consolidate_lumen``
    repeats until nothing changes). Two edges joining the same two nodes are
    left to degree-2 merging, which drops a duplicate route. *memo*, a dict
    a caller passes to every call on one graph, remembers across calls where
    zips have put junctions, so none is zipped back. *keep_apart*,
    called with each stretch, is evidence beyond the mask that the two
    centrelines are two vessels (the raw image, say): a stretch it returns
    True for is left alone. The mask alone cannot tell (grids H and H2).

    *along_centreline* (µm of a polyline on a thick vessel's centreline
    tree: a ``thick_vessel_junctions.ThickCentreline``'s ``along_um``) says
    which of the two is that vessel: a stretch mostly on the tree is never
    cut back or rejoined beside one that is not, and the other is resolved
    onto it, whichever lies nearer the wall.
    """
    runs = doubled_stretches(G, support, min_run_um=min_run_um, min_run_radii=min_run_radii, memo=memo)
    if not runs:
        return G
    editor = _Editor(G)
    if memo is not None:
        # Across the clean-up's rounds: where zips have put junctions, so no
        # round zips a fork back to where the round before had it.
        editor.zipped_at = memo.setdefault("zipped_at", set())
    counts = {"trimmed": 0, "zipped": 0, "rejoined": 0}
    # A run of a single sample has no length to rewire: zipping it moved the
    # junction one vertex along and left the same one-sample run at the new
    # junction, so the clean-up never settled (E14.5: ~24 forks re-zipped
    # every round to its limit).
    too_short = 0.5 * _sample_step(support.voxel_size_zyx)
    for run in runs:
        edge, partner = run.edge, run.partner
        if run.length_um < too_short:
            continue
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
        if along_centreline is not None:
            on = _mostly_on(along_centreline, run.samples[run.start : run.stop + 1])
            partner_on = _mostly_on(along_centreline, _partner_stretch(G, run))
            if on and not partner_on:
                continue
            if partner_on and not on:
                # The partner is the thick vessel: the lumen's middle.
                run = replace(run, partner_radius_um=float("inf"))
        # A dead end beside a vessel is cut back; of two dead ends side by
        # side, the one nearer the wall is.
        trims = run.radius_um <= run.partner_radius_um or min(G.degree[a], G.degree[b]) > 1
        done = None
        if run.at_stop and G.degree[v] == 1 and trims:
            done = "trimmed" if _trim(editor, run, tip_at_stop=True) else None
        elif run.at_start and G.degree[u] == 1 and trims:
            done = "trimmed" if _trim(editor, run, tip_at_stop=False) else None
        elif run.at_start and run.at_stop:
            # The whole edge runs beside the partner: a whole-edge pair, the
            # pair rule's case, not a stretch to rewire.
            pass
        elif run.at_start and u in (a, b) and G.degree[u] >= 3:
            # A fork is two edges leaving a junction. At a degree-2 node the
            # two are one vessel folding back (a hairpin) or kinking: zipped,
            # the degree-2 merge joined the pieces again and the next round
            # zipped them again, 11 of them on E14.5 to the round limit.
            done = "zipped" if _zip(editor, run, u, at_start=True, min_shared_um=2 * too_short) else None
        elif run.at_stop and v in (a, b) and G.degree[v] >= 3:
            done = "zipped" if _zip(editor, run, v, at_start=False, min_shared_um=2 * too_short) else None
        elif run.radius_um < run.partner_radius_um:
            done = "rejoined" if _rejoin(editor, run) else None
        if done:
            counts[done] += 1
    if any(counts.values()):
        logger.info(
            "Doubled stretches in one lumen: %d dead end(s) cut back, %d fork(s) zipped, "
            "%d stretch(es) cut out and rejoined; %.0f um of centreline taken out",
            counts["trimmed"], counts["zipped"], counts["rejoined"], editor.removed_um,
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
