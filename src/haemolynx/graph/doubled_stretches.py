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
rule that kept pairs the whole-edge rule had deleted. Here the pair rule
still decides which edge of a pair goes; with ``lumen_loops.REWIRE_PAIRS``
:func:`rewire_pair` takes it out the same way -- its stretch beside the
other cut out, what is left joined where it parts (PLAN_clean_graph.md 1d)
-- rather than deleting it whole; this step runs after it on what it leaves.

A join can be refused when the loop it closes lies inside one lumen
(``lumen_loops.loop_inside_one_lumen``): the shortest loop through it, round
the rest of the network, the edge rewired left out -- always for the pair
rewire, and for this step's rejoin with :data:`GUARD_REJOINS`. The failed
attempts (the parked branch, the missed-branches session's trim-and-rejoin)
raised the short loops because a rejoin could close one. Both switches are
off: on E14.5 each failed the plan's kill criteria (see their notes).
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
from .lumen_loops import LOOP_SEARCH_UM, loop_inside_one_lumen
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

#: Whether the stretch step's rejoin, by default, makes no join that would
#: close a loop inside one lumen (:func:`_join_closes_loop_in_one_lumen`; the
#: pair rewire always tests its joins). Off: on E14.5 (2026-10-25, the graph
#: as ``scripts/graph_artefacts.py`` saves it, against 7f7e699c) it took 73 um
#: of doubled centreline out and covered 0.9 points more of the mask, but left
#: 7 more short loops -- loops round holes of 10 um or more, which grids Q/Q2
#: found mostly real -- 3 more interior dead ends (a remainder not joined can
#: leave its junction a stub) and 1 more tip in another lumen.
#: PLAN_clean_graph.md's kill criteria allow none of those, so it waits on
#: Harvey, as the pair rewire does (``lumen_loops.REWIRE_PAIRS``).
GUARD_REJOINS = False

#: Edge mark: what :func:`rewire_pair` left of an edge it took out. Such an
#: edge, found in a pair again, is deleted rather than rewired once more: on
#: E14.5 a rewire made the next round's pair one round after another, and the
#: clean-up ran to its round limit (2026-10-25).
REWIRED = "rewired"

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


def _runs_of(partner: np.ndarray, gap_samples: int):
    """``(which, start, stop)`` for each run of samples beside one partner
    (*partner*: per sample, its partner's number or -1), skipping up to
    *gap_samples* samples beside nothing or beside another."""
    i = 0
    while i < len(partner):
        if partner[i] < 0:
            i += 1
            continue
        which, start, stop, gap, j = partner[i], i, i, 0, i + 1
        while j < len(partner):
            if partner[j] == which:
                stop, gap = j, 0
            elif gap < gap_samples:
                gap += 1
            else:
                break
            j += 1
        i = stop + 1
        yield which, start, stop


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
    for which, start, stop in _runs_of(partner, gap_samples):
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
        #: Joins not made, each closing a loop inside one lumen.
        self.refused = 0
        #: Attributes every edge added gets.
        self.marks: dict = {}

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
        attrs.update(self.marks)
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


def _hiding(hidden: set):
    """A Dijkstra weight over a MultiGraph: the shortest of the edges joining
    two nodes, the edges *hidden* (``edge_id``s) left out."""

    def weight(p, q, keyed: dict):
        lengths = [float(d.get("length") or 0.0) for k, d in keyed.items() if edge_id(p, q, k) not in hidden]
        return min(lengths) if lengths else None

    return weight


def _loop_through_join(
    G: nx.MultiGraph, node, partner, partner_path: np.ndarray, at: int, join_points: np.ndarray, hidden: set,
) -> np.ndarray | None:
    """The loop joining *node* to *partner* at vertex *at* of its path from
    its first node (along *join_points*, from *node*) would close: the
    shortest, within :data:`lumen_loops.LOOP_SEARCH_UM`, through the join, the
    partner from there to one of its ends, and the rest of the network back
    to *node*, the edges *hidden* left out. Its points in microns, or None
    when there is none that short."""
    a, b, _key = partner
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(partner_path, axis=0), axis=1))])
    legs = (
        (a, float(arc[at]), partner_path[: at + 1][::-1]),
        (b, float(arc[-1] - arc[at]), partner_path[at:]),
    )
    join_um = _length(join_points)
    limit = LOOP_SEARCH_UM - join_um - min(leg_um for _end, leg_um, _points in legs)
    if limit < 0:
        return None
    weight = _hiding(hidden)
    previous, distance = nx.dijkstra_predecessor_and_distance(G, node, cutoff=limit, weight=weight)
    best = None
    for end, leg_um, leg_points in legs:
        if end in distance:
            total = join_um + leg_um + distance[end]
            if total <= LOOP_SEARCH_UM and (best is None or total < best[0]):
                best = (total, end, leg_points)
    if best is None:
        return None
    _total, end, leg_points = best
    route = [end]
    while route[-1] != node:
        route.append(previous[route[-1]][0])
    pieces = [np.asarray(join_points, dtype=float), np.asarray(leg_points, dtype=float)]
    for p, q in zip(route[:-1], route[1:]):
        key = min(
            (k for k in G[p][q] if edge_id(p, q, k) not in hidden),
            key=lambda k: float(G.edges[p, q, k].get("length") or 0.0),
        )
        positions = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (p, q)}
        pieces.append(edge_sample_points(p, q, G.edges[p, q, key], positions))
    return np.vstack(pieces)


def _join_closes_loop_in_one_lumen(
    G: nx.MultiGraph, support: MaskSupport, node, partner, partner_path: np.ndarray, at: int,
    join_points: np.ndarray, hidden: set,
) -> bool:
    """Whether the join :func:`_loop_through_join` describes closes a loop
    lying inside one lumen (``lumen_loops.loop_inside_one_lumen``): that join
    is then not made. Lee thinning's rings and a strand rejoined to the
    vessel it doubled both close one; the in-lumen loop rule would break it
    again, taking its arc nearest the wall, whichever that was."""
    loop = _loop_through_join(G, node, partner, partner_path, at, join_points, hidden)
    return loop is not None and loop_inside_one_lumen(loop, support)


def _rejoin(editor: _Editor, run: DoubledStretch, support: MaskSupport | None = None) -> bool:
    """Cut the stretch out of the edge nearer the wall and join its two
    remaining parts to the partner where they leave it. With *support*, a
    part whose join would close a loop inside one lumen
    (:func:`_join_closes_loop_in_one_lumen`) is not joined -- it is joined to
    the network already, round that loop -- and goes with the stretch; when
    neither part can be joined the stretch is left as it is."""
    u, v, _key = run.edge
    a, b, _pkey = run.partner
    partner_path = editor.path(run.partner, a)
    last = len(partner_path) - 1
    ends = {0: a, last: b}

    def at_partner(point):
        """The partner's vertex nearest *point*: its index, or the end node
        it is when that vertex is one of the partner's two ends."""
        index = int(np.argmin(np.linalg.norm(partner_path - point, axis=1)))
        return (index, None) if 0 < index < last else (None, ends[index])

    first, first_node = at_partner(run.samples[run.start])
    second, second_node = at_partner(run.samples[run.stop])
    if first_node is not None and second_node is not None and first_node == second_node:
        return False
    # (the remainder's node, where it joins the partner, and its points from
    # the node). A remainder joining the partner at the node it already
    # starts from (the edge leaves a node the partner ends on) would be a
    # self-loop: it has nothing left, and goes.
    joins = [
        (node, index, end_node, remainder)
        for node, index, end_node, remainder in (
            (u, first, first_node, run.samples[: run.start + 1]),
            (v, second, second_node, run.samples[run.stop:][::-1]),
        )
        if node != end_node
    ]
    if support is not None and joins:
        hidden = {edge_id(*run.edge), edge_id(*run.partner)}

        def closes_loop(node, index, end_node, remainder) -> bool:
            vertex = index if index is not None else (0 if end_node == a else last)
            join = np.vstack([remainder, partner_path[vertex]])
            return _join_closes_loop_in_one_lumen(editor.G, support, node, run.partner, partner_path, vertex, join, hidden)

        kept = [join for join in joins if not closes_loop(*join)]
        editor.refused += len(joins) - len(kept)
        if not kept:
            return False
        joins = kept
    # Split the partner where the remainders join it -- once where both join
    # at the same vertex, not at all where they join at its own ends. The
    # chords are not held to the run's length (see _zip).
    indices = sorted({index for _node, index, _end, _points in joins if index is not None})
    at = dict(zip(indices, editor.split(run.partner, a, indices) if indices else []))
    pieces = []
    for node, index, end_node, _remainder in joins:
        target = end_node if index is None else at[index]
        if node == u:
            pieces.append((u, target, np.vstack([run.samples[: run.start + 1], editor.pos(target)])))
        else:
            pieces.append((target, v, np.vstack([editor.pos(target), run.samples[run.stop:]])))
    editor.replace(run.edge, pieces)
    return True


def _stretch_beside(G: nx.MultiGraph, support: MaskSupport, edge, partner) -> DoubledStretch | None:
    """The longest stretch of *edge* running beside *partner* in one lumen,
    judged sample by sample as :func:`doubled_stretches` judges it but
    against *partner* alone, or None when no sample is beside it."""
    u, v, _key = edge
    a, b, _pkey = partner
    node_pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v, a, b)}
    step = _sample_step(support.voxel_size_zyx)
    path = edge_sample_points(u, v, G.edges[edge], node_pos)
    beside_points = _densify(edge_sample_points(a, b, G.edges[partner], node_pos), step)
    samples, tangents, judged = _judged_samples(path, support.radius, step_um=step, margin_um=SAME_LUMEN_MARGIN_UM)
    if not judged.any() or len(beside_points) == 0:
        return None
    neighbour = np.full(len(samples), -1, dtype=np.intp)
    neighbour[np.flatnonzero(judged)] = _same_lumen_neighbours(
        samples[judged], tangents[judged], cKDTree(beside_points), beside_points, support.inside, support.radius,
        step_um=step, margin_um=SAME_LUMEN_MARGIN_UM, max_cosine=MAX_SHADOW_COSINE,
    )
    gap_samples = max(1, int(np.ceil(RUN_GAP_UM / step)))
    runs = [(start, stop) for _which, start, stop in _runs_of(np.where(neighbour >= 0, 0, -1), gap_samples)]
    if not runs:
        return None
    start, stop = max(runs, key=lambda run: _length(samples[run[0]:run[1] + 1]))
    judged_at = np.flatnonzero(judged)
    beside = neighbour[start:stop + 1]
    beside = beside[beside >= 0]
    return DoubledStretch(
        edge=tuple(edge),
        partner=tuple(partner),
        samples=samples,
        start=start,
        stop=stop,
        partner_start=beside_points[neighbour[start]],
        partner_stop=beside_points[neighbour[stop]],
        length_um=_length(samples[start:stop + 1]),
        radius_um=float(np.mean(support.radius(samples[start:stop + 1]))),
        partner_radius_um=float(np.mean(support.radius(beside_points[beside]))),
        at_start=start <= int(judged_at[0]) + gap_samples,
        at_stop=stop >= int(judged_at[-1]) - gap_samples,
    )


def rewire_pair(G: nx.MultiGraph, support: MaskSupport, edge, partner) -> bool:
    """Take *edge* out of a pair in one lumen without deleting it whole
    (PLAN_clean_graph.md 1d): its stretch beside *partner* is cut out, and
    what is left at each end is joined to *partner* where it leaves it.

    A remainder that is nothing but the stretch's own end -- at a node the
    two share, or at a dead end's tip -- has nothing to join. A join that
    would close a loop inside one lumen (:func:`_join_closes_loop_in_one_lumen`)
    is not made; what it would have joined is joined to the network already,
    round that loop. Returns whether anything was joined. When nothing is,
    *G* is as it was, and the caller deletes *edge* as the pair rule always
    did: grids D-F found the undoubled remainder vessel 56 % of the time and
    no measure that says when, so it is kept wherever it can be (decision D2).
    """
    u, v, _key = edge
    a, b, _pkey = partner
    if u == v or a == b or {u, v} == {a, b}:
        return False
    run = _stretch_beside(G, support, edge, partner)
    if run is None:
        return False
    editor = _Editor(G)
    partner_path = editor.path(partner, a)
    last = len(partner_path) - 1
    hidden = {edge_id(*edge), edge_id(*partner)}
    joins = []
    for node, reaches_end, remainder in (
        (u, run.at_start, run.samples[: run.start + 1]),
        (v, run.at_stop, run.samples[run.stop:][::-1]),
    ):
        if reaches_end and (node in (a, b) or G.degree[node] == 1):
            continue
        at = int(np.argmin(np.linalg.norm(partner_path - remainder[-1], axis=1)))
        attach = a if at == 0 else b if at == last else None
        if attach == node:
            continue
        join = np.vstack([remainder, partner_path[at]])
        if _join_closes_loop_in_one_lumen(G, support, node, partner, partner_path, at, join, hidden):
            continue
        joins.append((node, at, attach, join))
    if not joins:
        return False
    inner = sorted({at for _node, at, attach, _join in joins if attach is None})
    split_at = dict(zip(inner, editor.split(partner, a, inner))) if inner else {}
    editor.marks = {REWIRED: True}
    editor.replace(edge, [
        (node, attach if attach is not None else split_at[at], join) for node, at, attach, join in joins
    ])
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
    guard_joins: bool | None = None,
) -> nx.MultiGraph:
    """Resolve, in place, every stretch of centreline :func:`doubled_stretches`
    finds running beside another in one lumen (see the module docstring).

    *guard_joins* (default :data:`GUARD_REJOINS`, off) has a rejoin make no
    join that would close a loop inside one lumen
    (:func:`_join_closes_loop_in_one_lumen`).

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
    if guard_joins is None:
        guard_joins = GUARD_REJOINS
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
            done = "rejoined" if _rejoin(editor, run, support if guard_joins else None) else None
        if done:
            counts[done] += 1
    if any(counts.values()) or editor.refused:
        logger.info(
            "Doubled stretches in one lumen: %d dead end(s) cut back, %d fork(s) zipped, "
            "%d stretch(es) cut out and rejoined; %.0f um of centreline taken out; %d join(s) not "
            "made, each closing a loop inside one lumen",
            counts["trimmed"], counts["zipped"], counts["rejoined"], editor.removed_um, editor.refused,
        )
    G.graph["doubled_stretches_resolved"] = int(G.graph.get("doubled_stretches_resolved", 0)) + sum(counts.values())
    return G


__all__ = [
    "MIN_DOUBLED_RUN_RADII",
    "GUARD_REJOINS",
    "MIN_DOUBLED_RUN_UM",
    "REWIRED",
    "DoubledStretch",
    "doubled_stretches",
    "resolve_doubled_stretches",
    "rewire_pair",
]
