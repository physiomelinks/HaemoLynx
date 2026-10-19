"""Draw back, along the run's skeleton, a vessel the graph lost round a wide hole.

The lumen clean-up judges two centrelines through one segmented lumen, and a loop
inside one, by the mask alone. On the surface plexus, where the segmentation fuses
neighbouring vessels into sheets, that takes vessels of their own: of 30 skeleton
pieces the E14.5 graph had lost (Harvey's grid N, 2026-10-18,
``PLAN_surface_coverage.md``), every one whose loop with the final graph ran round a
tissue hole wider than 10.3 um was a vessel (9 of 9), and 17 of the 21 round a
smaller hole were not. The two rules that had taken the vessels -- pairs in one lumen
and loops inside one -- never read the hole; the short-loop rule, which does
(``lumen_loops.loop_hole_um``, the same measure), was right 8 times in 9. So a lost
piece is drawn back when the loop it closes runs round a hole wider than the one the
short-loop rule keeps a loop round.

A piece is a 26-connected run of skeleton voxels no centreline comes within
:data:`REACH_UM` of (the census measure the grids were drawn with). It is judged --
the pool grid N was drawn from -- when it holds at least :data:`MIN_PIECE_UM` of
skeleton, at least :data:`MIN_MASK_FRACTION` of them in the mask and at least
:data:`MIN_FAR_FRACTION` further than two lumen radii + 3 um (and 6 um) from every
centreline, and it meets skeleton the graph reaches in two places. It is drawn back
along its own skeleton between the two contacts farthest apart through it, not whole:
in the tangles Lee leaves in a wide sheet, Harvey drew one vessel, not the mesh. Each
further place it meets the graph is then joined along its skeleton when the loop that
join closes runs round a wide hole too: on grid N2 (2026-10-18, the 21 vessels this drew
on E14.5, 19 kept) every join Harvey added did that.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import MaskSupport

from ._helpers import (
    EdgeSampleIndex,
    densify_polyline,
    duplicates_existing_vessel,
    edge_id,
    edge_sample_points,
)
from .edit import insert_node_on_edge
from .lumen_loops import DEFAULT_SHORT_LOOP_KEEP_HOLE_UM, loop_hole_um
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

logger = logging.getLogger(__name__)

#: A skeleton voxel no centreline comes within this of (um) is lost.
REACH_UM = 3.0
#: A lost piece holds at least this much skeleton (um, each voxel counted as the
#: step to its nearest neighbour in the piece: on E14.5's near-cubic voxels the
#: 20 voxels grid N's pool was drawn with, on a coarse z axis the same length)...
MIN_PIECE_UM = 20.0
#: ...at least this share of them in the mask (the raw segmentation; the skeleton was
#: thinned from it hole-filled and closed, so it runs on through dropouts). Harvey drew
#: every vessel E14.5's surface still lacked (2026-10-18): each was at least 58% in the
#: mask, and at 80% four of them, at 69-78%, were refused with nothing else let in...
MIN_MASK_FRACTION = 0.5
#: ...and at least this share a separate tube from every centreline: further than
#: two lumen radii + :data:`FAR_MARGIN_UM`, and :data:`FAR_MIN_UM`.
MIN_FAR_FRACTION = 0.5
FAR_MARGIN_UM = 3.0
FAR_MIN_UM = 6.0

#: Edge attribute on a vessel drawn back, and the hole (um) its loop ran round;
#: a further join of a drawn piece to the graph carries ``retraced_join`` too.
RETRACED = "retraced"
RETRACED_HOLE_UM = "retraced_hole_um"
RETRACED_JOIN = "retraced_join"
#: Which piece a drawn edge belongs to while its further joins are drawn.
_PIECE = "_retrace_piece"

#: Why a judged piece is not drawn back (``Retrace.reason``).
REFUSED_SMALL = "shorter than MIN_PIECE_UM"
REFUSED_OFF_MASK = "under MIN_MASK_FRACTION in the mask"
REFUSED_BESIDE = "not a separate tube from the vessels kept"
REFUSED_ONE_END = "meets the graph in fewer than two places"
REFUSED_BACKGROUND = "a join to the graph crosses more background than a bridge may"
REFUSED_SAME_LUMEN = "runs beside a vessel in the same lumen"
REFUSED_NO_LOOP = "closes no loop with the graph"
REFUSED_SMALL_HOLE = "its loop runs round a hole no wider than the limit"

_STEP_UM = 0.5
_NEAREST_CANDIDATES = 256
_NEIGHBOUR = np.sqrt(3.0) + 1e-6


@dataclass(frozen=True)
class LostSkeletonPiece:
    """A 26-connected run of skeleton voxels no centreline reaches."""

    #: Its voxels, ``(N, 3)`` indices into the skeleton.
    voxels: np.ndarray
    #: ``(piece voxel, reached voxel)`` index pairs, one per reached skeleton
    #: voxel 26-adjacent to the piece.
    contacts: tuple


@dataclass(frozen=True)
class Retrace:
    """What :func:`judge_lost_skeleton` said of one piece."""

    piece: LostSkeletonPiece
    #: The path it would be drawn along, physical ``(z, y, x)`` um, from the graph
    #: point one end joins to the other's; None when it got no further than that.
    path_um: Optional[np.ndarray]
    far_fraction: float
    mask_fraction: float
    #: The hole its loop with the graph runs round (``lumen_loops.loop_hole_um``).
    hole_um: Optional[float]
    loop_length_um: Optional[float]
    #: None when it is drawn back, else one of the ``REFUSED_*`` reasons.
    reason: Optional[str]

    @property
    def accepted(self) -> bool:
        return self.reason is None


def _edge_list(G: nx.MultiGraph) -> list:
    """``(edge id, u, v, data)`` of every edge of *G*, in its order."""
    return [(edge_id(u, v, key), u, v, data) for u, v, key, data in G.edges(keys=True, data=True)]


def _trial_copy(G: nx.MultiGraph) -> nx.MultiGraph:
    """``G.copy()`` -- the same nodes, edges, keys and data, in the same order
    (so the same shortest paths) -- written straight into its adjacency:
    ``add_edges_from`` made most of the cost of a copy taken per piece judged,
    about a second a copy on a 36,000-edge graph."""
    H = G.__class__()
    H.graph.update(G.graph)
    H._node.update((n, d.copy()) for n, d in G._node.items())
    adj = H._adj
    for n in H._node:
        adj[n] = H.adjlist_inner_dict_factory()
    for u, neighbours in G._adj.items():
        mine = adj[u]
        for v, keydict in neighbours.items():
            if v not in mine:
                copied = H.edge_key_dict_factory()
                for key, data in keydict.items():
                    copied[key] = data.copy()
                mine[v] = adj[v][u] = copied
    return H


class _LiveSamples:
    """The centreline samples (*sample*: an edge's polyline -> its points) of
    every edge of a graph *keep* (``data -> bool``) accepts, kept up to date by
    :meth:`update` as edges are split and drawn. Rebuilt whole after every
    piece and join, they were 98% of the step's time on a 36,000-edge graph
    (28_3_d contra patella, 2026-10-19: 110 pieces drawn in 9 hours). The
    graph's samples as first read stay in one KD-tree, those of edges removed
    since are skipped, and those of edges added since sit in a small tree of
    their own: the same samples a tree built over the graph as it stands holds,
    so the same nearest distances (an exact tie may name the other sample)."""

    #: Added samples past this share of the first tree's, at least
    #: ``_COMPACT_MIN``, are merged into it (no edge sampled again).
    _COMPACT_SHARE = 0.125
    _COMPACT_MIN = 50_000

    def __init__(self, G: nx.MultiGraph, sample, keep=None):
        self._sample, self._keep = sample, keep
        self._data: dict = {}  # edge id -> its data dict, for every edge seen
        self._rows: dict = {}  # edge id -> (start, stop) of its samples, None when not sampled
        self._points = np.empty((0, 3))
        self._owners: list = []
        self._dead = np.zeros(0, dtype=bool)
        self._n = 0
        self._append(self._sampled(G, self._changes(G)[1]))
        self._compact()

    @property
    def points(self) -> np.ndarray:
        """Every sample, live or not, by row (:meth:`query`'s rows)."""
        return self._points[: self._n]

    @property
    def owners(self) -> list:
        return self._owners

    @property
    def empty(self) -> bool:
        return self._base_tree is None and self._added_tree is None

    def _changes(self, G: nx.MultiGraph, edges: list | None = None) -> tuple[set, list]:
        """``(every edge id in G, (id, u, v, data) of each not sampled as it is)``;
        *edges* is :func:`_edge_list` of *G*, when the caller has it."""
        edges = _edge_list(G) if edges is None else edges
        current = {e for e, *_rest in edges}
        return current, [(e, u, v, data) for e, u, v, data in edges if self._data.get(e) is not data]

    def _sampled(self, G: nx.MultiGraph, new: list) -> list:
        out = []
        for e, u, v, data in new:
            pos = {n: G.nodes[n].get("pos") for n in (u, v)}
            path = None
            if all(p is not None for p in pos.values()) and (self._keep is None or self._keep(data)):
                path = self._sample(edge_sample_points(u, v, data, pos))
            out.append((e, data, path))
        return out

    def _append(self, new: list) -> None:
        for e, data, path in new:
            self._data[e], self._rows[e] = data, None
            if path is None or not len(path):
                continue
            stop = self._n + len(path)
            if stop > len(self._points):
                grown = np.empty((max(stop, 2 * len(self._points)), 3))
                grown[: self._n] = self._points[: self._n]
                self._points = grown
                self._dead = np.concatenate([self._dead, np.zeros(len(grown) - len(self._dead), dtype=bool)])
            self._points[self._n:stop] = path
            self._owners.extend([e] * len(path))
            self._rows[e] = (self._n, stop)
            self._n = stop

    def _compact(self) -> None:
        """One tree over every live sample, in row order; none added."""
        alive = ~self._dead[: self._n]
        live = np.flatnonzero(alive)
        moved_to = np.cumsum(alive) - 1  # an edge's samples stay contiguous
        self._rows = {
            e: None if span is None else (int(moved_to[span[0]]), int(moved_to[span[0]]) + span[1] - span[0])
            for e, span in self._rows.items()
        }
        self._points = self._points[live]
        self._owners = [self._owners[i] for i in live]
        self._n = self._base = len(live)
        self._dead = np.zeros(self._n, dtype=bool)
        self._base_tree = cKDTree(self._points) if self._n else None
        self._added_tree, self._added_rows = None, np.empty(0, dtype=np.intp)

    def update(self, G: nx.MultiGraph, edges: list | None = None) -> None:
        """Follow *G*: drop the samples of edges no longer in it (or whose
        data was replaced), sample the edges added."""
        current, new = self._changes(G, edges)
        gone = [e for e in self._data if e not in current]
        gone += [e for e, *_rest in new if e in self._data]  # same id, new data
        if not gone and not new:
            return
        for e in gone:
            span = self._rows.pop(e)
            del self._data[e]
            if span is not None:
                self._dead[span[0]:span[1]] = True
        self._append(self._sampled(G, new))
        added = np.flatnonzero(~self._dead[self._base: self._n]) + self._base
        if len(added) > max(self._COMPACT_MIN, self._COMPACT_SHARE * self._base):
            self._compact()
            return
        self._added_rows = added
        self._added_tree = cKDTree(self._points[added]) if len(added) else None

    def _live_base(self, point: np.ndarray, k: int, bound: float = np.inf):
        """``(distances, rows)`` of up to *k* live samples of the first tree
        nearest *point*, closer than *bound*, nearest first."""
        tree, n = self._base_tree, self._base
        if tree is None:
            return np.empty(0), np.empty(0, dtype=np.intp)
        ask = min(max(k, 8), n)
        while True:
            dist, rows = tree.query(point, k=ask, distance_upper_bound=bound)
            dist, rows = np.atleast_1d(dist), np.atleast_1d(rows)
            found = np.isfinite(dist)
            live = found.copy()
            live[found] = ~self._dead[rows[found]]
            if live.sum() >= k or not found.all() or ask == n:
                return dist[live][:k], rows[live][:k]
            ask = min(8 * ask, n)

    def query(self, points_um):
        """``(distance, row)`` of the live sample nearest each point, as
        ``cKDTree.query`` answers (row ``len(points)`` and ``inf`` with none)."""
        x = np.asarray(points_um, dtype=float)
        single = x.ndim == 1
        x = x.reshape(-1, 3)
        dist, rows = np.full(len(x), np.inf), np.full(len(x), self._n, dtype=np.intp)
        if self._added_tree is not None:
            dist, j = self._added_tree.query(x)
            rows = self._added_rows[j]
        if self._base_tree is not None:
            base_dist, base_rows = self._base_tree.query(x)
            for i in np.flatnonzero(self._dead[base_rows]):
                d, r = self._live_base(x[i], 1, bound=dist[i])
                base_dist[i], base_rows[i] = (d[0], r[0]) if len(d) else (np.inf, self._n)
            take = base_dist <= dist
            dist[take], rows[take] = base_dist[take], base_rows[take]
        return (dist[0], rows[0]) if single else (dist, rows)

    def nearest_rows(self, point_um, k: int) -> np.ndarray:
        """The rows of the *k* live samples nearest *point_um*, nearest first."""
        point = np.asarray(point_um, dtype=float).reshape(3)
        dist, rows = self._live_base(point, k)
        if self._added_tree is not None:
            d, j = self._added_tree.query(point, k=min(k, len(self._added_rows)))
            dist = np.concatenate([dist, np.atleast_1d(d)])
            rows = np.concatenate([rows, self._added_rows[np.atleast_1d(j)]])
        return rows[np.argsort(dist, kind="stable")[:k]]


class _LiveIndex:
    """What :func:`duplicates_existing_vessel` reads of an
    :class:`EdgeSampleIndex` (every edge sampled every micron), kept up to
    date with :meth:`update` rather than built again."""

    added: tuple = ()

    def __init__(self, G: nx.MultiGraph):
        from haemolynx.preprocessing.bridge_mask_support import _densify

        self._samples = _LiveSamples(G, lambda path: _densify(path, 1.0))

    @property
    def tree(self):
        return None if self._samples.empty else self._samples

    @property
    def points(self) -> np.ndarray:
        return self._samples.points

    def update(self, G: nx.MultiGraph, edges: list | None = None) -> None:
        self._samples.update(G, edges)


class _Centrelines:
    """Every edge's centreline sampled every half micron: all of them for what
    the graph reaches, vessels alone (no thick-vessel bridge) to join to. Kept
    up to date with :meth:`update`."""

    def __init__(self, G: nx.MultiGraph):
        sample = lambda path: densify_polyline(path, max_step_um=_STEP_UM)  # noqa: E731
        self._all = _LiveSamples(G, sample)
        self._vessels = _LiveSamples(G, sample, keep=lambda data: not data.get(IS_ZERO_RESISTANCE))

    def update(self, G: nx.MultiGraph, edges: list | None = None) -> None:
        edges = _edge_list(G) if edges is None else edges
        self._all.update(G, edges)
        self._vessels.update(G, edges)

    def distance(self, points_um: np.ndarray) -> np.ndarray:
        if self._all.empty:
            return np.full(len(points_um), np.inf)
        return self._all.query(points_um)[0]

    def nearest_vessel(self, point_um: np.ndarray, skip=None):
        """``(point, (u, v, key))`` of the vessel centreline sample nearest
        *point_um* -- the nearest whose owner *skip* (``owner -> bool``) does
        not refuse, among the closest :data:`_NEAREST_CANDIDATES`."""
        vessels = self._vessels
        if vessels.empty:
            return None, None
        if skip is None:
            _, i = vessels.query(point_um)
            return vessels.points[i], vessels.owners[i]
        for i in vessels.nearest_rows(point_um, _NEAREST_CANDIDATES):
            if not skip(vessels.owners[i]):
                return vessels.points[i], vessels.owners[i]
        return None, None


def _length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1).sum())


def _components(voxels: np.ndarray) -> list[np.ndarray]:
    """26-connected groups of *voxels* (indices into it)."""
    parent = np.arange(len(voxels))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(voxels).query_pairs(_NEIGHBOUR, output_type="ndarray"):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([root(i) for i in range(len(voxels))])
    order = np.argsort(roots, kind="stable")
    bounds = np.flatnonzero(np.diff(roots[order])) + 1
    return np.split(order, bounds)


def lost_skeleton_pieces(
    G: nx.MultiGraph,
    skeleton: np.ndarray,
    voxel_size_zyx=(1.0, 1.0, 1.0),
    *,
    reach_um: float = REACH_UM,
    centrelines: _Centrelines | None = None,
) -> list[LostSkeletonPiece]:
    """The runs of *skeleton* (voxel-indexed) no centreline of *G* comes within
    *reach_um* of, largest first, each with where it meets skeleton the graph
    does reach."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    voxels = np.argwhere(np.asarray(skeleton, dtype=bool))
    if len(voxels) == 0:
        return []
    centrelines = _Centrelines(G) if centrelines is None else centrelines
    lost = centrelines.distance(voxels * spacing) > float(reach_um)
    lost_voxels, reached_voxels = voxels[lost], voxels[~lost]
    if len(lost_voxels) == 0:
        return []
    reached_tree = cKDTree(reached_voxels) if len(reached_voxels) else None
    pieces = []
    for group in _components(lost_voxels):
        piece = lost_voxels[group]
        contacts = []
        if reached_tree is not None:
            for i, near in enumerate(reached_tree.query_ball_point(piece, _NEIGHBOUR)):
                contacts.extend((tuple(piece[i]), tuple(reached_voxels[j])) for j in near)
        pieces.append(LostSkeletonPiece(voxels=piece, contacts=tuple(contacts)))
    pieces.sort(key=lambda p: -len(p.voxels))
    return pieces


def piece_length_um(piece: LostSkeletonPiece, voxel_size_zyx) -> float:
    """How much skeleton *piece* holds, in microns: each voxel counted as the
    step to its nearest neighbour in the piece."""
    if len(piece.voxels) < 2:
        return 0.0
    points = piece.voxels * np.asarray(voxel_size_zyx, dtype=float)
    return float(cKDTree(points).query(points, k=2)[0][:, 1].sum())


_OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def _sweep(sources, members: set):
    """Breadth-first through *members* (voxel tuples, 26-connected) from every
    voxel of *sources* at once: ``(previous, steps)`` per voxel reached."""
    previous = {source: None for source in sources}
    steps = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        x = queue.popleft()
        for o in _OFFSETS:
            y = (x[0] + o[0], x[1] + o[1], x[2] + o[2])
            if y in members and y not in previous:
                previous[y], steps[y] = x, steps[x] + 1
                queue.append(y)
    return previous, steps


def _back(previous: dict, end) -> list:
    path = []
    while end is not None:
        path.append(end)
        end = previous[end]
    return path


def _places(piece: LostSkeletonPiece) -> tuple[dict, dict]:
    """``(reached_of, place)``: for each piece voxel that meets the reached
    skeleton, a reached voxel it meets; and for each reached voxel the piece
    meets, which place it is -- one 26-connected run of them."""
    reached_of = {}
    for mine, theirs in piece.contacts:
        reached_of.setdefault(mine, theirs)
    touched = np.array(sorted({theirs for _, theirs in piece.contacts})).reshape(-1, 3)
    place = {}
    if len(touched):
        for label, group in enumerate(_components(touched)):
            for i in group:
                place[tuple(touched[i])] = label
    return reached_of, place


def _contact_path(piece: LostSkeletonPiece) -> Optional[np.ndarray]:
    """The piece's voxels from one contact to the contact farthest from it
    through the piece that meets the reached skeleton somewhere else (a
    double sweep), with the reached voxel at each end; None when it meets
    reached skeleton in one place only (one 26-connected run of it)."""
    reached_of, place = _places(piece)
    if len(set(place.values())) < 2:
        return None
    members = {tuple(v) for v in piece.voxels}
    _, steps = _sweep([next(iter(reached_of))], members)
    far = max((c for c in reached_of if c in steps), key=steps.get)
    previous, steps = _sweep([far], members)
    elsewhere = [c for c in reached_of if c in steps and place[reached_of[c]] != place[reached_of[far]]]
    if not elsewhere:
        return None
    other = max(elsewhere, key=steps.get)
    path = _back(previous, other)[::-1]  # far -> other
    return np.array([reached_of[far]] + path + [reached_of[other]], dtype=float)


def _other_places(piece: LostSkeletonPiece, route: np.ndarray, spacing: np.ndarray) -> list[np.ndarray]:
    """For each place the piece meets the graph that its *route* (voxel
    indices, as :func:`_contact_path` drew it) does not reach within
    :data:`REACH_UM`, the piece's own skeleton from the nearest route voxel to
    it, with the reached voxel at its end -- shortest first."""
    reached_of, place = _places(piece)
    route_points = route * spacing
    route_tree = cKDTree(route_points)
    by_place: dict = {}
    for mine, theirs in reached_of.items():
        by_place.setdefault(place[theirs], []).append(mine)
    members = {tuple(v) for v in piece.voxels}
    sources = [tuple(int(c) for c in v) for v in route[1:-1]]
    previous, steps = _sweep([v for v in sources if v in members], members)
    branches = []
    for label, mine in by_place.items():
        theirs = [reached_of[m] for m in mine]
        if route_tree.query(np.asarray(theirs, dtype=float) * spacing)[0].min() <= REACH_UM:
            continue
        reachable = [m for m in mine if m in steps]
        if not reachable:
            continue
        end = min(reachable, key=steps.get)
        path = _back(previous, end)[::-1]  # route -> place
        branches.append(np.array(path + [reached_of[end]], dtype=float))
    branches.sort(key=len)
    return branches


def _split_at(G: nx.MultiGraph, point_um: np.ndarray, owner, made: list) -> Any:
    """Join *point_um* to the network: the node it lies at, splitting the
    vessel *owner* there -- or, where a split this piece made already took
    that vessel, the piece of it nearest."""
    u, v, key = owner
    if not G.has_edge(u, v, key):
        pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True)}
        options = [(a, b, k) for n in made for a, b, k in G.edges(n, keys=True)]
        u, v, key = min(
            options,
            key=lambda e: float(np.min(np.linalg.norm(
                edge_sample_points(e[0], e[1], G.edges[e], pos) - point_um, axis=1))),
        )
    node = insert_node_on_edge(G, u, v, key, point_um)
    made.append(node)
    return node


def _loop_back(G: nx.MultiGraph, a: Any, b: Any) -> Optional[np.ndarray]:
    """The graph's shortest way from *b* back to *a*, as a polyline."""
    try:
        nodes = nx.shortest_path(G, b, a, weight="length")
    except nx.NetworkXNoPath:
        return None
    if len(nodes) < 2:
        return None
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in nodes}
    parts = []
    for x, y in zip(nodes[:-1], nodes[1:]):
        key = min(G[x][y], key=lambda k: float(G[x][y][k].get("length") or 0.0))
        parts.append(edge_sample_points(x, y, G[x][y][key], pos)[1:])
    return np.vstack(parts)


def _judge(
    G: nx.MultiGraph,
    piece: LostSkeletonPiece,
    support: MaskSupport,
    centrelines: _Centrelines,
    index: EdgeSampleIndex,
    spacing: np.ndarray,
    keep_hole_over_um: float,
):
    """``(Retrace, attachments)``: the verdict on *piece*, and where it would
    join the graph -- ``(point, owner)`` at each end -- when it is drawn back."""
    points = piece.voxels * spacing
    nothing = dict(path_um=None, hole_um=None, loop_length_um=None)
    mask_fraction = float(np.mean(support.inside(points)))
    far = centrelines.distance(points) > np.maximum(2.0 * support.radius(points) + FAR_MARGIN_UM, FAR_MIN_UM)
    far_fraction = float(np.mean(far))
    verdict = dict(piece=piece, far_fraction=far_fraction, mask_fraction=mask_fraction)
    if piece_length_um(piece, spacing) < MIN_PIECE_UM:
        return Retrace(**verdict, **nothing, reason=REFUSED_SMALL), None
    if mask_fraction < MIN_MASK_FRACTION:
        return Retrace(**verdict, **nothing, reason=REFUSED_OFF_MASK), None
    if far_fraction < MIN_FAR_FRACTION:
        return Retrace(**verdict, **nothing, reason=REFUSED_BESIDE), None
    route = _contact_path(piece)
    if route is None:
        return Retrace(**verdict, **nothing, reason=REFUSED_ONE_END), None
    route_um = route * spacing
    start, start_owner = centrelines.nearest_vessel(route_um[0])
    end, end_owner = centrelines.nearest_vessel(route_um[-1])
    if start is None or end is None:
        return Retrace(**verdict, **nothing, reason=REFUSED_ONE_END), None
    # Tried on a copy: the loop it would close is read with both ends joined.
    trial = _trial_copy(G)
    made: list = []
    a = _split_at(trial, start, start_owner, made)
    b = _split_at(trial, end, end_owner, made)
    path = np.vstack([trial.nodes[a]["pos"], route_um, trial.nodes[b]["pos"]]).astype(float)
    nothing["path_um"] = path
    # The bridge gate judges the two joins from the graph to the skeleton, which
    # this step draws. The route between is the run's own skeleton, which graph
    # building never holds to it: E14.5's skeleton, thinned from the hole-filled
    # mask, crosses dropouts of 3-10 um in the raw mask along vessels grid N
    # marked as vessels.
    joins = (path[:2], path[-2:])
    if not all(support.accepts(join) for join in joins):
        return Retrace(**verdict, **nothing, reason=REFUSED_BACKGROUND), None
    if support.support(path).inside_fraction < MIN_MASK_FRACTION:
        return Retrace(**verdict, **nothing, reason=REFUSED_OFF_MASK), None
    if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
        return Retrace(**verdict, **nothing, reason=REFUSED_SAME_LUMEN), None
    back = None if a == b else _loop_back(trial, a, b)
    if back is None:
        return Retrace(**verdict, **nothing, reason=REFUSED_NO_LOOP), None
    loop = np.vstack([path, back])
    hole = loop_hole_um(loop, support)
    measured = dict(path_um=path, hole_um=hole, loop_length_um=_length(loop))
    if hole is None or hole <= float(keep_hole_over_um):
        return Retrace(**verdict, **measured, reason=REFUSED_SMALL_HOLE), None
    return Retrace(**verdict, **measured, reason=None), ((start, start_owner), (end, end_owner), route)


def _nearest_on(G: nx.MultiGraph, point_um: np.ndarray, keep) -> tuple:
    """``(point, (u, v, key))``: the centreline sample nearest *point_um* on
    the edges *keep* (``data -> bool``) accepts; ``(None, None)`` with none."""
    pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True)}
    best, best_point, best_owner = np.inf, None, None
    for u, v, key, data in G.edges(keys=True, data=True):
        if not keep(data):
            continue
        path = densify_polyline(edge_sample_points(u, v, data, pos), max_step_um=_STEP_UM)
        d = np.linalg.norm(path - point_um, axis=1)
        i = int(np.argmin(d))
        if d[i] < best:
            best, best_point, best_owner = float(d[i]), path[i], (u, v, key)
    return best_point, best_owner


def _join_place(
    G: nx.MultiGraph,
    branch: np.ndarray,
    tag: int,
    support: MaskSupport,
    spacing: np.ndarray,
    keep_hole_over_um: float,
    centrelines: _Centrelines,
    index: EdgeSampleIndex,
) -> Optional[str]:
    """Draw, in place, one further place a drawn piece meets the graph: the
    piece's skeleton *branch* (voxel indices, from its route to the reached
    voxel there) from the route edge (the edges carrying ``_PIECE == tag``) to
    the vessel nearest that voxel, judged as the route was -- its join to the
    graph through the gate, at least half in the mask, not beside a vessel in the
    same lumen (judged at its middle when short, as a reconnect's bridge), and
    only when the loop it closes runs round a hole wider than
    *keep_hole_over_um*. Grid N2's drawings (N2-12, N2-14): every join
    Harvey added ran along the piece to such a place. Returns None when it is
    drawn, else why not (a ``REFUSED_*`` reason)."""
    branch_um = branch * spacing
    start, start_owner = _nearest_on(G, branch_um[0], lambda d: d.get(_PIECE) == tag)
    end, end_owner = centrelines.nearest_vessel(
        branch_um[-1], skip=lambda e: not G.has_edge(*e) or G.edges[e].get(_PIECE) == tag)
    if start is None or end is None:
        return REFUSED_ONE_END
    trial = _trial_copy(G)
    made: list = []
    a = _split_at(trial, start, start_owner, made)
    b = _split_at(trial, end, end_owner, made)
    if a == b:
        return REFUSED_NO_LOOP
    path = np.vstack([trial.nodes[a]["pos"], branch_um[1:], trial.nodes[b]["pos"]]).astype(float)
    if not support.accepts(path[-2:]):
        return REFUSED_BACKGROUND
    if support.support(path).inside_fraction < MIN_MASK_FRACTION:
        return REFUSED_OFF_MASK
    if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index, judge_short_at_middle=True):
        return REFUSED_SAME_LUMEN
    back = _loop_back(trial, a, b)
    if back is None:
        return REFUSED_NO_LOOP
    hole = loop_hole_um(np.vstack([path, back]), support)
    if hole is None or hole <= float(keep_hole_over_um):
        return REFUSED_SMALL_HOLE
    made = []
    a = _split_at(G, start, start_owner, made)
    b = _split_at(G, end, end_owner, made)
    path[0], path[-1] = G.nodes[a]["pos"], G.nodes[b]["pos"]
    G.add_edge(
        a, b, voxels=path.tolist(), length=_length(path),
        **{RETRACED: True, RETRACED_JOIN: True, RETRACED_HOLE_UM: round(float(hole), 2), _PIECE: tag},
    )
    return None


def judge_lost_skeleton(
    G: nx.MultiGraph,
    skeleton: np.ndarray,
    support: MaskSupport,
    *,
    keep_hole_over_um: float = DEFAULT_SHORT_LOOP_KEEP_HOLE_UM,
    reach_um: float = REACH_UM,
) -> list[Retrace]:
    """What :func:`retrace_lost_skeleton` would say of each lost piece of
    *skeleton*, judged against *G* as it stands -- read-only, largest piece
    first. Pieces drawn back one by one change what the next is judged
    against, so the run itself may draw a different set."""
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    centrelines, index = _Centrelines(G), _LiveIndex(G)
    return [
        _judge(G, piece, support, centrelines, index, spacing, keep_hole_over_um)[0]
        for piece in lost_skeleton_pieces(G, skeleton, spacing, reach_um=reach_um, centrelines=centrelines)
    ]


def _follow(G: nx.MultiGraph, centrelines: _Centrelines, index: _LiveIndex) -> None:
    edges = _edge_list(G)
    centrelines.update(G, edges)
    index.update(G, edges)


def retrace_lost_skeleton(
    G: nx.MultiGraph,
    skeleton: np.ndarray,
    support: MaskSupport,
    *,
    keep_hole_over_um: float = DEFAULT_SHORT_LOOP_KEEP_HOLE_UM,
    reach_um: float = REACH_UM,
) -> nx.MultiGraph:
    """Draw back, in place, each lost piece of *skeleton* the graph should
    follow: the module docstring's pool, joined at its two farthest contacts,
    where the loop it closes with the graph runs round a hole wider than
    *keep_hole_over_um* (the short-loop rule's ``short_loop_keep_hole_over_um``).
    Each is added as one vessel along its contact path, from the vessel point
    nearest one end to the one nearest the other (split there): the two joins
    to the graph never across more background than a bridge may cross
    (*support*'s gate), the whole at least ``MIN_MASK_FRACTION`` in the mask,
    and never beside a vessel in the same lumen. Then each further place the
    piece meets the graph, which its route does not reach, is joined along the
    piece's own skeleton under the same tests, the hole judged on the loop
    that join closes (:func:`_join_place`) -- so a tangle's mesh, round small
    holes, is not drawn. Largest piece first, each judged against the graph as
    the ones before it left it. Every vessel drawn carries ``retraced=True``
    and ``retraced_hole_um`` (a further join ``retraced_join`` too); its path
    is the skeleton's staircase, for the caller to smooth.
    """
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    centrelines, index = _Centrelines(G), _LiveIndex(G)
    pieces = lost_skeleton_pieces(G, skeleton, spacing, reach_um=reach_um, centrelines=centrelines)
    drawn = joins = 0
    joins_refused: dict[str, int] = {}
    refused: dict[str, int] = {}
    for piece in pieces:
        verdict, attachments = _judge(G, piece, support, centrelines, index, spacing, keep_hole_over_um)
        if attachments is None:
            refused[verdict.reason] = refused.get(verdict.reason, 0) + 1
            continue
        made: list = []
        (start, start_owner), (end, end_owner), route = attachments
        a = _split_at(G, start, start_owner, made)
        b = _split_at(G, end, end_owner, made)
        path = verdict.path_um.copy()
        path[0], path[-1] = G.nodes[a]["pos"], G.nodes[b]["pos"]
        G.add_edge(
            a, b,
            voxels=path.tolist(), length=_length(path),
            **{RETRACED: True, RETRACED_HOLE_UM: round(float(verdict.hole_um), 2), _PIECE: drawn},
        )
        # the route, and the vessels its ends split
        _follow(G, centrelines, index)
        for branch in _other_places(piece, route, spacing):
            why = _join_place(G, branch, drawn, support, spacing, keep_hole_over_um, centrelines, index)
            if why is None:
                joins += 1
                # its split took a vessel the indexes still name
                _follow(G, centrelines, index)
            else:
                joins_refused[why] = joins_refused.get(why, 0) + 1
        drawn += 1
    for *_e, data in G.edges(data=True):
        data.pop(_PIECE, None)
    if pieces:
        logger.info(
            "Skeleton re-trace: drew back %d of %d lost skeleton pieces round a hole over %.3g um "
            "(refused %s), and %d further joins of theirs (refused %s)",
            drawn, len(pieces), float(keep_hole_over_um), _reasons(refused), joins, _reasons(joins_refused),
        )
    return G


def _reasons(counts: dict) -> str:
    return ", ".join(f"{n} {why}" for why, n in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"


def retraced_edges(G: nx.MultiGraph) -> list:
    """``(u, v, key)`` of every vessel :func:`retrace_lost_skeleton` drew."""
    return [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get(RETRACED)]


__all__ = [
    "FAR_MARGIN_UM",
    "FAR_MIN_UM",
    "LostSkeletonPiece",
    "MIN_FAR_FRACTION",
    "MIN_MASK_FRACTION",
    "MIN_PIECE_UM",
    "REACH_UM",
    "REFUSED_BACKGROUND",
    "REFUSED_BESIDE",
    "REFUSED_NO_LOOP",
    "REFUSED_OFF_MASK",
    "REFUSED_ONE_END",
    "REFUSED_SAME_LUMEN",
    "REFUSED_SMALL",
    "REFUSED_SMALL_HOLE",
    "RETRACED",
    "RETRACED_HOLE_UM",
    "RETRACED_JOIN",
    "Retrace",
    "judge_lost_skeleton",
    "lost_skeleton_pieces",
    "piece_length_um",
    "retrace_lost_skeleton",
    "retraced_edges",
]
