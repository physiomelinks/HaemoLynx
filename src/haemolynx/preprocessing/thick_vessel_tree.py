"""The thick-vessel centreline tree on a large stack: the low-RAM option's
``skeleton_thick_vessel_large_stack_tree``.

:func:`~.thick_vessels._component_tree` draws a fat component's tree from
scipy's Dijkstra over a 26-neighbour graph of every voxel in it, built into
CSR: about 360 bytes a voxel. Where the fat region is one network crossing the
stack (Harvey's patella mask at a 3 um gate: 111 M voxels, 2026-10-18) that
graph alone is 40 GB. Its arm loop also rescanned the whole component for
every arm, and stopped at 500 arms.

Here each component's tree is built the same way -- the same root and far
end, arms taken farthest first, judged by the same tests -- from:

* shortest paths from the root found over the voxels themselves
  (:func:`shortest_path_parents`): a bucketed label-correcting search that
  looks each voxel's neighbours up in the component's sorted keys, keeping a
  distance and a parent a voxel (12 bytes) beside a fixed ~120 MB of batch
  arrays. Its distances are scipy's, bit for bit; where two routes cost
  exactly the same it may take the other one, so the tree can step a voxel
  aside, and an arm judged on that route can come out a little different. A
  component of at most :data:`EXACT_TREE_MAX_VOXELS` voxels still gets
  scipy's walk, and so the in-RAM tree exactly (bar the arm cap);
* the farthest uncovered high-radius voxel found from cells of candidates
  whose distances to the tree are measured again only when a cell could hold
  the farthest (:class:`_FarthestCandidates`), not every candidate per arm;
* no cap on the number of arms.

The in-RAM option and the low-RAM option without this setting keep
:func:`~.thick_vessels._component_tree` as it is.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
from scipy.spatial import cKDTree

from . import blockwise_edt, memmap_support, shared_volume_tasks, slab_labels
from .thick_vessels import (
    _OFFSETS_26,
    _STRUCTURE_FULL,
    _VoxelBoxes,
    _box_members,
    _build_dijkstra_graph_from_coords,
    _coords_of,
    _dijkstra_parents,
    _dominant_eigenvector_3x3,
    _expand_slices,
    _path_cuts_across_lumen,
    _path_length,
    _principal_endpoints,
    _project_onto_axis,
    _step_factor,
)

logger = logging.getLogger(__name__)

#: A component of at most this many voxels gets scipy's Dijkstra on the CSR
#: graph (about 360 bytes a voxel: 1.4 GB here), and so exactly the in-RAM
#: tree; a larger one, :func:`shortest_path_parents`.
EXACT_TREE_MAX_VOXELS = 4_000_000

#: Points read at once: the component's coordinates, the radius at them and
#: the shortest-path search's batch are held this many at a time.
_CHUNK = 1 << 21
_SEARCH_BATCH = 1 << 16
#: The bucket width of :func:`shortest_path_parents`, in its cost units (a
#: step into a voxel of radius r costs 1/r^2). Narrow buckets visit almost
#: every voxel once; the number of rounds is set by how many steps the
#: longest path takes, not by this.
_BUCKET_WIDTH = 0.02
#: Buckets are taken together until the frontier holds this many voxels.
_MIN_FRONTIER = 4096
#: Components under this many voxels are drawn several at once, on threads
#: (numpy's and scipy's work lets go of the interpreter): each holds at most
#: the exact walk's graph of one, about 360 bytes a voxel.
_THREADED_COMPONENT_VOXELS = 500_000
#: Threads for those, at most.
_COMPONENT_THREADS = 4
#: The radius step's block rows run on at most this many threads, each
#: holding one block's transform (``blockwise_edt.BLOCK_BYTES_PER_VOXEL`` a
#: voxel of it), and only as many as half the free memory holds.
_RADIUS_MAX_THREADS = 4
#: Edge of the cells (voxels) the farthest-candidate search groups by.
_CELL = 16
#: Path voxels covered at once.
_COVER_CHUNK = 32


class ComponentVoxels:
    """A component's voxels: their sorted C-order *keys* in its box of
    *shape*, each voxel's number its index into them."""

    def __init__(self, keys: np.ndarray, shape):
        self.keys = np.asarray(keys, dtype=np.int64)
        self.shape = tuple(int(s) for s in shape)
        self._plane = self.shape[1] * self.shape[2]
        self._row = self.shape[2]

    def __len__(self) -> int:
        return len(self.keys)

    def coords(self, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.keys[ids]
        z = k // self._plane
        r = k - z * self._plane
        y = r // self._row
        return z, y, r - y * self._row

    def coord_array(self, ids) -> np.ndarray:
        return np.column_stack(self.coords(ids))

    def ids_at(self, z, y, x) -> np.ndarray:
        """The voxel number at each ``(z, y, x)``, -1 where none is."""
        z, y, x = (np.asarray(v, dtype=np.int64) for v in (z, y, x))
        Z, Y, X = self.shape
        inside = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
        q = (z * Y + y) * X + x
        if not len(self.keys):
            return np.full(q.shape, -1, dtype=np.int64)
        at = np.minimum(np.searchsorted(self.keys, q), len(self.keys) - 1)
        return np.where(inside & (self.keys[at] == q), at, -1)

    def neighbours(self, ids: np.ndarray) -> np.ndarray:
        """``(len(ids), 26)`` voxel numbers of each voxel's neighbours in
        ``_OFFSETS_26`` order, -1 where there is none.

        A row's three neighbours along x are looked up together: one search
        for the first, the next two keys after it. The voxel's own row needs
        no search -- its neighbours are the keys either side of it."""
        ids = np.asarray(ids, dtype=np.int64)
        n = len(self.keys)
        k = self.keys[ids]
        z = k // self._plane
        r = k - z * self._plane
        y = r // self._row
        x = r - y * self._row
        Z, Y, X = self.shape
        out = np.full((len(ids), 26), -1, dtype=np.int64)
        left_ok = x >= 1
        right_ok = x <= X - 2
        last = n - 1
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                # Columns of dx = -1, 0, 1 in _OFFSETS_26's C order, which
                # skips the centre (13 of the full 27).
                west, middle, east = (
                    c - (c > 13) for c in range((dz + 1) * 9 + (dy + 1) * 3, (dz + 1) * 9 + (dy + 1) * 3 + 3)
                )
                if dz == 0 and dy == 0:
                    before = np.maximum(ids - 1, 0)
                    after = np.minimum(ids + 1, last)
                    hit = left_ok & (ids >= 1) & (self.keys[before] == k - 1)
                    out[hit, west] = before[hit]
                    hit = right_ok & (ids < last) & (self.keys[after] == k + 1)
                    out[hit, east] = after[hit]
                    continue
                valid = (z + dz >= 0) & (z + dz < Z) & (y + dy >= 0) & (y + dy < Y)
                t = k + dz * self._plane + dy * self._row
                at = np.searchsorted(self.keys, t - 1)
                # Keys are unique and sorted, so x-1, x and x+1 of the row
                # are the next keys from *at*, each there or not; a wrapped
                # x-1 (x = 0) still moves the pointer past itself.
                found = self.keys[np.minimum(at, last)] == t - 1
                hit = valid & left_ok & found
                out[hit, west] = at[hit]
                at = at + found
                found = self.keys[np.minimum(at, last)] == t
                hit = valid & found
                out[hit, middle] = at[hit]
                at = at + found
                hit = valid & right_ok & (self.keys[np.minimum(at, last)] == t + 1)
                out[hit, east] = at[hit]
        return out

    def touches(self, ids: np.ndarray, member: np.ndarray) -> np.ndarray:
        """Whether each voxel, or one of its 26 neighbours, is *member*
        (a boolean per voxel number): ``_touches_tree`` of each."""
        ids = np.asarray(ids, dtype=np.int64)
        nb = self.neighbours(ids)
        near = (nb >= 0) & member[np.maximum(nb, 0)]
        return member[ids] | near.any(axis=1)


def _cost(edt_at: np.ndarray) -> np.ndarray:
    """The cost of stepping into a voxel of radius *edt_at*, as
    ``_component_tree`` costs it."""
    return 1.0 / (np.square(edt_at) + 1e-6)


def _relax(voxels, f, dist, edt_at, factors):
    """``(to, value, came)``: every step out of the voxels *f* that would
    bring a neighbour nearer than it is now."""
    nb = voxels.neighbours(f)
    rows, cols = np.nonzero(nb >= 0)
    to = nb[rows, cols]
    del nb
    value = dist[f][rows] + _cost(edt_at[to]) * factors[cols]
    better = value < dist[to]
    return to[better], value[better], f[rows[better]]


def shortest_path_parents(
    voxels: ComponentVoxels,
    edt_at: np.ndarray,
    root: int,
    relative_spacing: np.ndarray,
    *,
    bucket_width: float = _BUCKET_WIDTH,
    batch: int | None = None,
    return_distances: bool = False,
):
    """Each voxel's predecessor on a shortest path from *root* (itself for
    the root, -1 where unreachable), stepping into a voxel costing
    ``1 / (edt^2 + 1e-6)`` times :func:`~.thick_vessels._step_factor` --
    ``_component_tree``'s graph, never built.

    Delta-stepping: voxels are relaxed a bucket of distances at a time, and
    one improved again within its bucket is relaxed again, so the distances
    are the shortest ones, sums of the same terms in the same order as
    scipy's. Of two routes that cost exactly the same it keeps the one into
    the lower voxel number, where scipy keeps the one its heap reached first.
    *return_distances* returns ``(parent, distance)``.
    """
    if batch is None:
        batch = _SEARCH_BATCH
    n = len(voxels)
    if n >= np.iinfo(np.int32).max:
        raise ValueError(f"a fat component of {n} voxels is too large to number")
    factors = np.array([_step_factor(o, relative_spacing) for o in _OFFSETS_26])
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int32)
    dist[root] = 0.0
    parent[root] = root
    buckets: dict[int, list[np.ndarray]] = {0: [np.array([root], dtype=np.int64)]}
    width = float(bucket_width)
    _search(voxels, edt_at, dist, parent, buckets, width, factors, batch)
    return (parent, dist) if return_distances else parent


def _search(voxels, edt_at, dist, parent, buckets, width, factors, batch):
    """:func:`shortest_path_parents`'s loop over the buckets."""
    n = len(dist)
    started = last_log = time.perf_counter()
    relaxed = 0
    while buckets:
        if time.perf_counter() - last_log > 60.0:
            last_log = time.perf_counter()
            logger.info(
                "Large-stack centreline tree: shortest paths reached %d of %d voxels "
                "(%d relaxed) in %.0fs",
                int(np.count_nonzero(parent >= 0)),
                n,
                relaxed,
                last_log - started,
            )
        # The lowest bucket, and the ones after it while the frontier is
        # small: a round costs as much in calls as in voxels, and a thin
        # vessel's cross-section spreads over many narrow buckets.
        b = lo = min(buckets)
        parts = buckets.pop(b)
        size = sum(len(p) for p in parts)
        while buckets and size < _MIN_FRONTIER:
            b = min(buckets)
            more = buckets.pop(b)
            parts.extend(more)
            size += sum(len(p) for p in more)
        frontier = np.unique(np.concatenate(parts))
        del parts
        # An entry left in a bucket after its voxel was improved into a lower
        # one, which has been relaxed already.
        frontier = frontier[np.floor(dist[frontier] / width) >= lo]
        while frontier.size:
            improved = []
            for start in range(0, frontier.size, batch):
                f = frontier[start:start + batch]
                relaxed += len(f)
                to, value, came = _relax(voxels, f, dist, edt_at, factors)
                if not to.size:
                    continue
                order = np.lexsort((came, value, to))
                to, value, came = to[order], value[order], came[order]
                first = np.r_[True, to[1:] != to[:-1]]
                to, value, came = to[first], value[first], came[first]
                dist[to] = value
                parent[to] = came
                improved.append(to)
            if not improved:
                break
            improved = np.unique(np.concatenate(improved))
            at = np.floor(dist[improved] / width).astype(np.int64)
            here = at <= b
            later, at = improved[~here], at[~here]
            if later.size:
                order = np.argsort(at, kind="stable")
                later, at = later[order], at[order]
                starts = np.flatnonzero(np.r_[True, at[1:] != at[:-1]])
                for s0, s1 in zip(starts, np.r_[starts[1:], later.size]):
                    buckets.setdefault(int(at[s0]), []).append(later[s0:s1])
            frontier = improved[here]


def _principal_endpoints_by_chunk(voxels: ComponentVoxels, relative: np.ndarray):
    """:func:`~.thick_vessels._principal_endpoints` of the component, its
    sums taken a chunk of voxels at a time (so not bit for bit its rounding)."""
    n = len(voxels)
    if n < 2:
        return 0, 0
    total = np.zeros(3)
    for s in range(0, n, _CHUNK):
        total += (voxels.coord_array(np.arange(s, min(s + _CHUNK, n))) * relative).sum(axis=0)
    mean = total / n
    cov = np.zeros((3, 3))
    for s in range(0, n, _CHUNK):
        centred = voxels.coord_array(np.arange(s, min(s + _CHUNK, n))) * relative - mean
        for i in range(3):
            for j in range(i, 3):
                cov[i, j] += float(np.sum(centred[:, i] * centred[:, j]))
    cov /= max(n - 1, 1)
    cov = np.triu(cov) + np.triu(cov, 1).T
    axis = _dominant_eigenvector_3x3(cov)
    lo = hi = None
    for s in range(0, n, _CHUNK):
        ids = np.arange(s, min(s + _CHUNK, n))
        projection = _project_onto_axis(voxels.coord_array(ids) * relative - mean, axis)
        i, j = int(np.argmin(projection)), int(np.argmax(projection))
        if lo is None or projection[i] < lo[0]:
            lo = (projection[i], s + i)
        if hi is None or projection[j] > hi[0]:
            hi = (projection[j], s + j)
    return lo[1], hi[1]


class _GrowingTree:
    """Nearest-point distances to a growing set of points: k-d trees of
    sizes doubling, merged as points are added (Bentley-Saxe)."""

    def __init__(self, points: np.ndarray):
        self._parts: list[tuple[np.ndarray, cKDTree]] = []
        self.add(points)

    def add(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        while self._parts and len(self._parts[-1][0]) <= len(points):
            points = np.concatenate([self._parts.pop()[0], points])
        self._parts.append((points, cKDTree(points)))

    def distance(self, query: np.ndarray) -> np.ndarray:
        best = None
        for _points, tree in self._parts:
            d, _ = tree.query(query, k=1)
            best = d if best is None else np.minimum(best, d)
        return best


class _FarthestCandidates:
    """The uncovered candidate farthest from the tree, the lowest-numbered of
    those as far: ``argmax`` of every uncovered candidate's distance, as
    ``_component_tree`` takes it.

    Candidates are grouped in cells of :data:`_CELL` voxels a side, each with
    the largest distance of its uncovered candidates when last measured.
    Distances only fall as the tree grows, so that bound only falls, and the
    farthest is in a cell whose bound reaches it. A cell's candidates are
    measured again only when a piece of tree added since lies within its
    bound of the cell -- else none of them can have come nearer."""

    def __init__(self, voxels, candidates, covered, relative, tree: _GrowingTree):
        self._voxels = voxels
        self._candidates = candidates
        self._covered = covered
        self._relative = np.asarray(relative, dtype=float)
        self._tree = tree
        grid = np.asarray(voxels.shape, dtype=np.int64) // _CELL + 1
        cell = np.empty(len(candidates), dtype=np.int64)
        for s in range(0, len(candidates), _CHUNK):
            z, y, x = voxels.coords(candidates[s:s + _CHUNK])
            cell[s:s + _CHUNK] = ((z // _CELL) * grid[1] + (y // _CELL)) * grid[2] + (x // _CELL)
        order = np.argsort(cell, kind="stable").astype(np.int32)
        cell = cell[order]
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        index = np.column_stack(np.unravel_index(cell[starts], tuple(int(g) for g in grid)))
        del cell
        self._cell_lo = index * _CELL * self._relative
        self._cell_hi = (index * _CELL + _CELL - 1) * self._relative
        self._members = order
        self._bounds = np.r_[starts, len(order)]
        # Every candidate measured once, a chunk at a time.
        self._distance = np.empty(len(candidates))
        for s in range(0, len(candidates), _CHUNK):
            points = voxels.coord_array(candidates[s:s + _CHUNK]).astype(np.float64)
            self._distance[s:s + _CHUNK] = tree.distance(points * self._relative)
        live = np.where(covered[self._members], -np.inf, self._distance[self._members])
        self._cell_bound = np.maximum.reduceat(live, starts) if len(starts) else np.empty(0)
        del live
        # The boxes of the pieces of tree added since the start, and from
        # which piece on each growth's pieces come.
        self._version = 0
        self._measured = np.zeros(len(starts), dtype=np.int64)
        self._piece_lo = np.empty((0, 3))
        self._piece_hi = np.empty((0, 3))
        self._first_piece = [0]
        self._stamp = np.where(np.isfinite(self._cell_bound), 0, -1).astype(np.int64)
        self._heap = [
            (-float(b), int(c), 0) for c, b in enumerate(self._cell_bound) if np.isfinite(b)
        ]
        heapq.heapify(self._heap)

    def tree_grew(self, points: np.ndarray) -> None:
        """The tree gained *points* (scaled as the tree's)."""
        points = np.asarray(points, dtype=float)
        pieces = range(0, len(points), _COVER_CHUNK)
        lo = np.array([points[s:s + _COVER_CHUNK].min(axis=0) for s in pieces])
        hi = np.array([points[s:s + _COVER_CHUNK].max(axis=0) for s in pieces])
        self._piece_lo = np.concatenate([self._piece_lo, lo.reshape(-1, 3)])
        self._piece_hi = np.concatenate([self._piece_hi, hi.reshape(-1, 3)])
        self._version += 1
        self._first_piece.append(len(self._piece_lo))

    def _measure(self, c: int):
        members = self._members[self._bounds[c]:self._bounds[c + 1]]
        live = members[~self._covered[members]]
        if not live.size:
            return None
        if self._measured[c] != self._version:
            since = self._first_piece[int(self._measured[c])]
            gap = np.maximum(
                np.maximum(
                    self._piece_lo[since:] - self._cell_hi[c], self._cell_lo[c] - self._piece_hi[since:]
                ),
                0.0,
            )
            nearest = float(np.sqrt(np.min(np.sum(gap * gap, axis=1)))) if len(gap) else np.inf
            if nearest <= self._cell_bound[c] * (1.0 + 1e-9) + 1e-9:
                points = self._voxels.coord_array(self._candidates[live]).astype(np.float64)
                self._distance[live] = self._tree.distance(points * self._relative)
            self._measured[c] = self._version
        # Members are in candidate order within a cell, so argmax's first is
        # the lowest-numbered of the cell's farthest.
        d = self._distance[live]
        i = int(np.argmax(d))
        self._cell_bound[c] = d[i]
        return float(d[i]), int(live[i])

    def farthest(self):
        """``(distance, candidate index)``, or None when every candidate is
        covered."""
        best = None
        held = []
        while self._heap:
            bound, c, stamp = self._heap[0]
            if stamp != self._stamp[c]:
                heapq.heappop(self._heap)
                continue
            if best is not None and -bound < best[0]:
                break
            heapq.heappop(self._heap)
            found = self._measure(c)
            if found is None:
                self._stamp[c] = -1
                continue
            if best is None or found[0] > best[0] or (found[0] == best[0] and found[1] < best[1]):
                best = found
            held.append((c, found[0]))
        for c, d in held:
            self._stamp[c] += 1
            heapq.heappush(self._heap, (-d, c, int(self._stamp[c])))
        return best


def _cover(voxels, path_ids, radius, relative, candidate_keys, covered) -> None:
    """Mark *covered* (one per candidate) the candidates
    ``_covered_around_path`` covers around the path: within *radius* of one
    of its voxels by ``distance_transform_edt``'s arithmetic. Chunked along
    the path -- a voxel is within the radius of the path exactly when it is
    within it of a piece -- so no box is ever the whole path's."""
    if not len(path_ids):
        return
    radius = max(1, int(radius))
    reach = np.ceil(radius / relative).astype(int)
    path = voxels.coord_array(np.asarray(path_ids, dtype=np.int64))
    shape = voxels.shape
    slack = radius * 1e-9 + 1e-9
    for s in range(0, len(path), _COVER_CHUNK):
        pts = path[s:s + _COVER_CHUNK]
        lo = np.maximum(pts.min(axis=0) - reach, 0)
        hi = np.minimum(pts.max(axis=0) + reach + 1, shape)
        _positions, ids = _box_members(
            candidate_keys, shape, tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        )
        ids = ids[~covered[ids]]
        if not ids.size:
            continue
        query = _coords_of(candidate_keys[ids], shape).astype(np.int64)
        tree = cKDTree(pts.astype(np.float64) * relative)
        distance, nearest = tree.query(
            query.astype(np.float64) * relative, k=1, distance_upper_bound=radius + slack
        )
        found = np.flatnonzero(np.isfinite(distance))
        within = np.zeros(len(ids), dtype=bool)
        if found.size:
            exact = blockwise_edt.displacement_distances(pts[nearest[found]] - query[found], relative)
            within[found] = exact <= radius
            for row in found[~within[found]]:
                near = tree.query_ball_point(query[row].astype(np.float64) * relative, radius + slack)
                if near:
                    within[row] = bool(
                        np.any(
                            blockwise_edt.displacement_distances(pts[near] - query[row], relative)
                            <= radius
                        )
                    )
        covered[ids[within]] = True


def _trace(parent: np.ndarray, start: int, stop) -> list[int]:
    """Voxel numbers from *start* back along *parent* to the first for which
    ``stop(ids)`` (a boolean per voxel of a batch) is True, that one
    included, in that order -- or to the root."""
    path: list[int] = []
    seen: set[int] = set()
    v = int(start)
    while True:
        batch = []
        while len(batch) < 64 and v >= 0 and v not in seen:
            seen.add(v)
            batch.append(v)
            nxt = int(parent[v])
            if nxt == v:
                v = -1
                break
            v = nxt
        if not batch:
            return path
        hits = np.flatnonzero(stop(np.asarray(batch, dtype=np.int64)))
        if hits.size:
            path.extend(batch[: int(hits[0]) + 1])
            return path
        path.extend(batch)
        if v < 0 or v in seen:
            return path


def component_tree(
    keys: np.ndarray,
    edt_at: np.ndarray,
    shape: tuple[int, int, int],
    relative_spacing: np.ndarray,
    *,
    exact_max_voxels: int | None = None,
) -> np.ndarray:
    """``_component_tree`` of the component whose sorted C-order *keys* in
    its box of *shape* have radius *edt_at*: the tree's keys, sorted. Up to
    *exact_max_voxels* voxels (:data:`EXACT_TREE_MAX_VOXELS` when not given)
    the shortest paths are scipy's walk; past it, :func:`shortest_path_parents`."""
    if exact_max_voxels is None:
        exact_max_voxels = EXACT_TREE_MAX_VOXELS
    relative = np.asarray(relative_spacing, dtype=float)
    keys = np.asarray(keys, dtype=np.int64)
    edt_at = np.asarray(edt_at, dtype=np.float64)
    n = len(keys)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    voxels = ComponentVoxels(keys, shape)
    max_edt = float(edt_at.max())
    t0 = time.perf_counter()
    if n <= int(exact_max_voxels):
        coords = _coords_of(keys, shape)
        root_v, far_v = _principal_endpoints(coords, relative)
        graph = _build_dijkstra_graph_from_coords(coords, shape, _cost(edt_at), relative)
        walked = _dijkstra_parents(None, None, root_v, precomputed_graph=graph)
        del graph, coords
        if walked is None:
            return keys[[int(np.argmax(edt_at))]]
        parent = walked[0]
        root = int(voxels.ids_at(*root_v))
        far = int(voxels.ids_at(*far_v))
    else:
        root, far = _principal_endpoints_by_chunk(voxels, relative)
        parent = shortest_path_parents(voxels, edt_at, root, relative)
    t_paths = time.perf_counter() - t0

    main = _trace(parent, far, lambda ids: ids == root)[::-1]
    if len(main) < 2 or main[0] != root:
        return keys[[int(np.argmax(edt_at))]]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[main] = True
    cover_r = max(1, int(round(max_edt)))
    min_arm_voxels = max(4, int(2.0 * max_edt))
    candidates = np.flatnonzero(edt_at >= 0.4 * max_edt).astype(np.int32)
    candidate_keys = keys[candidates]
    covered = np.zeros(len(candidates), dtype=bool)
    _cover(voxels, main, cover_r, relative, candidate_keys, covered)
    tree = _GrowingTree(voxels.coord_array(np.asarray(main)).astype(np.float64) * relative)
    farthest = _FarthestCandidates(voxels, candidates, covered, relative, tree)
    component = _VoxelBoxes(keys, shape)

    def touching(ids):
        return voxels.touches(ids, in_tree)

    arms = tried = 0
    last_log = time.perf_counter()
    while True:
        found = farthest.farthest()
        if found is None:
            break
        far_dist, which = found
        # Remaining high-EDT voxels sitting beside the trunk are the sheet, not arms.
        if far_dist < 2.0 * max_edt:
            break
        tried += 1
        target = int(candidates[which])
        branch_ids = _trace(parent, target, touching)[::-1]
        branch = [tuple(int(v) for v in row) for row in voxels.coord_array(np.asarray(branch_ids))]
        if _path_length(branch, relative) < min_arm_voxels:
            _cover(voxels, branch_ids, min(2, cover_r), relative, candidate_keys, covered)
            continue
        if touching(np.asarray([branch_ids[-1]]))[0]:
            # Tip already on the tree: this geodesic would close a loop.
            covered[which] = True
            continue
        if _path_cuts_across_lumen(branch, component, max_edt, relative):
            _cover(voxels, branch_ids, cover_r, relative, candidate_keys, covered)
            continue
        in_tree[branch_ids] = True
        _cover(voxels, branch_ids, cover_r, relative, candidate_keys, covered)
        added = voxels.coord_array(np.asarray(branch_ids)).astype(np.float64) * relative
        tree.add(added)
        farthest.tree_grew(added)
        arms += 1
        if time.perf_counter() - last_log > 60.0:
            last_log = time.perf_counter()
            logger.info(
                "Large-stack centreline tree: %d arms from %d targets so far "
                "(%d of %d high-radius voxels covered)",
                arms,
                tried,
                int(np.count_nonzero(covered)),
                len(candidates),
            )
    if n > 1_000_000:
        logger.info(
            "Large-stack centreline tree: %d voxels, %d arms from %d targets; shortest "
            "paths %.1f s, arms %.1f s",
            n,
            arms,
            tried,
            t_paths,
            time.perf_counter() - t0 - t_paths,
        )
    return keys[in_tree]


def _radius_at(mask, box, keys, shape, relative) -> np.ndarray:
    """The fat region's inscribed radius at the component's voxels (*keys*
    in its *box*, of *shape*), from blocks of the box padded by a voxel:
    ``blockwise_edt.distances_at_points``, handed the voxels a row of its
    first blocks at a time. Cut anywhere else, a block's transform was taken
    again for every piece that fell in it."""
    padded = _expand_slices(box, mask.shape, pad=1)
    origin = np.array([s.start for s in box], dtype=np.int64)
    p_origin = np.array([s.start for s in padded], dtype=np.int64)
    domain = np.array([s.stop - s.start for s in padded], dtype=np.int64)
    halo = np.minimum(
        blockwise_edt.reach_voxels(blockwise_edt.START_REACH_VOXELS * float(np.min(relative)), relative),
        domain,
    )
    core = blockwise_edt._core_extent(domain, halo, memmap_support.LOW_MEMORY_BLOCK_VOXELS)
    out = np.empty(len(keys))
    surfaces = _SurfaceOnce()
    plane = int(shape[1]) * int(shape[2])
    # Block rows in z, as key ranges (the keys are in C order, so by z).
    z_cuts = np.arange(0, int(domain[0]) + int(core[0]), int(core[0])) + p_origin[0] - origin[0]
    at = np.searchsorted(keys, np.clip(z_cuts, 0, int(shape[0])) * plane)
    y_cuts = np.arange(0, int(domain[1]) + int(core[1]), int(core[1])) + p_origin[1] - origin[1]

    def rows():
        for i0, i1 in zip(at[:-1], at[1:]):
            if i1 <= i0:
                continue
            coords = _coords_of(keys[i0:i1], shape)
            row = np.searchsorted(y_cuts, coords[:, 1], side="right")
            order = np.argsort(row, kind="stable")
            bounds = np.searchsorted(row[order], np.arange(row.min(), row.max() + 2))
            for j0, j1 in zip(bounds[:-1], bounds[1:]):
                if j1 > j0:
                    these = order[j0:j1]
                    yield i0 + these, coords[these] + origin

    def measure(row):
        where, points = row
        out[where] = blockwise_edt.distances_at_points(
            lambda region: mask[region], padded, points, sampling=relative, surface_cache=surfaces
        )

    threads = _radius_threads(int(np.prod(domain, dtype=np.int64)))
    if threads == 1:
        for row in rows():
            measure(row)
        return out
    with ThreadPoolExecutor(threads) as pool:
        pending = set()
        try:
            for row in rows():
                pending.add(pool.submit(measure, row))
                # A few rows queued a thread, so their points are not all held at once.
                if len(pending) >= 2 * threads:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
            for future in pending:
                future.result()
        except BaseException:
            surfaces.release()
            raise
    return out


def _radius_threads(domain_voxels: int) -> int:
    """How many threads :func:`_radius_at` measures on: one for a domain
    one block holds, else as many as half the free memory holds blocks of
    the low-RAM option's size, up to :data:`_RADIUS_MAX_THREADS` and the
    cores but one."""
    block = int(memmap_support.LOW_MEMORY_BLOCK_VOXELS)
    if domain_voxels <= block:
        return 1
    from haemolynx.haemodynamics.resistance import available_memory_bytes

    available = available_memory_bytes()
    if available is None:
        return 1
    fit = int(0.5 * float(available) // (block * blockwise_edt.BLOCK_BYTES_PER_VOXEL))
    return max(1, min(_RADIUS_MAX_THREADS, shared_volume_tasks.default_worker_count(), fit))


class _SurfaceOnce(dict):
    """The surface cache ``blockwise_edt.distances_at_points`` keeps, shared
    by threads: the first to ask for a domain's surface finds it, and any
    other asking meanwhile waits for it rather than finding it again (on a
    stack-wide component, gigabytes each)."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._busy: dict = {}

    def get(self, key, default=None):
        while True:
            with self._lock:
                if dict.__contains__(self, key):
                    return dict.__getitem__(self, key)
                event = self._busy.get(key)
                if event is None:
                    self._busy[key] = threading.Event()
                    return default
            event.wait()

    def __setitem__(self, key, value):
        with self._lock:
            dict.__setitem__(self, key, value)
            event = self._busy.pop(key, None)
        if event is not None:
            event.set()

    def release(self) -> None:
        """Wake every thread waiting (after a failure, when the surface
        being found may never come)."""
        with self._lock:
            events = list(self._busy.values())
            self._busy.clear()
        for event in events:
            event.set()


def edt_ridge_by_component(
    out: np.ndarray | None,
    mask: np.ndarray,
    relative: np.ndarray,
    *,
    drawn: list | None = None,
    exact_max_voxels: int | None = None,
) -> None:
    """Draw every fat component's tree (:func:`component_tree`) of *mask*
    into *out* (None: nowhere), adding each one's voxels to *drawn* when
    given: ``thick_vessels._edt_ridge_by_component`` with this module's tree,
    its voxels held as keys (8 bytes each, not 24 as coordinates) and grouped
    by component without an index array."""
    shape = tuple(int(s) for s in mask.shape)
    plane = shape[1] * shape[2]
    step = memmap_support.slab_step(shape)
    total = sum(int(np.count_nonzero(mask[z0:z0 + step])) for z0 in range(0, shape[0], step))
    if total == 0:
        return
    # Every fat voxel's key, with its component number above it, in one
    # array filled as the slabs are labelled: sorted, the voxels come by
    # component and, within one, in C order. Collected as key, label and
    # number arrays and joined, the labelling was 40 bytes a fat voxel at
    # once (5.8 GB on the patella's 144 M); this is 8.
    shift = max(1, int(np.prod(shape, dtype=np.int64) - 1).bit_length())
    if int(total).bit_length() + shift > 62:
        raise ValueError(f"a {shape} volume with {total} fat voxels is too large to sort")
    combined = np.empty(total, dtype=np.int64)
    filled = 0

    def visit(z0, _z1, labels, offset):
        nonlocal filled
        flat = labels.reshape(-1)
        at = np.flatnonzero(flat)
        part = combined[filled:filled + len(at)]
        part[:] = flat[at]
        part += int(offset)
        part <<= shift
        part |= at + int(z0) * plane
        filled += len(at)

    components = slab_labels.label_by_slab(
        lambda z0, z1: mask[z0:z1], shape, structure=_STRUCTURE_FULL, visit=visit
    )
    if filled != total:
        raise RuntimeError(f"labelled {filled} fat voxels of {total}")
    logger.info(
        "skeletonize_edt_ridge (large-stack tree): %d fat component(s) in a %s volume (%d voxels)",
        components.count,
        shape,
        total,
    )
    low = (1 << shift) - 1
    for s0 in range(0, total, _CHUNK):
        part = combined[s0:s0 + _CHUNK]
        numbers = components.number[part >> shift]
        part &= low
        part |= numbers << shift
    del numbers
    combined.sort()
    bounds = np.searchsorted(combined, np.arange(1, components.count + 2, dtype=np.int64) << shift)
    sizes = np.diff(bounds)
    # The largest last, so every voxel list but its own is freed first.
    largest = int(np.argmax(sizes)) + 1
    small = [c for c in range(1, components.count + 1) if c != largest and sizes[c - 1] < _THREADED_COMPONENT_VOXELS]
    large = [c for c in range(1, components.count + 1) if c != largest and sizes[c - 1] >= _THREADED_COMPONENT_VOXELS]

    def draw(component_id: int) -> None:
        part = combined[bounds[component_id - 1]:bounds[component_id]] & low
        _draw_component(out, drawn, mask, relative, components, component_id, part, shape, exact_max_voxels)

    threads = min(_COMPONENT_THREADS, shared_volume_tasks.default_worker_count())
    if threads > 1 and len(small) > 1:
        with ThreadPoolExecutor(threads) as pool:
            for future in [pool.submit(draw, c) for c in small]:
                future.result()
    else:
        for c in small:
            draw(c)
    for c in large:
        draw(c)
    part = combined[bounds[largest - 1]:bounds[largest]] & low
    del combined
    _draw_component(out, drawn, mask, relative, components, largest, part, shape, exact_max_voxels)


def _draw_component(out, drawn, mask, relative, components, component_id, keys, shape, exact_max_voxels):
    """Draw component *component_id*'s tree (its voxels' *keys* in the
    volume of *shape*) into *out* and *drawn*."""
    box = components.box(component_id)
    box_shape = tuple(int(s.stop) - int(s.start) for s in box)
    origin = np.array([s.start for s in box], dtype=np.int64)
    local = np.empty(len(keys), dtype=np.int64)
    for s in range(0, len(keys), _CHUNK):
        c = _coords_of(keys[s:s + _CHUNK], shape) - origin
        local[s:s + _CHUNK] = (c[:, 0] * box_shape[1] + c[:, 1]) * box_shape[2] + c[:, 2]
    count = len(keys)
    del keys
    t = time.perf_counter()
    edt_at = _radius_at(mask, box, local, box_shape, relative)
    if count > 1_000_000:
        logger.info(
            "skeletonize_edt_ridge: the radius at component %d's %d voxels in %.1fs",
            component_id,
            count,
            time.perf_counter() - t,
        )
    tree = component_tree(local, edt_at, box_shape, relative, exact_max_voxels=exact_max_voxels)
    del edt_at, local
    voxels = _coords_of(tree, box_shape) + origin
    # Components are apart, so threads drawing two never touch one voxel.
    if out is not None:
        out[tuple(voxels.T)] = True
    if drawn is not None:
        drawn.append(voxels)
    if components.count > 1 or count > 10_000:
        logger.info(
            "skeletonize_edt_ridge: component %d/%d (%d voxels) took %.2fs",
            component_id,
            components.count,
            count,
            time.perf_counter() - t,
        )
