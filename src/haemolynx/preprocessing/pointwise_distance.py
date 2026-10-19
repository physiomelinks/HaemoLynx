"""Euclidean distance-transform values at chosen voxels, without the volume.

``scipy.ndimage.distance_transform_edt`` of a whole stack needs roughly 70
bytes of RAM per voxel internally, whatever it is asked to write into -- the
thing the low-RAM option cannot afford. A caller that only *reads* the
transform at some voxels (a diameter sampled along each edge, a routing
window around two clicked points) does not need the rest of it.

The value at a voxel is its distance to the nearest *feature* voxel (for an
EDT of a mask, the nearest background voxel). That nearest feature always
touches the other side: stepping from a feature one voxel towards the query
along every axis where they differ shortens the distance, so if that
neighbour were a feature too the first could not have been nearest. So only
features 26-adjacent to a non-feature voxel -- the surface, whose size grows
with the vessels' surface area rather than the volume -- can ever be
nearest, and a KD-tree over them answers any query.

Distances are then recomputed exactly the way scipy computes them from the
displacement (scale each axis, square, add in axis order, square root), so a
voxel with one nearest feature gets scipy's value bit for bit. Where two
features are *exactly* equidistant, scipy keeps whichever its sweep met
first; with a voxel spacing that is not exactly representable, the two
formula values can differ in the last bit, so this can come out one ulp
(about 2e-16 relative) from scipy at such a voxel. With unit or
power-of-two spacings every term is exact and it never differs.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .memmap_support import LOW_MEMORY_BLOCK_VOXELS, iter_blocks

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)

#: Nearest candidates checked per query before falling back to a radius
#: search. Any feature tied with the nearest one must be among them.
_CANDIDATES = 8

#: How many voxels' lumen radii graph building's lookups remember
#: (``FeatureDistance(remember=...)``): more than the centreline voxels and
#: their neighbours a large stack's clean-up reads, at about 100 bytes each.
LUMEN_RADII_REMEMBERED = 2_000_000

#: Voxels along each axis of a tile of the low-RAM lookup
#: (``FeatureDistance(by_block=True)``): wide enough that most lumen radii
#: are answered inside the tile a voxel is in.
SURFACE_TILE_EDGE = 96

#: Bytes of tile surfaces and their k-d trees the low-RAM lookup keeps,
#: least recently used dropped first.
SURFACE_TILE_CACHE_BYTES = 256 << 20

#: Bytes a surface voxel costs in a tile: its coordinates and its share of
#: the tile's k-d tree.
_SURFACE_VOXEL_BYTES = 72


def surface_feature_voxels(
    features: np.ndarray,
    *,
    feature_value: bool = True,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
) -> np.ndarray:
    """Feature voxels 26-adjacent to a non-feature voxel, in C order.

    A voxel is a feature where ``features == feature_value`` -- pass
    ``feature_value=False`` for the background of a mask without inverting
    the whole mask. Worked out one padded block at a time.
    """
    parts = []
    for padded, inner, core in iter_blocks(features.shape, halo=1, block_voxels=block_voxels):
        found = _block_surface(np.asarray(features[padded], dtype=bool), feature_value, inner, core)
        if found.size:
            parts.append(found)
    if not parts:
        return np.empty((0, features.ndim), dtype=np.intp)
    found = np.concatenate(parts)
    return found[np.lexsort(found.T[::-1])]


def _block_surface(block: np.ndarray, feature_value: bool, inner, core) -> np.ndarray:
    """The surface features of *block* (a padded box of the volume) inside
    *inner*, in C order and the volume's coordinates (*core* is where
    *inner* lies in the volume)."""
    feature = block if feature_value else ~block
    other = ~feature
    if not other.any() or not feature.any():
        return np.empty((0, block.ndim), dtype=np.intp)
    found = np.argwhere((feature & binary_dilation(other, structure=_STRUCTURE_26))[inner])
    found += np.array([s.start for s in core])
    return found


def _scipy_distance(displacement: np.ndarray, sampling: np.ndarray) -> np.ndarray:
    """``distance_transform_edt``'s own arithmetic, for (n, ndim) displacements."""
    dt = displacement.T.astype(np.float64)
    for axis in range(dt.shape[0]):
        dt[axis] *= sampling[axis]
    np.multiply(dt, dt, dt)
    return np.sqrt(np.add.reduce(dt, axis=0))


class FeatureDistance:
    """Distance from any voxel to the nearest feature voxel of a volume.

    Built once from the volume's surface features; each query costs a KD-tree
    lookup. ``features`` and ``feature_value`` are as for
    :func:`surface_feature_voxels`; *sampling* is the per-axis spacing, as
    ``distance_transform_edt`` takes it (``None`` for unit spacing).

    *remember* keeps the distances of up to that many voxels once measured,
    for a caller that asks about the same voxels again and again: graph
    building's clean-up reads the lumen radius along the same centrelines
    round after round, and on E14.5 MCA these lookups were over a third of
    its time. Off (0) by default -- a caller visiting every voxel once, as
    :func:`distance_transform_edt_blockwise` does, would only pay its memory
    (about 100 bytes a voxel).

    *packed* (``bridge_mask_support.PackedMask``) is *features* as bits, read
    in place of *features* to tell whether a queried voxel is itself a
    feature: the same answer, from RAM rather than a page of a memmap read
    at random (``MaskSupport`` hands its own over once it has packed).

    *by_block* is the low-RAM option's lookup: no tree over the whole
    surface (about 69 bytes a surface voxel, 7-9 GB on a 5.5 G-voxel stack)
    but one per tile of *tile_edge* voxels a side, made when a query first
    needs it and kept, least recently used dropped first, to
    *tile_cache_bytes* (:class:`_SurfaceTiles`). The same distances: either
    way a query gets the least of scipy's arithmetic over every feature.
    """

    def __init__(
        self,
        features: np.ndarray,
        *,
        feature_value: bool = True,
        sampling: Sequence[float] | None = None,
        block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
        remember: int = 0,
        packed=None,
        by_block: bool = False,
        tile_edge: int = SURFACE_TILE_EDGE,
        tile_cache_bytes: int = SURFACE_TILE_CACHE_BYTES,
    ):
        self.features = features
        self.packed = packed
        self.feature_value = bool(feature_value)
        self.sampling = (
            np.ones(features.ndim) if sampling is None else np.asarray(sampling, dtype=float)
        )
        self._tiles = None
        if by_block:
            self.surface = None
            self._tree = None
            if _has_both_values(features, block_voxels):
                self._tiles = _SurfaceTiles(self, tile_edge, tile_cache_bytes)
        else:
            self.surface = surface_feature_voxels(
                features, feature_value=feature_value, block_voxels=block_voxels
            )
            self._tree = cKDTree(self.surface * self.sampling) if len(self.surface) else None
        self._remember = int(remember)
        #: Distances already measured, by flat voxel index.
        self._known: dict[int, float] = {}

    @property
    def has_surface(self) -> bool:
        """False when the volume is all feature or all non-feature -- where
        scipy's own transform is not a distance to anything."""
        return self._tree is not None or self._tiles is not None

    def at(self, voxels: np.ndarray) -> np.ndarray:
        """Distance at each integer voxel of *voxels*, shape ``(n, ndim)``.

        A feature voxel is at distance 0. With *remember* on, a voxel
        measured before is answered from memory: the same value.
        """
        voxels = np.asarray(voxels, dtype=np.intp).reshape(-1, self.features.ndim)
        shape = self.features.shape
        if (
            self._remember <= 0
            or not len(voxels)
            or not self.has_surface
            or (voxels < 0).any()
            or (voxels >= np.asarray(shape)).any()
        ):
            # Out-of-range voxels keep plain indexing's behaviour (a negative
            # index wraps, one past the end raises), never a remembered value.
            return self._measure(voxels)
        keys = np.ravel_multi_index(tuple(voxels.T), shape).tolist()
        known = self._known
        found = [known.get(key) for key in keys]
        missing = np.fromiter((value is None for value in found), dtype=bool, count=len(found))
        out = np.fromiter(
            (0.0 if value is None else value for value in found), dtype=np.float64, count=len(found)
        )
        if missing.any():
            fresh = self._measure(voxels[missing])
            out[missing] = fresh
            room = self._remember - len(known)
            if room > 0:
                new = [key for key, absent in zip(keys, missing) if absent]
                known.update(zip(new[:room], fresh[:room].tolist()))
        return out

    def _measure(self, voxels: np.ndarray) -> np.ndarray:
        """:meth:`at` without memory: the KD-tree lookup itself."""
        out = np.zeros(len(voxels), dtype=np.float64)
        if not len(voxels):
            return out
        if not self.has_surface:
            raise ValueError("no feature surface: every voxel is on the same side")
        if (
            self.packed is not None
            and (voxels >= 0).all()
            and (voxels < np.asarray(self.features.shape)).all()
        ):
            values = self.packed.contains(voxels)
        else:
            values = np.asarray(self.features[tuple(voxels.T)], dtype=bool)
        is_feature = values == self.feature_value
        query = voxels[~is_feature]
        if not len(query):
            return out
        if self._tiles is not None:
            out[~is_feature] = self._tiles.nearest(query)
        else:
            out[~is_feature] = _nearest_distances(self.surface, self._tree, query, self.sampling)
        return out


def _nearest_distances(surface: np.ndarray, tree: cKDTree, query: np.ndarray, sampling) -> np.ndarray:
    """The least of scipy's distances from each voxel of *query* to the
    features *surface*, held in *tree* (scaled by *sampling*).

    The tree's own distances round differently, so its nearest few are
    measured by scipy's arithmetic and the least kept; a feature tied with
    the nearest is among them unless all of them are tied, when the ball
    round the nearest is measured too. So this is the least over every
    feature in *surface*, whatever the tree's rounding.
    """
    k = min(_CANDIDATES, len(surface))
    tree_distance, index = tree.query(query * sampling, k=k)
    tree_distance = tree_distance.reshape(len(query), k)
    index = index.reshape(len(query), k)
    best = np.full(len(query), np.inf)
    for column in range(k):
        best = np.minimum(best, _scipy_distance(surface[index[:, column]] - query, sampling))
    # A tie can hide past the k-th candidate only if all k are (nearly) tied.
    if k == _CANDIDATES:
        slack = tree_distance[:, 0] * 1e-9 + 1e-12
        for row in np.flatnonzero(tree_distance[:, -1] <= tree_distance[:, 0] + slack):
            near = tree.query_ball_point(query[row] * sampling, tree_distance[row, 0] + slack[row])
            best[row] = min(best[row], _scipy_distance(surface[near] - query[row], sampling).min())
    return best


def _has_both_values(features: np.ndarray, block_voxels: int) -> bool:
    """Whether *features* holds a True voxel and a False one -- so has a
    surface -- read a slab of whole axis-0 planes at a time, stopping as
    soon as it has seen both."""
    per_plane = int(np.prod(features.shape[1:], dtype=np.int64)) or 1
    step = max(1, int(block_voxels) // per_plane)
    seen_true = seen_false = False
    for start in range(0, int(features.shape[0]), step):
        slab = np.asarray(features[start:start + step], dtype=bool)
        seen_true = seen_true or bool(slab.any())
        seen_false = seen_false or not bool(slab.all())
        if seen_true and seen_false:
            return True
    return False


class _SurfaceTiles:
    """:class:`FeatureDistance`'s low-RAM lookup: the volume cut into tiles,
    each tile's surface features (found with a voxel of context, so exactly
    the whole surface's inside the tile) and their k-d tree made when a
    query first needs them, and kept to a byte budget, least recently used
    dropped first.

    A query is measured against the tile it is in, then against the tiles
    one tile further out, then two, ..., each only where the tile's box is
    nearer than the best so far by scipy's own arithmetic. It stops once
    the best is no further than the nearest voxel outside the tiles
    searched: a feature there is at least that far along one axis, and
    scipy's distance is never less than one axis's term. So the answer is
    the least over every feature, as the whole surface's tree gives.
    """

    def __init__(self, owner: FeatureDistance, tile_edge: int, cache_bytes: int):
        self.owner = owner
        self.shape = np.asarray(owner.features.shape, dtype=np.int64)
        self.tile = np.minimum(np.full(len(self.shape), max(1, int(tile_edge)), np.int64), self.shape)
        self.grid = -(-self.shape // self.tile)
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[int, tuple[np.ndarray, cKDTree | None]] = OrderedDict()
        self._held = 0
        self._rings: dict[int, np.ndarray] = {}

    def _ring(self, r: int) -> np.ndarray:
        """Tile offsets at Chebyshev distance *r*."""
        found = self._rings.get(r)
        if found is None:
            span = np.arange(-r, r + 1)
            grid = np.stack(np.meshgrid(*([span] * len(self.shape)), indexing="ij"), -1)
            grid = grid.reshape(-1, len(self.shape))
            found = grid[np.abs(grid).max(axis=1) == r]
            self._rings[r] = found
        return found

    def _surface(self, key: int) -> tuple[np.ndarray, cKDTree | None]:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        index = np.array(np.unravel_index(key, tuple(self.grid)), dtype=np.int64)
        lo = index * self.tile
        hi = np.minimum(lo + self.tile, self.shape)
        p_lo = np.maximum(lo - 1, 0)
        p_hi = np.minimum(hi + 1, self.shape)
        padded = tuple(slice(int(a), int(b)) for a, b in zip(p_lo, p_hi))
        owner = self.owner
        if owner.packed is not None:
            block = owner.packed.box(padded)
        else:
            block = np.asarray(owner.features[padded], dtype=bool)
        inner = tuple(slice(int(a - p), int(b - p)) for a, b, p in zip(lo, hi, p_lo))
        core = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        surface = _block_surface(block, owner.feature_value, inner, core)
        entry = (surface, cKDTree(surface * owner.sampling) if len(surface) else None)
        self._cache[key] = entry
        self._held += len(surface) * _SURFACE_VOXEL_BYTES
        while self._held > self.cache_bytes and len(self._cache) > 1:
            _key, (old, _tree) = self._cache.popitem(last=False)
            self._held -= len(old) * _SURFACE_VOXEL_BYTES
        return entry

    def _search(self, query: np.ndarray, best: np.ndarray, which: np.ndarray, tiles: np.ndarray) -> None:
        """Lower ``best[which[i]]`` to the nearest feature of tile ``tiles[i]``
        (grid indices), where nearer."""
        keys = np.ravel_multi_index(tuple(tiles.T), tuple(self.grid))
        order = np.argsort(keys, kind="stable")
        keys, which = keys[order], which[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        for start, stop in zip(starts, np.r_[starts[1:], len(keys)]):
            surface, tree = self._surface(int(keys[start]))
            if tree is None:
                continue
            rows = which[start:stop]
            found = _nearest_distances(surface, tree, query[rows], self.owner.sampling)
            best[rows] = np.minimum(best[rows], found)

    def _outside(self, query: np.ndarray, home: np.ndarray, r: int) -> np.ndarray:
        """The distance from each query to the nearest voxel outside the
        tiles within *r* of its home tile (``inf`` where they hold the whole
        volume): no feature there is nearer, by scipy's arithmetic."""
        sampling = self.owner.sampling
        lo = np.maximum((home - r) * self.tile, 0)
        hi = np.minimum((home + r + 1) * self.tile, self.shape)
        bound = np.full(len(query), np.inf)
        for axis in range(len(self.shape)):
            below = lo[:, axis] > 0
            bound[below] = np.minimum(
                bound[below], (query[below, axis] - lo[below, axis] + 1) * sampling[axis]
            )
            above = hi[:, axis] < self.shape[axis]
            bound[above] = np.minimum(
                bound[above], (hi[above, axis] - query[above, axis]) * sampling[axis]
            )
        return bound

    def nearest(self, query: np.ndarray) -> np.ndarray:
        """The least of scipy's distances from each voxel of *query* to any feature."""
        query = np.asarray(query, dtype=np.intp)
        best = np.full(len(query), np.inf)
        home = np.clip(query // self.tile, 0, self.grid - 1)
        everyone = np.arange(len(query))
        self._search(query, best, everyone, home)
        todo = everyone
        for r in range(1, int(self.grid.max()) + 1):
            todo = todo[best[todo] > self._outside(query[todo], home[todo], r - 1)]
            if not todo.size:
                break
            ring = self._ring(r)
            which = np.repeat(todo, len(ring))
            tiles = home[which] + np.tile(ring, (len(todo), 1))
            inside = np.all((tiles >= 0) & (tiles < self.grid), axis=1)
            which, tiles = which[inside], tiles[inside]
            # Only tiles whose box could hold a feature nearer than the best.
            lo = tiles * self.tile
            hi = np.minimum(lo + self.tile, self.shape) - 1
            q = query[which]
            gap = np.maximum(np.maximum(lo - q, q - hi), 0)
            nearer = _scipy_distance(gap, self.owner.sampling) < best[which]
            if nearer.any():
                self._search(query, best, which[nearer], tiles[nearer])
        return best


#: Context read around each block by :func:`distance_transform_edt_blockwise`.
#: Voxels deeper than this inside a vessel are looked up instead, so it only
#: sets how many need looking up, never the result.
BLOCKWISE_EDT_HALO = 16


def distance_transform_edt_blockwise(
    features: np.ndarray,
    out: np.ndarray,
    *,
    feature_value: bool = False,
    sampling: Sequence[float] | None = None,
    halo: int = BLOCKWISE_EDT_HALO,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
) -> np.ndarray:
    """``distance_transform_edt`` written into *out*, one padded block at a time.

    With the default ``feature_value=False`` this is
    ``distance_transform_edt(features, sampling=sampling)`` -- each voxel's
    distance to the nearest zero voxel. Each block is transformed with
    *halo* voxels of context; a voxel's value there is kept when no feature
    outside the padded block could be nearer -- every such feature is at
    least one step beyond an inner face -- and is otherwise looked up with
    :class:`FeatureDistance`. Same values as the whole-volume call, subject
    to the exact-tie caveat in this module's docstring. Only one padded
    block (plus scipy's ~70 bytes per voxel of it) is ever in RAM.

    Raises ``ValueError`` for a volume with no feature voxel or no
    non-feature voxel, where scipy's own result is not a distance.
    """
    from scipy.ndimage import distance_transform_edt

    ndim = features.ndim
    spacing = np.ones(ndim) if sampling is None else np.asarray(sampling, dtype=float)
    lookup: FeatureDistance | None = None
    shape = np.asarray(features.shape)
    for _padded, _inner, core in iter_blocks(features.shape, halo=0, block_voxels=block_voxels):
        lo = np.array([s.start for s in core])
        hi = np.array([s.stop for s in core])
        plo = np.maximum(lo - halo, 0)
        phi = np.minimum(hi + halo, shape)
        crop = np.asarray(features[tuple(slice(a, b) for a, b in zip(plo, phi))], dtype=bool)
        is_feature = crop if feature_value else ~crop
        inner = tuple(slice(a - p, b - p) for a, b, p in zip(lo, hi, plo))
        if not is_feature[inner].all():
            if is_feature.any():
                local = distance_transform_edt(~is_feature, sampling=spacing)[inner]
                bound = np.full(local.shape, np.inf)
                for axis in range(ndim):
                    position = np.arange(lo[axis], hi[axis]).reshape(
                        [-1 if a == axis else 1 for a in range(ndim)]
                    )
                    if plo[axis] > 0:
                        bound = np.minimum(bound, (position - plo[axis] + 1) * spacing[axis])
                    if phi[axis] < shape[axis]:
                        bound = np.minimum(bound, (phi[axis] - position) * spacing[axis])
                unsure = local > bound
            else:
                local = np.zeros(tuple(hi - lo))
                unsure = np.ones(local.shape, dtype=bool)
            if unsure.any():
                if lookup is None:
                    lookup = FeatureDistance(
                        features,
                        feature_value=feature_value,
                        sampling=spacing,
                        block_voxels=block_voxels,
                    )
                    if not lookup.has_surface:
                        raise ValueError("no feature surface: every voxel is on the same side")
                local[unsure] = lookup.at(np.argwhere(unsure) + lo)
            out[core] = local
        else:
            out[core] = 0.0
    return out
