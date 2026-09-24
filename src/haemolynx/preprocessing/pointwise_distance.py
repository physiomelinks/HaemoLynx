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

from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from .memmap_support import LOW_MEMORY_BLOCK_VOXELS, iter_blocks

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)

#: Nearest candidates checked per query before falling back to a radius
#: search. Any feature tied with the nearest one must be among them.
_CANDIDATES = 8


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
        block = np.asarray(features[padded], dtype=bool)
        feature = block if feature_value else ~block
        other = ~feature
        if not other.any() or not feature.any():
            continue
        surface = (feature & binary_dilation(other, structure=_STRUCTURE_26))[inner]
        found = np.argwhere(surface)
        if found.size:
            found += np.array([s.start for s in core])
            parts.append(found)
    if not parts:
        return np.empty((0, features.ndim), dtype=np.intp)
    found = np.concatenate(parts)
    return found[np.lexsort(found.T[::-1])]


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
    """

    def __init__(
        self,
        features: np.ndarray,
        *,
        feature_value: bool = True,
        sampling: Sequence[float] | None = None,
        block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
    ):
        self.features = features
        self.feature_value = bool(feature_value)
        self.sampling = (
            np.ones(features.ndim) if sampling is None else np.asarray(sampling, dtype=float)
        )
        self.surface = surface_feature_voxels(
            features, feature_value=feature_value, block_voxels=block_voxels
        )
        self._tree = cKDTree(self.surface * self.sampling) if len(self.surface) else None

    @property
    def has_surface(self) -> bool:
        """False when the volume is all feature or all non-feature -- where
        scipy's own transform is not a distance to anything."""
        return self._tree is not None

    def at(self, voxels: np.ndarray) -> np.ndarray:
        """Distance at each integer voxel of *voxels*, shape ``(n, ndim)``.

        A feature voxel is at distance 0.
        """
        voxels = np.asarray(voxels, dtype=np.intp).reshape(-1, self.features.ndim)
        out = np.zeros(len(voxels), dtype=np.float64)
        if not len(voxels):
            return out
        if not self.has_surface:
            raise ValueError("no feature surface: every voxel is on the same side")
        is_feature = (
            np.asarray(self.features[tuple(voxels.T)], dtype=bool) == self.feature_value
        )
        query = voxels[~is_feature]
        if not len(query):
            return out
        k = min(_CANDIDATES, len(self.surface))
        tree_distance, index = self._tree.query(query * self.sampling, k=k)
        tree_distance = tree_distance.reshape(len(query), k)
        index = index.reshape(len(query), k)
        best = np.full(len(query), np.inf)
        for column in range(k):
            best = np.minimum(
                best, _scipy_distance(self.surface[index[:, column]] - query, self.sampling)
            )
        # A tie can hide past the k-th candidate only if all k are (nearly) tied.
        if k == _CANDIDATES:
            slack = tree_distance[:, 0] * 1e-9 + 1e-12
            for row in np.flatnonzero(tree_distance[:, -1] <= tree_distance[:, 0] + slack):
                near = self._tree.query_ball_point(
                    query[row] * self.sampling, tree_distance[row, 0] + slack[row]
                )
                best[row] = min(
                    best[row],
                    _scipy_distance(self.surface[near] - query[row], self.sampling).min(),
                )
        out[~is_feature] = best
        return out
