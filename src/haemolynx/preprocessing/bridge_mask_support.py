"""Whether a bridge between two pieces of vessel runs through the segmentation.

Every step that joins two pieces of vessel -- skeleton bridges, bundle-hub
spokes, the graph's gap, optimise and orphan reconnects, the recovery of mask
the graph left uncovered -- draws a path between them. Within reach of each
other is not enough for that path to be a vessel: in a dense capillary bed the
nearest end is often the next capillary over, across background. So a path is
sampled against the mask it was traced from, and accepted only where it stays
in the lumen bar a short dropout (:func:`bridge_mask_support`,
:class:`MaskSupport`).

A path through the mask can still be a vessel the network already has: a
second strand beside the first inside one wide lumen. :func:`path_shadows_existing_vessel`
is the guard against that, the same question at every site that adds a path.

Self-contained on purpose: :mod:`haemolynx.graph` imports this module, never
the other way round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "BridgeMaskSupport",
    "DEFAULT_MAX_BACKGROUND_GAP_UM",
    "DEFAULT_MIN_MASK_FRACTION",
    "MaskSupport",
    "bridge_is_supported",
    "bridge_mask_support",
    "path_shadows_existing_vessel",
    "shadowed_samples",
]

#: Longest run of background a bridge may cross: a one- or two-voxel dropout
#: in the segmentation of a vessel that is really there.
DEFAULT_MAX_BACKGROUND_GAP_UM = 2.0

#: Least fraction of a bridge's length that must lie in the mask.
DEFAULT_MIN_MASK_FRACTION = 0.8

#: Background a chord must cross to separate two paths, the slack on "within
#: two radii", and how far past its radius an attachment end is left out:
#: one micron, about one voxel of a typical stack.
SAME_LUMEN_MARGIN_UM = 1.0

#: A sample's nearest existing centreline point lies beside it, not ahead or
#: behind, when the chord to it makes at most this cosine with the path.
MAX_SHADOW_COSINE = 0.7


@dataclass(frozen=True)
class BridgeMaskSupport:
    """How much of a path lies in the mask, and its longest stretch outside."""

    inside_fraction: float
    longest_background_um: float
    length_um: float


def _sample_step(voxel_size_zyx: Sequence[float]) -> float:
    """At most 0.5 um, and an even fraction of the finest voxel: pieces that
    long, read at their midpoints, then never straddle a voxel boundary on a
    path stepping from voxel centre to voxel centre, so an n-voxel dropout
    along an axis reads n voxels exactly."""
    finest = float(np.min(voxel_size_zyx))
    return finest / (2 * max(1, int(np.ceil(finest - 1e-9))))


def _counts(lengths: np.ndarray, step_um: float) -> np.ndarray:
    return np.maximum(np.ceil(lengths / step_um - 1e-9).astype(int), 1)


def _split(arr: np.ndarray, step_um: float, offset: float):
    """Each segment of *arr* cut into equal pieces of at most *step_um*:
    the point *offset* (0..1) of the way along each piece, and its length."""
    segments = np.diff(arr, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    counts = _counts(lengths, step_um)
    piece = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    t = ((piece + offset) / np.repeat(counts, counts))[:, None]
    points = np.repeat(arr[:-1], counts, axis=0) + t * np.repeat(segments, counts, axis=0)
    return points, np.repeat(lengths / counts, counts)


def _densify(points: np.ndarray, step_um: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(arr) < 2:
        return arr
    return np.concatenate([arr[:1], _split(arr, step_um, 1.0)[0]])


def _pieces(points: np.ndarray, step_um: float) -> tuple[np.ndarray, np.ndarray]:
    """``(midpoints, lengths)`` of the path cut into equal pieces of at most
    *step_um* per segment."""
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(arr) < 2:
        return arr, np.zeros(len(arr))
    return _split(arr, step_um, 0.5)


def _points_in_mask(points_um: np.ndarray, mask: np.ndarray, voxel_size_zyx) -> np.ndarray:
    """True where the point's nearest voxel is in the mask; outside the volume
    is background."""
    points = np.asarray(points_um, dtype=float).reshape(-1, 3)
    idx = np.rint(points / np.asarray(voxel_size_zyx, dtype=float)).astype(np.intp)
    in_bounds = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
    inside = np.zeros(len(points), dtype=bool)
    valid = idx[in_bounds]
    if len(valid):
        inside[in_bounds] = np.asarray(mask[valid[:, 0], valid[:, 1], valid[:, 2]], dtype=bool)
    return inside


def _support_of(weights: np.ndarray, inside: np.ndarray) -> BridgeMaskSupport:
    total = float(weights.sum())
    if total <= 0.0:
        return BridgeMaskSupport(float(inside.all()) if len(inside) else 0.0, 0.0, 0.0)
    outside = np.where(~inside, weights, 0.0)
    # Cumulative sum restarted at every inside sample: the running length of
    # the background run each sample ends.
    run_start = np.maximum.accumulate(np.where(inside, np.arange(len(inside)), -1))
    cumulative = np.concatenate([[0.0], np.cumsum(outside)])
    runs = cumulative[1:] - cumulative[np.maximum(run_start, -1) + 1]
    return BridgeMaskSupport(
        inside_fraction=float(weights[inside].sum() / total),
        longest_background_um=float(runs.max()) if len(runs) else 0.0,
        length_um=total,
    )


def bridge_mask_support(
    path_um,
    mask: np.ndarray | None = None,
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    inside: Callable[[np.ndarray], np.ndarray] | None = None,
) -> BridgeMaskSupport:
    """How a physical ``(z, y, x)`` path sits in the mask.

    The path is cut into pieces of half a voxel of the finest axis (at most
    0.5 um), each read at its midpoint, so a one-voxel dropout reads one
    voxel of background (exactly, along an axis between voxel centres).
    A path with a single point has length 0 and no background.
    *inside* (``points -> bool``) replaces the nearest-voxel test on *mask*.
    """
    if inside is None and mask is None:
        raise ValueError("bridge_mask_support needs a mask or an inside test")
    mids, weights = _pieces(np.asarray(path_um, dtype=float), _sample_step(voxel_size_zyx))
    if inside is None:
        flags = _points_in_mask(mids, mask, voxel_size_zyx)
    else:
        flags = np.asarray(inside(mids), dtype=bool)
    return _support_of(weights, flags)


def bridge_is_supported(
    support: BridgeMaskSupport,
    *,
    max_background_gap_um: float = DEFAULT_MAX_BACKGROUND_GAP_UM,
    min_mask_fraction: float = DEFAULT_MIN_MASK_FRACTION,
) -> bool:
    """A bridge is a vessel when its longest background run is at most
    *max_background_gap_um* and at least *min_mask_fraction* of it is mask.

    The fraction is waived while the background, all told, is no more than
    one allowed run: a bridge across a two-voxel dropout is barely longer
    than the dropout, and held to the fraction no such bridge would ever be
    drawn. It is what stops a long path of many short dropouts.
    """
    gap = float(max_background_gap_um)
    background_um = (1.0 - support.inside_fraction) * support.length_um
    return support.longest_background_um <= gap + 1e-9 and (
        support.inside_fraction >= float(min_mask_fraction) - 1e-9
        or background_um <= gap + 1e-9
    )


def path_shadows_existing_vessel(
    path_um,
    existing_tree: cKDTree | None,
    existing_points_um: np.ndarray,
    inside: Callable[[np.ndarray], np.ndarray],
    radius_at: Callable[[np.ndarray], np.ndarray],
    *,
    step_um: float = 0.5,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    max_cosine: float = MAX_SHADOW_COSINE,
) -> bool:
    """Whether a new path runs beside an existing centreline in the same lumen.

    The path is sampled every *step_um*; samples within (the lumen radius at
    that end + *margin_um*) of either end are left out, since a branch meets
    the vessel it joins there. A remaining sample is *shadowed* when its
    nearest existing centreline point lies beside it (the chord makes a
    cosine under *max_cosine* with the path, so a continuation whose nearest
    point is straight ahead or behind does not count), the chord crosses
    less than *margin_um* of background (one lumen, not two), and the chord
    is no longer than two lumen radii at the sample plus *margin_um*. The
    path duplicates a vessel when at least half its remaining samples are
    shadowed; with none remaining it does not.

    *existing_points_um* are the centreline points (``(M, 3)``) and
    *existing_tree* a KD-tree over them (built here when ``None``);
    *inside* and *radius_at* map ``(N, 3)`` points to a bool / a lumen
    radius in microns each.
    """
    judged, shadowing = _shadowing_points(
        path_um, existing_tree, existing_points_um, inside, radius_at,
        step_um=step_um, margin_um=margin_um, max_cosine=max_cosine,
    )
    return judged > 0 and 2 * len(shadowing) >= judged


def _judged_samples(path_um, radius_at, *, step_um: float, margin_um: float):
    """*path_um* sampled every *step_um*, each sample's tangent, and which
    samples are judged: those further than (the lumen radius at that end +
    *margin_um*) from both ends, where a branch meets the vessel it joins."""
    samples = _densify(np.asarray(path_um, dtype=float), step_um)
    if len(samples) < 2:
        return samples, np.zeros_like(samples), np.zeros(len(samples), dtype=bool)
    tangents = np.gradient(samples, axis=0)
    ends = samples[[0, -1]]
    end_radii = np.asarray(radius_at(ends), dtype=float).reshape(2)
    keep = (np.linalg.norm(samples - ends[0], axis=1) > end_radii[0] + margin_um) & (
        np.linalg.norm(samples - ends[1], axis=1) > end_radii[1] + margin_um
    )
    return samples, tangents, keep


def _same_lumen_neighbours(
    samples,
    tangents,
    existing_tree,
    existing,
    inside,
    radius_at,
    *,
    step_um: float,
    margin_um: float,
    max_cosine: float,
    min_share: float = 0.0,
) -> np.ndarray:
    """Per sample, the index into *existing* of its nearest centreline point
    when that point lies beside it in the same lumen, else -1. With
    *min_share*, every sample is -1 as soon as fewer than that share of them
    could qualify: no chord is then sampled against the mask."""
    neighbour = np.full(len(samples), -1, dtype=np.intp)
    if len(samples) == 0:
        return neighbour
    tree = existing_tree if existing_tree is not None else cKDTree(existing)
    chord_length, nearest = tree.query(samples)
    chords = existing[nearest] - samples
    tangent_norm = np.linalg.norm(tangents, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.abs(np.sum(chords * tangents, axis=1)) / (chord_length * tangent_norm)
    beside = (chord_length < 1e-9) | (np.nan_to_num(cosine, nan=0.0) < max_cosine)
    close = chord_length <= 2.0 * np.asarray(radius_at(samples), dtype=float) + margin_um
    candidates = np.flatnonzero(beside & close)
    if len(candidates) == 0 or len(candidates) < min_share * len(samples):
        return neighbour
    lengths = chord_length[candidates]
    n = int(np.ceil(lengths.max() / step_um)) + 1
    if n > 1:
        t = np.linspace(0.0, 1.0, n)
        points = samples[candidates][:, None, :] + t[None, :, None] * chords[candidates][:, None, :]
        flags = np.asarray(inside(points.reshape(-1, 3)), dtype=bool).reshape(len(candidates), n)
        background_um = (~flags).sum(axis=1) * lengths / (n - 1)
        same_lumen = background_um < margin_um
    else:
        same_lumen = np.ones(len(candidates), dtype=bool)
    neighbour[candidates[same_lumen]] = nearest[candidates[same_lumen]]
    return neighbour


def _shadowing_points(
    path_um,
    existing_tree,
    existing_points_um,
    inside,
    radius_at,
    *,
    step_um: float,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    max_cosine: float = MAX_SHADOW_COSINE,
) -> tuple[int, np.ndarray]:
    """``(samples judged, index into *existing_points_um* of each shadowed
    sample's nearest point)`` -- see :func:`path_shadows_existing_vessel`."""
    nothing = (0, np.empty(0, dtype=np.intp))
    existing = np.asarray(existing_points_um, dtype=float).reshape(-1, 3)
    if len(existing) == 0:
        return nothing
    samples, tangents, keep = _judged_samples(path_um, radius_at, step_um=step_um, margin_um=margin_um)
    samples, tangents = samples[keep], tangents[keep]
    if len(samples) == 0:
        return nothing
    neighbour = _same_lumen_neighbours(
        samples, tangents, existing_tree, existing, inside, radius_at,
        step_um=step_um, margin_um=margin_um, max_cosine=max_cosine, min_share=0.5,
    )
    return len(samples), neighbour[neighbour >= 0]


def shadowed_samples(
    path_um,
    existing_tree: cKDTree | None,
    existing_points_um: np.ndarray,
    inside: Callable[[np.ndarray], np.ndarray],
    radius_at: Callable[[np.ndarray], np.ndarray],
    *,
    step_um: float = 0.5,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    max_cosine: float = MAX_SHADOW_COSINE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(samples, judged, beside)``: *path_um* sampled every *step_um*, which
    samples are judged (away from both ends), and which run beside an existing
    centreline in the same lumen -- :func:`path_shadows_existing_vessel` sample
    by sample, without its verdict on the path as a whole.

    That verdict needs half the judged samples; this answers how much of a
    path is a second strand, so a doubled stretch of a long edge counts too.
    """
    existing = np.asarray(existing_points_um, dtype=float).reshape(-1, 3)
    samples, tangents, judged = _judged_samples(path_um, radius_at, step_um=step_um, margin_um=margin_um)
    beside = np.zeros(len(samples), dtype=bool)
    if len(existing) == 0 or not judged.any():
        return samples, judged, beside
    neighbour = _same_lumen_neighbours(
        samples[judged], tangents[judged], existing_tree, existing, inside, radius_at,
        step_um=step_um, margin_um=margin_um, max_cosine=max_cosine,
    )
    beside[np.flatnonzero(judged)] = neighbour >= 0
    return samples, judged, beside


class MaskSupport:
    """One segmentation mask, read the ways a bridge site asks about it.

    *mask* is voxel-indexed (canonical ``(z, y, x)``, memmaps included) and
    paths are physical microns on *voxel_size_zyx*. The lumen radius --
    each point's distance to background -- comes from the mask's surface
    (``pointwise_distance.FeatureDistance``), built the first time a radius
    is asked for; pass *feature_distance* to share one already built.
    """

    def __init__(
        self,
        mask: np.ndarray,
        voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        max_background_gap_um: float = DEFAULT_MAX_BACKGROUND_GAP_UM,
        min_mask_fraction: float = DEFAULT_MIN_MASK_FRACTION,
        feature_distance=None,
    ):
        self.mask = mask
        self.voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx)
        self.max_background_gap_um = float(max_background_gap_um)
        self.min_mask_fraction = float(min_mask_fraction)
        self._distance = feature_distance

    def inside(self, points_um: np.ndarray) -> np.ndarray:
        return _points_in_mask(points_um, self.mask, self.voxel_size_zyx)

    def radius(self, points_um: np.ndarray) -> np.ndarray:
        """Distance to background at each point's nearest voxel, in microns
        (0 outside the mask)."""
        if self._distance is None:
            from .pointwise_distance import LUMEN_RADII_REMEMBERED, FeatureDistance

            # Remembering: the same centreline voxels are asked about again
            # by every check a path or a clean-up round makes.
            self._distance = FeatureDistance(
                self.mask,
                feature_value=False,
                sampling=self.voxel_size_zyx,
                remember=LUMEN_RADII_REMEMBERED,
            )
        points = np.asarray(points_um, dtype=float).reshape(-1, 3)
        if not self._distance.has_surface:
            return np.zeros(len(points))
        index = np.rint(points / np.asarray(self.voxel_size_zyx)).astype(np.intp)
        index = np.clip(index, 0, np.asarray(self.mask.shape) - 1)
        return self._distance.at(index)

    def support(self, path_um) -> BridgeMaskSupport:
        return bridge_mask_support(path_um, voxel_size_zyx=self.voxel_size_zyx, inside=self.inside)

    def accepts(self, path_um) -> bool:
        return self.accepts_support(self.support(path_um))

    def accepts_support(self, support: BridgeMaskSupport) -> bool:
        return bridge_is_supported(
            support,
            max_background_gap_um=self.max_background_gap_um,
            min_mask_fraction=self.min_mask_fraction,
        )

    def shadows(self, path_um, existing_tree: cKDTree | None, existing_points_um) -> bool:
        """:func:`path_shadows_existing_vessel` against this mask."""
        return path_shadows_existing_vessel(
            path_um,
            existing_tree,
            existing_points_um,
            self.inside,
            self.radius,
            step_um=_sample_step(self.voxel_size_zyx),
        )
