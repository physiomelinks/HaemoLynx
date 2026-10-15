"""A synthetic dense capillary bed: what mask-supported bridging and recovery
are for.

A lattice of 5 um capillaries 22 um apart in one plane, at the voxel size of
an E14.5 two-photon stack (0.98 x 0.98 x 1.0 um), with what goes wrong in
one:

- a 12 um vessel, wide enough that Lee thinning leaves side strands where
  capillaries meet it;
- two parallel 4 um vessels 8 um apart, background between them;
- two blind-ended sprouts whose tips face each other across 4.5 um of
  background, 8.5 um apart -- inside any reconnect threshold;
- two one-voxel dropouts in the segmentation, each across a vessel;
- one capillary whose skeleton has been deleted.

Geometry is in physical ``(z, y, x)`` microns; radii vary by a seeded
``+-0.2 um`` per vessel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize

VOXEL_SIZE_ZYX = (1.0, 0.98, 0.98)
SHAPE = (21, 117, 117)
PLANE_Z_UM = 10.0
#: 22 um apart, bar the last cell: 26 um, room for the two sprouts.
LINES_UM = (11.0, 33.0, 55.0, 77.0, 103.0)
CAPILLARY_RADIUS_UM = 2.5
WIDE_ROW_UM = 55.0
WIDE_HALF_WIDTH_UM = 6.0
#: ``(start, end)`` of each vessel beyond the lattice, ``(y, x)`` um: the two
#: parallel vessels, then the two sprouts (start on a row, end blind).
PAIR_YX_UM = (((18.0, 55.0), (18.0, 77.0)), ((26.0, 55.0), (26.0, 77.0)))
SPROUTS_YX_UM = (((77.0, 20.0), (87.0, 20.0)), ((103.0, 26.0), (93.0, 26.0)))
#: Their radius, before the per-vessel jitter.
PAIR_AND_SPROUT_RADIUS_UM = 2.0
#: The two one-voxel dropouts: the row they cut and where along it, then
#: the column and where along it.
ROW_DROPOUT_UM = (11.0, 44.0)
COLUMN_DROPOUT_UM = (33.0, 66.0)
SEED = 20261004


@dataclass(frozen=True)
class DenseCapillaryBed:
    mask: np.ndarray
    skeleton: np.ndarray
    voxel_size_zyx: tuple[float, float, float]
    #: Where each sprout's centreline ends, ``(z, y, x)`` um.
    sprout_tips_um: np.ndarray
    #: ``(start, end)`` centreline of each parallel vessel.
    parallel_pair_um: tuple[tuple[np.ndarray, np.ndarray], ...]
    #: ``(start, end)`` of the capillary whose skeleton was deleted.
    lost_capillary_um: tuple[np.ndarray, np.ndarray]
    extent_um: np.ndarray


def _grid_um() -> np.ndarray:
    return np.indices(SHAPE).transpose(1, 2, 3, 0) * np.asarray(VOXEL_SIZE_ZYX)


def _capsule(points: np.ndarray, start, end, radius: float) -> np.ndarray:
    """Points within *radius* of the segment start-end."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    t = np.clip(((points - start) @ axis) / float(axis @ axis), 0.0, 1.0)
    offset = points - (start + t[..., None] * axis)
    return np.sum(offset**2, axis=-1) <= radius**2


def _at(y: float, x: float) -> np.ndarray:
    return np.array([PLANE_Z_UM, y, x])


def dense_capillary_bed(seed: int = SEED) -> DenseCapillaryBed:
    rng = np.random.default_rng(seed)
    points = _grid_um()
    extent = (np.asarray(SHAPE) - 1) * np.asarray(VOXEL_SIZE_ZYX)
    far = float(extent[1])

    def jitter(radius: float) -> float:
        return radius + float(rng.uniform(-0.2, 0.2))

    mask = np.zeros(SHAPE, dtype=bool)
    for line in LINES_UM:
        radius = WIDE_HALF_WIDTH_UM if line == WIDE_ROW_UM else CAPILLARY_RADIUS_UM
        mask |= _capsule(points, _at(line, 0.0), _at(line, far), jitter(radius))
        mask |= _capsule(points, _at(0.0, line), _at(far, line), jitter(CAPILLARY_RADIUS_UM))

    pair = tuple((_at(*start), _at(*end)) for start, end in PAIR_YX_UM)
    for start, end in pair:
        mask |= _capsule(points, start, end, jitter(PAIR_AND_SPROUT_RADIUS_UM))

    sprouts = tuple((_at(*start), _at(*end)) for start, end in SPROUTS_YX_UM)
    for start, end in sprouts:
        mask |= _capsule(points, start, end, jitter(PAIR_AND_SPROUT_RADIUS_UM))

    spacing = np.asarray(VOXEL_SIZE_ZYX)
    row, along_row = ROW_DROPOUT_UM
    row_dropout = int(round(along_row / spacing[2]))
    near_row = np.abs(np.arange(SHAPE[1]) * spacing[1] - row) <= 4.0
    mask[:, near_row, row_dropout] = False
    column, along_column = COLUMN_DROPOUT_UM
    column_dropout = int(round(along_column / spacing[1]))
    near_column = np.abs(np.arange(SHAPE[2]) * spacing[2] - column) <= 4.0
    mask[:, column_dropout, near_column] = False

    skeleton = skeletonize(mask, method="lee").astype(bool)
    column = LINES_UM[-1]
    lost = (
        _at(WIDE_ROW_UM + WIDE_HALF_WIDTH_UM + 2.0, column),
        _at(77.0 - CAPILLARY_RADIUS_UM - 1.0, column),
    )
    ys = np.arange(SHAPE[1]) * spacing[1]
    xs = np.arange(SHAPE[2]) * spacing[2]
    in_gap = (ys >= lost[0][1]) & (ys <= lost[1][1])
    beside = np.abs(xs - column) <= 4.0
    skeleton[:, in_gap[:, None] & beside[None, :]] = False

    return DenseCapillaryBed(
        mask=mask,
        skeleton=skeleton,
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        sprout_tips_um=np.array([end for _start, end in sprouts]),
        parallel_pair_um=pair,
        lost_capillary_um=lost,
        extent_um=extent,
    )


def centreline_samples(start: np.ndarray, end: np.ndarray, step_um: float = 0.5) -> np.ndarray:
    """Points along the segment start-end, *step_um* apart."""
    count = max(2, int(np.ceil(np.linalg.norm(end - start) / step_um)) + 1)
    return start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)


@dataclass(frozen=True)
class DesignSegment:
    """One vessel between two of the bed's junctions or ends, as designed."""

    start_yx_um: tuple[float, float]
    end_yx_um: tuple[float, float]
    #: Nominal: twice the radius before the per-vessel jitter (+-0.4 um).
    diameter_um: float

    @property
    def length_um(self) -> float:
        return float(np.hypot(self.end_yx_um[0] - self.start_yx_um[0], self.end_yx_um[1] - self.start_yx_um[1]))


def design_segments(*, as_segmented: bool = False) -> list[DesignSegment]:
    """The bed's network as designed, every centreline straight in the plane
    ``z = PLANE_Z_UM``: each vessel split at the junctions on it, the lattice
    lines running face to face. *as_segmented* also cuts the two vessels the
    dropouts break, leaving out the voxel of each that is gone -- what a tool
    reading the mask literally should find."""
    far = float((SHAPE[1] - 1) * VOXEL_SIZE_ZYX[1])
    vessels = []
    for line in LINES_UM:
        row_diameter = 2 * (WIDE_HALF_WIDTH_UM if line == WIDE_ROW_UM else CAPILLARY_RADIUS_UM)
        vessels.append(((line, 0.0), (line, far), row_diameter))
        vessels.append(((0.0, line), (far, line), 2 * CAPILLARY_RADIUS_UM))
    for start, end in PAIR_YX_UM + SPROUTS_YX_UM:
        vessels.append((start, end, 2 * PAIR_AND_SPROUT_RADIUS_UM))

    def runs_along_x(start, end) -> bool:
        return start[0] == end[0]

    def on(point, start, end) -> bool:
        (y0, x0), (y1, x1) = start, end
        return min(y0, y1) <= point[0] <= max(y0, y1) and min(x0, x1) <= point[1] <= max(x0, x1)

    gaps = []
    if as_segmented:
        half = VOXEL_SIZE_ZYX[2] / 2
        row, along = ROW_DROPOUT_UM
        gaps.append(((row, along), half))
        column, along = COLUMN_DROPOUT_UM
        gaps.append(((along, column), half))

    segments = []
    for start, end, diameter in vessels:
        axis = 1 if runs_along_x(start, end) else 0
        cuts = {start[axis], end[axis]}
        for other_start, other_end, _d in vessels:
            if runs_along_x(other_start, other_end) == runs_along_x(start, end):
                continue
            crossing = (start[0], other_start[1]) if axis == 1 else (other_start[0], start[1])
            if on(crossing, start, end) and on(crossing, other_start, other_end):
                cuts.add(crossing[axis])
        stops = sorted(cuts)
        pieces = list(zip(stops[:-1], stops[1:]))
        for centre, half in gaps:
            if on(centre, start, end):
                pieces = [
                    piece
                    for a, b in pieces
                    for piece in (((a, centre[axis] - half), (centre[axis] + half, b))
                                  if a < centre[axis] < b else ((a, b),))
                ]
        for a, b in pieces:
            first, second = list(start), list(start)
            first[axis], second[axis] = a, b
            segments.append(DesignSegment(tuple(first), tuple(second), diameter))
    return segments


def design_topology(segments: list[DesignSegment]) -> dict[str, int]:
    """Branch points (degree >= 3) and end points (degree 1) of *segments*."""
    from collections import Counter

    degree = Counter(point for s in segments for point in (s.start_yx_um, s.end_yx_um))
    return {
        "branch_points": sum(1 for d in degree.values() if d >= 3),
        "end_points": sum(1 for d in degree.values() if d == 1),
    }
