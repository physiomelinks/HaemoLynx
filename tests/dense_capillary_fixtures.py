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

    pair = ((_at(18.0, 55.0), _at(18.0, 77.0)), (_at(26.0, 55.0), _at(26.0, 77.0)))
    for start, end in pair:
        mask |= _capsule(points, start, end, jitter(2.0))

    sprouts = ((_at(77.0, 20.0), _at(87.0, 20.0)), (_at(103.0, 26.0), _at(93.0, 26.0)))
    for start, end in sprouts:
        mask |= _capsule(points, start, end, jitter(2.0))

    spacing = np.asarray(VOXEL_SIZE_ZYX)
    row_dropout = int(round(44.0 / spacing[2]))
    near_row = np.abs(np.arange(SHAPE[1]) * spacing[1] - 11.0) <= 4.0
    mask[:, near_row, row_dropout] = False
    column_dropout = int(round(66.0 / spacing[1]))
    near_column = np.abs(np.arange(SHAPE[2]) * spacing[2] - 33.0) <= 4.0
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
