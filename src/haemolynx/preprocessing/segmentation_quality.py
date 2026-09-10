"""Score a raw segmented mask on how well it will skeletonise and graph.

A quick, pre-skeletonisation health check for the "Check segmented image"
button on the Input tab: five independent 0-2 sub-scores that sum to a 0-10
total, each aimed at a specific failure mode this pipeline is known to be
sensitive to, so a user can tell *why* a score is low, not just that it is:

- ``fragmentation``: how many small, disconnected specks sit alongside the
  real vessel network (segmentation noise that ``remove_small_segmented_
  volumes`` exists to clean up).
- ``connectivity``: how much of the total foreground volume is one single
  connected piece, as opposed to several large pieces that should be one
  network (what ``reconnect_vessel_like_components`` exists to repair).
- ``boundary_vessels``: how many separate places the mask touches the
  volume's own edge -- each one becomes an open end after skeletonisation,
  and a haemodynamic solve wants a small, known number of inlets/outlets,
  not dozens.
- ``noise``: how much the mask's own shape changes under a light surface
  smoothing (:func:`haemolynx.preprocessing.smooth_vessel_surfaces`) --
  a jagged, poorly-resolved vessel changes a lot; a clean one barely moves.
- ``resolution``: the typical vessel radius (from the mask's own distance
  transform) measured in units of the coarsest sampled axis -- a vessel
  only one or two voxels across is undersampled, whatever its physical
  size.

Pure numpy/scipy/skimage over an already-boolean array and a physical
``voxel_size_zyx``, matching the rest of this package (see
:mod:`haemolynx.preprocessing.segmentation_cleanup`'s own module docstring
for why): nothing here imports ``haemolynx.io``, ``haemolynx.graph`` or
``haemolynx.gui``, so the GUI's background-thread wiring is the only place
that ever has to know where the mask came from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .segmentation_cleanup import smooth_vessel_surfaces

__all__ = [
    "SegmentationQualityScore",
    "score_segmented_mask",
    "format_segmentation_quality_report",
]

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

#: A component smaller than this fraction of the largest component's own
#: size counts as a "small fragment" for the fragmentation score -- relative,
#: not an absolute voxel count, so the same score means the same thing on a
#: small crop and a whole-brain volume.
DEFAULT_SMALL_FRAGMENT_FRACTION = 0.02

#: Physical Gaussian sigma for the noise/resolution check -- matches
#: `smooth_vessel_surfaces`'s own default, a light touch meant to smooth
#: single-voxel surface roughness, not reshape a genuinely thin vessel.
DEFAULT_NOISE_SIGMA_UM = 1.0

#: A vessel this many voxels across its own radius (so roughly twice that
#: across its diameter), measured against the coarsest sampled axis, earns
#: full marks for the resolution score; less scales down proportionally.
DEFAULT_TARGET_VOXELS_ACROSS_RADIUS = 3.0

#: Up to this many boundary-touching patches costs nothing (a normal
#: single-inlet/single-outlet network already has two); the boundary-vessel
#: score decays past it.
DEFAULT_BOUNDARY_PATCH_ALLOWANCE = 2
DEFAULT_BOUNDARY_PATCH_DECAY_SCALE = 5.0

#: How many small fragments it takes to roughly halve the fragmentation
#: score (a smooth hyperbolic decay, never reaching zero at any finite count).
DEFAULT_FRAGMENT_COUNT_DECAY_SCALE = 5.0


@dataclass(frozen=True)
class SegmentationQualityScore:
    """A 0-10 segmented-mask quality score, broken into five 0-2 parts."""

    fragmentation: float
    connectivity: float
    boundary_vessels: float
    noise: float
    resolution: float
    #: Raw measurements behind each sub-score, for the human-readable report.
    small_fragment_count: int
    component_count: int
    largest_fraction: float
    boundary_patch_count: int
    surface_iou: float
    median_radius_um: float
    coarsest_voxel_um: float

    @property
    def total(self) -> float:
        return (
            self.fragmentation
            + self.connectivity
            + self.boundary_vessels
            + self.noise
            + self.resolution
        )


def _boundary_patch_count(mask: np.ndarray) -> int:
    """Distinct connected patches where *mask* touches any of its 6 faces.

    Each face is labelled on its own 2D slice (26-connectivity in 3D has no
    meaning on a single-voxel-thick face), so a vessel touching two opposite
    faces counts twice -- it opens the volume in two separate places, which
    is exactly what this is meant to count.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got {mask.ndim}D")
    count = 0
    for axis in range(3):
        for index in (0, -1):
            face = np.take(mask, index, axis=axis)
            if not face.any():
                continue
            _labeled, n = label(face, structure=_STRUCTURE_8)
            count += int(n)
    return count


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def score_segmented_mask(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    small_fragment_fraction: float = DEFAULT_SMALL_FRAGMENT_FRACTION,
    noise_sigma_um: float = DEFAULT_NOISE_SIGMA_UM,
    target_voxels_across_radius: float = DEFAULT_TARGET_VOXELS_ACROSS_RADIUS,
    boundary_patch_allowance: int = DEFAULT_BOUNDARY_PATCH_ALLOWANCE,
    boundary_patch_decay_scale: float = DEFAULT_BOUNDARY_PATCH_DECAY_SCALE,
    fragment_count_decay_scale: float = DEFAULT_FRAGMENT_COUNT_DECAY_SCALE,
) -> SegmentationQualityScore:
    """Score a raw segmented mask, 0-10, before it is ever skeletonised.

    An empty mask scores 0 on every axis -- there is nothing to measure, not
    a passing grade. Every other degenerate case (a single voxel, a mask
    that already touches every face) falls out of the same formulas below
    without a special case: a lone component has no fragments and is fully
    connected; a mask with no boundary faces scores full marks on that axis
    because ``boundary_patch_count`` is 0, at or under the free allowance.
    """
    mask = np.asarray(mask, dtype=bool)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    total_voxels = int(mask.sum())
    if total_voxels == 0:
        return SegmentationQualityScore(
            fragmentation=0.0,
            connectivity=0.0,
            boundary_vessels=0.0,
            noise=0.0,
            resolution=0.0,
            small_fragment_count=0,
            component_count=0,
            largest_fraction=0.0,
            boundary_patch_count=0,
            surface_iou=0.0,
            median_radius_um=0.0,
            coarsest_voxel_um=float(max(sampling)) if sampling else 0.0,
        )

    labeled, count = label(mask, structure=_STRUCTURE_26)
    sizes = np.bincount(labeled.ravel())[1:]
    largest = int(sizes.max())
    largest_fraction = float(largest) / float(total_voxels)

    small_threshold = max(1.0, float(small_fragment_fraction) * largest)
    small_fragment_count = int(np.sum(sizes < small_threshold))
    fragmentation = 2.0 / (1.0 + small_fragment_count / max(1e-9, fragment_count_decay_scale))

    connectivity = 2.0 * largest_fraction

    boundary_patches = _boundary_patch_count(mask)
    boundary_excess = max(0, boundary_patches - int(boundary_patch_allowance))
    boundary_vessels = 2.0 * float(
        np.exp(-boundary_excess / max(1e-9, boundary_patch_decay_scale))
    )

    smoothed = smooth_vessel_surfaces(mask, voxel_size_zyx=sampling, sigma_um=noise_sigma_um)
    surface_iou = _iou(mask, smoothed)
    noise = 2.0 * surface_iou

    edt = distance_transform_edt(mask, sampling=sampling)
    median_radius_um = float(np.median(edt[mask]))
    coarsest_voxel_um = float(max(sampling))
    voxels_across_radius = median_radius_um / max(1e-9, coarsest_voxel_um)
    resolution = 2.0 * min(1.0, voxels_across_radius / max(1e-9, target_voxels_across_radius))

    return SegmentationQualityScore(
        fragmentation=fragmentation,
        connectivity=connectivity,
        boundary_vessels=boundary_vessels,
        noise=noise,
        resolution=resolution,
        small_fragment_count=small_fragment_count,
        component_count=int(count),
        largest_fraction=largest_fraction,
        boundary_patch_count=boundary_patches,
        surface_iou=surface_iou,
        median_radius_um=median_radius_um,
        coarsest_voxel_um=coarsest_voxel_um,
    )


def format_segmentation_quality_report(score: SegmentationQualityScore) -> str:
    """A compact multiline report for the log window, mirroring this
    codebase's other ``format_*_report`` functions."""
    return (
        f"Segmented image quality: {score.total:.1f}/10\n"
        f"  fragmentation:    {score.fragmentation:.1f}/2  "
        f"({score.small_fragment_count} small fragment(s) of "
        f"{score.component_count} component(s))\n"
        f"  connectivity:     {score.connectivity:.1f}/2  "
        f"(largest component holds {score.largest_fraction * 100:.0f}% of the volume)\n"
        f"  boundary vessels: {score.boundary_vessels:.1f}/2  "
        f"({score.boundary_patch_count} place(s) the mask touches the image edge)\n"
        f"  noise:            {score.noise:.1f}/2  "
        f"(surface unchanged by light smoothing: {score.surface_iou * 100:.0f}%)\n"
        f"  resolution:       {score.resolution:.1f}/2  "
        f"(typical vessel radius {score.median_radius_um:.2f}um vs "
        f"{score.coarsest_voxel_um:.2f}um sampling)"
    )
