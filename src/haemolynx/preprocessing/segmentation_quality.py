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

The report this drives (:func:`format_segmentation_quality_report`) leads
with a verdict grouping these five into two tiers, not just the blended
0-10 total: ``resolution`` is a *source-data* problem (a physical sampling
limit fixed at acquisition time -- fixable only by resegmenting from a
less-downsampled source or re-imaging at higher resolution), while the
other four are *pipeline-fixable* (this package's own
``segmentation_cleanup_*`` steps routinely address them). A good
fragmentation/connectivity score must never be allowed to bury a bad
resolution score inside one number -- that distinction is the entire point
of this module: pushing a user toward resegmenting/re-imaging when that is
genuinely the only fix, rather than toward tuning cleanup settings that
cannot help.

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
    "DEFAULT_TARGET_VOXELS_ACROSS_RADIUS",
    "DEFAULT_BOUNDARY_PATCH_ALLOWANCE",
    "RESOLUTION_LIMITING_THRESHOLD",
    "PIPELINE_FIXABLE_ISSUE_THRESHOLD",
]

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

#: A component smaller than this fraction of the largest component's own
#: size counts as a "small fragment" for the fragmentation score -- relative,
#: not an absolute voxel count, so the same score means the same thing on a
#: small crop and a whole-brain volume. Reasoned, not empirically fitted: 2%
#: is small enough that a genuine bilateral second vessel tree (routinely
#: >20% of the largest) is never caught by this alone.
DEFAULT_SMALL_FRAGMENT_FRACTION = 0.02

#: Physical Gaussian sigma for the noise/resolution check -- matches
#: `smooth_vessel_surfaces`'s own default, a light touch meant to smooth
#: single-voxel surface roughness, not reshape a genuinely thin vessel.
#: Reasoned, not empirically fitted.
DEFAULT_NOISE_SIGMA_UM = 1.0

#: A vessel this many voxels across its own radius (so roughly twice that
#: across its diameter), measured against the coarsest sampled axis, earns
#: full marks for the resolution score; less scales down proportionally.
#: Reasoned (not empirically fitted) against this codebase's own diameter
#: measurements: both the FWHM Gaussian fit
#: (`haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`) and
#: the EDT inscribed-radius estimate (`haemodynamics.edt_diameter`) need
#: several samples across a vessel's profile to be trustworthy -- under
#: about 3 voxels of radius, both techniques are known to carry a
#: increasingly large, currently-uncorrectable discretisation bias.
#: Overridable per dataset via the `min_voxels_across_vessel_radius`
#: pipeline setting (`None` here means "use this default").
DEFAULT_TARGET_VOXELS_ACROSS_RADIUS = 3.0

#: Up to this many boundary-touching patches costs nothing (a normal
#: single-inlet/single-outlet network already has two); the boundary-vessel
#: score decays past it. Reasoned, not empirically fitted -- a whole-organ
#: scan or a multi-inlet preparation legitimately has more than two, which
#: is exactly why it is overridable per dataset via the
#: `expected_boundary_vessel_count` pipeline setting (`None` here means
#: "use this default").
DEFAULT_BOUNDARY_PATCH_ALLOWANCE = 2
#: How quickly the boundary-vessel score falls off past the allowance
#: (exponential decay rate). Reasoned, not empirically fitted.
DEFAULT_BOUNDARY_PATCH_DECAY_SCALE = 5.0

#: How many small fragments it takes to roughly halve the fragmentation
#: score (a smooth hyperbolic decay, never reaching zero at any finite
#: count). Reasoned, not empirically fitted.
DEFAULT_FRAGMENT_COUNT_DECAY_SCALE = 5.0

#: Below this fraction of the resolution sub-score's own 0-2 range, the
#: report calls resolution out as the limiting factor up front: half marks
#: means the vessel is sampled at under half of `target_voxels_across_radius`
#: -- reasoned, not empirically fitted, as the point past which no
#: downstream processing (cleanup, skeletonisation, or a better fit) can be
#: expected to compensate for the missing samples.
RESOLUTION_LIMITING_THRESHOLD = 1.0

#: Below this fraction of a pipeline-fixable sub-score's own 0-2 range, it
#: is counted in the report's "N pipeline-fixable issue(s)" tally. Reasoned,
#: not empirically fitted -- a three-quarter-marks bar.
PIPELINE_FIXABLE_ISSUE_THRESHOLD = 1.5


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

    @property
    def source_data_score(self) -> float:
        """0-2: sub-scores no amount of this pipeline's own processing can
        improve. Only `resolution` -- a physical voxel-sampling limit fixed
        at acquisition time. A low score here means "resegment from a
        less-downsampled source, or re-image at higher resolution", never
        "try a cleanup setting"."""
        return self.resolution

    @property
    def pipeline_fixable_score(self) -> float:
        """0-8: sub-scores this pipeline's own `segmentation_cleanup_*`
        settings routinely address (removing small fragments, reconnecting
        or closing gaps, de-whiskering/smoothing the surface, occasionally
        reducing boundary touches by reconnecting a vessel cut at a seam)."""
        return self.fragmentation + self.connectivity + self.boundary_vessels + self.noise


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
    target_voxels_across_radius: float | None = None,
    boundary_patch_allowance: int | None = None,
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

    *target_voxels_across_radius* and *boundary_patch_allowance* default to
    ``None``, meaning "use this module's own reasoned default" -- the same
    ``None``-means-default convention the pipeline settings that can
    override them (``min_voxels_across_vessel_radius``,
    ``expected_boundary_vessel_count``) already use elsewhere in this
    codebase for a dataset-specific value with no universal answer.
    """
    mask = np.asarray(mask, dtype=bool)
    sampling = tuple(float(v) for v in voxel_size_zyx)
    target_voxels_across_radius = (
        DEFAULT_TARGET_VOXELS_ACROSS_RADIUS
        if target_voxels_across_radius is None
        else float(target_voxels_across_radius)
    )
    boundary_patch_allowance = (
        DEFAULT_BOUNDARY_PATCH_ALLOWANCE
        if boundary_patch_allowance is None
        else int(boundary_patch_allowance)
    )
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


def _verdict_line(score: SegmentationQualityScore) -> str:
    """The report's leading, unmissable line: whether the *source* data
    itself is the problem (resegment/re-image -- nothing downstream fixes
    it) or whether what remains is pipeline-fixable (try cleanup settings).

    A good `fragmentation`/`connectivity` score must never bury a bad
    `resolution` score inside one blended total -- that is exactly the
    failure mode this function exists to prevent (see
    RESOLUTION_LIMITING_THRESHOLD's own docstring for the reasoning).
    """
    if score.resolution < RESOLUTION_LIMITING_THRESHOLD:
        voxels_across_radius = score.median_radius_um / max(1e-9, score.coarsest_voxel_um)
        return (
            f"⚠ Resolution is the limiting factor "
            f"({voxels_across_radius:.1f} voxels across a typical vessel radius) -- "
            f"no amount of cleanup fixes this. Re-image at higher "
            f"resolution/magnification, or resegment from a less-downsampled source."
        )
    fixable_issues = sum(
        1
        for sub_score in (
            score.fragmentation,
            score.connectivity,
            score.boundary_vessels,
            score.noise,
        )
        if sub_score < PIPELINE_FIXABLE_ISSUE_THRESHOLD
    )
    if fixable_issues == 0:
        return "✓ Source resolution looks adequate, and no pipeline-fixable issues stood out."
    plural = "s" if fixable_issues != 1 else ""
    return (
        f"✓ Source resolution looks adequate. {fixable_issues} pipeline-fixable "
        f"issue{plural} below -- try the segmentation cleanup settings."
    )


def format_segmentation_quality_report(
    score: SegmentationQualityScore,
    *,
    after_cleanup: SegmentationQualityScore | None = None,
) -> str:
    """A compact multiline report for the log window, mirroring this
    codebase's other ``format_*_report`` functions.

    Leads with :func:`_verdict_line` -- the one sentence that should
    actually drive a "resegment"/"re-image" decision -- ahead of the
    five-line breakdown, which always describes *score* (the raw,
    as-loaded mask). When *after_cleanup* is given (the same mask scored
    again after applying the run's current ``segmentation_cleanup_*``
    settings), the headline also reports it alongside the raw total, with
    an explicit caveat: a rescued-by-cleanup number is not evidence the
    original segmentation or imaging was adequate, which is the one thing
    this whole report exists to help a user judge for themselves.
    """
    if after_cleanup is not None:
        headline = (
            f"Raw segmentation: {score.total:.1f}/10  |  "
            f"After your current cleanup settings: {after_cleanup.total:.1f}/10 "
            f"(does not mean your source data improved)"
        )
    else:
        headline = f"Segmented image quality: {score.total:.1f}/10"
    return (
        f"{_verdict_line(score)}\n"
        f"{headline}\n"
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
