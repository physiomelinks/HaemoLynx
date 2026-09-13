"""Cross-check a segmented mask against its own raw (unsegmented) image.

Every check in :mod:`haemolynx.preprocessing.segmentation_quality` measures
the mask's own geometry -- self-referential, and blind to whether the
segmentation itself is *correct*, only whether it is internally consistent.
This module is the one check that looks outside the mask: given the raw
intensity volume the mask was (presumably) derived from, it finds

- **added voxels**: mask foreground with no support in the raw signal
  (over-segmentation -- background noise, or a classifier false positive),
- **removed voxels**: raw signal not captured by the mask (under-
  segmentation -- real foreground the classifier missed), and
- **missed structures**: whole connected components of raw-implied
  foreground with almost no overlap with the mask at all -- a genuine
  structural failure (an entire vessel the classifier never saw), distinct
  from the voxel-level counts above, which a single frayed edge on an
  otherwise-detected vessel would also produce.

The raw image is independently thresholded with Otsu's method
(:func:`skimage.filters.threshold_otsu`) -- a standard, parameter-free,
literature-established global threshold, not a tunable this module invents.
It is a coarse reference, not a second segmentation: real microscopy data is
not always bimodal, so this cannot replace visual inspection, only flag
gross disagreement worth a second look. A low agreement score can also mean
the two images are not perfectly spatially aligned with each other (a
different crop, or one resampled/registered and the other not), or simply
that the real segmentation legitimately uses more than raw intensity (ilastik
pixel classification reads texture and local context, not a single global
cut) -- :func:`format_segmentation_raw_comparison_report` says so explicitly
whenever agreement is low, rather than pointing only at "resegment."

``missed_structure_min_volume_um3`` matters more than it looks: an early
version of this module floored the "is this raw-implied component a real
structure" test at 5 voxels, and running it against a real, noisy dataset
(one raw-foreground voxel in the low tens of thousands, most of it single-
or few-voxel background-noise flecks from Otsu thresholding real
fluorescence) reported over 30,000 "missed structures" -- a physical volume
floor, big enough to exclude noise-scale flecks but well under any real
capillary cross-section, is what keeps this count meaningful on real data
instead of being dominated by noise.

Deliberately not part of :mod:`segmentation_quality`'s 0-10 score: that
score is what :mod:`haemolynx.optimisation`'s settings search directly
maximises, and this module needs a raw reference image that is only
sometimes available (see the "Check segmented image" GUI wiring) --
entangling an optional, external-data-dependent check into the always-
computable mask-only score would make the score's own scale depend on
whether a raw file happened to be supplied. This is reported as an
independent, additive section instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu

__all__ = [
    "RawImageForeground",
    "SegmentationRawComparison",
    "analyze_raw_image_foreground",
    "compare_segmentation_to_raw_image",
    "format_segmentation_raw_comparison_report",
]

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)

#: A raw-implied component with at most this fraction of its own volume
#: overlapping the segmented mask counts as "missed entirely" -- reasoned
#: (not empirically fitted) as the point past which this is a genuine
#: structural omission, not just an imprecise edge on a vessel the voxel-
#: level added/removed counts already describe.
DEFAULT_MISSED_STRUCTURE_MAX_OVERLAP_FRACTION = 0.1

#: Ignore a raw-implied component smaller than this physical volume when
#: looking for missed structures -- a physical size, not a raw voxel count,
#: so the same floor means the same thing at any resolution. Reasoned (not
#: empirically fitted) to sit well under a real capillary cross-section but
#: comfortably above single-/few-voxel Otsu-threshold noise on real,
#: imperfectly-bimodal fluorescence data -- see the module docstring for the
#: real-data test that found 5 *voxels* (this module's first, too-permissive
#: attempt) let tens of thousands of noise flecks through.
DEFAULT_MISSED_STRUCTURE_MIN_VOLUME_UM3 = 20.0

#: Below this overall intersection-over-union, the report adds a caveat
#: that low agreement is not proof the segmentation itself is wrong -- see
#: the module docstring. Reasoned, not empirically fitted.
DEFAULT_LOW_AGREEMENT_IOU_WARNING_THRESHOLD = 0.5


@dataclass(frozen=True)
class SegmentationRawComparison:
    """The result of comparing a segmented mask to its own raw image."""

    #: The Otsu threshold used to binarise the raw image.
    raw_threshold: float
    raw_foreground_voxel_count: int
    #: How much of *mask*'s own foreground has no support in the raw signal.
    added_voxel_count: int
    added_fraction: float
    #: How much of the raw-implied foreground the mask fails to cover.
    removed_voxel_count: int
    removed_fraction: float
    #: Intersection-over-union between the mask and the raw-implied
    #: foreground -- one overall agreement number.
    agreement_iou: float
    #: Whole raw-implied structures with (almost) no overlap with the mask.
    missed_structure_count: int
    missed_structure_total_voxels: int
    missed_structure_total_volume_um3: float


@dataclass(frozen=True)
class RawImageForeground:
    """The mask-independent half of :func:`compare_segmentation_to_raw_image`:
    the raw image's own Otsu threshold, its implied foreground, and that
    foreground's connected-component labelling.

    None of this depends on the segmented mask being compared against it --
    only on *raw_image* itself (and the *voxel_size_zyx*/
    *missed_structure_min_volume_um3* a caller keeps fixed across repeated
    comparisons). A caller checking many candidate masks against the same
    raw image -- a settings search sweeping segmentation-cleanup settings,
    one candidate mask per trial -- can compute this once with
    :func:`analyze_raw_image_foreground` and pass it to every
    :func:`compare_segmentation_to_raw_image` call via *precomputed*,
    instead of redoing Otsu thresholding and a full connected-components
    labelling pass over the (unchanging) raw image for every candidate.
    """

    threshold: float
    raw_foreground: np.ndarray
    raw_total: int
    labeled: np.ndarray
    sizes: np.ndarray
    missed_structure_min_voxels: int


def analyze_raw_image_foreground(
    raw_image: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    missed_structure_min_volume_um3: float = DEFAULT_MISSED_STRUCTURE_MIN_VOLUME_UM3,
) -> Optional[RawImageForeground]:
    """Precompute :class:`RawImageForeground` for *raw_image*, to reuse
    across many :func:`compare_segmentation_to_raw_image` calls.

    Returns ``None`` for a raw image with no intensity contrast at all
    (uniformly one value, or every voxel non-finite) -- passing that back
    in as *precomputed* reproduces exactly the same degenerate-case
    handling :func:`compare_segmentation_to_raw_image` gives without it.
    """
    raw_image = np.asarray(raw_image)
    voxel_volume_um3 = (
        float(voxel_size_zyx[0]) * float(voxel_size_zyx[1]) * float(voxel_size_zyx[2])
    )
    finite = raw_image[np.isfinite(raw_image)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return None

    threshold = float(threshold_otsu(finite))
    raw_foreground = raw_image > threshold
    raw_total = int(raw_foreground.sum())
    missed_structure_min_voxels = max(
        1, int(round(missed_structure_min_volume_um3 / max(1e-9, voxel_volume_um3)))
    )
    if raw_total:
        labeled, n_components = label(raw_foreground, structure=_STRUCTURE_26)
    else:
        labeled, n_components = np.zeros(raw_foreground.shape, dtype=np.int32), 0
    sizes = np.bincount(labeled.ravel()) if n_components else np.zeros(1, dtype=np.int64)
    return RawImageForeground(
        threshold=threshold,
        raw_foreground=raw_foreground,
        raw_total=raw_total,
        labeled=labeled,
        sizes=sizes,
        missed_structure_min_voxels=missed_structure_min_voxels,
    )


def compare_segmentation_to_raw_image(
    mask: np.ndarray,
    raw_image: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    missed_structure_max_overlap_fraction: float = DEFAULT_MISSED_STRUCTURE_MAX_OVERLAP_FRACTION,
    missed_structure_min_volume_um3: float = DEFAULT_MISSED_STRUCTURE_MIN_VOLUME_UM3,
    precomputed: Optional[RawImageForeground] = None,
) -> SegmentationRawComparison:
    """Compare *mask* against the raw intensity volume it was derived from.

    *mask* and *raw_image* must share shape -- both describe the same
    physical volume, sampled on the same grid; this never resamples,
    matching this codebase's own established convention for a second image
    that must align with a reference one (e.g.
    :func:`haemolynx.io.load_and_validate_vessel_masks`).

    A raw image with no intensity contrast at all (uniformly one value, or
    every voxel non-finite) has no Otsu threshold to compute; that
    degenerate case reports every mask voxel as "added" rather than raising
    -- there is genuinely nothing in the raw data to support any of it.

    *precomputed*, when given, is used instead of re-deriving
    :class:`RawImageForeground` from *raw_image* -- see
    :func:`analyze_raw_image_foreground`, which is what a caller uses to
    build it. The caller is responsible for it actually matching
    *raw_image*/*voxel_size_zyx*/*missed_structure_min_volume_um3*; nothing
    here re-validates that.
    """
    mask = np.asarray(mask, dtype=bool)
    raw_image = np.asarray(raw_image)
    if raw_image.shape != mask.shape:
        raise ValueError(
            "raw_image shape does not match the segmented mask shape: "
            f"{raw_image.shape} != {mask.shape}"
        )

    voxel_volume_um3 = (
        float(voxel_size_zyx[0]) * float(voxel_size_zyx[1]) * float(voxel_size_zyx[2])
    )
    mask_total = int(mask.sum())

    foreground = precomputed if precomputed is not None else analyze_raw_image_foreground(
        raw_image, voxel_size_zyx=voxel_size_zyx,
        missed_structure_min_volume_um3=missed_structure_min_volume_um3,
    )
    if foreground is None:
        return SegmentationRawComparison(
            raw_threshold=0.0,
            raw_foreground_voxel_count=0,
            added_voxel_count=mask_total,
            added_fraction=1.0 if mask_total else 0.0,
            removed_voxel_count=0,
            removed_fraction=0.0,
            agreement_iou=0.0,
            missed_structure_count=0,
            missed_structure_total_voxels=0,
            missed_structure_total_volume_um3=0.0,
        )

    raw_foreground = foreground.raw_foreground
    raw_total = foreground.raw_total

    added = mask & ~raw_foreground
    removed = raw_foreground & ~mask
    added_count = int(added.sum())
    removed_count = int(removed.sum())

    union = int((mask | raw_foreground).sum())
    intersection = int((mask & raw_foreground).sum())
    agreement_iou = float(intersection) / union if union else 1.0

    missed_count = 0
    missed_voxels = 0
    if raw_total:
        sizes = foreground.sizes
        overlap_counts = np.bincount(foreground.labeled[mask].ravel(), minlength=sizes.size)
        for component_id in range(1, sizes.size):
            size = int(sizes[component_id])
            if size < foreground.missed_structure_min_voxels:
                continue
            overlap_fraction = float(overlap_counts[component_id]) / size
            if overlap_fraction <= missed_structure_max_overlap_fraction:
                missed_count += 1
                missed_voxels += size

    return SegmentationRawComparison(
        raw_threshold=foreground.threshold,
        raw_foreground_voxel_count=raw_total,
        added_voxel_count=added_count,
        added_fraction=float(added_count) / mask_total if mask_total else 0.0,
        removed_voxel_count=removed_count,
        removed_fraction=float(removed_count) / raw_total if raw_total else 0.0,
        agreement_iou=agreement_iou,
        missed_structure_count=missed_count,
        missed_structure_total_voxels=missed_voxels,
        missed_structure_total_volume_um3=missed_voxels * voxel_volume_um3,
    )


def format_segmentation_raw_comparison_report(
    comparison: SegmentationRawComparison,
    *,
    low_agreement_warning_threshold: float = DEFAULT_LOW_AGREEMENT_IOU_WARNING_THRESHOLD,
) -> str:
    """A compact multiline report section, mirroring this codebase's other
    ``format_*_report`` functions -- meant to be appended after
    :func:`haemolynx.preprocessing.segmentation_quality.format_segmentation_quality_report`'s
    own output, not to replace it."""
    lines = [
        f"Raw-image cross-check (Otsu threshold {comparison.raw_threshold:.3g}):",
        f"  agreement (IoU):    {comparison.agreement_iou * 100:.0f}%",
        f"  added voxels:       {comparison.added_voxel_count} "
        f"({comparison.added_fraction * 100:.1f}% of the segmented volume has no raw support)",
        f"  removed voxels:     {comparison.removed_voxel_count} "
        f"({comparison.removed_fraction * 100:.1f}% of the raw signal is missing from the segmentation)",
        f"  missed structures:  {comparison.missed_structure_count} whole structure(s) "
        f"({comparison.missed_structure_total_volume_um3:.1f}um3 total) present in the raw "
        "image but not captured by the segmentation at all",
    ]
    if comparison.agreement_iou < low_agreement_warning_threshold:
        lines.append(
            "  note: low agreement is not proof the segmentation itself is wrong -- it can "
            "also mean the raw and segmented images are not perfectly spatially aligned, or "
            "that the segmentation legitimately used more than raw intensity to decide"
        )
    return "\n".join(lines)
