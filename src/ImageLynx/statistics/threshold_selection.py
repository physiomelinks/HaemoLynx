"""Choose the segmentation threshold from calibre, constrained by fragmentation.

The segmentation handover selects the threshold from connected-component statistics: the
value just above where component count climbs steeply and the largest component's share
starts falling. Measured on the real WKY-C probability field, no mask-component statistic
discriminates. The largest component's share never falls - it is *higher* at 0.99, where the
network has visibly shattered into 7151 pieces, than at 0.70 - because share is counted in
voxels and this network is one dominant mass at every threshold, with fragments too small to
move a voxel fraction. Counting components above a 50-voxel floor is equally flat, wandering
between 94 and 139 across the whole range with no structure.

That is a property of the data's topology rather than of any one classifier, so it survives
retraining: a vascular bed percolates, and a percolating mask stays connected long after its
centreline has started beading.

Two measurements do discriminate.

**Median inscribed diameter** moves monotonically - 13.96 um at 0.20 down to 3.73 um at 0.99
on that volume - and it has an external target rather than an internal optimum: the
handover's own validation table expects a capillary mode of 4-7 um, and its half-voxel
arithmetic gives the same window independently. So it can be an objective without anything
being fitted.

**Skeleton endpoint density** is flat while the network is intact and climbs sharply once it
beads. On a mid-stack subvolume it ran 2.3, 2.8, 3.2, 2.9, 2.4, 2.1 per mm from 0.30 to 0.97
and then jumped to 4.8 at 0.99, where skeleton components went 172 -> 467 and mean component
length halved. Each new fragment contributes two endpoints, which is why this sees what the
mask cannot.

Calibre is therefore the objective and fragmentation the constraint - the reverse of the
handover's ordering, where topology chose the value and calibre was checked afterwards, if at
all. When the two do not agree anywhere, that is reported rather than resolved: a segmentation
with no threshold that is simultaneously the right calibre and intact is a segmentation
problem, and picking the least-bad value would turn it into a plausible set of numbers.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

#: The capillary diameter mode the handover's validation table expects, and the same window
#: its half-voxel error arithmetic implies independently. An external target, not a fitted one.
CAPILLARY_DIAMETER_RANGE_UM: Tuple[float, float] = (4.0, 7.0)

#: Multiple of the sweep's baseline endpoint density above which the centreline is treated as
#: fragmenting. The baseline is the median across the sweep rather than the minimum: the
#: minimum is a single noisy sample and using it flags ordinary variation as fragmentation.
#: The median assumes fewer than half the sampled thresholds are fragmenting, which is a
#: property of choosing a sensible sweep range rather than an assumption about the data.
FRAGMENTATION_TOLERANCE = 1.5

#: Components below this are debris rather than vessel - the size filter prob_to_mask.py
#: applies, at roughly 325 um^3. Recorded for continuity with that tool; not decisive here.
MIN_COMPONENT_VOXELS = 50


@dataclass(frozen=True)
class ThresholdSample:
    """Everything measured at one threshold, decisive or not."""

    threshold: float
    foreground_fraction: float
    median_diameter_um: float
    p90_diameter_um: float
    mask_components: int
    mask_components_above_floor: int
    largest_mask_component_share: float
    skeleton_length_mm: float
    endpoints: int
    endpoint_density_per_mm: float
    skeleton_components: int

    @property
    def calibre_ok(self) -> bool:
        lo, hi = CAPILLARY_DIAMETER_RANGE_UM
        return lo <= self.median_diameter_um <= hi


@dataclass(frozen=True)
class ThresholdSelection:
    """The chosen threshold, or an explanation of why there is not one."""

    threshold: Optional[float]
    reason: str
    calibre_window: Tuple[float, ...]
    fragmentation_onset: Optional[float]
    baseline_endpoint_density_per_mm: float
    samples: Tuple[ThresholdSample, ...]

    def require(self) -> float:
        """The threshold, or an error. For callers that cannot proceed without one."""
        if self.threshold is None:
            raise ValueError(f"No threshold satisfies both criteria: {self.reason}")
        return self.threshold

    def format_table(self) -> str:
        lo, hi = CAPILLARY_DIAMETER_RANGE_UM
        lines = [
            f"threshold sweep  (capillary window {lo}-{hi} um, "
            f"fragmentation above {FRAGMENTATION_TOLERANCE:.1f}x "
            f"{self.baseline_endpoint_density_per_mm:.2f} ep/mm)",
            f"{'thr':>6}{'fg':>8}{'d_med':>8}{'d_p90':>8}{'ep/mm':>8}"
            f"{'skelcmp':>9}{'maskcmp':>9}{'share':>8}  verdict",
        ]
        for sample in self.samples:
            marks = []
            if sample.calibre_ok:
                marks.append("calibre")
            if (self.fragmentation_onset is not None
                    and sample.threshold >= self.fragmentation_onset):
                marks.append("FRAGMENTING")
            if sample.threshold == self.threshold:
                marks.append("<- chosen")
            lines.append(
                f"{sample.threshold:>6.2f}{sample.foreground_fraction:>8.3f}"
                f"{sample.median_diameter_um:>8.2f}{sample.p90_diameter_um:>8.2f}"
                f"{sample.endpoint_density_per_mm:>8.2f}{sample.skeleton_components:>9d}"
                f"{sample.mask_components:>9d}{sample.largest_mask_component_share:>8.3f}"
                f"  {' '.join(marks)}"
            )
        lines.append(f"result: {self.reason}")
        return "\n".join(lines)


def evaluate_threshold(
    probabilities: np.ndarray,
    threshold: float,
    voxel_size_zyx: Sequence[float],
    *,
    measure_skeleton: bool = True,
) -> Optional[ThresholdSample]:
    """Measure calibre and both topologies at one threshold.

    Returns ``None`` when the mask is empty, which is a legitimate outcome at the top of a
    sweep rather than an error.

    Skeletonisation dominates the cost, so ``measure_skeleton=False`` is available for a
    calibre-only pass; the fragmentation constraint cannot be applied without it.
    """
    from scipy.ndimage import convolve, distance_transform_edt
    from skimage.measure import label
    from skimage.morphology import skeletonize

    spacing = tuple(float(v) for v in voxel_size_zyx)
    binary = np.asarray(probabilities) > float(threshold)
    if not binary.any():
        return None

    edt = distance_transform_edt(binary, sampling=spacing)
    radii = edt[binary]
    median_diameter = 2.0 * float(np.median(radii))
    p90_diameter = 2.0 * float(np.percentile(radii, 90))

    labelled = label(binary, connectivity=3)
    counts = np.bincount(labelled.ravel())[1:]
    share = float(counts.max() / counts.sum()) if counts.size else 0.0
    above_floor = int((counts >= MIN_COMPONENT_VOXELS).sum())

    length_mm = 0.0
    endpoints = 0
    density = 0.0
    skeleton_components = 0
    if measure_skeleton:
        skeleton = skeletonize(binary)
        n_skeleton = int(skeleton.sum())
        if n_skeleton:
            # One voxel step is the in-plane pitch; the volume is near-isotropic (axial to
            # lateral 1.0011) so a single figure is exact enough for a density denominator.
            length_mm = n_skeleton * spacing[1] / 1000.0
            neighbourhood = np.ones((3, 3, 3), dtype=np.uint8)
            neighbourhood[1, 1, 1] = 0
            degree = convolve(skeleton.astype(np.uint8), neighbourhood, mode="constant")
            endpoints = int(((degree == 1) & skeleton).sum())
            density = endpoints / length_mm if length_mm else 0.0
            skeleton_components = int(label(skeleton, connectivity=3).max())

    return ThresholdSample(
        threshold=float(threshold),
        foreground_fraction=float(binary.mean()),
        median_diameter_um=median_diameter,
        p90_diameter_um=p90_diameter,
        mask_components=int(counts.size),
        mask_components_above_floor=above_floor,
        largest_mask_component_share=share,
        skeleton_length_mm=length_mm,
        endpoints=endpoints,
        endpoint_density_per_mm=density,
        skeleton_components=skeleton_components,
    )


def sweep_thresholds(
    probabilities: np.ndarray,
    thresholds: Iterable[float],
    voxel_size_zyx: Sequence[float],
    *,
    measure_skeleton: bool = True,
) -> Tuple[ThresholdSample, ...]:
    """Evaluate a sorted sweep, dropping thresholds whose mask is empty."""
    samples = []
    for threshold in sorted(float(t) for t in thresholds):
        sample = evaluate_threshold(
            probabilities, threshold, voxel_size_zyx, measure_skeleton=measure_skeleton
        )
        if sample is not None:
            samples.append(sample)
    return tuple(samples)


def select_threshold(
    samples: Sequence[ThresholdSample],
    *,
    diameter_range: Tuple[float, float] = CAPILLARY_DIAMETER_RANGE_UM,
    fragmentation_tolerance: float = FRAGMENTATION_TOLERANCE,
) -> ThresholdSelection:
    """Highest intact threshold whose calibre is capillary-scale, or a refusal.

    Calibre chooses and fragmentation vetoes. Within the calibre window the highest surviving
    threshold is taken, because calibre falls monotonically with threshold and the risk being
    traded against is over-inclusion: the lower the threshold the fatter the vessel, and
    resistance carries that as r^-4.
    """
    if not samples:
        return ThresholdSelection(None, "No thresholds were evaluated.", (), None, 0.0, ())

    ordered = tuple(sorted(samples, key=lambda s: s.threshold))
    lo, hi = diameter_range

    densities = [s.endpoint_density_per_mm for s in ordered if s.endpoint_density_per_mm > 0]
    baseline = float(np.median(densities)) if densities else 0.0
    onset = None
    if baseline > 0:
        limit = fragmentation_tolerance * baseline
        breaking = [s.threshold for s in ordered if s.endpoint_density_per_mm > limit]
        onset = min(breaking) if breaking else None

    window = tuple(s.threshold for s in ordered if lo <= s.median_diameter_um <= hi)
    if not window:
        diameters = [s.median_diameter_um for s in ordered]
        return ThresholdSelection(
            None,
            f"No threshold reaches capillary calibre: median diameter spans "
            f"{min(diameters):.2f}-{max(diameters):.2f} um against a {lo}-{hi} um window. "
            f"That is a segmentation problem rather than a thresholding one.",
            window, onset, baseline, ordered,
        )

    intact = [t for t in window if onset is None or t < onset]
    if not intact:
        return ThresholdSelection(
            None,
            f"Every threshold at capillary calibre ({window[0]:.2f}-{window[-1]:.2f}) is at "
            f"or beyond the fragmentation onset {onset:.2f}, so the calibre can only be "
            f"reached by breaking the network. That is a segmentation problem rather than a "
            f"thresholding one.",
            window, onset, baseline, ordered,
        )

    chosen = max(intact)
    return ThresholdSelection(
        chosen,
        f"{chosen:.2f}: highest intact threshold inside the {lo}-{hi} um calibre window "
        f"({len(intact)} candidate(s); fragmentation onset "
        f"{'none observed' if onset is None else f'{onset:.2f}'}).",
        window, onset, baseline, ordered,
    )
