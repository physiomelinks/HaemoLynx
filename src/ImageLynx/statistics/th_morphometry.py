"""H1 sections 1.3 and 1.5: morphometrics that need both channels at once.

Section 1.3 asks for the parenchymal volume of the TH-positive glomus clusters and the
centreline length density *within* those clusters. Section 1.5 asks for the distance from
every TH-positive voxel to the nearest lectin-positive centreline. Both are joins between the
two channels of one acquisition, which is only sound because they are two channels of one
acquisition: identical grid, co-registered by construction, no registration step involved.

Both channels are cropped to the same region of interest, placed by ``roi_placement.place_roi``
from each specimen's own data. That is the same ROI and the same frozen vessel threshold that
``examples/cb_h1_batch.py`` uses, verified by reproducing the foreground fractions recorded in
its threshold_selection.json to five decimal places.

The vessel mask and skeleton are recomputed here rather than read from the
``*_ilastik_Probabilities_cache`` directories. Those were written by a different run against a
different crop: at the H1 ROI only about 29% of their voxels fall inside the box at all, so
joining TH to them would have measured the overlap of two unrelated regions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Every unique 26-connected step, taken once per pair rather than once per direction.
_STEPS: Tuple[Tuple[int, int, int], ...] = tuple(
    offset for offset in product((-1, 0, 1), repeat=3)
    if offset > (0, 0, 0)
)


def centreline_length_um(
    skeleton: np.ndarray,
    voxel_um: Sequence[float],
    within: Optional[np.ndarray] = None,
) -> float:
    """Total centreline length, summing the real length of every step.

    Counting skeleton voxels and multiplying by the voxel size is the obvious estimator and
    it is wrong by up to sqrt(3): a diagonal step covers 3.23 um on this grid where an axial
    one covers 1.87. On a tortuous network that is not a small correction, and H1 section 1.4
    turns on tortuosity, so the two must not disagree about what length means.

    With ``within``, a step counts only when *both* of its endpoints are inside the mask. A
    step straddling the boundary belongs to neither side, and assigning it to the tissue it
    half touches would inflate whichever mask is more fragmented.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    if within is not None:
        within = np.asarray(within, dtype=bool)
        if within.shape != skeleton.shape:
            raise ValueError(
                f"within has shape {within.shape}, skeleton has {skeleton.shape}")

    total = 0.0
    for step in _STEPS:
        length = float(np.sqrt(sum((s * v) ** 2 for s, v in zip(step, voxel_um))))
        a, b = _shifted_pair(skeleton, step)
        joined = a & b
        if within is not None:
            wa, wb = _shifted_pair(within, step)
            joined &= wa & wb
        total += length * int(joined.sum())
    return total


def _shifted_pair(volume: np.ndarray, step) -> Tuple[np.ndarray, np.ndarray]:
    """The overlapping views of ``volume`` offset against itself by ``step``."""
    lo = tuple(slice(max(s, 0), volume.shape[i] + min(s, 0))
               for i, s in enumerate(step))
    hi = tuple(slice(max(-s, 0), volume.shape[i] + min(-s, 0))
               for i, s in enumerate(step))
    return volume[lo], volume[hi]


def tissue_to_vessel_distance_um(
    tissue: np.ndarray,
    skeleton: np.ndarray,
    voxel_um: Sequence[float],
) -> np.ndarray:
    """Distance from every tissue voxel to the nearest centreline voxel, in micrometres.

    To the centreline rather than to the vessel surface, which is what H1 section 1.5 asks
    for. The two differ by the local radius, so they are not interchangeable: on a 3 um
    capillary the surface is 1.5 um closer everywhere, and that offset would be absorbed
    into any group difference rather than showing up as one.

    ``sampling`` puts the result in micrometres directly and carries the 1.0011 axial to
    lateral ratio, so nothing downstream needs a second conversion.
    """
    from scipy import ndimage as ndi

    tissue = np.asarray(tissue, dtype=bool)
    skeleton = np.asarray(skeleton, dtype=bool)
    if tissue.shape != skeleton.shape:
        raise ValueError(f"shapes disagree: tissue {tissue.shape}, skeleton {skeleton.shape}")
    if not skeleton.any():
        raise ValueError(
            "no centreline in this volume, so there is no distance to measure. A distance "
            "transform against an empty mask returns infinity everywhere, which would "
            "propagate as a very large tissue-to-vessel distance rather than as an error."
        )
    if not tissue.any():
        return np.empty(0, dtype=np.float32)

    distance = ndi.distance_transform_edt(~skeleton, sampling=tuple(voxel_um))
    return distance[tissue].astype(np.float32)


@dataclass(frozen=True)
class ThMorphometry:
    """One specimen's section 1.3 and 1.5 results, at one pair of thresholds."""

    specimen_id: str
    group: str
    roi_voxels: int
    th_threshold: float
    vessel_threshold: float

    # Section 1.3
    th_volume_um3: float
    th_volume_fraction: float
    vessel_volume_um3: float
    vessel_volume_fraction: float
    centreline_length_um: float
    centreline_length_within_th_um: float
    length_density_mm_per_mm3: float

    # Section 1.5
    tvd_n: int
    tvd_median_um: float
    tvd_p25_um: float
    tvd_p75_um: float
    tvd_p90_um: float
    tvd_mean_um: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def summarise(
    specimen_id: str,
    group: str,
    th_mask: np.ndarray,
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
    voxel_um: Sequence[float],
    th_threshold: float,
    vessel_threshold: float,
) -> ThMorphometry:
    """Assemble both sections from masks that are already on a common grid."""
    voxel_volume = float(np.prod(voxel_um))
    n = int(th_mask.size)

    length_all = centreline_length_um(skeleton, voxel_um)
    length_in_th = centreline_length_um(skeleton, voxel_um, within=th_mask)
    th_volume = float(th_mask.sum()) * voxel_volume

    # Length per unit parenchymal volume, expressed as mm of centreline per mm3 of TH+
    # tissue. um/um3 is the same quantity, but the numbers are 1e6 apart and the mm form is
    # what the vascular morphometry literature reports.
    density = (length_in_th / th_volume) * 1e6 if th_volume > 0 else float("nan")

    if skeleton.any() and th_mask.any():
        tvd = tissue_to_vessel_distance_um(th_mask, skeleton, voxel_um)
    else:
        tvd = np.empty(0, dtype=np.float32)

    def q(percentile: float) -> float:
        return float(np.percentile(tvd, percentile)) if tvd.size else float("nan")

    return ThMorphometry(
        specimen_id=specimen_id,
        group=group,
        roi_voxels=n,
        th_threshold=float(th_threshold),
        vessel_threshold=float(vessel_threshold),
        th_volume_um3=th_volume,
        th_volume_fraction=float(th_mask.mean()),
        vessel_volume_um3=float(vessel_mask.sum()) * voxel_volume,
        vessel_volume_fraction=float(vessel_mask.mean()),
        centreline_length_um=length_all,
        centreline_length_within_th_um=length_in_th,
        length_density_mm_per_mm3=density,
        tvd_n=int(tvd.size),
        tvd_median_um=q(50),
        tvd_p25_um=q(25),
        tvd_p75_um=q(75),
        tvd_p90_um=q(90),
        tvd_mean_um=float(tvd.mean()) if tvd.size else float("nan"),
    )
