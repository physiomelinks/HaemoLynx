"""H1 sections 1.3 and 1.5: morphometrics that need both channels at once.

Section 1.3 asks for the parenchymal volume of the TH-positive glomus clusters and the
centreline length density *within* those clusters. Section 1.5 asks for the distance from
every TH-positive voxel to the nearest lectin-positive centreline. Both are joins between the
two channels of one acquisition, which is only sound because they are two channels of one
acquisition: identical grid, co-registered by construction, no registration step involved.

The vessel side is the batch network itself (``examples/cb_h1_batch.py --stage run``), not a
mask recomputed here: its graph for the §1.3 length, its skeleton for the §1.5 distance and
its mask for the vessel volume. The TH channel is cropped to the same placed ROI, and the
driver refuses a batch output whose box differs (``roi_placement.check_output_roi``).

**One definition of vessel length** (open item 40). Until then this module thresholded the
probability map plainly at the frozen cut and skeletonised that, while the network uses the
hysteresis band plus its own cleanup, and it measured length by summing every 26-adjacent
voxel pair, which counts all three links where three skeleton voxels touch at a corner. The
two together put WKY-A's §1.3 length at 199 mm against the network's 97 mm. The length is
now the network's edge polylines, classified against TH by the same sampling H2 §2.1 uses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np

from ImageLynx.haemodynamics.tissue_regions import edge_length_inside_um


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
    graph,
    th_mask: np.ndarray,
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
    voxel_um: Sequence[float],
    th_threshold: float,
    vessel_threshold: float,
) -> ThMorphometry:
    """Assemble both sections from the batch network and masks on its grid.

    ``graph`` gives the §1.3 lengths, ``skeleton`` the §1.5 distances; both come from the
    same pipeline run, so they describe one vessel set.
    """
    th_mask = np.asarray(th_mask, dtype=bool)
    for name, volume in (("vessel_mask", vessel_mask), ("skeleton", skeleton)):
        if np.shape(volume) != th_mask.shape:
            raise ValueError(
                f"{name} has shape {np.shape(volume)}, th_mask has {th_mask.shape}")
    voxel_volume = float(np.prod(voxel_um))
    n = int(th_mask.size)

    length_all, length_in_th = edge_length_inside_um(graph, th_mask, voxel_um)
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
