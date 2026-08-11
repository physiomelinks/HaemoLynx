"""Where to put the sub-volume in each specimen.

A matched ROI size makes the samples the same *size*; it does not make them the same
*anatomy*. The carotid body does not sit in the middle of its imaged block, and it does not
sit in the same place in every block: preprocess_cb.py recorded each volume's axial tissue
peak, and they range from slice 106 of 435 in WKY-B to 230 of 435 in WKY-A. A centred ROI
therefore lands mid-organ in one specimen and in its sparse margin in another, and the
resulting difference in vessel density is a difference in where the box was put.

Worse, the misplacement is not random with respect to the comparison: WKY peaks at a mean
depth fraction of 0.40 and SHR at 0.34, so a centred box systematically samples a different
part of the organ in each cohort.

Placement here is computed from the data rather than chosen by hand:

- **z** from the axial tissue peak in the volume's own QC record, which preprocess_cb.py
  derived from tissue extent per slice.
- **y and x** from the intensity-weighted centroid of the grayscale channel.

**The trade this makes.** Centring on signal samples the middle of the organ, which is
denser than its periphery, so the absolute densities reported are not representative of the
whole carotid body. What it buys is that the same rule is applied to all six, so the
*comparison* is like-for-like even though the absolute level is not. Given H1 is a
between-group claim rather than an absolute one, that is the right way round - but it has to
be stated, because an absolute vessel density quoted from these ROIs would be an
overestimate.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RoiPlacement:
    """Where one specimen's ROI sits, and what put it there."""

    specimen_id: str
    centre_zyx: Tuple[int, int, int]
    size_zyx: Tuple[int, int, int]
    offsets_zyx: Tuple[float, float, float]
    peak_slice: Optional[int]
    source: str

    @property
    def bounds(self) -> Tuple[slice, slice, slice]:
        return tuple(
            slice(c - s // 2, c - s // 2 + s)
            for c, s in zip(self.centre_zyx, self.size_zyx)
        )


def clamp_centre(centre_zyx, size_zyx, shape_zyx) -> Tuple[int, int, int]:
    """Pull the centre inwards until the box fits wholly inside the volume.

    A box hanging over the edge would be silently truncated, making the sample smaller than
    its neighbours' - the very thing a matched size exists to prevent.
    """
    clamped = []
    for centre, size, extent in zip(centre_zyx, size_zyx, shape_zyx):
        half = size // 2
        if size >= extent:
            clamped.append(extent // 2)
        else:
            clamped.append(int(np.clip(centre, half, extent - (size - half))))
    return tuple(clamped)


def centre_to_offsets(centre_zyx, shape_zyx) -> Tuple[float, float, float]:
    """crop_roi takes offsets from the volume centre as a fraction of each dimension."""
    return tuple(
        float((centre - extent / 2.0) / extent)
        for centre, extent in zip(centre_zyx, shape_zyx)
    )


def tissue_centroid_yx(volume: np.ndarray, percentile: float = 99.0) -> Tuple[int, int]:
    """In-plane centre of mass of the brightest tissue.

    Thresholded at a high percentile before weighting: the mean of a background-subtracted
    volume is dominated by the many near-zero voxels, which drags the centroid towards the
    geometric middle and defeats the point of measuring it.
    """
    data = np.asarray(volume, dtype=np.float32)
    projected = data.max(axis=0) if data.ndim == 3 else data
    cutoff = np.percentile(projected, percentile)
    mask = projected >= cutoff
    if not mask.any():
        return tuple(int(s // 2) for s in projected.shape)
    ys, xs = np.nonzero(mask)
    weights = projected[mask].astype(np.float64)
    return (int(round(np.average(ys, weights=weights))),
            int(round(np.average(xs, weights=weights))))


def place_roi(
    specimen,
    size_zyx: Sequence[int],
    subsample: Tuple[int, int, int] = (4, 2, 2),
) -> RoiPlacement:
    """Compute this specimen's ROI placement from its own data.

    Falls back to the volume centre, and says so in ``source``, when neither the QC record
    nor the preprocessed volume is reachable - a silent fallback to centred placement would
    reintroduce exactly the bias this function exists to remove.
    """
    import h5py

    size_zyx = tuple(int(v) for v in size_zyx)
    shape = specimen.shape_zyx
    sources = []

    record = specimen.qc_record()
    peak = None
    if record:
        peak = (record.get("z_profile") or {}).get("peak_slice")
    if peak is not None:
        centre_z = int(peak)
        sources.append("z=qc_peak_slice")
    else:
        centre_z = shape[0] // 2
        sources.append("z=volume_centre")

    centre_y, centre_x = shape[1] // 2, shape[2] // 2
    path = specimen.ilastik_input_path
    if path.exists():
        try:
            sz, sy, sx = subsample
            with h5py.File(path, "r") as handle:
                # Channel 0 is the background-subtracted grayscale; the vesselness channels
                # are derived from it and would weight the centroid towards whichever scale
                # the filter happened to favour.
                block = np.asarray(handle["data"][::sz, ::sy, ::sx, 0], dtype=np.float32)
            cy, cx = tissue_centroid_yx(block)
            centre_y, centre_x = cy * sy, cx * sx
            sources.append("yx=grayscale_centroid")
        except Exception:
            sources.append("yx=volume_centre (unreadable)")
    else:
        sources.append("yx=volume_centre (absent)")

    centre = clamp_centre((centre_z, centre_y, centre_x), size_zyx, shape)
    return RoiPlacement(
        specimen_id=specimen.specimen_id,
        centre_zyx=centre,
        size_zyx=size_zyx,
        offsets_zyx=centre_to_offsets(centre, shape),
        peak_slice=peak,
        source=", ".join(sources),
    )


def format_placement_table(placements: Sequence[RoiPlacement], specimens=None) -> str:
    """What was sampled from where, for the record."""
    lines = [
        f"{'spec':<7}{'volume zyx':>18}{'ROI centre zyx':>18}{'offsets zyx':>26}"
        f"{'peak z':>8}",
    ]
    for placement in placements:
        shape = next((s.shape_zyx for s in (specimens or [])
                      if s.specimen_id == placement.specimen_id), None)
        offsets = ", ".join(f"{o:+.3f}" for o in placement.offsets_zyx)
        lines.append(
            f"{placement.specimen_id:<7}{str(shape or '-'):>18}"
            f"{str(placement.centre_zyx):>18}{offsets:>26}"
            f"{str(placement.peak_slice or '-'):>8}"
        )
    return "\n".join(lines)
