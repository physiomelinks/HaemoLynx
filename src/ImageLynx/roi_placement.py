"""Where to put the sub-volume in each specimen.

A matched ROI size makes the samples the same *size*; it does not make them the same
*anatomy*. The carotid body does not sit in the middle of its imaged block, and it does not
sit in the same place in every block: preprocess_cb.py recorded each volume's axial tissue
peak. The stacks differ in depth - 435 slices for WKY, 495 for SHR - so the peak is compared
as a fraction of depth, and it ranges from 0.244 (WKY-B) to 0.529 (WKY-A). A centred ROI
therefore lands mid-organ in one specimen and in its sparse margin in another, and the
resulting difference in vessel density is a difference in where the box was put.

The misplacement is also group-correlated, but weakly: WKY means 0.402 against SHR's 0.371,
a gap of 0.031 sitting inside a within-WKY spread of 0.285. Read that as a reason not to
assume centring is neutral, not as a measured cohort effect.

Placement here is computed from the data rather than chosen by hand:

- **z** from the axial tissue peak in the volume's own QC record. preprocess_cb.py derived
  it as the argmax of a per-slice 99th-percentile brightness profile, smoothed along z by a
  moving average of max(3, n // 20) slices.
- **y and x** from the centroid of the grayscale channel's z-projection, thresholded at its
  99th percentile. The projection covers only the slices the box occupies (open item 13),
  not the whole stack. See tissue_centroid_yx.

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

    The weighting is not always inert. preprocess_cb.py clips the top 0.02% of voxels to
    1.0 and the projection takes a max over z, so over the *whole* stack 1.33-1.52% of the
    projection is saturated, the 99th percentile lands exactly on 1.0, and every survivor
    weighs the same. place_roi now projects only the ROI's own 160 slices (open item 13),
    where fewer columns saturate and the cutoff can fall below 1.0 (0.899 in WKY-A), so the
    intensity weighting takes effect there.
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

    # Clamp z before reading: the lateral centroid is measured over the slices the box will
    # actually occupy (open item 13). Projecting the whole stack let tissue that never enters
    # the box vote on where it goes laterally, which moved the centre 7-45 um.
    centre_z = clamp_centre((centre_z, 0, 0), (size_zyx[0], 1, 1), (shape[0], 1, 1))[0]
    z0 = centre_z - size_zyx[0] // 2
    z1 = z0 + size_zyx[0]

    centre_y, centre_x = shape[1] // 2, shape[2] // 2
    path = specimen.ilastik_input_path
    if path.exists():
        try:
            sz, sy, sx = subsample
            with h5py.File(path, "r") as handle:
                # Channel 0 is the background-subtracted grayscale; the vesselness channels
                # are derived from it and would weight the centroid towards whichever scale
                # the filter happened to favour.
                #
                # The stride is a memory decision, not a speed one. Chunks are
                # (32, 128, 128, 3) and gzip-compressed, so a z-stride of 4 still lands in
                # every chunk and each is decompressed whole. Measured on WKY-A: strided
                # 4.88 s for 25 MB, full-resolution 4.58 s for 402 MB. It costs 6.9 um of
                # centroid accuracy to hold 16x less in RAM.
                block = np.asarray(handle["data"][z0:z1:sz, ::sy, ::sx, 0], dtype=np.float32)
            cy, cx = tissue_centroid_yx(block)
            centre_y, centre_x = cy * sy, cx * sx
            sources.append(f"yx=grayscale_centroid over z {z0}-{z1}")
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


#: Sidecar the network pipeline writes next to its outputs, naming the box it cropped. Every
#: driver that reads a batch graph or ``per_edge_morphometry.csv`` checks it against
#: ``place_roi`` before using the output (open item 27).
ROI_RECORD_NAME = "roi_placement.json"


def placement_fell_back(placement: RoiPlacement) -> bool:
    """True when any axis of the placement came from the volume centre, not the data."""
    return "volume_centre" in placement.source


def centred_placement(specimen, size_zyx: Sequence[int]) -> RoiPlacement:
    """The array-centre box, as ``crop_roi`` cuts it with zero offsets (``extent // 2``)."""
    size_zyx = tuple(int(v) for v in size_zyx)
    shape = specimen.shape_zyx
    centre = clamp_centre(tuple(e // 2 for e in shape), size_zyx, shape)
    return RoiPlacement(
        specimen_id=specimen.specimen_id,
        centre_zyx=centre,
        size_zyx=size_zyx,
        offsets_zyx=centre_to_offsets(centre, shape),
        peak_slice=None,
        source="array_centre (requested)",
    )


def roi_record(placement: RoiPlacement, shape_zyx, *, centred: bool) -> dict:
    """What the sidecar holds: enough to rebuild the box and to see how it was chosen."""
    return {
        "specimen_id": placement.specimen_id,
        "volume_shape_zyx": [int(v) for v in shape_zyx],
        "centre_zyx": [int(v) for v in placement.centre_zyx],
        "size_zyx": [int(v) for v in placement.size_zyx],
        "bounds_zyx": [[int(s.start), int(s.stop)] for s in placement.bounds],
        "offsets_zyx": [float(v) for v in placement.offsets_zyx],
        "peak_slice": placement.peak_slice,
        "source": placement.source,
        "centred": bool(centred),
    }


def write_roi_record(directory, record: dict):
    """Write the sidecar into ``directory`` and return its path."""
    import json
    from pathlib import Path

    path = Path(directory) / ROI_RECORD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def check_output_roi(directory, specimen, size_zyx: Sequence[int],
                     placement: Optional[RoiPlacement] = None) -> dict:
    """Refuse a pipeline output that was not cropped at this specimen's placed ROI.

    Raises when the sidecar is missing (outputs from before open item 27, all centre-cropped),
    when the run asked for the array centre, or when its box differs from what ``place_roi``
    gives now (the placement rule has changed since the run). Returns the record otherwise.
    """
    import json
    from pathlib import Path

    path = Path(directory) / ROI_RECORD_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{specimen.specimen_id}: no {ROI_RECORD_NAME} in {directory}. Pipeline outputs "
            f"without it predate open item 27 and were cropped at the array centre, not the "
            f"placed ROI. Re-run cb_h1_batch.py --stage run."
        )
    record = json.loads(path.read_text())
    if record.get("specimen_id") != specimen.specimen_id:
        raise ValueError(
            f"{path} is for {record.get('specimen_id')}, not {specimen.specimen_id}.")
    if record.get("centred"):
        raise ValueError(
            f"{specimen.specimen_id}: the output in {directory} was cropped at the array "
            f"centre (--roi-centred), not at the placed ROI.")
    if placement is None:
        placement = place_roi(specimen, size_zyx)
    expected = [[int(s.start), int(s.stop)] for s in placement.bounds]
    if record.get("bounds_zyx") != expected:
        raise ValueError(
            f"{specimen.specimen_id}: the output in {directory} was cropped at "
            f"{record.get('bounds_zyx')}, but place_roi now gives {expected}. The placement "
            f"has changed since that run; re-run cb_h1_batch.py --stage run."
        )
    return record


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
