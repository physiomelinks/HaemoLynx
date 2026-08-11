"""Where the labels sit relative to the boundary the classifier has to place.

A pixel classifier learns a decision boundary, so it needs evidence where that boundary is.
Labelling vessel cores and far-field emptiness teaches it the two things it was never going
to get wrong, and leaves it to interpolate across the zone that actually decides the
segmentation.

That is what happened here. Tripling WKY-C's labels - 4488 to 13504 voxels, background up
3.9x, depths 2 to 4 - left the prediction unchanged to four decimal places and made the
uncertain band slightly worse. Measured against the resulting prediction:

    WKY-C                  n        p10    p25  median    p75    p90   in 2-10um band
    vessel labels       8217       0.00   0.00    0.00   0.00   2.64             6.0%
    background labels   5287       5.60  11.04   26.95  45.16  64.37            19.8%
    uncertain voxels   17.5M       1.87   3.73    8.13  16.05  29.21            41.5%

Half the volume sits at about 0.5, concentrated 2-10 um from confident vessel, and almost no
label occupies that band. Nothing told the classifier where a vessel ends, which is also why
the masks come out near twice capillary calibre at every usable threshold.

Measured from the labels alone, the same gap was in all six volumes: 1.0% to 8.6% of
background labels within 9.33 um of a vessel label.

**This measure deliberately uses no prediction.** Distance to the nearest *labelled* vessel
overstates distance to the nearest real one, since most vessels are unlabelled - on WKY-C the
label-only figure was 8.6% where the prediction-based figure was 19.8%, roughly a factor of
two. The trade is worth it: this runs in about two seconds on the .ilp alone, so it can be
checked between saves while labelling, where a prediction-based version would cost a headless
run per iteration. Read it as a comparative and directional measure, not an absolute one, and
see MIN_BOUNDARY_FRACTION for how the target accounts for the offset.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

#: The band, in micrometres from a labelled vessel, where the decision boundary lives. The
#: lower edge is one voxel - closer than that overlaps the vessel wall itself and is not
#: boundary evidence - and the upper edge is five voxels, beyond which the prediction is
#: already confidently background.
BOUNDARY_BAND_UM: Tuple[float, float] = (1.87, 9.33)

#: Target fraction of background labels inside the band.
#:
#: On the prediction-based measure the uncertainty sits 41.5% inside it, which is what the
#: labelling has to cover. This label-only measure reads roughly half that on the same data
#: (8.6% against 19.8% on WKY-C), so 20% here corresponds to something near 40% there. The
#: calibration comes from one specimen and is approximate; overshooting costs nothing, since
#: boundary labels are the ones that carry information.
MIN_BOUNDARY_FRACTION = 0.20


@dataclass(frozen=True)
class LabelPlacement:
    """Where one specimen's background labels sit relative to its vessel labels."""

    specimen_id: str
    group: str
    vessel_labels: int
    background_labels: int
    background_p25_um: float
    background_median_um: float
    background_p75_um: float
    background_within_band_fraction: float
    slices_measured: int
    slices_skipped: int

    @property
    def ok(self) -> bool:
        return self.background_within_band_fraction >= MIN_BOUNDARY_FRACTION


def _read_label_slices(classifier_path: Path) -> Dict[str, Dict[int, List[Tuple]]]:
    """Per lane, per z, the label blocks and where they sit."""
    import h5py

    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(classifier_path, "r") as project:
        infos = project["Input Data/infos"]
        lanes = []
        for key in sorted(infos.keys()):
            path = None
            for role in infos[key].keys():
                if "filePath" in infos[key][role]:
                    path = text(infos[key][role]["filePath"][()])
                    break
            lanes.append(path or "")

        label_sets = project["PixelClassification/LabelSets"]
        by_lane: Dict[str, Dict[int, List[Tuple]]] = {}
        for position, key in enumerate(sorted(label_sets.keys())):
            blocks: Dict[int, List[Tuple]] = {}

            def visit(_name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.size == 0:
                    return
                extent = text(obj.attrs["blockSlice"]).strip("[]").split(",")
                (z0, _), (y0, y1), (x0, x1), _ = [
                    tuple(int(v) for v in part.split(":")) for part in extent
                ]
                block = obj[()]
                plane = block[0, ..., 0] if block.ndim == 4 else np.squeeze(block)
                blocks.setdefault(z0, []).append((y0, y1, x0, x1, plane))

            label_sets[key].visititems(visit)
            if position < len(lanes):
                by_lane[lanes[position]] = blocks
    return by_lane


def analyse_label_placement(
    classifier_path: Optional[Path] = None,
    voxel_size_zyx: Optional[Tuple[float, float, float]] = None,
) -> Tuple[LabelPlacement, ...]:
    """Distance from every background label to the nearest vessel label, per specimen.

    Measured within each labelled z-slice, since labels are painted slice by slice and a
    3D distance across unlabelled slices would measure the labelling pattern rather than the
    geometry. Slices carrying only one class are skipped and counted: distance is undefined
    with nothing to measure from.
    """
    from scipy.ndimage import distance_transform_edt

    from ..specimens import POOLED_CLASSIFIER, PROCESSING_VOXEL_UM, SPECIMENS

    classifier_path = Path(classifier_path or POOLED_CLASSIFIER)
    spacing = tuple(voxel_size_zyx or PROCESSING_VOXEL_UM)[1:]
    by_lane = _read_label_slices(classifier_path)
    lo, hi = BOUNDARY_BAND_UM

    rows: List[LabelPlacement] = []
    for specimen in SPECIMENS:
        blocks = next(
            (b for path, b in by_lane.items() if specimen.preproc_stem in path), {}
        )
        distances: List[np.ndarray] = []
        vessel_total = background_total = measured = skipped = 0

        for _z, items in sorted(blocks.items()):
            height = max(i[1] for i in items)
            width = max(i[3] for i in items)
            vessel = np.zeros((height, width), dtype=bool)
            background = np.zeros((height, width), dtype=bool)
            for y0, y1, x0, x1, plane in items:
                vessel[y0:y1, x0:x1] |= plane == 1
                background[y0:y1, x0:x1] |= plane == 2

            vessel_total += int(vessel.sum())
            background_total += int(background.sum())
            if not vessel.any() or not background.any():
                skipped += 1
                continue
            measured += 1
            distances.append(distance_transform_edt(~vessel, sampling=spacing)[background])

        if distances:
            pooled = np.concatenate(distances)
            p25, median, p75 = (float(v) for v in np.percentile(pooled, [25, 50, 75]))
            in_band = float(((pooled >= lo) & (pooled <= hi)).mean())
        else:
            p25 = median = p75 = float("nan")
            in_band = 0.0

        rows.append(LabelPlacement(
            specimen_id=specimen.specimen_id,
            group=specimen.group,
            vessel_labels=vessel_total,
            background_labels=background_total,
            background_p25_um=p25,
            background_median_um=median,
            background_p75_um=p75,
            background_within_band_fraction=in_band,
            slices_measured=measured,
            slices_skipped=skipped,
        ))
    return tuple(rows)


def format_placement_table(rows: Tuple[LabelPlacement, ...]) -> str:
    """A table to read between saves while labelling."""
    lo, hi = BOUNDARY_BAND_UM
    lines = [
        f"background label placement  (target: >= {MIN_BOUNDARY_FRACTION:.0%} within "
        f"{lo:.2f}-{hi:.2f} um of a vessel label)",
        f"{'spec':<7}{'grp':<5}{'vessel':>8}{'backgrd':>9}{'p25':>8}{'median':>8}"
        f"{'p75':>8}{'in band':>9}  ",
    ]
    for row in rows:
        lines.append(
            f"{row.specimen_id:<7}{row.group:<5}{row.vessel_labels:>8}"
            f"{row.background_labels:>9}{row.background_p25_um:>8.1f}"
            f"{row.background_median_um:>8.1f}{row.background_p75_um:>8.1f}"
            f"{row.background_within_band_fraction:>9.1%}  "
            f"{'ok' if row.ok else '<- needs boundary labels'}"
        )
    failing = [r.specimen_id for r in rows if not r.ok]
    lines.append(
        "all specimens have boundary evidence" if not failing
        else "needs background painted 1-5 voxels outside the vessel wall: "
             + ", ".join(failing)
    )
    return "\n".join(lines)
