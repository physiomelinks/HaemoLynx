#!/usr/bin/env python3
"""H1 sections 1.3 and 1.5 across the cohort, from both channels at the H1 ROI.

    python3 examples/cb_h1_th_metrics.py                  # WKY only, the defensible half
    python3 examples/cb_h1_th_metrics.py --all            # both groups, SHR caveated
    python3 examples/cb_h1_th_metrics.py --all --th-threshold 0.5 0.7 0.9

Section 1.3 is the parenchymal volume of the TH-positive clusters and the centreline length
density within them. Section 1.5 is the distance from every TH-positive voxel to the nearest
lectin-positive centreline.

Both channels are cropped to the same ROI, placed by ``place_roi`` from each specimen's own
data, and the vessel channel is thresholded at the frozen 0.9 that cb_h1_batch selected. That
combination was verified against the foreground fractions recorded in its
threshold_selection.json before any of this was computed.

**On SHR.** The classifier that produced the TH channel carries 22.9x more glomus labels in
WKY than SHR, and SHR-B and SHR-C carry none at all. A between-group contrast drawn from it
would be partly the labelling and partly the biology, with no way to tell which. WKY alone is
the defensible half, so it is the default; ``--all`` adds SHR with the caveat attached to
every row rather than to a footnote.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ImageLynx.roi_placement import place_roi                          # noqa: E402
from ImageLynx.specimens import (                                      # noqa: E402
    PROCESSING_VOXEL_UM, SPECIMENS, TH_CHANNEL, VESSEL_CHANNEL,
)
from ImageLynx.statistics.th_morphometry import ThMorphometry, summarise  # noqa: E402

#: The threshold cb_h1_batch froze for all six after checking it does not split the cohorts.
FROZEN_VESSEL_THRESHOLD = 0.9
ROI = (160, 160, 160)

#: Why any SHR row here is provisional. Carried into the JSON and printed beside the table,
#: because a caveat that lives only in a commit message is a caveat nobody reads.
SHR_CAVEAT = (
    "The TH classifier carries 22.9x more glomus labels in WKY than SHR (23262 against "
    "1016); SHR-B and SHR-C carry none at all and SHR-A is labelled at a single depth. Any "
    "WKY against SHR difference below is partly the labelling and partly the biology, and "
    "this labelling cannot separate them."
)


def _crop(path, bounds, channel_index):
    with h5py.File(path, "r") as handle:
        data = handle["exported_data"]
        block = np.asarray(data[bounds[0], bounds[1], bounds[2], channel_index],
                           dtype=np.float32)
    # Ilastik writes uint8 when the export is 8-bit; rescale so the thresholds mean the same
    # thing whichever dtype the export happened to use.
    return block / 255.0 if block.max() > 1.5 else block


def _skeletonise(mask):
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    # Fill enclosed cavities so a hollow arteriole lumen does not skeletonise into a shell.
    # No dilation or gap bridging: bridge_gaps is a plain dilation that inflates narrow
    # vessels hardest, which is the wrong bias for a length measurement.
    return skeletonize(ndi.binary_fill_holes(mask))


def analyse(specimen, th_threshold, vessel_threshold=FROZEN_VESSEL_THRESHOLD, roi=ROI):
    bounds = place_roi(specimen, roi).bounds
    vessel = _crop(specimen.probabilities_path, bounds,
                   VESSEL_CHANNEL.target_index) > vessel_threshold
    th = _crop(specimen.th_probabilities_path, bounds,
               TH_CHANNEL.target_index) > th_threshold
    return summarise(
        specimen_id=specimen.specimen_id,
        group=specimen.group,
        th_mask=th,
        vessel_mask=vessel,
        skeleton=_skeletonise(vessel),
        voxel_um=PROCESSING_VOXEL_UM,
        th_threshold=th_threshold,
        vessel_threshold=vessel_threshold,
    )


def _table(rows):
    head = (f"  {'spec':6s} {'TH mm3':>9s} {'TH %ROI':>8s} {'fVV %':>7s} "
            f"{'len mm':>8s} {'in TH mm':>9s} {'density':>9s} {'TVD med':>8s} {'TVD p90':>8s}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        lines.append(
            f"  {r.specimen_id:6s} {r.th_volume_um3/1e9:9.5f} "
            f"{100*r.th_volume_fraction:7.2f}% {100*r.vessel_volume_fraction:6.2f}% "
            f"{r.centreline_length_um/1000:8.3f} {r.centreline_length_within_th_um/1000:9.3f} "
            f"{r.length_density_mm_per_mm3:9.1f} {r.tvd_median_um:7.2f}u {r.tvd_p90_um:7.2f}u")
    return "\n".join(lines)


def _group_summary(rows, field):
    out = {}
    for group in ("WKY", "SHR"):
        values = [getattr(r, field) for r in rows if r.group == group]
        if values:
            out[group] = (float(np.mean(values)), float(min(values)), float(max(values)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="Include SHR. Every SHR row carries the labelling caveat.")
    ap.add_argument("--th-threshold", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                    help="TH probability cutoffs. Several so the sensitivity is visible.")
    ap.add_argument("--roi", type=int, nargs=3, default=list(ROI), metavar=("Z", "Y", "X"))
    ap.add_argument("--out", default="examples/outputs/cb_h1_th_metrics.json")
    args = ap.parse_args()

    specimens = [s for s in SPECIMENS if args.all or s.group == "WKY"]
    missing = [s.specimen_id for s in specimens if not s.th_probabilities_path.exists()]
    if missing:
        sys.exit(f"No TH probability map for: {', '.join(missing)}")

    print(f"ROI {args.roi[0]}x{args.roi[1]}x{args.roi[2]} voxels = "
          f"{np.prod(args.roi) * float(np.prod(PROCESSING_VOXEL_UM)) / 1e9:.4f} mm3, "
          f"identical for every specimen")
    print(f"Vessel threshold frozen at {FROZEN_VESSEL_THRESHOLD} (cb_h1_batch)")
    if any(s.group == "SHR" for s in specimens):
        print(f"\n  !! {SHR_CAVEAT}\n")

    payload = {"roi_zyx": list(args.roi),
               "vessel_threshold": FROZEN_VESSEL_THRESHOLD,
               "shr_included": bool(args.all),
               "shr_caveat": SHR_CAVEAT if args.all else None,
               "by_threshold": {}}

    for th_threshold in args.th_threshold:
        rows = [analyse(s, th_threshold, roi=tuple(args.roi)) for s in specimens]
        print(f"\nTH threshold {th_threshold}")
        print(_table(rows))
        for field, label in (("th_volume_um3", "1.3 parenchymal volume um3"),
                             ("length_density_mm_per_mm3", "1.3 length density mm/mm3"),
                             ("tvd_median_um", "1.5 TVD median um")):
            summary = _group_summary(rows, field)
            parts = [f"{g} {m:.4g} ({lo:.4g}-{hi:.4g})" for g, (m, lo, hi) in summary.items()]
            line = f"    {label:32s} " + "   ".join(parts)
            if len(summary) == 2:
                w, s_ = summary["WKY"][0], summary["SHR"][0]
                line += f"   SHR/WKY {s_/w:.2f}x"
            print(line)
        payload["by_threshold"][str(th_threshold)] = [r.as_dict() for r in rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print("\nn = 3 per group, so the exact two-sided permutation floor is "
          "2/C(6,3) = 0.10.\nNo p below that is attainable and none is reported.")


if __name__ == "__main__":
    main()
