"""Run all six specimens through the pipeline on matched sub-volumes, and compare the groups.

Three stages, each of which can be run alone:

  --stage placement   where each ROI will sit, and why          (seconds)
  --stage threshold   choose one threshold for all six          (minutes)
  --stage run         run the pipeline and compare the groups   (~6 min per specimen)

Two design decisions are load-bearing and deliberate.

**One threshold for all six.** Per-specimen thresholds would absorb exactly the classifier
differences the shared classifier exists to prevent, converting an instrument artefact into
an apparently clean result. The threshold stage therefore reports each specimen's own choice
but freezes a single value, and checks whether those choices separate by cohort - because if
they do, part of any group difference is the segmentation rather than the tissue.

**Matched ROIs, placed on tissue rather than on array indices.** SHR volumes average
89 Mvoxel against 63 for WKY, so a percentage crop samples more of SHR; and the axial tissue
peak ranges from slice 106 to 230, so a centred crop lands mid-organ in one specimen and in
the sparse margin of another. Both effects are group-correlated. See ImageLynx.roi_placement.

Usage
-----
    python examples/cb_h1_batch.py --stage placement
    python examples/cb_h1_batch.py --stage threshold
    python examples/cb_h1_batch.py --stage run --threshold 0.90
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ImageLynx.io import read_ilastik_probabilities                     # noqa: E402
from ImageLynx.roi_placement import format_placement_table, place_roi   # noqa: E402
from ImageLynx.specimens import (                                       # noqa: E402
    PROCESSING_VOXEL_UM, SPECIMENS, get_specimen,
)
from ImageLynx.statistics.cohort_split import assess_cohort_split       # noqa: E402
from ImageLynx.statistics.threshold_selection import (                  # noqa: E402
    select_threshold, sweep_thresholds,
)

DEFAULT_ROI = (160, 160, 160)
DEFAULT_GRID = [0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cb_h1_batch"


def _predicted():
    missing = [s.specimen_id for s in SPECIMENS if not s.probabilities_path.exists()]
    if missing:
        print(f"No probability map for: {', '.join(missing)}")
    return [s for s in SPECIMENS if s.probabilities_path.exists()]


def stage_placement(roi):
    placements = [place_roi(s, roi) for s in SPECIMENS]
    print(f"ROI {roi[0]}x{roi[1]}x{roi[2]} voxels = "
          f"{np.prod(roi) * float(np.prod(PROCESSING_VOXEL_UM)) / 1e9:.4f} mm3, identical "
          f"for every specimen\n")
    print(format_placement_table(placements, SPECIMENS))
    print("\nPlacement rule: z from the volume's own axial tissue peak (QC record), y and x "
          "from the\ngrayscale centroid. Centring on signal samples mid-organ, so absolute "
          "densities are\noverestimates of the whole organ - the comparison is like-for-like, "
          "the absolute level is not.")
    for placement in placements:
        print(f"  {placement.specimen_id}: {placement.source}")
    return placements


def stage_threshold(roi, grid):
    """Choose one threshold for all six, and check the per-specimen choices for a cohort split."""
    chosen, foreground = {}, {}
    for specimen in _predicted():
        placement = place_roi(specimen, roi)
        volume = read_ilastik_probabilities(
            specimen.probabilities_path, expected_shape_zyx=specimen.shape_zyx)
        sub = volume[placement.bounds]
        samples = sweep_thresholds(sub, grid, PROCESSING_VOXEL_UM)
        selection = select_threshold(samples)
        print(f"\n########## {specimen.specimen_id} ({specimen.group}) ##########")
        print(selection.format_table())
        if selection.threshold is not None:
            chosen[specimen.specimen_id] = selection.threshold
        foreground[specimen.specimen_id] = {s.threshold: s.foreground_fraction
                                            for s in samples}
        del volume, sub

    if not chosen:
        print("\nNo specimen yielded a usable threshold. That is a segmentation problem, "
              "not a thresholding one.")
        return None

    print("\n" + "=" * 78)
    split = assess_cohort_split(chosen, quantity="selected threshold")
    for group, values in split.values_by_group.items():
        print(f"  {group}: {', '.join(f'{v:.2f}' for v in sorted(values))}")
    print(f"\n  {split.verdict}")

    frozen = float(np.median(list(chosen.values())))
    frozen = min(DEFAULT_GRID, key=lambda t: abs(t - frozen))
    print(f"\nFrozen threshold for all six: {frozen:.2f} (median of the per-specimen choices)")

    at_frozen = {sid: fg[frozen] for sid, fg in foreground.items() if frozen in fg}
    if len(at_frozen) >= 4:
        fg_split = assess_cohort_split(at_frozen, quantity="foreground fraction at the frozen threshold")
        print("\nForeground fraction at that threshold:")
        for sid, value in sorted(at_frozen.items()):
            print(f"  {sid}: {value:.4f}")
        print(f"\n  {fg_split.verdict}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "threshold_selection.json").write_text(json.dumps(
        {"per_specimen": chosen, "frozen": frozen,
         "threshold_split": split.verdict,
         "foreground_at_frozen": at_frozen}, indent=2))
    return frozen


def stage_run(roi, threshold):
    """Run the pipeline once per specimen with frozen parameters and matched ROIs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = Path(__file__).resolve().parent / "carotid_image_to_model.py"
    results = {}
    for specimen in _predicted():
        placement = place_roi(specimen, roi)
        out = OUTPUT_DIR / specimen.specimen_id
        out.mkdir(exist_ok=True)
        command = [
            sys.executable, str(pipeline),
            "--specimen", specimen.specimen_id,
            "--roi-voxels", *[str(v) for v in roi],
            "--hysteresis-low", str(threshold),
        ]
        print(f"\n=== {specimen.specimen_id} ({specimen.group}) centre "
              f"{placement.centre_zyx} ===", flush=True)
        log = out / "pipeline.log"
        with log.open("w") as handle:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
        results[specimen.specimen_id] = code
        print(f"  exit={code}  log={log}")
    failed = [s for s, c in results.items() if c != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} completed."
          + (f" Failed: {', '.join(failed)}" if failed else ""))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["placement", "threshold", "run"], required=True)
    parser.add_argument("--roi-voxels", type=int, nargs=3, default=list(DEFAULT_ROI))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Frozen threshold for --stage run. Required there.")
    args = parser.parse_args()
    roi = tuple(args.roi_voxels)

    if args.stage == "placement":
        stage_placement(roi)
    elif args.stage == "threshold":
        stage_threshold(roi, DEFAULT_GRID)
    else:
        if args.threshold is None:
            parser.error("--threshold is required for --stage run; take it from --stage threshold")
        stage_run(roi, args.threshold)


if __name__ == "__main__":
    main()
