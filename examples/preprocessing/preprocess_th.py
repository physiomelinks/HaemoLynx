#!/usr/bin/env python3
"""
Preprocessing for the TH (glomus cell) channel of LCFM carotid body Z-stacks.

Companion to preprocess_cb.py, which handles the lectin/vessel channel. This
module imports that one rather than reimplementing it, so both channels of the
same acquisition get identical IO, background subtraction, outlier handling and
HDF5 writing, and stay comparable downstream.

    raw ZCYX TIFF
      -> extract TH channel (index 1)                   preprocess_cb
      -> z-profile diagnosis, recorded not acted on     preprocess_cb
      -> rolling-ball background subtraction            preprocess_cb
      -> [optional] impulse-noise removal               preprocess_cb
      -> tissue-anchored normalisation                  here
      -> signed 3D Difference of Gaussians              here
      -> multi-channel HDF5 with ilastik axistags       preprocess_cb

What this deliberately does NOT do, and why. Each was measured on
CB3-WKY-CB-{A,B,C}; see th_glomus_preprocessing_review.md for the numbers.

  * No histogram-matching bleach correction. On WKY-C it multiplies slice 0 by
    26.9x and raises the TH-positive fraction in the near-empty top of the
    stack from 0.004% to 1.413%, a 350-fold increase. Worse, it forces every
    slice to the same p99 and the same TH-positive fraction, which hard-codes
    TH density to be uniform in z. Only a per-slice multiplicative gain is
    offered, and only where the z-profile is genuinely monotonic decay.

  * No blanket 3x3x3 median. It costs 18% of the doughnut contrast that the
    whole segmentation strategy depends on (core-to-ring 79.4% -> 65.2%,
    core floor up 62%). Use --remove-outliers, which touches only impulse
    noise, exactly as preprocess_cb does for the same reason.

  * No clipping of the DoG at zero. The dark nuclear core is where the DoG is
    negative: 99.8% of cores are negative before the clip and exactly zero
    after it, which maps 'inside the nucleus' and 'outside the cell' to the
    same value. The signed map is kept.

  * No tiling. Reflect padding to a 256 grid would make 52% of WKY-B a
    mirrored duplicate of real tissue, which the classifier will label as
    though it were real. These volumes are 35-100 M voxels and fit whole.

Typical use
-----------
    # 1. Look before you leap. Reports the z-profile and tissue occupancy.
    python3 preprocess_th.py --input . --diagnose

    # 2. Process every volume in a directory.
    python3 preprocess_th.py --input . --output-dir ilastik_inputs_th --save-tif

    # 3. Re-run with a shared cohort anchor once step 2 has reported per-volume
    #    ones, so the applied gain does not depend on how much empty frame was
    #    acquired.
    python3 preprocess_th.py --input . --anchors 1950 9700

Requires: numpy, tifffile, h5py, scipy, scikit-image.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter

from preprocess_cb import (
    VOXEL_ZYX,
    diagnose_z,
    human,
    percentile_anchors,
    read_vessel_channel,
    remove_outliers,
    subtract_background,
    write_h5,
    _wrap,
)

# ---------------------------------------------------------------- calibration

# The TH channel of the CB3 acquisition. Channel 0 is lectin: verified by
# byte-comparing it against the previously extracted C1-*_vessels.tif.
TH_CHANNEL = 1

# Rolling-ball radius in pixels. 12 px is a 22.4 um ball radius, comfortably
# above the 15 um maximum glomus soma diameter (8 px), so somas ride on top of
# the ball and survive while the diffuse haze trapped inside cell nests does
# not. Note this is a radius: the guide compared it against a cell diameter.
DEF_ROLLING_BALL = 12.0

# Difference-of-Gaussians scales in pixels. The small scale resolves the
# cytoplasmic ring; the large scale is set near the cell radius, measured at
# r = 4.0 px (7.46 um) as the peak of the mean radial profile about 2181
# detected nuclear cores in WKY-A parenchyma.
DEF_DOG_SIGMAS = (1.0, 3.0)

# Smoothing scale for the tissue mask, in pixels. 15 px = 28 um, a few cell
# diameters, so the mask follows nests rather than individual cells.
DEF_TISSUE_SMOOTH = 15


# ------------------------------------------------------------------ tissue mask

def tissue_mask(vol, smooth=DEF_TISSUE_SMOOTH):
    """Coarse foreground mask, used only to place normalisation anchors.

    The carotid body occupies a minority of each field (measured: 16.1%, 17.4%
    and 28.9% across three WKY volumes, a 1.8x spread), so a whole-volume
    percentile anchor is set largely by how much empty frame happened to be
    acquired. That is a cropping accident, and if WKY and SHR were framed
    differently it becomes a group-differential gain. Anchoring inside tissue
    removes the dependence.

    The mask is deliberately generous: it includes the interstitial space
    inside a nest, which is genuinely part of the organ, and on a synthetic
    blob it runs about 2.4x the true extent. That is fine for its one job.
    Only the HIGH anchor sets the gain, and the top percentile inside the mask
    is still drawn from tissue voxels. The LOW anchor stays at the background
    floor either way, which is the correct zero point.
    """
    from skimage.filters import threshold_otsu

    smoothed = uniform_filter(vol.astype(np.float32), size=smooth)
    try:
        thr = threshold_otsu(smoothed)
    except ValueError:
        return np.ones(vol.shape, dtype=bool), 1.0
    mask = smoothed > thr
    frac = float(mask.mean())
    if frac < 0.005:
        # Degenerate segmentation; fall back rather than anchor on a handful
        # of voxels, and let the caller see it in QC.
        return np.ones(vol.shape, dtype=bool), frac
    return mask, frac


def normalise_to_anchors(vol, lo, hi):
    """Scale to [0, 1] against externally chosen anchors."""
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol.astype(np.float32, copy=False) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out


# -------------------------------------------------- stage: z-profile correction

def multiplicative_z_correction(vol, verdict, force=False):
    """Per-slice scalar gain, matching each slice's robust signal level.

    Multiplicative rather than histogram matching: it rescales a slice without
    redistributing its intensities, so it cannot promote background noise to
    cell level or flatten real depth variation in cell density.

    Refuses to run unless the z-profile is genuinely monotonic decay. A hump
    is the extent of the tissue block, not photobleaching, and 'correcting' it
    amplifies the sparse end slices by several fold.
    """
    if verdict != "monotonic decay" and not force:
        raise ValueError(
            f"z-profile verdict is '{verdict}', not 'monotonic decay', so a "
            "depth correction is not justified: this shape is the extent of "
            "the tissue block, not photobleaching. Correcting it multiplies "
            "the sparse end slices, promoting background noise to cell-level "
            "intensity. Measured on WKY-C, histogram matching raised the "
            "TH-positive fraction in slices 0-99 from 0.004% to 1.413%. "
            "Label at several depths in ilastik instead, or pass --force-z-correct "
            "if you have a specific reason."
        )
    n = vol.shape[0]
    sig = np.array([np.percentile(vol[z], 99.0) for z in range(n)], dtype=np.float64)
    ref = float(np.median(sig[sig > 0])) if np.any(sig > 0) else 1.0
    gain = np.where(sig > 0, ref / np.maximum(sig, 1e-9), 1.0)
    # A gain far from 1 means the slice has almost no tissue, not that it is
    # dim. Cap it so those slices are left alone rather than amplified.
    gain = np.clip(gain, 0.25, 4.0)
    out = (vol * gain[:, None, None]).astype(np.float32)
    return out, {"reference_p99": ref, "gain_min": float(gain.min()),
                 "gain_max": float(gain.max())}


# ------------------------------------------------------- stage: soma DoG channel

def soma_dog(vol, sigmas, mask, saturated, split=False):
    """Signed 3D Difference of Gaussians about the glomus cell scale.

    Positive on the bright cytoplasmic ring, negative in the dark nuclear core.
    Both halves are informative and the sign is what separates 'core' from
    'outside the cell', so the map is NOT clipped at zero. Scaling is symmetric
    about zero so that the sign survives.

    With `split`, returns the positive and negative rectified parts as two
    non-negative channels instead, for classifiers that prefer them.
    """
    f = vol.astype(np.float32, copy=False)
    small, large = float(sigmas[0]), float(sigmas[1])
    if not small < large:
        raise ValueError(f"DoG sigmas must be increasing, got {sigmas}")
    dog = gaussian_filter(f, sigma=small) - gaussian_filter(f, sigma=large)

    # Symmetric anchor, taken inside tissue so that the vast empty background
    # cannot set the scale.
    inside = dog[mask] if mask.any() else dog
    scale = float(np.percentile(np.abs(inside), 100.0 - saturated))
    if scale <= 0:
        scale = 1.0
    out = np.clip(dog / scale, -1.0, 1.0)

    stats = {
        "dog_sigmas_px": [small, large],
        "dog_symmetric_scale": scale,
        "fraction_negative": float((dog < 0).mean()),
    }
    if split:
        return [np.clip(out, 0, None), np.clip(-out, 0, None)], stats
    return [out], stats


# ------------------------------------------------------------------ per-volume

def process_volume(path, args):
    base = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'=' * 70}\n{base}  (TH channel)\n{'=' * 70}")
    qc = {"input": path, "parameters": vars(args).copy(), "channel_used": args.channel}
    t_all = time.time()

    print("  [1/6] reading + TH channel extraction")
    vol, n_ch, axes = read_vessel_channel(path, args.channel)
    print(f"        axes={axes} channels={n_ch} -> TH volume {vol.shape}")
    qc["shape_zyx"] = list(vol.shape)
    qc["n_channels_in_file"] = n_ch

    print("  [2/6] z-profile diagnosis")
    qc["z_profile"] = diagnose_z(vol)
    verdict = qc["z_profile"]["verdict"]
    print(f"        verdict: {verdict} (peak at slice "
          f"{qc['z_profile']['peak_slice']}/{vol.shape[0]})")
    for line in _wrap(qc["z_profile"]["advice"], 62):
        print(f"        {line}")

    mask, occ = tissue_mask(vol, args.tissue_smooth)
    qc["tissue_occupancy"] = round(occ, 4)
    print(f"        tissue occupancy {100 * occ:.1f}% of the field")
    if occ < 0.005:
        print("        WARNING: tissue mask degenerate, anchoring on the whole "
              "volume instead")

    if args.diagnose:
        whole = percentile_anchors(vol, args.saturated)
        inside = percentile_anchors(vol[mask], args.saturated)
        print(f"        anchors whole volume {whole[0]:.0f}-{whole[1]:.0f}, "
              f"within tissue {inside[0]:.0f}-{inside[1]:.0f} "
              f"({inside[1] / max(whole[1], 1e-9):.2f}x)")
        qc["anchors_whole_volume"] = list(whole)
        qc["anchors_within_tissue"] = list(inside)
        return qc

    if args.z_correct == "multiplicative":
        print("  [3/6] per-slice multiplicative depth correction")
        vol, zstats = multiplicative_z_correction(
            vol, verdict, force=args.force_z_correct)
        print(f"        gains {zstats['gain_min']:.2f}-{zstats['gain_max']:.2f} "
              f"about p99={zstats['reference_p99']:.0f}")
        qc["z_correction"] = zstats
        mask, occ = tissue_mask(vol, args.tissue_smooth)
    else:
        print("  [3/6] depth correction SKIPPED (recorded in QC, not applied)")
        qc["z_correction"] = None

    if args.rolling_ball > 0:
        print(f"  [4/6] rolling-ball background subtraction "
              f"(radius={args.rolling_ball:g} px = "
              f"{args.rolling_ball * args.voxel[1]:.1f} um)")
        t = time.time()
        vol = subtract_background(vol, args.rolling_ball, args.workers)
        print(f"        done in {human(time.time() - t)}")
    else:
        print("  [4/6] background subtraction SKIPPED")

    if args.remove_outliers > 0:
        vol, n_hot = remove_outliers(vol, 1, args.remove_outliers)
        frac = 100.0 * n_hot / vol.size
        print(f"        impulse-noise removal replaced {n_hot} voxels ({frac:.4f}%)")
        qc["outliers_replaced"] = n_hot
        if frac > 1.0:
            print("        WARNING: >1% replaced -- threshold is too low, you "
                  "are median-filtering the image and will lose the nuclear core")

    print(f"  [5/6] normalisation (saturated={args.saturated}%, "
          f"{'supplied' if args.anchors else 'tissue-anchored'})")
    if args.anchors:
        lo, hi = float(args.anchors[0]), float(args.anchors[1])
        qc["anchor_source"] = "supplied"
    else:
        lo, hi = percentile_anchors(vol[mask], args.saturated)
        qc["anchor_source"] = "tissue"
    whole_lo, whole_hi = percentile_anchors(vol, args.saturated)
    print(f"        anchors [{lo:.1f}, {hi:.1f}] -> [0, 1]  "
          f"(whole-volume would be [{whole_lo:.1f}, {whole_hi:.1f}])")
    qc["normalisation_anchors"] = [lo, hi]
    qc["normalisation_anchors_whole_volume"] = [whole_lo, whole_hi]
    vol = normalise_to_anchors(vol, lo, hi)

    channels, names = [vol], ["grayscale"]

    print("  [6/6] signed soma Difference of Gaussians")
    dog_channels, dog_stats = soma_dog(vol, args.dog_sigmas, mask,
                                       args.saturated, split=args.split_dog)
    channels += dog_channels
    names += (["soma_dog_positive", "soma_dog_negative"] if args.split_dog
              else ["soma_dog_signed"])
    print(f"        sigmas {dog_stats['dog_sigmas_px']} px, "
          f"{100 * dog_stats['fraction_negative']:.1f}% of voxels negative "
          f"(core and background), sign retained")
    qc.update(dog_stats)

    if args.with_vessel_channel:
        print(f"        adding the lectin channel {args.vessel_channel} as context")
        ves, _, _ = read_vessel_channel(path, args.vessel_channel)
        if args.rolling_ball > 0:
            ves = subtract_background(ves, args.rolling_ball, args.workers)
        v_lo, v_hi = percentile_anchors(ves[mask], args.saturated)
        channels.append(normalise_to_anchors(ves, v_lo, v_hi))
        names.append("lectin_context")
        qc["lectin_anchors"] = [v_lo, v_hi]

    os.makedirs(args.output_dir, exist_ok=True)
    out_h5 = os.path.join(args.output_dir, f"{base}_TH_ilastik.h5")
    write_h5(out_h5, channels, names, args.voxel)
    qc["output_h5"] = out_h5
    qc["channel_names"] = names

    if args.save_tif:
        for arr, nm in zip(channels, names):
            p = os.path.join(args.output_dir, f"{base}_TH_{nm}.tif")
            tifffile.imwrite(p, arr.astype(np.float32), imagej=True,
                             resolution=(1.0 / args.voxel[2], 1.0 / args.voxel[1]),
                             metadata={"unit": "um", "spacing": args.voxel[0],
                                       "axes": "ZYX"})
        print(f"        wrote {len(channels)} inspection TIFFs")

    qc["elapsed_seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(args.output_dir, f"{base}_TH_qc.json"), "w") as f:
        json.dump(qc, f, indent=2)

    print(f"\n  -> {out_h5}")
    print(f"     shape {(*channels[0].shape, len(channels))}  float32  axistags zyxc")
    for i, nm in enumerate(names):
        print(f"     channel {i}: {nm}")
    print(f"     total {human(qc['elapsed_seconds'])}")
    return qc


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Raw LCFM z-stack -> Ilastik-ready HDF5 for the TH glomus "
                    "cell channel. Companion to preprocess_cb.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--input", nargs="+", required=True,
                    help="TIFF files, directories, or globs.")
    ap.add_argument("--output-dir", default="ilastik_inputs_th")
    ap.add_argument("--channel", type=int, default=TH_CHANNEL,
                    help="Index of the TH channel in the acquisition.")
    ap.add_argument("--with-vessel-channel", action="store_true",
                    help="Add the lectin channel as an extra ilastik input. The "
                         "two channels are one acquisition on an identical grid, "
                         "so they are co-registered by construction.")
    ap.add_argument("--vessel-channel", type=int, default=0)

    ap.add_argument("--z-correct", choices=("none", "multiplicative"),
                    default="none",
                    help="Depth correction. Histogram matching is deliberately "
                         "not offered; see the module docstring.")
    ap.add_argument("--force-z-correct", action="store_true",
                    help="Apply the multiplicative correction even where the "
                         "z-profile says it is not justified.")
    ap.add_argument("--rolling-ball", type=float, default=DEF_ROLLING_BALL,
                    help="Rolling-ball radius in PIXELS. 0 disables.")
    ap.add_argument("--remove-outliers", type=float, default=0,
                    help="Replace voxels exceeding the local median by this "
                         "much. 0 disables (the recommended default).")
    ap.add_argument("--saturated", type=float, default=0.35,
                    help="Percent clipped at each end during normalisation.")
    ap.add_argument("--anchors", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="Use these normalisation anchors instead of per-volume "
                         "tissue percentiles. Use a cohort-wide pair so the gain "
                         "cannot vary with how much empty frame was acquired.")
    ap.add_argument("--tissue-smooth", type=int, default=DEF_TISSUE_SMOOTH,
                    help="Smoothing width in PIXELS for the tissue mask.")

    ap.add_argument("--dog-sigmas", nargs=2, type=float,
                    default=list(DEF_DOG_SIGMAS), metavar=("SMALL", "LARGE"),
                    help="Difference-of-Gaussians scales in PIXELS.")
    ap.add_argument("--split-dog", action="store_true",
                    help="Emit the positive and negative parts of the DoG as two "
                         "non-negative channels instead of one signed channel.")

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--voxel", nargs=3, type=float, default=list(VOXEL_ZYX),
                    metavar=("Z", "Y", "X"), help="Voxel size in um.")
    ap.add_argument("--diagnose", action="store_true",
                    help="Report the z-profile, tissue occupancy and both anchor "
                         "choices for each volume, then stop. Run this first.")
    ap.add_argument("--save-tif", action="store_true",
                    help="Also write each channel as a calibrated TIFF.")
    args = ap.parse_args()

    paths = []
    for item in args.input:
        if os.path.isdir(item):
            paths += sorted(glob.glob(os.path.join(item, "*.tif"))
                            + glob.glob(os.path.join(item, "*.tiff")))
        else:
            paths += sorted(glob.glob(item)) or [item]
    # Skip the single-channel derivatives of a previous vessel extraction.
    paths = [p for p in dict.fromkeys(paths)
             if os.path.isfile(p) and "_vessels" not in os.path.basename(p)]
    if not paths:
        sys.exit("No input files matched.")

    print(f"Found {len(paths)} volume(s):")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    results, t0 = [], time.time()
    for p in paths:
        try:
            results.append(process_volume(p, args))
        except Exception as exc:  # keep the batch alive
            print(f"\n  FAILED: {p}\n    {type(exc).__name__}: {exc}")
            results.append({"input": p, "error": str(exc)})

    print(f"\n{'=' * 70}\nSUMMARY  ({human(time.time() - t0)} total)\n{'=' * 70}")
    for r in results:
        name = os.path.basename(r["input"])
        if "error" in r:
            print(f"  FAIL  {name}: {r['error']}")
        elif args.diagnose:
            print(f"  {name}: {r['z_profile']['verdict']}, tissue "
                  f"{100 * r['tissue_occupancy']:.1f}%, "
                  f"tissue anchors {r['anchors_within_tissue'][0]:.0f}-"
                  f"{r['anchors_within_tissue'][1]:.0f}")
        else:
            print(f"  OK    {name} -> {os.path.basename(r['output_h5'])} "
                  f"({human(r['elapsed_seconds'])})")

    if args.diagnose and all("error" not in r for r in results):
        his = [r["anchors_within_tissue"][1] for r in results]
        los = [r["anchors_within_tissue"][0] for r in results]
        print(f"\n  Cohort-wide anchors (median of per-volume tissue anchors):")
        print(f"    --anchors {np.median(los):.0f} {np.median(his):.0f}")
        print("  Pass those to the real run so the applied gain is identical "
              "across\n  specimens and cannot track how much empty frame was "
              "acquired.")


if __name__ == "__main__":
    main()
