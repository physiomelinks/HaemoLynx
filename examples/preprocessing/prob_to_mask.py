#!/usr/bin/env python3
"""
Turn an Ilastik probability map into a clean binary vessel mask plus a
calibrated Euclidean distance map, ready to hand to a skeletonisation pipeline.

This is the Fiji-free replacement for section 10 of the protocol. It stops at
the mask/EDT boundary on the assumption that skeletonisation and graph
assembly happen downstream in an existing pipeline.

Usage
-----
    python3 prob_to_mask.py --prob CB3-WKY-A_Probabilities.h5 \
                            --out-prefix CB3-WKY-A

    # sweep thresholds first to see where fragmentation sets in
    python3 prob_to_mask.py --prob ... --sweep

Outputs (as .npy, plus optional .tif for eyeballing in Fiji):
    <prefix>_mask.npy      bool   (z, y, x)
    <prefix>_edt_um.npy    float32 (z, y, x)  distance to background, in um

Requires: numpy, h5py, scipy, scikit-image, tifffile.
"""

import argparse
import sys

import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold

VOXEL_ZYX = (1.8639, 1.8660, 1.8660)  # binned 2x2x2 acquisition, micrometres


def drop_small(mask, min_size):
    """Remove connected components below min_size voxels.

    Done by hand rather than via skimage.morphology.remove_small_objects,
    whose min_size/max_size semantics changed in 0.26 -- this keeps the
    threshold meaning fixed ("strictly smaller than min_size is removed")
    across skimage versions.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return mask, 0, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= min_size
    return keep[lab], n, int(keep.sum())


def load_probability(path, dataset, channel):
    """Read an ilastik export and return the vessel probability as (z, y, x).

    Ilastik writes one channel per label class. If you did not restrict the
    channel range on export, the file has 2 channels and picking the wrong one
    silently gives you the inverted segmentation -- hence the explicit report
    of what was found.
    """
    with h5py.File(path, "r") as f:
        if dataset not in f:
            keys = list(f.keys())
            sys.exit(f"{path}: no dataset '{dataset}'. Found: {keys}")
        d = f[dataset]
        tags = d.attrs.get("axistags", "<none>")
        print(f"  dataset '{dataset}' shape={d.shape} dtype={d.dtype}")
        print(f"  axistags: {tags}")
        arr = np.squeeze(d[...])

    if arr.ndim == 4:
        # ilastik exports channel-last for zyxc input.
        n = arr.shape[-1]
        print(f"  {n} class channels present; taking channel {channel}")
        if channel >= n:
            sys.exit(f"--channel {channel} out of range (file has {n})")
        arr = arr[..., channel]
    elif arr.ndim != 3:
        sys.exit(f"Expected 3D or 4D probability data, got shape {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.5:
        # 8-bit export path; rescale so the thresholds below stay meaningful.
        print(f"  max={arr.max():.1f} -> rescaling from 8-bit to [0, 1]")
        arr = arr / 255.0
    print(f"  probability range [{arr.min():.3f}, {arr.max():.3f}], "
          f"mean {arr.mean():.3f}")
    return arr


def segment(prob, low, high, min_size, fill_holes=True, verbose=True):
    """Hysteresis threshold -> 3D cavity fill -> size filter.

    Hysteresis rather than a single cutoff because a global threshold forces a
    choice between fragmenting faint capillaries and fusing adjacent ones.
    Seeds come from `high`; growth into `low` happens only where connected to
    a seed.
    """
    mask = apply_hysteresis_threshold(prob, low, high)

    if fill_holes:
        # 3D fill closes genuine enclosed cavities (hollow arteriole lumens)
        # but leaves vascular loops alone. A 2D per-slice fill would fill the
        # interior of every in-plane loop and destroy network topology.
        mask = ndi.binary_fill_holes(mask)

    if min_size > 0:
        mask, before, after = drop_small(mask, min_size)
        if verbose:
            print(f"  size filter (<{min_size} vx): {before} -> {after} components")

    return mask


def report(mask, prob_shape):
    lab, n = ndi.label(mask)
    frac = 100.0 * mask.mean()
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        largest = 100.0 * sizes.max() / sizes.sum()
    else:
        largest = 0.0
    print(f"  vessel fraction {frac:.2f}%  components {n}  "
          f"largest holds {largest:.1f}% of vessel voxels")
    return n, frac, largest


def sweep(prob, highs, lows, min_size):
    """Print the fragmentation curve so a threshold can be chosen on evidence.

    The useful operating point is just above the `low` at which component
    count starts climbing steeply -- that is where the classifier stops
    linking real capillary segments and starts leaving them stranded.
    """
    print("\n  high    low   vessel%   components   largest%")
    for h in highs:
        for lo in lows:
            if lo >= h:
                continue
            m = segment(prob, lo, h, min_size, fill_holes=False, verbose=False)
            lab, n = ndi.label(m)
            if n:
                sizes = np.bincount(lab.ravel())[1:]
                largest = 100.0 * sizes.max() / sizes.sum()
            else:
                largest = 0.0
            print(f"  {h:4.2f}   {lo:4.2f}   {100 * m.mean():6.2f}   "
                  f"{n:10d}   {largest:7.1f}")
    print("\n  Pick the `low` just above where `components` starts climbing "
          "steeply\n  and `largest%` starts falling -- that is where genuine "
          "capillary\n  segments begin to be stranded from the network.")


def main():
    ap = argparse.ArgumentParser(
        description="Ilastik probability map -> clean binary mask + calibrated EDT."
    )
    ap.add_argument("--prob", required=True, help="Ilastik HDF5 probability export.")
    ap.add_argument("--dataset", default="exported_data",
                    help="Internal HDF5 path (ilastik default: exported_data).")
    ap.add_argument("--channel", type=int, default=0,
                    help="Class channel holding Vessel probability (default 0). "
                         "Only relevant if you exported all classes.")
    ap.add_argument("--high", type=float, default=0.70,
                    help="Hysteresis seed threshold (default 0.70).")
    ap.add_argument("--low", type=float, default=0.30,
                    help="Hysteresis growth threshold (default 0.30).")
    ap.add_argument("--min-size", type=int, default=50,
                    help="Drop connected components smaller than this many "
                         "voxels (default 50 ~ 325 um^3 at 1.866 um).")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="Skip 3D cavity filling.")
    ap.add_argument("--refine-radii", action="store_true",
                    help="Upsample the probability map 2x (cubic) before "
                         "thresholding, halving the radius quantisation step. "
                         "8x memory and time. See protocol section 0.1.")
    ap.add_argument("--voxel", nargs=3, type=float, default=list(VOXEL_ZYX),
                    metavar=("Z", "Y", "X"),
                    help="Voxel size in um (default: binned acquisition).")
    ap.add_argument("--sweep", action="store_true",
                    help="Print a threshold sweep and exit without writing.")
    ap.add_argument("--save-tif", action="store_true",
                    help="Also write TIFFs for visual inspection in Fiji.")
    ap.add_argument("--out-prefix", help="Output path prefix.")
    args = ap.parse_args()

    print(f"Loading {args.prob}")
    prob = load_probability(args.prob, args.dataset, args.channel)

    if args.sweep:
        sweep(prob, highs=(0.60, 0.70, 0.80),
              lows=(0.15, 0.20, 0.30, 0.40, 0.50), min_size=args.min_size)
        return

    if not args.out_prefix:
        sys.exit("--out-prefix is required unless --sweep is given")

    voxel = tuple(args.voxel)
    if args.refine_radii:
        from scipy.ndimage import zoom
        print("Upsampling probability 2x (cubic)...")
        prob = zoom(prob, 2, order=3, prefilter=True).astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0, out=prob)
        voxel = tuple(v / 2 for v in voxel)
        # min_size is a voxel count, so it must scale with the finer grid.
        args.min_size *= 8
        print(f"  new shape {prob.shape}, voxel {voxel}")

    print(f"Segmenting (hysteresis {args.low} -> {args.high})")
    mask = segment(prob, args.low, args.high, args.min_size,
                   fill_holes=not args.no_fill_holes)
    report(mask, prob.shape)

    print("Computing Euclidean distance map (calibrated, um)")
    # sampling= makes the output physical units directly, and handles the
    # 1.0011 axial:lateral ratio without a separate conversion step.
    edt = ndi.distance_transform_edt(mask, sampling=voxel).astype(np.float32)
    # EDT over ALL vessel voxels is dominated by surface voxels (distance ~1
    # voxel), which understates vessel calibre. Ridge voxels -- local maxima of
    # the EDT -- approximate the centerline, so they are the honest proxy for
    # radius until the real skeleton is computed downstream.
    ridge = mask & (edt >= ndi.maximum_filter(edt, size=3) - 1e-6)
    r = edt[ridge]
    if r.size:
        print(f"  radius at ridge (centerline proxy) voxels, n={r.size}:")
        print(f"    median {np.median(r):.2f} um, p95 {np.percentile(r, 95):.2f} um, "
              f"max {r.max():.2f} um")
        step = voxel[1]
        print(f"  radius quantisation step ~{step:.2f} um "
              f"-> +/-{100 * 0.5 * step / 3.0:.0f}% on a nominal 3.0 um "
              f"capillary radius,")
        print(f"     which propagates to ~{(1 + 0.5 * step / 3.0) ** 4:.1f}x "
              f"on Poiseuille resistance (r^-4).")

    np.save(f"{args.out_prefix}_mask.npy", mask)
    np.save(f"{args.out_prefix}_edt_um.npy", edt)
    print(f"\nWrote {args.out_prefix}_mask.npy and {args.out_prefix}_edt_um.npy")

    if args.save_tif:
        import tifffile
        tifffile.imwrite(f"{args.out_prefix}_mask.tif",
                         (mask * 255).astype(np.uint8))
        tifffile.imwrite(f"{args.out_prefix}_edt_um.tif", edt)
        print(f"Wrote {args.out_prefix}_mask.tif and {args.out_prefix}_edt_um.tif")

    print("\nHand off to your skeletonisation pipeline:")
    print("    mask = np.load(f'{prefix}_mask.npy')")
    print("    edt  = np.load(f'{prefix}_edt_um.npy')   # already in um")
    print("    radii = edt[skeleton]   # sample, do NOT multiply")


if __name__ == "__main__":
    main()
