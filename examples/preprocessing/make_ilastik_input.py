#!/usr/bin/env python3
"""
Build a multi-channel HDF5 for Ilastik Pixel Classification from LCFM carotid
body Z-stacks.

Replaces the non-existent `convert_tiles_to_hdf5.py` referenced by v4 of the
protocol, and performs the two steps that are impractical in the Fiji GUI:

  * per-scale normalisation followed by a voxelwise max across tubeness maps
    (Fiji's Tubeness is not gamma-normalised, so a raw max is dominated by the
    largest sigma -- see protocol section 6.1)
  * writing correct lowercase ilastik axistags ('zyxc')

Usage
-----
From Fiji-generated tubeness maps (already combined into VESS_* volumes, or
supplied as raw per-sigma maps -- either works, they are normalised here):

    python3 make_ilastik_input.py \
        --gray GRAY.tif \
        --vess VESS_FINE.tif VESS_COARSE.tif \
        --out  CB3-WKY-A_ilastik.h5

Computing vesselness in Python instead of running Tubeness five times:

    python3 make_ilastik_input.py \
        --gray GRAY.tif --compute-vesselness \
        --out  CB3-WKY-A_ilastik.h5

Requires: numpy, tifffile, h5py  (+ scikit-image for --compute-vesselness,
scipy for --upsample).
"""

import argparse
import json
import sys

import h5py
import numpy as np
import tifffile

# Acquisition calibration for the CB3 2x2x2 dataset, in micrometres.
VOXEL_Z = 1.8639
VOXEL_Y = 1.8660
VOXEL_X = 1.8660

# Sigmas in pixels for --compute-vesselness. Geometric spacing of ~1.4x across
# the capillary range (1.0-2.0 px) and 2x across the arteriole range, matching
# the sigma table in section 6.2 of the protocol.
FINE_SIGMAS = (1.0, 1.4, 2.0)
COARSE_SIGMAS = (4.0, 8.0)

# vigra AxisInfo type flags, as written by ilastik itself.
_TYPEFLAG = {"c": 1, "z": 2, "y": 2, "x": 2, "t": 8}


def robust_normalise(vol, saturated=0.05):
    """Scale to [0, 1] using percentile anchors.

    Equivalent to Fiji's Enhance Contrast + Normalize + Use stack histogram.
    Percentiles rather than min/max so that a handful of hot voxels cannot
    compress the whole dynamic range -- which is the failure mode that left
    0.175% of the DUP_ stack pinned at 65535.
    """
    lo = np.percentile(vol, saturated)
    hi = np.percentile(vol, 100.0 - saturated)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0, out=out)


def read_volume(path):
    """Read a 3D TIFF as (z, y, x) float32, squeezing singleton axes."""
    arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        sys.exit(f"{path}: expected a 3D stack, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def normalised_max(paths, saturated=0.05):
    """Voxelwise max over per-scale-normalised volumes.

    The normalisation is the point: without it the largest sigma wins almost
    everywhere and the capillary-scale response is erased.
    """
    acc = None
    for p in paths:
        v = robust_normalise(read_volume(p), saturated)
        acc = v if acc is None else np.maximum(acc, v, out=acc)
    return acc


def compute_vesselness(gray, sigmas):
    """Multiscale Sato tubeness, normalised per scale before the max.

    Sato (rather than Frangi) because it is the same ridge measure Fiji's
    Tubeness plugin implements, so results stay comparable to a Fiji run.
    """
    from skimage.filters import sato

    acc = None
    for s in sigmas:
        r = sato(gray, sigmas=[s], black_ridges=False).astype(np.float32)
        r = robust_normalise(r)
        acc = r if acc is None else np.maximum(acc, r, out=acc)
        print(f"    sigma={s:<4} ({s * VOXEL_X:.2f} um) done")
    return acc


def upsample(vol, factor):
    from scipy.ndimage import zoom

    return zoom(vol, factor, order=3, prefilter=True).astype(np.float32)


def write_h5(path, channels, names, voxel_size):
    """Write (z, y, x, c) float32 with ilastik-readable axistags.

    Channels are written one at a time and dropped as we go. np.stack would
    briefly hold two full copies, which at the unbinned 0.933 um resolution
    (3 channels ~ 9.7 GB) would exceed available RAM.
    """
    zyx = channels[0].shape
    shape = (*zyx, len(channels))

    axes = [
        {"key": k, "typeFlags": _TYPEFLAG[k], "resolution": 0, "description": ""}
        for k in "zyxc"
    ]
    tags = json.dumps({"axes": axes})

    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "data",
            shape=shape,
            dtype=np.float32,
            chunks=(min(32, shape[0]), min(128, shape[1]),
                    min(128, shape[2]), shape[3]),
            compression="gzip",
            compression_opts=4,
        )
        for i in range(len(channels)):
            ds[..., i] = channels[i]
            channels[i] = None  # release as we go
        ds.attrs["axistags"] = tags
        # Informational only; ilastik ignores these but they keep the file
        # self-describing for the downstream skeletonisation stage.
        ds.attrs["channel_names"] = json.dumps(names)
        ds.attrs["voxel_size_um_zyx"] = np.asarray(voxel_size, dtype=np.float64)

    print(f"\nWrote {path}")
    print(f"  dataset  /data  shape={shape}  dtype=float32  axistags=zyxc")
    for i, n in enumerate(names):
        print(f"  channel {i}: {n}")
    print(f"  voxel size (z, y, x) um: {voxel_size}")


def main():
    ap = argparse.ArgumentParser(
        description="Build a multi-channel Ilastik HDF5 from preprocessed LCFM stacks."
    )
    ap.add_argument(
        "--gray",
        required=True,
        help="Background-subtracted, globally normalised grayscale TIFF (protocol phase 5).",
    )
    ap.add_argument(
        "--vess",
        nargs="*",
        default=[],
        metavar="TIFF",
        help="Vesselness TIFFs. Each becomes one channel. Pass several sigma "
        "maps under a single flag only if you want them collapsed -- normally "
        "pass VESS_FINE.tif VESS_COARSE.tif as two separate channels.",
    )
    ap.add_argument(
        "--collapse-vess",
        action="store_true",
        help="Combine all --vess inputs into ONE channel via normalised max, "
        "instead of one channel each.",
    )
    ap.add_argument(
        "--compute-vesselness",
        action="store_true",
        help="Ignore --vess; compute fine and coarse vesselness from --gray with "
        "scikit-image (slower, but avoids five manual Tubeness runs).",
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="TIFF",
        help="Additional channels, e.g. the second acquisition channel C2. "
        "Normalised the same way as --gray.",
    )
    ap.add_argument(
        "--upsample",
        type=int,
        default=1,
        choices=(1, 2),
        help="Cubic upsampling factor. 2 halves the radius quantisation error "
        "on 2-4 voxel capillaries at 8x the memory cost (protocol section 0.1).",
    )
    ap.add_argument(
        "--saturated",
        type=float,
        default=0.05,
        help="Percent clipped at each end during normalisation (default 0.05).",
    )
    ap.add_argument("--out", required=True, help="Output .h5 path.")
    args = ap.parse_args()

    channels, names = [], []

    print(f"Reading grayscale: {args.gray}")
    gray = read_volume(args.gray)
    print(f"  shape {gray.shape}")
    gray = robust_normalise(gray, args.saturated)
    channels.append(gray)
    names.append("grayscale")

    if args.compute_vesselness:
        print("Computing multiscale vesselness (fine)...")
        channels.append(compute_vesselness(gray, FINE_SIGMAS))
        names.append(f"vesselness_fine_sigma{FINE_SIGMAS}")
        print("Computing multiscale vesselness (coarse)...")
        channels.append(compute_vesselness(gray, COARSE_SIGMAS))
        names.append(f"vesselness_coarse_sigma{COARSE_SIGMAS}")
    elif args.vess:
        if args.collapse_vess:
            print(f"Combining {len(args.vess)} vesselness maps into one channel...")
            channels.append(normalised_max(args.vess, args.saturated))
            names.append("vesselness_max")
        else:
            for p in args.vess:
                print(f"Reading vesselness: {p}")
                channels.append(robust_normalise(read_volume(p), args.saturated))
                names.append(f"vesselness:{p}")
    else:
        print("WARNING: no vesselness channel. Grayscale-only input is workable "
              "but loses the tubular shape prior.", file=sys.stderr)

    for p in args.extra:
        print(f"Reading extra channel: {p}")
        channels.append(robust_normalise(read_volume(p), args.saturated))
        names.append(f"extra:{p}")

    shapes = {c.shape for c in channels}
    if len(shapes) != 1:
        sys.exit(f"Channel shapes disagree: {shapes}")

    voxel = (VOXEL_Z, VOXEL_Y, VOXEL_X)
    if args.upsample > 1:
        print(f"Upsampling {args.upsample}x (cubic)...")
        channels = [upsample(c, args.upsample) for c in channels]
        voxel = tuple(v / args.upsample for v in voxel)

    write_h5(args.out, channels, names, voxel)


if __name__ == "__main__":
    main()
