#!/usr/bin/env python3
"""
End-to-end Python preprocessing for LCFM carotid body Z-stacks.

Takes raw multi-channel TIFFs straight to Ilastik-ready HDF5, replacing
phases 1-6 of the Fiji protocol:

    raw ZCYX TIFF
      -> extract vessel channel                        (protocol section 1)
      -> z-profile diagnosis                           (section 2)
      -> rolling-ball background subtraction           (section 3)
      -> [optional] impulse-noise removal              (section 4)
      -> global percentile normalisation               (section 5)
      -> multiscale vesselness, fine + coarse          (section 6)
      -> multi-channel HDF5 with ilastik axistags      (section 8)

Deliberately does NOT denoise by default and does NOT tile: at 1.866 um a
capillary is 2-4 voxels wide, so both operations cost more than they give.
See protocol sections 0 and 4.

Typical use
-----------
    # 1. Look before you leap -- no processing, just report.
    python3 preprocess_cb.py --input . --diagnose

    # 2. Process every volume in a directory.
    python3 preprocess_cb.py --input . --output-dir ilastik_inputs --save-tif

Requires: numpy, tifffile, h5py, scipy, scikit-image.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import tifffile
from scipy import ndimage as ndi

# ---------------------------------------------------------------- calibration

VOXEL_ZYX = (1.8639, 1.8660, 1.8660)  # binned 2x2x2 acquisition, micrometres

# Defaults are in PIXELS and assume the 1.866 um binned data. For the unbinned
# 0.933 um acquisition every pixel-valued default below doubles -- see protocol
# section 0.2, or just pass --voxel 0.932 0.933 0.933 and the doubled values.
DEF_ROLLING_BALL = 30.0
DEF_SIGMAS_FINE = (1.0, 1.4, 2.0)     # capillary scales:  1.9 - 3.7 um radius
DEF_SIGMAS_COARSE = (4.0, 8.0)        # arteriole scales:  7.5 - 14.9 um radius

_TYPEFLAG = {"c": 1, "z": 2, "y": 2, "x": 2, "t": 8}  # vigra AxisInfo flags

# Measured on this dataset: skimage's sato holds ~160 bytes of working set per
# voxel in flight (Hessian elements + eigenvalue intermediates, several in
# float64). Calibrated against two runs on volume A (101 M voxels):
#   workers=8, chunk_z=48 -> 19.4 GB peak, 264 s
#   workers=4, chunk_z=24 ->  9.1 GB peak, 373 s
# Used to auto-size the worker pool so the ridge filter cannot exhaust RAM.
_BYTES_PER_VOXEL_IN_FLIGHT = 160


def plan_workers(shape, max_sigma, chunk_z, budget_gb, requested_workers):
    """Pick a worker count that keeps the ridge filter inside `budget_gb`.

    Memory is dominated by workers x (chunk_z + 2*halo) x Y x X, not by the
    volume itself, so this is the only knob that reliably prevents an OOM --
    and the one that has to change when moving to the unbinned resolution.
    """
    yx = shape[1] * shape[2]
    halo = int(math.ceil(4 * max_sigma))
    per_worker = (chunk_z + 2 * halo) * yx * _BYTES_PER_VOXEL_IN_FLIGHT
    # Parent simultaneously holds the working volume plus both vesselness
    # results, with headroom for the copies made during normalisation.
    parent = 4 * float(np.prod(shape)) * 4
    avail = budget_gb * 1e9 - parent

    if avail <= per_worker:
        return 1, per_worker / 1e9, parent / 1e9
    n = int(avail // per_worker)
    n = max(1, min(n, requested_workers, os.cpu_count() or 1))
    return n, per_worker / 1e9, parent / 1e9


# ------------------------------------------------------------------ utilities

def human(seconds):
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.1f} s"


def percentile_anchors(vol, saturated, max_samples=20_000_000):
    """Robust [lo, hi] anchors from a subsample.

    Subsampling because np.percentile sorts, and sorting 100M+ voxels costs
    more than the accuracy is worth -- the anchors are stable to <0.1% here.
    """
    flat = vol.ravel()
    if flat.size > max_samples:
        step = flat.size // max_samples + 1
        flat = flat[::step]
    return (float(np.percentile(flat, saturated)),
            float(np.percentile(flat, 100.0 - saturated)))


def normalise(vol, saturated):
    """Scale to [0, 1] against the WHOLE-STACK histogram.

    This is Fiji's Enhance Contrast + Normalize + "Use stack histogram". The
    stack-wide part is what matters: per-slice normalisation would reintroduce
    slice-to-slice intensity jumps and break the 3D derivative features.
    """
    lo, hi = percentile_anchors(vol, saturated)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32), (lo, hi)
    out = (vol.astype(np.float32, copy=False) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out, (lo, hi)


# ------------------------------------------------------------------- stage: IO

def read_vessel_channel(path, channel):
    """Read a TIFF and return (vessel_volume_zyx, n_channels, axes_string)."""
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes, shape = series.axes, series.shape
        arr = series.asarray()

    if "C" in axes:
        ci = axes.index("C")
        n_ch = shape[ci]
        if channel >= n_ch:
            sys.exit(f"{path}: --channel {channel} but only {n_ch} channels")
        arr = np.take(arr, channel, axis=ci)
    else:
        n_ch = 1
        if channel != 0:
            print(f"    note: single-channel file, ignoring --channel {channel}")

    arr = np.squeeze(arr)
    if arr.ndim != 3:
        sys.exit(f"{path}: expected a 3D stack after channel selection, "
                 f"got shape {arr.shape} from axes {axes}")
    return arr.astype(np.float32), n_ch, axes


# ------------------------------------------------- stage: z-profile diagnosis

def diagnose_z(vol):
    """Decide whether axial intensity decay is real, or just tissue extent.

    Photobleaching / attenuation gives a monotonically falling profile.
    A blob of tissue in the middle of the block gives a hump, and "correcting"
    a hump with histogram matching multiplies the sparse end slices by several
    fold, promoting their background noise to vessel-level intensity.
    """
    n = vol.shape[0]
    sig = np.array([np.percentile(vol[z], 99.0) for z in range(n)])
    bg = np.array([np.percentile(vol[z], 50.0) for z in range(n)])

    k = max(3, n // 20)
    smooth = ndi.uniform_filter1d(sig, size=k, mode="nearest")
    peak = int(np.argmax(smooth))
    peak_frac = peak / max(n - 1, 1)

    first, last = smooth[: n // 4].mean(), smooth[-n // 4:].mean()
    if peak_frac < 0.15:
        verdict = "monotonic decay"
        advice = ("Real attenuation. Consider a multiplicative correction "
                  "(scale each slice by a single factor). Never histogram "
                  "matching.")
    elif peak_frac > 0.85:
        verdict = "inverse decay (rises with z)"
        advice = ("Unusual -- check stack orientation before correcting "
                  "anything.")
    else:
        verdict = "hump / tissue-extent dominated"
        advice = ("Do NOT apply bleach correction. This shape is the tissue "
                  "block, not photobleaching. Label at several depths in "
                  "ilastik instead.")

    return {
        "peak_slice": peak,
        "peak_position_fraction": round(peak_frac, 3),
        "p99_first_quarter_mean": round(float(first), 1),
        "p99_last_quarter_mean": round(float(last), 1),
        "background_p50_range": [round(float(bg.min()), 1),
                                 round(float(bg.max()), 1)],
        "verdict": verdict,
        "advice": advice,
    }


# --------------------------------------------- stage: background subtraction

def _rb_slice(args):
    from skimage.restoration import rolling_ball
    sl, radius = args
    # The thread-count keyword was renamed: `workers` in older scikit-image,
    # `num_threads` from 0.20. Either way it must be 1, because the outer
    # ProcessPoolExecutor already owns the parallelism.
    import inspect
    kw = ("num_threads" if "num_threads" in
          inspect.signature(rolling_ball).parameters else "workers")
    return sl - rolling_ball(sl, radius=radius, **{kw: 1})


def subtract_background(vol, radius, workers):
    """Per-slice rolling-ball subtraction, parallel over z.

    Per-slice (not 3D) to match ImageJ and because haze varies with depth.
    Verified bit-identical to the serial result; the parallelism is purely a
    wall-clock win (~12x on 16 workers).
    """
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_rb_slice, [(vol[z], radius) for z in range(vol.shape[0])],
                          chunksize=4))
    res = np.stack(out).astype(np.float32)
    np.clip(res, 0, None, out=res)
    return res


# ----------------------------------------------------- stage: outlier removal

def remove_outliers(vol, radius, threshold):
    """Replace only voxels far above their local median.

    The safe alternative to a median filter at this voxel size: a 3x3x3 median
    spans 5.6 um and would erase 2-4 voxel capillaries outright, whereas this
    touches only genuine impulse noise.
    """
    med = ndi.median_filter(vol, size=2 * radius + 1)
    hot = (vol - med) > threshold
    out = vol.copy()
    out[hot] = med[hot]
    return out, int(hot.sum())


# ---------------------------------------------------------- stage: vesselness

def _sato_chunk(args):
    from skimage.filters import sato
    block, sigma, take0, take_n = args
    r = sato(block, sigmas=[sigma], black_ridges=False)
    return r[take0:take0 + take_n]


def vesselness(vol, sigmas, workers, chunk_z, downscale=1, saturated=0.05):
    """Multiscale ridge filter: per-scale normalise, then voxelwise max.

    Two things this does that a naive implementation does not:

    1. Normalises EACH scale before the max. Sato/Frangi responses are not
       gamma-normalised across sigma -- large sigma responds ~3x stronger on
       this data, so a raw max is dominated by the coarsest scale and the
       capillary-scale selectivity is lost entirely.

    2. Processes in z-chunks with a 4*sigma halo. A monolithic call peaks at
       ~17.6 GB on a 100M-voxel volume; chunked, memory scales with chunk size.
       Verified identical to the monolithic result (corr 1.000000).

    `downscale=2` computes at half resolution with halved sigmas and upsamples
    the result: 8x faster, correlation 0.9989 against full resolution. Safe for
    coarse scales, where the response varies slowly by construction.
    """
    if downscale > 1:
        from scipy.ndimage import zoom
        work = zoom(vol, 1.0 / downscale, order=1)
        sigmas = [s / downscale for s in sigmas]
    else:
        work = vol

    nz = work.shape[0]
    acc = None
    for sigma in sigmas:
        halo = int(math.ceil(4 * sigma))
        jobs = []
        for z0 in range(0, nz, chunk_z):
            z1 = min(z0 + chunk_z, nz)
            a, b = max(0, z0 - halo), min(nz, z1 + halo)
            jobs.append((work[a:b], sigma, z0 - a, z1 - z0))

        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(_sato_chunk, jobs))
        else:
            parts = [_sato_chunk(j) for j in jobs]

        r = np.concatenate(parts, axis=0).astype(np.float32)
        lo, hi = percentile_anchors(r, saturated)
        if hi > lo:
            r = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
        else:
            r[:] = 0.0
        acc = r if acc is None else np.maximum(acc, r, out=acc)
        print(f"      sigma={sigma * downscale:<5.2f} "
              f"({sigma * downscale * VOXEL_ZYX[1]:5.2f} um) done")

    if downscale > 1:
        from scipy.ndimage import zoom
        factors = np.array(vol.shape) / np.array(acc.shape)
        acc = zoom(acc, factors, order=1).astype(np.float32)
        acc = acc[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
        pad = [(0, vol.shape[i] - acc.shape[i]) for i in range(3)]
        if any(p[1] for p in pad):
            acc = np.pad(acc, pad, mode="edge")
    return acc


# -------------------------------------------------------------- stage: output

def write_h5(path, channels, names, voxel):
    """Write (z, y, x, c) float32 with ilastik-readable lowercase axistags.

    Channels are written one at a time; np.stack would hold two full copies,
    which is 19 GB at the unbinned resolution.
    """
    zyx = channels[0].shape
    shape = (*zyx, len(channels))
    tags = json.dumps({"axes": [
        {"key": k, "typeFlags": _TYPEFLAG[k], "resolution": 0, "description": ""}
        for k in "zyxc"]})

    with h5py.File(path, "w") as f:
        # Chunks are clamped to the data shape: h5py rejects a chunk larger
        # than the dataset in any dimension, which a small volume would hit.
        chunks = (min(32, shape[0]), min(128, shape[1]),
                  min(128, shape[2]), shape[3])
        ds = f.create_dataset("data", shape=shape, dtype=np.float32,
                              chunks=chunks,
                              compression="gzip", compression_opts=4)
        for i in range(len(channels)):
            ds[..., i] = channels[i]
        ds.attrs["axistags"] = tags
        ds.attrs["channel_names"] = json.dumps(names)
        ds.attrs["voxel_size_um_zyx"] = np.asarray(voxel, dtype=np.float64)


# ------------------------------------------------------------------ per-volume

def process_volume(path, args):
    base = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'=' * 70}\n{base}\n{'=' * 70}")
    qc = {"input": path, "parameters": vars(args).copy()}
    t_all = time.time()

    print("  [1/6] reading + channel extraction")
    vol, n_ch, axes = read_vessel_channel(path, args.channel)
    print(f"        axes={axes} channels={n_ch} -> vessel volume {vol.shape}")
    qc["shape_zyx"] = list(vol.shape)
    qc["n_channels_in_file"] = n_ch

    print("  [2/6] z-profile diagnosis")
    qc["z_profile"] = diagnose_z(vol)
    print(f"        verdict: {qc['z_profile']['verdict']} "
          f"(peak at slice {qc['z_profile']['peak_slice']}"
          f"/{vol.shape[0]})")
    for line in _wrap(qc["z_profile"]["advice"], 62):
        print(f"        {line}")
    if args.diagnose:
        return qc

    if args.rolling_ball > 0:
        print(f"  [3/6] rolling-ball background subtraction "
              f"(radius={args.rolling_ball:g} px = "
              f"{args.rolling_ball * args.voxel[1]:.1f} um)")
        t = time.time()
        vol = subtract_background(vol, args.rolling_ball, args.workers)
        print(f"        done in {human(time.time() - t)}")
    else:
        print("  [3/6] background subtraction SKIPPED")

    if args.remove_outliers > 0:
        print(f"  [4/6] impulse-noise removal (threshold={args.remove_outliers})")
        vol, n_hot = remove_outliers(vol, 1, args.remove_outliers)
        frac = 100.0 * n_hot / vol.size
        print(f"        replaced {n_hot} voxels ({frac:.4f}%)")
        qc["outliers_replaced"] = n_hot
        if frac > 1.0:
            print("        WARNING: >1% replaced -- threshold is too low, "
                  "you are median-filtering the image")
    else:
        print("  [4/6] denoising SKIPPED (recommended -- see protocol section 4)")

    print(f"  [5/6] global normalisation (saturated={args.saturated}%, "
          f"stack histogram)")
    vol, (lo, hi) = normalise(vol, args.saturated)
    print(f"        anchors [{lo:.1f}, {hi:.1f}] -> [0, 1]")
    qc["normalisation_anchors"] = [lo, hi]

    channels, names = [vol], ["grayscale"]

    print("  [6/6] multiscale vesselness")
    n_w, per_w, parent_gb = plan_workers(
        vol.shape, max(args.sigmas_fine), args.chunk_z,
        args.max_memory_gb, args.workers)
    print(f"        memory plan: budget {args.max_memory_gb:g} GB = "
          f"{parent_gb:.1f} GB parent + {n_w} x {per_w:.1f} GB workers "
          f"(chunk_z={args.chunk_z})")
    if n_w < args.workers:
        print(f"        reduced workers {args.workers} -> {n_w} to stay in "
              f"budget; raise --max-memory-gb or lower --chunk-z for speed")
    qc["workers_used"] = n_w

    t = time.time()
    print(f"    fine scales {list(args.sigmas_fine)} px")
    fine = vesselness(vol, args.sigmas_fine, n_w, args.chunk_z,
                      downscale=1, saturated=args.saturated)
    ds = 2 if args.fast_coarse else 1
    print(f"    coarse scales {list(args.sigmas_coarse)} px"
          f"{' (at half resolution)' if ds > 1 else ''}")
    coarse = vesselness(vol, args.sigmas_coarse, n_w, args.chunk_z,
                        downscale=ds, saturated=args.saturated)
    print(f"        vesselness done in {human(time.time() - t)}")

    if args.single_vesselness:
        channels.append(np.maximum(fine, coarse))
        names.append("vesselness_max")
    else:
        channels += [fine, coarse]
        names += ["vesselness_fine", "vesselness_coarse"]

    if args.extra_channel is not None:
        print(f"        adding extra input channel {args.extra_channel}")
        extra, _, _ = read_vessel_channel(path, args.extra_channel)
        if args.rolling_ball > 0:
            extra = subtract_background(extra, args.rolling_ball, args.workers)
        extra, _ = normalise(extra, args.saturated)
        channels.append(extra)
        names.append(f"channel{args.extra_channel}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_h5 = os.path.join(args.output_dir, f"{base}_ilastik.h5")
    write_h5(out_h5, channels, names, args.voxel)
    qc["output_h5"] = out_h5
    qc["channel_names"] = names

    if args.save_tif:
        for arr, nm in zip(channels, names):
            p = os.path.join(args.output_dir, f"{base}_{nm}.tif")
            tifffile.imwrite(p, arr.astype(np.float32),
                             imagej=True,
                             resolution=(1.0 / args.voxel[2], 1.0 / args.voxel[1]),
                             metadata={"unit": "um", "spacing": args.voxel[0],
                                       "axes": "ZYX"})
        print(f"        wrote {len(channels)} inspection TIFFs")

    qc["elapsed_seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(args.output_dir, f"{base}_qc.json"), "w") as f:
        json.dump(qc, f, indent=2)

    print(f"\n  -> {out_h5}")
    print(f"     shape {(*vol.shape, len(channels))}  float32  axistags zyxc")
    for i, nm in enumerate(names):
        print(f"     channel {i}: {nm}")
    print(f"     total {human(qc['elapsed_seconds'])}")
    return qc


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Raw LCFM z-stack -> Ilastik-ready HDF5 (replaces Fiji "
                    "phases 1-6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--input", nargs="+", required=True,
                    help="TIFF files, directories, or globs.")
    ap.add_argument("--output-dir", default="ilastik_inputs")
    ap.add_argument("--channel", type=int, default=0,
                    help="Index of the vessel/lectin channel.")
    ap.add_argument("--extra-channel", type=int, default=None,
                    help="Add another acquisition channel (e.g. 1 for TH) as an "
                         "extra ilastik input. Does not affect downstream "
                         "output, which stays a single vessel probability map.")

    ap.add_argument("--rolling-ball", type=float, default=DEF_ROLLING_BALL,
                    help="Rolling-ball radius in PIXELS. 0 disables.")
    ap.add_argument("--remove-outliers", type=float, default=0,
                    help="Replace voxels exceeding the local median by this "
                         "much. 0 disables (the recommended default).")
    ap.add_argument("--saturated", type=float, default=0.02,
                    help="Percent clipped at each end during normalisation.")

    ap.add_argument("--sigmas-fine", type=float, nargs="+",
                    default=list(DEF_SIGMAS_FINE),
                    help="Capillary-scale sigmas in PIXELS.")
    ap.add_argument("--sigmas-coarse", type=float, nargs="+",
                    default=list(DEF_SIGMAS_COARSE),
                    help="Arteriole-scale sigmas in PIXELS.")
    ap.add_argument("--single-vesselness", action="store_true",
                    help="Collapse fine+coarse into one channel. Simpler, but "
                         "loses the fine/coarse disagreement cue that helps "
                         "ilastik separate touching parallel capillaries.")
    ap.add_argument("--fast-coarse", action="store_true", default=True,
                    help="Compute coarse scales at half resolution (8x faster, "
                         "correlation 0.9989).")
    ap.add_argument("--no-fast-coarse", dest="fast_coarse", action="store_false")

    ap.add_argument("--workers", type=int, default=8,
                    help="Upper bound on parallel processes. The actual count "
                         "is reduced automatically to fit --max-memory-gb.")
    ap.add_argument("--chunk-z", type=int, default=32,
                    help="Z-chunk for the ridge filter. Lower it if even one "
                         "worker will not fit.")
    ap.add_argument("--max-memory-gb", type=float, default=12.0,
                    help="Memory budget for the ridge filter. Worker count is "
                         "derived from this. Measured: volume A needs 19.4 GB "
                         "at workers=8/chunk_z=48 and 9.1 GB at "
                         "workers=4/chunk_z=24.")
    ap.add_argument("--voxel", nargs=3, type=float, default=list(VOXEL_ZYX),
                    metavar=("Z", "Y", "X"), help="Voxel size in um.")

    ap.add_argument("--diagnose", action="store_true",
                    help="Report the z-profile for each volume and stop. Run "
                         "this first.")
    ap.add_argument("--save-tif", action="store_true",
                    help="Also write each channel as a calibrated TIFF for "
                         "inspection in Fiji.")
    args = ap.parse_args()

    paths = []
    for item in args.input:
        if os.path.isdir(item):
            paths += sorted(glob.glob(os.path.join(item, "*.tif"))
                            + glob.glob(os.path.join(item, "*.tiff")))
        else:
            paths += sorted(glob.glob(item)) or [item]
    paths = [p for p in dict.fromkeys(paths) if os.path.isfile(p)]
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
            print(f"  {name}: {r['z_profile']['verdict']}")
        else:
            print(f"  OK    {name} -> {os.path.basename(r['output_h5'])} "
                  f"({human(r['elapsed_seconds'])})")

    if not args.diagnose and any("error" not in r for r in results):
        print("\nNext: open the .h5 files in ilastik Pixel Classification.")
        print("  Input Data -> add all volumes; axes should read as zyxc.")
        print("  See protocol sections 9 and 10.")


if __name__ == "__main__":
    main()
