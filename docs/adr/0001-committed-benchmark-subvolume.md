# Commit a frozen WKY-A sub-volume and its reference run

The end-to-end regression suite runs `carotid_image_to_model.py` on WKY-A's probability map cut at its placed ROI, and compares chosen outputs with a committed reference run. We commit the sub-volume (both channels, gzip HDF5) and the reference run under `tests/data/benchmark/` instead of building them from the real map at test time (`realdata`), so the suite also runs in CI, where the real data is absent. The price is a permanent 15-33 MB in git history for each fixture version, so the sub-volume is frozen: only `scripts/make_benchmark_subvolume.py` changes it, and its sidecar records the source map's sha256 and the placed box. Everything the suite commits stays under 100 MB, and a test checks this.

## Consequences

- The pipeline takes the pre-cut file through `--placed-input`, which skips `place_roi` and uses the sidecar's record. Nothing is patched in the test.
- The reference run must match in both the local venv and CI's fresh Python 3.10 install, so floats are compared with per-output tolerances. A package upgrade that moves a mask or skeleton voxel fails the suite and is not skipped.
