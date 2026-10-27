"""Cut and freeze the benchmark sub-volume: WKY-A's probability map at its placed ROI.

Run on the machine that holds the real maps. It writes the cut (both channels, float32 zyxc,
gzip HDF5) and its placement sidecar into ``tests/data/benchmark/``; the file name and the
sidecar format are the ones ``carotid_image_to_model.py --placed-input`` reads. Nothing else
writes these files. It refuses a tree with tracked changes, and an existing output unless
``--overwrite`` is given.

    venv/bin/python examples/make_benchmark_subvolume.py [--overwrite]

``tests/test_benchmark_subvolume.py`` rebuilds the cut with ``build_subvolume`` and checks it
against the committed files.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ImageLynx import cb_settings, specimens  # noqa: E402
from ImageLynx.artefact_provenance import read_provenance  # noqa: E402
from ImageLynx.preprocessing import crop_roi  # noqa: E402
from ImageLynx.roi_placement import (  # noqa: E402
    place_roi, placed_input_sidecar_path, placement_fell_back, roi_record,
)

SPECIMEN_ID = "WKY-A"
BENCHMARK_DIR = REPO / "tests" / "data" / "benchmark"
SUBVOLUME_PATH = BENCHMARK_DIR / "wky_a_benchmark_subvolume.h5"
DATASET = specimens.PROBABILITIES_DATASET
GZIP_LEVEL = 4


def repo_git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout


def tracked_changes(git=repo_git) -> str:
    """`git status` lines for tracked files that differ from HEAD (untracked files ignored)."""
    return git("status", "--porcelain", "--untracked-files=no").strip()


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_name(path: Path) -> str:
    """The map's path below the CB data root; its file name when it lies elsewhere."""
    try:
        return str(path.relative_to(specimens.CB_DATA_ROOT))
    except ValueError:
        return path.name


def build_subvolume(specimen=None):
    """The cut and its sidecar record, built from the real map, nothing written.

    The cut is what the pipeline's crop yields: ``crop_roi`` at the placed offsets. It is
    checked against the placed bounds, so a drift between the two raises.
    """
    import dask.array as da
    import h5py

    specimen = specimen or specimens.get_specimen(SPECIMEN_ID)
    size = cb_settings.ROI_VOXELS
    placement = place_roi(specimen, size)
    if placement_fell_back(placement):
        raise ValueError(f"{specimen.specimen_id}: the placement fell back to the volume "
                         f"centre ({placement.source}); the sub-volume must sit on tissue.")
    path = specimen.probabilities_path
    with h5py.File(path, "r") as handle:
        dataset = handle[DATASET]
        shape = tuple(int(v) for v in dataset.shape)
        cut = crop_roi(da.from_array(dataset, chunks="auto"), size_zyx=size,
                       offset_z=placement.offsets_zyx[0], offset_y=placement.offsets_zyx[1],
                       offset_x=placement.offsets_zyx[2])
        cut = np.ascontiguousarray(np.asarray(cut), dtype=np.float32)
        direct = np.asarray(dataset[placement.bounds + (slice(None),)], dtype=np.float32)
    if not np.array_equal(cut, direct):
        raise ValueError("crop_roi at the placed offsets does not give the placed bounds' voxels.")
    record = roi_record(placement, shape[:3], centred=False)
    record.update({
        "source_path": _source_name(path),
        "source_sha256": sha256_of(path),
        "classifier_provenance": read_provenance(path),
    })
    return cut, record


def write_subvolume(cut, record, commit, directory=BENCHMARK_DIR, name=SUBVOLUME_PATH.name):
    import h5py

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with h5py.File(path, "w") as handle:
        handle.create_dataset(DATASET, data=cut, compression="gzip",
                              compression_opts=GZIP_LEVEL, chunks=tuple(min(32, n) for n in cut.shape[:3]) + (cut.shape[-1],))
    placed_input_sidecar_path(path).write_text(
        json.dumps({**record, "commit": commit}, indent=2) + "\n")
    return path


def make(overwrite=False, directory=BENCHMARK_DIR, git=repo_git, build=build_subvolume):
    """Refuse a dirty tree or an existing output, then build and write. Returns the map path."""
    dirty = tracked_changes(git)
    if dirty:
        raise RuntimeError("Refusing to write the benchmark sub-volume from a tree with tracked "
                           f"changes (its sidecar records the commit):\n{dirty}")
    path = Path(directory) / SUBVOLUME_PATH.name
    sidecar = placed_input_sidecar_path(path)
    existing = [p for p in (path, sidecar) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{[str(p) for p in existing]} exist; pass --overwrite to replace "
                              "the frozen sub-volume.")
    cut, record = build()
    return write_subvolume(cut, record, git("rev-parse", "HEAD").strip(), directory=directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing sub-volume and sidecar.")
    args = parser.parse_args(argv)
    path = make(overwrite=args.overwrite)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.2f} MB) and its sidecar.")


if __name__ == "__main__":
    main()
