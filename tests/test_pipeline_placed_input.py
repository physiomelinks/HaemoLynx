"""``carotid_image_to_model.py --placed-input`` runs the pipeline on a pre-cut probability map.

A pre-cut map comes with a JSON sidecar holding its placement record, so a run needs neither the
specimen's full map nor ``place_roi``. Every test here goes through the command line. The refusals hand the script a map file that is
not an HDF5 file at all: a refusal that fires before any data loads never opens it, so the
message is the conflict and not an h5py error.
"""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ImageLynx.batch_outputs import EDGE_TABLE_NAME, VESSELS_VTP_NAME
from ImageLynx.roi_placement import (
    ROI_RECORD_KEYS,
    ROI_RECORD_NAME,
    RoiPlacement,
    centre_to_offsets,
    placed_input_sidecar_path,
    roi_record,
)
from ImageLynx.specimens import PROBABILITIES_DATASET

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "carotid_image_to_model.py"
SIZE = (48, 48, 48)
FULL_SHAPE = (100, 100, 100)
CENTRE = (50, 50, 50)


def _record(specimen_id="WKY-A", size=SIZE):
    placement = RoiPlacement(specimen_id=specimen_id, centre_zyx=CENTRE, size_zyx=size,
                             offsets_zyx=centre_to_offsets(CENTRE, FULL_SHAPE), peak_slice=50,
                             source="z=qc_peak_slice, yx=grayscale_centroid")
    return roi_record(placement, FULL_SHAPE, centred=False)


def _write_sidecar(map_path, record):
    placed_input_sidecar_path(map_path).write_text(json.dumps(record))


def _synthetic_map(size=SIZE):
    """Two tubes along y (the boundary axis) joined by two connectors; channel 0 is vessel."""
    nz, ny, nx = size
    z, y, x = np.mgrid[:nz, :ny, :nx]
    vessel = np.zeros(size, bool)
    for z0, x0 in ((14, 14), (34, 34)):
        vessel |= (z - z0) ** 2 + (x - x0) ** 2 <= 4
    vessel |= ((y - 24) ** 2 + (x - 14) ** 2 <= 4) & (z >= 14) & (z <= 34)
    vessel |= ((y - 24) ** 2 + (z - 34) ** 2 <= 4) & (x >= 14) & (x <= 34)
    volume = np.zeros(size + (2,), np.float32)
    volume[..., 0] = np.where(vessel, 1.0, 0.0)
    volume[..., 1] = 1.0 - volume[..., 0]
    return volume


@pytest.fixture
def placed(tmp_path):
    """A valid pre-cut map and sidecar; returns the map's path."""
    path = tmp_path / "pre_cut.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(PROBABILITIES_DATASET, data=_synthetic_map())
    _write_sidecar(path, _record())
    return path


@pytest.fixture
def not_hdf5(tmp_path):
    """A map file that cannot be opened, with a valid sidecar: proves a refusal opened nothing."""
    path = tmp_path / "not_data.h5"
    path.write_bytes(b"this is not an hdf5 file")
    _write_sidecar(path, _record())
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def _refusal(proc, *words):
    """The run died with a message that names every one of ``words``, and no h5py error."""
    assert proc.returncode != 0, proc.stdout[-2000:]
    for word in words:
        assert word in proc.stderr, f"{word!r} missing from:\n{proc.stderr[-2000:]}"
    assert "Unable to" not in proc.stderr and "OSError" not in proc.stderr, (
        "the map was opened before the refusal:\n" + proc.stderr[-2000:])


# --- A run ---------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
def test_a_pre_cut_map_runs_end_to_end_and_the_run_folder_holds_the_sidecar_record(placed, tmp_path):
    out = tmp_path / "run"
    proc = _run("--placed-input", placed, "--output-dir", out)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert json.loads((out / ROI_RECORD_NAME).read_text()) == _record()
    # It ran the rest of the pipeline on the map as given, with no second crop.
    assert "Applying ROI crop" not in proc.stdout
    for name in (EDGE_TABLE_NAME, VESSELS_VTP_NAME):
        assert (out / name).exists(), sorted(p.name for p in out.iterdir())


def test_the_record_keys_are_the_ones_roi_record_writes():
    assert list(ROI_RECORD_KEYS) == list(_record())


def test_keys_the_sidecar_adds_stay_out_of_roi_placement_json(placed, tmp_path):
    """A sidecar may carry more than the record (where the cut came from, say)."""
    _write_sidecar(placed, {**_record(), "source_sha256": "abc"})
    out = tmp_path / "run"
    proc = _run("--placed-input", placed, "--output-dir", out, "--exit-after-mask")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert json.loads((out / ROI_RECORD_NAME).read_text()) == _record()


# --- Refusals that need no data ------------------------------------------------------------

@pytest.mark.parametrize("flags, named", [
    (["--roi-voxels", "48", "48", "48"], ["--roi-voxels", "--placed-input"]),
    (["--roi-centred"], ["--roi-centred", "--placed-input"]),
    (["--use-cache-dir"], ["--use-cache-dir", "--placed-input"]),
    (["--chunk-fraction", "0.5"], ["--chunk-fraction", "--placed-input"]),
    (["--sub-volume", "0.5"], ["--sub-volume", "--placed-input"]),
])
def test_a_flag_that_would_cut_or_reuse_the_map_is_refused(not_hdf5, tmp_path, flags, named):
    proc = _run("--placed-input", not_hdf5, "--output-dir", tmp_path / "run", *flags)
    _refusal(proc, *named)


def test_a_chunk_fraction_of_one_is_not_chunking(not_hdf5, tmp_path):
    proc = _run("--placed-input", not_hdf5, "--output-dir", tmp_path / "run",
                "--chunk-fraction", "1.0")
    # Past the flag check, it opens the map, which is not an HDF5 file.
    assert proc.returncode != 0
    assert "Unable to synchronously open file" in proc.stderr, proc.stderr[-2000:]
    assert "--placed-input cannot be combined" not in proc.stderr


@pytest.mark.parametrize("yaml_text, named", [
    ("PipelineConfig:\n  chunk_fraction: 0.5\n", "--chunk-fraction"),
    ("PipelineConfig:\n  pre_generated_mask_and_skeleton: true\n", "--use-cache-dir"),
    ("SkeletonConfig:\n  sub_volume_voxels: [48, 48, 48]\n", "sub_volume_voxels"),
    ("SkeletonConfig:\n  sub_volume_offset_y: 0.1\n", "sub_volume_offset_zyx"),
    ("SkeletonConfig:\n  sub_volume_percentage: 0.5\n", "sub_volume_percentage"),
])
def test_a_crop_or_cache_setting_from_a_yaml_config_is_refused(not_hdf5, tmp_path, yaml_text,
                                                               named):
    config = tmp_path / "conflict.yaml"
    config.write_text(yaml_text)
    proc = _run("--placed-input", not_hdf5, "--output-dir", tmp_path / "run",
                "--config", config)
    _refusal(proc, named, "--placed-input")


def test_a_specimen_the_sidecar_does_not_name_is_refused(tmp_path):
    path = tmp_path / "wky_b.h5"
    path.write_bytes(b"not data")
    _write_sidecar(path, _record("WKY-B"))
    proc = _run("--placed-input", path, "--specimen", "WKY-A", "--output-dir", tmp_path / "run")
    _refusal(proc, "WKY-A", "WKY-B", "--specimen")


def test_the_default_specimen_is_checked_too(tmp_path):
    path = tmp_path / "shr_a.h5"
    path.write_bytes(b"not data")
    _write_sidecar(path, _record("SHR-A"))
    proc = _run("--placed-input", path, "--output-dir", tmp_path / "run")
    _refusal(proc, "WKY-A", "SHR-A", "--specimen")


def test_a_missing_sidecar_is_refused(tmp_path):
    path = tmp_path / "no_sidecar.h5"
    path.write_bytes(b"not data")
    proc = _run("--placed-input", path, "--output-dir", tmp_path / "run")
    _refusal(proc, str(placed_input_sidecar_path(path)), "sidecar")


def test_a_missing_map_is_refused(tmp_path):
    proc = _run("--placed-input", tmp_path / "absent.h5", "--output-dir", tmp_path / "run")
    _refusal(proc, "absent.h5")


@pytest.mark.parametrize("text, named", [
    ("{not json", "not valid JSON"),
    ("[1, 2, 3]", "JSON object"),
])
def test_a_malformed_sidecar_is_refused(tmp_path, text, named):
    path = tmp_path / "bad.h5"
    path.write_bytes(b"not data")
    placed_input_sidecar_path(path).write_text(text)
    proc = _run("--placed-input", path, "--output-dir", tmp_path / "run")
    _refusal(proc, named, str(placed_input_sidecar_path(path)))


@pytest.mark.parametrize("key", ROI_RECORD_KEYS)
def test_a_sidecar_without_a_record_key_is_refused(tmp_path, key):
    path = tmp_path / "short.h5"
    path.write_bytes(b"not data")
    record = _record()
    del record[key]
    _write_sidecar(path, record)
    proc = _run("--placed-input", path, "--output-dir", tmp_path / "run")
    _refusal(proc, key, "missing")


@pytest.mark.parametrize("change, named", [
    ({"centred": True}, "array centre"),
    ({"size_zyx": [48, 48]}, "size_zyx"),
    ({"bounds_zyx": [[26, 74], [26, 74], [26, 75]]}, "bounds_zyx"),
    ({"specimen_id": 7}, "specimen_id"),
])
def test_a_sidecar_that_contradicts_itself_is_refused(tmp_path, change, named):
    path = tmp_path / "odd.h5"
    path.write_bytes(b"not data")
    _write_sidecar(path, {**_record(), **change})
    proc = _run("--placed-input", path, "--specimen", "WKY-A", "--output-dir", tmp_path / "run")
    _refusal(proc, named)


def test_a_map_that_is_not_the_sidecars_size_is_refused(tmp_path):
    path = tmp_path / "wrong_size.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(PROBABILITIES_DATASET, data=_synthetic_map((48, 48, 40)))
    _write_sidecar(path, _record())
    out = tmp_path / "run"
    proc = _run("--placed-input", path, "--output-dir", out)
    _refusal(proc, "(48, 48, 40, 2)", "(48, 48, 48)", "size_zyx")
    assert not (out / ROI_RECORD_NAME).exists()


def test_a_map_without_the_probability_dataset_is_refused(tmp_path):
    path = tmp_path / "wrong_dataset.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("other", data=_synthetic_map())
    _write_sidecar(path, _record())
    proc = _run("--placed-input", path, "--output-dir", tmp_path / "run")
    _refusal(proc, PROBABILITIES_DATASET, "other")


def test_help_describes_the_flag():
    proc = _run("--help")
    assert proc.returncode == 0
    assert "--placed-input" in proc.stdout
    assert "sidecar" in proc.stdout
