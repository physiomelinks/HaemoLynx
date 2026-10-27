"""The benchmark sub-volume: committed, well-formed, and rebuildable from the real map.

``examples/make_benchmark_subvolume.py`` is the only writer. The fast tests check the committed
files and the script's refusals; the ``realdata`` test rebuilds the cut from the map on disk.
"""
import json
import subprocess

import h5py
import numpy as np
import pytest

import make_benchmark_subvolume as mk
from conftest import TOY_ENV, toy_git
from ImageLynx import cb_settings, specimens
from ImageLynx.roi_placement import (
    ROI_RECORD_KEYS, ROI_RECORD_NAME, check_placed_input_shape, placed_input_sidecar_path,
    read_placed_input_record,
)

PATH = mk.SUBVOLUME_PATH
SIDECAR = placed_input_sidecar_path(PATH)
BATCH_RECORD = specimens.get_specimen(mk.SPECIMEN_ID).batch_run_dir / ROI_RECORD_NAME


def _sidecar():
    return json.loads(SIDECAR.read_text())


def test_committed_file_is_one_float32_gzip_dataset_of_the_roi_shape():
    with h5py.File(PATH, "r") as handle:
        assert list(handle.keys()) == [mk.DATASET]
        dataset = handle[mk.DATASET]
        assert dataset.shape == (*cb_settings.ROI_VOXELS, 2)
        assert dataset.dtype == np.float32
        assert dataset.compression == "gzip"
    assert PATH.stat().st_size < 33e6


def test_sidecar_is_a_placed_input_record_with_the_provenance_fields():
    data = _sidecar()
    assert all(key in data for key in ROI_RECORD_KEYS)
    record = read_placed_input_record(PATH)
    assert record == {key: data[key] for key in ROI_RECORD_KEYS}
    check_placed_input_shape(PATH, record, mk.DATASET)
    assert record["specimen_id"] == mk.SPECIMEN_ID
    assert data["source"] != data["source_path"]
    assert len(data["source_sha256"]) == 64
    assert data["classifier_provenance"]["classifier_sha256"]
    assert len(data["commit"]) == 40


@pytest.mark.skipif(not BATCH_RECORD.exists(), reason="needs the WKY-A batch run on disk")
def test_sidecar_bounds_equal_the_batch_runs_placement():
    batch = json.loads(BATCH_RECORD.read_text())
    assert _sidecar()["bounds_zyx"] == batch["bounds_zyx"]


def _toy_repo(tmp_path):
    tmp_path.mkdir()
    toy_git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a")
    toy_git(tmp_path, "add", "f.txt")
    toy_git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _toy_git(repo):
    return lambda *args: subprocess.run(["git", *args], cwd=repo, env=TOY_ENV, check=True,
                                        capture_output=True, text=True).stdout


def _fake_build():
    cut = np.zeros((4, 4, 4, 2), dtype=np.float32)
    return cut, {"specimen_id": "X"}


def test_make_refuses_a_tree_with_tracked_changes(tmp_path):
    repo = _toy_repo(tmp_path / "repo")
    (repo / "f.txt").write_text("changed")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="tracked changes"):
        mk.make(directory=out, git=_toy_git(repo), build=_fake_build)
    assert not out.exists()


def test_make_refuses_an_existing_output_without_overwrite(tmp_path):
    repo = _toy_repo(tmp_path / "repo")
    out = tmp_path / "out"
    git = _toy_git(repo)
    written = mk.make(directory=out, git=git, build=_fake_build)
    assert written.exists() and placed_input_sidecar_path(written).exists()
    with pytest.raises(FileExistsError, match="--overwrite"):
        mk.make(directory=out, git=git, build=_fake_build)
    assert mk.make(overwrite=True, directory=out, git=git, build=_fake_build) == written
    assert json.loads(placed_input_sidecar_path(written).read_text())["commit"] == \
        git("rev-parse", "HEAD").strip()


@pytest.mark.realdata
def test_committed_subvolume_is_rebuilt_from_the_real_map():
    specimen = specimens.get_specimen(mk.SPECIMEN_ID)
    if not specimen.probabilities_path.exists():
        pytest.skip(f"no probability map at {specimen.probabilities_path}")
    cut, record = mk.build_subvolume(specimen)
    with h5py.File(PATH, "r") as handle:
        committed = handle[mk.DATASET][...]
    assert cut.shape == committed.shape and cut.dtype == committed.dtype
    assert np.array_equal(cut, committed)
    data = _sidecar()
    assert {key: data[key] for key in record} == record
