"""The manifest a run leaves beside its outputs (`pipeline/provenance.py`).

Without it an output folder cannot say which settings, which version and which
commit made it, so a result from before a default or a law changed looks like
one from after. These tests pin what the manifest holds, that every staged run
writes it -- on the seven-vessel fixture, and through a failure -- and that
nothing about writing it (no git, an unwritable folder) can fail a run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import haemolynx
from haemolynx.parsers import load_config
from haemolynx.pipeline import (
    default_schema,
    fill_derived_settings,
    record_run,
    resolve_settings,
    run_pipeline_stages,
    stages,
)
from haemolynx.pipeline import provenance
from haemolynx.pipeline.progress import STAGE_STARTED, STAGES
from haemolynx.pipeline.provenance import (
    KEY_DEPENDENCIES,
    MANIFEST_SUFFIX,
    SETTINGS_SUFFIX,
    current_rss_bytes,
    file_fingerprint,
    git_state,
    peak_rss_bytes,
    read_git_state,
)
from haemolynx.pipeline.stages import HaemodynamicModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"
STAGE_CALLS = [stage.call for stage in STAGES if stage.call]
SCHEMA = default_schema()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(folder: Path, stem: str) -> dict:
    return json.loads((folder / f"{stem}{MANIFEST_SUFFIX}").read_text(encoding="utf-8"))


def _settings(tmp_path: Path, **overrides) -> dict:
    """Every setting at its default, the image a small file of our own."""
    image = tmp_path / "image.tif"
    if not image.exists():
        image.write_bytes(b"not really a tiff, but bytes to fingerprint" * 100)
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": image,
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            **overrides,
        }
    )
    return resolve_settings(values, schema=SCHEMA, config_path=None)


@pytest.fixture
def fresh_git_state():
    """`git_state` is read once per process; a test that changes git clears it."""
    git_state.cache_clear()
    yield
    git_state.cache_clear()


# --- what the manifest holds ---------------------------------------------------


def test_the_manifest_holds_the_version_environment_inputs_and_stages(tmp_path):
    settings = _settings(tmp_path)
    with record_run(settings, SCHEMA) as run:
        with run.stage("segment"):
            pass
        with run.stage("skeletonise"):
            np.ones(1000).sum()

    manifest = _read(tmp_path / "out", "image")

    assert manifest["manifest_version"] == provenance.MANIFEST_VERSION
    assert manifest["haemolynx_version"] == haemolynx.__version__
    assert manifest["status"] == "finished" and manifest["error"] is None
    assert manifest["python"]["version"] == platform.python_version()
    assert manifest["python"]["executable"] == sys.executable
    assert set(manifest["dependencies"]) == set(KEY_DEPENDENCIES)
    assert manifest["dependencies"]["numpy"] == np.__version__
    assert manifest["dependencies"]["networkx"] == nx.__version__
    assert set(manifest["git"]) >= {"commit", "dirty", "branch", "describe"}
    assert manifest["memory_method"] == provenance.MEMORY_METHOD

    (image,) = manifest["inputs"]
    assert image["setting"] == "input_path"
    assert Path(image["path"]) == (tmp_path / "image.tif").resolve()
    assert image["size_bytes"] == (tmp_path / "image.tif").stat().st_size
    assert image["checksum"] == f"sha256:{_sha256(tmp_path / 'image.tif')}"
    assert image["checksum_method"] == "sha256 of the whole file"
    datetime.fromisoformat(image["modified"])

    assert [stage["stage"] for stage in manifest["stages"]] == ["segment", "skeletonise"]
    assert manifest["stages_finished"] == ["segment", "skeletonise"]
    for stage in manifest["stages"]:
        assert stage["status"] == "finished"
        assert stage["resumed"] is False
        started = datetime.fromisoformat(stage["started"])
        finished = datetime.fromisoformat(stage["finished"])
        assert started <= finished
        assert stage["wall_seconds"] >= 0
    assert manifest["stages"][0]["title"] == "1. Input"


def test_the_log_says_where_the_manifest_went_and_each_stages_time(tmp_path, caplog):
    """So a command-line run, which shows no progress unless asked, still logs
    every stage's wall time and the peak memory."""
    settings = _settings(tmp_path)
    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.provenance"):
        with record_run(settings, SCHEMA) as run:
            with run.stage("segment"):
                pass
            with run.stage("skeletonise"):
                pass

    (line,) = [r.getMessage() for r in caplog.records if "Run manifest" in r.getMessage()]
    assert str(tmp_path / "out" / f"image{MANIFEST_SUFFIX}") in line
    assert "segment 0.0 s, skeletonise 0.0 s" in line
    assert "peak memory" in line


def test_the_settings_are_recorded_upgraded_and_as_the_run_started(tmp_path):
    """A retired setting is recorded as the one that replaced it, and a value
    a stage writes back into the settings mid-run is not what is recorded."""
    settings = _settings(tmp_path)
    settings["diameter_basis"] = "anatomical"  # retired; read into segmentation_source
    expected = SCHEMA.upgrade(settings)["segmentation_source"]

    with record_run(settings, SCHEMA) as run:
        with run.stage("segment"):
            settings["inlet_nodes"][:] = [7, 8]  # what assign_boundaries does

    recorded = _read(tmp_path / "out", "image")["settings"]
    assert "diameter_basis" not in recorded
    assert expected != SCHEMA["segmentation_source"].default
    assert recorded["segmentation_source"] == expected
    assert recorded["inlet_nodes"] == []
    assert recorded["input_path"] == (tmp_path / "image.tif").as_posix()


def test_the_settings_file_loads_back_as_the_settings_that_ran(tmp_path):
    settings = _settings(tmp_path, viscosity_law="constant")
    before = copy.deepcopy(settings)
    with record_run(settings, SCHEMA) as run:
        with run.stage("segment"):
            pass

    manifest = _read(tmp_path / "out", "image")
    assert manifest["settings_file"] == f"image{SETTINGS_SUFFIX}"
    loaded = load_config(tmp_path / "out" / manifest["settings_file"], SCHEMA)
    fill_derived_settings(loaded)
    assert {name: loaded[name] for name in SCHEMA.names} == {
        name: before[name] for name in SCHEMA.names
    }


def test_a_stage_that_raises_the_process_peak_records_its_memory(tmp_path):
    before = current_rss_bytes()
    if before is None:
        pytest.skip("the current resident size cannot be read here (macOS)")
    settings = _settings(tmp_path)
    with record_run(settings, SCHEMA) as run:
        with run.stage("segment"):
            held = np.ones(25_000_000)  # 200 MB, touched, held to the stage's end
            held[::4096] = 2.0

    (stage,) = _read(tmp_path / "out", "image")["stages"]
    assert stage["peak_rss_method"] in {"process_peak", "sampled"}
    assert stage["peak_rss_bytes"] >= before + 150_000_000
    assert stage["process_peak_rss_bytes"] >= stage["peak_rss_bytes"]
    del held


def test_the_process_peak_is_read_with_the_standard_library():
    peak = peak_rss_bytes()
    assert isinstance(peak, int) and peak > 0
    current = current_rss_bytes()
    if current is not None:
        assert 0 < current <= peak
    if sys.platform == "win32":
        assert provenance.worker_peak_rss_bytes() is None


def test_stages_before_the_one_a_run_resumes_at_are_marked_resumed(tmp_path):
    settings = _settings(tmp_path)
    with record_run(settings, SCHEMA, start_from="solve") as run:
        for name in ("segment", "build_haemodynamic_model", "solve"):
            with run.stage(name):
                pass

    manifest = _read(tmp_path / "out", "image")
    assert manifest["start_from"] == "solve"
    assert [(stage["stage"], stage["resumed"]) for stage in manifest["stages"]] == [
        ("segment", True),
        ("build_haemodynamic_model", True),
        ("solve", False),
    ]


# --- input fingerprints --------------------------------------------------------


def test_a_small_file_is_checksummed_whole(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(bytes(range(256)) * 10)
    record = file_fingerprint(path)
    assert record["checksum"] == f"sha256:{_sha256(path)}"
    assert record["checksum_bytes_read"] == path.stat().st_size


def test_a_large_file_is_fingerprinted_from_a_fixed_number_of_blocks(tmp_path):
    """Reading a fixed amount, whatever the size: and what that does not see."""
    block = 64 * 1024
    options = {"full_hash_max_bytes": 2**20, "blocks": 4, "block_bytes": block}
    data = bytearray(np.random.default_rng(0).integers(0, 256, 3 * 2**20, dtype=np.uint8))
    path = tmp_path / "stack.bin"
    path.write_bytes(bytes(data))

    record = file_fingerprint(path, **options)
    assert record["checksum"].startswith("sha256-sampled:")
    assert record["checksum_bytes_read"] == 4 * block
    assert "4 blocks" in record["checksum_method"]

    def checksum_after(change) -> str:
        changed = bytearray(data)
        change(changed)
        path.write_bytes(bytes(changed))
        return file_fingerprint(path, **options)["checksum"]

    def flip(offset):
        return lambda buffer: buffer.__setitem__(offset, buffer[offset] ^ 0xFF)

    assert checksum_after(flip(0)) != record["checksum"]  # in the first block
    assert checksum_after(flip(len(data) - 1)) != record["checksum"]  # the last
    assert checksum_after(lambda buffer: buffer.append(0)) != record["checksum"]  # size
    # Between two blocks: the documented blind spot of a sampled fingerprint.
    assert checksum_after(flip(block + 10)) == record["checksum"]


def test_every_input_file_in_effect_is_fingerprinted_once(tmp_path):
    mask = tmp_path / "pericytes.tif"
    mask.write_bytes(b"pericyte mask")
    unread = tmp_path / "cells.tif"
    unread.write_bytes(b"read only with the cell-mask distances on")
    settings = _settings(
        tmp_path,
        use_pericyte_mask_constriction=True,
        pericyte_mask_path=mask,
        cell_mask_path=unread,
        measurement_3d_to_cell_mask=False,
        tissue_raw_tiff_path=tmp_path / "image.tif",  # the same file as input_path
        measure_tissue_volume=True,
    )

    records = provenance.input_files(settings, SCHEMA)

    assert [record["setting"] for record in records] == [
        "input_path",
        "pericyte_mask_path",
    ]
    assert records[1]["checksum"] == f"sha256:{_sha256(mask)}"


# --- a staged run on fixture data ----------------------------------------------


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    """`run_pipeline_stages` on the seven-vessel fixture, every stage real."""
    folder = tmp_path_factory.mktemp("fixture_run")
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": FIXTURE,
            "vtk_output_prefix": folder / "run",
            "plot_dir": folder / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    before = copy.deepcopy(settings)
    graph = run_pipeline_stages(settings, SCHEMA)
    return folder, before, graph


def test_a_staged_run_writes_its_manifest_beside_its_outputs(fixture_run):
    folder, _before, graph = fixture_run
    assert graph.number_of_edges() > 0

    manifest = _read(folder, FIXTURE.stem)
    assert (folder / f"{FIXTURE.stem}_graph.pkl").is_file()  # beside the other outputs
    assert manifest["status"] == "finished"
    assert manifest["stages_finished"] == STAGE_CALLS
    assert [stage["stage"] for stage in manifest["stages"]] == STAGE_CALLS
    assert all(stage["wall_seconds"] > 0 for stage in manifest["stages"][1:3])
    assert sum(stage["wall_seconds"] for stage in manifest["stages"]) <= manifest["wall_seconds"]
    (image,) = manifest["inputs"]
    assert image["checksum"] == f"sha256:{_sha256(FIXTURE)}"
    assert image["size_bytes"] == FIXTURE.stat().st_size
    if peak_rss_bytes() is not None:
        assert all(stage["peak_rss_bytes"] > 0 for stage in manifest["stages"])


def test_a_staged_runs_settings_file_reruns_the_same_settings(fixture_run):
    folder, before, _graph = fixture_run
    loaded = load_config(folder / f"{FIXTURE.stem}{SETTINGS_SUFFIX}", SCHEMA)
    fill_derived_settings(loaded)
    assert {name: loaded[name] for name in SCHEMA.names} == {
        name: before[name] for name in SCHEMA.names
    }


def test_the_example_pipeline_writes_a_manifest_too(tmp_path, monkeypatch):
    """The CLI example calls the stages itself, inside `record_run`."""
    import importlib.util

    path = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
    spec = importlib.util.spec_from_file_location("resistance_network_pipeline", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    model = HaemodynamicModel(graph=nx.MultiGraph())
    for name in (
        "segment", "skeletonise", "build_network", "assign_boundaries", "assign_diameters",
        "apply_network_handling", "build_haemodynamic_model", "solve", "run_perturbations",
        "export_results",
    ):
        monkeypatch.setattr(example, name, lambda *_args, **_kwargs: model)
    image = tmp_path / "image.tif"
    image.write_bytes(b"bytes")

    example.image_to_model_pipeline(input_path=image, vtk_output_prefix=tmp_path / "run")

    manifest = _read(tmp_path, "image")
    assert manifest["stages_finished"] == [
        name for name in STAGE_CALLS if name != "post_process"
    ]


# --- a run that fails ----------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch):
    """Every stage replaced by a stub; *fail_at* raises."""

    def install(fail_at=None):
        model = HaemodynamicModel(graph=nx.MultiGraph())

        def stub(name):
            def call(*_args, **_kwargs):
                if name == fail_at:
                    raise RuntimeError(f"{name} broke")
                return model

            return call

        for name in STAGE_CALLS:
            monkeypatch.setattr(stages, name, stub(name))
        monkeypatch.setattr(stages, "apply_network_handling", lambda *a, **k: model)

    return install


def test_a_stage_failure_still_writes_a_partial_manifest(tmp_path, stubbed):
    stubbed(fail_at="build_network")
    settings = _settings(tmp_path)

    with pytest.raises(RuntimeError, match="build_network broke"):
        run_pipeline_stages(settings, SCHEMA)

    manifest = _read(tmp_path / "out", "image")
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "build_network broke",
        "stage": "build_network",
    }
    assert manifest["stages_finished"] == ["segment", "skeletonise"]
    assert [(stage["stage"], stage["status"]) for stage in manifest["stages"]] == [
        ("segment", "finished"),
        ("skeletonise", "finished"),
        ("build_network", "failed"),
    ]
    assert manifest["stages"][-1]["wall_seconds"] is not None
    assert manifest["inputs"][0]["setting"] == "input_path"
    assert (tmp_path / "out" / f"image{SETTINGS_SUFFIX}").is_file()


def test_a_run_stopped_as_a_stage_starts_records_that_stage_interrupted(tmp_path, stubbed):
    """The panel's cancel raises from the progress callback, outside the stage."""
    stubbed()

    class Stopped(Exception):
        pass

    def stop_at_solve(event):
        if event.kind == STAGE_STARTED and event.stage == "solve":
            raise Stopped("cancelled")

    with pytest.raises(Stopped):
        run_pipeline_stages(_settings(tmp_path), SCHEMA, progress=stop_at_solve)

    manifest = _read(tmp_path / "out", "image")
    assert manifest["status"] == "failed"
    assert manifest["error"]["stage"] == "solve"
    assert manifest["stages"][-1]["stage"] == "solve"
    assert manifest["stages"][-1]["status"] == "interrupted"
    assert "solve" not in manifest["stages_finished"]


def test_a_run_that_fails_in_its_first_stage_writes_no_manifest(tmp_path, stubbed):
    """Nothing was written to describe; and an old manifest is left alone."""
    stubbed(fail_at="segment")
    folder = tmp_path / "out"
    folder.mkdir()
    earlier = folder / f"image{MANIFEST_SUFFIX}"
    earlier.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_pipeline_stages(_settings(tmp_path), SCHEMA)

    assert earlier.read_text(encoding="utf-8") == "{}"


def test_a_manifest_that_cannot_be_written_never_fails_the_run(tmp_path, stubbed, caplog):
    stubbed()
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder", encoding="utf-8")
    settings = _settings(tmp_path, vtk_output_prefix=blocker / "out" / "run")

    with caplog.at_level(logging.WARNING, logger="haemolynx.pipeline.provenance"):
        graph = run_pipeline_stages(settings, SCHEMA)

    assert graph is not None
    assert any("Could not write the run manifest" in r.getMessage() for r in caplog.records)


def test_a_run_with_no_output_folder_or_schema_writes_nothing_and_runs(stubbed, tmp_path):
    stubbed()
    assert run_pipeline_stages({}, schema=None) is not None


# --- git -----------------------------------------------------------------------


def test_no_git_executable_leaves_the_commit_null(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    package = tmp_path / "src" / "haemolynx"
    package.mkdir(parents=True)

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("[WinError 2] The system cannot find the file specified")

    monkeypatch.setattr(provenance.subprocess, "run", no_git)

    state = read_git_state(package)
    assert {key: state[key] for key in ("commit", "dirty", "branch", "describe")} == {
        "commit": None,
        "dirty": None,
        "branch": None,
        "describe": None,
    }


def test_an_installed_package_never_calls_git(tmp_path, monkeypatch):
    """No repository above a wheel's site-packages: nothing to ask git."""
    package = tmp_path / "site-packages" / "haemolynx"
    package.mkdir(parents=True)

    def refuse(*_args, **_kwargs):
        raise AssertionError("git must not be called outside a checkout")

    monkeypatch.setattr(provenance.subprocess, "run", refuse)
    assert read_git_state(package)["commit"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_package_inside_a_checkout_it_is_not_tracked_by_has_no_commit(tmp_path):
    """A wheel installed into a venv inside a checkout is not that checkout's code."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / ".venv" / "site-packages" / "haemolynx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    assert read_git_state(package)["commit"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_from_a_checkout_the_commit_is_heads(fresh_git_state):
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=provenance.PACKAGE_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        pytest.skip("the package is not running from a git checkout")

    state = git_state()
    assert state["commit"] == head
    assert isinstance(state["dirty"], bool)
    assert state["dirty_path_count"] >= len(state["dirty_paths"])
    assert all(not path.startswith(" ") for path in state["dirty_paths"])


def test_a_run_without_git_still_writes_its_manifest(
    tmp_path, stubbed, monkeypatch, fresh_git_state
):
    stubbed()

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance, "_git", no_git)

    run_pipeline_stages(_settings(tmp_path), SCHEMA)

    manifest = _read(tmp_path / "out", "image")
    assert manifest["status"] == "finished"
    assert manifest["git"]["commit"] is None
    assert manifest["git"]["dirty"] is None
    assert manifest["haemolynx_version"] == haemolynx.__version__
