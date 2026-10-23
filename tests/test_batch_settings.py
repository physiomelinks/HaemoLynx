"""One image's settings in a batch run, and the batch's own checks
(`pipeline/batch.py`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.io import BatchFolders, match_batch
from haemolynx.pipeline import default_schema
from haemolynx.pipeline import batch
from haemolynx.pipeline.batch import (
    NOT_WRITTEN_BY_A_BATCH,
    WINDOWS_MAX_FILE_PATH,
    WINDOWS_MAX_FOLDER_PATH,
    WRITTEN_WHILE,
    batch_mask_sizes,
    batch_preflight,
    check_batch,
    check_batch_item,
    check_output_path_lengths,
    default_output_root,
    item_settings,
    output_paths,
)


@pytest.fixture(scope="module")
def schema():
    return default_schema()


@pytest.fixture
def base(schema):
    return schema.defaults()


def _match(tmp_path: Path, *names: str, sizes=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return match_batch(BatchFolders(tmp_path), sizes)


def test_item_settings_change_only_the_inputs_and_outputs(tmp_path, base):
    (match,) = _match(tmp_path / "in", "E14_seg.tif", "E14.tif")
    out = tmp_path / "out"
    settings = item_settings(base, match, out)
    assert settings["input_path"] == tmp_path / "in" / "E14_seg.tif"
    assert settings["raw_image_path"] == tmp_path / "in" / "E14.tif"
    assert settings["vtk_output_prefix"] == out / "E14_seg" / "resistance_network"
    assert settings["base_plot_dir"] == out / "E14_seg" / "plots"
    # Unset, it follows the VTK output into the image's folder.
    assert settings["perturbation_output_dir"] is None
    configured = item_settings({**base, "perturbation_output_dir": "elsewhere"}, match, out)
    assert configured["perturbation_output_dir"] == out / "E14_seg" / "perturbations"
    assert settings["mid_run_postprocessing"] is False
    changed = {name for name in base if settings[name] != base[name]}
    assert changed <= {
        "input_path", "raw_image_path", "vtk_output_prefix", "base_plot_dir",
        "perturbation_output_dir", "sweep_output_dir", "ilastik_output_dir",
    }


def test_default_output_root_is_the_vtk_prefixs_folder():
    assert default_output_root({"vtk_output_prefix": "outputs/run"}) == Path("outputs")
    assert default_output_root({"vtk_output_prefix": "run"}) is None
    assert default_output_root({}) is None


def test_mask_sizes_follow_the_roles_in_use(base):
    assert batch_mask_sizes({**base, "automated_vessel_assignment": False}) == ()
    on = {**base, "automated_vessel_assignment": True, "use_large_vessel_masks": True}
    assert "large" in batch_mask_sizes(on)
    assert "large" not in batch_mask_sizes({**on, "use_ilastik_large_vessel_segmentation": True})


def test_the_defaults_pass_the_batch_checks(base, schema):
    report = check_batch(base, schema)
    assert report.errors == []


def test_node_ids_are_refused(base, schema):
    settings = {**base, "inlet_node_selection_method": "node_ids", "automated_vessel_assignment": False}
    assert any("node_ids" in error for error in check_batch(settings, schema).errors)


def test_coordinates_warn(base, schema):
    settings = {
        **base,
        "automated_vessel_assignment": False,
        "inlet_node_selection_method": "coordinates",
        "inlet_node_coordinates": [[1.0, 2.0, 3.0]],
    }
    report = check_batch(settings, schema)
    assert report.errors == []
    assert any("same positions" in warning for warning in report.warnings)


def test_a_single_per_image_file_in_effect_is_refused(base, schema, tmp_path):
    settings = {**base, "measurement_3d_to_cell_mask": True, "cell_mask_path": tmp_path / "cells.tif"}
    assert any("cell_mask_path" in error for error in check_batch(settings, schema).errors)
    # Set but not read: nothing to refuse.
    settings["measurement_3d_to_cell_mask"] = False
    assert not any("cell_mask_path" in error for error in check_batch(settings, schema).errors)


def test_synthetic_and_ilastik_runs_are_refused(base, schema):
    assert check_batch({**base, "network_source": "synthetic"}, schema).errors
    assert check_batch({**base, "use_ilastik_segmentation": True}, schema).errors


def test_mid_run_pause_and_skipped_stages_warn(base, schema):
    report = check_batch({**base, "mid_run_postprocessing": True, "do_skeletonize": False}, schema)
    assert any("cannot pause" in warning for warning in report.warnings)
    assert any("do_skeletonize" in warning for warning in report.warnings)


def test_a_match_problem_is_an_error_and_a_missing_raw_a_warning(tmp_path, base):
    (match,) = _match(tmp_path, "E14_seg.tif", sizes=["large"])
    report = check_batch_item(match, {**base, "measure_diameters_from_raw_image": True})
    assert report.errors == [f"No large vessel masks found in {tmp_path}."]
    assert any("measured from the mask" in warning for warning in report.warnings)


# -- output path lengths ---------------------------------------------------------

LIMITS = (WINDOWS_MAX_FILE_PATH, WINDOWS_MAX_FOLDER_PATH)


def _longest_file(settings) -> int:
    files, _folders = output_paths(settings)
    return max(len(str(path)) for path in files)


def _settings_with_longest_file(tmp_path, base, length: int):
    """One image's settings whose longest output file path is *length*
    characters: the output folder's name padded to it."""
    (match,) = _match(tmp_path / "seg", "E14_MCA_Simple Segmentation.tif")
    short = _longest_file(item_settings(base, match, tmp_path / "o"))
    root = tmp_path / ("o" * (1 + length - short))
    settings = item_settings(base, match, root)
    assert _longest_file(settings) == length
    return settings


def test_an_output_file_path_past_windows_limit_is_an_error(tmp_path, base):
    at_limit = _settings_with_longest_file(tmp_path, base, WINDOWS_MAX_FILE_PATH)
    assert check_output_path_lengths(at_limit, limits=LIMITS).ok
    past = _settings_with_longest_file(tmp_path, base, WINDOWS_MAX_FILE_PATH + 1)
    report = check_output_path_lengths(past, limits=LIMITS)
    (error,) = report.errors
    assert f"{WINDOWS_MAX_FILE_PATH + 1}-character file path" in error
    assert "_checkpoint_build_haemodynamic_model.pkl" in error
    assert "long paths" in error


def test_an_output_folder_past_windows_limit_is_an_error(tmp_path, base):
    (match,) = _match(tmp_path / "seg", "E14_seg.tif")
    settings = item_settings(base, match, tmp_path / "out")
    _files, folders = output_paths(settings)
    assert Path(settings["vtk_output_prefix"]).parent.absolute() in folders
    folder = max(len(str(path)) for path in folders)
    assert check_output_path_lengths(settings, limits=(10_000, folder)).ok
    (error,) = check_output_path_lengths(settings, limits=(10_000, folder - 1)).errors
    assert f"{folder}-character folder path" in error


def test_with_no_limit_any_path_passes(tmp_path, base, monkeypatch):
    past = _settings_with_longest_file(tmp_path, base, 400)
    monkeypatch.setattr(batch, "path_length_limits", lambda: None)
    assert check_output_path_lengths(past).ok


def test_the_paths_counted_are_the_files_these_settings_write(tmp_path, base):
    (match,) = _match(tmp_path / "seg", "E14_seg.tif")

    def names(**changes):
        files, _folders = output_paths(item_settings({**base, **changes}, match, tmp_path / "out"))
        return {path.name for path in files}

    plain = names(statistics=False, measurement_3d_to_cell_mask=False)
    assert "E14_seg_checkpoint_build_haemodynamic_model.pkl" in plain
    assert "E14_seg.haemorun.part" in plain
    assert "E14_seg_run_manifest.json" in plain
    assert "E14_seg_statistics.csv" not in plain
    assert "E14_seg_cell_to_vessel_3d_dual_distances_summary.csv" not in plain
    # Written by hand in the panel, never by a batch.
    assert not {name for name in plain if name.endswith(("_review.csv", "_connectivity.csv"))}
    with_statistics = names(statistics=True)
    assert "E14_seg_betweenness_communities_edge_length.json" in with_statistics
    assert "E14_seg_cell_to_vessel_3d_dual_distances_summary.csv" in names(
        measurement_3d_to_cell_mask=True
    )


def test_each_perturbations_folder_and_files_are_counted(tmp_path, base):
    (match,) = _match(tmp_path / "seg", "E14_seg.tif")
    settings = item_settings(
        {
            **base,
            "run_perturbations": True,
            "statistics": True,
            "perturbations": [{"name": "block10", "type": "capillary_block"}],
        },
        match,
        tmp_path / "out",
    )
    files, folders = output_paths(settings)
    own = (tmp_path / "out" / "E14_seg" / "block10_capillary_block").absolute()
    assert own in folders
    assert own / "block10_summary.csv" in files
    assert own / "block10_betweenness_communities_edge_flow.json" in files


def test_the_written_while_table_names_real_files_and_settings(schema):
    from haemolynx.pipeline.export_dictionary import FILES, RUN_FOLDER

    run_suffixes = {
        name[len("{stem}"):]
        for entry in FILES
        if entry.where == RUN_FOLDER
        for name in entry.name.split(", ")
        if name.startswith("{stem}_")
    }
    assert set(WRITTEN_WHILE) <= run_suffixes
    assert set(NOT_WRITTEN_BY_A_BATCH) <= run_suffixes
    assert all(setting in schema for setting in WRITTEN_WHILE.values())


def test_long_paths_turned_on_lift_the_limit(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(sys, "platform", "linux")
    assert batch.path_length_limits() is None

    def registry(enabled):
        class Key:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        return types.SimpleNamespace(
            HKEY_LOCAL_MACHINE=object(),
            OpenKey=lambda *_args: Key(),
            QueryValueEx=lambda _key, name: (enabled, 4),
        )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", registry(1))
    assert batch.path_length_limits() is None
    monkeypatch.setitem(sys.modules, "winreg", registry(0))
    assert batch.path_length_limits() == LIMITS


def test_batch_preflight_checks_the_paths_and_prints_nothing(tmp_path, base, schema, monkeypatch, capsys):
    (match,) = _match(tmp_path / "seg", "E14_seg.tif")
    monkeypatch.setattr(batch, "path_length_limits", lambda: (40, 40))
    report = batch_preflight(base, match, tmp_path / "out", schema)
    assert any("Windows allows (40)" in error for error in report.errors)
    assert capsys.readouterr().out == ""
