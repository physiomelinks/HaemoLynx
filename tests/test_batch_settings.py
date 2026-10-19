"""One image's settings in a batch run, and the batch's own checks
(`pipeline/batch.py`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.io import BatchFolders, match_batch
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.batch import (
    batch_mask_sizes,
    check_batch,
    check_batch_item,
    default_output_root,
    item_settings,
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
