"""Pre-run checks derived from the schema."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from haemolynx.parsers import ConfigError, Schema, Setting, check_settings  # noqa: E402
from haemolynx.parsers.checks import resolve_existing_path  # noqa: E402


def _schema() -> Schema:
    return Schema(
        [
            Setting("use_ilastik", "bool", False, "Segment with ilastik", "S"),
            Setting(
                "input_path", "path", None, "Read this segmented image", "S",
                requires=("!use_ilastik",), must_exist=True,
            ),
            Setting(
                "classifier_path", "path", None, "Use this trained classifier", "S",
                requires=("use_ilastik",), must_exist=True,
            ),
            Setting("output_dir", "path", "out", "Write results here", "S"),
        ]
    )


# --- what must exist, and when ---------------------------------------------


def test_a_required_path_that_exists_passes(tmp_path):
    existing = tmp_path / "mask.tif"
    existing.write_bytes(b"x")
    report = check_settings(_schema(), {"use_ilastik": False, "input_path": existing})
    assert report.ok
    assert report.passed[0][0] == "input_path"


def test_a_required_path_that_is_missing_is_an_error_naming_the_setting(tmp_path):
    report = check_settings(
        _schema(), {"use_ilastik": False, "input_path": tmp_path / "absent.tif"}
    )
    assert not report.ok
    assert "input_path" in report.errors[0]
    assert "absent.tif" in report.errors[0]
    assert "clear it" not in report.errors[0]


def test_a_path_the_run_would_generate_is_not_demanded(tmp_path):
    """With ilastik on, the segmented input is an output, not an input."""
    existing = tmp_path / "raw.tif"
    existing.write_bytes(b"x")
    report = check_settings(
        _schema(),
        {"use_ilastik": True, "input_path": None, "classifier_path": existing},
    )
    assert report.ok, report.errors


def test_a_path_a_switched_off_feature_needs_is_not_demanded():
    """The classifier only matters when ilastik is on."""
    report = check_settings(_schema(), {"use_ilastik": False, "input_path": None})
    assert [e for e in report.errors if "classifier" in e] == []


def test_an_unset_but_required_path_says_why_it_is_required():
    report = check_settings(_schema(), {"use_ilastik": True, "classifier_path": None})
    assert any("'use_ilastik' is on" in message for message in report.errors)


def test_paths_without_must_exist_are_not_checked(tmp_path):
    """Output directories are created by a run, not required before it."""
    report = check_settings(
        _schema(),
        {"use_ilastik": True, "classifier_path": None, "output_dir": tmp_path / "nope"},
    )
    assert not any("output_dir" in message for message in report.errors)


def _schema_with_an_optional_path() -> Schema:
    return Schema([
        Setting("measure", "bool", False, "Measure the tissue", "S"),
        Setting(
            "tissue_path", "path", None, "Find the tissue in this image", "S",
            requires=("measure",), must_exist=True, placeholder="the raw image",
        ),
    ])


def test_an_unset_path_with_a_placeholder_is_optional():
    """Unset, the run reads what the placeholder names: not an error."""
    report = check_settings(_schema_with_an_optional_path(), {"measure": True, "tissue_path": None})
    assert report.ok, report.errors
    assert report.passed == [("tissue_path", "not set: uses the raw image")]


def test_an_optional_path_that_is_set_must_still_exist(tmp_path):
    schema = _schema_with_an_optional_path()
    report = check_settings(schema, {"measure": True, "tissue_path": tmp_path / "gone.tif"})
    assert len(report.errors) == 1
    assert report.errors[0].startswith("tissue_path: ") and "gone.tif" in report.errors[0]
    assert report.errors[0].endswith(", or clear it to use the raw image.")

    existing = tmp_path / "tissue.tif"
    existing.write_bytes(b"x")
    report = check_settings(schema, {"measure": True, "tissue_path": existing})
    assert report.ok
    assert report.passed == [("tissue_path", str(existing))]


def test_an_optional_path_is_not_checked_while_its_feature_is_off(tmp_path):
    report = check_settings(
        _schema_with_an_optional_path(), {"measure": False, "tissue_path": tmp_path / "gone.tif"}
    )
    assert report.ok and report.passed == []


def test_a_placeholder_makes_only_a_path_optional():
    path = Setting("p", "path", None, "A path", "S", must_exist=True, placeholder="the default")
    assert path.optional_path
    assert not Setting("p", "path", None, "A path", "S", must_exist=True).optional_path
    assert not Setting("w", "float", None, "A width", "S", placeholder="auto").optional_path


def test_a_config_file_says_an_optional_path_must_exist_only_if_set():
    from haemolynx.parsers.config import _comment_lines

    optional = _schema_with_an_optional_path()["tissue_path"]
    required = _schema()["input_path"]
    assert _comment_lines(optional)[0].endswith("must exist if set]")
    assert _comment_lines(required)[0].endswith("must exist]")


def test_every_optional_must_exist_path_in_the_pipeline_falls_back_when_unset():
    """A placeholder on a must_exist path turns its unset check off, so each
    one is listed here with what it falls back to, checked in the run:
    tissue_raw_tiff_path to the raw image (pipeline.checks.tissue_raw_image,
    tests/test_tissue_volume.py), measurement_3d_vessel_mask_path to the run's
    segmentation (stages.export_results, tests/test_three_dim_distances.py),
    and the four venule masks and classifiers to the arteriole/global one
    read as a global mask (io.vessel_masks_are_global,
    tests/test_global_vessel_masks.py). A new one has to be added here, its
    fallback checked."""
    from haemolynx.pipeline import default_schema

    optional = {s.name for s in default_schema() if s.must_exist and s.optional_path}
    assert optional == {
        "tissue_raw_tiff_path",
        "measurement_3d_vessel_mask_path",
        "large_venule_mask_path",
        "small_venule_mask_path",
        "ilastik_venule_classifier_path",
        "ilastik_small_venule_classifier_path",
    }


def test_named_settings_can_be_skipped(tmp_path):
    report = check_settings(
        _schema(),
        {"use_ilastik": False, "input_path": tmp_path / "absent.tif"},
        skip=["input_path"],
    )
    assert report.ok


# --- the .zip convention ---------------------------------------------------


def test_a_zipped_file_satisfies_the_check(tmp_path):
    """Loaders accept a zipped input, so a missing path is not missing."""
    zipped = tmp_path / "mask.tif.zip"
    zipped.write_bytes(b"x")
    exists, detail = resolve_existing_path(tmp_path / "mask.tif")
    assert exists
    assert "zipped" in detail


def test_a_missing_path_reports_everything_it_tried(tmp_path):
    exists, detail = resolve_existing_path(tmp_path / "mask.tif")
    assert not exists
    assert "mask.tif" in detail and "mask.tif.zip" in detail


# --- the declaration itself ------------------------------------------------


def test_must_exist_is_rejected_on_a_setting_that_is_not_a_path():
    with pytest.raises(ConfigError, match="only a path can be checked"):
        Setting("count", "int", 1, "How many", "S", must_exist=True)


def test_must_exist_is_part_of_the_gui_description():
    described = _schema().describe()
    by_name = {
        setting["name"]: setting
        for section in described["sections"]
        for setting in section["settings"]
    }
    assert by_name["input_path"]["must_exist"] is True
    assert by_name["output_dir"]["must_exist"] is False


def test_a_negated_prerequisite_names_a_real_setting():
    with pytest.raises(ConfigError, match="requires 'missing'"):
        Schema([Setting("a", "path", None, "Only", "S", requires=("!missing",))])


def test_an_unset_path_needed_outside_one_mode_says_which_mode(tmp_path):
    schema = Schema([
        Setting("mode", "choice", "file", "Where masks come from", "S", choices=("file", "none")),
        Setting(
            "mask_path", "path", None, "Read this mask", "S",
            requires=("mode!=none",), must_exist=True,
        ),
    ])
    report = check_settings(schema, {"mode": "file", "mask_path": None})
    assert any("'mode' is not 'none'" in message for message in report.errors)


def test_a_must_exist_path_behind_a_path_gate_is_demanded_only_while_the_gate_is_set(tmp_path):
    """The reason names the gate as set, not on."""
    from haemolynx.parsers.checks import check_settings
    from haemolynx.parsers.schema import Schema, Setting

    schema = Schema([
        Setting("raw", "path", None, "The raw image", "S"),
        Setting("classifier", "path", None, "What segments it", "S",
                requires=("raw",), must_exist=True),
    ])

    assert check_settings(schema, {"raw": None, "classifier": None}).ok

    report = check_settings(schema, {"raw": str(tmp_path / "raw.tif"), "classifier": None})
    assert len(report.errors) == 1
    assert "required because 'raw' is set" in report.errors[0]
