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
