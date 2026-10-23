"""The vessel label each ilastik project is read at: its five settings, the
preflight check that reads each project's labels, and the stages handing the
label to the segmentation (`io.ilastik`, whose own tests are in test_io.py).

ilastik's Simple Segmentation numbers the labels 1 to N, and which is the
vessel cannot be read from the numbers: a two-label output used to be read
with its minority as vessel, inside out wherever vessels are the majority.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.io.vessel_masks import (
    VESSEL_MASK_SETTINGS,
    load_and_validate_vessel_masks,
    vessel_mask_arguments,
)
from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.checks import (
    check_ilastik_vessel_labels,
    ilastik_classifiers_in_effect,
    preflight,
)
from haemolynx.pipeline.stages import segment

h5py = pytest.importorskip("h5py")

SCHEMA = default_schema()

LABEL_SETTINGS = (
    "ilastik_vessel_label",
    "ilastik_arteriole_vessel_label",
    "ilastik_venule_vessel_label",
    "ilastik_small_arteriole_vessel_label",
    "ilastik_small_venule_vessel_label",
)


def _project(path: Path, names=("Vessel", "Background")) -> Path:
    with h5py.File(path, "w") as project:
        project.create_dataset(
            "PixelClassification/LabelNames",
            data=[name.encode() for name in names],
            dtype=h5py.string_dtype(),
        )
    return path


def _settings(tmp_path: Path, **overrides) -> dict:
    values = {"vtk_output_prefix": tmp_path / "run", **overrides}
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _main_ilastik(tmp_path: Path, **overrides) -> dict:
    raw = tmp_path / "raw.tif"
    tifffile.imwrite(raw, np.zeros((3, 4, 4), dtype=np.uint8))
    return _settings(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=raw,
        ilastik_classifier_path=_project(tmp_path / "main.ilp"),
        ilastik_output_dir=tmp_path / "segmentations",
        **overrides,
    )


def _large_ilastik(tmp_path: Path, *, venule: bool, **overrides) -> dict:
    values = dict(
        automated_vessel_assignment=True,
        use_large_vessel_masks=True,
        use_ilastik_large_vessel_segmentation=True,
        ilastik_arteriole_classifier_path=_project(tmp_path / "art.ilp", ("Artery", "Rest")),
    )
    if venule:
        values["ilastik_venule_classifier_path"] = _project(tmp_path / "ven.ilp", ("Rest", "Vein"))
    values.update(overrides)
    return _settings(tmp_path, **values)


# --- the settings ---------------------------------------------------------------


@pytest.mark.parametrize("name", LABEL_SETTINGS)
def test_each_label_setting_has_no_default_and_sits_under_its_classifier(name):
    setting = SCHEMA[name]
    assert setting.kind == "str"
    assert setting.default is None
    assert setting.placeholder


def test_the_venule_labels_are_read_only_with_a_venule_classifier():
    assert "ilastik_venule_classifier_path" in SCHEMA["ilastik_venule_vessel_label"].requires
    assert (
        "ilastik_small_venule_classifier_path"
        in SCHEMA["ilastik_small_venule_vessel_label"].requires
    )


def test_each_mask_role_maps_its_own_label_settings():
    for role, small in (("large", ""), ("small", "small_")):
        names = VESSEL_MASK_SETTINGS[role]
        assert names["ilastik_arteriole_vessel_label"] == f"ilastik_{small}arteriole_vessel_label"
        assert names["ilastik_venule_vessel_label"] == f"ilastik_{small}venule_vessel_label"
    arguments = vessel_mask_arguments(
        {"ilastik_small_arteriole_vessel_label": "A", "ilastik_small_venule_vessel_label": "2"},
        "small",
    )
    assert arguments["ilastik_arteriole_vessel_label"] == "A"
    assert arguments["ilastik_venule_vessel_label"] == "2"


# --- preflight ----------------------------------------------------------------


def test_preflight_refuses_a_main_run_with_no_vessel_label_and_lists_the_labels(tmp_path):
    report = check_ilastik_vessel_labels(_main_ilastik(tmp_path))

    assert not report.ok
    (error,) = report.errors
    assert "ilastik_vessel_label is not set" in error
    assert "1 ('Vessel'), 2 ('Background')" in error


def test_preflight_refuses_a_label_the_project_does_not_have(tmp_path):
    report = check_ilastik_vessel_labels(_main_ilastik(tmp_path, ilastik_vessel_label="Lumen"))

    assert not report.ok
    assert "ilastik_vessel_label='Lumen' is not a label in main.ilp" in report.errors[0]


@pytest.mark.parametrize("label", ["Vessel", "vessel", "1"])
def test_preflight_passes_a_label_by_name_or_number(tmp_path, label):
    report = check_ilastik_vessel_labels(_main_ilastik(tmp_path, ilastik_vessel_label=label))

    assert report.ok
    assert ("ilastik_vessel_label", "label 1 ('Vessel') of main.ilp; its labels are 1 ('Vessel'), 2 ('Background')") in report.passed


def test_the_whole_preflight_runs_the_label_check(tmp_path):
    report = preflight(_main_ilastik(tmp_path), SCHEMA)

    assert any("ilastik_vessel_label is not set" in error for error in report.errors)


def test_a_project_with_no_label_names_takes_a_number_with_a_warning(tmp_path):
    settings = _main_ilastik(tmp_path, ilastik_vessel_label="1")
    Path(settings["ilastik_classifier_path"]).write_bytes(b"not an hdf5 project")

    report = check_ilastik_vessel_labels(settings)
    assert report.ok
    assert any("no label names could be read" in warning for warning in report.warnings)

    settings["ilastik_vessel_label"] = "Vessel"
    assert not check_ilastik_vessel_labels(settings).ok


def test_no_label_is_asked_for_while_ilastik_segments_nothing(tmp_path):
    assert ilastik_classifiers_in_effect(_settings(tmp_path)) == []
    # ilastik on with no raw image: the segmented image is read instead.
    settings = _main_ilastik(tmp_path)
    settings["raw_image_path"] = None
    assert ilastik_classifiers_in_effect(settings) == []
    # Mask ilastik on under an unticked mask switch runs nothing.
    settings = _large_ilastik(tmp_path, venule=True, use_large_vessel_masks=False)
    assert ilastik_classifiers_in_effect(settings) == []
    assert check_ilastik_vessel_labels(settings).ok


def test_each_mask_classifier_needs_its_own_label(tmp_path):
    settings = _large_ilastik(
        tmp_path, venule=True, ilastik_arteriole_vessel_label="Artery"
    )
    assert ilastik_classifiers_in_effect(settings) == [
        ("ilastik_arteriole_classifier_path", "ilastik_arteriole_vessel_label"),
        ("ilastik_venule_classifier_path", "ilastik_venule_vessel_label"),
    ]

    report = check_ilastik_vessel_labels(settings)
    (error,) = report.errors
    assert "ilastik_venule_vessel_label is not set" in error and "2 ('Vein')" in error

    settings["ilastik_venule_vessel_label"] = "Vein"
    assert check_ilastik_vessel_labels(settings).ok


def test_a_global_mask_classifier_needs_only_its_own_label(tmp_path):
    settings = _large_ilastik(tmp_path, venule=False, ilastik_arteriole_vessel_label="Artery")

    assert ilastik_classifiers_in_effect(settings) == [
        ("ilastik_arteriole_classifier_path", "ilastik_arteriole_vessel_label")
    ]
    assert check_ilastik_vessel_labels(settings).ok


# --- the stages hand the label on -----------------------------------------------


def test_segment_hands_the_vessel_label_to_ilastik(tmp_path, monkeypatch):
    from haemolynx import io

    produced = tmp_path / "segmentations" / "raw_segmented.tif"
    produced.parent.mkdir()
    tifffile.imwrite(produced, np.ones((3, 4, 4), dtype=np.uint8))
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return produced

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", fake_run)
    segment(_main_ilastik(tmp_path, ilastik_vessel_label="Vessel"))

    assert seen["vessel_label"] == "Vessel"
    assert seen["label_setting"] == "ilastik_vessel_label"


def test_the_mask_loader_hands_each_classifier_its_own_label(tmp_path, monkeypatch):
    from haemolynx.io import vessel_masks as module

    shape = (3, 4, 5)
    seen = []

    def fake_run(**kwargs):
        seen.append((Path(kwargs["classifier_path"]).name, kwargs["vessel_label"], kwargs["label_setting"]))
        volume = np.zeros(shape, dtype=np.uint8)
        volume[1, 1, 1] = 1
        tifffile.imwrite(kwargs["output_path"], volume)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(module, "run_ilastik_headless_segmentation", fake_run)
    (tmp_path / "raw.tif").write_bytes(b"x")
    (tmp_path / "seg").mkdir()
    settings = _large_ilastik(
        tmp_path,
        venule=True,
        ilastik_arteriole_vessel_label="Artery",
        ilastik_venule_vessel_label="Vein",
        ilastik_unsegmented_arteriole_image_path=tmp_path / "raw.tif",
        ilastik_unsegmented_venule_image_path=tmp_path / "raw.tif",
        ilastik_output_dir=tmp_path / "seg",
    )
    arguments = vessel_mask_arguments(settings, "large")
    arguments.update(arteriole_mask_path=None, venule_mask_path=None)

    load_and_validate_vessel_masks(
        **arguments,
        image_shape=shape,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert seen == [
        ("art.ilp", "Artery", "ilastik_arteriole_vessel_label"),
        ("ven.ilp", "Vein", "ilastik_venule_vessel_label"),
    ]
