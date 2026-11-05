"""The classifier training window's pure parts: the job menu's names, the
labels' colours, strokes carried from one job to another, and the Boundaries
tab's vessel masks a classifier in the main image's row makes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from haemolynx.classifier.model import FILE_KIND, PURPOSES
from haemolynx.gui.classifier_training import (
    LABEL_COLOURS,
    PURPOSE_TITLES,
    classifier_filled_rows,
    classifier_vessel_mask_roles,
    remap_strokes,
    result_colours,
    stroke_colours,
    vessel_mask_note,
    vessel_mask_settings_for,
)


def _record_only(path, purpose):
    """A classifier file holding only its record: all the roles are read from."""
    record = {"kind": FILE_KIND, "version": 1, "purpose": purpose, "labels": list(PURPOSES[purpose])}
    with open(path, "wb") as handle:
        np.savez(handle, record=np.array(json.dumps(record)))
    return path


def test_the_job_menu_names_the_vessel_jobs_by_what_they_find():
    assert PURPOSE_TITLES["vessels"] == "All vessels"
    assert PURPOSE_TITLES["vessels_and_classes"] == "Capillaries, arterioles and venules"
    assert PURPOSE_TITLES["vessels_and_sized_classes"] == "Capillaries, large and small arterioles and venules"
    assert set(PURPOSE_TITLES) == set(PURPOSES)


def _rgb(name):
    return np.array(LABEL_COLOURS[name][:3])


def test_large_vessels_are_dark_and_small_ones_light_arterioles_red_venules_blue():
    for size in ("Large", "Small"):
        r, g, b = _rgb(f"{size} arteriole")
        assert r > g and r > b
        r, g, b = _rgb(f"{size} venule")
        assert b > r and b > g
    assert _rgb("Large arteriole").sum() < _rgb("Small arteriole").sum()
    assert _rgb("Large venule").sum() < _rgb("Small venule").sum()
    r, g, b = _rgb("Capillary")
    assert g > r and g > b
    # Every label of the job its own colour.
    colours = [LABEL_COLOURS[name] for name in PURPOSES["vessels_and_sized_classes"]]
    assert len(set(colours)) == len(colours)


def _strongest(colour):
    return "rgb"[max(range(3), key=lambda i: colour[i])]


@pytest.mark.parametrize("colours", [stroke_colours, result_colours])
def test_capillaries_are_green_arterioles_red_and_venules_blue(colours):
    shift = 1 if colours is stroke_colours else 0  # a stroke is its label's index + 1
    labels = PURPOSES["vessels_and_classes"]
    drawn = colours("vessels_and_classes")
    assert _strongest(drawn[labels.index("Capillary") + shift]) == "g"
    assert _strongest(drawn[labels.index("Arteriole") + shift]) == "r"
    assert _strongest(drawn[labels.index("Venule") + shift]) == "b"


def test_every_label_has_a_colour():
    assert {name for labels in PURPOSES.values() for name in labels} <= set(LABEL_COLOURS)


def test_a_vessel_stroke_is_a_capillary_stroke_in_the_combined_job_and_back():
    # Background 1, Vessel 2  <->  Background 1, Capillary 2, Arteriole 3, Venule 4
    assert remap_strokes("vessels", "vessels_and_classes") == {1: 1, 2: 2}
    assert remap_strokes("vessels_and_classes", "vessels") == {1: 1, 2: 2, 3: 0, 4: 0}
    assert remap_strokes("vessel_classes", "vessels_and_classes") == {1: 1, 2: 3, 3: 4}
    assert remap_strokes("vessels_and_classes", "vessel_classes") == {1: 1, 2: 0, 3: 2, 4: 3}
    # Capillaries carry over between the combined jobs; arterioles and venules
    # do not, as a large or a small one is not known.
    assert remap_strokes("vessels_and_classes", "vessels_and_sized_classes") == {1: 1, 2: 2, 3: 0, 4: 0}
    assert remap_strokes("vessels_and_sized_classes", "vessels") == {1: 1, 2: 2, 3: 0, 4: 0, 5: 0, 6: 0}


@pytest.mark.parametrize("purpose, roles", [
    ("vessels", ()),
    ("vessel_classes", ()),
    ("vessels_and_classes", ("large",)),
    ("vessels_and_sized_classes", ("large", "small")),
    ("pericytes", ()),
])
def test_the_roles_a_classifier_also_makes_masks_for(tmp_path, purpose, roles):
    assert classifier_vessel_mask_roles(_record_only(tmp_path / "c.haemoclf", purpose)) == roles


def test_no_file_or_not_a_classifier_makes_no_masks(tmp_path):
    assert classifier_vessel_mask_roles(None) == ()
    assert classifier_vessel_mask_roles(tmp_path / "missing.haemoclf") == ()
    (tmp_path / "not.haemoclf").write_bytes(b"not a classifier")
    assert classifier_vessel_mask_roles(tmp_path / "not.haemoclf") == ()


def test_each_role_gets_the_file_its_switch_and_its_masks_ilastik_off():
    settings = vessel_mask_settings_for("c.haemoclf", ("large", "small"))
    assert settings == {
        "pixel_classifier_large_vessel_path": "c.haemoclf",
        "use_ilastik_large_vessel_segmentation": False,
        "use_pixel_classifier_large_vessel_segmentation": True,
        "pixel_classifier_small_vessel_path": "c.haemoclf",
        "use_ilastik_small_vessel_segmentation": False,
        "use_pixel_classifier_small_vessel_segmentation": True,
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_small_vessel_masks_for_boundary_assignment": True,
    }
    # Each role's classifier switch before its mask switch: the panel's search
    # for masks beside the image then finds the role made from the raw image.
    order = list(settings)
    for role, masks in (("large", "use_large_vessel_masks"),
                        ("small", "use_small_vessel_masks_for_boundary_assignment")):
        assert order.index(f"use_pixel_classifier_{role}_vessel_segmentation") < order.index(masks)
    assert list(vessel_mask_settings_for("c.haemoclf", ("large",))) == [
        "pixel_classifier_large_vessel_path", "use_ilastik_large_vessel_segmentation",
        "use_pixel_classifier_large_vessel_segmentation", "automated_vessel_assignment",
        "use_large_vessel_masks",
    ]
    assert vessel_mask_settings_for("c.haemoclf", ()) == {}


def test_the_note_names_the_sizes_filled():
    assert "large and small vessel masks" in vessel_mask_note(("large", "small"))
    assert "large vessel masks" in vessel_mask_note(("large",))


def _record_with_counts(path, purpose, counts):
    """A classifier file holding only its record, shown *counts* voxels of each label."""
    record = {
        "kind": FILE_KIND, "version": 1, "purpose": purpose, "labels": list(PURPOSES[purpose]),
        "voxels_per_label": counts,
    }
    with open(path, "wb") as handle:
        np.savez(handle, record=np.array(json.dumps(record)))
    return path


def _run_settings(tmp_path, **overrides):
    from haemolynx.pipeline import default_schema

    return {**default_schema().defaults(), "output_dir": tmp_path / "out",
            "raw_image_path": tmp_path / "raw.tif", **overrides}


def _made_by_the_run(settings):
    """Every mask path the run's classifiers write (``stages.classifier_requests``)."""
    from haemolynx.pipeline.stages import classifier_requests

    return {path for _classifier, outputs in classifier_requests(settings) for path in outputs.values()}


def test_the_segmented_image_row_names_the_mask_the_main_classifier_writes(tmp_path):
    settings = _run_settings(tmp_path, use_pixel_classifier_segmentation=True,
                             pixel_classifier_path=_record_only(tmp_path / "v.haemoclf", "vessels"))
    filled = classifier_filled_rows(settings)
    assert filled == {"input_path": tmp_path / "out" / "raw_segmented.tif"}
    assert set(filled.values()) == _made_by_the_run(settings)
    # Named by the raw image alone: before a classifier is chosen too.
    assert classifier_filled_rows({**settings, "pixel_classifier_path": None}) == filled


@pytest.mark.parametrize("change", [
    {"use_pixel_classifier_segmentation": False},
    {"raw_image_path": None},  # the classifier has nothing to segment
    {"use_ilastik_segmentation": True},  # ilastik segments it instead
    {"network_source": "synthetic"},
])
def test_no_classifier_making_the_mask_fills_nothing(tmp_path, change):
    settings = _run_settings(tmp_path, use_pixel_classifier_segmentation=True)
    assert classifier_filled_rows({**settings, **change}) == {}


def test_a_combined_classifier_fills_the_large_arteriole_and_venule_rows(tmp_path):
    path = _record_with_counts(tmp_path / "c.haemoclf", "vessels_and_classes",
                               {"Background": 9, "Capillary": 9, "Arteriole": 9, "Venule": 9})
    settings = _run_settings(tmp_path, use_pixel_classifier_segmentation=True, pixel_classifier_path=path,
                             **vessel_mask_settings_for(path, classifier_vessel_mask_roles(path)))
    filled = classifier_filled_rows(settings)
    out = tmp_path / "out"
    # Every vessel to the segmented image, the arterioles and venules to the large masks.
    assert filled == {
        "input_path": out / "raw_segmented.tif",
        "large_arteriole_mask_path": out / "raw_large_arteriole_segmented.tif",
        "large_venule_mask_path": out / "raw_large_venule_segmented.tif",
    }
    assert set(filled.values()) == _made_by_the_run(settings)


def test_a_classifier_telling_large_from_small_fills_each_roles_rows(tmp_path):
    path = _record_with_counts(tmp_path / "s.haemoclf", "vessels_and_sized_classes", {
        "Background": 9, "Capillary": 9, "Large arteriole": 9, "Small arteriole": 9,
        "Large venule": 9, "Small venule": 9,
    })
    settings = _run_settings(tmp_path, use_pixel_classifier_segmentation=True, pixel_classifier_path=path,
                             **vessel_mask_settings_for(path, ("large", "small")))
    filled = classifier_filled_rows(settings)
    out = tmp_path / "out"
    assert filled == {
        "input_path": out / "raw_segmented.tif",
        "large_arteriole_mask_path": out / "raw_large_arteriole_segmented.tif",
        "large_venule_mask_path": out / "raw_large_venule_segmented.tif",
        "small_arteriole_mask_path": out / "raw_small_arteriole_segmented.tif",
        "small_venule_mask_path": out / "raw_small_venule_segmented.tif",
    }
    assert set(filled.values()) == _made_by_the_run(settings)


def test_a_classifier_never_shown_a_venule_fills_the_global_mask_and_empties_the_venule_row(tmp_path):
    from haemolynx.io.vessel_masks import vessel_masks_are_global

    path = _record_with_counts(tmp_path / "s.haemoclf", "vessels_and_sized_classes", {
        "Background": 9, "Capillary": 9, "Large arteriole": 9, "Small venule": 9,
    })
    settings = _run_settings(tmp_path, **vessel_mask_settings_for(path, ("large", "small")))
    filled = classifier_filled_rows(settings)
    out = tmp_path / "out"
    assert filled == {
        "large_arteriole_mask_path": out / "raw_large_global_segmented.tif",
        "large_venule_mask_path": None,
        "small_arteriole_mask_path": out / "raw_small_arteriole_segmented.tif",
        "small_venule_mask_path": out / "raw_small_venule_segmented.tif",
    }
    assert {p for p in filled.values() if p is not None} == _made_by_the_run(settings)
    # The rows read as the run reads the classifier: the large role global either way.
    off = {**settings, **filled, "use_pixel_classifier_large_vessel_segmentation": False,
           "use_pixel_classifier_small_vessel_segmentation": False}
    assert vessel_masks_are_global(settings) and vessel_masks_are_global(off)


@pytest.mark.parametrize("change", [
    {"pixel_classifier_large_vessel_path": None},
    {"pixel_classifier_large_vessel_path": "missing.haemoclf"},  # how it names its masks is unknown
    {"use_large_vessel_masks": False},
    {"use_ilastik_large_vessel_segmentation": True},
    {"raw_image_path": None},
])
def test_a_vessel_mask_role_its_classifier_does_not_make_fills_nothing(tmp_path, change):
    path = _record_with_counts(tmp_path / "c.haemoclf", "vessel_classes", {"Arteriole": 9, "Venule": 9})
    settings = _run_settings(tmp_path, **vessel_mask_settings_for(path, ("large",)))
    assert set(classifier_filled_rows(settings)) == {"large_arteriole_mask_path", "large_venule_mask_path"}
    assert classifier_filled_rows({**settings, **change}) == {}


def test_a_filled_row_that_does_not_exist_yet_is_not_asked_for_before_the_run(tmp_path):
    from haemolynx.parsers.checks import check_settings
    from haemolynx.pipeline import default_schema

    path = _record_with_counts(tmp_path / "c.haemoclf", "vessels_and_classes",
                               {"Capillary": 9, "Arteriole": 9, "Venule": 9})
    (tmp_path / "raw.tif").write_bytes(b"")
    settings = _run_settings(tmp_path, use_pixel_classifier_segmentation=True, pixel_classifier_path=path,
                             **vessel_mask_settings_for(path, ("large",)))
    settings.update(classifier_filled_rows(settings))
    errors = " ".join(check_settings(default_schema(), settings).errors)
    for name in ("input_path", "large_arteriole_mask_path", "large_venule_mask_path"):
        assert name not in errors, name
