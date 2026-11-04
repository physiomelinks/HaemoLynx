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


def test_large_vessels_are_dark_and_small_ones_light_arterioles_red_venules_green():
    for size in ("Large", "Small"):
        r, g, b = _rgb(f"{size} arteriole")
        assert r > g and r > b
        r, g, b = _rgb(f"{size} venule")
        assert g > r and g > b
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
