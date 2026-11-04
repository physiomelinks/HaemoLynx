"""The classifier training window's pure parts: the job menu's names, the
labels' colours, and strokes carried from one job to another."""
from __future__ import annotations

import pytest

from haemolynx.classifier.model import PURPOSES
from haemolynx.gui.classifier_training import (
    LABEL_COLOURS,
    PURPOSE_TITLES,
    remap_strokes,
    result_colours,
    stroke_colours,
)


def test_the_job_menu_names_the_two_vessel_jobs_by_what_they_find():
    assert PURPOSE_TITLES["vessels"] == "All vessels"
    assert PURPOSE_TITLES["vessels_and_classes"] == "Capillaries, arterioles and venules"
    assert set(PURPOSE_TITLES) == set(PURPOSES)


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
