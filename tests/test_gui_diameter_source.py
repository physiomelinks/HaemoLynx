"""The Diameters tab's one "Diameter measurement" choice, over two settings.

Pure: :mod:`haemolynx.gui.diameter_source` says what each choice means in the
two settings a config file holds, and what two settings amount to as a
choice. The panel's own wiring is tested in ``test_gui_diameter_source_widget``.
"""
from __future__ import annotations

import pytest

from haemolynx.gui.diameter_source import (
    DIAMETER_SOURCES,
    DIAMETER_SOURCE_SETTINGS,
    ENDOTHELIAL,
    FWHM,
    NO_MEASUREMENT,
    both_on,
    settings_for,
    source_from,
)
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.checks import check_one_primary_diameter_measurement

SCHEMA = default_schema()


def test_the_choice_stands_for_two_real_bool_settings():
    for name in DIAMETER_SOURCE_SETTINGS:
        assert SCHEMA[name].kind == "bool"
        assert SCHEMA[name].default is False
    assert source_from(SCHEMA.defaults()) == NO_MEASUREMENT


@pytest.mark.parametrize(
    "source, fwhm, endothelial",
    [(NO_MEASUREMENT, False, False), (FWHM, True, False), (ENDOTHELIAL, False, True)],
)
def test_each_choice_is_one_pair_of_settings_and_reads_back_as_itself(source, fwhm, endothelial):
    settings = settings_for(source)
    assert settings == {"use_fwhm_edge_diameters": fwhm, "use_endothelial_diameters": endothelial}
    assert source_from(settings) == source


@pytest.mark.parametrize("source", [value for _label, value in DIAMETER_SOURCES])
def test_no_choice_is_one_a_run_refuses(source):
    """Two checkboxes could both be ticked; no single choice turns both on."""
    settings = settings_for(source)
    assert not both_on(settings)
    assert check_one_primary_diameter_measurement(settings).ok


def test_both_on_is_the_combination_preflight_refuses():
    both = {"use_fwhm_edge_diameters": True, "use_endothelial_diameters": True}
    assert both_on(both)
    assert not check_one_primary_diameter_measurement(both).ok
    # Shown as FWHM until the panel settles it.
    assert source_from(both) == FWHM


def test_an_unknown_choice_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown diameter source 'mask'"):
        settings_for("mask")


def test_every_choice_has_its_own_display_text():
    labels = [label for label, _value in DIAMETER_SOURCES]
    values = [value for _label, value in DIAMETER_SOURCES]
    assert len(set(labels)) == len(labels) == 3
    assert values == [NO_MEASUREMENT, FWHM, ENDOTHELIAL]
