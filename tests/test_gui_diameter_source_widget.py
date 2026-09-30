"""The Diameters tab, built for real: one Diameter measurement choice where
the FWHM and endothelial checkboxes were, and no rows for FWHM's internal
label values.

Visibility is read with ``isVisibleTo(page)`` -- "would this show if the page
did" -- so a test need not put the tab on screen first.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    FORCED_HIDDEN_SETTINGS,
    forced_hidden_value,
    settings_widget,
)
from haemolynx.gui.diameter_source import (  # noqa: E402
    BOTH_ON_NOTE,
    DIAMETER_SOURCE_SETTINGS,
    ENDOTHELIAL,
    FWHM,
    NO_MEASUREMENT,
)
from haemolynx.pipeline import default_schema  # noqa: E402

pytestmark = pytest.mark.gui

SCHEMA = default_schema()


@pytest.fixture
def panel(make_napari_viewer):
    return settings_widget(napari_viewer=make_napari_viewer())


def _process():
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def _diameters_page(panel):
    tabs = panel._haemolynx_tabs
    index = next(i for i in range(tabs.count()) if "Diameters" in tabs.tabText(i))
    return tabs.widget(index).widget()


def _shown(panel, widget) -> bool:
    return widget.native.isVisibleTo(_diameters_page(panel))


def test_the_tab_shows_one_choice_where_the_two_checkboxes_were(panel):
    rows = panel._haemolynx_rows()
    choice = panel._haemolynx_diameter_source
    diameters = list(panel._haemolynx_diameters_settings)

    assert _shown(panel, choice)
    assert choice.label == "Diameter measurement"
    for name in DIAMETER_SOURCE_SETTINGS:
        assert rows[name] not in diameters, name
        assert not _shown(panel, rows[name]), name
    # Where the FWHM checkbox was: after the diameter table, before FWHM's rows.
    assert diameters.index(rows["default_diameter"]) < diameters.index(choice)
    assert diameters.index(choice) < diameters.index(rows["fwhm_raw_tiff_path"])
    assert choice.value == NO_MEASUREMENT


def test_choosing_a_source_writes_the_two_settings_and_shows_its_rows(panel):
    rows = panel._haemolynx_rows()
    choice = panel._haemolynx_diameter_source

    choice.value = FWHM
    _process()
    values = panel._haemolynx_values()
    assert (values["use_fwhm_edge_diameters"], values["use_endothelial_diameters"]) == (True, False)
    assert _shown(panel, rows["fwhm_raw_tiff_path"])
    assert not _shown(panel, rows["endothelial_image_path"])

    choice.value = ENDOTHELIAL
    _process()
    values = panel._haemolynx_values()
    assert (values["use_fwhm_edge_diameters"], values["use_endothelial_diameters"]) == (False, True)
    assert not _shown(panel, rows["fwhm_raw_tiff_path"])
    assert _shown(panel, rows["endothelial_image_path"])

    choice.value = NO_MEASUREMENT
    _process()
    values = panel._haemolynx_values()
    assert (values["use_fwhm_edge_diameters"], values["use_endothelial_diameters"]) == (False, False)
    assert not _shown(panel, rows["fwhm_raw_tiff_path"])
    assert not _shown(panel, rows["endothelial_image_path"])


def test_setting_the_two_settings_moves_the_choice(panel):
    rows = panel._haemolynx_rows()
    choice = panel._haemolynx_diameter_source
    rows["use_endothelial_diameters"].value = True
    _process()
    assert choice.value == ENDOTHELIAL
    rows["use_endothelial_diameters"].value = False
    _process()
    assert choice.value == NO_MEASUREMENT


def test_the_choice_hides_while_haemodynamics_is_off(panel):
    rows = panel._haemolynx_rows()
    rows["run_haemodynamics"].value = False
    _process()
    assert not _shown(panel, panel._haemolynx_diameter_source)
    rows["run_haemodynamics"].value = True
    _process()
    assert _shown(panel, panel._haemolynx_diameter_source)


def test_a_config_with_both_on_keeps_fwhm_and_says_so(panel, tmp_path):
    config = tmp_path / "both.yaml"
    config.write_text(
        "use_fwhm_edge_diameters: true\nuse_endothelial_diameters: true\n", encoding="utf-8"
    )
    panel._haemolynx_load_config(config)
    _process()
    values = panel._haemolynx_values()
    assert values["use_fwhm_edge_diameters"] is True
    assert values["use_endothelial_diameters"] is False
    assert panel._haemolynx_diameter_source.value == FWHM
    assert panel._haemolynx_report() == BOTH_ON_NOTE


def test_a_config_switching_source_is_not_mistaken_for_both_on(panel, tmp_path):
    """A load sets one row at a time, passing through both on on the way."""
    panel._haemolynx_diameter_source.value = ENDOTHELIAL
    _process()
    config = tmp_path / "fwhm.yaml"
    config.write_text(
        "use_fwhm_edge_diameters: true\nuse_endothelial_diameters: false\n", encoding="utf-8"
    )
    panel._haemolynx_load_config(config)
    _process()
    values = panel._haemolynx_values()
    assert (values["use_fwhm_edge_diameters"], values["use_endothelial_diameters"]) == (True, False)
    assert panel._haemolynx_diameter_source.value == FWHM
    assert panel._haemolynx_report() != BOTH_ON_NOTE


def test_a_saved_config_holds_the_two_settings_the_choice_wrote(panel, tmp_path):
    panel._haemolynx_diameter_source.value = ENDOTHELIAL
    _process()
    config = tmp_path / "saved.yaml"
    assert panel._haemolynx_save_config(config)
    text = config.read_text(encoding="utf-8")
    assert "use_endothelial_diameters: true" in text
    assert "use_fwhm_edge_diameters: false" in text


@pytest.mark.parametrize("name", ["fwhm_background_label", "fwhm_junction_label"])
def test_fwhms_internal_label_values_have_no_row_and_stay_at_their_defaults(panel, tmp_path, name):
    rows = panel._haemolynx_rows()
    assert FORCED_HIDDEN_SETTINGS[name] == SCHEMA[name].default
    panel._haemolynx_diameter_source.value = FWHM
    panel._haemolynx_expand_advanced(True)
    _process()
    assert rows[name] not in list(panel._haemolynx_diameters_settings)
    assert not _shown(panel, rows[name])
    assert all(name not in button.names for button in panel._haemolynx_advanced.values())

    # A config cannot move one out of sight.
    config = tmp_path / "labels.yaml"
    config.write_text(f"{name}: 7\n", encoding="utf-8")
    panel._haemolynx_load_config(config)
    _process()
    values = panel._haemolynx_values()
    assert values[name] == forced_hidden_value(name, values) == SCHEMA[name].default
