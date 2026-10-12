"""The panel with one raw image, built for real: its path, channel and stain on
the Input tab, and one "Measure diameters from the raw image" checkbox on the
Diameters tab whose method -- FWHM or the endothelial wall fit -- follows the
stain.

Visibility is read with ``isVisibleTo(page)`` -- "would this show if the page
did" -- so a test need not put the tab on screen first.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    FORCED_HIDDEN_SETTINGS,
    settings_widget,
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


def _page(panel, word):
    tabs = panel._haemolynx_tabs
    index = next(i for i in range(tabs.count()) if word in tabs.tabText(i))
    return tabs.widget(index).widget()


def _shown(panel, widget, word="Diameters") -> bool:
    return widget.native.isVisibleTo(_page(panel, word))


def test_the_input_tab_holds_the_segmented_and_the_raw_image(panel):
    rows = panel._haemolynx_rows()
    for name in ("input_path", "raw_image_path", "raw_image_channel", "raw_image_stain"):
        assert _shown(panel, rows[name], "Input"), name
    assert rows["input_path"].label == "Segmented image path"
    assert rows["raw_image_path"].label == "Raw image path"
    assert rows["raw_image_stain"].value == "auto"
    # The check and the optimiser read the schema's own rows: no mirrors.
    assert panel._haemolynx_raw_data_row is rows["raw_image_path"]
    assert panel._haemolynx_raw_channel_row is rows["raw_image_channel"]


def test_the_diameters_tab_has_one_switch_and_no_image_of_its_own(panel):
    rows = panel._haemolynx_rows()
    diameters = list(panel._haemolynx_diameters_settings)
    switch = rows["measure_diameters_from_raw_image"]
    assert switch in diameters and _shown(panel, switch)
    assert diameters.index(rows["default_diameter"]) < diameters.index(switch)
    for retired in ("fwhm_raw_tiff_path", "endothelial_image_path", "use_fwhm_edge_diameters",
                    "use_endothelial_diameters", "diameter_basis"):
        assert retired not in rows, retired


@pytest.mark.parametrize(
    "stain, fwhm_shown, endothelial_shown",
    [("auto", True, True), ("plasma_column", True, False), ("endothelial", False, True)],
)
def test_the_stain_decides_which_methods_rows_show(panel, stain, fwhm_shown, endothelial_shown):
    rows = panel._haemolynx_rows()
    rows["measure_diameters_from_raw_image"].value = True
    rows["raw_image_stain"].value = stain
    panel._haemolynx_expand_advanced(True)
    _process()
    assert _shown(panel, rows["fwhm_sample_spacing_along_edge_um"]) is fwhm_shown
    assert _shown(panel, rows["endothelial_min_ring_contrast"]) is endothelial_shown
    assert _shown(panel, panel._haemolynx_optimise_fwhm_box) is fwhm_shown


def test_measuring_nothing_shows_neither_method(panel):
    rows = panel._haemolynx_rows()
    rows["measure_diameters_from_raw_image"].value = False
    panel._haemolynx_expand_advanced(True)
    _process()
    assert not _shown(panel, rows["fwhm_sample_spacing_along_edge_um"])
    assert not _shown(panel, rows["endothelial_min_ring_contrast"])


def test_an_old_config_loads_into_the_raw_image_rows(panel, tmp_path):
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"")
    config = tmp_path / "old.yaml"
    config.write_text(
        f"use_fwhm_edge_diameters: true\nfwhm_raw_tiff_path: {raw.as_posix()}\ndiameter_basis: plasma_column\n",
        encoding="utf-8",
    )
    panel._haemolynx_load_config(config)
    _process()
    values = panel._haemolynx_values()
    assert values["measure_diameters_from_raw_image"] is True
    assert values["raw_image_stain"] == "plasma_column"
    assert str(values["raw_image_path"]).endswith("raw.tif")


def test_a_saved_config_holds_the_one_switch_and_the_stain(panel, tmp_path):
    rows = panel._haemolynx_rows()
    rows["measure_diameters_from_raw_image"].value = True
    rows["raw_image_stain"].value = "endothelial"
    _process()
    config = tmp_path / "saved.yaml"
    assert panel._haemolynx_save_config(config)
    text = config.read_text(encoding="utf-8")
    assert "measure_diameters_from_raw_image: true" in text
    assert "raw_image_stain: endothelial" in text
    assert "use_fwhm_edge_diameters" not in text


@pytest.mark.parametrize("name", ["fwhm_background_label", "fwhm_junction_label"])
def test_fwhms_internal_label_values_have_no_row_and_stay_at_their_defaults(panel, name):
    rows = panel._haemolynx_rows()
    assert FORCED_HIDDEN_SETTINGS[name] == SCHEMA[name].default
    rows["measure_diameters_from_raw_image"].value = True
    panel._haemolynx_expand_advanced(True)
    _process()
    assert rows[name] not in list(panel._haemolynx_diameters_settings)
    assert not _shown(panel, rows[name])
    assert all(name not in button.names for button in panel._haemolynx_advanced.values())
