"""The panel's Preset menu, under Load config... and Save config...

Choosing a preset sets the rows it names (`pipeline/presets.py`) and no
others, and holds nothing afterwards: a row changed by hand keeps its value,
and a config or run loaded afterwards replaces the preset and puts the menu
back to None.
"""
from __future__ import annotations

from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    DISPLAY_SETTINGS_OFF_IN_NAPARI,
    FORCED_HIDDEN_SETTINGS,
    settings_widget,
)
from haemolynx.gui.chrome_tooltips import PRESET_TOOLTIP  # noqa: E402
from haemolynx.gui.form import field_for  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from haemolynx.pipeline.presets import (  # noqa: E402
    NO_PRESET,
    PRESETS,
    preset_menu,
    preset_settings,
)

pytestmark = pytest.mark.gui

SCHEMA = default_schema()


@pytest.fixture
def panel(make_napari_viewer):
    make_napari_viewer()
    return settings_widget()


def _choose(panel, name: str | None) -> None:
    """Pick *name* in the menu as a user does (None: the "None" entry)."""
    menu = panel._haemolynx_preset
    menu.activated.emit(menu.findData(name) if name is not None else 0)


def test_the_menu_starts_at_none_and_offers_every_preset(panel):
    menu = panel._haemolynx_preset
    assert menu.currentIndex() == 0
    assert menu.currentText() == NO_PRESET
    assert menu.itemData(0) is None
    assert menu.toolTip() == PRESET_TOOLTIP
    entries = [entry for entry in preset_menu() if entry is not None]
    named = [
        (menu.itemText(i), menu.itemData(i))
        for i in range(menu.count())
        if menu.itemText(i)
    ]
    assert named == entries
    assert menu.count() == len(preset_menu())
    for name, spec in PRESETS.items():
        assert menu.itemData(menu.findData(name), Qt.ToolTipRole) == spec["description"]


def test_the_menu_sits_under_load_and_save_config(panel):
    from qtpy.QtWidgets import QPushButton

    row = panel._haemolynx_run_file_row
    assert row.layout().indexOf(panel._haemolynx_preset) == 1
    assert row.layout().indexOf(panel._haemolynx_view_button.native) > 1
    load_button = next(
        button for button in panel.findChildren(QPushButton) if button.text() == "Load config..."
    )
    buttons_row = load_button.parentWidget()
    layout = panel.layout()
    assert layout.indexOf(row) == layout.indexOf(buttons_row) + 1


def test_every_setting_a_preset_names_is_a_row_the_panel_leaves_to_the_user(panel):
    rows = panel._haemolynx_rows()
    for name, spec in PRESETS.items():
        for setting in spec["overrides"]:
            assert setting in rows, (name, setting)
            assert setting not in FORCED_HIDDEN_SETTINGS, (name, setting)
            assert setting not in DISPLAY_SETTINGS_OFF_IN_NAPARI, (name, setting)


def test_each_preset_lands_in_the_form_as_it_says(panel):
    """Every preset in turn, on one panel: each value reaches its widget and
    reads back as the setting it names."""
    for name in PRESETS:
        _choose(panel, name)
        values = panel._haemolynx_values()
        for setting, value in preset_settings(name, SCHEMA).items():
            assert values[setting] == value, (name, setting)
        assert panel._haemolynx_preset.currentData() == name
        assert panel._haemolynx_report().startswith(f"Loaded preset {PRESETS[name]['label']}")


def test_a_preset_sets_the_rows_it_names_and_leaves_the_others(panel):
    rows = panel._haemolynx_rows()
    rows["min_stub_length"].value = 42.5
    before = panel._haemolynx_values()

    _choose(panel, "large_image")

    after = panel._haemolynx_values()
    changed = {name for name in after if after[name] != before[name]}
    assert changed == {"use_memmap_loading", "haemodynamics_solver", "equivalent_resistance_solver"}
    assert after["min_stub_length"] == 42.5
    report = panel._haemolynx_report()
    assert report.startswith("Loaded preset Large image, low memory: set ")
    for setting in changed:
        assert field_for(SCHEMA[setting]).label in report


def test_a_row_changed_after_a_preset_keeps_its_value_until_a_preset_sets_it(panel):
    rows = panel._haemolynx_rows()
    _choose(panel, "quick_look")
    assert panel._haemolynx_values()["statistics"] is False

    rows["statistics"].value = True
    _choose(panel, "large_image")
    assert panel._haemolynx_values()["statistics"] is True

    # The preset already shown, chosen again, loads again.
    _choose(panel, "quick_look")
    rows["statistics"].value = True
    assert panel._haemolynx_preset.currentData() == "quick_look"
    _choose(panel, "quick_look")
    assert panel._haemolynx_values()["statistics"] is False


def test_none_loads_nothing(panel):
    _choose(panel, "network_structure")
    before = panel._haemolynx_values()
    report = panel._haemolynx_report()

    _choose(panel, None)

    assert panel._haemolynx_values() == before
    assert panel._haemolynx_report() == report


def test_a_preset_says_when_its_settings_were_already_set(panel):
    _choose(panel, "thick_vessels")
    _choose(panel, "thick_vessels")
    assert panel._haemolynx_report() == (
        "Loaded preset Thick vessels: its settings were already set"
    )


def test_a_config_loaded_after_a_preset_replaces_it_and_puts_the_menu_back(panel, tmp_path):
    from haemolynx.parsers import dump_config

    _choose(panel, "quick_look")
    assert panel._haemolynx_values()["vtk_export"] is False

    config = tmp_path / "config.yaml"
    dump_config(
        config, SCHEMA,
        values={**{s.name: s.default for s in SCHEMA}, "statistics": True, "vtk_export": True},
    )
    panel._haemolynx_load_config(config)

    values = panel._haemolynx_values()
    assert values["statistics"] is True
    assert values["vtk_export"] is True
    assert values["do_equiv_resistance_calculation"] is True
    assert panel._haemolynx_preset.currentIndex() == 0
    assert panel._haemolynx_preset.currentText() == NO_PRESET


def test_a_config_that_fails_to_load_leaves_the_preset(panel, tmp_path):
    _choose(panel, "quick_look")
    broken = tmp_path / "broken.yaml"
    broken.write_text("not_a_setting: 1\n", encoding="utf-8")

    panel._haemolynx_load_config(broken)

    assert panel._haemolynx_report().startswith("Could not load")
    assert panel._haemolynx_preset.currentData() == "quick_look"
    assert panel._haemolynx_values()["vtk_export"] is False


def test_a_run_loaded_after_a_preset_puts_the_menu_back(panel, tmp_path):
    from test_gui_batch_run_widget import _saved_run

    _choose(panel, "full_report")
    assert panel._haemolynx_load_run(_saved_run(tmp_path)) is True
    assert panel._haemolynx_preset.currentIndex() == 0


def test_ticking_mask_switches_by_preset_searches_for_the_masks(panel, tmp_path):
    """A preset ticks the switches as a user would, so the masks beside the
    image are found, and the report says so under the preset's line."""
    image, art, ven = (tmp_path / name for name in (
        "E14_MCA_segmented.tif", "E14_MCA_large_arterioles.tif", "E14_MCA_large_veins.tif",
    ))
    for path in (image, art, ven):
        path.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = image

    _choose(panel, "arteriole_venule_masks")

    values = panel._haemolynx_values()
    assert Path(values["large_arteriole_mask_path"]) == art
    assert Path(values["large_venule_mask_path"]) == ven
    report = panel._haemolynx_report()
    assert report.startswith("Loaded preset Arteriole and venule masks: set ")
    # Both searches' lines, the large masks' and the small masks'.
    assert "E14_MCA_large_arterioles.tif" in report
    assert "No small vessel masks found" in report
