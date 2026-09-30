"""The panel's Advanced buttons, built for real.

Where each button goes and what it holds is decided in
:mod:`haemolynx.gui.layout` and tested in ``test_gui_layout.py``; these check
that the panel draws it: rows start hidden behind their button, a button
follows the row it hangs off, and its count says what is changed out of sight.

Visibility is read with ``isVisibleTo(page)`` -- "would this show if the page
did" -- so a test need not put each tab on screen first.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.perturbation_editing import visible_tab_settings  # noqa: E402
from haemolynx.gui.tabs import tabs_for  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402

pytestmark = pytest.mark.gui

SCHEMA = default_schema()


@pytest.fixture
def panel(make_napari_viewer):
    return settings_widget(napari_viewer=make_napari_viewer())


def _page(panel, title: str):
    tabs = panel._haemolynx_tabs
    for index in range(tabs.count()):
        if title in tabs.tabText(index):
            return tabs.widget(index).widget()
    raise AssertionError(f"no tab {title!r}")


def _shown(panel, title: str, widget) -> bool:
    return widget.native.isVisibleTo(_page(panel, title))


def _process():
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def test_an_advanced_row_starts_hidden_and_shows_once_its_button_is_opened(panel):
    rows = panel._haemolynx_rows()
    button = panel._haemolynx_advanced["skeletonise:do_skeletonize"]
    bundle = rows["skeleton_bundle_scan_size"]

    assert _shown(panel, "Skeletonise", button.container)
    assert not _shown(panel, "Skeletonise", bundle)
    assert button.button.text.startswith("▸ Advanced skeletonisation")

    button.toggle()
    assert _shown(panel, "Skeletonise", bundle)
    assert button.button.text.startswith("▾")

    button.toggle()
    assert not _shown(panel, "Skeletonise", bundle)


def test_clicking_the_button_opens_it(panel):
    button = panel._haemolynx_advanced["skeletonise:do_skeletonize"]
    button.button.native.click()
    _process()
    assert button.expanded
    assert _shown(panel, "Skeletonise", panel._haemolynx_rows()["skeleton_bundle_scan_size"])


def test_a_button_is_hidden_while_the_row_it_hangs_off_is_off(panel):
    rows = panel._haemolynx_rows()
    fwhm = panel._haemolynx_advanced["assign_diameters:use_fwhm_edge_diameters"]

    rows["use_fwhm_edge_diameters"].value = False
    _process()
    assert not _shown(panel, "Diameters", fwhm.container)

    rows["use_fwhm_edge_diameters"].value = True
    _process()
    assert _shown(panel, "Diameters", fwhm.container)


def test_the_decoy_check_is_behind_a_button_under_do_fwhm_measurement(panel):
    rows = panel._haemolynx_rows()
    decoy = panel._haemolynx_advanced["assign_diameters:do_fwhm_measurement"]
    fwhm = panel._haemolynx_advanced["assign_diameters:use_fwhm_edge_diameters"]
    rows["use_fwhm_edge_diameters"].value = True
    _process()

    assert "fwhm_decoy_check" in decoy.names
    assert decoy.spec.level > fwhm.spec.level
    assert not _shown(panel, "Diameters", rows["fwhm_decoy_check"])
    decoy.toggle(True)
    assert _shown(panel, "Diameters", rows["fwhm_decoy_check"])

    # Nothing to check while FWHM keeps the diameters already on the graph.
    rows["do_fwhm_measurement"].value = False
    _process()
    assert not _shown(panel, "Diameters", decoy.container)
    assert _shown(panel, "Diameters", fwhm.container)

    diameters = panel._haemolynx_diameters_settings
    order = list(diameters)
    assert order.index(rows["do_fwhm_measurement"]) < order.index(rows["use_raw_section_fallback"])


def test_viscosity_law_is_an_ordinary_row(panel):
    rows = panel._haemolynx_rows()
    assert _shown(panel, "Haemodynamics", rows["viscosity_law"])
    assert _shown(panel, "Diameters", rows["do_fwhm_measurement"]) is bool(
        rows["use_fwhm_edge_diameters"].value
    )


def test_the_count_says_how_many_hidden_settings_are_changed(panel):
    rows = panel._haemolynx_rows()
    button = panel._haemolynx_advanced["skeletonise:do_skeletonize"]
    assert "changed" not in button.button.text

    rows["skeleton_bundle_scan_size"].value = 11
    _process()
    assert button.changed == 1
    assert button.button.text.endswith("(1 changed)")

    rows["skeleton_bundle_scan_size"].value = SCHEMA["skeleton_bundle_scan_size"].default
    _process()
    assert button.changed == 0
    assert "changed" not in button.button.text


def test_a_loaded_config_that_changes_a_hidden_setting_says_so(panel, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("skeleton_bundle_density_fraction: 0.5\n", encoding="utf-8")
    panel._haemolynx_load_config(config)
    _process()
    button = panel._haemolynx_advanced["skeletonise:do_skeletonize"]
    assert button.changed == 1


def test_ide_plots_being_off_in_napari_is_not_a_change(panel):
    export = panel._haemolynx_advanced["export_results:box:export"]
    assert panel._haemolynx_values()["visualize_results"] is False
    assert export.changed == 0


def test_every_advanced_setting_the_panel_lays_out_is_behind_a_button(panel):
    behind = {name for button in panel._haemolynx_advanced.values() for name in button.names}
    tabs = {tab.stage.call: tab for tab in tabs_for(SCHEMA)}
    perturbation_rows = {f.name for f in tabs["run_perturbations"].fields}
    elsewhere = perturbation_rows - set(visible_tab_settings(perturbation_rows))
    missing = [
        setting.name
        for setting in SCHEMA
        if setting.advanced
        and setting.name not in elsewhere
        and setting.name not in {"flow_direction_colouring", "flow_arrow_scale"}
        and setting.name not in behind
    ]
    assert missing == []


def test_the_shared_ilastik_rows_move_with_their_own_button(panel):
    rows = panel._haemolynx_rows()
    block = panel._haemolynx_shared_ilastik_block
    advanced = panel._haemolynx_advanced["shared:ilastik"]
    assert set(advanced.names) == {"ilastik_output_dir", "ilastik_output_suffix", "ilastik_timeout_seconds"}

    rows["use_ilastik_segmentation"].value = True
    _process()
    assert _page(panel, "Input").isAncestorOf(block.native)
    assert _shown(panel, "Input", rows["ilastik_executable"])
    assert not _shown(panel, "Input", rows["ilastik_output_dir"])
    advanced.toggle(True)
    assert _shown(panel, "Input", rows["ilastik_output_dir"])

    rows["use_ilastik_segmentation"].value = False
    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["use_ilastik_large_vessel_segmentation"].value = True
    _process()
    assert _page(panel, "Boundaries").isAncestorOf(block.native)
    assert block.native.isAncestorOf(advanced.container.native)
    assert _shown(panel, "Boundaries", rows["ilastik_output_dir"])


def test_the_check_and_optimise_box_keeps_its_options_behind_its_button(panel):
    button = panel._haemolynx_advanced["segment:box:check_and_optimise"]
    body = button.body.native
    for widget in (
        panel._haemolynx_optimise_downsample,
        panel._haemolynx_optimise_choose_groups,
        panel._haemolynx_optimise_group_checkboxes_container,
    ):
        assert body.isAncestorOf(widget.native)
    for widget in (
        panel._haemolynx_optimise_button,
        panel._haemolynx_check_image_button,
        panel._haemolynx_raw_data_row,
        panel._haemolynx_raw_channel_row,
    ):
        assert not body.isAncestorOf(widget.native)
        assert _shown(panel, "Input", widget)
    assert not _shown(panel, "Input", panel._haemolynx_optimise_downsample)


def test_an_h5_dataset_row_shows_only_for_an_hdf5_file(panel):
    rows = panel._haemolynx_rows()
    button = next(
        b for b in panel._haemolynx_advanced.values() if "cell_mask_h5_dataset_name" in b.names
    )
    button.toggle(True)
    rows["measurement_3d_to_cell_mask"].value = True
    rows["cell_mask_path"].value = "cells.tif"
    _process()
    assert not _shown(panel, "Additional measurements", rows["cell_mask_h5_dataset_name"])
    rows["cell_mask_path"].value = "cells.h5"
    _process()
    assert _shown(panel, "Additional measurements", rows["cell_mask_h5_dataset_name"])


def test_a_role_pages_node_ids_are_behind_its_own_button(panel):
    rows = panel._haemolynx_rows()
    boundaries = panel._haemolynx_boundaries
    holder = boundaries.holders["inlet"].native
    button = panel._haemolynx_advanced["assign_boundaries:role:inlet"]
    assert button.names == ("inlet_nodes",)
    assert holder.isAncestorOf(button.container.native)
    assert not rows["inlet_nodes"].native.isVisibleTo(holder)
    button.toggle(True)
    # No selection method hides the node IDs: they are the run's own record.
    for method in SCHEMA["inlet_node_selection_method"].choices:
        rows["inlet_node_selection_method"].value = method
        _process()
        assert rows["inlet_nodes"].native.isVisibleTo(holder), method


def test_a_perturbation_entry_keeps_its_advanced_options_behind_its_own_button(panel):
    rows = panel._haemolynx_rows()
    rows["run_perturbations"].value = True
    perturbations = panel._haemolynx_perturbations
    perturbations.add()
    perturbations.choose_type(0, "capillary_block")
    _process()
    (editor,) = perturbations.editors()
    container = editor.container.native
    body = editor.advanced.body.native

    assert set(editor.advanced.names) >= {"capillary_block_seed", "capillary_block_resistance_factor"}
    assert body.isAncestorOf(editor.editors["capillary_block_seed"].native)
    assert not body.isAncestorOf(editor.editors["capillary_block_probability"].native)
    assert not editor.editors["capillary_block_seed"].native.isVisibleTo(container)
    editor.advanced.toggle(True)
    assert editor.editors["capillary_block_seed"].native.isVisibleTo(container)

    # The entry's own selection decides which of its options show.
    assert editor.editors["capillary_block_probability"].native.isVisibleTo(container)
    assert not editor.editors["capillary_block_vessel_ids"].native.isVisibleTo(container)
    editor.editors["capillary_block_selection"].value = "vessel_ids"
    _process()
    assert not editor.editors["capillary_block_probability"].native.isVisibleTo(container)
    assert editor.editors["capillary_block_vessel_ids"].native.isVisibleTo(container)

    # A hidden option is not one of the entry's overrides either.
    overrides = perturbations.entries()[0]["overrides"]
    assert overrides["capillary_block_selection"] == "vessel_ids"
    assert "capillary_block_vessel_ids" in overrides
    for name in ("capillary_block_probability", "capillary_block_branch_orders", "capillary_block_seed"):
        assert name not in overrides, name


def test_expand_advanced_opens_and_closes_every_button(panel):
    panel._haemolynx_expand_advanced(True)
    assert all(button.expanded for button in panel._haemolynx_advanced.values())
    panel._haemolynx_expand_advanced(False)
    assert not any(button.expanded for button in panel._haemolynx_advanced.values())


def test_every_advanced_button_has_hover_text(panel):
    for key, button in panel._haemolynx_advanced.items():
        assert button.button.tooltip.strip(), key
