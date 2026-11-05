"""The Input tab's "Crop on import" box, built for real.

Which rows the box holds and where it sits is decided in
:mod:`haemolynx.gui.layout` (``test_crop_on_import.py`` checks that); these
check the panel draws it there, and that its rows follow the toggle and the
method as the user changes them.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from qtpy.QtWidgets import QApplication, QGroupBox  # noqa: E402

from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(make_napari_viewer):
    return settings_widget(napari_viewer=make_napari_viewer())


def _page(panel, title: str):
    tabs = panel._haemolynx_tabs
    for index in range(tabs.count()):
        if title in tabs.tabText(index):
            return tabs.widget(index).widget()
    raise AssertionError(f"no tab {title!r}")


def _shown(panel, widget) -> bool:
    return widget.native.isVisibleTo(_page(panel, "Input"))


def test_the_crop_box_is_drawn_on_the_input_tab_right_under_segmentation_cleanup(panel):
    page = _page(panel, "Input")
    cleanup = page.findChild(QGroupBox, "haemolynx_section_segmentation_cleanup")
    crop = page.findChild(QGroupBox, "haemolynx_section_crop_on_import")

    assert crop is not None and crop.title() == "Crop on import"
    layout = crop.parentWidget().layout()
    assert layout.indexOf(crop) == layout.indexOf(cleanup) + 1
    assert crop.isAncestorOf(panel._haemolynx_rows()["crop_on_import"].native)


def test_the_crop_rows_follow_the_toggle_and_the_method(panel):
    rows = panel._haemolynx_rows()
    by_size = ("crop_size_xyz", "crop_location", "crop_z_location")
    by_coordinates = ("crop_start_xyz", "crop_end_xyz")

    assert _shown(panel, rows["crop_on_import"])
    for name in ("crop_method", *by_size, *by_coordinates):
        assert not _shown(panel, rows[name]), name

    rows["crop_on_import"].value = True
    QApplication.processEvents()
    for name in ("crop_method", *by_size):
        assert _shown(panel, rows[name]), name
    for name in by_coordinates:
        assert not _shown(panel, rows[name]), name

    rows["crop_method"].value = "coordinates"
    QApplication.processEvents()
    for name in by_coordinates:
        assert _shown(panel, rows[name]), name
    for name in by_size:
        assert not _shown(panel, rows[name]), name
