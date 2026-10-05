"""The ? under the tab scrollbar opens a two-pane help window on the open tab."""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from qtpy.QtWidgets import QApplication  # noqa: E402

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.chrome_tooltips import HELP_TOOLTIP  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qapp):
    widget = settings_widget(napari_viewer=None)
    widget.resize(420, 900)
    widget.show()
    QApplication.processEvents()
    return widget


def _help_windows():
    return [
        window
        for window in QApplication.topLevelWidgets()
        if window.objectName() == "haemolynx_help_window"
    ]


def test_the_help_button_sits_to_the_right_of_the_show_checkboxes(panel):
    view_controls = panel._haemolynx_view_controls.native
    button = panel._haemolynx_help_button
    show_results = panel._haemolynx_show_results.native
    show_steps = panel._haemolynx_show_steps.native
    tabs = panel._haemolynx_tabs

    assert button.text() == "?"
    assert button.toolTip() == HELP_TOOLTIP
    assert view_controls.objectName() == "haemolynx_view_controls"
    ancestor = button.parentWidget()
    while ancestor is not None and ancestor is not view_controls:
        ancestor = ancestor.parentWidget()
    assert ancestor is view_controls

    layout = panel.layout()
    assert layout.indexOf(view_controls) > layout.indexOf(tabs)

    results_right = show_results.mapTo(panel, show_results.rect().topRight()).x()
    steps_right = show_steps.mapTo(panel, show_steps.rect().topRight()).x()
    help_left = button.mapTo(panel, button.rect().topLeft()).x()
    assert help_left >= max(results_right, steps_right) - 2

    column_top = min(
        show_results.mapTo(panel, show_results.rect().topLeft()).y(),
        show_steps.mapTo(panel, show_steps.rect().topLeft()).y(),
    )
    column_bottom = max(
        show_results.mapTo(panel, show_results.rect().bottomLeft()).y(),
        show_steps.mapTo(panel, show_steps.rect().bottomLeft()).y(),
    )
    help_mid = button.mapTo(panel, button.rect().center()).y()
    assert column_top <= help_mid <= column_bottom

    tabs_bottom = tabs.mapTo(panel, tabs.rect().bottomLeft()).y()
    help_top = button.mapTo(panel, button.rect().topLeft()).y()
    assert help_top >= tabs_bottom - 2


def test_help_opens_on_the_current_tab_with_an_empty_page(panel):
    tabs = panel._haemolynx_tabs
    tabs.setCurrentIndex(2)
    QApplication.processEvents()
    panel._haemolynx_help_button.click()
    QApplication.processEvents()

    windows = _help_windows()
    assert len(windows) == 1
    window = windows[0]
    assert window.isVisible()
    directory = window.directory
    titles = [directory.topLevelItem(i).text(0) for i in range(directory.topLevelItemCount())]
    assert titles == [tabs.tabText(i) for i in range(tabs.count())]
    assert directory.currentItem().text(0) == tabs.tabText(2)
    assert window.text.toPlainText() == ""
    assert window.text.isReadOnly()

    tabs.setCurrentIndex(4)
    panel._haemolynx_help_button.click()
    QApplication.processEvents()
    assert _help_windows() == [window]
    assert directory.currentItem().text(0) == tabs.tabText(4)
    assert window.text.toPlainText() == ""
    window.close()
