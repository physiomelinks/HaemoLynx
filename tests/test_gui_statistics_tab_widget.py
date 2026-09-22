"""Tab 8 boxes "Connectivity/Network Analysis" apart from the flat
"Statistics and measurements" list above it, instead of running both
together as one undifferentiated column of checkboxes.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from qtpy.QtWidgets import QGroupBox, QTabWidget  # noqa: E402

from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui


def test_network_analysis_measures_sit_in_their_own_boxed_group(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    tabs = panel.findChild(QTabWidget)
    index = next(
        i for i in range(tabs.count()) if tabs.tabText(i) == "8. Additional measurements"
    )
    page = tabs.widget(index)

    group = page.findChild(QGroupBox, "haemolynx_section_connectivity_network_analysis")
    assert group is not None, "no boxed Connectivity/Network Analysis group on tab 8"
    assert group.title() == "Connectivity/Network Analysis"

    rows = panel._haemolynx_rows()
    network_toggle = rows["statistics_network_analysis"].native
    a_network_measure = rows["statistics_cyclomatic_number"].native
    a_plain_statistic = rows["statistics_basic"].native

    assert group.isAncestorOf(network_toggle)
    assert group.isAncestorOf(a_network_measure)
    assert not group.isAncestorOf(a_plain_statistic), (
        "a plain Statistics row should stay in the flat list above the box, "
        "not fall inside the nested Connectivity/Network Analysis group"
    )
    assert page.isAncestorOf(a_plain_statistic)


def test_network_analysis_group_hides_when_statistics_is_off(make_napari_viewer):
    """Regression: with `statistics` unchecked, every row inside the
    Connectivity/Network Analysis box hides (statistics_network_analysis
    requires `statistics`), but the box itself used to stay visible with
    nothing in it -- an empty titled frame, not "no separate checkbox" as
    the nested group is supposed to read while off."""
    from qtpy.QtWidgets import QApplication

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    panel.show()
    tabs = panel.findChild(QTabWidget)
    index = next(
        i for i in range(tabs.count()) if tabs.tabText(i) == "8. Additional measurements"
    )
    tabs.setCurrentIndex(index)
    QApplication.processEvents()
    page = tabs.widget(index)
    group = page.findChild(QGroupBox, "haemolynx_section_connectivity_network_analysis")

    rows = panel._haemolynx_rows()
    rows["statistics"].value = False
    QApplication.processEvents()
    assert group.isVisible() is False

    rows["statistics"].value = True
    QApplication.processEvents()
    assert group.isVisible() is True
