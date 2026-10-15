"""The left panel's Statistics tab, built for real: the second of its four
tabs, a button per measure the last run's statistics computed with that
measure's results hidden behind it, and kept up to date as statistics rerun."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    LAYER_CONTROLS_TITLE,
    LAYER_LIST_TITLE,
    STATISTICS_DOCK_NAME,
    VIEW_DOCK_NAME,
    settings_widget,
)
from haemolynx.gui.results import ResultLayers  # noqa: E402
from haemolynx.gui.statistics_view import section_tooltip, statistics_sections  # noqa: E402
from haemolynx.pipeline.stages import HAEMODYNAMICS_SECTION, STATISTICS_REPORT  # noqa: E402
from test_gui_view_panel import _left_tabs, _settle  # noqa: E402

pytestmark = pytest.mark.gui

FIXTURE = Path(__file__).resolve().parent / "data" / "seven_vessel_noisy_3d.tif"


def _report(**sections):
    return {"mode": "fast", "csv": "out/run_statistics.csv", "sections": sections}


FIRST = _report(
    basic={"Total Nodes": 7, "Total Edges": 6},
    murray_law={"Mean Murray Ratio": 0.98},
    dead_ends={"Dead Ends": {"Dead Ends": 3}},
    **{HAEMODYNAMICS_SECTION: {"Two-Point Equivalent Resistance (Pa.s/m^3)": 2.5e15}},
)


def _show_network_with(panel, viewer, report) -> None:
    """Put a network carrying *report* on screen, as a finished stage does."""
    G = nx.MultiGraph()
    if report is not None:
        G.graph[STATISTICS_REPORT] = report
    results = ResultLayers()
    results._graph = G
    panel._haemolynx_view.results = results
    viewer._haemolynx_after_layers_applied()


def _buttons(panel):
    stats = panel._haemolynx_statistics_panel
    return {key: button.text() for key, button in stats.buttons.items()}


def test_the_statistics_tab_is_second_on_the_left_and_empty_before_a_run(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    panel = settings_widget(napari_viewer=viewer)
    _settle()

    assert _left_tabs(viewer) == (
        [VIEW_DOCK_NAME, STATISTICS_DOCK_NAME, LAYER_LIST_TITLE, LAYER_CONTROLS_TITLE],
        VIEW_DOCK_NAME,
    )
    dock = panel._haemolynx_statistics_dock
    stats = panel._haemolynx_statistics_panel
    assert dock.widget().widget() is stats.widget
    assert stats.buttons == {}
    assert stats.summary.text().startswith("No statistics yet")
    assert "9. Additional measurements" in stats.summary.text()


def test_a_button_per_section_each_hiding_its_results_until_pressed(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    panel = settings_widget(napari_viewer=viewer)
    stats = panel._haemolynx_statistics_panel

    _show_network_with(panel, viewer, FIRST)

    assert _buttons(panel) == {
        "basic": "▸ Basic",
        "murray_law": "▸ Murray law",
        "dead_ends": "▸ Dead ends",
        HAEMODYNAMICS_SECTION: "▸ Haemodynamics",
    }
    assert stats.summary.text().startswith("4 results from the last run in fast mode.")
    # Nothing behind a button is built, let alone shown, until it is pressed.
    assert stats.bodies == {}
    # Each button's hover text explains its own statistic.
    tooltips = {key: button.toolTip() for key, button in stats.buttons.items()}
    assert tooltips == {key: section_tooltip(key) for key in stats.buttons}
    assert len(set(tooltips.values())) == len(tooltips)
    assert "Murray's law" in tooltips["murray_law"]

    stats.buttons["dead_ends"].click()
    _settle()
    assert stats.buttons["dead_ends"].text() == "▾ Dead ends"
    assert stats.bodies["dead_ends"].isVisible()
    assert stats.rows_shown("dead_ends") == [("Dead Ends", ""), ("Dead Ends", "3")]
    assert stats.rows_shown("basic") == []
    # Below its own button, above the next one.
    layout = stats.bodies["dead_ends"].parentWidget().layout()
    below = layout.indexOf(stats.buttons["dead_ends"]) + 1
    assert layout.indexOf(stats.bodies["dead_ends"]) == below

    stats.buttons["basic"].click()
    assert stats.rows_shown("basic") == [("Total Nodes", "7"), ("Total Edges", "6")]
    assert stats.bodies["dead_ends"].isVisible(), "pressing one leaves the others as they are"

    stats.buttons["dead_ends"].click()
    assert not stats.bodies["dead_ends"].isVisible()
    assert stats.buttons["dead_ends"].text() == "▸ Dead ends"


def test_rerun_statistics_replace_the_buttons_and_keep_open_what_was_open(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    panel = settings_widget(napari_viewer=viewer)
    stats = panel._haemolynx_statistics_panel
    _show_network_with(panel, viewer, FIRST)
    stats.buttons["basic"].click()
    stats.buttons["murray_law"].click()

    rerun = _report(basic={"Total Nodes": 9}, k_core={"Max Core Number": 2})
    _show_network_with(panel, viewer, rerun)
    _settle()

    assert list(stats.buttons) == [section.key for section in statistics_sections(rerun)]
    assert _buttons(panel) == {"basic": "▾ Basic", "k_core": "▸ K core"}
    assert stats.bodies["basic"].isVisible()
    assert stats.rows_shown("basic") == [("Total Nodes", "9")]
    assert "k_core" not in stats.bodies

    # A measure back after a run without it opens as the user left it.
    _show_network_with(panel, viewer, FIRST)
    assert stats.buttons["murray_law"].isChecked()
    assert stats.rows_shown("murray_law") == [("Mean Murray Ratio", "0.98")]


def test_a_stage_before_the_export_empties_the_tab_and_clear_does_too(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    panel = settings_widget(napari_viewer=viewer)
    stats = panel._haemolynx_statistics_panel
    _show_network_with(panel, viewer, FIRST)

    # A run from an earlier tab: the network on screen is one no export measured.
    _show_network_with(panel, viewer, None)
    assert stats.buttons == {}
    assert stats.summary.text().startswith("No statistics yet")

    _show_network_with(panel, viewer, FIRST)
    panel._haemolynx_clear()
    assert stats.buttons == {}


def test_view_controls_rebuilds_a_closed_statistics_tab_second_with_its_results(
    make_napari_viewer,
):
    viewer = make_napari_viewer(show=True)
    panel = settings_widget(napari_viewer=viewer)
    _settle()
    _show_network_with(panel, viewer, FIRST)
    dock = panel._haemolynx_statistics_dock
    dock.destroyOnClose()
    _settle()

    panel._haemolynx_view_button.native.click()
    _settle()

    rebuilt = panel._haemolynx_statistics_dock
    assert rebuilt is not dock
    assert not rebuilt.isHidden()
    assert rebuilt.widget().widget() is panel._haemolynx_statistics_panel.widget
    assert _left_tabs(viewer) == (
        [VIEW_DOCK_NAME, STATISTICS_DOCK_NAME, LAYER_LIST_TITLE, LAYER_CONTROLS_TITLE],
        VIEW_DOCK_NAME,
    )
    assert "murray_law" in panel._haemolynx_statistics_panel.buttons


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("show_results", [True, False])
def test_a_real_run_fills_the_tab_and_a_rerun_of_the_statistics_updates_it(
    make_napari_viewer, qtbot, tmp_path, show_results
):
    """Shown in the viewer or not, the run's statistics reach the tab; Run
    from this stage on Additional measurements, with a measure turned off,
    takes its button away."""
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    rows = widget._haemolynx_rows()
    rows["input_path"].value = FIXTURE
    rows["vtk_output_prefix"].value = tmp_path / "out" / "run"
    rows["statistics"].value = True
    widget._haemolynx_show_results.value = show_results
    running = widget._haemolynx_run_state
    stats = widget._haemolynx_statistics_panel

    def wait():
        qtbot.waitUntil(lambda: not running.running, timeout=600_000)
        qtbot.wait(200)

    widget._haemolynx_run()
    wait()

    keys = list(stats.buttons)
    assert {"basic", "murray_law", "dead_ends", "branch_orders"} <= set(keys)
    assert keys.index("basic") < keys.index("murray_law") < keys.index("branch_orders")
    assert stats.summary.text().endswith(f"Saved to {stats.report['csv']}")
    assert Path(stats.report["csv"]).name == "seven_vessel_noisy_3d_statistics.csv"
    assert Path(stats.report["csv"]).is_file()
    stats.buttons["basic"].click()
    nodes = dict(stats.rows_shown("basic"))["Total Nodes"]
    assert int(nodes) > 0

    if not show_results:
        return
    rows["statistics_murray_law"].value = False
    widget._haemolynx_run_from("9. Additional measurements")
    wait()

    assert "murray_law" not in stats.buttons
    assert "basic" in stats.buttons and stats.buttons["basic"].isChecked()
    assert dict(stats.rows_shown("basic"))["Total Nodes"] == nodes
