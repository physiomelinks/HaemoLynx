"""The "Optimise FWHM settings" button, built for real, with a live
QApplication -- mirrors tests/test_gui_optimise_settings_widget.py's own
structure and rationale for the Input tab's "Optimise settings" button.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qapp):
    return settings_widget(napari_viewer=None)


def _fake_results(graph, voxel_size_zyx=(1.0, 1.0, 1.0)):
    return SimpleNamespace(_graph=graph, _voxel_size_zyx=voxel_size_zyx)


def _tiny_graph():
    import networkx as nx
    import numpy as np

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 10.0]))
    G.add_edge(0, 1, length=10.0, voxels=[(0.0, 0.0, float(x)) for x in range(11)])
    return G


# --- the button exists and is wired -----------------------------------------


def test_optimise_fwhm_button_exists_with_a_tooltip(panel):
    button = panel._haemolynx_optimise_fwhm_button
    assert button.text == "Optimise FWHM settings"
    assert button.tooltip.strip()


def test_optimise_fwhm_button_sits_right_after_the_raw_tiff_row(panel):
    """Per the feature request: 'under the master checkbox and input file
    line for FWHM, but before the other settings'."""
    container = panel._haemolynx_diameters_settings
    widgets = list(container)
    rows = panel._haemolynx_rows()
    raw_tiff_index = widgets.index(rows["fwhm_raw_tiff_path"])
    button_index = widgets.index(panel._haemolynx_optimise_fwhm_button)
    assert button_index == raw_tiff_index + 1
    # And before the next FWHM setting row.
    next_row_index = widgets.index(rows["fwhm_sample_spacing_along_edge_um"])
    assert button_index < next_row_index


def test_optimise_fwhm_button_hidden_until_fwhm_measurement_is_turned_on(panel):
    """A nested widget's own ``.visible`` only reflects reality once its
    top-level ancestor is shown *and* its tab is the active one (an inactive
    QTabWidget page stays hidden regardless) -- see
    test_gui_optimise_settings_widget.py's own
    test_ticking_choose_groups_reveals_the_checkbox_list for the same
    caveat; that one only needed `.show()` because its widget lives on the
    already-active "1. Input" tab, while this button lives on "5. Diameters"."""
    from haemolynx.gui.tabs import tab_titles

    panel.show()
    diameters_index = list(tab_titles()).index("5. Diameters")
    panel._haemolynx_tabs.setCurrentIndex(diameters_index)
    rows = panel._haemolynx_rows()
    rows["use_fwhm_edge_diameters"].value = False
    assert panel._haemolynx_optimise_fwhm_button.visible is False

    rows["use_fwhm_edge_diameters"].value = True
    assert panel._haemolynx_optimise_fwhm_button.visible is True

    rows["use_fwhm_edge_diameters"].value = False
    assert panel._haemolynx_optimise_fwhm_button.visible is False


# --- guards ------------------------------------------------------------------


def test_optimise_fwhm_settings_refuses_without_a_graph(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_optimise_fwhm_settings()
    assert not started
    assert "run the pipeline through at least Graph first" in panel._haemolynx_report()


def test_optimise_fwhm_settings_refuses_without_a_raw_path(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    panel._haemolynx_optimise_fwhm_settings()
    assert not started
    assert "Choose a raw FWHM image" in panel._haemolynx_report()


def test_optimise_fwhm_settings_refuses_while_a_run_is_already_going(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file

    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})
    try:
        panel._haemolynx_optimise_fwhm_settings()
    finally:
        panel._haemolynx_run_state.stopped()
    assert not started
    assert "already going" in panel._haemolynx_report()


# --- wiring: the button starts a background worker with the right arguments --


def test_optimise_fwhm_settings_starts_the_background_worker(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    graph = _tiny_graph()
    panel._haemolynx_view.results = _fake_results(graph, voxel_size_zyx=(2.0, 1.0, 1.0))
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file

    panel._haemolynx_optimise_fwhm_settings()

    assert len(started) == 1
    args, kwargs = started[0]
    passed_graph, voxel_size_zyx, settings, schema, passed_rows, report, button, bars = args
    assert passed_graph is not graph  # a copy, never the caller's own graph
    assert passed_graph.number_of_edges() == graph.number_of_edges()
    assert voxel_size_zyx == (2.0, 1.0, 1.0)
    assert passed_rows is panel._haemolynx_rows()
    assert button is panel._haemolynx_optimise_fwhm_button
    assert bars is panel._haemolynx_optimise_fwhm_bars
    assert kwargs["apply_prerequisites"] is not None
    assert kwargs["run_state"] is panel._haemolynx_run_state


def test_clicking_the_native_fwhm_button_triggers_optimisation(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_fwhm_optimisation_in_background",
        lambda *args, **kwargs: started.append(args),
    )
    panel._haemolynx_view.results = _fake_results(_tiny_graph())
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")
    panel._haemolynx_rows()["fwhm_raw_tiff_path"].value = raw_file
    panel._haemolynx_rows()["use_fwhm_edge_diameters"].value = True

    panel._haemolynx_optimise_fwhm_button.native.click()

    assert len(started) == 1


# --- clearing resets the button and bars --------------------------------------


def test_on_clear_resets_the_fwhm_optimise_button_and_bars(panel):
    panel._haemolynx_optimise_fwhm_button.enabled = False
    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})

    panel._haemolynx_clear(ask=False)

    assert panel._haemolynx_optimise_fwhm_button.enabled is True
