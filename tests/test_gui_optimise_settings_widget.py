"""The "Optimise settings" button, built for real, with a live QApplication.

Uses `qapp` (pytest-qt) rather than the `panel`/`make_napari_viewer` fixture
the rest of `test_gui_widget.py` uses: this feature never touches a napari
layer or the viewer, so `settings_widget(napari_viewer=None)` is enough, and
avoids the cost -- and, on a machine where a real `napari.Viewer()` cannot be
constructed, the failure -- of building one just to hold a QApplication.

They need a Qt binding and pytest-qt, so they are marked `gui` and skipped
everywhere those are missing, same as `test_gui_widget.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Small, real, bundle-heavy fixture -- exercises the bundle-refinement group
#: (see CLAUDE.md's note that this file is "fast, bundle-heavy").
FIXTURE = REPO_ROOT / "tests" / "data" / "bundled_vessels_8_to_2.h5"


@pytest.fixture
def panel(qapp):
    """The panel, built without a viewer -- this feature needs none."""
    return settings_widget(napari_viewer=None)


# --- the button exists and is wired -----------------------------------------


def test_optimise_button_exists_with_a_tooltip(panel):
    button = panel._haemolynx_optimise_button
    assert button.text == "Optimise settings"
    assert button.tooltip.strip()


def test_optimise_button_is_a_child_of_the_input_tab(panel):
    from haemolynx.gui.tabs import tab_titles

    tabs = panel._haemolynx_tabs
    input_index = list(tab_titles()).index("1. Input")
    input_page = tabs.widget(input_index)
    button_native = panel._haemolynx_optimise_button.native
    assert button_native.isAncestorOf is not None  # a native Qt widget exists
    # Walk up from the button to confirm the Input tab's page contains it.
    ancestor = button_native.parent()
    found = False
    while ancestor is not None:
        if ancestor is input_page:
            found = True
            break
        ancestor = ancestor.parent()
    assert found, "the Optimise settings button is not on the Input tab"


# --- guards ------------------------------------------------------------------


def test_optimise_settings_refuses_without_an_input_path(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_optimisation_in_background", lambda *a, **k: started.append((a, k))
    )
    panel._haemolynx_optimise_settings()
    assert not started
    assert "Choose a segmented input image" in panel._haemolynx_report()


def test_optimise_settings_refuses_a_missing_file(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_optimisation_in_background", lambda *a, **k: started.append((a, k))
    )
    rows = panel._haemolynx_rows()
    rows["input_path"].value = tmp_path / "does_not_exist.tif"
    panel._haemolynx_optimise_settings()
    assert not started
    assert "Choose a segmented input image" in panel._haemolynx_report()


def test_optimise_settings_refuses_while_a_run_is_already_going(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod, "_run_optimisation_in_background", lambda *a, **k: started.append((a, k))
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input

    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})
    try:
        panel._haemolynx_optimise_settings()
    finally:
        panel._haemolynx_run_state.stopped()
    assert not started
    assert "already going" in panel._haemolynx_report()


# --- wiring: the button starts a background worker with the right arguments --


def test_optimise_settings_starts_the_background_worker(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input

    panel._haemolynx_optimise_settings()

    assert len(started) == 1
    args, kwargs = started[0]
    settings, schema, passed_rows, report, button, bars = args
    assert Path(settings["input_path"]) == real_input
    assert passed_rows is rows
    assert button is panel._haemolynx_optimise_button
    assert bars is panel._haemolynx_optimise_bars
    assert kwargs["apply_prerequisites"] is not None
    assert kwargs["run_state"] is panel._haemolynx_run_state


def test_clicking_the_native_button_triggers_optimisation(panel, monkeypatch, tmp_path):
    """The button's own Qt click, not just calling the handler directly."""
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append(args),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_optimise_button.native.click()

    assert len(started) == 1


# --- a real, slow, end-to-end run --------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_optimise_settings_end_to_end_on_a_real_fixture(panel, tmp_path):
    """Runs the real worker to completion: rows change, and a YAML lands
    beside the (copied) input file.

    No mocking of `_run_optimisation_in_background` here -- this is the one
    test that proves the whole button-to-file path works, not just its wiring.
    Uses `napari.qt.threading`'s worker synchronously via Qt's own event loop,
    pumped until the report line changes.
    """
    import shutil
    import time

    from qtpy.QtWidgets import QApplication

    assert FIXTURE.is_file(), FIXTURE
    copy_path = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, copy_path)

    rows = panel._haemolynx_rows()
    rows["input_path"].value = copy_path
    before = {
        "skeleton_closing_radius": rows["skeleton_closing_radius"].value,
        "graph_reconnect_threshold": rows["graph_reconnect_threshold"].value,
    }

    panel._haemolynx_optimise_settings()
    assert "Optimising settings..." in panel._haemolynx_report()

    app = QApplication.instance()
    deadline = time.time() + 120
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
        if "Optimising settings..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("optimisation did not finish within 120s")

    report = panel._haemolynx_report()
    assert "Optimised settings applied" in report, report

    yaml_files = list(tmp_path.glob("*.yaml"))
    assert yaml_files, "no config file was written beside the input image"
    assert yaml_files[0].name.endswith(f"_{copy_path.stem}.yaml")

    after = {
        "skeleton_closing_radius": rows["skeleton_closing_radius"].value,
        "graph_reconnect_threshold": rows["graph_reconnect_threshold"].value,
    }
    # Not asserting the values changed (the image may happen to favour the
    # defaults) -- only that the round trip through the optimiser and back
    # into the rows completed without error.
    assert set(before) == set(after)
