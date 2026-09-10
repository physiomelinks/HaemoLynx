"""The "Check segmented image" button, built for real, with a live QApplication.

Uses `qapp` (pytest-qt) rather than the `panel`/`make_napari_viewer` fixture the
rest of `test_gui_widget.py` uses -- this feature never touches a napari layer
or the viewer, so `settings_widget(napari_viewer=None)` is enough (same
precedent as `test_gui_optimise_settings_widget.py`, which this file mirrors).

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
FIXTURE = REPO_ROOT / "tests" / "data" / "bundled_vessels_8_to_2.h5"


@pytest.fixture
def panel(qapp):
    """The panel, built without a viewer -- this feature needs none."""
    return settings_widget(napari_viewer=None)


# --- the button exists and is wired -----------------------------------------


def test_check_image_button_exists_with_a_tooltip(panel):
    button = panel._haemolynx_check_image_button
    assert button.text == "Check segmented image"
    assert button.tooltip.strip()


def test_check_image_button_is_a_child_of_the_input_tab(panel):
    from haemolynx.gui.tabs import tab_titles

    tabs = panel._haemolynx_tabs
    input_index = list(tab_titles()).index("1. Input")
    input_page = tabs.widget(input_index)
    button_native = panel._haemolynx_check_image_button.native
    ancestor = button_native.parent()
    found = False
    while ancestor is not None:
        if ancestor is input_page:
            found = True
            break
        ancestor = ancestor.parent()
    assert found, "the Check segmented image button is not on the Input tab"


# --- guards ------------------------------------------------------------------


def test_check_segmented_image_refuses_without_an_input_path(panel, monkeypatch):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_segmentation_quality_check_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    panel._haemolynx_check_segmented_image()
    assert not started
    assert "Choose a segmented input image" in panel._haemolynx_report()


def test_check_segmented_image_refuses_a_missing_file(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_segmentation_quality_check_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    rows = panel._haemolynx_rows()
    rows["input_path"].value = tmp_path / "does_not_exist.tif"
    panel._haemolynx_check_segmented_image()
    assert not started
    assert "Choose a segmented input image" in panel._haemolynx_report()


def test_check_segmented_image_refuses_while_a_run_is_already_going(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_segmentation_quality_check_in_background",
        lambda *a, **k: started.append((a, k)),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input

    panel._haemolynx_run_state.start(worker=None, cancel_flag={"cancelled": False})
    try:
        panel._haemolynx_check_segmented_image()
    finally:
        panel._haemolynx_run_state.stopped()
    assert not started
    assert "already going" in panel._haemolynx_report()


# --- wiring: the button starts a background worker with the right arguments --


def test_check_segmented_image_starts_the_background_worker(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_segmentation_quality_check_in_background",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input

    panel._haemolynx_check_segmented_image()

    assert len(started) == 1
    args, kwargs = started[0]
    settings, report, button = args
    assert Path(settings["input_path"]) == real_input
    assert button is panel._haemolynx_check_image_button
    assert kwargs["run_state"] is panel._haemolynx_run_state


def test_clicking_the_native_button_triggers_the_check(panel, monkeypatch, tmp_path):
    """The button's own Qt click, not just calling the handler directly."""
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_segmentation_quality_check_in_background",
        lambda *args, **kwargs: started.append(args),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_check_image_button.native.click()

    assert len(started) == 1


# --- a real, slow, end-to-end run --------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_check_segmented_image_end_to_end_on_a_real_fixture(panel, tmp_path):
    """Runs the real worker to completion: the report gets a 0-10 score.

    No mocking of `_run_segmentation_quality_check_in_background` here --
    the one test that proves the whole button-to-score path works, not just
    its wiring. Uses `napari.qt.threading`'s worker synchronously via Qt's
    own event loop, pumped until the report line changes, same technique as
    `test_gui_optimise_settings_widget.py`.
    """
    import shutil
    import time

    from qtpy.QtWidgets import QApplication

    assert FIXTURE.is_file(), FIXTURE
    copy_path = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, copy_path)

    rows = panel._haemolynx_rows()
    rows["input_path"].value = copy_path

    panel._haemolynx_check_segmented_image()
    assert "Checking segmented image..." in panel._haemolynx_report()

    app = QApplication.instance()
    deadline = time.time() + 60
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
        if "Checking segmented image..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("segmented image check did not finish within 60s")

    report = panel._haemolynx_report()
    assert "Segmented image quality:" in report, report
    assert "/10" in report
