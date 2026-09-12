"""The "Check segmented image" button, built for real, with a live QApplication.

Uses `qapp` (pytest-qt) rather than the `panel`/`make_napari_viewer` fixture the
rest of `test_gui_widget.py` uses -- this feature never touches a napari layer
or the viewer, so `settings_widget(napari_viewer=None)` is enough (same
precedent as `test_gui_optimise_settings_widget.py`, which this file mirrors).

They need a Qt binding and pytest-qt, so they are marked `gui` and skipped
everywhere those are missing, same as `test_gui_widget.py`.
"""
from __future__ import annotations

import re
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


# --- raw data file row: exists, placed correctly, two-way synced -------------------


def test_raw_data_row_exists_with_a_tooltip(panel):
    row = panel._haemolynx_raw_data_row
    assert row.tooltip.strip()


def test_raw_data_row_is_a_child_of_the_input_tab(panel):
    from haemolynx.gui.tabs import tab_titles

    tabs = panel._haemolynx_tabs
    input_index = list(tab_titles()).index("1. Input")
    input_page = tabs.widget(input_index)
    ancestor = panel._haemolynx_raw_data_row.native.parent()
    found = False
    while ancestor is not None:
        if ancestor is input_page:
            found = True
            break
        ancestor = ancestor.parent()
    assert found, "the raw data file row is not on the Input tab"


def test_raw_data_row_starts_in_sync_with_fwhm_raw_tiff_path(panel):
    rows = panel._haemolynx_rows()
    assert panel._haemolynx_raw_data_row.value == rows["fwhm_raw_tiff_path"].value


def test_setting_raw_data_row_updates_fwhm_raw_tiff_path(panel, tmp_path):
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")

    panel._haemolynx_raw_data_row.value = raw_file

    assert Path(rows["fwhm_raw_tiff_path"].value) == raw_file


def test_setting_fwhm_raw_tiff_path_updates_the_raw_data_row(panel, tmp_path):
    rows = panel._haemolynx_rows()
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")

    rows["fwhm_raw_tiff_path"].value = raw_file

    assert Path(panel._haemolynx_raw_data_row.value) == raw_file


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


def test_check_segmented_image_binarizes_a_normalized_float_probability_mask(
    panel, monkeypatch, tmp_path
):
    """Regression test: the check used to score
    ``np.asarray(image).astype(bool)`` -- equivalent to ``image != 0`` --
    directly on the *raw* loaded image, instead of running it through the
    canonical `_to_binary_volume_for_skeletonization` threshold the real
    `skeletonise()` stage always applies. A normalized [0, 1] probability
    mask with nonzero background noise then reads as 100% foreground in one
    giant "component", exactly the symptom a user reported (score 8.9/10,
    "largest component holds 100% of the volume", vessel radius in the
    hundreds of microns) for a genuinely well-segmented binary mask.
    """
    import time

    import numpy as np
    from qtpy.QtWidgets import QApplication

    rng = np.random.default_rng(0)
    # Background noise well under the 0.5 threshold; a real ~12.5%-occupied
    # vessel-like block well over it. Naive `!= 0` reads every noisy
    # background voxel as foreground; the real threshold-at-0.5 binarisation
    # reads only the block.
    image = rng.uniform(0.0, 0.2, size=(20, 20, 20)).astype(np.float32)
    image[5:15, 5:15, 5:15] = 0.9

    monkeypatch.setattr(
        widget_mod,
        "load_volume_for_skeletonise",
        lambda settings, input_format: (image, (1.0, 1.0, 1.0), {"status": "complete"}),
    )

    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_check_segmented_image()
    assert "Checking segmented image..." in panel._haemolynx_report()

    app = QApplication.instance()
    deadline = time.time() + 10
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if "Checking segmented image..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("segmented image check did not finish within 10s")

    report = panel._haemolynx_report()
    assert "Segmented image quality:" in report, report
    # The buggy naive `!= 0` binarisation reads all the background noise as
    # foreground too, which (a) reaches every face of the volume -- several
    # boundary-touching patches instead of none -- and (b) inflates the
    # measured vessel radius by an order of magnitude (the whole 20x20x20
    # volume's own inscribed radius, not the 10x10x10 thresholded block's).
    boundary_match = re.search(r"\((\d+) place\(s\) the mask touches the image edge\)", report)
    assert boundary_match, report
    assert int(boundary_match.group(1)) == 0, report
    radius_match = re.search(r"typical vessel radius ([\d.]+)um", report)
    assert radius_match, report
    assert float(radius_match.group(1)) < 5.0, report


def test_check_segmented_image_reports_raw_vs_after_cleanup_when_cleanup_is_on(
    panel, monkeypatch, tmp_path
):
    """When the run's segmentation_cleanup_* settings are on, the report
    must show both the raw score and what those settings achieve on top of
    it, with the caveat that the second number is not evidence the source
    data/imaging itself was adequate (see segmentation_quality's own module
    docstring for why conflating the two defeats the point of this check)."""
    import time

    import numpy as np
    from qtpy.QtWidgets import QApplication

    shape = (30, 30, 30)
    zz, yy, xx = np.indices(shape, dtype=float)
    body = (np.sqrt((zz - 15) ** 2 + (yy - 15) ** 2) <= 5.0) & (xx >= 5) & (xx <= 24)
    mask = body.copy()
    mask[1, 1, 1] = True  # an isolated single-voxel speck, far from the body

    monkeypatch.setattr(
        widget_mod,
        "load_volume_for_skeletonise",
        lambda settings, input_format: (mask, (1.0, 1.0, 1.0), {"status": "complete"}),
    )

    rows = panel._haemolynx_rows()
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    rows["input_path"].value = real_input
    rows["segmentation_cleanup_remove_small_volumes"].value = True

    panel._haemolynx_check_segmented_image()

    app = QApplication.instance()
    deadline = time.time() + 10
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if "Checking segmented image..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("segmented image check did not finish within 10s")

    report = panel._haemolynx_report()
    assert "Raw segmentation:" in report, report
    assert "After your current cleanup settings:" in report, report
    assert "does not mean your source data improved" in report, report


def test_check_segmented_image_cross_checks_against_a_raw_file_when_set(
    panel, monkeypatch, tmp_path
):
    """When the "Raw data file" row (shared with fwhm_raw_tiff_path) names a
    file, the report gains an added/removed/missed-structures section
    comparing the segmentation against it."""
    import time

    import numpy as np
    from qtpy.QtWidgets import QApplication

    import haemolynx.haemodynamics.automated as automated_mod

    shape = (20, 20, 20)
    zz, yy, xx = np.indices(shape, dtype=float)
    mask = (np.sqrt((zz - 10) ** 2 + (yy - 10) ** 2) <= 4.0) & (xx >= 2) & (xx <= 17)
    raw_image = np.where(mask, 200.0, 10.0).astype(np.float32)

    monkeypatch.setattr(
        widget_mod,
        "load_volume_for_skeletonise",
        lambda settings, input_format: (mask, (1.0, 1.0, 1.0), {"status": "complete"}),
    )
    monkeypatch.setattr(
        automated_mod,
        "load_single_channel_tiff_volume",
        lambda path, axis_order: raw_image,
    )

    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    raw_file = tmp_path / "raw.tif"
    raw_file.write_bytes(b"")

    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input
    panel._haemolynx_raw_data_row.value = raw_file

    panel._haemolynx_check_segmented_image()

    app = QApplication.instance()
    deadline = time.time() + 10
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if "Checking segmented image..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("segmented image check did not finish within 10s")

    report = panel._haemolynx_report()
    assert "Raw-image cross-check" in report, report
    assert "agreement" in report.lower(), report
    assert "added voxels" in report, report
    assert "removed voxels" in report, report
    assert "missed structures" in report, report


def test_check_segmented_image_degrades_gracefully_when_raw_file_is_missing(
    panel, monkeypatch, tmp_path
):
    """A raw path that fails to load must not crash the whole check -- the
    mask-only score is still useful; the raw cross-check just gets skipped
    with a note explaining why."""
    import time

    import numpy as np
    from qtpy.QtWidgets import QApplication

    shape = (15, 15, 15)
    zz, yy, xx = np.indices(shape, dtype=float)
    mask = (np.sqrt((zz - 7) ** 2 + (yy - 7) ** 2) <= 3.0) & (xx >= 2) & (xx <= 12)

    monkeypatch.setattr(
        widget_mod,
        "load_volume_for_skeletonise",
        lambda settings, input_format: (mask, (1.0, 1.0, 1.0), {"status": "complete"}),
    )

    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")

    rows = panel._haemolynx_rows()
    rows["input_path"].value = real_input
    panel._haemolynx_raw_data_row.value = tmp_path / "does_not_exist.tif"

    panel._haemolynx_check_segmented_image()

    app = QApplication.instance()
    deadline = time.time() + 10
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if "Checking segmented image..." not in panel._haemolynx_report():
            break
    else:
        pytest.fail("segmented image check did not finish within 10s")

    report = panel._haemolynx_report()
    assert "Segmented image quality:" in report, report
    assert "Raw-image cross-check skipped:" in report, report
