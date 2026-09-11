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
    assert kwargs["downsample_factor"] is None  # "Auto" is the default dropdown value
    assert kwargs["groups"] is None  # "Choose optimisation types" is unticked by default


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


# --- optimisation downsampling ------------------------------------------------


def test_downsample_dropdown_exists_with_the_right_choices_and_default(panel):
    dropdown = panel._haemolynx_optimise_downsample
    assert list(dropdown.choices) == ["Auto", "Off", "2x", "4x", "8x", "16x"]
    assert dropdown.value == "Auto"
    assert dropdown.tooltip.strip()
    assert "only" in dropdown.tooltip and "Optimise settings" in dropdown.tooltip


def test_downsample_dropdown_choice_is_passed_through(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_optimise_downsample.value = "4x"
    panel._haemolynx_optimise_settings()

    assert started[0]["downsample_factor"] == 4


def test_downsample_dropdown_off_maps_to_factor_1(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_optimise_downsample.value = "Off"
    panel._haemolynx_optimise_settings()

    assert started[0]["downsample_factor"] == 1


# --- choose optimisation types ------------------------------------------------


def test_choose_groups_checkbox_and_list_exist_and_start_hidden(panel):
    from haemolynx.optimisation import GROUP_LABELS, GROUP_NAMES

    checkbox = panel._haemolynx_optimise_choose_groups
    assert checkbox.value is False
    assert checkbox.tooltip.strip()

    group_checkboxes = panel._haemolynx_optimise_group_checkboxes
    assert set(group_checkboxes) == set(GROUP_NAMES)
    for name, box in group_checkboxes.items():
        assert box.text == GROUP_LABELS[name]
        assert box.value is True  # every group runs unless the user narrows it down


def test_ticking_choose_groups_reveals_the_checkbox_list(panel):
    """A nested Container's own ``.visible`` only reflects reality once its
    top-level ancestor is actually shown -- true in the real app (the panel is
    docked and displayed), not true of a bare, never-shown widget in a test,
    so this test shows the panel itself first."""
    panel.show()

    container = panel._haemolynx_optimise_group_checkboxes_container
    assert container.visible is False

    panel._haemolynx_optimise_choose_groups.value = True
    assert container.visible is True

    panel._haemolynx_optimise_choose_groups.value = False
    assert container.visible is False


def test_groups_kwarg_is_none_when_choose_groups_is_unticked(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    # Untick one group's box without ticking "Choose optimisation types" --
    # it must have no effect, since the list is not in play yet.
    panel._haemolynx_optimise_group_checkboxes["min_stub_length"].value = False
    panel._haemolynx_optimise_settings()

    assert started[0]["groups"] is None


def test_groups_kwarg_reflects_only_the_ticked_boxes_when_chosen(panel, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(
        widget_mod,
        "_run_optimisation_in_background",
        lambda *args, **kwargs: started.append(kwargs),
    )
    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_optimise_choose_groups.value = True
    group_checkboxes = panel._haemolynx_optimise_group_checkboxes
    for name, box in group_checkboxes.items():
        box.value = name in ("closing_radius", "min_stub_length")

    panel._haemolynx_optimise_settings()

    assert set(started[0]["groups"]) == {"closing_radius", "min_stub_length"}


# --- the generated config's schema must always be buildable ------------------


def test_names_with_prerequisite_closure_pulls_in_the_whole_requires_chain():
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    names = widget_mod._names_with_prerequisite_closure(
        schema, ("use_thick_vessel_skeletonisation", "input_path")
    )

    assert "use_thick_vessel_skeletonisation" in names
    assert "input_path" in names
    # use_thick_vessel_skeletonisation requires do_skeletonize;
    # input_path requires use_ilastik_segmentation -- both one-hop
    # prerequisites, and the closure must include every hop, not just one.
    assert "do_skeletonize" in names
    assert "use_ilastik_segmentation" in names
    # The whole point: a Schema built from just this closure must not raise
    # ConfigError for a missing requires target.
    schema.subset(names)


def test_optimise_settings_generated_config_schema_is_buildable():
    """Regression: every OPTIMISE_SETTING_NAMES entry's own requires chain
    must resolve inside the documented config schema the "finished" handler
    builds, or writing the optimised-settings YAML crashes after a real
    run completes (this crashed in production before
    `_names_with_prerequisite_closure` existed)."""
    from haemolynx.optimisation import OPTIMISE_SETTING_NAMES
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    names = widget_mod._names_with_prerequisite_closure(
        schema, (*OPTIMISE_SETTING_NAMES, "input_path")
    )

    documented_schema = schema.subset(names)  # must not raise ConfigError

    assert set(OPTIMISE_SETTING_NAMES) <= set(documented_schema.names)
    assert "input_path" in documented_schema.names


def test_optimise_settings_binarizes_a_normalized_float_probability_mask(
    panel, monkeypatch, tmp_path
):
    """Regression test for the same mis-binarization bug fixed in
    `_run_segmentation_quality_check_in_background` (see
    test_gui_check_segmented_image_widget.py) --
    `_run_optimisation_in_background` had the identical
    ``np.asarray(image).astype(bool)`` bug on the raw loaded image, before
    the optimiser ever saw it. Captures the ``raw_mask`` actually passed to
    :func:`optimise_skeleton_and_graph_settings` rather than running the
    full (slow) real search, to isolate just this fix.
    """
    import time

    import numpy as np
    from qtpy.QtWidgets import QApplication

    from haemolynx.optimisation.search import OptimisationResult

    rng = np.random.default_rng(0)
    # Background noise well under the 0.5 threshold; a real 6x6x6 block well
    # over it. Naive `!= 0` reads every noisy background voxel as foreground;
    # the real threshold-at-0.5 binarisation reads only the block.
    image = rng.uniform(0.0, 0.2, size=(12, 12, 12)).astype(np.float32)
    image[3:9, 3:9, 3:9] = 0.9

    monkeypatch.setattr(
        widget_mod,
        "load_volume_for_skeletonise",
        lambda settings, input_format: (image, (1.0, 1.0, 1.0), {"status": "complete"}),
    )

    captured = {}

    def fake_optimise(raw_mask, **kwargs):
        captured["raw_mask"] = raw_mask
        return OptimisationResult(settings={}, trials=())

    monkeypatch.setattr(widget_mod, "optimise_skeleton_and_graph_settings", fake_optimise)

    real_input = tmp_path / "mask.tif"
    real_input.write_bytes(b"")
    panel._haemolynx_rows()["input_path"].value = real_input

    panel._haemolynx_optimise_settings()

    app = QApplication.instance()
    deadline = time.time() + 10
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
        if "raw_mask" in captured:
            break
    else:
        pytest.fail("optimiser worker did not run within 10s")

    assert captured["raw_mask"].sum() == 6 ** 3  # only the thresholded block


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
