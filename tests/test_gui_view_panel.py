"""Floating view dock: Z-depth, scale bar, and snapshot, display-only.

The pipeline still reads the full volume / full graph. These controls only
change what napari shows. The snapshot is a cosmetic canvas TIFF. The dock
floats over the canvas so it does not steal height from the bottom run log.
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    SCALE_BAR_POSITION,
    SNAPSHOT_NO_OUTPUT_FOLDER,
    SNAPSHOT_STEM,
    VIEW_DOCK_NAME,
    _apply_layers,
    _float_dock_over_canvas,
    _scale_bar_overlay,
    _store_z_window_cache,
    data_for_pipeline,
    output_folder_from_settings,
    settings_widget,
    unique_snapshot_path,
)
from haemolynx.gui.chrome_tooltips import REOPEN_VIEW_TOOLTIP  # noqa: E402
from haemolynx.gui.results import IMAGE, NODES, VESSEL_TUBES, VESSELS  # noqa: E402
from test_gui_results_widget import a_run  # noqa: E402

pytestmark = pytest.mark.gui


def _stack_results():
    from types import SimpleNamespace

    from haemolynx.gui.results import ResultLayers

    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4), dtype=np.uint8),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=(0.5, 1.0, 2.0),
            voxel_size_zyx=(2.0, 1.0, 0.5),
        ),
    )
    return results


def _load_patterned_image(viewer):
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[0] = 1
    image[1] = 3
    image[2] = 2
    image[3] = 9
    layer = viewer.layers[IMAGE]
    layer.data = image
    _store_z_window_cache(layer, image)
    return layer, image


def test_selecting_the_image_layer_gives_it_a_clearer_3d_render(make_napari_viewer):
    """Plain MIP -- napari's own default -- has no depth cue; the layer being
    inspected should switch to attenuated MIP while it is the active
    selection, and fall back once something else is selected."""
    from haemolynx.gui._widget import (
        IMAGE_DEFAULT_RENDER_OPTIONS,
        IMAGE_FOCUS_RENDER_OPTIONS,
    )

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    image_layer = viewer.layers[IMAGE]
    for key, value in IMAGE_DEFAULT_RENDER_OPTIONS.items():
        assert getattr(image_layer, key) == value

    viewer.layers.selection.active = image_layer
    for key, value in IMAGE_FOCUS_RENDER_OPTIONS.items():
        assert getattr(image_layer, key) == value

    other = next(l for l in viewer.layers if l is not image_layer)
    viewer.layers.selection.active = other
    for key, value in IMAGE_DEFAULT_RENDER_OPTIONS.items():
        assert getattr(image_layer, key) == value


def test_the_view_panel_floats_over_the_canvas(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)

    dock = panel._haemolynx_view_dock
    assert dock is not None
    assert VIEW_DOCK_NAME in dock.windowTitle()
    assert dock.isFloating()
    assert panel._haemolynx_view_panel.objectName() == "haemolynx_view_panel"
    assert panel._haemolynx_z_depth_slider.objectName() == "haemolynx_z_depth_slider"
    assert panel._haemolynx_z_depth_row.parentWidget() is panel._haemolynx_display_group
    assert panel._haemolynx_vessel_draw.objectName() == "haemolynx_vessel_draw"
    assert panel._haemolynx_vessel_draw_row.objectName() == "haemolynx_vessel_draw_row"
    assert panel._haemolynx_vessel_draw_row.parentWidget() is panel._haemolynx_display_group
    assert panel._haemolynx_vessel_draw.currentText() == "Tubes"
    assert panel._haemolynx_scale_bar.objectName() == "haemolynx_scale_bar"
    assert panel._haemolynx_display_group.objectName() == "haemolynx_display_group"
    snapshot = panel._haemolynx_snapshot_group
    assert snapshot.objectName() == "haemolynx_snapshot_group"
    assert snapshot.parentWidget() is panel._haemolynx_view_panel
    layout = panel._haemolynx_view_panel.layout()
    assert layout.indexOf(panel._haemolynx_display_group) >= 0
    assert layout.indexOf(snapshot) == layout.indexOf(panel._haemolynx_display_group) + 1
    button = panel._haemolynx_snapshot_button
    assert button.objectName() == "haemolynx_snapshot_button"
    assert button.parentWidget() is snapshot
    assert snapshot.layout().indexOf(button) >= 0


def test_the_floating_view_panel_is_tall_enough_that_snapshot_is_not_clipped(
    make_napari_viewer,
):
    """Regression: a fresh widget's sizeHint() used to under-report, so
    "Save snapshot" sat partly below the floating dock's own bottom edge."""
    from qtpy.QtWidgets import QApplication

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    dock = panel._haemolynx_view_dock
    QApplication.processEvents()

    assert dock.height() >= 220

    button = panel._haemolynx_snapshot_button
    bottom_in_dock = button.mapTo(dock, button.rect().bottomLeft()).y()
    assert bottom_in_dock <= dock.height(), (
        "Save snapshot's own bottom edge falls outside the dock "
        f"(button bottom={bottom_in_dock}, dock height={dock.height()})"
    )


def test_float_dock_over_canvas_pads_past_the_size_hint():
    """The resize math itself, isolated from a real dock's own layout pass."""
    from types import SimpleNamespace

    class FakeSize:
        def __init__(self, width, height):
            self._width, self._height = width, height

        def width(self):
            return self._width

        def height(self):
            return self._height

    class FakeInner:
        def sizeHint(self):
            return FakeSize(300, 150)

    class FakeDock:
        def __init__(self):
            self.floated = False
            self.resized = None

        def setFloating(self, value):
            self.floated = value

        def widget(self):
            return FakeInner()

        def resize(self, width, height):
            self.resized = (width, height)

    dock = FakeDock()
    viewer = SimpleNamespace(window=SimpleNamespace())
    _float_dock_over_canvas(viewer, dock)

    assert dock.floated is True
    width, height = dock.resized
    assert width == 300
    assert height >= 150 + 24, "should pad past the hint, not just floor it"


def test_the_view_button_sits_left_of_edit_and_has_a_tooltip(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    view_button = panel._haemolynx_view_button
    edit_button = panel._haemolynx_edit_button

    assert view_button.text == "View"
    assert view_button.tooltip == REOPEN_VIEW_TOOLTIP
    assert view_button.enabled

    row = panel._haemolynx_run_file_row
    layout = row.layout()
    assert layout.indexOf(view_button.native) < layout.indexOf(edit_button.native)


def test_the_view_button_is_disabled_with_no_viewer():
    panel = settings_widget(napari_viewer=None)
    assert panel._haemolynx_view_dock is None
    assert not panel._haemolynx_view_button.enabled


def test_reopening_an_already_open_view_panel_is_a_no_op(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    dock = panel._haemolynx_view_dock
    assert dock.isVisible()

    panel._haemolynx_reopen_view()

    assert dock.isVisible()


def test_the_view_button_reopens_a_closed_view_panel(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    dock = panel._haemolynx_view_dock
    dock.hide()
    assert not dock.isVisible()

    panel._haemolynx_view_button.native.click()

    assert dock.isVisible()


def test_the_view_panel_is_not_on_the_right_settings_column(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for widget in (
        panel._haemolynx_z_depth_slider,
        panel._haemolynx_z_depth_row,
        panel._haemolynx_vessel_draw,
        panel._haemolynx_vessel_draw_row,
    ):
        ancestor = widget.parentWidget()
        while ancestor is not None:
            assert ancestor is not panel
            ancestor = ancestor.parentWidget()


def test_the_panel_still_builds_the_view_chrome_with_no_viewer():
    panel = settings_widget(napari_viewer=None)
    assert panel._haemolynx_view_dock is None
    assert panel._haemolynx_view_panel is not None
    assert panel._haemolynx_z_depth_slider is not None
    assert panel._haemolynx_scale_bar.isChecked() is False
    assert panel._haemolynx_scale_bar.isEnabled()
    assert panel._haemolynx_snapshot_button is not None
    assert panel._haemolynx_vessel_draw.currentText() == "Tubes"
    panel._haemolynx_snapshot_button.click()
    assert "viewer" in panel._haemolynx_report().lower()


def test_z_depth_filter_is_on_the_left_panel_not_the_right(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    row = panel._haemolynx_z_depth_row
    assert row.parentWidget() is panel._haemolynx_display_group
    ancestor = row
    found_view = False
    while ancestor is not None:
        assert ancestor is not panel
        if ancestor is panel._haemolynx_view_panel:
            found_view = True
        ancestor = ancestor.parentWidget()
    assert found_view
    assert panel.layout().indexOf(row) == -1


def test_z_depth_clips_every_layer_and_full_range_restores(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    layer, original = _load_patterned_image(viewer)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()

    vessel_count = len(viewer.layers[VESSELS].data)
    node_count = len(viewer.layers[NODES].data)

    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    displayed = np.asarray(layer.data)
    np.testing.assert_array_equal(displayed[0], 1)
    np.testing.assert_array_equal(displayed[1], 3)
    np.testing.assert_array_equal(displayed[2], 2)
    np.testing.assert_array_equal(displayed[3], 0)
    assert len(viewer.layers[VESSELS].data) < vessel_count
    assert len(viewer.layers[NODES].data) <= node_count
    np.testing.assert_array_equal(data_for_pipeline(layer), original)

    panel._haemolynx_z_depth_slider.setValue((0.0, 8.0))
    np.testing.assert_array_equal(np.asarray(layer.data), original)
    assert len(viewer.layers[VESSELS].data) == vessel_count
    assert len(viewer.layers[NODES].data) == node_count


def test_z_depth_slider_does_not_rebuild_while_dragging(
    make_napari_viewer, monkeypatch
):
    """valueChanged while a handle is down must not clip volumes or graph layers.

    The expensive work is a full-volume zeros+copy plus napari Vectors/Points
    rebuilds. Apply once on sliderReleased (and immediately on programmatic
    setValue when not dragging).
    """
    import haemolynx.gui._widget as widget_mod

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    layer, original = _load_patterned_image(viewer)
    panel._haemolynx_view.results = _stack_results()

    counts = {"volume": 0, "graph": 0}
    real_volume = widget_mod._apply_volume_z_display
    real_graph = widget_mod._apply_z_filter

    def counting_volume(*args, **kwargs):
        counts["volume"] += 1
        return real_volume(*args, **kwargs)

    def counting_graph(*args, **kwargs):
        counts["graph"] += 1
        return real_graph(*args, **kwargs)

    monkeypatch.setattr(widget_mod, "_apply_volume_z_display", counting_volume)
    monkeypatch.setattr(widget_mod, "_apply_z_filter", counting_graph)

    panel._haemolynx_after_layers_applied()
    baseline_volume = counts["volume"]
    baseline_graph = counts["graph"]
    assert baseline_volume >= 1
    assert baseline_graph >= 1

    slider = panel._haemolynx_z_depth_slider
    slider._hi.sliderPressed.emit()
    assert slider.isSliderDown()
    for hi in (7.5, 7.0, 6.5, 6.0, 5.5, 5.0):
        slider.setValue((0.0, hi))
    assert counts["volume"] == baseline_volume
    assert counts["graph"] == baseline_graph
    np.testing.assert_array_equal(np.asarray(layer.data), original)

    slider._hi.sliderReleased.emit()
    assert counts["volume"] == baseline_volume + 1
    assert counts["graph"] == baseline_graph + 1
    displayed = np.asarray(layer.data)
    np.testing.assert_array_equal(displayed[0], 1)
    np.testing.assert_array_equal(displayed[1], 3)
    np.testing.assert_array_equal(displayed[2], 2)
    np.testing.assert_array_equal(displayed[3], 0)
    np.testing.assert_array_equal(data_for_pipeline(layer), original)

    slider.setValue((0.0, 8.0))
    assert counts["volume"] == baseline_volume + 2
    assert counts["graph"] == baseline_graph + 2
    np.testing.assert_array_equal(np.asarray(layer.data), original)


def test_z_depth_leaves_the_pipeline_cache_full(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    layer, original = _load_patterned_image(viewer)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()
    settings_before = dict(panel._haemolynx_values())

    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    np.testing.assert_array_equal(data_for_pipeline(layer), original)
    assert dict(panel._haemolynx_values()) == settings_before


def _vispy_scale_bar_visuals(viewer):
    """Vispy overlays napari attaches to the scale-bar model, if the window exists."""
    overlay = _scale_bar_overlay(viewer)
    if overlay is None:
        return []
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window is not None else None
    mapping = getattr(
        getattr(qt_viewer, "canvas", None), "_viewer_overlay_to_visual", None
    )
    if mapping is None:
        return []
    return list(mapping.get(overlay, []))


def test_scale_bar_checkbox_toggles_viewer_scale_bar(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_scale_bar
    overlay = _scale_bar_overlay(viewer)
    assert box.isChecked() is False
    assert overlay is not None
    assert overlay.visible is False
    assert str(overlay.position) == SCALE_BAR_POSITION

    box.setChecked(True)
    assert overlay.visible is True
    assert str(overlay.position) == SCALE_BAR_POSITION
    visuals = _vispy_scale_bar_visuals(viewer)
    assert visuals, "napari should add a canvas scale-bar overlay when the box is ticked"
    assert visuals[0].node.visible is True

    box.setChecked(False)
    assert overlay.visible is False
    visuals = _vispy_scale_bar_visuals(viewer)
    if visuals:
        assert visuals[0].node.visible is False


def test_scale_bar_is_bottom_right_and_in_microns_once_layers_exist(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    overlay = _scale_bar_overlay(viewer)
    assert overlay.visible is False
    panel._haemolynx_scale_bar.setChecked(True)
    assert overlay.visible is True
    assert str(overlay.position) == SCALE_BAR_POSITION
    visuals = _vispy_scale_bar_visuals(viewer)
    assert visuals
    assert visuals[0].node.visible is True

    image = viewer.layers[IMAGE]
    assert all("micrometer" in str(unit) for unit in image.units)

    panel._haemolynx_scale_bar.setChecked(False)
    assert overlay.visible is False
    visuals = _vispy_scale_bar_visuals(viewer)
    if visuals:
        assert visuals[0].node.visible is False


def test_output_folder_is_the_vtk_prefix_parent(tmp_path):
    from pathlib import Path

    assert output_folder_from_settings({"vtk_output_prefix": Path("out/run")}) == Path(
        "out"
    )
    assert output_folder_from_settings({"vtk_output_prefix": None}) is None
    assert output_folder_from_settings({"vtk_output_prefix": ""}) is None
    assert output_folder_from_settings({"vtk_output_prefix": "."}) is None
    assert output_folder_from_settings({}) is None
    # A FileEdit left blank resolves to the working directory.
    assert output_folder_from_settings({"vtk_output_prefix": Path.cwd()}) is None
    chosen = tmp_path / "artifacts" / "run"
    assert output_folder_from_settings({"vtk_output_prefix": chosen}) == tmp_path / "artifacts"


def test_snapshot_filename_is_timestamped_and_unique(tmp_path):
    from datetime import datetime

    when = datetime(2026, 9, 4, 11, 32, 0)
    first = unique_snapshot_path(tmp_path, when=when)
    assert first.name == f"{SNAPSHOT_STEM}_20260904_113200.tif"
    first.write_bytes(b"x")
    second = unique_snapshot_path(tmp_path, when=when)
    assert second.name == f"{SNAPSHOT_STEM}_20260904_113200_2.tif"


def _point_snapshot_at(panel, tmp_path):
    rows = panel._haemolynx_rows()
    rows["vtk_output_prefix"].value = tmp_path / "artifacts" / "run"
    return tmp_path / "artifacts"


def _fake_view_rgb(*, mark=(3, 4, (7, 8, 9)), shape=(12, 16, 3)):
    rgb = np.zeros(shape, dtype=np.uint8)
    row, col, colour = mark
    rgb[row, col] = colour
    return rgb


def test_snapshot_button_writes_a_single_view_tiff(
    make_napari_viewer, tmp_path, monkeypatch
):
    import tifffile
    from qtpy.QtWidgets import QPushButton

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    out_dir = _point_snapshot_at(panel, tmp_path)
    rgb = _fake_view_rgb()
    monkeypatch.setattr(viewer.window, "screenshot", lambda **_k: rgb)

    settings_before = dict(panel._haemolynx_values())
    image_layer = viewer.layers[IMAGE]
    original = np.asarray(data_for_pipeline(image_layer)).copy()

    button = panel._haemolynx_snapshot_button
    assert isinstance(button, QPushButton)
    button.click()

    written = sorted(out_dir.glob(f"{SNAPSHOT_STEM}_*.tif"))
    assert len(written) == 1
    path = written[0]
    assert str(path) in panel._haemolynx_report()

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 1
        saved = tif.pages[0].asarray()
    np.testing.assert_array_equal(saved, rgb)
    assert saved.ndim == 3 and saved.shape[-1] == 3
    assert saved.shape != original.shape
    np.testing.assert_array_equal(data_for_pipeline(image_layer), original)
    assert dict(panel._haemolynx_values()) == settings_before


def test_snapshot_is_the_displayed_view_not_the_volume_cache(
    make_napari_viewer, tmp_path, monkeypatch
):
    """The Z-depth filter changes the canvas; the TIFF is that view, not z_window_full."""
    from types import SimpleNamespace

    import tifffile

    from haemolynx.gui.results import ResultLayers

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[0] = 1
    image[1] = 3
    image[2] = 2
    image[3] = 9
    layer = viewer.layers[IMAGE]
    layer.data = image
    _store_z_window_cache(layer, image)

    results = ResultLayers()
    results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4), dtype=np.uint8),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=(0.5, 1.0, 2.0),
            voxel_size_zyx=(2.0, 1.0, 0.5),
        ),
    )
    panel._haemolynx_view.results = results
    panel._haemolynx_after_layers_applied()

    original = image.copy()
    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    displayed = np.asarray(layer.data)
    np.testing.assert_array_equal(displayed[0], 1)
    np.testing.assert_array_equal(displayed[3], 0)

    out_dir = _point_snapshot_at(panel, tmp_path)
    rgb = _fake_view_rgb(mark=(1, 2, (11, 22, 33)), shape=(8, 10, 3))
    seen = {}

    def fake_screenshot(**_kwargs):
        seen["displayed"] = np.asarray(layer.data).copy()
        return rgb

    monkeypatch.setattr(viewer.window, "screenshot", fake_screenshot)
    panel._haemolynx_snapshot_button.click()

    np.testing.assert_array_equal(seen["displayed"], displayed)
    np.testing.assert_array_equal(seen["displayed"][0], 1)
    np.testing.assert_array_equal(seen["displayed"][3], 0)

    written = sorted(out_dir.glob(f"{SNAPSHOT_STEM}_*.tif"))
    assert len(written) == 1
    with tifffile.TiffFile(written[0]) as tif:
        assert len(tif.pages) == 1
        saved = tif.pages[0].asarray()
    np.testing.assert_array_equal(saved, rgb)
    assert saved.shape != original.shape
    np.testing.assert_array_equal(data_for_pipeline(layer), original)
    np.testing.assert_array_equal(np.asarray(layer.data), displayed)


def test_snapshot_without_output_folder_does_not_crash(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    panel._haemolynx_rows()["vtk_output_prefix"].value = ""
    assert output_folder_from_settings(panel._haemolynx_values()) is None

    panel._haemolynx_snapshot_button.click()
    assert panel._haemolynx_report() == SNAPSHOT_NO_OUTPUT_FOLDER


def test_two_snapshots_do_not_overwrite(make_napari_viewer, tmp_path, monkeypatch):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    out_dir = _point_snapshot_at(panel, tmp_path)
    monkeypatch.setattr(
        viewer.window, "screenshot", lambda **_k: _fake_view_rgb()
    )
    panel._haemolynx_snapshot_button.click()
    panel._haemolynx_snapshot_button.click()
    written = sorted(out_dir.glob(f"{SNAPSHOT_STEM}_*.tif"))
    assert len(written) == 2
    assert written[0] != written[1]


def _choose_vessel_draw(control, text: str) -> None:
    """Pick Tubes/Lines the way a click on the view-panel radios does."""
    button = control.lines if text == "Lines" else control.tubes
    button.click()


def test_vessels_default_to_tubes_and_toggle_to_lines(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    assert isinstance(viewer.layers[VESSELS], napari.layers.Vectors)
    assert VESSEL_TUBES in viewer.layers
    assert isinstance(viewer.layers[VESSEL_TUBES], napari.layers.Surface)
    assert viewer.layers[VESSELS].visible is False
    assert viewer.layers[VESSEL_TUBES].visible is True
    assert panel._haemolynx_vessel_draw.currentText() == "Tubes"

    n_segments = len(viewer.layers[VESSELS].data)
    n_verts = len(viewer.layers[VESSEL_TUBES].data[0])
    assert n_verts == n_segments * 6 * 2

    panel._haemolynx_vessel_draw.setCurrentText("Lines")
    assert viewer.layers[VESSELS].visible is True
    assert viewer.layers[VESSEL_TUBES].visible is False

    panel._haemolynx_vessel_draw.setCurrentText("Tubes")
    assert viewer.layers[VESSELS].visible is False
    assert viewer.layers[VESSEL_TUBES].visible is True


def test_vessel_draw_activated_toggles_visibility_both_ways(make_napari_viewer):
    """A click on the Tubes/Lines radios must swap what is drawn."""
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    combo = panel._haemolynx_vessel_draw
    n_segments = len(viewer.layers[VESSELS].data)
    assert n_segments > 0
    assert len(viewer.layers[VESSEL_TUBES].data[0]) == n_segments * 12

    _choose_vessel_draw(combo, "Lines")
    assert combo.currentText() == "Lines"
    assert viewer.layers[VESSELS].visible is True
    assert viewer.layers[VESSEL_TUBES].visible is False
    assert len(viewer.layers[VESSEL_TUBES].data[0]) == 0

    _choose_vessel_draw(combo, "Tubes")
    assert combo.currentText() == "Tubes"
    assert viewer.layers[VESSELS].visible is False
    assert viewer.layers[VESSEL_TUBES].visible is True
    assert len(viewer.layers[VESSEL_TUBES].data[0]) == n_segments * 12

    _choose_vessel_draw(combo, "Lines")
    _choose_vessel_draw(combo, "Tubes")
    assert viewer.layers[VESSELS].visible is False
    assert viewer.layers[VESSEL_TUBES].visible is True
    assert len(viewer.layers[VESSEL_TUBES].data[0]) > 0


def test_vessel_draw_choice_survives_a_layer_rebuild(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    panel._haemolynx_vessel_draw.setCurrentText("Lines")
    for group in a_run():
        _apply_layers(viewer, group)

    assert panel._haemolynx_vessel_draw.currentText() == "Lines"
    assert viewer.layers[VESSELS].visible is True
    if VESSEL_TUBES in viewer.layers:
        assert viewer.layers[VESSEL_TUBES].visible is False


def test_z_depth_filter_rebuilds_tubes_from_filtered_vectors(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()

    assert panel._haemolynx_vessel_draw.currentText() == "Tubes"
    full_segments = len(viewer.layers[VESSELS].data)
    full_verts = len(viewer.layers[VESSEL_TUBES].data[0])
    assert full_verts == full_segments * 12

    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    clipped_segments = len(viewer.layers[VESSELS].data)
    assert clipped_segments < full_segments
    assert len(viewer.layers[VESSEL_TUBES].data[0]) == clipped_segments * 12
    assert viewer.layers[VESSEL_TUBES].visible is True
    assert viewer.layers[VESSELS].visible is False

    panel._haemolynx_z_depth_slider.setValue((0.0, 8.0))
    assert len(viewer.layers[VESSELS].data) == full_segments
    assert len(viewer.layers[VESSEL_TUBES].data[0]) == full_verts



# --- XY / XZ / YZ view-snap buttons ------------------------------------------


def _snap_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    # A box, not a cube, so every plane's centre differs.
    viewer.add_image(np.zeros((10, 20, 40), dtype=np.uint8), scale=(2.0, 1.0, 0.5))
    return viewer


def test_the_view_snap_buttons_sit_in_the_canvas_bottom_left(make_napari_viewer):
    from haemolynx.gui._widget import VIEW_SNAP_MARGIN

    viewer = _snap_viewer(make_napari_viewer)
    bar = viewer._haemolynx_view_snap_buttons
    assert list(bar.buttons) == ["XY", "XZ", "YZ"]
    for plane, button in bar.buttons.items():
        assert button.text() == plane
        assert button.toolTip()

    qt_viewer = viewer.window._qt_viewer
    native = qt_viewer.canvas.native
    for size in ((500, 400), (800, 650)):
        native.resize(*size)
        bar.place()
        corner = native.mapTo(bar.parentWidget(), native.rect().bottomLeft())
        assert bar.x() == corner.x() + VIEW_SNAP_MARGIN
        assert bar.geometry().bottom() == corner.y() - VIEW_SNAP_MARGIN


def test_the_view_snap_buttons_are_not_a_pane_of_the_viewer_splitter(
    make_napari_viewer,
):
    """napari's Qt viewer is a QSplitter, which turns every child into a pane:
    a bar parented there squeezed the canvas down to the buttons' width."""
    viewer = _snap_viewer(make_napari_viewer)
    bar = viewer._haemolynx_view_snap_buttons
    assert viewer.window._qt_viewer.indexOf(bar) == -1
    assert bar.parentWidget() is viewer.window._qt_window


def test_a_second_panel_reuses_the_viewers_view_snap_buttons(make_napari_viewer):
    viewer = _snap_viewer(make_napari_viewer)
    first = viewer._haemolynx_view_snap_buttons
    settings_widget(napari_viewer=viewer)
    assert viewer._haemolynx_view_snap_buttons is first


@pytest.mark.parametrize(
    ("plane", "view", "up"),
    [
        ("XY", (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ("XZ", (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        ("YZ", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
    ],
)
def test_a_3d_snap_turns_the_camera_and_recentres(make_napari_viewer, plane, view, up):
    viewer = _snap_viewer(make_napari_viewer)
    viewer.dims.ndisplay = 3
    viewer.reset_view()
    centre = tuple(viewer.camera.center)
    # Somewhere else entirely, as a user leaves it after orbiting and panning.
    viewer.camera.angles = (35.0, -20.0, 70.0)
    viewer.camera.center = (0.0, 0.0, 0.0)
    viewer.dims.order = (2, 1, 0)

    viewer._haemolynx_view_snap_buttons.buttons[plane].click()

    np.testing.assert_allclose(viewer.camera.view_direction, view, atol=1e-9)
    np.testing.assert_allclose(viewer.camera.up_direction, up, atol=1e-9)
    np.testing.assert_allclose(viewer.camera.center, centre)
    assert tuple(viewer.dims.order) == (0, 1, 2)


@pytest.mark.parametrize(
    ("plane", "displayed"), [("XY", (1, 2)), ("XZ", (0, 2)), ("YZ", (0, 1))]
)
def test_a_2d_snap_shows_the_planes_axes(make_napari_viewer, plane, displayed):
    from haemolynx.gui._widget import snap_view_to_plane

    viewer = _snap_viewer(make_napari_viewer)
    viewer.dims.ndisplay = 2
    viewer.camera.zoom = 50.0

    assert snap_view_to_plane(viewer, plane)

    assert tuple(viewer.dims.displayed) == displayed
    assert viewer.camera.zoom != 50.0  # refitted to the data


def test_tubes_stay_on_the_vessel_lines_after_a_2d_snap_then_3d(make_napari_viewer):
    """napari 0.9 slices a fully displayed Surface in its stored axis order,
    whatever ``dims.order`` says, while Vectors (and Image) follow the order.
    A 2D YZ snap then a switch to 3D left the tubes rotated against the vessel
    lines and the image they were built on -- z drawn along x."""
    from haemolynx.gui._widget import snap_view_to_plane

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    viewer.dims.ndisplay = 2
    assert snap_view_to_plane(viewer, "YZ")
    viewer.dims.ndisplay = 3
    displayed = list(viewer.dims.displayed)
    assert displayed != sorted(displayed)

    vessels = viewer.layers[VESSELS]
    tubes = viewer.layers[VESSEL_TUBES]
    # The lines' own on-screen coordinates: stored (z, y, x) in displayed order.
    np.testing.assert_array_equal(
        np.asarray(vessels._view_data)[:, 0], np.asarray(vessels.data)[:, 0][:, displayed]
    )
    np.testing.assert_array_equal(
        np.asarray(tubes._view_vertices), np.asarray(tubes.data[0])[:, displayed]
    )


def test_snapping_without_3d_data_changes_nothing(make_napari_viewer):
    from haemolynx.gui._widget import snap_view_to_plane

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((20, 40), dtype=np.uint8))
    order = tuple(viewer.dims.order)

    assert snap_view_to_plane(viewer, "XZ") is False
    assert tuple(viewer.dims.order) == order


# --- the Z-depth filter's cost ------------------------------------------------


def test_a_z_depth_change_does_not_recompose_branch_tooltips(make_napari_viewer, monkeypatch):
    """The tooltip text does not depend on the Z window: a filtered or
    recreated layer takes its column from the full cache, which is kept in
    step with the checkbox selection. Recomposing it anyway on every change
    is what made the slider take tens of seconds per move on a real network."""
    from haemolynx.gui import branch_hover

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()
    full_tooltips = list(viewer.layers[VESSELS].features["tooltip"])
    assert full_tooltips and all(t.startswith("branchID: ") for t in full_tooltips)

    calls = []
    real = branch_hover.tooltips_from_feature_table
    monkeypatch.setattr(
        branch_hover,
        "tooltips_from_feature_table",
        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs),
    )
    slider = panel._haemolynx_z_depth_slider
    slider.setValue((0.0, 5.0))
    narrowed = list(viewer.layers[VESSELS].features["tooltip"])
    slider.setValue((0.0, 8.0))

    assert calls == []
    assert narrowed and set(narrowed) <= set(full_tooltips)
    assert list(viewer.layers[VESSELS].features["tooltip"]) == full_tooltips


def test_the_z_depth_cache_is_the_volume_itself_not_a_copy(make_napari_viewer):
    """A copy doubled every displayed volume's memory for nothing: the filter
    never writes into it, since a clipped view is always a new array."""
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    layer, image = _load_patterned_image(viewer)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()

    assert data_for_pipeline(layer) is image
    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    assert layer.data is not image
    assert data_for_pipeline(layer) is image
    panel._haemolynx_z_depth_slider.setValue((0.0, 8.0))
    # Back to showing the stored volume itself, not a fresh copy of it.
    assert layer.data is image


def test_a_disk_backed_volume_stays_on_disk_through_the_z_filter(make_napari_viewer, tmp_path):
    """The low-RAM option keeps big volumes in memory-mapped files; the Z
    filter used to read each one into RAM in full just to cache it."""
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    layer, image = _load_patterned_image(viewer)
    on_disk = np.memmap(tmp_path / "stack.dat", dtype=image.dtype, mode="w+", shape=image.shape)
    on_disk[:] = image
    layer.data = on_disk
    _store_z_window_cache(layer, on_disk)
    panel._haemolynx_view.results = _stack_results()
    panel._haemolynx_after_layers_applied()

    assert data_for_pipeline(layer) is on_disk
    panel._haemolynx_z_depth_slider.setValue((0.0, 5.0))
    assert isinstance(layer.data, np.memmap)
    np.testing.assert_array_equal(np.asarray(layer.data)[3], 0)
    np.testing.assert_array_equal(np.asarray(layer.data)[:3], image[:3])
    panel._haemolynx_z_depth_slider.setValue((0.0, 8.0))
    assert layer.data is on_disk


# --- the Z-depth filter is display-only: it must not change what flows look like


def _flow_network_results(n_vessels=40):
    """A run whose vessels each carry their own flow and branch order."""
    from types import SimpleNamespace

    import networkx as nx

    from haemolynx.gui.results import ResultLayers
    from test_gui_results import network

    rng = np.random.default_rng(1)
    graph = nx.MultiGraph()
    for i in range(n_vessels + 1):
        graph.add_node(i, pos=np.array([i * 2.0, float(i % 5), float(i % 3)]))
    for i in range(n_vessels):
        a, b = graph.nodes[i]["pos"], graph.nodes[i + 1]["pos"]
        flow = float(10 ** rng.uniform(-13, -10))
        graph.add_edge(
            i, i + 1, key=0,
            voxels=[a.tolist(), ((a + b) / 2).tolist(), b.tolist()],
            length=float(np.linalg.norm(b - a)), segment_id=i,
            flow_abs=flow, flow_signed=flow, resistance=1e15, conductance=1e-15,
            # The first order only near z = 0, so a window higher up lacks it.
            branch_order="A1" if i < 5 else ("B2" if i % 2 else "V3"),
        )
    shape = (2 * n_vessels + 1, 6, 4)
    results = ResultLayers()
    groups = [
        results.stage_finished("skeletonise", SimpleNamespace(
            image=np.zeros(shape, dtype=np.uint8), skeleton=np.zeros(shape, dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0))),
        results.stage_finished("build_network", network(graph, (1.0, 1.0, 1.0))),
    ]
    return results, groups, float(shape[0])


def _segment_colours(layer):
    """Each drawn segment's colour, keyed by the segment itself."""
    data = np.asarray(layer.data)
    colours = np.round(np.asarray(layer.edge_color), 6)
    return {data[i].tobytes(): tuple(colours[i]) for i in range(len(data))}


@pytest.mark.parametrize("colour_by", ["flow_abs", "branch_order"])
def test_the_z_depth_filter_keeps_every_vessels_colour(make_napari_viewer, colour_by):
    """Narrowing the window used to paste the full layer's per-row colours
    onto the shorter layer, drawing each remaining vessel in some other
    vessel's flow colour -- and widening again never put them back."""
    from haemolynx.gui._widget import _colour_layer
    from haemolynx.gui.results import colour_cycle_for

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    results, groups, extent = _flow_network_results()
    for group in groups:
        _apply_layers(viewer, group)
    panel._haemolynx_view.results = results
    panel._haemolynx_after_layers_applied()
    layer = viewer.layers[VESSELS]
    if colour_by == "flow_abs":
        _colour_layer(layer, "flow_abs", "continuous")
    else:
        _colour_layer(layer, "branch_order", "categorical",
                      colour_cycle_for(layer.features["branch_order"]))
    full = _segment_colours(layer)
    limits = tuple(layer.edge_contrast_limits) if colour_by == "flow_abs" else None

    slider = panel._haemolynx_z_depth_slider
    slider.setValue((20.0, 60.0))
    narrowed = viewer.layers[VESSELS]
    shown = _segment_colours(narrowed)
    assert 0 < len(shown) < len(full)
    assert {key: full[key] for key in shown} == shown
    if colour_by == "flow_abs":
        assert tuple(narrowed.edge_contrast_limits) == limits
    else:
        assert "A1" not in set(narrowed.features["branch_order"])

    slider.setValue((0.0, extent))
    assert _segment_colours(viewer.layers[VESSELS]) == full


def test_a_sweep_layer_keeps_its_grid_point_through_a_z_depth_change(make_napari_viewer, tmp_path):
    """Two ways the depth filter used to change a sweep's flows on screen: it
    redrew from a cache still holding the first grid point, and the sweep
    slider kept driving the layer object the filter had just replaced."""
    from qtpy.QtWidgets import QAbstractSlider

    from haemolynx.gui._widget import OURS as RESULT_OURS
    from haemolynx.gui.results import ResultLayers, perturbation_layer_names
    from haemolynx.pipeline import PerturbationRun
    from test_perturbation_stage import PRESSURE_SWEEP, _run

    result = _run(tmp_path, [PRESSURE_SWEEP]).results[0]
    # That fixture's vessels run along x; turn them to run along z, so a depth
    # window has something to leave out.
    for _node, data in result.graph.nodes(data=True):
        data["pos"] = np.asarray(data["pos"], dtype=float)[::-1].copy()
    for *_ends, data in result.graph.edges(keys=True, data=True):
        data["voxels"] = [list(point)[::-1] for point in data["voxels"]]
    sweep = result.sweep_flows
    last = len(sweep.axis_values[sweep.axis_names[0]]) - 1
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    _apply_layers(viewer, ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=[result], output_dir=tmp_path)))
    panel._haemolynx_after_layers_applied()
    name = perturbation_layer_names(result.name)[0]
    dock = viewer.window._dock_widgets[f"{name} sweep"]
    (grid_slider,) = dock.findChildren(QAbstractSlider)

    def expected(point):
        tag = viewer.layers[name].metadata[RESULT_OURS]
        per_edge = np.asarray(sweep.flow_abs_at(point), dtype=float)[tag["sweep_edge_index"]]
        return per_edge[np.asarray(tag["segment_owner"], dtype=int)]

    def shown():
        return np.asarray(viewer.layers[name].features["flow_abs"], dtype=float)

    grid_slider.setValue(last)
    np.testing.assert_allclose(shown(), expected(last))
    # Flows are ~1e-13, so np.allclose's default atol of 1e-8 would call any
    # two of them equal.
    assert not np.allclose(expected(last), expected(0), rtol=1e-6, atol=0.0)

    rows = len(viewer.layers[name].data)
    panel._haemolynx_z_depth_slider.setValue((0.0, 900.0))
    assert len(viewer.layers[name].data) < rows  # the layer was replaced
    np.testing.assert_allclose(shown(), expected(last))

    grid_slider.setValue(0)
    np.testing.assert_allclose(shown(), expected(0))
