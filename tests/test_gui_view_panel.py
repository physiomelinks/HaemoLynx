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
    _scale_bar_overlay,
    _store_z_window_cache,
    data_for_pipeline,
    output_folder_from_settings,
    settings_widget,
    unique_snapshot_path,
)
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

