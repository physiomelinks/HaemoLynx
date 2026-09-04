"""Left-hand view dock: Z project, scale bar, and snapshot, display-only.

The pipeline still reads the full volume / full graph. These controls only
change what napari shows. The snapshot is a cosmetic canvas TIFF.
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    SNAPSHOT_NO_OUTPUT_FOLDER,
    SNAPSHOT_STEM,
    VIEW_DOCK_NAME,
    _apply_layers,
    _store_z_project_cache,
    data_for_pipeline,
    output_folder_from_settings,
    settings_widget,
    unique_snapshot_path,
)
from haemolynx.gui.results import IMAGE, NODES, VESSELS  # noqa: E402
from test_gui_results_widget import a_run  # noqa: E402

pytestmark = pytest.mark.gui


def test_the_view_panel_docks_on_the_left(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)

    dock = panel._haemolynx_view_dock
    assert dock is not None
    assert VIEW_DOCK_NAME in dock.windowTitle()
    assert dock.area == "left"
    assert panel._haemolynx_view_panel.objectName() == "haemolynx_view_panel"
    assert panel._haemolynx_z_project_slider.objectName() == "haemolynx_z_project_slider"
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
    slider = panel._haemolynx_z_project_slider
    ancestor = slider.parentWidget()
    while ancestor is not None:
        assert ancestor is not panel
        ancestor = ancestor.parentWidget()


def test_the_panel_still_builds_the_view_chrome_with_no_viewer():
    panel = settings_widget(napari_viewer=None)
    assert panel._haemolynx_view_dock is None
    assert panel._haemolynx_view_panel is not None
    assert panel._haemolynx_z_project_slider is not None
    assert panel._haemolynx_scale_bar.isChecked() is False
    assert panel._haemolynx_snapshot_button is not None
    panel._haemolynx_snapshot_button.click()
    assert "viewer" in panel._haemolynx_report().lower()


def test_z_project_slider_changes_display_and_full_range_restores(make_napari_viewer):
    from types import SimpleNamespace

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
    _store_z_project_cache(layer, image)

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
    slider = panel._haemolynx_z_project_slider
    assert slider.isEnabled()
    full_z = float(results.image_z_extent_um())
    slider.setValue((0.0, full_z))
    vessel_count = len(viewer.layers[VESSELS].data)
    node_count = len(viewer.layers[NODES].data)
    assert vessel_count > 1

    slider.setValue((0.0, 5.0))
    displayed = np.asarray(viewer.layers[IMAGE].data)
    assert displayed.shape == original.shape
    np.testing.assert_array_equal(displayed[0], 3)
    np.testing.assert_array_equal(displayed[3], 0)
    assert len(viewer.layers[VESSELS].data) < vessel_count
    assert len(viewer.layers[NODES].data) <= node_count

    slider.setValue((0.0, full_z))
    restored = np.asarray(viewer.layers[IMAGE].data)
    np.testing.assert_array_equal(restored, original)
    assert len(viewer.layers[VESSELS].data) == vessel_count
    assert len(viewer.layers[NODES].data) == node_count


def test_full_range_z_project_does_not_crop_pipeline_inputs(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)

    image_layer = viewer.layers[IMAGE]
    original = np.asarray(data_for_pipeline(image_layer)).copy()
    settings_before = dict(panel._haemolynx_values())

    slider = panel._haemolynx_z_project_slider
    slider.setValue((0.0, 5.0))
    np.testing.assert_array_equal(data_for_pipeline(image_layer), original)
    assert np.asarray(image_layer.data).shape == original.shape
    assert dict(panel._haemolynx_values()) == settings_before

    slider.setValue((0.0, 8.0))
    np.testing.assert_array_equal(data_for_pipeline(image_layer), original)
    np.testing.assert_array_equal(np.asarray(image_layer.data), original)


def test_scale_bar_checkbox_toggles_viewer_scale_bar(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_scale_bar
    assert box.isChecked() is False
    assert viewer.scale_bar.visible is False

    box.setChecked(True)
    assert viewer.scale_bar.visible is True

    box.setChecked(False)
    assert viewer.scale_bar.visible is False


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
    """Z-project changes the canvas; the TIFF is that view, not z_project_full."""
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
    _store_z_project_cache(layer, image)

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
    panel._haemolynx_z_project_slider.setValue((0.0, 5.0))
    displayed = np.asarray(layer.data)
    np.testing.assert_array_equal(displayed[0], 3)
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
    np.testing.assert_array_equal(seen["displayed"][0], 3)
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
