"""The classifier training window, built in the real panel: a Train button
in each classifier row; painting trains the classifier and classifies the
plane in view, and only that plane; saving fills the row and its switch; the
whole image segmented in the window is what a run makes of it."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")
pytest.importorskip("sklearn")

from haemolynx.classifier import classify_volume, load_classifier, segment_with_classifier  # noqa: E402
from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.classifier_training import CLASSIFIER_ROWS  # noqa: E402

from test_pixel_classifier import VOXEL, strokes, tube_volume, write_raw  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    rows = widget._haemolynx_rows()
    rows["raw_image_path"].value = raw
    return viewer, widget, rows, volume, truth, raw


def test_every_classifier_row_has_a_train_button(panel):
    _viewer, _widget, rows, *_ = panel
    for name in CLASSIFIER_ROWS:
        button = rows[name]._haemolynx_train_button
        assert button.text == "Train..." and button.tooltip, name
        assert button.native.parent() is rows[name].native, name


def test_painting_classifies_only_the_plane_in_view(panel):
    viewer, widget, _rows, volume, truth, _raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window is not None and window.live is not None
    points, classes = strokes(truth, per_class=150)
    window.live_box.setChecked(False)
    window.strokes.data[tuple(points.T)] = classes + 1
    viewer.dims.ndisplay = 2
    viewer.dims.order = (0, 1, 2)
    viewer.dims.set_point(0, 10 * VOXEL[0])
    window.update(retrain=True)
    preview = np.asarray(window.preview.data)
    whole = classify_volume([volume], VOXEL, window.live.classifier, workers=2)
    assert np.array_equal(preview[10], whole[10])
    others = np.delete(preview, 10, axis=0)
    assert not others.any()
    # Moving to another plane clears the last one and shows the new one.
    viewer.dims.set_point(0, 4 * VOXEL[0])
    window.update(retrain=False)
    preview = np.asarray(window.preview.data)
    assert not preview[10].any() and np.array_equal(preview[4], whole[4])


def test_saving_fills_the_row_and_turns_its_switch_on(panel, tmp_path):
    _viewer, widget, rows, _volume, truth, _raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    points, classes = strokes(truth, per_class=100)
    window.strokes.data[tuple(points.T)] = classes + 1
    target = tmp_path / "mine.haemoclf"
    window.path_edit.setText(str(target))
    window.save()
    assert target.exists()
    values = widget._haemolynx_values()
    assert str(values["pixel_classifier_path"]) == str(target)
    assert values["use_pixel_classifier_segmentation"] is True
    saved = load_classifier(target)
    assert saved.voxels_per_label() == {"Background": 100, "Vessel": 100}


def test_reopening_a_saved_classifier_paints_its_strokes_back(panel, tmp_path):
    _viewer, widget, rows, _volume, truth, _raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    points, classes = strokes(truth, per_class=60)
    window.strokes.data[tuple(points.T)] = classes + 1
    window.path_edit.setText(str(tmp_path / "c.haemoclf"))
    window.save()
    painted = np.array(window.strokes.data)
    window.window.close()
    again = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert again is not window
    assert np.array_equal(np.asarray(again.strokes.data), painted)


def test_the_whole_image_segmented_in_the_window_is_the_runs(panel, tmp_path):
    _viewer, widget, _rows, _volume, truth, raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    points, classes = strokes(truth, per_class=150)
    window.strokes.data[tuple(points.T)] = classes + 1
    target = tmp_path / "c.haemoclf"
    window.path_edit.setText(str(target))
    window.save()
    window.segment_whole_image()
    shown = np.asarray(window.segmentation.data)
    made = segment_with_classifier(raw, target, {"Vessel": tmp_path / "run.tif"})["Vessel"]
    assert np.array_equal(shown == 1, tifffile.imread(made) == 1)


def test_low_ram_paints_sparsely_and_reads_lazily(make_napari_viewer, tmp_path):
    from haemolynx.classifier.lazy import LazyTiffChannel
    from haemolynx.classifier.sparse import SparseLabels

    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    rows = widget._haemolynx_rows()
    rows["raw_image_path"].value = raw
    rows["use_memmap_loading"].value = True
    rows["output_dir"].value = tmp_path / "out"
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert isinstance(window.strokes.data, SparseLabels) and isinstance(window.preview.data, SparseLabels)
    assert isinstance(window.live.volumes[0], LazyTiffChannel)
    points, classes = strokes(truth, per_class=120)
    window.strokes.data[tuple(points.T.astype(np.int64))] = classes + 1
    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, 6 * VOXEL[0])
    window.update(retrain=True)
    whole = classify_volume([volume], VOXEL, window.live.classifier, workers=2)
    assert np.array_equal(np.asarray(window.preview.data[6]), whole[6])
    window.path_edit.setText(str(tmp_path / "c.haemoclf"))
    window.segment_whole_image()
    made = tmp_path / "out" / "raw_segmented.tif"
    assert made.exists() and np.array_equal(tifffile.imread(made), (whole == 1).astype(np.uint8))
    assert widget._haemolynx_values()["use_pixel_classifier_segmentation"] is True


def test_choosing_the_combined_job_keeps_the_strokes_by_name(panel):
    _viewer, widget, _rows, _volume, truth, _raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_large_vessel_path", synchronous=True)
    assert window.purpose == "vessel_classes" and window.job_box.isVisible() is not None
    points, classes = strokes(truth, per_class=40)
    window.live_box.setChecked(False)
    window.strokes.data[tuple(points.T)] = np.where(classes == 1, 3, 1)  # Venule and Background
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_classes"))
    assert window.purpose == "vessels_and_classes"
    assert window.labels == ("Background", "Capillary", "Arteriole", "Venule")
    assert window.job_box.currentText() == "Capillaries, arterioles and venules"
    values = np.asarray(window.strokes.data)[tuple(points.T)]
    assert set(np.unique(values)) == {1, 4}  # Venule is now the fourth label


def test_no_raw_image_says_what_to_do(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    assert widget._haemolynx_open_classifier_training("pixel_classifier_path") is None
    assert "raw image" in widget._haemolynx_report()
