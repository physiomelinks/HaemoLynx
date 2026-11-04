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


def test_the_window_takes_the_stylesheet_napari_gives_the_panel(panel):
    # napari sets its (dark) theme's stylesheet on its main window, the
    # panel's ancestor; a parentless window never got it and came up light.
    _viewer, widget, *_ = panel
    widget.setStyleSheet("QWidget { background-color: rgb(38, 41, 48); color: rgb(240, 241, 242); }")
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window.window.isWindow()
    for part in (window.window, window.job_box, window.label_box, window.status):
        part.ensurePolished()
        palette = part.palette()
        assert palette.color(part.backgroundRole()).getRgb()[:3] == (38, 41, 48), part
        assert palette.color(part.foregroundRole()).getRgb()[:3] == (240, 241, 242), part


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


BOTH_ROLES = {
    "pixel_classifier_large_vessel_path", "pixel_classifier_small_vessel_path",
}


def test_saving_a_classifier_telling_large_from_small_makes_the_boundaries_masks(panel, tmp_path):
    _viewer, widget, rows, _volume, truth, _raw = panel
    from haemolynx.io.vessel_mask_discovery import find_vessel_masks

    mask_rows = ("large_arteriole_mask_path", "large_venule_mask_path",
                 "small_arteriole_mask_path", "small_venule_mask_path")
    # Masks the panel's search would put in those rows, were it to look.
    for word in ("large_arteriole", "large_venule", "small_arteriole", "small_venule"):
        tifffile.imwrite(tmp_path / f"raw_{word}.tif", np.zeros((2, 4, 4), np.uint8))
    assert all(found.is_pair for found in find_vessel_masks([_raw]).values())
    before = {name: widget._haemolynx_values()[name] for name in mask_rows}
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_sized_classes"))
    assert window.job_box.currentText() == "Capillaries, large and small arterioles and venules"
    assert [b.text() for b in window.label_group.buttons()] == list(window.labels)
    points, classes = strokes(truth, per_class=60)
    window.live_box.setChecked(False)
    window.strokes.data[tuple(points.T)] = classes + 1  # Background and Capillary
    target = tmp_path / "sized.haemoclf"
    window.path_edit.setText(str(target))
    window.save()
    values = widget._haemolynx_values()
    for name in BOTH_ROLES:
        assert str(values[name]) == str(target), name
    for switch in ("use_pixel_classifier_segmentation", "automated_vessel_assignment",
                   "use_large_vessel_masks", "use_small_vessel_masks_for_boundary_assignment",
                   "use_pixel_classifier_large_vessel_segmentation",
                   "use_pixel_classifier_small_vessel_segmentation"):
        assert values[switch] is True, switch
    # The roles are made from the raw image: the search for mask files beside
    # it left their rows as they were.
    assert {name: values[name] for name in mask_rows} == before
    assert "large and small vessel masks" in widget._haemolynx_report()
    # The rows it filled are shown: their switches' chains are all on.
    for name in BOTH_ROLES:
        assert not rows[name].native.isHidden(), name


def test_choosing_a_combined_classifier_in_the_row_makes_the_large_masks(panel, tmp_path):
    from test_gui_classifier_training import _record_only

    _viewer, widget, rows, *_ = panel
    rows["use_pixel_classifier_segmentation"].value = True
    rows["pixel_classifier_path"].value = _record_only(tmp_path / "combined.haemoclf", "vessels_and_classes")
    values = widget._haemolynx_values()
    assert str(values["pixel_classifier_large_vessel_path"]) == str(tmp_path / "combined.haemoclf")
    assert values["use_large_vessel_masks"] is True
    assert values["use_pixel_classifier_large_vessel_segmentation"] is True
    assert values["use_small_vessel_masks_for_boundary_assignment"] is False
    assert values["pixel_classifier_small_vessel_path"] in (None, "")
    # A classifier of vessels alone changes nothing more.
    rows["use_large_vessel_masks"].value = False
    rows["pixel_classifier_path"].value = _record_only(tmp_path / "vessels.haemoclf", "vessels")
    assert widget._haemolynx_values()["use_large_vessel_masks"] is False


def test_loading_a_config_naming_such_a_classifier_leaves_the_boundaries_as_it_says(panel, tmp_path):
    from test_gui_classifier_training import _record_only

    _viewer, widget, _rows, _volume, _truth, raw = panel
    path = _record_only(tmp_path / "sized.haemoclf", "vessels_and_sized_classes")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"raw_image_path: {raw.as_posix()}\nuse_pixel_classifier_segmentation: true\n"
        f"pixel_classifier_path: {path.as_posix()}\n",
        encoding="utf-8",
    )
    widget._haemolynx_load_config(config)
    values = widget._haemolynx_values()
    assert str(values["pixel_classifier_path"]) == str(path)
    assert values["use_large_vessel_masks"] is False
    assert values["pixel_classifier_large_vessel_path"] in (None, "")


EXCLUSIVE = (
    ("use_ilastik_segmentation", "use_pixel_classifier_segmentation", {}),
    ("use_ilastik_large_vessel_segmentation", "use_pixel_classifier_large_vessel_segmentation",
     {"automated_vessel_assignment": True, "use_large_vessel_masks": True}),
    ("use_ilastik_small_vessel_segmentation", "use_pixel_classifier_small_vessel_segmentation",
     {"automated_vessel_assignment": True, "use_small_vessel_masks_for_boundary_assignment": True}),
)


@pytest.mark.parametrize("ilastik, classifier, gates", EXCLUSIVE)
def test_ilastik_and_the_classifier_grey_each_other_and_say_why(panel, ilastik, classifier, gates):
    from haemolynx.gui.form import exclusive_reason

    _viewer, _widget, rows, *_ = panel
    for name, value in gates.items():
        rows[name].value = value
    assert rows[ilastik].enabled and rows[classifier].enabled
    rows[ilastik].value = True
    assert not rows[classifier].enabled and not rows[classifier].native.isHidden()
    assert rows[classifier].tooltip == exclusive_reason(ilastik)
    assert rows[ilastik].enabled
    rows[ilastik].value = False
    rows[classifier].value = True
    assert not rows[ilastik].enabled and not rows[ilastik].native.isHidden()
    assert rows[ilastik].tooltip == exclusive_reason(classifier)
    rows[classifier].value = False
    assert rows[ilastik].enabled and rows[classifier].enabled


def test_a_config_ticking_both_leaves_ilastik_free_to_untick(panel, tmp_path):
    _viewer, widget, rows, _volume, _truth, raw = panel
    config = tmp_path / "both.yaml"
    config.write_text(
        f"raw_image_path: {raw.as_posix()}\nuse_ilastik_segmentation: true\n"
        "use_pixel_classifier_segmentation: true\n",
        encoding="utf-8",
    )
    widget._haemolynx_load_config(config)
    assert rows["use_ilastik_segmentation"].enabled
    assert not rows["use_pixel_classifier_segmentation"].enabled
    rows["use_ilastik_segmentation"].value = False
    assert rows["use_pixel_classifier_segmentation"].enabled
    assert not rows["use_ilastik_segmentation"].enabled


def test_saving_a_classifier_chooses_it_over_ilastik(panel, tmp_path):
    _viewer, widget, rows, _volume, truth, _raw = panel
    rows["use_ilastik_segmentation"].value = True
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    points, classes = strokes(truth, per_class=40)
    window.live_box.setChecked(False)
    window.strokes.data[tuple(points.T)] = classes + 1
    window.path_edit.setText(str(tmp_path / "mine.haemoclf"))
    window.save()
    values = widget._haemolynx_values()
    assert values["use_ilastik_segmentation"] is False
    assert values["use_pixel_classifier_segmentation"] is True
    assert rows["use_pixel_classifier_segmentation"].enabled


def test_no_raw_image_says_what_to_do(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    assert widget._haemolynx_open_classifier_training("pixel_classifier_path") is None
    assert "raw image" in widget._haemolynx_report()
