"""The classifier training window, built in the real panel: a Train button
in each classifier row; painting trains the classifier and classifies the
plane in view, and only that plane; saving fills the row and its switch; the
whole image segmented in the window is what a run makes of it."""
from __future__ import annotations

import re

import numpy as np
import pytest
import tifffile

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")
pytest.importorskip("sklearn")

from haemolynx.classifier import (  # noqa: E402
    classify_volume,
    load_classifier,
    save_classifier,
    segment_with_classifier,
)
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
    # Loading ran to its end (it once stopped at napari's old camera name).
    assert window.status.text().startswith("Paint a few strokes of each label")
    assert window.save_button.isEnabled() and window.import_button.isEnabled()
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
    rows["output_dir"].value = tmp_path / "out"
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_sized_classes"))
    assert window.job_box.currentText() == "Capillaries, large and small arterioles and venules"
    assert [b.text() for b in window.label_group.buttons()] == [*window.labels, "Eraser"]
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
    # The roles are made from the raw image: their rows name the masks the
    # classifier writes -- one global mask a role, as it was shown no venule --
    # never the files the search for masks beside the image would have found.
    out = tmp_path / "out"
    assert _same_file(values["large_arteriole_mask_path"], out / "raw_large_global_segmented.tif")
    assert _same_file(values["small_arteriole_mask_path"], out / "raw_small_global_segmented.tif")
    assert values["large_venule_mask_path"] in (None, "")
    assert values["small_venule_mask_path"] in (None, "")
    for name in mask_rows:
        assert not rows[name].native.isHidden() and not rows[name].enabled, name
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


def _painted(window):
    from haemolynx.classifier.sparse import painted_points

    points, values = painted_points(window.strokes.data)
    return {tuple(int(v) for v in p): int(value) for p, value in zip(points, values)}


def test_importing_ilastik_labels_paints_them_and_trains(panel, tmp_path):
    from test_classifier_ilastik_labels import write_project

    _viewer, widget, _rows, volume, truth, _raw = panel
    block = np.zeros((1, 2, 3), np.uint8)
    block[0, 0, :] = 2  # Label 2: vessel
    block[0, 1, :] = 1  # Label 1: background
    project = write_project(tmp_path / "p.ilp", images=(("raw.tif", "zyx", volume.shape),),
                            blocks={0: [((5, 10, 20), block)]})
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.live_box.setChecked(False)
    assert window.import_button.toolTip()
    count = window.import_ilastik_labels(project, mapping={1: "Background", 2: "Vessel"})
    assert count == 6
    painted = _painted(window)
    assert all(painted[(5, 10, 20 + i)] == 2 and painted[(5, 11, 20 + i)] == 1 for i in range(3))
    assert window.live.classifier.voxels_per_label() == {"Background": 3, "Vessel": 3}
    assert "Imported 6 labelled voxels from p.ilp" in window.status.text()
    # A label left out is not painted.
    assert window.import_ilastik_labels(project, mapping={1: None, 2: "Vessel"}) == 3


def test_import_guesses_the_vessel_label_from_the_runs_ilastik_setting(panel, tmp_path, monkeypatch):
    from test_classifier_ilastik_labels import write_project

    _viewer, widget, rows, volume, _truth, _raw = panel
    rows["ilastik_vessel_label"].value = "Label 2"
    project = write_project(tmp_path / "p.ilp", images=(("raw.tif", "zyx", volume.shape),),
                            blocks={0: [((0, 0, 0), np.ones((1, 1, 1), np.uint8))]})
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    asked = {}

    def ask(_project, names, guess):
        asked.update(names=names, guess=guess)
        return None  # cancelled

    monkeypatch.setattr(window, "_ask_label_mapping", ask)
    assert window.import_ilastik_labels(project) == 0
    assert asked == {"names": ("Label 1", "Label 2"), "guess": {1: "Background", 2: "Vessel"}}
    assert not _painted(window)


def test_segment_whole_image_says_how_far_it_is_stops_and_picks_up(panel, tmp_path, monkeypatch):
    from haemolynx.classifier import apply as apply_module
    from haemolynx.classifier.slabs import SlabStore, slab_folder

    _viewer, widget, rows, volume, truth, _raw = panel
    rows["output_dir"].value = tmp_path / "out"
    monkeypatch.setattr(apply_module, "slab_planes", lambda *_a, **_k: 3)
    monkeypatch.setattr(apply_module, "PROGRESS_SECONDS", 0.0)
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.live_box.setChecked(False)
    points, classes = strokes(truth, per_class=60)
    window.strokes.data[tuple(points.T)] = classes + 1
    classifier_path = tmp_path / "c.haemoclf"
    window.path_edit.setText(str(classifier_path))
    original = SlabStore.put
    puts = [0]

    def put(store, *args):
        original(store, *args)
        puts[0] += 1
        if puts[0] == 2:
            window._stop_segmenting = True  # as the Stop button does

    monkeypatch.setattr(SlabStore, "put", put)
    window.segment_whole_image()
    assert window.status.text().startswith("Stopped. What was segmented is kept")
    assert window.segment_button.text() == "Segment whole image"
    assert window._progress_text.startswith("Classifying the raw image, ")
    assert re.search(r"\(\d+% complete\)\.$", window._progress_text)
    output = tmp_path / "out" / "raw_segmented.tif"
    assert not output.exists() and any(slab_folder(output).glob("*.tif"))
    saved = classifier_path.stat().st_mtime_ns
    monkeypatch.setattr(SlabStore, "put", original)
    window.segment_whole_image()
    assert output.exists() and not slab_folder(output).exists()
    assert classifier_path.stat().st_mtime_ns == saved  # the same classifier, taken up
    whole = classify_volume([volume], VOXEL, load_classifier(classifier_path), workers=2)
    assert np.array_equal(tifffile.imread(output), (whole == 1).astype(np.uint8))


def test_the_window_says_when_the_image_is_unlike_the_classifiers(make_napari_viewer, tmp_path):
    from test_classifier_familiar import _trained_with_profile

    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    rows = widget._haemolynx_rows()
    volume, truth = tube_volume()
    path = save_classifier(_trained_with_profile(volume, truth), tmp_path / "c.haemoclf")
    rows["raw_image_path"].value = write_raw(tmp_path / "inverted.tif", (volume.max() - volume).astype(np.uint16))
    rows["use_pixel_classifier_segmentation"].value = True
    rows["pixel_classifier_path"].value = path
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert "inverted.tif is unlike every image c.haemoclf was trained on" in window.status.text()


def _memory(monkeypatch, free, total=None):
    """The machine as having *free* bytes free, every read weighed against them."""
    from haemolynx.classifier import memory

    monkeypatch.setattr(memory, "CHECKED_READ_BYTES", 0)
    monkeypatch.setattr(memory, "memory_bytes", lambda: (free, total or free))


def test_an_image_too_large_to_read_whole_is_refused_until_low_ram_is_on(panel, monkeypatch):
    viewer, widget, rows, volume, _truth, _raw = panel
    _memory(monkeypatch, volume.nbytes)  # 70% of it is less than the image
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window.load_failed and window.live is None
    assert "takes about" in window.status.text() and "Low RAM option" in window.status.text()
    assert not any(layer.name.startswith("Classifier") for layer in viewer.layers)
    rows["use_memmap_loading"].value = True
    again = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert again is not window and not again.load_failed and again.live is not None
    assert again.status.text().startswith("Paint a few strokes")


def test_a_lazily_read_image_keeps_napari_in_2d(panel, monkeypatch):
    viewer, widget, rows, volume, _truth, _raw = panel
    rows["use_memmap_loading"].value = True
    viewer.dims.ndisplay = 3  # already in 3D: the window puts it back before adding its layers
    _memory(monkeypatch, volume.nbytes - 1)  # a read of the whole image raises
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window.sparse and viewer.dims.ndisplay == 2
    monkeypatch.setattr(window, "update", lambda **_kw: None)  # the preview is not what is tested
    viewer.dims.ndisplay = 3  # raised MemoryError before the guard: the whole image read
    assert viewer.dims.ndisplay == 2
    assert window.status.text().startswith("The view stays in 2D")
    # Painting and moving through planes in 2D still work.
    viewer.dims.set_point(0, 5 * VOXEL[0])
    window.strokes.data[5, 10, 10] = 2
    assert window.strokes.data[5, 10, 10] == 2
    # With the window's layers removed, 3D is the viewer's again.
    for layer in window._lazy_layers_open():
        viewer.layers.remove(layer)
    assert not window._blocking
    viewer.add_image(np.zeros((4, 8, 8)), name="small")
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3


def test_segment_whole_image_is_refused_when_reading_it_again_would_not_fit(panel, monkeypatch, tmp_path):
    from haemolynx.classifier import memory

    _viewer, widget, rows, volume, truth, _raw = panel
    rows["output_dir"].value = tmp_path / "out"
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.live_box.setChecked(False)
    points, classes = strokes(truth, per_class=40)
    window.strokes.data[tuple(points.T)] = classes + 1
    window.path_edit.setText(str(tmp_path / "c.haemoclf"))
    monkeypatch.setattr(memory, "memory_bytes", lambda: (volume.nbytes, volume.nbytes))
    window.segment_whole_image()
    assert "takes about" in window.status.text()
    assert not window._segmenting and not (tmp_path / "out" / "raw_segmented.tif").exists()
    assert not (tmp_path / "c.haemoclf").exists()  # refused before anything was saved


def test_opening_the_window_leaves_the_runs_input_alone(panel, tmp_path):
    """The window's image layer is the panel's own: it was once taken for an
    image dropped in, made the run's input and written out whole."""
    viewer, widget, rows, volume, _truth, _raw = panel
    segmented = tmp_path / "segmented.tif"
    tifffile.imwrite(segmented, (volume > 1000).astype(np.uint8))
    rows["input_path"].value = segmented
    window = widget._haemolynx_open_classifier_training("pixel_classifier_large_vessel_path", synchronous=True)
    assert window.live is not None
    assert str(widget._haemolynx_values()["input_path"]) == str(segmented)
    assert all(layer.metadata.get("haemolynx") for layer in viewer.layers if layer.name.startswith("Classifier"))


def test_no_raw_image_says_what_to_do(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    assert widget._haemolynx_open_classifier_training("pixel_classifier_path") is None
    assert "raw image" in widget._haemolynx_report()


def _same_file(value, path) -> bool:
    from pathlib import Path

    return value not in (None, "") and Path(value).resolve() == Path(path).resolve()


def test_the_classifier_keeps_the_segmented_image_row_in_view_naming_its_mask(panel, tmp_path):
    from haemolynx.gui.form import classifier_filled_reason

    _viewer, widget, rows, *_ = panel
    rows["output_dir"].value = tmp_path / "out"
    mask = tmp_path / "out" / "raw_segmented.tif"
    rows["use_pixel_classifier_segmentation"].value = True
    row = rows["input_path"]
    assert not row.native.isHidden() and not row.enabled
    assert row.tooltip == classifier_filled_reason("use_pixel_classifier_segmentation")
    assert _same_file(widget._haemolynx_values()["input_path"], mask)
    # The row names what the run reads while the classifier makes it.
    rows["input_path"].value = tmp_path / "other.tif"
    assert _same_file(widget._haemolynx_values()["input_path"], mask)
    rows["output_dir"].value = tmp_path / "elsewhere"
    assert _same_file(widget._haemolynx_values()["input_path"], tmp_path / "elsewhere" / "raw_segmented.tif")
    # Unticked, the row is the user's again, still naming the classifier's mask.
    rows["use_pixel_classifier_segmentation"].value = False
    assert not row.native.isHidden() and row.enabled
    assert _same_file(widget._haemolynx_values()["input_path"], tmp_path / "elsewhere" / "raw_segmented.tif")
    # ilastik segmenting the image hides it, as before.
    rows["use_ilastik_segmentation"].value = True
    assert row.native.isHidden()


def test_a_combined_classifier_fills_the_vessel_mask_rows_with_its_masks(panel, tmp_path):
    from test_gui_classifier_training import _record_with_counts

    _viewer, widget, rows, *_ = panel
    rows["output_dir"].value = tmp_path / "out"
    path = _record_with_counts(tmp_path / "c.haemoclf", "vessels_and_classes",
                               {"Background": 9, "Capillary": 9, "Arteriole": 9, "Venule": 9})
    rows["use_pixel_classifier_segmentation"].value = True
    rows["pixel_classifier_path"].value = path
    values = widget._haemolynx_values()
    out = tmp_path / "out"
    expected = {
        "input_path": out / "raw_segmented.tif",
        "large_arteriole_mask_path": out / "raw_large_arteriole_segmented.tif",
        "large_venule_mask_path": out / "raw_large_venule_segmented.tif",
    }
    for name, mask in expected.items():
        assert _same_file(values[name], mask), name
        assert not rows[name].native.isHidden() and not rows[name].enabled, name
    # A classifier never shown a venule makes one global mask: the venule row is emptied.
    _record_with_counts(path, "vessels_and_classes", {"Background": 9, "Capillary": 9, "Arteriole": 9})
    rows["output_dir"].value = tmp_path / "out2"
    values = widget._haemolynx_values()
    assert _same_file(values["large_arteriole_mask_path"], tmp_path / "out2" / "raw_large_global_segmented.tif")
    assert values["large_venule_mask_path"] in (None, "")


def test_the_object_mask_classifiers_keep_their_mask_rows_in_view_naming_their_masks(panel, tmp_path):
    from haemolynx.gui.form import classifier_filled_reason

    _viewer, widget, rows, *_ = panel
    out = tmp_path / "out"
    rows["output_dir"].value = out
    pericytes, cells = "use_pixel_classifier_pericyte_segmentation", "use_pixel_classifier_cell_segmentation"
    rows["measure_pericytes"].value = True
    rows["run_haemodynamics"].value = True
    rows["measurement_3d_to_cell_mask"].value = True
    rows[pericytes].value = True
    rows[cells].value = True
    values = widget._haemolynx_values()
    for name, switch, mask in (
        ("pericyte_measurement_mask_path", pericytes, out / "raw_pericytes_segmented.tif"),
        ("cell_mask_path", cells, out / "raw_cells_segmented.tif"),
    ):
        assert not rows[name].native.isHidden() and not rows[name].enabled, name
        assert rows[name].tooltip == classifier_filled_reason(switch), name
        assert _same_file(values[name], mask), name

    # A perturbation entry reading the block or the pericyte mask shows it
    # too, greyed, with the classifier's rows under it, and never takes it as
    # its override (the pericyte mask's only row is an entry's).
    rows["run_perturbations"].value = True
    perturbations = widget._haemolynx_perturbations
    perturbations.add()
    perturbations.choose_type(0, "capillary_block_mask")
    perturbations.add()
    perturbations.choose_type(1, "pericyte_measured_constriction")
    rows["use_pixel_classifier_block_segmentation"].value = True
    for index, (name, kind, switch, mask) in enumerate((
        ("capillary_block_mask_path", "blocks", "use_pixel_classifier_block_segmentation",
         out / "raw_blocks_segmented.tif"),
        ("pericyte_mask_path", "pericytes", pericytes, out / "raw_pericytes_segmented.tif"),
    )):
        editor = perturbations.editors()[index]
        row = editor.editors[name]
        assert not row.native.isHidden() and not row.enabled, name
        assert row.tooltip == classifier_filled_reason(switch), name
        assert _same_file(row.value, mask), name
        assert name not in (perturbations.entries()[index].get("overrides") or {}), name
        widgets = list(editor.container)
        slot = widgets.index(next(w for w in widgets if w is editor.classifier_slots[kind]))
        assert widgets[slot - 1] is row, name
    block = perturbations.editors()[0].editors["capillary_block_mask_path"]

    # Unticked, each row is the user's again, still naming the classifier's mask.
    rows["use_pixel_classifier_block_segmentation"].value = False
    rows[pericytes].value = False
    assert block.enabled and not block.native.isHidden()
    assert _same_file(block.value, out / "raw_blocks_segmented.tif")
    row = rows["pericyte_measurement_mask_path"]
    assert row.enabled and _same_file(widget._haemolynx_values()[row.name], out / "raw_pericytes_segmented.tif")


# ------------------------------------------------------------ the display


def _lut(colour):
    """An ImageJ display LUT ending on *colour* (its channels' on/off)."""
    table = np.zeros((3, 256), np.uint8)
    for index, on in enumerate(colour):
        if on:
            table[index] = np.arange(256)
    return table


def _composite(path, channels, *, luts=None):
    """A Z C Y X ImageJ composite of *channels*, as Fiji saves one."""
    metadata = {"axes": "ZCYX", "spacing": VOXEL[0], "unit": "um", "mode": "composite"}
    if luts is not None:
        metadata["LUTs"] = luts
    tifffile.imwrite(path, np.stack(channels, axis=1), imagej=True,
                     resolution=(1 / VOXEL[2], 1 / VOXEL[1]), metadata=metadata)
    return path


@pytest.fixture
def two_channels(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    volume, truth = tube_volume()
    other = np.random.default_rng(3).integers(100, 900, volume.shape).astype(np.uint16)
    raw = _composite(tmp_path / "composite.tif", [other, volume])
    rows = widget._haemolynx_rows()
    rows["raw_image_path"].value = raw
    return viewer, widget, rows, other, volume


def test_each_channel_is_shown_in_its_own_colour_whichever_the_classifier_reads(two_channels):
    from haemolynx.classifier.lazy import LazyTiffChannel

    viewer, widget, _rows, other, volume = two_channels
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window.live.classifier.spec.channels == (0,)  # the classifier reads C1 alone
    first, second = window.images[0], window.images[1]
    assert window.image is first
    assert [first.name, second.name] == ["Classifier image C1 (vessels)", "Classifier image C2 (vessels)"]
    assert (first.colormap.name, second.colormap.name) == ("red", "green")  # Fiji's, the file giving none
    assert (first.blending, second.blending) == ("translucent_no_depth", "additive")
    assert first.visible and second.visible
    assert np.array_equal(np.asarray(first.data), other)
    # The channel only shown is read where it is looked at, never whole.
    assert isinstance(second.data, LazyTiffChannel) and 1 not in window._volumes
    assert np.array_equal(np.asarray(second.data[7]), volume[7])
    rows = window.display_rows
    assert [rows[c].show.text() for c in (0, 1)] == ["C1", "C2"]
    assert [rows[c].colour.currentData() for c in (0, 1)] == ["red", "green"]
    assert all(rows[c].limits.isEnabled() for c in (0, 1))
    # Below what is painted and what is classified.
    names = [layer.name for layer in viewer.layers]
    assert names.index(second.name) < names.index(window.preview.name) < names.index(window.strokes.name)
    # Opened again, the channels' layers are replaced, not doubled.
    window.window.close()
    again = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert again is not window
    assert sorted(layer.name for layer in viewer.layers if layer.name.startswith("Classifier image")) == [
        "Classifier image C1 (vessels)", "Classifier image C2 (vessels)",
    ]


def test_a_channel_with_a_colour_of_its_own_is_shown_in_it(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    volume, _truth = tube_volume()
    raw = _composite(tmp_path / "lut.tif", [volume, volume], luts=[_lut((0, 1, 0)), _lut((1, 0, 1))])
    widget._haemolynx_rows()["raw_image_path"].value = raw
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert (window.images[0].colormap.name, window.images[1].colormap.name) == ("green", "magenta")
    assert window.display_rows[0].show.text() == "C1 (green)"


def test_the_display_box_sets_each_channels_range_colour_and_visibility_not_the_classifiers(two_channels):
    from haemolynx.classifier.features import display_range

    _viewer, widget, _rows, _other, volume = two_channels
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    layer, row = window.images[1], window.display_rows[1]
    limits, span = display_range(volume)
    assert tuple(layer.contrast_limits) == pytest.approx(limits)
    assert tuple(layer.contrast_limits_range) == pytest.approx(span)  # napari's own slider opens out too
    assert row.limits.value() == pytest.approx(limits)
    assert (row.limits.minimum(), row.limits.maximum()) == pytest.approx(span)
    ranges, spec = window.live.ranges, window.live.classifier.spec
    # Opened out past the range it was first shown at, then narrowed.
    row.limits.setValue(span)
    assert tuple(layer.contrast_limits) == pytest.approx(span)
    row.limits.setValue((1000.0, 2000.0))
    assert tuple(layer.contrast_limits) == pytest.approx((1000.0, 2000.0))
    assert row.limits.edge_labels() == ("1000", "2000")
    # Both handles at one place: still a range napari takes.
    row.limits.setValue((1500.0, 1500.0))
    low, high = layer.contrast_limits
    assert low == pytest.approx(1500.0) and high > low
    # napari's own layer controls move the box with them, and widen it.
    layer.contrast_limits = (500.0, 900.0)
    assert row.limits.value() == pytest.approx((500.0, 900.0))
    assert row.limits.edge_labels() == ("500", "900")
    layer.contrast_limits_range = (0.0, 65535.0)
    assert (row.limits.minimum(), row.limits.maximum()) == (0.0, 65535.0)
    # Colour and visibility, each way.
    row.colour.setCurrentIndex(row.colour.findData("magenta"))
    assert layer.colormap.name == "magenta"
    row.show.setChecked(False)
    assert not layer.visible
    layer.visible = True
    assert row.show.isChecked()
    # None of it reaches the classifier, nor touches the other channel.
    assert window.live.ranges == ranges and window.live.classifier.spec == spec
    assert window.images[0].colormap.name == "red" and window.images[0].visible


def test_an_image_of_one_channel_is_grey_and_its_range_opens_out(panel):
    viewer, widget, _rows, volume, _truth, _raw = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert list(window.images) == [None] and window.image is window.images[None]
    layer = window.image
    assert layer.name == "Classifier image (vessels)" and layer.colormap.name == "gray"
    assert window.display_rows[None].show.text() == "Raw image"
    limits = window.live.ranges[0]
    assert tuple(layer.contrast_limits) == pytest.approx(limits)
    # napari's slider once spanned the limits alone: the range could only narrow.
    low, high = layer.contrast_limits_range
    assert (low, high) == (float(volume.min()), float(volume.max()))
    assert low <= limits[0] < limits[1] < high  # the image is clipped at 0, its lowest percentile
    window.display_rows[None].limits.setValue((float(volume.min()), float(volume.max())))
    assert tuple(layer.contrast_limits) == pytest.approx((volume.min(), volume.max()))


def test_a_channel_read_only_whole_that_would_not_fit_is_left_out_and_said(two_channels, monkeypatch):
    from haemolynx.gui import classifier_training

    viewer, widget, rows, *_ = two_channels
    rows["image_axis_order"].value = "zxy"  # reordered: no channel can be read lazily
    refusals = iter([None, "too large"])  # the classifier's channel fits; the other does not
    monkeypatch.setattr(classifier_training, "memory_refusal", lambda *_a, **_k: next(refusals))
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    assert window.live is not None and list(window.images) == [0]
    assert not window.display_rows[1].show.isChecked() and not window.display_rows[1].limits.isEnabled()
    assert "Not shown, as it would not fit in memory: C2." in window.status.text()
    assert not any(layer.name == "Classifier image C2 (vessels)" for layer in viewer.layers)


# ------------------------------------------------------------ the eraser


def test_the_eraser_is_napari_s_erase_tool_and_follows_it(panel):
    _viewer, widget, *_ = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_large_vessel_path", synchronous=True)
    group, strokes_layer = window.label_group, window.strokes
    assert group.button(window.ERASER).text() == "Eraser" and group.button(window.ERASER).toolTip()
    group.button(window.ERASER).click()
    assert strokes_layer.mode == "erase"
    group.button(3).click()  # Venule
    assert strokes_layer.mode == "paint" and strokes_layer.selected_label == 3
    # Chosen in napari's own layer controls, the Paint box follows.
    strokes_layer.mode = "erase"
    assert group.checkedId() == window.ERASER
    strokes_layer.mode = "paint"
    assert group.checkedId() == 3
    # Another job: its labels, the eraser still under them, painting its second label.
    group.button(window.ERASER).click()
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_classes"))
    assert [b.text() for b in group.buttons()] == ["Background", "Capillary", "Arteriole", "Venule", "Eraser"]
    assert group.checkedId() == 2 and strokes_layer.mode == "paint" and strokes_layer.selected_label == 2


@pytest.mark.parametrize("low_ram", [False, True])
def test_strokes_rubbed_out_are_forgotten_when_it_trains_again(make_napari_viewer, tmp_path, low_ram):
    from haemolynx.classifier.sparse import painted_points

    viewer = make_napari_viewer()
    widget = settings_widget(napari_viewer=viewer)
    volume, truth = tube_volume()
    rows = widget._haemolynx_rows()
    rows["raw_image_path"].value = write_raw(tmp_path / "raw.tif", volume)
    rows["use_memmap_loading"].value = low_ram
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    points, classes = strokes(truth, per_class=60)
    window.strokes.data[tuple(points.T.astype(np.int64))] = classes + 1
    # A square painted Vessel by mistake, on plane 10, away from every other stroke there.
    others = points[points[:, 0] == 10][:, 1:]
    centre = max(((y, x) for y in range(6, 42) for x in range(6, 50)),
                  key=lambda p: np.min(np.abs(others - p).max(axis=1)) if len(others) else 0)
    square = np.array([(10, centre[0] + dy, centre[1] + dx) for dy in range(-2, 3) for dx in range(-2, 3)])
    window.strokes.data[tuple(square.T)] = 2
    viewer.dims.ndisplay = 2
    viewer.dims.order = (0, 1, 2)
    viewer.dims.set_point(0, 10 * VOXEL[0])
    window.update(retrain=True)
    trained = {tuple(p) for p in window.live.classifier.training.points_zyx.astype(int)}
    assert {tuple(p) for p in square} <= trained
    window.label_group.button(window.ERASER).click()
    window.strokes.brush_size = 7
    # One stroke of the eraser, as a drag makes it: napari says it painted once the stroke is done.
    with window.strokes.block_history():
        window.strokes.paint((10, *centre), window.strokes.colormap.background_value)
    assert not np.asarray([window.strokes.data[tuple(p)] for p in square]).any()
    # Trained again on the stroke, without what it rubbed out; the rest kept.
    left, _values = painted_points(window.strokes.data)
    trained = {tuple(p) for p in window.live.classifier.training.points_zyx.astype(int)}
    assert trained == {tuple(p) for p in left.astype(int)}
    assert not trained & {tuple(p) for p in square}
    assert {tuple(p) for p in points} <= trained


def test_the_brush_size_box_sizes_what_the_brush_paints_and_follows_napari(panel):
    viewer, widget, *_ = panel
    window = widget._haemolynx_open_classifier_training("pixel_classifier_path", synchronous=True)
    window.live_box.setChecked(False)
    box, strokes_layer = window.brush_box, window.strokes
    assert box.value() == strokes_layer.brush_size == 3 and box.toolTip()
    viewer.dims.ndisplay = 2
    viewer.dims.order = (0, 1, 2)
    viewer.dims.set_point(0, 10 * VOXEL[0])

    def painted_by_a_stroke(at):
        before = np.count_nonzero(np.asarray(strokes_layer.data))
        with strokes_layer.block_history():
            strokes_layer.paint((10, *at), 2)
        return np.count_nonzero(np.asarray(strokes_layer.data)) - before

    box.setValue(1)
    assert strokes_layer.brush_size == 1 and painted_by_a_stroke((10, 10)) == 1
    box.setValue(7)
    assert strokes_layer.brush_size == 7 and painted_by_a_stroke((30, 30)) > 20
    # The eraser rubs out as wide as the box says.
    window.label_group.button(window.ERASER).click()
    assert strokes_layer.brush_size == 7
    # napari's own slider, or its [ and ] keys: the box follows, its range growing past its end.
    strokes_layer.brush_size = 12
    assert box.value() == 12
    strokes_layer.brush_size = window.BRUSH_MAX + 20
    assert box.maximum() == box.value() == window.BRUSH_MAX + 20


# ------------------------------------------------------- jobs on worker threads


def _threaded_window(widget, qtbot, row="pixel_classifier_path"):
    """The window as the panel runs it: its jobs on worker threads. With the
    idle test to wait on: loaded, no job running or waiting, no preview due."""
    window = widget._haemolynx_open_classifier_training(row)

    def idle():
        return (
            window.live is not None and not window._busy and window._queued is None
            and getattr(window, "_kept", None) is None and not window.timer.isActive()
        )

    qtbot.waitUntil(idle, timeout=30000)
    return window, idle


def _sized_strokes(truth):
    """Strokes of every label of the job telling large from small: the
    background, and the vessels shared out among the five vessel labels."""
    points, classes = strokes(truth, per_class=200)
    vessel = classes == 1
    x = points[:, 2]
    shares = np.digitize(x, np.quantile(x[vessel], [0.2, 0.4, 0.6, 0.8]))
    return points, np.where(vessel, 2 + shares, 1).astype(np.uint8)


def test_a_job_chosen_while_a_preview_runs_is_not_lost(panel, qtbot):
    """One job at a time, the latest request waiting: the new job's classifier,
    asked for while a preview ran, was dropped for the next preview asked for.
    The window kept the old classifier, deaf to the new labels' strokes, with
    Update now greyed: arterioles and venules never showed."""
    _viewer, widget, _rows, _volume, truth, _raw = panel
    window, idle = _threaded_window(widget, qtbot)
    window.live_box.setChecked(False)
    points, classes = strokes(truth, per_class=100)
    window.strokes.data[tuple(points.T)] = classes + 1
    window.update(retrain=True)  # a preview runs...
    assert window._busy
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_sized_classes"))  # ...the job changes
    window.update(retrain=False)  # ...and a move asks for another preview
    qtbot.waitUntil(idle, timeout=30000)
    assert window.live.classifier.purpose == window.purpose == "vessels_and_sized_classes"
    assert window.update_button.isEnabled()
    points, values = _sized_strokes(truth)
    window.strokes.data[tuple(points.T)] = values
    window.update_button.click()
    qtbot.waitUntil(idle, timeout=30000)
    assert "Large arteriole" in window.status.text() and "Small venule" in window.status.text()
    shown = np.asarray(window.preview.data)[window._shown[1]]
    assert set(np.unique(shown).tolist()) & {2, 3, 4, 5}  # arterioles and venules classified, and shown


def test_a_training_message_is_said_in_the_window_not_raised_into_napari(panel, qtbot):
    """A new job with too few labels painted fails to train: said on the
    status line. superqt also raised it into napari, as an error, until the
    worker's handlers were given at its creation (pytest-qt fails a test on
    an error raised in Qt's event loop)."""
    _viewer, widget, *_ = panel
    window, idle = _threaded_window(widget, qtbot)
    window.job_box.setCurrentIndex(window.purposes.index("vessels_and_sized_classes"))
    qtbot.waitUntil(idle, timeout=30000)
    assert window.status.text().startswith("Paint at least two labels")
    assert window.update_button.isEnabled()
