"""A classifier segmentation stopped part-way and taken up again
(``classifier.slabs``), and how far one has got (``SegmentProgress``): the
masks the same bit for bit as one made in one go, whatever the slab size, the
grid, or how many masks share the pass."""
from __future__ import annotations

import logging

import numpy as np
import pytest
import tifffile

from haemolynx.classifier import apply as apply_module
from haemolynx.classifier import save_classifier, segment_with_classifier
from haemolynx.classifier.apply import SegmentProgress, remaining_seconds
from haemolynx.classifier.slabs import SlabStore, full_size_planes, slab_folder

from test_classifier_scale import trained
from test_pixel_classifier import trained_classifier, tube_volume, write_raw

pytest.importorskip("sklearn")


class Stop(Exception):
    pass


@pytest.fixture
def image(tmp_path):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    path = save_classifier(trained_classifier(volume, truth), tmp_path / "c.haemoclf")
    return volume, truth, raw, path


def _slabs_of(planes: int):
    return lambda *_args, **_kwargs: planes


def _stopping_after(monkeypatch, puts: int):
    """A poll that stops the pass once *puts* slabs have been handed over."""
    count = [0]
    original = SlabStore.put

    def put(store, *args):
        original(store, *args)
        count[0] += 1

    monkeypatch.setattr(SlabStore, "put", put)

    def poll():
        if count[0] >= puts:
            raise Stop()

    return poll


def _segment(raw, path, outputs, **options):
    return segment_with_classifier(raw, path, outputs, workers=1, **options)


def test_a_stopped_segmentation_is_taken_up_and_its_mask_is_the_same(image, tmp_path, monkeypatch, caplog):
    volume, _truth, raw, path = image
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(3))
    reference = tifffile.imread(_segment(raw, path, {"Vessel": tmp_path / "ref.tif"})["Vessel"])
    out = tmp_path / "out.tif"
    with pytest.raises(Stop):
        _segment(raw, path, {"Vessel": out}, poll=_stopping_after(monkeypatch, 2))
    assert not out.exists()
    kept = SlabStore.__new__(SlabStore)
    kept.folder = slab_folder(out)
    finished = kept.finished_planes()
    assert 0 < finished < volume.shape[0]
    monkeypatch.undo()
    # Taken up with slabs of another size: the mask is the same.
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(4))
    told: list[SegmentProgress] = []
    with caplog.at_level(logging.INFO, logger="haemolynx.classifier.apply"):
        _segment(raw, path, {"Vessel": out}, progress=told.append)
    assert np.array_equal(tifffile.imread(out), reference)
    assert not slab_folder(out).exists()
    assert f"at plane {finished} of {volume.shape[0]}" in caplog.text
    classified = [t for t in told if not t.writing]
    assert classified and all(t.start_plane == finished for t in classified)


def test_another_classifier_or_no_reuse_starts_afresh(image, tmp_path, monkeypatch):
    volume, truth, raw, path = image
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(3))
    out = tmp_path / "out.tif"
    with pytest.raises(Stop):
        _segment(raw, path, {"Vessel": out}, poll=_stopping_after(monkeypatch, 2))
    monkeypatch.undo()
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(3))
    from haemolynx.classifier import FeatureSpec

    other = save_classifier(trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 2.0))),
                            tmp_path / "other.haemoclf")
    told: list[SegmentProgress] = []
    _segment(raw, other, {"Vessel": out}, progress=told.append)
    assert all(t.start_plane == 0 for t in told)
    expected = tifffile.imread(_segment(raw, other, {"Vessel": tmp_path / "other_ref.tif"})["Vessel"])
    assert np.array_equal(tifffile.imread(out), expected)
    # And with reuse off, slabs a stopped run left are not used.
    out2 = tmp_path / "out2.tif"
    with pytest.raises(Stop):
        _segment(raw, path, {"Vessel": out2}, poll=_stopping_after(monkeypatch, 2))
    told.clear()
    monkeypatch.undo()
    _segment(raw, path, {"Vessel": out2}, progress=told.append, reuse_existing=False)
    assert all(t.start_plane == 0 for t in told if not t.writing)


def test_a_stopped_segmentation_on_a_coarser_grid_is_taken_up_the_same(image, tmp_path, monkeypatch):
    _volume, _truth, raw, path = image
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(2))
    reference = tifffile.imread(_segment(raw, path, {"Vessel": tmp_path / "ref.tif"}, downsample="2x")["Vessel"])
    out = tmp_path / "out.tif"
    with pytest.raises(Stop):
        _segment(raw, path, {"Vessel": out}, downsample="2x", poll=_stopping_after(monkeypatch, 2))
    monkeypatch.undo()
    _segment(raw, path, {"Vessel": out}, downsample="2x")
    assert np.array_equal(tifffile.imread(out), reference)


def test_masks_sharing_a_pass_are_taken_up_from_the_plane_they_all_reach(tmp_path, monkeypatch):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    path = save_classifier(trained(volume, truth, "vessels_and_classes"), tmp_path / "all.haemoclf")
    names = {"Vessel": "v.tif", "Arteriole": "a.tif", "Venule": "ven.tif"}
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(3))
    reference = _segment(raw, path, {m: tmp_path / "ref" / n for m, n in names.items()})
    outputs = {m: tmp_path / n for m, n in names.items()}
    with pytest.raises(Stop):
        _segment(raw, path, outputs, poll=_stopping_after(monkeypatch, 3))
    monkeypatch.undo()
    # One mask lost its last slab: every mask is taken up from where it ends.
    venule = sorted(slab_folder(outputs["Venule"]).glob("*.tif"))
    venule[-1].unlink()
    lost_at = int(venule[-1].name.split("-")[0])
    told: list[SegmentProgress] = []
    _segment(raw, path, outputs, progress=told.append)
    assert [t.start_plane for t in told if not t.writing][0] == lost_at
    for mask in names:
        assert np.array_equal(tifffile.imread(outputs[mask]), tifffile.imread(reference[mask])), mask


# --------------------------------------------------------------- the store


def test_the_store_keeps_slabs_only_from_the_first_plane_without_a_gap(tmp_path):
    record = {"classifier_sha256": "a", "label": "Vessel"}
    store = SlabStore(tmp_path / "m.tif", record, keep=True, shape_zyx=(10, 4, 5))
    for z0, z1 in ((0, 3), (3, 6), (8, 10)):
        store.put(z0, z1, np.ones((z1 - z0, 4, 5), np.uint8))
    store.close()
    assert store.slabs() == [(0, 3), (3, 6), (8, 10)]
    assert store.finished_planes() == 6
    store.keep_before(4)
    assert store.slabs() == [(0, 3)] and store.finished_planes() == 3
    # The same record keeps them; another, or keep=False, starts afresh.
    assert SlabStore(tmp_path / "m.tif", record, keep=True, shape_zyx=(10, 4, 5)).finished_planes() == 3
    assert SlabStore(tmp_path / "m.tif", {**record, "label": "Venule"}, keep=True,
                     shape_zyx=(10, 4, 5)).finished_planes() == 0
    store = SlabStore(tmp_path / "n.tif", record, keep=True, shape_zyx=(10, 4, 5))
    store.put(0, 2, np.ones((2, 4, 5), np.uint8))
    store.close()
    assert SlabStore(tmp_path / "n.tif", record, keep=False, shape_zyx=(10, 4, 5)).finished_planes() == 0


def test_a_slab_on_a_coarse_grid_is_kept_at_the_images_size(tmp_path):
    store = SlabStore(tmp_path / "m.tif", {}, keep=True, factors=(2, 2, 2), shape_zyx=(5, 5, 3))
    coarse = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)
    store.put(1, 3, coarse)  # grid planes 1-2: image planes 2-4 (the image ends at 5)
    store.close()
    planes = list(store.planes())
    assert len(planes) == 3 and all(p.shape == (5, 3) for p in planes)
    expected = [plane for _z, plane in full_size_planes(1, coarse, (2, 2, 2), (5, 5, 3))]
    assert all(np.array_equal(a, b) for a, b in zip(planes, expected))
    assert np.array_equal(planes[0][:2, :2], np.full((2, 2), coarse[0, 0, 0]))


# --------------------------------------------------------------- progress


def test_progress_runs_to_the_end_and_then_says_it_is_writing(image, tmp_path, monkeypatch):
    _volume, _truth, raw, path = image
    monkeypatch.setattr(apply_module, "slab_planes", _slabs_of(3))
    monkeypatch.setattr(apply_module, "PROGRESS_SECONDS", 0.0)
    told: list[SegmentProgress] = []
    _segment(raw, path, {"Vessel": tmp_path / "out.tif"}, progress=told.append)
    classified = [t for t in told if not t.writing]
    assert [t.done for t in classified] == sorted(t.done for t in classified)
    assert classified[-1].done == classified[-1].total > 1
    assert classified[0].remaining is None  # the first tile times nothing
    assert all(t.remaining is not None and t.remaining >= 0 for t in classified[1:])
    assert classified[-1].remaining == 0
    assert told[-1].writing


def test_the_time_left_is_the_pace_since_the_first_tile():
    assert remaining_seconds(1, 10, 5.0, 0) is None
    assert remaining_seconds(5, 10, 8.0, 4) == pytest.approx(10.0)
    assert remaining_seconds(10, 10, 8.0, 9) == 0


def test_the_segment_stage_fills_its_step_bar_and_says_the_time_left(image, tmp_path, monkeypatch):
    from haemolynx.gui.progress import ProgressDisplay
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.progress import STEP, RunProgress
    from haemolynx.pipeline.stages import segment_with_run_classifiers

    _volume, _truth, raw, path = image
    monkeypatch.setattr(apply_module, "PROGRESS_SECONDS", 0.0)
    settings = resolve_settings({
        "output_dir": tmp_path / "out", "raw_image_path": raw,
        "use_pixel_classifier_segmentation": True, "pixel_classifier_path": path,
    }, schema=default_schema(), config_path=None)
    events = []
    run = RunProgress(events.append)
    with run.stage("segment") as stage:
        segment_with_run_classifiers(settings, progress=stage)
    steps = [e for e in events if e.kind == STEP]
    assert steps and all(e.stage == "segment" for e in steps)
    assert steps[0].step.startswith("Classifying the raw image, ")
    assert any("left" in e.step for e in steps)
    assert steps[-1].step == "Writing the masks"
    display = ProgressDisplay()
    display.update(steps[-2])
    assert display.steps.total == steps[-2].step_total and display.steps.value == steps[-2].step_total
