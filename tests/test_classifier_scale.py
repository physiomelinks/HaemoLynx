"""The classifier at scale: lazy reading, one pass for many classifiers,
only the features some tree reads, the combined vessel job, sparse labels.
Each must give exactly the answer of the plain path it stands in for."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.classifier import (
    FeatureSpec,
    LazyTiffChannel,
    TrainingImage,
    TrainingVoxels,
    classify_volume,
    features_at,
    intensity_range,
    new_classifier,
    save_classifier,
    segment_with_classifier,
    segment_with_classifiers,
)
from haemolynx.classifier.apply import Job, classify_slabs, forest_features, shared_spec, spec_columns
from haemolynx.classifier.features import box_features, features_of_box, scaled
from haemolynx.classifier.sparse import SparseLabels, painted_points

from test_pixel_classifier import VOXEL, strokes, tube_volume, write_raw

pytest.importorskip("sklearn")


def trained(volume, truth, purpose="vessels", scales=(1.0, 2.0), channels=(None,), split=False):
    classifier = new_classifier(purpose, scales_um=scales, channels=channels)
    ranges = [intensity_range(volume)] * len(channels)
    points, classes = strokes(truth, per_class=200)
    classes = classes.copy()
    if split:  # tubes in the upper half are the next label
        classes[(classes == 1) & (points[:, 1] >= volume.shape[1] // 2)] = 2
    if purpose == "vessels_and_classes":  # Vessel, then Arteriole/Venule by x
        inside = classes > 0
        classes[inside & (points[:, 2] < volume.shape[2] // 3)] = 2
        classes[inside & (points[:, 2] > 2 * volume.shape[2] // 3)] = 3
    image = TrainingImage("t.tif", volume.shape, VOXEL, tuple(ranges))
    feats = features_at([volume] * len(channels), ranges, VOXEL, classifier.spec, points)
    voxels = TrainingVoxels(feats, classes.astype(np.uint8), np.zeros(len(classes), np.int32), points)
    return classifier.with_image_voxels(image, voxels).trained(n_jobs=1)


# ------------------------------------------------------------ lazy reading


@pytest.mark.parametrize("compressed", [False, True])
def test_a_lazily_read_channel_gives_the_files_voxels(tmp_path, compressed):
    volume, _ = tube_volume()
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, volume, imagej=True, compression="zlib" if compressed else None)
    lazy = LazyTiffChannel(path)
    assert lazy.shape == volume.shape and lazy.dtype == volume.dtype
    assert (lazy._mapped is None) == compressed
    for key in [(3,), (slice(2, 9), slice(5, 40), slice(None, None, 3)), (slice(None), 7, slice(1, 30)),
                (slice(-4, None), slice(None), -1), (np.array([1, 4, 2]),)]:
        assert np.array_equal(lazy[key], volume[key]), key
    assert np.array_equal(np.asarray(lazy), volume)
    lazy.close()


def test_a_composites_channel_is_read_lazily_alike(tmp_path):
    volume, _ = tube_volume()
    other = (volume // 2).astype(np.uint16)
    path = tmp_path / "composite.tif"
    tifffile.imwrite(path, np.stack([other, volume], axis=1), imagej=True, metadata={"axes": "ZCYX"})
    for channel, expected in ((0, other), (1, volume)):
        lazy = LazyTiffChannel(path, channel)
        assert np.array_equal(lazy[:, 3:20, 5:9], expected[:, 3:20, 5:9])
        lazy.close()


def test_a_compressed_raw_image_segments_lazily_to_the_same_mask(tmp_path):
    volume, truth = tube_volume()
    path = save_classifier(trained(volume, truth), tmp_path / "v.haemoclf")
    raw = tmp_path / "raw.tif"
    tifffile.imwrite(raw, volume, imagej=True, compression="zlib", resolution=(1 / VOXEL[2], 1 / VOXEL[1]),
                     metadata={"axes": "ZYX", "spacing": VOXEL[0], "unit": "um"})
    in_ram = segment_with_classifier(raw, path, {"Vessel": tmp_path / "a.tif"})
    lazy = segment_with_classifier(raw, path, {"Vessel": tmp_path / "b.tif"}, use_memmap=True,
                                   memmap_directory=tmp_path / "mm")
    assert np.array_equal(tifffile.imread(in_ram["Vessel"]), tifffile.imread(lazy["Vessel"]))


# ------------------------------------------------- only the features read


def test_features_a_forest_never_reads_are_skipped_without_changing_the_rest():
    volume, truth = tube_volume()
    classifier = trained(volume, truth, scales=(1.0, 2.0, 4.0))
    spec = classifier.spec
    used = forest_features(classifier)
    needed = np.zeros(spec.n_features, bool)
    needed[used] = True
    channel = scaled(volume, intensity_range(volume))
    core = tuple(slice(0, n) for n in volume.shape)
    full = box_features([channel], VOXEL, spec, core)
    part = box_features([channel], VOXEL, spec, core, needed=needed)
    assert np.array_equal(part[..., needed], full[..., needed])
    assert not part[..., ~needed].any()
    assert np.array_equal(
        classifier.forest.predict(part.reshape(-1, spec.n_features)),
        classifier.forest.predict(full.reshape(-1, spec.n_features)),
    )


def test_the_slab_pass_labels_every_voxel_as_the_whole_volumes_features_do():
    volume, truth = tube_volume()
    classifier = trained(volume, truth, scales=(1.0, 3.0))
    ranges = [intensity_range(volume)]
    whole = tuple(slice(0, n) for n in volume.shape)
    features = features_of_box([volume], ranges, VOXEL, classifier.spec, whole, whole)
    expected = classifier.forest.predict(features.reshape(-1, classifier.spec.n_features)).reshape(volume.shape)
    for planes in (1, 3, 20):
        got = classify_volume([volume], VOXEL, classifier, workers=3, planes_per_slab=planes, box_voxels=30_000)
        assert np.array_equal(got, expected), planes


# ------------------------------------------------ one pass, many classifiers


def test_one_pass_for_several_classifiers_makes_each_ones_masks(tmp_path):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    vessels = save_classifier(trained(volume, truth, scales=(1.0, 2.0)), tmp_path / "v.haemoclf")
    classes = save_classifier(
        trained(volume, truth, "vessel_classes", scales=(2.0, 4.0), split=True), tmp_path / "av.haemoclf"
    )
    together = segment_with_classifiers(raw, [
        (vessels, {"Vessel": tmp_path / "t_v.tif"}),
        (classes, {"Arteriole": tmp_path / "t_a.tif", "Venule": tmp_path / "t_ven.tif"}),
    ])
    alone_v = segment_with_classifier(raw, vessels, {"Vessel": tmp_path / "a_v.tif"})
    alone_c = segment_with_classifier(raw, classes, {"Arteriole": tmp_path / "a_a.tif", "Venule": tmp_path / "a_ven.tif"})
    assert np.array_equal(tifffile.imread(together[0]["Vessel"]), tifffile.imread(alone_v["Vessel"]))
    for mask in ("Arteriole", "Venule"):
        assert np.array_equal(tifffile.imread(together[1][mask]), tifffile.imread(alone_c[mask]))


def test_the_shared_features_are_each_classifiers_own():
    a = FeatureSpec(scales_um=(1.0, 2.0), channels=(1,))
    b = FeatureSpec(scales_um=(2.0, 8.0), channels=(0, 1))
    shared = shared_spec([a, b])
    assert shared.channels == (0, 1) and shared.scales_um == (1.0, 2.0, 8.0)
    names = shared.names()
    assert [names[i] for i in spec_columns(a, shared)] == a.names()
    with pytest.raises(ValueError, match="one-channel"):
        shared_spec([FeatureSpec(channels=(None,)), b])


def test_a_classifier_asked_twice_walks_once(tmp_path):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    path = save_classifier(trained(volume, truth), tmp_path / "v.haemoclf")
    calls = []
    jobs = [Job(trained(volume, truth), (("x", None),))]
    classify_slabs([volume], [intensity_range(volume)], VOXEL, jobs, sink=lambda *a: calls.append(a[0]))
    assert calls[0] == 0
    out = segment_with_classifiers(raw, [(path, {"Vessel": tmp_path / "1.tif"}), (path, {"Vessel": tmp_path / "2.tif"})])
    assert np.array_equal(tifffile.imread(out[0]["Vessel"]), tifffile.imread(out[1]["Vessel"]))


# ---------------------------------------------------- the combined vessel job


def test_a_combined_classifier_makes_the_vessel_mask_of_every_vessel_label(tmp_path):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    classifier = trained(volume, truth, "vessels_and_classes")
    assert classifier.labels == ("Background", "Vessel", "Arteriole", "Venule")
    path = save_classifier(classifier, tmp_path / "all.haemoclf")
    out = segment_with_classifier(raw, path, {
        "Vessel": tmp_path / "v.tif", "Arteriole": tmp_path / "a.tif", "Venule": tmp_path / "ven.tif",
    })
    labels = classify_volume([volume], VOXEL, classifier, workers=2)
    vessel = tifffile.imread(out["Vessel"]).astype(bool)
    assert np.array_equal(vessel, labels > 0)
    arteriole = tifffile.imread(out["Arteriole"]).astype(bool)
    assert np.array_equal(arteriole, labels == 2) and not (arteriole & ~vessel).any()


def test_preflight_takes_a_combined_classifier_for_the_main_mask_and_the_vessel_masks(tmp_path):
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.checks import check_pixel_classifiers

    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    path = save_classifier(trained(volume, truth, "vessels_and_classes"), tmp_path / "all.haemoclf")
    settings = resolve_settings({
        "output_dir": tmp_path / "out", "raw_image_path": raw, "use_pixel_classifier_segmentation": True,
        "pixel_classifier_path": path, "automated_vessel_assignment": True, "use_large_vessel_masks": True,
        "use_pixel_classifier_large_vessel_segmentation": True, "pixel_classifier_large_vessel_path": path,
    }, schema=default_schema(), config_path=None)
    assert check_pixel_classifiers(settings).errors == []


# --------------------------------------------------- cells and 3D vessels


def test_the_3d_distance_masks_come_from_the_classifier(tmp_path):
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import (
        classifier_requests,
        object_masks_made,
        segment_object_masks,
        segment_with_run_classifiers,
    )

    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    cells = save_classifier(trained(volume, truth, "cells"), tmp_path / "c.haemoclf")
    vessels = save_classifier(trained(volume, truth), tmp_path / "v.haemoclf")
    settings = resolve_settings({
        "output_dir": tmp_path / "out", "raw_image_path": raw, "input_path": raw,
        "measurement_3d_to_cell_mask": True,
        "use_pixel_classifier_cell_segmentation": True, "pixel_classifier_cell_path": cells,
        "use_pixel_classifier_3d_vessel_segmentation": True, "pixel_classifier_3d_vessel_path": vessels,
    }, schema=default_schema(), config_path=None)
    assert object_masks_made(settings) == ("cells", "vessels_3d")
    assert len(classifier_requests(settings)) == 2
    segment_with_run_classifiers(settings)
    made = segment_object_masks(settings)
    assert settings["cell_mask_path"] == made["cells"] == tmp_path / "out" / "raw_cells_segmented.tif"
    assert settings["measurement_3d_vessel_mask_path"] == made["vessels_3d"]
    assert tifffile.imread(made["cells"]).any()


# ------------------------------------------------------------ sparse labels


def test_sparse_labels_read_and_write_as_numpy_does():
    rng = np.random.default_rng(4)
    shape = (13, 300, 270)
    dense = np.zeros(shape, np.uint8)
    sparse = SparseLabels(shape, chunk=(4, 64, 64))
    for _ in range(30):
        points = tuple(rng.integers(0, n, 40) for n in shape)
        value = int(rng.integers(0, 4))
        dense[points] = value
        sparse[points] = value
    box = (slice(2, 11), slice(50, 260), slice(None, None, 7))
    dense[box] = 3
    sparse[box] = 3
    dense[5, 10:20, 30] = 0
    sparse[5, 10:20, 30] = 0
    for key in [(5,), (slice(None), 7), box, (slice(None), slice(None), -1), (4, slice(0, 70), 66)]:
        assert np.array_equal(sparse[key], dense[key]), key
    points = tuple(rng.integers(0, n, 200) for n in shape)
    assert np.array_equal(sparse[points], dense[points])
    got_points, got_values = painted_points(sparse)
    assert np.array_equal(got_points, np.argwhere(dense > 0)) and np.array_equal(got_values, dense[dense > 0])
    assert np.array_equal(np.asarray(sparse), dense)


def test_training_on_sparse_strokes_is_training_on_dense_ones():
    from haemolynx.classifier.preview import LiveClassifier

    volume, truth = tube_volume()
    points, classes = strokes(truth, per_class=100)
    dense = np.zeros(volume.shape, np.uint8)
    dense[tuple(points.T)] = classes + 1
    sparse = SparseLabels(volume.shape)
    sparse[tuple(points.T.astype(np.int64))] = classes + 1
    a = LiveClassifier(new_classifier("vessels", scales_um=(1.0,)), [volume], VOXEL, path="t").train(dense)
    b = LiveClassifier(new_classifier("vessels", scales_um=(1.0,)), [volume], VOXEL, path="t").train(sparse)
    assert np.array_equal(a.training.features, b.training.features)
    assert np.array_equal(a.forest.votes(a.training.features), b.forest.votes(a.training.features))


# --------------------------------------------------------- the coarse grid


@pytest.mark.parametrize("voxel", [(2.0, 0.6, 0.6), (1.0, 1.0, 1.0), (0.5, 0.7, 0.9)])
@pytest.mark.parametrize("core", [(1, 16, 19), (5, 23, 11), (9, 40, 56)])
def test_coarse_features_in_any_box_are_the_whole_volumes_bit_for_bit(voxel, core):
    from haemolynx.classifier.features import boxes, coarse_factors, feature_halo

    volume, _ = tube_volume(shape=(24, 60, 70))
    spec = FeatureSpec(scales_um=(1.0, 4.0, 8.0), coarse=True)
    assert coarse_factors(spec, 8.0, voxel) != (1, 1, 1)
    ranges = [intensity_range(volume)]
    whole = tuple(slice(0, n) for n in volume.shape)
    reference = features_of_box([volume], ranges, voxel, spec, whole, whole)
    for padded, inner, box_core in boxes(volume.shape, feature_halo(spec, voxel), core_zyx=core):
        part = features_of_box([volume], ranges, voxel, spec, padded, inner)
        assert np.array_equal(part, reference[box_core]), box_core


def test_a_coarse_classifier_still_finds_the_tubes_and_reads_unlike_an_old_one():
    from test_pixel_classifier import trained_classifier

    volume, truth = tube_volume(seed=0)
    coarse = trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 2.0, 4.0), coarse=True))
    fresh, fresh_truth = tube_volume(seed=5)
    found = classify_volume([fresh], (2.0, 0.8, 0.8), coarse, workers=2) == 1
    assert 2 * (found & fresh_truth).sum() / (found.sum() + fresh_truth.sum()) > 0.85
    assert FeatureSpec.from_record({"scales_um": [1.0]}).coarse is False  # a classifier saved before
    assert FeatureSpec.from_record(coarse.spec.to_record()) == coarse.spec


def test_an_old_and_a_new_classifier_share_a_run_each_on_its_own_grid(tmp_path):
    from test_pixel_classifier import trained_classifier

    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    old = save_classifier(trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 2.0))), tmp_path / "o.haemoclf")
    new = save_classifier(trained(volume, truth, "pericytes"), tmp_path / "n.haemoclf")
    assert trained(volume, truth, "pericytes").spec.coarse
    with pytest.raises(ValueError, match="coarse grid"):
        shared_spec([FeatureSpec(coarse=True), FeatureSpec()])
    both = segment_with_classifiers(raw, [(old, {"Vessel": tmp_path / "a.tif"}), (new, {"Pericyte": tmp_path / "b.tif"})])
    alone = segment_with_classifier(raw, old, {"Vessel": tmp_path / "c.tif"})
    assert np.array_equal(tifffile.imread(both[0]["Vessel"]), tifffile.imread(alone["Vessel"]))


@pytest.mark.parametrize("tile", [16, 23, 1024])
def test_coarse_scales_shared_by_a_tile_label_every_voxel_as_the_whole_volume_does(monkeypatch, tile):
    from haemolynx.classifier import apply as apply_module
    from test_pixel_classifier import trained_classifier

    voxel = (2.0, 0.6, 0.6)
    volume, truth = tube_volume(shape=(20, 64, 72))
    classifier = trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 4.0, 8.0), coarse=True), voxel=voxel)
    ranges = [intensity_range(volume)]
    whole = tuple(slice(0, n) for n in volume.shape)
    features = features_of_box([volume], ranges, voxel, classifier.spec, whole, whole)
    expected = classifier.forest.predict(features.reshape(-1, classifier.spec.n_features)).reshape(volume.shape)
    monkeypatch.setattr(apply_module, "COARSE_TILE", tile)
    monkeypatch.setattr(apply_module, "MIN_COARSE_TILE", tile)
    got = classify_volume([volume], voxel, classifier, workers=3, planes_per_slab=7, box_voxels=20_000)
    assert np.array_equal(got, expected)


# ------------------------------------------------- processes, coarser grid


@pytest.mark.parametrize("coarse", [False, True])
def test_worker_processes_label_every_voxel_as_threads_do(tmp_path, coarse):
    from test_pixel_classifier import trained_classifier

    voxel = (2.0, 0.6, 0.6)
    volume, truth = tube_volume(shape=(20, 64, 72))
    raw = write_raw(tmp_path / "raw.tif", volume, voxel_zyx=voxel)
    classifier = trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 4.0), coarse=coarse), voxel=voxel)
    threads = classify_volume([volume], voxel, classifier, workers=2, processes=False, planes_per_slab=6)
    lazy = LazyTiffChannel(raw)
    processes = classify_volume([lazy], voxel, classifier, workers=2, processes=True, planes_per_slab=6,
                                box_voxels=30_000)
    lazy.close()
    assert np.array_equal(processes, threads)


def test_a_block_mean_view_reads_numpys_block_means():
    from haemolynx.classifier.lazy import BlockMeanVolume

    volume = np.random.default_rng(0).integers(0, 1000, (7, 11, 13)).astype(np.uint16)
    view = BlockMeanVolume(volume, (2, 3, 3))
    assert view.shape == (4, 4, 5)
    padded = np.full((8, 12, 15), np.nan, np.float64)
    padded[:7, :11, :13] = volume
    expected = np.nanmean(padded.reshape(4, 2, 4, 3, 5, 3), axis=(1, 3, 5))
    np.testing.assert_allclose(view[:, :, :], expected, rtol=1e-6)
    np.testing.assert_allclose(view[2, 1:3, ::2], expected[2, 1:3, ::2], rtol=1e-6)


def test_a_coarser_grid_writes_full_size_masks_of_its_coarse_labels(tmp_path, caplog):
    from haemolynx.classifier.apply import grid_factors
    from haemolynx.classifier.lazy import BlockMeanVolume

    voxel = (2.0, 0.6, 0.6)
    volume, truth = tube_volume(shape=(20, 64, 72))
    raw = write_raw(tmp_path / "raw.tif", volume, voxel_zyx=voxel)
    classifier = trained(volume, truth)
    path = save_classifier(classifier, tmp_path / "c.haemoclf")
    factors = grid_factors("2x", voxel)
    assert factors == (1, 2, 2)  # z already as coarse as 2x makes y and x
    out = segment_with_classifier(raw, path, {"Vessel": tmp_path / "m.tif"}, downsample="2x")["Vessel"]
    mask = tifffile.imread(out)
    assert mask.shape == volume.shape
    coarse = classify_volume([BlockMeanVolume(volume, factors)], tuple(v * f for v, f in zip(voxel, factors)),
                             classifier, workers=1)
    spread = np.repeat(np.repeat(coarse == 1, 2, axis=1), 2, axis=2)[:, :64, :72]
    assert np.array_equal(mask.astype(bool), spread)
    caplog.set_level("INFO")
    segment_with_classifier(raw, path, {"Vessel": tmp_path / "m.tif"}, downsample="2x")
    assert "Reusing the segmentation" in caplog.text
    caplog.clear()
    segment_with_classifier(raw, path, {"Vessel": tmp_path / "m.tif"})
    assert "another grid" in caplog.text


def test_the_run_hands_the_classifier_its_grid(tmp_path):
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import classifier_run_arguments

    settings = resolve_settings({"output_dir": tmp_path, "raw_image_path": tmp_path / "r.tif",
                                 "pixel_classifier_downsample": "4x"}, schema=default_schema(), config_path=None)
    assert classifier_run_arguments(settings)["downsample"] == "4x"
