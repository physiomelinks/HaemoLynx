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
