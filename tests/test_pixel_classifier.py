"""The pixel classifier: its features, forest, file and segmentation.

What must hold whatever the machine does with it: a voxel's features are the
same computed in any box as over the whole volume; the forest's own walk
votes as scikit-learn's does; the compiled walk and the NumPy walk give the
same labels; and a segmentation is the same bit for bit in RAM or low-RAM,
with one worker or several, in boxes of any size.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from haemolynx.classifier import (
    FeatureSpec,
    Forest,
    TrainingImage,
    TrainingVoxels,
    boxes,
    classifier_output_is_current,
    classifier_provenance,
    classify_volume,
    feature_halo,
    features_at,
    intensity_range,
    load_classifier,
    new_classifier,
    read_classifier_record,
    save_classifier,
    segment_with_classifier,
)
from haemolynx.classifier import forest as forest_module
from haemolynx.classifier.features import features_of_box, symmetric_eigenvalues

sklearn = pytest.importorskip("sklearn")


def tube_volume(shape=(20, 48, 56), seed=0, n=6):
    """Bright tubes of known position on a noisy dark background."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in shape), indexing="ij", sparse=True)
    truth = np.zeros(shape, bool)
    for _ in range(n):
        p = rng.uniform(0.2, 0.8, 3) * np.array(shape)
        d = rng.normal(size=3)
        d[0] *= 0.2
        d /= np.linalg.norm(d)
        r = rng.uniform(1.5, 3.5)
        vz, vy, vx = z - p[0], y - p[1], x - p[2]
        t = vz * d[0] + vy * d[1] + vx * d[2]
        truth |= (vz**2 + vy**2 + vx**2 - t**2) < r * r
    image = ndi.gaussian_filter(truth.astype(np.float32), 0.8) * 3000 + 200
    image += rng.normal(0, 150, shape)
    return np.clip(image, 0, 65535).astype(np.uint16), truth


def strokes(truth, seed=0, per_class=400):
    """Labelled voxels a person might paint: inside tubes, and well away."""
    rng = np.random.default_rng(seed)
    inside = np.argwhere(ndi.binary_erosion(truth))
    outside = np.argwhere(~ndi.binary_dilation(truth, iterations=3))
    pick_in = inside[rng.choice(len(inside), min(per_class, len(inside)), replace=False)]
    pick_out = outside[rng.choice(len(outside), per_class, replace=False)]
    points = np.concatenate([pick_out, pick_in]).astype(np.int32)
    classes = np.concatenate([np.zeros(len(pick_out)), np.ones(len(pick_in))]).astype(np.uint8)
    return points, classes


VOXEL = (2.0, 0.8, 0.8)


def trained_classifier(volume, truth, spec=None, voxel=VOXEL):
    spec = spec or FeatureSpec(scales_um=(1.0, 2.0, 4.0))
    classifier = new_classifier("vessels", scales_um=spec.scales_um, channels=spec.channels)
    ranges = [intensity_range(volume)]
    points, classes = strokes(truth)
    features = features_at([volume], ranges, voxel, classifier.spec, points)
    image = TrainingImage(path="tubes.tif", shape_zyx=volume.shape, voxel_size_zyx=voxel, ranges=tuple(ranges))
    voxels = TrainingVoxels(features, classes, np.zeros(len(classes), np.int32), points)
    return classifier.with_image_voxels(image, voxels).trained(n_jobs=1)


def write_raw(path, volume, voxel_zyx=VOXEL, *, imagej=True):
    z, y, x = voxel_zyx
    if imagej:
        tifffile.imwrite(path, volume, imagej=True, resolution=(1 / x, 1 / y),
                         metadata={"axes": "ZYX", "spacing": z, "unit": "um"})
    else:
        tifffile.imwrite(path, volume, ome=True,
                         metadata={"axes": "ZYX", "PhysicalSizeX": x, "PhysicalSizeY": y, "PhysicalSizeZ": z})
    return path


# ---------------------------------------------------------------- features


def test_a_spec_names_each_feature_it_computes():
    spec = FeatureSpec(scales_um=(4.0, 1.0), channels=(0, 2))
    assert spec.scales_um == (1.0, 4.0)
    assert spec.n_features == 2 * (1 + 6 * 2) == len(spec.names())
    assert spec.names()[0] == "C1 intensity" and spec.names()[-1] == "C3 hessian_3 4 um"
    assert FeatureSpec.from_record(spec.to_record()) == spec


@pytest.mark.parametrize("bad", [dict(scales_um=()), dict(scales_um=(1.0, 1.0)), dict(scales_um=(-1.0,)),
                                 dict(channels=()), dict(channels=(None, 1)), dict(channels=(1, 1))])
def test_a_spec_refuses_what_it_cannot_compute(bad):
    with pytest.raises(ValueError):
        FeatureSpec(**bad)


def test_the_closed_form_eigenvalues_are_numpys():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 5000))
    mine = np.stack(symmetric_eigenvalues(*a), -1)
    matrices = np.array([[a[0], a[3], a[4]], [a[3], a[1], a[5]], [a[4], a[5], a[2]]]).transpose(2, 0, 1)
    np.testing.assert_allclose(mine, np.linalg.eigvalsh(matrices)[:, ::-1], atol=1e-10)


@pytest.mark.parametrize("core", [(1, 16, 19), (4, 13, 17), (7, 48, 3)])
@pytest.mark.parametrize("voxel", [(2.0, 0.8, 0.8), (1.0, 1.0, 1.0), (0.4, 1.3, 0.7)])
def test_features_in_any_box_are_the_whole_volumes_bit_for_bit(core, voxel):
    volume, _ = tube_volume()
    spec = FeatureSpec(scales_um=(0.7, 2.0, 3.5))
    ranges = [intensity_range(volume)]
    whole = tuple(slice(0, n) for n in volume.shape)
    reference = features_of_box([volume], ranges, voxel, spec, whole, whole)
    for padded, inner, box_core in boxes(volume.shape, feature_halo(spec, voxel), core_zyx=core):
        part = features_of_box([volume], ranges, voxel, spec, padded, inner)
        assert np.array_equal(part, reference[box_core]), box_core


def test_a_single_plane_image_has_features_too():
    volume, _ = tube_volume(shape=(1, 40, 40))
    spec = FeatureSpec(scales_um=(1.0, 2.0))
    whole = tuple(slice(0, n) for n in volume.shape)
    features = features_of_box([volume], [intensity_range(volume)], (1.0, 1.0, 1.0), spec, whole, whole)
    assert features.shape == (1, 40, 40, spec.n_features)
    assert np.isfinite(features).all()


def test_features_at_points_are_the_volumes_features_there():
    volume, truth = tube_volume()
    spec = FeatureSpec(scales_um=(1.0, 3.0))
    ranges = [intensity_range(volume)]
    whole = tuple(slice(0, n) for n in volume.shape)
    reference = features_of_box([volume], ranges, VOXEL, spec, whole, whole)
    points, _ = strokes(truth, per_class=50)
    got = features_at([volume], ranges, VOXEL, spec, points, box_voxels=20_000)
    assert np.array_equal(got, reference[points[:, 0], points[:, 1], points[:, 2]])


def test_the_intensity_range_reads_the_same_lattice_every_time():
    volume, _ = tube_volume()
    low, high = intensity_range(volume)
    assert (low, high) == intensity_range(volume.copy())
    assert 0 <= low < high <= volume.max()
    flat = np.full((4, 8, 8), 7, np.uint16)
    assert intensity_range(flat) == (7.0, 8.0)


# ------------------------------------------------------------------ forest


def fitted(seed=0, classes=3, n_features=9):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, n_features)).astype(np.float32)
    y = ((X[:, 0] + X[:, 3] * X[:, 5] > 0).astype(int) + (X[:, 2] > 1)) % classes
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(12, max_depth=9, random_state=0).fit(X, y)
    return model, Forest.from_sklearn(model, classes), rng


def test_the_forests_walk_votes_as_scikit_learn_does():
    model, forest, rng = fitted()
    X = rng.normal(size=(20000, 9)).astype(np.float32)
    np.testing.assert_allclose(forest.predict_proba(X), model.predict_proba(X), atol=1e-6)
    assert np.array_equal(forest.predict(X), model.predict(X).astype(np.uint8))


def test_features_at_a_threshold_go_the_way_scikit_learns_float64_comparison_does():
    model, forest, _ = fitted()
    # Every threshold, and the float32 values either side of it.
    t = forest.threshold[np.isfinite(forest.threshold)]
    values = np.concatenate([t.astype(np.float32), np.nextafter(t.astype(np.float32), np.float32(np.inf)),
                             np.nextafter(t.astype(np.float32), np.float32(-np.inf))])
    X = np.tile(values[:, None], (1, 9)).astype(np.float32)
    assert np.array_equal(forest.predict(X), model.predict(X).astype(np.uint8))


def test_the_compiled_walk_and_the_numpy_walk_give_the_same_labels(monkeypatch):
    pytest.importorskip("numba")
    _model, forest, rng = fitted(classes=3)
    X = rng.normal(size=(30000, 9)).astype(np.float32)
    compiled = forest.predict(X)
    monkeypatch.setenv(forest_module.DISABLE_NUMBA_ENVIRONMENT, "1")
    assert np.array_equal(compiled, forest.predict(X))
    assert np.array_equal(forest.predict(X[:7]), compiled[:7])
    assert np.array_equal(forest.votes(X, chunk=1000), forest.votes(X, chunk=30000))


def test_a_class_never_trained_gets_no_votes():
    model, forest, rng = fitted(classes=2)
    wider = Forest.from_sklearn(model, 3)
    X = rng.normal(size=(500, 9)).astype(np.float32)
    assert (wider.predict_proba(X)[:, 2] == 0).all()
    assert np.array_equal(wider.predict(X), forest.predict(X))


def test_a_forest_pointing_past_its_nodes_is_refused():
    _model, forest, _ = fitted()
    arrays = forest.arrays()
    arrays["forest_left"] = arrays["forest_left"].copy()
    arrays["forest_left"][0] = len(arrays["forest_left"]) + 5
    with pytest.raises(ValueError, match="past its nodes"):
        Forest.from_arrays(arrays)


# ------------------------------------------------------------------- model


def test_a_saved_classifier_reads_back_and_votes_alike(tmp_path):
    volume, truth = tube_volume()
    classifier = trained_classifier(volume, truth)
    path = save_classifier(classifier, tmp_path / "vessels.haemoclf")
    assert path.exists() and not path.with_name(path.name + ".part").exists()
    loaded = load_classifier(path)
    assert loaded.purpose == "vessels" and loaded.spec == classifier.spec
    assert loaded.voxels_per_label() == classifier.voxels_per_label()
    assert np.array_equal(loaded.training.features, classifier.training.features)
    X = classifier.training.features
    assert np.array_equal(loaded.forest.predict(X), classifier.forest.predict(X))
    record = read_classifier_record(path)
    assert record["labels"] == ["Background", "Vessel"] and record["images"][0]["path"] == "tubes.tif"


def test_training_the_same_voxels_twice_gives_the_same_file_contents(tmp_path):
    volume, truth = tube_volume()
    first = trained_classifier(volume, truth)
    second = trained_classifier(volume, truth)
    X = first.training.features
    assert np.array_equal(first.forest.votes(X), second.forest.votes(X))


def test_a_file_that_is_not_a_classifier_is_refused(tmp_path):
    other = tmp_path / "other.haemoclf"
    np.savez(open(other, "wb"), something=np.arange(3))
    with pytest.raises(ValueError, match="not a HaemoLynx classifier"):
        load_classifier(other)
    pickled = tmp_path / "pickled.haemoclf"
    np.savez(open(pickled, "wb"), record=np.array([{"kind": "x"}], dtype=object))
    with pytest.raises(ValueError):
        load_classifier(pickled)


def test_a_classifier_needs_two_labels_painted():
    classifier = new_classifier("vessels")
    voxels = TrainingVoxels(
        np.zeros((5, classifier.spec.n_features), np.float32), np.zeros(5, np.uint8),
        np.zeros(5, np.int32), np.zeros((5, 3), np.int32),
    )
    image = TrainingImage("a.tif", (1, 2, 3), (1.0, 1.0, 1.0), ((0.0, 1.0),))
    with pytest.raises(ValueError, match="at least two labels"):
        classifier.with_image_voxels(image, voxels).trained()


def test_an_images_strokes_replace_its_own_and_keep_anothers():
    classifier = new_classifier("vessel_classes", scales_um=(1.0,))
    n = classifier.spec.n_features

    def voxels(count, label):
        return TrainingVoxels(np.full((count, n), label, np.float32), np.full(count, label, np.uint8),
                              np.zeros(count, np.int32), np.zeros((count, 3), np.int32))

    a = TrainingImage("a.tif", (1, 2, 3), (1.0, 1.0, 1.0), ((0.0, 1.0),))
    b = TrainingImage("b.tif", (1, 2, 3), (1.0, 1.0, 1.0), ((0.0, 1.0),))
    both = classifier.with_image_voxels(a, voxels(3, 1)).with_image_voxels(b, voxels(4, 2))
    again = both.with_image_voxels(a, voxels(2, 0))
    assert again.voxels_per_label() == {"Background": 2, "Arteriole": 0, "Venule": 4}
    assert [image.path for image in again.images] == ["a.tif", "b.tif"]
    assert again.trained_labels() == ["Background", "Venule"]


# ------------------------------------------------------------ segmentation


def test_a_trained_classifier_finds_the_tubes_of_a_fresh_image():
    volume, truth = tube_volume(seed=0)
    classifier = trained_classifier(volume, truth)
    fresh, fresh_truth = tube_volume(seed=5)
    labels = classify_volume([fresh], VOXEL, classifier, workers=1)
    found = labels == 1
    dice = 2 * (found & fresh_truth).sum() / (found.sum() + fresh_truth.sum())
    assert dice > 0.85


def test_a_segmentation_is_the_same_with_any_workers_and_box_size():
    volume, truth = tube_volume()
    classifier = trained_classifier(volume, truth)
    one = classify_volume([volume], VOXEL, classifier, workers=1)
    many = classify_volume([volume], VOXEL, classifier, workers=4, box_voxels=8_000)
    assert np.array_equal(one, many)


def test_boxes_never_get_thinner_than_twice_their_margin():
    tiles = list(boxes((40, 300, 300), (8, 17, 17), box_voxels=1_000))
    steps = [sorted({core[a].start for _p, _i, core in tiles}) for a in range(3)]
    assert steps[0][1] - steps[0][0] >= 16 and steps[1][1] - steps[1][0] >= 34
    covered = np.zeros((40, 300, 300), np.int8)
    for _padded, _inner, core in tiles:
        covered[core] += 1
    assert (covered == 1).all()


def test_the_raw_image_is_segmented_into_a_mask_with_its_record(tmp_path, caplog):
    volume, truth = tube_volume()
    classifier_path = save_classifier(trained_classifier(volume, truth), tmp_path / "v.haemoclf")
    raw = write_raw(tmp_path / "raw.tif", volume)
    output = tmp_path / "out" / "raw_segmented.tif"
    got = segment_with_classifier(raw, classifier_path, {"Vessel": output}, workers=2)
    assert got == {"Vessel": output}
    with tifffile.TiffFile(output) as tif:
        mask = tif.asarray()
        assert tif.imagej_metadata["spacing"] == pytest.approx(VOXEL[0])
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 1}
    assert mask.shape == volume.shape
    record = json.loads(output.with_name(output.name + ".haemolynx.json").read_text())
    assert record["label"] == "Vessel" and record["output"]["vessel_voxels"] == int(mask.sum())
    stamp = output.stat().st_mtime_ns
    caplog.set_level("INFO")
    segment_with_classifier(raw, classifier_path, {"Vessel": output})
    assert output.stat().st_mtime_ns == stamp and "Reusing the classifier's segmentation" in caplog.text


def test_another_classifier_segments_again(tmp_path):
    volume, truth = tube_volume()
    first = save_classifier(trained_classifier(volume, truth), tmp_path / "a.haemoclf")
    second = save_classifier(
        trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 2.0))), tmp_path / "b.haemoclf"
    )
    raw = write_raw(tmp_path / "raw.tif", volume)
    output = tmp_path / "raw_segmented.tif"
    segment_with_classifier(raw, first, {"Vessel": output})
    channels = load_classifier(second).spec.channels
    provenance = classifier_provenance(raw, second, "Vessel", channels=channels, voxel_size_zyx=VOXEL,
                                       axis_order="zyx")
    current, reason = classifier_output_is_current(output, provenance)
    assert not current and "not the one" in reason


def test_low_ram_segments_the_same_mask_bit_for_bit(tmp_path):
    volume, truth = tube_volume()
    classifier_path = save_classifier(trained_classifier(volume, truth), tmp_path / "v.haemoclf")
    raw = write_raw(tmp_path / "raw.tif", volume)
    in_ram = segment_with_classifier(raw, classifier_path, {"Vessel": tmp_path / "a.tif"}, reuse_existing=False)
    low_ram = segment_with_classifier(
        raw, classifier_path, {"Vessel": tmp_path / "b.tif"}, reuse_existing=False,
        use_memmap=True, memmap_directory=tmp_path / "scratch", workers=3,
    )
    assert np.array_equal(tifffile.imread(in_ram["Vessel"]), tifffile.imread(low_ram["Vessel"]))
    assert not list((tmp_path / "scratch").glob("*.memmap"))


def test_a_raw_image_stored_x_first_gets_its_mask_stored_x_first(tmp_path):
    volume, truth = tube_volume()
    classifier_path = save_classifier(trained_classifier(volume, truth), tmp_path / "v.haemoclf")
    stored = np.ascontiguousarray(np.transpose(volume, (2, 1, 0)))
    raw = write_raw(tmp_path / "raw.tif", stored, voxel_zyx=(VOXEL[2], VOXEL[1], VOXEL[0]))
    zyx = write_raw(tmp_path / "zyx.tif", volume)
    a = segment_with_classifier(raw, classifier_path, {"Vessel": tmp_path / "a.tif"}, axis_order="xyz")
    b = segment_with_classifier(zyx, classifier_path, {"Vessel": tmp_path / "b.tif"})
    assert np.array_equal(np.transpose(tifffile.imread(a["Vessel"]), (2, 1, 0)), tifffile.imread(b["Vessel"]))


def test_a_composites_channels_are_read_by_number(tmp_path):
    volume, truth = tube_volume()
    noise = np.random.default_rng(9).integers(0, 400, volume.shape).astype(np.uint16)
    composite = np.stack([noise, volume], axis=1)  # Z C Y X
    raw = tmp_path / "composite.tif"
    tifffile.imwrite(raw, composite, imagej=True, resolution=(1 / VOXEL[2], 1 / VOXEL[1]),
                     metadata={"axes": "ZCYX", "spacing": VOXEL[0], "unit": "um"})
    classifier = trained_classifier(volume, truth, spec=FeatureSpec(scales_um=(1.0, 2.0, 4.0), channels=(1,)))
    path = save_classifier(classifier, tmp_path / "v.haemoclf")
    plain = save_classifier(trained_classifier(volume, truth), tmp_path / "p.haemoclf")
    single = write_raw(tmp_path / "single.tif", volume)
    a = segment_with_classifier(raw, path, {"Vessel": tmp_path / "a.tif"})
    b = segment_with_classifier(single, plain, {"Vessel": tmp_path / "b.tif"})
    assert np.array_equal(tifffile.imread(a["Vessel"]), tifffile.imread(b["Vessel"]))
    low = segment_with_classifier(raw, path, {"Vessel": tmp_path / "c.tif"}, use_memmap=True,
                                  memmap_directory=tmp_path / "scratch")
    assert np.array_equal(tifffile.imread(a["Vessel"]), tifffile.imread(low["Vessel"]))


# ----------------------------------------------------------- live preview


def painted(truth, seed=0):
    points, classes = strokes(truth, seed=seed, per_class=150)
    labels = np.zeros(truth.shape, np.uint8)
    labels[tuple(points.T)] = classes + 1
    return labels


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_planes_features_are_the_whole_volumes(axis):
    from haemolynx.classifier.preview import LiveClassifier

    volume, _ = tube_volume()
    spec = FeatureSpec(scales_um=(1.0, 2.5))
    live = LiveClassifier(new_classifier("vessels", scales_um=spec.scales_um), [volume], VOXEL, path="t.tif",
                          workers=2)
    whole = tuple(slice(0, n) for n in volume.shape)
    reference = features_of_box([volume], list(live.ranges), VOXEL, spec, whole, whole)
    for index in (0, 5, volume.shape[axis] - 1):
        assert np.array_equal(live.plane_features(axis, index), np.take(reference, index, axis=axis))


def test_the_live_preview_shows_what_segmenting_the_whole_image_gives_on_that_plane():
    from haemolynx.classifier.preview import LiveClassifier

    volume, truth = tube_volume()
    live = LiveClassifier(new_classifier("vessels", scales_um=(1.0, 2.0, 4.0)), [volume], VOXEL, path="t.tif",
                          workers=3)
    live.plane_features(0, 10)  # some strokes are read off a cached plane
    classifier = live.train(painted(truth))
    whole = classify_volume([volume], VOXEL, classifier, workers=2)
    for axis, index in ((0, 10), (1, 20), (2, 3)):
        assert np.array_equal(live.predict_plane(axis, index), np.take(whole, index, axis=axis))


def test_painting_trains_on_the_same_features_as_listing_the_strokes():
    from haemolynx.classifier.preview import LiveClassifier

    volume, truth = tube_volume()
    labels = painted(truth)
    live = LiveClassifier(new_classifier("vessels", scales_um=(1.0, 2.0)), [volume], VOXEL, path="t.tif")
    classifier = live.train(labels)
    points = np.argwhere(labels > 0)
    reference = features_at([volume], list(live.ranges), VOXEL, classifier.spec, points)
    assert np.array_equal(classifier.training.features, reference)
    assert np.array_equal(classifier.training.points_zyx, points)
    assert np.array_equal(live.painted_labels(), labels)


def test_the_plane_cache_keeps_within_its_budget():
    from haemolynx.classifier.preview import LiveClassifier

    volume, _ = tube_volume()
    spec = FeatureSpec(scales_um=(1.0,))
    per_plane = volume.shape[1] * volume.shape[2] * spec.n_features * 4
    live = LiveClassifier(new_classifier("vessels", scales_um=(1.0,)), [volume], VOXEL, path="t.tif",
                          cache_bytes=3 * per_plane)
    for index in range(volume.shape[0]):
        live.plane_features(0, index)
        assert len(live._planes) * per_plane <= 3 * per_plane


def test_a_label_the_classifier_never_saw_is_refused(tmp_path):
    volume, truth = tube_volume()
    classifier = new_classifier("vessel_classes", scales_um=(1.0, 2.0))
    ranges = [intensity_range(volume)]
    points, classes = strokes(truth)
    image = TrainingImage("t.tif", volume.shape, VOXEL, tuple(ranges))
    voxels = TrainingVoxels(features_at([volume], ranges, VOXEL, classifier.spec, points), classes,
                            np.zeros(len(classes), np.int32), points)
    path = save_classifier(classifier.with_image_voxels(image, voxels).trained(n_jobs=1), tmp_path / "c.haemoclf")
    raw = write_raw(tmp_path / "raw.tif", volume)
    with pytest.raises(ValueError, match="never shown any Venule"):
        segment_with_classifier(raw, path, {"Arteriole": tmp_path / "a.tif", "Venule": tmp_path / "v.tif"})
    assert not (tmp_path / "a.tif").exists()
