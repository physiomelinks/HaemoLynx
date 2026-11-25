"""An image unlike every image a classifier was trained on
(``classifier.familiar``): judged on the spread of its intensities on its own
range, which brightness does not change; warned about by a segmentation, by
preflight and by the training window, never refused."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from haemolynx.classifier import (
    intensity_range,
    load_classifier,
    save_classifier,
    segment_with_classifier,
)
from haemolynx.classifier.familiar import (
    UNFAMILIAR_DISTANCE,
    raw_image_profiles,
    unfamiliar_channels,
    unfamiliar_channels_in_record,
    unfamiliar_note,
)
from haemolynx.classifier.features import PROFILE_PERCENTILES, intensity_range_and_profile
from haemolynx.classifier.model import TrainingImage, read_classifier_record

from test_pixel_classifier import VOXEL, strokes, trained_classifier, tube_volume, write_raw

pytest.importorskip("sklearn")


def test_the_range_read_with_the_profile_is_the_range_exactly():
    for seed in (0, 1, 2):
        volume, _ = tube_volume(seed=seed)
        value_range, profile = intensity_range_and_profile(volume)
        assert value_range == intensity_range(volume)
        assert len(profile) == len(PROFILE_PERCENTILES) and list(profile) == sorted(profile)


def test_a_brighter_or_dimmer_image_has_the_same_profile():
    volume, _ = tube_volume()
    _range, profile = intensity_range_and_profile(volume)
    for scale, offset in ((3.0, 0), (0.5, 100), (2.0, 5000)):
        other = np.clip(volume.astype(np.float64) * scale + offset, 0, 65535).astype(np.uint16)
        assert max(abs(a - b) for a, b in zip(intensity_range_and_profile(other)[1], profile)) < 0.01


def _trained_with_profile(volume, truth):
    """A classifier trained as the window trains one: the image's profile kept."""
    from haemolynx.classifier.preview import LiveClassifier

    base = trained_classifier(volume, truth)
    live = LiveClassifier(base, [volume], VOXEL, path="tubes.tif", workers=1)
    points, classes = strokes(truth, per_class=100)
    labels = np.zeros(volume.shape, np.uint8)
    labels[tuple(points.T)] = classes + 1
    return live.train(labels)


def test_the_training_image_keeps_its_profile_through_the_file(tmp_path):
    volume, truth = tube_volume()
    classifier = _trained_with_profile(volume, truth)
    image = classifier.images[classifier.image_index("tubes.tif")]
    assert image.profile == (intensity_range_and_profile(volume)[1],)
    loaded = load_classifier(save_classifier(classifier, tmp_path / "c.haemoclf"))
    assert loaded.images == classifier.images
    # A record from before profiles were kept reads with none.
    old = image.to_record()
    old.pop("profile")
    assert TrainingImage.from_record(old).profile == ()


def test_an_inverted_image_is_unfamiliar_and_a_brighter_one_is_not():
    volume, truth = tube_volume()
    classifier = _trained_with_profile(volume, truth)
    brighter = np.clip(volume.astype(np.float64) * 2.5, 0, 65535).astype(np.uint16)
    inverted = (volume.max() - volume).astype(np.uint16)
    assert unfamiliar_channels(classifier, {None: intensity_range_and_profile(brighter)[1]}) == []
    found = unfamiliar_channels(classifier, {None: intensity_range_and_profile(inverted)[1]})
    assert len(found) == 1 and found[0][0] is None and found[0][1] > UNFAMILIAR_DISTANCE
    note = unfamiliar_note("inverted.tif", "c.haemoclf", found)
    assert "inverted.tif" in note and "c.haemoclf" in note and "its channel" in note


def test_a_classifier_saved_before_profiles_judges_nothing():
    volume, truth = tube_volume()
    classifier = trained_classifier(volume, truth)  # its image recorded without a profile
    inverted = (volume.max() - volume).astype(np.uint16)
    assert unfamiliar_channels(classifier, {None: intensity_range_and_profile(inverted)[1]}) == []


def test_a_segmentation_warns_of_an_unfamiliar_image_and_still_segments(tmp_path, caplog):
    volume, truth = tube_volume()
    path = save_classifier(_trained_with_profile(volume, truth), tmp_path / "c.haemoclf")
    own = write_raw(tmp_path / "own.tif", volume)
    other = write_raw(tmp_path / "inverted.tif", (volume.max() - volume).astype(np.uint16))
    with caplog.at_level(logging.WARNING, logger="haemolynx.classifier.apply"):
        segment_with_classifier(own, path, {"Vessel": tmp_path / "own_seg.tif"})
    assert "unlike every image" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="haemolynx.classifier.apply"):
        try:
            segment_with_classifier(other, path, {"Vessel": tmp_path / "other_seg.tif"})
        except ValueError:
            pass  # an inverted image may well have no vessel left: still warned first
    assert "inverted.tif is unlike every image c.haemoclf was trained on" in caplog.text


def test_preflight_warns_of_an_unfamiliar_raw_image(tmp_path):
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.checks import check_pixel_classifiers

    volume, truth = tube_volume()
    path = save_classifier(_trained_with_profile(volume, truth), tmp_path / "c.haemoclf")

    def report(raw):
        settings = resolve_settings({
            "output_dir": tmp_path / "out", "raw_image_path": raw,
            "use_pixel_classifier_segmentation": True, "pixel_classifier_path": path,
        }, schema=default_schema(), config_path=None)
        return check_pixel_classifiers(settings)

    own = report(write_raw(tmp_path / "own.tif", volume))
    assert own.errors == [] and not any("unlike" in w for w in own.warnings)
    other = report(write_raw(tmp_path / "inverted.tif", (volume.max() - volume).astype(np.uint16)))
    assert other.errors == []
    assert any("pixel_classifier_path" in w and "inverted.tif is unlike" in w for w in other.warnings)
    # Judged from the file's record alone, as preflight reads it.
    record = read_classifier_record(path)
    assert unfamiliar_channels_in_record(record, raw_image_profiles(tmp_path / "inverted.tif", (None,)))
