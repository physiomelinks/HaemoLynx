"""The HaemoLynx classifier in a run: the Segment stage, the vessel masks,
the pericyte and block masks, preflight, batches, names and citations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.classifier import (
    TrainingImage,
    TrainingVoxels,
    features_at,
    intensity_range,
    new_classifier,
    save_classifier,
    segment_with_classifier,
)
from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.naming import (
    classifier_segmented_path,
    classifier_segments,
    run_input_path,
    run_stem,
    segments_raw_image,
)
from haemolynx.pipeline.stages import (
    current_ilastik_output,
    object_masks_made,
    segment,
    segment_object_masks,
)

from test_pixel_classifier import VOXEL, strokes, tube_volume, write_raw

pytest.importorskip("sklearn")

SCHEMA = default_schema()


def settings_for(tmp_path: Path, **overrides) -> dict:
    values = {"output_dir": tmp_path / "out", **overrides}
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def classifier_file(path: Path, volume, truth, purpose="vessels", *, labels=None) -> Path:
    """A classifier for *purpose* trained on *volume*'s tubes: background
    and each label after it, the tubes split between the labels by y."""
    classifier = new_classifier(purpose, scales_um=(1.0, 2.0))
    ranges = [intensity_range(volume)]
    points, classes = strokes(truth, per_class=200)
    names = labels or classifier.labels[1:]
    if len(names) > 1:
        inside = classes == 1
        upper = points[:, 1] >= volume.shape[1] // 2
        classes = classes.copy()
        classes[inside & upper] = 2
    image = TrainingImage("t.tif", volume.shape, VOXEL, tuple(ranges))
    voxels = TrainingVoxels(
        features_at([volume], ranges, VOXEL, classifier.spec, points), classes.astype(np.uint8),
        np.zeros(len(classes), np.int32), points,
    )
    return save_classifier(classifier.with_image_voxels(image, voxels).trained(n_jobs=1), path)


@pytest.fixture
def image(tmp_path):
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    return volume, truth, raw


# ------------------------------------------------------------ main mask


def test_the_segment_stage_segments_the_raw_image_with_the_classifier(tmp_path, image):
    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "v.haemoclf", volume, truth)
    settings = settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=classifier
    )
    expected = tmp_path / "out" / "raw_segmented.tif"
    assert run_input_path(settings) == expected and run_stem(settings) == "raw_segmented"
    assert current_ilastik_output(settings)[0] is None
    inputs = segment(settings)
    assert Path(inputs.image_path) == expected
    reference = segment_with_classifier(raw, classifier, {"Vessel": tmp_path / "ref.tif"})["Vessel"]
    assert np.array_equal(tifffile.imread(expected), tifffile.imread(reference))
    found, why = current_ilastik_output(settings)
    assert Path(found) == expected, why


def test_low_ram_runs_segment_the_same_mask(tmp_path, image):
    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "v.haemoclf", volume, truth)
    common = dict(use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=classifier)
    in_ram = segment(settings_for(tmp_path, **common))
    low = segment(resolve_settings(
        {"output_dir": tmp_path / "low", "use_memmap_loading": True, "memmap_directory": tmp_path / "mm", **common},
        schema=SCHEMA, config_path=None,
    ))
    assert np.array_equal(tifffile.imread(in_ram.image_path), tifffile.imread(low.image_path))


def test_without_a_raw_image_the_segmented_image_is_read(tmp_path, caplog):
    mask = tmp_path / "mask.tif"
    tifffile.imwrite(mask, np.ones((3, 4, 4), np.uint8))
    settings = settings_for(tmp_path, use_pixel_classifier_segmentation=True, input_path=mask)
    assert not classifier_segments(settings) and not segments_raw_image(settings)
    assert Path(segment(settings).image_path) == mask
    assert "classifier is skipped" in caplog.text


def test_ilastik_takes_precedence_over_the_classifier(tmp_path, image):
    _volume, _truth, raw = image
    settings = settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, use_ilastik_segmentation=True, raw_image_path=raw,
    )
    assert not classifier_segments(settings) and segments_raw_image(settings)
    assert run_input_path(settings).parent != tmp_path / "out"


def test_the_classifier_output_is_named_after_the_raw_image(tmp_path):
    settings = {"raw_image_path": tmp_path / "StackA.tif", "output_dir": tmp_path / "o"}
    assert classifier_segmented_path(settings) == tmp_path / "o" / "StackA_segmented.tif"


# ---------------------------------------------------------- vessel masks


def _load_masks(tmp_path, raw, classifier, volume):
    from haemolynx.io.vessel_masks import load_and_validate_vessel_masks

    return load_and_validate_vessel_masks(
        mask_role="large", enabled=True, use_ilastik=False, arteriole_mask_path=None, venule_mask_path=None,
        image_shape=volume.shape, main_voxel_size_xyz=(VOXEL[2], VOXEL[1], VOXEL[0]), raw_image_path=raw,
        use_classifier=True, classifier_path=classifier, output_dir=tmp_path / "masks",
    )


def test_one_classifier_makes_the_arteriole_and_venule_masks(tmp_path, image):
    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "av.haemoclf", volume, truth, "vessel_classes")
    arteriole, venule, *_ = _load_masks(tmp_path, raw, classifier, volume)
    assert arteriole is not None and venule is not None
    written_a = tifffile.imread(tmp_path / "masks" / "raw_large_arteriole_segmented.tif")
    written_v = tifffile.imread(tmp_path / "masks" / "raw_large_venule_segmented.tif")
    assert np.array_equal(np.asarray(arteriole, bool), written_a > 0)
    assert np.array_equal(np.asarray(venule, bool), written_v > 0)
    assert not (written_a.astype(bool) & written_v.astype(bool)).any()


def test_a_classifier_never_shown_a_venule_makes_a_global_mask(tmp_path, image):
    from haemolynx.io.vessel_masks import vessel_mask_role_is_global

    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "g.haemoclf", volume, truth, "vessel_classes", labels=["Arteriole"])
    arteriole, venule, *_ = _load_masks(tmp_path, raw, classifier, volume)
    assert arteriole is not None and venule is None
    assert (tmp_path / "masks" / "raw_large_global_segmented.tif").exists()
    settings = settings_for(
        tmp_path, automated_vessel_assignment=True, use_large_vessel_masks=True, raw_image_path=raw,
        use_pixel_classifier_large_vessel_segmentation=True, pixel_classifier_large_vessel_path=classifier,
    )
    assert vessel_mask_role_is_global(settings, "large")


def test_roles_the_classifier_makes_are_not_matched_as_files(tmp_path, image):
    from haemolynx.graph.boundaries import vessel_masks_segmented_from_raw
    from haemolynx.pipeline.batch import batch_from_raw, batch_mask_sizes, classifier_output_paths

    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "av.haemoclf", volume, truth, "vessel_classes")
    settings = settings_for(
        tmp_path, automated_vessel_assignment=True, use_large_vessel_masks=True,
        use_small_vessel_masks_for_boundary_assignment=True, raw_image_path=raw,
        use_pixel_classifier_large_vessel_segmentation=True, pixel_classifier_large_vessel_path=classifier,
        use_pixel_classifier_segmentation=True, pixel_classifier_path=classifier,
    )
    assert vessel_masks_segmented_from_raw(settings, "large")
    assert not vessel_masks_segmented_from_raw(settings, "small")
    assert batch_mask_sizes(settings) == ("small",)
    assert batch_from_raw(settings)
    names = {path.name for path in classifier_output_paths(settings)}
    assert {"raw_segmented.tif", "raw_large_arteriole_segmented.tif", "raw_large_venule_segmented.tif"} <= names


# ------------------------------------------------ pericyte and block masks


def test_the_pericyte_mask_is_made_only_while_something_reads_it(tmp_path, image):
    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "p.haemoclf", volume, truth, "pericytes")
    common = dict(raw_image_path=raw, input_path=raw, use_pixel_classifier_pericyte_segmentation=True,
                  pixel_classifier_pericyte_path=classifier)
    assert object_masks_made(settings_for(tmp_path, **common)) == ()
    settings = settings_for(tmp_path, measure_pericytes=True, **common)
    assert object_masks_made(settings) == ("pericytes",)
    made = segment_object_masks(settings)
    expected = tmp_path / "out" / "raw_pericytes_segmented.tif"
    assert made == {"pericytes": expected}
    assert settings["pericyte_mask_path"] == expected == settings["pericyte_measurement_mask_path"]
    record = json.loads(expected.with_name(expected.name + ".haemolynx.json").read_text())
    assert record["label"] == "Pericyte"


def test_the_block_mask_is_made_for_a_block_mask_perturbation(tmp_path, image):
    from haemolynx.haemodynamics.perturbations import perturbation_problems

    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "b.haemoclf", volume, truth, "blocks")
    entry = {"name": "blocks", "type": "capillary_block_mask", "overrides": {}}
    settings = settings_for(
        tmp_path, raw_image_path=raw, input_path=raw, run_haemodynamics=True, run_perturbations=True,
        perturbations=[entry], use_pixel_classifier_block_segmentation=True, pixel_classifier_block_path=classifier,
    )
    assert object_masks_made(settings) == ("blocks",)
    assert not [p for p in perturbation_problems(settings, SCHEMA) if "names no mask" in p]
    segment_object_masks(settings)
    assert settings["capillary_block_mask_path"] == tmp_path / "out" / "raw_blocks_segmented.tif"


# ------------------------------------------------------------- preflight


def test_preflight_refuses_a_classifier_trained_for_another_job(tmp_path, image):
    from haemolynx.pipeline.checks import check_pixel_classifiers

    volume, truth, raw = image
    pericytes = classifier_file(tmp_path / "p.haemoclf", volume, truth, "pericytes")
    vessels = classifier_file(tmp_path / "v.haemoclf", volume, truth)
    wrong = check_pixel_classifiers(settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=pericytes,
    ))
    assert any("trained to find Pericyte" in e for e in wrong.errors)
    right = check_pixel_classifiers(settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=vessels,
    ))
    assert not right.errors


def test_preflight_refuses_channels_the_raw_image_lacks(tmp_path, image):
    from haemolynx.classifier import FeatureSpec
    from haemolynx.pipeline.checks import check_pixel_classifiers

    volume, truth, raw = image
    classifier = new_classifier("vessels", scales_um=(1.0,), channels=(2,))
    spec = FeatureSpec(scales_um=(1.0,), channels=(2,))
    points, classes = strokes(truth, per_class=50)
    ranges = [intensity_range(volume)]
    voxels = TrainingVoxels(features_at([volume], ranges, VOXEL, spec, points), classes,
                            np.zeros(len(classes), np.int32), points)
    path = save_classifier(
        classifier.with_image_voxels(TrainingImage("t", volume.shape, VOXEL, tuple(ranges)), voxels).trained(n_jobs=1),
        tmp_path / "c3.haemoclf",
    )
    report = check_pixel_classifiers(settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=path,
    ))
    assert any("one channel" in e for e in report.errors)


def test_a_run_that_used_the_classifier_cites_it(tmp_path, image):
    from haemolynx.pipeline.citations import _image_readers, _used_pixel_classifier

    volume, truth, raw = image
    classifier = classifier_file(tmp_path / "v.haemoclf", volume, truth)
    settings = settings_for(
        tmp_path, use_pixel_classifier_segmentation=True, raw_image_path=raw, pixel_classifier_path=classifier,
    )
    assert _used_pixel_classifier(settings)
    assert _image_readers(settings) == {"tifffile"}
    assert not _used_pixel_classifier(settings_for(tmp_path, raw_image_path=raw))
