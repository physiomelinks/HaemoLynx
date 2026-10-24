"""ilastik's vessel masks from chosen channels of the run's raw image
(`PLAN_batch_run.md`, Harvey 2026-10-24): each classifier is given its own
channels of the raw image, kept separate, written to a file of their own for
as long as ilastik runs; its output's record names them; preflight checks them;
the separate input files they replace still load."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from test_io import _count_runs, _fake_ilastik, _ilastik_project, _segment  # noqa: E402

from haemolynx.io.ilastik import (  # noqa: E402
    channel_input_path,
    sidecar_path,
    write_ilastik_channels,
)
from haemolynx.io.load import read_voxel_size_xyz  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402
from haemolynx.pipeline.checks import check_ilastik_vessel_mask_channels, preflight  # noqa: E402


def _composite(path: Path, *, ome: bool = False) -> np.ndarray:
    """A 4-plane, 3-channel stack, every voxel numbered so one channel cannot
    pass for another, as ImageJ's hyperstack (or an OME-TIFF), with 0.5 um
    pixels and 2 um between planes."""
    data = np.arange(4 * 3 * 6 * 8, dtype=np.uint16).reshape(4, 3, 6, 8)
    if ome:
        tifffile.imwrite(
            path, data, ome=True, photometric="minisblack",
            metadata={"axes": "ZCYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0},
        )
    else:
        tifffile.imwrite(
            path, data, imagej=True, resolution=(2.0, 2.0),
            metadata={"axes": "ZCYX", "spacing": 2.0, "unit": "um", "mode": "composite"},
        )
    return data


# --- the channels written for ilastik ----------------------------------------


@pytest.mark.parametrize("ome", [False, True])
def test_chosen_channels_are_written_separate_in_order_with_the_voxel_size(tmp_path, ome):
    data = _composite(tmp_path / "raw.tif", ome=ome)

    two = write_ilastik_channels(tmp_path / "raw.tif", [2, 0], tmp_path / "two.tif")
    with tifffile.TiffFile(two) as tif:
        series = tif.series[0]
        assert (series.axes, series.shape) == ("ZCYX", (4, 2, 6, 8))
        assert (tif.is_imagej, tif.is_ome) == (not ome, ome), "the same kind of file"
        written = series.asarray()
    np.testing.assert_array_equal(written, data[:, [2, 0]])
    assert read_voxel_size_xyz(two)[0] == pytest.approx((0.5, 0.5, 2.0))

    one = write_ilastik_channels(tmp_path / "raw.tif", [1], tmp_path / "one.tif")
    with tifffile.TiffFile(one) as tif:
        assert (tif.series[0].axes, tif.series[0].shape) == ("ZYX", (4, 6, 8))
        np.testing.assert_array_equal(tif.series[0].asarray(), data[:, 1])
    assert read_voxel_size_xyz(one)[0] == pytest.approx((0.5, 0.5, 2.0))


@pytest.mark.parametrize(
    "channels, message",
    [
        ([3], r"has 3 channels \(0-2, C1-C3\); channel\(s\) \[3\] do not exist"),
        ([1, 1], r"Channel\(s\) \[1\] are chosen more than once"),
        ([], "No channels chosen"),
    ],
)
def test_channels_a_file_cannot_give_are_refused(tmp_path, channels, message):
    _composite(tmp_path / "raw.tif")
    with pytest.raises(ValueError, match=message):
        write_ilastik_channels(tmp_path / "raw.tif", channels, tmp_path / "out.tif")


def test_a_single_channel_file_has_no_channels_to_give(tmp_path):
    tifffile.imwrite(tmp_path / "plain.tif", np.zeros((4, 6, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="no channels to choose from"):
        write_ilastik_channels(tmp_path / "plain.tif", [0], tmp_path / "out.tif")


# --- ilastik run on them -------------------------------------------------------


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "composite.tif"
    _composite(raw)
    project = _ilastik_project(tmp_path / "classifier.ilp")
    return raw, project, tmp_path / "out" / "composite_large_global_segmented.tif"


def test_ilastik_reads_the_chosen_channels_and_they_go_once_it_is_done(tmp_path, monkeypatch):
    raw, project, output = _inputs(tmp_path)
    arguments = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")
    monkeypatch.setenv("FAKE_ILASTIK_ARGS", str(arguments))

    assert _segment(tmp_path, raw, project, output, channels=[2]) == output

    assert Path(json.loads(arguments.read_text())[-1]) == channel_input_path(output)
    assert not channel_input_path(output).exists()
    assert not channel_input_path(output).parent.exists()
    record = json.loads(sidecar_path(output).read_text())
    assert record["input"]["channels"] == [2]
    assert Path(record["input"]["path"]) == raw.resolve()


def test_the_channels_go_when_ilastik_fails(tmp_path, monkeypatch):
    raw, project, output = _inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "fail")
    with pytest.raises(RuntimeError, match="exit code 3"):
        _segment(tmp_path, raw, project, output, channels=[0])
    assert not channel_input_path(output).exists()


def test_an_output_is_reused_for_the_same_channels_only(tmp_path, monkeypatch, caplog):
    raw, project, output = _inputs(tmp_path)
    runs = _count_runs(tmp_path, monkeypatch)

    _segment(tmp_path, raw, project, output, reuse_existing=True, channels=[2])
    _segment(tmp_path, raw, project, output, reuse_existing=True, channels=[2])
    assert runs() == 1, "the same channels: reused, nothing written for ilastik"
    with caplog.at_level(logging.WARNING, logger="haemolynx.io.ilastik"):
        _segment(tmp_path, raw, project, output, reuse_existing=True, channels=[1, 2])
    assert runs() == 2
    assert any("channels read of it, has changed" in record.getMessage() for record in caplog.records)
    _segment(tmp_path, raw, project, output, reuse_existing=True)
    assert runs() == 3, "the whole file is another input again"


def test_the_vessel_mask_loader_gives_each_classifier_its_channels(tmp_path, monkeypatch):
    """End to end through the fake ilastik: one global mask from channel 1
    of the run's raw image, named after the raw image."""
    from haemolynx.io import load_and_validate_vessel_masks

    raw, project, _output = _inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")
    mask, venule, *_ = load_and_validate_vessel_masks(
        mask_role="large", enabled=True, use_ilastik=True,
        arteriole_mask_path=None, venule_mask_path=None,
        image_shape=(4, 6, 8), main_voxel_size_xyz=(0.5, 0.5, 2.0),
        raw_image_path=raw,
        ilastik_arteriole_channels=(1,),
        ilastik_arteriole_classifier_path=project,
        ilastik_arteriole_vessel_label="Vessel",
        ilastik_output_dir=tmp_path / "seg",
        ilastik_executable=str(_fake_ilastik(tmp_path)),
    )
    output = tmp_path / "seg" / "composite_large_global_segmented.tif"
    assert venule is None
    assert int(np.count_nonzero(mask)) == 3 * 6 * 8  # the fake's vessel label
    assert json.loads(sidecar_path(output).read_text())["input"]["channels"] == [1]


# --- preflight -----------------------------------------------------------------


@pytest.fixture(scope="module")
def schema():
    return default_schema()


def _ilastik_masks(raw, **changes) -> dict:
    return {
        "automated_vessel_assignment": True,
        "use_large_vessel_masks": True,
        "use_ilastik_large_vessel_segmentation": True,
        "raw_image_path": raw,
        "ilastik_venule_classifier_path": None,
        **changes,
    }


def test_preflight_checks_each_classifiers_channels_against_the_raw_image(tmp_path):
    raw = tmp_path / "composite.tif"
    _composite(raw)
    ok = check_ilastik_vessel_mask_channels(_ilastik_masks(raw, ilastik_arteriole_channels=(0, 2)))
    assert ok.ok and ok.passed == [("ilastik_arteriole_channels", "C1, C3 of composite.tif")]
    (error,) = check_ilastik_vessel_mask_channels(_ilastik_masks(raw, ilastik_arteriole_channels=(3,))).errors
    assert "ilastik_arteriole_channels names channel(s) [3], but raw_image_path has 3 channels" in error
    (error,) = check_ilastik_vessel_mask_channels(_ilastik_masks(raw, ilastik_arteriole_channels=(1, 1))).errors
    assert "more than once" in error
    # The venule classifier's channels are read only with a venule classifier.
    venule = _ilastik_masks(
        raw, ilastik_venule_classifier_path=tmp_path / "ven.ilp", ilastik_venule_channels=(5,)
    )
    (error,) = check_ilastik_vessel_mask_channels(venule).errors
    assert error.startswith("ilastik_venule_channels")
    assert check_ilastik_vessel_mask_channels(
        {**venule, "ilastik_venule_classifier_path": None}
    ).ok


def test_preflight_wants_a_raw_image_for_ilastiks_vessel_masks(tmp_path):
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 6, 8), dtype=np.uint8))
    (error,) = check_ilastik_vessel_mask_channels(_ilastik_masks(None)).errors
    assert "raw_image_path is not set" in error
    (error,) = check_ilastik_vessel_mask_channels(_ilastik_masks(plain, ilastik_arteriole_channels=(0,))).errors
    assert "single-channel image" in error
    # No channels: the whole file, whatever it holds.
    assert check_ilastik_vessel_mask_channels(_ilastik_masks(plain)).ok
    # Masks not in use: nothing is segmented.
    assert check_ilastik_vessel_mask_channels(
        _ilastik_masks(None, automated_vessel_assignment=False)
    ).ok


def test_preflight_runs_the_vessel_mask_channel_check(tmp_path, schema):
    raw = tmp_path / "composite.tif"
    _composite(raw)
    settings = {**schema.defaults(), **_ilastik_masks(raw, ilastik_arteriole_channels=(4,))}
    report = preflight(settings, schema, echo=False)
    assert any("ilastik_arteriole_channels names channel(s) [4]" in error for error in report.errors)


def test_each_image_of_a_batch_gives_the_classifier_its_own_raw_image(tmp_path, schema):
    from haemolynx.io import BatchFolders, match_batch
    from haemolynx.pipeline.batch import batch_preflight, item_settings

    seg, raw = tmp_path / "seg", tmp_path / "raw"
    seg.mkdir()
    raw.mkdir()
    for stem in ("E14", "E15"):
        tifffile.imwrite(seg / f"{stem}_seg.tif", np.zeros((4, 6, 8), dtype=np.uint8))
        _composite(raw / f"{stem}.tif")
    base = {
        **schema.defaults(),
        **_ilastik_masks(None, ilastik_arteriole_channels=(1,)),
        "ilastik_arteriole_classifier_path": _ilastik_project(tmp_path / "vessels.ilp"),
        "ilastik_arteriole_vessel_label": "Vessel",
        "ilastik_executable": str(_fake_ilastik(tmp_path)),
    }
    out = tmp_path / "out"
    for match in match_batch(BatchFolders(seg, raw=raw)):
        settings = item_settings(base, match, out)
        assert settings["raw_image_path"] == match.raw
        assert settings["ilastik_output_dir"] == out / match.stem / "segmentations"
        report = batch_preflight(base, match, out, schema)
        assert not [error for error in report.errors if "ilastik" in error], report.errors
        assert ("ilastik_arteriole_channels", f"C2 of {match.raw.name}") in report.passed


# --- the input files they replace --------------------------------------------


def test_the_retired_input_files_still_load(schema, caplog):
    old = {
        "use_ilastik_large_vessel_segmentation": True,
        "raw_image_path": "raw.tif",
        "ilastik_unsegmented_arteriole_image_path": "raw.tif",
        "ilastik_unsegmented_venule_image_path": "venules.tif",
        "ilastik_unsegmented_small_arteriole_image_path": "small.tif",
    }
    with caplog.at_level(logging.WARNING, logger="haemolynx.pipeline.schema"):
        values = schema.validate(old)
    assert not [name for name in values if name.startswith("ilastik_unsegmented")]
    assert values["ilastik_arteriole_channels"] is None
    warned = [record.getMessage() for record in caplog.records]
    # The raw image itself keeps working; another file is a different input now.
    assert not any("ilastik_unsegmented_arteriole_image_path" in line for line in warned)
    assert any("ilastik_unsegmented_venule_image_path (venules.tif) is retired" in line for line in warned)
    # Its switch off, a small mask's old file read nothing either way.
    assert not any("small_arteriole" in line for line in warned)
