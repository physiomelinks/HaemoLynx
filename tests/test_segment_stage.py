"""What `segment()` accepts, and what it says when it cannot go on.

The stage settles one question -- which file the rest of the run reads -- and
there are two ways to answer it: name a segmented image, or ask ilastik to make
one. The second is the awkward case, because then `input_path` is the stage's
*output* rather than its input, and the documented way to ask for it is to leave
the setting empty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.stages import segment

SCHEMA = default_schema()


def settings_for(tmp_path: Path, **overrides) -> dict:
    values = {"output_dir": tmp_path, **overrides}
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def a_segmented_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "mask.tif"
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[1:3, 1:3, :] = 255
    tifffile.imwrite(path, volume)
    return path


def test_an_empty_input_path_is_not_a_crash_when_ilastik_will_fill_it(tmp_path):
    """#127: `Path(None)` before the branch that produces the path.

    `input_path` was converted on the first line, so the one configuration the
    setting exists to support -- leave it empty, let ilastik segment -- died
    with `TypeError: expected str, bytes or os.PathLike object, not NoneType`
    before ilastik was reached. Whatever happens next, it must not be that.
    """
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=tmp_path / "raw.tif",
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )

    with pytest.raises(Exception) as raised:  # noqa: PT011 - the type is the point
        segment(settings)

    assert not isinstance(raised.value, TypeError), raised.value
    # It gets far enough to complain about the thing that is actually missing.
    assert "raw.tif" in str(raised.value)


def test_no_input_path_and_no_ilastik_says_so_plainly(tmp_path):
    """The other half of the same branch: nothing to read and nothing to make.

    This used to reach `Path(None)` too. It is a real mistake either way, so it
    should be named rather than arriving as a TypeError from a conversion.
    """
    settings = settings_for(tmp_path, use_ilastik_segmentation=False, input_path=None)

    with pytest.raises(ValueError, match="input_path must name a segmented image"):
        segment(settings)


def test_a_named_image_is_used_as_it_stands(tmp_path):
    """The ordinary case, so the fix above cannot have quietly changed it."""
    mask = a_segmented_tiff(tmp_path)
    inputs = segment(settings_for(tmp_path, input_path=mask))

    assert inputs.image_path == mask
    assert inputs.input_format == "tif"
    assert inputs.output_dir == tmp_path


def test_a_format_the_pipeline_cannot_read_is_refused(tmp_path):
    other = tmp_path / "mask.png"
    other.write_bytes(b"not really a png")

    with pytest.raises(ValueError, match="Invalid image format"):
        segment(settings_for(tmp_path, input_path=other))


def test_ilastik_without_a_classifier_is_refused(tmp_path):
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=a_segmented_tiff(tmp_path),
        ilastik_classifier_path=None,
    )

    with pytest.raises(ValueError, match="ilastik_classifier_path"):
        segment(settings)


def test_ilastik_hands_on_the_path_it_produced(tmp_path, monkeypatch):
    """The whole point of leaving `input_path` empty: the stage fills it in."""
    from haemolynx import io

    raw = a_segmented_tiff(tmp_path)
    produced = tmp_path / "segmentations" / "mask_segmented.tif"
    produced.parent.mkdir()
    produced.write_bytes(raw.read_bytes())

    monkeypatch.setattr(
        io, "run_ilastik_headless_segmentation",
        lambda **_kwargs: produced,
    )

    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=raw,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )
    inputs = segment(settings)

    assert inputs.image_path == produced
    assert settings["input_path"] == produced


def test_ilastik_timeout_setting_reaches_the_subprocess_call(tmp_path, monkeypatch):
    """ilastik_timeout_seconds must actually be threaded through, not just declared."""
    from haemolynx import io

    raw = a_segmented_tiff(tmp_path)
    produced = tmp_path / "segmentations" / "mask_segmented.tif"
    produced.parent.mkdir()
    produced.write_bytes(raw.read_bytes())

    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return produced

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", fake_run)

    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=raw,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
        ilastik_timeout_seconds=42.0,
    )
    segment(settings)

    assert seen.get("timeout") == 42.0


def test_ilastik_reuse_setting_reaches_the_call_and_defaults_on(tmp_path, monkeypatch):
    """A current segmentation is reused unless the user asks for a new one."""
    from haemolynx import io

    raw = a_segmented_tiff(tmp_path)
    produced = tmp_path / "segmentations" / "mask_segmented.tif"
    produced.parent.mkdir()
    produced.write_bytes(raw.read_bytes())
    seen = []

    def fake_run(**kwargs):
        seen.append(kwargs)
        return produced

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", fake_run)
    common = dict(
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=raw,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )

    segment(settings_for(tmp_path, **common))
    segment(settings_for(tmp_path, ilastik_reuse_existing_output=False, **common))

    assert [call["reuse_existing"] for call in seen] == [True, False]


def test_a_stop_during_ilastik_arrives_through_the_stage_heartbeat(tmp_path, monkeypatch):
    """The progress callback is the only way a watcher can stop a run, and
    ilastik is one long call: the stage hands it a heartbeat to poll."""
    from haemolynx import io
    from haemolynx.pipeline.progress import HEARTBEAT, RunProgress

    raw = a_segmented_tiff(tmp_path)

    def fake_run(**kwargs):
        for _ in range(10):
            kwargs["poll"]()
        raise AssertionError("the stop was not honoured")

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", fake_run)
    kinds = []

    class Stopped(Exception):
        pass

    def watcher(event):
        kinds.append(event.kind)
        if kinds.count(HEARTBEAT) == 2:
            raise Stopped()

    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=raw,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )
    run = RunProgress(watcher)
    with pytest.raises(Stopped):
        with run.stage("segment") as segmenting:
            segment(settings, progress=segmenting)

    assert kinds.count(HEARTBEAT) == 2


def test_ilastik_without_a_raw_image_reads_the_segmented_image_and_says_so(tmp_path, caplog):
    """The raw image is optional: with none to segment, ilastik is skipped
    (the panel greys the switch out) and the run reads the segmented image,
    as it would with the switch off."""
    import logging

    from haemolynx.pipeline.stages import ilastik_segments, segmented_input_path

    mask = a_segmented_tiff(tmp_path)
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=mask,
        raw_image_path=None,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )
    assert not ilastik_segments(settings)
    assert segmented_input_path(settings) == mask

    with caplog.at_level(logging.WARNING, logger="haemolynx"):
        segment(settings)

    assert settings["input_path"] == mask
    assert not (tmp_path / "segmentations").exists()
    assert any(
        "ilastik is skipped" in record.getMessage() and "raw_image_path" in record.getMessage()
        for record in caplog.records
    )


def test_ilastik_without_a_raw_image_or_a_segmented_image_is_named(tmp_path):
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        raw_image_path=None,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
    )

    with pytest.raises(ValueError, match="raw_image_path is not set, and there is no segmented image"):
        segment(settings)


# -- an ilastik output read, never made (the panel's Optimise settings) ------


def _ilastik_project(path: Path, label_names=("Vessel", "Background")) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as project:
        project.create_dataset(
            "PixelClassification/LabelNames",
            data=[name.encode() for name in label_names],
            dtype=h5py.string_dtype(),
        )
    return path


def _ilastik_run(tmp_path: Path, *, record=True, **overrides):
    """Settings segmenting a raw image with ilastik, and the output a run
    of them made before -- recorded as ilastik's runner records it."""
    from haemolynx.io.ilastik import _write_sidecar, ilastik_label_names, ilastik_provenance

    raw = a_segmented_tiff(tmp_path)
    project = _ilastik_project(tmp_path / "classifier.ilp")
    output = tmp_path / "segmentations" / "mask_segmented.tif"
    output.parent.mkdir()
    output.write_bytes(raw.read_bytes())
    if record:
        provenance = ilastik_provenance(raw, project, 1, label_names=ilastik_label_names(project))
        _write_sidecar(output, provenance, vessel_voxels=16, voxels=64)
    settings = settings_for(
        tmp_path,
        **{
            "use_ilastik_segmentation": True,
            "input_path": None,
            "raw_image_path": raw,
            "ilastik_classifier_path": project,
            "ilastik_output_dir": output.parent,
            "ilastik_output_suffix": ".tif",
            "ilastik_vessel_label": "Vessel",
            **overrides,
        },
    )
    return settings, output, project


@pytest.fixture
def no_ilastik(monkeypatch):
    from haemolynx import io

    def never(**_kwargs):
        raise AssertionError("ilastik was run")

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", never)


def test_a_current_ilastik_output_is_found_without_running_ilastik(tmp_path, no_ilastik):
    from haemolynx.pipeline.stages import current_ilastik_output

    settings, output, _project = _ilastik_run(tmp_path)

    assert current_ilastik_output(settings) == (output, "")


def test_an_ilastik_output_is_not_current_and_says_why(tmp_path, no_ilastik):
    from haemolynx.pipeline.stages import current_ilastik_output

    settings, output, project = _ilastik_run(tmp_path)
    _ilastik_project(project, ("Vessel", "Background", "Other"))  # another project now
    path, reason = current_ilastik_output(settings)
    assert path is None
    assert str(output) in reason and "project" in reason

    other = tmp_path / "other"
    other.mkdir()
    unrecorded, _output, _project = _ilastik_run(other, record=False)
    path, reason = current_ilastik_output(unrecorded)
    assert path is None and "haemolynx.json" in reason


def test_an_ilastik_output_cannot_be_current_without_its_vessel_label(tmp_path, no_ilastik):
    from haemolynx.pipeline.stages import current_ilastik_output

    for label, named in ((None, "ilastik_vessel_label"), ("Lumen", "'Lumen'")):
        folder = tmp_path / str(label)
        folder.mkdir()
        settings, _output, _project = _ilastik_run(folder, ilastik_vessel_label=label)
        path, reason = current_ilastik_output(settings)
        assert path is None and named in reason


def test_no_ilastik_output_is_current_when_ilastik_does_not_segment(tmp_path, no_ilastik):
    from haemolynx.pipeline.stages import current_ilastik_output

    for name, overrides in (
        ("off", {"use_ilastik_segmentation": False}),
        ("no_raw", {"raw_image_path": None}),
        ("synthetic", {"network_source": "synthetic"}),
    ):
        folder = tmp_path / name
        folder.mkdir()
        settings, _output, _project = _ilastik_run(folder, **overrides)
        assert current_ilastik_output(settings)[0] is None


# -- preflight's quality check never runs ilastik ------------------------------


@pytest.fixture
def ilastik_calls(monkeypatch):
    """Every ilastik run asked for. Recorded rather than raised: the quality
    check turns any exception into a warning, which would hide the call."""
    from haemolynx import io

    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("ilastik was run")

    monkeypatch.setattr(io, "run_ilastik_headless_segmentation", record)
    return calls


def test_preflight_quality_check_does_not_run_ilastik(tmp_path, ilastik_calls):
    """Preflight runs on the panel's own thread: scoring the segmentation
    must not start an ilastik run, which can take hours."""
    from haemolynx.pipeline.checks import check_segmentation_quality

    settings, output, _project = _ilastik_run(tmp_path, min_acceptable_segmentation_quality=1.0)
    output.unlink()  # nothing segmented yet

    report = check_segmentation_quality(settings)

    assert ilastik_calls == []
    assert any("ilastik has not yet made" in warning for warning in report.warnings), report.warnings
    assert not report.errors


def test_preflight_quality_check_scores_ilastiks_current_output(tmp_path, ilastik_calls):
    from haemolynx.pipeline.checks import check_segmentation_quality

    settings, _output, _project = _ilastik_run(tmp_path, min_acceptable_segmentation_quality=0.0)

    report = check_segmentation_quality(settings)

    assert ilastik_calls == []
    assert [label for label, _detail in report.passed] == ["segmentation quality"], (report.passed, report.warnings)


# -- preflight reads ilastik as the stage does ---------------------------------


def _preflight(settings):
    from haemolynx.pipeline import preflight

    return preflight(settings, SCHEMA, echo=False)


def test_preflight_with_ilastik_on_but_no_raw_image_checks_the_segmented_image(tmp_path):
    """Without a raw image the stage skips ilastik and reads input_path, its
    outputs named after it: preflight asked for ilastik's program and project
    and looked for a cached skeleton named 'input'."""
    mask = a_segmented_tiff(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "mask_skeleton.npy").write_bytes(b"")
    settings = settings_for(
        tmp_path,
        output_dir=out,
        use_ilastik_segmentation=True,
        raw_image_path=None,
        input_path=mask,
        ilastik_classifier_path=tmp_path / "missing.ilp",
        ilastik_executable="",
        do_skeletonize=False,
    )

    assert segment(dict(settings)).image_path == mask
    report = _preflight(settings)

    assert not [e for e in report.errors if "ilastik" in e or "skeleton" in e], report.errors


def test_preflight_refuses_a_raw_image_that_is_not_a_tiff(tmp_path):
    """The build reads the raw image whenever it is set, as a TIFF: an HDF5
    one passed preflight, then stopped the Graph stage."""
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"")
    settings = settings_for(tmp_path, input_path=a_segmented_tiff(tmp_path), raw_image_path=raw)

    report = _preflight(settings)

    assert any("raw_image_path" in e and "not a TIFF" in e for e in report.errors), report.errors


@pytest.mark.parametrize(
    "overrides, named",
    [
        ({"ilastik_output_suffix": "tif"}, "ilastik_output_suffix"),
        ({"ilastik_output_suffix": ".png"}, "ilastik_output_suffix"),
        ({"ilastik_output_dir": None}, "ilastik_output_dir"),
    ],
)
def test_preflight_refuses_ilastik_output_settings_the_runner_would(tmp_path, overrides, named):
    settings, _output, _project = _ilastik_run(tmp_path, **overrides)

    report = _preflight(settings)

    assert any(named in e for e in report.errors), report.errors
