"""Tests for io module."""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from haemolynx.io import (
    crop_tiff_volume_from_corners,
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    bridge_gaps,
    simplify_to_3d,
)


def test_bridge_gaps():
    arr = np.zeros((5, 5, 5), dtype=bool)
    arr[2, 2, :] = True
    arr[2, 2, 1] = False  # 1-voxel gap
    result = bridge_gaps(arr, max_gap=2)
    assert result[2, 2, 1]


def test_simplify_to_3d():
    img3 = np.random.rand(4, 4, 4)
    assert simplify_to_3d(img3).shape == (4, 4, 4)
    img4 = np.random.rand(4, 4, 4, 2)
    out = simplify_to_3d(img4)
    assert out.shape == (4, 4, 4)


def test_simplify_to_3d_raises():
    with pytest.raises(ValueError):
        simplify_to_3d(np.zeros((3, 3)))


def test_load_and_skeletonize_3d_tif(tmp_path):
    f = tmp_path / "test.tif"
    img = np.random.randint(0, 255, (6, 6, 6), dtype=np.uint16)
    import tifffile
    tifffile.imwrite(f, img)
    (
        image,
        skeleton,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_meta_status,
    ) = load_and_skeletonize_3d_tif(str(f))
    assert image.shape == (6, 6, 6)
    assert skeleton.shape == (6, 6, 6)
    assert skeleton.dtype == bool
    assert (voxel_size_x, voxel_size_y, voxel_size_z) == (1.0, 1.0, 1.0)
    assert voxel_meta_status["status"] in {"missing", "partial"}


def test_crop_tiff_volume_from_corners(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    arr = np.arange(10 * 8 * 6, dtype=np.uint16).reshape((10, 8, 6))
    import tifffile

    tifffile.imwrite(src, arr)
    info = crop_tiff_volume_from_corners(
        src,
        dst,
        corner_a=(9, 7, 5),
        corner_b=(7, 4, 0),
    )

    out = tifffile.imread(dst)
    assert out.shape == (3, 4, 6)
    assert tuple(info["source_shape"]) == (10, 8, 6)
    assert tuple(info["cropped_shape"]) == (3, 4, 6)


# --- reading a voxel size without reading the pixels -------------------------


ANISOTROPIC_XYZ = (0.4, 0.5, 2.0)


def _tiff_with_voxel_metadata(path, shape=(4, 6, 8)):
    import numpy as np
    import tifffile

    tifffile.imwrite(
        path,
        np.zeros(shape, dtype=np.uint8),
        imagej=True,
        resolution=(1.0 / ANISOTROPIC_XYZ[0], 1.0 / ANISOTROPIC_XYZ[1]),
        metadata={"spacing": ANISOTROPIC_XYZ[2], "unit": "um"},
    )
    return path


def test_read_voxel_size_xyz_agrees_with_the_full_loader(tmp_path):
    """The cheap read and the real one must not disagree, or the panel scales
    a layer to something the run will not use."""
    from haemolynx.io import load_3d_tif_with_voxel_size, read_voxel_size_xyz

    path = _tiff_with_voxel_metadata(tmp_path / "aniso.tif")

    found, _status = read_voxel_size_xyz(path)
    _image, x, y, z, _meta = load_3d_tif_with_voxel_size(str(path))

    assert found == pytest.approx((x, y, z))
    assert found == pytest.approx(ANISOTROPIC_XYZ)


def test_read_voxel_size_xyz_does_not_read_the_pixels(tmp_path, monkeypatch):
    """The point of it: the tags are in the header, the stack can be huge."""
    import tifffile

    from haemolynx.io import read_voxel_size_xyz

    path = _tiff_with_voxel_metadata(tmp_path / "aniso.tif")

    def _refuse(self, *args, **kwargs):
        raise AssertionError("asarray was called; this should be a header read")

    monkeypatch.setattr(tifffile.TiffFile, "asarray", _refuse)
    assert read_voxel_size_xyz(path) is not None


def test_read_voxel_size_xyz_is_none_when_the_file_says_nothing(tmp_path):
    """All ones is the absence of an answer, not an answer."""
    import numpy as np
    import tifffile

    from haemolynx.io import read_voxel_size_xyz

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 6, 8), dtype=np.uint8))
    assert read_voxel_size_xyz(path) is None


def test_read_voxel_size_xyz_is_none_for_what_it_cannot_read(tmp_path):
    from haemolynx.io import read_voxel_size_xyz

    assert read_voxel_size_xyz(tmp_path / "absent.tif") is None
    not_a_tiff = tmp_path / "notes.txt"
    not_a_tiff.write_text("hello")
    assert read_voxel_size_xyz(not_a_tiff) is None


# --- an OME-TIFF's voxel size, from its OME header ---------------------------
#
# tifffile writes one pixel per unit into the resolution tags of a file it is
# given no resolution for, so an OME-TIFF's tags say 1 micron whatever its
# voxels are. Its size is the OME-XML header's PhysicalSizeX/Y/Z.

MUSCLE_XYZ = (0.91, 0.91, 4.94)


def _ome_tiff(path, sizes, shape=(4, 6, 8), **kwargs):
    import tifffile

    tifffile.imwrite(
        path,
        np.zeros(shape, dtype=np.uint8),
        ome=True,
        metadata={"axes": "ZYX", **sizes},
        **kwargs,
    )
    return path


def _ome_sizes(xyz, unit=None):
    sizes = {}
    for axis, size in zip("XYZ", xyz):
        sizes[f"PhysicalSize{axis}"] = size
        if unit is not None:
            sizes[f"PhysicalSize{axis}Unit"] = unit
    return sizes


def test_an_ome_tiffs_voxel_size_is_read_from_its_ome_header(tmp_path, caplog):
    """Regression: the skeletal-muscle stack, 0.913 x 0.913 x 4.94 um in its OME
    header, ran on 1 um on every axis -- read from tags saying one pixel per
    unit -- with a warning about z alone."""
    import logging

    import tifffile

    from haemolynx.io import load_3d_tif_with_voxel_size, resolve_voxel_size_xyz

    path = _ome_tiff(tmp_path / "muscle.ome.tif", _ome_sizes(MUSCLE_XYZ))
    with tifffile.TiffFile(path) as tif:
        # The trap: the tags are there, and say 1.
        assert tif.pages[0].tags["XResolution"].value == (1, 1)

    with caplog.at_level(logging.WARNING, logger="haemolynx"):
        _image, x, y, z, status = load_3d_tif_with_voxel_size(str(path))
        resolved, source = resolve_voxel_size_xyz((x, y, z), status, None, "auto")

    assert (x, y, z) == pytest.approx(MUSCLE_XYZ)
    assert status["status"] == "complete"
    assert status["source"] == "ome_metadata"
    assert status["axis_sources"] == {"x": "ome_metadata", "y": "ome_metadata", "z": "ome_metadata"}
    assert resolved == pytest.approx(MUSCLE_XYZ)
    assert source == "metadata"
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize(
    "unit, per_micron",
    [("nm", 1000.0), ("mm", 1e-3), ("cm", 1e-4), ("µm", 1.0)],
)
def test_ome_sizes_are_converted_to_microns_from_their_unit(tmp_path, unit, per_micron):
    from haemolynx.io import load_3d_tif_with_voxel_size

    in_unit = tuple(size * per_micron for size in ANISOTROPIC_XYZ)
    path = _ome_tiff(tmp_path / "units.ome.tif", _ome_sizes(in_unit, unit))

    _image, x, y, z, status = load_3d_tif_with_voxel_size(str(path))

    assert (x, y, z) == pytest.approx(ANISOTROPIC_XYZ)
    assert status["status"] == "complete"


def test_ome_sizes_follow_the_axis_order_the_pages_are_read_with(tmp_path):
    """OME's X, Y and Z are the file's width, height and planes, as the tags
    are: in a stack whose pages step along x, PhysicalSizeZ is the x spacing."""
    from haemolynx.io import load_3d_tif_with_voxel_size, read_voxel_size_xyz

    # Width 0.5, height 0.25, pages 2.0.
    path = _ome_tiff(tmp_path / "pages_along_x.ome.tif", _ome_sizes((0.5, 0.25, 2.0)))

    _image, x, y, z, status = load_3d_tif_with_voxel_size(str(path), axis_order="xyz")
    assert (x, y, z) == pytest.approx((2.0, 0.25, 0.5))
    assert status["status"] == "complete"
    assert read_voxel_size_xyz(path, "xyz")[0] == pytest.approx((2.0, 0.25, 0.5))

    _image, x, y, z, _status = load_3d_tif_with_voxel_size(str(path))
    assert (x, y, z) == pytest.approx((0.5, 0.25, 2.0))


def test_read_voxel_size_xyz_reads_an_ome_tiffs_header(tmp_path, monkeypatch):
    """Regression: it returned None for the muscle stack, whose tags of 1 and
    missing ImageJ spacing looked like a file saying nothing, so the panel drew
    it at 1 um. Still a header read."""
    import tifffile

    from haemolynx.io import read_voxel_size_xyz

    path = _ome_tiff(tmp_path / "muscle.ome.tif", _ome_sizes(MUSCLE_XYZ))

    def _refuse(self, *args, **kwargs):
        raise AssertionError("asarray was called; this should be a header read")

    monkeypatch.setattr(tifffile.TiffFile, "asarray", _refuse)
    found, status = read_voxel_size_xyz(path)

    assert found == pytest.approx(MUSCLE_XYZ)
    assert status["source"] == "ome_metadata"
    assert status["status"] == "complete"


def test_an_axis_the_ome_header_has_no_size_for_is_read_from_the_tags(tmp_path):
    from haemolynx.io import load_3d_tif_with_voxel_size

    path = _ome_tiff(
        tmp_path / "z_only.ome.tif",
        {"PhysicalSizeZ": ANISOTROPIC_XYZ[2]},
        resolution=(1.0 / ANISOTROPIC_XYZ[0], 1.0 / ANISOTROPIC_XYZ[1]),
    )

    _image, x, y, z, status = load_3d_tif_with_voxel_size(str(path))

    assert (x, y, z) == pytest.approx(ANISOTROPIC_XYZ)
    assert status["status"] == "complete"
    assert status["axis_sources"] == {"x": "tiff_metadata", "y": "tiff_metadata", "z": "ome_metadata"}


def test_an_imagej_stack_with_only_a_spacing_reads_as_it_did_before(tmp_path):
    """The ImageJ path is unchanged. ImageJ writes ResolutionUnit NONE beside
    unit=micron, so a stricter reading of the tags' unit would break files
    that load correctly."""
    import tifffile

    from haemolynx.io import load_3d_tif_with_voxel_size

    path = tmp_path / "spacing_only.tif"
    tifffile.imwrite(
        path, np.zeros((4, 6, 8), dtype=np.uint8), imagej=True,
        metadata={"spacing": 2.0, "unit": "um"},
    )

    _image, x, y, z, status = load_3d_tif_with_voxel_size(str(path))

    assert (x, y, z) == pytest.approx((1.0, 1.0, 2.0))
    assert status["status"] == "complete"
    assert status["source"] == "tiff_metadata"
    assert status["axis_sources"] == {"x": "tiff_metadata", "y": "tiff_metadata", "z": "tiff_metadata"}


def _ome_xml(pixels_attributes):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" {pixels_attributes}/></Image>'
        "</OME>"
    )


@pytest.mark.parametrize(
    "attributes, expected",
    [
        ('PhysicalSizeX="0.4" PhysicalSizeY="0.5" PhysicalSizeZ="2"', {"x": 0.4, "y": 0.5, "z": 2.0}),
        ('PhysicalSizeX="400" PhysicalSizeXUnit="nm"', {"x": 0.4}),
        # Greek mu, which some writers use in place of the schema's micro sign.
        ('PhysicalSizeX="0.4" PhysicalSizeXUnit="μm"', {"x": 0.4}),
        ('PhysicalSizeX="4" PhysicalSizeXUnit="Å"', {"x": 4e-4}),
        ('SizeX="8"', {}),
    ],
)
def test_ome_physical_sizes_are_read_in_microns(attributes, expected):
    from haemolynx.io.load import _ome_physical_sizes_um

    assert _ome_physical_sizes_um(_ome_xml(attributes)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "attributes",
    [
        'PhysicalSizeX="3" PhysicalSizeXUnit="pixel"',
        'PhysicalSizeX="0"',
        'PhysicalSizeX="-0.4"',
        'PhysicalSizeX="wide"',
    ],
)
def test_an_ome_size_that_is_not_a_voxel_size_is_left_out_with_a_warning(attributes, caplog):
    """Left out, so the tags are read for that axis instead; warned about,
    since those tags may well say 1."""
    import logging

    from haemolynx.io.load import _ome_physical_sizes_um

    with caplog.at_level(logging.WARNING, logger="haemolynx"):
        sizes = _ome_physical_sizes_um(_ome_xml(attributes + ' PhysicalSizeY="0.5"'))

    assert sizes == pytest.approx({"y": 0.5})
    [warning] = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert "PhysicalSizeX" in warning


@pytest.mark.parametrize("ome_xml", [None, "", "<OME", "<OME/>"])
def test_an_ome_header_without_a_pixels_element_gives_no_sizes(ome_xml):
    from haemolynx.io.load import _ome_physical_sizes_um

    assert _ome_physical_sizes_um(ome_xml) == {}


# --- run_ilastik_headless_segmentation ---------------------------------------
#
# These run a real process: a small Python stand-in for ilastik behind a
# ``.bat`` (Windows) or shell (POSIX) wrapper, which is also what ilastik's own
# launcher is -- so killing the process *tree* on a stop is what is tested.

_FAKE_ILASTIK = r'''
import os, sys, time
mode = os.environ.get("FAKE_ILASTIK_MODE", "ok")
pattern = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_filename_format="))
image = sys.argv[-1]
nickname = os.path.splitext(os.path.basename(image))[0]
print("fake ilastik starting", flush=True)
if mode == "ok":
    print("predicting 50%", flush=True)
    with open(pattern.replace("{nickname}", nickname), "wb") as out:
        out.write(b"segmented")
    print("done", flush=True)
elif mode == "fail":
    print("ERROR: project file is broken", flush=True)
    sys.exit(3)
elif mode == "silent":
    sys.exit(0)
elif mode == "hang":
    beat = os.environ["FAKE_ILASTIK_BEAT"]
    while True:
        with open(beat, "a") as f:
            f.write(".")
        time.sleep(0.05)
'''


def _fake_ilastik(tmp_path):
    """An executable path that behaves like ilastik per ``FAKE_ILASTIK_MODE``."""
    import os
    import stat
    import sys

    script = tmp_path / "fake_ilastik.py"
    script.write_text(_FAKE_ILASTIK)
    if os.name == "nt":
        wrapper = tmp_path / "fake_ilastik.bat"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        wrapper = tmp_path / "fake_ilastik"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def _ilastik_inputs(tmp_path):
    image = tmp_path / "raw.tif"
    image.write_bytes(b"not a real tiff, never opened by this function")
    project = tmp_path / "classifier.ilp"
    project.write_bytes(b"not a real project, never opened by this function")
    return image, project, tmp_path / "out" / "raw_segmented.tif"


def _stopped_growing(path, *, settle=0.6):
    """Whether *path* stops changing: the process writing to it is dead."""
    import time

    time.sleep(settle)
    before = path.stat().st_size
    time.sleep(settle)
    return path.stat().st_size == before


def test_ilastik_output_is_moved_to_the_requested_name(tmp_path, monkeypatch):
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    produced = run_ilastik_headless_segmentation(
        image, project, output, ilastik_executable=_fake_ilastik(tmp_path)
    )

    assert produced == output
    assert output.read_bytes() == b"segmented"
    assert not (output.parent / "raw.tif").exists()


def test_ilastik_output_lines_reach_the_log_as_they_arrive(tmp_path, monkeypatch, caplog):
    """A long segmentation shows its progress, not nothing until it ends."""
    import logging

    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    with caplog.at_level(logging.INFO, logger="haemolynx.io.ilastik"):
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=_fake_ilastik(tmp_path)
        )

    logged = [record.getMessage() for record in caplog.records]
    assert "[ilastik] fake ilastik starting" in logged
    assert "[ilastik] predicting 50%" in logged


def test_ilastik_timeout_kills_the_process_and_raises_an_actionable_error(tmp_path, monkeypatch):
    """A hung ilastik is killed -- the whole tree -- not left running."""
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    beat = tmp_path / "beat.txt"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "hang")
    monkeypatch.setenv("FAKE_ILASTIK_BEAT", str(beat))

    with pytest.raises(RuntimeError, match="ilastik_timeout_seconds"):
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=_fake_ilastik(tmp_path), timeout=1.5
        )

    assert beat.exists()
    assert _stopped_growing(beat)


def test_a_poll_that_raises_stops_ilastik_and_propagates(tmp_path, monkeypatch):
    """How the panel's stop reaches a segmentation that can take hours."""
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    beat = tmp_path / "beat.txt"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "hang")
    monkeypatch.setenv("FAKE_ILASTIK_BEAT", str(beat))
    polls = []

    class Stopped(Exception):
        pass

    def poll():
        polls.append(1)
        if len(polls) >= 3:
            raise Stopped()

    with pytest.raises(Stopped):
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=_fake_ilastik(tmp_path),
            timeout=60.0, poll=poll,
        )

    assert len(polls) == 3
    assert _stopped_growing(beat)


def test_a_failing_ilastik_reports_its_exit_code_and_last_output(tmp_path, monkeypatch):
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "fail")

    with pytest.raises(RuntimeError, match="exit code 3") as raised:
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=_fake_ilastik(tmp_path)
        )

    assert "project file is broken" in str(raised.value)


def test_an_old_output_is_not_returned_when_ilastik_writes_nothing(tmp_path, monkeypatch):
    """Regression: a stale segmentation passed for this run's if ilastik exited
    cleanly without writing one."""
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"from an earlier run")
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "silent")

    with pytest.raises(RuntimeError, match="no output file"):
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=_fake_ilastik(tmp_path)
        )


def test_missing_ilastik_executable_names_the_real_setting(tmp_path):
    """Regression: this used to tell users to set an env var nothing reads.

    ``ILASTIK_EXECUTABLE`` is never read as an environment variable anywhere
    in this codebase -- the real knob is the ``ilastik_executable`` setting.
    """
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="ilastik_executable") as raised:
        run_ilastik_headless_segmentation(
            image, project, output, ilastik_executable=tmp_path / "no_such_ilastik.exe"
        )

    assert "ILASTIK_EXECUTABLE" not in str(raised.value)


def _age(path, seconds):
    import os
    import time

    then = time.time() - seconds
    os.utime(path, (then, then))


def test_an_up_to_date_output_is_reused_without_running_ilastik(tmp_path, monkeypatch):
    import subprocess

    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"made earlier")
    _age(image, 100)
    _age(project, 100)

    def refuse(*_args, **_kwargs):
        raise AssertionError("ilastik was started although its output is current")

    monkeypatch.setattr(subprocess, "Popen", refuse)

    produced = run_ilastik_headless_segmentation(
        image, project, output, ilastik_executable="ilastik.exe", reuse_existing=True
    )

    assert produced == output
    assert output.read_bytes() == b"made earlier"


@pytest.mark.parametrize("changed", ["image", "project"])
def test_an_output_older_than_its_image_or_project_is_made_again(tmp_path, monkeypatch, changed):
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"made earlier")
    _age(output, 100)
    _age(image if changed == "project" else project, 200)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    run_ilastik_headless_segmentation(
        image, project, output, ilastik_executable=_fake_ilastik(tmp_path), reuse_existing=True
    )

    assert output.read_bytes() == b"segmented"


def test_reuse_is_off_unless_asked_for(tmp_path, monkeypatch):
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"made earlier")
    _age(image, 100)
    _age(project, 100)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    run_ilastik_headless_segmentation(
        image, project, output, ilastik_executable=_fake_ilastik(tmp_path)
    )

    assert output.read_bytes() == b"segmented"


def test_ilastik_output_is_up_to_date_needs_every_source_older(tmp_path):
    from haemolynx.io.ilastik import ilastik_output_is_up_to_date

    image, project, output = _ilastik_inputs(tmp_path)
    assert not ilastik_output_is_up_to_date(output, image, project)

    output.parent.mkdir(parents=True)
    output.write_bytes(b"x")
    _age(image, 100)
    _age(project, 100)
    assert ilastik_output_is_up_to_date(output, image, project)

    _age(output, 200)
    assert not ilastik_output_is_up_to_date(output, image, project)
    assert not ilastik_output_is_up_to_date(output, tmp_path / "missing.tif")
