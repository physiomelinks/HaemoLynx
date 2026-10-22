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
# Like ilastik it writes a Simple Segmentation: 1 the vessel, 2 the rest, with
# the vessel the *majority* (three quarters of the voxels), which is where a
# guess at the vessel label goes wrong.

_FAKE_ILASTIK = r"""
import json, os, sys, time
import numpy as np
mode = os.environ.get("FAKE_ILASTIK_MODE", "ok")
pattern = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_filename_format="))
image = sys.argv[-1]
nickname = os.path.splitext(os.path.basename(image))[0]
if os.environ.get("FAKE_ILASTIK_ARGS"):
    with open(os.environ["FAKE_ILASTIK_ARGS"], "w") as f:
        json.dump(sys.argv[1:], f)
if os.environ.get("FAKE_ILASTIK_RUNS"):
    with open(os.environ["FAKE_ILASTIK_RUNS"], "a") as f:
        f.write(".")
print("fake ilastik starting", flush=True)
if mode in ("ok", "one_class"):
    print("predicting 50%", flush=True)
    labels = np.full((4, 6, 8), 2, dtype=np.uint8)
    if mode == "ok":
        labels[:3] = 1
    out = pattern.replace("{nickname}", nickname)
    if out.endswith(".h5"):
        import h5py
        with h5py.File(out, "w") as h:
            d = h.create_dataset("exported_data", data=labels[..., None])
            d.attrs["axistags"] = "zyxc axistags"
    else:
        import tifffile
        tifffile.imwrite(out, labels, ome=True, metadata={"axes": "ZYX"}, software="ilastik")
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
"""

#: The voxels the fake writes, and how many of them are its vessel label 1.
_FAKE_VOXELS = 4 * 6 * 8
_FAKE_VESSEL_VOXELS = 3 * 6 * 8


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


def _ilastik_project(path, label_names=("Vessel", "Background")):
    """A file built like an ilastik pixel-classification project, as far as
    its label names go."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as project:
        project.create_dataset(
            "PixelClassification/LabelNames",
            data=[name.encode() for name in label_names],
            dtype=h5py.string_dtype(),
        )
    return path


def _ilastik_inputs(tmp_path, suffix=".tif"):
    image = tmp_path / "raw.tif"
    image.write_bytes(b"not a real tiff, never opened by this function")
    project = _ilastik_project(tmp_path / "classifier.ilp")
    return image, project, tmp_path / "out" / f"raw_segmented{suffix}"


def _read_output(path):
    if Path(path).suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as handle:
            return handle["exported_data"][()]
    import tifffile

    return tifffile.imread(path)


def _stopped_growing(path, *, settle=0.6):
    """Whether *path* stops changing: the process writing to it is dead."""
    import time

    time.sleep(settle)
    before = path.stat().st_size
    time.sleep(settle)
    return path.stat().st_size == before


def _segment(tmp_path, image, project, output, **kwargs):
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    kwargs.setdefault("vessel_label", "Vessel")
    return run_ilastik_headless_segmentation(
        image, project, output, ilastik_executable=_fake_ilastik(tmp_path), **kwargs
    )


@pytest.mark.parametrize("suffix", [".tif", ".h5"])
def test_ilastik_output_is_moved_and_binarised_at_the_vessel_label(tmp_path, monkeypatch, suffix):
    """Regression: a {1, 2} export was read with its minority as vessel, so an
    output where vessels are the majority came back inside out."""
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    image, project, output = _ilastik_inputs(tmp_path, suffix)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    produced = _segment(tmp_path, image, project, output)

    assert produced == output
    assert sorted(path.name for path in output.parent.iterdir()) == [
        output.name, output.name + ".haemolynx.json"
    ]
    written = _read_output(output)
    assert set(np.unique(written).tolist()) == {0, 1}
    mask = _to_binary_volume_for_skeletonization(np.squeeze(written))
    assert int(mask.sum()) == _FAKE_VESSEL_VOXELS
    assert mask[:3].all() and not mask[3:].any()


def test_the_vessel_label_may_be_the_second_label_by_name_or_number(tmp_path, monkeypatch):
    image, project, output = _ilastik_inputs(tmp_path)
    _ilastik_project(project, ("Background", "Vessel"))
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    _segment(tmp_path, image, project, output, vessel_label="vessel")
    by_name = _read_output(output)
    _segment(tmp_path, image, project, output, vessel_label=2)

    assert int(by_name.sum()) == _FAKE_VOXELS - _FAKE_VESSEL_VOXELS
    np.testing.assert_array_equal(_read_output(output), by_name)


def test_ilastik_is_asked_for_the_format_its_file_name_says_and_never_saves_the_project(
    tmp_path, monkeypatch
):
    """Without --output_format ilastik writes the project's saved export format
    (HDF5 unless changed), and it gives the file that format's extension
    whatever it was asked for (a multipage TIFF is always .tiff); without
    --readonly a headless run may save the project, changing the bytes its
    outputs are recorded by."""
    import json

    image, project, output = _ilastik_inputs(tmp_path)
    arguments = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")
    monkeypatch.setenv("FAKE_ILASTIK_ARGS", str(arguments))

    _segment(tmp_path, image, project, output)

    passed = json.loads(arguments.read_text())
    assert "--readonly" in passed
    assert "--output_format=multipage tiff" in passed
    assert "--export_source=Simple Segmentation" in passed
    (pattern,) = [a for a in passed if a.startswith("--output_filename_format=")]
    assert pattern.endswith("{nickname}.tiff")


def test_an_output_with_no_voxel_of_the_vessel_label_raises_and_is_left_as_written(
    tmp_path, monkeypatch
):
    """Regression: a one-class output (2 everywhere) read as solid vessel."""
    from haemolynx.io.ilastik import sidecar_path

    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "one_class")

    with pytest.raises(ValueError, match="no voxel of the vessel label 1"):
        _segment(tmp_path, image, project, output)

    assert (_read_output(output) == 2).all()
    assert not sidecar_path(output).exists()


def test_an_unknown_vessel_label_is_refused_before_ilastik_runs(tmp_path, monkeypatch):
    import subprocess

    image, project, output = _ilastik_inputs(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("ilastik was started with a label the project lacks")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    for label in (None, "Lumen", 3, 0):
        with pytest.raises(
            ValueError, match=r"ilastik_vessel_label.*1 \('Vessel'\), 2 \('Background'\)"
        ):
            _segment(
                tmp_path, image, project, output,
                vessel_label=label, label_setting="ilastik_vessel_label",
            )


def test_ilastik_output_lines_reach_the_log_as_they_arrive(tmp_path, monkeypatch, caplog):
    """A long segmentation shows its progress, not nothing until it ends."""
    import logging

    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")

    with caplog.at_level(logging.INFO, logger="haemolynx.io.ilastik"):
        _segment(tmp_path, image, project, output)

    logged = [record.getMessage() for record in caplog.records]
    assert "[ilastik] fake ilastik starting" in logged
    assert "[ilastik] predicting 50%" in logged
    assert any("binarised at vessel label 1 ('Vessel'): 75.0%" in line for line in logged)


def test_ilastik_timeout_kills_the_process_and_raises_an_actionable_error(tmp_path, monkeypatch):
    """A hung ilastik is killed -- the whole tree -- not left running."""
    image, project, output = _ilastik_inputs(tmp_path)
    beat = tmp_path / "beat.txt"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "hang")
    monkeypatch.setenv("FAKE_ILASTIK_BEAT", str(beat))

    with pytest.raises(RuntimeError, match="ilastik_timeout_seconds"):
        _segment(tmp_path, image, project, output, timeout=1.5)

    assert beat.exists()
    assert _stopped_growing(beat)


def test_a_poll_that_raises_stops_ilastik_and_propagates(tmp_path, monkeypatch):
    """How the panel's stop reaches a segmentation that can take hours."""
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
        _segment(tmp_path, image, project, output, timeout=60.0, poll=poll)

    assert len(polls) == 3
    assert _stopped_growing(beat)


def test_a_failing_ilastik_reports_its_exit_code_and_last_output(tmp_path, monkeypatch):
    image, project, output = _ilastik_inputs(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "fail")

    with pytest.raises(RuntimeError, match="exit code 3") as raised:
        _segment(tmp_path, image, project, output)

    assert "project file is broken" in str(raised.value)


def test_an_old_output_is_not_returned_when_ilastik_writes_nothing(tmp_path, monkeypatch):
    """Regression: a stale segmentation passed for this run's if ilastik exited
    cleanly without writing one."""
    image, project, output = _ilastik_inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"from an earlier run")
    # And under the name ilastik itself writes, before it is moved.
    (output.parent / "raw.tiff").write_bytes(b"from an earlier failed run")
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "silent")

    with pytest.raises(RuntimeError, match="no output file"):
        _segment(tmp_path, image, project, output)


def test_missing_ilastik_executable_names_the_real_setting(tmp_path):
    """Regression: this used to tell users to set an env var nothing reads.

    ``ILASTIK_EXECUTABLE`` is never read as an environment variable anywhere
    in this codebase -- the real knob is the ``ilastik_executable`` setting.
    """
    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="ilastik_executable") as raised:
        run_ilastik_headless_segmentation(
            image, project, output,
            ilastik_executable=tmp_path / "no_such_ilastik.exe", vessel_label=1,
        )

    assert "ILASTIK_EXECUTABLE" not in str(raised.value)


# --- reusing an output: the sidecar ------------------------------------------


def _age(path, seconds):
    import os
    import time

    then = time.time() - seconds
    os.utime(path, (then, then))


def _count_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs.txt"
    monkeypatch.setenv("FAKE_ILASTIK_MODE", "ok")
    monkeypatch.setenv("FAKE_ILASTIK_RUNS", str(runs))
    return lambda: len(runs.read_text()) if runs.exists() else 0


def test_an_output_made_by_this_project_is_reused_without_running_ilastik(tmp_path, monkeypatch):
    import json

    from haemolynx.io.ilastik import file_sha256, sidecar_path

    image, project, output = _ilastik_inputs(tmp_path)
    runs = _count_runs(tmp_path, monkeypatch)

    _segment(tmp_path, image, project, output, reuse_existing=True)
    made = output.read_bytes()
    _segment(tmp_path, image, project, output, reuse_existing=True)

    assert runs() == 1
    assert output.read_bytes() == made
    record = json.loads(sidecar_path(output).read_text())
    assert record["project_sha256"] == file_sha256(project)
    assert record["vessel_label"] == 1 and record["vessel_label_name"] == "Vessel"
    assert record["output"]["vessel_voxels"] == _FAKE_VESSEL_VOXELS


def test_a_project_copied_in_with_an_older_time_is_run_again(tmp_path, monkeypatch, caplog):
    """Regression: reuse was decided by file times, and copying a file keeps
    its modification time on Windows, so a colleague's project, older than
    the output, was never run."""
    import logging

    image, project, output = _ilastik_inputs(tmp_path)
    runs = _count_runs(tmp_path, monkeypatch)
    _segment(tmp_path, image, project, output, reuse_existing=True)

    _ilastik_project(project, ("Vessel", "Background", "Wall"))
    _age(project, 10_000)
    with caplog.at_level(logging.WARNING, logger="haemolynx.io.ilastik"):
        _segment(tmp_path, image, project, output, reuse_existing=True)

    assert runs() == 2
    assert any(
        "project is not the one it was made with" in record.getMessage()
        for record in caplog.records
    )


def test_the_same_project_touched_again_is_still_reused(tmp_path, monkeypatch):
    import os

    image, project, output = _ilastik_inputs(tmp_path)
    runs = _count_runs(tmp_path, monkeypatch)
    _segment(tmp_path, image, project, output, reuse_existing=True)

    os.utime(project)  # newer than the output, the same bytes
    _segment(tmp_path, image, project, output, reuse_existing=True)

    assert runs() == 1


@pytest.mark.parametrize("change", ["no sidecar", "image", "label", "output", "reuse off"])
def test_an_output_that_cannot_vouch_for_this_run_is_made_again(tmp_path, monkeypatch, change):
    from haemolynx.io.ilastik import sidecar_path

    image, project, output = _ilastik_inputs(tmp_path)
    runs = _count_runs(tmp_path, monkeypatch)
    _segment(tmp_path, image, project, output, reuse_existing=True)

    label = "Vessel"
    reuse = True
    if change == "no sidecar":
        sidecar_path(output).unlink()  # made before HaemoLynx recorded it
    elif change == "image":
        image.write_bytes(b"another raw stack")
    elif change == "label":
        label = "Background"
    elif change == "output":
        output.write_bytes(output.read_bytes() + b"\0")
    else:
        reuse = False
    _segment(tmp_path, image, project, output, reuse_existing=reuse, vessel_label=label)

    assert runs() == 2


def test_ilastik_output_is_current_says_why_not(tmp_path):
    from haemolynx.io.ilastik import ilastik_output_is_current

    current, reason = ilastik_output_is_current(tmp_path / "missing.tif", {})
    assert not current and reason == "there is no output yet"
    (tmp_path / "out.tif").write_bytes(b"x")
    current, reason = ilastik_output_is_current(tmp_path / "out.tif", {})
    assert not current and "out.tif.haemolynx.json" in reason


# --- the vessel label and the binarisation, without ilastik ------------------


def test_resolve_vessel_label_reads_names_then_numbers():
    from haemolynx.io.ilastik import resolve_vessel_label

    names = ["Background", "Vessel", "vessel wall"]
    assert resolve_vessel_label("Vessel", names) == 2
    assert resolve_vessel_label(" VESSEL WALL ", names) == 3
    assert resolve_vessel_label("1", names) == 1
    assert resolve_vessel_label(3, names) == 3
    assert resolve_vessel_label(7, None) == 7
    for bad in ("", None, "Lumen", 4, 0, -1, "2.5"):
        with pytest.raises(ValueError, match="my_label"):
            resolve_vessel_label(bad, names, setting="my_label")
    with pytest.raises(ValueError, match="a label number"):
        resolve_vessel_label("Vessel", None)


def test_resolve_vessel_label_prefers_an_exact_name_over_a_number():
    from haemolynx.io.ilastik import resolve_vessel_label

    assert resolve_vessel_label("2", ["2", "1"]) == 1


def test_label_names_are_read_from_the_project_or_none(tmp_path):
    from haemolynx.io.ilastik import ilastik_label_names

    assert ilastik_label_names(_ilastik_project(tmp_path / "p.ilp", ("A", "B"))) == ["A", "B"]
    (tmp_path / "junk.ilp").write_bytes(b"not hdf5")
    assert ilastik_label_names(tmp_path / "junk.ilp") is None
    assert ilastik_label_names(tmp_path / "missing.ilp") is None


def test_a_compressed_tiff_is_binarised_with_its_axes_kept(tmp_path):
    import tifffile

    from haemolynx.io.ilastik import binarise_ilastik_output

    labels = np.full((3, 4, 5), 1, dtype=np.uint8)
    labels[0] = 2
    path = tmp_path / "seg.tif"
    tifffile.imwrite(
        path, labels, compression="zlib", photometric="minisblack", metadata={"axes": "ZYX"}
    )

    assert binarise_ilastik_output(path, 1) == (40, 60)
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "ZYX"
        written = tif.asarray()
    assert written.dtype == np.uint8
    np.testing.assert_array_equal(written, (labels == 1).astype(np.uint8))


def test_an_h5_output_keeps_its_dataset_and_attributes(tmp_path):
    h5py = pytest.importorskip("h5py")
    from haemolynx.io.ilastik import binarise_ilastik_output

    path = tmp_path / "seg.h5"
    labels = np.array([[[1, 2], [2, 3]]], dtype=np.uint8)[..., None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exported_data", data=labels).attrs["axistags"] = "tags"

    assert binarise_ilastik_output(path, 2) == (2, 4)
    with h5py.File(path, "r") as handle:
        assert handle["exported_data"].attrs["axistags"] == "tags"
        np.testing.assert_array_equal(handle["exported_data"][()], (labels == 2).astype(np.uint8))


def test_the_minority_guess_for_a_mask_of_two_labels_is_warned_about_once(caplog):
    import logging

    from haemolynx.io import load

    load._MINORITY_GUESSES_WARNED.clear()
    mask = np.full((4, 4, 4), 2, dtype=np.uint8)
    mask[0] = 1
    with caplog.at_level(logging.WARNING, logger="haemolynx.io.load"):
        first = load._to_binary_volume_for_skeletonization(mask)
        load._to_binary_volume_for_skeletonization(mask)
        load._to_binary_volume_for_skeletonization((mask == 1).astype(np.uint8))

    guesses = [r.getMessage() for r in caplog.records if "a guess" in r.getMessage()]
    assert len(guesses) == 1
    assert "1 (25.0%), 2 (75.0%)" in guesses[0] and "rarest, 1," in guesses[0]
    assert int(first.sum()) == 16


# --- H5 exports with a channel axis (ilastik) -------------------------------


def _ilastik_axistags(keys: str) -> str:
    import json

    return json.dumps({
        "axes": [
            {"key": key, "typeFlags": 1 if key == "c" else 2, "resolution": 0, "description": ""}
            for key in keys
        ]
    })


def _ilastik_segmentation(shape_zyx=(9, 11, 15)) -> np.ndarray:
    """A Simple Segmentation's labels: 1 the vessel (the minority, a rod 5
    voxels across along x), 2 the rest."""
    labels = np.full(shape_zyx, 2, dtype=np.uint8)
    z, y, _x = (n // 2 for n in shape_zyx)
    labels[max(z - 2, 0):z + 3, max(y - 2, 0):y + 3, 2:-2] = 1
    return labels


def _write_h5(path, data, *, axistags=None, dimension_labels=None, chunks=None, **attrs):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("exported_data", data=data, chunks=chunks)
        if axistags is not None:
            dataset.attrs["axistags"] = axistags
        for axis, label in enumerate(dimension_labels or ()):
            dataset.dims[axis].label = label
        for key, value in attrs.items():
            dataset.attrs[key] = value
    return path


@pytest.mark.parametrize("use_memmap", [False, True])
def test_h5_loader_drops_an_ilastik_exports_single_channel_axis(tmp_path, use_memmap):
    """Regression: an ilastik Simple Segmentation exported as H5 is ``zyxc``
    with one channel, and the loader stopped the run with "Expected 3D image
    after simplification, got shape: (z, y, x, 1)"."""
    from haemolynx.io import load_3d_h5_with_voxel_size

    labels = _ilastik_segmentation()
    path = _write_h5(
        tmp_path / "vessels_Simple Segmentation.h5",
        labels[..., np.newaxis],
        axistags=_ilastik_axistags("zyxc"),
        dimension_labels="zyxc",
        chunks=(2, 9, 11, 1),
    )

    image, *_ = load_3d_h5_with_voxel_size(
        str(path), use_memmap=use_memmap, memmap_directory=tmp_path / "memmap"
    )

    assert image.shape == labels.shape
    assert np.array_equal(image, labels)
    assert isinstance(image, np.memmap) == use_memmap


def test_h5_ilastik_export_reaches_the_pipeline_as_its_vessel_mask(tmp_path):
    """The run's own read (the 2D-aware loader) and the skeletoniser's both
    get the (z, y, x) labels, and the vessel label is what is foreground."""
    from haemolynx.io import load_and_skeletonize_3d_h5, load_image_with_voxel_size_2d_aware

    labels = _ilastik_segmentation()
    path = _write_h5(
        tmp_path / "vessels.h5", labels[..., np.newaxis], axistags=_ilastik_axistags("zyxc")
    )

    image, *_rest, status = load_image_with_voxel_size_2d_aware(str(path), input_format="h5")
    assert image.shape == labels.shape
    assert "promoted_from_2d" not in status

    loaded, skeleton, *_ = load_and_skeletonize_3d_h5(str(path))
    assert loaded.shape == skeleton.shape == labels.shape
    assert skeleton.any()
    assert not skeleton[labels == 2].any()


def test_h5_loader_drops_a_trailing_singleton_axis_without_axis_names(tmp_path):
    from haemolynx.io import load_3d_h5_with_voxel_size

    labels = _ilastik_segmentation()
    path = _write_h5(tmp_path / "untagged.h5", labels[..., np.newaxis])

    image, *_ = load_3d_h5_with_voxel_size(str(path))

    assert np.array_equal(image, labels)


@pytest.mark.parametrize("use_memmap", [False, True])
def test_h5_loader_drops_a_leading_channel_axis_and_its_element_size(tmp_path, use_memmap):
    """A ``czyx`` export loses axis 0, and ``element_size_um``'s entry for it."""
    from haemolynx.io import load_3d_h5_with_voxel_size

    labels = _ilastik_segmentation()
    path = _write_h5(
        tmp_path / "czyx.h5",
        labels[np.newaxis],
        axistags=_ilastik_axistags("czyx"),
        chunks=(1, 2, 9, 11),
        element_size_um=np.array([1.0, 2.0, 0.5, 0.25]),
    )

    image, x, y, z, status = load_3d_h5_with_voxel_size(
        str(path), use_memmap=use_memmap, memmap_directory=tmp_path / "memmap"
    )

    assert np.array_equal(image, labels)
    assert (x, y, z) == (0.25, 0.5, 2.0)
    assert status["status"] == "complete"


def test_h5_loader_keeps_a_named_z_axis_of_one_slice(tmp_path):
    """Only channel and time axes are dropped: a one-slice ``zyx`` stack stays 3D."""
    from haemolynx.io import load_3d_h5_with_voxel_size

    data = _ilastik_segmentation((1, 9, 11))
    path = _write_h5(tmp_path / "one_slice.h5", data, axistags=_ilastik_axistags("zyx"))

    image, *_ = load_3d_h5_with_voxel_size(str(path))

    assert image.shape == (1, 9, 11)


def test_h5_loader_promotes_a_2d_ilastik_export_through_the_2d_aware_loader(tmp_path):
    from haemolynx.io import load_image_with_voxel_size_2d_aware

    plane = _ilastik_segmentation((1, 9, 11))[0]
    path = _write_h5(
        tmp_path / "plane.h5", plane[..., np.newaxis], axistags=_ilastik_axistags("yxc")
    )

    image, *_rest, status = load_image_with_voxel_size_2d_aware(str(path), input_format="h5")

    assert image.shape == (1, 9, 11)
    assert np.array_equal(image[0], plane)
    assert status["promoted_from_2d"] is True


def test_h5_loader_still_rejects_several_channels(tmp_path):
    """A Probabilities export has one channel per label: which is the vessel
    is not the loader's to guess."""
    from haemolynx.io import load_3d_h5_with_voxel_size

    probabilities = np.zeros((6, 9, 11, 2), dtype=np.float32)
    path = _write_h5(
        tmp_path / "probabilities.h5", probabilities, axistags=_ilastik_axistags("zyxc")
    )

    with pytest.raises(ValueError, match="Simple Segmentation"):
        load_3d_h5_with_voxel_size(str(path))
