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


# --- run_ilastik_headless_segmentation: timeout and error messages ---------


def _ilastik_inputs(tmp_path):
    image = tmp_path / "raw.tif"
    image.write_bytes(b"not a real tiff, never opened by this function")
    project = tmp_path / "classifier.ilp"
    project.write_bytes(b"not a real project, never opened by this function")
    return image, project, tmp_path / "out" / "raw_segmented.tif"


def test_ilastik_timeout_is_passed_to_the_subprocess(tmp_path, monkeypatch):
    """A hung ilastik process must be killable, not block the caller forever."""
    import subprocess

    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)
    seen_kwargs = {}

    def fake_run(command, **kwargs):
        seen_kwargs.update(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"segmented")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_ilastik_headless_segmentation(image, project, output, timeout=123.0)

    assert seen_kwargs.get("timeout") == 123.0


def test_ilastik_timeout_expiring_raises_an_actionable_error(tmp_path, monkeypatch):
    """Regression: an unhandled ``TimeoutExpired`` used to propagate as-is."""
    import subprocess

    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ilastik_timeout_seconds"):
        run_ilastik_headless_segmentation(image, project, output, timeout=5.0)


def test_missing_ilastik_executable_names_the_real_setting(tmp_path, monkeypatch):
    """Regression: this used to tell users to set an env var nothing reads.

    ``ILASTIK_EXECUTABLE`` is never read as an environment variable anywhere
    in this codebase -- the real knob is the ``ilastik_executable`` setting.
    """
    import subprocess

    from haemolynx.io.ilastik import run_ilastik_headless_segmentation

    image, project, output = _ilastik_inputs(tmp_path)

    def fake_run(command, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="ilastik_executable") as raised:
        run_ilastik_headless_segmentation(image, project, output)

    assert "ILASTIK_EXECUTABLE" not in str(raised.value)
