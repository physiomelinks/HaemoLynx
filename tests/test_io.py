"""Tests for io module."""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from ImageLynx.io import (
    crop_tiff_volume_from_corners,
    load_and_skeletonize_3d_tif,
    load_and_skeletonize_3d_h5,
    bridge_gaps,
    simplify_to_3d,
    get_tif_spacing
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
    image, skeleton, vx, vy, vz = load_and_skeletonize_3d_tif(str(f))
    assert image.shape == (6, 6, 6)
    assert skeleton.shape == (6, 6, 6)
    assert skeleton.dtype == bool


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


def test_get_tif_spacing(tmp_path):
    import tifffile
    f = tmp_path / "test_spacing.tif"
    img = np.zeros((2, 2, 2), dtype=np.uint8)
    
    # Test setting specific spacing (Z, Y, X) = (2.0, 0.5, 0.5)
    # tifffile uses resolution = (1/X, 1/Y) in cm/inch etc.
    # spacing tag is for Z.
    tifffile.imwrite(
        f,
        img,
        resolution=(2.0, 2.0), # 1/0.5
        metadata={'spacing': 2.0},
        imagej=True
    )
    
    spacing = get_tif_spacing(str(f))
    assert spacing == (2.0, 0.5, 0.5)
    
    # Test fallback to defaults
    f2 = tmp_path / "test_no_spacing.tif"
    tifffile.imwrite(f2, img)
    spacing_default = get_tif_spacing(str(f2))
    assert spacing_default == (1.0, 1.0, 1.0)


# --- Reading the Ilastik probability export (#98 Phase A) ---------------------------------
#
# The headless export is a 4-D HDF5 with one probability channel per class, in label order.
# Picking the wrong channel does not fail: it returns the background probability, and every
# downstream number is then computed from the inverse segmentation.

@pytest.fixture
def probability_h5(tmp_path):
    """A 2-class export shaped the way ilastik writes one: (z, y, x, c), float32."""
    h5py = pytest.importorskip("h5py")

    vessel = np.zeros((8, 12, 12), dtype=np.float32)
    vessel[:, 5:7, 5:7] = 0.9                     # a thin tube, a few percent of the volume
    stack = np.stack([vessel, 1.0 - vessel], axis=-1).astype(np.float32)

    path = tmp_path / "VOL_ilastik_Probabilities.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exported_data", data=stack)
    return path, vessel


def test_reads_the_named_vessel_class_channel(probability_h5):
    from ImageLynx.io import read_ilastik_probabilities

    path, vessel = probability_h5
    got = read_ilastik_probabilities(path, vessel_class_index=0)

    assert got.shape == vessel.shape
    assert got.dtype == np.float32
    assert got == pytest.approx(vessel)


def test_refuses_to_guess_the_vessel_class(probability_h5, monkeypatch):
    """There is no safe default when the trained project's label order is unrecorded.

    It is recorded now - LabelNames are ['vessel', 'background'], so vessel is channel 0 -
    but the refusal is what protects the next classifier, whose label order is whatever it
    was clicked in.
    """
    import ImageLynx.specimens as specimens
    from ImageLynx.io import read_ilastik_probabilities

    monkeypatch.setattr(specimens, "VESSEL_CLASS_INDEX", None)
    path, _ = probability_h5
    with pytest.raises(ValueError, match="vessel class index is not recorded"):
        read_ilastik_probabilities(path)


def test_uses_the_recorded_vessel_class_by_default(probability_h5):
    """With the index recorded, reading needs no argument - and gets channel 0."""
    from ImageLynx.io import read_ilastik_probabilities

    path, vessel = probability_h5
    assert read_ilastik_probabilities(path) == pytest.approx(vessel)


def test_rejects_the_inverse_segmentation(probability_h5):
    """Mean vessel probability is a few percent. Near 1 - expected means the wrong channel.

    This is the one failure mode of the whole chain that produces a complete, plausible set
    of results rather than an error, so it has to be caught where the data is read.
    """
    from ImageLynx.io import read_ilastik_probabilities

    path, _ = probability_h5
    with pytest.raises(ValueError, match="background"):
        read_ilastik_probabilities(path, vessel_class_index=1)

    # Still reachable deliberately, for anyone who really does want the other class.
    got = read_ilastik_probabilities(path, vessel_class_index=1, check_calibration=False)
    assert got.mean() > 0.9


def test_names_the_available_datasets_when_the_export_name_is_wrong(probability_h5):
    from ImageLynx.io import read_ilastik_probabilities

    path, _ = probability_h5
    with pytest.raises(KeyError, match="exported_data"):
        read_ilastik_probabilities(path, vessel_class_index=0, dataset="probabilities")


def test_rejects_a_volume_with_no_class_axis(tmp_path):
    """A 3-D export means the class axis was already collapsed, so which class is unknowable."""
    h5py = pytest.importorskip("h5py")
    from ImageLynx.io import read_ilastik_probabilities

    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exported_data", data=np.zeros((8, 12, 12), dtype=np.float32))

    with pytest.raises(ValueError, match="class axis"):
        read_ilastik_probabilities(path, vessel_class_index=0)


def test_checks_the_volume_against_the_expected_shape(probability_h5):
    """Catches a probability map paired with the wrong specimen's registry entry."""
    from ImageLynx.io import read_ilastik_probabilities

    path, _ = probability_h5
    with pytest.raises(ValueError, match="shape"):
        read_ilastik_probabilities(path, vessel_class_index=0, expected_shape_zyx=(435, 456, 507))
