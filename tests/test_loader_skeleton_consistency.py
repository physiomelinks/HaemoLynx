"""The same volume must skeletonize the same way whatever file it arrived in.

``load_and_skeletonize_3d_tif`` filled holes in the skeleton and
``load_and_skeletonize_3d_h5`` did not, so saving one volume as TIFF and as H5
and loading both gave two different skeletons — and therefore two different
graphs, resistances and statistics — with nothing in the run to say why.

The hole fill was added to the TIFF path deliberately (issue #4, skeletonization
quality) and everything downstream has been tuned against skeletons that have
it, so H5 was the path that was wrong. It also costs nothing in the normal
case: a thin 3D curve encloses no background, which
``tests/test_pipeline_performance.py`` pins separately.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.io import load_and_skeletonize_3d_h5, load_and_skeletonize_3d_tif
from haemolynx.preprocessing import fill_binary_holes

H5_DATASET_NAME = "data"


def _hollow_shell() -> np.ndarray:
    """A shell around a sealed cavity: its skeleton is a closed surface.

    Chosen so the hole fill actually does something — a straight vessel would
    make the comparison vacuous, since a curve skeleton encloses nothing and
    the two loaders would have agreed even while they disagreed in code.
    """
    volume = np.zeros((21, 21, 21), dtype=np.uint8)
    volume[4:17, 4:17, 4:17] = 1
    volume[7:14, 7:14, 7:14] = 0
    return volume


def _straight_vessel() -> np.ndarray:
    """The pipeline's real case: a thick tube whose skeleton is a curve."""
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    volume[6:10, 6:10, 2:14] = 1
    return volume


def _write_pair(tmp_path, volume: np.ndarray):
    h5py = pytest.importorskip("h5py")
    tif_path = tmp_path / "volume.tif"
    h5_path = tmp_path / "volume.h5"
    tifffile.imwrite(str(tif_path), volume)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset(H5_DATASET_NAME, data=volume)
    return tif_path, h5_path


def _both_skeletons(tmp_path, volume: np.ndarray):
    tif_path, h5_path = _write_pair(tmp_path, volume)
    _image, tif_skeleton, *_rest = load_and_skeletonize_3d_tif(str(tif_path))
    _image, h5_skeleton, *_rest = load_and_skeletonize_3d_h5(
        str(h5_path), dataset_name=H5_DATASET_NAME
    )
    return tif_skeleton, h5_skeleton


@pytest.mark.parametrize(
    "volume, label",
    [(_hollow_shell(), "hollow shell"), (_straight_vessel(), "straight vessel")],
    ids=["hollow_shell", "straight_vessel"],
)
def test_tiff_and_h5_loaders_return_the_same_skeleton(tmp_path, volume, label):
    tif_skeleton, h5_skeleton = _both_skeletons(tmp_path, volume)

    assert tif_skeleton.dtype == bool
    assert h5_skeleton.dtype == bool
    differing = int(np.logical_xor(tif_skeleton, h5_skeleton).sum())
    assert differing == 0, f"{label}: {differing} skeleton voxels differ by file format"


def test_the_shell_fixture_is_a_case_the_hole_fill_changes(tmp_path):
    """Otherwise the comparison above would pass however the loaders behaved."""
    tif_skeleton, _h5_skeleton = _both_skeletons(tmp_path, _hollow_shell())

    from skimage.morphology import skeletonize

    unfilled = skeletonize(_hollow_shell().astype(bool), method="lee")
    assert not np.array_equal(unfilled, fill_binary_holes(unfilled))
    assert tif_skeleton.sum() > unfilled.sum()


def test_the_sparse_case_is_unchanged_by_the_hole_fill(tmp_path):
    """What the H5 path gains is consistency, not a different normal answer."""
    tif_skeleton, _h5_skeleton = _both_skeletons(tmp_path, _straight_vessel())

    from skimage.morphology import skeletonize

    unfilled = skeletonize(_straight_vessel().astype(bool), method="lee")
    assert np.array_equal(tif_skeleton, unfilled.astype(bool))
