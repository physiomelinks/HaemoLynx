"""``use_memmap_loading`` changes where a volume lives, not what it contains.

Every test here compares a memmap-backed result against the ordinary
in-RAM one on the same small fixture, so the invariant these all pin is:
turning the toggle on must never change a single value it returns, only
back it with a disk-backed ``numpy.memmap`` instead of a plain array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.io import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    load_and_skeletonize_3d_tif,
)
from haemolynx.io.load import _skeletonize_loaded_volume
from haemolynx.preprocessing import bridge_gaps, fill_binary_holes, preprocess_skeleton_for_graph
from haemolynx.preprocessing.skeleton import MAX_BALL_DILATION_RADIUS


def _random_volume(shape=(6, 7, 8), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


# --- io.load: TIFF -----------------------------------------------------------


def test_tiff_memmap_loading_returns_a_memmap_with_identical_pixels(tmp_path):
    path = tmp_path / "volume.tif"
    raw = _random_volume()
    tifffile.imwrite(path, raw)

    memmap_image, mx, my, mz, m_status = load_3d_tif_with_voxel_size(
        str(path), use_memmap=True
    )
    eager_image, ex, ey, ez, e_status = load_3d_tif_with_voxel_size(
        str(path), use_memmap=False
    )

    assert isinstance(memmap_image, np.memmap)
    assert not isinstance(eager_image, np.memmap)
    assert np.array_equal(memmap_image, eager_image)
    assert (mx, my, mz) == (ex, ey, ez)
    assert m_status == e_status


def test_tiff_memmap_loading_honours_memmap_directory(tmp_path):
    """Not just 'still a memmap' -- the backing file must actually land in
    the requested directory, tifffile's own 'memmap:<dir>' string syntax,
    not silently fall back to the OS temp directory."""
    path = tmp_path / "volume.tif"
    tifffile.imwrite(path, _random_volume())
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    image, *_ = load_3d_tif_with_voxel_size(
        str(path), use_memmap=True, memmap_directory=custom_dir
    )

    assert isinstance(image, np.memmap)
    assert Path(image.filename).parent == custom_dir


def test_tiff_memmap_loading_works_on_a_compressed_file(tmp_path):
    """The real-world case this exists for: a well-compressed file still
    has to decompress to its full shape somewhere -- memmap just moves
    where that "somewhere" is."""
    path = tmp_path / "compressed.tif"
    raw = np.zeros((8, 40, 40), dtype=np.uint8)
    raw[3:5, 10:30, 10:30] = 255
    tifffile.imwrite(path, raw, compression="zlib")

    image, *_ = load_3d_tif_with_voxel_size(str(path), use_memmap=True)

    assert isinstance(image, np.memmap)
    assert np.array_equal(image, raw)


# --- io.load: H5 ---------------------------------------------------------


def test_h5_memmap_loading_returns_a_memmap_with_identical_pixels(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "volume.h5"
    raw = _random_volume()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=raw)

    memmap_image, mx, my, mz, m_status = load_3d_h5_with_voxel_size(
        str(path), "data", use_memmap=True
    )
    eager_image, ex, ey, ez, e_status = load_3d_h5_with_voxel_size(
        str(path), "data", use_memmap=False
    )

    assert isinstance(memmap_image, np.memmap)
    assert not isinstance(eager_image, np.memmap)
    assert np.array_equal(memmap_image, eager_image)
    assert (mx, my, mz) == (ex, ey, ez)
    assert m_status == e_status


def test_h5_memmap_loading_honours_memmap_directory(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "volume.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=_random_volume())
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    image, *_ = load_3d_h5_with_voxel_size(
        str(path), "data", use_memmap=True, memmap_directory=custom_dir
    )

    assert isinstance(image, np.memmap)
    assert Path(image.filename).parent == custom_dir


# --- io.load._skeletonize_loaded_volume / load_and_skeletonize_3d_tif --------


def test_skeletonize_loaded_volume_gives_the_same_skeleton_with_memmap_on():
    mask = np.zeros((6, 20, 20), dtype=bool)
    mask[2:4, 5:15, 5:15] = True

    memmap_skeleton = _skeletonize_loaded_volume(mask, use_memmap=True)
    eager_skeleton = _skeletonize_loaded_volume(mask, use_memmap=False)

    assert np.array_equal(memmap_skeleton, eager_skeleton)


def test_load_and_skeletonize_3d_tif_gives_the_same_result_with_memmap_on(tmp_path):
    path = tmp_path / "volume.tif"
    raw = np.zeros((6, 20, 20), dtype=np.uint8)
    raw[2:4, 5:15, 5:15] = 255
    tifffile.imwrite(path, raw)

    (memmap_image, memmap_skeleton, *_rest) = load_and_skeletonize_3d_tif(
        str(path), use_memmap=True
    )
    (eager_image, eager_skeleton, *_rest) = load_and_skeletonize_3d_tif(
        str(path), use_memmap=False
    )

    assert isinstance(memmap_image, np.memmap)
    assert np.array_equal(memmap_image, eager_image)
    assert np.array_equal(memmap_skeleton, eager_skeleton)


def test_load_and_skeletonize_3d_tif_honours_memmap_directory(tmp_path):
    path = tmp_path / "volume.tif"
    raw = np.zeros((6, 20, 20), dtype=np.uint8)
    raw[2:4, 5:15, 5:15] = 255
    tifffile.imwrite(path, raw)
    custom_dir = tmp_path / "custom_memmap_dir"

    image, *_rest = load_and_skeletonize_3d_tif(
        str(path), use_memmap=True, memmap_directory=custom_dir
    )

    assert isinstance(image, np.memmap)
    assert Path(image.filename).parent == custom_dir


# --- preprocessing.skeleton: fill_binary_holes / bridge_gaps -----------------


def test_fill_binary_holes_gives_the_same_result_with_memmap_on():
    # A hollow shell: the interior is background enclosed by foreground, the
    # exact case fill_binary_holes exists to close.
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False

    memmap_result = fill_binary_holes(mask, use_memmap=True)
    eager_result = fill_binary_holes(mask, use_memmap=False)

    assert np.array_equal(memmap_result, eager_result)
    assert memmap_result[5, 5, 5], "the enclosed cavity should have been filled"


def test_fill_binary_holes_with_memmap_on_leaves_no_backing_file_behind():
    import glob
    import tempfile

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    mask[2, 2, 2] = False

    before = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))
    fill_binary_holes(mask, use_memmap=True)
    after = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))

    assert after - before == set()


def test_fill_binary_holes_honours_memmap_directory_and_cleans_up_after(tmp_path):
    """fill_binary_holes releases its memmap before returning, so there is no
    handle left to inspect afterwards -- but the custom directory getting
    created at all (new_memmap_array's own side effect) proves the parameter
    actually reached the allocation, and it being empty afterwards proves
    the transient file was still cleaned up."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    fill_binary_holes(mask, use_memmap=True, memmap_directory=custom_dir)

    assert custom_dir.is_dir()
    assert list(custom_dir.iterdir()) == []


def test_bridge_gaps_distance_transform_path_gives_the_same_result_with_memmap_on():
    arr = np.zeros((5, 15, 15), dtype=bool)
    arr[2, 2, :] = True
    gap = MAX_BALL_DILATION_RADIUS + 2
    arr[2, 2, 5:5 + gap] = False

    memmap_result = bridge_gaps(arr, max_gap=gap, use_memmap=True)
    eager_result = bridge_gaps(arr, max_gap=gap, use_memmap=False)

    assert np.array_equal(memmap_result, eager_result)
    assert memmap_result[2, 2, 5:5 + gap].all(), "the gap should have been bridged"


def test_bridge_gaps_dilation_path_ignores_use_memmap():
    """max_gap at or below MAX_BALL_DILATION_RADIUS never reaches the
    distance-transform code at all, so the flag has nothing to do there."""
    arr = np.zeros((5, 5, 5), dtype=bool)
    arr[2, 2, :] = True
    arr[2, 2, 2] = False

    memmap_result = bridge_gaps(arr, max_gap=1, use_memmap=True)
    eager_result = bridge_gaps(arr, max_gap=1, use_memmap=False)

    assert np.array_equal(memmap_result, eager_result)


def test_preprocess_skeleton_for_graph_gives_the_same_result_with_memmap_on():
    skeleton = np.zeros((5, 15, 15), dtype=bool)
    skeleton[2, 2, :] = True
    gap = MAX_BALL_DILATION_RADIUS + 2
    skeleton[2, 2, 5:5 + gap] = False

    memmap_result = preprocess_skeleton_for_graph(
        skeleton, bridge_gap_size=gap, max_bridge_distance=0, use_memmap=True
    )
    eager_result = preprocess_skeleton_for_graph(
        skeleton, bridge_gap_size=gap, max_bridge_distance=0, use_memmap=False
    )

    assert np.array_equal(memmap_result, eager_result)
