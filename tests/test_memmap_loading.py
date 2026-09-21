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
from skimage.morphology import skeletonize

from haemolynx.io import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    load_and_skeletonize_3d_tif,
)
from haemolynx.io.load import _skeletonize_loaded_volume
from haemolynx.preprocessing import (
    bridge_gaps,
    drop_small_components,
    fill_binary_holes,
    preprocess_skeleton_for_graph,
    skeletonize_by_component,
)
from haemolynx.preprocessing.memmap_support import release_memmap_array
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

    assert isinstance(memmap_skeleton, np.memmap)
    assert not isinstance(eager_skeleton, np.memmap)
    assert np.array_equal(memmap_skeleton, eager_skeleton)
    release_memmap_array(memmap_skeleton)


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
    assert isinstance(memmap_skeleton, np.memmap)
    assert not isinstance(eager_skeleton, np.memmap)
    assert np.array_equal(memmap_image, eager_image)
    assert np.array_equal(memmap_skeleton, eager_skeleton)
    release_memmap_array(memmap_skeleton)


def test_load_and_skeletonize_3d_tif_honours_memmap_directory(tmp_path):
    path = tmp_path / "volume.tif"
    raw = np.zeros((6, 20, 20), dtype=np.uint8)
    raw[2:4, 5:15, 5:15] = 255
    tifffile.imwrite(path, raw)
    custom_dir = tmp_path / "custom_memmap_dir"

    image, skeleton, *_rest = load_and_skeletonize_3d_tif(
        str(path), use_memmap=True, memmap_directory=custom_dir
    )

    assert isinstance(image, np.memmap)
    assert Path(image.filename).parent == custom_dir
    assert isinstance(skeleton, np.memmap)
    assert Path(skeleton.filename).parent == custom_dir
    release_memmap_array(skeleton)


# --- preprocessing.skeleton: fill_binary_holes / bridge_gaps -----------------


def test_fill_binary_holes_gives_the_same_result_with_memmap_on():
    # A hollow shell: the interior is background enclosed by foreground, the
    # exact case fill_binary_holes exists to close.
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False

    memmap_result = fill_binary_holes(mask, use_memmap=True)
    eager_result = fill_binary_holes(mask, use_memmap=False)

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    assert memmap_result[5, 5, 5], "the enclosed cavity should have been filled"
    release_memmap_array(memmap_result)


def test_fill_binary_holes_with_memmap_on_returns_a_memmap_the_caller_must_release():
    """The *returned* mask is the whole point of use_memmap_loading reaching
    this far -- it is what preprocess_skeleton_for_graph, and eventually
    SkeletonisedVolume.skeleton, get handed next. Only the *intermediate*
    labelled-background buffer is fully cleaned up internally; this one is
    the caller's to release, the same convention SkeletonisedVolume.image
    and clean_segmented_mask_for_skeletonisation's raw output already use."""
    import glob
    import tempfile

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    mask[2, 2, 2] = False

    before = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))
    result = fill_binary_holes(mask, use_memmap=True)
    after = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))

    assert isinstance(result, np.memmap)
    assert len(after - before) == 1
    release_memmap_array(result)
    assert set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap")) == before


def test_fill_binary_holes_honours_memmap_directory(tmp_path):
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    result = fill_binary_holes(mask, use_memmap=True, memmap_directory=custom_dir)

    assert Path(result.filename).parent == custom_dir
    release_memmap_array(result)


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


# --- preprocessing.skeleton: drop_small_components ---------------------------


def test_drop_small_components_gives_the_same_result_with_memmap_on():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[2:8, 5:15, 5:15] = True  # a large component, kept
    mask[0, 0, 0] = True  # a lone voxel, dropped
    mask[0, 0, 2] = True
    mask[0, 0, 3] = True  # a 2-voxel component, dropped at min_size=3
    mask[0, 5, 0:3] = True  # exactly min_size voxels -- the threshold boundary

    memmap_result = drop_small_components(mask, min_size=3, use_memmap=True)
    eager_result = drop_small_components(mask, min_size=3, use_memmap=False)

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    assert not memmap_result[0, 0, 0], "the lone voxel should have been dropped"
    assert not memmap_result[0, 0, 2], "the 2-voxel component should have been dropped"
    assert memmap_result[5, 10, 10], "the large component should survive"
    # Not hard-coded to "removed" -- this pins agreement with whatever this
    # installed skimage version's own min_size threshold actually does at
    # the boundary (inclusive vs exclusive changed between skimage
    # releases, see drop_small_components's docstring), rather than an
    # assumption this test would need updating for on every skimage bump.
    from skimage.morphology import remove_small_objects

    boundary_component = np.zeros(3, dtype=bool)
    boundary_component[:] = True
    boundary_survives = remove_small_objects(boundary_component, min_size=3, connectivity=1).any()
    assert bool(memmap_result[0, 5, 0]) == boundary_survives
    release_memmap_array(memmap_result)


def test_drop_small_components_with_memmap_off_calls_skimage_directly():
    """The documented contract: use_memmap=False must be byte-for-byte
    skimage.morphology.remove_small_objects, not a reimplementation --
    this pins that no separate code path was accidentally used instead."""
    from skimage.morphology import remove_small_objects

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    mask[0, 0, 0] = True

    ours = drop_small_components(mask, min_size=4, connectivity=2)
    theirs = remove_small_objects(mask, min_size=4, connectivity=2)

    assert np.array_equal(ours, theirs)


def test_drop_small_components_honours_memmap_directory(tmp_path):
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    result = drop_small_components(mask, min_size=3, use_memmap=True, memmap_directory=custom_dir)

    assert Path(result.filename).parent == custom_dir
    release_memmap_array(result)


# --- preprocessing.skeleton: skeletonize_by_component -------------------------


def test_skeletonize_by_component_gives_the_same_result_with_memmap_on():
    mask = np.zeros((10, 30, 30), dtype=bool)
    mask[2:8, 5:15, 5:15] = True  # component A
    mask[2:8, 20:28, 5:15] = True  # component B, disconnected from A

    memmap_result = skeletonize_by_component(mask, use_memmap=True)
    eager_result = skeletonize_by_component(mask, use_memmap=False)
    monolithic = skeletonize(mask, method="lee")

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    assert np.array_equal(memmap_result, monolithic)
    release_memmap_array(memmap_result)


def test_skeletonize_by_component_with_memmap_off_calls_skimage_directly():
    """The documented contract: use_memmap=False must be byte-for-byte
    skimage.morphology.skeletonize, not a reimplementation."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    ours = skeletonize_by_component(mask)
    theirs = skeletonize(mask, method="lee")

    assert np.array_equal(ours, theirs)


def test_skeletonize_by_component_matches_monolithic_with_overlapping_bounding_boxes():
    """Two disconnected components whose bounding boxes overlap in space
    (an L-shape and a separate bar crossing through the L's bbox) --
    proves the per-component crop masks out the OTHER component's voxels
    rather than accidentally including them via a naive bbox-only crop."""
    mask = np.zeros((3, 20, 20), dtype=bool)
    mask[1, 2:15, 2] = True
    mask[1, 2, 2:15] = True  # together, an L-shaped component A
    mask[1, 17, 5:18] = True  # component B, bbox overlaps A's in y/x

    memmap_result = skeletonize_by_component(mask, use_memmap=True)

    assert np.array_equal(memmap_result, skeletonize(mask, method="lee"))
    release_memmap_array(memmap_result)


def test_skeletonize_by_component_with_memmap_on_handles_an_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=bool)

    result = skeletonize_by_component(mask, use_memmap=True)

    assert isinstance(result, np.memmap)
    assert not result.any()
    release_memmap_array(result)


def test_skeletonize_by_component_honours_memmap_directory(tmp_path):
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    result = skeletonize_by_component(mask, use_memmap=True, memmap_directory=custom_dir)

    assert Path(result.filename).parent == custom_dir
    release_memmap_array(result)


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
