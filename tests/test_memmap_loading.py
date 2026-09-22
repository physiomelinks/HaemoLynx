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
from haemolynx.io.load import (
    _skeletonize_loaded_volume,
    _to_binary_volume_for_skeletonization,
    _unique_with_counts,
)
from haemolynx.preprocessing import (
    bridge_gaps,
    connect_skeleton_components,
    drop_small_components,
    fill_binary_holes,
    preprocess_skeleton_for_graph,
    skeletonize_by_component,
)
from haemolynx.preprocessing.memmap_support import release_memmap_array
from haemolynx.preprocessing.skeleton import (
    MAX_BALL_DILATION_RADIUS,
    _filter_components_by_total_fraction,
)


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


# --- io.load._unique_with_counts / _to_binary_volume_for_skeletonization -----


class _RaisesIfFlattenedWhole(np.ndarray):
    """Proves a caller never sees this array as one flat/raveled unit.

    ``np.unique`` flattens its input via ``.flatten()`` before sorting it --
    for a ``use_memmap_loading`` volume, that flatten is a fresh, full-size,
    plain-RAM copy regardless of how the source pixels are stored, which is
    exactly the allocation ``_unique_with_counts`` exists to avoid. Slicing
    (``arr[index]``, the per-slice access the memmap-safe path uses) comes
    back as a plain ``np.ndarray``, so a per-slice ``np.unique`` call is
    unaffected; only asking this class to flatten *itself* raises.
    """

    def flatten(self, *args, **kwargs):
        raise AssertionError("flattened the whole array instead of per-slice access")

    def ravel(self, *args, **kwargs):
        raise AssertionError("raveled the whole array instead of per-slice access")

    def __getitem__(self, item):
        result = np.ndarray.__getitem__(self, item)
        return np.asarray(result) if isinstance(result, np.ndarray) else result


def _watched(arr: np.ndarray) -> _RaisesIfFlattenedWhole:
    return arr.view(_RaisesIfFlattenedWhole)


def test_unique_with_counts_matches_np_unique_with_memmap_off():
    arr = _random_volume(shape=(4, 5, 6), seed=1)
    values, counts = _unique_with_counts(arr, use_memmap=False)
    expected_values, expected_counts = np.unique(arr, return_counts=True)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)


def test_unique_with_counts_matches_np_unique_with_memmap_on():
    """Some values only appear in some slices, so this also pins that
    per-slice tallies are correctly merged, not just each slice's own."""
    arr = np.zeros((5, 4, 4), dtype=np.uint8)
    arr[0] = 1
    arr[1] = 2
    arr[2, 0, 0] = 3
    arr[3:] = 1
    expected_values, expected_counts = np.unique(arr, return_counts=True)

    values, counts = _unique_with_counts(arr, use_memmap=True)

    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)


def test_unique_with_counts_with_memmap_on_never_flattens_the_whole_array():
    arr = _watched(_random_volume(shape=(4, 5, 6), seed=2))

    values, counts = _unique_with_counts(arr, use_memmap=True)

    expected_values, expected_counts = np.unique(np.asarray(arr), return_counts=True)
    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)


def test_unique_with_counts_with_memmap_off_does_flatten_the_whole_array():
    """Confirms the watcher above is a real discriminator, not a no-op --
    the plain (unprotected) path really does flatten its input whole."""
    arr = _watched(_random_volume(shape=(4, 5, 6), seed=2))

    with pytest.raises(AssertionError):
        _unique_with_counts(arr, use_memmap=False)


@pytest.mark.parametrize(
    "raw",
    [
        np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.uint8),
        np.array([[[0, 255], [255, 0]], [[255, 255], [0, 0]]], dtype=np.uint8),
        np.array([[[1, 2], [2, 1]], [[2, 2], [1, 1]]], dtype=np.uint8),
        np.full((2, 2, 2), 7, dtype=np.uint8),
    ],
)
def test_to_binary_volume_for_skeletonization_gives_the_same_result_with_memmap_on(raw):
    memmap_result = _to_binary_volume_for_skeletonization(raw, use_memmap=True)
    eager_result = _to_binary_volume_for_skeletonization(raw, use_memmap=False)

    assert memmap_result.dtype == bool
    assert np.array_equal(memmap_result, eager_result)


def test_to_binary_volume_for_skeletonization_with_memmap_on_never_flattens_a_low_cardinality_mask():
    raw = _watched(
        np.array([[[0, 255], [255, 0]], [[255, 255], [0, 0]]], dtype=np.uint8)
    )

    result = _to_binary_volume_for_skeletonization(raw, use_memmap=True)

    assert np.array_equal(result, np.asarray(raw) == 255)


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


def test_fill_binary_holes_releases_the_inverted_memmap_even_if_the_inversion_loop_raises():
    """Regression for a leak found in code review: the per-slice loop that
    built the transient `inverted` buffer used to run *before* the
    try/finally that released it, so an exception raised while writing
    any slice leaked the backing file -- only a failure in the labelling
    call *after* the loop was ever protected. The loop now runs inside a
    ``with temporary_memmap_array(...)`` block that covers it directly, so
    this forces the failure inside the loop itself (not the labelling
    call) and checks nothing is left behind."""
    import glob
    import tempfile

    class PoisonedAtSliceTwo(np.ndarray):
        def __getitem__(self, key):
            if key == 2:
                raise RuntimeError("boom")
            return super().__getitem__(key)

    mask = np.zeros((5, 5, 5), dtype=bool).view(PoisonedAtSliceTwo)

    before = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))
    with pytest.raises(RuntimeError, match="boom"):
        fill_binary_holes(mask, use_memmap=True)
    after = set(glob.glob(str(tempfile.gettempdir()) + "/haemolynx_*.memmap"))

    assert after == before


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


def test_skeletonize_by_component_with_memmap_on_gives_each_component_a_memmap_mask(monkeypatch):
    """Regression: ``labeled[bbox] == component_id`` always allocated a
    fresh, plain-RAM array the size of the component's bbox, regardless of
    `use_memmap` -- for a single component filling (almost) the whole
    volume, exactly the case Tier 2 tiling exists for, that bbox *is* the
    whole volume, so every component's own mask must be a memmap too, not
    only the labelled array and the accumulated result."""
    import haemolynx.preprocessing.skeleton as skeleton_module

    mask = np.zeros((10, 30, 30), dtype=bool)
    mask[2:8, 5:15, 5:15] = True  # component A
    mask[2:8, 20:28, 5:15] = True  # component B, disconnected from A

    seen_types = []
    original = skeleton_module._skeletonize_component_into

    def spy(component_mask, out, **kwargs):
        seen_types.append(type(component_mask))
        return original(component_mask, out, **kwargs)

    monkeypatch.setattr(skeleton_module, "_skeletonize_component_into", spy)

    result = skeletonize_by_component(mask, use_memmap=True)

    assert len(seen_types) == 2, "expected one call per connected component"
    assert all(issubclass(t, np.memmap) for t in seen_types), seen_types
    release_memmap_array(result)


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


# --- skeletonize_by_component: tile_large_components (Tier 2) ----------------


def test_tiling_with_a_generous_halo_matches_the_monolithic_result():
    """A straight tube long enough in Z that a small tile_max_voxels forces
    several tiles; tile_halo_voxels generously larger than the tube's own
    radius should make tiling indistinguishable from the untiled result --
    and from tiling turned off entirely, proving the *toggle* is what
    changes behaviour, not incidental tile-size bookkeeping."""
    mask = np.zeros((60, 20, 20), dtype=bool)
    mask[:, 8:12, 8:12] = True
    monolithic = skeletonize(mask, method="lee")

    tiled = skeletonize_by_component(
        mask,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=20 * 20 * 10,  # 10 z-slices per tile -> 6 tiles
        tile_halo_voxels=15,
    )
    untiled = skeletonize_by_component(
        mask,
        use_memmap=True,
        tile_large_components=False,
        tile_max_voxels=20 * 20 * 10,
        tile_halo_voxels=15,
    )

    assert isinstance(tiled, np.memmap)
    assert np.array_equal(tiled, monolithic)
    assert np.array_equal(untiled, monolithic)
    release_memmap_array(tiled)
    release_memmap_array(untiled)


def test_tiling_toggle_off_ignores_tile_max_voxels_even_if_it_would_force_tiling():
    """tile_large_components defaults False -- a component larger than
    tile_max_voxels must still be skeletonized as one piece, exactly like
    before tiling existed, unless the toggle is explicitly on."""
    mask = np.zeros((60, 20, 20), dtype=bool)
    mask[:, 8:12, 8:12] = True

    result = skeletonize_by_component(
        mask, use_memmap=True, tile_max_voxels=1, tile_halo_voxels=0
    )

    assert np.array_equal(result, skeletonize(mask, method="lee"))
    release_memmap_array(result)


@pytest.mark.parametrize(
    "tile_max_voxels",
    [
        500,  # depth=20, 10x10 slices -> tile_depth=5, exact multiple (4 tiles)
        700,  # tile_depth=7, 20 has a remainder tile (2 full + 6 left over)
        50,  # smaller than one slice's own 100 voxels -> tile_depth forced to 1
        2000,  # >= mask.size -> fits in a single tile, no actual tiling
    ],
)
def test_tiling_boundary_math_matches_the_monolithic_result(tile_max_voxels):
    mask = np.zeros((20, 10, 10), dtype=bool)
    mask[:, 3:7, 3:7] = True

    result = skeletonize_by_component(
        mask,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=tile_max_voxels,
        tile_halo_voxels=8,
    )

    assert np.array_equal(result, skeletonize(mask, method="lee"))
    release_memmap_array(result)


def test_a_too_small_halo_gives_a_small_bounded_difference_not_an_unbounded_one():
    """Documents the actual tradeoff with concrete numbers instead of an
    unverified claim: a thick cylinder (needs more erosion context per
    slab than a thin tube) shows a real but small, bounded difference from
    the monolithic result at an inadequate halo, and none at a generous
    one -- confirmed empirically before writing this test that halo=10
    already fully closes the gap for this shape, so this pins both sides
    of that boundary rather than just asserting inequality."""
    zz, yy, xx = np.indices((60, 40, 40))
    mask = (yy - 20) ** 2 + (xx - 20) ** 2 <= 15**2
    monolithic = skeletonize(mask, method="lee")

    inadequate = skeletonize_by_component(
        mask,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=40 * 40 * 8,
        tile_halo_voxels=0,
    )
    diff_count = int(np.count_nonzero(inadequate != monolithic))
    assert 0 < diff_count < 200, (
        "expected a real but bounded difference at an inadequate halo, "
        f"got {diff_count} differing voxels"
    )
    release_memmap_array(inadequate)

    generous = skeletonize_by_component(
        mask,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=40 * 40 * 8,
        tile_halo_voxels=10,
    )
    assert np.array_equal(generous, monolithic)
    release_memmap_array(generous)


def test_tiling_honours_memmap_directory(tmp_path):
    mask = np.zeros((60, 20, 20), dtype=bool)
    mask[:, 8:12, 8:12] = True
    custom_dir = tmp_path / "custom_memmap_dir"
    assert not custom_dir.exists()

    result = skeletonize_by_component(
        mask,
        use_memmap=True,
        memmap_directory=custom_dir,
        tile_large_components=True,
        tile_max_voxels=20 * 20 * 10,
        tile_halo_voxels=15,
    )

    assert Path(result.filename).parent == custom_dir
    assert np.array_equal(result, skeletonize(mask, method="lee"))
    release_memmap_array(result)


# --- preprocessing.skeleton: connect_skeleton_components / -------------------
# --- _filter_components_by_total_fraction -------------------------------


def _two_disconnected_lines(shape=(5, 20, 20)):
    mask = np.zeros(shape, dtype=bool)
    mask[2, 2, 2:8] = True  # component A, 6 voxels
    mask[2, 2, 12:18] = True  # component B, 6 voxels, 4 voxels away from A
    return mask


def test_connect_skeleton_components_gives_the_same_result_with_memmap_on():
    skeleton = _two_disconnected_lines()

    memmap_result = connect_skeleton_components(
        skeleton, max_bridge_distance=10, use_memmap=True
    )
    eager_result = connect_skeleton_components(
        skeleton, max_bridge_distance=10, use_memmap=False
    )

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    # The gap must actually have been bridged, or this test would pass
    # vacuously (both sides just returning the untouched input).
    assert memmap_result.sum() > skeleton.sum()
    release_memmap_array(memmap_result)


def test_connect_skeleton_components_with_memmap_on_reaches_no_bridge_early_return():
    """n_components <= 1 (or, as here, a single line with nothing to bridge
    to) returns the original array from inside the labelled-components
    block -- must not raise on the way out even though that block's own
    memmap gets released first."""
    skeleton = np.zeros((5, 20, 20), dtype=bool)
    skeleton[2, 2, 2:8] = True

    result = connect_skeleton_components(skeleton, max_bridge_distance=10, use_memmap=True)

    assert result is skeleton


def test_connect_skeleton_components_with_memmap_on_re_skeletonizes_through_skeletonize_by_component(
    monkeypatch,
):
    """Regression: the re-skeletonize step after a successful bridge called
    the plain, untiled skeletonize_volume -- same bug as
    preprocess_skeleton_for_graph's own re-skeletonize step, in a second
    place that reaches it independently (connect_skeleton_components is
    also called directly by preprocess_skeleton_for_graph, not only
    through that other step)."""
    import haemolynx.preprocessing.skeleton as skeleton_module

    skeleton = _two_disconnected_lines()
    calls = []
    original = skeleton_module.skeletonize_by_component

    def spy(mask, **kwargs):
        calls.append(kwargs)
        return original(mask, **kwargs)

    monkeypatch.setattr(skeleton_module, "skeletonize_by_component", spy)

    connect_skeleton_components(
        skeleton,
        max_bridge_distance=10,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=321,
        tile_halo_voxels=7,
    )

    assert len(calls) == 1
    assert calls[0]["use_memmap"] is True
    assert calls[0]["tile_large_components"] is True
    assert calls[0]["tile_max_voxels"] == 321
    assert calls[0]["tile_halo_voxels"] == 7


def test_filter_components_by_total_fraction_gives_the_same_result_with_memmap_on():
    skeleton = np.zeros((5, 20, 20), dtype=bool)
    skeleton[2, 2, 2:14] = True  # 12 voxels, kept
    skeleton[2, 10, 10] = True  # 1 voxel, dropped at fraction=0.5

    memmap_result = _filter_components_by_total_fraction(
        skeleton, min_component_fraction=0.5, use_memmap=True
    )
    eager_result = _filter_components_by_total_fraction(
        skeleton, min_component_fraction=0.5, use_memmap=False
    )

    assert isinstance(memmap_result, np.memmap)
    assert not isinstance(eager_result, np.memmap)
    assert np.array_equal(memmap_result, eager_result)
    assert memmap_result.sum() == 12
    release_memmap_array(memmap_result)


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


def test_preprocess_skeleton_for_graph_re_skeletonizes_through_skeletonize_by_component(
    monkeypatch,
):
    """Regression: the re-skeletonize step after bundle collapse called the
    plain, untiled ``skeletonize_volume`` unconditionally -- a second,
    full-volume Lee-thinning call with none of the first call's
    use_memmap/tiling protection. It must route through
    ``skeletonize_by_component`` instead, with the tiling settings this
    function was given forwarded to it."""
    import haemolynx.preprocessing.skeleton as skeleton_module

    skeleton = np.zeros((5, 15, 15), dtype=bool)
    skeleton[2, 2, :] = True

    # skeletonize_voxel_bundles_into_paths (the step just before the one
    # under test) legitimately calls skeletonize_volume itself -- only the
    # re-skeletonize step after it is the regression under test, so this
    # spies on skeletonize_by_component rather than forbidding
    # skeletonize_volume outright.
    calls = []
    original = skeleton_module.skeletonize_by_component

    def spy(mask, **kwargs):
        calls.append(kwargs)
        return original(mask, **kwargs)

    monkeypatch.setattr(skeleton_module, "skeletonize_by_component", spy)

    result = preprocess_skeleton_for_graph(
        skeleton,
        max_bridge_distance=0,
        use_memmap=True,
        tile_large_components=True,
        tile_max_voxels=123,
        tile_halo_voxels=4,
    )

    assert len(calls) == 1
    assert calls[0]["use_memmap"] is True
    assert calls[0]["tile_large_components"] is True
    assert calls[0]["tile_max_voxels"] == 123
    assert calls[0]["tile_halo_voxels"] == 4
    assert np.array_equal(result, skeleton)
