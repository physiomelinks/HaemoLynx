"""Regression tests for the low-RAM option audit.

Each fix either removes a whole-volume allocation or speeds a step up, and
none may change a result: every test below compares against the default path
(or scipy/numpy's own whole-volume call) exactly, on volumes cut into many
small blocks so the block edges are exercised.
"""
from __future__ import annotations

import functools
import threading
import time

import h5py
import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import haemolynx.graph.reconnect as reconnect_mod
import haemolynx.preprocessing.memmap_support as memmap_mod
import haemolynx.preprocessing.segmentation_cleanup as cleanup_mod
import haemolynx.preprocessing.skeleton as skeleton_mod
from haemolynx.haemodynamics.edt_diameter import _pointwise_radius_sampler
from haemolynx.io.axis_order import apply_axis_order
from haemolynx.io.load import _unique_with_counts, load_3d_h5_with_voxel_size
from haemolynx.preprocessing.memmap_support import bincount_by_slab, release_memmap_array
from haemolynx.preprocessing.segmentation_cleanup import (
    _component_descriptors,
    smooth_vessel_surfaces,
    split_narrow_neck_components,
)
from haemolynx.preprocessing.skeleton import (
    compute_skeleton_connectivity_stats,
    drop_small_components,
)
from haemolynx.preprocessing.thick_vessels import inscribed_radius_map


@pytest.fixture
def small_blocks(monkeypatch):
    """Slabs and blocks of a few hundred voxels, so a small test volume is
    cut into many of them."""
    monkeypatch.setattr(memmap_mod, "LOW_MEMORY_BLOCK_VOXELS", 400)
    monkeypatch.setattr(skeleton_mod, "LOW_MEMORY_BLOCK_VOXELS", 400)
    small = functools.partial(memmap_mod.iter_blocks, block_voxels=400)
    monkeypatch.setattr(memmap_mod, "iter_blocks", small)
    monkeypatch.setattr(
        cleanup_mod,
        "map_blockwise",
        functools.partial(memmap_mod.map_blockwise, block_voxels=400),
    )


def _tubes(shape=(16, 24, 24), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.indices(shape)
    for _ in range(4):
        cy, cx, r = rng.uniform(4, shape[1] - 4), rng.uniform(4, shape[2] - 4), rng.uniform(1.5, 3.5)
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    mask &= rng.random(shape) > 0.02
    return mask


# --- counting components without an int64 copy of the labels ----------------


def test_bincount_by_slab_is_bincount(small_blocks):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 50, size=(9, 13, 11)).astype(np.int32)
    where = rng.random(labels.shape) > 0.5

    assert np.array_equal(bincount_by_slab(labels), np.bincount(labels.ravel()))
    assert np.array_equal(
        bincount_by_slab(labels, where=where, minlength=60),
        np.bincount(labels[where], minlength=60),
    )


def test_component_counts_never_cast_the_whole_label_volume(small_blocks, monkeypatch):
    """np.bincount casts its whole input to intp: on an int32 label memmap
    that was an 8-byte-per-voxel copy of the volume."""
    mask = _tubes()
    sizes = []
    real = np.bincount

    def recording(values, *args, **kwargs):
        sizes.append(np.size(values))
        return real(values, *args, **kwargs)

    monkeypatch.setattr(np, "bincount", recording)
    low_ram = drop_small_components(mask, min_size=30, use_memmap=True)
    stats = compute_skeleton_connectivity_stats(mask, use_memmap=True)
    monkeypatch.setattr(np, "bincount", real)

    assert sizes and max(sizes) < mask.size
    assert np.array_equal(low_ram, drop_small_components(mask, min_size=30))
    assert stats == compute_skeleton_connectivity_stats(mask)
    release_memmap_array(low_ram)


# --- segmentation cleanup ----------------------------------------------------


def test_component_descriptors_do_not_keep_every_voxel():
    """Nothing read a component's full coordinate list after it was
    described, but every one was kept for the whole reconnect step."""
    from scipy.ndimage import distance_transform_edt, label

    mask = _tubes()
    labeled, count = label(mask, structure=np.ones((3, 3, 3)))
    descriptors = _component_descriptors(
        labeled=labeled, count=count, edt_inside=distance_transform_edt(mask)
    )

    assert descriptors
    assert all("coords" not in d for d in descriptors.values())


def test_low_ram_neck_split_never_transforms_the_whole_volume(small_blocks, monkeypatch):
    mask = np.zeros((12, 20, 44), dtype=bool)
    zz, yy, xx = np.indices(mask.shape)
    mask |= (zz - 6) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2 <= 36
    mask |= (zz - 6) ** 2 + (yy - 10) ** 2 + (xx - 33) ** 2 <= 36
    mask |= ((zz - 6) ** 2 + (yy - 10) ** 2 <= 2) & (xx > 10) & (xx < 33)

    plain, plain_stats = split_narrow_neck_components(mask, voxel_size_zyx=(1.0, 0.5, 0.5))
    shapes = []
    real = cleanup_mod.distance_transform_edt

    def recording(array, *args, **kwargs):
        shapes.append(np.shape(array))
        return real(array, *args, **kwargs)

    monkeypatch.setattr(cleanup_mod, "distance_transform_edt", recording)
    low_ram, low_ram_stats = split_narrow_neck_components(
        mask, voxel_size_zyx=(1.0, 0.5, 0.5), use_memmap=True
    )

    assert plain_stats["cuts_made"] >= 1, "the fixture must have a neck to cut"
    assert mask.shape not in shapes
    assert low_ram_stats == plain_stats
    assert np.array_equal(low_ram, plain)


@pytest.mark.parametrize("voxel_size_zyx", [(1.0, 1.0, 1.0), (1.3, 0.4, 0.55)])
def test_low_ram_gaussian_smoothing_is_exact_in_blocks(small_blocks, voxel_size_zyx):
    mask = _tubes(seed=3)

    plain = smooth_vessel_surfaces(mask, voxel_size_zyx=voxel_size_zyx, sigma_um=0.8)
    low_ram = smooth_vessel_surfaces(
        mask, voxel_size_zyx=voxel_size_zyx, sigma_um=0.8, use_memmap=True
    )

    assert isinstance(low_ram, np.memmap)
    assert np.array_equal(low_ram, plain)
    release_memmap_array(low_ram)


# --- loading -----------------------------------------------------------------


@pytest.mark.parametrize("axis_order", ["xyz", "yxz", "zxy"])
def test_blocked_axis_reorder_matches_the_transpose(small_blocks, axis_order):
    volume = np.arange(7 * 9 * 11, dtype=np.uint16).reshape(7, 9, 11)

    low_ram = apply_axis_order(volume, axis_order, use_memmap=True)

    assert np.array_equal(low_ram, apply_axis_order(volume, axis_order))
    release_memmap_array(low_ram)


@pytest.mark.parametrize("chunks", [None, (3, 5, 5), (4, 9, 11)])
def test_low_ram_h5_read_in_chunk_slabs_is_the_dataset(small_blocks, tmp_path, chunks):
    data = np.random.default_rng(0).integers(0, 255, size=(10, 9, 11), dtype=np.uint8)
    path = tmp_path / "volume.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, chunks=chunks)

    image, *_ = load_3d_h5_with_voxel_size(str(path), dataset_name="data", use_memmap=True)

    assert np.array_equal(image, data)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32])
def test_slab_value_counts_match_numpy_unique(small_blocks, dtype):
    rng = np.random.default_rng(2)
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    low, high = (info.min, info.max) if info else (-3, 3)
    volume = rng.integers(max(low, -300), min(high, 300), size=(6, 7, 8)).astype(dtype)

    values, counts = _unique_with_counts(volume, use_memmap=True)
    expected_values, expected_counts = np.unique(volume, return_counts=True)

    assert values.dtype == volume.dtype
    assert np.array_equal(values, expected_values)
    assert np.array_equal(counts, expected_counts)


# --- EDT cross-check sampling --------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_batched_radius_sampling_is_the_whole_map_interpolation(seed):
    """Including points past the volume's edge, where each clipped box keeps
    the edge handling a call of its own gave it."""
    rng = np.random.default_rng(seed)
    mask = rng.random((10, 13, 12)) > 0.45
    voxel_size_zyx = (0.7, 0.31, 0.45)
    shape = np.array(mask.shape)
    points = rng.uniform(-1.5, 0.5, size=(3000, 3)) + rng.uniform(0, 1, (3000, 3)) * (shape + 1)

    expected = map_coordinates(
        inscribed_radius_map(mask, voxel_size_zyx), points.T, order=1, mode="constant", cval=0.0
    )

    assert np.array_equal(_pointwise_radius_sampler(mask, voxel_size_zyx)(points), expected)


# --- loop reconnection ---------------------------------------------------------


def _rings(count=3, extent=60):
    """*count* copies of a ring with a tail each side, in separate z planes:
    one candidate pair each, so several windows can be routed at once."""
    skeleton = np.zeros((4 * count + 1, extent, extent), dtype=bool)
    G = nx.MultiGraph()
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ys = np.round(30 + 20 * np.sin(t)).astype(int)
    xs = np.round(30 + 20 * np.cos(t)).astype(int)
    for ring in range(count):
        z = 4 * ring + 2
        skeleton[z, ys, xs] = True
        skeleton[z, 30, 0:11] = True
        skeleton[z, 30, 50:60] = True
        top = list(dict.fromkeys((z, y, x) for y, x, a in zip(ys, xs, t) if a <= np.pi))
        base = 4 * ring
        for offset, position in enumerate([(z, 30, 0), (z, 30, 50), (z, 30, 10), (z, 30, 59)]):
            G.add_node(base + offset, pos=np.array(position, dtype=float))
        G.add_edge(base, base + 2, voxels=[[float(z), 30.0, float(x)] for x in range(0, 11)])
        G.add_edge(base + 1, base + 3, voxels=[[float(z), 30.0, float(x)] for x in range(50, 60)])
        G.add_edge(base + 1, base + 2, voxels=[[float(c) for c in v] for v in reversed(top)])
    return skeleton, G


def _secondary_edges(G):
    return sorted(
        (min(u, v), max(u, v), round(d["length"], 9))
        for u, v, d in G.edges(data=True)
        if d.get("secondary")
    )


def test_low_ram_reconnect_routes_windows_bigger_than_one_block(monkeypatch):
    """They used to be skipped, losing their loop edges; now their distance
    transform runs in blocks, with the same values."""
    skeleton, G = _rings(count=1)
    plain = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1
    )
    monkeypatch.setattr(reconnect_mod, "LOW_MEMORY_BLOCK_VOXELS", 2_000)
    monkeypatch.setattr(
        memmap_mod, "iter_blocks", functools.partial(memmap_mod.iter_blocks, block_voxels=2_000)
    )
    import haemolynx.preprocessing.pointwise_distance as pointwise_mod

    monkeypatch.setattr(
        pointwise_mod, "iter_blocks", functools.partial(memmap_mod.iter_blocks, block_voxels=2_000)
    )

    low_ram = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1, use_memmap=True
    )

    assert _secondary_edges(plain)
    assert _secondary_edges(low_ram) == _secondary_edges(plain)


def test_low_ram_reconnect_threads_share_one_window_budget(monkeypatch):
    """Four threads each holding a window at the old per-window limit could
    reach ~15 GB; under the low-RAM option they wait for room instead."""
    skeleton, G = _rings(count=3)
    active = [0]
    peak = [0]
    lock = threading.Lock()
    real = reconnect_mod.route_through_array

    def slow_route(*args, **kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(reconnect_mod, "route_through_array", slow_route)

    unbudgeted = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=3
    )
    assert peak[0] > 1, "the fixture must route windows in parallel"

    # Each padded window is clipped to this small volume: room for one, not two.
    peak[0] = 0
    budgeted = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=3, use_memmap=True,
        low_memory_window_voxels=skeleton.size,
    )

    assert peak[0] == 1
    assert _secondary_edges(budgeted) == _secondary_edges(unbudgeted)
