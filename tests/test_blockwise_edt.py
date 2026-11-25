"""preprocessing/blockwise_edt.py: a block's own distance transform, with
enough context, answers as the whole volume's does.

Spacings here are binary fractions, so every distance is computed exactly and
two equidistant features give the same value bit for bit; with others two
such features can differ by a rounding step (the caveat in
pointwise_distance), which the thick-vessel tests cover on real fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from haemolynx.preprocessing import blockwise_edt

SPACINGS = [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5), (1.5, 0.75, 0.5)]


def _random_volume(seed, shape=(18, 23, 21), fraction=None):
    rng = np.random.default_rng(seed)
    return rng.random(shape) > (rng.uniform(0.5, 0.97) if fraction is None else fraction)


def _box(volume):
    return lambda slices: volume[slices]


def _balls(shape, spacing, centres_radii):
    grids = np.indices(shape).astype(float)
    out = np.zeros(shape, dtype=bool)
    for centre, radius in centres_radii:
        d2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, centre, spacing))
        out |= d2 <= radius**2
    return out


@pytest.mark.parametrize("spacing", SPACINGS + [(1.3, 0.913, 0.6)])
@pytest.mark.parametrize("seed", range(3))
def test_distances_at_reads_scipys_values_from_the_feature_transform(spacing, seed):
    volume = _random_volume(seed)
    coords = np.argwhere(np.ones(volume.shape, dtype=bool))

    read = blockwise_edt.distances_at(blockwise_edt.feature_transform(volume, spacing), coords, spacing)

    assert np.array_equal(read, distance_transform_edt(volume, sampling=spacing).ravel())


def test_displacement_distances_is_the_same_arithmetic():
    spacing = (1.3, 0.913, 0.6)
    displacement = np.array([[3, -4, 0], [0, 0, 5], [1, 2, 2]])
    expected = np.sqrt(((displacement * np.array(spacing)) ** 2).sum(axis=1))
    assert np.allclose(blockwise_edt.displacement_distances(displacement, spacing), expected)
    volume = np.ones((9, 11, 13), dtype=bool)
    volume[4, 5, 6] = False
    whole = distance_transform_edt(volume, sampling=spacing)
    for d in displacement:
        assert blockwise_edt.displacement_distances(d[None], spacing)[0] == whole[tuple(np.array([4, 5, 6]) + d)]


@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("reach", [0.0, 1.0, 2.5, 6.0])
@pytest.mark.parametrize("seed", range(3))
def test_near_distances_compare_with_the_reach_as_the_whole_transform_does(spacing, reach, seed):
    volume = _random_volume(seed)
    wanted = _random_volume(seed + 100, fraction=0.4)
    whole = distance_transform_edt(volume, sampling=spacing)

    seen = np.zeros(volume.shape, dtype=bool)
    for core, where, distance in blockwise_edt.near_distances_by_block(
        _box(volume), _box(wanted), volume.shape, sampling=spacing, reach=reach, block_voxels=900
    ):
        truth = whole[core][where]
        near = truth <= reach
        assert np.array_equal(distance <= reach, near)
        assert np.array_equal(distance[near], truth[near])
        assert np.all(distance[~near] > reach)
        assert not seen[core].any(), "blocks overlap"
        seen[core] = where
    assert np.array_equal(seen, wanted), "every wanted voxel once"


def test_near_distances_skip_the_transform_where_no_feature_is_in_reach(monkeypatch):
    volume = np.ones((30, 30, 30), dtype=bool)
    volume[0, 0, 0] = False
    calls = []
    real = blockwise_edt.feature_transform

    def counting(block, sampling):
        calls.append(block.shape)
        return real(block, sampling)

    monkeypatch.setattr(blockwise_edt, "feature_transform", counting)
    blocks = list(
        blockwise_edt.near_distances_by_block(
            _box(volume), _box(volume), volume.shape, sampling=(1.0, 1.0, 1.0), reach=2.0, block_voxels=1000
        )
    )
    far = [distance for core, _where, distance in blocks if all(s.start >= 10 for s in core)]
    assert far and all(np.all(np.isinf(d)) for d in far)
    assert len(calls) < len(blocks)


@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("block_voxels", [300, 5_000, 200_000, 10**9])
def test_distances_at_points_are_the_domains_transform_even_far_inside_a_wide_vessel(spacing, block_voxels):
    """Balls wider than the starting reach (16 voxels), cut by the domain's
    faces, so the halo has to double and the faces stop it -- and, on the
    smaller budgets, outgrows them, leaving the deepest points to the
    surface."""
    shape = (40, 70, 70)
    volume = _balls(
        shape, spacing, [((20, 35, 35), 22.0), ((5, 5, 60), 9.0), ((38, 66, 2), 6.0)]
    )
    domain = (slice(2, 40), slice(1, 69), slice(0, 70))
    points = np.argwhere(volume[domain]) + np.array([s.start for s in domain])
    rng = np.random.default_rng(0)
    points = points[rng.choice(len(points), size=min(len(points), 3_000), replace=False)]

    read = blockwise_edt.distances_at_points(
        _box(volume), domain, points, sampling=spacing, block_voxels=block_voxels
    )

    whole = distance_transform_edt(volume[domain], sampling=spacing)
    local = points - np.array([s.start for s in domain])
    assert np.array_equal(read, whole[tuple(local.T)])


def test_points_deeper_than_the_budget_reaches_are_measured_to_the_surface(monkeypatch):
    """Doubling until one block held a point's nearest feature transformed
    a block per point once cores shrank to a voxel; past the reach the budget
    holds, the surface's features stand in, found once per domain."""
    spacing = (2.0, 0.5, 0.5)
    shape = (30, 90, 90)
    volume = _balls(shape, spacing, [((15, 45, 45), 21.0)])
    domain = tuple(slice(0, s) for s in shape)
    points = np.argwhere(volume)
    calls = []
    real = blockwise_edt._surface_features

    def counting(*args):
        calls.append(args[1])
        return real(*args)

    monkeypatch.setattr(blockwise_edt, "_surface_features", counting)
    cache = {}
    whole = distance_transform_edt(volume, sampling=spacing)
    for _ in range(2):
        read = blockwise_edt.distances_at_points(
            _box(volume), domain, points, sampling=spacing, block_voxels=20_000, surface_cache=cache
        )
        assert np.array_equal(read, whole[tuple(points.T)])
    assert calls == [domain], "the surface is found once, for the domain"


def test_distances_at_points_on_a_domain_with_no_background_are_scipys_too():
    volume = np.ones((12, 13, 14), dtype=bool)
    domain = tuple(slice(0, s) for s in volume.shape)
    points = np.argwhere(volume)
    for block_voxels in (200, 10**6):
        read = blockwise_edt.distances_at_points(
            _box(volume), domain, points, sampling=(1.0, 1.0, 1.0), block_voxels=block_voxels
        )
        assert np.array_equal(read, distance_transform_edt(volume).ravel())


def test_distances_at_points_with_no_points_is_empty():
    volume = np.ones((5, 5, 5), dtype=bool)
    read = blockwise_edt.distances_at_points(
        _box(volume), tuple(slice(0, 5) for _ in range(3)), np.empty((0, 3), int), sampling=(1, 1, 1)
    )
    assert read.shape == (0,)


def test_the_parallel_plan_counts_this_process_s_own_volumes_as_free(monkeypatch):
    """The plan hands this process's disk-backed volumes' pages back before
    it reads the free memory: on Windows a written memmap's pages are not
    counted free until then, and the plan took them for the machine's."""
    import os

    import haemolynx.haemodynamics.resistance as resistance
    from haemolynx.preprocessing import memmap_support

    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    handed_back = []
    monkeypatch.setattr(memmap_support, "release_resident_pages", lambda *a: handed_back.append(a))
    monkeypatch.setattr(
        resistance, "available_memory_bytes", lambda: (8 << 30) if handed_back else (300 << 20)
    )

    block, workers = blockwise_edt.parallel_plan((200, 1200, 1250), 32_000_000)

    assert handed_back and workers > 1, (block, workers)


def test_the_parallel_plan_follows_cores_free_memory_and_the_volume(monkeypatch):
    """Workers: one per core but one, as many as half the free memory holds
    at the smallest block, and no more than there are blocks; blocks as large
    as that share allows. One process for a small volume, or when the free
    memory is unknown or too little."""
    import os

    import haemolynx.haemodynamics.resistance as resistance

    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    free = {"bytes": 8 << 30}
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: free["bytes"])
    plan = blockwise_edt.parallel_plan

    block, workers = plan((200, 1200, 1250), 32_000_000)
    budget = blockwise_edt.PARALLEL_MEMORY_FRACTION * free["bytes"]
    assert 1 < workers <= 19
    assert blockwise_edt.MIN_PARALLEL_BLOCK_VOXELS <= block <= 32_000_000
    assert workers * (block * blockwise_edt.BLOCK_BYTES_PER_VOXEL + blockwise_edt.WORKER_BASE_BYTES) <= budget

    free["bytes"] = 64 << 30
    assert plan((200, 1200, 1250), 32_000_000)[1] == 19, "cores bound it"

    assert plan((40, 150, 150), 32_000_000) == (32_000_000, 1), "a small volume stays here"
    free["bytes"] = 300 << 20
    assert plan((200, 1200, 1250), 32_000_000) == (32_000_000, 1), "too little memory"
    free["bytes"] = None
    assert plan((200, 1200, 1250), 32_000_000) == (32_000_000, 1), "memory unknown"

    free["bytes"] = 8 << 30
    monkeypatch.setattr(blockwise_edt, "PARALLEL_MIN_VOXELS", 0)
    block, workers = plan((10, 40, 60), 6_000)
    assert block == 6_000 and workers == 4, "no more workers than blocks"
