"""Speed-ups to skeletonisation and graph building that must not change a result.

Each test pins a faster computation against the one it replaced, exactly:
bundle refinement's per-hub work in a box instead of the whole volume, the
paired consistency diagnostics sharing one distance transform, the loop
router's repulsion blur on the edge's own neighbourhood, and routing handed
to worker processes.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    generate_binary_structure,
    maximum_filter,
    uniform_filter,
)

import haemolynx.graph.reconnect as reconnect_mod
from haemolynx.graph.diagnostics import (
    diagnose_graph_against_mask,
    diagnose_graph_mask_consistency,
    diagnose_vessels_missing_from_graph,
)
from haemolynx.preprocessing.skeleton import (
    _draw_hub_links,
    _select_hub_centres,
    skeletonize_volume,
    skeletonize_voxel_bundles_into_paths,
)
from haemolynx.preprocessing.skeleton_consistency import (
    diagnose_skeleton_against_mask,
    diagnose_skeleton_mask_consistency,
    diagnose_vessels_missing_from_skeleton,
)


def _vessels(seed, shape=(14, 40, 40)):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    for _ in range(5):
        cy, cx, r = rng.uniform(3, shape[1] - 3), rng.uniform(3, shape[2] - 3), rng.uniform(1.5, 4)
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask |= (zz - rng.uniform(2, shape[0] - 2)) ** 2 + (yy - cy) ** 2 <= r * r
    mask[rng.random(shape) > 0.995] = True  # specks: small "vessels"
    return mask


# --- bundle refinement -------------------------------------------------------


def _bundles_whole_volume(binary_mask, scan_size=9, density_fraction=0.35,
                          max_connections_per_hub=8, hub_min_spacing=None):
    """The per-hub loop as it was: a volume-sized array per hub."""
    mask = binary_mask.astype(bool)
    scan = (max(3, int(scan_size)),) * mask.ndim
    if hub_min_spacing is None:
        hub_min_spacing = max(1, int(min(scan) / 2))
    base_skeleton = skeletonize_volume(mask)
    density = uniform_filter(mask.astype(np.float32), size=scan, mode="constant")
    dense_volume = density >= density_fraction
    if not dense_volume.any():
        return base_skeleton.astype(bool)
    peak_map = dense_volume & (density == maximum_filter(density, size=scan, mode="nearest"))
    peak_coords = np.argwhere(peak_map)
    selected_hubs = _select_hub_centres(peak_coords, density[tuple(peak_coords.T)], hub_min_spacing)
    result = base_skeleton.astype(bool).copy()
    shape = np.array(mask.shape)
    half_window = np.array(scan) // 2
    structure = generate_binary_structure(mask.ndim, 1)
    for hub in selected_hubs:
        lo = np.maximum(hub - half_window, 0)
        hi = np.minimum(hub + half_window + 1, shape)
        slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(mask.ndim))
        local_dense = np.zeros_like(mask, dtype=bool)
        local_dense[slices] = dense_volume[slices]
        if not local_dense.any():
            continue
        local_mask_coords = np.argwhere(mask[slices])
        if local_mask_coords.size == 0:
            continue
        nearest = int(np.argmin(np.sum((local_mask_coords - (hub - lo).astype(float)) ** 2, axis=1)))
        center = (local_mask_coords[nearest] + lo).astype(int)
        shell = binary_dilation(local_dense, structure=structure) & ~local_dense
        boundary_points = np.argwhere(result & shell)
        result[local_dense] = False
        result[tuple(center.tolist())] = True
        if boundary_points.size:
            _draw_hub_links(result, center, boundary_points, max_connections_per_hub)
    return skeletonize_volume(result.astype(bool)).astype(bool)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("scan_size", [5, 9])
def test_bundle_refinement_in_hub_boxes_is_the_whole_volume_result(seed, scan_size):
    mask = _vessels(seed)

    expected = _bundles_whole_volume(mask, scan_size=scan_size)

    assert np.array_equal(skeletonize_voxel_bundles_into_paths(mask, scan_size=scan_size), expected)


# --- consistency diagnostics -----------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("voxel_size_zyx", [(1.0, 1.0, 1.0), (1.6, 0.4, 0.5)])
def test_skeleton_diagnostics_together_are_the_two_reports(seed, voxel_size_zyx):
    mask = _vessels(seed)
    skeleton = skeletonize_volume(mask)
    skeleton[: mask.shape[0] // 2] = False  # leave some vessels unexplained

    together = diagnose_skeleton_against_mask(
        skeleton, mask, voxel_size_zyx=voxel_size_zyx, min_vessel_voxels=3
    )

    assert together == (
        diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=voxel_size_zyx),
        diagnose_vessels_missing_from_skeleton(
            skeleton, mask, voxel_size_zyx=voxel_size_zyx, min_vessel_voxels=3
        ),
    )


def _graph_of(skeleton, voxel_size_zyx):
    G = nx.MultiGraph()
    voxels = np.argwhere(skeleton)[:200] * np.asarray(voxel_size_zyx)
    for i in range(0, len(voxels) - 5, 5):
        G.add_edge(i, i + 5, voxels=voxels[i:i + 6].tolist())
    return G


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("voxel_size_zyx", [(1.0, 1.0, 1.0), (1.6, 0.4, 0.5)])
def test_graph_diagnostics_together_are_the_two_reports(seed, voxel_size_zyx):
    mask = _vessels(seed)
    G = _graph_of(skeletonize_volume(mask), voxel_size_zyx)

    together = diagnose_graph_against_mask(
        G, mask, voxel_size_zyx=voxel_size_zyx, min_vessel_voxels=3
    )

    assert together == (
        diagnose_graph_mask_consistency(G, mask, voxel_size_zyx=voxel_size_zyx),
        diagnose_vessels_missing_from_graph(
            G, mask, voxel_size_zyx=voxel_size_zyx, min_vessel_voxels=3
        ),
    )


def test_diagnostics_together_on_an_empty_mask():
    empty = np.zeros((4, 5, 6), dtype=bool)
    assert diagnose_skeleton_against_mask(empty, empty) == (
        diagnose_skeleton_mask_consistency(empty, empty),
        diagnose_vessels_missing_from_skeleton(empty, empty),
    )


# --- loop reconnection -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sigma", [0.7, 2.0, 3.3])
def test_repulsion_blur_on_the_edge_neighbourhood_is_the_window_blur(seed, sigma):
    """Including windows smaller than the blur's reach, and edges touching a
    window face, where the whole-window blur reflects."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        shape = tuple(int(v) for v in rng.integers(1, 30, 3))
        mask = np.zeros(shape)
        start = np.array([rng.integers(0, s) for s in shape])
        path = np.clip(start + np.cumsum(rng.integers(-1, 2, (15, 3)), axis=0), 0, np.array(shape) - 1)
        mask[tuple(path.T)] = 1.0

        assert np.array_equal(
            reconnect_mod._gaussian_of_sparse(mask, sigma), gaussian_filter(mask, sigma=sigma)
        )


def _rings(count=3, extent=60):
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
        (min(u, v), max(u, v), d["voxels"])
        for u, v, d in G.edges(data=True)
        if d.get("secondary")
    )


def test_routing_in_worker_processes_finds_the_same_paths():
    skeleton, G = _rings()

    in_thread = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, routing_processes=0
    )
    in_processes = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, routing_processes=2
    )

    assert _secondary_edges(in_thread), "the fixture must produce secondary edges"
    assert _secondary_edges(in_processes) == _secondary_edges(in_thread)


def test_a_routing_worker_that_dies_falls_back_to_routing_in_thread():
    router = reconnect_mod._Router(2)
    try:
        assert router._workers, "workers should start"
        for worker in router._workers:
            worker.kill()
            worker.wait()
        cost = 1 + np.random.default_rng(0).random((5, 6, 7))

        result = router(cost, (0, 0, 0), (4, 5, 6))

        assert router._broken
        assert result == reconnect_mod.route_through_array(
            cost, (0, 0, 0), (4, 5, 6), fully_connected=True
        )
    finally:
        router.close()


def test_a_routing_error_in_a_worker_is_raised_as_in_thread():
    router = reconnect_mod._Router(2)
    try:
        cost = np.ones((3, 3, 3))
        with pytest.raises(Exception) as in_worker:
            router(cost, (0, 0, 0), (9, 9, 9))  # end outside the array
        with pytest.raises(Exception) as in_thread:
            reconnect_mod.route_through_array(cost, (0, 0, 0), (9, 9, 9), fully_connected=True)
        assert type(in_worker.value) is type(in_thread.value)
        assert not router._broken
    finally:
        router.close()


def test_small_runs_route_in_thread(monkeypatch):
    """Workers cost a second or two to start: only a run with enough pairs gets them."""
    made = []
    real = reconnect_mod._Router

    class Recording(real):
        def __init__(self, processes):
            made.append(processes)
            super().__init__(processes)

    monkeypatch.setattr(reconnect_mod, "_Router", Recording)
    skeleton, G = _rings(count=1)
    reconnect_mod.reconnect_secondary_loop_edges(G.copy(), skeleton, debug=False)

    assert made == [0]


@pytest.mark.parametrize("cores", [1, 3, 12])
def test_reconnect_threads_follow_the_core_count(monkeypatch, cores):
    """One thread per core by default (never more than there are pairs), and
    one routing worker per thread once a run is big enough for workers."""
    threads, routers = [], []
    real_pool, real_router = reconnect_mod.ThreadPoolExecutor, reconnect_mod._Router

    class RecordingPool(real_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            threads.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    class RecordingRouter(real_router):
        def __init__(self, processes):
            routers.append(processes)
            super().__init__(0)  # no real workers in a unit test

    monkeypatch.setattr(reconnect_mod.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(reconnect_mod, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(reconnect_mod, "_Router", RecordingRouter)
    monkeypatch.setattr(reconnect_mod, "ROUTING_PROCESS_MIN_PAIRS", 2)
    skeleton, G = _rings(count=3)  # three candidate pairs

    reconnect_mod.reconnect_secondary_loop_edges(G.copy(), skeleton, debug=False)

    assert threads == [min(cores, 3)]
    assert routers == [min(cores, 3)]
