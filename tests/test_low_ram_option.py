"""The Low RAM option (``use_memmap_loading``) changes where volumes live, never
what a run produces.

Each test compares a low-RAM code path against the ordinary one on the same
input. Where the low-RAM path works one block at a time, the block budget is
shrunk so the volume really is split, since a single block would trivially
agree with the whole-volume call.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pytest
import tifffile
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt
from skan import csr
from skimage.morphology import skeletonize

import haemolynx.graph.build as build_mod
import haemolynx.graph.reconnect as reconnect_mod
import haemolynx.preprocessing.skeleton as skeleton_mod
from haemolynx.graph import build_graph_from_skeleton
from haemolynx.graph.build import skan_skeleton
from haemolynx.graph.smoothing import smooth_graph_centrelines
from haemolynx.io.axis_order import apply_axis_order
from haemolynx.io.load import _to_binary_volume_for_skeletonization
from haemolynx.pipeline import default_schema
from haemolynx.pipeline import stages
from haemolynx.preprocessing.memmap_support import (
    iter_blocks,
    map_blockwise,
    new_memmap_array,
    release_superseded,
)
from haemolynx.preprocessing.skeleton import (
    compute_skeleton_connectivity_stats,
    skeletonize_voxel_bundles_into_paths,
)

DATA = Path(__file__).parent / "data"


def _ball(radius: int) -> np.ndarray:
    grid = np.indices((2 * radius + 1,) * 3) - radius
    return (grid**2).sum(axis=0) <= radius**2


def _random_mask(shape=(20, 26, 30), fraction=0.08, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape) < fraction


def _random_skeleton(seed: int, shape=(30, 40, 35)) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    volume = gaussian_filter(rng.random(shape), 2) > 0.52
    return skeletonize(volume, method="lee").astype(bool)


# --- memmap_support.iter_blocks / map_blockwise ------------------------------


@pytest.mark.parametrize(
    "shape, halo, block_voxels",
    [
        ((17, 23, 31), 0, 1_000),
        ((17, 23, 31), 3, 2_000),
        ((40, 9, 12), 5, 3_000),
        ((8, 8, 8), 2, 10**9),
        ((5, 6), 1, 20),
    ],
)
def test_iter_blocks_cores_partition_the_shape(shape, halo, block_voxels):
    cover = np.zeros(shape, dtype=np.int32)
    blocks = list(iter_blocks(shape, halo=halo, block_voxels=block_voxels))

    for padded, inner, core in blocks:
        cover[core] += 1
        for p, i, c, size in zip(padded, inner, core, shape):
            assert p.start == max(0, c.start - halo)
            assert p.stop == min(size, c.stop + halo)
            assert (i.start + p.start, i.stop + p.start) == (c.start, c.stop)

    assert (cover == 1).all(), "every voxel belongs to exactly one core"
    if block_voxels < np.prod(shape):
        assert len(blocks) > 1


def test_iter_blocks_keeps_every_padded_block_within_budget():
    for padded, _inner, _core in iter_blocks((40, 50, 60), halo=4, block_voxels=5_000):
        assert np.prod([s.stop - s.start for s in padded]) <= 5_000


def test_map_blockwise_matches_a_whole_volume_dilation():
    mask = _random_mask(seed=1)
    ball = _ball(2)
    out = np.zeros_like(mask)

    map_blockwise(mask, lambda b: binary_dilation(b, structure=ball), out, halo=2, block_voxels=900)

    assert np.array_equal(out, binary_dilation(mask, structure=ball))


def test_map_blockwise_matches_a_whole_volume_closing_with_a_two_radius_halo():
    mask = _random_mask(seed=2, fraction=0.15)
    ball = _ball(2)
    out = np.zeros_like(mask)

    map_blockwise(mask, lambda b: binary_closing(b, structure=ball), out, halo=4, block_voxels=1_500)

    assert np.array_equal(out, binary_closing(mask, structure=ball))


def test_map_blockwise_closing_with_too_small_a_halo_does_differ():
    """Proves the test above discriminates: a closing reaches 2r, not r."""
    mask = _random_mask(seed=2, fraction=0.15)
    ball = _ball(2)
    out = np.zeros_like(mask)

    map_blockwise(mask, lambda b: binary_closing(b, structure=ball), out, halo=2, block_voxels=1_500)

    assert not np.array_equal(out, binary_closing(mask, structure=ball))


def test_map_blockwise_matches_a_whole_volume_distance_threshold():
    mask = _random_mask(seed=3, fraction=0.01)
    max_gap = 3

    def within_gap(block):
        if not block.any():
            return block.copy()
        inverted = ~block
        return block | ((distance_transform_edt(inverted) <= max_gap) & inverted)

    out = np.zeros_like(mask)
    map_blockwise(mask, within_gap, out, halo=max_gap, block_voxels=1_200)

    assert np.array_equal(out, within_gap(mask))


# --- memmap_support.release_superseded ---------------------------------------


def test_release_superseded_deletes_an_intermediate_memmap():
    previous = new_memmap_array((3, 3, 3), bool)
    path = Path(previous.filename)

    release_superseded(previous, np.zeros((3, 3, 3), bool), keep=np.zeros(1))

    assert not path.exists()


def test_release_superseded_keeps_a_step_that_returned_its_input():
    same = new_memmap_array((3, 3, 3), bool)
    try:
        release_superseded(same, same, keep=np.zeros(1))
        assert Path(same.filename).exists()
    finally:
        Path(same.filename).unlink()


def test_release_superseded_keeps_the_callers_own_input():
    callers = new_memmap_array((3, 3, 3), bool)
    try:
        release_superseded(callers, np.zeros((3, 3, 3), bool), keep=callers)
        assert Path(callers.filename).exists()
    finally:
        Path(callers.filename).unlink()


def test_release_superseded_is_a_no_op_on_plain_arrays():
    previous = np.ones((3, 3, 3), bool)

    release_superseded(previous, np.zeros((3, 3, 3), bool), keep=np.zeros(1))

    assert previous.all()


# --- io.load._to_binary_volume_for_skeletonization ---------------------------


def _labels(values, shape=(4, 5, 6), seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(values), size=shape, p=None)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(_labels(np.array([0, 1], np.uint8)), id="0/1"),
        pytest.param(_labels(np.array([0, 255], np.uint8)), id="0/255"),
        pytest.param(
            np.where(_random_mask((4, 5, 6), 0.3), 2, 1).astype(np.uint8), id="1/2"
        ),
        pytest.param(_labels(np.array([0, 3, 5, 7], np.uint16)), id="few labels"),
        pytest.param(
            np.random.default_rng(4).integers(0, 255, (4, 5, 6), dtype=np.uint8),
            id="grayscale",
        ),
        pytest.param(np.random.default_rng(5).random((4, 5, 6)), id="float 0..1"),
        pytest.param(
            np.random.default_rng(6).normal(size=(4, 5, 6)).astype(np.float32),
            id="other float",
        ),
        pytest.param(np.full((4, 5, 6), np.nan), id="no finite values"),
    ],
)
def test_binarisation_writes_the_same_mask_to_a_memmap(raw):
    plain = _to_binary_volume_for_skeletonization(raw, use_memmap=False)
    low_ram = _to_binary_volume_for_skeletonization(raw, use_memmap=True)

    assert isinstance(low_ram, np.memmap)
    assert low_ram.dtype == bool
    assert np.array_equal(low_ram, plain)


# --- io.axis_order.apply_axis_order ------------------------------------------


@pytest.mark.parametrize("axis_order", ["xyz", "yxz", "zxy", "xzy"])
def test_apply_axis_order_into_a_memmap_matches_the_contiguous_copy(axis_order):
    volume = np.random.default_rng(7).integers(0, 9, (4, 5, 6), dtype=np.uint8)

    plain = apply_axis_order(volume, axis_order)
    low_ram = apply_axis_order(volume, axis_order, use_memmap=True)

    assert isinstance(low_ram, np.memmap)
    assert plain.flags.c_contiguous
    assert np.array_equal(low_ram, plain)


# --- preprocessing.skeleton: bundle refinement, connectivity stats -----------


def _dense_hub_fixture(seed: int) -> np.ndarray:
    """Three crowded voxel bundles, each crossed by three straight vessels."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((40, 60, 60), dtype=bool)
    for z, y, x in [(12, 15, 15), (28, 40, 42), (20, 15, 45)]:
        mask[z - 4:z + 5, y - 4:y + 5, x - 4:x + 5] = rng.random((9, 9, 9)) < 0.6
        mask[z, y, :] = True
        mask[z, :, x] = True
        mask[:, y, x] = True
    return mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_low_ram_bundle_refinement_matches_the_default_on_real_hubs(seed, monkeypatch):
    mask = _dense_hub_fixture(seed)
    plain = skeletonize_voxel_bundles_into_paths(mask)
    # The fixture must actually exercise hub collapse, not just skeletonize.
    assert not np.array_equal(plain, skeletonize(mask, method="lee").astype(bool))

    small_blocks = functools.partial(iter_blocks, block_voxels=20_000)
    monkeypatch.setattr(skeleton_mod, "iter_blocks", small_blocks)
    assert len(list(small_blocks(mask.shape, halo=8))) > 1

    low_ram = skeletonize_voxel_bundles_into_paths(mask, use_memmap=True)

    assert np.array_equal(np.asarray(low_ram), plain)


def test_connectivity_stats_are_the_same_in_low_ram_mode():
    skeleton = _random_skeleton(8)

    plain = compute_skeleton_connectivity_stats(skeleton)
    low_ram = compute_skeleton_connectivity_stats(skeleton, use_memmap=True)

    assert low_ram == plain


# --- graph.build.skan_skeleton -----------------------------------------------


def _assert_same_paths(expected, actual):
    assert actual.n_paths == expected.n_paths
    for index in range(expected.n_paths):
        assert np.array_equal(actual.path_coordinates(index), expected.path_coordinates(index))


@pytest.mark.parametrize("seed", range(6))
def test_skan_skeleton_without_the_image_matches_skan_on_random_skeletons(seed):
    skeleton = _random_skeleton(seed)

    _assert_same_paths(csr.Skeleton(skeleton), skan_skeleton(skeleton, use_memmap=True))


def test_skan_skeleton_without_the_image_matches_skan_on_the_fixtures():
    with h5py.File(DATA / "bundled_vessels_8_to_2.h5") as handle:
        bundled = np.asarray(handle["data"]) > 0
    seven = skeletonize(tifffile.imread(DATA / "seven_vessel_noisy_3d.tif") > 0, method="lee")

    for skeleton in (bundled, seven.astype(bool), seven.astype(np.uint8)):
        _assert_same_paths(csr.Skeleton(skeleton), skan_skeleton(skeleton, use_memmap=True))


def test_skan_skeleton_batches_neighbour_lookups_without_changing_the_paths(monkeypatch):
    skeleton = _random_skeleton(11)
    monkeypatch.setattr(build_mod, "_ADJACENCY_BATCH_NODES", 7)

    _assert_same_paths(csr.Skeleton(skeleton), skan_skeleton(skeleton, use_memmap=True))


def test_skan_skeleton_reads_a_memmap_without_copying_or_padding_it(monkeypatch, tmp_path):
    skeleton = _random_skeleton(12)
    mapped = np.lib.format.open_memmap(
        tmp_path / "skeleton.npy", mode="w+", dtype=bool, shape=skeleton.shape
    )
    mapped[...] = skeleton
    expected = csr.Skeleton(skeleton)

    def no_pad(*args, **kwargs):
        raise AssertionError("the whole volume was padded")

    monkeypatch.setattr(np, "pad", no_pad)

    _assert_same_paths(expected, skan_skeleton(mapped, use_memmap=True))


def test_skan_skeleton_of_an_empty_volume_has_no_paths():
    assert skan_skeleton(np.zeros((5, 6, 7), bool), use_memmap=True).n_paths == 0


def test_skan_skeleton_off_is_skans_own_object():
    assert isinstance(skan_skeleton(_random_skeleton(0)), csr.Skeleton)


# --- graph.reconnect under use_memmap ----------------------------------------


def _ring_fixture(extent=60):
    """A ring with a tail each side: nodes 1 and 2 are degree 2, joined along
    the top half, so the bottom half is a secondary route between them.

    *extent* past 60 leaves empty space beyond the ring, so a routing window
    is smaller than the volume."""
    skeleton = np.zeros((9, extent, extent), dtype=bool)
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ys = np.round(30 + 20 * np.sin(t)).astype(int)
    xs = np.round(30 + 20 * np.cos(t)).astype(int)
    skeleton[4, ys, xs] = True
    skeleton[4, 30, 0:11] = True
    skeleton[4, 30, 50:60] = True
    top = list(dict.fromkeys((4, y, x) for y, x, a in zip(ys, xs, t) if a <= np.pi))

    G = nx.MultiGraph()
    for node, position in enumerate([(4, 30, 0), (4, 30, 50), (4, 30, 10), (4, 30, 59)]):
        G.add_node(node, pos=np.array(position, dtype=float))
    G.add_edge(0, 2, voxels=[[4.0, 30.0, float(x)] for x in range(0, 11)])
    G.add_edge(1, 3, voxels=[[4.0, 30.0, float(x)] for x in range(50, 60)])
    G.add_edge(1, 2, voxels=[[float(c) for c in v] for v in reversed(top)])
    return skeleton, G


def _secondary_edges(G):
    return sorted(
        (min(u, v), max(u, v), round(d["length"], 9))
        for u, v, d in G.edges(data=True)
        if d.get("secondary")
    )


def test_low_ram_reconnect_finds_the_same_secondary_edges():
    skeleton, G = _ring_fixture()

    plain = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1
    )
    low_ram = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1, use_memmap=True
    )

    assert _secondary_edges(plain), "the fixture must produce a secondary edge"
    assert _secondary_edges(low_ram) == _secondary_edges(plain)


def test_low_ram_reconnect_never_transforms_the_whole_volume(monkeypatch):
    """Even when the windows are big enough that the default switches to the
    whole-volume cost field."""
    skeleton, G = _ring_fixture(extent=200)
    shapes = []
    real = reconnect_mod.distance_transform_edt

    def recording(array, *args, **kwargs):
        shapes.append(array.shape)
        return real(array, *args, **kwargs)

    monkeypatch.setattr(reconnect_mod, "distance_transform_edt", recording)
    monkeypatch.setattr(reconnect_mod, "GLOBAL_FIELD_WINDOW_FRACTION", 0.0)

    reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1
    )
    assert skeleton.shape in shapes, "the default did not reach the fallback"

    shapes.clear()
    reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), skeleton, debug=False, max_workers=1, use_memmap=True
    )
    assert shapes and skeleton.shape not in shapes


def test_low_ram_reconnect_skips_and_reports_windows_over_the_bound(caplog):
    skeleton, G = _ring_fixture()

    with caplog.at_level(logging.WARNING, logger="haemolynx.graph.reconnect"):
        result = reconnect_mod.reconnect_secondary_loop_edges(
            G.copy(),
            skeleton,
            debug=False,
            max_workers=1,
            use_memmap=True,
            low_memory_window_voxels=100,
        )

    assert _secondary_edges(result) == []
    assert any("too large for the low-RAM option" in r.getMessage() for r in caplog.records)


def test_low_ram_reconnect_reads_a_bool_skeleton_in_place(monkeypatch):
    skeleton, G = _ring_fixture()

    def no_astype(self, *args, **kwargs):
        raise AssertionError("the skeleton was copied")

    class NoCopy(np.ndarray):
        astype = no_astype

    guarded = skeleton.view(NoCopy)
    result = reconnect_mod.reconnect_secondary_loop_edges(
        G.copy(), guarded, debug=False, max_workers=1, use_memmap=True
    )

    assert _secondary_edges(result)


# --- graph.assemble / graph.smoothing ----------------------------------------


def _graph_signature(G):
    return (
        sorted(tuple(np.round(p, 6)) for p in nx.get_node_attributes(G, "pos").values()),
        sorted(round(d["length"], 6) for *_, d in G.edges(data=True)),
    )


@pytest.mark.parametrize("seed", [0, 3])
def test_build_graph_from_skeleton_gives_the_same_graph_in_low_ram_mode(seed):
    skeleton = _random_skeleton(seed)

    plain = build_graph_from_skeleton(skeleton)
    low_ram = build_graph_from_skeleton(skeleton, use_memmap=True)

    assert plain.number_of_edges() > 0
    assert _graph_signature(low_ram) == _graph_signature(plain)


def test_build_graph_from_skeleton_routes_use_memmap_to_both_steps(monkeypatch):
    import haemolynx.graph.assemble as assemble_mod

    seen = {}
    real_skan, real_reconnect = assemble_mod.skan_skeleton, assemble_mod.reconnect_secondary_loop_edges

    def spy_skan(skeleton, *, use_memmap=False):
        seen["skan"] = use_memmap
        return real_skan(skeleton, use_memmap=use_memmap)

    def spy_reconnect(*args, use_memmap=False, **kwargs):
        seen["reconnect"] = use_memmap
        return real_reconnect(*args, use_memmap=use_memmap, **kwargs)

    monkeypatch.setattr(assemble_mod, "skan_skeleton", spy_skan)
    monkeypatch.setattr(assemble_mod, "reconnect_secondary_loop_edges", spy_reconnect)

    build_graph_from_skeleton(_random_skeleton(0), use_memmap=True)

    assert seen == {"skan": True, "reconnect": True}


def test_centreline_smoothing_is_the_same_for_bool_and_integer_skeletons():
    skeleton = _random_skeleton(5)
    G = build_graph_from_skeleton(skeleton)
    as_bool, as_int = G.copy(), G.copy()

    counts_bool = smooth_graph_centrelines(as_bool, skeleton)
    counts_int = smooth_graph_centrelines(as_int, skeleton.astype(np.uint8))

    assert counts_bool == counts_int
    assert counts_bool["smoothed"] + counts_bool["relaxed"] > 0
    assert _graph_signature(as_bool) == _graph_signature(as_int)


# --- pipeline.stages ----------------------------------------------------------

SCHEMA = default_schema()


def _stage_settings(tmp_path: Path, mask_path: Path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": mask_path,
            "vtk_output_prefix": tmp_path / "run" / "network",
            "plot_dir": tmp_path / "plots",
            "do_skeletonize": True,
            **overrides,
        }
    )
    return values


def _write_rod_mask(tmp_path: Path) -> Path:
    mask = np.zeros((12, 30, 30), dtype=np.uint8)
    mask[5:8, 14:17, 3:27] = 255
    mask[5:8, 3:27, 14:17] = 255
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, mask)
    return path


def test_skeletonise_in_low_ram_mode_logs_the_skipped_diagnostic(tmp_path, monkeypatch, caplog):
    def forbidden(*args, **kwargs):
        raise AssertionError("the whole-volume diagnostic ran in low-RAM mode")

    monkeypatch.setattr(stages.preprocessing, "diagnose_skeleton_mask_consistency", forbidden)
    settings = _stage_settings(tmp_path, _write_rod_mask(tmp_path), use_memmap_loading=True)

    with caplog.at_level(logging.INFO, logger=stages.logger.name):
        stages.skeletonise(settings, stages.segment(settings))

    expected = stages.LOW_RAM_SKIPPED_DIAGNOSTICS_MESSAGE % "skeleton/mask consistency"
    assert expected in [r.getMessage() for r in caplog.records]


def test_skeletonise_without_low_ram_still_runs_the_diagnostic(tmp_path, monkeypatch):
    calls = []
    real = stages.preprocessing.diagnose_skeleton_mask_consistency

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(stages.preprocessing, "diagnose_skeleton_mask_consistency", spy)
    settings = _stage_settings(tmp_path, _write_rod_mask(tmp_path), use_memmap_loading=False)

    stages.skeletonise(settings, stages.segment(settings))

    assert calls == [1]


def test_resuming_from_a_saved_skeleton_memory_maps_it(tmp_path):
    mask_path = _write_rod_mask(tmp_path)
    first = _stage_settings(tmp_path, mask_path, use_memmap_loading=True)
    produced = stages.skeletonise(first, stages.segment(first))

    resumed_settings = _stage_settings(
        tmp_path, mask_path, use_memmap_loading=True, do_skeletonize=False
    )
    resumed = stages.skeletonise(resumed_settings, stages.segment(resumed_settings))

    assert isinstance(resumed.skeleton, np.memmap)
    assert np.array_equal(resumed.skeleton, np.asarray(produced.skeleton))


def test_build_network_passes_the_low_ram_setting_to_graph_building(tmp_path, monkeypatch):
    seen = []
    real = stages.graph.build_graph_from_skeleton

    def spy(*args, **kwargs):
        seen.append(kwargs.get("use_memmap"))
        return real(*args, **kwargs)

    monkeypatch.setattr(stages.graph, "build_graph_from_skeleton", spy)
    settings = _stage_settings(
        tmp_path, _write_rod_mask(tmp_path), use_memmap_loading=True, visualize_results=False
    )
    volume = stages.skeletonise(settings, stages.segment(settings))

    stages.build_network(settings, volume, SCHEMA)

    assert seen == [True]
