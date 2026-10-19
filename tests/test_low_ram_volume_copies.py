"""The low-RAM option (``use_memmap_loading``) without whole-volume copies a
5.5 G-voxel stack cannot hold: each step gives the in-RAM answer bit for
bit, and where it is practical its traced peak no longer grows with the
volume.

The thick region's centreline tree handed on rather than drawn again, the
gated skeleton on disk, the lumen radius from tiles of the surface; the
other items of the audit have their tests beside their modules' own.
"""
from __future__ import annotations

import tracemalloc

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import thick_vessel_junctions as tvj
from haemolynx.preprocessing import memmap_support
from haemolynx.preprocessing.pointwise_distance import FeatureDistance
from haemolynx.preprocessing.thick_vessels import (
    edt_ridge_voxels,
    skeletonize_edt_ridge,
    skeletonize_thickness_gated,
)


def _capsule_into(mask, spacing, a_um, b_um, radius_um):
    spacing = np.asarray(spacing, float)
    a, b = np.asarray(a_um, float), np.asarray(b_um, float)
    lo = np.maximum(np.floor((np.minimum(a, b) - radius_um) / spacing).astype(int) - 1, 0)
    hi = np.minimum(np.ceil((np.maximum(a, b) + radius_um) / spacing).astype(int) + 2, mask.shape)
    if np.any(hi <= lo):
        return
    grids = np.meshgrid(*[np.arange(l, h) * s for l, h, s in zip(lo, hi, spacing)], indexing="ij")
    p = np.stack(grids, axis=-1)
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab or 1.0), 0.0, 1.0)
    d = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
    mask[tuple(slice(l, h) for l, h in zip(lo, hi))] |= d <= radius_um


SPACING = (1.5, 0.6, 0.6)


def _network(seed=1, shape=(30, 90, 90)):
    """A fat trunk with a fat branch, thin capillaries fused onto both, and a
    loose capillary."""
    rng = np.random.default_rng(seed)
    spacing = np.asarray(SPACING)
    extent = np.asarray(shape) * spacing
    mask = np.zeros(shape, dtype=bool)
    a = np.array([-5.0, 0.4 * extent[1], 0.5 * extent[2]])
    b = np.array([extent[0] + 5.0, 0.6 * extent[1], 0.45 * extent[2]])
    _capsule_into(mask, spacing, a, b, 7.5)
    mid = a + 0.5 * (b - a)
    _capsule_into(mask, spacing, mid, mid + np.array([0.0, 20.0, 18.0]), 6.0)
    for _ in range(7):
        on = a + rng.uniform(0.15, 0.85) * (b - a)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        _capsule_into(mask, spacing, on, on + direction * rng.uniform(15.0, 40.0), rng.uniform(1.2, 2.2))
    _capsule_into(mask, spacing, (5.0, 5.0, 5.0), (25.0, 10.0, 20.0), 1.5)
    return mask


# --- the thick region's centreline tree, handed on -----------------------------


@pytest.mark.parametrize("use_memmap", [False, True])
def test_the_trees_voxels_are_argwhere_of_the_tree_drawn(use_memmap, monkeypatch):
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 30_000)
    trunk = np.zeros((26, 80, 90), dtype=bool)
    _capsule_into(trunk, SPACING, (-4.0, 30.0, 10.0), (60.0, 18.0, 45.0), 9.0)
    _capsule_into(trunk, SPACING, (20.0, 20.0, 30.0), (10.0, 45.0, 52.0), 6.0)
    trunk[:, :, 3] = False  # two components

    voxels = edt_ridge_voxels(trunk, SPACING, use_memmap=use_memmap)
    expected = np.argwhere(skeletonize_edt_ridge(trunk, SPACING, use_memmap=use_memmap))

    assert len(expected) and voxels.dtype == expected.dtype
    assert np.array_equal(voxels, expected)
    assert np.array_equal(tvj.thick_vessel_centreline_voxels(trunk, SPACING, use_memmap=use_memmap), expected)
    assert edt_ridge_voxels(np.zeros((4, 5, 6), bool), SPACING).shape == (0, 3)


@pytest.mark.parametrize("use_memmap", [False, True])
def test_the_gated_skeletonisation_hands_on_the_tree_it_drew(tmp_path, monkeypatch, use_memmap):
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 30_000)
    mask = _network()
    drawn: list = []

    skeleton, thick = skeletonize_thickness_gated(
        mask, voxel_size_zyx=SPACING, return_thick_mask=True, use_memmap=use_memmap,
        memmap_directory=tmp_path, centreline_voxels=drawn,
    )

    assert thick is not None and thick.any()
    assert len(drawn) == 1
    assert np.array_equal(drawn[0], np.argwhere(tvj.thick_vessel_centreline(thick, SPACING)))
    assert np.asarray(skeleton)[tuple(drawn[0].T)].all(), "the tree is in the skeleton"


def test_the_low_ram_gated_skeleton_is_on_disk_and_the_in_ram_one(tmp_path, monkeypatch):
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 30_000)
    mask = _network()
    plain, plain_thick = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, return_thick_mask=True)
    low, low_thick = skeletonize_thickness_gated(
        mask, voxel_size_zyx=SPACING, return_thick_mask=True, use_memmap=True, memmap_directory=tmp_path,
    )

    assert isinstance(low, np.memmap) and low.dtype == bool
    assert np.array_equal(low, plain) and np.array_equal(low_thick, plain_thick)
    # No thick region, and none asked for: Lee of the whole mask, on disk too.
    for kwargs in (dict(min_radius_um=0.0), dict(min_radius_um=50.0)):
        plain_lee, none = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, return_thick_mask=True, **kwargs)
        low_lee, low_none = skeletonize_thickness_gated(
            mask, voxel_size_zyx=SPACING, return_thick_mask=True, use_memmap=True,
            memmap_directory=tmp_path, **kwargs,
        )
        assert none is None and low_none is None
        assert isinstance(low_lee, np.memmap)
        assert np.array_equal(low_lee, plain_lee)


def test_a_region_found_before_to_be_empty_is_no_volume(tmp_path, monkeypatch):
    """A cached "no region" was a volume of zeros; a broadcast False reads the same."""
    from haemolynx.preprocessing import thick_vessels

    mask = _network()
    plain = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, thick_region=np.zeros(mask.shape, bool))
    empty = np.broadcast_to(np.False_, mask.shape)
    assert not thick_vessels._any(empty) and thick_vessels._any(np.broadcast_to(np.True_, (3, 4)))
    assert not thick_vessels._any(np.zeros((3, 4), bool)) and thick_vessels._any(np.eye(3, dtype=bool))
    broadcast = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, thick_region=empty)
    assert np.array_equal(broadcast, plain)


def _graph_through(mask):
    """A graph with a vessel running into the trunk and one along it."""
    G = nx.MultiGraph()
    spacing = np.asarray(SPACING)
    z = np.linspace(2.0, 40.0, 60)
    trunk = np.column_stack([z, np.full_like(z, 0.5 * 90 * 0.6), np.full_like(z, 0.47 * 90 * 0.6)])
    G.add_node(0, pos=trunk[0])
    G.add_node(1, pos=trunk[-1])
    G.add_edge(0, 1, voxels=trunk.tolist(), length=float(np.linalg.norm(np.diff(trunk, axis=0), axis=1).sum()))
    side = np.linspace(trunk[30], trunk[30] + np.array([0.0, 22.0, 0.0]), 30)
    G.add_node(2, pos=side[-1])
    G.add_edge(1, 2, voxels=np.vstack([trunk[30:][::-1][:1], side]).tolist(), length=22.0)
    return G


def test_the_split_reads_the_trees_voxels_as_it_read_the_tree():
    mask = _network()
    _skeleton, thick = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, return_thick_mask=True)
    G = _graph_through(mask)
    volume_tree = tvj.thick_vessel_centreline(thick, SPACING)
    voxels_tree = tvj.thick_vessel_centreline_voxels(thick, SPACING)

    by_volume = tvj.insert_thick_vessel_junction_nodes(G, thick, voxel_size_zyx=SPACING, segmentation=mask, centreline=volume_tree)
    by_voxels = tvj.insert_thick_vessel_junction_nodes(G, thick, voxel_size_zyx=SPACING, segmentation=mask, centreline=voxels_tree)
    found = tvj.insert_thick_vessel_junction_nodes(G, thick, voxel_size_zyx=SPACING, segmentation=mask)

    assert nx.utils.graphs_equal(by_volume, by_voxels) and nx.utils.graphs_equal(by_volume, found)
    one = tvj.ThickCentreline.from_region(thick, SPACING, segmentation=mask, centreline=volume_tree)
    two = tvj.ThickCentreline.from_region(thick, SPACING, segmentation=mask, centreline=voxels_tree)
    assert np.array_equal(one._points, two._points) and np.array_equal(one.radius_um, two.radius_um)


def test_the_low_ram_tree_voxels_hold_no_volume(monkeypatch):
    """Drawn as voxels, the tree needs no volume of zeros the region's
    shape: four times the stack, the same trunk, the same peak (a slab's
    labels), where the volume's grows with it."""
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 200_000)

    def peaks(depth):
        region = np.zeros((depth, 300, 300), dtype=bool)  # 3.6 M voxels a 40 planes
        _capsule_into(region, (1.0, 1.0, 1.0), (-2.0, 150.0, 20.0), (42.0, 150.0, 60.0), 6.0)
        measured = []
        for function in (skeletonize_edt_ridge, edt_ridge_voxels):
            tracemalloc.start()
            try:
                result = function(region, (1.0, 1.0, 1.0), use_memmap=True)
                measured.append((result, tracemalloc.get_traced_memory()[1]))
            finally:
                tracemalloc.stop()
        (volume, volume_peak), (voxels, voxels_peak) = measured
        assert np.array_equal(voxels, np.argwhere(volume))
        return region.size, volume_peak, voxels_peak

    small, small_volume_peak, small_voxels_peak = peaks(40)
    big, big_volume_peak, big_voxels_peak = peaks(160)
    assert big_volume_peak >= big
    assert big_voxels_peak < 1.5 * small_voxels_peak
    assert big_voxels_peak < 0.3 * big


# --- the lumen radius from tiles of the surface ---------------------------------


def test_the_tiled_lumen_radius_holds_no_whole_surface():
    """A speckled mask is nearly all surface: the whole tree's coordinates
    alone are 24 bytes a surface voxel, the tiles' a budget."""
    rng = np.random.default_rng(4)
    mask = rng.random((60, 100, 100)) > 0.5
    voxels = np.column_stack([rng.integers(0, n, 300) for n in mask.shape])

    def peak(**kwargs):
        tracemalloc.start()
        try:
            distance = FeatureDistance(mask, feature_value=False, sampling=(1.0, 0.7, 0.7), **kwargs)
            values = distance.at(voxels)
            return values, tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    whole, whole_peak = peak()
    tiled, tiled_peak = peak(by_block=True, tile_edge=16, tile_cache_bytes=1 << 20)
    assert np.array_equal(tiled, whole)
    assert whole_peak > 10 * mask.size
    assert tiled_peak < 2 * mask.size


# --- components labelled a slab at a time, not into an int32 volume -------------


def _broken_skeleton(seed=0, shape=(18, 50, 50)):
    """Lines broken into pieces in line with each other, specks, and a piece
    one plane thick: components of every size, crossing slab boundaries."""
    rng = np.random.default_rng(seed)
    skeleton = np.zeros(shape, dtype=bool)
    for _ in range(6):
        a = rng.uniform(0, 1, 3) * (np.asarray(shape) - 1)
        b = rng.uniform(0, 1, 3) * (np.asarray(shape) - 1)
        for t in np.linspace(0, 1, 120):
            if (t * 10) % 3 < 2.4:  # gaps along the line
                skeleton[tuple(np.rint(a + t * (b - a)).astype(int))] = True
    skeleton |= rng.random(shape) > 0.995
    skeleton[9, 10:30, 40] = True
    return skeleton


@pytest.fixture
def no_label_volume(monkeypatch):
    """Small slabs, and any volume-shaped non-boolean memmap refused."""
    import haemolynx.preprocessing.skeleton as skeleton_module

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 5_000)
    real_new, real_temporary = skeleton_module.new_memmap_array, skeleton_module.temporary_memmap_array

    def checked_new(shape, dtype, **kwargs):
        assert np.dtype(dtype) == np.dtype(bool), f"a {np.dtype(dtype)} volume {shape}"
        return real_new(shape, dtype, **kwargs)

    def checked_temporary(shape, dtype, **kwargs):
        assert np.dtype(dtype) == np.dtype(bool), f"a {np.dtype(dtype)} volume {shape}"
        return real_temporary(shape, dtype, **kwargs)

    monkeypatch.setattr(skeleton_module, "new_memmap_array", checked_new)
    monkeypatch.setattr(skeleton_module, "temporary_memmap_array", checked_temporary)


@pytest.mark.parametrize("connectivity", [1, 3])
def test_slab_components_count_and_filter_as_the_label_volume(tmp_path, no_label_volume, connectivity):
    from haemolynx.preprocessing.skeleton import (
        _filter_components_by_total_fraction,
        compute_skeleton_connectivity_stats,
    )

    skeleton = _broken_skeleton()
    assert compute_skeleton_connectivity_stats(
        skeleton, connectivity, use_memmap=True, memmap_directory=tmp_path
    ) == compute_skeleton_connectivity_stats(skeleton, connectivity)
    for fraction in (0.01, 0.05, 0.3):
        plain = _filter_components_by_total_fraction(skeleton, fraction, connectivity)
        low = _filter_components_by_total_fraction(
            skeleton, fraction, connectivity, use_memmap=True, memmap_directory=tmp_path
        )
        assert np.array_equal(low, plain)


@pytest.mark.parametrize("voxel_size", [None, (2.0, 0.5, 0.5)])
@pytest.mark.parametrize("connectivity", [1, 3])
def test_slab_components_drop_small_ones_as_the_label_volume(tmp_path, no_label_volume, voxel_size, connectivity):
    from haemolynx.preprocessing.skeleton import drop_small_components

    skeleton = _broken_skeleton(seed=3)
    for min_size in (2, 5, 12):
        plain = drop_small_components(skeleton, min_size=min_size, connectivity=connectivity, voxel_size_zyx=voxel_size)
        low = drop_small_components(
            skeleton, min_size=min_size, connectivity=connectivity, voxel_size_zyx=voxel_size,
            use_memmap=True, memmap_directory=tmp_path,
        )
        assert np.array_equal(low, plain)


@pytest.mark.parametrize("voxel_size", [None, (2.0, 0.5, 0.5)])
def test_slab_components_bridge_and_keep_joinable_pieces_as_the_label_volume(tmp_path, no_label_volume, voxel_size):
    from haemolynx.preprocessing.skeleton import (
        _bridge_metric,
        _drop_small_components_bridging_cannot_join,
        connect_skeleton_components,
    )

    skeleton = _broken_skeleton(seed=5)
    plain = connect_skeleton_components(skeleton, 6, 3, voxel_size_zyx=voxel_size, min_facing_cosine=-1.0)
    low = connect_skeleton_components(
        skeleton, 6, 3, voxel_size_zyx=voxel_size, min_facing_cosine=-1.0,
        use_memmap=True, memmap_directory=tmp_path,
    )
    assert not np.array_equal(plain, skeleton), "the fixture must be bridged"
    assert np.array_equal(low, plain)

    weights, reach = _bridge_metric(3, voxel_size, 1.0, 6)
    kwargs = dict(min_size=6, weights=weights, reach=reach, min_facing_cosine=0.0, connectivity=3, voxel_size_zyx=voxel_size)
    kept = _drop_small_components_bridging_cannot_join(skeleton, **kwargs)
    low_kept = _drop_small_components_bridging_cannot_join(skeleton, use_memmap=True, memmap_directory=tmp_path, **kwargs)
    assert not np.array_equal(kept, skeleton)
    assert np.array_equal(low_kept, kept)


@pytest.mark.parametrize("tile", [False, True])
def test_lee_by_slab_components_is_lee_of_the_volume(tmp_path, monkeypatch, tile):
    """Components across many slabs, one plane thick, boxes overlapping, and
    one too big for RAM (thinned from a box on disk)."""
    from skimage.morphology import skeletonize

    from haemolynx.preprocessing.skeleton import skeletonize_by_component

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 4_000)
    import haemolynx.preprocessing.skeleton as skeleton_module

    monkeypatch.setattr(skeleton_module, "LOW_MEMORY_BLOCK_VOXELS", 4_000)
    mask = np.zeros((24, 40, 40), dtype=bool)
    _capsule_into(mask, (1.0, 1.0, 1.0), (1.0, 5.0, 5.0), (22.0, 30.0, 34.0), 3.2)
    _capsule_into(mask, (1.0, 1.0, 1.0), (4.0, 35.0, 3.0), (20.0, 3.0, 20.0), 2.0)
    mask[12, 2:20, 30:33] = True  # one plane thick
    mask |= np.random.default_rng(1).random(mask.shape) > 0.995
    expected = skeletonize(mask, method="lee")

    low = skeletonize_by_component(
        mask, use_memmap=True, memmap_directory=tmp_path,
        tile_large_components=tile, tile_max_voxels=6_000, tile_halo_voxels=30,
    )

    assert isinstance(low, np.memmap)
    assert np.array_equal(low, expected)
    memmap_support.release_memmap_array(low)


def test_slab_component_sizes_hold_no_volume(tmp_path, monkeypatch):
    """The labels of a slab, not of the volume: four times the stack, the
    same peak."""
    from haemolynx.preprocessing.skeleton import compute_skeleton_connectivity_stats

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 50_000)

    def peak(depth):
        skeleton = np.zeros((depth, 200, 200), dtype=bool)
        skeleton[:, 100, 100] = True
        skeleton[::7, 20, 20:180] = True
        tracemalloc.start()
        try:
            stats = compute_skeleton_connectivity_stats(skeleton, 3, use_memmap=True, memmap_directory=tmp_path)
            return stats, tracemalloc.get_traced_memory()[1], skeleton.size
        finally:
            tracemalloc.stop()

    small_stats, small_peak, _ = peak(40)
    big_stats, big_peak, big = peak(160)
    assert big_stats.voxel_count > small_stats.voxel_count
    assert big_peak < 1.5 * small_peak
    assert big_peak < 0.25 * big


# --- the thin-arm join's component numbers, at the voxels asked about ------------


def test_numbers_anywhere_are_labels_numbers_at_any_voxel(monkeypatch):
    from scipy.ndimage import label

    from haemolynx.preprocessing import slab_labels

    rng = np.random.default_rng(7)
    volume = rng.random((30, 20, 20)) > 0.6
    structure = np.ones((3, 3, 3), dtype=bool)
    expected, count = label(volume, structure=structure)
    known_at = rng.random(volume.shape) > 0.9

    components, known = slab_labels.numbers_at(
        lambda z0, z1: volume[z0:z1], lambda z0, z1: known_at[z0:z1], volume.shape,
        structure=structure, block_voxels=1_200,
    )
    anywhere = slab_labels.NumbersAnywhere(components, known, lambda z0, z1: volume[z0:z1])
    relabelled = []
    real = anywhere._slab_numbers
    monkeypatch.setattr(anywhere, "_slab_numbers", lambda index: relabelled.append(index) or real(index))

    assert components.count == count and components.number_dtype() == np.int32
    every = tuple(np.indices(volume.shape).reshape(3, -1))
    assert np.array_equal(anywhere[every], expected[every])
    assert len(set(relabelled)) == len(components.offsets) > 1
    z, y, x = np.argwhere(volume)[17]
    assert anywhere[(int(z), int(y), int(x))] == expected[z, y, x]


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("reach", [True, False])
def test_the_low_ram_join_reads_bridge_voxels_where_they_land(tmp_path, monkeypatch, cap, reach):
    """The join's numbers at the skeleton only, a bridge's voxels read from
    their slab labelled again (as it was before the flake filter and the
    bridges changed the skeleton): the same skeleton as in RAM."""
    from haemolynx.preprocessing import slab_labels
    from haemolynx.preprocessing import thick_vessels as tv

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 20_000)
    mask = _network(seed=2)
    _skeleton, thick = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, return_thick_mask=True)
    lee = np.asarray(skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, min_radius_um=0.0))
    start = (lee & ~thick) | tv.skeletonize_edt_ridge(thick, SPACING)
    relabelled = []
    real = slab_labels.NumbersAnywhere._slab_numbers

    def counting(self, index):
        relabelled.append(index)
        return real(self, index)

    monkeypatch.setattr(slab_labels.NumbersAnywhere, "_slab_numbers", counting)
    kwargs = dict(
        voxel_size_zyx=SPACING, min_arm_extent_um=2.0, max_bridge_radius_multiple=cap,
        arms_reach_extent=reach,
    )
    plain = tv._join_thin_arms_to_fat_ridge(start.copy(), thick, mask, **kwargs)
    low = tv._join_thin_arms_to_fat_ridge(start.copy(), thick, mask, use_memmap=True, **kwargs)

    assert not np.array_equal(np.asarray(plain), start), "the fixture must be joined"
    assert np.array_equal(np.asarray(low), np.asarray(plain))
    assert relabelled, "some bridge voxel was read where it landed"


# --- vessel and pericyte masks: labels at the foreground voxels only ------------


def _blobs(shape, fraction, sigma, seed):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random(shape), sigma)
    return field > np.quantile(field, 1.0 - fraction)


@pytest.mark.parametrize("structure", [None, np.ones((3, 3, 3), bool)])
@pytest.mark.parametrize("block", [None, 900])
def test_foreground_labels_read_as_the_label_volume(structure, block):
    from scipy.ndimage import find_objects, label

    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    rng = np.random.default_rng(11)
    mask = _blobs((16, 30, 28), 0.15, 1.0, 3)
    expected, count = label(mask, structure=structure)
    labels = ForegroundLabels(mask, structure, block_voxels=block)

    assert labels.count == count and labels.numbers.dtype == np.int32
    assert np.array_equal(labels.sizes, np.bincount(expected.ravel(), minlength=count + 1))
    assert labels.find_objects() == find_objects(expected)
    assert np.array_equal(labels.coords(), np.argwhere(expected))
    points = tuple(np.array([rng.integers(0, n, 300) for n in mask.shape]))
    assert np.array_equal(labels.at(points), expected[points])
    for box in (np.s_[2:9, 0:30, 5:6], np.s_[0:16, 3:4, 0:28], np.s_[5:6, 10:20, 12:27]):
        assert np.array_equal(labels.read(box), expected[box])
    keep = rng.random(count + 1) > 0.5
    assert np.array_equal(labels.write(np.empty(mask.shape, bool), keep, block_voxels=700), keep[expected])
    some = count // 2 + 1
    assert np.array_equal(labels.coords()[labels.members(some)], np.argwhere(expected == some))


def test_foreground_labels_hold_the_foreground_not_the_volume():
    """The same vessels in four times the stack: the same peak, where a
    label volume's grows with the stack."""
    from scipy.ndimage import label

    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    def peaks(depth):
        mask = np.zeros((depth, 160, 160), dtype=bool)
        _capsule_into(mask, (1.0, 1.0, 1.0), (5.0, 20.0, 20.0), (30.0, 140.0, 130.0), 4.0)
        measured = []
        for make in (lambda: label(mask, structure=np.ones((3, 3, 3))), lambda: ForegroundLabels(mask, np.ones((3, 3, 3)), block_voxels=100_000)):
            tracemalloc.start()
            try:
                make()
                measured.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        return mask.size, measured

    small, (_small_label, small_foreground) = peaks(40)
    big, (big_label, big_foreground) = peaks(160)
    assert big_label >= 4 * big
    assert big_foreground < 1.5 * small_foreground
    assert big_foreground < 0.25 * big


def test_pericyte_centroids_are_scipys_bit_for_bit():
    from scipy.ndimage import center_of_mass, label

    from haemolynx.haemodynamics import pericyte_mask as pm

    mask = _blobs((20, 36, 30), 0.03, 0.8, 5)
    labels, count = label(mask)
    expected = np.asarray(center_of_mass(mask, labels, list(range(1, count + 1))), dtype=float)
    spacing = (2.0, 0.37, 0.41)

    centroids, diameters = pm._extract_pericyte_component_properties(mask, spacing)

    assert count > 5
    assert np.array_equal(centroids, expected * np.asarray(spacing).reshape(1, 3))
    assert np.array_equal(pm._extract_pericyte_centroids_physical(mask, spacing), centroids)
    volumes = np.bincount(labels.ravel())[1:] * float(np.prod(spacing))
    assert np.array_equal(diameters, ((6.0 * volumes) / np.pi) ** (1.0 / 3.0))


def _vessel_masks(shape=(20, 40, 44), seed=0):
    art = _blobs(shape, 0.06, 1.2, seed)
    ven = _blobs(shape, 0.06, 1.2, seed + 50) | (np.roll(art, 1, axis=2) & _blobs(shape, 0.5, 2, seed + 9))
    return art, ven


@pytest.mark.parametrize("seed", [0, 1])
def test_low_ram_vessel_mask_clean_ups_give_the_same_masks_on_disk(tmp_path, monkeypatch, seed):
    from haemolynx.graph import large_vessels as lv
    from haemolynx.graph.mask_component_volume import remove_small_vessel_components_by_volume

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 3_000)
    art, ven = _vessel_masks(seed=seed)
    low = dict(low_memory=True, memmap_directory=tmp_path)
    calls = [
        lambda **kw: remove_small_vessel_components_by_volume(art, ven, voxel_size_xyz=(0.5, 0.5, 2.0), min_component_volume_um3=20.0, **kw)[:2],
        lambda **kw: lv.swap_minority_touching_vessel_components(art, ven, max_size_ratio=0.9, min_contact_fraction=0.05, voxel_size_zyx=(2.0, 0.5, 0.5), **kw)[:2],
        lambda **kw: lv.remove_small_opposite_attached_large_vessel_components(art, ven, voxel_size_xyz=(0.5, 0.5, 2.0), max_component_volume_um3=30.0, max_attach_distance_microns=3.0, **kw)[:2],
        lambda **kw: lv.exclude_smaller_overlapping_large_vessel_components(art, ven | (art & _blobs(art.shape, 0.5, 2, seed + 3)), **kw),
    ]
    for call in calls:
        plain = call()
        on_disk = call(**low)
        for a, b in zip(plain, on_disk):
            assert np.array_equal(a, b)
        assert any(isinstance(b, np.memmap) for b in on_disk) or all(b is a for a, b in zip((art, ven), on_disk))


def test_low_ram_vessel_masks_are_read_and_remembered_on_disk(tmp_path, monkeypatch):
    import tifffile

    from haemolynx.io import automated_vessel_assignment as ava

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 3_000)
    art, ven = _vessel_masks(seed=2)
    tifffile.imwrite(tmp_path / "art.tif", art.astype(np.uint8) * 255)
    tifffile.imwrite(tmp_path / "ven.tif", ven.astype(np.uint8) * 255)
    monkeypatch.setattr(ava, "_modified_too_recently", lambda _status: False)
    kwargs = dict(
        mask_role="large", enabled=True, use_ilastik=False,
        arteriole_mask_path=tmp_path / "art.tif", venule_mask_path=tmp_path / "ven.tif",
        image_shape=art.shape, main_voxel_size_xyz=(1.0, 1.0, 1.0),
        min_component_volume_um3=4.0, swap_minority_components=True, swap_max_size_ratio=0.9,
        swap_min_contact_fraction=0.05, remove_small_opposite_attached_components=True,
        opposite_attached_max_component_volume_um3=30.0, exclude_smaller_overlapping_volumes=True,
    )
    ava.clear_vessel_mask_cache()
    plain = ava.load_and_validate_vessel_masks(**kwargs)
    ava.clear_vessel_mask_cache()
    on_disk = ava.load_and_validate_vessel_masks(use_memmap=True, memmap_directory=tmp_path, **kwargs)
    remembered = ava.load_and_validate_vessel_masks(use_memmap=True, memmap_directory=tmp_path, **kwargs)
    ava.clear_vessel_mask_cache()

    for got in (on_disk, remembered):
        assert isinstance(got[0], np.memmap) and isinstance(got[1], np.memmap)
        assert np.array_equal(got[0], plain[0]) and np.array_equal(got[1], plain[1])
        assert got[2:] == plain[2:]
    assert remembered[0] is not on_disk[0]


# --- the overlay projection, a slab at a time -----------------------------------


def _projection_as_it_was(arr):
    """overlay_z_projection before slabs: np.unique and arr == value whole."""
    if arr.dtype == bool:
        return np.max(arr, axis=0).astype(float)
    if np.issubdtype(arr.dtype, np.integer):
        values, counts = np.unique(arr, return_counts=True)
        if values.size == 1:
            return np.max(arr, axis=0)
        if values.size == 2:
            fg = values[values != 0][0] if 0 in values else values[int(np.argmin(counts))]
            return np.max(arr == fg, axis=0).astype(float)
        if values.size <= 4:
            nonzero = values[values != 0]
            if nonzero.size:
                fg = nonzero[int(np.argmin([counts[np.where(values == v)[0][0]] for v in nonzero]))]
                return np.max(arr == fg, axis=0).astype(float)
    return np.max(arr, axis=0)


@pytest.mark.parametrize("kind", ["bool", "zero_one", "one_two", "three_labels", "grey", "float_nan"])
def test_the_overlay_projection_by_slab_is_the_whole_volumes(tmp_path, monkeypatch, kind):
    from haemolynx.visualization.plot import overlay_z_projection

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 500)
    rng = np.random.default_rng(3)
    shape = (14, 20, 22)
    volume = {
        "bool": rng.random(shape) > 0.9,
        "zero_one": (rng.random(shape) > 0.9).astype(np.uint8),
        "one_two": np.where(rng.random(shape) > 0.9, 1, 2).astype(np.uint16),
        "three_labels": rng.choice(np.array([0, 3, 7], dtype=np.int32), shape, p=[0.8, 0.15, 0.05]),
        "grey": rng.integers(0, 4000, shape).astype(np.uint16),
        "float_nan": np.where(rng.random(shape) > 0.97, np.nan, rng.random(shape)).astype(np.float32),
    }[kind]
    on_disk = np.lib.format.open_memmap(tmp_path / "v.npy", mode="w+", dtype=volume.dtype, shape=shape)
    on_disk[...] = volume
    expected = _projection_as_it_was(volume)

    got = overlay_z_projection(on_disk)

    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected, equal_nan=np.issubdtype(expected.dtype, np.floating))
    del on_disk


def test_the_overlay_projection_holds_no_volume(monkeypatch):
    from haemolynx.visualization.plot import overlay_z_projection

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 100_000)

    def peaks(depth):
        volume = np.where(np.random.default_rng(1).random((depth, 200, 200)) > 0.95, 1, 2).astype(np.uint8)
        measured = []
        for project in (_projection_as_it_was, overlay_z_projection):
            tracemalloc.start()
            try:
                project(volume)
                measured.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        return volume.size, measured

    _small, (_was, small_peak) = peaks(40)
    big, (big_was, big_peak) = peaks(160)
    assert big_was >= big  # np.unique sorted a copy, and arr == value was a volume
    assert big_peak < 1.5 * small_peak
    assert big_peak < 0.25 * big


# --- mask recovery's worker files, in the run's memmap directory -----------------


def test_mask_recovery_shares_its_bits_through_the_runs_memmap_directory(tmp_path, monkeypatch):
    from haemolynx.graph import mask_recovery
    from haemolynx.preprocessing import shared_volume_tasks
    from haemolynx.preprocessing.bridge_mask_support import MaskSupport

    mask = np.zeros((10, 30, 30), dtype=bool)
    _capsule_into(mask, (1.0, 1.0, 1.0), (5.0, 3.0, 3.0), (5.0, 26.0, 26.0), 2.5)
    G = nx.MultiGraph()
    line = np.array([[5.0, 3.0 + t, 3.0 + t] for t in range(24)])
    G.add_node(0, pos=line[0])
    G.add_node(1, pos=line[-1])
    G.add_edge(0, 1, voxels=line.tolist())
    support = MaskSupport(mask, (1.0, 1.0, 1.0), memmap_directory=tmp_path)
    assert support.limited_to(5.0, 0.2).memmap_directory == tmp_path
    seen = []
    real = shared_volume_tasks.run_volume_tasks

    def recording(*args, **kwargs):
        seen.append(kwargs.get("memmap_directory"))
        return real(*args, **kwargs)

    monkeypatch.setattr(shared_volume_tasks, "run_volume_tasks", recording)
    index, radii = mask_recovery._centreline_samples(G, support)
    in_workers = mask_recovery._cross_sections(index, radii, support, reach_radii=4.0, margin_um=0.5, workers=2)
    in_process = mask_recovery._cross_sections(index, radii, support, reach_radii=4.0, margin_um=0.5, workers=1)

    assert seen == [tmp_path, tmp_path]
    assert np.array_equal(in_workers.reach, in_process.reach)


# --- coordinates laid out as np.argwhere lays them out ----------------------------


def test_coordinates_standing_in_for_argwhere_have_its_layout():
    """np.argwhere's answer is Fortran-ordered, and a BLAS product of it (a
    covariance, x.T @ x) rounds by the layout: rows picked out of it come
    back C-ordered, and a piece's principal axis then differed in its last
    bits -- once in sign, moving a small-mask continuity bridge."""
    from scipy.ndimage import find_objects, label

    from haemolynx.preprocessing.skeleton import _SlabComponents
    from haemolynx.preprocessing.slab_labels import ForegroundLabels
    from haemolynx.preprocessing.thick_vessels import drawn_voxels

    def layout(a):
        return (a.dtype, a.flags["C_CONTIGUOUS"], a.flags["F_CONTIGUOUS"])

    mask = _blobs((14, 22, 25), 0.08, 1.0, 9)
    structure = np.ones((3, 3, 3), bool)
    labels, count = label(mask, structure=structure)
    found = ForegroundLabels(mask, structure, block_voxels=900)
    assert count > 3
    assert layout(found.coords()) == layout(np.argwhere(labels))
    for number, box in enumerate(find_objects(labels), start=1):
        old = np.argwhere(labels[box] == number) + np.asarray([s.start for s in box]).reshape(1, 3)
        new = found.component_coords(number)
        assert np.array_equal(new, old) and layout(new) == layout(old)
        if len(old) >= 3:
            spacing = np.array([[0.98, 0.98, 0.98]])
            assert np.cov((new * spacing).T).tobytes() == np.cov((old * spacing).T).tobytes()
    assert layout(_SlabComponents(mask, structure, voxels=True).coords) == layout(np.argwhere(labels))
    keys = [np.argwhere(labels == 1), np.argwhere(labels == 2)]
    both = (labels == 1) | (labels == 2)
    assert np.array_equal(drawn_voxels(keys, mask.shape), np.argwhere(both))
    assert layout(drawn_voxels(keys, mask.shape)) == layout(np.argwhere(both))
