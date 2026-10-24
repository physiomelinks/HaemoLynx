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
from scipy.ndimage import distance_transform_edt

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
def test_the_low_ram_join_carries_bridge_numbers_along_the_bridge(tmp_path, monkeypatch, cap, reach):
    """The join's numbers at the skeleton only. A bridge's new fat voxels take
    theirs along the bridge drawn -- `allowed`'s from the arm voxel it starts
    at, the fat structures' from the tree voxel it stopped against -- and only
    a voxel neither reaches is read from its slab labelled again (as it was
    before the flake filter and the bridges changed the skeleton): every
    number carried is the one that read gives, the skeleton is the in-RAM
    one, and with nothing carried it is too, read slab by slab. On the
    patella (2026-10-20) the slab reads were four fifths of the join."""
    from haemolynx.preprocessing import slab_labels
    from haemolynx.preprocessing import thick_vessels as tv

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 20_000)
    mask = _network(seed=2)
    _skeleton, thick = skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, return_thick_mask=True)
    lee = np.asarray(skeletonize_thickness_gated(mask, voxel_size_zyx=SPACING, min_radius_um=0.0))
    start = (lee & ~thick) | tv.skeletonize_edt_ridge(thick, SPACING)
    relabelled = []
    real_numbers = slab_labels.NumbersAnywhere._slab_numbers

    def counting(self, index):
        relabelled.append(index)
        return real_numbers(self, index)

    lookups = []
    real_init = slab_labels.NumbersAnywhere.__init__

    def keeping(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        lookups.append(self)

    carried = []
    real_carry = tv._numbers_carried_along

    def recording(path, inside, first, number, step):
        out = real_carry(path, inside, first, number, step)
        carried.append((step, out))
        return out

    monkeypatch.setattr(slab_labels.NumbersAnywhere, "_slab_numbers", counting)
    monkeypatch.setattr(slab_labels.NumbersAnywhere, "__init__", keeping)
    monkeypatch.setattr(tv, "_numbers_carried_along", recording)
    kwargs = dict(
        voxel_size_zyx=SPACING, min_arm_extent_um=2.0, max_bridge_radius_multiple=cap,
        arms_reach_extent=reach,
    )
    plain = tv._join_thin_arms_to_fat_ridge(start.copy(), thick, mask, **kwargs)
    low = tv._join_thin_arms_to_fat_ridge(start.copy(), thick, mask, use_memmap=True, **kwargs)

    assert not np.array_equal(np.asarray(plain), start), "the fixture must be joined"
    assert np.array_equal(np.asarray(low), np.asarray(plain))
    assert not relabelled, "every bridge voxel's numbers came along its bridge"
    roi = [n for n in lookups if getattr(n.foreground, "__name__", "") == "_roi_planes_then"]
    allowed = [n for n in lookups if n not in roi]
    assert len(allowed) == 1 and len(roi) == (0 if reach and cap is None else 1)
    checked = {+1: 0, -1: 0}
    for step, numbers in carried:
        if not numbers:
            continue
        lookup = (allowed if step == +1 else roi)[0]
        voxels = tuple(np.array(list(numbers)).T)
        assert np.array_equal(lookup[voxels], np.array(list(numbers.values())))
        checked[step] += len(numbers)
    assert checked[+1] > 0 and (checked[-1] > 0) == bool(roi)

    relabelled.clear()
    monkeypatch.setattr(tv, "_numbers_carried_along", lambda *args: {})
    looked_up = tv._join_thin_arms_to_fat_ridge(start.copy(), thick, mask, use_memmap=True, **kwargs)
    assert np.array_equal(np.asarray(looked_up), np.asarray(plain))
    assert relabelled, "with nothing carried, a bridge voxel is read where it landed"


def _ridge_trees(coords, labels, scale):
    """The join's `_ridge_tree_for`: each structure's ridge voxels and their tree."""
    from scipy.spatial import cKDTree

    trees = {}

    def ridge_of(number):
        if number not in trees:
            mine = coords[labels == number]
            trees[number] = (mine, cKDTree(mine.astype(np.float64) * scale)) if len(mine) else None
        return trees[number]

    return ridge_of


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.0, 0.98, 0.98), (2.0, 0.3, 0.3)])
def test_the_nearest_fat_of_a_structure_is_the_voxel_its_whole_tree_finds(spacing):
    """The low-RAM join's nearest fat voxel to an arm, from the ridge's tree
    and the trees of doubling size the bridges' voxels go into: exactly
    `_nearest_point` with a cKDTree of the structure's fat voxels in the
    order the in-RAM join lists them (`fat_coords[same_component]`), with the
    grid's ties and voxels two bridges drew. Without taking ties from that
    whole tree, 1-4% of these answers differed."""
    from scipy.spatial import cKDTree

    from haemolynx.preprocessing import thick_vessels as tv

    scale = np.asarray(spacing, dtype=np.float64)
    shape = (24, 24, 24)
    rng = np.random.default_rng(3)
    ridge_keys = np.unique(rng.integers(0, 24 ** 3, 400))
    ridge = tv._coords_of(ridge_keys, shape)

    def structure_of(coords):  # one number a voxel, as `allowed`'s labels give it
        return (1 + (coords[:, 2] >= 12)).astype(np.int32)

    labels = structure_of(ridge)
    fat = tv._FatByStructure(_ridge_trees(ridge, labels, scale), ridge_keys, labels, scale, shape)
    every, every_label = ridge, labels
    asked = 0
    for _bridge in range(80):
        drawn = rng.integers(0, 24, (int(rng.integers(1, 12)), 3)).astype(np.intp)
        if rng.random() < 0.3:  # across an earlier bridge, or the ridge
            drawn = np.concatenate([drawn, every[rng.integers(0, len(every), 3)]])
        fat.add(drawn, structure_of(drawn))
        every = np.concatenate([every, drawn])
        every_label = np.concatenate([every_label, structure_of(drawn)])
        for _arm in range(4):
            number = int(rng.integers(1, 3))
            points_um = rng.integers(0, 24, (int(rng.integers(1, 8)), 3)) * scale
            mine = every[every_label == number]
            expected = tv._nearest_point(points_um, mine, cKDTree(mine.astype(np.float64) * scale))
            assert fat.nearest(number, points_um) == expected
            asked += 1
    assert 0 < fat.full_trees_built < asked / 4


def test_a_tie_for_the_nearest_fat_is_the_whole_trees_choice():
    """Two fat voxels as near an arm, one on the ridge and one a bridge drew:
    which a cKDTree returns depends on how it split its points, so the answer
    is a tree of them all's, built then and kept until the structure gains a
    voxel. Two arm points as near the fat, each with one voxel nearest it, are
    the k-d tree's own distances either way (the first wins); one of them
    with two voxels as near it needs the whole tree again."""
    from scipy.spatial import cKDTree

    from haemolynx.preprocessing import thick_vessels as tv

    scale = np.ones(3)
    shape = (10, 10, 10)
    ridge = np.array([[5, 5, 2]], dtype=np.intp)
    labels = np.array([1], dtype=np.int32)
    fat = tv._FatByStructure(
        _ridge_trees(ridge, labels, scale), tv._keys_of(ridge, shape), labels, scale, shape
    )
    fat.add(np.array([[5, 5, 8]]), np.array([1], dtype=np.int32))
    every = np.array([[5, 5, 2], [5, 5, 8]], dtype=np.intp)

    def whole(points):
        return tv._nearest_point(points, every, cKDTree(every * scale))

    between = np.array([[5.0, 5.0, 5.0]])
    assert fat.nearest(1, between) == whole(between)
    assert fat.full_trees_built == 1
    assert fat.nearest(1, between) == whole(between) and fat.full_trees_built == 1, "kept"

    fat.add(np.array([[1, 1, 1]]), np.array([1], dtype=np.int32))
    every = np.concatenate([every, [[1, 1, 1]]])
    assert fat.nearest(1, np.array([[5.0, 5.0, 6.0]])) == (0, 2.0, (5, 5, 8))
    assert fat.full_trees_built == 1, "one voxel nearest: no tree of them all"
    two = np.array([[5.0, 5.0, 4.0], [5.0, 5.0, 6.0]])
    assert fat.nearest(1, two) == whole(two) == (0, 2.0, (5, 5, 2))
    assert fat.full_trees_built == 1
    tied = np.array([[1.0, 1.0, 4.0], [5.0, 5.0, 5.0]])
    assert fat.nearest(1, tied) == whole(tied)
    assert fat.full_trees_built == 2


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.0, 0.98, 0.98), (1.5, 0.6, 0.6)])
def test_the_cap_radius_from_a_box_transformed_once_is_each_windows_own(spacing, monkeypatch):
    """The bridge cap's radius over a window of a box distances_at_points
    transforms whole, read from that box's radius map made once: the float32
    of the window's largest distance, bit for bit, the empty window's 0
    included. Boxes are let go, least recently read first, past twice
    LOW_MEMORY_BLOCK_VOXELS voxels between them."""
    from haemolynx.preprocessing import blockwise_edt
    from haemolynx.preprocessing import thick_vessels as tv

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 40_000)
    rng = np.random.default_rng(5)
    thick = np.zeros((30, 40, 40), dtype=bool)
    for _ in range(6):
        a = rng.uniform(0, 1, 3) * np.array(thick.shape) * spacing
        _capsule_into(thick, spacing, a, a + rng.normal(size=3) * 12.0, rng.uniform(2.0, 6.0))
    boxes = [
        (slice(0, 30), slice(0, 30), slice(0, 40)),
        (slice(2, 30), slice(10, 40), slice(5, 40)),
        (slice(0, 25), slice(0, 40), slice(0, 35)),
    ]
    radii = tv._BoxRadii(thick, spacing)
    for _ in range(60):
        box = boxes[int(rng.integers(0, len(boxes)))]
        assert radii.fits(box)
        corner = [int(rng.integers(b.start, b.stop)) for b in box]
        window = tuple(
            slice(c, min(b.stop, c + int(rng.integers(1, 12)))) for c, b in zip(corner, box)
        )
        points = np.argwhere(thick[window]) + np.array([w.start for w in window])
        found = blockwise_edt.distances_at_points(
            lambda region: thick[region], box, points, sampling=spacing
        )
        expected = float(np.float32(found.max())) if len(found) else 0.0
        assert radii.largest(box, window) == expected
    assert len(radii._boxes) == 2 and radii.made > len(boxes), "two boxes fit; the third evicts one"
    assert not radii.fits((slice(0, 30), slice(0, 40), slice(0, 40)))


@pytest.mark.parametrize("spacing", [(1.0, 0.98, 0.98), (1.5, 0.6, 0.6)])
def test_the_cap_radius_from_voxels_measured_once_is_each_windows_own(spacing, monkeypatch):
    """Over a box too large to transform whole, the bridge cap's radius from
    each fat voxel's distance measured once and kept in cells -- windows of
    nearby arms overlapping, cells let go past the budget and measured again
    -- is each window's own reading by distances_at_points, bit for bit."""
    from haemolynx.preprocessing import blockwise_edt
    from haemolynx.preprocessing import thick_vessels as tv

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 6_000)
    rng = np.random.default_rng(11)
    thick = np.zeros((24, 60, 60), dtype=bool)
    for _ in range(6):
        a = rng.uniform(0, 1, 3) * np.array(thick.shape) * spacing
        _capsule_into(thick, spacing, a, a + rng.normal(size=3) * 20.0, rng.uniform(2.0, 9.0))
    box = (slice(1, 24), slice(0, 58), slice(2, 60))
    radii = tv._CellRadii(thick, spacing, {})
    assert not tv._BoxRadii.fits(box)
    fat = np.argwhere(thick[box]) + [b.start for b in box]
    ends = fat[rng.integers(0, len(fat), 50)]
    ends = np.concatenate([ends, ends[:10] + 1])  # arms beside earlier ones
    for end in ends:
        window = tuple(
            slice(max(b.start, int(v) - 5), min(b.stop, int(v) + 6)) for v, b in zip(end, box)
        )
        points = np.argwhere(thick[window]) + np.array([w.start for w in window])
        found = blockwise_edt.distances_at_points(
            lambda region: thick[region], box, points, sampling=spacing
        )
        expected = float(np.float32(found.max())) if len(found) else 0.0
        assert radii.largest(box, window) == expected
    assert radii.measured < sum(
        int(np.count_nonzero(thick[tuple(slice(max(b.start, int(v) - 5), min(b.stop, int(v) + 6))
                                       for v, b in zip(end, box))])) for end in ends
    ), "overlapping windows share measured voxels"
    assert radii._held <= 2 * memmap_support.LOW_MEMORY_BLOCK_VOXELS + 27 * 16 ** 3


def test_numbers_are_carried_only_through_neighbours_inside_the_mask():
    """Two 26-adjacent voxels of a mask are in one of its components: a
    number goes along a path while each step is to a neighbour inside, and
    every voxel it reaches is in the first one's component."""
    from scipy.ndimage import label

    from haemolynx.preprocessing.thick_vessels import _numbers_carried_along

    rng = np.random.default_rng(5)
    mask = rng.random((12, 12, 12)) > 0.45
    labels, _ = label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    for _ in range(200):
        voxel = np.array([rng.integers(0, 12) for _ in range(3)])
        path = [tuple(int(v) for v in voxel)]
        for _ in range(15):
            voxel = np.clip(voxel + rng.integers(-1, 2, 3), 0, 11)
            path.append(tuple(int(v) for v in voxel))
        if rng.random() < 0.3:
            path.insert(8, (0, 0, 0))  # a jump in the path
        for first, step in ((0, +1), (len(path) - 1, -1)):
            numbers = _numbers_carried_along(path, mask, first, 7, step)
            assert all(n == 7 for n in numbers.values())
            if numbers:
                assert set(labels[tuple(np.array(list(numbers)).T)]) == {labels[path[first]]}
            # it runs exactly as far as the first voxel outside or jump
            i, reached = first, []
            while 0 <= i < len(path) and mask[path[i]] and (
                not reached or max(abs(a - b) for a, b in zip(path[i], reached[-1])) <= 1
            ):
                reached.append(path[i])
                i += step
            assert set(numbers) == set(reached)

    line = [(0, 0, x) for x in range(5)]
    mask = np.zeros((1, 1, 5), dtype=bool)
    mask[0, 0, [0, 1, 3, 4]] = True
    assert _numbers_carried_along(line, mask, 0, 3, +1) == {(0, 0, 0): 3, (0, 0, 1): 3}
    assert _numbers_carried_along(line, mask, 4, 9, -1) == {(0, 0, 4): 9, (0, 0, 3): 9}
    assert _numbers_carried_along(line, mask, 2, 9, -1) == {}


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

    from haemolynx.io import vessel_masks as ava

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


# --- vessel-mask dilation without a distance volume ------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_low_ram_mask_dilation_is_the_distance_transforms(tmp_path, monkeypatch, seed):
    from haemolynx.graph.large_vessels import dilate_binary_mask_by_microns

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 3_000)
    rng = np.random.default_rng(seed)
    mask = _blobs((14, 26, 30), rng.uniform(0.005, 0.2), 1.0, seed)
    spacing = [(1.0, 1.0, 1.0), (0.98, 0.98, 0.98), (2.0, 0.41, 0.41)][seed % 3]
    for dilation in (0.5, 0.98, 1.96, 3.0, 4.9):
        # The whole transform both modes made before the in-RAM run took the
        # blocks too (faster, a quarter of the memory).
        reference = mask | (distance_transform_edt(~mask, sampling=spacing) <= dilation)
        plain = dilate_binary_mask_by_microns(mask, dilation, spacing)
        low = dilate_binary_mask_by_microns(mask, dilation, spacing, low_memory=True, memmap_directory=tmp_path)
        assert isinstance(low, np.memmap)
        assert type(plain) is np.ndarray
        assert np.array_equal(plain, reference)
        assert np.array_equal(low, reference)


def test_an_empty_mask_dilates_to_an_empty_mask_and_a_full_one_stays_full(tmp_path):
    """scipy reads a volume with no feature as that far from one at (-1, 0, 0),
    so the whole transform "dilated" an empty mask into a blob at the corner,
    and the block path copied it: a vessel touching three faces, which stump
    detection took for a large vessel cut off by the image. Nothing now, in
    either mode, the 2D path included."""
    from haemolynx.graph.large_vessel_network import find_large_vessel_mask_stump_points
    from haemolynx.graph.large_vessels import dilate_binary_mask_by_microns

    spacing = (1.0, 0.5, 0.5)
    empty = np.zeros((6, 9, 10), bool)
    assert (distance_transform_edt(~empty, sampling=spacing) <= 1.6).any(), "scipy's corner, still there"
    for dilation in (1.6, 15.0):
        plain = dilate_binary_mask_by_microns(empty, dilation, spacing)
        low = dilate_binary_mask_by_microns(empty, dilation, spacing, low_memory=True, memmap_directory=tmp_path)
        assert type(plain) is np.ndarray and plain.dtype == bool and not plain.any()
        assert isinstance(low, np.memmap) and not np.asarray(low).any()
        assert find_large_vessel_mask_stump_points(plain, voxel_size_zyx=spacing) == []
    assert not dilate_binary_mask_by_microns(np.zeros((9, 10), bool), 1.6, (0.5, 0.5)).any()

    full = np.ones((6, 9, 10), bool)
    assert dilate_binary_mask_by_microns(full, 1.6, spacing).all()
    assert np.asarray(
        dilate_binary_mask_by_microns(full, 1.6, spacing, low_memory=True, memmap_directory=tmp_path)
    ).all()


def test_mask_dilation_holds_no_distance_volume(tmp_path, monkeypatch):
    """Four times the stack, the same vessels: the low-RAM peak stays a
    block's feature transform, and the in-RAM one is the mask returned (a
    byte a voxel), not a float64 distance volume (8 bytes a voxel and
    scipy's working arrays), which both modes made before."""
    from haemolynx.graph.large_vessels import dilate_binary_mask_by_microns

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 60_000)

    def peaks(depth):
        mask = np.zeros((depth, 120, 120), dtype=bool)
        _capsule_into(mask, (1.0, 0.6, 0.6), (5.0, 10.0, 10.0), (30.0, 60.0, 60.0), 3.0)
        measured = []
        for kwargs in ({}, dict(low_memory=True, memmap_directory=tmp_path)):
            tracemalloc.start()
            try:
                dilate_binary_mask_by_microns(mask, 2.0, (1.0, 0.6, 0.6), **kwargs)
                measured.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        return mask.size, measured

    _small, (_in_ram, small_low) = peaks(40)
    big, (big_in_ram, big_low) = peaks(160)
    assert big_in_ram < 3 * big
    assert big_low < 1.5 * small_low


def test_low_ram_confidence_assignment_dilates_as_before(tmp_path, monkeypatch):
    import networkx as nx

    from haemolynx.graph import (
        select_terminal_nodes_from_large_vessel_masks_progressive_dilation_confidence as select,
    )

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 2_000)
    G = nx.MultiGraph()
    positions = [(4.0, 3.0, 2.0), (4.0, 3.0, 9.0), (4.0, 3.0, 16.0), (4.0, 12.0, 9.0), (4.0, 18.0, 9.0)]
    for node, pos in enumerate(positions):
        G.add_node(node, pos=np.array(pos))
    for u, v in ((0, 1), (1, 2), (1, 3), (3, 4)):
        G.add_edge(u, v, voxels=[positions[u], positions[v]], length=1.0)
    arteriole = np.zeros((9, 22, 20), bool)
    venule = np.zeros((9, 22, 20), bool)
    arteriole[2:7, 1:5, 0:3] = True
    venule[2:7, 17:21, 7:12] = True
    kwargs = dict(
        voxel_size_zyx=(1.0, 1.0, 1.0), max_dilation_microns=10.0, dilation_step_microns=2.0,
        confidence_margin=0.02, minimum_confidence=0.02, topology_penalty=0.0,
    )
    plain = select(G, large_arteriole_mask=arteriole, large_venule_mask=venule, **kwargs)
    low = select(
        G, large_arteriole_mask=arteriole, large_venule_mask=venule, low_memory=True,
        memmap_directory=tmp_path, **kwargs,
    )
    assert plain["input_nodes"] or plain["output_nodes"]
    for key in ("input_nodes", "output_nodes", "unresolved_nodes", "node_confidence"):
        assert repr(low[key]) == repr(plain[key])


# --- small-mask continuity, read at the voxels it asks about ----------------------


def _broken_tubes(shape, rng, count, radius, spacing):
    mask = np.zeros(shape, bool)
    points = np.indices(shape, dtype=float).reshape(3, -1).T * spacing
    extent = np.asarray(shape) * spacing
    for _ in range(count):
        start = rng.uniform(0.1, 0.9, 3) * extent
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        offset = points - start
        along = offset @ direction
        t = np.clip(along, 0.0, rng.uniform(5.0, 0.6 * extent.max()))
        radial = np.linalg.norm(offset - t[:, None] * direction, axis=1)
        gap = np.abs((along % 9.0) - 4.5) < 0.8
        mask |= ((radial <= radius) & ~gap).reshape(shape)
    return mask


@pytest.mark.parametrize("seed", range(6))
def test_low_ram_small_mask_continuity_is_the_in_ram_one(tmp_path, monkeypatch, seed):
    from haemolynx.graph import enforce_small_vessel_mask_continuity

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 4_000)
    rng = np.random.default_rng(seed)
    spacing = np.array([[1.5, 0.5, 0.5], [0.98, 0.98, 0.98], [1.0, 1.0, 1.0]][seed % 3])
    shape = (14, 40, 44)
    small_art = _broken_tubes(shape, rng, 4, 1.4, spacing)
    small_ven = _broken_tubes(shape, rng, 4, 1.4, spacing) & ~small_art
    large_art = _broken_tubes(shape, rng, 1, 2.6, spacing) & ~small_art & ~small_ven
    large_ven = _broken_tubes(shape, rng, 1, 2.6, spacing) & ~small_art & ~small_ven & ~large_art
    kwargs = dict(
        small_arteriole_mask=small_art, small_venule_mask=small_ven,
        large_arteriole_mask=large_art if seed % 2 else None, large_venule_mask=large_ven,
        voxel_size_zyx=tuple(spacing), enforce_cylinder_only=bool(seed % 2), min_cylindricality=0.3,
        max_bridge_distance_microns=12.0, corridor_max_distance_microns=6.0,
        opposite_exclusion_distance_microns=1.5, min_facing_cosine=0.5, max_axis_angle_degrees=60.0,
        max_radius_ratio=4.0,
    )
    plain = enforce_small_vessel_mask_continuity(**kwargs)
    low = enforce_small_vessel_mask_continuity(low_memory=True, memmap_directory=tmp_path, **kwargs)

    assert sum(plain["stats"][kind]["accepted_bridges"] for kind in ("arteriole", "venule")) > 0
    for key in ("small_arteriole_mask", "small_venule_mask"):
        assert isinstance(low[key], np.memmap)
        assert np.array_equal(low[key], plain[key])

    def without_times(stats):
        return {
            k: ({kk: vv for kk, vv in v.items() if "elapsed" not in kk} if isinstance(v, dict) else v)
            for k, v in stats.items()
        }

    assert without_times(low["stats"]) == without_times(plain["stats"])


class _WholeTransform:
    """The whole-volume transform the in-RAM continuity read before
    :class:`~haemolynx.graph.mask_continuity._DistanceAt` replaced it."""

    def __init__(self, volume, *, feature_value, sampling, memmap_directory=None):
        values = np.asarray(volume[tuple(slice(None) for _ in volume.shape)], dtype=bool)
        self.distances = distance_transform_edt(values != bool(feature_value), sampling=sampling)

    def __getitem__(self, index):
        return self.distances[index]


@pytest.mark.parametrize("seed", range(8))
def test_in_ram_small_mask_continuity_is_the_whole_transforms(monkeypatch, seed):
    """Normal runs read continuity's distances at the voxels its gates ask
    about too (four times faster, an eighth of the memory): the same masks
    and the same decisions as the whole transforms it made before."""
    from haemolynx.graph import enforce_small_vessel_mask_continuity
    from haemolynx.graph import mask_continuity

    rng = np.random.default_rng(100 + seed)
    spacing = np.array([[1.5, 0.5, 0.5], [0.98, 0.98, 0.98], [1.0, 1.0, 1.0], [2.0, 0.41, 0.41]][seed % 4])
    shape = (16, 44, 48)
    small_art = _broken_tubes(shape, rng, 5, 1.4, spacing)
    small_ven = _broken_tubes(shape, rng, 5, 1.4, spacing) & ~small_art
    large_art = _broken_tubes(shape, rng, 1, 2.6, spacing) & ~small_art & ~small_ven
    large_ven = _broken_tubes(shape, rng, 1, 2.6, spacing) & ~small_art & ~small_ven & ~large_art
    kwargs = dict(
        small_arteriole_mask=small_art, small_venule_mask=small_ven,
        large_arteriole_mask=large_art if seed % 2 else None, large_venule_mask=large_ven if seed % 3 else None,
        voxel_size_zyx=tuple(spacing), enforce_cylinder_only=bool(seed % 2), min_cylindricality=0.3,
        max_bridge_distance_microns=12.0, corridor_max_distance_microns=6.0,
        opposite_exclusion_distance_microns=1.5, min_facing_cosine=0.5, max_axis_angle_degrees=60.0,
        max_radius_ratio=4.0,
    )
    # No GPU is asked for, so none is looked for.
    monkeypatch.setattr(mask_continuity, "_init_gpu_edt_backend", lambda: pytest.fail("looked for a GPU"))
    pointwise = enforce_small_vessel_mask_continuity(**kwargs)
    monkeypatch.setattr(mask_continuity, "_DistanceAt", _WholeTransform)
    whole = enforce_small_vessel_mask_continuity(**kwargs)

    for key in ("small_arteriole_mask", "small_venule_mask"):
        assert type(pointwise[key]) is np.ndarray
        assert np.array_equal(pointwise[key], whole[key])

    def without_times(stats):
        return {
            k: ({kk: vv for kk, vv in v.items() if "elapsed" not in kk} if isinstance(v, dict) else v)
            for k, v in stats.items()
        }

    assert without_times(pointwise["stats"]) == without_times(whole["stats"])
    if seed in (0, 1):
        assert sum(pointwise["stats"][kind]["accepted_bridges"] for kind in ("arteriole", "venule")) > 0


def test_continuity_asked_for_a_gpu_without_one_reads_pointwise(monkeypatch):
    """*use_gpu_acceleration* with no GPU: the pointwise reads, not a CPU
    whole transform -- the same masks either way."""
    from haemolynx.graph import enforce_small_vessel_mask_continuity
    from haemolynx.graph import mask_continuity

    rng = np.random.default_rng(3)
    spacing = (1.5, 0.5, 0.5)
    small_art = _broken_tubes((14, 40, 44), rng, 4, 1.4, np.asarray(spacing))
    small_ven = _broken_tubes((14, 40, 44), rng, 4, 1.4, np.asarray(spacing)) & ~small_art
    kwargs = dict(
        small_arteriole_mask=small_art, small_venule_mask=small_ven,
        large_arteriole_mask=None, large_venule_mask=None, voxel_size_zyx=spacing,
    )
    monkeypatch.setattr(mask_continuity, "_init_gpu_edt_backend", lambda: False)
    monkeypatch.setattr(mask_continuity, "_edt", lambda *a, **k: pytest.fail("made a whole transform"))
    asked = enforce_small_vessel_mask_continuity(use_gpu_acceleration=True, **kwargs)
    plain = enforce_small_vessel_mask_continuity(**kwargs)
    for key in ("small_arteriole_mask", "small_venule_mask"):
        assert np.array_equal(asked[key], plain[key])


def test_continuity_distance_reader_is_scipys_transform():
    from scipy.ndimage import distance_transform_edt

    from haemolynx.graph.mask_continuity import _DistanceAt
    from haemolynx.preprocessing.memmap_support import MaskUnion

    a = _blobs((10, 18, 20), 0.05, 1.0, 1)
    b = _blobs((10, 18, 20), 0.05, 1.0, 2)
    every = tuple(np.indices(a.shape).reshape(3, -1))
    for spacing in ((1.0, 1.0, 1.0), (1.5, 0.5, 0.5)):
        inside = _DistanceAt(a, feature_value=False, sampling=spacing)
        assert np.array_equal(inside[every], distance_transform_edt(a, sampling=spacing)[every])
        union = _DistanceAt(MaskUnion(a, b), feature_value=True, sampling=spacing)
        assert np.array_equal(union[every], distance_transform_edt(~(a | b), sampling=spacing)[every])
        for volume in (np.zeros(a.shape, bool), np.ones(a.shape, bool)):
            reader = _DistanceAt(volume, feature_value=True, sampling=spacing)
            assert np.array_equal(reader[every], distance_transform_edt(~volume, sampling=spacing)[every])


def test_a_continuity_bridge_drawn_in_its_box_is_the_whole_volumes():
    from haemolynx.graph.mask_continuity import (
        _bridge_box_from_line,
        _bridge_mask_from_line,
        _line_indices_zyx,
    )

    rng = np.random.default_rng(4)
    shape = (9, 20, 24)
    for _ in range(30):
        a, b = (rng.integers(-2, np.asarray(shape) + 2) for _ in range(2))
        line = _line_indices_zyx(a, b)
        for radius in (0, 1, 3):
            for spacing in (None, (2.0, 0.5, 0.5)):
                whole = _bridge_mask_from_line(line, shape, radius_voxels=radius, sampling_zyx=spacing)
                boxed = _bridge_box_from_line(line, shape, radius_voxels=radius, sampling_zyx=spacing)
                drawn = np.zeros(shape, bool)
                if boxed is not None:
                    drawn[boxed[0]] = boxed[1]
                assert np.array_equal(drawn, whole)


# --- arteriole | venule, read or written without a volume in RAM -----------------


def test_mask_union_reads_as_the_union():
    from haemolynx.graph._helpers import points_inside_mask
    from haemolynx.preprocessing.memmap_support import MaskUnion

    a = _blobs((10, 18, 20), 0.05, 1.0, 1)
    b = _blobs((10, 18, 20), 0.05, 1.0, 2)
    union = MaskUnion(a, None, b)
    assert union.shape == a.shape and np.array_equal(union[...], a | b)
    assert np.array_equal(union[2:7, ::3, 5], (a | b)[2:7, ::3, 5])
    points = np.random.default_rng(0).uniform(-1, 19, (300, 3))
    assert np.array_equal(
        points_inside_mask(points, union, voxel_size_zyx=(1.0, 1.0, 1.0)),
        points_inside_mask(points, a | b, voxel_size_zyx=(1.0, 1.0, 1.0)),
    )
    with pytest.raises(ValueError):
        MaskUnion(a, np.zeros((3, 3, 3), bool))


def test_low_ram_global_mask_is_the_union_on_disk(tmp_path, monkeypatch):
    from haemolynx.io import as_global_vessel_mask

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 700)
    a = _blobs((10, 18, 20), 0.05, 1.0, 1)
    b = _blobs((10, 18, 20), 0.05, 1.0, 2)
    union = as_global_vessel_mask(a, b, use_memmap=True, memmap_directory=tmp_path)
    assert isinstance(union, np.memmap) and np.array_equal(union, a | b)
    assert np.array_equal(as_global_vessel_mask(None, b, use_memmap=True), b)


def test_low_ram_restriction_union_is_the_in_ram_one(tmp_path, monkeypatch):
    from haemolynx.pipeline import default_schema, stages

    a = _blobs((10, 18, 20), 0.05, 1.0, 1)
    b = _blobs((10, 18, 20), 0.05, 1.0, 2)
    c = _blobs((10, 18, 20), 0.05, 1.0, 3)

    def fake_load(**kwargs):
        if kwargs["mask_role"] == "large":
            return a, b, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        return c, None, (1.0, 1.0, 1.0), None

    monkeypatch.setattr(stages.io, "load_and_validate_vessel_masks", fake_load)
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 700)
    settings = default_schema().defaults()
    settings.update(
        skeleton_thick_vessel_restrict_to_mask="both",
        skeleton_thick_vessel_restrict_to_mask_warn_below=0.0,
        use_large_vessel_masks=True,
        use_small_vessel_masks_for_boundary_assignment=True,
        automated_vessel_assignment=True,
    )
    image = a | b | c
    plain = stages.thick_vessel_restriction_mask(
        image, dict(settings, use_memmap_loading=False, memmap_directory=None), (1.0, 1.0, 1.0)
    )
    low = stages.thick_vessel_restriction_mask(
        image, dict(settings, use_memmap_loading=True, memmap_directory=str(tmp_path)), (1.0, 1.0, 1.0)
    )
    assert isinstance(low, np.memmap)
    assert np.array_equal(low, plain) and np.array_equal(plain, image)


# --- segmentation cleanup's reconnect: labels at the mask's voxels ----------------


def test_low_ram_cleanup_reconnect_labels_no_volume(tmp_path, monkeypatch):
    from haemolynx.preprocessing import segmentation_cleanup as cleanup

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 3_000)
    zz, yy, xx = np.indices((12, 24, 44), dtype=float)
    mask = np.zeros((12, 24, 44), dtype=bool)
    for z, y, pieces in ((4, 5, ((0, 14), (20, 34))), (8, 16, ((2, 12), (17, 30), (35, 43)))):
        tube = (zz - z) ** 2 + (yy - y) ** 2 <= 1.0
        for x0, x1 in pieces:
            mask |= tube & (xx >= x0) & (xx <= x1)
    made = []
    real = cleanup.new_memmap_array

    def recording(shape, dtype, **kwargs):
        made.append(np.dtype(dtype))
        return real(shape, dtype, **kwargs)

    monkeypatch.setattr(cleanup, "new_memmap_array", recording)
    args = dict(voxel_size_zyx=(1.0, 1.0, 1.0), max_bridge_distance_um=10.0)
    plain, plain_stats = cleanup.reconnect_vessel_like_components(mask, **args)
    low, low_stats = cleanup.reconnect_vessel_like_components(
        mask, use_memmap=True, memmap_directory=tmp_path, **args
    )

    assert plain_stats["accepted_bridges"] > 0
    assert np.array_equal(np.asarray(low), plain) and low_stats == plain_stats
    assert np.dtype(np.int32) not in made
