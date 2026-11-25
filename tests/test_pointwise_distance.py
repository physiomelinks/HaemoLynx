"""``preprocessing.pointwise_distance.FeatureDistance``: the distance
transform read at chosen voxels, and the memory graph building turns on for
voxels it asks about again and again."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from haemolynx.preprocessing.pointwise_distance import FeatureDistance

SPACING = (1.0, 0.98, 0.98)


def _mask() -> np.ndarray:
    """Two tubes of different radii and a speck, in a small anisotropic stack."""
    z, y, x = np.indices((14, 30, 40))
    mask = ((y - 10) ** 2 + (z - 7) ** 2 <= 16) | ((x - 28) ** 2 + (z - 6) ** 2 <= 6)
    mask[2, 25, 5] = True
    return mask


def _voxels(mask, count=400, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.integers(0, n, count) for n in mask.shape])


def test_a_lookup_is_scipys_own_transform_at_those_voxels():
    mask = _mask()
    voxels = _voxels(mask)

    looked_up = FeatureDistance(mask, feature_value=False, sampling=SPACING).at(voxels)

    expected = distance_transform_edt(mask, sampling=SPACING)[tuple(voxels.T)]
    assert looked_up == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_remembered_distances_are_the_same_and_asked_of_the_tree_once(monkeypatch):
    mask = _mask()
    voxels = _voxels(mask)
    plain = FeatureDistance(mask, feature_value=False, sampling=SPACING)
    remembering = FeatureDistance(mask, feature_value=False, sampling=SPACING, remember=10_000)
    measured = []
    real = remembering._measure
    monkeypatch.setattr(remembering, "_measure", lambda v: measured.append(len(v)) or real(v))

    first = remembering.at(voxels)
    again = remembering.at(voxels[::-1])

    assert np.array_equal(first, plain.at(voxels))
    assert np.array_equal(again, first[::-1])
    assert measured == [len(voxels)], "the second call is answered from memory"
    some_new = np.vstack([voxels[:5], [[0, 0, 0]]])
    remembering.at(some_new)
    assert measured[-1] <= 1, "only a voxel not seen before is measured"


def test_memory_keeps_no_more_voxels_than_it_is_allowed():
    mask = _mask()
    remembering = FeatureDistance(mask, feature_value=False, sampling=SPACING, remember=50)

    remembering.at(_voxels(mask, count=400))

    assert len(remembering._known) == 50
    assert len(FeatureDistance(mask, feature_value=False)._known) == 0
    plain = FeatureDistance(mask, feature_value=False)
    plain.at(_voxels(mask))
    assert len(plain._known) == 0, "off by default"


def test_out_of_range_voxels_behave_as_without_memory():
    """A negative index wraps and one past the end raises, as plain indexing
    does -- never a remembered value of another voxel."""
    mask = _mask()
    plain = FeatureDistance(mask, feature_value=False, sampling=SPACING)
    remembering = FeatureDistance(mask, feature_value=False, sampling=SPACING, remember=10_000)
    wrapped = np.array([[-1, -1, -1]])

    assert np.array_equal(remembering.at(wrapped), plain.at(wrapped))
    with pytest.raises(IndexError):
        remembering.at(np.array([[mask.shape[0], 0, 0]]))
    assert len(remembering._known) == 0


def test_the_mask_as_bits_gives_the_same_distances_without_reading_the_mask():
    """``packed`` (a ``PackedMask`` of the features) answers whether a queried
    voxel is itself a feature: the same distances, and in range the volume
    itself is not read. Out of range keeps plain indexing's behaviour."""
    from haemolynx.preprocessing.bridge_mask_support import PackedMask

    mask = _mask()
    voxels = _voxels(mask)
    plain = FeatureDistance(mask, feature_value=False, sampling=SPACING)
    packed = FeatureDistance(
        mask, feature_value=False, sampling=SPACING, packed=PackedMask.from_mask(mask)
    )
    expected = plain.at(voxels)

    class Unread(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("the volume was read")

    packed.features = mask.view(Unread)
    assert np.array_equal(packed.at(voxels), expected)

    packed.features = mask
    wrapped = np.array([[-1, -1, -1]])
    assert np.array_equal(packed.at(wrapped), plain.at(wrapped))
    with pytest.raises(IndexError):
        packed.at(np.array([[mask.shape[0], 0, 0]]))


# --- by_block: the low-RAM option's tiles of the surface -----------------------


def _all_voxels(shape) -> np.ndarray:
    return np.indices(shape).reshape(len(shape), -1).T


@pytest.mark.parametrize("tile_edge", [3, 5, 8, 96])
@pytest.mark.parametrize("spacing", [SPACING, (1.0, 1.0, 1.0), (2.1, 0.33, 0.47)])
def test_tiles_of_the_surface_give_the_whole_trees_distances_bit_for_bit(tile_edge, spacing):
    mask = _mask()
    voxels = _all_voxels(mask.shape)
    whole = FeatureDistance(mask, feature_value=False, sampling=spacing)
    tiled = FeatureDistance(mask, feature_value=False, sampling=spacing, by_block=True, tile_edge=tile_edge)

    assert tiled.has_surface and tiled.surface is None
    assert np.array_equal(tiled.at(voxels), whole.at(voxels))


def test_tiles_find_a_surface_many_tiles_away():
    """Deep inside a block wider than several tiles: the tile a voxel is in,
    and the next ring out, hold no surface at all."""
    mask = np.zeros((40, 44, 46), dtype=bool)
    mask[2:38, 3:41, 4:43] = True
    mask[20, 22, 23] = False  # one hole in the middle, a nearer surface for some
    voxels = _all_voxels(mask.shape)[::7]
    whole = FeatureDistance(mask, feature_value=False, sampling=(1.3, 0.7, 0.9))
    tiled = FeatureDistance(mask, feature_value=False, sampling=(1.3, 0.7, 0.9), by_block=True, tile_edge=4)

    assert np.array_equal(tiled.at(voxels), whole.at(voxels))


def test_tiles_measure_features_too_and_the_background_side():
    mask = _mask()
    voxels = _all_voxels(mask.shape)
    whole = FeatureDistance(mask, feature_value=True, sampling=SPACING)
    tiled = FeatureDistance(mask, feature_value=True, sampling=SPACING, by_block=True, tile_edge=6)
    assert np.array_equal(tiled.at(voxels), whole.at(voxels))


def test_tiles_kept_to_their_budget_still_give_the_same_distances():
    mask = _mask()
    voxels = _voxels(mask, count=2000, seed=5)
    whole = FeatureDistance(mask, feature_value=False, sampling=SPACING)
    tiled = FeatureDistance(
        mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4, tile_cache_bytes=4000
    )

    assert np.array_equal(tiled.at(voxels), whole.at(voxels))
    cache = tiled._tiles._cache
    biggest = max(len(surface) for surface, _tree in cache.values()) * 62
    assert tiled._tiles._held <= 4000 + biggest


def test_tiles_with_no_surface_anywhere():
    for value in (False, True):
        volume = np.full((5, 6, 7), value)
        assert not FeatureDistance(volume, feature_value=False, by_block=True).has_surface
        assert not FeatureDistance(volume, feature_value=False).has_surface


def test_tiles_out_of_range_voxels_behave_as_the_whole_tree():
    mask = _mask()
    whole = FeatureDistance(mask, feature_value=False, sampling=SPACING)
    tiled = FeatureDistance(mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=5)
    wrapped = np.array([[-1, -1, -1], [-3, 2, -7]])

    assert np.array_equal(tiled.at(wrapped), whole.at(wrapped))
    with pytest.raises(IndexError):
        tiled.at(np.array([[mask.shape[0], 0, 0]]))


def test_tiles_read_the_bits_when_packed_and_remember_as_before():
    from haemolynx.preprocessing.bridge_mask_support import PackedMask

    mask = _mask()
    voxels = _voxels(mask, count=600)
    expected = FeatureDistance(mask, feature_value=False, sampling=SPACING).at(voxels)
    tiled = FeatureDistance(
        mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=6,
        packed=PackedMask.from_mask(mask), remember=10_000,
    )

    class Unread(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("the volume was read")

    tiled.features = mask.view(Unread)
    assert np.array_equal(tiled.at(voxels), expected)
    assert np.array_equal(tiled.at(voxels[::-1]), expected[::-1])


def test_mask_support_low_memory_radius_is_the_same():
    from haemolynx.preprocessing.bridge_mask_support import MaskSupport

    mask = _mask()
    points = _voxels(mask, count=500, seed=8) * np.asarray(SPACING)
    plain = MaskSupport(mask, SPACING)
    low = MaskSupport(mask, SPACING, low_memory=True)

    assert np.array_equal(low.radius(points), plain.radius(points))
    assert low._distance._tiles is not None and plain._distance._tiles is None
    assert low.limited_to(5.0, 0.1).low_memory
    assert low.limited_to(5.0, 0.1)._distance is low._distance


def test_radius_sampler_low_memory_is_the_same():
    from haemolynx.graph.assemble import mask_radius_sampler

    mask = _mask()
    plain = mask_radius_sampler(mask, SPACING, 1.0)
    low = mask_radius_sampler(mask, SPACING, 1.0, low_memory=True)
    for point in _voxels(mask, count=200, seed=9) * np.asarray(SPACING):
        assert low(point) == plain(point)
    assert low.feature_distance._tiles is not None


# --- making tiles: cheaper, the same surface; a budget that follows free memory -


@pytest.mark.parametrize("shape", [(9, 11, 13), (1, 6, 7), (2, 1, 5), (17,), (6, 8), (3, 4, 5, 6)])
def test_the_box_dilation_is_scipys_with_a_full_structure(shape):
    from scipy.ndimage import binary_dilation

    from haemolynx.preprocessing.pointwise_distance import _dilate_box

    rng = np.random.default_rng(len(shape) * 100 + sum(shape))
    for density in (0.02, 0.3, 0.9):
        volume = rng.random(shape) < density
        expected = binary_dilation(volume, structure=np.ones((3,) * len(shape), dtype=bool))
        got = _dilate_box(volume)
        assert np.array_equal(got, expected)
        assert not np.shares_memory(got, volume)


def test_a_tile_is_made_once_while_memory_is_free_and_again_when_it_is_not(monkeypatch):
    from haemolynx.haemodynamics import resistance
    from haemolynx.preprocessing import pointwise_distance

    mask = _mask()
    voxels = _voxels(mask, count=3000, seed=11)
    expected = FeatureDistance(mask, feature_value=False, sampling=SPACING).at(voxels)
    # A floor far below one tile, so only the memory free keeps tiles; the
    # memory free read at every tile made.
    monkeypatch.setattr(pointwise_distance, "SURFACE_TILE_CACHE_BYTES", 1)
    monkeypatch.setattr(pointwise_distance, "_SURFACE_TILE_RECHECK_BYTES", 0)

    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 64 << 30)
    roomy = FeatureDistance(mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4)
    for part in np.array_split(voxels, 30):
        roomy.at(part)
    assert np.array_equal(roomy.at(voxels), expected)
    tiles = roomy._tiles
    assert tiles.made == len(tiles._cache) > 10  # never one made twice

    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 0)
    tight = FeatureDistance(mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4)
    for part in np.array_split(voxels, 30):
        tight.at(part)
    assert np.array_equal(tight.at(voxels), expected)
    assert tight._tiles.made > tiles.made and len(tight._tiles._cache) == 1

    # Memory gone later: the next tile the roomy lookup makes drops all but
    # a quarter of what it holds (what it holds would be free once dropped).
    _key, (dropped, _tree) = tiles._cache.popitem()
    tiles._held -= len(dropped) * 62
    held, made = tiles._held, tiles.made
    biggest = max(len(surface) for surface, _tree in tiles._cache.values()) * 62
    roomy.at(voxels[::-1])
    assert tiles.made > made and tiles._held <= 0.25 * (held + biggest) + biggest


def test_lookups_alive_together_share_one_allowance(monkeypatch):
    from haemolynx.haemodynamics import resistance
    from haemolynx.preprocessing import pointwise_distance

    import gc

    gc.collect()  # no lookup left alive by another test
    mask = _mask()
    monkeypatch.setattr(pointwise_distance, "SURFACE_TILE_CACHE_BYTES", 1)
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 4_000_000)
    first = FeatureDistance(mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4)
    second = FeatureDistance(mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4)
    first._tiles._held = 600_000
    # A quarter of (4 MB free + 0.6 MB held), less the other's 0.6 MB.
    assert second._tiles._allowance() == int(0.25 * 4_600_000) - 600_000
    assert first._tiles._allowance() == int(0.25 * 4_600_000)
    fixed = FeatureDistance(
        mask, feature_value=False, sampling=SPACING, by_block=True, tile_edge=4, tile_cache_bytes=5000
    )
    assert fixed._tiles._allowance() == 5000
    del first
    gc.collect()
    assert second._tiles._allowance() == int(0.25 * 4_000_000)
