"""preprocessing/slab_labels.py: labelling a slab of z-planes at a time finds
scipy's components, numbered as scipy numbers them, with its boxes."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import find_objects, generate_binary_structure, label

from haemolynx.preprocessing import slab_labels

STRUCTURES = {
    "6": generate_binary_structure(3, 1),
    "18": generate_binary_structure(3, 2),
    "26": generate_binary_structure(3, 3),
}


def _planes(volume):
    return lambda z0, z1: volume[z0:z1]


def _relabelled(volume, structure, planes_per_slab):
    """The whole label volume put back together from the slabs."""
    plane = volume.shape[1] * volume.shape[2]
    components = slab_labels.label_by_slab(
        _planes(volume), volume.shape, structure=structure, block_voxels=planes_per_slab * plane
    )
    out = np.zeros(volume.shape, dtype=np.int64)
    for z0, z1, labels, offset in components.replay(_planes(volume)):
        out[z0:z1] = components.numbers(labels, offset)
    return components, out


@pytest.mark.parametrize("connectivity", sorted(STRUCTURES))
@pytest.mark.parametrize("planes_per_slab", [1, 2, 5])
@pytest.mark.parametrize("seed", range(4))
def test_slab_labels_are_scipys_labels(connectivity, planes_per_slab, seed):
    rng = np.random.default_rng(seed)
    volume = rng.random((13, 17, 11)) > rng.uniform(0.55, 0.8)
    structure = STRUCTURES[connectivity]

    components, relabelled = _relabelled(volume, structure, planes_per_slab)
    expected, n = label(volume, structure=structure)

    assert components.count == n
    assert components.planes_per_slab == planes_per_slab
    assert np.array_equal(relabelled, expected)
    for number, box in enumerate(find_objects(expected), start=1):
        assert components.box(number) == box


def test_a_component_joined_only_in_a_later_slab_keeps_scipys_number():
    """A U opening upwards: its two arms are two pieces until the slab that
    holds the bottom of the U, and a component starting between them in
    raster order must still be numbered after the U."""
    volume = np.zeros((9, 5, 9), dtype=bool)
    volume[:, 2, 1] = True  # left arm, all the way down
    volume[:, 2, 7] = True  # right arm
    volume[8, 2, 1:8] = True  # the bottom, in the last plane only
    volume[0, 2, 4] = True  # a speck between the arms in raster order

    components, relabelled = _relabelled(volume, STRUCTURES["6"], 1)
    expected, n = label(volume, structure=STRUCTURES["6"])

    assert components.count == n == 2
    assert np.array_equal(relabelled, expected)


def test_a_diagonal_through_every_slab_is_one_component_only_when_the_structure_says_so():
    volume = np.zeros((8, 8, 8), dtype=bool)
    idx = np.arange(8)
    volume[idx, idx, idx] = True

    for connectivity, count in (("6", 8), ("18", 8), ("26", 1)):
        components, relabelled = _relabelled(volume, STRUCTURES[connectivity], 1)
        assert components.count == count
        assert np.array_equal(relabelled, label(volume, structure=STRUCTURES[connectivity])[0])


def test_an_empty_volume_has_no_components():
    volume = np.zeros((4, 5, 6), dtype=bool)
    components = slab_labels.label_by_slab(_planes(volume), volume.shape, structure=STRUCTURES["26"])
    assert components.count == 0
    keys_components, keys, numbers = slab_labels.voxels_by_component(
        _planes(volume), volume.shape, structure=STRUCTURES["26"]
    )
    assert keys_components.count == 0 and keys.size == 0 and numbers.size == 0


def test_voxels_by_component_lists_every_voxel_in_c_order_with_its_number():
    rng = np.random.default_rng(7)
    volume = rng.random((10, 12, 9)) > 0.7
    expected, _n = label(volume, structure=STRUCTURES["26"])

    _components, keys, numbers = slab_labels.voxels_by_component(
        _planes(volume), volume.shape, structure=STRUCTURES["26"], block_voxels=2 * 12 * 9
    )

    assert np.array_equal(keys, np.flatnonzero(volume))
    assert np.array_equal(numbers, expected.ravel()[keys])


def test_numbers_at_looks_up_the_query_voxels_and_refuses_any_other():
    rng = np.random.default_rng(8)
    volume = rng.random((10, 12, 9)) > 0.7
    query = rng.random(volume.shape) > 0.5
    expected, _n = label(volume, structure=STRUCTURES["18"])

    _components, lookup = slab_labels.numbers_at(
        _planes(volume), _planes(query), volume.shape, structure=STRUCTURES["18"], block_voxels=3 * 12 * 9
    )

    z, y, x = np.nonzero(query)
    assert np.array_equal(lookup[z, y, x], expected[z, y, x])
    assert lookup[int(z[0]), int(y[0]), int(x[0])] == expected[z[0], y[0], x[0]]
    outside = np.argwhere(~query)[0]
    with pytest.raises(KeyError):
        lookup[tuple(int(v) for v in outside)]


def _planes_of(volumes, z0, z1, params):
    return np.asarray(volumes[0][z0:z1], dtype=bool)


def _labels_seen(volumes, z0, z1, labels, params):
    return np.unique(labels[labels > 0]).astype(np.int64)


def _write_numbers(volumes, z0, z1, labels, offset, n, params):
    from haemolynx.preprocessing.shared_volume_tasks import open_whole_file

    out = open_whole_file(params["out"])
    if labels is None:
        # Every component here answers alike: the first one's number.
        out[z0:z1] = np.where(np.asarray(volumes[0][z0:z1], dtype=bool), volumes[1][offset + 1], 0)
    else:
        out[z0:z1] = slab_labels.offset_lookup(labels, offset, volumes[1])
    del out
    return labels is not None


@pytest.mark.parametrize("workers", [1, 2])
def test_slabs_labelled_in_workers_are_scipys_labels_numbered_alike(tmp_path, workers):
    """Each slab labelled in a worker and the faces joined here, in z order:
    scipy's components with scipy's numbers and boxes, every slab's own
    labels handed back with its offset, and a replay that labels again only
    the slabs whose components do not answer alike."""
    from haemolynx.preprocessing.memmap_support import new_memmap_array, release_memmap_array
    from haemolynx.preprocessing.shared_volume_tasks import whole_file_spec

    rng = np.random.default_rng(11)
    volume = rng.random((13, 17, 11)) > 0.7
    volume[:, 2, 3] = True  # a column through every slab
    plane = volume.shape[1] * volume.shape[2]
    structure = STRUCTURES["26"]

    components, seen = slab_labels.label_by_slab_in_workers(
        _planes_of, (volume,), volume.shape, structure=structure, collect=_labels_seen,
        block_voxels=2 * plane, workers=workers, memmap_directory=tmp_path,
    )
    expected, n = label(volume, structure=structure)
    serial = slab_labels.label_by_slab(_planes(volume), volume.shape, structure=structure, block_voxels=2 * plane)

    assert components.count == n
    assert np.array_equal(components.number, serial.number)
    assert np.array_equal(components.slab_counts, serial.slab_counts)
    for number, box in enumerate(find_objects(expected), start=1):
        assert components.box(number) == box
    for (offset, labels), (z0, z1, slab_offset, count) in zip(seen, components.slabs()):
        assert offset == slab_offset and np.array_equal(labels, np.arange(1, count + 1))

    table = components.number.copy()  # each offset label's own component number
    out = new_memmap_array(volume.shape, np.int64, directory=tmp_path)
    again = slab_labels.replay_in_workers(
        components, _planes_of, _write_numbers, (volume, table),
        relabel=lambda offset, count: len(set(table[offset + 1:offset + count + 1])) > 1,
        params=dict(out=whole_file_spec(out)), workers=workers, memmap_directory=tmp_path,
    )
    out.flush()
    assert np.array_equal(np.asarray(out), expected)
    single = [len(set(table[o + 1:o + c + 1])) <= 1 for _z0, _z1, o, c in components.slabs()]
    assert again == [not s for s in single], "only a slab with two components labelled again"
    release_memmap_array(out)
