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
