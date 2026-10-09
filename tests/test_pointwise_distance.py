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
