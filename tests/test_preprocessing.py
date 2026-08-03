"""Tests for preprocessing module."""
import pytest
import numpy as np

from ImageLynx.preprocessing import (
    bridge_gaps,
    close_binary_mask,
    skeletonize_3d,
    preprocess_skeleton_for_graph,
)


def test_bridge_gaps(small_binary_3d):
    result = bridge_gaps(small_binary_3d, max_gap=1)
    assert result.shape == small_binary_3d.shape
    assert np.any(result)


def test_bridge_gaps_dilates_but_close_binary_mask_does_not():
    """Why the mask cleanup path uses close_binary_mask rather than bridge_gaps.

    bridge_gaps is a plain dilation - it never erodes back - so an isolated solid object
    grows by max_gap in every direction. On a vessel mask that adds a voxel of radius to
    every vessel unconditionally, and because cross-sectional area goes as the square of the
    radius the bias is not size-neutral: +1 voxel is +125% area on a 2-voxel radius but only
    +36% on a 6-voxel one, so narrow capillaries are inflated hardest. A closing bridges the
    same gaps while leaving an isolated object's size untouched.
    """
    volume = np.zeros((21, 21, 21), dtype=bool)
    volume[8:13, 8:13, 8:13] = True   # isolated 5x5x5 cube, well clear of the borders
    original_count = int(volume.sum())

    dilated = bridge_gaps(volume, max_gap=1)
    closed = close_binary_mask(volume, radius=1)

    assert int(dilated.sum()) > original_count, "bridge_gaps is expected to dilate"
    assert dilated[7, 10, 10], "bridge_gaps should expand outward by one voxel"

    assert int(closed.sum()) == original_count, (
        "closing changed the size of an isolated object "
        f"({original_count} -> {int(closed.sum())} voxels)"
    )
    assert not closed[7, 10, 10], "closing must not expand boundaries"


def test_close_binary_mask_bridges_a_gap_between_thick_structures():
    """The behaviour the mask path relies on: a closing does still bridge a narrow gap.

    This holds for thick structures. It does not hold for a 1-voxel-thick skeleton, where
    the erosion step removes the bridge again - which is why bridge_gaps is kept for that
    case rather than being replaced outright.
    """
    volume = np.zeros((21, 21, 21), dtype=bool)
    volume[8:13, 8:13, 4:10] = True    # two thick blocks separated by a 1-voxel gap at x=10
    volume[8:13, 8:13, 11:17] = True

    closed = close_binary_mask(volume, radius=1)

    assert closed[10, 10, 10], "the gap between two thick structures was not bridged"


def test_close_binary_mask_is_a_noop_for_non_positive_radius():
    volume = np.zeros((10, 10, 10), dtype=bool)
    volume[5, 5, 5] = True
    assert np.array_equal(close_binary_mask(volume, radius=0), volume)


def test_skeletonize_3d(small_binary_3d):
    out = skeletonize_3d(small_binary_3d)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool


def test_preprocess_skeleton_for_graph(small_binary_3d):
    out = preprocess_skeleton_for_graph(small_binary_3d, min_branch_length=2)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool
