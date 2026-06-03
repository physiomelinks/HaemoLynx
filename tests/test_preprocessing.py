"""Tests for preprocessing module."""
import pytest
import numpy as np

from ImageLynx.preprocessing import (
    bridge_gaps,
    preprocess_skeleton_for_graph,
    skeletonize_volume,
)


def test_bridge_gaps(small_binary_3d):
    result = bridge_gaps(small_binary_3d, max_gap=1)
    assert result.shape == small_binary_3d.shape
    assert np.any(result)


def test_skeletonize_volume(small_binary_3d):
    out = skeletonize_volume(small_binary_3d)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool


def test_preprocess_skeleton_for_graph(small_binary_3d):
    out = preprocess_skeleton_for_graph(small_binary_3d, min_branch_length=2)
    assert out.shape == small_binary_3d.shape
    assert out.dtype == bool
