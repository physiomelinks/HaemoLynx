"""Tests for preprocessing module."""
import pytest
import numpy as np

from haemolynx.preprocessing import (
    bridge_gaps,
    compute_skeleton_connectivity_stats,
    inter_component_gap_distances,
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


def test_compute_skeleton_connectivity_stats_on_empty_skeleton():
    empty = np.zeros((8, 8, 8), dtype=bool)
    stats = compute_skeleton_connectivity_stats(empty)
    assert stats.voxel_count == 0
    assert stats.n_components == 0
    assert stats.largest_fraction == 0.0
    assert stats.component_sizes == ()


def test_compute_skeleton_connectivity_stats_counts_components_largest_first():
    skel = np.zeros((10, 10, 10), dtype=bool)
    skel[1, 1, 1:5] = True  # 4-voxel component
    skel[8, 8, 8:9] = True  # 1-voxel component, far away
    stats = compute_skeleton_connectivity_stats(skel)
    assert stats.voxel_count == 5
    assert stats.n_components == 2
    assert stats.largest_component_size == 4
    assert stats.component_sizes == (4, 1)
    assert stats.largest_fraction == pytest.approx(4 / 5)


def test_inter_component_gap_distances_empty_for_single_component():
    skel = np.zeros((8, 8, 8), dtype=bool)
    skel[1, 1, 1:5] = True
    assert inter_component_gap_distances(skel).size == 0

    also_empty = np.zeros((8, 8, 8), dtype=bool)
    assert inter_component_gap_distances(also_empty).size == 0


def test_inter_component_gap_distances_measures_known_gaps():
    skel = np.zeros((1, 1, 20), dtype=bool)
    skel[0, 0, 0] = True
    skel[0, 0, 5] = True  # 5 voxels from the first point
    skel[0, 0, 15] = True  # 10 voxels from the second point, 15 from the first
    gaps = inter_component_gap_distances(skel, component_connectivity=1)
    assert gaps.size == 3  # 3 unordered pairs among 3 components
    assert gaps[0] == pytest.approx(5.0)
    assert gaps[1] == pytest.approx(10.0)
    assert gaps[2] == pytest.approx(15.0)


def test_inter_component_gap_distances_scales_with_voxel_size():
    skel = np.zeros((1, 1, 10), dtype=bool)
    skel[0, 0, 0] = True
    skel[0, 0, 5] = True
    gaps = inter_component_gap_distances(
        skel, component_connectivity=1, voxel_size_zyx=(1.0, 1.0, 2.0)
    )
    assert gaps == pytest.approx([10.0])
