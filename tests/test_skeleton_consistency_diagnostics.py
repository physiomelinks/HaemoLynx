"""How well a skeleton represents its mask, and a graph represents its
skeleton: hand-verified geometry, no pipeline involved.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.diagnostics import (
    diagnose_skeleton_graph_consistency,
    format_skeleton_graph_consistency_report,
)
from haemolynx.preprocessing.skeleton_consistency import (
    diagnose_skeleton_mask_consistency,
    format_skeleton_mask_consistency_report,
)


# --- diagnose_skeleton_mask_consistency -------------------------------------


def test_an_empty_mask_is_fully_explained_trivially():
    empty = np.zeros((5, 5, 5), dtype=bool)
    report = diagnose_skeleton_mask_consistency(empty, empty)
    assert report == {
        "mask_voxel_count": 0,
        "explained_voxel_count": 0,
        "coverage_fraction": 1.0,
    }


def test_a_centred_skeleton_explains_the_whole_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True  # a 2-voxel-wide bar along x
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[3:7, 4, :] = True  # runs along one wall of the bar -- within 1 voxel of every point

    report = diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["mask_voxel_count"] == int(mask.sum())
    assert report["explained_voxel_count"] == report["mask_voxel_count"]
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_a_skeleton_far_from_the_mask_explains_none_of_it():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[0, 0, 0] = True

    report = diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["explained_voxel_count"] == 0
    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_an_empty_skeleton_against_a_real_mask_explains_none_of_it():
    mask = np.ones((3, 3, 3), dtype=bool)
    empty_skeleton = np.zeros((3, 3, 3), dtype=bool)

    report = diagnose_skeleton_mask_consistency(empty_skeleton, mask)

    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_a_partially_covering_skeleton_reports_the_exact_fraction():
    """Two separate 1-voxel-radius blobs, only one of which has a skeleton
    voxel through its centre -- hand counted: half the mask is explained."""
    mask = np.zeros((5, 9, 5), dtype=bool)
    mask[2, 1, 2] = True  # blob A, isolated single voxel
    mask[2, 7, 2] = True  # blob B, isolated single voxel
    skeleton = np.zeros((5, 9, 5), dtype=bool)
    skeleton[2, 1, 2] = True  # sits exactly on blob A; nowhere near blob B

    report = diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["mask_voxel_count"] == 2
    assert report["explained_voxel_count"] == 1
    assert report["coverage_fraction"] == pytest.approx(0.5)


def test_the_mask_consistency_report_format_is_a_readable_one_liner():
    report = {"explained_voxel_count": 7, "mask_voxel_count": 10, "coverage_fraction": 0.7}
    text = format_skeleton_mask_consistency_report(report)
    assert "7" in text and "10" in text and "70.0%" in text


# --- diagnose_skeleton_graph_consistency ------------------------------------


def test_an_empty_skeleton_is_fully_traced_trivially():
    G = nx.MultiGraph()
    report = diagnose_skeleton_graph_consistency(G, np.zeros((5, 5, 5), dtype=bool))
    assert report == {
        "skeleton_voxel_count": 0,
        "graph_voxel_count": 0,
        "matched_voxel_count": 0,
        "coverage_fraction": 1.0,
    }


def test_an_edge_that_exactly_traces_the_skeleton_scores_full_coverage():
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:8, 5, 5] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 5.0, 5.0]))
    G.add_node(1, pos=np.array([7.0, 5.0, 5.0]))
    voxels = [[float(z), 5.0, 5.0] for z in range(2, 8)]
    G.add_edge(0, 1, length=5.0, voxels=voxels)

    report = diagnose_skeleton_graph_consistency(G, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["skeleton_voxel_count"] == 6
    assert report["matched_voxel_count"] == 6
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_a_graph_missing_most_of_the_skeleton_reports_the_shortfall():
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:8, 5, 5] = True  # 6 voxels

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 5.0, 5.0]))
    G.add_node(1, pos=np.array([4.0, 5.0, 5.0]))
    voxels = [[float(z), 5.0, 5.0] for z in range(2, 4)]  # only 2 of 6 voxels
    G.add_edge(0, 1, length=2.0, voxels=voxels)

    report = diagnose_skeleton_graph_consistency(G, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["matched_voxel_count"] == 2
    assert report["coverage_fraction"] == pytest.approx(2 / 6)


def test_an_edge_with_no_voxels_is_skipped_not_a_crash():
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 2] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 1.0, 1.0]))
    G.add_edge(0, 1, length=1.0)  # no "voxels" key at all

    report = diagnose_skeleton_graph_consistency(G, skeleton)

    assert report["graph_voxel_count"] == 0
    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_a_voxel_point_outside_the_volume_is_clipped_not_a_crash():
    """A collapse-patched endpoint or a smoothed point can round just past
    the array edge; this must not raise an IndexError."""
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[4, 4, 4] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([4.0, 4.0, 4.0]))
    G.add_node(1, pos=np.array([9.0, 9.0, 9.0]))
    G.add_edge(0, 1, length=1.0, voxels=[[4.0, 4.0, 4.0], [9.0, 9.0, 9.0]])

    report = diagnose_skeleton_graph_consistency(G, skeleton)

    assert report["matched_voxel_count"] == 1


def test_voxel_size_converts_physical_microns_back_to_voxel_indices():
    """voxels are stored in physical microns; a non-isotropic voxel size
    must be divided out correctly to land back on the right index."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[4, 4, 4] = True  # one voxel, at physical position (8, 2, 1) with spacing below

    voxel_size_zyx = (2.0, 0.5, 0.25)
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([8.0, 2.0, 1.0]))
    G.add_node(1, pos=np.array([8.0, 2.0, 1.0]))
    G.add_edge(0, 1, length=0.0, voxels=[[8.0, 2.0, 1.0]])

    report = diagnose_skeleton_graph_consistency(G, skeleton, voxel_size_zyx=voxel_size_zyx)

    assert report["matched_voxel_count"] == 1
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_the_graph_consistency_report_format_is_a_readable_one_liner():
    report = {"matched_voxel_count": 3, "skeleton_voxel_count": 4, "coverage_fraction": 0.75}
    text = format_skeleton_graph_consistency_report(report)
    assert "3" in text and "4" in text and "75.0%" in text
