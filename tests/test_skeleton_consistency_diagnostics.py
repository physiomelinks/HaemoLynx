"""How well a skeleton represents its mask, a graph represents its skeleton,
and a graph represents the original mask directly: hand-verified geometry,
no pipeline involved.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.diagnostics import (
    diagnose_graph_mask_consistency,
    diagnose_skeleton_graph_consistency,
    format_graph_mask_consistency_report,
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


def test_the_mask_is_read_via_canonical_binarisation_not_a_bare_nonzero_test():
    """Regression test for a genuine miscalibration found on a real noisy
    fixture (seven_vessel_noisy_3d.tif): treating every nonzero voxel as
    foreground (a bare ``!= 0`` test) read 95.5% of that volume as "mask",
    when the pipeline's own canonical binarisation
    (``io.load._to_binary_volume_for_skeletonization``) reads only ~1% of
    it as foreground once the 1/2-labelled convention it actually uses is
    accounted for. Here: a background label of 1 and a minority foreground
    label of 2, with no zero voxel anywhere in the array -- a bare
    ``!= 0`` test would (wrongly) call the entire volume foreground.
    """
    mask = np.full((6, 6, 6), 1, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 2  # minority label: the true foreground, 8 voxels
    skeleton = np.zeros((6, 6, 6), dtype=bool)
    skeleton[2:4, 2:4, 2:4] = True  # sits exactly on the true foreground

    report = diagnose_skeleton_mask_consistency(skeleton, mask)

    assert report["mask_voxel_count"] == 8  # not 216, which a bare != 0 test would give
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_the_discretisation_margin_lets_a_well_centred_skeleton_of_a_thin_square_vessel_score_full_coverage():
    """Regression test for a genuine geometric miscalibration found on a
    real thin-vessel fixture: a discretised medial axis running along one
    corner of a 2x2-voxel cross-section rod is, correctly, up to a full
    voxel diagonal away from the section's *opposite* corner -- there is
    no single-voxel-wide line that could do better for a vessel this thin.
    Without the discretisation margin, comparing against the bare inscribed
    radius reads that opposite corner as unexplained even though the
    skeleton is exactly where a perfect one would be.
    """
    mask = np.zeros((8, 8, 10), dtype=bool)
    mask[4:6, 4:6, :] = True  # a rod, 2x2 voxels in (z, y), full length in x
    skeleton = np.zeros((8, 8, 10), dtype=bool)
    skeleton[4, 4, :] = True  # runs along one corner of the cross-section

    report = diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["coverage_fraction"] == pytest.approx(1.0)


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


# --- diagnose_graph_mask_consistency -----------------------------------------


def test_an_empty_mask_is_fully_explained_by_the_graph_trivially():
    G = nx.MultiGraph()
    report = diagnose_graph_mask_consistency(G, np.zeros((5, 5, 5), dtype=bool))
    assert report == {
        "mask_voxel_count": 0,
        "graph_voxel_count": 0,
        "explained_voxel_count": 0,
        "coverage_fraction": 1.0,
    }


def test_a_centred_graph_edge_explains_the_whole_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True  # a 2-voxel-wide bar along x

    voxels = [[float(z), 4.0, float(x)] for z in range(3, 7) for x in range(10)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=9.0, voxels=voxels)  # runs along one wall of the bar

    report = diagnose_graph_mask_consistency(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["mask_voxel_count"] == int(mask.sum())
    assert report["explained_voxel_count"] == report["mask_voxel_count"]
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_a_graph_far_from_the_mask_explains_none_of_it():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_edge(0, 1, length=0.0, voxels=[[0.0, 0.0, 0.0]])

    report = diagnose_graph_mask_consistency(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["explained_voxel_count"] == 0
    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_an_empty_graph_against_a_real_mask_explains_none_of_it():
    mask = np.ones((3, 3, 3), dtype=bool)
    report = diagnose_graph_mask_consistency(nx.MultiGraph(), mask)
    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_an_edge_with_no_voxels_is_skipped_not_a_crash():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 1.0, 1.0]))
    G.add_edge(0, 1, length=1.0)  # no "voxels" key at all

    report = diagnose_graph_mask_consistency(G, mask)

    assert report["graph_voxel_count"] == 0
    assert report["coverage_fraction"] == pytest.approx(0.0)


def test_the_mask_is_read_via_canonical_binarisation_for_the_graph_check_too():
    """Same Phase-9 canonical-binarisation fix as
    diagnose_skeleton_mask_consistency above, applied here since this check
    reads the same segmented image."""
    mask = np.full((6, 6, 6), 1, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 2  # minority label: the true foreground, 8 voxels

    voxels = [
        [float(z), float(y), float(x)]
        for z in range(2, 4)
        for y in range(2, 4)
        for x in range(2, 4)
    ]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=1.0, voxels=voxels)

    report = diagnose_graph_mask_consistency(G, mask)

    assert report["mask_voxel_count"] == 8  # not 216, which a bare != 0 test would give
    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_the_discretisation_margin_lets_a_well_centred_graph_edge_of_a_thin_square_vessel_score_full_coverage():
    """Same geometric-miscalibration fix as the skeleton/mask check above,
    applied here for the same reason: this check uses the same
    local-radius-plus-margin criterion against the same kind of thin
    vessel."""
    mask = np.zeros((8, 8, 10), dtype=bool)
    mask[4:6, 4:6, :] = True  # a rod, 2x2 voxels in (z, y), full length in x

    voxels = [[4.0, 4.0, float(x)] for x in range(10)]  # one corner of the cross-section
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=9.0, voxels=voxels)

    report = diagnose_graph_mask_consistency(G, mask, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_the_graph_mask_consistency_report_format_is_a_readable_one_liner():
    report = {"explained_voxel_count": 6, "mask_voxel_count": 8, "coverage_fraction": 0.75}
    text = format_graph_mask_consistency_report(report)
    assert "6" in text and "8" in text and "75.0%" in text
