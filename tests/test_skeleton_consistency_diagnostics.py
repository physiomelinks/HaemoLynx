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


def test_the_mask_consistency_report_formatter_reads_the_real_functions_own_keys():
    """The format test above hand-types a dict matching the formatter's own
    ``.get(key, default)`` calls -- if diagnose_skeleton_mask_consistency's
    actual keys ever drifted from what the formatter reads, that test would
    still pass (both sides would have been edited to match), silently
    defaulting the formatter's output to 0/100% forever. Run the real
    function's own output through the formatter instead, so a key rename on
    one side without the other shows up as a wrong number in the text."""
    mask = np.zeros((5, 9, 5), dtype=bool)
    mask[2, 1, 2] = True
    mask[2, 7, 2] = True
    skeleton = np.zeros((5, 9, 5), dtype=bool)
    skeleton[2, 1, 2] = True

    report = diagnose_skeleton_mask_consistency(skeleton, mask)
    text = format_skeleton_mask_consistency_report(report)

    assert str(report["explained_voxel_count"]) in text
    assert str(report["mask_voxel_count"]) in text
    assert f"{report['coverage_fraction']:.1%}" in text


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


def test_the_local_radius_is_applied_per_voxel_not_as_one_scalar_for_the_whole_mask():
    """Regression test that the tolerance genuinely comes from
    ``inscribed_radius_map``'s per-voxel map, not some single scalar (e.g.
    its max or mean) applied everywhere -- a mistake that would not show up
    on any of this file's other fixtures, since every one of them has a
    single, constant local radius throughout.

    One mask holds two unconnected bars of different width: a 6-voxel-wide
    bar (local radius 1-3, peaking at its centre) and a 40-voxel-wide bar
    (local radius up to 20) elsewhere in the same array. The skeleton runs
    along the narrow bar's own near wall only -- nowhere near the wide bar
    at all. Hand-verified per voxel (see the distances/tolerances in the
    comments): only the narrow bar's nearest 4 of 6 rows are explained.

    A version that used the mask-wide *maximum* local radius (20, from the
    wide bar) would use a hugely more generous tolerance for the narrow
    bar too, and would incorrectly explain all 6 of its rows -- this test
    fails under that mistake and passes only with genuine per-voxel radii.
    """
    mask = np.zeros((2, 70, 1), dtype=bool)
    mask[:, 10:16, :] = True  # narrow bar: y=10..15 (6 wide), local radius 1..3
    mask[:, 20:60, :] = True  # wide bar: y=20..59 (40 wide), local radius up to 20

    skeleton = np.zeros((2, 70, 1), dtype=bool)
    skeleton[:, 10, :] = True  # narrow bar's near wall only

    report = diagnose_skeleton_mask_consistency(skeleton, mask)

    # y=10..13: distance from the skeleton (0,1,2,3) <= local radius (1,2,3,3)
    # + margin (sqrt(3)=1.73) -- explained. y=14,15: distance (4,5) exceeds
    # radius (2,1) + margin -- not explained. Both z-slices: 4 * 2 = 8.
    assert report["mask_voxel_count"] == 2 * (6 + 40)
    assert report["explained_voxel_count"] == 8
    assert report["coverage_fraction"] == pytest.approx(8 / 92)


def test_anisotropic_voxel_size_is_accounted_for_in_skeleton_mask_consistency():
    """Neither mask-based consistency check was exercised with a non-cubic
    voxel size anywhere in this file -- only diagnose_skeleton_graph_
    consistency's voxel-index conversion was. A well-centred skeleton must
    still fully explain the mask once physical (not voxel-index) distances
    are computed with genuinely anisotropic spacing."""
    voxel_size_zyx = (2.0, 0.5, 0.25)  # coarse z, fine x -- same convention
    # as test_graph_assemble.py's VOXEL_SIZE_ZYX, so a z/x mixup would show up.
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[3:7, 4, :] = True

    report = diagnose_skeleton_mask_consistency(skeleton, mask, voxel_size_zyx=voxel_size_zyx)

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


def test_the_graph_consistency_report_formatter_reads_the_real_functions_own_keys():
    """See test_the_mask_consistency_report_formatter_reads_the_real_functions_
    own_keys above for why this runs the real diagnose function's output
    through its formatter instead of a hand-typed dict."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:8, 5, 5] = True

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 5.0, 5.0]))
    G.add_node(1, pos=np.array([4.0, 5.0, 5.0]))
    G.add_edge(0, 1, length=2.0, voxels=[[float(z), 5.0, 5.0] for z in range(2, 4)])

    report = diagnose_skeleton_graph_consistency(G, skeleton)
    text = format_skeleton_graph_consistency_report(report)

    assert str(report["matched_voxel_count"]) in text
    assert str(report["skeleton_voxel_count"]) in text
    assert f"{report['coverage_fraction']:.1%}" in text


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


def test_a_graph_mask_edge_with_no_voxels_is_skipped_not_a_crash():
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


def test_anisotropic_voxel_size_is_accounted_for_in_graph_mask_consistency():
    """The graph-mask counterpart of test_anisotropic_voxel_size_is_
    accounted_for_in_skeleton_mask_consistency above: an edge's ``voxels``
    are physical microns, so its coordinates here are voxel-index * spacing
    (matching test_voxel_size_converts_physical_microns_back_to_voxel_
    indices's convention) -- a well-centred edge must still fully explain
    the mask once that's divided back out correctly per anisotropic axis."""
    voxel_size_zyx = (2.0, 0.5, 0.25)
    sz, sy, sx = voxel_size_zyx
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[4:6, 4:6, :] = True  # a rod, 2x2 voxels in (z, y), full length in x

    voxels = [[4.0 * sz, 4.0 * sy, float(x) * sx] for x in range(10)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=9.0 * sx, voxels=voxels)

    report = diagnose_graph_mask_consistency(G, mask, voxel_size_zyx=voxel_size_zyx)

    assert report["coverage_fraction"] == pytest.approx(1.0)


def test_the_graph_mask_consistency_report_format_is_a_readable_one_liner():
    report = {"explained_voxel_count": 6, "mask_voxel_count": 8, "coverage_fraction": 0.75}
    text = format_graph_mask_consistency_report(report)
    assert "6" in text and "8" in text and "75.0%" in text


def test_the_graph_mask_consistency_report_formatter_reads_the_real_functions_own_keys():
    """See test_the_mask_consistency_report_formatter_reads_the_real_functions_
    own_keys above for why this runs the real diagnose function's output
    through its formatter instead of a hand-typed dict."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True
    voxels = [[float(z), 4.0, float(x)] for z in range(3, 7) for x in range(10)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=9.0, voxels=voxels)

    report = diagnose_graph_mask_consistency(G, mask)
    text = format_graph_mask_consistency_report(report)

    assert str(report["explained_voxel_count"]) in text
    assert str(report["mask_voxel_count"]) in text
    assert f"{report['coverage_fraction']:.1%}" in text


# --- all three together: a compounding-loss scenario ------------------------


def test_healthy_individual_checks_can_still_compound_into_a_poor_graph_mask_reading():
    """The whole reason diagnose_graph_mask_consistency exists, per its own
    docstring: a loss that compounds across the earlier two steps can leave
    individually reasonable skeleton/mask and skeleton/graph numbers, while
    the graph still runs through only a fraction of the segmented image.

    Construction: a long, thin segment (many skeleton voxels, small local
    radius each) plus a large, thick blob elsewhere (few skeleton voxels
    needed for a good local-radius-based mask reading, but a big fraction
    of the mask's total *volume*). The skeleton covers both well. The graph
    traces the thin segment exactly but drops the thick blob's core
    entirely -- realistic, since a thick region's centreline is exactly
    the kind of low-voxel-count structure a topology-repair pass could
    prune or fail to reconnect without leaving much of a dent in a
    voxel-count-based check.

    Losing those few voxels barely moves skeleton/graph (they are a small
    fraction of the *total skeleton voxel count*) and skeleton/mask is
    untouched entirely (it never looks at the graph). But graph/mask drops
    sharply, because those few voxels were explaining a much larger
    fraction of the mask's *volume* (the thick blob's own local radius is
    large). Both individual numbers stay at or above the pipeline's own
    default warn-below threshold (0.7); graph/mask alone falls well short.
    """
    shape = (60, 24, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[0:50, 4:6, :2] = True    # thin segment: 50 long, 2 wide -- local radius 1
    mask[50:60, 0:24, :] = True   # thick blob: 10 x 24 x 5 -- local radius up to ~10

    skeleton = np.zeros(shape, dtype=bool)
    skeleton[0:50, 4, 0] = True        # thin segment's own corner column: 50 voxels
    skeleton[50:60, 11:13, 2] = True   # thick blob's centred core: 20 voxels

    skeleton_mask = diagnose_skeleton_mask_consistency(skeleton, mask)

    # The graph traces the thin segment exactly and stops there -- it never
    # reaches the thick blob's core at all.
    voxels = [[float(z), 4.0, 0.0] for z in range(50)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=49.0, voxels=voxels)

    skeleton_graph = diagnose_skeleton_graph_consistency(G, skeleton)
    graph_mask = diagnose_graph_mask_consistency(G, mask)

    default_warn_below = 0.7
    assert skeleton_mask["coverage_fraction"] >= default_warn_below, (
        "fixture bug: skeleton/mask should look individually healthy"
    )
    assert skeleton_graph["coverage_fraction"] >= default_warn_below, (
        "fixture bug: skeleton/graph should look individually healthy"
    )
    assert graph_mask["coverage_fraction"] < 0.5, (
        "graph/mask should read far worse than either check taken alone"
    )
    # Exact values, verified against the functions themselves (a mixed-
    # thickness fixture like this one is not hand-countable by inspection).
    assert skeleton_mask["coverage_fraction"] == pytest.approx(1088 / 1400)
    assert skeleton_graph["coverage_fraction"] == pytest.approx(50 / 70)
    assert graph_mask["coverage_fraction"] == pytest.approx(555 / 1400)
