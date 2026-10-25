"""How well a skeleton represents its mask, a graph represents its skeleton,
a graph represents the original mask directly, and -- the inverse question
-- whether any genuine vessel is missing from the skeleton or graph
entirely: hand-verified geometry, no pipeline involved.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph.diagnostics import (
    diagnose_graph_mask_consistency,
    diagnose_skeleton_graph_consistency,
    diagnose_vessels_missing_from_graph,
    format_graph_mask_consistency_report,
    format_skeleton_graph_consistency_report,
    format_vessels_missing_from_graph_report,
)
from haemolynx.preprocessing.skeleton_consistency import (
    diagnose_centreline_dropped_by_thick_gate,
    diagnose_skeleton_mask_consistency,
    diagnose_vessels_missing_from_skeleton,
    format_centreline_dropped_by_thick_gate_report,
    format_skeleton_mask_consistency_report,
    format_vessels_missing_from_skeleton_report,
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


def test_a_supplied_local_radius_map_is_used_instead_of_being_recomputed():
    """A caller-supplied `local_radius_map` must actually be used -- not
    silently ignored in favour of recomputing it from `mask` internally --
    since the settings optimiser relies on exactly this to avoid a fresh
    full-volume EDT on every single candidate it evaluates (see
    `optimisation.search._Search._raw_mask_local_radius_map`).
    """
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 4:6, :] = True
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[0, 0, 0] = True  # far from the mask -- explains none of it by default

    # A radius map large enough that "no farther than the local radius" is
    # true everywhere flips a normally-unexplained mask to fully explained
    # -- if this were ignored and recomputed internally instead, the result
    # would be unchanged from the no-argument case above (coverage ~ 0).
    huge_radius_map = np.full(mask.shape, 1000.0)
    report = diagnose_skeleton_mask_consistency(
        skeleton, mask, voxel_size_zyx=(1.0, 1.0, 1.0), local_radius_map=huge_radius_map,
    )
    assert report["coverage_fraction"] == pytest.approx(1.0)


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


# --- diagnose_vessels_missing_from_skeleton (the inverse question) ---------


def _two_vessels_and_a_noise_speck():
    """Two well-separated 2-voxel-wide bars (16 voxels each) plus one
    isolated single-voxel speck -- 3 connected components, only 2 of which
    are big enough to count as a genuine vessel at the default
    min_vessel_voxels=2."""
    shape = (2, 20, 4)
    mask = np.zeros(shape, dtype=bool)
    mask[:, 2:4, :] = True   # vessel A: 16 voxels
    mask[:, 10:12, :] = True  # vessel B: 16 voxels
    mask[0, 16, 0] = True     # noise speck: 1 voxel
    return mask


def test_vessels_missing_from_skeleton_is_trivial_for_an_empty_mask():
    empty = np.zeros((5, 5, 5), dtype=bool)
    report = diagnose_vessels_missing_from_skeleton(empty, empty)
    assert report == {
        "vessel_count": 0,
        "missing_vessel_count": 0,
        "missing_vessel_voxel_counts": [],
        "explained_vessel_fraction": 1.0,
        "mask_voxel_count": 0,
        "missing_voxel_count": 0,
        "missing_volume_fraction": 0.0,
    }


def test_a_vessel_with_no_skeleton_at_all_is_flagged_while_a_covered_one_is_not():
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True  # fully explains vessel A only; nothing near B

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)

    # The noise speck never counts as a vessel at the default floor.
    assert report["vessel_count"] == 2
    assert report["missing_vessel_count"] == 1
    assert report["missing_vessel_voxel_counts"] == [16]
    assert report["explained_vessel_fraction"] == pytest.approx(0.5)


def test_an_empty_skeleton_flags_every_genuine_vessel_as_missing():
    mask = _two_vessels_and_a_noise_speck()
    report = diagnose_vessels_missing_from_skeleton(np.zeros(mask.shape, dtype=bool), mask)
    assert report["vessel_count"] == 2
    assert report["missing_vessel_count"] == 2
    assert sorted(report["missing_vessel_voxel_counts"]) == [16, 16]
    assert report["explained_vessel_fraction"] == pytest.approx(0.0)


def test_vessels_missing_from_skeleton_uses_a_supplied_local_radius_map():
    """Same contract as `diagnose_skeleton_mask_consistency`'s own
    `local_radius_map` parameter -- a caller-supplied map must actually be
    used, not silently recomputed, since the settings optimiser relies on
    this to avoid a fresh full-volume EDT on every candidate it tries."""
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True  # fully explains vessel A only; nothing near B by default

    default_report = diagnose_vessels_missing_from_skeleton(skeleton, mask)
    assert default_report["missing_vessel_count"] == 1

    huge_radius_map = np.full(mask.shape, 1000.0)
    report = diagnose_vessels_missing_from_skeleton(
        skeleton, mask, local_radius_map=huge_radius_map,
    )
    assert report["missing_vessel_count"] == 0, (
        "a supplied local_radius_map large enough to explain any distance "
        "was not used -- diagnose_vessels_missing_from_skeleton must be "
        "reading the parameter, not silently recomputing its own radius map"
    )


def test_min_vessel_voxels_controls_the_noise_size_floor():
    """The same fixture's noise speck is excluded at the default floor, but
    becomes a (missing) vessel of its own once the floor drops to 1."""
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True

    default_floor = diagnose_vessels_missing_from_skeleton(skeleton, mask)
    assert default_floor["vessel_count"] == 2

    no_floor = diagnose_vessels_missing_from_skeleton(
        skeleton, mask, min_vessel_voxels=1
    )
    assert no_floor["vessel_count"] == 3
    assert no_floor["missing_vessel_count"] == 2
    assert sorted(no_floor["missing_vessel_voxel_counts"]) == [1, 16]


def test_missing_vessels_from_skeleton_uses_canonical_binarisation():
    """Same Phase-9 canonical-binarisation fix as diagnose_skeleton_mask_
    consistency: a 1/2-labelled mask must not be read as bare != 0."""
    mask = np.full((6, 6, 6), 1, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 2  # minority label: the true foreground, 8 voxels
    skeleton = np.zeros((6, 6, 6), dtype=bool)
    skeleton[2:4, 2:4, 2:4] = True

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)

    assert report["vessel_count"] == 1  # not treating all 208 background voxels as vessels
    assert report["missing_vessel_count"] == 0


def test_vessels_missing_from_skeleton_report_formatter_reads_the_real_functions_own_keys():
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)
    text = format_vessels_missing_from_skeleton_report(report)

    assert str(report["missing_vessel_count"]) in text
    assert str(report["vessel_count"]) in text
    assert f"{report['explained_vessel_fraction']:.1%}" in text


def test_missing_vessels_report_their_share_of_the_whole_mask_volume():
    """The denominator is every mask voxel, the speck below the noise floor
    included: 16 missing of 16 + 16 + 1."""
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True  # explains vessel A only

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)

    assert report["mask_voxel_count"] == 33
    assert report["missing_voxel_count"] == 16
    assert report["missing_volume_fraction"] == pytest.approx(16 / 33)


def test_one_large_vessel_among_many_missing_specks_is_mostly_missing_by_count_but_not_by_volume():
    """The E14.5 case in miniature (832 of 1,205 components missing, 0.36%
    of the mask): the vasculature is one large component and the skeleton
    leaves out a hundred 2-voxel specks. By count almost every "vessel" is
    missing; by volume very little is, and the report must say both."""
    mask = np.zeros((10, 40, 40), dtype=bool)
    mask[2:8, 5:35, 5:35] = True  # 5,400 voxels, the vasculature
    for y in range(0, 40, 4):
        for x in range(0, 40, 4):
            mask[0, y, x:x + 2] = True  # 100 separate 2-voxel specks, 200 voxels
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[5, 20, 5:35] = True  # runs through the large component only

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)
    text = format_vessels_missing_from_skeleton_report(report)

    assert report["vessel_count"] == 101
    assert report["missing_vessel_count"] == 100
    assert report["missing_volume_fraction"] == pytest.approx(200 / 5600)
    assert "100 of 101" in text
    assert f"{200 / 5600:.2%} of the segmented volume" in text
    assert "the largest is 2 voxels" in text


def test_the_missing_vessels_report_names_the_volume_share_and_the_largest_piece():
    mask = _two_vessels_and_a_noise_speck()
    report = diagnose_vessels_missing_from_skeleton(np.zeros(mask.shape, dtype=bool), mask)

    text = format_vessels_missing_from_skeleton_report(report)

    assert f"{32 / 33:.2%} of the segmented volume" in text
    assert "the largest is 16 voxels" in text


def test_the_volume_share_is_left_out_when_no_vessel_is_missing():
    mask = _two_vessels_and_a_noise_speck()
    skeleton = np.zeros(mask.shape, dtype=bool)
    skeleton[:, 2, :] = True
    skeleton[:, 10, :] = True  # explains both vessels

    report = diagnose_vessels_missing_from_skeleton(skeleton, mask)
    text = format_vessels_missing_from_skeleton_report(report)

    assert report["missing_volume_fraction"] == 0.0
    assert text.endswith("(100.0% represented).")
    assert "segmented volume" not in text


def test_a_tiny_missing_share_is_not_printed_as_zero():
    """A share under 0.01% would round to "0.00%", which reads as nothing
    missing while the count says otherwise."""
    report = {
        "vessel_count": 2,
        "missing_vessel_count": 1,
        "missing_vessel_voxel_counts": [1],
        "explained_vessel_fraction": 0.5,
        "missing_volume_fraction": 1e-6,
    }

    text = format_vessels_missing_from_skeleton_report(report)

    assert "<0.01% of the segmented volume" in text
    assert "0.00%" not in text
    assert "the largest is 1 voxel." in text


# --- diagnose_vessels_missing_from_graph (the inverse question) ------------


def test_vessels_missing_from_graph_is_trivial_for_an_empty_mask():
    empty = np.zeros((5, 5, 5), dtype=bool)
    report = diagnose_vessels_missing_from_graph(nx.MultiGraph(), empty)
    assert report == {
        "vessel_count": 0,
        "missing_vessel_count": 0,
        "missing_vessel_voxel_counts": [],
        "explained_vessel_fraction": 1.0,
        "mask_voxel_count": 0,
        "missing_voxel_count": 0,
        "missing_volume_fraction": 0.0,
    }


def test_a_vessel_with_no_graph_edge_at_all_is_flagged_while_a_covered_one_is_not():
    mask = _two_vessels_and_a_noise_speck()
    voxels = [[float(z), 2.0, float(x)] for z in range(2) for x in range(4)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=1.0, voxels=voxels)  # traces vessel A only

    report = diagnose_vessels_missing_from_graph(G, mask)

    assert report["vessel_count"] == 2
    assert report["missing_vessel_count"] == 1
    assert report["missing_vessel_voxel_counts"] == [16]
    assert report["explained_vessel_fraction"] == pytest.approx(0.5)


def test_an_empty_graph_flags_every_genuine_vessel_as_missing():
    mask = _two_vessels_and_a_noise_speck()
    report = diagnose_vessels_missing_from_graph(nx.MultiGraph(), mask)
    assert report["vessel_count"] == 2
    assert report["missing_vessel_count"] == 2
    assert report["explained_vessel_fraction"] == pytest.approx(0.0)


def test_missing_vessels_from_graph_uses_canonical_binarisation():
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

    report = diagnose_vessels_missing_from_graph(G, mask)

    assert report["vessel_count"] == 1
    assert report["missing_vessel_count"] == 0


def test_vessels_missing_from_graph_report_formatter_reads_the_real_functions_own_keys():
    mask = _two_vessels_and_a_noise_speck()
    voxels = [[float(z), 2.0, float(x)] for z in range(2) for x in range(4)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=1.0, voxels=voxels)

    report = diagnose_vessels_missing_from_graph(G, mask)
    text = format_vessels_missing_from_graph_report(report)

    assert str(report["missing_vessel_count"]) in text
    assert str(report["vessel_count"]) in text
    assert f"{report['explained_vessel_fraction']:.1%}" in text


def test_vessels_missing_from_graph_report_their_volume_share_too():
    mask = _two_vessels_and_a_noise_speck()
    voxels = [[float(z), 2.0, float(x)] for z in range(2) for x in range(4)]
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, length=1.0, voxels=voxels)  # traces vessel A only

    report = diagnose_vessels_missing_from_graph(G, mask)
    text = format_vessels_missing_from_graph_report(report)

    assert report["missing_volume_fraction"] == pytest.approx(16 / 33)
    assert f"{16 / 33:.2%} of the segmented volume" in text
    assert "the largest is 16 voxels" in text


# --- diagnose_centreline_dropped_by_thick_gate --------------------------------


def _line_along_x(shape, y: int, x0: int, x1: int, z: int = 5) -> np.ndarray:
    line = np.zeros(shape, dtype=bool)
    line[z, y, x0:x1] = True
    return line


def test_dropped_centreline_is_split_by_place_round_the_thick_region():
    shape = (11, 11, 50)
    region = np.zeros(shape, dtype=bool)
    region[:, :, :20] = True
    plain = _line_along_x(shape, 5, 10, 40)  # x 10-19 inside, 20-22 within 3 um, 23-39 beyond
    gated = np.zeros(shape, dtype=bool)

    report = diagnose_centreline_dropped_by_thick_gate(
        gated, plain, region, plain_skeleton=plain, shell_um=3.0,
    )

    assert report["dropped_pieces"] == 1
    assert report["dropped_voxels"] == 30
    assert report["dropped_um"] == pytest.approx(30.0)
    assert report["by_place"] == {
        "thick_region": {"voxels": 10, "um": pytest.approx(10.0)},
        "shell": {"voxels": 3, "um": pytest.approx(3.0)},
        "beyond": {"voxels": 17, "um": pytest.approx(17.0)},
    }
    assert report["inside_large_um"] is None
    (piece,) = report["pieces"]
    assert piece["thick_region_um"] == pytest.approx(10.0)
    assert piece["beyond_um"] == pytest.approx(17.0)


def test_plain_lee_within_reach_of_the_skeleton_is_not_dropped():
    shape = (11, 11, 40)
    region = np.zeros(shape, dtype=bool)
    plain = _line_along_x(shape, 2, 5, 35)
    near = _line_along_x(shape, 5, 5, 35)  # 3 um away: within reach
    far = _line_along_x(shape, 6, 5, 35)  # 4 um away: dropped

    assert diagnose_centreline_dropped_by_thick_gate(near, plain, region, plain_skeleton=plain)["dropped_voxels"] == 0
    assert diagnose_centreline_dropped_by_thick_gate(far, plain, region, plain_skeleton=plain)["dropped_voxels"] == 30


def test_pieces_under_the_size_floor_are_left_out():
    shape = (11, 11, 40)
    region = np.zeros(shape, dtype=bool)
    gated = np.zeros(shape, dtype=bool)
    plain = _line_along_x(shape, 5, 2, 11) | _line_along_x(shape, 5, 20, 30)  # 9 and 10 voxels

    report = diagnose_centreline_dropped_by_thick_gate(gated, plain, region, plain_skeleton=plain, min_piece_voxels=10)

    assert report["dropped_pieces"] == 1
    assert report["dropped_voxels"] == 10


def test_dropped_centreline_is_measured_in_microns_on_anisotropic_voxels():
    shape = (11, 11, 40)
    region = np.zeros(shape, dtype=bool)
    gated = _line_along_x(shape, 3, 5, 35)  # 2 voxels away along y: 1.0 um at 0.5 um a voxel
    plain = _line_along_x(shape, 5, 5, 35)
    spacing = (1.0, 0.5, 0.25)

    near = diagnose_centreline_dropped_by_thick_gate(gated, plain, region, plain_skeleton=plain, voxel_size_zyx=spacing)
    alone = diagnose_centreline_dropped_by_thick_gate(
        np.zeros(shape, dtype=bool), plain, region, plain_skeleton=plain, voxel_size_zyx=spacing,
    )

    assert near["dropped_voxels"] == 0
    assert alone["dropped_um"] == pytest.approx(30 * 0.25)


def test_dropped_centreline_inside_the_large_masks_is_reported_apart():
    shape = (11, 11, 40)
    region = np.zeros(shape, dtype=bool)
    large = np.zeros(shape, dtype=bool)
    large[:, :, :15] = True
    plain = _line_along_x(shape, 5, 5, 35)

    report = diagnose_centreline_dropped_by_thick_gate(
        np.zeros(shape, dtype=bool), plain, region, plain_skeleton=plain, large_mask=large,
    )

    assert report["inside_large_um"] == pytest.approx(10.0)
    assert report["pieces"][0]["inside_large_um"] == pytest.approx(10.0)


def test_plain_lee_defaults_to_lee_of_the_hole_filled_mask():
    from haemolynx.preprocessing.skeleton import fill_binary_holes, skeletonize_volume

    mask = np.zeros((11, 11, 40), dtype=bool)
    mask[3:8, 3:8, 5:35] = True  # an odd-width rod: Lee leaves a line
    empty = np.zeros_like(mask)

    report = diagnose_centreline_dropped_by_thick_gate(empty, mask, empty)
    expected = skeletonize_volume(fill_binary_holes(mask))

    assert expected.sum() > 10
    assert report["plain_voxels"] == int(expected.sum())
    assert report["dropped_voxels"] == int(expected.sum())


def _thick_vessel_with_capillary(capillary_y: int):
    """A thick vessel (radius 7 voxels, its own large-vessel mask) along x,
    and a capillary of radius 1 voxel beside it, its axis at row
    *capillary_y*, joined to it at both ends."""
    shape = (32, 48, 90)
    z, y, x = np.indices(shape)
    thick = ((z - 16) ** 2 + (y - 16) ** 2 <= 49) & (x >= 5) & (x < 85)
    capillary = ((z - 16) ** 2 + (y - capillary_y) ** 2 <= 1) & (x >= 15) & (x < 75)
    mask = thick | capillary
    for x0 in (15, 72):
        mask |= (abs(z - 16) <= 1) & (y >= 16) & (y <= capillary_y) & (x >= x0) & (x < x0 + 3)
    return mask, thick


def _gated(mask, large):
    from haemolynx.preprocessing.thick_vessels import skeletonize_thickness_gated

    # E14.5's settings: the large masks are the thick body, 3 um of wall, 3 um of flake filter.
    return skeletonize_thickness_gated(
        mask, min_radius_um=3.0, voxel_size_zyx=(1.0, 1.0, 1.0), restrict_thick_to_mask=large,
        return_thick_mask=True, wall_absorption_um=3.0, flake_filter_um=3.0,
    )


def test_a_capillary_4_um_beside_a_thick_vessel_is_reported_dropped_by_the_gate():
    mask, large = _thick_vessel_with_capillary(capillary_y=27)  # its axis 4 um beyond the wall (row 23)
    skeleton, region = _gated(mask, large)
    assert not skeleton[:, 26:29, 20:70].any()  # the gate drew no centreline along it

    report = diagnose_centreline_dropped_by_thick_gate(skeleton, mask, region, large_mask=large)

    assert report["dropped_pieces"] == 1
    assert 60.0 <= report["dropped_um"] <= 75.0
    places = report["by_place"]
    assert places["shell"]["um"] > places["thick_region"]["um"] > 0
    assert places["beyond"]["um"] == 0
    assert report["inside_large_um"] < 0.2 * report["dropped_um"]


def test_a_capillary_beyond_the_shell_keeps_its_centreline_and_nothing_is_reported():
    mask, large = _thick_vessel_with_capillary(capillary_y=31)
    skeleton, region = _gated(mask, large)
    assert skeleton[:, 30:33, 20:70].sum() >= 45

    report = diagnose_centreline_dropped_by_thick_gate(skeleton, mask, region, large_mask=large)

    assert report["dropped_voxels"] == 0


def test_the_thick_vessel_alone_drops_no_centreline():
    _mask, large = _thick_vessel_with_capillary(capillary_y=31)
    skeleton, region = _gated(large, large)

    report = diagnose_centreline_dropped_by_thick_gate(skeleton, large, region, large_mask=large)

    assert report["dropped_voxels"] == 0


def test_the_dropped_centreline_report_formatter_reads_the_real_functions_own_keys():
    shape = (11, 11, 50)
    region = np.zeros(shape, dtype=bool)
    region[:, :, :20] = True
    plain = _line_along_x(shape, 5, 10, 40)
    report = diagnose_centreline_dropped_by_thick_gate(
        np.zeros(shape, dtype=bool), plain, region, plain_skeleton=plain, large_mask=region,
    )

    text = format_centreline_dropped_by_thick_gate_report(report, shell_um=3.0)

    assert text == (
        "Centreline the thick gate drops: 30 um of plain-Lee centreline in 1 piece(s) has no skeleton "
        "near it -- 10 um inside the thick region, 3 um within 3 um of it, 17 um beyond; "
        "10 um inside the large-vessel masks."
    )
