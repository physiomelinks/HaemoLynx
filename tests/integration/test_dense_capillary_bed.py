"""Skeleton cleaning and graph building on a dense capillary bed, end to end.

Every bridge the build draws stays in the segmentation, no segmented vessel
is drawn twice or left out, and the only free ends are the bed's own blind
sprouts and the vessels the image cuts -- see
``tests/dense_capillary_fixtures.py`` for what the bed holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

pytest.importorskip("skan")

from haemolynx.graph import (
    build_graph_from_skeleton,
    diagnose_graph_against_mask,
    diagnose_lumen_artefacts,
    diagnose_parallel_duplicates_in_lumen,
    lumen_artefact_counts,
)
from haemolynx.graph._helpers import EdgeSampleIndex
from haemolynx.graph.assemble import mask_radius_sampler
from haemolynx.graph.mask_recovery import recover_uncovered_mask_vessels, uncovered_mask_voxels
from haemolynx.preprocessing import MaskSupport, preprocess_skeleton_for_graph
from haemolynx.preprocessing.skeleton_consistency import diagnose_skeleton_mask_consistency

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from dense_capillary_fixtures import centreline_samples, dense_capillary_bed  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: How far a free end may sit from what explains it: a vessel's own radius
#: plus the voxel or two Lee thinning stops short of a blind end.
END_TOLERANCE_UM = 4.0


@pytest.fixture(scope="module")
def built():
    bed = dense_capillary_bed()
    voxel = bed.voxel_size_zyx
    skeleton = preprocess_skeleton_for_graph(
        bed.skeleton, min_branch_length=3, max_bridge_distance=4, closing_radius=1,
        bridge_gap_size=3, voxel_size_zyx=voxel, segmentation_mask=bed.mask,
    )
    bridges = []
    before_recovery = {}

    def record_bridges(G, label):
        bridges.extend(
            float(data["bridge_background_um"])
            for *_, data in G.edges(data=True) if "bridge_background_um" in data
        )
        if label == "reconnect_orphan_and_dangling_nodes":
            before_recovery["graph"] = G.copy()

    G = build_graph_from_skeleton(
        skeleton, voxel_size=voxel, graph_reconnect_threshold=10.0,
        final_orphan_reconnect_threshold=3.0, cluster_collapse_distance=5.0,
        min_stub_length=10.0, min_stub_length_radius_multiple=1.5,
        stub_radius_at=mask_radius_sampler(bed.mask, voxel, 1.0),
        segmentation_mask=bed.mask, step_callback=record_bridges,
    )
    return bed, G, bridges, before_recovery["graph"]


@pytest.fixture(scope="module")
def recovered_from_the_mask(built):
    """Recovery without the skeleton -- the mask Lee-thinned again -- on the
    build's graph as it stood before recovery: the one way the capillary
    whose skeleton was deleted can come back."""
    bed, _G, _bridges, before = built
    return recover_uncovered_mask_vessels(before.copy(), MaskSupport(bed.mask, bed.voxel_size_zyx))


def test_every_bridge_crosses_at_most_two_microns_of_background(built, recovered_from_the_mask):
    _bed, _G, bridges, _before = built
    bridges = bridges + [
        float(data["bridge_background_um"])
        for *_, data in recovered_from_the_mask.edges(data=True) if "bridge_background_um" in data
    ]
    assert bridges
    assert max(bridges) <= 2.0 + 1e-9


def test_no_vessel_is_drawn_twice_in_one_lumen(built):
    bed, G, _bridges, _before = built
    report = diagnose_parallel_duplicates_in_lumen(G, bed.mask, voxel_size_zyx=bed.voxel_size_zyx)
    assert report["duplicate_pairs"] == []


def test_the_lumen_artefact_report_finds_nothing_that_is_never_a_vessel(built):
    """The report ``build_network`` logs and keeps (``lumen_artefact_counts``),
    held as the Nerve fixture's is (``test_pipeline_invariants.py``): no loop
    inside one lumen, no dead end whose tip is in another vessel's lumen,
    beside another centreline or mostly off the mask, and no pair in one
    lumen -- nothing its warning would name."""
    bed, G, _bridges, _before = built
    report = diagnose_lumen_artefacts(G, bed.mask, voxel_size_zyx=bed.voxel_size_zyx)
    counts = lumen_artefact_counts(report)

    assert counts["found"] is False, counts
    assert counts["pairs_in_one_lumen"] == 0
    assert counts["loops_inside_one_lumen"] == 0
    assert counts["tips_in_another_lumen"] == 0
    assert counts["dead_ends_beside_another_centreline"] == 0
    assert counts["dead_ends_off_the_mask"] == 0


def test_nothing_segmented_is_left_out_of_the_graph(built):
    """Recovery's own measure -- a mask voxel within the local lumen of a
    centreline -- reaches 95%. The pipeline's coverage check counts a voxel
    only within the inscribed radius at its nearest centreline point, which
    a vessel's rounded ends at the image faces never are: there the graph
    must explain as much as the mask's own Lee skeleton does, near enough."""
    bed, G, _bridges, _before = built
    consistency, missing = diagnose_graph_against_mask(
        G, bed.mask, voxel_size_zyx=bed.voxel_size_zyx
    )
    skeleton_report = diagnose_skeleton_mask_consistency(
        bed.skeleton, bed.mask, voxel_size_zyx=bed.voxel_size_zyx
    )
    uncovered = uncovered_mask_voxels(G, MaskSupport(bed.mask, bed.voxel_size_zyx))

    assert missing["missing_vessel_count"] == 0
    assert 1.0 - len(uncovered) / bed.mask.sum() >= 0.95
    assert consistency["coverage_fraction"] >= 0.95 * skeleton_report["coverage_fraction"]


def test_the_only_free_ends_are_the_sprouts_and_the_image_faces(built):
    bed, G, _bridges, _before = built
    ends = np.array([G.nodes[n]["pos"] for n in G.nodes if G.degree[n] == 1], dtype=float)
    to_a_face = np.minimum(ends, bed.extent_um - ends)[:, 1:].min(axis=1)
    to_a_sprout = cKDTree(bed.sprout_tips_um).query(ends)[0]

    unexplained = ends[(to_a_face > END_TOLERANCE_UM) & (to_a_sprout > END_TOLERANCE_UM)]
    assert unexplained.tolist() == []
    assert np.all(cKDTree(ends).query(bed.sprout_tips_um)[0] <= END_TOLERANCE_UM)


@pytest.mark.parametrize("vessel", ["first of the pair", "second of the pair"])
def test_neighbours_across_background_are_each_traced(built, vessel):
    bed, G, _bridges, _before = built
    start, end = {
        "first of the pair": bed.parallel_pair_um[0],
        "second of the pair": bed.parallel_pair_um[1],
    }[vessel]
    samples = centreline_samples(start, end)
    distance, _ = EdgeSampleIndex(G, step_um=0.5).tree.query(samples)

    assert np.mean(distance <= 1.5) >= 0.9


def _traced_fraction(G, start, end) -> float:
    distance, _ = EdgeSampleIndex(G, step_um=0.5).tree.query(centreline_samples(start, end))
    return float(np.mean(distance <= 1.5))


def test_a_capillary_the_skeleton_lacks_is_left_to_the_mask_recovery_without_it(built, recovered_from_the_mask):
    """The build recovers along its skeleton, so a capillary the skeleton
    lacks stays out of the graph (the Segmented vessels without a
    centreline review list shows it); recovery thinning the mask again traces
    it end to end."""
    bed, G, _bridges, _before = built

    assert _traced_fraction(G, *bed.lost_capillary_um) < 0.1
    assert _traced_fraction(recovered_from_the_mask, *bed.lost_capillary_um) >= 0.9
