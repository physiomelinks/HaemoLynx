"""The lumen-artefact report: what graph building leaves in a segmented vessel
that is not a vessel, measured against the mask.

``graph.diagnose_lumen_artefacts`` is read-only; these tests pin what it
reports on one synthetic case of each kind (``tests/lumen_artefact_fixtures.py``),
and that the per-sample "beside another vessel in the same lumen" test it
counts with agrees with the per-path guard every reconnect uses.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from haemolynx.graph import (
    diagnose_lumen_artefacts,
    format_lumen_artefacts_report,
    lumen_artefacts_found,
)
from haemolynx.graph.diagnostics import DEAD_END_KINDS, classify_dead_ends
from haemolynx.graph.lumen_loops import iter_short_loops
from haemolynx.preprocessing import MaskSupport, path_shadows_existing_vessel, shadowed_samples

from lumen_artefact_fixtures import (
    VOXEL_SIZE,
    dead_end_cases,
    fork_in_one_lumen,
    loop_round_tissue,
    polyline_graph,
    ring_in_one_lumen,
    two_strands_in_one_lumen,
    wide_vessel,
)


def _report(mask, G, **kwargs):
    kwargs.setdefault("image_face_margin_um", 0.0)
    return diagnose_lumen_artefacts(G, mask, voxel_size_zyx=VOXEL_SIZE, **kwargs)


def _line(z, y, x0, x1):
    return np.array([[z, y, x] for x in np.arange(x0, x1 + 0.25, 0.5)], dtype=float)


# --- shadowed_samples: the per-sample form of the guard ---------------------


def test_a_second_strand_is_beside_the_first_along_its_middle_and_not_at_its_ends():
    support = MaskSupport(wide_vessel(), VOXEL_SIZE)
    first = _line(7, 5, 2, 57)
    samples, judged, beside = shadowed_samples(
        _line(7, 9, 2, 57), cKDTree(first), first, support.inside, support.radius
    )
    assert len(samples) == len(judged) == len(beside)
    assert beside[judged].mean() > 0.9
    # Within a lumen radius of either end nothing is judged, so nothing is beside.
    assert not beside[~judged].any()
    assert not judged[0] and not judged[-1]


def test_a_vessel_behind_background_is_never_beside():
    mask = np.zeros((9, 22, 60), dtype=bool)
    mask[1:8, 1:8] = True
    mask[1:8, 11:18] = True
    support = MaskSupport(mask, VOXEL_SIZE)
    first = _line(4, 4, 2, 57)
    _samples, judged, beside = shadowed_samples(
        _line(4, 14, 2, 57), cKDTree(first), first, support.inside, support.radius
    )
    assert judged.any()
    assert not beside.any()


@pytest.mark.parametrize(
    "path",
    [_line(7, 9, 2, 57), _line(7, 9, 2, 30), np.vstack([_line(7, 9, 2, 20), [[7, 14, 40]]])],
)
def test_the_per_path_guard_is_half_the_judged_samples_beside(path):
    """``path_shadows_existing_vessel`` and ``shadowed_samples`` ask one
    question; the guard's verdict is the per-sample answer on half the path."""
    support = MaskSupport(wide_vessel(), VOXEL_SIZE)
    first = _line(7, 5, 2, 57)
    _samples, judged, beside = shadowed_samples(path, cKDTree(first), first, support.inside, support.radius)
    verdict = path_shadows_existing_vessel(path, cKDTree(first), first, support.inside, support.radius)
    assert verdict == bool(judged.any() and 2 * beside[judged].sum() >= judged.sum())


# --- iter_short_loops -------------------------------------------------------


def test_each_short_loop_is_listed_once():
    _mask, G = loop_round_tissue()
    loops = list(iter_short_loops(G))
    assert len(loops) == 1
    cycle, polyline = loops[0]
    assert len(cycle) == 4
    assert polyline.shape[1] == 3


def test_a_tree_has_no_loops():
    G = polyline_graph([[(0, 0, 0), (0, 0, 10)], [(0, 0, 10), (0, 0, 20)], [(0, 0, 10), (0, 10, 10)]])
    assert list(iter_short_loops(G)) == []


# --- diagnose_lumen_artefacts -----------------------------------------------


def test_two_strands_in_one_lumen_are_a_pair_and_their_middles_are_duplicated_centreline():
    mask, G = two_strands_in_one_lumen()
    report = _report(mask, G)
    assert report["duplicate_pair_count"] == 1
    assert report["fork_pair_count"] == 0
    # Each strand is 55 um; all but about a lumen radius at each end runs beside the other.
    assert 70.0 < report["duplicated_centreline_um"] < 110.0
    assert report["duplicated_centreline_fraction"] == pytest.approx(
        report["duplicated_centreline_um"] / report["centreline_um"]
    )
    assert lumen_artefacts_found(report)


def test_two_edges_leaving_one_node_through_one_lumen_are_a_fork():
    mask, G = fork_in_one_lumen()
    report = _report(mask, G)
    assert report["duplicate_pair_count"] == 1
    assert report["fork_pair_count"] == 1


def test_a_ring_round_nothing_is_a_loop_inside_one_lumen():
    mask, G = ring_in_one_lumen()
    report = _report(mask, G)
    assert report["short_loop_count"] == 1
    assert report["loops_inside_one_lumen"] == 1
    assert lumen_artefacts_found(report)


def test_a_loop_round_tissue_is_not_an_artefact():
    mask, G = loop_round_tissue()
    report = _report(mask, G)
    assert report["short_loop_count"] == 1
    assert report["loops_inside_one_lumen"] == 0
    assert report["duplicate_pair_count"] == 0
    assert report["duplicated_centreline_um"] == pytest.approx(0.0, abs=1.0)
    assert not lumen_artefacts_found(report)


def test_each_kind_of_dead_end_is_reported_and_a_real_branch_is_none_of_them():
    mask, G, tips = dead_end_cases()
    report = _report(mask, G)
    dead_ends = report["dead_ends"]
    assert tips["inside_other_lumen"] in dead_ends["inside_other_lumen"]
    assert tips["off_mask"] in dead_ends["off_mask"]
    assert tips["mask_continues"] in dead_ends["mask_continues"]
    assert tips["short"] in dead_ends["short"]
    for kind, nodes in dead_ends.items():
        assert tips["real_branch"] not in nodes, kind
    assert tips["off_mask"] not in dead_ends["mask_continues"]
    assert tips["short"] not in dead_ends["mask_continues"]
    assert report["isolated_single_edges"] == 1
    # The trunk's two ends lie on the image's x faces.
    assert report["dead_ends_at_image_face"] == 2
    assert report["interior_dead_ends"] == 5
    assert report["dead_end_counts"] == {kind: len(nodes) for kind, nodes in dead_ends.items()}


def test_dead_ends_near_an_image_face_are_not_judged():
    mask, G, _tips = dead_end_cases()
    report = diagnose_lumen_artefacts(G, mask, voxel_size_zyx=VOXEL_SIZE, image_face_margin_um=200.0)
    assert report["interior_dead_ends"] == 0
    assert report["dead_ends_at_image_face"] == 7
    assert all(not nodes for nodes in report["dead_ends"].values())


def test_the_short_rule_follows_the_radius_multiple():
    mask, G, tips = dead_end_cases()
    assert tips["short"] in _report(mask, G, stub_radius_multiple=3.0)["dead_ends"]["short"]
    assert tips["short"] not in _report(mask, G, stub_radius_multiple=1.0)["dead_ends"]["short"]


def test_the_report_changes_nothing_and_reads_as_one_line():
    mask, G = fork_in_one_lumen()
    before = (sorted(map(str, G.nodes)), sorted(map(str, G.edges(keys=True))))
    report = _report(mask, G)
    assert (sorted(map(str, G.nodes)), sorted(map(str, G.edges(keys=True)))) == before
    line = format_lumen_artefacts_report(report)
    assert "\n" not in line
    assert line.startswith("Lumen artefacts: 1 pair(s) of edges share one lumen (1 leaving one node side by side)")


def test_an_empty_graph_reports_nothing():
    report = _report(wide_vessel(), nx.MultiGraph())
    assert report["duplicate_pair_count"] == 0
    assert report["short_loop_count"] == 0
    assert report["interior_dead_ends"] == 0
    assert not lumen_artefacts_found(report)


# --- classify_dead_ends: the dead ends alone, tip by tip ---------------------


def test_classifying_dead_ends_alone_agrees_with_the_report_tip_for_tip():
    mask, G, tips = dead_end_cases()
    for margin in (0.0, 10.0):
        report = _report(mask, G, image_face_margin_um=margin)
        judged = classify_dead_ends(G, mask, voxel_size_zyx=VOXEL_SIZE, image_face_margin_um=margin)

        assert judged["dead_ends"] == report["dead_ends"]
        assert len(judged["interior"]) == report["interior_dead_ends"]
        assert len(judged["at_image_face"]) == report["dead_ends_at_image_face"]
        assert judged["isolated_single_edges"] == report["isolated_single_edges"]
        for tip, kinds in judged["kinds"].items():
            assert kinds == tuple(k for k in DEAD_END_KINDS if tip in report["dead_ends"][k])


def test_each_tip_carries_its_own_kinds_and_a_real_branch_none():
    mask, G, tips = dead_end_cases()
    kinds = classify_dead_ends(G, mask, voxel_size_zyx=VOXEL_SIZE, image_face_margin_um=0.0)["kinds"]

    assert kinds[tips["real_branch"]] == ()
    assert "off_mask" in kinds[tips["off_mask"]]
    assert "short" in kinds[tips["short"]]
    assert kinds[tips["inside_other_lumen"]][0] == "inside_other_lumen"
    assert set(kinds) == set(tips.values())
