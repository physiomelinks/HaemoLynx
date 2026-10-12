"""One definition of what a dead end is (graph/dead_ends.py), and every caller
agreeing on it (PLAN_dead_ends.md, Phase 1)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from dead_end_fixtures import CASES, SHAPE, VOXEL_SIZE
from haemolynx.graph import dead_ends as de
from haemolynx.graph import network_review as nr
from haemolynx.graph.diagnostics import classify_dead_ends as diagnostics_classify
from haemolynx.graph.diagnostics import diagnose_lumen_artefacts
from haemolynx.preprocessing import MaskSupport

EXTENT = de.image_extent_um(SHAPE, VOXEL_SIZE)


def _judge(case):
    support = MaskSupport(case.mask, VOXEL_SIZE)
    return de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support), support


@pytest.mark.parametrize("make", CASES, ids=lambda make: make.__name__)
def test_each_fixture_gets_its_kind(make):
    case = make()
    ends, _support = _judge(case)
    assert case.tip in ends, f"{case.name}: {case.tip} is not a dead end"
    end = ends[case.tip]
    assert end.kind == case.kind, f"{case.name} ({case.why}): {end.kind} {end.measures()}"


def test_the_trunk_ends_leave_the_image():
    case = CASES[0]()
    ends, _support = _judge(case)
    trunk_ends = [n for n in ends if n[2] in (0.0, 79.0)]
    assert len(trunk_ends) == 2
    assert all(ends[n].kind == de.LEAVES_IMAGE for n in trunk_ends)


def test_a_dead_end_is_judged_as_its_whole_chain():
    """A chain in three pieces through degree-2 nodes: one dead end, its
    length the chain's, its junction the first node of degree three."""
    case = CASES[2]()  # interior_near_face: one 21 um branch
    G = case.graph
    tip = case.tip
    (_t, junction, key), = [(a, b, k) for a, b, k in G.edges(tip, keys=True)]
    data = G.edges[tip, junction, key]
    voxels = np.asarray(data["voxels"])
    if np.linalg.norm(voxels[0] - np.asarray(tip)) > np.linalg.norm(voxels[-1] - np.asarray(tip)):
        voxels = voxels[::-1]
    G.remove_edge(tip, junction, key)
    cut = [0, len(voxels) // 3, 2 * len(voxels) // 3, len(voxels) - 1]
    nodes = [tip] + [tuple(float(c) for c in voxels[i]) for i in cut[1:3]] + [junction]
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        if b not in G:
            G.add_node(b, pos=np.asarray(b))
        piece = voxels[cut[i]: cut[i + 1] + 1]
        G.add_edge(a, b, voxels=piece.tolist(), length=float(np.sum(np.linalg.norm(np.diff(piece, axis=0), axis=1))))
    ends, _support = _judge(case)
    end = ends[tip]
    assert end.junction == junction
    assert len(end.chain) == 3
    assert end.chain_length_um == pytest.approx(21.0, abs=0.6)
    assert end.kind == case.kind


def test_the_heading_points_out_of_the_tip():
    path = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 9.0], [0.0, 0.0, 5.0], [0.0, 3.0, 0.0]])
    assert np.allclose(de.tip_heading(path), [0.0, 0.0, 1.0])
    assert de.tip_heading(path[:1]) is None


def test_leaves_image_by_touch_by_reach_and_not_otherwise():
    case = CASES[1]()  # stops_short_of_face
    support = MaskSupport(case.mask, VOXEL_SIZE)
    up = np.array([0.0, 1.0, 0.0])
    # On the face, whatever the mask.
    assert de.leaves_image((10, 59, 40), EXTENT) == (True, False)
    # 8 um short, the tube running on to the face: reaches it.
    assert de.leaves_image((10, 51, 40), EXTENT, heading=up, support=support, tip_radius_um=2.5) == (True, True)
    # The same tip heading away from the face does not.
    assert de.leaves_image((10, 51, 40), EXTENT, heading=-up, support=support, tip_radius_um=2.5)[0] is False
    # Without the mask, 8 um is just 8 um from the face.
    assert de.leaves_image((10, 51, 40), EXTENT, heading=up)[0] is False
    # Too far to reach, even with the tube running all the way.
    assert de.leaves_image((10, 40, 40), EXTENT, heading=up, support=support, tip_radius_um=2.5)[0] is False


def test_a_fixed_face_margin_still_overrides_the_test():
    case = CASES[1]()
    support = MaskSupport(case.mask, VOXEL_SIZE)
    none_at_face = de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support, face_margin_um=0.0)
    assert none_at_face[case.tip].kind != de.LEAVES_IMAGE
    all_at_face = de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support, face_margin_um=200.0)
    assert all(end.kind in (de.LEAVES_IMAGE, de.ISLAND_END) for end in all_at_face.values())


def test_without_a_mask_only_the_geometric_kinds_are_told():
    case = CASES[3]()  # mask_runs_on: needs the mask to say so
    ends = de.classify_dead_ends(case.graph, extent_um=EXTENT)
    assert ends[case.tip].kind == de.UNKNOWN
    assert ends[case.tip].in_mask_fraction is None
    face = CASES[0]()
    assert de.classify_dead_ends(face.graph, extent_um=EXTENT)[face.tip].kind == de.LEAVES_IMAGE


def _second_island_ends(G, ends):
    """The far end of each island chain: the dead-end list shows a piece
    with no junction once, from its first end."""
    from haemolynx.graph._helpers import edge_id

    seen, second = set(), set()
    for tip, end in ends.items():
        chain = frozenset(edge_id(*e) for e in end.chain)
        if end.kind == de.ISLAND_END and chain in seen:
            second.add(tip)
        seen.add(chain)
    return second


def test_every_caller_gives_the_same_verdict():
    """The lumen-artefact report, its tip-by-tip classifier, the Post
    processing dead-end list and the boundary review all read the one
    definition: on one graph they agree tip for tip."""
    for make in CASES:
        case = make()
        support = MaskSupport(case.mask, VOXEL_SIZE)
        ends = de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support)
        leaving = {tip for tip, end in ends.items() if end.kind == de.LEAVES_IMAGE}
        judged = diagnostics_classify(case.graph, case.mask, voxel_size_zyx=VOXEL_SIZE, mask_support=support)
        assert set(judged["at_image_face"]) == leaving, case.name
        assert {tip: end.kind for tip, end in judged["ends"].items()} == {t: e.kind for t, e in ends.items()}
        report = diagnose_lumen_artefacts(case.graph, case.mask, voxel_size_zyx=VOXEL_SIZE, mask_support=support)
        assert report["dead_ends_at_image_face"] == len(leaving), case.name
        assert sum(report["dead_end_kind_counts"].values()) == len(ends)
        listed = {item.nodes[0] for item in nr.find_dead_ends(case.graph, support)}
        assert listed == set(ends) - leaving - _second_island_ends(case.graph, ends), case.name
        # The boundary review has no mask: it reads the kinds the run marked.
        de.mark_dead_end_kinds(case.graph, ends)
        open_at_face = {
            item.nodes[0] for item in nr.find_boundary_issues(case.graph, {}, extent_um=EXTENT)
            if item.measures["issue"] == "open end at a face"
        }
        assert open_at_face == {t for t in leaving if case.graph.degree(next(iter(case.graph.neighbors(t)))) > 1}, case.name


def test_marking_writes_the_kind_and_a_persons_verdict_stays():
    case = CASES[7]()  # blind_sprout
    support = MaskSupport(case.mask, VOXEL_SIZE)
    G = case.graph
    ends = de.classify_and_mark(G, extent_um=EXTENT, support=support)
    assert G.nodes[case.tip][de.DEAD_END_KIND] == de.UNKNOWN
    assert all(G.nodes[t][de.DEAD_END_KIND] == end.kind for t, end in ends.items())
    assert de.DEAD_END_KIND not in G.nodes[(10.0, 30.0, 40.0)]
    de.set_dead_end_kind(G, case.tip, de.BLIND)
    again = de.classify_and_mark(G, extent_um=EXTENT, support=support)
    assert again[case.tip].kind == de.BLIND
    assert again[case.tip].source == de.SOURCE_REVIEW
    assert G.nodes[case.tip][de.DEAD_END_KIND] == de.BLIND
    # Once the node is no longer a dead end, the verdict goes with it.
    G.add_edge(case.tip, (10.0, 30.0, 0.0), length=1.0)
    de.classify_and_mark(G, extent_um=EXTENT, support=support)
    assert de.DEAD_END_KIND not in G.nodes[case.tip]
    with pytest.raises(ValueError):
        de.set_dead_end_kind(G, case.tip, de.BLIND)
    with pytest.raises(ValueError):
        de.set_dead_end_kind(G, (10.0, 30.0, 79.0), "sprout")


def test_open_end_nodes_read_the_kinds_and_fall_back_to_distance():
    case = CASES[1]()  # its tip leaves only by the mask
    support = MaskSupport(case.mask, VOXEL_SIZE)
    G = case.graph
    de.record_image_extent(G, SHAPE, VOXEL_SIZE)
    # No kinds yet: by distance alone, only the tips on a face.
    assert set(de.open_end_nodes(G)) == {(10.0, 30.0, 0.0), (10.0, 30.0, 79.0)}
    de.classify_and_mark(G, support=support)
    assert set(de.open_end_nodes(G)) == {(10.0, 30.0, 0.0), (10.0, 30.0, 79.0), case.tip}


def test_measures_carry_what_a_grid_key_needs():
    case = CASES[4]()  # broken_towards_side
    ends, _support = _judge(case)
    row = ends[case.tip].measures()
    for name in ("kind", "chain_length_um", "tip_radius_um", "junction_radius_um", "face_distance_um",
                 "mask_run_um", "ahead_um", "ahead_junctions_away", "ahead_background_stretches", "depth_um"):
        assert name in row
    assert row["ahead_um"] == pytest.approx(9.0, abs=0.6)
    assert row["ahead_junctions_away"] is None  # another piece of the network
    assert row["ahead_background_stretches"] == 1


def test_a_neighbour_ahead_is_not_a_broken_vessel():
    """A vessel ahead that the network reaches one junction from the dead
    end's own -- a sibling round a corner -- is not a vessel the segmentation
    broke."""
    from dead_end_fixtures import JUNCTION

    case = CASES[4]()
    G = case.graph
    far, near, mid = (10.0, 10.0, 0.0), (10.0, 10.0, 79.0), (10.0, 10.0, 60.0)
    voxels = np.asarray(G.edges[far, near, 0]["voxels"])
    G.remove_edge(far, near, 0)
    G.add_node(mid, pos=np.asarray(mid))
    i = int(np.argmin(np.abs(voxels[:, 2] - 60.0)))
    G.add_edge(far, mid, voxels=voxels[: i + 1].tolist(), length=60.0)
    G.add_edge(mid, near, voxels=voxels[i:].tolist(), length=19.0)
    G.add_edge(mid, JUNCTION, voxels=[list(mid), list(JUNCTION)], length=28.3)
    ends, _support = _judge(case)
    end = ends[case.tip]
    assert end.ahead_um == pytest.approx(9.0, abs=0.6)
    assert end.ahead_junctions_away == 1
    assert end.kind == de.UNKNOWN


def test_census_lists_every_kind():
    case = CASES[0]()
    ends, _support = _judge(case)
    census = de.dead_end_census(ends)
    assert tuple(census) == de.DEAD_END_KINDS
    assert census[de.LEAVES_IMAGE] == 3
