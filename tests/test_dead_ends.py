"""One definition of what a dead end is (graph/dead_ends.py), and every caller
agreeing on it (PLAN_dead_ends.md, Phase 1)."""
from __future__ import annotations

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
        # Mask recovery's face test, on each tip with the heading out of it
        # (an island's end is an island end before it is anything else).
        for tip, end in ends.items():
            if end.heading is not None and end.kind != de.ISLAND_END:
                assert _recovery_leaves_image(end, support) == (tip in leaving), (case.name, tip)


def _recovery_leaves_image(end, support) -> bool:
    from haemolynx.graph.mask_recovery import _leaves_image_at

    return _leaves_image_at(np.asarray(end.tip_um), support, np.asarray(end.heading))


def test_the_facing_join_reads_the_chain_the_definition_walks():
    """PLAN_dead_ends.md 3.2, item 6: the facing join's heading and length
    come from the definition's chain (chain_to_junction), not a walk of its
    own."""
    from haemolynx.graph.facing_ends import _heading

    for make in CASES:
        case = make()
        for tip in [n for n in case.graph if case.graph.degree(n) == 1]:
            _nodes, edges = de.chain_to_junction(case.graph, tip)
            path = de.chain_path(case.graph, edges)
            direction, length = _heading(case.graph, tip)
            assert length == pytest.approx(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))), case.name
            assert direction is not None and float(direction @ de.tip_heading(path)) > 0.99, (case.name, tip)


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


def _island_nodes(G, ends):
    return {n for tip, end in ends.items() if end.kind == de.ISLAND_END for edge in end.chain for n in edge[:2]}


@pytest.mark.parametrize("make", CASES, ids=lambda make: make.__name__)
def test_removing_pieces_without_a_junction_takes_exactly_the_island_ends_chains(make):
    """The removal reads the one definition: it takes the nodes of every chain
    the classifier calls an island end, and nothing else."""
    case = make()
    support = MaskSupport(case.mask, VOXEL_SIZE)
    ends = de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support)
    islands = _island_nodes(case.graph, ends)
    before = set(case.graph.nodes)

    removed = de.remove_pieces_without_junction(case.graph)

    assert set(case.graph.nodes) == before - islands, case.name
    assert removed["nodes"] == len(islands)
    after = de.classify_dead_ends(case.graph, extent_um=EXTENT, support=support)
    assert de.ISLAND_END not in {end.kind for end in after.values()}, case.name


def test_an_island_of_two_edges_counts_as_one_piece_and_its_length():
    from dead_end_fixtures import island

    case = island()
    removed = de.remove_pieces_without_junction(case.graph)

    assert removed == {"pieces": 1, "vessels": 2, "nodes": 3, "length_um": pytest.approx(20.0)}
    assert case.graph.number_of_edges() == 3     # the trunk's two halves and the short spur
    assert (10.0, 50.0, 20.0) not in case.graph


def test_rings_lone_nodes_and_short_pieces_with_a_junction_stay_and_long_islands_go():
    from lumen_artefact_fixtures import polyline_graph

    star = [[(0, 0, 0), (0, 0, 3)], [(0, 0, 3), (0, 0, 6)], [(0, 0, 3), (0, 3, 3)]]   # 9 um, one junction
    ring = [[(0, 20, 0), (0, 20, 10)], [(0, 20, 10), (0, 30, 10)], [(0, 30, 10), (0, 20, 0)]]
    long_island = [[(0, 50, 0), (0, 50, 400)]]
    G = polyline_graph(star + ring + long_island)
    G.add_node((5.0, 5.0, 5.0), pos=np.asarray((5.0, 5.0, 5.0)))
    kept = set(G.nodes) - {(0.0, 50.0, 0.0), (0.0, 50.0, 400.0)}

    removed = de.remove_pieces_without_junction(G)

    assert set(G.nodes) == kept
    assert removed == {"pieces": 1, "vessels": 1, "nodes": 2, "length_um": pytest.approx(400.0)}
    assert de.remove_pieces_without_junction(G) == {"pieces": 0, "vessels": 0, "nodes": 0, "length_um": 0.0}


# --- remove_interior_dead_ends -------------------------------------------------


@pytest.mark.parametrize("make", CASES, ids=lambda make: make.__name__)
def test_removing_interior_dead_ends_takes_exactly_the_interior_kinds_chains(make):
    """It reads the kinds marked on the nodes: every chain whose tip is an
    interior kind goes back to its junction, and no other node does."""
    case = make()
    support = MaskSupport(case.mask, VOXEL_SIZE)
    ends = de.classify_and_mark(case.graph, extent_um=EXTENT, support=support)
    G = case.graph
    chains = [end.chain for end in ends.values()
              if end.kind in de.INTERIOR_DEAD_END_KINDS and G.degree(end.junction) > 1
              and not any(G.edges[a, b, k].get("is_zero_resistance") for a, b, k in end.chain)]
    H = G.copy()
    for chain in chains:
        H.remove_edges_from(chain)
    expected = {n for chain in chains for edge in chain for n in edge[:2] if H.degree(n) == 0}
    before = set(G.nodes)

    removed = de.remove_interior_dead_ends(G)

    assert before - set(G.nodes) == expected, case.name
    assert removed["nodes"] == len(expected) and removed["dead_ends"] == len(chains)
    assert not any(G.nodes[n].get(de.DEAD_END_KIND) in de.INTERIOR_DEAD_END_KINDS for n in G if G.degree(n) == 1)


def test_interior_dead_ends_go_to_their_junction_and_every_other_kind_stays():
    from lumen_artefact_fixtures import polyline_graph

    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    hub = (0.0, 0.0, 0.0)
    G = polyline_graph([
        [hub, (0, 0, 10)], [(0, 0, 10), (0, 0, 20)],     # stops short: two edges through a degree-2 node
        [hub, (0, 10, 0)],                                 # leaves the image
        [hub, (0, -10, 0)],                                # a person's real blind end
        [hub, (0, 0, -10)],                                # unknown, but its chain is a bridge
        [hub, (10, 0, 0)],                                 # no kind marked: not judged
    ])
    kinds = {(0.0, 0.0, 20.0): de.STOPS_SHORT, (0.0, 10.0, 0.0): de.LEAVES_IMAGE,
             (0.0, -10.0, 0.0): de.BLIND, (0.0, 0.0, -10.0): de.UNKNOWN}
    for node, kind in kinds.items():
        G.nodes[node][de.DEAD_END_KIND] = kind
    for *_e, data in G.edges((0.0, 0.0, -10.0), data=True):
        data[IS_ZERO_RESISTANCE] = True

    removed = de.remove_interior_dead_ends(G)

    assert removed == {"dead_ends": 1, "vessels": 2, "nodes": 2, "length_um": pytest.approx(20.0)}
    assert (0.0, 0.0, 10.0) not in G and (0.0, 0.0, 20.0) not in G
    assert hub in G and G.degree(hub) == 4
    assert de.remove_interior_dead_ends(G)["dead_ends"] == 0


def test_a_tree_loses_every_interior_chain_whatever_the_node_order():
    """Taken one at a time, the third arm of a star ran through the hub left
    with two vessels and was skipped as an island's, and which arms went hung
    on the node order."""
    from lumen_artefact_fixtures import polyline_graph

    hub = (0.0, 0.0, 0.0)
    arms = [(0, 0, 10), (0, 10, 0), (0, 0, -10), (0, -10, 0)]
    for order in (arms, arms[::-1]):
        G = polyline_graph([[hub, arm] for arm in order])
        for arm in order:
            G.nodes[tuple(float(c) for c in arm)][de.DEAD_END_KIND] = de.UNKNOWN

        removed = de.remove_interior_dead_ends(G)

        assert removed == {"dead_ends": 4, "vessels": 4, "nodes": 5, "length_um": pytest.approx(40.0)}
        assert G.number_of_nodes() == 0


def test_fixture_5_the_facing_pair_is_what_the_facing_join_joins():
    """PLAN_dead_ends.md Phase 0, fixture 5: the two tips are a pair the join
    accepts, and joining them leaves no dead end on either vessel."""
    from dead_end_fixtures import facing_pair
    from haemolynx.graph.facing_ends import facing_dead_end_pairs, join_facing_dead_ends

    case = facing_pair()
    support = MaskSupport(case.mask, VOXEL_SIZE)
    other_tip = (10.0, 46.0, 40.0)
    (pair,) = [p for p in facing_dead_end_pairs(case.graph, support) if {p.a, p.b} == {case.tip, other_tip}]
    assert pair.refused is None and pair.gap_um == pytest.approx(6.0)

    joined = join_facing_dead_ends(case.graph.copy(), support)

    assert joined.degree(case.tip) == 2 and joined.degree(other_tip) == 2
    assert case.tip not in de.classify_dead_ends(joined, extent_um=EXTENT, support=support)


def test_fixture_10_cluster_collapse_takes_a_tip_within_5_um_of_its_junction():
    """PLAN_dead_ends.md Phase 0, fixture 10 (inconsistency 8): cluster
    collapse merges every node within 5 um, so the 4 um stub's tip goes into
    its junction -- before any dead-end rule could judge it -- and the
    junction is left with the trunk alone."""
    from dead_end_fixtures import JUNCTION, tip_near_own_junction
    from haemolynx.graph.collapse import collapse_node_clusters

    case = tip_near_own_junction()
    collapsed = collapse_node_clusters(case.graph.copy(), distance_threshold=5.0)

    assert case.tip not in collapsed
    dead_ends = [n for n in collapsed if collapsed.degree(n) == 1]
    assert all(min(np.min(np.asarray(collapsed.nodes[n]["pos"])), 79.0 - np.max(collapsed.nodes[n]["pos"])) < 1.0
               for n in dead_ends), "only the trunk's two face ends are left"
    junctions = [n for n in collapsed if np.linalg.norm(np.asarray(collapsed.nodes[n]["pos"]) - JUNCTION) < 5.0]
    assert len(junctions) == 1


def test_the_vessels_the_image_faces_cut_are_the_chains_of_the_ends_leaving_it():
    """Decision D5: a vessel on a dead end's chain that leaves the image has
    a censored length; the VTK and the connectivity CSV flag it from here."""
    from dead_end_fixtures import leaves_through_face
    from haemolynx.graph._helpers import edge_id

    case = leaves_through_face()
    G = case.graph
    de.record_image_extent(G, SHAPE, VOXEL_SIZE)
    support = MaskSupport(case.mask, VOXEL_SIZE)
    de.classify_and_mark(G, support=support)

    cut = de.vessels_cut_by_image_faces(G)

    # The trunk's two halves leave through the x faces, the branch through y.
    assert cut == {edge_id(*e) for e in G.edges(keys=True)}
    G.nodes[case.tip][de.DEAD_END_KIND] = "unknown"
    assert len(de.vessels_cut_by_image_faces(G)) == 2
