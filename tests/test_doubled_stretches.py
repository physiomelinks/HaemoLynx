"""Stretches of centreline doubled in one lumen, rewired by their shape
(``graph.doubled_stretches``): a dead end beside a vessel is cut back, a fork
is zipped, a stretch inside an edge is cut out and its ends joined to the
centreline kept. Nothing is deleted whole, nothing cut off, and a
thick-vessel bridge is never touched."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import diagnose_lumen_artefacts, diagnose_parallel_duplicates_in_lumen
from haemolynx.graph.assemble import consolidate_lumen, lumen_cleanup
from haemolynx.graph.doubled_stretches import doubled_stretches, resolve_doubled_stretches
from haemolynx.graph.lumen_loops import remove_parallel_edges_in_lumen
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.preprocessing import MaskSupport

from lumen_artefact_fixtures import (
    VOXEL_SIZE,
    capsule,
    dead_end_beside_vessel,
    fork_that_parts,
    polyline_graph,
    strand_through_a_trunk,
    two_strands_in_one_lumen,
)


def _support(mask):
    return MaskSupport(mask, VOXEL_SIZE)


def _doubled_um(G, mask):
    report = diagnose_lumen_artefacts(G, mask, voxel_size_zyx=VOXEL_SIZE, image_face_margin_um=0.0)
    return report["duplicated_centreline_um"]


def _pairs(G, mask):
    return diagnose_parallel_duplicates_in_lumen(G, mask, voxel_size_zyx=VOXEL_SIZE)["duplicate_pair_count"]


def _cleanup(mask, min_stub_length=10.0, multiple=3.0):
    return lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape,
                         min_stub_length=min_stub_length, min_stub_length_radius_multiple=multiple)


def _length(G):
    return sum(d["length"] for _u, _v, d in G.edges(data=True))


def test_a_dead_end_beside_its_vessel_is_cut_back_to_where_it_leaves_it():
    mask, G, tip, junction = dead_end_beside_vessel()
    before = _doubled_um(G, mask)
    G = resolve_doubled_stretches(G, _support(mask))
    assert not G.has_node(tip)
    trunk_ends = {(7.0, 7.0, 0.0), (7.0, 7.0, 59.0)}
    (stub,) = [n for n in G.neighbors(junction) if G.degree[n] == 1 and n not in trunk_ends]
    # Cut back to within a lumen radius or so of the junction it leaves.
    assert np.linalg.norm(np.asarray(G.nodes[stub]["pos"]) - np.asarray(junction)) < 10.0
    assert _doubled_um(G, mask) < 0.2 * before
    # The trunk is untouched.
    assert G.has_edge(junction, (7.0, 7.0, 59.0)) and G.has_edge((7.0, 7.0, 0.0), junction)


def test_the_clean_up_then_prunes_what_is_left_of_that_dead_end():
    mask, G, _tip, _junction = dead_end_beside_vessel()
    G = consolidate_lumen(G, _cleanup(mask))
    assert G.number_of_edges() == 1
    assert {float(np.asarray(G.nodes[n]["pos"])[1]) for n in G.nodes} == {7.0}


def test_a_fork_is_zipped_up_to_where_its_edges_part():
    mask, G, (trunk_end, tip_a, tip_b) = fork_that_parts()
    G = resolve_doubled_stretches(G, _support(mask))
    parting = [n for n in G.nodes if G.degree[n] == 3]
    assert len(parting) == 1
    where = np.asarray(G.nodes[parting[0]]["pos"])
    assert 30.0 < where[2] < 42.0  # where the two branches leave the wide lumen
    for tip in (tip_a, tip_b):
        assert nx.shortest_path(G, trunk_end, tip)[-2] == parting[0]
    assert _pairs(G, mask) == 0
    assert nx.cycle_basis(nx.Graph(G)) == []


def test_a_strand_through_a_trunk_is_cut_out_and_its_branches_joined_to_the_trunk():
    mask, G, (tip_a, tip_b, end_a, end_b) = strand_through_a_trunk()
    assert nx.number_connected_components(G) == 2
    before = _doubled_um(G, mask)
    G = resolve_doubled_stretches(G, _support(mask))
    assert nx.number_connected_components(G) == 1
    # The trunk still runs end to end, and each branch now hangs off it.
    assert nx.has_path(G, end_a, end_b)
    for tip in (tip_a, tip_b):
        assert nx.has_path(G, tip, end_a)
    assert _doubled_um(G, mask) < 0.2 * before
    assert nx.cycle_basis(nx.Graph(G)) == []


def test_resolving_shortens_the_centreline_and_repeating_settles():
    for build in (fork_that_parts, strand_through_a_trunk):
        mask, G, _ends = build()
        length = _length(G)
        G = resolve_doubled_stretches(G, _support(mask))
        assert _length(G) < length
        cleanup = _cleanup(mask, min_stub_length=0.0, multiple=0.0)
        once = consolidate_lumen(G, cleanup)
        twice = consolidate_lumen(once.copy(), cleanup)
        assert sorted(map(str, once.edges(keys=True))) == sorted(map(str, twice.edges(keys=True)))


def test_the_pair_rule_still_decides_whole_pairs_and_the_stretch_step_what_it_leaves():
    """The strand through the trunk is a pair with it (half of it beside the
    trunk) and the whole-edge rule takes the strand whole, as before; what
    the rule leaves -- a stretch shorter than half an edge -- is the stretch
    step's. Both end with the trunk drawn once and nothing cut off."""
    mask, G, (tip_a, tip_b, end_a, end_b) = strand_through_a_trunk()
    assert _pairs(G, mask) == 1
    G = consolidate_lumen(G, _cleanup(mask, min_stub_length=0.0, multiple=0.0))
    assert nx.has_path(G, end_a, end_b)
    assert _pairs(G, mask) == 0
    assert _doubled_um(G, mask) < 5.0


def test_a_thick_vessel_bridge_is_never_a_stretch_nor_a_partner():
    mask, G = two_strands_in_one_lumen()
    assert doubled_stretches(G, _support(mask))
    for u, v, key in list(G.edges(keys=True))[:1]:
        G.edges[u, v, key][IS_ZERO_RESISTANCE] = True
    assert doubled_stretches(G, _support(mask)) == []
    edges = sorted(map(str, G.edges(keys=True)))
    resolve_doubled_stretches(G, _support(mask))
    assert sorted(map(str, G.edges(keys=True))) == edges


def test_a_stretch_shorter_than_the_minimum_is_left_alone():
    mask, G, _tip, _junction = dead_end_beside_vessel()
    support = _support(mask)
    assert doubled_stretches(G, support, min_run_um=100.0) == []
    edges = sorted(map(str, G.edges(keys=True)))
    resolve_doubled_stretches(G, support, min_run_um=100.0)
    assert sorted(map(str, G.edges(keys=True))) == edges


def test_the_middle_edge_goes_when_it_lies_wholly_beside_one_carrying_a_branch():
    mask = capsule(np.zeros((15, 45, 70), dtype=bool), (7, 7, -10), (7, 7, 60), 6)
    capsule(mask, (7, 9, 57), (7, 40, 57), 2)
    G = polyline_graph([
        [(7, 7, 2), (7, 6, 12), (7, 6, 57)],
        [(7, 7, 2), (7, 9, 12), (7, 9, 57)],
        [(7, 9, 57), (7, 40, 57)],
    ])
    G = consolidate_lumen(G, _cleanup(mask))
    assert nx.number_connected_components(G) == 1
    assert G.has_node((7.0, 40.0, 57.0))
    assert _pairs(G, mask) == 0


def test_evidence_beyond_the_mask_keeps_a_stretch_apart():
    """A caller with evidence the mask lacks (the raw image) says which
    stretches are two vessels: those are left alone, the rest resolved."""
    mask, G, _tip, _junction = dead_end_beside_vessel()
    support = _support(mask)
    asked = []

    def two_vessels(run):
        asked.append(run)
        return True

    edges = sorted(map(str, G.edges(keys=True)))
    resolve_doubled_stretches(G, support, keep_apart=two_vessels)
    assert asked and sorted(map(str, G.edges(keys=True))) == edges
    resolve_doubled_stretches(G, support, keep_apart=lambda run: False)
    assert sorted(map(str, G.edges(keys=True))) != edges


def test_the_clean_up_hands_the_evidence_to_the_pair_rule_and_the_stretch_step():
    mask, G = two_strands_in_one_lumen()
    cleanup = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0,
                            min_stub_length_radius_multiple=0.0, keep_apart=lambda what: True)
    assert cleanup.keep_apart is not None
    G = consolidate_lumen(G, cleanup)
    assert G.number_of_edges() == 2  # the pair stays: two vessels by the evidence


# --- the clean-up settles: no round zips a fork back ---------------------------


def test_a_fork_is_zipped_only_where_its_edges_part_at_a_similar_distance():
    from haemolynx.graph.doubled_stretches import _part_together

    assert _part_together(10.0, 12.0, 1.0)
    assert _part_together(10.0, 5.0, 1.0) and _part_together(10.0, 20.0, 1.0)
    assert not _part_together(10.0, 4.9, 1.0) and not _part_together(10.0, 20.1, 1.0)
    assert not _part_together(0.5, 0.5, 1.0)  # shorter than the minimum shared stretch


def test_the_memo_remembers_where_zips_put_junctions_and_a_second_pass_changes_nothing():
    mask, G, _ends = fork_that_parts()
    support = _support(mask)
    memo = {}
    resolve_doubled_stretches(G, support, memo=memo)
    assert memo["zipped_at"], "the zip recorded the junction it left"
    edges = sorted(map(str, G.edges(keys=True)))
    resolve_doubled_stretches(G, support, memo=memo)
    assert sorted(map(str, G.edges(keys=True))) == edges


def test_a_zip_that_would_put_the_junction_back_where_one_was_is_refused():
    """A fork is not zipped to a junction a previous round's zip left behind:
    the memo holds that place and the zip is refused, so no round undoes
    the one before (E14.5's forks moved to and fro to the round limit)."""
    from haemolynx.graph.doubled_stretches import _Editor

    mask, G, (trunk_end, _a, _b) = fork_that_parts()
    support = _support(mask)
    resolve_doubled_stretches(G, support)
    (landed,) = [n for n in G.nodes if G.degree[n] == 3]
    place = _Editor.place(G.nodes[landed]["pos"])

    mask, G, _ends = fork_that_parts()
    resolve_doubled_stretches(G, _support(mask), memo={"zipped_at": {place}})
    # The fork may still be zipped the other way round (the other edge kept,
    # its parting vertex a few um away), but never onto the remembered place.
    assert all(_Editor.place(G.nodes[n]["pos"]) != place for n in G.nodes if G.degree[n] == 3)


# --- the finder judges only what changed between rounds ----------------------


def test_with_a_memo_an_unchanged_graph_is_not_judged_again_and_gives_the_same_runs(monkeypatch):
    import haemolynx.graph.doubled_stretches as module

    mask, G, _ends = strand_through_a_trunk()
    support = _support(mask)
    judged = []
    original = module._judge_edge

    def counting(*args, **kwargs):
        judged.append(args[:3])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_judge_edge", counting)
    memo = {}
    first = doubled_stretches(G, support, memo=memo)
    assert judged and memo["edges"]
    judged.clear()
    second = doubled_stretches(G, support, memo=memo)
    assert judged == [], "nothing changed, so no edge is judged again"
    assert [(r.edge, r.partner, r.start, r.stop) for r in second] == [(r.edge, r.partner, r.start, r.stop) for r in first]


def test_with_a_memo_an_edge_near_a_change_is_judged_again_and_one_far_away_is_not(monkeypatch):
    import haemolynx.graph.doubled_stretches as module

    mask, G, _ends = strand_through_a_trunk()
    # A capillary far from the trunk, on its own: judged once, then never again.
    capsule(mask, (7, 38, 60), (7, 38, 79), 1.5)
    G.add_node("far_a", pos=np.array([7.0, 38.0, 61.0]))
    G.add_node("far_b", pos=np.array([7.0, 38.0, 78.0]))
    G.add_edge("far_a", "far_b", voxels=[(7.0, 38.0, 61.0), (7.0, 38.0, 78.0)], length=17.0)
    support = _support(mask)
    judged = []
    original = module._judge_edge

    def counting(*args, **kwargs):
        judged.append((args[0], args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_judge_edge", counting)
    memo = {}
    doubled_stretches(G, support, memo=memo)
    judged.clear()
    # Move the strand a little: the strand and the trunk are judged again, the far capillary is not.
    strand = next(e for e in G.edges(keys=True) if (7.0, 37.0, 10.0) in e[:2])
    G.edges[strand]["voxels"] = [tuple(p[:2]) + (p[2] + 0.3,) for p in G.edges[strand]["voxels"]]
    again = doubled_stretches(G, support, memo=memo)
    assert again
    assert ("far_a", "far_b") not in judged and ("far_b", "far_a") not in judged
    assert len(judged) >= 1


def test_a_thick_vessel_taper_is_never_a_stretch_nor_a_partner():
    """The thin vessel a thick one narrows into runs inside the thick lumen
    from the thick centreline's end (``THICK_TAPER``, set by the split): the
    stretch step rewired it as a second strand, and the curved vessels round
    it came back as straight lines (E14.5, S25, 2026-10-13)."""
    from haemolynx.graph.thick_vessel_junctions import THICK_TAPER

    mask, G = two_strands_in_one_lumen()
    for u, v, key in list(G.edges(keys=True))[:1]:
        G.edges[u, v, key][THICK_TAPER] = True
    assert doubled_stretches(G, _support(mask)) == []
    edges = sorted(map(str, G.edges(keys=True)))
    resolve_doubled_stretches(G, _support(mask))
    assert sorted(map(str, G.edges(keys=True))) == edges


# --- a thick vessel's centreline: the strand that stays -------------------------


def _rows_between(G, x0, x1):
    points = np.vstack([np.asarray(d["voxels"], dtype=float) for *_e, d in G.edges(data=True)])
    inside = (points[:, 2] > x0) & (points[:, 2] < x1)
    return sorted(set(np.round(points[inside, 1]).tolist()))


def test_the_strand_on_a_thick_vessels_centreline_is_kept_and_the_other_rejoined_to_it():
    """Rewiring by the wall alone, the clean-up took the stretch on the thick
    centreline tree as often as not (E14.5, 2026-10-15). The strand through
    the trunk is, here, the one on the tree: the trunk's own stretch beside
    it is cut out instead, and its two ends joined to the strand."""
    from test_lumen_loops import _on_row

    mask, G, (_tip_a, _tip_b, end_a, end_b) = strand_through_a_trunk()
    plain = resolve_doubled_stretches(G.copy(), _support(mask))
    assert _rows_between(plain, 20, 40) == [20.0]

    G = resolve_doubled_stretches(G, _support(mask), along_centreline=_on_row(23))
    assert _rows_between(G, 20, 40) == [23.0]
    assert nx.is_connected(G) and nx.has_path(G, end_a, end_b)
    assert _doubled_um(G, mask) < 1.0


def test_a_dead_end_on_a_thick_vessels_centreline_is_not_cut_back():
    from test_lumen_loops import _on_row

    mask, G, tip, _junction = dead_end_beside_vessel()
    assert not resolve_doubled_stretches(G.copy(), _support(mask)).has_node(tip)

    length = _length(G)
    G = resolve_doubled_stretches(G, _support(mask), along_centreline=_on_row(10))
    assert G.has_node(tip) and _length(G) == pytest.approx(length)


@pytest.mark.parametrize("build", [dead_end_beside_vessel, fork_that_parts, strand_through_a_trunk])
def test_with_no_thick_centreline_near_every_stretch_is_resolved_as_before(build):
    """Far from any thick vessel the stretch step decides as it always did."""
    mask, G, *_ends = build()
    plain = resolve_doubled_stretches(G.copy(), _support(mask))
    nowhere = resolve_doubled_stretches(G.copy(), _support(mask), along_centreline=lambda points: 0.0)
    assert sorted(map(str, nowhere.edges())) == sorted(map(str, plain.edges()))
    assert _length(nowhere) == pytest.approx(_length(plain))


def test_a_stretch_with_no_length_on_the_tree_is_never_on_it():
    """Regression: a partner stretch narrowed to one point has no length, and
    "at least half of it on the tree" held for it (0 >= 0) -- every capillary
    beside one, far from any thick vessel, was taken for the wall side and
    rewired: on E14.5, 2.7 mm of vessel lost."""
    from haemolynx.graph.doubled_stretches import _mostly_on

    nowhere = lambda points: 0.0
    everywhere = lambda points: float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    point = np.array([[7.0, 7.0, 10.0]])
    line = np.array([[7.0, 7.0, 10.0], [7.0, 7.0, 20.0]])
    assert not _mostly_on(nowhere, point)
    assert not _mostly_on(everywhere, point)
    assert not _mostly_on(nowhere, line)
    assert _mostly_on(everywhere, line)
    assert not _mostly_on(lambda points: 4.0, line) and _mostly_on(lambda points: 5.0, line)
