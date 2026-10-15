"""The clean-up that ends graph building: degree-2 merging, edges off the mask,
two edges in one lumen, loops inside one lumen and the stub prune, round after
round until nothing changes (``graph.assemble.consolidate_lumen``).

Run once in a fixed order, what the later steps left was never judged: a
second strand in pieces is only recognised once its pieces are merged, which
the old single pass did last.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import diagnose_parallel_duplicates_in_lumen
from haemolynx.graph.assemble import consolidate_lumen, lumen_cleanup
from haemolynx.graph.degree2 import smart_multigraph_degree2_removal
from haemolynx.graph.lumen_loops import (
    cuts_off_vessels,
    remove_loops_inside_one_lumen,
    remove_parallel_edges_in_lumen,
)
from haemolynx.graph.mask_recovery import remove_edges_off_the_mask
from haemolynx.preprocessing import MaskSupport

from lumen_artefact_fixtures import (
    VOXEL_SIZE,
    capsule,
    polyline_graph,
    ring_in_one_lumen,
    two_strands_in_one_lumen,
    wide_vessel,
)


def _chain(y, x_from, x_to, piece):
    """A strand along x at height y, in pieces of *piece* um joined at degree-2 nodes."""
    xs = np.arange(x_from, x_to + 0.5, piece)
    return [[(7, y, a), (7, y, b)] for a, b in zip(xs[:-1], xs[1:])]


def _strands_in_pieces():
    """A medial strand and a second strand 3 um from it in one wide lumen, both
    in 8 um pieces: too short, once a lumen radius is left off each end, for
    any one piece to be judged a duplicate."""
    return wide_vessel(), polyline_graph(_chain(7, 4, 52, 8) + _chain(10, 4, 52, 8))


def _no_length_prune(mask):
    return lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape,
                         min_stub_length=0.0, min_stub_length_radius_multiple=0.0)


def _pairs(G, mask):
    return diagnose_parallel_duplicates_in_lumen(G, mask, voxel_size_zyx=VOXEL_SIZE)["duplicate_pair_count"]


def test_a_second_strand_in_pieces_is_found_once_merged_and_taken_out():
    mask, G = _strands_in_pieces()
    assert _pairs(G, mask) == 0  # in pieces, nothing is judged a duplicate

    G = consolidate_lumen(G, _no_length_prune(mask))

    assert _pairs(G, mask) == 0
    assert G.number_of_edges() == 1
    (u, v), = G.edges()
    assert {np.asarray(G.nodes[n]["pos"])[1] for n in (u, v)} == {7.0}  # the medial strand stays
    assert all(G.degree[n] != 2 for n in G.nodes)


def test_the_old_single_pass_left_that_strand_behind():
    """Loops, pairs and prune before the last merge, as graph building ran
    them before: the pair only exists after the merge, and nothing judges it.
    (The prune of the time: no length rule, and none of today's mask rules,
    one of which would take this strand out on its own.)"""
    from haemolynx.graph.prune import prune_vascular_stubs

    mask, G = _strands_in_pieces()
    cleanup = _no_length_prune(mask)
    G = remove_loops_inside_one_lumen(G, cleanup.support)
    G = remove_parallel_edges_in_lumen(G, cleanup.support)
    G = prune_vascular_stubs(G, min_stub_length=0.0)
    G = smart_multigraph_degree2_removal(G, None, max_degree=4, inside_lumen=cleanup.inside_lumen)
    G = remove_edges_off_the_mask(G, cleanup.support)
    assert _pairs(G, mask) == 1


def test_a_ring_inside_one_lumen_is_broken_and_the_rest_merged():
    mask, G = ring_in_one_lumen()
    G = consolidate_lumen(G, _no_length_prune(mask))
    assert nx.cycle_basis(nx.Graph(G)) == []
    assert G.number_of_edges() == 1
    assert G.number_of_edges(*next(iter(G.edges()))) == 1


def test_without_the_mask_only_degree2_nodes_are_merged():
    mask, G = _strands_in_pieces()
    cleanup = lumen_cleanup(None, VOXEL_SIZE, image_shape=mask.shape)
    assert cleanup.support is None
    G = consolidate_lumen(G, cleanup)
    assert G.number_of_edges() == 2  # one per strand: merged, nothing else judged
    assert all(G.degree[n] == 1 for n in G.nodes)


def test_consolidation_settles_and_running_it_again_changes_nothing():
    mask, G = _strands_in_pieces()
    cleanup = _no_length_prune(mask)
    once = consolidate_lumen(G, cleanup)
    twice = consolidate_lumen(once.copy(), cleanup)
    assert sorted(map(str, once.edges(keys=True))) == sorted(map(str, twice.edges(keys=True)))


# --- the duplicate removal never cuts vessels off ---------------------------


def _fork_with_a_branch(branch_on_both: bool):
    """Two edges leaving one node through one lumen; the wall-side one (y=9)
    carries a branch out of the vessel, and with *branch_on_both* the medial
    one (y=6) carries another."""
    mask = capsule(np.zeros((15, 45, 70), dtype=bool), (7, 7, -10), (7, 7, 60), 6)
    capsule(mask, (7, 9, 57), (7, 40, 57), 2)
    paths = [
        [(7, 7, 2), (7, 6, 12), (7, 6, 57)],
        [(7, 7, 2), (7, 9, 12), (7, 9, 57)],
        [(7, 9, 57), (7, 40, 57)],
    ]
    if branch_on_both:
        capsule(mask, (7, 6, 57), (7, 6, 68), 2)
        paths.append([(7, 6, 57), (7, 6, 68)])
    return mask, polyline_graph(paths)


def _edge(G, a, b):
    a, b = tuple(map(float, a)), tuple(map(float, b))
    return (a, b, next(iter(G[a][b])))


def test_the_wall_side_duplicate_stays_when_it_alone_carries_a_branch():
    mask, G = _fork_with_a_branch(branch_on_both=False)
    support = MaskSupport(mask, VOXEL_SIZE)
    assert _pairs(G, mask) == 1
    G = remove_parallel_edges_in_lumen(G, support)
    assert nx.number_connected_components(G) == 1
    assert G.has_edge((7.0, 9.0, 57.0), (7.0, 40.0, 57.0))
    assert not G.has_node((7.0, 6.0, 57.0))  # the medial one went, its dead end with it


def test_a_pair_both_of_whose_edges_carry_vessels_is_left_whole():
    mask, G = _fork_with_a_branch(branch_on_both=True)
    support = MaskSupport(mask, VOXEL_SIZE)
    edges = G.number_of_edges()
    G = remove_parallel_edges_in_lumen(G, support)
    assert G.number_of_edges() == edges
    assert nx.number_connected_components(G) == 1


def test_cutting_off_is_judged_on_what_lies_beyond_the_edge():
    _mask, G = _fork_with_a_branch(branch_on_both=False)
    assert cuts_off_vessels(G, _edge(G, (7, 7, 2), (7, 9, 57)))
    # A dead end goes with its edge: nothing is cut off.
    assert not cuts_off_vessels(G, _edge(G, (7, 7, 2), (7, 6, 57)))
    G.add_edge((7.0, 6.0, 57.0), (7.0, 9.0, 57.0))
    # A loop closed round the two: either can go.
    assert not cuts_off_vessels(G, _edge(G, (7, 7, 2), (7, 9, 57)))


# --- after the thick-vessel split: bridges in the clean-up ---------------------


def _mark_bridge(G, a, b):
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    G.edges[_edge(G, a, b)][IS_ZERO_RESISTANCE] = True


def _bridges(G):
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    return [(u, v) for u, v, d in G.edges(data=True) if d.get(IS_ZERO_RESISTANCE)]


def _centreline_um(G):
    return sum(float(d["length"]) for *_e, d in G.edges(data=True))


def test_a_degree_two_node_between_a_bridge_and_a_vessel_is_never_merged():
    """A bridge is not vessel: merged into one, the vessel would lose its
    resistance or the bridge gain one."""
    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)]])
    _mark_bridge(G, (7, 7, 20), (7, 7, 30))
    G = smart_multigraph_degree2_removal(G, None)
    assert G.number_of_edges() == 3
    assert len(_bridges(G)) == 1


def test_a_loop_inside_one_lumen_through_two_bridges_to_one_node_goes_whole():
    """A vessel the split bridged at both ends to the same centreline node,
    with 20 um of it left between the bridges inside the thick lumen: the
    network meets the ring once, so strand and bridges all go, and the
    thick centreline is left merged."""
    mask = wide_vessel()
    G = polyline_graph([[(7, 7, 2), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)], [(7, 11, 20), (7, 11, 40)],
                        [(7, 11, 20), (7, 7, 30)], [(7, 11, 40), (7, 7, 30)]])
    _mark_bridge(G, (7, 11, 20), (7, 7, 30))
    _mark_bridge(G, (7, 11, 40), (7, 7, 30))
    G = consolidate_lumen(G, _no_length_prune(mask))
    assert nx.cycle_basis(nx.Graph(G)) == []
    assert not _bridges(G)
    assert G.number_of_edges() == 1
    assert 54.0 <= _centreline_um(G) <= 56.0


def test_a_loop_through_two_bridges_to_two_nodes_loses_the_strand_with_its_bridges():
    """Bridged to two places on the centreline, the ring has two sides: the
    strand along the wall goes, bridges and all, and the centreline stays."""
    mask = wide_vessel()
    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 40)], [(7, 7, 40), (7, 7, 57)],
                        [(7, 11, 20), (7, 11, 40)], [(7, 11, 20), (7, 7, 20)], [(7, 11, 40), (7, 7, 40)]])
    _mark_bridge(G, (7, 11, 20), (7, 7, 20))
    _mark_bridge(G, (7, 11, 40), (7, 7, 40))
    G = consolidate_lumen(G, _no_length_prune(mask))
    assert nx.cycle_basis(nx.Graph(G)) == []
    assert not _bridges(G)
    assert G.number_of_edges() == 1
    assert 54.0 <= _centreline_um(G) <= 56.0


def test_a_bridge_left_dangling_by_a_clean_up_goes_too():
    """The pair step takes out a strand beside the centreline; the bridge
    that joined the strand's end to the centreline is then a dead end, and
    a bridge is never one."""
    from haemolynx.graph.thick_vessel_junctions import remove_dangling_bridges

    mask = wide_vessel()
    G = polyline_graph([[(7, 7, 2), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)], [(7, 11, 5), (7, 11, 50)],
                        [(7, 11, 50), (7, 7, 30)]])
    _mark_bridge(G, (7, 11, 50), (7, 7, 30))
    strand = G.copy()
    strand.remove_edge(*_edge(strand, (7, 11, 5), (7, 11, 50)))
    assert remove_dangling_bridges(strand) == 1
    assert not _bridges(strand) and not strand.has_node((7.0, 11.0, 50.0))

    G = consolidate_lumen(G, _no_length_prune(mask))
    assert not _bridges(G)
    assert G.number_of_edges() == 1
    assert 54.0 <= _centreline_um(G) <= 56.0



def test_a_piece_bridged_at_both_ends_to_one_node_goes_with_its_bridges():
    """A short strand in the thick lumen, each end only a bridge, both to the
    centreline's node at y = 30 (what the tree fill left at grid V's V13 on
    E14.5): not a vessel opening into the thick one, and contracted for the
    statistics a loop on one node. A vessel bridged at both ends to two
    places on the centreline is a loop through tissue and stays."""
    from haemolynx.graph.thick_vessel_junctions import remove_dangling_bridges

    G = polyline_graph([[(7, 7, 2), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)], [(7, 11, 27), (7, 11, 33)],
                        [(7, 11, 27), (7, 7, 30)], [(7, 11, 33), (7, 7, 30)]])
    _mark_bridge(G, (7, 11, 27), (7, 7, 30))
    _mark_bridge(G, (7, 11, 33), (7, 7, 30))
    assert remove_dangling_bridges(G) == 2
    assert not _bridges(G) and G.number_of_edges() == 2
    assert not G.has_node((7.0, 11.0, 27.0)) and not G.has_node((7.0, 11.0, 33.0))

    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 40)], [(7, 7, 40), (7, 7, 57)],
                        [(7, 11, 20), (7, 11, 40)], [(7, 11, 20), (7, 7, 20)], [(7, 11, 40), (7, 7, 40)]])
    _mark_bridge(G, (7, 11, 20), (7, 7, 20))
    _mark_bridge(G, (7, 11, 40), (7, 7, 40))
    assert remove_dangling_bridges(G) == 0
    assert len(_bridges(G)) == 2 and G.number_of_edges() == 6


def test_a_second_clean_up_on_the_same_object_judges_only_what_changed(monkeypatch):
    """build_network cleans up after smoothing and again after the split:
    the verdicts live on the LumenCleanup, so the second call judges no
    unchanged edge again."""
    import haemolynx.graph.doubled_stretches as stretches

    mask, G = _strands_in_pieces()
    cleanup = _no_length_prune(mask)
    G = consolidate_lumen(G, cleanup)
    assert cleanup.memos["stretches"] is not None and "pairs" in cleanup.memos
    judged = []
    original = stretches._judge_edge
    monkeypatch.setattr(stretches, "_judge_edge", lambda *a, **k: (judged.append(1), original(*a, **k))[1])
    again = consolidate_lumen(G.copy(), cleanup)
    assert judged == []
    assert sorted(map(str, again.edges(keys=True))) == sorted(map(str, G.edges(keys=True)))


def test_the_clean_up_breaks_short_loops_round_tissue_only_when_asked():
    """``remove_short_loops_under_um`` reaches the clean-up: on by default in
    the schema (200 um), a loop the in-lumen rule keeps goes; at 0 it stays."""
    import networkx as nx

    from haemolynx.graph.lumen_loops import LOOP_SEARCH_UM

    mask = np.zeros((15, 20, 60), dtype=bool)
    idx = np.indices(mask.shape)
    mask |= (idx[0] - 7) ** 2 + (idx[1] - 7) ** 2 <= 9
    mask[:, 6:9, 22:39] = False  # tissue between the two strands

    def loop_graph():
        G = nx.MultiGraph()
        for node, x in (("A", 0.0), ("u", 20.0), ("v", 40.0), ("B", 59.0)):
            G.add_node(node, pos=np.array([7.0, 7.0, x]))
        straight = lambda x0, x1: [[7.0, 7.0, float(x)] for x in range(x0, x1 + 1)]
        length = lambda p: float(np.sum(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1)))
        G.add_edge("A", "u", voxels=straight(0, 20), length=length(straight(0, 20)))
        G.add_edge("v", "B", voxels=straight(40, 59), length=length(straight(40, 59)))
        for row in (4, 10):
            path = [[7.0, 7.0, 20.0]] + [[7.0, float(row), float(x)] for x in range(22, 39)] + [[7.0, 7.0, 40.0]]
            G.add_edge("u", "v", voxels=path, length=length(path), row=row)
        return G

    # Not asked: the clean-up carries no limit (the other steps may still judge
    # the loop by the mask; test_lumen_loops pins that the in-lumen rule keeps it).
    assert lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0).max_short_loop_um == 0.0
    cleanup = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0,
                            remove_short_loops_under_um=LOOP_SEARCH_UM)
    assert cleanup.max_short_loop_um == LOOP_SEARCH_UM
    broken = consolidate_lumen(loop_graph(), cleanup)
    assert nx.cycle_basis(nx.Graph(broken)) == []
    assert nx.is_connected(broken)
    assert set(broken.nodes) >= {"A", "B"}


# --- a thick vessel narrowing into a thin one: its taper in the clean-up --------


def test_a_pair_with_a_thick_vessel_taper_is_left_alone():
    """A taper lies in the thick lumen beside whatever else does: it is the
    thick vessel narrowing, never the second strand of a pair."""
    from haemolynx.graph.thick_vessel_junctions import THICK_TAPER

    from lumen_artefact_fixtures import two_strands_in_one_lumen

    mask, G = two_strands_in_one_lumen()
    support = MaskSupport(mask, VOXEL_SIZE)
    assert remove_parallel_edges_in_lumen(G.copy(), support).number_of_edges() == 1
    for u, v, key in list(G.edges(keys=True))[:1]:
        G.edges[u, v, key][THICK_TAPER] = True
    assert remove_parallel_edges_in_lumen(G, support).number_of_edges() == 2


def test_a_taper_is_never_merged_with_a_vessel_so_its_mark_stays_on_it():
    """Regression (review of 100293f..e1cf508): merged at a degree-2 node,
    the taper lent the whole vessel beyond it the clean-up's protection, and
    each merge after spread it further, to the thick vessel's own centreline
    and to vessels nowhere near a thick lumen. Dropped at the merge instead,
    the mark would protect nothing in the next round. So a taper is kept
    apart from a vessel as a bridge is; two halves of one taper merge."""
    from haemolynx.graph.degree2 import create_trivial_merged_edge
    from haemolynx.graph.thick_vessel_junctions import THICK_TAPER

    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)]])
    G.edges[_edge(G, (7, 7, 20), (7, 7, 30))][THICK_TAPER] = True
    G = smart_multigraph_degree2_removal(G, None)
    assert G.number_of_edges() == 3
    (taper,) = [d for *_e, d in G.edges(data=True) if d.get(THICK_TAPER)]
    assert taper["length"] == pytest.approx(10.0)

    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 30)], [(7, 7, 30), (7, 7, 57)]])
    for a, b in (((7, 7, 20), (7, 7, 30)), ((7, 7, 30), (7, 7, 57))):
        G.edges[_edge(G, a, b)][THICK_TAPER] = True
    G = smart_multigraph_degree2_removal(G, None)
    assert G.number_of_edges() == 2
    (taper,) = [d for *_e, d in G.edges(data=True) if d.get(THICK_TAPER)]
    assert taper["length"] == pytest.approx(37.0)

    vessel, end = {"voxels": [(0, 0, 0), (0, 0, 5)]}, {"voxels": [(0, 0, 5), (0, 0, 9)], THICK_TAPER: True}
    assert not create_trivial_merged_edge(vessel, end, (0, 0, 5)).get(THICK_TAPER)
    assert not create_trivial_merged_edge(end, vessel, (0, 0, 5)).get(THICK_TAPER)
    assert create_trivial_merged_edge(end, dict(end), (0, 0, 5)).get(THICK_TAPER) is True


def test_merging_keeps_what_smoothing_did_saying_the_less_smoothed_half():
    """Regression: both merges dropped ``centreline_smoothing``, so the
    vessels a clean-up after smoothing merged said nothing of what happened
    to them (test_centreline_smoothing's provenance test failed)."""
    from haemolynx.graph.degree2 import create_trivial_merged_edge

    G = polyline_graph([[(7, 7, 2), (7, 7, 20)], [(7, 7, 20), (7, 7, 57)]])
    G.edges[_edge(G, (7, 7, 2), (7, 7, 20))]["centreline_smoothing"] = "smoothed"
    G.edges[_edge(G, (7, 7, 20), (7, 7, 57))]["centreline_smoothing"] = "kept_raw"
    G = smart_multigraph_degree2_removal(G, None)
    ((_u, _v, data),) = G.edges(data=True)
    assert data["centreline_smoothing"] == "kept_raw"

    both = create_trivial_merged_edge(
        {"voxels": [(0, 0, 0), (0, 0, 5)], "centreline_smoothing": "smoothed"},
        {"voxels": [(0, 0, 5), (0, 0, 9)], "centreline_smoothing": "smoothed"}, (0, 0, 5),
    )
    assert both["centreline_smoothing"] == "smoothed"


def test_the_clean_up_leaves_a_short_loop_round_a_wide_hole():
    """``short_loop_keep_hole_over_um`` reaches the clean-up: a capillary ring
    round a block of tissue (grids Q and Q2's real loops) survives every
    round by default and is broken at 0."""
    from test_lumen_loops import _capillary_ring

    from haemolynx.graph.lumen_loops import DEFAULT_SHORT_LOOP_KEEP_HOLE_UM, LOOP_SEARCH_UM

    G, mask = _capillary_ring()
    cleanup = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0,
                            remove_short_loops_under_um=LOOP_SEARCH_UM)
    assert cleanup.short_loop_keep_hole_um == DEFAULT_SHORT_LOOP_KEEP_HOLE_UM
    kept = consolidate_lumen(G.copy(), cleanup)
    loops = lambda H: H.number_of_edges() - H.number_of_nodes() + nx.number_connected_components(H)
    length = lambda H: sum(d["length"] for *_e, d in H.edges(data=True))
    assert loops(kept) == 1 and length(kept) >= 180.0

    every = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0,
                          remove_short_loops_under_um=LOOP_SEARCH_UM, short_loop_keep_hole_over_um=0.0)
    broken = consolidate_lumen(G.copy(), every)
    assert loops(broken) == 0 and length(broken) < length(kept) - 80.0
    assert nx.is_connected(broken)


# --- a thick vessel's centreline: the strand that stays -------------------------


def _row(y):
    """A stand-in for a thick vessel's centreline tree along row *y*: the µm
    of a polyline on it."""
    from test_lumen_loops import _on_row

    return _on_row(y)


def test_the_pair_rule_keeps_the_strand_on_a_thick_vessels_centreline():
    """Of two strands in one lumen the one nearer the wall goes -- unless it
    is the thick vessel's own centreline (E14.5, 2026-10-15: the pair rule
    took the strand on the tree as often as not, breaking the vessel)."""
    mask = wide_vessel()
    support = MaskSupport(mask, VOXEL_SIZE)
    strands = lambda: polyline_graph([[(7, 4, 2), (7, 4, 57)], [(7, 8, 2), (7, 8, 57)]])
    assert _pairs(strands(), mask) == 1
    rows = lambda G: sorted(float(np.asarray(d["voxels"])[0][1]) for *_e, d in G.edges(data=True))

    assert rows(remove_parallel_edges_in_lumen(strands(), support)) == [8.0]
    assert rows(remove_parallel_edges_in_lumen(strands(), support, along_centreline=_row(4))) == [4.0]


def test_the_stub_prune_judges_a_dead_end_on_a_thick_vessels_centreline_as_any_other():
    """Kept as the thick vessel running on, such a dead end was wrong at both
    places it decided on Harvey's grid W (2026-10-15, 5 and 6)."""
    from lumen_artefact_fixtures import dead_end_cases

    mask, G, tips = dead_end_cases()
    tip = tips["inside_other_lumen"]  # a hair along the trunk, on row 32
    on_tree = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, along_thick_centreline=_row(32))
    pruned = on_tree.prune(G.copy())
    assert not pruned.has_node(tip)
    assert not pruned.has_node(tips["off_mask"]) and pruned.has_node(tips["real_branch"])


def test_the_short_loop_rule_is_not_given_the_thick_centreline(monkeypatch):
    """Keeping the side on the tree, the short-loop rule took the capillary
    beside it: both places it decided that way on grid W (2 and 22) were
    marked wrong. The pair, in-lumen loop and stretch rules are given it."""
    import haemolynx.graph.assemble as assemble

    seen = {}
    for name in ("remove_parallel_edges_in_lumen", "remove_loops_inside_one_lumen",
                 "remove_short_loops", "resolve_doubled_stretches"):
        original = getattr(assemble, name)

        def spy(G, *args, _name=name, _original=original, **kwargs):
            seen.setdefault(_name, []).append(kwargs.get("along_centreline"))
            return _original(G, *args, **kwargs)

        monkeypatch.setattr(assemble, name, spy)
    mask, G = two_strands_in_one_lumen()
    along = _row(5)
    cleanup = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, remove_short_loops_under_um=200.0,
                            along_thick_centreline=along)

    consolidate_lumen(G, cleanup)

    assert set(seen["remove_short_loops"]) == {None}
    for name in ("remove_parallel_edges_in_lumen", "remove_loops_inside_one_lumen", "resolve_doubled_stretches"):
        assert set(seen[name]) == {along}


def _thick_vessel_running_on_into_a_narrow_one():
    """Harvey's breaks (Forclaude.haemorun, E14.5, 2026-10-15) in miniature:
    a 12 um vessel (the thick region) runs on into a 4 um one. Its
    centreline is one edge from the thick vessel well into the narrow one; a
    second strand 2 um off it lies wholly in the wide lumen. Averaged along
    each, the centreline runs nearer the wall, and the clean-up kept the
    second strand."""
    mask = np.zeros((15, 40, 90), dtype=bool)
    capsule(mask, (7, 20, -10), (7, 20, 40), 6)
    capsule(mask, (7, 20, 40), (7, 20, 100), 2)
    thick = capsule(np.zeros_like(mask), (7, 20, -10), (7, 20, 36), 6)
    G = polyline_graph([
        [(7, 20, 0), (7, 20, 2)],
        [(7, 20, 2), (7, 20, 78)],
        [(7, 20, 2), (7, 18, 6), (7, 18, 34), (7, 20, 38)],
    ])
    return mask, thick, G


def _followed(G, centreline) -> float:
    """The share of *centreline*'s points within 1.5 um of an edge of *G*."""
    points = np.vstack([np.asarray(d["voxels"], dtype=float) for *_e, d in G.edges(data=True)])
    distance = np.linalg.norm(centreline[:, None] - points[None], axis=2).min(axis=1)
    return float(np.mean(distance <= 1.5))


def test_the_clean_up_keeps_a_thick_vessels_centreline_running_on_into_a_narrow_vessel():
    from haemolynx.graph.thick_vessel_junctions import ThickCentreline

    mask, thick, G = _thick_vessel_running_on_into_a_narrow_one()
    middle = np.array([[7.0, 20.0, float(x)] for x in range(4, 34)])
    tree = ThickCentreline.from_region(thick, VOXEL_SIZE, segmentation=mask)
    assert tree.along_um(middle) == pytest.approx(29.0, abs=1.0)

    def clean(along):
        cleanup = lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=0.0,
                                min_stub_length_radius_multiple=0.0, along_thick_centreline=along)
        return consolidate_lumen(G.copy(), cleanup)

    assert _followed(clean(None), middle) < 0.2  # the wall alone kept the second strand
    kept = clean(tree.along_um)
    assert _followed(kept, middle) == 1.0
    assert nx.is_connected(kept) and nx.cycle_basis(nx.Graph(kept)) == []
    assert kept.has_node((7.0, 20.0, 78.0))
    off_centre = [p for *_e, d in kept.edges(data=True) for p in d["voxels"] if abs(p[1] - 18.0) < 0.5]
    assert off_centre == []
