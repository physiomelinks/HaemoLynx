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
