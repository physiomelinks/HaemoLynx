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
