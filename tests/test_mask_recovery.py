"""``graph.mask_recovery``: vessels the mask has and the graph lost.

A branch whose bridge was refused, or a fragment the skeleton filter dropped,
leaves segmented vessel with no edge through it. Recovery traces those pieces
and joins them to the network through the mask -- never as a second strand
beside a vessel the graph already has, and never across background.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import diagnose_parallel_duplicates_in_lumen
from haemolynx.graph.mask_recovery import recover_uncovered_mask_vessels, uncovered_mask_voxels
from haemolynx.preprocessing import MaskSupport

pytest.importorskip("skan")


def _tube(mask, axis, centre, radius, lo, hi):
    """Fill a tube of *radius* along *axis* through *centre* from *lo* to *hi*."""
    idx = np.indices(mask.shape)
    others = [a for a in range(3) if a != axis]
    r2 = sum((idx[a] - centre[a]) ** 2 for a in others)
    mask |= (r2 <= radius**2) & (idx[axis] >= lo) & (idx[axis] <= hi)


def _trunk_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.graph["voxel_size"] = (1.0, 1.0, 1.0)
    G.add_node(0, pos=np.array([7.0, 7.0, 0.0]))
    G.add_node(1, pos=np.array([7.0, 7.0, 59.0]))
    G.add_edge(0, 1, voxels=[[7.0, 7.0, float(x)] for x in range(60)], length=59.0)
    return G


def _trunk_mask():
    mask = np.zeros((15, 40, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 3, 0, 59)
    return mask


def _support(mask):
    return MaskSupport(mask, (1.0, 1.0, 1.0))


def test_a_covered_vessel_leaves_nothing_to_recover():
    mask = _trunk_mask()

    assert len(uncovered_mask_voxels(_trunk_graph(), _support(mask))) == 0
    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert G.number_of_edges() == 1


def test_a_branch_the_graph_lost_is_traced_and_joined_to_its_trunk():
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 7, 35)  # a branch along y off the trunk

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    recovered = [d for *_, d in G.edges(data=True) if d.get("recovered")]
    assert recovered
    assert sum(d["length"] for d in recovered) > 15.0
    assert nx.number_connected_components(G) == 1
    junctions = [n for n in G.nodes if G.degree[n] == 3]
    assert len(junctions) == 1
    assert np.allclose(np.asarray(G.nodes[junctions[0]]["pos"])[[0, 2]], (7.0, 30.0), atol=1.0)
    assert all(d.get("bridge_kind") == "recovered" for *_, d in G.edges(data=True) if d.get("reconnected"))
    assert diagnose_parallel_duplicates_in_lumen(G, mask)["duplicate_pair_count"] == 0
    assert len(uncovered_mask_voxels(G, _support(mask))) < 0.05 * mask.sum()


def test_a_speck_and_a_vessel_out_of_reach_are_left_alone():
    """Recovery adds network, not islands: a piece below the volume floor,
    and a vessel no end of which lies near the network, add nothing."""
    mask = _trunk_mask()
    mask[7, 30:32, 10:12] = True  # 4 um^3 of speck
    _tube(mask, 2, (7, 30, 0), 2, 20, 50)  # a vessel 23 um from the trunk

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert G.number_of_edges() == 1


def _two_trunks_and_a_lost_capillary():
    """Trunks along x at y = 7 and y = 31, and a capillary between them at
    x = 30 that the graph does not have: 24 um centreline to centreline."""
    mask = np.zeros((15, 40, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 3, 0, 59)
    _tube(mask, 2, (7, 31, 0), 3, 0, 59)
    _tube(mask, 1, (7, 0, 30), 2, 7, 31)
    G = _trunk_graph()
    G.add_node(2, pos=np.array([7.0, 31.0, 0.0]))
    G.add_node(3, pos=np.array([7.0, 31.0, 59.0]))
    G.add_edge(2, 3, voxels=[[7.0, 31.0, float(x)] for x in range(60)], length=59.0)
    return G, mask


def test_a_capillary_lost_between_two_vessels_is_recovered_end_to_end():
    G, mask = _two_trunks_and_a_lost_capillary()
    before = len(uncovered_mask_voxels(G, _support(mask)))

    G = recover_uncovered_mask_vessels(G, _support(mask))

    recovered = [d for *_, d in G.edges(data=True) if d.get("recovered")]
    assert sum(d["length"] for d in recovered) == pytest.approx(24.0, rel=0.1)
    assert nx.number_connected_components(G) == 1
    junctions = sorted(
        tuple(np.round(np.asarray(G.nodes[n]["pos"])[[1, 2]]).astype(int))
        for n in G.nodes if G.degree[n] == 3
    )
    assert len(junctions) == 2
    assert np.allclose(junctions, [(7, 30), (31, 30)], atol=1.0)
    assert len(uncovered_mask_voxels(G, _support(mask))) <= before
    assert diagnose_parallel_duplicates_in_lumen(G, mask)["duplicate_pair_count"] == 0


def test_splitting_a_vessel_whose_path_runs_backwards_keeps_each_piece_on_its_nodes():
    """Regression: a trunk whose voxels ran from v to u was split as if they
    ran from u to v, so each new edge carried the other piece's path."""
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 20), 2, 7, 35)
    G = _trunk_graph()
    G.edges[0, 1, 0]["voxels"] = G.edges[0, 1, 0]["voxels"][::-1]

    G = recover_uncovered_mask_vessels(G, _support(mask))

    assert G.number_of_edges() > 1
    for u, v, data in G.edges(data=True):
        path = np.asarray(data["voxels"], dtype=float)
        ends = {tuple(np.round(path[0], 6)), tuple(np.round(path[-1], 6))}
        assert ends == {
            tuple(np.round(np.asarray(G.nodes[u]["pos"], dtype=float), 6)),
            tuple(np.round(np.asarray(G.nodes[v]["pos"], dtype=float), 6)),
        }
        assert data["length"] == pytest.approx(
            float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )


def test_a_vessel_behind_background_is_not_joined_across_it():
    """A parallel vessel 3 um of background away from the trunk's wall:
    the join to it would cross that background."""
    mask = _trunk_mask()
    _tube(mask, 2, (7, 16, 0), 2, 15, 45)

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert G.number_of_edges() == 1


def test_a_piece_whose_centreline_is_only_isolated_voxels_traces_nothing():
    """Cut to an uncovered region, the mask's centreline can be a scatter of
    lone voxels, which skan cannot build a skeleton from at all."""
    from haemolynx.graph.mask_recovery import _trace_piece

    mask = np.zeros((5, 5, 12), dtype=bool)
    mask[2, 2, 1:11:3] = True
    region = np.argwhere(mask)

    piece = _trace_piece(region, _support(mask), pad=1)

    assert piece.paths == [] and piece.ends == []
