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
from haemolynx.graph.mask_recovery import (
    recover_uncovered_mask_vessels,
    remove_edges_off_the_mask,
    uncovered_mask_voxels,
)
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


def test_a_strand_with_no_join_is_not_left_as_an_island(monkeypatch):
    """Two strands in one piece: only the one that joins is added.

    The piece used to count as attached once any single end joined, so the
    other strand stayed in the graph with no path to the network.
    """
    from haemolynx.graph.mask_recovery import _Piece

    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 8, 30)
    _tube(mask, 2, (7, 35, 0), 2, 12, 40)
    near = np.array([[7.0, float(y), 30.0] for y in range(11, 31)])
    far = np.array([[7.0, 35.0, float(x)] for x in range(14, 38)])
    calls = {"n": 0}

    def both_strands(region, support, pad):
        calls["n"] += 1
        if calls["n"] > 1:
            return _Piece()
        piece = _Piece()
        piece.paths = [near, far]
        piece.ends = [((7, 11, 30), (7, 30, 30)), ((7, 35, 14), (7, 35, 37))]
        return piece

    monkeypatch.setattr("haemolynx.graph.mask_recovery._trace_piece", both_strands)

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert nx.number_connected_components(G) == 1
    recovered = [d for *_, d in G.edges(data=True) if d.get("recovered")]
    assert recovered
    points = np.vstack([np.asarray(d["voxels"]) for d in recovered])
    assert points[:, 1].max() < 32.0


def test_a_ring_round_a_hole_in_the_mask_is_not_added_as_a_self_loop(monkeypatch):
    """Regression: Lee thinning round a hole traces a path that ends where it
    starts. It was added as a self-loop -- 193 on the E14.5 stack, 33 of them
    left as islands once the stub prune took their tails -- and the flow
    solve refused the graph."""
    from haemolynx.graph.mask_recovery import _Piece

    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 8, 30)
    branch = np.array([[7.0, float(y), 30.0] for y in range(11, 31)])
    ring = np.array([[7.0, 26.0, 30.0], [7.0, 27.0, 31.0], [7.0, 28.0, 30.0],
                     [7.0, 27.0, 29.0], [7.0, 26.0, 30.0]])
    calls = {"n": 0}

    def branch_and_ring(region, support, pad):
        calls["n"] += 1
        if calls["n"] > 1:
            return _Piece()
        piece = _Piece()
        piece.paths = [branch, ring]
        piece.ends = [((7, 11, 30), (7, 30, 30)), ((7, 26, 30), (7, 26, 30))]
        return piece

    monkeypatch.setattr("haemolynx.graph.mask_recovery._trace_piece", branch_and_ring)

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert nx.number_of_selfloops(G) == 0
    assert any(d.get("recovered") for *_, d in G.edges(data=True))
    assert nx.number_connected_components(G) == 1


def test_a_join_that_cannot_be_made_does_not_leave_the_strand(monkeypatch):
    """Paths used to be added before the join, so a failed attach left them."""
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 7, 35)
    monkeypatch.setattr(
        "haemolynx.graph.mask_recovery._Attachments.attach_node",
        lambda self, owner, point_um, reserved: None,
    )

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    assert G.number_of_edges() == 1
    assert nx.number_connected_components(G) == 1
    assert not any(data.get("recovered") for *_, data in G.edges(data=True))


def test_a_piece_whose_centreline_is_only_isolated_voxels_traces_nothing():
    """Cut to an uncovered region, the mask's centreline can be a scatter of
    lone voxels, which skan cannot build a skeleton from at all."""
    from haemolynx.graph.mask_recovery import _trace_piece

    mask = np.zeros((5, 5, 12), dtype=bool)
    mask[2, 2, 1:11:3] = True
    region = np.argwhere(mask)

    piece = _trace_piece(region, _support(mask), pad=1)

    assert piece.paths == [] and piece.ends == []


def test_a_piece_whose_centreline_is_a_tiny_ring_traces_nothing(monkeypatch):
    """Lee thinning of a small blob can leave a three-voxel ring, for which
    skan's path buffer is too small ("index pointer size 1 should be 2").
    Its one path would be a self-loop, which recovery drops anyway."""
    from haemolynx.graph.mask_recovery import _trace_piece

    mask = np.zeros((5, 6, 6), dtype=bool)
    mask[2, 2, 2] = mask[2, 2, 3] = mask[2, 3, 2] = True
    monkeypatch.setattr("skimage.morphology.skeletonize", lambda image, method=None: image.copy())

    piece = _trace_piece(np.argwhere(mask), _support(mask), pad=1)

    assert piece.paths == [] and piece.ends == []


def test_a_tiny_ring_beside_a_centreline_goes_and_the_centreline_stays(monkeypatch):
    from haemolynx.graph.mask_recovery import _trace_piece

    mask = np.zeros((5, 8, 12), dtype=bool)
    mask[2, 2, 1:11] = True
    mask[2, 5, 1] = mask[2, 5, 2] = mask[2, 6, 1] = True
    monkeypatch.setattr("skimage.morphology.skeletonize", lambda image, method=None: image.copy())

    piece = _trace_piece(np.argwhere(mask), _support(mask), pad=1)

    assert piece.ends == [((2, 2, 1), (2, 2, 10))]


# --- loops: round tissue, or inside one lumen ---------------------------------


def _square_loop(z, lo, hi):
    """A closed square in the plane *z* with corners (lo, lo) and (hi, hi) in (y, x)."""
    corners = [(lo, lo), (lo, hi), (hi, hi), (hi, lo), (lo, lo)]
    return np.array([[z, y, x] for y, x in corners], dtype=float)


def _slab():
    mask = np.zeros((9, 30, 30), dtype=bool)
    mask[2:7, 2:28, 2:28] = True
    return mask


def test_a_ring_inside_one_lumen_loses_a_path_and_one_round_tissue_keeps_both():
    """Lee thinning of a blob traces rings: two paths between the same ends."""
    from haemolynx.graph.mask_recovery import _without_loops_in_one_lumen

    loop = _square_loop(4.0, 6.0, 22.0)
    a, b = (4, 6, 6), (4, 22, 22)
    strand = [(loop[:3], a, b, 0.0), (loop[2:][::-1], a, b, 0.0)]
    round_tissue = _slab()
    round_tissue[:, 9:20, 9:20] = False

    kept, dropped = _without_loops_in_one_lumen(strand, _support(_slab()))
    assert dropped == 1 and len(kept) == 1
    kept, dropped = _without_loops_in_one_lumen(strand, _support(round_tissue))
    assert dropped == 0 and len(kept) == 2


def _arch_off_the_trunk(filled: bool):
    """A vessel leaving the trunk at x = 20, running along y = 20 and back
    down to the trunk at x = 40, and its centreline. *filled* segments the
    tissue it encloses too, as a blob joined to the trunk."""
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 20), 2, 8, 20)
    _tube(mask, 1, (7, 0, 40), 2, 8, 20)
    _tube(mask, 2, (7, 20, 0), 2, 20, 40)
    if filled:
        mask[5:10, 7:21, 20:41] = True
    path = np.array(
        [[7.0, float(y), 20.0] for y in range(11, 20)]
        + [[7.0, 20.0, float(x)] for x in range(20, 41)]
        + [[7.0, float(y), 40.0] for y in range(19, 10, -1)]
    )
    return mask, path


def _recover_one_strand(monkeypatch, mask, path):
    from haemolynx.graph.mask_recovery import _Piece

    calls = {"n": 0}

    def the_strand(region, support, pad):
        calls["n"] += 1
        piece = _Piece()
        if calls["n"] == 1:
            piece.paths = [path]
            piece.ends = [((7, 11, 20), (7, 11, 40))]
        return piece

    monkeypatch.setattr("haemolynx.graph.mask_recovery._trace_piece", the_strand)
    return recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))


def _joins(G):
    return [d for *_, d in G.edges(data=True) if d.get("reconnected")]


def test_a_strand_in_a_blob_beside_its_vessel_is_not_left_as_a_spur(monkeypatch):
    """A strand lying in a blob round the vessel it came off used to be
    joined once and left as a dead end. That join closes a loop inside the
    lumen, so the dead end is taken back and the split it opened collapses."""
    mask, path = _arch_off_the_trunk(filled=True)

    G = _recover_one_strand(monkeypatch, mask, path)

    assert not any(d.get("recovered") for *_, d in G.edges(data=True))
    assert G.number_of_edges() == 1
    assert set(G.nodes) == {0, 1}
    assert nx.cycle_basis(nx.Graph(G)) == []


def test_a_vessel_leaving_and_rejoining_round_tissue_keeps_both_joins(monkeypatch):
    mask, path = _arch_off_the_trunk(filled=False)

    G = _recover_one_strand(monkeypatch, mask, path)

    assert len(_joins(G)) == 2
    assert len(nx.cycle_basis(nx.Graph(G))) == 1


def test_a_split_vessel_is_searched_without_the_joins_hanging_off_it():
    """Regression: a point on a vessel already split by a join could be taken
    to lie on that join, and the next strand was attached to the join -- 206
    times on the E14.5 stack -- instead of to the vessel."""
    from haemolynx.graph.mask_recovery import _Attachments

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([7.0, 7.0, 0.0]))
    G.add_node(1, pos=np.array([7.0, 7.0, 20.0]))
    G.add_node(2, pos=np.array([7.0, 7.0, 10.0]), _recovery_split=True)
    G.add_node(3, pos=np.array([7.0, 12.0, 14.0]))
    G.add_edge(0, 2, voxels=[[7.0, 7.0, float(x)] for x in range(0, 11)])
    G.add_edge(2, 1, voxels=[[7.0, 7.0, float(x)] for x in range(10, 21)])
    point = np.array([7.0, 7.4, 10.6])
    G.add_edge(2, 3, voxels=[[7.0, 7.0, 10.0], point.tolist(), [7.0, 12.0, 14.0]],
               recovered=True, reconnected=True)
    attachments = _Attachments(G, None, np.zeros(0), 4.0)

    u, v, _key = attachments._sub_edge(0, 1, point)

    assert {u, v} == {1, 2}


def test_new_node_ids_follow_on_from_the_graph_and_are_never_reused():
    from haemolynx.graph._helpers import next_node_id
    from haemolynx.graph.mask_recovery import _NodeIds

    G = nx.MultiGraph()
    G.add_nodes_from([0, 5, "a"])
    ids = _NodeIds(G)

    first = ids.take()
    assert first == next_node_id(G, set()) == 6
    G.add_node(first)
    G.remove_node(first)
    assert ids.take() == 7


def _trace(monkeypatch, paths, ends):
    from haemolynx.graph.mask_recovery import _Piece

    calls = {"n": 0}

    def the_piece(region, support, pad):
        calls["n"] += 1
        piece = _Piece()
        if calls["n"] == 1:
            piece.paths = paths
            piece.ends = ends
        return piece

    monkeypatch.setattr("haemolynx.graph.mask_recovery._trace_piece", the_piece)


def test_a_tip_within_reach_joins_across_a_short_dropout_at_a_new_node():
    """The circled case, scaled down: one voxel of background between the
    tip and the trunk, inside today's reach, lands on a new node."""
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 12, 35)

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    junctions = [n for n in G.nodes if G.degree[n] == 3]
    assert len(junctions) == 1
    assert junctions[0] not in (0, 1)
    assert np.allclose(np.asarray(G.nodes[junctions[0]]["pos"])[[0, 2]], (7.0, 30.0), atol=2.0)
    tips = [n for n in G.nodes if G.degree[n] == 1 and n not in (0, 1)]
    assert len(tips) == 1
    assert np.asarray(G.nodes[tips[0]]["pos"])[1] > 25.0
    assert diagnose_parallel_duplicates_in_lumen(G, mask)["duplicate_pair_count"] == 0


def test_a_joined_branch_whose_tip_stops_inside_the_mask_is_kept(monkeypatch):
    """The trace stops short of the mask end and nothing further is in reach.
    That tip is the vessel: it stays, joined at the near end, instead of the
    whole branch being deleted."""
    mask = _trunk_mask()
    _tube(mask, 1, (7, 0, 30), 2, 8, 35)
    path = np.array([[7.0, float(y), 30.0] for y in range(11, 22)])
    _trace(monkeypatch, [path], [((7, 11, 30), (7, 21, 30))])

    G = recover_uncovered_mask_vessels(_trunk_graph(), _support(mask))

    recovered = [d for *_, d in G.edges(data=True) if d.get("recovered") and not d.get("reconnected")]
    assert recovered
    tips = [n for n in G.nodes if G.degree[n] == 1 and n not in (0, 1)]
    assert len(tips) == 1
    assert float(np.asarray(G.nodes[tips[0]]["pos"])[1]) > 18.0
    assert G.graph["recovered_dead_ends_removed"] == 0
    assert G.graph["recovered_joins"] >= 1


def test_an_edge_through_empty_space_is_removed_and_one_in_the_mask_stays():
    """A centreline in the gap is not a vessel. The trunk stays, and so does
    a vessel that only clips outside the mask. The floating edge's nodes go
    with it."""
    mask = _trunk_mask()
    G = _trunk_graph()
    G.add_node(2, pos=np.array([2.0, 30.0, 10.0]))
    G.add_node(3, pos=np.array([2.0, 30.0, 25.0]))
    G.add_edge(2, 3, voxels=[[2.0, 30.0, float(x)] for x in range(10, 26)], length=15.0)
    clip = [[7.0, 7.0, float(x)] for x in range(0, 20)]
    clip += [[7.0, 12.0, float(x)] for x in range(20, 25)]
    clip += [[7.0, 7.0, float(x)] for x in range(25, 45)]
    G.add_node(4, pos=np.array([7.0, 7.0, 0.0]))
    G.add_node(5, pos=np.array([7.0, 7.0, 44.0]))
    G.add_edge(4, 5, voxels=clip, length=44.0)

    remove_edges_off_the_mask(G, _support(mask))

    assert G.has_edge(0, 1)
    assert G.has_edge(4, 5)
    assert not G.has_node(2) and not G.has_node(3)
    assert G.graph["edges_off_mask_removed"] == 1


def test_a_recovered_end_on_the_image_face_is_kept(monkeypatch):
    """The far tip lies on the volume face. A vessel beside it is in reach,
    but the join would fold back, and the end is kept anyway."""
    mask = np.zeros((15, 40, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 3, 0, 59)
    _tube(mask, 2, (7, 36, 0), 2, 0, 59)
    _tube(mask, 1, (7, 0, 30), 2, 10, 39)
    G = _trunk_graph()
    G.add_node(2, pos=np.array([7.0, 36.0, 0.0]))
    G.add_node(3, pos=np.array([7.0, 36.0, 59.0]))
    G.add_edge(2, 3, voxels=[[7.0, 36.0, float(x)] for x in range(60)], length=59.0)
    path = np.array([[7.0, float(y), 30.0] for y in range(12, 40)])
    _trace(monkeypatch, [path], [((7, 12, 30), (7, 39, 30))])

    G = recover_uncovered_mask_vessels(G, _support(mask))

    face = [n for n in G.nodes if G.degree[n] == 1 and np.asarray(G.nodes[n]["pos"])[1] > 38.0]
    assert len(face) == 1
    assert any(d.get("recovered") and not d.get("reconnected") for *_, d in G.edges(data=True))
    assert len(_joins(G)) == 1


def test_a_strand_with_two_joins_loses_only_its_spur(monkeypatch):
    """Two ends meet two trunks. A third end, in reach of the lower trunk,
    would close a loop inside that lumen and is the only part removed."""
    mask = np.zeros((15, 40, 60), dtype=bool)
    _tube(mask, 2, (7, 7, 0), 3, 0, 59)
    _tube(mask, 2, (7, 31, 0), 3, 0, 59)
    _tube(mask, 1, (7, 0, 30), 2, 8, 28)
    mask[5:10, 7:21, 20:50] = True
    lower = np.array([[7.0, float(y), 30.0] for y in range(11, 20)])
    upper = np.array([[7.0, float(y), 30.0] for y in range(19, 28)])
    spur = np.array([[7.0, 19.0 - i * 0.5, 30.0 + i] for i in range(13)])
    G = _trunk_graph()
    G.add_node(2, pos=np.array([7.0, 31.0, 0.0]))
    G.add_node(3, pos=np.array([7.0, 31.0, 59.0]))
    G.add_edge(2, 3, voxels=[[7.0, 31.0, float(x)] for x in range(60)], length=59.0)
    _trace(
        monkeypatch,
        [lower, upper, spur],
        [((7, 11, 30), (7, 19, 30)), ((7, 19, 30), (7, 27, 30)), ((7, 19, 30), (7, 13, 42))],
    )

    G = recover_uncovered_mask_vessels(G, _support(mask))

    assert len(_joins(G)) == 2
    assert G.graph["recovered_dead_ends_removed"] == 1
    assert not any(
        abs(float(np.asarray(G.nodes[n]["pos"])[1]) - 13.0) < 1.0
        and float(np.asarray(G.nodes[n]["pos"])[2]) > 35.0
        for n in G.nodes
    )
    assert nx.number_connected_components(G) == 1


def test_parallel_edges_in_one_lumen_lose_the_wall_side_and_a_tissue_loop_stays():
    from haemolynx.graph.lumen_loops import remove_loops_inside_one_lumen, remove_parallel_edges_in_lumen

    wide = np.zeros((15, 15, 60), dtype=bool)
    idx = np.indices(wide.shape)
    wide |= (idx[0] - 7) ** 2 + (idx[1] - 7) ** 2 <= 36
    G = nx.MultiGraph()
    for name, y in (("middle", 7.0), ("wall", 2.0)):
        a, b = f"{name}-a", f"{name}-b"
        path = [[7.0, y, float(x)] for x in range(5, 55)]
        G.add_node(a, pos=np.array([7.0, y, 5.0]))
        G.add_node(b, pos=np.array([7.0, y, 54.0]))
        G.add_edge(a, b, voxels=path, length=49.0)
    support = _support(wide)

    remove_parallel_edges_in_lumen(G, support)

    assert G.number_of_edges() == 1
    assert G.has_node("middle-a") and not G.has_node("wall-a")
    assert G.graph["parallel_lumen_edges_removed"] == 1

    tube = np.zeros((15, 20, 60), dtype=bool)
    idx = np.indices(tube.shape)
    tube |= (idx[0] - 7) ** 2 + (idx[1] - 7) ** 2 <= 9
    ring = nx.MultiGraph()
    ring.add_node("u", pos=np.array([7.0, 7.0, 20.0]))
    ring.add_node("v", pos=np.array([7.0, 7.0, 40.0]))
    for row in (4, 9):
        path = [[7.0, 7.0, 20.0]] + [[7.0, float(row), float(x)] for x in range(22, 39)] + [[7.0, 7.0, 40.0]]
        ring.add_edge("u", "v", voxels=path, length=24.0, row=row)
    holed = tube.copy()
    holed[:, 6:9, 22:39] = False

    remove_loops_inside_one_lumen(ring, _support(tube))
    assert [d["row"] for *_, d in ring.edges(data=True)] == [9]

    around = nx.MultiGraph()
    around.add_node("u", pos=np.array([7.0, 7.0, 20.0]))
    around.add_node("v", pos=np.array([7.0, 7.0, 40.0]))
    for row in (4, 9):
        path = [[7.0, 7.0, 20.0]] + [[7.0, float(row), float(x)] for x in range(22, 39)] + [[7.0, 7.0, 40.0]]
        around.add_edge("u", "v", voxels=path, length=24.0, row=row)
    remove_loops_inside_one_lumen(around, _support(holed))
    assert sorted(d["row"] for *_, d in around.edges(data=True)) == [4, 9]
