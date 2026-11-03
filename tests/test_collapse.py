"""``graph.collapse``: clusters stay bounded, and edges follow the node they end on.

Each test here failed on the collapse this replaced: a chain of nearby nodes
became one node however long it was; the representative moved to its
cluster's centroid while its own edges went on ending where it used to be;
moved endpoints were rounded to a whole micron; and no edge's ``length`` was
measured again after any of it.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph._helpers import calculate_path_length
from haemolynx.graph.collapse import collapse_node_clusters, move_node_with_its_edges
from haemolynx.graph.direction_aware_collapse import collapse_node_clusters_direction_aware
from haemolynx.graph.persistence_collapse import collapse_node_clusters_persistence


def _chain(n: int, spacing: float) -> nx.MultiGraph:
    """Nodes 0..n-1 along x, *spacing* um apart, joined in a path."""
    G = nx.MultiGraph()
    for i in range(n):
        G.add_node(i, pos=np.array([0.0, 0.0, spacing * i]))
    for i in range(n - 1):
        voxels = [[0.0, 0.0, spacing * i], [0.0, 0.0, spacing * (i + 1)]]
        G.add_edge(i, i + 1, length=spacing, voxels=voxels)
    return G


def _edges_meet_their_nodes(G) -> None:
    """Every edge's path starts at one of its nodes and ends at the other, and
    its length is what that path measures."""
    for u, v, data in G.edges(data=True):
        voxels = np.asarray(data["voxels"], dtype=float)
        ends = {tuple(np.round(voxels[0], 9)), tuple(np.round(voxels[-1], 9))}
        nodes = {tuple(np.round(np.asarray(G.nodes[n]["pos"], dtype=float), 9)) for n in (u, v)}
        assert ends == nodes, f"edge {u}-{v} ends at {ends}, its nodes are at {nodes}"
        assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))


def test_a_long_chain_of_nearby_nodes_is_not_collapsed_into_one():
    """Regression: clusters were whole linked groups, so seven nodes 4 um
    apart -- a 24 um stretch of vessel -- became a single node."""
    G = _chain(7, 4.0)

    out = collapse_node_clusters(G, distance_threshold=5.0)

    assert out.number_of_nodes() >= 3
    positions = np.array([out.nodes[n]["pos"] for n in out.nodes])
    assert np.ptp(positions[:, 2]) >= 15.0, "the chain's extent survives"


def test_every_cluster_stays_within_the_threshold_of_its_representative():
    G = _chain(9, 3.0)
    original = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in G.nodes}

    out = collapse_node_clusters(G, distance_threshold=5.0, max_iterations=1)

    # One pass: each surviving node is one cluster's representative, now at
    # its cluster's centroid, which is within the threshold of its members.
    for node in out.nodes:
        centre = np.asarray(out.nodes[node]["pos"], dtype=float)
        assert np.linalg.norm(centre - original[node]) <= 5.0


def test_a_tight_cluster_still_collapses_to_one_node():
    G = nx.MultiGraph()
    for i, offset in enumerate(([0, 0, 0], [0, 1, 0], [0, 0, 1.5], [1, 1, 0])):
        G.add_node(i, pos=np.asarray(offset, dtype=float))
    for i, far in enumerate(([0, 20, 0], [0, -20, 0], [0, 0, 20], [20, 0, 0])):
        G.add_node(10 + i, pos=np.asarray(far, dtype=float))
        G.add_edge(i, 10 + i, voxels=[G.nodes[i]["pos"].tolist(), list(map(float, far))], length=0.0)

    out = collapse_node_clusters(G, distance_threshold=5.0)

    assert out.number_of_nodes() == 5
    _edges_meet_their_nodes(out)


@pytest.mark.parametrize(
    "collapse",
    [collapse_node_clusters, collapse_node_clusters_direction_aware, collapse_node_clusters_persistence],
)
def test_the_representatives_own_edges_follow_it_to_the_centroid(collapse):
    """Regression: only the merged members' edges were re-ended; the
    representative's own kept ending where it had been, up to the threshold
    away, and every length was left as it was measured before the move."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))  # degree 2: the representative
    G.add_node(1, pos=np.array([0.0, 0.0, 3.3]))
    G.add_node(10, pos=np.array([0.0, 25.0, 0.0]))
    G.add_node(11, pos=np.array([0.0, -25.0, 0.0]))
    G.add_node(12, pos=np.array([25.0, 0.0, 3.3]))
    for u, v in ((0, 10), (0, 11), (1, 12)):
        path = [G.nodes[u]["pos"].tolist(), G.nodes[v]["pos"].tolist()]
        G.add_edge(u, v, voxels=path, length=calculate_path_length(path))

    out = collapse(G, distance_threshold=5.0)

    assert 1 not in out.nodes
    assert np.allclose(out.nodes[0]["pos"], [0.0, 0.0, 1.65])
    _edges_meet_their_nodes(out)


def test_moved_endpoints_keep_their_fractional_microns():
    """Regression: a moved endpoint was rounded to a whole micron."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.3, 0.2, 0.7]))
    G.add_node(2, pos=np.array([0.3, 20.2, 0.7]))
    path = [[0.3, 0.2, 0.7], [0.3, 20.2, 0.7]]
    G.add_edge(1, 2, voxels=path, length=20.0)
    G.add_edge(0, 1, voxels=[[0.0, 0.0, 0.0], [0.3, 0.2, 0.7]], length=0.79)

    out = collapse_node_clusters(G, distance_threshold=5.0)

    rep = next(n for n in out.nodes if n != 2)
    (edge,) = [data for _, _, data in out.edges(data=True)]
    assert np.allclose(edge["voxels"][0], out.nodes[rep]["pos"])
    assert not np.allclose(edge["voxels"][0], np.round(out.nodes[rep]["pos"]))


def test_move_node_with_its_edges_re_ends_and_re_measures_every_edge():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 10.0]))
    path = [[0.0, 0.0, 0.0], [0.0, 3.0, 5.0], [0.0, 0.0, 10.0]]
    G.add_edge(0, 1, voxels=path, length=calculate_path_length(path))

    move_node_with_its_edges(G, 0, [0.0, 0.0, -2.0])

    (data,) = [d for _, _, d in G.edges(data=True)]
    assert data["voxels"][0] == [0.0, 0.0, -2.0]
    assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
    _edges_meet_their_nodes(G)


def _loop_between_close_junctions(arc_length_um: float) -> nx.MultiGraph:
    """Two junctions 3 um apart joined twice, straight and by an arc of
    *arc_length_um* (a half-ellipse in y), each leading on to an anchor."""
    import math

    G = nx.MultiGraph()
    a, b = [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]
    G.add_node(0, pos=np.array(a))
    G.add_node(1, pos=np.array(b))
    G.add_node(2, pos=np.array([-30.0, 0.0, 0.0]))
    G.add_node(3, pos=np.array([33.0, 0.0, 0.0]))
    G.add_edge(0, 1, voxels=[a, b], length=3.0)
    t = np.linspace(0.0, math.pi, 200)
    lo, hi = 0.0, arc_length_um
    for _ in range(60):
        h = 0.5 * (lo + hi)
        arc = np.stack([1.5 - 1.5 * np.cos(t), h * np.sin(t), np.zeros_like(t)], axis=1).tolist()
        lo, hi = (h, hi) if calculate_path_length(arc) < arc_length_um else (lo, h)
    G.add_edge(0, 1, voxels=arc, length=calculate_path_length(arc))
    G.add_edge(2, 0, voxels=[[-30.0, 0.0, 0.0], a], length=30.0)
    G.add_edge(1, 3, voxels=[b, [33.0, 0.0, 0.0]], length=30.0)
    return G


def test_the_plain_collapse_keeps_a_loop_longer_than_twice_the_threshold():
    """A 20 um arc between junctions 3 um apart is a vessel, not the gap
    between them: merging the two deleted it as a self-loop whatever its
    length. The second junction is left out of the cluster instead."""
    out = collapse_node_clusters(_loop_between_close_junctions(20.0), distance_threshold=5.0)

    assert out.has_node(0) and out.has_node(1)
    assert out.number_of_edges(0, 1) == 2
    assert max(d["length"] for d in out.get_edge_data(0, 1).values()) == pytest.approx(20.0, abs=0.1)
    assert not list(nx.selfloop_edges(out))


def test_the_plain_collapse_still_absorbs_a_ring_no_longer_than_twice_the_threshold():
    out = collapse_node_clusters(_loop_between_close_junctions(8.0), distance_threshold=5.0)

    assert out.number_of_nodes() == 3
    assert out.number_of_edges() == 2
    assert not list(nx.selfloop_edges(out))


# --- self-loops, tie order and the persistence collapse's guards -------------

COLLAPSES = (
    collapse_node_clusters, collapse_node_clusters_direction_aware, collapse_node_clusters_persistence,
)


def _total_length(G) -> float:
    return float(sum(d["length"] for *_e, d in G.edges(data=True)))


def _ring_on_a_stem() -> nx.MultiGraph:
    """A stem from node 0 to junction 1 and a 120 um ring leaving 1 and
    coming back to it, as skan traces it: one self-loop."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 30.0]))
    G.add_edge(0, 1, voxels=[[0.0, 0.0, float(x)] for x in range(31)], length=30.0)
    angles = np.linspace(np.pi, -np.pi, 121)
    ring = np.stack([np.zeros_like(angles), 19.1 * np.sin(angles), 49.1 + 19.1 * np.cos(angles)], axis=1)
    ring[0] = ring[-1] = (0.0, 0.0, 30.0)
    G.add_edge(1, 1, voxels=ring.tolist(), length=calculate_path_length(ring.tolist()))
    return G


@pytest.mark.parametrize("collapse", COLLAPSES, ids=lambda f: f.__name__)
def test_a_ring_on_one_junction_is_split_for_the_loop_rules_not_deleted(collapse):
    """Regression: every collapse deleted every self-loop, so a ring joined
    to the network at one junction went with no rule judging it, though no
    node merged. A long one is split at its far point into two edges, which
    the loop rules read; a short one, the gap a merge closed, still goes."""
    G = _ring_on_a_stem()
    G.add_edge(0, 0, voxels=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], length=2.0)

    collapsed = collapse(G, 5.0)

    assert nx.number_of_selfloops(collapsed) == 0
    assert _total_length(collapsed) == pytest.approx(_total_length(_ring_on_a_stem()))
    (far,) = set(collapsed) - {0, 1}
    assert collapsed.number_of_edges(1, far) == 2
    assert np.asarray(collapsed.nodes[far]["pos"])[2] == pytest.approx(49.1 + 19.1, abs=0.6)


def test_split_self_loops_puts_each_half_from_its_node_to_the_far_point():
    from haemolynx.graph._helpers import split_self_loops

    G = _ring_on_a_stem()
    assert split_self_loops(G, min_length_um=200.0) == 0  # shorter: left as it is
    assert split_self_loops(G) == 1
    (far,) = set(G) - {0, 1}
    for _u, _v, data in G.edges(far, data=True):
        voxels = np.asarray(data["voxels"])
        assert np.allclose(voxels[0], G.nodes[1]["pos"]) and np.allclose(voxels[-1], G.nodes[far]["pos"])
        assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))


def _junction_ladder(ids) -> nx.MultiGraph:
    """Ten junctions 4 um apart along a vessel, each with a 20 um side
    branch, the nodes numbered by *ids* (ten for the junctions, then ten)."""
    G = nx.MultiGraph()
    for i in range(10):
        G.add_node(ids[i], pos=np.array([0.0, 0.0, 4.0 * i]))
        G.add_node(ids[10 + i], pos=np.array([0.0, 20.0, 4.0 * i]))
        G.add_edge(ids[10 + i], ids[i], voxels=[[0, 20.0, 4.0 * i], [0, 0.0, 4.0 * i]], length=20.0)
    for i in range(9):
        G.add_edge(ids[i], ids[i + 1], voxels=[[0, 0, 4.0 * i], [0, 0, 4.0 * (i + 1)]], length=4.0)
    return G


def _shape(G):
    return sorted(
        tuple(sorted((tuple(np.round(G.nodes[u]["pos"], 6)), tuple(np.round(G.nodes[v]["pos"], 6)))))
        for u, v in G.edges()
    )


@pytest.mark.parametrize("collapse", COLLAPSES, ids=lambda f: f.__name__)
def test_a_collapse_does_not_depend_on_the_nodes_ids(collapse):
    """Regression: ties between representatives of one degree went to the
    lowest id, so the same network numbered another way collapsed another
    way, and a change elsewhere renumbering it moved its junctions. Ties go
    by position."""
    ids = list(range(20))
    shapes = {
        repr(_shape(collapse(_junction_ladder(order), 5.0)))
        for order in (ids, ids[::-1], [7, 3, 19, 0, 12, 5, 16, 1, 9, 14, 2, 18, 6, 11, 4, 17, 8, 13, 10, 15])
    }
    assert len(shapes) == 1


def test_the_persistence_collapse_neither_chains_nor_deletes_a_loops_arc():
    """Regression: with no gap to cut at, the persistence collapse took every
    merge within the collapse distance: the ladder of junctions chained into
    two nodes and one vessel, and a 60 um arc between two nodes 3 um apart
    was merged away. It holds a cluster to the collapse distance and absorbs
    no long vessel, as the other two methods do."""
    ladder = collapse_node_clusters_persistence(_junction_ladder(list(range(20))), 5.0)
    assert ladder.number_of_nodes() >= 8
    assert np.ptp([ladder.nodes[n]["pos"][2] for n in ladder]) > 30.0

    G = nx.MultiGraph()
    points = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 20.0), 2: (0.0, 0.0, 23.0), 3: (0.0, 0.0, 43.0)}
    for node, pos in points.items():
        G.add_node(node, pos=np.array(pos))
    t = np.linspace(0.0, 1.0, 40)
    arc = np.stack([np.zeros_like(t), 30.0 * np.sin(np.pi * t), 20.0 + 3.0 * t], axis=1)
    for u, v, voxels in ((0, 1, [points[0], points[1]]), (2, 3, [points[2], points[3]]),
                         (1, 2, arc.tolist()), (1, 2, [points[1], points[2]])):
        G.add_edge(u, v, voxels=[list(p) for p in voxels], length=calculate_path_length([list(p) for p in voxels]))

    kept = collapse_node_clusters_persistence(G, 5.0)

    assert _total_length(kept) == pytest.approx(_total_length(G))
