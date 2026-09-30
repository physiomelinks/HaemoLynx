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
