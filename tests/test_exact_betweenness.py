"""statistics/exact_betweenness.py: exact betweenness in worker processes,
equal (``==``, not approximately) to ``nx.betweenness_centrality``, and the
batched pass the unweighted undirected graphs take, here and in the workers."""
from __future__ import annotations

import functools
import logging
import random

import networkx as nx
import numpy as np
import pytest

from haemolynx.statistics import exact_betweenness as eb
from haemolynx.statistics import network_measures
from haemolynx.statistics.comprehensive import compute_vessel_statistics_by_measure

#: Every parallel call here runs in two workers, whatever the graph's size.
parallel = functools.partial(eb.exact_betweenness_centrality, workers=2, min_nodes=0)
#: And every call of this one in this process.
here = functools.partial(eb.exact_betweenness_centrality, workers=1, min_nodes=0)


def _assert_bitwise_equal(got, expected):
    assert list(got) == list(expected), "nodes in another order"
    assert got == expected
    # == on floats already compares bits for these finite values; say so.
    assert np.array_equal(
        np.array(list(got.values()), dtype=float).view(np.int64),
        np.array(list(expected.values()), dtype=float).view(np.int64),
    )
    assert all(type(got[node]) is type(expected[node]) for node in expected)


def _assert_identical(G, **arguments):
    expected = nx.betweenness_centrality(G, **arguments)
    _assert_bitwise_equal(parallel(G, **arguments), expected)
    _assert_bitwise_equal(here(G, **arguments), expected)


def _scrambled(G, seed):
    """*G* with its nodes and each node's neighbours in a shuffled order and
    relabelled to strings: networkx's answer depends on both orders."""
    rng = random.Random(seed)
    nodes = list(G)
    rng.shuffle(nodes)
    edges = list(G.edges())
    rng.shuffle(edges)
    H = nx.Graph()
    H.add_nodes_from(f"n{node}" for node in nodes)
    H.add_edges_from((f"n{u}", f"n{v}") for u, v in edges)
    return H


def _random_vessel_like_graph(seed, n=400):
    """Several components: a few loopy beds, trees, pairs and lone nodes,
    with a self-loop or two."""
    rng = random.Random(seed)
    G = nx.Graph()
    offset = 0
    for size in (n, n // 3, 40, 7, 3, 2, 1, 1):
        part = nx.gnm_random_graph(size, int(size * 1.2), seed=rng.randrange(1 << 30))
        G.update(nx.relabel_nodes(part, {i: i + offset for i in part}))
        offset += size
    for _ in range(3):
        node = rng.randrange(offset)
        G.add_edge(node, node)
    return G


def _weighted_random_graph(n=120, p=0.04, seed=3):
    G = nx.gnp_random_graph(n, p, seed=seed)
    rng = random.Random(seed)
    for u, v in G.edges():
        # Few distinct lengths, so many shortest paths tie: the sums of
        # sigma and the tie-breaking are what the parallel pass must repeat.
        G[u][v]["length"] = rng.choice([1.0, 1.5, 2.0, 0.1 + 0.2])
    return G


def _vessel_multigraph(seed=5):
    """A grid of vessels with lengths, a few parallel vessels of other
    lengths, a self-loop and two pieces cut off from the rest."""
    rng = random.Random(seed)
    G = nx.MultiGraph()
    grid = nx.grid_2d_graph(7, 6)
    for node in grid:
        G.add_node(node, pos=np.array([0.0, float(node[0]), float(node[1])]))
    for u, v in grid.edges():
        G.add_edge(u, v, length=rng.choice([10.0, 12.5, 20.0]))
    for u, v in list(grid.edges())[::5]:
        G.add_edge(u, v, length=rng.choice([5.0, 30.0]))
    G.add_edge((3, 3), (3, 3), length=4.0)
    for i, node in enumerate([("piece", 0), ("piece", 1), ("piece", 2)]):
        G.add_node(node, pos=np.array([5.0, float(i), 0.0]))
    G.add_edge(("piece", 0), ("piece", 1), length=3.0)
    G.add_edge(("piece", 1), ("piece", 2), length=3.0)
    G.add_node("alone", pos=np.array([9.0, 9.0, 9.0]))
    return G


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("endpoints", [False, True])
def test_unweighted_shapes_are_identical(normalized, endpoints):
    arguments = dict(normalized=normalized, endpoints=endpoints)
    for G in (
        nx.path_graph(40),
        nx.grid_2d_graph(9, 8),
        nx.gnp_random_graph(150, 0.03, seed=1),
    ):
        _assert_identical(G, **arguments)


@pytest.mark.parametrize("normalized", [True, False])
def test_weighted_random_graph_is_identical(normalized):
    _assert_identical(_weighted_random_graph(), weight="length", normalized=normalized)


@pytest.mark.parametrize("weight", [None, "length"])
def test_multigraph_with_parallel_edges_and_pieces_is_identical(weight):
    G = _vessel_multigraph()
    assert any(G.number_of_edges(u, v) > 1 for u, v in G.edges())
    assert nx.number_connected_components(G) == 3
    _assert_identical(G, weight=weight)


def test_many_small_components_are_identical():
    """Each source reaches a few nodes of many, which the workers send back
    as node numbers and values rather than a row a node."""
    G = nx.disjoint_union_all(
        [nx.path_graph(5), nx.cycle_graph(7), nx.star_graph(6)] * 8
    )
    _assert_identical(G)
    _assert_identical(_weighted_random_graph(n=200, p=0.006, seed=9), weight="length")


def test_directed_graph_and_unordered_labels_are_identical():
    G = nx.gnp_random_graph(90, 0.05, seed=4, directed=True)
    G = nx.relabel_nodes(G, {v: f"n{(v * 37) % 90}" for v in G})
    for u, v in G.edges():
        G[u][v]["length"] = 1.0 + (len(u) + len(v)) % 3
    _assert_identical(G)
    _assert_identical(G, weight="length")


def test_callable_weight_is_identical():
    G = _weighted_random_graph()
    weight = lambda u, v, d: d["length"] * 2.0 + 0.5  # noqa: E731
    _assert_identical(G, weight=weight)


@pytest.mark.parametrize("seed", range(4))
def test_batched_pass_matches_networkx_on_scrambled_random_networks(seed):
    _assert_identical(_scrambled(_random_vessel_like_graph(seed), seed))


def test_batched_pass_matches_networkx_where_path_counts_pass_2_to_the_53():
    # On a 30 x 30 grid the corner-to-corner shortest paths number C(58, 29),
    # about 3e16: past 2**53, so each path count's float depends on the order
    # its terms were added in, and a dependency's on the order it was passed back.
    _assert_identical(_scrambled(nx.grid_2d_graph(30, 30), 7))


@pytest.mark.parametrize("slots, most", [(1, 128), (5, 3), (64, 128), (1 << 20, 1), (1 << 20, 128)])
def test_batched_pass_is_the_same_in_batches_and_tasks_of_any_size(monkeypatch, slots, most):
    G = _scrambled(nx.grid_2d_graph(9, 7), 3)
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    monkeypatch.setattr(eb, "BATCH_SLOTS", slots)
    monkeypatch.setattr(eb, "MAX_BATCH_SOURCES", most)
    _assert_identical(G)


@pytest.mark.parametrize(
    "G",
    [
        nx.Graph(),
        nx.empty_graph(1),
        nx.path_graph(2),
        nx.path_graph(3),
        nx.star_graph(6),
        nx.cycle_graph(11),
        nx.balanced_tree(3, 4),
        nx.complete_graph(9),
        nx.MultiGraph([(0, 1), (0, 1), (1, 2), (2, 3), (3, 0), (2, 2)]),
    ],
    ids=["empty", "one", "two", "path3", "star", "cycle", "tree", "complete", "multigraph"],
)
@pytest.mark.parametrize("normalized", [True, False])
def test_batched_pass_matches_networkx_on_small_shapes(G, normalized):
    _assert_identical(G, normalized=normalized)


def test_batched_tasks_keep_every_components_sources_in_order():
    for workers in (1, 3):
        tasks = eb._batched_tasks([5, 3000, 1, 40], workers)
        jobs = [job for task in tasks for job in task]
        for component, size in enumerate([5, 3000, 1, 40]):
            ranges = [(start, stop) for c, start, stop in jobs if c == component]
            assert ranges[0][0] == 0 and ranges[-1][1] == size
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        assert all(
            sum((stop - start) * [5, 3000, 1, 40][c] for c, start, stop in task) <= eb.BATCH_SLOTS
            or len(task) == 1
            for task in tasks
        )
    # Split for the workers: at least four batches a worker of a large component.
    assert len([job for task in eb._batched_tasks([3000], 3) for job in task]) >= 12


def test_below_min_nodes_or_one_worker_it_runs_here(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("worker processes started")

    monkeypatch.setattr(eb, "_start_workers", never)
    for G, weight in ((nx.grid_2d_graph(6, 6), None), (_weighted_random_graph(n=40), "length")):
        expected = nx.betweenness_centrality(G, weight=weight)
        _assert_bitwise_equal(eb.exact_betweenness_centrality(G, workers=4, weight=weight), expected)  # < min_nodes
        _assert_bitwise_equal(eb.exact_betweenness_centrality(G, workers=1, min_nodes=0, weight=weight), expected)
        with monkeypatch.context() as m:
            m.setattr(eb, "betweenness_worker_count", lambda n, m: 1)
            _assert_bitwise_equal(eb.exact_betweenness_centrality(G, min_nodes=0, weight=weight), expected)
    assert eb.exact_betweenness_centrality(nx.Graph(), workers=4, min_nodes=0) == {}


@pytest.mark.parametrize(
    "code_name, G, weight",
    [
        ("_WORKER_CODE", _weighted_random_graph(n=40), "length"),
        ("_BATCHED_WORKER_CODE", nx.grid_2d_graph(6, 6), None),
    ],
    ids=["networkx", "batched"],
)
def test_a_worker_that_fails_falls_back_to_this_process(monkeypatch, caplog, code_name, G, weight):
    monkeypatch.setattr(eb, code_name, "import sys; sys.exit(3)")
    with caplog.at_level(logging.WARNING, logger=eb.__name__):
        got = parallel(G, weight=weight)
    _assert_bitwise_equal(got, nx.betweenness_centrality(G, weight=weight))
    assert "worker processes failed" in caplog.text


@pytest.mark.parametrize(
    "code_name, line, weight",
    [("_WORKER_CODE", "rows = []", "length"), ("_BATCHED_WORKER_CODE", "blocks = []", None)],
    ids=["networkx", "batched"],
)
def test_a_worker_that_raises_falls_back_to_this_process(monkeypatch, caplog, code_name, line, weight):
    """A worker's own error comes back as one, not as a hang."""
    original = getattr(eb, code_name)
    code = original.replace(line, "raise ValueError('boom')")
    assert code != original
    monkeypatch.setattr(eb, code_name, code)
    G = nx.path_graph(30)
    nx.set_edge_attributes(G, 1.5, "length")
    with caplog.at_level(logging.WARNING, logger=eb.__name__):
        _assert_bitwise_equal(parallel(G, weight=weight), nx.betweenness_centrality(G, weight=weight))
    assert "boom" in caplog.text


def test_worker_count_is_the_default_capped_by_free_memory(monkeypatch):
    from haemolynx.haemodynamics import resistance
    from haemolynx.preprocessing import shared_volume_tasks

    monkeypatch.setattr(shared_volume_tasks, "default_worker_count", lambda: 8)
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: None)
    assert eb.betweenness_worker_count(30_000, 40_000) == 8
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 64 << 30)
    assert eb.betweenness_worker_count(30_000, 40_000) == 8
    per_worker = (
        eb.WORKER_BASE_BYTES
        + eb.WORKER_BYTES_PER_NODE * 30_000
        + eb.WORKER_BYTES_PER_EDGE * 2 * 40_000
    )
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 6 * per_worker)
    assert eb.betweenness_worker_count(30_000, 40_000) == 3
    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 0)
    assert eb.betweenness_worker_count(30_000, 40_000) == 1


def test_statistics_report_is_unchanged_by_the_workers(monkeypatch):
    """The full report's betweenness and the weighted betweenness summaries,
    from worker processes, are the very rows networkx gives."""
    G = _vessel_multigraph()
    for _, _, data in G.edges(data=True):
        data["resistance"] = data["length"] * 3.0
        data["flow_abs"] = 1.0 / data["length"]
    arguments = dict(
        statistics_mode="full", enabled_measures=frozenset({"betweenness", "basic"})
    )

    def networkx_only(G, **kwargs):
        return nx.betweenness_centrality(G, **kwargs)

    monkeypatch.setattr(network_measures, "exact_betweenness_centrality", networkx_only)
    expected_sections = compute_vessel_statistics_by_measure(G, **arguments)
    expected_weighted = network_measures.compute_betweenness_and_community_measurements(G)

    calls = []

    def counted(G, **kwargs):
        calls.append(kwargs.get("weight"))
        return parallel(G, **kwargs)

    monkeypatch.setattr(network_measures, "exact_betweenness_centrality", counted)
    sections = compute_vessel_statistics_by_measure(G, **arguments)
    weighted = network_measures.compute_betweenness_and_community_measurements(G)

    assert sections == expected_sections
    assert weighted == expected_weighted
    assert calls == [None] + ["analysis_weight"] * 3
    assert len(sections["betweenness"]) == G.number_of_nodes()
