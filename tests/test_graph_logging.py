"""What each topology pass did, without having to ask for debug.

The counts a user reads off a run -- how many terminal stubs pruning removed,
how many nodes and edges the graph had after each topology step -- were
``logger.debug`` behind ``if debug:``, so at default settings nothing said them
at all: not the console, and not anything else watching the ``haemolynx``
logger. They are ungated INFO now.

The tests here pin both halves of that. The summaries are said at INFO with
``debug=False``, and the numbers in them are the real ones -- the count
reported as removed equals the drop in node count, not just a number in a
sentence. And nothing finer joined them: a per-node INFO line would be a
hundred thousand lines on a real volume, so a default build emits no DEBUG
records at all and stays under a stated ceiling of records in total.
"""
from __future__ import annotations

import logging
import re

import networkx as nx

from haemolynx.graph import (
    STEP_LABELS,
    build_graph_from_skeleton,
    prune_vascular_stubs,
    smart_multigraph_degree2_removal,
)

GRAPH_LOGGER = "haemolynx.graph"

#: How many records a build of a tiny skeleton may emit at ``debug=False``.
#: Three lines of preamble, eleven steps, one summary per pass, and slack for
#: a warning; anything that adds a record *per node* or *per edge* blows
#: through it long before a real volume does.
RECORD_CEILING = 40


def _graph_with_two_short_stubs() -> nx.MultiGraph:
    """A three-node vessel with a two-micron stub hanging off each end node."""
    G = nx.MultiGraph()
    for node, pos in {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 20.0),
        2: (0.0, 0.0, 40.0),
        3: (0.0, 2.0, 20.0),
        4: (0.0, 2.0, 40.0),
    }.items():
        G.add_node(node, pos=pos)
    G.add_edge(0, 1, length=20.0)
    G.add_edge(1, 2, length=20.0)
    G.add_edge(1, 3, length=2.0)
    G.add_edge(2, 4, length=2.0)
    return G


def _graph_with_two_degree2_nodes() -> nx.MultiGraph:
    """A straight chain: the two interior nodes are degree-2 and removable."""
    G = nx.MultiGraph()
    positions = [(0.0, 0.0, float(10 * i)) for i in range(4)]
    for node, pos in enumerate(positions):
        G.add_node(node, pos=pos)
    for node in range(3):
        G.add_edge(
            node,
            node + 1,
            length=10.0,
            voxels=[positions[node], positions[node + 1]],
        )
    return G


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith(GRAPH_LOGGER)]


def _messages(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in _records(caplog) if r.levelno == level]


# --- the regression: the counts are said at all ------------------------------


def test_pruning_says_how_many_terminal_stubs_it_removed(caplog) -> None:
    """The line the user asked for, at the settings they actually run.

    This is the regression: with ``debug=False`` -- which is what
    ``verbose_logging`` is by default -- pruning used to log nothing.
    """
    G = _graph_with_two_short_stubs()
    before = G.number_of_nodes()

    with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
        pruned = prune_vascular_stubs(G, min_stub_length=10.0, debug=False)

    removed = before - pruned.number_of_nodes()
    assert removed == 2, "the fixture must lose exactly its two stubs"

    summaries = [m for m in _messages(caplog, logging.INFO) if "Pruning complete" in m]
    assert len(summaries) == 1, _messages(caplog, logging.INFO)
    assert re.search(rf"\b{removed}\b", summaries[0]), summaries[0]
    assert f"{pruned.number_of_nodes()} nodes" in summaries[0]
    assert f"{pruned.number_of_edges()} edges" in summaries[0]


def test_the_pruning_count_is_the_real_drop_in_nodes(caplog) -> None:
    """A number in a sentence is only worth reading if it is the true one."""
    G = _graph_with_two_short_stubs()
    # Every edge is now short, so pruning eats the whole graph iteratively.
    for _, _, data in G.edges(data=True):
        data["length"] = 1.0
    before = G.number_of_nodes()

    with caplog.at_level(logging.INFO, logger=GRAPH_LOGGER):
        pruned = prune_vascular_stubs(G, min_stub_length=10.0, debug=False)

    removed = before - pruned.number_of_nodes()
    summaries = [m for m in _messages(caplog, logging.INFO) if "Pruning complete" in m]
    assert len(summaries) == 1
    reported = int(re.search(r"removed (\d+)", summaries[0]).group(1))
    assert reported == removed


def test_degree2_removal_says_how_many_nodes_it_removed(caplog) -> None:
    G = _graph_with_two_degree2_nodes()
    before = G.number_of_nodes()

    with caplog.at_level(logging.INFO, logger=GRAPH_LOGGER):
        result = smart_multigraph_degree2_removal(G, skeleton_data=None, debug=False)

    removed = before - result.number_of_nodes()
    assert removed > 0, "the fixture must lose at least one degree-2 node"
    summaries = [m for m in _messages(caplog, logging.INFO) if "Smart removal" in m]
    assert len(summaries) == 1, _messages(caplog, logging.INFO)
    reported = int(re.search(r"(\d+) removed", summaries[0]).group(1))
    assert reported == removed


# --- one line per topology step ----------------------------------------------


def test_each_topology_step_reports_its_node_and_edge_count(caplog, tiny_skeleton) -> None:
    """Eleven lines per run: the label, and the graph it left behind.

    This is the "total number of branches, as per pipeline" the user reads off
    a run, and it has to be the count *at that step* rather than at the end.
    """
    snapshots: dict[str, tuple[int, int]] = {}

    def remember(G: nx.MultiGraph, label: str) -> None:
        snapshots[label] = (G.number_of_nodes(), G.number_of_edges())

    with caplog.at_level(logging.INFO, logger=GRAPH_LOGGER):
        build_graph_from_skeleton(tiny_skeleton, debug=False, step_callback=remember)

    info = _messages(caplog, logging.INFO)
    assert set(snapshots) == set(STEP_LABELS)
    for position, label in enumerate(STEP_LABELS, start=1):
        prefix = f"Step {position}/{len(STEP_LABELS)} {label}:"
        naming = [m for m in info if m.startswith(prefix)]
        assert len(naming) == 1, f"{label}: {naming}"
        nodes, edges = snapshots[label]
        assert f"{nodes} nodes" in naming[0], naming[0]
        assert f"{edges} edges" in naming[0], naming[0]
    steps = [m for m in info if m.startswith("Step ")]
    assert len(steps) == len(STEP_LABELS)


# --- and nothing finer -------------------------------------------------------


def test_a_default_build_emits_no_debug_records(caplog, tiny_skeleton) -> None:
    """`debug=False` is a bound on the volume, not just on the detail."""
    with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
        build_graph_from_skeleton(tiny_skeleton, debug=False)

    assert _messages(caplog, logging.DEBUG) == []


def test_a_default_build_stays_under_a_line_ceiling(caplog, tiny_skeleton) -> None:
    """A live log window has to survive a real run; per-node INFO would sink it."""
    with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
        build_graph_from_skeleton(tiny_skeleton, debug=False)

    emitted = _records(caplog)
    assert len(emitted) <= RECORD_CEILING, [r.getMessage() for r in emitted]


def test_asking_for_debug_still_says_more(caplog, tiny_skeleton) -> None:
    """The ungated summaries are additions, not a replacement for `debug=True`."""
    with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
        build_graph_from_skeleton(tiny_skeleton, debug=False)
    quiet = len(_records(caplog))

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
        build_graph_from_skeleton(tiny_skeleton, debug=True)
    loud = len(_records(caplog))

    assert loud > quiet
