"""The shared stand-in batch run (``make_batch_run`` in conftest.py).

Tests that need a batch run take it from the factory, which writes real files and opens them
through ``open_batch_run``. These tests pin what the factory hands out, so a driver test built
on it can trust the run it gets.
"""
import networkx as nx
import numpy as np
import pytest


def _triangle():
    G = nx.MultiGraph()
    for node, x in enumerate((0.0, 5.0, 10.0)):
        G.add_node(node, pos=np.array([x, 0.0, 0.0]))
    G.add_edge(0, 1, key=0)
    G.add_edge(1, 2, key=0)
    G.add_edge(1, 2, key=1)
    return G


def test_a_given_graph_and_columns_come_back_through_the_reader(make_batch_run):
    G = _triangle()
    lengths = {(0, 1, 0): 30.0, (1, 2, 0): 40.0, (1, 2, 1): 50.0}
    diameters = {(0, 1, 0): 7.0, (1, 2, 0): 5.5, (1, 2, 1): 4.0}
    run = make_batch_run(G=G, columns={"length_um": lengths,
                                       "assigned_diameter_um": diameters}).run

    assert run.numeric_column("length_um") == lengths
    joined = run.graph()
    assert {(u, v, k): d["assigned_diameter_um"]
            for u, v, k, d in joined.edges(keys=True, data=True)} == diameters
    assert np.array_equal(joined.nodes[2]["pos"], [10.0, 0.0, 0.0])


def _three_edge_columns(drop=None, extra=None):
    lengths = {edge: 10.0 for edge in _triangle().edges(keys=True)}
    diameters = dict(lengths)
    if drop:
        del lengths[drop]
    if extra:
        diameters[extra] = 5.0
    return {"length_um": lengths, "assigned_diameter_um": diameters}


@pytest.mark.parametrize("columns, message", [
    (_three_edge_columns(drop=(1, 2, 1)), r"no length_um given for edge \(1, 2, 1\)"),
    (_three_edge_columns(extra=(7, 8, 0)), r"assigned_diameter_um.*not in the graph.*\(7, 8, 0\)"),
], ids=["graph edge with no value", "value for an edge the graph lacks"])
def test_columns_that_do_not_match_the_graph_one_to_one_are_refused(make_batch_run, columns,
                                                                     message):
    """A missing value is not filled and an extra key is not ignored, as the reader's strict
    join would."""
    with pytest.raises(ValueError, match=message):
        make_batch_run(G=_triangle(), columns=columns)


def test_a_run_can_be_written_under_a_given_folder(make_batch_run, tmp_path):
    run_dir = tmp_path / "old" / "TEST-A"
    stand_in = make_batch_run(run_dir=run_dir)
    assert stand_in.run_dir == run_dir
    assert stand_in.run.run_dir == run_dir
    assert stand_in.cache.parent == run_dir


def test_each_specimen_is_placed_at_its_own_roi_after_another_is_made(make_batch_run):
    first = make_batch_run("TEST-A")
    second = make_batch_run("TEST-B")
    assert first.run.placement == first.placed
    assert second.run.placement == second.placed
    assert first.placed.specimen_id == "TEST-A" and second.placed.specimen_id == "TEST-B"
