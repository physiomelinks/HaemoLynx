"""The shared stand-in batch run (``make_batch_run`` in conftest.py).

Tests that need a batch run take it from the factory, which writes real files and opens them
through ``open_batch_run``. These tests pin what the factory hands out, so a driver test built
on it can trust the run it gets.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.batch_outputs import BatchRun


def _triangle():
    G = nx.MultiGraph()
    for node, x in enumerate((0.0, 5.0, 10.0)):
        G.add_node(node, pos=np.array([x, 0.0, 0.0]))
    G.add_edge(0, 1, key=0)
    G.add_edge(1, 2, key=0)
    G.add_edge(1, 2, key=1)
    return G


def test_the_default_run_is_the_real_reader_on_the_four_edge_run(make_batch_run):
    stand_in = make_batch_run()
    run = stand_in.run
    assert isinstance(run, BatchRun)
    assert list(run.edge_table()) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
    assert run.numeric_column("length_um")[(1, 2, 0)] == 20.0
    assert run.placement == stand_in.placed
    assert run.skeleton().shape == run.vessel_mask().shape == run.placement.size_zyx


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


def test_a_graph_edge_with_no_value_in_a_column_is_refused_not_filled(make_batch_run):
    lengths = {(0, 1, 0): 30.0, (1, 2, 0): 40.0}      # (1, 2, 1) has none
    diameters = {(0, 1, 0): 7.0, (1, 2, 0): 5.5, (1, 2, 1): 4.0}
    with pytest.raises(ValueError, match=r"length_um.*\(1, 2, 1\)"):
        make_batch_run(G=_triangle(), columns={"length_um": lengths,
                                               "assigned_diameter_um": diameters})


def test_a_column_value_for_an_edge_the_graph_lacks_is_refused_not_ignored(make_batch_run):
    G = _triangle()
    lengths = {(u, v, k): 10.0 for u, v, k in G.edges(keys=True)}
    diameters = {**lengths, (7, 8, 0): 5.0}
    with pytest.raises(ValueError, match=r"assigned_diameter_um.*not in the graph.*\(7, 8, 0\)"):
        make_batch_run(G=G, columns={"length_um": lengths, "assigned_diameter_um": diameters})
