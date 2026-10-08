"""``cb_h2_error_propagation.py`` must place pressure where the pipeline does (package C).

It used to pick inlets and outlets with its own 25% band on another axis, so the calibre error
floors it reports were measured on pressure nodes no H2 flow uses. It also counted edges on a
simple Graph, which merges parallel edges. These pin the face rule and the MultiGraph count.
"""
import ast
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import cb_h2_error_propagation as module
from ImageLynx import cb_settings
from ImageLynx.graph.boundaries import select_boundary_terminal_nodes_by_face
from ImageLynx.specimens import PROCESSING_VOXEL_UM

SOURCE = Path(module.__file__)


def _graph():
    """Centre node with terminals on both faces of every axis, and a parallel pair to node 7."""
    G = nx.MultiGraph()
    mid, top = 148.0, 296.0
    G.add_node(0, pos=np.array([mid, mid, mid]))
    faces = [(0.0, mid, mid), (top, mid, mid), (mid, 0.0, mid),
             (mid, top, mid), (mid, mid, 0.0), (mid, mid, top)]
    for i, pos in enumerate(faces, start=1):
        G.add_node(i, pos=np.array(pos))
        G.add_edge(0, i)
    # An interior dead end: a terminal on no face, which neither rule should make a boundary.
    G.add_node(7, pos=np.array([mid, 60.0, 100.0]))
    G.add_node(8, pos=np.array([mid, 30.0, 100.0]))
    G.add_edge(0, 7)
    G.add_edge(0, 7)
    G.add_edge(7, 8)
    return G


def _inputs(G, skip=None):
    """What the batch-run reader hands out: the graph with calibre on each edge, and the
    ``length_um`` column as a float per edge."""
    lengths = {}
    for a, b, k, data in G.edges(keys=True, data=True):
        data["assigned_diameter_um"] = 6.0
        if (a, b, k) != skip:
            lengths[(a, b, k)] = 40.0
    return G, lengths


def test_face_boundaries_are_the_library_face_rule():
    G = _graph()
    index = {n: i for i, n in enumerate(G.nodes())}
    expected_in, expected_out = select_boundary_terminal_nodes_by_face(
        G, cb_settings.ROI_VOXELS, axis=cb_settings.BOUNDARY_AXIS,
        face_tolerance_voxels=cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS,
        voxel_size=PROCESSING_VOXEL_UM)
    inlets, outlets = module.face_boundaries(G, index)
    assert sorted(inlets.tolist()) == sorted(index[n] for n in expected_in)
    assert sorted(outlets.tolist()) == sorted(index[n] for n in expected_out)
    # On axis 1 that is node 3 (low face) and node 4 (high face), and not the interior node 8.
    assert inlets.tolist() == [index[3]] and outlets.tolist() == [index[4]]


def test_face_boundaries_default_to_the_frozen_axis_and_tolerance():
    assert module.BOUNDARY_AXIS == cb_settings.BOUNDARY_AXIS
    assert module.FACE_TOLERANCE == cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS
    assert module.ROI == cb_settings.ROI_VOXELS


def test_the_band_rule_is_gone():
    tree = ast.parse(SOURCE.read_text())
    assigned = {t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not assigned & {"AXIS", "EDGE_PERCENT", "END_PERCENT"}
    assert "boundary_nodes" not in functions


def test_edge_count_keeps_parallel_edges():
    G = _graph()
    solvable, total = module.edge_count_between_boundaries(G, {3}, {4})
    # 6 face spokes + the parallel pair + the stub: 9, where a simple Graph would say 8.
    assert total == 9 == G.number_of_edges()
    assert nx.Graph(G).number_of_edges() == 8
    assert solvable == 9


def test_network_arrays_keep_parallel_edges_as_separate_conductances():
    G = _graph()
    u, v, length, diameter, index = module.network_arrays(*_inputs(G))
    assert len(u) == 9
    pair = sorted([index[0], index[7]])
    assert sum(sorted([a, b]) == pair for a, b in zip(u.tolist(), v.tolist())) == 2


def test_a_missing_morphometry_row_raises():
    G = _graph()
    with pytest.raises(ValueError, match="no row"):
        module.network_arrays(*_inputs(G, skip=(0, 7, 1)))


def test_a_node_left_without_edges_raises():
    G, lengths = _inputs(_graph())
    # A zero length on the only edge to node 8 would leave it as a zero Laplacian row.
    lengths[(7, 8, 0)] = 0.0
    with pytest.raises(ValueError, match="singular"):
        module.network_arrays(G, lengths)


def test_load_network_reads_lengths_through_the_reader_and_refuses_a_blank_one(
        tmp_path, monkeypatch):
    """A blank length used to become NaN and the edge was quietly dropped from the solve."""
    from types import SimpleNamespace
    from ImageLynx.batch_outputs import EDGE_TABLE_NAME, BatchRun

    G, lengths = _inputs(_graph())
    rows = ["u,v,key,length_um,assigned_diameter_um"]
    rows += [f"{a},{b},{k},{'' if (a, b, k) == (0, 7, 1) else 40.0},6.0" for a, b, k in lengths]
    (tmp_path / EDGE_TABLE_NAME).write_text("\n".join(rows) + "\n")
    run = BatchRun(SimpleNamespace(specimen_id="TEST-A"), tmp_path, None, None)
    monkeypatch.setattr(run, "graph", lambda: G)
    monkeypatch.setattr(module, "open_batch_run", lambda specimen: run)
    monkeypatch.setattr(module, "get_specimen", lambda specimen_id: None)

    with pytest.raises(ValueError, match=r"TEST-A.*empty or non-finite length_um.*\(0, 7, 1\)"):
        module.load_network("TEST-A")
