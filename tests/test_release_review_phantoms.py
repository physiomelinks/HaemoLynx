"""The phantoms of known topology and the graph metrics `graph.8a` reads."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from release_review import graph_metrics, phantoms


def test_the_tree_and_mesh_truths():
    tree, mesh = phantoms.tree(), phantoms.mesh()
    assert tree.junctions() == ["j1", "j2", "j3"]
    assert tree.loops() == 0
    assert tree.junction_radius("j1") == 5.0
    assert len(mesh.junctions()) == 8
    assert all(mesh.degree(node) == 3 for node in mesh.junctions())
    assert mesh.loops() == 2
    # Every open end lies outside the box: the image cuts every vessel it leaves by.
    for phantom in (tree, mesh):
        ends = [node for node in phantom.nodes if phantom.degree(node) == 1]
        assert ends and not any(phantom.inside(np.asarray(phantom.nodes[node])) for node in ends)
        assert all(phantom.inside(np.asarray(phantom.nodes[node])) for node in phantom.junctions())


def test_a_segment_is_clipped_to_the_box():
    size = np.array([10.0, 10.0, 10.0])
    assert phantoms._clipped_length(np.array([5.0, 5.0, -5.0]), np.array([5.0, 5.0, 15.0]), size) == pytest.approx(10.0)
    assert phantoms._clipped_length(np.array([5.0, 5.0, 2.0]), np.array([5.0, 5.0, 6.0]), size) == pytest.approx(4.0)
    assert phantoms._clipped_length(np.array([20.0, 5.0, 2.0]), np.array([20.0, 5.0, 6.0]), size) == 0.0
    diagonal = phantoms._clipped_length(np.array([-10.0, -10.0, -10.0]), np.array([20.0, 20.0, 20.0]), size)
    assert diagonal == pytest.approx(10.0 * math.sqrt(3.0))


def test_the_tubes_hold_the_volume_of_their_cylinders():
    one = phantoms.Phantom("rod", (20.0, 20.0, 40.0), {"a": (10.0, 10.0, -5.0), "b": (10.0, 10.0, 45.0)},
                           (("a", "b", 4.0),))
    mask = phantoms.tubes(one, (1.0, 1.0, 1.0))
    assert mask.shape == (21, 21, 41)
    per_slice = mask.sum(axis=(0, 1))
    assert np.all(per_slice == per_slice[0])
    assert per_slice[0] == pytest.approx(math.pi * 16.0, rel=0.06)
    coarse = phantoms.tubes(one, (3.0, 1.0, 1.0))
    assert coarse.shape == (7, 21, 41)
    assert coarse.sum() * 3.0 == pytest.approx(mask.sum(), rel=0.15)


def test_rendering_is_seeded_and_the_noisy_mask_is_the_thresholded_raw():
    phantom = phantoms.tree()
    mask_a, raw_a = phantoms.render(phantom, (3.0, 1.0, 1.0), "noisy", seed=3)
    mask_b, raw_b = phantoms.render(phantom, (3.0, 1.0, 1.0), "noisy", seed=3)
    assert np.array_equal(raw_a, raw_b) and np.array_equal(mask_a, mask_b)
    assert np.array_equal(mask_a, (raw_a > 0.5).astype(np.uint8))
    clean, _ = phantoms.render(phantom, (3.0, 1.0, 1.0), "clean")
    assert np.array_equal(clean, phantoms.tubes(phantom, (3.0, 1.0, 1.0)))
    assert len(phantoms.variants()) == 8


def test_junctions_match_one_to_one_within_the_local_radius():
    truth = [(0.0, 0.0, 0.0), (0.0, 0.0, 50.0)]
    radii = [4.0, 2.0]
    built = [(0.0, 1.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.0, 53.0), (0.0, 30.0, 30.0)]
    match = graph_metrics.match_junctions(truth, radii, built)
    # Two built nodes sit in the first junction's radius: the nearer takes it,
    # the other is extra. The second junction's node is 3 um away, beyond 2.
    assert (match.true_count, match.built_count, match.matched) == (2, 4, 1)
    assert match.precision == 0.25 and match.recall == 0.5
    assert match.missed == ((0.0, 0.0, 50.0),)
    coarse = graph_metrics.match_junctions(truth, radii, built, voxel_size_zyx=(3.0, 1.0, 1.0))
    assert coarse.matched == 2  # no finer than a voxel's longest side
    empty = graph_metrics.match_junctions(truth, radii, [])
    assert empty.precision is None and empty.recall == 0.0


def test_a_built_graph_is_scored_against_the_truth():
    truth = phantoms.truth(phantoms.tree())
    graph = nx.MultiGraph()
    positions = {name: np.asarray(pos) for name, pos in phantoms.tree().nodes.items()}
    for name, pos in positions.items():
        graph.add_node(name, pos=pos)
    for u, v, _radius in phantoms.tree().vessels:
        graph.add_edge(u, v, length=phantoms._clipped_length(positions[u], positions[v], np.array([120.0] * 3)))
    score = graph_metrics.score_graph(graph, truth, (1.0, 1.0, 1.0))
    assert (score["precision"], score["recall"], score["length_error"], score["loops"]) == (1.0, 1.0, 0.0, 0)
    assert graph_metrics.meets_targets(score, 0.95, 0.05)
    graph.add_edge("j2", "j3", length=40.0)
    worse = graph_metrics.score_graph(graph, truth, (1.0, 1.0, 1.0))
    assert worse["loops"] == 1 and worse["length_error"] > 0.05
    assert not graph_metrics.meets_targets(worse, 0.95, 0.05)
