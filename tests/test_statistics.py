"""Tests for statistics module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.statistics import (
    compute_basic_statistics,
    compute_tortuosity_measures,
    compute_branching_statistics,
    compute_tree_asymmetry,
    compute_fractal_dimension,
    compute_path_efficiency,
    compute_vessel_density,
    compute_comprehensive_vessel_statistics,
)


def test_compute_basic_statistics(simple_graph):
    s = compute_basic_statistics(simple_graph, False)
    assert s["Total Nodes"] == 3
    assert s["Total Edges"] == 2


def test_compute_tortuosity_measures(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_tortuosity_measures(simple_graph, pos, False)
    assert "Average Tortuosity Index" in s


def test_compute_branching_statistics(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_branching_statistics(simple_graph, pos)
    assert "Average Branching Angle (degrees)" in s


def test_compute_tree_asymmetry(simple_graph):
    s = compute_tree_asymmetry(simple_graph)
    assert "Tree Asymmetry Index" in s


def test_compute_fractal_dimension(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_fractal_dimension(simple_graph, pos)
    assert "Fractal Dimension" in s


def test_compute_path_efficiency(simple_graph):
    s = compute_path_efficiency(simple_graph, False)
    assert "Path Efficiency" in s


def test_compute_vessel_density(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_vessel_density(
        simple_graph, pos, (1, 1, 1), (10, 10, 10), False
    )
    assert "Total Vessel Length (microns)" in s


def test_compute_comprehensive_vessel_statistics(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10)
    )
    assert "Total Nodes" in s
    assert "Fractal Dimension" in s
