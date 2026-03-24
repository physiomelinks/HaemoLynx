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
    export_statistics_to_csv,
    compute_branch_order_statistics,
    export_branch_order_statistics_to_csv,
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


def test_export_statistics_to_csv(tmp_path):
    stats = {
        "Total Nodes": 5,
        "Average Edge Length (microns)": 10.0,
        "Path Efficiency Pair Coverage": 0.5,
        "Statistics Mode": "fast",
        "nested": {"Community Count": 2},
    }
    output_csv = tmp_path / "example_statistics.csv"
    path = export_statistics_to_csv(stats, output_csv)

    assert path == output_csv
    assert output_csv.exists()
    text = output_csv.read_text(encoding="utf-8")
    assert "Section,Metric,Value,Unit,Notes" in text
    assert "Average Edge Length,10,microns" in text
    assert "Path Efficiency Pair Coverage,50.00%,,Fraction of all node-pairs included." in text
    assert "nested,Community Count,2,," in text


def test_compute_branch_order_statistics_sorted_and_aggregated():
    G = nx.MultiGraph()
    G.add_node(1, pos=(0.0, 0.0, 0.0))
    G.add_node(2, pos=(1.0, 0.0, 0.0))
    G.add_node(3, pos=(2.0, 0.0, 0.0))
    G.add_node(4, pos=(3.0, 0.0, 0.0))

    G.add_edge(1, 2, key=0, branch_order="Art2", length=2.0)
    G.add_edge(2, 3, key=0, branch_order="B01", length=2.0)
    G.add_edge(3, 4, key=0, branch_order="Ven1", length=2.0)
    G.add_edge(2, 4, key=0, branch_order="BO3", length=4.0)

    pos = nx.get_node_attributes(G, "pos")
    s = compute_branch_order_statistics(G, node_positions=pos)
    assert list(s.keys()) == ["Art2", "BO1", "BO3", "Ven1"]
    assert s["BO1"]["Edge Count"] == 1
    assert s["BO1"]["Mean Length (microns)"] == 2.0
    assert s["Art2"]["Mean Tortuosity Index"] == 2.0


def test_export_branch_order_statistics_to_csv(tmp_path):
    branch_stats = {
        "Art1": {
            "Branch Order": "Art1",
            "Edge Count": 3,
            "Mean Length (microns)": 12.5,
            "Mean Tortuosity Index": 1.1,
            "Tortuosity Sample Count": 3,
        },
        "BO2": {
            "Branch Order": "BO2",
            "Edge Count": 2,
            "Mean Length (microns)": 8.0,
            "Mean Tortuosity Index": "N/A (insufficient position data)",
            "Tortuosity Sample Count": 0,
        },
    }
    out_csv = tmp_path / "sample_branch_statistics.csv"
    out = export_branch_order_statistics_to_csv(branch_stats, out_csv)

    assert out == out_csv
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "Branch Order,Edge Count,Mean Length (microns),Mean Tortuosity Index,Notes" in text
    assert "Art1,3,12.5,1.1,Mean tortuosity is path length / straight distance." in text
    assert "BO2,2,8,N/A (insufficient position data),Tortuosity unavailable (missing/insufficient node positions)." in text
