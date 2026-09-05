"""Tests for statistics module."""
import re

import pytest
import numpy as np
import networkx as nx

from haemolynx.statistics import (
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
    compute_emergence_angles_by_branch_order,
    export_branch_order_statistics_to_csv,
)


def _straight_edge(G, u, v, branch_order):
    pos_u = np.asarray(G.nodes[u]["pos"], dtype=float)
    pos_v = np.asarray(G.nodes[v]["pos"], dtype=float)
    G.add_edge(
        u,
        v,
        key=0,
        branch_order=branch_order,
        length=float(np.linalg.norm(pos_v - pos_u)),
        voxels=[tuple(pos_u), tuple(pos_v)],
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


def test_tree_asymmetry_uses_edge_length_not_an_arbitrary_spanning_tree():
    """The spanning tree reduction must follow length, not edge insertion order.

    The triangle's long edge (2-0, length 100) is inserted first. Without an
    explicit weight, every edge is treated as weight 1 and Kruskal's stable
    tie-breaking keeps whichever cycle edge was seen first -- here the long
    one -- dropping the short 1-2 edge instead. That produces a perfectly
    symmetric star at node 0 (asymmetry 0), hiding the asymmetry the two
    pendants on node 0 actually introduce once the real (short) triangle
    edges are correctly kept and the long one is dropped (asymmetry 0.2).
    """
    G = nx.Graph()
    for n in range(5):
        G.add_node(n)
    G.add_edge(2, 0, length=100.0)
    G.add_edge(0, 1, length=1.0)
    G.add_edge(1, 2, length=1.0)
    G.add_edge(0, 3, length=1.0)
    G.add_edge(0, 4, length=1.0)

    result = compute_tree_asymmetry(G)

    assert result["Tree Asymmetry Index"] == pytest.approx(0.2)


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
    assert s["BO1"]["Mean Emergence Angle (degrees)"] == pytest.approx(0.0, abs=1e-9)
    assert s["BO3"]["Mean Emergence Angle (degrees)"] == pytest.approx(0.0, abs=1e-9)
    assert s["Art2"]["Mean Emergence Angle (degrees)"] == "N/A (no unique parent junction)"
    assert s["Ven1"]["Mean Emergence Angle (degrees)"] == "N/A (no unique parent junction)"


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
    assert (
        "Branch Order,Edge Count,Mean Length (microns),Mean Tortuosity Index,"
        "Mean Emergence Angle (degrees),Notes"
    ) in text
    assert "Art1,3,12.5,1.1,N/A (no unique parent junction)," in text
    assert "Mean tortuosity is path length / straight distance." in text
    assert "Emergence angle unavailable (no unique lower-order parent junction)." in text
    assert "BO2,2,8,N/A (insufficient position data),N/A (no unique parent junction)," in text
    assert "Tortuosity unavailable (missing/insufficient node positions)." in text


def test_emergence_angle_is_deflection_from_the_parent_branch(tmp_path):
    """A collinear daughter is 0°; a perpendicular side branch is 90°."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 20.0))
    G.add_node(2, pos=(0.0, 0.0, 40.0))
    G.add_node(3, pos=(0.0, 20.0, 20.0))
    _straight_edge(G, 0, 1, "B01")
    _straight_edge(G, 1, 2, "B02")
    _straight_edge(G, 1, 3, "B03")

    angles = compute_emergence_angles_by_branch_order(G)
    assert angles["BO2"]["Mean Emergence Angle (degrees)"] == pytest.approx(0.0, abs=1e-9)
    assert angles["BO3"]["Mean Emergence Angle (degrees)"] == pytest.approx(90.0, abs=1e-9)
    assert "BO1" not in angles

    stats = compute_branch_order_statistics(G, node_positions=nx.get_node_attributes(G, "pos"))
    assert stats["BO2"]["Mean Emergence Angle (degrees)"] == pytest.approx(0.0, abs=1e-9)
    assert stats["BO3"]["Mean Emergence Angle (degrees)"] == pytest.approx(90.0, abs=1e-9)
    assert stats["BO1"]["Mean Emergence Angle (degrees)"] == "N/A (no unique parent junction)"

    out_csv = tmp_path / "emergence_branch_statistics.csv"
    export_branch_order_statistics_to_csv(stats, out_csv)
    text = out_csv.read_text(encoding="utf-8")
    assert "Mean Emergence Angle (degrees)" in text.splitlines()[0]
    assert re.search(r"^BO2,1,20,1,0,", text, flags=re.MULTILINE)
    assert re.search(r"^BO3,1,20,1,90,", text, flags=re.MULTILINE)


def test_emergence_angle_uses_local_centreline_not_node_span():
    """The parent tangent is taken from the centreline near the junction.

    Node-to-node the parent runs along +x, which would make the daughter 0°.
    The last centreline segment at the junction is not along +x, so the
    measured angle must follow that local tangent instead.
    """
    G = nx.MultiGraph()
    parent_other = (0.0, 20.0, -30.0)
    junction = (0.0, 0.0, 0.0)
    daughter = (0.0, 0.0, 20.0)
    G.add_node(0, pos=parent_other)
    G.add_node(1, pos=junction)
    G.add_node(2, pos=(0.0, 20.0, 0.0))
    G.add_node(3, pos=daughter)
    parent_voxels = [parent_other, (0.0, 20.0, -10.0), junction]
    G.add_edge(
        0,
        1,
        key=0,
        branch_order="B01",
        length=float(
            np.linalg.norm(np.subtract(parent_voxels[1], parent_voxels[0]))
            + np.linalg.norm(np.subtract(parent_voxels[2], parent_voxels[1]))
        ),
        voxels=parent_voxels,
    )
    _straight_edge(G, 1, 2, "B02")
    _straight_edge(G, 1, 3, "B03")

    incoming_parent = np.subtract(junction, (0.0, 20.0, -10.0))
    outgoing_daughter = np.subtract(daughter, junction)
    expected = np.degrees(
        np.arccos(
            np.clip(
                np.dot(incoming_parent, outgoing_daughter)
                / (
                    np.linalg.norm(incoming_parent)
                    * np.linalg.norm(outgoing_daughter)
                ),
                -1.0,
                1.0,
            )
        )
    )
    node_span_parent = np.subtract(junction, parent_other)
    node_span_angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(node_span_parent, outgoing_daughter)
                / (
                    np.linalg.norm(node_span_parent)
                    * np.linalg.norm(outgoing_daughter)
                ),
                -1.0,
                1.0,
            )
        )
    )
    assert expected != pytest.approx(node_span_angle, abs=1.0)

    angles = compute_emergence_angles_by_branch_order(G)
    assert angles["BO3"]["Mean Emergence Angle (degrees)"] == pytest.approx(
        expected, abs=1e-9
    )
    assert angles["BO3"]["Mean Emergence Angle (degrees)"] != pytest.approx(
        node_span_angle, abs=1.0
    )


def test_emergence_angle_skipped_when_parent_order_is_tied():
    """Two equal-order stems at a confluence are not a unique parent."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 10.0, 0.0))
    G.add_node(1, pos=(0.0, -10.0, 0.0))
    G.add_node(2, pos=(0.0, 0.0, 0.0))
    G.add_node(3, pos=(0.0, 0.0, 10.0))
    _straight_edge(G, 0, 2, "B02")
    _straight_edge(G, 1, 2, "B02")
    _straight_edge(G, 2, 3, "Ven1")

    angles = compute_emergence_angles_by_branch_order(G)
    assert angles == {}


def test_emergence_angle_counts_a_bridging_edge_only_once():
    """A capillary that is a daughter at both its junctions is one sample.

    node 2 and node 3 are both junctions with their own lower-order (Art1)
    parent, so the BO5 edge between them is a "daughter" from either end --
    not a strict parent -> child step, the way a real anastomosis or a
    capillary bridging two comparable-order neighbourhoods looks. Processing
    each junction independently must still count that one edge's emergence
    once, not once per end.
    """
    G = nx.MultiGraph()
    G.add_node(0, pos=(-20.0, 0.0, 0.0))
    G.add_node(5, pos=(0.0, -20.0, 0.0))
    G.add_node(2, pos=(0.0, 0.0, 0.0))
    G.add_node(3, pos=(0.0, 0.0, 20.0))
    G.add_node(1, pos=(20.0, 0.0, 20.0))
    G.add_node(6, pos=(0.0, 20.0, 20.0))
    _straight_edge(G, 0, 2, "Art1")
    _straight_edge(G, 5, 2, "BO6")
    _straight_edge(G, 2, 3, "BO5")
    _straight_edge(G, 1, 3, "Art1")
    _straight_edge(G, 6, 3, "Ven2")

    angles = compute_emergence_angles_by_branch_order(G)

    assert angles["BO5"]["Emergence Angle Sample Count"] == 1
    assert angles["BO6"]["Emergence Angle Sample Count"] == 1
    assert angles["Ven2"]["Emergence Angle Sample Count"] == 1
