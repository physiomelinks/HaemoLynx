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
    compute_daughter_daughter_angles,
    compute_emergence_angles_by_branch_order,
    compute_intercapillary_distance,
    compute_murray_law_compliance,
    compute_network_robustness,
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
    s = compute_branching_statistics(simple_graph)
    assert s == {"Number of Branching Points": 0}


def test_compute_branching_statistics_counts_junctions():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6)])
    s = compute_branching_statistics(G)
    assert s == {"Number of Branching Points": 2}


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


def test_tree_asymmetry_on_an_empty_graph_does_not_raise():
    """max() over G.nodes() used to raise ValueError on an empty graph."""
    result = compute_tree_asymmetry(nx.Graph())

    assert result["Tree Asymmetry Index"] == "N/A (empty graph)"


def test_tree_asymmetry_handles_a_chain_deeper_than_the_recursion_limit():
    """A long, sparsely-branching stretch of vessel is a real shape, not a
    pathological one, once a network is reduced to a spanning tree -- the
    old node-by-node recursive walk would raise RecursionError on it.

    Root selection (max degree, ties broken by node order) lands on node 1,
    the first degree-2 node in a 0..4999 chain, splitting it into a
    1-node arm and a 4998-node arm: asymmetry (4998 - 1) / 5000.
    """
    G = nx.path_graph(5000)

    result = compute_tree_asymmetry(G)

    assert result["Tree Asymmetry Index"] == pytest.approx(4997 / 5000)


def test_compute_fractal_dimension(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_fractal_dimension(simple_graph, pos)
    assert "Fractal Dimension (Node Positions)" in s
    assert "Fractal Dimension (Centreline)" in s


def test_fractal_dimension_centreline_sees_the_polyline_the_node_only_one_misses():
    """The two estimates are deliberately different lenses on the network.

    Both graphs share the exact same two node positions, so the node-only
    box count cannot tell them apart -- it never looks at the polyline in
    between. Only the graph whose single edge actually zigzags between
    those two points should read differently once every point along its
    real centreline is counted instead of just its two ends.
    """
    pos = {0: (0.0, 0.0, 0.0), 1: (10.0, 0.0, 0.0)}

    straight = nx.Graph()
    straight.add_node(0, pos=pos[0])
    straight.add_node(1, pos=pos[1])
    straight.add_edge(0, 1, voxels=[pos[0], pos[1]])

    zigzag = nx.Graph()
    zigzag.add_node(0, pos=pos[0])
    zigzag.add_node(1, pos=pos[1])
    zigzag.add_edge(
        0, 1, voxels=[(float(x), 5.0 if x % 2 else 0.0, 0.0) for x in range(11)]
    )

    straight_result = compute_fractal_dimension(straight, pos)
    zigzag_result = compute_fractal_dimension(zigzag, pos)

    assert straight_result["Fractal Dimension (Node Positions)"] == pytest.approx(
        zigzag_result["Fractal Dimension (Node Positions)"], abs=1e-9
    )
    assert straight_result["Fractal Dimension (Centreline)"] != pytest.approx(
        zigzag_result["Fractal Dimension (Centreline)"], abs=1e-9
    )


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
    assert "Fractal Dimension (Node Positions)" in s
    assert "Fractal Dimension (Centreline)" in s
    assert "Mean Murray Ratio" in s
    assert "Mean Daughter-Daughter Angle (degrees)" in s
    assert "Mean Intercapillary Distance (microns)" in s
    assert "Bridge Edge Count" in s
    assert "Articulation Point Count" in s


def test_murray_law_compliance_matches_the_cube_law_by_hand():
    """8**3 + 8**3 = 1024 vs a parent diameter of 10 (1000): ratio 1.024."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    G.add_node(2, pos=(0.0, 5.0, 20.0))
    G.add_node(3, pos=(0.0, -5.0, 20.0))
    G.add_edge(0, 1, key=0, branch_order="Art1", diameter_um=10.0)
    G.add_edge(1, 2, key=0, branch_order="BO1", diameter_um=8.0)
    G.add_edge(1, 3, key=0, branch_order="BO2", diameter_um=8.0)

    result = compute_murray_law_compliance(G)

    assert result["Murray Ratio Sample Count"] == 1
    assert result["Mean Murray Ratio"] == pytest.approx(1.024)
    assert result["Murray Law Exponent"] == 3.0


def test_murray_law_skips_a_junction_missing_any_one_diameter():
    """A ratio built from a partial set of daughters is not comparable to
    one built from all of them, so the whole junction is skipped."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    G.add_node(2, pos=(0.0, 5.0, 20.0))
    G.add_node(3, pos=(0.0, -5.0, 20.0))
    G.add_edge(0, 1, key=0, branch_order="Art1", diameter_um=10.0)
    G.add_edge(1, 2, key=0, branch_order="BO1", diameter_um=8.0)
    G.add_edge(1, 3, key=0, branch_order="BO2")  # no diameter_um

    result = compute_murray_law_compliance(G)

    assert result["Murray Ratio Sample Count"] == 0
    assert result["Mean Murray Ratio"] == "N/A (no junction had diameters on every branch)"


def test_murray_law_averages_across_several_junctions():
    parent_d = 8.0
    # Two equal daughters give ratio = 2 * daughter**3 / parent**3.
    daughter_for_ratio_1 = (parent_d**3 / 2) ** (1 / 3)  # ratio exactly 1.0
    daughter_for_ratio_2 = parent_d  # 2 * parent**3 / parent**3 == 2.0

    G = nx.MultiGraph()
    for node in range(7):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    # Junction A: exact compliance (ratio 1.0).
    G.add_edge(0, 1, key=0, branch_order="Art1", diameter_um=parent_d)
    G.add_edge(1, 2, key=0, branch_order="BO1", diameter_um=daughter_for_ratio_1)
    G.add_edge(1, 3, key=0, branch_order="BO2", diameter_um=daughter_for_ratio_1)
    # Junction B: double compliance (ratio 2.0).
    G.add_edge(3, 4, key=0, branch_order="Art2", diameter_um=parent_d)
    G.add_edge(4, 5, key=0, branch_order="BO3", diameter_um=daughter_for_ratio_2)
    G.add_edge(4, 6, key=0, branch_order="BO4", diameter_um=daughter_for_ratio_2)

    result = compute_murray_law_compliance(G)

    assert result["Murray Ratio Sample Count"] == 2
    assert result["Mean Murray Ratio"] == pytest.approx(1.5, rel=1e-6)


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
    assert s["BO1"]["Mean Pressure Drop (Pa)"] == "N/A (no flow solved)"
    assert s["BO1"]["Total Pressure Drop (Pa)"] == "N/A (no flow solved)"
    assert s["BO1"]["Mean Diameter (microns)"] == "N/A (no diameter assigned)"
    assert s["BO1"]["Diameter Coefficient of Variation"] == "N/A (no diameter assigned)"
    assert "_diameter_sum_sq" not in s["BO1"]


def test_branch_order_pressure_drop_aggregates_magnitude_not_signed_value():
    """pressure_drop's sign is an artefact of an edge's arbitrary (u, v)
    storage order, not physically meaningful -- aggregation must use
    magnitude, and mean/total must reflect only the edges flow was solved
    for."""
    G = nx.MultiGraph()
    for node in range(4):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    G.add_edge(0, 1, key=0, branch_order="B01", length=1.0, pressure_drop=10.0)
    G.add_edge(1, 2, key=0, branch_order="B01", length=1.0, pressure_drop=-20.0)
    G.add_edge(2, 3, key=0, branch_order="B01", length=1.0)  # flow not solved

    s = compute_branch_order_statistics(G, node_positions=nx.get_node_attributes(G, "pos"))

    assert s["BO1"]["Pressure Drop Sample Count"] == 2
    assert s["BO1"]["Mean Pressure Drop (Pa)"] == pytest.approx(15.0)
    assert s["BO1"]["Total Pressure Drop (Pa)"] == pytest.approx(30.0)


def test_branch_order_diameter_mean_and_coefficient_of_variation():
    G = nx.MultiGraph()
    for node in range(3):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    G.add_edge(0, 1, key=0, branch_order="B01", length=1.0, diameter_um=4.0)
    G.add_edge(1, 2, key=0, branch_order="B01", length=1.0, diameter_um=6.0)

    s = compute_branch_order_statistics(G, node_positions=nx.get_node_attributes(G, "pos"))

    assert s["BO1"]["Diameter Sample Count"] == 2
    assert s["BO1"]["Mean Diameter (microns)"] == pytest.approx(5.0)
    # population std of [4, 6] is 1.0, so CoV = 1.0 / 5.0
    assert s["BO1"]["Diameter Coefficient of Variation"] == pytest.approx(0.2)
    assert "_diameter_sum_sq" not in s["BO1"]


def test_branch_order_diameter_ignores_non_positive_or_missing_values():
    G = nx.MultiGraph()
    for node in range(3):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    G.add_edge(0, 1, key=0, branch_order="B01", length=1.0, diameter_um=0.0)
    G.add_edge(1, 2, key=0, branch_order="B01", length=1.0)  # no diameter_um at all

    s = compute_branch_order_statistics(G, node_positions=nx.get_node_attributes(G, "pos"))

    assert s["BO1"]["Diameter Sample Count"] == 0
    assert s["BO1"]["Mean Diameter (microns)"] == "N/A (no diameter assigned)"
    assert s["BO1"]["Diameter Coefficient of Variation"] == "N/A (no diameter assigned)"


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
        "Mean Emergence Angle (degrees),Mean Pressure Drop (Pa),"
        "Total Pressure Drop (Pa),Mean Diameter (microns),"
        "Diameter Coefficient of Variation,Notes"
    ) in text
    assert (
        "Art1,3,12.5,1.1,N/A (no unique parent junction),"
        "N/A (no flow solved),N/A (no flow solved),"
        "N/A (no diameter assigned),N/A (no diameter assigned),"
    ) in text
    assert "Mean tortuosity is path length / straight distance." in text
    assert "Emergence angle unavailable (no unique lower-order parent junction)." in text
    assert (
        "BO2,2,8,N/A (insufficient position data),N/A (no unique parent junction),"
        "N/A (no flow solved),N/A (no flow solved),"
        "N/A (no diameter assigned),N/A (no diameter assigned),"
    ) in text
    assert "Tortuosity unavailable (missing/insufficient node positions)." in text
    assert "Pressure drop unavailable (flow not solved)." in text
    assert "Diameter unavailable (no diameter assigned)." in text


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


def test_daughter_daughter_angle_is_between_the_two_daughters_not_the_parent():
    """The other half of bifurcation morphometry: BO2 continues the parent
    straight ahead, BO3 leaves at a right angle, so the angle *between*
    BO2 and BO3 themselves is 90 degrees -- for a different reason than
    BO3's emergence angle happens to also be 90 degrees in this fixture."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 20.0))
    G.add_node(2, pos=(0.0, 0.0, 40.0))
    G.add_node(3, pos=(0.0, 20.0, 20.0))
    _straight_edge(G, 0, 1, "B01")
    _straight_edge(G, 1, 2, "B02")
    _straight_edge(G, 1, 3, "B03")

    result = compute_daughter_daughter_angles(G)

    assert result["Daughter-Daughter Angle Sample Count"] == 1
    assert result["Mean Daughter-Daughter Angle (degrees)"] == pytest.approx(90.0, abs=1e-9)


def test_daughter_daughter_angle_needs_at_least_two_daughters():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 20.0))
    _straight_edge(G, 0, 1, "B01")

    result = compute_daughter_daughter_angles(G)

    assert result["Daughter-Daughter Angle Sample Count"] == 0
    assert result["Mean Daughter-Daughter Angle (degrees)"] == (
        "N/A (no bifurcation with two measurable daughters)"
    )


def test_daughter_daughter_angle_counts_every_pair_at_a_trifurcation():
    """Three daughters at one junction have C(3, 2) = 3 pairs, not just one."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 20.0))
    G.add_node(2, pos=(0.0, 0.0, 40.0))
    G.add_node(3, pos=(0.0, 20.0, 20.0))
    G.add_node(4, pos=(20.0, 0.0, 20.0))
    _straight_edge(G, 0, 1, "Art1")
    _straight_edge(G, 1, 2, "BO1")
    _straight_edge(G, 1, 3, "BO2")
    _straight_edge(G, 1, 4, "BO3")

    result = compute_daughter_daughter_angles(G)

    assert result["Daughter-Daughter Angle Sample Count"] == 3


def test_intercapillary_distance_between_two_isolated_parallel_edges():
    """Two edges sharing no node, offset by exactly 100 microns in y: the
    nearest point on the other edge is 100 microns away for each of them."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    G.add_node(2, pos=(0.0, 100.0, 0.0))
    G.add_node(3, pos=(0.0, 100.0, 10.0))
    _straight_edge(G, 0, 1, "BO1")
    _straight_edge(G, 2, 3, "BO2")

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Sample Count"] == 2
    assert result["Mean Intercapillary Distance (microns)"] == pytest.approx(100.0)
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(100.0)


def test_intercapillary_distance_excludes_edges_sharing_a_node():
    """A and B meet at node 1, so their shared point (distance 0) must not
    count as A's or B's nearest neighbour -- only the unrelated edge C can.
    A naive nearest-point search that does not exclude adjacent edges would
    report 0.0 for both A and B instead of 50.0."""
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    G.add_node(2, pos=(0.0, 0.0, 20.0))
    G.add_node(3, pos=(0.0, 50.0, 0.0))
    G.add_node(4, pos=(0.0, 50.0, 10.0))
    _straight_edge(G, 0, 1, "BO1")  # A
    _straight_edge(G, 1, 2, "BO2")  # B, adjacent to A via node 1
    _straight_edge(G, 3, 4, "BO3")  # C, shares no node with A or B

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Sample Count"] == 3
    assert result["Mean Intercapillary Distance (microns)"] == pytest.approx(50.0)
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(50.0)


def test_intercapillary_distance_needs_at_least_two_edges():
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    _straight_edge(G, 0, 1, "BO1")

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Sample Count"] == 0
    assert result["Mean Intercapillary Distance (microns)"] == "N/A (fewer than two edges)"
    assert result["Median Intercapillary Distance (microns)"] == "N/A (fewer than two edges)"


def test_network_robustness_finds_the_single_bridge_joining_two_triangles():
    """Two triangles joined by one edge: that edge is the sole connection,
    so it is the only bridge, and both of its endpoints are articulation
    points (removing either splits the network into two pieces) -- the
    other four nodes are each protected by their triangle's second path."""
    G = nx.MultiGraph()
    G.add_edges_from(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    )

    result = compute_network_robustness(G)

    assert result["Bridge Edge Count"] == 1
    assert result["Bridge Edge Fraction"] == pytest.approx(1 / 7)
    assert result["Articulation Point Count"] == 2
    assert result["Articulation Point Fraction"] == pytest.approx(2 / 6)


def test_network_robustness_a_parallel_edge_is_not_a_bridge():
    """Two physically distinct vessels between the same pair of junctions:
    occluding either one leaves the other, so neither is a bridge -- a
    naive collapse-to-simple-graph check would wrongly flag one."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0)
    G.add_edge(0, 1, key=1)

    result = compute_network_robustness(G)

    assert result["Bridge Edge Count"] == 0
    assert result["Bridge Edge Fraction"] == pytest.approx(0.0)
    assert result["Articulation Point Count"] == 0


def test_network_robustness_a_single_edge_is_its_own_bridge():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0)

    result = compute_network_robustness(G)

    assert result["Bridge Edge Count"] == 1
    assert result["Bridge Edge Fraction"] == pytest.approx(1.0)
    assert result["Articulation Point Count"] == 0


def test_network_robustness_handles_an_empty_graph():
    G = nx.MultiGraph()

    result = compute_network_robustness(G)

    assert result["Bridge Edge Count"] == 0
    assert result["Bridge Edge Fraction"] == "N/A (no edges)"
    assert result["Articulation Point Count"] == 0
    assert result["Articulation Point Fraction"] == "N/A (no nodes)"


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
