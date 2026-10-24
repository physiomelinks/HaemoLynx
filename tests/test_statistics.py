"""Tests for statistics module."""
import math
import re

import pytest
import numpy as np
import networkx as nx

from haemolynx.statistics import (
    STATISTIC_MEASURES,
    ZERO_RESISTANCE_BRIDGES,
    compute_basic_statistics,
    compute_tortuosity_measures,
    compute_branching_statistics,
    compute_tree_asymmetry,
    compute_fractal_dimension,
    compute_path_efficiency,
    compute_vessel_density,
    compute_comprehensive_vessel_statistics,
    compute_vessel_statistics_by_measure,
    export_statistics_to_csv,
    flatten_statistics_sections,
    compute_branch_order_statistics,
    compute_daughter_daughter_angles,
    compute_emergence_angles_by_branch_order,
    compute_intercapillary_distance,
    compute_murray_law_compliance,
    compute_network_robustness,
    compute_cyclomatic_number,
    compute_degree_assortativity,
    compute_rich_club_coefficient,
    compute_k_core_structure,
    export_branch_order_statistics_to_csv,
)
from haemolynx.statistics.shape import box_counting_fractal_dimension


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


def test_box_counting_fractal_dimension_degenerate_cloud_does_not_raise():
    """Regression: an all-coincident point cloud (zero spatial extent) used
    to crash with `LinAlgError: SVD did not converge` -- a zero range feeds
    `log10(0) = -inf` into the log-spaced box sizes -- instead of degrading
    like every other degenerate-input case here already does."""
    from haemolynx.statistics.shape import _box_counting_fractal_dimension

    points = np.zeros((5, 3))
    assert _box_counting_fractal_dimension(points) == 0.0


def test_compute_fractal_dimension_with_coincident_node_positions_does_not_raise():
    """Same regression through the public entry point: every node at the
    same physical position (a degenerate but real possibility -- a graph
    edit or a malformed skeleton merge could leave duplicate positions)."""
    G = nx.Graph()
    G.add_edge(0, 1, length=1.0, voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    pos = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}

    result = compute_fractal_dimension(G, pos)

    assert result["Fractal Dimension (Node Positions)"] == 0.0
    assert result["Fractal Dimension (Centreline)"] == 0.0


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


def _unit_lattice(*sides: int) -> np.ndarray:
    """Points 1 um apart filling a box of *sides* (z, y, x), a side of 1 flat."""
    axes = [np.arange(float(n)) for n in sides]
    return np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])


def _sierpinski_triangle(depth: int) -> np.ndarray:
    """The corners of a depth-*depth* Sierpinski triangle, 1 um apart, in the
    z = 0 plane: dimension log 3 / log 2."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    points = np.zeros((1, 2))
    for _ in range(depth):
        points = np.concatenate([(points + corner) / 2.0 for corner in corners])
    points *= 2**depth
    return np.column_stack([np.zeros(len(points)), points])


@pytest.mark.parametrize(
    ("shape", "points", "expected", "tolerance"),
    [
        ("line", _unit_lattice(1, 1, 400), 1.0, 0.05),
        ("plane", _unit_lattice(1, 80, 80), 2.0, 0.15),
        # A solid this small reads low: boxes at its faces are part empty.
        ("solid", _unit_lattice(40, 40, 40), 3.0, 0.3),
        ("sierpinski", _sierpinski_triangle(8), np.log(3.0) / np.log(2.0), 0.05),
    ],
)
def test_box_counting_fractal_dimension_reads_known_dimensions(
    shape, points, expected, tolerance
):
    """Known answers, not just keys: a line is 1, a plane 2, a solid 3 and a
    Sierpinski triangle log 3 / log 2, each fitted cleanly. The bands do not
    overlap, so a dimension read off the wrong slope cannot pass as its
    neighbour."""
    fit = box_counting_fractal_dimension(points)

    assert fit["dimension"] == pytest.approx(expected, abs=tolerance), shape
    assert fit["r_squared"] > 0.98
    assert fit["box_min_um"] == pytest.approx(2.0)
    assert fit["sizes"] >= 10


def test_box_counting_fractal_dimension_needs_no_axis_alignment():
    """A line along the cube's diagonal reads as one along an axis."""
    t = np.arange(400.0)
    diagonal = np.column_stack([t, t, t]) / np.sqrt(3.0)

    assert box_counting_fractal_dimension(diagonal)["dimension"] == pytest.approx(
        1.0, abs=0.05
    )


def test_a_straight_vessels_centreline_has_dimension_one():
    """Through the statistics entry point: one straight 300 um vessel, its
    centreline resampled every micron, is one-dimensional, and the report
    gives the fit and its box sizes beside it."""
    ends = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 300.0)}
    G = nx.Graph()
    for node, pos in ends.items():
        G.add_node(node, pos=pos)
    G.add_edge(0, 1, voxels=[ends[0], ends[1]])

    result = compute_fractal_dimension(G, ends)

    assert result["Fractal Dimension (Centreline)"] == pytest.approx(1.0, abs=0.05)
    assert result["Fractal Dimension (Centreline) R^2"] > 0.98
    assert result["Fractal Dimension (Centreline) Box Sizes (microns)"].startswith("2-")
    # Two node positions have no scaling range to fit.
    assert result["Fractal Dimension (Node Positions) R^2"] == (
        "N/A (no scaling range to fit)"
    )


def _lines_graph(lines) -> nx.Graph:
    G = nx.Graph()
    for i, (start, end) in enumerate(lines):
        G.add_node(2 * i, pos=start)
        G.add_node(2 * i + 1, pos=end)
        G.add_edge(2 * i, 2 * i + 1, voxels=[start, end])
    return G


def test_a_straight_vessel_drawn_by_its_two_ends_has_dimension_one():
    """Regression: the box count took the centreline's own vertices, so a
    straight vessel stored as its two ends was two points -- dimension 0 --
    and a fixed range of box sizes was fitted whatever the counts did."""
    G = _lines_graph([((0.0, 0.0, 0.0), (0.0, 0.0, 200.0))])

    result = compute_fractal_dimension(G, nx.get_node_attributes(G, "pos"))

    assert result["Fractal Dimension (Centreline)"] == pytest.approx(1.0, abs=0.05)
    assert result["Fractal Dimension (Centreline) R^2"] > 0.99


def test_a_plane_filling_mesh_has_dimension_two_above_its_spacing():
    """Parallel vessels every 2 um across a 100 um square fill the plane at
    every box size above that spacing."""
    lines = [((0.0, float(y), 0.0), (0.0, float(y), 100.0)) for y in range(0, 101, 2)]
    G = _lines_graph(lines)

    result = compute_fractal_dimension(G, nx.get_node_attributes(G, "pos"))

    assert result["Fractal Dimension (Centreline)"] == pytest.approx(2.0, abs=0.15)
    low, high = (float(v) for v in result["Fractal Dimension (Centreline) Box Sizes (microns)"].split("-"))
    assert low >= 2.0 and high <= 50.0


def test_box_counts_are_the_fewest_over_grid_offsets():
    """Two points one box apart straddle a grid line at one offset and share
    a box at another; the count is the fewest."""
    from haemolynx.statistics.shape import _occupied_boxes

    points = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.9]])
    assert _occupied_boxes(points, 1.0) == 1


def test_compute_path_efficiency(simple_graph):
    s = compute_path_efficiency(simple_graph, False)
    assert "Path Efficiency" in s


def _brute_force_mean_path_length(G):
    simple = nx.Graph()
    simple.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        w = d.get("length", 1)
        if not simple.has_edge(u, v) or w < simple[u][v]["length"]:
            simple.add_edge(u, v, length=w)
    nodes = list(simple)
    lengths = [
        nx.shortest_path_length(simple, a, b, weight="length")
        for i, a in enumerate(nodes) for b in nodes[i + 1:]
    ]
    return float(np.mean(lengths)), len(lengths)


@pytest.mark.parametrize("batch", [256, 3])
def test_exact_path_efficiency_matches_a_search_per_pair(monkeypatch, batch):
    """full mode searched once per node *pair* -- hours of Export on a real
    network. One search per node gives the same mean: parallel vessels take
    the lighter, a zero-length vessel is still a vessel, a missing length
    counts 1, a self-loop changes nothing; also across source batches."""
    from haemolynx.statistics import shape

    monkeypatch.setattr(shape, "_ALL_PAIRS_SOURCE_BATCH", batch)
    rng = np.random.default_rng(3)
    G = nx.MultiGraph()
    for u in range(11):  # a ring with chords, so paths have choices
        G.add_edge(u, (u + 1) % 11, length=float(rng.uniform(1, 9)))
    G.add_edge(0, 5, length=2.0)
    G.add_edge(0, 5, length=30.0)  # parallel, heavier: ignored
    G.add_edge(3, 8, length=0.0)  # collapsed junction
    G.add_edge(2, 9)  # no length
    G.add_edge(4, 4, length=1.0)  # self-loop
    expected, pairs = _brute_force_mean_path_length(G)

    result = compute_path_efficiency(G, True, max_pairs=None)

    assert result["Average Shortest Path Length (microns)"] == pytest.approx(expected, rel=1e-12)
    assert result["Path Efficiency"] == pytest.approx(1 / expected, rel=1e-12)
    assert result["Path Efficiency Pair Sample Size"] == pairs == 55
    assert result["Path Efficiency Pair Coverage"] == 1.0
    simple = nx.Graph(G)
    assert compute_path_efficiency(simple, False, max_pairs=None)[
        "Average Shortest Path Length (microns)"
    ] == pytest.approx(_brute_force_mean_path_length(simple)[0], rel=1e-12)


@pytest.mark.parametrize("batch", [256, 2])
def test_sampled_path_efficiency_matches_a_search_per_sampled_pair(monkeypatch, batch):
    """fast mode's sampled pairs used one Dijkstra each (29 s on a 2,203-node
    network, slower than the exact mean); they are now read off one search
    per distinct source. Same pairs (same seed), same answer."""
    from haemolynx.statistics import shape

    monkeypatch.setattr(shape, "_ALL_PAIRS_SOURCE_BATCH", batch)
    rng = np.random.default_rng(5)
    G = nx.MultiGraph()
    for u in range(14):
        G.add_edge(u, (u + 1) % 14, length=float(rng.uniform(1, 9)))
    G.add_edge(2, 9, length=1.5)
    G.add_edge(2, 9, length=40.0)
    G.add_edge(5, 12, length=0.0)
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())  # the node order the sample indexes into
    for u, v, d in G.edges(data=True):
        if not simple.has_edge(u, v) or d["length"] < simple[u][v]["length"]:
            simple.add_edge(u, v, length=d["length"])
    nodes = list(simple.nodes())
    sampler = np.random.default_rng(shape.REPRODUCIBILITY_SEED)
    pairs = set()
    while len(pairs) < 30:
        i, j = int(sampler.integers(0, len(nodes))), int(sampler.integers(0, len(nodes)))
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    expected = np.mean([
        nx.shortest_path_length(simple, nodes[i], nodes[j], weight="length") for i, j in pairs
    ])

    result = compute_path_efficiency(G, True, max_pairs=30)

    assert result["Average Shortest Path Length (microns)"] == pytest.approx(expected, rel=1e-12)
    assert result["Path Efficiency Pair Sample Size"] == 30
    assert result["Path Efficiency Pair Coverage"] == pytest.approx(30 / 91)


def _critical_points_by_removal(G, inlets, outlets):
    """What robustness used to do: copy the graph once per point, remove it."""
    count = 0
    for point in nx.articulation_points(G):
        H = G.copy()
        H.remove_node(point)
        if any(
            ((inlets & c) and (outlets - c)) or ((outlets & c) and (inlets - c))
            for c in nx.connected_components(H)
        ):
            count += 1
    return count


@pytest.mark.parametrize("seed", range(60))
def test_critical_articulation_points_match_removing_each_one(seed):
    """One depth-first pass gives what a graph copy per articulation point
    gave: sparse trees and meshes, several components, parallel vessels,
    self-loops, terminals on articulation points, terminals not in the graph."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    for _ in range(int(rng.integers(n - 2, 2 * n))):
        u, v = (int(x) for x in rng.integers(0, n, size=2))
        G.add_edge(u, v, length=1.0)
    if seed % 3 == 0:  # a second, separate piece
        G.add_edges_from([(100, 101), (101, 102), (102, 100), (102, 103)])
    nodes = list(G.nodes())
    inlets = {nodes[int(i)] for i in rng.integers(0, len(nodes), size=int(rng.integers(1, 4)))}
    outlets = {nodes[int(i)] for i in rng.integers(0, len(nodes), size=int(rng.integers(1, 4)))}
    if seed % 5 == 0:
        outlets.add("not in the graph")

    result = compute_network_robustness(G, inlet_nodes=inlets, outlet_nodes=outlets)

    assert result["Critical Articulation Point Count"] == _critical_points_by_removal(
        G, inlets, outlets
    )


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
    assert "Cyclomatic Number" in s
    assert "Degree Assortativity" in s
    assert "Rich Club Coefficient" in s
    assert "Max Core Number" in s
    assert "Flow Hierarchy" in s


def test_enabled_measures_none_matches_every_measure_running(simple_graph):
    """The default (no enabled_measures argument) must be unchanged from
    before this parameter existed: every measure runs, keyed identically."""
    pos = nx.get_node_attributes(simple_graph, "pos")
    default = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10)
    )
    explicit_all = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
        enabled_measures=frozenset(STATISTIC_MEASURES),
    )
    assert default == explicit_all


def test_disabling_a_measure_removes_exactly_its_own_keys(simple_graph):
    """Murray's law is a clean, isolated measure to switch off: its own two
    keys disappear and nothing else does."""
    pos = nx.get_node_attributes(simple_graph, "pos")
    full = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10)
    )
    without_murray = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
        enabled_measures=frozenset(STATISTIC_MEASURES) - {"murray_law"},
    )
    assert "Mean Murray Ratio" in full
    assert "Mean Murray Ratio" not in without_murray
    assert {**without_murray, **compute_murray_law_compliance(simple_graph)} == full


def test_each_measure_that_ran_is_a_section_of_its_own_rows(simple_graph):
    """What the panel's Statistics tab draws a button for: one section per
    measure asked for, in the order the measures run, each holding exactly
    what that measure computes."""
    pos = nx.get_node_attributes(simple_graph, "pos")
    sections = compute_vessel_statistics_by_measure(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
        enabled_measures=frozenset({"cyclomatic_number", "basic", "murray_law"}),
    )

    assert list(sections) == ["basic", "murray_law", "cyclomatic_number"]
    assert sections["basic"] == compute_basic_statistics(simple_graph, False)
    assert sections["murray_law"] == compute_murray_law_compliance(simple_graph)
    assert sections["cyclomatic_number"] == compute_cyclomatic_number(simple_graph)


@pytest.mark.parametrize("mode", ["fast", "full"])
def test_the_flat_report_is_every_section_in_turn_and_no_row_is_in_two(simple_graph, mode):
    pos = nx.get_node_attributes(simple_graph, "pos")
    arguments = dict(node_positions=pos, image_dimensions=(10, 10, 10), statistics_mode=mode)
    sections = compute_vessel_statistics_by_measure(simple_graph, **arguments)
    flat = compute_comprehensive_vessel_statistics(simple_graph, **arguments)

    assert list(sections) == list(STATISTIC_MEASURES)
    rows = [name for section in sections.values() for name in section]
    assert len(rows) == len(set(rows)), "a row reported by two measures"
    assert list(flat) == rows + ["Statistics Mode"]
    assert flat == flatten_statistics_sections(sections, mode)
    assert flat["Statistics Mode"] == mode
    if mode == "full":
        assert list(sections["community"]) == ["communities"]


def test_zero_resistance_bridges_are_a_section_where_the_csv_always_had_them():
    """After the last measure that runs in both modes and before the three
    that differ between them, so the statistics CSV keeps its row order."""
    from haemolynx.graph import IS_ZERO_RESISTANCE

    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 0, 40), 2: (0, 9, 40)}.items():
        G.add_node(node, pos=np.asarray(pos, float))
    G.add_edge(0, 1, length=40.0, voxels=[[0, 0, 0], [0, 0, 40]])
    G.add_edge(1, 2, length=9.0, voxels=[[0, 0, 40], [0, 9, 40]], **{IS_ZERO_RESISTANCE: True})

    sections = compute_vessel_statistics_by_measure(
        G, enabled_measures=frozenset({"basic", "flow_hierarchy", "path_efficiency"})
    )

    assert list(sections) == ["basic", "flow_hierarchy", ZERO_RESISTANCE_BRIDGES, "path_efficiency"]
    assert sections[ZERO_RESISTANCE_BRIDGES] == {
        "Zero-resistance Bridges (not counted as vessels)": 1,
        "Zero-resistance Bridge Length (microns)": 9.0,
    }
    assert ZERO_RESISTANCE_BRIDGES not in compute_vessel_statistics_by_measure(
        nx.MultiGraph(nx.path_graph(2)), enabled_measures=frozenset({"basic"})
    )


def test_enabled_measures_with_nothing_enabled_still_reports_the_mode(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
        enabled_measures=frozenset(),
    )
    assert s == {"Statistics Mode": "fast"}


def test_an_unknown_enabled_measure_name_is_rejected(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    with pytest.raises(ValueError, match="Unknown statistic measure"):
        compute_comprehensive_vessel_statistics(
            simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
            enabled_measures=frozenset({"not_a_real_measure"}),
        )


# --- compute_cyclomatic_number ------------------------------------------------


def test_compute_cyclomatic_number_on_a_tree_is_zero(simple_graph):
    s = compute_cyclomatic_number(simple_graph)
    assert s["Cyclomatic Number"] == 0
    assert s["Cyclomatic Number Per Node"] == 0.0


def test_compute_cyclomatic_number_counts_a_loop():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    s = compute_cyclomatic_number(G)
    assert s["Cyclomatic Number"] == 1


def test_compute_cyclomatic_number_counts_parallel_edges_as_their_own_loop():
    """Regression: a naive nx.Graph(G) collapse before counting would merge
    two parallel vessels between the same junctions into one edge and erase
    the loop they form -- the cyclomatic number must be computed on the
    MultiGraph directly."""
    G = nx.MultiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)  # a second, parallel vessel between the same junctions
    s = compute_cyclomatic_number(G)
    assert s["Cyclomatic Number"] == 1


def test_compute_cyclomatic_number_on_an_empty_graph_does_not_raise():
    s = compute_cyclomatic_number(nx.Graph())
    assert s["Cyclomatic Number"] == 0
    assert s["Cyclomatic Number Per Node"] == "N/A (no nodes)"


# --- compute_degree_assortativity ---------------------------------------------


def test_compute_degree_assortativity_disassortative_star():
    """A star (one hub, several leaves) is maximally disassortative."""
    G = nx.star_graph(5)  # node 0 is the hub, degree 5; leaves are degree 1
    s = compute_degree_assortativity(G)
    assert s["Degree Assortativity"] < 0


def test_compute_degree_assortativity_too_few_edges_is_na():
    G = nx.Graph()
    G.add_edge(0, 1)
    assert compute_degree_assortativity(G) == {"Degree Assortativity": "N/A (fewer than 2 edges)"}


def test_compute_degree_assortativity_no_variance_is_na():
    """Every node the same degree (a cycle) has no degree variance to
    correlate -- nx.degree_assortativity_coefficient returns NaN for this,
    which must come back as a readable N/A, not a silent NaN in the report."""
    G = nx.cycle_graph(4)
    result = compute_degree_assortativity(G)
    assert result["Degree Assortativity"] == "N/A (no degree variance)"


# --- compute_rich_club_coefficient --------------------------------------------


def test_compute_rich_club_coefficient_reports_the_tightest_tier():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (3, 5)])
    s = compute_rich_club_coefficient(G)
    max_degree = max(dict(G.degree()).values())
    assert s["Rich Club Coefficient Degree Threshold"] == max_degree - 1
    assert 0.0 <= s["Rich Club Coefficient"] <= 1.0


def test_compute_rich_club_coefficient_handles_a_multigraph():
    G = nx.MultiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    s = compute_rich_club_coefficient(G)
    assert s["Rich Club Coefficient"] != "N/A (no edges)"


def test_a_loop_vessel_changes_neither_rich_club_nor_k_core():
    """Contracting thick-vessel bridges makes a self-loop wherever a vessel
    opens into one thick vessel at both ends; NetworkX refuses self-loops,
    and E14.5's statistics stopped there (2026-10-12)."""
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    without = (compute_rich_club_coefficient(G), compute_k_core_structure(G))
    G.add_edge(2, 2, length=18.0)
    assert (compute_rich_club_coefficient(G), compute_k_core_structure(G)) == without


def test_compute_rich_club_coefficient_no_edges_is_na():
    G = nx.Graph()
    G.add_node(0)
    assert compute_rich_club_coefficient(G) == {"Rich Club Coefficient": "N/A (no edges)"}


# --- compute_k_core_structure -------------------------------------------------


def test_compute_k_core_structure_a_triangle_plus_pendant():
    """A fully-connected triangle (core number 2) with one pendant node
    hanging off it (core number 1) -- the pendant should not water down the
    innermost core's own size."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    s = compute_k_core_structure(G)
    assert s["Max Core Number"] == 2
    assert s["Innermost Core Size"] == 3
    assert s["Mean Core Number"] == pytest.approx((2 + 2 + 2 + 1) / 4)


def test_compute_k_core_structure_empty_graph_does_not_raise():
    s = compute_k_core_structure(nx.Graph())
    assert s["Max Core Number"] == 0
    assert s["Innermost Core Size"] == 0


def test_compute_k_core_structure_handles_a_multigraph():
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edge(0, 1)  # parallel edge -- must not raise on collapse to simple
    s = compute_k_core_structure(G)
    assert s["Max Core Number"] == 2


# --- compute_network_robustness: perfusion-critical bridges/articulation -----


def _dead_end_and_main_path_graph() -> nx.Graph:
    """0 (inlet) -1- 1 -2- 2 (outlet), plus a dead-end spur 1-3 that leads
    nowhere -- edges (0,1) and (1,2) are both bridges on the only
    inlet-to-outlet path; edge (1,3) is a bridge to a spur that is neither
    inlet nor outlet. Node 1 is the one articulation point, and it is
    perfusion-critical (removing it separates 0 from 2)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return G


def test_compute_network_robustness_without_boundaries_is_unchanged():
    """Backward compatibility: omitting inlet/outlet nodes must report
    exactly the same keys as before this parameter existed."""
    G = _dead_end_and_main_path_graph()
    s = compute_network_robustness(G)
    assert set(s) == {
        "Bridge Edge Count", "Bridge Edge Fraction",
        "Articulation Point Count", "Articulation Point Fraction",
    }
    assert s["Bridge Edge Count"] == 3
    assert s["Articulation Point Count"] == 1


def test_compute_network_robustness_flags_only_the_perfusion_path_bridges_as_critical():
    G = _dead_end_and_main_path_graph()
    s = compute_network_robustness(G, inlet_nodes=[0], outlet_nodes=[2])
    assert s["Critical Bridge Edge Count"] == 2  # (0,1) and (1,2), not the (1,3) spur
    assert s["Critical Bridge Edge Fraction"] == pytest.approx(2 / 3)


def test_compute_network_robustness_flags_the_articulation_point_between_inlet_and_outlet():
    G = _dead_end_and_main_path_graph()
    s = compute_network_robustness(G, inlet_nodes=[0], outlet_nodes=[2])
    assert s["Critical Articulation Point Count"] == 1
    assert s["Critical Articulation Point Fraction"] == 1.0


def test_compute_network_robustness_a_bridge_off_the_perfusion_path_is_not_critical():
    """A dead-end spur bridging off a node that is itself neither inlet nor
    outlet, and not on the only path between them, must not be flagged."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # 0=inlet ... 2=outlet, 2-3 is a spur
    s = compute_network_robustness(G, inlet_nodes=[0], outlet_nodes=[2])
    assert s["Critical Bridge Edge Count"] == 2  # (0,1) and (1,2)
    assert s["Bridge Edge Count"] == 3  # (0,1), (1,2), (2,3)


def test_compute_network_robustness_ignores_boundaries_when_only_one_side_given():
    G = _dead_end_and_main_path_graph()
    s = compute_network_robustness(G, inlet_nodes=[0], outlet_nodes=[])
    assert "Critical Bridge Edge Count" not in s


def test_compute_network_robustness_no_bridges_reports_na_for_critical_fraction():
    G = nx.cycle_graph(4)  # no bridges at all
    s = compute_network_robustness(G, inlet_nodes=[0], outlet_nodes=[2])
    assert s["Bridge Edge Count"] == 0
    assert s["Critical Bridge Edge Fraction"] == "N/A (no bridges)"


def test_community_and_betweenness_measures_are_independently_toggleable(simple_graph):
    """These two back the "fast/slow community and betweenness" GUI
    toggle's underlying statistics_mode -- confirm they can each be
    switched off without disturbing the other, in both modes."""
    pos = nx.get_node_attributes(simple_graph, "pos")
    for mode in ("fast", "full"):
        full = compute_comprehensive_vessel_statistics(
            simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
            statistics_mode=mode,
        )
        no_community = compute_comprehensive_vessel_statistics(
            simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
            statistics_mode=mode,
            enabled_measures=frozenset(STATISTIC_MEASURES) - {"community"},
        )
        no_betweenness = compute_comprehensive_vessel_statistics(
            simple_graph, node_positions=pos, image_dimensions=(10, 10, 10),
            statistics_mode=mode,
            enabled_measures=frozenset(STATISTIC_MEASURES) - {"betweenness"},
        )
        community_keys = set(full) - set(no_community)
        betweenness_keys = set(full) - set(no_betweenness)
        assert community_keys, f"mode={mode}: disabling community removed nothing"
        assert betweenness_keys, f"mode={mode}: disabling betweenness removed nothing"
        assert community_keys.isdisjoint(betweenness_keys)


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
    assert result["Mean Murray Ratio"] == (
        "N/A (no junction had a measured diameter on every branch)"
    )
    assert result["Murray Junctions Without Measured Diameters"] == 1


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


def _bifurcation(parent, daughters, *, sources=None):
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 10.0))
    G.add_edge(0, 1, key=0, branch_order="Art1", diameter_um=parent)
    for i, d in enumerate(daughters):
        G.add_node(2 + i, pos=(0.0, 5.0 * (i + 1), 20.0))
        G.add_edge(1, 2 + i, key=0, branch_order=f"BO{i + 1}", diameter_um=d)
    for (u, v, key), source in (sources or {}).items():
        G[u][v][key]["diameter_source"] = source
    return G


def test_murray_law_ignores_diameters_the_run_filled_in_from_the_table():
    """Regression: every diameter counted whatever its source, so with the
    defaults -- every vessel at the table's 4 um -- every junction of two
    daughters scored exactly 2.0, a statement about the table, not the
    network."""
    G = _bifurcation(4.0, [4.0, 4.0], sources={
        (0, 1, 0): "table", (1, 2, 0): "table", (1, 3, 0): "table",
    })

    result = compute_murray_law_compliance(G)

    assert result["Murray Ratio Sample Count"] == 0
    assert result["Murray Junctions Without Measured Diameters"] == 1


@pytest.mark.parametrize("filled_in", ["table", "class_median", "override"])
def test_one_filled_in_daughter_skips_the_junction(filled_in):
    G = _bifurcation(10.0, [8.0, 8.0], sources={
        (0, 1, 0): "measured", (1, 2, 0): "measured", (1, 3, 0): filled_in,
    })

    assert compute_murray_law_compliance(G)["Murray Ratio Sample Count"] == 0


def test_measured_sources_all_count():
    G = _bifurcation(10.0, [8.0, 8.0], sources={
        (0, 1, 0): "measured", (1, 2, 0): "raw_section", (1, 3, 0): "edt_mask",
    })

    assert compute_murray_law_compliance(G)["Mean Murray Ratio"] == pytest.approx(1.024)


def test_a_synthetic_networks_drawn_widths_count_as_measured():
    """A synthetic network's widths are what it was drawn with -- the
    template's measurements -- so Murray's law reads them; a filled-in width
    beside them still skips the junction."""
    G = _bifurcation(10.0, [8.0, 8.0], sources={
        (0, 1, 0): "synthetic", (1, 2, 0): "synthetic", (1, 3, 0): "synthetic",
    })
    result = compute_murray_law_compliance(G)
    assert result["Murray Ratio Sample Count"] == 1
    assert result["Mean Murray Ratio"] == pytest.approx(1.024)
    assert result["Median Fitted Murray Exponent"] == pytest.approx(
        math.log(2) / math.log(10.0 / 8.0), rel=1e-6
    )
    G = _bifurcation(10.0, [8.0, 8.0], sources={
        (0, 1, 0): "synthetic", (1, 2, 0): "synthetic", (1, 3, 0): "table",
    })
    assert compute_murray_law_compliance(G)["Murray Ratio Sample Count"] == 0


def test_murray_reads_a_generated_networks_exponent_off_its_trees():
    """The generator holds Murray's law exactly at every tree junction with
    the template's exponent: the statistic, reading the drawn widths, finds
    that exponent at every junction of the trees, and skips none."""
    import sys
    from dataclasses import replace
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from haemolynx.graph.branch_order import assign_vessel_branch_orders
    from haemolynx.synthetic import boundary_nodes, generate_network
    from synthetic_fixtures import fixture_template

    G = generate_network(replace(fixture_template(), murray_exponent=2.6), seed=4)
    roles = boundary_nodes(G)
    assign_vessel_branch_orders(
        G, roles["inlet"], outlet_nodes=roles["outlet"],
        arteriole_boundary_nodes=roles["arteriole_boundary"], venule_boundary_nodes=roles["venule_boundary"],
    )
    assert compute_murray_law_compliance(G)["Murray Junctions Without Measured Diameters"] == 0
    trees = G.edge_subgraph(
        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d["synthetic_class"] in ("arteriole", "venule")
    )
    result = compute_murray_law_compliance(trees)
    assert result["Fitted Murray Exponent Sample Count"] >= 6
    assert result["Median Fitted Murray Exponent"] == pytest.approx(2.6, rel=1e-6)
    assert result["Fitted Murray Exponent IQR Low"] == pytest.approx(2.6, rel=1e-6)
    assert result["Fitted Murray Exponent IQR High"] == pytest.approx(2.6, rel=1e-6)


def test_each_junction_gets_its_own_fitted_exponent():
    """Two equal daughters of 2**(-1/n) times the parent satisfy the law at
    exactly n; the median and IQR are over the junctions' own exponents."""
    from haemolynx.statistics.bifurcation import fitted_murray_exponent

    for n in (2.0, 2.7, 3.0):
        assert fitted_murray_exponent(10.0, [10.0 * 2 ** (-1 / n)] * 2) == pytest.approx(n)

    G = _bifurcation(10.0, [10.0 * 2 ** (-1 / 3)] * 2)
    result = compute_murray_law_compliance(G)
    assert result["Median Fitted Murray Exponent"] == pytest.approx(3.0)
    assert result["Fitted Murray Exponent Sample Count"] == 1


@pytest.mark.parametrize("daughters", [[8.0], [10.0, 6.0], [12.0, 5.0]])
def test_no_exponent_fits_a_single_daughter_or_one_as_wide_as_its_parent(daughters):
    from haemolynx.statistics.bifurcation import fitted_murray_exponent

    assert fitted_murray_exponent(10.0, daughters) is None


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
    assert 'Average Edge Length,10,um,"Total Edge Length divided by Total Edges: the mean vessel length, the' in text
    assert "Path Efficiency Pair Coverage,0.5,fraction,Node pairs measured over all node pairs." in text
    assert "nested,Community Count,2,count," in text


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


def test_compute_branch_order_statistics_orders_large_vessel_tiers_outermost():
    """Large_Art sits before Art, Large_Ven sits after Ven -- mirroring the
    hierarchical BFS tier order in graph/branch_order.py, not sorted
    alphabetically (which would put Large_Art after BO and Large_Ven
    before Ven)."""
    G = nx.MultiGraph()
    for node in range(6):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    G.add_edge(0, 1, key=0, branch_order="Large_Art1", length=1.0)
    G.add_edge(1, 2, key=0, branch_order="Art1", length=1.0)
    G.add_edge(2, 3, key=0, branch_order="B01", length=1.0)
    G.add_edge(3, 4, key=0, branch_order="Ven1", length=1.0)
    G.add_edge(4, 5, key=0, branch_order="Large_Ven1", length=1.0)

    pos = nx.get_node_attributes(G, "pos")
    s = compute_branch_order_statistics(G, node_positions=pos)

    assert list(s.keys()) == ["Large_Art1", "Art1", "BO1", "Ven1", "Large_Ven1"]
    assert s["Large_Art1"]["Edge Count"] == 1
    assert s["Large_Ven1"]["Edge Count"] == 1


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


def test_branch_order_pressure_drop_leaves_out_vessels_carrying_no_flow():
    """A vessel behind a dead end or in a piece with no inlet and outlet has a
    drop of exactly 0 that is no pressure loss: it pulled the order's mean
    towards 0 (PLAN_dead_ends.md, inconsistency 10)."""
    G = nx.MultiGraph()
    for node in range(5):
        G.add_node(node, pos=(0.0, 0.0, float(node)))
    G.add_edge(0, 1, branch_order="B01", length=1.0, pressure_drop=10.0, flow_state="solved")
    G.add_edge(1, 2, branch_order="B01", length=1.0, pressure_drop=0.0, flow_state="blind_end")
    G.add_edge(2, 3, branch_order="B01", length=1.0, pressure_drop=0.0, flow_state="no_boundary")
    G.add_edge(3, 4, branch_order="B01", length=1.0, pressure_drop=-20.0)  # solved before flow states

    s = compute_branch_order_statistics(G, node_positions=nx.get_node_attributes(G, "pos"))

    assert s["BO1"]["Pressure Drop Sample Count"] == 2
    assert s["BO1"]["Mean Pressure Drop (Pa)"] == pytest.approx(15.0)


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
        "Branch Order,Edge Count,Mean Length (um),Mean Tortuosity Index,"
        "Mean Emergence Angle (degrees),Mean Pressure Drop (Pa),"
        "Total Pressure Drop (Pa),Mean Diameter (um),"
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
    # A and C are 50 apart all along; B runs on past C's end, so its median
    # distance along its length is a little more (50.26), never the 0 of
    # the junction it shares with A.
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(50.0)
    assert result["Mean Intercapillary Distance (microns)"] == pytest.approx(
        (50.0 + 50.0 + (np.hypot(50.0, 4.0) + np.hypot(50.0, 6.0)) / 2) / 3
    )


def test_intercapillary_distance_is_the_median_along_a_vessel_not_its_closest_approach():
    """Regression: each vessel counted only its closest approach to another,
    which usually falls at one end, so values ran low. A and C run 100 um,
    40 um apart all along, and a short stub D sits 5 um beside A's end: the
    spacing is 40 um, where the closest approaches gave 5, 35 and 5."""
    G = nx.MultiGraph()
    for node, pos in {
        0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 100.0),  # A
        2: (0.0, 40.0, 0.0), 3: (0.0, 40.0, 100.0),  # C
        4: (0.0, 5.0, 100.0), 5: (0.0, 5.0, 104.0),  # D, a stub beside A's end
    }.items():
        G.add_node(node, pos=pos)
    _straight_edge(G, 0, 1, "BO1")
    _straight_edge(G, 2, 3, "BO2")
    _straight_edge(G, 4, 5, "BO3")

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Sample Count"] == 3
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(40.0, abs=0.5)


def test_intercapillary_distance_weights_microns_not_centreline_vertices():
    """A centreline's vertices are unevenly spaced (a voxel apart, four times
    denser per micron along a fine axis than a coarse one); the median is over
    the vessel's length. A has 80 vertices over its first 20 um, beside B
    (30 um away), and 21 over the other 80 um, which reach from B's end out
    to E (60 um away): along its length its median is sqrt(30^2 + 30^2)."""
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 100.0),
                      2: (0.0, 30.0, 0.0), 3: (0.0, 30.0, 20.0),
                      4: (0.0, 60.0, 20.0), 5: (0.0, 60.0, 100.0)}.items():
        G.add_node(node, pos=pos)
    dense = [[0.0, 0.0, x] for x in np.arange(0.0, 20.0, 0.25)]
    sparse = [[0.0, 0.0, x] for x in np.arange(20.0, 100.1, 4.0)]
    G.add_edge(0, 1, key=0, branch_order="BO1", voxels=dense + sparse)
    _straight_edge(G, 2, 3, "BO2")  # B
    _straight_edge(G, 4, 5, "BO3")  # E

    result = compute_intercapillary_distance(G)

    # A ~42.4, B 30, E ~50: the median is A's. Counted per vertex, A's would
    # be 30.
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(
        np.hypot(30.0, 30.0), abs=1.0
    )


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


def test_statistics_leave_zero_resistance_bridges_out():
    """A bridge is not a vessel: not counted, not in the length, and a vessel
    opening into a thick vessel meets its centreline directly; the bridges
    are reported on their own. Per-vessel results still land on the graph."""
    from haemolynx.graph import IS_ZERO_RESISTANCE

    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 0, 40), 2: (0, 0, 80), 3: (0, 9, 40), 4: (0, 40, 40)}.items():
        G.add_node(node, pos=np.asarray(pos, float))
    for u, v, length in ((0, 1, 40.0), (1, 2, 40.0), (3, 4, 31.0)):
        G.add_edge(u, v, length=length, voxels=[G.nodes[u]["pos"].tolist(), G.nodes[v]["pos"].tolist()])
    G.add_edge(1, 3, length=9.0, voxels=[[0, 0, 40], [0, 9, 40]], **{IS_ZERO_RESISTANCE: True})

    stats = compute_comprehensive_vessel_statistics(
        G, enabled_measures=frozenset({"basic", "strahler"}), inlet_nodes=[0], outlet_nodes=[2]
    )

    assert stats["Total Edges"] == 3
    assert stats["Total Nodes"] == 4
    assert stats["Total Edge Length (microns)"] == pytest.approx(111.0)
    assert stats["Zero-resistance Bridges (not counted as vessels)"] == 1
    assert stats["Zero-resistance Bridge Length (microns)"] == pytest.approx(9.0)
    assert "strahler_order" in G.edges[3, 4, 0]
    assert "strahler_order" not in G.edges[1, 3, 0]
    assert G.number_of_edges() == 4, "the graph itself keeps its bridge"


def test_the_dead_ends_section_counts_kinds_density_cut_length_and_why_there_is_no_flow():
    """PLAN_dead_ends.md Phase 7: by kind, the interior density (open ends
    and artefacts never in it), chain lengths, the share the faces cut, and
    the flow states' lengths."""
    from haemolynx.graph import FLOW_STATE
    from haemolynx.graph.dead_ends import DEAD_END_KIND

    G = nx.MultiGraph()
    pos = {0: (5, 5, 0), 1: (5, 5, 30), 2: (5, 5, 60), 3: (5, 5, 99), 4: (5, 0, 30), 5: (5, 8, 60),
           6: (5, 5, 80), 7: (5, 8, 85)}
    for n, p in pos.items():
        G.add_node(n, pos=np.asarray(p, dtype=float))
    states = {(0, 1): "solved", (1, 2): "solved", (2, 3): "solved", (1, 4): "closed_end",
              (2, 5): "blind_end", (6, 7): "no_boundary"}
    for (u, v), state in states.items():
        G.add_edge(u, v, length=float(np.linalg.norm(np.subtract(pos[u], pos[v]))), **{FLOW_STATE: state})
    for n, kind in {0: "leaves_image", 3: "leaves_image", 4: "leaves_image", 5: "unknown",
                    6: "island_end", 7: "island_end"}.items():
        G.nodes[n][DEAD_END_KIND] = kind

    rows = compute_comprehensive_vessel_statistics(
        G, enabled_measures=frozenset({"dead_ends"}), tissue_volume_um3=1e6,
    )["Dead Ends"]

    assert rows["Dead Ends"] == 6
    assert rows["Dead Ends Leaving the Image"] == 3
    assert rows["Interior Dead Ends, Blind or Broken (unknown)"] == 1
    assert rows["Ends of Pieces Without a Junction"] == 2
    total = sum(d["length"] for *_e, d in G.edges(data=True))
    assert rows["Interior Dead Ends per mm of Vessel (1/mm)"] == pytest.approx(1 / (total / 1000))
    assert rows["Interior Dead Ends per mm^3 of Tissue (1/mm^3)"] == pytest.approx(1000.0)
    assert rows["Interior Dead-End Chain Length, median (microns)"] == pytest.approx(3.0)
    # The chains ending at the three face ends: 0-1, 2-3 and 1-4.
    assert rows["Vessel Length on Chains Cut by the Image Faces (microns)"] == pytest.approx(30 + 39 + 5)
    assert rows["Pieces Without a Junction"] == 1
    assert rows["Vessel Length Behind Interior Dead Ends, No Flow (microns)"] == pytest.approx(3.0)
    assert rows["Vessel Length Behind Closed Image-Face Ends, No Flow (microns)"] == pytest.approx(5.0)
    assert rows["Vessel Length Carrying Flow (microns)"] == pytest.approx(99.0)
    # Decision D5: the lengths without the three cut vessels, and of them alone.
    island = float(np.hypot(3, 5))
    assert rows["Vessels Cut by the Image Faces"] == 3
    assert rows["Mean Vessel Length, Not Cut by the Image Faces (microns)"] == pytest.approx((30 + 3 + island) / 3)
    assert rows["Median Vessel Length, Not Cut by the Image Faces (microns)"] == pytest.approx(island)
    assert rows["Mean Vessel Length, Cut by the Image Faces, Censored (microns)"] == pytest.approx((30 + 39 + 5) / 3)
    assert rows["Median Vessel Length, Cut by the Image Faces, Censored (microns)"] == pytest.approx(30.0)


# --- intercapillary distance between the vessels carrying flow ---------------------------


def _three_parallel_vessels(flows):
    """A, B and C, 10 um long, side by side 50 um apart (A at y=0, B at 50,
    C at 100), each given one of *flows* as a solved flow (None: no flow)."""
    G = nx.MultiGraph()
    for i, y in enumerate((0.0, 50.0, 100.0)):
        G.add_node(2 * i, pos=(0.0, y, 0.0))
        G.add_node(2 * i + 1, pos=(0.0, y, 10.0))
        _straight_edge(G, 2 * i, 2 * i + 1, f"BO{i + 1}")
        flow = flows[i]
        if flow is not None:
            G[2 * i][2 * i + 1][0].update(
                flow_abs=float(flow),
                flow_state="solved" if flow > 0 else "blind_end",
                flow_solved=flow > 0,
            )
    return G


def test_intercapillary_distance_between_flowing_vessels_leaves_out_the_vessel_carrying_none():
    """B, in the middle, carries no flow: every vessel's nearest neighbour is
    50 um away, but the nearest *flowing* neighbour of A and C is each other,
    100 um away -- the spacing the perfused vessels leave for oxygen to cross."""
    G = _three_parallel_vessels([1e-12, 0.0, 1e-12])

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Sample Count"] == 3
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(50.0)
    assert result["Intercapillary Distance Between Flowing Vessels Sample Count"] == 2
    assert result["Mean Intercapillary Distance Between Flowing Vessels (microns)"] == pytest.approx(100.0)
    assert result["Median Intercapillary Distance Between Flowing Vessels (microns)"] == pytest.approx(100.0)
    assert result["Flowing Over All Median Intercapillary Distance"] == pytest.approx(2.0)


def test_intercapillary_distance_between_flowing_vessels_is_the_whole_when_every_vessel_flows():
    G = _three_parallel_vessels([1e-12, 2e-12, 3e-12])

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Between Flowing Vessels Sample Count"] == 3
    assert result["Median Intercapillary Distance Between Flowing Vessels (microns)"] == pytest.approx(
        result["Median Intercapillary Distance (microns)"]
    )
    assert result["Flowing Over All Median Intercapillary Distance"] == pytest.approx(1.0)


def test_intercapillary_distance_between_flowing_vessels_needs_a_solved_flow():
    G = _three_parallel_vessels([None, None, None])

    result = compute_intercapillary_distance(G)

    # The rows over every vessel are as they always were.
    assert result["Intercapillary Distance Sample Count"] == 3
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(50.0)
    reason = "N/A (needs a solved flow (run haemodynamics))"
    assert result["Mean Intercapillary Distance Between Flowing Vessels (microns)"] == reason
    assert result["Median Intercapillary Distance Between Flowing Vessels (microns)"] == reason
    assert result["Intercapillary Distance Between Flowing Vessels Sample Count"] == 0
    assert result["Flowing Over All Median Intercapillary Distance"] == reason


def test_intercapillary_distance_between_flowing_vessels_needs_two_of_them():
    G = _three_parallel_vessels([1e-12, 0.0, 0.0])

    result = compute_intercapillary_distance(G)

    reason = "N/A (fewer than two vessels carry flow)"
    assert result["Intercapillary Distance Between Flowing Vessels Sample Count"] == 0
    assert result["Median Intercapillary Distance Between Flowing Vessels (microns)"] == reason
    assert result["Flowing Over All Median Intercapillary Distance"] == reason
    assert result["Median Intercapillary Distance (microns)"] == pytest.approx(50.0)


def test_intercapillary_distance_between_flowing_vessels_uses_the_carrying_floor():
    """A flow the solve balanced to rounding is no flow: B's 1e-24 against
    A's and C's 1e-12 is below a billionth of the largest."""
    G = _three_parallel_vessels([1e-12, 1e-24, 1e-12])

    result = compute_intercapillary_distance(G)

    assert result["Intercapillary Distance Between Flowing Vessels Sample Count"] == 2
    assert result["Median Intercapillary Distance Between Flowing Vessels (microns)"] == pytest.approx(100.0)
