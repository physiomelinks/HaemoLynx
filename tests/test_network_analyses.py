"""Transit time, perfusion territories, loop scale, Horton-Strahler and
algebraic connectivity on graphs with hand-computed answers."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.statistics.loops import compute_loop_hierarchy
from haemolynx.statistics.spectral import compute_algebraic_connectivity
from haemolynx.statistics.strahler import compute_strahler_orders
from haemolynx.statistics.territories import compute_perfusion_territories
from haemolynx.statistics.transit_time import compute_transit_times


def _tau(diameter_um, length_um, flow):
    return math.pi * (diameter_um * 0.5e-6) ** 2 * (length_um * 1e-6) / flow


def _flow_vessel(G, u, v, flow, length=100.0, diameter=10.0, **extra):
    G.add_edge(u, v, flow_abs=float(flow), length=float(length), diameter_um=float(diameter), **extra)


# --- transit time ------------------------------------------------------------------


def _series_chain():
    G = nx.MultiGraph()
    for node, p in ((0, 3.0), (1, 2.0), (2, 1.0)):
        G.add_node(node, pressure=p)
    _flow_vessel(G, 0, 1, 1e-12, branch_order="Art1")
    _flow_vessel(G, 1, 2, 1e-12, branch_order="B01")
    return G


def test_series_transit_times_add_and_have_no_spread():
    G = _series_chain()
    result = compute_transit_times(G, [0], [2])
    tau = _tau(10.0, 100.0, 1e-12)
    assert result["Mean Transit Time (s)"] == pytest.approx(2 * tau)
    assert result["Transit Time SD (s)"] == pytest.approx(0.0, abs=1e-12)
    assert result["Capillary Mean Transit Time (s)"] == pytest.approx(tau)
    assert G[0][1][0]["transit_time_s"] == pytest.approx(tau)
    assert G[1][2][0]["arrival_time_s"] == pytest.approx(2 * tau)


def test_unequal_parallel_branches_give_the_flow_weighted_spread():
    G = nx.MultiGraph()
    G.add_node(0, pressure=2.0)
    G.add_node(1, pressure=1.0)
    _flow_vessel(G, 0, 1, 1.0e-12, length=100.0)
    _flow_vessel(G, 0, 1, 3.0e-12, length=900.0)
    ta, tb = _tau(10, 100, 1e-12), _tau(10, 900, 3e-12)
    mean = (1 * ta + 3 * tb) / 4
    sd = math.sqrt((1 * ta**2 + 3 * tb**2) / 4 - mean**2)

    result = compute_transit_times(G, [0], [1])

    assert result["Mean Transit Time (s)"] == pytest.approx(mean)
    assert result["Transit Time SD (s)"] == pytest.approx(sd)
    assert result["Capillary Transit Time Status"].startswith("N/A")


def test_transit_time_needs_a_solved_flow():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=10.0, diameter_um=5.0)
    result = compute_transit_times(G, [0], [1])
    assert "solved flow" in result["Transit Time Status"]
    assert all("transit_time_s" not in d for _u, _v, d in G.edges(data=True))


# --- perfusion territories ------------------------------------------------------------


def _two_inlet_chain():
    """Inlets at both ends of 0-1-2-3-4-5 (10 um vessels), one outlet (6)
    hanging off node 2: nodes 0-2 are nearer inlet 0, nodes 3-5 inlet 5."""
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)):
        G.add_edge(u, v, length=10.0)
    return G


def test_the_vessel_between_two_territories_is_the_watershed():
    G = _two_inlet_chain()
    result = compute_perfusion_territories(G, [0, 5], [6])

    assert G[0][1][0]["arteriolar_territory"] == "A01"
    assert G[4][5][0]["arteriolar_territory"] == "A02"
    assert G[2][3][0]["watershed"] == "arteriolar"
    assert G[0][1][0]["watershed"] == "none"
    assert result["Arteriolar Territory Count"] == 2
    assert result["Arteriolar Watershed Vessel Count"] == 1
    assert result["Venular Territory Count"] == 1
    assert result["Venular Watershed Vessel Count"] == 0


def test_watershed_margin_is_lowest_at_the_border():
    G = _two_inlet_chain()
    compute_perfusion_territories(G, [0, 5], [6])
    # node 2: nearest inlet 20 um, second 30 um -> (30-20)/(30+20) = 0.2
    assert G[2][3][0]["arteriolar_watershed_margin"] == pytest.approx(0.2)
    # node 1: 10 um vs 40 um -> 0.6 (node 0 itself is 1.0)
    assert G[0][1][0]["arteriolar_watershed_margin"] == pytest.approx(0.6)


def test_territories_need_inlets_and_outlets():
    G = _two_inlet_chain()
    result = compute_perfusion_territories(G, [], [6])
    assert str(result["Territory Status"]).startswith("N/A")
    assert all("watershed" not in d for _u, _v, d in G.edges(data=True))


# --- loops ---------------------------------------------------------------------------


def test_every_grid_vessel_is_on_a_40_um_loop_and_a_pendant_on_none():
    grid = nx.grid_2d_graph(5, 5)
    G = nx.MultiGraph()
    for u, v in grid.edges():
        G.add_edge(u, v, length=10.0)
    G.add_edge((0, 0), "tip", length=10.0)

    result = compute_loop_hierarchy(G, statistics_mode="full")

    loops = [d["loop_length_um"] for u, v, d in G.edges(data=True) if "tip" not in (u, v)]
    assert loops == pytest.approx([40.0] * len(loops))
    assert math.isnan(G[(0, 0)]["tip"][0]["loop_length_um"])
    assert result["Vessels On No Loop"] == 1
    assert result["Loop Length Median (um)"] == pytest.approx(40.0)


def test_parallel_vessels_form_a_loop_with_each_other():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=5.0)
    G.add_edge(0, 1, length=7.0)
    compute_loop_hierarchy(G)
    assert sorted(d["loop_length_um"] for _u, _v, d in G.edges(data=True)) == [12.0, 12.0]


def test_loop_density_uses_the_imaged_volume():
    grid = nx.grid_2d_graph(3, 3)  # 12 vessels, 9 nodes, 1 component -> 4 loops
    G = nx.MultiGraph()
    for u, v in grid.edges():
        G.add_edge(u, v, length=10.0)
    result = compute_loop_hierarchy(G, image_dimensions=(10, 100, 100), voxel_size=(1.0, 1.0, 1.0))
    assert result["Independent Loops Per mm^3"] == pytest.approx(4 / 1e-4)


# --- Horton-Strahler ----------------------------------------------------------------------


def _binary_arterial_tree(branch_order="Art1"):
    """Inlet 0 -> trunk to 1, then three levels of symmetric bifurcation."""
    G = nx.MultiGraph()
    extra = {} if branch_order is None else {"branch_order": branch_order}
    G.add_edge(0, 1, length=10.0, diameter_um=16.0, **extra)
    frontier, next_id = [1], 2
    for depth, diameter in enumerate((8.0, 4.0, 2.0)):
        new = []
        for parent in frontier:
            for _ in range(2):
                G.add_edge(parent, next_id, length=10.0, diameter_um=diameter, **extra)
                new.append(next_id)
                next_id += 1
        frontier = new
    return G


def test_symmetric_binary_tree_has_bifurcation_ratio_two():
    G = _binary_arterial_tree()
    result = compute_strahler_orders(G, [0], [])

    arterial = result["Strahler Arterial"]
    assert arterial["Max Order"] == 4
    counts = {k: row["Element Count"] for k, row in arterial["Orders"].items()}
    assert counts == {"1": 8, "2": 4, "3": 2, "4": 1}
    assert arterial["Bifurcation Ratio R_B"] == pytest.approx(2.0)
    assert arterial["Length Ratio R_L"] == pytest.approx(1.0)
    assert arterial["Diameter Ratio R_D"] == pytest.approx(2.0)
    assert G[0][1][0]["strahler_order"] == 4
    leaf_orders = {
        d["strahler_order"]
        for u, v, d in G.edges(data=True)
        if 1 in (G.degree(u), G.degree(v)) and 0 not in (u, v)
    }
    assert leaf_orders == {1.0}


def test_capillaries_are_left_unordered():
    G = _binary_arterial_tree()
    G.add_edge(8, 100, length=10.0, branch_order="B01")
    compute_strahler_orders(G, [0], [])
    assert math.isnan(G[8][100][0]["strahler_order"])


def test_untyped_network_falls_back_to_the_whole_network_tree():
    G = _binary_arterial_tree(branch_order=None)
    result = compute_strahler_orders(G, [0], [])
    assert result["Strahler Method"].startswith("approximate")
    assert result["Strahler Network"]["Max Order"] == 4


# --- algebraic connectivity ----------------------------------------------------------------


def test_path_graph_algebraic_connectivity_and_fiedler_split():
    n = 6
    G = nx.MultiGraph()
    for i in range(n - 1):
        G.add_edge(i, i + 1, length=10.0)

    result = compute_algebraic_connectivity(G, weighting="topology")

    assert result["Algebraic Connectivity lambda_2"] == pytest.approx(2 * (1 - math.cos(math.pi / n)), rel=1e-6)
    assert result["Fiedler Cut Vessel Count"] == 1
    assert G[2][3][0]["fiedler_side"] == "cut"
    assert sorted(result["Fiedler Split Vessel Counts [A, B]"]) == [2, 2]


def test_lambda_2_scales_with_the_conductance_units():
    G = nx.MultiGraph()
    for i in range(4):
        G.add_edge(i, i + 1, length=2.0)  # conductance 1/2 per vessel
    by_topology = compute_algebraic_connectivity(G.copy(), weighting="topology")
    by_length = compute_algebraic_connectivity(G, weighting="length")
    assert by_length["Algebraic Connectivity lambda_2"] == pytest.approx(
        by_topology["Algebraic Connectivity lambda_2"] / 2, rel=1e-6
    )
    assert by_length["Normalised Algebraic Connectivity"] == pytest.approx(
        by_topology["Normalised Algebraic Connectivity"], rel=1e-6
    )
