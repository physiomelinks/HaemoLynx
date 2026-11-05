"""Inlet->outlet bottlenecks and shunts on graphs with known answers."""
from __future__ import annotations

import math

import networkx as nx
import pytest

from haemolynx.statistics.inlet_outlet_routes import (
    compute_inlet_outlet_bottlenecks,
    compute_inlet_outlet_shunts,
)


def _vessel(G, u, v, length=10.0, resistance=None, flow=None):
    data = {"length": float(length)}
    if resistance is not None:
        data["resistance"] = float(resistance)
        data["conductance"] = 1.0 / float(resistance)
    if flow is not None:
        data["flow_abs"] = float(flow)
    return G.add_edge(u, v, **data)


def _edge_attr(G, u, v, name):
    return [data[name] for data in G.get_edge_data(u, v).values()]


def _two_branches_into_a_trunk():
    """Inlet 0 splits into two branches (1, 2) that rejoin at 3, then a
    single trunk 3-4 runs to outlet 4: the trunk is the only bottleneck."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1)
    _vessel(G, 0, 2)
    _vessel(G, 1, 3)
    _vessel(G, 2, 3)
    _vessel(G, 3, 4)
    return G


# --- bottlenecks ---------------------------------------------------------------


def test_the_single_trunk_is_the_min_cut_and_carries_every_route():
    G = _two_branches_into_a_trunk()

    result = compute_inlet_outlet_bottlenecks(G, [0], [4])

    assert result["Bottleneck Min-Cut Edge Count"] == 1
    assert result["Bottleneck Min-Cut Capacity"] == pytest.approx(1.0)
    assert _edge_attr(G, 3, 4, "bottleneck_min_cut") == ["min_cut"]
    assert _edge_attr(G, 0, 1, "bottleneck_min_cut") == ["not_min_cut"]
    assert _edge_attr(G, 3, 4, "bottleneck_route_share") == [pytest.approx(1.0)]
    assert result["Bottleneck Route Share Max"] == pytest.approx(1.0)


def test_a_parallel_vessel_doubles_capacity_so_it_is_not_the_bottleneck():
    G = _two_branches_into_a_trunk()
    _vessel(G, 3, 4)  # a second, parallel trunk vessel
    _vessel(G, 4, 5)  # then one single vessel to the outlet
    result = compute_inlet_outlet_bottlenecks(G, [0], [5])

    assert result["Bottleneck Min-Cut Edge Count"] == 1
    assert _edge_attr(G, 4, 5, "bottleneck_min_cut") == ["min_cut"]
    assert _edge_attr(G, 3, 4, "bottleneck_min_cut") == ["not_min_cut", "not_min_cut"]


def test_resistance_weighting_cuts_the_lowest_conductance_cross_section():
    """Two vessels in the inlet cross-section with tiny conductance vs one
    fat trunk: fewest vessels is the trunk, but hydraulically the two thin
    vessels are the narrower cross-section."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, resistance=1e6)
    _vessel(G, 0, 2, resistance=1e6)
    _vessel(G, 1, 3, resistance=1.0)
    _vessel(G, 2, 3, resistance=1.0)
    _vessel(G, 3, 4, resistance=1.0)

    by_count = compute_inlet_outlet_bottlenecks(G.copy(), [0], [4], weighting="topology")
    result = compute_inlet_outlet_bottlenecks(G, [0], [4], weighting="resistance")

    assert by_count["Bottleneck Min-Cut Edge Count"] == 1
    assert result["Bottleneck Route Weighting"] == "resistance"
    assert result["Bottleneck Min-Cut Edge Count"] == 2
    assert result["Bottleneck Min-Cut Capacity"] == pytest.approx(2e-6)
    assert _edge_attr(G, 0, 1, "bottleneck_min_cut") == ["min_cut"]
    assert _edge_attr(G, 3, 4, "bottleneck_min_cut") == ["not_min_cut"]


def test_resistance_weighting_falls_back_to_length_without_haemodynamics():
    G = _two_branches_into_a_trunk()
    result = compute_inlet_outlet_bottlenecks(G, [0], [4], weighting="resistance")
    assert result["Bottleneck Route Weighting"] == "length (resistance unavailable)"
    assert result["Bottleneck Min-Cut Edge Count"] == 1


def test_route_share_follows_the_shorter_branch_under_length_weighting():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, length=1.0)
    _vessel(G, 1, 3, length=1.0)
    _vessel(G, 0, 2, length=50.0)
    _vessel(G, 2, 3, length=50.0)

    compute_inlet_outlet_bottlenecks(G, [0], [3], weighting="length")

    assert _edge_attr(G, 0, 1, "bottleneck_route_share") == [pytest.approx(1.0)]
    assert _edge_attr(G, 0, 2, "bottleneck_route_share") == [pytest.approx(0.0)]


def test_route_share_is_a_fraction_of_inlet_outlet_pairs():
    """Two inlets both feed a shared trunk, but only one of them uses spur 1-2."""
    G = nx.MultiGraph()
    _vessel(G, 10, 1)
    _vessel(G, 11, 2)
    _vessel(G, 1, 2)
    _vessel(G, 2, 3)

    compute_inlet_outlet_bottlenecks(G, [10, 11], [3])

    assert _edge_attr(G, 2, 3, "bottleneck_route_share") == [pytest.approx(1.0)]
    assert _edge_attr(G, 1, 2, "bottleneck_route_share") == [pytest.approx(0.5)]


def test_fast_mode_samples_sources_only_above_the_cap():
    G = _two_branches_into_a_trunk()
    exact = compute_inlet_outlet_bottlenecks(G.copy(), [0], [4], statistics_mode="fast")
    full = compute_inlet_outlet_bottlenecks(G, [0], [4], statistics_mode="full")
    assert exact["Bottleneck Route Share Method"] == "exact"
    assert exact["Bottleneck Route Share Max"] == full["Bottleneck Route Share Max"]

    star = nx.MultiGraph()
    for leaf in range(10):
        _vessel(star, leaf, 100)
    _vessel(star, 100, 200)
    sampled = compute_inlet_outlet_bottlenecks(
        star, list(range(10)), [200], max_sources=3
    )
    assert sampled["Bottleneck Route Share Method"] == "sampled_sources=3"
    assert _edge_attr(star, 100, 200, "bottleneck_route_share") == [pytest.approx(1.0)]


# --- shunts ----------------------------------------------------------------------


def _capillary_bed_with_a_shunt():
    """A 5x5 grid capillary bed between inlet corner (0, 0) and outlet
    corner (4, 4) -- typical route length 80 um -- plus one direct 10 um
    vessel from the inlet straight to the outlet."""
    grid = nx.grid_2d_graph(5, 5)
    G = nx.MultiGraph()
    for u, v in grid.edges():
        _vessel(G, u, v, length=10.0)
    _vessel(G, (0, 0), (4, 4), length=10.0)
    return G


def test_a_direct_arteriole_venule_vessel_is_flagged_as_a_shunt():
    G = _capillary_bed_with_a_shunt()

    result = compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert _edge_attr(G, (0, 0), (4, 4), "shunt") == ["shunt"]
    assert _edge_attr(G, (0, 0), (4, 4), "shunt_route_ratio")[0] < 0.5
    assert _edge_attr(G, (2, 2), (2, 3), "shunt") == ["not_shunt"]
    assert result["Shunt Edge Count"] == 1
    assert result["Shunt Pathway Count"] == 1
    assert result["Shortest Inlet-Outlet Route"] == pytest.approx(10.0)


def test_bed_vessels_beside_a_shunt_do_not_borrow_its_shortness():
    """Regression: the bed vessel leaving the inlet must be scored on its own
    route through the bed (80 um), not on the walk "inlet -> vessel -> back
    through the inlet -> shunt" (30 um), which flagged it as a shunt too."""
    G = _capillary_bed_with_a_shunt()

    compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    for u, v in [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((3, 4), (4, 4)), ((4, 3), (4, 4))]:
        assert _edge_attr(G, u, v, "shunt") == ["not_shunt"], (u, v)
        assert _edge_attr(G, u, v, "shunt_route_ratio")[0] == pytest.approx(1.0), (u, v)


def test_a_bed_branch_leaving_a_shunt_midway_is_not_itself_a_shunt():
    G = _capillary_bed_with_a_shunt()
    G.remove_edge((0, 0), (4, 4))
    _vessel(G, (0, 0), "s1", length=5.0)
    _vessel(G, "s1", (4, 4), length=5.0)
    _vessel(G, "s1", (2, 2), length=5.0)

    compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert _edge_attr(G, (0, 0), "s1", "shunt") == ["shunt"]
    assert _edge_attr(G, "s1", (4, 4), "shunt") == ["shunt"]
    assert _edge_attr(G, "s1", (2, 2), "shunt") == ["not_shunt"]


def test_a_dead_end_spur_has_no_route():
    G = _capillary_bed_with_a_shunt()
    _vessel(G, (0, 0), "spur_tip", length=1.0)

    compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert math.isnan(_edge_attr(G, (0, 0), "spur_tip", "shunt_route_ratio")[0])
    assert _edge_attr(G, (0, 0), "spur_tip", "shunt") == ["not_shunt"]


def test_a_bed_with_no_shortcut_has_no_shunt():
    grid = nx.grid_2d_graph(5, 5)
    G = nx.MultiGraph()
    for u, v in grid.edges():
        _vessel(G, u, v)

    result = compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert result["Shunt Edge Count"] == 0
    assert result["Shunt Pathway Count"] == 0
    assert all(data["shunt"] == "not_shunt" for _u, _v, data in G.edges(data=True))


def test_a_multi_vessel_shunt_is_one_pathway():
    G = _capillary_bed_with_a_shunt()
    G.remove_edge((0, 0), (4, 4))
    _vessel(G, (0, 0), "s1", length=5.0)
    _vessel(G, "s1", (4, 4), length=5.0)

    result = compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert result["Shunt Edge Count"] == 2
    assert result["Shunt Pathway Count"] == 1


def test_shunt_outflow_fraction_reads_the_solved_flow_at_the_outlets():
    G = _capillary_bed_with_a_shunt()
    for _u, _v, data in G.edges(data=True):
        data["flow_abs"] = 1.0
    G[(0, 0)][(4, 4)][0]["flow_abs"] = 6.0  # the shunt; two bed vessels also reach the outlet

    result = compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    assert result["Shunt Outflow Fraction"] == pytest.approx(6.0 / 8.0)


def test_shunt_outflow_fraction_is_na_without_a_solved_flow():
    G = _capillary_bed_with_a_shunt()
    result = compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])
    assert str(result["Shunt Outflow Fraction"]).startswith("N/A")


def test_a_vessel_with_no_route_gets_a_nan_ratio():
    G = _capillary_bed_with_a_shunt()
    _vessel(G, "island_a", "island_b")

    compute_inlet_outlet_shunts(G, [(0, 0)], [(4, 4)])

    ratio = _edge_attr(G, "island_a", "island_b", "shunt_route_ratio")[0]
    assert math.isnan(ratio)
    assert _edge_attr(G, "island_a", "island_b", "shunt") == ["not_shunt"]


# --- shared edge cases -----------------------------------------------------------


@pytest.mark.parametrize(
    "compute", [compute_inlet_outlet_bottlenecks, compute_inlet_outlet_shunts]
)
def test_no_inlets_reports_na_and_writes_no_attributes(compute):
    G = _two_branches_into_a_trunk()
    result = compute(G, [], [4])
    assert any(str(v).startswith("N/A") for v in result.values())
    assert all(
        not (set(data) & {"bottleneck_route_share", "bottleneck_min_cut", "shunt", "shunt_route_ratio"})
        for _u, _v, data in G.edges(data=True)
    )


@pytest.mark.parametrize(
    "compute", [compute_inlet_outlet_bottlenecks, compute_inlet_outlet_shunts]
)
def test_disconnected_inlet_and_outlet_report_na(compute):
    G = nx.MultiGraph()
    _vessel(G, 0, 1)
    _vessel(G, 2, 3)
    result = compute(G, [0], [3])
    assert any("no inlet is connected" in str(v) for v in result.values())


@pytest.mark.parametrize(
    "compute", [compute_inlet_outlet_bottlenecks, compute_inlet_outlet_shunts]
)
def test_stale_attributes_are_cleared_when_a_rerun_has_nothing_to_analyse(compute):
    G = _two_branches_into_a_trunk()
    compute(G, [0], [4])
    compute(G, [], [])
    names = {"bottleneck_route_share", "bottleneck_min_cut", "shunt", "shunt_route_ratio"}
    assert all(not (set(data) & names) for _u, _v, data in G.edges(data=True))


def test_a_node_given_as_both_inlet_and_outlet_is_excluded_from_both():
    G = _two_branches_into_a_trunk()
    result = compute_inlet_outlet_bottlenecks(G, [0, 4], [4])
    assert result["Bottleneck Nodes Both Inlet And Outlet (excluded)"] == 1
    assert "Bottleneck Status" in result  # 4 excluded -> no outlets left


# --- through the comprehensive statistics report ---------------------------------


def test_comprehensive_statistics_runs_both_and_forwards_their_settings():
    from haemolynx.statistics import compute_comprehensive_vessel_statistics

    G = _capillary_bed_with_a_shunt()
    stats = compute_comprehensive_vessel_statistics(
        G,
        enabled_measures=frozenset({"bottlenecks", "shunts"}),
        inlet_nodes=[(0, 0)],
        outlet_nodes=[(4, 4)],
        route_weighting="topology",
        shunt_max_route_fraction=0.2,
    )

    assert stats["Bottleneck Route Weighting"] == "topology"
    assert stats["Shunt Route Weighting"] == "topology"
    assert stats["Shunt Route Fraction Threshold"] == pytest.approx(0.2)
    assert stats["Shunt Edge Count"] == 1  # 1 hop vs a median of 8
    assert _edge_attr(G, (0, 0), (4, 4), "shunt") == ["shunt"]


def test_comprehensive_statistics_writes_nothing_when_the_measures_are_off():
    from haemolynx.statistics import STATISTIC_MEASURES, compute_comprehensive_vessel_statistics

    G = _capillary_bed_with_a_shunt()
    stats = compute_comprehensive_vessel_statistics(
        G,
        enabled_measures=frozenset(STATISTIC_MEASURES) - {"bottlenecks", "shunts"},
        inlet_nodes=[(0, 0)],
        outlet_nodes=[(4, 4)],
    )

    assert not any(key.startswith(("Bottleneck", "Shunt")) for key in stats)
    names = {"bottleneck_route_share", "bottleneck_min_cut", "shunt", "shunt_route_ratio"}
    assert all(not (set(data) & names) for _u, _v, data in G.edges(data=True))


def test_new_measures_are_network_analysis_measures():
    from haemolynx.statistics import NETWORK_ANALYSIS_MEASURES, STATISTIC_MEASURES

    assert {"bottlenecks", "shunts"} <= set(STATISTIC_MEASURES)
    assert {"bottlenecks", "shunts"} <= NETWORK_ANALYSIS_MEASURES


def test_schema_route_weighting_choices_match_the_module():
    from haemolynx.pipeline import default_schema
    from haemolynx.statistics import ROUTE_WEIGHTINGS

    schema = default_schema()
    assert tuple(schema["statistics_route_weighting"].choices) == ROUTE_WEIGHTINGS
    assert schema["statistics_route_weighting"].default == "length"
    assert schema["statistics_shunt_max_route_fraction"].default == pytest.approx(0.5)


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError, match="route weighting"):
        compute_inlet_outlet_bottlenecks(_two_branches_into_a_trunk(), [0], [4], weighting="bogus")
