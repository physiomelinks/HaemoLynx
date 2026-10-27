"""The capillary branch orders between the arterioles and the venules, on
graphs with hand-counted routes, and its statistics and stage wiring."""
from __future__ import annotations

import csv
import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.branch_order import assign_vessel_branch_orders
from haemolynx.statistics import (
    NETWORK_ANALYSIS_MEASURES,
    STATISTIC_MEASURES,
    compute_vessel_statistics_by_measure,
    glossary,
)
from haemolynx.statistics.capillary_paths import SECTION, compute_capillary_path_orders
from test_capillary_bed_joins import _large_tier_bed
from test_statistics_without_haemodynamics import _export_with_haemodynamics_off

ORDERS = "Capillary Orders Between Arteriole and Venule"
LENGTHS = "Capillary Route Length Between Arteriole and Venule"
STRAIGHT = "Straight-Line Distance Between Arteriole and Venule"
RATIO = "Capillary Route Length Over Straight-Line Distance"
STATUS = "Capillary Path Orders Status"
LENGTH_STATUS = f"{LENGTHS} Status"
STRAIGHT_STATUS = f"{STRAIGHT} Status"
RATIO_STATUS = f"{RATIO} Status"


def _length(rows, statistic):
    return rows[f"{LENGTHS}, {statistic} (microns)"]


def _straight(rows, statistic):
    return rows[f"{STRAIGHT}, {statistic} (microns)"]


def _ratio(rows, statistic):
    return rows[f"{RATIO}, {statistic}"]


def _vessel(G, u, v, label, length=10.0):
    G.add_edge(u, v, length=float(length), branch_order=label)


def _rows(G, feeds, drains):
    return compute_capillary_path_orders(G, feeds, drains)[SECTION]


def _chain(capillaries):
    """An arteriole 0-1 (feed 1), *capillaries* capillaries in a row from 1
    to the venule end at 1 + capillaries, and the venule on from there: a
    straight line, each node 10 um further along x."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    for i in range(1, capillaries + 1):
        _vessel(G, i, i + 1, f"B{i:02d}")
    end = capillaries + 1
    _vessel(G, end, end + 1, "Ven1")
    for node in G:
        G.nodes[node]["pos"] = (0.0, 0.0, 10.0 * node)
    return G, end


# --- the count -----------------------------------------------------------------


@pytest.mark.parametrize("capillaries", [1, 2, 3, 6])
def test_a_single_route_holds_one_order_per_capillary(capillaries):
    G, end = _chain(capillaries)

    rows = _rows(G, [1], [end])

    assert rows[f"{ORDERS}, Mean"] == capillaries
    assert rows[f"{ORDERS}, Median"] == capillaries
    assert rows[f"{ORDERS}, Min"] == rows[f"{ORDERS}, Max"] == capillaries
    assert rows[f"{ORDERS}, SD"] == "N/A (one route)"
    assert _length(rows, "Mean") == _length(rows, "Min") == _length(rows, "Max") == 10.0 * capillaries
    assert _length(rows, "SD") == "N/A (one route)"
    assert _straight(rows, "Mean") == _straight(rows, "Max") == 10.0 * capillaries
    assert _ratio(rows, "Mean") == _ratio(rows, "Min") == 1.0
    assert _straight(rows, "SD") == _ratio(rows, "SD") == "N/A (one route)"
    assert rows["Capillaries Draining to a Venule"] == 1
    assert rows["Capillaries Draining to a Venule No Arteriole Reaches"] == 0


def test_every_capillary_draining_into_a_venule_ends_a_route_of_its_own():
    """From feed 1: 1-2-9 drains after 2 capillaries, 1-3-4-10 after 3, and
    the two parallel capillaries 4-11 after 3 each: orders 2, 3, 3, 3. Every
    vessel is 10 um except one of the parallel pair, 25 um: routes of 20, 30,
    30 and 45 um."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    for u, v, label in ((1, 2, "B01"), (2, 9, "B02"), (1, 3, "B01"), (3, 4, "B02"), (4, 10, "B03")):
        _vessel(G, u, v, label)
    _vessel(G, 4, 11, "B03")
    _vessel(G, 4, 11, "B03", length=25.0)
    for end in (9, 10, 11):
        _vessel(G, end, 20, "Ven1")

    rows = _rows(G, [1], [9, 10, 11])

    assert rows["Capillaries Draining to a Venule"] == 4
    assert rows[f"{ORDERS}, Mean"] == pytest.approx(11 / 4)
    assert rows[f"{ORDERS}, Median"] == 3.0
    assert rows[f"{ORDERS}, SD"] == pytest.approx(np.std([2, 3, 3, 3], ddof=1))
    assert (rows[f"{ORDERS}, Min"], rows[f"{ORDERS}, Max"]) == (2, 3)
    assert _length(rows, "Mean") == pytest.approx(125 / 4)
    assert _length(rows, "Median") == 30.0
    assert _length(rows, "SD") == pytest.approx(np.std([20, 30, 30, 45], ddof=1))
    assert (_length(rows, "Min"), _length(rows, "Max")) == (20.0, 45.0)


def _short_and_long_route():
    """Inlet 0, arteriole 0-1, arteriole boundary node 1; two capillary routes
    into the venule boundary node 9 -- 1-2-9 and 1-3-4-5-6-9 -- and the
    venule 9-10 to the outlet 10. Ordered by ``graph.branch_order``."""
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 2), (2, 9), (1, 3), (3, 4), (4, 5), (5, 6), (6, 9), (9, 10)):
        G.add_edge(u, v, length=10.0)
    assign_vessel_branch_orders(
        G, [0], outlet_nodes=[10], arteriole_boundary_nodes=[1], venule_boundary_nodes=[9]
    )
    return G


def test_the_draining_capillary_is_counted_along_its_own_route_not_by_its_label():
    """Regression for reading the labels: the B orders are counted through
    the venule end, so 6-9, the fifth capillary of its route, is labelled B03
    (one past 9, reached along the short route). The route it ends holds five
    capillaries, and the short one two."""
    G = _short_and_long_route()
    assert [G[0][1][0]["branch_order"], G[9][10][0]["branch_order"]] == ["Art1", "Ven1"]
    assert [G[2][9][0]["branch_order"], G[6][9][0]["branch_order"]] == ["B02", "B03"]

    rows = _rows(G, [1], [9])

    assert (rows[f"{ORDERS}, Min"], rows[f"{ORDERS}, Max"]) == (2, 5)
    assert rows[f"{ORDERS}, Mean"] == 3.5
    assert rows[f"{ORDERS}, SD"] == pytest.approx(math.sqrt(4.5))
    assert (_length(rows, "Min"), _length(rows, "Max")) == (20.0, 50.0)


def test_the_shortest_route_by_length_need_not_have_the_fewest_capillaries():
    """Feed 1 to the venule end 9 through the one capillary 2-9 (5 um): 1-2
    is a single capillary of 100 um, 1-3-4-2 three of 10 um. The fewest
    capillaries is the long way round (2 orders, 105 um); the shortest route
    is 35 um, through four."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    _vessel(G, 1, 2, "B01", length=100.0)
    for u, v, label in ((1, 3, "B01"), (3, 4, "B02"), (4, 2, "B02")):
        _vessel(G, u, v, label)
    _vessel(G, 2, 9, "B02", length=5.0)
    _vessel(G, 9, 10, "Ven1")

    rows = _rows(G, [1], [9])

    assert rows["Capillaries Draining to a Venule"] == 1
    assert rows[f"{ORDERS}, Mean"] == 2.0
    assert _length(rows, "Mean") == _length(rows, "Min") == 35.0


@pytest.mark.parametrize("length", [None, 0.0, float("nan")])
def test_without_every_capillary_length_it_counts_orders_and_says_why_there_are_no_lengths(length):
    G, end = _chain(3)
    if length is None:
        del G[2][3][0]["length"]
    else:
        G[2][3][0]["length"] = length

    rows = _rows(G, [1], [end])

    assert rows[f"{ORDERS}, Mean"] == 3.0
    assert rows[LENGTH_STATUS] == "N/A (vessel lengths unavailable)"
    assert not any(name.startswith((f"{LENGTHS},", STRAIGHT, RATIO)) for name in rows)


# --- the straight line between a route's ends ---------------------------------------


def _positioned(G, positions):
    for node, (y, x) in positions.items():
        G.nodes[node]["pos"] = (0.0, float(y), float(x))
    return G


def test_a_winding_route_is_its_length_over_the_straight_line_between_its_ends():
    """Feed 1 at the origin, up 40 um to 2, across 30 to 3 and down 40 to the
    venule end 9: a 110 um route between ends 30 um apart."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    _vessel(G, 1, 2, "B01", length=40.0)
    _vessel(G, 2, 3, "B02", length=30.0)
    _vessel(G, 3, 9, "B03", length=40.0)
    _vessel(G, 9, 10, "Ven1")
    _positioned(G, {0: (-10, 0), 1: (0, 0), 2: (40, 0), 3: (40, 30), 9: (0, 30), 10: (-10, 30)})

    rows = _rows(G, [1], [9])

    assert _length(rows, "Mean") == 110.0
    assert _straight(rows, "Mean") == 30.0
    assert _ratio(rows, "Mean") == pytest.approx(110 / 30)


def test_the_straight_line_runs_from_the_feed_its_route_starts_at_not_the_nearest():
    """Venule end 9 at x = 30. Feed 1 (x = 0) reaches it along 30 um of straight
    capillary through 2; feed 5, only 5 um from it, along 200 um through 6.
    Each draining capillary's straight line runs from its own route's feed:
    30 um from 1 (ratio 1) and 5 um from 5 (ratio 40), though 5 is nearer 9
    than 1 is."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    _vessel(G, 0, 5, "Art2")
    _vessel(G, 1, 2, "B01", length=15.0)
    _vessel(G, 2, 9, "B02", length=15.0)
    _vessel(G, 5, 6, "B01", length=100.0)
    _vessel(G, 6, 9, "B02", length=100.0)
    _vessel(G, 9, 10, "Ven1")
    _positioned(G, {0: (-10, 0), 1: (0, 0), 2: (0, 15), 5: (0, 25), 6: (60, 28), 9: (0, 30), 10: (-10, 30)})

    rows = _rows(G, [1, 5], [9])

    assert (_length(rows, "Min"), _length(rows, "Max")) == (30.0, 200.0)
    assert (_straight(rows, "Min"), _straight(rows, "Max")) == (5.0, 30.0)
    assert (_ratio(rows, "Min"), _ratio(rows, "Max")) == (1.0, 40.0)
    assert _ratio(rows, "Mean") == 20.5


def test_without_a_position_at_a_routes_end_it_says_why_there_is_no_straight_line():
    G, end = _chain(3)
    del G.nodes[end]["pos"]

    rows = _rows(G, [1], [end])

    assert _length(rows, "Mean") == 30.0
    assert rows[STRAIGHT_STATUS] == "N/A (node positions unavailable)"
    assert not any(name.startswith((f"{STRAIGHT},", RATIO)) for name in rows)


def test_a_route_ending_where_it_started_has_no_ratio():
    """The venule end 4 lies at the feed's own point: a loop of three
    capillaries, 30 um long, 0 um from end to end."""
    G, end = _chain(3)
    G.nodes[end]["pos"] = G.nodes[1]["pos"]

    rows = _rows(G, [1], [end])

    assert _straight(rows, "Mean") == 0.0
    assert rows[RATIO_STATUS] == "N/A (no route's two ends are apart)"


def test_a_route_runs_through_capillaries_alone_and_past_no_venule_end():
    """Feed 1 drains into venule end 20 after 2 capillaries. Capillary 20-21-30
    runs on between venule ends 20 and 30: nothing from an arteriole reaches
    21 but through 20, so its two capillaries drain into a venule no arteriole
    reaches. The capillary 20-30 joins two venule ends and ends no route."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    for u, v, label in ((1, 2, "B01"), (2, 20, "B02"), (20, 21, "B03"), (21, 30, "B04"), (20, 30, "B03")):
        _vessel(G, u, v, label)
    _vessel(G, 20, 40, "Ven1")
    _vessel(G, 30, 40, "Ven1")

    rows = _rows(G, [1], [20, 30])

    assert rows["Capillaries Draining to a Venule"] == 3
    assert rows["Capillaries Draining to a Venule No Arteriole Reaches"] == 2
    assert rows[f"{ORDERS}, Mean"] == 2.0


def test_a_capillary_leaving_its_feed_straight_into_a_venule_is_one_order():
    G, end = _chain(1)
    _vessel(G, 1, 7, "B01")
    _vessel(G, 7, 8, "Ven1")

    rows = _rows(G, [1], [end, 7])

    assert rows[f"{ORDERS}, Mean"] == 1.0
    assert rows["Capillaries Draining to a Venule"] == 2


def test_a_node_given_as_feed_and_drain_is_a_feed():
    G, end = _chain(3)

    rows = _rows(G, [1], [1, end])

    assert rows[f"{ORDERS}, Mean"] == 3.0
    assert rows["Capillaries Draining to a Venule"] == 1


# --- when it cannot count --------------------------------------------------------


def test_without_arteriole_or_venule_orders_every_vessel_is_a_b_order_and_it_reports_why():
    """Inlets and outlets alone: B orders count every vessel from the inlets,
    the arteriole and the venule included, so there is nothing to lie between."""
    G = nx.MultiGraph()
    for u in range(4):
        G.add_edge(u, u + 1, length=10.0)
    assign_vessel_branch_orders(G, [0], outlet_nodes=[4])
    assert {data["branch_order"] for *_e, data in G.edges(data=True)} == {"B01", "B02", "B03", "B04"}

    rows = _rows(G, [0], [4])

    assert rows == {
        STATUS: "N/A (no arteriole or venule branch orders: the B orders count every vessel from the inlets)"
    }


@pytest.mark.parametrize(
    "case, reason",
    [
        ("unknown_classes", "no capillary branch orders"),
        ("no_feed", "no capillary feed nodes"),
        ("no_drain", "no capillary drains into a venule"),
        ("unreached", "no arteriole reaches a venule through capillaries"),
    ],
)
def test_it_reports_why_it_counted_no_route(case, reason):
    G, end = _chain(3)
    feeds, drains = [1], [end]
    if case == "unknown_classes":
        for *_e, data in G.edges(data=True):
            data["branch_order"] = "Unknown_Cap"
    elif case == "no_feed":
        feeds = [0]
    elif case == "no_drain":
        drains = [99]
    else:
        G.remove_edge(2, 3)

    rows = _rows(G, feeds, drains)

    assert rows[STATUS] == f"N/A ({reason})"
    assert not any(name.startswith(ORDERS) for name in rows)
    if case == "unreached":
        assert rows["Capillaries Draining to a Venule"] == 1
        assert rows["Capillaries Draining to a Venule No Arteriole Reaches"] == 1


def test_every_row_it_writes_has_a_glossary_term():
    G, end = _chain(3)
    cut, _end = _chain(3)
    cut.remove_edge(2, 3)
    unmeasured, _end = _chain(3)
    del unmeasured[2][3][0]["length"]
    unplaced, _end = _chain(3)
    del unplaced.nodes[end]["pos"]
    closed, _end = _chain(3)
    closed.nodes[end]["pos"] = closed.nodes[1]["pos"]
    written = {}
    for graph in (G, cut, unmeasured, unplaced, closed):
        written.update(_rows(graph, [1], [end]))
    assert len(written) == 26

    for name in written:
        term = glossary.term_for(name)
        assert term is not None, name
        if name.endswith(" Status"):
            unit = ""
        elif name.startswith((f"{LENGTHS},", f"{STRAIGHT},")):
            unit = "um"
        elif name.startswith(f"{RATIO},"):
            unit = "dimensionless"
        else:
            unit = "vessels"
        assert term.unit == unit, name


# --- as a statistics measure ------------------------------------------------------


def _statistics_rows(G, nodes):
    sections = compute_vessel_statistics_by_measure(
        G, enabled_measures=frozenset({"capillary_path_orders"}), **nodes
    )
    return sections["capillary_path_orders"][SECTION]


@pytest.mark.parametrize("bridge", [False, True])
def test_it_counts_from_where_capillaries_leave_and_join_the_large_tier(bridge):
    """The large-tier bed (see test_capillary_bed_joins): one capillary route
    leaves the large arteriole at 1 and drains straight into the large vein
    at 8 (1-5-8), another runs from the arteriole boundary node 3 to the
    venule boundary node 7 (3-4-7). The hand-offs 1 and 8 are read from the
    labels, with the bridge into the large arteriole contracted: two routes
    of two 10 um capillaries each, the bridge no part of either. Placed 10 um
    a node id along x, their straight lines run 1 to 8 (70 um) and 3 to 7
    (40 um): the bridge's centreline end 1, not its far end 11."""
    G, nodes = _large_tier_bed(bridge=bridge)
    for node in G:
        G.nodes[node]["pos"] = (0.0, 0.0, 10.0 * node)

    rows = _statistics_rows(G, nodes)

    assert rows["Capillaries Draining to a Venule"] == 2
    assert rows["Capillaries Draining to a Venule No Arteriole Reaches"] == 0
    assert rows[f"{ORDERS}, Mean"] == 2.0
    assert _length(rows, "Min") == _length(rows, "Max") == 20.0
    assert (_straight(rows, "Min"), _straight(rows, "Max")) == (40.0, 70.0)


def test_it_is_a_network_analysis_measure_after_the_bed_joins():
    position = STATISTIC_MEASURES.index("capillary_path_orders")
    assert STATISTIC_MEASURES[position - 1] == "capillary_bed_joins"
    assert "capillary_path_orders" in NETWORK_ANALYSIS_MEASURES

    G = _short_and_long_route()
    sections = compute_vessel_statistics_by_measure(
        G,
        enabled_measures=frozenset({"capillary_bed_joins", "capillary_path_orders", "transit_time"}),
        inlet_nodes=[0],
        outlet_nodes=[10],
        arteriole_boundary_nodes=[1],
        venule_boundary_nodes=[9],
    )

    assert list(sections) == ["capillary_bed_joins", "capillary_path_orders", "transit_time"]
    assert sections["capillary_path_orders"][SECTION][f"{ORDERS}, Mean"] == 3.5


def _hierarchical_line():
    """Inlet 0, arteriole 0-1, four capillaries 1..5, venule 5-6, outlet 6:
    positions and centrelines as a built network has them."""
    G = nx.MultiGraph()
    for node in range(7):
        G.add_node(node, pos=np.asarray((0.0, 0.0, 3.0 * node)))
    for u in range(6):
        G.add_edge(
            u, u + 1, key=0, length=3.0,
            voxels=[(0.0, 0.0, 3.0 * u), (0.0, 0.0, 3.0 * (u + 1))],
        )
    assign_vessel_branch_orders(
        G, [0], outlet_nodes=[6], arteriole_boundary_nodes=[1], venule_boundary_nodes=[5]
    )
    return G


def _statistics_csv_rows(tmp_path):
    with (tmp_path / "no_haemodynamics_statistics.csv").open(newline="", encoding="utf-8") as handle:
        return {row["Metric"]: row for row in csv.DictReader(handle)}


def test_the_export_stage_writes_it_to_the_statistics_csv_with_its_unit(tmp_path):
    _export_with_haemodynamics_off(
        tmp_path, graph=_hierarchical_line(), inlet_nodes=[0], outlet_nodes=[6],
        arteriole_boundary_nodes=[1], venule_boundary_nodes=[5],
        statistics_capillary_path_orders=True,
    )

    rows = _statistics_csv_rows(tmp_path)
    mean = rows[f"{ORDERS}, Mean"]
    assert (mean["Section"], mean["Value"], mean["Unit"]) == (SECTION, "4", "vessels")
    shortest = rows[f"{LENGTHS}, Min"]
    assert (shortest["Section"], shortest["Value"], shortest["Unit"]) == (SECTION, "12", "um")
    straight = rows[f"{STRAIGHT}, Min"]
    assert (straight["Section"], straight["Value"], straight["Unit"]) == (SECTION, "12", "um")
    ratio = rows[f"{RATIO}, Mean"]
    assert (ratio["Section"], ratio["Value"], ratio["Unit"]) == (SECTION, "1", "dimensionless")
    assert rows["Capillaries Draining to a Venule"]["Value"] == "1"


def test_the_export_stage_skips_it_when_toggled_off(tmp_path):
    _export_with_haemodynamics_off(
        tmp_path, graph=_hierarchical_line(), inlet_nodes=[0], outlet_nodes=[6],
        arteriole_boundary_nodes=[1], venule_boundary_nodes=[5],
        statistics_capillary_path_orders=False,
    )

    rows = _statistics_csv_rows(tmp_path)
    assert "Capillary Bed Join Order, Kaplan-Meier Median" in rows
    assert not any(
        name.startswith((ORDERS, LENGTHS, STRAIGHT, RATIO, "Capillaries Draining", "Capillary Path"))
        for name in rows
    )
