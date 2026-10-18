"""Where the capillary beds of different feeds join, on graphs with
hand-computed answers, and its Kaplan-Meier summary and stage wiring."""
from __future__ import annotations

import csv
import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.branch_order import (
    LARGE_MASK_VESSEL_TYPE,
    assign_branch_orders,
    assign_vessel_branch_orders,
    capillary_start_nodes,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.statistics import (
    STATISTIC_MEASURES,
    compute_vessel_statistics_by_measure,
)
from haemolynx.statistics.capillary_bed_joins import (
    BED_JOIN_EDGE_ATTRIBUTES,
    BED_TABLE,
    BED_TABLE_COLUMNS,
    SECTION,
    compute_capillary_bed_joins,
    export_capillary_bed_joins_csv,
    kaplan_meier,
    kaplan_meier_median,
)
from test_statistics_without_haemodynamics import _export_with_haemodynamics_off


def _vessel(G, u, v, length=10.0, **extra):
    G.add_edge(u, v, length=float(length), **extra)


def _labelled(G, feeds):
    """*G* with the B orders ``graph.branch_order`` counts from *feeds*."""
    assign_branch_orders(G, feeds)
    return G


def _rows(result):
    return result[SECTION]


def _table(G):
    return {row["bed"]: row for row in G.graph[BED_TABLE]}


def _ladder(segments):
    """Two rails of *segments* 10 um vessels, joined by a rung at every node;
    rail 0 runs 0..segments, rail 1 runs 100..100+segments."""
    G = nx.MultiGraph()
    for i in range(segments):
        _vessel(G, i, i + 1)
        _vessel(G, 100 + i, 101 + i)
    for i in range(segments + 1):
        _vessel(G, i, 100 + i)
    return G


# --- the join order --------------------------------------------------------------


@pytest.mark.parametrize("segments", [1, 2, 3, 4, 5, 6, 7])
def test_two_feeds_on_a_ladder_join_halfway_between_them(segments):
    """Feeds *segments* vessels apart along one rail: their beds meet at the
    vessel halfway, B order ceil(segments / 2)."""
    G = _labelled(_ladder(segments), [0, segments])

    rows = _rows(compute_capillary_bed_joins(G, [0, segments]))

    expected = math.ceil(segments / 2)
    assert rows["Capillary Beds"] == 2
    assert rows["Capillary Beds Joined"] == 2
    assert rows["Capillary Bed Join Order, Kaplan-Meier Median"] == expected
    table = _table(G)
    assert [table[bed]["join_order"] for bed in ("F01", "F02")] == [expected, expected]
    assert table["F01"]["joins_bed"] == "F02"
    assert not table["F01"]["censored"]
    beds = {data["capillary_bed"] for *_e, data in G.edges(data=True)}
    assert beds == {"F01", "F02"}
    assert all(data["bed_join_order"] == expected for *_e, data in G.edges(data=True))


def _uneven_chain():
    """Feeds 0 and 4 four vessels apart; the vessel off feed 0 is 100 um, the
    other three 10 um. By hops node 2 is halfway; by length node 1 is nearer
    feed 4 (30 um) than feed 0 (100 um)."""
    G = nx.MultiGraph()
    for (u, v), length in zip(((0, 1), (1, 2), (2, 3), (3, 4)), (100.0, 10.0, 10.0, 10.0)):
        _vessel(G, u, v, length)
    return _labelled(G, [0, 4])


def test_hop_and_length_weighting_put_the_watershed_in_different_places():
    hops = _uneven_chain()
    by_hops = _rows(compute_capillary_bed_joins(hops, [0, 4], weighting="topology"))
    lengths = _uneven_chain()
    by_length = _rows(compute_capillary_bed_joins(lengths, [0, 4], weighting="length"))

    assert by_hops["Capillary Bed Join Route Weighting"] == "topology"
    assert by_length["Capillary Bed Join Route Weighting"] == "length"
    # Hops: the beds meet on 2-3, a B02; by length on 0-1, a B01.
    assert by_hops["Capillary Bed Join Order, Kaplan-Meier Median"] == 2
    assert by_length["Capillary Bed Join Order, Kaplan-Meier Median"] == 1
    assert hops[1][2][0]["capillary_bed"] == "F01"
    assert lengths[1][2][0]["capillary_bed"] == "F02"
    # Half the 130 um feed-to-feed path either way.
    assert _table(hops)["F01"]["join_distance_um"] == pytest.approx(65.0)
    assert _table(lengths)["F01"]["join_distance_um"] == pytest.approx(65.0)


def test_a_bridge_is_no_step():
    """Feed 9 opens into the bed through a 5 um zero-resistance bridge to node
    1. As no step, node 2 is one vessel from feed 9 and two from feed 4, so the
    beds meet on 2-3 (a B02); counted as a step, node 2 would be equidistant
    and the watershed slide to the B01 at 1-2."""
    G = nx.MultiGraph()
    G.add_edge(9, 1, length=5.0, **{IS_ZERO_RESISTANCE: True})
    for u, v in ((1, 2), (2, 3), (3, 4)):
        _vessel(G, u, v)
    _labelled(G, [9, 4])

    rows = _rows(compute_capillary_bed_joins(G, [9, 4]))

    assert rows["Capillary Bed Join Order, Kaplan-Meier Median"] == 2
    table = _table(G)
    assert table["F02"]["feed_node"] == 9
    assert table["F02"]["join_order"] == 2
    # Nodes 2 and 3 are each 10 um from their feed: the bridge adds nothing.
    assert table["F02"]["join_distance_um"] == pytest.approx(15.0)
    assert G[1][2][0]["capillary_bed"] == "F02"
    assert G[3][4][0]["capillary_bed"] == "F01"
    assert not any(name in G[9][1][0] for name in BED_JOIN_EDGE_ATTRIBUTES)


# --- censoring ---------------------------------------------------------------------


def _three_unjoined_beds():
    """One arteriolar tree (inlet 100) feeding 0 and 10, and feed 40 on none.
    Feed 0's bed runs to a dead end at the image face (node 2), feed 10's
    drains into venule node 12, feed 40's ends blind at node 41. No capillary
    joins another feed's."""
    G = nx.MultiGraph()
    _vessel(G, 100, 101)
    _vessel(G, 101, 0)
    _vessel(G, 101, 10)
    for u, v in ((0, 1), (1, 2), (1, 3), (10, 11), (11, 12), (40, 41)):
        _vessel(G, u, v)
    _vessel(G, 12, 30)
    _labelled(G, [0, 10, 40])
    for u, v, label in ((100, 101, "Art1"), (101, 0, "Art2"), (101, 10, "Art2"), (12, 30, "Ven1")):
        G[u][v][0]["branch_order"] = label
    G.nodes[2]["dead_end_kind"] = "leaves_image"
    G.nodes[3]["dead_end_kind"] = "blind"
    return G


def test_a_tree_with_no_anastomosis_has_every_bed_censored_with_its_reason():
    G = _three_unjoined_beds()

    rows = _rows(compute_capillary_bed_joins(G, [0, 10, 40], drain_nodes=[12]))

    assert rows["Capillary Beds"] == 3
    assert rows["Capillary Beds Joined"] == 0
    assert rows["Capillary Beds Joined (fraction)"] == 0.0
    assert rows["Capillary Beds Censored, Leaving the Image"] == 1
    assert rows["Capillary Beds Censored, Draining to a Venule"] == 1
    assert rows["Capillary Beds Censored, Ending Blind"] == 1
    assert rows["Capillary Bed Join Order, Kaplan-Meier Median"].startswith("N/A")
    table = _table(G)
    assert [table[bed]["censor_reason"] for bed in ("F01", "F02", "F03")] == [
        "leaves_image",
        "reaches_venule",
        "blind_end",
    ]
    assert all(row["censored"] for row in table.values())
    # Censored at the highest B order each bed was followed to.
    assert [table[bed]["join_order"] for bed in ("F01", "F02", "F03")] == [2, 2, 1]
    assert [table[bed]["arteriolar_tree"] for bed in ("F01", "F02", "F03")] == ["T01", "T01", ""]
    assert all(np.isnan(data["bed_join_order"]) for *_e, data in G.edges(data=True) if "bed_join_order" in data)
    # Only capillaries belong to a bed.
    assert "capillary_bed" not in G[101][0][0]
    assert "capillary_bed" not in G[12][30][0]


def test_a_bed_that_both_leaves_the_image_and_drains_is_censored_by_the_image():
    """It may yet join another beyond the field of view."""
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 2), (1, 3)):
        _vessel(G, u, v)
    _labelled(G, [0])
    G.nodes[2]["dead_end_kind"] = "leaves_image"

    compute_capillary_bed_joins(G, [0], drain_nodes=[3])

    assert _table(G)["F01"]["censor_reason"] == "leaves_image"


def test_beds_meeting_only_at_a_venule_do_not_join():
    """Two feeds' capillaries drain into the same venule node: blood meets in
    the venule, not in the capillary bed."""
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 5), (4, 3), (3, 5)):
        _vessel(G, u, v)
    _labelled(G, [0, 4])

    rows = _rows(compute_capillary_bed_joins(G, [0, 4], drain_nodes=[5]))

    assert rows["Capillary Beds Joined"] == 0
    assert rows["Capillary Beds Censored, Draining to a Venule"] == 2


# --- arteriolar trees -------------------------------------------------------------


def test_joins_are_classed_within_or_between_arteriolar_trees():
    """Inlet 100's tree feeds 0 and 10, inlet 200's feeds 20. Beds 0 and 10
    meet at B01 (within one tree); bed 20 meets bed 10 at B02 (between)."""
    G = nx.MultiGraph()
    for u, v in ((100, 101), (101, 0), (101, 10), (200, 20)):
        _vessel(G, u, v)
    for u, v in ((0, 1), (1, 10), (10, 11), (11, 12), (12, 20)):
        _vessel(G, u, v)
    _labelled(G, [0, 10, 20])
    for u, v in ((100, 101), (101, 0), (101, 10), (200, 20)):
        G[u][v][0]["branch_order"] = "Art1"

    rows = _rows(compute_capillary_bed_joins(G, [0, 10, 20]))

    table = _table(G)
    assert [table[bed]["arteriolar_tree"] for bed in ("F01", "F02", "F03")] == ["T01", "T01", "T02"]
    assert [table[bed]["join_order"] for bed in ("F01", "F02", "F03")] == [1, 1, 2]
    assert [table[bed]["joins_bed"] for bed in ("F01", "F02", "F03")] == ["F02", "F01", "F02"]
    assert [table[bed]["join_class"] for bed in ("F01", "F02", "F03")] == [
        "within_tree",
        "within_tree",
        "between_trees",
    ]
    assert rows["Capillary Beds Joining Within Their Arteriolar Tree"] == 2
    assert rows["Capillary Beds Joining Another Arteriolar Tree"] == 1


# --- Kaplan-Meier ----------------------------------------------------------------


def test_kaplan_meier_on_a_hand_computed_censored_example():
    """Joins at 1, 2, 3 and 5; censored at 2 and 4. At risk 6, 5, 3, 1: the
    fraction not yet joined is 5/6, 2/3, 4/9, then 0 -- under a half first at 3."""
    times = [1, 2, 2, 3, 4, 5]
    joined = [True, False, True, True, False, True]

    event_times, survival = kaplan_meier(times, joined)

    assert list(event_times) == [1, 2, 3, 5]
    assert survival == pytest.approx([5 / 6, 2 / 3, 4 / 9, 0.0])
    assert kaplan_meier_median(times, joined) == 3


def test_kaplan_meier_median_is_nan_when_fewer_than_half_join():
    assert np.isnan(kaplan_meier_median([1, 2, 3, 4], [True, False, False, False]))
    assert np.isnan(kaplan_meier_median([], []))


# --- what it needs, and where it is wired --------------------------------------


def test_without_capillary_orders_or_feeds_it_reports_why_and_writes_nothing(tmp_path):
    G = _ladder(3)
    rows = _rows(compute_capillary_bed_joins(G, [0, 3]))
    assert rows["Capillary Bed Join Status"] == "N/A (no capillary branch orders)"

    _labelled(G, [0, 3])
    rows = _rows(compute_capillary_bed_joins(G, [999]))
    assert rows["Capillary Bed Join Status"] == "N/A (no capillary feed nodes)"

    assert not any(name in data for *_e, data in G.edges(data=True) for name in BED_JOIN_EDGE_ATTRIBUTES)
    assert BED_TABLE not in G.graph
    assert export_capillary_bed_joins_csv(G, tmp_path / "beds.csv") is None
    assert not (tmp_path / "beds.csv").exists()


def test_the_per_bed_csv_has_one_row_per_bed_and_leaves_the_graph(tmp_path):
    G = _labelled(_ladder(4), [0, 4])
    compute_capillary_bed_joins(G, [0, 4])

    path = export_capillary_bed_joins_csv(G, tmp_path / "beds.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == BED_TABLE_COLUMNS
    assert [row["bed"] for row in rows] == ["F01", "F02"]
    assert [row["join_order"] for row in rows] == ["2", "2"]
    assert [row["censored"] for row in rows] == ["False", "False"]
    assert BED_TABLE not in G.graph


def test_capillary_start_nodes_follow_the_branch_order_mode():
    """Hierarchical (arteriole and venule boundaries and outlets): the
    boundaries; otherwise the inlets and outlets."""
    assert capillary_start_nodes([1], [2], [3], [4]) == ([3], [4])
    assert capillary_start_nodes([1], [2], [3], []) == ([1], [2])
    assert capillary_start_nodes([1], [], [3], [4]) == ([1], [])


# --- a Large_Art/Large_Ven tier ---------------------------------------------------


def _large_tier_bed(rung=True, bridge=False):
    """A large arteriole 0-1-2 (inlet 0) and a large vein 8-9-10 (outlet 10),
    tagged in their masks and run with a Large tier. A capillary leaves the
    large arteriole at 1 and drains straight into the large vein at 8, through
    5; a small arteriole leaves at 2 and meets the bed at its boundary node 3,
    whose capillaries 3-4-7 drain through a small venule (boundary 7) into the
    vein at 8. *rung* adds the capillary 5-4 between the two; *bridge* has the
    first capillary open into the large arteriole through a zero-resistance
    bridge 1-11, as a vessel opening into a thick one does.

    Ordered by ``graph.branch_order``: B counts from 3 and from the hand-off
    node 1, so 1-5 and 3-4 are B01, and 5-8, 4-7 and the rung B02.
    """
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 2)):
        _vessel(G, u, v, **{LARGE_MASK_VESSEL_TYPE: "arteriole"})
    for u, v in ((8, 9), (9, 10)):
        _vessel(G, u, v, **{LARGE_MASK_VESSEL_TYPE: "venule"})
    first = 1
    if bridge:
        G.add_edge(1, 11, length=5.0, **{IS_ZERO_RESISTANCE: True})
        first = 11
    for u, v in ((first, 5), (5, 8), (2, 3), (3, 4), (4, 7), (7, 8)):
        _vessel(G, u, v)
    if rung:
        _vessel(G, 5, 4)
    nodes = dict(
        inlet_nodes=[0],
        outlet_nodes=[10],
        arteriole_boundary_nodes=[3],
        venule_boundary_nodes=[7],
    )
    assign_vessel_branch_orders(
        G, **nodes, large_arteriole_boundary_nodes=[1, 2], large_venule_boundary_nodes=[8]
    )
    return G, nodes


def _statistics_rows(G, nodes):
    """The bed joins as the statistics compute them from the run's terminals
    and boundary nodes."""
    sections = compute_vessel_statistics_by_measure(
        G,
        enabled_measures=frozenset({"capillary_bed_joins"}),
        route_weighting="topology",
        **nodes,
    )
    return sections["capillary_bed_joins"][SECTION]


def test_capillary_start_nodes_add_where_capillaries_leave_and_join_the_large_tier():
    """Given the labelled graph, the feeds are also where a capillary leaves
    the Large_Art tier (1) and the drains where one joins the Large_Ven tier
    (8). Node 2, where only the small arteriole leaves, feeds no capillary."""
    G, nodes = _large_tier_bed()
    terminals = [
        nodes[name]
        for name in ("inlet_nodes", "outlet_nodes", "arteriole_boundary_nodes", "venule_boundary_nodes")
    ]
    assert [G[1][2][0]["branch_order"], G[1][5][0]["branch_order"], G[8][9][0]["branch_order"]] == [
        "Large_Art2", "B01", "Large_Ven2",
    ]

    assert capillary_start_nodes(*terminals, graph=G) == ([3, 1], [7, 8])
    assert capillary_start_nodes(*terminals) == ([3], [7])
    # B orders from the inlets alone count from the inlets, Large tier or not.
    assert capillary_start_nodes([0], [10], [3], [], graph=G) == ([0], [10])


def test_a_capillary_leaving_the_large_arteriole_feeds_a_bed_of_its_own():
    """Regression: the statistics fed the beds from the arteriole boundary
    nodes alone, so the capillary leaving the large arteriole at 1 -- B01 from
    it -- had no feed; bed 3 took it from its far end, and no bed joined
    another. Fed from 1 as well, the two beds meet on the rung, a B02."""
    G, nodes = _large_tier_bed()

    rows = _statistics_rows(G, nodes)

    assert rows["Capillary Beds"] == 2
    assert rows["Capillary Beds Joined"] == 2
    assert rows["Capillary Bed Join Order, Kaplan-Meier Median"] == 2
    table = _table(G)
    assert [table[bed]["feed_node"] for bed in ("F01", "F02")] == [1, 3]
    assert [table[bed]["joins_bed"] for bed in ("F01", "F02")] == ["F02", "F01"]
    # Both hang from the one arterial tree, the large arteriole and its branch.
    assert [table[bed]["join_class"] for bed in ("F01", "F02")] == ["within_tree"] * 2
    assert G[1][5][0]["capillary_bed"] == G[5][8][0]["capillary_bed"] == "F01"
    assert G[3][4][0]["capillary_bed"] == G[4][7][0]["capillary_bed"] == "F02"
    assert "Capillaries No Feed Reaches" not in rows


def test_a_capillary_draining_straight_into_the_large_vein_drains_its_bed():
    """With no rung, bed 1 runs straight into the large vein at 8. With only
    the venule boundary nodes as drains it would end blind there; it drains
    to a venule, as bed 3 does through the small venule."""
    G, nodes = _large_tier_bed(rung=False)

    rows = _statistics_rows(G, nodes)

    assert rows["Capillary Beds"] == 2
    assert rows["Capillary Beds Joined"] == 0
    assert rows["Capillary Beds Censored, Draining to a Venule"] == 2
    assert rows["Capillary Beds Censored, Ending Blind"] == 0
    assert _table(G)["F01"]["feed_node"] == 1


def test_the_hand_off_is_read_with_the_bridge_into_the_large_arteriole_contracted():
    """The capillary opens into the large arteriole through the bridge 1-11.
    The statistics see the bridge's two ends as one node, 1 (the centreline
    end), where a Large_Art and a B vessel meet: it feeds the bed."""
    G, nodes = _large_tier_bed(bridge=True)
    assert G[11][5][0]["branch_order"] == "B01"

    rows = _statistics_rows(G, nodes)

    assert rows["Capillary Beds"] == 2
    assert rows["Capillary Beds Joined"] == 2
    assert _table(G)["F01"]["feed_node"] == 1
    assert G[11][5][0]["capillary_bed"] == "F01"
    assert "capillary_bed" not in G[1][11][0]


def test_without_a_large_tier_the_beds_are_the_boundary_nodes_alone():
    """The same orders with the large vessels labelled plain Art and Ven, as
    a run without a Large tier has them: no hand-off is read, and the beds are
    those of the arteriole and venule boundary nodes alone, as before."""
    G, nodes = _large_tier_bed()
    for *_e, data in G.edges(data=True):
        data["branch_order"] = data["branch_order"].replace("Large_", "")
    expected = G.copy()
    compute_capillary_bed_joins(expected, [3], drain_nodes=[7])

    _statistics_rows(G, nodes)

    assert G.graph[BED_TABLE] == expected.graph[BED_TABLE]
    assert [row["feed_node"] for row in G.graph[BED_TABLE]] == [3]
    assert {d["capillary_bed"] for *_e, d in G.edges(data=True) if "capillary_bed" in d} == {"F01"}


def test_it_runs_as_a_statistics_measure_from_the_arteriole_boundaries():
    G = _labelled(_ladder(4), [0, 4])
    sections = compute_vessel_statistics_by_measure(
        G,
        enabled_measures=frozenset({"capillary_bed_joins"}),
        inlet_nodes=[0],
        outlet_nodes=[104],
        arteriole_boundary_nodes=[0, 4],
        venule_boundary_nodes=[104],
        route_weighting="topology",
        short_loop_removal_um=200.0,
    )

    rows = sections["capillary_bed_joins"][SECTION]
    assert rows["Capillary Beds"] == 2
    assert rows["Graph Built With Short Loops Removed Under (microns)"] == 200.0
    assert "capillary_bed_joins" in STATISTIC_MEASURES


def test_turned_off_it_computes_and_writes_nothing():
    G = _labelled(_ladder(4), [0, 4])
    sections = compute_vessel_statistics_by_measure(
        G,
        enabled_measures=frozenset({"basic", "perfusion_territories"}),
        inlet_nodes=[0, 4],
        outlet_nodes=[104],
    )

    assert "capillary_bed_joins" not in sections
    assert not any(name in data for *_e, data in G.edges(data=True) for name in BED_JOIN_EDGE_ATTRIBUTES)
    assert BED_TABLE not in G.graph


def test_its_vessel_columns_are_declared_for_the_vessels_layer():
    from haemolynx.gui.results import OPTIONAL_EDGE_COLUMNS, TEXT_COLUMNS, edge_columns_for_settings

    for name in BED_JOIN_EDGE_ATTRIBUTES:
        assert OPTIONAL_EDGE_COLUMNS[name] == "export_results"
    assert "capillary_bed" in TEXT_COLUMNS
    assert "bed_join_order" not in TEXT_COLUMNS
    assert set(BED_JOIN_EDGE_ATTRIBUTES) <= set(edge_columns_for_settings())


def _two_feed_chain():
    """Inlets 0 and 3 three vessels apart on a straight line, for the export
    stage: positions and centrelines as a built network has them."""
    G = nx.MultiGraph()
    for node in range(4):
        G.add_node(node, pos=np.asarray((0.0, 0.0, 3.0 * node)))
    for u in range(3):
        G.add_edge(
            u, u + 1, key=0, length=3.0,
            voxels=[(0.0, 0.0, 3.0 * u), (0.0, 0.0, 3.0 * (u + 1))],
        )
    return _labelled(G, [0, 3])


def test_the_export_stage_writes_the_per_bed_csv_beside_the_statistics_csv(tmp_path):
    G = _export_with_haemodynamics_off(
        tmp_path, graph=_two_feed_chain(), inlet_nodes=[0, 3], outlet_nodes=[],
        statistics_capillary_bed_joins=True, statistics_route_weighting="topology",
    )

    stats_csv = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert "Capillary Bed Join Order" in stats_csv
    assert "Short Loops Removed Under" in stats_csv
    with (tmp_path / "no_haemodynamics_capillary_bed_joins.csv").open(newline="") as handle:
        beds = list(csv.DictReader(handle))
    assert [row["join_order"] for row in beds] == ["2", "2"]
    assert all(data["capillary_bed"] in {"F01", "F02"} for *_e, data in G.edges(data=True))
    assert BED_TABLE not in G.graph


def test_the_export_stage_skips_it_when_toggled_off(tmp_path):
    G = _export_with_haemodynamics_off(
        tmp_path, graph=_two_feed_chain(), inlet_nodes=[0, 3], outlet_nodes=[],
        statistics_capillary_bed_joins=False,
    )

    assert "Capillary Bed" not in (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert not (tmp_path / "no_haemodynamics_capillary_bed_joins.csv").exists()
    assert not any(name in data for *_e, data in G.edges(data=True) for name in BED_JOIN_EDGE_ATTRIBUTES)
