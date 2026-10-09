"""``graph.network_review``: what each review list of the Post processing tab
finds, on one small synthetic network per list.

Each finder is read-only and returns its items in the order the list shows
them; these tests pin which items, in what order, and why.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import network_review as nr
from haemolynx.graph.facing_ends import REFUSED_GAP_TOO_WIDE
from haemolynx.graph.prune import FLOW_SOLVED
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics.poiseuille import FWHM_DEMOTED_ATTR
from haemolynx.preprocessing import MaskSupport

from lumen_artefact_fixtures import VOXEL_SIZE, dead_end_cases, two_strands_in_one_lumen


def _vessel(G, u, v, *, points=None, **attrs):
    """A vessel from u to v along *points* (default straight), its length measured."""
    path = np.asarray(points if points is not None else [G.nodes[u]["pos"], G.nodes[v]["pos"]], dtype=float)
    attrs.setdefault("length", float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    return G.add_edge(u, v, voxels=path.tolist(), **attrs)


def _graph(positions) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    return G


def _unchanged(G):
    return sorted(map(str, G.nodes(data="pos"))), sorted(map(str, G.edges(keys=True, data=True)))


# --- junctions close together -------------------------------------------------


def _two_junctions(apart_um: float, **connector):
    G = _graph({
        "a": (0, 0, 0), "b": (0, 0, apart_um),
        "a1": (0, 20, -10), "a2": (0, -20, -10), "b1": (0, 20, apart_um + 10), "b2": (0, -20, apart_um + 10),
    })
    for u, v in (("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2")):
        _vessel(G, u, v, diameter_um=6.0)
    _vessel(G, "a", "b", diameter_um=6.0, **connector)
    return G


def test_two_junctions_within_a_vessel_radius_are_one_junction_drawn_as_two():
    G = _two_junctions(1.0)
    before = _unchanged(G)

    (item,) = nr.find_close_junctions(G)

    assert _unchanged(G) == before
    assert item.kind == nr.CLOSE_JUNCTIONS
    assert set(item.nodes) == {"a", "b"}
    assert item.edges[0] in {("a", "b", 0), ("b", "a", 0)}
    assert len(item.edges) == 5
    assert item.measures["vessels"] == 4
    assert item.measures["connector_um"] == pytest.approx(1.0)
    assert item.centre_um == pytest.approx((0.0, 0.0, 0.5))


def test_junctions_further_apart_than_the_radius_or_joined_by_a_bridge_are_not_listed():
    assert nr.find_close_junctions(_two_junctions(10.0)) == []
    assert nr.find_close_junctions(_two_junctions(1.0, **{IS_ZERO_RESISTANCE: True})) == []


def test_close_junctions_are_listed_closest_first():
    G = _two_junctions(2.0)
    H = _two_junctions(1.0)
    G = nx.disjoint_union(G, H)
    distances = [item.measures["connector_um"] for item in nr.find_close_junctions(G)]
    assert distances == pytest.approx([1.0, 2.0])


# --- hairpins ------------------------------------------------------------------------


def test_a_vessel_doubling_back_is_a_hairpin_and_a_straight_or_short_one_is_not():
    G = _graph({0: (0, 0, 0), 1: (0, 5, 0), 2: (0, 0, 50), 3: (0, 50, 50), 4: (0, 2, 60), 5: (0, 2, 61)})
    _vessel(G, 0, 1, points=[(0, 0, 0), (0, 0, 15), (0, 5, 15), (0, 5, 0)])  # 35 um, ends 5 um apart
    _vessel(G, 2, 3)  # straight
    _vessel(G, 4, 5, points=[(0, 2, 60), (0, 2, 63), (0, 3, 63), (0, 2, 61)])  # doubles back, but 6 um

    (item,) = nr.find_hairpins(G)

    assert item.nodes == (0, 1)
    assert item.measures["ratio"] == pytest.approx(7.0)
    assert item.measures["chord_um"] == pytest.approx(5.0)
    assert len(item.points_um) == 4


# --- diameter jumps ----------------------------------------------------------------


def test_a_measured_vessel_running_on_into_one_three_times_as_wide_is_a_jump():
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 0, 20), 3: (0, 0, 30), 4: (0, 0, 40)})
    _vessel(G, 0, 1, diameter_um=4.0, diameter_source="measured")
    _vessel(G, 1, 2, diameter_um=12.0, diameter_source="measured")
    _vessel(G, 2, 3, diameter_um=13.0, diameter_source="table")  # a table width says nothing
    _vessel(G, 3, 4, diameter_um=4.0, diameter_source="measured")

    (item,) = nr.find_diameter_jumps(G)

    assert item.nodes == (1,)
    assert item.measures["ratio"] == pytest.approx(3.0)
    assert item.measures["at"] == "pass-through"
    assert item.centre_um == pytest.approx((0.0, 0.0, 10.0))


def test_a_daughter_wider_than_its_parent_is_a_jump():
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 10, 20), 3: (0, -10, 20)})
    _vessel(G, 0, 1, diameter_um=5.0, diameter_source="measured", branch_order="Art1")
    _vessel(G, 1, 2, diameter_um=9.0, diameter_source="measured", branch_order="Art2")
    _vessel(G, 1, 3, diameter_um=4.0, diameter_source="measured", branch_order="Art2")

    (item,) = nr.find_diameter_jumps(G)

    assert item.nodes == (1,)
    assert item.measures["at"] == "junction"
    assert item.measures["ratio"] == pytest.approx(1.8)
    assert len(item.edges) == 2
    assert "Art1" in item.why


# --- unmeasured diameters -------------------------------------------------------


def test_widths_not_measured_or_in_doubt_are_listed_by_their_share_of_the_flow():
    G = _graph({n: (0, 0, 10 * n) for n in range(6)})
    G.nodes[0]["pressure"] = 100.0
    for n in range(1, 6):
        G.nodes[n]["pressure"] = 100.0 - 10 * n
    _vessel(G, 0, 1, diameter_um=5.0, diameter_source="measured", flow_abs=10.0)
    _vessel(G, 1, 2, diameter_um=5.0, diameter_source="table", flow_abs=2.0)
    _vessel(G, 2, 3, diameter_um=5.0, diameter_source="class_median", flow_abs=8.0)
    _vessel(G, 3, 4, diameter_um=5.0, diameter_source="edt_mask", flow_abs=5.0, **{FWHM_DEMOTED_ATTR: "speck width"})
    _vessel(G, 4, 5, diameter_um=5.0, diameter_source="measured", flow_abs=1.0, fwhm_low_confidence_vs_edt=True)

    items = nr.find_unmeasured_diameters(G, inlets=[0])

    assert [item.nodes for item in items] == [(2, 3), (3, 4), (1, 2), (4, 5)]
    assert [item.measures["flow_share"] for item in items] == pytest.approx([0.8, 0.5, 0.2, 0.1])
    assert items[0].why == "Its width is its class's median width."
    assert items[1].why.startswith("FWHM set aside (speck width)")
    assert items[3].measures["reasons"] == "FWHM at odds with the mask's width"


def test_an_unsolved_vessel_keeps_its_place_at_the_end_with_no_share():
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 0, 20)})
    _vessel(G, 0, 1, diameter_um=5.0, diameter_source="table", flow_abs=1.0, **{FLOW_SOLVED: False})
    _vessel(G, 1, 2, diameter_um=5.0, diameter_source="table", flow_abs=3.0)
    items = nr.find_unmeasured_diameters(G)
    assert [item.nodes for item in items] == [(1, 2), (0, 1)]
    assert items[1].measures["flow_share"] is None


# --- unsolved pieces ------------------------------------------------------------------


def test_unsolved_pieces_are_items_saying_where_they_hang():
    G = _graph({n: (0, 0, 10 * n) for n in range(4)} | {9: (0, 20, 10), 20: (0, 50, 0), 21: (0, 50, 10)})
    for u, v in ((0, 1), (1, 2), (2, 3), (1, 9), (20, 21)):
        _vessel(G, u, v)

    tree, island = nr.find_unsolved_pieces(G, [0], [3])

    assert tree.nodes == (1,) and "hangs off node(s) 1" in tree.why
    assert island.nodes == () and "a piece of its own" in island.why
    assert island.measures["disconnected"] is True


# --- flow against the branch orders -------------------------------------------------


def _chain(orders, pressures, flows=None):
    """A chain of vessels 0-1-2-..., each labelled, the nodes at *pressures*."""
    G = _graph({n: (0, 0, 10 * n) for n in range(len(pressures))})
    for n, p in enumerate(pressures):
        G.nodes[n]["pressure"] = float(p)
    for n, order in enumerate(orders):
        flow = 1.0 if flows is None else flows[n]
        _vessel(G, n, n + 1, branch_order=order, flow_abs=flow, diameter_um=5.0)
    return G


def test_flow_from_arteriole_to_capillary_to_venule_is_in_order():
    G = _chain(["Art1", "Art2", "B01", "Ven2", "Ven1"], [100, 90, 80, 70, 60, 50])
    assert nr.find_flow_against_branch_order(G) == []


def test_flow_from_a_capillary_back_into_an_arteriole_is_listed():
    G = _chain(["Art1", "B01", "Art2", "B01"], [100, 90, 80, 70, 60])

    (item,) = nr.find_flow_against_branch_order(G, inlets=[0])

    assert item.nodes == (2,)
    assert item.why == "Flow runs from B01 into Art2, back towards the arterial side."
    assert item.measures["arteriole_meets_venule"] is False


def test_the_direction_is_read_from_node_pressures_not_from_how_the_edge_is_named():
    # The same chain as above, the pressures reversed: blood now runs 4 -> 0,
    # so B01 at 3-4 feeds Art2... and it is still Art2 being fed by a capillary.
    G = _chain(["B01", "Art2", "B01", "Art1"], [60, 70, 80, 90, 100])
    (item,) = nr.find_flow_against_branch_order(G)
    assert item.nodes == (2,)


def test_an_arteriole_meeting_a_venule_with_no_capillary_is_listed():
    G = _chain(["Art1", "Ven1"], [100, 90, 80])
    (item,) = nr.find_flow_against_branch_order(G)
    assert item.nodes == (1,)
    assert item.measures["arteriole_meets_venule"] is True


def test_unsolved_vessels_are_not_judged_for_direction():
    G = _chain(["Art1", "B01", "Art2", "B01"], [100, 90, 80, 70, 60])
    for _u, _v, data in G.edges(data=True):
        data[FLOW_SOLVED] = False
    assert nr.find_flow_against_branch_order(G) == []


# --- flow outliers ------------------------------------------------------------------------


#: Capillary flows a few percent apart, and one 15% off: alike, not outliers.
_ALIKE = (1.0, 1.05, 0.95, 1.1, 0.9, 1.02, 0.98, 1.08, 0.92, 0.85, 1.03, 0.97)


def _capillary_bed(fast_factor: float = 10.0, count: int = 10):
    G = _graph({n: (0, 0, 10 * n) for n in range(count + 1)})
    for n in range(count):
        _vessel(G, n, n + 1, branch_order="B01", diameter_um=5.0, flow_abs=1e-15 * _ALIKE[n])
    data = G.edges[count - 1, count, 0]
    data["flow_abs"] *= fast_factor
    return G


def test_a_capillary_ten_times_as_fast_as_the_rest_is_an_outlier():
    G = _capillary_bed()

    (item,) = nr.find_flow_outliers(G)

    assert item.nodes == (9, 10)
    assert item.measures["class"] == "capillary"
    assert item.measures["robust_z"] > nr.FLOW_OUTLIER_ROBUST_Z
    assert "faster" in item.why


def test_a_class_too_small_to_judge_has_no_outliers_and_a_still_vessel_is_always_listed():
    assert nr.find_flow_outliers(_capillary_bed(count=5)) == []
    assert nr.find_flow_outliers(_capillary_bed(fast_factor=1.0, count=12)) == []
    G = _capillary_bed(fast_factor=1.0)
    G.edges[0, 1, 0]["flow_abs"] = 0.0
    (item,) = nr.find_flow_outliers(G)
    assert item.nodes == (0, 1)
    assert item.why == "A solved vessel with no flow through it."


# --- boundary nodes --------------------------------------------------------------------


def test_each_boundary_issue_is_listed_in_turn():
    # Image 100 um across in x; inlet 0 at the x = 0 face, outlet 5 inside it.
    G = _graph({0: (50, 50, 0), 1: (50, 50, 20), 2: (50, 50, 40), 3: (50, 50, 60), 4: (50, 30, 20),
                5: (50, 50, 80), 6: (50, 70, 40), 7: (50, 99, 40), 8: (50, 50, 99)})
    pressures = {0: 100, 1: 90, 2: 80, 3: 70, 4: 85, 5: 60, 6: 75, 7: 75, 8: 55}
    for node, p in pressures.items():
        G.nodes[node]["pressure"] = float(p)
    _vessel(G, 0, 1, flow_abs=1.0)
    _vessel(G, 1, 2, flow_abs=1.0)
    _vessel(G, 2, 3, flow_abs=1.0)
    _vessel(G, 3, 5, flow_abs=1.0)
    _vessel(G, 1, 4, flow_abs=0.0)       # a dead end off the inlet's vessel, inside
    _vessel(G, 2, 6, flow_abs=0.0)
    _vessel(G, 6, 7, flow_abs=0.0)       # an open end at the y face, no boundary node
    _vessel(G, 3, 8, flow_abs=0.00001)   # an outlet at the x face carrying almost nothing
    roles = {"inlet": [0], "outlet": [5, 8, 2]}

    items = nr.find_boundary_issues(G, roles, extent_um=(100, 100, 100))

    issues = [(item.nodes[0], item.measures["issue"]) for item in items]
    assert issues == [
        (2, "not a dead end"),
        (5, "inside the image"),
        (2, "inside the image"),
        (7, "open end at a face"),
        (8, "little flow"),
    ]
    assert items[0].measures["role"] == "outlet"
    assert items[1].measures["depth_um"] == pytest.approx(20.0)


def test_without_the_image_size_only_what_the_graph_shows_is_judged():
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 0, 20)})
    _vessel(G, 0, 1)
    _vessel(G, 1, 2)
    assert nr.find_boundary_issues(G, {"inlet": [0], "outlet": [2]}) == []
    (item,) = nr.find_boundary_issues(G, {"inlet": [1], "outlet": [2]})
    assert item.measures["issue"] == "not a dead end"


# --- needing the mask -----------------------------------------------------------------


def test_dead_ends_come_artefacts_first_each_with_its_chain_to_the_junction():
    mask, G, tips = dead_end_cases()
    support = MaskSupport(mask, VOXEL_SIZE)

    items = nr.find_dead_ends(G, support, image_face_margin_um=0.0)

    order = [item.nodes[0] for item in items]
    assert order == [
        tips["inside_other_lumen"], tips["off_mask"], tips["mask_continues"], tips["short"], tips["real_branch"],
    ]
    assert items[-1].measures["kinds"] == ""
    assert "none of the artefact kinds" in items[-1].why
    off = items[1]
    assert off.nodes == (tips["off_mask"], (5.0, 30.0, 30.0))
    assert off.measures["length_um"] == pytest.approx(20.0)
    assert len(off.edges) == 1 and G.has_edge(*off.edges[0])


def test_two_strands_in_one_lumen_are_one_item_saying_whether_deleting_cuts_anything_off():
    mask, G = two_strands_in_one_lumen()

    (item,) = nr.find_two_in_one_lumen(G, MaskSupport(mask, VOXEL_SIZE))

    assert len(item.edges) == 2 and all(G.has_edge(*e) for e in item.edges)
    assert item.measures["cuts_off"] == "False;False"
    assert item.measures["shorter_um"] == pytest.approx(55.0)


def _broken_vessel(gap_um: float):
    """A vessel along x broken by a gap in its mask, drawn as two pieces."""
    mask = np.zeros((15, 20, 80), dtype=bool)
    idx = np.indices(mask.shape)
    mask |= ((idx[0] - 7) ** 2 + (idx[1] - 7) ** 2) <= 4
    lo = 30
    hi = int(lo + gap_um)
    mask[:, :, lo + 1:hi] = False
    G = _graph({"start": (7, 7, 0), "left": (7, 7, lo), "right": (7, 7, hi), "end": (7, 7, 79)})
    _vessel(G, "start", "left")
    _vessel(G, "right", "end")
    return G, MaskSupport(mask, (1.0, 1.0, 1.0))


def test_facing_ends_past_the_joins_reach_are_listed_with_why():
    G, support = _broken_vessel(15.0)

    (item,) = nr.find_facing_ends(G, support, max_gap_um=10.0)

    assert set(item.nodes) == {"left", "right"}
    assert item.measures["gap_um"] == pytest.approx(15.0)
    assert item.measures["refused"] == REFUSED_GAP_TOO_WIDE
    assert len(item.edges) == 2
    assert item.centre_um == pytest.approx((7.0, 7.0, 37.5))
    assert nr.find_facing_ends(G, support, max_gap_um=5.0) == []  # 15 > 2 x 5


def test_a_segmented_vessel_without_a_centreline_is_an_item_to_trace():
    mask, G, tips = dead_end_cases()

    items = nr.find_uncovered_mask(G, MaskSupport(mask, VOXEL_SIZE))

    assert items and all(item.kind == nr.UNCOVERED_MASK and item.edges == () for item in items)
    # The mask running on past the mask_continues tip, beyond its lumen.
    assert items[0].centre_um[2] == pytest.approx(50.0, abs=1.0)
    assert items[0].centre_um[1] > tips["mask_continues"][1]
    assert len(items[0].points_um) > 0


# --- the region a scan is limited to ------------------------------------------------


def test_within_keeps_items_round_the_view_centre():
    G = _two_junctions(1.0)
    H = _two_junctions(1.0)
    shift = np.array([0.0, 0.0, 500.0])
    for node in H.nodes:
        H.nodes[node]["pos"] = H.nodes[node]["pos"] + shift
    for _u, _v, data in H.edges(data=True):
        data["voxels"] = (np.asarray(data["voxels"]) + shift).tolist()
    items = nr.find_close_junctions(nx.disjoint_union(G, H))
    assert len(items) == 2
    assert len(nr.within(items, (0.0, 0.0, 0.0), 50.0)) == 1
    assert len(nr.within(items, (0.0, 0.0, 0.0), 0.0)) == 2
    assert len(nr.within(items, None, 50.0)) == 2


def test_z_range_around_covers_the_slices_within_reach():
    # z = 15..25 um on 2 um slices: slices 7.5..12.5, rounded outwards.
    assert nr.z_range_around((20.0, 0.0, 0.0), 5.0, (2.0, 1.0, 1.0), depth=100) == (7, 14)
    assert nr.z_range_around((2.0, 0.0, 0.0), 50.0, (1.0, 1.0, 1.0), depth=30) == (0, 30)
    assert nr.z_range_around((2.0, 0.0, 0.0), 0.0, (1.0, 1.0, 1.0), depth=30) is None
