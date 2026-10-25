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
from haemolynx.graph import skeleton_retrace as R
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


def test_a_thick_vessel_bridge_drawn_round_a_bend_is_not_a_hairpin():
    """A bridge keeps its drawn path where a straight one would leave the
    segmentation (a curved thick vessel), so it can double back; it is not a
    vessel, and is not listed."""
    G = _graph({0: (0, 0, 0), 1: (0, 5, 0)})
    _vessel(G, 0, 1, points=[(0, 0, 0), (0, 0, 15), (0, 5, 15), (0, 5, 0)], **{IS_ZERO_RESISTANCE: True})
    assert nr.find_hairpins(G) == []


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


def test_a_thick_vessel_bridge_is_no_side_of_a_diameter_jump():
    """A bridge takes the widest diameter at its ends (``diameter_source=
    "bridge"``), for drawing: a thin vessel running on into its bridge, or a
    bridge a hand edit gave a branch order, is no jump in a measured width."""
    bridge = {IS_ZERO_RESISTANCE: True, "diameter_source": "bridge"}
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 0, 20)})
    _vessel(G, 0, 1, diameter_um=4.0, diameter_source="measured")
    _vessel(G, 1, 2, diameter_um=12.0, **bridge)
    assert nr.find_diameter_jumps(G) == []

    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 10, 20), 3: (0, -10, 20)})
    _vessel(G, 0, 1, diameter_um=5.0, diameter_source="measured", branch_order="Art1")
    _vessel(G, 1, 2, diameter_um=12.0, branch_order="Art2", **bridge)
    _vessel(G, 1, 3, diameter_um=4.0, diameter_source="measured", branch_order="Art2")
    assert nr.find_diameter_jumps(G) == []
    G.edges[1, 2, 0].pop(IS_ZERO_RESISTANCE)
    (item,) = nr.find_diameter_jumps(G)
    assert item.measures["at"] == "junction" and item.measures["ratio"] == pytest.approx(2.4)


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


def test_a_thick_vessel_bridge_is_never_an_unmeasured_width():
    """A bridge is not measured, by design: its resistance is zeroed whatever
    its width, so a width in doubt there moves nothing."""
    G = _graph({0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 0, 20)})
    _vessel(G, 0, 1, diameter_um=5.0, diameter_source="table", flow_abs=1.0)
    _vessel(G, 1, 2, diameter_um=5.0, diameter_source="table", flow_abs=1.0,
            fwhm_low_confidence_vs_edt=True, **{IS_ZERO_RESISTANCE: True})
    assert [item.nodes for item in nr.find_unmeasured_diameters(G)] == [(0, 1)]


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


def test_a_thick_vessel_bridge_is_never_a_flow_outlier():
    """A bridge's velocity is its flow over a width it was never measured at,
    and a bridge whose vessel was cut off carries none: neither says anything
    about the network, fast or still."""
    G = _capillary_bed(count=12)
    G.edges[0, 1, 0]["flow_abs"] = 0.0
    assert {item.nodes for item in nr.find_flow_outliers(G)} == {(0, 1), (11, 12)}
    G.edges[11, 12, 0][IS_ZERO_RESISTANCE] = True
    G.edges[0, 1, 0][IS_ZERO_RESISTANCE] = True
    assert nr.find_flow_outliers(G) == []


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

    # Grouped by kind (graph.dead_ends), artefacts first and longest first
    # within each; the island is listed once, last.
    order = [item.nodes[0] for item in items]
    assert order[:5] == [
        tips["off_mask"], tips["inside_other_lumen"], tips["short"], tips["mask_continues"], tips["real_branch"],
    ]
    assert [item.measures["kind"] for item in items] == [
        "artefact", "artefact", "artefact", "stops_short", "unknown", "island_end",
    ]
    real = items[4]
    assert real.measures["kinds"] == ""
    assert "a blind end, or a broken vessel" in real.why
    off = items[0]
    assert off.nodes == (tips["off_mask"], (5.0, 30.0, 30.0))
    assert off.measures["length_um"] == pytest.approx(20.0)
    assert len(off.edges) == 1 and G.has_edge(*off.edges[0])


def test_two_strands_in_one_lumen_are_one_item_saying_whether_deleting_cuts_anything_off():
    mask, G = two_strands_in_one_lumen()

    (item,) = nr.find_two_in_one_lumen(G, MaskSupport(mask, VOXEL_SIZE))

    assert len(item.edges) == 2 and all(G.has_edge(*e) for e in item.edges)
    assert item.measures["cuts_off"] == "False;False"
    assert item.measures["shorter_um"] == pytest.approx(55.0)


def _narrow_strand_along_a_trunk():
    """A capillary fused to a 14 um trunk's wall: its centreline runs 5.5 um
    from the trunk's, inside the trunk's lumen, and leaves by a side branch.
    The trunk is long enough that the two are no whole-edge pair."""
    from lumen_artefact_fixtures import capsule, polyline_graph

    mask = capsule(np.zeros((21, 50, 120), dtype=bool), (10, 20, -10), (10, 20, 130), 7)
    capsule(mask, (10, 25.5, 40), (10, 44, 44), 2)
    G = polyline_graph([[(10, 20, 0), (10, 20, 119)], [(10, 25.5, 10), (10, 25.5, 40), (10, 43, 44)]])
    return mask, G


def test_doubling_seen_only_from_the_wider_vessel_is_listed_and_left_by_the_clean_up():
    """PLAN_clean_graph.md section 6: a sample is judged with its own vessel's
    radius, so the capillary is beside the trunk from the trunk's side only.
    The stretch step leaves that (grid H2 found half such chains vessels of
    their own and no rule), and this list shows it for a person to decide."""
    from haemolynx.graph.doubled_stretches import resolve_doubled_stretches

    mask, G = _narrow_strand_along_a_trunk()
    support = MaskSupport(mask, VOXEL_SIZE)
    assert nr.find_two_in_one_lumen(G, support) == []
    before = _unchanged(G)
    resolve_doubled_stretches(G, support)
    assert _unchanged(G) == before

    (item,) = nr.find_doubled_stretches(G, support)
    assert item.kind == nr.DOUBLED_STRETCHES
    assert item.measures["seen_from"] == nr.SEEN_FROM_WIDER
    assert item.measures["dead_end"] is True
    assert item.measures["vessel_radius_um"] > 2 * item.measures["other_radius_um"]
    assert 25.0 < item.measures["run_um"] < 40.0
    assert len(item.edges) == 2 and all(G.has_edge(*e) for e in item.edges)
    assert 10.0 < item.centre_um[2] < 40.0
    assert "seen only from the wider one" in item.why
    assert _unchanged(G) == before

    # The Post processing tab's "Stretches doubled in one lumen" makes it.
    from haemolynx.gui.post_processing import REVIEW_LIST_BY_KEY, find_review_items

    assert REVIEW_LIST_BY_KEY[nr.DOUBLED_STRETCHES].uses_mask
    assert find_review_items(nr.DOUBLED_STRETCHES, G, support=support) == [item]


def test_a_whole_edge_pair_is_left_to_the_two_vessels_list():
    mask, G = two_strands_in_one_lumen()
    support = MaskSupport(mask, VOXEL_SIZE)
    assert len(nr.find_two_in_one_lumen(G, support)) == 1
    assert nr.find_doubled_stretches(G, support) == []


def test_runs_shorter_than_the_listing_minimum_are_not_listed():
    mask, G = _narrow_strand_along_a_trunk()
    support = MaskSupport(mask, VOXEL_SIZE)
    assert nr.find_doubled_stretches(G, support, min_run_um=100.0) == []


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


# --- skeleton the graph does not follow (PLAN_surface_coverage.md) ----------------


def test_skeleton_the_graph_does_not_follow_is_listed_with_the_route_the_retrace_would_draw():
    from test_skeleton_retrace import _ring

    scene = _ring()
    support = scene.support()

    (item,) = nr.find_lost_skeleton(scene.G, scene.skeleton, support)

    assert item.kind == nr.LOST_SKELETON and item.nodes == () and item.edges == ()
    assert item.measures["reason"] == nr.LOST_SKELETON_WHY[None]
    assert 30.0 <= item.measures["length_um"] <= 40.0
    assert item.measures["hole_um"] > 20.0
    assert item.measures["mask_fraction"] == pytest.approx(1.0)
    assert len(item.points_um) == item.measures["voxels"]
    # The route runs from the vessel at one end (A) to the one at the other (D).
    ends = sorted(tuple(np.round(p, 6)) for p in (item.path_um[0], item.path_um[-1]))
    assert ends == sorted(tuple(np.round(scene.G.nodes[n]["pos"], 6)) for n in ("A", "D"))
    assert "skeleton no centreline follows" in item.why
    # A wider limit: still listed, now refused for its hole.
    (refused,) = nr.find_lost_skeleton(scene.G, scene.skeleton, support, keep_hole_over_um=1000.0)
    assert refused.measures["reason"] == nr.LOST_SKELETON_WHY[R.REFUSED_SMALL_HOLE]
    assert refused.path_um is not None


def test_a_lost_branch_is_listed_with_no_route_and_short_pieces_are_not():
    from test_skeleton_retrace import Scene

    scene = Scene((13, 60, 70))
    scene.line((6.0, 15.0, 10.0), (6.0, 15.0, 60.0), node_names=("a", "b"))
    scene.line((6.0, 15.0, 35.0), (6.0, 50.0, 35.0), in_graph=False)  # 35 um branch
    scene.line((6.0, 15.0, 20.0), (6.0, 26.0, 20.0), in_graph=False)  # 11 um stub

    (item,) = nr.find_lost_skeleton(scene.G, scene.skeleton, scene.support())

    assert item.measures["reason"] == nr.LOST_SKELETON_WHY[R.REFUSED_ONE_END]
    assert item.path_um is None
    assert item.centre_um[1] > 25.0  # the branch, not the stub


def test_lost_skeleton_is_judged_only_where_the_scan_reads_and_listed_longest_first():
    from test_skeleton_retrace import Scene

    scene = Scene((40, 60, 70))
    scene.line((6.0, 15.0, 10.0), (6.0, 15.0, 60.0), node_names=("a", "b"))
    scene.line((6.0, 15.0, 30.0), (6.0, 50.0, 30.0), in_graph=False)  # 35 um, at z 6
    scene.line((6.0, 15.0, 50.0), (30.0, 15.0, 50.0), in_graph=False)  # 24 um, up through z 7-30

    lengths = [item.measures["length_um"] for item in nr.find_lost_skeleton(scene.G, scene.skeleton, scene.support())]
    deep = nr.find_lost_skeleton(scene.G, scene.skeleton, scene.support(), z_range=(20, 40))

    assert len(lengths) == 2 and lengths == sorted(lengths, reverse=True)
    assert len(deep) == 1 and deep[0].measures["length_um"] == pytest.approx(min(lengths))


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


# --- terminal nodes ---------------------------------------------------------------------


def _terminal_network():
    """Image 100 um across. A trunk 0-1-2-3-5 from the x = 0 face to the
    x = 99 face, inlet 0 and outlet 5; off it an interior dead end 4, open
    ends 7 (6 um wide) and 9 (10 um wide) at the y faces, and a lone vessel
    10-11 with a dead end at each end."""
    G = _graph({0: (50, 50, 0), 1: (50, 50, 20), 2: (50, 50, 40), 3: (50, 50, 60), 5: (50, 50, 99),
                4: (50, 30, 20), 7: (50, 99, 40), 9: (50, 0, 60), 10: (10, 10, 10), 11: (10, 10, 30)})
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 5)):
        _vessel(G, u, v, diameter_um=12.0)
    _vessel(G, 1, 4, diameter_um=4.0)
    _vessel(G, 2, 7, diameter_um=6.0)
    _vessel(G, 3, 9, diameter_um=10.0)
    _vessel(G, 10, 11, diameter_um=8.0)
    return G


def test_terminal_nodes_list_inlets_outlets_then_open_ends_widest_first_then_other_dead_ends():
    G = _terminal_network()
    before = _unchanged(G)
    roles = {"inlet": [0], "outlet": [5], "arteriole_boundary": [4]}

    items = nr.find_terminal_nodes(G, roles, extent_um=(100, 100, 100))

    assert _unchanged(G) == before
    assert [item.nodes[0] for item in items] == [0, 5, 9, 7, 4]
    assert all(item.kind == nr.TERMINAL_NODES for item in items)
    assert [item.measures["role"] for item in items] == ["inlet", "outlet", "", "", ""]
    assert [item.measures["leaves_image"] for item in items] == [True, True, True, True, False]
    assert [item.measures["diameter_um"] for item in items] == [12.0, 12.0, 10.0, 6.0, 4.0]
    assert items[4].measures["other_roles"] == "arteriole_boundary"
    assert items[4].measures["depth_um"] == pytest.approx(20.0)
    assert items[2].centre_um == pytest.approx((50.0, 0.0, 60.0))
    assert "open end" in items[2].why and "inside the image" in items[4].why
    assert items[0].why.startswith("Node 0 is an inlet")


def test_a_terminal_role_on_a_junction_or_inside_the_image_is_listed_saying_so():
    G = _terminal_network()
    items = nr.find_terminal_nodes(G, {"inlet": [0], "outlet": [2]}, extent_um=(100, 100, 100))

    outlet = items[1]
    assert outlet.nodes == (2,) and outlet.measures["role"] == "outlet"
    assert outlet.measures["vessels"] == 3
    assert "3 vessels meet there" in outlet.why
    assert outlet.measures["leaves_image"] is False
    # Node 5 is no boundary node now: an open end like the others.
    assert [item.nodes[0] for item in items[2:]] == [5, 9, 7, 4]


def test_terminal_nodes_read_the_flow_through_their_vessel_when_solved():
    G = _terminal_network()
    for u, v, k, data in G.edges(keys=True, data=True):
        data["flow_abs"] = 2.0 if {u, v} <= {0, 1, 2, 3, 5} else 0.5
    items = nr.find_terminal_nodes(G, {"inlet": [0], "outlet": [5]}, extent_um=(100, 100, 100))
    shares = {item.nodes[0]: item.measures["flow_share"] for item in items}
    assert shares[0] == pytest.approx(1.0)
    assert shares[9] == pytest.approx(0.25)


# --- "Broken vessels" and the turn onto the vessel ahead (PLAN_dead_ends.md Phase 6) -------


def test_broken_vessels_list_facing_pairs_and_dead_ends_pointing_at_a_side():
    """The list renamed "Broken vessels" shows the facing pairs the join
    left open and, beside them, the dead ends the definition calls broken
    with another vessel's side ahead -- each saying how it would be joined,
    with the same measures, so one CSV holds both."""
    from dead_end_fixtures import VOXEL_SIZE as FIXTURE_VOXELS
    from dead_end_fixtures import broken_towards_side

    G, support = _broken_vessel(15.0)
    (pair,) = nr.find_broken_vessels(G, support, max_gap_um=10.0)
    assert pair.kind == nr.BROKEN_VESSELS == nr.FACING_ENDS
    assert pair.measures["join"] == nr.JOIN_ENDS and pair.measures["refused"] == REFUSED_GAP_TOO_WIDE
    assert pair.measures["ahead_edge"] is None

    case = broken_towards_side()
    side_support = MaskSupport(case.mask, FIXTURE_VOXELS)
    before = _unchanged(case.graph)
    # The trunk's and the far vessel's ends at the x faces face each other
    # 20 um apart, but all four leave the image: two vessels, not one broken.
    (side,) = nr.find_broken_vessels(case.graph, side_support)
    assert _unchanged(case.graph) == before
    assert len(nr.find_facing_ends(case.graph, side_support)) == 2
    assert side.nodes[0] == case.tip and side.measures["join"] == nr.JOIN_SIDE
    assert side.measures["gap_um"] == pytest.approx(9.0, abs=0.6)
    assert side.measures["turn_deg"] < 5.0
    assert case.graph.has_edge(*side.measures["ahead_edge"])
    assert side.measures["ahead_point_um"][1] == pytest.approx(10.0, abs=0.6)
    assert all(case.graph.has_edge(*e) for e in side.edges)
    assert set(pair.measures) == set(side.measures)


def test_a_dead_end_in_a_facing_pair_is_not_listed_again_as_a_side_join():
    from dead_end_fixtures import VOXEL_SIZE as FIXTURE_VOXELS
    from dead_end_fixtures import facing_pair

    case = facing_pair()
    items = nr.find_broken_vessels(case.graph, MaskSupport(case.mask, FIXTURE_VOXELS))
    assert [item.measures["join"] for item in items] == [nr.JOIN_ENDS]
    assert case.tip in items[0].nodes


def test_dead_ends_stay_longest_first_within_their_kind_whatever_their_turn(monkeypatch):
    """A turn-first order was tried and taken out: the fresh grid G3′
    (Harvey, 2026-10-25) scored the turn rule 21/30 against 29/30 for marking
    every join wrong. The turn stays a measure of each item."""
    from haemolynx.graph import dead_ends as de

    junction = (0.0, 0.0, 0.0)
    tips = {"a": (0.0, 10.0, 0.0), "b": (0.0, 0.0, 10.0), "c": (0.0, -10.0, 0.0), "d": (0.0, 0.0, -10.0),
            "e": (5.0, 0.0, 0.0)}
    G = _graph({"j": junction, **tips})
    for tip in tips:
        _vessel(G, tip, "j")
    kinds = {"a": ("broken", 40.0, 10.0), "b": ("broken", 10.0, 20.0), "c": ("broken", None, 50.0),
             "d": ("artefact", 5.0, 2.0), "e": ("artefact", None, 4.0)}

    def fake(G, **_kw):
        return {
            tip: de.DeadEnd(tip=tip, junction="j", kind=kind, chain=((tip, "j", 0),), chain_length_um=length,
                            tip_um=tips[tip], heading=(0.0, 1.0, 0.0), ahead_turn_deg=turn)
            for tip, (kind, turn, length) in kinds.items()
        }

    monkeypatch.setattr(de, "classify_dead_ends", fake)
    items = nr.find_dead_ends(G, MaskSupport(np.zeros((12, 12, 12), dtype=bool), (1.0, 1.0, 1.0)))

    assert [item.nodes[0] for item in items] == ["e", "d", "c", "b", "a"]
    assert [item.measures["ahead_turn_deg"] for item in items] == [None, 5.0, None, 10.0, 40.0]


def test_broken_vessels_come_closest_first(monkeypatch):
    from haemolynx.graph import dead_ends as de

    G, support = _broken_vessel(15.0)
    (template,) = nr.find_facing_ends(G, support, max_gap_um=10.0)
    pairs = [
        nr.ReviewItem(kind=nr.FACING_ENDS, nodes=(f"t{n}", f"u{n}"), edges=(), centre_um=(0.0, 0.0, 0.0), why="",
                      measures={**template.measures, "turn_deg": turn, "gap_um": gap})
        for n, (turn, gap) in enumerate([(5.0, 9.0), (40.0, 1.0), (5.0, 2.0)])
    ]
    monkeypatch.setattr(nr, "find_facing_ends", lambda *_a, **_k: pairs)
    monkeypatch.setattr(de, "classify_dead_ends", lambda *_a, **_k: {})

    items = nr.find_broken_vessels(G, support)

    assert [item.nodes[0] for item in items] == ["t1", "t2", "t0"]


# --- "Dead ends at thick vessels" (PLAN_bridging.md 0.2, 1.3) -----------------------


def test_dead_ends_at_thick_walls_are_listed_with_where_a_join_lands():
    """Each dead end the definition calls ``thick_wall``: its chain to the
    junction (what Delete takes, never the thick vessel), how far it stopped
    from the lumen, and the point on the thick vessel's own centreline that
    "Join to the vessel ahead" joins it to."""
    from dead_end_fixtures import THICK_AXIS, at_a_thick_wall
    from dead_end_fixtures import VOXEL_SIZE as FIXTURE_VOXELS

    case = at_a_thick_wall()
    support = MaskSupport(case.mask, FIXTURE_VOXELS)
    before = _unchanged(case.graph)

    (item,) = nr.find_dead_ends_at_thick_walls(case.graph, support)

    assert _unchanged(case.graph) == before
    assert item.kind == nr.THICK_WALL_ENDS and item.nodes[0] == case.tip
    assert item.measures["kind"] == "thick_wall"
    assert item.measures["thick_lumen_um"] == pytest.approx(6.0)
    assert item.measures["thick_centreline_um"] == pytest.approx(12.0)
    assert item.measures["thick_turn_deg"] == pytest.approx(0.0, abs=1.0)
    assert set(item.measures["ahead_edge"][:2]) == set(THICK_AXIS)
    assert case.graph.has_edge(*item.measures["ahead_edge"])
    assert item.measures["ahead_point_um"] == pytest.approx((10.0, 7.0, 40.0), abs=0.6)
    (chain,) = item.edges
    assert case.tip in chain[:2] and not set(chain[:2]) & set(THICK_AXIS)
    assert "6 µm from a thick vessel's lumen" in item.why

    unrecorded = at_a_thick_wall(recorded=False)
    assert nr.find_dead_ends_at_thick_walls(unrecorded.graph, support) == []


def test_dead_ends_at_thick_walls_come_nearest_the_lumen_first(monkeypatch):
    from haemolynx.graph import dead_ends as de

    junction = (0.0, 0.0, 0.0)
    tips = {"a": (0.0, 10.0, 0.0), "b": (0.0, 0.0, 10.0), "c": (0.0, -10.0, 0.0), "d": (0.0, 0.0, -10.0)}
    G = _graph({"j": junction, **tips})
    for tip in tips:
        _vessel(G, tip, "j")
    lumen = {"a": 8.0, "b": -2.0, "c": 3.0, "d": None}

    def fake(G, **_kw):
        return {
            tip: de.DeadEnd(tip=tip, junction="j", kind=de.THICK_WALL if lumen[tip] is not None else de.UNKNOWN,
                            chain=((tip, "j", 0),), chain_length_um=10.0, tip_um=tips[tip], heading=None,
                            thick_lumen_um=lumen[tip], thick_centreline_um=lumen[tip],
                            at_thick_wall=lumen[tip] is not None)
            for tip in tips
        }

    monkeypatch.setattr(de, "classify_dead_ends", fake)
    items = nr.find_dead_ends_at_thick_walls(G, MaskSupport(np.zeros((12, 12, 12), dtype=bool), (1.0, 1.0, 1.0)))

    assert [item.nodes[0] for item in items] == ["b", "c", "a"]
    assert "2 µm in" in items[0].why and items[0].measures["ahead_edge"] is None
