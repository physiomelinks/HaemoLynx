"""Grid P's interactive page (``scripts/thick_junction_review``): what each
case hands the page -- the segmentation and the thick-vessel region a plane
at a time across the sheets' slab, and every line in the sheet's own view --
and the page built from it. The grid itself is drawn from a run and never in
CI."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from thick_bridge_review.build_grid import HALF_SLAB_UM, HALF_WINDOW_UM, PLANES
from thick_junction_review.grid_page import PLACEHOLDER, build_page
from thick_junction_review.phase1_rules_grid import case_view

VS = np.array([1.0, 0.5, 0.5])


def _unpack(text: str, shape) -> np.ndarray:
    flat = np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))
    return flat[: int(np.prod(shape))].reshape(shape).astype(bool)


def _scene():
    """A 60 x 120 x 120 voxel stack (60 x 60 x 60 um): a thick vessel along y
    at z = 30 um, x = 30 um, its region the vessel's own voxels, and a thin
    vessel's centreline along x at y = 30 um ending at the wall."""
    shape = (60, 120, 120)
    zz, yy, xx = np.indices(shape)
    z_um, x_um = zz * VS[0], xx * VS[2]
    thick = (z_um - 30) ** 2 + (x_um - 30) ** 2 <= 6.0 ** 2
    mask = thick.copy()
    mask[28:33, 58:63, 72:] = True
    thin = np.array([[30.0, 30.0, x] for x in np.arange(36.0, 59.5, 0.5)])
    far = np.array([[30.0 + d, 10.0, 25.0] for d in np.arange(-15.0, 15.5, 0.5)])
    tree = np.array([[30.0, y, 30.0] for y in np.arange(0.0, 60.0, 0.5)])
    return mask, thick, tree, [(thin, False), (far, True)], thin


def test_a_case_hands_the_page_its_slab_a_plane_at_a_time():
    mask, region, tree, paths, thin = _scene()
    case = {"id": "P01", "rule": "wall", "vox": (30, 60, 72), "view": "down z", "axis": 0,
            "centre": np.array([30.0, 30.0, 36.0]), "piece": thin[:10], "route": thin[[0, -1]]}

    view = case_view(case, mask, region, tree, paths, VS)

    label, h, w = PLANES[0]
    assert (view["depth"], view["down"], view["across"]) == ("z", "y", "x")
    n, rows, cols = view["shape"]
    # The sheets' slab: the case's plane and HALF_SLAB_UM either side.
    assert n == int(np.ceil(2 * HALF_SLAB_UM / VS[0])) + 1
    assert cols == int(np.ceil(2 * HALF_WINDOW_UM / VS[2])) + 1 or view["origin_um"][0] == 0.0
    lo = np.round(np.asarray([view["origin_um"][2], view["origin_um"][1], view["origin_um"][0]]) / VS).astype(int)
    expected = mask[lo[0]:lo[0] + n, lo[1]:lo[1] + rows, lo[2]:lo[2] + cols]
    assert np.array_equal(_unpack(view["seg"], view["shape"]), expected)
    assert np.array_equal(_unpack(view["region"], view["shape"]),
                          region[lo[0]:lo[0] + n, lo[1]:lo[1] + rows, lo[2]:lo[2] + cols])
    # Lines are [across, down, depth] microns: x, y, z for a view down z.
    assert view["piece"][0] == [36.0, 30.0, 30.0]
    assert view["route"] == [[36.0, 30.0, 30.0], [59.0, 30.0, 30.0]]


def test_only_what_the_slab_holds_is_handed_over_split_where_it_leaves():
    """The bridge runs through z 15-45 um: the slab is z 24-36 (and 2 um
    either side for lines), so only that stretch reaches the page, as one
    run; the thick vessel's tree is kept within the window."""
    mask, region, tree, paths, thin = _scene()
    case = {"id": "P02", "rule": "taper", "vox": (30, 20, 50), "view": "down z", "axis": 0,
            "centre": np.array([30.0, 10.0, 25.0]), "piece": thin[:4]}

    view = case_view(case, mask, region, tree, paths, VS)

    (bridge,) = [line for line in view["lines"] if line["kind"] == "bridge"]
    depths = [p[2] for p in bridge["points"]]
    assert min(depths) >= 30.0 - HALF_SLAB_UM - 2.0 and max(depths) <= 30.0 + HALF_SLAB_UM + 2.0
    assert all(abs(p[1] - 10.0) <= HALF_WINDOW_UM + 5 for p in view["tree"])
    assert view["route"] is None and view["tick"] is None


def test_the_page_holds_the_data_once_and_nothing_in_it_closes_its_script():
    data = {"dataset": "test", "voxel_size_zyx": [1, 1, 1], "slab_um": 12,
            "sheets": [{"rule": "wall", "title": "Walls", "question": "a </script> question",
                        "marks": [{"key": "J", "label": "Join"}], "cases": []}]}

    page = build_page(data)

    assert PLACEHOLDER not in page
    block = page.split('<script id="grid-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(block) == data


def test_grid_p2_asks_the_rest_of_the_wall_joins_two_questions():
    """Grid P2 (Harvey's call after grid P): every dead end past the reach
    heading for the thick vessel, and every one the in-lumen loop guard
    refused, that grid P did not mark already -- and nothing else."""
    from thick_junction_review.wall_rest_grid import rest_of_pool

    def case(name, decision, turn, centre):
        return {"node": name, "decision": decision, "chord_turn_deg": turn, "centre": np.asarray(centre, float)}

    pool = [
        case("far facing", "beyond the reach", 45.0, (1.0, 0.0, 0.0)),
        case("far away", "beyond the reach", 95.0, (2.0, 0.0, 0.0)),
        case("far no heading", "beyond the reach", float("nan"), (3.0, 0.0, 0.0)),
        case("guard", "beside a vessel, or a loop", 70.0, (4.0, 0.0, 0.0)),
        case("guard, marked", "beside a vessel, or a loop", 70.0, (50.0, 0.0, 0.0)),
        case("far, marked", "beyond the reach", 14.0, (60.0, 0.0, 0.0)),
        case("joined", "joined", 30.0, (5.0, 0.0, 0.0)),
        case("turning away", "heading away (chord)", 120.0, (6.0, 0.0, 0.0)),
    ]
    marked = [(50.5, 0.0, 0.0), (60.0, 1.0, 0.0)]

    rest = rest_of_pool(pool, marked, max_turn_deg=80.0)

    assert [(c["node"], c["question"]) for c in rest] == [("far facing", "reach"), ("guard", "loop guard")]
    assert "question" not in pool[0], "the pool is left as it was"


def test_a_page_is_named_for_its_grid_and_keeps_its_marks_apart():
    """Grid P2's page says P2, and the marks it keeps in the browser sit
    under their own key: under grid P's they would mix with its 30."""
    import pytest

    data = {"dataset": "test", "voxel_size_zyx": [1, 1, 1], "slab_um": 12, "sheets": []}

    grid_p = build_page(data)
    grid_p2 = build_page(dict(data, grid="P2"))

    assert "<title>Thick-Vessel Grid P</title>" in grid_p and '"grid-p-marks-v1"' in grid_p
    assert "<title>Thick-Vessel Grid P2</title>" in grid_p2 and '"grid-p2-marks-v1"' in grid_p2
    assert "Thick-vessel grid P2</h1>" in grid_p2 and "__GRID" not in grid_p2
    with pytest.raises(ValueError, match="letters and digits"):
        build_page(dict(data, grid='P"><script>'))


# --- grid K: the connectors the split leaves unbridged ---------------------------


def test_grid_k_records_the_connector_the_split_leaves_and_where_its_bridge_would_go(caplog):
    """A vessel running 40 um inside a thick vessel's lumen before it meets
    the centreline tree, the graph's thick centreline 37 um from where it
    leaves the region: past its 17 um reach, so the split keeps it as vessel.
    The recorder sees that connector -- the one the split's warning counts --
    with the reach read at its exit and its bridge to where it met the tree."""
    import logging

    from test_graph_thick_vessel_junctions import T1, _bridges, _graph, _split, _thick, _thin_vessel
    from thick_junction_review.connector_grid import connector_case, failed_connectors_recorded

    from haemolynx.graph.thick_vessel_junctions import _bridge_reach_um

    lumen, body, tree = _thick()
    _thin_vessel(lumen, (20, 2, 40), (20, 2, 24))
    G = _graph(("T0", "T1", (20, 52, 20), T1),
               ("X", "J", (20, 2, 40), (20, 2, 24), (20, 34, 24), (20, 38, 20)))
    found = []
    with caplog.at_level(logging.WARNING), failed_connectors_recorded(found):
        result = _split(G, lumen, body, tree)

    assert not _bridges(result)
    assert "1 connector(s) left as they were: 1 with no thick centreline" in caplog.text
    (record,) = found
    (exit_,) = record["exits"]
    assert exit_["fails"] and exit_["where"][2] > 26.0, "it leaves at the region's wall"
    assert exit_["reach_um"] == _bridge_reach_um(exit_["tree_radius_um"])
    assert exit_["to_centreline_um"] > exit_["reach_um"]
    case = connector_case(record)
    assert case["exits"] == case["failed_exits"] == 1 and not case["spur"]
    assert case["inside_um"] > 35.0
    np.testing.assert_allclose(case["route"][1], (20.0, 38.0, 20.0), atol=1.0)


def test_grid_k_measures_both_causes():
    """A reach read at a spur of the tree a third as wide as the thick vessel
    beside it, and a connector with one exit in reach sunk by the other."""
    from thick_junction_review.connector_grid import connector_case

    def exit_(x, radius, widest, fails):
        return {"where": np.array([0.0, 0.0, x]), "tree_radius_um": radius, "reach_um": 2 * radius + 5,
                "to_centreline_um": 12.0, "target": np.array([0.0, 5.0, x]), "widest_near_um": widest,
                "fails": fails}

    runs = [np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), np.array([[0.0, 0.0, 30.0], [0.0, 4.0, 30.0]])]
    case = connector_case({"runs": runs, "exits": [exit_(0.0, 6.0, 6.0, False), exit_(30.0, 1.4, 4.5, True)]})

    assert case["exits"] == 2 and case["failed_exits"] == 1
    assert case["spur"] and case["reach_um"] == pytest.approx(7.8)
    assert case["centre"][2] == 30.0, "centred on the exit that sank it"
    assert case["piece"][0][2] == 30.0, "with the stretch beside that exit"
    assert case["inside_um"] == pytest.approx(7.0)


def test_grid_k_takes_a_contact_held_twice_as_one():
    """Regression: on E14.5 a connector's contacts held one sample twice, and
    sorting them by distance went on to compare the two points."""
    from thick_junction_review.connector_grid import by_distance

    twice = np.array([3.0, 4.0, 0.0])
    ranked = by_distance([twice, np.array([0.0, 0.0, 9.0]), twice.copy()], np.zeros(3))

    assert [d for d, _p in ranked] == [5.0, 5.0, 9.0]
    assert ranked[0][1] is twice


# --- the E15 grid: the tree fill, the wall join and grid K's measures, fresh ------


def _gap_on_a_straight_tree(detour=None, filled=False):
    """A thick centreline tree along x (z 5, y 5), the graph following it
    from x 0 to 15 and from 25 to 40 and not between: a 10 um gap. With
    *detour* (y), the graph also runs round the gap that far off the tree;
    with *filled*, the fill drew the tree across it."""
    import networkx as nx

    from haemolynx.graph.thick_vessel_junctions import FROM_TREE

    def edge(G, u, v, corners):
        voxels = []
        for a, b in zip(corners, corners[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            steps = max(int(np.ceil(np.linalg.norm(b - a))), 1)
            voxels += [tuple(a + (b - a) * t) for t in np.linspace(0.0, 1.0, steps + 1)[(1 if voxels else 0):]]
        G.add_node(u, pos=tuple(float(c) for c in corners[0]))
        G.add_node(v, pos=tuple(float(c) for c in corners[-1]))
        length = float(np.sum(np.linalg.norm(np.diff(np.asarray(voxels), axis=0), axis=1)))
        G.add_edge(u, v, voxels=voxels, length=length)

    G = nx.MultiGraph()
    edge(G, 0, 1, [(5, 5, 0), (5, 5, 15)])
    edge(G, 2, 3, [(5, 5, 25), (5, 5, 40)])
    if detour is not None:
        edge(G, 1, 2, [(5, 5, 15), (5, detour, 15), (5, detour, 25), (5, 5, 25)])
    after = G.copy()
    if filled:
        after.add_edge(1, 2, voxels=[(5.0, 5.0, float(x)) for x in range(15, 26)], length=10.0, **{FROM_TREE: True})
    tree_vox = np.array([(5, 5, x) for x in range(41)])
    return {
        "fill_in": G, "fill_out": after,
        "fill_args": {"tree_voxels": tree_vox, "vs": np.ones(3), "tree_points": tree_vox.astype(float),
                      "tolerance": np.full(len(tree_vox), 0.5), "tree_radius": np.full(len(tree_vox), 6.0)},
    }


@pytest.mark.parametrize(
    "detour, ratio, says",
    [(None, None, "filled"), (15.0, 3.0, "filled"), (7.0, 1.4, "left as it was")],
    ids=["no-way-round", "a-long-way-round", "about-as-direct"],
)
def test_the_e15_grid_reads_each_gaps_way_round_to_the_end(detour, ratio, says):
    """The tree fill cuts its search at 2.1 times the tree's path; the grid
    reads the graph's own way between the gap's sides to the end, so its key
    places every gap on the measure, whichever side of 2.1."""
    from thick_junction_review.e15_grid import fill_cases

    (case,) = fill_cases(_gap_on_a_straight_tree(detour))

    assert case["sides"] == 2 and case["gap_um"] == pytest.approx(10.0)
    assert case["tree_way_um"] == pytest.approx(10.0)
    assert case["rule_says"] == says
    if ratio is None:
        assert case["own_way_ratio"] is None
    else:
        assert case["own_way_ratio"] == pytest.approx(ratio, rel=0.02)
    np.testing.assert_allclose(case["piece"][[0, -1], 2], [15.0, 25.0])
    assert case["decision"] == "left as it was", "nothing was drawn"


def test_the_e15_grid_reads_whether_the_fill_drew_the_gap():
    from thick_junction_review.e15_grid import fill_cases

    (case,) = fill_cases(_gap_on_a_straight_tree(filled=True))

    assert case["decision"] == "filled" and case["drawn_share"] == 1.0


def test_the_e15_grids_wall_sheet_takes_joins_and_ends_past_the_reach_heading_in():
    """Six joins and three dead ends past the 15 um reach heading for the
    thick vessel (one turned away is not one of the three), then the other
    decisions."""
    import random

    from thick_junction_review.e15_grid import COUNTS, WALL_BEYOND, WALL_JOINED, pick_wall

    def case(decision, past, turn):
        return {"decision": decision, "past_lumen_um": past, "chord_turn_deg": turn, "route_turn_deg": turn}

    pool = ([case("joined", p, 20.0) for p in (1.0, 6.0, 11.0, 12.0, 13.0, 14.0, 2.0, 3.0)]
            + [case("beyond the reach", p, 30.0) for p in (16.0, 17.0, 18.0, 19.0)]
            + [case("beyond the reach", 16.5, 150.0)]
            + [case("heading away (chord)", 4.0, 120.0) for _ in range(6)])

    picked = pick_wall(pool, random.Random(0))

    assert len(picked) == COUNTS["wall"]
    assert sum(c["decision"] == "joined" for c in picked[:WALL_JOINED]) == WALL_JOINED
    beyond = picked[WALL_JOINED:WALL_JOINED + WALL_BEYOND]
    assert all(c["decision"] == "beyond the reach" and c["chord_turn_deg"] < 80.0 for c in beyond)
    bins = {int(c["past_lumen_um"] // 5) for c in picked[:WALL_JOINED]}
    assert bins == {0, 1, 2}, "spread over how far past the lumen they end: 0-5, 5-10 and 10-15 um"


def test_the_e15_grids_wall_sheet_takes_what_the_other_sheets_leave_of_thirty():
    """E15 gave the fill and the connectors no case, so the wall sheet asks
    all thirty; it never asks fewer than its own share, nor more than its
    pool holds."""
    import random

    from thick_junction_review.e15_grid import COUNTS, TOTAL, pick_wall, wall_count

    assert wall_count(COUNTS["fill"], COUNTS["connector"]) == COUNTS["wall"]
    assert wall_count(0, 0) == TOTAL == 30
    assert wall_count(3, 0) == 27
    assert wall_count(20, 8) == COUNTS["wall"]

    pool = [{"decision": "heading away (chord)", "past_lumen_um": float(p), "chord_turn_deg": 120.0,
             "route_turn_deg": 120.0} for p in range(40)]
    assert len(pick_wall(pool, random.Random(0), wall_count(0, 0))) == 30
    assert len(pick_wall(pool[:20], random.Random(0), wall_count(0, 0))) == 20


def test_the_e15_grid_says_whether_each_exit_on_its_own_would_bridge_a_connector():
    """Grid K's second measure: a connector one of whose exits is in reach
    would be bridged there, were each exit taken on its own."""
    from thick_junction_review.e15_grid import connector_cases

    def exit_(x, fails):
        return {"where": np.array([0.0, 0.0, x]), "tree_radius_um": 3.0, "reach_um": 11.0,
                "to_centreline_um": 20.0 if fails else 5.0, "target": np.array([0.0, 5.0, x]),
                "widest_near_um": 3.0, "fails": fails}

    runs = [np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])]
    seen = {"connectors": [
        {"runs": runs, "exits": [exit_(0.0, False), exit_(30.0, True)]},
        {"runs": runs, "exits": [exit_(0.0, True)]},
        {"runs": runs, "exits": []},
    ]}

    cases = connector_cases(seen)

    assert [c["each_exit_on_its_own"] for c in cases] == ["bridged", "left"]
