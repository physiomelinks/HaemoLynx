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
