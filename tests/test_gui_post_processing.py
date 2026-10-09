"""What the "7. Post processing" tab draws and lists (pure, no napari)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import VesselEnd, edge_keys
from haemolynx.graph._helpers import calculate_path_length
from haemolynx.gui.post_processing import (
    ADDED,
    ADDED_NODES,
    AT_JUNCTION,
    CONNECTED,
    HIGH_DEGREE_JUNCTIONS,
    JUNCTION_TABLE_COLUMNS,
    NEW_VESSEL_POINTS,
    NEW_VESSEL_TRACE,
    SELECTED,
    STATUS_COLOURS,
    VesselTrace,
    added_nodes_layer,
    added_vessel_ids,
    branch_id_of,
    camera_center_for,
    default_connectivity_csv_path,
    describe_vessels,
    edits_lost_by_running_from,
    junction_label,
    junction_marker_layer,
    junction_table_rows,
    nearest_node,
    parse_branch_ids,
    point_on_path_under_click,
    point_under_click,
    scan_network,
    status_colours,
    trace_layers,
    vessel_status,
    zoom_for_canvas,
)


def _add(G, u, v, **attrs):
    voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
    G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels), **attrs)


def _network() -> nx.MultiGraph:
    """0 -> 4-way junction 1 -> {2, 3} -> 4, and 1 -> 5."""
    G = nx.MultiGraph()
    positions = {
        0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 10, 20), 3: (0, -10, 20),
        4: (0, 0, 30), 5: (0, 20, 10),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5)]:
        _add(G, u, v, branch_order="B01", diameter_um=5.0)
    return G


def test_scan_finds_the_four_way():
    G = _network()
    scan = scan_network(G)
    assert scan.junctions == (1,)
    assert scan.vessel_count == 6
    assert scan.summary == "1 junction(s) where 4+ vessels meet, in 6 vessels."


def test_vessel_status_orders_selected_over_junction():
    # Segments of vessels 0, 0, 1, 2, 3 (vessel 0 drawn as two segments).
    status = vessel_status([0, 0, 1, 2, 3], at_junction=[1, 2], selected=[2])
    assert list(status) == [CONNECTED, CONNECTED, AT_JUNCTION, SELECTED, CONNECTED]
    assert list(vessel_status([5, 6])) == [CONNECTED, CONNECTED]


def test_status_colours_are_the_declared_rgba():
    colours = status_colours([CONNECTED, AT_JUNCTION, SELECTED])
    assert colours.shape == (3, 4)
    for row, label in zip(colours, [CONNECTED, AT_JUNCTION, SELECTED]):
        assert tuple(row) == STATUS_COLOURS[label]


def test_junction_marker_layer_rings_the_four_way():
    G = _network()
    spec = junction_marker_layer(G, scan_network(G))
    assert spec.name == HIGH_DEGREE_JUNCTIONS and spec.kind == "points"
    assert np.allclose(spec.data, [[0, 0, 10]])
    assert list(spec.features["node_id"]) == [1]
    assert list(spec.features["degree"]) == [4.0]


def test_junction_table_rows_match_the_vessels_listed():
    G = _network()
    vessels, rows = junction_table_rows(G, 1)
    assert len(rows) == len(vessels) == 4
    assert all(len(row) == len(JUNCTION_TABLE_COLUMNS) for row in rows)
    keys = edge_keys(G)
    for vessel, row in zip(vessels, rows):
        assert row[0] == str(vessel.branch_id)
        assert keys[vessel.branch_id] == vessel.edge
        assert row[2] == "5" and row[3] == "B01"
        assert row[4] == str(vessel.other_node)
    assert rows[0][1] == "10"  # the 0-1 vessel is 10 um long


def test_junction_label_names_the_node_and_the_decision():
    G = _network()
    assert junction_label(G, 1) == "Node 1 - 4 vessels"
    assert junction_label(G, 1, "left as is") == "Node 1 - 4 vessels (left as is)"


def test_camera_center_follows_the_displayed_dims():
    position = (4.0, 50.0, 70.0)  # (z, y, x)
    assert camera_center_for(position, (0, 1, 2)) == (4.0, 50.0, 70.0)
    assert camera_center_for(position, (1, 2)) == (0.0, 50.0, 70.0)  # 2D, XY
    assert camera_center_for(position, (0, 2)) == (0.0, 4.0, 70.0)   # 2D, XZ snap


@pytest.mark.parametrize(
    "canvas, box, expected",
    [
        ((600, 800), 60.0, 10.0),   # the shorter side fits the box
        ((0, 0), 60.0, None),       # no canvas yet: keep the zoom
        ((600, 800), 0.0, None),    # nonsense box: keep the zoom
    ],
)
def test_zoom_for_canvas(canvas, box, expected):
    result = zoom_for_canvas(canvas, box)
    assert result == (pytest.approx(expected) if expected is not None else None)


def test_nearest_node_picks_what_is_under_the_click():
    G = _network()  # node 1 at (0, 0, 10), node 5 at (0, 20, 10)
    assert nearest_node(G, (0.0, 0.5, 10.5)) == 1
    assert nearest_node(G, (0.0, 19.0, 10.0)) == 5
    assert nearest_node(G, (0.0, 40.0, 40.0)) is None  # nothing near
    # 3D: a click ray looking down z through (y, x) = (0, 10) hits node 1
    # whatever depth the click position sits at.
    assert nearest_node(G, (50.0, 0.0, 10.0), view_direction=(1, 0, 0)) == 1
    # 2D: only the displayed axes count.
    assert nearest_node(G, (99.0, 20.0, 10.0), dims=(1, 2)) == 5


def test_parse_branch_ids_reads_commas_spaces_and_semicolons_once_each():
    assert parse_branch_ids("12, 40 3;12", 50) == [12, 40, 3]
    assert parse_branch_ids("0", 1) == [0]


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "Type one or more"),
        ("  , ", "Type one or more"),
        ("12, x", "'x' is not a branchID"),
        ("1.5", "'1.5' is not a branchID"),
        ("50", "branchID 50 is not in this network \\(0 to 49\\)"),
        ("-1", "branchID -1 is not in this network"),
    ],
)
def test_parse_branch_ids_says_what_is_wrong(text, message):
    with pytest.raises(ValueError, match=message):
        parse_branch_ids(text, 50)


def test_a_run_from_before_post_processing_would_lose_the_edits_still_in_the_tab():
    from haemolynx.graph import add_vessel_between

    G = _network()
    assert not edits_lost_by_running_from("assign_diameters", G, None)
    add_vessel_between(G, 0, 4)
    assert edits_lost_by_running_from("assign_diameters", G, None)
    assert edits_lost_by_running_from("skeletonise", G, None)
    # Post processing follows the solve: a run from Haemodynamics starts again
    # from Diameters' network, which never had the edits.
    assert edits_lost_by_running_from("build_haemodynamic_model", G, None)


def test_a_run_from_before_post_processing_would_lose_edits_already_applied():
    from haemolynx.graph.post_processing import APPLIED

    applied = _network()
    applied.graph[APPLIED] = True
    assert edits_lost_by_running_from("assign_boundaries", None, applied)
    assert not edits_lost_by_running_from("assign_boundaries", None, _network())


def test_a_run_from_post_processing_or_later_loses_nothing():
    from haemolynx.graph import add_vessel_between
    from haemolynx.graph.post_processing import APPLIED

    G = _network()
    add_vessel_between(G, 0, 4)
    applied = _network()
    applied.graph[APPLIED] = True
    for start in ("post_process", "run_perturbations", "export_results", None, "nope"):
        assert not edits_lost_by_running_from(start, G, applied)


def test_describe_vessels_names_branch_id_ends_length_and_diameter():
    G = _network()
    keys = edge_keys(G)
    dead_end = next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})
    picked = [keys[dead_end], keys[0]]  # given out of order: listed by branchID
    assert describe_vessels(G, picked) == (
        f"branchID 0 (node 0-1, 10 µm, 5 µm); branchID {dead_end} (node 1-5, 20 µm, 5 µm)"
    )
    del G.edges[keys[0]]["diameter_um"]
    assert describe_vessels(G, [keys[0]]) == "branchID 0 (node 0-1, 10 µm)"
    assert describe_vessels(G, [(98, 99, 0)]) == ""  # not in the graph


# --- Add vessel: tracing a new vessel -------------------------------------------


def test_vessel_status_colours_added_vessels_under_the_junction_and_selection():
    status = vessel_status([0, 1, 2, 3], added=[1, 2, 3], at_junction=[2], selected=[3])
    assert list(status) == [CONNECTED, ADDED, AT_JUNCTION, SELECTED]
    assert tuple(status_colours([ADDED])[0]) == STATUS_COLOURS[ADDED]


def test_added_vessel_ids_are_the_branch_ids_add_vessel_drew():
    G = _network()
    _add(G, 5, 3, post_processing_added=True)
    assert added_vessel_ids(G) == [branch_id_of(G, (5, 3, 0))]
    assert added_vessel_ids(_network()) == []


def test_branch_id_of_and_describe_vessels_take_an_edge_either_way_round():
    G = _network()
    _add(G, 5, 3)  # listed by G.edges as (3, 5, 0): node 3 comes first
    assert (3, 5, 0) in edge_keys(G)
    branch_id = edge_keys(G).index((3, 5, 0))
    assert branch_id_of(G, (5, 3, 0)) == branch_id_of(G, (3, 5, 0)) == branch_id
    assert branch_id_of(G, (5, 3, 9)) is None
    assert describe_vessels(G, [(5, 3, 0)]).startswith(f"branchID {branch_id} (node 5-3")


def test_point_on_path_under_click_lands_between_path_points():
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 20.0)]
    # 2D, looking down z: the click's own z (the slice) does not matter.
    point = point_on_path_under_click(path, (7.0, 3.0, 12.0), dims=(1, 2))
    assert np.allclose(point, (0, 0, 12))
    # 3D: whatever lies under the cursor along the view ray.
    point = point_on_path_under_click(
        path, (50.0, 0.0, 5.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2)
    )
    assert np.allclose(point, (0, 0, 5))


def test_point_under_click_in_2d_is_the_click_itself():
    assert np.allclose(point_under_click((4.0, 5.0, 6.0), dims=(1, 2)), (4, 5, 6))


def _two_vessels_along_z() -> np.ndarray:
    """A mask with two vessels on one line of sight down z: one at z 2..4,
    one at z 7..8, both at (y, x) = (5, 5)."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 5, 5] = True
    mask[7:9, 5, 5] = True
    return mask


def test_point_under_click_in_3d_picks_the_middle_of_the_first_vessel_hit():
    mask = _two_vessels_along_z()
    point = point_under_click(
        (0.0, 5.0, 5.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2),
        volume=mask, pick="first",
    )
    assert np.allclose(point[1:], (5, 5)) and 2.5 <= point[0] <= 3.5
    # Looking the other way, the vessel at z 7..8 is in front.
    point = point_under_click(
        (0.0, 5.0, 5.0), view_direction=(-1.0, 0.0, 0.0), dims=(0, 1, 2),
        volume=mask, pick="first",
    )
    assert 7.0 <= point[0] <= 8.0


def test_point_under_click_in_3d_picks_the_brightest_voxel_of_raw_data():
    raw = np.ones((10, 10, 10), dtype=np.float32)
    raw[3, 5, 5], raw[7, 5, 5] = 50.0, 90.0
    point = point_under_click(
        (0.0, 5.0, 5.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2),
        volume=raw, pick="brightest",
    )
    assert np.allclose(point, (7, 5, 5), atol=0.5)
    # Anisotropic voxels: 2 um apart in z, so voxel 7 sits at 14 um.
    point = point_under_click(
        (0.0, 5.0, 5.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2),
        volume=raw, voxel_size_zyx=(2.0, 1.0, 1.0), pick="brightest",
    )
    assert np.allclose(point, (14, 5, 5), atol=1.0)


def test_point_under_click_missing_every_vessel_keeps_the_trace_depth():
    empty = np.zeros((10, 10, 10), dtype=bool)
    point = point_under_click(
        (0.0, 2.0, 3.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2),
        volume=empty, near_um=(6.0, 9.0, 9.0),
    )
    assert np.allclose(point, (6, 2, 3))
    # With no volume at all, the same.
    point = point_under_click(
        (0.0, 2.0, 3.0), view_direction=(1.0, 0.0, 0.0), dims=(0, 1, 2), near_um=(6.0, 9.0, 9.0)
    )
    assert np.allclose(point, (6, 2, 3))


def test_vessel_trace_grows_one_leg_per_click():
    G = _network()
    trace = VesselTrace.starting_at(VesselEnd.on_vessel((1, 5, 0), (0.0, 12.0, 10.0)), G)
    assert trace.points_um == [(0.0, 12.0, 10.0)] and trace.last_point == (0.0, 12.0, 10.0)
    trace.extend([(0.0, 12.0, 10.0), (0.0, 13.0, 11.0), (0.0, 14.0, 12.0)], (0.0, 14.0, 12.0), "through the segmented mask")
    trace.extend([(0.0, 14.0, 12.0), (0.0, 14.0, 15.0)], (0.0, 14.0, 15.0), "through the raw data")
    assert trace.points_um == [
        (0.0, 12.0, 10.0), (0.0, 13.0, 11.0), (0.0, 14.0, 12.0), (0.0, 14.0, 15.0),
    ]
    assert trace.waypoints_um == [(0.0, 14.0, 12.0), (0.0, 14.0, 15.0)]
    assert trace.routes == ["through the segmented mask", "through the raw data"]
    # extended() previews a leg without taking it.
    assert trace.extended([(0.0, 14.0, 15.0), (0.0, 20.0, 15.0)])[-1] == (0.0, 20.0, 15.0)
    assert len(trace.points_um) == 4


def test_trace_layers_draw_the_path_so_far_and_every_click():
    G = _network()
    trace = VesselTrace.starting_at(VesselEnd.at_node(5), G)
    (points,) = trace_layers(trace)  # one click: just the start, no path yet
    assert points.name == NEW_VESSEL_POINTS
    assert list(points.features["role"]) == ["start node"]

    trace.extend([(0.0, 20.0, 10.0), (0.0, 20.0, 15.0), (0.0, 25.0, 15.0)], (0.0, 25.0, 15.0), "x")
    path, points = trace_layers(trace)
    assert path.name == NEW_VESSEL_TRACE and path.kind == "vectors"
    assert path.data.shape == (2, 2, 3)  # two segments, 1 -> 2 then 2 -> 3
    assert np.allclose(path.data[0, 0] + path.data[0, 1], (0, 20, 15))
    assert list(points.features["role"]) == ["start node", "waypoint"]
    assert np.allclose(points.data, [(0, 20, 10), (0, 25, 15)])

    on_vessel = VesselTrace.starting_at(VesselEnd.on_vessel((1, 5, 0), (0.0, 12.0, 10.0)), G)
    (points,) = trace_layers(on_vessel)
    assert list(points.features["role"]) == ["new node"]
    assert tuple(points.options["face_color"][0]) == STATUS_COLOURS[ADDED]


def test_added_nodes_layer_marks_the_new_nodes_still_in_the_graph():
    G = _network()
    spec = added_nodes_layer(G, [5, 99, 5])
    assert spec.name == ADDED_NODES
    assert list(spec.features["node_id"]) == [5]
    assert np.allclose(spec.data, [(0, 20, 10)])
    assert len(added_nodes_layer(G, []).data) == 0


def test_default_connectivity_csv_path_sits_beside_the_vtk_output(tmp_path):
    values = {"vtk_output_prefix": tmp_path / "stack"}
    assert default_connectivity_csv_path(values) == str(tmp_path / "stack_connectivity.csv")
    # A run from another machine names a folder that is not here: just the name.
    elsewhere = {"vtk_output_prefix": tmp_path / "missing" / "stack"}
    assert default_connectivity_csv_path(elsewhere) == "stack_connectivity.csv"
    assert default_connectivity_csv_path(None) == "haemolynx_connectivity.csv"


# --- Manual loop review -----------------------------------------------------------

import csv  # noqa: E402
from dataclasses import replace  # noqa: E402

from haemolynx.graph import short_loops  # noqa: E402
from haemolynx.gui.post_processing import (  # noqa: E402
    KEPT,
    LOOP_MATCH_UM,
    LOOP_REVIEW_COLUMNS,
    LOOP_TABLE_COLUMNS,
    LOOP_ZOOM_MARGIN,
    LOOP_ZOOM_MIN_BOX_UM,
    SIDE_DELETED,
    LoopReview,
    append_loop_review,
    loop_branch_ids,
    loop_label,
    loop_review_csv_path,
    loop_review_row,
    loop_side_rows,
    loop_view,
    loops_to_review,
)


def _loop_network() -> nx.MultiGraph:
    """The 1-{2,3}-4 loop of :func:`_network`, met at 1 and at 4 (a vessel
    on to node 6), with a 7 um vessel on one side."""
    G = _network()
    G.add_node(6, pos=np.asarray((0.0, 0.0, 40.0)))
    _add(G, 4, 6, branch_order="B01", diameter_um=5.0)
    for u, v, k in edge_keys(G):
        if {u, v} == {2, 4}:
            G.edges[u, v, k]["diameter_um"] = 7.0
    return G


def _the_loop(G):
    (loop,) = short_loops(G)
    return loop


def _ids(G, pairs) -> list[int]:
    return sorted(i for i, k in enumerate(edge_keys(G)) if set(k[:2]) in [set(p) for p in pairs])


def test_a_loop_is_listed_by_its_place_length_and_sides():
    loop = _the_loop(_loop_network())
    assert loop_label(loop, 3) == f"Loop 3: {loop.length_um:.3g} µm round, 2 side(s)"


def test_the_loop_table_lists_each_side_with_its_vessels_length_and_diameter():
    G = _loop_network()
    loop = _the_loop(G)

    rows = loop_side_rows(G, loop)

    assert len(rows) == 2 and all(len(row) == len(LOOP_TABLE_COLUMNS) for row in rows)
    assert [row[0] for row in rows] == ["1", "2"]
    by_side = {frozenset(n for e in side for n in e[:2]): row for side, row in zip(loop.sides, rows)}
    upper = by_side[frozenset({1, 2, 4})]
    assert sorted(int(i) for i in upper[1].split(", ")) == _ids(G, [(1, 2), (2, 4)])
    assert float(upper[2]) == pytest.approx(2 * np.hypot(10, 10), rel=1e-2)
    assert float(upper[3]) == pytest.approx(6.0, rel=1e-2), "length-weighted mean of 5 and 7"
    assert {upper[4], upper[5]} == {"1", "4"}


def test_loop_branch_ids_name_the_loop_or_one_side_and_skip_what_has_gone():
    G = _loop_network()
    loop = _the_loop(G)
    loop_ids = _ids(G, [(1, 2), (2, 4), (1, 3), (3, 4)])

    assert sorted(loop_branch_ids(G, loop)) == loop_ids
    assert sorted(loop_branch_ids(G, loop, 0) + loop_branch_ids(G, loop, 1)) == loop_ids
    assert loop_branch_ids(G, loop, 5) == []
    G.remove_edge(*loop.edges[0])
    assert len(loop_branch_ids(G, loop)) == 3


def test_a_review_matches_its_loop_by_position_and_length_not_node_ids():
    loop = _the_loop(_loop_network())
    review = LoopReview.of(loop, KEPT, time="2026-10-09T15:00:00")
    centre = np.asarray(loop.centre_um)

    assert review.matches(loop)
    assert review.matches(replace(loop, centre_um=tuple(centre + [0, 0.5 * LOOP_MATCH_UM, 0])))
    assert not review.matches(replace(loop, centre_um=tuple(centre + [0, 2 * LOOP_MATCH_UM, 0])))
    assert review.matches(replace(loop, length_um=loop.length_um * 1.03))
    assert not review.matches(replace(loop, length_um=loop.length_um * 1.2))
    renumbered = replace(loop, edges=(), sides=())
    assert renumbered != loop and review.matches(renumbered)


def test_a_review_survives_a_round_trip_through_plain_values():
    loop = _the_loop(_loop_network())
    review = LoopReview.of(loop, SIDE_DELETED, side=2, time="2026-10-09T15:00:00")

    assert LoopReview.from_dict(review.as_dict()) == review
    assert LoopReview.from_dict({"decision": KEPT}) is None
    assert LoopReview.from_dict("not a review") is None
    assert LoopReview.from_dict({**review.as_dict(), "centre_um": [1.0, 2.0]}) is None


def test_kept_loops_are_not_listed_again_but_an_opened_one_that_came_back_is():
    loop = _the_loop(_loop_network())

    assert loops_to_review([loop], []) == [loop]
    assert loops_to_review([loop], [LoopReview.of(loop, KEPT)]) == []
    assert loops_to_review([loop], [LoopReview.of(loop, SIDE_DELETED, side=1)]) == [loop]


def _tilted_circle(radius=10.0, centre=(5.0, 20.0, 30.0)):
    t = np.linspace(0.0, 2 * np.pi, 361)
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    normal = np.cross(e1, e2)
    points = np.asarray(centre) + radius * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    return points, normal


def test_the_loop_view_centres_fits_and_looks_along_the_loops_normal():
    points, normal = _tilted_circle()

    view = loop_view(points, towards=normal)

    assert view.centre_um == pytest.approx((5.0, 20.0, 30.0), abs=1e-6)
    assert view.box_um == pytest.approx(LOOP_ZOOM_MARGIN * 20.0, rel=1e-3)
    assert np.dot(view.view_direction, normal) == pytest.approx(1.0, abs=1e-6)
    assert np.dot(view.up_direction, normal) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(view.up_direction) == pytest.approx(1.0)
    flipped = loop_view(points, towards=-normal)
    assert np.dot(flipped.view_direction, normal) == pytest.approx(-1.0, abs=1e-6)


def test_the_loop_view_keeps_the_cameras_up_laid_into_the_loops_plane():
    points, normal = _tilted_circle()
    wanted = np.array([0.0, -1.0, 0.0])  # already in the plane

    view = loop_view(points, towards=normal, up=wanted)

    assert view.up_direction == pytest.approx(tuple(wanted), abs=1e-6)
    along_normal = loop_view(points, towards=normal, up=normal)
    assert np.dot(along_normal.up_direction, normal) == pytest.approx(0.0, abs=1e-6)


def test_a_small_loop_is_zoomed_to_no_tighter_than_the_smallest_box():
    points, _normal = _tilted_circle(radius=2.0)
    assert loop_view(points).box_um == LOOP_ZOOM_MIN_BOX_UM


def test_a_review_row_has_every_column_with_the_loop_its_sides_and_the_decision():
    G = _loop_network()
    loop = _the_loop(G)
    review = LoopReview.of(loop, SIDE_DELETED, side=1, time="2026-10-09T15:00:00")

    row = loop_review_row(G, loop, review)

    assert list(row) == list(LOOP_REVIEW_COLUMNS)
    assert row["decision"] == SIDE_DELETED and row["side_deleted"] == "1"
    assert row["time"] == "2026-10-09T15:00:00"
    assert float(row["loop_length_um"]) == pytest.approx(loop.length_um, rel=1e-3)
    assert row["sides"] == "2"
    assert sorted(int(i) for i in row["loop_branch_ids"].split(";")) == sorted(loop_branch_ids(G, loop))
    assert row["centre_zyx_vox"] == "0 0 20"
    assert row["zero_resistance_bridges"] == "0"
    # Without a mask, what the mask would say is left blank.
    assert row["path_in"] == "" and row["wall_side"] == "" and row["inside_one_lumen"] == ""
    assert float(row["out_of_plane"]) == pytest.approx(0.0, abs=1e-6)


def _shifted(G, offset):
    """*G* moved by *offset* microns, so all of it sits inside a mask."""
    H = G.copy()
    for node in H.nodes:
        H.nodes[node]["pos"] = np.asarray(H.nodes[node]["pos"], dtype=float) + offset
    for _u, _v, _k, data in H.edges(keys=True, data=True):
        data["voxels"] = [tuple(np.asarray(p, dtype=float) + offset) for p in data["voxels"]]
    return H


def _tube_round(G, edges, shape, radius=2.0) -> np.ndarray:
    """A mask of tubes *radius* microns round *edges* of *G*, on 1 um voxels."""
    points = np.indices(shape).reshape(3, -1).T.astype(float)
    inside = np.zeros(len(points), dtype=bool)
    for a, b, _k in edges:
        pa, pb = (np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b))
        t = np.clip((points - pa) @ (pb - pa) / np.dot(pb - pa, pb - pa), 0.0, 1.0)
        inside |= np.linalg.norm(points - (pa + t[:, None] * (pb - pa)), axis=1) <= radius
    return inside.reshape(shape)


def test_a_review_row_reads_the_mask_round_the_loop_when_given_one():
    H = _shifted(_loop_network(), np.array([4.0, 15.0, 5.0]))
    loop = _the_loop(H)
    mask = _tube_round(H, loop.edges, (9, 31, 40))

    row = loop_review_row(H, loop, LoopReview.of(loop, KEPT), mask=mask)

    assert float(row["path_in"]) == pytest.approx(1.0)
    assert row["inside_one_lumen"] == "False", "a loop round tissue"
    assert float(row["chords_in_lumen"]) < 0.5
    assert row["wall_side"] in {"1", "2"}
    assert len(row["side_lumen_r_um"].split(";")) == 2


def test_the_loop_review_csv_sits_beside_the_vtk_output(tmp_path):
    expected = tmp_path / "run1_loop_review.csv"
    assert loop_review_csv_path({"vtk_output_prefix": str(tmp_path / "run1")}) == expected
    assert loop_review_csv_path({"vtk_output_prefix": str(tmp_path / "missing" / "run1")}) is None
    assert loop_review_csv_path({}) is None and loop_review_csv_path(None) is None


def test_decisions_collect_in_one_csv_under_one_header(tmp_path):
    G = _loop_network()
    loop = _the_loop(G)
    first = loop_review_row(G, loop, LoopReview.of(loop, KEPT, time="t1"))
    second = loop_review_row(G, loop, LoopReview.of(loop, SIDE_DELETED, side=2, time="t2"))
    path = tmp_path / "run_loop_review.csv"

    append_loop_review(path, [first])
    append_loop_review(path, [second])

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["decision"] for r in rows] == [KEPT, SIDE_DELETED]
    assert rows[0] == first and rows[1] == second
    assert path.read_text(encoding="utf-8").count("loop_length_um") == 1
