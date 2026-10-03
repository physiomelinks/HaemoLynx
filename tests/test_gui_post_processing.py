"""What the "6. Post processing" tab draws and lists (pure, no napari)."""
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
    for start in ("post_process", "build_haemodynamic_model", "export_results", None, "nope"):
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
