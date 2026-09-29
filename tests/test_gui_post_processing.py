"""What the "10. Post processing" tab draws and lists (pure, no napari)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import edge_keys
from haemolynx.graph._helpers import calculate_path_length
from haemolynx.gui.post_processing import (
    AT_JUNCTION,
    CONNECTED,
    HIGH_DEGREE_JUNCTIONS,
    JUNCTION_TABLE_COLUMNS,
    SELECTED,
    STATUS_COLOURS,
    boundaries_following_graph,
    camera_center_for,
    junction_label,
    junction_marker_layer,
    junction_table_rows,
    nearest_node,
    parse_branch_ids,
    scan_network,
    status_colours,
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


def test_boundaries_following_graph_drops_pruned_boundary_nodes():
    from haemolynx.pipeline.stages import PipelineResume

    G = _network()
    G.remove_node(0)  # say a prune took inlet 0 with it
    resume = PipelineResume(
        start_from="assign_diameters", graph=G,
        inlet_nodes=(0, 2), outlet_nodes=(4,),
        arteriole_boundary_nodes=(0, 1), venule_boundary_nodes=(3,),
        resistance_node_pair=(0, 4),
    )
    after = boundaries_following_graph(resume)
    assert after.inlet_nodes == (2,) and after.outlet_nodes == (4,)
    assert after.arteriole_boundary_nodes == (1,) and after.venule_boundary_nodes == (3,)
    assert after.resistance_node_pair == (2, 4)  # re-picked: 0 is gone
    assert after.graph is G and after.start_from == "assign_diameters"
    # A pair that survived is left alone.
    kept = boundaries_following_graph(
        PipelineResume(start_from="assign_diameters", graph=G, inlet_nodes=(2,),
                       outlet_nodes=(4,), resistance_node_pair=(1, 4))
    )
    assert kept.resistance_node_pair == (1, 4)
