"""The "10. Post processing" tab, built for real, with a viewer.

The graph rules are pinned in ``test_graph_post_processing.py`` and the
colours in ``test_gui_post_processing.py``; these check the Qt glue between
them: a scan lists the 4+ junctions and recolours the vessels layer, the
table turns vessels yellow, each button changes the network the viewer
shows, and clicks in the viewer delete and add vessels. Marked ``gui`` like
the other ``*_widget.py`` tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.graph import edge_keys  # noqa: E402
from haemolynx.graph._helpers import calculate_path_length  # noqa: E402
from haemolynx.gui._widget import (  # noqa: E402
    POST_PROCESSING_TAB,
    _apply_layers,
    _post_processing_controls,
    settings_widget,
)
from haemolynx.gui.post_processing import (  # noqa: E402
    AT_JUNCTION,
    CONNECTED,
    HIGH_DEGREE_JUNCTIONS,
    SELECTED,
    STATUS_COLOURS,
)
from haemolynx.gui.results import VESSELS, ResultLayers  # noqa: E402

from test_gui_results import network  # noqa: E402

pytestmark = pytest.mark.gui


def _four_way_network() -> nx.MultiGraph:
    """Inlet 0 -> 4-way junction 1 -> {2, 3} -> outlet 4, plus 1 -> 5."""
    G = nx.MultiGraph()
    positions = {
        0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 10, 0), 3: (20, -10, 0),
        4: (30, 0, 0), 5: (10, 20, 0),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5)]:
        voxels = [tuple(map(float, positions[u])), tuple(map(float, positions[v]))]
        G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels),
                   branch_order="B01", diameter_um=5.0)
    return G


@pytest.fixture
def page(make_napari_viewer):
    viewer = make_napari_viewer()
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(_four_way_network())))
    report = SimpleNamespace(value="")
    regenerated: list = []
    controls = _post_processing_controls(
        viewer,
        report,
        results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (4,)},
        regenerate=regenerated.append,
        running=lambda: False,
    )
    return SimpleNamespace(
        controls=controls, viewer=viewer, results=results, report=report,
        regenerated=regenerated,
    )


def _select_rows(table, rows) -> None:
    """Select *rows* together, as ctrl-clicking each would (``selectRow``
    alone replaces the selection in extended-selection mode)."""
    from qtpy.QtCore import QItemSelectionModel

    table.clearSelection()
    flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
    for row in rows:
        table.selectionModel().select(table.model().index(row, 0), flags)


def _row_to(table, other_end: str) -> int:
    return next(r for r in range(table.rowCount()) if table.item(r, 4).text() == other_end)


def _colour_of(viewer, graph, pair) -> tuple:
    """The colour the vessels layer draws the vessel between *pair* in."""
    layer = viewer.layers[VESSELS]
    keys = edge_keys(graph)
    index = next(i for i, key in enumerate(keys) if set(key[:2]) == set(pair))
    segment = list(np.asarray(layer.features["edge_index"])).index(index)
    return tuple(np.round(np.asarray(layer.edge_color)[segment], 3))


def _rgba(status) -> tuple:
    return tuple(np.round(STATUS_COLOURS[status], 3))


def _click(page, position):
    event = SimpleNamespace(position=position, dims_displayed=(0, 1, 2), view_direction=None)
    page.controls.on_click(page.viewer.layers[VESSELS], event)


def test_the_panel_ends_with_the_post_processing_tab(make_napari_viewer):
    from qtpy.QtWidgets import QStackedWidget, QTabWidget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    tabs = panel.findChild(QTabWidget)
    assert tabs.tabText(tabs.count() - 1) == POST_PROCESSING_TAB
    # One Revert page per tab, so the chrome below still follows the tabs.
    stack = panel.findChild(QStackedWidget, "haemolynx_revert_stack")
    assert stack.count() == tabs.count()
    assert panel._haemolynx_post_processing.scan_button.toolTip()
    # Editing moved into the tab: the old Edit button is no longer shown.
    assert not panel._haemolynx_edit_button.visible
    assert panel._haemolynx_post_processing.add_button.toolTip()


def test_scan_before_a_run_reports_instead_of_raising(make_napari_viewer):
    viewer = make_napari_viewer()
    controls = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: None,
        boundary_roles=lambda: {}, regenerate=lambda graph: None, running=lambda: False,
    )
    controls.scan_button.click()
    assert "Nothing to check yet" in controls.status.text()
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers


def test_scan_lists_the_junction_recolours_the_vessels_and_zooms(page):
    c, viewer = page.controls, page.viewer
    layer_before = viewer.layers[VESSELS]
    c.scan_button.click()

    assert c.junction_list.count() == 1
    assert c.junction_list.item(0).text() == "Node 1 - 4 vessels"
    assert c.status.text() == "1 junction(s) where 4+ vessels meet, in 6 vessels."
    # The first junction is picked for you: its four vessels are listed.
    assert c.table.rowCount() == 4
    assert {c.table.item(r, 4).text() for r in range(4)} == {"0", "2", "3", "5"}

    # The vessels layer on screen is recoloured, not rebuilt or hidden.
    assert viewer.layers[VESSELS] is layer_before and layer_before.visible
    graph = c.state.graph
    assert _colour_of(viewer, graph, (2, 4)) == _rgba(CONNECTED)
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(AT_JUNCTION)
    assert np.allclose(viewer.layers[HIGH_DEGREE_JUNCTIONS].data, [[10, 0, 0]])
    centre = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(centre, [10, 0, 0][-len(viewer.dims.displayed):], atol=1e-6)


def test_selecting_table_rows_turns_those_vessels_yellow(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    graph = c.state.graph
    _select_rows(c.table, [_row_to(c.table, "2"), _row_to(c.table, "5")])
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(SELECTED)
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(SELECTED)
    assert _colour_of(viewer, graph, (1, 3)) == _rgba(AT_JUNCTION)
    _select_rows(c.table, [])
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(AT_JUNCTION)


def test_delete_selected_vessels_updates_the_network(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    _select_rows(c.table, [_row_to(c.table, "5")])
    c.delete_button.click()

    graph = c.state.graph
    assert 5 not in graph and graph.degree(1) == 3
    assert c.junction_list.count() == 0
    assert c.status.text() == "0 junction(s) where 4+ vessels meet, in 5 vessels."
    assert len(viewer.layers[VESSELS].data) == 5
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers  # no 4+ junction left
    assert "deleted 1 vessel(s) at node 1" in page.report.value


def test_delete_refuses_to_strand_the_inlet(page):
    c = page.controls
    c.scan_button.click()
    _select_rows(c.table, [_row_to(c.table, "0")])
    before = sorted(edge_keys(c.state.graph))
    c.delete_button.click()
    assert "boundary node" in c.status.text()
    assert sorted(edge_keys(c.state.graph)) == before


def test_split_turns_the_junction_into_bifurcations(page):
    c = page.controls
    c.scan_button.click()
    c.connector.setValue(12.0)
    c.split_button.click()

    graph = c.state.graph
    assert graph.degree(1) == 3
    connectors = [d for _u, _v, d in graph.edges(data=True) if d.get("junction_split_connector")]
    assert len(connectors) == 1 and connectors[0]["length"] == pytest.approx(12.0)
    assert c.junction_list.count() == 0
    assert "split node 1" in page.report.value


def test_leave_as_is_marks_the_junction(page):
    c = page.controls
    c.scan_button.click()
    c.leave_button.click()
    assert c.junction_list.item(0).text() == "Node 1 - 4 vessels (left as is)"
    assert c.state.graph.degree(1) == 4


def test_clicking_a_vessel_in_delete_mode_removes_it(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.click_delete_button.click()
    assert "click a vessel" in c.edit_status.text()
    _click(page, (10.0, 10.0, 0.0))  # halfway along the 1 -> 5 vessel

    graph = c.state.graph
    assert 5 not in graph and graph.degree(1) == 3
    assert len(viewer.layers[VESSELS].data) == 5
    c.stop_button.click()
    _click(page, (25.0, 5.0, 0.0))  # not armed any more: nothing happens
    assert graph.number_of_edges() == 5


def test_click_delete_never_strands_the_outlet(page):
    c = page.controls
    c.scan_button.click()
    c.click_delete_button.click()
    _click(page, (25.0, 5.0, 0.0))  # one of the two vessels into outlet 4
    assert c.state.graph.degree(4) == 1
    _click(page, (25.0, -5.0, 0.0))  # the last one: refused
    assert c.state.graph.degree(4) == 1
    assert "boundary node" in c.edit_status.text()


def test_add_vessel_joins_two_clicked_nodes_with_their_mean_diameter(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    _click(page, (10.0, 20.0, 0.0))  # node 5
    assert "Start node 5 picked" in c.edit_status.text()
    _click(page, (20.0, -10.0, 0.0))  # node 3

    graph = c.state.graph
    added = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("post_processing_added")
    ]
    assert len(added) == 1 and {added[0][0], added[0][1]} == {3, 5}
    data = added[0][2]
    # No image layer here, so it is drawn straight: 5 -> 3 is sqrt(10^2 + 30^2).
    assert data["length"] == pytest.approx(np.hypot(10.0, 30.0))
    # Every vessel at nodes 3 and 5 is 5 um across, so the mean is 5 um.
    assert data["diameter_um"] == pytest.approx(5.0)
    assert data["diameter_source"] == "override"
    assert "as a straight line" in c.edit_status.text()
    assert len(viewer.layers[VESSELS].data) == 7


def test_add_vessel_ignores_a_click_on_the_same_node_twice(page):
    c = page.controls
    c.scan_button.click()
    c.add_button.click()
    _click(page, (10.0, 20.0, 0.0))
    _click(page, (10.0, 20.0, 0.0))
    assert "node 5 again" in c.edit_status.text()
    assert c.state.graph.number_of_edges() == 6


def test_add_vessel_routes_through_the_segmented_image(make_napari_viewer):
    """With the segmented image loaded, a new vessel follows it."""
    viewer = make_napari_viewer()
    mask = np.zeros((1, 25, 35), dtype=np.uint8)
    mask[0, 0, 10:31] = 1          # an L-shaped vessel from (0, 0, 10) ...
    mask[0, 0:21, 30] = 1
    mask[0, 20, 10:31] = 1         # ... round to (0, 20, 10)
    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 20, 0), 3: (0, 20, 10)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in [(0, 1), (2, 3)]:
        voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
        G.add_edge(u, v, voxels=voxels, length=10.0, diameter_um=4.0)
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=mask, skeleton=np.zeros_like(mask, dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0),
        ),
    ))
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (2,)},
        regenerate=lambda graph: None, running=lambda: False,
    )
    c.scan_button.click()
    c.add_button.click()
    page = SimpleNamespace(controls=c, viewer=viewer)
    _click(page, (0.0, 0.0, 10.0))   # node 1
    _click(page, (0.0, 20.0, 10.0))  # node 3

    added = [d for _u, _v, d in c.state.graph.edges(data=True) if d.get("post_processing_added")]
    assert len(added) == 1
    assert "routed through the image" in c.edit_status.text()
    assert max(p[2] for p in added[0]["voxels"]) >= 29   # went round the L
    assert added[0]["length"] > 50
    assert added[0]["diameter_um"] == pytest.approx(4.0)


def test_regenerate_hands_over_the_edited_graph_and_clears_the_tab(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.split_button.click()
    edited = c.state.graph
    c.regenerate_button.click()

    assert page.regenerated == [edited]
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers
    assert c.on_click not in viewer.layers[VESSELS].mouse_drag_callbacks
    assert c.state.graph is None and c.junction_list.count() == 0
