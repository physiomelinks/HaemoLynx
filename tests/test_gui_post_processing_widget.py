"""The "6. Post processing" tab, built for real, with a viewer.

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
    ADDED,
    ADDED_NODES,
    AT_JUNCTION,
    CONNECTED,
    HIGH_DEGREE_JUNCTIONS,
    NEW_VESSEL_POINTS,
    NEW_VESSEL_TRACE,
    SELECTED,
    STATUS_COLOURS,
    TRACE_SOURCES,
    TRACE_THROUGH_RAW,
)
from haemolynx.gui.results import FWHM_RAW, VESSELS, ResultLayers  # noqa: E402

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
    """The tab on the four-way network, its junction correction ticked."""
    return _four_way_page(make_napari_viewer(), junction_correction=True)


def _four_way_page(viewer, *, junction_correction: bool):
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(_four_way_network())))
    report = SimpleNamespace(value="")
    regenerated: list = []
    stops: list = []
    #: What the panel would say: a run paused here, one finished, whether a
    #: run started when asked.
    run = SimpleNamespace(paused=False, complete=True, starts=True)

    def regenerate(graph, stop_after=None):
        regenerated.append(graph)
        stops.append(stop_after)
        return run.starts

    controls = _post_processing_controls(
        viewer,
        report,
        results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (4,)},
        regenerate=regenerate,
        running=lambda: False,
        paused=lambda: run.paused,
        complete=lambda: run.complete,
    )
    controls.refresh()
    controls.junction_toggle.setChecked(junction_correction)
    return SimpleNamespace(
        controls=controls, viewer=viewer, results=results, report=report,
        regenerated=regenerated, stops=stops, run=run,
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


def test_the_post_processing_tab_sits_between_diameters_and_haemodynamics(make_napari_viewer):
    from qtpy.QtWidgets import QStackedWidget, QTabWidget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    tabs = panel.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert POST_PROCESSING_TAB == "6. Post processing"
    at = titles.index(POST_PROCESSING_TAB)
    assert titles[at - 1] == "5. Diameters" and titles[at + 1] == "7. Haemodynamics"
    assert titles[-1] == "10. Export"
    # Its own buttons start a run there, not "Run from this stage".
    assert POST_PROCESSING_TAB not in panel._haemolynx_revert_buttons
    # One Revert page per tab, so the chrome below still follows the tabs.
    stack = panel.findChild(QStackedWidget, "haemolynx_revert_stack")
    assert stack.count() == tabs.count()
    assert panel._haemolynx_post_processing.scan_button.toolTip()
    # Editing moved into the tab: the old Edit button is no longer shown.
    assert not panel._haemolynx_edit_button.visible
    controls = panel._haemolynx_post_processing
    assert controls.add_button.toolTip() and controls.trace_source.toolTip()
    # Add vessel finishes by itself and both modes are toggles: no Stop button.
    assert not hasattr(controls, "stop_button")
    assert controls.add_button.isCheckable() and controls.click_delete_button.isCheckable()
    items = [controls.trace_source.itemText(i) for i in range(controls.trace_source.count())]
    assert items == list(TRACE_SOURCES)


def test_scan_before_a_run_reports_instead_of_raising(make_napari_viewer):
    viewer = make_napari_viewer()
    controls = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: None,
        boundary_roles=lambda: {}, regenerate=lambda graph: None, running=lambda: False,
    )
    controls.scan_button.click()
    assert "Nothing to check yet" in controls.status.text()
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers


def test_junction_correction_is_off_and_hidden_until_ticked(make_napari_viewer):
    """Unticked, a scan picks no junction, marks none, tints none cyan and
    leaves the camera where it was; ticking brings all of it, unticking
    takes it away again. The other edits work either way."""
    page = _four_way_page(make_napari_viewer(), junction_correction=False)
    c, viewer = page.controls, page.viewer
    assert c.junction_toggle.text() == "Manual 4+ vessel junction correction"
    assert c.junction_toggle.toolTip()
    fresh = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: None,
        boundary_roles=lambda: {}, regenerate=lambda graph: None, running=lambda: False,
    )
    assert not fresh.junction_toggle.isChecked(), "off by default"
    assert fresh.junction_box.isHidden()
    for widget in (c.junction_list, c.table, c.delete_button, c.leave_button,
                   c.split_button, c.connector):
        assert c.junction_box.isAncestorOf(widget)
    for widget in (c.scan_button, c.click_delete_button, c.add_button, c.prune_button):
        assert not c.junction_box.isAncestorOf(widget)

    centre_before = np.asarray(viewer.camera.center, dtype=float).copy()
    c.scan_button.click()
    graph = c.state.graph
    assert graph is not None
    assert c.state.node is None and c.table.rowCount() == 0
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(CONNECTED)
    assert np.allclose(viewer.camera.center, centre_before)
    assert c.status.text() == "1 junction(s) where 4+ vessels meet, in 6 vessels."

    # An edit rescans: still no junction picked or marked.
    keys = edge_keys(graph)
    c.branch_ids.setText(str(next(i for i, k in enumerate(keys) if set(k[:2]) == {2, 4})))
    c.delete_ids_button.click()
    assert c.state.graph.degree(1) == 4
    assert c.state.node is None and HIGH_DEGREE_JUNCTIONS not in viewer.layers

    c.junction_toggle.setChecked(True)
    graph = c.state.graph
    assert not c.junction_box.isHidden()
    assert c.state.node == 1 and c.table.rowCount() == 4
    assert np.allclose(viewer.layers[HIGH_DEGREE_JUNCTIONS].data, [[10, 0, 0]])
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(AT_JUNCTION)
    centre = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(centre, [10, 0, 0][-len(viewer.dims.displayed):], atol=1e-6)

    c.junction_toggle.setChecked(False)
    assert c.junction_box.isHidden()
    assert c.state.node is None and c.table.rowCount() == 0
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(CONNECTED)


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
    assert c.click_delete_button.isChecked()  # still armed for the next one
    c.click_delete_button.click()  # pressed again: stops
    assert not c.click_delete_button.isChecked()
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


def _added(graph):
    return [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("post_processing_added")]


def test_add_vessel_from_a_node_to_a_node_finishes_by_itself(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    assert c.add_button.isChecked()
    _click(page, (10.0, 20.0, 0.0))  # node 5
    assert "Starting at node 5" in c.edit_status.text()
    assert list(viewer.layers[NEW_VESSEL_POINTS].features["role"]) == ["start node"]
    _click(page, (20.0, -10.0, 0.0))  # node 3: the vessel is finished

    graph = c.state.graph
    added = _added(graph)
    assert len(added) == 1 and {added[0][0], added[0][1]} == {3, 5}
    data = added[0][2]
    # No image layer here, so it is drawn straight: 5 -> 3 is sqrt(10^2 + 30^2).
    assert data["length"] == pytest.approx(np.hypot(10.0, 30.0))
    # Every vessel at nodes 3 and 5 is 5 um across, so the mean is 5 um.
    assert data["diameter_um"] == pytest.approx(5.0)
    assert data["diameter_source"] == "override"
    assert "as a straight line" in c.edit_status.text()
    assert len(viewer.layers[VESSELS].data) == 7
    # Finished: the mode ends and the trace is no longer drawn.
    assert c.state.mode == "idle" and not c.add_button.isChecked()
    assert c.state.trace is None
    assert NEW_VESSEL_TRACE not in viewer.layers and NEW_VESSEL_POINTS not in viewer.layers
    assert _colour_of(viewer, graph, (3, 5)) == _rgba(ADDED)
    assert "Added branchID" in c.log_box.toPlainText()


def test_add_vessel_refuses_to_end_where_it_started_and_can_be_cancelled(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    _click(page, (10.0, 20.0, 0.0))
    _click(page, (10.0, 20.0, 0.0))
    assert "start and end at node 5" in c.edit_status.text()
    assert c.state.graph.number_of_edges() == 6
    assert c.state.trace is not None  # still tracing
    c.add_button.click()  # pressed again: cancelled
    assert c.state.trace is None and c.state.mode == "idle"
    assert "cancelled" in c.edit_status.text()
    assert NEW_VESSEL_POINTS not in viewer.layers
    assert c.state.graph.number_of_edges() == 6


def test_add_vessel_from_one_vessel_to_another_forms_a_node_on_each(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    _click(page, (10.0, 10.0, 0.0))  # halfway along 1 -> 5
    assert "a new node on branchID" in c.edit_status.text()
    assert list(viewer.layers[NEW_VESSEL_POINTS].features["role"]) == ["new node"]
    assert c.state.graph.number_of_edges() == 6  # nothing changes until it finishes

    _click(page, (0.0, 10.0, 0.0))  # a point along the way
    assert "Point 1 traced as a straight line" in c.edit_status.text()
    trace = viewer.layers[NEW_VESSEL_TRACE].data
    assert np.allclose(trace[0, 0], (10, 10, 0))
    assert np.allclose(trace[-1, 0] + trace[-1, 1], (0, 10, 0))
    assert len(viewer.layers[NEW_VESSEL_POINTS].data) == 2

    _click(page, (25.0, 5.0, 0.0))  # halfway along 2 -> 4: finished there
    graph = c.state.graph
    new_nodes = c.state.added_nodes
    assert len(new_nodes) == 2
    assert np.allclose(graph.nodes[new_nodes[0]]["pos"], (10, 10, 0))
    assert np.allclose(graph.nodes[new_nodes[1]]["pos"], (25, 5, 0))
    # Both vessels cut in two, each half measured again; plus the new vessel.
    assert graph.number_of_edges() == 9
    assert not graph.has_edge(1, 5) and not graph.has_edge(2, 4)
    assert graph.edges[1, new_nodes[0], 0]["length"] == pytest.approx(10.0)
    assert graph.edges[new_nodes[0], 5, 0]["length"] == pytest.approx(10.0)
    assert graph.edges[2, new_nodes[1], 0]["length"] == pytest.approx(np.hypot(5.0, 5.0))
    (u, v, data), = _added(graph)
    assert {u, v} == set(new_nodes)
    assert data["length"] == pytest.approx(10.0 + np.hypot(25.0, 5.0))
    assert data["diameter_um"] == pytest.approx(5.0)
    # The new nodes stay marked until Regenerate, and the log says what was cut.
    assert np.allclose(viewer.layers[ADDED_NODES].data, [(10, 10, 0), (25, 5, 0)])
    log = c.log_box.toPlainText()
    assert f"new node(s) {new_nodes[0]}, {new_nodes[1]} cut branchID" in log
    assert "in 3 clicks" in log


def test_a_drag_turns_the_view_and_adds_nothing(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    layer = viewer.layers[VESSELS]
    assert c.on_press in layer.mouse_drag_callbacks

    def press_and_release(moved_to):
        event = SimpleNamespace(
            position=(10.0, 20.0, 0.0), dims_displayed=(0, 1, 2), view_direction=None,
            pos=(100.0, 100.0), type="mouse_press", button=1,
        )
        gen = c.on_press(layer, event)
        next(gen)
        event.type, event.pos = "mouse_move", moved_to
        next(gen)
        event.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(gen)

    press_and_release((160.0, 100.0))  # dragged: turning the view
    assert c.state.trace is None
    press_and_release((101.0, 100.0))  # a twitch of the hand: still a click
    assert c.state.trace is not None and c.state.trace.start.node == 5


def _l_shaped_network():
    """Two short vessels whose ends 1 (0, 0, 10) and 3 (0, 20, 10) an
    L-shaped piece of image joins, out round x = 30."""
    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 20, 0), 3: (0, 20, 10)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in [(0, 1), (2, 3)]:
        voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
        G.add_edge(u, v, voxels=voxels, length=10.0, diameter_um=4.0)
    return G


def _l_mask():
    mask = np.zeros((1, 25, 35), dtype=np.uint8)
    mask[0, 0, 10:31] = 1          # an L-shaped vessel from (0, 0, 10) ...
    mask[0, 0:21, 30] = 1
    mask[0, 20, 10:31] = 1         # ... round to (0, 20, 10)
    return mask


def _l_page(viewer, *, image=None, settings=None):
    results = ResultLayers()
    if image is not None:
        _apply_layers(viewer, results.stage_finished(
            "skeletonise",
            SimpleNamespace(
                image=image, skeleton=np.zeros_like(image, dtype=bool),
                voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0),
            ),
        ))
    _apply_layers(viewer, results.stage_finished("build_network", network(_l_shaped_network())))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (2,)},
        regenerate=lambda graph: None, running=lambda: False, settings=settings,
    )
    c.scan_button.click()
    c.add_button.click()
    return SimpleNamespace(controls=c, viewer=viewer)


def test_add_vessel_traces_each_leg_through_the_segmented_mask(make_napari_viewer):
    """Each click is traced from the last through the mask, drawn at once."""
    page = _l_page(make_napari_viewer(), image=_l_mask())
    c, viewer = page.controls, page.viewer
    _click(page, (0.0, 0.0, 10.0))   # node 1
    _click(page, (0.0, 20.0, 30.0))  # the far corner of the L
    assert "Point 1 traced through the segmented mask" in c.edit_status.text()
    first_leg = viewer.layers[NEW_VESSEL_TRACE].data
    ends = first_leg[:, 0] + first_leg[:, 1]
    assert ends[:, 2].max() >= 29 and np.allclose(ends[-1], (0, 20, 30))
    # It went along the mask (out to x = 30, then down), not straight across.
    assert (first_leg[:, 0, 1] < 1).sum() >= 15

    _click(page, (0.0, 20.0, 10.0))  # node 3: finished
    (u, v, data), = _added(c.state.graph)
    assert {u, v} == {1, 3}
    assert "through the segmented mask" in c.edit_status.text()
    assert max(p[2] for p in data["voxels"]) >= 29
    assert data["length"] > 50
    assert data["diameter_um"] == pytest.approx(4.0)
    # Smoothed as the pipeline smooths its own centrelines.
    assert data["centreline_smoothing"] in ("smoothed", "relaxed", "kept_raw")


def _bright_l():
    raw = np.random.default_rng(0).uniform(0.0, 20.0, size=(1, 25, 35)).astype(np.float32)
    raw[_l_mask().astype(bool)] = 200.0
    return raw


def test_add_vessel_traces_through_the_raw_data_when_chosen(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(_bright_l(), name=FWHM_RAW)
    page = _l_page(viewer)  # no segmented image: only the raw data to follow
    c = page.controls
    c.trace_source.setCurrentText(TRACE_THROUGH_RAW)
    _click(page, (0.0, 0.0, 10.0))
    _click(page, (0.0, 20.0, 10.0))
    (_u, _v, data), = _added(c.state.graph)
    assert "through the raw data" in c.edit_status.text()
    assert max(p[2] for p in data["voxels"]) >= 29


def test_add_vessel_reads_the_raw_data_file_from_the_settings(make_napari_viewer, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, _bright_l())
    settings = {"fwhm_raw_tiff_path": str(path), "image_axis_order": "zyx"}
    page = _l_page(make_napari_viewer(), image=_l_mask(), settings=lambda: settings)
    c = page.controls
    c.trace_source.setCurrentText(TRACE_THROUGH_RAW)
    _click(page, (0.0, 0.0, 10.0))
    _click(page, (0.0, 20.0, 10.0))
    assert "through the raw data" in c.edit_status.text()
    assert c.state.raw is not None and c.state.raw[0].shape == (1, 25, 35)


def test_add_vessel_without_raw_data_says_so_and_uses_the_mask(make_napari_viewer, tmp_path):
    settings = {"image_axis_order": "zyx"}
    page = _l_page(make_napari_viewer(), image=_l_mask(), settings=lambda: settings)
    c = page.controls
    c.trace_source.setCurrentText(TRACE_THROUGH_RAW)
    _click(page, (0.0, 0.0, 10.0))
    _click(page, (0.0, 20.0, 30.0))
    text = c.edit_status.text()
    assert "through the segmented mask" in text and "no raw data to trace through" in text
    assert "Raw data file" in text

    # A Raw data file set afterwards is read once Raw data is chosen again.
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "raw.tif", _bright_l())
    settings["fwhm_raw_tiff_path"] = str(tmp_path / "raw.tif")
    c.trace_source.setCurrentIndex(0)
    c.trace_source.setCurrentText(TRACE_THROUGH_RAW)
    _click(page, (0.0, 20.0, 20.0))
    assert "Point 2 traced through the raw data" in c.edit_status.text()


def test_delete_by_branch_id_removes_the_vessels_typed(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    keys = edge_keys(c.state.graph)
    dead_end = next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})
    c.branch_ids.setText(str(dead_end))
    c.delete_ids_button.click()

    graph = c.state.graph
    assert 5 not in graph and graph.number_of_edges() == 5
    assert len(viewer.layers[VESSELS].data) == 5
    assert f"Deleted branchID(s) {dead_end}" in c.edit_status.text()
    assert c.branch_ids.text() == ""


def test_delete_by_branch_id_reports_bad_input_and_protects_boundaries(page):
    c = page.controls
    c.scan_button.click()
    c.branch_ids.setText("99")
    c.delete_ids_button.click()
    assert "branchID 99 is not in this network (0 to 5)" in c.edit_status.text()
    keys = edge_keys(c.state.graph)
    into_outlet = [i for i, k in enumerate(keys) if 4 in k[:2]]
    c.branch_ids.setText(", ".join(map(str, into_outlet)))
    c.branch_ids.returnPressed.emit()
    assert "boundary node" in c.edit_status.text()
    assert c.state.graph.number_of_edges() == 6
    assert c.branch_ids.text() != ""  # kept, so it can be corrected


def test_prune_removes_a_branch_a_delete_cut_off(make_napari_viewer):
    """1 -> 5 -> 6 hangs off the junction; deleting 1 -> 5 cuts 5 -> 6 off."""
    viewer = make_napari_viewer()
    G = _four_way_network()
    G.add_node(6, pos=np.asarray((10.0, 30.0, 0.0)))
    G.add_edge(5, 6, voxels=[(10.0, 20.0, 0.0), (10.0, 30.0, 0.0)], length=10.0,
               branch_order="B02", diameter_um=5.0)
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    report = SimpleNamespace(value="")
    c = _post_processing_controls(
        viewer, report, results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (4,)},
        regenerate=lambda graph: None, running=lambda: False,
    )
    c.scan_button.click()
    c.prune_button.click()
    assert "Nothing to prune" in c.status.text()

    keys = edge_keys(c.state.graph)
    c.branch_ids.setText(str(next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})))
    c.delete_ids_button.click()
    assert c.state.graph.has_edge(5, 6)  # cut off, but still there
    c.prune_button.click()

    graph = c.state.graph
    assert 5 not in graph and 6 not in graph
    assert graph.number_of_edges() == 5
    assert len(viewer.layers[VESSELS].data) == 5
    assert "pruned 1 disconnected piece(s), 1 vessel(s)" in report.value


def test_the_log_records_each_change_with_its_branch_ids(make_napari_viewer):
    viewer = make_napari_viewer()
    G = _four_way_network()
    G.add_node(6, pos=np.asarray((10.0, 30.0, 0.0)))
    G.add_edge(5, 6, voxels=[(10.0, 20.0, 0.0), (10.0, 30.0, 0.0)], length=10.0,
               branch_order="B02", diameter_um=5.0)
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (4,)},
        regenerate=lambda graph: None, running=lambda: False,
    )
    c.junction_toggle.setChecked(True)
    c.scan_button.click()
    keys = edge_keys(c.state.graph)
    to_5 = next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})

    _select_rows(c.table, [_row_to(c.table, "5")])
    c.delete_button.click()
    c.prune_button.click()
    c.branch_ids.setText("0")          # the vessel into inlet 0: refused
    c.delete_ids_button.click()

    lines = c.log_box.toPlainText().splitlines()
    assert "Scanned the network: 1 junction(s) where 4+ vessels meet, in 7 vessels." in lines[0]
    assert f"Node 1 (4 vessels): deleted branchID {to_5} (node 1-5, 20 µm, 5 µm)" in lines[1]
    # The prune names the vessel it removed by its branchID at the time.
    assert "Pruned 1 disconnected piece(s), 1 vessel(s): branchID" in lines[2]
    assert "(node 5-6, 10 µm, 5 µm)" in lines[2]
    assert "refused" in lines[3] and "boundary node" in lines[3]
    assert all(line[:2].isdigit() and line[2] == ":" for line in lines)  # timestamped


def test_regenerate_hands_over_the_edited_graph_and_clears_the_tab(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.split_button.click()
    edited = c.state.graph
    c.regenerate_button.click()

    assert page.regenerated == [edited]
    assert page.stops == [None], "after a finished run: on to the end"
    assert HIGH_DEGREE_JUNCTIONS not in viewer.layers
    assert c.on_press not in viewer.layers[VESSELS].mouse_drag_callbacks
    assert c.state.graph is None and c.junction_list.count() == 0


def test_regenerate_hands_over_a_traced_vessel_with_its_cut_vessels(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.add_button.click()
    _click(page, (10.0, 10.0, 0.0))  # halfway along 1 -> 5
    _click(page, (20.0, -10.0, 0.0))  # node 3
    c.regenerate_button.click()

    (graph,) = page.regenerated
    assert graph.number_of_edges() == 8
    for _u, _v, data in graph.edges(data=True):
        assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
    assert len(_added(graph)) == 1
    assert ADDED_NODES not in viewer.layers and c.state.added_nodes == []


# --- Regenerate graph and Continue ----------------------------------------------


def test_the_hand_over_buttons_wait_for_something_to_hand_over(page):
    c = page.controls
    assert c.regenerate_graph_button.toolTip() and c.continue_button.toolTip()
    c.scan_button.click()
    assert not c.regenerate_graph_button.isEnabled(), "nothing edited yet"
    assert not c.regenerate_button.isEnabled()
    assert not c.continue_button.isEnabled(), "no run is paused"

    c.split_button.click()

    assert c.regenerate_graph_button.isEnabled()
    assert c.regenerate_button.isEnabled()
    assert "Regenerate graph" in c.commit_status.text()


def test_regenerate_graph_while_paused_keeps_the_run_paused(page):
    c = page.controls
    page.run.paused, page.run.complete = True, False
    c.scan_button.click()
    c.split_button.click()
    assert c.continue_button.isEnabled()
    assert not c.regenerate_button.isEnabled(), "Continue, not this, while paused"
    edited = c.state.graph

    c.regenerate_graph_button.click()

    assert page.regenerated == [edited]
    assert page.stops == ["post_process"]
    assert c.state.graph is None


def test_regenerate_graph_after_a_finished_run_re_solves_to_the_end(page):
    c = page.controls
    c.scan_button.click()
    c.split_button.click()

    c.regenerate_graph_button.click()

    assert page.stops == [None]


def test_continue_carries_a_paused_run_on_with_the_edits(page):
    c = page.controls
    page.run.paused, page.run.complete = True, False
    c.refresh()
    c.scan_button.click()
    c.split_button.click()
    edited = c.state.graph

    c.continue_button.click()

    assert page.regenerated == [edited]
    assert page.stops == [None]


def test_continue_without_a_scan_carries_on_the_network_as_it_paused(page):
    c = page.controls
    page.run.paused = True
    c.refresh()

    c.continue_button.click()

    (graph,) = page.regenerated
    assert graph is page.results._graph


def test_a_run_that_does_not_start_leaves_the_edits_in_the_tab(page):
    c = page.controls
    page.run.starts = False
    c.scan_button.click()
    c.split_button.click()
    edited = c.state.graph

    c.regenerate_graph_button.click()

    assert c.state.graph is edited, "a refused run loses nothing"
    assert c.junction_list.count() > 0 or c.state.scan is not None


def test_forgetting_the_tab_drops_its_edits(page):
    c = page.controls
    c.scan_button.click()
    c.split_button.click()

    c.forget()

    assert c.state.graph is None
    assert not c.regenerate_graph_button.isEnabled()
    assert c.status.text() == "Not scanned yet."


def test_mid_run_postprocessing_is_an_input_tab_checkbox_off_by_default(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    row = panel._haemolynx_rows()["mid_run_postprocessing"]
    assert row.value is False
    assert row.label == "Mid-run postprocessing"
    from haemolynx.gui.tabs import assign_to_stages
    from haemolynx.pipeline import default_schema

    assert assign_to_stages(default_schema())["mid_run_postprocessing"] == "1. Input"


def test_export_connectivity_writes_the_network_on_screen(page, tmp_path):
    import csv

    c = page.controls
    # Before a scan it exports the run's own graph.
    path = c.export_connectivity(tmp_path / "before.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    inlet = next(r for r in rows if r["Edge u"] == "0")
    assert (inlet["From node ID"], inlet["To node ID"], inlet["Notes"]) == ("", "1", "Inlet")
    assert next(r for r in rows if r["Edge v"] == "5")["Notes"] == "Dead end"

    # After an edit it exports the edited network, and says so in the log.
    c.scan_button.click()
    keys = edge_keys(c.state.graph)
    c.branch_ids.setText(str(next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})))
    c.delete_ids_button.click()
    path = c.export_connectivity(tmp_path / "after.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert "Exported the connectivity of all 5 vessels" in c.log_box.toPlainText()
    assert "may be out of date" in c.connectivity_status.text()


def test_export_connectivity_before_a_run_reports_instead_of_raising(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    controls = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: None,
        boundary_roles=lambda: {}, regenerate=lambda graph, stop_after=None: True,
        running=lambda: False,
    )
    assert controls.export_connectivity(tmp_path / "x.csv") is None
    assert "Nothing to export yet" in controls.connectivity_status.text()
    assert not (tmp_path / "x.csv").exists()
    assert controls.export_connectivity_button.toolTip()


def test_export_connectivity_sits_on_the_export_tab(make_napari_viewer):
    from qtpy.QtWidgets import QGroupBox, QPushButton, QTabWidget

    panel = settings_widget(napari_viewer=make_napari_viewer())
    tabs = panel.findChild(QTabWidget)
    box = panel.findChild(QGroupBox, "haemolynx_connectivity_box")
    export_page = next(
        tabs.widget(i) for i in range(tabs.count()) if tabs.tabText(i) == "10. Export"
    )
    assert export_page.isAncestorOf(box)
    for name in ("haemolynx_post_processing_export_connectivity", "haemolynx_connectivity_map"):
        assert box.isAncestorOf(panel.findChild(QPushButton, name))
    assert not panel._haemolynx_post_processing.page.isAncestorOf(box)


def test_export_only_inlet_to_outlet_drops_the_dead_end(page, tmp_path):
    import csv

    from haemolynx.gui.post_processing import INLET_TO_OUTLET_ONLY

    c = page.controls
    c.connectivity_choice.setCurrentText(INLET_TO_OUTLET_ONLY)
    path = c.export_connectivity(tmp_path / "through.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # 1 -> 5 leads nowhere; the five vessels from inlet 0 to outlet 4 remain.
    assert len(rows) == 5
    assert all(r["Edge v"] != "5" for r in rows)
    assert "5 of 6 vessels (inlet to outlet only)" in c.connectivity_status.text()


def test_open_connectivity_map_draws_the_last_export(page, tmp_path, monkeypatch):
    import webbrowser

    c = page.controls
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    c.export_connectivity(tmp_path / "net.csv")
    html_path = c.open_connectivity_map()
    assert html_path == tmp_path / "net_map.html" and html_path.is_file()
    assert opened == [html_path.resolve().as_uri()]
    assert "Opened the 2D connectivity map" in c.connectivity_status.text()
