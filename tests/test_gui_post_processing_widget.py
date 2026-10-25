"""The "7. Post processing" tab, built for real, with a viewer.

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
    REVIEW_LISTS,
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


#: The Review dropdown's entries the tests choose.
NO_LIST = "None (edit by clicking only)"
JUNCTIONS = "4+ junctions"
LOOPS = "Short loops"


@pytest.fixture
def page(make_napari_viewer):
    """The tab on the four-way network, 4+ junctions chosen under Review."""
    return _four_way_page(make_napari_viewer(), junction_correction=True)


def _four_way_page(viewer, *, junction_correction: bool, graph=None, settings=None, image=None):
    results = ResultLayers()
    if image is not None:
        _apply_layers(viewer, results.stage_finished(
            "skeletonise",
            SimpleNamespace(
                image=image, skeleton=np.zeros_like(image, dtype=bool),
                voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0),
            ),
        ))
    _apply_layers(viewer, results.stage_finished(
        "build_network", network(graph if graph is not None else _four_way_network())
    ))
    report = SimpleNamespace(value="")
    regenerated: list = []
    stops: list = []
    #: What the panel would say: a run paused here, whether a run started
    #: when asked.
    run = SimpleNamespace(paused=False, starts=True)

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
        settings=settings,
        paused=lambda: run.paused,
    )
    controls.refresh()
    controls.review_choice.setCurrentText(JUNCTIONS if junction_correction else NO_LIST)
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


def test_the_post_processing_tab_sits_between_haemodynamics_and_perturbations(make_napari_viewer):
    from qtpy.QtWidgets import QStackedWidget, QTabWidget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    tabs = panel.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert POST_PROCESSING_TAB == "7. Post processing"
    at = titles.index(POST_PROCESSING_TAB)
    assert titles[at - 1] == "6. Haemodynamics" and titles[at + 1] == "8. Perturbations"
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


def test_no_list_is_chosen_at_first_and_choosing_one_waits_for_a_scan(make_napari_viewer):
    """With no list chosen a scan picks no junction, marks none, tints none
    cyan and leaves the camera where it was. Choosing 4+ junctions shows its
    page; the next scan lists, marks and zooms; choosing None again takes it
    all away. The other edits work either way."""
    page = _four_way_page(make_napari_viewer(), junction_correction=False)
    c, viewer = page.controls, page.viewer
    assert c.review_choice.toolTip() and c.region.toolTip()
    fresh = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: None,
        boundary_roles=lambda: {}, regenerate=lambda graph: None, running=lambda: False,
    )
    assert fresh.review_choice.currentText() == NO_LIST, "nothing listed by default"
    choices = [fresh.review_choice.itemText(i) for i in range(fresh.review_choice.count())]
    assert choices == [entry.label for entry in REVIEW_LISTS]
    assert fresh.junction_box.isHidden() and fresh.loop_box.isHidden() and fresh.review_box.isHidden()
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
    assert c.status.text() == "6 vessels; no list chosen."

    # An edit rescans: still no junction picked or marked.
    keys = edge_keys(graph)
    c.branch_ids.setText(str(next(i for i, k in enumerate(keys) if set(k[:2]) == {2, 4})))
    c.delete_ids_button.click()
    assert c.state.graph.degree(1) == 4
    assert c.state.node is None and HIGH_DEGREE_JUNCTIONS not in viewer.layers

    c.review_choice.setCurrentText(JUNCTIONS)
    assert not c.junction_box.isHidden()
    assert c.state.node is None and c.junction_list.count() == 0, "listed only by a scan"
    assert c.status.text() == "Press Scan network to list 4+ junctions."
    c.scan_button.click()
    graph = c.state.graph
    assert c.state.node == 1 and c.table.rowCount() == 4
    assert np.allclose(viewer.layers[HIGH_DEGREE_JUNCTIONS].data, [[10, 0, 0]])
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(AT_JUNCTION)
    centre = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(centre, [10, 0, 0][-len(viewer.dims.displayed):], atol=1e-6)

    c.review_choice.setCurrentText(NO_LIST)
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


def test_add_vessel_logs_how_the_vessel_sits_in_the_segmentation(make_napari_viewer):
    """The log line says what the post_process stage will write on the
    vessel: traced along the L, all of it is in the segmentation; drawn
    straight across the gap, most of it is not -- and it is added all the same."""
    from haemolynx.graph import DRAWN_BACKGROUND_RUN_UM, DRAWN_MASK_FRACTION

    page = _l_page(make_napari_viewer(), image=_l_mask())
    c = page.controls
    _click(page, (0.0, 0.0, 10.0))   # node 1
    _click(page, (0.0, 20.0, 10.0))  # node 3, round the L
    log = c.log_box.toPlainText()
    assert "all of it in the segmentation" in log
    (_u, _v, along), = _added(c.state.graph)
    assert along[DRAWN_MASK_FRACTION] == pytest.approx(1.0)

    c.add_button.click()
    _click(page, (0.0, 0.0, 10.0))
    _click(page, (0.0, 10.0, 0.0))   # a waypoint in the background, left of the L
    _click(page, (0.0, 20.0, 10.0))
    off = [d for _u, _v, d in _added(c.state.graph) if d is not along]
    assert len(off) == 1
    assert off[0][DRAWN_MASK_FRACTION] < 0.9
    assert off[0][DRAWN_BACKGROUND_RUN_UM] > 5.0
    assert "% of it in the segmentation" in c.log_box.toPlainText().splitlines()[-1]


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
    settings = {"raw_image_path": str(path), "image_axis_order": "zyx"}
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
    assert "no raw image is set on the Input tab" in text

    # A raw image set afterwards is read once Raw data is chosen again.
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "raw.tif", _bright_l())
    settings["raw_image_path"] = str(tmp_path / "raw.tif")
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


def test_prune_removes_a_dead_end_tree_and_keeps_the_routes(make_napari_viewer):
    """1 -> 5 -> 6 hangs off the junction and reaches no outlet: pruned, with
    no delete first. The two routes 1 -> {2, 3} -> 4 stay."""
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

    graph = c.state.graph
    assert 5 not in graph and 6 not in graph
    assert graph.number_of_edges() == 5
    assert all(graph.has_edge(u, v) for u, v in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
    assert len(viewer.layers[VESSELS].data) == 5
    assert "pruned 0 disconnected piece(s), 2 dead-end vessel(s) and 0 on loops off one node, 2 vessel(s)" in report.value

    c.prune_button.click()
    assert "Nothing to prune" in c.status.text()


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
    c.review_choice.setCurrentText(JUNCTIONS)
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
    assert "Pruned 1 disconnected piece(s), 0 dead-end vessel(s) and 0 on loops off one node, 1 vessel(s): branchID" in lines[2]
    assert "(node 5-6, 10 µm, 5 µm)" in lines[2]
    assert "refused" in lines[3] and "boundary node" in lines[3]
    assert all(line[:2].isdigit() and line[2] == ":" for line in lines)  # timestamped


def test_regenerate_hands_over_the_edited_graph_and_clears_the_tab(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.split_button.click()
    edited = c.state.graph
    c.regenerate_graph_button.click()

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
    c.regenerate_graph_button.click()

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
    assert not c.continue_button.isEnabled(), "no run is paused"

    c.split_button.click()

    assert c.regenerate_graph_button.isEnabled()
    assert "Regenerate graph" in c.commit_status.text()


def test_the_page_hands_over_through_regenerate_graph_and_continue_alone(page):
    """"Regenerate from the edited network" did what Regenerate graph does
    after a finished run, and was clickable only when Regenerate graph was."""
    from qtpy.QtWidgets import QPushButton

    texts = {b.text() for b in page.controls.page.findChildren(QPushButton)}
    assert {"Regenerate graph", "Continue"} <= texts
    assert "Regenerate from the edited network" not in texts


def test_regenerate_graph_while_paused_keeps_the_run_paused(page):
    c = page.controls
    page.run.paused = True
    c.scan_button.click()
    c.split_button.click()
    assert c.continue_button.isEnabled()
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
    page.run.paused = True
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


# --- Manual loop review ------------------------------------------------------------

import csv  # noqa: E402

from haemolynx.gui.post_processing import KEPT, SIDE_DELETED  # noqa: E402


def _loop_page(viewer, tmp_path, *, loop_review: bool = True):
    """The tab on the four-way network, whose loop 1-{2,3}-4 runs through
    outlet 4, with Short loops chosen under Review and the run's output in *tmp_path*."""
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(_four_way_network())))
    report = SimpleNamespace(value="")
    controls = _post_processing_controls(
        viewer,
        report,
        results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (4,)},
        regenerate=lambda graph, stop_after=None: True,
        running=lambda: False,
        settings=lambda: {"vtk_output_prefix": str(tmp_path / "run")},
        paused=lambda: False,
    )
    controls.refresh()
    controls.review_choice.setCurrentText(LOOPS if loop_review else NO_LIST)
    return SimpleNamespace(
        controls=controls, viewer=viewer, report=report, csv=tmp_path / "run_loop_review.csv",
    )


def _side_through(controls, node) -> int:
    return next(
        i for i, side in enumerate(controls.state.loop.sides) if any(node in e[:2] for e in side)
    )


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_short_loops_are_listed_only_once_chosen_and_scanned(make_napari_viewer, tmp_path):
    page = _loop_page(make_napari_viewer(), tmp_path, loop_review=False)
    c, viewer = page.controls, page.viewer
    assert c.loop_box.isHidden()
    for widget in (c.loop_list, c.loop_table, c.loop_delete_button, c.loop_keep_button):
        assert c.loop_box.isAncestorOf(widget) and widget.toolTip()

    c.scan_button.click()
    graph = c.state.graph
    assert c.loop_list.count() == 0 and c.state.loop is None
    assert _colour_of(viewer, graph, (2, 4)) == _rgba(CONNECTED)

    c.review_choice.setCurrentText(LOOPS)
    assert not c.loop_box.isHidden()
    assert c.loop_list.count() == 0, "listed only by a scan"
    c.scan_button.click()
    graph = c.state.graph
    assert c.loop_list.count() == 1
    assert c.loop_list.item(0).text() == "Loop 1: 56.6 µm round, 2 side(s)"
    assert c.loop_table.rowCount() == 2, "cut at outlet 4 as well as at junction 1"
    for pair in [(1, 2), (2, 4), (3, 4), (1, 3)]:
        assert _colour_of(viewer, graph, pair) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, (0, 1)) == _rgba(CONNECTED)
    # Zoomed to the loop's middle, (20, 0, 0).
    centre = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(centre, [20, 0, 0][-len(viewer.dims.displayed):], atol=1e-6)

    c.review_choice.setCurrentText(NO_LIST)
    assert c.loop_box.isHidden() and c.state.loop is None and c.loop_list.count() == 0
    assert _colour_of(viewer, graph, (2, 4)) == _rgba(CONNECTED)


def test_picking_a_loop_in_3d_looks_at_it_in_its_own_plane(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    page = _loop_page(viewer, tmp_path)

    page.controls.scan_button.click()

    # The loop lies in x = 0, so the camera looks along x at its middle.
    view = np.asarray(viewer.camera.view_direction, dtype=float)
    assert abs(view[2]) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(np.asarray(viewer.camera.center), [20, 0, 0], atol=1e-3)


def test_keep_records_the_loop_takes_it_off_the_list_and_writes_the_csv(make_napari_viewer, tmp_path):
    page = _loop_page(make_napari_viewer(), tmp_path)
    c = page.controls
    c.scan_button.click()
    before = sorted(edge_keys(c.state.graph))

    c.loop_keep_button.click()

    assert c.loop_list.count() == 0 and c.state.loop is None
    assert sorted(edge_keys(c.state.graph)) == before, "keeping changes nothing"
    assert [review.decision for review in c.state.loop_reviews] == [KEPT]
    assert [row["decision"] for row in _csv_rows(page.csv)] == [KEPT]
    assert "kept" in c.log_box.toPlainText()
    assert "0 short loop(s) to review" in c.loop_summary.text()
    assert "1 kept" in c.loop_summary.text() and page.csv.name in c.loop_summary.text()
    assert not c.regenerate_graph_button.isEnabled(), "nothing to regenerate"
    # A kept loop is not asked about again.
    c.scan_button.click()
    assert c.loop_list.count() == 0


def test_delete_this_side_opens_the_loop_and_records_which_side(make_napari_viewer, tmp_path):
    page = _loop_page(make_napari_viewer(), tmp_path)
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    c.loop_delete_button.click()
    assert "Select the side" in c.loop_summary.text()
    assert c.state.graph.has_edge(1, 2)

    side = _side_through(c, 2)
    c.loop_table.selectRow(side)
    graph = c.state.graph
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(SELECTED)
    assert _colour_of(viewer, graph, (2, 4)) == _rgba(SELECTED)
    assert _colour_of(viewer, graph, (1, 3)) == _rgba(AT_JUNCTION)
    c.loop_delete_button.click()

    graph = c.state.graph
    assert 2 not in graph and graph.degree(1) == 3 and graph.degree(4) == 1
    assert len(viewer.layers[VESSELS].data) == 4
    assert c.loop_list.count() == 0
    assert [(r.decision, r.side) for r in c.state.loop_reviews] == [(SIDE_DELETED, side + 1)]
    (row,) = _csv_rows(page.csv)
    assert row["decision"] == SIDE_DELETED and row["side_deleted"] == str(side + 1)
    assert row["sides"] == "2"
    assert c.regenerate_graph_button.isEnabled()
    assert "opened: deleted side" in c.log_box.toPlainText()
    assert "deleted side" in page.report.value

    c.forget()
    assert c.state.loop_reviews == [] and c.loop_list.count() == 0


def test_a_loop_decision_without_an_output_folder_still_counts(make_napari_viewer, tmp_path):
    page = _loop_page(make_napari_viewer(), tmp_path / "not made yet")
    c = page.controls
    c.scan_button.click()

    c.loop_keep_button.click()

    assert [review.decision for review in c.state.loop_reviews] == [KEPT]
    assert "Not written to a CSV" in c.loop_summary.text()
    assert not page.csv.exists()


# --- the other review lists ------------------------------------------------------

from haemolynx.gui.post_processing import (  # noqa: E402
    ACTION_DELETE_ITEM,
    ACTION_DELETE_SELECTED,
    ACTION_JOIN_AHEAD,
    ACTION_KEEP,
    ACTION_MARK_BLIND,
    ACTION_MARK_OPEN,
    ACTION_MERGE,
    ACTION_SET_DIAMETER,
    ACTION_TRACE_ON,
    REVIEW_MARKERS,
)

UNSOLVED = "Unsolved pieces"
UNMEASURED = "Unmeasured diameters"
DEAD_ENDS = "Dead ends"
CLOSE = "Junctions close together"


def _review_page(viewer, tmp_path=None, **kwargs):
    settings = (lambda: {"vtk_output_prefix": str(tmp_path / "run")}) if tmp_path is not None else None
    return _four_way_page(viewer, junction_correction=False, settings=settings, **kwargs)


def _shown_buttons(c) -> set:
    return {action for action, button in c.review_buttons.items() if not button.isHidden()}


def test_a_review_list_shows_its_own_page_and_buttons_and_lists_only_once_scanned(make_napari_viewer, tmp_path):
    page = _review_page(make_napari_viewer(), tmp_path)
    c, viewer = page.controls, page.viewer
    for widget in (c.review_items, c.review_table, c.review_diameter, *c.review_buttons.values()):
        assert c.review_box.isAncestorOf(widget) and widget.toolTip()
    c.scan_button.click()

    c.review_choice.setCurrentText(UNSOLVED)
    assert not c.review_box.isHidden() and c.junction_box.isHidden() and c.loop_box.isHidden()
    assert _shown_buttons(c) == {ACTION_DELETE_ITEM, ACTION_KEEP}
    assert c.review_items.count() == 0 and c.status.text() == "Press Scan network to list unsolved pieces."
    c.scan_button.click()

    # 1 -> 5 is a dead end: no inlet-to-outlet path runs along it.
    assert c.review_items.count() == 1
    assert c.review_items.item(0).text() == "1. 1 vessel(s), 20 µm, off node 1"
    assert c.review_table.rowCount() == 1
    graph = c.state.graph
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(CONNECTED)
    assert np.allclose(viewer.layers[REVIEW_MARKERS].data, [[10, 10, 0]])
    centre = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(centre, [10, 10, 0][-len(viewer.dims.displayed):], atol=1e-6)
    assert "1 item(s) in unsolved pieces to review" in c.review_summary.text()

    c.review_choice.setCurrentText(UNMEASURED)
    assert c.review_items.count() == 0 and REVIEW_MARKERS not in viewer.layers
    assert _shown_buttons(c) == {ACTION_SET_DIAMETER, ACTION_KEEP} and not c.review_diameter.isHidden()


def test_deleting_an_unsolved_piece_records_the_decision_and_empties_the_list(make_napari_viewer, tmp_path):
    page = _review_page(make_napari_viewer(), tmp_path)
    c, viewer = page.controls, page.viewer
    c.review_choice.setCurrentText(UNSOLVED)
    c.scan_button.click()

    c.review_buttons[ACTION_DELETE_ITEM].click()

    graph = c.state.graph
    assert 5 not in graph and graph.number_of_edges() == 5
    assert len(viewer.layers[VESSELS].data) == 5
    assert c.review_items.count() == 0 and REVIEW_MARKERS not in viewer.layers
    assert [d.decision for d in c.state.review_decisions] == ["deleted"]
    (row,) = _csv_rows(tmp_path / "run_unsolved_pieces_review.csv")
    assert row["decision"] == "deleted" and row["kind"] == "unsolved_pieces"
    assert row["attached_at"] == "1" and row["length_um"] == "20"
    assert "Deleted branchID" in c.log_box.toPlainText()
    assert c.regenerate_graph_button.isEnabled()


def test_a_kept_item_is_not_listed_again(make_napari_viewer):
    page = _review_page(make_napari_viewer())
    c = page.controls
    c.review_choice.setCurrentText(UNSOLVED)
    c.scan_button.click()
    before = sorted(edge_keys(c.state.graph))

    c.review_buttons[ACTION_KEEP].click()

    assert c.review_items.count() == 0
    assert sorted(edge_keys(c.state.graph)) == before, "keeping changes nothing"
    assert [d.decision for d in c.state.review_decisions] == [KEPT]
    assert "Not written to a CSV" in c.review_summary.text()
    c.scan_button.click()
    assert c.review_items.count() == 0
    c.forget()
    assert c.state.review_decisions == []


def test_set_diameter_gives_the_vessel_a_width_regenerate_keeps(make_napari_viewer):
    G = _four_way_network()
    G.edges[1, 5, 0]["diameter_source"] = "table"
    page = _review_page(make_napari_viewer(), graph=G)
    c = page.controls
    c.review_choice.setCurrentText(UNMEASURED)
    c.scan_button.click()
    assert c.review_items.count() == 1

    c.review_diameter.setValue(9.0)
    c.review_buttons[ACTION_SET_DIAMETER].click()

    data = c.state.graph.edges[1, 5, 0]
    assert data["diameter_um"] == 9.0 and data["diameter_source"] == "override"
    assert "post_processing_edited" not in data
    assert c.review_items.count() == 0, "an override is no longer unmeasured"
    assert c.regenerate_graph_button.isEnabled()
    assert "to 9 µm" in c.log_box.toPlainText()


def test_merge_makes_two_close_junctions_one(make_napari_viewer):
    G = _four_way_network()
    G.remove_edge(1, 2)
    G.add_node(6, pos=np.asarray((11.0, 0.0, 0.0)))  # a micron past junction 1
    for u, v in [(1, 6), (6, 2)]:
        voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
        G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01", diameter_um=5.0)
    G.add_node(7, pos=np.asarray((11.0, -20.0, 0.0)))
    G.add_edge(6, 7, voxels=[(11.0, 0.0, 0.0), (11.0, -20.0, 0.0)], length=20.0, diameter_um=5.0)
    page = _review_page(make_napari_viewer(), graph=G)
    c = page.controls
    c.review_choice.setCurrentText(CLOSE)
    c.scan_button.click()
    assert c.review_items.item(0).text().startswith("1. Nodes 1, 6: 1 µm apart")
    assert _shown_buttons(c) == {ACTION_MERGE, ACTION_DELETE_SELECTED, ACTION_KEEP}

    c.review_buttons[ACTION_MERGE].click()

    graph = c.state.graph
    assert 6 not in graph and graph.degree(1) == 5
    assert c.review_items.count() == 0
    assert "Merged junctions 1 and 6 into node 1" in c.log_box.toPlainText()


def test_a_list_without_what_it_needs_says_so(make_napari_viewer):
    page = _review_page(make_napari_viewer())
    c = page.controls
    c.review_choice.setCurrentText(DEAD_ENDS)
    c.scan_button.click()
    assert c.review_items.count() == 0
    assert "Dead ends needs the segmented image in the viewer" in c.review_summary.text()
    c.review_choice.setCurrentText("Flow outliers")
    c.scan_button.click()
    assert "needs a solved network" in c.status.text()


def test_the_region_limits_a_list_to_round_the_view_centre(make_napari_viewer):
    page = _review_page(make_napari_viewer())
    c, viewer = page.controls, page.viewer
    c.review_choice.setCurrentText(UNSOLVED)
    viewer.camera.center = (0.0, 500.0, 500.0)
    c.region.setValue(30.0)
    c.scan_button.click()
    assert c.review_items.count() == 0
    viewer.camera.center = (0.0, 10.0, 10.0)
    c.scan_button.click()
    assert c.review_items.count() == 1
    c.region.setValue(0.0)
    viewer.camera.center = (0.0, 500.0, 500.0)
    c.scan_button.click()
    assert c.review_items.count() == 1


def _t_network_and_mask():
    """A trunk 0 - 1 - 2 along x in a segmented tube, and 1 -> 3 off it into
    background: a dead end mostly off the mask, well inside the image."""
    mask = np.zeros((25, 60, 60), dtype=np.uint8)
    idx = np.indices(mask.shape)
    mask[((idx[0] - 12) ** 2 + (idx[1] - 10) ** 2) <= 9] = 1
    G = nx.MultiGraph()
    for node, pos in {0: (12, 10, 0), 1: (12, 10, 30), 2: (12, 10, 59), 3: (12, 35, 30)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
        G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01", diameter_um=6.0)
    return G, mask


def _t_page(viewer, tmp_path):
    G, mask = _t_network_and_mask()
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished(
        "skeletonise",
        SimpleNamespace(image=mask, skeleton=np.zeros_like(mask, dtype=bool),
                        voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0)),
    ))
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (0,), "outlet": (2,)},
        regenerate=lambda graph, stop_after=None: True, running=lambda: False,
        settings=lambda: {"vtk_output_prefix": str(tmp_path / "run")},
    )
    c.refresh()
    c.review_choice.setCurrentText(DEAD_ENDS)
    return c


def test_a_mask_list_is_made_in_the_background_and_its_dead_end_traced_on(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    c = _t_page(viewer, tmp_path)

    c.scan_button.click()
    qtbot.waitUntil(lambda: not c.state.scanning, timeout=60000)

    assert c.scan_button.isEnabled()
    assert c.review_items.count() == 1
    assert c.review_items.item(0).text() == "1. Node 3: off mask, 25 µm"
    assert _shown_buttons(c) == {
        ACTION_DELETE_ITEM, ACTION_TRACE_ON, ACTION_JOIN_AHEAD, ACTION_MARK_BLIND, ACTION_MARK_OPEN, ACTION_KEEP,
    }
    # No vessel was found ahead of this tip, so there is nothing to join it to.
    assert not c.review_buttons[ACTION_JOIN_AHEAD].isEnabled()
    assert c.review_buttons[ACTION_MARK_BLIND].isEnabled() and c.review_buttons[ACTION_MARK_OPEN].isEnabled()
    assert c.state.support is not None, "the segmented image is read once and kept"

    c.review_buttons[ACTION_TRACE_ON].click()
    assert c.state.mode == "add" and c.state.trace.start.node == 3
    assert "Starting at node 3" in c.edit_status.text()
    # Recorded when the trace is finished: until then nothing was traced on.
    assert c.state.review_decisions == []
    c.add_button.click()  # cancel the trace

    c.review_buttons[ACTION_DELETE_ITEM].click()
    assert 3 not in c.state.graph
    assert c.review_items.count() == 0
    assert "Scan network again" in c.review_summary.text(), "a mask list may be out of date after an edit"
    rows = _csv_rows(tmp_path / "run_dead_ends_review.csv")
    assert [r["decision"] for r in rows] == ["deleted"], "a cancelled trace leaves no decision"
    assert rows[0]["kinds"] == "off_mask"


@pytest.mark.parametrize(
    "action, kind, words",
    [(ACTION_MARK_BLIND, "blind", "a real blind end"), (ACTION_MARK_OPEN, "leaves_image", "leaving the image")],
)
def test_a_dead_end_marked_by_hand_keeps_its_kind_and_leaves_the_list(
    make_napari_viewer, qtbot, tmp_path, action, kind, words
):
    """PLAN_dead_ends.md Phase 9: a person's verdict goes on the node, where
    reclassifying keeps it, and into the review CSV a rule is fitted from."""
    from haemolynx.graph.dead_ends import DEAD_END_KIND, classify_and_mark

    c = _t_page(make_napari_viewer(), tmp_path)
    c.scan_button.click()
    qtbot.waitUntil(lambda: not c.state.scanning, timeout=60000)
    assert c.review_buttons[action].isEnabled()

    c.review_buttons[action].click()
    assert c.state.graph.nodes[3][DEAD_END_KIND] == kind
    assert c.review_items.count() == 0
    rows = _csv_rows(tmp_path / "run_dead_ends_review.csv")
    assert [r["decision"] for r in rows] == [words]
    classify_and_mark(c.state.graph)
    assert c.state.graph.nodes[3][DEAD_END_KIND] == kind, "a person's mark outlives reclassifying"


def _broken_page(viewer, tmp_path, make=None, choice=DEAD_ENDS):
    """dead_end_fixtures' broken vessel (or *make*'s case): a branch ending
    9 um short of another vessel's side across a gap in the segmentation."""
    from dead_end_fixtures import broken_towards_side

    case = (make or broken_towards_side)()
    G = nx.convert_node_labels_to_integers(case.graph, ordering="sorted")
    tip = next(n for n, d in G.nodes(data=True) if tuple(map(float, d["pos"])) == case.tip)
    for _u, _v, d in G.edges(data=True):
        d.setdefault("branch_order", "B01")
        d.setdefault("diameter_um", 5.0)
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished(
        "skeletonise",
        SimpleNamespace(image=case.mask.astype(np.uint8), skeleton=np.zeros_like(case.mask, dtype=bool),
                        voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0)),
    ))
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (), "outlet": ()},
        regenerate=lambda graph, stop_after=None: True, running=lambda: False,
        settings=lambda: {"vtk_output_prefix": str(tmp_path / "run")},
    )
    c.refresh()
    c.review_choice.setCurrentText(choice)
    return c, tip


def test_a_broken_dead_end_is_joined_to_the_side_of_the_vessel_ahead(make_napari_viewer, qtbot, tmp_path):
    c, tip = _broken_page(make_napari_viewer(), tmp_path)
    c.scan_button.click()
    qtbot.waitUntil(lambda: not c.state.scanning, timeout=60000)
    row = next(i for i, item in enumerate(c.state.review_items) if item.nodes[0] == tip)
    c.review_items.setCurrentRow(row)
    assert c.review_buttons[ACTION_JOIN_AHEAD].isEnabled()
    assert nx.number_connected_components(c.state.graph) == 2

    c.review_buttons[ACTION_JOIN_AHEAD].click()
    assert c.state.graph.degree(tip) == 2
    assert nx.number_connected_components(c.state.graph) == 1
    rows = _csv_rows(tmp_path / "run_dead_ends_review.csv")
    assert [r["decision"] for r in rows] == ["joined to the vessel ahead"]


def test_a_dead_end_at_a_thick_wall_is_joined_to_the_thick_centreline(make_napari_viewer, qtbot, tmp_path):
    """PLAN_bridging.md 0.2: "Dead ends at thick vessels" lists the dead end
    the wall join left, and "Join to the vessel ahead" draws it on to the
    thick vessel's own centreline beside it."""
    from dead_end_fixtures import THICK_AXIS, at_a_thick_wall

    c, tip = _broken_page(make_napari_viewer(), tmp_path, make=at_a_thick_wall, choice="Dead ends at thick vessels")
    c.scan_button.click()
    qtbot.waitUntil(lambda: not c.state.scanning, timeout=60000)
    (item,) = c.state.review_items
    assert item.nodes[0] == tip and item.measures["kind"] == "thick_wall"
    c.review_items.setCurrentRow(0)
    assert _shown_buttons(c) == {ACTION_TRACE_ON, ACTION_JOIN_AHEAD, ACTION_DELETE_ITEM, ACTION_MARK_BLIND, ACTION_KEEP}
    assert c.review_buttons[ACTION_JOIN_AHEAD].isEnabled()
    assert nx.number_connected_components(c.state.graph) == 2

    c.review_buttons[ACTION_JOIN_AHEAD].click()

    assert c.state.graph.degree(tip) == 2
    assert nx.number_connected_components(c.state.graph) == 1
    landed = [n for n in c.state.graph if c.state.graph.degree(n) == 3
              and abs(float(c.state.graph.nodes[n]["pos"][1]) - THICK_AXIS[0][1]) < 0.6]
    assert len(landed) == 1, "the join lands on the thick vessel's centreline"
    rows = _csv_rows(tmp_path / "run_thick_wall_ends_review.csv")
    assert [r["decision"] for r in rows] == ["joined to the vessel ahead"]


def test_the_scan_after_a_regenerate_says_what_it_changed(page):
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    keys = edge_keys(c.state.graph)
    c.branch_ids.setText(str(next(i for i, k in enumerate(keys) if set(k[:2]) == {1, 5})))
    c.delete_ids_button.click()
    c.regenerate_graph_button.click()
    (edited,) = page.regenerated

    # The regenerated run puts a new network in the viewer.
    _apply_layers(viewer, page.results.stage_finished("build_network", network(edited.copy())))
    c.scan_button.click()

    assert "Since Regenerate: vessels 6 → 5; length 87 → 67 µm; unsolved 1 → 0." in c.status.text()
    assert "Since Regenerate" in c.log_box.toPlainText()
    c.scan_button.click()
    assert "Since Regenerate" not in c.status.text(), "said once"


# --- rings: what they ring is coloured, and a click on one chooses it ---------------

from haemolynx.gui.post_processing import LISTED  # noqa: E402


def _ring_colours(viewer, name) -> list:
    return [tuple(np.round(row, 3)) for row in np.asarray(viewer.layers[name].border_color)]


def _add_vessels(G, positions, pairs) -> nx.MultiGraph:
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v in pairs:
        voxels = [tuple(map(float, G.nodes[u]["pos"])), tuple(map(float, G.nodes[v]["pos"]))]
        G.add_edge(u, v, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01", diameter_um=5.0)
    return G


def test_ringed_junctions_vessels_are_magenta_and_a_click_on_a_ring_chooses_it(make_napari_viewer):
    # Outlet 4 becomes a second 4+ junction; 6 -> 8 is at neither.
    G = _add_vessels(
        _four_way_network(), {6: (40, 0, 0), 7: (30, 20, 0), 8: (50, 0, 0)}, [(4, 6), (4, 7), (6, 8)]
    )
    page = _four_way_page(make_napari_viewer(), junction_correction=True, graph=G)
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    graph = c.state.graph
    assert c.state.node == 1 and list(c.state.scan.junctions) == [1, 4]
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, (4, 7)) == _rgba(LISTED)
    assert _colour_of(viewer, graph, (6, 8)) == _rgba(CONNECTED)
    assert _ring_colours(viewer, HIGH_DEGREE_JUNCTIONS) == [_rgba(AT_JUNCTION), _rgba(LISTED)]

    viewer.camera.zoom = 10.0
    centre = np.asarray(viewer.camera.center, dtype=float).copy()
    _click(page, (31.0, 1.0, 0.0))  # inside junction 4's ring, nothing armed

    assert c.state.node == 4 and c.junction_list.currentRow() == 1
    assert {c.table.item(r, 4).text() for r in range(c.table.rowCount())} == {"2", "3", "6", "7"}
    assert _colour_of(viewer, graph, (4, 7)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, (1, 5)) == _rgba(LISTED)
    assert _ring_colours(viewer, HIGH_DEGREE_JUNCTIONS) == [_rgba(LISTED), _rgba(AT_JUNCTION)]
    assert np.allclose(viewer.camera.center, centre), "the ring is on screen already: no zoom"
    assert sorted(edge_keys(graph)) == sorted(edge_keys(G)), "choosing edits nothing"

    _click(page, (20.0, 20.0, 0.0))  # in no ring
    assert c.state.node == 4
    # Picking from the list still zooms, and recolours the rings the same way.
    c.junction_list.setCurrentRow(0)
    assert _ring_colours(viewer, HIGH_DEGREE_JUNCTIONS) == [_rgba(AT_JUNCTION), _rgba(LISTED)]
    shown = np.asarray(viewer.camera.center)[-len(viewer.dims.displayed):]
    assert np.allclose(shown, [10, 0, 0][-len(viewer.dims.displayed):], atol=1e-6)


def test_a_drag_over_a_ring_turns_the_view_and_chooses_nothing(make_napari_viewer):
    G = _add_vessels(_four_way_network(), {6: (40, 0, 0), 7: (30, 20, 0)}, [(4, 6), (4, 7)])
    page = _four_way_page(make_napari_viewer(), junction_correction=True, graph=G)
    c, viewer = page.controls, page.viewer
    c.scan_button.click()
    layer = viewer.layers[VESSELS]
    assert c.state.mode == "idle" and c.on_press in layer.mouse_drag_callbacks

    def press_and_release(moved_to):
        event = SimpleNamespace(
            position=(30.0, 0.0, 0.0), dims_displayed=(0, 1, 2), view_direction=None,
            pos=(100.0, 100.0), type="mouse_press", button=1,
        )
        gen = c.on_press(layer, event)
        next(gen)
        event.type, event.pos = "mouse_move", moved_to
        next(gen)
        event.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(gen)

    press_and_release((160.0, 100.0))
    assert c.state.node == 1
    press_and_release((101.0, 100.0))
    assert c.state.node == 4


def test_ringed_items_vessels_are_magenta_and_a_click_on_a_ring_chooses_the_item(make_napari_viewer):
    # Two dead ends no inlet-to-outlet path runs along: 1 -> 5 and 3 -> 6.
    G = _add_vessels(_four_way_network(), {6: (20, -30, 0)}, [(3, 6)])
    page = _review_page(make_napari_viewer(), graph=G)
    c, viewer = page.controls, page.viewer
    c.review_choice.setCurrentText(UNSOLVED)
    c.scan_button.click()
    graph = c.state.graph
    assert c.review_items.count() == 2
    items = list(c.state.review_items)
    chosen = c.state.review_item
    assert chosen is items[0]
    other = items[1]

    def far_end(item):
        return set(item.edges[0][:2])

    assert _colour_of(viewer, graph, far_end(chosen)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, far_end(other)) == _rgba(LISTED)
    assert _colour_of(viewer, graph, (1, 2)) == _rgba(CONNECTED)
    assert _ring_colours(viewer, REVIEW_MARKERS) == [_rgba(AT_JUNCTION), _rgba(LISTED)]

    viewer.camera.zoom = 10.0
    centre = np.asarray(viewer.camera.center, dtype=float).copy()
    _click(page, tuple(np.asarray(other.centre_um) + (0.0, 1.0, 1.0)))

    assert c.state.review_item is other and c.review_items.currentRow() == 1
    assert c.review_table.rowCount() == 1
    assert _colour_of(viewer, graph, far_end(other)) == _rgba(AT_JUNCTION)
    assert _colour_of(viewer, graph, far_end(chosen)) == _rgba(LISTED)
    assert _ring_colours(viewer, REVIEW_MARKERS) == [_rgba(LISTED), _rgba(AT_JUNCTION)]
    assert np.allclose(viewer.camera.center, centre), "the ring is on screen already: no zoom"

    # Kept, an item is no longer ringed, and its vessel no longer magenta.
    c.review_buttons[ACTION_KEEP].click()
    assert c.review_items.count() == 1 and c.state.review_item is chosen
    assert _colour_of(viewer, graph, far_end(other)) == _rgba(CONNECTED)
    assert _colour_of(viewer, graph, far_end(chosen)) == _rgba(AT_JUNCTION)
    assert _ring_colours(viewer, REVIEW_MARKERS) == [_rgba(AT_JUNCTION)]


def test_with_an_edit_armed_a_click_on_a_ring_edits_instead(make_napari_viewer):
    G = _add_vessels(_four_way_network(), {6: (20, -30, 0)}, [(3, 6)])
    page = _review_page(make_napari_viewer(), graph=G)
    c = page.controls
    c.review_choice.setCurrentText(UNSOLVED)
    c.scan_button.click()
    first, other = c.state.review_items
    c.click_delete_button.click()
    _click(page, tuple(other.centre_um))  # the middle of the other dead end
    assert not c.state.graph.has_edge(*other.edges[0][:2]), "deleted, not chosen"


def _island_page(viewer, tmp_path):
    """dead_end_fixtures' island case: the trunk, a short spur off it, and a
    two-edge piece of its own."""
    from dead_end_fixtures import island

    case = island()
    G = nx.convert_node_labels_to_integers(case.graph, ordering="sorted")
    island_nodes = {n for n, d in G.nodes(data=True) if abs(float(d["pos"][1]) - 50.0) < 1e-6}
    for _u, _v, d in G.edges(data=True):
        d.setdefault("branch_order", "B01")
        d.setdefault("diameter_um", 5.0)
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished(
        "skeletonise",
        SimpleNamespace(image=case.mask.astype(np.uint8), skeleton=np.zeros_like(case.mask, dtype=bool),
                        voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0)),
    ))
    _apply_layers(viewer, results.stage_finished("build_network", network(G)))
    c = _post_processing_controls(
        viewer, SimpleNamespace(value=""), results=lambda: results,
        boundary_roles=lambda: {"inlet": (), "outlet": ()},
        regenerate=lambda graph, stop_after=None: True, running=lambda: False,
        settings=lambda: {"vtk_output_prefix": str(tmp_path / "run")},
    )
    c.refresh()
    c.review_choice.setCurrentText(DEAD_ENDS)
    return c, island_nodes


def test_dead_ends_filtered_to_pieces_with_no_junction_can_all_be_removed(make_napari_viewer, qtbot, tmp_path):
    c, island_nodes = _island_page(make_napari_viewer(), tmp_path)
    assert not c.dead_end_show.isHidden() and c.dead_end_show.toolTip() and c.remove_pieces_button.toolTip()
    assert c.review_box.isAncestorOf(c.dead_end_show) and c.review_box.isAncestorOf(c.remove_pieces_button)

    c.scan_button.click()
    qtbot.waitUntil(lambda: not c.state.scanning, timeout=60000)
    assert c.review_items.count() == 2, "the spur and the island"
    assert c.remove_pieces_button.isHidden(), "only offered while the list shows pieces with no junction"

    c.dead_end_show.setCurrentText("Pieces with no junction")
    assert c.review_items.count() == 1
    assert not c.remove_pieces_button.isHidden() and c.remove_pieces_button.isEnabled()
    assert c.remove_pieces_button.text() == "Remove all 1 listed piece with no junction"

    c.remove_pieces_button.click()
    assert not island_nodes & set(c.state.graph), "the whole piece is gone"
    assert c.state.graph.number_of_edges() == 3, "the trunk's two halves and the spur stay"
    assert c.review_items.count() == 0
    assert not c.remove_pieces_button.isEnabled()
    rows = _csv_rows(tmp_path / "run_dead_ends_review.csv")
    assert [(r["decision"], r["kind"]) for r in rows] == [("deleted", "dead_ends")]
    assert "Removed 1 piece(s) with no junction" in c.log_box.toPlainText()

    c.dead_end_show.setCurrentText("All kinds")
    assert c.review_items.count() == 1, "the spur is still listed, the removed piece is not"
    assert c.remove_pieces_button.isHidden()


# --- Terminal nodes: inlets and outlets added and removed by hand ------------------

from haemolynx.graph.post_processing import BOUNDARY_ROLES  # noqa: E402
from haemolynx.gui.post_processing import (  # noqa: E402
    ACTION_MAKE_INLET,
    ACTION_MAKE_OUTLET,
    ACTION_REMOVE_TERMINAL,
)

TERMINALS = "Terminal nodes (add or remove inlets and outlets)"


def test_a_dead_end_made_an_outlet_is_kept_on_the_graph_for_regenerate(make_napari_viewer, tmp_path):
    page = _review_page(make_napari_viewer(), tmp_path)
    c = page.controls
    c.review_choice.setCurrentText(TERMINALS)
    c.scan_button.click()
    assert [c.review_items.item(i).text().split(":")[0] for i in range(c.review_items.count())] == [
        "1. Node 0 (inlet)", "2. Node 4 (outlet)", "3. Node 5 (dead end)",
    ]
    assert _shown_buttons(c) == {ACTION_MAKE_INLET, ACTION_MAKE_OUTLET, ACTION_REMOVE_TERMINAL, ACTION_KEEP}
    # The only inlet: no role to make it again, and taking it away is refused.
    assert not c.review_buttons[ACTION_MAKE_INLET].isEnabled()
    c.review_buttons[ACTION_REMOVE_TERMINAL].click()
    assert "only inlet" in c.review_summary.text()
    assert BOUNDARY_ROLES not in c.state.graph.graph and not c.regenerate_graph_button.isEnabled()

    c.review_items.setCurrentRow(2)
    c.review_buttons[ACTION_MAKE_OUTLET].click()

    assert c.state.outlets == (4, 5) and c.state.inlets == (0,)
    assert 5 in c.state.protected, "an outlet cannot be deleted by a later edit"
    assert c.state.graph.graph[BOUNDARY_ROLES] == {"inlet": [0], "outlet": [4, 5]}
    assert c.regenerate_graph_button.isEnabled()
    (row,) = _csv_rows(tmp_path / "run_terminal_nodes_review.csv")
    assert row["decision"] == "made an outlet" and row["nodes"] == "5" and row["role"] == ""
    assert "Node 5 made an outlet: 1 inlet(s), 2 outlet(s)" in c.log_box.toPlainText()

    # A scan reads the roles back off the graph, not the run's own lists.
    c.scan_button.click()
    assert c.review_items.item(2).text().startswith("3. Node 5 (outlet)")
    c.review_items.setCurrentRow(1)
    c.review_buttons[ACTION_REMOVE_TERMINAL].click()
    assert c.state.outlets == (5,)

    c.regenerate_graph_button.click()

    assert page.regenerated[-1].graph[BOUNDARY_ROLES] == {"inlet": [0], "outlet": [5]}


def _raw_panel_page(viewer, *, raw: bool):
    """The four-way network over a segmentation round its loop, Short loops
    chosen and scanned; with *raw*, the run's raw image in the viewer (the
    Diameters stage's copy), bright where the segmentation is."""
    image = np.zeros((36, 26, 6), dtype=np.uint8)
    image[6:34, 0:14, 0:3] = 1
    if raw:
        bright = np.random.default_rng(0).uniform(0.0, 20.0, size=image.shape).astype(np.float32)
        bright[image.astype(bool)] = 400.0
        viewer.add_image(bright, name=FWHM_RAW)
    page = _four_way_page(viewer, junction_correction=False, image=image,
                          settings=lambda: {"image_axis_order": "zyx"})
    c = page.controls
    c.review_choice.setCurrentText(LOOPS)
    c.scan_button.click()
    return c


def _grey_at_middle(label) -> int:
    from qtpy.QtGui import QColor

    image = label.pixmap().toImage()
    return QColor(image.pixel(image.width() // 2, image.height() // 2)).value()


def test_the_raw_panel_draws_the_chosen_loop_in_the_segmentation_and_the_raw_image(make_napari_viewer):
    c = _raw_panel_page(make_napari_viewer(), raw=True)
    assert c.state.loop is not None
    assert not c.raw_box.isHidden() and c.raw_check.toolTip()
    assert not c.raw_check.isChecked() and c.raw_grid.isHidden(), "off until ticked"

    c.raw_check.setChecked(True)
    assert not c.raw_grid.isHidden()
    assert len(c.raw_pictures) == 4
    for (plane, name), label in c.raw_pictures.items():
        assert not label.pixmap().isNull(), (plane, name)
        assert label.toolTip()
    # The loop lies in x = 0 round (20, 0, 0), inside the segmented block:
    # its own plane is white there in the segmentation, bright in the raw image.
    assert _grey_at_middle(c.raw_pictures[("Its own plane", "Segmentation")]) > 200
    assert _grey_at_middle(c.raw_pictures[("Its own plane", "Raw image")]) > 150
    assert "the chosen item cyan" in c.raw_status.text()

    c.raw_check.setChecked(False)
    assert c.raw_grid.isHidden() and c.raw_status.text() == ""
    assert all(label.pixmap() is None or label.pixmap().isNull() for label in c.raw_pictures.values())

    # Only beside a list that has items to draw.
    c.raw_check.setChecked(True)
    c.review_choice.setCurrentText(JUNCTIONS)
    assert c.raw_box.isHidden()
    c.review_choice.setCurrentText(NO_LIST)
    assert c.raw_box.isHidden()


def test_the_raw_panel_without_a_raw_image_draws_the_segmentation_and_says_why(make_napari_viewer):
    c = _raw_panel_page(make_napari_viewer(), raw=False)
    c.raw_check.setChecked(True)
    assert not c.raw_pictures[("Its own plane", "Segmentation")].pixmap().isNull()
    assert c.raw_pictures[("Its own plane", "Raw image")].text() == "None"
    assert "No raw image: no raw image is set on the Input tab." in c.raw_status.text()


def test_the_raw_panel_draws_the_chosen_item_of_any_other_list(make_napari_viewer):
    """The same pictures for a review list's item: here a terminal node's."""
    c = _raw_panel_page(make_napari_viewer(), raw=True)
    c.raw_check.setChecked(True)
    c.review_choice.setCurrentText("Terminal nodes (add or remove inlets and outlets)")
    assert not c.raw_box.isHidden()
    assert c.raw_status.text() == "Choose an item in the list to see it here."
    c.scan_button.click()
    assert c.state.review_item is not None
    assert all(not label.pixmap().isNull() for label in c.raw_pictures.values())
    assert "the chosen item cyan" in c.raw_status.text()
