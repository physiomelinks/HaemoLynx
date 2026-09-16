"""The "Edit" button and its floating window, built for real, with a viewer.

Uses the same `panel`/`make_napari_viewer` fixture as `test_gui_widget.py`:
unlike "Check segmented image", this feature has to click on real vessels/
nodes napari layers, so it needs an actual viewer, not just a QApplication.

They need napari, a Qt binding and a display, so they are marked `gui` and
skipped everywhere those are missing, same as `test_gui_widget.py`.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import _apply_layers, settings_widget  # noqa: E402
from haemolynx.gui.results import EDIT_DRAFT, NODES, VESSELS, ResultLayers  # noqa: E402

from test_gui_results import a_graph, network  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(make_napari_viewer):
    viewer = make_napari_viewer()
    return settings_widget(napari_viewer=viewer), viewer


def _with_vessel_layers(panel_obj, viewer, graph=None) -> None:
    """Put a real image/vessels/nodes layer set in *viewer*, from *graph*.

    Mirrors what a real run's skeletonise + build_network stages do, without
    running the whole pipeline -- the Edit window only needs the layers (the
    IMAGE one supplies Add branch's segmented-mask cost field), not a real run.
    """
    results = ResultLayers()
    _apply_layers(
        viewer,
        results.stage_finished(
            "skeletonise",
            SimpleNamespace(
                image=np.ones((10, 10, 10), dtype=np.uint8),
                skeleton=np.zeros((10, 10, 10), dtype=bool),
                voxel_size_xyz=(1.0, 1.0, 1.0),
                voxel_size_zyx=(1.0, 1.0, 1.0),
            ),
        ),
    )
    _apply_layers(
        viewer, results.stage_finished("build_network", network(graph or a_graph()))
    )
    panel_obj._haemolynx_view.results = results


# --- the button exists and is wired -----------------------------------------


def test_edit_button_exists_with_a_tooltip(panel):
    widget, _viewer = panel
    button = widget._haemolynx_edit_button
    assert button.text == "Edit"
    assert button.tooltip


def test_opening_the_editor_with_nothing_to_edit_reports_instead_of_raising(panel):
    widget, _viewer = panel
    widget._haemolynx_open_graph_editor()
    assert "Nothing to edit yet" in widget._haemolynx_report()
    assert widget._haemolynx_graph_editor["state"] is None


# --- opening the editor -------------------------------------------------------


def test_opening_the_editor_arms_click_handling_on_the_real_layers(panel):
    widget, viewer = panel
    _with_vessel_layers(widget, viewer)

    widget._haemolynx_open_graph_editor()

    graph_editor = widget._haemolynx_graph_editor
    assert graph_editor["state"] is not None
    assert graph_editor["window"] is not None
    assert graph_editor["window"].is_open()


def test_opening_the_editor_installs_the_click_callback_on_vessels_and_nodes(panel):
    widget, viewer = panel
    _with_vessel_layers(widget, viewer)

    widget._haemolynx_open_graph_editor()

    for name in (VESSELS, NODES):
        layer = viewer.layers[name]
        assert len(layer.mouse_drag_callbacks) >= 1


# --- driving the editor state directly (the click-hit-testing itself is ---
# --- exercised at the unit level in test_gui_graph_click.py / ---
# --- test_gui_graph_editor.py) -------------------------------------------


def test_add_then_regenerate_round_trip_updates_the_real_vessels_layer(panel):
    widget, viewer = panel
    _with_vessel_layers(widget, viewer)
    widget._haemolynx_open_graph_editor()

    graph_editor = widget._haemolynx_graph_editor
    state = graph_editor["state"]
    before = len(viewer.layers[VESSELS].data)

    from haemolynx.gui.graph_click import NodeHit

    state.start_add()
    state.click_add((0.0, 0.0, 0.0), NodeHit(node_id=0))
    state.click_add((2.0, 0.0, 0.0), None)
    new_node = state.finish_add()

    # The window's own refresh path (what a real click would trigger too).
    group = widget._haemolynx_view.results.layers_for_graph(state.graph)
    _apply_layers(viewer, group)

    assert new_node is not None
    after = len(viewer.layers[VESSELS].data)
    assert after > before
    assert new_node in list(viewer.layers[NODES].features["node_id"])


def test_delete_updates_the_real_vessels_layer(panel):
    widget, viewer = panel
    _with_vessel_layers(widget, viewer)
    widget._haemolynx_open_graph_editor()

    state = widget._haemolynx_graph_editor["state"]
    before_edges = state.graph.number_of_edges()

    from haemolynx.gui.graph_click import EdgeHit

    state.start_delete()
    state.click_delete(EdgeHit(u=0, v=1, key=0, point_um=(5.0, 0.0, 0.0)))
    group = widget._haemolynx_view.results.layers_for_graph(state.graph)
    _apply_layers(viewer, group)

    assert state.graph.number_of_edges() < before_edges
    assert not state.graph.has_edge(0, 1)


# --- the real, wired-up mouse callback (not just the pure state machine) ---
# --- -- this is the entry point a real click actually goes through, and ---
# --- the one none of the tests above (which drive `state` directly) touch ---


def _fake_click_event(position):
    """Just enough of a napari mouse event for `_graph_editor_click` to read."""
    return SimpleNamespace(position=position, dims_displayed=(0, 1, 2), view_direction=None)


def test_a_real_click_that_misses_everything_extends_the_draft_and_shows_it(panel):
    """Regression: a click that resolves to no hit (e.g. empty space, or
    just outside the segmented volume) used to reach an unguarded
    route_through_array through this exact real callback and raise -- and
    even when it did not raise, the extended draft was never drawn anywhere.
    Driving the real `mouse_drag_callbacks` entry (not `state` directly)
    is what would have caught both."""
    from haemolynx.gui.graph_click import NodeHit

    widget, viewer = panel
    _with_vessel_layers(widget, viewer)
    widget._haemolynx_open_graph_editor()

    state = widget._haemolynx_graph_editor["state"]
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), NodeHit(node_id=0))

    click = viewer.layers[VESSELS].mouse_drag_callbacks[-1]
    # Far outside the tiny 10x10x10 fixture volume -- guaranteed to miss
    # both the vessels and nodes layers, and past the cost field's own
    # bounds -- astar_path clamps this to the volume's own far corner
    # (voxel (9, 9, 9), 1um voxels here) rather than raising.
    click(viewer.layers[VESSELS], _fake_click_event((500.0, 500.0, 500.0)))

    assert state.mode == "add"
    assert state.draft is not None
    assert state.draft.points_um[-1] == (9.0, 9.0, 9.0)
    assert EDIT_DRAFT in viewer.layers
    drawn = np.asarray(viewer.layers[EDIT_DRAFT].data[0])
    assert tuple(drawn[-1]) == (9.0, 9.0, 9.0)


def test_finishing_a_draft_through_the_real_flow_clears_the_preview_layer(panel):
    """Regression: the draft-preview layer was never cleaned up -- once
    drawn (see the test above), a finished/committed branch would leave its
    last in-progress segment on screen forever."""
    from haemolynx.gui.graph_click import NodeHit

    widget, viewer = panel
    _with_vessel_layers(widget, viewer)
    widget._haemolynx_open_graph_editor()

    state = widget._haemolynx_graph_editor["state"]
    state.start_add()
    state.click_add((0.0, 0.0, 0.0), NodeHit(node_id=0))
    click = viewer.layers[VESSELS].mouse_drag_callbacks[-1]
    click(viewer.layers[VESSELS], _fake_click_event((2.0, 0.0, 0.0)))
    assert EDIT_DRAFT in viewer.layers

    widget._haemolynx_finish_graph_editor_branch()

    assert state.draft is None
    assert EDIT_DRAFT not in viewer.layers


# --- Regenerate with nothing recorded yet -------------------------------------


def test_regenerate_with_no_checkpoint_reports_instead_of_running(panel):
    widget, viewer = panel
    _with_vessel_layers(widget, viewer)
    widget._haemolynx_open_graph_editor()

    widget._haemolynx_regenerate_from_edit()

    assert "Nothing to regenerate from" in widget._haemolynx_report()
