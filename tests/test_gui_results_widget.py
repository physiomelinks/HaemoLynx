"""Putting a run's results in a real viewer.

`test_gui_results.py` decides *what* each stage should show, without a display.
What is left for here is the part only a real viewer can answer: that the specs
become layers of the right type at the right scale, that a second run updates
its own work and never the user's, and that the results reach the viewer at all
-- the delivery path from `_run_in_background` to `finished`, which nothing
covered before, which is how the finished graph came to be dropped on the floor
unnoticed.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    _add_or_update,
    _apply_layers,
    _clear_our_layers,
    _run_in_background,
)
from haemolynx.gui.results import (  # noqa: E402
    IMAGE,
    NODES,
    SKELETON,
    VESSELS,
    LayerSpec,
    ResultLayers,
    StageLayers,
)
from tests.test_gui_results import a_graph, network  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


def a_run(voxel_size_zyx=(2.0, 1.0, 0.5)) -> list[StageLayers]:
    """The groups a small run would produce, in order."""
    results = ResultLayers()
    graph = a_graph()
    return [
        results.stage_finished(
            "skeletonise",
            SimpleNamespace(
                image=np.zeros((4, 4, 4), dtype=np.uint8),
                skeleton=np.zeros((4, 4, 4), dtype=bool),
                voxel_size_xyz=tuple(reversed(voxel_size_zyx)),
                voxel_size_zyx=voxel_size_zyx,
            ),
        ),
        results.stage_finished("build_network", network(graph, voxel_size_zyx)),
    ]


# --- the specs become layers -------------------------------------------------


def test_a_stage_becomes_layers_of_the_right_type_and_scale(viewer):
    for group in a_run():
        _apply_layers(viewer, group)

    by_name = {layer.name: layer for layer in viewer.layers}
    assert isinstance(by_name[IMAGE], napari.layers.Image)
    assert isinstance(by_name[SKELETON], napari.layers.Labels)
    assert isinstance(by_name[VESSELS], napari.layers.Vectors)
    assert isinstance(by_name[NODES], napari.layers.Points)

    # The registration rule, checked where it can actually be checked.
    assert tuple(by_name[SKELETON].scale) == (2.0, 1.0, 0.5)
    assert tuple(by_name[VESSELS].scale) == (1.0, 1.0, 1.0)


def test_the_view_turns_to_3d_when_the_geometry_arrives(viewer):
    assert viewer.dims.ndisplay == 2
    for group in a_run():
        _apply_layers(viewer, group)
    assert viewer.dims.ndisplay == 3


def test_the_features_reach_the_layer(viewer):
    for group in a_run():
        _apply_layers(viewer, group)
    vessels = viewer.layers[VESSELS]
    assert "length" in vessels.features
    assert len(vessels.features) > 0


# --- a second run ------------------------------------------------------------


def test_running_again_updates_our_layers_rather_than_piling_them_up(viewer):
    for group in a_run():
        _apply_layers(viewer, group)
    before = len(viewer.layers)
    same_layer = viewer.layers[VESSELS]

    for group in a_run():
        _apply_layers(viewer, group)

    assert len(viewer.layers) == before
    assert viewer.layers[VESSELS] is same_layer


def test_what_the_user_changed_survives_a_second_run(viewer):
    """Someone who hid a layer does not want it back every run."""
    for group in a_run():
        _apply_layers(viewer, group)
    viewer.layers[SKELETON].visible = False
    viewer.layers[VESSELS].opacity = 0.25

    for group in a_run():
        _apply_layers(viewer, group)

    assert viewer.layers[SKELETON].visible is False
    assert viewer.layers[VESSELS].opacity == 0.25


def test_a_layer_of_the_users_with_the_same_name_is_never_touched(viewer):
    """The one unrecoverable failure: silently overwriting someone's work."""
    mine = viewer.add_points(np.zeros((3, 3)), name=VESSELS)
    mine_data = mine.data.copy()

    for group in a_run():
        _apply_layers(viewer, group)

    assert viewer.layers[VESSELS] is mine
    assert np.array_equal(viewer.layers[VESSELS].data, mine_data)
    assert f"{VESSELS} (HaemoLynx)" in {layer.name for layer in viewer.layers}


def test_clearing_removes_only_our_layers(viewer):
    theirs = viewer.add_image(np.zeros((4, 4, 4)), name="their data")
    for group in a_run():
        _apply_layers(viewer, group)

    removed = _clear_our_layers(viewer)

    assert removed >= 4
    assert [layer.name for layer in viewer.layers] == [theirs.name]


def test_a_layer_that_changed_type_is_replaced(viewer):
    spec = LayerSpec(kind="points", name="HaemoLynx nodes", data=np.zeros((2, 3)))
    _add_or_update(viewer, spec)
    assert isinstance(viewer.layers[NODES], napari.layers.Points)

    _add_or_update(
        viewer,
        LayerSpec(kind="image", name="HaemoLynx nodes", data=np.zeros((4, 4, 4))),
    )
    assert isinstance(viewer.layers[NODES], napari.layers.Image)


# --- the delivery path -------------------------------------------------------


def test_a_run_puts_its_results_in_the_viewer(viewer, qtbot, monkeypatch):
    """`_run_in_background` -> `finished`, which nothing covered before."""
    from haemolynx.gui import _widget

    graph = a_graph()
    scripted = [
        ("skeletonise", SimpleNamespace(
            image=np.zeros((4, 4, 4), dtype=np.uint8),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0), voxel_size_zyx=(1.0, 1.0, 1.0))),
        ("build_network", network(graph)),
    ]

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        for name, output in scripted:
            if on_stage_output is not None:
                on_stage_output(name, output)
        return graph

    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    _run_in_background({}, None, report, button, None,
                       viewer=viewer, results=ResultLayers())

    qtbot.waitUntil(lambda: VESSELS in viewer.layers, timeout=5000)
    qtbot.waitUntil(lambda: button.enabled, timeout=5000)
    assert {IMAGE, SKELETON, VESSELS, NODES} <= {layer.name for layer in viewer.layers}
    assert "Finished" in report.value


def test_a_failure_to_draw_does_not_stop_the_run(viewer, qtbot, monkeypatch):
    """A view bug must never end a run -- an eight-hour one least of all."""
    from haemolynx.gui import _widget

    graph = a_graph()

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        if on_stage_output is not None:
            on_stage_output("build_network", network(graph))
        return graph

    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)
    monkeypatch.setattr(
        ResultLayers, "stage_finished",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("drawing broke")),
    )

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    _run_in_background({}, None, report, button, None,
                       viewer=viewer, results=ResultLayers())

    qtbot.waitUntil(lambda: button.enabled, timeout=5000)
    assert "Finished" in report.value
    assert VESSELS not in viewer.layers


def test_a_run_with_the_viewer_switched_off_shows_nothing(viewer, qtbot, monkeypatch):
    from haemolynx.gui import _widget

    graph = a_graph()

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        assert on_stage_output is None, "outputs should not be asked for"
        return graph

    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    _run_in_background({}, None, report, button, None, viewer=None, results=None)

    qtbot.waitUntil(lambda: button.enabled, timeout=5000)
    assert len(viewer.layers) == 0


# --- the panel's own controls ------------------------------------------------


def test_the_panel_offers_the_view_controls(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    panel = settings_widget(napari_viewer=make_napari_viewer())

    assert panel._haemolynx_show_results.value is True
    assert set(panel._haemolynx_colour) == {"vessels", "nodes"}
    assert panel._haemolynx_colour["vessels"].value == "none"
