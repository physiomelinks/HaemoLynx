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
from test_gui_results import a_graph, network  # noqa: E402

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
    # Colouring is not the panel's business at all: which feature the colours
    # follow, and the range they span, both live in napari's layer controls on
    # the left. The panel configures and runs the pipeline.
    assert not hasattr(panel, "_haemolynx_colour")
    assert not hasattr(panel, "_haemolynx_scales")


# --- the colour-by dropdowns learn what a stage made available ---------------


def _solved_run() -> list[StageLayers]:
    """A run through to `solve`, which is where flow and pressure appear."""
    results = ResultLayers()
    graph = a_graph(conductance=1e-18, resistance=1e18)
    groups = [results.stage_finished("build_network", network(graph))]
    for index, (u, v, key, data) in enumerate(graph.edges(keys=True, data=True)):
        data["pressure_u"] = 1000.0
        data["pressure_v"] = 500.0
        data["pressure_drop"] = 500.0
        data["flow_signed"] = 5e-16 * (1 if index % 2 else -1)
        data["flow_abs"] = 5e-16
    groups.append(
        results.stage_finished(
            "solve",
            SimpleNamespace(
                node_list=list(graph.nodes),
                pressure=np.array([1000.0, 900.0, 700.0, 500.0]),
                equivalent_resistance=1e18,
            ),
        )
    )
    return groups





def test_choosing_none_actually_uncolours_the_layer(make_napari_viewer):
    """"none" is offered, so it has to do something."""
    import numpy.testing as npt

    from haemolynx.gui._widget import UNCOLOURED, _colour_layer

    viewer = make_napari_viewer()
    for group in _solved_run():
        _apply_layers(viewer, group)
    layer = viewer.layers[VESSELS]

    _colour_layer(layer, "flow_abs")
    coloured = np.array(layer.edge_color, copy=True)

    _colour_layer(layer, "none")
    flat = np.asarray(layer.edge_color)

    assert len(np.unique(flat, axis=0)) == 1, "every vessel should look the same"
    assert not np.allclose(flat, coloured) or len(np.unique(coloured, axis=0)) == 1
    assert UNCOLOURED == "#cccccc"
    npt.assert_allclose(flat[0][:3], 0.8, atol=0.02)




def test_branch_order_then_flow_does_not_raise_keyerror_nan(make_napari_viewer):
    """The real sequence of a run, which used to end in `KeyError: nan`.

    The diameters stage colours by `branch_order`, which is text, so the layer
    is left in cycle mode. The solve then colours by `flow_abs`, which is full
    of NaN because `set_edge_flows` skips any edge with no conductance. napari's
    `CategoricalColormap.map` tests membership with `np.isin`, and NaN is never
    equal to itself, so the value is filed under a key that can never be looked
    up again and the very next lookup raises. No user interaction needed -- the
    two default colourings are enough.
    """
    from haemolynx.gui._widget import _colour_layer

    viewer = make_napari_viewer()
    graph = a_graph(conductance=1e-18, branch_order="BO1")
    results = ResultLayers()
    _apply_layers(viewer, results.stage_finished("build_network", network(graph)))
    layer = viewer.layers[VESSELS]

    _colour_layer(layer, "branch_order", "categorical",
                  (("BO1", (1.0, 0.0, 0.0, 1.0)),))
    assert layer.edge_color_mode == "cycle"

    # Only some edges get a flow, which is the case that bites.
    partial = np.array([1e-16, np.nan, 5e-17])
    features = dict(layer.features)
    features["flow_abs"] = np.repeat(partial, len(layer.data) // 3)[: len(layer.data)]
    layer.features = features

    _colour_layer(layer, "flow_abs", "continuous")

    assert layer.edge_color_mode == "colormap"
    assert len(layer.edge_color) == len(layer.data)


def test_flow_then_branch_order_does_not_raise_either(make_napari_viewer):
    """The mirror image: colormap mode meeting a text column."""
    from haemolynx.gui._widget import _colour_layer

    viewer = make_napari_viewer()
    graph = a_graph(conductance=1e-18, branch_order="BO1")
    _apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(graph)))
    layer = viewer.layers[VESSELS]

    _colour_layer(layer, "length", "continuous")
    assert layer.edge_color_mode == "colormap"

    _colour_layer(layer, "branch_order", "categorical",
                  (("BO1", (1.0, 0.0, 0.0, 1.0)),))
    assert layer.edge_color_mode == "cycle"
    assert len(layer.edge_color) == len(layer.data)


def test_a_stage_with_no_opinion_leaves_the_colouring_alone(make_napari_viewer):
    """`None` means "say nothing", `"none"` means "clear it". Not the same.

    Most stages after build_network name no colouring. Treating that as a
    request to blank the layer threw away the previous stage's colouring at
    every one of them -- the vessels went flat grey at assign_boundaries and
    stayed that way until a stage that did have an opinion.
    """
    from haemolynx.gui._widget import UNCOLOURED, _colour_layer

    viewer = make_napari_viewer()
    _apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(a_graph())))
    layer = viewer.layers[VESSELS]

    _colour_layer(layer, "length", "continuous")
    coloured = np.array(layer.edge_color, copy=True)

    _colour_layer(layer, None)                       # a stage with nothing to say
    assert np.allclose(np.asarray(layer.edge_color), coloured)

    _colour_layer(layer, "none")                     # the user, deliberately
    assert len(np.unique(np.asarray(layer.edge_color), axis=0)) == 1
    assert UNCOLOURED == "#cccccc"


def test_the_contrast_limits_attribute_is_one_napari_actually_has(make_napari_viewer):
    """A napari layer accepts setattr of a name it does not have.

    `layer.edge_color_contrast_limits = ...` raises nothing and does nothing:
    the value lands on a stray attribute while the real limits keep their old
    value. We set that name for a while, so colouring by `segment_id` (0..9)
    and then by `flow_abs` (0..1.5e-13) left the flow mapped against 0..9 --
    every vessel at the bottom of the colormap, one flat colour, no error.

    So assert the names exist rather than trusting a try/except to catch a typo.
    """
    viewer = make_napari_viewer()
    _apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(a_graph())))

    assert hasattr(viewer.layers[VESSELS], "edge_contrast_limits")
    assert hasattr(viewer.layers[NODES], "face_contrast_limits")
    assert not hasattr(type(viewer.layers[VESSELS]), "edge_color_contrast_limits")


def test_colouring_by_a_small_column_after_a_large_one_still_spreads(make_napari_viewer):
    """The symptom the wrong attribute produced, stated as a test."""
    from haemolynx.gui._widget import _colour_layer

    viewer = make_napari_viewer()
    _apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(a_graph())))
    layer = viewer.layers[VESSELS]

    big = np.linspace(0.0, 9.0, len(layer.data))
    tiny = np.linspace(0.0, 1.5e-13, len(layer.data))
    features = dict(layer.features)
    features["big"], features["tiny"] = big, tiny
    layer.features = features

    _colour_layer(layer, "big", "continuous", limits=(0.0, 9.0))
    _colour_layer(layer, "tiny", "continuous", limits=(0.0, 1.5e-13))

    assert layer.edge_contrast_limits == (0.0, 1.5e-13)
    assert len(np.unique(np.asarray(layer.edge_color), axis=0)) > 1, (
        "the tiny column is mapped against the big column's range"
    )


# --- the colour bar, which lives in the layer controls on the left ----------


def a_drawn_run(make_napari_viewer, **edge_attributes):
    """A viewer with our layers in it, and the vessels' colour bar."""
    from haemolynx.gui._widget import _layer_controls, settings_widget

    viewer = make_napari_viewer()
    settings_widget(napari_viewer=viewer)          # the panel builds no bars
    _apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(a_graph(**edge_attributes))))
    layer = viewer.layers[VESSELS]
    controls = _layer_controls(viewer, layer)
    return viewer, layer, controls._haemolynx_scale


def test_the_colour_bar_is_added_to_the_layers_own_controls(make_napari_viewer):
    """napari draws none for a feature colouring, so we add one where it belongs."""
    from haemolynx.gui._widget import _layer_controls

    viewer, layer, scale = a_drawn_run(make_napari_viewer)

    assert scale.shown is True
    assert scale.heading.text() == "segment_id"
    assert scale.bar.pixmap() is not None and not scale.bar.pixmap().isNull()
    assert float(scale.low.text()) == 0.0

    # And the nodes get their own, in their own controls.
    nodes_scale = _layer_controls(viewer, viewer.layers[NODES])._haemolynx_scale
    assert nodes_scale is not scale
    assert nodes_scale.shown is True


def test_the_bar_follows_a_colouring_chosen_on_the_left(make_napari_viewer):
    """The choice is napari's now, so the bar cannot wait to be told."""
    _viewer, layer, scale = a_drawn_run(make_napari_viewer)

    layer.edge_color = "length"          # as napari's own dropdown does it

    assert scale.heading.text() == "length"
    assert float(scale.high.text()) > 0


def test_only_one_bar_is_added_however_often_layers_are_applied(make_napari_viewer):
    from haemolynx.gui._widget import _layer_controls

    viewer, _layer, scale = a_drawn_run(make_napari_viewer)
    rows = _layer_controls(viewer, viewer.layers[VESSELS]).layout().rowCount()

    for _ in range(3):
        _apply_layers(viewer, ResultLayers().stage_finished(
            "build_network", network(a_graph())))

    controls = _layer_controls(viewer, viewer.layers[VESSELS])
    assert controls.layout().rowCount() == rows
    assert controls._haemolynx_scale is scale


def test_typing_a_range_changes_what_the_colours_span(make_napari_viewer):
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _colour_layer(layer, "length", "continuous", limits=(0.0, 10.0))
    scale.follow_the_layer()

    scale.low.setText("2")
    scale.high.setText("8")
    scale.low.editingFinished.emit()

    assert layer.edge_contrast_limits == (2.0, 8.0)


def test_a_range_that_makes_no_sense_is_put_back(make_napari_viewer):
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _colour_layer(layer, "length", "continuous", limits=(0.0, 10.0))
    scale.follow_the_layer()

    for bad_low, bad_high in (("not a number", "8"), ("9", "3")):
        scale.low.setText(bad_low)
        scale.high.setText(bad_high)
        scale.low.editingFinished.emit()
        assert layer.edge_contrast_limits == (0.0, 10.0)
        assert float(scale.low.text()) == 0.0


def test_both_fit_buttons_apply_a_range(make_napari_viewer):
    """Why "Fit 1-99%" exists: one huge vessel flattens all the others.

    With only a handful of segments a percentile cannot exclude the outlier;
    what the trimming does to a real distribution is checked in
    test_gui_results.py, without a display. Here: both buttons apply.
    """
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    features = dict(layer.features)
    skewed = np.linspace(1e-16, 2e-16, len(layer.data))
    skewed[-1] = 1e-9
    features["flow_abs"] = skewed
    layer.features = features
    _colour_layer(layer, "flow_abs", "continuous", limits=(0.0, 1e-9))
    scale.follow_the_layer()

    assert scale.autoscale(0.0, 100.0)
    wide = layer.edge_contrast_limits
    assert scale.autoscale(1.0, 99.0)

    assert layer.edge_contrast_limits[1] <= wide[1]
    assert float(scale.high.text()) == pytest.approx(layer.edge_contrast_limits[1])
    assert len(np.unique(np.asarray(layer.edge_color), axis=0)) > 1


def test_a_text_colouring_has_no_range_to_show(make_napari_viewer):
    """branch_order is a cycle of colours, not a scale; hide the bar."""
    _viewer, _layer, scale = a_drawn_run(make_napari_viewer, branch_order="BO1")

    scale.refresh("branch_order")
    assert scale.shown is False, "a colour cycle has no range to draw"

    scale.refresh("length")
    assert scale.shown is True


def test_a_missing_layer_controls_panel_is_not_fatal(make_napari_viewer):
    """All of this is private napari API, so it has to fail quietly."""
    from haemolynx.gui import _widget

    viewer = make_napari_viewer()
    _widget._apply_layers(viewer, ResultLayers().stage_finished(
        "build_network", network(a_graph())))
    assert VESSELS in viewer.layers

    # Pretend a napari version moved the controls out from under us.
    original = _widget._layer_controls
    _widget._layer_controls = lambda *_a, **_k: None
    try:
        _widget._apply_layers(viewer, ResultLayers().stage_finished(
            "build_network", network(a_graph())))
    finally:
        _widget._layer_controls = original
    assert VESSELS in viewer.layers


def test_rescaling_repaints_without_reselecting_the_feature(make_napari_viewer):
    """The colours have to change when the range does, not later.

    `ColorManager.contrast_limits` is a plain field, so setting it recolours
    the model and emits nothing: the canvas keeps drawing the buffer it has,
    and the change only showed up once you picked another feature and came
    back -- that assignment is what fires the event. Pressing "Fit 1-99%"
    looked like it had done nothing at all.
    """
    from haemolynx.gui._widget import _apply_contrast_limits, _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _colour_layer(layer, "length", "continuous", limits=(0.0, 10.0))
    scale.follow_the_layer()

    repaints = []
    layer.events.edge_color.connect(lambda *_a: repaints.append(1))

    assert _apply_contrast_limits(layer, 0.0, 1.0)

    assert repaints, "the canvas was never told the colours changed"
    assert layer.edge_contrast_limits == (0.0, 1.0)


def test_the_fit_buttons_repaint_too(make_napari_viewer):
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _colour_layer(layer, "length", "continuous", limits=(0.0, 100.0))
    scale.follow_the_layer()

    repaints = []
    layer.events.edge_color.connect(lambda *_a: repaints.append(1))

    assert scale.autoscale(1.0, 99.0)
    assert repaints, "Fit 1-99% changed the range but not the picture"


# --- the colour bar in the canvas -------------------------------------------


def test_a_checkbox_puts_the_colour_bar_in_the_viewer(make_napari_viewer):
    """napari registers a colorbar overlay for Points and none for Vectors.

    What is asserted is the overlay and its visibility, not pixels: an
    offscreen viewer builds no overlay visuals at all -- not for Image, Points
    or anything else -- so a screenshot would prove nothing either way. What
    makes a late-added overlay real is that the canvas connects to
    `_overlays.events.added`, so the test below checks that event fires.
    """
    from haemolynx.gui._widget import _layer_controls, _viewer_colorbar

    viewer, layer, scale = a_drawn_run(make_napari_viewer)

    assert scale.in_viewer.isChecked() is False
    assert _viewer_colorbar(layer).visible is False

    scale.in_viewer.setChecked(True)
    assert _viewer_colorbar(layer).visible is True

    scale.in_viewer.setChecked(False)
    assert _viewer_colorbar(layer).visible is False

    # Points come with theirs already registered; vessels needed one adding.
    nodes = viewer.layers[NODES]
    nodes_scale = _layer_controls(viewer, nodes)._haemolynx_scale
    nodes_scale.in_viewer.setChecked(True)
    assert nodes._overlays["face_colorbar"].visible is True
    assert "edge_colorbar" in layer._overlays


def test_adding_the_overlay_tells_the_canvas(make_napari_viewer):
    """A Vectors colorbar is added after the layer was built, so it has to.

    `VispyCanvas` connects to each layer's `_overlays.events.added` and builds
    the visual from there. Without that event the overlay would sit in the dict
    and never draw, and the checkbox would be a lie.
    """
    from haemolynx.gui._widget import _viewer_colorbar

    _viewer, layer, _scale = a_drawn_run(make_napari_viewer)
    assert "edge_colorbar" not in layer._overlays

    added = []
    layer._overlays.events.added.connect(lambda *_a: added.append(1))
    _viewer_colorbar(layer)

    assert added, "the canvas was never told a new overlay exists"


def test_the_checkbox_reports_the_overlay_it_did_not_set(make_napari_viewer):
    """Someone may switch the overlay on from the console; do not lie about it."""
    from haemolynx.gui._widget import _viewer_colorbar

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _viewer_colorbar(layer).visible = True

    scale.follow_the_layer()

    assert scale.in_viewer.isChecked() is True


# --- the range is worked out, not left over ---------------------------------


def test_a_colouring_is_scaled_to_its_own_column(make_napari_viewer):
    """Selected and shown are not the same thing.

    The range used to be applied after the column had already been mapped, and
    by a route that re-mapped nothing, so the colours stayed scaled to whatever
    the previous stage had used. `flow_abs` was the chosen colouring at the
    solve and the network was a single flat colour, because flows of 1e-13 were
    being mapped against segment ids of 0..9.
    """
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, _scale = a_drawn_run(make_napari_viewer)
    features = dict(layer.features)
    features["flow_abs"] = np.linspace(0.0, 1.5e-13, len(layer.data))
    layer.features = features

    _colour_layer(layer, "segment_id", "continuous", limits=(0.0, 9.0))
    _colour_layer(layer, "flow_abs", "continuous")      # no limits given

    assert layer.edge_contrast_limits == pytest.approx((0.0, 1.5e-13))
    assert len(np.unique(np.asarray(layer.edge_color), axis=0)) > 1


def test_choosing_a_feature_on_the_left_rescales_it(make_napari_viewer):
    """napari applies a new column with the old range; nothing else would fit it."""
    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    features = dict(layer.features)
    features["flow_abs"] = np.linspace(0.0, 1.5e-13, len(layer.data))
    layer.features = features

    layer.edge_color = "segment_id"
    scale.follow_the_layer()
    layer.edge_color = "flow_abs"          # as napari's own dropdown does it
    scale.follow_the_layer()

    assert layer.edge_contrast_limits == pytest.approx((0.0, 1.5e-13))
    assert float(scale.high.text()) == pytest.approx(1.5e-13)


def test_a_range_you_set_yourself_is_not_refitted(make_napari_viewer):
    """Autofitting on every event would undo the range you just typed."""
    from haemolynx.gui._widget import _colour_layer

    _viewer, layer, scale = a_drawn_run(make_napari_viewer)
    _colour_layer(layer, "length", "continuous")
    scale.follow_the_layer()

    scale.low.setText("2")
    scale.high.setText("8")
    scale.low.editingFinished.emit()
    scale.follow_the_layer()               # same column, so leave it be

    assert layer.edge_contrast_limits == (2.0, 8.0)


# --- the node feature dropdown napari does not provide ----------------------


def test_the_nodes_get_a_feature_dropdown(make_napari_viewer):
    """QtPointsControls has a colour swatch and no way to colour by a column."""
    from haemolynx.gui._widget import _active_column, _layer_controls

    viewer, _vessels, _scale = a_drawn_run(make_napari_viewer)
    nodes = viewer.layers[NODES]
    chooser = _layer_controls(viewer, nodes)._haemolynx_feature

    offered = [chooser.native.itemText(i) for i in range(chooser.native.count())]
    assert "degree" in offered and "pressure" in offered
    # Identifiers are not quantities; colouring by one shows nothing.
    assert "node_id" not in offered

    chooser.native.setCurrentText("pressure")
    assert _active_column(nodes) == "pressure"


def test_the_node_dropdown_gains_columns_as_stages_land(make_napari_viewer):
    """Rebuilt from the layer, unlike napari's own, which is filled once."""
    from haemolynx.gui._widget import _layer_controls

    viewer, _vessels, _scale = a_drawn_run(make_napari_viewer)
    nodes = viewer.layers[NODES]
    chooser = _layer_controls(viewer, nodes)._haemolynx_feature

    features = dict(nodes.features)
    features["something_new"] = np.zeros(len(nodes.data))
    nodes.features = features
    chooser.refresh()

    offered = [chooser.native.itemText(i) for i in range(chooser.native.count())]
    assert "something_new" in offered


def test_the_vessels_keep_napari_s_own_dropdown(make_napari_viewer):
    """Vectors already has "edge feature:"; do not add a second one."""
    from haemolynx.gui._widget import _layer_controls

    viewer, vessels, _scale = a_drawn_run(make_napari_viewer)
    assert getattr(_layer_controls(viewer, vessels), "_haemolynx_feature", None) is None
