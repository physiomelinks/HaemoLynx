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

import threading
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from pytestqt.exceptions import capture_exceptions  # noqa: E402

from haemolynx.gui._widget import (  # noqa: E402
    _add_or_update,
    _apply_layers,
    _clear_our_layers,
    _run_in_background,
)
from haemolynx.gui.progress import BarState  # noqa: E402
from haemolynx.gui.run_state import ALREADY_RUNNING, CANCELLED, RunState  # noqa: E402
from haemolynx.gui.results import (  # noqa: E402
    BOUNDARY_NODES,
    IMAGE,
    NODES,
    SKELETON,
    VESSELS,
    LayerSpec,
    ResultLayers,
    StageLayers,
    perturbation_layer_names,
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


def test_vessel_mask_volumes_land_as_coloured_image_layers(viewer):
    """Masks used by the run become translucent 3D Image volumes, not Labels."""
    from haemolynx.gui.results import MASK_COLOURS, MASK_LAYERS

    results = ResultLayers()
    voxel_size_zyx = (2.0, 1.0, 0.5)
    art = np.zeros((4, 4, 4), dtype=bool)
    art[2, 2, 2] = True
    ven = np.zeros((4, 4, 4), dtype=bool)
    ven[1, 1, 1] = True
    group = results.stage_finished(
        "skeletonise",
        SimpleNamespace(
            image=np.zeros((4, 4, 4), dtype=np.uint8),
            skeleton=np.zeros((4, 4, 4), dtype=bool),
            voxel_size_xyz=tuple(reversed(voxel_size_zyx)),
            voxel_size_zyx=voxel_size_zyx,
        ),
    )
    _apply_layers(viewer, group)
    group = results.stage_finished(
        "build_network",
        network(
            a_graph(),
            voxel_size_zyx=voxel_size_zyx,
            large_arteriole_mask=art,
            small_venule_mask=ven,
        ),
    )
    _apply_layers(viewer, group)

    art_name = MASK_LAYERS["large_arteriole_mask"]
    ven_name = MASK_LAYERS["small_venule_mask"]
    assert art_name in viewer.layers
    assert ven_name in viewer.layers
    assert MASK_LAYERS["large_venule_mask"] not in {
        layer.name for layer in viewer.layers
    }

    art_layer = viewer.layers[art_name]
    ven_layer = viewer.layers[ven_name]
    assert isinstance(art_layer, napari.layers.Image)
    assert isinstance(ven_layer, napari.layers.Image)
    assert tuple(art_layer.scale) == voxel_size_zyx
    assert art_layer.visible is True
    assert art_layer.rendering == "mip"
    assert art_layer.blending == "translucent"
    # Colormap stops: transparent at 0, role colour at 1.
    art_high = tuple(float(c) for c in art_layer.colormap.colors[-1])
    ven_high = tuple(float(c) for c in ven_layer.colormap.colors[-1])
    assert art_high == pytest.approx(MASK_COLOURS["large_arteriole_mask"], abs=1e-5)
    assert ven_high == pytest.approx(MASK_COLOURS["small_venule_mask"], abs=1e-5)


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


def a_perturbation_group(*names):
    """The layers one perturbations stage would produce, for real."""
    from test_gui_results import a_perturbation, a_perturbation_run, built

    return built().stage_finished(
        "run_perturbations",
        a_perturbation_run(*(a_perturbation(name) for name in names)),
    )


def test_each_perturbation_becomes_its_own_pair_of_layers(viewer):
    for group in a_run():
        _apply_layers(viewer, group)

    _apply_layers(viewer, a_perturbation_group("art_dilate_20", "art_constrict_20"))

    names = {layer.name for layer in viewer.layers}
    assert perturbation_layer_names("art_dilate_20")[0] in names
    assert perturbation_layer_names("art_constrict_20")[0] in names
    assert isinstance(
        viewer.layers[perturbation_layer_names("art_dilate_20")[0]],
        napari.layers.Vectors,
    )


def test_the_baselines_own_layers_survive_a_perturbation(viewer):
    """The comparison only exists if what it is compared against is still there."""
    for group in a_run():
        _apply_layers(viewer, group)
    baseline = viewer.layers[VESSELS]
    baseline_data = baseline.data.copy()

    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))

    assert viewer.layers[VESSELS] is baseline
    assert np.array_equal(viewer.layers[VESSELS].data, baseline_data)


def test_a_perturbation_layer_lands_in_microns(viewer):
    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))
    for name in perturbation_layer_names("art_dilate_20"):
        assert tuple(viewer.layers[name].scale) == (1.0, 1.0, 1.0)


def test_a_users_layer_of_the_same_name_is_never_overwritten(viewer):
    """A perturbation's name comes from a config, so a collision is plausible."""
    vessels_name = perturbation_layer_names("art_dilate_20")[0]
    mine = viewer.add_points(np.zeros((3, 3)), name=vessels_name)
    mine_data = mine.data.copy()

    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))

    assert viewer.layers[vessels_name] is mine
    assert np.array_equal(viewer.layers[vessels_name].data, mine_data)
    assert f"{vessels_name} (HaemoLynx)" in {layer.name for layer in viewer.layers}


def test_clearing_takes_the_perturbation_layers_with_it(viewer):
    """They are ours by metadata, which is what "clear ours" reads -- and they
    are not in `LAYER_NAMES`, which cannot list a name a config invents."""
    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))
    assert perturbation_layer_names("art_dilate_20")[0] in viewer.layers

    removed = _clear_our_layers(viewer)

    assert removed == 2
    assert len(viewer.layers) == 0


def test_a_perturbation_layer_is_added_hidden(viewer):
    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))
    for name in perturbation_layer_names("art_dilate_20"):
        assert viewer.layers[name].visible is False


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


# --- stopping a run on purpose -----------------------------------------------


class Quittable:
    """Stands in for the run's worker where no run is really going."""

    def __init__(self) -> None:
        self.quits = 0

    def quit(self) -> None:
        self.quits += 1


def a_graph_with_depth() -> nx.MultiGraph:
    """`a_graph`'s four nodes, moved off its straight line.

    napari fits the camera to the layers by dividing the canvas by their
    bounding box, so a box with a zero-width side divides by zero and vispy
    then refuses the projection. `a_graph` runs dead straight along z, giving
    it no width at all in y or x. Nothing notices while the viewer is in 2D, or
    while a layer is being updated rather than added -- but the second of two
    runs adds one into a viewer the first already turned to 3D, and that is the
    run these tests are about. No real vessel network is a straight line.
    """
    graph = nx.MultiGraph()
    corners = ((0.0, 0.0, 0.0), (10.0, 4.0, 2.0), (20.0, 1.0, 6.0), (30.0, 7.0, 3.0))
    for node_id, pos in enumerate(corners):
        graph.add_node(node_id, pos=np.array(pos))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        start, end = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
        graph.add_edge(
            u, v, key=0,
            voxels=[start.tolist(), end.tolist()],
            length=float(np.linalg.norm(end - start)),
            segment_id=u,
        )
    return graph


@pytest.fixture
def paused_run(monkeypatch):
    """A fake pipeline that waits after `build_network` until it is released.

    The bug needs a run that really is in progress when the button is pressed,
    and `solve` afterwards to show whether it carried on regardless: the run
    reports its stage output there, which is one of the two points a
    cancellation acts on.
    """
    from haemolynx.gui import _widget

    graph = a_graph_with_depth()
    script = SimpleNamespace(
        graph=graph,
        drawn=threading.Event(),
        resume=threading.Event(),
        stages=[],
    )

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        if on_stage_output is not None:
            on_stage_output("build_network", network(graph))
        script.stages.append("build_network")
        script.drawn.set()
        script.resume.wait(20)
        if on_stage_output is not None:
            on_stage_output(
                "solve",
                SimpleNamespace(node_list=(), pressure=None, equivalent_resistance=None),
            )
        script.stages.append("solve")
        return graph

    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)
    return script


def test_clearing_the_layers_mid_run_stops_it_and_frees_the_panel(
    viewer, qtbot, paused_run
):
    """The bug: this used to leave the run going and the Run button dead."""
    from haemolynx.gui._widget import ProgressBars

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    bars = ProgressBars()
    results = ResultLayers()
    state = RunState(bars=bars)

    _run_in_background({}, None, report, button, bars, viewer=viewer,
                       results=results, state=state)
    qtbot.waitUntil(paused_run.drawn.is_set, timeout=5000)
    qtbot.waitUntil(lambda: VESSELS in viewer.layers, timeout=5000)
    assert state.running is True

    # What pressing "Clear layers" does, in the order the panel does it.
    assert _clear_our_layers(viewer) >= 2
    assert state.cancel() is True
    paused_run.resume.set()

    qtbot.waitUntil(lambda: not state.running, timeout=5000)

    # It stopped at its next checkpoint rather than running to the end.
    assert paused_run.stages == ["build_network"]
    # And nothing already in flight put the cleared layers back.
    assert VESSELS not in viewer.layers
    # The panel is free again: guard clear, button back.
    assert button.enabled is True
    # Reported as an intention, not as a fault.
    assert report.value == CANCELLED
    # Every stateful thing the run left behind is back to nothing.
    assert bars.display.stages == BarState()
    assert bars.display.steps == BarState()
    assert results.colour_options() == []


def test_a_new_run_can_be_started_straight_after_a_cancel(viewer, qtbot, paused_run):
    """The whole point of the fix: no restarting the plugin."""
    from haemolynx.gui._widget import ProgressBars

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    bars = ProgressBars()
    state = RunState(bars=bars)

    _run_in_background({}, None, report, button, bars, viewer=viewer,
                       results=ResultLayers(), state=state)
    qtbot.waitUntil(paused_run.drawn.is_set, timeout=5000)
    _clear_our_layers(viewer)
    state.cancel()
    paused_run.resume.set()
    qtbot.waitUntil(lambda: not state.running, timeout=5000)

    _run_in_background({}, None, report, button, bars, viewer=viewer,
                       results=ResultLayers(), state=state)
    qtbot.waitUntil(lambda: not state.running, timeout=5000)

    assert paused_run.stages == ["build_network", "build_network", "solve"]
    assert VESSELS in viewer.layers
    assert "Finished" in report.value


def test_a_cancelled_run_puts_no_exception_in_front_of_the_user(
    viewer, qtbot, paused_run
):
    """A cancellation is a sentence in the report box, not an error dialog.

    `RunCancelled` leaves the run through the worker's `errored` signal, like
    any other exception -- and superqt gives a worker whose `errored` nobody
    claimed a handler that re-raises. Re-raised, it reaches napari's excepthook
    and the user gets `RunCancelled` and a stack trace for having pressed a
    button, which is the report this replaces.
    """
    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)
    state = RunState()

    with capture_exceptions() as raised:
        _run_in_background({}, None, report, button, None, viewer=viewer,
                           results=ResultLayers(), state=state)
        qtbot.waitUntil(paused_run.drawn.is_set, timeout=5000)
        state.cancel()
        paused_run.resume.set()
        qtbot.waitUntil(lambda: not state.running, timeout=5000)

    assert [kind for kind, _value, _tb in raised] == []
    assert report.value == CANCELLED


def test_a_run_that_really_fails_still_raises_where_napari_can_show_it(
    qtbot, monkeypatch
):
    """The other half: silencing the cancellation must not silence a failure.

    Claiming `errored` takes superqt's re-raise with it, so the handler does it
    instead. A broken run still reaches the excepthook that reports it.
    """
    from haemolynx.gui import _widget

    def fake_run(settings, schema, progress=None, on_stage_output=None):
        raise ValueError("the pipeline broke")

    monkeypatch.setattr(_widget, "run_pipeline_stages", fake_run)

    report = SimpleNamespace(value="")
    button = SimpleNamespace(enabled=True)

    with capture_exceptions() as raised:
        _run_in_background({}, None, report, button, None, viewer=None, results=None)
        qtbot.waitUntil(lambda: button.enabled, timeout=5000)

    assert [kind for kind, _value, _tb in raised] == [ValueError]
    assert report.value == "ValueError: the pipeline broke"


def test_the_clear_button_is_what_stops_the_panels_run(make_napari_viewer):
    """The wiring: the button the user presses is the one that cancels."""
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))

    results = ResultLayers()
    results.stage_finished("build_network", network(a_graph()))
    worker = Quittable()
    panel._haemolynx_run_state.start(worker=worker, results=results)
    panel._haemolynx_run_button.enabled = False
    panel._haemolynx_progress.start()

    panel._haemolynx_clear()

    assert panel._haemolynx_run_state.cancelled is True
    assert worker.quits == 1
    assert "Stopping the run" in panel._haemolynx_report()
    assert results.colour_options() == []
    assert panel._haemolynx_progress.display.stages == BarState()
    assert len(viewer.layers) == 0


def test_clearing_the_layers_with_no_run_going_is_unchanged(make_napari_viewer):
    from haemolynx.gui._widget import settings_widget

    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    # Points, not an image: an image dropped in becomes the run's input, which
    # is a different behaviour and not the one under test.
    theirs = viewer.add_points(np.zeros((3, 3)), name="their data")
    _apply_layers(viewer, a_perturbation_group("art_dilate_20"))

    panel._haemolynx_clear()

    assert [layer.name for layer in viewer.layers] == [theirs.name]
    assert panel._haemolynx_report() == "Removed 2 HaemoLynx layer(s)."
    assert panel._haemolynx_run_button.enabled is True
    assert panel._haemolynx_run_state.cancelled is False


def test_run_says_how_to_stop_the_run_that_is_already_going(make_napari_viewer):
    """The panel's own answer to "why will Run not do anything?"."""
    from haemolynx.gui._widget import settings_widget

    panel = settings_widget(napari_viewer=make_napari_viewer())
    panel._haemolynx_run_state.start(worker=Quittable())

    panel._haemolynx_run()

    assert panel._haemolynx_report() == ALREADY_RUNNING


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
    # Direct, not cycle: a categorical colouring is now looked up per item and
    # written as an array, so `CategoricalColormap.map` -- the thing that
    # raised on NaN -- is never reached at all. This sequence is now safe by
    # construction rather than by ordering.
    assert layer.edge_color_mode == "direct"

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
    assert layer.edge_color_mode == "direct"      # see the note above
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


def test_a_text_column_is_recognised_by_its_data_not_its_name(make_napari_viewer):
    """Choosing "role" raised `could not convert string to float: 'inlet'`.

    Text-or-number was decided by membership of `TEXT_COLUMNS`, which names the
    text columns the results module happens to write. `role`, on the boundary
    nodes, is not one of them, so it was taken for a quantity and its labels
    were handed to a colormap. Any column any layer carries has a dtype, and
    that is what is asked now.
    """
    from haemolynx.gui._widget import (
        _active_column, _is_text_column, _layer_controls,
    )

    viewer, _vessels, _scale = a_drawn_run(make_napari_viewer)
    results = ResultLayers()
    results.stage_finished("build_network", network(a_graph()))
    _apply_layers(viewer, results.stage_finished("assign_boundaries", SimpleNamespace(
        inlet_nodes=[0], outlet_nodes=[3], arteriole_boundary_nodes=[1],
        venule_boundary_nodes=[2], resistance_node_pair=None)))

    layer = viewer.layers[BOUNDARY_NODES]
    controls = _layer_controls(viewer, layer)
    assert _is_text_column(layer, "role") is True

    controls._haemolynx_feature.native.setCurrentText("role")

    assert _active_column(layer) == "role"
    assert len(np.unique(np.asarray(layer.face_color), axis=0)) > 1
    # A cycle of labels has no range, so there is nothing to draw a bar for.
    assert controls._haemolynx_scale.shown is False


def test_a_text_column_has_no_range(make_napari_viewer):
    from haemolynx.gui._widget import _data_range

    _viewer, layer, _scale = a_drawn_run(make_napari_viewer, branch_order="BO1")
    assert _data_range(layer, "branch_order") is None
    assert _data_range(layer, "length") is not None
