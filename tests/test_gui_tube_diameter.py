"""The "Tube diameter" choice, in a real viewer: on the view panel and on the
tubes layer's own controls.

Per vessel draws each tube at its vessel's own diameter, as before; Uniform
draws every one at the width in the µm box beside it. Display only: the
vessels keep theirs.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import _apply_layers, _layer_controls, settings_widget  # noqa: E402
from haemolynx.gui.results import VESSEL_TUBES, VESSELS, ResultLayers  # noqa: E402
from haemolynx.gui.vessel_tubes import (  # noqa: E402
    DEFAULT_TUBE_DIAMETER,
    DEFAULT_UNIFORM_TUBE_DIAMETER_UM,
    DEFAULT_VESSEL_DRAW,
    TUBE_DIAMETER_LABELS,
    TUBE_DIAMETER_PER_VESSEL,
    TUBE_DIAMETER_UNIFORM,
    TUBE_RADIUS_UM,
    vessel_tube_mesh,
)
from test_gui_results import a_graph, network  # noqa: E402

pytestmark = pytest.mark.gui

#: `a_graph`'s three vessels, along z, each its own width.
DIAMETERS_UM = (3.0, 6.0, 12.0)


@pytest.fixture(autouse=True)
def _original_choice():
    """The choices are the session's; put them back so other tests see the defaults."""
    widget_mod._tube_diameter = DEFAULT_TUBE_DIAMETER
    widget_mod._tube_uniform_diameter_um = DEFAULT_UNIFORM_TUBE_DIAMETER_UM
    widget_mod._vessel_draw_mode = DEFAULT_VESSEL_DRAW
    yield
    widget_mod._tube_diameter = DEFAULT_TUBE_DIAMETER
    widget_mod._tube_uniform_diameter_um = DEFAULT_UNIFORM_TUBE_DIAMETER_UM
    widget_mod._vessel_draw_mode = DEFAULT_VESSEL_DRAW


def a_graph_with_diameters():
    graph = a_graph(diameter_source="measured", resistance=1e15, conductance=1e-15,
                    flow_abs=1e-12, flow_signed=1e-12)
    for (u, v, key), diameter in zip(graph.edges(keys=True), DIAMETERS_UM):
        graph.edges[u, v, key]["diameter_um"] = diameter
    return graph


def a_run_with_diameters():
    """The groups a run gives as far as the Diameters stage.

    The image spans the vessels' 30 µm in z: the view clips every layer to
    the image's depth, which would leave only the first vessel's tube.
    """
    results = ResultLayers()
    graph = a_graph_with_diameters()
    shape = (32, 4, 4)
    return [
        results.stage_finished("skeletonise", SimpleNamespace(
            image=np.zeros(shape, dtype=np.uint8),
            skeleton=np.zeros(shape, dtype=bool),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
        )),
        results.stage_finished("build_network", network(graph)),
        results.stage_finished("assign_diameters", SimpleNamespace(graph=graph)),
    ]


@pytest.fixture
def run(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run_with_diameters():
        _apply_layers(viewer, group)
    return panel, viewer


def _combo(viewer, name=VESSEL_TUBES):
    controls = _layer_controls(viewer, viewer.layers[name])
    return controls._haemolynx_tube_diameter


def _choose_on_layer(viewer, choice, name=VESSEL_TUBES):
    combo = _combo(viewer, name)
    combo.setCurrentIndex(combo.findData(choice))


def _layer_width(viewer, name=VESSEL_TUBES):
    """The uniform-width box on a tubes layer's controls, and its label."""
    controls = _layer_controls(viewer, viewer.layers[name])
    return controls._haemolynx_tube_uniform_width, controls._haemolynx_tube_uniform_width_label


def _pick(panel, choice):
    """Choose on the view panel the way a click on its menu does."""
    menu = panel._haemolynx_tube_diameter_menu
    (action,) = [action for action in menu.actions() if action.data() == choice]
    action.trigger()


def _ticked(panel):
    menu = panel._haemolynx_tube_diameter_menu
    return [action.data() for action in menu.actions() if action.isChecked()]


def _widest(viewer, name=VESSEL_TUBES) -> float:
    """How far the tubes reach from the line their vessels run along (z)."""
    vertices = np.asarray(viewer.layers[name].data[0], dtype=float)
    return float(np.linalg.norm(vertices[:, 1:], axis=1).max())


def _mesh(viewer, name=VESSEL_TUBES):
    vertices, faces = viewer.layers[name].data[:2]
    return np.asarray(vertices).copy(), np.asarray(faces).copy()


def test_the_choice_is_on_the_view_panel_and_the_tubes_layer_and_starts_per_vessel(run):
    from haemolynx.gui.chrome_tooltips import TUBE_DIAMETER_TOOLTIP

    panel, viewer = run
    row = panel._haemolynx_tube_diameter_row
    display = panel._haemolynx_display_group
    assert row.parentWidget() is display
    # Under the Tubes/Lines row it qualifies.
    form = display.layout()
    assert form.indexOf(row) == form.indexOf(panel._haemolynx_vessel_draw_row) + 1
    button = panel._haemolynx_tube_diameter
    menu = panel._haemolynx_tube_diameter_menu
    assert [action.text() for action in menu.actions()] == ["Per vessel", "Uniform"]
    assert button.text() == TUBE_DIAMETER_LABELS[TUBE_DIAMETER_PER_VESSEL]
    assert _ticked(panel) == [TUBE_DIAMETER_PER_VESSEL]
    assert button.isEnabled()
    assert button.toolTip() == TUBE_DIAMETER_TOOLTIP

    combo = _combo(viewer)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Per vessel", "Uniform"]
    assert combo.currentData() == TUBE_DIAMETER_PER_VESSEL
    assert combo.toolTip() == TUBE_DIAMETER_TOOLTIP
    # Each vessel at its own diameter: half the widest one's, at the widest.
    assert _widest(viewer) == pytest.approx(max(DIAMETERS_UM) / 2)


def test_uniform_on_the_view_panel_draws_every_tube_at_one_width(run):
    panel, viewer = run
    vessels, tubes = viewer.layers[VESSELS], viewer.layers[VESSEL_TUBES]
    diameters = np.asarray(vessels.features["diameter_um"], dtype=float).copy()
    assert sorted(set(diameters.tolist())) == list(DIAMETERS_UM)

    _pick(panel, TUBE_DIAMETER_UNIFORM)

    assert _widest(viewer) == pytest.approx(TUBE_RADIUS_UM)
    want, _faces, _index = vessel_tube_mesh(
        vessels.data, vessels.features, quality=widget_mod._tube_quality,
        edge_width=vessels.edge_width, diameter=TUBE_DIAMETER_UNIFORM,
    )
    np.testing.assert_allclose(tubes.data[0], want, atol=1e-6)
    # Still coloured vertex by vertex from the vessels' own colours.
    assert np.asarray(tubes.vertex_colors).shape[0] == len(want)
    assert tubes.visible and not vessels.visible
    assert panel._haemolynx_tube_diameter.text() == "Uniform"
    assert _ticked(panel) == [TUBE_DIAMETER_UNIFORM]
    assert _combo(viewer).currentData() == TUBE_DIAMETER_UNIFORM
    # Drawing only: the vessels keep their diameters.
    np.testing.assert_array_equal(vessels.features["diameter_um"], diameters)


def test_the_tubes_layer_controls_choose_it_too_and_the_view_panel_follows(run):
    panel, viewer = run

    _choose_on_layer(viewer, TUBE_DIAMETER_UNIFORM)

    assert _widest(viewer) == pytest.approx(TUBE_RADIUS_UM)
    assert panel._haemolynx_tube_diameter.text() == "Uniform"
    assert _ticked(panel) == [TUBE_DIAMETER_UNIFORM]


def test_back_to_per_vessel_restores_the_drawing_there_was(run):
    panel, viewer = run
    original = _mesh(viewer)

    _pick(panel, TUBE_DIAMETER_UNIFORM)
    assert _widest(viewer) == pytest.approx(TUBE_RADIUS_UM)
    _pick(panel, TUBE_DIAMETER_PER_VESSEL)

    for got, want in zip(_mesh(viewer), original):
        np.testing.assert_array_equal(got, want)
    assert _combo(viewer).currentData() == TUBE_DIAMETER_PER_VESSEL
    assert _ticked(panel) == [TUBE_DIAMETER_PER_VESSEL]


def test_picking_the_ticked_entry_again_keeps_it_ticked(run):
    """A checkable action unticks itself when clicked; the choice stays."""
    panel, viewer = run
    _pick(panel, TUBE_DIAMETER_UNIFORM)

    _pick(panel, TUBE_DIAMETER_UNIFORM)

    assert _ticked(panel) == [TUBE_DIAMETER_UNIFORM]
    assert _widest(viewer) == pytest.approx(TUBE_RADIUS_UM)


def test_the_choice_holds_through_a_redraw_and_the_lines_toggle(run):
    panel, viewer = run
    _pick(panel, TUBE_DIAMETER_UNIFORM)
    uniform = _mesh(viewer)

    panel._haemolynx_vessel_draw.lines.click()
    # Lines have no diameter to choose.
    assert not panel._haemolynx_tube_diameter.isEnabled()
    assert not panel._haemolynx_tube_uniform_width.isEnabled()
    panel._haemolynx_vessel_draw.tubes.click()
    assert panel._haemolynx_tube_diameter.isEnabled()
    assert panel._haemolynx_tube_uniform_width.isEnabled()
    for group in a_run_with_diameters():
        _apply_layers(viewer, group)

    assert _combo(viewer).currentData() == TUBE_DIAMETER_UNIFORM
    assert _ticked(panel) == [TUBE_DIAMETER_UNIFORM]
    for got, want in zip(_mesh(viewer), uniform):
        np.testing.assert_array_equal(got, want)


def test_every_tubes_layer_shares_the_choice(run):
    """A perturbation's tubes follow the same choice, and show it made."""
    from haemolynx.gui.results import perturbation_layer_names
    from haemolynx.gui.vessel_tubes import vessel_tubes_layer_name
    from haemolynx.pipeline import PerturbationResult
    from test_gui_results import a_perturbation_run, built

    panel, viewer = run
    graph = a_graph_with_diameters()
    dilate = PerturbationResult(name="dilate", type="arteriole_diameter_change", graph=graph)
    _apply_layers(viewer, built(graph).stage_finished(
        "run_perturbations", a_perturbation_run(dilate)))
    panel._haemolynx_choose_layer_set("dilate")
    tubes = vessel_tubes_layer_name(perturbation_layer_names("dilate")[0])
    assert _combo(viewer, tubes).currentData() == TUBE_DIAMETER_PER_VESSEL
    assert _widest(viewer, tubes) == pytest.approx(max(DIAMETERS_UM) / 2)

    _choose_on_layer(viewer, TUBE_DIAMETER_UNIFORM, tubes)
    assert _widest(viewer, tubes) == pytest.approx(TUBE_RADIUS_UM)
    panel._haemolynx_choose_layer_set(None)

    assert _combo(viewer).currentData() == TUBE_DIAMETER_UNIFORM
    assert _ticked(panel) == [TUBE_DIAMETER_UNIFORM]
    assert _widest(viewer) == pytest.approx(TUBE_RADIUS_UM)


def test_the_width_box_shows_only_while_uniform_and_starts_at_four_microns(run):
    from haemolynx.gui.chrome_tooltips import TUBE_UNIFORM_WIDTH_TOOLTIP

    panel, viewer = run
    box = panel._haemolynx_tube_uniform_width
    layer_box, layer_label = _layer_width(viewer)
    assert box.isHidden() and layer_box.isHidden() and layer_label.isHidden()
    for widget in (box, layer_box):
        assert widget.value() == DEFAULT_UNIFORM_TUBE_DIAMETER_UM == 4.0
        assert widget.suffix() == " µm"
        assert widget.toolTip() == TUBE_UNIFORM_WIDTH_TOOLTIP
        # Applied on Enter or on leaving the box, not on every key typed.
        assert not widget.keyboardTracking()
    # Beside the choice it is the width for.
    assert box.parentWidget() is panel._haemolynx_tube_diameter.parentWidget()

    _pick(panel, TUBE_DIAMETER_UNIFORM)
    assert not box.isHidden()
    assert not layer_box.isHidden() and not layer_label.isHidden()

    _pick(panel, TUBE_DIAMETER_PER_VESSEL)
    assert box.isHidden() and layer_box.isHidden() and layer_label.isHidden()


def test_a_width_on_the_view_panel_redraws_every_tube_that_wide(run):
    panel, viewer = run
    vessels = viewer.layers[VESSELS]
    diameters = np.asarray(vessels.features["diameter_um"], dtype=float).copy()
    _pick(panel, TUBE_DIAMETER_UNIFORM)

    panel._haemolynx_tube_uniform_width.setValue(20.0)

    assert _widest(viewer) == pytest.approx(10.0)
    want, _faces, _index = vessel_tube_mesh(
        vessels.data, vessels.features, quality=widget_mod._tube_quality,
        edge_width=vessels.edge_width, diameter=TUBE_DIAMETER_UNIFORM,
        uniform_diameter_um=20.0,
    )
    np.testing.assert_allclose(viewer.layers[VESSEL_TUBES].data[0], want, atol=1e-6)
    assert _layer_width(viewer)[0].value() == 20.0
    # Drawing only: the vessels keep their diameters.
    np.testing.assert_array_equal(vessels.features["diameter_um"], diameters)


def test_a_width_on_the_tubes_layer_controls_moves_the_view_panel(run):
    panel, viewer = run
    _choose_on_layer(viewer, TUBE_DIAMETER_UNIFORM)

    _layer_width(viewer)[0].setValue(1.5)

    assert _widest(viewer) == pytest.approx(0.75)
    assert panel._haemolynx_tube_uniform_width.value() == 1.5


def test_the_width_is_kept_through_per_vessel_and_back(run):
    panel, viewer = run
    _pick(panel, TUBE_DIAMETER_UNIFORM)
    panel._haemolynx_tube_uniform_width.setValue(20.0)
    wide = _mesh(viewer)

    _pick(panel, TUBE_DIAMETER_PER_VESSEL)
    assert _widest(viewer) == pytest.approx(max(DIAMETERS_UM) / 2)
    _pick(panel, TUBE_DIAMETER_UNIFORM)

    assert panel._haemolynx_tube_uniform_width.value() == 20.0
    for got, want in zip(_mesh(viewer), wide):
        np.testing.assert_array_equal(got, want)
