"""The view panel's "Showing" menu, in a real viewer: baseline or a perturbation.

A perturbation's layers lie exactly on the baseline's, so the menu swaps
whole networks -- vessels (as tubes or lines), nodes and flow direction --
and only appears once there is a perturbation to swap to.
"""
from __future__ import annotations

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import _apply_layers, settings_widget  # noqa: E402
from haemolynx.gui.results import (  # noqa: E402
    NODES,
    VESSEL_TUBES,
    VESSELS,
    perturbation_layer_names,
)
from haemolynx.gui.vessel_tubes import vessel_tubes_layer_name  # noqa: E402
from test_gui_results import a_perturbation, a_perturbation_run, built  # noqa: E402
from test_gui_results_widget import a_run  # noqa: E402

pytestmark = pytest.mark.gui

DILATE_VESSELS, DILATE_NODES = perturbation_layer_names("dilate")
DILATE_TUBES = vessel_tubes_layer_name(DILATE_VESSELS)
BLOCK_VESSELS, BLOCK_NODES = perturbation_layer_names("block")


def _perturbations(*names):
    return built().stage_finished(
        "run_perturbations", a_perturbation_run(*(a_perturbation(name) for name in names))
    )


@pytest.fixture
def run(make_napari_viewer):
    """A panel over a baseline network, with no perturbations yet."""
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    return panel, viewer


def _labels(panel):
    return [action.text() for action in panel._haemolynx_layer_set_menu.actions()]


def _checked(panel):
    return [a.text() for a in panel._haemolynx_layer_set_menu.actions() if a.isChecked()]


def _choose(panel, label):
    """Pick an entry the way a click on the menu does."""
    (action,) = [a for a in panel._haemolynx_layer_set_menu.actions() if a.text() == label]
    action.trigger()


def _drawn(viewer, name):
    """Whether a network's vessels are on screen, as tubes or as lines."""
    tubes = vessel_tubes_layer_name(name)
    return bool(viewer.layers[name].visible) or (
        tubes in viewer.layers and bool(viewer.layers[tubes].visible)
    )


def test_the_menu_is_hidden_until_a_perturbation_has_layers(run):
    panel, viewer = run
    assert panel._haemolynx_layer_set_row.isHidden()

    _apply_layers(viewer, _perturbations("dilate", "block"))

    assert not panel._haemolynx_layer_set_row.isHidden()
    assert _labels(panel) == ["Baseline", "dilate", "block"]
    assert _checked(panel) == ["Baseline"]
    assert panel._haemolynx_layer_set_button.text() == "Baseline"


def test_choosing_a_perturbation_swaps_the_whole_network(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate", "block"))
    viewer.layers[NODES].visible = True

    _choose(panel, "dilate")

    assert _drawn(viewer, DILATE_VESSELS)
    assert not _drawn(viewer, VESSELS)
    assert not _drawn(viewer, BLOCK_VESSELS)
    # Nodes were on for the baseline, so they are on for the perturbation.
    assert viewer.layers[DILATE_NODES].visible
    assert not viewer.layers[NODES].visible
    assert not viewer.layers[BLOCK_NODES].visible
    assert _checked(panel) == ["dilate"]
    assert panel._haemolynx_layer_set_button.text() == "dilate"


def test_a_perturbation_is_drawn_as_tubes_like_the_baseline(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    assert viewer.layers[VESSEL_TUBES].visible  # the default drawing

    _choose(panel, "dilate")

    assert DILATE_TUBES in viewer.layers
    assert viewer.layers[DILATE_TUBES].visible
    assert not viewer.layers[DILATE_VESSELS].visible
    assert not viewer.layers[VESSEL_TUBES].visible


def test_switching_back_restores_the_baseline(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    viewer.layers[NODES].visible = False

    _choose(panel, "dilate")
    viewer.layers[DILATE_NODES].visible = True
    _choose(panel, "Baseline")

    assert _drawn(viewer, VESSELS)
    assert viewer.layers[NODES].visible  # carried back from the perturbation
    assert not _drawn(viewer, DILATE_VESSELS)
    assert not viewer.layers[DILATE_NODES].visible


def test_tubes_and_lines_follow_the_network_shown(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    _choose(panel, "dilate")
    draw = panel._haemolynx_vessel_draw

    draw.lines.click()
    assert viewer.layers[DILATE_VESSELS].visible
    assert not _drawn(viewer, VESSELS)

    draw.tubes.click()
    assert viewer.layers[DILATE_TUBES].visible
    assert not _drawn(viewer, VESSELS)


def test_a_redraw_keeps_the_chosen_perturbation_shown(run):
    """Every perturbation layer is drawn hidden; a re-run must not undo the choice."""
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    _choose(panel, "dilate")

    _apply_layers(viewer, _perturbations("dilate"))
    for group in a_run():
        _apply_layers(viewer, group)

    assert _drawn(viewer, DILATE_VESSELS)
    assert not _drawn(viewer, VESSELS)
    assert _checked(panel) == ["dilate"]


def test_when_the_shown_perturbation_goes_the_baseline_comes_back(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate", "block"))
    _choose(panel, "dilate")

    for name in [layer.name for layer in viewer.layers if "dilate" in layer.name]:
        viewer.layers.remove(name)
    # The menu stays while it would be the only way back ...
    assert _labels(panel) == ["Baseline", "block"]
    # ... and the next redraw puts the baseline back on its own.
    panel._haemolynx_after_layers_applied()

    assert _drawn(viewer, VESSELS)
    assert _checked(panel) == ["Baseline"]


def test_clear_forgets_the_choice_and_hides_the_menu(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    _choose(panel, "dilate")

    panel._haemolynx_clear()

    assert panel._haemolynx_layer_set_row.isHidden()
    assert getattr(viewer, "_haemolynx_layer_set", None) is None


def test_a_z_depth_window_keeps_the_perturbation_shown(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))
    panel._haemolynx_after_layers_applied()
    _choose(panel, "dilate")

    low, high = panel._haemolynx_z_depth_slider.value()
    panel._haemolynx_z_depth_slider.setValue((low, low + 0.5 * (high - low)))

    assert _drawn(viewer, DILATE_VESSELS)
    assert not _drawn(viewer, VESSELS)
    assert _checked(panel) == ["dilate"]


# --- sweep sliders live in the view panel, for the sweep shown --------------


@pytest.fixture
def sweeps(run, tmp_path):
    """The baseline plus two real sweep perturbations, one slider each."""
    from haemolynx.gui.results import ResultLayers
    from haemolynx.pipeline import PerturbationRun
    from test_perturbation_stage import DILATION_SWEEP, PRESSURE_SWEEP, _run

    panel, viewer = run
    results = _run(tmp_path, [PRESSURE_SWEEP, DILATION_SWEEP]).results
    _apply_layers(viewer, ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(results=results, output_dir=tmp_path)))
    return panel, viewer, [result.name for result in results]


def _sweep_slider(panel, name):
    from qtpy.QtWidgets import QAbstractSlider

    _key, container = panel._haemolynx_sweep_controls[perturbation_layer_names(name)[0]]
    (slider,) = container.native.findChildren(QAbstractSlider)
    return container, slider


def test_sweep_sliders_are_in_the_view_panel_not_docks_of_their_own(sweeps):
    panel, viewer, names = sweeps

    docks = [name for name in viewer.window._wrapped_dock_widgets if name.endswith(" sweep")]
    assert docks == []
    assert set(panel._haemolynx_sweep_controls) == {
        perturbation_layer_names(name)[0] for name in names
    }
    view_panel = panel._haemolynx_view_panel
    for name in names:
        container, _slider = _sweep_slider(panel, name)
        assert view_panel.isAncestorOf(container.native)


def test_only_the_shown_sweep_s_sliders_are_shown(sweeps):
    panel, _viewer, (first, second) = sweeps
    group = panel._haemolynx_sweep_group
    assert group.isHidden()  # the baseline has no sweep

    _choose(panel, first)
    assert not group.isHidden()
    assert first in group.title()
    assert not _sweep_slider(panel, first)[0].native.isHidden()
    assert _sweep_slider(panel, second)[0].native.isHidden()

    _choose(panel, second)
    assert _sweep_slider(panel, first)[0].native.isHidden()
    assert not _sweep_slider(panel, second)[0].native.isHidden()

    _choose(panel, "Baseline")
    assert group.isHidden()


def test_a_single_re_solve_perturbation_shows_no_sweep_box(run):
    panel, viewer = run
    _apply_layers(viewer, _perturbations("dilate"))

    _choose(panel, "dilate")

    assert panel._haemolynx_sweep_group.isHidden()


def test_the_panel_slider_recolours_the_shown_sweep_s_tubes(sweeps):
    import numpy as np

    panel, viewer, (first, second) = sweeps
    _choose(panel, first)
    vessels = perturbation_layer_names(first)[0]
    tubes = vessel_tubes_layer_name(vessels)
    other = perturbation_layer_names(second)[0]
    flows = np.asarray(viewer.layers[vessels].features["flow_abs"], dtype=float).copy()
    colours = np.asarray(viewer.layers[tubes].vertex_colors, dtype=float).copy()
    other_flows = np.asarray(viewer.layers[other].features["flow_abs"], dtype=float).copy()

    _container, slider = _sweep_slider(panel, first)
    slider.setValue(slider.maximum())

    # Flows are ~1e-13, so np.allclose's default atol would call any two equal.
    now = np.asarray(viewer.layers[vessels].features["flow_abs"], dtype=float)
    assert not np.allclose(now, flows, rtol=1e-6, atol=0.0)
    assert not np.allclose(np.asarray(viewer.layers[tubes].vertex_colors, dtype=float), colours)
    np.testing.assert_array_equal(
        np.asarray(viewer.layers[other].features["flow_abs"], dtype=float), other_flows
    )


def test_each_sweep_keeps_its_grid_point_while_another_is_shown(sweeps):
    panel, _viewer, (first, second) = sweeps
    _choose(panel, first)
    _container, slider = _sweep_slider(panel, first)
    slider.setValue(slider.maximum())

    _choose(panel, second)
    _choose(panel, first)

    assert _sweep_slider(panel, first)[1].value() == slider.maximum()
