"""The tubes layer's "render quality" slider, in a real viewer.

Level 0 is the original drawing: separate six-sided prisms, flat shaded,
which read as bands. Each step right redraws one continuous, capped,
smooth-shaded tube per vessel with a rounder cross-section.
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui import _widget as widget_mod  # noqa: E402
from haemolynx.gui._widget import _apply_layers, _layer_controls, settings_widget  # noqa: E402
from haemolynx.gui.results import VESSEL_TUBES, VESSELS  # noqa: E402
from haemolynx.gui.vessel_tubes import TUBE_QUALITY_SIDES, TUBE_SHADING  # noqa: E402
from test_gui_results_widget import a_run  # noqa: E402

pytestmark = pytest.mark.gui

TOP = len(TUBE_QUALITY_SIDES) - 1


@pytest.fixture(autouse=True)
def _original_quality():
    """The level is the session's; put it back so other tests see level 0."""
    widget_mod._tube_quality = 0
    yield
    widget_mod._tube_quality = 0


@pytest.fixture
def run(make_napari_viewer):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    for group in a_run():
        _apply_layers(viewer, group)
    return panel, viewer


def _slider(viewer, name=VESSEL_TUBES):
    controls = _layer_controls(viewer, viewer.layers[name])
    return controls._haemolynx_tube_quality


def _mesh(viewer, name=VESSEL_TUBES):
    vertices, faces = viewer.layers[name].data[:2]
    return np.asarray(vertices).copy(), np.asarray(faces).copy()


def test_the_slider_is_on_the_tubes_layer_controls_and_starts_at_the_original(run):
    _panel, viewer = run
    slider = _slider(viewer)

    assert (slider.minimum(), slider.maximum()) == (0, TOP)
    assert slider.value() == 0
    assert slider.toolTip().strip()
    assert viewer.layers[VESSEL_TUBES].shading == TUBE_SHADING


def test_the_top_of_the_slider_redraws_joined_smooth_tubes(run):
    _panel, viewer = run
    before_vertices, _before_faces = _mesh(viewer)

    _slider(viewer).setValue(TOP)

    vertices, faces = _mesh(viewer)
    assert len(vertices) > len(before_vertices)
    assert faces.max() < len(vertices)
    assert viewer.layers[VESSEL_TUBES].shading == "smooth"
    # Still coloured vertex by vertex from the vessels' own colours.
    colours = np.asarray(viewer.layers[VESSEL_TUBES].vertex_colors)
    assert colours.shape[0] == len(vertices)


def test_back_to_the_start_restores_the_original_drawing(run):
    _panel, viewer = run
    original = _mesh(viewer)
    slider = _slider(viewer)

    slider.setValue(TOP)
    slider.setValue(0)

    for got, want in zip(_mesh(viewer), original):
        np.testing.assert_array_equal(got, want)
    assert viewer.layers[VESSEL_TUBES].shading == TUBE_SHADING


def test_the_quality_holds_through_a_redraw_and_the_lines_toggle(run):
    panel, viewer = run
    _slider(viewer).setValue(TOP)
    top_mesh = _mesh(viewer)

    for group in a_run():
        _apply_layers(viewer, group)
    panel._haemolynx_vessel_draw.lines.click()
    panel._haemolynx_vessel_draw.tubes.click()

    assert _slider(viewer).value() == TOP
    for got, want in zip(_mesh(viewer), top_mesh):
        np.testing.assert_array_equal(got, want)


def test_every_tubes_layer_shares_the_quality(run):
    """A perturbation's tubes follow the same slider, and show it moved."""
    from haemolynx.gui.results import perturbation_layer_names
    from haemolynx.gui.vessel_tubes import vessel_tubes_layer_name
    from test_gui_results import a_perturbation, a_perturbation_run, built

    panel, viewer = run
    _apply_layers(viewer, built().stage_finished(
        "run_perturbations", a_perturbation_run(a_perturbation("dilate"))))
    panel._haemolynx_choose_layer_set("dilate")
    tubes = vessel_tubes_layer_name(perturbation_layer_names("dilate")[0])
    assert _slider(viewer, tubes).value() == 0

    _slider(viewer, tubes).setValue(TOP)
    panel._haemolynx_choose_layer_set(None)

    assert _slider(viewer).value() == TOP
    assert viewer.layers[VESSEL_TUBES].shading == "smooth"
