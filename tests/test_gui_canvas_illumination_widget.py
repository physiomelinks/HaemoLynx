"""The view window's Illumination box, on a real canvas."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")

from qtpy.QtWidgets import QApplication  # noqa: E402

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.canvas_illumination import DEFAULT_AMBIENT  # noqa: E402

pytestmark = pytest.mark.gui


def _triangle():
    return (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        np.array([[0, 1, 2]]),
    )


def _light_alpha(viewer, layer) -> float:
    filt = viewer.window._qt_viewer.layer_to_visual[layer].node.shading_filter
    assert filt is not None
    return float(np.asarray(filt.ambient_light.rgba).reshape(-1)[-1])


def test_the_box_starts_at_the_canvas_defaults_without_changing_a_surface(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_illumination
    assert box.state().shading == "smooth"
    assert box.state().ambient == pytest.approx(DEFAULT_AMBIENT)
    assert box.adjusted is False
    assert box.ambient.isEnabled()

    layer = viewer.add_surface(_triangle(), shading="flat")
    QApplication.processEvents()
    assert layer.shading == "flat"
    assert box.adjusted is False


def test_moving_the_lamp_lights_every_surface_and_one_added_later(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    panel = settings_widget(napari_viewer=viewer)
    box = panel._haemolynx_illumination
    first = viewer.add_surface(_triangle(), shading="flat")
    second = viewer.add_surface(_triangle(), shading="flat")

    box.ambient.setValue(0.55)
    QApplication.processEvents()

    assert box.adjusted is True
    assert first.shading == "smooth"
    assert second.shading == "smooth"
    assert _light_alpha(viewer, first) == pytest.approx(0.55)
    assert _light_alpha(viewer, second) == pytest.approx(0.55)

    box.shading.setCurrentIndex(box.shading.findData("none"))
    QApplication.processEvents()
    assert first.shading == "none"
    assert not box.ambient.isEnabled()

    box.shading.setCurrentIndex(box.shading.findData("flat"))
    QApplication.processEvents()
    later = viewer.add_surface(_triangle(), shading="smooth")
    QApplication.processEvents()
    assert later.shading == "flat"
    assert first.shading == "flat"
    assert box.diffuse.isEnabled()
