"""Unsolved vessels on screen: light grey under every flow-based colouring.

napari's colormaps draw NaN transparent, so a vessel whose flow the solve
could not give (blanked by ``results.mask_unsolved_flow_columns``) used to
vanish. The panel colours through a copy of each map whose NaN colour is the
uncoloured grey. The pure side is in ``test_gui_unsolved_vessels.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import (  # noqa: E402
    GREY_NAN_SUFFIX,
    UNCOLOURED_RGBA,
    _apply_colormap,
    _apply_layers,
    _base_colormap_name,
    _colour_by_columns,
    _colour_layer,
    _colormap_name,
    _flow_dir_rgba,
    _grey_nan_colormap,
)
from haemolynx.gui.results import FLOW_SOLUTION, VESSELS  # noqa: E402
from test_gui_unsolved_vessels import _graph, _solve_layers  # noqa: E402

pytestmark = pytest.mark.gui

GREY = np.asarray(UNCOLOURED_RGBA)


def _a_vectors_layer():
    """Three one-segment vessels, the last unsolved, as the panel draws them."""
    data = np.zeros((3, 2, 3))
    data[:, 0, 0] = [0.0, 10.0, 20.0]
    data[:, 1, 0] = 10.0
    return napari.layers.Vectors(
        data,
        features={
            "flow_abs": [1e-12, 1e-11, np.nan],
            "flow_dir_z": [1.0, -1.0, np.nan],
            "flow_dir_y": [0.0, 0.0, np.nan],
            "flow_dir_x": [0.0, 0.0, np.nan],
            FLOW_SOLUTION: ["Solved", "Solved", "Unsolved"],
        },
    )


def test_the_grey_nan_copy_is_the_same_map_with_a_grey_nan():
    from napari.utils.colormaps import ensure_colormap

    greyed = _grey_nan_colormap("viridis")
    values = np.array([0.0, 0.5, 1.0])

    assert greyed.name == "viridis" + GREY_NAN_SUFFIX
    np.testing.assert_allclose(greyed.map(values), ensure_colormap("viridis").map(values))
    np.testing.assert_allclose(greyed.map(np.array([np.nan]))[0], GREY)
    assert _base_colormap_name(greyed.name) == "viridis"
    # Copying a copy gives the copy, not a copy of it.
    assert _grey_nan_colormap(greyed.name).name == greyed.name
    assert _grey_nan_colormap("not a colormap") == "not a colormap"


def test_a_flow_colouring_draws_the_unsolved_vessel_light_grey():
    layer = _a_vectors_layer()

    _colour_layer(layer, "flow_abs", "continuous")

    colours = np.asarray(layer.edge_color)
    np.testing.assert_allclose(colours[2], GREY)
    assert not np.allclose(colours[0], GREY) and not np.allclose(colours[1], GREY)
    assert _colormap_name(layer) == "viridis"


def test_choosing_another_map_keeps_the_unsolved_vessel_grey():
    layer = _a_vectors_layer()
    _colour_layer(layer, "flow_abs", "continuous")
    before = np.array(layer.edge_color, copy=True)

    assert _apply_colormap(layer, "plasma")

    colours = np.asarray(layer.edge_color)
    assert _colormap_name(layer) == "plasma"
    assert not np.allclose(colours[:2], before[:2])
    np.testing.assert_allclose(colours[2], GREY)


def test_the_direction_map_draws_the_unsolved_vessel_the_same_grey():
    rgba = _flow_dir_rgba(_a_vectors_layer())

    np.testing.assert_allclose(rgba[2], GREY)
    np.testing.assert_allclose(rgba[0], [0.5, 0.5, 1.0, 1.0])


def test_solved_or_not_is_not_offered_as_a_colouring():
    assert FLOW_SOLUTION not in _colour_by_columns(_a_vectors_layer())


def test_a_solved_run_draws_its_unsolved_vessel_grey(make_napari_viewer):
    viewer = make_napari_viewer()
    _apply_layers(viewer, _solve_layers(_graph()))
    vessels = viewer.layers[VESSELS]

    assert list(vessels.features[FLOW_SOLUTION]) == ["Solved", "Solved", "Unsolved"]
    colours = np.asarray(vessels.edge_color)
    np.testing.assert_allclose(colours[2], GREY)
    assert not np.allclose(colours[0], GREY)
