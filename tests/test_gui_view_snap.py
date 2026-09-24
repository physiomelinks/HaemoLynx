"""The plane maths behind the XY / XZ / YZ view-snap buttons (pure, no napari)."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.gui.view_snap import (
    VIEW_PLANES,
    camera_directions_for_plane,
    dims_order_for_plane,
)


def test_the_buttons_are_the_three_planes_in_reading_order():
    assert VIEW_PLANES == ("XY", "XZ", "YZ")


@pytest.mark.parametrize(
    ("plane", "order"),
    [("XY", (0, 1, 2)), ("XZ", (1, 0, 2)), ("YZ", (2, 0, 1))],
)
def test_a_2d_view_puts_the_planes_two_axes_on_screen(plane, order):
    """napari shows the last two dims as rows then columns; the first is the
    slider through the axis the plane looks along."""
    assert dims_order_for_plane(plane) == order


def test_leading_non_spatial_axes_stay_in_front():
    assert dims_order_for_plane("XZ", ndim=4) == (0, 2, 1, 3)


@pytest.mark.parametrize(
    ("plane", "view", "up"),
    [
        # napari's own default 3D view is exactly this XY.
        ("XY", (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ("XZ", (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        ("YZ", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
    ],
)
def test_a_3d_view_looks_along_the_missing_axis(plane, view, up):
    got_view, got_up = camera_directions_for_plane(plane)
    assert got_view == view
    assert got_up == up


@pytest.mark.parametrize("plane", VIEW_PLANES)
def test_every_plane_keeps_the_default_views_handedness(plane):
    """napari's screen right is view x up. It must be +1 along the plane's
    horizontal axis, as +x is in the default view -- a view direction with
    the other sign would show the plane mirrored."""
    view, up = camera_directions_for_plane(plane)
    right = np.cross(view, up)
    horizontal = {"XY": 2, "XZ": 2, "YZ": 1}[plane]
    expected = np.zeros(3)
    expected[horizontal] = 1.0
    np.testing.assert_array_equal(right, expected)


def test_plane_names_are_case_insensitive_and_unknown_ones_refused():
    assert dims_order_for_plane("xz") == dims_order_for_plane("XZ")
    with pytest.raises(ValueError, match="XY"):
        camera_directions_for_plane("ZX")
    with pytest.raises(ValueError, match="3D volume"):
        dims_order_for_plane("XY", ndim=2)
