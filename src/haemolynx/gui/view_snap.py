"""Snapping the viewer to look straight at one of the volume's planes.

Pure: a plane name in, the napari dims order (2D display) or camera
directions (3D display) out. ``_widget.py`` applies them to a viewer and puts
the XY / XZ / YZ buttons on the canvas.

Arrays are canonical ``(z, y, x)`` (see ``io/axis_order.py``), so the plane
names read as the two axes left on screen: XY looks down z, XZ down y and YZ
down x. Each keeps napari's own default for XY -- the first on-screen axis
increasing downwards -- so XZ and YZ show z increasing down the screen, the
way a stack is read from its top slice.
"""
from __future__ import annotations

#: The planes a view can snap to, in button order.
VIEW_PLANES: tuple[str, ...] = ("XY", "XZ", "YZ")

#: Canonical ``(z, y, x)`` index of the axis each plane looks along.
_VIEW_AXIS = {"XY": 0, "XZ": 1, "YZ": 2}

#: ``(vertical, horizontal)`` on-screen axes of each plane, as ``(z, y, x)``
#: indices -- napari draws the last two displayed dims as rows then columns.
_SCREEN_AXES = {"XY": (1, 2), "XZ": (0, 2), "YZ": (0, 1)}


def _checked(plane: str) -> str:
    key = str(plane).upper()
    if key not in _VIEW_AXIS:
        raise ValueError(f"view plane must be one of {VIEW_PLANES}, got {plane!r}")
    return key


def dims_order_for_plane(plane: str, ndim: int = 3) -> tuple[int, ...]:
    """The ``viewer.dims.order`` that shows *plane* in a 2D view.

    The two plane axes go last (rows, then columns); the axis looked along
    and any leading non-spatial axes stay sliders in front of them.
    """
    key = _checked(plane)
    if ndim < 3:
        raise ValueError(f"snapping to a plane needs a 3D volume, got {ndim} dims")
    offset = ndim - 3
    vertical, horizontal = (offset + axis for axis in _SCREEN_AXES[key])
    sliced = offset + _VIEW_AXIS[key]
    return (*range(offset), sliced, vertical, horizontal)


def camera_directions_for_plane(
    plane: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(view_direction, up_direction)`` in displayed ``(z, y, x)`` order for
    napari's ``Camera.set_view_direction`` in a 3D view of *plane*.

    The camera looks along the plane's missing axis with the plane's
    vertical axis increasing down the screen and its horizontal axis
    increasing to the right, as napari's default XY view does. napari's
    screen right is ``view x up``, so the view is ``up x right``: picking the
    sign of the view by hand instead mirrors XZ left-to-right.
    """
    key = _checked(plane)
    vertical, horizontal = _SCREEN_AXES[key]
    up = [0.0, 0.0, 0.0]
    up[vertical] = -1.0
    right = [0.0, 0.0, 0.0]
    right[horizontal] = 1.0
    view = (
        up[1] * right[2] - up[2] * right[1],
        up[2] * right[0] - up[0] * right[2],
        up[0] * right[1] - up[1] * right[0],
    )
    return tuple(float(value) + 0.0 for value in view), tuple(up)
