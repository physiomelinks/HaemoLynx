"""Boundary conditions as things you can point at.

The four boundary roles are configured by two settings each: a list of
coordinates and a list of volume boxes, both in physical ``(z, y, x)`` microns.
Typed into a text box they are unreadable, and nothing says whether they land on
a vessel -- the first sign that they do not is the run stopping with "No
inlet or outlet nodes found".

This turns those settings into two napari layers and back:

``HaemoLynx BC coordinates``
    A Points layer, one ring per coordinate. Its data *is* the setting: napari
    world coordinates are already microns, because graph layers carry
    ``scale=(1, 1, 1)`` and node ``pos`` is physical. So there is no conversion
    anywhere in this module, only bookkeeping.

``HaemoLynx BC <role> regions``
    A Shapes layer per role, one rectangle per volume box, drawn at the box's z
    centre with the box's y/x extent. The z extent it came from rides alongside
    as a ``depth`` feature, so every region keeps its own rather than sharing
    one. :func:`rectangle_from_box` and :func:`box_from_rectangle` are exact
    inverses, which is what lets the layer be treated as the setting. Nothing
    but those rectangles goes in it: it is the layer napari's own drawing tool
    works on, and anything else in it is something that tool can trip over.

``HaemoLynx BC <role> boxes``
    A Surface layer per role showing what the rectangles stand for: each box as
    a translucent solid in a shade of its own (:func:`box_mesh`), plus the band
    an ``edge_percent`` role selects from. Drawn from the settings only, so it
    is never edited and never read back, and it is the one that shows in 3D.

``HaemoLynx BC box nodes``
    A Points layer marking every node inside the box being edited on a role's
    page (:func:`nodes_in_box`): yellow and larger than the network's own
    dots, open ends (degree 1) larger still with a red rim. Drawn from the
    graph and the box, never read back; choosing one of those nodes for the
    role writes its ID (:func:`use_node_for_role`).

``HaemoLynx BC node IDs``
    A Points layer marking the nodes each ``node_ids`` role lists, at their
    positions in the run's graph. The IDs are the setting, not the positions,
    so this one is drawn from the settings and never read back: a node is
    added or removed by clicking it on the graph (:func:`toggle_node_id`).

The coordinates and regions layers are editable, and everything here is pure:
settings in, layer specs out, layer data in, settings out. Nothing imports
napari, so it is all testable without a display -- the same contract
:mod:`haemolynx.gui.results` keeps.

One hazard is worth naming, because it is silent. The settings rows are edited
by magicgui's ``LiteralEvalLineEdit``, which stores ``str(value)`` and reads it
back with ``ast.literal_eval``. ``repr(np.float64(1.5))`` is
``'np.float64(1.5)'`` and ``str(np.array([[1., 2.]]))`` has no commas -- both
raise on the way back in, and ``yaml.safe_dump`` refuses a ``np.float64``
outright. So every value leaving this module for a settings row goes through
:func:`plain`, and there are tests that would fail if it did not.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from haemolynx.graph.boundaries import (
    BOUNDARY_ROLE_SETTINGS,
    inlets_outlets_from_vessel_masks,
)
from haemolynx.gui.results import (
    BOUNDARY_COORDINATE_POINT_SIZE,
    PREFIX,
    LayerSpec,
    StageLayers,
    role_colours,
)

__all__ = [
    "AUTOMATED_OVERRIDES_MANUAL_NOTE",
    "BC_BOX_NAMES",
    "BC_BOX_NODES",
    "BC_COORDINATES",
    "BOX_NODE_COLUMNS",
    "BoxNode",
    "DEFAULT_BOX_SIZE_UM",
    "MOVE_DIRECTIONS",
    "box_around",
    "box_centre",
    "box_node_rows",
    "box_nodes_spec",
    "box_size",
    "move_box",
    "nodes_in_box",
    "resize_box",
    "use_node_for_role",
    "view_centre_zyx",
    "BC_LAYER_NAMES",
    "BC_NODE_IDS",
    "BC_REGION_NAMES",
    "boxes_name",
    "regions_name",
    "ROLES",
    "LARGE_AUTO_ROLES",
    "SMALL_AUTO_ROLES",
    "LARGE_VESSEL_NETWORK_ROLES",
    "SMALL_VESSEL_OVERRIDES_MANUAL_NOTE",
    "LARGE_VESSEL_NETWORK_MODE_OFF_NOTE",
    "DISABLED_ROLE_TOOLTIP",
    "BoundaryPicks",
    "box_colours",
    "box_from_rectangle",
    "box_from_view_drag",
    "box_mesh",
    "region_shapes",
    "role_boxes",
    "coordinate_setting",
    "node_id_note",
    "node_id_points",
    "node_id_setting",
    "orderable_settings",
    "role_manual_controls_enabled",
    "settings_for_method",
    "visible_settings",
    "group_for",
    "method_setting",
    "plain",
    "rectangle_from_box",
    "settings_from_layers",
    "snap",
    "specs_for",
    "terminal_points",
    "toggle_node_id",
    "PERCENT_FOR_NODE_ROLE",
    "band_boxes",
    "terminal_axis_span",
    "outside_extent",
    "role_settings",
    "role_title",
    "shared_settings",
    "volume_setting",
    "wanted_rows",
]

#: Shown on the Boundaries tab between the automated mask rows and the
#: manual role controls. Kept here so tests can assert the exact wording
#: without importing Qt.
AUTOMATED_OVERRIDES_MANUAL_NOTE = (
    "While automated vessel assignment and 'Inlets outlets from vessel masks' "
    "are both on, the masks choose the inlets and outlets and override the "
    "manual Inlet and Outlet tabs below. Turn 'Inlets outlets from vessel "
    "masks' off to set them by hand; the masks still drive the large-vessel "
    "treatment."
)

#: The boundary roles, in the order a run assigns them. Taken from the
#: selector's own table rather than restated, so a new role would reach the
#: panel without anything here changing.
ROLES: tuple[str, ...] = tuple(BOUNDARY_ROLE_SETTINGS)

#: Inlet/outlet: greyed while the large-vessel masks choose them
#: (``automated_vessel_assignment`` and ``inlets_outlets_from_vessel_masks``).
LARGE_AUTO_ROLES: frozenset[str] = frozenset({"inlet", "outlet"})

#: Arteriole/venule: greyed when small-vessel mask boundary assignment is on.
SMALL_AUTO_ROLES: frozenset[str] = frozenset(
    {"arteriole_boundary", "venule_boundary"}
)

#: Large-vessel network inlet/outlet: greyed while the feature itself is off,
#: the opposite polarity from LARGE_AUTO_ROLES/SMALL_AUTO_ROLES above (which
#: grey when some *other* automation has taken the role over). There is
#: nothing to override until assign_large_vessel_branch_orders is on.
LARGE_VESSEL_NETWORK_ROLES: frozenset[str] = frozenset(
    {"large_vessel_inlet", "large_vessel_outlet"}
)


def role_manual_controls_enabled(role: str, values: Mapping[str, Any]) -> bool:
    """Whether *role*'s manual sub-tab should stay interactive.

    The large-vessel masks replace manual inlet/outlet picking only while
    they choose the inlets and outlets (:func:`inlets_outlets_from_vessel_masks`:
    ``automated_vessel_assignment`` and ``inlets_outlets_from_vessel_masks``);
    small-vessel mask assignment
    (``use_small_vessel_masks_for_boundary_assignment``) replaces manual
    arteriole/venule boundary picking. The sub-tab stays visible but is
    greyed out -- unlike vessel-mask option rows, which hide.
    """
    if role in LARGE_AUTO_ROLES and inlets_outlets_from_vessel_masks(values):
        return False
    if role in SMALL_AUTO_ROLES and values.get(
        "use_small_vessel_masks_for_boundary_assignment"
    ):
        return False
    if role in LARGE_VESSEL_NETWORK_ROLES and not values.get(
        "assign_large_vessel_branch_orders"
    ):
        return False
    return True


#: Same wording _widget.py used inline for the small-vessel-mask case, now
#: shared so a third disabled reason does not need adding to two call sites.
SMALL_VESSEL_OVERRIDES_MANUAL_NOTE = (
    "Small-vessel mask assignment overrides manual arteriole/venule "
    "boundary selection."
)
LARGE_VESSEL_NETWORK_MODE_OFF_NOTE = (
    "Only used when assign_large_vessel_branch_orders is on."
)

#: Why a role's sub-tab/actions are greyed, keyed by role. One lookup instead
#: of an if/elif duplicated at both call sites in _widget.py.
DISABLED_ROLE_TOOLTIP: dict[str, str] = {
    "inlet": AUTOMATED_OVERRIDES_MANUAL_NOTE,
    "outlet": AUTOMATED_OVERRIDES_MANUAL_NOTE,
    "arteriole_boundary": SMALL_VESSEL_OVERRIDES_MANUAL_NOTE,
    "venule_boundary": SMALL_VESSEL_OVERRIDES_MANUAL_NOTE,
    "large_vessel_inlet": LARGE_VESSEL_NETWORK_MODE_OFF_NOTE,
    "large_vessel_outlet": LARGE_VESSEL_NETWORK_MODE_OFF_NOTE,
}


BC_COORDINATES = f"{PREFIX}BC coordinates"

#: The nodes the ``node_ids`` roles list, marked where the graph has them.
BC_NODE_IDS = f"{PREFIX}BC node IDs"

#: How many decimal places a picked coordinate keeps. A micron is the unit and
#: a nanometre is far below what a click can mean, so three keeps the config
#: readable without throwing anything away.
DECIMALS = 3


#: What each selection method needs configured beyond itself. A method absent
#: from here needs nothing: `all_degree_1` takes every terminal there is.
METHOD_SETTINGS: Mapping[str, str] = {
    "coordinates": "coordinates",
    "volume": "volume_boxes",
    "node_ids": "node_ids",
}

#: Settings shared by every role that selects by band or by distance, so they
#: cannot belong under any one role.
BAND_SETTINGS: Mapping[str, tuple[str, ...]] = {
    "edge_percent": ("boundary_axis", "boundary_first_percent", "boundary_last_percent"),
    "degree_1_from_inlet": ("boundary_distance_from_inlet_node",),
}


#: Which of the two band percentages a role reads. `edge_percent` splits the
#: network along an axis and takes terminals from each end; a run computes both
#: ends every time, but a role only ever takes the one for its own end, so
#: showing an inlet role the outlet percentage invites setting a number that
#: does nothing.
PERCENT_FOR_NODE_ROLE: Mapping[str, str] = {
    "inlet": "boundary_first_percent",
    "outlet": "boundary_last_percent",
}


def settings_for_method(role: str, method: str) -> tuple[str, ...]:
    """The settings *role* reads when it selects nodes by *method*."""
    key = METHOD_SETTINGS.get(str(method))
    if key is not None:
        return (BOUNDARY_ROLE_SETTINGS[role][key],)
    if str(method) == "edge_percent":
        return ("boundary_axis",
                PERCENT_FOR_NODE_ROLE[BOUNDARY_ROLE_SETTINGS[role]["node_role"]])
    return BAND_SETTINGS.get(str(method), ())


def visible_settings(values: Mapping[str, Any]) -> set[str]:
    """Which boundary settings the chosen methods will actually read.

    The Boundaries tab declares many rows and a run reads a handful of them:
    one method per role, plus whatever each role's chosen method asks for.
    Showing the rest invites filling in a coordinate list that nothing will
    look at.
    """
    wanted: set[str] = set()
    for role in ROLES:
        wanted.update(settings_for_method(role, values.get(method_setting(role))))
    return wanted


def orderable_settings() -> list[str]:
    """Every boundary setting this module places, method-first, role by role."""
    ordered: list[str] = []
    for role in ROLES:
        ordered += list(role_settings(role))
    ordered += list(shared_settings())
    return ordered


def role_settings(role: str) -> tuple[str, ...]:
    """Everything that belongs to one role, its method first.

    A role's own page: how it selects, whatever that method reads, and the node
    IDs the run fills in. The band settings are deliberately absent -- one axis
    and one pair of bands describe the whole network, so they belong to no role
    (see :data:`BAND_SETTINGS`).
    """
    names = [
        method_setting(role),
        coordinate_setting(role),
        volume_setting(role),
        node_id_setting(role),
        f"{role}_nodes",
    ]
    return tuple(dict.fromkeys(names))


def shared_settings() -> tuple[str, ...]:
    """The settings more than one role can read, so they sit under all of them."""
    ordered: list[str] = []
    for names in BAND_SETTINGS.values():
        ordered += list(names)
    return tuple(dict.fromkeys(ordered))


def role_title(role: str) -> str:
    """What a role's tab is called: 'arteriole_boundary' -> 'Arteriole'."""
    stem = role[: -len("_boundary")] if role.endswith("_boundary") else role
    return stem.replace("_", " ").capitalize()


def regions_name(role: str) -> str:
    """The layer a role's regions and band are drawn in.

    One per role rather than one for all four: a layer carries one colour and
    one visibility, so sharing them meant inlets and outlets could not be told
    apart in the layer list, nor either hidden on its own.
    """
    return f"{PREFIX}BC {role_title(role).lower()} regions"


def boxes_name(role: str) -> str:
    """The layer a role's regions are drawn in as solid boxes."""
    return f"{PREFIX}BC {role_title(role).lower()} boxes"


#: Every region layer, in role order.
BC_REGION_NAMES = tuple(regions_name(role) for role in ROLES)

#: Every box layer, in role order.
BC_BOX_NAMES = tuple(boxes_name(role) for role in ROLES)

#: All of them, for "is this one of the picking layers?".
#: The nodes inside the box being edited, drawn to choose from.
BC_BOX_NODES = f"{PREFIX}BC box nodes"

BC_LAYER_NAMES = frozenset(
    {BC_COORDINATES, BC_NODE_IDS, BC_BOX_NODES, *BC_REGION_NAMES, *BC_BOX_NAMES}
)


def outside_extent(
    values: Mapping[str, Any],
    lo: Sequence[float],
    hi: Sequence[float],
) -> tuple[str, ...]:
    """Which configured coordinates fall outside the image, role by role.

    The one mistake this cannot self-correct. Every one of these settings is
    microns, and a coordinate read off a viewer that was showing voxel indices
    looks entirely plausible -- it is a small positive triple in the right
    ballpark. On an anisotropic stack it lands at a fraction of the depth it
    should, on no vessel, and the run snaps it to whatever terminal happens to
    be nearest instead of failing. Outside the volume altogether is the part
    that can be detected, and it is the common half of the mistake, since a
    voxel index is smaller than the micron it stands for.
    """
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    notes = []
    for role in ROLES:
        points = BoundaryPicks.from_settings(values).coordinates[role]
        if not points:
            continue
        stray = sum(
            1 for point in points
            if np.any(np.asarray(point, dtype=float) < low)
            or np.any(np.asarray(point, dtype=float) > high)
        )
        if stray:
            notes.append(f"{stray} of {len(points)} {role} coordinate(s)")
    return tuple(notes)


def coordinate_setting(role: str) -> str:
    """The setting holding *role*'s picked coordinates."""
    return BOUNDARY_ROLE_SETTINGS[role]["coordinates"]


def volume_setting(role: str) -> str:
    """The setting holding *role*'s volume boxes."""
    return BOUNDARY_ROLE_SETTINGS[role]["volume_boxes"]


def method_setting(role: str) -> str:
    """The setting saying how *role*'s nodes are selected."""
    return BOUNDARY_ROLE_SETTINGS[role]["method"]


def node_id_setting(role: str) -> str:
    """The setting holding the node IDs *role* takes with ``node_ids``."""
    return BOUNDARY_ROLE_SETTINGS[role]["node_ids"]


def plain(value: Any) -> Any:
    """*value* as builtin floats and lists, however deeply nested.

    The one boundary between numpy and a settings row. A numpy scalar's ``repr``
    is ``np.float64(1.5)``, which ``ast.literal_eval`` rejects, and
    ``yaml.safe_dump`` will not represent it at all -- so a picked coordinate
    that reached a row still wearing its numpy type would break the row the next
    time anyone touched it, and break saving the config outright.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def _entries(value: Any) -> list[Any]:
    """*value* as a list of entries, whatever container it arrived in.

    Not `value or ()`: a numpy array raises "truth value is ambiguous" there,
    and a settings dict handed straight from a caller can hold one.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _finite_point(value: Any) -> list[float] | None:
    """*value* as three finite floats, or None if it is not a point."""
    try:
        numbers = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(numbers) != 3 or not all(math.isfinite(n) for n in numbers):
        return None
    return numbers


def _node_id(value: Any) -> Any | None:
    """*value* as a plain node ID, or None if it cannot be one.

    Node IDs are ints; one that arrived as ``np.int64`` or ``12.0`` is the
    same node, and goes back to a row as a plain int (see :func:`plain`).
    """
    value = plain(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and value.strip():
        text = value.strip()
        return int(text) if text.lstrip("-").isdigit() else text
    return None


def _node_id_entries(value: Any) -> list[Any]:
    """A node-ID setting as a list of entries: a lone ID typed bare is one."""
    if isinstance(value, (str, int, float, np.generic)) and not isinstance(value, bool):
        return [value]
    return _entries(value)


def _corner_pair(value: Any) -> list[list[float]] | None:
    """*value* as exactly two corner points, or None if it is not a box."""
    try:
        corners = list(value)
    except TypeError:
        return None
    if len(corners) != 2:
        return None
    lo, hi = _finite_point(corners[0]), _finite_point(corners[1])
    if lo is None or hi is None:
        return None
    return [lo, hi]


def rectangle_from_box(
    corner_a: Sequence[float], corner_b: Sequence[float]
) -> tuple[np.ndarray, float]:
    """A box as the rectangle that draws it, and the depth it was drawn from.

    The rectangle is planar, at the box's z centre, carrying its y/x extent --
    which is the only thing napari can draw and edit, since a Shapes layer has
    no 3D box and cannot be edited in the 3D view at all. The z extent comes
    back as *depth*, so nothing about the box is lost.
    """
    lo = np.minimum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
    hi = np.maximum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
    centre_z = float((lo[0] + hi[0]) / 2.0)
    corners = np.array(
        [
            [centre_z, lo[1], lo[2]],
            [centre_z, lo[1], hi[2]],
            [centre_z, hi[1], hi[2]],
            [centre_z, hi[1], lo[2]],
        ],
        dtype=float,
    )
    return corners, float(hi[0] - lo[0])


def box_from_rectangle(
    corners: Sequence[Sequence[float]], *, depth: float
) -> list[list[float]]:
    """The two opposite corners a rectangle and a depth describe.

    The inverse of :func:`rectangle_from_box`. The rectangle's own z is the
    centre of the box, so the depth is spread symmetrically about it: a
    rectangle drawn on a slice grows equally into the slices either side, which
    is what someone drawing on the middle of a stack means by it.
    """
    points = np.asarray(corners, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(
            f"A region needs at least three (z, y, x) corners, got {points.shape}."
        )
    half = abs(float(depth)) / 2.0
    centre_z = float(points[:, 0].mean())
    lo = [centre_z - half, float(points[:, 1].min()), float(points[:, 2].min())]
    hi = [centre_z + half, float(points[:, 1].max()), float(points[:, 2].max())]
    return [plain([round(v, DECIMALS) for v in lo]), plain([round(v, DECIMALS) for v in hi])]


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    """A unit cube's eight corners and its twelve outward-wound triangles.

    Corner ``i`` has bit 4 for z, 2 for y and 1 for x, so ``lo + corner * size``
    places it. The winding is settled once here, by turning each triangle to
    face away from the centre, because flat shading lights a face by its
    normal and an inward one reads as a hole in the box.
    """
    corners = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
    quads = []
    for axis in range(3):
        for side in (0.0, 1.0):
            quad = [i for i in range(8) if corners[i][axis] == side]
            # Round the face, not across it: swap the last two into order.
            quads.append([quad[0], quad[1], quad[3], quad[2]])
    faces = []
    centre = np.full(3, 0.5)
    for a, b, c, d in quads:
        for tri in ((a, b, c), (a, c, d)):
            p = corners[list(tri)]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            if np.dot(normal, p.mean(axis=0) - centre) < 0:
                tri = (tri[0], tri[2], tri[1])
            faces.append(tri)
    return corners, np.asarray(faces, dtype=int)


_CUBE_CORNERS, _CUBE_FACES = _unit_cube()


def box_mesh(
    boxes: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every box as a closed solid: vertices, triangles and whose each vertex is.

    Each box has its own eight vertices rather than sharing any, so it can
    carry a colour of its own (:func:`box_colours`) -- the whole point of
    drawing them as separate solids is that two regions of one role can be
    told apart in 3D.
    """
    if not len(boxes):
        return (np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int),
                np.empty((0,), dtype=int))
    vertices, faces, owner = [], [], []
    for index, (corner_a, corner_b) in enumerate(boxes):
        lo = np.minimum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
        hi = np.maximum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
        vertices.append(lo + _CUBE_CORNERS * (hi - lo))
        faces.append(_CUBE_FACES + 8 * index)
        owner.append(np.full(8, index, dtype=int))
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(owner)


def box_colours(
    colour: Sequence[float], count: int
) -> list[tuple[float, float, float, float]]:
    """*count* shades of one role's colour, one per box, all telling apart.

    The role stays readable (an inlet box is still green) while two inlet
    boxes are not the same green: the shades run from darker to lighter than
    the role colour, which a single box keeps exactly.
    """
    base = np.asarray(colour, dtype=float)[:3]
    alpha = float(colour[3]) if len(colour) > 3 else 1.0
    if count <= 1:
        return [(*(float(v) for v in base), alpha)] * max(count, 0)
    shades = []
    for step in np.linspace(-0.45, 0.45, count):
        towards = np.ones(3) if step > 0 else np.zeros(3)
        rgb = base + abs(step) * (towards - base)
        shades.append((*(float(v) for v in rgb), alpha))
    return shades


def role_boxes(
    picks: "BoundaryPicks",
    bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    *,
    extra: Mapping[str, Sequence[tuple[Sequence[float], Sequence[float]]]] | None = None,
) -> dict[str, list[tuple[list[float], list[float]]]]:
    """The boxes each role's box layer shows: its band, its regions, *extra*.

    *extra* is a box still being dragged out in 3D -- shown so the drag has
    something to look at, but in no setting until the mouse comes up.
    """
    out: dict[str, list] = {}
    for role in ROLES:
        boxes = []
        if bands and role in bands:
            lo, hi = bands[role]
            boxes.append(([float(v) for v in lo], [float(v) for v in hi]))
        for lo, hi in picks.volumes.get(role, ()):
            boxes.append(([float(v) for v in lo], [float(v) for v in hi]))
        for lo, hi in (extra or {}).get(role, ()):
            boxes.append(([float(v) for v in lo], [float(v) for v in hi]))
        if boxes:
            out[role] = boxes
    return out


def box_from_view_drag(
    start: Sequence[float],
    end: Sequence[float],
    view_direction: Sequence[float],
    up_direction: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    depth: float | None = None,
) -> list[list[float]] | None:
    """The box a mouse drag across the 3D view describes, or None.

    A drag on screen is a rectangle in the plane facing the camera: across it,
    the box takes the rectangle; along the line of sight -- the axis the camera
    looks down most nearly -- it runs through the whole image, since a 3D view
    shows no depth to point at. On a z view, *depth* trims that to a slab
    centred on the image, which is what the Region depth slider says. Looking
    straight down an axis gives exactly the rectangle dragged; at an angle, the
    smallest axis-aligned box holding it.

    None for a drag with no area across the screen, or one that misses the
    image altogether: there is no box to add, and adding a flat one would
    select nothing.
    """
    start = np.asarray(start, dtype=float)[-3:]
    end = np.asarray(end, dtype=float)[-3:]
    view = np.asarray(view_direction, dtype=float)[-3:]
    up = np.asarray(up_direction, dtype=float)[-3:]
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    if not np.linalg.norm(view):
        return None
    view = view / np.linalg.norm(view)
    up = up - np.dot(up, view) * view
    if not np.linalg.norm(up):
        return None
    up = up / np.linalg.norm(up)
    right = np.cross(view, up)
    delta = end - start
    across, along = float(np.dot(delta, right)), float(np.dot(delta, up))
    corners = np.array([
        start,
        start + across * right,
        start + along * up,
        start + across * right + along * up,
    ])
    box_lo, box_hi = corners.min(axis=0), corners.max(axis=0)
    sight = int(np.argmax(np.abs(view)))
    box_lo[sight], box_hi[sight] = low[sight], high[sight]
    if depth is not None and sight == 0 and math.isfinite(depth) and depth > 0:
        centre = (low[0] + high[0]) / 2.0
        box_lo[0], box_hi[0] = centre - depth / 2.0, centre + depth / 2.0
    box_lo, box_hi = np.maximum(box_lo, low), np.minimum(box_hi, high)
    if any(box_hi[axis] - box_lo[axis] <= 0 for axis in range(3) if axis != sight):
        return None
    return [plain([round(float(v), DECIMALS) for v in box_lo]),
            plain([round(float(v), DECIMALS) for v in box_hi])]


def terminal_axis_span(graph, axis: int):
    """The lowest and highest terminal along *axis*, or None without a graph.

    What the selector measures its bands across -- not the image. A network
    rarely reaches the edge of its image, so an image-relative band routinely
    holds no terminal at all (see `select_boundary_terminal_nodes`).
    """
    points, _ids = terminal_points(graph)
    if not len(points) or not 0 <= int(axis) < points.shape[1]:
        return None
    column = points[:, int(axis)]
    return float(column.min()), float(column.max())


def band_boxes(
    values: Mapping[str, Any],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    axis_span: tuple[float, float] | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """The slab each `edge_percent` role takes its terminals from.

    `edge_percent` is the one method whose region is implied rather than
    written down: a percentage and an axis describe a box, but nothing draws
    it, so the only way to find out what it selected was to run and look at
    the answer. The slab spans everything across the other two axes, because
    the selector's rule is on one coordinate alone.

    *axis_span* is where the terminals actually reach, which is what a run
    measures across; without a graph the caller passes None and the extent
    stands in, which is close enough to point at and marked as an estimate.
    """
    low = [float(v) for v in lo]
    high = [float(v) for v in hi]
    axis = values.get("boundary_axis")
    try:
        axis = int(axis)
    except (TypeError, ValueError):
        return {}
    if not 0 <= axis < len(low):
        return {}

    start, end = axis_span if axis_span is not None else (low[axis], high[axis])
    span = float(end) - float(start)
    boxes: dict[str, tuple[list[float], list[float]]] = {}
    for role in ROLES:
        if str(values.get(method_setting(role))) != "edge_percent":
            continue
        name = PERCENT_FOR_NODE_ROLE[BOUNDARY_ROLE_SETTINGS[role]["node_role"]]
        try:
            percent = float(values.get(name))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(percent):
            continue
        reach = span * max(0.0, min(100.0, percent)) / 100.0
        corner_lo, corner_hi = list(low), list(high)
        if name == "boundary_first_percent":
            corner_lo[axis], corner_hi[axis] = float(start), float(start) + reach
        else:
            corner_lo[axis], corner_hi[axis] = float(end) - reach, float(end)
        boxes[role] = (corner_lo, corner_hi)
    return boxes


def region_shapes(
    picks: "BoundaryPicks",
    *,
    only: str | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, np.ndarray]]:
    """Every region as the one editable rectangle that stands for it.

    Only rectangles: the box each one describes is drawn in its own layer
    (:func:`box_mesh`). It used to be twelve line segments in this one, which
    made thirteen shapes per region for napari to rebuild on every edit and
    left its drawing tool working on a layer that was being rewritten
    underneath it.
    """
    data: list[np.ndarray] = []
    roles: list[str] = []
    depths: list[float] = []
    for role in ROLES if only is None else (only,):
        for lo, hi in picks.volumes.get(role, ()):
            corners, depth = rectangle_from_box(lo, hi)
            data.append(corners)
            roles.append(role)
            depths.append(depth)
    features = {
        "role": np.asarray(roles, dtype=object),
        "depth": np.asarray(depths, dtype=float),
    }
    return data, ["rectangle"] * len(data), features


@dataclass(frozen=True)
class BoundaryPicks:
    """What the four roles' coordinate and volume settings currently say."""

    coordinates: Mapping[str, tuple[tuple[float, float, float], ...]]
    volumes: Mapping[str, tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]]
    #: One line per entry that could not be read, for the report box. A panel
    #: must not fall over on a hand-edited config; the run raises for the same
    #: value later, which is the right place for a hard error.
    problems: tuple[str, ...] = field(default=())
    #: The node IDs each role lists for the ``node_ids`` method, in order.
    node_ids: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, values: Mapping[str, Any]) -> "BoundaryPicks":
        """Read all four roles out of a settings dict, skipping what will not read."""
        coordinates: dict[str, tuple] = {}
        volumes: dict[str, tuple] = {}
        node_ids: dict[str, tuple] = {}
        problems: list[str] = []
        for role in ROLES:
            ids: list[Any] = []
            for index, entry in enumerate(_node_id_entries(values.get(node_id_setting(role)))):
                node = _node_id(entry)
                if node is None:
                    problems.append(
                        f"{node_id_setting(role)}[{index}] is not a node ID: {entry!r}"
                    )
                    continue
                if node not in ids:
                    ids.append(node)
            node_ids[role] = tuple(ids)

            points: list[tuple[float, float, float]] = []
            for index, entry in enumerate(_entries(values.get(coordinate_setting(role)))):
                point = _finite_point(entry)
                if point is None:
                    problems.append(
                        f"{coordinate_setting(role)}[{index}] is not a "
                        f"(z, y, x) point: {entry!r}"
                    )
                    continue
                points.append(tuple(point))
            coordinates[role] = tuple(points)

            boxes: list[tuple] = []
            for index, entry in enumerate(_entries(values.get(volume_setting(role)))):
                pair = _corner_pair(entry)
                if pair is None:
                    problems.append(
                        f"{volume_setting(role)}[{index}] is not two "
                        f"(z, y, x) corners: {entry!r}"
                    )
                    continue
                boxes.append((tuple(pair[0]), tuple(pair[1])))
            volumes[role] = tuple(boxes)
        return cls(coordinates, volumes, tuple(problems), node_ids)

    def to_settings(self) -> dict[str, list]:
        """The eight settings these picks describe, as plain lists of floats."""
        out: dict[str, list] = {}
        for role in ROLES:
            out[coordinate_setting(role)] = [
                plain([round(float(v), DECIMALS) for v in point])
                for point in self.coordinates.get(role, ())
            ]
            out[volume_setting(role)] = [
                [
                    plain([round(float(v), DECIMALS) for v in lo]),
                    plain([round(float(v), DECIMALS) for v in hi]),
                ]
                for lo, hi in self.volumes.get(role, ())
            ]
        return out

    def points(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Every coordinate as ``(N, 3)`` plus the role each one belongs to."""
        positions: list[tuple[float, float, float]] = []
        roles: list[str] = []
        for role in ROLES:
            for point in self.coordinates.get(role, ()):
                positions.append(point)
                roles.append(role)
        data = (
            np.asarray(positions, dtype=float)
            if positions
            else np.empty((0, 3), dtype=float)
        )
        return data, {"role": np.asarray(roles, dtype=object)}

    def rectangles(self) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Every box as the rectangle that draws it, plus role and depth."""
        shapes: list[np.ndarray] = []
        roles: list[str] = []
        depths: list[float] = []
        for role in ROLES:
            for lo, hi in self.volumes.get(role, ()):
                corners, depth = rectangle_from_box(lo, hi)
                shapes.append(corners)
                roles.append(role)
                depths.append(depth)
        return shapes, {
            "role": np.asarray(roles, dtype=object),
            "depth": np.asarray(depths, dtype=float),
        }

    def summary(self) -> str:
        """One line naming what is configured, for the report box."""
        parts = [
            f"{len(self.coordinates[role])} {role} "
            f"coordinate{'' if len(self.coordinates[role]) == 1 else 's'}"
            for role in ROLES
            if self.coordinates.get(role)
        ]
        parts += [
            f"{len(self.volumes.get(role, ()))} {role} "
            f"region{'' if len(self.volumes[role]) == 1 else 's'}"
            for role in ROLES
            if self.volumes.get(role)
        ]
        parts += [
            f"{role} node ID{'' if len(self.node_ids[role]) == 1 else 's'} "
            f"{list(self.node_ids[role])}"
            for role in ROLES
            if self.node_ids.get(role)
        ]
        if not parts:
            return "No boundary conditions configured."
        return ", ".join(parts)


def toggle_node_id(
    values: Mapping[str, Any], role: str, node_id: Any
) -> tuple[dict[str, list], str]:
    """The node-ID settings after clicking *node_id* for *role*, and what changed.

    A click adds the node to *role*, or takes it off again if *role* already
    has it -- so a misclick is undone by clicking the same node. A node another
    role lists moves to this one rather than being listed twice: a run gives
    each node to the first role that claims it (inlets, outlets, then the
    vessel boundaries), so a second listing would only be silently ignored.
    Entries that cannot be read are kept as they are; the report box names them.
    """
    node = _node_id(node_id)
    if node is None:
        return {}, f"{node_id!r} is not a node ID."
    proposed: dict[str, list] = {}
    moved_from: list[str] = []
    for other in ROLES:
        if other == role:
            continue
        entries = _node_id_entries(values.get(node_id_setting(other)))
        kept = [plain(entry) for entry in entries if _node_id(entry) != node]
        if len(kept) != len(entries):
            proposed[node_id_setting(other)] = kept
            moved_from.append(other)
    entries = _node_id_entries(values.get(node_id_setting(role)))
    mine = [plain(entry) for entry in entries if _node_id(entry) != node]
    if len(mine) != len(entries) and not moved_from:
        proposed[node_id_setting(role)] = mine
        return proposed, f"Removed node {node} from {role}."
    proposed[node_id_setting(role)] = [*mine, node]
    moved = f" (moved from {', '.join(moved_from)})" if moved_from else ""
    return proposed, f"Added node {node} to {role}{moved}."


def node_id_points(
    graph: Any, values: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list]]:
    """Where each listed node is, its role and ID, and the IDs *graph* lacks.

    Without a graph nothing can be placed and nothing can be called missing:
    node IDs only mean something against the graph a run built.
    """
    picks = BoundaryPicks.from_settings(values)
    positions: list[np.ndarray] = []
    roles: list[str] = []
    ids: list[Any] = []
    missing: dict[str, list] = {}
    for role in ROLES:
        for node in picks.node_ids.get(role, ()):
            if graph is None:
                continue
            # `_node_id` has already turned "12" into 12, as the run does.
            position = graph.nodes[node].get("pos") if node in graph.nodes else None
            if position is None:
                missing.setdefault(role, []).append(node)
                continue
            positions.append(np.asarray(position, dtype=float)[:3])
            roles.append(role)
            ids.append(node)
    data = np.asarray(positions, dtype=float) if positions else np.empty((0, 3), dtype=float)
    features = {
        "role": np.asarray(roles, dtype=object),
        "node_id": np.asarray(ids, dtype=object),
    }
    return data, features, missing


def specs_for(values: Mapping[str, Any], graph: Any = None) -> tuple[LayerSpec, ...]:
    """The editable layers that draw what *values* describes.

    The coordinates layer is emitted even when empty -- it is the surface the
    user clicks into, so it has to exist before there is anything on it. The
    regions layer is not: an empty Shapes layer draws nothing and would only be
    one more row in the layer list until a region is drawn. The box layers are
    not specs at all: napari has no Surface in `LayerSpec`'s vocabulary, and
    they are drawn from :func:`role_boxes` by the panel. The node IDs layer is
    emitted whenever a role lists any, placed on *graph* -- empty without one,
    since an ID has no position until a run has built the graph it names.
    """
    picks = BoundaryPicks.from_settings(values)
    points, point_features = picks.points()
    specs = [
        LayerSpec(
            kind="points",
            name=BC_COORDINATES,
            data=points,
            features=point_features,
            colour_by="role",
            colour_kind="categorical",
            colour_cycle=role_colours(),
            options={
                # A ring, so what you asked for cannot be mistaken for
                # `HaemoLynx boundary nodes`, which is what the run snapped to.
                "symbol": "ring",
                "size": BOUNDARY_COORDINATE_POINT_SIZE,
                "border_width": 0.25,
                "out_of_slice_display": True,
            },
        )
    ]
    for role in ROLES:
        shapes, kinds, shape_features = region_shapes(picks, only=role)
        if not shapes:
            continue
        specs.append(
            LayerSpec(
                kind="shapes",
                name=regions_name(role),
                data=shapes,
                features=shape_features,
                colour_by="role",
                colour_kind="categorical",
                colour_cycle=role_colours(),
                options={"shape_type": kinds, "edge_width": 1.5, "opacity": 0.25},
            )
        )
    if any(picks.node_ids.values()):
        nodes, node_features, _missing = node_id_points(graph, values)
        specs.append(
            LayerSpec(
                kind="points",
                name=BC_NODE_IDS,
                data=nodes,
                features=node_features,
                colour_by="role",
                colour_kind="categorical",
                colour_cycle=role_colours(),
                options={
                    # A square, so a listed node cannot be mistaken for a
                    # coordinate's ring or for the run's own boundary nodes.
                    "symbol": "square",
                    "size": BOUNDARY_COORDINATE_POINT_SIZE,
                    "border_width": 0.25,
                    "out_of_slice_display": True,
                },
            )
        )
    return tuple(specs)


def node_id_note(graph: Any, values: Mapping[str, Any]) -> str:
    """What the report box says about the listed node IDs, if anything."""
    picks = BoundaryPicks.from_settings(values)
    if not any(picks.node_ids.values()):
        return ""
    if graph is None:
        return (
            "  Node IDs are placed once a run has built the graph ('3. Graph'); "
            "until then they cannot be shown or checked."
        )
    _points, _features, missing = node_id_points(graph, values)
    if not missing:
        return ""
    listed = "; ".join(f"{role} {ids}" for role, ids in missing.items())
    return (
        f"  Not in the current graph: {listed}. A run with these IDs stops and "
        "says so -- rebuilding the graph renumbers its nodes, so pick them again."
    )


def group_for(values: Mapping[str, Any], bands=None, graph: Any = None) -> StageLayers:
    """The picking layers as a group the panel can hand to `_apply_layers`."""
    picks = BoundaryPicks.from_settings(values)
    note = picks.summary()
    if bands:
        drawn = ", ".join(f"{role} band" for role in bands)
        configured = (any(picks.coordinates.values()) or any(picks.volumes.values())
                      or any(picks.node_ids.values()))
        note = f"{note}, {drawn}" if configured else drawn
    if picks.problems:
        note = f"{note} ({len(picks.problems)} entry could not be read: {picks.problems[0]})"
    note += node_id_note(graph, values)
    return StageLayers(
        stage="boundary_picking",
        title="Boundary conditions",
        layers=specs_for(values, graph),
        note=note,
    )


def settings_from_layers(
    *,
    points: Any = None,
    point_roles: Iterable[str] | None = None,
    rectangles: Sequence[Any] | None = None,
    rectangle_roles: Iterable[str] | None = None,
    depths: Iterable[float] | None = None,
) -> dict[str, list]:
    """The eight settings the layers' current contents describe.

    Anything absent is left out rather than emptied, so syncing one layer never
    clears the other's settings.
    """
    out: dict[str, list] = {}
    if points is not None:
        by_role: dict[str, list] = {role: [] for role in ROLES}
        roles = list(point_roles or ())
        for index, point in enumerate(np.asarray(points, dtype=float)):
            role = roles[index] if index < len(roles) else ROLES[0]
            by_role.setdefault(str(role), []).append(tuple(point))
        for role in ROLES:
            out[coordinate_setting(role)] = [
                plain([round(float(v), DECIMALS) for v in point])
                for point in by_role.get(role, ())
            ]
    if rectangles is not None:
        boxes: dict[str, list] = {role: [] for role in ROLES}
        roles = list(rectangle_roles or ())
        sizes = list(depths or ())
        for index, corners in enumerate(rectangles):
            role = str(roles[index]) if index < len(roles) else ROLES[0]
            depth = float(sizes[index]) if index < len(sizes) else 0.0
            boxes.setdefault(role, []).append(box_from_rectangle(corners, depth=depth))
        for role in ROLES:
            out[volume_setting(role)] = boxes.get(role, [])
    return out


def wanted_rows(
    proposed: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Only the settings whose value would actually change.

    The compare-before-write that keeps a sync from cascading: writing a row
    fires its `changed`, so writing rows that already hold the value would set
    the panel talking to itself.
    """
    changed: dict[str, Any] = {}
    for name, value in proposed.items():
        if plain(value) != plain(current.get(name)):
            changed[name] = value
    return changed


def terminal_points(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    """Positions and ids of every degree-1 node that has a position.

    The same candidates the selector uses, so the panel cannot promise a node
    the run would not choose.
    """
    positions: list[Any] = []
    ids: list[Any] = []
    if graph is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=object)
    for node, degree in graph.degree():
        if degree != 1:
            continue
        position = graph.nodes[node].get("pos")
        if position is None:
            continue
        positions.append(np.asarray(position, dtype=float)[:3])
        ids.append(node)
    data = (
        np.asarray(positions, dtype=float)
        if positions
        else np.empty((0, 3), dtype=float)
    )
    return data, np.asarray(ids, dtype=object)


def snap(points: Any, candidates: Any) -> tuple[np.ndarray, np.ndarray]:
    """Move each point onto its nearest candidate; report how far each moved.

    With no candidates the points are returned untouched and every distance is
    zero -- there is nothing to snap to before a run has built a graph, and
    that is not an error.
    """
    moved = np.asarray(points, dtype=float).reshape(-1, 3).copy()
    targets = np.asarray(candidates, dtype=float).reshape(-1, 3)
    if not len(moved) or not len(targets):
        return moved, np.zeros(len(moved), dtype=float)
    distances = np.linalg.norm(moved[:, None, :] - targets[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)          # ties take the lowest index
    return targets[nearest].copy(), distances[np.arange(len(moved)), nearest]


# --- inserting a box and choosing a node from it ------------------------------

#: The size, (z, y, x) in microns, of a box Insert a box places.
DEFAULT_BOX_SIZE_UM: tuple[float, float, float] = (10.0, 10.0, 10.0)

#: The box-moving buttons, in the order the panel lays them out, each with the
#: screen direction it stands for. "back"/"forward" move through the axis not
#: on screen -- the slices, in the usual XY view.
MOVE_DIRECTIONS: tuple[str, ...] = ("left", "right", "up", "down", "back", "forward")

#: Header of the "nodes in this box" table, one entry per :func:`box_node_rows` cell.
BOX_NODE_COLUMNS = ("Node ID", "Vessels", "Open end", "z (um)", "y (um)", "x (um)")


def _corners(box: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    pair = _corner_pair(box)
    if pair is None:
        raise ValueError(f"A box is two (z, y, x) corners, got {box!r}")
    a, b = np.asarray(pair[0], dtype=float), np.asarray(pair[1], dtype=float)
    return np.minimum(a, b), np.maximum(a, b)


def _as_box(lo: np.ndarray, hi: np.ndarray) -> list[list[float]]:
    return [plain([round(float(v), DECIMALS) for v in lo]),
            plain([round(float(v), DECIMALS) for v in hi])]


def box_around(
    centre: Sequence[float], size: Sequence[float] = DEFAULT_BOX_SIZE_UM
) -> list[list[float]]:
    """The box of *size* (z, y, x) microns centred on *centre*, as two corners."""
    middle = np.asarray(centre, dtype=float)[-3:]
    half = np.abs(np.asarray(size, dtype=float)[-3:]) / 2.0
    if middle.shape != (3,) or half.shape != (3,) or not np.all(np.isfinite([*middle, *half])):
        raise ValueError(f"A box needs a (z, y, x) centre and size, got {centre!r}, {size!r}")
    return _as_box(middle - half, middle + half)


def box_centre(box: Sequence[Sequence[float]]) -> list[float]:
    """The (z, y, x) middle of *box*."""
    lo, hi = _corners(box)
    return plain([round(float(v), DECIMALS) for v in (lo + hi) / 2.0])


def box_size(box: Sequence[Sequence[float]]) -> list[float]:
    """The (z, y, x) extent of *box*, in microns."""
    lo, hi = _corners(box)
    return plain([round(float(v), DECIMALS) for v in hi - lo])


def resize_box(box: Sequence[Sequence[float]], size: Sequence[float]) -> list[list[float]]:
    """*box* at a new (z, y, x) *size*, about the same centre."""
    return box_around(box_centre(box), size)


def move_box(
    box: Sequence[Sequence[float]],
    direction: str,
    step: float,
    displayed: Sequence[int] = (1, 2),
    *,
    view_direction: Sequence[float] | None = None,
    up_direction: Sequence[float] | None = None,
) -> list[list[float]]:
    """*box* moved *step* microns in a *direction* as the viewer shows it.

    In 2D, *displayed* is napari's ``dims.displayed``: the last axis runs
    across the screen (left/right), the one before it down the screen
    (up/down -- napari draws rows downwards, so "up" lowers that coordinate),
    and back/forward step through the axis not on screen, the slices.

    In 3D, pass the camera's *view_direction* and *up_direction* (napari's,
    in displayed-axis order): left/right follow the screen's right, up/down
    its up, and forward/back run away from and towards you. The box moves
    along whichever image axis lies closest to that direction, so it stays
    aligned with the image and moves by exactly *step* -- left is left however
    the view is turned. Without the camera, 3D falls back to z/y/x.
    """
    if direction not in MOVE_DIRECTIONS:
        raise ValueError(f"direction must be one of {MOVE_DIRECTIONS}, not {direction!r}")
    axes = [int(a) for a in displayed if 0 <= int(a) < 3]
    lo, hi = _corners(box)
    shift = np.zeros(3)
    if len(axes) == 3 and view_direction is not None and up_direction is not None:
        view = np.asarray(view_direction, dtype=float)[-3:]
        up = np.asarray(up_direction, dtype=float)[-3:]
        right = np.cross(view, up)
        on_screen = {"right": right, "left": -right, "up": up, "down": -up,
                     "forward": view, "back": -view}[direction]
        index = int(np.argmax(np.abs(on_screen)))
        shift[axes[index]] = np.sign(on_screen[index]) * abs(float(step))
        return _as_box(lo + shift, hi + shift)
    axes = axes[-2:] if len(axes) >= 2 else [1, 2]
    vertical, horizontal = axes
    depth = next((a for a in range(3) if a not in axes), 0)
    axis, sign = {
        "left": (horizontal, -1.0), "right": (horizontal, 1.0),
        "up": (vertical, -1.0), "down": (vertical, 1.0),
        "back": (depth, -1.0), "forward": (depth, 1.0),
    }[direction]
    shift[axis] = sign * abs(float(step))
    return _as_box(lo + shift, hi + shift)


@dataclass(frozen=True)
class BoxNode:
    """One node of the graph inside a box, as the panel lists it."""

    node_id: Any
    degree: int
    position: tuple[float, float, float]

    @property
    def open_end(self) -> bool:
        """A vessel ends here: the kind of node an inlet or outlet usually is."""
        return self.degree == 1


def nodes_in_box(graph: Any, box: Sequence[Sequence[float]]) -> list[BoxNode]:
    """Every node of *graph* inside *box* (edges included), open ends first.

    Then by distance from the box's centre, so the node the box was put round
    comes to the top of its group.
    """
    if graph is None:
        return []
    lo, hi = _corners(box)
    middle = (lo + hi) / 2.0
    found = []
    for node, degree in graph.degree():
        pos = graph.nodes[node].get("pos")
        if pos is None:
            continue
        p = np.asarray(pos, dtype=float)[:3]
        if np.all(p >= lo) and np.all(p <= hi):
            found.append(BoxNode(node, int(degree), tuple(float(v) for v in p)))
    found.sort(key=lambda n: (not n.open_end,
                              float(np.linalg.norm(np.asarray(n.position) - middle)), str(n.node_id)))
    return found


def box_node_rows(nodes: Sequence[BoxNode]) -> list[tuple[str, ...]]:
    """The "nodes in this box" table's cells, one row per node, in order."""
    return [
        (
            str(n.node_id),
            str(n.degree),
            "open end" if n.open_end else "",
            *(f"{v:.1f}" for v in n.position),
        )
        for n in nodes
    ]


def box_nodes_spec(nodes: Sequence[BoxNode]) -> LayerSpec:
    """The nodes in the box, yellow and larger than the network's own dots.

    Open ends are larger again and rimmed in red, so the candidates for an
    inlet or outlet stand out from the junctions around them.
    """
    data = (np.asarray([n.position for n in nodes], dtype=float)
            if nodes else np.empty((0, 3), dtype=float))
    open_end = np.asarray([n.open_end for n in nodes], dtype=bool)
    yellow, red = (1.0, 0.9, 0.0, 1.0), (1.0, 0.15, 0.1, 1.0)
    return LayerSpec(
        kind="points",
        name=BC_BOX_NODES,
        data=data,
        features={
            "node_id": np.asarray([n.node_id for n in nodes], dtype=object),
            "open_end": open_end,
        },
        options={
            "size": np.where(open_end, BOUNDARY_COORDINATE_POINT_SIZE * 1.5,
                             BOUNDARY_COORDINATE_POINT_SIZE),
            "face_color": np.asarray([yellow] * len(nodes)).reshape(-1, 4),
            "border_color": np.asarray(
                [red if o else yellow for o in open_end]).reshape(-1, 4),
            "border_width": 0.3,
            "out_of_slice_display": True,
        },
    )


def use_node_for_role(
    values: Mapping[str, Any], role: str, node_id: Any
) -> tuple[dict[str, Any], str]:
    """The settings that make *node_id* one of *role*'s nodes, and what changed.

    The node is added to the role's node IDs (taken off any other role that
    listed it, as a click does -- :func:`toggle_node_id`) and the role is
    switched to the ``node_ids`` method, so a run takes exactly the nodes
    chosen rather than every open end in the box. Choosing a node the role
    already has changes nothing.
    """
    node = _node_id(node_id)
    if node is None:
        return {}, f"{node_id!r} is not a node ID."
    proposed: dict[str, Any] = {}
    already = node in [_node_id(e) for e in _node_id_entries(values.get(node_id_setting(role)))]
    if not already:
        proposed, _toggled = toggle_node_id(values, role, node)
    if str(values.get(method_setting(role))) != "node_ids":
        proposed[method_setting(role)] = "node_ids"
    if already and not proposed:
        return {}, f"Node {node} is already one of {role}'s nodes."
    return proposed, f"Node {node} is now one of {role}'s nodes."


def view_centre_zyx(
    point: Sequence[float], displayed: Sequence[int], camera_centre: Sequence[float]
) -> list[float]:
    """The (z, y, x) micron point in the middle of what the viewer shows.

    *point* is napari's ``dims.point`` (the slice each axis is at, in world
    units), *displayed* its ``dims.displayed`` and *camera_centre* its
    ``camera.center``, whose last entries are the on-screen axes in displayed
    order. Axes on screen take the camera's centre, the rest the slice -- so a
    box inserted in the XY view lands on the slice being looked at.
    """
    full = [float(v) for v in point]
    offset = max(0, len(full) - 3)
    centre = [float(v) for v in camera_centre][-len(displayed):] if displayed else []
    for axis, value in zip(displayed, centre):
        if 0 <= int(axis) < len(full):
            full[int(axis)] = value
    out = full[offset:offset + 3]
    while len(out) < 3:
        out.insert(0, 0.0)
    return out
