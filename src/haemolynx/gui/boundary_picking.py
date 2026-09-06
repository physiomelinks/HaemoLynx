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

``HaemoLynx BC regions``
    A Shapes layer, one rectangle per volume box, drawn at the box's z centre
    with the box's y/x extent. The z extent it came from rides alongside as a
    ``depth`` feature, so every region keeps its own rather than sharing one.
    :func:`rectangle_from_box` and :func:`box_from_rectangle` are exact
    inverses, which is what lets the layer be treated as the setting.

Both layers are editable, and everything here is pure: settings in, layer specs
out, layer data in, settings out. Nothing imports napari, so it is all testable
without a display -- the same contract :mod:`haemolynx.gui.results` keeps.

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

from haemolynx.graph.boundaries import BOUNDARY_ROLE_SETTINGS
from haemolynx.gui.results import (
    BOUNDARY_COORDINATE_POINT_SIZE,
    PREFIX,
    LayerSpec,
    StageLayers,
    role_colours,
)

__all__ = [
    "AUTOMATED_OVERRIDES_MANUAL_NOTE",
    "BC_COORDINATES",
    "BC_LAYER_NAMES",
    "BC_REGION_NAMES",
    "regions_name",
    "ROLES",
    "LARGE_AUTO_ROLES",
    "SMALL_AUTO_ROLES",
    "LARGE_VESSEL_NETWORK_ROLES",
    "SMALL_VESSEL_OVERRIDES_MANUAL_NOTE",
    "LARGE_VESSEL_NETWORK_MODE_OFF_NOTE",
    "DISABLED_ROLE_TOOLTIP",
    "BoundaryPicks",
    "box_from_rectangle",
    "box_outline",
    "region_shapes",
    "coordinate_setting",
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
    "BAND",
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
    "When automated vessel assignment is on, it overrides the other "
    "(manual) inlet/outlet selection methods below."
)

#: The boundary roles, in the order a run assigns them. Taken from the
#: selector's own table rather than restated, so a new role would reach the
#: panel without anything here changing.
ROLES: tuple[str, ...] = tuple(BOUNDARY_ROLE_SETTINGS)

#: Inlet/outlet: greyed when ``automated_vessel_assignment`` is on (large-
#: vessel automatic assignment; ``use_large_vessel_masks`` loads the masks).
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

    Large-vessel automated assignment (``automated_vessel_assignment``)
    replaces manual inlet/outlet picking; small-vessel mask assignment
    (``use_small_vessel_masks_for_boundary_assignment``) replaces manual
    arteriole/venule boundary picking. The sub-tab stays visible but is
    greyed out -- unlike vessel-mask option rows, which hide.
    """
    if role in LARGE_AUTO_ROLES and values.get("automated_vessel_assignment"):
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

#: How many decimal places a picked coordinate keeps. A micron is the unit and
#: a nanometre is far below what a click can mean, so three keeps the config
#: readable without throwing anything away.
DECIMALS = 3


#: What each selection method needs configured beyond itself. A method absent
#: from here needs nothing: `all_degree_1` takes every terminal there is.
METHOD_SETTINGS: Mapping[str, str] = {
    "coordinates": "coordinates",
    "volume": "volume_boxes",
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


#: Every region layer, in role order.
BC_REGION_NAMES = tuple(regions_name(role) for role in ROLES)

#: All of them, for "is this one of the picking layers?".
BC_LAYER_NAMES = frozenset({BC_COORDINATES, *BC_REGION_NAMES})


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


def box_outline(
    corner_a: Sequence[float], corner_b: Sequence[float]
) -> list[np.ndarray]:
    """The twelve edges of a box, as two-point segments.

    A rectangle is the only thing napari can *edit*, and it is flat -- so on
    its own it says nothing about how deep a region is. These draw the box the
    rectangle stands for. They are regenerated from the setting every time, so
    dragging one does nothing lasting; the rectangle is the handle.
    """
    lo = np.minimum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
    hi = np.maximum(np.asarray(corner_a, dtype=float), np.asarray(corner_b, dtype=float))
    corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=float)
    edges = []
    for a, b in itertools.combinations(range(len(corners)), 2):
        # An edge joins two corners that differ along exactly one axis.
        if int(np.count_nonzero(corners[a] != corners[b])) == 1:
            edges.append(np.array([corners[a], corners[b]], dtype=float))
    return edges


#: What a shape in the regions layer is for: the editable rectangle, or one of
#: the twelve segments drawing the box it stands for.
BAND = "band"
HANDLE = "handle"
OUTLINE = "outline"


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
    bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    *,
    only: str | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, np.ndarray]]:
    """Every region as one editable rectangle plus the box it describes.

    A band gets the box and no rectangle: it is not a region anyone typed, it
    is what a percentage works out to, so there is no handle to drag and
    nothing for the settings to read back.
    """
    data: list[np.ndarray] = []
    kinds: list[str] = []
    roles: list[str] = []
    depths: list[float] = []
    parts: list[str] = []
    for role, (lo, hi) in (bands or {}).items():
        if only is not None and role != only:
            continue
        for edge in box_outline(lo, hi):
            data.append(edge)
            kinds.append("line")
            roles.append(role)
            depths.append(abs(float(hi[0]) - float(lo[0])))
            parts.append(BAND)
    for role in ROLES if only is None else (only,):
        for lo, hi in picks.volumes.get(role, ()):
            corners, depth = rectangle_from_box(lo, hi)
            data.append(corners)
            kinds.append("rectangle")
            roles.append(role)
            depths.append(depth)
            parts.append(HANDLE)
            for edge in box_outline(lo, hi):
                data.append(edge)
                kinds.append("line")
                roles.append(role)
                depths.append(depth)
                parts.append(OUTLINE)
    features = {
        "role": np.asarray(roles, dtype=object),
        "depth": np.asarray(depths, dtype=float),
        "part": np.asarray(parts, dtype=object),
    }
    return data, kinds, features


@dataclass(frozen=True)
class BoundaryPicks:
    """What the four roles' coordinate and volume settings currently say."""

    coordinates: Mapping[str, tuple[tuple[float, float, float], ...]]
    volumes: Mapping[str, tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]]
    #: One line per entry that could not be read, for the report box. A panel
    #: must not fall over on a hand-edited config; the run raises for the same
    #: value later, which is the right place for a hard error.
    problems: tuple[str, ...] = field(default=())

    @classmethod
    def from_settings(cls, values: Mapping[str, Any]) -> "BoundaryPicks":
        """Read all four roles out of a settings dict, skipping what will not read."""
        coordinates: dict[str, tuple] = {}
        volumes: dict[str, tuple] = {}
        problems: list[str] = []
        for role in ROLES:
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
        return cls(coordinates, volumes, tuple(problems))

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
        if not parts:
            return "No boundary conditions configured."
        return ", ".join(parts)


def specs_for(values: Mapping[str, Any], bands=None) -> tuple[LayerSpec, ...]:
    """The layers that draw what *values* describes.

    The coordinates layer is emitted even when empty -- it is the surface the
    user clicks into, so it has to exist before there is anything on it. The
    regions layer is not: an empty Shapes layer draws nothing and would only be
    one more row in the layer list until a region is drawn.
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
        shapes, kinds, shape_features = region_shapes(picks, bands, only=role)
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
    return tuple(specs)


def group_for(values: Mapping[str, Any], bands=None) -> StageLayers:
    """The picking layers as a group the panel can hand to `_apply_layers`."""
    picks = BoundaryPicks.from_settings(values)
    note = picks.summary()
    if bands:
        drawn = ", ".join(f"{role} band" for role in bands)
        configured = any(picks.coordinates.values()) or any(picks.volumes.values())
        note = f"{note}, {drawn}" if configured else drawn
    if picks.problems:
        note = f"{note} ({len(picks.problems)} entry could not be read: {picks.problems[0]})"
    return StageLayers(
        stage="boundary_picking",
        title="Boundary conditions",
        layers=specs_for(values, bands),
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
