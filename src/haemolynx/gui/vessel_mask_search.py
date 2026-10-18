"""Fill the vessel-mask rows from the masks found beside the input image.

When a vessel-mask role comes into use -- "Use large vessel masks", or the
small masks' switch, ticked with ``automated_vessel_assignment`` on -- or the
input image changes while one is in use, the panel looks in the input image's
folder for that role's masks (`io.vessel_mask_discovery.find_vessel_masks`)
and fills its mask rows. A row holding a file the user chose is never
replaced: only an empty row, one naming no file that exists, or one still
holding what the panel filled in itself. A role whose masks ilastik makes is
not searched, since its mask rows are not read.

Nothing here imports Qt: settings in, the rows to set and the report out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from haemolynx.graph.boundaries import vessel_masks_in_use
from haemolynx.io.automated_vessel_assignment import VESSEL_MASK_SETTINGS
from haemolynx.io.vessel_mask_discovery import FoundVesselMasks, find_vessel_masks

#: The images a run's masks are looked for beside: its segmented image, then
#: its raw image (with ilastik segmenting, the segmented image is not there yet).
REFERENCE_SETTINGS = ("input_path", "raw_image_path")

#: The settings whose change can start a search.
TRIGGER_SETTINGS = (
    "automated_vessel_assignment",
    *(
        names[key]
        for names in VESSEL_MASK_SETTINGS.values()
        for key in ("enabled", "use_ilastik")
    ),
    *REFERENCE_SETTINGS,
)


@dataclass(frozen=True)
class MaskFill:
    """What a search puts in the panel."""

    #: Mask-path settings to set, each to a path or None (an emptied row).
    values: dict[str, Path | None] = field(default_factory=dict)
    #: Lines for the panel's report box; empty when there is nothing to say.
    lines: tuple[str, ...] = ()


def _path_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(Path(value))
    except TypeError:
        return str(value)


def _names_a_file(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return Path(value).is_file()
    except (OSError, TypeError, ValueError):
        return False


def roles_searched(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The mask roles (``"large"``, ``"small"``) whose rows a search fills:
    in use (`graph.vessel_masks_in_use`), and read from files, not ilastik."""
    return tuple(
        role
        for role, names in VESSEL_MASK_SETTINGS.items()
        if vessel_masks_in_use(settings, role) and not settings.get(names["use_ilastik"])
    )


def roles_to_search(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    """The roles a change of the settings from *before* to *after* searches for:
    each role newly in use, or every role in use once an input image changed."""
    now = roles_searched(after)
    if any(_path_key(before.get(name)) != _path_key(after.get(name)) for name in REFERENCE_SETTINGS):
        return now
    was = set(roles_searched(before))
    return tuple(role for role in now if role not in was)


def _replaceable(name: str, settings: Mapping[str, Any], filled_before: Mapping[str, Any]) -> bool:
    """Whether a search may set row *name*: it names no file that exists, or
    still holds what the panel filled it with."""
    value = settings.get(name)
    if not _names_a_file(value):
        return True
    return name in filled_before and _path_key(filled_before[name]) == _path_key(value)


def _found_lines(found: FoundVesselMasks, settings: Mapping[str, Any]) -> list[str]:
    size = found.size
    names = VESSEL_MASK_SETTINGS[size]
    if found.is_pair:
        return [
            f"Found the {size} vessel masks beside the input image: arteriole "
            f"{found.arteriole.name}, venule {found.venule.name}."
        ]
    if found.global_mask is not None:
        return [
            f"Found a {size} vessel mask beside the input image with no vessel type "
            f"in its name, {found.global_mask.name}: read as a global mask of every "
            f"{size} vessel."
        ]
    if found.arteriole is not None:
        line = f"Found a {size} arteriole mask beside the input image, {found.arteriole.name}."
        if settings.get(names["venule_mask_path"]) in (None, ""):
            line += (
                f" No {size} venule mask was found, so with the venule row empty the "
                f"run reads it as a global mask of every {size} vessel."
            )
        return [line]
    return [
        f"Found a {size} venule mask beside the input image, {found.venule.name}, "
        f"but no {size} arteriole mask to go with it."
    ]


def fill_vessel_mask_rows(
    settings: Mapping[str, Any],
    roles: tuple[str, ...],
    filled_before: Mapping[str, Any] | None = None,
) -> MaskFill:
    """Look beside the input image for each of *roles*' masks, and say which
    mask rows to fill.

    *filled_before* is what earlier searches filled each row with: a row still
    holding it is the panel's to replace (a new input image's masks replace
    the last image's), while any other file in a row is the user's and keeps
    its whole role as it is."""
    if not roles:
        return MaskFill()
    filled_before = filled_before or {}
    references = [settings.get(name) for name in REFERENCE_SETTINGS]
    references = [reference for reference in references if reference not in (None, "")]
    sizes = " and ".join(roles)
    if not references:
        return MaskFill(lines=(f"No input image to look beside for the {sizes} vessel masks.",))

    values: dict[str, Path | None] = {}
    lines: list[str] = []
    for role, found in find_vessel_masks(references, roles).items():
        proposed = found.settings()
        kept = [
            name
            for name, value in proposed.items()
            if _path_key(value) != _path_key(settings.get(name))
            and not _replaceable(name, settings, filled_before)
        ]
        if kept:
            lines.append(
                f"Kept the {role} vessel masks already chosen rather than those "
                f"found beside the input image ("
                + ", ".join(path.name for path in proposed.values() if path is not None)
                + ")."
            )
        elif proposed:
            values.update(proposed)
            lines.extend(_found_lines(found, {**settings, **proposed}))
            if not found.size_named:
                lines.append(
                    "No file beside it says 'large' or 'small', so the masks naming "
                    "only a vessel type are read as the large ones."
                )
        elif not found.ambiguous:
            lines.append(
                f"No {role} vessel masks found beside the input image (a mask's file "
                f"name is the image's name with '{role}' added, and 'arteriole' or "
                f"'venule' unless it is a global mask"
                + ("; 'arteriole' or 'venule' alone counts as large)." if role == "large" else ").")
            )
        for kind, tied in found.ambiguous.items():
            lines.append(
                f"Several files could be the {role} {kind} mask ("
                + ", ".join(path.name for path in tied)
                + "): choose one."
            )
    return MaskFill(values=values, lines=tuple(lines))
