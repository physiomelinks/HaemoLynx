"""Fill the pericyte mask rows from a mask beside the input image, and keep
them naming the same mask.

Two places read a pericyte mask: the Pericyte measurements box
(``pericyte_measurement_mask_path``) and each perturbation entry that places
pericytes from a mask (its own ``pericyte_mask_path``). A row is *in use* while
the panel shows it: Measure pericytes ticked, or an entry's mask row revealed.

* When a row comes into use, or the input image changes while one is, the
  panel looks in the input image's folder for a mask file whose name contains
  "pericyte" ("pericytes" too: :func:`find_pericyte_mask`) and fills the rows
  in use with it.
* A mask entered into one row by hand fills the other rows in use, over what
  the panel filled them with: the user's mask overrides the one found.
* A row holding a file the user chose is never replaced -- only an empty row,
  one naming no file that exists, or one still holding what the panel filled
  in itself (as :mod:`haemolynx.gui.vessel_mask_search` fills the vessel masks).

Nothing here imports Qt: rows in, the rows to set and the report out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from haemolynx.io.vessel_mask_discovery import file_name_words, list_mask_files

#: The Pericyte measurements box's mask row.
MEASUREMENT_FIELD = "pericyte_measurement_mask_path"
#: The setting a perturbation entry's mask row overrides.
PERTURBATION_SETTING = "pericyte_mask_path"
#: The images a pericyte mask is looked for beside.
REFERENCE_SETTINGS = ("input_path", "raw_image_path")
#: What a pericyte mask's file name contains (any case); "pericytes" does too.
NAME_TERM = "pericyte"

#: The baseline settings whose change can bring a row into use or out of it,
#: or start a search: the measurement's switch, the images, and the gates an
#: entry's mask row reads from the run's own settings.
TRIGGER_SETTINGS = (
    "measure_pericytes",
    MEASUREMENT_FIELD,
    "use_pericyte_mask_constriction",
    "run_haemodynamics",
    *REFERENCE_SETTINGS,
)

_ENTRY_PREFIX = "perturbations["
_ENTRY_SUFFIX = f"].{PERTURBATION_SETTING}"


def perturbation_field(index: int) -> str:
    """The row key of perturbation entry *index*'s mask."""
    return f"{_ENTRY_PREFIX}{int(index)}{_ENTRY_SUFFIX}"


def perturbation_index(name: str) -> int | None:
    """The entry a :func:`perturbation_field` key names, or None for another row."""
    if not (name.startswith(_ENTRY_PREFIX) and name.endswith(_ENTRY_SUFFIX)):
        return None
    try:
        return int(name[len(_ENTRY_PREFIX):-len(_ENTRY_SUFFIX)])
    except ValueError:
        return None


@dataclass(frozen=True)
class FoundPericyteMask:
    """What :func:`find_pericyte_mask` found: the mask, or the files tied for it."""

    path: Path | None = None
    tied: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PericyteMaskFill:
    """What an update puts in the panel."""

    #: Row key -> the mask to set it to.
    values: dict[str, Path] = field(default_factory=dict)
    #: Lines for the panel's report box; empty when there is nothing to say.
    lines: tuple[str, ...] = ()
    #: Rows the user just changed: theirs now, never the panel's to replace.
    released: tuple[str, ...] = ()


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


def _shared_words(first: list[str], second: list[str]) -> tuple[int, int]:
    """Words *first* and *second* share from their start, then in all."""
    start = 0
    for a, b in zip(first, second):
        if a != b:
            break
        start += 1
    return start, len(set(first) & set(second))


def find_pericyte_mask(references: Iterable[Any]) -> FoundPericyteMask:
    """The pericyte mask beside the images *references* (unset ones skipped):
    a mask file (``.tif``, ``.tiff``, ``.h5``) in their folders whose name
    contains :data:`NAME_TERM`, the images themselves left out. Of several,
    the one sharing an image's name furthest from its start, then in most
    words; a tie is left unchosen."""
    images = [Path(reference) for reference in references if reference not in (None, "")]
    candidates = [
        candidate
        for candidate in list_mask_files([image.parent for image in images], exclude=images)
        if NAME_TERM in candidate.stem.lower()
    ]
    if len(candidates) <= 1:
        return FoundPericyteMask(candidates[0] if candidates else None)
    image_words = [file_name_words(image.stem) for image in images]

    def rank(candidate: Path) -> tuple[int, int]:
        words = file_name_words(candidate.stem)
        return max(_shared_words(words, other) for other in image_words)

    ranks = {candidate: rank(candidate) for candidate in candidates}
    best = max(ranks.values())
    chosen = [candidate for candidate, value in ranks.items() if value == best]
    if len(chosen) == 1:
        return FoundPericyteMask(chosen[0])
    return FoundPericyteMask(tied=tuple(chosen))


def _replaceable(name: str, values: Mapping[str, Any], filled: Mapping[str, Any]) -> bool:
    """Whether the panel may set row *name*: it names no file that exists, or
    still holds what the panel filled it with."""
    value = values.get(name)
    if not _names_a_file(value):
        return True
    return name in filled and _path_key(filled[name]) == _path_key(value)


def _row_names(names: Iterable[str]) -> str:
    labels = [
        "the Pericyte measurements mask" if name == MEASUREMENT_FIELD
        else f"perturbation {perturbation_index(name) + 1}'s mask"
        for name in names
    ]
    return ", ".join(labels)


def update_pericyte_mask_rows(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    filled: Mapping[str, Any],
    *,
    references: Iterable[Any] = (),
    references_changed: bool = False,
) -> PericyteMaskFill:
    """The rows to set when the pericyte mask rows in use go from *before* to
    *after* (row key -> its value, for the rows in use only).

    *filled* is what the panel last filled each row with. A row whose value
    changed to anything else was changed by the user: it is released, and a
    mask it now names fills every other row in use that is the panel's to
    replace. Otherwise the rows newly in use -- every row in use once the
    input image changed (*references_changed*) -- are filled with a mask the
    user chose in another row, else with the one found beside *references*."""
    released = tuple(
        name
        for name, value in after.items()
        if name in before
        and _path_key(before[name]) != _path_key(value)
        and not (name in filled and _path_key(filled[name]) == _path_key(value))
    )
    still_filled = {name: value for name, value in filled.items() if name not in released}
    entered = [name for name in released if _names_a_file(after[name])]
    if entered:
        source = Path(after[entered[-1]])
        values = {
            name: source
            for name in after
            if name not in entered
            and _path_key(after[name]) != _path_key(source)
            and _replaceable(name, after, still_filled)
        }
        lines = (f"Set {_row_names(values)} to {source.name}, the pericyte mask just chosen.",) if values else ()
        return PericyteMaskFill(values=values, lines=lines, released=released)

    searched = list(after) if references_changed else [name for name in after if name not in before]
    targets = [name for name in searched if _replaceable(name, after, still_filled)]
    if not targets:
        return PericyteMaskFill(released=released)
    chosen = [
        after[name]
        for name in after
        if _names_a_file(after[name]) and not _replaceable(name, after, still_filled)
    ]
    if chosen:
        source = Path(chosen[0])
        why = "the pericyte mask already chosen"
    else:
        references = [reference for reference in references if reference not in (None, "")]
        if not references:
            return PericyteMaskFill(released=released)
        found = find_pericyte_mask(references)
        if found.tied:
            return PericyteMaskFill(
                lines=(
                    "Several files beside the input image could be the pericyte mask ("
                    + ", ".join(path.name for path in found.tied)
                    + "): choose one.",
                ),
                released=released,
            )
        if found.path is None:
            return PericyteMaskFill(
                lines=(
                    f"No pericyte mask found beside the input image (a .tif, .tiff or .h5 "
                    f"file whose name contains '{NAME_TERM}').",
                ),
                released=released,
            )
        source = found.path
        why = "found beside the input image"
    values = {name: source for name in targets if _path_key(after[name]) != _path_key(source)}
    lines = (f"Set {_row_names(values)} to {source.name}, {why}.",) if values else ()
    return PericyteMaskFill(values=values, lines=lines, released=released)
