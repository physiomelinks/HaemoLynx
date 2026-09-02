"""A list of perturbations as something you can build up a row at a time.

The `perturbations` setting is a list of ``{name, type, overrides}`` entries: a
nested literal nobody can edit safely in a text box. What makes it awkward is
the same thing that makes it useful -- which keys an entry may carry depends on
its ``type``, and no ``requires`` can say so, because a prerequisite must be a
bool and a type is a choice. So the panel asks for the type first and reveals
that type's options, exactly as the Boundaries tab reveals what a selection
method reads.

Everything here is pure: entries in, entries out, never in place, so the panel
can hold the returned list as its new state and a test can drive the whole
editor with no widgets at all. :mod:`haemolynx.gui._widget` is the only part
that touches Qt.

Two hazards, both silent, both handled at the one boundary:

* **The values must be builtins.** The row that holds this list is a magicgui
  ``LiteralEvalLineEdit``: it stores ``str(value)`` and reads it back with
  ``ast.literal_eval``, and ``repr(np.float64(1.5))`` is
  ``'np.float64(1.5)'``, which raises on the way back in -- as does
  ``yaml.safe_dump`` on the same value. Every entry leaving here has been
  through :func:`~haemolynx.haemodynamics.perturbations.plain`, which is what
  ``PerturbationSpec.to_entry`` does.
* **An override must not reach the flat settings.** `arteriole_diameter_scale`
  is an ordinary row on the Diameters tab, and the panel sends every row to a
  run. An editor here writes into ``entries[i]["overrides"]`` and nowhere else,
  so a perturbation cannot move the baseline it is measured against.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from haemolynx.haemodynamics.perturbations import (
    PERTURBATION_TYPES,
    SETTINGS_FOR_TYPE,
    PerturbationSpec,
    is_usable_as_a_directory_name,
    perturbations_from_settings,
    perturbations_to_settings,
    settings_for_perturbation_type,
)

__all__ = [
    "EDITOR_SETTINGS",
    "PERTURBATION_TYPES",
    "UNCHOSEN",
    "add_entry",
    "default_name",
    "from_settings",
    "hidden_for_type",
    "name_problems",
    "new_entry",
    "remove_entry",
    "rows_for_type",
    "set_name",
    "set_overrides",
    "set_type",
    "summary",
    "to_settings",
]

#: The type an entry has before a user has chosen one, and the one that does
#: nothing. A new row is this, so pressing "+" adds a perturbation that runs
#: nothing until it is told what to be.
UNCHOSEN = "none"


def rows_for_type(perturbation_type: Any) -> tuple[str, ...]:
    """The settings an entry of this type shows, in the order it shows them."""
    return settings_for_perturbation_type(perturbation_type)


def _editor_settings() -> tuple[str, ...]:
    """Every setting any type reads, each once, in declaration order.

    The panel builds one editor per name in this list per entry and hides the
    ones the chosen type does not read, rather than building and destroying
    widgets as the dropdown changes -- which is how a container ends up holding
    a row nobody can see and nothing can reach.
    """
    seen: list[str] = []
    for perturbation_type in PERTURBATION_TYPES:
        for name in SETTINGS_FOR_TYPE[perturbation_type]:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


#: Every setting an entry could show, over all types.
EDITOR_SETTINGS: tuple[str, ...] = _editor_settings()


def hidden_for_type(perturbation_type: Any) -> tuple[str, ...]:
    """The editors an entry of this type hides. Hidden, not greyed out.

    A greyed-out row still says "there is a setting here you cannot have";
    fourteen of them, thirteen greyed, says nothing at all.
    """
    shown = set(rows_for_type(perturbation_type))
    return tuple(name for name in EDITOR_SETTINGS if name not in shown)


def from_settings(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The entries the `perturbations` setting describes, as editable dicts.

    Read leniently, like everything that meets a hand-edited config: a
    malformed entry becomes an editable one rather than an exception, so the
    panel opens and the user can see what is wrong with it.
    """
    return [spec.to_entry() for spec in perturbations_from_settings(values)]


def to_settings(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The `perturbations` setting these entries describe, as builtins.

    The inverse of :func:`from_settings`, and the only way a value leaves this
    module: everything goes through `PerturbationSpec`, so no numpy scalar and
    no `Path` can reach the row that has to survive `literal_eval`.
    """
    return perturbations_to_settings(
        [
            PerturbationSpec.from_entry(entry, index=index)
            for index, entry in enumerate(entries)
        ]
    )


def _normalised(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """*entries* as this module holds them: named, typed, and plain."""
    return from_settings({"perturbations": list(entries)})


def default_name(entries: Sequence[Mapping[str, Any]]) -> str:
    """A name for a new entry that no existing one already has."""
    taken = {str(entry.get("name")) for entry in entries}
    for number in range(1, len(taken) + 2):
        candidate = f"perturbation_{number}"
        if candidate not in taken:
            return candidate
    raise AssertionError("unreachable: one more candidate than there are names")


def new_entry(
    entries: Sequence[Mapping[str, Any]], perturbation_type: str = UNCHOSEN
) -> dict[str, Any]:
    """One fresh entry, named so as not to collide with *entries*."""
    return {
        "name": default_name(entries),
        "type": str(perturbation_type),
        "overrides": {},
    }


def add_entry(
    entries: Sequence[Mapping[str, Any]], perturbation_type: str = UNCHOSEN
) -> list[dict[str, Any]]:
    """*entries* with one more on the end. What the "+" button does."""
    kept = _normalised(entries)
    return kept + [new_entry(kept, perturbation_type)]


def remove_entry(
    entries: Sequence[Mapping[str, Any]], index: int
) -> list[dict[str, Any]]:
    """*entries* without the one at *index*. Out of range removes nothing."""
    kept = _normalised(entries)
    if 0 <= index < len(kept):
        del kept[index]
    return kept


def set_name(
    entries: Sequence[Mapping[str, Any]], index: int, name: str
) -> list[dict[str, Any]]:
    """Rename one entry, which also renames the directory it writes into."""
    return _with(entries, index, name=str(name))


def set_type(
    entries: Sequence[Mapping[str, Any]], index: int, perturbation_type: str
) -> list[dict[str, Any]]:
    """Change one entry's type, dropping the overrides it no longer reads.

    Keeping them would leave a `pressure_sweep` entry carrying an arteriole
    scale that nothing applies -- reported as an unused override, when what
    actually happened is that the user changed their mind.
    """
    keep = set(rows_for_type(perturbation_type))
    kept = _normalised(entries)
    if not 0 <= index < len(kept):
        return kept
    overrides = {
        name: value
        for name, value in (kept[index].get("overrides") or {}).items()
        if name in keep
    }
    return _with(kept, index, type=str(perturbation_type), overrides=overrides)


def set_overrides(
    entries: Sequence[Mapping[str, Any]], index: int, overrides: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Replace one entry's overrides with what its visible editors say.

    Replace rather than merge: the visible editors *are* the entry's
    overrides, so a key that has stopped being visible has stopped being set.
    """
    return _with(entries, index, overrides=dict(overrides))


def _with(
    entries: Sequence[Mapping[str, Any]], index: int, **changes: Any
) -> list[dict[str, Any]]:
    """*entries* with one entry's fields changed, as builtins, out of place."""
    kept = _normalised(entries)
    if not 0 <= index < len(kept):
        return kept
    kept[index] = {**kept[index], **changes}
    # Back through the specs, so a value typed into an editor is made plain
    # before it can reach the row that has to survive `literal_eval`.
    return _normalised(kept)


def name_problems(entries: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """What is wrong with the names, which are also the output directories."""
    problems: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("name") or "")
        if not is_usable_as_a_directory_name(name):
            problems.append(
                f"{name!r} cannot be a directory name, and each perturbation "
                "writes its output into a directory called after it"
            )
        elif name in seen:
            problems.append(f"two perturbations are called {name!r}")
        seen.add(name)
    return tuple(problems)


def summary(entries: Sequence[Mapping[str, Any]]) -> str:
    """One line saying what would run, for the panel's report box."""
    if not entries:
        return "none; the run solves the baseline only"
    described = ", ".join(
        f"{entry.get('name')} ({entry.get('type')})" for entry in entries
    )
    doing = sum(1 for entry in entries if str(entry.get("type")) != UNCHOSEN)
    return f"{len(entries)}, of which {doing} would re-solve: {described}"
