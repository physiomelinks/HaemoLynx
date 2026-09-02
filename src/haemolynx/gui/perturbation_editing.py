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
* **An override must not reach the flat settings.** `arteriole_diameter_change_percent`
  is also declared for the Diameters / Perturbation schema, and the panel sends
  every ordinary row to a run. An editor here writes into
  ``entries[i]["overrides"]`` and nowhere else, so a perturbation cannot move
  the baseline it is measured against.
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
    "ADD_TOOLTIP",
    "ALWAYS_VISIBLE_TAB_SETTINGS",
    "EDITOR_SETTINGS",
    "NAME_TOOLTIP",
    "PERTURBATION_TYPES",
    "PERTURBATION_TYPE_DISPLAY_NAMES",
    "REMOVE_TOOLTIP",
    "SETTING_DISPLAY_LABELS",
    "TYPE_TOOLTIP",
    "UNCHOSEN",
    "add_entry",
    "default_name",
    "display_label_for_setting",
    "display_name_for_type",
    "editor_layout_order",
    "from_settings",
    "hidden_for_type",
    "name_problems",
    "new_entry",
    "perturbation_type_choices",
    "remove_entry",
    "rows_for_type",
    "set_name",
    "set_overrides",
    "set_type",
    "summary",
    "to_settings",
    "visible_tab_settings",
]

#: The type an entry has before a user has chosen one, and the one that does
#: nothing. A new row is this, so pressing "+" adds a perturbation that runs
#: nothing until it is told what to be.
UNCHOSEN = "none"

#: GUI / report wording for each API type key. Identifiers in
#: :data:`PERTURBATION_TYPES` stay stable for configs; the dropdown and
#: summaries show these labels. Diameter-percent types say
#: "constriction/dilation" because the same control narrows or widens.
PERTURBATION_TYPE_DISPLAY_NAMES: Mapping[str, str] = {
    "none": "none",
    "pressure_sweep": "pressure sweep",
    "pressure_and_pericyte_sweep": (
        "pressure and pericyte constriction/dilation sweep"
    ),
    "pericyte_dilation_sweep": "pericyte constriction/dilation sweep",
    "arteriole_diameter_change": "arteriole constriction/dilation",
    "arteriole_diameter_sweep": "arteriole constriction/dilation sweep",
    "pressure_and_arteriole_sweep": (
        "pressure and arteriole constriction/dilation sweep"
    ),
    "capillary_diameter_sweep": "capillary constriction/dilation sweep",
    "pressure_and_capillary_sweep": (
        "pressure and capillary constriction/dilation sweep"
    ),
    "pericyte_spacing_sweep": "pericyte spacing sweep",
    "pericyte_length_sweep": "pericyte length sweep",
    "pericyte_diameter_change": "pericyte constriction/dilation",
    "arteriole_and_pericyte_diameter_change": (
        "arteriole and pericyte constriction/dilation"
    ),
}

#: GUI row labels for bidirectional diameter / tone settings. Schema ``name``
#: keys and plot file stems stay as-is; only what the panel prints changes.
SETTING_DISPLAY_LABELS: Mapping[str, str] = {
    # Leading "%" so the combined arteriole+pericyte entry's first knob is
    # obviously the whole-branch percent, not another pericyte tone control.
    "arteriole_diameter_change_percent": "Arteriole % constriction/dilation",
    "arteriole_dilation_min_percent": "Arteriole constriction/dilation min",
    "arteriole_dilation_max_percent": "Arteriole constriction/dilation max",
    "arteriole_dilation_step_percent": "Arteriole constriction/dilation step",
    "capillary_dilation_min_percent": "Capillary constriction/dilation min",
    "capillary_dilation_max_percent": "Capillary constriction/dilation max",
    "capillary_dilation_step_percent": "Capillary constriction/dilation step",
    "pericyte_dilation_min_percent": "Pericyte constriction/dilation min",
    "pericyte_dilation_max_percent": "Pericyte constriction/dilation max",
    "pericyte_dilation_step_percent": "Pericyte constriction/dilation step",
    "pericyte_geometry_dilation_percent": (
        "Pericyte geometry constriction/dilation"
    ),
    "pericyte_constriction_factor": "Pericyte constriction/dilation factor",
    "constriction_by_branch_order": "Constriction/dilation by branch order",
}

#: Hover / focus text for controls that are not schema settings. Setting-row
#: tooltips come from ``Setting.help`` (plus unit) via :func:`haemolynx.gui.form.field_for`.
NAME_TOOLTIP = (
    "Also the directory this perturbation's output goes in, so it has "
    "to be unique and usable as a directory name"
)
TYPE_TOOLTIP = (
    "What to change before re-solving (diameter types use constriction/"
    "dilation wording). Choosing one shows its options"
)
ADD_TOOLTIP = (
    "Add another perturbation that re-solves from the same baseline; "
    "pick its type to reveal the settings it overrides"
)
REMOVE_TOOLTIP = (
    "Remove this perturbation from the list; later entries keep their "
    "settings and shift up"
)


def display_name_for_type(perturbation_type: Any) -> str:
    """Human label for a perturbation type key (API string unchanged)."""
    text = str(perturbation_type)
    return PERTURBATION_TYPE_DISPLAY_NAMES.get(text, text.replace("_", " "))


def display_label_for_setting(name: str, unit: str | None = None) -> str:
    """Form-row label for a setting, with constriction/dilation wording.

    Falls back to capitalising the snake_case name when *name* has no entry in
    :data:`SETTING_DISPLAY_LABELS`, matching :func:`haemolynx.gui.form.label_for`.
    """
    label = SETTING_DISPLAY_LABELS.get(name) or name.replace("_", " ").capitalize()
    return f"{label} ({unit})" if unit else label


def perturbation_type_choices() -> list[tuple[str, str]]:
    """``(display label, API value)`` pairs for the type ComboBox."""
    return [(display_name_for_type(name), name) for name in PERTURBATION_TYPES]


#: Ordinary form rows on the Perturbations tab. Everything else under
#: "Perturbation runs" is either a type's option (revealed only when that type
#: is chosen) or a leftover always-on sweep flag the whole-brain example still
#: reads from its config -- neither belongs as a permanent tab row.
ALWAYS_VISIBLE_TAB_SETTINGS: tuple[str, ...] = (
    "run_perturbations",
    "perturbations",
    "perturbation_output_dir",
)


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


def editor_layout_order(perturbation_type: Any) -> tuple[str, ...]:
    """Widget order for one entry: this type's rows first, then the rest.

    Editors are built once for every type (see :data:`EDITOR_SETTINGS`), but
    laying them out in that global order buries a type's own knobs -- the
    combined arteriole+pericyte type declares ``arteriole_diameter_change_percent``
    first in :data:`SETTINGS_FOR_TYPE`, yet global order put six pericyte
    geometry rows above it. Hidden editors stay after the visible ones so
    reveal/hide on type change does not recreate widgets.
    """
    shown = rows_for_type(perturbation_type)
    shown_set = set(shown)
    rest = tuple(name for name in EDITOR_SETTINGS if name not in shown_set)
    return shown + rest


def visible_tab_settings(names: Sequence[str]) -> tuple[str, ...]:
    """*names* kept as ordinary Perturbations-tab rows, in the given order.

    The stage still *claims* the whole Perturbation-runs section so Field
    objects exist for the editor to clone, but the dilation/pressure ranges
    and the brain-script sweep flag must not sit on the tab as always-on
    controls -- they appear only when a ``pericyte_dilation_sweep`` entry is
    chosen (or in the brain example's own config).
    """
    keep = set(ALWAYS_VISIBLE_TAB_SETTINGS)
    return tuple(name for name in names if name in keep)


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

    Keeping them would leave a `pericyte_dilation_sweep` entry carrying an
    arteriole scale that nothing applies -- reported as an unused override,
    when what actually happened is that the user changed their mind.
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
        f"{entry.get('name')} ({display_name_for_type(entry.get('type'))})"
        for entry in entries
    )
    doing = sum(1 for entry in entries if str(entry.get("type")) != UNCHOSEN)
    return f"{len(entries)}, of which {doing} would re-solve: {described}"
