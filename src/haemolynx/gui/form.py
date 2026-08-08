"""Turning the settings schema into a form, without knowing what draws it.

The schema already carries everything a form needs -- a kind, a default, a
range, choices, help text, the section to group under, and the prerequisite
that makes a setting relevant. This module reads that and says what widget each
setting wants; :mod:`haemolynx.gui._widget` builds the actual napari panel from
it.

Keeping the two apart means the mapping is testable without Qt, napari, or a
display, which is most of what there is to get wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from haemolynx.parsers.schema import Schema, Setting, is_prerequisite_met

#: Schema kind -> the magicgui widget that edits it. `any`, `mapping` and the
#: list kinds have no dedicated widget, so they are edited as Python literals:
#: crude, but honest about what the value is, and it round-trips.
WIDGET_TYPES = {
    "bool": "CheckBox",
    "int": "SpinBox",
    "float": "FloatSpinBox",
    "str": "LineEdit",
    "choice": "ComboBox",
    "path": "FileEdit",
    "int_list": "LiteralEvalLineEdit",
    "float_list": "LiteralEvalLineEdit",
    "mapping": "LiteralEvalLineEdit",
    "any": "LiteralEvalLineEdit",
}

#: Spin boxes need bounds. A setting with no declared range still gets these,
#: because magicgui's defaults (0-1000) silently clamp values the schema allows
#: -- `pericyte_constriction_seed` alone defaults to 20240917. The integer
#: bounds are Qt's own limits: QSpinBox is 32-bit, so nothing wider would
#: survive the round trip anyway.
DEFAULT_INT_RANGE = (-(2**31), 2**31 - 1)
DEFAULT_FLOAT_RANGE = (-1e12, 1e12)


@dataclass(frozen=True)
class Field:
    """One row of the form, and everything needed to draw and gate it."""

    name: str
    label: str
    widget_type: str
    value: Any
    options: dict[str, Any]
    help: str
    section: str
    advanced: bool
    #: Prerequisites from the schema, e.g. ``("use_ilastik_segmentation",)`` or
    #: ``("!use_ilastik_segmentation",)``. The form greys the row out until
    #: they hold, rather than hiding it, so a user can see why it is off.
    enabled_by: tuple[str, ...]

    def is_enabled(self, values: Mapping[str, Any]) -> bool:
        """Whether this setting can take effect, given the other values."""
        return all(is_prerequisite_met(rule, values) for rule in self.enabled_by)

    def why_disabled(self, values: Mapping[str, Any]) -> str:
        """A sentence saying which prerequisite is unmet, for the tooltip."""
        unmet = [rule for rule in self.enabled_by if not is_prerequisite_met(rule, values)]
        if not unmet:
            return ""
        parts = [
            f"'{rule[1:]}' is on" if rule.startswith("!") else f"'{rule}' is off"
            for rule in unmet
        ]
        return f"Not used while {' and '.join(parts)}."


def label_for(name: str) -> str:
    """`skeleton_closing_radius` -> `Skeleton closing radius`."""
    return name.replace("_", " ").capitalize()


def _options_for(setting: Setting) -> dict[str, Any]:
    """The magicgui keyword options this setting's widget needs."""
    options: dict[str, Any] = {}
    if setting.kind == "choice":
        options["choices"] = list(setting.choices or ())
    elif setting.kind == "int":
        low, high = DEFAULT_INT_RANGE
        options["min"] = int(setting.minimum) if setting.minimum is not None else low
        options["max"] = int(setting.maximum) if setting.maximum is not None else high
    elif setting.kind == "float":
        low, high = DEFAULT_FLOAT_RANGE
        options["min"] = float(setting.minimum) if setting.minimum is not None else low
        options["max"] = float(setting.maximum) if setting.maximum is not None else high
        options["step"] = None  # let magicgui pick, rather than forcing 1.0
    elif setting.kind == "path":
        # A setting named `*_dir` picks a directory; one that must already
        # exist opens for reading, and one the run will write opens for saving.
        if setting.name.endswith("_dir"):
            options["mode"] = "d"
        elif setting.must_exist:
            options["mode"] = "r"
        else:
            options["mode"] = "w"
    return options


def field_for(setting: Setting, value: Any = None) -> Field:
    """The form row for one setting, showing *value* or the schema default."""
    return Field(
        name=setting.name,
        label=label_for(setting.name),
        widget_type=WIDGET_TYPES[setting.kind],
        value=setting.default if value is None else value,
        options=_options_for(setting),
        help=setting.help + (f" ({setting.unit})" if setting.unit else ""),
        section=setting.section,
        advanced=setting.advanced,
        enabled_by=tuple(setting.requires or ()),
    )


def fields_for(schema: Schema, values: Mapping[str, Any] | None = None) -> list[Field]:
    """Every setting as a form row, in schema order."""
    values = values or {}
    return [field_for(setting, values.get(setting.name)) for setting in schema]


def sections_for(
    schema: Schema, values: Mapping[str, Any] | None = None
) -> dict[str, list[Field]]:
    """Form rows grouped into the sections the schema declares, in order."""
    grouped: dict[str, list[Field]] = {}
    for field in fields_for(schema, values):
        grouped.setdefault(field.section, []).append(field)
    return grouped


def values_from(fields: Sequence[Field]) -> dict[str, Any]:
    """The settings dict these rows describe, ready for `schema.validate`."""
    return {field.name: field.value for field in fields}
