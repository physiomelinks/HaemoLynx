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

    #: The kind this row came from, needed to read its value back.
    kind: str = "any"
    #: True when the setting has no default, so an empty box means None.
    nullable: bool = False

    def to_setting_value(self, raw: Any) -> Any:
        """A widget's value as the setting's own value.

        The inverse of what the widget did to it: an empty picker or an empty
        box is *unset*, not the working directory and not zero.
        """
        if isinstance(raw, str) and not raw.strip():
            return None if self.nullable or self.kind == "path" else raw
        if self.kind == "path":
            text = str(raw)
            if text in {"", "."} and self.nullable:
                return None
            return raw
        if self.nullable and self.kind in {"int", "float"} and isinstance(raw, str):
            return int(raw) if self.kind == "int" else float(raw)
        return raw

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


#: Which options each widget accepts. magicgui raises on anything else, so a
#: setting's options follow the widget it actually gets -- not its kind. A
#: float with no default is edited in a LineEdit, which has no min or max.
OPTIONS_BY_WIDGET = {
    "ComboBox": {"choices"},
    "SpinBox": {"min", "max"},
    "FloatSpinBox": {"min", "max", "step"},
    "FileEdit": {"mode", "filter"},
    "CheckBox": set(),
    "LineEdit": set(),
    "LiteralEvalLineEdit": set(),
}


def _options_for(setting: Setting, widget_type: str) -> dict[str, Any]:
    """The magicgui keyword options the widget this setting gets will accept."""
    options: dict[str, Any] = {}
    if setting.kind == "choice":
        options["choices"] = list(setting.choices or ())
    elif setting.kind == "int" and widget_type == "SpinBox":
        low, high = DEFAULT_INT_RANGE
        options["min"] = int(setting.minimum) if setting.minimum is not None else low
        options["max"] = int(setting.maximum) if setting.maximum is not None else high
    elif setting.kind == "float" and widget_type == "FloatSpinBox":
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


#: Widgets that cannot hold "unset". A setting with no default needs one that
#: can, or the panel reports a value nobody chose -- a FloatSpinBox says 0.0
#: and a FileEdit says the working directory, and the schema then warns that a
#: setting is set while the feature reading it is off.
NUMERIC_WIDGETS = {"SpinBox", "FloatSpinBox"}


def widget_type_for(setting: Setting) -> str:
    """The widget that can express this setting, including its unset state."""
    widget = WIDGET_TYPES[setting.kind]
    if setting.default is None and widget in NUMERIC_WIDGETS:
        return "LineEdit"
    return widget


#: Widgets whose empty state is an empty string. `LiteralEvalLineEdit` is not
#: one of them: it parses its text with `ast.literal_eval`, and "" is not a
#: literal, so an empty one raises `SyntaxError: invalid syntax`. It takes None
#: instead, which it renders as "None" and parses straight back.
BLANK_WHEN_UNSET = {"FileEdit", "LineEdit"}


def _display_value(setting: Setting, value: Any, widget_type: str) -> Any:
    """What the widget starts with.

    An unset value shows as an empty box rather than magicgui's fallback --
    the working directory for a path, zero for a number -- which would look
    like a choice somebody made. Whether "empty" is an empty string depends on
    the widget: see :data:`BLANK_WHEN_UNSET`.
    """
    if value is None and widget_type in BLANK_WHEN_UNSET:
        return ""
    return value


def field_for(setting: Setting, value: Any = None) -> Field:
    """The form row for one setting, showing *value* or the schema default."""
    widget_type = widget_type_for(setting)
    return Field(
        name=setting.name,
        label=label_for(setting.name),
        widget_type=widget_type,
        value=_display_value(
            setting, setting.default if value is None else value, widget_type
        ),
        options=_options_for(setting, widget_type),
        help=setting.help + (f" ({setting.unit})" if setting.unit else ""),
        section=setting.section,
        advanced=setting.advanced,
        enabled_by=tuple(setting.requires or ()),
        kind=setting.kind,
        nullable=setting.default is None,
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
    return {field.name: field.to_setting_value(field.value) for field in fields}
