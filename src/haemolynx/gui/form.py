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

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from haemolynx.parsers.schema import (
    PREREQUISITE_EQUALS,
    PREREQUISITE_NOT_EQUALS,
    PREREQUISITE_OFF,
    Schema,
    Setting,
    is_prerequisite_met,
    parse_prerequisite,
    prerequisite_name,
)

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

#: Sections whose gated rows *disappear* until their ``requires`` hold, rather
#: than staying visible and greyed. Input swaps segmented-file vs ilastik
#: children; Vessel masks nests under ``automated_vessel_assignment``;
#: Diameters nests constant vs per-order tables and FWHM under its parents;
#: Statistics nests cell-mask under its parent bool;
#: EDT mask diameter estimate nests under ``use_edt_diameter_crosscheck``
#: (the mask step of the diameter chain, after FWHM or the endothelium and
#: before the table -- see haemodynamics.apply.assign_edge_diameters), with
#: only its FWHM disagreement warning also needing FWHM on;
#: Connectivity/Network Analysis nests entirely under
#: ``statistics`` then its own ``statistics_network_analysis`` toggle, the
#: same two-level pattern -- it groups every graph-theoretic connectivity
#: measure (bridges, loops, centrality, community structure) apart from
#: Statistics' own geometric/morphometric ones.
HIDE_WHEN_UNMET_SECTIONS = frozenset({
    "Input and segmentation",
    "Vessel masks",
    "Diameters and pericytes",
    "FWHM diameter measurement",
    "EDT mask diameter estimate",
    "Endothelial diameter",
    "Statistics and measurements",
    "Connectivity/Network Analysis",
    "Vascular communities",
})

#: Shared across main / large / small ilastik. Declared once under Input;
#: schema ``requires`` cannot OR, so the form hosts them on Input or
#: Boundaries (same setting names / widgets) via :func:`shared_ilastik_host`.
SHARED_ILASTIK_SETTINGS: tuple[str, ...] = (
    "ilastik_executable",
    "ilastik_output_dir",
    "ilastik_output_suffix",
    "ilastik_timeout_seconds",
    "ilastik_reuse_existing_output",
)
SHARED_ILASTIK_SETTING_SET = frozenset(SHARED_ILASTIK_SETTINGS)

#: Parents whose children hide when unmet even outside
#: :data:`HIDE_WHEN_UNMET_SECTIONS` (Graph centreline knobs live under
#: ``Pipeline stages``; Export IDE-plot knobs live under ``Solver and output``;
#: cartwheel hub guard knobs live under their own ``Cartwheel hub guard``
#: section, which is too small a slice of "3. Graph" to earn a whole-section
#: entry in :data:`HIDE_WHEN_UNMET_SECTIONS`; likewise ``run_perturbations``'s
#: one same-tab child in "Perturbation runs" -- the other settings in that
#: section are only ever edited inside a perturbation entry's own typed
#: editor, gated by that entry's type dropdown rather than this mechanism,
#: so they are not listed here). ``cluster_collapse_method`` hides the knob
#: each collapse method alone reads, and ``capillary_block_selection`` the
#: vessels a capillary block's other way of choosing them would read, inside
#: a perturbation entry.
HIDE_WHEN_UNMET_PARENTS = frozenset(
    {
        "smooth_centrelines",
        "use_thick_vessel_skeletonisation",
        "visualize_results",
        "show_plots_in_ide",
        "detect_cartwheel_hub_artifacts",
        "run_perturbations",
        "cluster_collapse_method",
        "capillary_block_selection",
    }
)

#: A dataset-name setting -> the path setting naming the file it is a dataset
#: *of*. A dataset name means nothing unless that file is HDF5, so its row
#: shows only while the path names one.
H5_DATASET_SETTINGS: dict[str, str] = {
    "cell_mask_h5_dataset_name": "cell_mask_path",
    "measurement_3d_vessel_mask_h5_dataset_name": "measurement_3d_vessel_mask_path",
    "measurement_3d_reference_h5_dataset_name": "measurement_3d_reference_image_path",
    "pericyte_mask_h5_dataset_name": "pericyte_mask_path",
}

#: What the loaders read as HDF5 (see haemolynx.io.load).
H5_SUFFIXES = frozenset({".h5", ".hdf5"})


def is_h5_path(value: Any) -> bool:
    """True when *value* names an HDF5 file by its suffix."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and Path(text).suffix.lower() in H5_SUFFIXES


def shared_ilastik_host(values: Mapping[str, Any]) -> str | None:
    """Where the shared ilastik knobs should appear, or ``None`` if nowhere.

    Prefer Input when main-image ilastik is on. When only vessel-mask ilastik
    is on, host them on Boundaries so they are not stranded while Input hides
    them. One schema value each — the panel reparents the same rows.
    """
    if values.get("use_ilastik_segmentation"):
        return "input"
    if values.get("use_ilastik_large_vessel_segmentation") or values.get(
        "use_ilastik_small_vessel_segmentation"
    ):
        return "boundaries"
    return None


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
    #: Prerequisites from the schema, e.g. ``("use_ilastik_segmentation",)``,
    #: ``("!use_ilastik_segmentation",)``, or ``("haematocrit_model=distributed_iterative",)``
    #: for a row that only applies under one value of a ``choice`` setting.
    #: Most sections grey the row out until they hold so a user can see why it
    #: is off. Rows in :data:`HIDE_WHEN_UNMET_SECTIONS` instead *hide* when
    #: unmet — see :meth:`is_visible`.
    enabled_by: tuple[str, ...]

    #: The kind this row came from, needed to read its value back.
    kind: str = "any"
    #: True when the setting has no default, so an empty box means None.
    nullable: bool = False
    #: Hint text for an empty box, e.g. "auto"; never becomes the value.
    #: magicgui's ``create_widget`` does not accept a ``placeholder`` kwarg,
    #: so this is applied to the built widget separately, not through
    #: :attr:`options`.
    placeholder: str | None = None

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
            # A FileEdit built empty reads back as ".", but one *assigned*
            # empty -- which is what opening a config file does -- resolves it
            # to the working directory first, so the blank arrives here looking
            # like a real choice. Every nullable path setting picks a file, and
            # a file picker cannot legitimately yield a directory, so a
            # directory here is that blank rather than something anyone chose.
            if self.nullable and Path(text).is_dir():
                return None
            return raw
        if self.nullable and self.kind in {"int", "float"} and isinstance(raw, str):
            return int(raw) if self.kind == "int" else float(raw)
        return raw

    def is_enabled(self, values: Mapping[str, Any]) -> bool:
        """Whether this setting can take effect, given the other values."""
        return all(is_prerequisite_met(rule, values) for rule in self.enabled_by)

    @property
    def hide_when_unmet(self) -> bool:
        """True when unmet prerequisites should hide this row, not grey it.

        Input, Diameters/FWHM, Vessel-mask, and Statistics options nest under
        parent toggles; showing every greyed child makes those tabs unreadable.
        Shared ilastik knobs, centreline children, thick-vessel children, and
        Export IDE-plot children also hide (hosted or parent-gated). Other
        sections still grey so the reason stays visible.
        """
        if self.name in SHARED_ILASTIK_SETTING_SET:
            return True
        if not self.enabled_by:
            return False
        if self.section in HIDE_WHEN_UNMET_SECTIONS:
            return True
        return any(
            prerequisite_name(rule) in HIDE_WHEN_UNMET_PARENTS
            for rule in self.enabled_by
        )

    def is_visible(self, values: Mapping[str, Any]) -> bool:
        """Whether this row should appear given the other values.

        Shared ilastik rows use Input-tab visibility here (hosted on Input).
        Boundaries hosting is layered on by :func:`visible_vessel_mask_settings`.
        An HDF5 dataset-name row (:data:`H5_DATASET_SETTINGS`) also needs the
        file it names a dataset of to be HDF5.
        """
        if self.name in SHARED_ILASTIK_SETTING_SET:
            return shared_ilastik_host(values) == "input"
        pair = H5_DATASET_SETTINGS.get(self.name)
        if pair is not None and not is_h5_path(values.get(pair)):
            return False
        if self.hide_when_unmet:
            return self.is_enabled(values)
        return True

    def why_disabled(self, values: Mapping[str, Any]) -> str:
        """A sentence saying which prerequisite is unmet, for the tooltip."""
        unmet = [rule for rule in self.enabled_by if not is_prerequisite_met(rule, values)]
        if not unmet:
            return ""
        parts = []
        for rule in unmet:
            name, test, expected = parse_prerequisite(rule)
            if test == PREREQUISITE_EQUALS:
                parts.append(f"'{name}' is not '{expected}'")
            elif test == PREREQUISITE_NOT_EQUALS:
                parts.append(f"'{name}' is '{expected}'")
            elif test == PREREQUISITE_OFF:
                parts.append(f"'{name}' is on")
            else:
                parts.append(f"'{name}' is off")
        return f"Not used while {' and '.join(parts)}."


def prerequisite_chain(schema: Schema, name: str) -> tuple[str, ...]:
    """Every ``requires`` rule *name* depends on: its own, then its gates' own.

    A setting's ``requires`` names only its nearest gates -- the FWHM decoy
    check's sample size names the decoy check, not FWHM itself -- but it
    takes effect only while every gate above those is met too. The panel
    reads the whole chain, so a row never lingers after a toggle further up
    has hidden the row it hangs under.
    """
    chain: list[str] = []
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop(0)
        if current in seen or current not in schema:
            continue
        seen.add(current)
        for rule in schema[current].requires:
            if rule not in chain:
                chain.append(rule)
            pending.append(prerequisite_name(rule))
    return tuple(chain)


def _visible_settings_in_section(
    schema: Schema, values: Mapping[str, Any], section: str
) -> set[str]:
    """Setting names in *section* that should appear for *values*."""
    shown: set[str] = set()
    for setting in schema:
        if setting.section != section:
            continue
        field = field_for(setting, values.get(setting.name))
        if field.is_visible(values):
            shown.add(field.name)
    return shown


def visible_vessel_mask_settings(
    schema: Schema, values: Mapping[str, Any]
) -> set[str]:
    """Vessel-mask setting names that should appear for *values*.

    The root ``automated_vessel_assignment`` toggle is always included; every
    other Vessel masks row follows its ``requires`` chain. Shared ilastik
    executable/output knobs join in when :func:`shared_ilastik_host` is
    ``"boundaries"`` (same setting names as on Input).
    """
    shown = _visible_settings_in_section(schema, values, "Vessel masks")
    if shared_ilastik_host(values) == "boundaries":
        shown.update(SHARED_ILASTIK_SETTING_SET)
    return shown


def visible_input_segmentation_settings(
    schema: Schema, values: Mapping[str, Any]
) -> set[str]:
    """Input-tab setting names that should appear for *values*.

    Ungated rows (toggle, voxel metadata) stay; ``input_path`` vs main-ilastik
    children follow ``use_ilastik_segmentation``. Shared executable/output knobs
    appear only while hosted on Input (see :func:`shared_ilastik_host`).
    """
    return _visible_settings_in_section(schema, values, "Input and segmentation")


def visible_graph_centreline_settings(
    schema: Schema, values: Mapping[str, Any]
) -> set[str]:
    """Graph-tab centreline setting names that should appear for *values*.

    ``smooth_centrelines`` stays whenever it is shown; its method / iterations /
    max-deviation children hide until smoothing is on.
    """
    names = (
        "smooth_centrelines",
        "centreline_smoothing_method",
        "centreline_smoothing_iterations",
        "centreline_max_deviation",
    )
    shown: set[str] = set()
    for name in names:
        setting = schema[name]
        field = field_for(setting, values.get(name))
        if field.is_visible(values):
            shown.add(name)
    return shown


def visible_diameter_settings(
    schema: Schema, values: Mapping[str, Any]
) -> set[str]:
    """Diameters-tab diameter + FWHM + EDT setting names that should appear.

    Parent toggles (``all_diams_const``, ``use_fwhm_edge_diameters``,
    ``use_edt_diameter_crosscheck``, …) stay; gated children follow their
    ``requires`` chains. EDT mask diameter estimate nests two levels under
    ``use_fwhm_edge_diameters`` then ``use_edt_diameter_crosscheck``.
    """
    return (
        _visible_settings_in_section(schema, values, "Diameters and pericytes")
        | _visible_settings_in_section(schema, values, "FWHM diameter measurement")
        | _visible_settings_in_section(schema, values, "EDT mask diameter estimate")
        | _visible_settings_in_section(schema, values, "Endothelial diameter")
    )


def visible_statistics_settings(
    schema: Schema, values: Mapping[str, Any]
) -> set[str]:
    """Export-tab Statistics + Connectivity/Network Analysis setting names
    that should appear for *values*.

    Parent toggles (``statistics``, ``measurement_3d_to_cell_mask``) stay;
    gated children follow their ``requires`` chains. Under
    ``measurement_3d_to_cell_mask`` that means ``cell_mask_path``,
    ``cell_mask_h5_dataset_name``, ``measurement_3d_vessel_mask_path``,
    ``measurement_3d_vessel_mask_h5_dataset_name``,
    ``measurement_3d_reference_image_path``, and
    ``measurement_3d_reference_h5_dataset_name``. Connectivity/Network
    Analysis (every graph-theoretic connectivity measure: bridges, loops,
    centrality, community structure) nests two levels under ``statistics``
    then its own ``statistics_network_analysis`` toggle, the same pattern
    :func:`visible_diameter_settings` uses for EDT under FWHM.
    """
    return _visible_settings_in_section(
        schema, values, "Statistics and measurements"
    ) | _visible_settings_in_section(schema, values, "Connectivity/Network Analysis")


#: Export-tab IDE plot rows: schema keys stay snake_case for YAML.
SETTING_ROW_LABELS: dict[str, str] = {
    "visualize_results": "Produce IDE plots",
    "show_plots_in_ide": "Show plots in IDE",
    "ide_plot_mode": "IDE plot mode",
    "hold_ide_plots_open": "Hold IDE plots open",
    "use_thick_vessel_skeletonisation": "Use alternate skeletonisation for thick vessels",
    "use_memmap_loading": "Low RAM option for large networks (Slow)",
    "measurement_3d_to_cell_mask": "Measure distance between vessels and 3D object mask",
}

#: Display text for a ``choice`` setting's values, keyed by setting name then
#: raw value. The stored/YAML value stays a plain snake_case identifier, like
#: every other choice setting -- only the dropdown's own visible text reads
#: as a sentence for the ones a user picks a whole model from, rather than a
#: mode word like "pries" or "anatomical".
CHOICE_VALUE_LABELS: dict[str, dict[str, str]] = {
    "haematocrit_model": {
        "fixed": "Fixed haematocrit",
        "distributed_iterative": "Calculated haematocrit distribution - iterative",
    },
}


def label_for(name: str, unit: str | None = None) -> str:
    """`skeleton_closing_radius` -> `Skeleton closing radius (voxels)`.

    The unit belongs on the label rather than only in the tooltip. This
    pipeline measures some lengths in voxels and some in microns, 10 is a
    reasonable value for either, and a tooltip is only read by someone who
    already suspects there is something to check.

    Spelled exactly as the schema spells it, so the row and the config file
    agree -- a GUI that says "µm" where the file says "um" reads as two units.
    IDE plot rows use :data:`SETTING_ROW_LABELS` so "IDE" stays capitalized.
    """
    label = SETTING_ROW_LABELS.get(name) or name.replace("_", " ").capitalize()
    return f"{label} ({unit})" if unit else label


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


def _options_for(setting: Setting, widget_type: str, value: Any = None) -> dict[str, Any]:
    """The magicgui keyword options the widget this setting gets will accept."""
    options: dict[str, Any] = {}
    if setting.name in CHANNEL_SETTINGS:
        # Until the panel reads the file it names, only the current channel.
        options["choices"] = channel_choices([], value)
    elif setting.kind == "choice":
        labels = CHOICE_VALUE_LABELS.get(setting.name)
        options["choices"] = (
            [(labels.get(c, c), c) for c in (setting.choices or ())]
            if labels
            else list(setting.choices or ())
        )
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


#: Settings that pick one channel of the image another setting names: the
#: panel shows them as a drop-down of that file's channels, Fiji-named (C1,
#: C2, ...), where a number box would leave the user to count from 0.
CHANNEL_SETTINGS = {
    "endothelial_channel": "endothelial_image_path",
    "fwhm_raw_channel": "fwhm_raw_tiff_path",
}

#: The drop-down's "no channel" entry: a single-channel file has nothing to
#: pick, and a multi-channel one has to be told which.
NO_CHANNEL_SINGLE = "single-channel file"
NO_CHANNEL_CHOOSE = "choose a channel"


def channel_choices(channels, current: int | None = None) -> list[tuple[str, int | None]]:
    """``(label, value)`` entries for a channel drop-down.

    *channels* is what :func:`haemolynx.io.tiff_channels` read from the file
    (empty for a single-channel one, or none chosen yet). A channel the file
    does not have stays listed when it is the current one -- marked, so the
    mistake is visible rather than silently changed to another channel.
    """
    entries: list[tuple[str, int | None]] = [
        (NO_CHANNEL_CHOOSE if channels else NO_CHANNEL_SINGLE, None)
    ]
    entries += [(channel.label, channel.index) for channel in channels]
    known = {channel.index for channel in channels}
    if current is not None and int(current) not in known:
        label = f"C{int(current) + 1}"
        entries.append((f"{label} (not in this file)" if channels else label, int(current)))
    return entries


#: Widgets that cannot hold "unset". A setting with no default needs one that
#: can, or the panel reports a value nobody chose -- a FloatSpinBox says 0.0
#: and a FileEdit says the working directory, and the schema then warns that a
#: setting is set while the feature reading it is off.
NUMERIC_WIDGETS = {"SpinBox", "FloatSpinBox"}


def widget_type_for(setting: Setting) -> str:
    """The widget that can express this setting, including its unset state."""
    if setting.name in CHANNEL_SETTINGS:
        return "ComboBox"
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


def display_value_for(setting: Setting, value: Any) -> Any:
    """What *value* has to become before a built widget will take it.

    :func:`field_for` puts every value through :func:`_display_value` when it
    builds a row. Anything writing to a row afterwards -- opening a config file
    into an open form -- has to do the same, or an unset value arrives as None
    at a widget that raises on it.
    """
    return _display_value(setting, value, widget_type_for(setting))


def field_for(setting: Setting, value: Any = None) -> Field:
    """The form row for one setting, showing *value* or the schema default."""
    widget_type = widget_type_for(setting)
    return Field(
        name=setting.name,
        label=label_for(setting.name, setting.unit),
        widget_type=widget_type,
        value=_display_value(
            setting, setting.default if value is None else value, widget_type
        ),
        options=_options_for(
            setting, widget_type, setting.default if value is None else value
        ),
        help=setting.help + (f" ({setting.unit})" if setting.unit else ""),
        section=setting.section,
        advanced=setting.advanced,
        enabled_by=tuple(setting.requires or ()),
        kind=setting.kind,
        nullable=setting.default is None,
        placeholder=setting.placeholder,
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
