"""The schema-to-form mapping, checked without a GUI.

`Schema.describe()` was built so a form could be generated rather than
hand-written; this is the first thing to use it. The mapping is pure, so the
part most likely to be wrong -- a setting with no widget, a range that silently
clamps a legal value, a prerequisite that greys out the wrong row -- is
testable with no napari, no Qt and no display.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from haemolynx.gui.form import (
    DEFAULT_FLOAT_RANGE,
    DEFAULT_INT_RANGE,
    OPTIONS_BY_WIDGET,
    WIDGET_TYPES,
    Field,
    field_for,
    fields_for,
    label_for,
    sections_for,
    values_from,
)
from haemolynx.parsers import Schema, Setting
from haemolynx.pipeline import default_schema

SCHEMA = default_schema()


# --- every setting reaches the form -----------------------------------------


def test_every_setting_becomes_a_row():
    """A setting with no row is a setting a GUI user cannot reach."""
    names = {field.name for field in fields_for(SCHEMA)}
    assert names == set(SCHEMA.names)


def test_every_schema_kind_has_a_widget():
    """A new kind must be given a widget, not silently dropped."""
    kinds = {setting.kind for setting in SCHEMA}
    missing = kinds - set(WIDGET_TYPES)
    assert not missing, f"schema kinds with no widget: {sorted(missing)}"


def test_rows_are_grouped_into_the_schema_sections():
    grouped = sections_for(SCHEMA)
    assert list(grouped) == list(dict.fromkeys(s.section for s in SCHEMA))
    assert sum(len(rows) for rows in grouped.values()) == len(SCHEMA.names)


def test_the_defaults_the_form_starts_with_are_valid_settings():
    """Opening the panel and pressing run must not fail validation."""
    values = values_from(fields_for(SCHEMA))
    validated = SCHEMA.validate(values)
    assert set(validated) == set(SCHEMA.names)


def test_supplied_values_win_over_the_defaults():
    fields = fields_for(SCHEMA, {"min_stub_length": 42.0})
    by_name = {field.name: field for field in fields}
    assert by_name["min_stub_length"].value == 42.0
    assert by_name["cluster_collapse_distance"].value == SCHEMA["cluster_collapse_distance"].default


# --- what each widget is told ------------------------------------------------


def test_a_choice_setting_offers_exactly_its_choices():
    setting = next(s for s in SCHEMA if s.kind == "choice")
    field = field_for(setting)
    assert field.widget_type == "ComboBox"
    assert field.options["choices"] == list(setting.choices)


def test_a_declared_range_reaches_the_spin_box():
    field = field_for(
        Setting("count", "int", 3, "How many", "S", minimum=1, maximum=9)
    )
    assert (field.options["min"], field.options["max"]) == (1, 9)


def test_an_undeclared_range_is_wide_rather_than_absent():
    """magicgui's own default is 0-1000, which would clamp legal values."""
    field = field_for(Setting("count", "int", 3, "How many", "S"))
    assert (field.options["min"], field.options["max"]) == DEFAULT_INT_RANGE

    field = field_for(Setting("size", "float", 3.0, "How big", "S"))
    assert (field.options["min"], field.options["max"]) == DEFAULT_FLOAT_RANGE


def test_no_spin_box_can_clamp_a_value_the_schema_allows():
    """Every numeric default must sit inside the bounds the form gives it."""
    for field in fields_for(SCHEMA):
        if field.widget_type not in {"SpinBox", "FloatSpinBox"}:
            continue
        if field.value is None:
            continue
        assert field.options["min"] <= field.value <= field.options["max"], (
            f"{field.name}: default {field.value} is outside the widget range "
            f"{field.options['min']}..{field.options['max']}"
        )


@pytest.mark.parametrize(
    "name,must_exist,expected_mode",
    [
        ("output_dir", False, "d"),   # a directory, whether or not it exists
        ("plot_dir", True, "d"),
        ("input_path", True, "r"),    # must already be there: open it
        ("report_path", False, "w"),  # the run writes it: save dialogue
    ],
)
def test_a_path_opens_the_right_kind_of_dialogue(name, must_exist, expected_mode):
    field = field_for(Setting(name, "path", None, "A path", "S", must_exist=must_exist))
    assert field.widget_type == "FileEdit"
    assert field.options["mode"] == expected_mode


def test_the_unit_is_shown_with_the_help_text():
    field = field_for(Setting("length", "float", 1.0, "How long", "S", unit="um"))
    assert field.help == "How long (um)"


def test_labels_read_as_words():
    assert label_for("skeleton_closing_radius") == "Skeleton closing radius"


def test_a_label_carries_the_unit_so_the_row_itself_says_it():
    """A tooltip is only read by someone who already suspects a problem.

    Some lengths here are voxels and some are microns, and 10 is a reasonable
    value for either, so the row has to say which without being hovered.
    """
    assert label_for("min_stub_length", "um") == "Min stub length (um)"
    assert label_for("skeleton_bridge_gap_size", "voxels") == (
        "Skeleton bridge gap size (voxels)"
    )


def test_a_label_with_no_unit_gains_no_brackets():
    assert label_for("do_skeletonize", None) == "Do skeletonize"
    assert label_for("do_skeletonize", "") == "Do skeletonize"


def test_the_row_for_a_setting_with_a_unit_is_labelled_with_it():
    field = field_for(Setting("length", "float", 1.0, "How long", "S", unit="um"))
    assert field.label == "Length (um)"


def test_every_setting_that_declares_a_unit_shows_it_on_its_row():
    """The whole point, across the real schema rather than one example."""
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    for field in fields_for(schema):
        unit = schema[field.name].unit
        if unit:
            assert field.label.endswith(f" ({unit})"), (
                f"{field.name} declares {unit!r} but its row reads {field.label!r}"
            )


def test_the_label_spells_the_unit_the_way_the_config_file_does():
    """"µm" on the row and "um" in the file reads as two different units."""
    from haemolynx.pipeline import default_schema

    for field in fields_for(default_schema()):
        assert "µ" not in field.label, field.label


# --- prerequisites gate the row ---------------------------------------------


def _gated_schema() -> Schema:
    return Schema(
        [
            Setting("use_ilastik", "bool", False, "Segment with ilastik", "S"),
            Setting("classifier", "path", None, "The trained classifier", "S",
                    requires=("use_ilastik",)),
            Setting("mask", "path", None, "An existing mask", "S",
                    requires=("!use_ilastik",)),
        ]
    )


def test_a_row_is_enabled_only_when_its_prerequisite_holds():
    fields = {field.name: field for field in fields_for(_gated_schema())}

    off = {"use_ilastik": False}
    assert not fields["classifier"].is_enabled(off)
    assert fields["mask"].is_enabled(off)

    on = {"use_ilastik": True}
    assert fields["classifier"].is_enabled(on)
    assert not fields["mask"].is_enabled(on)


def test_a_disabled_row_says_which_setting_disabled_it():
    fields = {field.name: field for field in fields_for(_gated_schema())}

    assert fields["classifier"].why_disabled({"use_ilastik": False}) == (
        "Not used while 'use_ilastik' is off."
    )
    assert fields["mask"].why_disabled({"use_ilastik": True}) == (
        "Not used while 'use_ilastik' is on."
    )


def test_an_enabled_row_gives_no_reason():
    fields = {field.name: field for field in fields_for(_gated_schema())}
    assert fields["classifier"].why_disabled({"use_ilastik": True}) == ""


def test_a_row_with_no_prerequisite_is_always_enabled():
    field = field_for(Setting("always", "bool", True, "On", "S"))
    assert field.enabled_by == ()
    assert field.is_enabled({})


def test_fields_are_immutable():
    """The form reads the schema; it must not be able to edit it by accident."""
    field = fields_for(SCHEMA)[0]
    with pytest.raises(Exception):
        field.value = "changed"  # type: ignore[misc]
    assert isinstance(field, Field)


# --- an unset path shows as empty -------------------------------------------


def test_an_unset_path_starts_empty_rather_than_at_the_working_directory():
    """`input_path` has no default: the picker must not invent one.

    magicgui's FileEdit falls back to the current directory for a null value,
    which reads as a choice somebody made. An empty box says "not set", which
    is what the pre-run checks will also say.
    """
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert SCHEMA["input_path"].default is None
    assert by_name["input_path"].value == ""


def test_a_path_that_has_a_default_still_shows_it():
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert by_name["ilastik_output_dir"].value == SCHEMA["ilastik_output_dir"].default


def test_a_row_is_blank_only_when_the_setting_is_unset():
    """An empty box means "not set" -- so it must not appear for a set value."""
    for field in fields_for(SCHEMA):
        if field.value != "":
            continue
        default = SCHEMA[field.name].default
        assert default is None or default == "", (
            f"{field.name} shows as empty but its default is {default!r}"
        )


# --- unset must survive the round trip through a widget ----------------------


def test_a_setting_with_no_default_gets_a_widget_that_can_be_empty():
    """A FloatSpinBox cannot hold "unset"; it reports 0.0, which is a value.

    `fwhm_diameter_guess_um` is a float with no default. Shown in a spin box,
    the panel reads back 0.0, the schema sees a setting that is neither None
    nor its default, and warns that it is set while the feature that reads it
    is off -- for a panel nobody has touched.
    """
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert SCHEMA["fwhm_diameter_guess_um"].default is None
    assert by_name["fwhm_diameter_guess_um"].widget_type == "LineEdit"
    assert by_name["fwhm_diameter_guess_um"].value == ""


def test_a_numeric_setting_that_has_a_default_keeps_its_spin_box():
    by_name = {field.name: field for field in fields_for(SCHEMA)}
    assert by_name["min_stub_length"].widget_type == "FloatSpinBox"


@pytest.mark.parametrize(
    "kind,default,raw,expected",
    [
        ("path", None, Path("."), None),        # FileEdit's empty value
        ("path", None, "", None),
        ("path", None, Path("/data/x.tif"), Path("/data/x.tif")),
        ("path", "out", Path("out"), Path("out")),
        ("float", None, "", None),              # LineEdit left empty
        ("float", None, "4.5", 4.5),
        ("int", None, "7", 7),
        ("float", 1.0, 2.5, 2.5),               # ordinary spin box
        ("str", None, "", None),
        ("str", "auto", "", ""),                # a real empty string, not unset
    ],
)
def test_a_widget_value_reads_back_as_the_setting_value(kind, default, raw, expected):
    from haemolynx.parsers import Setting

    field = field_for(Setting("thing", kind, default, "A thing", "S"))
    assert field.to_setting_value(raw) == expected


def test_the_untouched_panel_warns_about_nothing():
    """Opening the panel and reading it back must equal the schema defaults.

    This is the whole bug: a widget that invents a value for a setting with no
    default makes the schema report "set but nothing will read it" on a panel
    the user has not touched.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SCHEMA.validate(values_from(fields_for(SCHEMA)))

    assert [str(warning.message) for warning in caught] == []


def test_the_untouched_panel_produces_the_schema_defaults():
    read_back = SCHEMA.validate(values_from(fields_for(SCHEMA)))
    defaults = SCHEMA.validate({s.name: s.default for s in SCHEMA})
    assert read_back == defaults


# --- options must suit the widget, not the kind ------------------------------


def test_no_row_passes_an_option_its_widget_would_reject():
    """magicgui raises on an unknown keyword, and the panel fails to open.

    The options used to be chosen from the setting's kind. When a float with no
    default started using a LineEdit so it could be empty, it kept the spin
    box's options and magicgui refused it:

        LineEdit got an unexpected keyword argument: min, max, step

    Options now follow the widget the setting actually gets. This checks every
    row, so the next widget swap cannot repeat it.
    """
    offenders = {}
    for field in fields_for(SCHEMA):
        allowed = OPTIONS_BY_WIDGET[field.widget_type]
        extra = sorted(set(field.options) - allowed)
        if extra:
            offenders[field.name] = (field.widget_type, extra)
    assert offenders == {}, (
        f"rows whose options their widget would reject: {offenders}"
    )


def test_every_widget_in_use_declares_which_options_it_takes():
    used = {field.widget_type for field in fields_for(SCHEMA)}
    assert used <= set(OPTIONS_BY_WIDGET), (
        f"widgets with no declared options: {sorted(used - set(OPTIONS_BY_WIDGET))}"
    )


def test_a_numeric_setting_with_no_default_gets_no_range_options():
    from haemolynx.parsers import Setting

    field = field_for(Setting("guess", "float", None, "A guess", "S", minimum=0.0))
    assert field.widget_type == "LineEdit"
    assert field.options == {}


def test_a_numeric_setting_with_a_default_keeps_its_range():
    from haemolynx.parsers import Setting

    field = field_for(Setting("guess", "float", 1.0, "A guess", "S", minimum=0.0))
    assert field.widget_type == "FloatSpinBox"
    assert field.options["min"] == 0.0


def test_every_row_starts_with_a_value_its_widget_can_read_back():
    """A LiteralEvalLineEdit parses its text; "" is not a literal.

    Blanking an unset value made three rows -- voxel_size_override_xyz and the
    two diameter tables -- start empty, and magicgui raised
    `SyntaxError: invalid syntax (<unknown>, line 0)` from `ast.literal_eval`
    as soon as the panel read them, which is at construction. None renders as
    "None" and parses straight back, so that is what they get.
    """
    import ast

    for field in fields_for(SCHEMA):
        if field.widget_type != "LiteralEvalLineEdit":
            continue
        try:
            ast.literal_eval(str(field.value))
        except (ValueError, SyntaxError) as error:  # pragma: no cover - the failure
            raise AssertionError(
                f"{field.name} starts as {field.value!r}, which its "
                f"LiteralEvalLineEdit cannot parse: {error}"
            ) from error


def test_an_unset_literal_row_is_none_rather_than_empty():
    from haemolynx.parsers import Setting

    field = field_for(Setting("table", "mapping", None, "A table", "S"))
    assert field.widget_type == "LiteralEvalLineEdit"
    assert field.value is None
    assert field.to_setting_value(None) is None


def test_an_unset_path_or_number_is_still_blank():
    """The empty string is right for those two; only literals need None."""
    from haemolynx.parsers import Setting

    assert field_for(Setting("where", "path", None, "A path", "S")).value == ""
    assert field_for(Setting("guess", "float", None, "A guess", "S")).value == ""
