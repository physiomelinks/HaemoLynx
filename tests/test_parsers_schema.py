"""The settings schema: coercion, validation, and the GUI contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from ImageLynx.parsers import ConfigError, Schema, Setting


def _schema() -> Schema:
    return Schema(
        [
            Setting("input_path", "path", "in.tif", "Mask to analyse", "Input"),
            Setting(
                "voxel_size_um", "float", 1.0, "Voxel edge length", "Input",
                unit="um", minimum=0.0,
            ),
            Setting("passes", "int", 2, "Smoothing passes", "Input", minimum=1),
            Setting("use_masks", "bool", False, "Assign boundaries from masks", "Boundaries"),
            Setting(
                "mask_path", "path", None, "Arteriole mask", "Boundaries",
                requires=("use_masks",),
            ),
            Setting(
                "mode", "choice", "fast", "Statistics detail", "Output",
                choices=("fast", "full"),
            ),
            Setting("spacing_xyz", "float_list", (1.0, 1.0, 1.0), "Voxel spacing", "Input"),
            Setting("diameters", "mapping", {"B01": 5.0}, "Diameter per order", "Output"),
        ],
        title="Test schema",
    )


# --- coercion --------------------------------------------------------------


def test_values_are_coerced_to_their_declared_kind():
    resolved = _schema().validate(
        {"input_path": "a/b.tif", "voxel_size_um": 2, "passes": "3", "spacing_xyz": [1, 2, 3]}
    )
    assert resolved["input_path"] == Path("a/b.tif")
    assert isinstance(resolved["voxel_size_um"], float)
    assert resolved["passes"] == 3
    assert resolved["spacing_xyz"] == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "text,expected",
    [("true", True), ("False", False), ("yes", True), ("off", False)],
)
def test_bools_accept_yaml_and_cli_spellings(text, expected):
    assert _schema().validate({"use_masks": text})["use_masks"] is expected


def test_bools_reject_arbitrary_strings():
    with pytest.raises(ConfigError, match="expects true or false"):
        _schema().validate({"use_masks": "maybe"})


def test_ints_reject_bools():
    """True is an int in Python; silently accepting it hides a config mistake."""
    with pytest.raises(ConfigError, match="expects an int, got a bool"):
        _schema().validate({"passes": True})


def test_missing_keys_take_their_default():
    resolved = _schema().validate({})
    assert resolved["mode"] == "fast"
    assert resolved["voxel_size_um"] == 1.0
    assert set(resolved) == set(_schema().names)


# --- validation ------------------------------------------------------------


def test_unknown_key_is_rejected_with_a_spelling_suggestion():
    with pytest.raises(ConfigError, match="Unknown setting 'inptu_path'.*Did you mean: input_path"):
        _schema().validate({"inptu_path": "x"})


def test_value_outside_declared_bounds_is_rejected_with_the_unit():
    with pytest.raises(ConfigError, match="below its minimum 0.0 um"):
        _schema().validate({"voxel_size_um": -1.0})


def test_value_outside_choices_lists_the_allowed_values():
    with pytest.raises(ConfigError, match="allowed values are 'fast', 'full'"):
        _schema().validate({"mode": "medium"})


def test_a_setting_whose_prerequisite_is_off_is_reported():
    """The silent class of bug: a value that is set but can never be read."""
    with pytest.raises(ConfigError, match="has no effect while 'use_masks' is false"):
        _schema().validate({"mask_path": "art.tif"})


def test_a_setting_whose_prerequisite_is_on_is_accepted():
    resolved = _schema().validate({"mask_path": "art.tif", "use_masks": True})
    assert resolved["mask_path"] == Path("art.tif")


def test_every_problem_is_reported_at_once():
    """One run of the checker must tell the user everything that is wrong."""
    with pytest.raises(ConfigError) as excinfo:
        _schema().validate({"mode": "medium", "voxel_size_um": -1.0, "nope": 1})
    message = str(excinfo.value)
    assert "3 configuration problems" in message
    assert "mode" in message and "voxel_size_um" in message and "nope" in message


# --- schema construction ---------------------------------------------------


def test_duplicate_setting_names_are_rejected():
    with pytest.raises(ConfigError, match="Duplicate setting 'a'"):
        Schema([
            Setting("a", "int", 1, "First", "S"),
            Setting("a", "int", 2, "Second", "S"),
        ])


def test_requires_must_name_a_setting_in_the_schema():
    with pytest.raises(ConfigError, match="requires 'missing', which is not in the schema"):
        Schema([Setting("a", "int", 1, "Only", "S", requires=("missing",))])


def test_requires_must_name_a_bool():
    with pytest.raises(ConfigError, match="which is not a bool"):
        Schema([
            Setting("gate", "int", 1, "Not a flag", "S"),
            Setting("a", "int", 1, "Dependent", "S", requires=("gate",)),
        ])


def test_a_default_that_breaks_its_own_rules_fails_at_import():
    """A schema bug must surface when the schema loads, not on a user's run."""
    with pytest.raises(ConfigError, match="below its minimum"):
        Setting("a", "float", -1.0, "Bad default", "S", minimum=0.0)
    with pytest.raises(ConfigError, match="allowed values are"):
        Setting("m", "choice", "nope", "Bad default", "S", choices=("x", "y"))


def test_unknown_kind_is_rejected():
    with pytest.raises(ConfigError, match="unknown kind 'colour'"):
        Setting("a", "colour", None, "Nonsense", "S")


def test_settings_need_help_text():
    """Help is the GUI tooltip and the YAML comment, so it is not optional."""
    with pytest.raises(ConfigError, match="needs a help string"):
        Setting("a", "int", 1, "", "S")


def test_bounds_are_rejected_on_non_numeric_kinds():
    with pytest.raises(ConfigError, match="declares bounds but is kind 'str'"):
        Setting("a", "str", "x", "Text", "S", minimum=0.0)


def test_choices_are_rejected_on_non_choice_kinds():
    with pytest.raises(ConfigError, match="declares choices but is kind 'int'"):
        Setting("a", "int", 1, "Number", "S", choices=(1, 2))


# --- the GUI contract ------------------------------------------------------


def test_describe_is_json_serialisable_and_keeps_sections_in_order():
    import json

    described = _schema().describe()
    json.dumps(described)  # must not raise: Paths and tuples are converted

    assert described["title"] == "Test schema"
    assert [s["name"] for s in described["sections"]] == ["Input", "Boundaries", "Output"]


def test_describe_carries_every_field_a_form_needs():
    described = _schema().describe()
    by_name = {
        setting["name"]: setting
        for section in described["sections"]
        for setting in section["settings"]
    }
    assert by_name["mode"]["choices"] == ["fast", "full"]
    assert by_name["voxel_size_um"]["unit"] == "um"
    assert by_name["voxel_size_um"]["minimum"] == 0.0
    assert by_name["mask_path"]["requires"] == ["use_masks"]
    assert set(by_name) == set(_schema().names)


def test_sections_group_settings_in_declaration_order():
    assert [s.name for s in _schema().sections()["Input"]] == [
        "input_path",
        "voxel_size_um",
        "passes",
        "spacing_xyz",
    ]


def test_subset_keeps_only_the_named_settings():
    subset = _schema().subset(["mode", "use_masks"])
    assert subset.names == ("mode", "use_masks")


def test_indexing_an_unknown_name_suggests_a_close_one():
    with pytest.raises(ConfigError, match="Did you mean: input_path"):
        _schema()["input_pth"]
