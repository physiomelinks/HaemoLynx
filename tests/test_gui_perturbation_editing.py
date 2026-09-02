"""Editing a list of perturbations, decided without a display.

`gui/perturbation_editing.py` is the whole of what the Perturbations tab
decides: which editors a type reveals, what "+" and "Remove" do to the list,
and what a list of entries looks like as a settings value. None of it needs
Qt, so all of it is tested here; `test_gui_perturbation_widget.py` is left with
only what a real viewer answers.

Two properties are load-bearing rather than tidy:

* **Nothing is edited in place.** The panel holds the returned list as its new
  state, so a function that mutated its argument would leave the widgets and
  the settings row describing different lists.
* **The values that leave are builtins.** The row this list travels in is a
  magicgui `LiteralEvalLineEdit`, which stores `str(value)` and reads it back
  with `ast.literal_eval` -- so a `np.float64` typed into an editor comes back
  as `'np.float64(20.0)'` and raises. The same value would stop
  `yaml.safe_dump` writing a config out at all.
"""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

yaml = pytest.importorskip("yaml")

from haemolynx.gui.perturbation_editing import (  # noqa: E402
    ADD_TOOLTIP,
    ALWAYS_VISIBLE_TAB_SETTINGS,
    EDITOR_SETTINGS,
    NAME_TOOLTIP,
    PERTURBATION_TYPES,
    PERTURBATION_TYPE_DISPLAY_NAMES,
    REMOVE_TOOLTIP,
    SETTING_DISPLAY_LABELS,
    TYPE_TOOLTIP,
    UNCHOSEN,
    add_entry,
    default_name,
    display_label_for_setting,
    display_name_for_type,
    from_settings,
    hidden_for_type,
    name_problems,
    new_entry,
    perturbation_type_choices,
    remove_entry,
    rows_for_type,
    set_name,
    set_overrides,
    set_type,
    summary,
    to_settings,
    visible_tab_settings,
)
from haemolynx.gui.form import field_for  # noqa: E402
from haemolynx.haemodynamics.perturbations import SETTINGS_FOR_TYPE  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402

SCHEMA = default_schema()

A_DILATION = {
    "name": "art_dilate_20",
    "type": "arteriole_diameter_change",
    "overrides": {"arteriole_diameter_change_percent": 20},
}
A_SWEEP = {
    "name": "higher_inlet",
    "type": "pressure_sweep",
    "overrides": {"inlet_pressure_min_pa": 5000},
}


# --- which editors a type reveals --------------------------------------------


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_a_type_reveals_exactly_the_settings_it_reads(perturbation_type: str):
    """The panel shows a type's options because the run reads them."""
    assert rows_for_type(perturbation_type) == SETTINGS_FOR_TYPE[perturbation_type]


def test_the_unchosen_type_reveals_nothing():
    """A fresh row asks one question -- "what kind?" -- and no others."""
    assert rows_for_type(UNCHOSEN) == ()
    assert hidden_for_type(UNCHOSEN) == EDITOR_SETTINGS


def test_an_unknown_type_reveals_nothing_rather_than_the_last_ones_rows():
    assert rows_for_type("no_such_type") == ()


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_shown_and_hidden_are_every_editor_between_them(perturbation_type: str):
    """Hidden, not greyed out: an editor is in exactly one of the two lists."""
    shown = set(rows_for_type(perturbation_type))
    hidden = set(hidden_for_type(perturbation_type))
    assert shown.isdisjoint(hidden)
    assert shown | hidden == set(EDITOR_SETTINGS)


def test_every_editor_is_a_declared_setting():
    """An editor for a setting no schema declares could not be built at all."""
    for name in EDITOR_SETTINGS:
        assert name in SCHEMA, f"{name} is revealed by a type but not declared"


def test_every_perturbation_setting_has_descriptive_help():
    """Schema help is the GUI tooltip (plus unit) for every SETTINGS_FOR_TYPE row."""
    for name in EDITOR_SETTINGS:
        setting = SCHEMA[name]
        help_text = setting.help.strip()
        assert help_text, f"{name} has empty help"
        assert not help_text.endswith("."), (
            f"{name} help should read as a label, without a full stop"
        )
        field = field_for(setting)
        assert field.help.strip(), f"{name} form field has empty tooltip"
        if setting.unit:
            assert field.help.endswith(f"({setting.unit})")


def test_non_schema_perturbation_controls_expose_tooltip_strings():
    """Name, type, Add and Remove are not settings; their tooltips live here."""
    for text in (NAME_TOOLTIP, TYPE_TOOLTIP, ADD_TOOLTIP, REMOVE_TOOLTIP):
        assert text.strip()
        assert " " in text  # a real sentence, not a token
    assert "directory" in NAME_TOOLTIP
    assert "constriction" in TYPE_TOOLTIP and "dilation" in TYPE_TOOLTIP
    assert "baseline" in ADD_TOOLTIP
    assert "Remove this perturbation" in REMOVE_TOOLTIP


def test_no_editor_is_built_twice():
    """`pericyte_dilation_*` is read by one type; a repeat would be two rows."""
    assert len(set(EDITOR_SETTINGS)) == len(EDITOR_SETTINGS)


def test_the_tab_keeps_only_the_always_on_run_settings():
    """Sweep ranges and pericyte knobs are type options, not permanent rows."""
    from haemolynx.haemodynamics.perturbations import (
        PERICYTE_CONSTRICTION_SETTINGS,
        PERICYTE_ENTRY_GEOMETRY_SETTINGS,
    )

    claimed = (
        *ALWAYS_VISIBLE_TAB_SETTINGS,
        "run_pericyte_dilation_sweep",
        "pericyte_dilation_min_percent",
        "inlet_pressure_min_pa",
        "sweep_output_dir",
        *EDITOR_SETTINGS,
        *PERICYTE_CONSTRICTION_SETTINGS,
    )
    assert visible_tab_settings(claimed) == ALWAYS_VISIBLE_TAB_SETTINGS
    assert "pericyte_dilation_min_percent" in rows_for_type("pericyte_dilation_sweep")
    assert "inlet_pressure_min_pa" in rows_for_type("pressure_sweep")
    assert "inlet_pressure_min_pa" in rows_for_type("pressure_and_pericyte_sweep")
    assert "arteriole_diameter_change_percent" in rows_for_type(
        "arteriole_diameter_change"
    )
    assert "arteriole_dilation_min_percent" in rows_for_type("arteriole_diameter_sweep")
    assert "inlet_pressure_min_pa" in rows_for_type("pressure_and_arteriole_sweep")
    assert "arteriole_dilation_min_percent" in rows_for_type(
        "pressure_and_arteriole_sweep"
    )
    assert "capillary_dilation_min_percent" in rows_for_type("capillary_diameter_sweep")
    assert "inlet_pressure_min_pa" in rows_for_type("pressure_and_capillary_sweep")
    assert "capillary_dilation_min_percent" in rows_for_type(
        "pressure_and_capillary_sweep"
    )
    assert "pericyte_mask_path" in rows_for_type("pericyte_diameter_change")
    for name in PERICYTE_ENTRY_GEOMETRY_SETTINGS:
        assert name in rows_for_type("pericyte_dilation_sweep"), name
        assert name in rows_for_type("pressure_and_pericyte_sweep"), name
        assert name in rows_for_type("pericyte_diameter_change"), name
        assert name in rows_for_type("arteriole_and_pericyte_diameter_change"), name
    assert "constriction_by_branch_order" in rows_for_type("pericyte_spacing_sweep")
    assert "constriction_by_branch_order" in rows_for_type("pericyte_length_sweep")
    assert "pericyte_constriction_factor" in rows_for_type("pericyte_dilation_sweep")
    assert "pericyte_constriction_factor" in rows_for_type("pericyte_spacing_sweep")
    assert "run_pericyte_dilation_sweep" not in EDITOR_SETTINGS
    assert "sweep_output_dir" not in EDITOR_SETTINGS
    assert "do_pericyte_construction" not in ALWAYS_VISIBLE_TAB_SETTINGS
    assert "constriction_by_branch_order" not in ALWAYS_VISIBLE_TAB_SETTINGS
    assert "pericyte_constriction_factor" not in ALWAYS_VISIBLE_TAB_SETTINGS


# --- adding and removing -----------------------------------------------------


def test_adding_appends_one_unchosen_entry():
    """The "+" button: a row that asks what it should be and runs nothing."""
    added = add_entry([])
    assert len(added) == 1
    assert added[0]["type"] == UNCHOSEN
    assert added[0]["overrides"] == {}


def test_adding_names_the_new_entry_after_the_ones_already_there():
    """A name is a directory, so two entries may not share one."""
    entries = add_entry(add_entry(add_entry([])))
    assert len({entry["name"] for entry in entries}) == 3


def test_a_new_entry_avoids_a_name_a_user_typed():
    taken = [{"name": "perturbation_1", "type": UNCHOSEN, "overrides": {}}]
    assert new_entry(taken)["name"] != "perturbation_1"
    assert default_name(taken) == "perturbation_2"


def test_removing_what_was_added_leaves_what_was_there():
    """Add then remove is a round trip, not an approximation of one."""
    before = from_settings({"perturbations": [A_DILATION, A_SWEEP]})
    after = remove_entry(add_entry(before), len(before))
    assert after == before


def test_removing_the_first_shifts_the_rest_up():
    entries = from_settings({"perturbations": [A_DILATION, A_SWEEP]})
    kept = remove_entry(entries, 0)
    assert [entry["name"] for entry in kept] == ["higher_inlet"]


@pytest.mark.parametrize("index", (-1, 2, 99))
def test_removing_an_index_that_is_not_there_removes_nothing(index: int):
    entries = from_settings({"perturbations": [A_DILATION, A_SWEEP]})
    assert remove_entry(entries, index) == entries


# --- changing one entry ------------------------------------------------------


def test_choosing_a_type_keeps_the_overrides_that_type_reads():
    entries = set_type(from_settings({"perturbations": [A_DILATION]}), 0, "arteriole_diameter_change")
    assert entries[0]["overrides"] == {"arteriole_diameter_change_percent": 20}


def test_changing_the_type_drops_the_overrides_it_no_longer_reads():
    """Otherwise a sweep carries an arteriole scale that nothing applies, and
    the run reports it as an unused override -- when what happened is that the
    user changed their mind."""
    entries = set_type(from_settings({"perturbations": [A_DILATION]}), 0, "pressure_sweep")
    assert entries[0]["type"] == "pressure_sweep"
    assert entries[0]["overrides"] == {}


def test_setting_the_overrides_replaces_them_rather_than_merging():
    """The visible editors *are* the overrides: a hidden one is not set."""
    entries = set_overrides(
        from_settings({"perturbations": [A_SWEEP]}), 0, {"inlet_pressure_max_pa": 7000}
    )
    assert entries[0]["overrides"] == {"inlet_pressure_max_pa": 7000}


def test_renaming_an_entry_renames_only_that_one():
    entries = set_name(from_settings({"perturbations": [A_DILATION, A_SWEEP]}), 1, "later")
    assert [entry["name"] for entry in entries] == ["art_dilate_20", "later"]


@pytest.mark.parametrize(
    "change",
    (
        lambda entries: set_name(entries, 5, "x"),
        lambda entries: set_type(entries, 5, "pressure_sweep"),
        lambda entries: set_overrides(entries, 5, {"inlet_pressure_min_pa": 1}),
    ),
)
def test_editing_an_entry_that_is_not_there_changes_nothing(change):
    entries = from_settings({"perturbations": [A_DILATION]})
    assert change(entries) == entries


# --- nothing is edited in place ----------------------------------------------


@pytest.mark.parametrize(
    "change",
    (
        lambda entries: add_entry(entries),
        lambda entries: remove_entry(entries, 0),
        lambda entries: set_name(entries, 0, "renamed"),
        lambda entries: set_type(entries, 0, "pressure_sweep"),
        lambda entries: set_overrides(entries, 0, {"arteriole_diameter_change_percent": 100}),
    ),
)
def test_no_edit_touches_the_list_it_was_given(change):
    """The panel keeps what it was given until it is handed a replacement."""
    entries = from_settings({"perturbations": [A_DILATION, A_SWEEP]})
    untouched = copy.deepcopy(entries)

    change(entries)

    assert entries == untouched


# --- the settings round trip -------------------------------------------------


def test_to_settings_and_from_settings_are_inverses():
    entries = from_settings({"perturbations": [A_DILATION, A_SWEEP]})
    assert from_settings(to_settings(entries)) == entries


def test_reading_a_settings_value_back_gives_the_same_value():
    values = {"perturbations": [A_DILATION, A_SWEEP]}
    assert to_settings(from_settings(values)) == values


def test_an_empty_list_survives_the_round_trip():
    assert from_settings({"perturbations": []}) == []
    assert to_settings([]) == {"perturbations": []}


def test_an_unset_setting_reads_as_no_perturbations():
    assert from_settings({}) == []


def test_a_hand_edited_entry_becomes_editable_rather_than_raising():
    """The panel has to open on a config a user got wrong."""
    entries = from_settings({"perturbations": ["not an entry", {"type": "nonsense"}]})
    assert len(entries) == 2
    assert all(set(entry) == {"name", "type", "overrides"} for entry in entries)


# --- the values that leave are builtins --------------------------------------


def test_a_numpy_value_typed_into_an_editor_leaves_as_a_builtin():
    """`repr(np.float64(20.0))` is `'np.float64(20.0)'`, which literal_eval
    refuses -- so the row would fail to read back what the panel put in it."""
    entries = set_overrides(
        from_settings({"perturbations": [A_DILATION]}),
        0,
        {"arteriole_diameter_change_percent": np.float64(20.0)},
    )
    value = entries[0]["overrides"]["arteriole_diameter_change_percent"]
    assert type(value) is float


def test_the_settings_value_survives_the_rows_literal_eval():
    entries = set_overrides(
        from_settings({"perturbations": [A_SWEEP]}),
        0,
        {"inlet_pressure_min_pa": np.int64(5000)},
    )
    written = to_settings(entries)["perturbations"]
    assert ast.literal_eval(str(written)) == written


def test_the_settings_value_can_be_written_to_a_config():
    entries = set_overrides(
        from_settings({"perturbations": [A_DILATION]}),
        0,
        {"arteriole_diameter_change_percent": np.float64(20.0)},
    )
    dumped = yaml.safe_dump(to_settings(entries))
    assert yaml.safe_load(dumped) == to_settings(entries)


def test_a_path_editor_leaves_as_a_forward_slashed_string():
    """The same reason the config writer does it: one file on two machines."""
    entries = set_overrides(
        from_settings(
            {"perturbations": [{"name": "tone", "type": "pericyte_diameter_change"}]}
        ),
        0,
        {"pericyte_mask_path": Path("masks") / "pericytes.tif"},
    )
    assert entries[0]["overrides"]["pericyte_mask_path"] == "masks/pericytes.tif"


# --- what the panel says -----------------------------------------------------


def test_a_name_that_is_a_path_is_reported():
    problems = name_problems([{"name": "../elsewhere", "type": UNCHOSEN}])
    assert len(problems) == 1
    assert "directory" in problems[0]


def test_two_entries_with_one_name_are_reported():
    entries = [{"name": "same", "type": UNCHOSEN}, {"name": "same", "type": UNCHOSEN}]
    assert any("two perturbations" in problem for problem in name_problems(entries))


def test_ordinary_names_are_left_alone():
    assert name_problems(from_settings({"perturbations": [A_DILATION, A_SWEEP]})) == ()


def test_the_summary_counts_what_would_actually_re_solve():
    entries = from_settings(
        {"perturbations": [A_DILATION, {"name": "off", "type": UNCHOSEN}]}
    )
    said = summary(entries)
    assert "2" in said and "1 would re-solve" in said
    assert "art_dilate_20" in said
    assert "arteriole constriction/dilation" in said


def test_the_summary_says_so_when_there_are_none():
    assert "baseline only" in summary([])


def test_every_perturbation_type_has_a_constriction_dilation_display_name():
    """Dropdown labels cover every API type; diameter types say both directions."""
    assert set(PERTURBATION_TYPE_DISPLAY_NAMES) == set(PERTURBATION_TYPES)
    for name in PERTURBATION_TYPES:
        label = display_name_for_type(name)
        assert label
        if "diameter" in name or name in {
            "pericyte_dilation_sweep",
            "pressure_and_pericyte_sweep",
        }:
            assert "constriction/dilation" in label, name
    choices = perturbation_type_choices()
    assert [value for _label, value in choices] == list(PERTURBATION_TYPES)
    assert any("constriction/dilation" in label for label, _value in choices)


def test_bidirectional_settings_use_constriction_dilation_labels():
    for name in (
        "arteriole_diameter_change_percent",
        "arteriole_dilation_min_percent",
        "capillary_dilation_min_percent",
        "pericyte_dilation_min_percent",
        "pericyte_geometry_dilation_percent",
        "pericyte_constriction_factor",
        "constriction_by_branch_order",
    ):
        assert name in SETTING_DISPLAY_LABELS
        assert "constriction/dilation" in display_label_for_setting(name).lower()
    assert display_label_for_setting(
        "arteriole_diameter_change_percent", "percent"
    ) == "Arteriole constriction/dilation (percent)"
    # Unmapped names still capitalise like form.label_for.
    assert display_label_for_setting("inlet_pressure_min_pa") == (
        "Inlet pressure min pa"
    )
