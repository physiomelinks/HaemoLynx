"""The presets: named sets of settings to start a run from (`pipeline/presets.py`).

The panel's Preset menu and the examples' ``--preset`` flag both read them, so
these check what a preset may name, that everything it sets is read once the
files it leaves to the user are chosen, and that a usage chosen after another
gives the same run as one chosen first.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

from haemolynx.gui.form import unmet_chain
from haemolynx.parsers import ConfigError, prerequisite_name
from haemolynx.parsers.cli import settings_from_command_line
from haemolynx.pipeline import default_schema, write_default_config
from haemolynx.pipeline.presets import (
    NO_PRESET,
    PRESETS,
    USAGE_PRESETS,
    check_presets,
    preset_menu,
    preset_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = default_schema()
ADD_ONS = tuple(name for name in PRESETS if name not in USAGE_PRESETS)


def _loaded(*names: str, over: dict | None = None) -> dict:
    """The settings after loading *names*, in turn, over *over* (the defaults)."""
    values = dict(SCHEMA.defaults() if over is None else over)
    for name in names:
        values.update(preset_settings(name, SCHEMA))
    return values


def test_every_preset_names_settings_the_schema_has_with_values_they_take():
    check_presets(PRESETS, SCHEMA)
    for name in PRESETS:
        assert set(preset_settings(name, SCHEMA)) == set(PRESETS[name]["overrides"])


def test_check_presets_refuses_an_unknown_setting_and_a_value_it_cannot_take():
    with pytest.raises(ValueError, match="no_such_setting"):
        check_presets({"bad": {"overrides": {"no_such_setting": True}}}, SCHEMA)
    with pytest.raises(ConfigError):
        check_presets({"bad": {"overrides": {"statistics_mode": "neither"}}}, SCHEMA)


def test_preset_settings_refuses_an_unknown_preset_and_coerces_the_rest():
    with pytest.raises(KeyError, match="no_such_preset"):
        preset_settings("no_such_preset", SCHEMA)
    assert preset_settings("large_image", SCHEMA) == {
        "use_memmap_loading": True,
        "haemodynamics_solver": "sparse",
        "equivalent_resistance_solver": "sparse",
    }


def test_a_preset_picks_switches_and_choices_never_files_or_thresholds():
    """A file belongs to the image being analysed, and a threshold is
    calibrated on marked data, not chosen for a usage."""
    for name, spec in PRESETS.items():
        kinds = {setting: SCHEMA[setting].kind for setting in spec["overrides"]}
        assert set(kinds.values()) <= {"bool", "choice"}, (name, kinds)


def test_every_setting_a_preset_names_is_read_once_its_files_are_chosen():
    """Loaded over the defaults, nothing a preset sets is left out of effect
    by another switch, only by a file the user still has to choose (the raw
    image) -- otherwise the preset would quietly do nothing."""
    for name in PRESETS:
        values = _loaded(name)
        for setting in PRESETS[name]["overrides"]:
            unmet = unmet_chain(SCHEMA, setting, values)
            assert all(SCHEMA[prerequisite_name(rule)].kind == "path" for rule in unmet), (
                name, setting, unmet,
            )


def test_the_raw_image_preset_measures_once_the_raw_image_is_set():
    values = _loaded("flow_raw_image_diameters")
    assert unmet_chain(SCHEMA, "measure_diameters_from_raw_image", values) == ("raw_image_path",)
    values["raw_image_path"] = "raw.tif"
    assert not unmet_chain(SCHEMA, "measure_diameters_from_raw_image", values)


def test_choosing_a_usage_gives_the_same_run_whichever_usage_came_before():
    """Regression: the flow presets left the equivalent resistance off after
    Quick look, though either one chosen first computes it."""
    base = {**SCHEMA.defaults(), "raw_image_path": "raw.tif", "cell_mask_path": "cells.tif"}
    named = set().union(*(PRESETS[name]["overrides"] for name in USAGE_PRESETS))
    for before, chosen in itertools.permutations(USAGE_PRESETS, 2):
        alone = _loaded(chosen, over=base)
        after = _loaded(before, chosen, over=base)
        for setting in sorted(named):
            if not unmet_chain(SCHEMA, setting, alone):
                assert after[setting] == alone[setting], (before, chosen, setting)


def test_an_add_on_keeps_the_usage_it_is_loaded_over():
    """An add-on names only its own settings: what flows, and how diameters
    are measured, stay the usage's."""
    deciding = ("run_haemodynamics", "measure_diameters_from_raw_image", "measurement_3d_to_cell_mask")
    for usage, add_on in itertools.product(USAGE_PRESETS, ADD_ONS):
        usage_alone = _loaded(usage)
        with_add_on = _loaded(usage, add_on)
        for setting in deciding:
            assert with_add_on[setting] == usage_alone[setting], (usage, add_on, setting)


def test_the_menu_starts_at_none_then_lists_the_usages_then_the_add_ons():
    menu = preset_menu()
    assert menu[0] == (NO_PRESET, None)
    assert menu[1] is None
    assert [entry[1] for entry in menu[2:2 + len(USAGE_PRESETS)]] == list(USAGE_PRESETS)
    assert menu[2 + len(USAGE_PRESETS)] is None
    assert [entry[1] for entry in menu[3 + len(USAGE_PRESETS):]] == list(ADD_ONS)
    assert set(USAGE_PRESETS) <= set(PRESETS)
    assert [entry[0] for entry in menu if entry and entry[1]] == [
        PRESETS[name]["label"] for name in (*USAGE_PRESETS, *ADD_ONS)
    ]


def test_each_preset_has_its_own_label_and_a_description_in_the_tooltip_voice():
    """The description is the menu entry's hover text, written as the
    panel's other tooltips are."""
    labels = [spec["label"] for spec in PRESETS.values()]
    assert all(label.strip() for label in labels)
    assert len(set(labels)) == len(labels)
    assert NO_PRESET not in labels
    for name, spec in PRESETS.items():
        description = spec["description"]
        assert len(description.split()) >= 3, name
        assert not description.rstrip().endswith("."), name


def test_the_command_line_offers_the_panel_presets_beside_the_examples(tmp_path, capsys):
    examples = str(REPO_ROOT / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    from pipeline_presets import PRESETS as COMMAND_LINE_PRESETS

    for name, spec in PRESETS.items():
        assert COMMAND_LINE_PRESETS[name] is spec
    assert "quick_debug" in COMMAND_LINE_PRESETS

    config = tmp_path / "config.yaml"
    write_default_config(config)
    settings = settings_from_command_line(
        SCHEMA, config, presets=COMMAND_LINE_PRESETS, argv=["--preset", "large_image"]
    )
    assert settings["use_memmap_loading"] is True
    assert settings["haemodynamics_solver"] == "sparse"
    assert settings["equivalent_resistance_solver"] == "sparse"
    assert "Applying preset 'large_image'" in capsys.readouterr().out
