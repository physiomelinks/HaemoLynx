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
    DELETE_PRESET,
    DELETE_PRESET_TEXT,
    NO_PRESET,
    PRESETS,
    SAVE_PRESET,
    SAVE_PRESET_TEXT,
    USAGE_PRESETS,
    check_presets,
    preset_menu,
    preset_settings,
)
from haemolynx.pipeline.saved_presets import save_preset, saved_presets

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


def test_the_mask_preset_keeps_the_large_vessels_over_a_config_that_cut_them():
    """The cut at the large vessel masks is off by default (Harvey,
    2026-11-04), and the masks preset leaves it off: chosen over a config
    that cut there, it turns the cut off rather than keeping it."""
    cut = {**SCHEMA.defaults(), "cut_network_at_large_vessel_volumes": True}
    for over in (None, cut):
        values = _loaded("arteriole_venule_masks", over=over)
        assert values["cut_network_at_large_vessel_volumes"] is False
        assert values["use_large_vessel_masks"] is True


def _usage_bases() -> tuple[dict, dict]:
    """The defaults, and the defaults with every switch a usage names
    flipped: a usage's own settings are the same from either."""
    base = {**SCHEMA.defaults(), "raw_image_path": "raw.tif", "cell_mask_path": "cells.tif"}
    flipped = dict(base)
    for setting in set().union(*(PRESETS[name]["overrides"] for name in USAGE_PRESETS)):
        assert SCHEMA[setting].kind == "bool", setting
        flipped[setting] = not base[setting]
    return base, flipped


def test_choosing_a_usage_gives_the_same_run_whichever_usage_came_before():
    """Regression: the flow presets left the equivalent resistance off after
    Quick look, though either one chosen first computes it. And Quick look
    turned the tissue volume and the vascular communities off with no other
    usage naming them, so they stayed off after it: unseen from the
    defaults, where both are off already, hence the flipped base."""
    named = set().union(*(PRESETS[name]["overrides"] for name in USAGE_PRESETS))
    for base in _usage_bases():
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


def test_an_add_on_turns_on_what_its_own_settings_need_over_any_usage():
    """Full report names the statistics and VTK switches beside its own, as
    the docstring says: over Cell-to-vessel distances, which turns them off,
    it turns them back on, and the cell measurement stays the usage's. A
    setting an add-on names is left unread only by a file still to choose or
    by what the usage decided (Large image's solver for an equivalent
    resistance Quick look does not compute)."""
    decided = set().union(*(PRESETS[name]["overrides"] for name in USAGE_PRESETS))
    for usage, add_on in itertools.product(USAGE_PRESETS, ADD_ONS):
        values = _loaded(usage, add_on)
        for setting in PRESETS[add_on]["overrides"]:
            for rule in unmet_chain(SCHEMA, setting, values):
                gate = prerequisite_name(rule)
                assert SCHEMA[gate].kind == "path" or (
                    gate in decided and gate not in PRESETS[add_on]["overrides"]
                ), (usage, add_on, setting, rule)
    report = _loaded("cell_distances", "full_report")
    assert report["statistics"] and report["vtk_export"] and report["compute_vascular_communities"]
    assert report["measurement_3d_to_cell_mask"] and not report["run_haemodynamics"]
    assert not report["measure_tissue_volume"]


def test_the_menu_starts_at_none_then_lists_the_usages_then_the_add_ons():
    menu = preset_menu()
    assert menu[0] == (NO_PRESET, None)
    assert menu[1] is None
    assert [entry[1] for entry in menu[2:2 + len(USAGE_PRESETS)]] == list(USAGE_PRESETS)
    assert menu[2 + len(USAGE_PRESETS)] is None
    add_ons = menu[3 + len(USAGE_PRESETS):3 + len(USAGE_PRESETS) + len(ADD_ONS)]
    assert [entry[1] for entry in add_ons] == list(ADD_ONS)
    # With nothing saved, the menu ends by offering to save these settings.
    assert menu[-2:] == [None, (SAVE_PRESET_TEXT, SAVE_PRESET)]
    assert set(USAGE_PRESETS) <= set(PRESETS)
    assert [entry[0] for entry in menu if entry and entry[1] in PRESETS] == [
        PRESETS[name]["label"] for name in (*USAGE_PRESETS, *ADD_ONS)
    ]


def test_the_presets_saved_on_this_machine_follow_the_standard_ones(tmp_path):
    defaults = SCHEMA.defaults()
    save_preset("Zeta", defaults, SCHEMA, directory=tmp_path)
    save_preset("alpha", defaults, SCHEMA, directory=tmp_path)
    saved = saved_presets(tmp_path)
    menu = preset_menu(saved)
    assert menu[:len(preset_menu()) - 2] == preset_menu()[:-2]
    assert menu[len(preset_menu()) - 2:] == [
        None,
        ("alpha", "saved:alpha"),
        ("Zeta", "saved:Zeta"),
        None,
        (SAVE_PRESET_TEXT, SAVE_PRESET),
        (DELETE_PRESET_TEXT, DELETE_PRESET),
    ]
    keys = [entry[1] for entry in menu if entry]
    assert len(set(keys)) == len(keys)


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


def test_each_layer_overrides_the_one_before_as_the_readme_says(tmp_path):
    """README "Settings and presets": schema default -> config file ->
    --preset -> --<setting-name> flag, and a preset sets only what it names."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "schema default  ->  config file (--config)  ->  --preset NAME  ->  "
        "--<setting-name> VALUE"
    ) in readme

    config = tmp_path / "config.yaml"
    config.write_text(
        "haemodynamics_solver: dense\n"
        "equivalent_resistance_solver: eigendecomposition\n"
        "vtk_export: false\n",
        encoding="utf-8",
    )
    settings = settings_from_command_line(
        SCHEMA,
        config,
        presets=PRESETS,
        argv=["--preset", "large_image", "--haemodynamics-solver", "dense"],
    )
    defaults = SCHEMA.defaults()
    # Named by nothing: the schema default.
    assert settings["export_citations"] == defaults["export_citations"]
    # Named by the config only: the config, though the default differs.
    assert defaults["vtk_export"] is True and settings["vtk_export"] is False
    # Named by the config and the preset: the preset.
    assert settings["equivalent_resistance_solver"] == "sparse"
    # Named by the preset and a flag: the flag.
    assert settings["haemodynamics_solver"] == "dense"


def test_the_readme_names_the_preset_files_that_exist():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for path in (
        "src/haemolynx/pipeline/schema.py",
        "src/haemolynx/pipeline/presets.py",
        "examples/pipeline_presets.py",
    ):
        assert f"[`{path}`]({path})" in readme
        assert (REPO_ROOT / path).is_file()
