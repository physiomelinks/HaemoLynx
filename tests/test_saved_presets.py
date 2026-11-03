"""Saved presets: the panel's settings kept on this machine for later runs.

A saved preset holds every setting but what belongs to one image, writes only
those away from their defaults, and gives the settings back as they were
saved: the ones it names at its values, every other one it may hold at its
default (`pipeline/saved_presets.py`).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

from haemolynx.parsers.schema import ConfigError
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.saved_presets import (
    PRESET_DIR_ENV,
    SAVED_PRESET_KEY_PREFIX,
    delete_saved_preset,
    held_by_saved_presets,
    is_saved_preset_key,
    preset_file_stem,
    read_saved_preset,
    save_preset,
    saved_preset_dir,
    saved_preset_for_key,
    saved_preset_path,
    saved_preset_settings,
    saved_preset_values,
    saved_presets,
)

SCHEMA = default_schema()
DEFAULTS = {setting.name: setting.default for setting in SCHEMA}


def _held(name: str) -> bool:
    return held_by_saved_presets(SCHEMA[name])


def _other_value(setting):
    """A value of *setting* away from its default, or None to pass it over."""
    default = setting.default
    if setting.kind == "bool":
        return not default
    if setting.kind == "choice":
        return next(choice for choice in setting.choices if choice != default)
    if setting.kind in {"int", "float"}:
        step = 1 if setting.kind == "int" else 0.5
        start = default if default is not None else (setting.minimum or step)
        for value in (start + step, start - step):
            if setting.minimum is not None and value < setting.minimum:
                continue
            if setting.maximum is not None and value > setting.maximum:
                continue
            return value
    return None


def test_a_saved_preset_holds_no_file_or_folder():
    paths = [setting.name for setting in SCHEMA if setting.kind == "path"]
    assert {"input_path", "raw_image_path", "output_dir", "ilastik_classifier_path"} <= set(paths)
    assert not any(_held(name) for name in paths)


def test_a_saved_preset_holds_nothing_describing_one_image():
    one_image = [
        name
        for name in SCHEMA.names
        if name.endswith(("_node_ids", "_node_coordinates", "_node_volumes", "_nodes"))
    ]
    assert len(one_image) == 26
    one_image += [
        "raw_image_channel",
        "tissue_raw_channel",
        "ilastik_arteriole_channels",
        "ilastik_vessel_label",
        "ilastik_small_venule_vessel_label",
        "cell_mask_h5_dataset_name",
        "voxel_size_override_xyz",
        "image_axis_order",
        "crop_on_import",
        "crop_method",
        "crop_size_xyz",
        "crop_location",
        "crop_z_location",
        "crop_start_xyz",
        "crop_end_xyz",
        "custom_edges",
        "capillary_block_vessel_ids",
        "ilastik_executable",
    ]
    assert [name for name in one_image if _held(name)] == []


def test_the_stage_skips_left_out_are_the_ones_reusing_cached_outputs():
    from haemolynx.gui.stage_checkpoints import SKIP_FOR_RESUME

    assert [name for name in SKIP_FOR_RESUME if _held(name)] == []


def test_a_saved_preset_holds_how_images_are_analysed():
    for name in (
        "statistics_mode",
        "segmentation_cleanup",
        "segmentation_cleanup_close_gaps_radius_um",
        "use_thick_vessel_skeletonisation",
        "viscosity_law",
        "open_end_model",
        "perturbations",
        "manual_capillary_diameter_by_branch_order",
        "voxel_size_policy",
        "segmented_image_downsample",
        "use_large_vessel_masks",
        "inlet_node_selection_method",
        "raw_image_stain",
        "segmentation_source",
    ):
        assert _held(name), name


def test_only_held_settings_away_from_their_defaults_are_written():
    values = dict(DEFAULTS)
    values.update(
        statistics_mode="full",
        segmentation_cleanup_close_gaps_radius_um=1.25,
        input_path="C:/data/image.tif",
        inlet_node_ids=[3, 4],
        raw_image_channel=2,
        do_skeletonize=False,
        crop_on_import=True,
        crop_size_xyz=(64, 64, 32),
    )
    saved = saved_preset_values(values, SCHEMA)
    assert set(saved) == {
        "statistics_mode",
        "segmentation_cleanup_close_gaps_radius_um",
    }
    # In the schema's order, as the file lists them.
    assert list(saved) == [name for name in SCHEMA.names if name in saved]


def test_the_panels_defaults_save_an_empty_preset(tmp_path):
    preset = save_preset("Defaults", DEFAULTS, SCHEMA, directory=tmp_path)
    assert preset.settings == {}
    settings, passed_over = saved_preset_settings(read_saved_preset(preset.path), SCHEMA)
    assert passed_over == []
    assert settings == {
        setting.name: setting.coerce(setting.default) for setting in SCHEMA if held_by_saved_presets(setting)
    }


def test_every_held_setting_comes_back_as_it_was_saved(tmp_path):
    values = dict(DEFAULTS)
    changed = {}
    for setting in SCHEMA:
        if held_by_saved_presets(setting) and (other := _other_value(setting)) is not None:
            values[setting.name] = other
            changed[setting.name] = setting.coerce(other)
    values["manual_arteriole_diameter_by_branch_order"] = {"Art1": 12.5, "Art2": 9.0}
    values["perturbations"] = [
        {"name": "half", "type": "pericyte_diameter_change", "overrides": {"pericyte_constriction_factor": 0.5}}
    ]
    assert len(changed) > 300

    save_preset("Everything changed", values, SCHEMA, directory=tmp_path)
    [preset] = saved_presets(tmp_path)
    settings, passed_over = saved_preset_settings(preset, SCHEMA)

    assert passed_over == []
    for name, value in changed.items():
        assert settings[name] == value, name
    assert settings["manual_arteriole_diameter_by_branch_order"] == {"Art1": 12.5, "Art2": 9.0}
    assert settings["perturbations"] == values["perturbations"]
    assert set(settings) == {setting.name for setting in SCHEMA if held_by_saved_presets(setting)}


def test_loading_puts_every_held_setting_it_does_not_name_back_to_its_default(tmp_path):
    preset = save_preset("Structure", {**DEFAULTS, "run_haemodynamics": False}, SCHEMA, directory=tmp_path)
    settings, _ = saved_preset_settings(preset, SCHEMA)
    assert settings["run_haemodynamics"] is False
    assert settings["statistics_mode"] == SCHEMA["statistics_mode"].default
    assert settings["perturbations"] == [] and settings["perturbations"] is not SCHEMA["perturbations"].default
    assert "input_path" not in settings and "inlet_node_ids" not in settings


def test_perturbation_overrides_naming_a_file_or_vessels_are_not_kept(tmp_path):
    entry = {
        "name": "blocked",
        "type": "capillary_block",
        "overrides": {
            "capillary_block_mask_path": "C:/data/blocks.tif",
            "capillary_block_vessel_ids": [7, 9],
            "capillary_block_resistance_factor": 25.0,
        },
    }
    saved = saved_preset_values({**DEFAULTS, "perturbations": [entry]}, SCHEMA)
    assert saved["perturbations"] == [
        {"name": "blocked", "type": "capillary_block", "overrides": {"capillary_block_resistance_factor": 25.0}}
    ]
    # Written into the file by hand, they are still not loaded.
    path = tmp_path / "by_hand.yaml"
    path.write_text(yaml.safe_dump({"settings": {"perturbations": [entry]}}), encoding="utf-8")
    settings, _ = saved_preset_settings(read_saved_preset(path), SCHEMA)
    assert settings["perturbations"][0]["overrides"] == {"capillary_block_resistance_factor": 25.0}


def test_the_file_reads_as_what_the_preset_changes(tmp_path):
    preset = save_preset(
        "  Quick E14 look  ", {**DEFAULTS, "statistics": True, "vtk_export": False}, SCHEMA, directory=tmp_path
    )
    text = preset.path.read_text(encoding="utf-8")
    assert text.startswith("# A HaemoLynx saved preset.")
    raw = yaml.safe_load(text)
    assert list(raw) == ["name", "saved", "settings"]
    assert raw["name"] == "Quick E14 look" == preset.name
    assert raw["settings"] == {"statistics": True, "vtk_export": False}
    assert preset.path == tmp_path / "Quick E14 look.yaml"
    assert not list(tmp_path.glob("*.part"))


def test_a_name_becomes_a_file_name_every_system_takes():
    assert preset_file_stem("My E14.5: run/2") == "My E14_5_ run_2"
    assert preset_file_stem("  thick vessels  ") == "thick vessels"
    assert preset_file_stem("con") == "_con"
    assert preset_file_stem("LPT1") == "_LPT1"
    assert len(preset_file_stem("x" * 300)) == 100
    for empty in ("", "   "):
        with pytest.raises(ValueError):
            preset_file_stem(empty)
    assert saved_preset_path("a/b", "folder") == Path("folder") / "a_b.yaml"


def test_saving_under_a_name_already_saved_replaces_it(tmp_path):
    save_preset("Mine", {**DEFAULTS, "statistics": True}, SCHEMA, directory=tmp_path)
    save_preset("Mine", {**DEFAULTS, "vtk_export": False}, SCHEMA, directory=tmp_path)
    [preset] = saved_presets(tmp_path)
    assert preset.settings == {"vtk_export": False}


def test_presets_are_listed_by_name_and_a_file_that_is_not_one_is_passed_over(tmp_path, caplog):
    save_preset("beta", DEFAULTS, SCHEMA, directory=tmp_path)
    save_preset("Alpha", DEFAULTS, SCHEMA, directory=tmp_path)
    (tmp_path / "broken.yaml").write_text("settings: [unclosed", encoding="utf-8")
    (tmp_path / "a_list.yaml").write_text("- one\n- two\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not yaml at all", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        found = saved_presets(tmp_path)
    assert [preset.name for preset in found] == ["Alpha", "beta"]
    assert "broken.yaml" in caplog.text and "a_list.yaml" in caplog.text
    assert saved_presets(tmp_path / "missing") == []


def test_a_preset_naming_a_renamed_unknown_or_one_image_setting_loads_the_rest(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "Old one",
                "settings": {
                    "diameter_basis": "anatomical",
                    "segmented_image_downsample": "2",
                    "no_such_setting": 1,
                    "input_path": "C:/data/image.tif",
                    "statistics": True,
                },
            }
        ),
        encoding="utf-8",
    )
    settings, passed_over = saved_preset_settings(read_saved_preset(path), SCHEMA)
    assert settings["segmentation_source"] == "endothelial"
    assert settings["segmented_image_downsample"] == "2x"
    assert settings["statistics"] is True
    assert passed_over == ["input_path", "no_such_setting"]
    assert "input_path" not in settings


def test_a_value_its_setting_cannot_take_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"settings": {"statistics_mode": "every way"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="statistics_mode"):
        saved_preset_settings(read_saved_preset(path), SCHEMA)


def test_a_preset_is_keyed_by_its_file_and_read_again_by_its_key(tmp_path):
    preset = save_preset("Key: test", {**DEFAULTS, "statistics": True}, SCHEMA, directory=tmp_path)
    assert preset.key == SAVED_PRESET_KEY_PREFIX + "Key_ test"
    assert is_saved_preset_key(preset.key)
    assert not any(is_saved_preset_key(key) for key in (None, "quick_look", ":save", 3))
    again = saved_preset_for_key(preset.key, tmp_path)
    assert again.name == "Key: test" and again.settings == {"statistics": True}
    delete_saved_preset(again)
    assert not preset.path.exists()
    delete_saved_preset(again)
    with pytest.raises(ConfigError, match="no longer there"):
        saved_preset_for_key(preset.key, tmp_path)


def test_presets_are_kept_where_the_environment_says_then_in_the_users_settings(monkeypatch, tmp_path):
    monkeypatch.setenv(PRESET_DIR_ENV, str(tmp_path / "chosen"))
    assert saved_preset_dir() == tmp_path / "chosen"

    monkeypatch.delenv(PRESET_DIR_ENV)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert saved_preset_dir() == tmp_path / "roaming" / "HaemoLynx" / "presets"
    monkeypatch.delenv("APPDATA")
    assert saved_preset_dir() == tmp_path / "home" / "AppData" / "Roaming" / "HaemoLynx" / "presets"

    monkeypatch.setattr(sys, "platform", "darwin")
    assert saved_preset_dir() == tmp_path / "home" / "Library" / "Application Support" / "HaemoLynx" / "presets"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert saved_preset_dir() == tmp_path / "xdg" / "haemolynx" / "presets"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert saved_preset_dir() == tmp_path / "home" / ".config" / "haemolynx" / "presets"


def test_the_tests_never_see_the_users_own_presets():
    assert saved_presets() == []
    assert "saved_presets" in saved_preset_dir().name
