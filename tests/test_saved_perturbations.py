"""Saved perturbations: one perturbation kept on this machine for later runs.

A saved perturbation holds its entry -- name, type and the overrides its type
reads -- less what belongs to one image, and gives back an entry a run's list
can take: the overrides as this schema spells them, those it cannot use
passed over and named (`pipeline/saved_perturbations.py`).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from haemolynx.haemodynamics.perturbations import PerturbationSpec
from haemolynx.parsers.schema import ConfigError, Schema
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.saved_perturbations import (
    PERTURBATION_DIR_ENV,
    delete_saved_perturbation,
    held_entry,
    read_saved_perturbation,
    save_perturbation,
    saved_perturbation_dir,
    saved_perturbation_entry,
    saved_perturbation_for_key,
    saved_perturbation_path,
    saved_perturbations,
)
from haemolynx.pipeline.saved_presets import PRESET_DIR_ENV, saved_preset_dir

SCHEMA = default_schema()

ARTERIOLE = {
    "name": "art_dilate_20",
    "type": "arteriole_diameter_change",
    "overrides": {"arteriole_diameter_change_percent": 20},
}
PERICYTE = {
    "name": "tight pericytes",
    "type": "pericyte_diameter_change",
    "overrides": {
        "pericyte_constriction_factor": 0.6,
        "constriction_length_um": 5.0,
        "use_pericyte_mask_constriction": True,
        "pericyte_mask_path": "C:/data/image_pericytes.tif",
        "pericyte_mask_h5_dataset_name": "exported_data",
    },
}
LISTED_BLOCK = {
    "name": "block_three",
    "type": "capillary_block",
    "overrides": {
        "capillary_block_selection": "vessel_ids",
        "capillary_block_vessel_ids": [4, 9, 12],
        "capillary_block_resistance_factor": 1000.0,
    },
}


# --- what is kept ------------------------------------------------------------


def test_a_saved_perturbation_keeps_its_type_and_options_but_not_this_images_mask(tmp_path):
    saved = save_perturbation(PERICYTE, SCHEMA, directory=tmp_path)

    assert saved.path == tmp_path / "tight pericytes.yaml"
    assert saved.entry == {
        "name": "tight pericytes",
        "type": "pericyte_diameter_change",
        "overrides": {
            "pericyte_constriction_factor": 0.6,
            "constriction_length_um": 5.0,
            "use_pericyte_mask_constriction": True,
        },
    }
    assert saved.left_out == ("pericyte_mask_h5_dataset_name", "pericyte_mask_path")


def test_a_capillary_block_keeps_how_it_chooses_but_not_the_vessels_by_number(tmp_path):
    saved = save_perturbation(LISTED_BLOCK, SCHEMA, directory=tmp_path)

    assert saved.entry["overrides"] == {
        "capillary_block_selection": "vessel_ids",
        "capillary_block_resistance_factor": 1000.0,
    }
    assert saved.left_out == ("capillary_block_vessel_ids",)


def test_an_override_its_type_does_not_read_is_left_out():
    held, left_out = held_entry(
        {**ARTERIOLE, "overrides": {**ARTERIOLE["overrides"], "inlet_p_bc": 9000.0}}, SCHEMA
    )
    assert held["overrides"] == {"arteriole_diameter_change_percent": 20}
    assert left_out == ("inlet_p_bc",)


def test_the_file_holds_an_entry_a_configs_list_takes_as_it_is(tmp_path):
    saved = save_perturbation(ARTERIOLE, SCHEMA, directory=tmp_path)

    text = saved.path.read_text(encoding="utf-8")
    assert text.startswith("# A HaemoLynx saved perturbation.")
    raw = yaml.safe_load(text)
    assert set(raw) == {"saved", "perturbation"}
    assert raw["saved"] == saved.saved
    entry = raw["perturbation"]
    assert entry == ARTERIOLE
    spec = PerturbationSpec.from_entry(entry)
    assert spec.problems == ()
    assert spec.schema_problems(SCHEMA) == ()
    assert not list(tmp_path.glob("*.part"))


def test_saving_under_the_same_name_replaces_the_one_saved_before(tmp_path):
    save_perturbation(ARTERIOLE, SCHEMA, directory=tmp_path)
    again = {**ARTERIOLE, "overrides": {"arteriole_diameter_change_percent": -15}}

    save_perturbation(again, SCHEMA, directory=tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["art_dilate_20.yaml"]
    entry = read_saved_perturbation(tmp_path / "art_dilate_20.yaml").entry
    assert entry["overrides"] == {"arteriole_diameter_change_percent": -15}


# --- what is refused ---------------------------------------------------------


@pytest.mark.parametrize("perturbation_type", ["none", "no_such_type"])
def test_an_entry_with_no_type_to_run_is_not_saved(tmp_path, perturbation_type):
    with pytest.raises(ValueError, match="no type yet"):
        save_perturbation(
            {"name": "unfinished", "type": perturbation_type, "overrides": {}},
            SCHEMA,
            directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_an_entry_the_schema_rejects_is_not_saved(tmp_path):
    bad = {**ARTERIOLE, "overrides": {"arteriole_diameter_change_percent": "lots"}}
    with pytest.raises(ValueError, match="art_dilate_20"):
        save_perturbation(bad, SCHEMA, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_name_with_nothing_in_it_has_no_file():
    with pytest.raises(ValueError, match="needs a name"):
        saved_perturbation_path("  ")


# --- reading them back -------------------------------------------------------


def test_one_saved_comes_back_as_the_entry_it_was(tmp_path):
    saved = save_perturbation(PERICYTE, SCHEMA, directory=tmp_path)

    entry, passed_over = saved_perturbation_entry(
        saved_perturbation_for_key(saved.key, tmp_path), SCHEMA
    )

    assert entry == saved.entry
    assert passed_over == ()


def test_what_a_run_cannot_take_from_a_hand_written_file_is_passed_over_and_named(tmp_path):
    path = tmp_path / "by_hand.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "perturbation": {
                    "name": "by_hand",
                    "type": "arteriole_diameter_change",
                    "overrides": {
                        "arteriole_diameter_change_percent": "15",
                        "arteriole_diameter_chnage_percent": 3,
                        "inlet_p_bc": 9000.0,
                        "viscosity_law": "constant",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    entry, passed_over = saved_perturbation_entry(read_saved_perturbation(path), SCHEMA)

    # Coerced to the setting's kind, as the run reads it.
    assert entry["overrides"] == {"arteriole_diameter_change_percent": 15.0}
    assert passed_over == ("arteriole_diameter_chnage_percent", "inlet_p_bc", "viscosity_law")


def test_a_mask_written_into_the_file_by_hand_is_not_added(tmp_path):
    path = tmp_path / "masked.yaml"
    path.write_text(yaml.safe_dump({"perturbation": PERICYTE}), encoding="utf-8")

    entry, passed_over = saved_perturbation_entry(read_saved_perturbation(path), SCHEMA)

    assert "pericyte_mask_path" not in entry["overrides"]
    assert passed_over == ("pericyte_mask_h5_dataset_name", "pericyte_mask_path")


def test_one_saved_before_a_value_was_renamed_loads_as_its_new_spelling(tmp_path):
    """Through the schema's own upgrade, as a config or a saved preset does:
    here a choice renamed from ``listed`` to ``vessel_ids``."""
    schema = Schema(
        [
            replace(setting, value_aliases={"listed": "vessel_ids"})
            if setting.name == "capillary_block_selection"
            else setting
            for setting in SCHEMA
        ],
        title=SCHEMA.title,
    )
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "perturbation": {
                    "name": "old",
                    "type": "capillary_block",
                    "overrides": {"capillary_block_selection": "listed"},
                }
            }
        ),
        encoding="utf-8",
    )

    entry, passed_over = saved_perturbation_entry(read_saved_perturbation(path), schema)

    assert entry["overrides"] == {"capillary_block_selection": "vessel_ids"}
    assert passed_over == ()
    # Without the alias the old spelling is a value the setting refuses.
    _entry, refused = saved_perturbation_entry(read_saved_perturbation(path), SCHEMA)
    assert refused == ("capillary_block_selection",)


def test_a_type_this_version_does_not_know_is_refused(tmp_path):
    path = tmp_path / "future.yaml"
    path.write_text(
        yaml.safe_dump({"perturbation": {"name": "future", "type": "teleport", "overrides": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="teleport"):
        saved_perturbation_entry(read_saved_perturbation(path), SCHEMA)


def test_every_saved_one_is_listed_by_name_and_a_stray_file_is_passed_over(tmp_path, caplog):
    save_perturbation(PERICYTE, SCHEMA, directory=tmp_path)
    save_perturbation(ARTERIOLE, SCHEMA, directory=tmp_path)
    (tmp_path / "notes.yaml").write_text("just: a note\n", encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("perturbation: [\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        found = saved_perturbations(tmp_path)

    assert [item.name for item in found] == ["art_dilate_20", "tight pericytes"]
    assert [item.type for item in found] == ["arteriole_diameter_change", "pericyte_diameter_change"]
    assert [item.key for item in found] == ["art_dilate_20", "tight pericytes"]
    assert all(item.saved for item in found)
    assert "notes.yaml is not a saved perturbation" in caplog.text
    assert "broken.yaml" in caplog.text
    assert saved_perturbations(tmp_path / "nothing here") == []


def test_one_deleted_is_gone_and_deleting_it_again_does_nothing(tmp_path):
    saved = save_perturbation(ARTERIOLE, SCHEMA, directory=tmp_path)

    delete_saved_perturbation(saved)
    delete_saved_perturbation(saved)

    assert not saved.path.exists()
    with pytest.raises(ConfigError, match="no longer there"):
        saved_perturbation_for_key(saved.key, tmp_path)


# --- where they are kept -----------------------------------------------------


def test_they_are_kept_where_the_environment_says_then_beside_the_presets(monkeypatch, tmp_path):
    monkeypatch.setenv(PERTURBATION_DIR_ENV, str(tmp_path / "chosen"))
    assert saved_perturbation_dir() == tmp_path / "chosen"

    monkeypatch.delenv(PERTURBATION_DIR_ENV)
    monkeypatch.delenv(PRESET_DIR_ENV)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert saved_perturbation_dir() == tmp_path / "roaming" / "HaemoLynx" / "perturbations"
    assert saved_perturbation_dir().parent == saved_preset_dir().parent

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert saved_perturbation_dir() == tmp_path / "home" / ".config" / "haemolynx" / "perturbations"


def test_the_tests_never_see_the_users_own_saved_perturbations():
    assert saved_perturbations() == []
    assert "saved_perturbations" in saved_perturbation_dir().name
