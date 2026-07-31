"""Every example's config file must match its schema, key for key.

The schema is the single declaration a GUI, the CLI and the YAML file are all
built from, so a setting that exists in one but not the other is the failure
mode this whole design exists to prevent. These tests run over every example
registered in ``examples/regenerate_configs.py``, so a new example is covered
the moment it is added there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

yaml = pytest.importorskip("yaml")

from ImageLynx.parsers import Schema, dump_config, load_config  # noqa: E402
from regenerate_configs import CONFIGS, regenerate  # noqa: E402

CASES = sorted(CONFIGS.items())
IDS = [Path(config).stem for config, _ in CASES]


def _schema(module_name: str) -> Schema:
    return __import__(module_name).SCHEMA


def _keys_in(config_path: Path) -> set[str]:
    """Every setting key in the file, flattening the one level of sections."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    keys: set[str] = set()
    for key, value in raw.items():
        if isinstance(value, dict):
            keys.update(value)
        else:
            keys.add(key)
    return keys


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_every_config_key_has_a_schema_entry(config_relpath, module_name):
    schema = _schema(module_name)
    extra = _keys_in(REPO_ROOT / config_relpath) - set(schema.names)
    assert not extra, (
        f"{config_relpath} contains settings missing from {module_name}.SCHEMA: "
        f"{sorted(extra)}"
    )


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_every_schema_entry_appears_in_the_config(config_relpath, module_name):
    """A GUI renders the schema; a setting absent from the file is undiscoverable."""
    schema = _schema(module_name)
    missing = set(schema.names) - _keys_in(REPO_ROOT / config_relpath)
    assert not missing, (
        f"{config_relpath} is missing settings declared in {module_name}.SCHEMA: "
        f"{sorted(missing)}. Run python examples/regenerate_configs.py"
    )


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_the_committed_config_loads_and_validates(config_relpath, module_name):
    settings = load_config(REPO_ROOT / config_relpath, _schema(module_name))
    assert set(settings) == set(_schema(module_name).names)


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_the_committed_config_is_what_the_generator_would_write(
    config_relpath, module_name, tmp_path
):
    """Guards against hand-edits that drop a setting's documentation."""
    schema = _schema(module_name)
    committed = REPO_ROOT / config_relpath
    regenerated = dump_config(
        tmp_path / committed.name, schema, values=load_config(committed, schema)
    )
    assert regenerated.read_text(encoding="utf-8") == committed.read_text(
        encoding="utf-8"
    ), f"{config_relpath} is stale. Run python examples/regenerate_configs.py"


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_every_setting_is_documented_and_grouped(config_relpath, module_name):
    """Help text and section become the GUI tooltip and form grouping."""
    for setting in _schema(module_name):
        assert setting.help.strip(), f"{setting.name} has no help text"
        assert not setting.help.endswith("."), (
            f"{setting.name} help should read as a label, without a full stop"
        )
        assert setting.section.strip(), f"{setting.name} has no section"


@pytest.mark.parametrize("config_relpath,module_name", CASES, ids=IDS)
def test_the_schema_describes_itself_for_a_gui(config_relpath, module_name):
    import json

    described = _schema(module_name).describe()
    json.dumps(described)
    assert described["sections"], f"{module_name}.SCHEMA has no sections"


def test_regenerating_is_idempotent_and_preserves_committed_values(tmp_path):
    """Running the generator twice must not keep changing the files."""
    before = {
        relpath: (REPO_ROOT / relpath).read_text(encoding="utf-8") for relpath in CONFIGS
    }
    try:
        regenerate()
        after = {
            relpath: (REPO_ROOT / relpath).read_text(encoding="utf-8")
            for relpath in CONFIGS
        }
        assert after == before
    finally:
        for relpath, text in before.items():
            (REPO_ROOT / relpath).write_text(text, encoding="utf-8")
