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

from haemolynx.parsers import dump_config, load_config  # noqa: E402
from regenerate_configs import CONFIGS, regenerate, schema_for as _schema  # noqa: E402

CASES = sorted(CONFIGS.items())
IDS = [Path(config).stem for config, _ in CASES]


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


#: What ``brain_pipeline_config.yaml`` has always configured the sweep with.
#: Pinned by value, not by "it loads": these moved from a schema beside the
#: example into the package's own, and a migration that quietly reset one of
#: them to a package default would still load.
BRAIN_SWEEP_VALUES = {
    "pericyte_dilation_min_percent": 1,
    "pericyte_dilation_max_percent": 30,
    "pericyte_dilation_step_percent": 1,
    "inlet_pressure_min_pa": 4500,
    "inlet_pressure_max_pa": 6000,
    "inlet_pressure_step_pa": 500,
    "sweep_output_dir": "examples/outputs/brain_dilation_sweep",
}


def test_the_sweep_settings_are_declared_in_the_package_schema():
    """The napari panel builds its form from `default_schema()` and nothing else.

    While these lived in `examples/brain_pipeline_schema.py` the panel could
    not show them at all, whatever tab claimed the section.
    """
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    section = set(schema.section_names("Perturbation runs"))
    assert section == {"run_pericyte_dilation_sweep", *BRAIN_SWEEP_VALUES}


def test_the_brain_schema_adds_nothing_the_pipeline_schema_does_not_have():
    """It is a title over the package schema now; adding them twice would raise.

    `Schema(list(default_schema()) + SWEEP_SETTINGS)` is a duplicate-setting
    error once they are declared in the package, so the example's own list has
    to go rather than be kept alongside.
    """
    import brain_pipeline_schema
    from haemolynx.pipeline import default_schema

    assert not hasattr(brain_pipeline_schema, "SWEEP_SETTINGS")
    assert brain_pipeline_schema.SCHEMA.names == default_schema().names


def test_the_brain_config_still_configures_the_sweep_it_always_did():
    from haemolynx.pipeline import default_schema

    settings = load_config(
        REPO_ROOT / "examples" / "brain_pipeline_config.yaml", default_schema()
    )

    assert settings["run_pericyte_dilation_sweep"] is True
    for name, expected in BRAIN_SWEEP_VALUES.items():
        actual = settings[name]
        actual = actual.as_posix() if isinstance(actual, Path) else actual
        assert actual == expected, f"{name} is {actual!r}, was {expected!r}"


def test_the_constriction_geometry_the_sweep_reads_is_declared():
    """`pericyte_sweep.py` read these with hardcoded fallbacks and no declaration.

    An undeclared setting cannot be put in a config file at all -- the loader
    rejects the key -- so the 40/100 um the sweep ran with could not be
    changed without editing the source.
    """
    from haemolynx.haemodynamics import pericyte_sweep
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    for name, fallback in (
        ("constriction_length_um", 40.0),
        ("constriction_spacing_um", 100.0),
    ):
        assert name in schema, f"{name} is read by the sweep but not declared"
        assert schema[name].default == fallback, (
            f"{name} must default to the value the sweep fell back to, or a "
            "run's numbers change without anyone editing a config"
        )
        assert schema[name].section == "Diameters and pericytes"
    source = Path(pericyte_sweep.__file__).read_text(encoding="utf-8")
    for name in ("constriction_length_um", "constriction_spacing_um"):
        assert name in source


def test_the_resistance_config_names_boundaries_at_both_ends():
    """A config that selects no outlets cannot run, and this one did not.

    `outlet_node_selection_method` was "coordinates" with an empty
    `outlet_node_coordinates`, so the run died at the boundary stage every
    time. Nothing caught it because nothing ran the shipped config: the
    branch-comparison tool replaced these settings with its own, and the
    integration tests pass their own.
    """
    from haemolynx.parsers import load_config
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    settings = load_config(
        REPO_ROOT / "examples" / "resistance_pipeline_config.yaml", schema
    )

    for role in ("inlet", "outlet"):
        method = settings[f"{role}_node_selection_method"]
        source = {
            "coordinates": f"{role}_node_coordinates",
            "volume": f"{role}_node_volumes",
        }.get(method)
        if source is None:
            continue  # edge_percent and friends need no values of their own
        assert settings[source], (
            f"{role}_node_selection_method is {method!r} but {source} is empty, "
            "so the run selects no nodes for that end and stops"
        )


def test_the_resistance_config_points_at_an_image_that_is_here():
    """It named `examples/images/brain_microvessels.tiff`, which never existed.

    Preflight stops on it, so `python examples/resistance_network_pipeline.py`
    failed before doing anything -- the same error the napari panel showed.
    """
    from haemolynx.parsers import load_config
    from haemolynx.pipeline import default_schema

    settings = load_config(
        REPO_ROOT / "examples" / "resistance_pipeline_config.yaml", default_schema()
    )
    named = REPO_ROOT / settings["input_path"]
    assert named.name == "Nerve_capillaries.tif", (
        f"the config names {settings['input_path']}, which is not the dataset "
        "this repository ships"
    )
