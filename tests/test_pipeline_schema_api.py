"""The pipeline's schema is part of the installed package, not example code.

`resolve_settings` cannot be called without a schema, so while the schema lived
in `examples/resistance_pipeline_schema.py` a pip-installed copy of HaemoLynx
had no way to configure a run at all. These tests pin the public route in --
`default_schema()` and `write_default_config()` -- and, in a subprocess with
only the package importable, prove it does not reach back into the repository.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from haemolynx.parsers import ConfigError, Schema, load_config  # noqa: E402
from haemolynx.pipeline import default_schema, write_default_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


# --- the schema itself -----------------------------------------------------


def test_default_schema_is_the_pipelines_settings():
    schema = default_schema()
    assert isinstance(schema, Schema)
    for name in (
        "input_path",
        "voxel_size_override_xyz",
        "inlet_p_bc",
        "vtk_output_prefix",
    ):
        assert name in schema, f"{name} missing from the pipeline schema"
    assert schema.title


def test_default_schema_is_stable_between_calls():
    """A GUI and a run must be configuring the same settings."""
    assert default_schema().names == default_schema().names


def test_the_schema_module_is_importable_without_the_examples_directory():
    """`import haemolynx.pipeline.schema` must not need a repository checkout."""
    import haemolynx.pipeline.schema as module

    assert Path(module.__file__).is_relative_to(SRC)
    assert module.SCHEMA is default_schema()


def test_no_path_default_points_inside_the_examples_directory():
    """Library defaults describe the user's working directory, not this repo."""
    inside_repo = [
        setting.name
        for setting in default_schema()
        if setting.kind == "path"
        and setting.default is not None
        and str(setting.default).startswith("examples/")
    ]
    assert not inside_repo, (
        "path defaults still point at this repository's examples/ tree: "
        f"{inside_repo}"
    )


# --- the generated config file ---------------------------------------------


def test_write_default_config_writes_every_setting_with_its_documentation(tmp_path):
    path = write_default_config(tmp_path / "cfg.yaml")
    text = path.read_text(encoding="utf-8")
    schema = default_schema()

    written = yaml.safe_load(text)
    keys = {key for section in written.values() for key in section}
    assert keys == set(schema.names)
    for setting in schema:
        assert f"# {setting.help}" in text, f"{setting.name} lost its help comment"


def test_the_generated_config_loads_back_as_the_schema_defaults(tmp_path):
    path = write_default_config(tmp_path / "cfg.yaml")
    assert load_config(path, default_schema()) == default_schema().defaults()


def test_write_default_config_keeps_the_values_it_is_given(tmp_path):
    path = write_default_config(
        tmp_path / "cfg.yaml", values={"inlet_p_bc": 1234.0, "do_skeletonize": False}
    )
    settings = load_config(path, default_schema())
    assert settings["inlet_p_bc"] == 1234.0
    assert settings["do_skeletonize"] is False


def test_write_default_config_refuses_a_value_the_schema_rejects(tmp_path):
    with pytest.raises(ConfigError, match="not_a_setting"):
        write_default_config(tmp_path / "cfg.yaml", values={"not_a_setting": 1})


def test_write_default_config_accepts_an_extended_schema(tmp_path):
    """Extending the pipeline schema is how the whole-brain example works."""
    from haemolynx.parsers import Setting

    extended = Schema(
        list(default_schema())
        + [Setting(name="my_sweep", kind="bool", default=True, help="Sweep", section="Mine")]
    )
    path = write_default_config(tmp_path / "cfg.yaml", schema=extended)
    assert load_config(path, extended)["my_sweep"] is True


# --- the installed-package case --------------------------------------------


def test_a_bare_package_import_can_configure_a_run(tmp_path):
    """No repository, no examples/ on the path: still able to write a config.

    This is the failure the move exists to fix, so it is checked the only way
    that is honest -- in a fresh interpreter that cannot see this checkout.
    """
    script = textwrap.dedent(
        """
        from haemolynx.parsers import load_config
        from haemolynx.pipeline import default_schema, write_default_config

        write_default_config("cfg.yaml")
        settings = load_config("cfg.yaml", default_schema())
        assert set(settings) == set(default_schema().names)
        print(len(default_schema()))
"""
    )
    (tmp_path / "run.py").write_text(script, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip().splitlines()[-1]) == len(default_schema())
    assert (tmp_path / "cfg.yaml").is_file()
