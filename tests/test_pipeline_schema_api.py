"""The pipeline's schema is part of the installed package, not example code.

`resolve_settings` cannot be called without a schema, so while the schema lived
in `examples/resistance_pipeline_schema.py` a pip-installed copy of HaemoLynx
had no way to configure a run at all. These tests pin the public route in --
`default_schema()` and `write_default_config()` -- and, in a subprocess with
only the package importable, prove it does not reach back into the repository.
"""
from __future__ import annotations

import os
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
        "cut_network_at_large_vessel_volumes",
    ):
        assert name in schema, f"{name} missing from the pipeline schema"
    assert "cut_large_vessel_sample_densely" not in schema
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


# --- nested/gated settings must not be silently read while hidden ----------


def _non_default_probe_value(setting):
    """A value that differs from *setting*'s default, generic across kinds.

    Prefers a declared bound (minimum/maximum) over an arbitrary offset, so
    it stays inside a narrow range (a 0.0-1.0 fraction, say) instead of
    guessing a step size that happens to land out of bounds. Returns None
    when no value can be constructed generically (mapping/any/list kinds),
    which the caller skips rather than guesses at.
    """
    default = setting.default
    kind = setting.kind
    if kind == "bool":
        return not bool(default)
    if kind == "choice":
        for choice in setting.choices or ():
            if choice != default:
                return choice
        return None
    if kind in ("int", "float"):
        for bound in (setting.minimum, setting.maximum):
            if bound is not None and bound != default:
                return bound
        base = default if default is not None else 0
        return base + (1 if kind == "int" else 1.0)
    if kind in ("str", "path"):
        base = "" if default is None else str(default)
        return base + "_probe"
    return None


def _first_unmet_prerequisite(setting) -> tuple[str, bool]:
    """One ``(setting_name, value)`` pair that makes *setting*'s first
    prerequisite unmet -- ``all()`` over requires means unmeeting just one
    is enough to make the whole setting ineffective."""
    prerequisite = setting.requires[0]
    if prerequisite.startswith("!"):
        return prerequisite[1:], True
    return prerequisite, False


def test_every_requires_prerequisite_names_a_real_setting():
    """A typo'd or renamed prerequisite is invisible to the ineffectiveness
    check below: ``is_prerequisite_met`` reads an unknown key as `False` via
    a plain dict `.get`, the same as a real setting deliberately turned off,
    so a stale `requires` entry would still "work" there by accident. This
    is the one check that actually looks at the name itself.
    """
    schema = default_schema()
    names = set(schema.names)
    bad = [
        (setting.name, prerequisite)
        for setting in schema
        for prerequisite in (setting.requires or ())
        if (prerequisite[1:] if prerequisite.startswith("!") else prerequisite) not in names
    ]
    assert bad == []


def test_every_gated_setting_is_ineffective_when_its_prerequisite_is_off():
    """Nested options must not silently be read once their prerequisite is
    off -- the GUI hides or greys them out on exactly that condition, and
    Schema.ineffective_settings is the general mechanism that is supposed to
    catch every such setting, not just the one hand-picked case in
    test_parsers_schema.py. This walks the real pipeline schema's 180+ gated
    settings and proves the claim for each one a probe value can be built
    for, so a `requires` tuple that stops matching what a stage actually
    reads (a typo, a renamed setting) shows up here instead of only as a
    silently-wrong run.
    """
    schema = default_schema()
    defaults = schema.defaults()
    checked = 0
    skipped = []

    for setting in schema:
        if not setting.requires:
            continue
        probe = _non_default_probe_value(setting)
        if probe is None:
            skipped.append(setting.name)
            continue
        unmet_name, unmet_value = _first_unmet_prerequisite(setting)
        resolved = {
            **defaults,
            setting.name: setting.coerce(probe),
            unmet_name: unmet_value,
        }
        messages = schema.ineffective_settings(resolved)
        assert any(f"Setting '{setting.name}'" in message for message in messages), (
            f"{setting.name} requires {setting.requires!r}, but setting it to "
            f"{probe!r} while {unmet_name}={unmet_value!r} did not get flagged "
            "as ineffective -- something may still read it while its own "
            "prerequisite is off."
        )
        checked += 1

    # A floor, not the exact count: proves the probe-value generator is
    # actually covering the bulk of the schema's gated settings rather than
    # skipping nearly all of them without anyone noticing.
    assert checked > 100, (
        f"only checked {checked} of the schema's gated settings "
        f"(skipped, no generic probe value: {skipped})"
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


def isolated_import_env(pythonpath: Path) -> dict:
    """This process's environment with `pythonpath` as the only import root.

    Isolation here means one thing -- a known `PYTHONPATH` -- and it is spelled
    as a replacement rather than a hand-built environment on purpose. An
    interpreter cannot be started in an empty environment: on Windows one
    without `SYSTEMROOT` cannot initialise its socket layer and dies importing
    `_overlapped`, so an environment listing only the paths the test cares
    about tests nothing but the platform. What the environment must not do is
    smuggle a repository checkout in through an inherited `PYTHONPATH`; that
    the checkout really is out of reach is then asserted by the subprocess
    itself, which is the only place it can be observed.
    """
    return {**os.environ, "PYTHONPATH": str(pythonpath)}


def test_the_isolated_environment_replaces_only_the_import_path(monkeypatch):
    """Everything the platform needs to start an interpreter must survive."""
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    env = isolated_import_env(SRC)

    assert env["PYTHONPATH"] == str(SRC), "an inherited PYTHONPATH leaked through"
    assert {name: value for name, value in env.items() if name != "PYTHONPATH"} == {
        name: value for name, value in os.environ.items() if name != "PYTHONPATH"
    }


def test_a_bare_package_import_can_configure_a_run(tmp_path):
    """No repository, no examples/ on the path: still able to write a config.

    This is the failure the move exists to fix, so it is checked the only way
    that is honest -- in a fresh interpreter that cannot see this checkout.
    """
    script = textwrap.dedent(
        f"""
        import importlib.util
        import pathlib
        import sys

        repo_root = pathlib.Path({str(REPO_ROOT)!r})
        reachable = [
            entry
            for entry in sys.path
            if entry
            and pathlib.Path(entry).resolve() in (repo_root, repo_root / "examples")
        ]
        assert not reachable, "the repository is importable from: " + repr(reachable)
        for name in ("examples", "resistance_network_pipeline", "pipeline_presets"):
            assert importlib.util.find_spec(name) is None, name + " is importable"

        import haemolynx
        from haemolynx.parsers import load_config
        from haemolynx.pipeline import default_schema, write_default_config

        package_dir = pathlib.Path(haemolynx.__file__).resolve().parent
        assert package_dir == pathlib.Path({str(SRC)!r}).resolve() / "haemolynx", package_dir

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
        env=isolated_import_env(SRC),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip().splitlines()[-1]) == len(default_schema())
    assert (tmp_path / "cfg.yaml").is_file()
