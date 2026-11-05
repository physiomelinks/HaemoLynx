"""What `pip install HaemoLynx` actually gives someone.

Every other test runs against the source tree, with `src` on the path. That
cannot see the ways a distribution differs from a checkout: a module missing
from the wheel, a package directory without `__init__.py` so setuptools skips
it, an import that only resolves because the repository happens to be the
working directory. These tests build the wheel and use it the way a user would.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "haemolynx"

#: A run through the public API, executed in a directory that is not the
#: repository. It lives in its own file so the CI job can run it directly
#: against a wheel installed into an empty environment.
SMOKE_SCRIPT_PATH = Path(__file__).with_name("installed_package_smoke.py")


def _packaged_modules() -> set[str]:
    """Every module the source tree expects to ship, as wheel-relative paths.

    Zip entry names always use forward slashes, whatever the platform building
    the wheel spells its own separator as.
    """
    return {
        path.relative_to(PACKAGE_ROOT.parent).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """Build the wheel once for the module."""
    out_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out_dir),
         str(REPO_ROOT)],
        check=True, capture_output=True, text=True,
    )
    built = list(out_dir.glob("*.whl"))
    assert len(built) == 1, f"expected one wheel, got {built}"
    return built[0]


# --- what the distribution contains -----------------------------------------


@pytest.mark.slow
def test_the_wheel_ships_every_module_in_the_package(wheel):
    """A subpackage with no `__init__.py` is silently skipped by setuptools."""
    shipped = {name for name in zipfile.ZipFile(wheel).namelist() if name.endswith(".py")}
    missing = _packaged_modules() - shipped
    assert not missing, (
        f"modules in src/ that the wheel does not contain: {sorted(missing)}. "
        "Check [tool.setuptools.packages.find] and that each directory has "
        "__init__.py."
    )


@pytest.mark.slow
def test_the_wheel_ships_nothing_but_the_package(wheel):
    """Tests, examples and tutorials are not part of what a user installs."""
    contents = zipfile.ZipFile(wheel).namelist()
    # The metadata directory is named for the distribution and version, so it
    # is taken from the wheel rather than spelled out and left to rot.
    dist_info = f"{wheel.name.split('-')[0]}-{wheel.name.split('-')[1]}.dist-info/"
    strays = [
        name
        for name in contents
        if not name.startswith((f"{PACKAGE_ROOT.name}/", dist_info))
    ]
    assert strays == [], f"unexpected files in the wheel: {strays}"


# --- what the distribution does ---------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_the_installed_package_works_from_outside_the_repository(wheel, tmp_path):
    """Unpack the wheel on its own and drive the public API from elsewhere.

    The wheel is installed into a directory of its own and put on the path
    alone, so every haemolynx import must be satisfied by what the wheel
    contains. The third-party dependencies come from the running interpreter --
    they are not what is under test here; the CI job installs them from scratch.
    """
    site_dir = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
         "--target", str(site_dir), str(wheel)],
        check=True, capture_output=True, text=True,
    )

    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir()
    script = work_dir / SMOKE_SCRIPT_PATH.name
    script.write_text(SMOKE_SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    env = dict(os.environ)
    # The wheel, and nothing of the repository: an import that only worked
    # because `src` was on the path fails here, which is the point.
    env["PYTHONPATH"] = str(site_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYVISTA_OFF_SCREEN"] = "true"
    completed = subprocess.run(
        [sys.executable, "-P", str(script)] if sys.version_info >= (3, 11)
        else [sys.executable, str(script)],
        cwd=work_dir, capture_output=True, text=True, env=env,
    )
    assert completed.returncode == 0, (
        f"the installed package could not run:\n{completed.stdout}\n{completed.stderr}"
    )

    payload = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert payload, f"smoke script printed no result:\n{completed.stdout}"
    result = json.loads(payload[-1][len("RESULT "):])

    # It ran from the unpacked wheel, not from a stray copy of the source tree.
    assert result["package_dir"] == str(site_dir / "haemolynx")
    assert str(REPO_ROOT / "src") not in result["package_dir"]

    assert result["edges"] == 2
    assert result["total_length"] == pytest.approx(200.0)
    # Two identical vessels: equal resistance, and a real number for each.
    assert len(result["resistances"]) == 2
    assert result["resistances"][0] == pytest.approx(result["resistances"][1])
    assert result["resistances"][0] > 0.0
    assert any(name.endswith(".vtp") for name in result["vtk_files"])


@pytest.mark.slow
def test_the_config_a_fresh_install_writes_is_complete(wheel, tmp_path):
    """`write_default_config` is how an installed user gets a runnable config."""
    from haemolynx.parsers import load_config
    from haemolynx.pipeline import default_schema, write_default_config

    schema = default_schema()
    written = write_default_config(tmp_path / "config.yaml")
    settings = load_config(written, schema)

    assert set(settings) == set(schema.names)


def test_the_distribution_name_matches_what_users_pip_install():
    """`pip install HaemoLynx` must resolve to this package, not a near-miss."""
    from importlib import metadata

    distribution = metadata.distribution("HaemoLynx")
    assert distribution.metadata["Name"] == "HaemoLynx"
    assert metadata.distribution("HaemoLynx").locate_file("haemolynx").name == "haemolynx"


def test_the_version_is_single_sourced():
    """A version in two places drifts; the metadata must agree with the module."""
    from importlib import metadata

    import haemolynx

    assert metadata.version("HaemoLynx") == haemolynx.__version__


def test_no_module_imports_the_repository_layout():
    """A library that reads `examples/` or `tests/` breaks once installed."""
    offenders = {}
    for module in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = module.read_text(encoding="utf-8")
        hits = [
            needle
            for needle in ('"examples/', "'examples/", '"tests/', "'tests/",
                           '"tutorials/', "'tutorials/")
            if needle in source
        ]
        if hits:
            offenders[str(module.relative_to(PACKAGE_ROOT))] = hits
    assert offenders == {}, (
        f"library code referencing repository directories: {offenders}. Those "
        "paths do not exist for an installed user."
    )


def test_declared_console_scripts_exist_after_installation():
    """Anything in [project.scripts] must actually land on PATH.

    There are no console scripts today, so this passes vacuously; it starts
    doing work the moment one is declared.
    """
    from importlib import metadata

    scripts = metadata.distribution("HaemoLynx").entry_points.select(group="console_scripts")
    for script in scripts:
        assert shutil.which(script.name), (
            f"{script.name} is declared in [project.scripts] but is not on PATH "
            "after installation."
        )
