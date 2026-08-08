"""The napari plugin declaration, checked without installing napari.

A plugin that is declared wrongly does not fail loudly: napari simply does not
list it, and the only symptom is an empty Plugins menu. The manifest, the entry
point that points at it, and the callable it names are all checkable here --
and npe2, which needs no Qt, validates the manifest itself.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "src" / "haemolynx" / "gui" / "napari.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


# --- the manifest itself ----------------------------------------------------


def test_the_manifest_exists_where_the_entry_point_says():
    assert MANIFEST.is_file(), f"no manifest at {MANIFEST}"


def test_npe2_accepts_the_manifest():
    """npe2 is napari's own validator, and needs no Qt to run it."""
    npe2 = pytest.importorskip("npe2")
    parsed = npe2.PluginManifest.from_file(MANIFEST)
    assert parsed.contributions.widgets, "the manifest declares no widget"


def test_npe2_discovers_the_plugin_from_the_installed_entry_point():
    """The check that matters: would napari find this plugin at all?

    Every other test here reads the manifest directly. This one goes the way
    napari does -- entry point, then manifest, then contributions -- which is
    the only path that catches a manifest that is valid on its own but wrong
    about the distribution it belongs to.
    """
    npe2 = pytest.importorskip("npe2")

    manager = npe2.PluginManager()
    manager.discover()
    declared = PYPROJECT.read_text(encoding="utf-8").split('name = "', 1)[1].split('"', 1)[0]
    if declared not in manager._manifests:
        pytest.skip(f"{declared} is not installed in this environment")

    found = manager.get_manifest(declared)
    assert [widget.display_name for widget in found.contributions.widgets] == [
        "Pipeline settings",
        "Run a saved config",
    ]


def test_the_plugin_name_is_the_distribution_name_exactly(manifest):
    """npe2 compares against the *declared* name, not its normalised form.

    `name: haemolynx` looks right -- it is what PyPI normalises to, and what
    every other tool accepts -- but npe2 refuses it with "The name field in the
    manifest ('haemolynx') must match the package name ('HaemoLynx')" and the
    plugin is silently absent from napari.
    """
    declared = PYPROJECT.read_text(encoding="utf-8").split('name = "', 1)[1].split('"', 1)[0]
    assert manifest["name"] == declared


def test_command_ids_are_prefixed_with_the_plugin_name(manifest):
    """npe2 requires it, and rejects the whole manifest when it is not so."""
    for command in manifest["contributions"]["commands"]:
        assert command["id"].startswith(f"{manifest['name']}."), (
            f"{command['id']} must start with '{manifest['name']}.'"
        )


def test_every_widget_names_a_command_that_exists(manifest):
    command_ids = {command["id"] for command in manifest["contributions"]["commands"]}
    for widget in manifest["contributions"]["widgets"]:
        assert widget["command"] in command_ids, (
            f"widget {widget['display_name']} names an undeclared command"
        )


def test_every_command_points_at_something_importable(manifest):
    """`python_name` is resolved at click time, so a typo shows up as a crash."""
    for command in manifest["contributions"]["commands"]:
        module_name, _, attribute = command["python_name"].partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), (
            f"{command['id']} names {command['python_name']}, which does not exist"
        )
        assert callable(getattr(module, attribute))


def test_importing_the_widget_module_does_not_need_a_gui():
    """The library must import on a machine with no napari and no Qt.

    Checked in a fresh interpreter rather than in this one: napari ships a
    pytest plugin, so `import napari` has already happened in any test session
    where napari is installed -- which is exactly the environment the GUI runs
    in. Asserting on this process's sys.modules would test the test runner.
    """
    probe = (
        "import sys; import haemolynx.gui._widget; "
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'napari', 'magicgui', 'qtpy', 'PyQt6', 'PyQt5', 'PySide6'}); "
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"importing the widget module failed:\n{result.stdout}\n{result.stderr}"
    )
    assert result.stdout.strip() == "", (
        f"importing haemolynx.gui._widget pulled in {result.stdout.strip()}. Keep "
        "those imports inside the functions, so the library works without a GUI."
    )


def test_importing_the_library_does_not_need_a_gui():
    """The same, for the package as a whole: `import haemolynx` must stay cheap."""
    probe = (
        "import sys; import haemolynx; "
        "print('napari' in sys.modules or 'qtpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --- how it is declared and shipped -----------------------------------------


def test_pyproject_declares_the_manifest_entry_point():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '[project.entry-points."napari.manifest"]' in text
    assert 'haemolynx = "haemolynx.gui:napari.yaml"' in text


def test_the_manifest_is_declared_as_package_data():
    """Without this the yaml is missing from the wheel and the plugin vanishes."""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '"haemolynx.gui" = ["napari.yaml"]' in text


def test_napari_is_an_extra_not_a_dependency():
    """napari needs Python 3.11; the library supports 3.9 and must keep doing so."""
    text = PYPROJECT.read_text(encoding="utf-8")
    runtime = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "napari" not in runtime, "napari must not be a runtime dependency"
    extras = text.split("[project.optional-dependencies]", 1)[1].split("[project", 1)[0]
    assert "napari[pyqt6]>=0.8" in extras


def test_the_napari_extra_brings_a_qt_binding():
    """napari declares no Qt binding, and exits with "No Qt bindings found".

    `pip install "HaemoLynx[napari]"` has to give a panel that opens, so the
    extra names one. `napari-plugin` is the same thing without it, for someone
    who already runs napari and has chosen their own.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    extras = text.split("[project.optional-dependencies]", 1)[1].split("[project", 1)[0]
    napari_extra = extras.split("napari = ", 1)[1].split("\n", 1)[0]
    assert "pyqt" in napari_extra.lower() or "pyside" in napari_extra.lower()
    plugin_extra = extras.split("napari-plugin = ", 1)[1].split("\n", 1)[0]
    assert "pyqt" not in plugin_extra.lower()


def test_the_napari_framework_classifier_is_declared():
    """It is what makes the package findable as a plugin once published."""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '"Framework :: napari"' in text


def test_more_than_one_widget_is_contributed(manifest):
    """This is what makes the menu read "HaemoLynx" rather than "x (HaemoLynx)".

    napari renders a lone widget as `menu_item_template = "{1} ({0})"`, and
    `needs_full_title` is True whenever a plugin provides only one, so the
    suffix cannot be turned off. With two or more it builds a submenu titled
    with the plugin's display name and lists the bare widget names inside.
    """
    assert len(manifest["contributions"]["widgets"]) >= 2, (
        "napari only groups a plugin's widgets under its own name when there "
        "is more than one; with a single widget the menu shows "
        "'<widget> (HaemoLynx)'."
    )


def test_the_display_name_is_what_should_appear_in_the_menu(manifest):
    assert manifest["display_name"] == "HaemoLynx"
