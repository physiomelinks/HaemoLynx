"""The declared dependency floors must be the ones that were actually tested.

`pyproject.toml` promises a lower bound for each dependency, and
`constraints-floor.txt` pins exactly those versions so CI can run the suite
against them. The promise is only worth something if the two agree, so that is
what these tests check. A floor raised in one file and not the other is the
failure this prevents.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONSTRAINTS = REPO_ROOT / "constraints-floor.txt"

REQUIREMENT = re.compile(r'^\s*"(?P<name>[A-Za-z0-9_.-]+)\s*(?P<spec>[^"]*)"')
PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s#]+)")


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def _declared_dependencies() -> dict[str, str]:
    """Every runtime dependency in pyproject, as name -> specifier."""
    text = PYPROJECT.read_text(encoding="utf-8")
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    found = {}
    for line in block.splitlines():
        match = REQUIREMENT.match(line)
        if match:
            found[_normalise(match["name"])] = match["spec"].strip()
    return found


def _pinned_floors() -> dict[str, str]:
    """Every pin in the constraints file, as name -> version."""
    found = {}
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            found[_normalise(match["name"])] = match["version"]
    return found


def test_every_dependency_declares_a_lower_bound():
    """An unbounded dependency is a promise to work with versions never tried."""
    unbounded = [
        name for name, spec in _declared_dependencies().items() if ">=" not in spec
    ]
    assert unbounded == [], (
        f"dependencies without a lower bound: {unbounded}. Give each one the "
        "oldest version the suite passes on."
    )


def test_the_constraints_file_covers_every_dependency():
    missing = set(_declared_dependencies()) - set(_pinned_floors())
    assert not missing, (
        f"{CONSTRAINTS.name} does not pin: {sorted(missing)}. CI cannot test a "
        "floor it does not install."
    )


def test_the_constraints_file_pins_nothing_extra():
    extra = set(_pinned_floors()) - set(_declared_dependencies())
    assert not extra, (
        f"{CONSTRAINTS.name} pins packages that are not dependencies: {sorted(extra)}"
    )


@pytest.mark.parametrize("name", sorted(_declared_dependencies()))
def test_each_pin_is_exactly_the_declared_floor(name):
    """`numpy>=1.23` must be tested as `numpy==1.23.0`, not something newer."""
    declared = _declared_dependencies()[name]
    pinned = _pinned_floors()[name]
    floor = declared.split(">=", 1)[1].split(",")[0].strip()

    pinned_parts = pinned.split(".")
    floor_parts = floor.split(".")
    assert pinned_parts[: len(floor_parts)] == floor_parts, (
        f"{name}: pyproject says >={floor} but {CONSTRAINTS.name} pins {pinned}. "
        "The tested version must be the declared floor."
    )


def test_the_type_marker_is_declared_as_package_data():
    """`py.typed` is only honoured if it is installed alongside the modules."""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert (REPO_ROOT / "src" / "ImageLynx" / "py.typed").exists()
    assert 'ImageLynx = ["py.typed"]' in text, (
        "py.typed exists but is not listed in [tool.setuptools.package-data]; "
        "it would not be installed."
    )
