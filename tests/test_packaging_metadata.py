"""What `pyproject.toml` declares must match what the repository actually has.

Packaging metadata is only checked when someone builds and uploads, which is
too late to find out that a floor was never tested, that `py.typed` is not
installed, or that the licence the project claims is not the one in the file.
Each of those is a statement about the repository, so each is checked here.
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


# --- licence ----------------------------------------------------------------

LICENSE_FILE = REPO_ROOT / "LICENSE"
COPYRIGHT_HOLDERS = ("Finbar Argus", "Harvey Davis")


def test_the_licence_file_exists_and_is_apache_2():
    """Without a licence, the default is all-rights-reserved: nobody may use it."""
    text = LICENSE_FILE.read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text


def test_the_licence_names_the_copyright_holders():
    """Apache 2.0's appendix is boilerplate until the holders are filled in."""
    text = LICENSE_FILE.read_text(encoding="utf-8")
    for holder in COPYRIGHT_HOLDERS:
        assert holder in text, f"{holder} is not named in LICENSE"
    assert "[name of copyright owner]" not in text, (
        "the Apache appendix still has its placeholder; fill in the year and holders"
    )


def test_pyproject_declares_the_same_licence_as_the_file():
    """A metadata licence that disagrees with LICENSE misleads every consumer."""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in text, (
        "declare the SPDX expression, so the wheel carries License-Expression"
    )
    assert 'license-files = ["LICENSE"]' in text, (
        "LICENSE must be listed, or it is not shipped in the distribution"
    )


def test_every_author_is_a_copyright_holder():
    """The people named in the metadata and in the licence must be the same."""
    text = PYPROJECT.read_text(encoding="utf-8")
    block = text.split("authors = [", 1)[1].split("]", 1)[0]
    for holder in COPYRIGHT_HOLDERS:
        assert holder in block, f"{holder} holds copyright but is not an author"
