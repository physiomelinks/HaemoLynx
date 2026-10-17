"""Every package in requirements.txt is installed by `pip install -e .[dev]`.

CI installs only pyproject.toml's dependencies and its `[dev]` extra, so a package that
requirements.txt names but pyproject.toml lacks is missing there, and the tests that import
it fail at collection (retro-14-19-followups/01).
"""
from pathlib import Path

from packaging.requirements import Requirement  # pytest requires packaging, so [dev] installs it
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11, as in CI; pytest itself requires tomli there
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]


def _requirement_lines(text):
    """The requirement on each line of a requirements file, without comments or blanks."""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield Requirement(line)


def _key(req):
    return canonicalize_name(req.name), frozenset(req.extras), str(req.specifier), str(req.marker)


def missing_from_pyproject(requirements_text, pyproject_text):
    """requirements.txt entries that neither `dependencies` nor the `[dev]` extra lists,
    with the same extras, version bounds and environment markers."""
    project = tomllib.loads(pyproject_text)["project"]
    declared = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    have = {_key(Requirement(d)) for d in declared}
    return [str(r) for r in _requirement_lines(requirements_text) if _key(r) not in have]


def test_pyproject_installs_every_package_in_requirements_txt():
    missing = missing_from_pyproject(
        (REPO / "requirements.txt").read_text(), (REPO / "pyproject.toml").read_text())
    assert missing == [], (
        f"requirements.txt names {missing}, which pyproject.toml's `dependencies` and "
        "`[dev]` extra do not list (with the same extras and bounds); CI would not install them")


def test_a_package_dropped_from_pyproject_is_reported():
    requirements = "numpy\ndask[array]  # chunked masks\npytest-cov\nnumba\n"
    pyproject = (
        '[project]\ndependencies = ["numpy", "dask", "numba; python_version < \'3.11\'"]\n'
        '[project.optional-dependencies]\ndev = ["pytest"]\n')
    assert missing_from_pyproject(requirements, pyproject) == [
        "dask[array]", "pytest-cov", "numba"]
