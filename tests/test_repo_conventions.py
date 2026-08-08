"""Conventions a normal test run cannot check, because it satisfies them itself.

The one here is a `sys.path` trap. ``python -m pytest`` puts the working
directory on ``sys.path``, so ``from tests.foo import ...`` resolves when run
from the repo root -- and CI runs bare ``pytest``, which does not, so the import
fails there and only there. It has cost three green local runs and three red CI
runs. Sibling test modules are importable by their bare name (pytest prepends
the test file's own directory), so that is the form to use.
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).resolve()

#: `from tests.x import`, `import tests.x`, `from tests import x` -- any of them.
IMPORTS_THE_TESTS_PACKAGE = re.compile(r"^\s*(?:from\s+tests[.\s]|import\s+tests\b)", re.M)


def test_no_test_module_imports_its_siblings_through_a_tests_package():
    offenders = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path == THIS_FILE:
            continue
        if IMPORTS_THE_TESTS_PACKAGE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert offenders == [], (
        "These modules import through a `tests` package, which only resolves "
        "under `python -m pytest` from the repo root and fails under the bare "
        f"`pytest` CI runs: {offenders}. Import the sibling by its bare name."
    )
