"""The quick tier's collectors, each on fabricated facts: a workflow, a root,
a few modules."""
from __future__ import annotations

import textwrap

from release_review import quick
from release_review.ladder import FAIL, PASS

WORKFLOW = textwrap.dedent("""\
    name: Tests
    on:
      pull_request:
      push:
        branches: [main, Devel_GUI_improvements_HD]
      schedule:
        - cron: "17 3 * * 1"
    jobs:
      lint:
        runs-on: ubuntu-latest
        steps:
          - run: ruff check .
      pytest:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: ["3.9", "3.12"]
        steps:
          - run: |
              pip install -e .[dev]
              pytest
      windows:
        runs-on: windows-latest
        steps:
          - run: pip install -e .[dev]
          - run: pytest -m "not slow and not gui" -n auto
      installed-package:
        runs-on: ubuntu-latest
        steps:
          - run: python -m build --wheel --outdir dist/
          - run: /tmp/fresh/bin/python installed_package_smoke.py
          - run: pytest tests/test_installed_package.py -v
      floors:
        runs-on: ubuntu-latest
        steps:
          - run: pip install -c constraints-floor.txt -e ".[dev]"
      gui:
        runs-on: ubuntu-latest
        steps:
          - run: python -m npe2 validate HaemoLynx
          - run: xvfb-run -a pytest -m gui -v
    """)


def _facts(workflow=WORKFLOW):
    return quick.ci_facts({"pytest-pr.yml": workflow})


def test_the_workflow_is_read_job_by_job():
    facts = _facts()
    assert set(facts.triggers["pytest-pr.yml"]) == {"pull_request", "push", "schedule"}
    assert facts.runners["pytest-pr.yml:windows"] == ["windows-latest"]
    assert facts.python_versions["pytest-pr.yml:pytest"] == ["3.9", "3.12"]
    expressions = {(run.job.split(":")[1], run.marker_expression, run.names_files) for run in facts.pytest_runs}
    assert ("pytest", None, False) in expressions
    assert ("windows", "not slow and not gui", False) in expressions
    assert ("gui", "gui", False) in expressions
    assert ("installed-package", None, True) in expressions


def test_tests_4a_needs_the_whole_suite_and_the_gui_tests_on_pull_requests():
    result = quick.tests_4a(_facts())
    assert result.status == PASS
    assert result.value == {"whole_suite": ["pytest-pr.yml:pytest"], "gui": ["pytest-pr.yml:gui"]}
    without_gui = WORKFLOW.replace("xvfb-run -a pytest -m gui -v", "echo no gui")
    assert quick.tests_4a(_facts(without_gui)).status == FAIL
    push_only = WORKFLOW.replace("  pull_request:\n", "")
    assert quick.tests_4a(_facts(push_only)).status == FAIL


def test_tests_6b_needs_a_push_or_a_schedule():
    assert quick.tests_6b(_facts()).status == PASS
    pull_requests_only = WORKFLOW.split("  push:")[0] + "jobs:" + WORKFLOW.split("jobs:")[1]
    assert quick.tests_6b(_facts(pull_requests_only)).status == FAIL


def test_tests_8a_needs_windows_and_macos():
    result = quick.tests_8a(_facts())
    assert result.status == FAIL and "no macOS runner" in result.evidence
    with_mac = WORKFLOW.replace("runs-on: windows-latest", "runs-on: ${{ matrix.os }}\n    strategy:\n      matrix:\n        os: [windows-latest, macos-latest]")
    assert quick.tests_8a(_facts(with_mac)).status == PASS


def test_packaging_jobs_and_the_editable_install():
    assert quick.package_4a(_facts()).status == PASS
    no_floors = WORKFLOW.replace("-c constraints-floor.txt ", "")
    result = quick.package_4a(_facts(no_floors))
    assert result.status == FAIL and "dependency floors: none" in result.evidence
    assert quick.package_2a(_facts()).status == PASS
    assert quick.package_2a(_facts(WORKFLOW.replace("pip install -e", "pip install"))).status == FAIL


def test_code_6a_needs_a_clean_linter_and_a_formatter_in_ci():
    pyproject = {"tool": {"ruff": {"lint": {"select": ["F", "E9"]}}}}
    result = quick.code_6a(_facts(), pyproject, ruff_clean=True)
    assert result.status == FAIL and "formatter: none in CI" in result.evidence
    formatted = WORKFLOW.replace("- run: ruff check .", "- run: ruff check .\n      - run: ruff format --check .")
    assert quick.ci_facts({"pytest-pr.yml": formatted}).commands["pytest-pr.yml:lint"] == ["ruff check .", "ruff format --check ."]
    assert quick.code_6a(_facts(formatted), pyproject, ruff_clean=True).status == PASS
    assert quick.code_6a(_facts(formatted), pyproject, ruff_clean=False).status == FAIL


def test_docs_4a_needs_the_tutorial_test_and_a_whole_suite_job():
    assert quick.docs_4a(_facts(), has_tutorial_test=True).status == PASS
    assert quick.docs_4a(_facts(), has_tutorial_test=False).status == FAIL


def test_test_counts_per_marker():
    collected = [
        {"id": "a", "markers": ["slow", "integration"]},
        {"id": "b", "markers": ["gui"]},
        {"id": "c", "markers": ["parametrize"]},
    ]
    assert quick.count_tests(collected) == {"collected": 3, "slow": 1, "integration": 1, "gui": 1, "plotting": 0}
    assert quick.tests_2a(collected, []).status == PASS
    assert quick.tests_2a(collected, ["tests/test_broken.py"]).status == FAIL
    assert quick.tests_2a(None, []) is None


def test_coverage_from_a_cobertura_report():
    xml = '<?xml version="1.0" ?><coverage line-rate="0.927" branch-rate="0" version="7"></coverage>'
    result = quick.tests_6a(xml)
    assert result.status == PASS and result.value == 0.927
    assert quick.tests_6a(xml.replace("0.927", "0.65")).status == FAIL
    assert quick.tests_6a(None) is None


def test_root_files():
    project = ["pyproject.toml", "README.md", "LICENSE", ".gitignore", "CLAUDE.md", "RELEASE_REVIEW.md"]
    assert quick.code_6c(project).status == PASS
    cluttered = quick.code_6c(project + ["hang_debug.log", "mason_claude.md", ".coverage"])
    assert cluttered.status == FAIL and cluttered.value == [".coverage", "hang_debug.log", "mason_claude.md"]
    assert quick.package_6c(project).status == FAIL
    assert quick.package_6c(project + ["CITATION.cff", "CONTRIBUTING.md"]).status == PASS


def test_the_changelog():
    assert quick.docs_6c(["README.md"], None, "0.1.0").status == FAIL
    assert quick.docs_6c(["CHANGELOG.md"], "## 0.2.0\n- new", "0.2.0").status == PASS
    assert quick.docs_6c(["CHANGELOG.md"], "## 0.1.0\n- first", "0.2.0").status == FAIL
    assert quick.package_6a(["README.md"], "0.1.0", 457).status == FAIL
    assert quick.package_6a(["CHANGELOG.md"], "0.2.0", 3) is None  # a reviewer's call


def test_settings_help_and_the_help_pages_counted_never_passed():
    probe = {"settings": 3, "settings_without_help": ["x"], "help_pages": 10, "help_pages_written": 0}
    assert quick.docs_4b(probe).status == FAIL
    assert quick.docs_4b({**probe, "settings_without_help": []}).status == PASS
    pages = quick.docs_6a(probe)
    assert pages.status == FAIL and "0 of 10" in pages.evidence
    assert quick.docs_6a({**probe, "help_pages_written": 10}) is None  # Harvey's record decides
    assert quick.docs_4b(None) is None


SOURCES = {
    "haemolynx/__init__.py": "",
    "haemolynx/graph/build.py": (
        "class A:\n"
        "    def method(self, x: int) -> int:\n        return x\n"
        "    def __init__(self, y):\n        pass\n"
        "def free(a, b: str, *args, **kwargs) -> None:\n    pass\n"
    ),
    "haemolynx/pipeline/schema.py": "x = 1\n" * 3001,
    "haemolynx/gui/_widget.py": "y = 2\n" * 3002,
}


def test_type_hints_are_counted_per_return_and_parameter():
    counts = quick.type_hint_counts(SOURCES)
    assert counts.functions == 3
    assert counts.returns_annotated == 3  # __init__'s return counts as annotated
    assert (counts.parameters, counts.parameters_annotated) == (6, 2)  # x, y, a, b, *args, **kwargs
    result = quick.code_4b(SOURCES)
    assert result.status == FAIL and result.value == {"returns": 1.0, "parameters": 0.333}


def test_long_modules_need_a_stated_reason():
    result = quick.code_6b(SOURCES)
    assert result.status == FAIL
    assert result.value == {"haemolynx/gui/_widget.py": 3002}
    assert "declarations" in result.evidence
    del SOURCES["haemolynx/gui/_widget.py"]
    try:
        assert quick.code_6b(SOURCES).status == PASS
    finally:
        SOURCES["haemolynx/gui/_widget.py"] = "y = 2\n" * 3002


def test_the_package_layout():
    result = quick.code_2a(SOURCES, has_build_system=True)
    assert result.status == PASS and result.value["subpackages"] == ["graph", "gui", "pipeline"]
    assert quick.code_2a(SOURCES, has_build_system=False).status == FAIL


def test_collect_runs_every_collector_the_facts_allow():
    facts = {
        "workflows": {"pytest-pr.yml": WORKFLOW}, "root_files": ["pyproject.toml", "LICENSE"],
        "sources": SOURCES, "pyproject": {"build-system": {}, "project": {"version": "0.1.0"}},
        "probe": None, "collected": None,
    }
    found = quick.collect(facts)
    assert {"tests.4a", "tests.6b", "tests.8a", "package.2a", "package.4a", "package.6a", "package.6c",
            "docs.4a", "docs.6c", "code.2a", "code.4b", "code.6a", "code.6b", "code.6c"} == set(found)
