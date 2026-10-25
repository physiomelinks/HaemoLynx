"""The quick tier: checks decided from static facts, in a minute or two.

Each collector here is pure: it takes facts the runner gathered (the CI
workflows' text, the tests collected with their markers, the package's
sources, the files at the root, a probe of the settings and help pages) and
returns a :class:`~.ladder.CheckResult`, or None when the facts cannot decide
the check, so it falls back to a recorded mark or its row.
"""
from __future__ import annotations

import ast
import re
import shlex
import xml.etree.ElementTree as ElementTree
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .ladder import CheckResult, measured
from .rubric import target

#: Markers counted per test (``[tool.pytest.ini_options]`` in ``pyproject.toml``).
MARKERS = ("slow", "integration", "gui", "plotting")

#: Modules over the size limit with a reason stated in ``RELEASE_REVIEW.md``
#: (``code.6b``'s row): a module this long is otherwise a module to split.
MODULE_SIZE_REASONS = {
    "haemolynx/pipeline/schema.py": "declarations: one Setting per option",
}

#: Files that belong at a project's root. Anything else tracked there is
#: personal or debugging clutter (``code.6c``).
ROOT_PROJECT_FILES = re.compile(
    r"^(\.gitignore|\.gitattributes|\.pre-commit-config\.yaml|\.readthedocs\.ya?ml|"
    r"LICEN[CS]E(\.\w+)?|README(\.\w+)?|CHANGELOG(\.\w+)?|CITATION\.cff|CONTRIBUTING(\.\w+)?|"
    r"CODE_OF_CONDUCT(\.\w+)?|SECURITY(\.\w+)?|pyproject\.toml|setup\.(py|cfg)|MANIFEST\.in|"
    r"tox\.ini|noxfile\.py|conftest\.py|mkdocs\.ya?ml|environment\.ya?ml|constraints-floor\.txt|"
    r"CLAUDE\.md|RELEASE_REVIEW\.md)$"
)


# ---------------------------------------------------------------------------
# Tests collected
# ---------------------------------------------------------------------------


def count_tests(collected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many tests were collected, in all and per marker."""
    counts = Counter()
    for item in collected:
        for marker in set(item.get("markers", ())) & set(MARKERS):
            counts[marker] += 1
    return {"collected": len(collected), **{marker: counts[marker] for marker in MARKERS}}


def tests_2a(collected: Optional[Sequence[Mapping[str, Any]]], errors: Sequence[str]) -> Optional[CheckResult]:
    if collected is None:
        return None
    counts = count_tests(collected)
    evidence = ", ".join(f"{value:,} {name}" for name, value in counts.items())
    if errors:
        evidence += f"; {len(errors)} collection error(s): {', '.join(errors[:5])}"
    return measured("tests.2a", counts["collected"] > 0 and not errors, counts, evidence)


# ---------------------------------------------------------------------------
# CI workflows
# ---------------------------------------------------------------------------


@dataclass
class PytestRun:
    job: str
    marker_expression: Optional[str]
    names_files: bool
    command: str


@dataclass
class CIFacts:
    triggers: dict[str, Any] = field(default_factory=dict)
    runners: dict[str, list[str]] = field(default_factory=dict)
    python_versions: dict[str, list[str]] = field(default_factory=dict)
    commands: dict[str, list[str]] = field(default_factory=dict)
    pytest_runs: list[PytestRun] = field(default_factory=list)

    def any_command(self, pattern: str) -> list[str]:
        """The jobs with a step whose command matches *pattern*."""
        regex = re.compile(pattern)
        return sorted(job for job, lines in self.commands.items() if any(regex.search(line) for line in lines))


def _pytest_runs(job: str, command: str) -> list[PytestRun]:
    runs = []
    for line in command.splitlines():
        if not re.search(r"(^|\s)pytest(\s|$)", line):
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            words = line.split()
        start = words.index("pytest") + 1 if "pytest" in words else len(words)
        arguments = words[start:]
        expression = None
        for index, word in enumerate(arguments):
            if word == "-m" and index + 1 < len(arguments):
                expression = arguments[index + 1]
        names_files = any(word.endswith(".py") or "::" in word for word in arguments)
        runs.append(PytestRun(job, expression, names_files, line.strip()))
    return runs


def ci_facts(workflows: Mapping[str, str]) -> CIFacts:
    """What the workflows' YAML says: triggers, runners, Python versions,
    each job's commands and each pytest call's marker expression."""
    import yaml

    facts = CIFacts()
    for name, text in sorted(workflows.items()):
        data = yaml.safe_load(text) or {}
        # PyYAML reads the bare key `on` as the boolean True.
        triggers = data.get("on", data.get(True)) or {}
        if isinstance(triggers, str):
            triggers = {triggers: None}
        elif isinstance(triggers, list):
            triggers = {each: None for each in triggers}
        facts.triggers[name] = triggers
        for job_name, job in (data.get("jobs") or {}).items():
            key = f"{name}:{job_name}"
            matrix = ((job.get("strategy") or {}).get("matrix") or {})
            runs_on = job.get("runs-on", "")
            runners = matrix.get("os") if "matrix.os" in str(runs_on) else runs_on
            facts.runners[key] = [str(each) for each in (runners if isinstance(runners, list) else [runners])]
            versions = list(matrix.get("python-version") or [])
            for step in job.get("steps") or ():
                version = (step.get("with") or {}).get("python-version")
                if version and "matrix" not in str(version):
                    versions.append(version)
            facts.python_versions[key] = [str(each) for each in versions]
            lines = [str(step["run"]) for step in job.get("steps") or () if step.get("run")]
            facts.commands[key] = lines
            for command in lines:
                facts.pytest_runs.extend(_pytest_runs(key, command))
    return facts


def _on_pull_requests(facts: CIFacts, job: str) -> bool:
    return "pull_request" in facts.triggers.get(job.split(":", 1)[0], {})


def _suite_runs(facts: CIFacts) -> list[PytestRun]:
    return [run for run in facts.pytest_runs if not run.names_files]


def tests_4a(facts: CIFacts) -> CheckResult:
    """Unit, integration and GUI tests, each run by a job on every PR."""
    runs = [run for run in _suite_runs(facts) if _on_pull_requests(facts, run.job)]
    whole = sorted({run.job for run in runs if run.marker_expression is None})
    gui = sorted({
        run.job for run in runs
        if run.marker_expression is not None and "gui" in run.marker_expression and "not gui" not in run.marker_expression
    })
    evidence = f"on pull requests, the whole suite in {', '.join(whole) or 'no job'}; `-m gui` in {', '.join(gui) or 'no job'}"
    return measured("tests.4a", bool(whole) and bool(gui), {"whole_suite": whole, "gui": gui}, evidence)


def tests_6b(facts: CIFacts) -> CheckResult:
    """Pushes to the shared branches, or a schedule, not only pull requests."""
    pushes = {
        name: (triggers.get("push") or {}).get("branches", [])
        for name, triggers in facts.triggers.items() if "push" in triggers
    }
    schedules = {name: triggers["schedule"] for name, triggers in facts.triggers.items() if triggers.get("schedule")}
    suite_files = {run.job.split(":", 1)[0] for run in _suite_runs(facts)}
    on_push = {name: branches for name, branches in pushes.items() if name in suite_files}
    scheduled = {name: value for name, value in schedules.items() if name in suite_files}
    evidence = "; ".join(
        [f"{name} on push to {', '.join(branches) or 'every branch'}" for name, branches in on_push.items()]
        + [f"{name} on schedule {', '.join(each.get('cron', '?') for each in value)}" for name, value in scheduled.items()]
    ) or "the suite runs on pull requests only"
    return measured("tests.6b", bool(on_push or scheduled), {"push": on_push, "schedule": list(scheduled)}, evidence)


def tests_8a(facts: CIFacts) -> CheckResult:
    """Windows and macOS runners running the suite."""
    runners = sorted({runner for run in _suite_runs(facts) for runner in facts.runners.get(run.job, [])})
    windows = any(runner.startswith("windows") for runner in runners)
    macos = any(runner.startswith("macos") for runner in runners)
    evidence = f"the suite runs on {', '.join(runners)}" + ("" if macos else "; no macOS runner")
    return measured("tests.8a", windows and macos, runners, evidence)


def coverage_fraction(coverage_xml: str) -> float:
    """The line rate a ``coverage.xml`` (Cobertura) reports."""
    root = ElementTree.fromstring(coverage_xml)
    return float(root.attrib["line-rate"])


def tests_6a(coverage_xml: Optional[str], source: str = "coverage.xml") -> Optional[CheckResult]:
    if coverage_xml is None:
        return None
    fraction = coverage_fraction(coverage_xml)
    goal = target("tests.6a")
    return measured(
        "tests.6a", fraction >= goal, round(fraction, 4),
        f"{100 * fraction:.1f}% line coverage of non-GUI code ({source})", target=goal,
    )


def package_2a(facts: CIFacts) -> CheckResult:
    jobs = facts.any_command(r"pip install -e\b")
    evidence = f"every CI run installs it editable (`pip install -e`) in {', '.join(jobs) or 'no job'}"
    return measured("package.2a", bool(jobs), jobs, evidence)


def package_4a(facts: CIFacts) -> CheckResult:
    """The wheel built and run outside the repository, the dependency floors,
    and the napari manifest, each checked by a CI job."""
    parts = {
        "wheel": facts.any_command(r"python -m build\b.*--wheel"),
        "outside the repository": facts.any_command(r"installed_package_smoke"),
        "dependency floors": facts.any_command(r"constraints-floor"),
        "napari manifest": facts.any_command(r"npe2 validate"),
    }
    evidence = "; ".join(f"{name}: {', '.join(jobs) or 'none'}" for name, jobs in parts.items())
    return measured("package.4a", all(parts.values()), parts, evidence)


def code_6a(facts: CIFacts, pyproject: Mapping[str, Any], ruff_clean: Optional[bool]) -> CheckResult:
    """A linter and a formatter, configured and clean in CI."""
    lint_rules = (((pyproject.get("tool") or {}).get("ruff") or {}).get("lint") or {}).get("select")
    lint_jobs = facts.any_command(r"ruff check|flake8|pylint")
    format_jobs = facts.any_command(r"ruff format --check|black (--check|--diff)")
    evidence = (
        f"lint: rules {lint_rules or 'none'} in pyproject.toml, run by {', '.join(lint_jobs) or 'no job'}, "
        f"{'clean' if ruff_clean else 'not clean' if ruff_clean is False else 'not run'} here; "
        f"formatter: {', '.join(format_jobs) or 'none in CI'}"
    )
    passed = bool(lint_rules) and bool(lint_jobs) and bool(ruff_clean) and bool(format_jobs)
    return measured("code.6a", passed, {"lint": lint_jobs, "format": format_jobs, "clean": ruff_clean}, evidence)


def docs_4a(facts: CIFacts, has_tutorial_test: bool) -> CheckResult:
    whole = sorted({run.job for run in _suite_runs(facts) if run.marker_expression is None})
    evidence = (
        f"`tests/integration/test_pipeline_tutorial.py` {'present' if has_tutorial_test else 'missing'}; "
        f"the whole suite runs in {', '.join(whole) or 'no job'}"
    )
    return measured("docs.4a", has_tutorial_test and bool(whole), whole, evidence)


# ---------------------------------------------------------------------------
# Packaging and docs
# ---------------------------------------------------------------------------


def _root_file(root_files: Iterable[str], pattern: str) -> Optional[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return next((name for name in sorted(root_files) if regex.match(name)), None)


def package_6c(root_files: Sequence[str]) -> CheckResult:
    found = {
        "CITATION.cff": _root_file(root_files, r"^CITATION\.cff$"),
        "CONTRIBUTING": _root_file(root_files, r"^CONTRIBUTING(\.\w+)?$"),
        "licence": _root_file(root_files, r"^LICEN[CS]E(\.\w+)?$"),
    }
    missing = [name for name, path in found.items() if path is None]
    evidence = "present: " + (", ".join(path for path in found.values() if path) or "none")
    if missing:
        evidence += f"; missing: {', '.join(missing)}"
    return measured("package.6c", not missing, found, evidence)


def _changelog(root_files: Sequence[str]) -> Optional[str]:
    return _root_file(root_files, r"^CHANGELOG(\.\w+)?$")


def docs_6c(root_files: Sequence[str], changelog_text: Optional[str], version: str) -> CheckResult:
    name = _changelog(root_files)
    if name is None:
        return measured("docs.6c", False, None, "no CHANGELOG at the root")
    names_version = changelog_text is not None and version in changelog_text
    return measured(
        "docs.6c", names_version, version,
        f"`{name}` {'has' if names_version else 'has no'} entry for {version}",
    )


def package_6a(root_files: Sequence[str], version: str, commits_since_tag: Optional[int]) -> Optional[CheckResult]:
    """Fails without a CHANGELOG; with one, whether the version moves with
    the changes is a reviewer's call, so it is left to the record."""
    since = "" if commits_since_tag is None else f", {commits_since_tag:,} commits after its tag"
    if _changelog(root_files) is None:
        return measured("package.6a", False, {"version": version, "commits_since_tag": commits_since_tag},
                        f"version {version}{since}; no CHANGELOG")
    return None


def docs_4b(probe: Optional[Mapping[str, Any]]) -> Optional[CheckResult]:
    if not probe or "settings" not in probe:
        return None
    empty = list(probe.get("settings_without_help", []))
    evidence = f"{len(empty)} of {probe['settings']} settings with empty help"
    if empty:
        evidence += f": {', '.join(empty[:10])}"
    return measured("docs.4b", not empty, {"settings": probe["settings"], "empty": empty}, evidence)


def docs_6a(probe: Optional[Mapping[str, Any]]) -> Optional[CheckResult]:
    """Counted only, never written: an H check, Harvey's to pass. Fails while
    any page is empty; once every page has text it is his record that passes it."""
    if not probe or "help_pages" not in probe:
        return None
    pages, written = probe["help_pages"], probe["help_pages_written"]
    if written < pages:
        return measured("docs.6a", False, {"pages": pages, "written": written},
                        f"{written} of {pages} help pages have text (`gui/stage_help.py`)")
    return None


# ---------------------------------------------------------------------------
# Code metrics
# ---------------------------------------------------------------------------


@dataclass
class HintCounts:
    functions: int = 0
    returns_annotated: int = 0
    parameters: int = 0
    parameters_annotated: int = 0


def type_hint_counts(sources: Mapping[str, str]) -> HintCounts:
    """Annotated returns and parameters over every function in *sources*;
    ``self`` and ``cls`` are not counted, nor are ``__init__``'s returns."""
    counts = HintCounts()
    for text in sources.values():
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            counts.functions += 1
            if node.name != "__init__":
                counts.returns_annotated += node.returns is not None
            else:
                counts.returns_annotated += 1
            arguments = node.args
            every = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            every += [each for each in (arguments.vararg, arguments.kwarg) if each is not None]
            for index, argument in enumerate(every):
                if index == 0 and argument.arg in ("self", "cls"):
                    continue
                counts.parameters += 1
                counts.parameters_annotated += argument.annotation is not None
    return counts


def code_4b(sources: Mapping[str, str]) -> CheckResult:
    counts = type_hint_counts(sources)
    returns = counts.returns_annotated / max(counts.functions, 1)
    parameters = counts.parameters_annotated / max(counts.parameters, 1)
    goal = target("code.4b")
    evidence = (
        f"{100 * returns:.0f}% of {counts.functions:,} functions' returns and "
        f"{100 * parameters:.0f}% of {counts.parameters:,} parameters annotated"
    )
    return measured("code.4b", returns >= goal and parameters >= goal,
                    {"returns": round(returns, 3), "parameters": round(parameters, 3)}, evidence, target=goal)


def line_counts(sources: Mapping[str, str]) -> dict[str, int]:
    return {path: len(text.splitlines()) for path, text in sources.items()}


def code_2a(sources: Mapping[str, str], has_build_system: bool) -> CheckResult:
    counts = line_counts(sources)
    subpackages = sorted({path.split("/")[1] for path in counts if path.count("/") >= 2})
    evidence = (
        f"{len(counts)} modules, {sum(counts.values()):,} lines, in {len(subpackages)} subpackages "
        f"({', '.join(subpackages)}); {'a' if has_build_system else 'no'} build system in pyproject.toml"
    )
    return measured("code.2a", has_build_system and len(subpackages) >= 2,
                    {"modules": len(counts), "lines": sum(counts.values()), "subpackages": subpackages}, evidence)


def code_6b(sources: Mapping[str, str]) -> CheckResult:
    limit = int(target("code.6b"))
    long = {path: lines for path, lines in line_counts(sources).items() if lines > limit}
    unexplained = {path: lines for path, lines in long.items() if path not in MODULE_SIZE_REASONS}
    listed = ", ".join(f"`{path.split('/', 1)[1]}` {lines:,}" for path, lines in sorted(long.items(), key=lambda kv: -kv[1]))
    evidence = f"over {limit:,} lines: {listed or 'none'}"
    if long.keys() - unexplained.keys():
        evidence += "; with a stated reason: " + ", ".join(
            f"`{path.split('/', 1)[1]}` ({MODULE_SIZE_REASONS[path]})" for path in sorted(long.keys() - unexplained.keys())
        )
    return measured("code.6b", not unexplained, unexplained, evidence, target=limit)


def code_6c(root_files: Sequence[str]) -> CheckResult:
    clutter = sorted(name for name in root_files if not ROOT_PROJECT_FILES.match(name))
    evidence = f"tracked at the root besides project files: {', '.join(clutter) or 'nothing'}"
    return measured("code.6c", not clutter, clutter, evidence)


def collect(facts: Mapping[str, Any]) -> dict[str, CheckResult]:
    """Every quick-tier result the facts can decide."""
    ci = ci_facts(facts["workflows"])
    root_files = facts["root_files"]
    sources = facts["sources"]
    pyproject = facts["pyproject"]
    probe = facts.get("probe")
    version = (probe or {}).get("version") or (pyproject.get("project") or {}).get("version", "?")
    results = [
        tests_2a(facts.get("collected"), facts.get("collection_errors", ())),
        tests_4a(ci),
        tests_6b(ci),
        tests_8a(ci),
        tests_6a(facts.get("coverage_xml")),
        package_2a(ci),
        package_4a(ci),
        package_6a(root_files, version, facts.get("commits_since_tag")),
        package_6c(root_files),
        docs_4a(ci, facts.get("has_tutorial_test", False)),
        docs_4b(probe),
        docs_6a(probe),
        docs_6c(root_files, facts.get("changelog_text"), version),
        code_2a(sources, "build-system" in pyproject),
        code_4b(sources),
        code_6a(ci, pyproject, facts.get("ruff_clean")),
        code_6b(sources),
        code_6c(root_files),
    ]
    return {result.id: result for result in results if result is not None}
