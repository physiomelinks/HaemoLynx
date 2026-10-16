"""Per-module line coverage of the in-scope CB source, saved to JSON and compared run to run.

Proves a test cleanup lost no coverage: run it before the edit and after, with the same
pytest arguments, then compare.

    venv/bin/python scripts/coverage_by_module.py run BEFORE.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py run AFTER.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py compare BEFORE.json AFTER.json
    venv/bin/python scripts/coverage_by_module.py table BEFORE.json AFTER.json [--modules PATH ...]
    venv/bin/python scripts/coverage_by_module.py sole RUN.json TEST_ID

``run`` runs pytest under pytest-cov with ``--cov-context=test`` and writes, per in-scope
module, its statement count and covered lines, and per test the in-scope lines it ran. The
coverage data file goes to a temporary folder, never the repo. With no PYTEST_ARGS it runs
pytest's default (the whole suite). Exit code: pytest's.

``compare`` prints the passed and skipped counts of both runs, the test ids only in BEFORE and
only in AFTER, then every module whose covered count changed. Exit 0 when no module's covered
count dropped, 1 when one did, 2 on a usage error, 3 when the two runs cannot be compared
(different pytest arguments, or a run whose tests failed).

``table`` prints a Markdown table (module, statements, covered before, covered after, change)
to paste under a ticket's ``## Answer``, led by a passed/skipped count line for each run. With ``--modules`` it prints those modules, changed or
not; without, only modules whose covered count changed. A ``--modules`` path in neither run is
a usage error (exit 2, nothing printed), as is a module listed in only one run.

``sole`` lists the lines TEST_ID ran that no other test in the run ran (its sole-guard lines).

Lines run at import time (``def``, module constants) count as covered but belong to no test.
Subprocesses are not traced, so a driver run as a subprocess shows no coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Nerve-only or archived source, left out of "in-scope" (test-suite-pruning map, Scope).
EXCLUDED_PACKAGE = {
    "src/ImageLynx/haemodynamics/pericyte_mask.py",
    "src/ImageLynx/haemodynamics/pericyte_comparison.py",
    "src/ImageLynx/haemodynamics/probability.py",
    "src/ImageLynx/statistics/auto_tuner.py",
}
INCLUDED_EXAMPLES = {
    "examples/carotid_image_to_model.py",
    "examples/preprocessing/preprocess_cb.py",
    "examples/preprocessing/preprocess_th.py",
    "examples/preprocessing/prob_to_mask.py",
}


def in_scope(rel: str) -> bool:
    """True for a repo-relative path in the CB source the suite must keep covered."""
    if rel.startswith("src/ImageLynx/"):
        return rel.endswith(".py") and rel not in EXCLUDED_PACKAGE
    if rel.startswith("examples/"):
        return rel in INCLUDED_EXAMPLES or (rel.startswith("examples/cb_") and "/" not in rel[9:])
    return False


def in_scope_files() -> list[Path]:
    paths = sorted((REPO / "src/ImageLynx").rglob("*.py")) + sorted((REPO / "examples").rglob("*.py"))
    return [p for p in paths if in_scope(str(p.relative_to(REPO)))]


def read_data(data_file: Path) -> dict:
    """Modules and per-test lines from a coverage data file written with --cov-context=test."""
    import coverage

    cov = coverage.Coverage(data_file=str(data_file))
    cov.load()  # get_data() on an unloaded Coverage returns an empty data set
    data = cov.get_data()
    measured = {str(Path(f).resolve()): f for f in data.measured_files()}
    modules, tests = {}, defaultdict(lambda: defaultdict(list))
    for path in in_scope_files():
        rel = str(path.relative_to(REPO))
        _, statements, _, missing, _ = cov.analysis2(str(path))
        statements, missing = set(statements), set(missing)
        modules[rel] = {"statements": len(statements), "covered": sorted(statements - missing)}
        f = measured.get(str(path.resolve()))
        if f is None:
            continue  # never imported: 0 %
        for line, ctxs in (data.contexts_by_lineno(f) or {}).items():
            if line not in statements:
                continue
            for ctx in ctxs:
                if ctx:  # "" is import time, inside no test
                    tests[ctx.rsplit("|", 1)[0]][rel].append(line)
    tests = {t: {rel: sorted(set(ls)) for rel, ls in by.items()} for t, by in sorted(tests.items())}
    return {"modules": modules, "tests": tests}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def run(out: Path, pytest_args: list[str]) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / ".coverage"
        rc = Path(tmp) / "coveragerc"
        rc.write_text(f"[run]\ndata_file = {data_file}\n")
        outcomes_file = Path(tmp) / "outcomes.json"
        cmd = [sys.executable, "-m", "pytest", "-p", "pytest_cov", "-p", "_cbm_outcomes",
               "--cov=src/ImageLynx", "--cov=examples", "--cov-context=test",
               f"--cov-config={rc}", "--cov-report=", *pytest_args]
        env = {**os.environ, "COVERAGE_FILE": str(data_file), "CBM_OUTCOMES_FILE": str(outcomes_file),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))}
        code = subprocess.run(cmd, cwd=REPO, env=env).returncode
        if not data_file.exists():
            print(f"pytest wrote no coverage data (exit {code})", file=sys.stderr)
            return code or 1
        result = read_data(data_file)
        if not outcomes_file.exists():
            print(f"pytest wrote no test outcomes (exit {code})", file=sys.stderr)
            return code or 1
        outcomes = json.loads(outcomes_file.read_text())
    result = {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "pytest_args": pytest_args,
        "pytest_exit": code,
        "outcomes": outcomes,
        **result,
    }
    out.write_text(json.dumps(result, indent=1))
    total_s = sum(m["statements"] for m in result["modules"].values())
    total_c = sum(len(m["covered"]) for m in result["modules"].values())
    print(f"{out}: {len(result['tests'])} tests, {total_c}/{total_s} in-scope lines covered")
    return code


def _ids(result: dict, outcome: str) -> set[str]:
    return {t for t, o in result["outcomes"].items() if o == outcome}


def count_line(name: str, result: dict) -> str:
    """'before: 1471 passed, 12 skipped', or says the run saved no outcomes (an older run)."""
    if "outcomes" not in result:
        return f"{name}: test counts not recorded (run saved before outcomes were recorded)"
    return f"{name}: {len(_ids(result, 'passed'))} passed, {len(_ids(result, 'skipped'))} skipped"


def outcome_lines(before: dict, after: dict) -> list[str]:
    """Count lines for both runs, then the test ids only in BEFORE and only in AFTER."""
    lines = [count_line("before", before), count_line("after", after)]
    if "outcomes" in before and "outcomes" in after:
        for label, only in (("only in BEFORE", set(before["outcomes"]) - set(after["outcomes"])),
                            ("only in AFTER", set(after["outcomes"]) - set(before["outcomes"]))):
            lines.append(f"{label} ({len(only)}):")
            lines.extend(f"  {t}" for t in sorted(only))
    return lines


def compare(before: dict, after: dict) -> tuple[int, list[str]]:
    """(exit code, report lines) for two run results."""
    lines = [f"before: {before['commit'][:7]}{' (dirty)' if before['dirty'] else ''}",
             f"after:  {after['commit'][:7]}{' (dirty)' if after['dirty'] else ''}"]
    if before["pytest_args"] != after["pytest_args"]:
        return 3, lines + [f"cannot compare: pytest args differ: {before['pytest_args']} vs {after['pytest_args']}"]
    for name, r in (("before", before), ("after", after)):
        if r["pytest_exit"] != 0:
            return 3, lines + [f"cannot compare: the {name} run's pytest exited {r['pytest_exit']}"]
    lines += outcome_lines(before, after)
    dropped = False
    for rel in sorted(set(before["modules"]) | set(after["modules"])):
        b = before["modules"].get(rel, {"covered": []})["covered"]
        a = after["modules"].get(rel, {"covered": []})["covered"]
        if len(a) == len(b):
            continue
        if len(a) < len(b):
            dropped = True
            lost = sorted(set(b) - set(a))
            hint = f"; lines no longer run: {_ranges(lost)}" if lost else ""
            lines.append(f"DROP {rel}: {len(b)} -> {len(a)}{hint}")
        else:
            lines.append(f"gain {rel}: {len(b)} -> {len(a)}")
    lines.append("covered count dropped" if dropped else "no module lost covered lines")
    return int(dropped), lines


def table(before: dict, after: dict, modules: list[str] | None = None) -> list[str]:
    """Markdown table lines: the named modules, or those whose covered count changed.

    Raises KeyError for a named module that is in neither run, and for any module in only one
    run (its other count would be invented).
    """
    unknown = [m for m in modules or [] if m not in before["modules"] and m not in after["modules"]]
    if unknown:
        raise KeyError(f"not in either run: {', '.join(unknown)}")
    lopsided = sorted(set(before["modules"]) ^ set(after["modules"]))
    if lopsided:
        raise KeyError(f"in only one run: {', '.join(lopsided)}")
    lines = [count_line("before", before) + "  ", count_line("after", after), "",
             "| Module | Statements | Covered before | Covered after | Change |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for rel in modules or sorted(set(before["modules"]) | set(after["modules"])):
        b, a = before["modules"][rel], after["modules"][rel]
        change = len(a["covered"]) - len(b["covered"])
        if change or modules:
            shown = f"{change:+d}" if change else "0"
            lines.append(f"| {rel} | {a['statements']} | {len(b['covered'])} "
                         f"| {len(a['covered'])} | {shown} |")
    return lines


def sole(result: dict, test: str) -> dict[str, list[int]]:
    """Per module, the lines `test` ran that no other test in the run ran."""
    if test not in result["tests"]:
        raise KeyError(f"{test} ran no in-scope line in this run (or is not in it)")
    others = defaultdict(set)
    for t, by in result["tests"].items():
        if t != test:
            for rel, ls in by.items():
                others[rel].update(ls)
    out = {}
    for rel, ls in result["tests"][test].items():
        only = sorted(set(ls) - others[rel])
        if only:
            out[rel] = only
    return out


def _ranges(nums: list[int]) -> str:
    parts, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            parts.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run pytest under coverage and save per-module lines")
    p.add_argument("out", type=Path)
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("compare", help="exit 1 if any module's covered count dropped")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p = sub.add_parser("table", help="Markdown per-module covered counts, before and after")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--modules", nargs="+", metavar="PATH", help="only these modules, changed or not")
    p = sub.add_parser("sole", help="lines only TEST_ID ran in this run")
    p.add_argument("result", type=Path)
    p.add_argument("test")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run(args.out, args.pytest_args)
    if args.cmd == "compare":
        code, report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        print("\n".join(report))
        return code
    if args.cmd == "table":
        try:
            report = table(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.modules)
        except KeyError as e:
            print(e.args[0], file=sys.stderr)
            return 2
        print("\n".join(report))
        return 0
    try:
        found = sole(json.loads(args.result.read_text()), args.test)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    for rel, ls in found.items():
        print(f"{rel}: {_ranges(ls)}")
    print(f"{sum(map(len, found.values()))} sole-guard lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
