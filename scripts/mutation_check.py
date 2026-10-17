"""Mutation checks: break a source line on purpose and show the test meant to guard it fails.
It never edits the main checkout: every mutation is made in a throwaway git worktree.

    venv/bin/python scripts/mutation_check.py SPEC.toml [--only NAME ...] [--timeout SECONDS]

SPEC.toml holds one ``[[mutation]]`` table per check (a ticket keeps its spec next to it, as
``.scratch/<feature>/issues/NN-mutations.toml``):

    [[mutation]]
    name = "crop_roi: offset ignored"               # short label, unique in the file
    file = "src/ImageLynx/preprocessing/image.py"   # repo-relative .py file, not under tests/
    old = "extent / 2.0 + extent * offsets[i] - 0.5"  # must appear exactly once in the file
    new = "extent / 2.0 + 0 * offsets[i] - 0.5"
    test = "tests/test_efficiency.py::test_crop_roi"  # one pytest node id
    expect = "survived"                             # optional; default "caught"

Use ``'''...'''`` for text over several lines (TOML drops a newline straight after the opening
quotes). The spec is checked in full before anything runs, and anything it does not pin down
exactly is refused: an unknown key, a duplicate name, empty ``old``, ``new`` equal to ``old``.
Every error in the spec is listed in one run. An ``old`` that is not in its file exactly once
gets the count, the lines it starts on, and any whitespace at its ends (the usual cause).

The main checkout is never edited, not even for a moment. The checks run in a temporary git
worktree outside the repo, made from HEAD plus every uncommitted change in this checkout
(tracked edits, deletions, and untracked files that are not ignored, all named in the output's
header, so a stray note copied in is plain to see). The gitignored data folders
(``_worktree.DATA_FOLDERS``) are symlinked in, not copied, so ``realdata`` tests
run on the real data; a run that changes anything through those links stops with exit 3. The
worktree is removed afterwards, also after an error or Ctrl-C. ``scripts/_worktree.py`` builds
it, shared with ``coverage_by_module.py run --at``.

Each distinct test first runs alone under coverage on the unmutated worktree, with Python
subprocesses it starts traced too. It must pass. The main checkout's copies of the mutated
files are measured as well: a test that ran one of them imported the wrong tree, and the run
stops. Then each mutation is applied, its test run alone without coverage, and the file put
back. No run writes bytecode, so a same-size edit is never masked by a cached ``.pyc``.
Results: ``caught`` (pytest exit 1), ``survived`` (exit 0), ``skipped`` (the test skipped, which
pytest also reports as exit 0; the first failure cell then holds pytest's ``SKIPPED`` line with
the reason), ``timed out``, ``error`` (any other pytest exit, such as a collection error; not a
catch). ``timed out`` matches neither ``expect``: a hang is a finding to look at, whatever the
spec expected. A skip is no verdict, so ``skipped`` counts as an error, as does a skip in the
clean run (a ``realdata`` test skips where the data is missing from this checkout too).

``Ran line`` says whether the clean run ran a statement on a line the mutation changes
(unchanged context lines in ``old`` don't count); the clean run includes import time, so a
module constant counts.

Output: a header naming the worktree's commit and the files copied in, a Markdown table to
paste under a ticket's ``## Answer``, and a summary line. The table gives the mutation, test,
expected, result, ran line, and the first failure: the test file's ``file:line`` (its last
traceback frame in that file) and the first ``E`` line, where a bare ``AssertionError:`` gets
the first non-empty ``E`` line after it; or pytest's last line when there is no ``E`` line, such
as ``1 passed`` for a survivor. A catch made by an error other than an assert (a ``TypeError``
from the broken code, say, rather than the test's own check) shows as ``caught (TypeError, not
an assert)`` and is counted in the summary; it still counts as ``caught``.

Exit code: 0 when every result matches its ``expect`` and every test ran its mutated line;
1 when one does not; 2 on a usage error (the spec, or --only/--timeout); 3 when the checks
could not run (git failed, a test failed, skipped or timed out before any mutation, a test
imported the main checkout, pytest errored on or skipped a mutated run, or a run changed the
linked data).
"""
from __future__ import annotations

import argparse
import ast
import difflib
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11, as in CI; pytest itself requires tomli there
    import tomli as tomllib

# scripts/ is not a package. No .pyc is written for the helper: the checkout stays as it was.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
import _worktree  # noqa: E402
sys.dont_write_bytecode = _bytecode

REPO = Path(__file__).resolve().parents[1]


class SpecError(Exception):
    """The spec cannot be used as written (exit 2)."""


class CannotRun(Exception):
    """The checks could not be run, so there is no verdict (exit 3)."""


# A mutated run's outcome. A spec may expect one of EXPECTS; one of NO_VERDICT makes the run exit 3.
CAUGHT, SURVIVED, SKIPPED, TIMED_OUT, ERROR = "caught", "survived", "skipped", "timed out", "error"
EXPECTS = (CAUGHT, SURVIVED)
NO_VERDICT = (ERROR, SKIPPED)


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str
    old: str
    new: str
    test: str
    expect: str = CAUGHT


@dataclass(frozen=True)
class Result:
    mutation: Mutation
    outcome: str  # CAUGHT, SURVIVED, SKIPPED, TIMED_OUT or ERROR
    ran: bool
    failure: str
    error: str | None = None  # the exception type that failed the test, when pytest showed one

    @property
    def as_expected(self) -> bool:
        return self.outcome == self.mutation.expect

    @property
    def caught_by_non_assert(self) -> bool:
        """A catch the test's own checks did not make: say, a TypeError from the broken code."""
        return self.outcome == CAUGHT and self.error not in ASSERTS

    @property
    def result_cell(self) -> str:
        if self.caught_by_non_assert:
            return f"caught ({self.error or 'no E line'}, not an assert)"
        return self.outcome


REQUIRED = ("name", "file", "old", "new", "test")
# Errors a test's own checks raise: assert (numpy.testing's too), and pytest.fail or
# pytest.raises' DID NOT RAISE (pytest's Failed). A catch by any other error is marked.
ASSERTS = ("AssertionError", "Failed")


def load_spec(path: Path, repo: Path, only: list[str] | None = None) -> list[Mutation]:
    """The spec's mutations (those named in `only`, if given), in spec order.

    Raises SpecError for anything the spec does not pin down exactly: a missing or unknown key,
    a file outside the repo, under tests/ or not .py, or old text that is not in the file once.
    Every error is found before it raises, one per line, so one run shows them all. (A check that
    needs an earlier one to pass, such as counting old in a file that is not there, is skipped.)
    """
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SpecError(f"cannot read {path}: {error}") from None
    errors = []
    unknown = sorted(set(data) - {"mutation"})
    if unknown:
        errors.append(f"unknown top-level key(s): {', '.join(unknown)}")
    entries = data.get("mutation")
    if not isinstance(entries, list) or not entries:
        raise SpecError("\n".join(errors + ["no [[mutation]] tables"]))
    mutations, names = [], set()
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            errors.append(f"mutation {i} is not a table")
            continue
        problems = _entry_problems(entry, repo, names)
        errors += [f"mutation {i} ({entry.get('name', '?')}): {p}" for p in problems]
        if isinstance(entry.get("name"), str):
            names.add(entry["name"])
        if not problems:
            mutations.append(Mutation(**{**entry, "file": Path(entry["file"]).as_posix()}))
    if only is not None:
        unknown = [n for n in only if n not in names]
        if unknown:
            errors.append(f"--only name(s) not in the spec: {', '.join(unknown)}")
        mutations = [m for m in mutations if m.name in only]
    if errors:
        raise SpecError("\n".join(errors))
    return mutations


def _entry_problems(entry: dict, repo: Path, names: set[str]) -> list[str]:
    """What is wrong with one [[mutation]] table; `names` are the names used before it."""
    missing = [k for k in REQUIRED if k not in entry]
    extra = sorted(set(entry) - set(REQUIRED) - {"expect"})
    problems = [f"missing {', '.join(missing)}"] if missing else []
    if extra:
        problems.append(f"unknown key(s): {', '.join(extra)}")
    problems += [f"{key} must be a string" for key, value in entry.items() if not isinstance(value, str)]
    if problems:
        return problems  # the checks below need every key, as a string
    old, new, expect = entry["old"], entry["new"], entry.get("expect", CAUGHT)
    if expect not in EXPECTS:
        problems.append(f"expect must be {' or '.join(map(repr, EXPECTS))}, not {expect!r}")
    if entry["name"] in names:
        problems.append("name already used")
    if not old:
        problems.append("old is empty")
    if new == old:
        problems.append("new is the same as old")
    rel = Path(entry["file"])  # Path drops ./ and doubled slashes, as coverage's paths do
    file = rel.as_posix()
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        return problems + ["file must be a path inside the repo"]
    before = len(problems)
    if rel.parts[0] == "tests":
        problems.append(f"{file} is under tests/")
    if not (repo / rel).is_file():
        problems.append(f"{file} is not a file in the repo")
    if rel.suffix != ".py":
        problems.append(f"{file} is not a .py file")
    if old and len(problems) == before:
        problems += _old_not_once(old, (repo / rel).read_text(), file)
    return problems


def _old_not_once(old: str, source: str, file: str) -> list[str]:
    """[] when `old` is in `source` once; else how often and where, and any whitespace at its ends.

    Overlapping matches count, since either could be the one meant.
    """
    starts = [i for i in range(len(source)) if source.startswith(old, i)]
    if len(starts) == 1:
        return []
    lines = ", ".join(str(source.count("\n", 0, i) + 1) for i in starts)
    message = f"old appears {len(starts)} times in {file}{f' (lines {lines})' if starts else ''}, not once"
    stripped = old.strip()
    # Trailing whitespace (often a newline a TOML string kept or dropped) was the usual cause.
    ends = [f"{end} with whitespace {text!r}" for end, text in
            (("starts", old[:len(old) - len(old.lstrip())]), ("ends", old[len(old.rstrip()):])) if text]
    if ends and stripped:
        message += (f"; it {' and '.join(ends)}, and without that whitespace it appears "
                    f"{source.count(stripped)} time(s)")
    return [message]


def changed_lines(source: str, old: str, new: str) -> set[int]:
    """The lines of `source` (1-based) that replacing `old` with `new` changes.

    Only the old lines that differ from `new` count, not unchanged context around them. A pure
    insertion counts the old line it follows (or, at the very start, the one it precedes).
    """
    first = source.count("\n", 0, source.index(old)) + 1
    old_lines = old.splitlines(keepends=True)
    out = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, old_lines, new.splitlines(keepends=True)).get_opcodes():
        if tag in ("replace", "delete"):
            out.update(range(first + i1, first + i2))
        elif tag == "insert":
            out.add(first + max(i1 - 1, 0))
    return out


def ran_line(source: str, old: str, new: str, executed: set[int]) -> bool:
    """True when a statement on a line the mutation changes has a line in `executed`.

    A statement's own lines are all its lines for a simple statement, and only the header (with
    any decorators) for a compound one, so a body that never ran does not count for its `if`.
    `except` clauses count as statements. Comments and blank lines belong to none.
    """
    changed = changed_lines(source, old, new)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
            continue
        own_start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        body = getattr(node, "body", None)
        own_end = max(node.lineno, body[0].lineno - 1) if isinstance(body, list) and body else node.end_lineno
        own = set(range(own_start, own_end + 1))
        if own & changed and own & executed:
            return True
    return False


def _pytest(wt: Path, test: str, timeout: float, prefix: tuple[str, ...] = ()) -> tuple[int | None, str]:
    """(pytest's exit code, or None on a timeout; its output) for `test` run alone in `wt`."""
    # -rfEs: pytest's default short summary (fE) plus one SKIPPED line, with its reason, per skip.
    cmd = [sys.executable, *prefix, "-m", "pytest", test, "-q", "-p", "no:cacheprovider", "--tb=short",
           "-rfEs"]
    # No bytecode: Python trusts a cached .pyc when the source's mtime (to the second) and size
    # match, so a same-size mutation would otherwise run the code it replaced.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    # Own session, so a timeout kills pytest's children too. Ctrl-C then reaches only this
    # script, which kills the session before the worktree is removed.
    proc = subprocess.Popen(cmd, cwd=wt, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return None, out
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    return proc.returncode, out


def _skips(out: str) -> list[str]:
    """pytest's SKIPPED summary lines in `out` (one per skip, with its reason; needs -rs)."""
    return [line.strip() for line in out.splitlines() if line.startswith("SKIPPED ")]


def clean_run(wt: Path, test: str, files: list[str], timeout: float, repo: Path) -> dict[str, set[int]]:
    """Run `test` alone under coverage on the unmutated worktree; per file, the lines it ran.

    Python subprocesses the test starts are traced too. The main checkout's copies of `files`
    are measured as well, so a test that imports them instead of the worktree's is refused.
    """
    import coverage

    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / ".coverage"
        rc = Path(tmp) / "coveragerc"
        include = ", ".join(str(root / f) for root in (wt, repo) for f in files)
        rc.write_text(f"[run]\ndata_file = {data_file}\nparallel = true\npatch = subprocess\n"
                      f"include = {include}\n")
        code, out = _pytest(wt, test, timeout, ("-m", "coverage", "run", f"--rcfile={rc}"))
        if code != 0:
            why = f"times out ({timeout:g} s)" if code is None else f"fails before any mutation (pytest exit {code})"
            raise CannotRun(f"{test} {why}:\n{out}")
        if skips := _skips(out):
            raise CannotRun(f"{test} skips before any mutation, so no mutation of it can be judged:\n"
                            + "\n".join(skips))
        cov = coverage.Coverage(config_file=str(rc))
        cov.combine(data_paths=[tmp])
        data = cov.get_data()
        lines_run = {}
        for measured in data.measured_files():
            path = Path(measured).resolve()
            if path.is_relative_to(wt.resolve()):
                lines_run[path.relative_to(wt.resolve()).as_posix()] = set(data.lines(measured) or ())
            else:
                rel = path.relative_to(repo.resolve())
                raise CannotRun(f"{test} imported the main checkout's {rel}, not the worktree's")
        return lines_run


def mutate(wt: Path, m: Mutation, executed: dict[str, set[int]], timeout: float) -> Result:
    path = wt / m.file
    original = path.read_text()
    if original.count(m.old) != 1:
        raise CannotRun(f"{m.name}: the worktree's {m.file} does not hold old text once")
    path.write_text(original.replace(m.old, m.new))
    try:
        code, out = _pytest(wt, m.test, timeout)
    finally:
        path.write_text(original)
    skips = _skips(out)
    outcome = SKIPPED if skips else {None: TIMED_OUT, 0: SURVIVED, 1: CAUGHT}.get(code, ERROR)
    cell, error = (skips[0], None) if skips else failure(out, m.test)
    return Result(m, outcome, ran_line(original, m.old, m.new, executed.get(m.file, set())), cell, error)


def failure(out: str, test: str) -> tuple[str, str | None]:
    """(the failure cell, the exception type that failed the test or None), from pytest's output.

    The cell is the first ``E`` line, with its ``E`` dropped, after the test file's ``file:line``
    from the last traceback frame in that file before it. A bare ``Type:`` line (numpy's
    assert_allclose starts its message with blank lines) gets the first non-empty ``E`` line after
    it. With no ``E`` line, the cell is pytest's last line, such as ``1 passed``.
    """
    lines = out.splitlines()
    first = next((i for i, line in enumerate(lines) if line.startswith("E ")), None)
    if first is None:
        return (lines[-1].strip() if lines else ""), None
    detail = lines[first][1:].strip()
    if detail == "assert" or detail.startswith("assert "):
        error = "AssertionError"  # pytest rewrote the assert and shows the expression only
    else:
        match = re.match(r"([A-Za-z_][\w.]*)(:|$)", detail)
        error = match.group(1) if match else None
        if match and not detail[match.end():].strip():
            after = (line[1:].strip() for line in lines[first + 1:] if line.startswith("E"))
            if more := next((text for text in after if text), None):
                detail = f"{match.group(1)}: {more}"
    test_file = test.split("::")[0]
    frame = re.compile(rf"(?:^|/){re.escape(test_file)}:(\d+): ")
    hits = [match.group(1) for line in lines[:first] if (match := frame.search(line))]
    return (f"{test_file}:{hits[-1]}: {detail}" if hits else detail), error


def exit_code(results: list[Result]) -> int:
    """3 when pytest errored on a mutation or its test skipped; 1 when a result went against its
    expect or its test never ran the mutated line; else 0."""
    if any(r.outcome in NO_VERDICT for r in results):
        return 3
    return int(any(not r.as_expected or not r.ran for r in results))


def report(results: list[Result]) -> list[str]:
    lines = ["| Mutation | Test | Expected | Result | Ran line | First failure (test file:line) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        cells = [r.mutation.name, r.mutation.test, r.mutation.expect, r.result_cell,
                 "yes" if r.ran else "no", r.failure]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    expected = sum(r.as_expected for r in results)
    errors = sum(r.outcome in NO_VERDICT for r in results)
    summary = (f"\n{len(results)} mutation(s): {expected} as expected, {len(results) - expected} not; "
               f"{sum(not r.ran for r in results)} never ran the mutated line; {errors} error(s)")
    if non_assert := sum(r.caught_by_non_assert for r in results):
        summary += f"; {non_assert} caught by an error other than an assert"
    lines.append(summary + ".")
    return lines


def _timeout(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"--timeout must be a finite number > 0, not {text}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--only", nargs="+", action="extend", metavar="NAME",
                        help="run only these mutations, by name (may be given more than once)")
    parser.add_argument("--timeout", type=_timeout, default=300.0, metavar="SECONDS",
                        help="per pytest run (default 300); a mutation that runs longer is 'timed out'")
    args = parser.parse_args(argv)
    try:
        mutations = load_spec(args.spec, REPO, args.only)
    except SpecError as error:
        problems = str(error).splitlines()
        print(f"{len(problems)} spec error(s):", *problems, sep="\n", file=sys.stderr)
        return 2
    try:
        with _worktree.worktree(REPO, "HEAD", copy_uncommitted=True) as wt:
            files = sorted({m.file for m in mutations})
            executed = {test: clean_run(wt.path, test, files, args.timeout, REPO)
                        for test in dict.fromkeys(m.test for m in mutations)}
            results = [mutate(wt.path, m, executed[m.test], args.timeout) for m in mutations]
            if changed := wt.data_changes():
                raise CannotRun("the runs changed the linked data in the main checkout: " + ", ".join(changed))
    except (CannotRun, _worktree.WorktreeError) as error:
        print(f"cannot run: {error}", file=sys.stderr)
        return 3
    print(f"Worktree: {wt.short} + {len(wt.copied)} uncommitted file(s) copied in, "
          f"{len(wt.linked)} data path(s) linked in.")
    if wt.copied:  # named, so an untracked note copied in is not taken for a test input
        print("Copied in: " + ", ".join(wt.copied))
    print()
    print("\n".join(report(results)))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
