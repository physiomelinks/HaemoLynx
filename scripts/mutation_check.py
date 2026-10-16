"""Mutation checks: break a source line on purpose and show the test meant to guard it fails.

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

The checks run in a temporary git worktree outside the repo, made from HEAD plus every
uncommitted change in this checkout (tracked edits, deletions, and untracked files that are
not ignored), so the main checkout is never edited. The worktree is removed afterwards, also
after an error or Ctrl-C.

Each distinct test first runs alone under coverage on the unmutated worktree, with Python
subprocesses it starts traced too. It must pass. The main checkout's copies of the mutated
files are measured as well: a test that ran one of them imported the wrong tree, and the run
stops. Then each mutation is applied, its test run alone without coverage, and the file put
back. No run writes bytecode, so a same-size edit is never masked by a cached ``.pyc``.
Results: ``caught`` (pytest exit 1), ``survived`` (exit 0), ``timed out``, ``error`` (any other
pytest exit, such as a collection error; not a catch). ``timed out`` matches neither
``expect``: a hang is a finding to look at, whatever the spec expected.

``Ran line`` says whether the clean run ran a statement on a line the mutation changes
(unchanged context lines in ``old`` don't count); the clean run includes import time, so a
module constant counts.

Output: a header naming the worktree's commit, a Markdown table to paste under a ticket's
``## Answer`` (mutation, test, expected, result, ran line, and the first ``E`` line of the
failure, or pytest's last line when there is none, such as ``1 passed`` for a survivor), and
a summary line.

Exit code: 0 when every result matches its ``expect`` and every test ran its mutated line;
1 when one does not; 2 on a usage error (the spec, or --only/--timeout); 3 when the checks
could not run (git failed, a test failed or timed out before any mutation, a test imported
the main checkout, or pytest errored on a mutation).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class SpecError(Exception):
    """The spec cannot be used as written (exit 2)."""


class CannotRun(Exception):
    """The checks could not be run, so there is no verdict (exit 3)."""


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str
    old: str
    new: str
    test: str
    expect: str = "caught"


@dataclass(frozen=True)
class Result:
    mutation: Mutation
    outcome: str  # caught, survived, timed out, error
    ran: bool
    failure: str

    @property
    def as_expected(self) -> bool:
        return self.outcome == self.mutation.expect


REQUIRED = ("name", "file", "old", "new", "test")
EXPECTS = ("caught", "survived")


def load_spec(path: Path, repo: Path, only: list[str] | None = None) -> list[Mutation]:
    """The spec's mutations (those named in `only`, if given), in spec order.

    Raises SpecError for anything the spec does not pin down exactly: a missing or unknown key,
    a file outside the repo, under tests/ or not .py, or old text that is not in the file once.
    """
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SpecError(f"cannot read {path}: {error}") from None
    unknown = sorted(set(data) - {"mutation"})
    if unknown:
        raise SpecError(f"unknown top-level key(s): {', '.join(unknown)}")
    entries = data.get("mutation")
    if not isinstance(entries, list) or not entries:
        raise SpecError("no [[mutation]] tables")
    mutations, names = [], set()
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise SpecError(f"mutation {i} is not a table")
        label = f"mutation {i} ({entry.get('name', '?')})"
        missing = [k for k in REQUIRED if k not in entry]
        if missing:
            raise SpecError(f"{label}: missing {', '.join(missing)}")
        extra = sorted(set(entry) - set(REQUIRED) - {"expect"})
        if extra:
            raise SpecError(f"{label}: unknown key(s): {', '.join(extra)}")
        for key, value in entry.items():
            if not isinstance(value, str):
                raise SpecError(f"{label}: {key} must be a string")
        rel = Path(entry["file"])  # Path drops ./ and doubled slashes, as coverage's paths do
        m = Mutation(**{**entry, "file": rel.as_posix()})
        if m.expect not in EXPECTS:
            raise SpecError(f"{label}: expect must be 'caught' or 'survived', not {m.expect!r}")
        if m.name in names:
            raise SpecError(f"{label}: name already used")
        names.add(m.name)
        if not m.old:
            raise SpecError(f"{label}: old is empty")
        if m.new == m.old:
            raise SpecError(f"{label}: new is the same as old")
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise SpecError(f"{label}: file must be a path inside the repo")
        if rel.parts[0] == "tests":
            raise SpecError(f"{label}: {m.file} is under tests/")
        if not (repo / rel).is_file():
            raise SpecError(f"{label}: {m.file} is not a file in the repo")
        if rel.suffix != ".py":
            raise SpecError(f"{label}: {m.file} is not a .py file")
        count = (repo / rel).read_text().count(m.old)
        if count != 1:
            raise SpecError(f"{label}: old appears {count} times in {m.file}, not once")
        mutations.append(m)
    if only is not None:
        unknown = [n for n in only if n not in names]
        if unknown:
            raise SpecError(f"--only name(s) not in the spec: {', '.join(unknown)}")
        mutations = [m for m in mutations if m.name in only]
    return mutations


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


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CannotRun(f"git {' '.join(args[:2])} failed: {proc.stderr.strip()}")
    return proc.stdout


@contextlib.contextmanager
def worktree(repo: Path):
    """(worktree, HEAD's short hash, number of uncommitted files copied in or deleted): a detached
    worktree of HEAD with this checkout's uncommitted changes applied, removed on exit."""
    tmp = Path(tempfile.mkdtemp(prefix="mutation_check_"))
    wt = tmp / "wt"
    try:
        _git(repo, "worktree", "add", "-q", "--detach", str(wt), "HEAD")
    except CannotRun:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        changed = [line.split("\t", 1) for line in
                   _git(repo, "diff", "--name-status", "--no-renames", "HEAD").splitlines()]
        untracked = [["?", rel] for rel in _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()]
        for status, rel in changed + untracked:
            if status == "D":
                (wt / rel).unlink()
            else:
                (wt / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / rel, wt / rel, follow_symlinks=False)
        yield wt, _git(repo, "rev-parse", "--short=7", "HEAD").strip(), len(changed) + len(untracked)
    finally:
        # Forced twice, remove also takes a locked worktree. Its failure is not raised: raising
        # here would hide an error already on its way out, and rmtree and prune finish the job.
        subprocess.run(["git", "worktree", "remove", "--force", "--force", str(wt)], cwd=repo,
                       capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        _git(repo, "worktree", "prune")


def _pytest(wt: Path, test: str, timeout: float, prefix: tuple[str, ...] = ()) -> tuple[int | None, str]:
    """(pytest's exit code, or None on a timeout; its output) for `test` run alone in `wt`."""
    cmd = [sys.executable, *prefix, "-m", "pytest", test, "-q", "-p", "no:cacheprovider", "--tb=short"]
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
    outcome = {None: "timed out", 0: "survived", 1: "caught"}.get(code, "error")
    lines = out.splitlines()
    failure = next((line.strip() for line in lines if line.startswith("E ")), lines[-1] if lines else "")
    return Result(m, outcome, ran_line(original, m.old, m.new, executed.get(m.file, set())), failure)


def exit_code(results: list[Result]) -> int:
    """3 when pytest errored on a mutation; 1 when a result went against its expect or its test
    never ran the mutated line; else 0."""
    if any(r.outcome == "error" for r in results):
        return 3
    return int(any(not r.as_expected or not r.ran for r in results))


def report(results: list[Result]) -> list[str]:
    lines = ["| Mutation | Test | Expected | Result | Ran line | First failure line |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        cells = [r.mutation.name, r.mutation.test, r.mutation.expect, r.outcome,
                 "yes" if r.ran else "no", r.failure]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    expected = sum(r.as_expected for r in results)
    errors = sum(r.outcome == "error" for r in results)
    lines.append(f"\n{len(results)} mutation(s): {expected} as expected, {len(results) - expected} not; "
                 f"{sum(not r.ran for r in results)} never ran the mutated line; {errors} error(s).")
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
        print(error, file=sys.stderr)
        return 2
    try:
        with worktree(REPO) as (wt, head, copied):
            files = sorted({m.file for m in mutations})
            executed = {test: clean_run(wt, test, files, args.timeout, REPO)
                        for test in dict.fromkeys(m.test for m in mutations)}
            results = [mutate(wt, m, executed[m.test], args.timeout) for m in mutations]
    except CannotRun as error:
        print(f"cannot run: {error}", file=sys.stderr)
        return 3
    print(f"Worktree: {head} + {copied} uncommitted file(s) copied in.\n")
    print("\n".join(report(results)))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
