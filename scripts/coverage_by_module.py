"""Per-module line coverage of the in-scope CB source, saved to JSON and compared run to run.

Proves a test cleanup lost no coverage: run it before the edit and after, with the same
pytest arguments, then compare.

    venv/bin/python scripts/coverage_by_module.py run BEFORE.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py run AFTER.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py run --files-from FILE OUT.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py run --at REV BEFORE.json [PYTEST_ARGS...]
    venv/bin/python scripts/coverage_by_module.py compare BEFORE.json AFTER.json
    venv/bin/python scripts/coverage_by_module.py check RUN.json REV
    venv/bin/python scripts/coverage_by_module.py table BEFORE.json AFTER.json [--modules PATH ...]
    venv/bin/python scripts/coverage_by_module.py sole RUN.json TEST_ID

``run`` runs pytest under pytest-cov with ``--cov-context=test`` and writes, per in-scope
module, its statement count and covered lines, and per test the in-scope lines it ran. The
coverage data file goes to a temporary folder, never the repo. With no PYTEST_ARGS it runs
pytest's default (the whole suite). With ``--files-from FILE`` it runs only the test files
listed in FILE (one path per line, relative to the repo; blank lines and ``#`` comments are
ignored), appended to PYTEST_ARGS and saved with them, so ``compare`` refuses runs over different
lists. PYTEST_ARGS then carries options only: a test path given there runs too. One such list is
``scripts/coverage_in_scope_tests.txt``; its header says where it comes from. A missing list,
a missing listed file or a list with no files is a usage error (exit 2, nothing run).

``run`` measures this checkout, and refuses to start (exit 2) when its tracked files differ
from HEAD, since the result would then belong to no commit; ``--allow-dirty`` runs anyway and
records ``"dirty": true``. Untracked files do not count. The tracked files are checked again
after the run: a change while it ran (an edit, or a test writing to the tree) fails it.

``run`` saves the commit it measured, the ``tree`` id of the tracked files it measured (what
``git write-tree`` gives for them as they stood on disk) and a ``dirty`` flag, so a file
labelled "after = the commit" can be checked: a clean run's tree is its commit's tree. The tree
id is computed in a throwaway index and object folder, so nothing is written to the repo.

``run --at REV`` measures commit REV instead, for a "before" baseline without stashing or
copying anything. It makes a detached worktree of REV in a temporary folder outside the repo,
symlinks the gitignored data folders (``_worktree.DATA_FOLDERS``) in from this
checkout, runs there, and removes the worktree, also after an error or Ctrl-C. This checkout
is never edited, so it may be dirty. A symlink cannot be made read-only, so a run that adds,
changes or removes anything through the links fails and names the paths.
``scripts/_worktree.py`` builds the worktree, shared with ``mutation_check.py``.

Every ``run`` fails when a ``realdata`` test skips: a baseline without the real data is not
like for like. Exit code: pytest's when it is not 0; else 3 when a ``realdata`` test skipped,
the tracked files changed during the run, the run changed the linked data, a test imported
this checkout's source instead of the worktree's, or git could not make the worktree; else 0.
The JSON is still written, with those findings in it, and ``compare`` refuses it.

``compare`` prints each run's commit and tree id, then a ``WARNING`` line (the exit code is
unchanged) for a run whose tree is not the tree of its commit, or whose commit this repository
does not have. It then prints the passed and skipped counts of both runs, the test ids only in
BEFORE and only in AFTER, then every module whose covered count changed. ``check RUN.json REV``
says whether the run's tree is the tree of commit REV: exit 0 yes, 1 no, 2 REV is no commit
here, 3 the run recorded no tree. For ``compare``, exit 0 when no module's covered
count dropped, 1 when one did, 2 on a usage error, 3 when the two runs cannot be compared
(different pytest arguments, a run whose tests failed, or a run ``run`` failed as above).

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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# scripts/ is not a package. No .pyc is written for the helper: the checkout stays as it was.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
import _worktree  # noqa: E402
sys.dont_write_bytecode = _bytecode

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


def in_scope_files(root: Path = REPO) -> list[Path]:
    paths = sorted((root / "src/ImageLynx").rglob("*.py")) + sorted((root / "examples").rglob("*.py"))
    return [p for p in paths if in_scope(str(p.relative_to(root)))]


def read_data(data_file: Path, root: Path = REPO) -> dict:
    """Modules and per-test lines from a coverage data file written with --cov-context=test.

    `root` is the tree the run measured. Raises ValueError when it is a worktree and the run
    measured an in-scope file of this checkout instead (a test imported the wrong tree).
    """
    import coverage

    cov = coverage.Coverage(data_file=str(data_file))
    cov.load()  # get_data() on an unloaded Coverage returns an empty data set
    data = cov.get_data()
    measured = {str(Path(f).resolve()): f for f in data.measured_files()}
    if root.resolve() != REPO.resolve():
        here = [Path(f).relative_to(REPO.resolve()).as_posix() for f in measured
                if Path(f).is_relative_to(REPO.resolve())]
        foreign = sorted(rel for rel in here if in_scope(rel))
        if foreign:
            raise ValueError(f"the run imported this checkout's {', '.join(foreign)}, not the worktree's")
    modules, tests = {}, defaultdict(lambda: defaultdict(list))
    for path in in_scope_files(root):
        rel = str(path.relative_to(root))
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


def read_files_from(path: Path, root: Path = REPO) -> list[str]:
    """The test files listed in `path`. Raises ValueError on a file missing from `root`, or an
    empty list."""
    files = [s for line in path.read_text().splitlines() if (s := line.split("#", 1)[0].strip())]
    if not files:
        raise ValueError(f"{path} lists no test files")
    missing = [f for f in files if not (root / f).exists()]
    if missing:
        raise ValueError(f"{path} lists files that do not exist: {', '.join(missing)}")
    return files


def tracked_state(root: Path) -> tuple[bool, str]:
    """(dirty, fingerprint) of `root`: whether its tracked files differ from HEAD, and a hash of
    HEAD and that difference, which changes when a tracked file does."""
    head = _worktree.git(root, "rev-parse", "HEAD").strip()
    diff = _worktree.git(root, "diff", "--binary", "HEAD")
    return bool(diff), hashlib.sha256(f"{head}\0{diff}".encode()).hexdigest()


def working_tree(root: Path) -> str:
    """The tree id of `root`'s tracked files as they stand on disk (`git write-tree` of them).

    Taken in a copy of the index, with a throwaway object folder behind the repo's own, so the
    index, the object database and the working tree of `root` are all left as they were.
    Untracked files are not in it; a file staged as new is.
    """
    def git_path(name):
        return Path(_worktree.git(root, "rev-parse", "--path-format=absolute", "--git-path", name).strip())

    index, objects = git_path("index"), git_path("objects")
    with tempfile.TemporaryDirectory() as tmp:
        if index.exists():
            shutil.copy2(index, Path(tmp) / "index")
        (Path(tmp) / "objects").mkdir()
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index"),
               "GIT_OBJECT_DIRECTORY": str(Path(tmp) / "objects"),
               "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(objects)}
        _worktree.git(root, "add", "--update", env=env)
        return _worktree.git(root, "write-tree", env=env).strip()


def commit_tree(repo: Path, commit: str) -> str | None:
    """The tree id of `commit` in `repo`, or None when `repo` has no such commit."""
    try:
        return _worktree.git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}").strip()
    except _worktree.WorktreeError:
        return None


DIRTY_REFUSAL = ("refusing to run: tracked files differ from HEAD, so the run would measure no "
                 "commit. Commit them, measure a commit with --at REV, or pass --allow-dirty.")


def run(out: Path, pytest_args: list[str], files_from: Path | None = None, at: str | None = None,
        allow_dirty: bool = False) -> int:
    """Run pytest under coverage in this checkout, or with `at` in a worktree of that commit."""
    try:
        if at is None:
            return _run_in(out, pytest_args, files_from, allow_dirty)
        with _worktree.worktree(REPO, at) as wt:
            return _run_in(out, pytest_args, files_from, allow_dirty, wt)
    except _worktree.WorktreeError as error:
        print(f"cannot run: {error}", file=sys.stderr)
        return 3


def _run_in(out: Path, pytest_args: list[str], files_from: Path | None, allow_dirty: bool,
            wt: _worktree.Worktree | None = None) -> int:
    """The run, in worktree `wt` when given, else in this checkout."""
    root = wt.path if wt is not None else REPO
    if files_from is not None:
        try:
            pytest_args = [*pytest_args, *read_files_from(files_from, root)]
        except (OSError, ValueError) as e:
            print(e, file=sys.stderr)
            return 2
    dirty, state_before = tracked_state(root)
    if dirty and not allow_dirty:
        print(DIRTY_REFUSAL, file=sys.stderr)
        return 2
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
        if wt is not None:
            env["PYTHONDONTWRITEBYTECODE"] = "1"  # nor a .pyc for _cbm_outcomes in this checkout
        code = subprocess.run(cmd, cwd=root, env=env).returncode
        if not data_file.exists():
            print(f"pytest wrote no coverage data (exit {code})", file=sys.stderr)
            return code or 1
        try:
            result = read_data(data_file, root)
        except ValueError as error:
            print(f"cannot run: {error}", file=sys.stderr)
            return 3
        if not outcomes_file.exists():
            print(f"pytest wrote no test outcomes (exit {code})", file=sys.stderr)
            return code or 1
        recorded = json.loads(outcomes_file.read_text())
    dirty_after, state_after = tracked_state(root)
    tree = working_tree(root)
    outcomes = recorded["outcomes"]
    result = {
        "commit": _worktree.git(root, "rev-parse", "HEAD").strip(),
        "tree": tree,
        "at": wt.rev if wt is not None else None,
        "dirty": dirty or dirty_after,
        "changed_during_run": state_before != state_after,
        "pytest_args": pytest_args,
        **({"files_from": str(files_from)} if files_from else {}),
        "pytest_exit": code,
        "outcomes": outcomes,
        "realdata_skipped": sorted(t for t in recorded["realdata"] if outcomes.get(t) == "skipped"),
        "data_changed": wt.data_changes() if wt is not None else [],
        **result,
    }
    out.write_text(json.dumps(result, indent=1))
    total_s = sum(m["statements"] for m in result["modules"].values())
    total_c = sum(len(m["covered"]) for m in result["modules"].values())
    print(f"{out}: {len(result['tests'])} tests, {total_c}/{total_s} in-scope lines covered")
    problems = run_problems(result)
    for problem in problems:
        print(problem, file=sys.stderr)
    return code or (3 if problems else 0)


def run_problems(result: dict) -> list[str]:
    """Why `result` is no baseline: realdata skips, a tree changed mid-run, data written."""
    problems = []
    if skipped := result.get("realdata_skipped"):
        problems.append(f"{len(skipped)} realdata test(s) skipped, so this run is no baseline:\n"
                        + "\n".join(f"  {t}" for t in skipped))
    if result.get("changed_during_run"):
        problems.append("the tracked files changed during the run (edited while it ran, or by a test)")
    if changed := result.get("data_changed"):
        problems.append("the run changed the linked data in the main checkout: " + ", ".join(changed))
    return problems


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


def tree_warning(name: str, result: dict, repo: Path) -> str | None:
    """A WARNING line when `result`'s tree is not the tree of its commit in `repo`, else None.

    A run saved before trees were recorded has nothing to check.
    """
    if "tree" not in result:
        return None
    commit = result["commit"][:7]
    expected = commit_tree(repo, result["commit"])
    if expected is None:
        return (f"WARNING: the {name} run's commit {commit} is not in this repository, "
                "so its tree cannot be checked")
    if expected != result["tree"]:
        return (f"WARNING: the {name} run's tree {result['tree'][:7]} is not the tree of its commit "
                f"{commit} ({expected[:7]}): it measured edits to that commit, or another tree")
    return None


def compare(before: dict, after: dict, repo: Path = REPO) -> tuple[int, list[str]]:
    """(exit code, report lines) for two run results. `repo` is where the commits are looked up."""
    lines = [f"{label} {r['commit'][:7]}, tree "
             f"{r['tree'][:7] if 'tree' in r else 'not recorded'}{' (dirty)' if r['dirty'] else ''}"
             for label, r in (("before:", before), ("after: ", after))]
    lines += [w for name, r in (("before", before), ("after", after)) if (w := tree_warning(name, r, repo))]
    if before["pytest_args"] != after["pytest_args"]:
        return 3, lines + [f"cannot compare: pytest args differ: {before['pytest_args']} vs {after['pytest_args']}"]
    for name, r in (("before", before), ("after", after)):
        if r["pytest_exit"] != 0:
            return 3, lines + [f"cannot compare: the {name} run's pytest exited {r['pytest_exit']}"]
        if problems := run_problems(r):
            return 3, lines + [f"cannot compare: the {name} run is no baseline", *problems]
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


def check(result: dict, rev: str, repo: Path = REPO) -> tuple[int, str]:
    """(exit code, message): is `result`'s tree the tree of commit `rev`? 0 yes, 1 no, 2 `rev`
    is no commit in `repo`, 3 the run recorded no tree."""
    if "tree" not in result:
        return 3, "no tree recorded in this run (saved before trees were recorded)"
    expected = commit_tree(repo, rev)
    if expected is None:
        return 2, f"{rev} is not a commit in this repository"
    short = rev[:7]
    if expected == result["tree"]:
        return 0, f"tree {result['tree'][:7]} is the tree of {short}"
    return 1, f"tree {result['tree'][:7]} is not the tree of {short} ({expected[:7]})"


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
    p = sub.add_parser("run", help="run pytest under coverage and save per-module lines",
                       description="Run pytest under coverage and save per-module lines. Refuses to "
                       "start (exit 2) when this checkout's tracked files differ from HEAD; pass "
                       "--allow-dirty to run anyway, or --at REV to measure a commit instead. Fails "
                       "(exit 3) when a realdata test skips.")
    p.add_argument("--files-from", type=Path, metavar="FILE",
                   help="run only the test files listed in FILE (one per line, # comments)")
    p.add_argument("--at", metavar="REV",
                   help="measure commit REV in a temporary worktree, with the gitignored data folders "
                   "symlinked in from this checkout; this checkout is never edited, and may be dirty")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run even when tracked files differ from HEAD (recorded as dirty in the JSON)")
    p.add_argument("out", type=Path)
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p = sub.add_parser("compare", help="exit 1 if any module's covered count dropped")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p = sub.add_parser("check", help="exit 0 if a run's tree is the tree of commit REV")
    p.add_argument("run", type=Path)
    p.add_argument("rev")
    p = sub.add_parser("table", help="Markdown per-module covered counts, before and after")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--modules", nargs="+", metavar="PATH", help="only these modules, changed or not")
    p = sub.add_parser("sole", help="lines only TEST_ID ran in this run")
    p.add_argument("result", type=Path)
    p.add_argument("test")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return run(args.out, args.pytest_args, args.files_from, args.at, args.allow_dirty)
    if args.cmd == "compare":
        code, report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        print("\n".join(report))
        return code
    if args.cmd == "check":
        code, message = check(json.loads(args.run.read_text()), args.rev)
        print(message, file=sys.stdout if code in (0, 1) else sys.stderr)
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
