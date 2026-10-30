"""The pre-commit hook runs the committing checkout's own copy, whatever `core.hooksPath` says.

Claude Code's worktree setup rewrites a relative `core.hooksPath` to the main checkout's
absolute `scripts/git-hooks`; this checkout now sets it absolute itself. Git then runs the main
checkout's `pre-commit` from every worktree, which used to source that worktree's own
`_stamp.sh`: two versions mixed. Each test builds a toy repo with a worktree, never this
checkout. Most stage only an edit to a `.md` file, so the real hook stops at its "only .md
files staged" step, which finds no pointer test in the toy and runs nothing; one stages a toy
test and runs it through the hook's pytest step.

The hook also refuses a commit that adds a `.md` file other than the three kept docs: every
other `.md` is a local note, and the hook asks for a `.gitignore` line instead. A merge may
bring new ones in, so a merge commit is let through. A commit that stages only `.md` files
runs the local-note pointer test alone, since an edited doc can still point at a note.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import TOY_ENV, toy_git

HOOKS = Path(__file__).resolve().parent.parent / "scripts" / "git-hooks"
GUARD = "tests/test_tracked_files_name_no_local_note.py"
# Toy note names, built from pieces so the pointer test does not read them as local notes.
PLAN = "docs/plan" + ".md"
GUIDE = "tutorials/guide" + ".md"
ONLY_DOCS = f"pre-commit: only .md files staged, and this checkout has no {GUARD}, so no tests ran."

# Stands in for a branch's changed hook: it only leaves a marker in its checkout.
STUB = '#!/usr/bin/env bash\ntouch "$(git rev-parse --show-toplevel)/stub-ran"\n'


def _toy(tmp_path, stub=False):
    """A toy main checkout holding copies of this checkout's hook files (untracked), and a
    worktree `wt` on branch `wt`, which has a committed stub `pre-commit` when `stub`."""
    main = tmp_path / "main"
    (main / "scripts" / "git-hooks").mkdir(parents=True)
    for name in ("pre-commit", "_stamp.sh"):
        shutil.copy2(HOOKS / name, main / "scripts" / "git-hooks" / name)
    toy_git(main, "init", "-q", "-b", "main")
    (main / "README.md").write_text("toy\n")
    toy_git(main, "add", "README.md")
    toy_git(main, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    toy_git(main, "worktree", "add", "-q", "-b", "wt", str(wt))
    if stub:
        hook = wt / "scripts" / "git-hooks" / "pre-commit"
        hook.parent.mkdir(parents=True)
        hook.write_text(STUB)
        hook.chmod(0o755)
        toy_git(wt, "add", "scripts/git-hooks/pre-commit")
        toy_git(wt, "commit", "-q", "-m", "stub hook")
    return main, wt


def _commit(checkout, message):
    """Commit what is staged in `checkout`, hook included; the finished process."""
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                           "-m", message], cwd=checkout, env=TOY_ENV, capture_output=True,
                          text=True)


def _commit_md(checkout):
    """Add a line to the toy's README.md in `checkout` and commit it; the finished process."""
    readme = checkout / "README.md"
    readme.write_text(readme.read_text() + "more\n")
    toy_git(checkout, "add", "README.md")
    return _commit(checkout, "readme")


def _commit_new(checkout, *names):
    """Stage new files `names` in `checkout` and commit them; the finished process."""
    for name in names:
        (checkout / name).parent.mkdir(parents=True, exist_ok=True)
        (checkout / name).write_text("text\n")
        toy_git(checkout, "add", name)
    return _commit(checkout, "new files")


def _with_python(checkout):
    """Give `checkout` the venv/bin/python the hook looks for: this interpreter."""
    (checkout / "venv" / "bin").mkdir(parents=True)
    (checkout / "venv" / "bin" / "python").symlink_to(sys.executable)


def _absolute_hooks_path(main):
    toy_git(main, "config", "core.hooksPath", str(main / "scripts" / "git-hooks"))


def test_a_worktree_commit_runs_the_worktrees_own_hook(tmp_path):
    main, wt = _toy(tmp_path, stub=True)
    _absolute_hooks_path(main)

    proc = _commit_md(wt)

    assert (wt / "stub-ran").exists(), proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_a_main_checkout_commit_runs_the_hook_once(tmp_path):
    main, _ = _toy(tmp_path)
    for i, hooks_path in enumerate(("scripts/git-hooks", str(main / "scripts" / "git-hooks"))):
        toy_git(main, "config", "core.hooksPath", hooks_path)

        proc = _commit_md(main)

        assert proc.returncode == 0, (hooks_path, proc.stderr)
        assert proc.stderr.count(ONLY_DOCS) == 1, (hooks_path, proc.stderr)


def test_a_worktree_without_its_own_hook_commits_with_one_line_saying_so(tmp_path):
    main, wt = _toy(tmp_path)
    _absolute_hooks_path(main)

    proc = _commit_md(wt)

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.splitlines() == [
        f"pre-commit: {wt.resolve()} has no scripts/git-hooks/pre-commit, so no hook ran."], proc.stderr


def test_the_hook_runs_the_fast_tests_without_git_variables(tmp_path):
    """In a worktree git hands the hook GIT_DIR and GIT_INDEX_FILE; a test that runs `git` in a
    temp folder would then write into the committing repo."""
    main, wt = _toy(tmp_path)
    shutil.copytree(main / "scripts", wt / "scripts")
    _absolute_hooks_path(main)
    _with_python(wt)
    (wt / "tests").mkdir()
    (wt / "tests" / "test_env.py").write_text(
        "import os\n\n\ndef test_no_git_variables():\n"
        "    assert [k for k in os.environ if k.startswith('GIT_')] == []\n")
    toy_git(wt, "add", "tests/test_env.py")
    proc = _commit(wt, "env test")

    log = wt / toy_git(wt, "rev-parse", "--git-path", "hook-stamp").strip() / "fast-tests.log"
    assert proc.returncode == 0, proc.stderr
    assert "1 passed" in log.read_text(), log.read_text()


def test_a_commit_adding_a_local_note_is_refused(tmp_path):
    main, _ = _toy(tmp_path)
    _absolute_hooks_path(main)

    proc = _commit_new(main, PLAN)

    assert proc.returncode == 1, proc.stderr
    assert PLAN in proc.stderr and ".gitignore" in proc.stderr, proc.stderr
    assert toy_git(main, "log", "--format=%s") == "init\n"


def test_a_rename_to_a_new_md_name_is_refused(tmp_path):
    main, _ = _toy(tmp_path)
    _absolute_hooks_path(main)
    toy_git(main, "mv", "README.md", "notes.md")

    proc = _commit(main, "rename")

    assert proc.returncode == 1, proc.stderr
    assert "notes.md" in proc.stderr, proc.stderr


def test_a_commit_adding_a_kept_doc_is_let_through(tmp_path):
    main, _ = _toy(tmp_path)
    _absolute_hooks_path(main)

    proc = _commit_new(main, "GLOSSARY.md", "CODING_STANDARDS.md", "tutorials/README.md")

    assert proc.returncode == 0, proc.stderr


def test_a_merge_may_bring_in_new_md_files(tmp_path):
    main, wt = _toy(tmp_path)
    _absolute_hooks_path(main)
    (wt / "tutorials").mkdir()
    (wt / GUIDE).write_text("upstream's doc\n")
    toy_git(wt, "add", GUIDE)
    toy_git(wt, "commit", "-q", "--no-verify", "-m", "upstream doc")
    toy_git(main, "merge", "-q", "--no-ff", "--no-commit", "wt")

    proc = _commit(main, "merge")

    assert proc.returncode == 0, proc.stderr
    assert (main / GUIDE).exists()


def _docs_only_toy(tmp_path, guard_passes):
    """A toy checkout with a stand-in pointer test that passes or fails, and another test that
    always fails, so a commit shows which of them the hook ran."""
    main, _ = _toy(tmp_path)
    _absolute_hooks_path(main)
    _with_python(main)
    (main / "tests").mkdir()
    (main / GUARD).write_text(f"def test_guard():\n    assert {guard_passes}, 'pointer found'\n")
    (main / "tests" / "test_other.py").write_text("def test_other():\n    assert False\n")
    toy_git(main, "add", "tests")
    toy_git(main, "commit", "-q", "--no-verify", "-m", "tests")
    return main


def test_a_docs_only_commit_runs_the_pointer_test_alone(tmp_path):
    main = _docs_only_toy(tmp_path, guard_passes=True)

    proc = _commit_md(main)

    assert proc.returncode == 0, proc.stderr
    assert "1 passed" in proc.stderr, proc.stderr


def test_a_docs_only_commit_is_refused_when_the_pointer_test_fails(tmp_path):
    main = _docs_only_toy(tmp_path, guard_passes=False)

    proc = _commit_md(main)

    assert proc.returncode == 1, proc.stderr
    assert "pointer found" in proc.stderr, proc.stderr
