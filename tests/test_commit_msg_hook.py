"""The commit-msg hook refuses a message that points at a local note.

A commit message is read in a clone that has no local notes, so it states its reasons in words,
like tracked files do. The hook runs the rules of ``scripts/local_note_pointers.py`` on the
message. Each test builds a toy repo with copies of this checkout's hook and scripts and commits
through git, never in this checkout. The toy text is built from pieces, so this file does not
match the rules itself.
"""
import shutil
import subprocess
from pathlib import Path

from conftest import TOY_ENV, toy_git

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
OPEN_ITEM = "open" + " item 3"
PLAN = "plan" + ".md"


def _toy(tmp_path):
    """A toy checkout using copies of this checkout's commit-msg hook and pointer rules."""
    repo = tmp_path / "main"
    (repo / "scripts" / "git-hooks").mkdir(parents=True)
    shutil.copy2(SCRIPTS / "git-hooks" / "commit-msg", repo / "scripts" / "git-hooks")
    for name in ("local_note_pointers.py", "_worktree.py"):
        shutil.copy2(SCRIPTS / name, repo / "scripts")
    toy_git(repo, "init", "-q", "-b", "main")
    toy_git(repo, "config", "core.hooksPath", str(repo / "scripts" / "git-hooks"))
    (repo / "README.md").write_text("toy\n")
    toy_git(repo, "add", "README.md")
    return repo


def _commit(repo, *args, env=TOY_ENV):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                           *args], cwd=repo, env=env, capture_output=True, text=True)


def test_a_message_citing_an_open_item_is_refused(tmp_path):
    repo = _toy(tmp_path)

    proc = _commit(repo, "-m", "fix: the solver", "-m", f"Closes {OPEN_ITEM}.")

    assert proc.returncode == 1, proc.stderr
    assert f"Closes {OPEN_ITEM}." in proc.stderr, proc.stderr
    assert toy_git(repo, "rev-list", "--all") == ""


def test_a_message_naming_an_untracked_note_or_an_issue_item_is_refused(tmp_path):
    repo = _toy(tmp_path)
    (repo / PLAN).write_text("On disk, never tracked.\n")
    issue_item = "#" + "98 item 7"

    proc = _commit(repo, "-m", f"fix: as {PLAN} says", "-m", f"Done under {issue_item}.")

    assert proc.returncode == 1, proc.stderr
    assert PLAN in proc.stderr and issue_item in proc.stderr, proc.stderr


def test_a_message_stating_its_reasons_in_words_is_let_through(tmp_path):
    repo = _toy(tmp_path)

    proc = _commit(repo, "-m", "fix (#98): say the reason in words",
                   "-m", "README.md says how to run it.")

    assert proc.returncode == 0, proc.stderr


def test_gits_comment_lines_and_the_verbose_diff_are_not_read(tmp_path):
    """`commit -v` puts the staged diff below a scissors line; git strips it, and its comment
    lines, after the hook has run."""
    repo = _toy(tmp_path)
    (repo / "calc.py").write_text(f"# Fixed under {OPEN_ITEM}.\n")
    toy_git(repo, "add", "calc.py")

    proc = _commit(repo, "-v", "-e", "-m", "feat: a calculator",
                   env={**TOY_ENV, "GIT_EDITOR": "true"})

    assert proc.returncode == 0, proc.stderr
    assert toy_git(repo, "log", "--format=%B").strip() == "feat: a calculator"


def test_a_worktree_without_its_own_hook_commits_with_one_line_saying_so(tmp_path):
    repo = _toy(tmp_path)
    toy_git(repo, "commit", "-q", "--no-verify", "-m", "init")
    wt = tmp_path / "wt"
    toy_git(repo, "worktree", "add", "-q", "--detach", str(wt), "HEAD")
    (wt / "README.md").write_text("more\n")
    toy_git(wt, "add", "README.md")

    proc = _commit(wt, "-m", f"docs: {OPEN_ITEM}")

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.splitlines() == [
        f"commit-msg: {wt.resolve()} has no scripts/git-hooks/commit-msg, so no hook ran."]
