"""The pre-commit hook runs the committing checkout's own copy, whatever `core.hooksPath` says.

Claude Code's worktree setup rewrites a relative `core.hooksPath` to the main checkout's
absolute `scripts/git-hooks` (ticket retro-14-19-followups/04). Git then runs the main
checkout's `pre-commit` from every worktree, which used to source that worktree's own
`_stamp.sh`: two versions mixed. Each test builds a toy repo with a worktree, never this
checkout, and stages only a `.md` file, so the real hook stops at its "only .md files staged"
exit without running pyflakes or pytest.
"""
import shutil
import subprocess
from pathlib import Path

from conftest import TOY_ENV, toy_git

HOOKS = Path(__file__).resolve().parent.parent / "scripts" / "git-hooks"
ONLY_DOCS = "pre-commit: only .md files staged, skipping fast tests."

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


def _commit_md(checkout, name="notes.md"):
    """Stage one new .md file `name` in `checkout` and commit it; the finished process."""
    (checkout / name).write_text("notes\n")
    toy_git(checkout, "add", name)
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                           "-m", "notes"], cwd=checkout, env=TOY_ENV, capture_output=True,
                          text=True)


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

        proc = _commit_md(main, f"notes{i}.md")

        assert proc.returncode == 0, (hooks_path, proc.stderr)
        assert proc.stderr.count(ONLY_DOCS) == 1, (hooks_path, proc.stderr)


def test_a_worktree_without_its_own_hook_commits_with_one_line_saying_so(tmp_path):
    main, wt = _toy(tmp_path)
    _absolute_hooks_path(main)

    proc = _commit_md(wt)

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.splitlines() == [
        f"pre-commit: {wt.resolve()} has no scripts/git-hooks/pre-commit, so no hook ran."], proc.stderr
