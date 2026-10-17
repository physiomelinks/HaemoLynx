"""The git-driving script tests leave the repository a hook's GIT_* variables name untouched.

A pre-commit hook runs with GIT_DIR and GIT_INDEX_FILE set to the committing checkout. A test
that runs `git init`/`git add` in a toy folder with them still set writes into that checkout
instead: on 2026-10-17 a leak like this set `core.bare = true` in this repo's shared config.
Here the leaked variables name a throwaway toy repo's worktree, never this checkout, and the
toy's config and index must come out byte-identical.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import TOY_ENV, toy_git

REPO = Path(__file__).resolve().parent.parent

# The tests that drive git on toy repos through scripts/_worktree.py or conftest's TOY_ENV.
GIT_DRIVING_TESTS = ("tests/test_worktree.py", "tests/test_mutation_check.py",
                     "tests/test_coverage_by_module.py", "tests/test_check_stale_test_names.py",
                     "tests/test_pre_commit_hook.py")


@pytest.mark.slow
def test_git_driving_tests_leave_a_leaked_git_dir_untouched(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    toy_git(main, "init", "-q", "-b", "main")
    (main / "a.txt").write_text("a\n")
    toy_git(main, "add", "a.txt")
    toy_git(main, "commit", "-q", "-m", "a")
    toy_git(main, "worktree", "add", "-q", str(tmp_path / "wt"))
    gitdir = main / ".git" / "worktrees" / "wt"
    watched = [main / ".git" / "config", main / ".git" / "index", gitdir / "index"]
    before = {p: p.read_bytes() for p in watched}

    # As a hook in the toy worktree would leave them; TOY_ENV first drops any GIT_* of our own.
    env = {**TOY_ENV, "GIT_DIR": str(gitdir), "GIT_INDEX_FILE": str(gitdir / "index")}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           *GIT_DRIVING_TESTS], cwd=REPO, env=env, capture_output=True, text=True)

    changed = [str(p.relative_to(tmp_path)) for p in watched if p.read_bytes() != before[p]]
    assert changed == [], proc.stdout[-3000:]
    assert proc.returncode == 0, proc.stdout[-3000:]
