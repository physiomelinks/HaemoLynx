"""The git hooks that bring back local-only files after a pull
(``scripts/git_hooks/``, installed by ``scripts/install_git_hooks.py``).

Each test builds throwaway repositories: an origin, a clone that still tracks
a plan file and run configs, and a second clone that takes them off origin as
ce7c005c did, with a real deletion beside them.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "install_git_hooks.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

LOCAL_ONLY = {
    "PLAN_example.md": "# A plan\n\nStep one.\n",
    "config.yaml": "do_skeletonize: true\n",
    "Run.yml": "image: E14.5.tif\n",
}
TRACKED = {
    "examples/example_config.yaml": "do_skeletonize: false\n",
    "docs/PLAN_notes.md": "Not at the root.\n",
}
DELETED_FOR_REAL = "src/module.py"
GITIGNORE = "/PLAN_*.md\n/*.yaml\n/*.yml\n"


def git(cwd, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def write(root: Path, files: dict) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def install(clone: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(INSTALLER)], cwd=clone, capture_output=True, text=True
    )


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """(mine, seed): ``mine`` tracks the local-only files; origin has dropped them."""
    # The user's and the system's git config (autocrlf, hooksPath) stay out.
    (tmp_path / "gitconfig").write_text("")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.com")

    git(tmp_path, "init", "-q", "--bare", "-b", "main", "origin.git")
    seed, mine = tmp_path / "seed", tmp_path / "mine"
    git(tmp_path, "clone", "-q", "origin.git", str(seed))
    write(seed, {**LOCAL_ONLY, **TRACKED, DELETED_FOR_REAL: "x = 1\n"})
    git(seed, "add", ".")
    git(seed, "commit", "-q", "-m", "everything tracked")
    git(seed, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", "origin.git", str(mine))

    git(seed, "rm", "-q", "--cached", *LOCAL_ONLY)
    git(seed, "rm", "-q", DELETED_FOR_REAL)
    write(seed, {".gitignore": GITIGNORE})
    git(seed, "add", ".gitignore")
    git(seed, "commit", "-q", "-m", "local-only files off origin")
    git(seed, "push", "-q", "origin", "HEAD:main")
    return mine, seed


def contents(clone: Path, names) -> dict:
    return {
        name: (clone / name).read_bytes().decode("utf-8") if (clone / name).exists() else None
        for name in names
    }


def test_without_the_hooks_a_pull_deletes_them(repos):
    mine, _ = repos
    git(mine, "pull", "-q", "--ff-only")
    assert contents(mine, LOCAL_ONLY) == dict.fromkeys(LOCAL_ONLY)


@pytest.mark.parametrize(
    "update, local_commit",
    [
        (["pull", "-q", "--ff-only"], False),
        (["pull", "-q", "--rebase"], True),
        (["pull", "-q", "--no-rebase", "--no-edit"], True),
        (["checkout", "-q", "-b", "updated", "origin/main"], False),
    ],
    ids=["fast-forward pull", "rebasing pull", "merging pull", "checkout"],
)
def test_an_update_brings_back_the_files_it_takes_off_origin(repos, update, local_commit):
    mine, _ = repos
    assert install(mine).returncode == 0
    if local_commit:
        write(mine, {"examples/example_config.yaml": "do_skeletonize: true\n"})
        git(mine, "commit", "-q", "-am", "a commit of mine")
    git(mine, "fetch", "-q")

    git(mine, *update)

    assert contents(mine, LOCAL_ONLY) == LOCAL_ONLY
    assert not (mine / DELETED_FOR_REAL).exists()
    tracked = git(mine, "ls-files").split()
    assert set(TRACKED) <= set(tracked)
    assert not set(LOCAL_ONLY) & set(tracked)
    assert git(mine, "status", "--porcelain") == ""
    if local_commit:
        assert git(mine, "log", "-1", "--format=%s", "--no-merges") == "a commit of mine\n"


def test_the_hook_by_hand_brings_back_an_earlier_pull(repos):
    mine, _ = repos
    git(mine, "pull", "-q", "--ff-only")
    assert install(mine).returncode == 0

    git(mine, "hook", "run", "post-checkout", "--", "HEAD~1", "HEAD", "1")

    assert contents(mine, LOCAL_ONLY) == LOCAL_ONLY
    assert not (mine / DELETED_FOR_REAL).exists()


def test_a_file_already_there_is_never_written_over(repos):
    mine, _ = repos
    git(mine, "pull", "-q", "--ff-only")
    write(mine, {"config.yaml": "edited since\n"})
    assert install(mine).returncode == 0

    git(mine, "hook", "run", "post-checkout", "--", "HEAD~1", "HEAD", "1")

    assert contents(mine, LOCAL_ONLY) == {**LOCAL_ONLY, "config.yaml": "edited since\n"}


def test_a_file_checkout_restores_nothing(repos):
    mine, _ = repos
    git(mine, "pull", "-q", "--ff-only")
    assert install(mine).returncode == 0

    git(mine, "hook", "run", "post-checkout", "--", "HEAD~1", "HEAD", "0")

    assert contents(mine, LOCAL_ONLY) == dict.fromkeys(LOCAL_ONLY)


def test_the_installer_leaves_someone_elses_hook_alone(repos):
    mine, _ = repos
    hooks = mine / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "post-merge").write_bytes(b"#!/bin/sh\necho theirs\n")

    result = install(mine)

    assert result.returncode == 1
    assert "post-merge: another hook is already there, left alone" in result.stdout
    assert (hooks / "post-merge").read_bytes() == b"#!/bin/sh\necho theirs\n"
    assert b"local-only" in (hooks / "post-checkout").read_bytes()


def test_installing_again_replaces_our_own_hooks_with_lf_endings(repos):
    mine, _ = repos
    hooks = mine / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "post-checkout").write_bytes(b"#!/bin/sh\r\n# HaemoLynx: local-only files\r\nexit 0\r\n")

    assert install(mine).returncode == 0
    assert install(mine).returncode == 0

    for name in ("post-checkout", "post-merge"):
        text = (hooks / name).read_bytes()
        assert b"\r" not in text
        assert text == (INSTALLER.parent / "git_hooks" / name).read_bytes().replace(b"\r\n", b"\n")
