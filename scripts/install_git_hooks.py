#!/usr/bin/env python3
"""Install the repository's git hooks (``scripts/git_hooks/``) into the clone
this is run in.

They bring back the local-only files -- ``PLAN_*.md`` and the run configs at
the repository root, git-ignored since ce7c005c -- that a pull, a rebase or a
checkout deletes because the commit before still tracked them. They go where
git looks for hooks (``git rev-parse --git-path hooks``), which every worktree
of a clone shares, so one install covers them all. A hook of the same name
that is not one of these is left alone, and the install fails naming it.

Usage::

    python scripts/install_git_hooks.py

A clone that pulled before installing them gets its files back by running the
hook once by hand, from the commit before that pull::

    git hook run post-checkout -- <commit before the pull> HEAD 1
"""
from __future__ import annotations

import subprocess
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parent / "git_hooks"
HOOKS = ("post-checkout", "post-merge")
#: The line each of our hooks carries: a reinstall replaces a hook holding it,
#: and never one without it.
MARKER = "# HaemoLynx: local-only files"


def hooks_path(cwd=None) -> Path:
    """Where git runs this clone's hooks from (``core.hooksPath`` honoured)."""
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(cwd or ".") / out


def install(cwd=None) -> list[str]:
    """Write each hook into the clone at ``cwd``; returns the hooks left alone."""
    target = hooks_path(cwd)
    target.mkdir(parents=True, exist_ok=True)
    left_alone = []
    for name in HOOKS:
        destination = target / name
        if destination.exists() and MARKER not in destination.read_text(
            encoding="utf-8", errors="replace"
        ):
            left_alone.append(name)
            continue
        # sh reads a CR as part of the command, whatever the checkout's line endings.
        text = (HOOKS_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        destination.write_bytes(text.encode("utf-8"))
        destination.chmod(0o755)
    return left_alone


def main() -> int:
    left_alone = install()
    target = hooks_path()
    for name in HOOKS:
        if name in left_alone:
            print(f"{target / name}: another hook is already there, left alone")
        else:
            print(f"Installed {target / name}")
    return 1 if left_alone else 0


if __name__ == "__main__":
    raise SystemExit(main())
