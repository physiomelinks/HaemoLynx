"""A throwaway git worktree with the repo's gitignored data linked in.

Shared by ``coverage_by_module.py run --at REV`` and ``mutation_check.py``, so neither edits
the main checkout and neither loses its ``realdata`` tests to missing data.

``worktree(repo, rev)`` makes a detached worktree of ``rev`` in a temporary folder outside the
repo, and removes it on exit, also after an error or Ctrl-C. With ``copy_uncommitted`` it first
copies this checkout's uncommitted changes in (tracked edits, deletions, and untracked files
that are not ignored).

Then the data folders (``DATA_FOLDERS``, gitignored and so missing from any worktree) are
linked in: every entry of this checkout's copy that the worktree lacks becomes a symlink to it.
Nothing is copied. A folder the worktree already has (because ``rev`` tracks a file in it) is
filled entry by entry, and a file ``rev`` tracks keeps its tracked version.

A symlink cannot be made read-only, so the linked data is listed (path, size, mtime) when the
worktree is made, and ``Worktree.data_changes()`` names every path a run added, changed or
removed through the links. Callers treat any such path as an error.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The gitignored in-repo data the realdata tests read (repo-relative).
DATA_FOLDERS = ("examples/images", "examples/outputs")


class WorktreeError(Exception):
    """git failed, so there is no worktree to run in."""


def git(repo: Path, *args: str) -> str:
    """git's stdout, run in `repo`. Raises WorktreeError when git exits non-zero."""
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise WorktreeError(f"git {' '.join(args[:2])} failed: {proc.stderr.strip()}")
    return proc.stdout


def link_data(source: Path, wt: Path, folders=DATA_FOLDERS) -> list[str]:
    """Symlink into `wt` every entry of `source`'s data folders that `wt` lacks; the linked paths."""
    linked: list[str] = []

    def overlay(src: Path, dst: Path, rel: str) -> None:
        if not dst.exists() and not dst.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())
            linked.append(rel)
        elif dst.is_dir() and not dst.is_symlink() and src.is_dir():
            for child in sorted(src.iterdir()):
                overlay(child, dst / child.name, f"{rel}/{child.name}")
        # else: the worktree has its own (tracked) copy, which wins

    for folder in folders:
        if (source / folder).exists():
            overlay(source / folder, wt / folder, folder)
    return linked


def snapshot(wt: Path, linked: list[str]) -> dict[str, tuple]:
    """{path: (kind, size, mtime_ns)} for everything reachable through the `linked` paths."""
    out = {}
    for rel in linked:
        target = (wt / rel).resolve()
        if not target.is_dir():
            st = target.stat()
            out[rel] = ("file", st.st_size, st.st_mtime_ns)
            continue
        for dirpath, dirnames, filenames in os.walk(target):
            base = Path(dirpath)
            sub = base.relative_to(target).as_posix()
            key = rel if sub == "." else f"{rel}/{sub}"
            out[key] = ("dir", 0, base.stat().st_mtime_ns)  # a dir's mtime moves when an entry is added or removed
            for name in filenames + [d for d in dirnames if (base / d).is_symlink()]:
                st = (base / name).lstat()
                out[f"{key}/{name}"] = ("file", st.st_size, st.st_mtime_ns)
    return out


@dataclass
class Worktree:
    path: Path
    commit: str  # full hash of the commit checked out
    copied: list[str] = field(default_factory=list)  # uncommitted files copied in (or deleted)
    linked: list[str] = field(default_factory=list)  # data paths symlinked in
    _before: dict = field(default_factory=dict, repr=False)

    @property
    def short(self) -> str:
        return self.commit[:7]

    def data_changes(self) -> list[str]:
        """Linked data paths added, changed or removed since the worktree was made."""
        after = snapshot(self.path, self.linked)
        return sorted(k for k in self._before.keys() | after.keys() if self._before.get(k) != after.get(k))


def _copy_uncommitted(repo: Path, wt: Path) -> list[str]:
    changed = [line.split("\t", 1) for line in
               git(repo, "diff", "--name-status", "--no-renames", "HEAD").splitlines()]
    untracked = [["?", rel] for rel in git(repo, "ls-files", "--others", "--exclude-standard").splitlines()]
    for status, rel in changed + untracked:
        if status == "D":
            (wt / rel).unlink()
        else:
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, wt / rel, follow_symlinks=False)
    return [rel for _, rel in changed + untracked]


@contextlib.contextmanager
def worktree(repo: Path, rev: str = "HEAD", *, copy_uncommitted: bool = False, folders=DATA_FOLDERS):
    """A Worktree of `rev` (detached, outside `repo`, data linked in), removed on exit.

    Raises WorktreeError when `rev` is not a commit or git cannot make the worktree.
    """
    tmp = Path(tempfile.mkdtemp(prefix="imagelynx_wt_"))
    path = tmp / "wt"
    try:
        git(repo, "worktree", "add", "-q", "--detach", str(path), rev)
    except WorktreeError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    wt = Worktree(path, "")
    try:
        wt.commit = git(path, "rev-parse", "HEAD").strip()
        if copy_uncommitted:
            wt.copied = _copy_uncommitted(repo, path)
        wt.linked = link_data(repo, path, folders)
        wt._before = snapshot(path, wt.linked)
        yield wt
    finally:
        # Links go first, so nothing below can reach the data through them.
        for rel in wt.linked:
            with contextlib.suppress(OSError):
                (path / rel).unlink()
        # Forced twice, remove also takes a locked worktree. Its failure is not raised: raising
        # here would hide an error already on its way out, and rmtree and prune finish the job.
        subprocess.run(["git", "worktree", "remove", "--force", "--force", str(path)], cwd=repo,
                       capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        git(repo, "worktree", "prune")
