"""Lines in tracked files that point at a local note.

A local note is a markdown file kept on disk beside the code but never tracked, such as a
plan or a ticket. A clone does not have them, so a tracked file that cites one points at
nothing; code, tests and comments state the reason in words instead.
``tests/test_tracked_files_name_no_local_note.py`` runs this on the repo.

``pointers(root)`` returns ``path:line: text`` for every line of a file git tracks in `root`
that does one of four things:

1. names a markdown file that git does not track, by the path as written or by its file name;
2. holds ``OPEN_ITEM`` in any case (how the notes number unresolved items);
3. holds a path into the scratch folder, where tickets and plans are kept;
4. gives an issue number and then an item number in the same clause (``ISSUE_ITEM``): the
   items are numbered in a note, not in the issue. A bare issue number is fine.

Names the tests and the pipeline write themselves (``TOY_NOTES``) are not local notes. Not read
at all: binary files, ``.gitignore`` (it names the local notes on purpose) and ``tests/data/``.

``message_pointers(root, message)`` runs the same rules on a commit message, for the
``commit-msg`` hook.
"""
from __future__ import annotations

import re
from pathlib import Path

from _worktree import git

# Built from pieces, so this file does not match its own rules.
OPEN_ITEM = "open" + " item"
SCRATCH = "." + "scratch/"
ISSUE_ITEM = re.compile("#" + r"\d+[^.;\n]*?\bitem\s+\d+", re.IGNORECASE)
#: .md files the tests and the pipeline write themselves. model_results.md is also spelled in
#: ImageLynx.visualization.reporting, but this script runs on toy repos without the package and
#: must not import it, so the name is written out here.
TOY_NOTES = {"notes.md", "model_results.md"}
# git commit -v puts the staged diff below a line holding this mark.
SCISSORS = ">8"
# A file name ending in .md. A glob such as *.md is not a name, so "*" is left out.
MD_NAME = re.compile(r"[\w./-]+\.md\b")


def pointers(root: Path) -> list[str]:
    """``path:line: text`` for each line of a tracked file in `root` that points at a local note."""
    tracked = _tracked(root)
    names = _known_names(tracked)
    found = []
    for rel in tracked:
        if _not_read(rel) or not (root / rel).is_file():
            continue  # a tracked file deleted on disk has no lines to read
        data = (root / rel).read_bytes()
        if b"\0" in data:
            continue  # binary
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary
        found += [f"{rel}:{number}: {line.strip()}"
                  for number, line in enumerate(text.splitlines(), 1)
                  if _points_at_a_note(line, names)]
    return found


def message_pointers(root: Path, message: str) -> list[str]:
    """Lines of commit message `message`, for repo `root`, that point at a local note.

    Git's comment lines (``#`` then a space, or ``#`` alone) and everything below the scissors
    line of ``commit -v`` are not read: git strips them from the message after the hook runs.
    """
    names = _known_names(_tracked(root))
    found = []
    for line in message.splitlines():
        if line.startswith("# ") and SCISSORS in line:
            break
        if line == "#" or line.startswith(("# ", "#\t")):
            continue
        if _points_at_a_note(line, names):
            found.append(line.strip())
    return found


def _tracked(root: Path) -> list[str]:
    return [rel for rel in git(root, "ls-files", "-z").split("\0") if rel]


def _known_names(tracked: list[str]) -> set[str]:
    return set(tracked) | {Path(rel).name for rel in tracked} | TOY_NOTES


def _points_at_a_note(line: str, known_names: set[str]) -> bool:
    # Rule 1. The regex backtracks over long runs of name characters (base64 in a notebook),
    # so it only runs on lines that could hold a name.
    if ".md" in line and any(name not in known_names and Path(name).name not in known_names
                             for name in MD_NAME.findall(line)):
        return True
    if OPEN_ITEM in line.lower():  # rule 2
        return True
    if SCRATCH in line:  # rule 3
        return True
    return ISSUE_ITEM.search(line) is not None  # rule 4


def _not_read(rel: str) -> bool:
    return rel == ".gitignore" or rel.startswith("tests/data/")
