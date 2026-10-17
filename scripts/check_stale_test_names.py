"""List test names that documents cite but tests/ no longer defines.

    venv/bin/python scripts/check_stale_test_names.py [--root DIR]

Two sections, never mixed:

1. The documents CLAUDE.md lists under "Key references" (bullets only; a `*` in a name is a glob).
   The list is read from CLAUDE.md each run, so it cannot drift from it.
2. Every other tracked `.md` file.

A cited `test_x` counts as alive when `def test_x` appears in any file under tests/. A cited
`test_x.py` is checked against the test file names instead. Each hit prints `path:line:` and
the citing line. A hit on a line that says "renamed" (or whose next line does) is a rename
note, not a dead citation: it is printed apart and left out of the count.

A Key references document that is not on disk (a gitignored one, in a fresh worktree) is listed
as "not present", never skipped silently. Point `--root` at the checkout that has them.

Exit 0 when the report ran (a report, not a gate); 2 when CLAUDE.md has no Key references list.
"""
import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"\btest_\w+(?:\.py)?")
DEF_RE = re.compile(r"def (test_\w+)")


@dataclass(frozen=True)
class Hit:
    lineno: int
    name: str
    line: str
    rename_note: bool


def key_reference_patterns(claude_md):
    """The `.md` names in the bullets of CLAUDE.md's "Key references" section, in order."""
    lines = claude_md.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "## Key references")
    except StopIteration:
        raise ValueError("CLAUDE.md has no '## Key references' section") from None
    patterns = []
    for ln in lines[start + 1:]:
        if ln.startswith(("## ", "### ")):
            break
        if ln.startswith("- "):
            for token in re.findall(r"`([^`]+)`", ln):
                if token.endswith(".md") and token not in patterns:
                    patterns.append(token)
    if not patterns:
        raise ValueError("the '## Key references' section of CLAUDE.md names no .md file")
    return patterns


def expand_patterns(patterns, root):
    """Expand glob patterns against root (sorted); a plain name stays whether or not it exists."""
    out = []
    for pat in patterns:
        found = sorted(str(p.relative_to(root)) for p in Path(root).glob(pat)) if "*" in pat else []
        found = found or [pat]  # a pattern that matches nothing is reported as absent, not dropped
        for name in found:
            if name not in out:
                out.append(name)
    return out


def tests_inventory(root):
    """(names defined by `def test_...` anywhere under tests/, names of the .py files there)."""
    defined, files = set(), set()
    for p in (Path(root) / "tests").rglob("*.py"):
        files.add(p.name)
        defined |= set(DEF_RE.findall(p.read_text()))
    return defined, files


def find_stale(text, defined, files):
    """Every cited test name that tests/ does not have, one Hit per citation."""
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        for name in NAME_RE.findall(line):
            if name.endswith(".py"):
                alive = name in files
            elif name.endswith("_"):  # `test_x_*` written as a family
                alive = any(d.startswith(name) for d in defined)
            else:
                alive = name in defined
            if alive:
                continue
            nearby = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
            hits.append(Hit(i + 1, name, line.strip(), "renamed" in nearby))
    return hits


def tracked_markdown(root):
    # A pre-commit hook sets GIT_INDEX_FILE and the like; they would point git at the wrong repo.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=root, env=env, capture_output=True, text=True, check=True)
    return out.stdout.split()


def _scan(docs, root, defined, files):
    lines, dead, notes, missing = [], 0, 0, 0
    for doc in docs:
        path = Path(root) / doc
        if not path.exists():
            lines.append(f"(not present: {doc})")
            missing += 1
            continue
        for h in find_stale(path.read_text(), defined, files):
            if h.rename_note:
                notes += 1
                lines.append(f"{doc}:{h.lineno}: [rename note] {h.name}")
            else:
                dead += 1
                lines.append(f"{doc}:{h.lineno}: {h.line}")
                lines.append(f"    stale: {h.name}")
    return lines, dead, notes, missing


def report(root):
    root = Path(root)
    keys = expand_patterns(key_reference_patterns((root / "CLAUDE.md").read_text()), root)
    defined, files = tests_inventory(root)
    others = [d for d in tracked_markdown(root) if d not in keys]
    out = []
    for title, docs in (("Key references (CLAUDE.md)", keys), ("Other tracked documents", others)):
        lines, dead, notes, missing = _scan(docs, root, defined, files)
        out.append(f"== {title}: {len(docs)} document(s) ==")
        out += lines
        out.append(f"{dead} stale citation(s), {notes} rename note(s), {missing} document(s) not present")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO), help="checkout to scan (default: this one)")
    args = ap.parse_args(argv)
    try:
        print(report(args.root))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
