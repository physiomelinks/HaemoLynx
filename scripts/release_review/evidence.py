"""Checking that the files a row cites as evidence still exist.

A check passes only with evidence in its row, and evidence that has gone is
no evidence. On 2026-10-24 rows marked passing cited output folders that had
been deleted with the worktree that made them (``outputs/p6/o2_current/``,
``outputs/release_review/2026-10-15_9b679b8/``), and nothing noticed. This
reads each row's backticked paths and looks for them:

- source, test and script paths (``tests/test_x.py:12``, ``pipeline/checks.py``,
  a bare ``test_x.py``) in the tree being scored;
- ``outputs/...`` paths, which are git-ignored and local, under each output
  root given (the main checkout and its worktrees). Without one -- in CI --
  they are not checked.

A cited line past the end of its file is reported, not failed: lines drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

_BACKTICKED = re.compile(r"`([^`]+)`")
_FILE_EXTENSIONS = (
    "py", "md", "json", "log", "csv", "txt", "yaml", "yml", "toml", "pkl", "npy",
    "tif", "tiff", "h5", "html", "png", "ps1", "ipynb", "cfg",
)
_PATH = re.compile(
    r"^(?P<path>[\w.\-/]+?\.(?:" + "|".join(_FILE_EXTENSIONS) + r"))(?::(?P<line>\d+)(?:-(?P<end>\d+))?)?$"
)
_DIRECTORY = re.compile(r"^(?P<path>outputs/[\w.\-/]+/)$")
#: Where a path written relative to the package or a test folder may sit.
SOURCE_PREFIXES = ("", "src/haemolynx/", "tests/", "tests/integration/", "scripts/", "examples/")


@dataclass(frozen=True)
class Citation:
    text: str
    path: str
    line: Optional[int] = None
    end: Optional[int] = None

    @property
    def is_output(self) -> bool:
        return self.path.startswith("outputs/")

    @property
    def is_bare(self) -> bool:
        return "/" not in self.path


def citations(evidence: str) -> list[Citation]:
    """The file and folder paths cited in backticks in *evidence*.

    Bare names are kept only for Python files (``test_flow_solvers.py:125``),
    which can be found by name; a bare ``REPORT.md`` names a file beside an
    earlier citation and cannot be placed. Templated names (``{stem}_...``)
    and globs are not paths.
    """
    found = []
    for token in _BACKTICKED.findall(evidence):
        # `test_flow_solvers.py:125, 207-217` cites one file at several lines.
        token = token.split(",")[0].strip()
        if any(mark in token for mark in "{}*<> "):
            continue
        directory = _DIRECTORY.match(token)
        if directory:
            found.append(Citation(token, directory.group("path")))
            continue
        match = _PATH.match(token)
        if not match:
            continue
        path = match.group("path")
        if "/" not in path and not path.endswith(".py"):
            continue
        line = int(match.group("line")) if match.group("line") else None
        end = int(match.group("end")) if match.group("end") else None
        found.append(Citation(token, path, line, end))
    return found


@dataclass(frozen=True)
class Finding:
    check_id: str
    citation: Citation
    problem: str  # "missing" or "past the end"
    detail: str = ""


class Tree:
    """Where cited paths are looked for: the tree being scored, and the
    local output roots (main checkout and worktrees) for ``outputs/``."""

    def __init__(self, root: Path, output_roots: Sequence[Path] = ()) -> None:
        self.root = Path(root)
        self.output_roots = [Path(each) for each in output_roots]
        self._by_name: Optional[dict[str, list[Path]]] = None

    def _names(self) -> dict[str, list[Path]]:
        if self._by_name is None:
            self._by_name = {}
            for folder in ("src", "tests", "scripts", "examples", "tutorials"):
                for path in sorted((self.root / folder).rglob("*.py")):
                    self._by_name.setdefault(path.name, []).append(path)
        return self._by_name

    def placeable(self, citation: Citation) -> bool:
        """Whether the path can be looked for at all. One that starts in a
        folder neither the tree nor the package has (``grid_P/marks.csv``)
        is written relative to a folder named earlier in its row."""
        if citation.is_output or citation.is_bare:
            return True
        first = citation.path.split("/", 1)[0]
        return any((self.root / prefix / first).is_dir() for prefix in ("", "src/haemolynx"))

    def locate(self, citation: Citation) -> list[Path]:
        """Every file or folder the citation may mean: one, or for a bare
        name several (``schema.py`` is in ``parsers/`` and ``pipeline/``)."""
        if citation.is_output:
            return [root / citation.path for root in self.output_roots if (root / citation.path).exists()][:1]
        if citation.is_bare:
            return list(self._names().get(citation.path, []))
        for prefix in SOURCE_PREFIXES:
            candidate = self.root / (prefix + citation.path)
            if candidate.exists():
                return [candidate]
        return []


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def audit(
    rows: Iterable[tuple[str, str]],
    tree: Tree,
    *,
    check_outputs: bool,
    line_count: Callable[[Path], int] = _line_count,
) -> list[Finding]:
    """Each cited path in *rows* (``(check_id, evidence)`` pairs) that is
    missing, or whose cited line is past its end."""
    findings = []
    for check_id, evidence in rows:
        for citation in citations(evidence):
            if (citation.is_output and not check_outputs) or not tree.placeable(citation):
                continue
            located = tree.locate(citation)
            if not located:
                findings.append(Finding(check_id, citation, "missing"))
                continue
            last = citation.end or citation.line
            files = [path for path in located if path.is_file() and path.suffix == ".py"]
            if last is not None and files:
                count = max(line_count(path) for path in files)
                if last > count:
                    findings.append(Finding(check_id, citation, "past the end", f"{files[0].name} has {count} lines"))
    return findings


def missing_by_check(findings: Iterable[Finding]) -> dict[str, list[str]]:
    by_check: dict[str, list[str]] = {}
    for finding in findings:
        if finding.problem == "missing":
            by_check.setdefault(finding.check_id, []).append(finding.citation.path)
    return by_check
