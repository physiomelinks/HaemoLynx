"""Reading ``RELEASE_REVIEW.md``: its ladders, its scorecard and its log.

The document is the second copy of the rubric (``rubric.py`` is the first) and
the record of every check's evidence. The scorer reads each row's mark (✓ or
✗) and its evidence, so a check nobody can re-measure still counts what its
row says, and the citations in that evidence can be checked for files that no
longer exist (:mod:`.evidence`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PASSED_MARK, FAILED_MARK = "✓", "✗"

_ASPECT_HEADING = re.compile(r"^### (\d+)\. (.+?) — `(\w+)`\s*$")
_LADDER_ROW = re.compile(
    r"^\| (?P<id>[a-z]+\.\d+[a-z]) \| (?P<kind>[ARH]) \| (?P<passes>.*?) \| "
    r"(?P<mark>[✓✗]) \| (?P<evidence>.*?)\s*\|\s*$"
)
_SCORECARD_ROW = re.compile(
    r"^\| (?P<number>\d+) \| \[(?P<title>[^\]]+)\]\(#[^)]*--(?P<key>\w+)\) \| "
    r"\*\*(?P<score>\d+)\*\* \| (?P<holds>.*?)\s*\|\s*$"
)
_LOG_ROW = re.compile(
    r"^\| (?P<date>\d{4}-\d{2}-\d{2}) \| (?P<commit>[^|]*?) \| (?P<by>[^|]*?) \| (?P<what>.*?)\s*\|\s*$"
)
_HEADLINE = re.compile(r"\*\*Headline: the lowest score is (\d+)\*\*")
_MEAN = re.compile(r"The mean is (\d+(?:\.\d+)?)\.")


@dataclass(frozen=True)
class LadderRow:
    """One check's row: what passes it, its mark at the last review, and the
    evidence or gap written beside it."""

    id: str
    kind: str
    passes_when: str
    passed: bool
    evidence: str
    line: int
    aspect: str


@dataclass(frozen=True)
class ScorecardRow:
    number: int
    title: str
    key: str
    score: int
    holds: str


@dataclass(frozen=True)
class LogRow:
    date: str
    commit: str
    by: str
    what: str


@dataclass
class ReviewDocument:
    rows: dict[str, LadderRow] = field(default_factory=dict)
    #: (number, title, key) for each aspect heading, in order.
    aspects: list[tuple[int, str, str]] = field(default_factory=list)
    scorecard: dict[str, ScorecardRow] = field(default_factory=dict)
    log: list[LogRow] = field(default_factory=list)
    headline: Optional[int] = None
    mean: Optional[float] = None


def parse(text: str) -> ReviewDocument:
    """Every ladder row, scorecard row and log row of the review's text.

    A check ID appearing in two rows raises: the document would then say two
    things about one check.
    """
    document = ReviewDocument()
    aspect = None
    in_log = False
    for number, line in enumerate(text.splitlines(), start=1):
        heading = _ASPECT_HEADING.match(line)
        if heading:
            aspect = heading.group(3)
            document.aspects.append((int(heading.group(1)), heading.group(2), aspect))
            continue
        if line.startswith("## "):
            in_log = line.strip() == "## Review log"
            aspect = None
            continue
        row = _LADDER_ROW.match(line)
        if row and aspect is not None:
            check_id = row.group("id")
            if check_id in document.rows:
                raise ValueError(f"{check_id} has two rows (lines {document.rows[check_id].line} and {number})")
            document.rows[check_id] = LadderRow(
                id=check_id,
                kind=row.group("kind"),
                passes_when=row.group("passes"),
                passed=row.group("mark") == PASSED_MARK,
                evidence=row.group("evidence").strip(),
                line=number,
                aspect=aspect,
            )
            continue
        card = _SCORECARD_ROW.match(line)
        if card:
            key = card.group("key")
            document.scorecard[key] = ScorecardRow(
                number=int(card.group("number")),
                title=card.group("title"),
                key=key,
                score=int(card.group("score")),
                holds=card.group("holds"),
            )
            continue
        if in_log:
            logged = _LOG_ROW.match(line)
            if logged:
                document.log.append(LogRow(**{name: logged.group(name).strip() for name in ("date", "commit", "by", "what")}))
            continue
        headline = _HEADLINE.search(line)
        if headline:
            document.headline = int(headline.group(1))
        mean = _MEAN.search(line)
        if mean and document.mean is None:
            document.mean = float(mean.group(1))
    return document


def read(path: Path) -> ReviewDocument:
    return parse(Path(path).read_text(encoding="utf-8"))
