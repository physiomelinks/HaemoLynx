"""Deciding each check from what this run measured, what is recorded and
what its row says, and scoring the result.

A check's result comes from, in order:

1. a collector that measured it in this run (:data:`~.ladder.MEASURED`);
2. a recorded mark in ``human_checks.yaml`` (:data:`~.ladder.RECORDED`);
3. its row in ``RELEASE_REVIEW.md``, carried forward unmeasured
   (:data:`~.ladder.DOCUMENT`).

Two rules hold whatever the source. An H check passes only on a named
tester's recorded mark: a collector may fail one (the help pages counted
empty) but never pass it, and its row's ✓ is not a mark. And a row's ✓
carries forward only with evidence: an empty evidence cell, or a cited file
that no longer exists, makes it unknown, which scores as a fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from . import human
from .ladder import DOCUMENT, FAIL, PASS, UNKNOWN, CheckResult, gate, headline, score_aspects
from .review_doc import LadderRow, ReviewDocument
from .rubric import CHECKS, HUMAN, Check


def decide(
    check: Check,
    measured: Optional[CheckResult],
    marks: Sequence[human.Mark],
    row: Optional[LadderRow],
    missing_citations: Sequence[str] = (),
) -> CheckResult:
    if check.kind == HUMAN:
        recorded = human.decide(check.id, list(marks))
        if measured is not None and measured.status == FAIL:
            return measured
        if recorded is not None:
            return recorded
        note = "no tester's mark recorded"
        if measured is not None and measured.status == PASS:
            note += f"; measured {measured.value!r}, but only a tester can pass it"
        return CheckResult(check.id, UNKNOWN, evidence=row.evidence if row else "", source=DOCUMENT, note=note)
    if measured is not None:
        return measured
    recorded = human.decide(check.id, list(marks))
    if recorded is not None:
        return recorded
    if row is None:
        return CheckResult(check.id, UNKNOWN, source=DOCUMENT, note="no row in RELEASE_REVIEW.md")
    if not row.passed:
        return CheckResult(check.id, FAIL, evidence=row.evidence, source=DOCUMENT)
    if not row.evidence:
        return CheckResult(check.id, UNKNOWN, source=DOCUMENT, note="marked passing with no evidence in its row")
    if missing_citations:
        return CheckResult(
            check.id, UNKNOWN, evidence=row.evidence, source=DOCUMENT,
            note=f"cited evidence missing: {', '.join(missing_citations)}",
        )
    return CheckResult(check.id, PASS, evidence=row.evidence, source=DOCUMENT)


@dataclass
class Scorecard:
    commit: str
    date: str
    tiers: tuple[str, ...]
    results: dict[str, CheckResult]
    scores: dict[str, int]
    document_scores: dict[str, int] = field(default_factory=dict)
    #: Checks whose result here differs from their row's mark: (id, row says, result).
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)
    lowest: int = 0
    mean: float = 0.0
    gate_short: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build(
    document: ReviewDocument,
    measured: Mapping[str, CheckResult],
    marks: Mapping[str, Sequence[human.Mark]],
    *,
    commit: str,
    date: str,
    tiers: Sequence[str],
    missing_citations: Optional[Mapping[str, Sequence[str]]] = None,
    checks: Sequence[Check] = CHECKS,
) -> Scorecard:
    missing_citations = missing_citations or {}
    results = {}
    disagreements = []
    for each in checks:
        row = document.rows.get(each.id)
        result = decide(each, measured.get(each.id), marks.get(each.id, ()), row, missing_citations.get(each.id, ()))
        results[each.id] = result
        if row is not None and result.source != DOCUMENT:
            said = PASS if row.passed else FAIL
            if result.status != said:
                disagreements.append((each.id, said, result.status))
        elif row is not None and row.passed and result.status != PASS:
            disagreements.append((each.id, PASS, result.status))
    scores = score_aspects(results, checks)
    lowest, mean = headline(scores)
    return Scorecard(
        commit=commit,
        date=date,
        tiers=tuple(tiers),
        results=results,
        scores=scores,
        document_scores={key: row.score for key, row in document.scorecard.items()},
        disagreements=disagreements,
        lowest=lowest,
        mean=mean,
        gate_short=gate(scores),
    )
