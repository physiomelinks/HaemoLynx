"""A check's result, and the ladder that turns results into a score.

The rule (``RELEASE_REVIEW.md`` § The ladder rule): an aspect's score is the
highest even level whose checks, and every lower level's, all pass, plus one
if at least half of the next level's checks pass. ``unknown`` scores as
``fail``: missing evidence is never the benefit of the doubt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from .rubric import (
    ASPECT_KEYS,
    CHECKS,
    GATE_EVERY_ASPECT,
    GATE_SCIENCE,
    LEVELS,
    SCIENCE_ASPECTS,
    Check,
)

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
STATUSES = (PASS, FAIL, UNKNOWN)

#: Where a result came from: a collector that measured it in this run, a
#: tester's or reviewer's recorded mark (``human_checks.yaml``), or the
#: check's row in ``RELEASE_REVIEW.md`` carried forward unmeasured.
MEASURED, RECORDED, DOCUMENT = "measured", "recorded", "document"


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    value: Any = None
    target: Any = None
    evidence: str = ""
    commit: Optional[str] = None
    date: Optional[str] = None
    source: str = MEASURED
    #: Why a result is unknown, or what was set aside to reach it.
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: status {self.status!r} is not one of {STATUSES}")

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measured(check_id: str, passed: Optional[bool], value: Any = None, evidence: str = "", *, target: Any = None, note: str = "") -> CheckResult:
    """A collector's result: pass or fail, or unknown when *passed* is None."""
    status = UNKNOWN if passed is None else (PASS if passed else FAIL)
    return CheckResult(check_id, status, value=value, target=target, evidence=evidence, source=MEASURED, note=note)


def aspect_score(statuses_by_level: Mapping[int, Sequence[str]]) -> int:
    """The ladder rule, over each level's check statuses.

    Every level from 2 to 10 must hold at least one check: an empty level
    would pass vacuously and lift whatever sits above it.
    """
    score = 0
    for level in LEVELS:
        statuses = list(statuses_by_level.get(level, ()))
        if not statuses:
            raise ValueError(f"level {level} has no checks")
        passed = sum(status == PASS for status in statuses)
        if passed == len(statuses):
            score = level
            continue
        if 2 * passed >= len(statuses):
            score += 1
        break
    return score


def score_aspects(
    results: Mapping[str, CheckResult], checks: Sequence[Check] = CHECKS
) -> dict[str, int]:
    """Each aspect's score; a check with no result counts as unknown."""
    by_aspect: dict[str, dict[int, list[str]]] = {key: {} for key in ASPECT_KEYS}
    for each in checks:
        status = results[each.id].status if each.id in results else UNKNOWN
        by_aspect.setdefault(each.aspect, {}).setdefault(each.level, []).append(status)
    return {key: aspect_score(levels) for key, levels in by_aspect.items() if levels}


def headline(scores: Mapping[str, int]) -> tuple[int, float]:
    """The lowest score and the mean, rounded to one decimal as the
    scorecard writes it."""
    values = list(scores.values())
    return min(values), round(sum(values) / len(values), 1)


def gate(scores: Mapping[str, int]) -> list[str]:
    """What stands between these scores and the release gate; empty when met."""
    short = []
    for key, value in scores.items():
        needed = GATE_SCIENCE if key in SCIENCE_ASPECTS else GATE_EVERY_ASPECT
        if value < needed:
            short.append(f"{key} {value} < {needed}")
    return short
