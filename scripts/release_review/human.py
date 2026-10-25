"""The recorded marks: ``human_checks.yaml``.

Only one of the named testers can pass an H check (``RELEASE_REVIEW.md``
§ Testers), and some need an independent one, or several. Each mark names
who, when, the result and where its evidence is kept. The scorer reads the
file and never writes it; the testers' results go in by hand, and an agent may
copy a tester's result in with their name, never one of its own.

An R check may be recorded here too, by whoever reviewed it, with the file
and line that decided it. Most R checks keep their evidence in their row of
``RELEASE_REVIEW.md`` instead, which the scorer reads when nothing is
recorded here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

from .ladder import FAIL, PASS, RECORDED, UNKNOWN, CheckResult
from .rubric import CHECKS_BY_ID, HUMAN

RESULTS = (PASS, FAIL)


@dataclass(frozen=True)
class Mark:
    check_id: str
    by: str
    date: str
    result: str
    evidence: str
    value: Any = None


def _date_text(value: Any) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def parse(data: Optional[Mapping[str, Any]]) -> dict[str, list[Mark]]:
    """The marks in a loaded ``human_checks.yaml``, by check.

    Raises ValueError listing every problem at once: a check the rubric does
    not have, a result other than pass or fail, a mark with no name, date or
    evidence, or an H check marked by someone who may not mark it.
    """
    marks: dict[str, list[Mark]] = {}
    problems = []
    checks = (data or {}).get("checks") or {}
    for check_id, entries in checks.items():
        rubric_check = CHECKS_BY_ID.get(check_id)
        if rubric_check is None:
            problems.append(f"{check_id}: not a check of the rubric")
            continue
        for entry in entries or ():
            by = entry.get("tester") or entry.get("reviewer")
            result = entry.get("result")
            missing = [name for name, value in (("tester or reviewer", by), ("date", entry.get("date")), ("evidence", entry.get("evidence"))) if not value]
            if missing:
                problems.append(f"{check_id}: a mark without {', '.join(missing)}")
                continue
            if result not in RESULTS:
                problems.append(f"{check_id}: result {result!r} is not pass or fail")
                continue
            if rubric_check.kind == HUMAN and by not in rubric_check.testers:
                problems.append(
                    f"{check_id}: {by} may not mark it (its testers: {', '.join(rubric_check.testers)})"
                )
                continue
            marks.setdefault(check_id, []).append(
                Mark(check_id, str(by), _date_text(entry["date"]), result, str(entry["evidence"]), entry.get("value"))
            )
    if problems:
        raise ValueError("human_checks.yaml: " + "; ".join(problems))
    return marks


def load(path: Path) -> dict[str, list[Mark]]:
    import yaml

    path = Path(path)
    if not path.is_file():
        return {}
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")))


def decide(check_id: str, marks: list[Mark]) -> Optional[CheckResult]:
    """The check's result from its marks, or None when it has none.

    An H check passes when enough distinct testers (``testers_needed``) have
    each passed it; while too few have, it fails if any has failed it and is
    unknown otherwise. Any other check takes its latest mark.
    """
    if not marks:
        return None
    rubric_check = CHECKS_BY_ID[check_id]
    ordered = sorted(marks, key=lambda mark: mark.date)
    latest = ordered[-1]
    if rubric_check.kind != HUMAN:
        return CheckResult(
            check_id, latest.result, value=latest.value, evidence=latest.evidence,
            date=latest.date, source=RECORDED, note=f"reviewed by {latest.by}",
        )
    last_by_tester = {mark.by: mark for mark in ordered}
    passed = sorted(name for name, mark in last_by_tester.items() if mark.result == PASS)
    failed = sorted(name for name, mark in last_by_tester.items() if mark.result == FAIL)
    evidence = "; ".join(f"{mark.by} {mark.date} {mark.result}: {mark.evidence}" for mark in last_by_tester.values())
    if len(passed) >= rubric_check.testers_needed:
        status, note = PASS, f"passed by {', '.join(passed)}"
    elif failed:
        status, note = FAIL, f"failed by {', '.join(failed)}"
    else:
        status = UNKNOWN
        note = f"{len(passed)} of the {rubric_check.testers_needed} testers it needs"
    return CheckResult(
        check_id, status, value=[last_by_tester[name].value for name in passed] or None,
        evidence=evidence, date=latest.date, source=RECORDED, note=note,
    )
