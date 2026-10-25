"""Deciding a check from what was measured, what is recorded and what its
row says; the testers' marks; reading the review document."""
from __future__ import annotations

import pytest

from release_review import human, review_doc, rubric, scorecard
from release_review.ladder import DOCUMENT, FAIL, MEASURED, PASS, RECORDED, UNKNOWN, CheckResult, measured

DOCUMENT_TEXT = """\
# Review

## Scorecard

| # | Aspect | Score | What holds it there (the next checks to pass) |
|---|---|---|---|
| 1 | [Graph accuracy](#1-graph-accuracy--graph) | **4** | 6b, 6c |

**Headline: the lowest score is 4** (graph).
The mean is 4.0. The release gate needs more.

## The ladders

### 1. Graph accuracy — `graph`

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| graph.2a | A | A graph builds. | ✓ | E14.5: 3,146 edges (`outputs/x/REPORT.md`) |
| graph.4a | R | Steps tested. | ✓ | |
| graph.6b | H | Rules fitted to marks. | ✗ | Harvey marked G1 |
| graph.8a | A | Phantoms ≥ 0.95 *(proposed)*. | ✗ | No phantoms |

## Review log

| Date | Commit | By | What changed |
|---|---|---|---|
| 2026-10-10 | `9c6a15a` | Claude | First scoring. |
"""


@pytest.fixture
def document():
    return review_doc.parse(DOCUMENT_TEXT)


def test_the_document_is_read_row_by_row(document):
    assert list(document.rows) == ["graph.2a", "graph.4a", "graph.6b", "graph.8a"]
    row = document.rows["graph.2a"]
    assert (row.kind, row.passed, row.aspect) == ("A", True, "graph")
    assert row.evidence.startswith("E14.5: 3,146 edges")
    assert document.rows["graph.4a"].evidence == ""
    assert document.rows["graph.8a"].passed is False
    assert document.scorecard["graph"].score == 4
    assert (document.headline, document.mean) == (4, 4.0)
    assert [(row.date, row.by) for row in document.log] == [("2026-10-10", "Claude")]


def test_a_check_with_two_rows_is_refused():
    doubled = DOCUMENT_TEXT.replace("| graph.4a | R |", "| graph.2a | R |")
    with pytest.raises(ValueError, match="graph.2a has two rows"):
        review_doc.parse(doubled)


def test_a_measured_result_wins_over_the_row(document):
    check = rubric.check("graph.2a")
    found = measured("graph.2a", False, evidence="no graph")
    result = scorecard.decide(check, found, [], document.rows["graph.2a"])
    assert (result.status, result.source) == (FAIL, MEASURED)


def test_a_row_passes_on_its_evidence_and_only_with_some(document):
    assert scorecard.decide(rubric.check("graph.2a"), None, [], document.rows["graph.2a"]).status == PASS
    empty = scorecard.decide(rubric.check("graph.4a"), None, [], document.rows["graph.4a"])
    assert (empty.status, empty.source) == (UNKNOWN, DOCUMENT)
    assert "no evidence" in empty.note
    assert scorecard.decide(rubric.check("graph.8a"), None, [], document.rows["graph.8a"]).status == FAIL


def test_a_row_whose_cited_file_is_gone_is_unknown(document):
    result = scorecard.decide(rubric.check("graph.2a"), None, [], document.rows["graph.2a"], ["outputs/x/REPORT.md"])
    assert result.status == UNKNOWN
    assert "outputs/x/REPORT.md" in result.note


def test_an_h_check_passes_only_on_a_testers_mark(document):
    check = rubric.check("graph.6b")
    passing_row = document.rows["graph.6b"].__class__(**{**document.rows["graph.6b"].__dict__, "passed": True})
    assert scorecard.decide(check, None, [], passing_row).status == UNKNOWN
    counted = scorecard.decide(check, measured("graph.6b", True, 30), [], passing_row)
    assert counted.status == UNKNOWN and "only a tester can pass it" in counted.note
    mark = human.Mark("graph.6b", "Harvey", "2026-10-25", PASS, "outputs/dead_end_review/FIT.md")
    assert scorecard.decide(check, None, [mark], passing_row).status == PASS


def test_a_collector_may_fail_an_h_check_whatever_is_recorded():
    """docs.6a: the help pages counted empty fail it, even with a stale mark."""
    check = rubric.check("docs.6a")
    mark = human.Mark("docs.6a", "Harvey", "2026-10-25", PASS, "gui/stage_help.py")
    result = scorecard.decide(check, measured("docs.6a", False, evidence="0 of 10"), [mark], None)
    assert (result.status, result.source) == (FAIL, MEASURED)


def test_the_scorecard_scores_and_lists_disagreements(document):
    results = {"graph.2a": measured("graph.2a", False, evidence="build failed")}
    card = scorecard.build(document, results, {}, commit="abc", date="2026-10-25", tiers=["quick"])
    assert card.results["graph.2a"].status == FAIL
    assert ("graph.2a", PASS, FAIL) in card.disagreements
    assert ("graph.4a", PASS, UNKNOWN) in card.disagreements  # carried, but no evidence
    assert card.scores["graph"] == 0
    assert card.document_scores == {"graph": 4}


# --- the testers' marks -----------------------------------------------------


def _mark(check_id, by, result=PASS, date="2026-11-01"):
    return {"tester": by, "date": date, "result": result, "evidence": "sheet kept at X"}


def test_marks_are_validated_all_at_once():
    data = {"checks": {
        "graph.99z": [_mark("graph.99z", "Harvey")],
        "docs.6a": [_mark("docs.6a", "Finbar")],
        "visual.8a": [_mark("visual.8a", "Mason")],
        "graph.6b": [{"tester": "Harvey", "result": "pass"}],
        "speed.6c": [_mark("speed.6c", "Harvey", result="maybe")],
    }}
    with pytest.raises(ValueError) as raised:
        human.parse(data)
    message = str(raised.value)
    for expected in ("graph.99z: not a check", "docs.6a: Finbar may not mark it", "visual.8a: Mason may not",
                     "graph.6b: a mark without date, evidence", "speed.6c: result 'maybe'"):
        assert expected in message


def test_an_h_check_needs_as_many_testers_as_it_asks():
    marks = human.parse({"checks": {"usability.8a": [_mark("usability.8a", "George")]}})["usability.8a"]
    one = human.decide("usability.8a", marks)
    assert one.status == UNKNOWN and "1 of the 2" in one.note
    marks += human.parse({"checks": {"usability.8a": [_mark("usability.8a", "Rebecca")]}})["usability.8a"]
    two = human.decide("usability.8a", marks)
    assert (two.status, two.source) == (PASS, RECORDED)
    assert "George" in two.evidence and "Rebecca" in two.evidence


def test_a_testers_latest_mark_counts_and_a_fail_fails_while_too_few_pass():
    marks = human.parse({"checks": {"package.6b": [
        _mark("package.6b", "Harvey", FAIL, "2026-11-01"),
        _mark("package.6b", "Harvey", PASS, "2026-11-03"),
        _mark("package.6b", "Mason", FAIL, "2026-11-02"),
    ]}})["package.6b"]
    result = human.decide("package.6b", marks)
    assert result.status == FAIL and "Mason" in result.note


def test_an_r_check_takes_its_latest_review():
    marks = human.parse({"checks": {"tests.4b": [
        {"reviewer": "Claude", "date": "2026-10-20", "result": "fail", "evidence": "a:1"},
        {"reviewer": "Claude", "date": "2026-10-25", "result": "pass", "evidence": "b:2"},
    ]}})["tests.4b"]
    result = human.decide("tests.4b", marks)
    assert (result.status, result.evidence) == (PASS, "b:2")


def test_no_marks_decide_nothing():
    assert human.decide("graph.6b", []) is None
    assert human.parse(None) == {}
    assert human.parse({"checks": {"graph.6b": []}}) == {}


def test_a_result_of_any_source_round_trips_as_a_dict():
    result = CheckResult("graph.2a", PASS, value={"edges": 3}, evidence="x", source=DOCUMENT)
    assert CheckResult(**result.as_dict()) == result
