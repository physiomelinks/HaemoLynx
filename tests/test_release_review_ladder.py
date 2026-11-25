"""The ladder rule (`RELEASE_REVIEW.md` § The ladder rule), exhaustively."""
from __future__ import annotations

import itertools

import pytest

from release_review import ladder, rubric
from release_review.ladder import FAIL, PASS, UNKNOWN, CheckResult, aspect_score


def _levels(**per_level):
    return {int(name[1:]): statuses for name, statuses in per_level.items()}


def test_every_check_passing_scores_ten():
    assert aspect_score(_levels(l2=[PASS], l4=[PASS, PASS], l6=[PASS], l8=[PASS], l10=[PASS])) == 10


def test_a_failing_level_two_scores_zero_or_one_at_half():
    assert aspect_score(_levels(l2=[FAIL], l4=[PASS], l6=[PASS], l8=[PASS], l10=[PASS])) == 0
    assert aspect_score(_levels(l2=[PASS, FAIL], l4=[PASS], l6=[PASS], l8=[PASS], l10=[PASS])) == 1
    assert aspect_score(_levels(l2=[PASS, FAIL, FAIL], l4=[PASS], l6=[PASS], l8=[PASS], l10=[PASS])) == 0


def test_half_the_next_level_adds_one():
    assert aspect_score(_levels(l2=[PASS], l4=[PASS], l6=[PASS, FAIL], l8=[FAIL], l10=[FAIL])) == 5
    assert aspect_score(_levels(l2=[PASS], l4=[PASS], l6=[PASS, FAIL, FAIL], l8=[FAIL], l10=[FAIL])) == 4
    assert aspect_score(_levels(l2=[PASS], l4=[PASS, PASS, FAIL], l6=[FAIL], l8=[FAIL], l10=[FAIL])) == 3


def test_unknown_scores_as_fail():
    assert aspect_score(_levels(l2=[PASS], l4=[UNKNOWN], l6=[PASS], l8=[PASS], l10=[PASS])) == 2
    assert aspect_score(_levels(l2=[UNKNOWN], l4=[PASS], l6=[PASS], l8=[PASS], l10=[PASS])) == 0


def test_a_passing_high_check_cannot_lift_a_failing_low_one():
    """A level-8 result never carries an aspect with a level-4 check failing."""
    capped = aspect_score(_levels(l2=[PASS], l4=[PASS, FAIL, FAIL], l6=[PASS, PASS], l8=[PASS, PASS], l10=[PASS]))
    assert capped == 2


def test_the_rule_over_every_combination_of_one_check_a_level():
    """With one check a level, the score is the last level of an unbroken run
    of passes from 2, and nothing above a fail counts (a single check's
    half is itself)."""
    for outcome in itertools.product([PASS, FAIL], repeat=5):
        levels = dict(zip(rubric.LEVELS, ([status] for status in outcome)))
        run = 0
        for level, status in zip(rubric.LEVELS, outcome):
            if status != PASS:
                break
            run = level
        assert aspect_score(levels) == run, outcome


def test_the_rule_over_every_combination_of_two_checks_a_level():
    for outcome in itertools.product([PASS, FAIL], repeat=10):
        levels = {level: list(outcome[2 * i : 2 * i + 2]) for i, level in enumerate(rubric.LEVELS)}
        expected = 0
        for level in rubric.LEVELS:
            passed = levels[level].count(PASS)
            if passed == 2:
                expected = level
                continue
            expected += 1 if passed == 1 else 0
            break
        assert aspect_score(levels) == expected, outcome


def test_an_empty_level_is_refused():
    with pytest.raises(ValueError, match="level 6"):
        aspect_score(_levels(l2=[PASS], l4=[PASS], l6=[], l8=[PASS], l10=[PASS]))


def test_scores_of_every_aspect_and_a_missing_result_counts_as_unknown():
    results = {check.id: CheckResult(check.id, PASS) for check in rubric.CHECKS}
    assert set(ladder.score_aspects(results).values()) == {10}
    del results["graph.4a"]
    scores = ladder.score_aspects(results)
    assert scores["graph"] == 3  # 4a, 4b, 4c: two of three pass
    assert scores["diameter"] == 10


def test_headline_and_gate():
    scores = {key: 8 for key in rubric.ASPECT_KEYS}
    assert ladder.gate(scores) == []
    scores.update(graph=7, usability=7)
    assert ladder.gate(scores) == ["graph 7 < 8"]
    scores["code"] = 6
    assert ladder.gate(scores) == ["graph 7 < 8", "code 6 < 7"]
    assert ladder.headline({"a": 4, "b": 4, "c": 5, "d": 5}) == (4, 4.5)
    # The first scoring: sixteen aspects summing to 70 is a mean of 4.4.
    assert ladder.headline({str(i): value for i, value in enumerate([4, 4, 4, 7, 6, 4, 4, 4, 4, 6, 4, 4, 5, 3, 4, 3])}) == (3, 4.4)


def test_a_result_with_an_unknown_status_is_refused():
    with pytest.raises(ValueError):
        CheckResult("graph.2a", "maybe")
    assert ladder.measured("graph.2a", None).status == UNKNOWN
    assert ladder.measured("graph.2a", True).status == PASS
