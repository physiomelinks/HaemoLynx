"""The rubric is written down twice -- `scripts/release_review/rubric.py` and
`RELEASE_REVIEW.md`'s ladders -- and the two must agree on every check: its
ID, aspect, level, kind and targets. A target changed in one place only, or a
check added to one only, fails here (CLAUDE.md § Release review: never move a
target to make a check pass)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from release_review import review_doc, rubric

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def document():
    return review_doc.read(REPO_ROOT / "RELEASE_REVIEW.md")


def test_every_check_is_in_both_and_nowhere_else(document):
    in_rubric = {check.id for check in rubric.CHECKS}
    in_document = set(document.rows)
    assert sorted(in_document - in_rubric) == [], "rows of RELEASE_REVIEW.md the rubric lacks"
    assert sorted(in_rubric - in_document) == [], "checks of the rubric RELEASE_REVIEW.md lacks"
    assert len(rubric.CHECKS) == len(in_rubric), "a check is listed twice in the rubric"


def test_kind_and_aspect_agree(document):
    for check in rubric.CHECKS:
        row = document.rows[check.id]
        assert (row.kind, row.aspect) == (check.kind, check.aspect), check.id


def test_the_aspects_and_their_order_agree(document):
    assert [(number, key) for number, _title, key in document.aspects] == [
        (aspect.number, aspect.key) for aspect in rubric.ASPECTS
    ]
    assert [title for _number, title, _key in document.aspects] == [aspect.title for aspect in rubric.ASPECTS]
    assert sorted(document.scorecard) == sorted(rubric.ASPECT_KEYS)


def test_every_aspect_has_a_check_at_every_level():
    for aspect in rubric.ASPECT_KEYS:
        levels = {check.level for check in rubric.checks_of(aspect)}
        assert levels == set(rubric.LEVELS), aspect


def test_every_target_is_stated_in_its_row(document):
    for check in rubric.CHECKS:
        for goal in check.targets:
            assert goal.text in document.rows[check.id].passes_when, (check.id, goal.text)


def test_every_proposed_target_is_in_the_rubric(document):
    """A row marking a number *(proposed)* has that number in the rubric, so
    a collector compares against the document's figure, not its own."""
    for row in document.rows.values():
        if "(proposed)" in row.passes_when:
            assert rubric.check(row.id).targets, row.id


def test_percentage_targets_are_fractions_of_the_stated_number():
    for check in rubric.CHECKS:
        for goal in check.targets:
            percent = re.search(r"(\d+(?:\.\d+)?)%", goal.text)
            if percent:
                assert goal.value == pytest.approx(float(percent.group(1)) / 100), check.id


def test_human_checks_name_only_testers_and_the_independent_ones_where_asked():
    for check in rubric.CHECKS:
        assert set(check.testers) <= set(rubric.TESTERS), check.id
        assert 1 <= check.testers_needed <= len(check.testers), check.id
    independent = {"graph.10a", "diameter.10a", "stats.10a", "visual.8a", "usability.8a"}
    for check_id in independent:
        assert set(rubric.check(check_id).testers) == set(rubric.INDEPENDENT), check_id
    assert rubric.check("docs.6a").testers == ("Harvey",)
    assert rubric.check("usability.8a").testers_needed == 2
    assert set(rubric.check("package.6b").testers) == {"Harvey", "Mason", "Finbar"}


def test_only_h_checks_carry_testers():
    for check in rubric.CHECKS:
        if check.kind != rubric.HUMAN:
            assert check.testers == rubric.TESTERS and check.testers_needed == 1, check.id


def test_human_checks_yaml_lists_every_h_check_and_reads_cleanly():
    from release_review import human

    path = REPO_ROOT / "scripts" / "release_review" / "human_checks.yaml"
    listed = set(yaml.safe_load(path.read_text(encoding="utf-8"))["checks"])
    assert {check.id for check in rubric.CHECKS if check.kind == rubric.HUMAN} <= listed
    human.load(path)  # every mark in it valid: a known check, a tester allowed to mark it


def test_the_document_parses_its_own_scorecard_and_log(document):
    assert document.headline is not None and document.mean is not None
    assert all(0 <= row.score <= 10 for row in document.scorecard.values())
    assert document.log and all(re.match(r"\d{4}-\d{2}-\d{2}$", row.date) for row in document.log)
