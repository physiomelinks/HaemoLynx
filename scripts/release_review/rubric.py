"""Every check of the release review, as data.

The rubric is written down twice: here, and in ``RELEASE_REVIEW.md``'s
ladders. ``tests/test_release_review_rubric.py`` parses the document and fails
when the two disagree on a check's ID, level, kind or target, so neither can
move without the other. A target changes only with a line in the review log
saying why (``CLAUDE.md`` § Release review: never move a target to make a
check pass).

A check's ``tier`` names the scorer tier whose collector decides it
(:data:`QUICK`, :data:`FIXTURES`, :data:`FULL`). A check with no tier is
decided by a recorded mark (``human_checks.yaml``) or, failing that, by its
row in the document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LEVELS = (2, 4, 6, 8, 10)

#: Kinds of check: a script or test decides it, someone reviews the code or
#: docs and records the file and line, or one of the named testers does it.
AUTOMATED, REVIEWED, HUMAN = "A", "R", "H"
KINDS = (AUTOMATED, REVIEWED, HUMAN)

QUICK, FIXTURES, FULL = "quick", "fixtures", "full"
TIERS = (QUICK, FIXTURES, FULL)

#: The six testers Harvey named on 2026-10-10 (``RELEASE_REVIEW.md`` § Testers).
DEVELOPERS = ("Harvey", "Finbar", "Mason")
INDEPENDENT = ("George", "Rebecca", "Mike")
TESTERS = DEVELOPERS + INDEPENDENT

#: The release gate: every aspect at 7 or more, the science at 8 or more.
GATE_EVERY_ASPECT = 7
GATE_SCIENCE = 8
SCIENCE_ASPECTS = ("graph", "diameter", "flow", "stats")


@dataclass(frozen=True)
class Aspect:
    number: int
    key: str
    title: str


ASPECTS = (
    Aspect(1, "graph", "Graph accuracy"),
    Aspect(2, "diameter", "Diameter accuracy"),
    Aspect(3, "flow", "Haemodynamic accuracy"),
    Aspect(4, "stats", "Statistics"),
    Aspect(5, "input", "Inputs and segmentation"),
    Aspect(6, "speed", "Speed"),
    Aspect(7, "memory", "Memory and scale"),
    Aspect(8, "usability", "Usability"),
    Aspect(9, "visual", "Visualisation"),
    Aspect(10, "export", "Exports"),
    Aspect(11, "docs", "Documentation and help"),
    Aspect(12, "robust", "Robustness"),
    Aspect(13, "repro", "Reproducibility"),
    Aspect(14, "tests", "Testing and CI"),
    Aspect(15, "package", "Installation and packaging"),
    Aspect(16, "code", "Code maintainability"),
)
ASPECT_KEYS = tuple(aspect.key for aspect in ASPECTS)


@dataclass(frozen=True)
class Target:
    """A number a check is held to: ``text`` exactly as the document's
    "Passes when" cell states it, ``value`` as a collector compares with it
    (fractions for percentages, seconds for times)."""

    text: str
    value: float


@dataclass(frozen=True)
class Check:
    id: str
    kind: str
    tier: Optional[str] = None
    targets: tuple[Target, ...] = ()
    #: H checks: who may mark it, and how many of them must.
    testers: tuple[str, ...] = TESTERS
    testers_needed: int = 1

    @property
    def aspect(self) -> str:
        return self.id.split(".", 1)[0]

    @property
    def level(self) -> int:
        return int(self.id.split(".", 1)[1][:-1])


def _h(check_id: str, testers: tuple[str, ...] = TESTERS, needed: int = 1, *targets: Target) -> Check:
    return Check(check_id, HUMAN, None, tuple(targets), testers, needed)


CHECKS: tuple[Check, ...] = (
    # 1. Graph accuracy
    Check("graph.2a", AUTOMATED, FIXTURES),
    Check("graph.4a", REVIEWED),
    Check("graph.4b", AUTOMATED),
    Check("graph.4c", AUTOMATED, targets=(Target("5%", 0.05),)),
    Check("graph.6a", AUTOMATED, FULL),
    _h("graph.6b"),
    Check("graph.6c", REVIEWED),
    Check("graph.6d", REVIEWED),
    Check("graph.8a", AUTOMATED, FIXTURES, (Target("≥ 0.95", 0.95), Target("5%", 0.05))),
    _h("graph.8b", TESTERS, 1, Target("≥ 90%", 0.90)),
    Check("graph.8c", AUTOMATED),
    _h("graph.10a", INDEPENDENT, 1, Target("≥ 0.9", 0.9)),
    Check("graph.10b", AUTOMATED),
    # 2. Diameter accuracy
    Check("diameter.2a", AUTOMATED, FIXTURES),
    Check("diameter.4a", AUTOMATED),
    Check("diameter.4b", AUTOMATED),
    Check("diameter.6a", AUTOMATED),
    Check("diameter.6b", AUTOMATED),
    Check("diameter.8a", AUTOMATED, targets=(Target("≤ 10%", 0.10),)),
    _h("diameter.8b", TESTERS, 1, Target("30", 30), Target("15%", 0.15)),
    _h("diameter.10a", INDEPENDENT),
    # 3. Haemodynamic accuracy
    Check("flow.2a", AUTOMATED, FIXTURES),
    Check("flow.4a", AUTOMATED),
    Check("flow.4b", REVIEWED),
    Check("flow.6a", AUTOMATED),
    Check("flow.6b", AUTOMATED, FIXTURES),
    Check("flow.6c", AUTOMATED, FULL),
    Check("flow.8a", AUTOMATED, targets=(Target("1%", 0.01),)),
    Check("flow.8b", AUTOMATED),
    _h("flow.10a"),
    # 4. Statistics
    Check("stats.2a", AUTOMATED, FIXTURES),
    Check("stats.4a", AUTOMATED),
    Check("stats.4b", AUTOMATED),
    Check("stats.6a", REVIEWED),
    Check("stats.6b", AUTOMATED),
    Check("stats.6c", AUTOMATED),
    Check("stats.6d", AUTOMATED),
    Check("stats.8a", AUTOMATED),
    Check("stats.8b", AUTOMATED),
    _h("stats.10a", INDEPENDENT),
    # 5. Inputs and segmentation
    Check("input.2a", AUTOMATED, FIXTURES),
    Check("input.4a", AUTOMATED),
    Check("input.4b", AUTOMATED),
    Check("input.6a", AUTOMATED),
    Check("input.6b", AUTOMATED),
    Check("input.6c", REVIEWED),
    Check("input.8a", AUTOMATED),
    Check("input.8b", REVIEWED),
    _h("input.10a"),
    # 6. Speed
    Check("speed.2a", AUTOMATED, FULL),
    Check("speed.4a", AUTOMATED),
    Check("speed.4b", REVIEWED),
    Check("speed.4c", REVIEWED),
    Check("speed.6a", AUTOMATED, FULL),
    Check("speed.6b", REVIEWED, targets=(Target("2 min", 120), Target("20 min", 1200))),
    _h("speed.6c"),
    Check("speed.8a", AUTOMATED),
    Check("speed.8b", AUTOMATED, FULL, (Target("20%", 0.20),)),
    Check("speed.10a", AUTOMATED, FULL, (Target("three or more sizes", 3),)),
    # 7. Memory and scale
    Check("memory.2a", AUTOMATED, FULL),
    Check("memory.4a", AUTOMATED),
    Check("memory.4b", AUTOMATED),
    Check("memory.6a", AUTOMATED, FULL),
    Check("memory.6b", REVIEWED),
    Check("memory.6c", REVIEWED, targets=(Target("20,000", 20000),)),
    Check("memory.8a", AUTOMATED),
    Check("memory.8b", AUTOMATED),
    Check(
        "memory.10a", AUTOMATED, FULL,
        (Target("three or more sizes", 3), Target("16 GB", 16 * 2**30)),
    ),
    # 8. Usability
    Check("usability.2a", AUTOMATED),
    Check("usability.4a", AUTOMATED),
    Check("usability.4b", AUTOMATED),
    Check("usability.4c", AUTOMATED),
    Check("usability.6a", REVIEWED),
    Check("usability.6b", AUTOMATED),
    Check("usability.6c", AUTOMATED),
    Check("usability.6d", AUTOMATED),
    _h("usability.8a", INDEPENDENT, 2),
    _h("usability.8b", INDEPENDENT, 2, Target("≥ 68", 68)),
    _h("usability.10a", TESTERS, 5, Target("≥ 80", 80)),
    # 9. Visualisation
    Check("visual.2a", AUTOMATED),
    Check("visual.4a", AUTOMATED),
    Check("visual.4b", AUTOMATED),
    Check("visual.6a", REVIEWED),
    Check("visual.6b", REVIEWED),
    Check("visual.6c", AUTOMATED),
    Check("visual.6d", REVIEWED, targets=(Target("≥ 300", 300),)),
    _h("visual.8a", INDEPENDENT, 1, Target("≥ 80%", 0.80)),
    _h("visual.10a"),
    # 10. Exports
    Check("export.2a", AUTOMATED, FIXTURES),
    Check("export.4a", AUTOMATED),
    Check("export.4b", REVIEWED),
    Check("export.6a", REVIEWED),
    Check("export.6b", AUTOMATED),
    Check("export.6c", AUTOMATED),
    Check("export.6d", AUTOMATED),
    Check("export.8a", AUTOMATED),
    Check("export.8b", AUTOMATED, FIXTURES),
    Check("export.10a", AUTOMATED),
    # 11. Documentation and help
    Check("docs.2a", REVIEWED),
    Check("docs.4a", AUTOMATED, QUICK),
    Check("docs.4b", AUTOMATED, QUICK),
    Check("docs.4c", REVIEWED),
    _h("docs.6a", ("Harvey",)),
    Check("docs.6b", REVIEWED),
    Check("docs.6c", REVIEWED, QUICK),
    Check("docs.6d", REVIEWED),
    _h("docs.8a", INDEPENDENT),
    Check("docs.8b", REVIEWED),
    Check("docs.10a", REVIEWED),
    # 12. Robustness
    Check("robust.2a", AUTOMATED),
    Check("robust.4a", AUTOMATED),
    Check("robust.4b", AUTOMATED),
    Check("robust.6a", AUTOMATED, FIXTURES),
    Check("robust.6b", REVIEWED),
    _h("robust.6c"),
    Check("robust.8a", AUTOMATED, targets=(Target("Three or more", 3),)),
    Check("robust.10a", AUTOMATED),
    # 13. Reproducibility
    Check("repro.2a", AUTOMATED, FIXTURES),
    Check("repro.4a", AUTOMATED),
    Check("repro.4b", AUTOMATED),
    Check("repro.6a", AUTOMATED, FIXTURES),
    Check("repro.6b", AUTOMATED),
    Check("repro.6c", AUTOMATED, FIXTURES),
    Check("repro.8a", AUTOMATED),
    Check("repro.8b", AUTOMATED, FIXTURES),
    Check("repro.10a", REVIEWED),
    # 14. Testing and CI
    Check("tests.2a", AUTOMATED, QUICK),
    Check("tests.4a", AUTOMATED, QUICK),
    Check("tests.4b", REVIEWED),
    Check("tests.4c", AUTOMATED, targets=(Target("two weeks", 14),)),
    Check("tests.6a", AUTOMATED, QUICK, (Target("≥ 70%", 0.70),)),
    Check("tests.6b", AUTOMATED, QUICK),
    Check("tests.6c", AUTOMATED, targets=(Target("10 min", 600),)),
    Check("tests.8a", AUTOMATED, QUICK),
    Check("tests.8b", AUTOMATED, targets=(Target("≥ 80%", 0.80),)),
    Check("tests.10a", AUTOMATED, targets=(Target("≥ 90%", 0.90),)),
    # 15. Installation and packaging
    Check("package.2a", AUTOMATED, QUICK),
    Check("package.4a", AUTOMATED, QUICK),
    Check("package.4b", AUTOMATED),
    Check("package.6a", REVIEWED, QUICK),
    _h("package.6b", ("Harvey", "Mason", "Finbar"), 3),
    Check("package.6c", REVIEWED, QUICK),
    Check("package.8a", AUTOMATED),
    _h("package.8b", INDEPENDENT),
    Check("package.10a", AUTOMATED),
    # 16. Code maintainability
    Check("code.2a", REVIEWED, QUICK),
    Check("code.4a", AUTOMATED),
    Check("code.4b", AUTOMATED, QUICK, (Target("80%", 0.80),)),
    Check("code.4c", REVIEWED),
    Check("code.6a", AUTOMATED, QUICK),
    Check("code.6b", REVIEWED, QUICK, (Target("3,000", 3000),)),
    Check("code.6c", REVIEWED, QUICK),
    Check("code.8a", AUTOMATED),
    Check("code.8b", REVIEWED),
    _h("code.10a", INDEPENDENT),
)

CHECKS_BY_ID = {check.id: check for check in CHECKS}


def check(check_id: str) -> Check:
    return CHECKS_BY_ID[check_id]


def target(check_id: str, index: int = 0) -> float:
    """The value of a check's *index*-th target."""
    return CHECKS_BY_ID[check_id].targets[index].value


def checks_of(aspect: str) -> tuple[Check, ...]:
    return tuple(each for each in CHECKS if each.aspect == aspect)


def aspect(key: str) -> Aspect:
    return next(each for each in ASPECTS if each.key == key)
