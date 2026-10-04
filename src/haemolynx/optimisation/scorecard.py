"""The settings a search started from against the ones it chose, on one set
of measures.

Each sweep judges its candidates by its own measure, and guards a few others
against getting worse -- but only against the settings that sweep started
from. Nothing checked that the settings a whole run ends with are no worse
than the ones it was given. A scorecard measures both on the same grid and
the same measures, and says where the result is worse.

Pure: rows in, rows and text out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScoreRow:
    """One measure of the starting settings (*before*) and the chosen ones
    (*after*)."""

    name: str
    before: float
    after: float
    higher_is_better: bool
    #: How far *after* may move the wrong way from *before* before it counts
    #: as worse: rounding between two near-identical results is not a
    #: regression.
    tolerance: float = 0.0

    def _gain(self) -> float:
        change = self.after - self.before
        return change if self.higher_is_better else -change

    @property
    def worse(self) -> bool:
        return self._gain() < -self.tolerance

    @property
    def better(self) -> bool:
        return self._gain() > self.tolerance


def worse_rows(rows: Iterable[ScoreRow]) -> tuple[ScoreRow, ...]:
    """The rows on which the chosen settings are worse than the starting ones."""
    return tuple(row for row in rows if row.worse)


def _number(value: float) -> str:
    return f"{value:.3g}"


def scorecard_lines(rows: Sequence[ScoreRow]) -> list[str]:
    """The scorecard as report lines: one per measure, marked where it got
    better or worse, then a warning naming every worse one."""
    if not rows:
        return []
    width = max(len(row.name) for row in rows)
    lines = ["Your starting settings against these, measured the same way on the same grid:"]
    for row in rows:
        mark = "  worse" if row.worse else ("  better" if row.better else "")
        lines.append(
            f"    {row.name.ljust(width)}  {_number(row.before)} -> {_number(row.after)}{mark}"
        )
    worse = worse_rows(rows)
    if worse:
        lines.append(
            "WARNING: worse than your starting settings on "
            + ", ".join(f"{row.name} ({_number(row.before)} -> {_number(row.after)})" for row in worse)
            + ". Check these before using the settings."
        )
    return lines
