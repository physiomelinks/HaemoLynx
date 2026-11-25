"""An optimiser run's proposal set against what the panel holds now.

"Optimise settings" and "Optimise FWHM settings" used to write their winners
straight into the panel. Now a run proposes: the panel shows which settings
would change, from what to what, and any measure the run made worse (see
:mod:`haemolynx.optimisation.scorecard`), and writes nothing until Apply.

Pure: values in, changes and text out -- nothing here imports Qt or napari.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from haemolynx.optimisation.scorecard import ScoreRow, worse_rows


@dataclass(frozen=True)
class ProposedChange:
    """One setting a run would change: its value now, and the proposed one."""

    name: str
    was: Any
    now: Any


def _same(a: Any, b: Any) -> bool:
    """Equal as settings: numbers to rounding, booleans only to booleans,
    sequences item by item."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def proposed_changes(
    proposed: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[ProposedChange, ...]:
    """Every *proposed* setting whose value differs from the panel's
    *current* one, in the proposal's order. Both in the settings' own terms
    (what ``current_values()`` returns), not the widgets' display form."""
    return tuple(
        ProposedChange(name, current.get(name), value)
        for name, value in proposed.items()
        if not _same(current.get(name), value)
    )


def _shown(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def review_text(changes: Sequence[ProposedChange], scorecard: Sequence[ScoreRow] = ()) -> str:
    """The review a run's proposal is shown with: a "was -> now" line per
    change, then a warning naming every measure the run made worse."""
    if changes:
        count = len(changes)
        lines = [f"{count} setting{'s' if count != 1 else ''} would change:"]
        lines += [f"  {c.name}: {_shown(c.was)} -> {_shown(c.now)}" for c in changes]
    else:
        lines = ["No setting would change: the search kept every value you have."]
    worse = worse_rows(scorecard)
    if worse:
        lines.append(
            "WARNING: worse than your current settings on "
            + ", ".join(f"{row.name} ({row.before:.3g} -> {row.after:.3g})" for row in worse)
            + "."
        )
    lines.append("Apply writes these into the panel; Discard leaves it as it is.")
    return "\n".join(lines)
