"""One record per tool, and the table that sets them side by side.

Pure: every tool's side turns its own output into a :class:`Measurements`
(totals and per-segment arrays), and nothing here runs a tool. A *segment* is
whatever that tool calls one -- a branch between two branch points or end
points of its graph; HaemoLynx's is an edge of the graph its statistics read
(zero-resistance bridges contracted, as :mod:`haemolynx.statistics` does).

Tools cut the same vessels into very different numbers of segments (a raw
skeleton into thousands of pieces a few microns long), so the table gives the
tortuosity and diameter medians weighted by segment length as well: the value
at a typical micron of centreline, which does not depend on how it was cut.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

#: A segment this much longer than its chord counts as tortuous in the table.
TORTUOUS_ABOVE = 1.1

_ARRAYS = ("segment_lengths_um", "tortuosity", "diameters_um")


@dataclass
class Measurements:
    """What one tool measured on one mask.

    The three arrays run segment by segment, one entry each. ``tortuosity``
    is ``nan`` for a closed loop (its two ends one point: each tool says so in
    its own way -- VesselVio writes 0, skan's ratio is infinite -- and
    ``loops`` counts them); ``diameters_um`` is ``nan`` where the tool gave
    none.
    """

    tool: str
    total_length_um: float
    segments: int
    branch_points: int
    end_points: int
    segment_lengths_um: np.ndarray
    tortuosity: np.ndarray
    diameters_um: np.ndarray
    loops: int = 0
    #: How this tool defines each quantity, in a sentence or two each.
    definitions: dict = field(default_factory=dict)
    #: Anything else the tool reported that the table should show.
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _ARRAYS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        sizes = {getattr(self, name).size for name in _ARRAYS}
        if len(sizes) != 1:
            raise ValueError(f"{self.tool}: per-segment arrays of different lengths {sizes}")

    def to_json(self) -> dict:
        data = asdict(self)
        for name in _ARRAYS:
            data[name] = [None if not np.isfinite(v) else float(v) for v in getattr(self, name)]
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Measurements":
        return cls(**{**data, **{
            name: np.asarray([np.nan if v is None else v for v in data[name]], dtype=float)
            for name in _ARRAYS
        }})


def tortuosity_of(lengths: Sequence[float], chords: Sequence[float]) -> tuple[np.ndarray, int]:
    """Length over chord for each segment (``nan`` for a loop, chord 0), and
    how many were loops."""
    lengths = np.asarray(lengths, dtype=float)
    chords = np.asarray(chords, dtype=float)
    closed = ~(chords > 0)
    tortuosity = np.full(lengths.shape, np.nan)
    tortuosity[~closed] = lengths[~closed] / chords[~closed]
    return tortuosity, int(np.count_nonzero(closed))


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    """25th, 50th and 75th percentiles; ``nan`` for none."""
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at half the total weight, values sorted; ``nan`` for none."""
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return math.nan
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def summary(m: Measurements) -> dict[str, float]:
    """The numbers the table shows for one tool."""
    lengths = m.segment_lengths_um[np.isfinite(m.segment_lengths_um)]
    tortuous = np.isfinite(m.tortuosity)
    tortuosity = m.tortuosity[tortuous]
    measured = np.isfinite(m.diameters_um)
    diameters = m.diameters_um[measured]
    lq1, lmed, lq3 = quartiles(lengths)
    tq1, tmed, tq3 = quartiles(tortuosity)
    dq1, dmed, dq3 = quartiles(diameters)
    return {
        "Total centreline length (um)": m.total_length_um,
        "Segments": m.segments,
        "Branch points (degree >= 3)": m.branch_points,
        "End points (degree 1)": m.end_points,
        "Loops (segment whose ends meet)": m.loops,
        "Segment length, median (um)": lmed,
        "Segment length, IQR (um)": lq3 - lq1 if np.isfinite(lq1) else math.nan,
        "Segment length, mean (um)": float(np.mean(lengths)) if lengths.size else math.nan,
        "Tortuosity, median": tmed,
        "Tortuosity, mean": float(np.mean(tortuosity)) if tortuosity.size else math.nan,
        "Tortuosity, length-weighted median": weighted_median(m.tortuosity, m.segment_lengths_um),
        f"Segments with tortuosity > {TORTUOUS_ABOVE}": int(np.count_nonzero(tortuosity > TORTUOUS_ABOVE)),
        "Diameter, median (um)": dmed,
        "Diameter, IQR (um)": dq3 - dq1 if np.isfinite(dq1) else math.nan,
        "Diameter, mean (um)": float(np.mean(diameters)) if diameters.size else math.nan,
        "Diameter, length-weighted median (um)": weighted_median(m.diameters_um, m.segment_lengths_um),
        "Segments with a diameter": int(diameters.size),
    }


def relative_difference(value: float, reference: float) -> float:
    """``value / reference - 1``; ``nan`` when the reference is 0 or missing."""
    if reference is None or not np.isfinite(reference) or reference == 0 or not np.isfinite(value):
        return math.nan
    return float(value) / float(reference) - 1.0


def _cell(value: float) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if value is None or not np.isfinite(value):
        return "-"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}" if abs(value) < 10 else f"{value:.1f}"


def comparison_table(
    sides: Sequence[Measurements],
    *,
    reference: str = "HaemoLynx",
    truth: dict[str, float] | None = None,
) -> str:
    """A markdown table, one column per tool (and one for *truth*, the design
    values of a synthetic mask, when given), each cell with its difference from
    *reference*'s in brackets."""
    summaries = {m.tool: summary(m) for m in sides}
    rows = list(next(iter(summaries.values())))
    ref = summaries.get(reference)
    header = ["Quantity"] + (["Design"] if truth else []) + [m.tool for m in sides]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row]
        if truth:
            cells.append(_cell(truth[row]) if row in truth else "")
        for m in sides:
            value = summaries[m.tool][row]
            text = _cell(value)
            if ref is not None and m.tool != reference:
                diff = relative_difference(value, ref[row])
                if np.isfinite(diff) and "IQR" not in row:
                    text += f" ({diff:+.0%})"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_measurements(sides: Sequence[Measurements], path: Path) -> None:
    path.write_text(json.dumps([m.to_json() for m in sides], indent=1), encoding="utf-8")


def read_measurements(path: Path) -> list[Measurements]:
    return [Measurements.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]
