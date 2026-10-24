"""What the left panel's Statistics tab shows, without Qt.

The export stage leaves what the run's statistics found on the graph it
measured (``pipeline.stages.STATISTICS_REPORT``): one section per statistic
measure that ran, then the resistances, the branch orders, the weighted
betweenness and communities, and the tissue volume. The tab draws one button
per section with that section's rows hidden behind it; this module says which
buttons, in what order, under what titles, and what each row reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from haemolynx.pipeline.stages import (
    BRANCH_ORDER_SECTION,
    HAEMODYNAMICS_SECTION,
    STATISTICS_REPORT,
    TISSUE_VOLUME_SECTION,
    WEIGHTED_MEASURES_SECTION,
)
from haemolynx.statistics import STATISTIC_MEASURES, ZERO_RESISTANCE_BRIDGES
from haemolynx.synthetic.compare import COMPARISON_SECTION

#: The sections that are not a statistic measure, in the order their buttons
#: follow the measures'.
OTHER_SECTIONS: tuple[str, ...] = (
    HAEMODYNAMICS_SECTION,
    BRANCH_ORDER_SECTION,
    WEIGHTED_MEASURES_SECTION,
    ZERO_RESISTANCE_BRIDGES,
    TISSUE_VOLUME_SECTION,
    COMPARISON_SECTION,
)

#: Their buttons' titles. A measure's is its name as its checkbox on the
#: Additional measurements tab spells it ("Statistics murray law").
SECTION_TITLES: dict[str, str] = {
    HAEMODYNAMICS_SECTION: "Haemodynamics",
    BRANCH_ORDER_SECTION: "Branch orders",
    WEIGHTED_MEASURES_SECTION: "Weighted betweenness and communities",
    ZERO_RESISTANCE_BRIDGES: "Zero-resistance bridges",
    TISSUE_VOLUME_SECTION: "Tissue volume",
    COMPARISON_SECTION: "Template comparison",
}

#: The weighted measures' three edge weightings, as their headings read.
WEIGHTING_TITLES: dict[str, str] = {
    "edge_resistance": "Weighted by resistance",
    "edge_length": "Weighted by length",
    "edge_flow_abs": "Weighted by |flow|",
}

#: Rows shown under one heading. Full mode's betweenness is a row per node;
#: the exported files have them all.
MAX_ROWS = 60

#: Items of a list shown in one value, and characters of the whole value.
MAX_LIST_ITEMS = 12
MAX_VALUE_CHARS = 240


@dataclass(frozen=True)
class StatisticRow:
    """One line behind a button: a name and its value, or a heading (no
    value) over the deeper lines that follow it."""

    name: str
    value: str = ""
    depth: int = 0
    heading: bool = False


@dataclass(frozen=True)
class StatisticsSection:
    """One button: the measure (or other section) and the rows behind it."""

    key: str
    title: str
    rows: tuple[StatisticRow, ...]


def statistics_report(graph: Any) -> Mapping[str, Any] | None:
    """The report the last export left on *graph*, or None."""
    record = getattr(graph, "graph", None)
    if not isinstance(record, Mapping):
        return None
    report = record.get(STATISTICS_REPORT)
    return report if isinstance(report, Mapping) else None


def section_title(key: str) -> str:
    """The button's text for section *key*."""
    return SECTION_TITLES.get(key) or key.replace("_", " ").capitalize()


def section_tooltip(key: str) -> str:
    """The button's hover text for section *key*: what its statistic
    measures, or what the button does for a section with no entry."""
    from haemolynx.gui.chrome_tooltips import (
        STATISTICS_SECTION_TOOLTIP,
        STATISTICS_SECTION_TOOLTIPS,
    )

    return STATISTICS_SECTION_TOOLTIPS.get(key, STATISTICS_SECTION_TOOLTIP)


def _order(key: str, position: int) -> tuple[int, int]:
    if key in STATISTIC_MEASURES:
        return (0, STATISTIC_MEASURES.index(key))
    if key in OTHER_SECTIONS:
        return (1, OTHER_SECTIONS.index(key))
    return (2, position)


def statistics_sections(report: Mapping[str, Any] | None) -> tuple[StatisticsSection, ...]:
    """One section per measure that ran, in the order the Additional
    measurements tab lists them, then the others; none without a report."""
    if not report:
        return ()
    sections = report.get("sections") or {}
    keys = sorted(
        enumerate(sections), key=lambda item: _order(item[1], item[0])
    )
    return tuple(
        StatisticsSection(
            key=key, title=section_title(key), rows=section_rows(key, sections[key])
        )
        for _position, key in keys
    )


def section_rows(key: str, values: Any) -> tuple[StatisticRow, ...]:
    """What section *key* shows behind its button."""
    if not isinstance(values, Mapping):
        return (StatisticRow(section_title(key), value_text(values)),)
    if key == BRANCH_ORDER_SECTION:
        # Each order's record repeats its own name, already its heading.
        values = {
            order: (
                {name: v for name, v in record.items() if name != "Branch Order"}
                if isinstance(record, Mapping)
                else record
            )
            for order, record in values.items()
        }
    elif key == WEIGHTED_MEASURES_SECTION:
        values = {WEIGHTING_TITLES.get(name, name): v for name, v in values.items()}
    else:
        # Full mode's betweenness is keyed by node.
        values = {name if isinstance(name, str) else f"Node {name}": v for name, v in values.items()}
    rows = _rows(values, 0)
    return rows or (StatisticRow("Nothing to report"),)


def _rows(values: Mapping[Any, Any], depth: int) -> tuple[StatisticRow, ...]:
    rows: list[StatisticRow] = []
    items = list(values.items())
    for name, value in items[:MAX_ROWS]:
        if isinstance(value, Mapping):
            rows.append(StatisticRow(str(name), depth=depth, heading=True))
            rows.extend(_rows(value, depth + 1) or (StatisticRow("None", depth=depth + 1),))
        else:
            rows.append(StatisticRow(str(name), value_text(value), depth))
    hidden = len(items) - MAX_ROWS
    if hidden > 0:
        rows.append(StatisticRow(f"… and {hidden:,} more in the exported files", depth=depth))
    return tuple(rows)


def value_text(value: Any) -> str:
    """*value* as one line: numbers to six significant figures, as the
    statistics CSV writes them, and a long list cut short with its length."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, Mapping):
        texts = [f"{name}: {value_text(item)}" for name, item in value.items()]
        return _joined(texts, "{}")
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value)
        except TypeError:
            items = list(value)
        return _joined([value_text(item) for item in items], "{}")
    if isinstance(value, (list, tuple)):
        return _joined([value_text(item) for item in value], "[]")
    return str(value)


def _joined(texts: list[str], braces: str) -> str:
    """*texts* in *braces*, cut short (and counted) past the limits."""
    body = ", ".join(texts[:MAX_LIST_ITEMS])
    cut = len(texts) > MAX_LIST_ITEMS or len(body) > MAX_VALUE_CHARS
    if not cut:
        return f"{braces[0]}{body}{braces[1]}"
    body = body[:MAX_VALUE_CHARS].rstrip(", ")
    return f"{braces[0]}{body}, …{braces[1]} ({len(texts):,} items)"


#: Where a label may break a word too long for the tab: after each path
#: separator and underscore, so a file path or a setting's name wraps
#: instead of holding the tab wider than its dock.
_BREAK_AFTER = ("/", "\\", "_")
ZERO_WIDTH_SPACE = "​"


def breakable(text: str) -> str:
    """*text* with a zero-width space after each path separator and
    underscore, which a word-wrapping label may break at."""
    for character in _BREAK_AFTER:
        text = text.replace(character, character + ZERO_WIDTH_SPACE)
    return text


def statistics_summary(report: Mapping[str, Any] | None, statistics_tab: str) -> str:
    """The line above the buttons: what the report is, or how to get one.
    *statistics_tab* is the title of the tab the Statistics setting is on."""
    if not report or not report.get("sections"):
        return (
            "No statistics yet. They show here once a run's export has "
            f"computed them (Statistics, on {statistics_tab})."
        )
    count = len(report["sections"])
    parts = [f"{count} result{'s' if count != 1 else ''} from the last run"]
    mode = report.get("mode")
    if mode:
        parts.append(f"in {mode} mode")
    text = " ".join(parts) + ". Press one to show it."
    csv = report.get("csv")
    if csv:
        text += f"\nSaved to {csv}"
    return text
