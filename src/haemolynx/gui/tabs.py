"""Which settings belong to which pipeline stage.

The panel shows one tab per stage, in the order the pipeline runs them, so a
user configures a run the way it executes rather than the way the config file
is laid out. The stages themselves -- their order, titles, summaries, and which
settings steer each -- are :data:`haemolynx.pipeline.progress.STAGES`, which is
also what a progress bar counts through; this module only turns them into tabs.

The schema groups settings by *section*, which is close but not the same: a
section can span stages (`Pipeline stages` holds the skeleton thresholds, the
graph thresholds and the output prefix), and a stage can span sections
(`assign_boundaries` reads the boundary section plus toggles from elsewhere).
So the split is declared rather than derived -- but it was *built* from which
stage function actually reads which setting, and :mod:`tests.test_gui_tabs`
fails if a setting reaches no tab or more than one.

Claims are resolved in two passes: named settings first, in stage order, then
whole sections. A stage can therefore take one setting out of a section another
stage owns, which is what `base_plot_dir` (declared under boundary assignment,
used for output) needs.

Stages and tabs are not quite one for one either: a stage that names another
stage's title in `Stage.tab` puts its rows on that tab and opens none of its
own, which is how `solve` shows its boundary pressures next to the
haemodynamics they belong with while staying a stage a run reports through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from haemolynx.gui.form import Field, fields_for
from haemolynx.parsers.schema import Schema
from haemolynx.pipeline.progress import STAGES, Stage

__all__ = [
    "STAGES",
    "Stage",
    "Tab",
    "assign_to_stages",
    "tab_title",
    "tab_titles",
    "tabs_for",
    "unassigned",
]


def tab_title(stage: Stage) -> str:
    """Which tab *stage*'s rows go on: its own title unless it names another's."""
    return stage.tab or stage.title


def tab_titles(stages: Sequence[Stage] | None = None) -> tuple[str, ...]:
    """The tabs the panel opens, in order: one per stage that names no other.

    Raises rather than dropping rows when a stage points at a tab that does not
    exist -- a mistyped title would otherwise take every setting that stage
    claims off the panel altogether, and `unassigned` would still read empty.
    """
    stages = STAGES if stages is None else stages
    own = tuple(stage.title for stage in stages if stage.tab is None)
    unknown = sorted(
        {
            stage.tab
            for stage in stages
            if stage.tab is not None and stage.tab not in own
        }
    )
    if unknown:
        raise ValueError(
            f"Stage tab(s) {unknown} name no tab. Tabs are: {list(own)}. Set "
            "Stage.tab to the title of a stage that opens one, or leave it None."
        )
    return own


def assign_to_stages(schema: Schema) -> dict[str, str]:
    """Setting name -> the title of the tab it belongs on.

    Named claims win over section claims, so a stage can take one setting out
    of a section another stage owns.
    """
    tab_titles()
    owner: dict[str, str] = {}
    for stage in STAGES:
        for name in stage.settings:
            if name in schema:
                owner.setdefault(name, tab_title(stage))
    for stage in STAGES:
        for section in stage.sections:
            for setting in schema:
                if setting.section == section:
                    owner.setdefault(setting.name, tab_title(stage))
    return owner


def unassigned(schema: Schema) -> list[str]:
    """Settings no tab claims. Must be empty: a new setting needs a home."""
    owner = assign_to_stages(schema)
    return sorted(name for name in schema.names if name not in owner)


@dataclass
class Tab:
    """A stage and the form rows shown on its tab."""

    stage: Stage
    fields: list[Field] = field(default_factory=list)


def tabs_for(schema: Schema, values=None) -> list[Tab]:
    """The panel's tabs, each carrying its own rows, in pipeline order."""
    owner = assign_to_stages(schema)
    by_title = {stage.title: Tab(stage=stage) for stage in STAGES if stage.tab is None}
    for row in fields_for(schema, values):
        title = owner.get(row.name)
        if title is not None:
            by_title[title].fields.append(row)
    return list(by_title.values())
