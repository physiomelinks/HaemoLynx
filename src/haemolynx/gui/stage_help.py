"""One help entry per panel tab.

The text of each entry is empty until a person writes it. Nothing here
composes that text: see the Help menu section of ``CLAUDE.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HelpTopic:
    """One page in the help window: a tab's title, and the text a person wrote."""

    title: str
    body: str = ""


def topics_for_tabs(titles: Sequence[str]) -> tuple[HelpTopic, ...]:
    """One empty page per tab, in the panel's own order."""
    return tuple(HelpTopic(title=title) for title in titles)


def topic_index(topics: Sequence[HelpTopic], title: str) -> int:
    """Where *title* sits, or the first page when the window has no such tab."""
    for index, topic in enumerate(topics):
        if topic.title == title:
            return index
    return 0
