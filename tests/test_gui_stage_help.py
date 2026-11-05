"""Help pages: one empty entry per tab, until a person writes the text."""
from __future__ import annotations

from haemolynx.gui.stage_help import topic_index, topics_for_tabs
from haemolynx.gui.tabs import tab_titles


def test_each_tab_has_an_empty_help_page():
    titles = tab_titles()
    topics = topics_for_tabs(titles)

    assert [topic.title for topic in topics] == list(titles)
    assert all(topic.body == "" for topic in topics)
    assert topic_index(topics, titles[0]) == 0
    assert topic_index(topics, titles[-1]) == len(titles) - 1
    assert topic_index(topics, "no such tab") == 0
