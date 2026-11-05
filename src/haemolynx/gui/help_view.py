"""The panel's help window: a directory of tabs, and the text of the one open.

Qt is imported inside :func:`help_window`, so importing this module does not
need a GUI. The words on each page come from :mod:`haemolynx.gui.stage_help`
and are left as that module stored them.
"""
from __future__ import annotations

from typing import Sequence

from haemolynx.gui.stage_help import HelpTopic, topic_index


def help_window(parent, topics: Sequence[HelpTopic], current_title: str):
    """A non-modal window opened on *current_title*.

    The left pane lists every topic. The right pane is that topic's text,
    empty until someone has written it. ``show_topic`` moves the directory
    to a title and shows its text; the window opens already on
    *current_title*.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QHBoxLayout,
        QSplitter,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
    )

    dialog = QDialog(parent)
    dialog.setObjectName("haemolynx_help_window")
    dialog.setWindowTitle("Help")
    dialog.resize(760, 480)
    dialog.setModal(False)

    tree = QTreeWidget()
    tree.setObjectName("haemolynx_help_directory")
    tree.setHeaderHidden(True)
    tree.setSelectionMode(QAbstractItemView.SingleSelection)
    text = QTextEdit()
    text.setObjectName("haemolynx_help_text")
    text.setReadOnly(True)

    items: list[QTreeWidgetItem] = []
    for topic in topics:
        item = QTreeWidgetItem([topic.title])
        item.setData(0, Qt.ItemDataRole.UserRole, topic.body)
        tree.addTopLevelItem(item)
        items.append(item)

    def show_body(item) -> None:
        if item is None:
            text.setPlainText("")
            return
        body = item.data(0, Qt.ItemDataRole.UserRole)
        text.setPlainText("" if body is None else str(body))

    tree.currentItemChanged.connect(lambda current, _previous: show_body(current))

    def show_topic(title: str) -> None:
        if not items:
            text.setPlainText("")
            return
        chosen = items[topic_index(topics, title)]
        tree.setCurrentItem(chosen)
        show_body(chosen)

    dialog.show_topic = show_topic
    dialog.directory = tree
    dialog.text = text

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setObjectName("haemolynx_help_splitter")
    splitter.addWidget(tree)
    splitter.addWidget(text)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([220, 540])

    layout = QHBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.addWidget(splitter)
    show_topic(current_title)
    return dialog
