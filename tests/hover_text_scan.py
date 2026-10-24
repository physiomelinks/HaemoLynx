"""The controls under a built widget that have no hover text.

Shared by the GUI tests that scan what HaemoLynx builds for controls a user
can hover without being told what they do (``usability.6d``): the panel, its
docks and windows, the Batch run window. Needs a Qt binding; import it from
a test that already skips without one.
"""
from __future__ import annotations


def _compound_value_widget(widget) -> bool:
    """A magicgui control made of others that holds one value: a path row's
    FileEdit, a text box and a Select file button."""
    try:
        from magicgui.widgets.bases import ValuedContainerWidget as compound
    except ImportError:  # magicgui before 0.10, where FileEdit was a Container
        from magicgui.widgets import FileEdit as compound
    return isinstance(getattr(widget, "_magic_widget", None), compound)


def controls_without_hover_text(root) -> list[str]:
    """Every control at or under *root* -- button, field, choice, list,
    table, slider, tab -- whose hover text is empty, shown or not.

    A control's own parts count as that control's, since Qt shows a control's
    hover text over any part of it that has none: Qt's own (a choice's list, a
    spin box's text, a table's headers, corner button and scroll bars, a tab
    bar's arrows, a labelled slider's readout) and magicgui's (a path row's
    text box and button).
    """
    from qtpy.QtWidgets import (
        QAbstractButton,
        QAbstractItemView,
        QAbstractScrollArea,
        QAbstractSlider,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QScrollBar,
        QTabBar,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    controls = (
        QAbstractButton,
        QAbstractItemView,
        QAbstractSlider,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
    )

    def part_of_another(widget) -> bool:
        if isinstance(widget, QScrollBar) or isinstance(widget.parentWidget(), QAbstractScrollArea):
            return True
        ancestor = widget.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, (QComboBox, QAbstractSpinBox, QTabBar, QAbstractSlider)):
                return True
            if _compound_value_widget(ancestor):
                return True
            ancestor = ancestor.parentWidget()
        return False

    def name(widget) -> str:
        magic = getattr(widget, "_magic_widget", None)
        if _compound_value_widget(widget):
            return f"{type(magic).__name__} {magic.name!r}"
        return f"{type(widget).__name__} {getattr(widget, 'text', lambda: '')()!r}"

    widgets = [root, *root.findChildren(QWidget)]
    missing = [
        name(widget)
        for widget in widgets
        if (isinstance(widget, controls) or _compound_value_widget(widget))
        and not part_of_another(widget)
        and not widget.toolTip().strip()
    ]
    for tabs in (widget for widget in widgets if isinstance(widget, QTabWidget)):
        missing += [f"tab {tabs.tabText(i)!r}" for i in range(tabs.count()) if not tabs.tabToolTip(i).strip()]
    return missing
