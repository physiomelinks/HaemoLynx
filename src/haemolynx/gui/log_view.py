"""A run's log in a window, drawn without letting the run wait for the window.

:mod:`haemolynx.gui.run_log` decides *what* a log window shows -- how a record
reads as text, how many lines may wait, what happens when one has to be thrown
away -- and needs no GUI to do it. This module is the widget that shows it, and
nothing more: a read-only text box, a timer that empties the buffer onto it ten
times a second, and the five controls a log is no use without.

The split is the thread boundary. A run appends to the buffer from its worker
thread and from graph building's thread pool, thousands of times; this drains it
on the GUI thread, in batches, on its own timer. Nothing here is ever called
from a run.

Why this is not a magicgui widget, in one place so nobody has to guess:
magicgui's ``TextEdit`` wraps a ``QTextEdit``, which has no block-count cap. A
raw ``QPlainTextEdit`` has :meth:`~qtpy.QtGui.QTextDocument.setMaximumBlockCount`,
and that cap is what lets a two-hour run be shown at all -- Qt drops the oldest
block in C++ in constant time, with nothing on the Python side counting lines.
It is a deliberate inconsistency with the rest of the panel, in the company of
the raw ``QProgressBar`` and raw ``QScrollArea`` already there.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from haemolynx.gui.run_log import (
    DEFAULT_LEVEL,
    DRAIN_INTERVAL_MS,
    LEVELS,
    MAX_LINES,
    RUN_CANCELLED,
    RUN_STARTED,
    RunLog,
    banner,
)

logger = logging.getLogger(__name__)

#: What the window puts where lines it could not keep up with would have been.
#: It names the level, because that is the part the reader can do something
#: about; :func:`haemolynx.gui.run_log.dropped_note` is the same admission
#: without the advice, for a caller with no level to blame.
DROPPED_FORMAT = "... {count} lines dropped (log level {level}) ..."

#: The choice that shows everything, for a run whose settings asked for it
#: (`verbose_logging`). Named here so a caller setting the level from a setting
#: does not have to spell a combo entry.
VERBOSE_LEVEL = "Debug"

#: What "Save..." offers to call the file.
DEFAULT_FILENAME = "haemolynx-run-log.txt"

#: The filter that dialog opens with.
SAVE_FILTER = "Text (*.txt);;All files (*)"


class LogView:
    """The panel's log window: what a run says, while it is saying it.

    What to show is :class:`haemolynx.gui.run_log.RunLog`, which needs no GUI
    and is tested without one; this only moves it onto a widget. Every method
    here touches Qt, so all of them must be called on the GUI thread -- the
    buffer is the boundary a run writes to from its own thread, and
    :meth:`drain` is the only thing that crosses it.

    Constructing this needs a ``QApplication``, which in a napari session
    already exists. Pass a *log* to share a buffer with whatever is capturing
    the library's logger (:func:`haemolynx.gui.run_log.attach`); leave it out
    and the view makes its own, which :attr:`run_log` then hands over.
    """

    def __init__(self, log: Optional[RunLog] = None) -> None:
        from qtpy.QtCore import QTimer
        from qtpy.QtGui import QFont
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QPlainTextEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self.run_log = log if log is not None else RunLog(level=LEVELS[DEFAULT_LEVEL])

        # See the module docstring for why this is not magicgui's TextEdit.
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setUndoRedoEnabled(False)
        # Wrapping is off on purpose: a statistics line runs to a few hundred
        # characters, and wrapped it becomes four rows that no longer line up
        # with anything. A log is read by scanning down its left edge.
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        # The bound. Qt trims the oldest block itself, in C++; nothing here
        # counts lines or rewrites the document.
        self.view.document().setMaximumBlockCount(MAX_LINES)
        monospace = QFont("Courier New")
        monospace.setStyleHint(QFont.Monospace)
        self.view.setFont(monospace)

        self.level_combo = QComboBox()
        self.level_combo.addItems(list(LEVELS))
        self.level_combo.setCurrentText(DEFAULT_LEVEL)
        self.level_combo.setToolTip(
            "How much of the run to show. Debug is per node and per iteration:"
            " thousands of lines a second on a real volume."
        )
        self.level_combo.currentTextChanged.connect(self._level_chosen)

        self.follow_box = QCheckBox("Follow")
        self.follow_box.setChecked(True)
        self.follow_box.setToolTip(
            "Keep the newest line in view. Turn it off, or just scroll up, to"
            " read something without being pulled back to the bottom."
        )

        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy the whole log to the clipboard.")
        self.copy_button.clicked.connect(self.copy)

        self.save_button = QPushButton("Save...")
        self.save_button.setToolTip("Write the whole log to a text file.")
        self.save_button.clicked.connect(self.save)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Empty the window and the buffer behind it.")
        self.clear_button.clicked.connect(self.reset)

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Log level"))
        row.addWidget(self.level_combo)
        row.addWidget(self.follow_box)
        row.addStretch(1)
        for button in (self.copy_button, self.save_button, self.clear_button):
            row.addWidget(button)

        self._native = QWidget()
        layout = QVBoxLayout(self._native)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controls)
        layout.addWidget(self.view)

        # Parented to the widget, so it goes when the panel does rather than
        # firing at a text box that has been deleted underneath it.
        self._timer = QTimer(self._native)
        self._timer.setInterval(DRAIN_INTERVAL_MS)
        self._timer.timeout.connect(self.drain)

    # -- what the panel puts in its layout ----------------------------------

    @property
    def native(self):
        """The widget to dock, or to add to a layout."""
        return self._native

    @property
    def level(self) -> int:
        """The level the window is set to, as :mod:`logging` numbers it.

        What :func:`haemolynx.gui.run_log.attach` should capture at.
        """
        return LEVELS.get(self.level_combo.currentText(), LEVELS[DEFAULT_LEVEL])

    @property
    def follow(self) -> bool:
        """Whether the newest line is being kept in view."""
        return self.follow_box.isChecked()

    @follow.setter
    def follow(self, follow: bool) -> None:
        self.follow_box.setChecked(bool(follow))

    def set_level(self, name: str) -> None:
        """Show *name*'s worth of the run from here on: a key of :data:`LEVELS`.

        An unknown name is ignored rather than raised on: this is a display
        control, and losing the log window is a worse answer than showing the
        wrong amount of it.
        """
        if name in LEVELS:
            self.level_combo.setCurrentText(name)
        else:
            logger.debug("no such log level to show: %r", name)

    @property
    def running(self) -> bool:
        """Whether the drain timer is going."""
        return self._timer.isActive()

    # -- the run's lifecycle ------------------------------------------------

    def start(self) -> None:
        """A run has begun: mark it in the log and start draining.

        The log is *not* cleared. Comparing the counts one set of parameters
        gives with the next is the reason to have this window at all, so a new
        run opens with a banner and the last run stays above it.
        """
        self.note(banner(RUN_STARTED))
        self.drain()
        self._timer.start()

    def stop(self) -> None:
        """The run has ended, however it ended: show the last of it and stop.

        Draining before stopping matters -- a run's final lines are written in
        the few milliseconds before it returns, and would otherwise sit in the
        buffer until the next run started.
        """
        self.drain()
        self._timer.stop()

    def cancelled(self) -> None:
        """Mark a run the user stopped, so the gap in the log is explained."""
        self.note(banner(RUN_CANCELLED))
        self.drain()

    def note(self, line: str) -> None:
        """Put *line* in the log as it stands -- a banner, or a failure."""
        self.run_log.add_line(line)

    def reset(self) -> None:
        """Empty the window and the buffer behind it. What "Clear" does."""
        self.run_log.reset()
        self.view.clear()

    # -- the tick -----------------------------------------------------------

    def drain(self) -> int:
        """Move what is waiting onto the widget; return how many lines that was.

        One :meth:`~qtpy.QtWidgets.QPlainTextEdit.appendPlainText` per tick,
        never one per record: at a hundred appends a second the run would be
        drawing the window instead of the other way round. The buffer caps what
        one call can hand over at
        :data:`~haemolynx.gui.run_log.MAX_PER_DRAIN`, so a burst of DEBUG
        cannot stall a tick either -- the newest are kept and the rest are
        counted.
        """
        try:
            drained = self.run_log.drain()
        except Exception:  # noqa: BLE001 - a timer slot must not raise
            logger.debug("could not read the run log", exc_info=True)
            return 0

        lines = list(drained.lines)
        if drained.dropped:
            # Where they would have been, since what was dropped is older than
            # what survived. One line per tick, whatever the count.
            lines.insert(
                0,
                DROPPED_FORMAT.format(
                    count=drained.dropped, level=self.level_combo.currentText()
                ),
            )
        if not lines:
            return 0
        try:
            self._append(lines)
        except Exception:  # noqa: BLE001 - the same
            logger.debug("could not draw %d log line(s)", len(lines), exc_info=True)
            return 0
        return len(lines)

    def _append(self, lines: Sequence[str]) -> None:
        bar = self.view.verticalScrollBar()
        # Asked *before* the append, or the answer is always "no": the new
        # lines have already moved the maximum by then. A reader who scrolled
        # up to look at a count must not be pulled back down every 100 ms, so
        # following is the scrollbar's own position and the checkbox, not the
        # checkbox alone.
        at_bottom = bar.value() >= bar.maximum()
        self.view.appendPlainText("\n".join(lines))
        if self.follow and at_bottom:
            bar.setValue(bar.maximum())

    # -- the buttons --------------------------------------------------------

    def copy(self) -> None:
        """Put the whole log on the clipboard.

        From the buffer, never from the widget: what the widget holds has been
        trimmed to the last :data:`MAX_LINES` blocks, and scraping it back out
        would also lose anything the buffer dropped and admitted to.
        """
        from qtpy.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:  # a platform with no clipboard at all
            logger.debug("no clipboard to copy the log to")
            return
        clipboard.setText(self.run_log.text())

    def save(self) -> Optional[str]:
        """Ask where to write the log, and write it. Returns the path, or None."""
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            None, "Save this run's log", DEFAULT_FILENAME, SAVE_FILTER
        )
        if not path:
            return None
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.run_log.text())
        return path

    def _level_chosen(self, name: str) -> None:
        """The combo changed: from here on, keep that much.

        Only what arrives next -- what is already in the buffer stays, because
        a window that erased its own history when the level moved would lose
        the one line the user had turned it up to find.
        """
        self.run_log.level = LEVELS.get(name, LEVELS[DEFAULT_LEVEL])
