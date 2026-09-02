"""The lines a run's log window shows, worked out without a GUI.

The pipeline already narrates itself through ``logging`` -- how many terminal
stubs pruning removed, how many nodes and edges each topology step left behind
-- and a napari user sees none of it: nothing in a napari session configures
logging, so every record is dropped by the root logger's level check before it
reaches anywhere at all.

This module is the half of a live log window that is not Qt: what a record
looks like as text, a bounded buffer to hold the text in between the run's
thread writing it and the GUI thread drawing it, and the handler that fills the
buffer for the length of a run. The widget's job is then only to copy
:meth:`RunLog.drain`'s lines into a text box on a timer, the same division as
:mod:`haemolynx.gui.progress` and :mod:`haemolynx.gui.run_state`.

**Capturing means lowering a level, not just adding a handler.** This is the
fact that makes the window possible at all: :func:`attach` sets the
``haemolynx`` logger's own level as well as the handler's, because a record
below the logger's effective level never reaches a handler to begin with. It
attaches to ``haemolynx`` rather than to the root logger on purpose -- lowering
root's level pulls in matplotlib's font manager, PIL and napari itself, which
between them out-talk the pipeline several times over -- and it leaves
propagation alone, so a console handler from
:func:`haemolynx.parsers.configure_console_logging` still receives everything.
It is an additional handler, never a replacement.

Three more decisions worth knowing before reading:

**Formatting happens on the way in.** :meth:`RunLog.add_record` turns the
record into strings immediately, on the emitting thread, so the buffer holds
text and not :class:`logging.LogRecord`s. A record keeps ``record.args``
alive, and in this library those arguments are graphs, skeletons and numpy
arrays: a buffer of two thousand records would pin a run's worth of volumes in
memory until the window was cleared.

**The log accumulates across runs.** Comparing the counts from one set of
parameters with the next is the reason to want the window at all, so a new run
adds a :data:`RUN_STARTED` banner rather than clearing what came before.
:meth:`RunLog.reset` exists for a Clear button to call.

**Overflow drops the oldest, and says so.** A run that out-talks the window
loses its oldest unshown lines, never its newest, and :attr:`RunLog.dropped`
counts them: a log with a silent hole in it is worse than one that admits to
the hole.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

#: The logger a run reports through: every module in the library gets its
#: logger from ``logging.getLogger(__name__)``, so this is the one ancestor
#: they all share.
LOGGER_NAME = "haemolynx"

#: What a user can ask to see, and the level each choice means. The keys are
#: what a combo box shows, in the order it should show them.
LEVELS: "dict[str, int]" = {
    "Warnings only": logging.WARNING,
    "Info": logging.INFO,
    "Debug": logging.DEBUG,
}

#: What the window starts on. The pipeline's own narration is INFO, and
#: DEBUG is per-node -- hundreds of thousands of lines on a real volume.
DEFAULT_LEVEL = "Info"

#: How many lines may wait to be shown before the oldest are dropped.
MAX_PENDING = 2000

#: How many lines the window keeps once they have been shown.
MAX_LINES = 5000

#: How many lines one :meth:`RunLog.drain` hands over. A text box cannot
#: render two thousand lines between two frames, and the newest are the ones
#: worth rendering.
MAX_PER_DRAIN = 500

#: How often the window should drain, in milliseconds. Ten times a second
#: reads as live and costs one timer callback.
DRAIN_INTERVAL_MS = 100

#: The banner a run opens with, so a log that keeps every run still shows
#: where one ended and the next began. Filled in by :func:`banner`.
RUN_STARTED = "=== Run started {time} ==="

#: And the one a run the user stopped closes with.
RUN_CANCELLED = "=== Run cancelled {time} ==="

#: How each line of a record reads: what it was, where it came from, what it
#: said. No timestamp: the window is live, and the width is better spent on
#: the message.
LINE_FORMAT = "{level:<7} {source}: {message}"

#: A multi-line message's later lines line up under the first, rather than
#: repeating its heading.
CONTINUATION_INDENT = "    "


def banner(template: str, when: Optional[datetime] = None) -> str:
    """*template* -- :data:`RUN_STARTED` or :data:`RUN_CANCELLED` -- as a line."""
    moment = when if when is not None else datetime.now()
    return template.format(time=moment.strftime("%H:%M:%S"))


def dropped_note(count: int) -> str:
    """The line that owns up to *count* lines the window could not keep up with."""
    return f"... {count} line(s) dropped: the log could not keep up ..."


def source_name(name: str) -> str:
    """A record's logger name without the package every one of them is inside.

    ``haemolynx.graph.prune`` reads as ``graph.prune``: the prefix is on every
    line, so it is width spent saying nothing.
    """
    prefix = LOGGER_NAME + "."
    return name[len(prefix):] if name.startswith(prefix) else name


def format_record(record: logging.LogRecord) -> "tuple[str, ...]":
    """*record* as one string per physical line of its message.

    One entry per line, rather than one string with newlines in it, because
    everything downstream counts entries: a buffer that held a fifty-line
    diagnostics report as a single "line" would report a bound it was not
    keeping.
    """
    message = record.getMessage()
    lines = message.splitlines() or [""]
    head = LINE_FORMAT.format(
        level=record.levelname,
        source=source_name(record.name),
        message=lines[0],
    )
    return (head, *(CONTINUATION_INDENT + line for line in lines[1:]))


@dataclass(frozen=True)
class Drained:
    """What one :meth:`RunLog.drain` handed over.

    *dropped* is the lines the caller will never see, counted since the last
    drain, so a window can say so where they would have been.
    """

    lines: "tuple[str, ...]"
    dropped: int


class RunLog:
    """A bounded buffer of log lines, written by a run and read by a window.

    Records arrive from the run's worker thread -- and, inside graph building,
    from a :class:`~concurrent.futures.ThreadPoolExecutor` -- while the GUI
    thread drains. Both ends take a lock, which is what makes
    :attr:`dropped` an exact count rather than an estimate: deciding to drop
    the oldest line and dropping it have to be one step. It is only ever held
    for a few list operations, and nothing is called while it is held.
    """

    def __init__(
        self,
        level: int = LEVELS[DEFAULT_LEVEL],
        max_pending: int = MAX_PENDING,
        max_lines: int = MAX_LINES,
        max_per_drain: int = MAX_PER_DRAIN,
    ) -> None:
        self._lock = threading.Lock()
        self._pending: "deque[str]" = deque()
        self._lines: "deque[str]" = deque(maxlen=max_lines)
        self._max_pending = max_pending
        self._max_per_drain = max_per_drain
        self._level = level
        self._dropped = 0
        #: How much of `_dropped` a drain has already owned up to.
        self._reported = 0

    # -- what the window sets -----------------------------------------------

    @property
    def level(self) -> int:
        """The lowest level of record this buffer keeps."""
        return self._level

    @level.setter
    def level(self, level: int) -> None:
        self._level = level

    @property
    def dropped(self) -> int:
        """How many lines this buffer has thrown away, in total."""
        return self._dropped

    @property
    def pending(self) -> int:
        """How many lines are waiting to be shown."""
        return len(self._pending)

    # -- what a run writes --------------------------------------------------

    def add_record(self, record: logging.LogRecord) -> None:
        """Keep *record*, as text, if it is at or above :attr:`level`.

        Called on whichever thread emitted it, which is why the formatting is
        done here: see the module docstring.
        """
        if record.levelno < self._level:
            return
        self._extend(format_record(record))

    def add_line(self, line: str) -> None:
        """Keep *line* as it stands -- a banner, or a note about the log itself."""
        self._extend((line,))

    def _extend(self, lines: "tuple[str, ...]") -> None:
        if not lines:
            return
        with self._lock:
            pending = self._pending
            pending.extend(lines)
            over = len(pending) - self._max_pending
            if over > 0:
                for _ in range(over):
                    pending.popleft()
                self._dropped += over

    # -- what the window reads ----------------------------------------------

    def drain(self) -> Drained:
        """Take everything waiting; hand back the newest of it.

        Everything taken leaves the buffer, so a second drain with nothing
        added in between returns nothing. What is over
        :data:`MAX_PER_DRAIN` is dropped rather than deferred: a window behind
        by two thousand lines should catch up to the present, not crawl
        through the past at five hundred lines a frame.
        """
        with self._lock:
            taken = list(self._pending)
            self._pending.clear()
            over = len(taken) - self._max_per_drain
            if over > 0:
                del taken[:over]
                self._dropped += over
            unreported = self._dropped - self._reported
            self._reported = self._dropped
            self._lines.extend(taken)
            return Drained(lines=tuple(taken), dropped=unreported)

    def text(self) -> str:
        """Every line the window has been given, oldest first.

        Bounded by :data:`MAX_LINES`: what a scrollback holds is a fixed
        amount of memory, whatever the run does.
        """
        with self._lock:
            return "\n".join(self._lines)

    def reset(self) -> None:
        """Forget everything -- what a Clear button does."""
        with self._lock:
            self._pending.clear()
            self._lines.clear()
            self._dropped = 0
            self._reported = 0


class RunLogHandler(logging.Handler):
    """A handler that puts each record into a :class:`RunLog` and stops there.

    Deliberately the cheapest thing that could work. Records arrive on the
    run's worker thread and, inside graph building, on a
    :class:`~concurrent.futures.ThreadPoolExecutor`'s threads; a handler that
    emitted a Qt signal per record would cross threads tens of thousands of
    times in a run and queue the GUI thread solid. All this does is append,
    and the GUI thread reads the buffer on its own timer.

    :meth:`emit` never raises into the run. A log line is the least important
    thing happening on that thread, so a bad ``%`` argument in a message goes
    to :meth:`logging.Handler.handleError` -- the same place the standard
    library sends a broken stream -- and the run carries on.
    """

    def __init__(self, log: RunLog, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self._log = log

    @property
    def log(self) -> RunLog:
        """The buffer this handler fills."""
        return self._log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log.add_record(record)
        except Exception:  # noqa: BLE001 - a log line must not break a run
            self.handleError(record)


def installed_handlers(
    logger: Optional[logging.Logger] = None,
) -> "tuple[RunLogHandler, ...]":
    """Every :class:`RunLogHandler` on *logger*, which should be one or none.

    Exists so that "attaching twice leaves one handler behind" is something a
    test can look at directly rather than infer from a line appearing once.
    """
    target = logger if logger is not None else logging.getLogger(LOGGER_NAME)
    return tuple(h for h in target.handlers if isinstance(h, RunLogHandler))


class Attachment:
    """What :func:`attach` did to the logger, and how to undo exactly that.

    Both the level and the handler list are put back from the snapshot taken
    when the handler went on, so a run cannot leave the library's logging any
    different from how it found it. Restoring the list rather than removing
    one handler from it does mean a handler some *other* code added to
    ``haemolynx`` during the run goes too; for the length of one run, on one
    library's own logger, that is worth a guarantee that nothing accumulates.
    """

    def __init__(
        self,
        logger: logging.Logger,
        handler: RunLogHandler,
        previous_level: int,
        previous_handlers: "tuple[logging.Handler, ...]",
    ) -> None:
        self.logger = logger
        self.handler = handler
        self.previous_level = previous_level
        self.previous_handlers = tuple(previous_handlers)
        self._lock = threading.Lock()
        self._detached = False

    @property
    def detached(self) -> bool:
        """Whether the logger has already been put back."""
        return self._detached

    def detach(self) -> None:
        """Put the logger back. Safe to call twice, and from either thread.

        A run can end in more ways than it has handlers for -- finished,
        failed, cancelled, or the panel closed under it -- so more than one of
        them calls this, and the second call has to be a no-op rather than a
        second restore of a stale snapshot.
        """
        with self._lock:
            if self._detached:
                return
            self._detached = True
        self.logger.handlers[:] = list(self.previous_handlers)
        self.logger.setLevel(self.previous_level)
        self.handler.close()

    def __enter__(self) -> "Attachment":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.detach()
        return False


def attach(
    log: RunLog,
    level: Optional[int] = None,
    logger_name: str = LOGGER_NAME,
) -> Attachment:
    """Send everything the library logs into *log*, until the run is over.

    *level* is what to capture, defaulting to whatever *log* is already set
    to, and *log* is set to it either way, so the two cannot silently disagree
    -- a handler at DEBUG feeding a buffer filtering at INFO would do the
    expensive half of the work and throw the result away.

    Any :class:`RunLogHandler` already on the logger comes off first, so a
    detach that never happened -- a run that ended in a way nobody handled,
    a panel closed mid-run -- costs one leaked handler and not a growing pile
    of them, each writing into a buffer nothing reads.
    """
    logger = logging.getLogger(logger_name)
    for stale in installed_handlers(logger):
        logger.removeHandler(stale)
        stale.close()

    # After the sweep, so a detach cannot put a leaked handler back.
    previous_level = logger.level
    previous_handlers = tuple(logger.handlers)

    wanted = log.level if level is None else level
    log.level = wanted
    handler = RunLogHandler(log, level=wanted)
    logger.addHandler(handler)
    # The whole point. In a napari session nothing has configured logging, so
    # this logger's effective level is root's WARNING and every INFO record is
    # dropped before any handler is consulted: adding a handler alone captures
    # nothing at all. `propagate` is left as it is, so a console handler
    # installed by the command line still receives the same records.
    logger.setLevel(wanted)
    return Attachment(logger, handler, previous_level, previous_handlers)
