"""Whether the panel has a run going, and how one is stopped on purpose.

The panel runs the pipeline on a napari ``thread_worker``, and used to keep no
record of it at all: "a run is in progress" *was* the Run button's own
``enabled`` flag, cleared only by that run's own success and failure handlers,
and the worker was a local variable nobody held on to. So anything that ended a
run another way left the panel unusable -- pressing "Clear layers" mid-run
pulled the layers out from under the run, and left the Run button greyed out,
the bars stopped part-way and no handle to cancel with. Closing the plugin and
opening it again was the only way back.

This is that missing record, and it needs no GUI: the guard, the worker to ask
to stop, and the two stateful things a run leaves behind -- the progress bars,
and the :class:`~haemolynx.gui.results.ResultLayers` that remembers the graph
from one stage to the next. Both are put back through a ``reset()`` method, so
a test can hand over the real, pure ones and the panel can hand over its Qt
wrapper.

**Cancelling is cooperative.** :func:`~haemolynx.pipeline.run_pipeline_stages`
takes no cancel argument, so a run is stopped where it already reports:
:meth:`RunState.check` raises :class:`RunCancelled` from the progress and
stage-output callbacks. Every one of those lands between stages, or between
graph building's eleven topology steps, so the run stops with nothing
half-written -- and it stops within a topology step of being asked, rather than
at the end of the stage.

The exception then arrives at the worker's ``errored`` signal like any other,
which is why a cancellation has to be told apart from a failure: it is not one,
and reporting it as ``RunCancelled: ...`` beside a stack trace would say the
run broke when the user stopped it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: What the report box says once a cancelled run has actually stopped.
CANCELLED = "Cancelled: the run was stopped when the layers were cleared. Ready."

#: What it says when the run beat the cancellation to the finish line. Rare,
#: and worth saying plainly rather than reporting either a finish or a cancel.
FINISHED_FIRST = "The run finished before it could be stopped. Its layers were cleared."

#: What the Run button says for itself when a run is already going.
ALREADY_RUNNING = (
    "A run is already going. Press 'Clear layers and state' to stop it."
)


class RunCancelled(Exception):
    """Raised inside a run to stop it, because the user asked it to stop.

    Deliberately not a subclass of anything the pipeline raises: it means the
    run was ended from outside, at a point of the run's own choosing, and
    everything that reports a run has to treat it as an intention rather than
    a fault.
    """


class RunState:
    """The panel's one record of the run it has going.

    Nothing here is Qt: *bars* and the per-run *results* are only ever asked
    to ``reset()``, so the pure :class:`~haemolynx.gui.progress.ProgressDisplay`
    and :class:`~haemolynx.gui.results.ResultLayers` serve a test exactly as
    the panel's own wrappers serve the panel.

    Each run owns a ``cancel_flag`` dict that its callbacks close over. Clear
    mutates that dict and then :meth:`supersede` frees the panel, so a dying
    worker cannot cancel or paint over the next run.
    """

    def __init__(self, bars: Any = None) -> None:
        #: The panel's progress bars, or the pure display underneath them.
        self.bars = bars
        self._worker: Any = None
        self._results: Any = None
        self._running = False
        self._flag: dict[str, bool] = {"cancelled": False}

    # -- what the panel asks ------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether a run is in flight, and so whether Run should refuse."""
        return self._running

    @property
    def cancelled(self) -> bool:
        """Whether the run in flight has been told to stop.

        Read on the GUI thread as well as the run's: events and layer groups
        emitted just before the cancel are still crossing between the two, and
        applying them would put back the layers that were just cleared.
        """
        return bool(self._flag.get("cancelled"))

    @property
    def cancel_flag(self) -> dict[str, bool]:
        """The dict this run's callbacks close over. A new run gets a new one."""
        return self._flag

    @property
    def worker(self) -> Any:
        """The worker running the pipeline, while there is one."""
        return self._worker

    @property
    def results(self) -> Any:
        """What the run in flight is turning its stages into."""
        return self._results

    # -- the run's own lifecycle -------------------------------------------

    def start(
        self,
        worker: Any = None,
        results: Any = None,
        cancel_flag: dict[str, bool] | None = None,
    ) -> dict[str, bool]:
        """A run has just been handed to *worker*.

        Returns the cancel flag that run's callbacks must close over, so a
        later Clear cannot make a dying worker look like the next run.
        Pass *cancel_flag* when the callbacks have already closed over one.
        """
        self._flag = cancel_flag if cancel_flag is not None else {"cancelled": False}
        self._flag["cancelled"] = False
        self._worker = worker
        self._results = results
        self._running = True
        return self._flag

    def stopped(self) -> None:
        """The run has ended, however it ended: the panel is free again.

        ``cancelled`` is deliberately left set until the next run starts, so a
        straggling event that arrives after the worker has gone is still
        ignored.
        """
        self._running = False
        self._worker = None
        self._results = None

    def supersede(self) -> None:
        """Free the panel while a cancelled worker is still winding down.

        The dying run keeps its own ``cancel_flag``; the next :meth:`start`
        installs a new one. Run pipeline can be pressed immediately.
        """
        if self._running:
            self._flag["cancelled"] = True
        self._running = False
        self._worker = None
        self._results = None

    # -- stopping one on purpose -------------------------------------------

    def cancel(self) -> bool:
        """Ask the run to stop, and forget what it has drawn so far.

        Returns whether there was a run to stop, which is what decides whether
        the user is told about one: pressing "Clear layers" with nothing going
        must behave exactly as it always did.

        The worker's own ``quit()`` is called as well as the flag being set. It
        is napari's abort mechanism, it is what a generator worker would stop
        at, and it makes ``worker.abort_requested`` true for anything else
        watching. It does not by itself interrupt a plain function worker,
        which is what :meth:`check` is for.
        """
        if not self._running:
            return False
        self._flag["cancelled"] = True
        # Before the worker has stopped, on purpose: the graph this remembers
        # is what a stage drawn after the cancel would be drawn against.
        _reset(self._results)
        _reset(self.bars)
        quit_ = getattr(self._worker, "quit", None)
        if callable(quit_):
            try:
                quit_()
            except Exception:  # noqa: BLE001 - a worker already gone is not a problem
                logger.debug("could not ask the run's worker to quit", exc_info=True)
        return True

    def check(self, flag: dict[str, bool] | None = None) -> None:
        """Stop the run here if it has been cancelled. Runs on the run's thread.

        Called from the callbacks the pipeline already offers, every one of
        which lands between stages or between topology steps. Pass the flag
        :meth:`start` returned so a superseded run still stops itself.
        """
        owned = flag if flag is not None else self._flag
        if owned.get("cancelled"):
            raise RunCancelled(CANCELLED)


def clear_message(
    removed: int,
    stopping: bool,
    *,
    discarded_artefacts: bool = False,
    restored_skips: bool = False,
) -> str:
    """What the report box says when "Clear layers and state" has been pressed."""
    note = f"Removed {removed} HaemoLynx layer(s)."
    if stopping:
        note += " Stopping the run; the panel is ready for another run."
    if discarded_artefacts:
        note += " Discarded cached checkpoint and resume pickles."
    if restored_skips:
        note += " Restored skeletonize and graph-building toggles."
    return note


def _reset(thing: Optional[Any]) -> None:
    """Put *thing* back to how it started, if it knows how."""
    reset = getattr(thing, "reset", None)
    if callable(reset):
        reset()
