"""What the panel's progress bars should read, worked out without a GUI.

The panel shows two bars: one across the run's eight stages, and one across the
steps of the stage that is running -- only graph building has any, and it is
the stage long enough to need them.

Turning the stream of :class:`~haemolynx.pipeline.progress.ProgressEvent`s into
what those bars say is ordinary logic, so it lives here rather than in
:mod:`haemolynx.gui._widget`, where it would need napari, a Qt binding and a
display to test. The widget's job is then only to copy :class:`BarState` onto a
QProgressBar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from haemolynx.pipeline.progress import (
    STAGE_FAILED,
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    STEP,
    ProgressEvent,
)

#: How many stages a run has; what the overall bar counts to.
TOTAL_STAGES = len([stage for stage in STAGES if stage.call])


@dataclass
class BarState:
    """One progress bar: how far along, out of how many, and what it says.

    ``total`` of 0 is Qt's own convention for "no end in sight": a bar with
    minimum and maximum both 0 animates rather than filling, which is what an
    unknown step count should look like.
    """

    value: int = 0
    total: int = 0
    text: str = ""
    visible: bool = False


@dataclass
class ProgressDisplay:
    """The two bars, kept up to date event by event.

    Nothing here needs the events to arrive in order or in full: a consumer
    that misses one still gets the right reading from the next, because every
    event carries the stage's own position rather than an increment.
    """

    stages: BarState = field(default_factory=BarState)
    steps: BarState = field(default_factory=BarState)

    def start(self) -> None:
        """A run is about to begin: show an empty bar rather than nothing."""
        self.stages = BarState(
            value=0, total=TOTAL_STAGES, text=f"Starting... 0/{TOTAL_STAGES}", visible=True
        )
        self.steps = BarState()

    def update(self, event: ProgressEvent) -> None:
        """Take in one event from the run."""
        if event.kind == STAGE_STARTED:
            self.stages = BarState(
                value=event.index,
                total=event.total,
                text=f"{event.title} ({event.index + 1}/{event.total})",
                visible=True,
            )
            self.steps = BarState()
        elif event.kind == STAGE_FINISHED:
            self.stages = BarState(
                value=event.completed,
                total=event.total,
                text=f"{event.title} ({event.completed}/{event.total})",
                visible=True,
            )
            self.steps = BarState()
        elif event.kind == STAGE_FAILED:
            self.stages = BarState(
                value=event.index,
                total=event.total,
                text=f"Failed at {event.title}",
                visible=True,
            )
            self.steps = BarState()
        elif event.kind == STEP:
            done = (event.step_index or 0) + 1
            total = event.step_total or 0
            of_total = f"/{total}" if total else ""
            self.steps = BarState(
                value=done,
                total=total,
                text=f"{event.step} ({done}{of_total})",
                visible=True,
            )

    def finish(self, message: str = "Finished") -> None:
        """The run is over: fill the bar in, and drop the step one."""
        self.stages = BarState(
            value=self.stages.total,
            total=self.stages.total,
            text=message,
            visible=True,
        )
        self.steps = BarState()

    def fail(self, message: str = "Failed") -> None:
        """The run stopped early: leave the bar where it got to, and say so."""
        self.stages = BarState(
            value=self.stages.value,
            total=self.stages.total,
            text=message,
            visible=True,
        )
        self.steps = BarState()
