"""What the panel's "Optimise settings" progress bars should read, worked out
without a GUI.

Mirrors :mod:`haemolynx.gui.progress`'s shape (`BarState` / `ProgressDisplay`)
one-for-one, but consumes :class:`haemolynx.optimisation.progress.OptimisationEvent`
instead of a pipeline run's `ProgressEvent` -- kept separate rather than
generalising the pipeline-run bars, so their well-tested behaviour is
untouched. No napari, no Qt: the widget's job is only to copy this onto two
`QProgressBar`s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from haemolynx.optimisation.progress import (
    CANDIDATE_EVALUATED,
    GROUP_FINISHED,
    GROUP_STARTED,
    OptimisationEvent,
)


@dataclass
class OptimisationBarState:
    """One progress bar: how far along, out of how many, and what it says."""

    value: int = 0
    total: int = 0
    text: str = ""
    visible: bool = False


@dataclass
class OptimisationProgressDisplay:
    """The two bars an optimisation run shows, kept up to date event by event."""

    groups: OptimisationBarState = field(default_factory=OptimisationBarState)
    candidates: OptimisationBarState = field(default_factory=OptimisationBarState)

    def start(self) -> None:
        """A run is about to begin: show an empty bar rather than nothing."""
        self.groups = OptimisationBarState(value=0, total=0, text="Optimising settings...", visible=True)
        self.candidates = OptimisationBarState()

    def update(self, event: OptimisationEvent) -> None:
        """Take in one event from the run."""
        if event.kind == GROUP_STARTED:
            self.groups = OptimisationBarState(
                value=event.group_index,
                total=event.group_total,
                text=f"{event.group_name} ({event.group_index + 1}/{event.group_total})",
                visible=True,
            )
            self.candidates = OptimisationBarState()
        elif event.kind == GROUP_FINISHED:
            winner = event.winner or {}
            summary = ", ".join(f"{name}={value}" for name, value in winner.items())
            self.groups = OptimisationBarState(
                value=event.group_index + 1,
                total=event.group_total,
                text=f"{event.group_name}: {summary}" if summary else event.group_name,
                visible=True,
            )
            self.candidates = OptimisationBarState()
        elif event.kind == CANDIDATE_EVALUATED:
            done = (event.candidate_index or 0) + 1
            total = event.candidate_total or 0
            of_total = f"/{total}" if total else ""
            self.candidates = OptimisationBarState(
                value=done,
                total=total,
                text=f"{event.group_name} candidate {done}{of_total}",
                visible=True,
            )

    def finish(self, message: str = "Optimised") -> None:
        """The run is over: fill the bar in, and drop the candidate one."""
        self.groups = OptimisationBarState(
            value=max(self.groups.value, self.groups.total), total=self.groups.total, text=message, visible=True
        )
        self.candidates = OptimisationBarState()

    def fail(self, message: str = "Failed") -> None:
        """The run stopped early: leave the bar where it got to, and say so."""
        self.groups = OptimisationBarState(
            value=self.groups.value, total=self.groups.total, text=message, visible=True
        )
        self.candidates = OptimisationBarState()

    def reset(self) -> None:
        """Both bars back to nothing."""
        self.groups = OptimisationBarState()
        self.candidates = OptimisationBarState()
