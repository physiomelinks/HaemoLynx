"""Progress events for :func:`haemolynx.optimisation.search.optimise_skeleton_and_graph_settings`.

Mirrors the shape of :mod:`haemolynx.pipeline.progress` -- a plain callback
taking one event -- but for the optimiser's own run, which is not a pipeline
run and has no ``Stage``/``STAGES`` of its own. Nothing here imports napari,
Qt, or the pipeline package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

#: A sweep over one (or, for centreline smoothing, one joint) setting started.
GROUP_STARTED = "group_started"
#: A sweep finished; its winning value(s) are in ``winner``.
GROUP_FINISHED = "group_finished"
#: One candidate within the current sweep was just scored.
CANDIDATE_EVALUATED = "candidate_evaluated"

KINDS: tuple[str, ...] = (GROUP_STARTED, GROUP_FINISHED, CANDIDATE_EVALUATED)


@dataclass(frozen=True)
class OptimisationEvent:
    """One point in a settings-optimisation run at which there is something new to show."""

    kind: str
    #: 0-based position of the current sweep among every sweep the run will do.
    group_index: int
    #: How many sweeps the run has -- an upper bound; guarded sweeps that are
    #: skipped (e.g. thickness gating on a mask with nothing fat) mean the
    #: real count can finish before this is reached.
    group_total: int
    #: A short name for the setting (or settings) this sweep is deciding.
    group_name: str
    #: For CANDIDATE_EVALUATED: 0-based position of the candidate just scored.
    candidate_index: Optional[int] = None
    #: For CANDIDATE_EVALUATED: how many candidates this sweep has.
    candidate_total: Optional[int] = None
    #: For GROUP_FINISHED: the setting(s) this sweep decided, name to value.
    winner: Optional[dict] = None


ProgressCallback = Callable[[OptimisationEvent], None]
