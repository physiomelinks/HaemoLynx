"""Config filename and a human-readable trial summary for an optimisation run.

Kept to plain Python (stdlib only) plus this package's own dataclasses -- no
``haemolynx.parsers``, ``haemolynx.pipeline``, or ``haemolynx.gui`` -- so
building the report never depends on the schema machinery that later writes it
into a YAML file's comment header. That happens in the GUI wiring, the one
layer that needs both this package and ``haemolynx.parsers``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .search import _GUARD_PENALTY, GROUP_NAMES, OptimisationResult, TrialRecord

#: A trial scoring at or above this counts as "rejected by an input-data
#: guard" for the report below -- comfortably under `_GUARD_PENALTY` itself
#: (1000) to allow for the underlying quality/topology term's own (possibly
#: negative) contribution, comfortably over any score a real, un-guarded
#: trial has ever been observed to reach.
_GUARD_REJECTED_SCORE_THRESHOLD = _GUARD_PENALTY / 2


def config_filename(input_path: Union[str, Path], now: Optional[datetime] = None) -> str:
    """``{date}_{time}_{input filename stem}.yaml``, e.g. ``20260910_143022_mouse_cortex.yaml``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = Path(input_path).stem
    return f"{stamp}_{stem}.yaml"


def build_report_text(result: OptimisationResult) -> str:
    """One line per sweep: how many candidates were tried, and what won.

    This is the text ``dump_config`` writes as a leading comment block on the
    optimised config file (see ``parsers.config.dump_config``'s use of
    ``schema.description``), and it is also shown as the GUI's status message.
    """
    lines = ["HaemoLynx settings, optimised from the segmented input image:"]
    if result.downsample_factor > 1:
        lines.append(
            f"Searched on a {result.downsample_factor}x downsampled copy for speed; "
            "voxel-based settings below are already scaled back up to the full-resolution grid."
        )
    skipped_groups = [name for name in GROUP_NAMES if name not in result.groups_run]
    if skipped_groups:
        lines.append(
            "Not optimised (left at their starting value): " + ", ".join(skipped_groups)
        )

    by_group: dict[str, list[TrialRecord]] = {}
    for trial in result.trials:
        by_group.setdefault(trial.group, []).append(trial)

    for group, trials in by_group.items():
        settings_tried = sorted({trial.setting for trial in trials})
        winners = ", ".join(
            f"{name}={result.settings[name]!r}" for name in settings_tried if name in result.settings
        )
        failures = sum(1 for trial in trials if trial.note.startswith("failed"))
        failure_note = f", {failures} failed" if failures else ""
        # A trial that raised is already counted above as "failed", not here
        # -- this counts a candidate that ran fine but was rejected by one
        # of the input-data-consistency guards (dropped real coverage of
        # the segmented mask, or -- for segmentation_cleanup with a raw
        # image supplied -- invented foreground the raw signal does not
        # support). See `search._Search._regression_penalty`.
        guarded = sum(
            1
            for trial in trials
            if not trial.note.startswith("failed") and trial.score >= _GUARD_REJECTED_SCORE_THRESHOLD
        )
        guard_note = f", {guarded} rejected by an input-data guard" if guarded else ""
        lines.append(
            f"- {group}: tried {len(trials)} candidate(s){failure_note}{guard_note} -> {winners}"
        )

    return "\n".join(lines)
