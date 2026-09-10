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

from .search import OptimisationResult, TrialRecord


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
        lines.append(f"- {group}: tried {len(trials)} candidate(s){failure_note} -> {winners}")

    return "\n".join(lines)
