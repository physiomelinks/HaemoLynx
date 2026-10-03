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
from typing import Optional, Sequence, Union

from .search import (
    _GUARD_PENALTY,
    _VOXEL_SCALED_SETTING_NAMES,
    AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS,
    GROUP_NAMES,
    OptimisationResult,
    TrialRecord,
)

#: A trial scoring at or above this counts as "rejected by an input-data
#: guard" for the report below -- comfortably under `_GUARD_PENALTY` itself
#: (1000) to allow for the underlying quality/topology term's own (possibly
#: negative) contribution, comfortably over any score a real, un-guarded
#: trial has ever been observed to reach.
_GUARD_REJECTED_SCORE_THRESHOLD = _GUARD_PENALTY / 2

#: The heading :func:`build_report_text` starts with unless told otherwise.
DEFAULT_REPORT_HEADING = "HaemoLynx settings, optimised from the segmented input image:"


def config_filename(input_path: Union[str, Path], now: Optional[datetime] = None) -> str:
    """``{date}_{time}_{input filename stem}.yaml``, e.g. ``20260910_143022_mouse_cortex.yaml``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = Path(input_path).stem
    return f"{stamp}_{stem}.yaml"


def format_seconds(seconds: float) -> str:
    """*seconds* as a person reads a duration: ``4.2 s``, ``38 s``,
    ``3 min 05 s``, ``1 h 02 min``."""
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _number(value: float) -> str:
    return f"{value:.3g}"


def _metric_comparison(before: Optional[TrialRecord], after: TrialRecord) -> str:
    """The score and each measurement *after* noted, beside *before*'s when
    there is one: ``score -0.91 -> -0.94, largest_fraction 0.91 -> 0.94``.
    A guard penalty is left out while neither trial paid one."""
    parts = []
    pairs: list[tuple[str, Optional[float], float]] = [
        ("score", before.score if before is not None else None, after.score)
    ]
    for name, value in after.metrics.items():
        pairs.append((name, before.metrics.get(name) if before is not None else None, value))
    for name, old, new in pairs:
        if name == "guard_penalty" and not new and not old:
            continue
        if old is None:
            parts.append(f"{name} {_number(new)}")
        else:
            parts.append(f"{name} {_number(old)} -> {_number(new)}")
    return ", ".join(parts)


def _changes(trials: Sequence[TrialRecord], *, search_grid_factor: int = 1) -> list[str]:
    """One line per sweep in *trials* that moved its setting: the value it
    started from, the one it chose, and the measurements behind the choice.
    On a downsampled search, a voxel-counted setting's values are the search
    grid's, and say so."""
    by_sweep: dict[int, list[TrialRecord]] = {}
    for trial in trials:
        if trial.sweep >= 0:
            by_sweep.setdefault(trial.sweep, []).append(trial)
    lines = []
    for sweep_trials in by_sweep.values():
        chosen = next((t for t in sweep_trials if t.chosen), None)
        if chosen is None or chosen.incumbent:
            continue
        incumbent = next((t for t in sweep_trials if t.incumbent), None)
        moved = (
            f"{incumbent.value!r} -> {chosen.value!r}" if incumbent is not None
            else f"set to {chosen.value!r}"
        )
        units = (
            " [search-grid voxels]"
            if search_grid_factor > 1 and chosen.setting in _VOXEL_SCALED_SETTING_NAMES
            else ""
        )
        lines.append(
            f"    {chosen.setting}: {moved}{units} ({_metric_comparison(incumbent, chosen)})"
        )
    return lines


def build_report_text(
    result: OptimisationResult,
    *,
    group_names: Sequence[str] = GROUP_NAMES,
    heading: str = DEFAULT_REPORT_HEADING,
) -> str:
    """One line per group: how many candidates were tried, what won, and how
    long it took -- then, under it, one line per setting a sweep moved, with
    the score and measurements it moved for (see :class:`.search.TrialRecord`).

    This is the text ``dump_config`` writes as a leading comment block on the
    optimised config file (see ``parsers.config.dump_config``'s use of
    ``schema.description``), and it is also shown as the GUI's status message.

    *group_names* is the full set of groups the search that produced *result*
    could have run -- :data:`.search.GROUP_NAMES` (the Skeletonise/Graph
    search) by default; :mod:`.fwhm_search` passes its own
    ``FWHM_GROUP_NAMES`` so "not optimised" is reported against the groups
    that search actually has, not this module's unrelated default; and
    *heading* its own first line.
    """
    lines = [heading]
    if result.seconds > 0:
        timing = f"Search took {format_seconds(result.seconds)}"
        if result.estimated_seconds is not None:
            timing += f" (Auto estimated {format_seconds(result.estimated_seconds)})"
        lines.append(timing + ".")
    if result.downsample_factor > 1:
        factors = tuple(getattr(result, "downsample_factors_zyx", ()) or ())
        per_axis = (
            f" (z, y, x reduced {factors[0]}x, {factors[1]}x, {factors[2]}x, so its voxels "
            "are as near the same size on every axis as the factors allow)"
            if len(factors) == 3 and len(set(factors)) > 1
            else ""
        )
        lines.append(
            f"Searched on a {result.downsample_factor}x downsampled copy for speed{per_axis}; "
            "voxel-based settings below are already scaled back up to the full-resolution grid."
        )
    if result.typical_radius_um is not None and result.resolution_cap is not None:
        lines.append(
            f"Auto measured a typical vessel radius of {result.typical_radius_um:.2f} um, so searched "
            f"no coarser than {result.resolution_cap}x, which keeps such a vessel at least "
            f"{2 * AUTO_MIN_VOXELS_ACROSS_TYPICAL_RADIUS:g} search-grid voxels across."
        )
    if result.finer_than_search_grid:
        lines.append(
            f"Under one voxel of the {result.downsample_factor}x search grid, so left as you had them: "
            + ", ".join(
                f"{name}={result.settings[name]!r}"
                for name in result.finer_than_search_grid
                if name in result.settings
            )
        )
    skipped_groups = [name for name in group_names if name not in result.groups_run]
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
        if not winners:
            # A joint sweep (centreline smoothing) tries several settings as
            # one value under a name of its own: say what it last chose.
            chosen = [trial for trial in trials if trial.chosen]
            if chosen:
                winners = f"{chosen[-1].setting}={chosen[-1].value!r}"
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
        group_seconds = result.group_seconds.get(group, sum(trial.seconds for trial in trials))
        time_note = f" (took {format_seconds(group_seconds)})" if group_seconds > 0 else ""
        lines.append(
            f"- {group}: tried {len(trials)} candidate(s){failure_note}{guard_note} -> {winners}"
            f"{time_note}"
        )
        lines.extend(_changes(trials, search_grid_factor=result.downsample_factor))

    return "\n".join(lines)
