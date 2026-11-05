"""Stage times and peak memory, read from each run's manifest.

``PLAN_release_review.md`` step R2 asked for a ``{stem}_run_timings.csv``
beside every run. What landed instead (2026-10-15, ``9b679b84``) is
``{stem}_run_manifest.json`` (``haemolynx.pipeline.provenance``), which holds
each stage's start, end, ``wall_seconds`` and peak memory with how it was
measured, beside the settings, version and input checksums. The scorer reads
the manifests rather than a separate file, and keeps its own per-run rows in
``benchmarks.csv`` (:mod:`.trends`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

MANIFEST_SUFFIX = "_run_manifest.json"

#: The columns of one benchmark row: a stage of a dataset's run at a commit.
BENCHMARK_COLUMNS = (
    "date", "commit", "dataset", "stage", "status", "wall_seconds", "peak_rss_mib",
    "peak_rss_method", "voxels", "machine_ram_gib", "platform",
)

_MIB = 2**20


@dataclass(frozen=True)
class StageRecord:
    stage: str
    status: str
    wall_seconds: Optional[float]
    peak_rss_bytes: Optional[int]
    peak_rss_method: Optional[str]
    resumed: bool = False


@dataclass(frozen=True)
class RunRecord:
    """One run, as its manifest records it."""

    path: str
    status: str
    wall_seconds: Optional[float]
    peak_rss_bytes: Optional[int]
    stages: tuple[StageRecord, ...]
    version: Optional[str]
    commit: Optional[str]
    error: Optional[Mapping[str, Any]]
    inputs: tuple[Mapping[str, Any], ...]
    platform: Optional[str]
    machine_ram_bytes: Optional[int]
    has_settings: bool

    @property
    def finished(self) -> bool:
        return self.status == "finished"

    def stage(self, name: str) -> Optional[StageRecord]:
        return next((each for each in self.stages if each.stage == name), None)


def parse(manifest: Mapping[str, Any], path: str = "") -> RunRecord:
    stages = tuple(
        StageRecord(
            stage=record.get("stage", "?"),
            status=record.get("status", "?"),
            wall_seconds=record.get("wall_seconds"),
            peak_rss_bytes=record.get("peak_rss_bytes"),
            peak_rss_method=record.get("peak_rss_method"),
            resumed=bool(record.get("resumed")),
        )
        for record in manifest.get("stages") or ()
    )
    git = manifest.get("git") or {}
    machine = manifest.get("platform") if isinstance(manifest.get("platform"), Mapping) else {}
    return RunRecord(
        path=path,
        status=manifest.get("status", "?"),
        wall_seconds=manifest.get("wall_seconds"),
        peak_rss_bytes=manifest.get("peak_rss_bytes"),
        stages=stages,
        version=manifest.get("haemolynx_version"),
        commit=git.get("commit") if isinstance(git, Mapping) else None,
        error=manifest.get("error"),
        inputs=tuple(manifest.get("inputs") or ()),
        platform=machine.get("system"),
        machine_ram_bytes=machine.get("total_ram_bytes"),
        has_settings=bool(manifest.get("settings")),
    )


def read(path: Path) -> RunRecord:
    path = Path(path)
    return parse(json.loads(path.read_text(encoding="utf-8")), path.as_posix())


def find(folder: Path) -> list[Path]:
    """Every manifest under *folder*."""
    return sorted(Path(folder).rglob(f"*{MANIFEST_SUFFIX}"))


def _mib(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(value / _MIB, 1)


def benchmark_rows(
    run: RunRecord, *, dataset: str, commit: str, date: str, voxels: Optional[int] = None,
) -> list[dict[str, Any]]:
    """One :data:`BENCHMARK_COLUMNS` row per stage the run recorded, and one
    for the whole run (stage ``total``)."""
    ram = None if run.machine_ram_bytes is None else round(run.machine_ram_bytes / 2**30, 1)
    common = {"date": date, "commit": commit, "dataset": dataset, "voxels": voxels,
              "machine_ram_gib": ram, "platform": run.platform}
    rows = [
        {**common, "stage": stage.stage, "status": stage.status, "wall_seconds": stage.wall_seconds,
         "peak_rss_mib": _mib(stage.peak_rss_bytes), "peak_rss_method": stage.peak_rss_method}
        for stage in run.stages if not stage.resumed
    ]
    rows.append({**common, "stage": "total", "status": run.status, "wall_seconds": run.wall_seconds,
                 "peak_rss_mib": _mib(run.peak_rss_bytes), "peak_rss_method": "process_peak"})
    return rows


def _seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.1f}" if value < 100 else f"{value:,.0f}"


def stage_table(run: RunRecord) -> str:
    """The run's stages as a Markdown table: time, peak memory and how the
    peak was read (``process_peak`` exact; ``sampled`` a lower bound)."""
    lines = ["| Stage | Status | Wall (s) | Peak RSS (MiB) | Peak measured by |", "|---|---|---:|---:|---|"]
    for stage in run.stages:
        lines.append(
            f"| {stage.stage}{' (resumed)' if stage.resumed else ''} | {stage.status} | {_seconds(stage.wall_seconds)} | "
            f"{_mib(stage.peak_rss_bytes) if stage.peak_rss_bytes is not None else '-'} | {stage.peak_rss_method or '-'} |"
        )
    lines.append(f"| **whole run** | {run.status} | {_seconds(run.wall_seconds)} | {_mib(run.peak_rss_bytes) or '-'} | process peak |")
    return "\n".join(lines)


def per_dataset(
    runs: Mapping[str, Optional[RunRecord]], reference: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Which reference datasets have a finished run with every stage timed
    and its peak memory recorded, and which do not."""
    measured, missing = [], []
    for name in reference:
        run = runs.get(name)
        complete = (
            run is not None and run.finished and run.stages
            and all(stage.wall_seconds is not None and stage.peak_rss_bytes is not None for stage in run.stages)
        )
        (measured if complete else missing).append(name)
    return measured, missing


def solved_share(edges: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The share of centreline length with a solved flow, and the rest by
    why it carries none (``flow_state``): what ``flow.6c`` records per
    dataset. Each edge is its attribute mapping."""
    total = solved = 0.0
    unsolved: dict[str, float] = {}
    vessels = unsolved_vessels = 0
    for data in edges:
        length = float(data.get("length") or 0.0)
        total += length
        vessels += 1
        # The solve writes flow_state on every vessel; none means it never ran.
        state = data.get("flow_state")
        if state == "solved":
            solved += length
        else:
            unsolved_vessels += 1
            state = str(state or "not solved")
            unsolved[state] = unsolved.get(state, 0.0) + length
    return {
        "length_um": round(total, 1),
        "solved_um": round(solved, 1),
        "solved_fraction": round(solved / total, 4) if total else None,
        "vessels": vessels,
        "unsolved_vessels": unsolved_vessels,
        "unsolved_um_by_state": {state: round(value, 1) for state, value in sorted(unsolved.items())},
    }
