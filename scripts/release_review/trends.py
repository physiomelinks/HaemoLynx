"""Benchmarks over time: the rows each full run appends, a stage that slowed,
and how each stage grows with the volume.

``benchmarks.csv`` (``outputs/release_review/``) gets one row per stage of
every dataset run (:data:`~.manifests.BENCHMARK_COLUMNS`). ``speed.8b`` asks
for a stage more than 20% slower than at the last scored commit to be
flagged; ``speed.10a`` and ``memory.10a`` for time and memory against volume
at three or more sizes.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .manifests import BENCHMARK_COLUMNS
from .rubric import target

#: Stages faster than this are not compared: their times are mostly noise.
MIN_COMPARED_SECONDS = 1.0

#: A growth exponent within this of 1 reads as proportional to the volume:
#: timings on a shared workstation vary by more than that.
GROWTH_TOLERANCE = 0.1


def read_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append *rows*, writing the header first when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in BENCHMARK_COLUMNS})


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class Slowdown:
    dataset: str
    stage: str
    before_commit: str
    before_seconds: float
    seconds: float

    @property
    def ratio(self) -> float:
        return self.seconds / self.before_seconds


def slower_stages(
    rows: Sequence[Mapping[str, Any]],
    commit: str,
    *,
    threshold: Optional[float] = None,
    min_seconds: float = MIN_COMPARED_SECONDS,
) -> list[Slowdown]:
    """Each stage of *commit*'s runs more than *threshold* (``speed.8b``'s
    20%) slower than the same dataset's same stage at the latest earlier
    commit in *rows*. Rows are in the order they were appended; only finished
    stages are compared, and a commit's repeated runs by their fastest."""
    threshold = target("speed.8b") if threshold is None else threshold
    order: list[str] = []
    best: dict[tuple[str, str, str], float] = {}
    for row in rows:
        seconds = _float(row.get("wall_seconds"))
        if seconds is None or row.get("status") not in ("finished", "", None):
            continue
        key = (str(row["commit"]), str(row["dataset"]), str(row["stage"]))
        if row["commit"] not in order:
            order.append(str(row["commit"]))
        best[key] = min(seconds, best.get(key, math.inf))
    if commit not in order:
        return []
    earlier = order[: order.index(commit)]
    flagged = []
    for (row_commit, dataset, stage), seconds in best.items():
        if row_commit != commit:
            continue
        before = next(
            ((other, best[(other, dataset, stage)]) for other in reversed(earlier) if (other, dataset, stage) in best),
            None,
        )
        if before is None or before[1] < min_seconds:
            continue
        if seconds > before[1] * (1.0 + threshold):
            flagged.append(Slowdown(dataset, stage, before[0], before[1], seconds))
    return sorted(flagged, key=lambda each: -each.ratio)


def growth_exponent(points: Sequence[tuple[float, float]]) -> Optional[float]:
    """The slope of log(value) against log(volume), by least squares: 1 when
    a stage's time (or memory) grows in proportion to the volume, more when
    it grows faster. None with fewer than two distinct volumes."""
    usable = [(math.log(volume), math.log(value)) for volume, value in points if volume > 0 and value > 0]
    if len({x for x, _ in usable}) < 2:
        return None
    mean_x = sum(x for x, _ in usable) / len(usable)
    mean_y = sum(y for _, y in usable) / len(usable)
    spread = sum((x - mean_x) ** 2 for x, _ in usable)
    return sum((x - mean_x) * (y - mean_y) for x, y in usable) / spread


def scaling(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str], field: str = "wall_seconds") -> dict[str, Any]:
    """Each stage's growth exponent over *datasets* (one dataset cut to
    several sizes), from their rows' ``voxels`` and *field*."""
    points: dict[str, list[tuple[float, float]]] = {}
    sizes = set()
    for row in rows:
        if row.get("dataset") not in datasets or row.get("status") not in ("finished", "", None):
            continue
        voxels, value = _float(row.get("voxels")), _float(row.get(field))
        if voxels is None or value is None:
            continue
        sizes.add(voxels)
        points.setdefault(str(row["stage"]), []).append((voxels, value))
    exponents = {stage: growth_exponent(values) for stage, values in points.items()}
    faster = sorted(
        stage for stage, exponent in exponents.items() if exponent is not None and exponent > 1.0 + GROWTH_TOLERANCE
    )
    return {"sizes": len(sizes), "exponents": exponents, "faster_than_volume": faster}
