"""Turning two pipeline runs' artefacts into comparable numbers.

Nothing here runs a pipeline, spawns a worktree, or touches git. Every function
takes artefacts (graph pickles, statistics CSVs, VTK files) or already-loaded
summaries and returns plain data, so the comparison logic can be unit-tested in
milliseconds instead of behind the 15-minute run the tool exists to drive.
"""
from __future__ import annotations

import csv
import hashlib
import math
import pickle
import statistics as _stats
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

#: Two runs of the same code are bit-identical, so anything above pure float
#: repr noise is a real difference worth showing a reviewer.
DEFAULT_REL_TOL = 1e-12

#: Written into every statistics CSV at export time; never a code difference.
IGNORED_STATISTICS_METRICS = ("Exported At (UTC)",)

#: ``{image_stem}_graph_after_{step}.pkl`` -- one per graph-building step.
STAGE_SNAPSHOT_INFIX = "_graph_after_"


# ---------------------------------------------------------------------------
# Metric rows: one number from each side, plus whether they differ
# ---------------------------------------------------------------------------


def _both_nan(a: float | None, b: float | None) -> bool:
    return (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    )


@dataclass(frozen=True)
class MetricRow:
    """One quantity measured on both sides.

    ``current`` or ``reference`` is ``None`` when that side did not produce the
    quantity at all -- a missing attribute is itself a difference, so it is
    reported rather than dropped.
    """

    name: str
    current: float | str | None
    reference: float | str | None
    unit: str = ""
    rel_tol: float = DEFAULT_REL_TOL
    #: Expected to vary between runs (wall-clock time), so shown but never
    #: highlighted as a change in behaviour.
    informational: bool = False

    @property
    def delta(self) -> float | None:
        if not self._both_numeric():
            return None
        return float(self.current) - float(self.reference)  # type: ignore[arg-type]

    @property
    def percent_change(self) -> float | None:
        delta = self.delta
        if delta is None or not self.reference:
            return None
        return 100.0 * delta / float(self.reference)  # type: ignore[arg-type]

    @property
    def differs(self) -> bool:
        if self.current is None or self.reference is None:
            return not (self.current is None and self.reference is None)
        if _both_nan(self.current, self.reference):  # type: ignore[arg-type]
            return False
        if self._both_numeric():
            return not math.isclose(
                float(self.current),  # type: ignore[arg-type]
                float(self.reference),  # type: ignore[arg-type]
                rel_tol=self.rel_tol,
                abs_tol=0.0,
            )
        return str(self.current) != str(self.reference)

    @property
    def flagged(self) -> bool:
        """Whether the report should call this row out as a change."""
        return self.differs and not self.informational

    def _both_numeric(self) -> bool:
        return isinstance(self.current, (int, float)) and isinstance(
            self.reference, (int, float)
        )


def any_differences(rows: Iterable[MetricRow]) -> bool:
    """True when at least one row is a real difference between the sides."""
    return any(row.flagged for row in rows)


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

#: (key, human label, unit) -- the order they appear in the report.
METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("nodes", "Nodes", ""),
    ("edges", "Edges", ""),
    ("total_edge_length", "Total edge length", "um"),
    ("mean_edge_length", "Mean edge length", "um"),
    ("median_edge_length", "Median edge length", "um"),
    ("max_edge_length", "Max edge length", "um"),
    ("branching_points", "Branching points (degree >= 3)", ""),
    ("average_degree", "Average degree", ""),
)

_METRIC_UNITS = {key: unit for key, _, unit in METRIC_SPECS}
_METRIC_LABELS = {key: label for key, label, _ in METRIC_SPECS}


def load_graph(path: Path | str) -> nx.Graph:
    """Read a graph pickle written by the pipeline."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def edge_lengths(graph: nx.Graph) -> list[float]:
    """Every finite ``length`` edge attribute, in microns."""
    lengths: list[float] = []
    for *_, data in graph.edges(data=True):
        value = data.get("length")
        if isinstance(value, (int, float)) and math.isfinite(value):
            lengths.append(float(value))
    return lengths


def graph_metrics(graph: nx.Graph) -> dict[str, float | None]:
    """The headline numbers a reviewer compares between two runs.

    Length metrics are ``None`` when no edge carries a ``length``, rather than
    a misleading zero.
    """
    lengths = edge_lengths(graph)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "total_edge_length": math.fsum(lengths) if lengths else None,
        "mean_edge_length": _stats.fmean(lengths) if lengths else None,
        "median_edge_length": _stats.median(lengths) if lengths else None,
        "max_edge_length": max(lengths) if lengths else None,
        "branching_points": sum(1 for _, deg in graph.degree() if deg >= 3),
        "average_degree": (2.0 * n_edges / n_nodes) if n_nodes else None,
    }


def compare_metrics(
    current: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    *,
    specs: Sequence[tuple[str, str, str]] = METRIC_SPECS,
    rel_tol: float = DEFAULT_REL_TOL,
) -> list[MetricRow]:
    """Pair up two metric dicts, in ``specs`` order then any extras."""
    current = dict(current or {})
    reference = dict(reference or {})
    ordered = [key for key, _, _ in specs]
    extras = [k for k in list(current) + list(reference) if k not in ordered]
    seen: set[str] = set()
    keys: list[str] = []
    for key in ordered + extras:
        if key not in seen and (key in current or key in reference):
            seen.add(key)
            keys.append(key)
    return [
        MetricRow(
            name=_METRIC_LABELS.get(key, key),
            current=current.get(key),
            reference=reference.get(key),
            unit=_METRIC_UNITS.get(key, ""),
            rel_tol=rel_tol,
        )
        for key in keys
    ]


# ---------------------------------------------------------------------------
# Per-stage divergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageFingerprint:
    """A stage's graph, reduced to something comparable across processes.

    Node ids are assigned by the build and carry no meaning, so the geometry is
    hashed instead: the sorted rounded node positions and the sorted rounded
    edge lengths.
    """

    nodes: int
    edges: int
    position_digest: str
    length_digest: str

    def differences(self, other: "StageFingerprint") -> list[str]:
        """Which parts of the fingerprint disagree, in reporting order."""
        reasons = []
        if self.nodes != other.nodes:
            reasons.append(f"nodes {self.nodes} vs {other.nodes}")
        if self.edges != other.edges:
            reasons.append(f"edges {self.edges} vs {other.edges}")
        if self.position_digest != other.position_digest:
            reasons.append("node positions")
        if self.length_digest != other.length_digest:
            reasons.append("edge lengths")
        return reasons


def _digest(values: Iterable[Any]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(repr(value).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def graph_fingerprint(graph: nx.Graph, *, decimals: int = 6) -> StageFingerprint:
    """Reduce a graph to counts plus geometry digests."""
    positions = []
    for _, data in graph.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            positions.append(None)
            continue
        positions.append(tuple(round(float(c), decimals) for c in tuple(pos)))
    lengths = sorted(round(value, decimals) for value in edge_lengths(graph))
    return StageFingerprint(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        position_digest=_digest(sorted(positions, key=repr)),
        length_digest=_digest(lengths),
    )


@dataclass(frozen=True)
class StageDiff:
    """One graph-building step, on both sides."""

    label: str
    status: str  # "same" | "differs" | "only_current" | "only_reference"
    reasons: tuple[str, ...] = ()

    @property
    def differs(self) -> bool:
        return self.status != "same"


def discover_stage_snapshots(output_dir: Path | str, image_stem: str) -> list[Path]:
    """The ``*_graph_after_*.pkl`` files, in the order the run wrote them.

    The filenames carry no ordinal, so write order (mtime) is what puts the
    steps in pipeline order; ties break on name for determinism.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    pattern = f"{image_stem}{STAGE_SNAPSHOT_INFIX}*.pkl"
    return sorted(output_dir.glob(pattern), key=lambda p: (p.stat().st_mtime_ns, p.name))


def stage_label(path: Path | str, image_stem: str) -> str:
    """The step name embedded in a snapshot filename."""
    name = Path(path).stem
    prefix = f"{image_stem}{STAGE_SNAPSHOT_INFIX}"
    return name[len(prefix):] if name.startswith(prefix) else name


def fingerprint_stage_files(
    output_dir: Path | str, image_stem: str
) -> list[tuple[str, StageFingerprint]]:
    """Label plus fingerprint for every stage snapshot a run left behind."""
    result = []
    for path in discover_stage_snapshots(output_dir, image_stem):
        result.append((stage_label(path, image_stem), graph_fingerprint(load_graph(path))))
    return result


def compare_stages(
    current: Sequence[tuple[str, StageFingerprint]],
    reference: Sequence[tuple[str, StageFingerprint]],
) -> list[StageDiff]:
    """Line the two stage sequences up by label, keeping the current order.

    Steps only the reference ran are appended, so a stage removed by the branch
    under review still shows up.
    """
    current_map = dict(current)
    reference_map = dict(reference)
    labels = [label for label, _ in current]
    labels += [label for label, _ in reference if label not in current_map]

    diffs: list[StageDiff] = []
    for label in labels:
        cur = current_map.get(label)
        ref = reference_map.get(label)
        if cur is None:
            diffs.append(StageDiff(label, "only_reference"))
        elif ref is None:
            diffs.append(StageDiff(label, "only_current"))
        else:
            reasons = cur.differences(ref)
            diffs.append(
                StageDiff(label, "differs" if reasons else "same", tuple(reasons))
            )
    return diffs


def first_differing_stage(diffs: Sequence[StageDiff]) -> StageDiff | None:
    """The earliest step whose graphs disagree -- where a regression starts."""
    for diff in diffs:
        if diff.differs:
            return diff
    return None


def stage_order_mismatch(
    current: Sequence[tuple[str, StageFingerprint]],
    reference: Sequence[tuple[str, StageFingerprint]],
) -> bool:
    """True when the shared steps ran in a different order on the two sides."""
    shared = {label for label, _ in current} & {label for label, _ in reference}
    return [l for l, _ in current if l in shared] != [
        l for l, _ in reference if l in shared
    ]


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

_EDGE_ATTRIBUTE_STATS = ("count", "mean", "min", "max")


def edge_attribute_summary(graph: nx.Graph) -> dict[str, dict[str, Any]]:
    """Which attributes the edges carry, and summary stats for numeric ones."""
    values: dict[str, list[Any]] = {}
    for *_, data in graph.edges(data=True):
        for key, value in data.items():
            values.setdefault(key, []).append(value)

    summary: dict[str, dict[str, Any]] = {}
    for key, raw in sorted(values.items()):
        numeric = [
            float(v)
            for v in raw
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
        ]
        entry: dict[str, Any] = {"count": len(raw), "numeric": bool(numeric)}
        if numeric:
            entry.update(
                mean=_stats.fmean(numeric), min=min(numeric), max=max(numeric)
            )
        summary[key] = entry
    return summary


@dataclass
class EdgeAttributeComparison:
    only_current: list[str] = field(default_factory=list)
    only_reference: list[str] = field(default_factory=list)
    shared_rows: list[MetricRow] = field(default_factory=list)

    @property
    def differs(self) -> bool:
        return bool(self.only_current or self.only_reference) or any_differences(
            self.shared_rows
        )


def compare_edge_attributes(
    current: Mapping[str, Mapping[str, Any]] | None,
    reference: Mapping[str, Mapping[str, Any]] | None,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
) -> EdgeAttributeComparison:
    """Attribute names on each side, and stats for the ones both produced."""
    current = dict(current or {})
    reference = dict(reference or {})
    comparison = EdgeAttributeComparison(
        only_current=sorted(set(current) - set(reference)),
        only_reference=sorted(set(reference) - set(current)),
    )
    for name in sorted(set(current) & set(reference)):
        cur, ref = current[name], reference[name]
        for stat in _EDGE_ATTRIBUTE_STATS:
            if stat not in cur and stat not in ref:
                continue
            comparison.shared_rows.append(
                MetricRow(
                    name=f"{name}.{stat}",
                    current=cur.get(stat),
                    reference=ref.get(stat),
                    rel_tol=rel_tol,
                )
            )
    return comparison


# ---------------------------------------------------------------------------
# Statistics CSVs
# ---------------------------------------------------------------------------


def read_statistics_csv(
    path: Path | str,
    *,
    ignored_metrics: Sequence[str] = IGNORED_STATISTICS_METRICS,
) -> dict[str, str]:
    """``"Section / Metric" -> value`` from a statistics CSV.

    ``ignored_metrics`` drops rows that are stamped per run (the export
    timestamp) and would otherwise report a difference on every comparison.
    """
    path = Path(path)
    rows: dict[str, str] = {}
    if not path.is_file():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if len(row) < 3:
                continue
            section, metric, value = row[0], row[1], row[2]
            if metric in ignored_metrics:
                continue
            unit = row[3] if len(row) > 3 else ""
            key = f"{section} / {metric}" + (f" [{unit}]" if unit else "")
            rows[key] = value
    return rows


def _as_float(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def compare_statistics_csv(
    current: Mapping[str, str] | None,
    reference: Mapping[str, str] | None,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
) -> list[MetricRow]:
    """One row per metric, comparing numerically where both sides parse."""
    current = dict(current or {})
    reference = dict(reference or {})
    keys = list(current)
    keys += [key for key in reference if key not in current]
    rows = []
    for key in keys:
        cur_raw, ref_raw = current.get(key), reference.get(key)
        cur_num, ref_num = _as_float(cur_raw), _as_float(ref_raw)
        if cur_num is not None and ref_num is not None:
            rows.append(MetricRow(key, cur_num, ref_num, rel_tol=rel_tol))
        else:
            rows.append(MetricRow(key, cur_raw, ref_raw, rel_tol=rel_tol))
    return rows


# ---------------------------------------------------------------------------
# VTK exports
# ---------------------------------------------------------------------------


def vtk_summary(path: Path | str) -> dict[str, Any] | None:
    """Cell/point counts and cell-array stats for one ``.vtp`` file.

    ``None`` when the file is missing or unreadable, so a partial export is
    visible in the report rather than mistaken for an empty mesh.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        import pyvista as pv
    except ImportError:
        return None
    try:
        mesh = pv.read(str(path))
    except Exception:  # noqa: BLE001 - an unreadable export is a finding, not a crash
        return None

    arrays: dict[str, dict[str, Any]] = {}
    for name in mesh.cell_data.keys():
        values = mesh.cell_data[name]
        entry: dict[str, Any] = {"n": int(len(values))}
        try:
            numeric = [float(v) for v in values]
        except (TypeError, ValueError):
            numeric = []
        finite = [v for v in numeric if math.isfinite(v)]
        if finite:
            entry.update(mean=_stats.fmean(finite), min=min(finite), max=max(finite))
        arrays[name] = entry
    return {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "cell_arrays": arrays,
    }


@dataclass
class VtkComparison:
    name: str
    count_rows: list[MetricRow] = field(default_factory=list)
    only_current: list[str] = field(default_factory=list)
    only_reference: list[str] = field(default_factory=list)
    shared_rows: list[MetricRow] = field(default_factory=list)
    missing: tuple[bool, bool] = (False, False)

    @property
    def differs(self) -> bool:
        return (
            bool(self.only_current or self.only_reference)
            or any_differences(self.count_rows)
            or any_differences(self.shared_rows)
        )


def compare_vtk(
    name: str,
    current: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
) -> VtkComparison:
    """Compare one exported mesh: counts, then per-cell-array statistics."""
    comparison = VtkComparison(
        name=name, missing=(current is None, reference is None)
    )
    cur = dict(current or {})
    ref = dict(reference or {})
    for key in ("n_cells", "n_points"):
        comparison.count_rows.append(
            MetricRow(f"{name}.{key}", cur.get(key), ref.get(key), rel_tol=rel_tol)
        )
    cur_arrays = dict(cur.get("cell_arrays") or {})
    ref_arrays = dict(ref.get("cell_arrays") or {})
    comparison.only_current = sorted(set(cur_arrays) - set(ref_arrays))
    comparison.only_reference = sorted(set(ref_arrays) - set(cur_arrays))
    for array in sorted(set(cur_arrays) & set(ref_arrays)):
        for stat in ("n", "mean", "min", "max"):
            if stat not in cur_arrays[array] and stat not in ref_arrays[array]:
                continue
            comparison.shared_rows.append(
                MetricRow(
                    f"{name}.{array}.{stat}",
                    cur_arrays[array].get(stat),
                    ref_arrays[array].get(stat),
                    rel_tol=rel_tol,
                )
            )
    return comparison
