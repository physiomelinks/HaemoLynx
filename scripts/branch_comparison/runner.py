"""Driving both sides of a comparison and collecting what they produced.

The reference branch is checked out into a throwaway ``git worktree`` and run
by a subprocess whose ``PYTHONPATH`` points at that checkout, because the
editable development install otherwise resolves ``haemolynx`` to whichever
working tree it was installed from -- which would make every comparison report
"no differences".
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import metrics, run_settings
from .report import ComparisonReport, PlotPair, SideReport

RUNNER_SCRIPT = Path(__file__).resolve().parent / "_run_side.py"

#: Suffixes worth showing side by side in the HTML report.
PLOT_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".html")

#: Shown first when both sides export them; anything else a branch writes is
#: discovered from disk and appended, so an export only one branch produces is
#: still reported.
PREFERRED_VTK_PARTS = ("vessels", "pericytes", "nodes")


class ComparisonError(RuntimeError):
    """Something went wrong before either pipeline could be run."""


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def git(*args: str, cwd: Path) -> str:
    """Run a git command, returning stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ComparisonError(
            f"git {' '.join(args)} failed in {cwd}:\n{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    """The working tree containing *start* (the current directory by default)."""
    start = Path(start or Path.cwd()).resolve()
    return Path(git("rev-parse", "--show-toplevel", cwd=start))


def current_ref(repo: Path) -> str:
    """The checked-out branch name, or the commit when detached."""
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    return branch if branch != "HEAD" else git("rev-parse", "--short", "HEAD", cwd=repo)


def commit_of(repo: Path, ref: str) -> str:
    return git("rev-parse", "--short", ref, cwd=repo)


@contextlib.contextmanager
def reference_worktree(repo: Path, ref: str, *, keep: bool = False) -> Iterator[Path]:
    """A detached worktree of *ref*, removed again on the way out."""
    try:
        commit_of(repo, ref)
    except ComparisonError as error:
        raise ComparisonError(
            f"Reference '{ref}' is not a ref in {repo}. Fetch it first, or pass "
            "an existing branch with --ref."
        ) from error

    parent = Path(tempfile.mkdtemp(prefix="imagelynx-compare-"))
    path = parent / "reference"
    git("worktree", "add", "--detach", str(path), ref, cwd=repo)
    try:
        yield path
    finally:
        if keep:
            print(f"Keeping reference worktree at {path}")
        else:
            with contextlib.suppress(ComparisonError):
                git("worktree", "remove", "--force", str(path), cwd=repo)
            shutil.rmtree(parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Running one side
# ---------------------------------------------------------------------------


@dataclass
class SidePaths:
    """Where one side's run puts everything."""

    label: str
    root: Path

    @property
    def plot_dir(self) -> Path:
        return self.root / "plots"

    @property
    def output_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def vtk_output_prefix(self) -> Path:
        return self.output_dir / "comparison_run"

    @property
    def graph_path(self) -> Path:
        return self.root / "final_graph.pkl"

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.json"

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    @property
    def log_path(self) -> Path:
        return self.root / "run.log"


def invoke_side(
    *,
    checkout: Path,
    paths: SidePaths,
    settings: dict[str, Any],
    required: Sequence[str],
    mode: str,
    python: str,
    timeout: float | None,
    allow_unapplied: bool,
    path_settings: Sequence[str] = (),
) -> dict[str, Any]:
    """Run ``_run_side.py`` against *checkout*, returning its result dict."""
    paths.root.mkdir(parents=True, exist_ok=True)
    spec = {
        "checkout": str(checkout),
        "mode": mode,
        "settings": settings,
        "required": list(required),
        "aliases": {
            name: list(run_settings.aliases_for(name)) for name in settings
        },
        "derived": list(run_settings.DERIVED_SETTINGS),
        "path_settings": list(path_settings),
        "allow_unapplied": allow_unapplied,
        "graph_capture_path": str(paths.graph_path),
        "result_path": str(paths.result_path),
    }
    paths.spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(checkout / "src"), str(checkout / "examples")]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    environment["MPLBACKEND"] = "Agg"
    environment["PYVISTA_OFF_SCREEN"] = "true"
    environment["PYTHONUNBUFFERED"] = "1"

    with paths.log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                [python, str(RUNNER_SCRIPT), str(paths.spec_path)],
                cwd=str(checkout),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Timed out after {timeout} s. See {paths.log_path}.",
            }

    if not paths.result_path.is_file():
        tail = _tail(paths.log_path)
        return {
            "ok": False,
            "error": (
                "The run produced no result file -- it died before it could "
                f"report. Last lines of {paths.log_path}:\n{tail}"
            ),
        }
    result = json.loads(paths.result_path.read_text(encoding="utf-8"))
    if not result.get("ok") and not result.get("error"):
        result["error"] = f"Run failed without a message. See {paths.log_path}."
    return result


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
    except OSError:
        return "(log unavailable)"


def resolve_pinned_settings(
    *,
    checkout: Path,
    paths: SidePaths,
    settings: dict[str, Any],
    required: Sequence[str],
    python: str,
    timeout: float | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Every setting the current checkout resolves to, for pinning both sides.

    Falls back to the explicit settings alone -- with a warning -- when the
    checkout has no resolver, which is the case for the older entry points.
    """
    result = invoke_side(
        checkout=checkout,
        paths=paths,
        settings=settings,
        required=required,
        mode="resolve",
        python=python,
        timeout=timeout,
        allow_unapplied=True,
    )
    if not result.get("ok"):
        return dict(settings), [
            "Could not read the current checkout's full settings, so only the "
            "settings named by the tool were pinned; anything else takes each "
            "branch's own default. Reason: "
            + (result.get("error") or "unknown").strip().splitlines()[-1]
        ], list(run_settings.PER_SIDE_SETTINGS)

    resolved = dict(result.get("resolved_settings") or {})
    for name in run_settings.PER_SIDE_SETTINGS:
        resolved.pop(name, None)
    pinned = {**resolved, **settings}
    return pinned, [], list(result.get("path_settings") or ())


# ---------------------------------------------------------------------------
# Collecting artefacts
# ---------------------------------------------------------------------------


@dataclass
class SideArtefacts:
    """What one finished run left on disk."""

    metrics: dict[str, Any] | None = None
    edge_attributes: dict[str, Any] | None = None
    stages: list[tuple[str, metrics.StageFingerprint]] | None = None
    statistics: dict[str, str] | None = None
    branch_statistics: dict[str, str] | None = None
    vtk: dict[str, dict[str, Any] | None] | None = None
    plots: dict[str, str] | None = None


def collect_side(paths: SidePaths, image_stem: str, *, root: Path) -> SideArtefacts:
    """Read back everything the comparison needs from one side's outputs."""
    artefacts = SideArtefacts()

    if paths.graph_path.is_file():
        graph = metrics.load_graph(paths.graph_path)
        artefacts.metrics = metrics.graph_metrics(graph)
        artefacts.edge_attributes = metrics.edge_attribute_summary(graph)

    artefacts.stages = metrics.fingerprint_stage_files(paths.output_dir, image_stem)
    artefacts.statistics = metrics.read_statistics_csv(
        paths.output_dir / f"{image_stem}_statistics.csv"
    )
    artefacts.branch_statistics = metrics.read_statistics_csv(
        paths.output_dir / f"{image_stem}_branch_statistics.csv"
    )
    artefacts.vtk = _collect_vtk(paths.vtk_output_prefix)
    artefacts.plots = _collect_plots(paths.plot_dir, root=root)
    return artefacts


def _collect_vtk(prefix: Path) -> dict[str, dict[str, Any] | None]:
    """Every ``{prefix}_*.vtp`` this run exported, keyed by what follows it."""
    summaries: dict[str, dict[str, Any] | None] = {}
    if not prefix.parent.is_dir():
        return summaries
    for path in sorted(prefix.parent.glob(f"{prefix.name}_*.vtp")):
        summaries[path.stem[len(prefix.name) + 1:]] = metrics.vtk_summary(path)
    return summaries


def ordered_vtk_parts(*sides: dict[str, Any] | None) -> list[str]:
    """The exports to report, preferred names first then whatever else exists."""
    found: set[str] = set()
    for side in sides:
        found |= set(side or {})
    ordered = [part for part in PREFERRED_VTK_PARTS if part in found]
    return ordered + sorted(found - set(ordered))


def _collect_plots(plot_dir: Path, *, root: Path) -> dict[str, str]:
    """``name -> path relative to the report`` for every plot in *plot_dir*."""
    plots: dict[str, str] = {}
    if not plot_dir.is_dir():
        return plots
    for path in sorted(plot_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in PLOT_SUFFIXES:
            name = path.relative_to(plot_dir).as_posix()
            plots[name] = path.relative_to(root).as_posix()
    return plots


# ---------------------------------------------------------------------------
# Assembling the report
# ---------------------------------------------------------------------------


def build_side_report(
    label: str,
    ref: str,
    commit: str,
    checkout: Path,
    paths: SidePaths,
    result: dict,
    required: Sequence[str] = (),
) -> SideReport:
    """One side's run, with unapplied settings split by how much they matter."""
    unapplied = dict(result.get("unapplied") or {})
    required = set(required)
    return SideReport(
        label=label,
        ref=ref,
        commit=commit,
        checkout=str(checkout),
        ok=bool(result.get("ok")),
        error=result.get("error"),
        runtime_seconds=result.get("runtime_seconds"),
        api_style=result.get("api_style", "unknown"),
        applied_settings=len(result.get("applied") or {}),
        unapplied_required={
            k: v for k, v in unapplied.items() if k in required
        },
        unapplied_optional={
            k: v for k, v in unapplied.items() if k not in required
        },
        final_graph_source=result.get("final_graph_source"),
        output_dir=str(paths.output_dir),
        plot_dir=str(paths.plot_dir),
    )


def build_report(
    *,
    current: SideReport,
    reference: SideReport,
    current_artefacts: SideArtefacts,
    reference_artefacts: SideArtefacts,
    image_path: str,
    warnings: Sequence[str] = (),
) -> ComparisonReport:
    """Everything the two sides produced, paired up."""
    report = ComparisonReport(
        current=current,
        reference=reference,
        image_path=image_path,
        warnings=list(warnings),
    )
    report.runtime_row = metrics.MetricRow(
        "Runtime",
        current.runtime_seconds,
        reference.runtime_seconds,
        unit="s",
        informational=True,
    )

    if report.failures:
        return report

    for label, side, artefacts in (
        (current.label, current, current_artefacts),
        (reference.label, reference, reference_artefacts),
    ):
        if artefacts.metrics is None:
            report.warnings.append(
                f"The {label} run finished but no final graph was captured, so "
                "its graph metrics and edge attributes are missing."
            )

    if current.final_graph_source != reference.final_graph_source:
        report.warnings.append(
            "The two sides' final graphs were captured at different points "
            f"({current.label}: {current.final_graph_source}; "
            f"{reference.label}: {reference.final_graph_source}). Branches that "
            "expose the finished graph differently are compared at the closest "
            "equivalent point, so treat edge-attribute differences as a "
            "starting point for investigation rather than a verdict."
        )

    report.metric_rows = metrics.compare_metrics(
        current_artefacts.metrics, reference_artefacts.metrics
    )
    report.stage_diffs = metrics.compare_stages(
        current_artefacts.stages or [], reference_artefacts.stages or []
    )
    report.first_stage_difference = metrics.first_differing_stage(report.stage_diffs)
    if metrics.stage_order_mismatch(
        current_artefacts.stages or [], reference_artefacts.stages or []
    ):
        report.warnings.append(
            "The two runs wrote their stage snapshots in different orders; the "
            "stage table follows the current run's order."
        )
    report.edge_attributes = metrics.compare_edge_attributes(
        current_artefacts.edge_attributes, reference_artefacts.edge_attributes
    )
    report.statistics_rows = metrics.compare_statistics_csv(
        current_artefacts.statistics, reference_artefacts.statistics
    )
    report.branch_statistics_rows = metrics.compare_statistics_csv(
        current_artefacts.branch_statistics, reference_artefacts.branch_statistics
    )
    report.vtk = [
        metrics.compare_vtk(
            part,
            (current_artefacts.vtk or {}).get(part),
            (reference_artefacts.vtk or {}).get(part),
        )
        for part in ordered_vtk_parts(current_artefacts.vtk, reference_artefacts.vtk)
    ]
    report.plots = pair_plots(
        current_artefacts.plots or {}, reference_artefacts.plots or {}
    )
    return report


def pair_plots(current: dict[str, str], reference: dict[str, str]) -> list[PlotPair]:
    """Every plot filename either run produced, in a stable order."""
    names = sorted(set(current) | set(reference))
    return [PlotPair(name, current.get(name), reference.get(name)) for name in names]
