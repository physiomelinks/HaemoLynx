"""Running the tiers: the slow, impure part of the scorer.

A scoring measures one commit. By default the runner exports it with
``git archive`` into the report folder and measures that copy, with
``PYTHONPATH`` at its ``src``: peer sessions edit the live tree mid-run, and
the editable install would otherwise import whatever the main checkout holds
(``CLAUDE.md`` § Release review). ``--tree`` measures a checkout as it is
instead, which is what CI does with its clean one.

Reports go to the main checkout's ``outputs/release_review/<date>_<sha>/``
(git-ignored), not a worktree's: the evidence of 2026-10-15 was deleted with
the worktree that measured it.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import (
    edge_cases,
    evidence,
    exports,
    fixtures,
    full,
    graph_metrics,
    human,
    manifests,
    phantoms,
    quick,
    report,
    review_doc,
    scorecard,
    trends,
)
from .rubric import FIXTURES, FULL, QUICK

SCORER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCORER_DIR.parent
HUMAN_CHECKS = SCORER_DIR / "human_checks.yaml"


class ScoringError(RuntimeError):
    """The scoring could not start: no such commit, no git, no document."""


# ---------------------------------------------------------------------------
# git and the tree being scored
# ---------------------------------------------------------------------------


def git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ScoringError(f"git {' '.join(args)} failed in {cwd}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def main_checkout(repo: Path) -> Path:
    """The main working tree of *repo*'s repository, from any worktree."""
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo))
    return common.parent


def worktrees(repo: Path) -> list[Path]:
    listed = git("worktree", "list", "--porcelain", cwd=repo)
    return [Path(line.split(" ", 1)[1]) for line in listed.splitlines() if line.startswith("worktree ")]


def archive(repo: Path, commit: str, destination: Path) -> Path:
    """``git archive`` of *commit*, unpacked into *destination*."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    data = subprocess.run(["git", "archive", "--format=tar", commit], cwd=str(repo), capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        bundle.extractall(destination)
    return destination


def tracked_root_files(tree: Path, is_checkout: bool) -> list[str]:
    if is_checkout:
        return sorted(name for name in git("ls-files", cwd=tree).splitlines() if "/" not in name)
    return sorted(path.name for path in tree.iterdir() if path.is_file())


# ---------------------------------------------------------------------------
# Subprocesses
# ---------------------------------------------------------------------------


def _environment(pythonpath: Sequence[Path], extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env


def run_python(
    arguments: Sequence[str], *, cwd: Path, pythonpath: Sequence[Path], log: Path,
    timeout: Optional[float] = None, extra_env: Optional[Mapping[str, str]] = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, *arguments], cwd=str(cwd), env=_environment(pythonpath, extra_env),
            stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    return completed.returncode


def free_ram_bytes() -> Optional[int]:
    """Memory available now, read without psutil."""
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
        return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# The quick tier
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # Python < 3.11: pytest brings tomli
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def quick_facts(tree: Path, out: Path, *, repo: Path, commit: str, is_checkout: bool,
                coverage_xml: Optional[Path]) -> dict[str, Any]:
    work = out / "quick"
    work.mkdir(parents=True, exist_ok=True)
    pythonpath = [tree / "src", SCRIPTS_DIR]
    run_python([str(SCORER_DIR / "_probe.py"), str(work / "probe.json")], cwd=tree,
               pythonpath=[tree / "src"], log=work / "probe.log")
    probe = _read_json(work / "probe.json")
    collected_path = work / "collected.json"
    run_python(
        ["-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-p", "release_review._collect_plugin"],
        cwd=tree, pythonpath=pythonpath, log=work / "collect.log",
        extra_env={"RELEASE_REVIEW_COLLECTED": str(collected_path)},
    )
    collected = _read_json(collected_path)
    ruff = run_python(["-m", "ruff", "check", "."], cwd=tree, pythonpath=[], log=work / "ruff.log")
    ruff_ran = "No module named ruff" not in (work / "ruff.log").read_text(encoding="utf-8", errors="replace")
    version = (probe or {}).get("version")
    commits_since_tag = None
    if version:
        try:
            commits_since_tag = int(git("rev-list", "--count", f"v{version}..{commit}", cwd=repo))
        except (ScoringError, ValueError):
            commits_since_tag = None
    root_files = tracked_root_files(tree, is_checkout)
    changelog = next((tree / name for name in root_files if name.upper().startswith("CHANGELOG")), None)
    sources = {
        path.relative_to(tree / "src").as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((tree / "src" / "haemolynx").rglob("*.py"))
    }
    return {
        "workflows": {path.name: path.read_text(encoding="utf-8") for path in sorted((tree / ".github" / "workflows").glob("*.y*ml"))},
        "root_files": root_files,
        "sources": sources,
        "pyproject": _load_toml(tree / "pyproject.toml"),
        "probe": probe,
        "collected": None if collected is None else collected["items"],
        "collection_errors": [] if collected is None else collected["errors"],
        "ruff_clean": (ruff == 0) if ruff_ran else None,
        "coverage_xml": coverage_xml.read_text(encoding="utf-8") if coverage_xml else None,
        "has_tutorial_test": (tree / "tests" / "integration" / "test_pipeline_tutorial.py").is_file(),
        "changelog_text": changelog.read_text(encoding="utf-8") if changelog else None,
        "commits_since_tag": commits_since_tag,
    }


# ---------------------------------------------------------------------------
# The fixture tier
# ---------------------------------------------------------------------------


def run_dataset(tree: Path, folder: Path, name: str, settings: Mapping[str, Any], *,
                config: Optional[Path] = None, single_process: bool = False,
                stop_after: Optional[str] = None, save_graph: bool = False, lumen_report: bool = False,
                timeout: Optional[float] = None) -> Optional[dict[str, Any]]:
    """One run in its own process, against *tree*'s ``src``; its ``result.json``."""
    folder.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": name, "output_dir": str(folder), "settings": dict(settings),
        "config": None if config is None else str(config), "single_process": single_process,
        "stop_after": stop_after, "save_graph": save_graph, "lumen_report": lumen_report,
    }
    spec_path = folder / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    try:
        run_python([str(SCORER_DIR / "_run_dataset.py"), str(spec_path)], cwd=tree,
                   pythonpath=[tree / "src"], log=folder / "run.log", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "failed", "error": {"type": "Timeout", "raised_in": "scorer",
                                                             "message": f"no result within {timeout} s"}}
    return _read_json(folder / "result.json")


def read_back(tree: Path, folder: Path) -> list[dict[str, Any]]:
    run_python([str(SCORER_DIR / "_read_back.py"), str(folder)], cwd=tree,
               pythonpath=[tree / "src", SCRIPTS_DIR], log=folder.parent / f"{folder.name}_readback.log")
    return _read_json(folder / "readback.json") or []


def _move(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    source.rename(destination)
    return destination


def _manifest(folder: Path) -> Optional[dict[str, Any]]:
    found = manifests.find(folder)
    return _read_json(found[0]) if found else None


def fixture_tier(tree: Path, out: Path, *, commit: str, log=print) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every fixture-tier result, and the details the report shows."""
    work = out / "fixtures"
    results, readbacks, runs = {}, {}, {}
    for name, settings in fixtures.fixture_datasets().items():
        log(f"fixture {name}")
        settings = {**settings, "input_path": str(tree / settings["input_path"])}
        results[name] = run_dataset(tree, work / name, name, settings)
        readbacks[name] = read_back(tree, work / name)
        manifest_path = manifests.find(work / name)
        runs[name] = manifests.read(manifest_path[0]) if manifest_path else None

    log("identity: the Nerve crop three times")
    same = work / "identity" / "run"
    identity = {**fixtures.identity_settings(), "input_path": str(tree / fixtures.NERVE_CROP)}
    first_result = run_dataset(tree, same, "identity_a", identity)
    first = _move(same, work / "identity" / "a")
    second_result = run_dataset(tree, same, "identity_b", identity)
    second = _move(same, work / "identity" / "b")
    alone_result = run_dataset(tree, same, "identity_one_process", identity, single_process=True)
    alone = _move(same, work / "identity" / "one_process")
    twice = [asdict(each) for each in exports.compare_folders(first, second)]
    workers = [asdict(each) for each in exports.compare_folders(first, alone)]

    cases = {}
    for case in edge_cases.CASES:
        log(f"edge case {case.name}")
        folder = work / "edge" / case.name
        folder.mkdir(parents=True, exist_ok=True)
        image = folder / "input.tif"
        import tifffile

        tifffile.imwrite(str(image), case.make())
        cases[case.name] = run_dataset(tree, folder / "run", case.name, {
            "input_path": str(image), "show_plots_in_ide": False, "interactive_plots": False, **case.settings,
        }, timeout=900)

    scores = {}
    for name, voxels, quality in phantoms.variants():
        variant = f"{name}_{voxels}_{quality}"
        log(f"phantom {variant}")
        folder = work / "phantoms" / variant
        folder.mkdir(parents=True, exist_ok=True)
        phantom = phantoms.PHANTOMS[name]()
        voxel_size = phantoms.VOXEL_SIZES[voxels]
        mask, raw = phantoms.render(phantom, voxel_size, quality, seed=0)
        import tifffile

        tifffile.imwrite(str(folder / "mask.tif"), mask)
        tifffile.imwrite(str(folder / "raw.tif"), raw)
        truth = phantoms.truth(phantom)
        (folder / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        built = run_dataset(tree, folder / "run", variant, {
            "input_path": str(folder / "mask.tif"), "show_plots_in_ide": False, "interactive_plots": False,
            "voxel_size_override_xyz": [voxel_size[2], voxel_size[1], voxel_size[0]], "voxel_size_policy": "override",
        }, stop_after="build_network", save_graph=True, timeout=900)
        graph_file = folder / "run" / "graph_returned.pkl"
        if built and built.get("status") == "finished" and graph_file.is_file():
            scores[variant] = graph_metrics.score_graph(exports.load_graph(graph_file), truth, voxel_size)
        else:
            scores[variant] = {"error": (built or {}).get("error")}

    seven_files = [path.name for path in (work / "seven_vessel").glob("*") if path.is_file()]
    collected = [
        fixtures.input_2a(results),
        fixtures.graph_2a(results.get("nerve_crop")),
        fixtures.diameter_2a(results.get("nerve_crop"), {"the Nerve crop at the skeleton defaults": first_result}),
        fixtures.flow_2a(results.get("nerve_crop")),
        fixtures.flow_6b(results.get("nerve_crop")),
        fixtures.stats_2a(results.get("seven_vessel"), seven_files),
        fixtures.export_2a(results.get("seven_vessel"), seven_files),
        fixtures.repro_2a(results),
        fixtures.repro_6a({name: _manifest(work / name) for name in results}, commit),
        fixtures.repro_6c(twice, second_result),
        fixtures.repro_8b(workers, first_result, alone_result),
        fixtures.robust_6a(cases),
        fixtures.export_8b(readbacks),
        fixtures.graph_8a(scores),
    ]
    details = {
        "results": results, "runs": runs, "readbacks": readbacks, "identity_twice": twice,
        "identity_workers": workers, "identity_results": [first_result, second_result, alone_result],
        "edge_cases": cases, "phantoms": scores,
        "undocumented_outputs": fixtures.undocumented_outputs(readbacks),
    }
    return {result.id: result for result in collected if result is not None}, details


# ---------------------------------------------------------------------------
# The full tier
# ---------------------------------------------------------------------------


def full_tier(tree: Path, out: Path, *, commit: str, date: str, datasets_file: Path,
              benchmarks_csv: Path, fixture_runs: Mapping[str, Any],
              fixture_results: Optional[Mapping[str, Any]] = None, only: Optional[Sequence[str]] = None,
              log=print) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = full.load_datasets(datasets_file)
    runs: dict[str, Any] = dict(fixture_runs)
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name, dataset in datasets.items():
        if only and name not in only:
            skipped[name] = "not chosen for this scoring (--datasets)"
            continue
        missing = full.missing_inputs(dataset)
        if missing:
            skipped[name] = f"input not on this machine: {', '.join(missing)}"
            continue
        needed = dataset.get("needs_gib")
        free = free_ram_bytes()
        if needed and free is not None and free < needed * 2**30:
            skipped[name] = f"needs {needed} GiB free, {free / 2**30:.1f} GiB free"
            continue
        log(f"full {name}")
        folder = out / name
        settings = full.dataset_settings(dataset)
        started = time.time()
        results[name] = run_dataset(tree, folder, name, settings, config=full.config_path(dataset),
                                    lumen_report=name in full.LUMEN_REPORT_DATASETS)
        log(f"full {name}: {(results[name] or {}).get('status')} in {time.time() - started:.0f} s")
        found = manifests.find(folder)
        runs[name] = manifests.read(found[0]) if found else None
    rows = []
    for name, run in runs.items():
        if run is not None:
            rows += manifests.benchmark_rows(run, dataset=name, commit=commit, date=date,
                                             voxels=(datasets.get(name) or {}).get("voxels"))
    earlier = trends.read_rows(benchmarks_csv)
    trends.append_rows(benchmarks_csv, rows)
    flagged = trends.slower_stages(earlier + rows, commit)
    compared_with = full.compared_commit(earlier, commit)
    collected = full.collect(runs, results, datasets, skipped, flagged=flagged, compared_with=compared_with,
                             rows=earlier + rows, fixture_results=fixture_results, commit=commit)
    return collected, {"runs": runs, "results": results, "skipped": skipped, "slower": flagged,
                       "compared_with": compared_with}


# ---------------------------------------------------------------------------
# A scoring
# ---------------------------------------------------------------------------


def score(
    *, repo: Path, commit: str, tiers: Sequence[str], out_root: Optional[Path] = None,
    tree: Optional[Path] = None, coverage_xml: Optional[Path] = None,
    datasets_file: Optional[Path] = None, only_datasets: Optional[Sequence[str]] = None,
    check_outputs: Optional[bool] = None, log=print,
) -> tuple[Path, scorecard.Scorecard]:
    repo = Path(repo).resolve()
    sha = git("rev-parse", "--short=8", commit, cwd=repo)
    date = time.strftime("%Y-%m-%d")
    main = main_checkout(repo)
    # Absolute: every run's subprocess works in the scored tree.
    out_root = (Path(out_root) if out_root else main / "outputs" / "release_review").resolve()
    out = out_root / f"{date}_{sha}"
    out.mkdir(parents=True, exist_ok=True)
    is_checkout = tree is not None
    if tree is None:
        log(f"archiving {sha} into {out / 'code'}")
        tree = archive(repo, sha, out / "code")
    tree = Path(tree).resolve()
    document = review_doc.read(tree / "RELEASE_REVIEW.md")
    # The marks as the scored commit has them; this scorer's own file only
    # for a commit from before the file existed.
    recorded = tree / "scripts" / "release_review" / "human_checks.yaml"
    marks = human.load(recorded if recorded.is_file() else HUMAN_CHECKS)

    measured: dict[str, Any] = {}
    sections: list[tuple[str, str]] = []
    extra: dict[str, Any] = {}
    if QUICK in tiers:
        log("quick tier")
        facts = quick_facts(tree, out, repo=repo, commit=sha, is_checkout=is_checkout, coverage_xml=coverage_xml)
        measured.update(quick.collect(facts))
        extra["probe"] = facts.get("probe")
    fixture_runs: dict[str, Any] = {}
    fixture_results: dict[str, Any] = {}
    if FIXTURES in tiers:
        log("fixture tier")
        found, details = fixture_tier(tree, out, commit=sha, log=log)
        measured.update(found)
        fixture_runs = {name: run for name, run in details["runs"].items() if run is not None}
        fixture_results = details["results"]
        sections.append(("Fixture tier", full.fixture_section(details)))
        extra["fixtures"] = {key: value for key, value in details.items() if key != "runs"}
    if FULL in tiers:
        log("full tier")
        datasets_file = Path(datasets_file) if datasets_file else main / full.DATASETS_FILE
        found, details = full_tier(tree, out, commit=sha, date=date, datasets_file=datasets_file,
                                   benchmarks_csv=out_root / "benchmarks.csv", fixture_runs=fixture_runs,
                                   fixture_results=fixture_results, only=only_datasets, log=log)
        measured.update(found)
        sections.append(("Full tier", full.full_section(details)))
        extra["full"] = {key: value for key, value in details.items() if key != "runs"}

    if check_outputs is None:
        check_outputs = not os.environ.get("CI")
    roots = [main, *worktrees(repo)] if check_outputs else []
    findings = evidence.audit(
        ((row.id, row.evidence) for row in document.rows.values()),
        evidence.Tree(tree, roots), check_outputs=check_outputs,
    )
    card = scorecard.build(
        document, measured, marks, commit=sha, date=date, tiers=tiers,
        missing_citations=evidence.missing_by_check(findings),
    )
    sections.append(("Evidence audit", full.evidence_section(findings, check_outputs)))
    previous = report.latest_previous(out_root, out)
    moved = report.changes(previous, card)
    (out / "scorecard.md").write_text(report.markdown(card, moved=moved, sections=sections), encoding="utf-8")
    report.write_json(out / "scorecard.json", card, {"moved": moved, "evidence": [asdict(each) for each in findings], **extra})
    return out, card
