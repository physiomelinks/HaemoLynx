"""The full tier: the reference datasets, and the report's longer sections.

The datasets are local and never committed: their paths live in the main
checkout's ``release_review_datasets.yaml`` (git-ignored, as every root
``*.yaml`` is). One entry per dataset::

    e14p5:
      config: config.yaml          # relative to this file's folder
      settings: {statistics: true} # beside the config's
      needs_gib: 8                 # free memory to start it with
      shape_zyx: [287, 512, 512]
    e14p5_eighth:                  # the same, cut to an eighth (speed.10a)
      of: e14p5
      crop_fraction: 0.125

A ``*_path`` setting given relative is read from that folder too. A dataset
whose input is missing is reported as not on this machine, never run on a
stand-in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import manifests, trends
from .evidence import Finding
from .ladder import CheckResult, measured
from .rubric import target

DATASETS_FILE = "release_review_datasets.yaml"

#: The reference datasets (``RELEASE_REVIEW.md`` § Reference datasets): the
#: three fixtures, from the fixture tier, and the four local stacks.
FIXTURE_RUNS = ("seven_vessel", "bundled_h5", "nerve_crop")
REFERENCE = FIXTURE_RUNS + ("e14p5", "nerve", "mason_sm", "brain")
TITLES = {
    "seven_vessel": "seven-vessel fixture", "bundled_h5": "bundled h5 fixture", "nerve_crop": "Nerve crop fixture",
    "e14p5": "E14.5", "nerve": "Nerve", "mason_sm": "Mason SM", "brain": "Brain",
}


def load_datasets(path: Path) -> dict[str, dict[str, Any]]:
    """The datasets file, each entry's paths made absolute and each crop
    filled in from the dataset it is cut from."""
    import yaml

    path = Path(path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    datasets = {}
    for name, entry in raw.items():
        entry = dict(entry or {})
        parent = raw.get(entry.get("of")) if entry.get("of") else None
        if parent is not None:
            fraction = float(entry["crop_fraction"])
            shape = parent["shape_zyx"]
            entry = {
                **{key: value for key, value in parent.items() if key not in ("shape_zyx", "voxels")},
                **entry,
                "settings": {**(parent.get("settings") or {}), **crop_settings(shape, fraction), **(entry.get("settings") or {})},
                "voxels": cropped_voxels(shape, fraction),
            }
        elif entry.get("shape_zyx") and not entry.get("voxels"):
            shape = entry["shape_zyx"]
            entry["voxels"] = int(shape[0]) * int(shape[1]) * int(shape[2])
        if entry.get("config"):
            entry["config"] = str((base / entry["config"]).resolve())
        entry["settings"] = {
            key: (str((base / value).resolve()) if key.endswith("_path") and isinstance(value, str) and not Path(value).is_absolute() else value)
            for key, value in (entry.get("settings") or {}).items()
        }
        datasets[name] = entry
    return datasets


def crop_size_xyz(shape_zyx: Sequence[int], fraction: float) -> list[int]:
    """A centred box holding *fraction* of the volume, each axis cut by the
    same factor (x, y, z, as ``crop_size_xyz`` takes it)."""
    factor = fraction ** (1.0 / 3.0)
    z, y, x = (max(1, int(round(side * factor))) for side in shape_zyx)
    return [x, y, z]


def cropped_voxels(shape_zyx: Sequence[int], fraction: float) -> int:
    x, y, z = crop_size_xyz(shape_zyx, fraction)
    return x * y * z


def crop_settings(shape_zyx: Sequence[int], fraction: float) -> dict[str, Any]:
    return {
        "crop_on_import": True, "crop_method": "size", "crop_size_xyz": crop_size_xyz(shape_zyx, fraction),
        "crop_location": "centre", "crop_z_location": "centre",
    }


def config_path(dataset: Mapping[str, Any]) -> Optional[Path]:
    return Path(dataset["config"]) if dataset.get("config") else None


def dataset_settings(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {"show_plots_in_ide": False, "interactive_plots": False, **(dataset.get("settings") or {})}


def missing_inputs(dataset: Mapping[str, Any]) -> list[str]:
    """The dataset's config, and its input image when named here or in its
    config, that do not exist on this machine."""
    missing = []
    config = config_path(dataset)
    if config is not None and not config.is_file():
        return [str(config)]
    image = (dataset.get("settings") or {}).get("input_path")
    if image is None and config is not None:
        import yaml

        def find(node: Any) -> Optional[str]:
            if isinstance(node, Mapping):
                if isinstance(node.get("input_path"), str):
                    return node["input_path"]
                for value in node.values():
                    found = find(value)
                    if found:
                        return found
            return None

        image = find(yaml.safe_load(config.read_text(encoding="utf-8")))
        if image and not Path(image).is_absolute():
            image = str(config.parent / image)
    if image and not Path(image).exists():
        missing.append(image)
    return missing


def compared_commit(earlier_rows: Sequence[Mapping[str, Any]], commit: str) -> Optional[str]:
    """The latest earlier commit with benchmark rows: what a slowdown is
    measured against."""
    commits = [str(row.get("commit")) for row in earlier_rows if str(row.get("commit")) != commit]
    return commits[-1] if commits else None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def _gib(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 2**30:.1f} GiB"


def _not_run(name: str, skipped: Mapping[str, str]) -> str:
    return f"{TITLES.get(name, name)}: {skipped.get(name, 'not in ' + DATASETS_FILE)}"


def speed_2a(run: Optional[manifests.RunRecord]) -> Optional[CheckResult]:
    """Decided by an E14.5 run of this scoring; with none (not run, or no
    room to run it) the check's row decides, so no room is never a fail."""
    if run is None:
        return None
    stages = ", ".join(f"{stage.stage} {stage.wall_seconds:,.0f} s" for stage in run.stages if stage.wall_seconds)
    return measured("speed.2a", run.finished, run.wall_seconds,
                    f"E14.5 {run.status} in {run.wall_seconds:,.0f} s ({stages})")


def memory_2a(run: Optional[manifests.RunRecord]) -> Optional[CheckResult]:
    if run is None:
        return None
    return measured("memory.2a", run.finished, run.peak_rss_bytes,
                    f"E14.5 {run.status}, peak {_gib(run.peak_rss_bytes)} on a {_gib(run.machine_ram_bytes)} {run.platform} machine")


#: The real stacks whose saved graph's lumen-artefact report counts for
#: ``graph.6a``.
LUMEN_REPORT_DATASETS = ("e14p5", "nerve", "mason_sm", "brain")


def graph_6a(results: Mapping[str, Optional[Mapping[str, Any]]]) -> Optional[CheckResult]:
    """The lumen-artefact report on a reference dataset's saved graph,
    recorded at this commit; None when no such dataset ran."""
    reports = {
        name: ((results.get(name) or {}).get("lumen_artefacts") or {})
        for name in LUMEN_REPORT_DATASETS if results.get(name)
    }
    if not reports:
        return None
    done = {name: report for name, report in reports.items() if report.get("text")}
    evidence = "; ".join(f"{TITLES[name]}: {report['text']}" for name, report in done.items())
    missing = [TITLES[name] for name in reports if name not in done]
    if missing:
        evidence += ("; " if evidence else "") + f"no report for {', '.join(missing)}"
    return measured("graph.6a", bool(done), {name: report.get("counts") for name, report in done.items()}, evidence)


def measured_before(rows: Sequence[Mapping[str, Any]], dataset: str) -> Optional[str]:
    """The latest commit whose benchmark rows give *dataset* a finished run
    with every stage's time and peak memory."""
    by_commit: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("dataset") == dataset:
            by_commit.setdefault(str(row.get("commit")), []).append(row)
    for commit in reversed(list(by_commit)):
        stages = by_commit[commit]
        total = [row for row in stages if row.get("stage") == "total"]
        complete = all(
            trends._float(row.get("wall_seconds")) is not None and trends._float(row.get("peak_rss_mib")) is not None
            for row in stages if row.get("stage") != "total"
        )
        if total and total[-1].get("status") == "finished" and complete and len(stages) > 1:
            return commit
    return None


def per_stage_check(
    check_id: str, runs: Mapping[str, Optional[manifests.RunRecord]], skipped: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]] = (),
) -> CheckResult:
    """A per-stage table for each reference dataset at a recorded commit:
    this scoring's run, or failing that an earlier one in ``benchmarks.csv``."""
    done, missing = manifests.per_dataset(runs, REFERENCE)
    earlier = {name: measured_before(rows, name) for name in missing}
    done += [f"{name} (at {commit})" for name, commit in earlier.items() if commit]
    missing = [name for name, commit in earlier.items() if not commit]
    what = "time" if check_id.startswith("speed") else "peak memory"
    shown = [TITLES.get(name.split(" ")[0], name) + name[len(name.split(" ")[0]):] for name in done]
    evidence = f"each stage's {what} recorded for {', '.join(shown) or 'no dataset'}"
    if missing:
        evidence += "; not for " + "; ".join(_not_run(name, skipped) for name in missing)
    return measured(check_id, not missing, {"measured": done, "missing": missing}, evidence)


def flow_6c(results: Mapping[str, Optional[Mapping[str, Any]]], skipped: Mapping[str, str]) -> CheckResult:
    """Each reference dataset's solved share, and its unsolved centreline
    explained by a ``flow_state`` (behind a dead end, a closed end, no
    boundary) rather than a solve that never ran."""
    parts, missing = [], []
    for name in REFERENCE:
        graph = (results.get(name) or {}).get("graph") or {}
        fraction = graph.get("solved_fraction")
        states = graph.get("length_um_by_flow_state") or {}
        if fraction is None or "not solved" in states:
            missing.append(name)
            reason = "no solve" if results.get(name) else skipped.get(name, "not run")
            parts.append(f"{TITLES[name]}: {reason}")
            continue
        unsolved = ", ".join(f"{state} {length:,.0f} um" for state, length in states.items() if state != "solved")
        parts.append(f"{TITLES[name]} {100 * fraction:.1f}% of {graph.get('length_um', 0):,.0f} um solved ({unsolved or 'all'})")
    return measured("flow.6c", not missing, {"missing": missing}, "; ".join(parts))


def speed_8b(flagged: Sequence[trends.Slowdown], compared_with: Optional[str]) -> Optional[CheckResult]:
    if compared_with is None:
        return None
    slower = ", ".join(f"{each.dataset} {each.stage} {each.before_seconds:.0f} -> {each.seconds:.0f} s" for each in flagged)
    return measured(
        "speed.8b", True, [each.__dict__ for each in flagged],
        f"each stage compared with {compared_with} in benchmarks.csv; more than "
        f"{100 * target('speed.8b'):.0f}% slower: {slower or 'none'}",
        target=target("speed.8b"),
    )


def scaling_check(
    check_id: str, rows: Sequence[Mapping[str, Any]], datasets: Mapping[str, Any], commit: Optional[str] = None,
) -> Optional[CheckResult]:
    """Every stage's time (or peak memory) against volume, on E14.5 and its
    crops at one commit: each stage needs the target's number of sizes, and
    none may grow faster than its volume."""
    if commit is not None:
        rows = [row for row in rows if str(row.get("commit")) == commit]
    family = ["e14p5"] + [name for name, entry in datasets.items() if entry.get("of") == "e14p5"]
    field = "wall_seconds" if check_id.startswith("speed") else "peak_rss_mib"
    found = trends.scaling(rows, family, field)
    if found["sizes"] == 0:
        return None
    needed = int(target(check_id, 0))
    exponents = ", ".join(f"{stage} {value:.2f}" for stage, value in sorted(found["exponents"].items()) if value is not None)
    short = sorted(
        stage for stage, sizes in found["stage_sizes"].items() if stage != "total" and sizes < needed
    )
    evidence = f"E14.5 at {found['sizes']} sizes; growth with volume: {exponents or 'none'}"
    if short:
        evidence += f"; measured at fewer than {needed} sizes: {', '.join(short)}"
    passed = found["sizes"] >= needed and not short and not found["faster_than_volume"]
    if check_id == "memory.10a":
        whole = [
            row for row in rows
            if row.get("dataset") == "e14p5" and row.get("stage") == "total" and row.get("status") == "finished"
        ]
        peak = trends._float(whole[-1].get("peak_rss_mib")) if whole else None
        limit = target("memory.10a", 1) / 2**20
        shown = "-" if peak is None else f"{peak / 1024:.1f} GiB"
        evidence += f"; E14.5 peak {shown} against {limit / 1024:.0f} GiB"
        passed = passed and peak is not None and peak <= limit
    if found["faster_than_volume"]:
        evidence += f"; faster than volume, with no stated reason: {', '.join(found['faster_than_volume'])}"
    return measured(check_id, passed, found, evidence)


def collect(
    runs: Mapping[str, Optional[manifests.RunRecord]],
    results: Mapping[str, Optional[Mapping[str, Any]]],
    datasets: Mapping[str, Any],
    skipped: Mapping[str, str],
    *,
    flagged: Sequence[trends.Slowdown],
    compared_with: Optional[str],
    rows: Sequence[Mapping[str, Any]],
    fixture_results: Optional[Mapping[str, Any]] = None,
    commit: Optional[str] = None,
) -> dict[str, CheckResult]:
    every_result = {**(fixture_results or {}), **results}
    found = [
        graph_6a(results),
        speed_2a(runs.get("e14p5")),
        memory_2a(runs.get("e14p5")),
        per_stage_check("speed.6a", runs, skipped, rows),
        per_stage_check("memory.6a", runs, skipped, rows),
        flow_6c(every_result, skipped),
        speed_8b(flagged, compared_with),
        scaling_check("speed.10a", rows, datasets, commit),
        scaling_check("memory.10a", rows, datasets, commit),
    ]
    return {result.id: result for result in found if result is not None}


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _run_line(name: str, result: Optional[Mapping[str, Any]]) -> str:
    if not result:
        return f"- **{name}**: no result"
    graph = result.get("graph") or {}
    line = f"- **{name}**: {result.get('status')} in {result.get('seconds')} s"
    if graph.get("edges") is not None:
        line += f"; {graph['edges']} vessels, {graph.get('length_um', 0):,} um"
    if graph.get("flow_residual"):
        line += f"; flow balance {graph['flow_residual']['relative']:.1e} of the largest flow"
    if (result.get("lumen_artefacts") or {}).get("text"):
        line += f"; {result['lumen_artefacts']['text']}"
    if result.get("error"):
        error = result["error"]
        line += f"; {error.get('type')} in {error.get('raised_in')}: {str(error.get('message'))[:200]}"
    return line


def fixture_section(details: Mapping[str, Any]) -> str:
    lines = ["### Runs", ""]
    for name, result in details["results"].items():
        lines.append(_run_line(name, result))
    for name, run in details["runs"].items():
        if run is not None:
            lines += ["", f"#### {name}", "", manifests.stage_table(run)]
    lines += ["", "### Identity (the Nerve crop)", ""]
    for label, key in (("Run twice", "identity_twice"), ("With workers and in one process", "identity_workers")):
        differences = details[key]
        lines.append(f"- {label}: " + ("identical" if not differences else
                                       "; ".join(f"{each['file']}: {each['detail']}" for each in differences[:8])))
    for result in details["identity_results"]:
        if result:
            lines.append(f"  - {result.get('name')}: {result.get('status')}, {result.get('worker_pools_started')} worker pools")
    lines += ["", "### Edge cases", ""]
    for name, result in details["edge_cases"].items():
        lines.append(_run_line(name, result))
    lines += ["", "### Exports read back", ""]
    for name, files in details["readbacks"].items():
        failed = [each for each in files if not each["opened"]]
        lines.append(f"- {name}: {len(files) - len(failed)} of {len(files)} opened"
                     + (": " + ", ".join(f"{each['file']} ({each['error']})" for each in failed) if failed else ""))
    undocumented = details.get("undocumented_outputs") or []
    lines.append(f"- Not in the data dictionary: {', '.join(undocumented) or 'nothing'}")
    lines += ["", "### Phantoms (graph.8a)", "",
              "| Variant | Junctions true / built / matched | Precision | Recall | Length (um) built / true | Error | Loops built / true |",
              "|---|---|---|---|---|---|---|"]
    for name, score in details["phantoms"].items():
        if not score or "precision" not in score:
            lines.append(f"| {name} | not built: {(score or {}).get('error')} | | | | | |")
            continue
        error = score["length_error"]
        lines.append(
            f"| {name} | {score['junctions_true']} / {score['junctions_built']} / {score['junctions_matched']} | "
            f"{_number(score['precision'])} | {_number(score['recall'])} | {score['length_um']:,} / {score['true_length_um']:,} | "
            f"{'-' if error is None else format(100 * error, '+.1f') + '%'} | {score['loops']} / {score['true_loops']} |"
        )
    return "\n".join(lines)


def _number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def full_section(details: Mapping[str, Any]) -> str:
    lines = []
    for name, reason in details["skipped"].items():
        lines.append(f"- **{name}** not run: {reason}")
    for name, result in details["results"].items():
        lines.append(_run_line(name, result))
    for name, run in details["runs"].items():
        if run is not None and name not in FIXTURE_RUNS:
            lines += ["", f"### {name}", "", manifests.stage_table(run)]
    lines += ["", "### Slower than the last scored commit", ""]
    if details.get("compared_with") is None:
        lines.append("No earlier benchmark rows to compare with.")
    else:
        flagged = details.get("slower") or []
        lines.append(f"Compared with {details['compared_with']}: " + (
            "; ".join(f"{each.dataset} {each.stage} {each.before_seconds:.1f} -> {each.seconds:.1f} s" for each in flagged)
            or "no stage more than 20% slower"))
    return "\n".join(lines)


def evidence_section(findings: Sequence[Finding], checked_outputs: bool) -> str:
    lines = [
        "Backticked paths in every row's evidence, looked for in the scored tree"
        + (" and, for `outputs/`, the main checkout and its worktrees." if checked_outputs else "; `outputs/` paths not checked here."),
        "",
    ]
    if not findings:
        return "\n".join(lines + ["Every cited path exists."])
    for finding in findings:
        lines.append(f"- `{finding.check_id}`: `{finding.citation.text}` {finding.problem}"
                     + (f" ({finding.detail})" if finding.detail else ""))
    return "\n".join(lines)
