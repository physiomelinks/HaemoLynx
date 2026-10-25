"""The fixture tier: what the committed fixtures, the edge cases and the
phantoms decide.

The runner runs each dataset in its own process (``_run_dataset.py``) and
hands the collectors here what came back: each run's ``result.json``, its
manifest, its exports read back, the differences between repeated runs and
the phantoms' scores. The collectors are pure.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .edge_cases import classify_failure
from .ladder import CheckResult, measured
from .rubric import target

#: The three committed fixtures (``RELEASE_REVIEW.md`` § Reference datasets),
#: each with the settings it runs with beside the schema's defaults. The
#: Nerve crop takes ``tests/integration/test_nerve_pipeline.py``'s boundary
#: boxes: on the defaults its solve refuses to run (``flow.6c``).
SEVEN_VESSEL = "tests/data/seven_vessel_noisy_3d.tif"
BUNDLED_H5 = "tests/data/bundled_vessels_8_to_2.h5"
NERVE_CROP = "tests/data/Nerve_capillaries_cropped.tif"

_QUIET = {"show_plots_in_ide": False, "interactive_plots": False}

#: The Nerve crop's shape, (z, y, x) voxels; its boxes split y at 40%.
NERVE_CROP_SHAPE = (63, 696, 941)


def nerve_crop_boundaries(shape: Sequence[int] = NERVE_CROP_SHAPE) -> dict[str, Any]:
    """``test_nerve_pipeline.py``'s settings: inlets in the first 40% of y,
    outlets in the rest, by volume, with its skeleton clean-up."""
    z, y, x = shape
    split = max(1, int(0.4 * y))
    return {
        "skeleton_closing_radius": 1,
        "skeleton_bridge_gap_size": 1,
        "skeleton_min_branch_length": 3,
        "skeleton_max_bridge_distance": 2,
        "skeleton_component_connectivity": 3,
        "skeleton_min_component_percent": 30.0,
        "inlet_node_selection_method": "volume",
        "outlet_node_selection_method": "volume",
        "inlet_node_volumes": [[[0.0, 0.0, 0.0], [float(z - 1), float(split - 1), float(x - 1)]]],
        "outlet_node_volumes": [[[0.0, float(split), 0.0], [float(z - 1), float(y - 1), float(x - 1)]]],
        "inlet_nodes": [],
        "outlet_nodes": [],
        "inlet_p_bc": 1000.0,
        "outlet_p_bc": 500.0,
        "min_stub_length": 3.0,
    }


def fixture_datasets() -> dict[str, dict[str, Any]]:
    """Each fixture run: its name, input and settings."""
    return {
        "seven_vessel": {"input_path": SEVEN_VESSEL, "statistics": True, **_QUIET},
        "bundled_h5": {"input_path": BUNDLED_H5, **_QUIET},
        "nerve_crop": {"input_path": NERVE_CROP, **_QUIET, **nerve_crop_boundaries()},
    }


#: The identity runs (``repro.6c``, ``repro.8b``): the Nerve crop with its
#: boundary boxes but the schema's skeleton defaults, which leave it 129
#: vessels, more than ``haemodynamics.sections.MIN_EDGES_FOR_WORKERS``; with
#: the mask's widths measured for every vessel the diameters stage then
#: starts worker processes, and with statistics on, so do its measures. Off
#: (the defaults read diameters from the table), no worker starts and the
#: worker count is never tested.
def identity_settings() -> dict[str, Any]:
    boxes = nerve_crop_boundaries()
    keep = ("inlet_node_selection_method", "outlet_node_selection_method", "inlet_node_volumes",
            "outlet_node_volumes", "inlet_nodes", "outlet_nodes", "inlet_p_bc", "outlet_p_bc")
    return {
        "input_path": NERVE_CROP, **_QUIET, **{name: boxes[name] for name in keep},
        "use_edt_diameter_crosscheck": True, "statistics": True,
    }


def _finished(result: Optional[Mapping[str, Any]]) -> bool:
    return bool(result) and result.get("status") == "finished"


def _graph(result: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return (result or {}).get("graph") or {}


def _where(name: str, result: Optional[Mapping[str, Any]]) -> str:
    if not result:
        return f"{name}: not run"
    if result.get("status") != "finished":
        error = result.get("error") or {}
        return f"{name}: failed, {error.get('type')}: {str(error.get('message'))[:120]}"
    return f"{name}: finished in {result.get('seconds')} s"


def input_2a(results: Mapping[str, Mapping[str, Any]]) -> CheckResult:
    tif, h5 = results.get("seven_vessel"), results.get("bundled_h5")
    evidence = f"{_where('seven_vessel_noisy_3d.tif', tif)}; {_where('bundled_vessels_8_to_2.h5', h5)}"
    return measured("input.2a", _finished(tif) and _finished(h5), None, evidence)


def graph_2a(nerve: Optional[Mapping[str, Any]]) -> CheckResult:
    graph = _graph(nerve)
    evidence = f"{_where('Nerve crop', nerve)}: {graph.get('edges', 0)} vessels, {graph.get('length_um', 0):,} um"
    return measured("graph.2a", _finished(nerve) and graph.get("edges", 0) > 0, graph.get("edges"), evidence)


def _diameters(graph: Mapping[str, Any]) -> str:
    sources = graph.get("diameter_sources") or {}
    return (
        f"{graph.get('edges_with_diameter', 0)} of {graph.get('edges', 0)} vessels with a diameter and its source "
        f"({', '.join(f'{name} {count}' for name, count in sorted(sources.items()))})"
    )


def diameter_2a(nerve: Optional[Mapping[str, Any]], others: Optional[Mapping[str, Mapping[str, Any]]] = None) -> CheckResult:
    """Level 2 is "it works, at least once, on real data": every vessel of
    the Nerve crop's run has a diameter and its source. Other real runs that
    leave some without are reported beside it."""
    graph = _graph(nerve)
    edges, with_diameter = graph.get("edges", 0), graph.get("edges_with_diameter", 0)
    evidence = f"Nerve crop (test_nerve_pipeline.py's settings): {_diameters(graph)}"
    for name, result in (others or {}).items():
        other = _graph(result)
        if _finished(result) and other.get("edges_with_diameter", 0) < other.get("edges", 0):
            states = other.get("length_um_by_flow_state") or {}
            evidence += f"; {name}: {_diameters(other)}, flow states {states}"
    passed = _finished(nerve) and edges > 0 and with_diameter == edges
    return measured("diameter.2a", passed, graph.get("diameter_sources"), evidence)


def flow_2a(nerve: Optional[Mapping[str, Any]]) -> CheckResult:
    graph = _graph(nerve)
    fraction = graph.get("solved_fraction")
    evidence = (
        f"Nerve crop (a real network): flows on {graph.get('edges_with_flow', 0)} of {graph.get('edges', 0)} vessels, "
        f"pressures on {graph.get('nodes_with_pressure', 0)} nodes, "
        f"{graph.get('solved_um', 0):,} of {graph.get('length_um', 0):,} um solved"
    )
    passed = _finished(nerve) and bool(fraction) and graph.get("edges_with_flow", 0) > 0
    return measured("flow.2a", passed, fraction, evidence)


def stats_2a(seven: Optional[Mapping[str, Any]], files: Sequence[str]) -> CheckResult:
    written = [name for name in files if name.endswith("_statistics.csv") and "branch" not in name]
    evidence = f"seven-vessel fixture with statistics on: {', '.join(written) or 'no statistics CSV'}"
    return measured("stats.2a", _finished(seven) and bool(written), written, evidence)


def export_2a(seven: Optional[Mapping[str, Any]], files: Sequence[str]) -> CheckResult:
    vtk = [name for name in files if name.endswith((".vtp", ".vtu", ".vtk"))]
    csv = [name for name in files if name.endswith(".csv")]
    evidence = f"seven-vessel fixture: {len(vtk)} VTK files, {len(csv)} CSVs"
    return measured("export.2a", _finished(seven) and bool(vtk) and bool(csv), {"vtk": vtk, "csv": csv}, evidence)


def repro_2a(results: Mapping[str, Mapping[str, Any]]) -> CheckResult:
    reloaded = {name: result.get("settings_reload_equal") for name, result in results.items()}
    evidence = "each run's `{stem}_run_settings.yaml` loads back to its settings: " + ", ".join(
        f"{name} {'yes' if value else 'no' if value is False else 'not written'}" for name, value in reloaded.items()
    )
    return measured("repro.2a", bool(reloaded) and all(reloaded.values()), reloaded, evidence)


def repro_6a(manifests: Mapping[str, Optional[Mapping[str, Any]]], commit: str) -> CheckResult:
    """Each run's folder holds its settings, the version, the commit (the
    runner records the archive's commit beside a manifest whose ``git`` is
    null: an archive has no ``.git``) and every input's checksum."""
    problems = []
    for name, manifest in manifests.items():
        if not manifest:
            problems.append(f"{name}: no manifest")
            continue
        missing = [key for key in ("settings", "haemolynx_version") if not manifest.get(key)]
        inputs = manifest.get("inputs") or []
        if not inputs or not all(str(each.get("checksum", "")).startswith("sha256:") for each in inputs):
            missing.append("input checksums")
        if missing:
            problems.append(f"{name}: no {', '.join(missing)}")
    evidence = (
        f"{len(manifests)} fixture runs' manifests hold the settings, version and input sha256; "
        f"commit {commit} recorded beside them"
    )
    if problems:
        evidence = "; ".join(problems)
    return measured("repro.6a", not problems and bool(manifests), None, evidence)


def _difference_text(differences: Sequence[Mapping[str, Any]]) -> str:
    counted = [each for each in differences if each.get("counted", True)]
    if not counted:
        aside = len(differences)
        return "identical" + (f" ({aside} differ only in a plot's bytes or an export timestamp)" if aside else "")
    return f"{len(counted)} outputs differ: " + "; ".join(f"{each['file']} {each['detail']}" for each in counted[:4])


def repro_6c(differences: Optional[Sequence[Mapping[str, Any]]], result: Optional[Mapping[str, Any]]) -> Optional[CheckResult]:
    if differences is None:
        return None
    counted = [each for each in differences if each.get("counted", True)]
    evidence = f"the Nerve crop run twice into the same folder: {_difference_text(differences)}"
    return measured("repro.6c", _finished(result) and not counted, len(counted), evidence)


def repro_8b(
    differences: Optional[Sequence[Mapping[str, Any]]], with_workers: Optional[Mapping[str, Any]],
    alone: Optional[Mapping[str, Any]],
) -> Optional[CheckResult]:
    if differences is None:
        return None
    pools = (with_workers or {}).get("worker_pools_started", 0)
    counted = [each for each in differences if each.get("counted", True)]
    evidence = (
        f"the Nerve crop with workers ({pools} pools started) and held to one process: "
        f"{_difference_text(differences)}"
    )
    passed = _finished(with_workers) and _finished(alone) and pools > 0 and not counted
    note = "" if pools else "no worker pool started, so the run did not test the worker count"
    return measured("repro.8b", passed, len(counted), evidence, note=note)


def robust_6a(cases: Mapping[str, Optional[Mapping[str, Any]]]) -> CheckResult:
    outcomes = {}
    for name, result in cases.items():
        if not result:
            outcomes[name] = "not run"
            continue
        error = result.get("error") or {}
        outcomes[name] = classify_failure(error.get("type"), error.get("raised_in"))
    failing = {name: outcome for name, outcome in outcomes.items() if outcome not in ("finished", "named")}
    parts = []
    for name, outcome in outcomes.items():
        error = (cases.get(name) or {}).get("error") or {}
        detail = f" ({error.get('type')} in {error.get('raised_in')})" if error else ""
        parts.append(f"{name} {outcome}{detail}")
    return measured("robust.6a", bool(outcomes) and not failing, outcomes, "; ".join(parts))


def export_8b(readbacks: Mapping[str, Sequence[Mapping[str, Any]]]) -> CheckResult:
    total = sum(len(files) for files in readbacks.values())
    failed = [
        f"{name}/{each['file']} ({each['error']})" for name, files in readbacks.items() for each in files if not each["opened"]
    ]
    readers = sorted({each["reader"].split(" ")[0] for files in readbacks.values() for each in files if each["opened"]})
    evidence = f"{total - len(failed)} of {total} exports of {len(readbacks)} fixture runs opened ({', '.join(readers)})"
    if failed:
        evidence += "; failed: " + ", ".join(failed[:5])
    return measured("export.8b", total > 0 and not failed, {"opened": total - len(failed), "total": total}, evidence)


def undocumented_outputs(readbacks: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """Files no dictionary entry names, and columns their entry does not
    describe (``export.6a``'s reach), across the runs."""
    found = set()
    for files in readbacks.values():
        for each in files:
            if each["entry"] is None:
                found.add(f"{each['file'].split('/')[-1]} (no entry)")
            for column in each.get("undocumented") or ():
                found.add(f"{each['file'].split('/')[-1]}: {column}")
    return sorted(found)


def graph_8a(scores: Mapping[str, Optional[Mapping[str, Any]]]) -> Optional[CheckResult]:
    """Every phantom variant's junction precision and recall at or above the
    target, and its length within the target."""
    from .graph_metrics import meets_targets

    if not scores:
        return None
    junction_goal, length_goal = target("graph.8a", 0), target("graph.8a", 1)
    failing, parts = [], []
    for name, score in scores.items():
        if not score or "precision" not in score:
            failing.append(name)
            parts.append(f"{name} not built")
            continue
        if not meets_targets(score, junction_goal, length_goal):
            failing.append(name)
        precision = "-" if score["precision"] is None else f"{score['precision']:.2f}"
        recall = "-" if score["recall"] is None else f"{score['recall']:.2f}"
        error = "-" if score["length_error"] is None else f"{100 * score['length_error']:+.1f}%"
        parts.append(f"{name} P {precision} R {recall} length {error}")
    return measured(
        "graph.8a", not failing, {"failing": failing}, "; ".join(parts),
        target={"junctions": junction_goal, "length": length_goal},
    )
