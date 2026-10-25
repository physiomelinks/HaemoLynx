"""One pipeline run for the scorer, in its own process.

``python _run_dataset.py <spec.json>``, run by :mod:`.runner` with
``PYTHONPATH`` at the scored tree's ``src`` so the run is that tree's code
and not the editable install's. The spec names the input, the settings
beside the schema's defaults (or a config file's), the output folder and
whether to hold the run to one process. It writes ``result.json`` into the
output folder: how the run ended (and, if it raised, the error's type, the
module it was raised in and its message), which ``haemolynx`` ran, how many
worker pools started, and a summary of the graph it left.

Imports nothing from the scorer, so it runs against any tree.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _innermost_module(trace) -> str | None:
    frames = traceback.extract_tb(trace)
    if not frames:
        return None
    filename = Path(frames[-1].filename).as_posix()
    if "/haemolynx/" in filename:
        tail = filename.rsplit("/haemolynx/", 1)[1]
        return "haemolynx." + tail[:-3].replace("/", ".") if tail.endswith(".py") else "haemolynx"
    for marker in ("/site-packages/", "/lib/python", "/Lib/"):
        if marker in filename:
            return filename.rsplit(marker, 1)[1].split("/", 1)[0]
    return filename


def _graph_summary(graph) -> dict:
    if graph is None:
        return {}
    edges = [data for _u, _v, data in graph.edges(data=True)]
    lengths = [float(data.get("length") or 0.0) for data in edges]
    with_diameter = [
        data for data in edges if data.get("diameter_um") is not None and data.get("diameter_source") not in (None, "unset")
    ]
    sources = {}
    for data in edges:
        source = data.get("diameter_source")
        sources[str(source)] = sources.get(str(source), 0) + 1
    states = {}
    solved_um = 0.0
    for data, length in zip(edges, lengths):
        state = str(data.get("flow_state") or "not solved")
        states[state] = round(states.get(state, 0.0) + length, 1)
        if state == "solved":
            solved_um += length
    total = sum(lengths)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "length_um": round(total, 1),
        "edges_with_diameter": len(with_diameter),
        "diameter_sources": sources,
        "edges_with_flow": sum("flow_signed" in data for data in edges),
        "nodes_with_pressure": sum(data.get("pressure") is not None for _n, data in graph.nodes(data=True)),
        "solved_um": round(solved_um, 1),
        "solved_fraction": round(solved_um / total, 4) if total else None,
        "length_um_by_flow_state": states,
    }


def _settings_reload(output: Path, before: dict, schema) -> bool | None:
    """Whether the run's settings file loads back to the settings it ran
    with (``repro.2a``); None when it wrote none."""
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline.settings import fill_derived_settings

    written = sorted(output.glob("*_run_settings.yaml"))
    if not written:
        return None
    loaded = load_config(written[0], schema)
    fill_derived_settings(loaded)
    return all(loaded.get(name) == before.get(name) for name in schema.names)


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("single_process"):
        os.cpu_count = lambda: 1  # every worker count is read from it, at call time
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    import concurrent.futures

    import haemolynx
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline.schema import SCHEMA
    from haemolynx.pipeline.settings import resolve_settings
    from haemolynx.pipeline.stages import run_pipeline_stages

    pools = {"started": 0}
    original = concurrent.futures.ProcessPoolExecutor.__init__

    def counted(self, *args, **kwargs):
        pools["started"] += 1
        original(self, *args, **kwargs)

    concurrent.futures.ProcessPoolExecutor.__init__ = counted

    config = spec.get("config")
    if config:
        values = load_config(Path(config), SCHEMA)
    else:
        values = {setting.name: setting.default for setting in SCHEMA}
    values.update(spec.get("settings") or {})
    values["vtk_output_prefix"] = str(output / "run")
    values["plot_dir"] = str(output / "plots")
    settings = resolve_settings(values, schema=SCHEMA, config_path=Path(config) if config else None)

    result = {
        "name": spec.get("name"),
        "haemolynx_file": Path(haemolynx.__file__).as_posix(),
        "single_process": bool(spec.get("single_process")),
        "status": "finished",
        "error": None,
    }
    before = copy.deepcopy(settings)
    started = time.perf_counter()
    graph = None
    try:
        graph = run_pipeline_stages(settings, SCHEMA, stop_after=spec.get("stop_after"))
    except BaseException as error:  # noqa: BLE001 - recorded, not hidden
        result["status"] = "failed"
        result["error"] = {
            "type": type(error).__name__,
            "type_module": type(error).__module__,
            "raised_in": _innermost_module(error.__traceback__),
            "message": str(error)[:2000],
            "traceback": traceback.format_exc()[-6000:],
        }
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["worker_pools_started"] = pools["started"]
    result["settings_reload_equal"] = _settings_reload(output, before, SCHEMA)
    if spec.get("save_graph") and graph is not None:
        import pickle

        with (output / "graph_returned.pkl").open("wb") as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        result["graph"] = _graph_summary(graph)
    except Exception as error:  # noqa: BLE001
        result["graph"] = {"error": f"{type(error).__name__}: {error}"}
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
