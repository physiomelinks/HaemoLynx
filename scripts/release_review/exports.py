"""Reading a run's exports back with other people's tools, and comparing two
runs' outputs.

``export.8b`` asks that each export opens in a third-party tool: the VTK
files in VTK (through pyvista), the CSVs in pandas, the graph in NetworkX
alone -- a pickle that needs HaemoLynx's own classes to load is not open to
anyone without it. Each file is also matched to its entry in the data
dictionary (``haemolynx.pipeline.export_dictionary``), and the columns or
arrays the entry does not describe are listed (``export.6a``).

``repro.6c`` and ``repro.8b`` ask that the same input and settings give
identical outputs, run twice and with or without worker processes.
:func:`compare_folders` says file by file where two runs' outputs differ:
the CSVs byte for byte, the graph node for node and edge for edge with every
attribute, the VTK files array for array; the manifest only outside the
timings it exists to record.
"""
from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

#: Modules a graph pickle may need: NetworkX, NumPy and Python's own.
GRAPH_PICKLE_MODULES = ("networkx", "numpy", "builtins", "collections", "copyreg", "_codecs", "datetime")

#: The scorer's own files in a run's folder, never the run's outputs.
SCORER_FILES = frozenset({"result.json", "spec.json", "run.log", "readback.json"})

#: Manifest keys that hold when and how long, and so differ between any two runs.
VOLATILE_MANIFEST_KEYS = frozenset({
    "started", "finished", "wall_seconds", "peak_rss_bytes", "process_peak_rss_bytes",
    "worker_peak_rss_bytes", "peak_rss_method", "argv", "mtime", "modified", "checksum_seconds",
    "python", "platform", "dependencies", "package_dir", "git",
})

#: Outputs whose bytes may differ with nothing changed: plots carry render
#: IDs and timestamps. Compared, but a difference there is not counted.
PLOT_SUFFIXES = (".png", ".html", ".svg", ".jpg")

#: A CSV line stamping when it was written (the statistics CSVs' metadata
#: row): the one line two identical runs' CSVs may differ in.
EXPORT_TIMESTAMP_LINE = re.compile(rb"Exported At")


class ForeignClass(pickle.UnpicklingError):
    """A pickle needed a class from outside :data:`GRAPH_PICKLE_MODULES`."""


class _GraphUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".", 1)[0] not in GRAPH_PICKLE_MODULES:
            raise ForeignClass(f"needs {module}.{name}")
        return super().find_class(module, name)


def load_graph(path: Path):
    """The pickled graph, loaded with NetworkX and NumPy only."""
    with Path(path).open("rb") as handle:
        return _GraphUnpickler(handle).load()


# ---------------------------------------------------------------------------
# Opening each export
# ---------------------------------------------------------------------------


def _open_vtk(path: Path) -> dict[str, Any]:
    import pyvista

    mesh = pyvista.read(str(path))
    return {
        "reader": f"VTK (pyvista {pyvista.__version__})",
        "names": list(mesh.point_data.keys()) + list(mesh.cell_data.keys()),
        "field_data": list(mesh.field_data.keys()),
        "cells": int(mesh.n_cells),
    }


def _open_csv(path: Path) -> dict[str, Any]:
    import pandas

    frame = pandas.read_csv(path)
    return {"reader": f"pandas {pandas.__version__}", "names": [str(name) for name in frame.columns], "rows": len(frame)}


def _open_graph(path: Path) -> dict[str, Any]:
    import networkx

    graph = load_graph(path)
    if not isinstance(graph, networkx.Graph):
        raise TypeError(f"holds a {type(graph).__name__}, not a NetworkX graph")
    return {"reader": f"NetworkX {networkx.__version__} alone",
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


def _open_npy(path: Path) -> dict[str, Any]:
    import numpy

    array = numpy.load(path, allow_pickle=False)
    return {"reader": f"NumPy {numpy.__version__}", "shape": list(array.shape)}


def _open_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"reader": "json", "names": list(data) if isinstance(data, dict) else []}


def _open_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {"reader": "PyYAML", "keys": len(data) if isinstance(data, dict) else 0}


def _open_png(path: Path) -> dict[str, Any]:
    from matplotlib import image

    return {"reader": "matplotlib", "shape": list(image.imread(path).shape)}


def _open_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("empty")
    return {"reader": "text", "characters": len(text)}


OPENERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    ".vtp": _open_vtk, ".vtu": _open_vtk, ".vtk": _open_vtk,
    ".csv": _open_csv, ".pkl": _open_graph, ".npy": _open_npy, ".json": _open_json,
    ".yaml": _open_yaml, ".yml": _open_yaml, ".png": _open_png, ".html": _open_text, ".txt": _open_text,
}


def name_pattern(name: str) -> "re.Pattern[str]":
    """A dictionary entry's file name as a pattern: ``{a|b}`` is either,
    any other ``{...}``, ``<...>`` or ``*`` any text."""
    parts = []
    for token in re.split(r"(\{[^}]*\}|<[^>]*>|\*)", name):
        if token.startswith("{") and "|" in token:
            parts.append("(?:" + "|".join(re.escape(each) for each in token[1:-1].split("|")) + ")")
        elif token.startswith(("{", "<")) or token == "*":
            parts.append(".+")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts))


def match_entry(file_name: str, entry_names: Sequence[str]) -> Optional[str]:
    """The dictionary entry naming *file_name*; an entry may list several
    names, comma-separated. Where several match, the one with the most
    literal text wins: ``{stem}_branch_statistics.csv`` over
    ``{stem}_statistics.csv``."""
    best, best_literal = None, -1
    for entry in entry_names:
        for each in (part.strip() for part in entry.split(",")):
            if each.startswith("...") or not name_pattern(each).fullmatch(file_name):
                continue
            literal = len(re.sub(r"\{[^}]*\}|<[^>]*>|\*", "", each))
            if literal > best_literal:
                best, best_literal = entry, literal
    return best


@dataclass
class ReadBack:
    file: str
    entry: Optional[str]
    opened: bool
    reader: str = ""
    error: str = ""
    undocumented: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def read_back(
    folder: Path,
    entry_names: Sequence[str],
    undocumented: Optional[Callable[[str, Iterable[str]], list[str]]] = None,
) -> list[ReadBack]:
    """Open every output under *folder* with its third-party reader, and
    match it to its dictionary entry."""
    found = []
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.name in SCORER_FILES:
            continue
        relative = path.relative_to(folder).as_posix()
        entry = match_entry(path.name, entry_names)
        opener = OPENERS.get(path.suffix.lower())
        if opener is None:
            found.append(ReadBack(relative, entry, False, error=f"no reader for {path.suffix}"))
            continue
        try:
            details = opener(path)
        except Exception as error:  # noqa: BLE001 - the export failed to open
            found.append(ReadBack(relative, entry, False, error=f"{type(error).__name__}: {error}"))
            continue
        names = details.pop("names", None)
        missing = []
        # A CSV's columns and a VTK file's arrays; a JSON file's keys are
        # the report's own rows, described by reference.
        if entry is not None and names and undocumented is not None and path.suffix.lower() != ".json":
            try:
                missing = undocumented(entry, names)
            except KeyError:
                missing = []
        found.append(ReadBack(relative, entry, True, details.pop("reader"), "", missing, details))
    return found


# ---------------------------------------------------------------------------
# Two runs' outputs
# ---------------------------------------------------------------------------


def same_value(a: Any, b: Any) -> bool:
    """Equal, with NaN equal to NaN and arrays compared element by element."""
    try:
        import numpy
    except ImportError:  # pragma: no cover
        numpy = None
    if numpy is not None and (isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray)):
        a, b = numpy.asarray(a), numpy.asarray(b)
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            return False
        if a.dtype.kind in "fc":
            return bool(numpy.array_equal(a, b, equal_nan=True))
        return bool(numpy.array_equal(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(same_value(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - incomparable means different
        return False


def graph_differences(first, second, *, limit: int = 5) -> list[str]:
    """How two graphs differ: nodes, edges, and any attribute of either."""
    differences = []
    if set(first.nodes) != set(second.nodes):
        differences.append(f"nodes differ: {len(set(first.nodes) ^ set(second.nodes))} in only one")
    if first.is_multigraph() and second.is_multigraph():
        edges_a = {(u, v, k): d for u, v, k, d in first.edges(keys=True, data=True)}
        edges_b = {(u, v, k): d for u, v, k, d in second.edges(keys=True, data=True)}
    else:
        edges_a = {(u, v): d for u, v, d in first.edges(data=True)}
        edges_b = {(u, v): d for u, v, d in second.edges(data=True)}
    if edges_a.keys() != edges_b.keys():
        differences.append(f"edges differ: {len(edges_a.keys() ^ edges_b.keys())} in only one")
    if not same_value(dict(first.graph), dict(second.graph)):
        keys = sorted(key for key in set(first.graph) | set(second.graph)
                      if not same_value(first.graph.get(key), second.graph.get(key)))
        differences.append(f"graph attributes differ: {', '.join(map(str, keys[:limit]))}")
    for node in set(first.nodes) & set(second.nodes):
        if not same_value(first.nodes[node], second.nodes[node]):
            differences.append(f"node {node!r} attributes differ")
            break
    changed = [key for key in edges_a.keys() & edges_b.keys() if not same_value(edges_a[key], edges_b[key])]
    if changed:
        attributes = sorted({
            name for key in changed for name in set(edges_a[key]) | set(edges_b[key])
            if not same_value(edges_a[key].get(name), edges_b[key].get(name))
        })
        differences.append(f"{len(changed)} edges' attributes differ: {', '.join(attributes[:limit])}")
    return differences


def _without_timestamp(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().splitlines() if not EXPORT_TIMESTAMP_LINE.search(line)]


def _without(data: Any, keys: frozenset) -> Any:
    if isinstance(data, Mapping):
        return {key: _without(value, keys) for key, value in data.items() if key not in keys}
    if isinstance(data, list):
        return [_without(value, keys) for value in data]
    return data


def _vtk_differences(first: Path, second: Path) -> list[str]:
    import pyvista

    a, b = pyvista.read(str(first)), pyvista.read(str(second))
    differences = []
    if not same_value(a.points, b.points):
        differences.append("points differ")
    for kind in ("point_data", "cell_data", "field_data"):
        data_a, data_b = getattr(a, kind), getattr(b, kind)
        if set(data_a.keys()) != set(data_b.keys()):
            differences.append(f"{kind} arrays differ in name")
        for name in set(data_a.keys()) & set(data_b.keys()):
            if not same_value(data_a[name], data_b[name]):
                differences.append(f"{kind} {name} differs")
    return differences


@dataclass(frozen=True)
class Difference:
    file: str
    detail: str
    counted: bool = True


def compare_folders(first: Path, second: Path) -> list[Difference]:
    """Every way the outputs in *second* differ from those in *first*."""
    first, second = Path(first), Path(second)

    def files(folder: Path) -> dict[str, Path]:
        return {
            path.relative_to(folder).as_posix(): path
            for path in folder.rglob("*") if path.is_file() and path.name not in SCORER_FILES
        }

    a, b = files(first), files(second)
    differences = [Difference(name, "only in the first run") for name in sorted(a.keys() - b.keys())]
    differences += [Difference(name, "only in the second run") for name in sorted(b.keys() - a.keys())]
    for name in sorted(a.keys() & b.keys()):
        path_a, path_b = a[name], b[name]
        if path_a.read_bytes() == path_b.read_bytes():
            continue
        suffix = path_a.suffix.lower()
        if suffix in PLOT_SUFFIXES:
            differences.append(Difference(name, "bytes differ (a plot)", counted=False))
        elif suffix == ".pkl":
            found = graph_differences(load_graph(path_a), load_graph(path_b))
            if found:
                differences.append(Difference(name, "; ".join(found)))
        elif suffix in (".vtp", ".vtu", ".vtk"):
            found = _vtk_differences(path_a, path_b)
            if found:
                differences.append(Difference(name, "; ".join(found)))
        elif suffix == ".csv" and _without_timestamp(path_a) == _without_timestamp(path_b):
            differences.append(Difference(name, "the export timestamp only", counted=False))
        elif suffix == ".json" and name.endswith("_run_manifest.json"):
            if not same_value(
                _without(json.loads(path_a.read_text(encoding="utf-8")), VOLATILE_MANIFEST_KEYS),
                _without(json.loads(path_b.read_text(encoding="utf-8")), VOLATILE_MANIFEST_KEYS),
            ):
                differences.append(Difference(name, "differs beyond its times and memory"))
        else:
            differences.append(Difference(name, "bytes differ"))
    return differences
