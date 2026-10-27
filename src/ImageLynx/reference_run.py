"""The reference run: chosen outputs of one pipeline run on the benchmark sub-volume, kept to
compare later runs with, part by part.

``write_reference_run`` stores the parts of a pipeline run folder in a reference-run folder;
``compare_with_reference`` checks a run folder against one and returns a verdict per part;
``format_report`` prints it in the style of ``batch_compare.format_report``. A run on the
sub-volume is not a batch run, so nothing here goes through ``open_batch_run`` (which keeps its
``place_roi`` check); the run folder is read directly, its mask, skeleton and graph from its one
``*_cache`` folder.

Parts (``PARTS``), how each is stored, and what counts as equal:

- **vessel mask**, **skeleton**: both in ``masks.npz`` (compressed, bit-packed, with their
  shapes). Every voxel, exactly.
- **graph**: ``graph.json``: the graph type, the ``graph`` attributes, the nodes, and the edges
  on forward ``(u, v, key)``, every attribute (arrays and tuples as lists). The node and edge
  sets, integers, booleans and text exactly; floats under the geometry rule.
- **edge table**: ``per_edge_morphometry.csv`` as written, assigned diameters included. The
  columns (and their order) and row keys exactly; ``batch_compare.TEXT_COLUMNS`` as text and
  ``batch_compare.INTEGER_COLUMNS`` exactly; every other column under the geometry rule. Every
  number is read through ``batch_outputs.numeric_cells``, so a blank or text cell raises.

The tolerance rule: a float passes when ``|new - ref| <= rtol * |ref| + atol`` with
``atol = rtol * max|ref|``, the largest finite magnitude of that graph attribute or edge-table
column in the reference. Geometry floats use ``GEOMETRY_RTOL``. A NaN equals a NaN.

``manifest.json`` (commit, Python, package versions, command line) is not a part: a different
environment is a note in the report, so a failure can be read against it, never a failure itself.
A file missing on either side is a difference in its part.
"""
from __future__ import annotations

import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from .batch_compare import INTEGER_COLUMNS, KEY_COLUMNS, MAX_LISTED, TEXT_COLUMNS, PartVerdict
from .batch_outputs import (CACHE_GLOB, EDGE_TABLE_NAME, GRAPH_NAME, SKELETON_NAME,
                            VESSEL_MASK_NAME, numeric_cells, read_edge_table)

VESSEL_MASK, SKELETON, GRAPH, EDGE_TABLE = PARTS = (
    "vessel mask", "skeleton", "graph", "edge table")

MASKS_NAME = "masks.npz"
GRAPH_JSON_NAME = "graph.json"
MANIFEST_NAME = "manifest.json"
# The voxel parts: the array name inside masks.npz and the file the pipeline caches.
VOXEL_PARTS = {VESSEL_MASK: ("vessel_mask", VESSEL_MASK_NAME),
               SKELETON: ("skeleton", SKELETON_NAME)}
SHAPE_SUFFIX = "_shape"

# Relative tolerance for geometry floats: coordinates, lengths, diameters, tortuosity.
GEOMETRY_RTOL = 1e-9

# Packages whose versions the manifest records: the ones the pipeline's numbers go through.
PACKAGES = ("numpy", "scipy", "scikit-image", "skan", "networkx", "python-igraph", "numba",
            "dask", "h5py", "joblib", "pandas", "pyvista", "vtk")

REFERENCE, NEW = "REFERENCE", "NEW"
# Stands for an attribute a node or edge does not carry.
_ABSENT = object()


@dataclass(frozen=True)
class ReferenceComparison:
    """Every part of a run compared with a reference run, and how their environments differ."""
    parts: tuple[PartVerdict, ...]
    environment: tuple[str, ...]

    @property
    def same(self) -> bool:
        return all(verdict.same for verdict in self.parts)

    def verdict(self, part: str) -> PartVerdict:
        return next(verdict for verdict in self.parts if verdict.part == part)


def environment() -> dict:
    """This interpreter's Python and package versions, as the manifest records them."""
    return {"python": sys.version.split()[0],
            "packages": {name: metadata.version(name) for name in PACKAGES}}


def write_reference_run(run_dir, reference_dir, command, commit: str) -> dict:
    """Store ``run_dir``'s parts in ``reference_dir`` with a manifest; returns the manifest.

    ``command`` is the pipeline command line that made the run, as it should be recorded
    (machine paths already replaced); ``commit`` the HEAD it ran at. The folder is replaced
    whole, so a file left from an earlier part list does not stay. A part file missing from the
    run raises: a reference run is never partial.
    """
    run_dir, reference_dir = Path(run_dir), Path(reference_dir)
    cache = _cache_dir(run_dir)
    if cache is None:
        raise FileNotFoundError(f"{run_dir} has no {CACHE_GLOB} folder; it is not a run folder.")
    arrays = {}
    for part, (name, file_name) in VOXEL_PARTS.items():
        voxels = _voxels(cache / file_name)
        arrays[name] = np.packbits(voxels, axis=None)
        arrays[name + SHAPE_SUFFIX] = np.array(voxels.shape, dtype=np.int64)
    graph = graph_to_json(_load_pickle(cache / GRAPH_NAME))
    edge_table = run_dir / EDGE_TABLE_NAME
    read_edge_table(edge_table, str(run_dir))  # a duplicate or unkeyed row raises here
    staging = reference_dir.with_name(reference_dir.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    np.savez_compressed(staging / MASKS_NAME, **arrays)
    (staging / GRAPH_JSON_NAME).write_text(json.dumps(graph, indent=1) + "\n")
    shutil.copyfile(edge_table, staging / EDGE_TABLE_NAME)
    manifest = {"commit": commit, **environment(), "command": list(command)}
    (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    staging.rename(reference_dir)
    return manifest


def compare_with_reference(run_dir, reference_dir, command=None) -> ReferenceComparison:
    """Compare ``run_dir`` with ``reference_dir`` in every part of ``PARTS``, without stopping
    early. A missing file is a difference in its part; a blank or text number raises.

    ``command``, the recorded command line that made ``run_dir``, is checked against the
    manifest's when given; like the Python and package versions, a difference is a note.
    """
    run_dir, reference_dir = Path(run_dir), Path(reference_dir)
    cache = _cache_dir(run_dir)
    problems = {}
    for part, (name, file_name) in VOXEL_PARTS.items():
        problems[part] = _voxel_part(reference_dir / MASKS_NAME, name,
                                     None if cache is None else cache / file_name)
    problems[GRAPH] = _graph_part(reference_dir / GRAPH_JSON_NAME,
                                  None if cache is None else cache / GRAPH_NAME)
    problems[EDGE_TABLE] = _edge_table_part(reference_dir / EDGE_TABLE_NAME,
                                            run_dir / EDGE_TABLE_NAME)
    return ReferenceComparison(
        tuple(PartVerdict(part, not problems[part], "; ".join(problems[part])) for part in PARTS),
        _environment_notes(reference_dir / MANIFEST_NAME, command))


def format_report(comparison: ReferenceComparison, label: str = "run") -> str:
    """A line per part, a line per environment difference, and a closing count."""
    lines = [label]
    for verdict in comparison.parts:
        text = "same" if verdict.same else f"DIFFERENT: {verdict.detail}"
        lines.append(f"  {verdict.part:<12} {text}")
    lines += [f"  environment  {note}" for note in comparison.environment]
    differing = sum(not verdict.same for verdict in comparison.parts)
    lines.append(f"{len(comparison.parts)} parts compared: "
                 + (f"{differing} differ." if differing else "all match."))
    return "\n".join(lines)


# --- Reading a run folder ------------------------------------------------------------------

def _cache_dir(run_dir: Path) -> Path | None:
    """The run's one cache folder, None when it has none; more than one raises."""
    caches = sorted(path for path in run_dir.glob(CACHE_GLOB) if path.is_dir())
    if len(caches) > 1:
        raise ValueError(f"{run_dir} holds {len(caches)} {CACHE_GLOB} folders "
                         f"({[path.name for path in caches]}); a run folder has one.")
    return caches[0] if caches else None


def _voxels(path: Path) -> np.ndarray:
    """A cached mask or skeleton as booleans; any value other than 0 or 1 raises."""
    array = np.load(path)
    if array.dtype != bool:
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{path} holds values other than 0 and 1; it is not a mask.")
        array = array.astype(bool)
    return array


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def graph_to_json(G) -> dict:
    """The graph as plain JSON values: type, ``graph`` attributes, nodes, and edges on forward
    ``(u, v, key)``, in the graph's own order. A value of a type JSON cannot hold raises."""
    return {"type": type(G).__name__,
            "graph": _json_value(dict(G.graph)),
            "nodes": [{"id": _json_value(node), "attributes": _json_value(data)}
                      for node, data in G.nodes(data=True)],
            "edges": [{"u": int(u), "v": int(v), "key": int(k), "attributes": _json_value(data)}
                      for u, v, k, data in G.edges(keys=True, data=True)]}


def _json_value(value):
    """``value`` with numpy scalars and arrays and tuples made into JSON values."""
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        if len(set(keys)) != len(keys):
            raise TypeError(f"dict keys {list(value)} collide as text.")
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{type(value).__name__} value {value!r} has no JSON form.")


# --- Comparing parts -----------------------------------------------------------------------

def _missing(reference: Path, new: Path | None) -> list[str]:
    problems = []
    if not reference.exists():
        problems.append(f"missing in {REFERENCE}: {reference.name}")
    if new is None or not new.exists():
        problems.append(f"missing in {NEW}: {new.name if new else 'the ' + CACHE_GLOB + ' folder'}")
    return problems


def _voxel_part(masks: Path, name: str, new_path: Path | None) -> list[str]:
    missing = _missing(masks, new_path)
    if missing:
        return missing
    with np.load(masks) as stored:
        if name not in stored:
            return [f"missing in {REFERENCE}: {name} in {masks.name}"]
        shape = tuple(int(n) for n in stored[name + SHAPE_SUFFIX])
        reference = np.unpackbits(stored[name], count=math.prod(shape)).reshape(shape).astype(bool)
    new = _voxels(new_path)
    if new.shape != reference.shape:
        return [f"shape {reference.shape} in {REFERENCE}, {new.shape} in {NEW}"]
    differing = int(np.count_nonzero(reference != new))
    return [f"{differing} of {reference.size} voxels differ"] if differing else []


def _graph_part(reference_path: Path, new_path: Path | None) -> list[str]:
    missing = _missing(reference_path, new_path)
    if missing:
        return missing
    reference = json.loads(reference_path.read_text())
    # Through text and back, so both sides are in the stored form (a tuple is a list, say).
    new = json.loads(json.dumps(graph_to_json(_load_pickle(new_path))))
    return _graph_problems(reference, new, GEOMETRY_RTOL)


def _graph_problems(reference: dict, new: dict, rtol: float) -> list[str]:
    """Where two graphs in ``graph_to_json`` form differ: one line per graph attribute and per
    node or edge attribute name, with counts and the largest gap. Stops after ``MAX_LISTED``."""
    problems = []
    if reference["type"] != new["type"]:
        problems.append(f"graph type {reference['type']} in {REFERENCE}, {new['type']} in {NEW}")
    problems += _attribute_problems({(): reference["graph"]}, {(): new["graph"]}, "graph", rtol)
    for what, key in (("node", lambda item: item["id"]),
                      ("edge", lambda item: (item["u"], item["v"], item["key"]))):
        ref_items = _keyed_items(reference[what + "s"], key, what)
        new_items = _keyed_items(new[what + "s"], key, what)
        problems += _one_side_only(ref_items, new_items, what)
        shared = [k for k in ref_items if k in new_items]
        problems += _attribute_problems({k: ref_items[k] for k in shared},
                                        {k: new_items[k] for k in shared}, what, rtol)
    if len(problems) > MAX_LISTED:
        problems = problems[:MAX_LISTED] + [f"and {len(problems) - MAX_LISTED} more"]
    return problems


def _keyed_items(items: list, key, what: str) -> dict:
    keyed = {}
    for item in items:
        k = key(item)
        k = tuple(k) if isinstance(k, list) else k
        if k in keyed:
            raise ValueError(f"{what} {k} appears twice in one graph.")
        keyed[k] = item["attributes"]
    return keyed


def _one_side_only(reference: dict, new: dict, what: str) -> list[str]:
    problems = []
    for side, mine, theirs in ((REFERENCE, reference, new), (NEW, new, reference)):
        only = [key for key in mine if key not in theirs]
        if only:
            problems.append(f"{len(only)} {what}s only in {side}: {only[:MAX_LISTED]}")
    return problems


def _attribute_problems(reference: dict, new: dict, what: str, rtol: float) -> list[str]:
    """``reference`` and ``new`` map the same keys to attribute dicts. One line per attribute
    name that differs anywhere; floats under the tolerance rule, scaled per attribute name."""
    names = sorted({name for attrs in (*reference.values(), *new.values()) for name in attrs})
    problems = []
    for name in names:
        values = {key: (attrs.get(name, _ABSENT), new[key].get(name, _ABSENT))
                  for key, attrs in reference.items()}
        scale = max((abs(x) for ref, _ in values.values() for x in _floats(ref)
                     if math.isfinite(x)), default=0.0)
        atol = rtol * scale
        differing, gaps = [], []
        for key, (ref, other) in values.items():
            pairs = _float_pairs(ref, other)
            if pairs is None:
                differing.append(key)
                continue
            outside = [(r, n) for r, n in pairs if not _close(r, n, rtol, atol)]
            if outside:
                differing.append(key)
                gaps += outside
        if differing:
            problems.append(f"{what} {name}: {len(differing)} of {len(values)} {what}s differ"
                            + _gap_text(gaps) + f": {differing[:MAX_LISTED]}")
    return problems


def _floats(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _floats(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)


def _float_pairs(ref, new):
    """The float leaves of two values side by side, or None when they differ in anything but
    a float (a type, a length, a key, an integer, text, or the value missing on one side). A
    value missing on both sides (an attribute only some edges carry) is the same."""
    if ref is _ABSENT and new is _ABSENT:
        return []
    if ref is _ABSENT or new is _ABSENT or type(ref) is not type(new):
        return None
    if isinstance(ref, float):
        return [(ref, new)]
    if isinstance(ref, list):
        if len(ref) != len(new):
            return None
        pairs = []
        for a, b in zip(ref, new):
            inner = _float_pairs(a, b)
            if inner is None:
                return None
            pairs += inner
        return pairs
    if isinstance(ref, dict):
        if ref.keys() != new.keys():
            return None
        return _float_pairs(list(ref.values()), list(new.values()))
    return [] if ref == new else None


def _close(ref: float, new: float, rtol: float, atol: float) -> bool:
    """``|new - ref| <= rtol * |ref| + atol``; a NaN equals a NaN, an infinity only itself."""
    if math.isnan(ref) or math.isnan(new):
        return math.isnan(ref) and math.isnan(new)
    if math.isinf(ref) or math.isinf(new):
        return ref == new
    return abs(new - ref) <= rtol * abs(ref) + atol


def _gap_text(pairs) -> str:
    """The largest absolute and relative gap among float pairs, if any is finite."""
    gaps = [(abs(n - r), abs(n - r) / abs(r) if r else math.inf) for r, n in pairs
            if math.isfinite(r) and math.isfinite(n)]
    if not gaps:
        return ""
    return (f", max abs {max(g for g, _ in gaps):.3g}, "
            f"max rel {max(rel for _, rel in gaps):.3g}")


def _edge_table_part(reference_path: Path, new_path: Path) -> list[str]:
    missing = _missing(reference_path, new_path)
    if missing:
        return missing
    reference = read_edge_table(reference_path, REFERENCE)
    new = read_edge_table(new_path, NEW)
    return _edge_table_problems(reference, new, GEOMETRY_RTOL)


def _edge_table_problems(reference: dict, new: dict, rtol: float) -> list[str]:
    """Where two edge tables from ``read_edge_table`` differ: the header, the row keys, then one
    line per differing column over the shared rows. None is left out."""
    ref_columns, new_columns = (list(next(iter(table.values()), {})) for table in (reference, new))
    problems = [f"columns only in {side}: {[c for c in mine if c not in theirs]}"
                for side, mine, theirs in ((REFERENCE, ref_columns, new_columns),
                                           (NEW, new_columns, ref_columns))
                if any(c not in theirs for c in mine)]
    if not problems and ref_columns != new_columns:
        problems.append(f"column order {ref_columns} in {REFERENCE}, {new_columns} in {NEW}")
    problems += _one_side_only(reference, new, "row")
    shared = [edge for edge in reference if edge in new]
    for column in ref_columns:
        if column in KEY_COLUMNS or column not in new_columns:
            continue
        if column in TEXT_COLUMNS:
            differing = [edge for edge in shared if reference[edge][column] != new[edge][column]]
            if differing:
                problems.append(f"{column}: {len(differing)} of {len(shared)} rows differ: "
                                f"{differing[:MAX_LISTED]}")
            continue
        ref_values = numeric_cells(reference, column, REFERENCE)
        new_values = numeric_cells(new, column, NEW)
        column_rtol = 0.0 if column in INTEGER_COLUMNS else rtol
        atol = column_rtol * max(abs(value) for value in ref_values.values())
        differing = [edge for edge in shared
                     if not _close(ref_values[edge], new_values[edge], column_rtol, atol)]
        if differing:
            problems.append(f"{column}: {len(differing)} of {len(shared)} rows differ"
                            + _gap_text([(ref_values[e], new_values[e]) for e in differing])
                            + f": {differing[:MAX_LISTED]}")
    return problems


def _environment_notes(manifest_path: Path, command) -> tuple[str, ...]:
    """How this run's interpreter (and command, when given) differ from the reference run's.
    A manifest without its keys raises."""
    if not manifest_path.exists():
        return (f"no {MANIFEST_NAME} in {REFERENCE}",)
    stored = json.loads(manifest_path.read_text())
    lacking = [key for key in ("python", "packages", "command") if key not in stored]
    if lacking:
        raise ValueError(f"{manifest_path} has no {lacking}; rewrite the reference run.")
    here = environment()
    notes = []
    if stored["python"] != here["python"]:
        notes.append(f"python {stored['python']} in {REFERENCE}, {here['python']} here")
    for name, version in here["packages"].items():
        if stored["packages"].get(name) != version:
            notes.append(f"{name} {stored['packages'].get(name)} in {REFERENCE}, {version} here")
    if command is not None and list(command) != stored["command"]:
        notes.append(f"command {stored['command']} in {REFERENCE}, {list(command)} here")
    return tuple(notes)
