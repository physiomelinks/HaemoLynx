"""Compare two batch runs by content, part by part, through the batch-run reader.

Use it to check a re-run against its archive: pickle bytes change between runs even when the
graph does not, so files are compared by what they hold. Each side is opened with
``open_batch_run``, so a run not cut at the placed ROI is refused before anything is compared.

What counts as equal, part by part (``PARTS``):

- **graph**: the same graph type and ``graph`` attributes, the same nodes, the same edges on
  forward ``(u, v, key)``, and every node and edge attribute equal. The diameter the reader
  joins on is left to the next part.
- **diameters**: ``assigned_diameter_um`` on each graph edge, as the reader joined it.
- **edge table**: the same columns and row keys, then every shared cell. ``TEXT_COLUMNS`` are
  compared as written. ``INTEGER_COLUMNS`` and every other column are read through
  ``BatchRun.numeric_column``, so a blank or text cell raises instead of being compared.
- **skeleton**, **vessel mask**: every voxel.
- **TH mask**: every voxel. The TH channel is not stored in a run: both sides read the
  specimen's one TH export at the placed ROI, which the reader already makes both sides share.
  So under the reader this part can only show that both runs share source and box, and the
  report says so.

Floats are compared exactly unless ``rtol`` is given; then a float ``a`` against ``b`` is equal
if ``|a - b| <= rtol * |b|``. Keys, integers, text and masks are always exact. A NaN equals a
NaN, so a value missing on both sides in the same way is the same content.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .batch_outputs import DIAMETER, open_batch_run
from .specimens import is_sensitivity_run_name

GRAPH, DIAMETERS, EDGE_TABLE, SKELETON, VESSEL_MASK, TH_MASK = PARTS = (
    "graph", "diameters", "edge table", "skeleton", "vessel mask", "TH mask")

# What the report adds to a matching TH mask: see the module docstring.
TH_MASK_SAME_MEANS = "same source, same box"

# Edge-table columns that key a row rather than hold a value.
KEY_COLUMNS = ("u", "v", "key")
# Edge-table columns compared as the text the file holds. Every other column is a number,
# read through ``BatchRun.numeric_column`` so a blank or text cell raises.
TEXT_COLUMNS = (
    "diameter_provenance", "edt_junction_trim", "centreline_smoothing", "branch_order",
    "reconnected",
    # Blank in every row of every run as of 2026-10-10 (no FWHM calibre is measured), which
    # numeric_column refuses. Compared as text, blank equals blank and a value appearing on
    # one side still differs; nothing is turned into a number.
    "fwhm_diameter_um",
)
# Numeric edge-table columns that hold counts, which a relative tolerance never loosens.
INTEGER_COLUMNS = ("n_centreline_points",)

# How many keys or attribute names a detail lists before it stops.
MAX_LISTED = 10


@dataclass(frozen=True)
class PartVerdict:
    """One part of one run: ``same``, or not with a short ``detail`` of what differs."""
    part: str
    same: bool
    detail: str = ""


@dataclass(frozen=True)
class RunComparison:
    """Every part of one run compared, or ``only_in`` the side ("OLD" or "NEW") that has it."""
    label: str
    parts: tuple[PartVerdict, ...] = ()
    only_in: str | None = None

    @property
    def same(self) -> bool:
        return self.only_in is None and all(verdict.same for verdict in self.parts)


class RootsDoNotMatch(ValueError):
    """One root holds sensitivity runs and the other holds batch runs."""


def compare_roots(old_root, new_root, specimens, rtol: float = 0.0) -> list[RunComparison]:
    """Compare every specimen's run under two roots, in the order of ``specimens``.

    A root is a batch root (``<root>/<specimen>/``) or a sensitivity root
    (``<root>/t<threshold>/<specimen>/``); both roots must be the same kind. A threshold or
    specimen folder that only one root has is a difference. One that neither has is opened
    anyway, so the reader refuses it rather than the comparison passing without it.
    """
    old_root, new_root = Path(old_root), Path(new_root)
    old_thresholds, new_thresholds = _thresholds(old_root), _thresholds(new_root)
    if bool(old_thresholds) != bool(new_thresholds):
        raise RootsDoNotMatch(
            f"{old_root} and {new_root} are not the same kind of root: one holds "
            f"t<threshold>/ sensitivity folders and the other does not.")
    if not old_thresholds:
        return _compare_specimens(old_root, new_root, specimens, rtol, prefix="")
    comparisons = []
    for name in sorted(set(old_thresholds) | set(new_thresholds)):
        only_in = _only_in(name in old_thresholds, name in new_thresholds)
        if only_in:
            comparisons.append(RunComparison(name, only_in=only_in))
        else:
            comparisons += _compare_specimens(old_root / name, new_root / name, specimens,
                                              rtol, prefix=f"{name}/")
    return comparisons


def _thresholds(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir()
                  if path.is_dir() and is_sensitivity_run_name(path.name))


def _only_in(in_old: bool, in_new: bool) -> str | None:
    if in_old == in_new:
        return None
    return "OLD" if in_old else "NEW"


def _compare_specimens(old_root: Path, new_root: Path, specimens, rtol, prefix: str) -> list:
    comparisons = []
    for specimen in specimens:
        label = prefix + specimen.specimen_id
        old_dir, new_dir = old_root / specimen.specimen_id, new_root / specimen.specimen_id
        only_in = _only_in(old_dir.is_dir(), new_dir.is_dir())
        if only_in:
            comparisons.append(RunComparison(label, only_in=only_in))
            continue
        comparisons.append(compare_runs(open_batch_run(specimen, old_dir),
                                        open_batch_run(specimen, new_dir), rtol, label))
    return comparisons


def compare_runs(old, new, rtol: float = 0.0, label: str = "") -> RunComparison:
    """Compare two opened batch runs in every part of ``PARTS``, without stopping early."""
    old_graph, new_graph = old.graph(), new.graph()
    problems = {
        GRAPH: _graph_problems(old_graph, new_graph, rtol),
        DIAMETERS: _numbers(*(_diameters(G) for G in (old_graph, new_graph)), "edge", rtol),
        EDGE_TABLE: _edge_table_problems(old, new, rtol),
        SKELETON: _voxel_problems(old.skeleton(), new.skeleton()),
        VESSEL_MASK: _voxel_problems(old.vessel_mask(), new.vessel_mask()),
        TH_MASK: _voxel_problems(old.th_mask(), new.th_mask()),
    }
    return RunComparison(label or old.specimen.specimen_id, tuple(
        PartVerdict(part, not problems[part], "; ".join(problems[part])) for part in PARTS))


def format_report(comparisons) -> str:
    """One block per run, a line per part, and a closing count."""
    lines = []
    for comparison in comparisons:
        if comparison.only_in:
            lines.append(f"{comparison.label}: DIFFERENT, only in {comparison.only_in}")
            continue
        lines.append(comparison.label)
        lines += [f"  {verdict.part:<12} {_verdict_text(verdict)}" for verdict in comparison.parts]
    differing = sum(not comparison.same for comparison in comparisons)
    lines.append(f"{len(comparisons)} runs compared: "
                 + (f"{differing} differ." if differing else "all match."))
    return "\n".join(lines)


def _verdict_text(verdict: PartVerdict) -> str:
    if not verdict.same:
        return f"DIFFERENT: {verdict.detail}"
    return f"same ({TH_MASK_SAME_MEANS})" if verdict.part == TH_MASK else "same"


def _voxel_problems(old, new) -> list[str]:
    differing = int(np.count_nonzero(old != new))
    return [f"{differing} of {old.size} voxels differ"] if differing else []


def _graph_problems(old, new, rtol) -> list[str]:
    """The graph type, its attributes, then the nodes and edges and their attributes. A graph
    can differ in many places, so the list stops after ``MAX_LISTED``."""
    problems = []
    if type(old) is not type(new):
        problems.append(f"graph type {type(old).__name__} in OLD, {type(new).__name__} in NEW")
    problems += [f"graph {name}" for name in _differing(old.graph, new.graph, rtol)]
    problems += _keyed(dict(old.nodes(data=True)), dict(new.nodes(data=True)), "node", rtol)
    problems += _keyed(_edges(old), _edges(new), "edge", rtol)
    if len(problems) > MAX_LISTED:
        problems = problems[:MAX_LISTED] + [f"and {len(problems) - MAX_LISTED} more"]
    return problems


def _edges(G) -> dict:
    """Each edge's attributes by forward ``(u, v, key)``, without the joined-on diameter."""
    return {(int(u), int(v), int(k)): {name: value for name, value in data.items()
                                       if name != DIAMETER}
            for u, v, k, data in G.edges(keys=True, data=True)}


def _diameters(G) -> dict:
    return {(int(u), int(v), int(k)): data[DIAMETER]
            for u, v, k, data in G.edges(keys=True, data=True)}


def _keyed(old: dict, new: dict, what: str, rtol) -> list[str]:
    """Keys on one side only, then each shared key's differing attribute names."""
    problems = _one_side_only(old, new, what)
    for key in sorted(key for key in old if key in new):
        problems += [f"{what} {key} {name}" for name in _differing(old[key], new[key], rtol)]
    return problems


def _one_side_only(old, new, what: str) -> list[str]:
    problems = []
    for side, mine, theirs in (("OLD", old, new), ("NEW", new, old)):
        only = sorted(key for key in mine if key not in theirs)
        if only:
            problems.append(f"{len(only)} {what}s only in {side}: {only[:MAX_LISTED]}")
    return problems


def _edge_table_problems(old, new, rtol) -> list[str]:
    """The header, the row keys, then each shared column over the shared rows. One line per
    differing column, none left out."""
    old_rows, new_rows = old.edge_table(), new.edge_table()
    old_columns, new_columns = (list(next(iter(rows.values()), {}))
                                for rows in (old_rows, new_rows))
    problems = [f"columns only in {side}: {[c for c in mine if c not in theirs]}"
                for side, mine, theirs in (("OLD", old_columns, new_columns),
                                           ("NEW", new_columns, old_columns))
                if any(c not in theirs for c in mine)]
    problems += _one_side_only(old_rows, new_rows, "row")
    shared = [edge for edge in old_rows if edge in new_rows]
    for column in old_columns:
        if column in KEY_COLUMNS or column not in new_columns:
            continue
        if column in TEXT_COLUMNS:
            differing = [edge for edge in shared
                         if old_rows[edge][column] != new_rows[edge][column]]
            if differing:
                problems.append(f"{column}: {len(differing)} of {len(shared)} rows differ: "
                                f"{differing[:MAX_LISTED]}")
            continue
        old_values, new_values = old.numeric_column(column), new.numeric_column(column)
        problems += [f"{column}: {problem}" for problem in _numbers(
            {edge: old_values[edge] for edge in shared},
            {edge: new_values[edge] for edge in shared}, "row",
            0.0 if column in INTEGER_COLUMNS else rtol)]
    return problems


def _numbers(old: dict, new: dict, what: str, rtol) -> list[str]:
    """Keys on one side only, then how many shared values differ and by how much."""
    problems = _one_side_only(old, new, what)
    shared = [key for key in old if key in new]
    differing = [key for key in shared if not _values_equal(old[key], new[key], rtol)]
    if differing:
        gaps = np.array([abs(new[key] - old[key]) for key in differing])
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = gaps / np.array([abs(old[key]) for key in differing])
        problems.append(f"{len(differing)} of {len(shared)} {what}s differ, max abs "
                        f"{gaps.max():.3g}, max rel {relative.max():.3g}: "
                        f"{differing[:MAX_LISTED]}")
    return problems


def _differing(old: dict, new: dict, rtol) -> list:
    """Names of attributes missing on one side or holding different values."""
    return [name for name in sorted(set(old) | set(new), key=str)
            if name not in old or name not in new
            or not _values_equal(old[name], new[name], rtol)]


def _values_equal(old, new, rtol) -> bool:
    """Same type and same value; floats within ``rtol`` (exact at 0), NaN equal to NaN."""
    if type(old) is not type(new):
        return False
    if isinstance(old, dict):
        return not _differing(old, new, rtol)
    if isinstance(old, (list, tuple)):
        return len(old) == len(new) and all(
            _values_equal(a, b, rtol) for a, b in zip(old, new))
    if isinstance(old, np.ndarray):
        if old.shape != new.shape or old.dtype != new.dtype:
            return False
        if old.dtype.kind == "f":
            return bool(np.all(np.isclose(old, new, rtol=rtol, atol=0, equal_nan=True)))
        return bool(np.array_equal(old, new))
    if isinstance(old, (float, np.floating)):
        return bool(np.isclose(old, new, rtol=rtol, atol=0, equal_nan=True))
    return bool(old == new)
