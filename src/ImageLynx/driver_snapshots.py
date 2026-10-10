"""Prove a driver refactor left its outputs unchanged, by running drivers and comparing by content.

A **driver snapshot** (``GLOSSARY.md``) is one folder holding what chosen ``examples/cb_*.py``
drivers wrote, plus their stdout, so a later run can be compared with it:

    <snapshot>/manifest.json            commit, versions, and each driver's command and exit code
    <snapshot>/<driver>/stdout.txt      what the driver printed
    <snapshot>/<driver>/stderr.txt      saved, never compared
    <snapshot>/<driver>/out/...         the files the driver wrote (its ``--out``)

``take_snapshot`` runs the drivers; ``compare_snapshots`` compares two snapshots. The snapshot
holds outputs only, never a driver's internals, so a refactor that changes a private function
cannot make the comparison fail.

What counts as equal is exact. In every text file (stdout, JSON, anything else) the snapshot
folder's own path is first replaced by ``<SNAPSHOT>``. Then, by file type:

- **stdout**: line for line.
- **JSON**: the parsed values; key order is ignored, types must match (``1`` is not ``1.0``),
  floats are exact, NaN equals NaN.
- **PNG**: the decoded pixels, equal in shape and values; metadata is ignored.
- **VTK** (``.vtp``, ``.vti``): the same points and cells (or grid), and the same point, cell
  and field arrays by name, dtype, shape and value; NaN equals NaN.
- **other files**: bytes.

A file only one snapshot has is a difference. ``compare_snapshots`` refuses, rather than
reporting a verdict, when the two snapshots ran different drivers or command lines, a driver
failed, or the Python or a package version differs: those outputs are not comparable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from .batch_compare import MAX_LISTED

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

MANIFEST_NAME = "manifest.json"
STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
OUT_DIR_NAME = "out"
SNAPSHOT_TOKEN = "<SNAPSHOT>"

# Rendering must not need a display, and must be the same backend on both sides.
RUN_ENV = {"MPLBACKEND": "Agg", "PYVISTA_OFF_SCREEN": "true"}
# Their versions are recorded in the manifest; a mismatch makes two snapshots not comparable.
PACKAGES = ("numpy", "scipy", "matplotlib", "pyvista", "vtk", "networkx")

DIR, FILE, STDOUT_ONLY = "dir", "file", "stdout"


@dataclass(frozen=True)
class Driver:
    """One driver and the one command line it is run with.

    ``output`` is how the driver is given somewhere to write: ``dir`` (``--out`` is a folder),
    ``file`` (``--out`` is ``<name>.json``) or ``stdout`` (it writes nothing, so it takes no
    ``--out``). ``inputs`` are ``(flag, driver)`` pairs: the flag points at another driver's
    ``out`` folder in the same snapshot, and that driver runs first.
    """
    name: str
    output: str
    args: tuple[str, ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()
    slow: bool = False


# Every output-writing cb_h1_* and cb_h2_* driver. cb_h1_batch (it runs the pipeline) and
# cb_compare_batch_runs (it only compares) are left out. Order matters: an input comes first.
DRIVERS = (
    Driver("cb_h1_figures", DIR),
    Driver("cb_h1_vtk", DIR),
    Driver("cb_h1_renders", DIR, inputs=(("--vtk-dir", "cb_h1_vtk"),)),
    Driver("cb_h1_th_metrics", FILE, args=("--all",)),
    Driver("cb_h2_glomus_perfusion", FILE),
    Driver("cb_h2_absolute_perfusion", FILE),
    Driver("cb_h2_boundary_selection", STDOUT_ONLY),
    Driver("cb_h2_error_propagation", STDOUT_ONLY),
    Driver("cb_h2_threshold_calibre", STDOUT_ONLY),
    # About 40 and 14 minutes: named with --driver or included with --slow.
    Driver("cb_h2_hypoxic_fraction", FILE, slow=True),
    Driver("cb_h2_vtk", DIR, slow=True),
)


class SnapshotsNotComparable(ValueError):
    """Two snapshots whose outputs cannot be compared: different runs, or a failed driver."""


@dataclass(frozen=True)
class FileVerdict:
    """One file of one driver: ``same``, or not with a short ``detail`` of what differs."""
    path: str
    same: bool
    detail: str = ""


@dataclass(frozen=True)
class DriverComparison:
    driver: str
    files: tuple[FileVerdict, ...]

    @property
    def same(self) -> bool:
        return all(verdict.same for verdict in self.files)


def select_drivers(names=None, slow=False, registry=DRIVERS) -> list[Driver]:
    """The drivers to run, in registry order, with any driver they read from pulled in first.

    With no ``names`` it is every fast driver, plus the slow ones when ``slow``. A slow driver
    named in ``names`` runs whatever ``slow`` says. An unknown name raises.
    """
    by_name = {driver.name: driver for driver in registry}
    if names:
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise ValueError(f"unknown driver(s) {', '.join(unknown)}; "
                             f"known: {', '.join(by_name)}")
        chosen = set(names)
    else:
        chosen = {driver.name for driver in registry if slow or not driver.slow}
    for driver in registry:
        if driver.name in chosen:
            chosen.update(source for _, source in driver.inputs)
    return [driver for driver in registry if driver.name in chosen]


def command_for(driver: Driver, snapshot: Path, scripts_dir: Path = EXAMPLES_DIR) -> list[str]:
    """The driver's full command line, with the interpreter first."""
    command = [sys.executable, str(scripts_dir / f"{driver.name}.py"), *driver.args]
    for flag, source in driver.inputs:
        command += [flag, str(snapshot / source / OUT_DIR_NAME)]
    if driver.output == DIR:
        command += ["--out", str(snapshot / driver.name / OUT_DIR_NAME)]
    elif driver.output == FILE:
        command += ["--out", str(snapshot / driver.name / OUT_DIR_NAME / f"{driver.name}.json")]
    return command


def take_snapshot(snapshot, drivers, scripts_dir: Path = EXAMPLES_DIR) -> dict:
    """Run ``drivers`` one at a time into the new folder ``snapshot`` and write its manifest.

    Stops at the first driver that exits non-zero, keeping its stdout and stderr and recording
    the exit code; the returned manifest says which. An existing ``snapshot`` raises.
    """
    snapshot = Path(snapshot).resolve()
    snapshot.mkdir(parents=True)
    manifest = {"commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
                "python": sys.version.split()[0],
                "packages": {name: metadata.version(name) for name in PACKAGES},
                "drivers": []}
    env = {**os.environ, **RUN_ENV}
    for driver in drivers:
        folder = snapshot / driver.name
        (folder / OUT_DIR_NAME).mkdir(parents=True)
        command = command_for(driver, snapshot, scripts_dir)
        done = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True)
        (folder / STDOUT_NAME).write_bytes(done.stdout)
        (folder / STDERR_NAME).write_bytes(done.stderr)
        manifest["drivers"].append({
            "name": driver.name,
            "command": [Path(command[1]).name] + [
                part.replace(str(snapshot), SNAPSHOT_TOKEN) for part in command[2:]],
            "exit_code": done.returncode})
        if done.returncode != 0:
            break
    (snapshot / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def failed_driver(manifest: dict) -> str | None:
    """The name of the driver that exited non-zero, if any."""
    return next((entry["name"] for entry in manifest["drivers"] if entry["exit_code"] != 0), None)


def _git(*arguments) -> str:
    done = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True)
    return done.stdout.strip()


def compare_snapshots(old, new) -> list[DriverComparison]:
    """Compare every driver's files in two snapshots, in the manifest's order.

    Raises ``SnapshotsNotComparable`` when the manifests do not allow a comparison (see the
    module docstring); a file that cannot be read raises its own error. Neither is a verdict.
    """
    old, new = Path(old).resolve(), Path(new).resolve()
    old_manifest, new_manifest = _manifest(old), _manifest(new)
    for label, manifest in (("OLD", old_manifest), ("NEW", new_manifest)):
        failed = failed_driver(manifest)
        if failed:
            raise SnapshotsNotComparable(f"{label} snapshot: driver {failed} exited non-zero.")
    _require_same(old_manifest, new_manifest)
    return [_compare_driver(entry["name"], old, new) for entry in old_manifest["drivers"]]


def _manifest(snapshot: Path) -> dict:
    path = snapshot / MANIFEST_NAME
    if not path.is_file():
        raise SnapshotsNotComparable(f"{snapshot} has no {MANIFEST_NAME}; it is not a snapshot.")
    return json.loads(path.read_text())


def _require_same(old: dict, new: dict) -> None:
    old_runs = [(entry["name"], entry["command"]) for entry in old["drivers"]]
    new_runs = [(entry["name"], entry["command"]) for entry in new["drivers"]]
    if old_runs != new_runs:
        raise SnapshotsNotComparable(
            "the snapshots ran different drivers or command lines: "
            f"OLD {old_runs}, NEW {new_runs}.")
    for key in ("python", "packages"):
        if old[key] != new[key]:
            raise SnapshotsNotComparable(
                f"the snapshots were taken with different {key}: OLD {old[key]}, NEW {new[key]}.")


def _compare_driver(name: str, old: Path, new: Path) -> DriverComparison:
    old_files, new_files = _files(old / name), _files(new / name)
    verdicts = []
    for relative in sorted(set(old_files) | set(new_files)):
        side = _only_in(relative, old_files, new_files)
        if side:
            verdicts.append(FileVerdict(relative, False, side))
        else:
            problems = _file_problems(old_files[relative], new_files[relative], old, new)
            verdicts.append(FileVerdict(relative, not problems, "; ".join(problems)))
    return DriverComparison(name, tuple(verdicts))


def _only_in(key, old, new) -> str | None:
    """``"only in OLD"`` or ``"only in NEW"`` when ``key`` is on one side only, else None."""
    if key not in new:
        return "only in OLD"
    if key not in old:
        return "only in NEW"
    return None


def _files(folder: Path) -> dict[str, Path]:
    """Every compared file under a driver's folder, by path relative to it. stderr is skipped."""
    return {path.relative_to(folder).as_posix(): path
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.relative_to(folder).as_posix() != STDERR_NAME}


def _file_problems(old_path: Path, new_path: Path, old_root: Path, new_root: Path) -> list[str]:
    suffix = old_path.suffix.lower()
    if suffix == ".png":
        return _png_problems(old_path, new_path)
    if suffix in (".vtp", ".vti"):
        return _vtk_problems(old_path, new_path)
    old_bytes = old_path.read_bytes().replace(str(old_root).encode(), SNAPSHOT_TOKEN.encode())
    new_bytes = new_path.read_bytes().replace(str(new_root).encode(), SNAPSHOT_TOKEN.encode())
    if suffix == ".json":
        return _json_problems(json.loads(old_bytes), json.loads(new_bytes))
    return _text_problems(old_bytes, new_bytes)


def _text_problems(old: bytes, new: bytes) -> list[str]:
    if old == new:
        return []
    old_lines, new_lines = old.split(b"\n"), new.split(b"\n")
    differing = [i + 1 for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    problems = []
    if differing:
        listed = ", ".join(str(n) for n in differing[:MAX_LISTED])
        more = f" and {len(differing) - MAX_LISTED} more" if len(differing) > MAX_LISTED else ""
        problems.append(f"line {listed}{more} differ")
    if len(old_lines) != len(new_lines):
        problems.append(f"{len(old_lines)} lines in OLD, {len(new_lines)} in NEW")
    return problems or ["bytes differ"]


def _json_problems(old, new, where: str = "") -> list[str]:
    """Where two parsed JSON values differ, as a path like ``a.b[2]``; NaN equals NaN."""
    here = where or "(root)"
    if isinstance(old, dict) and isinstance(new, dict):
        problems = []
        for key in sorted(set(old) | set(new)):
            side = _only_in(key, old, new)
            if side:
                problems.append(f"{where}.{key}: {side}".lstrip("."))
            else:
                problems += _json_problems(old[key], new[key], f"{where}.{key}".lstrip("."))
        return problems[:MAX_LISTED]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{here}: {len(old)} items in OLD, {len(new)} in NEW"]
        problems = []
        for i, (a, b) in enumerate(zip(old, new)):
            problems += _json_problems(a, b, f"{where}[{i}]")
        return problems[:MAX_LISTED]
    if type(old) is not type(new):
        return [f"{here}: {type(old).__name__} in OLD, {type(new).__name__} in NEW"]
    if isinstance(old, float) and old != old and new != new:
        return []
    return [] if old == new else [f"{here}: value differs"]


def _png_problems(old_path: Path, new_path: Path) -> list[str]:
    from PIL import Image
    with Image.open(old_path) as image:
        old = np.asarray(image)
    with Image.open(new_path) as image:
        new = np.asarray(image)
    if old.shape != new.shape or old.dtype != new.dtype:
        return [f"image {old.shape} {old.dtype} in OLD, {new.shape} {new.dtype} in NEW"]
    differ = old != new
    if differ.ndim == 3:
        differ = differ.any(axis=-1)
    count = int(np.count_nonzero(differ))
    return [f"{count} pixels differ"] if count else []


def _vtk_problems(old_path: Path, new_path: Path) -> list[str]:
    import pyvista as pv
    old, new = pv.read(old_path), pv.read(new_path)
    if type(old) is not type(new):
        return [f"{type(old).__name__} in OLD, {type(new).__name__} in NEW"]
    problems = []
    if isinstance(old, pv.ImageData):
        for attribute in ("dimensions", "spacing", "origin", "direction_matrix"):
            if not np.array_equal(getattr(old, attribute), getattr(new, attribute)):
                problems.append(f"grid {attribute} differs")
    else:
        if not _same_values(old.points, new.points):
            problems.append(f"points differ ({old.n_points} in OLD, {new.n_points} in NEW)")
        for attribute in ("verts", "lines", "faces", "strips", "cells", "celltypes"):
            if hasattr(old, attribute) and not _same_values(
                    getattr(old, attribute), getattr(new, attribute)):
                problems.append(f"{attribute} differ")
    for kind in ("point_data", "cell_data", "field_data"):
        problems += _array_problems(getattr(old, kind), getattr(new, kind), kind)
    return problems[:MAX_LISTED]


def _array_problems(old, new, kind: str) -> list[str]:
    label = kind.replace("_data", "")
    problems = []
    for name in sorted(set(old.keys()) | set(new.keys())):
        side = _only_in(name, old, new)
        if side:
            problems.append(f"{label} array '{name}' {side}")
        else:
            a, b = np.asarray(old[name]), np.asarray(new[name])
            if a.dtype != b.dtype or a.shape != b.shape:
                problems.append(f"{label} array '{name}' is {a.dtype} {a.shape} in OLD, "
                                f"{b.dtype} {b.shape} in NEW")
            elif not _same_values(a, b):
                problems.append(f"{label} array '{name}' values differ")
    return problems


def _same_values(old, new) -> bool:
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.dtype != new.dtype:
        return False
    return bool(np.array_equal(old, new, equal_nan=old.dtype.kind in "fc"))


def format_report(comparisons) -> str:
    """One line per driver, then each differing file with its detail, and a closing count."""
    lines = []
    for comparison in comparisons:
        differing = [verdict for verdict in comparison.files if not verdict.same]
        if differing:
            lines.append(f"{comparison.driver}: DIFFERENT, {len(differing)} of "
                         f"{len(comparison.files)} files differ")
            lines += [f"  {verdict.path}: {verdict.detail}" for verdict in differing]
        else:
            lines.append(f"{comparison.driver}: same ({len(comparison.files)} files)")
    differing = sum(not comparison.same for comparison in comparisons)
    lines.append(f"{len(comparisons)} drivers compared: "
                 + (f"{differing} differ." if differing else "all match."))
    return "\n".join(lines)
