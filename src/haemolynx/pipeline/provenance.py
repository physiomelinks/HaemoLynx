"""What made a run's outputs: written beside them as the run's manifest.

A result is only as good as the ability to say what produced it. Defaults move
(``open_end_model`` on 2026-10-12) and laws are corrected (the Pries in vitro
viscosity on 2026-10-13, about 3x on capillary resistance), and an output
folder on its own cannot say whether it came before or after. So every run
writes ``{stem}_run_manifest.json`` into its output folder, with
``{stem}_run_settings.yaml`` beside it::

    with record_run(settings, schema) as run:
        with run.stage("segment"):
            inputs = segment(settings)
        ...

:func:`~haemolynx.pipeline.run_pipeline_stages` does this itself, so the panel
and any script calling it get a manifest without asking. The manifest holds:

- the full settings, after :meth:`Schema.upgrade`, as they stood when the run
  started (the stages write found boundary nodes back into them);
- the HaemoLynx version, and the git commit and whether the tree was dirty
  when the package runs from a checkout (``null`` for an installed wheel);
- the Python version, the platform and the key dependencies' versions;
- each input file's path, size, modification time and checksum -- the whole
  file's sha256 up to :data:`FULL_HASH_MAX_BYTES`, above that a sha256 of
  :data:`SAMPLE_BLOCKS` evenly spaced blocks, which reads a fixed amount of a
  multi-GB stack (``checksum_method`` says which);
- each stage's start, end, wall time and peak memory (see
  :data:`MEMORY_METHOD`).

A run that fails still writes one, with ``status: "failed"``, the error and the
stages it finished. A run that fails before its first stage finished has
written nothing to describe, and writes none. Writing it never fails the run:
anything that goes wrong here is logged as a warning and the run goes on.

Peak memory comes from the standard library, not psutil (which HaemoLynx does
not depend on): ``GetProcessMemoryInfo`` through ctypes on Windows,
``getrusage`` (and ``/proc/self/statm`` for the current size) elsewhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Any, Iterator, Mapping, Optional

from haemolynx.parsers.schema import Schema, is_active, section_key
from haemolynx.pipeline.naming import run_input_path, run_stem
from haemolynx.pipeline.progress import (
    STAGE_FAILED,
    STAGE_FINISHED,
    STAGE_STARTED,
    STAGES,
    ProgressCallback,
    ProgressEvent,
    RunProgress,
    format_seconds,
)
from haemolynx.pipeline.settings import fill_derived_settings

logger = logging.getLogger(__name__)

#: The manifest's own layout version. Raise it when a field changes meaning.
MANIFEST_VERSION = 1

#: ``{stem}`` + this, in the run's output folder.
MANIFEST_SUFFIX = "_run_manifest.json"
#: ``{stem}`` + this, beside the manifest: the settings as a config file.
SETTINGS_SUFFIX = "_run_settings.yaml"

#: Files up to this size are checksummed whole.
FULL_HASH_MAX_BYTES = 64 * 2**20
#: Larger files: this many blocks, evenly spaced from the first byte to the last.
SAMPLE_BLOCKS = 32
#: Of this many bytes each -- 32 MiB read, whatever the file's size.
SAMPLE_BLOCK_BYTES = 2**20

#: The distributions whose versions change what a run computes.
KEY_DEPENDENCIES = (
    "numpy",
    "scipy",
    "networkx",
    "scikit-image",
    "tifffile",
    "skan",
    "h5py",
    "pandas",
    "pyvista",
)

#: How often the current resident size is read while a stage runs, seconds.
MEMORY_SAMPLE_SECONDS = 0.2

#: How long any one git command may take before it is given up on.
GIT_TIMEOUT_SECONDS = 10.0

#: At most this many dirty paths are listed; ``dirty_path_count`` has them all.
MAX_DIRTY_PATHS = 100

MEMORY_METHOD = (
    "Resident set size (the working set on Windows) of this Python process, read with "
    "the standard library: GetProcessMemoryInfo through ctypes on Windows, getrusage "
    "and /proc/self/statm on Linux, getrusage on macOS. Pages of memory-mapped files "
    "count while they are resident. A stage that raised the process's peak has that "
    "peak exactly (peak_rss_method 'process_peak'). Otherwise the process peak cannot "
    "be reset between stages, and the stage's peak is the largest resident size read "
    f"every {MEMORY_SAMPLE_SECONDS} s while it ran ('sampled', a lower bound) or, where "
    "the current size cannot be read (macOS), the process's peak so far "
    "('process_peak_so_far', an upper bound). Worker processes are not in this "
    "process's figure: worker_peak_rss_bytes is the largest finished worker's peak, "
    "from getrusage(RUSAGE_CHILDREN), and is null on Windows, which has no equivalent."
)

#: The run's stages in order, by call name.
_STAGE_CALLS = tuple(stage.call for stage in STAGES if stage.call)
_STAGE_TITLES = {stage.call: stage.title for stage in STAGES if stage.call}


# --- memory --------------------------------------------------------------------


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    @lru_cache(maxsize=None)
    def _kernel32():
        # Its own handle, not the shared `ctypes.windll.kernel32`, so the
        # signatures set here change nobody else's calls.
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        return kernel32

    def _windows_counters() -> Optional[_ProcessMemoryCounters]:
        kernel32 = _kernel32()
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        if not kernel32.K32GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return counters


def current_rss_bytes() -> Optional[int]:
    """This process's resident size now, or None where it cannot be read."""
    try:
        if sys.platform == "win32":
            counters = _windows_counters()
            return None if counters is None else int(counters.WorkingSetSize)
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", "rb") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001 - a measurement must not end a run
        return None
    return None


def _rusage_peak_bytes(who: str) -> Optional[int]:
    import resource

    peak = resource.getrusage(getattr(resource, who)).ru_maxrss
    # Linux reports kibibytes, macOS bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def peak_rss_bytes() -> Optional[int]:
    """The most this process has ever had resident, or None where unknown."""
    try:
        if sys.platform == "win32":
            counters = _windows_counters()
            return None if counters is None else int(counters.PeakWorkingSetSize)
        return _rusage_peak_bytes("RUSAGE_SELF")
    except Exception:  # noqa: BLE001
        return None


def worker_peak_rss_bytes() -> Optional[int]:
    """The largest finished child process's peak; None on Windows."""
    if sys.platform == "win32":
        return None
    try:
        return _rusage_peak_bytes("RUSAGE_CHILDREN")
    except Exception:  # noqa: BLE001
        return None


def total_ram_bytes() -> Optional[int]:
    """The machine's physical memory, or None where unknown."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        return None


class _MemorySampler:
    """Reads the current resident size every few tenths of a second.

    The process peak only ever rises, so a stage that stays below an earlier
    stage's peak would otherwise have no figure of its own. One daemon thread
    per run; :meth:`restart` at each stage's start.
    """

    def __init__(self, interval: float = MEMORY_SAMPLE_SECONDS) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._peak: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if current_rss_bytes() is None:
            return
        self._thread = threading.Thread(
            target=self._sample, name="haemolynx-memory-sampler", daemon=True
        )
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self.note(current_rss_bytes())

    def note(self, value: Optional[int]) -> None:
        if value is None:
            return
        with self._lock:
            if self._peak is None or value > self._peak:
                self._peak = value

    def restart(self) -> None:
        with self._lock:
            self._peak = current_rss_bytes()

    def peak(self) -> Optional[int]:
        with self._lock:
            return self._peak

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self._interval)
            self._thread = None


# --- the code and the machine --------------------------------------------------


#: The package's own directory: a checkout's ``src/haemolynx``, or an installed
#: wheel's ``site-packages/haemolynx``.
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _git(arguments: list[str], cwd: Path) -> str:
    """One git command's output, unstripped (a status line starts with a
    space). Raises if git is missing, slow or unhappy."""
    completed = subprocess.run(
        ["git", "--no-optional-locks", *arguments],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=True,
        # No console window flashing up behind the napari panel.
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return completed.stdout


def read_git_state(package_dir: Path = PACKAGE_DIR) -> dict[str, Any]:
    """The commit the package's source is at, and whether its tree is dirty.

    Every field is None for an installed wheel -- no repository above it, or
    (a wheel installed into a venv that happens to sit inside a checkout) a
    repository that does not track the package's own files -- and wherever
    git is missing, slow or fails. Never raises.
    """
    state: dict[str, Any] = {
        "commit": None,
        "dirty": None,
        "branch": None,
        "describe": None,
        "dirty_paths": None,
        "dirty_path_count": None,
        "read_at": _now(),
    }
    package_dir = Path(package_dir)
    if not any((folder / ".git").exists() for folder in (package_dir, *package_dir.parents)):
        return state
    try:
        _git(["ls-files", "--error-unmatch", "--", "__init__.py"], package_dir)
        commit = _git(["rev-parse", "HEAD"], package_dir).strip()
        status = _git(["status", "--porcelain", "--untracked-files=no"], package_dir)
    except Exception as error:  # noqa: BLE001 - provenance must not end a run
        logger.debug(f"No git commit recorded: {error}")
        return state
    # Porcelain lines are "XY path": two status letters (either may be a space).
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    state.update(
        commit=commit,
        dirty=bool(dirty),
        dirty_paths=dirty[:MAX_DIRTY_PATHS],
        dirty_path_count=len(dirty),
    )
    for field, arguments in (
        ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
        ("describe", ["describe", "--tags", "--always"]),
    ):
        try:
            state[field] = _git(arguments, package_dir).strip()
        except Exception:  # noqa: BLE001
            pass
    return state


@lru_cache(maxsize=None)
def git_state() -> dict[str, Any]:
    """:func:`read_git_state` for this package, read once per process.

    Once, because the code a long-lived process runs is what it imported: a
    commit made in the checkout after that is not what ran, so re-reading the
    checkout at every run would misreport the panel's later runs.
    """
    return read_git_state()


def dependency_versions(names: tuple[str, ...] = KEY_DEPENDENCIES) -> dict[str, Optional[str]]:
    """Each distribution's installed version; None for one not installed."""
    from importlib import metadata

    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - PackageNotFoundError, or broken metadata
            versions[name] = None
    return versions


def environment() -> dict[str, Any]:
    """Python, the platform, and the dependencies a run computed with."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "total_ram_bytes": total_ram_bytes(),
        },
        "dependencies": dependency_versions(),
    }


# --- inputs --------------------------------------------------------------------


def file_fingerprint(
    path: Path | str,
    *,
    full_hash_max_bytes: int = FULL_HASH_MAX_BYTES,
    blocks: int = SAMPLE_BLOCKS,
    block_bytes: int = SAMPLE_BLOCK_BYTES,
) -> dict[str, Any]:
    """*path*'s size, modification time and checksum, and how it was taken.

    Up to *full_hash_max_bytes* the whole file is hashed (``sha256:...``).
    Above, the hash is of the file's size and *blocks* blocks of
    *block_bytes*, the first at the start, the last at the end and the rest
    evenly spaced between (``sha256-sampled:...``): a change confined to the
    bytes between blocks is not seen, which for an image stack means a change
    to a few voxels, not a different stack. The prefixes differ so the two
    kinds are never compared as one.
    """
    path = Path(path)
    stat = path.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    started = time.perf_counter()
    with path.open("rb") as handle:
        if size <= full_hash_max_bytes:
            for chunk in iter(lambda: handle.read(2**20), b""):
                digest.update(chunk)
            checksum = f"sha256:{digest.hexdigest()}"
            method = "sha256 of the whole file"
            read = size
        else:
            digest.update(size.to_bytes(8, "little"))
            last = size - block_bytes
            read = 0
            for index in range(blocks):
                handle.seek(index * last // max(blocks - 1, 1))
                block = handle.read(block_bytes)
                digest.update(block)
                read += len(block)
            checksum = f"sha256-sampled:{digest.hexdigest()}"
            method = (
                f"sha256 of the file size (8 bytes, little-endian) and {blocks} blocks of "
                f"{block_bytes} bytes at offsets i*(size-{block_bytes})//{blocks - 1}, "
                f"i = 0..{blocks - 1}"
            )
    return {
        "path": PurePath(path.resolve()).as_posix(),
        "size_bytes": size,
        "modified": _timestamp(stat.st_mtime),
        "checksum": checksum,
        "checksum_method": method,
        "checksum_bytes_read": read,
        "checksum_seconds": round(time.perf_counter() - started, 4),
    }


def _existing_file(value: Any) -> Optional[Path]:
    """The file a path setting names, or its ``.zip`` sibling, if either is there."""
    if value is None or str(value) == "":
        return None
    path = Path(value)
    if path.is_file():
        return path
    zipped = path.with_suffix(path.suffix + ".zip")
    return zipped if zipped.is_file() else None


def input_files(settings: Mapping[str, Any], schema: Optional[Schema]) -> list[dict[str, Any]]:
    """Every input file the run read, fingerprinted.

    The image analysed (``input_path``, which is ilastik's output once
    ``segment`` has run it), then every path setting in effect that names an
    existing file: output folders and prefixes name no file, so they are left
    out. A file two settings name is fingerprinted once, under the first.
    """
    named: list[tuple[str, Any]] = [("input_path", run_input_path(settings))]
    if schema is not None:
        for setting in schema:
            if setting.kind != "path" or setting.name == "input_path":
                continue
            if setting.name in settings and is_active(setting, settings):
                named.append((setting.name, settings[setting.name]))
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for name, value in named:
        path = _existing_file(value)
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            records.append({"setting": name, **file_fingerprint(path)})
        except OSError as error:
            records.append(
                {"setting": name, "path": PurePath(resolved).as_posix(), "error": str(error)}
            )
    return records


# --- settings ------------------------------------------------------------------


def plain(value: Any, *, finite_only: bool = True) -> Any:
    """*value* as JSON can hold it: paths as posix strings, containers as lists
    and dicts, numpy scalars and small arrays as numbers. With *finite_only*,
    an infinite or NaN float becomes its name, which strict JSON has no
    number for; YAML has (``.inf``), so the settings file keeps the float."""
    # Subclasses (numpy's float64 is one) become the plain type: YAML's safe
    # dumper refuses anything else.
    if value is None or isinstance(value, bool):
        return value if value is None else bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        value = float(value)
        return value if math.isfinite(value) or not finite_only else repr(value)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            (key if isinstance(key, (str, int, float, bool)) or key is None else str(key)):
            plain(item, finite_only=finite_only)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [plain(item, finite_only=finite_only) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((plain(item, finite_only=finite_only) for item in value), key=repr)
    item = getattr(value, "item", None)
    shape = getattr(value, "shape", None)
    if shape == () and callable(item):
        return plain(item(), finite_only=finite_only)
    if shape is not None and getattr(value, "size", 0) <= 10_000 and hasattr(value, "tolist"):
        return plain(value.tolist(), finite_only=finite_only)
    if shape is not None:
        return f"<array shape={tuple(shape)} dtype={getattr(value, 'dtype', '?')}>"
    return repr(value)


def _upgraded(settings: Mapping[str, Any], schema: Optional[Schema]) -> dict[str, Any]:
    return dict(settings) if schema is None else schema.upgrade(settings)


#: Settings :func:`fill_derived_settings` computes from others when left null.
DERIVED_SETTINGS = ("diameter_by_branch_order", "constriction_by_branch_order")


def as_configured(values: Mapping[str, Any]) -> dict[str, Any]:
    """*values* with each derived setting back to null where the other settings
    derive exactly it, as the config file it came from had it -- a derived
    table written out in full warns on loading when nothing reads it."""
    configured = dict(values)
    for name in DERIVED_SETTINGS:
        if configured.get(name) is None:
            continue
        trial = {**configured, name: None}
        try:
            fill_derived_settings(trial)
        except Exception:  # noqa: BLE001 - keep the value as it is
            continue
        if trial.get(name) == configured[name]:
            configured[name] = None
    return configured


def write_settings_yaml(path: Path, values: Mapping[str, Any], schema: Schema) -> Path:
    """*values* as a config file ``--config`` loads, one heading per section.

    Every setting, not only those away from their defaults: a default may move
    between versions, and this file should still say what ran.
    """
    import yaml

    values = as_configured(values)
    by_section: dict[str, dict[str, Any]] = {}
    for section, settings in schema.sections().items():
        entries = {
            setting.name: plain(values[setting.name], finite_only=False)
            for setting in settings
            if setting.name in values
        }
        if entries:
            by_section[section_key(section)] = entries
    header = (
        "# The settings of the HaemoLynx run described by the run manifest beside\n"
        "# this file, as they stood when it started. Load it with --config.\n"
    )
    path.write_text(
        header + yaml.safe_dump(by_section, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# --- the record ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _timestamp(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _format_bytes(count: Optional[int]) -> str:
    if count is None:
        return "unknown"
    return f"{count / 2**30:.2f} GiB" if count >= 2**30 else f"{count / 2**20:.0f} MiB"


def manifest_location(settings: Mapping[str, Any]) -> Optional[tuple[Path, str]]:
    """The output folder and file stem a run's manifest goes under, or None.

    The folder is ``vtk_output_prefix``'s, the stem the analysed image's
    (``input_path``, ilastik's output when ilastik ran): the folder and stem
    every other output of the run is named by.
    """
    prefix = settings.get("vtk_output_prefix")
    stem = run_stem(settings)
    if not prefix or not stem:
        return None
    return Path(prefix).parent, stem


class RunRecorder:
    """Collects one run's provenance and writes it as the run's manifest.

    Built when the run starts, so the settings are recorded before any stage
    writes into them; fed every progress event through :meth:`watch`; written
    once by :meth:`write` when the run ends, however it ends. Nothing here may
    stop a run: each step that could fail records the failure instead.
    """

    def __init__(
        self,
        settings: dict,
        schema: Optional[Schema],
        *,
        start_from: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._schema = schema
        self.start_from = start_from
        self.stop_after = stop_after
        self.started = _now()
        self._clock = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self._open: Optional[dict[str, Any]] = None
        self._peak_at_open: Optional[int] = None
        self.error: Optional[BaseException] = None
        self._error_stage: Optional[str] = None
        self.settings_error: Optional[str] = None
        self.recorded_settings: Optional[dict[str, Any]] = None
        self._config_values: Optional[dict[str, Any]] = None
        try:
            upgraded = _upgraded(settings, schema)
            self.recorded_settings = plain(upgraded)
            if schema is not None:
                self._config_values = copy.deepcopy(
                    {name: value for name, value in upgraded.items() if name in schema}
                )
        except Exception as error:  # noqa: BLE001
            self.settings_error = f"{type(error).__name__}: {error}"
        self._sampler = _MemorySampler()
        try:
            self._sampler.start()
        except Exception:  # noqa: BLE001 - the stages then go unsampled
            logger.debug("could not start the memory sampler", exc_info=True)

    # -- what the run reports ------------------------------------------------

    def watch(self, callback: Optional[ProgressCallback]) -> ProgressCallback:
        """*callback*, with this recorder hearing every event first.

        First, so a watcher that raises to stop the run -- the panel's cancel
        -- cannot keep a stage's end out of the record.
        """

        def watched(event: ProgressEvent) -> None:
            self.observe(event)
            if callback is not None:
                callback(event)

        return watched

    def observe(self, event: ProgressEvent) -> None:
        try:
            if event.kind == STAGE_STARTED:
                self._begin(event.stage)
            elif event.kind in (STAGE_FINISHED, STAGE_FAILED):
                status = "finished" if event.kind == STAGE_FINISHED else "failed"
                self._end(status, event.seconds)
                if event.kind == STAGE_FAILED and self._error_stage is None:
                    self._error_stage = event.stage
        except Exception:  # noqa: BLE001
            logger.debug("could not record a stage boundary", exc_info=True)

    def _begin(self, stage: str) -> None:
        if self._open is not None:
            self._end("interrupted", None)
        self._peak_at_open = peak_rss_bytes()
        self._sampler.restart()
        self._open = {
            "stage": stage,
            "title": _STAGE_TITLES.get(stage, stage),
            "status": "running",
            "resumed": self._resumed(stage),
            "started": _now(),
            "finished": None,
            "wall_seconds": None,
            "peak_rss_bytes": None,
            "peak_rss_method": None,
            "process_peak_rss_bytes": None,
            "worker_peak_rss_bytes": None,
            "_clock": time.perf_counter(),
        }

    def _resumed(self, stage: str) -> bool:
        """Whether *stage* comes before the one the run resumed at: its time
        is that of handing on the saved result, not of computing it."""
        if self.start_from not in _STAGE_CALLS or stage not in _STAGE_CALLS:
            return False
        return _STAGE_CALLS.index(stage) < _STAGE_CALLS.index(self.start_from)

    def _end(self, status: str, seconds: Optional[float]) -> None:
        record = self._open
        if record is None:
            return
        self._open = None
        clock = record.pop("_clock")
        process_peak = peak_rss_bytes()
        self._sampler.note(current_rss_bytes())
        sampled = self._sampler.peak()
        if (
            process_peak is not None
            and self._peak_at_open is not None
            and process_peak > self._peak_at_open
        ):
            peak, method = process_peak, "process_peak"
        elif sampled is not None:
            peak, method = sampled, "sampled"
        else:
            peak, method = process_peak, (None if process_peak is None else "process_peak_so_far")
        record.update(
            status=status,
            finished=_now(),
            wall_seconds=round(seconds if seconds is not None else time.perf_counter() - clock, 4),
            peak_rss_bytes=peak,
            peak_rss_method=method,
            process_peak_rss_bytes=process_peak,
            worker_peak_rss_bytes=worker_peak_rss_bytes(),
        )
        self.stages.append(record)

    def fail(self, error: BaseException) -> None:
        """The run raised *error*: close the stage it was in as interrupted."""
        self.error = error
        if self._open is not None:
            if self._error_stage is None:
                self._error_stage = self._open["stage"]
            self._end("interrupted", None)

    # -- the manifest --------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Everything recorded, as the JSON the manifest holds."""
        import haemolynx

        finished = [record for record in self.stages if record["status"] == "finished"]
        error = None
        if self.error is not None:
            error = {
                "type": type(self.error).__name__,
                "message": str(self.error),
                "stage": self._error_stage,
            }
        try:
            inputs = input_files(self._settings, self._schema)
        except Exception as failure:  # noqa: BLE001
            inputs = [{"error": f"{type(failure).__name__}: {failure}"}]
        recorded = self.recorded_settings if isinstance(self.recorded_settings, Mapping) else {}
        synthetic = None
        if run_stem(recorded) is not None and recorded.get("network_source") == "synthetic":
            template = recorded.get("synthetic_template_path")
            synthetic = {
                "template": None if template is None else PurePath(str(template)).as_posix(),
                "source": recorded.get("synthetic_template_source"),
                "seed": recorded.get("synthetic_seed"),
                "scale": recorded.get("synthetic_scale"),
                "interior_dead_ends": recorded.get("synthetic_interior_dead_ends"),
                "bed_model": recorded.get("synthetic_bed_model"),
            }
        return {
            "manifest_version": MANIFEST_VERSION,
            "network_source": recorded.get("network_source", "segmented_image") if recorded else None,
            "synthetic": synthetic,
            "haemolynx_version": getattr(haemolynx, "__version__", None),
            "status": "failed" if self.error is not None else "finished",
            "error": error,
            "started": self.started,
            "finished": _now(),
            "wall_seconds": round(time.perf_counter() - self._clock, 4),
            "start_from": self.start_from,
            "stop_after": self.stop_after,
            "stages_finished": [record["stage"] for record in finished],
            "argv": list(sys.argv),
            "package_dir": PACKAGE_DIR.as_posix(),
            "git": git_state(),
            **environment(),
            "memory_method": MEMORY_METHOD,
            "peak_rss_bytes": peak_rss_bytes(),
            "stages": self.stages,
            "inputs": inputs,
            "settings_file": None,
            "settings_error": self.settings_error,
            "settings": self.recorded_settings,
        }

    def write(self) -> Optional[Path]:
        """Write the manifest and the settings file; the manifest's path.

        None, with nothing written, when the run finished no stage (it left
        no output to describe) or has no output folder. Never raises.
        """
        self._sampler.stop()
        try:
            return self._write()
        except Exception as error:  # noqa: BLE001 - provenance must not end a run
            logger.warning(f"Could not write the run manifest: {type(error).__name__}: {error}")
            return None

    def _write(self) -> Optional[Path]:
        if not any(record["status"] == "finished" for record in self.stages):
            logger.debug("No stage finished, so no run manifest is written.")
            return None
        location = manifest_location(self._settings)
        if location is None:
            logger.debug("The run has no output folder, so no run manifest is written.")
            return None
        folder, stem = location
        folder.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        if self._schema is not None and self._config_values is not None:
            settings_path = folder / f"{stem}{SETTINGS_SUFFIX}"
            try:
                write_settings_yaml(settings_path, self._config_values, self._schema)
                manifest["settings_file"] = settings_path.name
            except Exception as error:  # noqa: BLE001
                manifest["settings_error"] = f"{type(error).__name__}: {error}"
        path = folder / f"{stem}{MANIFEST_SUFFIX}"
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(partial, path)
        times = ", ".join(
            f"{record['stage']} {format_seconds(record['wall_seconds'])}"
            for record in self.stages
        )
        logger.info(
            f"Run manifest written to {path} ({times}; "
            f"peak memory {_format_bytes(manifest['peak_rss_bytes'])})."
        )
        return path


@contextmanager
def record_run(
    settings: dict,
    schema: Optional[Schema],
    *,
    progress: Optional[ProgressCallback] = None,
    start_from: Optional[str] = None,
    stop_after: Optional[str] = None,
) -> Iterator[RunProgress]:
    """A run's :class:`RunProgress`, its manifest written when the block ends.

    *progress* hears every event as it would have without the recording. The
    manifest is written whether the block finishes or raises, and the error,
    if any, propagates unchanged.
    """
    recorder = RunRecorder(settings, schema, start_from=start_from, stop_after=stop_after)
    try:
        yield RunProgress(recorder.watch(progress))
    except BaseException as error:
        recorder.fail(error)
        raise
    finally:
        recorder.write()
