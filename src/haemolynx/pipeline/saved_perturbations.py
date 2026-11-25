"""Saved perturbations: one perturbation kept on this machine, for later runs.

A perturbation is a name, a type and the settings it changes
(:mod:`haemolynx.haemodynamics.perturbations`), so one worth asking again of
another network is kept as that entry, one YAML file per perturbation in
:func:`saved_perturbation_dir`. Unlike a saved preset
(:mod:`haemolynx.pipeline.saved_presets`), which sets the panel's whole
settings, a saved perturbation is *added* to a run's list, so perturbations
saved from different runs can be asked of one network together.

It holds only what its type reads, and nothing of one image's -- the rule a
saved preset follows (:func:`~haemolynx.pipeline.saved_presets.held_by_saved_presets`):
a pericyte or block mask, an H5 dataset, vessels by ID. The run it is added to
supplies those. The file's ``perturbation`` is an entry as a config's
``perturbations`` list holds it, so it can be pasted into one as it is.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from haemolynx.haemodynamics.perturbations import (
    PERTURBATION_TYPES,
    PerturbationSpec,
    settings_for_perturbation_type,
)
from haemolynx.parsers.schema import ConfigError, Schema
from haemolynx.pipeline.saved_presets import (
    held_by_saved_presets,
    preset_file_stem,
    user_settings_dir,
)

logger = logging.getLogger(__name__)

#: Set to a folder to keep saved perturbations there instead of the user's own
#: settings folder (the tests point it at an empty one).
PERTURBATION_DIR_ENV = "HAEMOLYNX_PERTURBATION_DIR"

#: A saved perturbation's file suffix.
PERTURBATION_SUFFIX = ".yaml"

_HEADER = (
    "# A HaemoLynx saved perturbation. The panel's Perturbations tab adds it to\n"
    "# a run's list; the entry under perturbation can also go in a config's\n"
    "# perturbations list as it is. The files and vessels of one image are not kept.\n"
)


def saved_perturbation_dir() -> Path:
    """Where saved perturbations are kept: :data:`PERTURBATION_DIR_ENV` if
    set, else a ``perturbations`` folder beside the saved presets'."""
    override = os.environ.get(PERTURBATION_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return user_settings_dir() / "perturbations"


@dataclass(frozen=True)
class SavedPerturbation:
    """One saved perturbation: its file and the entry it holds."""

    path: Path
    entry: Mapping[str, Any]
    saved: Optional[str] = None
    #: The overrides left out when it was saved, as one image's or not read
    #: by its type; empty for one read back from its file.
    left_out: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return str(self.entry.get("name") or self.path.stem)

    @property
    def type(self) -> str:
        return str(self.entry.get("type") or "none")

    @property
    def key(self) -> str:
        """What the panel's menu keys this perturbation by: its file's stem."""
        return self.path.stem


def saved_perturbation_path(name: str, directory: Path | str | None = None) -> Path:
    """The file perturbation *name* is (or would be) saved in; ValueError
    for a name with nothing in it."""
    folder = Path(directory) if directory is not None else saved_perturbation_dir()
    return folder / f"{preset_file_stem(name)}{PERTURBATION_SUFFIX}"


def held_entry(entry: Any, schema: Schema) -> tuple[dict[str, Any], tuple[str, ...]]:
    """*entry* as a saved perturbation holds it, and the overrides left out:
    those its type does not read, and those of one image's."""
    spec = PerturbationSpec.from_entry(entry)
    reads = set(settings_for_perturbation_type(spec.type))
    kept: dict[str, Any] = {}
    left_out: list[str] = []
    for key, value in spec.overrides.items():
        if key in reads and key in schema and held_by_saved_presets(schema[key]):
            kept[key] = value
        else:
            left_out.append(key)
    held = PerturbationSpec(name=spec.name, type=spec.type, overrides=kept).to_entry()
    return held, tuple(sorted(left_out))


def save_perturbation(
    entry: Any, schema: Schema, *, directory: Path | str | None = None
) -> SavedPerturbation:
    """Write *entry*, one of the `perturbations` list, under its own name,
    over any perturbation saved under the same file name, and return it.

    ValueError for an entry with no type yet, or with a value the schema
    rejects: kept, it would be refused by every run it was added to.
    """
    import yaml

    spec = PerturbationSpec.from_entry(entry)
    if spec.type == "none" or spec.type not in PERTURBATION_TYPES:
        raise ValueError(f"The perturbation {spec.name!r} has no type yet: choose one before saving it")
    problems = spec.schema_problems(schema)
    if problems:
        raise ValueError("; ".join(problems))
    path = saved_perturbation_path(spec.name, directory)
    held, left_out = held_entry(entry, schema)
    saved = datetime.now().isoformat(timespec="seconds")
    body = yaml.safe_dump(
        {"saved": saved, "perturbation": held},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10**9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside it and moved over the name, so a saved perturbation is
    # never left half written.
    partial = path.with_name(path.name + ".part")
    partial.write_text(_HEADER + body, encoding="utf-8")
    os.replace(partial, path)
    return SavedPerturbation(path=path, entry=held, saved=saved, left_out=left_out)


def read_saved_perturbation(path: Path | str) -> SavedPerturbation:
    """The perturbation saved in *path*; ConfigError if it is not one."""
    import yaml

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Could not read the saved perturbation {path}: {error}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("perturbation"), Mapping):
        raise ConfigError(f"{path} is not a saved perturbation: it has no perturbation entry")
    return SavedPerturbation(
        path=path,
        entry=dict(raw["perturbation"]),
        saved=str(raw["saved"]) if raw.get("saved") else None,
    )


def saved_perturbations(directory: Path | str | None = None) -> list[SavedPerturbation]:
    """Every perturbation saved in *directory* (default
    :func:`saved_perturbation_dir`), by name; a file that is not one is
    logged and passed over."""
    folder = Path(directory) if directory is not None else saved_perturbation_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob(f"*{PERTURBATION_SUFFIX}")):
        try:
            found.append(read_saved_perturbation(path))
        except ConfigError as error:
            logger.warning("%s", error)
    return sorted(found, key=lambda saved: (saved.name.casefold(), saved.path.name))


def saved_perturbation_for_key(
    key: str, directory: Path | str | None = None
) -> SavedPerturbation:
    """The perturbation the panel's menu entry *key* stands for, read from
    its file now; ConfigError if it is gone or unreadable."""
    folder = Path(directory) if directory is not None else saved_perturbation_dir()
    path = folder / f"{key}{PERTURBATION_SUFFIX}"
    if not path.is_file():
        raise ConfigError(f"The saved perturbation {path} is no longer there")
    return read_saved_perturbation(path)


def saved_perturbation_entry(
    saved: SavedPerturbation, schema: Schema
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The entry *saved* adds to a run's list, and the overrides passed over.

    Its overrides as this schema spells them (one saved under a setting since
    renamed loads through :meth:`Schema.upgrade`), each coerced to its kind,
    and only those its type reads and a saved perturbation may hold: a name
    this schema lacks, a value it rejects, or one written into the file by
    hand that a saved perturbation may not hold is passed over and returned.
    ConfigError for a type this version does not know.
    """
    spec = PerturbationSpec.from_entry(saved.entry)
    if spec.type not in PERTURBATION_TYPES:
        raise ConfigError(
            f"The saved perturbation {saved.name} ({saved.path}) is of type "
            f"{spec.type!r}, which this version does not know; one of "
            f"{', '.join(PERTURBATION_TYPES)}"
        )
    upgraded = PerturbationSpec(
        name=spec.name, type=spec.type, overrides=schema.upgrade(spec.overrides)
    )
    kept = {
        key: value
        for key, value in upgraded.applied_overrides(schema).items()
        if held_by_saved_presets(schema[key])
    }
    passed_over = tuple(sorted(key for key in upgraded.overrides if key not in kept))
    entry = PerturbationSpec(name=spec.name, type=spec.type, overrides=kept).to_entry()
    return entry, passed_over


def delete_saved_perturbation(saved: SavedPerturbation) -> None:
    """Remove *saved*'s file; nothing if it is already gone."""
    try:
        saved.path.unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "PERTURBATION_DIR_ENV",
    "PERTURBATION_SUFFIX",
    "SavedPerturbation",
    "delete_saved_perturbation",
    "held_entry",
    "read_saved_perturbation",
    "save_perturbation",
    "saved_perturbation_dir",
    "saved_perturbation_entry",
    "saved_perturbation_for_key",
    "saved_perturbation_path",
    "saved_perturbations",
]
