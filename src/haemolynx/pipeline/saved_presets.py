"""Saved presets: the panel's settings kept on this machine for later runs.

A saved preset is the panel's settings as they were when it was saved, less
what belongs to one image (:func:`held_by_saved_presets`): the files and
folders a run reads and writes, the channels, labels, voxel size and axis
order that describe those files, the crop on import, the nodes, vessels and
positions of one image's network, and the switches that reuse one image's
cached outputs.

Unlike the shipped presets (:mod:`haemolynx.pipeline.presets`), which name a
few switches and leave every other setting alone, a saved preset is a whole
set: loading it sets every setting it may hold, those it names to its values
and the rest to their defaults, so the settings come back as they were saved.
What it may not hold is left as it is. Only the settings away from their
defaults are written, so the file reads as what the preset changes.

One YAML file per preset, in :func:`saved_preset_dir` -- the user's own
settings folder, or :data:`PRESET_DIR_ENV` -- so a preset outlives the panel
and is shared by every checkout and environment on the machine.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, Mapping, Optional

from haemolynx.parsers.schema import ConfigError, Schema, Setting

logger = logging.getLogger(__name__)

#: Set to a folder to keep saved presets there instead of the user's own
#: settings folder (the tests point it at an empty one).
PRESET_DIR_ENV = "HAEMOLYNX_PRESET_DIR"

#: A saved preset's file suffix.
PRESET_SUFFIX = ".yaml"

#: What the panel's Preset menu keys a saved preset by: this, then its file's
#: stem. The shipped presets are keyed by identifiers, which hold no colon.
SAVED_PRESET_KEY_PREFIX = "saved:"

#: Endings of the names of settings that describe one image's files or name
#: places in its network: a channel of the raw image, an H5 file's dataset, a
#: classifier's label, and the nodes, positions and boxes of its boundaries.
_ONE_IMAGE_ENDINGS = (
    "_channel",
    "_channels",
    "_h5_dataset_name",
    "_vessel_label",
    "_node_ids",
    "_node_coordinates",
    "_node_volumes",
    "_nodes",
)

#: The rest of one image's settings: how its file is laid out, its vessels by
#: ID, the ilastik program it was segmented with, and the switches that reuse
#: its cached skeleton, graph and diameters (the panel's ``SKIP_FOR_RESUME``).
#: The crop on import is one image's too, every setting of it
#: (:data:`_ONE_IMAGE_PREFIXES`).
_ONE_IMAGE_NAMES = frozenset(
    {
        "voxel_size_override_xyz",
        "image_axis_order",
        "custom_edges",
        "capillary_block_vessel_ids",
        "ilastik_executable",
        "do_skeletonize",
        "do_graph_building",
        "do_fwhm_measurement",
    }
)

#: Beginnings of the names of one image's settings: the crop on import, its
#: switch, method, size, place and corners alike (Harvey, 2026-11-03). Kept
#: with its switch but not its corners, a preset turned on a crop for the
#: next image with that image's corners, or none.
_ONE_IMAGE_PREFIXES = ("crop_",)

#: Names Windows refuses as a file's stem, whatever its suffix.
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)
#: Longest stem a preset's name becomes.
_MAX_STEM = 100

_HEADER = (
    "# A HaemoLynx saved preset. Loading it sets every setting listed under\n"
    "# settings, and puts every other one a preset may hold back to its default;\n"
    "# the files, folders and details of one image are left as they are.\n"
)


def held_by_saved_presets(setting: Setting) -> bool:
    """Whether a saved preset holds *setting*: anything but a file or folder,
    or a setting of one image's (see the module docstring)."""
    if setting.kind == "path":
        return False
    if setting.name in _ONE_IMAGE_NAMES or setting.name.startswith(_ONE_IMAGE_PREFIXES):
        return False
    return not setting.name.endswith(_ONE_IMAGE_ENDINGS)


def saved_preset_dir() -> Path:
    """Where saved presets are kept: :data:`PRESET_DIR_ENV` if set, else a
    ``presets`` folder in the user's own settings folder for HaemoLynx."""
    override = os.environ.get(PRESET_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "HaemoLynx" / "presets"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "HaemoLynx" / "presets"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "haemolynx" / "presets"


@dataclass(frozen=True)
class SavedPreset:
    """One saved preset: its name, its file, and the settings it names."""

    name: str
    path: Path
    settings: Mapping[str, Any] = field(default_factory=dict)
    saved: Optional[str] = None

    @property
    def key(self) -> str:
        """What the panel's Preset menu keys this preset by."""
        return SAVED_PRESET_KEY_PREFIX + self.path.stem


def is_saved_preset_key(key: Any) -> bool:
    """Whether a Preset menu entry's *key* is a saved preset's."""
    return isinstance(key, str) and key.startswith(SAVED_PRESET_KEY_PREFIX)


def preset_file_stem(name: str) -> str:
    """The file stem preset *name* is saved under: its letters, digits,
    spaces, hyphens and underscores, anything else an underscore.

    Raises ValueError for a name with nothing in it.
    """
    stem = re.sub(r"[^\w\- ]", "_", name.strip()).strip()[:_MAX_STEM].strip()
    if not stem:
        raise ValueError("A preset needs a name")
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    return stem


def saved_preset_path(name: str, directory: Path | str | None = None) -> Path:
    """The file preset *name* is (or would be) saved in."""
    folder = Path(directory) if directory is not None else saved_preset_dir()
    return folder / f"{preset_file_stem(name)}{PRESET_SUFFIX}"


def _held_perturbations(entries: Any, schema: Schema) -> Any:
    """The perturbation entries, each without the overrides a saved preset
    may not hold (a mask file, vessels by ID): the run's own then apply."""
    if not isinstance(entries, (list, tuple)):
        return entries
    held = []
    for entry in entries:
        overrides = entry.get("overrides") if isinstance(entry, Mapping) else None
        if isinstance(overrides, Mapping):
            entry = {
                **entry,
                "overrides": {
                    key: value
                    for key, value in overrides.items()
                    if key not in schema or held_by_saved_presets(schema[key])
                },
            }
        held.append(entry)
    return held


def _yaml_value(value: Any) -> Any:
    """*value* as YAML's safe dumper takes it: lists, dicts and scalars."""
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _plain(setting: Setting, value: Any) -> Any:
    return _yaml_value(setting.coerce(value))


def saved_preset_values(values: Mapping[str, Any], schema: Schema) -> dict[str, Any]:
    """What a preset saved from the panel's *values* names: each setting it
    may hold that is away from its default, in the schema's order, as YAML
    writes it."""
    saved: dict[str, Any] = {}
    for setting in schema:
        if setting.name not in values or not held_by_saved_presets(setting):
            continue
        value = values[setting.name]
        if setting.name == "perturbations":
            value = _held_perturbations(value, schema)
        plain = _plain(setting, value)
        if plain != _plain(setting, setting.default):
            saved[setting.name] = plain
    return saved


def save_preset(
    name: str,
    values: Mapping[str, Any],
    schema: Schema,
    *,
    directory: Path | str | None = None,
) -> SavedPreset:
    """Write the panel's *values* as preset *name*, over any preset saved
    under the same file name, and return it."""
    import yaml

    path = saved_preset_path(name, directory)
    settings = saved_preset_values(values, schema)
    saved = datetime.now().isoformat(timespec="seconds")
    body = yaml.safe_dump(
        {"name": name.strip(), "saved": saved, "settings": settings},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10**9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside it and moved over the name, so a preset is never left
    # half written.
    partial = path.with_name(path.name + ".part")
    partial.write_text(_HEADER + body, encoding="utf-8")
    os.replace(partial, path)
    return SavedPreset(name=name.strip(), path=path, settings=settings, saved=saved)


def read_saved_preset(path: Path | str) -> SavedPreset:
    """The preset saved in *path*; ConfigError if it is not one."""
    import yaml

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Could not read the preset {path}: {error}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("settings") or {}, Mapping):
        raise ConfigError(f"{path} is not a saved preset: it has no mapping of settings")
    name = raw.get("name")
    return SavedPreset(
        name=str(name).strip() if name else path.stem,
        path=path,
        settings=dict(raw.get("settings") or {}),
        saved=str(raw["saved"]) if raw.get("saved") else None,
    )


def saved_presets(directory: Path | str | None = None) -> list[SavedPreset]:
    """Every preset saved in *directory* (default :func:`saved_preset_dir`),
    by name; a file that is not a preset is logged and passed over."""
    folder = Path(directory) if directory is not None else saved_preset_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob(f"*{PRESET_SUFFIX}")):
        try:
            found.append(read_saved_preset(path))
        except ConfigError as error:
            logger.warning("%s", error)
    return sorted(found, key=lambda preset: (preset.name.casefold(), preset.path.name))


def saved_preset_for_key(
    key: str, directory: Path | str | None = None
) -> SavedPreset:
    """The preset the Preset menu entry *key* stands for, read from its file
    now; ConfigError if it is gone or unreadable."""
    folder = Path(directory) if directory is not None else saved_preset_dir()
    path = folder / f"{key[len(SAVED_PRESET_KEY_PREFIX):]}{PRESET_SUFFIX}"
    if not path.is_file():
        raise ConfigError(f"The preset file {path} is no longer there")
    return read_saved_preset(path)


def saved_preset_settings(
    preset: SavedPreset, schema: Schema
) -> tuple[dict[str, Any], list[str]]:
    """The settings loading *preset* sets, and the names in it passed over.

    Every setting a preset may hold, at the preset's value or else its
    default, each coerced to its kind: ConfigError for a value it cannot
    take. A preset saved under a setting since renamed loads through
    :meth:`Schema.upgrade`; a name this schema lacks, or one a preset may not
    hold (written into the file by hand), is passed over and returned.
    """
    named = schema.upgrade(preset.settings)
    passed_over = sorted(
        name for name in named if name not in schema or not held_by_saved_presets(schema[name])
    )
    settings: dict[str, Any] = {}
    for setting in schema:
        if not held_by_saved_presets(setting):
            continue
        value = copy.deepcopy(named.get(setting.name, setting.default))
        if setting.name == "perturbations":
            value = _held_perturbations(value, schema)
        settings[setting.name] = setting.coerce(value)
    return settings, passed_over


def delete_saved_preset(preset: SavedPreset) -> None:
    """Remove *preset*'s file; nothing if it is already gone."""
    try:
        preset.path.unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "PRESET_DIR_ENV",
    "PRESET_SUFFIX",
    "SAVED_PRESET_KEY_PREFIX",
    "SavedPreset",
    "delete_saved_preset",
    "held_by_saved_presets",
    "is_saved_preset_key",
    "preset_file_stem",
    "read_saved_preset",
    "save_preset",
    "saved_preset_dir",
    "saved_preset_for_key",
    "saved_preset_path",
    "saved_preset_settings",
    "saved_preset_values",
    "saved_presets",
]
