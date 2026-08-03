"""Read, write and override settings described by a :class:`Schema`.

The config file is the single place a run is configured from. Changing a
setting means editing the YAML (or overriding it on the command line), never
editing a call site — so the examples read as a sequence of pipeline stages
rather than a wall of keyword arguments.

Precedence, lowest to highest: schema defaults, config file, explicit
overrides (``--set name=value``).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import ConfigError, Schema, Setting, _jsonify, section_key

#: Threshold from the project convention: a call taking more than this many
#: settings is passed the config dict instead of individual keyword arguments.
DICT_ARGUMENT_THRESHOLD = 6


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ConfigError(
            "PyYAML is required to read config files. Install it with "
            "`pip install pyyaml`, or `pip install -e .` to pick up the "
            "declared dependency."
        ) from exc
    return yaml


def load_config(
    config_path: Path | str,
    schema: Schema,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load *config_path*, validate it against *schema*, return a settings dict.

    Missing keys take their schema default, so a config file need only state
    what it changes.
    """
    yaml = _require_yaml()
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Config file {path} must contain a mapping of settings, got "
            f"{type(raw).__name__}."
        )

    flat = _flatten_sections(raw, schema, source=path)
    if overrides:
        flat.update(overrides)
    try:
        return schema.validate(flat)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def _flatten_sections(
    raw: Mapping[str, Any], schema: Schema, *, source: Path | None = None
) -> dict[str, Any]:
    """Accept both a flat mapping and one nested one level by section."""
    section_names = set(schema.sections())
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if key in schema:
            flat[key] = value
        elif isinstance(value, Mapping) and (key in section_names or value):
            # A section heading: its keys are settings. Unknown names inside are
            # still reported by validate(), with the section named for context.
            for inner_key, inner_value in value.items():
                if inner_key in flat:
                    where = f" in {source}" if source else ""
                    raise ConfigError(
                        f"Setting '{inner_key}' appears more than once{where}."
                    )
                flat[inner_key] = inner_value
        else:
            flat[key] = value
    return flat


def dump_config(
    config_path: Path | str,
    schema: Schema,
    *,
    values: Mapping[str, Any] | None = None,
    include_help: bool = True,
) -> Path:
    """Write a commented YAML config for *schema*.

    The file is generated from the schema, so every setting appears with its
    help text, unit, allowed values and prerequisites as comments. Regenerating
    is how a config file is kept in step with the schema.
    """
    resolved = schema.validate(values or {})
    lines: list[str] = []
    if schema.title:
        lines.append(f"# {schema.title}")
    if schema.description:
        lines.extend(f"# {line}" for line in schema.description.splitlines())
    if lines:
        lines.append("")

    for section, settings in schema.sections().items():
        lines.append("# " + "-" * 72)
        lines.append(f"# {section}")
        lines.append("# " + "-" * 72)
        lines.append(f"{section_key(section)}:")
        for setting in settings:
            if include_help:
                lines.extend(_comment_lines(setting))
            lines.append(
                f"  {setting.name}: {_scalar(resolved.get(setting.name))}"
                if not _is_block(resolved.get(setting.name))
                else _block(setting.name, resolved.get(setting.name))
            )
        lines.append("")

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _comment_lines(setting: Setting) -> list[str]:
    notes: list[str] = []
    if setting.unit:
        notes.append(setting.unit)
    if setting.choices:
        notes.append("one of: " + ", ".join(str(c) for c in setting.choices))
    if setting.minimum is not None or setting.maximum is not None:
        low = "" if setting.minimum is None else str(setting.minimum)
        high = "" if setting.maximum is None else str(setting.maximum)
        notes.append(f"range {low}..{high}")
    if setting.requires:
        notes.append(
            "needs "
            + " and ".join(
                f"{name[1:]} off" if name.startswith("!") else name
                for name in setting.requires
            )
        )
    if setting.must_exist:
        notes.append("must exist")
    suffix = f"  [{'; '.join(notes)}]" if notes else ""
    return [f"  # {setting.help}{suffix}"]


def _is_block(value: Any) -> bool:
    """True when *value* needs an indented block rather than one inline line."""
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (tuple, list)):
        # Short coordinate-like lists read better inline: [0.0, 0.0, 600.0].
        return len(value) > 4 or any(
            isinstance(item, (Mapping, tuple, list)) for item in value
        )
    return False


def _block(name: str, value: Any) -> str:
    yaml = _require_yaml()
    dumped = yaml.safe_dump(
        {name: _jsonify(value)}, default_flow_style=False, sort_keys=False
    ).rstrip()
    return "\n".join(f"  {line}" for line in dumped.splitlines())


def _scalar(value: Any) -> str:
    yaml = _require_yaml()
    return yaml.safe_dump(
        _jsonify(value), default_flow_style=True, sort_keys=False
    ).strip().rstrip("...").strip()


def settings_for(values: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """The subset of *values* named by *names*.

    Used to hand one stage of a pipeline exactly the settings it consumes, so a
    function's dependencies stay visible even when it is passed a dict.
    """
    missing = [name for name in names if name not in values]
    if missing:
        raise ConfigError(f"Settings not present: {', '.join(missing)}.")
    return {name: values[name] for name in names}


def prefixed_arguments(
    settings: Mapping[str, Any], prefix: str, valid_parameters: Iterable[str]
) -> dict[str, Any]:
    """Settings named ``<prefix><parameter>`` as keyword arguments.

    A group of settings that share a prefix and otherwise match a function's
    parameters can be handed over in one go. Only names the function actually
    accepts are passed, so a setting that stops being read shows up as a
    mismatch here rather than as a value that silently does nothing::

        preprocess_skeleton_for_graph(
            skeleton, **prefixed_arguments(settings, "skeleton_", parameters)
        )
    """
    valid = set(valid_parameters)
    return {
        name[len(prefix):]: value
        for name, value in settings.items()
        if name.startswith(prefix) and name[len(prefix):] in valid
    }


def parameters_of(function) -> tuple[str, ...]:
    """Parameter names of *function*, for pairing with :func:`prefixed_arguments`."""
    return tuple(inspect.signature(function).parameters)


def add_schema_arguments(parser, schema: Schema, *, prefix: str = "") -> None:
    """Add one ``--flag`` per setting to an ``argparse`` parser.

    Flags default to ``None`` so that "not given" stays distinguishable from
    "given the default value"; feed the result through :func:`cli_overrides`.
    """
    for setting in schema:
        flag = f"--{prefix}{setting.name.replace('_', '-')}"
        help_text = setting.help + (f" [{setting.unit}]" if setting.unit else "")
        if setting.kind == "bool":
            parser.add_argument(
                flag,
                dest=f"set__{setting.name}",
                type=_parse_bool_text,
                metavar="true|false",
                default=None,
                help=help_text,
            )
        elif setting.kind == "choice":
            parser.add_argument(
                flag,
                dest=f"set__{setting.name}",
                choices=list(setting.choices or ()),
                default=None,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=f"set__{setting.name}",
                metavar=setting.kind.upper(),
                default=None,
                help=help_text,
            )


def _parse_bool_text(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"true", "yes", "1", "on"}:
        return True
    if lowered in {"false", "no", "0", "off"}:
        return False
    raise ValueError(f"expected true or false, got {text!r}")


def cli_overrides(namespace) -> dict[str, Any]:
    """Collect the ``--flag`` values that were actually supplied."""
    return {
        key[len("set__"):]: value
        for key, value in vars(namespace).items()
        if key.startswith("set__") and value is not None
    }
