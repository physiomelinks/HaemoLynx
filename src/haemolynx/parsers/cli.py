"""The command line every config-driven script gets for free.

A script that has a :class:`~haemolynx.parsers.Schema` and a config file needs
no argument parsing of its own::

    if __name__ == "__main__":
        main(settings_from_command_line(SCHEMA, CONFIG_PATH, description=__doc__))

which gives it ``--config``, a ``--<setting-name>`` flag for every setting,
``--preset``, ``--list-settings`` and ``--save-config``, all generated from the
schema so they cannot drift from it.

The direct ``print`` calls below are deliberate. This module is the command
line itself: everything it writes is the answer to a flag the user typed
(``--list-settings``, ``--list-presets``, ``--save-config``), not a library
reporting on work it is doing in the background. Those belong on stdout
unconditionally, and must not disappear because the caller has not configured
logging. Library modules log instead; :func:`configure_console_logging` is how
a script turns that logging into console output.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import add_schema_arguments, cli_overrides, dump_config, load_config
from .schema import ConfigError, Schema

#: Every log record carries its level, so a warning is visible as one.
CONSOLE_LOG_FORMAT = "[%(levelname)s] %(message)s"


def configure_console_logging(*, verbose: bool = False) -> None:
    """Send HaemoLynx's log records to the console.

    Call this from a script's entry point, never from library code: choosing
    where log records go is the application's decision, and a library that
    calls :func:`logging.basicConfig` makes it for every caller that imports
    it — including test runners and notebooks.

    Records go to stdout so that they interleave in order with anything the
    script prints itself.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=CONSOLE_LOG_FORMAT,
        stream=sys.stdout,
    )


def build_parser(
    schema: Schema,
    config_path: Path | str,
    *,
    description: str = "",
    presets: Mapping[str, Mapping[str, Any]] | None = None,
) -> argparse.ArgumentParser:
    """An argument parser for *schema*: the standard flags plus one per setting."""
    parser = argparse.ArgumentParser(
        description=description.strip().splitlines()[0] if description else None
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(config_path),
        help="YAML config to run from (default: %(default)s).",
    )
    parser.add_argument(
        "--list-settings",
        action="store_true",
        help="Print every setting with its value for this run, and exit.",
    )
    parser.add_argument(
        "--check-only",
        "--preflight-only",
        dest="check_only",
        action="store_true",
        help="Run the pre-run checks and exit without running anything.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Write the settings this run would use to a YAML file, and exit.",
    )
    if presets:
        parser.add_argument(
            "--preset",
            choices=sorted(presets),
            default=None,
            help="Apply a named set of overrides on top of the config file.",
        )
        parser.add_argument(
            "--list-presets",
            action="store_true",
            help="List the available presets and exit.",
        )
    add_schema_arguments(parser, schema)
    return parser


def settings_from_command_line(
    schema: Schema,
    config_path: Path | str,
    *,
    description: str = "",
    presets: Mapping[str, Mapping[str, Any]] | None = None,
    resolver: Callable[..., dict] | None = None,
    check: Callable[[dict], Any] | None = None,
    parser: argparse.ArgumentParser | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve one run's settings from the command line.

    Handles the flags that end a run early — ``--list-presets``,
    ``--list-settings``, ``--save-config`` — by printing and raising
    ``SystemExit``, and otherwise returns the settings dict.

    *resolver* is called as ``resolver(schema=..., config_path=..., overrides=...)``
    when a caller needs more than :func:`load_config` does; the pipeline passes
    its own so that derived settings are filled in. *check* is run on the
    resolved settings before returning, and ``--check-only`` exits after it.
    """
    parser = parser or build_parser(
        schema, config_path, description=description, presets=presets
    )
    parsed = parser.parse_args(argv)

    if presets and getattr(parsed, "list_presets", False):
        print("Available presets:")
        for name in sorted(presets):
            print(f"  - {name}: {presets[name].get('description', '')}")
        raise SystemExit(0)

    overrides: dict[str, Any] = {}
    preset_name = getattr(parsed, "preset", None)
    if preset_name:
        # A preset contributes only what it names; everything else still comes
        # from the config file.
        overrides.update(dict(presets[preset_name].get("overrides", {})))
        print(f"Applying preset '{preset_name}': {sorted(overrides) or 'no overrides'}")
    overrides.update(cli_overrides(parsed))

    if resolver is None:
        settings = load_config(parsed.config, schema, overrides=overrides or None)
    else:
        settings = resolver(
            schema=schema, config_path=parsed.config, overrides=overrides or None
        )
    print(f"Settings from: {parsed.config}")
    if overrides:
        print(f"Overridden for this run: {sorted(overrides)}")

    if parsed.list_settings:
        print_settings(schema, settings)
        raise SystemExit(0)

    if parsed.save_config is not None:
        known = {name: settings[name] for name in schema.names if name in settings}
        saved = dump_config(parsed.save_config, schema, values=known)
        print(f"Saved the settings for this run to: {saved}")
        raise SystemExit(0)

    if check is not None:
        check(settings)
    if parsed.check_only:
        print("Checks passed; exiting before the run.")
        raise SystemExit(0)

    return settings


def print_settings(schema: Schema, settings: Mapping[str, Any]) -> None:
    """Print every setting under its section heading, in schema order."""
    for section, section_settings in schema.sections().items():
        print(f"\n{section}")
        for setting in section_settings:
            if setting.name in settings:
                print(f"  {setting.name:52s} {settings[setting.name]!r}")


__all__ = [
    "build_parser",
    "configure_console_logging",
    "print_settings",
    "settings_from_command_line",
    "ConfigError",
]
