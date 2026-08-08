"""Config parsing: declarative setting schemas, YAML files, CLI overrides.

A pipeline declares its settings once, as a :class:`Schema` of :class:`Setting`
entries. That single declaration drives the YAML config file, the command-line
flags, validation, and — because :meth:`Schema.describe` is plain JSON — a GUI
form, without any of them repeating the list of settings.

Typical use in an example script::

    from haemolynx.parsers import load_config
    from my_example_schema import SCHEMA

    settings = load_config("my_example_config.yaml", SCHEMA)
    run_stage(settings)                      # >6 settings: pass the dict
    plot(settings["plot_dir"], settings["show"])   # <=6: pass the entries
"""
from .config import (
    DICT_ARGUMENT_THRESHOLD,
    add_schema_arguments,
    cli_overrides,
    dump_config,
    load_config,
    parameters_of,
    prefixed_arguments,
    settings_for,
)
from .checks import CheckReport, check_settings
from .cli import (
    build_parser,
    configure_console_logging,
    print_settings,
    settings_from_command_line,
)
from .schema import ConfigError, IneffectiveSettingWarning, Schema, Setting

__all__ = [
    "CheckReport",
    "ConfigError",
    "check_settings",
    "IneffectiveSettingWarning",
    "DICT_ARGUMENT_THRESHOLD",
    "Schema",
    "Setting",
    "add_schema_arguments",
    "build_parser",
    "cli_overrides",
    "configure_console_logging",
    "dump_config",
    "load_config",
    "parameters_of",
    "prefixed_arguments",
    "print_settings",
    "settings_for",
    "settings_from_command_line",
]
