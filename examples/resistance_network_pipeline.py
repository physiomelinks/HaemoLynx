#!/usr/bin/env python3
"""ImageLynx main pipeline package."""
import sys
from pathlib import Path

# Ensure package and sibling example modules are importable.
root_dir = Path(__file__).resolve().parents[1]
examples_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


from ImageLynx import haemodynamics
from ImageLynx.parsers import (
    add_schema_arguments,
    cli_overrides,
    dump_config,
    load_config,
)
from ImageLynx.pipeline import run_pipeline_stages
from preflight import run_preflight_checklist
from resistance_pipeline_schema import SCHEMA
from resistance_pipeline_settings import (
    PRESET_DEFINITIONS,
    build_settings_for_preset,
    list_presets,
)
from wizard import run_interactive_setup_wizard




# ---------------------------------------------------------------------------
# Settings -> pipeline arguments
#
# `resistance_pipeline_config.yaml` is the source of every setting, described by
# `resistance_pipeline_schema.py`. This section is the only place that knows how
# a setting name maps onto the pipeline stage arguments.
# ---------------------------------------------------------------------------

CONFIG_PATH = examples_dir / "resistance_pipeline_config.yaml"


#: Overrides may name a setting or the argument the old signature used.
ARGUMENT_TO_SETTING = {
    "image_path": "input_path",
    "axis_order": "image_axis_order",
    "do_pericyte_constriction": "do_pericyte_construction",
}


def resolve_settings(
    settings: dict | None = None,
    *,
    overrides: dict | None = None,
    config_path: Path | str = CONFIG_PATH,
    schema=SCHEMA,
) -> dict:
    """Every setting for one run, validated against the schema.

    With no arguments this is exactly what the config file says. Pass
    ``settings`` to supply an already-loaded dict, and ``overrides`` to change
    individual values on top of either.
    """
    overrides = {
        ARGUMENT_TO_SETTING.get(name, name): value
        for name, value in (overrides or {}).items()
    }
    plot_dir = overrides.pop("plot_dir", None)
    if settings is None:
        resolved = load_config(config_path, schema, overrides=overrides or None)
    else:
        # An already-resolved dict carries the derived entries too; they are not
        # schema settings, so set them aside rather than failing validation.
        merged = {**settings, **overrides}
        plot_dir = plot_dir or merged.pop("plot_dir", None)
        resolved = schema.validate({k: v for k, v in merged.items() if k in schema})
    _fill_derived_settings(resolved)
    if plot_dir is not None:
        resolved["plot_dir"] = Path(plot_dir)
    return resolved


def _fill_derived_settings(settings: dict) -> None:
    """Add the settings computed from other settings.

    The branch-order tables are functions of the manual diameter settings, so
    the config file states those and leaves these null rather than duplicating
    a 150-entry table that could then disagree with them.
    """
    settings.setdefault("plot_dir", Path(settings["base_plot_dir"]) / "nerve")
    if settings.get("diameter_by_branch_order") is None:
        settings["diameter_by_branch_order"] = haemodynamics.build_diameter_by_branch_order(
            all_diams_const=settings["all_diams_const"],
            max_branch_order=settings["max_branch_order"],
            default_diameter=settings["default_diameter"],
            manual_capillary_diameter_by_branch_order=settings[
                "manual_capillary_diameter_by_branch_order"
            ],
            manual_arteriole_diameter_by_branch_order=settings[
                "manual_arteriole_diameter_by_branch_order"
            ],
            manual_venule_diameter_by_branch_order=settings[
                "manual_venule_diameter_by_branch_order"
            ],
        )
    if settings.get("constriction_by_branch_order") is None:
        max_order = int(settings["max_branch_order"])
        constriction = {"B01": 1.0, "Art1": 1.0, "Ven1": 1.0}
        for order in range(2, max_order + 1):
            constriction[f"B{order:02d}"] = 0.8
            constriction[f"Art{order}"] = 0.8
            constriction[f"Ven{order}"] = 0.8
        settings["constriction_by_branch_order"] = constriction


def image_to_model_pipeline(settings: dict | None = None, **overrides):
    """Run the pipeline for one settings dict.

    ``image_to_model_pipeline()`` runs exactly what the config file says;
    ``overrides`` changes individual values for a single call without editing
    it, naming either a setting or the argument the old signature used::

        image_to_model_pipeline()
        image_to_model_pipeline(settings)
        image_to_model_pipeline(image_path="other.tif", do_skeletonize=False)
    """
    return run_pipeline_stages(resolve_settings(settings, overrides=overrides or None), SCHEMA)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the resistance network pipeline. Settings come from "
            "resistance_pipeline_config.yaml; every one of them can be overridden "
            "with a flag of the same name for a single run."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="YAML config to run from (default: %(default)s).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESET_DEFINITIONS.keys()),
        help="Apply a named preset's overrides on top of the config file.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--list-settings",
        action="store_true",
        help="List every setting with its value for this run, and exit.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Write the settings this run would use to a YAML file, and exit.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the preflight checklist and exit without executing the pipeline.",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Answer setup prompts instead of editing the config file.",
    )
    # One flag per setting, generated from the schema.
    add_schema_arguments(parser, SCHEMA)
    cli = parser.parse_args()

    if cli.list_presets:
        print("Available presets:")
        for preset_name, description in list_presets().items():
            print(f"  - {preset_name}: {description}")
        raise SystemExit(0)

    overrides: dict[str, object] = {}
    if cli.preset:
        # Preset overrides are still written in SCREAMING_SNAKE; the schema
        # names are the lower-case form of the same settings.
        preset_settings = build_settings_for_preset(preset_name=cli.preset)
        overrides.update(
            {
                name.lower(): value
                for name, value in preset_settings.items()
                if name.lower() in SCHEMA
            }
        )
        print(f"Applying preset '{cli.preset}'")
    if cli.wizard:
        wizard_results = run_interactive_setup_wizard(
            default_preset=cli.preset or "default",
            available_presets=sorted(PRESET_DEFINITIONS.keys()),
        )
        overrides.update(
            {
                name.lower(): value
                for name, value in wizard_results["settings_overrides"].items()
                if name.lower() in SCHEMA
            }
        )
        for name, value in wizard_results["pipeline_overrides"].items():
            setting_name = ARGUMENT_TO_SETTING.get(name, name)
            if setting_name in SCHEMA:
                overrides[setting_name] = value
    overrides.update(cli_overrides(cli))

    settings = resolve_settings(config_path=cli.config, overrides=overrides or None)
    print(f"Settings from: {cli.config}")
    if overrides:
        print(f"Overridden for this run: {sorted(overrides)}")

    if cli.list_settings:
        for section, section_settings in SCHEMA.sections().items():
            print(f"\n{section}")
            for setting in section_settings:
                print(f"  {setting.name:52s} {settings[setting.name]!r}")
        raise SystemExit(0)

    if cli.save_config is not None:
        saved_path = dump_config(cli.save_config, SCHEMA, values=settings)
        print(f"Saved the settings for this run to: {saved_path}")
        raise SystemExit(0)

    preflight_arguments = {
        **settings,
        **{
            argument: settings[setting]
            for argument, setting in ARGUMENT_TO_SETTING.items()
        },
    }
    preflight_report = run_preflight_checklist(preflight_arguments)
    if not preflight_report["ok"]:
        raise SystemExit(2)
    if cli.preflight_only:
        print("Preflight-only mode: exiting before pipeline execution.")
        raise SystemExit(0)

    image_to_model_pipeline(settings)
