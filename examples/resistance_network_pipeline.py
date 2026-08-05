#!/usr/bin/env python3
"""ImageLynx main pipeline package."""
from __future__ import annotations

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
from ImageLynx.parsers import settings_from_command_line
from ImageLynx.pipeline import resolve_settings as _resolve_settings
from ImageLynx.pipeline import (
    preflight,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    segment,
    skeletonise,
    solve,
)
from resistance_pipeline_schema import SCHEMA
from pipeline_presets import PRESETS




# ---------------------------------------------------------------------------
# Settings -> pipeline arguments
#
# `resistance_pipeline_config.yaml` is the source of every setting, described by
# `resistance_pipeline_schema.py`. This section is the only place that knows how
# a setting name maps onto the pipeline stage arguments.
# ---------------------------------------------------------------------------

CONFIG_PATH = examples_dir / "resistance_pipeline_config.yaml"


def _preflight_or_exit(settings: dict) -> None:
    """Run the pre-run checks; stop before doing any work if any failed."""
    if not preflight(settings, SCHEMA).ok:
        raise SystemExit(2)


def resolve_settings(settings=None, *, overrides=None, config_path=CONFIG_PATH, schema=SCHEMA):
    """This example's settings: the shared resolver, with its config and schema."""
    return _resolve_settings(
        settings, schema=schema, config_path=config_path, overrides=overrides
    )






def image_to_model_pipeline(settings: dict | None = None, **overrides):
    """Run the pipeline for one settings dict.

    ``image_to_model_pipeline()`` runs exactly what the config file says;
    ``overrides`` changes individual values for a single call without editing
    it, naming either a setting or the argument the old signature used::

        image_to_model_pipeline()
        image_to_model_pipeline(settings)
        image_to_model_pipeline(image_path="other.tif", do_skeletonize=False)
    """
    settings = resolve_settings(settings, overrides=overrides or None)

    inputs = segment(settings)
    volume = skeletonise(settings, inputs)
    network = build_network(settings, volume, SCHEMA)
    boundaries = assign_boundaries(settings, network)
    diameters = assign_diameters(settings, network, boundaries, SCHEMA)
    model = build_haemodynamic_model(settings, diameters)
    solution = solve(settings, model, boundaries)
    export_results(settings, network, model, solution)

    return model.graph


if __name__ == "__main__":
    image_to_model_pipeline(
        settings_from_command_line(
            SCHEMA,
            CONFIG_PATH,
            description=__doc__,
            presets=PRESETS,
            resolver=resolve_settings,
            check=lambda settings: _preflight_or_exit(settings),
        )
    )
