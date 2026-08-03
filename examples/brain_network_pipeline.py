#!/usr/bin/env python3
"""Whole-brain vascular network: the standard pipeline, then a dilation sweep.

Runs ``ImageLynx.pipeline`` exactly as ``resistance_network_pipeline.py`` does,
then sweeps pericyte dilation against inlet pressure over the resulting network
and plots the flow and resistance curves.

Every setting lives in ``brain_pipeline_config.yaml``, described by
``brain_pipeline_schema.py`` -- the standard pipeline's settings plus the
sweep's. Change a value there rather than editing this script::

    python examples/brain_network_pipeline.py
    python examples/brain_network_pipeline.py --config other_config.yaml
    python examples/brain_network_pipeline.py --pericyte-dilation-max-percent 10
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
examples_dir = root_dir / "examples"
for _path in (root_dir / "src", examples_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.haemodynamics.pericyte_sweep import run_pericyte_dilation_pressure_sweep
from ImageLynx.parsers import settings_from_command_line
from ImageLynx.pipeline import (
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    resolve_settings,
    segment,
    skeletonise,
    solve,
)
from ImageLynx.visualization.dilation_curves import plot_dilation_curves
from brain_pipeline_schema import SCHEMA
from pipeline_presets import PRESETS
from resistance_network_pipeline import run_preflight

CONFIG_PATH = examples_dir / "brain_pipeline_config.yaml"


def main(settings: dict) -> dict:
    """Run the pipeline, then the dilation sweep over the network it built."""
    inputs = segment(settings)
    volume = skeletonise(settings, inputs)
    network = build_network(settings, volume, SCHEMA)
    boundaries = assign_boundaries(settings, network)
    diameters = assign_diameters(settings, network, boundaries, SCHEMA)
    model = build_haemodynamic_model(settings, diameters)
    solution = solve(settings, model, boundaries)
    export_results(settings, network, model, solution)

    G = model.graph
    if not settings["run_pericyte_dilation_sweep"]:
        return {"graph": G, "sweep": None, "curves": None}

    sweep = run_pericyte_dilation_pressure_sweep(
        G,
        settings,
        starting_nodes=boundaries.starting_nodes,
        output_nodes=boundaries.output_nodes,
        output_dir=settings["sweep_output_dir"],
    )
    curves = plot_dilation_curves(sweep["results"], settings["sweep_output_dir"])
    return {"graph": G, "sweep": sweep, "curves": curves}


if __name__ == "__main__":
    main(
        settings_from_command_line(
            SCHEMA,
            CONFIG_PATH,
            description=__doc__,
            presets=PRESETS,
            resolver=resolve_settings,
            check=run_preflight,
        )
    )
