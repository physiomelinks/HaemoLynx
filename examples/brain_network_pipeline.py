#!/usr/bin/env python3
"""Whole-brain vascular network: the standard pipeline, then a dilation sweep.

Runs ``haemolynx.pipeline`` exactly as ``resistance_network_pipeline.py`` does,
then sweeps pericyte dilation against inlet pressure over the resulting network
and plots the flow and resistance curves.

The napari panel offers the same combined sweep as a
``pressure_and_pericyte_sweep`` perturbation (and separate pressure-only /
dilation-only types). This script still uses the always-on
``run_pericyte_dilation_sweep`` flag for the post-step, so a CLI brain run does
not need a ``perturbations`` list.

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

from haemolynx.haemodynamics.pericyte_sweep import run_pericyte_dilation_pressure_sweep
from haemolynx.parsers import configure_console_logging, settings_from_command_line
from haemolynx.pipeline import (
    apply_network_handling,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    record_run,
    resolve_settings,
    segment,
    skeletonise,
    solve,
)
from haemolynx.visualization.dilation_curves import plot_dilation_curves
from brain_pipeline_schema import SCHEMA
from pipeline_presets import PRESETS
from resistance_network_pipeline import _preflight_or_exit

CONFIG_PATH = examples_dir / "brain_pipeline_config.yaml"


def main(settings: dict) -> dict:
    """Run the pipeline, then the dilation sweep over the network it built."""
    # The pipeline's manifest goes beside its outputs as `record_run` ends,
    # before the sweep, which writes to a folder of its own.
    with record_run(settings, SCHEMA) as run:
        with run.stage("segment"):
            inputs = segment(settings)
        with run.stage("skeletonise"):
            volume = skeletonise(settings, inputs)
        with run.stage("build_network"):
            network = build_network(settings, volume, SCHEMA)
        with run.stage("assign_boundaries"):
            boundaries = assign_boundaries(settings, network)
        with run.stage("assign_diameters"):
            diameters = assign_diameters(settings, network, boundaries, SCHEMA)
        with run.stage("build_haemodynamic_model"):
            diameters = apply_network_handling(settings, diameters, boundaries, network)
            model = build_haemodynamic_model(settings, diameters)
        with run.stage("solve"):
            solution = solve(settings, model, boundaries)
        with run.stage("export_results"):
            export_results(settings, network, model, solution)

    G = model.graph
    if not settings["run_pericyte_dilation_sweep"]:
        return {"graph": G, "sweep": None, "curves": None}

    sweep = run_pericyte_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=boundaries.inlet_nodes,
        outlet_nodes=boundaries.outlet_nodes,
        output_dir=settings["sweep_output_dir"],
    )
    curves = plot_dilation_curves(sweep["results"], settings["sweep_output_dir"])
    return {"graph": G, "sweep": sweep, "curves": curves}


if __name__ == "__main__":
    settings = settings_from_command_line(
        SCHEMA,
        CONFIG_PATH,
        description=__doc__,
        presets=PRESETS,
        resolver=resolve_settings,
        check=_preflight_or_exit,
    )
    # The pipeline and the sweep report progress through `logging`; sending it
    # to the console is this script's decision to make, not the library's.
    configure_console_logging(verbose=settings["verbose_logging"])
    main(settings)
