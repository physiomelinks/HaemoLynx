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
import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
examples_dir = root_dir / "examples"
for _path in (root_dir / "src", examples_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.haemodynamics.pericyte_sweep import run_pericyte_dilation_pressure_sweep
from ImageLynx.parsers import add_schema_arguments, cli_overrides
from ImageLynx.pipeline import run_pipeline_stages
from ImageLynx.visualization.dilation_curves import plot_dilation_curves
from brain_pipeline_schema import SCHEMA
from resistance_network_pipeline import resolve_settings

CONFIG_PATH = examples_dir / "brain_pipeline_config.yaml"


def main(settings: dict) -> dict:
    """Run the pipeline, then the dilation sweep over the network it built."""
    G = run_pipeline_stages(settings, SCHEMA)
    if G is None:
        raise RuntimeError(
            "The pipeline produced no graph; enable do_graph_building or point "
            "graph_pickle_path at a saved run."
        )

    if not settings["run_pericyte_dilation_sweep"]:
        return {"graph": G, "sweep": None, "curves": None}

    sweep = run_pericyte_dilation_pressure_sweep(
        G,
        settings,
        starting_nodes=settings["starting_nodes"],
        output_nodes=settings["output_nodes"],
        output_dir=settings["sweep_output_dir"],
    )
    curves = plot_dilation_curves(sweep["results"], settings["sweep_output_dir"])
    return {"graph": G, "sweep": sweep, "curves": curves}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="YAML config to run from."
    )
    add_schema_arguments(parser, SCHEMA)
    cli = parser.parse_args()

    main(
        resolve_settings(
            overrides=cli_overrides(cli) or None,
            config_path=cli.config,
            schema=SCHEMA,
        )
    )
