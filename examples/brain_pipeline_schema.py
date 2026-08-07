"""Settings for ``brain_network_pipeline.py``.

The whole-brain run is the standard pipeline followed by a pericyte dilation
sweep, so this schema is the standard one plus the sweep's own settings.
Everything in ``resistance_pipeline_config.yaml`` therefore means the same
thing in ``brain_pipeline_config.yaml``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.parsers import Schema, Setting  # noqa: E402
from ImageLynx.pipeline.schema import default_schema  # noqa: E402

_SWEEP = "Pericyte dilation sweep"

#: Settings the sweep adds on top of the standard pipeline.
SWEEP_SETTINGS = [
    Setting(
        name="run_pericyte_dilation_sweep",
        kind="bool",
        default=True,
        help="Sweep pericyte dilation against inlet pressure after the pipeline run",
        section=_SWEEP,
    ),
    Setting(
        name="pericyte_dilation_min_percent",
        kind="int",
        default=1,
        help="Start the dilation sweep at this percentage",
        section=_SWEEP,
        unit="percent",
        minimum=0,
        maximum=100,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="pericyte_dilation_max_percent",
        kind="int",
        default=30,
        help="End the dilation sweep at this percentage",
        section=_SWEEP,
        unit="percent",
        minimum=0,
        maximum=100,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="pericyte_dilation_step_percent",
        kind="int",
        default=1,
        help="Step the dilation sweep by this percentage",
        section=_SWEEP,
        unit="percent",
        minimum=1,
        maximum=100,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="inlet_pressure_min_pa",
        kind="int",
        default=4500,
        help="Start the inlet-pressure sweep at this pressure",
        section=_SWEEP,
        unit="Pa",
        minimum=0,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="inlet_pressure_max_pa",
        kind="int",
        default=6000,
        help="End the inlet-pressure sweep at this pressure",
        section=_SWEEP,
        unit="Pa",
        minimum=0,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="inlet_pressure_step_pa",
        kind="int",
        default=500,
        help="Step the inlet-pressure sweep by this much",
        section=_SWEEP,
        unit="Pa",
        minimum=1,
        requires=("run_pericyte_dilation_sweep",),
    ),
    Setting(
        name="sweep_output_dir",
        kind="path",
        default="examples/outputs/brain_dilation_sweep",
        help="Write the sweep CSV and its curves here",
        section=_SWEEP,
        requires=("run_pericyte_dilation_sweep",),
    ),
]

SCHEMA = Schema(
    list(default_schema()) + SWEEP_SETTINGS,
    title="Whole-brain network pipeline",
    description=(
        "The standard image-to-model pipeline plus a pericyte dilation and\n"
        "inlet-pressure sweep over the resulting network."
    ),
)
