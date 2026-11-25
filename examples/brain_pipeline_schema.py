"""Settings for ``brain_network_pipeline.py``.

The whole-brain run is the standard pipeline followed by a pericyte dilation
sweep, and the sweep's settings are now part of the pipeline's own schema --
the napari panel builds its form from ``default_schema()`` alone, so a setting
declared out here beside an example was one the panel could not show. So this
adds nothing but a title: everything in ``brain_pipeline_config.yaml`` means
exactly what it means in ``resistance_pipeline_config.yaml``.

The module stays because ``regenerate_configs.CONFIGS`` names it, and because
the next setting this example alone needs belongs here rather than in the
package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx.parsers import Schema  # noqa: E402
from haemolynx.pipeline.schema import default_schema  # noqa: E402

SCHEMA = Schema(
    list(default_schema()),
    title="Whole-brain network pipeline",
    description=(
        "The standard image-to-model pipeline plus a pericyte dilation and\n"
        "inlet-pressure sweep over the resulting network."
    ),
)
