"""Settings for ``carotid_image_to_model.py``.

The carotid run is the standard image-to-model pipeline pointed at one
dataset, so this schema is the standard one with the defaults that dataset
needs. Every setting therefore means in ``carotid_config.yaml`` exactly what it
means in ``resistance_pipeline_config.yaml``; only the values below differ.

Regenerate the config file after changing this schema::

    python examples/regenerate_configs.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx.parsers import Schema  # noqa: E402
from haemolynx.pipeline.schema import default_schema  # noqa: E402

#: Where this dataset's inlets and outlets are: the carotid stack runs along
#: the y axis, so the terminal nodes in the first and last 10% of the network
#: along that axis are the ones flow enters and leaves through, rather than
#: hand-picked coordinates. `boundary_axis`, `boundary_first_percent` and
#: `boundary_last_percent` keep their defaults of axis 1, 10% and 10%, which is
#: what this dataset needs. This is also the pipeline default now, so it is
#: named here only to say that it was chosen rather than inherited.
_BOUNDARIES_FROM_NETWORK_ENDS = {
    "starting_node_selection_method": "edge_percent",
    "output_node_selection_method": "edge_percent",
    "starting_node_coordinates": [],
    "output_node_coordinates": [],
}

#: Defaults that differ from the standard pipeline's, keyed by setting name.
CAROTID_DEFAULTS: dict[str, Any] = {
    # Input: an already-segmented mask, or the raw stack when ilastik runs.
    "input_path": "examples/images/carotid_mask.tif",
    "ilastik_unsegmented_image_path": "examples/images/carotid.tif",
    "ilastik_classifier_path": "examples/classifiers/carotid_classifier.ilp",
    **_BOUNDARIES_FROM_NETWORK_ENDS,
    # Perfusion pressures measured for this preparation.
    "input_p_bc": 1000.0,
    "output_p_bc": 500.0,
    # This stack skeletonises into well-separated fragments, so bridging is off
    # and the small components are dropped rather than reconnected.
    "skeleton_max_bridge_distance": 0,
    "skeleton_min_component_percent": 5.0,
    "vtk_output_prefix": "examples/outputs/carotid/carotid_network",
}

PIPELINE_SCHEMA = default_schema()

_unknown = set(CAROTID_DEFAULTS) - set(PIPELINE_SCHEMA.names)
if _unknown:
    raise ValueError(
        "carotid defaults name settings that the pipeline schema does not "
        f"have: {sorted(_unknown)}"
    )

SCHEMA = Schema(
    [
        replace(setting, default=CAROTID_DEFAULTS[setting.name])
        if setting.name in CAROTID_DEFAULTS
        else setting
        for setting in PIPELINE_SCHEMA
    ],
    title="Carotid network pipeline",
    description=(
        "The standard image-to-model pipeline over a single carotid dataset.\n"
        "Same settings as resistance_pipeline_config.yaml, different values."
    ),
)
