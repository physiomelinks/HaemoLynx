#!/usr/bin/env python3
"""Carotid vascular network: the standard pipeline over one carotid dataset.

Runs ``ImageLynx.pipeline`` exactly as ``resistance_network_pipeline.py`` does
-- segmentation, skeletonisation, graph building, boundary and branch-order
assignment, haemodynamics, export -- with the settings this dataset needs. The
run itself holds nothing carotid-specific, so there is no forked copy of the
pipeline here; what makes it a carotid run lives in ``carotid_config.yaml``,
described by ``carotid_schema.py``.

The input is an **already-segmented** mask by default
(``examples/images/carotid_mask.tif``). To segment the raw stack in-repo
instead, set in the config file::

    use_ilastik_segmentation: true
    ilastik_unsegmented_image_path: examples/images/carotid.tif
    ilastik_classifier_path: <your trained .ilp>

Training the classifier stays manual, in the ilastik GUI. Either way the image
files live outside the repository, so a run needs that data on disk; the
pre-run checks say which paths are missing before any work starts.

Change values in the config file rather than editing this script::

    python examples/carotid_image_to_model.py
    python examples/carotid_image_to_model.py --config other_config.yaml
    python examples/carotid_image_to_model.py --input-path my_mask.tif
    python examples/carotid_image_to_model.py --list-settings
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

root_dir = Path(__file__).resolve().parents[1]
examples_dir = root_dir / "examples"
for _path in (root_dir / "src", examples_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ImageLynx.parsers import configure_console_logging, settings_from_command_line
from ImageLynx.pipeline import preflight, run_pipeline_stages
from ImageLynx.pipeline import resolve_settings as _resolve_settings
from carotid_schema import SCHEMA

CONFIG_PATH = examples_dir / "carotid_config.yaml"


def _preflight_or_exit(settings: dict) -> None:
    """Run the pre-run checks; stop before doing any work if any failed."""
    if not preflight(settings, SCHEMA).ok:
        raise SystemExit(2)


def resolve_settings(settings=None, *, overrides=None, config_path=CONFIG_PATH, schema=SCHEMA):
    """This example's settings: the shared resolver, with its config and schema.

    The plots go in their own directory, so one dataset's figures never
    overwrite another's.
    """
    overrides = dict(overrides or {})
    asked_for = overrides.get("plot_dir") or (settings or {}).get("plot_dir")
    resolved = _resolve_settings(
        settings, schema=schema, config_path=config_path, overrides=overrides
    )
    if asked_for is None:
        resolved["plot_dir"] = Path(resolved["base_plot_dir"]) / "carotid"
    return resolved


def main(settings: dict | None = None, **overrides) -> nx.MultiGraph | None:
    """Run the pipeline for one settings dict, as loaded from the config file."""
    return run_pipeline_stages(
        resolve_settings(settings, overrides=overrides or None), SCHEMA
    )


if __name__ == "__main__":
    settings = settings_from_command_line(
        SCHEMA,
        CONFIG_PATH,
        description=__doc__,
        resolver=resolve_settings,
        check=_preflight_or_exit,
    )
    # The pipeline reports its progress through `logging`; sending this run's
    # progress to the console is the script's call, not the library's.
    configure_console_logging(verbose=settings["verbose_logging"])
    main(settings)
