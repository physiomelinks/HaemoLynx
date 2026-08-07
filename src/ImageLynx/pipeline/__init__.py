"""The image-to-model pipeline: its settings, and the stages of a run.

Every setting is declared once, in :mod:`ImageLynx.pipeline.schema`. Start from
that declaration to write yourself a config file, then run the whole thing::

    write_default_config("my_config.yaml")      # commented, every setting
    schema = default_schema()
    settings = resolve_settings(schema=schema, config_path="my_config.yaml")
    graph = run_pipeline_stages(settings, schema)

or call the stages yourself, which is what the examples do so that each step of
a run is visible and can be intervened in::

    inputs   = segment(settings)
    volume   = skeletonise(settings, inputs)
    network  = build_network(settings, volume, SCHEMA)
    ...
"""
from .checks import preflight
from .schema import SCHEMA, default_schema, write_default_config
from .settings import fill_derived_settings, resolve_settings
from .stages import (
    BoundaryNodes,
    HaemodynamicModel,
    SegmentedInputs,
    SkeletonisedVolume,
    Solution,
    VesselNetwork,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    export_results,
    run_pipeline_stages,
    segment,
    skeletonise,
    solve,
)

__all__ = [
    "SCHEMA",
    "BoundaryNodes",
    "HaemodynamicModel",
    "SegmentedInputs",
    "SkeletonisedVolume",
    "Solution",
    "VesselNetwork",
    "assign_boundaries",
    "assign_diameters",
    "build_haemodynamic_model",
    "build_network",
    "default_schema",
    "export_results",
    "preflight",
    "fill_derived_settings",
    "resolve_settings",
    "run_pipeline_stages",
    "segment",
    "skeletonise",
    "solve",
    "write_default_config",
]
